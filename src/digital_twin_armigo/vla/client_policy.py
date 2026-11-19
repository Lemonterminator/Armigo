"""Sync/async chunk client policy implementing ``SegmentPolicy`` for S10 closed-loop evaluation."""

from __future__ import annotations

import contextlib
import http.client
import json
import threading
from collections.abc import Mapping
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Self
from urllib.parse import urlsplit

import numpy as np

from ..constants import TOP_CROP, TOP_FULL
from .contract import (
	ACTION_DIM,
	INITIAL_Q,
	POLICY_FIELDS,
	STUDENT_CAMERA_NAMES,
	assert_no_planner_diagnostics,
)
from .server import (
	FORBIDDEN_IMAGE_PAYLOAD_KEYS,
	ChunkPredictor,
	LocalChunkPredictor,
	encode_image,
)

DEFAULT_ASYNC_REPLAN_INTERVAL = 25


def extract_student_images(raw_images: Mapping[str, Any]) -> dict[str, np.ndarray]:
	"""Extract D1 student camera arrays from ``obs.images()`` without ever reading ``front`` or ``top_full``.

	- ``top_rgb`` -> cropped via ``constants.TOP_CROP`` to ``top_rgb_v2`` ``(416, 512, 3)``
	- ``top_depth`` -> ``top_depth_v2`` ``(416, 512, 3)``
	- ``wrist`` -> ``wrist`` ``(480, 640, 3)``
	"""
	x, y, w, h = TOP_CROP
	full_h, full_w = TOP_FULL[1], TOP_FULL[0]

	try:
		top_rgb_raw = raw_images["top_rgb"]
	except KeyError:
		top_rgb_raw = raw_images["top_rgb_v2"]
	top_rgb_arr = np.asarray(top_rgb_raw)
	if top_rgb_arr.ndim != 3 or top_rgb_arr.shape[2] != 3 or top_rgb_arr.dtype != np.uint8:
		raise ValueError(f"top_rgb must be (H, W, 3) uint8, got shape={top_rgb_arr.shape} dtype={top_rgb_arr.dtype}")
	if top_rgb_arr.shape[:2] == (full_h, full_w):
		top_rgb_v2 = np.ascontiguousarray(top_rgb_arr[y : y + h, x : x + w], dtype=np.uint8)
	elif top_rgb_arr.shape[:2] == (h, w):
		top_rgb_v2 = np.ascontiguousarray(top_rgb_arr, dtype=np.uint8)
	else:
		raise ValueError(f"unexpected top_rgb shape {top_rgb_arr.shape}; expected {(full_h, full_w, 3)} or {(h, w, 3)}")

	try:
		top_depth_raw = raw_images["top_depth"]
	except KeyError:
		top_depth_raw = raw_images["top_depth_v2"]
	top_depth_arr = np.asarray(top_depth_raw)
	if top_depth_arr.ndim != 3 or top_depth_arr.shape[2] != 3 or top_depth_arr.dtype != np.uint8:
		raise ValueError(
			f"top_depth must be (H, W, 3) uint8, got shape={top_depth_arr.shape} dtype={top_depth_arr.dtype}"
		)
	if top_depth_arr.shape[:2] == (full_h, full_w):
		top_depth_v2 = np.ascontiguousarray(top_depth_arr[y : y + h, x : x + w], dtype=np.uint8)
	elif top_depth_arr.shape[:2] == (h, w):
		top_depth_v2 = np.ascontiguousarray(top_depth_arr, dtype=np.uint8)
	else:
		raise ValueError(
			f"unexpected top_depth shape {top_depth_arr.shape}; expected {(h, w, 3)} or {(full_h, full_w, 3)}"
		)

	wrist_arr = np.asarray(raw_images["wrist"])
	if wrist_arr.ndim != 3 or wrist_arr.shape[2] != 3 or wrist_arr.dtype != np.uint8:
		raise ValueError(f"wrist must be (H, W, 3) uint8, got shape={wrist_arr.shape} dtype={wrist_arr.dtype}")
	wrist = np.ascontiguousarray(wrist_arr, dtype=np.uint8)

	return {
		"top_rgb_v2": top_rgb_v2,
		"top_depth_v2": top_depth_v2,
		"wrist": wrist,
	}


class RemoteChunkPredictor:
	"""Thread-safe HTTP client implementing ``ChunkPredictor`` against ``vla/server.py``."""

	def __init__(
		self,
		url: str,
		*,
		timeout_s: float = 30.0,
		image_encoding: str = "npy",
	) -> None:
		parsed = urlsplit(url.strip())
		if parsed.scheme != "http" or not parsed.hostname:
			raise ValueError(f"unsupported server URL {url!r}; expected http://host:port")
		if image_encoding not in ("npy", "png", "raw"):
			raise ValueError(f"unsupported image_encoding {image_encoding!r}")
		self.url = url.rstrip("/")
		self._host = parsed.hostname
		self._port = int(parsed.port or 80)
		self._base_path = parsed.path.rstrip("/")
		self._timeout_s = float(timeout_s)
		self._image_encoding = image_encoding
		self._local = threading.local()
		self._conns_lock = threading.Lock()
		self._conns: list[http.client.HTTPConnection] = []
		self._closed = False

	def _get_conn(self) -> http.client.HTTPConnection:
		conn = getattr(self._local, "conn", None)
		if conn is None:
			conn = http.client.HTTPConnection(self._host, self._port, timeout=self._timeout_s)
			self._local.conn = conn
			with self._conns_lock:
				if self._closed:
					conn.close()
					raise RuntimeError("RemoteChunkPredictor is closed")
				self._conns.append(conn)
		return conn

	def _drop_conn(self) -> None:
		conn = getattr(self._local, "conn", None)
		if conn is not None:
			self._local.conn = None
			with self._conns_lock, contextlib.suppress(ValueError):
				self._conns.remove(conn)
			with contextlib.suppress(Exception):
				conn.close()

	def _request_json(self, method: str, endpoint: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
		if self._closed:
			raise RuntimeError("RemoteChunkPredictor is closed")
		path = f"{self._base_path}{endpoint}"
		raw_body = json.dumps(body, allow_nan=False).encode("utf-8") if body is not None else None
		headers = {"Content-Type": "application/json"} if raw_body is not None else {}

		last_exc: Exception | None = None
		for attempt in range(2):
			conn = self._get_conn()
			try:
				conn.request(method, path, body=raw_body, headers=headers)
				resp = conn.getresponse()
				resp_bytes = resp.read()
				break
			except (http.client.HTTPException, OSError) as exc:
				last_exc = exc
				self._drop_conn()
				if attempt == 1:
					raise RuntimeError(f"HTTP {method} {self.url}{endpoint} failed: {exc}") from exc
		else:  # pragma: no cover
			raise RuntimeError(f"HTTP {method} {self.url}{endpoint} failed: {last_exc}")

		try:
			parsed = json.loads(resp_bytes.decode("utf-8"))
		except (UnicodeDecodeError, json.JSONDecodeError) as exc:
			raise RuntimeError(f"invalid JSON from {self.url}{endpoint} (status {resp.status}): {exc}") from exc
		if resp.status != 200:
			detail = parsed.get("error") if isinstance(parsed, dict) else parsed
			raise RuntimeError(f"server error HTTP {resp.status} from {endpoint}: {detail}")
		return parsed

	def health(self) -> dict[str, Any]:
		return self._request_json("GET", "/health")

	def predict(self, payload: Mapping[str, Any]) -> dict[str, Any]:
		forbidden = set(payload) & set(FORBIDDEN_IMAGE_PAYLOAD_KEYS)
		if forbidden:
			raise ValueError(f"forbidden camera key(s) in client predict payload: {sorted(forbidden)}")
		task = payload["task"]
		assert_no_planner_diagnostics(task)

		wire_payload: dict[str, Any] = {
			"q_t": np.asarray(payload["q_t"], dtype=np.float32).reshape(6).tolist(),
			"q_prev": None
			if payload.get("q_prev") is None
			else np.asarray(payload["q_prev"], dtype=np.float32).reshape(6).tolist(),
			"prev_action": None
			if payload.get("prev_action") is None
			else np.asarray(payload["prev_action"], dtype=np.float32).reshape(6).tolist(),
			"gripper_cmd": float(payload["gripper_cmd"]),
			"task": {k: task.get(k) for k in POLICY_FIELDS},
			"segment_frame": int(payload["segment_frame"]),
		}
		for cam_name in STUDENT_CAMERA_NAMES:
			wire_payload[cam_name] = encode_image(
				np.asarray(payload[cam_name], dtype=np.uint8), encoding=self._image_encoding
			)
		return self._request_json("POST", "/predict", wire_payload)

	def close(self) -> None:
		with self._conns_lock:
			if self._closed:
				return
			self._closed = True
			conns = list(self._conns)
			self._conns.clear()
		for conn in conns:
			with contextlib.suppress(Exception):
				conn.close()

	def __enter__(self) -> Self:
		return self

	def __exit__(self, *exc_info: object) -> None:
		self.close()


@dataclass
class ClientPolicyStats:
	"""Telemetry counters for S10 closed-loop execution."""

	calls: int = 0
	sync_calls: int = 0
	async_prefetches: int = 0
	cancelled_prefetches: int = 0
	stale_discarded: int = 0
	stop_events: int = 0
	total_latency_ms: float = 0.0

	@property
	def mean_latency_ms(self) -> float | None:
		return (self.total_latency_ms / self.calls) if self.calls > 0 else None


class ChunkClientPolicy:
	"""``SegmentPolicy`` client executing action chunks synchronously or asynchronously.

	Implements S10 queue reset and end-step stopping:
	- On ``start(task, segment_id)``: clears the action queue, cancels/invalidates any in-flight
	  async prefetch from a prior segment (tagged by ``(segment_id, generation, request_id)``),
	  and resets ``planned_stop_step = None``.
	- Whenever a chunk anchored at ``base_frame`` predicts ``k_hat = predicted_end_step`` in
	  ``0..H`` (passing optional ``end_threshold``), sets ``planned_stop_step = base_frame + k_hat``.
	- Whenever ``obs.segment_frame >= planned_stop_step`` (including ``k_hat == 0`` at frame 0),
	  immediately returns ``PolicyStep(done=True)`` without another predictor call.
	"""

	def __init__(
		self,
		predictor: ChunkPredictor,
		*,
		mode: str = "sync",
		replan_interval: int | None = None,
		prefetch_ahead: int | None = None,
		end_threshold: float | None = None,
	) -> None:
		if mode not in ("sync", "async"):
			raise ValueError(f"mode must be 'sync' or 'async', got {mode!r}")
		if replan_interval is not None and int(replan_interval) <= 0:
			raise ValueError(f"replan_interval must be > 0, got {replan_interval!r}")
		if prefetch_ahead is not None and int(prefetch_ahead) < 0:
			raise ValueError(f"prefetch_ahead must be >= 0, got {prefetch_ahead!r}")
		if end_threshold is not None and not (0.0 < float(end_threshold) <= 1.0):
			raise ValueError(f"end_threshold must be in (0, 1], got {end_threshold!r}")

		self.predictor = predictor
		self.mode = mode
		self.replan_interval = int(replan_interval) if replan_interval is not None else None
		self.prefetch_ahead = int(prefetch_ahead) if prefetch_ahead is not None else None
		self.end_threshold = float(end_threshold) if end_threshold is not None else None

		self._lock = threading.Lock()
		self._closed = False
		self._executor: ThreadPoolExecutor | None = (
			ThreadPoolExecutor(max_workers=2, thread_name_prefix="vla-prefetch") if self.mode == "async" else None
		)

		self._segment_id: int | None = None
		self._generation: int = 0
		self._request_seq: int = 0
		self._current_task: dict[str, Any] | None = None
		self._chunk_actions: np.ndarray | None = None
		self._chunk_anchor_frame: int = 0
		self._next_expected_frame: int = 0
		self._planned_stop_step: int | None = None
		self._pending_future: Future[dict[str, Any]] | None = None
		self._pending_tag: tuple[int, int, int] | None = None
		self._pending_base_frame: int = 0

		self.stats = ClientPolicyStats()

	@property
	def planned_stop_step(self) -> int | None:
		with self._lock:
			return self._planned_stop_step

	@property
	def queue_remaining(self) -> int:
		with self._lock:
			if self._chunk_actions is None:
				return 0
			end = self._chunk_anchor_frame + len(self._chunk_actions)
			return max(0, end - self._next_expected_frame)

	def start(self, task: dict[str, Any], segment_id: int) -> None:
		assert_no_planner_diagnostics(task)
		clean_task = {k: task.get(k) for k in POLICY_FIELDS}
		with self._lock:
			if self._closed:
				raise RuntimeError("ChunkClientPolicy is closed")
			self._generation += 1
			self._segment_id = int(segment_id)
			self._current_task = clean_task
			self._chunk_actions = None
			self._chunk_anchor_frame = 0
			self._next_expected_frame = 0
			self._planned_stop_step = None
			if self._pending_future is not None:
				if self._pending_future.done():
					self.stats.stale_discarded += 1
				else:
					self._pending_future.cancel()
					self.stats.cancelled_prefetches += 1
				self._pending_future = None
				self._pending_tag = None

	def _build_payload(self, obs: Any) -> dict[str, Any]:
		assert_no_planner_diagnostics(obs.task)
		clean_task = {k: obs.task.get(k) for k in POLICY_FIELDS}
		images = extract_student_images(obs.images())
		q_t = np.asarray(obs.q, dtype=np.float32).reshape(ACTION_DIM)
		q_prev = (
			q_t
			if getattr(obs, "q_prev", None) is None
			else np.asarray(obs.q_prev, dtype=np.float32).reshape(ACTION_DIM)
		)
		prev_action = (
			INITIAL_Q.astype(np.float32)
			if getattr(obs, "prev_action", None) is None
			else np.asarray(obs.prev_action, dtype=np.float32).reshape(ACTION_DIM)
		)
		return {
			**images,
			"q_t": q_t,
			"q_prev": q_prev,
			"prev_action": prev_action,
			"gripper_cmd": float(getattr(obs, "gripper_cmd", 0.0)),
			"task": clean_task,
			"segment_frame": int(obs.segment_frame),
		}

	def _apply_prediction_locked(
		self,
		result: Mapping[str, Any],
		*,
		base_frame: int,
		current_frame: int,
		tag: tuple[int, int, int],
		from_async: bool = False,
	) -> bool:
		if tag[:2] != (self._segment_id, self._generation):
			if not from_async:
				self.stats.stale_discarded += 1
			return False

		actions = np.asarray(result["actions"], dtype=np.float64)
		if actions.ndim != 2 or actions.shape[1] != ACTION_DIM or actions.shape[0] < 1:
			raise ValueError(f"invalid predicted actions shape {actions.shape}")
		horizon = int(actions.shape[0])

		probs = np.asarray(result["end_step_probs"], dtype=np.float64).reshape(-1)
		if probs.shape != (horizon + 2,):
			raise ValueError(f"end_step_probs shape {probs.shape} != expected ({horizon + 2},)")

		k_hat = int(result.get("predicted_end_step", int(np.argmax(probs))))
		if 0 <= k_hat <= horizon:
			if self.end_threshold is None or float(probs[k_hat]) >= self.end_threshold:
				self._planned_stop_step = base_frame + k_hat
			else:
				self._planned_stop_step = None
		else:
			self._planned_stop_step = None

		if current_frame < base_frame + horizon:
			self._chunk_actions = actions
			self._chunk_anchor_frame = base_frame
			self._next_expected_frame = max(self._next_expected_frame, current_frame)
			return True
		return False

	def _queue_covers_locked(self, segment_frame: int) -> bool:
		if self._chunk_actions is None:
			return False
		idx = segment_frame - self._chunk_anchor_frame
		return 0 <= idx < len(self._chunk_actions)

	def _maybe_consume_pending(self, *, current_frame: int, wait: bool) -> None:
		with self._lock:
			fut = self._pending_future
			tag = self._pending_tag
			base_frame = self._pending_base_frame
		if fut is None or tag is None:
			return
		if not wait and not fut.done():
			return

		try:
			result = fut.result()
		except CancelledError:
			with self._lock:
				if self._pending_future is fut:
					self._pending_future = None
					self._pending_tag = None
			return

		with self._lock:
			if self._pending_future is fut:
				self._pending_future = None
				self._pending_tag = None
			self._apply_prediction_locked(
				result, base_frame=base_frame, current_frame=current_frame, tag=tag, from_async=True
			)

	def _run_sync_predict(self, obs: Any) -> None:
		payload = self._build_payload(obs)
		base_frame = int(obs.segment_frame)
		with self._lock:
			if self._segment_id is None:
				raise RuntimeError("start() must be called before act()")
			self._request_seq += 1
			tag = (self._segment_id, self._generation, self._request_seq)

		result = self.predictor.predict(payload)

		with self._lock:
			self.stats.calls += 1
			self.stats.sync_calls += 1
			self.stats.total_latency_ms += float(result.get("latency_ms", 0.0))
			self._apply_prediction_locked(result, base_frame=base_frame, current_frame=base_frame, tag=tag)

	def _should_launch_prefetch_locked(self, segment_frame: int) -> bool:
		if self.mode != "async" or self._pending_future is not None or self._chunk_actions is None:
			return False
		if self._planned_stop_step is not None and self._planned_stop_step <= segment_frame + 2:
			return False
		horizon = len(self._chunk_actions)
		elapsed = segment_frame - self._chunk_anchor_frame
		remaining = (self._chunk_anchor_frame + horizon) - segment_frame
		if self.replan_interval is None and self.prefetch_ahead is None:
			interval = min(DEFAULT_ASYNC_REPLAN_INTERVAL, max(1, horizon // 2))
			return elapsed >= interval
		trigger = False
		if self.replan_interval is not None and elapsed >= self.replan_interval:
			trigger = True
		if self.prefetch_ahead is not None and remaining <= self.prefetch_ahead:
			trigger = True
		return trigger

	def _launch_async_prefetch(self, obs: Any) -> None:
		assert self._executor is not None
		payload = self._build_payload(obs)
		base_frame = int(obs.segment_frame)
		with self._lock:
			if self._closed or self._pending_future is not None or self._segment_id is None:
				return
			self._request_seq += 1
			tag = (self._segment_id, self._generation, self._request_seq)
			self.stats.async_prefetches += 1

			def _worker() -> dict[str, Any]:
				result = self.predictor.predict(payload)
				with self._lock:
					self.stats.calls += 1
					self.stats.total_latency_ms += float(result.get("latency_ms", 0.0))
					if tag[:2] != (self._segment_id, self._generation):
						self.stats.stale_discarded += 1
				return result

			self._pending_tag = tag
			self._pending_base_frame = base_frame
			self._pending_future = self._executor.submit(_worker)

	def _cancel_pending_locked(self) -> None:
		if self._pending_future is not None:
			self._pending_future.cancel()
			self._pending_future = None
			self._pending_tag = None
			self.stats.cancelled_prefetches += 1

	def act(self, obs: Any):
		from ..agent.student_rollout import PolicyStep

		assert_no_planner_diagnostics(obs.task)
		seg_frame = int(obs.segment_frame)
		with self._lock:
			if self._closed:
				raise RuntimeError("ChunkClientPolicy is closed")
			if self._segment_id is None or int(obs.segment_id) != self._segment_id:
				raise RuntimeError(f"act() got segment_id={obs.segment_id!r}, expected {self._segment_id!r}")

		# 1. Harvest any finished async prefetch matching current (segment_id, generation).
		self._maybe_consume_pending(current_frame=seg_frame, wait=False)

		# 2. Check planned end step BEFORE waiting/running any new inference.
		with self._lock:
			if self._planned_stop_step is not None and seg_frame >= self._planned_stop_step:
				self._cancel_pending_locked()
				self.stats.stop_events += 1
				return PolicyStep(done=True)
			need_sync = not self._queue_covers_locked(seg_frame)
			if (
				self.mode == "sync"
				and self.replan_interval is not None
				and self._chunk_actions is not None
				and seg_frame - self._chunk_anchor_frame >= self.replan_interval
			):
				need_sync = True
			has_pending = self._pending_future is not None

		# 3. Handle missing queue / sync replan.
		if need_sync:
			if self.mode == "async" and has_pending:
				self._maybe_consume_pending(current_frame=seg_frame, wait=True)
				with self._lock:
					if self._planned_stop_step is not None and seg_frame >= self._planned_stop_step:
						self._cancel_pending_locked()
						self.stats.stop_events += 1
						return PolicyStep(done=True)
					still_missing = not self._queue_covers_locked(seg_frame)
			else:
				still_missing = True

			if still_missing:
				self._run_sync_predict(obs)
				with self._lock:
					if self._planned_stop_step is not None and seg_frame >= self._planned_stop_step:
						self._cancel_pending_locked()
						self.stats.stop_events += 1
						return PolicyStep(done=True)

		# 4. In async mode, fire a non-blocking prefetch on the main-thread-extracted payload if due.
		with self._lock:
			should_prefetch = self._should_launch_prefetch_locked(seg_frame)
		if should_prefetch:
			self._launch_async_prefetch(obs)

		# 5. Execute action at seg_frame.
		with self._lock:
			if self._chunk_actions is None:
				raise RuntimeError(f"no action chunk available at segment_frame={seg_frame}")
			idx = seg_frame - self._chunk_anchor_frame
			if not (0 <= idx < len(self._chunk_actions)):
				raise RuntimeError(
					f"segment_frame={seg_frame} out of active chunk range "
					f"[{self._chunk_anchor_frame}, {self._chunk_anchor_frame + len(self._chunk_actions)})"
				)
			target = np.asarray(self._chunk_actions[idx], dtype=np.float64).copy()
			self._next_expected_frame = max(self._next_expected_frame, seg_frame + 1)
		return PolicyStep(target=target)

	def close(self) -> None:
		with self._lock:
			if self._closed:
				return
			self._closed = True
			self._cancel_pending_locked()
			executor = self._executor
			self._executor = None
		if executor is not None:
			executor.shutdown(wait=True, cancel_futures=True)

	def __enter__(self) -> Self:
		return self

	def __exit__(self, *exc_info: object) -> None:
		self.close()

	def __del__(self) -> None:
		with contextlib.suppress(Exception):
			self.close()


__all__ = [
	"DEFAULT_ASYNC_REPLAN_INTERVAL",
	"ChunkClientPolicy",
	"ClientPolicyStats",
	"LocalChunkPredictor",
	"RemoteChunkPredictor",
	"extract_student_images",
]
