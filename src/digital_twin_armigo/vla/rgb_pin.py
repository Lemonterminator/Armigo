"""Shared pre-intervention prefix for the first-grasp tail experiment (docs/vla/GRASP_TAIL_AB_HANDOFF_2026-11-23.md r3).

Two runs of one layout diverge long before any intervention: rendering one simulator state twice gives RGB images
that differ by one 8-bit level on a few dozen pixels near shadow edges (``student_rollout.RENDER_MAX_LEVEL``), and the
ACT server's CUDA bf16 inference turns that into a different command (first command up to ~1e-2 rad). Physics,
depth noise, perception and inference are otherwise repeatable.

So arm A **records** the two RGB streams of every ACT request (``top_rgb_v2``, ``wrist``) together with a digest of
everything else in the request. Every other arm (A', A+wait, B) runs the same live loop, renders and builds its own
request, and **pins** only those two RGB arrays to A's, request by request, until its own hand-off point:

- A' (``act``) and B (``expert-once``): A's switch observation (``GraspTail`` ``switch.frame``) when A triggered;
- A+wait (``act-wait``): A's first-grasp ``done`` observation when A triggered and the clamp was not yet confirmed
  there (the observation A+wait starts waiting at);
- no hand-off (the arm cannot differ from A): every request of the episode.

From the hand-off on, the arm's requests are its own renders (live). A pinned request must match A's request exactly
in everything but the two RGB arrays (state, previous command, gripper command, task, segment frame, top depth): the
first mismatch makes the rest of the episode live and is logged (``rgb_pin.json``). The renders still happen, so the
depth-noise RNG (shared with perception) advances exactly as in A.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..mujoco_cell.physics_expert import ExpertParams
from .client_policy import ChunkClientPolicy

PIN_VERSION = "rgb-pin-v1"
PIN_RGB_KEYS = ("top_rgb_v2", "wrist")
RECORD_FILE = "act_rgb.npz"
LOG_FILE = "rgb_pin.json"
ROLES = ("record", "pin")
# Arguments that may differ between A and a pinned arm (everything else in the manifest must match).
ARM_ARGS = ("grasp_tail_mode", "record_act_rgb", "pin_rgb_from", "clamp_wait", "clamp_wait_obstruct", "pin_handoff")


def meta_digest(payload: dict[str, Any]) -> str:
	"""SHA-256 of every request field except the pinned RGB arrays (exact bytes, dtype and shape)."""
	h = hashlib.sha256()
	for k in sorted(payload):
		if k in PIN_RGB_KEYS:
			continue
		v = payload[k]
		h.update(k.encode())
		if isinstance(v, np.ndarray):
			h.update(f"{v.dtype.str}{v.shape}".encode())
			h.update(np.ascontiguousarray(v).tobytes())
		else:
			h.update(json.dumps(v, sort_keys=True).encode())
	return h.hexdigest()


def _key(i: int, obs: Any) -> list[int]:
	return [int(i), int(obs.frame), int(obs.segment_id), int(obs.segment_frame)]


class RgbRecorder:
	"""Arm A: the request sequence of one episode."""

	def __init__(self):
		self.keys: list[list[int]] = []
		self.digests: list[str] = []
		self.q_t: list[np.ndarray] = []
		self.rgb: dict[str, list[np.ndarray]] = {k: [] for k in PIN_RGB_KEYS}

	def add(self, i: int, obs: Any, payload: dict[str, Any]) -> None:
		if i != len(self.keys):
			raise RuntimeError(f"request {i} recorded out of order (have {len(self.keys)})")
		self.keys.append(_key(i, obs))
		self.digests.append(meta_digest(payload))
		self.q_t.append(np.asarray(payload["q_t"]).copy())
		for k in PIN_RGB_KEYS:
			self.rgb[k].append(np.asarray(payload[k], dtype=np.uint8).copy())

	def save(self, path: Path) -> None:
		n = len(self.keys)
		np.savez_compressed(
			path,
			version=np.array(PIN_VERSION),
			keys=np.asarray(self.keys, dtype=np.int64).reshape(n, 4),
			digests=np.asarray(self.digests, dtype="U64").reshape(n),
			q_t=np.asarray(self.q_t, dtype=np.float32).reshape(n, -1),
			**{k: np.asarray(v, dtype=np.uint8) for k, v in self.rgb.items()},
		)

	def log(self) -> dict[str, Any]:
		return {"version": PIN_VERSION, "role": "record", "requests": len(self.keys)}


def handoff_frame(mode: str, first_grasp: dict | None) -> tuple[int | None, str]:
	"""The episode frame from which arm ``mode`` may differ from A, from A's ``grasp_tail.json`` ``first_grasp``
	(None: never, pin the whole episode) and why."""
	if first_grasp is None:
		return None, "no_first_grasp"
	if mode == "clamp-wait":  # StudentTwin(clamp_wait=True): any first grasp ACT ends before the clamp is confirmed
		done = [e for e in first_grasp["events"] if e["kind"] == "act_done"]
		if not done:
			return None, "no_act_done"
		if done[0]["clamp_run"] >= ExpertParams.clamp_still_frames:
			return None, "clamped_at_done"
		return int(done[0]["frame"]), "act_done_unclamped"
	if first_grasp.get("decision") != "triggered":
		return None, f"not_triggered:{first_grasp.get('decision')}"
	if mode in ("act", "expert-once"):
		return int(first_grasp["switch"]["frame"]), "switch"
	if mode == "act-wait":
		done = [e for e in first_grasp["events"] if e["kind"] == "act_done"]
		if not done:
			return None, "no_act_done"
		if not done[0].get("wait_eligible"):
			return None, "clamped_at_done"
		return int(done[0]["frame"]), "act_done_unclamped"
	raise ValueError(f"unknown grasp_tail mode {mode!r}")


class RgbPin:
	"""A pinned arm: replaces the RGB of request i by A's request i until the hand-off or the first mismatch."""

	def __init__(self, source_episode: Path, mode: str):
		source_episode = Path(source_episode)
		with np.load(source_episode / RECORD_FILE) as z:
			if str(z["version"]) != PIN_VERSION:
				raise ValueError(f"{source_episode / RECORD_FILE} is {z['version']}, not {PIN_VERSION}")
			self.keys = z["keys"].tolist()
			self.digests = [str(d) for d in z["digests"]]
			self.q_t = z["q_t"]
			self.rgb = {k: z[k] for k in PIN_RGB_KEYS}
		tail = json.loads((source_episode / "grasp_tail.json").read_text(encoding="utf-8"))
		self.handoff, self.handoff_reason = handoff_frame(mode, tail.get("first_grasp"))
		n = len(self.keys)
		self.cut = n if self.handoff is None else next((i for i, k in enumerate(self.keys) if k[1] >= self.handoff), n)
		self.source = str(source_episode)
		self.mode = mode
		self.live_from: dict[str, Any] | None = None  # first live request and why
		self.pinned = 0
		self.requests = 0
		self.rgb_diff: list[dict[str, Any]] = []  # pinned requests whose own render differed from A's

	def _go_live(self, i: int, obs: Any, reason: str, **info) -> None:
		self.live_from = {
			"request": int(i),
			"frame": int(obs.frame),
			"segment_id": int(obs.segment_id),
			"segment_frame": int(obs.segment_frame),
			"reason": reason,
			**info,
		}

	def apply(self, i: int, obs: Any, payload: dict[str, Any]) -> dict[str, Any]:
		self.requests += 1
		if self.live_from is not None:
			return payload
		if i >= self.cut:
			at_handoff = self.handoff is not None and int(obs.frame) >= self.handoff
			self._go_live(i, obs, "handoff" if at_handoff else "source_exhausted")
			return payload
		key = _key(i, obs)
		if key != self.keys[i]:
			self._go_live(i, obs, "mismatch_key", source_key=self.keys[i])
			return payload
		if meta_digest(payload) != self.digests[i]:
			dq = float(np.abs(np.asarray(payload["q_t"], dtype=np.float64) - self.q_t[i]).max())
			self._go_live(i, obs, "mismatch_request", q_t_max_abs=dq)
			return payload
		diff = {}
		for k in PIN_RGB_KEYS:
			own = np.asarray(payload[k])
			d = np.abs(own.astype(np.int16) - self.rgb[k][i].astype(np.int16))
			if d.any():
				diff[k] = [int(np.any(d, axis=-1).sum()), int(d.max())]
		if diff:
			self.rgb_diff.append({"request": int(i), "frame": int(obs.frame), **diff})
		self.pinned += 1
		return {**payload, **{k: self.rgb[k][i].copy() for k in PIN_RGB_KEYS}}

	def log(self) -> dict[str, Any]:
		return {
			"version": PIN_VERSION,
			"role": "pin",
			"mode": self.mode,
			"source": self.source,
			"source_requests": len(self.keys),
			"handoff_frame": self.handoff,
			"handoff_reason": self.handoff_reason,
			"cut_request": self.cut,
			"requests": self.requests,
			"pinned": self.pinned,
			"live_from": self.live_from,
			"mismatch": bool(self.live_from and self.live_from["reason"].startswith("mismatch")),
			"rgb_diff_requests": len(self.rgb_diff),
			"rgb_diff": self.rgb_diff,
		}


class RgbPinClientPolicy(ChunkClientPolicy):
	"""``ChunkClientPolicy`` that records (arm A) or pins (other arms) the RGB of each request. Sync mode only, so
	request i is always built on the main thread right before its prediction."""

	def __init__(self, predictor, *, recorder: RgbRecorder | None = None, pin: RgbPin | None = None, **kw):
		if (recorder is None) == (pin is None):
			raise ValueError("give exactly one of recorder / pin")
		if kw.get("mode", "sync") != "sync":
			raise ValueError("RGB pinning needs mode='sync'")
		super().__init__(predictor, **kw)
		self.recorder, self.pin = recorder, pin
		self._request_index = 0

	def _build_payload(self, obs: Any) -> dict[str, Any]:
		payload = super()._build_payload(obs)
		i = self._request_index
		self._request_index += 1
		if self.recorder is not None:
			self.recorder.add(i, obs, payload)
			return payload
		return self.pin.apply(i, obs, payload)

	def save(self, ep_dir: Path) -> None:
		ep_dir = Path(ep_dir)
		if not ep_dir.is_dir():
			return
		if self.recorder is not None:
			self.recorder.save(ep_dir / RECORD_FILE)
			log = self.recorder.log()
		else:
			log = self.pin.log()
		(ep_dir / LOG_FILE).write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")


def check_source_manifest(source: dict[str, Any], own: dict[str, Any]) -> None:
	"""A pinned arm may only pin from an A run with the same code, inputs and arguments (bar the arm's own)."""
	if source.get("args", {}).get("record_act_rgb") is not True:
		raise ValueError("the pin source was not run with --record-act-rgb")
	for k in ("protocol", "code"):
		if source.get(k) != own.get(k):
			raise ValueError(f"the pin source differs from this run in {k!r}")
	src_in = {k: v for k, v in (source.get("inputs_sha256") or {}).items() if k != "pin_source"}
	own_in = {k: v for k, v in (own.get("inputs_sha256") or {}).items() if k != "pin_source"}
	if src_in != own_in:
		raise ValueError("the pin source differs from this run in its inputs")
	a = {k: v for k, v in source.get("args", {}).items() if k not in ARM_ARGS}
	b = {k: v for k, v in own.get("args", {}).items() if k not in ARM_ARGS}
	diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
	if diff:
		raise ValueError(f"the pin source differs from this run in arguments {diff}")
