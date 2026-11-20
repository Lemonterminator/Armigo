"""Closed-loop VLA chunk prediction server and local predictor adapter (plan S10).

Runs inside ``E:\\il\\lerobot-cuda`` to serve trained ``SmolVLAWithEndHead`` / ``ACTWithEndHead``
checkpoints over HTTP (``/health`` and ``/predict``) using only stdlib ``http.server`` + JSON
with base64 ``.npy`` / PNG images. Also cleanly importable in the repo ``.venv`` without
requiring ``torch`` or ``lerobot`` at module import time.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import threading
import time
from collections.abc import Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Protocol, Self, runtime_checkable

import numpy as np

from .contract import (
	ACTION_DIM,
	FORBIDDEN_CAMERA_NAMES,
	STATE_DIM,
	STUDENT_CAMERA_NAMES,
	STUDENT_CAMERAS,
	assert_no_planner_diagnostics,
	encode_state_from_task,
)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000

CAMERA_INPUT_SHAPES: dict[str, tuple[int, int, int]] = {
	"top_rgb_v2": (416, 512, 3),
	"top_depth_v2": (416, 512, 3),
	"wrist": (480, 640, 3),
}

FORBIDDEN_IMAGE_PAYLOAD_KEYS: tuple[str, ...] = (
	*FORBIDDEN_CAMERA_NAMES,
	*(f"observation.images.{name}" for name in FORBIDDEN_CAMERA_NAMES),
)

REQUIRED_PREDICT_KEYS: tuple[str, ...] = (
	*STUDENT_CAMERA_NAMES,
	"q_t",
	"q_prev",
	"prev_action",
	"gripper_cmd",
	"task",
	"segment_frame",
)

_NPY_MAGIC = b"\x93NUMPY"
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def encode_image(img: np.ndarray, *, encoding: str = "npy") -> dict[str, Any]:
	"""Encode an ``(H, W, 3)`` uint8 RGB image into a JSON-serializable payload dict."""
	arr = np.asarray(img)
	if arr.dtype != np.uint8:
		raise TypeError(f"expected uint8 image, got {arr.dtype}")
	if arr.ndim != 3 or arr.shape[2] != 3:
		raise ValueError(f"expected (H, W, 3) image, got shape {arr.shape}")
	arr = np.ascontiguousarray(arr)
	if encoding == "npy":
		buf = io.BytesIO()
		np.save(buf, arr, allow_pickle=False)
		return {"encoding": "npy", "data": base64.b64encode(buf.getvalue()).decode("ascii")}
	if encoding == "png":
		import cv2

		bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
		ok, encoded = cv2.imencode(".png", bgr)
		if not ok:
			raise RuntimeError("cv2.imencode('.png', ...) failed")
		return {"encoding": "png", "data": base64.b64encode(encoded.tobytes()).decode("ascii")}
	if encoding == "raw":
		return {
			"encoding": "raw",
			"shape": list(arr.shape),
			"dtype": "uint8",
			"data": base64.b64encode(arr.tobytes()).decode("ascii"),
		}
	raise ValueError(f"unsupported image encoding {encoding!r}; expected 'npy', 'png', or 'raw'")


def _decode_png_bytes(raw: bytes) -> np.ndarray:
	import cv2

	buf = np.frombuffer(raw, dtype=np.uint8)
	bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
	if bgr is None:
		raise ValueError("failed to decode PNG image bytes")
	return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def decode_image(value: Any) -> np.ndarray:
	"""Decode an image from ndarray, nested list, base64 string, or ``{encoding, data}`` dict."""
	if isinstance(value, np.ndarray):
		arr = value
	elif isinstance(value, Mapping):
		enc = str(value.get("encoding", "")).lower()
		data = value.get("data")
		if not isinstance(data, str):
			raise TypeError("image payload dict must contain base64 string 'data'")
		raw = base64.b64decode(data.encode("ascii"))
		if enc == "npy":
			arr = np.load(io.BytesIO(raw), allow_pickle=False)
		elif enc == "png":
			arr = _decode_png_bytes(raw)
		elif enc == "raw":
			shape = tuple(int(v) for v in value["shape"])
			dtype = np.dtype(value.get("dtype", "uint8"))
			arr = np.frombuffer(raw, dtype=dtype).reshape(shape)
		else:
			raise ValueError(f"unknown image payload encoding {enc!r}")
	elif isinstance(value, str):
		raw = base64.b64decode(value.encode("ascii"))
		if raw.startswith(_NPY_MAGIC):
			arr = np.load(io.BytesIO(raw), allow_pickle=False)
		elif raw.startswith(_PNG_MAGIC):
			arr = _decode_png_bytes(raw)
		else:
			raise ValueError("unrecognized base64 image payload (expected .npy or .png)")
	else:
		arr = np.asarray(value)

	if arr.dtype != np.uint8:
		raise ValueError(f"decoded image must be uint8, got {arr.dtype}")
	if arr.ndim != 3 or arr.shape[2] != 3:
		raise ValueError(f"decoded image must have shape (H, W, 3), got {arr.shape}")
	return np.ascontiguousarray(arr)


def validate_predict_payload(
	payload: Mapping[str, Any],
	expected_shapes: Mapping[str, tuple[int, int, int]] | None = None,
) -> dict[str, Any]:
	"""Validate a ``/predict`` payload, enforce D1/R1 leakage guards, and convert to model inputs."""
	if not isinstance(payload, Mapping):
		raise TypeError("predict payload must be a mapping")

	forbidden = set(payload) & set(FORBIDDEN_IMAGE_PAYLOAD_KEYS)
	if forbidden:
		raise ValueError(f"forbidden camera key(s) in student prediction payload: {sorted(forbidden)}")

	missing = [k for k in REQUIRED_PREDICT_KEYS if k not in payload]
	if missing:
		raise ValueError(f"missing required prediction payload key(s): {missing}")

	task = payload["task"]
	if not isinstance(task, Mapping):
		raise TypeError("payload['task'] must be a mapping")
	assert_no_planner_diagnostics(task)
	if "kind" not in task or "instruction" not in task:
		raise ValueError("payload['task'] must include 'kind' and 'instruction'")

	seg_frame_raw = payload["segment_frame"]
	if isinstance(seg_frame_raw, bool) or not isinstance(seg_frame_raw, (int, np.integer)):
		raise TypeError(f"segment_frame must be an integer, got {seg_frame_raw!r}")
	segment_frame = int(seg_frame_raw)

	shapes = dict(expected_shapes) if expected_shapes is not None else dict(CAMERA_INPUT_SHAPES)
	images_chw: dict[str, np.ndarray] = {}
	for cam_name, cam_key in zip(STUDENT_CAMERA_NAMES, STUDENT_CAMERAS, strict=True):
		img_hwc = decode_image(payload[cam_name])
		exp_shape = shapes.get(cam_name)
		if exp_shape is not None and img_hwc.shape != tuple(exp_shape):
			raise ValueError(f"{cam_name} shape {img_hwc.shape} != expected {tuple(exp_shape)}")
		chw = np.transpose(img_hwc, (2, 0, 1)).astype(np.float32) / 255.0
		images_chw[cam_key] = np.ascontiguousarray(chw, dtype=np.float32)

	state_vec = encode_state_from_task(
		q_t=payload["q_t"],
		q_prev=payload["q_prev"],
		prev_action=payload["prev_action"],
		gripper_cmd=payload["gripper_cmd"],
		task=task,
		segment_frame=segment_frame,
	)

	return {
		"images": images_chw,
		"state": state_vec,
		"task": str(task["instruction"]),
		"raw_task": dict(task),
		"segment_frame": segment_frame,
	}


@runtime_checkable
class ChunkPredictor(Protocol):
	"""Common interface shared by ``LocalChunkPredictor`` and ``RemoteChunkPredictor``."""

	def health(self) -> dict[str, Any]: ...

	def predict(self, payload: Mapping[str, Any]) -> dict[str, Any]: ...


def _to_numpy(arr: Any) -> np.ndarray:
	if type(arr).__module__.startswith("torch"):
		return arr.detach().cpu().numpy()
	return np.asarray(arr)


def _shapes_from_model(model: Any) -> dict[str, tuple[int, int, int]] | None:
	cam_shapes = getattr(model, "camera_shapes", None)
	if not isinstance(cam_shapes, Mapping):
		return None
	out: dict[str, tuple[int, int, int]] = {}
	for cam_name, cam_key in zip(STUDENT_CAMERA_NAMES, STUDENT_CAMERAS, strict=True):
		if cam_key in cam_shapes:
			c, h, w = (int(v) for v in cam_shapes[cam_key])
			out[cam_name] = (h, w, c)
	return out if len(out) == len(STUDENT_CAMERA_NAMES) else None


class LocalChunkPredictor:
	"""In-process ``ChunkPredictor`` wrapping a VLA model (``predict_chunk``) or callable.

	Works identically for real PyTorch checkpoints (``from_checkpoint``) and lightweight
	pure-numpy test doubles without sockets or ``torch`` dependencies.
	"""

	def __init__(
		self,
		model_or_fn: Any,
		*,
		family: str | None = None,
		horizon: int | None = None,
		device: str = "cpu",
		expected_shapes: Mapping[str, tuple[int, int, int]] | None = None,
	) -> None:
		self._target = model_or_fn
		self._lock = threading.Lock()
		self.family = str(family if family is not None else getattr(model_or_fn, "family", "unknown"))
		raw_h = horizon if horizon is not None else getattr(model_or_fn, "horizon", None)
		self.horizon: int | None = int(raw_h) if raw_h is not None else None
		self.device = str(device)
		inferred_shapes = _shapes_from_model(model_or_fn)
		self.expected_shapes: dict[str, tuple[int, int, int]] = (
			{k: tuple(v) for k, v in expected_shapes.items()}
			if expected_shapes is not None
			else (inferred_shapes if inferred_shapes is not None else dict(CAMERA_INPUT_SHAPES))
		)
		self._is_chunk_model = hasattr(model_or_fn, "predict_chunk") and callable(model_or_fn.predict_chunk)
		self._is_torch_model = self._is_chunk_model and type(model_or_fn).__module__.startswith(
			"digital_twin_armigo.vla.models"
		)
		if not self._is_torch_model and "torch" in sys.modules:
			torch_mod = sys.modules["torch"]
			if isinstance(model_or_fn, torch_mod.nn.Module):
				self._is_torch_model = True

	@classmethod
	def from_checkpoint(
		cls,
		ckpt_dir: str | Path,
		*,
		device: str = "auto",
		local_files_only: bool = True,
	) -> LocalChunkPredictor:
		import torch

		from .models import load_vla_checkpoint

		resolved = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else str(device)
		model = load_vla_checkpoint(ckpt_dir, device=resolved, local_files_only=local_files_only)
		return cls(
			model,
			family=getattr(model, "family", "unknown"),
			horizon=int(model.horizon),
			device=resolved,
		)

	def health(self) -> dict[str, Any]:
		return {
			"status": "ok",
			"family": self.family,
			"horizon": self.horizon,
			"num_end_classes": (self.horizon + 2) if self.horizon is not None else None,
			"state_dim": STATE_DIM,
			"action_dim": ACTION_DIM,
			"device": self.device,
			"cameras": list(STUDENT_CAMERA_NAMES),
		}

	def predict(self, payload: Mapping[str, Any]) -> dict[str, Any]:
		prepared = validate_predict_payload(payload, expected_shapes=self.expected_shapes)
		t0 = time.perf_counter()
		with self._lock:
			if self._is_chunk_model:
				if self._is_torch_model:
					import torch

					batch: dict[str, Any] = {
						cam_key: torch.from_numpy(arr).unsqueeze(0) for cam_key, arr in prepared["images"].items()
					}
					batch["observation.state"] = torch.from_numpy(prepared["state"]).unsqueeze(0)
					batch["task"] = [prepared["task"]]
				else:
					batch = {cam_key: arr[None, ...] for cam_key, arr in prepared["images"].items()}
					batch["observation.state"] = prepared["state"][None, ...]
					batch["task"] = [prepared["task"]]
					batch["segment_frame"] = prepared["segment_frame"]
					batch["raw_task"] = prepared["raw_task"]
				actions_raw, probs_raw = self._target.predict_chunk(batch)
				actions = _to_numpy(actions_raw).astype(np.float64)
				probs = _to_numpy(probs_raw).astype(np.float64)
				if actions.ndim == 3 and actions.shape[0] == 1:
					actions = actions[0]
				if probs.ndim == 2 and probs.shape[0] == 1:
					probs = probs[0]
			elif callable(self._target):
				out = self._target(prepared)
				actions = np.asarray(out["actions"], dtype=np.float64)
				probs = np.asarray(out["end_step_probs"], dtype=np.float64)
			elif hasattr(self._target, "predict") and callable(self._target.predict):
				out = self._target.predict(payload)
				actions = np.asarray(out["actions"], dtype=np.float64)
				probs = np.asarray(out["end_step_probs"], dtype=np.float64)
			else:
				raise TypeError(f"unsupported predictor target {type(self._target).__name__}")
		latency_ms = (time.perf_counter() - t0) * 1000.0

		if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
			raise ValueError(f"predicted actions must have shape (H, {ACTION_DIM}), got {actions.shape}")
		horizon = actions.shape[0]
		probs = probs.reshape(-1)
		if probs.shape != (horizon + 2,):
			raise ValueError(f"end_step_probs must have length {horizon + 2}, got {probs.shape}")
		if not np.isfinite(actions).all() or not np.isfinite(probs).all():
			raise ValueError("model prediction contains non-finite values")
		if self.horizon is None:
			self.horizon = horizon

		predicted_end_step = int(np.argmax(probs))
		return {
			"actions": actions.tolist(),
			"end_step_probs": probs.tolist(),
			"predicted_end_step": predicted_end_step,
			"latency_ms": float(latency_ms),
		}


class _VLARequestHandler(BaseHTTPRequestHandler):
	protocol_version = "HTTP/1.1"

	def log_message(self, format: str, *args: Any) -> None:
		if getattr(self.server, "verbose", False):
			super().log_message(format, *args)

	def _send_json(self, status: int, body: dict[str, Any]) -> None:
		raw = json.dumps(body, allow_nan=False).encode("utf-8")
		self.send_response(status)
		self.send_header("Content-Type", "application/json")
		self.send_header("Content-Length", str(len(raw)))
		self.end_headers()
		self.wfile.write(raw)

	def do_GET(self) -> None:
		if self.path.split("?", 1)[0] == "/health":
			predictor: ChunkPredictor = self.server.predictor  # type: ignore[attr-defined]
			self._send_json(HTTPStatus.OK, predictor.health())
			return
		self._send_json(HTTPStatus.NOT_FOUND, {"error": f"unknown endpoint {self.path!r}"})

	def do_POST(self) -> None:
		endpoint = self.path.split("?", 1)[0]
		if endpoint == "/shutdown":
			self._send_json(HTTPStatus.OK, {"status": "shutting_down"})
			threading.Thread(target=self.server.shutdown, daemon=True).start()
			return
		if endpoint != "/predict":
			self._send_json(HTTPStatus.NOT_FOUND, {"error": f"unknown endpoint {self.path!r}"})
			return
		try:
			length = int(self.headers.get("Content-Length", "0"))
		except ValueError:
			self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid Content-Length"})
			return
		if length <= 0:
			self._send_json(HTTPStatus.BAD_REQUEST, {"error": "empty request body"})
			return
		raw_body = self.rfile.read(length)
		try:
			payload = json.loads(raw_body.decode("utf-8"))
		except (UnicodeDecodeError, json.JSONDecodeError) as exc:
			self._send_json(HTTPStatus.BAD_REQUEST, {"error": f"invalid JSON: {exc}"})
			return
		predictor: ChunkPredictor = self.server.predictor  # type: ignore[attr-defined]
		try:
			result = predictor.predict(payload)
		except (ValueError, KeyError, TypeError) as exc:
			self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
			return
		except Exception as exc:  # noqa: BLE001
			self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"{type(exc).__name__}: {exc}"})
			return
		self._send_json(HTTPStatus.OK, result)


class _ThreadingVLAHTTPServer(ThreadingHTTPServer):
	allow_reuse_address = True
	daemon_threads = True

	def __init__(self, server_address: tuple[str, int], predictor: ChunkPredictor, *, verbose: bool = False) -> None:
		self.predictor = predictor
		self.verbose = verbose
		super().__init__(server_address, _VLARequestHandler)


class VLAPolicyServer:
	"""Threaded stdlib HTTP server exposing ``/health`` and ``/predict`` for a ``ChunkPredictor``."""

	def __init__(
		self,
		predictor: ChunkPredictor,
		*,
		host: str = DEFAULT_HOST,
		port: int = DEFAULT_PORT,
		verbose: bool = False,
	) -> None:
		self.predictor = predictor
		self._httpd = _ThreadingVLAHTTPServer((host, int(port)), predictor, verbose=verbose)
		self.host = str(self._httpd.server_address[0])
		self.port = int(self._httpd.server_address[1])
		self._thread: threading.Thread | None = None
		self._closed = False

	@property
	def url(self) -> str:
		return f"http://{self.host}:{self.port}"

	def start(self) -> VLAPolicyServer:
		if self._thread is not None:
			return self
		self._thread = threading.Thread(target=self._httpd.serve_forever, name="vla-policy-server", daemon=True)
		self._thread.start()
		return self

	def serve_forever(self) -> None:
		self._httpd.serve_forever()

	def close(self) -> None:
		if self._closed:
			return
		self._closed = True
		self._httpd.shutdown()
		self._httpd.server_close()
		if self._thread is not None:
			self._thread.join(timeout=5.0)
			self._thread = None

	def __enter__(self) -> Self:
		return self.start()

	def __exit__(self, *exc_info: object) -> None:
		self.close()


def _write_ready_file(path: Path, payload: dict[str, Any]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	tmp = path.with_suffix(path.suffix + ".tmp")
	tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
	tmp.replace(path)


def build_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	p.add_argument("--ckpt", type=Path, required=True, help="Trained VLA checkpoint directory")
	p.add_argument("--host", default=DEFAULT_HOST)
	p.add_argument("--port", type=int, default=DEFAULT_PORT)
	p.add_argument("--device", default="auto", help="auto | cpu | cuda[:idx]")
	p.add_argument("--allow-download", action="store_true", help="Allow HF downloads if weights are not cached")
	p.add_argument("--ready-file", type=Path, default=None, help="Optional JSON file written once bound and ready")
	p.add_argument("--quiet", action="store_true", help="Suppress per-request HTTP logging")
	return p


def main(argv: list[str] | None = None) -> int:
	args = build_parser().parse_args(argv)
	predictor = LocalChunkPredictor.from_checkpoint(
		args.ckpt,
		device=args.device,
		local_files_only=not args.allow_download,
	)
	server = VLAPolicyServer(
		predictor,
		host=args.host,
		port=args.port,
		verbose=not args.quiet,
	)
	ready_payload = {
		"ready": True,
		"host": server.host,
		"port": server.port,
		"url": server.url,
		"health": predictor.health(),
	}
	if args.ready_file is not None:
		_write_ready_file(args.ready_file, ready_payload)
	print(json.dumps(ready_payload), flush=True)
	try:
		server.serve_forever()
	except KeyboardInterrupt:
		pass
	finally:
		server.close()
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
