"""Long-lived anomaly score server for the simulator (run in the anomaly env, see ``requirements.txt``).

Usage::

    E:/venvs/armigo-anomaly/Scripts/python scripts/anomaly/score_server.py \
        --model top=models/patchcore_top --model wrist_cam=models/patchcore_wrist_cam [--cpu]

Loads one exported model per camera (``best.pt`` + ``best.json`` written by ``train_eval.py``), and/or the supervised
cross detector of a spec written by ``export_sup.py`` (``--sup SPEC.json``: scores of its cameras are ``u`` values with
threshold 0, see ``sup_scorer.py``; ``--model`` for the same camera wins), and then speaks a
line-delimited JSON protocol on stdin/stdout, flushing after every line:

* server after loading: ``{"ready": true, "cameras": {"top": {"threshold": .., "model": .., "crop_size": ..}, ..}}``
* request: ``{"id": any, "camera": "top", "shape": [h, w, 3], "rgb_b64": "<raw uint8 RGB>"}``
* response: ``{"id": same, "score": float, "threshold": float, "anomalous": bool}`` (score >= threshold) or ``{"id": same, "error": ..}``
* ``{"cmd": "quit"}`` or EOF exits 0.

Crops arrive already RGB; they are resized to the model's ``crop_size`` with ``cv2.INTER_AREA`` exactly as
``score.py`` does. Only protocol lines go to stdout; everything else goes to stderr.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from score import load
from sup_scorer import SupScorer
from train_eval import score


def _protocol_stream():
	"""Return a stdout-like stream for protocol lines and point ``sys.stdout`` at stderr.

	The file descriptor behind stdout is redirected to stderr too, so C libraries and any stray ``print`` cannot
	corrupt the protocol. The returned stream writes to the original stdout (a private duplicate of its fd).
	"""
	stream = sys.stdout
	try:
		stream.flush()
		fd = os.dup(stream.fileno())
		os.dup2(sys.stderr.fileno(), stream.fileno())
		sys.stdout = sys.stderr
		return os.fdopen(fd, "w", encoding="utf-8", buffering=1)
	except (OSError, ValueError, AttributeError):
		sys.stdout = sys.stderr
		return stream


def _send(stream, obj: dict[str, Any]) -> None:
	stream.write(json.dumps(obj) + "\n")
	stream.flush()


def _parse_model_spec(spec: str) -> tuple[str, Path]:
	camera, sep, model_dir = spec.partition("=")
	if not sep or not camera or not model_dir:
		raise argparse.ArgumentTypeError(f"expected CAMERA=MODEL_DIR, got {spec!r}")
	return camera, Path(model_dir)


def _decode_image(request: dict[str, Any]) -> np.ndarray:
	shape = request.get("shape")
	if not (isinstance(shape, list) and len(shape) == 3 and all(isinstance(v, int) and v > 0 for v in shape)):
		raise ValueError(f"bad shape: {shape!r}")
	h, w, c = shape
	if c != 3:
		raise ValueError(f"bad shape: {shape!r}")
	try:
		raw = base64.b64decode(request["rgb_b64"], validate=True)
	except (KeyError, TypeError, ValueError, binascii.Error) as exc:
		raise ValueError(f"bad rgb_b64: {exc}") from None
	if len(raw) != h * w * c:
		raise ValueError(f"expected {h * w * c} bytes for shape {shape!r}, got {len(raw)}")
	return np.frombuffer(raw, dtype=np.uint8).reshape(h, w, c)


def _handle(request: dict[str, Any], cameras: dict[str, dict[str, Any]], device: torch.device) -> dict[str, Any]:
	request_id = request.get("id")
	camera = request.get("camera")
	entry = cameras.get(camera)
	if entry is None:
		return {"id": request_id, "error": f"unknown camera: {camera!r}"}
	try:
		image = _decode_image(request)
	except ValueError as exc:
		return {"id": request_id, "error": str(exc)}
	crop_size = entry["crop_size"]
	if image.shape[:2] != (crop_size, crop_size):
		# Same resize as score.py; the crop is already RGB, so no colour conversion here.
		image = cv2.resize(image, (crop_size, crop_size), interpolation=cv2.INTER_AREA)
	try:
		if entry.get("sup") is not None:
			value = float(entry["sup"].score(camera, image))
		else:
			value = float(score(entry["model"], image[None], entry["input_size"], device, 1)[0])
	except (RuntimeError, ValueError, TypeError, IndexError) as exc:  # a scoring failure must not kill the server
		return {"id": request_id, "error": f"scoring failed: {exc}"}
	threshold = entry["threshold"]
	return {"id": request_id, "score": value, "threshold": threshold, "anomalous": bool(value >= threshold)}


def main() -> int:
	ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	ap.add_argument(
		"--model",
		action="append",
		default=[],
		metavar="CAMERA=MODEL_DIR",
		help="exported model directory per camera",
	)
	ap.add_argument("--sup", type=Path, default=None, help="supervised detector spec (export_sup.py) for its cameras")
	ap.add_argument("--cpu", action="store_true")
	args = ap.parse_args()
	if not args.model and args.sup is None:
		ap.error("give at least one --model or a --sup spec")

	protocol = _protocol_stream()
	device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
	cameras: dict[str, dict[str, Any]] = {}
	for spec in args.model:
		try:
			camera, model_dir = _parse_model_spec(spec)
		except argparse.ArgumentTypeError as exc:
			ap.error(str(exc))
		if camera in cameras:
			ap.error(f"camera {camera!r} is configured twice")
		model, meta = load(model_dir, device)
		cameras[camera] = {
			"model": model,
			"input_size": int(meta["input_size"]),
			"crop_size": int(meta.get("crop_size", 128)),
			"threshold": float(meta["threshold"]),
			"name": meta["model"],
		}
		print(
			f"loaded {camera} <- {model_dir} ({meta['model']}, threshold {float(meta['threshold']):.3f})",
			file=sys.stderr,
			flush=True,
		)
	if args.sup is not None:
		sup = SupScorer(args.sup, device)
		for camera in sup.cameras:
			if camera in cameras:
				continue
			cameras[camera] = {
				"model": None,
				"sup": sup,
				"input_size": sup.input_size,
				"crop_size": sup.crop_size,
				"threshold": 0.0,
				"name": "sup_lora",
			}
			print(f"loaded {camera} <- {args.sup} (supervised, threshold 0)", file=sys.stderr, flush=True)
	_send(
		protocol,
		{
			"ready": True,
			"cameras": {
				camera: {
					"threshold": entry["threshold"],
					"model": entry["name"],
					"crop_size": entry["crop_size"],
				}
				for camera, entry in cameras.items()
			},
		},
	)

	if hasattr(sys.stdin, "reconfigure"):
		sys.stdin.reconfigure(encoding="utf-8")
	try:
		for line in sys.stdin:
			line = line.strip()
			if not line:
				continue
			try:
				request = json.loads(line)
			except json.JSONDecodeError as exc:
				_send(protocol, {"id": None, "error": f"bad json: {exc}"})
				continue
			if not isinstance(request, dict):
				_send(protocol, {"id": None, "error": "request must be a JSON object"})
				continue
			if request.get("cmd") == "quit":
				return 0
			_send(protocol, _handle(request, cameras, device))
	except OSError:  # client went away
		return 0
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
