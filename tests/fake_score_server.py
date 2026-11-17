"""Stand-in for ``scripts/anomaly/score_server.py`` with the same line protocol, without torch or anomalib.

Run by the main venv Python in tests. ``--model camera=THRESHOLD`` takes a float threshold instead of a model
directory, and the score is the image's mean / 255 * 100.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import sys
from typing import Any

import numpy as np


def _send(obj: dict[str, Any]) -> None:
	sys.stdout.write(json.dumps(obj) + "\n")
	sys.stdout.flush()


def _handle(request: dict[str, Any], thresholds: dict[str, float]) -> dict[str, Any]:
	request_id = request.get("id")
	camera = request.get("camera")
	if camera not in thresholds:
		return {"id": request_id, "error": f"unknown camera: {camera!r}"}
	shape = request.get("shape")
	if not (isinstance(shape, list) and len(shape) == 3 and all(isinstance(v, int) and v > 0 for v in shape)):
		return {"id": request_id, "error": f"bad shape: {shape!r}"}
	h, w, c = shape
	if c != 3:
		return {"id": request_id, "error": f"bad shape: {shape!r}"}
	try:
		raw = base64.b64decode(request["rgb_b64"], validate=True)
	except (KeyError, TypeError, ValueError, binascii.Error) as exc:
		return {"id": request_id, "error": f"bad rgb_b64: {exc}"}
	if len(raw) != h * w * c:
		return {"id": request_id, "error": f"expected {h * w * c} bytes for shape {shape!r}, got {len(raw)}"}
	value = float(np.frombuffer(raw, dtype=np.uint8).mean() / 255.0 * 100.0)
	threshold = thresholds[camera]
	return {"id": request_id, "score": value, "threshold": threshold, "anomalous": bool(value > threshold)}


def main() -> int:
	ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	ap.add_argument("--model", action="append", required=True, metavar="CAMERA=THRESHOLD")
	args = ap.parse_args()
	thresholds: dict[str, float] = {}
	for spec in args.model:
		camera, sep, value = spec.partition("=")
		if not sep or not camera or not value:
			ap.error(f"expected CAMERA=THRESHOLD, got {spec!r}")
		thresholds[camera] = float(value)
	if hasattr(sys.stdin, "reconfigure"):
		sys.stdin.reconfigure(encoding="utf-8")
	_send(
		{
			"ready": True,
			"cameras": {
				camera: {"threshold": threshold, "model": "fake", "crop_size": 0}
				for camera, threshold in thresholds.items()
			},
		}
	)
	for line in sys.stdin:
		line = line.strip()
		if not line:
			continue
		try:
			request = json.loads(line)
		except json.JSONDecodeError as exc:
			_send({"id": None, "error": f"bad json: {exc}"})
			continue
		if not isinstance(request, dict):
			_send({"id": None, "error": "request must be a JSON object"})
			continue
		if request.get("cmd") == "quit":
			return 0
		_send(_handle(request, thresholds))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
