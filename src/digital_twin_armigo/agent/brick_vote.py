"""Which top-view blob is the brick: a shape rule and a VLM vote on numbered candidates (set-of-marks).

:func:`shape_pick` takes the largest blob whose top rectangle is no longer than :data:`SHAPE_MAX_LONG_M` (a 2x4 Duplo
is 63.6 mm; bowl-rim crescents that survive a misplaced bowl mask are longer than 85 mm). :func:`vlm_pick` draws the
candidates as numbered boxes on the top RGB image and asks a vision-language model which one is the brick.

Offline on 60 v3 drift cases (``scripts/vla/brick_candidates_vlm.py``, Qwen3.8 on the local vLLM server): shape 58/60
correct, VLM 54/60; where both agree (52) both were always right; of the 8 disagreements the VLM was right in 2 and
the shape rule in 6. The VLM's mistakes are self-consistent over repeated asks, so agreement with the independent
rule, not repetition, is the check; a disagreement goes to a wrist-camera confirmation.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Sequence

import numpy as np

from .perception import BrickBlob

MIN_CELLS = 150  # blobs smaller than this (mm^2 of top area) are not candidates
MAX_CANDIDATES = 8
SHAPE_MAX_LONG_M = 0.08
COLOURS = [(255, 40, 40), (40, 160, 255), (255, 200, 0), (200, 60, 255), (0, 220, 120), (255, 120, 0)]
COLOURS += [(255, 255, 255), (0, 0, 0)]

PROMPT = (
	"This is a top-down camera image of a robot work table. Each numbered box marks one raised object found in the "
	"depth image. The target is the one LEGO Duplo brick (a small rectangular toy block with round studs on top) "
	"lying on the table outside the bowls; bricks inside a bowl are already sorted and are never the target. The "
	"target may be partly hidden under the robot gripper. Which numbered box contains the target brick? A box that "
	"only covers part of a bowl rim, the arm or another object is not it. Answer 'none' if no box contains it."
)


def candidates(blobs: Sequence[BrickBlob]) -> list[BrickBlob]:
	"""The blobs shown to the vote: at least ``MIN_CELLS``, the ``MAX_CANDIDATES`` largest."""
	return [b for b in blobs if b.cells >= MIN_CELLS][:MAX_CANDIDATES]


def shape_pick(cands: Sequence[BrickBlob]) -> int | None:
	"""Index of the largest candidate no longer than ``SHAPE_MAX_LONG_M``, or None."""
	return next((i for i, b in enumerate(cands) if b.long_m <= SHAPE_MAX_LONG_M), None)


def draw(rgb: np.ndarray, cands: Sequence[BrickBlob]) -> np.ndarray:
	"""``rgb`` with each candidate's pixel box drawn and numbered from 1."""
	import cv2

	img = np.ascontiguousarray(rgb).copy()
	for i, b in enumerate(cands):
		col = COLOURS[i % len(COLOURS)]
		u, v = b.pixels[:, 0], b.pixels[:, 1]
		x0, y0, x1, y1 = int(u.min()), int(v.min()), int(u.max()), int(v.max())
		cv2.rectangle(img, (x0 - 3, y0 - 3), (x1 + 3, y1 + 3), col, 2)
		org = (x0 - 3, max(14, y0 - 6))
		cv2.putText(img, str(i + 1), org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
		cv2.putText(img, str(i + 1), org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
	return img


def png_b64(rgb: np.ndarray) -> str:
	import cv2

	ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
	if not ok:
		raise ValueError("PNG encoding failed")
	return base64.b64encode(buf.tobytes()).decode("ascii")


def vlm_pick(llm, rgb: np.ndarray, cands: Sequence[BrickBlob]) -> tuple[int | None, dict]:
	"""Ask ``llm`` (an :class:`~.llm.OpenAICompatLLM`) which numbered candidate is the brick. Returns (index or None,
	info); info["error"] is set when the request failed (the caller then has no VLM vote)."""
	ids = [str(i + 1) for i in range(len(cands))] + ["none"]
	schema = {
		"type": "object",
		"properties": {"candidate": {"type": "string", "enum": ids}},
		"required": ["candidate"],
		"additionalProperties": False,
	}
	msg = [
		{
			"role": "user",
			"content": [
				{"type": "image_url", "image_url": {"url": "data:image/png;base64," + png_b64(draw(rgb, cands))}},
				{"type": "text", "text": PROMPT + ' Reply as JSON {"candidate": "<number or none>"}.'},
			],
		}
	]
	try:
		res = llm.chat(msg, json_schema=schema)
		answer = str(json.loads(res.content).get("candidate"))
	except Exception as e:  # noqa: BLE001 - a failed request is no vote, never a crash
		return None, {"error": f"{type(e).__name__}: {e}"[:200]}
	info = {"answer": answer, "latency_s": round(float(res.latency_s), 3)}
	return (int(answer) - 1 if answer.isdigit() and 1 <= int(answer) <= len(cands) else None), info
