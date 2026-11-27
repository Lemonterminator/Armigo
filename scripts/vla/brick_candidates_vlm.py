"""Offline test: can the local VLM pick the brick among numbered top-view candidates (set-of-marks)?

``--dump`` (CPU) rebuilds each v3 case (cameras ``drift``, realistic sensing, bowls from the oracle, as the grasp
evals), takes the top depth image ``locate_brick`` would see and lists every raised blob the same way it does
(reach band, bowl exclusion, table plane, height band, 1 mm grid, closing, connected components), keeping each
blob's image pixels. The blobs are drawn as numbered boxes on the top RGB image. The correct number is the blob
whose pixels contain the true brick's top centre projected through the *true* camera ("none" when no blob does).
Rule baselines: ``largest`` (today's ``locate_brick``) and ``shape`` (the largest blob whose top rectangle's long side
is at most 80 mm).

``--ask`` sends each image to the OpenAI-compatible VLM (default: the local vLLM Qwen server, one image per request)
and asks for the number; ``--votes`` asks again with a reworded prompt to measure self-consistency.

	PYTHONPATH=src python scripts/vla/brick_candidates_vlm.py --out C:/scratch/reverse-act/candidates --dump
	PYTHONPATH=src python scripts/vla/brick_candidates_vlm.py --out C:/scratch/reverse-act/candidates --ask
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
	sys.path.insert(0, str(ROOT / "src"))

MIN_CELLS = 150  # blobs smaller than this (mm^2 of top area) are not shown
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


def make_case(seed: int) -> list:
	return [seed, bool(seed % 2), seed % 6, {"scene": "v3", "split": "train", "cameras": "drift", "oracle": "bowls"}]


def blob_candidates(depth, k, world_from_cam, bowls) -> list[dict]:
	"""The raised blobs of ``locate_brick``'s pipeline, largest first, with their image pixels."""
	import cv2

	from digital_twin_armigo.agent import perception as pc

	_, w = depth.shape
	p = pc.depth_points(depth, k, world_from_cam).reshape(-1, 3)
	idx = np.arange(p.shape[0])
	fin = np.isfinite(p).all(axis=1)
	p, idx = p[fin], idx[fin]
	r = np.hypot(p[:, 0], p[:, 1])
	area = (r > pc.REACH_M[0]) & (r < 0.45)
	for bx, by in bowls:
		area &= np.hypot(p[:, 0] - bx, p[:, 1] - by) > pc.BOWL_R + pc.BOWL_MARGIN_M
	plane = pc.table_plane(p[area & (np.abs(p[:, 2]) < pc.TABLE_BAND_M)])
	if plane is None:
		return []
	keep = area & (r < pc.REACH_M[1])
	p, idx = p[keep].copy(), idx[keep]
	p[:, 2] -= p[:, :2] @ plane[:2] + plane[2]
	keep = (p[:, 2] > pc.MIN_HEIGHT_M) & (p[:, 2] < pc.MAX_HEIGHT_M)
	p, idx = p[keep], idx[keep]
	if len(p) < pc.MIN_TOP_POINTS:
		return []
	lo = p[:, :2].min(axis=0) - 0.005
	cells = np.floor((p[:, :2] - lo) / pc.GRID_M).astype(int)
	shape = cells.max(axis=0) + 6
	occ = np.zeros((shape[1], shape[0]), np.uint8)
	occ[cells[:, 1], cells[:, 0]] = 255
	occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
	n, labels, stats, _ = cv2.connectedComponentsWithStats(occ, connectivity=8)
	lab = labels[cells[:, 1], cells[:, 0]]
	out = []
	for label in range(1, n):
		m = lab == label
		if not m.any():
			continue
		blob = p[m]
		z_top = float(np.percentile(blob[:, 2], 90))
		top = blob[blob[:, 2] > z_top - pc.TOP_BAND_M]
		(_, _), (rw, rh), _ = cv2.minAreaRect(top[:, :2].astype(np.float32)) if len(top) >= 3 else ((0, 0), (0, 0), 0)
		v, u = np.divmod(idx[m], w)
		out.append(
			{
				"label": int(label),
				"cells": int(stats[label, cv2.CC_STAT_AREA]),
				"height_mm": round(z_top * 1000, 1),
				"long_mm": round(max(rw, rh) * 1000, 1),
				"short_mm": round(min(rw, rh) * 1000, 1),
				"xy_m": [round(float(v_), 4) for v_ in blob[:, :2].mean(axis=0)],
				"bbox_px": [int(u.min()), int(v.min()), int(u.max()), int(v.max())],
				"pixels": np.stack([u, v], axis=1),
			}
		)
	out.sort(key=lambda c: -c["cells"])
	return out


def draw(rgb: np.ndarray, cands: list[dict]) -> np.ndarray:
	import cv2

	img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
	for i, c in enumerate(cands):
		col = COLOURS[i % len(COLOURS)][::-1]
		x0, y0, x1, y1 = c["bbox_px"]
		cv2.rectangle(img, (x0 - 3, y0 - 3), (x1 + 3, y1 + 3), col, 2)
		txt = str(i + 1)
		org = (x0 - 3, max(14, y0 - 6))
		cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
		cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
	return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def dump_case(args: tuple) -> dict:
	os.environ["CUDA_VISIBLE_DEVICES"] = ""
	seed, out = args
	import shutil

	import cv2

	from digital_twin_armigo.agent.perception import locate_brick
	from digital_twin_armigo.agent.student_rollout import ExpertTracer
	from digital_twin_armigo.agent.twin_mujoco import sample_case, true_world_from_cam
	from digital_twin_armigo.agent.twin_record import EpisodeTrace

	shutil.rmtree(out / "trace" / f"case-{seed}", ignore_errors=True)  # our own scratch trace of an earlier dump
	layout = sample_case(*make_case(seed))
	twin = ExpertTracer(layout, None, EpisodeTrace(out / "trace" / f"case-{seed}", video=False), sensing="realistic")
	try:
		if err := twin.prepare():
			return {"seed": seed, "error": err}
		depth = twin._top_depth()  # the draw _perceive would make next
		rgb = twin.cell.rgb("top").copy()
		bowls = list(twin.bowls.values())
		est = locate_brick(depth, twin._cam_k, twin._cam_est, bowls)
		cands = [c for c in blob_candidates(depth, twin._cam_k, twin._cam_est, bowls) if c["cells"] >= MIN_CELLS]
		cands = cands[:MAX_CANDIDATES]
		# Truth: the candidate with the most pixels inside the brick's top face projected through the true camera.
		cell = twin.cell
		rot = cell.data.xmat[cell.brick_body].reshape(3, 3)
		sx, sy, sz = layout.brick_size
		cam_from_world = np.linalg.inv(true_world_from_cam(layout.top))
		poly = []
		for a, b in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
			c3 = cell.brick_pos() + rot @ np.array([a * sx / 2, b * sy / 2, sz / 2])
			cc = cam_from_world @ np.array([*c3, 1.0])
			poly.append((twin._cam_k @ (cc[:3] / cc[2]))[:2])
		poly = np.array(poly, dtype=np.float32)
		uv = poly.mean(axis=0)
		answer, best_d, best_in = "none", 1e9, 0
		for i, c in enumerate(cands):
			pts = c["pixels"][::2]
			inside = sum(cv2.pointPolygonTest(poly, (float(x), float(y)), True) >= -8.0 for x, y in pts)
			best_d = min(best_d, float(np.min(np.linalg.norm(c["pixels"] - uv, axis=1))))
			c["truth_poly_px"] = round(max(cv2.pointPolygonTest(poly, (float(x), float(y)), True) for x, y in pts), 1)
			if inside > best_in:
				best_in, answer = inside, str(i + 1)
		shape = next((str(i + 1) for i, c in enumerate(cands) if c["long_mm"] <= SHAPE_MAX_LONG_M * 1000), "none")
		img = draw(rgb, cands)
		cv2.imwrite(str(out / "img" / f"{seed}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
		true_xy = cell.brick_pos()[:2]
		return {
			"seed": seed,
			"truth_px": [round(float(uv[0]), 1), round(float(uv[1]), 1)],
			"answer": answer,
			"answer_px_dist": round(best_d, 1),
			"truth_poly": poly.round(1).tolist(),
			"largest": "1" if cands else "none",
			"shape": shape,
			"top_err_mm": None if est is None else round(float(np.linalg.norm(est.pos[:2] - true_xy) * 1000), 2),
			"candidates": [{k: v for k, v in c.items() if k != "pixels"} for c in cands],
		}
	except Exception as e:  # noqa: BLE001
		return {"seed": seed, "error": f"crash:{type(e).__name__}:{e}"}
	finally:
		twin.cell.close()


def _png_b64(path: Path) -> str:
	return base64.b64encode(path.read_bytes()).decode("ascii")


def ask(out: Path, rows: list[dict], url: str, model: str, votes: int) -> list[dict]:
	from digital_twin_armigo.agent.llm import OpenAICompatLLM

	llm = OpenAICompatLLM(url, model, max_tokens=32, timeout_s=120, retries=1)
	for r in rows:
		if "error" in r:
			continue
		ids = [str(i + 1) for i in range(len(r["candidates"]))] + ["none"]
		schema = {
			"type": "object",
			"properties": {"candidate": {"type": "string", "enum": ids}},
			"required": ["candidate"],
			"additionalProperties": False,
		}
		img = _png_b64(out / "img" / f"{r['seed']}.png")
		picks, lat = [], 0.0
		for v in range(votes):
			text = PROMPT if v == 0 else PROMPT + " Look carefully at the shapes and colours inside each box."
			msg = [
				{
					"role": "user",
					"content": [
						{"type": "image_url", "image_url": {"url": "data:image/png;base64," + img}},
						{"type": "text", "text": text + ' Reply as JSON {"candidate": "<number or none>"}.'},
					],
				}
			]
			try:
				res = llm.chat(msg, json_schema=schema)
				lat += res.latency_s
				picks.append(str(json.loads(res.content).get("candidate")))
			except Exception as e:  # noqa: BLE001
				picks.append(f"error:{type(e).__name__}")
		r["vlm"] = picks[0]
		r["vlm_votes"] = picks
		r["vlm_latency_s"] = round(lat / max(votes, 1), 2)
		print(r["seed"], "truth", r["answer"], "vlm", picks, "largest", r["largest"], "shape", r["shape"], flush=True)
	return rows


def summarize(rows: list[dict]) -> dict:
	ok = [r for r in rows if "error" not in r]
	s: dict = {"n": len(rows), "errors": [r["error"] for r in rows if "error" in r]}
	s["truth_none"] = sum(r["answer"] == "none" for r in ok)
	for key in ("largest", "shape", "vlm"):
		have = [r for r in ok if key in r]
		if not have:
			continue
		s[key] = {
			"correct": sum(r[key] == r["answer"] for r in have),
			"n": len(have),
			"correct_when_largest_wrong": sum(r[key] == r["answer"] for r in have if r["largest"] != r["answer"]),
			"largest_wrong": sum(r["largest"] != r["answer"] for r in have),
		}
	have = [r for r in ok if "vlm" in r]
	if have:
		agree = [r for r in have if r["vlm"] == r["shape"]]
		s["vlm_shape_agree"] = {"n": len(agree), "correct": sum(r["vlm"] == r["answer"] for r in agree)}
		s["vlm_shape_disagree"] = {
			"n": len(have) - len(agree),
			"vlm_correct": sum(r["vlm"] == r["answer"] for r in have if r["vlm"] != r["shape"]),
			"shape_correct": sum(r["shape"] == r["answer"] for r in have if r["vlm"] != r["shape"]),
		}
		s["vlm_self_consistent"] = sum(len(set(r["vlm_votes"])) == 1 for r in have)
		s["vlm_latency_s_median"] = float(np.median([r["vlm_latency_s"] for r in have]))
	return s


def main(argv: list[str] | None = None) -> int:
	ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	ap.add_argument("--out", type=Path, required=True)
	ap.add_argument("--seed0", type=int, default=3001201)
	ap.add_argument("--n", type=int, default=60)
	ap.add_argument("--workers", type=int, default=8)
	ap.add_argument("--dump", action="store_true")
	ap.add_argument("--ask", action="store_true")
	ap.add_argument("--url", default="http://127.0.0.1:8038/v1")
	ap.add_argument("--model", default="qwen3.8-vllm")
	ap.add_argument("--votes", type=int, default=3)
	a = ap.parse_args(argv)
	res = a.out / "results.jsonl"
	if a.dump:
		(a.out / "img").mkdir(parents=True, exist_ok=True)
		with ProcessPoolExecutor(a.workers) as pool:
			rows = list(pool.map(dump_case, [(s, a.out) for s in range(a.seed0, a.seed0 + a.n)]))
	else:
		rows = [json.loads(line) for line in res.read_text(encoding="utf-8").splitlines()]
	if a.ask:
		rows = ask(a.out, rows, a.url, a.model, a.votes)
	res.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
	summary = summarize(rows)
	(a.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
	print(json.dumps(summary, indent=2))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
