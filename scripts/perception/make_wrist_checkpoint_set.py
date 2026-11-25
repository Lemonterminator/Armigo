"""Wrist checkpoint-stratum set (``wrist-checkpoint-v1``, brickseg_prereg section 4): the wrist frames a closed-loop
grasp actually sees at its re-localisation checkpoints, with the belief the agent actually holds there.

For every case (scene v3, split ``val`` or ``ood-appearance``, cameras ``drift``, realistic sensing) the twin perceives
the brick from the top camera and descends to grasp with a synchronous checkpoint at every height used by
``grasp_loop.VARIANTS`` (hover, 0.050, 0.045, 0.040, 0.035, 0.030 m). The look at each checkpoint records the frame and
returns no replan, so the descent follows the open-loop plan and each frame's prior is the top-camera estimate.

``--stress`` adds one 2x2 clutter brick of the brick's own colour whose centre is 3-6 cm from the brick's centre (the
search gate is 4 cm), footprints at least 3 mm apart: the same-colour confuser stress set.

Every frame carries a per-confuser label (clutter bricks and distractor boxes): where it projects (``out_of_view``,
``in_view_outside_gate``, ``in_gate``, ``touching``) and the rendered Lab colour difference to the brick (CIE76 ΔE,
on the same scale as ``wrist_localize._blob``'s ``min_delta=35``).

Output: ``<out>/shard_NNNN.npz`` (``rgb`` (N, H, W, 3) uint8, ``meta`` (N,) JSON) and ``<out>/manifest.json``. CPU
only; no file outside ``--out`` is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

HEIGHTS = ("hover", 0.050, 0.045, 0.040, 0.035, 0.030)
# Disjoint from U1 brickseg-v1 (val 3100001-3100250, ood 3500001-3500200), p2_dev12 and loop_01_96 (3600001-3600048).
SEED_START = {"val": 3_100_501, "ood-appearance": 3_500_501}
STRESS_OFFSET = 10_000  # stress cases use their own seeds
CLUTTER_SIZE = (0.0318, 0.0318, 0.0192)
TOUCH_GAP_M = 0.003


def _footprint(xy, yaw, size):
	c, s = math.cos(yaw), math.sin(yaw)
	hx, hy = size[0] / 2, size[1] / 2
	return np.array(
		[[xy[0] + c * a - s * b, xy[1] + s * a + c * b] for a, b in ((hx, hy), (-hx, hy), (-hx, -hy), (hx, -hy))]
	)


def _seg_dist(p, a, b):
	ab = b - a
	t = np.clip(((p - a) @ ab) / max(ab @ ab, 1e-12), 0.0, 1.0)
	return float(np.linalg.norm(p - (a + t * ab)))


def _inside(p, poly):
	sign = None
	for i in range(len(poly)):
		a, b = poly[i], poly[(i + 1) % len(poly)]
		cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
		if sign is None:
			sign = cross > 0
		elif (cross > 0) != sign:
			return False
	return True


def footprint_gap(p1, p2) -> float:
	"""Smallest distance (m) between two convex footprints, 0 when they overlap."""
	if any(_inside(p, p2) for p in p1) or any(_inside(p, p1) for p in p2):
		return 0.0
	d = [_seg_dist(p, p2[i], p2[(i + 1) % 4]) for p in p1 for i in range(4)]
	d += [_seg_dist(p, p1[i], p1[(i + 1) % 4]) for p in p2 for i in range(4)]
	return min(d)


def stress_clutter(layout, rng):
	"""One 2x2 brick of the brick's colour, centre 3-6 cm from the brick centre, footprints at least 3 mm apart."""
	bxy, byaw = np.asarray(layout.brick_xy), math.radians(layout.brick_yaw_deg)
	brick = _footprint(bxy, byaw, layout.brick_size)
	for _ in range(500):
		a, d = rng.uniform(0, 2 * math.pi), rng.uniform(0.030, 0.060)
		xy = bxy + d * np.array([math.cos(a), math.sin(a)])
		yaw = rng.uniform(0, math.pi / 2)
		gap = footprint_gap(brick, _footprint(xy, yaw, CLUTTER_SIZE))
		if gap >= TOUCH_GAP_M:
			return (float(xy[0]), float(xy[1]), float(yaw), *map(float, layout.brick_rgb)), gap
	raise RuntimeError("no stress clutter placement")


def _lab(rgb_patch):
	import cv2

	lab = cv2.cvtColor(rgb_patch.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float64)
	lab[:, 0] *= 100.0 / 255.0
	lab[:, 1:] -= 128.0
	return np.median(lab, axis=0)


def _patch(rgb, uv, r=2):
	h, w = rgb.shape[:2]
	u, v = int(np.rint(uv[0])), int(np.rint(uv[1]))
	if not (r <= u < w - r and r <= v < h - r):
		return None
	return rgb[v - r : v + r + 1, u - r : u + r + 1]


def confusers(layout, rgb, k, w2c_actual, brick_xy, brick_yaw, prior_xy, top_z, gate_m=0.04):
	"""Per-object confuser labels for one frame, projected with the actual camera."""
	from digital_twin_armigo.agent.wrist_localize import _project

	centre, depth = _project(k, w2c_actual, np.array([prior_xy[0], prior_xy[1], top_z]))
	gate_px = gate_m * k[0, 0] / max(depth, 1e-6)
	bp = _patch(rgb, _project(k, w2c_actual, np.array([brick_xy[0], brick_xy[1], top_z]))[0])
	brick_lab = None if bp is None else _lab(bp)
	brick_fp = _footprint(brick_xy, brick_yaw, layout.brick_size)
	objs = [("clutter", c[0], c[1], c[2], CLUTTER_SIZE, c[3:6]) for c in layout.clutter]
	objs += [
		("distractor", d[0], d[1], d[5] if len(d) > 5 else 0.0, (2 * d[3], 2 * d[4], 2 * d[2]), None)
		for d in layout.distractors
	]
	h, w = rgb.shape[:2]
	out = []
	for kind, x, y, yaw, size, colour in objs:
		top = size[2]
		uv, z = _project(k, w2c_actual, np.array([x, y, top]))
		gap = footprint_gap(brick_fp, _footprint((x, y), yaw, size))
		if z <= 0 or not (0 <= uv[0] < w and 0 <= uv[1] < h):
			bucket = "out_of_view"
		elif gap <= TOUCH_GAP_M:
			bucket = "touching"
		elif np.linalg.norm(uv - centre) <= gate_px:
			bucket = "in_gate"
		else:
			bucket = "in_view_outside_gate"
		de = None
		if bucket != "out_of_view" and brick_lab is not None and (p := _patch(rgb, uv)) is not None:
			de = round(float(np.linalg.norm(_lab(p) - brick_lab)), 2)
		out.append(
			{
				"kind": kind,
				"xy": [round(float(x), 5), round(float(y), 5)],
				"rgb": None if colour is None else [round(float(v), 4) for v in colour],
				"same_rgb": colour is not None and np.allclose(colour, layout.brick_rgb),
				"bucket": bucket,
				"gap_mm": round(gap * 1000, 2),
				"delta_e": de,
			}
		)
	return out


def run_case(job):
	"""Frames of one case: list of (rgb, meta) or an error string."""
	seed, split, stress = job
	import mujoco

	from digital_twin_armigo.agent.grasp_loop import GraspLoop
	from digital_twin_armigo.agent.student_rollout import ExpertTracer
	from digital_twin_armigo.agent.twin_mujoco import sample_case
	from digital_twin_armigo.agent.twin_record import EpisodeTrace
	from digital_twin_armigo.agent.wrist_localize import believed_wrist_from_cam, wrist_k
	from digital_twin_armigo.inspection_scorer import FixedScorer

	class _Scorer(FixedScorer):
		def close(self) -> None:
			pass

	rng = np.random.default_rng([seed, 0xC4EC])
	positive = bool(rng.integers(2))
	face = int(rng.integers(6))
	try:
		layout = sample_case(seed, positive, face, {"scene": "v3", "split": split, "cameras": "drift"})
	except RuntimeError as exc:
		return {"seed": seed, "error": f"sample_case: {exc}"}
	stress_gap = None
	if stress:
		try:
			extra, stress_gap = stress_clutter(layout, rng)
		except RuntimeError as exc:
			return {"seed": seed, "error": str(exc)}
		layout.clutter = [*layout.clutter, extra]
	frames: list = []
	with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
		trace = EpisodeTrace(Path(tmp) / "ep", video=False)
		twin = ExpertTracer(
			layout,
			_Scorer(lambda _cam, _img: -1.0, {"top": 0.0, "wrist_cam": 0.0}),
			trace,
			sensing="realistic",
			motion="w1s",
		)
		try:
			cell = twin.cell

			def look(h):
				if len(frames) >= len(HEIGHTS):  # keep the first descent; a regrasp after a failed grasp repeats it
					return
				ex = twin.expert
				rgb = cell.rgb("wrist_cam").copy()
				ex.kin.data.qpos[:] = cell.data.qpos
				w2c = believed_wrist_from_cam(cell, ex.kin.data)
				cid = mujoco.mj_name2id(cell.model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
				k = wrist_k(float(cell.model.cam_fovy[cid]), rgb.shape)
				actual = np.eye(4)
				actual[:3, :3] = cell.data.cam_xmat[cid].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
				actual[:3, 3] = cell.data.cam_xpos[cid]
				size = tuple(float(v) for v in twin._believed_brick_size())
				est = twin.brick_est
				top_z = float(est.pos[2] + size[2] / 2)
				rot = cell.data.xmat[cell.brick_body].reshape(3, 3).copy()
				bpos = cell.brick_pos().copy()
				true_top = float(bpos[2] + layout.brick_size[2] / 2)
				byaw = math.atan2(rot[1, 0], rot[0, 0])
				meta = {
					"seed": seed,
					"split": split,
					"stress": stress,
					"stress_gap_mm": None if stress_gap is None else round(stress_gap * 1000, 2),
					"positive": positive,
					"face": face,
					"height_m": round(float(h), 4),
					"height_index": len(frames),
					"k": k.tolist(),
					"world_from_cam": w2c.tolist(),
					"world_from_cam_actual": actual.tolist(),
					"prior_xy": [float(v) for v in est.pos[:2]],
					"prior_yaw": float(est.yaw),
					"believed_size": list(size),
					"top_z": top_z,
					"brick_pos": bpos.tolist(),
					"brick_rot": rot.tolist(),
					"brick_size": [float(v) for v in layout.brick_size],
					"brick_rgb": [float(v) for v in layout.brick_rgb],
					"true_top_z": true_top,
					"confusers": confusers(layout, rgb, k, actual, bpos[:2], byaw, est.pos[:2], true_top),
				}
				frames.append((rgb, meta))
				return

			error = twin.prepare()
			if error is None:
				twin.expert.recheck = GraspLoop(HEIGHTS, "sync", look)
				error = twin._grasp()
		finally:
			twin.close()
			trace.close()
	return {"seed": seed, "frames": frames, "grasp": error, "positive": positive, "face": face}


def _git_commit() -> str | None:
	try:
		return subprocess.run(
			["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
		).stdout.strip()
	except (OSError, subprocess.CalledProcessError):
		return None


def main(argv=None) -> int:
	p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
	p.add_argument("--out", type=Path, required=True)
	p.add_argument("--splits", nargs="+", default=["val", "ood-appearance"], choices=sorted(SEED_START))
	p.add_argument("--cases", type=int, default=100, help="cases per split")
	p.add_argument("--stress", action="store_true", help="add a same-colour 2x2 confuser near the brick")
	p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
	p.add_argument("--cases-per-shard", type=int, default=25)
	a = p.parse_args(argv)
	os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
	a.out.mkdir(parents=True, exist_ok=True)
	if any(a.out.glob("shard_*.npz")):
		raise SystemExit(f"{a.out} already holds shards; refusing to overwrite")
	jobs = [
		(SEED_START[s] + (STRESS_OFFSET if a.stress else 0) + i, s, bool(a.stress))
		for s in a.splits
		for i in range(a.cases)
	]
	shards, pending, cases, shard_i = [], [], [], 0

	def flush():
		nonlocal pending, shard_i
		if not pending:
			return
		path = a.out / f"shard_{shard_i:04d}.npz"
		np.savez_compressed(
			path,
			rgb=np.stack([f[0] for f in pending]),
			meta=np.array([json.dumps(f[1], sort_keys=True) for f in pending]),
		)
		shards.append(
			{"path": path.name, "frames": len(pending), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
		)
		pending, shard_i = [], shard_i + 1

	with ProcessPoolExecutor(a.workers) as pool:
		for n, res in enumerate(pool.map(run_case, jobs), 1):
			cases.append({k: v for k, v in res.items() if k != "frames"} | {"frames": len(res.get("frames", []))})
			pending += res.get("frames", [])
			if n % a.cases_per_shard == 0:
				flush()
			print(
				f"[{n}/{len(jobs)}] seed {res['seed']} frames {len(res.get('frames', []))} {res.get('error') or res.get('grasp') or ''}",
				flush=True,
			)
	flush()
	manifest = {
		"name": a.out.name,
		"code_commit": _git_commit(),
		"splits": a.splits,
		"cases_per_split": a.cases,
		"stress": bool(a.stress),
		"heights": list(HEIGHTS),
		"seed_start": SEED_START,
		"stress_seed_offset": STRESS_OFFSET if a.stress else 0,
		"cameras": "drift",
		"sensing": "realistic",
		"shards": shards,
		"cases": cases,
		"total_frames": sum(s["frames"] for s in shards),
	}
	(a.out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
	print(f"wrote {manifest['total_frames']} frames from {len(jobs)} cases to {a.out}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
