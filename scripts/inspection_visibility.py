"""How much of the held brick do the expert's show poses actually show? Measures defect / face visibility per camera.

Runs the closed-loop expert in inspection mode on random layouts whose defect face is drawn uniformly from all six
faces (``sample_layout`` alone always leaves it on the top face). After every show hold it records, for each
inspection camera, the visible pixels of each brick face and of the defect cross (exact, from segmentation renders
of hidden face probes), and asks a :class:`PixelOracleInspector` for a label; the expert places into the bowl the
label picks, or aborts with ``inspection_unknown``. Every episode is scored by the unchanged audited evaluator
against the ground-truth bowl.

At the first show hold it also sweeps a grid of wrist flex x wrist roll poses on a kinematic copy (brick carried
rigidly by the gripper) and records per-face pixels for each pose that keeps the arm and the brick clear of the
table: which poses could show which face. Writes ``episodes.jsonl``, ``pose_grid.jsonl`` and ``summary.json``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import bench_physics_expert as bpe

from digital_twin_armigo.benchmarks import SENSITIVITY_MATRIX
from digital_twin_armigo.constants import SO101_MJCF_DIR, TOP_CROP
from digital_twin_armigo.physics import load_physics_spec
from digital_twin_armigo.presentation import FACES

CAMERAS = {"top": TOP_CROP, "wrist_cam": None}
FACE_NAMES = ("+x", "-x", "+y", "-y", "top", "bottom")
GRID_FLEX = tuple(np.round(np.linspace(-1.2, 1.6, 8), 3))
GRID_ROLL_DEG = tuple(range(-150, 151, 30))
TABLE_CLEARANCE_M = 0.02  # a grid pose counts only if the arm and the brick stay this far above the table


def _layouts(n: int, seed: int, top_cam) -> list:
	rng = np.random.default_rng(seed + 10_000)
	lays = bpe.sample_layouts(n, seed, top_cam)
	for lay in lays:
		lay.defect_face = int(rng.integers(len(FACES)))
	return lays


def _pose_grid(cell, q_now: np.ndarray) -> list[dict]:
	from digital_twin_armigo.mujoco_cell.inspection_views import HeldBrickPoser, capture_view

	poser = HeldBrickPoser(cell)
	lo, hi = cell.jnt_range[3], cell.jnt_range[4]
	rows = []
	for flex in GRID_FLEX:
		for roll in GRID_ROLL_DEG:
			if not (lo[0] <= flex <= lo[1] and hi[0] <= math.radians(roll) <= hi[1]):
				continue
			q = q_now.copy()
			q[3], q[4] = flex, math.radians(roll)
			d = poser.at(q)
			clear = min(cell.robot_min_z(d), poser.brick_min_z())
			row = {"flex": float(flex), "roll_deg": int(roll), "clearance_m": round(clear, 4)}
			if clear >= TABLE_CLEARANCE_M:
				for cam, crop in CAMERAS.items():
					v = capture_view(cell, cam, crop=crop, data=d)
					row[cam] = {"faces_px": list(v.faces_px), "brick_px": v.brick_px}
			rows.append(row)
	return rows


def run_one(lay, spec, k: int, params, grid: bool) -> tuple[dict[str, Any], list[dict]]:
	from digital_twin_armigo.mujoco_cell import Cell
	from digital_twin_armigo.mujoco_cell.inspection_views import PixelOracleInspector, capture_view
	from digital_twin_armigo.mujoco_cell.physics_expert import run_closed_loop_episode

	base = {c.name: c for c in SENSITIVITY_MATRIX}["baseline_2x2"]
	cell = Cell(lay, SO101_MJCF_DIR, physics=spec, dynamic_clutter=True, assisted_grasp=False, face_probes=True)
	oracle = PixelOracleInspector(lay.defective)
	per_cam = {cam: PixelOracleInspector(lay.defective) for cam in CAMERAS}
	views: list[dict] = []
	grid_rows: list[dict] = []

	def inspect(flex: float, roll_deg: float, frame: int):
		if grid and not grid_rows:
			grid_rows.extend(_pose_grid(cell, cell.joints()))
		res = None
		rec = {"frame": frame, "flex": flex, "roll_deg": roll_deg}
		for cam, crop in CAMERAS.items():
			v = capture_view(cell, cam, crop=crop, frame=frame)
			res = oracle.observe(v)
			per_cam[cam].observe(v)
			rec[cam] = {"faces_px": list(v.faces_px), "defect_px": v.defect_px, "brick_px": v.brick_px}
		views.append(rec)
		return res

	bowls = {"normal": lay.green, "defective": lay.pink}
	gt_bowl = lay.pink if lay.defective else lay.green
	try:
		tel, phases, info = run_closed_loop_episode(
			cell,
			params,
			q_start=bpe._q_start(),
			brick_xy=lay.brick_xy,
			brick_yaw_deg=lay.brick_yaw_deg,
			non_ideal=bpe.bench._mj_non_ideal(base, spec),
			bowls=bowls,
			inspect=inspect,
		)
	finally:
		cell.close()
	if info["infeasible_reason"] is not None:
		verdict, reasons = "INFEASIBLE", [info["infeasible_reason"]]
	else:
		health = bpe.bench._evaluate_mj_rollout(tel, phases, spec, tuple(lay.brick_size), gt_bowl)
		verdict, reasons = health["verdict"], list(health["reason_codes"])
	ep = {
		"layout_index": k,
		"defective": bool(lay.defective),
		"defect_face": int(lay.defect_face),
		"brick_size": [float(v) for v in lay.brick_size],
		"verdict": verdict,
		"reason_codes": reasons,
		"aborted": info["aborted"],
		"label": info.get("inspection_label"),
		"label_correct": info.get("inspection_label") == ("defective" if lay.defective else "normal"),
		"n_frames": info["n_frames"],
		"grasp_roll": info["grasp_geometry"].get("roll"),
		"views": views,
		"observed_faces": {cam: sorted(insp.observed) for cam, insp in per_cam.items()},
		"defect_seen": {cam: insp.defect_seen for cam, insp in per_cam.items()},
	}
	for r in grid_rows:
		r["layout_index"] = k
		r["defect_face"] = int(lay.defect_face)
	return ep, grid_rows


def summarize(eps: list[dict], grid: list[dict]) -> dict:
	out: dict[str, Any] = {"n_episodes": len(eps)}
	shown = [e for e in eps if e["views"]]
	out["n_reached_show"] = len(shown)
	out["verdicts"] = dict(Counter(e["verdict"] for e in eps))
	out["aborted"] = dict(Counter(str(e["aborted"]) for e in eps))
	out["labels"] = {
		cls: dict(Counter(str(e["label"]) for e in shown if e["defective"] == (cls == "defective")))
		for cls in ("defective", "normal")
	}
	out["wrong_labels"] = sum(1 for e in shown if e["label"] is not None and not e["label_correct"])
	# face coverage over the base show poses, per camera and for either camera
	cov: dict[str, list[float]] = {}
	for cam in [*CAMERAS, "either"]:
		frac = []
		for f in range(len(FACES)):
			n = 0
			for e in shown:
				obs = (
					set().union(*(e["observed_faces"][c] for c in CAMERAS))
					if cam == "either"
					else set(e["observed_faces"][cam])
				)
				n += f in obs
			frac.append(round(n / max(len(shown), 1), 3))
		cov[cam] = dict(zip(FACE_NAMES, frac, strict=True))
	out["face_coverage_base_poses"] = cov
	# defect visible rate by defect face (defective bricks that reached the show poses)
	by_face: dict[str, list[int]] = defaultdict(lambda: [0, 0])
	for e in shown:
		if e["defective"]:
			seen = any(e["defect_seen"].values())
			by_face[FACE_NAMES[e["defect_face"]]][0] += seen
			by_face[FACE_NAMES[e["defect_face"]]][1] += 1
	out["defect_visible_by_face"] = {k: f"{a}/{b}" for k, (a, b) in sorted(by_face.items())}
	# best achievable per face over the safe grid poses (fraction of episodes in which some safe pose shows it)
	if grid:
		per_ep: dict[int, set[int]] = defaultdict(set)
		pose_hits: Counter = Counter()
		safe = Counter()
		for r in grid:
			if r["clearance_m"] < TABLE_CLEARANCE_M:
				continue
			safe[(r["flex"], r["roll_deg"])] += 1
			for cam in CAMERAS:
				for f, n in enumerate(r[cam]["faces_px"]):
					if n >= 60:
						per_ep[r["layout_index"]].add(f)
						pose_hits[(r["flex"], r["roll_deg"], cam, FACE_NAMES[f])] += 1
		n_ep = max(len({r["layout_index"] for r in grid}), 1)
		out["grid_face_reachable"] = {
			FACE_NAMES[f]: round(sum(f in s for s in per_ep.values()) / n_ep, 3) for f in range(len(FACES))
		}
		best: dict[str, list] = {}
		for f in FACE_NAMES:
			hits = sorted(((n, k) for k, n in pose_hits.items() if k[3] == f), reverse=True)[:3]
			best[f] = [{"flex": k[0], "roll_deg": k[1], "camera": k[2], "episodes": n} for n, k in hits]
		out["grid_best_poses_per_face"] = best
		out["grid_safe_pose_fraction"] = round(len(safe) / max(len(GRID_FLEX) * len(GRID_ROLL_DEG), 1), 3)
	return out


def main() -> None:
	from digital_twin_armigo.mujoco_cell.physics_expert import ExpertParams

	ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	ap.add_argument("--n", type=int, default=40)
	ap.add_argument("--seed", type=int, default=2026)
	ap.add_argument("--no-grid", action="store_true")
	ap.add_argument("--stop-when-known", action="store_true", help="stop showing once the label is KNOWN")
	ap.add_argument(
		"--extra-poses",
		default="",
		help="extra show poses tried while UNKNOWN, 'flex:roll_deg,flex:roll_deg,...' (rad, degrees)",
	)
	ap.add_argument(
		"--max-frames",
		type=int,
		default=None,
		help="episode frame cap (default ExpertParams.max_frames); extra show poses need more",
	)
	ap.add_argument("--out", type=Path, default=ROOT / "results" / "inspection-visibility")
	args = ap.parse_args()
	extra = tuple((float(a), float(b)) for a, b in (p.split(":") for p in args.extra_poses.split(",") if p.strip()))
	params = ExpertParams(stop_when_known=args.stop_when_known, extra_show_poses=extra)
	if args.max_frames is not None:
		params = dataclasses.replace(params, max_frames=args.max_frames)
	spec = load_physics_spec()
	top = bpe.load_top_camera()
	args.out.mkdir(parents=True, exist_ok=True)
	eps, grid = [], []
	t0 = time.perf_counter()
	for k, lay in enumerate(_layouts(args.n, args.seed, top)):
		ep, rows = run_one(lay, spec, k, params, grid=not args.no_grid)
		eps.append(ep)
		grid.extend(rows)
		print(
			f"[{k:3d}] def={int(ep['defective'])} face={FACE_NAMES[ep['defect_face']]:6s} {ep['verdict']:10s} "
			f"label={ep['label']} aborted={ep['aborted']} obs={ep['observed_faces']} "
			f"({time.perf_counter() - t0:.0f}s)",
			flush=True,
		)
	(args.out / "episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in eps), encoding="utf-8")
	(args.out / "pose_grid.jsonl").write_text("".join(json.dumps(r) + "\n" for r in grid), encoding="utf-8")
	summary = summarize(eps, grid)
	summary["config"] = {
		"n": args.n,
		"seed": args.seed,
		"stop_when_known": args.stop_when_known,
		"extra_show_poses": [list(p) for p in extra],
		"cameras": list(CAMERAS),
		"min_face_px": 60,
		"min_defect_px": 25,
		"max_frames": params.max_frames,
	}
	(args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
	print(json.dumps(summary, indent=2))


if __name__ == "__main__":
	main()
