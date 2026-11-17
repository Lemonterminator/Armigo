"""How large can the held brick get in the top camera? Sweep whole-arm poses with the brick carried kinematically.

For ``--n`` benchmark layouts the closed-loop expert runs (inspection mode) to its first show hold; there the arm is
re-posed kinematically (:class:`~digital_twin_armigo.mujoco_cell.inspection_views.HeldBrickPoser`) over ``--samples``
random joint vectors (all five arm joints, within joint limits) plus the expert's own show pose. Each pose keeping the
arm and the brick ``--clearance`` above the table is measured in the FULL top-camera frame (no depth-bridge crop): brick
pixels, visible fraction (occlusion by the arm), per-face pixels and the brick's distance to the camera. Writes
``poses.jsonl`` (one row per measured pose) and ``summary.json`` into ``--out``.

Kinematic only: no self-collision check and no reachability-in-time check; the chosen poses must still be executed
by the physics expert.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

FACE_NAMES = ("+x", "-x", "+y", "-y", "top", "bottom")


class _Reached(Exception):
	pass


def measure_layout(k: int, lay, samples: int, clearance: float, seed: int) -> list[dict]:
	import bench_physics_expert as bpe

	from digital_twin_armigo.benchmarks import SENSITIVITY_MATRIX
	from digital_twin_armigo.constants import SO101_MJCF_DIR
	from digital_twin_armigo.mujoco_cell import Cell
	from digital_twin_armigo.mujoco_cell.inspection_views import HeldBrickPoser, capture_view
	from digital_twin_armigo.mujoco_cell.physics_expert import ExpertParams, run_closed_loop_episode
	from digital_twin_armigo.physics import load_physics_spec

	spec = load_physics_spec()
	base = {c.name: c for c in SENSITIVITY_MATRIX}["baseline_2x2"]
	cell = Cell(lay, SO101_MJCF_DIR, physics=spec, dynamic_clutter=True, assisted_grasp=False, face_probes=True)
	cam_id = cell.model.camera("top").id
	rng = np.random.default_rng([seed, k])
	rows: list[dict] = []

	def inspect(flex: float, roll_deg: float, frame: int):
		poser = HeldBrickPoser(cell)
		q0 = cell.joints().copy()
		lo, hi = cell.jnt_range[:5, 0], cell.jnt_range[:5, 1]
		qs = [("show", q0)]
		for _ in range(samples):
			q = q0.copy()
			q[:5] = rng.uniform(lo, hi)
			qs.append(("random", q))
		for src, q in qs:
			d = poser.at(q)
			if min(cell.robot_min_z(d), poser.brick_min_z()) < clearance:
				continue
			v = capture_view(cell, "top", crop=None, data=d)
			dist = float(np.linalg.norm(d.xpos[cell.brick_body] - d.cam_xpos[cam_id]))
			rows.append(
				{
					"layout": k,
					"source": src,
					"q": [round(float(x), 4) for x in q[:5]],
					"brick_xyz": [round(float(x), 4) for x in d.xpos[cell.brick_body]],
					"dist_m": round(dist, 4),
					"brick_px": v.brick_px,
					"brick_full_px": v.brick_full_px,
					"visible_fraction": round(v.visible_fraction, 3),
					"faces_px": list(v.faces_px),
					"bbox": list(v.bbox) if v.bbox else None,
				}
			)
		raise _Reached

	try:
		run_closed_loop_episode(
			cell,
			ExpertParams(),
			q_start=bpe._q_start(),
			brick_xy=lay.brick_xy,
			brick_yaw_deg=lay.brick_yaw_deg,
			non_ideal=bpe.bench._mj_non_ideal(base, spec),
			bowls={"normal": lay.green, "defective": lay.pink},
			inspect=inspect,
		)
	except _Reached:
		pass
	finally:
		cell.close()
	return rows


def summarize(rows: list[dict]) -> dict:
	def stats(sel):
		px = np.array([r["brick_px"] for r in sel]) if sel else np.zeros(0)
		return {
			"n": len(sel),
			"brick_px_median": float(np.median(px)) if len(px) else None,
			"brick_px_p95": float(np.percentile(px, 95)) if len(px) else None,
			"brick_px_max": int(px.max()) if len(px) else None,
		}

	show = [r for r in rows if r["source"] == "show"]
	rand = [r for r in rows if r["source"] == "random" and r["visible_fraction"] >= 0.6]
	out = {"show_pose": stats(show), "random_unoccluded": stats(rand)}
	# best achievable pixels per face (unoccluded random poses), median over layouts of the per-layout max
	per_face = {}
	for f, name in enumerate(FACE_NAMES):
		best = []
		for k in sorted({r["layout"] for r in rows}):
			sel = [r["faces_px"][f] for r in rand if r["layout"] == k]
			show_k = [r["faces_px"][f] for r in show if r["layout"] == k]
			best.append((max(sel) if sel else 0, show_k[0] if show_k else 0))
		per_face[name] = {
			"best_random_median": float(np.median([b for b, _ in best])),
			"show_median": float(np.median([s for _, s in best])),
		}
	out["face_px"] = per_face
	return out


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	ap.add_argument("--n", type=int, default=6)
	ap.add_argument("--seed", type=int, default=7)
	ap.add_argument("--samples", type=int, default=1500)
	ap.add_argument("--clearance", type=float, default=0.02)
	ap.add_argument("--out", type=Path, default=ROOT / "results" / "top-camera-reach")
	args = ap.parse_args()
	import bench_physics_expert as bpe

	args.out.mkdir(parents=True, exist_ok=True)
	rows: list[dict] = []
	for k, lay in enumerate(bpe.sample_layouts(args.n, args.seed, bpe.load_top_camera())):
		r = measure_layout(k, lay, args.samples, args.clearance, args.seed)
		print(f"layout {k}: {len(r)} poses measured", flush=True)
		rows += r
	with open(args.out / "poses.jsonl", "w", encoding="utf-8") as f:
		for r in rows:
			f.write(json.dumps(r) + "\n")
	s = summarize(rows)
	(args.out / "summary.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
	print(json.dumps(s, indent=2))


if __name__ == "__main__":
	main()
