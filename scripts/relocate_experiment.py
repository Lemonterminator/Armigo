"""Validate "put the brick back down on the table closer, then inspect it" on real MuJoCo episodes.

Cell changes for this experiment only: the bowls are moved out of the workspace (behind the robot), clutter is
removed, and the top camera is moved closer: ``--cam-factor`` x the gripper's maximum reach from the shoulder-pan
anchor (measured, :func:`max_tip_reach`), at elevation ``--cam-elev-deg`` in front of the robot, looking at the
inspection zone.

Each trial starts the brick at a random position in the band where it is graspable but cannot be scanned straight
down (``--r-start``), at a random azimuth and yaw. The :class:`RelocateExpert` picks it, puts it down on the table
at the relocation target (same azimuth, ``--zone-outer - --margin`` from the shoulder axis), scans the top and four
sides with the wrist camera from above, then regrasps and shows the bottom face to the top camera, and puts the
brick down again. Measured per trial: relocation (placed, xy error, tilt, yaw change), the wrist scan's faces
actually seen on the physics state (>= ``--min-px`` each), the bottom face's pixels in the top camera, and the top
camera's view of the brick on the table before / after relocation.

Writes ``trials.jsonl`` and ``summary.json`` into ``--out``.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

FACE_NAMES = ("+x", "-x", "+y", "-y", "top", "bottom")


def camera_pose(anchor: np.ndarray, reach: float, factor: float, elev_deg: float, look_r: float):
	"""Top camera at ``factor * reach`` from the shoulder anchor, in front of the robot (azimuth 0) at elevation
	``elev_deg``, looking at the point ``look_r`` in front of the anchor on the table."""
	from digital_twin_armigo.camera import CameraPose

	e = math.radians(elev_deg)
	pos = anchor + factor * reach * np.array([math.cos(e), 0.0, math.sin(e)])
	look = np.array([anchor[0] + look_r, anchor[1], 0.02])
	f = (look - pos) / np.linalg.norm(look - pos)
	return CameraPose(
		x=float(pos[0]),
		y=float(pos[1]),
		height=float(pos[2]),
		yaw_deg=math.degrees(math.atan2(f[1], f[0])),
		tilt_deg=math.degrees(math.acos(-f[2])),
		roll_deg=0.0,
		focal_px=612.47,
	), pos


def run_trial(job: dict) -> dict:
	import bench_physics_expert as bpe

	from digital_twin_armigo.benchmarks import SENSITIVITY_MATRIX
	from digital_twin_armigo.camera import CameraPose
	from digital_twin_armigo.constants import SO101_MJCF_DIR
	from digital_twin_armigo.mujoco_cell import Cell
	from digital_twin_armigo.mujoco_cell.episode import PhysicsEpisode, observe
	from digital_twin_armigo.mujoco_cell.inspection_views import capture_view
	from digital_twin_armigo.mujoco_cell.physics_expert import ExpertParams
	from digital_twin_armigo.mujoco_cell.table_relocation import RelocateExpert, RelocateParams
	from digital_twin_armigo.physics import load_physics_spec

	t0 = time.perf_counter()
	spec = load_physics_spec()
	lay = bpe.sample_layouts(1, job["layout_seed"], bpe.load_top_camera())[0]
	lay.green, lay.pink = tuple(job["bowls"][0]), tuple(job["bowls"][1])
	lay.clutter = []
	lay.top = CameraPose(**job["camera"])
	lay.brick_xy = tuple(job["start_xy"])
	lay.brick_yaw_deg = job["yaw_deg"]
	lay.defect_face = 5
	cell = Cell(lay, SO101_MJCF_DIR, physics=spec, dynamic_clutter=True, assisted_grasp=False, face_probes=True)
	views: list[dict] = []

	def on_view(stage: str, ex) -> None:
		cam = "wrist_cam" if stage == "scan" else "top"
		v = capture_view(cell, cam)
		views.append({"stage": stage, "camera": cam, "faces_px": list(v.faces_px), "brick_px": v.brick_px})

	base = {c.name: c for c in SENSITIVITY_MATRIX}["baseline_2x2"]
	params = ExpertParams(max_frames=job["max_frames"])
	rp = RelocateParams(min_scan_px=job["min_px"])
	try:
		ep = PhysicsEpisode(
			cell, bpe._q_start(), brick_xy=lay.brick_xy, brick_yaw_deg=lay.brick_yaw_deg,
			non_ideal=bpe.bench._mj_non_ideal(base, spec),
		)  # fmt: skip
		ex = RelocateExpert(cell, bpe._q_start(), job["target_xy"], params, rp, on_view)
		while ep.n_frames < params.max_frames:
			target, _ = ex.act(observe(cell, ep))
			if ex.done:
				break
			ep.step(target)
		bpos = cell.data.xpos[cell.brick_body].copy()
		rot = cell.data.xmat[cell.brick_body].reshape(3, 3)
	finally:
		cell.close()
	res = dict(ex.result)
	placed = res.get("placed_brick_pos")
	out = {
		**{k: job[k] for k in ("trial", "start_r", "azimuth_deg", "yaw_deg", "carry_m")},
		"start_xy": job["start_xy"],
		"target_xy": [round(v, 4) for v in job["target_xy"]],
		"stages": res["stages"],
		"failed": res.get("failed"),
		"truncated": not ex.done,
		"n_frames": ep.n_frames,
		"scan_mode": res.get("scan_mode"),
		"bottom_failed": res.get("bottom_failed"),
		"bottom_planned_px": res.get("bottom_planned_px"),
		"lost": bool(ex.lost),
	}
	if placed is not None:
		out["place_xy_err_m"] = round(float(np.linalg.norm(np.array(placed[:2]) - np.array(job["target_xy"]))), 4)
	out["final_xy_err_m"] = round(float(np.linalg.norm(bpos[:2] - np.array(job["target_xy"]))), 4)
	out["final_tilt_deg"] = round(math.degrees(math.acos(min(1.0, abs(float(rot[2, 2]))))), 2)
	out["final_on_table"] = bool(abs(bpos[2] - lay.brick_size[2] / 2.0) < 0.004)
	scan = [v for v in views if v["stage"] == "scan"]
	best = np.max([v["faces_px"] for v in scan], axis=0) if scan else np.zeros(6)
	out["scan_views"] = len(scan)
	out["scan_faces_px"] = [int(x) for x in best]
	out["scan_faces_covered"] = [FACE_NAMES[f] for f in range(5) if best[f] >= job["min_px"]]
	out["scan_all5"] = len(out["scan_faces_covered"]) == 5
	for st in ("start_top", "placed_top", "bottom"):
		vs = [v for v in views if v["stage"] == st]
		if vs:
			out[f"{st}_faces_px"] = vs[0]["faces_px"]
			out[f"{st}_brick_px"] = vs[0]["brick_px"]
	out["bottom_px"] = out.get("bottom_faces_px", [0] * 6)[5]
	out["wall_s"] = round(time.perf_counter() - t0, 1)
	return out


def summarize(trials: list[dict], min_bottom_px: int) -> dict:
	n = len(trials)

	def rate(f) -> float:
		return round(sum(bool(f(t)) for t in trials) / max(n, 1), 3)

	def med(key: str):
		xs = [t[key] for t in trials if t.get(key) is not None]
		return round(statistics.median(xs), 4) if xs else None

	return {
		"n": n,
		"grasped": rate(lambda t: "grasped" in t["stages"]),
		"placed": rate(lambda t: "placed" in t["stages"]),
		"scan_all5": rate(lambda t: t["scan_all5"]),
		"scan_mode": {m: sum(t["scan_mode"] == m for t in trials) for m in {t["scan_mode"] for t in trials}},
		"bottom_shown": rate(lambda t: "bottom_shown" in t["stages"]),
		f"bottom_px>={min_bottom_px}": rate(lambda t: t["bottom_px"] >= min_bottom_px),
		"full_pipeline": rate(lambda t: t["scan_all5"] and t["bottom_px"] >= min_bottom_px and t["final_on_table"]),
		"failures": {str(k): sum(t["failed"] == k for t in trials) for k in {t["failed"] for t in trials}},
		"bottom_failures": {
			str(k): sum(t["bottom_failed"] == k for t in trials) for k in {t["bottom_failed"] for t in trials}
		},
		"lost": rate(lambda t: t["lost"]),
		"truncated": rate(lambda t: t["truncated"]),
		"place_xy_err_m_median": med("place_xy_err_m"),
		"final_xy_err_m_median": med("final_xy_err_m"),
		"final_tilt_deg_median": med("final_tilt_deg"),
		"final_on_table": rate(lambda t: t["final_on_table"]),
		"carry_m_median": med("carry_m"),
		"bottom_px_median": med("bottom_px"),
		"top_brick_px_start_median": med("start_top_brick_px"),
		"top_brick_px_placed_median": med("placed_top_brick_px"),
		"frames_median": med("n_frames"),
		"wall_s_median": med("wall_s"),
	}


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	ap.add_argument("--n", type=int, default=40)
	ap.add_argument("--seed", type=int, default=11)
	ap.add_argument("--r-start", default="0.30,0.38", help="start radius band (m) around the shoulder axis")
	ap.add_argument("--az-max", type=float, default=75.0, help="start azimuth uniform in [-az, az] (deg)")
	ap.add_argument("--zone-outer", type=float, default=0.26, help="outer radius of the straight-down scan zone (m)")
	ap.add_argument("--margin", type=float, default=0.04)
	ap.add_argument("--cam-factor", type=float, default=1.35, help="camera distance / max gripper reach (1.2-1.5)")
	ap.add_argument("--cam-elev-deg", type=float, default=45.0)
	ap.add_argument("--min-px", type=int, default=800, help="wrist-scan pixels a face needs to count as seen")
	ap.add_argument("--min-bottom-px", type=int, default=400, help="top-camera bottom-face pixels counted as seen")
	ap.add_argument("--max-frames", type=int, default=3600)
	ap.add_argument("--workers", type=int, default=8)
	ap.add_argument("--out", type=Path, default=ROOT / "results" / "relocate")
	a = ap.parse_args()

	import bench_physics_expert as bpe

	from digital_twin_armigo.constants import SO101_MJCF_DIR
	from digital_twin_armigo.mujoco_cell import Cell
	from digital_twin_armigo.mujoco_cell._mj import mujoco
	from digital_twin_armigo.mujoco_cell.table_relocation import max_tip_reach, relocation_target, shoulder_anchor
	from digital_twin_armigo.physics import load_physics_spec

	probe = Cell(bpe.sample_layouts(1, a.seed, bpe.load_top_camera())[0], SO101_MJCF_DIR, physics=load_physics_spec())
	mujoco.mj_forward(probe.model, probe.data)
	anchor = shoulder_anchor(probe)
	reach = max_tip_reach(probe)
	probe.close()
	cam, cam_pos = camera_pose(anchor, reach, a.cam_factor, a.cam_elev_deg, a.zone_outer - a.margin)
	# bowls behind the robot, beyond its reach
	bowls = [
		(float(anchor[0] - 0.6 * reach * 1.2), float(anchor[1] + 0.6 * reach * 1.2)),
		(float(anchor[0] - 0.6 * reach * 1.2), float(anchor[1] - 0.6 * reach * 1.2)),
	]
	r0, r1 = (float(v) for v in a.r_start.split(","))
	rng = np.random.default_rng(a.seed)
	jobs = []
	for i in range(a.n):
		r = float(rng.uniform(r0, r1))
		az = float(rng.uniform(-a.az_max, a.az_max))
		xy = anchor[:2] + r * np.array([math.cos(math.radians(az)), math.sin(math.radians(az))])
		tgt, carry = relocation_target(xy, anchor, a.zone_outer, a.margin)
		jobs.append(
			{
				"trial": i,
				"layout_seed": int(a.seed * 1000 + i),
				"start_r": round(r, 4),
				"azimuth_deg": round(az, 2),
				"yaw_deg": round(float(rng.uniform(-45, 45)), 2),
				"start_xy": [float(v) for v in xy],
				"target_xy": [float(v) for v in tgt],
				"carry_m": round(carry, 4),
				"bowls": bowls,
				"camera": {
					k: getattr(cam, k) for k in ("x", "y", "height", "yaw_deg", "tilt_deg", "roll_deg", "focal_px")
				},
				"min_px": a.min_px,
				"max_frames": a.max_frames,
			}
		)
	a.out.mkdir(parents=True, exist_ok=True)
	trials = []
	t0 = time.perf_counter()
	with ProcessPoolExecutor(a.workers) as pool:
		futs = [pool.submit(run_trial, j) for j in jobs]
		for fut in as_completed(futs):
			t = fut.result()
			trials.append(t)
			print(
				f"[{t['trial']:3d}] r={t['start_r']:.3f} carry={t['carry_m']:.3f} stages={t['stages']} "
				f"fail={t['failed']} scan={t['scan_mode']}/{len(t['scan_faces_covered'])} bottom={t['bottom_px']} "
				f"tilt={t['final_tilt_deg']} ({time.perf_counter() - t0:.0f}s, {len(trials)}/{len(jobs)})",
				flush=True,
			)
	trials.sort(key=lambda t: t["trial"])
	(a.out / "trials.jsonl").write_text("".join(json.dumps(t) + "\n" for t in trials), encoding="utf-8")
	summary = {
		"results": summarize(trials, a.min_bottom_px),
		"setup": {
			"shoulder_anchor": [round(float(v), 4) for v in anchor],
			"max_tip_reach_m": round(reach, 4),
			"camera_pos": [round(float(v), 4) for v in cam_pos],
			"camera_dist_from_anchor_m": round(float(np.linalg.norm(cam_pos - anchor)), 4),
			"camera": jobs[0]["camera"],
			"bowls": bowls,
			"args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
		},
	}
	(a.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
	print(json.dumps(summary, indent=2))


if __name__ == "__main__":
	main()
