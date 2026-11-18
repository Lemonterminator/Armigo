"""Where on the table can the wrist camera scan a brick's top + four side faces before grasping? (kinematic map)

For a grid of brick positions (radius x azimuth around the shoulder-pan axis) the brick is moved there on a kinematic
copy of a benchmark cell and three things are checked, without stepping the physics:

* **graspable** -- the closed-loop expert's grasp candidates for that pose include one whose IK reaches the grasp
  point (error < ``--grasp-tol``) with non-negative clearance to bowls / clutter;
* **wrist scan** -- the gripper hovers over the brick with the jaws pointing straight down (``ik`` with the jaws-down
  constraint, position error < ``--ik-tol`` and approach axis within ``--down-tol-deg`` of vertical), and the wrist
  roll sweeps its range. Because the wrist camera sits off the roll axis, the sweep orbits the camera around the
  brick. Hover configurations: height x horizontal offset of the roll axis from the brick centre (centred / off-centre
  rotation) x an optional wrist-flex tilt added after the IK (0 = strictly vertical). A configuration scans the brick
  when, over the reachable rolls, each of the five faces (top, +-x, +-y) shows at least ``--min-px`` pixels in some
  view. ``scan_vertical`` = some configuration with tilt 0 scans the brick; ``scan_any`` = some configuration does;
* **outside scan** -- the same sweep with :func:`ik_reach` (the jaws may tilt to reach further, so the camera looks
  at the brick from outside rather than from above); ``scan_outside`` / ``scan_any``;
* positions over a bowl are flagged (``over_bowl``), not skipped: bowls are fixed in this cell.

Writes ``positions.jsonl`` and ``summary.json`` (per radius: fractions graspable / scannable, and the radial band where
the scan works) into ``--out``. Kinematic only: no self-collision or clutter-collision check during the sweep beyond
keeping every robot geom ``--min-z`` above the table.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

FACE_NAMES = ("+x", "-x", "+y", "-y", "top", "bottom")
SCAN_FACES = (0, 1, 2, 3, 4)

_W: dict = {}


def _init(layout_seed: int) -> None:
	import bench_physics_expert as bpe

	from digital_twin_armigo.constants import SO101_MJCF_DIR
	from digital_twin_armigo.mujoco_cell import Cell
	from digital_twin_armigo.mujoco_cell._mj import mujoco
	from digital_twin_armigo.physics import load_physics_spec

	lay = bpe.sample_layouts(1, layout_seed, bpe.load_top_camera())[0]
	cell = Cell(lay, SO101_MJCF_DIR, physics=load_physics_spec(), assisted_grasp=False, face_probes=True)
	mujoco.mj_forward(cell.model, cell.data)
	_W["cell"] = cell
	_W["q_start"] = bpe._q_start()


def pan_axis_xy(cell) -> np.ndarray:
	from digital_twin_armigo.mujoco_cell._mj import mujoco

	jid = mujoco.mj_name2id(cell.model, mujoco.mjtObj.mjOBJ_JOINT, "shoulder_pan")
	return cell.data.xanchor[jid, :2].copy()


def _vertical_seed(cell, target, q_seed, seeds, tol, cos_tol) -> np.ndarray | None:
	"""A jaws-down IK solution at ``target`` (roll 0) from the ready pose or from random restarts, or None."""
	from digital_twin_armigo.mujoco_cell.ik import ik
	from digital_twin_armigo.mujoco_cell.table_relocation import site_err_down

	d = copy.copy(cell.data)
	for s in [q_seed, *seeds]:
		q0 = np.asarray(s, dtype=np.float64).copy()
		q0[4] = 0.0
		q = ik(cell, target, q0, 0.0, iters=600)
		err, down = site_err_down(cell, d, q, target)
		if err <= tol and down >= cos_tol:
			return q
	return None


def check_position(job: tuple) -> dict:
	from digital_twin_armigo.constants import BOWL_R
	from digital_twin_armigo.mujoco_cell._mj import mujoco
	from digital_twin_armigo.mujoco_cell.ik import ik
	from digital_twin_armigo.mujoco_cell.inspection_views import capture_view
	from digital_twin_armigo.mujoco_cell.physics_expert import ClosedLoopExpert, ExpertParams
	from digital_twin_armigo.mujoco_cell.table_relocation import site_err_down

	radius, az_deg, yaw_deg, a = job
	cell = _W["cell"]
	m = cell.model
	base = pan_axis_xy(cell)
	xy = base + radius * np.array([math.cos(math.radians(az_deg)), math.sin(math.radians(az_deg))])
	lay = cell.layout
	row: dict = {"radius": radius, "azimuth_deg": az_deg, "yaw_deg": yaw_deg, "xy": [round(float(v), 4) for v in xy]}
	row["over_bowl"] = bool(
		min(np.linalg.norm(xy - np.asarray(lay.green)), np.linalg.norm(xy - np.asarray(lay.pink))) < BOWL_R + 0.02
	)
	z = lay.brick_size[2] / 2.0 + 0.0005
	yaw = math.radians(yaw_deg)
	quat_wxyz = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
	t0 = time.perf_counter()

	# graspable?
	params = ExpertParams()
	ex = ClosedLoopExpert(cell, _W["q_start"], (0.0, 0.0), params)
	ex.obs = {
		"q": _W["q_start"],
		"brick_pos": np.array([xy[0], xy[1], z]),
		"brick_quat": np.array([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]),
	}
	cands = ex.grasp_candidates()
	ok = [c for c in cands if c["ik_error_m"] < a.grasp_tol and c["clearance_m"] >= 0.0]
	row["graspable"] = bool(ok)
	row["grasp_ik_error_m"] = round(min((c["ik_error_m"] for c in cands), default=float("nan")), 4)

	# wrist scan
	d = copy.copy(cell.data)
	adr = cell.brick_qadr
	d.qpos[adr : adr + 3] = [xy[0], xy[1], z]
	d.qpos[adr + 3 : adr + 7] = quat_wxyz
	lo, hi = cell.jnt_range[4]
	rolls = np.linspace(lo, hi, a.rolls)
	radial = (xy - base) / max(float(np.linalg.norm(xy - base)), 1e-9)
	tang = np.array([-radial[1], radial[0]])
	offsets = {"centre": np.zeros(2)}
	for r in a.offsets:
		offsets[f"in{r:.2f}"] = -r * radial
		offsets[f"out{r:.2f}"] = r * radial
		offsets[f"side{r:.2f}"] = r * tang
	cos_tol = math.cos(math.radians(a.down_tol_deg))
	configs = []
	q_seed = np.array([*params.ready_q, params.open_g])
	rng_seeds = np.random.default_rng(0).uniform(cell.jnt_range[:, 0], cell.jnt_range[:, 1], (a.seeds, 6))
	for h in a.heights:
		for oname, off in offsets.items():
			target = np.array([xy[0] + off[0], xy[1] + off[1], h])
			best = {tilt: np.zeros(6) for tilt in a.tilts}
			n_reach = 0
			seed = _vertical_seed(cell, target, q_seed, rng_seeds, a.ik_tol, cos_tol)
			q_prev = seed.copy() if seed is not None else None
			for roll in rolls:
				if q_prev is None:
					break
				q0 = q_prev.copy()
				q0[4] = roll
				q = ik(cell, target, q0, float(roll), iters=400)
				err, down = site_err_down(cell, d, q, target)
				if err > a.ik_tol or down < cos_tol:
					continue
				n_reach += 1
				q_prev = q
				for tilt in a.tilts:
					qt = q.copy()
					qt[3] = np.clip(qt[3] + tilt, *cell.jnt_range[3])
					qt[5] = params.open_g
					d.qpos[cell.qadr] = qt
					mujoco.mj_kinematics(m, d)
					mujoco.mj_camlight(m, d)
					if cell.robot_min_z(d) < a.min_z:
						continue
					v = capture_view(cell, "wrist_cam", data=d)
					best[tilt] = np.maximum(best[tilt], v.faces_px)
			for tilt in a.tilts:
				covered = [int(best[tilt][f] >= a.min_px) for f in SCAN_FACES]
				configs.append(
					{
						"height": h,
						"offset": oname,
						"tilt": tilt,
						"rolls_reached": n_reach,
						"faces_px": [int(v) for v in best[tilt]],
						"n_faces": sum(covered),
					}
				)
	# "outside" scan: the jaws may tilt (ik_reach), the camera then looks at the brick from further out
	from digital_twin_armigo.mujoco_cell.ik import ik_reach

	for h in a.reach_heights:
		for oname, off in offsets.items():
			target = np.array([xy[0] + off[0], xy[1] + off[1], h])
			best_r = np.zeros(6)
			n_reach = 0
			q_prev = q_seed.copy()
			for roll in rolls:
				q0 = q_prev.copy()
				q0[4] = roll
				q = ik_reach(cell, target, q0, float(roll), tol=a.ik_tol)
				err, _ = site_err_down(cell, d, q, target)
				if err > a.ik_tol:
					continue
				n_reach += 1
				q_prev = q
				q = q.copy()
				q[5] = params.open_g
				d.qpos[cell.qadr] = q
				mujoco.mj_kinematics(m, d)
				mujoco.mj_camlight(m, d)
				if cell.robot_min_z(d) < a.min_z:
					continue
				best_r = np.maximum(best_r, capture_view(cell, "wrist_cam", data=d).faces_px)
			configs.append(
				{
					"height": h,
					"offset": oname,
					"tilt": "reach",
					"rolls_reached": n_reach,
					"faces_px": [int(v) for v in best_r],
					"n_faces": sum(int(best_r[f] >= a.min_px) for f in SCAN_FACES),
				}
			)
	full = [c for c in configs if c["n_faces"] == len(SCAN_FACES)]
	vert = [c for c in full if c["tilt"] == 0.0]
	row["scan_vertical"] = bool(vert)
	row["scan_any"] = bool(full)
	row["scan_outside"] = any(c["tilt"] == "reach" for c in full)
	row["max_faces_vertical"] = max((c["n_faces"] for c in configs if c["tilt"] == 0.0), default=0)
	row["max_faces_any"] = max((c["n_faces"] for c in configs), default=0)
	pick = (vert or full or sorted(configs, key=lambda c: -c["n_faces"]))[:1]
	row["best_config"] = pick[0] if pick else None
	row["scan_configs_vertical"] = [f"h{c['height']}/{c['offset']}" for c in vert]
	row["seconds"] = round(time.perf_counter() - t0, 2)
	return row


def summarize(rows: list[dict]) -> dict:
	out: dict = {"n_positions": len(rows), "by_radius": {}}
	for rad in sorted({r["radius"] for r in rows}):
		sel = [r for r in rows if r["radius"] == rad]
		out["by_radius"][f"{rad:.3f}"] = {
			"n": len(sel),
			"graspable": round(sum(r["graspable"] for r in sel) / len(sel), 2),
			"scan_vertical": round(sum(r["scan_vertical"] for r in sel) / len(sel), 2),
			"scan_outside": round(sum(r["scan_outside"] for r in sel) / len(sel), 2),
			"scan_any": round(sum(r["scan_any"] for r in sel) / len(sel), 2),
			"graspable_not_scannable_vertical": round(
				sum(r["graspable"] and not r["scan_vertical"] for r in sel) / len(sel), 2
			),
		}

	def band(key: str) -> list[float] | None:
		ok = [r["radius"] for r in rows if r[key]]
		return [min(ok), max(ok)] if ok else None

	out["radius_band_m"] = {k: band(k) for k in ("graspable", "scan_vertical", "scan_outside", "scan_any")}
	return out


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	ap.add_argument("--r-min", type=float, default=0.08)
	ap.add_argument("--r-max", type=float, default=0.46)
	ap.add_argument("--r-step", type=float, default=0.02)
	ap.add_argument("--az", default="-90:90:15", help="azimuth start:stop:step (deg, around the pan axis)")
	ap.add_argument("--yaws", default="0,30", help="brick yaws (deg)")
	ap.add_argument("--heights", default="0.06,0.075,0.09")
	ap.add_argument("--reach-heights", default="0.07,0.10,0.14", help="hover heights of the tilted (outside) scan")
	ap.add_argument("--offsets", default="0.03", help="off-centre distances of the roll axis (m)")
	ap.add_argument("--tilts", default="0,-0.3,0.3", help="wrist-flex tilts added after the IK (rad); 0 = vertical")
	ap.add_argument("--rolls", type=int, default=16)
	ap.add_argument("--seeds", type=int, default=12, help="random IK restarts when the ready-pose seed fails")
	ap.add_argument("--min-px", type=int, default=800)
	ap.add_argument("--ik-tol", type=float, default=0.004)
	ap.add_argument("--grasp-tol", type=float, default=0.005)
	ap.add_argument("--down-tol-deg", type=float, default=8.0)
	ap.add_argument("--min-z", type=float, default=0.02)
	ap.add_argument("--layout-seed", type=int, default=7)
	ap.add_argument("--workers", type=int, default=6)
	ap.add_argument("--out", type=Path, default=ROOT / "results" / "scan-zone")
	a = ap.parse_args()
	a.heights = [float(v) for v in a.heights.split(",")]
	a.reach_heights = [float(v) for v in a.reach_heights.split(",")]
	a.offsets = [float(v) for v in a.offsets.split(",") if v]
	a.tilts = [float(v) for v in a.tilts.split(",")]
	s0, s1, st = (float(v) for v in a.az.split(":"))
	azs = list(np.arange(s0, s1 + 1e-9, st))
	radii = list(np.round(np.arange(a.r_min, a.r_max + 1e-9, a.r_step), 3))
	yaws = [float(v) for v in a.yaws.split(",")]
	jobs = [(float(r), float(az), y, a) for r in radii for az in azs for y in yaws]
	a.out.mkdir(parents=True, exist_ok=True)
	rows = []
	t0 = time.perf_counter()
	with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.layout_seed,)) as pool:
		for i, r in enumerate(pool.map(check_position, jobs, chunksize=2)):
			rows.append(r)
			if i % 25 == 0:
				print(f"{i + 1}/{len(jobs)} ({time.perf_counter() - t0:.0f}s)", flush=True)
	with open(a.out / "positions.jsonl", "w", encoding="utf-8") as f:
		for r in rows:
			f.write(json.dumps(r) + "\n")
	s = summarize(rows)
	s["config"] = {k: (v if not isinstance(v, Path) else str(v)) for k, v in vars(a).items()}
	(a.out / "summary.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
	print(json.dumps({k: v for k, v in s.items() if k != "config"}, indent=2))


if __name__ == "__main__":
	main()
