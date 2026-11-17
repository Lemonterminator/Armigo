"""Audit GripperMonitor across the 660-episode benchmark suite (540 matrix/held-out + 120 layouts)."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
	sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
	sys.path.insert(0, str(ROOT / "scripts"))

from digital_twin_armigo.benchmarks import HELD_OUT_PERTURBATIONS, SENSITIVITY_MATRIX
from digital_twin_armigo.constants import CELL_MARKERS, SO101_MJCF_DIR
from digital_twin_armigo.mujoco_cell.scene import write_marker_textures

OUT_PATH = ROOT / "results" / "gripper-state" / "audit.json"
TRUTH_FORCE_N = 2.0
TRUTH_MIN_FRAMES = 5
FP_JAW_ENV_MAX_N = 0.5


def _run_audited_episode(task: tuple[str, str, int]) -> dict[str, Any]:
	import bench_physics_expert as bp

	from digital_twin_armigo.mujoco_cell import physics_expert as pe
	from digital_twin_armigo.physics import load_physics_spec

	kind, name, key = task
	spec = load_physics_spec()
	top = bp.load_top_camera()
	conds = {c.name: c for c in (*SENSITIVITY_MATRIX, *HELD_OUT_PERTURBATIONS)}

	frame_hits: list[tuple[float, list[str], str]] = []
	orig_step = pe.PhysicsEpisode.step
	orig_act = pe.ClosedLoopExpert.act
	cur_phase = ["rest"]

	def act(self, obs):
		t, p = orig_act(self, obs)
		cur_phase[0] = p
		return t, p

	def step(self, target):
		orig_step(self, target)
		cell = self.cell
		m, d = cell.model, cell.data
		jaws = cell.fixed_jaw_geoms | cell.moving_jaw_geoms
		brick = {int(g) for g in cell.brick_geoms}
		objs: list[str] = []
		cf = np.zeros(6)
		peak = 0.0
		for cid in range(d.ncon):
			con = d.contact[cid]
			g1, g2 = int(con.geom1), int(con.geom2)
			other = g2 if g1 in jaws else (g1 if g2 in jaws else -1)
			if other < 0 or other in cell.robot_all_geoms or other in brick:
				continue
			mujoco.mj_contactForce(m, d, cid, cf)
			fmag = float(np.linalg.norm(cf[:3]))
			if fmag >= TRUTH_FORCE_N:
				bname = (
					mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[other]) or f"b{m.geom_bodyid[other]}"
				)
				gname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, other) or f"g{other}"
				objs.append(f"{bname}/{gname}")
			peak = max(peak, fmag)
		frame_hits.append((peak, sorted(set(objs)), cur_phase[0]))

	pe.ClosedLoopExpert.act = act
	pe.PhysicsEpisode.step = step
	try:
		if kind == "matrix":
			row = bp.run_episode(conds[name], spec, top, key)
			ep_id = f"{name}_seed{key}"
		else:
			layouts = bp.sample_layouts(60, int(name), top)
			row = bp.run_layout_episode(layouts[key], spec, key)
			ep_id = f"layouts_seed{name}_idx{key:02d}"
	finally:
		pe.ClosedLoopExpert.act = orig_act
		pe.PhysicsEpisode.step = orig_step

	# Find truth events: >= TRUTH_MIN_FRAMES consecutive frames with jaw_env_force >= TRUTH_FORCE_N
	truth_start = None
	run = 0
	truth_objs: set[str] = set()
	truth_phases: set[str] = set()
	for idx, (fmag, objs, ph) in enumerate(frame_hits):
		if fmag >= TRUTH_FORCE_N:
			run += 1
			truth_objs.update(objs)
			truth_phases.add(ph)
			if run == TRUTH_MIN_FRAMES and truth_start is None:
				truth_start = idx - TRUTH_MIN_FRAMES + 1
		else:
			run = 0

	info = row["info"]
	h = hashlib.sha256()
	tel = row.get("_telemetry", {})
	for tk in sorted(tel.keys()):
		h.update(tk.encode())
		h.update(tel[tk].tobytes())
	h.update(b"__phases__")
	if "_phases" in row:
		h.update(row["_phases"].tobytes())
	return {
		"id": ep_id,
		"kind": kind,
		"group": name,
		"key": key,
		"verdict": row["verdict"],
		"reason_codes": list(row["reason_codes"]),
		"n_frames": int(info.get("n_frames", 0)),
		"release_peak_speed_mps": info.get("release_peak_speed_mps"),
		"open_g_used": info.get("grasp_geometry", {}).get("open_g"),
		"block_replans": int(info.get("block_replans", 0)),
		"aborted": info.get("aborted"),
		"infeasible_reason": info.get("infeasible_reason"),
		"jaw_env_contact_max_N": float(info.get("jaw_env_contact_max_N", 0.0)),
		"gripper_blocked_frames": int(info.get("gripper_blocked_frames", 0)),
		"gripper_first_blocked_frame": info.get("gripper_first_blocked_frame"),
		"gripper_block_direction": int(info.get("gripper_block_direction", 0)),
		"telemetry_hash": h.hexdigest(),
		"truth_event": truth_start is not None,
		"truth_start_frame": truth_start,
		"truth_objects": sorted(truth_objs),
		"truth_phases": sorted(truth_phases),
	}


def main() -> int:
	t0 = time.time()
	write_marker_textures(SO101_MJCF_DIR, set(CELL_MARKERS))
	cond_names = [c.name for c in (*SENSITIVITY_MATRIX, *HELD_OUT_PERTURBATIONS)]
	tasks: list[tuple[str, str, int]] = [
		("matrix", cn, s) for cn in cond_names for s in (*range(30), *range(100, 130))
	] + [("layout", str(ls), idx) for ls in (1000, 1001) for idx in range(60)]

	with concurrent.futures.ProcessPoolExecutor(max_workers=12) as pool:
		rows = list(pool.map(_run_audited_episode, tasks))

	fp = [r for r in rows if r["gripper_blocked_frames"] > 0 and r["jaw_env_contact_max_N"] < FP_JAW_ENV_MAX_N]
	fn = [r for r in rows if r["truth_event"] and r["gripper_blocked_frames"] == 0]
	detected_truth = [r for r in rows if r["truth_event"] and r["gripper_blocked_frames"] > 0]
	jaw_over_5n = [r for r in rows if r["jaw_env_contact_max_N"] > 5.0]
	latencies = {
		r["id"]: int(r["gripper_first_blocked_frame"] - r["truth_start_frame"])
		for r in detected_truth
		if r["gripper_first_blocked_frame"] is not None and r["truth_start_frame"] is not None
	}
	med_lat = float(np.median(list(latencies.values()))) if latencies else None

	# Separate misses during non-grasp/open-jaw brushing ("other_cause") from grasp-phase misses
	other_cause = [r for r in fn if "grasp" not in r["truth_phases"]]
	grasp_fn = [r for r in fn if "grasp" in r["truth_phases"]]

	matrix_rows = [r for r in rows if r["kind"] == "matrix"]
	layout_rows = [r for r in rows if r["kind"] == "layout"]
	per_cond: dict[str, list[int]] = {}
	for r in matrix_rows:
		ok, n = per_cond.get(r["group"], [0, 0])
		per_cond[r["group"]] = [ok + (1 if r["verdict"] == "SUCCESS" else 0), n + 1]

	report = {
		"episodes_audited": len(rows),
		"wall_s": round(time.time() - t0, 2),
		"matrix_success": sum(v[0] for v in per_cond.values()),
		"matrix_total": len(matrix_rows),
		"per_condition": per_cond,
		"layouts": [sum(1 for r in layout_rows if r["verdict"] == "SUCCESS"), len(layout_rows)],
		"bounce_outs": sum(1 for r in rows if "bounce_out" in r["reason_codes"]),
		"fast_releases": sum(1 for r in rows if (r["release_peak_speed_mps"] or 0.0) > 1.2),
		"frames_ge_890": [r["id"] for r in rows if r["n_frames"] >= 890],
		"false_positive_episodes": fp,
		"false_negative_episodes": grasp_fn,
		"other_cause": other_cause,
		"detected_episodes": detected_truth,
		"jaw_env_over_5N": jaw_over_5n,
		"detection_latency_frames": latencies,
		"median_latency_frames": med_lat,
		"telemetry_hashes": {r["id"]: r["telemetry_hash"] for r in rows},
		"blocked_episode_ids": [r["id"] for r in rows if r["gripper_blocked_frames"] > 0],
	}
	OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
	OUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
	print(
		f"audited {len(rows)} episodes in {report['wall_s']}s: "
		f"FP={len(fp)} grasp_FN={len(grasp_fn)} other_cause={len(other_cause)} "
		f"detected={len(detected_truth)} jaw>5N={len(jaw_over_5n)} median_latency={med_lat}"
	)
	for r in jaw_over_5n:
		print(
			f"  jaw>5N {r['id']:26s} max={r['jaw_env_contact_max_N']:5.1f}N "
			f"blocked={r['gripper_blocked_frames']:3d} first={r['gripper_first_blocked_frame']} "
			f"truth_start={r['truth_start_frame']} phases={r['truth_phases']} objs={r['truth_objects']}"
		)
	return 0 if (len(fp) == 0 and (med_lat is None or med_lat <= 6.0)) else 1


if __name__ == "__main__":
	raise SystemExit(main())
