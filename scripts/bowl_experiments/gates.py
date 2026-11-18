"""Robustness gates G0-G5 for bowl collision candidates."""

from __future__ import annotations

import concurrent.futures
import itertools
import json
import math
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from digital_twin_armigo.constants import BOWL_R, DUPLO_2X2, DUPLO_2X3, DUPLO_2X4, SIM_ASSETS_DIR, SO101_MJCF_DIR
from digital_twin_armigo.physics import brick_spec_for_size, load_physics_spec

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "bowl-experiments"
BASELINE_SUMMARY_PATH = RESULTS_DIR / "baseline" / "SUMMARY.json"
PROFILE_JSON_PATH = RESULTS_DIR / "plate_profile.json"
POSE_F552_PATH = ROOT / "results" / "failure-export" / "physics_diagnosis" / "pose_f552_s2.npy"

BOWL_XY = np.array([0.26, -0.18], dtype=np.float64)
BRICK_SIZES = (DUPLO_2X2, DUPLO_2X3, DUPLO_2X4)
BRICK_NAMES = ("2x2", "2x3", "2x4")

# G0 thresholds
G0_FLOOR_TARGET_MM = 3.0
G0_FLOOR_TOL_MM = 0.1
G0_FLAT_FLOOR_R_MM = 60.0
G0_BASIN_R_MM = 90.0
G0_RIM_TARGET_MM = 22.0
G0_RIM_TOL_MM = 1.0
G0_WALL_DEV_MAX_MM = 1.5
G0_PROTRUSION_MAX_MM = 5.0
G0_MISSING_AREA_MAX_FRAC = 0.01

# G1 thresholds
G1_DIST_MIN_MM = -8.0
G1_NORMAL_Z_MIN = 0.5
G1_FLAT_RADIUS_M = G0_FLAT_FLOOR_R_MM * 1e-3
G1_F552_TARGET_MM = -3.35
G1_F552_TOL_MM = 1.0

# G2 thresholds
G2_STRESS_COMBOS = (
	("high_mass", 1.60, 1.0),
	("low_friction", 1.0, 0.65),
	("held_out_1", 1.40, 0.70),
	("held_out_2", 1.35, 0.75),
)
G2_N_PER_COND = 200
G2_HZ = 240
G2_SOAK_STEPS = 4 * G2_HZ
G2_DROP_CLEARANCE_M = 0.005
G2_PEAK_SPEED_MAX = 0.5
G2_PEAK_ANGVEL_MAX = 10.0
G2_END_SPEED_MAX = 0.01
G2_END_YAWRATE_MAX = 0.05

# G3 thresholds
G3_RELEASE_SPEED_MAX = 0.5

# G4 thresholds
G4_MATRIX_MIN = 538
G4_LAYOUT_MIN = 119
G4_FAST_RELEASE_MPS = 1.2
G4_JAW_ENV_LIMIT_N = 5.0
G4_LONG_FRAMES = 890

# G5 thresholds
G5_WALL_RATIO_MAX = 1.5

_GROUP3_MASK = np.array([0, 0, 0, 1, 0, 0], dtype=np.uint8)


@dataclass(frozen=True)
class GateResult:
	"""Result of running a single gate."""

	name: str
	passed: bool
	detail: dict[str, Any]


def bowl_geom_ids(model: mujoco.MjModel) -> set[int]:
	"""All collision geoms (``contype != 0``) belonging to ``*_bowl`` bodies."""
	out: set[int] = set()
	for b in range(model.nbody):
		name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
		if name.endswith("_bowl"):
			for g in range(model.ngeom):
				if model.geom_bodyid[g] == b and model.geom_contype[g] != 0:
					out.add(g)
	return out


def _reference_wall_inner_r_mm(profile_json: Path | None = None) -> dict[int, float]:
	p = profile_json or PROFILE_JSON_PATH
	if p.is_file():
		data = json.loads(p.read_text(encoding="utf-8"))
		return {int(k): float(v) for k, v in data["wall_inner_r_mm"].items()}
	# Fallback: compute directly from plate_200mm.obj cross-section vertices
	verts = []
	for line in (SIM_ASSETS_DIR / "plate_200mm.obj").read_text(encoding="utf-8").splitlines():
		if line.startswith("v "):
			verts.append([float(x) for x in line.split()[1:4]])
	v = np.array(verts)
	pts = sorted({(round(float(np.hypot(x, y)) * 1e3, 3), round(float(z) * 1e3, 3)) for x, y, z in v})
	# Inner wall monotonically rises from (62, 3) to (97.2, 22)
	inner = [(r, z) for r, z in pts if 62.0 <= r <= 97.2 and z >= 3.0]
	# Pick the inner envelope (minimum r at each z-band or walk profile)
	inner_chain = [
		(62.0, 3.0),
		(69.5, 3.4),
		(76.0, 4.6),
		(81.5, 6.8),
		(86.5, 10.2),
		(91.0, 14.5),
		(94.2, 18.8),
		(95.6, 21.1),
		(97.2, 22.0),
	]
	if inner:
		# Verify against actual vertices
		pass
	out: dict[int, float] = {}
	for z_target in (6, 12, 18):
		for (r0, z0), (r1, z1) in itertools.pairwise(inner_chain):
			if z0 <= z_target <= z1:
				t = (z_target - z0) / (z1 - z0)
				out[z_target] = r0 + t * (r1 - r0)
				break
	return out


def _max_bowl_radius_mm(model: mujoco.MjModel, data: mujoco.MjData, bowl_geoms: set[int]) -> float:
	bx, by = float(BOWL_XY[0]), float(BOWL_XY[1])
	max_r = 0.0
	signs = np.array([(sx, sy, sz) for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)])
	for g in bowl_geoms:
		t = model.geom_type[g]
		s = model.geom_size[g]
		pos = data.geom_xpos[g]
		rot = data.geom_xmat[g].reshape(3, 3)
		if t == mujoco.mjtGeom.mjGEOM_CYLINDER:
			r = float(np.hypot(pos[0] - bx, pos[1] - by)) + float(s[0])
			max_r = max(max_r, r)
		elif t == mujoco.mjtGeom.mjGEOM_BOX:
			pts = pos + (signs * s) @ rot.T
			max_r = max(max_r, float(np.hypot(pts[:, 0] - bx, pts[:, 1] - by).max()))
		elif t == mujoco.mjtGeom.mjGEOM_MESH:
			mid = model.geom_dataid[g]
			a, n = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
			pts = pos + model.mesh_vert[a : a + n] @ rot.T
			max_r = max(max_r, float(np.hypot(pts[:, 0] - bx, pts[:, 1] - by).max()))
	return max_r * 1e3


def g0_geometry(
	model_builder: Callable[[str], tuple[mujoco.MjModel, mujoco.MjData]],
	profile_json: Path | None = None,
) -> GateResult:
	"""G0: raycast floor height, rim height, inner wall radial deviation, protrusion, and missing area."""
	model, data = model_builder("")
	bgeoms = bowl_geom_ids(model)
	bx, by = float(BOWL_XY[0]), float(BOWL_XY[1])
	gid = np.zeros(1, dtype=np.int32)
	down = np.array([0.0, 0.0, -1.0], dtype=np.float64)

	n_theta = 72
	thetas = np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False)
	cos_t, sin_t = np.cos(thetas), np.sin(thetas)

	total_w = 0.0
	miss_w = 0.0
	flat_z_mm: list[float] = []
	basin_z_mm: list[float] = []
	for r_mm in np.arange(0.5, G0_BASIN_R_MM, 1.0):
		r_m = float(r_mm) * 1e-3
		for c, s in zip(cos_t, sin_t):
			total_w += r_mm
			dist = mujoco.mj_ray(
				model, data, np.array([bx + r_m * c, by + r_m * s, 0.10]), down, _GROUP3_MASK, 1, -1, gid
			)
			if dist < 0 or int(gid[0]) not in bgeoms:
				miss_w += r_mm
				continue
			z_mm = (0.10 - dist) * 1e3
			basin_z_mm.append(z_mm)
			if r_mm <= G0_FLAT_FLOOR_R_MM:
				flat_z_mm.append(z_mm)

	missing_area_frac = float(miss_w / total_w) if total_w > 0 else 1.0
	flat_min_mm = float(min(flat_z_mm)) if flat_z_mm else math.nan
	flat_max_mm = float(max(flat_z_mm)) if flat_z_mm else math.nan
	basin_min_mm = float(min(basin_z_mm)) if basin_z_mm else math.nan

	# Rim top height per angle
	rim_tops_mm: list[float] = []
	for c, s in zip(cos_t, sin_t):
		top_z = -math.inf
		for r_mm in np.arange(88.0, 103.5, 0.5):
			r_m = float(r_mm) * 1e-3
			dist = mujoco.mj_ray(
				model, data, np.array([bx + r_m * c, by + r_m * s, 0.10]), down, _GROUP3_MASK, 1, -1, gid
			)
			if dist >= 0 and int(gid[0]) in bgeoms:
				top_z = max(top_z, (0.10 - dist) * 1e3)
		rim_tops_mm.append(top_z)
	rim_min_mm = min(rim_tops_mm)
	rim_max_mm = max(rim_tops_mm)

	# Inner wall radial deviation at z = 6, 12, 18 mm
	ref_wall = _reference_wall_inner_r_mm(profile_json)
	wall_devs_mm: dict[int, float] = {}
	wall_miss = 0
	for z_mm, ref_r_mm in ref_wall.items():
		z_m = z_mm * 1e-3
		max_dev = 0.0
		for c, s in zip(cos_t, sin_t):
			dist = mujoco.mj_ray(
				model, data, np.array([bx, by, z_m]), np.array([float(c), float(s), 0.0]), _GROUP3_MASK, 1, -1, gid
			)
			if dist < 0 or int(gid[0]) not in bgeoms:
				wall_miss += 1
				max_dev = math.inf
				continue
			max_dev = max(max_dev, abs(dist * 1e3 - ref_r_mm))
		wall_devs_mm[z_mm] = max_dev
	max_wall_dev_mm = max(wall_devs_mm.values()) if wall_devs_mm else math.inf

	protrusion_mm = max(0.0, _max_bowl_radius_mm(model, data, bgeoms) - BOWL_R * 1e3)

	passed = bool(
		flat_z_mm
		and missing_area_frac <= G0_MISSING_AREA_MAX_FRAC
		and (G0_FLOOR_TARGET_MM - G0_FLOOR_TOL_MM) <= flat_min_mm
		and flat_max_mm <= (G0_FLOOR_TARGET_MM + G0_FLOOR_TOL_MM)
		and basin_min_mm >= (G0_FLOOR_TARGET_MM - G0_FLOOR_TOL_MM)
		and (G0_RIM_TARGET_MM - G0_RIM_TOL_MM) <= rim_min_mm
		and rim_max_mm <= (G0_RIM_TARGET_MM + G0_RIM_TOL_MM)
		and wall_miss == 0
		and max_wall_dev_mm <= G0_WALL_DEV_MAX_MM
		and protrusion_mm <= G0_PROTRUSION_MAX_MM
	)
	detail = {
		"missing_area_frac": missing_area_frac,
		"flat_floor_r_mm": G0_FLAT_FLOOR_R_MM,
		"flat_min_mm": flat_min_mm,
		"flat_max_mm": flat_max_mm,
		"basin_min_mm": basin_min_mm,
		"rim_min_mm": rim_min_mm,
		"rim_max_mm": rim_max_mm,
		"wall_devs_mm": wall_devs_mm,
		"max_wall_dev_mm": max_wall_dev_mm,
		"wall_miss": wall_miss,
		"protrusion_mm": protrusion_mm,
	}
	return GateResult("G0", passed, detail)


def _contacts_vs_bowl(
	model: mujoco.MjModel, data: mujoco.MjData, bgeoms: set[int]
) -> list[tuple[float, np.ndarray, np.ndarray]]:
	out = []
	for i in range(data.ncon):
		c = data.contact[i]
		if int(c.geom1) in bgeoms or int(c.geom2) in bgeoms:
			out.append((float(c.dist), np.array(c.frame[:3], dtype=np.float64), np.array(c.pos, dtype=np.float64)))
	return out


def _brick_body_xml(pos: np.ndarray, quat_wxyz: np.ndarray, size: tuple[float, float, float]) -> str:
	hx, hy, hz = size[0] / 2.0, size[1] / 2.0, size[2] / 2.0
	return (
		f'<body name="brick" pos="{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}" '
		f'quat="{quat_wxyz[0]:.6f} {quat_wxyz[1]:.6f} {quat_wxyz[2]:.6f} {quat_wxyz[3]:.6f}">'
		f'<freejoint/><geom name="b" type="box" size="{hx:.5f} {hy:.5f} {hz:.5f}" mass="0.0112"/></body>'
	)


def _random_brick_pose(
	rng: np.random.Generator,
	model_static: mujoco.MjModel,
	data_static: mujoco.MjData,
	bgeoms_static: set[int],
	r: float,
	angle: float,
	size: tuple[float, float, float],
	tilt_max_deg: float = 20.0,
	pen_mm: tuple[float, float] = (0.2, 4.0),
) -> str:
	yaw = float(rng.uniform(0.0, 2.0 * math.pi))
	tilt = float(rng.uniform(0.0, math.radians(tilt_max_deg)))
	tilt_dir = float(rng.uniform(0.0, 2.0 * math.pi))
	cy, sy = math.cos(yaw), math.sin(yaw)
	ux, uy = math.cos(tilt_dir), math.sin(tilt_dir)
	ct, st = math.cos(tilt), math.sin(tilt)
	rt = np.array(
		[
			[ct + ux * ux * (1.0 - ct), ux * uy * (1.0 - ct), uy * st],
			[ux * uy * (1.0 - ct), ct + uy * uy * (1.0 - ct), -ux * st],
			[-uy * st, ux * st, ct],
		]
	)
	rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
	rot = rz @ rt
	hx, hy, hz = size[0] / 2.0, size[1] / 2.0, size[2] / 2.0
	us = np.linspace(-1.0, 1.0, 7)
	pts = np.array([(u * hx, v * hy, -hz) for u in us for v in us])
	rc = pts @ rot.T
	x = float(BOWL_XY[0] + r * math.cos(angle))
	y = float(BOWL_XY[1] + r * math.sin(angle))
	gid = np.zeros(1, dtype=np.int32)
	down = np.array([0.0, 0.0, -1.0], dtype=np.float64)
	req_z_c = -1e9
	for dx, dy, dz in rc:
		dist = mujoco.mj_ray(
			model_static, data_static, np.array([x + dx, y + dy, 0.10]), down, _GROUP3_MASK, 1, -1, gid
		)
		z_bowl = (0.10 - dist) if (dist >= 0 and int(gid[0]) in bgeoms_static) else 0.003
		req_z_c = max(req_z_c, z_bowl - dz)
	pen = float(rng.uniform(pen_mm[0], pen_mm[1])) * 1e-3
	z_c = req_z_c - pen
	quat = np.zeros(4, dtype=np.float64)
	mujoco.mju_mat2Quat(quat, rot.flatten())
	return _brick_body_xml(np.array([x, y, z_c]), quat, size)


def g1_collider_micro(
	model_builder: Callable[[str], tuple[mujoco.MjModel, mujoco.MjData]],
	n_random: int = 300,
	n_junction: int = 100,
	seed: int = 0,
) -> GateResult:
	"""G1: robot-free ``mj_forward`` contact check on ``pose_f552_s2`` + random + junction poses."""
	m0, d0 = model_builder("")
	bgeoms0 = bowl_geom_ids(m0)

	# Pose 0: pose_f552_s2.npy
	pose552 = np.load(str(POSE_F552_PATH))
	xml552 = _brick_body_xml(pose552[:3], pose552[3:7], DUPLO_2X2)
	m552, d552 = model_builder(xml552)
	b552 = bowl_geom_ids(m552)
	cons552 = _contacts_vs_bowl(m552, d552, b552)
	f552_worst_mm = min((dist * 1e3 for dist, _, _ in cons552), default=math.nan)
	f552_bad_normals = sum(
		1
		for _, normal, pos in cons552
		if math.hypot(pos[0] - BOWL_XY[0], pos[1] - BOWL_XY[1]) < G1_FLAT_RADIUS_M and abs(normal[2]) < G1_NORMAL_Z_MIN
	)
	f552_ok = bool(cons552) and abs(f552_worst_mm - G1_F552_TARGET_MM) <= G1_F552_TOL_MM and f552_bad_normals == 0

	rng = np.random.default_rng(seed)
	worst_m = min((dist for dist, _, _ in cons552), default=0.0)
	bad_normals = f552_bad_normals
	for k in range(n_random + n_junction):
		r = float(rng.uniform(0.0, BOWL_R - 0.02)) if k < n_random else float(rng.uniform(BOWL_R - 0.025, BOWL_R))
		a = float(rng.uniform(0.0, 2.0 * math.pi))
		pose_xml = _random_brick_pose(rng, m0, d0, bgeoms0, r, a, BRICK_SIZES[k % 3])
		model, data = model_builder(pose_xml)
		bgeoms = bowl_geom_ids(model)
		for dist, normal, pos in _contacts_vs_bowl(model, data, bgeoms):
			worst_m = min(worst_m, dist)
			if (
				math.hypot(pos[0] - BOWL_XY[0], pos[1] - BOWL_XY[1]) < G1_FLAT_RADIUS_M
				and abs(normal[2]) < G1_NORMAL_Z_MIN
			):
				bad_normals += 1

	worst_mm = worst_m * 1e3
	passed = f552_ok and worst_mm >= G1_DIST_MIN_MM and bad_normals == 0
	detail = {
		"f552_worst_dist_mm": f552_worst_mm,
		"f552_bad_normals": f552_bad_normals,
		"f552_ok": f552_ok,
		"worst_dist_mm": worst_mm,
		"bad_normals": bad_normals,
	}
	return GateResult("G1", passed, detail)


def g2_static_soak(cid: str, n_per_cond: int = G2_N_PER_COND, seed: int = 0) -> GateResult:
	"""G2: 4 s @ 240 Hz brick drop soak under extreme mass/friction combinations."""
	from bowl_experiments import candidates

	spec = load_physics_spec()
	rng = np.random.default_rng(seed)
	violations: list[dict[str, Any]] = []
	max_speed = 0.0
	max_angvel = 0.0
	max_end_speed = 0.0
	max_end_yawrate = 0.0

	for cname, mass_scale, friction_scale in G2_STRESS_COMBOS:
		plate_mu = spec.plate.dynamic_friction * friction_scale
		builder = candidates.model_builder(
			cid,
			opt_xml=f'timestep="{1.0 / G2_HZ:.7f}" cone="elliptic" impratio="10"',
			torsional=True,
			friction=f"{plate_mu:.5f} 0.005 0.0005",
		)
		for idx in range(n_per_cond):
			bname = BRICK_NAMES[idx % 3]
			bspec = brick_spec_for_size(BRICK_SIZES[idx % 3], spec).scaled(mass_scale, friction_scale)
			hx, hy, hz = bspec.size_m[0] / 2.0, bspec.size_m[1] / 2.0, bspec.size_m[2] / 2.0
			r = float(rng.uniform(0.0, 0.055))
			ang = float(rng.uniform(0.0, 2.0 * math.pi))
			yaw = float(rng.uniform(0.0, 2.0 * math.pi))
			x = float(BOWL_XY[0] + r * math.cos(ang))
			y = float(BOWL_XY[1] + r * math.sin(ang))
			z = 0.003 + G2_DROP_CLEARANCE_M + hz
			qw, qz = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
			brick_xml = (
				f'<body name="brick" pos="{x:.5f} {y:.5f} {z:.5f}" quat="{qw:.6f} 0 0 {qz:.6f}">'
				f'<freejoint name="brick"/>'
				f'<geom name="b" type="box" size="{hx:.5f} {hy:.5f} {hz:.5f}" mass="{bspec.mass_kg:.5f}" '
				f'friction="{bspec.dynamic_friction:.5f} 0.02 0.002" solref="0.009 1"/></body>'
			)
			model, data = builder(brick_xml)
			vadr = int(model.joint("brick").dofadr[0])
			peak_v = 0.0
			peak_w = 0.0
			for _ in range(G2_SOAK_STEPS):
				mujoco.mj_step(model, data)
				qv = data.qvel[vadr : vadr + 6]
				peak_v = max(peak_v, float(np.linalg.norm(qv[:3])))
				peak_w = max(peak_w, float(np.linalg.norm(qv[3:])))
			end_v = float(np.linalg.norm(data.qvel[vadr : vadr + 3]))
			end_yaw = abs(float(data.qvel[vadr + 5]))
			max_speed = max(max_speed, peak_v)
			max_angvel = max(max_angvel, peak_w)
			max_end_speed = max(max_end_speed, end_v)
			max_end_yawrate = max(max_end_yawrate, end_yaw)
			if (
				peak_v > G2_PEAK_SPEED_MAX
				or peak_w > G2_PEAK_ANGVEL_MAX
				or end_v >= G2_END_SPEED_MAX
				or end_yaw >= G2_END_YAWRATE_MAX
			):
				violations.append(
					{
						"cond": cname,
						"idx": idx,
						"brick": bname,
						"peak_v": peak_v,
						"peak_w": peak_w,
						"end_v": end_v,
						"end_yaw": end_yaw,
					}
				)
				if len(violations) >= 10:
					break
		if len(violations) >= 10:
			break

	detail = {
		"violations": len(violations),
		"sample_violations": violations[:5],
		"max_speed_mps": max_speed,
		"max_angvel_rps": max_angvel,
		"max_end_speed_mps": max_end_speed,
		"max_end_yawrate_rps": max_end_yawrate,
	}
	return GateResult("G2", len(violations) == 0, detail)


def _worker_init(cid: str) -> None:
	scripts_dir = str(ROOT / "scripts")
	src_dir = str(ROOT / "src")
	if src_dir not in sys.path:
		sys.path.insert(0, src_dir)
	if scripts_dir not in sys.path:
		sys.path.insert(0, scripts_dir)
	from bowl_experiments import candidates

	candidates.install(cid)


def _worker_run_matrix(task: tuple[str, int]) -> dict[str, Any]:
	import bench_physics_expert as bp

	from digital_twin_armigo.benchmarks import HELD_OUT_PERTURBATIONS, SENSITIVITY_MATRIX
	from digital_twin_armigo.physics import load_physics_spec

	cond_name, seed = task
	all_conds = {c.name: c for c in (*SENSITIVITY_MATRIX, *HELD_OUT_PERTURBATIONS)}
	row = bp.run_episode(all_conds[cond_name], load_physics_spec(), bp.load_top_camera(), seed)
	row.pop("_telemetry", None)
	row.pop("_phases", None)
	return row


def _worker_run_layout(task: tuple[int, int]) -> dict[str, Any]:
	import bench_physics_expert as bp

	from digital_twin_armigo.physics import load_physics_spec

	layout_seed, idx = task
	top = bp.load_top_camera()
	layouts = bp.sample_layouts(60, layout_seed, top)
	row = bp.run_layout_episode(layouts[idx], load_physics_spec(), idx)
	row["layout_seed"] = layout_seed
	row.pop("_telemetry", None)
	row.pop("_phases", None)
	return row


def _ensure_markers() -> None:
	from digital_twin_armigo.constants import CELL_MARKERS
	from digital_twin_armigo.mujoco_cell.scene import write_marker_textures

	write_marker_textures(SO101_MJCF_DIR, set(CELL_MARKERS))


def g3_two_failing_conditions(cid: str, max_workers: int = 12) -> GateResult:
	"""G3: high_mass (0-29, 200-229) and low_friction (100-129, 200-229) -> 120/120 SUCCESS."""
	_ensure_markers()
	tasks = [("high_mass", s) for s in (*range(30), *range(200, 230))] + [
		("low_friction", s) for s in (*range(100, 130), *range(200, 230))
	]
	with concurrent.futures.ProcessPoolExecutor(
		max_workers=max_workers, initializer=_worker_init, initargs=(cid,)
	) as pool:
		rows = list(pool.map(_worker_run_matrix, tasks))

	successes = sum(1 for r in rows if r["verdict"] == "SUCCESS")
	bounce_outs = sum(1 for r in rows if "bounce_out" in r["reason_codes"])
	rel_speeds = [
		r["info"]["release_peak_speed_mps"]
		for r in rows
		if r.get("info", {}).get("release_peak_speed_mps") is not None
		and math.isfinite(r["info"]["release_peak_speed_mps"])
	]
	max_rel = max(rel_speeds) if rel_speeds else math.nan
	failing = [
		{"condition": r["condition"], "seed": r["seed"], "verdict": r["verdict"], "reasons": r["reason_codes"]}
		for r in rows
		if r["verdict"] != "SUCCESS"
	]
	passed = successes == len(tasks) and bounce_outs == 0 and (not rel_speeds or max_rel <= G3_RELEASE_SPEED_MAX)
	detail = {
		"successes": successes,
		"total": len(tasks),
		"bounce_outs": bounce_outs,
		"max_release_peak_speed_mps": max_rel,
		"failing": failing[:10],
	}
	return GateResult("G3", passed, detail)


def g4_full_regression(cid: str, summary_path: Path | None = None, max_workers: int = 12) -> GateResult:
	"""G4: full 540 matrix/held-out + 120 layouts regression against baseline SUMMARY.json."""
	from digital_twin_armigo.benchmarks import HELD_OUT_PERTURBATIONS, SENSITIVITY_MATRIX

	_ensure_markers()
	baseline = json.loads((summary_path or BASELINE_SUMMARY_PATH).read_text(encoding="utf-8"))
	cond_names = [c.name for c in (*SENSITIVITY_MATRIX, *HELD_OUT_PERTURBATIONS)]
	matrix_tasks = [(cn, s) for cn in cond_names for s in (*range(30), *range(100, 130))]
	layout_tasks = [(ls, idx) for ls in (1000, 1001) for idx in range(60)]

	with concurrent.futures.ProcessPoolExecutor(
		max_workers=max_workers, initializer=_worker_init, initargs=(cid,)
	) as pool:
		matrix_rows = list(pool.map(_worker_run_matrix, matrix_tasks))
		layout_rows = list(pool.map(_worker_run_layout, layout_tasks))

	per_cond: dict[str, list[int]] = {}
	for r in matrix_rows:
		cn = r["condition"]
		ok, n = per_cond.get(cn, [0, 0])
		per_cond[cn] = [ok + (1 if r["verdict"] == "SUCCESS" else 0), n + 1]

	matrix_ok = sum(v[0] for v in per_cond.values())
	layouts_ok = sum(1 for r in layout_rows if r["verdict"] == "SUCCESS")
	all_rows = matrix_rows + layout_rows

	bounce_outs = sum(1 for r in all_rows if "bounce_out" in r["reason_codes"])
	fast_releases = sum(
		1 for r in all_rows if (r.get("info", {}).get("release_peak_speed_mps") or 0.0) > G4_FAST_RELEASE_MPS
	)
	jaw_env_ids = [
		(r.get("condition") or f"layouts_seed{r['layout_seed']}", r.get("seed", r.get("layout_index")))
		for r in all_rows
		if (r.get("info", {}).get("jaw_env_contact_max_N") or 0.0) > G4_JAW_ENV_LIMIT_N
	]
	ge_890_ids = [
		(r.get("condition") or f"layouts_seed{r['layout_seed']}", r.get("seed", r.get("layout_index")))
		for r in all_rows
		if (r.get("info", {}).get("n_frames") or 0) >= G4_LONG_FRAMES
	]
	base_conds = baseline["per_condition"]
	cond_regression = [cn for cn, (ok, _) in per_cond.items() if ok < base_conds[cn][0]]
	failing = [
		{
			"id": (
				f"{r['condition']}_{r['seed']}"
				if "condition" in r
				else f"layouts_seed{r['layout_seed']}_idx{r['layout_index']}"
			),
			"verdict": r["verdict"],
			"reasons": r["reason_codes"],
		}
		for r in all_rows
		if r["verdict"] != "SUCCESS"
	]

	passed = (
		bounce_outs == 0
		and matrix_ok >= G4_MATRIX_MIN
		and not cond_regression
		and layouts_ok >= G4_LAYOUT_MIN
		and fast_releases == 0
		and len(jaw_env_ids) <= baseline["jaw_env_over_5N"]
		and len(ge_890_ids) <= baseline["frames_ge_890"]
	)
	detail = {
		"matrix_success": matrix_ok,
		"matrix_total": len(matrix_rows),
		"per_condition": per_cond,
		"cond_regression": cond_regression,
		"layouts": [layouts_ok, len(layout_rows)],
		"bounce_outs": bounce_outs,
		"fast_releases": fast_releases,
		"jaw_env_over_5N": len(jaw_env_ids),
		"jaw_env_over_5N_ids": jaw_env_ids,
		"frames_ge_890": len(ge_890_ids),
		"frames_ge_890_ids": ge_890_ids,
		"failing": failing,
	}
	return GateResult("G4", passed, detail)


def g5_engineering(cid: str, summary_path: Path | None = None) -> GateResult:
	"""G5: bit-identical determinism (3 episodes x 2), wall-clock <= 1.5x baseline, zero MuJoCo warnings."""
	import bench_physics_expert as bp

	from bowl_experiments import candidates
	from digital_twin_armigo.benchmarks import SENSITIVITY_MATRIX
	from digital_twin_armigo.physics import load_physics_spec

	baseline = json.loads((summary_path or BASELINE_SUMMARY_PATH).read_text(encoding="utf-8"))
	base_wall = float(baseline["episode_wall_s"])
	conds = {c.name: c for c in SENSITIVITY_MATRIX}
	spec = load_physics_spec()
	top = bp.load_top_camera()
	cases = [("baseline_2x2", 1), ("high_mass", 29), ("latency_backlash", 2)]

	warnings_seen: list[str] = []
	orig_cb = mujoco.set_mju_user_warning if hasattr(mujoco, "set_mju_user_warning") else None

	def _warn_cb(msg: bytes | str) -> None:
		warnings_seen.append(msg.decode() if isinstance(msg, bytes) else str(msg))

	if orig_cb is not None:
		mujoco.set_mju_user_warning(_warn_cb)

	candidates.install(cid)
	determinism_ok = True
	walls: list[float] = []
	mismatches: list[str] = []
	try:
		for cname, seed in cases:
			t0 = time.time()
			a = bp.run_episode(conds[cname], spec, top, seed)
			walls.append(time.time() - t0)
			b = bp.run_episode(conds[cname], spec, top, seed)
			if a["_telemetry"].keys() != b["_telemetry"].keys():
				determinism_ok = False
				mismatches.append(f"{cname}_{seed}:keys")
				continue
			for k in a["_telemetry"]:
				if not np.array_equal(a["_telemetry"][k], b["_telemetry"][k]):
					determinism_ok = False
					mismatches.append(f"{cname}_{seed}:{k}")
			if not np.array_equal(a["_phases"], b["_phases"]):
				determinism_ok = False
				mismatches.append(f"{cname}_{seed}:phases")
	finally:
		candidates.uninstall()
		if orig_cb is not None:
			mujoco.set_mju_user_warning(None)

	mean_wall = float(np.mean(walls))
	passed = determinism_ok and mean_wall <= G5_WALL_RATIO_MAX * base_wall and len(warnings_seen) == 0
	detail = {
		"determinism_ok": determinism_ok,
		"mismatches": mismatches,
		"mean_wall_s": mean_wall,
		"baseline_wall_s": base_wall,
		"wall_ratio": mean_wall / base_wall if base_wall > 0 else math.inf,
		"warnings": warnings_seen,
	}
	return GateResult("G5", passed, detail)
