"""Shared episode recorder parity and the closed-loop pure-physics expert (MuJoCo sorting cell)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from digital_twin_armigo.benchmarks import SENSITIVITY_MATRIX
from digital_twin_armigo.camera import CameraPose
from digital_twin_armigo.constants import PHASES
from digital_twin_armigo.physics import load_physics_spec

ROOT = Path(__file__).resolve().parents[1]
PHYSICS = load_physics_spec()
TOP_CAM = CameraPose(**json.loads((ROOT / "config" / "sim" / "top_camera_joint.json").read_text())["camera"])
CONDS = {c.name: c for c in SENSITIVITY_MATRIX}


def _load_bench():
	if "bench_cross_engine" in sys.modules:
		return sys.modules["bench_cross_engine"]
	spec = importlib.util.spec_from_file_location("bench_cross_engine", ROOT / "scripts" / "bench_cross_engine.py")
	mod = importlib.util.module_from_spec(spec)
	sys.modules["bench_cross_engine"] = mod
	spec.loader.exec_module(mod)
	return mod


# Captured on MuJoCo 4.0.1 (updated after adopting R6_box_floor_wedge_walls bowl collision).
OPEN_LOOP_REFERENCE = (
	(
		"baseline_2x2",
		0,
		"SUCCESS",
		(),
		{
			"max_slip_pos_m": 0.008016529166149533,
			"max_slip_rot_deg": 5.272327190835066,
			"dual_contact_ratio": 0.976027397260274,
			"mean_pinch_force_N": 7.2025412462213865,
			"max_pinch_force_N": 11.548849419701877,
			"pinch_normal_peak_N": 13.222388687473659,
			"pinch_tangential_peak_N": 1.352132374333331,
			"pinch_impulse_Ns": 69.76834169623876,
			"max_tracking_error_rad": 0.5568083051940329,
			"tracking_error_ratio": 0.05695142378559464,
			"max_tracking_lag_duration_s": 0.4,
			"max_stall_duration_s": 1.3,
			"torque_saturation_ratio": 0.3685092127303183,
			"max_unintended_force_N": 0.0,
			"unintended_collision_impulse_Ns": 0.0,
			"final_dist_to_bowl_m": 0.01598654973060319,
			"final_brick_speed_mps": 2.842153472593232e-16,
			"final_brick_ang_vel_rps": 9.703879612542032e-11,
		},
	),
	(
		"baseline_2x2",
		3,
		"FAILURE",
		("grasp_slip", "place_miss"),
		{
			"max_slip_pos_m": 0.19409158347297747,
			"max_slip_rot_deg": 179.9082000132406,
			"dual_contact_ratio": 0.6689895470383276,
			"mean_pinch_force_N": 4.6882148468608955,
			"max_pinch_force_N": 13.480641166136877,
			"pinch_normal_peak_N": 13.393384989542728,
			"pinch_tangential_peak_N": 5.198654346906064,
			"pinch_impulse_Ns": 47.98374218279589,
			"max_tracking_error_rad": 0.5569368185498892,
			"tracking_error_ratio": 0.057432432432432436,
			"max_tracking_lag_duration_s": 0.4,
			"max_stall_duration_s": 1.3,
			"torque_saturation_ratio": 0.38175675675675674,
			"max_unintended_force_N": 0.0,
			"unintended_collision_impulse_Ns": 0.0,
			"final_dist_to_bowl_m": 0.14773074589751667,
			"final_brick_speed_mps": 1.374752657592418e-16,
			"final_brick_ang_vel_rps": 4.573710619163708e-10,
		},
	),
	(
		"brick_2x3",
		1,
		"FAILURE",
		("grasp_slip", "place_miss"),
		{
			"max_slip_pos_m": 0.21746459739777813,
			"max_slip_rot_deg": 113.85309198498267,
			"dual_contact_ratio": 0.6701030927835051,
			"mean_pinch_force_N": 4.633081512510382,
			"max_pinch_force_N": 12.563852241089492,
			"pinch_normal_peak_N": 13.88338983965243,
			"pinch_tangential_peak_N": 5.332758646230095,
			"pinch_impulse_Ns": 47.945986716015085,
			"max_tracking_error_rad": 0.5554048963298792,
			"tracking_error_ratio": 0.05536912751677853,
			"max_tracking_lag_duration_s": 0.4,
			"max_stall_duration_s": 1.3,
			"torque_saturation_ratio": 0.37416107382550334,
			"max_unintended_force_N": 0.0,
			"unintended_collision_impulse_Ns": 0.0,
			"final_dist_to_bowl_m": 0.17887430635299278,
			"final_brick_speed_mps": 3.459538231702451e-11,
			"final_brick_ang_vel_rps": 2.027417727943683e-09,
		},
	),
	(
		"latency_backlash",
		3,
		"FAILURE",
		("grasp_slip",),
		{
			"max_slip_pos_m": 0.09603191915451315,
			"max_slip_rot_deg": 104.75487736639278,
			"dual_contact_ratio": 0.8581314878892734,
			"mean_pinch_force_N": 6.026910382240132,
			"max_pinch_force_N": 10.797336960992693,
			"pinch_normal_peak_N": 10.697147965629632,
			"pinch_tangential_peak_N": 3.708897199913767,
			"pinch_impulse_Ns": 59.71681458943737,
			"max_tracking_error_rad": 0.5209246087771847,
			"tracking_error_ratio": 0.05405405405405406,
			"max_tracking_lag_duration_s": 0.36666666666666664,
			"max_stall_duration_s": 1.3,
			"torque_saturation_ratio": 0.38175675675675674,
			"max_unintended_force_N": 0.0,
			"unintended_collision_impulse_Ns": 0.0,
			"final_dist_to_bowl_m": 0.047643392800519144,
			"final_brick_speed_mps": 8.862695693548982e-06,
			"final_brick_ang_vel_rps": 0.0002500281080478251,
		},
	),
	(
		"dynamic_clutter",
		2,
		"SUCCESS",
		(),
		{
			"max_slip_pos_m": 0.00913831868232905,
			"max_slip_rot_deg": 7.797035364661856,
			"dual_contact_ratio": 0.9756944444444444,
			"mean_pinch_force_N": 7.129510087248068,
			"max_pinch_force_N": 10.984974865402503,
			"pinch_normal_peak_N": 12.471504782902315,
			"pinch_tangential_peak_N": 2.6468914350585155,
			"pinch_impulse_Ns": 68.87253315586108,
			"max_tracking_error_rad": 0.5567771724882659,
			"tracking_error_ratio": 0.05733558178752108,
			"max_tracking_lag_duration_s": 0.4,
			"max_stall_duration_s": 1.3,
			"torque_saturation_ratio": 0.37099494097807756,
			"max_unintended_force_N": 0.0,
			"unintended_collision_impulse_Ns": 0.0,
			"final_dist_to_bowl_m": 0.020523193623034414,
			"final_brick_speed_mps": 0.0006523924874616907,
			"final_brick_ang_vel_rps": 0.055699853506223106,
		},
	),
	(
		"high_mass",
		4,
		"SUCCESS",
		(),
		{
			"max_slip_pos_m": 0.009530388599825558,
			"max_slip_rot_deg": 4.465128939390012,
			"dual_contact_ratio": 0.9761904761904762,
			"mean_pinch_force_N": 7.2575755777176125,
			"max_pinch_force_N": 14.650041589487882,
			"pinch_normal_peak_N": 14.867349891213347,
			"pinch_tangential_peak_N": 1.6269812935811765,
			"pinch_impulse_Ns": 70.84277744995005,
			"max_tracking_error_rad": 0.5560096912853796,
			"tracking_error_ratio": 0.05509181969949917,
			"max_tracking_lag_duration_s": 0.4,
			"max_stall_duration_s": 1.3,
			"torque_saturation_ratio": 0.3656093489148581,
			"max_unintended_force_N": 0.0,
			"unintended_collision_impulse_Ns": 0.0,
			"final_dist_to_bowl_m": 0.008292026414899336,
			"final_brick_speed_mps": 1.1029972226957287e-15,
			"final_brick_ang_vel_rps": 1.4964902094279297e-07,
		},
	),
)


@pytest.mark.parametrize(("name", "seed", "verdict", "reasons", "floats"), OPEN_LOOP_REFERENCE)
@pytest.mark.skipif(mujoco.__version__ != "4.0.1", reason="reference values were captured on MuJoCo 4.0.1")
def test_open_loop_path_is_bit_identical(name, seed, verdict, reasons, floats) -> None:
	bench = _load_bench()
	h = bench._mj_run_matrix_condition(CONDS[name], PHYSICS, TOP_CAM, seed=seed)
	assert h["verdict"] == verdict
	assert tuple(h["reason_codes"]) == reasons
	for key, value in floats.items():
		assert h[key] == value, key


def test_matrix_helpers_are_shared() -> None:
	bench = _load_bench()
	cond_j = bench._mj_jitter_condition(CONDS["baseline_2x2"], 5)
	assert cond_j.brick_xy != CONDS["baseline_2x2"].brick_xy
	assert bench._mj_jitter_condition(CONDS["baseline_2x2"], 0) == CONDS["baseline_2x2"]
	assert bench._mj_non_ideal(CONDS["latency_backlash"], PHYSICS).latency_frames == 2


_EPISODES: dict[tuple[str, int], dict] = {}


def _closed_loop(name: str, seed: int) -> dict:
	key = (name, seed)
	if key not in _EPISODES:
		sys.path.insert(0, str(ROOT / "scripts"))
		_load_bench()
		import bench_physics_expert

		_EPISODES[key] = bench_physics_expert.run_episode(CONDS[name], PHYSICS, TOP_CAM, seed)
	return _EPISODES[key]


def test_jaw_pad_faces_come_from_the_model() -> None:
	info = _closed_loop("baseline_2x2", 1)["info"]
	geom = info["grasp_geometry"]
	# fixed pad on the negative side of the gripper frame, a couple of centimetres from it; moving jaw clear when open
	assert -0.03 < geom["fixed_face_m"] < -0.01
	assert geom["moving_face_m"] > 0.016


def test_closed_loop_episode_is_deterministic() -> None:
	from bench_physics_expert import run_episode

	a = _closed_loop("latency_backlash", 2)
	b = run_episode(CONDS["latency_backlash"], PHYSICS, TOP_CAM, 2)
	assert a["_telemetry"].keys() == b["_telemetry"].keys()
	for k in a["_telemetry"]:
		assert np.array_equal(a["_telemetry"][k], b["_telemetry"][k]), k
	assert np.array_equal(a["_phases"], b["_phases"])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_grasp_is_centred_and_clamped_before_lift(seed: int) -> None:
	info = _closed_loop("baseline_2x2", seed)["info"]
	assert info["grasp_push_mm"] <= 5.0
	assert info["jaw_normal_at_lift_N"] >= 1.0


@pytest.mark.parametrize(("name", "seed"), [("baseline_2x2", 0), ("brick_2x4", 3), ("low_friction", 5)])
def test_release_is_gentle(name: str, seed: int) -> None:
	assert _closed_loop(name, seed)["info"]["release_peak_speed_mps"] <= 1.2


@pytest.mark.parametrize(
	("name", "seed"),
	[("baseline_2x2", 1), ("brick_2x4", 3), ("low_friction", 5), ("dynamic_clutter", 4), ("latency_backlash", 2)],
)
def test_end_to_end_success_under_audited_evaluator(name: str, seed: int) -> None:
	ep = _closed_loop(name, seed)
	assert ep["verdict"] == "SUCCESS", ep["reason_codes"]
	assert ep["health"]["evaluator_version"] == "v2-audited"


def _check_phase_order(ph: np.ndarray) -> None:
	"""Non-decreasing phases, except that show_move and show_hold may alternate."""
	sm, sh = PHASES.index("show_move"), PHASES.index("show_hold")
	rank = np.where(ph == sh, sm, ph).astype(int)  # show_move and show_hold share one rank
	assert np.all(np.diff(rank) >= 0)


@pytest.mark.parametrize(("name", "seed"), [("baseline_2x2", 1), ("latency_backlash", 2)])
def test_phase_labels_follow_the_open_loop_convention(name: str, seed: int) -> None:
	ph = _closed_loop(name, seed)["_phases"]
	assert len(ph) == _closed_loop(name, seed)["info"]["n_frames"] <= 900
	assert ph[0] == PHASES.index("rest") and ph[-1] == PHASES.index("return")
	_check_phase_order(ph)
	# three show poses: show_move into each, show_hold while presenting
	show = ph[np.isin(ph, [PHASES.index("show_move"), PHASES.index("show_hold")])]
	starts = np.flatnonzero(np.diff(np.concatenate([[-1], show.astype(int)])) != 0)
	assert [PHASES[show[i]] for i in starts] == ["show_move", "show_hold"] * 3


@pytest.mark.parametrize(("name", "seed"), [("latency_backlash", 2), ("baseline_2x2", 1)])
def test_show_poses_are_held_as_measured(name: str, seed: int) -> None:
	held = _closed_loop(name, seed)["info"]["show_hold_measured_s"]
	assert len(held) == 3
	assert min(held) >= 0.35, held


def test_expert_acts_only_through_the_joint_target() -> None:
	"""Replaying the recorded targets through the open-loop recorder on a fresh cell reproduces the telemetry."""
	from digital_twin_armigo.mujoco_cell.physics_expert import PhysicsEpisode

	bench = _load_bench()
	name, seed = "latency_backlash", 2
	ep = _closed_loop(name, seed)
	assert ep["info"]["assisted_grasp"] is False  # recorded on the cell the episode actually ran on
	cond_j = bench._mj_jitter_condition(CONDS[name], seed)
	cell = bench._mj_make_cell(PHYSICS, cond_j, TOP_CAM, seed=seed)
	try:
		q_start = sys.modules["bench_physics_expert"]._q_start()
		replay = PhysicsEpisode(
			cell, q_start, cond_j.brick_xy, cond_j.brick_yaw_deg, bench._mj_non_ideal(cond_j, PHYSICS)
		)
		for target in ep["info"]["_targets"]:
			replay.step(target)
		tel = replay.telemetry()
	finally:
		cell.close()
	for k, v in ep["_telemetry"].items():
		assert np.array_equal(v, tel[k]), k


def _forced(params_over: dict) -> dict:
	import dataclasses

	from digital_twin_armigo.mujoco_cell.physics_expert import ExpertParams

	_closed_loop("baseline_2x2", 1)  # makes bench_physics_expert importable
	import bench_physics_expert

	params = dataclasses.replace(ExpertParams(), **params_over)
	return bench_physics_expert.run_episode(CONDS["baseline_2x2"], PHYSICS, TOP_CAM, 1, params)


def test_no_clamp_after_retries_aborts_without_lifting() -> None:
	ep = _forced({"clamp_force_n": 1000.0, "max_grasp_retries": 1})
	info, ph = ep["info"], ep["_phases"]
	assert info["grasp_retries"] == 1
	assert info["aborted"] == "no_clamp"
	assert PHASES.index("lift") not in ph
	assert ph[-1] == PHASES.index("return")
	assert ep["verdict"] == "FAILURE"
	_check_phase_order(ph)


def test_lost_brick_aborts_to_return() -> None:
	# a "clamp" that accepts zero force and a gripper command that never squeezes: the brick stays on the table
	ep = _forced({"clamp_force_n": 0.0, "clamp_still_frames": 1, "hold_g": 0.7})
	info, ph = ep["info"], ep["_phases"]
	assert info["aborted"] == "brick_lost"
	assert PHASES.index("place") not in ph
	assert ph[-1] == PHASES.index("return")
	assert ep["verdict"] == "FAILURE"


def test_frame_cap_is_reported() -> None:
	ep = _forced({"max_frames": 120})
	assert ep["info"]["truncated"] is True
	assert ep["info"]["n_frames"] == 120
	assert _closed_loop("baseline_2x2", 1)["info"]["truncated"] is False


def test_jaw_environment_contact_is_reported() -> None:
	info = _closed_loop("baseline_2x2", 1)["info"]
	assert info["jaw_env_contact_max_N"] == 0.0


def _telemetry_hash(row: dict) -> str:
	import hashlib

	h = hashlib.sha256()
	tel = row["_telemetry"]
	for k in sorted(tel.keys()):
		h.update(k.encode())
		h.update(tel[k].tobytes())
	h.update(b"__phases__")
	h.update(row["_phases"].tobytes())
	return h.hexdigest()


def test_observing_gripper_state_does_not_change_the_trajectory() -> None:
	_load_bench()
	if str(ROOT / "scripts") not in sys.path:
		sys.path.insert(0, str(ROOT / "scripts"))
	import bench_physics_expert

	golden = json.loads((ROOT / "tests" / "data" / "gripper_state_golden.json").read_text(encoding="utf-8"))
	for key, expected in golden.items():
		parts = key.split(":")
		if parts[0] == "layouts":
			seed, idx = int(parts[1]), int(parts[2])
			row = bench_physics_expert.run_layout_episode(
				bench_physics_expert.sample_layouts(60, seed, TOP_CAM)[idx], PHYSICS, idx
			)
		else:
			row = _closed_loop(parts[0], int(parts[1]))
		assert _telemetry_hash(row) == expected, key


def test_idx32_reports_blocked_within_the_closing_window() -> None:
	info = _run_layout_case(1000, 32, {"pre_narrow_open": False})["info"]
	assert info["gripper_first_blocked_frame"] is not None
	assert 95 <= info["gripper_first_blocked_frame"] <= 125
	assert info["gripper_blocked_frames"] > 0
	assert info["gripper_block_direction"] == -1


def test_clean_episode_never_reports_blocked() -> None:
	assert _closed_loop("baseline_2x2", 1)["info"]["gripper_blocked_frames"] == 0


def test_bowl_floor_height_matches_ray_and_constant() -> None:
	from digital_twin_armigo.constants import BOWL_FLOOR_TOP_M
	from digital_twin_armigo.mujoco_cell.physics_expert import bowl_floor_z
	from digital_twin_armigo.mujoco_cell.scene import BOWL_COLLISION_ID

	assert BOWL_COLLISION_ID == "R6_box_floor_wedge_walls"
	bench = _load_bench()
	cell = bench._mj_make_cell(PHYSICS, CONDS["baseline_2x2"], TOP_CAM, seed=0)
	try:
		m, d = cell.model, cell.data
		mujoco.mj_forward(m, d)
		pt = np.array([*cell.layout.pink, 0.2])
		geomid = np.zeros(1, dtype=np.int32)
		gg = np.array([0, 0, 0, 1, 0, 0], dtype=np.uint8)
		dist = mujoco.mj_ray(m, d, pt, np.array([0.0, 0.0, -1.0]), gg, 1, -1, geomid)
		assert abs((0.2 - dist) - BOWL_FLOOR_TOP_M) < 5e-5
		assert abs(bowl_floor_z(m, d, np.array(cell.layout.pink)) - BOWL_FLOOR_TOP_M) < 1e-9
		for bname in ("green_bowl", "pink_bowl"):
			bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, bname)
			col_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == bid and m.geom_contype[g] != 0]
			assert len(col_geoms) > 1
			for g in col_geoms:
				assert np.isclose(m.geom_friction[g, 0], PHYSICS.plate.dynamic_friction)
	finally:
		cell.close()


def _run_layout_case(layout_seed: int, idx: int, params_over: dict | None = None) -> dict:
	import dataclasses

	_load_bench()
	if str(ROOT / "scripts") not in sys.path:
		sys.path.insert(0, str(ROOT / "scripts"))
	import bench_physics_expert

	from digital_twin_armigo.mujoco_cell.physics_expert import ExpertParams

	params = dataclasses.replace(ExpertParams(), **(params_over or {}))
	lay = bench_physics_expert.sample_layouts(60, layout_seed, TOP_CAM)[idx]
	return bench_physics_expert.run_layout_episode(lay, PHYSICS, idx, params)


def test_idx32_recovers_by_narrowing_the_opening() -> None:
	row = _run_layout_case(1000, 32)
	info = row["info"]
	assert row["verdict"] == "SUCCESS", row["reason_codes"]
	assert info["grasp_geometry"]["open_g"] < 0.85
	assert info["jaw_env_contact_max_N"] < 5.0


def test_reactive_block_recovery_narrows_and_succeeds() -> None:
	row = _run_layout_case(1000, 32, {"pre_narrow_open": False, "max_frames": 960})
	info = row["info"]
	assert row["verdict"] == "SUCCESS", row["reason_codes"]
	assert info["block_replans"] >= 1


def test_baseline_keeps_default_open_g() -> None:
	assert _closed_loop("baseline_2x2", 1)["info"]["grasp_geometry"]["open_g"] == 0.85


def test_assess_feasibility_flags_environment_blocked_brick() -> None:
	import dataclasses

	_load_bench()
	if str(ROOT / "scripts") not in sys.path:
		sys.path.insert(0, str(ROOT / "scripts"))
	import bench_physics_expert

	from digital_twin_armigo.constants import SO101_MJCF_DIR
	from digital_twin_armigo.mujoco_cell import Cell
	from digital_twin_armigo.mujoco_cell.physics_expert import ExpertParams, assess_feasibility

	lay = bench_physics_expert.sample_layouts(60, 1000, TOP_CAM)[0]
	# Place the green bowl rim directly overlapping the brick so every (roll, open_g) is blocked
	blocked_lay = dataclasses.replace(lay, green=(lay.brick_xy[0] + 0.095, lay.brick_xy[1]))
	cell = Cell(
		blocked_lay, SO101_MJCF_DIR, physics=PHYSICS, dynamic_clutter=True, assisted_grasp=False, render_enabled=False
	)
	try:
		reason = assess_feasibility(
			cell,
			bench_physics_expert._q_start(),
			blocked_lay.pink,
			ExpertParams(),
			blocked_lay.brick_xy,
			blocked_lay.brick_yaw_deg,
		)
		assert reason == "grasp_blocked_by_environment"
	finally:
		cell.close()


def test_unrecoverable_block_aborts_without_lifting() -> None:
	row = _run_layout_case(1000, 32, {"pre_narrow_open": False, "open_ladder": (0.85,), "max_block_replans": 1})
	assert row["info"]["aborted"] == "gripper_blocked"
	assert PHASES.index("lift") not in row["_phases"]
