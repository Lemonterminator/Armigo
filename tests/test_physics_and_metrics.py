"""Tests for unified physics specifications, actuator non-idealities, and process-health metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from digital_twin_armigo.constants import DUPLO_2X2, DUPLO_2X3, DUPLO_2X4, FPS, PHASES
from digital_twin_armigo.metrics import compare_trajectories, evaluate_episode_health
from digital_twin_armigo.physics import (
	DEFAULT_PHYSICS,
	ActuatorCommandFilter,
	NonIdealityCfg,
	brick_spec_for_size,
	load_physics_spec,
)


def test_unified_brick_specs_cover_all_duplo_sizes_with_consistent_masses():
	s2 = brick_spec_for_size(DUPLO_2X2)
	s3 = brick_spec_for_size(DUPLO_2X3)
	s4 = brick_spec_for_size(DUPLO_2X4)
	assert s2.mass_kg == pytest.approx(0.007, abs=1e-4)
	assert s3.mass_kg == pytest.approx(0.0096, abs=1e-4)
	assert s4.mass_kg == pytest.approx(0.0135, abs=1e-4)
	assert s2.mass_kg < s3.mass_kg < s4.mass_kg
	assert DEFAULT_PHYSICS.version == "v1-nominal-assumed"
	assert DEFAULT_PHYSICS.gripper_actuator.effort_limit < DEFAULT_PHYSICS.arm_actuator.effort_limit


def test_load_physics_spec_from_repo_config_matches_defaults():
	spec = load_physics_spec()
	assert spec.version == DEFAULT_PHYSICS.version
	assert spec.bricks["2x2"].mass_kg == pytest.approx(DEFAULT_PHYSICS.bricks["2x2"].mass_kg)
	assert spec.arm_actuator.stiffness == pytest.approx(DEFAULT_PHYSICS.arm_actuator.stiffness)
	assert spec.gripper_actuator.effort_limit == pytest.approx(DEFAULT_PHYSICS.gripper_actuator.effort_limit)


def test_actuator_command_filter_applies_latency_backlash_and_velocity_limit():
	cfg = NonIdealityCfg(latency_frames=2, backlash_rad=0.02, velocity_limit_rad_s=3.0, encoder_ticks=4096)
	filt = ActuatorCommandFilter(cfg, n_joints=6, fps=30)
	q0 = np.zeros(6)
	filt.reset(q0)
	cmd = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
	# Latency = 2 frames -> first two outputs remain q0
	assert np.allclose(filt.step_command(cmd, q0), q0)
	assert np.allclose(filt.step_command(cmd, q0), q0)
	# Frame 3: delayed command arrives, clamped to 3.0 / 30 = 0.1 rad, minus half-backlash 0.01 = 0.09 rad
	out3 = filt.step_command(cmd, q0)
	assert out3[0] == pytest.approx(0.09, abs=1e-6)

	# Backlash deadband: small reversal inside half-backlash (0.01 rad) holds previous direction
	filt_b = ActuatorCommandFilter(NonIdealityCfg(backlash_rad=0.04), n_joints=1, fps=30)
	filt_b.reset(np.array([0.0]))
	p1 = filt_b.step_command(np.array([0.10]), np.array([0.0]))
	assert p1[0] == pytest.approx(0.08, abs=1e-6)  # lags by half backlash (0.02)
	p2 = filt_b.step_command(np.array([0.09]), np.array([0.08]))
	assert p2[0] == pytest.approx(0.08, abs=1e-6)  # inside deadband [0.06, 0.10]

	# Encoder quantization: 4096 ticks / 2pi
	rng = np.random.default_rng(0)
	obs = filt.observe_joints(np.array([0.12345, 0.0, 0.0, 0.0, 0.0, 0.0]), rng)
	step = 2.0 * math.pi / 4096
	assert abs(obs[0] - round(0.12345 / step) * step) < 1e-9


def _synthetic_episode(drop_during_show: bool = False, bounce_out: bool = False, overload: bool = False):
	n = 120
	phases = np.zeros(n, dtype=np.uint8)
	phases[0:20] = PHASES.index("approach")
	phases[20:40] = PHASES.index("grasp")
	phases[40:60] = PHASES.index("lift")
	phases[60:80] = PHASES.index("show_hold")
	phases[80:105] = PHASES.index("place")
	phases[105:120] = PHASES.index("return")

	q_cmd = np.zeros((n, 6))
	q_act = np.zeros((n, 6))
	q_cmd[:, 5] = 0.85
	q_cmd[20:85, 5] = 0.06  # closed
	# Smooth 10-frame opening ramp in place (frames 85..95): brick releases around frame 88 when q_cmd > 0.18
	q_cmd[85:95, 5] = np.linspace(0.06, 0.85, 10)
	q_act[:, 5] = 0.85
	q_act[20:88, 5] = 0.186
	if overload:
		# Sustained high torque saturation on gripper or arm
		q_cmd[20:100, 1] = 1.0
		q_act[20:100, 1] = 0.0

	tcp = np.tile([0.25, 0.0, 0.10], (n, 1))
	tcp[20:40, 2] = 0.015
	tcp[40:80, 2] = 0.18
	tcp[80:95] = [0.26, -0.18, 0.065]

	brick_pos = np.tile([0.25, 0.0, 0.01], (n, 1))
	brick_pos[25:88] = tcp[25:88]
	if drop_during_show:
		brick_pos[65:] = [0.25, 0.0, 0.01]
	else:
		brick_pos[88:] = [0.26, -0.18, 0.012]
		if bounce_out:
			brick_pos[105:] = [0.45, -0.18, 0.012]

	brick_quat = np.tile([0.0, 0.0, 0.0, 1.0], (n, 1))
	jaw_forces = np.zeros((n, 2))
	jaw_forces[25:88] = [6.5, 6.4]
	if drop_during_show:
		jaw_forces[65:] = 0.0
	unintended = np.zeros(n)
	brick_vel = np.zeros((n, 3))
	brick_ang_vel = np.zeros((n, 3))
	return {
		"q_cmd": q_cmd,
		"q_act": q_act,
		"tcp_pos": tcp,
		"brick_pos": brick_pos,
		"brick_quat": brick_quat,
		"jaw_forces": jaw_forces,
		"unintended_forces": unintended,
		"phases": phases,
		"target_bowl_xy": (0.26, -0.18),
		"bowl_r": 0.10,
		"brick_vel": brick_vel,
		"brick_ang_vel": brick_ang_vel,
		"fps": FPS,
	}


def test_episode_health_detects_ok_drop_bounce_and_tracking_vs_torque_stall():
	ok = evaluate_episode_health(**_synthetic_episode())
	assert ok.success is True
	assert ok.verdict == "SUCCESS"
	assert ok.failure_mode == "ok"
	assert ok.settled_in_bowl is True
	assert ok.max_slip_pos_m == pytest.approx(0.0, abs=1e-6)

	drop = evaluate_episode_health(**_synthetic_episode(drop_during_show=True))
	assert drop.success is False
	assert drop.verdict == "FAILURE"
	assert drop.failure_mode == "show_drop"

	bounce = evaluate_episode_health(**_synthetic_episode(bounce_out=True))
	assert bounce.success is False
	assert bounce.verdict == "FAILURE"
	assert bounce.failure_mode == "bounce_out"

	# Position tracking lag WITHOUT measured torque evidence must NOT be mislabeled as torque stall
	lag = evaluate_episode_health(**_synthetic_episode(overload=True))
	assert lag.success is False
	assert lag.verdict == "FAILURE"
	assert lag.failure_mode == "tracking_lag"
	assert lag.torque_saturation_ratio is None
	assert "torque_stall" not in lag.reason_codes


def test_counterexamples_false_bowl_containment_and_settle_timeout_vs_bounce_out():
	# 1. 2x4 brick whose center is inside bowl_r (0.085 < 0.10) but half overhangs rim (0.085 + 0.0318 = 0.1168 > 0.10)
	ep_overhang = _synthetic_episode()
	ep_overhang["brick_pos"][88:] = [0.26 + 0.085, -0.18, 0.012]
	ep_overhang["brick_size"] = DUPLO_2X4
	res_overhang = evaluate_episode_health(**ep_overhang)
	assert res_overhang.verdict == "FAILURE"
	assert res_overhang.settled_in_bowl is False
	assert "place_miss" in res_overhang.reason_codes or "bounce_out" in res_overhang.reason_codes

	# 2. Brick perched high above the bowl rim (center xy inside radius, z=0.050 above bowl height 0.022)
	ep_high = _synthetic_episode()
	ep_high["brick_pos"][88:] = [0.26 + 0.075, -0.18, 0.050]
	res_high = evaluate_episode_health(**ep_high)
	assert res_high.verdict == "FAILURE"
	assert res_high.settled_in_bowl is False

	# 3. Brick lands inside bowl and stays inside until end, still micro-rocking -> settle_timeout (NOT bounce_out)
	ep_rock = _synthetic_episode()
	ep_rock["brick_vel"][88:, 0] = 0.045  # >= 0.03 m/s, remains inside bowl center
	res_rock = evaluate_episode_health(**ep_rock)
	assert res_rock.verdict == "FAILURE"
	assert "settle_timeout" in res_rock.reason_codes
	assert "bounce_out" not in res_rock.reason_codes


def test_counterexamples_settled_early_then_moved_again_or_spun():
	# Calm for frames 88..112 (0.80 s > 0.50 s), then spins/moves in final frames -> must FAIL
	ep_spin = _synthetic_episode()
	ep_spin["brick_ang_vel"][115:] = [0.0, 0.0, 1.2]  # > 0.35 rad/s at end while linear vel == 0
	res_spin = evaluate_episode_health(**ep_spin)
	assert res_spin.verdict == "FAILURE"
	assert res_spin.settled_in_bowl is False
	assert "settle_timeout" in res_spin.reason_codes

	# Brief burst inside last 0.5 s [105:120], then vel=0 on final frame -1 -> trailing 0.5 s NOT continuous
	ep_bump = _synthetic_episode()
	ep_bump["brick_vel"][112:117, 0] = 0.08
	res_bump = evaluate_episode_health(**ep_bump)
	assert res_bump.verdict == "FAILURE"
	assert res_bump.settled_in_bowl is False

	# Re-grasped inside settle window -> must FAIL
	ep_regrab = _synthetic_episode()
	ep_regrab["jaw_forces"][115:] = [3.0, 3.0]
	res_regrab = evaluate_episode_health(**ep_regrab)
	assert res_regrab.verdict == "FAILURE"
	assert res_regrab.settled_in_bowl is False


def test_counterexamples_missing_telemetry_empty_nan_and_bad_quaternions_return_unknown():
	# 1. Missing angular velocity -> MUST be UNKNOWN (not silently filled with zeros)
	ep_no_ang = _synthetic_episode()
	ep_no_ang["brick_ang_vel"] = None
	r1 = evaluate_episode_health(**ep_no_ang)
	assert r1.verdict == "UNKNOWN"
	assert r1.success is False
	assert "missing_brick_ang_vel" in r1.reason_codes

	# 2. Empty trajectory -> UNKNOWN without IndexError
	ep_empty = _synthetic_episode()
	for k in (
		"q_cmd",
		"q_act",
		"tcp_pos",
		"brick_pos",
		"brick_quat",
		"jaw_forces",
		"unintended_forces",
		"phases",
		"brick_vel",
		"brick_ang_vel",
	):
		ep_empty[k] = ep_empty[k][:0]
	r2 = evaluate_episode_health(**ep_empty)
	assert r2.verdict == "UNKNOWN"
	assert "empty_trajectory" in r2.reason_codes

	# 3. NaN in telemetry -> UNKNOWN
	ep_nan = _synthetic_episode()
	ep_nan["brick_pos"][50, 0] = float("nan")
	r3 = evaluate_episode_health(**ep_nan)
	assert r3.verdict == "UNKNOWN"
	assert "non_finite_telemetry" in r3.reason_codes

	# 4. Zero / non-unit quaternion -> UNKNOWN
	ep_q = _synthetic_episode()
	ep_q["brick_quat"][45] = [0.0, 0.0, 0.0, 0.0]
	r4 = evaluate_episode_health(**ep_q)
	assert r4.verdict == "UNKNOWN"
	assert "invalid_quaternion" in r4.reason_codes

	ep_q2 = _synthetic_episode()
	ep_q2["ee_quat"] = np.tile([0.0, 0.0, 0.0, 3.0], (len(ep_q2["phases"]), 1))
	r5 = evaluate_episode_health(**ep_q2)
	assert r5.verdict == "UNKNOWN"
	assert "invalid_quaternion" in r5.reason_codes


def test_counterexample_early_lift_slip_is_detected():
	# Slips 35 mm right at the start of lift (frames 40..48), then stays shifted through frames 50..88.
	# Old evaluator chose reference at lift midpoint (frame 50) and missed this completely.
	ep = _synthetic_episode()
	ep["brick_pos"][42:88, 2] -= 0.035
	res = evaluate_episode_health(**ep)
	assert res.verdict == "FAILURE"
	assert res.max_slip_pos_m >= 0.034
	assert "grasp_slip" in res.reason_codes


def test_counterexample_substep_collision_even_when_terminal_bowl_succeeds():
	# 30 Hz end-of-frame unintended_forces is 0.0 N, but substep 3 of frame 50 hits 18.0 N
	# while the brick still ends settled in the bowl -> process_ok=False -> verdict=FAILURE.
	ep = _synthetic_episode()
	n = len(ep["phases"])
	sub_unint = np.zeros((n, 8), dtype=np.float64)
	sub_unint[50, 3] = 18.0
	ep["substep_unintended_forces"] = sub_unint
	res = evaluate_episode_health(**ep)
	assert res.terminal_ok is True
	assert res.process_ok is False
	assert res.verdict == "FAILURE"
	assert res.unintended_collision_count >= 1
	assert res.max_unintended_force_N == pytest.approx(18.0)
	assert "unintended_collision" in res.reason_codes


def test_measured_torque_saturation_reported_separately_from_tracking_lag():
	ep = _synthetic_episode()
	n = len(ep["phases"])
	torques = np.zeros((n, 6), dtype=np.float64)
	torques[30:90, 1] = 2.94  # saturated at effort limit for 2.0 s (> 1.5 s)
	limits = np.array([2.94, 2.94, 2.94, 2.94, 2.94, 0.55])
	res = evaluate_episode_health(**ep, actuator_torques=torques, torque_limits=limits, torque_source="solver_measured")
	assert res.verdict == "FAILURE"
	assert res.torque_saturation_ratio is not None
	assert res.torque_saturation_ratio > 0.4
	assert "torque_stall" in res.reason_codes


def test_missing_physics_config_raises_by_default():
	from pathlib import Path

	with pytest.raises(FileNotFoundError):
		load_physics_spec(Path("E:/digital_twin_armigo/config/sim/does_not_exist_12345.json"))


def test_counterexample_compare_trajectories_detects_time_shift_length_mismatch_and_final_vector_diff():
	n_a, n_b = 90, 75
	t_a = np.arange(n_a) / float(FPS)
	t_b = np.arange(n_b) / float(FPS)
	# Trajectory B is delayed by 3 frames (0.10 s) relative to A and ends earlier + at a different XY endpoint
	pos_a = np.zeros((n_a, 3))
	pos_a[:, 0] = np.minimum(t_a * 0.2, 0.25)
	pos_b = np.zeros((n_b, 3))
	pos_b[:, 0] = np.minimum(np.maximum(t_b - 0.10, 0.0) * 0.2, 0.22)
	pos_b[-1, 1] = 0.014  # 14 mm vector diff in Y even if scalar x disp close

	cmp = compare_trajectories(
		{"t": t_a, "brick_pos": pos_a},
		{"t": t_b, "brick_pos": pos_b},
		fps=FPS,
	)
	assert cmp["length_mismatch"] is True
	assert cmp["len_a"] == 90
	assert cmp["len_b"] == 75
	assert cmp["duration_diff_s"] == pytest.approx(0.5, abs=1e-4)
	assert cmp["brick_final_xy_diff_m"] > 0.03
	assert cmp["estimated_time_shift_s"] == pytest.approx(-0.10, abs=0.04) or cmp[
		"estimated_time_shift_s"
	] == pytest.approx(0.10, abs=0.04)


def test_episode_health_measures_slip_in_gripper_frame_during_presentation_rotation():
	ep = _synthetic_episode()
	n = len(ep["phases"])
	# Rotate gripper + brick together by 90 deg around X during show_hold (frames 60..80)
	q_rot = np.array([math.sin(math.pi / 4.0), 0.0, 0.0, math.cos(math.pi / 4.0)])  # xyzw
	ee_quat = np.tile([0.0, 0.0, 0.0, 1.0], (n, 1))
	ee_quat[60:80] = q_rot
	ep["brick_quat"][60:80] = q_rot
	# Brick offset 0.02 m from TCP in gripper local Z -> rotates with gripper
	ep["brick_pos"][25:60, 2] += 0.02
	ep["brick_pos"][60:80, 1] -= 0.02  # rotated +Z by +90 deg around X -> -Y
	ep["brick_pos"][80:90, 2] += 0.02
	ep["ee_quat"] = ee_quat

	res = evaluate_episode_health(**ep)
	assert res.success is True
	assert res.max_slip_pos_m == pytest.approx(0.0, abs=1e-6)
	assert res.max_slip_rot_deg == pytest.approx(0.0, abs=1e-4)


def test_compare_trajectories_computes_exact_rmse_and_max_errors():
	t = np.linspace(0.0, 1.0, 31)
	a = {"qpos": np.column_stack([t, np.zeros((31, 5))]), "brick_pos": np.column_stack([t, np.zeros((31, 2))])}
	b = {
		"qpos": np.column_stack([t + 0.01, np.zeros((31, 5))]),
		"brick_pos": np.column_stack([t + 0.005, np.zeros((31, 2))]),
	}
	cmp = compare_trajectories(a, b)
	assert cmp["qpos_rmse_rad"] == pytest.approx(0.01 / math.sqrt(6), abs=1e-6)
	assert cmp["qpos_max_rad"] == pytest.approx(0.01, abs=1e-6)
	assert cmp["brick_pos_rmse_m"] == pytest.approx(0.005, abs=1e-6)
	assert cmp["brick_pos_max_m"] == pytest.approx(0.005, abs=1e-6)
