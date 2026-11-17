"""Tests for cross-engine benchmark trajectory generators and summary evaluators."""

from __future__ import annotations

import math

import numpy as np
import pytest

from digital_twin_armigo.benchmarks import (
	HELD_OUT_PERTURBATIONS,
	SENSITIVITY_MATRIX,
	joint_tracking_trajectory,
	summarize_flick_experiment,
	summarize_spin_release_experiment,
	wilson_ci,
)
from digital_twin_armigo.constants import FPS


def test_joint_tracking_trajectory_is_bounded_and_smooth():
	q_traj = joint_tracking_trajectory(fps=FPS)
	assert q_traj.ndim == 2 and q_traj.shape[1] == 6
	assert len(q_traj) == 120
	# All joints within conservative physical limits
	assert np.all(np.abs(q_traj[:, :5]) < 2.9)
	assert np.all((q_traj[:, 5] >= -0.2) & (q_traj[:, 5] <= 1.8))


def test_summarize_flick_experiment_extracts_displacement_and_stop_time():
	n = 60
	pos = np.zeros((n, 3))
	vel = np.zeros((n, 3))
	quat = np.tile([0.0, 0.0, 0.0, 1.0], (n, 1))
	# Brick rests at (0.26, 0.0, 0.0096) until frame 15, then slides +0.18 m in Y and stops at frame 35
	pos[:, 0] = 0.26
	pos[:, 2] = 0.0096
	for k in range(15, 36):
		s = (k - 15) / 20.0
		pos[k, 1] = 0.18 * (1.0 - (1.0 - s) ** 2)
		vel[k, 1] = 0.54 * (1.0 - s)
	pos[36:, 1] = 0.18
	yaw_end = math.radians(35.0)
	quat[36:] = [0.0, 0.0, math.sin(yaw_end / 2.0), math.cos(yaw_end / 2.0)]

	res = summarize_flick_experiment(pos, vel, quat, strike_idx=15, fps=FPS)
	assert res["peak_speed_mps"] == pytest.approx(0.54, abs=1e-4)
	assert res["final_displacement_m"] == pytest.approx(0.18, abs=1e-4)
	assert res["final_yaw_deg"] == pytest.approx(35.0, abs=0.1)
	assert res["stop_time_s"] == pytest.approx(21 / FPS, abs=0.05)


def test_summarize_spin_release_experiment_extracts_flight_and_bounce():
	n = 60
	pos = np.zeros((n, 3))
	vel = np.zeros((n, 3))
	ang_vel = np.zeros((n, 3))
	quat = np.tile([0.0, 0.0, 0.0, 1.0], (n, 1))

	pos[:10, :] = [0.26, 0.0, 0.16]
	ang_vel[10, :] = [0.0, 4.5, 0.0]
	vel[10, :] = [0.05, 0.0, 0.0]
	# Fall 10..15, impact z=0.0096 at frame 15, bounce to z=0.032 at frame 18, settle at frame 22..59
	pos[10:16, 2] = np.linspace(0.16, 0.0096, 6)
	pos[16:22, 2] = [0.018, 0.028, 0.032, 0.024, 0.014, 0.0096]
	pos[22:, 2] = 0.0096
	pos[15:, 0] = np.linspace(0.26, 0.30, n - 15)

	res = summarize_spin_release_experiment(pos, vel, quat, ang_vel, release_idx=10, fps=FPS)
	assert res["release_speed_mps"] == pytest.approx(0.05, abs=1e-4)
	assert res["release_ang_vel_rad_s"] == pytest.approx(4.5, abs=1e-4)
	assert res["flight_time_s"] == pytest.approx(5 / FPS, abs=1e-4)
	assert res["first_bounce_peak_z_m"] == pytest.approx(0.032, abs=1e-4)
	assert res["slide_after_impact_m"] == pytest.approx(0.04, abs=1e-4)


def test_sensitivity_matrix_covers_required_variants():
	names = {c.name for c in SENSITIVITY_MATRIX}
	assert {
		"baseline_2x2",
		"brick_2x3",
		"brick_2x4",
		"dynamic_clutter",
		"low_friction",
		"high_mass",
		"latency_backlash",
	} <= names
	assert len(HELD_OUT_PERTURBATIONS) >= 2
	lo, hi = wilson_ci(27, 30)
	assert 0.70 < lo < 0.90 < hi < 1.0
