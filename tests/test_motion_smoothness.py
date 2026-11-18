"""Motion smoothness metric, the suite aggregate, and min-jerk moves planned inside the joint range."""

from __future__ import annotations

import numpy as np

from digital_twin_armigo.agent.twin_mcp import aggregate_smoothness
from digital_twin_armigo.constants import FPS, PHASES
from digital_twin_armigo.metrics import motion_smoothness
from digital_twin_armigo.mujoco_cell.physics_expert import MIN_JERK_PEAK_RATIO, ClosedLoopExpert, ExpertParams


def _min_jerk(n):
	s = np.linspace(0.0, 1.0, n)
	return 10 * s**3 - 15 * s**4 + 6 * s**5


def _arm(path):
	return np.repeat(np.asarray(path, dtype=np.float64)[:, None], 6, axis=1)


def test_continuous_move_is_smoother_than_a_clipped_one():
	smooth = _arm(_min_jerk(60))
	clipped = smooth.copy()
	clipped[:, 3] = np.minimum(clipped[:, 3], 0.6)  # a joint stopped dead at a limit halfway through
	a = motion_smoothness(smooth, smooth)
	b = motion_smoothness(clipped, clipped)
	assert b["jerk_peak_meas"] > 10 * a["jerk_peak_meas"]
	assert b["jerk_rms_cmd"] > a["jerk_rms_cmd"]


def test_stops_count_rest_then_move_segments_but_not_the_initial_rest():
	still = np.zeros(10)
	move = _min_jerk(30)
	path = np.concatenate([still, move, np.full(10, 1.0), 1.0 + move, np.full(10, 2.0), 2.0 + move])
	out = motion_smoothness(_arm(path), _arm(path))
	assert out["stops"] == 2
	assert 0.0 < out["still_fraction"] < 1.0


def test_phases_name_the_worst_segment():
	path = np.concatenate([_min_jerk(40), np.minimum(1.0 + _min_jerk(40), 1.5)])
	phases = np.array([PHASES.index("approach")] * 40 + [PHASES.index("return")] * 40)
	out = motion_smoothness(_arm(path), _arm(path), phases)
	assert out["jerk_peak_phase"] == "return"
	assert out["worst_phases"][0]["phase"] == "return"


def test_short_series_reports_only_frames():
	assert motion_smoothness(np.zeros((3, 6)), np.zeros((3, 6))) == {"frames": 3}


def test_aggregate_uses_medians_and_the_worst_peak():
	rows = [
		{
			"jerk_rms_cmd": 1.0,
			"jerk_rms_meas": 2.0,
			"jerk_peak_meas": 10.0,
			"jerk_peak_phase": "lift",
			"stops": 3,
			"still_fraction": 0.1,
			"clipped_frames": 0,
		},
		{
			"jerk_rms_cmd": 3.0,
			"jerk_rms_meas": 4.0,
			"jerk_peak_meas": 90.0,
			"jerk_peak_phase": "return",
			"stops": 5,
			"still_fraction": 0.3,
			"clipped_frames": 7,
		},
		{"frames": 2},
	]
	out = aggregate_smoothness(rows)
	assert out["episodes"] == 2
	assert out["jerk_rms_meas_median"] == 3.0
	assert out["jerk_peak_meas_max"] == 90.0 and out["jerk_peak_phase"] == "return"
	assert out["stops_total"] == 8 and out["clipped_frames_total"] == 7
	assert aggregate_smoothness([{"frames": 1}]) == {}


def _bare_expert(lo, hi, last):
	expert = ClosedLoopExpert.__new__(ClosedLoopExpert)
	expert.p = ExpertParams()
	expert._lo, expert._hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
	expert.last = np.asarray(last, dtype=np.float64)
	return expert


def test_move_plans_to_the_reachable_target_so_no_joint_is_clipped_midway():
	lo, hi = np.full(6, -1.0), np.full(6, 1.0)
	expert = _bare_expert(lo, hi, np.zeros(6))
	goal = np.array([0.5, -1.2, 0.0, -1.3, 0.2, 0.9])  # joints 1 and 3 lie outside the range, like the rest pose
	path = np.array([q for q, phase in expert._move(goal, 1.0, "return")])
	assert np.all(path >= lo - 1e-12) and np.all(path <= hi + 1e-12)
	np.testing.assert_allclose(path[-1], np.clip(goal, lo, hi))
	# Every joint follows one min-jerk profile: no joint reaches its end value before the others.
	progress = (path - path[0]) / np.where(path[-1] == path[0], 1.0, path[-1] - path[0])
	moving = path[-1] != path[0]
	assert np.allclose(progress[:, moving], progress[:, [np.flatnonzero(moving)[0]]])
	jerk = np.abs(np.diff(path[:, :5], 3, axis=0)).max() * FPS**3
	assert jerk < 200.0


def test_return_duration_scales_with_distance_but_stays_one_segment():
	p = ExpertParams()
	lo, hi = np.full(6, -2.0), np.full(6, 2.0)

	def return_frames(distance):
		expert = _bare_expert(lo, hi, np.r_[np.full(5, distance), p.open_g])
		expert.q_start = np.zeros(6)
		return sum(1 for _, phase in expert._go_home() if phase == "return")

	hold = round(p.final_hold_s * FPS)
	assert return_frames(0.5) - hold == round(p.return_s * FPS)  # short returns keep the default duration
	long = return_frames(3.0) - hold
	assert round(p.return_s * FPS) < long <= round(p.max_move_chunk_s * FPS)
	assert MIN_JERK_PEAK_RATIO * 3.0 / (long / FPS) <= p.return_peak_speed_rad_s + 1e-6
