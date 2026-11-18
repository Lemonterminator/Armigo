"""Motion smoothness metric and the suite aggregate."""

from __future__ import annotations

import numpy as np

from digital_twin_armigo.agent.twin_mcp import aggregate_smoothness
from digital_twin_armigo.constants import PHASES
from digital_twin_armigo.metrics import motion_smoothness


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
