"""Unit tests for the pure-numpy GripperMonitor state machine."""

from __future__ import annotations

from digital_twin_armigo.mujoco_cell.gripper_state import (
	GripperMonitor,
	GripperMonitorParams,
	GripperState,
)

P = GripperMonitorParams(err_rad=0.05, still_rad_s=0.05, effort_min=0.70)


def feed(mon: GripperMonitor, rows: list[tuple[float, float, float, tuple[float, float]]]):
	return [mon.update(*r) for r in rows]


def test_blocked_needs_consecutive_stalled_frames():
	mon = GripperMonitor(P)
	rows = [(0.06, 0.76, 0.9, (0.0, 0.0))] * (P.blocked_frames - 1)
	assert all(s is not GripperState.BLOCKED for s, _ in feed(mon, rows))
	state, direction = mon.update(0.06, 0.76, 0.9, (0.0, 0.0))
	assert state is GripperState.BLOCKED and direction == -1


def test_clamped_on_brick_is_never_blocked():
	mon = GripperMonitor(P)
	rows = [(0.06, 0.20, 0.9, (3.0, 3.0))] * 20
	assert all(s is GripperState.CLAMPED for s, _ in feed(mon, rows))


def test_normal_closing_ramp_is_moving_not_blocked():
	mon = GripperMonitor(P)
	rows = [(0.85 - 0.053 * k, 0.85 - 0.053 * k + P.err_rad * 1.5, 0.4, (0.0, 0.0)) for k in range(15)]
	assert all(s is not GripperState.BLOCKED for s, _ in feed(mon, rows))


def test_brick_force_on_one_jaw_prevents_blocked():
	mon = GripperMonitor(P)
	rows = [(0.06, 0.40, 0.9, (2.0, 0.0))] * 10  # fixed jaw touches brick: "no clamp", not "blocked"
	assert all(s is not GripperState.BLOCKED for s, _ in feed(mon, rows))


def test_opening_block_reports_positive_direction():
	mon = GripperMonitor(P)
	rows = [(0.85, 0.30, 0.95, (0.0, 0.0))] * P.blocked_frames
	out = feed(mon, rows)
	assert out[-1] == (GripperState.BLOCKED, 1)
