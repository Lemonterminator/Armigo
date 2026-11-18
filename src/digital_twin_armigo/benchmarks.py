"""Cross-engine SO-101 brick manipulation benchmarks and sensitivity matrix definitions."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .constants import FPS, PHASES, REST_U
from .joints import normalized_to_rad


@dataclass(frozen=True)
class MatrixCondition:
	"""One condition row in the pure-physics pick-inspect-place sensitivity matrix."""

	name: str
	brick_size: str = "2x2"
	brick_xy: tuple[float, float] = (0.26, 0.01)
	brick_yaw_deg: float = 20.0
	defective: bool = True
	clutter_per_bowl: int = 0
	mass_scale: float = 1.0
	friction_scale: float = 1.0
	latency_frames: int = 0
	backlash_deg: float = 0.0

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


SENSITIVITY_MATRIX: tuple[MatrixCondition, ...] = (
	MatrixCondition("baseline_2x2", brick_size="2x2", brick_yaw_deg=20.0),
	MatrixCondition("brick_2x3", brick_size="2x3", brick_yaw_deg=35.0),
	MatrixCondition("brick_2x4", brick_size="2x4", brick_yaw_deg=-25.0),
	MatrixCondition("dynamic_clutter", brick_size="2x2", brick_yaw_deg=15.0, clutter_per_bowl=3),
	MatrixCondition("low_friction", brick_size="2x2", brick_yaw_deg=20.0, friction_scale=0.65),
	MatrixCondition("high_mass", brick_size="2x2", brick_yaw_deg=20.0, mass_scale=1.60),
	MatrixCondition("latency_backlash", brick_size="2x2", brick_yaw_deg=20.0, latency_frames=2, backlash_deg=0.5),
)

HELD_OUT_PERTURBATIONS: tuple[MatrixCondition, ...] = (
	MatrixCondition(
		"combined_friction_mass",
		brick_size="2x3",
		brick_yaw_deg=-30.0,
		mass_scale=1.40,
		friction_scale=0.70,
	),
	MatrixCondition(
		"combined_all_nominal_stress",
		brick_size="2x2",
		brick_yaw_deg=25.0,
		clutter_per_bowl=2,
		mass_scale=1.35,
		friction_scale=0.75,
		latency_frames=2,
		backlash_deg=0.5,
	),
)


def wilson_ci(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
	"""95% Wilson score binomial confidence interval ``(low, high)``."""
	if total <= 0:
		return (0.0, 0.0)
	p = successes / float(total)
	denom = 1.0 + (z * z) / float(total)
	center = (p + (z * z) / (2.0 * float(total))) / denom
	half = (z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * float(total))) / float(total))) / denom
	return (max(0.0, center - half), min(1.0, center + half))


def _min_jerk_seg(q0: np.ndarray, q1: np.ndarray, n: int) -> np.ndarray:
	s = np.linspace(0.0, 1.0, max(2, n))
	w = 10 * s**3 - 15 * s**4 + 6 * s**5
	return q0[None, :] + w[:, None] * (q1 - q0)[None, :]


def joint_tracking_trajectory(fps: int = FPS) -> np.ndarray:
	"""120-frame multi-joint excitation trajectory starting from ``REST_U`` (6 joints, radians)."""
	q_rest = normalized_to_rad(np.array(REST_U, dtype=np.float64))
	q_rest[5] = 0.85
	q_mid = np.array([0.35, -0.20, 0.30, 1.15, -0.60, 0.25], dtype=np.float64)
	q_far = np.array([-0.40, 0.10, -0.15, 0.85, 0.90, 1.05], dtype=np.float64)
	seg1 = _min_jerk_seg(q_rest, q_mid, round(1.2 * fps))
	hold1 = np.repeat(q_mid[None, :], round(0.4 * fps), axis=0)
	seg2 = _min_jerk_seg(q_mid, q_far, round(1.2 * fps))
	hold2 = np.repeat(q_far[None, :], round(0.4 * fps), axis=0)
	seg3 = _min_jerk_seg(q_far, q_rest, round(0.8 * fps))
	traj = np.vstack([seg1, hold1, seg2, hold2, seg3])
	if len(traj) < 120:
		traj = np.vstack([traj, np.repeat(traj[-1:], 120 - len(traj), axis=0)])
	return traj[:120]


def _quat_to_yaw_deg(quat_xyzw: np.ndarray) -> float:
	x, y, z, w = (float(v) for v in quat_xyzw)
	siny_cosp = 2.0 * (w * z + x * y)
	cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
	return math.degrees(math.atan2(siny_cosp, cosy_cosp))


def summarize_flick_experiment(
	brick_pos: np.ndarray,
	brick_vel: np.ndarray,
	brick_quat_xyzw: np.ndarray,
	strike_idx: int,
	fps: int = FPS,
	stop_speed_thresh: float = 0.015,
) -> dict[str, Any]:
	"""Quantify brick launch speed, sliding displacement, yaw rotation, and stop time after arm impact."""
	pos = np.asarray(brick_pos, dtype=np.float64)
	vel = np.asarray(brick_vel, dtype=np.float64)
	quat = np.asarray(brick_quat_xyzw, dtype=np.float64)
	speeds = np.linalg.norm(vel[strike_idx:, :2], axis=1)
	peak_speed = float(speeds.max()) if len(speeds) else 0.0

	p0 = pos[max(0, strike_idx - 1), :2]
	p_end = pos[-1, :2]
	disp = float(np.hypot(*(p_end - p0)))
	yaw0 = _quat_to_yaw_deg(quat[max(0, strike_idx - 1)])
	yaw_end = _quat_to_yaw_deg(quat[-1])
	dyaw = (yaw_end - yaw0 + 180.0) % 360.0 - 180.0

	stop_rel = len(speeds) - 1
	consec = 0
	for i, sp in enumerate(speeds):
		if i > 2 and sp < stop_speed_thresh:
			consec += 1
			if consec >= 5:
				stop_rel = i - 4
				break
		else:
			consec = 0

	return {
		"peak_speed_mps": peak_speed,
		"final_displacement_m": disp,
		"final_xy": [float(p_end[0]), float(p_end[1])],
		"final_yaw_deg": float(dyaw),
		"stop_time_s": float(stop_rel / float(fps)),
	}


def summarize_spin_release_experiment(
	brick_pos: np.ndarray,
	brick_vel: np.ndarray,
	brick_quat_xyzw: np.ndarray,
	brick_ang_vel: np.ndarray,
	release_idx: int,
	fps: int = FPS,
	impact_z_thresh: float = 0.018,
) -> dict[str, Any]:
	"""Quantify free-fall flight time, impact bounce peak, spin rate, and resting pose after spin-release."""
	pos = np.asarray(brick_pos, dtype=np.float64)
	vel = np.asarray(brick_vel, dtype=np.float64)
	quat = np.asarray(brick_quat_xyzw, dtype=np.float64)
	ang = np.asarray(brick_ang_vel, dtype=np.float64)

	rel_speed = float(np.linalg.norm(vel[release_idx]))
	rel_ang_vel = float(np.linalg.norm(ang[release_idx]))
	z_post = pos[release_idx:, 2]

	impact_rel = len(z_post) - 1
	for i, z in enumerate(z_post):
		if i > 0 and z <= impact_z_thresh:
			impact_rel = i
			break

	bounce_peak = float(z_post[impact_rel:].max()) if impact_rel < len(z_post) else float(z_post[-1])
	impact_xy = pos[release_idx + impact_rel, :2]
	final_xyz = pos[-1]
	slide_after_impact = float(np.hypot(*(final_xyz[:2] - impact_xy)))

	return {
		"release_xyz": [float(v) for v in pos[release_idx]],
		"release_speed_mps": rel_speed,
		"release_ang_vel_rad_s": rel_ang_vel,
		"flight_time_s": float(impact_rel / float(fps)),
		"first_bounce_peak_z_m": bounce_peak,
		"slide_after_impact_m": slide_after_impact,
		"final_xyz": [float(v) for v in final_xyz],
		"final_yaw_deg": _quat_to_yaw_deg(quat[-1]),
	}


def build_flick_trajectory(
	ik_fn, q_start: np.ndarray, brick_xy: tuple[float, float] = (0.26, 0.0), fps: int = FPS
) -> tuple[np.ndarray, int]:
	"""Build a shared joint trajectory that positions behind ``brick_xy`` and sweeps laterally to flick it."""
	bx, by = brick_xy
	q_ready = np.array([0.0, -0.35, 0.25, 1.35, 0.0, 0.05], dtype=np.float64)
	q_hover = ik_fn(np.array([bx, by - 0.075, 0.08]), q_ready, 0.0)
	q_hover[5] = 0.05
	q_behind = ik_fn(np.array([bx, by - 0.075, 0.018]), q_hover, 0.0)
	q_behind[5] = 0.05
	q_sweep = q_behind.copy()
	q_sweep[0] = q_behind[0] - 0.55  # negative pan delta sweeps +Y across (0.26, 0.0)

	frames = [
		_min_jerk_seg(q_start, q_ready, round(0.5 * fps)),
		_min_jerk_seg(q_ready, q_hover, round(0.8 * fps)),
		_min_jerk_seg(q_hover, q_behind, round(0.6 * fps)),
		np.repeat(q_behind[None, :], round(0.2 * fps), axis=0),
	]
	strike_idx = sum(len(f) for f in frames)
	frames.append(_min_jerk_seg(q_behind, q_sweep, round(0.30 * fps)))
	frames.append(np.repeat(q_sweep[None, :], round(1.5 * fps), axis=0))
	return np.vstack(frames), strike_idx


def build_spin_release_trajectory(
	ik_fn,
	q_start: np.ndarray,
	brick_xy: tuple[float, float] = (0.26, 0.0),
	open_g: float = 0.85,
	hold_g: float = 0.06,
	fps: int = FPS,
) -> tuple[np.ndarray, int, np.ndarray]:
	"""Build a shared pick -> lift -> wind-up -> spin-release joint trajectory and phase labels."""
	bx, by = brick_xy
	q_pre = np.array([0.0, -0.35, 0.25, 1.35, 0.0, open_g], dtype=np.float64)
	q_above = ik_fn(np.array([bx, by, 0.09]), q_pre, 0.0)
	q_above[5] = open_g
	q_desc = ik_fn(np.array([bx, by, 0.014]), q_above, 0.0)
	q_desc[5] = open_g
	q_close = q_desc.copy()
	q_close[5] = hold_g
	q_lift = ik_fn(np.array([bx, by, 0.165]), q_close, 0.0)
	q_lift[5] = hold_g
	q_wind = q_lift.copy()
	q_wind[4] = -1.2
	q_spin_mid = q_lift.copy()
	q_spin_mid[4] = 0.0
	q_spin_end = q_lift.copy()
	q_spin_end[4] = 1.2
	q_spin_end[5] = open_g

	segs: list[tuple[np.ndarray, int]] = [
		(_min_jerk_seg(q_start, q_pre, round(0.5 * fps)), PHASES.index("approach")),
		(_min_jerk_seg(q_pre, q_above, round(0.8 * fps)), PHASES.index("approach")),
		(_min_jerk_seg(q_above, q_desc, round(0.7 * fps)), PHASES.index("approach")),
		(np.repeat(q_desc[None, :], round(0.25 * fps), axis=0), PHASES.index("grasp")),
		(_min_jerk_seg(q_desc, q_close, round(0.8 * fps)), PHASES.index("grasp")),
		(np.repeat(q_close[None, :], round(0.25 * fps), axis=0), PHASES.index("grasp")),
		(_min_jerk_seg(q_close, q_lift, round(0.9 * fps)), PHASES.index("lift")),
		(_min_jerk_seg(q_lift, q_wind, round(0.7 * fps)), PHASES.index("show_move")),
		(_min_jerk_seg(q_wind, q_spin_mid, round(0.35 * fps)), PHASES.index("show_move")),
	]
	release_idx = sum(len(s) for s, _ in segs)
	spin_release = _min_jerk_seg(q_spin_mid, q_spin_end, round(0.35 * fps))
	# Smooth 3-frame jaw release while spinning to avoid instantaneous edge kick
	n_open = min(3, len(spin_release))
	spin_release[:n_open, 5] = np.linspace(hold_g, open_g, n_open)
	spin_release[n_open:, 5] = open_g
	segs.append((spin_release, PHASES.index("place")))
	segs.append((np.repeat(q_spin_end[None, :], round(1.5 * fps), axis=0), PHASES.index("return")))

	traj = np.vstack([s for s, _ in segs])
	phases = np.concatenate([np.full(len(s), p, dtype=np.uint8) for s, p in segs])
	return traj, release_idx, phases
