"""Audited process-health telemetry, verdict contract, substep contact aggregation, and trajectory comparison."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

from .constants import BOWL_H, DUPLO_2X2, FPS, PHASES

EVALUATOR_VERSION = "v2-audited"
Verdict = Literal["SUCCESS", "FAILURE", "UNKNOWN"]


@dataclass(frozen=True)
class SubstepContactSummary:
	"""Per-control-frame normal, tangential, total peak forces (N), and impulses (N*s) aggregated over substeps."""

	peak_normal_N: np.ndarray
	peak_tangential_N: np.ndarray
	peak_total_N: np.ndarray
	normal_impulse_Ns: np.ndarray
	tangential_impulse_Ns: np.ndarray
	total_impulse_Ns: np.ndarray


def aggregate_substep_contacts(
	normal_forces: np.ndarray,
	tangential_forces: np.ndarray,
	substep_dt_s: float,
) -> SubstepContactSummary:
	"""Aggregate ``(T, S, ...)`` substep force magnitudes into per-frame peaks and impulses."""
	fn = np.asarray(normal_forces, dtype=np.float64)
	ft = np.asarray(tangential_forces, dtype=np.float64)
	ftot = np.hypot(fn, ft)
	dt = float(substep_dt_s)
	return SubstepContactSummary(
		peak_normal_N=fn.max(axis=1),
		peak_tangential_N=ft.max(axis=1),
		peak_total_N=ftot.max(axis=1),
		normal_impulse_Ns=fn.sum(axis=1) * dt,
		tangential_impulse_Ns=ft.sum(axis=1) * dt,
		total_impulse_Ns=ftot.sum(axis=1) * dt,
	)


@dataclass(frozen=True)
class EpisodeHealthMetrics:
	"""Versioned episode verdict with process/terminal separation, multi-reason codes, and traceable evidence."""

	verdict: Verdict
	reason_codes: tuple[str, ...]
	evidence_refs: dict[str, Any]
	evaluator_version: str
	terminal_ok: bool
	process_ok: bool
	success: bool
	failure_mode: str
	grasped_ok: bool
	max_slip_pos_m: float
	max_slip_rot_deg: float
	dual_contact_ratio: float
	mean_pinch_force_N: float
	max_pinch_force_N: float
	pinch_normal_peak_N: float
	pinch_tangential_peak_N: float
	pinch_impulse_Ns: float
	max_tracking_error_rad: float
	tracking_error_ratio: float
	max_tracking_lag_duration_s: float
	max_stall_duration_s: float
	torque_saturation_ratio: float | None
	torque_evidence_source: str
	unintended_collision_count: int
	max_unintended_force_N: float
	unintended_collision_impulse_Ns: float
	settled_in_bowl: bool
	settle_consecutive_frames: int
	final_dist_to_bowl_m: float
	final_brick_speed_mps: float
	final_brick_ang_vel_rps: float

	def to_dict(self) -> dict[str, Any]:
		out = asdict(self)
		out["reason_codes"] = list(self.reason_codes)
		return out


def _quat_angle_deg(q1: np.ndarray, q2: np.ndarray) -> float:
	dot = float(np.clip(abs(np.dot(q1, q2)), 0.0, 1.0))
	return math.degrees(2.0 * math.acos(dot))


def _quat_conj(q: np.ndarray) -> np.ndarray:
	return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
	ax, ay, az, aw = a
	bx, by, bz, bw = b
	return np.array(
		[
			aw * bx + ax * bw + ay * bz - az * by,
			aw * by - ax * bz + ay * bw + az * bx,
			aw * bz + ax * by - ay * bx + az * bw,
			aw * bw - ax * bx - ay * by - az * bz,
		],
		dtype=np.float64,
	)


def _quat_rotate_inv(q_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
	"""Rotate 3-vector ``v`` by inverse of quaternion ``q_xyzw``."""
	qv = np.array([v[0], v[1], v[2], 0.0], dtype=np.float64)
	res = _quat_mul(_quat_mul(_quat_conj(q_xyzw), qv), q_xyzw)
	return res[:3]


def _quat_to_rotmat(q_xyzw: np.ndarray) -> np.ndarray:
	x, y, z, w = (float(v) for v in q_xyzw)
	return np.array(
		[
			[1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
			[2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
			[2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
		],
		dtype=np.float64,
	)


def _quats_valid(quats: np.ndarray, tol: float = 1e-2) -> bool:
	if quats.ndim != 2 or quats.shape[1] != 4:
		return False
	if not np.all(np.isfinite(quats)):
		return False
	norms = np.linalg.norm(quats, axis=1)
	return bool(np.all(np.abs(norms - 1.0) <= tol))


def _brick_contained_in_bowl(
	pos: np.ndarray,
	quat_xyzw: np.ndarray,
	brick_size: tuple[float, float, float],
	bowl_xy: np.ndarray,
	bowl_r: float,
	bowl_h: float,
) -> bool:
	"""Verify all 8 oriented brick box corners lie inside the bowl radius and cavity height."""
	hx, hy, hz = 0.5 * brick_size[0], 0.5 * brick_size[1], 0.5 * brick_size[2]
	corners_local = np.array(
		[(sx * hx, sy * hy, sz * hz) for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)],
		dtype=np.float64,
	)
	rot = _quat_to_rotmat(quat_xyzw)
	corners_w = pos[None, :] + corners_local @ rot.T
	radii = np.hypot(corners_w[:, 0] - bowl_xy[0], corners_w[:, 1] - bowl_xy[1])
	min_z = float(corners_w[:, 2].min())
	max_z = float(corners_w[:, 2].max())
	max_dim = max(brick_size)
	return bool(
		float(radii.max()) <= bowl_r + 1e-4 and -0.004 <= min_z <= bowl_h + 0.003 and max_z <= bowl_h + max_dim + 0.015
	)


def _unknown_result(reasons: list[str], evidence: dict[str, Any]) -> EpisodeHealthMetrics:
	return EpisodeHealthMetrics(
		verdict="UNKNOWN",
		reason_codes=tuple(dict.fromkeys(reasons)),
		evidence_refs=evidence,
		evaluator_version=EVALUATOR_VERSION,
		terminal_ok=False,
		process_ok=False,
		success=False,
		failure_mode="unknown",
		grasped_ok=False,
		max_slip_pos_m=float("nan"),
		max_slip_rot_deg=float("nan"),
		dual_contact_ratio=0.0,
		mean_pinch_force_N=0.0,
		max_pinch_force_N=0.0,
		pinch_normal_peak_N=0.0,
		pinch_tangential_peak_N=0.0,
		pinch_impulse_Ns=0.0,
		max_tracking_error_rad=float("nan"),
		tracking_error_ratio=0.0,
		max_tracking_lag_duration_s=0.0,
		max_stall_duration_s=0.0,
		torque_saturation_ratio=None,
		torque_evidence_source="none",
		unintended_collision_count=0,
		max_unintended_force_N=0.0,
		unintended_collision_impulse_Ns=0.0,
		settled_in_bowl=False,
		settle_consecutive_frames=0,
		final_dist_to_bowl_m=float("nan"),
		final_brick_speed_mps=float("nan"),
		final_brick_ang_vel_rps=float("nan"),
	)


def evaluate_episode_health(
	q_cmd: np.ndarray,
	q_act: np.ndarray,
	tcp_pos: np.ndarray,
	brick_pos: np.ndarray,
	brick_quat: np.ndarray,
	jaw_forces: np.ndarray,
	unintended_forces: np.ndarray,
	phases: np.ndarray,
	target_bowl_xy: tuple[float, float],
	bowl_r: float,
	brick_vel: np.ndarray | None = None,
	brick_ang_vel: np.ndarray | None = None,
	ee_quat: np.ndarray | None = None,
	fps: int = FPS,
	brick_size: tuple[float, float, float] = DUPLO_2X2,
	bowl_h: float = BOWL_H,
	slip_pos_limit_m: float = 0.025,
	slip_rot_limit_deg: float = 25.0,
	stall_err_rad: float = 0.35,
	max_stall_secs: float = 1.5,
	settle_frames_required: int | None = None,
	settle_window_s: float = 0.5,
	post_release_observe_max_s: float = 3.0,
	max_settle_speed_mps: float = 0.03,
	max_settle_ang_vel_rad_s: float = 0.35,
	unintended_force_limit_N: float = 3.0,
	substep_unintended_forces: np.ndarray | None = None,
	pinch_normal_forces: np.ndarray | None = None,
	pinch_tangential_forces: np.ndarray | None = None,
	actuator_torques: np.ndarray | None = None,
	torque_limits: np.ndarray | None = None,
	torque_source: str = "none",
	timestamps: np.ndarray | None = None,
) -> EpisodeHealthMetrics:
	"""Evaluate physical grip stability, contact integrity, tracking/torque limits, and bowl containment."""
	unknown_reasons: list[str] = []
	evidence: dict[str, Any] = {}

	phases_arr = np.asarray(phases)
	n = len(phases_arr)
	if n == 0:
		return _unknown_result(["empty_trajectory"], {"n_frames": 0})

	q_cmd_a = np.asarray(q_cmd, dtype=np.float64)
	q_act_a = np.asarray(q_act, dtype=np.float64)
	tcp_a = np.asarray(tcp_pos, dtype=np.float64)
	bpos_a = np.asarray(brick_pos, dtype=np.float64)
	bquat_a = np.asarray(brick_quat, dtype=np.float64)
	jaw_a = np.asarray(jaw_forces, dtype=np.float64)
	unint_a = np.asarray(unintended_forces, dtype=np.float64)

	if brick_vel is None:
		unknown_reasons.append("missing_brick_vel")
	if brick_ang_vel is None:
		unknown_reasons.append("missing_brick_ang_vel")

	req_arrays: list[tuple[str, np.ndarray, tuple[int, ...]]] = [
		("q_cmd", q_cmd_a, (n, 6)),
		("q_act", q_act_a, (n, 6)),
		("tcp_pos", tcp_a, (n, 3)),
		("brick_pos", bpos_a, (n, 3)),
		("brick_quat", bquat_a, (n, 4)),
		("jaw_forces", jaw_a, (n, 2)),
		("unintended_forces", unint_a, (n,)),
	]
	if brick_vel is not None:
		bvel_a = np.asarray(brick_vel, dtype=np.float64)
		req_arrays.append(("brick_vel", bvel_a, (n, 3)))
	else:
		bvel_a = np.empty((0, 3))
	if brick_ang_vel is not None:
		bang_a = np.asarray(brick_ang_vel, dtype=np.float64)
		req_arrays.append(("brick_ang_vel", bang_a, (n, 3)))
	else:
		bang_a = np.empty((0, 3))

	for name, arr, expected_shape in req_arrays:
		if arr.shape != expected_shape:
			unknown_reasons.append("corrupted_time_series")
			evidence["shape_error"] = {"field": name, "got": list(arr.shape), "expected": list(expected_shape)}
			break
		if not np.all(np.isfinite(arr)):
			unknown_reasons.append("non_finite_telemetry")
			bad_idx = int(np.argwhere(~np.isfinite(arr))[0, 0])
			evidence.setdefault("non_finite", {})[name] = bad_idx

	if bquat_a.shape == (n, 4) and np.all(np.isfinite(bquat_a)) and not _quats_valid(bquat_a):
		unknown_reasons.append("invalid_quaternion")
		norms = np.linalg.norm(bquat_a, axis=1)
		bad_q = int(np.argmax(np.abs(norms - 1.0)))
		evidence["invalid_brick_quat_frame"] = bad_q

	ee_quat_a: np.ndarray | None = None
	if ee_quat is not None:
		ee_quat_a = np.asarray(ee_quat, dtype=np.float64)
		if ee_quat_a.shape != (n, 4):
			unknown_reasons.append("corrupted_time_series")
		elif not np.all(np.isfinite(ee_quat_a)):
			unknown_reasons.append("non_finite_telemetry")
		elif not _quats_valid(ee_quat_a):
			unknown_reasons.append("invalid_quaternion")
			norms = np.linalg.norm(ee_quat_a, axis=1)
			evidence["invalid_ee_quat_frame"] = int(np.argmax(np.abs(norms - 1.0)))
		else:
			ee_quat_a = ee_quat_a / np.linalg.norm(ee_quat_a, axis=1, keepdims=True)

	if timestamps is not None:
		ts_a = np.asarray(timestamps, dtype=np.float64)
		if ts_a.shape != (n,) or not np.all(np.isfinite(ts_a)) or (n > 1 and np.any(np.diff(ts_a) <= 0.0)):
			unknown_reasons.append("corrupted_timestamps")

	if unknown_reasons:
		return _unknown_result(unknown_reasons, evidence)

	bquat_a = bquat_a / np.linalg.norm(bquat_a, axis=1, keepdims=True)

	p_grasp = PHASES.index("grasp")
	p_lift = PHASES.index("lift")
	p_show_move = PHASES.index("show_move")
	p_show_hold = PHASES.index("show_hold")
	p_handoff = PHASES.index("handoff")
	p_place = PHASES.index("place")
	p_return = PHASES.index("return")

	hold_mask = np.isin(phases_arr, [p_lift, p_show_move, p_show_hold, p_handoff])
	place_idx = np.flatnonzero(phases_arr == p_place)
	release_start = int(place_idx[-1]) + 1 if len(place_idx) else n
	if len(place_idx):
		hold_baseline = float(np.min(q_cmd_a[place_idx, 5]))
		opening = place_idx[q_cmd_a[place_idx, 5] > hold_baseline + 0.02]
		if len(opening):
			release_start = int(opening[0])
	closed_place = (phases_arr == p_place) & (np.arange(n) < release_start)
	carry_mask = hold_mask | closed_place
	carry_idx = np.flatnonzero(carry_mask)

	# 1. Dual-jaw contact & substep normal/tangential/impulse pinch forces
	dual = (jaw_a[:, 0] > 0.02) & (jaw_a[:, 1] > 0.02)
	if len(carry_idx) > 0:
		dual_ratio = float(dual[carry_idx].mean())
		pinch = 0.5 * (jaw_a[carry_idx, 0] + jaw_a[carry_idx, 1])
		mean_pinch = float(pinch.mean())
		max_pinch = float(pinch.max())
	else:
		dual_ratio = 0.0
		mean_pinch = 0.0
		max_pinch = 0.0

	dt_frame = 1.0 / float(fps)
	if pinch_normal_forces is not None:
		pn_arr = np.asarray(pinch_normal_forces, dtype=np.float64)
		if pn_arr.ndim == 2 and len(pn_arr) == n:
			pinch_normal_peak = float(pn_arr[carry_idx].max()) if len(carry_idx) else float(pn_arr.max())
			sub_dt = dt_frame / float(pn_arr.shape[1])
			pinch_impulse = float(pn_arr[carry_idx].sum() * sub_dt) if len(carry_idx) else 0.0
		else:
			pinch_normal_peak = float(pn_arr[carry_idx].max()) if len(carry_idx) else 0.0
			pinch_impulse = float(pn_arr[carry_idx].sum() * dt_frame) if len(carry_idx) else 0.0
	else:
		pinch_normal_peak = max_pinch
		pinch_impulse = float(pinch.sum() * dt_frame) if len(carry_idx) else 0.0

	if pinch_tangential_forces is not None:
		pt_arr = np.asarray(pinch_tangential_forces, dtype=np.float64)
		pinch_tangential_peak = float(pt_arr[carry_idx].max()) if len(carry_idx) else float(pt_arr.max())
	else:
		pinch_tangential_peak = 0.0

	# 2. Early & full-carry slip in gripper frame (measured from lift onset, NOT midpoint!)
	def _rel_pose(i: int) -> tuple[np.ndarray, np.ndarray]:
		dp = bpos_a[i] - tcp_a[i]
		if ee_quat_a is not None:
			qe = ee_quat_a[i]
			return _quat_rotate_inv(qe, dp), _quat_mul(_quat_conj(qe), bquat_a[i])
		return dp, bquat_a[i]

	lift_idx = np.flatnonzero(phases_arr == p_lift)
	max_slip_m = 0.0
	max_slip_deg = 0.0
	grasped_ok = False
	ref_idx: int | None = None
	slip_peak_frame: int | None = None
	first_slip_frame: int | None = None
	if len(lift_idx) > 0 and len(carry_idx) > 0:
		first_lift = int(lift_idx[0])
		# Check missing ee_quat when wrist flex/roll rotates significantly during carry
		if ee_quat_a is None:
			wrist_span = np.ptp(q_act_a[carry_idx, 3:5], axis=0)
			if np.any(wrist_span > 0.25):
				return _unknown_result(["missing_ee_quat"], {"wrist_span_rad": wrist_span.tolist()})

		if first_lift > 0 and phases_arr[first_lift - 1] == p_grasp and dual[first_lift - 1]:
			ref_idx = first_lift - 1
		else:
			dual_in_lift = [int(i) for i in lift_idx if dual[int(i)]]
			ref_idx = dual_in_lift[0] if dual_in_lift else first_lift

		rel0, q0 = _rel_pose(ref_idx)
		grasped_ok = bool(np.linalg.norm(rel0) < 0.04 and dual[ref_idx])
		# Verify brick actually lifted off table with the gripper during carry
		if grasped_ok and float(bpos_a[carry_idx, 2].max()) < 0.035 and float(tcp_a[carry_idx, 2].max()) > 0.06:
			grasped_ok = False
		if grasped_ok:
			for idx_val in carry_idx:
				i = int(idx_val)
				if i < ref_idx:
					continue
				ri, qi = _rel_pose(i)
				sp = float(np.linalg.norm(ri - rel0))
				sr = _quat_angle_deg(qi, q0)
				if first_slip_frame is None and (sp > slip_pos_limit_m or sr > slip_rot_limit_deg):
					first_slip_frame = i
				if sp > max_slip_m:
					max_slip_m = sp
					slip_peak_frame = i
				if sr > max_slip_deg:
					max_slip_deg = sr
					if slip_peak_frame is None:
						slip_peak_frame = i

	# 3. Separate position tracking lag from measured actuator torque saturation
	arm_err = np.abs(q_cmd_a[:, :5] - q_act_a[:, :5]).max(axis=1)
	max_track_err = float(arm_err.max())
	lag_flags = arm_err > stall_err_rad
	track_err_ratio = float(lag_flags.mean())
	cur_run = 0
	max_lag_run = 0
	for flag in lag_flags:
		cur_run = cur_run + 1 if flag else 0
		max_lag_run = max(max_lag_run, cur_run)
	max_lag_s = max_lag_run / float(fps)

	torque_sat_ratio: float | None = None
	max_torque_overload_s: float = 0.0
	t_source = str(torque_source)
	if actuator_torques is not None and torque_limits is not None:
		tq_a = np.asarray(actuator_torques, dtype=np.float64)
		lim_a = np.asarray(torque_limits, dtype=np.float64)
		if tq_a.shape != (n, 6) or lim_a.shape != (6,) or not np.all(np.isfinite(tq_a)) or np.any(lim_a <= 0):
			return _unknown_result(["invalid_torque_telemetry"], {"torque_shape": list(tq_a.shape)})
		sat_t = (np.abs(tq_a[:, :5]) >= 0.95 * lim_a[:5][None, :]).any(axis=1)
		torque_sat_ratio = float(sat_t.mean())
		c_run = 0
		m_run = 0
		for fl in sat_t:
			c_run = c_run + 1 if fl else 0
			m_run = max(m_run, c_run)
		max_torque_overload_s = m_run / float(fps)
		if t_source == "none":
			t_source = "solver_measured"
	else:
		t_source = "none"

	# 4. Unintended link collisions (including substep peaks and impulse)
	frame_unint_peak = unint_a.copy()
	unint_impulse = float(unint_a.sum() * dt_frame)
	substep_hit_frames: list[int] = []
	if substep_unintended_forces is not None:
		sub_a = np.asarray(substep_unintended_forces, dtype=np.float64)
		if sub_a.ndim != 2 or sub_a.shape[0] != n or sub_a.shape[1] == 0 or not np.all(np.isfinite(sub_a)):
			return _unknown_result(["corrupted_substep_telemetry"], {"shape": list(sub_a.shape)})
		sub_peaks = sub_a.max(axis=1)
		frame_unint_peak = np.maximum(frame_unint_peak, sub_peaks)
		unint_impulse = float(sub_a.sum() * (dt_frame / float(sub_a.shape[1])))
		substep_hit_frames = [int(i) for i in np.flatnonzero(sub_peaks > unintended_force_limit_N)]

	unintended_hits = frame_unint_peak > unintended_force_limit_N
	unintended_count = int(unintended_hits.sum())
	max_unintended = float(frame_unint_peak.max())

	# 5. Terminal bowl geometric containment & post-release settle window (trailing continuous 0.5 s)
	open_after_place = np.flatnonzero(((phases_arr == p_place) | (phases_arr == p_return)) & (q_cmd_a[:, 5] > 0.35))
	bowl_xy = np.asarray(target_bowl_xy, dtype=np.float64)
	speeds = np.linalg.norm(bvel_a, axis=1)
	ang_speeds = np.linalg.norm(bang_a, axis=1)
	req_settle = (
		int(settle_frames_required) if settle_frames_required is not None else max(1, round(settle_window_s * fps))
	)

	obs_start = int(open_after_place[0]) if len(open_after_place) > 0 else n
	obs_end = min(n, obs_start + max(1, round(post_release_observe_max_s * fps))) if obs_start < n else n
	final_idx = obs_end - 1
	final_dist = float(np.hypot(*(bpos_a[final_idx, :2] - bowl_xy)))
	final_speed = float(speeds[final_idx])
	final_ang_vel = float(ang_speeds[final_idx])

	contained_post: list[bool] = []
	calm_post: list[bool] = []
	if obs_start < obs_end:
		for idx in range(obs_start, obs_end):
			in_b = _brick_contained_in_bowl(bpos_a[idx], bquat_a[idx], brick_size, bowl_xy, bowl_r, bowl_h)
			not_regrasped = bool((not dual[idx]) and (q_cmd_a[idx, 5] > 0.35))
			calm = bool(
				speeds[idx] < max_settle_speed_mps and ang_speeds[idx] < max_settle_ang_vel_rad_s and not_regrasped
			)
			contained_post.append(in_b)
			calm_post.append(in_b and calm)

	# Trailing continuous settle streak right up to obs_end - 1 ("结束前连续 0.5 秒")
	trailing_consec = 0
	for ok_c in reversed(calm_post):
		if ok_c:
			trailing_consec += 1
		else:
			break
	settled = bool(len(calm_post) >= req_settle and trailing_consec >= req_settle)
	final_contained = bool(contained_post and contained_post[-1])
	ever_contained = any(contained_post)

	# 6. Collect ALL process and terminal failure reasons independently
	process_reasons: list[str] = []
	terminal_reasons: list[str] = []

	if not grasped_ok:
		terminal_reasons.append("pick_miss")
	if max_torque_overload_s > max_stall_secs and t_source == "solver_measured":
		process_reasons.append("torque_stall")
	if max_lag_s > max_stall_secs:
		process_reasons.append("tracking_lag")
	if unintended_count > 0:
		process_reasons.append("unintended_collision")

	if grasped_ok and (max_slip_m > slip_pos_limit_m or max_slip_deg > slip_rot_limit_deg or dual_ratio < 0.70):
		if max_slip_m > slip_pos_limit_m or max_slip_deg > slip_rot_limit_deg:
			# Was the violation already present in lift, or only starting in show/handoff?
			lift_set = {int(i) for i in lift_idx}
			early_lift_viol = False
			if ref_idx is not None:
				rel0, q0 = _rel_pose(ref_idx)
				for i in carry_idx:
					if int(i) in lift_set and int(i) >= ref_idx:
						ri, qi = _rel_pose(int(i))
						if (
							float(np.linalg.norm(ri - rel0)) > slip_pos_limit_m
							or _quat_angle_deg(qi, q0) > slip_rot_limit_deg
						):
							early_lift_viol = True
							break
			show_set = {int(i) for i in np.flatnonzero(np.isin(phases_arr, [p_show_move, p_show_hold, p_handoff]))}
			if not early_lift_viol and first_slip_frame in show_set:
				process_reasons.append("show_drop")
			else:
				process_reasons.append("grasp_slip")
		if dual_ratio < 0.70 and "grasp_slip" not in process_reasons and "show_drop" not in process_reasons:
			process_reasons.append("dual_contact_loss")

	if not settled:
		if not grasped_ok:
			if "pick_miss" not in terminal_reasons:
				terminal_reasons.append("pick_miss")
		elif obs_start >= obs_end:
			terminal_reasons.append("place_miss")
		elif final_contained:
			terminal_reasons.append("settle_timeout")
		elif ever_contained:
			terminal_reasons.append("bounce_out")
		else:
			terminal_reasons.append("place_miss")

	terminal_ok = grasped_ok and settled and len(terminal_reasons) == 0
	process_ok = len(process_reasons) == 0
	ordered_pool = [
		code
		for code in (
			"pick_miss",
			"torque_stall",
			"unintended_collision",
			"show_drop",
			"grasp_slip",
			"dual_contact_loss",
			"tracking_lag",
			"bounce_out",
			"settle_timeout",
			"place_miss",
		)
		if code in terminal_reasons or code in process_reasons
	]
	all_reasons = tuple(dict.fromkeys([*ordered_pool, *terminal_reasons, *process_reasons]))
	verdict: Verdict = "SUCCESS" if (terminal_ok and process_ok) else "FAILURE"
	failure_mode = "ok" if verdict == "SUCCESS" else (all_reasons[0] if all_reasons else "failure")

	evidence.update(
		{
			"n_frames": n,
			"ref_frame": ref_idx,
			"slip_peak_frame": slip_peak_frame,
			"release_obs_window": [obs_start, obs_end],
			"req_settle_frames": req_settle,
			"trailing_settle_frames": trailing_consec,
			"final_contained_in_bowl": final_contained,
			"ever_contained_in_bowl": ever_contained,
			"substep_collision_frames": substep_hit_frames[:10],
			"unintended_collision_frames": [int(i) for i in np.flatnonzero(unintended_hits)[:10]],
		}
	)

	return EpisodeHealthMetrics(
		verdict=verdict,
		reason_codes=all_reasons,
		evidence_refs=evidence,
		evaluator_version=EVALUATOR_VERSION,
		terminal_ok=terminal_ok,
		process_ok=process_ok,
		success=(verdict == "SUCCESS"),
		failure_mode=failure_mode,
		grasped_ok=grasped_ok,
		max_slip_pos_m=max_slip_m,
		max_slip_rot_deg=max_slip_deg,
		dual_contact_ratio=dual_ratio,
		mean_pinch_force_N=mean_pinch,
		max_pinch_force_N=max_pinch,
		pinch_normal_peak_N=pinch_normal_peak,
		pinch_tangential_peak_N=pinch_tangential_peak,
		pinch_impulse_Ns=pinch_impulse,
		max_tracking_error_rad=max_track_err,
		tracking_error_ratio=track_err_ratio,
		max_tracking_lag_duration_s=max_lag_s,
		max_stall_duration_s=max_torque_overload_s,
		torque_saturation_ratio=torque_sat_ratio,
		torque_evidence_source=t_source,
		unintended_collision_count=unintended_count,
		max_unintended_force_N=max_unintended,
		unintended_collision_impulse_Ns=unint_impulse,
		settled_in_bowl=settled,
		settle_consecutive_frames=trailing_consec,
		final_dist_to_bowl_m=final_dist,
		final_brick_speed_mps=final_speed,
		final_brick_ang_vel_rps=final_ang_vel,
	)


def _interp_series(t_src: np.ndarray, y_src: np.ndarray, t_dst: np.ndarray, wrap_angle: bool = False) -> np.ndarray:
	arr = np.asarray(y_src, dtype=np.float64)
	if wrap_angle:
		arr = np.unwrap(arr)
	if arr.ndim == 1:
		out = np.interp(t_dst, t_src, arr)
		if wrap_angle:
			out = (out + math.pi) % (2.0 * math.pi) - math.pi
		return out
	cols = [np.interp(t_dst, t_src, arr[:, j]) for j in range(arr.shape[1])]
	return np.column_stack(cols)


def _estimate_time_shift_s(t_common: np.ndarray, a: np.ndarray, b: np.ndarray, max_lag_steps: int = 10) -> float:
	if len(t_common) < 6:
		return 0.0
	dt = float(np.median(np.diff(t_common)))
	best_shift = 0
	best_mse = float("inf")
	for lag in range(-max_lag_steps, max_lag_steps + 1):
		if lag >= 0:
			sa = a[lag:]
			sb = b[: len(b) - lag]
		else:
			sa = a[: len(a) + lag]
			sb = b[-lag:]
		if len(sa) < max(4, len(a) // 3):
			continue
		mse = float(np.mean((sa - sb) ** 2))
		if mse < best_mse - 1e-12 or (abs(mse - best_mse) <= 1e-12 and abs(lag) < abs(best_shift)):
			best_mse = mse
			best_shift = lag
	# Positive lag means `a` at index (t+lag) matches `b` at index t -> `b` is delayed by +lag * dt relative to `a`
	return float(-best_shift * dt)


def compare_trajectories(
	traj_a: dict[str, Any],
	traj_b: dict[str, Any],
	fps: int = FPS,
) -> dict[str, Any]:
	"""Compare trajectories over common timestamps without hiding duration/length mismatches or terminal deltas."""
	signal_keys = ("qpos", "tcp_pos", "brick_pos", "brick_yaw")
	present = [k for k in signal_keys if k in traj_a and k in traj_b]
	if not present:
		return {"valid": False, "reason": "no_common_signals"}

	first_k = present[0]
	len_a = len(traj_a[first_k])
	len_b = len(traj_b[first_k])
	if len_a == 0 or len_b == 0:
		return {"valid": False, "reason": "empty_trajectory", "len_a": len_a, "len_b": len_b}

	t_a = (
		np.asarray(traj_a["t"], dtype=np.float64) if "t" in traj_a else np.arange(len_a, dtype=np.float64) / float(fps)
	)
	t_b = (
		np.asarray(traj_b["t"], dtype=np.float64) if "t" in traj_b else np.arange(len_b, dtype=np.float64) / float(fps)
	)
	if (
		len(t_a) != len_a
		or len(t_b) != len_b
		or not np.all(np.isfinite(t_a))
		or not np.all(np.isfinite(t_b))
		or (len_a > 1 and np.any(np.diff(t_a) <= 0.0))
		or (len_b > 1 and np.any(np.diff(t_b) <= 0.0))
	):
		return {"valid": False, "reason": "invalid_timestamps", "len_a": len_a, "len_b": len_b}

	out: dict[str, Any] = {
		"valid": True,
		"len_a": len_a,
		"len_b": len_b,
		"length_mismatch": bool(len_a != len_b),
		"duration_a_s": float(t_a[-1] - t_a[0]),
		"duration_b_s": float(t_b[-1] - t_b[0]),
		"duration_diff_s": float((t_a[-1] - t_a[0]) - (t_b[-1] - t_b[0])),
	}

	t0 = max(float(t_a[0]), float(t_b[0]))
	t1 = min(float(t_a[-1]), float(t_b[-1]))
	if t1 < t0:
		return {**out, "valid": False, "reason": "disjoint_time_intervals"}

	overlap_mask = (t_a >= t0 - 1e-9) & (t_a <= t1 + 1e-9)
	t_common = t_a[overlap_mask]
	if len(t_common) == 0:
		t_common = np.array([t0], dtype=np.float64)

	for key, suffix in (("qpos", "rad"), ("tcp_pos", "m"), ("brick_pos", "m"), ("brick_yaw", "rad")):
		if key not in traj_a or key not in traj_b:
			continue
		a_raw = np.asarray(traj_a[key], dtype=np.float64)
		b_raw = np.asarray(traj_b[key], dtype=np.float64)
		if len(a_raw) != len_a or len(b_raw) != len_b:
			return {**out, "valid": False, "reason": f"length_mismatch_in_{key}"}
		if not np.all(np.isfinite(a_raw)) or not np.all(np.isfinite(b_raw)):
			return {**out, "valid": False, "reason": f"non_finite_{key}"}

		a = a_raw[overlap_mask]
		b = _interp_series(t_b, b_raw, t_common, wrap_angle=(key == "brick_yaw"))
		diff = a - b
		if key == "brick_yaw":
			diff = (diff + math.pi) % (2.0 * math.pi) - math.pi
			err = np.abs(diff)
			out[f"{key}_rmse_{suffix}"] = float(np.sqrt(np.mean(err**2)))
			out[f"{key}_max_{suffix}"] = float(np.max(err))
		elif diff.ndim == 2 and key in ("tcp_pos", "brick_pos"):
			dist = np.linalg.norm(diff, axis=1)
			out[f"{key}_rmse_{suffix}"] = float(np.sqrt(np.mean(dist**2)))
			out[f"{key}_max_{suffix}"] = float(np.max(dist))
			if key == "brick_pos":
				# Full-horizon un-truncated terminal position vector differences
				out["brick_final_pos_diff_m"] = float(np.linalg.norm(a_raw[-1] - b_raw[-1]))
				out["brick_final_xy_diff_m"] = float(np.hypot(*(a_raw[-1, :2] - b_raw[-1, :2])))
				out["estimated_time_shift_s"] = _estimate_time_shift_s(t_common, a, b)
		else:
			out[f"{key}_rmse_{suffix}"] = float(np.sqrt(np.mean(diff**2)))
			out[f"{key}_max_{suffix}"] = float(np.max(np.abs(diff)))

	if "estimated_time_shift_s" not in out and "qpos" in present:
		a_q = np.asarray(traj_a["qpos"], dtype=np.float64)[overlap_mask]
		b_q = _interp_series(t_b, np.asarray(traj_b["qpos"], dtype=np.float64), t_common)
		out["estimated_time_shift_s"] = _estimate_time_shift_s(t_common, a_q, b_q)

	if "events" in traj_a and "events" in traj_b:
		ev_a = traj_a["events"]
		ev_b = traj_b["events"]
		out["event_time_diff_s"] = {str(k): float(ev_a[k] - ev_b[k]) for k in ev_a if k in ev_b}
	return out
