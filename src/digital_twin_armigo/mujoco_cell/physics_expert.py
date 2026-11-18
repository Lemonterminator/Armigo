"""Closed-loop pure-physics pick → show → handoff → place expert on the MuJoCo cell.

``ClosedLoopExpert`` reads privileged simulator state every frame and returns the next joint target; it acts only
through that target, which goes through the same :class:`~digital_twin_armigo.physics.ActuatorCommandFilter` as the
open-loop plan. The episode recorder lives in :mod:`.episode`, the jaw / grasp geometry in :mod:`.grasp_geometry` and
the post-hoc diagnostics in :mod:`.episode_diagnostics`; their public names are re-exported here.

Privileged inputs the expert reads (never writes): arm joint positions, gripper-frame pose, brick pose and velocity,
per-jaw contact normal forces, the target bowl centre, the clutter bricks' positions (visible to the cameras; used
for the release point and the grasp clearance check), and the compiled model's geometry (jaw collision geoms, bowl
floor, bowls / clutter for clearance). IK and forward kinematics run on a private ``MjData``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

from ..constants import BOWL_FLOOR_TOP_M, BOWL_R, FPS, HANDOFF_Q, PHASES
from ..physics import NonIdealityCfg
from ._mj import mujoco
from .cell import Cell
from .episode import PHYSICS_HZ, SUBSTEPS, WARMUP_STEPS, PhysicsEpisode, jaw_env_contact_force, observe
from .episode_diagnostics import EVALUATOR_RELEASE_DELTA_RAD, episode_diagnostics, show_hold_measured
from .grasp_geometry import Kinematics, jaw_pad_faces, jaw_sweep_low_point, quat_xyzw_to_mat
from .gripper_state import GripperMonitor, GripperState

if TYPE_CHECKING:
	from ..inspection import InspectionResult

InspectFn = Callable[[float, float, int], "InspectionResult"]
"""Inspection callback: (wrist flex rad, wrist roll deg, frame index) -> result for the current simulator state."""

NextPoseFn = Callable[[Sequence[Mapping]], "tuple[float, float] | None"]
"""Show-pose policy: the inspection history so far (:data:`POLICY_HISTORY_KEYS` of each record in
``ClosedLoopExpert.inspections``) -> the next (wrist flex rad, wrist roll deg) to show, or None to stop. Asked only
while the label is UNKNOWN after the base poses."""


class EpisodePlanner(Protocol):
	"""A whole-episode inspection strategy driving the expert through extra stages (used instead of ``inspect``).

	The expert calls, in order: :meth:`prescan_poses` once after the ready move (arm poses to visit with the open
	gripper while the brick is still on the table) and :meth:`after_prescan_view` after holding each; then
	:meth:`choose_grasp` with the grasp candidates (best first) for every grasp plan; :meth:`after_grasp` once the
	brick is lifted; then :meth:`next_present_pose` / :meth:`after_present` alternately until the former returns None
	(whole-arm poses to carry the brick to and hold); finally :meth:`label` picks the bowl (None aborts the episode
	with ``inspection_unknown``). Every method may read ``expert.cell`` (the live simulator state) and ``expert``."""

	def prescan_poses(self, expert: ClosedLoopExpert) -> list[np.ndarray]: ...

	def after_prescan_view(self, expert: ClosedLoopExpert, k: int) -> bool: ...

	def choose_grasp(self, expert: ClosedLoopExpert, cands: list[dict]) -> int: ...

	def after_grasp(self, expert: ClosedLoopExpert) -> None: ...

	def next_present_pose(self, expert: ClosedLoopExpert, q_lift: np.ndarray) -> np.ndarray | None: ...

	def after_present(self, expert: ClosedLoopExpert) -> None: ...

	def label(self) -> str | None: ...


POLICY_HISTORY_KEYS = ("frame", "flex", "roll_deg", "status", "visible_faces", "observed_faces")
"""What a ``next_pose`` policy sees of each inspection. Not ``reason``: an inspector may word its UNKNOWN reasons
differently per class (``inspection.inspect_object_sim_oracle`` does), which would reveal the label; not ``label``,
which is always None while the policy is asked."""

__all__ = [
	"EVALUATOR_RELEASE_DELTA_RAD",
	"PHYSICS_HZ",
	"SUBSTEPS",
	"WARMUP_STEPS",
	"ClosedLoopExpert",
	"EpisodePlanner",
	"ExpertParams",
	"InspectFn",
	"NextPoseFn",
	"PhysicsEpisode",
	"assess_feasibility",
	"bowl_floor_z",
	"episode_diagnostics",
	"jaw_env_contact_force",
	"jaw_pad_faces",
	"jaw_sweep_low_point",
	"run_closed_loop_episode",
	"show_hold_measured",
]


@dataclass(frozen=True)
class ExpertParams:
	"""Every tunable number of :class:`ClosedLoopExpert`. Lengths in metres, angles in radians unless the name says
	degrees, durations in seconds, forces in newtons, speeds in metres per second."""

	open_g: float = 0.85  # gripper command with the jaws open (rad)
	hold_g: float = 0.06  # gripper command while pinching (rad); the gripper torque saturates below contact
	rest_s: float = 0.3  # initial hold at rest
	ready_q: tuple[float, float, float, float, float] = (0.0, -0.35, 0.25, 1.35, 0.0)  # intermediate arm pose (rad)
	ready_move_s: float = 0.6
	hover_z_m: float = 0.06  # gripper-frame height above the grasp point before descending
	hover_move_s: float = 1.0
	pregrasp_dz_m: float = 0.02  # height above the grasp height where the final descent starts
	descend_s: float = 0.7
	final_descend_s: float = 0.6
	jaw_table_clearance_m: float = 0.003  # lowest point of either jaw over the closing sweep above the table
	pad_clearance_m: float = 0.002  # gap between the fixed-jaw pad and the brick face before closing
	servo_tol_m: float = 0.0008  # gripper-frame position error accepted before closing
	servo_gain: float = 0.35  # fraction of the measured gripper-frame error fed back into the target per frame
	servo_max_correction_m: float = 0.02  # cap on the accumulated correction
	servo_max_frames: int = 30
	sweep_samples: int = 25  # gripper angles sampled over the closing sweep for the jaw low point
	roll_scan_steps: int = 721  # wrist-roll samples over its range when lining the jaws up with the brick
	roll_align_min: float = 0.98  # cosine between jaw axis and brick axis a roll candidate must reach
	roll_penalty_m_per_rad: float = 0.002  # candidate score penalty per radian of wrist roll (m of clearance)
	clearance_cap_m: float = 0.02  # clearance beyond this does not improve a grasp candidate's score
	min_grasp_clearance_m: float = 0.002  # candidates closer than this to bowls / clutter count as blocked
	clearance_sweep_samples: int = 6  # gripper angles checked for clearance over the closing sweep
	grasp_refine_iters: int = 3  # target / pad-face refinement iterations for one grasp candidate
	place_servo_tol_m: float = 0.003  # release-point error accepted before opening
	close_s: float = 0.5  # gripper closing ramp
	clamp_force_n: float = 1.5  # both jaws' normal force needed for a clamp
	clamp_still_speed_mps: float = 0.005  # brick speed below which it counts as stationary
	clamp_still_frames: int = 4  # consecutive clamped, stationary frames before lifting
	clamp_timeout_frames: int = 45  # frames after the closing ramp before a grasp attempt is abandoned
	max_grasp_retries: int = 2
	reopen_s: float = 0.5
	lift_z_m: float = 0.06  # lift height; low, so the jaws-down creep phase is short (measured)
	lift_s: float = 1.2  # minimum lift duration
	carry_peak_speed_rad_s: float = 0.5  # peak arm joint speed of any move while carrying the brick
	show_flex: float = 0.65  # wrist flex of the show poses (rad)
	show_roll_deg: tuple[float, float, float] = (-35.0, 0.0, 35.0)
	show_move_s: float = 0.9  # minimum show move duration
	show_peak_speed_rad_s: float = 1.2  # peak joint speed of the show moves (wrist only; jaws near horizontal)
	show_hold_s: float = 0.4  # measured hold per show pose (timer starts once the pose is reached and still)
	show_hold_timeout_s: float = 2.0  # give up waiting for the measured show pose after this long
	show_flex_tol: float = 0.05  # wrist flex tolerance of the measured show-pose criterion (rad)
	show_roll_tol_deg: float = 2.5  # wrist roll tolerance of the measured show-pose criterion
	show_still_speed_rad_s: float = 0.05  # arm joints slower than this count as still
	carry_joint_gain: float = 0.3  # per-frame integral gain on the joint error while holding a carry pose
	carry_joint_max_corr: float = 0.05  # cap on that correction (rad)
	handoff_move_s: float = 1.5
	handoff_hold_s: float = 0.6
	transfer_z_m: float = 0.12  # gripper-frame height when moving over the bowl
	transfer_s: float = 1.8
	place_roll: float = 0.0  # wrist roll over the bowl (rad): the handoff pose's, so the transfer does not roll
	release_clearance_m: float = 0.004  # brick bottom above the bowl floor at release
	place_descend_s: float = 1.2
	open_ramp_s: float = 1.0  # gripper opening ramp at release
	settle_speed_mps: float = 0.01
	settle_frames: int = 6
	settle_timeout_s: float = 1.0
	retreat_dz_m: float = 0.06
	retreat_s: float = 0.8
	return_s: float = 1.2
	# Peak joint speed of the return move. Under the 2.94 N m torque limit the wrist flex tops out near 4.6 rad/s;
	# planning faster leaves it lagging and then snapping to the decelerating target (measured jerk spike).
	return_peak_speed_rad_s: float = 4.4
	final_hold_s: float = 0.6
	lost_force_n: float = 0.02  # a jaw below this normal force has lost contact
	lost_frames: int = 10  # consecutive frames without dual contact after lift: brick lost
	max_frames: int = 900
	max_move_chunk_s: float = 1.4  # longest single min-jerk segment; longer moves are chunked
	chunk_pause_s: float = 0.1  # pause between chunks of a long move
	bowl_margin_m: float = 0.012  # keep the brick's corners this far inside the bowl radius
	clutter_clearance_m: float = 0.03  # preferred distance of the release point from clutter bricks
	release_grid_radii: int = 6  # radial samples of the in-bowl release-point search
	release_grid_angles: int = 24  # angular samples of the in-bowl release-point search
	release_off_centre_weight: float = 0.1  # score penalty per metre the release point lies off the bowl centre
	block_backoff_rad: float = 0.10  # gripper backoff opposite to block direction when BLOCKED
	max_block_replans: int = 2  # maximum reactive replans when gripper is BLOCKED (separate from max_grasp_retries)
	open_ladder: tuple[float, ...] = (0.85, 0.70, 0.55, 0.40)  # candidate pre-grasp openings from widest to narrowest
	pre_narrow_open: bool = True  # Stage 3: pre-narrow along open_ladder only when default open_g is blocked
	# Inspection (used only with an ``inspect`` callback): extra (wrist flex rad, wrist roll deg) show poses tried in
	# order while the label is still UNKNOWN after the ``show_roll_deg`` poses, and whether to stop showing as soon
	# as the label is KNOWN.
	extra_show_poses: tuple[tuple[float, float], ...] = ()
	stop_when_known: bool = True
	max_extra_shows: int = 4  # cap on the poses a ``next_pose`` policy may add after the base poses
	# Planner stages (used only with a ``planner``): pre-grasp viewpoint moves / holds and the present-pose holds.
	prescan_move_s: float = 0.6
	prescan_peak_speed_rad_s: float = 1.0
	prescan_hold_s: float = 0.25
	present_hold_s: float = 0.5
	present_peak_speed_rad_s: float = 0.3  # whole-arm close-up moves while carrying: slower than ordinary carries


def bowl_floor_z(model, data, bowl_xy: np.ndarray) -> float:
	"""Top of the bowl inner flat floor (metres): constant ``BOWL_FLOOR_TOP_M`` verified against raycasts."""
	return BOWL_FLOOR_TOP_M


MIN_JERK_PEAK_RATIO = 1.875  # peak / mean speed of a min-jerk profile


def _min_jerk(n: int) -> np.ndarray:
	s = np.linspace(0.0, 1.0, n)
	return 10 * s**3 - 15 * s**4 + 6 * s**5


@dataclass(frozen=True)
class _Grasp:
	"""A grasp the expert executed: gripper-frame target, wrist roll and the servo correction it settled on."""

	target: np.ndarray
	roll: float
	corr: np.ndarray


class ClosedLoopExpert:
	"""Per-frame pick → show → handoff → place policy on privileged simulator state.

	``act(obs)`` takes the current observation (``q``, ``tcp``, ``brick_pos``, ``brick_quat``, ``brick_vel``,
	``jaw_normal``) and returns ``(joint_target, phase)``; ``done`` turns True when the episode script has ended.
	"""

	def __init__(
		self,
		cell: Cell,
		q_start: np.ndarray,
		bowl_xy: tuple[float, float] | None,
		params: ExpertParams,
		*,
		bowls: Mapping[str, tuple[float, float]] | None = None,
		inspect: InspectFn | None = None,
		next_pose: NextPoseFn | None = None,
		planner: EpisodePlanner | None = None,
	) -> None:
		"""``bowl_xy`` fixes the target bowl up front (the label read from ground truth). With ``inspect`` instead,
		the target is chosen after the show poses from the inspection label: ``bowls`` maps each label
		(``"normal"`` / ``"defective"``) to a bowl centre. ``planner`` replaces ``inspect`` and the show poses with a
		whole-episode strategy (see :class:`EpisodePlanner`)."""
		n_modes = sum(x is not None for x in (bowl_xy, inspect, planner))
		if n_modes != 1:
			raise ValueError("give exactly one of bowl_xy (fixed target), inspect (+ bowls) or planner (+ bowls)")
		if (inspect is not None or planner is not None) and not bowls:
			raise ValueError("inspect / planner needs bowls: label -> bowl centre")
		if next_pose is not None and (inspect is None or params.extra_show_poses):
			raise ValueError("next_pose needs inspect and replaces extra_show_poses")
		self.planner = planner
		self.cell = cell
		self.p = params
		self.q_start = np.asarray(q_start, dtype=np.float64).copy()
		self.bowl_xy = None if bowl_xy is None else np.asarray(bowl_xy, dtype=np.float64)
		self.bowls = {k: np.asarray(v, dtype=np.float64) for k, v in (bowls or {}).items()}
		self.inspect = inspect
		self.next_pose = next_pose
		self.inspections: list[dict] = []  # one record per inspection call
		self.label: str | None = None
		self.kin = Kinematics(cell)
		self.size = tuple(cell.layout.brick_size)
		self.obs: dict[str, np.ndarray] = {}
		self.last = self.q_start.copy()
		self.grasp_retries = 0
		self.block_replans = 0
		self.grasp_info: dict[str, float] = {}
		self.gripper_monitor = GripperMonitor()
		self.gripper_state = GripperState.FREE
		self.gripper_block_direction = 0
		self._last_block_dir = 0
		self.blocked_frames: list[int] = []
		self._blocked_poses: list[tuple[np.ndarray, float]] = []
		self._active_open_g = float(params.open_g)
		self._frame_idx = 0
		self.lost = False
		self.aborted: str | None = None
		self.done = False
		self._prev_q = self.q_start.copy()
		self._no_dual = 0
		self._lo, self._hi = cell.jnt_range[:, 0].copy(), cell.jnt_range[:, 1].copy()
		self.clipped_frames = 0  # frames whose scripted target fell outside the joint range and was clipped
		self._gen = self._script()

	# -- plumbing --------------------------------------------------------------------------------------------------

	def act(self, obs: dict[str, np.ndarray]) -> tuple[np.ndarray, str]:
		self._prev_q = self.obs["q"] if self.obs else obs["q"]
		self.obs = obs
		self.gripper_state, direction = self.gripper_monitor.update(
			float(obs["gripper_cmd"]),
			float(obs["q"][5]),
			float(obs["effort_frac"]),
			obs["jaw_normal"],
		)
		if self.gripper_state is GripperState.BLOCKED:
			self.blocked_frames.append(self._frame_idx)
			self._last_block_dir = direction
			if self.gripper_block_direction == 0:
				self.gripper_block_direction = direction
		try:
			target, phase = next(self._gen)
		except StopIteration:
			self.done = True
			return self.last.copy(), "return"
		self._frame_idx += 1
		clipped = np.clip(target, self._lo, self._hi)
		self.clipped_frames += bool(np.any(clipped != target))
		target = clipped
		self.last = target.copy()
		return target, phase

	def _frames(self, secs: float) -> int:
		return max(1, round(secs * FPS))

	def _move(self, q_to: np.ndarray, secs: float, phase: str):
		"""Min-jerk move from the last target to ``q_to``. Moves longer than ``max_move_chunk_s`` are split into
		equal chunks with a short pause between them: a joint moving faster than about 0.25 rad/s saturates its
		actuator (damping 12 N m s/rad against a 2.94 N m limit), and the evaluator fails continuous saturation
		longer than 1.5 s."""
		p = self.p
		q_from = self.last.copy()
		# Plan to the reachable target. Clipping a min-jerk path at a joint limit instead stops that joint dead
		# mid-move (the rest pose lies just outside two limits, which made a jerk spike at the end of every return).
		q_to = np.clip(q_to, self._lo, self._hi)
		chunks = max(1, math.ceil(secs / p.max_move_chunk_s))
		for c in range(chunks):
			a, b = q_from + (q_to - q_from) * (c / chunks), q_from + (q_to - q_from) * ((c + 1) / chunks)
			n = max(2, self._frames(secs / chunks) + 1)
			for s in _min_jerk(n)[1:]:
				yield a + s * (b - a), phase
			if c < chunks - 1:
				for _ in range(self._frames(p.chunk_pause_s)):
					yield b.copy(), phase

	def _carry_secs(self, q_to: np.ndarray, min_s: float, peak_speed: float | None = None) -> float:
		"""Duration of a carry move from the last target to ``q_to``: at least ``min_s`` and long enough that the
		min-jerk peak arm joint speed (1.875 x mean speed) stays below ``carry_peak_speed_rad_s``. Fast carry
		moves make the friction pinch creep out of the jaw tips (measured), so big moves are slowed down."""
		dq = float(np.max(np.abs(np.asarray(q_to)[:5] - self.last[:5])))
		speed = self.p.carry_peak_speed_rad_s if peak_speed is None else peak_speed
		return max(min_s, MIN_JERK_PEAK_RATIO * dq / speed)

	def _hold(self, secs: float, phase: str):
		for _ in range(self._frames(secs)):
			yield self.last.copy(), phase

	def _hold_servo(self, q_des: np.ndarray, secs: float, phase: str):
		"""Hold arm pose ``q_des`` while integrating the measured joint error into the target (gravity sag,
		backlash), so the joints settle on ``q_des`` rather than short of it."""
		p = self.p
		corr = np.zeros(5)
		for _ in range(self._frames(secs)):
			err = q_des[:5] - self.obs["q"][:5]
			corr = np.clip(corr + p.carry_joint_gain * err, -p.carry_joint_max_corr, p.carry_joint_max_corr)
			q = q_des.copy()
			q[:5] += corr
			yield q, phase

	def _ik(self, target: np.ndarray, q0: np.ndarray, roll: float) -> np.ndarray:
		from .ik import ik_reach

		q = ik_reach(self.kin, np.asarray(target, dtype=np.float64), q0, roll)
		q[5] = q0[5]
		return q

	def _dual(self) -> bool:
		f = self.obs.get("jaw_normal", np.zeros(2))
		return bool(f[0] > self.p.lost_force_n and f[1] > self.p.lost_force_n)

	# -- grasp geometry --------------------------------------------------------------------------------------------

	def _env_geoms(self) -> np.ndarray:
		"""Collision geoms that are neither the robot nor the target brick, nor the table (bowls, clutter)."""
		m = self.cell.model
		table = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table")
		brick = {int(g) for g in self.cell.brick_geoms}
		return np.array(
			[
				g
				for g in range(m.ngeom)
				if g not in self.cell.robot_all_geoms
				and g not in brick
				and g != table
				and (m.geom_contype[g] or m.geom_conaffinity[g])
			],
			dtype=int,
		)

	def _clearance(self, qs: list[np.ndarray]) -> float:
		"""Smallest distance (m) between the gripper / moving links and bowls or clutter over the arm poses ``qs``,
		on the private kinematic data (capped at ``clearance_cap_m``)."""
		p = self.p
		m, d = self.kin.model, self.kin.data
		robot = sorted(self.cell.fixed_jaw_geoms | self.cell.moving_jaw_geoms | {int(g) for g in self.cell.link_geoms})
		env = self._env_geoms()
		best = p.clearance_cap_m
		for q in qs:
			self.kin.site_pose(q)
			for g1 in robot:
				for g2 in env:
					dist = mujoco.mj_geomDistance(m, d, int(g1), int(g2), best, None)
					best = min(best, float(dist))
		return best

	def _grasp_candidate(
		self,
		roll: float,
		axis: np.ndarray,
		half_w: float,
		q_hover: np.ndarray,
		open_g: float | None = None,
	) -> dict:
		"""Grasp target, joints and clearance for wrist roll ``roll`` pinching across brick axis ``axis``."""
		p = self.p
		g_open = p.open_g if open_g is None else float(open_g)
		bpos = self.obs["brick_pos"]
		q_hover = q_hover.copy()
		q_hover[4] = roll
		z_grasp = (
			jaw_sweep_low_point(self.cell, self.kin, q_hover, g_open, p.hold_g, p.sweep_samples)
			+ p.jaw_table_clearance_m
		)
		# the brick occupies this slab of the gripper frame's approach axis (x, pointing down)
		band = (z_grasp - (bpos[2] + self.size[2] / 2.0), z_grasp - (bpos[2] - self.size[2] / 2.0))
		target = np.array([bpos[0], bpos[1], z_grasp])
		q = q_hover.copy()
		fixed_face, moving_face, offset = -np.inf, np.inf, 0.0
		for _ in range(p.grasp_refine_iters):  # the pad faces and the jaw axis depend on the solved pose
			q = self._ik(target, q, roll)
			q[5] = g_open
			_, rot = self.kin.site_pose(q)
			zdir = rot[:, 2].copy()
			zdir[2] = 0.0
			zdir /= max(np.linalg.norm(zdir), 1e-9)
			fixed_face, moving_face = jaw_pad_faces(self.cell, self.kin, q, band)
			# brick centre at gripper-frame z = fixed face + pad clearance + half width
			offset = -(fixed_face + p.pad_clearance_m + half_w)
			target = np.array([bpos[0] + offset * zdir[0], bpos[1] + offset * zdir[1], z_grasp])
		pos, _ = self.kin.site_pose(q)
		q_pre = self._ik(target + np.array([0.0, 0.0, p.pregrasp_dz_m]), q, roll)
		sweep = []
		for g in np.linspace(g_open, p.hold_g, p.clearance_sweep_samples):
			qg = q.copy()
			qg[5] = g
			sweep.append(qg)
		clear = self._clearance([q_pre, *sweep])
		jaw_gap = float(moving_face - fixed_face)
		min_gap = 2.0 * half_w + p.pad_clearance_m + 2.0 * p.servo_tol_m + 0.004
		return {
			"target": target,
			"roll": float(roll),
			"q": q,
			"open_g": g_open,
			"clearance_m": clear,
			"gap_ok": jaw_gap >= min_gap,
			"ik_error_m": float(np.linalg.norm(pos - target)),
			"info": {
				"fixed_face_m": fixed_face,
				"moving_face_m": moving_face,
				"offset_m": offset,
				"grasp_z_m": z_grasp,
				"roll": float(roll),
				"clearance_m": clear,
				"open_g": g_open,
			},
		}

	def grasp_candidates(self, open_g: float | None = None) -> list[dict]:
		"""Every jaw-side / roll option for the brick's current pose, best first.

		Candidates are the wrist rolls (local maxima over a scan of the roll range) that line the jaw axis up with
		the brick's short axis (and, for a square brick, the other axis too), in either direction. Each is scored by
		its clearance to bowls and clutter (capped) minus a small penalty on the roll magnitude.
		"""
		p = self.p
		g_open = p.open_g if open_g is None else float(open_g)
		self.kin.data.qpos[:] = self.cell.data.qpos  # current clutter / bowl state for the clearance checks
		bpos = self.obs["brick_pos"]
		rot_b = quat_xyzw_to_mat(self.obs["brick_quat"])
		axes = [(rot_b[:2, 1], 0.5 * self.size[1], 1)]  # (axis, half width, brick-local axis index)
		if abs(self.size[0] - self.size[1]) < 1e-6:
			axes.append((rot_b[:2, 0], 0.5 * self.size[0], 0))
		q_seed = np.array([*p.ready_q, g_open])
		q_hover = self._ik(np.array([bpos[0], bpos[1], p.hover_z_m]), q_seed, 0.0)
		lo, hi = self.cell.jnt_range[4]
		rolls = np.linspace(lo, hi, p.roll_scan_steps)
		zs = []
		for roll in rolls:
			q_hover[4] = roll
			_, rot = self.kin.site_pose(q_hover)
			z = rot[:2, 2]
			zs.append(z / max(np.linalg.norm(z), 1e-9))
		zs = np.array(zs)
		cands = []
		for axis, half_w, local in axes:
			axis = axis / max(np.linalg.norm(axis), 1e-9)
			for sign in (1.0, -1.0):
				align = sign * (zs @ axis)
				i = int(np.argmax(align))
				if align[i] < p.roll_align_min:
					continue
				c = self._grasp_candidate(float(rolls[i]), sign * axis, half_w, q_hover, open_g=g_open)
				c["score"] = min(c["clearance_m"], p.clearance_cap_m) - p.roll_penalty_m_per_rad * abs(c["roll"])
				c["pinched_faces"] = (2 * local, 2 * local + 1)  # the jaws cover the brick's +/- faces of this axis
				cands.append(c)
		cands.sort(key=lambda c: -c["score"])
		return cands

	def _grasp_plan(self, open_g: float | None = None) -> tuple[np.ndarray, float, np.ndarray]:
		"""Gripper-frame target, wrist roll and the grasp joint angles for the brick's current pose."""
		p = self.p
		cands = self.grasp_candidates(open_g=open_g)
		if not cands:
			raise RuntimeError("no wrist roll lines the jaws up with the brick")
		if self.planner is not None:
			k = int(self.planner.choose_grasp(self, cands))
			cands.insert(0, cands.pop(k))
		c = cands[0]
		if (
			p.pre_narrow_open
			and self.block_replans == 0
			and (open_g is None or abs(open_g - p.open_g) < 1e-6)
			and c["clearance_m"] < p.min_grasp_clearance_m
		):
			pool = [cand for cand in cands if cand["gap_ok"]]
			for g in p.open_ladder:
				if g >= p.open_g - 1e-6:
					continue
				pool.extend(cand for cand in self.grasp_candidates(open_g=g) if cand["gap_ok"])
			if self.planner is not None:  # keep the planner's choice of which faces the jaws cover
				same = [cand for cand in pool if cand.get("pinched_faces") == c.get("pinched_faces")]
				pool = same or pool
			if pool:
				c = max(pool, key=lambda cand: cand["score"])
				self._active_open_g = float(c["open_g"])
		self.z_grasp = c["info"]["grasp_z_m"]
		self.grasp_info = dict(c["info"])
		return c["target"], c["roll"], c["q"]

	def _next_clear_opening(self, current_open_g: float) -> float | None:
		"""Next narrower rung in ``open_ladder`` whose best candidate clears obstacles and fits the brick."""
		for g in self.p.open_ladder:
			if g >= current_open_g - 1e-6:
				continue
			cands = [c for c in self.grasp_candidates(open_g=g) if c["gap_ok"]]
			if cands and cands[0]["clearance_m"] >= self.p.min_grasp_clearance_m:
				return float(g)
		return None

	# -- the episode script ----------------------------------------------------------------------------------------

	def _servo(self, target: np.ndarray, roll: float, phase: str, tol: float):
		"""Hold the gripper frame at ``target``, feeding back the measured error, until within ``tol`` metres."""
		p = self.p
		corr = np.zeros(3)
		for _ in range(p.servo_max_frames):
			err = target - self.obs["tcp"]
			if np.linalg.norm(err) < tol:
				break
			corr = np.clip(corr + p.servo_gain * err, -p.servo_max_correction_m, p.servo_max_correction_m)
			yield self._ik(target + corr, self.last, roll), phase
		return corr

	def _script(self):
		"""The episode as a sequence of phases. Each phase is a generator of ``(joint_target, phase)`` frames; a phase
		that can fail sets ``aborted`` (or ``lost``) and the script goes home."""
		p = self.p
		yield from self._hold(p.rest_s, "rest")
		yield from self._move(np.array([*p.ready_q, p.open_g]), p.ready_move_s, "approach")
		if self.planner is not None:
			yield from self._prescan()
		grasp = yield from self._grasp()
		if grasp is None:
			yield from self._go_home()
			return
		q_lift = yield from self._lift(grasp)
		if self.planner is not None:
			yield from self._planned_present(q_lift)
		else:
			yield from self._present(q_lift)
		if self.aborted is not None:
			yield from self._go_home()
			return
		if not self.lost:
			yield from self._handoff()
		if not self.lost:
			yield from self._place(q_lift)
		if self.lost:
			self.aborted = "brick_lost"
		yield from self._go_home()

	def _approach_and_descend(self, phase: str):
		"""Plan a grasp for the brick's current pose, move over it, descend and servo onto the grasp point."""
		p = self.p
		target, roll, q_grasp = self._grasp_plan(open_g=self._active_open_g)
		hover = target + np.array([0.0, 0.0, p.hover_z_m - self.z_grasp])
		pre = target + np.array([0.0, 0.0, p.pregrasp_dz_m])
		q = self._ik(hover, q_grasp, roll)
		q[5] = self._active_open_g
		yield from self._move(q, p.hover_move_s, phase)
		q = self._ik(pre, q, roll)
		yield from self._move(q, p.descend_s, phase)
		q = self._ik(target, q, roll)
		yield from self._move(q, p.final_descend_s, "grasp")
		corr = yield from self._servo(target, roll, "grasp", p.servo_tol_m)
		return _Grasp(target=target, roll=roll, corr=corr)

	def _close_and_clamp(self):
		"""Close the gripper and wait for a verified clamp; returns ``"clamped"``, ``"blocked"`` or ``"timeout"``."""
		p = self.p
		q_closed = self.last.copy()
		q_closed[5] = p.hold_g
		for frame in self._move(q_closed, p.close_s, "grasp"):
			if self.gripper_state is GripperState.BLOCKED:
				return "blocked"
			yield frame
		still = 0
		for _ in range(p.clamp_timeout_frames):
			if self.gripper_state is GripperState.BLOCKED:
				return "blocked"
			f = self.obs["jaw_normal"]
			v = float(np.linalg.norm(self.obs["brick_vel"]))
			still = still + 1 if (min(f) >= p.clamp_force_n and v < p.clamp_still_speed_mps) else 0
			if still >= p.clamp_still_frames:
				return "clamped"
			yield self.last.copy(), "grasp"
		return "timeout"

	def _rise_to_hover(self, roll: float, g: float):
		"""Straight up from the current gripper frame to hover height with gripper command ``g``."""
		p = self.p
		q_up = self._ik(self.obs["tcp"] + np.array([0.0, 0.0, p.hover_z_m - self.z_grasp]), self.last, roll)
		q_up[5] = g
		yield from self._move(q_up, p.descend_s, "grasp")

	def _recover_blocked(self, roll: float):
		"""Back the gripper off the block, rise, and pick the next narrower opening; False when out of options."""
		p = self.p
		q_back = self.last.copy()
		q_back[5] = float(
			np.clip(self.obs["q"][5] - self._last_block_dir * p.block_backoff_rad, self._lo[5], self._hi[5])
		)
		yield q_back, "grasp"
		self._blocked_poses.append((self.obs["q"][:5].copy(), float(self.obs["q"][5])))
		yield from self._rise_to_hover(roll, q_back[5])
		self.block_replans += 1
		next_open = self._next_clear_opening(self._active_open_g)
		if self.block_replans > p.max_block_replans or next_open is None:
			return False
		self._active_open_g = next_open
		return True

	def _grasp(self):
		"""Approach, close and verify a clamp, recovering from a BLOCKED gripper and retrying a missed clamp.

		Returns the successful :class:`_Grasp`, or None with ``aborted`` set (``gripper_blocked`` / ``no_clamp``)."""
		p = self.p
		attempts = 0
		phase = "approach"
		while True:
			grasp = yield from self._approach_and_descend(phase)
			outcome = yield from self._close_and_clamp()
			phase = "grasp"
			if outcome == "blocked":
				if not (yield from self._recover_blocked(grasp.roll)):
					self.aborted = "gripper_blocked"
					return None
				continue
			if outcome == "clamped":
				return grasp
			if attempts >= p.max_grasp_retries:
				# no verified clamp after the retries: do not lift; the episode records the miss
				self.aborted = "no_clamp"
				return None
			attempts += 1
			self.grasp_retries = attempts
			q_open = self.last.copy()
			q_open[5] = self._active_open_g
			yield from self._move(q_open, p.reopen_s, "grasp")
			yield from self._rise_to_hover(grasp.roll, self._active_open_g)

	def _lift(self, grasp: _Grasp):
		"""Lift the clamped brick straight up; returns the lift pose (the base of the show and place moves)."""
		p = self.p
		lift_target = np.array([grasp.target[0], grasp.target[1], p.lift_z_m]) + grasp.corr
		q_lift = self._ik(lift_target, self.last, grasp.roll)
		q_lift[5] = p.hold_g
		yield from self._carry(self._move(q_lift, self._carry_secs(q_lift, p.lift_s), "lift"))
		return q_lift

	def _show_pose(self, q_lift: np.ndarray, roll_deg: float, flex: float | None = None) -> np.ndarray:
		q_show = q_lift.copy()
		q_show[3] = self.p.show_flex if flex is None else flex
		q_show[4] = math.radians(roll_deg)
		return q_show

	def _show_poses(self) -> Iterator[tuple[float, float]]:
		"""The (flex, roll deg) show poses in order: the base poses, then (inspection only, while UNKNOWN) the static
		``extra_show_poses`` or up to ``max_extra_shows`` poses from the ``next_pose`` policy. Lazy, so the policy
		sees every inspection before it picks."""
		p = self.p
		for wr in p.show_roll_deg:
			yield p.show_flex, wr
		if self.inspect is None:
			return
		if self.next_pose is None:
			for flex, wr in p.extra_show_poses:
				yield self._clip_show(flex, wr)
			return
		for _ in range(p.max_extra_shows):
			if self.label is not None:
				return
			pose = self.next_pose([{k: r[k] for k in POLICY_HISTORY_KEYS} for r in self.inspections])
			if pose is None:
				return
			flex, wr = (float(v) for v in pose)
			if not (math.isfinite(flex) and math.isfinite(wr)):
				return
			yield self._clip_show(flex, wr)

	def _clip_show(self, flex: float, roll_deg: float) -> tuple[float, float]:
		"""A show pose inside the wrist joint ranges (the commanded target is clipped there anyway; clipping here keeps
		the show-pose criterion reachable and the inspection record truthful)."""
		lo, hi = self._lo, self._hi
		return float(np.clip(flex, lo[3], hi[3])), math.degrees(float(np.clip(math.radians(roll_deg), lo[4], hi[4])))

	def _present(self, q_lift: np.ndarray):
		"""Show the brick to the camera at each show pose in turn (see :meth:`_show_poses`).

		With an ``inspect`` callback, inspect after each show hold, stop once the label is KNOWN (``stop_when_known``;
		extra poses always stop), then pick the target bowl from the label; still UNKNOWN after the last pose sets
		``aborted = "inspection_unknown"``."""
		p = self.p
		for i, (flex, wr) in enumerate(self._show_poses()):
			if self.lost:
				return
			if self.inspect is not None and self.label is not None and (p.stop_when_known or i >= len(p.show_roll_deg)):
				break
			q_show = self._show_pose(q_lift, wr, flex)
			secs = self._carry_secs(q_show, p.show_move_s, p.show_peak_speed_rad_s)
			yield from self._carry(self._move(q_show, secs, "show_move"))
			yield from self._carry(self._show_hold(q_show))
			if self.inspect is not None and not self.lost:
				self._inspect_now(flex, wr)
		if self.inspect is not None and not self.lost:
			if self.label is None:
				self.aborted = "inspection_unknown"
			else:
				self.bowl_xy = self.bowls[self.label].copy()

	def _prescan(self):
		"""Planner stage 1: visit the planner's pre-grasp viewpoints with the open gripper, holding each still."""
		p = self.p
		for k, q in enumerate(self.planner.prescan_poses(self)):
			q = np.asarray(q, dtype=np.float64).copy()
			q[5] = self._active_open_g
			dq = float(np.max(np.abs(q[:5] - self.last[:5])))
			yield from self._move(
				q, max(p.prescan_move_s, MIN_JERK_PEAK_RATIO * dq / p.prescan_peak_speed_rad_s), "approach"
			)
			yield from self._hold_servo(q, p.prescan_hold_s, "approach")
			if not self.planner.after_prescan_view(self, k):
				break

	def _planned_present(self, q_lift: np.ndarray):
		"""Planner stages 3-4: the held-brick view at the lift pose, then the planner's whole-arm present poses (each
		carried to and held), then the label picks the bowl."""
		p = self.p
		self.planner.after_grasp(self)
		while not self.lost:
			q = self.planner.next_present_pose(self, q_lift)
			if q is None:
				break
			q = np.asarray(q, dtype=np.float64).copy()
			q[5] = p.hold_g
			secs = self._carry_secs(q, p.show_move_s, p.present_peak_speed_rad_s)
			yield from self._carry(self._move(q, secs, "show_move"))
			yield from self._carry(self._hold_servo(q, p.present_hold_s, "show_hold"))
			if not self.lost:
				self.planner.after_present(self)
		if self.lost:
			return
		# back to the lift pose before the handoff, so the handoff move starts where it always does
		secs = self._carry_secs(q_lift, p.show_move_s, p.present_peak_speed_rad_s)
		yield from self._carry(self._move(q_lift, secs, "show_move"))
		self.label = self.planner.label()
		if self.label is None:
			self.aborted = "inspection_unknown"
		elif self.label not in self.bowls:
			raise ValueError(f"planner returned label {self.label!r}; bowls has {sorted(self.bowls)}")
		else:
			self.bowl_xy = self.bowls[self.label].copy()

	def _inspect_now(self, flex: float, roll_deg: float) -> None:
		"""One inspection of the current state at show pose (``flex``, ``roll_deg``); the first KNOWN label sticks."""
		res = self.inspect(flex, roll_deg, self._frame_idx)
		self.inspections.append(
			{
				"frame": int(self._frame_idx),
				"flex": float(flex),
				"roll_deg": float(roll_deg),
				"status": res.status,
				"label": res.label,
				"visible_faces": list(res.visible_faces),
				"observed_faces": list(res.observed_faces),
				"reason": res.reason,
			}
		)
		if self.label is None and res.status == "KNOWN":
			if res.label not in self.bowls:
				raise ValueError(f"inspector returned KNOWN label {res.label!r}; bowls has {sorted(self.bowls)}")
			self.label = res.label

	def _handoff(self):
		p = self.p
		q_hand = np.array([*HANDOFF_Q, p.hold_g])
		yield from self._carry(self._move(q_hand, self._carry_secs(q_hand, p.handoff_move_s), "handoff"))
		yield from self._carry(self._hold_servo(q_hand, p.handoff_hold_s, "handoff"))

	def _go_home(self):
		q_rest = self.q_start.copy()
		q_rest[5] = self.p.open_g
		p = self.p
		dq = float(np.max(np.abs(np.clip(q_rest, self._lo, self._hi)[:5] - self.last[:5])))
		# Long enough to stay under the speed the actuators can reach, short enough to stay one unchunked segment.
		# Rounded up to whole frames so the frame-quantised move still stays under the peak speed.
		secs = min(
			max(p.return_s, math.ceil(MIN_JERK_PEAK_RATIO * dq / p.return_peak_speed_rad_s * FPS) / FPS),
			p.max_move_chunk_s,
		)
		yield from self._move(q_rest, secs, "return")
		yield from self._hold(self.p.final_hold_s, "return")

	def _show_pose_reached(self, q_show: np.ndarray) -> bool:
		"""The measured show-pose criterion: wrist flex and roll within tolerance and the arm still."""
		p = self.p
		q = self.obs["q"]
		speed = float(np.max(np.abs(q[:5] - self._prev_q[:5]))) * FPS
		return bool(
			abs(q[3] - q_show[3]) < p.show_flex_tol
			and abs(q[4] - q_show[4]) < math.radians(p.show_roll_tol_deg)
			and speed < p.show_still_speed_rad_s
		)

	def _show_hold(self, q_show: np.ndarray):
		"""Present the brick: hold ``q_show`` (with the joint-error integral) until the measured pose has been
		inside tolerance and still for ``show_hold_s``; the timer starts only once it is."""
		p = self.p
		corr = np.zeros(5)
		held = 0
		need = self._frames(p.show_hold_s)
		for _ in range(self._frames(p.show_hold_timeout_s)):
			held = held + 1 if self._show_pose_reached(q_show) else 0
			if held >= need:
				return
			err = q_show[:5] - self.obs["q"][:5]
			corr = np.clip(corr + p.carry_joint_gain * err, -p.carry_joint_max_corr, p.carry_joint_max_corr)
			q = q_show.copy()
			q[:5] += corr
			yield q, "show_hold"

	def _carry(self, frames):
		"""Pass carry frames through, watching for a lost brick (no recovery: the episode just ends)."""
		for item in frames:
			if self.lost:
				return
			yield item
			self._no_dual = 0 if self._dual() else self._no_dual + 1
			if self._no_dual >= self.p.lost_frames:
				self.lost = True

	def _release_xy(self) -> np.ndarray:
		"""Release point: the bowl centre, or the point inside the bowl farthest from any clutter brick."""
		p = self.p
		m, d = self.cell.model, self.cell.data
		clutter = []
		for b in range(m.nbody):
			name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
			if name.startswith("clutter_"):
				clutter.append(d.xpos[b, :2].copy())
		if not clutter:
			return self.bowl_xy.copy()
		half_diag = 0.5 * math.hypot(self.size[0], self.size[1])
		r_max = max(BOWL_R - half_diag - p.bowl_margin_m, 0.0)
		best, best_score = self.bowl_xy.copy(), -np.inf
		for r in np.linspace(0.0, r_max, p.release_grid_radii):
			for a in np.linspace(0.0, 2 * math.pi, p.release_grid_angles, endpoint=False):
				c = self.bowl_xy + r * np.array([math.cos(a), math.sin(a)])
				dmin = min(float(np.linalg.norm(c - x)) for x in clutter)
				score = min(dmin, p.clutter_clearance_m + half_diag) - p.release_off_centre_weight * r
				if score > best_score:
					best, best_score = c, score
		return best

	def _place(self, q_lift: np.ndarray):
		p = self.p
		roll = p.place_roll
		rxy = self._release_xy()
		q = self._ik(np.array([rxy[0], rxy[1], p.transfer_z_m]), q_lift, roll)
		q[5] = p.hold_g
		yield from self._carry(self._move(q, self._carry_secs(q, p.transfer_s), "place"))
		if self.lost:
			return
		dz = self.obs["brick_pos"][2] - self.obs["tcp"][2]
		z_tcp = (
			bowl_floor_z(self.kin.model, self.kin.data, self.bowl_xy) + p.release_clearance_m + self.size[2] / 2.0 - dz
		)
		q = self._ik(np.array([rxy[0], rxy[1], z_tcp]), q, roll)
		q[5] = p.hold_g
		yield from self._carry(self._move(q, self._carry_secs(q, p.place_descend_s), "place"))
		if self.lost:
			return
		# close the loop on the release point: servo the gripper frame onto it before opening
		yield from self._carry(self._servo(np.array([rxy[0], rxy[1], z_tcp]), roll, "place", p.place_servo_tol_m))
		if self.lost:
			return
		q_open = self.last.copy()
		q_open[5] = p.open_g
		yield from self._move(q_open, p.open_ramp_s, "place")
		still = 0
		for _ in range(self._frames(p.settle_timeout_s)):
			v = float(np.linalg.norm(self.obs["brick_vel"]))
			still = still + 1 if v < p.settle_speed_mps else 0
			if still >= p.settle_frames:
				break
			yield self.last.copy(), "place"
		q_up = self._ik(self.obs["tcp"] + np.array([0.0, 0.0, p.retreat_dz_m]), self.last, roll)
		q_up[5] = p.open_g
		yield from self._move(q_up, p.retreat_s, "place")


def assess_feasibility(
	cell: Cell,
	q_start: np.ndarray,
	bowl_xy: tuple[float, float] | None,
	params: ExpertParams,
	brick_xy: tuple[float, float],
	brick_yaw_deg: float,
) -> str | None:
	"""Kinematic-only check before an episode: a reason string when the layout is shown unreachable, else None."""
	if not params.pre_narrow_open:
		return None
	# the target bowl plays no part in the kinematic check
	expert = ClosedLoopExpert(cell, q_start, (0.0, 0.0) if bowl_xy is None else bowl_xy, params)
	yaw_rad = math.radians(brick_yaw_deg)
	expert.obs = {
		"q": np.asarray(q_start, dtype=np.float64),
		"brick_pos": np.array([brick_xy[0], brick_xy[1], cell.layout.brick_size[2] / 2.0 + 0.0005], dtype=np.float64),
		"brick_quat": np.array([0.0, 0.0, math.sin(yaw_rad / 2.0), math.cos(yaw_rad / 2.0)], dtype=np.float64),
	}
	cands = expert.grasp_candidates(open_g=params.open_g)
	if any(c["clearance_m"] >= 0.0 for c in cands):
		return None
	for g in params.open_ladder:
		if g >= params.open_g - 1e-6:
			continue
		narrow = [c for c in expert.grasp_candidates(open_g=g) if c["gap_ok"]]
		if any(c["clearance_m"] >= 0.0 for c in narrow):
			return None
	return "grasp_blocked_by_environment"


def run_closed_loop_episode(
	cell: Cell,
	params: ExpertParams | None = None,
	*,
	q_start: np.ndarray,
	bowl_xy: tuple[float, float] | None = None,
	brick_xy: tuple[float, float],
	brick_yaw_deg: float,
	non_ideal: NonIdealityCfg | None = None,
	bowls: Mapping[str, tuple[float, float]] | None = None,
	inspect: InspectFn | None = None,
	next_pose: NextPoseFn | None = None,
	planner: EpisodePlanner | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict]:
	"""Run one closed-loop episode on ``cell``; returns (telemetry, phases, info).

	Give ``bowl_xy`` for a fixed target, or ``inspect`` (or ``planner``) and ``bowls`` to choose it from the label (see
	:class:`ClosedLoopExpert`); ``info["target_bowl_xy"]`` is the bowl the expert placed into (None if it never chose)."""
	params = params or ExpertParams()
	episode = PhysicsEpisode(cell, q_start, brick_xy=brick_xy, brick_yaw_deg=brick_yaw_deg, non_ideal=non_ideal)
	infeasible_reason = assess_feasibility(cell, q_start, bowl_xy, params, brick_xy, brick_yaw_deg)
	expert = None
	phases: list[int] = []
	targets: list[np.ndarray] = []
	if infeasible_reason is None:
		expert = ClosedLoopExpert(
			cell, q_start, bowl_xy, params, bowls=bowls, inspect=inspect, next_pose=next_pose, planner=planner
		)
		while episode.n_frames < params.max_frames:
			target, phase = expert.act(observe(cell, episode))
			if expert.done:
				break
			episode.step(target)
			targets.append(target)
			phases.append(PHASES.index(phase))
	tel = episode.telemetry()
	ph = np.array(phases, dtype=np.uint8)
	info = episode_diagnostics(tel, ph, params)
	blocked = expert.blocked_frames if expert is not None else []
	info.update(
		n_frames=len(ph),
		grasp_retries=int(expert.grasp_retries) if expert else 0,
		block_replans=int(expert.block_replans) if expert else 0,
		brick_lost=bool(expert.lost) if expert else False,
		aborted=expert.aborted if expert else infeasible_reason,
		infeasible_reason=infeasible_reason,
		truncated=bool(expert is not None and not expert.done and episode.n_frames >= params.max_frames),
		assisted_grasp=cell.assisted_grasp,
		grasp_geometry=dict(expert.grasp_info) if expert else {},
		jaw_env_contact_max_N=float(max(episode.jaw_env_force, default=0.0)),
		gripper_blocked_frames=len(blocked),
		gripper_first_blocked_frame=(blocked[0] if blocked else None),
		gripper_block_direction=int(expert.gripper_block_direction) if expert else 0,
		_targets=np.array(targets) if targets else np.zeros((0, 6), dtype=np.float64),
	)
	if inspect is not None or planner is not None:
		chosen = expert.bowl_xy if expert is not None else None
		info.update(
			inspection_label=expert.label if expert else None,
			inspections=list(expert.inspections) if expert else [],
			target_bowl_xy=None if chosen is None else [float(v) for v in chosen],
		)
	return tel, ph, info
