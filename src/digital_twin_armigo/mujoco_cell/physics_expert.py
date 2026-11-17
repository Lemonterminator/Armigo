"""Pure-physics pick → show → handoff → place episodes on the MuJoCo cell: a per-frame recorder and a closed-loop
expert.

``PhysicsEpisode`` owns the 240 Hz stepping and the per-frame telemetry the audited evaluator reads (the open-loop
benchmark in ``scripts/bench_cross_engine.py`` replays its joint targets through it). ``ClosedLoopExpert`` reads
privileged simulator state every frame and returns the next joint target; it acts only through that target, which
goes through the same :class:`~digital_twin_armigo.physics.ActuatorCommandFilter` as the open-loop plan.

Privileged inputs the expert reads (never writes): arm joint positions, gripper-frame pose, brick pose and velocity,
per-jaw contact normal forces, the target bowl centre, the clutter bricks' positions (visible to the cameras; used
for the release point and the grasp clearance check), and the compiled model's geometry (jaw collision geoms, bowl
floor, bowls / clutter for clearance). IK and forward kinematics run on a private ``MjData``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..constants import BOWL_FLOOR_TOP_M, BOWL_R, FPS, HANDOFF_Q, PHASES
from ..physics import DEFAULT_PHYSICS, ActuatorCommandFilter, NonIdealityCfg
from ._mj import mujoco
from .cell import Cell
from .gripper_state import GripperMonitor, GripperState

PHYSICS_HZ = 240
SUBSTEPS = PHYSICS_HZ // FPS  # 8 substeps @ 240 Hz per 30 Hz frame
WARMUP_STEPS = 12


def jaw_env_contact_force(cell: Cell) -> float:
	"""Peak contact force (N) between a jaw geom and anything that is neither the robot nor the target brick
	(table, bowls, clutter). Reads the current contacts only. The evaluator's unintended-collision check excludes
	the jaws, so this is the only record of jaws pressing on the environment."""
	jaws = cell.fixed_jaw_geoms | cell.moving_jaw_geoms
	brick = {int(g) for g in cell.brick_geoms}
	peak = 0.0
	f = np.zeros(6)
	for cid in range(cell.data.ncon):
		con = cell.data.contact[cid]
		g1, g2 = int(con.geom1), int(con.geom2)
		other = g2 if g1 in jaws else (g1 if g2 in jaws else -1)
		if other < 0 or other in cell.robot_all_geoms or other in brick:
			continue
		mujoco.mj_contactForce(cell.model, cell.data, cid, f)
		peak = max(peak, float(np.linalg.norm(f[:3])))
	return peak


class PhysicsEpisode:
	"""Reset the cell for one pure-physics episode, step it one 30 Hz frame at a time and log telemetry.

	``telemetry()`` returns the arrays the audited evaluator consumes (see ``_evaluate_mj_rollout``): one row per
	frame, sampled after the frame's substeps.
	"""

	def __init__(
		self,
		cell: Cell,
		q_start: np.ndarray,
		brick_xy: tuple[float, float] = (0.26, 0.0),
		brick_yaw_deg: float = 0.0,
		non_ideal: NonIdealityCfg | None = None,
	) -> None:
		self.cell = cell
		cell.model.opt.timestep = 1.0 / PHYSICS_HZ
		cell.set_joints(q_start)
		cell.data.qvel[:] = 0.0
		yaw_rad = math.radians(brick_yaw_deg)
		b_quat = (0.0, 0.0, math.sin(yaw_rad / 2.0), math.cos(yaw_rad / 2.0))
		cell.set_brick_pose(
			np.array([brick_xy[0], brick_xy[1], cell.layout.brick_size[2] / 2.0 + 0.0005]), np.array(b_quat)
		)
		cell.data.ctrl[:6] = q_start
		for _ in range(WARMUP_STEPS):
			mujoco.mj_step(cell.model, cell.data)

		self.filt = ActuatorCommandFilter(non_ideal, n_joints=6, fps=FPS) if non_ideal is not None else None
		if self.filt is not None:
			self.filt.reset(q_start)
		# Peak jaw ↔ environment contact force per frame (end-of-frame contacts); kept out of ``telemetry()``,
		# whose keys are fixed by the evaluator contract.
		self.jaw_env_force: list[float] = []
		self._logs: dict[str, list] = {
			k: []
			for k in (
				"q_cmd",
				"qpos",
				"tcp_pos",
				"ee_quat",
				"brick_pos",
				"brick_quat",
				"brick_vel",
				"brick_ang_vel",
				"jaw_forces",
				"pinch_normal",
				"pinch_tangential",
				"unintended_forces",
				"substep_unintended",
				"actuator_torques",
			)
		}

	@property
	def n_frames(self) -> int:
		return len(self._logs["q_cmd"])

	def step(self, target: np.ndarray) -> None:
		"""Advance one 30 Hz frame towards the 6-D joint position ``target`` (through the command filter)."""
		cell = self.cell
		q_cmd = self.filt.step_command(target, cell.joints()) if self.filt is not None else target
		sub = cell.step_with_telemetry(q_cmd, substeps=SUBSTEPS)
		log = self._logs
		log["q_cmd"].append(q_cmd.copy())
		log["qpos"].append(cell.joints())
		log["tcp_pos"].append(cell.tcp_pos())
		wxyz = cell.data.xquat[cell.grip_body]
		log["ee_quat"].append(np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]], dtype=np.float64))
		log["brick_pos"].append(cell.brick_pos())
		log["brick_quat"].append(cell.brick_quat_xyzw())
		lv, av = cell.brick_vel()
		log["brick_vel"].append(lv)
		log["brick_ang_vel"].append(av)
		log["jaw_forces"].append(sub["jaw_total"][-1])
		log["pinch_normal"].append(sub["jaw_normal"].mean(axis=1))
		log["pinch_tangential"].append(sub["jaw_tangential"].mean(axis=1))
		log["unintended_forces"].append(float(sub["unintended"][-1]))
		log["substep_unintended"].append(sub["unintended"])
		log["actuator_torques"].append(sub["torques"][np.argmax(np.abs(sub["torques"]), axis=0), np.arange(6)])
		self.jaw_env_force.append(jaw_env_contact_force(cell))
		observer = getattr(cell, "frame_observer", None)
		if observer is not None:
			observer(self)

	def last(self, key: str):
		"""The most recent logged row of ``key``."""
		return self._logs[key][-1]

	def telemetry(self) -> dict[str, np.ndarray]:
		out = {"t": np.arange(self.n_frames, dtype=np.float64) / float(FPS)}
		out.update({k: np.array(v) for k, v in self._logs.items()})
		return out


# --- closed-loop expert -----------------------------------------------------------------------------------------


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


class _Kinematics:
	"""A private ``MjData`` on the cell's model for IK and forward kinematics: the simulation state is never
	touched. Exposes the attributes the solvers in :mod:`.ik` read."""

	def __init__(self, cell: Cell) -> None:
		self.model = cell.model
		self.data = mujoco.MjData(cell.model)
		self.data.qpos[:] = cell.data.qpos
		mujoco.mj_forward(self.model, self.data)
		for name in ("site", "qadr", "dofadr", "jnt_range", "jaw_body", "grip_body"):
			setattr(self, name, getattr(cell, name))

	def site_pose(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		"""Gripper-frame position and rotation matrix at joint angles ``q``."""
		self.data.qpos[self.qadr] = q
		mujoco.mj_kinematics(self.model, self.data)
		return self.data.site_xpos[self.site].copy(), self.data.site_xmat[self.site].reshape(3, 3).copy()


def _geom_points(model, gid: int) -> np.ndarray:
	"""Vertices (mesh) or bounding-box corners (primitives) of geom ``gid`` in its own frame."""
	t = model.geom_type[gid]
	s = model.geom_size[gid]
	signs = np.array([(x, y, z) for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)])
	if t == mujoco.mjtGeom.mjGEOM_MESH:
		mid = model.geom_dataid[gid]
		a, n = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
		return model.mesh_vert[a : a + n].astype(np.float64)
	if t == mujoco.mjtGeom.mjGEOM_BOX:
		return signs * s
	if t == mujoco.mjtGeom.mjGEOM_SPHERE:
		return signs * s[0]
	if t == mujoco.mjtGeom.mjGEOM_CAPSULE:
		return signs * np.array([s[0], s[0], s[1] + s[0]])
	raise ValueError(f"jaw geom {gid} has unsupported type {int(t)} for the pad-geometry derivation")


def jaw_pad_faces(cell: Cell, kin: _Kinematics, q: np.ndarray, band: tuple[float, float]) -> tuple[float, float]:
	"""Inner faces of the fixed and moving jaw along the gripper frame's z axis (fixed → moving jaw), metres.

	Computed from the compiled collision geoms at joint angles ``q``, keeping only geometry whose gripper-frame x
	(the approach axis, pointing down) lies in ``band`` — the slab the brick occupies. Returns the largest z of the
	fixed jaw and the smallest z of the moving jaw (``inf`` when the moving jaw is clear of the slab).
	"""
	pos, rot = kin.site_pose(q)
	m, d = kin.model, kin.data
	faces = []
	for geoms, pick in ((cell.fixed_jaw_geoms, np.max), (cell.moving_jaw_geoms, np.min)):
		if not geoms:
			raise ValueError("the compiled model has no fixed / moving jaw collision geoms (groups 3 and 4)")
		pts = []
		for gid in sorted(geoms):
			w = d.geom_xpos[gid] + _geom_points(m, gid) @ d.geom_xmat[gid].reshape(3, 3).T
			pts.append((w - pos) @ rot)
		p = np.vstack(pts)
		p = p[(p[:, 0] >= band[0]) & (p[:, 0] <= band[1])]
		faces.append(float(pick(p[:, 2])) if len(p) else (np.inf if pick is np.min else -np.inf))
	return faces[0], faces[1]


def jaw_sweep_low_point(cell: Cell, kin: _Kinematics, q: np.ndarray, g_from: float, g_to: float, samples: int) -> float:
	"""Deepest point of either jaw below the gripper frame (along its approach axis) while the gripper closes from
	``g_from`` to ``g_to`` at arm pose ``q``, from the compiled collision geoms; metres."""
	m, d = kin.model, kin.data
	geoms = sorted(cell.fixed_jaw_geoms | cell.moving_jaw_geoms)
	if not geoms:
		raise ValueError("the compiled model has no jaw collision geoms (groups 3 and 4)")
	low = -np.inf
	qq = np.asarray(q, dtype=np.float64).copy()
	for g in np.linspace(g_from, g_to, samples):
		qq[5] = g
		pos, rot = kin.site_pose(qq)
		for gid in geoms:
			w = d.geom_xpos[gid] + _geom_points(m, gid) @ d.geom_xmat[gid].reshape(3, 3).T
			low = max(low, float(((w - pos) @ rot)[:, 0].max()))
	return low


def bowl_floor_z(model, data, bowl_xy: np.ndarray) -> float:
	"""Top of the bowl inner flat floor (metres): constant ``BOWL_FLOOR_TOP_M`` verified against raycasts."""
	return BOWL_FLOOR_TOP_M


def _quat_xyzw_to_mat(q: np.ndarray) -> np.ndarray:
	x, y, z, w = q
	return np.array(
		[
			[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
			[2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
			[2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
		]
	)


MIN_JERK_PEAK_RATIO = 1.875  # peak / mean speed of a min-jerk profile


def _min_jerk(n: int) -> np.ndarray:
	s = np.linspace(0.0, 1.0, n)
	return 10 * s**3 - 15 * s**4 + 6 * s**5


class ClosedLoopExpert:
	"""Per-frame pick → show → handoff → place policy on privileged simulator state.

	``act(obs)`` takes the current observation (``q``, ``tcp``, ``brick_pos``, ``brick_quat``, ``brick_vel``,
	``jaw_normal``) and returns ``(joint_target, phase)``; ``done`` turns True when the episode script has ended.
	"""

	def __init__(self, cell: Cell, q_start: np.ndarray, bowl_xy: tuple[float, float], params: ExpertParams) -> None:
		self.cell = cell
		self.p = params
		self.q_start = np.asarray(q_start, dtype=np.float64).copy()
		self.bowl_xy = np.asarray(bowl_xy, dtype=np.float64)
		self.kin = _Kinematics(cell)
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
		target = np.clip(target, self._lo, self._hi)
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
		rot_b = _quat_xyzw_to_mat(self.obs["brick_quat"])
		axes = [(rot_b[:2, 1], 0.5 * self.size[1])]
		if abs(self.size[0] - self.size[1]) < 1e-6:
			axes.append((rot_b[:2, 0], 0.5 * self.size[0]))
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
		for axis, half_w in axes:
			axis = axis / max(np.linalg.norm(axis), 1e-9)
			for sign in (1.0, -1.0):
				align = sign * (zs @ axis)
				i = int(np.argmax(align))
				if align[i] < p.roll_align_min:
					continue
				c = self._grasp_candidate(float(rolls[i]), sign * axis, half_w, q_hover, open_g=g_open)
				c["score"] = min(c["clearance_m"], p.clearance_cap_m) - p.roll_penalty_m_per_rad * abs(c["roll"])
				cands.append(c)
		cands.sort(key=lambda c: -c["score"])
		return cands

	def _grasp_plan(self, open_g: float | None = None) -> tuple[np.ndarray, float, np.ndarray]:
		"""Gripper-frame target, wrist roll and the grasp joint angles for the brick's current pose."""
		p = self.p
		cands = self.grasp_candidates(open_g=open_g)
		if not cands:
			raise RuntimeError("no wrist roll lines the jaws up with the brick")
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
		p = self.p
		yield from self._hold(p.rest_s, "rest")
		yield from self._move(np.array([*p.ready_q, p.open_g]), p.ready_move_s, "approach")
		attempts = 0
		clamped = False
		phase = "approach"
		while True:
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
			q_closed = self.last.copy()
			q_closed[5] = p.hold_g
			blocked = False
			for frame in self._move(q_closed, p.close_s, "grasp"):
				if self.gripper_state is GripperState.BLOCKED:
					blocked = True
					break
				yield frame
			still = 0
			if not blocked:
				for _ in range(p.clamp_timeout_frames):
					if self.gripper_state is GripperState.BLOCKED:
						blocked = True
						break
					f = self.obs["jaw_normal"]
					v = float(np.linalg.norm(self.obs["brick_vel"]))
					still = still + 1 if (min(f) >= p.clamp_force_n and v < p.clamp_still_speed_mps) else 0
					if still >= p.clamp_still_frames:
						clamped = True
						break
					yield self.last.copy(), "grasp"
			if blocked:
				q_back = self.last.copy()
				q_back[5] = float(
					np.clip(self.obs["q"][5] - self._last_block_dir * p.block_backoff_rad, self._lo[5], self._hi[5])
				)
				yield q_back, "grasp"
				self._blocked_poses.append((self.obs["q"][:5].copy(), float(self.obs["q"][5])))
				q_up = self._ik(self.obs["tcp"] + np.array([0.0, 0.0, p.hover_z_m - self.z_grasp]), self.last, roll)
				q_up[5] = q_back[5]
				yield from self._move(q_up, p.descend_s, "grasp")
				self.block_replans += 1
				next_open = self._next_clear_opening(self._active_open_g)
				if self.block_replans > p.max_block_replans or next_open is None:
					self.aborted = "gripper_blocked"
					yield from self._go_home()
					return
				self._active_open_g = next_open
				phase = "grasp"
				continue
			if clamped or attempts >= p.max_grasp_retries:
				break
			attempts += 1
			self.grasp_retries = attempts
			phase = "grasp"
			q_open = self.last.copy()
			q_open[5] = self._active_open_g
			yield from self._move(q_open, p.reopen_s, "grasp")
			q_up = self._ik(self.obs["tcp"] + np.array([0.0, 0.0, p.hover_z_m - self.z_grasp]), self.last, roll)
			q_up[5] = self._active_open_g
			yield from self._move(q_up, p.descend_s, "grasp")
		if not clamped:
			# no verified clamp after the retries: do not lift, go home and let the episode record the miss
			self.aborted = "no_clamp"
			yield from self._go_home()
			return
		lift_target = np.array([target[0], target[1], p.lift_z_m]) + corr
		q_lift = self._ik(lift_target, self.last, roll)
		q_lift[5] = p.hold_g
		yield from self._carry(self._move(q_lift, self._carry_secs(q_lift, p.lift_s), "lift"))
		for wr in p.show_roll_deg:
			if self.lost:
				break
			q_show = q_lift.copy()
			q_show[3] = p.show_flex
			q_show[4] = math.radians(wr)
			secs = self._carry_secs(q_show, p.show_move_s, p.show_peak_speed_rad_s)
			yield from self._carry(self._move(q_show, secs, "show_move"))
			yield from self._carry(self._show_hold(q_show))
		if not self.lost:
			q_hand = np.array([*HANDOFF_Q, p.hold_g])
			yield from self._carry(self._move(q_hand, self._carry_secs(q_hand, p.handoff_move_s), "handoff"))
			yield from self._carry(self._hold_servo(q_hand, p.handoff_hold_s, "handoff"))
		if not self.lost:
			yield from self._place(q_lift)
		if self.lost:
			self.aborted = "brick_lost"
		yield from self._go_home()

	def _go_home(self):
		q_rest = self.q_start.copy()
		q_rest[5] = self.p.open_g
		yield from self._move(q_rest, self.p.return_s, "return")
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
	bowl_xy: tuple[float, float],
	params: ExpertParams,
	brick_xy: tuple[float, float],
	brick_yaw_deg: float,
) -> str | None:
	"""Kinematic-only check before an episode: a reason string when the layout is shown unreachable, else None."""
	if not params.pre_narrow_open:
		return None
	expert = ClosedLoopExpert(cell, q_start, bowl_xy, params)
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


def _observe(cell: Cell, episode: PhysicsEpisode) -> dict[str, np.ndarray | float]:
	q = cell.joints()
	lv, _ = cell.brick_vel()
	jaw = episode.last("pinch_normal") if episode.n_frames else np.zeros(2)
	effort_limit = (cell.physics or DEFAULT_PHYSICS).gripper_actuator.effort_limit
	gripper_cmd = float(episode.last("q_cmd")[5]) if episode.n_frames else float(q[5])
	effort_frac = abs(float(episode.last("actuator_torques")[5])) / effort_limit if episode.n_frames else 0.0
	return {
		"q": q,
		"tcp": cell.tcp_pos(),
		"brick_pos": cell.brick_pos(),
		"brick_quat": cell.brick_quat_xyzw(),
		"brick_vel": lv,
		"jaw_normal": np.asarray(jaw, dtype=np.float64),
		"gripper_cmd": gripper_cmd,
		"effort_frac": effort_frac,
	}


# The evaluator's release rule (``metrics.evaluate_episode_health``: ``opening = place_idx[q_cmd_a[place_idx, 5] >
# hold_baseline + 0.02]``): release starts at the first ``place`` frame whose gripper command exceeds the minimum
# gripper command in ``place`` by this much (rad).
EVALUATOR_RELEASE_DELTA_RAD = 0.02


def show_hold_measured(tel: dict[str, np.ndarray], phases: np.ndarray, params: ExpertParams) -> list[float]:
	"""For each show pose, the longest run (s) of show-phase frames in which the measured wrist flex and roll are
	within tolerance of that pose and every arm joint moves slower than ``show_still_speed_rad_s``."""
	ph = np.asarray(phases)
	q = tel["qpos"]
	speed = np.zeros(len(q))
	if len(q) > 1:
		speed[1:] = np.max(np.abs(np.diff(q[:, :5], axis=0)), axis=1) * FPS
	show = np.isin(ph, [PHASES.index("show_move"), PHASES.index("show_hold")])
	out = []
	for wr in params.show_roll_deg:
		ok = (
			show
			& (np.abs(q[:, 3] - params.show_flex) < params.show_flex_tol)
			& (np.abs(q[:, 4] - math.radians(wr)) < math.radians(params.show_roll_tol_deg))
			& (speed < params.show_still_speed_rad_s)
		)
		best = run = 0
		for flag in ok:
			run = run + 1 if flag else 0
			best = max(best, run)
		out.append(best / float(FPS))
	return out


def episode_diagnostics(tel: dict[str, np.ndarray], phases: np.ndarray, params: ExpertParams) -> dict:
	"""``grasp_push_mm``, ``jaw_normal_at_lift_N``, ``release_peak_speed_mps`` and ``show_hold_measured_s`` from an
	episode's telemetry (``nan`` when the episode never reached the phase a metric needs)."""
	ph = np.asarray(phases)
	g = np.flatnonzero(ph == PHASES.index("grasp"))
	lift = np.flatnonzero(ph == PHASES.index("lift"))
	place = np.flatnonzero(ph == PHASES.index("place"))
	out = {"grasp_push_mm": math.nan, "jaw_normal_at_lift_N": math.nan, "release_peak_speed_mps": math.nan}
	if len(g) and len(lift):
		d = tel["brick_pos"][lift[0], :2] - tel["brick_pos"][g[0], :2]
		out["grasp_push_mm"] = float(np.linalg.norm(d) * 1000.0)
		out["jaw_normal_at_lift_N"] = float(np.min(tel["pinch_normal"][lift[0]]))
	if len(place):
		g_cmd = tel["q_cmd"][place, 5]
		opening = place[g_cmd > g_cmd.min() + EVALUATOR_RELEASE_DELTA_RAD]
		if len(opening):
			out["release_peak_speed_mps"] = float(np.linalg.norm(tel["brick_vel"][opening[0] :], axis=1).max())
	out["show_hold_measured_s"] = show_hold_measured(tel, ph, params)
	return out


def run_closed_loop_episode(
	cell: Cell,
	params: ExpertParams | None = None,
	*,
	q_start: np.ndarray,
	bowl_xy: tuple[float, float],
	brick_xy: tuple[float, float],
	brick_yaw_deg: float,
	non_ideal: NonIdealityCfg | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict]:
	"""Run one closed-loop episode on ``cell``; returns (telemetry, phases, info)."""
	params = params or ExpertParams()
	episode = PhysicsEpisode(cell, q_start, brick_xy=brick_xy, brick_yaw_deg=brick_yaw_deg, non_ideal=non_ideal)
	infeasible_reason = assess_feasibility(cell, q_start, bowl_xy, params, brick_xy, brick_yaw_deg)
	if infeasible_reason is not None:
		tel = episode.telemetry()
		ph = np.array([], dtype=np.uint8)
		info = episode_diagnostics(tel, ph, params)
		info.update(
			n_frames=0,
			grasp_retries=0,
			block_replans=0,
			brick_lost=False,
			aborted=infeasible_reason,
			infeasible_reason=infeasible_reason,
			truncated=False,
			assisted_grasp=cell.assisted_grasp,
			grasp_geometry={},
			jaw_env_contact_max_N=0.0,
			gripper_blocked_frames=0,
			gripper_first_blocked_frame=None,
			gripper_block_direction=0,
			_targets=np.zeros((0, 6), dtype=np.float64),
		)
		return tel, ph, info
	expert = ClosedLoopExpert(cell, q_start, bowl_xy, params)
	phases: list[int] = []
	targets: list[np.ndarray] = []
	while episode.n_frames < params.max_frames:
		target, phase = expert.act(_observe(cell, episode))
		if expert.done:
			break
		episode.step(target)
		targets.append(target)
		phases.append(PHASES.index(phase))
	truncated = not expert.done and episode.n_frames >= params.max_frames
	tel = episode.telemetry()
	ph = np.array(phases, dtype=np.uint8)
	info = episode_diagnostics(tel, ph, params)
	info.update(
		n_frames=len(ph),
		grasp_retries=int(expert.grasp_retries),
		block_replans=int(expert.block_replans),
		brick_lost=bool(expert.lost),
		aborted=expert.aborted,
		infeasible_reason=None,
		truncated=bool(truncated),
		assisted_grasp=cell.assisted_grasp,
		grasp_geometry=dict(expert.grasp_info),
		jaw_env_contact_max_N=float(max(episode.jaw_env_force, default=0.0)),
		gripper_blocked_frames=len(expert.blocked_frames),
		gripper_first_blocked_frame=(expert.blocked_frames[0] if expert.blocked_frames else None),
		gripper_block_direction=int(expert.gripper_block_direction),
		_targets=np.array(targets),
	)
	return tel, ph, info
