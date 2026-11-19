"""Trusted MuJoCo skills and skill-boundary checkpoints for the six-face graph."""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import ClassVar

import numpy as np

from ..benchmarks import SENSITIVITY_MATRIX
from ..camera import CameraPose, look_from
from ..constants import BOWL_R, FPS, PHASES, REST_U, SO101_MJCF_DIR
from ..inspection import assert_no_defect_leakage
from ..joints import normalized_to_rad
from ..metrics import evaluate_episode_health, motion_smoothness
from ..mujoco_cell import Cell
from ..mujoco_cell._mj import mujoco
from ..mujoco_cell.episode import PhysicsEpisode, observe
from ..mujoco_cell.handeye_rig import calibrate_layout
from ..mujoco_cell.inspection_planning import path_min_z
from ..mujoco_cell.inspection_views import capture_view, detector_crop
from ..mujoco_cell.layout import sample_layout
from ..mujoco_cell.orbit_scan import OrbitParams, plan_orbit_scan
from ..mujoco_cell.physics_expert import MIN_JERK_PEAK_RATIO, ExpertParams, _min_jerk
from ..mujoco_cell.table_relocation import RelocateExpert, RelocateParams, plan_bottom_view, plan_table_scan
from ..physics import NonIdealityCfg, load_physics_spec
from .perception import PerceptionCfg, bowl_fill, bowl_filled, locate_bowls, locate_brick, yaw_error_deg
from .sensing import GripperContact
from .twin_graph import Policy, decision, select_skill

ROOT = Path(__file__).resolve().parents[3]
INITIAL_Q = np.array([*normalized_to_rad(np.array(REST_U, dtype=np.float64))[:5], ExpertParams().open_g])
STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION

PROFILE = "mujoco-close-top-1.35"
BOTTOM_MIN_PX = 400
WRIST_MIN_PX = 800
MIN_BRICK_PX = 150
MAX_TOTAL_FRAMES = 6300
# ``truth``: skills read the simulator (brick, bowls, contact forces), optionally with the brick perceived (see
# .perception). ``realistic``: skills see only what a real cell senses, see TwinMujoco.
SENSING = ("truth", "realistic")
NEUTRAL_XY = (0.25, 0.0)  # constructor target of a realistic expert; every target it uses is set later
RETRY_MOVED_M = 0.005  # a brick found this far from where it was believed has moved: its face numbering is lost
RETRY_MOVED_DEG = 3.0
BOTTOM_BORDER_PX = 40  # realistic bottom views keep the believed brick this far inside the top image


def sample_case(seed: int, positive: bool, face: int):
	if type(seed) is not int or not 0 <= seed <= 2**32 - 1 or type(positive) is not bool:
		raise ValueError("invalid fixture seed or class")
	if type(face) is not int or not 0 <= face < 6:
		raise ValueError("face must be in [0, 5]")
	raw = json.loads((ROOT / "config/sim/top_camera_joint.json").read_text(encoding="utf-8"))
	layout = sample_layout(np.random.default_rng(seed), CameraPose(**raw["camera"]), defective=positive)
	layout.top = CameraPose(0.4895033468, 0.0, 0.5130680468, 180.0, 25.0712240655, 0.0, 612.47)
	layout.front = look_from((0.20, 0.0), 0.55, 0.35, 35, focal_px=560)
	layout.defect_kind = "cross_var"
	layout.defect_face = face
	layout.defect_seed = 19_199 + seed * 100_019
	layout.clutter = []
	layout.distractors = []
	return layout


@contextmanager
def blind_geometry(cell):
	# Group 4 is disabled in both visibility render options. The model is session-local.
	original = cell.model.geom_group[cell.defect_geoms].copy()
	cell.model.geom_group[cell.defect_geoms] = 4
	try:
		yield
	finally:
		cell.model.geom_group[cell.defect_geoms] = original


@dataclass
class Checkpoint:
	id: str
	branch: int
	frame: int
	integration: np.ndarray
	expert: dict
	filter: dict
	wrist_count: int
	bottom_count: int
	views: list[dict]


# Bounded recovery variants. Each is a fixed, named, deterministic transform of the default expert parameters. The
# agent (or LLM) only ever picks a name; no number crosses that boundary. Every value must lie in RECOVERY_BOUNDS and
# every variant only slows motion down or raises the transfer height, so a variant cannot be more aggressive than the
# defaults. Keys of "expert" / "relocate" are ExpertParams / RelocateParams fields; "scan_peak_speed" is the peak
# joint speed (rad/s) of the wrist-scan moves (default SCAN_PEAK_SPEED, the value the scan used before variants).
SCAN_PEAK_SPEED = 1.0
RECOVERY_BOUNDS = MappingProxyType(
	{
		"scan_peak_speed": (0.25, SCAN_PEAK_SPEED),
		"scan_move_s": (0.6, 1.5),
		"ready_move_s": (0.6, 1.2),
		"carry_peak_speed_rad_s": (0.2, 0.5),
		"show_move_s": (0.9, 2.0),
		"transfer_s": (1.8, 3.0),
		"transfer_z_m": (0.12, 0.15),
		"place_descend_s": (1.2, 2.0),
		"retreat_s": (0.8, 1.5),
		"return_s": (1.2, 2.0),
	}
)
_SLOW_SCAN = {"scan_peak_speed": 0.35, "scan_move_s": 1.0, "ready_move_s": 1.0}
_SLOW_CARRY = {"carry_peak_speed_rad_s": 0.3, "show_move_s": 1.4, "transfer_s": 2.6}
_GENTLE_PLACE = {"place_descend_s": 1.8, "retreat_s": 1.2, "return_s": 1.8}
RECOVERY_VARIANTS = MappingProxyType(
	{
		"slow_scan": MappingProxyType(_SLOW_SCAN),
		"slow_carry": MappingProxyType(_SLOW_CARRY),
		"gentle_place": MappingProxyType(_GENTLE_PLACE),
		"high_place": MappingProxyType({**_GENTLE_PLACE, "transfer_z_m": 0.14}),
		"cautious": MappingProxyType({**_SLOW_SCAN, **_SLOW_CARRY, **_GENTLE_PLACE}),
	}
)
VARIANT_NOTES = MappingProxyType(
	{
		"slow_scan": "slower wrist-scan and ready moves, so joints stop saturating",
		"slow_carry": "slower carry and show moves",
		"gentle_place": "slower place descent, retreat and return",
		"high_place": "slower place moves with a higher approach height over the bowl",
		"cautious": "slow scan, carry and place together",
	}
)
# Failure evidence -> recommended variant. A torque stall is attributed to the skill that was running when the
# saturated run started; other physical or placement failures map by code.
STALL_VARIANT_BY_NODE = MappingProxyType(
	{"scan_wrist": "slow_scan", "present_bottom": "slow_carry", "place": "gentle_place"}
)
VARIANT_BY_CODE = MappingProxyType(
	{
		"BRICK_LOST": "slow_carry",
		"PLACEMENT_UNVERIFIED": "high_place",
		"PLACE_RETURN_PATH_BLOCKED": "high_place",
		"PARK_PATH_BLOCKED": "slow_carry",
	}
)
# Orbit-scan rings per wrist-scan attempt (first scan, then repeats and scans after a rewind): later attempts keep
# the jaws higher and the ring wider, so a repeat does not retrace a ring that already failed.
ORBIT_ATTEMPTS = (
	OrbitParams(),
	OrbitParams(tip_heights_m=(0.025, 0.04, 0.055)),
	OrbitParams(tip_heights_m=(0.04, 0.06), standoffs_m=(0.04, 0.06)),
)
RELOCATE_KEYS = frozenset({"scan_move_s"})  # RelocateParams fields a variant may set
STALL_REASONS = ("torque_stall", "tracking_lag")
STALL_SECONDS = 1.5  # metrics.evaluate_episode_health max_stall_secs
STALL_FRACTION = 0.95  # fraction of the torque limit that counts as saturated


def apply_variant(name, params, relocate):
	"""Return ``(ExpertParams, RelocateParams, scan_peak_speed)`` for variant ``name`` applied to the given defaults.
	Raises ValueError for an unknown name or a value outside RECOVERY_BOUNDS."""
	if name not in RECOVERY_VARIANTS:
		raise ValueError(f"unknown recovery variant: {name}")
	values = RECOVERY_VARIANTS[name]
	for key, value in values.items():
		lo, hi = RECOVERY_BOUNDS[key]
		if not lo <= value <= hi:
			raise ValueError(f"variant {name}: {key}={value} outside [{lo}, {hi}]")
	# transfer_z_m exists in both parameter classes: only the ExpertParams one (the bowl approach) is a variant key.
	rel = {k: v for k, v in values.items() if k in RELOCATE_KEYS}
	expert = {k: v for k, v in values.items() if k not in RELOCATE_KEYS and k != "scan_peak_speed"}
	return (
		replace(params, **expert),
		replace(relocate, **rel),
		float(values.get("scan_peak_speed", SCAN_PEAK_SPEED)),
	)


def sweep(expert, path, peak_speed, min_s, phase="approach", pause_s=None):
	"""Targets that move the arm from ``expert.last`` through the joint vectors of ``path`` without stopping at them:
	min-jerk timing along the piecewise-linear joint path (arc length measured as the largest joint change, so no
	joint exceeds ``peak_speed``), split into chunks of at most ``max_move_chunk_s`` with the expert's chunk pause
	between them, as :meth:`ClosedLoopExpert._move` does for one segment. Lasts at least ``min_s``."""
	p = expert.p
	points = [expert.last.copy(), *(np.clip(np.asarray(q, dtype=np.float64), expert._lo, expert._hi) for q in path)]
	if len(points) < 2:
		return
	# Arc length: the largest arm-joint change, plus a tiny share of the rest so a path that only moves the
	# gripper still advances to its end.
	steps = [float(np.max(np.abs(b[:5] - a[:5])) + 1e-6 * np.sum(np.abs(b - a))) for a, b in itertools.pairwise(points)]
	arc = np.concatenate([[0.0], np.cumsum(steps)])
	total = float(arc[-1])
	secs = max(min_s, MIN_JERK_PEAK_RATIO * total / peak_speed)
	chunks = max(1, math.ceil(secs / p.max_move_chunk_s))
	stack = np.array(points)

	def at(s):
		k = int(np.clip(np.searchsorted(arc, s, side="right") - 1, 0, len(steps) - 1))
		f = 0.0 if steps[k] <= 0 else (s - arc[k]) / steps[k]
		return stack[k] + min(max(f, 0.0), 1.0) * (stack[k + 1] - stack[k])

	for c in range(chunks):
		a, b = total * c / chunks, total * (c + 1) / chunks
		n = max(2, expert._frames(secs / chunks) + 1)
		for s in _min_jerk(n)[1:]:
			yield at(a + s * (b - a)), phase
		if c < chunks - 1:
			for _ in range(expert._frames(p.chunk_pause_s if pause_s is None else pause_s)):
				yield at(b), phase


def stall_start(torques, limit, fps=FPS):
	"""First frame of a continuous run of saturated arm-joint torque longer than STALL_SECONDS, else None."""
	saturated = (np.abs(np.asarray(torques)[:, :5]) >= STALL_FRACTION * limit).any(axis=1)
	run = 0
	for i, flag in enumerate(saturated):
		run = run + 1 if flag else 0
		if run / fps > STALL_SECONDS:
			return i - run + 1
	return None


def node_at(events, frame):
	"""Skill node running at active-branch ``frame``, replaying node and rewound events of the trace."""
	timeline = []
	for e in events:
		if e["kind"] == "rewound":
			timeline = [t for t in timeline if t[0] <= e["frame"]]
		elif e["kind"] == "node" and e["node"] != "recover":
			timeline.append((e["frame"], e["node"]))
	return next((node for f, node in reversed(timeline) if f <= frame), None)


def rewind_options(checkpoints, failure, stall_frame, stall_node):
	"""Ordered ``(choice, note)`` pairs for a rewind. The first entry is the deterministic default: the latest
	checkpoint taken before the stall began (a checkpoint after it keeps the stall in the health history, so it can
	never clear it) with the variant recommended for the failure, then the plain rewind of that checkpoint."""
	reasons = failure.get("reason_codes") or []
	recommended = VARIANT_BY_CODE.get(failure.get("code"))
	if stall_frame is not None and any(r in STALL_REASONS for r in reasons):
		recommended = STALL_VARIANT_BY_NODE.get(stall_node, "cautious")
	latest = list(reversed(checkpoints))
	if stall_frame is not None:
		clean = [c for c in latest if c.frame <= stall_frame]
		latest = clean + [c for c in latest if c not in clean]
	options = []
	for c in latest:
		hint = ""
		if stall_frame is not None and c.frame > stall_frame:
			hint = f"; keeps the stall that began at frame {stall_frame}"
		variants = [recommended] if recommended else []
		variants += ["cautious"] if recommended not in (None, "cautious") else []
		for v in variants:
			options.append((f"{c.id}+{v}", f"rewind to {c.id} (frame {c.frame}) with {VARIANT_NOTES[v]}{hint}"))
		options.append((c.id, f"rewind to {c.id} (frame {c.frame}) and retry with unchanged parameters{hint}"))
	return options


class ContinuousCarryExpert(RelocateExpert):
	"""Retime long carry moves along the same joint-space line."""

	smoothed_moves = 0

	def _move(self, q_to: np.ndarray, secs: float, phase: str):
		q_from = self.last.copy()
		q_to = np.clip(q_to, self._lo, self._hi)  # plan to the reachable target, as the base move does
		chunks = max(1, math.ceil(secs / self.p.max_move_chunk_s))
		if chunks == 1 or phase not in ("lift", "show_move", "place") or q_from[5] != q_to[5]:
			yield from super()._move(q_to, secs, phase)
			return

		frames = chunks * self._frames(secs / chunks) + (chunks - 1) * self._frames(self.p.chunk_pause_s)
		t = np.arange(1, frames + 1, dtype=np.float64) / frames
		ramp = 0.2
		edge = np.minimum(t, 1 - t)
		x = np.minimum(edge / ramp, 1.0)
		# Integrate a quintic velocity ramp into a constant-speed middle, then mirror at the end.
		progress = (ramp * x**4 * (2.5 - 3 * x + x * x) + np.maximum(edge - ramp, 0.0)) / (1 - ramp)
		progress[t > 0.5] = 1 - progress[t > 0.5]
		self.smoothed_moves += 1
		for s in progress[:-1]:
			yield q_from + s * (q_to - q_from), phase
		yield q_to.copy(), phase


class TwinMujoco:
	"""The six-face skills on one simulated cell.

	With ``sensing="realistic"`` the skills use only what a real cell senses, the camera images, the joint and
	gripper readings and the robot model:

	- the top camera is hand-eye calibrated before the episode, and the brick and both bowls are located in its images;
	- a carried brick is believed to move rigidly with the gripper from the grasp on;
	- planners and the detector crop see the believed brick (the simulated one is moved there while they run);
	- the expert's contact signals come from the gripper position and effort (.sensing), and brick speed reads zero;
	- a placement is checked by the top camera (depth against an empty-bowl image);
	- a failure is recovered from the current state (park or drop, relocate the brick) instead of a physics rewind;
	- the motion-health check moves to the final evaluation, which, as in ``truth``, is ground truth.

	Still from the simulator: the bowl and clutter collision geometry in the grasp clearance checks (at the true bowl
	positions, about 1 mm from the perceived ones).
	"""

	scan_peak_speed = SCAN_PEAK_SPEED
	sweep_pause_s = None  # pause between chunks of a multi-pose scan sweep; None: the expert's chunk pause
	recovery_variants: tuple = ()

	def __init__(
		self,
		layout,
		scorer,
		record,
		*,
		llm=None,
		bottom_threshold=None,
		crop_size=256,
		fault=None,
		on_frame=None,
		smooth_carry=False,
		sensing="truth",
	):
		if sensing not in SENSING:
			raise ValueError(f"sensing must be one of {SENSING}")
		self.sensing = sensing
		self.realistic = sensing == "realistic"
		self._expert_xy = NEUTRAL_XY if self.realistic else layout.brick_xy
		self.layout = layout
		self.scorer = scorer
		self.record = record
		self.llm = llm
		self.bottom_threshold = bottom_threshold
		self.crop_size = crop_size
		self.on_frame = on_frame  # Called once per simulated frame; may raise to cancel the episode.
		self.fault = fault  # Internal acceptance hook. It is absent from the public MCP API.
		self.spec = load_physics_spec()
		base = next(c for c in SENSITIVITY_MATRIX if c.name == "baseline_2x2")
		cfg = NonIdealityCfg(
			latency_frames=base.latency_frames,
			backlash_rad=math.radians(base.backlash_deg),
			velocity_limit_rad_s=self.spec.arm_actuator.velocity_limit,
			encoder_ticks=4096,
		)
		self.cell = Cell(
			layout, SO101_MJCF_DIR, physics=self.spec, dynamic_clutter=True, assisted_grasp=False, face_probes=True
		)
		self.ep = PhysicsEpisode(self.cell, INITIAL_Q, layout.brick_xy, layout.brick_yaw_deg, cfg)
		self.expert_type = ContinuousCarryExpert if smooth_carry else RelocateExpert
		self.expert = self.expert_type(
			self.cell, INITIAL_Q, self._expert_xy, ExpertParams(), RelocateParams(do_bottom=False, do_scan=False)
		)
		self.node = ""
		self.phases: list[int] = []
		self.targets: list[np.ndarray] = []  # policy target per frame (before the command filter)
		self._clipped_frames = 0  # target clips of experts replaced by rewinds
		self.checkpoints: list[Checkpoint] = []
		self.branch = 0
		self.rewinds = 0
		self.total_frames = 0
		self.wrist_count = 0
		self.bottom_count = 0
		self.held: tuple[np.ndarray, float] | None = None
		self.actual_destination = None
		self.execution_error = None
		self.record.start(self.cell)
		self.record.initial(self.cell, INITIAL_Q)
		# Brick perception (opt-in, see .perception): the top camera is calibrated before the episode, and the brick
		# is located in its depth image whenever a skill needs it on the table.
		self.perception = PerceptionCfg.from_env()
		if self.realistic and self.perception is None:
			self.perception = PerceptionCfg.realistic()
		self.brick_est = None
		self._belief_obs = False
		self._grip_from_brick = None  # realistic: the carried brick in the gripper frame, believed at the grasp
		self._expected_yaw = None  # realistic: yaw the brick should show after the agent put it down itself
		self._checkpoint_belief: dict[str, object] = {}
		self._park_xy = np.asarray(layout.brick_xy, dtype=np.float64)
		self.bowls = {
			"green": np.asarray(layout.green, dtype=np.float64),
			"pink": np.asarray(layout.pink, dtype=np.float64),
		}
		self.scene_error = None
		if self.perception is not None:
			seed = int(getattr(layout, "defect_seed", 0))
			self._cam_k, self._cam_est, info = calibrate_layout(layout, seed, self.perception.calibration)
			self._depth_rng = np.random.default_rng(seed + 30_000)
			self.event("camera_calibrated", **info)
		if self.realistic:
			self._contact = GripperContact()
			self._sense_bowls()

	def _sense_bowls(self):
		"""Locate both bowls in the top images and keep the depth image as the empty-bowl reference."""
		depth = self._top_depth()
		found = locate_bowls(self.cell.rgb("top"), depth, self._cam_k, self._cam_est)
		self._baseline_depth = depth
		errors = {}
		for name, est in found.items():
			if est is None:
				self.scene_error = "BOWL_NOT_FOUND"
				continue
			self.bowls[name] = est.xy
			truth = np.asarray(getattr(self.layout, name))
			errors[name] = round(float(np.linalg.norm(est.xy - truth) * 1000), 3)  # scoring only
		self.event(
			"bowls_perceived",
			positions_m={n: None if e is None else e.xy.round(5).tolist() for n, e in found.items()},
			error_xy_mm=errors,
		)

	def _grip_pose(self) -> np.ndarray:
		"""World-from-gripper frame from the joint readings (forward kinematics of the robot model)."""
		d, site = self.cell.data, self.cell.site
		tf = np.eye(4)
		tf[:3, :3] = d.site_xmat[site].reshape(3, 3)
		tf[:3, 3] = d.site_xpos[site]
		return tf

	def _brick_belief(self):
		"""``(position, quaternion xyzw)`` the realistic agent believes for the brick: carried with the gripper since
		the grasp, else as last perceived on the table; None before the first localisation."""
		if self._grip_from_brick is not None:
			tf = self._grip_pose() @ self._grip_from_brick
			quat = np.zeros(4)
			mujoco.mju_mat2Quat(quat, tf[:3, :3].flatten())
			return tf[:3, 3].copy(), quat[[1, 2, 3, 0]]
		if self.brick_est is not None:
			return self.brick_est.pos.copy(), self.brick_est.quat_xyzw()
		return None

	def _table_belief_tf(self) -> np.ndarray:
		est = self.brick_est
		tf = np.eye(4)
		c, s = math.cos(est.yaw), math.sin(est.yaw)
		tf[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
		tf[:3, 3] = est.pos
		return tf

	def _top_depth(self):
		r = self.cell.renderer
		r.enable_depth_rendering()
		try:
			r.update_scene(self.cell.data, "top")
			depth = r.render().astype(np.float64)
		finally:
			r.disable_depth_rendering()
		sigma = self.perception.depth_noise_m * (depth / 0.5) ** 2
		return depth + self._depth_rng.normal(0.0, 1.0, depth.shape) * sigma

	def _perceive(self):
		"""Locate the brick on the table through the calibrated top camera; None, or a failure code."""
		if self.perception is None:
			return None
		est = locate_brick(self._top_depth(), self._cam_k, self._cam_est, list(self.bowls.values()))
		if est is None:
			self.event("brick_perceived", found=False)
			return "BRICK_NOT_FOUND"
		if self._expected_yaw is not None:
			# The footprint only fixes the yaw up to its symmetry; after a put-down, keep the face numbering the
			# agent has been using by taking the symmetric yaw nearest the one it put the brick down with.
			period = math.pi if est.size[0] != est.size[1] else math.pi / 2
			est.yaw += period * round((self._expected_yaw - est.yaw) / period)
			self._expected_yaw = None
		self.brick_est = est
		# Truth is logged for scoring only; nothing in the agent reads it back.
		rot = self.cell.data.xmat[self.cell.brick_body].reshape(3, 3)
		true_yaw = math.atan2(rot[1, 0], rot[0, 0])
		self.event(
			"brick_perceived",
			found=True,
			position_m=est.pos.round(5).tolist(),
			yaw_deg=round(math.degrees(est.yaw), 3),
			footprint_mm=[round(v * 1000, 2) for v in est.footprint_m],
			top_points=est.top_points,
			occlusion_extended_mm=list(est.extended_mm),
			error_xy_mm=round(float(np.linalg.norm(est.pos[:2] - self.cell.brick_pos()[:2]) * 1000), 3),
			error_yaw_deg=round(yaw_error_deg(est.yaw, true_yaw, self.layout.brick_size), 3),
			size_correct=bool(np.allclose(est.size, self.layout.brick_size)),
		)
		return None

	@contextmanager
	def _belief(self):
		"""Show planners that read the brick from the simulation the believed pose instead: the perceived one on the
		table, and in ``realistic`` also the carried one."""
		if self.realistic:
			belief = self._brick_belief()
		elif self.brick_est is None or self.held is not None:
			belief = None
		else:
			belief = (self.brick_est.pos, self.brick_est.quat_xyzw())
		if belief is None:
			yield
			return
		m, d = self.cell.model, self.cell.data
		adr = self.cell.brick_qadr
		saved = d.qpos[adr : adr + 7].copy()
		x, y, z, w = belief[1]
		d.qpos[adr : adr + 3] = belief[0]
		d.qpos[adr + 3 : adr + 7] = (w, x, y, z)
		mujoco.mj_kinematics(m, d)
		try:
			yield
		finally:
			d.qpos[adr : adr + 7] = saved
			mujoco.mj_forward(m, d)

	def close(self):
		self.cell.close()

	def event(self, kind, **payload):
		self.record.event(kind, self.node, self.branch, self.ep.n_frames, **payload)

	def enter(self, node):
		self.node = node
		self.event("node")

	def _execute(self, generator):
		expert = self.expert
		expert._gen = generator
		expert.done = False
		while self.total_frames < MAX_TOTAL_FRAMES:
			if self.on_frame is not None:
				self.on_frame(self)
			obs = self._observe()
			target, phase = expert.act(obs)
			if expert.done:
				return None
			if error := self._apply_target(obs, target, phase):
				expert._gen.close()
				return error
		expert._gen.close()
		return "ACTION_BUDGET_EXHAUSTED"

	# Per-frame hooks shared by _execute and the student segment executor (agent/student_rollout.py).
	def _observe(self):
		"""The expert's per-frame observation (sensor-derived fields in ``realistic``). Updates the contact estimate."""
		obs = observe(self.cell, self.ep)
		if self.realistic:
			belief = self._brick_belief()
			if belief is None:
				obs["brick_pos"], obs["brick_quat"] = np.full(3, np.nan), np.array([0.0, 0.0, 0.0, 1.0])
			else:
				obs["brick_pos"], obs["brick_quat"] = belief
			obs["brick_vel"] = np.zeros(3)
			obs["jaw_normal"] = self._contact.update(
				float(obs["gripper_cmd"]), float(obs["q"][5]), float(obs["effort_frac"])
			)
		elif self._belief_obs and self.brick_est is not None:
			obs["brick_pos"] = self.brick_est.pos.copy()
			obs["brick_quat"] = self.brick_est.quat_xyzw()
		return obs

	def _apply_target(self, obs, target, phase):
		"""Record and apply one joint target through the command filter; returns a fault code or None."""
		# Observation at t is paired with the next unfiltered policy command.
		self.record.command(obs["q"], target, phase)
		self.ep.step(target)
		self.total_frames += 1
		self.phases.append(PHASES.index(phase))
		self.targets.append(np.asarray(target, dtype=np.float64).copy())
		self.record.step(self.cell, self.node)
		if self.fault and self.fault(self.node, self.branch, self.ep.n_frames):
			return "INJECTED_MOTION_FAILURE"
		return None

	def _save(self, views):
		if self.held is not None:
			return
		integration = np.empty(mujoco.mj_stateSize(self.cell.model, STATE_SPEC))
		mujoco.mj_getState(self.cell.model, self.cell.data, integration, STATE_SPEC)
		ex, filt = self.expert, self.ep.filt
		skill_state = {
			"last": ex.last.tolist(),
			"active_open": ex._active_open_g,
			"frame_idx": ex._frame_idx,
			"monitor_position": ex.gripper_monitor._prev_meas,
			"monitor_count": ex.gripper_monitor._run,
		}
		filter_state = {
			"queue": [q.tolist() for q in filt._queue],
			"prev": filt._prev_out.tolist(),
			"backlash": filt._backlash_pos.tolist(),
		}
		checkpoint = Checkpoint(
			f"checkpoint-{len(self.checkpoints)}",
			self.branch,
			self.ep.n_frames,
			integration,
			skill_state,
			filter_state,
			self.wrist_count,
			self.bottom_count,
			copy.deepcopy(views),
		)
		self.record.checkpoint(checkpoint)
		self.checkpoints.append(checkpoint)
		self._checkpoint_belief[checkpoint.id] = copy.deepcopy(self.brick_est)
		self.event("checkpoint", checkpoint=checkpoint.id)

	def prepare(self):
		def ready():
			yield from self.expert._hold(self.expert.p.rest_s, "rest")
			yield from self.expert._go_ready()

		if self.scene_error:
			return self.scene_error
		error = self._execute(ready())
		if not error:
			if self.realistic:  # the arm hid other pixels at rest: keep the farther reading of each
				self._baseline_depth = np.maximum(self._baseline_depth, self._top_depth())
			self._save([])
		return error

	def can_scan_wrist(self, policy):
		self.policy = policy  # read by decision_context for the remaining-budget fields
		return self.wrist_count < policy.max_wrist_views

	def can_present_bottom(self, policy):
		self.policy = policy
		return self.bottom_count < policy.max_bottom_views

	def _geometry(self):
		cell = self.cell
		m, d = cell.model, cell.data

		def rounded(a):
			return np.asarray(a).round(5).tolist()

		# World poses of all simulator bodies. Repeated meshes such as the bowl-wall wedges share
		# a parent, so their 336 constant local transforms need not be retransmitted at every decision.
		bodies = [
			[m.body(i).name, i, int(m.body_parentid[i]), *rounded(d.xpos[i]), *rounded(d.xquat[i][[1, 2, 3, 0]])]
			for i in range(m.nbody)
		]

		return {
			"brick_size_m": self.layout.brick_size,
			"brick_position_m": rounded(cell.brick_pos()),
			"brick_rotation_xyzw": rounded(cell.brick_quat_xyzw()),
			"joints_rad": rounded(cell.joints()),
			"tcp_position_m": rounded(cell.tcp_pos()),
			"grasped": self.held is not None,
			"joint_limits_rad": rounded(cell.jnt_range),
			"body_columns": ["name", "body_id", "parent_id", "xyz_m", "quaternion_xyzw"],
			"bodies": bodies,
			"bowl_radius_m": BOWL_R,
			"collision_model": "SO-101 / R6_box_floor_wedge_walls",
			"bowl_positions_m": {"green": self.layout.green, "pink": self.layout.pink},
			"cameras": {
				m.camera(i).name: {
					"position_m": rounded(d.cam_xpos[i]),
					"rotation": rounded(d.cam_xmat[i]),
					"intrinsic": rounded(m.cam_intrinsic[i]),
					"fovy_deg": float(m.cam_fovy[i]),
				}
				for i in range(m.ncam)
			},
			"pinch_normal_n": rounded(self.ep.last("pinch_normal")) if self.ep.n_frames else [0.0, 0.0],
		}

	SKILL_NOTES: ClassVar[dict[str, str]] = {
		"scan_wrist": "move the wrist camera over the brick to read the side and top faces (0-4) not yet cleared",
		"present_bottom": "grasp the brick and show its bottom face (5) to the top camera",
		"stop": "give up: the episode ends as a failure with the brick unsorted",
	}

	def decision_context(self, allowed, views, failures, notes=None):
		"""Compact, leak-checked state for the LLM: coverage, per-view scores, failures, budgets, checkpoints."""
		covered = decision(views)[1]
		cell = self.cell
		if self.realistic:
			belief = self._brick_belief()
			brick = np.full(3, np.nan) if belief is None else belief[0]
		else:
			brick = cell.brick_pos()
		policy = getattr(self, "policy", None)  # set by whoever owns the Policy; absent means limits are unknown
		budget = {
			"wrist_views_used": self.wrist_count,
			"bottom_views_used": self.bottom_count,
			"rewinds_used": self.rewinds,
			"frames_left": MAX_TOTAL_FRAMES - self.total_frames,
		}
		if policy is not None:
			budget["wrist_views_left"] = policy.max_wrist_views - self.wrist_count
			budget["bottom_views_left"] = policy.max_bottom_views - self.bottom_count
			budget["rewinds_left"] = policy.max_rewinds - self.rewinds
		context = {
			"covered_faces": sorted(covered),
			"missing_faces": [f for f in range(6) if f not in covered],
			"views": [
				{
					"id": v["id"],
					"camera": v["camera"],
					"faces": v["faces"],
					"score": None if v["score"] is None else round(v["score"], 3),
					"threshold": round(v["threshold"], 3),
					"margin": None if v["score"] is None else round(v["score"] - v["threshold"], 3),
				}
				for v in views
			],
			"failures": [{k: f[k] for k in ("code", "skill", "reason_codes") if k in f} for f in failures],
			"budget": budget,
			"grasped": self.held is not None,
		}
		if np.isfinite(brick).all():
			context["brick"] = {"xy_m": np.asarray(brick[:2]).round(3).tolist(), "z_m": round(float(brick[2]), 3)}
		described = {k: self.SKILL_NOTES[k] for k in allowed if k in self.SKILL_NOTES}
		checkpoints = []
		for c in self.checkpoints:
			if c.id in allowed:
				faces = sorted(decision(c.views)[1])
				checkpoints.append({"id": c.id, "frame": c.frame, "faces_covered": faces})
				described[c.id] = f"rewind to frame {c.frame} with faces {faces} covered"
		if checkpoints:
			context["checkpoints"] = checkpoints
		context["choice_notes"] = {**described, **{k: str(v) for k, v in (notes or {}).items() if k in allowed}}
		# Nothing about the defect, its face or the label may reach the model.
		assert_no_defect_leakage(context)
		return context

	def _note_llm(self, source, info):
		stats = getattr(self, "_llm_stats", None)
		if stats is None:
			stats = self._llm_stats = {
				"consulted": 0,
				"adopted": 0,
				"fallback_reasons": {},
				"prompt_tokens": [],
				"latency_s_total": 0.0,
				"attempts": 0,
			}
		stats["consulted"] += 1
		stats["adopted"] += source == "llm"
		if info.get("reason"):
			stats["fallback_reasons"][info["reason"]] = stats["fallback_reasons"].get(info["reason"], 0) + 1
		stats["prompt_tokens"].append(int(info.get("prompt_tokens") or 0))
		stats["latency_s_total"] = round(stats["latency_s_total"] + float(info.get("latency_s") or 0.0), 4)
		stats["attempts"] += int(info.get("attempts") or 0)

	def _llm_summary(self):
		stats = getattr(self, "_llm_stats", None) or {
			"consulted": 0,
			"adopted": 0,
			"fallback_reasons": {},
			"prompt_tokens": [],
			"latency_s_total": 0.0,
			"attempts": 0,
		}
		tokens = stats["prompt_tokens"]
		return {
			**stats,
			"prompt_tokens_median": float(np.median(tokens)) if tokens else None,
			"prompt_tokens_max": max(tokens) if tokens else None,
		}

	def choose(self, allowed, views, failures, notes=None):
		info = {}
		forced, self.forced_choice = getattr(self, "forced_choice", None), None
		if forced is not None:  # an outside agent (the step MCP) already chose
			if forced not in allowed:
				raise ValueError(f"{forced!r} is not one of {allowed}")
			action, source = forced, "agent"
		elif self.llm is None or len(allowed) == 1 and not failures:
			action, source = allowed[0], "policy"
		else:
			context = self.decision_context(allowed, views, failures, notes)
			action, source, info = select_skill(self.llm, allowed, context)
			self._note_llm(source, info)
		self.event(
			"decision",
			choices=allowed,
			choice=action,
			source=source,
			evidence=[v["id"] for v in views],
			prior_failure=bool(failures),
			fallback_reason=info.get("reason"),
			prompt_tokens=info.get("prompt_tokens"),
			latency_s=info.get("latency_s"),
			attempts=info.get("attempts"),
		)
		return action

	def _capture(self, camera):
		cell = self.cell
		rgb = cell.rgb(camera)
		if self.realistic:
			# Visible faces and the crop come from the believed brick; the image is the real one.
			with blind_geometry(cell):
				true_faces = capture_view(cell, camera).faces_px
			with self._belief(), blind_geometry(cell):
				geometry = capture_view(cell, camera)
				image, bbox = detector_crop(cell, camera, None, cell.data, self.crop_size, clamp=True, rgb=rgb)
		else:
			with blind_geometry(cell):
				geometry = capture_view(cell, camera)
			image, bbox = detector_crop(cell, camera, None, cell.data, self.crop_size, clamp=True, rgb=rgb)
		min_px = BOTTOM_MIN_PX if camera == "top" else WRIST_MIN_PX
		faces = [i for i, n in enumerate(geometry.faces_px) if n >= min_px and ((i == 5) == (camera == "top"))]
		threshold = (
			self.bottom_threshold
			if camera == "top" and self.bottom_threshold is not None
			else self.scorer.threshold(camera)
		)
		score = elapsed_ms = None
		heads = {}
		error = None
		reason = None
		if image is None:
			reason = "NO_CROP"
		elif geometry.brick_px < MIN_BRICK_PX:
			reason = "TOO_FEW_BRICK_PIXELS"
		elif not faces:
			reason = "NO_READABLE_FACE"
		else:
			from ..inspection_scorer import ScoreServerError

			self.event("detector_started", camera=camera)
			started = time.perf_counter()
			try:
				score_details = getattr(self.scorer, "score_with_details", None)
				if score_details is None:
					score = float(self.scorer.score(camera, image))
				else:
					score, heads = score_details(camera, image)
				if not math.isfinite(score):
					score = None
					reason = "NONFINITE_SCORE"
			except ScoreServerError as exc:
				reason = "DETECTOR_UNAVAILABLE"
				error = exc
			elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
		view = {
			"id": f"{self.branch}:{self.ep.n_frames}:{camera}",
			"branch": self.branch,
			"frame": self.ep.n_frames,
			"camera": camera,
			"faces": faces,
			"face_pixels": list(geometry.faces_px),
			"score": score,
			"threshold": float(threshold),
			"crop_bbox": bbox,
		}
		self.record.view(view, rgb, image)
		if self.realistic:  # scoring only: how far the believed visible faces were from the true ones
			belief = self._brick_belief()
			belief_err = (
				None if belief is None else round(float(np.linalg.norm(belief[0] - cell.brick_pos()) * 1000), 2)
			)
			self.event(
				"view_truth", id=view["id"], true_face_pixels=[int(v) for v in true_faces], belief_error_mm=belief_err
			)
		# Keep diagnostics in the trace, separate from the decision evidence supplied to the agent.
		self.event(
			"view" if error is None else "detector_error",
			observation=view,
			detector={"head_scores": heads, "elapsed_ms": elapsed_ms, "reason": reason},
		)
		if error is not None:
			raise error
		return view

	def _straight_clear(self, first, last, *, held=False):
		with self._belief():
			self.expert.kin.data.qpos[:] = self.cell.data.qpos
			# Sample the swept arm and bowl/clutter clearance in addition to table height.
			qs = [first + a * (last - first) for a in np.linspace(0, 1, 10)]
			return path_min_z(self.cell, first, last, carried=held) >= 0.015 and self.expert._clearance(qs) >= 0

	def _park_held(self):
		q_lift, roll = self.held
		ex = self.expert
		# Put the brick back where it was found: the perceived spot, or its layout position without perception.
		if self.brick_est is not None:
			self._park_xy = self.brick_est.pos[:2].copy()
		if not self._straight_clear(ex.last, q_lift, held=True):
			return "PARK_PATH_BLOCKED"

		def park():
			yield from ex._carry(ex._move(q_lift, ex._carry_secs(q_lift, ex.p.show_move_s), "show_move"))
			if not ex.lost:
				yield from ex._put_down(self._park_xy, roll, q_lift)
				if self.realistic and self._grip_from_brick is not None:
					# The put-down retreats straight up, so the gripper still has the release orientation.
					rot = (self._grip_pose() @ self._grip_from_brick)[:3, :3]
					self._expected_yaw = math.atan2(rot[1, 0], rot[0, 0])
			if not ex.lost:
				yield from ex._go_ready()

		error = self._execute(park())
		if error or ex.lost:
			return error or "BRICK_LOST"
		self.held = None
		if self.realistic:  # released: where it landed is found by looking again
			self._grip_from_brick = None
			self.brick_est = None
		return None

	def _wrist(self, previous, views):
		self.wrist_count += 1
		if self.held is not None:
			error = self._park_held()
			if error:
				return error
		if error := self._perceive():
			return error
		ex, cell = self.expert, self.cell
		with self._belief():
			ex.kin.data.qpos[:] = cell.data.qpos
		attempt = min(self.wrist_count - 1 + self.rewinds, 2)
		steps, mode, extra = self._scan_plan(previous, attempt)
		self.event("scan_plan", mode=mode, candidates=len(steps), **extra)
		pending: list[np.ndarray] = []  # ring poses of skipped views, still swept through to stay on the ring
		for candidate, via in steps:
			path = [*pending, *via, candidate.q]
			_, covered = decision([*previous, *views])
			if not any(f not in covered and candidate.faces_px[f] >= WRIST_MIN_PX for f in range(5)):
				pending = path if via else []
				continue
			pending = []
			if not all(self._straight_clear(a, b) for a, b in itertools.pairwise([ex.last, *path])):
				continue

			def look(path):
				yield from sweep(
					ex,
					path,
					self.scan_peak_speed,
					ex.rp.scan_move_s,
					pause_s=self.sweep_pause_s if len(path) > 1 else None,
				)
				yield from ex._hold_servo(path[-1], ex.rp.scan_hold_s, "approach")

			if error := self._execute(look(path)):
				return error
			views.append(self._capture("wrist_cam"))
			if decision(views)[0] == "defective":
				break
		if error := self._execute(ex._go_ready()):
			return error
		self._save([*previous, *views])
		return None

	def _scan_plan(self, previous, attempt):
		"""``([(view, via poses)], mode, event fields)``: the orbit scan around the upward face when it covers every
		side and top face not cleared yet, else the hover scan of :func:`plan_table_scan`."""
		ex, cell = self.expert, self.cell
		need = [f for f in range(5) if f not in decision(previous)[1]]
		with self._belief(), blind_geometry(cell):
			plan = plan_orbit_scan(
				cell,
				ex.kin,
				ex.last,
				gripper=ex.p.open_g,
				params=ORBIT_ATTEMPTS[attempt],
				faces=need,
				min_px=WRIST_MIN_PX,
			)
		reached = {f for v in plan.views for f in need if v.faces_px[f] >= WRIST_MIN_PX}
		if plan.views and reached == set(need):
			extra = {"tip_height_m": plan.tip_height_m, "standoff_m": plan.standoff_m, "ring_poses": len(plan.ring)}
			return list(zip(plan.views, plan.via, strict=True)), plan.mode, extra
		modes = (
			{"heights": (0.06, 0.075, 0.09), "offset_m": 0.03},
			{"heights": (0.07, 0.085, 0.10), "offset_m": 0.02},
			{"heights": (0.055, 0.08, 0.095), "offset_m": 0.04},
		)
		with self._belief(), blind_geometry(cell):
			candidates, mode = plan_table_scan(
				cell, ex.kin, ex.last, gripper=ex.p.open_g, min_px=WRIST_MIN_PX, **modes[attempt]
			)
		return [(c, []) for c in candidates], mode, {"orbit_faces": sorted(reached)}

	def _grasp(self):
		ex = self.expert
		ex.grasp_retries = ex.block_replans = 0
		ex._active_open_g = float(ex.p.open_g)
		if error := self._perceive():
			return error

		def grasp_and_lift():
			self._belief_obs = True  # the grasp is planned on the perceived pose
			try:
				grasp = yield from ex._grasp()
			finally:
				self._belief_obs = False
			if grasp is not None and self.realistic:
				# From now on the brick is believed to move with the gripper, as it was perceived before the grasp.
				self._grip_from_brick = np.linalg.inv(self._grip_pose()) @ self._table_belief_tf()
			if grasp is not None:
				q_lift = yield from ex._lift(grasp)
				self.held = (q_lift, grasp.roll)

		error = self._execute(grasp_and_lift())
		return error or ("BRICK_LOST" if ex.lost else ex.aborted)

	def _bottom(self, views):
		self.bottom_count += 1
		if self.held is None and (error := self._grasp()):
			return error
		ex, cell = self.expert, self.cell
		with self._belief(), blind_geometry(cell):
			candidate = plan_bottom_view(
				cell,
				ex.last,
				gripper=ex.p.hold_g,
				seed=self.rewinds * 71 + self.bottom_count - 1,
				clearance=lambda q: self._straight_clear(ex.last, q, held=True),
				border_px=BOTTOM_BORDER_PX if self.realistic else 0,
			)
		if candidate is None or candidate.faces_px[5] < BOTTOM_MIN_PX:
			self.event("view_unavailable", camera="top", reason="NO_BOTTOM_PATH")
			return None

		def look():
			pose = candidate.q.copy()
			yield from ex._carry(ex._move(pose, ex._carry_secs(pose, ex.p.show_move_s), "show_move"))
			yield from ex._carry(ex._hold_servo(pose, ex.rp.bottom_hold_s, "show_hold"))

		if error := self._execute(look()):
			return error
		if ex.lost:
			return "BRICK_LOST"
		views.append(self._capture("top"))
		return None

	def inspect(self, node, previous):
		views = []
		try:
			error = self._wrist(previous, views) if node == "scan_wrist" else self._bottom(views)
		except RuntimeError as exc:
			# The scoring process can stop independently of simulation. Keep the physical state and trace.
			from ..inspection_scorer import ScoreServerError

			if not isinstance(exc, ScoreServerError):
				raise
			error = "DETECTOR_UNAVAILABLE"
		return views, error

	def failure(self, reason, views):
		self.execution_error = reason
		failed_node = next(
			(e["node"] for e in reversed(self.record.events) if e["kind"] == "node" and e["node"] != "recover"), None
		)

		payload = {
			"code": reason,
			"skill": failed_node,
			"frame": self.ep.n_frames,
			"branch": self.branch,
			"prior_views": [{"id": v["id"], "faces": v["faces"], "score": v["score"]} for v in views],
		}
		if reason == "PHYSICAL_HEALTH_FAILED":
			payload["reason_codes"] = next(
				event["reason_codes"]
				for event in reversed(self.record.events)
				if event["kind"] == "motion_health" and event["branch"] == self.branch
			)
		self.event("failure", **{k: v for k, v in payload.items() if k not in ("frame", "branch")})
		return payload

	def restore_state(self, checkpoint, params=None, relocate=None):
		"""Physics, command filter, a rebuilt expert and the view counters at a skill-boundary checkpoint (brick on
		the table, gripper open). Shared by in-process ``rewind`` and cross-process ``twin_resume.restore``."""
		m, d = self.cell.model, self.cell.data
		mujoco.mj_setState(m, d, checkpoint.integration, STATE_SPEC)
		mujoco.mj_forward(m, d)
		mujoco.mj_setState(m, d, checkpoint.integration, STATE_SPEC)
		filt = self.ep.filt
		filt._queue = deque((np.array(q) for q in checkpoint.filter["queue"]), maxlen=filt._queue.maxlen)
		filt._prev_out = np.array(checkpoint.filter["prev"])
		filt._backlash_pos = np.array(checkpoint.filter["backlash"])
		params = self.expert.p if params is None else params
		relocate = self.expert.rp if relocate is None else relocate
		self._clipped_frames += getattr(self.expert, "clipped_frames", 0)
		self.expert = ex = self.expert_type(self.cell, INITIAL_Q, self._expert_xy, params, relocate)
		ex.last = np.array(checkpoint.expert["last"])
		ex._active_open_g = checkpoint.expert["active_open"]
		ex._frame_idx = checkpoint.expert["frame_idx"]
		ex.gripper_monitor._prev_meas = checkpoint.expert["monitor_position"]
		ex.gripper_monitor._run = checkpoint.expert["monitor_count"]
		self.held = None
		self._grip_from_brick = None
		if self.realistic:
			self._contact.reset()
			self.brick_est = copy.deepcopy(self._checkpoint_belief.get(checkpoint.id))
		self.actual_destination = None
		self.execution_error = None
		self.wrist_count = checkpoint.wrist_count
		self.bottom_count = checkpoint.bottom_count

	def recovery_choices(self, policy: Policy, failures) -> list[tuple[str, str]]:
		"""``(choice, note)`` pairs the recover node offers after ``failures[-1]``, the default first and ``stop``
		last; empty when the budget leaves no recovery. ``truth`` rewinds the physics to a skill-boundary checkpoint;
		``realistic`` recovers from the current state (see :meth:`_retry`)."""
		stop = ("stop", "give up and report the failure")
		if self.rewinds >= policy.max_rewinds or self.total_frames >= MAX_TOTAL_FRAMES:
			return []
		failure = failures[-1]
		if self.realistic:
			if failure.get("code") in ("BOWL_NOT_FOUND", "DETECTOR_UNAVAILABLE"):
				return []
			recommended = VARIANT_BY_CODE.get(failure.get("code"))
			variants = [v for v in dict.fromkeys((recommended, "cautious")) if v]
			note = "recover from the current state and retry with"
			options = [(f"retry+{v}", f"{note} {VARIANT_NOTES[v]}") for v in variants]
			return [*options, ("retry", f"{note} unchanged parameters"), stop]
		if not self.checkpoints:
			return []
		stall = stall_start(self.ep.telemetry()["actuator_torques"], self.spec.arm_actuator.effort_limit)
		stall_node = None if stall is None else node_at(self.record.events, stall)
		return [*rewind_options(self.checkpoints, failure, stall, stall_node), stop]

	def rewind(self, policy: Policy, failures):
		if self.realistic:
			return self._retry(policy, failures)
		self.policy = policy
		options = self.recovery_choices(policy, failures)
		if not options:
			return None
		choice = self.choose([c for c, _ in options], self.checkpoints[-1].views, failures, notes=dict(options))
		if choice == "stop":
			return None
		checkpoint_id, _, variant = choice.partition("+")
		checkpoint = next(c for c in self.checkpoints if c.id == checkpoint_id)
		self.rewinds += 1
		self.branch = self.rewinds
		self.record.fork(self.branch, checkpoint)
		for rows in self.ep._logs.values():
			del rows[checkpoint.frame :]
		del self.ep.jaw_env_force[checkpoint.frame :]
		del self.phases[checkpoint.frame :]
		del self.targets[checkpoint.frame :]
		params, relocate = self.expert.p, self.expert.rp
		if variant:
			# Variants transform the first attempt's defaults, so they never stack on an earlier variant.
			self._base_params = getattr(self, "_base_params", (params, relocate))
			params, relocate, self.scan_peak_speed = apply_variant(variant, *self._base_params)
		self.recovery_variants = (*self.recovery_variants, variant or "none")
		self.restore_state(checkpoint, params, relocate)
		del self.checkpoints[self.checkpoints.index(checkpoint) + 1 :]
		self.record.restored(self.cell)
		self.event(
			"rewound",
			parent_branch=checkpoint.branch,
			checkpoint=checkpoint.id,
			reason=failures[-1]["code"],
			variant=variant or None,
		)
		return copy.deepcopy(checkpoint.views)

	def _retry(self, policy: Policy, failures):
		"""Realistic recovery: no physics rewind. Put a carried brick down (or let go of it), go to the ready pose,
		locate the brick again and carry on with a bounded variant. The views so far still count only when the brick
		is found where it was believed at the last skill boundary; otherwise its face numbering is lost."""
		self.policy = policy
		failure = failures[-1]
		options = self.recovery_choices(policy, failures)
		if not options:
			return None
		last = self.checkpoints[-1] if self.checkpoints else None
		kept = copy.deepcopy(last.views) if last else []
		choice = self.choose([c for c, _ in options], kept, failures, notes=dict(options))
		if choice == "stop":
			return None
		_, _, variant = choice.partition("+")
		self.rewinds += 1
		params, relocate = self.expert.p, self.expert.rp
		if variant:
			self._base_params = getattr(self, "_base_params", (params, relocate))
			params, relocate, self.scan_peak_speed = apply_variant(variant, *self._base_params)
		self.recovery_variants = (*self.recovery_variants, variant or "none")
		self.event("retry", reason=failure.get("code"), variant=variant or None)
		lost = self.expert.lost
		self._clipped_frames += getattr(self.expert, "clipped_frames", 0)
		self.expert = ex = self.expert_type(self.cell, INITIAL_Q, self._expert_xy, params, relocate)
		ex.last = self.cell.joints().copy()
		self._contact.reset()
		error = None
		if self.held is not None and not lost:
			error = self._park_held()
		if self.held is not None or self._grip_from_brick is not None or error:

			def let_go():  # open where the gripper is, then rise
				q = ex.last.copy()
				q[5] = ex.p.open_g
				yield from ex._move(q, ex.p.close_s, "place")
				up = ex._ik(self.cell.tcp_pos() + np.array([0.0, 0.0, 0.06]), q, float(q[4]))
				up[5] = ex.p.open_g
				yield from ex._move(up, ex.p.descend_s, "place")

			error = self._execute(let_go())
		self.held = None
		self._grip_from_brick = None
		self.brick_est = None
		if error or self._execute(ex._go_ready()):
			return None
		believed = self._checkpoint_belief.get(last.id) if last else None
		if self._perceive():
			return None
		moved = believed is None or bool(
			np.linalg.norm(self.brick_est.pos[:2] - believed.pos[:2]) > RETRY_MOVED_M
			or yaw_error_deg(self.brick_est.yaw, believed.yaw, self.brick_est.size) > RETRY_MOVED_DEG
			or self.brick_est.size != believed.size
		)
		views = [] if moved else kept
		self.event("retried", brick_moved=moved, views_kept=len(views))
		self._save(views)
		return views

	def place(self, label):
		if label == "unknown":
			self.event("abstain", destination="inspection_table")
			error = self._park_held() if self.held is not None else None
			if error:
				return error
			if self.realistic:  # the brick must be seen back on the table
				if self.brick_est is None and self._perceive():
					return "INSPECTION_TABLE_PARK_UNVERIFIED"
				self.actual_destination = "inspection_table"
				return None
			position = self.cell.brick_pos()
			if (
				np.linalg.norm(position[:2] - self._park_xy) > 0.03
				or position[2] > self.layout.brick_size[2] / 2 + 0.01
			):
				return "INSPECTION_TABLE_PARK_UNVERIFIED"
			self.actual_destination = "inspection_table"
			return None
		if self.held is None and (error := self._grasp()):
			return error
		ex = self.expert
		q_lift, _ = self.held
		ex.bowl_xy = np.array(self.bowls["pink" if label == "defective" else "green"])
		if not self._straight_clear(ex.last, q_lift, held=True):
			return "PLACE_RETURN_PATH_BLOCKED"

		def release():
			if np.max(np.abs(ex.last - q_lift)) > 0.01:
				yield from ex._carry(ex._move(q_lift, ex._carry_secs(q_lift, ex.p.show_move_s), "show_move"))
			if not ex.lost:
				yield from ex._place(q_lift)
			if not ex.lost:
				yield from ex._go_home()

		error = self._execute(release())
		if error or ex.lost:
			return error or "BRICK_LOST"
		self.held = None
		position = self.cell.brick_pos()
		for name, xy in (("green", self.layout.green), ("pink", self.layout.pink)):
			if np.linalg.norm(position[:2] - xy) < BOWL_R and position[2] < 0.055:
				self.actual_destination = name  # ground truth, read only by evaluate in realistic
				break
		target = "pink" if label == "defective" else "green"
		if self.realistic:
			self._grip_from_brick = None
			self.brick_est = None
			depth = self._top_depth()
			fill = {
				n: bowl_fill(depth, self._baseline_depth, self._cam_k, self._cam_est, xy)
				for n, xy in self.bowls.items()
			}
			self.event("placement_check", fill_pixels=fill)
			if not bowl_filled(fill[target]):
				return "PLACEMENT_UNVERIFIED"
			return None  # the motion-health check is part of the final evaluation
		if self.actual_destination != target:
			return "PLACEMENT_UNVERIFIED"
		health = self._health(ex.bowl_xy)
		if health["verdict"] != "SUCCESS":
			self.event("motion_health", reason_codes=health["reason_codes"])
			return "PHYSICAL_HEALTH_FAILED"
		return None

	def evaluate(self, label, views, failures):
		# Ground truth enters only after the graph has selected and executed its final action.
		expected = "defective" if self.layout.defective else "normal"
		target = self.layout.pink if self.layout.defective else self.layout.green
		health = self._health(target) if self.ep.n_frames and label != "unknown" else {}
		ok = label == expected and self.actual_destination == ("pink" if self.layout.defective else "green")
		result = {
			"verdict": "SUCCESS"
			if ok and health.get("verdict") == "SUCCESS"
			else "UNKNOWN"
			if label == "unknown" and not self.execution_error
			else "FAILED",
			"label": label,
			"label_correct": label == expected,
			"destination": self.actual_destination,
			"active_branch": self.branch,
			"rewinds": self.rewinds,
			"recovery_variants": list(self.recovery_variants),
			"failures": failures,
			"active_views": len(views),
			"views": sum(e["kind"] == "view" for e in self.record.events),
			"sim_frames": self.total_frames,
			"active_frames": self.ep.n_frames,
			"evidence": [v["id"] for v in views],
			"cleared_faces": sorted(decision(views)[1]),
			"health": health,
			"llm": self._llm_summary(),
			"smoothness": self._smoothness(),
			"student_eligible": bool(ok and health.get("verdict") == "SUCCESS" and not self.rewinds),
			"placement_evidence": next(
				(v["id"] for v in views if v["score"] is not None and v["score"] >= v["threshold"]), None
			),
		}
		self.record.audit(
			{"true_label": expected, "true_face": self.layout.defect_face if self.layout.defective else None}
		)
		return result

	def _smoothness(self):
		n = min(len(self.targets), self.ep.n_frames)
		if n < 4:
			return {"frames": n}
		qpos = self.ep.telemetry()["qpos"][:n]
		out = motion_smoothness(np.asarray(self.targets[:n]), qpos, np.asarray(self.phases[:n]))
		out["clipped_frames"] = int(self._clipped_frames + getattr(self.expert, "clipped_frames", 0))
		return out

	def _health(self, target):
		tel = self.ep.telemetry()
		return evaluate_episode_health(
			q_cmd=tel["q_cmd"],
			q_act=tel["qpos"],
			tcp_pos=tel["tcp_pos"],
			brick_pos=tel["brick_pos"],
			brick_quat=tel["brick_quat"],
			jaw_forces=tel["jaw_forces"],
			unintended_forces=tel["unintended_forces"],
			phases=np.asarray(self.phases),
			target_bowl_xy=target,
			bowl_r=BOWL_R,
			brick_vel=tel["brick_vel"],
			brick_ang_vel=tel["brick_ang_vel"],
			ee_quat=tel["ee_quat"],
			fps=FPS,
			brick_size=self.layout.brick_size,
			substep_unintended_forces=tel["substep_unintended"],
			pinch_normal_forces=tel["pinch_normal"],
			pinch_tangential_forces=tel["pinch_tangential"],
			actuator_torques=tel["actuator_torques"],
			torque_limits=np.array(
				[self.spec.arm_actuator.effort_limit] * 5 + [self.spec.gripper_actuator.effort_limit]
			),
			torque_source="solver_measured",
			timestamps=tel["t"],
		).to_dict()


def file_digest(path):
	return hashlib.sha256(Path(path).read_bytes()).hexdigest()
