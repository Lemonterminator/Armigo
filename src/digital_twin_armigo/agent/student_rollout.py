"""Student execution interface: the six-face skills run as motion segments driven by a per-frame policy.

A **segment** is the motion between two decision points of the expert's skill code: places where the skill waits for
a sensor result (a capture and detector score, a perception, a clamp or lost-brick check, a reading of the gripper
pose) before it decides what comes next. The graph and the skills' graph-side logic keep deciding; a policy only
moves the arm within one segment and says when the segment is done (``segment_done``).

Three pieces:

- :class:`Segment` names a segment (``kind``, JSON ``params``) and gives its text instruction (a VLA task string).
  Its params mix the instruction's own fields with planner outputs (target joints, predicted visible faces, pixel
  counts). A policy sees only :func:`policy_inputs` (the ``POLICY_FIELDS``: kind, instruction and the fields the
  instruction is formatted from); everything else is a diagnostic for replay and grading (``Segment.diagnostics``),
  reachable only through the executor's separate ``segment_monitor`` channel.
- :class:`StudentTwin` is a :class:`~.twin_mujoco.TwinMujoco` whose motion generators are never run. Each call of
  ``_execute(generator)`` is mapped to one segment, or to a short sequence of segments for the three skills whose
  generator has a decision point inside (grasp and lift, park, release into a bowl), and each segment is executed
  by :meth:`StudentTwin.run_segment`: observation -> ``policy.act`` -> the expert's own command path
  (``TwinMujoco._apply_target``: joint-range clip, record, command filter, ``Episode.step``). All graph-side logic
  (scan planning and skipping, clearance checks, captures, early stop after a defective view, perception, placement
  checks, in-place recovery) is the unchanged ``TwinMujoco`` code; the few decisions taken inside a generator are
  reproduced between segments (clamp check, carried-brick belief at the grasp, lost-brick monitor, put-down yaw).
- :class:`ExpertTracer` is the unchanged expert with segment markers: the same mapping labels its frames, so expert
  recordings carry the same segment ids, instructions and ends a student is driven with. It is the expert backend of
  every recording entry (``twin_mcp run`` / ``serve`` / resume through ``TwinService.run``, and the step MCP), so
  production recordings and replay-gate traces are labelled by this one implementation.

Recording: both twins emit ``segment`` / ``segment_end`` events and set ``EpisodeTrace.segment``, which writes a
per-frame ``segment`` column (and exact ``state_f64`` / ``action_f64``) to ``actions.npz``. The segment-end
observation (``segment_done``) is the first observation of the next segment (no step happens between them).

:class:`ReplayPolicy` plays a recorded expert episode's float64 actions and segment ends; ``scripts/vla/replay_gate.py``
checks that the replay reproduces the expert bit for bit.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from ..constants import FPS, PHASES
from ..mujoco_cell.gripper_state import GripperMonitor, GripperMonitorParams, GripperState
from .grasp_tail import GRASP_TAIL_MODES, GraspPlan, GraspTail, expert_tail
from .twin_mujoco import INITIAL_Q, MAX_TOTAL_FRAMES, WRIST_MIN_PX, TwinMujoco
from .twin_record import (  # noqa: F401 - POLICY_FIELDS / POLICY_PARAMS / SEGMENT_KINDS are re-exported
	POLICY_FIELDS,
	POLICY_PARAMS,
	SEGMENT_KINDS,
	Segment,
	policy_inputs,
)

# Composite expert generators: (segment kind, frame predicate (phase, carried) -> bool) in order. A composite
# generator's frames belong to the first part, from the current one on, whose predicate holds. ``carried`` marks
# frames yielded through the expert's lost-brick watch (``ClosedLoopExpert._carry``).
_PARTS: dict[str, tuple[tuple[str, Callable[[str, bool], bool]], ...]] = {
	"grasp_and_lift": (("grasp", lambda ph, c: ph != "lift"), ("lift", lambda ph, c: ph == "lift")),
	"park": (
		("return_to_lift", lambda ph, c: ph == "show_move"),
		("lower_on_table", lambda ph, c: ph == "place" and c),
		("release_on_table", lambda ph, c: ph == "place" and not c),
		("return_ready", lambda ph, c: ph == "approach"),
	),
	"release": (
		("return_to_lift", lambda ph, c: ph == "show_move"),
		("carry_to_bowl", lambda ph, c: ph == "place" and c),
		("release_in_bowl", lambda ph, c: not c and ph in ("place", "return")),
	),
}


class SegmentMappingError(RuntimeError):
	"""A skill ran a motion the segment mapping does not know, or its frames do not fit the expected parts."""


@dataclass
class StudentObservation:
	"""What a policy sees at one frame. ``task`` is :func:`policy_inputs` of the current segment (no planner
	diagnostics). ``images()`` gives the recorder's camera images of this frame (rendered on demand when the trace
	records no video); rendering is skipped unless a policy asks for it."""

	frame: int
	task: dict[str, Any]  # policy_inputs(segment): POLICY_FIELDS only
	segment_id: int
	segment_frame: int  # actions applied so far in this segment
	q: np.ndarray  # joint reading (rad), the recorded ``state``
	_images: Callable[[], dict[str, np.ndarray]]
	q_prev: np.ndarray | None = None  # previous frame joint reading (rad), (6,) float64
	prev_action: np.ndarray | None = None  # previous applied target (rad), INITIAL_Q at frame 0, (6,) float64
	gripper_cmd: float = 0.0  # filtered gripper command from obs["gripper_cmd"]

	def images(self) -> dict[str, np.ndarray]:
		return self._images()


@dataclass
class PolicyStep:
	target: np.ndarray | None = None  # 6 joint targets (rad); ignored when done
	done: bool = False  # segment_done: end the segment at this observation, without a step
	phase: str | None = None  # annotation for the health scorer; default: the segment's phase


class SegmentPolicy(Protocol):
	def start(self, task: dict[str, Any], segment_id: int) -> None:
		"""A new segment begins; ``task`` is its :func:`policy_inputs`."""

	def act(self, obs: StudentObservation) -> PolicyStep: ...


def _rounded(values, digits=6):
	return [round(float(v), digits) for v in values]


def roll_deg(roll_rad: float) -> float:
	"""A segment's ``roll_deg`` param: the controller's wrist roll in degrees, rounded to 0.1."""
	return round(math.degrees(float(roll_rad)), 1)


def with_grasp_roll(seg: Segment, roll_rad: float) -> Segment:
	"""``seg`` (a ``grasp`` segment) with the planned grasp roll as its ``roll_deg`` param."""
	return Segment(seg.kind, {**seg.params, "roll_deg": roll_deg(roll_rad)}, variant=seg.variant)


class _SegmentMixin:
	"""Segment bookkeeping and the generator -> segment mapping shared by the expert tracer and the student."""

	def _segment_setup(self):
		self.segments: list[dict] = []  # one row per segment: id, kind, instruction, params, start, end, reason
		self.lift_poses: list[np.ndarray] = []  # the held lift pose after each grasp (diagnostics)
		self._open_segment: dict | None = None
		self._scan_steps: list = []

	def _checkpoint_extra(self):
		"""Segment rows and lift poses so far (a resumed episode continues the segment ids)."""
		if self._open_segment is not None:
			raise RuntimeError("checkpoint inside a segment")
		return {
			"rows": [{k: v for k, v in row.items() if k != "segment"} for row in self.segments],
			"lift_poses": [np.asarray(q, dtype=np.float64).tolist() for q in self.lift_poses],
		}

	def _instruction_variant(self) -> int:
		if not hasattr(self, "layout"):
			return int(getattr(self, "variant", 0) or 0)
		layout = getattr(self, "layout", None)
		if layout is None:
			return 0
		v3 = getattr(layout, "_scene_v3", None)
		if v3 is not None and hasattr(v3, "variant"):
			return int(v3.variant)
		if isinstance(layout, dict):
			return int(layout.get("variant", 0))
		return int(getattr(layout, "variant", 0) or 0)

	def _restore_extra(self, extra):
		if extra is None:
			return
		self.segments = [
			{
				**row,
				"segment": Segment(
					row["kind"],
					row["params"],
					variant=int(row.get("variant", 0)),
				),
			}
			for row in extra["rows"]
		]
		self.lift_poses = [np.array(q, dtype=np.float64) for q in extra["lift_poses"]]
		self._open_segment = None
		self.record.segment = -1

	def _scan_plan(self, previous, attempt):
		steps, mode, extra = super()._scan_plan(previous, attempt)
		self._scan_steps, self._scan_mode = steps, mode
		return steps, mode, extra

	def _segment_begin(self, seg: Segment) -> int:
		if self._open_segment is not None:
			raise RuntimeError("a segment is already open")
		row = {
			"id": len(self.segments),
			"kind": seg.kind,
			"instruction": seg.instruction,
			"params": seg.params,
			**({"variant": seg.variant} if seg.variant else {}),
			"start": self.ep.n_frames,
			"segment": seg,
		}
		self.segments.append(row)
		self._open_segment = row
		self.record.segment = row["id"]
		self.event(
			"segment",
			id=row["id"],
			segment_kind=seg.kind,
			instruction=seg.instruction,
			params=seg.params,
			**({"variant": seg.variant} if seg.variant else {}),
		)
		return row["id"]

	def _segment_end(self, reason: str):
		row, self._open_segment = self._open_segment, None
		row["end"], row["reason"] = self.ep.n_frames, reason
		self.record.segment = -1
		self.event(
			"segment_end", id=row["id"], segment_kind=row["kind"], frames=row["end"] - row["start"], reason=reason
		)

	def _bowl_name(self) -> str:
		xy = np.asarray(self.expert.bowl_xy)
		slot = next(name for name, b in self.bowls.items() if np.array_equal(np.asarray(b), xy))
		return self.bowl_colour(slot) if hasattr(self, "bowl_colour") else slot

	def _motion(self, generator) -> tuple[str, Any]:
		"""``("simple", Segment)`` or ``("composite", (name, {kind: Segment}))`` for a skill's motion generator."""
		name = generator.__name__
		local = generator.gi_frame.f_locals if generator.gi_frame is not None else {}
		v = (
			self._instruction_variant()
			if hasattr(self, "_instruction_variant")
			else _SegmentMixin._instruction_variant(self)
		)
		if name == "ready":
			return "simple", Segment("prepare", variant=v)
		if name == "_go_ready":
			return "simple", Segment("return_ready", variant=v)
		if name == "clear_view":  # TwinMujoco._clear_top_depth: to rest for an unoccluded top image, then ready
			return "simple", Segment("prepare", {"mode": "clear_view"}, variant=v)
		if name == "let_go":
			return "simple", Segment("let_go", variant=v)
		# TwinMujoco._review_once: re-views of a detector alarm reuse the student's fixed segment vocabulary
		# (SEGMENT_KINDS is a VLA contract); ``mode="review"`` marks them.
		if name == "review_wrist":
			path = local["path"]
			views = len(self._scan_steps) + 1
			return "simple", Segment(
				"scan_view",
				{
					"mode": "review",
					"view": views,
					"views": views,
					"target_q": _rounded(path[-1]),
					"via_poses": len(path) - 1,
					"roll_deg": roll_deg(path[-1][4]),
				},
				variant=v,
			)
		if name == "ring_locate":  # TwinMujoco._ring_localise: one wrist view of the localisation ring
			path = local["path"]
			return "simple", Segment(
				"scan_view",
				{
					"mode": "locate",
					"view": local["view"],
					"views": local["views"],
					"target_q": _rounded(path[-1]),
					"via_poses": len(path) - 1,
					"roll_deg": roll_deg(path[-1][4]),
				},
				variant=v,
			)
		if name == "review_bottom":
			pose = local["pose"]
			return "simple", Segment(
				"present_bottom",
				{"mode": "review", "target_q": _rounded(pose), "roll_deg": roll_deg(pose[4])},
				variant=v,
			)
		if name == "dense_scan":
			path = local["path"]
			faces = sorted({f for cand, _ in self._scan_steps for f in range(5) if cand.faces_px[f] >= WRIST_MIN_PX})
			return "simple", Segment(
				"scan_view",
				{
					"mode": getattr(self, "_scan_mode", "orbit"),
					"evidence": getattr(self, "evidence", "dense"),
					"view": 1,
					"views": 1,
					"faces": faces,
					"target_q": _rounded(path[-1]),
					"via_poses": len(path) - 1,
					"roll_deg": roll_deg(path[-1][4]),
				},
				variant=v,
			)
		if name == "look" and "path" in local:  # TwinMujoco._wrist: sweep through ``path`` and hold its last pose
			path = local["path"]
			k = next((i for i, (c, _) in enumerate(self._scan_steps) if c.q is path[-1]), None)
			if k is None:
				raise SegmentMappingError("scan view not found in the scan plan")
			cand = self._scan_steps[k][0]
			params = {
				"mode": self._scan_mode,
				"view": k + 1,
				"views": len(self._scan_steps),
				"faces": [f for f in range(5) if cand.faces_px[f] >= WRIST_MIN_PX],
				"target_q": _rounded(cand.q),
				"via_poses": len(path) - 1,
				"roll_deg": roll_deg(cand.q[4]),
			}
			return "simple", Segment("scan_view", params, variant=v)
		if name == "look" and "candidate" in local:  # TwinMujoco._bottom
			cand = local["candidate"]
			return "simple", Segment(
				"present_bottom",
				{"target_q": _rounded(cand.q), "bottom_px": int(cand.faces_px[5]), "roll_deg": roll_deg(cand.q[4])},
				variant=v,
			)
		if name in _PARTS:
			segs = {}
			for kind, _ in _PARTS[name]:
				params = {}
				if kind == "carry_to_bowl":
					params = {"bowl": self._bowl_name()}
				elif kind == "lower_on_table":
					params = {"xy": _rounded(self._park_xy, 4)}
				segs[kind] = Segment(kind, params, variant=v)
			return "composite", (name, segs)
		raise SegmentMappingError(f"no segment for motion generator {name!r} in node {self.node!r}")


DEFAULT_ACTION_NOISE_KINDS: tuple[str, ...] = (
	"grasp",
	"lift",
	"carry_to_bowl",
	"release_in_bowl",
	"scan_view",
)


@dataclass(frozen=True)
class ActionNoise:
	"""Ornstein-Uhlenbeck action-noise spec for arm joints (0..4; gripper joint 5 is untouched).

	Discrete-time update at ``dt = 1 / FPS``:
	``n[t + 1] = rho * n[t] + sigma_rad * sqrt(1 - rho**2) * N(0, 1)``, where ``rho = exp(-dt / tau_s)``.
	"""

	sigma_rad: float = 0.02
	tau_s: float = 0.3
	kinds: tuple[str, ...] = DEFAULT_ACTION_NOISE_KINDS
	seed: int = 0
	clip_rad: float | None = None

	def __post_init__(self) -> None:
		if self.sigma_rad < 0.0 or not math.isfinite(self.sigma_rad):
			raise ValueError(f"sigma_rad must be finite and >= 0, got {self.sigma_rad!r}")
		if self.tau_s <= 0.0 or not math.isfinite(self.tau_s):
			raise ValueError(f"tau_s must be finite and > 0, got {self.tau_s!r}")
		if self.clip_rad is not None and (self.clip_rad <= 0.0 or not math.isfinite(self.clip_rad)):
			raise ValueError(f"clip_rad must be None or finite > 0, got {self.clip_rad!r}")
		bad = set(self.kinds) - set(SEGMENT_KINDS)
		if bad:
			raise ValueError(f"unknown segment kind(s) in ActionNoise.kinds: {sorted(bad)}")
		object.__setattr__(self, "kinds", tuple(self.kinds))

	def rho(self, dt: float = 1.0 / FPS) -> float:
		return math.exp(-float(dt) / float(self.tau_s))

	def innovation_std(self, dt: float = 1.0 / FPS) -> float:
		r = self.rho(dt)
		return float(self.sigma_rad) * math.sqrt(max(0.0, 1.0 - r * r))

	def make_process(self, dt: float = 1.0 / FPS) -> _OUNoiseProcess:
		return _OUNoiseProcess(self, dt=dt)


class _OUNoiseProcess:
	"""Stateful 5-DOF OU noise generator with per-segment reset."""

	def __init__(self, spec: ActionNoise, dt: float = 1.0 / FPS) -> None:
		self.spec = spec
		self.dt = float(dt)
		self.rho = spec.rho(self.dt)
		self.innov_std = spec.innovation_std(self.dt)
		self.rng = np.random.default_rng(int(spec.seed))
		self.state = np.zeros(5, dtype=np.float64)

	def reset(self) -> None:
		self.state.fill(0.0)

	def step(self) -> np.ndarray:
		if self.spec.sigma_rad == 0.0:
			self.state.fill(0.0)
		else:
			eps = self.rng.normal(0.0, 1.0, size=5)
			self.state = self.rho * self.state + self.innov_std * eps
			if self.spec.clip_rad is not None:
				self.state = np.clip(self.state, -self.spec.clip_rad, self.spec.clip_rad)
		out = np.zeros(6, dtype=np.float64)
		out[:5] = self.state
		return out


class ExpertTracer(_SegmentMixin, TwinMujoco):
	"""The unchanged expert, recording which segment each of its frames belongs to (events and the per-frame
	``segment`` column). With ``action_noise`` set, OU noise is added to the executed arm targets on segments in
	``action_noise.kinds`` while the clean teacher target is recorded and flagged with ``noisy=True``."""

	def __init__(self, *args, action_noise: ActionNoise | None = None, **kwargs):
		if action_noise is not None and not isinstance(action_noise, ActionNoise):
			raise TypeError(f"action_noise must be ActionNoise or None, got {type(action_noise).__name__}")
		self.action_noise = action_noise
		self._ou_noise = None if action_noise is None else action_noise.make_process()
		super().__init__(*args, **kwargs)
		self._segment_setup()

	def _segment_begin(self, seg: Segment) -> int:
		if self._ou_noise is not None:
			self._ou_noise.reset()
		return super()._segment_begin(seg)

	def _segment_end(self, reason: str):
		if self._open_segment is not None and self._open_segment["kind"] == "carry_to_bowl":
			from ..mujoco_cell.grasp_geometry import Kinematics

			kin = getattr(self, "_carry_fk", None)
			if kin is None or kin.model is not self.cell.model:
				kin = self._carry_fk = Kinematics(self.cell)
			pos, _ = kin.site_pose(self.cell.joints())
			self._last_carry_tcp = pos.copy()
		if self._ou_noise is not None:
			self._ou_noise.reset()
		super()._segment_end(reason)

	def _apply_target(self, obs, target, phase):
		if (
			self._ou_noise is not None
			and self.action_noise is not None
			and self._open_segment is not None
			and self._open_segment["kind"] in self.action_noise.kinds
			and self.action_noise.sigma_rad > 0.0
		):
			delta = self._ou_noise.step()
			executed = np.clip(target + delta, self.expert._lo, self.expert._hi)
			return super()._apply_target(obs, target, phase, executed_target=executed, noisy=True)
		return super()._apply_target(obs, target, phase)

	def _tag_carry(self):
		ex = self.expert
		if "_carry" in vars(ex):
			return
		original = ex._carry  # bound method of the class
		ex._carry_frame = -1

		def carry(frames):
			for item in original(frames):
				ex._carry_frame = ex._frame_idx  # index this frame gets in act()
				yield item

		ex._carry = carry

	def _split(self, generator, name, segs):
		ex, parts = self.expert, _PARTS[name]
		current = -1
		try:
			for target, phase in generator:
				carried = ex._carry_frame == ex._frame_idx
				if current < 0 or not parts[current][1](phase, carried):
					nxt = next((j for j in range(max(current, 0), len(parts)) if parts[j][1](phase, carried)), None)
					if nxt is None or nxt == current:
						raise SegmentMappingError(f"{name}: frame (phase {phase}, carried {carried}) fits no part")
					if current >= 0:
						self._segment_end("done")
					current = nxt
					seg = segs[parts[nxt][0]]
					if seg.kind == "grasp":
						# ClosedLoopExpert._approach_and_descend has planned the grasp before its first frame.
						seg = with_grasp_roll(seg, ex.grasp_info["roll"])
					self._segment_begin(seg)
				yield target, phase
		finally:
			generator.close()

	def _execute(self, generator):
		self._tag_carry()
		form, value = self._motion(generator)
		if form == "simple":
			self._segment_begin(value)
			wrapped = generator
		else:
			wrapped = self._split(generator, *value)
		try:
			error = super()._execute(wrapped)
		except BaseException:
			if self._open_segment is not None:
				self._segment_end("error")
			raise
		if self._open_segment is not None:
			lost = self._open_segment["segment"].held and self.expert.lost
			self._segment_end(error or ("brick_lost" if lost else "done"))
		if form == "composite" and value[0] == "grasp_and_lift" and self.held is not None:
			self.lift_poses.append(np.asarray(self.held[0], dtype=np.float64).copy())
		return error


# Obstruction exit of the clamp wait (docs/vla/CLAMP_WAIT_VALIDATION_2026-11-23.md v2): the held command is fixed, so
# an arm that stays away from it is pushed back by something. Consecutive observations in the wait with the arm joints
# (j0-j4) at least ``clamp_wait_obstruct_rad`` from the held command end the wait as ``"obstructed"``.
CLAMP_WAIT_OBSTRUCT_FRAMES = 3
# Closed-empty exit of the clamp wait (same doc, v3): a brick between the jaws holds the gripper open well above its
# command (clamped holds stay >= 0.049 rad above it on val2/val3), so a gripper within ``clamp_wait_empty_rad`` of the
# held gripper command has closed on nothing. Consecutive such observations end the wait as ``"empty"``. Gripper
# position only (no effort): the realistic sensing contract does not assume the servo load is read.
CLAMP_WAIT_EMPTY_FRAMES = 3


def clamp_wait_outcome(
	gstate: GripperState | None, clamp_run: int, waited: int, p, obstruct_run: int = 0, empty_run: int = 0
) -> str | None:
	"""``ClosedLoopExpert._close_and_clamp``'s wait, one held frame at a time: ``"blocked"``, ``"clamped"`` (clamped and
	still for ``p.clamp_still_frames``), ``"obstructed"`` after ``CLAMP_WAIT_OBSTRUCT_FRAMES`` obstructed frames,
	``"empty"`` after ``CLAMP_WAIT_EMPTY_FRAMES`` closed-empty frames (each counted only with its exit on), ``"timeout"``
	after ``p.clamp_timeout_frames`` held frames, else None."""
	if gstate is GripperState.BLOCKED:
		return "blocked"
	if clamp_run >= p.clamp_still_frames:
		return "clamped"
	if obstruct_run >= CLAMP_WAIT_OBSTRUCT_FRAMES:
		return "obstructed"
	if empty_run >= CLAMP_WAIT_EMPTY_FRAMES:
		return "empty"
	if waited >= p.clamp_timeout_frames:
		return "timeout"
	return None


class StudentTwin(_SegmentMixin, TwinMujoco):
	"""Graph backend whose skills issue segments to ``policy`` instead of running the expert's motion generators.

	The expert object stays as the planners' kinematic model and parameter set: ``expert.last`` follows the applied
	targets, and its ``lost`` / ``_no_dual`` / ``aborted`` fields carry the lost-brick and clamp checks, so the
	inherited skill code reads them as it reads the expert's."""

	def __init__(
		self,
		layout,
		scorer,
		record,
		policy: SegmentPolicy,
		*,
		segment_monitor: Callable[[Segment, int], None] | None = None,
		grasp_tail: str | None = None,
		clamp_wait: bool = False,
		clamp_wait_obstruct_rad: float | None = None,
		clamp_wait_empty_rad: float | None = None,
		**kwargs,
	):
		"""``grasp_tail``: first-grasp tail experiment mode (agent/grasp_tail.py: ``act`` logs only, ``act-wait``,
		``expert-once``); None runs without the instrumentation.

		``clamp_wait``: clamp-verified grasp termination. When the policy ends ANY grasp segment before the clamp is
		confirmed, hold its last command until clamped and still, blocked, or ``clamp_timeout_frames`` (the expert's
		``_close_and_clamp`` wait). The A+wait arm of the grasp-tail A/B did this on the first triggered grasp only.

		``clamp_wait_obstruct_rad``: obstruction exit of the clamp wait (``CLAMP_WAIT_OBSTRUCT_FRAMES``); None keeps the
		v1 rule (hold to clamp, blocked or timeout).

		``clamp_wait_empty_rad``: closed-empty exit of the clamp wait (``CLAMP_WAIT_EMPTY_FRAMES``); None leaves it off."""
		if grasp_tail is not None and grasp_tail not in GRASP_TAIL_MODES:
			raise ValueError(f"grasp_tail must be None or one of {GRASP_TAIL_MODES}, got {grasp_tail!r}")
		if clamp_wait and grasp_tail == "act-wait":
			raise ValueError("clamp_wait already covers the act-wait arm; use one of them")
		self.grasp_tail = grasp_tail
		if clamp_wait_obstruct_rad is not None and not (clamp_wait and clamp_wait_obstruct_rad > 0):
			raise ValueError("clamp_wait_obstruct_rad needs clamp_wait and a positive threshold")
		self.clamp_wait = bool(clamp_wait)
		self.clamp_wait_obstruct_rad = clamp_wait_obstruct_rad
		if clamp_wait_empty_rad is not None and not (clamp_wait and clamp_wait_empty_rad > 0):
			raise ValueError("clamp_wait_empty_rad needs clamp_wait and a positive threshold")
		self.clamp_wait_empty_rad = clamp_wait_empty_rad
		self.grasp_tail_log: GraspTail | None = None  # the first grasp's switch decision and telemetry
		self._tail_pending: GraspPlan | None = None
		self._first_grasp_done = False
		self.segment_policy = policy  # not ``policy``: TwinMujoco keeps the graph's Policy there
		# Separate channel for replay checks and grading: receives the whole segment, diagnostics included. The policy
		# itself only ever gets policy_inputs(segment).
		self.segment_monitor = segment_monitor
		super().__init__(layout, scorer, record, **kwargs)
		self._segment_setup()

	def _segment_setup(self):
		super()._segment_setup()
		self._q_prev: np.ndarray | None = None
		self._prev_action: np.ndarray = INITIAL_Q.copy()

	def _checkpoint_extra(self):
		extra = super()._checkpoint_extra()
		extra["q_prev"] = None if self._q_prev is None else self._q_prev.tolist()
		extra["prev_action"] = self._prev_action.tolist()
		return extra

	def _restore_extra(self, extra):
		super()._restore_extra(extra)
		if extra is None:
			return
		qp = extra.get("q_prev")
		self._q_prev = None if qp is None else np.array(qp, dtype=np.float64)
		pa = extra.get("prev_action")
		self._prev_action = INITIAL_Q.copy() if pa is None else np.array(pa, dtype=np.float64)

	# -- the segment executor -----------------------------------------------------------------------------------

	def run_segment(self, seg: Segment) -> str | None:
		"""Run one segment to ``segment_done``. Returns a failure code (``ACTION_BUDGET_EXHAUSTED``,
		``SEGMENT_FRAME_BUDGET``, a fault) or None; a lost brick ends a held segment with ``expert.lost`` set."""
		ex, p = self.expert, self.expert.p
		sid = self._segment_begin(seg)
		if self.segment_monitor is not None:
			self.segment_monitor(seg, sid)
		task = policy_inputs(seg)
		self.segment_policy.start(dict(task), sid)
		frames = 0
		after_step = False  # the next observation follows a step of this segment (the checks read it)
		clamp_run = 0
		monitor = (
			GripperMonitor(GripperMonitorParams(effort_min=0.0, use_effort=False))
			if self.realistic
			else GripperMonitor()
		)
		self._clamped = self._blocked = False
		# First-grasp tail experiment (agent/grasp_tail.py); None leaves the segment loop exactly as before.
		tail = None
		if seg.kind == "grasp" and self._tail_pending is not None:
			tail = GraspTail(self.grasp_tail, self._tail_pending, p.pregrasp_dz_m, sid)
			self._tail_pending = None
			self.grasp_tail_log = tail
		# Where this frame's command comes from: act, act_wait_hold (A+wait), expert_tail (B), clamp_wait_hold.
		source = "act"
		waited = 0  # hold frames of a clamp wait (act_wait_hold / clamp_wait_hold)
		obstruct_run = 0  # consecutive held frames with the arm away from the held command (obstruction exit)
		arm_err_max = 0.0
		empty_run = 0  # consecutive held frames with the gripper at its held command (closed-empty exit)
		grip_err_min = float("inf")
		executed: np.ndarray | None = None  # last command executed in this segment
		gstate = None  # the segment monitor's state at the current observation
		expert_gen = hold = saved_expert = None

		def end(reason: str, outcome: str | None = None):
			if tail is not None:
				tail.outcome = outcome if outcome is not None else tail.outcome
				tail.finish(self.ep.n_frames, frames, reason)
			if saved_expert is not None:  # the expert's per-frame fields are the student's again
				ex.obs, ex.gripper_state = saved_expert
			self._segment_end(reason)

		while True:
			if self.total_frames >= MAX_TOTAL_FRAMES:
				end("budget", "budget" if source != "act" else None)
				return "ACTION_BUDGET_EXHAUSTED"
			if self.on_frame is not None:
				self.on_frame(self)
			obs = self._observe()
			if after_step:
				after_step = False
				f = np.asarray(obs["jaw_normal"], dtype=np.float64)
				if seg.held:  # ClosedLoopExpert._carry: consecutive frames without contact on both jaws
					dual = bool(f[0] > p.lost_force_n and f[1] > p.lost_force_n)
					ex._no_dual = 0 if dual else ex._no_dual + 1
					if ex._no_dual >= p.lost_frames:
						ex.lost = True
				if seg.kind == "grasp":  # ClosedLoopExpert._close_and_clamp: clamped and still
					v = float(np.linalg.norm(obs["brick_vel"]))
					clamp_run = clamp_run + 1 if (min(f) >= p.clamp_force_n and v < p.clamp_still_speed_mps) else 0
					state, _ = monitor.update(
						float(obs["gripper_cmd"]), float(obs["q"][5]), float(obs["effort_frac"]), obs["jaw_normal"]
					)
					gstate = state
					self._blocked |= state is GripperState.BLOCKED
			if seg.held and ex.lost:
				self._segment_end("brick_lost")
				return None
			q = np.asarray(obs["q"], dtype=np.float64).copy()
			if tail is not None:
				fired = tail.observe(
					frame=self.ep.n_frames,
					segment_frame=frames,
					obs=obs,
					q=q,
					executed=executed,
					clamp_run=clamp_run,
					gripper_state=gstate,
					clamp_frames=p.clamp_still_frames,
				)
				if fired and source == "act" and tail.mode == "expert-once":
					source = "expert_tail"
					saved_expert = (ex.obs, ex.gripper_state)
					ex.obs = obs
					expert_gen = expert_tail(ex, tail.plan)
					tail.event("expert_start", self.ep.n_frames, frames)
			phase = None
			if source == "expert_tail":
				ex.obs = obs  # ClosedLoopExpert.act: the generator reads this frame's observation and gripper state
				ex.gripper_state = gstate if gstate is not None else GripperState.FREE
				try:
					target, phase = next(expert_gen)
				except StopIteration as stop:
					outcome = stop.value
					self._clamped = outcome == "clamped"
					self._blocked |= outcome == "blocked"
					tail.event("expert_end", self.ep.n_frames, frames, outcome=outcome)
					end("done", outcome)
					return None
				raw = np.asarray(target, dtype=np.float64)
			else:
				step = None
				if source == "act":
					q_prev = q.copy() if self._q_prev is None else self._q_prev.copy()
					prev_action = self._prev_action.copy()
					gripper_cmd = float(obs["gripper_cmd"])
					step = self.segment_policy.act(
						StudentObservation(
							frame=self.ep.n_frames,
							task=dict(task),
							segment_id=sid,
							segment_frame=frames,
							q=q,
							_images=self._images,
							q_prev=q_prev,
							prev_action=prev_action,
							gripper_cmd=gripper_cmd,
						)
					)
					if step.done:
						eligible = tail is not None and tail.triggered and clamp_run < p.clamp_still_frames
						if tail is not None:  # every arm: where A+wait would start waiting (vla/rgb_pin.py)
							tail.event(
								"act_done", self.ep.n_frames, frames, clamp_run=int(clamp_run), wait_eligible=eligible
							)
						if self.clamp_wait and seg.kind == "grasp" and clamp_run < p.clamp_still_frames:
							source = "clamp_wait_hold"
						elif eligible and tail.mode == "act-wait":
							source = "act_wait_hold"
						else:
							self._clamped = clamp_run >= p.clamp_still_frames
							end("done")
							return None
						hold = self._prev_action.copy()
						if tail is not None:
							tail.event("wait_start", self.ep.n_frames, frames)
				if source in ("act_wait_hold", "clamp_wait_hold"):
					if source == "clamp_wait_hold" and self.clamp_wait_obstruct_rad is not None:
						arm_err = float(np.max(np.abs(hold[:5] - q[:5])))
						arm_err_max = max(arm_err_max, arm_err)
						obstruct_run = obstruct_run + 1 if arm_err >= self.clamp_wait_obstruct_rad else 0
					if source == "clamp_wait_hold" and self.clamp_wait_empty_rad is not None:
						grip_err = abs(float(q[5]) - float(hold[5]))
						grip_err_min = min(grip_err_min, grip_err)
						empty_run = empty_run + 1 if grip_err < self.clamp_wait_empty_rad else 0
					outcome = clamp_wait_outcome(gstate, clamp_run, waited, p, obstruct_run, empty_run)
					if outcome is not None:
						self._clamped = outcome == "clamped"
						if tail is not None:
							tail.event("wait_end", self.ep.n_frames, frames, outcome=outcome, wait_frames=waited)
						if source == "clamp_wait_hold":
							extra = (
								{} if self.clamp_wait_obstruct_rad is None else {"arm_err_max": round(arm_err_max, 5)}
							)
							if self.clamp_wait_empty_rad is not None:
								extra["grip_err_min"] = round(grip_err_min, 5)
							self.event("clamp_wait", id=sid, frames=waited, outcome=outcome, **extra)
						end("done", outcome)
						return None
					waited += 1
					if tail is not None:
						tail.wait_frames = waited
					raw, phase = hold.copy(), "grasp"
				else:
					raw = np.asarray(step.target, dtype=np.float64)
					phase = step.phase
			if frames >= seg.max_frames:
				end("frame_budget", "frame_budget" if source != "act" else None)
				return "SEGMENT_FRAME_BUDGET"
			if raw.shape != (6,) or not np.isfinite(raw).all():
				raise ValueError("a policy target must be 6 finite joint positions")
			target = np.clip(raw, ex._lo, ex._hi)  # as ClosedLoopExpert.act
			ex.clipped_frames += bool(np.any(target != raw))
			ex.last = target.copy()
			ex._frame_idx += 1
			phase = phase or seg.phase
			if phase not in PHASES:
				raise ValueError(f"unknown phase {phase!r}")
			if tail is not None:
				tail.command(source, target)
			error = self._apply_target(obs, target, phase)
			self._q_prev = q
			self._prev_action = target.copy()
			executed = target.copy()
			frames += 1
			after_step = True
			if error:
				end("fault", "fault" if source != "act" else None)
				return error

	def _images(self):
		if self.record.video and self.record.images is not None:
			return self.record.images
		return self.record._render(self.cell)

	# -- motion generators -> segments --------------------------------------------------------------------------

	def _execute(self, generator):
		form, value = self._motion(generator)
		generator.close()  # never run: the policy moves the arm
		if form == "simple":
			return self.run_segment(value)
		name, segs = value
		return getattr(self, f"_student_{name}")(segs)

	def _student_grasp_and_lift(self, segs):
		"""TwinMujoco._grasp's grasp_and_lift: grasp, clamp check, carried-brick belief at the grasp, lift."""
		ex = self.expert
		target, roll, open_g = self._planned_grasp()
		seg = with_grasp_roll(segs["grasp"], roll)
		if self.grasp_tail is not None and not self._first_grasp_done:
			# The plan snapshot of this very call: the tail never re-plans inside the segment.
			self._tail_pending = GraspPlan(target, roll, open_g, self.ep.n_frames, self._perception_frame())
		self._first_grasp_done = True
		self._belief_obs = True
		try:
			error = self.run_segment(seg)
		finally:
			self._belief_obs = False
		if error:
			return error
		if not self._clamped:
			ex.aborted = "gripper_blocked" if self._blocked else "no_clamp"
			return None
		if self.realistic:
			self._grip_from_brick = np.linalg.inv(self._grip_pose()) @ self._table_belief_tf()
		error = self.run_segment(segs["lift"])
		if error:
			return error
		# The held lift pose (base of the show, park and place moves) is where the lift ended. The grasp roll is the
		# wrist-roll command the brick was grasped and lifted with (the expert's lift pose has q[4] = its grasp roll).
		self.held = (ex.last.copy(), float(ex.last[4]))
		self.lift_poses.append(ex.last.copy())
		return None

	def _perception_frame(self) -> int | None:
		"""Episode frame of the last ``brick_perceived`` event that found the brick."""
		for e in reversed(self.record.events):
			if e.get("kind") == "brick_perceived" and e.get("found"):
				return int(e["frame"])
		return None

	def _planned_grasp_roll(self) -> float:
		"""The wrist roll ``ClosedLoopExpert._grasp_plan`` picks for the perceived brick, as the expert's grasp would
		plan it at its first frame. Leaves the expert's state as it was (the student, not the expert, moves next)."""
		return self._planned_grasp()[1]

	def _planned_grasp(self) -> tuple[np.ndarray, float, float]:
		"""``(gripper-frame target, wrist roll, jaw opening)`` of the grasp ``ClosedLoopExpert._grasp_plan`` plans for
		the perceived brick at its first frame. Leaves the expert's state as it was."""
		ex = self.expert
		obs = dict(ex.obs)
		if self.realistic:
			belief = self._brick_belief()
			if belief is not None:
				obs["brick_pos"], obs["brick_quat"] = belief
		elif self.brick_est is not None:  # TwinMujoco._grasp plans on the perceived pose (``_belief_obs``)
			obs["brick_pos"], obs["brick_quat"] = self.brick_est.pos.copy(), self.brick_est.quat_xyzw()
		saved = (ex.obs, ex._active_open_g, dict(ex.grasp_info), ex.kin.data.qpos.copy())
		z_saved = vars(ex).get("z_grasp")
		try:
			ex.obs = obs
			target, roll, _ = ex._grasp_plan(open_g=ex._active_open_g)
			open_g = float(ex._active_open_g)  # _grasp_plan may pick a narrower opening (pre_narrow_open)
		finally:
			ex.obs, ex._active_open_g, ex.grasp_info, qpos = saved
			ex.kin.data.qpos[:] = qpos
			if z_saved is None:
				vars(ex).pop("z_grasp", None)
			else:
				ex.z_grasp = z_saved
		return np.asarray(target, dtype=np.float64).copy(), float(roll), open_g

	def _student_park(self, segs):
		"""TwinMujoco._park_held's park: back to the lift pose, put down at the park spot, ready pose."""
		ex = self.expert
		if error := self.run_segment(segs["return_to_lift"]):
			return error
		if ex.lost:
			return None
		if error := self.run_segment(segs["lower_on_table"]):
			return error
		if not ex.lost and (error := self.run_segment(segs["release_on_table"])):
			return error
		if self.realistic and self._grip_from_brick is not None:
			# The put-down retreats straight up, so the gripper still has the release orientation.
			rot = (self._grip_pose() @ self._grip_from_brick)[:3, :3]
			self._expected_yaw = math.atan2(rot[1, 0], rot[0, 0])
		if not ex.lost:
			return self.run_segment(segs["return_ready"])
		return None

	def _student_release(self, segs):
		"""TwinMujoco.place's release: back to the lift pose if away from it, carry into the bowl, release, home."""
		ex = self.expert
		q_lift, _ = self.held
		if np.max(np.abs(ex.last - q_lift)) > 0.01 and (error := self.run_segment(segs["return_to_lift"])):
			return error
		if ex.lost:
			return None
		if error := self.run_segment(segs["carry_to_bowl"]):
			return error
		if not ex.lost:
			return self.run_segment(segs["release_in_bowl"])
		return None


# -- replay -------------------------------------------------------------------------------------------------------


class ReplayMismatch(RuntimeError):
	"""The student asked for a segment the recording does not have next."""


@dataclass
class RecordedSegment:
	segment: Segment
	actions: np.ndarray  # (n, 6) float64, executed targets
	states: np.ndarray  # (n, 6) float64, joint readings before each action
	phases: list[str]
	reason: str


class ReplayPolicy:
	"""Plays a recorded expert episode segment by segment: the recorded float64 targets, then ``segment_done`` where
	the expert's segment ended. Each observed joint reading is compared with the recorded one (bitwise).

	As a policy it sees only the task (``policy_inputs``). Pass :meth:`check_segment` as the student's
	``segment_monitor`` to also match each whole segment, diagnostics included, with the recorded one."""

	def __init__(self, segments: list[RecordedSegment]):
		self.segments = segments
		self.index = -1
		self.state_mismatches = 0
		self.max_state_error = 0.0

	@classmethod
	def from_trace(cls, trace_dir: Path) -> ReplayPolicy:
		trace_dir = Path(trace_dir)
		events = [json.loads(line) for line in (trace_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
		if any(e["kind"] == "rewound" for e in events):
			raise ValueError("replay of rewound (multi-branch) episodes is not supported")
		with np.load(trace_dir / "branch-0" / "actions.npz", allow_pickle=False) as rows:
			ids, actions, states = rows["segment"], rows["action_f64"], rows["state_f64"]
			phases = [str(p) for p in rows["phase"]]
		starts = {e["id"]: e for e in events if e["kind"] == "segment"}
		ends = {e["id"]: e for e in events if e["kind"] == "segment_end"}
		out = []
		for sid in sorted(starts):
			start, end = starts[sid], ends[sid]
			rows_of = np.flatnonzero(ids == sid)
			if len(rows_of) != end["frames"] or (len(rows_of) and rows_of[0] != start["frame"]):
				raise ValueError(f"segment {sid}: frames in actions.npz do not match its events")
			if len(rows_of) and not np.array_equal(rows_of, np.arange(rows_of[0], rows_of[0] + len(rows_of))):
				raise ValueError(f"segment {sid} is not contiguous")
			out.append(
				RecordedSegment(
					Segment(start["segment_kind"], start["params"]),
					actions[rows_of],
					states[rows_of],
					[phases[i] for i in rows_of],
					end["reason"],
				)
			)
		if (ids < 0).any():
			raise ValueError("actions outside any segment")
		return cls(out)

	def start(self, task: dict[str, Any], segment_id: int) -> None:
		self.index += 1
		if self.index >= len(self.segments):
			raise ReplayMismatch(f"segment {segment_id} ({task['kind']}) is past the end of the recording")
		recorded = policy_inputs(self.segments[self.index].segment)
		if task != recorded or segment_id != self.index:
			raise ReplayMismatch(f"segment {segment_id}: student task {task} != recorded {recorded}")

	def check_segment(self, segment: Segment, segment_id: int) -> None:
		"""Diagnostics channel (``StudentTwin(segment_monitor=...)``): the whole segment must match the recording."""
		if segment_id >= len(self.segments):
			raise ReplayMismatch(f"segment {segment_id} ({segment.kind}) is past the end of the recording")
		rec = self.segments[segment_id]
		if rec.segment.key() != segment.key():
			raise ReplayMismatch(f"segment {segment_id}: student {segment.key()} != recorded {rec.segment.key()}")

	def act(self, obs: StudentObservation) -> PolicyStep:
		rec, i = self.segments[self.index], obs.segment_frame
		if i < len(rec.states):
			err = float(np.max(np.abs(obs.q - rec.states[i])))
			self.max_state_error = max(self.max_state_error, err)
			self.state_mismatches += err != 0.0
		elif i == len(rec.states) and self.index + 1 < len(self.segments):
			nxt = self.segments[self.index + 1]
			if len(nxt.states):  # the segment-end observation is the next segment's first one
				err = float(np.max(np.abs(obs.q - nxt.states[0])))
				self.max_state_error = max(self.max_state_error, err)
				self.state_mismatches += err != 0.0
		if i >= len(rec.actions):
			return PolicyStep(done=True)
		return PolicyStep(target=rec.actions[i].copy(), phase=rec.phases[i])


# -- comparison ---------------------------------------------------------------------------------------------------

VOLATILE_EVENT_KEYS = ("wall_time_s",)


def comparable_events(events: list[dict]) -> list[dict]:
	"""Events without wall-clock fields (``wall_time_s``, the detector's ``elapsed_ms``)."""
	out = []
	for e in events:
		e = {k: v for k, v in e.items() if k not in VOLATILE_EVENT_KEYS}
		if isinstance(e.get("detector"), dict):
			e["detector"] = {k: v for k, v in e["detector"].items() if k != "elapsed_ms"}
		out.append(e)
	return out


def telemetry_digest(twin: TwinMujoco) -> str:
	tel = twin.ep.telemetry()
	h = hashlib.sha256()
	for key in ("q_cmd", "qpos", "tcp_pos", "brick_pos", "brick_quat", "jaw_forces", "actuator_torques"):
		h.update(np.ascontiguousarray(tel[key], dtype=np.float64).tobytes())
	return h.hexdigest()


def _segments_of(events):
	ends = {e["id"]: e for e in events if e["kind"] == "segment_end"}
	return [
		(e["segment_kind"], json.dumps(e["params"], sort_keys=True), e["frame"], e["frame"] + ends[e["id"]]["frames"])
		for e in events
		if e["kind"] == "segment"
	]


def _scan_stops(events):
	"""Per ``scan_wrist`` visit: (views captured, stopped early on a defective view)."""
	out, cur = [], None
	for e in events:
		if e["kind"] == "node":
			cur = [0, False] if e["node"] == "scan_wrist" else None
			if cur is not None:
				out.append(cur)
		elif e["kind"] == "view" and cur is not None:
			o = e["observation"]
			cur[0] += 1
			cur[1] = cur[1] or (o["score"] is not None and bool(o["faces"]) and o["score"] >= o["threshold"])
	return [tuple(x) for x in out]


# The MuJoCo renderer is not repeatable: rendering one simulator state several times in a row gives images that
# differ by one 8-bit level on a few dozen pixels near shadow edges (measured on bench16 case 10, wrist camera: 8
# distinct images in 12 renders, max difference 1, at most 46 of 307200 pixels; repeatable with shadows off or
# without multisampling). Captured images (full frame and detector crop) are therefore compared within this bound.
# The detector sees only the crop: when a view's crop is bitwise equal its total score and every head score are
# compared exactly, even if its full frame differs by render noise; only a crop that differs exempts the scores
# (reported instead). The decision (score >= threshold) is always compared exactly.
RENDER_MAX_LEVEL = 1
RENDER_MAX_FRACTION = 0.001


def _image_diff(a: Path, b: Path) -> tuple[int, float]:
	import cv2

	x, y = cv2.imread(str(a)), cv2.imread(str(b))
	if x is None or y is None or x.shape != y.shape:
		return 255, 1.0
	d = np.abs(x.astype(np.int16) - y.astype(np.int16)).max(axis=2)
	return int(d.max()), float((d > 0).mean())


def compare_runs(expert: dict, student: dict, expert_dir: Path | None = None, student_dir: Path | None = None):
	"""Gate checks between two runs; returns ``(checks, info)``. Each run: ``result``, ``events``, ``digest``,
	``crops`` (sha256 per keyframe file: ``<view>-frame.png`` full image, ``<view>-crop.png`` detector input). With
	the trace directories, keyframes that are not bitwise equal are compared within the renderer's repeatability
	(RENDER_MAX_LEVEL / RENDER_MAX_FRACTION). Full-frame noise and crop noise are flagged separately: only a view whose
	crop differs has its scores reported instead of checked; with a bitwise-equal crop the total and per-head
	detector scores must match exactly. ``smoothness.clipped_frames`` is excluded from the result check: it counts
	the expert generator's raw targets that act() clipped, which a replay of the executed (already clipped) targets
	cannot see."""
	ee, se = comparable_events(expert["events"]), comparable_events(student["events"])
	image_ok, worst = expert["crops"].keys() == student["crops"].keys(), (0, 0.0)
	noisy_frames, noisy = set(), set()  # view ids whose full frame / detector crop differ by render noise
	for name in sorted(set(expert["crops"]) & set(student["crops"])):
		if expert["crops"][name] == student["crops"][name]:
			continue
		stem, _, suffix = name.removesuffix(".png").rpartition("-")
		view_id = stem.replace("-", ":", 2)  # keyframe stem -> view id
		(noisy if suffix == "crop" else noisy_frames).add(view_id)
		if expert_dir is None or student_dir is None:
			image_ok = False
			continue
		level, frac = _image_diff(Path(expert_dir, "keyframes", name), Path(student_dir, "keyframes", name))
		worst = (max(worst[0], level), max(worst[1], frac))
		image_ok &= level <= RENDER_MAX_LEVEL and frac <= RENDER_MAX_FRACTION

	def views(ev):
		return [e["observation"] for e in ev if e["kind"] == "view"]

	def decided(v):
		return v["score"] is not None and v["score"] >= v["threshold"]

	def masked(ev):  # scores of views whose detector crop differs by render noise are reported, not compared
		out = json.loads(json.dumps(ev))
		for e in out:
			if e["kind"] == "view" and e["observation"]["id"] in noisy:
				e["observation"]["score"] = decided(e["observation"])
				e["detector"]["head_scores"] = None
		return out

	def result(r):
		r = json.loads(json.dumps(r))
		r.get("smoothness", {}).pop("clipped_frames", None)
		return r

	def exact_scores(ev):  # total and every head score of each view whose detector crop is bitwise equal
		return [
			(e["observation"]["id"], e["observation"]["score"], (e.get("detector") or {}).get("head_scores"))
			for e in ev
			if e["kind"] == "view" and e["observation"]["id"] not in noisy
		]

	ev_, sv_ = views(ee), views(se)
	score_delta = [
		abs(a["score"] - b["score"])
		for a, b in zip(ev_, sv_, strict=False)
		if a["id"] in noisy and a["score"] is not None and b["score"] is not None
	]
	checks = {
		"captures": [(v["camera"], v["frame"]) for v in ev_] == [(v["camera"], v["frame"]) for v in sv_],
		"images": image_ok,
		"decisions": [(v["faces"], v["threshold"], decided(v)) for v in ev_]
		== [(v["faces"], v["threshold"], decided(v)) for v in sv_],
		"scores": exact_scores(ee) == exact_scores(se),
		"early_stops": _scan_stops(ee) == _scan_stops(se),
		"segments": _segments_of(ee) == _segments_of(se),
		"destination": expert["result"]["destination"] == student["result"]["destination"],
		"evaluate": result(expert["result"]) == result(student["result"]),
		"telemetry": expert["digest"] == student["digest"],
		"events": masked(ee) == masked(se),
	}
	info = {
		"render_noise_views": sorted(noisy | noisy_frames),  # any keyframe of the view differs
		"render_noise_frame_views": sorted(noisy_frames),  # full frame differs (crop may be equal)
		"render_noise_crop_views": sorted(noisy),  # detector crop differs: scores reported, not compared
		"render_noise_max_level": worst[0],
		"render_noise_max_fraction": worst[1],
		"render_noise_max_score_delta": max(score_delta, default=0.0),
	}
	return checks, info


def crop_digests(trace_dir: Path) -> dict[str, str]:
	return {
		p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(trace_dir, "keyframes").glob("*.png"))
	}


def run_episode(
	case,
	scorer,
	out: Path,
	*,
	policy: SegmentPolicy | None = None,
	segment_monitor: Callable[[Segment, int], None] | None = None,
	sensing="realistic",
	video=False,
	grasp_tail: str | None = None,
	clamp_wait: bool = False,
	clamp_wait_obstruct_rad: float | None = None,
	clamp_wait_empty_rad: float | None = None,
	action_noise: ActionNoise | None = None,
):
	"""Run one case through the graph: the expert (with segment markers) when ``policy`` is None, else the student
	(``segment_monitor``: the diagnostics channel, see :class:`StudentTwin`). Writes the trace to ``out`` and returns
	``result``, ``events``, ``digest``, ``crops`` and the twin's counters."""
	from .twin_graph import Policy, build_graph
	from .twin_mujoco import sample_case
	from .twin_record import EpisodeTrace

	if policy is not None and action_noise is not None:
		raise ValueError("action_noise is only supported for the expert tracer (policy=None)")
	trace = EpisodeTrace(out, video=video)
	layout = sample_case(*case)
	if policy is None:
		twin = ExpertTracer(layout, scorer, trace, sensing=sensing, action_noise=action_noise)
	else:
		twin = StudentTwin(
			layout,
			scorer,
			trace,
			policy,
			segment_monitor=segment_monitor,
			sensing=sensing,
			grasp_tail=grasp_tail,
			clamp_wait=clamp_wait,
			clamp_wait_obstruct_rad=clamp_wait_obstruct_rad,
			clamp_wait_empty_rad=clamp_wait_empty_rad,
		)
	try:
		result = build_graph(twin, Policy()).invoke({}, config={"recursion_limit": 64})["result"]
		provenance = {"engine": "mujoco", "sensing": sensing, "case": list(case), "student": policy is not None}
		if action_noise is not None:
			provenance["action_noise"] = {
				"sigma_rad": action_noise.sigma_rad,
				"tau_s": action_noise.tau_s,
				"kinds": list(action_noise.kinds),
				"seed": action_noise.seed,
			}
		if grasp_tail is not None:
			provenance["grasp_tail"] = grasp_tail
		if clamp_wait:
			provenance["clamp_wait"] = True
		if clamp_wait_obstruct_rad is not None:
			provenance["clamp_wait_obstruct_rad"] = clamp_wait_obstruct_rad
		if clamp_wait_empty_rad is not None:
			provenance["clamp_wait_empty_rad"] = clamp_wait_empty_rad
		trace.finish(result, provenance)
		if grasp_tail is not None:
			log = twin.grasp_tail_log
			payload = {
				"version": "grasp-tail-v1",
				"mode": grasp_tail,
				"first_grasp": None if log is None else log.to_json(),
			}
			(out / "grasp_tail.json").write_text(json.dumps(payload, allow_nan=True), encoding="utf-8")
		run = {
			"case": list(case),
			"result": json.loads(json.dumps(result)),
			"events": trace.events,
			"digest": telemetry_digest(twin),
			"crops": crop_digests(out),
			"frames": twin.ep.n_frames,
			"lift_poses": [q.tolist() for q in twin.lift_poses],
			"clipped_frames": int(twin._smoothness().get("clipped_frames", 0)),
		}
		(out / "run.json").write_text(json.dumps(run, allow_nan=False), encoding="utf-8")
		return run
	finally:
		twin.close()
