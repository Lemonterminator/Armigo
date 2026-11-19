"""Student execution interface: the six-face skills run as motion segments driven by a per-frame policy.

A **segment** is the motion between two decision points of the expert's skill code: places where the skill waits for
a sensor result (a capture and detector score, a perception, a clamp or lost-brick check, a reading of the gripper
pose) before it decides what comes next. The graph and the skills' graph-side logic keep deciding; a policy only
moves the arm within one segment and says when the segment is done (``segment_done``).

Three pieces:

- :class:`Segment` names a segment (``kind``, JSON ``params``) and gives its text instruction (a VLA task string).
- :class:`StudentTwin` is a :class:`~.twin_mujoco.TwinMujoco` whose motion generators are never run. Each call of
  ``_execute(generator)`` is mapped to one segment, or to a short sequence of segments for the three skills whose
  generator has a decision point inside (grasp and lift, park, release into a bowl), and each segment is executed
  by :meth:`StudentTwin.run_segment`: observation -> ``policy.act`` -> the expert's own command path
  (``TwinMujoco._apply_target``: joint-range clip, record, command filter, ``Episode.step``). All graph-side logic
  (scan planning and skipping, clearance checks, captures, early stop after a defective view, perception, placement
  checks, in-place recovery) is the unchanged ``TwinMujoco`` code; the few decisions taken inside a generator are
  reproduced between segments (clamp check, carried-brick belief at the grasp, lost-brick monitor, put-down yaw).
- :class:`ExpertTracer` is the unchanged expert with segment markers: the same mapping labels its frames, so expert
  recordings carry the same segment ids, instructions and ends a student is driven with.

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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from ..constants import PHASES
from ..mujoco_cell.gripper_state import GripperMonitor, GripperState
from .twin_mujoco import MAX_TOTAL_FRAMES, WRIST_MIN_PX, TwinMujoco

# kind -> (instruction template, held: the brick is carried (lost-brick monitor on), default phase, frame budget)
SEGMENT_KINDS: dict[str, tuple[str, bool, str, int]] = {
	"prepare": ("Wait at rest, then move the arm to the ready pose with the gripper open.", False, "approach", 600),
	"scan_view": (
		"Move the wrist camera to inspection view {view} of {views} around the brick and hold still.",
		False,
		"approach",
		900,
	),
	"return_ready": ("Return the arm to the ready pose with the gripper open.", False, "approach", 600),
	"grasp": ("Grasp the brick on the table.", False, "grasp", 1500),
	"lift": ("Lift the grasped brick straight up.", True, "lift", 600),
	"present_bottom": (
		"Show the bottom face of the held brick to the top camera and hold still.",
		True,
		"show_hold",
		900,
	),
	"return_to_lift": ("Move the held brick back to the lift pose.", True, "show_move", 600),
	"lower_on_table": (
		"Carry the held brick back to its spot on the table and lower it onto the table.",
		True,
		"place",
		1200,
	),
	"release_on_table": (
		"Open the gripper, let the brick settle on the table and lift the gripper clear.",
		False,
		"place",
		600,
	),
	"carry_to_bowl": ("Carry the held brick over the {bowl} bowl and lower it into the bowl.", True, "place", 1200),
	"release_in_bowl": (
		"Open the gripper to drop the brick into the bowl, lift clear and return to the rest pose.",
		False,
		"place",
		900,
	),
	# Recovery (TwinMujoco._retry): let go of a brick that cannot be put down, then return_ready.
	"let_go": ("Open the gripper where it is and rise.", False, "place", 300),
}

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


@dataclass(frozen=True)
class Segment:
	kind: str
	params: dict[str, Any] = field(default_factory=dict)

	def __post_init__(self):
		if self.kind not in SEGMENT_KINDS:
			raise ValueError(f"unknown segment kind {self.kind!r}")
		json.dumps(self.params, allow_nan=False)  # params must be plain JSON

	@property
	def instruction(self) -> str:
		return SEGMENT_KINDS[self.kind][0].format(**self.params)

	@property
	def held(self) -> bool:
		return SEGMENT_KINDS[self.kind][1]

	@property
	def phase(self) -> str:
		return SEGMENT_KINDS[self.kind][2]

	@property
	def max_frames(self) -> int:
		return SEGMENT_KINDS[self.kind][3]

	def key(self) -> str:
		"""Canonical identity used to match a student's segment with a recorded one."""
		return json.dumps({"kind": self.kind, "params": self.params}, sort_keys=True, allow_nan=False)


@dataclass
class StudentObservation:
	"""What a policy sees at one frame. ``images()`` gives the recorder's camera images of this frame (rendered on
	demand when the trace records no video); rendering is skipped unless a policy asks for it."""

	frame: int
	segment: Segment
	segment_id: int
	segment_frame: int  # actions applied so far in this segment
	q: np.ndarray  # joint reading (rad), the recorded ``state``
	_images: Callable[[], dict[str, np.ndarray]]

	def images(self) -> dict[str, np.ndarray]:
		return self._images()


@dataclass
class PolicyStep:
	target: np.ndarray | None = None  # 6 joint targets (rad); ignored when done
	done: bool = False  # segment_done: end the segment at this observation, without a step
	phase: str | None = None  # annotation for the health scorer; default: the segment's phase


class SegmentPolicy(Protocol):
	def start(self, segment: Segment, segment_id: int) -> None: ...

	def act(self, obs: StudentObservation) -> PolicyStep: ...


def _rounded(values, digits=6):
	return [round(float(v), digits) for v in values]


class _SegmentMixin:
	"""Segment bookkeeping and the generator -> segment mapping shared by the expert tracer and the student."""

	def _segment_setup(self):
		self.segments: list[dict] = []  # one row per segment: id, kind, instruction, params, start, end, reason
		self.lift_poses: list[np.ndarray] = []  # the held lift pose after each grasp (diagnostics)
		self._open_segment: dict | None = None
		self._scan_steps: list = []

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
			"start": self.ep.n_frames,
			"segment": seg,
		}
		self.segments.append(row)
		self._open_segment = row
		self.record.segment = row["id"]
		self.event("segment", id=row["id"], segment_kind=seg.kind, instruction=seg.instruction, params=seg.params)
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
		return next(name for name, b in self.bowls.items() if np.array_equal(np.asarray(b), xy))

	def _motion(self, generator) -> tuple[str, Any]:
		"""``("simple", Segment)`` or ``("composite", (name, {kind: Segment}))`` for a skill's motion generator."""
		name = generator.__name__
		local = generator.gi_frame.f_locals if generator.gi_frame is not None else {}
		if name == "ready":
			return "simple", Segment("prepare")
		if name == "_go_ready":
			return "simple", Segment("return_ready")
		if name == "let_go":
			return "simple", Segment("let_go")
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
			}
			return "simple", Segment("scan_view", params)
		if name == "look" and "candidate" in local:  # TwinMujoco._bottom
			cand = local["candidate"]
			return "simple", Segment(
				"present_bottom", {"target_q": _rounded(cand.q), "bottom_px": int(cand.faces_px[5])}
			)
		if name in _PARTS:
			segs = {}
			for kind, _ in _PARTS[name]:
				params = {}
				if kind == "carry_to_bowl":
					params = {"bowl": self._bowl_name()}
				elif kind == "lower_on_table":
					params = {"xy": _rounded(self._park_xy, 4)}
				segs[kind] = Segment(kind, params)
			return "composite", (name, segs)
		raise SegmentMappingError(f"no segment for motion generator {name!r} in node {self.node!r}")


class ExpertTracer(_SegmentMixin, TwinMujoco):
	"""The unchanged expert, recording which segment each of its frames belongs to (events and the per-frame
	``segment`` column). Behaviour is identical to :class:`TwinMujoco`: the targets pass through unchanged."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._segment_setup()

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
					self._segment_begin(segs[parts[nxt][0]])
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
		error = super()._execute(wrapped)
		if self._open_segment is not None:
			lost = self._open_segment["segment"].held and self.expert.lost
			self._segment_end(error or ("brick_lost" if lost else "done"))
		if form == "composite" and value[0] == "grasp_and_lift" and self.held is not None:
			self.lift_poses.append(np.asarray(self.held[0], dtype=np.float64).copy())
		return error


class StudentTwin(_SegmentMixin, TwinMujoco):
	"""Graph backend whose skills issue segments to ``policy`` instead of running the expert's motion generators.

	The expert object stays as the planners' kinematic model and parameter set: ``expert.last`` follows the applied
	targets, and its ``lost`` / ``_no_dual`` / ``aborted`` fields carry the lost-brick and clamp checks, so the
	inherited skill code reads them as it reads the expert's."""

	def __init__(self, layout, scorer, record, policy: SegmentPolicy, **kwargs):
		self.segment_policy = policy  # not ``policy``: TwinMujoco keeps the graph's Policy there
		super().__init__(layout, scorer, record, **kwargs)
		self._segment_setup()

	# -- the segment executor -----------------------------------------------------------------------------------

	def run_segment(self, seg: Segment) -> str | None:
		"""Run one segment to ``segment_done``. Returns a failure code (``ACTION_BUDGET_EXHAUSTED``,
		``SEGMENT_FRAME_BUDGET``, a fault) or None; a lost brick ends a held segment with ``expert.lost`` set."""
		ex, p = self.expert, self.expert.p
		sid = self._segment_begin(seg)
		self.segment_policy.start(seg, sid)
		frames = 0
		after_step = False  # the next observation follows a step of this segment (the checks read it)
		clamp_run = 0
		monitor = GripperMonitor()
		self._clamped = self._blocked = False
		while True:
			if self.total_frames >= MAX_TOTAL_FRAMES:
				self._segment_end("budget")
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
					self._blocked |= state is GripperState.BLOCKED
			if seg.held and ex.lost:
				self._segment_end("brick_lost")
				return None
			step = self.segment_policy.act(
				StudentObservation(
					frame=self.ep.n_frames,
					segment=seg,
					segment_id=sid,
					segment_frame=frames,
					q=np.asarray(obs["q"], dtype=np.float64).copy(),
					_images=self._images,
				)
			)
			if step.done:
				self._clamped = clamp_run >= p.clamp_still_frames
				self._segment_end("done")
				return None
			if frames >= seg.max_frames:
				self._segment_end("frame_budget")
				return "SEGMENT_FRAME_BUDGET"
			raw = np.asarray(step.target, dtype=np.float64)
			if raw.shape != (6,) or not np.isfinite(raw).all():
				raise ValueError("a policy target must be 6 finite joint positions")
			target = np.clip(raw, ex._lo, ex._hi)  # as ClosedLoopExpert.act
			ex.clipped_frames += bool(np.any(target != raw))
			ex.last = target.copy()
			ex._frame_idx += 1
			phase = step.phase or seg.phase
			if phase not in PHASES:
				raise ValueError(f"unknown phase {phase!r}")
			error = self._apply_target(obs, target, phase)
			frames += 1
			after_step = True
			if error:
				self._segment_end("fault")
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
		self._belief_obs = True
		try:
			error = self.run_segment(segs["grasp"])
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
		# The held lift pose (base of the show, park and place moves) is where the lift ended.
		self.held = (ex.last.copy(), None)
		self.lift_poses.append(ex.last.copy())
		return None

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
	the expert's segment ended. Each observed joint reading is compared with the recorded one (bitwise)."""

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

	def start(self, segment: Segment, segment_id: int) -> None:
		self.index += 1
		if self.index >= len(self.segments):
			raise ReplayMismatch(f"segment {segment_id} ({segment.kind}) is past the end of the recording")
		rec = self.segments[self.index]
		if rec.segment.key() != segment.key() or segment_id != self.index:
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
# without multisampling). Captured images are therefore compared within this bound, and the detector scores of a
# view are compared exactly only when its crop is bitwise equal; the decision (score >= threshold) always exactly.
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
	``crops`` (sha256 per keyframe file). With the trace directories, keyframes that are not bitwise equal are
	compared within the renderer's repeatability (RENDER_MAX_LEVEL / RENDER_MAX_FRACTION) and their views' scores
	are reported instead of checked. ``smoothness.clipped_frames`` is excluded from the result check: it counts the
	expert generator's raw targets that act() clipped, which a replay of the executed (already clipped) targets
	cannot see."""
	ee, se = comparable_events(expert["events"]), comparable_events(student["events"])
	noisy, image_ok, worst = set(), expert["crops"].keys() == student["crops"].keys(), (0, 0.0)
	for name in sorted(set(expert["crops"]) & set(student["crops"])):
		if expert["crops"][name] == student["crops"][name]:
			continue
		noisy.add(name.rsplit("-", 1)[0].replace("-", ":", 2))  # keyframe stem -> view id
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

	def masked(ev):  # scores of views whose crop differs by render noise are reported, not compared
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
		"scores": [(v["id"], v["score"]) for v in ev_ if v["id"] not in noisy]
		== [(v["id"], v["score"]) for v in sv_ if v["id"] not in noisy],
		"early_stops": _scan_stops(ee) == _scan_stops(se),
		"segments": _segments_of(ee) == _segments_of(se),
		"destination": expert["result"]["destination"] == student["result"]["destination"],
		"evaluate": result(expert["result"]) == result(student["result"]),
		"telemetry": expert["digest"] == student["digest"],
		"events": masked(ee) == masked(se),
	}
	info = {
		"render_noise_views": sorted(noisy),
		"render_noise_max_level": worst[0],
		"render_noise_max_fraction": worst[1],
		"render_noise_max_score_delta": max(score_delta, default=0.0),
	}
	return checks, info


def crop_digests(trace_dir: Path) -> dict[str, str]:
	return {
		p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(trace_dir, "keyframes").glob("*.png"))
	}


def run_episode(case, scorer, out: Path, *, policy: SegmentPolicy | None = None, sensing="realistic", video=False):
	"""Run one case through the graph: the expert (with segment markers) when ``policy`` is None, else the student.
	Writes the trace to ``out`` and returns ``result``, ``events``, ``digest``, ``crops`` and the twin's counters."""
	from .twin_graph import Policy, build_graph
	from .twin_mujoco import sample_case
	from .twin_record import EpisodeTrace

	trace = EpisodeTrace(out, video=video)
	layout = sample_case(*case)
	if policy is None:
		twin = ExpertTracer(layout, scorer, trace, sensing=sensing)
	else:
		twin = StudentTwin(layout, scorer, trace, policy, sensing=sensing)
	try:
		result = build_graph(twin, Policy()).invoke({}, config={"recursion_limit": 64})["result"]
		trace.finish(
			result, {"engine": "mujoco", "sensing": sensing, "case": list(case), "student": policy is not None}
		)
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
