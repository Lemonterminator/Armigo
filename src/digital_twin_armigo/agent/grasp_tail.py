"""First-grasp tail experiment (docs/vla/GRASP_TAIL_AB_HANDOFF_2026-11-23.md): instrumentation and the two arms.

Only the episode's **first** ``grasp`` segment of a :class:`~.student_rollout.StudentTwin` is touched, and only when
the twin is built with a ``grasp_tail`` mode:

- ``act``: the ACT policy drives every frame; the switch point, target and errors are only logged (arm A / A').
- ``act-wait``: as ``act``, but when the switch point fired (near-field gate included) and ACT ends the segment
  without ``clamp_still_frames`` consecutive clamped frames, hold ACT's last executed command for at most
  ``clamp_timeout_frames`` frames until the clamp is confirmed (arm A+wait). No other motion is added.
- ``expert-once``: at the switch observation, before any action of that frame is issued, a truncated expert tail
  takes over the rest of the segment: line up over the camera-planned target at the current height, descend, servo,
  close and confirm the clamp within the timeout (arm B). No reset, re-perception, blocked recovery or expert lift.

The switch point is decided at observations, from proprioception and the plan snapshot only: the first downward
crossing of ``target_z + pregrasp_dz_m + CROSS_MARGIN_M`` by the gripper frame, after it was seen above the line,
before the gripper started closing (largest executed gripper command of the segment minus the last executed one above
``CLOSE_START_RAD``), and within the near-field gate (XY <= ``NEAR_XY_M``, ``|q[4] - roll| <= NEAR_ROLL_RAD``).

Every quantity is in the world frame of the gripper-frame target (``ClosedLoopExpert._grasp_plan``), metres and
radians. A telemetry row is one observation t paired with the command issued after it (none at the segment end).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

GRASP_TAIL_MODES = ("act", "act-wait", "expert-once")
GRASP_TAIL_VERSION = "grasp-tail-v1"
CROSS_MARGIN_M = 0.005
CLOSE_START_RAD = 0.08
NEAR_XY_M = 0.015  # = TakeoverConfig.in_place_near_m (vla/dagger.py)
NEAR_ROLL_RAD = 0.15  # = the roll part of StudentDaggerTwin._in_place_near

# Why a first grasp was not switched (kept in the denominator, reported per reason).
NOT_TRIGGERED = (
	"no_target",  # the plan snapshot is not finite
	"closed_early",  # the gripper started closing before the crossing
	"near_field_xy",  # crossed, horizontal error above NEAR_XY_M
	"near_field_roll",  # crossed, wrist-roll error above NEAR_ROLL_RAD
	"near_field_xy_roll",  # both
	"never_above",  # the gripper frame was never seen above the line
	"never_crossed",  # seen above, the segment ended before the crossing
)


@dataclass(frozen=True)
class GraspPlan:
	"""Snapshot of the planned grasp at the segment entry (the same ``_planned_grasp()`` call that gives the roll)."""

	target: np.ndarray  # gripper-frame target (m, world)
	roll: float  # wrist roll (rad)
	open_g: float  # jaw opening command (rad)
	frame: int  # episode frame of the snapshot
	perception_frame: int | None  # episode frame of the brick perception it was planned on

	@property
	def valid(self) -> bool:
		return bool(np.isfinite(self.target).all() and np.isfinite(self.roll) and np.isfinite(self.open_g))

	def to_json(self) -> dict[str, Any]:
		return {
			"target_xyz_m": [float(v) for v in self.target],
			"roll_rad": float(self.roll),
			"open_g_rad": float(self.open_g),
			"frame": int(self.frame),
			"perception_frame": self.perception_frame,
			"valid": self.valid,
		}


_COLUMNS = (
	"segment_frame",
	"frame",
	"source",
	"command",
	"q",
	"tcp",
	"err_xyz",
	"err_roll",
	"jaw_normal_est",
	"gripper_cmd",
	"effort_frac",
	"gripper_state",
	"clamp_run",
)


class GraspTail:
	"""Switch detector and telemetry of one first grasp segment. ``observe`` is called once per observation, before
	the command of that frame is chosen; ``command`` once per issued command."""

	def __init__(self, mode: str, plan: GraspPlan, pregrasp_dz_m: float, segment_id: int):
		if mode not in GRASP_TAIL_MODES:
			raise ValueError(f"grasp_tail mode must be one of {GRASP_TAIL_MODES}, got {mode!r}")
		self.mode = mode
		self.plan = plan
		self.segment_id = segment_id
		self.line_z = float(plan.target[2]) + float(pregrasp_dz_m) + CROSS_MARGIN_M
		self.above_seen = False
		self.closing = False
		self.max_open_cmd: float | None = None
		self.decision: str | None = None  # "triggered" or a NOT_TRIGGERED reason
		self.switch: dict[str, Any] | None = None
		self.outcome: str | None = None  # how the takeover / wait ended
		self.wait_frames = 0
		self.events: list[dict[str, Any]] = []
		self.rows: dict[str, list] = {c: [] for c in _COLUMNS}
		self._clamp_seen = False
		if not plan.valid:
			self._decide("no_target", None)

	@property
	def triggered(self) -> bool:
		return self.decision == "triggered"

	def event(self, kind: str, frame: int, segment_frame: int, **payload) -> None:
		self.events.append({"kind": kind, "frame": int(frame), "segment_frame": int(segment_frame), **payload})

	def _decide(self, decision: str, info: dict | None) -> None:
		self.decision = decision
		if info is not None:
			self.switch = info

	def errors(self, q: np.ndarray, tcp: np.ndarray) -> tuple[np.ndarray, float]:
		return np.asarray(self.plan.target, dtype=np.float64) - tcp, float(q[4]) - float(self.plan.roll)

	def observe(
		self,
		*,
		frame: int,
		segment_frame: int,
		obs: dict,
		q: np.ndarray,
		executed: np.ndarray | None,
		clamp_run: int,
		gripper_state: Any,
		clamp_frames: int,
	) -> bool:
		"""Log observation t and update the switch decision; True when the switch fires at this observation.
		``executed`` is the last command executed in this segment (None at its first observation)."""
		tcp = np.asarray(obs["tcp"], dtype=np.float64)
		err, err_roll = self.errors(q, tcp)
		r = self.rows
		r["segment_frame"].append(int(segment_frame))
		r["frame"].append(int(frame))
		r["source"].append("")
		r["command"].append([float("nan")] * 6)
		r["q"].append([float(v) for v in q])
		r["tcp"].append([float(v) for v in tcp])
		r["err_xyz"].append([float(v) for v in err])
		r["err_roll"].append(err_roll)
		r["jaw_normal_est"].append([float(v) for v in np.asarray(obs["jaw_normal"], dtype=np.float64)])
		r["gripper_cmd"].append(float(obs["gripper_cmd"]))
		r["effort_frac"].append(float(obs["effort_frac"]))
		r["gripper_state"].append(None if gripper_state is None else str(getattr(gripper_state, "name", gripper_state)))
		r["clamp_run"].append(int(clamp_run))
		if clamp_run >= clamp_frames and not self._clamp_seen:
			self._clamp_seen = True
			self.event("clamp_confirmed", frame, segment_frame, clamp_run=int(clamp_run))
		if executed is not None and not self.closing:
			g = float(executed[5])
			self.max_open_cmd = g if self.max_open_cmd is None else max(self.max_open_cmd, g)
			if self.max_open_cmd - g > CLOSE_START_RAD:
				self.closing = True
				self.event("closing_started", frame, segment_frame, max_open_cmd=self.max_open_cmd, command=g)
		if self.decision is not None:
			return False
		if self.closing:
			self._decide("closed_early", {"frame": int(frame), "segment_frame": int(segment_frame)})
			self.event("not_triggered", frame, segment_frame, reason="closed_early")
			return False
		z = float(tcp[2])
		if z > self.line_z:
			self.above_seen = True
			return False
		if not self.above_seen:
			return False
		xy = float(np.linalg.norm(err[:2]))
		xy_ok, roll_ok = xy <= NEAR_XY_M, abs(err_roll) <= NEAR_ROLL_RAD
		info = {
			"frame": int(frame),
			"segment_frame": int(segment_frame),
			"tcp_xyz_m": [float(v) for v in tcp],
			"err_xyz_m": [float(v) for v in err],
			"err_xy_m": xy,
			"err_roll_rad": err_roll,
			"line_z_m": self.line_z,
			"near_xy_ok": xy_ok,
			"near_roll_ok": roll_ok,
		}
		self.event(
			"crossed", frame, segment_frame, **{k: v for k, v in info.items() if k not in ("frame", "segment_frame")}
		)
		if xy_ok and roll_ok:
			self._decide("triggered", info)
			self.event("triggered", frame, segment_frame)
			return True
		reason = (
			"near_field_xy_roll" if not (xy_ok or roll_ok) else ("near_field_xy" if not xy_ok else "near_field_roll")
		)
		self._decide(reason, info)
		self.event("not_triggered", frame, segment_frame, reason=reason)
		return False

	def command(self, source: str, target: np.ndarray) -> None:
		"""The command issued after the last observed row and where it came from."""
		self.rows["source"][-1] = source
		self.rows["command"][-1] = [float(v) for v in target]

	def finish(self, frame: int, segment_frame: int, reason: str) -> None:
		"""The segment ended (``reason``: the segment-end reason). Fixes a missing decision."""
		if self.decision is None:
			self._decide("never_crossed" if self.above_seen else "never_above", None)
			self.event("not_triggered", frame, segment_frame, reason=self.decision)
		self.event("segment_end", frame, segment_frame, reason=reason, outcome=self.outcome)

	def to_json(self) -> dict[str, Any]:
		sources = [s for s in self.rows["source"] if s]
		return {
			"version": GRASP_TAIL_VERSION,
			"mode": self.mode,
			"segment_id": self.segment_id,
			"frame_convention": "row = observation t and the command issued after it ('' = none)",
			"units": "m, rad; world frame of the gripper-frame target",
			"criteria": {
				"cross_margin_m": CROSS_MARGIN_M,
				"line_z_m": self.line_z,
				"close_start_rad": CLOSE_START_RAD,
				"near_xy_m": NEAR_XY_M,
				"near_roll_rad": NEAR_ROLL_RAD,
			},
			"plan": self.plan.to_json(),
			"decision": self.decision,
			"switch": self.switch,
			"outcome": self.outcome,
			"wait_frames": self.wait_frames,
			"source_counts": {s: sources.count(s) for s in sorted(set(sources))},
			"events": self.events,
			"rows": self.rows,
		}


def expert_tail(ex, plan: GraspPlan):
	"""Truncated expert tail from where ACT left the gripper (``ex.last``, ``ex.obs``): line up over the planned
	target at the current height, descend, servo, close and wait for a verified clamp. Returns the outcome of
	``ClosedLoopExpert._close_and_clamp`` (``clamped`` / ``blocked`` / ``timeout``). The alignment timing is the
	in-place recovery's (``StudentDaggerTwin._in_place_grasp_gen``)."""
	p = ex.p
	target, roll, g_open = np.asarray(plan.target, dtype=np.float64), float(plan.roll), float(plan.open_g)
	tcp = np.asarray(ex.obs["tcp"], dtype=np.float64)
	dz = float(tcp[2] - target[2])
	if dz > 0.003:
		q = ex._ik(target + np.array([0.0, 0.0, dz]), ex.last, roll)
		q[5] = g_open
		shift = float(np.linalg.norm(tcp[:2] - target[:2]))
		yield from ex._move(q, max(0.5, p.final_descend_s * shift / 0.005), "grasp")
	q = ex._ik(target, ex.last, roll)
	q[5] = g_open
	yield from ex._move(q, p.final_descend_s, "grasp")
	yield from ex._servo(target, roll, "grasp", p.servo_tol_m)
	return (yield from ex._close_and_clamp())
