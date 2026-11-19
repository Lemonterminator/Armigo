"""Explicit gripper contact/stall state monitor using hardware-observable signals only."""

from __future__ import annotations

import enum
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from ..constants import FPS


class GripperState(enum.IntEnum):
	"""Discrete gripper status inferred from command, angle, servo effort, and jaw brick-normal forces."""

	FREE = 0
	MOVING = 1
	CLAMPED = 2
	BLOCKED = 3


@dataclass(frozen=True)
class GripperMonitorParams:
	"""Thresholds calibrated from simulation signal distributions (Task 0)."""

	err_rad: float = 0.05
	still_rad_s: float = 0.05
	effort_min: float = 0.70
	brick_force_n: float = 0.5
	clamp_force_n: float = 1.5
	blocked_frames: int = 4
	fps: float = float(FPS)
	use_effort: bool = True


class GripperMonitor:
	"""Pure-function stateful monitor tracking whether the gripper is FREE, MOVING, CLAMPED, or BLOCKED."""

	def __init__(self, params: GripperMonitorParams | None = None, *, use_effort: bool | None = None) -> None:
		p = params or GripperMonitorParams()
		if use_effort is not None:
			p = replace(p, use_effort=use_effort)
		self.p = p
		self.reset()

	def reset(self) -> None:
		"""Reset per-episode velocity and stall counters."""
		self._prev_meas: float | None = None
		self._run = 0

	def update(
		self,
		cmd: float,
		meas: float,
		effort_frac: float,
		jaw_normal: Sequence[float],
	) -> tuple[GripperState, int]:
		"""Return ``(state, direction)`` where ``direction = sign(cmd - meas)`` when BLOCKED, else 0."""
		p = self.p
		vel = 0.0 if self._prev_meas is None else (float(meas) - self._prev_meas) * p.fps
		self._prev_meas = float(meas)
		fn = [float(x) for x in jaw_normal]
		if fn and min(fn) >= p.clamp_force_n:
			self._run = 0
			return GripperState.CLAMPED, 0
		err = float(cmd) - float(meas)
		effort_ok = (not p.use_effort) or (float(effort_frac) >= p.effort_min)
		stalled = (
			abs(err) > p.err_rad
			and abs(vel) < p.still_rad_s
			and (max(fn) if fn else 0.0) < p.brick_force_n
			and effort_ok
		)
		self._run = self._run + 1 if stalled else 0
		if self._run >= p.blocked_frames:
			return GripperState.BLOCKED, int(np.sign(err))
		if abs(err) > p.err_rad or abs(vel) >= p.still_rad_s:
			return GripperState.MOVING, 0
		return GripperState.FREE, 0
