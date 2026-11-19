"""What a real arm's own sensors give in place of simulator contact signals.

The SO-101 gripper servo reports its position and load (effort), not per-jaw forces. :class:`GripperContact` turns
those into the two jaw-force values the expert's clamp and lost-brick checks read: the jaws touch something when the
gripper has stopped short of its command (position above command by more than ``GAP_RAD``) under high effort. In
simulation this agrees with "both jaws above the expert's lost-brick force" on 98-99 % of frames over six bench16
episodes; the disagreements are the frames of a closing stroke before the gripper settles.
"""

from __future__ import annotations

import numpy as np

from ..constants import FPS

GAP_RAD = 0.05  # gripper position above its command: the jaws are held open by something
EFFORT_MIN = 0.5  # fraction of the gripper torque limit
STILL_RAD_S = 0.3  # gripper speed under which the reading is a settled contact, not a closing stroke
FORCE_AT_FULL_EFFORT_N = 3.0  # reported jaw force at the torque limit (above the expert's 1.5 N clamp threshold)


class GripperContact:
	def __init__(self) -> None:
		self.prev: float | None = None

	def reset(self) -> None:
		self.prev = None

	def update(self, cmd: float, meas: float, effort_frac: float) -> np.ndarray:
		"""Estimated (left, right) jaw normal force in newtons from the gripper command, position and effort."""
		speed = 0.0 if self.prev is None else abs(meas - self.prev) * FPS
		self.prev = float(meas)
		touching = meas - cmd > GAP_RAD and effort_frac >= EFFORT_MIN and speed < STILL_RAD_S
		force = FORCE_AT_FULL_EFFORT_N * min(effort_frac, 1.0) if touching else 0.0
		return np.array([force, force])
