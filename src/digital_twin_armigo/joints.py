"""Joint unit conversion between Studio's normalised readings and model joint angles.

Joint values in the datasets are in Studio's normalised units (body joints -100…100, gripper 0…100), produced with
the follower calibration that recorded sort-depth-v2. The mapping assumes Studio's "zero at range midpoint"
convention and the Menagerie joint zero; the camera fit (``fit-camera``) is the check.
"""

from __future__ import annotations

import numpy as np

from .constants import CALIBRATION, GRIPPER_RAD, JOINTS, OFFSETS_DEG, TICKS_PER_DEG

_LO = np.array([CALIBRATION[j][0] for j in JOINTS[:5]], dtype=np.float64)
_HI = np.array([CALIBRATION[j][1] for j in JOINTS[:5]], dtype=np.float64)
_OFFSET_DEG = np.array([OFFSETS_DEG.get(j, 0.0) for j in JOINTS[:5]], dtype=np.float64)
_GRIPPER_SPAN = GRIPPER_RAD[1] - GRIPPER_RAD[0]


def normalized_to_rad(u: np.ndarray) -> np.ndarray:
	"""Studio normalised joint values (6,) → model joint angles in radians."""
	u = np.asarray(u, dtype=np.float64)
	out = np.empty(6)
	out[:5] = np.radians(u[:5] / 100.0 * (_HI - _LO) / 2.0 / TICKS_PER_DEG + _OFFSET_DEG)
	out[5] = GRIPPER_RAD[0] + np.clip(u[5], 0, 100) / 100.0 * _GRIPPER_SPAN
	return out


def rad_to_normalized(q: np.ndarray) -> np.ndarray:
	"""Inverse of :func:`normalized_to_rad` (float32, as stored in the datasets)."""
	q = np.asarray(q, dtype=np.float64)
	out = np.empty(6)
	out[:5] = (np.degrees(q[:5]) - _OFFSET_DEG) * TICKS_PER_DEG * 2.0 / (_HI - _LO) * 100.0
	out[5] = (q[5] - GRIPPER_RAD[0]) / _GRIPPER_SPAN * 100.0
	return out.astype(np.float32)
