"""Joint unit conversion between Studio's normalised readings and model joint angles.

Joint values in the datasets are in Studio's normalised units (body joints -100…100, gripper 0…100), produced with
the follower calibration that recorded sort-depth-v2. The mapping assumes Studio's "zero at range midpoint"
convention and the Menagerie joint zero; the camera fit (``fit-camera``) is the check. The agent's student datasets
(``agent.twin_record``) use radians instead, with :func:`joint_normalization` recorded alongside.
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


# Student datasets store joint angles in radians (canonical). Studio units do not bound the expert: wrist_roll
# commands reach about +-190 there. The normalisation below maps each joint's MJCF range (``assets/so101_mjcf``, the
# range the expert clips every command to, so it covers the expert's span) affinely onto [-1, 1]. Measured joints can
# sit slightly outside the soft limits, so values just beyond +-1 are valid; the map is invertible everywhere.
JOINT_RANGE_RAD = np.array(
	[
		[-1.91986, 1.91986],
		[-1.7453293, 1.7453293],
		[-1.69, 1.69],
		[-1.658063, 1.658063],
		[-2.7438473, 2.8412063],
		[-0.174533, 1.7453292],
	],
	dtype=np.float64,
)
_MID = JOINT_RANGE_RAD.mean(axis=1)
_HALF = (JOINT_RANGE_RAD[:, 1] - JOINT_RANGE_RAD[:, 0]) / 2.0


def rad_to_unit(q: np.ndarray) -> np.ndarray:
	"""Joint angles (..., 6) in radians -> the joint range mapped onto [-1, 1] (float64)."""
	return (np.asarray(q, dtype=np.float64) - _MID) / _HALF


def unit_to_rad(u: np.ndarray) -> np.ndarray:
	"""Inverse of :func:`rad_to_unit` (float64)."""
	return np.asarray(u, dtype=np.float64) * _HALF + _MID


def joint_normalization(names) -> dict:
	"""Dataset metadata: units of ``observation.state`` / ``action`` and the invertible per-joint normalisation."""
	return {
		"unit": "rad",
		"names": list(names),
		"low_rad": JOINT_RANGE_RAD[:, 0].tolist(),
		"high_rad": JOINT_RANGE_RAD[:, 1].tolist(),
		"normalized": "u = 2 * (q - low) / (high - low) - 1",
		"inverse": "q = low + (u + 1) * (high - low) / 2",
		"source": "SO-101 MJCF joint ranges; the expert clips every command to them",
	}
