"""Damped-least-squares inverse kinematics on the MuJoCo cell, and the wrist roll that lines the jaws up.

Every solver here works on the cell's own ``MjData``: it overwrites the arm joints, runs kinematics only, and
restores ``qpos`` (followed by one ``mj_forward``) before returning, so the simulation state is unchanged.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np

from ._mj import mujoco
from .cell import Cell

logger = logging.getLogger(__name__)

DAMPING = 1e-4
POSITION_TOL_M = 5e-4
_DOWN = np.array([0.0, 0.0, -1.0])
_DAMP6 = DAMPING * np.eye(6)
_DAMP3 = DAMPING * np.eye(3)


@contextmanager
def _scratch_qpos(cell: Cell) -> Iterator[None]:
	"""Let the body overwrite ``qpos`` for kinematics; put the simulation state back afterwards."""
	saved = cell.data.qpos.copy()
	try:
		yield
	finally:
		cell.data.qpos[:] = saved
		mujoco.mj_forward(cell.model, cell.data)


def ik(cell: Cell, target: np.ndarray, q0: np.ndarray, roll: float, iters: int = 200, w_rot: float = 0.3) -> np.ndarray:
	"""Joint angles putting ``gripperframe`` at ``target`` with the jaws pointing down (damped least squares).

	``w_rot`` weighs the jaws-down error against the position error; a smaller weight lets the jaws tilt to reach
	further.
	"""
	m, d = cell.model, cell.data
	q = q0.copy()
	jacp = np.zeros((3, m.nv))
	jacr = np.zeros((3, m.nv))
	lo, hi = cell.jnt_range[:5, 0], cell.jnt_range[:5, 1]
	qadr, cols = cell.qadr[:5], cell.dofadr[:4]
	with _scratch_qpos(cell):
		for _ in range(iters):
			d.qpos[qadr] = q[:5]
			mujoco.mj_kinematics(m, d)
			mujoco.mj_comPos(m, d)
			err_p = target - d.site_xpos[cell.site]
			approach = d.site_xmat[cell.site].reshape(3, 3)[:, 0]
			err_r = np.cross(approach, _DOWN)
			mujoco.mj_jacSite(m, d, jacp, jacr, cell.site)
			j = np.vstack([jacp[:, cols], w_rot * jacr[:, cols]])
			e = np.concatenate([err_p, w_rot * err_r])
			dq = j.T @ np.linalg.solve(j @ j.T + _DAMP6, e)
			q[:4] = np.clip(q[:4] + dq, lo[:4], hi[:4])
			if np.linalg.norm(err_p) < POSITION_TOL_M:
				break
		q[4] = np.clip(roll, lo[4], hi[4])
	return q


def site_error(cell: Cell, q: np.ndarray, target: np.ndarray) -> float:
	"""Distance from ``gripperframe`` at joint angles ``q`` to ``target`` (state restored)."""
	m, d = cell.model, cell.data
	with _scratch_qpos(cell):
		d.qpos[cell.qadr] = q
		mujoco.mj_kinematics(m, d)
		return float(np.linalg.norm(d.site_xpos[cell.site] - target))


def ik_reach(cell: Cell, target: np.ndarray, q0: np.ndarray, roll: float, tol: float = 0.005) -> np.ndarray:
	""":func:`ik` jaws down if that reaches ``target``, otherwise with the jaws allowed to tilt progressively more."""
	best, best_err = q0, np.inf
	for w in (0.3, 0.1, 0.03):
		q = ik(cell, target, q0, roll, w_rot=w)
		err = site_error(cell, q, target)
		if err < tol:
			return q
		if err < best_err:
			best, best_err = q, err
	logger.debug("IK missed %s by %.3f m", target, best_err)
	return best


def _jaw_direction(cell: Cell, q: np.ndarray) -> np.ndarray:
	"""Jaw direction at ``q`` on scratch state (the caller restores ``qpos``)."""
	m, d = cell.model, cell.data
	d.qpos[cell.qadr] = q
	mujoco.mj_kinematics(m, d)
	v = (d.xipos[cell.jaw_body] - d.site_xpos[cell.site])[:2]
	return v / max(np.linalg.norm(v), 1e-9)


def jaw_direction(cell: Cell, q: np.ndarray) -> np.ndarray:
	"""Horizontal unit vector from the fixed jaw (``gripperframe``) towards the moving jaw at joint angles ``q``."""
	with _scratch_qpos(cell):
		return _jaw_direction(cell, q)


def best_roll(cell: Cell, q: np.ndarray, axis_xy: np.ndarray) -> tuple[float, float]:
	"""Wrist roll that lines the jaw opening up with ``axis_xy`` (either sign), and that sign."""
	lo, hi = cell.jnt_range[4]
	best = (-np.inf, 0.0, 1.0)
	qq = q.copy()
	qq[5] = 0.75
	with _scratch_qpos(cell):
		for roll in np.linspace(lo, hi, 181):
			qq[4] = roll
			c = float(_jaw_direction(cell, qq) @ axis_xy)
			score = abs(c) - 0.02 * abs(roll)
			if score > best[0]:
				best = (score, float(roll), math.copysign(1.0, c))
	return best[1], best[2]


def ik_held(cell: Cell, target: np.ndarray, q0: np.ndarray, iters: int = 120) -> tuple[np.ndarray, float]:
	"""Move shoulder pan, shoulder lift and elbow so the held brick's centre reaches ``target``; wrist flex, roll and
	gripper stay as in ``q0``. Returns the joints and the remaining position error in metres.

	Unlike the solvers above this leaves the arm at the solution: the presentation search evaluates the pose next
	and restores the state itself.
	"""
	m, d = cell.model, cell.data
	rel_p = cell.held[0]
	q = q0.copy()
	jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
	lo, hi = cell.jnt_range[:3, 0], cell.jnt_range[:3, 1]
	cols = cell.dofadr[:3]
	err = np.full(3, np.inf)
	for _ in range(iters):
		d.qpos[cell.qadr] = q
		mujoco.mj_kinematics(m, d)
		mujoco.mj_comPos(m, d)
		point = d.xpos[cell.grip_body] + d.xmat[cell.grip_body].reshape(3, 3) @ rel_p
		err = target - point
		if np.linalg.norm(err) < POSITION_TOL_M:
			break
		mujoco.mj_jac(m, d, jacp, jacr, point, cell.grip_body)
		j = jacp[:, cols]
		q[:3] = np.clip(q[:3] + j.T @ np.linalg.solve(j @ j.T + _DAMP3, err), lo, hi)
	return q, float(np.linalg.norm(err))
