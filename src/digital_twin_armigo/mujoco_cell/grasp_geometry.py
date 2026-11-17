"""Jaw and grasp geometry on a private kinematic copy of the cell: pad faces, the closing sweep's low point.

Everything here runs on :class:`Kinematics`, a private ``MjData`` on the cell's model, so the simulation state is
never touched.
"""

from __future__ import annotations

import numpy as np

from ._mj import mujoco
from .cell import Cell


class Kinematics:
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


def geom_points(model, gid: int) -> np.ndarray:
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


def jaw_pad_faces(cell: Cell, kin: Kinematics, q: np.ndarray, band: tuple[float, float]) -> tuple[float, float]:
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
			w = d.geom_xpos[gid] + geom_points(m, gid) @ d.geom_xmat[gid].reshape(3, 3).T
			pts.append((w - pos) @ rot)
		p = np.vstack(pts)
		p = p[(p[:, 0] >= band[0]) & (p[:, 0] <= band[1])]
		faces.append(float(pick(p[:, 2])) if len(p) else (np.inf if pick is np.min else -np.inf))
	return faces[0], faces[1]


def jaw_sweep_low_point(cell: Cell, kin: Kinematics, q: np.ndarray, g_from: float, g_to: float, samples: int) -> float:
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
			w = d.geom_xpos[gid] + geom_points(m, gid) @ d.geom_xmat[gid].reshape(3, 3).T
			low = max(low, float(((w - pos) @ rot)[:, 0].max()))
	return low


def quat_xyzw_to_mat(q: np.ndarray) -> np.ndarray:
	"""Rotation matrix of a unit quaternion in (x, y, z, w) order."""
	x, y, z, w = q
	return np.array(
		[
			[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
			[2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
			[2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
		]
	)
