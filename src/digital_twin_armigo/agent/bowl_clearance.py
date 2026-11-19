"""Bowl clearance on perceived geometry, for ``realistic`` sensing.

The physics expert's own clearance (:meth:`ClosedLoopExpert._clearance`) measures the robot against the simulator's
bowl and clutter collision geoms, i.e. at the true bowl positions. A real cell knows the bowls only from the top
camera, so in ``realistic`` sensing the clearance is measured against a proxy instead:

- each bowl is the **nominal** plate cross-section :data:`BOWL_PROFILE` (the outer envelope of the collision wall
  profile in ``mujoco_cell/scene.py``, its hollow filled up to the rim), revolved around the bowl's **perceived**
  centre (``TwinMujoco.bowls``, from :func:`.perception.locate_bowls`) and dilated by ``margin_m``. The proxy contains
  the true bowl whenever the localisation error is below ``margin_m``. The default margin (5 mm) is about twice the
  largest bowl localisation error measured on bench16 ``realistic`` (2.6 mm; median 1.5 mm). The flared rim is kept,
  so a jaw may pass under it as it may under the real plate.
- the robot is the same set of moving collision geoms the expert checks (jaws and links): mesh geoms as the convex
  hull MuJoCo collides them with (``mesh_graph``), every other geom as its local bounding box (exact for boxes, an
  over-approximation for the tiny jaw-tip spheres and capsules). Each surface is sampled with spacing ``spacing_m``;
  every surface point lies within ``spacing_m / sqrt(2)`` of a sample, and a signed distance is 1-Lipschitz, so
  subtracting that slack makes the sampled distance a lower bound on the surface-to-proxy distance.

Measured against the simulator's clearance over 267 random arm poses near the bowls of three bench16 layouts: with zero margin at the true centres the proxy agrees with ``mj_geomDistance`` within about 1 mm;
with the margin at the perceived centres it never reports clear where the true geometry collides, and it is
1-3.5 mm more conservative on average (margin and slack minus the localisation error).

The proxy reads only the robot model, the joint vector being checked and the perceived bowl centres. It never reads
the simulator's bowl or clutter geoms or their positions. ``realistic`` layouts have no clutter
(:func:`.twin_mujoco.sample_case`); clutter is not perceived, so a ``realistic`` twin refuses layouts with clutter.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from ..mujoco_cell._mj import mujoco

PERCEIVED_BOWL_MARGIN_M = 0.005  # dilation of the nominal bowl: covers the bowl localisation error
SURFACE_SPACING_M = 0.002  # grid spacing of the robot surface samples (slack spacing / sqrt(2), about 1.4 mm)
# Nominal plate cross-section (radius m, height m above the table), counter-clockwise from the axis: the underside of
# the wall, the rim, then filled flat at rim height back to the axis. Outer envelope of scene._BOWL_WALL_POLYS.
BOWL_PROFILE = np.array(
	[
		(0.0, 0.0),
		(0.073, 0.0),
		(0.0755, 0.0007),
		(0.081, 0.0028),
		(0.087, 0.0064),
		(0.0928, 0.011),
		(0.0975, 0.0155),
		(0.0998, 0.0182),
		(0.1, 0.0201),
		(0.0989, 0.0216),
		(0.0972, 0.022),
		(0.0, 0.022),
	]
)
# Mirrored about the axis, so the axis is not an edge: the signed distance is then right for every radius r >= 0.
_SECTION = np.vstack([BOWL_PROFILE[1:-1], BOWL_PROFILE[-2:0:-1] * (-1.0, 1.0)])


def box_surface_points(center: np.ndarray, half: np.ndarray, spacing: float) -> np.ndarray:
	"""Points on the surface of the box ``center +- half`` (its own frame), on a grid no coarser than ``spacing``."""
	axes = [np.linspace(-h, h, max(2, math.ceil(2.0 * h / spacing) + 1)) for h in half]
	pts = []
	for k in range(3):
		i, j = [a for a in range(3) if a != k]
		u, v = np.meshgrid(axes[i], axes[j], indexing="ij")
		for sign in (-1.0, 1.0):
			face = np.empty((u.size, 3))
			face[:, i], face[:, j], face[:, k] = u.ravel(), v.ravel(), sign * half[k]
			pts.append(face)
	return np.unique(np.vstack(pts), axis=0) + center


def hull_surface_points(model, gid: int, spacing: float) -> np.ndarray | None:
	"""Points on the convex hull MuJoCo collides mesh geom ``gid`` with (its own frame), every hull triangle sampled on
	a barycentric grid whose sub-triangle edges are at most ``spacing``; None without a hull graph."""
	mid = model.geom_dataid[gid]
	adr = model.mesh_graphadr[mid]
	if adr < 0:
		return None
	graph = model.mesh_graph
	nv, nf = int(graph[adr]), int(graph[adr + 1])
	start = adr + 2 + 3 * nv + 3 * nf  # skip vert_edgeadr, vert_globalid and edge_localid
	faces = np.asarray(graph[start : start + 3 * nf]).reshape(nf, 3)
	verts = model.mesh_vert[model.mesh_vertadr[mid] : model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
	pts = []
	for tri in verts[faces].astype(np.float64):
		longest = max(np.linalg.norm(tri[i] - tri[i - 1]) for i in range(3))
		n = max(1, math.ceil(longest / spacing))
		ij = np.array([(i, j) for i in range(n + 1) for j in range(n + 1 - i)], dtype=np.float64) / n
		pts.append(tri[0] + ij[:, :1] * (tri[1] - tri[0]) + ij[:, 1:] * (tri[2] - tri[0]))
	return np.unique(np.vstack(pts), axis=0)


def polygon_sdf(p: np.ndarray, poly: np.ndarray) -> np.ndarray:
	"""Signed distance of 2D points ``p`` (N x 2) to the simple polygon ``poly`` (M x 2); negative inside."""
	a, b = poly, np.roll(poly, -1, axis=0)
	ab = b - a
	ap = p[:, None, :] - a[None]
	t = np.clip(np.einsum("nmk,mk->nm", ap, ab) / np.einsum("mk,mk->m", ab, ab), 0.0, 1.0)
	dist = np.linalg.norm(ap - t[..., None] * ab, axis=2).min(axis=1)
	py = p[:, 1:2]
	crosses = (a[None, :, 1] > py) != (b[None, :, 1] > py)
	with np.errstate(divide="ignore", invalid="ignore"):
		x_at = a[None, :, 0] + (py - a[None, :, 1]) * ab[None, :, 0] / ab[None, :, 1]
	inside = np.count_nonzero(crosses & (p[:, 0:1] < x_at), axis=1) % 2 == 1
	return np.where(inside, -dist, dist)


def bowl_distance(points: np.ndarray, center_xy: np.ndarray) -> float:
	"""Smallest signed distance (m) from ``points`` (N x 3, world) to the nominal bowl at ``center_xy``. The bowl is
	a solid of revolution, so the distance is the 2D distance in the point's (radius, height) half-plane."""
	r = np.hypot(points[:, 0] - center_xy[0], points[:, 1] - center_xy[1])
	return float(polygon_sdf(np.stack([r, points[:, 2]], axis=1), _SECTION).min())


class PerceivedBowlClearance:
	"""Clearance (m) between the robot's moving collision geoms and the nominal bowls at the perceived centres.

	``bowls`` is a live mapping name -> perceived (x, y); it is read at every call, so a re-localisation is seen."""

	def __init__(
		self,
		cell,
		bowls: Mapping[str, Sequence[float]],
		*,
		margin_m: float = PERCEIVED_BOWL_MARGIN_M,
		spacing_m: float = SURFACE_SPACING_M,
	) -> None:
		m = cell.model
		# The geoms ClosedLoopExpert._clearance checks: both jaws and every link that moves with the shoulder lift.
		self.geoms = sorted(cell.fixed_jaw_geoms | cell.moving_jaw_geoms | {int(g) for g in cell.link_geoms})
		local = []
		for g in self.geoms:
			hull = hull_surface_points(m, g, spacing_m) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH else None
			local.append(
				hull if hull is not None else box_surface_points(m.geom_aabb[g, :3], m.geom_aabb[g, 3:], spacing_m)
			)
		self._counts = [len(p) for p in local]
		self._local = np.vstack(local)
		self.bowls = bowls
		self.margin_m = float(margin_m)
		self.slack = spacing_m / math.sqrt(2.0)

	def robot_points(self, kin, q: np.ndarray) -> np.ndarray:
		"""World positions of the robot surface samples at joint angles ``q`` (forward kinematics on ``kin``)."""
		kin.site_pose(q)
		d = kin.data
		out = np.empty_like(self._local)
		start = 0
		for g, n in zip(self.geoms, self._counts, strict=True):
			out[start : start + n] = d.geom_xpos[g] + self._local[start : start + n] @ d.geom_xmat[g].reshape(3, 3).T
			start += n
		return out

	def __call__(self, kin, qs: Sequence[np.ndarray], cap: float) -> float:
		"""Smallest clearance over the arm poses ``qs``, capped at ``cap`` (as ``ClosedLoopExpert._clearance``)."""
		best = cap
		centres = [np.asarray(xy, dtype=np.float64) for xy in self.bowls.values()]
		reach = self.margin_m + cap + self.slack  # samples farther than this from the nominal bowl cannot lower best
		r_max, z_max = BOWL_PROFILE[:, 0].max() + reach, BOWL_PROFILE[:, 1].max() + reach
		for q in qs:
			pts = self.robot_points(kin, np.asarray(q, dtype=np.float64))
			pts = pts[pts[:, 2] < z_max]
			for c in centres:
				near = pts[np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1]) < r_max]
				if len(near):
					best = min(best, bowl_distance(near, c) - self.margin_m - self.slack)
		return float(best)


def with_perceived_bowl_clearance(expert_type: type, clearance: PerceivedBowlClearance) -> type:
	"""Subclass of the expert class ``expert_type`` whose ``_clearance`` (grasp candidates, and every path check
	that calls it) uses ``clearance`` instead of the simulator's bowl / clutter geoms."""

	class PerceivedClearanceExpert(expert_type):
		def _clearance(self, qs):
			return clearance(self.kin, qs, self.p.clearance_cap_m)

	PerceivedClearanceExpert.__name__ = PerceivedClearanceExpert.__qualname__ = f"Perceived{expert_type.__name__}"
	return PerceivedClearanceExpert
