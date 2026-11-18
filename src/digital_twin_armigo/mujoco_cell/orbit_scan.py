"""Orbit scan: the wrist camera circles the brick around the vertical axis through its upward face, looking inwards.

The axis is the vertical line through the x-y centre of the brick face that points most upwards (the top face of an
upright brick, a side face of a brick lying down). The SO-101 wrist camera sits about 57 mm off the jaw axis and
looks inwards, about 57 degrees below the horizon, so with the jaws held straight down the wrist roll swings the
camera around the jaw axis, on the outside, looking back towards the arm. The orbit puts the jaw axis over the brick
and drives the camera along a horizontal ellipse around the brick axis: a circle for a square face, an ellipse
stretched along the long side of a rectangular one. Where the ellipse is not the camera's natural 57 mm circle the
jaw axis moves off centre (eccentric orbit), so the camera keeps about the same stand-off from every side face.

The arm has five joints; with the jaws straight down the tilt of the approach axis is not free (it stays in the arm's
vertical plane), so the camera's viewing angle is fixed by its mount. The knobs are the jaw-tip height above the
upward face (``tip_heights_m``: lower shows more of the side faces) and the stand-off of the ring beyond the face
edges (``standoffs_m``). Every kept pose has the jaws within ``down_tol_deg`` of straight down, the robot at least
``min_robot_z_m`` above the table, ``brick_clearance_m`` from the brick, and the gripper and wrist camera
``self_clearance_m`` from the base, shoulder and upper arm (on bricks close to the base the far side of the ring swings
the camera back into the shoulder). The same checks run on poses between neighbouring ring poses; the ring is split
into arcs where they fail.

:func:`plan_orbit_scan` tries each ring and each arc, keeps the arc whose views cover the requested faces best (fewest
views on a tie), and returns the chosen views in wrist-roll order with the arc poses between them as via points, so
the arm sweeps along the ring in one direction instead of cutting across it.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from ..presentation import FACES
from ._mj import mujoco
from .cell import Cell
from .ik import ik
from .inspection_planning import ViewCandidate, greedy_face_cover
from .inspection_views import capture_view

__all__ = [
	"OrbitParams",
	"OrbitPlan",
	"TopFace",
	"jaw_axis_xy",
	"plan_orbit_scan",
	"ring_point",
	"top_face",
]


@dataclass(frozen=True)
class TopFace:
	"""The brick face pointing most upwards, in world coordinates (metres)."""

	face: int  # index into FACES
	centre: np.ndarray  # face centre (x, y, z)
	u: np.ndarray  # horizontal unit vector along the face's longer horizontal extent
	v: np.ndarray  # u turned +90 degrees about world z
	half_u: float  # horizontal half extent of the face along u
	half_v: float  # along v


def top_face(pos: np.ndarray, rot: np.ndarray, size: Sequence[float]) -> TopFace:
	"""Upward face of a box at ``pos`` with rotation matrix ``rot`` (columns: brick axes in world) and full extents
	``size`` (brick x, y, z)."""
	pos, rot = np.asarray(pos, dtype=np.float64), np.asarray(rot, dtype=np.float64).reshape(3, 3)
	face = int(np.argmax([sign * rot[2, axis] for axis, sign in FACES]))
	axis, sign = FACES[face]
	centre = pos + sign * rot[:, axis] * size[axis] / 2.0
	flat = [rot[:2, k] * size[k] / 2.0 for k in range(3) if k != axis]
	main = max(flat, key=lambda f: float(np.linalg.norm(f)))
	u2 = main / max(float(np.linalg.norm(main)), 1e-12)
	v2 = np.array([-u2[1], u2[0]])
	half_u = sum(abs(float(f @ u2)) for f in flat)
	half_v = sum(abs(float(f @ v2)) for f in flat)
	return TopFace(face, centre, np.array([*u2, 0.0]), np.array([*v2, 0.0]), half_u, half_v)


def ring_point(top: TopFace, phi: float, standoff: float) -> np.ndarray:
	"""Horizontal camera position (x, y) on the ring at azimuth ``phi`` (radians, from ``u`` towards ``v``):
	an ellipse ``standoff`` beyond the face edge on each of its axes."""
	a, b = top.half_u + standoff, top.half_v + standoff
	return top.centre[:2] + a * math.cos(phi) * top.u[:2] + b * math.sin(phi) * top.v[:2]


def jaw_axis_xy(top: TopFace, phi: float, standoff: float, mount_m: float) -> np.ndarray:
	"""Where the (vertical) jaw axis goes for the camera, ``mount_m`` off that axis and on the outside, to sit at
	:func:`ring_point`: the jaw axis is off the brick axis by the ring radius minus ``mount_m`` (eccentric orbit)."""
	cam = ring_point(top, phi, standoff)
	radial = cam - top.centre[:2]
	return cam - mount_m * radial / max(float(np.linalg.norm(radial)), 1e-12)


@dataclass(frozen=True)
class OrbitParams:
	tip_heights_m: tuple[float, ...] = (0.012, 0.025, 0.04)  # jaw tips above the upward face centre
	standoffs_m: tuple[float, ...] = (0.025, 0.04)  # camera ring beyond the face edges
	azimuths: int = 24
	down_tol_deg: float = 8.0
	site_tol_m: float = 0.004
	cam_tol_m: float = 0.006  # camera off its ring point
	min_robot_z_m: float = 0.012  # lowest robot point above the table
	brick_clearance_m: float = 0.006  # robot to brick
	self_clearance_m: float = 0.01  # gripper and wrist camera to the base, shoulder and upper arm
	segment_checks: int = 3  # interior poses checked between neighbouring ring poses
	restarts: int = 6


@dataclass
class OrbitPlan:
	views: list[ViewCandidate]  # in wrist-roll order
	via: list[list[np.ndarray]]  # kept ring poses between the previous view (none for the first) and each view
	tip_height_m: float | None = None
	standoff_m: float | None = None
	ring: list[ViewCandidate] = field(default_factory=list)  # the arc of safe ring poses the sweep runs along

	@property
	def mode(self) -> str:
		return "orbit" if self.views else "none"


def _brick_geoms(cell: Cell) -> list[int]:
	return [int(g) for g in np.flatnonzero(cell.model.geom_bodyid == cell.brick_body)]


def _robot_geoms(cell: Cell) -> list[int]:
	return sorted(cell.fixed_jaw_geoms | cell.moving_jaw_geoms | {int(g) for g in cell.link_geoms})


def _body_geoms(cell: Cell, names: Iterable[str]) -> list[int]:
	"""Collision geoms of the named bodies."""
	m = cell.model
	ids = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in names}
	return [int(g) for g in range(m.ngeom) if m.geom_bodyid[g] in ids and (m.geom_contype[g] or m.geom_conaffinity[g])]


def _min_distance(m, d, a: Iterable[int], b: Iterable[int], cap: float) -> float:
	best = cap
	for g1 in a:
		for g2 in b:
			best = min(best, float(mujoco.mj_geomDistance(m, d, g1, g2, best, None)))
	return best


def _wrap(a: float) -> float:
	return (a + math.pi) % (2 * math.pi) - math.pi


class _Poser:
	"""Vertical-jaw IK that also turns the wrist roll until the camera lands on a given horizontal point."""

	def __init__(self, cell: Cell, kin, params: OrbitParams) -> None:
		self.cell, self.kin, self.p = cell, kin, params
		self.cam = mujoco.mj_name2id(cell.model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
		self.lo, self.hi = cell.jnt_range[4]
		self.cos_tol = math.cos(math.radians(params.down_tol_deg))

	def fk(self, q):
		m, d = self.kin.model, self.kin.data
		d.qpos[self.kin.qadr] = q
		mujoco.mj_kinematics(m, d)
		mujoco.mj_camlight(m, d)  # mj_kinematics does not move cameras
		site = d.site_xpos[self.kin.site].copy()
		down = float(-d.site_xmat[self.kin.site].reshape(3, 3)[2, 0])
		return site, down, d.cam_xpos[self.cam].copy()

	def mount(self, q) -> float:
		site, _, cam = self.fk(q)
		return float(np.linalg.norm(cam[:2] - site[:2]))

	def solve(self, cam_xy: np.ndarray, centre_xy: np.ndarray, site_z: float, q0: np.ndarray, mount: float):
		"""Joints with the jaws down, the camera over ``cam_xy`` on the side away from ``centre_xy``, or None."""
		radial = cam_xy - centre_xy
		radial = radial / max(float(np.linalg.norm(radial)), 1e-12)
		site_xy = cam_xy - mount * radial
		q = np.asarray(q0, dtype=np.float64).copy()
		for _ in range(4):
			q = ik(self.kin, np.array([*site_xy, site_z]), q, float(q[4]), iters=300)
			site, down, cam = self.fk(q)
			# Turn the roll so the camera's bearing from the jaw axis matches the radial direction.
			off = cam[:2] - site[:2]
			turn = _wrap(math.atan2(radial[1], radial[0]) - math.atan2(off[1], off[0]))
			roll = q[4] + turn
			if not self.lo <= roll <= self.hi:
				alt = roll - math.copysign(2 * math.pi, roll)
				if not self.lo <= alt <= self.hi:
					return None
				roll = alt
			q[4] = roll
			site, down, cam = self.fk(q)
			site_xy = site_xy + (cam_xy - cam[:2])
		q = ik(self.kin, np.array([*site_xy, site_z]), q, float(q[4]), iters=300)
		site, down, cam = self.fk(q)
		if (
			np.linalg.norm(site - np.array([*site_xy, site_z])) > self.p.site_tol_m
			or down < self.cos_tol
			or np.linalg.norm(cam[:2] - cam_xy) > self.p.cam_tol_m
		):
			return None
		return q


def _coverage(views: Sequence[ViewCandidate], faces: Sequence[int], min_px: int) -> tuple[int, int, int]:
	"""(faces covered, smallest best-pixel count over the faces, -views): larger is better."""
	best = [max((v.faces_px[f] for v in views), default=0) for f in faces]
	return sum(b >= min_px for b in best), min(best, default=0), -len(views)


def plan_orbit_scan(
	cell: Cell,
	kin,
	q_seed: np.ndarray,
	*,
	gripper: float,
	params: OrbitParams = OrbitParams(),
	faces: Sequence[int] = (0, 1, 2, 3, 4),
	min_px: int = 800,
	max_views: int = 5,
) -> OrbitPlan:
	"""Orbit scan of the brick at its current simulated pose (see the module docstring). ``kin`` is a private
	:class:`~.grasp_geometry.Kinematics` (IK runs there). Returns an empty plan when no ring pose is kept."""
	m = cell.model
	pos = cell.data.xpos[cell.brick_body].copy()
	rot = cell.data.xmat[cell.brick_body].reshape(3, 3).copy()
	top = top_face(pos, rot, cell.layout.brick_size)
	brick, robot = _brick_geoms(cell), _robot_geoms(cell)
	distal = _body_geoms(cell, ("gripper", "camera_mount", "moving_jaw_so101_v1"))
	proximal = _body_geoms(cell, ("base", "shoulder", "upper_arm"))
	poser = _Poser(cell, kin, params)
	d = copy.copy(cell.data)
	seeds = [np.asarray(q_seed, dtype=np.float64).copy()]
	seeds += list(np.random.default_rng(0).uniform(cell.jnt_range[:, 0], cell.jnt_range[:, 1], (params.restarts, 6)))
	phis = np.linspace(0.0, 2 * math.pi, params.azimuths, endpoint=False)

	def safe(q) -> bool:
		d.qpos[cell.qadr] = q
		mujoco.mj_kinematics(m, d)
		return (
			cell.robot_min_z(d) >= params.min_robot_z_m
			and _min_distance(m, d, robot, brick, params.brick_clearance_m) >= params.brick_clearance_m
			and _min_distance(m, d, distal, proximal, params.self_clearance_m) >= params.self_clearance_m
		)

	def segment_safe(a, b) -> bool:
		n = params.segment_checks
		return all(safe(a.q + (k / (n + 1)) * (b.q - a.q)) for k in range(1, n + 1))

	best_plan, best_score = OrbitPlan([], []), (-1, -1, 0)
	for tip in params.tip_heights_m:
		site_z = top.centre[2] + tip
		for standoff in params.standoffs_m:
			ring: list[ViewCandidate] = []
			prev = None
			for phi in phis:
				cam_xy = ring_point(top, phi, standoff)
				q = None
				for s in [prev] if prev is not None else seeds:
					q = poser.solve(cam_xy, top.centre[:2], site_z, s, poser.mount(s))
					if q is not None:
						break
				if q is None and prev is not None:
					for s in seeds:
						q = poser.solve(cam_xy, top.centre[:2], site_z, s, poser.mount(s))
						if q is not None:
							break
				if q is None:
					prev = None
					continue
				q = q.copy()
				q[5] = gripper
				if not safe(q):
					prev = None
					continue
				mujoco.mj_camlight(m, d)
				v = capture_view(cell, "wrist_cam", data=d)
				ring.append(ViewCandidate(q, "wrist_cam", v.faces_px, v.brick_px, v.visible_fraction))
				prev = q
			# Split the ring, in roll order, where the straight joint path between neighbours is not safe: one sweep
			# stays inside one arc.
			ring.sort(key=lambda c: float(c.q[4]))
			arcs: list[list[ViewCandidate]] = []
			for c in ring:
				if arcs and segment_safe(arcs[-1][-1], c):
					arcs[-1].append(c)
				else:
					arcs.append([c])
			for arc in arcs:
				chosen = greedy_face_cover(arc, faces, min_px, max_views)
				score = _coverage(chosen, faces, min_px)
				if chosen and score > best_score:
					best_score, best_plan = score, _ordered(arc, chosen, tip, standoff)
	return best_plan


def _ordered(ring: list[ViewCandidate], chosen: list[ViewCandidate], tip: float, standoff: float) -> OrbitPlan:
	"""The chosen views in wrist-roll order, each with the ring poses whose roll lies between it and the previous
	view as via points (one sweep in one roll direction)."""
	ring = sorted(ring, key=lambda c: float(c.q[4]))
	chosen = sorted(chosen, key=lambda c: float(c.q[4]))
	via, last = [], None
	for c in chosen:
		lo = -math.inf if last is None else float(last.q[4])
		via.append([r.q for r in ring if lo < float(r.q[4]) < float(c.q[4])] if last is not None else [])
		last = c
	return OrbitPlan(chosen, via, tip, standoff, ring)
