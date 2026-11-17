"""Viewpoint planning for the heuristic inspection: where to put the arm so a camera sees given brick faces.

Two kinds of viewpoints, both scored kinematically on a copy of the simulator state (no physics step):

* **pre-grasp wrist views** (:func:`prescan_candidates`, :func:`greedy_face_cover`): the brick still lies on the
  table and the gripper hovers above it at a few heights, offsets and wrist rolls, tilted by a little wrist flex, so
  the wrist camera (rigid on the gripper) looks at the side faces from different directions. After the grasp the
  wrist camera sees a fixed part of the brick, so this is the only time it can survey the faces;
* **top-camera close-ups** (:func:`closeup_candidates`, :func:`best_closeup`): the held brick carried by the whole arm
  towards the fixed top camera (larger in its image) with a chosen face turned to it.

Face visibility comes from the face-probe segmentation (geometry, not the label); a deployable system would get it
from the brick pose estimate. The planners only choose poses; the expert executes them with the physics.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from ._mj import mujoco
from .cell import Cell
from .inspection_views import HeldBrickPoser, capture_view

__all__ = [
	"ViewCandidate",
	"best_closeup",
	"closeup_candidates",
	"closeup_library",
	"greedy_face_cover",
	"path_min_z",
	"prescan_candidates",
]


@dataclass(frozen=True, eq=False)
class ViewCandidate:
	"""An arm pose (6 joints, gripper included) and what ``camera`` sees of the brick from it."""

	q: np.ndarray
	camera: str
	faces_px: tuple[int, ...]
	brick_px: int
	visible_fraction: float


def _view(cell: Cell, camera: str, d, q: np.ndarray) -> ViewCandidate:
	v = capture_view(cell, camera, data=d)
	return ViewCandidate(np.asarray(q, dtype=np.float64).copy(), camera, v.faces_px, v.brick_px, v.visible_fraction)


def prescan_candidates(
	cell: Cell,
	ik: Callable[[np.ndarray, np.ndarray, float], np.ndarray],
	q_seed: np.ndarray,
	*,
	heights: Sequence[float] = (0.08, 0.11),
	offsets: Sequence[tuple[float, float]] = ((0.0, 0.0),),
	rolls: Iterable[float] = tuple(np.linspace(-2.5, 2.5, 9)),
	dflex: Sequence[float] = (-0.3, 0.0, 0.3),
	min_z: float = 0.03,
	clearance: Callable[[list[np.ndarray]], float] | None = None,
	min_clearance: float = 0.01,
	gripper: float | None = None,
) -> list[ViewCandidate]:
	"""Wrist-camera views of the brick lying on the table, from hover poses above it: for each height, horizontal
	offset and wrist roll the gripper frame is put (``ik(target, q0, roll)``) over the brick, then the wrist flex is
	tilted by each of ``dflex``. Poses whose robot geoms come within ``min_z`` of the table or (with ``clearance``)
	within ``min_clearance`` of bowls / clutter are dropped."""
	m = cell.model
	d = copy.copy(cell.data)
	bpos = cell.data.xpos[cell.brick_body].copy()
	g = float(q_seed[5]) if gripper is None else float(gripper)
	out: list[ViewCandidate] = []
	for h in heights:
		for dx, dy in offsets:
			for roll in rolls:
				q0 = ik(np.array([bpos[0] + dx, bpos[1] + dy, h]), np.asarray(q_seed, dtype=np.float64), float(roll))
				for df in dflex:
					q = q0.copy()
					q[3] += df
					q[5] = g
					q = np.clip(q, cell.jnt_range[:, 0], cell.jnt_range[:, 1])
					d.qpos[cell.qadr] = q
					mujoco.mj_kinematics(m, d)
					mujoco.mj_camlight(m, d)
					if cell.robot_min_z(d) < min_z:
						continue
					if clearance is not None and clearance([q]) < min_clearance:
						continue
					out.append(_view(cell, "wrist_cam", d, q))
	return out


def greedy_face_cover(
	cands: Sequence[ViewCandidate], faces: Iterable[int], min_px: int, max_views: int, times: int = 1
) -> list[ViewCandidate]:
	"""At most ``max_views`` candidates covering ``faces`` (a face is covered by a view showing it with at least
	``min_px`` pixels; with ``times`` > 1 each face must be covered by that many views): greedily the view covering
	the most still-uncovered faces, ties broken by the pixels it shows of them. Stops early once every face is covered
	or no view adds one."""
	need = {int(f): int(times) for f in faces}
	chosen: list[ViewCandidate] = []
	pool = list(cands)

	def gain(c: ViewCandidate) -> tuple[int, int]:
		new = [f for f, n in need.items() if n > 0 and c.faces_px[f] >= min_px]
		return len(new), sum(c.faces_px[f] for f in new)

	while any(n > 0 for n in need.values()) and pool and len(chosen) < max_views:
		best = max(pool, key=gain)
		if gain(best)[0] == 0:
			break
		chosen.append(best)
		pool.remove(best)
		for f in need:
			if best.faces_px[f] >= min_px:
				need[f] -= 1
	return chosen


def closeup_library(rows: Iterable[dict], per_layout: int = 40, min_visible: float = 0.6) -> np.ndarray:
	"""Arm joint vectors (N, 5) worth trying for top-camera close-ups: from a ``top_camera_reach.py`` sweep, the
	``per_layout`` poses with the most unoccluded brick pixels per layout."""
	by_layout: dict[int, list[dict]] = {}
	for r in rows:
		if r.get("source") == "random" and r["visible_fraction"] >= min_visible:
			by_layout.setdefault(int(r["layout"]), []).append(r)
	qs = []
	for sel in by_layout.values():
		sel.sort(key=lambda r: -r["brick_px"])
		qs += [r["q"] for r in sel[:per_layout]]
	return np.asarray(qs, dtype=np.float64).reshape(-1, 5)


def closeup_candidates(
	cell: Cell,
	library: np.ndarray,
	*,
	gripper: float,
	rolls_deg: Sequence[float] = (-120.0, -60.0, 0.0, 60.0, 120.0),
	clearance_m: float = 0.04,
	min_visible: float = 0.5,
) -> list[ViewCandidate]:
	"""Top-camera views of the HELD brick (carried rigidly by the gripper, see :class:`HeldBrickPoser`) at every
	library arm pose, with the wrist roll replaced by each of ``rolls_deg`` (the roll turns the brick in front of the
	camera); poses that bring the arm or the brick within ``clearance_m`` of the table or leave the brick mostly
	occluded are dropped."""
	poser = HeldBrickPoser(cell)
	lo, hi = cell.jnt_range[:, 0], cell.jnt_range[:, 1]
	out: list[ViewCandidate] = []
	for q5 in library:
		for r in rolls_deg:
			q = np.array([*q5, gripper], dtype=np.float64)
			q[4] = np.deg2rad(r)
			q = np.clip(q, lo, hi)
			d = poser.at(q)
			if min(cell.robot_min_z(d), poser.brick_min_z()) < clearance_m:
				continue
			c = _view(cell, "top", d, q)
			if c.visible_fraction >= min_visible:
				out.append(c)
	return out


def best_closeup(cands: Sequence[ViewCandidate], face: int, min_share: float = 0.0) -> ViewCandidate | None:
	"""The candidate showing ``face`` with the most pixels (None if no candidate shows it at all). ``min_share``
	optionally requires the face to make up at least that fraction of the visible brick."""
	best, best_px = None, 0
	for c in cands:
		px = c.faces_px[face]
		if px > best_px and (min_share <= 0 or px >= min_share * max(c.brick_px, 1)):
			best, best_px = c, px
	return best


def path_min_z(cell: Cell, q_from: np.ndarray, q_to: np.ndarray, *, carried: bool, steps: int = 12) -> float:
	"""Lowest robot (and, if ``carried``, held brick) point above the table along the straight joint-space path
	(what a min-jerk move follows)."""
	poser = HeldBrickPoser(cell) if carried else None
	d = copy.copy(cell.data)
	low = np.inf
	for s in np.linspace(0.0, 1.0, steps):
		q = np.asarray(q_from) + s * (np.asarray(q_to) - np.asarray(q_from))
		if poser is not None:
			dd = poser.at(q)
			low = min(low, cell.robot_min_z(dd), poser.brick_min_z())
		else:
			d.qpos[cell.qadr] = q
			mujoco.mj_kinematics(cell.model, d)
			low = min(low, cell.robot_min_z(d))
	return float(low)
