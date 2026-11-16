"""Scripted expert: pick, present the brick to the main camera, hand off and place, as per-frame joint targets."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..camera import CameraPose, camera_forward
from ..constants import FPS, HANDOFF_Q, PHASES, TOP_CROP
from ..presentation import order_show_poses, show_points, view_cosine
from ._mj import mujoco
from .cell import Cell
from .ik import best_roll, ik, ik_held, ik_reach
from .layout import release_point


def min_jerk(n: int) -> np.ndarray:
	s = np.linspace(0.0, 1.0, n)
	return 10 * s**3 - 15 * s**4 + 6 * s**5


class Plan:
	"""Joint targets (radians, one per frame) with a phase label per frame, built from min-jerk segments."""

	def __init__(self, start: np.ndarray, speed: float) -> None:
		self.frames: list[np.ndarray] = []
		self.phases: list[int] = []
		self.last = start.copy()
		self.speed = speed

	def hold(self, secs: float, phase: str) -> None:
		n = round(secs * FPS)
		self.frames.extend([self.last.copy()] * n)
		self.phases.extend([PHASES.index(phase)] * n)

	def move(self, target: np.ndarray, secs: float, phase: str, settle: float = 0.2) -> None:
		n = max(2, int(secs * self.speed * FPS))
		for s in min_jerk(n)[1:]:
			self.frames.append(self.last + s * (target - self.last))
			self.phases.append(PHASES.index(phase))
		self.last = target.copy()
		self.hold(settle, phase)


@dataclass
class Grip:
	"""What the pick decided and the place needs: wrist roll for jaws-down moves and the gripper commands."""

	roll: float
	q_lift: np.ndarray | None = None
	open_g: float = 1.3
	closed_g: float = -0.17


def expert_pick(cell: Cell, rest: np.ndarray, rng: np.random.Generator) -> tuple[Plan, Grip]:
	"""Rest → hover → descend → close → lift, jaws down across the brick's short side."""
	lay = cell.layout
	bx, by = lay.brick_xy
	yaw = math.radians(lay.brick_yaw_deg)
	short_axis = np.array([-math.sin(yaw), math.cos(yaw)])
	q_hover = ik(cell, np.array([bx, by, 0.09]), rest.copy(), 0.0)
	roll, sign = best_roll(cell, q_hover, short_axis)
	grip = Grip(roll)
	# gripperframe sits on the fixed jaw: put that jaw just outside the brick's long side, the moving jaw opposite.
	offset = -sign * short_axis * (lay.brick_size[1] / 2 + 0.006)
	gx, gy = bx + offset[0], by + offset[1]
	z_grip = 0.004
	plan = Plan(rest, rng.uniform(0.8, 1.25))
	plan.hold(rng.uniform(0.5, 1.2), "rest")
	q = rest.copy()
	for x, y, z, g, secs, phase in (
		(gx, gy, 0.09, grip.open_g, 2.5, "approach"),
		(gx, gy, z_grip + 0.02, grip.open_g, 1.5, "approach"),
		(gx, gy, z_grip, grip.open_g, 0.8, "grasp"),
		(gx, gy, z_grip, grip.closed_g, 1.0, "grasp"),
		(gx, gy, 0.10, grip.closed_g, 1.2, "lift"),
	):
		q = ik(cell, np.array([x, y, z]), q, roll)
		q[5] = g
		plan.move(q, secs, phase)
	grip.q_lift = q.copy()
	return plan, grip


def expert_handoff(cell: Cell, plan: Plan, grip: Grip, rng: np.random.Generator) -> None:
	"""Append the move to the standard holding pose and a short hold there (gripper stays closed)."""
	q = np.array([*HANDOFF_Q, grip.closed_g])
	plan.move(q, 1.5, "handoff", settle=0.0)
	plan.hold(rng.uniform(0.5, 0.8), "handoff")


def expert_place(cell: Cell, plan: Plan, grip: Grip, rest: np.ndarray) -> None:
	"""Append bowl approach → release → rest to ``plan`` (jaws down, from wherever the plan ended)."""
	lay = cell.layout
	bowl = release_point(lay.pink if lay.defective else lay.green)
	# IK starts from the jaws-down lift pose: from a presentation or handoff pose it may not converge
	q = (grip.q_lift if grip.q_lift is not None else plan.last).copy()
	for z, g, secs in (
		(0.11, grip.closed_g, 2.2),
		(0.06, grip.closed_g, 0.8),
		(0.06, grip.open_g, 0.6),
		(0.12, grip.open_g, 0.8),
	):
		q = ik_reach(cell, np.array([bowl[0], bowl[1], z]), q, grip.roll)
		q[5] = g
		plan.move(q, secs, "place")
	plan.move(rest.copy(), 2.5, "return")


SHOW_DISTANCES_M = (0.30, 0.35, 0.40, 0.45, 0.50)

SHOW_OFF_AXIS_DEG = 12.0

SHOW_REACH_M = (0.12, 0.40)

SHOW_Z_MIN_M = 0.10


def presentation_search(cell: Cell, q_now: np.ndarray, main: CameraPose, max_points: int = 3) -> dict:
	"""Score presentation poses of the held brick in front of the main ("top") camera.

	Show points come from :func:`defect_sorter.presentation.show_points` (0.3 to 0.5 m from the camera, near its
	optical axis, in reach). At each point a wrist flex x wrist roll grid is solved for the brick centre; a pose
	counts when the arm stays clear of the table, the brick stays inside the crop and at least half of it is not
	hidden by the arm. Its score is the visible defect pixels times the cosine between the defect face normal and the
	view direction. The first show point with any scoring pose wins. The simulation state is restored afterwards.
	"""
	m, d = cell.model, cell.data
	saved_q, saved_v = d.qpos.copy(), d.qvel.copy()
	cam = np.array([main.x, main.y, main.height])
	points = show_points(cam, camera_forward(main), SHOW_DISTANCES_M, SHOW_OFF_AXIS_DEG, SHOW_REACH_M, SHOW_Z_MIN_M)
	flex_lo, flex_hi = m.jnt_range[m.joint("wrist_flex").id]
	roll_lo, roll_hi = m.jnt_range[m.joint("wrist_roll").id]
	flexes = np.linspace(flex_lo + 0.15, flex_hi - 0.15, 7)
	rolls = np.linspace(roll_lo, roll_hi, 9)
	half_diag = 0.5 * float(np.linalg.norm(cell.layout.brick_size))
	result: dict = {"show_point": None, "poses": [], "points_tried": 0}
	try:
		for point in points[:max_points]:
			result["points_tried"] += 1
			poses = []
			warm = q_now.copy()
			for flex in flexes:
				for roll in rolls:
					q0 = warm.copy()
					q0[3], q0[4] = flex, roll
					q, err = ik_held(cell, point, q0)
					if err > 0.006:
						continue
					warm = q
					d.qpos[cell.qadr] = q
					cell.place_held()
					if cell.robot_min_z() < 0.02 or cell.brick_pos()[2] - half_diag < 0.03:
						continue
					vis = cell.visibility("top", TOP_CROP)
					if vis["brick_full"] < 50 or vis["brick"] < 0.5 * vis["brick_full"]:
						continue
					centre, normal = cell.defect_normal()
					cos = view_cosine(normal, centre, cam)
					poses.append({"q": q.tolist(), "score": vis["defect"] * max(cos, 0.0), "cos": cos, **vis})
			if any(p["score"] > 0 for p in poses):
				result["show_point"] = point.tolist()
				result["poses"] = poses
				break
	finally:
		d.qpos[:], d.qvel[:] = saved_q, saved_v
		mujoco.mj_forward(m, d)
	return result


def expert_present(plan: Plan, search: dict, rng: np.random.Generator, closed_g: float) -> list[dict]:
	"""Append the presentation to ``plan``: up to three distinct poses, best defect view first, each held 0.3 to 0.5 s.

	Returns the chosen poses (with their scores) in presentation order.
	"""
	poses = search["poses"]
	if not poses:
		return []
	keys = np.array([[p["q"][3], p["q"][4]] for p in poses])
	order = order_show_poses(
		keys, np.array([p["score"] for p in poses]), k=int(rng.integers(2, 4)), min_sep=0.6, rel_min=0.3
	)
	chosen = []
	for n, i in enumerate(order):
		q = np.array(poses[i]["q"])
		q[5] = closed_g
		plan.move(q, 1.8 if n == 0 else 1.0, "show_move", settle=0.0)
		plan.hold(rng.uniform(0.3, 0.5), "show_hold")
		chosen.append(poses[i])
	return chosen
