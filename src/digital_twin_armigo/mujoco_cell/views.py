"""Camera randomisation for rendering and the checks that keep a sampled camera plausible."""

from __future__ import annotations

import math

import numpy as np

from ..camera import CameraPose, look_from, project
from ..constants import BOWL_H, BOWL_R, DUPLO_2X2, TOP_CROP, TOP_FULL


def sample_top_camera(rng: np.random.Generator, nominal: CameraPose) -> CameraPose:
	"""A top camera near the fitted real one (±3 cm, ±5 deg yaw, ±3 deg tilt, ±3 % focal)."""
	return CameraPose(
		nominal.x + rng.uniform(-0.03, 0.03),
		nominal.y + rng.uniform(-0.03, 0.03),
		nominal.height + rng.uniform(-0.03, 0.03),
		nominal.yaw_deg + rng.uniform(-5.0, 5.0),
		nominal.tilt_deg + rng.uniform(-3.0, 3.0),
		nominal.roll_deg + rng.uniform(-1.5, 1.5),
		nominal.focal_px * rng.uniform(0.97, 1.03),
	)


def sample_front_camera(rng: np.random.Generator) -> CameraPose:
	"""A low side camera looking across the workspace from the side opposite the robot."""
	return look_from(
		(0.22, 0.0),
		rng.uniform(0.45, 0.65),
		rng.uniform(0.22, 0.38),
		rng.uniform(155, 205),
		focal_px=rng.uniform(480, 560),
	)


def sample_side_camera(rng: np.random.Generator) -> CameraPose:
	"""A fully random camera above the table looking down at the workspace from any side (checked by the caller)."""
	return look_from(
		(rng.uniform(0.15, 0.35), rng.uniform(-0.10, 0.20)),
		rng.uniform(0.30, 0.70),
		rng.uniform(0.20, 0.70),
		rng.uniform(0.0, 360.0),
		roll_deg=rng.normal(0, 2.0),
		focal_px=rng.uniform(450, 620),
	)


def perturb_main_camera(rng: np.random.Generator, main: CameraPose) -> CameraPose:
	"""The simulated main camera moved slightly (±5 mm, ±0.5 deg, ±0.5 % focal): show poses stay valid."""
	return CameraPose(
		main.x + rng.uniform(-0.005, 0.005),
		main.y + rng.uniform(-0.005, 0.005),
		main.height + rng.uniform(-0.005, 0.005),
		main.yaw_deg + rng.uniform(-0.5, 0.5),
		main.tilt_deg + rng.uniform(-0.5, 0.5),
		main.roll_deg + rng.uniform(-0.5, 0.5),
		main.focal_px * rng.uniform(0.995, 1.005),
	)


def bowl_rim_points(xy: tuple[float, float], n: int = 16) -> np.ndarray:
	a = np.linspace(0, 2 * math.pi, n, endpoint=False)
	return np.stack([xy[0] + BOWL_R * np.cos(a), xy[1] + BOWL_R * np.sin(a), np.full(n, BOWL_H)], axis=-1)


def top_view_ok(pose: CameraPose, layout: dict, tcp_path: np.ndarray, margin_px: float = 20.0) -> tuple[bool, str]:
	"""Whether a top camera is plausible: above the table, sees the brick, the gripper path and both bowl centres
	inside the crop with a margin and at least 60 % of each bowl rim inside the crop (the real top view cuts the far
	rim of the bowls), and shows the brick at least 15 px across."""
	if pose.height < 0.25 or not 10.0 <= pose.tilt_deg <= 45.0:
		return False, "pose out of range"
	size = layout.get("brick_size", DUPLO_2X2)
	brick = np.array([[*layout["brick_xy"], size[2]]])
	x0, y0, w, h = TOP_CROP

	def inside(points: np.ndarray, margin: float) -> np.ndarray:
		uv, front = project(pose, points, TOP_FULL)
		ok = (uv[:, 0] > x0 + margin) & (uv[:, 0] < x0 + w - margin) & (uv[:, 1] > y0 + margin)
		return ok & (uv[:, 1] < y0 + h - margin) & front

	centres = np.array([[*layout[b], BOWL_H] for b in ("green", "pink")])
	key = inside(np.vstack([centres, brick, tcp_path]), margin_px)
	if not key.all():
		return False, f"{int((~key).sum())} key points outside the crop"
	for b in ("green", "pink"):
		if inside(bowl_rim_points(layout[b]), 0.0).mean() < 0.6:
			return False, f"{b} bowl mostly outside the crop"
	ends, _ = project(pose, brick + np.array([[size[0] / 2, 0, 0], [-size[0] / 2, 0, 0]]), TOP_FULL)
	if np.linalg.norm(ends[0] - ends[1]) < 15.0:
		return False, "brick too small"
	return True, ""
