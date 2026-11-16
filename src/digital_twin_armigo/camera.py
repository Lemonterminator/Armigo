"""Pinhole camera above the table: pose parametrisation, projection and ray casting. Pure NumPy."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .constants import PLANE_HEIGHT_M, PLANE_TILT_DEG, TOP_FULL


@dataclass
class CameraPose:
	"""A camera above the table: horizontal position, height, view yaw, tilt from vertical, roll, focal length."""

	x: float
	y: float
	height: float = PLANE_HEIGHT_M
	yaw_deg: float = 0.0
	tilt_deg: float = PLANE_TILT_DEG
	roll_deg: float = 0.0
	focal_px: float = 615.0

	def frame(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
		"""Image right, image up and viewing direction of the camera, as unit vectors in the world."""
		t, p = math.radians(self.tilt_deg), math.radians(self.yaw_deg)
		forward = np.array([math.sin(t) * math.cos(p), math.sin(t) * math.sin(p), -math.cos(t)])
		up = np.array([math.cos(t) * math.cos(p), math.cos(t) * math.sin(p), math.sin(t)])
		right = np.cross(forward, up)
		r = math.radians(self.roll_deg)
		right, up = math.cos(r) * right + math.sin(r) * up, -math.sin(r) * right + math.cos(r) * up
		return right, up, forward

	def axes(self) -> tuple[np.ndarray, np.ndarray]:
		"""MuJoCo camera x (image right) and y (image up) axes in the world."""
		right, up, _ = self.frame()
		return right, up

	@property
	def position(self) -> np.ndarray:
		return np.array([self.x, self.y, self.height])

	def mjcf(self, name: str, resolution: tuple[int, int]) -> str:
		right, up = self.axes()
		xy = " ".join(f"{v:.6f}" for v in (*right, *up))
		return (
			f'<camera name="{name}" pos="{self.x:.5f} {self.y:.5f} {self.height:.5f}" xyaxes="{xy}" '
			f'resolution="{resolution[0]} {resolution[1]}" focalpixel="{self.focal_px:.2f} {self.focal_px:.2f}" '
			f'sensorsize="{resolution[0] * 1e-5:.6f} {resolution[1] * 1e-5:.6f}"/>'
		)


def as_pose(cam: CameraPose | Mapping[str, float]) -> CameraPose:
	"""A :class:`CameraPose` from itself or from the ``camera`` mapping of a camera JSON file."""
	return cam if isinstance(cam, CameraPose) else CameraPose(**cam)


def opengl_axes(cam: CameraPose | Mapping[str, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Columns of the camera-to-world rotation in the OpenGL convention: +x right, +y up, -z viewing direction."""
	right, up, forward = as_pose(cam).frame()
	return right, up, -forward


def look_from(
	target: tuple[float, float], distance_xy: float, height: float, yaw_deg: float, **kw: float
) -> CameraPose:
	"""A camera ``distance_xy`` behind ``target`` (along -yaw) at ``height``, tilted to look at the target."""
	p = math.radians(yaw_deg)
	tilt = math.degrees(math.atan2(distance_xy, height))
	return CameraPose(
		target[0] - distance_xy * math.cos(p),
		target[1] - distance_xy * math.sin(p),
		height,
		yaw_deg,
		tilt,
		**kw,
	)


def camera_forward(pose: CameraPose) -> np.ndarray:
	right, up = pose.axes()
	return np.cross(up, right)


def project(pose: CameraPose, points: np.ndarray, resolution: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
	"""Pixel coordinates (u right, v down) of world points, and whether each lies in front of the camera."""
	right, up = pose.axes()
	forward = np.cross(up, right)
	rel = np.asarray(points, dtype=np.float64) - np.array([pose.x, pose.y, pose.height])
	x, y, z = rel @ right, rel @ up, rel @ forward
	in_front = z > 0.05
	z = np.where(in_front, z, 1.0)
	u = resolution[0] / 2 + pose.focal_px * x / z
	v = resolution[1] / 2 - pose.focal_px * y / z
	return np.stack([u, v], axis=-1), in_front


def table_point(pose: CameraPose, u: np.ndarray, v: np.ndarray, z: float) -> np.ndarray:
	"""Where the rays through full-frame pixels (u, v) meet the horizontal plane at height ``z``."""
	right, up = pose.axes()
	forward = np.cross(up, right)
	x = (np.asarray(u, np.float64) - TOP_FULL[0] / 2) / pose.focal_px
	y = -(np.asarray(v, np.float64) - TOP_FULL[1] / 2) / pose.focal_px
	ray = x[:, None] * right + y[:, None] * up + forward
	t = (z - pose.height) / ray[:, 2]
	return np.array([pose.x, pose.y]) + t[:, None] * ray[:, :2]
