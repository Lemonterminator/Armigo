"""Brick localisation from the top camera's depth image through a hand-eye-calibrated camera pose.

Enabled by pointing ``ARMIGO_BRICK_PERCEPTION`` at a JSON :class:`PerceptionCfg` (worker processes inherit it).
The agent then calibrates the top camera before each episode (:func:`~..mujoco_cell.handeye_rig.calibrate_layout`)
and, whenever the brick lies on the table and a skill needs it, locates it in a depth image:

1. back-project every pixel through the *estimated* camera pose into the robot frame;
2. fit the table plane to the points near z = 0 (a camera rotation error tilts the table in the estimated frame,
   and a fixed height threshold would then pick up the far table), and measure heights from that plane;
3. keep points 6 to 50 mm above the table, within reach and outside the two bowls;
4. take the largest connected blob on a 1 mm grid, and its top face (points within 4 mm of the blob's top);
5. fit the smallest rectangle around the top face: centre, yaw and footprint (2x2 or 2x4);
6. if a side comes out shorter than the Duplo size, something nearer the camera (usually the arm) may hide part of
   the top face: extend the rectangle towards the side whose missing strip the depth image shows covered.

The brick is assumed to rest upright on the table, so its height comes from the size class.

:func:`locate_bowls` finds the two bowls by colour and fits a circle to the 3D points of each bowl's outer rim, and
:func:`bowl_fill` tells whether something now lies in a bowl by comparing the depth image with one taken before.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..constants import BOWL_R, DUPLO_2X2, DUPLO_2X4
from ..mujoco_cell.handeye_rig import CalibrationCfg

ENV = "ARMIGO_BRICK_PERCEPTION"
MIN_HEIGHT_M = 0.006
MAX_HEIGHT_M = 0.05
REACH_M = (0.10, 0.32)
BOWL_MARGIN_M = 0.02
TOP_BAND_M = 0.004
GRID_M = 0.001
MIN_TOP_POINTS = 40
TABLE_BAND_M = 0.03  # first guess of table points around z = 0
TABLE_INLIER_M = 0.003
MIN_DEFICIT_M = 0.003  # a fitted side this much shorter than the brick is checked for occlusion
OCCLUDER_GAP_M = 0.01  # a pixel is covered when its depth is this much nearer than the hidden top face
# Bowl colours as OpenCV HSV bounds (H 0-180): hue low, hue high, minimum saturation, minimum value.
BOWL_HSV = {"green": (45, 85, 70, 30), "pink": (150, 178, 40, 90)}
BOWL_MIN_PIXELS = 3000
RIM_INLIER_M = 0.004
FILL_GAP_M = 0.008  # a bowl pixel is filled when it came this much nearer the camera
FILL_MIN_PIXELS = 500  # a 2x2 brick in a bowl shows about 1400; arm edges and noise up to about 300


@dataclass(frozen=True)
class PerceptionCfg:
	calibration: CalibrationCfg = field(default_factory=CalibrationCfg)
	depth_noise_m: float = 0.0015  # σ at 0.5 m; grows with the square of the depth, as a stereo camera's does

	@classmethod
	def realistic(cls) -> PerceptionCfg:
		"""``config/agent_bench/perception/realistic.json``: joint noise 0.1°, zero-offset σ 0.5°, offsets solved."""
		return cls(calibration=CalibrationCfg(joint_noise_deg=0.1, joint_bias_deg=0.5, offsets=True))

	@classmethod
	def from_env(cls) -> PerceptionCfg | None:
		path = os.environ.get(ENV)
		if not path:
			return None
		raw = json.loads(Path(path).read_text(encoding="utf-8"))
		return cls(calibration=CalibrationCfg(**raw.pop("calibration", {})), **raw)


@dataclass
class BrickEstimate:
	pos: np.ndarray  # centre, robot frame (m)
	yaw: float  # radians, of the brick's long axis (any of the four sides for a 2x2)
	size: tuple[float, float, float]
	footprint_m: tuple[float, float]  # fitted top-face rectangle, long side first
	top_points: int
	extended_mm: tuple[float, float] = (0.0, 0.0)  # occlusion correction along the long and short axes

	def quat_xyzw(self) -> np.ndarray:
		return np.array([0.0, 0.0, math.sin(self.yaw / 2), math.cos(self.yaw / 2)])


def depth_points(depth: np.ndarray, k: np.ndarray, world_from_cam: np.ndarray) -> np.ndarray:
	"""(h, w, 3) robot-frame points of a planar depth image (metres along the optical axis)."""
	h, w = depth.shape
	u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
	cam = np.stack([(u - k[0, 2]) / k[0, 0] * depth, (v - k[1, 2]) / k[1, 1] * depth, depth], axis=-1)
	return cam @ world_from_cam[:3, :3].T + world_from_cam[:3, 3]


def yaw_error_deg(est: float, truth: float, size: tuple[float, float, float]) -> float:
	"""Yaw difference modulo the brick's footprint symmetry (90° for a square, 180° otherwise)."""
	period = math.pi / 2 if abs(size[0] - size[1]) < 1e-6 else math.pi
	d = (est - truth) % period
	return math.degrees(min(d, period - d))


def _covered_fraction(strip: np.ndarray, depth: np.ndarray, k: np.ndarray, cam_from_world: np.ndarray) -> float:
	"""Fraction of robot-frame points ``strip`` that the depth image shows hidden behind something nearer."""
	cam = strip @ cam_from_world[:3, :3].T + cam_from_world[:3, 3]
	u = np.round(cam[:, 0] / cam[:, 2] * k[0, 0] + k[0, 2]).astype(int)
	v = np.round(cam[:, 1] / cam[:, 2] * k[1, 1] + k[1, 2]).astype(int)
	inside = (u >= 0) & (u < depth.shape[1]) & (v >= 0) & (v < depth.shape[0])
	if not inside.any():
		return 0.0
	return float(np.mean(depth[v[inside], u[inside]] < cam[inside, 2] - OCCLUDER_GAP_M))


def table_plane(points: np.ndarray) -> np.ndarray | None:
	"""(a, b, c) of the table z = a x + b y + c, by least squares refitted on its inliers."""
	if len(points) < 500:
		return None
	keep = np.ones(len(points), bool)
	for _ in range(3):
		a = np.column_stack([points[keep, :2], np.ones(int(keep.sum()))])
		coef, *_ = np.linalg.lstsq(a, points[keep, 2], rcond=None)
		res = points[:, 2] - (points[:, :2] @ coef[:2] + coef[2])
		keep = np.abs(res) < TABLE_INLIER_M
		if keep.sum() < 500:
			return None
	return coef


def locate_brick(
	depth: np.ndarray, k: np.ndarray, world_from_cam: np.ndarray, bowls: list[tuple[float, float]]
) -> BrickEstimate | None:
	"""The brick on the table in a planar depth image seen through ``world_from_cam``, or None when no blob has a
	usable top face."""
	import cv2

	p = depth_points(depth, k, world_from_cam).reshape(-1, 3)
	p = p[np.isfinite(p).all(axis=1)]
	r = np.hypot(p[:, 0], p[:, 1])
	area = (r > REACH_M[0]) & (r < 0.45)
	for bx, by in bowls:
		area &= np.hypot(p[:, 0] - bx, p[:, 1] - by) > BOWL_R + BOWL_MARGIN_M
	plane = table_plane(p[area & (np.abs(p[:, 2]) < TABLE_BAND_M)])
	if plane is None:
		return None
	p = p[area & (r < REACH_M[1])]
	p[:, 2] -= p[:, :2] @ plane[:2] + plane[2]  # height above the fitted table
	p = p[(p[:, 2] > MIN_HEIGHT_M) & (p[:, 2] < MAX_HEIGHT_M)]
	if len(p) < MIN_TOP_POINTS:
		return None
	lo = p[:, :2].min(axis=0) - 0.005
	cells = np.floor((p[:, :2] - lo) / GRID_M).astype(int)
	shape = cells.max(axis=0) + 6
	occ = np.zeros((shape[1], shape[0]), np.uint8)
	occ[cells[:, 1], cells[:, 0]] = 255
	occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
	n, labels, stats, _ = cv2.connectedComponentsWithStats(occ, connectivity=8)
	if n < 2:
		return None
	best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
	blob = p[labels[cells[:, 1], cells[:, 0]] == best]
	z_top = float(np.percentile(blob[:, 2], 90))
	top = blob[blob[:, 2] > z_top - TOP_BAND_M]
	if len(top) < MIN_TOP_POINTS:
		return None
	(cx, cy), (w, h), angle = cv2.minAreaRect(top[:, :2].astype(np.float32))
	yaw = math.radians(angle) if w >= h else math.radians(angle + 90)
	long_side, short_side = max(w, h), min(w, h)
	size = DUPLO_2X4 if long_side > 0.048 else DUPLO_2X2
	period = math.pi if size is DUPLO_2X4 else math.pi / 2
	yaw = (yaw + period / 2) % period - period / 2
	centre = np.array([cx, cy])
	cam_from_world = np.linalg.inv(world_from_cam)
	extended = []
	axes = (np.array([math.cos(yaw), math.sin(yaw)]), np.array([-math.sin(yaw), math.cos(yaw)]))
	for axis, fitted, full, other in ((axes[0], long_side, size[0], axes[1]), (axes[1], short_side, size[1], axes[0])):
		deficit = full - fitted
		if deficit < MIN_DEFICIT_M:
			extended.append(0.0)
			continue
		# the missing strip on each side, sampled on the top-face plane
		half_other = (short_side if axis is axes[0] else long_side) / 2
		a = np.linspace(fitted / 2, full - fitted / 2, 6)
		b = np.linspace(-half_other, half_other, 9)
		covered = []
		for sign in (1.0, -1.0):
			xy = centre + sign * a[:, None, None] * axis + b[None, :, None] * other
			xy = xy.reshape(-1, 2)
			strip = np.column_stack([xy, xy @ plane[:2] + plane[2] + z_top])
			covered.append(_covered_fraction(strip, depth, k, cam_from_world))
		if max(covered) < 0.5:
			extended.append(0.0)  # the shortfall is not explained by an occluder: keep the fit
			continue
		sign = 1.0 if covered[0] >= covered[1] else -1.0
		centre = centre + sign * deficit / 2 * axis
		extended.append(round(deficit * 1000, 2))
	table_z = float(centre @ plane[:2] + plane[2])
	pos = np.array([centre[0], centre[1], table_z + size[2] / 2 + 0.0005])  # resting upright on the table
	return BrickEstimate(pos, yaw, size, (float(long_side), float(short_side)), len(top), tuple(extended))


@dataclass
class BowlEstimate:
	xy: np.ndarray  # rim centre, robot frame (m)
	radius_m: float
	pixels: int
	rim_points: int


def _plane_heights(p: np.ndarray, bowls_hint=()) -> np.ndarray | None:
	"""Heights of robot-frame points ``p`` (n, 3) above the table plane fitted to them."""
	r = np.hypot(p[:, 0], p[:, 1])
	near = np.isfinite(p).all(axis=1) & (r > REACH_M[0]) & (r < 0.45) & (np.abs(p[:, 2]) < TABLE_BAND_M)
	plane = table_plane(p[near])
	if plane is None:
		return None
	return p[:, 2] - (p[:, :2] @ plane[:2] + plane[2])


def locate_bowls(
	rgb: np.ndarray, depth: np.ndarray, k: np.ndarray, world_from_cam: np.ndarray
) -> dict[str, BowlEstimate | None]:
	"""The green and pink bowls in a top image: the largest blob of each colour, then a circle fitted to the 3D points
	of its outer contour that stand on the rim (the contour also runs along occluders and the image edge; those
	points sit lower or off the circle and are dropped by the height cut and the inlier refit)."""
	import cv2

	hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
	pts = depth_points(depth, k, world_from_cam)
	heights = _plane_heights(pts.reshape(-1, 3))
	out: dict[str, BowlEstimate | None] = {}
	for name, (lo, hi, smin, vmin) in BOWL_HSV.items():
		out[name] = None
		mask = (hsv[..., 0] >= lo) & (hsv[..., 0] <= hi) & (hsv[..., 1] >= smin) & (hsv[..., 2] >= vmin)
		n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
		if n < 2:
			continue
		best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
		area = int(stats[best, cv2.CC_STAT_AREA])
		if area < BOWL_MIN_PIXELS:
			continue
		comp = labels == best
		contours, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		edge = max(contours, key=len)[:, 0]
		rim = pts[edge[:, 1], edge[:, 0]]
		if heights is not None:
			h = heights.reshape(depth.shape)
			blob_top = float(np.percentile(h[comp], 95))
			rim = rim[h[edge[:, 1], edge[:, 0]] > 0.6 * blob_top]
		rim = rim[np.isfinite(rim).all(axis=1)]
		if len(rim) < 20:
			continue
		keep = np.ones(len(rim), bool)
		for _ in range(4):
			a = np.column_stack([2 * rim[keep, :2], np.ones(int(keep.sum()))])
			(cx, cy, c), *_ = np.linalg.lstsq(a, (rim[keep, :2] ** 2).sum(axis=1), rcond=None)
			radius = math.sqrt(max(c + cx * cx + cy * cy, 0.0))
			keep = np.abs(np.hypot(rim[:, 0] - cx, rim[:, 1] - cy) - radius) < RIM_INLIER_M
			if keep.sum() < 10:
				break
		if abs(radius - BOWL_R) > 0.015:
			continue
		out[name] = BowlEstimate(np.array([cx, cy]), float(radius), area, int(keep.sum()))
	return out


def bowl_fill(
	depth: np.ndarray, baseline: np.ndarray, k: np.ndarray, world_from_cam: np.ndarray, xy, margin_m: float = 0.015
) -> int:
	"""Pixels inside the bowl at ``xy`` that came at least FILL_GAP_M nearer the camera than in ``baseline`` and lie
	low (under 6 cm above the table, so the arm passing over does not count)."""
	p = depth_points(depth, k, world_from_cam)
	heights = _plane_heights(p.reshape(-1, 3))
	if heights is None:
		return 0
	inside = np.hypot(p[..., 0] - xy[0], p[..., 1] - xy[1]) < BOWL_R - margin_m
	low = heights.reshape(depth.shape) < 0.06
	return int(np.count_nonzero(inside & low & (depth < baseline - FILL_GAP_M)))


def bowl_filled(count: int) -> bool:
	return count >= FILL_MIN_PIXELS
