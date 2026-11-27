"""Height-above-table images in the depth bridge ``layers`` encoding (dataset key ``top_depth_v2``). Pure NumPy."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from .camera import CameraPose, as_pose
from .constants import COARSE_MAX_M, FINE_MAX_M


def layered_height_image(height_m: np.ndarray, valid: np.ndarray) -> np.ndarray:
	"""Height above the table → RGB uint8 in the depth bridge ``layers`` encoding (R fine, G coarse, B valid)."""
	h = np.clip(np.where(valid, height_m, 0.0), 0.0, None)
	out = np.empty((*h.shape, 3), np.uint8)
	out[..., 0] = np.rint(255.0 * np.minimum(h / FINE_MAX_M, 1.0))
	out[..., 1] = np.rint(255.0 * np.minimum(h / COARSE_MAX_M, 1.0))
	out[..., 2] = np.where(valid, 255, 0)
	out[~valid, :2] = 0
	return out


def decode_height(depth_rgb: np.ndarray) -> np.ndarray:
	"""Height above the table (m, NaN where invalid) from a ``layers``-encoded depth image."""
	r, g, b = (depth_rgb[..., k].astype(np.float64) for k in range(3))
	fine = g < 255 * 0.035 / COARSE_MAX_M
	h = np.where(fine, r / 255 * FINE_MAX_M, g / 255 * COARSE_MAX_M)
	return np.where(b > 128, h, np.nan)


def _pixel_grid(
	full_shape: tuple[int, int], crop: tuple[int, int, int, int] | None, f: float
) -> tuple[np.ndarray, np.ndarray]:
	"""Normalised image coordinates (right, down) of the pixel centres, about the full-frame centre."""
	full_h, full_w = full_shape
	u0, v0, wid, hgt = (0, 0, full_w, full_h) if crop is None else crop
	u, v = np.meshgrid(np.arange(u0, u0 + wid) + 0.5, np.arange(v0, v0 + hgt) + 0.5)
	return (u - full_w / 2.0) / f, (v - full_h / 2.0) / f


def _ray_world_z_grid(
	pose: CameraPose,
	full_shape: tuple[int, int],
	crop: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
	"""Compute world Z component of camera viewing rays per pixel using full-frame principal point.

	``full_shape`` is (height, width), as a NumPy image shape; note that ``TOP_FULL`` is (width, height).

	The grid depends only on the camera's orientation, focal length and the pixel window, which stay fixed for a whole
	episode, so it is cached (read-only). Only the Z component is formed; it is the same floating-point expression as
	``(x * right + (-y) * up + forward)[..., 2]``, so the result is bit-identical."""
	right, up, forward = pose.frame()
	key = (
		right[2].item(),
		up[2].item(),
		forward[2].item(),
		float(pose.focal_px),
		tuple(int(s) for s in full_shape),
		None if crop is None else tuple(int(c) for c in crop),
	)
	hit = _RAY_Z_CACHE.get(key)
	if hit is not None:
		return hit
	x, y = _pixel_grid(full_shape, crop, pose.focal_px)
	ray_z = x * right[2] + (-y) * up[2] + forward[2]
	ray_z.flags.writeable = False
	if len(_RAY_Z_CACHE) >= _RAY_Z_CACHE_SIZE:
		_RAY_Z_CACHE.pop(next(iter(_RAY_Z_CACHE)))
	_RAY_Z_CACHE[key] = ray_z
	return ray_z


_RAY_Z_CACHE: dict[tuple, np.ndarray] = {}
_RAY_Z_CACHE_SIZE = 32


def depth_plane_to_height(
	cam: CameraPose | Mapping[str, float],
	depth: np.ndarray,
	crop: tuple[int, int, int, int] | None = None,
	full_shape: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
	"""World height per pixel and a validity mask from a ``distance_to_image_plane`` image (H, W) of camera ``cam``.

	With ``crop`` (x, y, width, height), ``depth`` is that crop of a ``full_shape`` (height, width) image."""
	pose = as_pose(cam)
	shape = full_shape if full_shape is not None else depth.shape
	ray_z = _ray_world_z_grid(pose, shape, crop=crop if full_shape is not None or crop is None else None)
	valid = np.isfinite(depth) & (depth > 0.05) & (depth < 5.0)
	world_z = pose.height + ray_z * np.where(valid, depth, 0.0)
	return world_z, valid


def height_to_depth_plane(
	cam: CameraPose | Mapping[str, float],
	z_world: np.ndarray,
	crop: tuple[int, int, int, int] | None = None,
	full_shape: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
	"""Convert world height ``z_world`` back to camera optical-axis depth ``distance_to_image_plane``."""
	pose = as_pose(cam)
	shape = full_shape if full_shape is not None else z_world.shape
	ray_z = _ray_world_z_grid(pose, shape, crop=crop if full_shape is not None or crop is None else None)
	valid = np.isfinite(z_world) & (ray_z < -1e-4)
	safe_rz = np.where(valid, ray_z, -1.0)
	depth_m = np.where(valid, (z_world - pose.height) / safe_rz, 0.0)
	valid &= (depth_m > 0.05) & (depth_m < 5.0)
	return depth_m, valid


def fit_table_pose(
	depth: np.ndarray,
	focal_px: float,
	crop: tuple[int, int, int, int] | None = None,
	full_shape: tuple[int, int] | None = None,
	seed: int = 0,
	iterations: int = 300,
) -> CameraPose:
	"""The camera's height, tilt and roll above the table, from a depth image alone (the table must fill a good part
	of it). RANSAC finds the largest plane, which is refitted on its inliers.

	Only the focal length is needed: heights do not depend on the camera's x, y or yaw, which are returned as 0, so
	``depth_plane_to_height`` with the result gives the height above the fitted table. This replaces fixed mounting
	constants on a real camera, and reproduces the simulator's true-camera heights to a fraction of a millimetre.
	``full_shape`` is (height, width); ``crop`` as in :func:`depth_plane_to_height`."""
	shape = full_shape if full_shape is not None else depth.shape
	x, y = _pixel_grid(shape, crop if full_shape is not None else None, focal_px)
	valid = np.isfinite(depth) & (depth > 0.05) & (depth < 5.0)
	pts = np.stack([x * depth, y * depth, depth], axis=-1)[valid]  # camera frame: right, down, forward
	if len(pts) < 500:
		raise ValueError(f"only {len(pts)} valid depth pixels")
	rng = np.random.default_rng(seed)
	sample = pts[rng.choice(len(pts), min(len(pts), 20_000), replace=False)]
	best, best_count = None, -1
	for _ in range(iterations):
		a, b, c = sample[rng.choice(len(sample), 3, replace=False)]
		n = np.cross(b - a, c - a)
		norm = np.linalg.norm(n)
		if norm < 1e-12:
			continue
		n /= norm
		count = int(np.count_nonzero(np.abs((sample - a) @ n) < 0.004))
		if count > best_count:
			best, best_count = (n, a), count
	n, a = best
	inliers = sample[np.abs((sample - a) @ n) < 0.004]
	for _ in range(3):
		centre = inliers.mean(axis=0)
		n = np.linalg.svd(inliers - centre, full_matrices=False)[2][2]
		inliers = sample[np.abs((sample - centre) @ n) < 0.002]
	d = -float(n @ centre)
	if d < 0:  # orient the normal towards the camera: then n . p + d is the height of camera point p
		n, d = -n, -d
	# n is world up in camera axes: right . up = sin(roll) sin(tilt), image-up . up = cos(roll) sin(tilt), forward . up = -cos(tilt)
	tilt = math.degrees(math.acos(float(np.clip(-n[2], -1.0, 1.0))))
	roll = math.degrees(math.atan2(float(n[0]), float(-n[1])))
	return CameraPose(0.0, 0.0, d, 0.0, tilt, roll, focal_px)
