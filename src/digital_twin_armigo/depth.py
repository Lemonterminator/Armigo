"""Height-above-table images in the depth bridge ``layers`` encoding (dataset key ``top_depth_v2``). Pure NumPy."""

from __future__ import annotations

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


def _ray_world_z_grid(
	pose: CameraPose,
	full_shape: tuple[int, int],
	crop: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
	"""Compute world Z component of camera viewing rays per pixel using full-frame principal point."""
	right, up, forward = pose.frame()
	full_h, full_w = full_shape
	if crop is None:
		u0, v0, wid, hgt = 0, 0, full_w, full_h
	else:
		u0, v0, wid, hgt = crop
	u, v = np.meshgrid(np.arange(u0, u0 + wid) + 0.5, np.arange(v0, v0 + hgt) + 0.5)
	f = pose.focal_px
	ray = ((u - full_w / 2.0) / f)[..., None] * right + (-(v - full_h / 2.0) / f)[..., None] * up + forward
	return ray[..., 2]


def depth_plane_to_height(
	cam: CameraPose | Mapping[str, float],
	depth: np.ndarray,
	crop: tuple[int, int, int, int] | None = None,
	full_shape: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
	"""World height per pixel and a validity mask from a ``distance_to_image_plane`` image (H, W) of camera ``cam``."""
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
