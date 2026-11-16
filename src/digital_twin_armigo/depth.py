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


def depth_plane_to_height(cam: CameraPose | Mapping[str, float], depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	"""World height per pixel and a validity mask from a ``distance_to_image_plane`` image (H, W) of camera ``cam``."""
	pose = as_pose(cam)
	right, up, forward = pose.frame()
	hgt, wid = depth.shape
	u, v = np.meshgrid(np.arange(wid) + 0.5, np.arange(hgt) + 0.5)
	f = pose.focal_px
	ray = ((u - wid / 2.0) / f)[..., None] * right + (-(v - hgt / 2.0) / f)[..., None] * up + forward
	valid = np.isfinite(depth) & (depth > 0.05) & (depth < 5.0)
	world_z = pose.height + ray[..., 2] * np.where(valid, depth, 0.0)
	return world_z, valid
