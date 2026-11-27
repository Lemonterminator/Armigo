"""Training labels for the brick segmentation network, computed from simulator truth (never used by the agent)."""

from __future__ import annotations

import numpy as np

from ..constants import SO101_MJCF_DIR

__all__ = ["TOP_XY", "_demo_cell", "brick_corners_world", "corner_labels"]

TOP_XY = ((1, 1), (-1, 1), (-1, -1), (1, -1))  # counter-clockwise seen from +z


def brick_corners_world(pos, rot, size) -> np.ndarray:
	"""Return all 8 brick corners ``(8, 3)`` in world coords; first 4 are +z top face CCW."""
	hx, hy, hz = (float(s) / 2.0 for s in size)
	local = [(sx * hx, sy * hy, hz) for sx, sy in TOP_XY] + [(sx * hx, sy * hy, -hz) for sx, sy in TOP_XY]
	return np.asarray(local, dtype=np.float64) @ np.asarray(rot, dtype=np.float64).T + np.asarray(pos, dtype=np.float64)


def corner_labels(corners_w, k, world_from_cam, depth, tol_m: float = 0.002) -> np.ndarray:
	"""Project ``corners_w`` to ``(8, 3)`` float32 ``(u, v, visible)`` checked against depth + ``tol_m``."""
	c = np.linalg.inv(np.asarray(world_from_cam, dtype=np.float64))
	p = np.asarray(corners_w, dtype=np.float64) @ c[:3, :3].T + c[:3, 3]
	k_arr = np.asarray(k, dtype=np.float64)
	uv = (p[:, :2] / np.where(p[:, 2:3] > 1e-9, p[:, 2:3], 1.0)) @ k_arr[:2, :2].T + k_arr[:2, 2]
	h, w = depth.shape
	out = np.zeros((len(corners_w), 3), dtype=np.float32)
	for i, ((u, v), z) in enumerate(zip(uv, p[:, 2], strict=True)):
		if not (np.isfinite(u) and np.isfinite(v) and np.isfinite(z)):
			continue
		ui, vi = round(float(u)), round(float(v))
		inside = z > 0 and 0 <= ui < w and 0 <= vi < h
		vis = float(inside and float(z) <= float(depth[vi, ui]) + tol_m)
		out[i] = (float(u), float(v), vis)
	return out


def _demo_cell(seed: int = 3_000_101, *, positive: bool = False, face: int = 0, split: str = "train"):
	"""Build a face-probe MuJoCo ``Cell`` for tests and dataset generation (mirrors ``TwinMujoco``)."""
	from ..agent.twin_mujoco import INITIAL_Q, sample_case
	from ..mujoco_cell.cell import Cell

	layout = sample_case(seed, positive=positive, face=face, options={"scene": "v3", "cameras": "wide", "split": split})
	cell = Cell(layout, SO101_MJCF_DIR, dynamic_clutter=True, assisted_grasp=False, face_probes=True)
	cell.set_joints(INITIAL_Q)
	return cell
