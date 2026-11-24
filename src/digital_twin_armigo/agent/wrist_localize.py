"""Wrist-camera brick localisation from RGB only (sensor side, no simulator truth).

The camera pose comes from the joint readings and the *designed* wrist mount (``Cell.nominal_wrist_cam``), never from
``model.cam_pos`` of the simulation, which holds the actual, possibly offset, mount. The brick is the colour blob nearest
to its predicted image position; the top-face rectangle's corners are intersected with the top-face plane."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import cv2
import numpy as np

from ..mujoco_cell._mj import mujoco
from .perception import BrickEstimate, yaw_error_deg

__all__ = ["FuseCfg", "WristFix", "believed_wrist_from_cam", "fuse", "localize_from_wrist", "wrist_k"]


@dataclass(frozen=True)
class WristFix:
	pos_xy: np.ndarray
	yaw: float
	area_px: int
	fill: float
	footprint_m: tuple[float, float]
	tilt_deg: float


@dataclass(frozen=True)
class FuseCfg:
	gate_xy_m: float = 0.04
	gate_yaw_deg: float = 30.0
	w_wrist: float = 1.0


def believed_wrist_from_cam(cell, kin_data) -> np.ndarray:
	"""World-from-camera (OpenCV) of the wrist camera on its designed mount at ``kin_data.qpos``."""
	m = cell.model
	cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
	mujoco.mj_kinematics(m, kin_data)
	with cell.nominal_wrist_cam(kin_data):
		mujoco.mj_camlight(m, kin_data)
		tf = np.eye(4)
		tf[:3, :3] = kin_data.cam_xmat[cid].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
		tf[:3, 3] = kin_data.cam_xpos[cid]
	return tf


def wrist_k(fovy_deg: float, shape) -> np.ndarray:
	h, w = shape[:2]
	f = 0.5 * h / math.tan(math.radians(fovy_deg) / 2)
	return np.array([[f, 0.0, w / 2 - 0.5], [0.0, f, h / 2 - 0.5], [0.0, 0.0, 1.0]])


def _project(k, world_from_cam, p):
	c = np.linalg.inv(world_from_cam)
	q = c[:3, :3] @ p + c[:3, 3]
	return (k @ (q / q[2]))[:2], float(q[2])


def _rays_to_plane(px, k, world_from_cam, z):
	d = np.c_[px, np.ones(len(px))] @ np.linalg.inv(k).T @ world_from_cam[:3, :3].T
	o = world_from_cam[:3, 3]
	return o + ((z - o[2]) / d[:, 2])[:, None] * d


def _blob(rgb, centre, gate_px, min_delta=35.0):
	lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
	h, w = rgb.shape[:2]
	yy, xx = np.mgrid[0:h, 0:w]
	r = np.hypot(xx - centre[0], yy - centre[1])
	ring = (r > 1.6 * gate_px) & (r < 2.4 * gate_px)
	if ring.sum() < 50:
		ring = r > 1.6 * gate_px
	if not ring.any():
		return None
	bg = np.median(lab[ring], axis=0)
	mask = ((np.linalg.norm(lab - bg, axis=-1) > min_delta) & (r < gate_px)).astype(np.uint8)
	mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
	n, labels, _, cent = cv2.connectedComponentsWithStats(mask)
	if n <= 1:
		return None
	i = 1 + int(np.argmin(np.hypot(cent[1:, 0] - centre[0], cent[1:, 1] - centre[1])))
	return (labels == i).astype(np.uint8)


def localize_from_wrist(rgb, k, world_from_cam, prior_xy, prior_yaw, size, top_z, *, gate_m=0.04, max_tilt_deg=50.0):
	axis = world_from_cam[:3, 2]
	tilt = math.degrees(math.acos(float(np.clip(-axis[2], -1.0, 1.0))))
	if tilt > max_tilt_deg:
		return None, "TILTED"
	centre, depth = _project(k, world_from_cam, np.array([prior_xy[0], prior_xy[1], top_z]))
	if depth <= 1e-4:
		return None, "NO_BLOB"
	mask = _blob(rgb, centre, gate_m * k[0, 0] / depth)
	if mask is None:
		return None, "NO_BLOB"
	if mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
		return None, "CLIPPED"
	contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	contour = max(contours, key=cv2.contourArea)
	rect = cv2.minAreaRect(contour)
	corners = _rays_to_plane(cv2.boxPoints(rect), k, world_from_cam, top_z)
	edges = np.roll(corners, -1, axis=0) - corners
	lengths = np.linalg.norm(edges[:, :2], axis=1)
	long_i = int(np.argmax(lengths[:2]))
	footprint = (float(lengths[long_i]), float(lengths[1 - long_i]))
	want = sorted(size[:2], reverse=True)
	if abs(footprint[0] - want[0]) > 0.3 * want[0] or abs(footprint[1] - want[1]) > 0.3 * want[1]:
		return None, "BAD_FOOTPRINT"
	yaw = math.atan2(edges[long_i, 1], edges[long_i, 0])
	period = math.pi / 2 if abs(size[0] - size[1]) < 1e-6 else math.pi
	yaw += period * round((prior_yaw - yaw) / period)
	area = float(cv2.contourArea(contour))
	box = max(rect[1][0] * rect[1][1], 1.0)
	return WristFix(corners.mean(axis=0)[:2], yaw, int(mask.sum()), area / box, footprint, tilt), None


def fuse(prior: BrickEstimate, fix: WristFix, cfg: FuseCfg):
	d_xy = fix.pos_xy - prior.pos[:2]
	d_yaw_deg = yaw_error_deg(fix.yaw, prior.yaw, prior.size)
	info = {
		"innovation_xy_mm": round(float(np.linalg.norm(d_xy) * 1000), 3),
		"innovation_yaw_deg": round(d_yaw_deg, 3),
		"accepted": False,
	}
	if np.linalg.norm(d_xy) > cfg.gate_xy_m or d_yaw_deg > cfg.gate_yaw_deg:
		return None, info
	info["accepted"] = True
	pos = prior.pos.copy()
	pos[:2] += cfg.w_wrist * d_xy
	yaw = prior.yaw + cfg.w_wrist * (fix.yaw - prior.yaw)
	return replace(prior, pos=pos, yaw=yaw), info
