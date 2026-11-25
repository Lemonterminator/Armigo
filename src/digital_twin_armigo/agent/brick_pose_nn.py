"""ONNX wrist/top brick-face segmentation inference (OpenCV DNN, no torch) and symmetric PnP pose recovery."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

__all__ = [
	"BrickSegNet",
	"NetOut",
	"PoseFit",
	"localize_with_net",
	"peaks",
	"pose_from_keypoints",
]

CLS_BG, CLS_TOP, CLS_BOTTOM, CLS_SIDE = 0, 1, 2, 3
_TOP_XY = ((1, 1), (-1, 1), (-1, -1), (1, -1))


def _top_corners_local(size: tuple[float, float, float]) -> np.ndarray:
	hx, hy, hz = (float(s) / 2.0 for s in size)
	return np.array([(sx * hx, sy * hy, hz) for sx, sy in _TOP_XY], dtype=np.float64)


def _all_corners_local(size: tuple[float, float, float]) -> np.ndarray:
	hx, hy, hz = (float(s) / 2.0 for s in size)
	pts = [(sx * hx, sy * hy, hz) for sx, sy in _TOP_XY] + [(sx * hx, sy * hy, -hz) for sx, sy in _TOP_XY]
	return np.array(pts, dtype=np.float64)


@dataclass
class NetOut:
	cls: np.ndarray  # (H_in, W_in) int64
	heat: np.ndarray  # (2, H_in, W_in) float32 sigmoid probabilities
	scale: tuple[float, float]  # (sx, sy) mapping network coords -> original image coords


@dataclass(frozen=True)
class PoseFit:
	pos: np.ndarray  # (3,) world frame
	rot: np.ndarray  # (3, 3) world frame
	yaw: float
	tilt_deg: float
	reproj_px: float
	n_points: int
	mode: str


class BrickSegNet:
	"""CPU inference wrapper using ``cv2.dnn.readNetFromONNX``."""

	def __init__(self, onnx_path: Path, meta_path: Path | None = None) -> None:
		onnx_path = Path(onnx_path)
		if meta_path is None:
			meta_path = onnx_path.with_suffix(".json")
		self.meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
		self.input_hw = tuple(int(v) for v in self.meta.get("input_hw", (240, 320)))
		self.mean = np.asarray(self.meta.get("mean", (0.485, 0.456, 0.406)), dtype=np.float32)
		self.std = np.asarray(self.meta.get("std", (0.229, 0.224, 0.225)), dtype=np.float32)
		self.net = cv2.dnn.readNetFromONNX(str(onnx_path))

	def predict(self, rgb: np.ndarray) -> NetOut:
		orig_h, orig_w = rgb.shape[:2]
		in_h, in_w = self.input_hw
		resized = cv2.resize(rgb, (in_w, in_h), interpolation=cv2.INTER_AREA)
		x = (resized.astype(np.float32) / 255.0 - self.mean) / self.std
		blob = np.ascontiguousarray(x.transpose(2, 0, 1)[None, ...], dtype=np.float32)
		self.net.setInput(blob)
		logits = self.net.forward()[0]  # (6, in_h, in_w)
		cls = np.argmax(logits[:4], axis=0).astype(np.int64)
		heat = (1.0 / (1.0 + np.exp(-np.clip(logits[4:6], -60.0, 60.0)))).astype(np.float32)
		return NetOut(cls=cls, heat=heat, scale=(orig_w / float(in_w), orig_h / float(in_h)))


def peaks(heat: np.ndarray, thr: float = 0.3, radius: int = 2, scale: tuple[float, float] = (1.0, 1.0)) -> np.ndarray:
	"""Find local maxima >= ``thr`` in a 2D heatmap with 3x3 weighted sub-pixel refinement."""
	h_arr = np.asarray(heat, dtype=np.float32)
	if h_arr.ndim != 2:
		raise ValueError(f"heat must be 2D, got shape {h_arr.shape}")
	ksize = 2 * int(radius) + 1
	dilated = cv2.dilate(h_arr, np.ones((ksize, ksize), dtype=np.uint8))
	is_peak = (h_arr >= float(thr)) & (h_arr == dilated) & (h_arr > 0)
	ys, xs = np.nonzero(is_peak)
	if not len(xs):
		return np.zeros((0, 3), dtype=np.float32)
	# Keep strict peaks first or break flat plateaus cleanly by coordinate scan after sorting descending by value
	order = np.argsort(-h_arr[ys, xs], kind="stable")
	kept: list[tuple[float, float, float]] = []
	suppressed = np.zeros_like(is_peak, dtype=bool)
	h_img, w_img = h_arr.shape
	for idx in order:
		py, px = int(ys[idx]), int(xs[idx])
		if suppressed[py, px]:
			continue
		y0, y1 = max(0, py - radius), min(h_img, py + radius + 1)
		x0, x1 = max(0, px - radius), min(w_img, px + radius + 1)
		suppressed[y0:y1, x0:x1] = True
		wy0, wy1 = max(0, py - 1), min(h_img, py + 2)
		wx0, wx1 = max(0, px - 1), min(w_img, px + 2)
		patch = h_arr[wy0:wy1, wx0:wx1].astype(np.float64)
		weights = np.maximum(patch - float(patch.min()), 0.0)
		wsum = float(weights.sum())
		if wsum > 1e-9:
			yy, xx = np.mgrid[wy0:wy1, wx0:wx1]
			u = float((xx * weights).sum() / wsum)
			v = float((yy * weights).sum() / wsum)
		else:
			u, v = float(px), float(py)
		kept.append((u * float(scale[0]), v * float(scale[1]), float(h_arr[py, px])))
	return np.asarray(kept, dtype=np.float32)


def _cyclic_sort_quad(pts: np.ndarray) -> np.ndarray:
	c = pts.mean(axis=0)
	ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
	return pts[np.argsort(ang)]


def _mask_to_quad(mask: np.ndarray) -> np.ndarray | None:
	m = (np.asarray(mask) > 0).astype(np.uint8)
	contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	if not contours:
		return None
	contour = max(contours, key=cv2.contourArea)
	if cv2.contourArea(contour) < 16.0:
		return None
	perim = cv2.arcLength(contour, True)
	for frac in np.linspace(0.01, 0.15, 29):
		approx = cv2.approxPolyDP(contour, frac * perim, True)
		if len(approx) == 4:
			return approx.reshape(4, 2).astype(np.float64)
	rect = cv2.minAreaRect(contour)
	return cv2.boxPoints(rect).astype(np.float64)


def _reproj_err(obj_pts: np.ndarray, img_pts: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, k: np.ndarray) -> float:
	proj, _ = cv2.projectPoints(obj_pts.astype(np.float64), rvec, tvec, k.astype(np.float64), None)
	err = np.linalg.norm(proj.reshape(-1, 2) - img_pts.astype(np.float64), axis=1)
	return float(np.sqrt(np.mean(err**2)))


def _solve_planar_hypotheses(
	quad_px: np.ndarray,
	k: np.ndarray,
	world_from_cam: np.ndarray,
	size: tuple[float, float, float],
	prior_yaw: float,
	mode: str,
	*,
	max_tilt_deg: float = 60.0,
) -> tuple[PoseFit | None, str | None]:
	obj_top = _top_corners_local(size)
	hz = float(size[2]) / 2.0
	obj_plane = obj_top.copy()
	obj_plane[:, 2] = 0.0
	ordered = _cyclic_sort_quad(np.asarray(quad_px, dtype=np.float64))
	k_f64 = np.asarray(k, dtype=np.float64)
	w2c = np.asarray(world_from_cam, dtype=np.float64)
	cands: list[tuple[float, float, PoseFit]] = []

	for seq in (ordered, ordered[::-1]):
		for shift in range(4):
			img_pts = np.ascontiguousarray(np.roll(seq, shift, axis=0), dtype=np.float64)
			for flag in (cv2.SOLVEPNP_IPPE, cv2.SOLVEPNP_SQPNP):
				ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj_plane, img_pts, k_f64, None, flags=flag)
				if not ok:
					continue
				for rvec, tvec in zip(rvecs, tvecs, strict=True):
					t_cam = np.asarray(tvec, dtype=np.float64).reshape(3)
					if t_cam[2] <= 1e-4:
						continue
					r_cam, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
					tf_cam = np.eye(4)
					tf_cam[:3, :3] = r_cam
					tf_cam[:3, 3] = t_cam
					tf_w = w2c @ tf_cam
					rot_w = tf_w[:3, :3]
					tilt = math.degrees(math.acos(float(np.clip(rot_w[2, 2], -1.0, 1.0))))
					if tilt > max_tilt_deg:
						continue
					err = _reproj_err(obj_plane, img_pts, rvec, tvec, k_f64)
					pos_w = tf_w[:3, 3] - rot_w[:, 2] * hz
					raw_yaw = math.atan2(rot_w[1, 0], rot_w[0, 0])
					yaw = raw_yaw + 2.0 * math.pi * round((prior_yaw - raw_yaw) / (2.0 * math.pi))
					yaw_dist = abs(yaw - prior_yaw)
					cands.append((err, yaw_dist, PoseFit(pos_w.copy(), rot_w.copy(), yaw, tilt, err, 4, mode)))

	if not cands:
		return None, "NO_POSE"
	min_err = min(c[0] for c in cands)
	if min_err > 2.5:
		return None, "BAD_REPROJ"
	near_best = [c for c in cands if c[0] <= min_err + 0.5]
	best = min(near_best, key=lambda c: (c[1], c[0]))[2]
	return best, None


def pose_from_keypoints(
	top_px: np.ndarray,
	bottom_px: np.ndarray,
	top_mask: np.ndarray | None,
	k: np.ndarray,
	world_from_cam: np.ndarray,
	size: tuple[float, float, float],
	prior_yaw: float,
) -> tuple[PoseFit | None, str | None]:
	top_arr = np.asarray(top_px, dtype=np.float64)[:, :2] if len(top_px) else np.zeros((0, 2), dtype=np.float64)
	bot_arr = np.asarray(bottom_px, dtype=np.float64)[:, :2] if len(bottom_px) else np.zeros((0, 2), dtype=np.float64)

	if len(bot_arr) >= 2 and len(top_arr) >= 4:
		fit6 = _solve_6d_ransac(top_arr[:4], bot_arr[:4], k, world_from_cam, size, prior_yaw)
		if fit6 is not None and fit6.reproj_px <= 2.5:
			return fit6, None

	if len(top_arr) >= 4:
		return _solve_planar_hypotheses(top_arr[:4], k, world_from_cam, size, prior_yaw, "ippe4")

	if top_mask is not None:
		quad = _mask_to_quad(top_mask)
		if quad is not None:
			return _solve_planar_hypotheses(quad, k, world_from_cam, size, prior_yaw, "mask4")
	return None, "NO_CORNERS"


def _solve_6d_ransac(
	top4: np.ndarray,
	bot_pts: np.ndarray,
	k: np.ndarray,
	world_from_cam: np.ndarray,
	size: tuple[float, float, float],
	prior_yaw: float,
) -> PoseFit | None:
	planar, _ = _solve_planar_hypotheses(top4, k, world_from_cam, size, prior_yaw, "ippe4", max_tilt_deg=179.0)
	if planar is None:
		return None
	all_obj = _all_corners_local(size)
	# Match bottom image points greedily to projections of local bottom corners under the planar init
	w2c = np.asarray(world_from_cam, dtype=np.float64)
	c2w = np.linalg.inv(w2c)
	tf_w = np.eye(4)
	tf_w[:3, :3] = planar.rot
	tf_w[:3, 3] = planar.pos
	tf_cam = c2w @ tf_w
	bot_cam = all_obj[4:] @ tf_cam[:3, :3].T + tf_cam[:3, 3]
	if (bot_cam[:, 2] <= 1e-4).any():
		return None
	bot_proj = (bot_cam[:, :2] / bot_cam[:, 2:3]) @ k[:2, :2].T + k[:2, 2]
	used = set()
	obj_list = list(all_obj[:4])
	# Reorder top4 to best match planar projection
	top_cam = all_obj[:4] @ tf_cam[:3, :3].T + tf_cam[:3, 3]
	top_proj = (top_cam[:, :2] / top_cam[:, 2:3]) @ k[:2, :2].T + k[:2, 2]
	img_top_matched = []
	avail = list(range(len(top4)))
	for p in top_proj:
		j = min(avail, key=lambda idx: float(np.linalg.norm(top4[idx] - p)))
		avail.remove(j)
		img_top_matched.append(top4[j])
	img_list = img_top_matched
	for bp in bot_pts:
		j = min(
			(i for i in range(4) if i not in used),
			key=lambda idx: float(np.linalg.norm(bot_proj[idx] - bp)),
			default=-1,
		)
		if j < 0:
			break
		used.add(j)
		obj_list.append(all_obj[4 + j])
		img_list.append(bp)
	if len(obj_list) < 6:
		return None
	obj_arr = np.ascontiguousarray(obj_list, dtype=np.float64)
	img_arr = np.ascontiguousarray(img_list, dtype=np.float64)
	ok, rvec, tvec, _ = cv2.solvePnPRansac(
		obj_arr, img_arr, np.asarray(k, dtype=np.float64), None, reprojectionError=2.5
	)
	if not ok:
		return None
	r_cam, _ = cv2.Rodrigues(rvec)
	tf_c = np.eye(4)
	tf_c[:3, :3] = r_cam
	tf_c[:3, 3] = tvec.reshape(3)
	tf_w_out = w2c @ tf_c
	rot_w = tf_w_out[:3, :3]
	tilt = math.degrees(math.acos(float(np.clip(rot_w[2, 2], -1.0, 1.0))))
	raw_yaw = math.atan2(rot_w[1, 0], rot_w[0, 0])
	yaw = raw_yaw + 2.0 * math.pi * round((prior_yaw - raw_yaw) / (2.0 * math.pi))
	err = _reproj_err(obj_arr, img_arr, rvec, tvec, k)
	return PoseFit(tf_w_out[:3, 3].copy(), rot_w.copy(), yaw, tilt, err, len(obj_arr), "ransac6d")


def _nearest_top_component(cls_map: np.ndarray, centre_net: np.ndarray, gate_net_px: float) -> np.ndarray | None:
	top_mask = (cls_map == CLS_TOP).astype(np.uint8)
	n, labels, stats, cent = cv2.connectedComponentsWithStats(top_mask)
	if n <= 1:
		return None
	best_i, best_d = -1, float("inf")
	for i in range(1, n):
		if stats[i, cv2.CC_STAT_AREA] < 10:
			continue
		d = float(np.hypot(cent[i, 0] - centre_net[0], cent[i, 1] - centre_net[1]))
		if d <= gate_net_px and d < best_d:
			best_i, best_d = i, d
	if best_i < 0:
		return None
	return (labels == best_i).astype(np.uint8)


def localize_with_net(
	net: BrickSegNet,
	rgb: np.ndarray,
	k: np.ndarray,
	world_from_cam: np.ndarray,
	prior_xy,
	prior_yaw: float,
	size: tuple[float, float, float],
	top_z: float | None = None,
	*,
	gate_m: float = 0.04,
	max_tilt_deg: float = 50.0,
):
	from .wrist_localize import WristFix, _project

	axis = world_from_cam[:3, 2]
	tilt_cam = math.degrees(math.acos(float(np.clip(-axis[2], -1.0, 1.0))))
	if tilt_cam > max_tilt_deg:
		return None, "TILTED"
	z_ref = float(size[2]) if top_z is None else float(top_z)
	centre, depth = _project(k, world_from_cam, np.array([prior_xy[0], prior_xy[1], z_ref]))
	if depth <= 1e-4:
		return None, "NO_BLOB"

	out = net.predict(rgb)
	sx, sy = out.scale
	gate_px = gate_m * float(k[0, 0]) / depth
	centre_net = np.array([centre[0] / sx, centre[1] / sy])
	gate_net = gate_px / max(sx, sy)
	top_comp_net = _nearest_top_component(out.cls, centre_net, gate_net)
	if top_comp_net is None:
		return None, "NO_BLOB"

	dilated = cv2.dilate(top_comp_net, np.ones((7, 7), np.uint8))  # 3 px neighborhood in 320x240
	h_img, w_img = rgb.shape[:2]
	top_comp_full = cv2.resize(top_comp_net, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
	if top_comp_full[0].any() or top_comp_full[-1].any() or top_comp_full[:, 0].any() or top_comp_full[:, -1].any():
		return None, "CLIPPED"

	top_pks = peaks(out.heat[0], thr=0.3, radius=2, scale=out.scale)
	if len(top_pks):
		keep = []
		for u, v, _ in top_pks:
			nx, ny = round(u / sx), round(v / sy)
			in_net = 0 <= ny < dilated.shape[0] and 0 <= nx < dilated.shape[1] and bool(dilated[ny, nx])
			in_gate = math.hypot(u - centre[0], v - centre[1]) <= gate_px
			keep.append(in_net and in_gate)
		top_pks = top_pks[np.asarray(keep, dtype=bool)]

	bot_pks = peaks(out.heat[1], thr=0.3, radius=2, scale=out.scale)
	if len(bot_pks):
		keep_b = [math.hypot(u - centre[0], v - centre[1]) <= gate_px * 1.2 for u, v, _ in bot_pks]
		bot_pks = bot_pks[np.asarray(keep_b, dtype=bool)]

	fit, why = pose_from_keypoints(top_pks, bot_pks, top_comp_full, k, world_from_cam, size, prior_yaw)
	if fit is None:
		return None, why
	if float(np.linalg.norm(fit.pos[:2] - np.asarray(prior_xy, dtype=np.float64)[:2])) > gate_m:
		return None, "OUT_OF_GATE"

	area_px = int(top_comp_full.sum())
	contours, _ = cv2.findContours(top_comp_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	box = cv2.minAreaRect(max(contours, key=cv2.contourArea))[1] if contours else (1.0, 1.0)
	fill = float(area_px / max(box[0] * box[1], 1.0))
	return (
		WristFix(
			pos_xy=fit.pos[:2].copy(),
			yaw=float(fit.yaw),
			area_px=area_px,
			fill=fill,
			footprint_m=(float(size[0]), float(size[1])),
			tilt_deg=float(tilt_cam),
			z=float(fit.pos[2]),
			tilt_brick_deg=float(fit.tilt_deg),
			reproj_px=float(fit.reproj_px),
		),
		None,
	)
