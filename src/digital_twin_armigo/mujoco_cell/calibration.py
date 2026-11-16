"""Fitting the simulated cell to the real recordings: top camera pose and the layout prior."""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..camera import CameraPose, look_from, table_point
from ..constants import BOWL_H, COARSE_MAX_M, PLANE_HEIGHT_M, PLANE_TILT_DEG, TOP_CROP
from ..depth import decode_height
from ..joints import normalized_to_rad
from .cell import Cell
from .layout import Layout

logger = logging.getLogger(__name__)


def _real_frames(dataset: Path, indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
	import av
	import pyarrow.parquet as pq

	table = pq.read_table(dataset / "data/chunk-000/file-000.parquet", columns=["observation.state", "index"])
	states = np.stack(table.column("observation.state").to_numpy(zero_copy_only=False))
	wanted = set(indices)
	frames = {}
	with av.open(str(dataset / "videos/observation.images.top_depth_v2/chunk-000/file-000.mp4")) as c:
		for i, f in enumerate(c.decode(video=0)):
			if i in wanted:
				frames[i] = f.to_ndarray(format="rgb24")
			if i >= max(indices):
				break
	return np.stack([frames[i] for i in indices]), states[indices]


def fit_camera(dataset: Path, menagerie: Path, indices: list[int], out: Path) -> CameraPose:
	"""Fit top camera x, y, yaw (then height, tilt, focal) so the rendered arm matches the real depth silhouettes."""
	from scipy.optimize import minimize

	real, states = _real_frames(dataset, indices)
	real_mask = real[..., 1] > int(255 * 0.04 / COARSE_MAX_M)
	layout = Layout(green=(0.0, 1.0), pink=(0.0, -1.0), brick_xy=(2.0, 2.0), brick_yaw_deg=0.0, defective=False)

	layout.top = look_from((0.2, 0.0), 0.25, PLANE_HEIGHT_M, 0.0)
	cell = Cell(layout, menagerie)

	def score(p: CameraPose) -> float:
		cell.set_camera("top", p)
		ious = []
		for k, s in enumerate(states):
			cell.set_joints(normalized_to_rad(s))
			z, valid = cell.height("top")
			x0, y0, w, h = TOP_CROP
			sim = (z[y0 : y0 + h, x0 : x0 + w] > 0.04) & valid[y0 : y0 + h, x0 : x0 + w]
			union = np.logical_or(sim, real_mask[k]).sum()
			ious.append(np.logical_and(sim, real_mask[k]).sum() / max(union, 1))
		return float(np.mean(ious))

	best = (-1.0, None)
	t0 = time.perf_counter()
	for yaw in range(0, 360, 15):
		for dist in (0.15, 0.25, 0.35):
			for tx in (0.1, 0.2, 0.3):
				for ty in (-0.15, 0.0, 0.15):
					p = look_from((tx, ty), dist, PLANE_HEIGHT_M, yaw)
					p.tilt_deg = PLANE_TILT_DEG
					s = score(p)
					if s > best[0]:
						best = (s, p)
	logger.info("grid best IoU %.3f at %s (%.0f s)", best[0], best[1], time.perf_counter() - t0)

	def loss(v: np.ndarray) -> float:
		return -score(CameraPose(v[0], v[1], v[2], v[3], v[4], v[5], v[6]))

	b = best[1]
	v0 = np.array([b.x, b.y, b.height, b.yaw_deg, b.tilt_deg, 0.0, b.focal_px])
	simplex = np.vstack([v0, v0 + np.diag([0.03, 0.03, 0.02, 6.0, 3.0, 2.0, 25.0])])
	res = minimize(loss, v0, method="Nelder-Mead", options={"maxfev": 250, "initial_simplex": simplex})
	pose = CameraPose(*res.x)
	result = {"camera": asdict(pose), "iou": -res.fun, "grid_iou": best[0], "frames": indices}
	out.parent.mkdir(parents=True, exist_ok=True)
	out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
	logger.info("refined IoU %.3f: %s", -res.fun, pose)
	return pose


def layout_prior(dataset: Path, camera_file: Path, out: Path) -> list[dict]:
	"""Bowl centres and the brick position of every real episode, measured on its first and last top frames.

	Bowls are the largest green / pink regions in ``top_rgb_v2``; their outline is projected onto the rim plane
	through the nominal top camera and fitted with a circle. The brick is where the height dropped between the
	first and last ``top_depth_v2`` frames outside the bowls (it was picked up), projected at its top face.
	Positions are only as good as the camera; rendering through the same camera puts them back where they were.
	"""
	import av
	import cv2
	import pyarrow.parquet as pq

	pose = CameraPose(**json.loads(camera_file.read_text())["camera"])
	ep = pq.read_table(dataset / "meta/episodes/chunk-000/file-000.parquet").to_pydict()
	starts, ends = list(ep["dataset_from_index"]), list(ep["dataset_to_index"])
	want = {s + 3: (e, "first") for e, s in enumerate(starts)} | {t - 3: (e, "last") for e, t in enumerate(ends)}
	frames: dict[tuple[str, int, str], np.ndarray] = {}
	for cam in ("top_rgb_v2", "top_depth_v2"):
		with av.open(str(dataset / f"videos/observation.images.{cam}/chunk-000/file-000.mp4")) as c:
			for i, f in enumerate(c.decode(video=0)):
				if i in want:
					frames[(cam, *want[i])] = f.to_ndarray(format="rgb24")
				if i >= max(want):
					break
	x0, y0 = TOP_CROP[:2]
	result = []
	for e in range(len(starts)):
		hsv = cv2.cvtColor(frames[("top_rgb_v2", e, "first")], cv2.COLOR_RGB2HSV)
		hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
		masks = {
			"green": (hue > 45) & (hue < 90) & (sat > 80),
			"pink": ((hue > 160) | (hue < 8)) & (sat > 40) & (sat < 160) & (val > 140),
		}
		entry: dict = {"episode": e}
		for name, mask in masks.items():
			m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
			n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
			if n < 2:
				continue
			k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
			hull = cv2.convexHull(cv2.findNonZero((lab == k).astype(np.uint8)))[:, 0]
			w = table_point(pose, hull[:, 0] + x0, hull[:, 1] + y0, BOWL_H)
			a = np.c_[2 * w, np.ones(len(w))]
			cx, cy, c0 = np.linalg.lstsq(a, (w**2).sum(1), rcond=None)[0]
			entry[name] = [float(cx), float(cy)]
			entry[f"{name}_r"] = float(math.sqrt(max(c0 + cx * cx + cy * cy, 0.0)))
		h0 = decode_height(frames[("top_depth_v2", e, "first")])
		h1 = decode_height(frames[("top_depth_v2", e, "last")])
		drop = (np.nan_to_num(h0 - h1) > 0.008) & (np.nan_to_num(h0) < 0.06)
		drop = cv2.morphologyEx(drop.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
		n, lab, stats, cen = cv2.connectedComponentsWithStats(drop)
		for k in np.argsort(-stats[1:, cv2.CC_STAT_AREA]) + 1:
			if not 300 < stats[k, cv2.CC_STAT_AREA] < 4000:
				continue
			xy = table_point(pose, np.array([cen[k][0] + x0]), np.array([cen[k][1] + y0]), 0.02)[0]
			inside = [np.hypot(*(xy - entry[b])) < entry[f"{b}_r"] for b in ("green", "pink") if b in entry]
			if not any(inside):
				entry["brick"] = [float(xy[0]), float(xy[1])]
				break
		result.append(entry)
		logger.info("episode %d: %s", e, entry)
	out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
	return result
