#!/usr/bin/env python
"""Offline CPU evaluation of lab vs unet-rect vs unet-pnp on U1 val and ood-appearance shards.

Plan: docs/superpowers/plans/2026-11-24-failure-observability-harness-learning.md Task U4
Preregistered pass gate: docs/experiments/brickseg_prereg.md
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
	sys.path.insert(0, str(ROOT / "src"))

from digital_twin_armigo.agent.brick_pose_nn import (
	BrickSegNet,
	NetOut,
	PoseFit,
	_nearest_top_component,
	peaks,
	pose_from_keypoints,
)
from digital_twin_armigo.agent.perception import yaw_error_deg
from digital_twin_armigo.agent.wrist_localize import (
	WristFix,
	_project,
	_rays_to_plane,
	localize_from_wrist,
)

DEFAULT_DATA = Path("E:/Armigo_data_v3/perception/brickseg-v1")
METHODS = ("lab", "unet-rect", "unet-pnp")

PREREG_THRESHOLDS = {
	"sim_val_top_iou_min": 0.90,
	"sim_val_pck_2px_min": 0.90,
	"sim_val_wrist_xy_med_max_mm": 1.5,
	"sim_val_wrist_yaw_med_max_deg": 1.5,
	"sim_val_wrist_rate_min_pct": 98.0,
}


def _rot_z(yaw: float) -> np.ndarray:
	c, s = math.cos(yaw), math.sin(yaw)
	return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _table_sat_bin(sat: float | None) -> str:
	if sat is None:
		return "unknown"
	s = float(sat)
	if s < 0.15:
		return "low"
	if s <= 0.35:
		return "mid"
	return "high"


def held_pose_error_in_gripper(
	pred_pos: np.ndarray,
	pred_rot: np.ndarray,
	true_pos: np.ndarray,
	true_rot: np.ndarray,
	gripper_from_world: np.ndarray,
	size: tuple[float, float, float],
) -> tuple[float, float, np.ndarray]:
	"""Compute relative position error (mm), symmetry-aware 3D orientation error (deg), and xyz error vector (mm)
	of a predicted brick pose against ground truth in the gripper coordinate frame.
	"""
	g2w = np.asarray(gripper_from_world, dtype=np.float64)
	r_gw = g2w[:3, :3]
	t_gw = g2w[:3, 3]

	p_g_true = r_gw @ np.asarray(true_pos, dtype=np.float64) + t_gw
	p_g_pred = r_gw @ np.asarray(pred_pos, dtype=np.float64) + t_gw
	err_vec_mm = (p_g_pred - p_g_true) * 1000.0
	pos_mm = float(np.linalg.norm(err_vec_mm))

	r_g_true = r_gw @ np.asarray(true_rot, dtype=np.float64)
	r_g_pred = r_gw @ np.asarray(pred_rot, dtype=np.float64)

	sym_angles = (0.0, math.pi / 2.0, math.pi, 1.5 * math.pi) if abs(size[0] - size[1]) < 1e-6 else (0.0, math.pi)
	best_deg = 180.0
	for ang in sym_angles:
		r_rel = (r_g_true @ _rot_z(ang)).T @ r_g_pred
		cos_th = float(np.clip((float(np.trace(r_rel)) - 1.0) / 2.0, -1.0, 1.0))
		deg = math.degrees(math.acos(cos_th))
		best_deg = min(best_deg, deg)
	return pos_mm, best_deg, err_vec_mm


def _rect_from_mask(
	mask_full: np.ndarray,
	k: np.ndarray,
	w2c: np.ndarray,
	prior_yaw: float,
	size: tuple[float, float, float],
	top_z: float,
	tilt_cam: float,
) -> tuple[WristFix | None, str | None]:
	if mask_full is None or not mask_full.any():
		return None, "NO_BLOB"
	if mask_full[0].any() or mask_full[-1].any() or mask_full[:, 0].any() or mask_full[:, -1].any():
		return None, "CLIPPED"
	contours, _ = cv2.findContours(mask_full.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	if not contours:
		return None, "NO_BLOB"
	contour = max(contours, key=cv2.contourArea)
	rect = cv2.minAreaRect(contour)
	corners = _rays_to_plane(cv2.boxPoints(rect), k, w2c, top_z)
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
	return WristFix(corners.mean(axis=0)[:2], yaw, int(mask_full.sum()), area / box, footprint, tilt_cam), None


def _unet_rect_from_net_out(
	out: NetOut,
	rgb_hw: tuple[int, int],
	k: np.ndarray,
	w2c: np.ndarray,
	prior_xy: np.ndarray,
	prior_yaw: float,
	size: tuple[float, float, float],
	top_z: float,
	*,
	gate_m: float = 0.04,
	max_tilt_deg: float = 50.0,
) -> tuple[WristFix | None, str | None]:
	axis = w2c[:3, 2]
	tilt_cam = math.degrees(math.acos(float(np.clip(-axis[2], -1.0, 1.0))))
	if tilt_cam > max_tilt_deg:
		return None, "TILTED"
	centre, depth = _project(k, w2c, np.array([prior_xy[0], prior_xy[1], top_z]))
	if depth <= 1e-4:
		return None, "NO_BLOB"
	sx, sy = out.scale
	gate_px = gate_m * float(k[0, 0]) / depth
	comp_net = _nearest_top_component(out.cls, np.array([centre[0] / sx, centre[1] / sy]), gate_px / max(sx, sy))
	if comp_net is None:
		return None, "NO_BLOB"
	h_img, w_img = rgb_hw
	mask_full = cv2.resize(comp_net, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
	return _rect_from_mask(mask_full, k, w2c, prior_yaw, size, top_z, tilt_cam)


def _unet_rect_localize(
	net: BrickSegNet,
	rgb: np.ndarray,
	k: np.ndarray,
	w2c: np.ndarray,
	prior_xy: np.ndarray,
	prior_yaw: float,
	size: tuple[float, float, float],
	top_z: float,
	*,
	gate_m: float = 0.04,
	max_tilt_deg: float = 50.0,
) -> tuple[WristFix | None, str | None]:
	return _unet_rect_from_net_out(
		net.predict(rgb),
		rgb.shape[:2],
		k,
		w2c,
		prior_xy,
		prior_yaw,
		size,
		top_z,
		gate_m=gate_m,
		max_tilt_deg=max_tilt_deg,
	)


def _nearest_any_brick_component(cls_map: np.ndarray, centre_net: np.ndarray, gate_net_px: float) -> np.ndarray | None:
	brick_mask = (cls_map > 0).astype(np.uint8)
	n, labels, stats, cent = cv2.connectedComponentsWithStats(brick_mask)
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


def _unet_pnp_from_net_out(
	out: NetOut,
	rgb_hw: tuple[int, int],
	k: np.ndarray,
	w2c: np.ndarray,
	prior_xy: np.ndarray,
	prior_yaw: float,
	size: tuple[float, float, float],
	top_z: float,
	*,
	gate_m: float = 0.04,
	max_tilt_deg: float = 50.0,
	allow_3d_fallback: bool = False,
) -> tuple[PoseFit | None, str | None]:
	"""Recover brick 6-DoF ``PoseFit`` via ``pose_from_keypoints`` using precomputed ``NetOut``.

	Matches ``brick_pose_nn.localize_with_net`` exactly when ``allow_3d_fallback=False`` (wrist views),
	and supports top/flipped/held views via keypoint-only 6-DoF PnP fallback when ``allow_3d_fallback=True``.
	"""
	axis = w2c[:3, 2]
	tilt_cam = math.degrees(math.acos(float(np.clip(-axis[2], -1.0, 1.0))))
	if tilt_cam > max_tilt_deg:
		return None, "TILTED"
	centre, depth = _project(k, w2c, np.array([prior_xy[0], prior_xy[1], top_z]))
	if depth <= 1e-4:
		return None, "NO_BLOB"

	sx, sy = out.scale
	gate_px = gate_m * float(k[0, 0]) / depth
	centre_net = np.array([centre[0] / sx, centre[1] / sy])
	gate_net = gate_px / max(sx, sy)
	h_img, w_img = rgb_hw

	top_comp_net = _nearest_top_component(out.cls, centre_net, gate_net)
	if top_comp_net is not None:
		dilated = cv2.dilate(top_comp_net, np.ones((7, 7), np.uint8))
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

		fit, why = pose_from_keypoints(top_pks, bot_pks, top_comp_full, k, w2c, size, prior_yaw)
	elif allow_3d_fallback:
		any_comp = _nearest_any_brick_component(out.cls, centre_net, gate_net)
		if any_comp is None:
			return None, "NO_BLOB"
		any_full = cv2.resize(any_comp, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
		if any_full[0].any() or any_full[-1].any() or any_full[:, 0].any() or any_full[:, -1].any():
			return None, "CLIPPED"
		top_pks = peaks(out.heat[0], thr=0.3, radius=2, scale=out.scale)
		if len(top_pks):
			keep_t = [math.hypot(u - centre[0], v - centre[1]) <= gate_px * 1.5 for u, v, _ in top_pks]
			top_pks = top_pks[np.asarray(keep_t, dtype=bool)]
		bot_pks = peaks(out.heat[1], thr=0.3, radius=2, scale=out.scale)
		if len(bot_pks):
			keep_b = [math.hypot(u - centre[0], v - centre[1]) <= gate_px * 1.5 for u, v, _ in bot_pks]
			bot_pks = bot_pks[np.asarray(keep_b, dtype=bool)]
		if len(top_pks) < 4:
			return None, "TOO_FEW_KEYPOINTS"
		fit, why = pose_from_keypoints(top_pks, bot_pks, None, k, w2c, size, prior_yaw)
	else:
		return None, "NO_BLOB"

	if fit is None:
		return None, why or "PNP_FAILED"
	if float(np.linalg.norm(fit.pos[:2] - np.asarray(prior_xy, dtype=np.float64)[:2])) > gate_m:
		return None, "OUT_OF_GATE"
	return fit, None


def _run_method(
	method: str,
	net_out: NetOut,
	rgb: np.ndarray,
	k: np.ndarray,
	w2c: np.ndarray,
	prior_xy: np.ndarray,
	prior_yaw: float,
	size: tuple[float, float, float],
	top_z: float,
	*,
	is_top_or_held: bool,
) -> tuple[np.ndarray | None, float | None, np.ndarray | None, np.ndarray | None, str | None]:
	if method == "lab":
		fix, why = localize_from_wrist(rgb, k, w2c, prior_xy, prior_yaw, size, top_z, segmenter="lab")
		if fix is None:
			return None, None, None, None, why or "NO_BLOB"
		pred_pos = np.array([float(fix.pos_xy[0]), float(fix.pos_xy[1]), top_z - size[2] / 2.0], dtype=np.float64)
		return fix.pos_xy.copy(), float(fix.yaw), pred_pos, _rot_z(float(fix.yaw)), None
	if method == "unet-rect":
		fix, why = _unet_rect_from_net_out(net_out, rgb.shape[:2], k, w2c, prior_xy, prior_yaw, size, top_z)
		if fix is None:
			return None, None, None, None, why or "NO_BLOB"
		pred_pos = np.array([float(fix.pos_xy[0]), float(fix.pos_xy[1]), top_z - size[2] / 2.0], dtype=np.float64)
		return fix.pos_xy.copy(), float(fix.yaw), pred_pos, _rot_z(float(fix.yaw)), None
	fit, why = _unet_pnp_from_net_out(
		net_out,
		rgb.shape[:2],
		k,
		w2c,
		prior_xy,
		prior_yaw,
		size,
		top_z,
		gate_m=0.06 if is_top_or_held else 0.04,
		allow_3d_fallback=is_top_or_held,
	)
	if fit is None:
		return None, None, None, None, why or "PNP_FAILED"
	return fit.pos[:2].copy(), float(fit.yaw), fit.pos.copy(), fit.rot.copy(), None


def _median_p90(vals: list[float]) -> tuple[float | None, float | None]:
	if not vals:
		return None, None
	s = sorted(vals)
	p90 = s[min(len(s) - 1, int(0.9 * (len(s) - 1)))]
	return round(statistics.median(s), 3), round(p90, 3)


def _view_kind_group(view_kind: str, camera: str) -> str:
	if view_kind.startswith("wrist"):
		return "wrist"
	if view_kind in ("top_table", "top_present"):
		return view_kind
	return camera


def _aggregate_pose_bucket(recs: list[dict]) -> dict:
	n = len(recs)
	found = [r for r in recs if r["found"]]
	med_xy, p90_xy = _median_p90([float(r["xy_mm"]) for r in found if r["xy_mm"] is not None])
	med_y, p90_y = _median_p90([float(r["yaw_deg"]) for r in found if r["yaw_deg"] is not None])
	return {
		"n": n,
		"found": len(found),
		"rate_pct": round(100.0 * len(found) / n, 2) if n else 0.0,
		"xy_med_mm": med_xy,
		"xy_p90_mm": p90_xy,
		"yaw_med_deg": med_y,
		"yaw_p90_deg": p90_y,
	}


def _aggregate_held_bucket(recs: list[dict]) -> dict:
	n = len(recs)
	found = [r for r in recs if r["found"] and r["held_pos_mm"] is not None]
	med_pos, p90_pos = _median_p90([float(r["held_pos_mm"]) for r in found])
	med_rot, p90_rot = _median_p90([float(r["held_rot_deg"]) for r in found if r["held_rot_deg"] is not None])
	return {
		"n": n,
		"found": len(found),
		"rate_pct": round(100.0 * len(found) / n, 2) if n else 0.0,
		"held_pos_med_mm": med_pos,
		"held_pos_p90_mm": p90_pos,
		"held_rot_med_deg": med_rot,
		"held_rot_p90_deg": p90_rot,
	}


def _compute_ood_paired_vs_lab(per_frame: list[dict], *, camera_filter: str | None = "wrist_cam") -> dict[str, dict]:
	by_frame: dict[tuple[str, int], dict[str, dict]] = {}
	for rec in per_frame:
		if rec["split"] != "ood-appearance":
			continue
		if camera_filter is not None and rec["camera"] != camera_filter:
			continue
		by_frame.setdefault((rec["shard"], int(rec["index"])), {})[rec["method"]] = rec

	out: dict[str, dict] = {}
	for cand in ("unet-rect", "unet-pnp"):
		pairs = [(m["lab"], m[cand]) for m in by_frame.values() if "lab" in m and cand in m]
		n = len(pairs)
		lab_found = sum(1 for lb, _ in pairs if lb["found"])
		cand_found = sum(1 for _, cd in pairs if cd["found"])
		both_found = sum(1 for lb, cd in pairs if lb["found"] and cd["found"])
		cand_only = sum(1 for lb, cd in pairs if (not lb["found"]) and cd["found"])
		lab_only = sum(1 for lb, cd in pairs if lb["found"] and (not cd["found"]))
		neither = sum(1 for lb, cd in pairs if (not lb["found"]) and (not cd["found"]))

		lab_rate = round(100.0 * lab_found / n, 2) if n else 0.0
		cand_rate = round(100.0 * cand_found / n, 2) if n else 0.0
		lab_xy_med, _ = _median_p90([float(lb["xy_mm"]) for lb, _ in pairs if lb["found"] and lb["xy_mm"] is not None])
		cand_xy_med, _ = _median_p90([float(cd["xy_mm"]) for _, cd in pairs if cd["found"] and cd["xy_mm"] is not None])
		lab_yaw_med, _ = _median_p90(
			[float(lb["yaw_deg"]) for lb, _ in pairs if lb["found"] and lb["yaw_deg"] is not None]
		)
		cand_yaw_med, _ = _median_p90(
			[float(cd["yaw_deg"]) for _, cd in pairs if cd["found"] and cd["yaw_deg"] is not None]
		)
		paired_xy_diff, _ = _median_p90(
			[float(cd["xy_mm"]) - float(lb["xy_mm"]) for lb, cd in pairs if lb["found"] and cd["found"]]
		)
		paired_yaw_diff, _ = _median_p90(
			[float(cd["yaw_deg"]) - float(lb["yaw_deg"]) for lb, cd in pairs if lb["found"] and cd["found"]]
		)

		not_worse_rate = n > 0 and cand_rate >= lab_rate
		not_worse_xy = n > 0 and (
			(cand_xy_med is not None and lab_xy_med is not None and cand_xy_med <= lab_xy_med)
			or (cand_xy_med is not None and lab_xy_med is None)
			or (cand_xy_med is None and lab_xy_med is None and cand_found == lab_found)
		)
		not_worse_yaw = n > 0 and (
			(cand_yaw_med is not None and lab_yaw_med is not None and cand_yaw_med <= lab_yaw_med)
			or (cand_yaw_med is not None and lab_yaw_med is None)
			or (cand_yaw_med is None and lab_yaw_med is None and cand_found == lab_found)
		)
		out[cand] = {
			"camera_scope": camera_filter or "all",
			"n_frames": n,
			"lab_found": lab_found,
			"cand_found": cand_found,
			"lab_rate_pct": lab_rate,
			"cand_rate_pct": cand_rate,
			"delta_rate_pct": round(cand_rate - lab_rate, 2),
			"both_found_n": both_found,
			"cand_only_found_n": cand_only,
			"lab_only_found_n": lab_only,
			"neither_found_n": neither,
			"lab_xy_med_mm": lab_xy_med,
			"cand_xy_med_mm": cand_xy_med,
			"paired_xy_med_diff_mm": paired_xy_diff,
			"lab_yaw_med_deg": lab_yaw_med,
			"cand_yaw_med_deg": cand_yaw_med,
			"paired_yaw_med_diff_deg": paired_yaw_diff,
			"not_worse_rate": bool(not_worse_rate),
			"not_worse_xy": bool(not_worse_xy),
			"not_worse_yaw": bool(not_worse_yaw),
			"passed": bool(not_worse_rate and not_worse_xy and not_worse_yaw),
		}
	return out


def _compute_prereg_checks(
	val_ious: dict[str, float],
	val_pck_2px: float,
	per_frame: list[dict],
	ood_paired_vs_lab: dict[str, dict],
	*,
	primary_method: str = "unet-pnp",
) -> dict:
	val_wrist = [
		r for r in per_frame if r["split"] == "val" and r["camera"] == "wrist_cam" and r["method"] == primary_method
	]
	wrist_agg = _aggregate_pose_bucket(val_wrist)
	top_iou = float(val_ious.get("iou_c1", 0.0))
	pck = float(val_pck_2px)
	xy_med = wrist_agg["xy_med_mm"]
	yaw_med = wrist_agg["yaw_med_deg"]
	rate_pct = float(wrist_agg["rate_pct"])
	ood_cand = ood_paired_vs_lab[primary_method]

	checks = {
		"primary_method": primary_method,
		"sim_val_top_iou": {
			"metric": "sim-val top-face IoU",
			"op": ">=",
			"threshold": PREREG_THRESHOLDS["sim_val_top_iou_min"],
			"actual": top_iou,
			"passed": bool(top_iou >= PREREG_THRESHOLDS["sim_val_top_iou_min"]),
		},
		"sim_val_pck_2px": {
			"metric": "sim-val visible corner PCK@2px (320x240)",
			"op": ">=",
			"threshold": PREREG_THRESHOLDS["sim_val_pck_2px_min"],
			"actual": pck,
			"passed": bool(pck >= PREREG_THRESHOLDS["sim_val_pck_2px_min"]),
		},
		"sim_val_wrist_xy_med_mm": {
			"metric": "sim-val wrist pose median xy error (mm)",
			"op": "<=",
			"threshold": PREREG_THRESHOLDS["sim_val_wrist_xy_med_max_mm"],
			"actual": xy_med,
			"passed": bool(xy_med is not None and xy_med <= PREREG_THRESHOLDS["sim_val_wrist_xy_med_max_mm"]),
		},
		"sim_val_wrist_yaw_med_deg": {
			"metric": "sim-val wrist pose median yaw error (deg)",
			"op": "<=",
			"threshold": PREREG_THRESHOLDS["sim_val_wrist_yaw_med_max_deg"],
			"actual": yaw_med,
			"passed": bool(yaw_med is not None and yaw_med <= PREREG_THRESHOLDS["sim_val_wrist_yaw_med_max_deg"]),
		},
		"sim_val_wrist_rate_pct": {
			"metric": "sim-val wrist localization success rate (%)",
			"op": ">=",
			"threshold": PREREG_THRESHOLDS["sim_val_wrist_rate_min_pct"],
			"actual": rate_pct,
			"n": wrist_agg["n"],
			"passed": bool(wrist_agg["n"] > 0 and rate_pct >= PREREG_THRESHOLDS["sim_val_wrist_rate_min_pct"]),
		},
		"sim_ood_paired_vs_lab": {
			"metric": "sim-ood-appearance wrist paired vs lab",
			"op": "not_worse_than_lab",
			"threshold": "rate>=lab, xy_med<=lab, yaw_med<=lab",
			"actual": {
				"delta_rate_pct": ood_cand["delta_rate_pct"],
				"cand_xy_med_mm": ood_cand["cand_xy_med_mm"],
				"lab_xy_med_mm": ood_cand["lab_xy_med_mm"],
				"cand_yaw_med_deg": ood_cand["cand_yaw_med_deg"],
				"lab_yaw_med_deg": ood_cand["lab_yaw_med_deg"],
			},
			"passed": bool(ood_cand["passed"]),
		},
	}
	gate_keys = (
		"sim_val_top_iou",
		"sim_val_pck_2px",
		"sim_val_wrist_xy_med_mm",
		"sim_val_wrist_yaw_med_deg",
		"sim_val_wrist_rate_pct",
		"sim_ood_paired_vs_lab",
	)
	checks["all_passed"] = all(bool(checks[k]["passed"]) for k in gate_keys)
	return checks


def _load_aux_meta(aux_meta_path: Path | None) -> tuple[dict[str, dict], dict | None]:
	if aux_meta_path is None:
		return {}, None
	raw = json.loads(Path(aux_meta_path).read_text(encoding="utf-8"))
	frames = raw.get("frames") if isinstance(raw, dict) and "frames" in raw else raw
	alignment = raw.get("alignment") if isinstance(raw, dict) else None
	return dict(frames), alignment


def _format_iou_pck(inter: dict[int, int], union: dict[int, int], hit: int, tot: int) -> dict:
	ious = {f"iou_c{c}": round(inter[c] / union[c], 4) if union[c] else 0.0 for c in range(4)}
	pck = round(hit / tot, 4) if tot else 0.0
	return {"ious": ious, "pck_2px": pck, "pck_hit": hit, "pck_tot": tot}


def evaluate_shards(
	data_dir: Path,
	onnx_path: Path,
	meta_path: Path | None = None,
	*,
	aux_meta_path: Path | None = None,
	seed: int = 0,
	limit: int | None = None,
	per_frame_out: Path | None = None,
) -> dict:
	data_dir = Path(data_dir)
	net = BrickSegNet(onnx_path, meta_path)
	rng = np.random.default_rng(int(seed))
	aux_frames, aux_alignment = _load_aux_meta(aux_meta_path)
	shards = sorted(data_dir.rglob("shard_*.npz"))
	if not shards:
		raise FileNotFoundError(f"no shard_*.npz under {data_dir}")

	inter = {c: 0 for c in range(4)}
	union = {c: 0 for c in range(4)}
	pck_hit = 0
	pck_tot = 0
	cam_seg: dict[str, dict] = {}
	infer_ms: list[float] = []
	per_frame: list[dict] = []
	n_eval = 0

	pf_handle = None
	if per_frame_out is not None:
		per_frame_out = Path(per_frame_out)
		per_frame_out.parent.mkdir(parents=True, exist_ok=True)
		pf_handle = per_frame_out.open("w", encoding="utf-8", newline="\n")

	try:
		for spath in shards:
			with np.load(spath, allow_pickle=False) as z:
				rgbs, clss, corners = z["rgb"], z["cls"], z["corners"]
				metas = z["meta"] if "meta" in z.files else None
				for i in range(len(rgbs)):
					if limit is not None and n_eval >= limit:
						break
					m = json.loads(str(metas[i])) if metas is not None else {}
					split = str(m.get("split") or spath.parent.name)
					if split not in ("val", "ood-appearance"):
						continue
					aux = aux_frames.get(f"{spath.name}:{i}")
					if aux:
						for ak in (
							"table_saturation",
							"table_sat_bin",
							"same_color_clutter",
							"held",
							"world_from_gripper",
							"gripper_from_world",
							"brick_in_gripper_pos",
							"brick_in_gripper_rot",
						):
							if ak not in m and ak in aux:
								m[ak] = aux[ak]

					n_eval += 1
					rgb = rgbs[i]
					k_full = np.asarray(m["k"], dtype=np.float64)
					# Scale K down to 320x240 shard resolution
					k = k_full.copy()
					k[0, :] *= 0.5
					k[1, :] *= 0.5
					w2c = np.asarray(m["world_from_cam"], dtype=np.float64)
					bpos = np.asarray(m["brick_pos"], dtype=np.float64)
					brot = np.asarray(m["brick_rot"], dtype=np.float64)
					bsize = tuple(float(v) for v in m["brick_size"])
					byaw = math.atan2(brot[1, 0], brot[0, 0])
					btilt = math.degrees(math.acos(float(np.clip(brot[2, 2], -1.0, 1.0))))
					camera = str(m.get("camera") or "wrist_cam")
					view_kind = str(m.get("view_kind") or camera)
					flipped = btilt > 30.0
					held = bool(m["held"]) if "held" in m else (view_kind == "top_present")
					table_sat = float(m["table_saturation"]) if m.get("table_saturation") is not None else None
					sat_bin = str(m["table_sat_bin"]) if m.get("table_sat_bin") else _table_sat_bin(table_sat)
					scc = bool(m["same_color_clutter"]) if m.get("same_color_clutter") is not None else None
					g2w = (
						np.asarray(m["gripper_from_world"], dtype=np.float64)
						if m.get("gripper_from_world") is not None
						else None
					)

					# Single deterministic prior perturbation shared across ALL three methods on this frame
					prior_xy = bpos[:2] + rng.uniform(-0.015, 0.015, 2)
					prior_yaw = byaw + math.radians(float(rng.uniform(-15.0, 15.0)))
					top_z = float(bpos[2] + bsize[2] / 2.0)

					t0 = time.perf_counter()
					net_out = net.predict(rgb)
					infer_ms.append((time.perf_counter() - t0) * 1000.0)

					if split == "val":
						true_c = clss[i]
						c_acc = cam_seg.setdefault(
							camera,
							{
								"inter": {c: 0 for c in range(4)},
								"union": {c: 0 for c in range(4)},
								"hit": 0,
								"tot": 0,
								"n": 0,
							},
						)
						c_acc["n"] += 1
						for c in range(4):
							inter_c = int(((net_out.cls == c) & (true_c == c)).sum())
							union_c = int(((net_out.cls == c) | (true_c == c)).sum())
							inter[c] += inter_c
							union[c] += union_c
							c_acc["inter"][c] += inter_c
							c_acc["union"][c] += union_c
						for ch, sl in ((0, slice(0, 4)), (1, slice(4, 8))):
							pks = peaks(net_out.heat[ch], thr=0.3, radius=2, scale=net_out.scale)[:, :2]
							for u, v, vis in corners[i, sl]:
								if vis <= 0.5:
									continue
								pck_tot += 1
								c_acc["tot"] += 1
								if len(pks) and float(np.linalg.norm(pks - [u, v], axis=1).min()) <= 2.0:
									pck_hit += 1
									c_acc["hit"] += 1

					is_top_or_held = camera == "top" or held or flipped
					prior_xy_list = [round(float(prior_xy[0]), 6), round(float(prior_xy[1]), 6)]
					prior_yaw_r = round(float(prior_yaw), 6)

					for method in METHODS:
						pos_xy, yaw_est, pred_pos, pred_rot, fail_reason = _run_method(
							method,
							net_out,
							rgb,
							k,
							w2c,
							prior_xy,
							prior_yaw,
							bsize,
							top_z,
							is_top_or_held=is_top_or_held,
						)
						found = pos_xy is not None and yaw_est is not None
						xy_mm: float | None = None
						yaw_deg: float | None = None
						held_pos_mm: float | None = None
						held_rot_deg: float | None = None
						held_err_vec: list[float] | None = None
						if found:
							fail_reason = None
							xy_mm = round(float(np.linalg.norm(pos_xy - bpos[:2]) * 1000.0), 4)
							yaw_deg = round(float(yaw_error_deg(float(yaw_est), byaw, bsize)), 4)
							if held and g2w is not None and pred_pos is not None and pred_rot is not None:
								hp_mm, hr_deg, hv_mm = held_pose_error_in_gripper(
									pred_pos, pred_rot, bpos, brot, g2w, bsize
								)
								held_pos_mm = round(hp_mm, 4)
								held_rot_deg = round(hr_deg, 4)
								held_err_vec = [round(float(v), 4) for v in hv_mm]
						else:
							fail_reason = str(fail_reason or "UNKNOWN")

						rec = {
							"shard": spath.name,
							"index": int(i),
							"seed": int(m.get("seed", -1)),
							"split": split,
							"camera": camera,
							"view_kind": view_kind,
							"flipped": bool(flipped),
							"held": bool(held),
							"table_saturation": None if table_sat is None else round(table_sat, 6),
							"table_sat_bin": sat_bin,
							"same_color_clutter": scc,
							"prior_xy": prior_xy_list,
							"prior_yaw": prior_yaw_r,
							"method": method,
							"found": bool(found),
							"fail_reason": fail_reason,
							"xy_mm": xy_mm,
							"yaw_deg": yaw_deg,
							"held_pos_mm": held_pos_mm,
							"held_rot_deg": held_rot_deg,
							"held_err_gripper_mm": held_err_vec,
						}
						per_frame.append(rec)
						if pf_handle is not None:
							pf_handle.write(json.dumps(rec, separators=(",", ":")) + "\n")
			if limit is not None and n_eval >= limit:
				break
	finally:
		if pf_handle is not None:
			pf_handle.close()

	# 1. Pose summary by split x camera x orientation x method
	by_cam_orient: dict[tuple[str, str, str, str], list[dict]] = {}
	by_vkind: dict[tuple[str, str, str, str], list[dict]] = {}
	by_held: dict[tuple[str, str, str], list[dict]] = {}
	by_sat: dict[tuple[str, str, str, str], list[dict]] = {}
	by_scc: dict[tuple[str, str, bool | str, str], list[dict]] = {}
	fail_counts: Counter[tuple[str, str, str, str]] = Counter()

	for rec in per_frame:
		orient = "flipped" if rec["flipped"] else "upright"
		by_cam_orient.setdefault((rec["split"], rec["camera"], orient, rec["method"]), []).append(rec)
		by_vkind.setdefault(
			(rec["split"], _view_kind_group(rec["view_kind"], rec["camera"]), orient, rec["method"]), []
		).append(rec)
		if rec["held"]:
			by_held.setdefault((rec["split"], orient, rec["method"]), []).append(rec)
		by_sat.setdefault((rec["split"], rec["camera"], rec["table_sat_bin"], rec["method"]), []).append(rec)
		scc_key: bool | str = "unknown" if rec["same_color_clutter"] is None else bool(rec["same_color_clutter"])
		by_scc.setdefault((rec["split"], rec["camera"], scc_key, rec["method"]), []).append(rec)
		if not rec["found"]:
			fail_counts[(rec["split"], rec["camera"], rec["method"], str(rec["fail_reason"]))] += 1

	summary = [
		{"split": sp, "camera": cam, "orientation": orient, "method": m, **_aggregate_pose_bucket(recs)}
		for (sp, cam, orient, m), recs in sorted(by_cam_orient.items())
	]
	by_view_kind = [
		{"split": sp, "view_kind": vk, "orientation": orient, "method": m, **_aggregate_pose_bucket(recs)}
		for (sp, vk, orient, m), recs in sorted(by_vkind.items())
	]
	held_summary = [
		{"split": sp, "orientation": orient, "method": m, **_aggregate_held_bucket(recs)}
		for (sp, orient, m), recs in sorted(by_held.items())
	]
	sat_order = {"low": 0, "mid": 1, "high": 2, "unknown": 3}
	by_table_saturation = [
		{"split": sp, "camera": cam, "table_sat_bin": sb, "method": m, **_aggregate_pose_bucket(recs)}
		for (sp, cam, sb, m), recs in sorted(
			by_sat.items(), key=lambda kv: (kv[0][0], kv[0][1], sat_order.get(kv[0][2], 9), kv[0][3])
		)
	]
	by_same_color_clutter = [
		{"split": sp, "camera": cam, "same_color_clutter": sc, "method": m, **_aggregate_pose_bucket(recs)}
		for (sp, cam, sc, m), recs in sorted(
			by_scc.items(), key=lambda kv: (kv[0][0], kv[0][1], str(kv[0][2]), kv[0][3])
		)
	]
	failure_reasons = [
		{"split": sp, "camera": cam, "method": m, "fail_reason": reason, "count": cnt}
		for (sp, cam, m, reason), cnt in sorted(fail_counts.items())
	]

	val_overall = _format_iou_pck(inter, union, pck_hit, pck_tot)
	val_by_cam = {
		cam: {"n": acc["n"], **_format_iou_pck(acc["inter"], acc["union"], acc["hit"], acc["tot"])}
		for cam, acc in sorted(cam_seg.items())
	}
	cpu_med, cpu_p90 = _median_p90(infer_ms)
	ood_paired_wrist = _compute_ood_paired_vs_lab(per_frame, camera_filter="wrist_cam")
	ood_paired_all = _compute_ood_paired_vs_lab(per_frame, camera_filter=None)
	prereg_checks = _compute_prereg_checks(val_overall["ious"], val_overall["pck_2px"], per_frame, ood_paired_wrist)

	return {
		"n_frames_evaluated": n_eval,
		"aux_alignment": aux_alignment,
		"val_ious": val_overall["ious"],
		"val_pck_2px": val_overall["pck_2px"],
		"val_metrics_by_camera": val_by_cam,
		"cpu_ms": {"med": cpu_med, "p90": cpu_p90},
		"prereg_checks": prereg_checks,
		"ood_paired_vs_lab": ood_paired_wrist,
		"ood_paired_vs_lab_all_cameras": ood_paired_all,
		"summary": summary,
		"by_view_kind": by_view_kind,
		"held_summary": held_summary,
		"by_table_saturation": by_table_saturation,
		"by_same_color_clutter": by_same_color_clutter,
		"failure_reasons": failure_reasons,
		"per_frame": per_frame,
	}


def render_markdown(res: dict) -> str:
	lines = [
		"# BrickSeg Offline Pose Evaluation",
		"",
		f"- `frames_evaluated`: `{res.get('n_frames_evaluated', 0)}`",
		f"- `cpu_infer_ms`: median `{res['cpu_ms']['med']}` ms, p90 `{res['cpu_ms']['p90']}` ms",
	]
	if res.get("aux_alignment"):
		al = res["aux_alignment"]
		lines.append(
			f"- `aux_meta_alignment`: `{al['aligned_frames']}/{al['total_frames']}` aligned "
			f"(mismatched `{al['mismatched_frames']}`, max_pos `{al['max_pos_diff_m']}` m, "
			f"max_rot `{al['max_rot_diff']}`, max_cam `{al['max_cam_diff']}`)"
		)

	checks = res["prereg_checks"]
	lines += [
		"",
		"## Preregistered Pass Gate (`docs/experiments/brickseg_prereg.md`)",
		"",
		f"- `primary_method`: `{checks['primary_method']}`",
		f"- `overall_gate`: `{'PASS' if checks['all_passed'] else 'FAIL'}`",
		"",
		"| check | criterion | actual | status |",
		"|---|---|---|---|",
	]
	for key in (
		"sim_val_top_iou",
		"sim_val_pck_2px",
		"sim_val_wrist_xy_med_mm",
		"sim_val_wrist_yaw_med_deg",
		"sim_val_wrist_rate_pct",
		"sim_ood_paired_vs_lab",
	):
		c = checks[key]
		crit = f"{c['op']} {c['threshold']}"
		act = json.dumps(c["actual"], separators=(",", ":")) if isinstance(c["actual"], dict) else str(c["actual"])
		lines.append(f"| {c['metric']} | `{crit}` | `{act}` | **{'PASS' if c['passed'] else 'FAIL'}** |")

	lines += [
		"",
		"## Semantic Segmentation & Corner Metrics (`val`)",
		"",
		"### Overall (`val`)",
		"",
		*[f"- `{k}`: `{v}`" for k, v in res["val_ious"].items()],
		f"- `pck_2px`: `{res['val_pck_2px']}`",
		"",
		"### Camera-Stratified (`val`)",
		"",
		"| camera | n | iou_c0 | iou_c1 (top) | iou_c2 (bottom) | iou_c3 (side) | pck_2px |",
		"|---|---:|---:|---:|---:|---:|---:|",
	]
	for cam, cmet in sorted(res.get("val_metrics_by_camera", {}).items()):
		ci = cmet["ious"]
		lines.append(
			f"| {cam} | {cmet['n']} | {ci['iou_c0']} | {ci['iou_c1']} | {ci['iou_c2']} | {ci['iou_c3']} | {cmet['pck_2px']} |"
		)

	lines += [
		"",
		"## OOD Paired Comparison vs `lab` (`ood-appearance`, wrist)",
		"",
		"| method | n | lab_rate% | cand_rate% | delta_rate% | both | cand_only | lab_only | lab_xy_med | cand_xy_med | paired_dxy | lab_yaw_med | cand_yaw_med | paired_dyaw | pass |",
		"|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
	]
	for cand, row in sorted(res.get("ood_paired_vs_lab", {}).items()):
		lines.append(
			f"| {cand} | {row['n_frames']} | {row['lab_rate_pct']} | {row['cand_rate_pct']} | {row['delta_rate_pct']} "
			f"| {row['both_found_n']} | {row['cand_only_found_n']} | {row['lab_only_found_n']} "
			f"| {row['lab_xy_med_mm']} | {row['cand_xy_med_mm']} | {row['paired_xy_med_diff_mm']} "
			f"| {row['lab_yaw_med_deg']} | {row['cand_yaw_med_deg']} | {row['paired_yaw_med_diff_deg']} "
			f"| {'PASS' if row['passed'] else 'FAIL'} |"
		)

	lines += [
		"",
		"## Pose Accuracy by Split x Camera x Orientation x Method",
		"",
		"| split | camera | orient | method | n | found | rate% | xy_med_mm | xy_p90_mm | yaw_med_deg | yaw_p90_deg |",
		"|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
	]
	for r in res["summary"]:
		lines.append(
			f"| {r['split']} | {r['camera']} | {r['orientation']} | {r['method']} | {r['n']} | {r['found']} | {r['rate_pct']} "
			f"| {r['xy_med_mm']} | {r['xy_p90_mm']} | {r['yaw_med_deg']} | {r['yaw_p90_deg']} |"
		)

	lines += [
		"",
		"## Informational: Pose Accuracy by View Kind (`wrist`, `top_table`, `top_present`)",
		"",
		"| split | view_kind | orient | method | n | found | rate% | xy_med_mm | xy_p90_mm | yaw_med_deg | yaw_p90_deg |",
		"|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
	]
	for r in res.get("by_view_kind", []):
		lines.append(
			f"| {r['split']} | {r['view_kind']} | {r['orientation']} | {r['method']} | {r['n']} | {r['found']} | {r['rate_pct']} "
			f"| {r['xy_med_mm']} | {r['xy_p90_mm']} | {r['yaw_med_deg']} | {r['yaw_p90_deg']} |"
		)

	lines += [
		"",
		"## Informational: Held-in-Gripper Relative Pose (`gripper_from_world`, outside pass gate)",
		"",
		"| split | orient | method | n | found | rate% | held_pos_med_mm | held_pos_p90_mm | held_rot_med_deg | held_rot_p90_deg |",
		"|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
	]
	for r in res.get("held_summary", []):
		lines.append(
			f"| {r['split']} | {r['orientation']} | {r['method']} | {r['n']} | {r['found']} | {r['rate_pct']} "
			f"| {r['held_pos_med_mm']} | {r['held_pos_p90_mm']} | {r['held_rot_med_deg']} | {r['held_rot_p90_deg']} |"
		)

	lines += [
		"",
		"## Informational: Table Saturation Strata (`low` <0.15, `mid` [0.15, 0.35], `high` >0.35)",
		"",
		"| split | camera | sat_bin | method | n | found | rate% | xy_med_mm | xy_p90_mm | yaw_med_deg | yaw_p90_deg |",
		"|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
	]
	for r in res.get("by_table_saturation", []):
		lines.append(
			f"| {r['split']} | {r['camera']} | {r['table_sat_bin']} | {r['method']} | {r['n']} | {r['found']} | {r['rate_pct']} "
			f"| {r['xy_med_mm']} | {r['xy_p90_mm']} | {r['yaw_med_deg']} | {r['yaw_p90_deg']} |"
		)

	lines += [
		"",
		"## Informational: Same-Color Clutter Strata",
		"",
		"| split | camera | same_color_clutter | method | n | found | rate% | xy_med_mm | xy_p90_mm | yaw_med_deg | yaw_p90_deg |",
		"|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
	]
	for r in res.get("by_same_color_clutter", []):
		lines.append(
			f"| {r['split']} | {r['camera']} | {r['same_color_clutter']} | {r['method']} | {r['n']} | {r['found']} | {r['rate_pct']} "
			f"| {r['xy_med_mm']} | {r['xy_p90_mm']} | {r['yaw_med_deg']} | {r['yaw_p90_deg']} |"
		)

	lines += [
		"",
		"## Failure Reasons by Split x Camera x Method",
		"",
		"| split | camera | method | fail_reason | count |",
		"|---|---|---|---|---:|",
	]
	for r in res.get("failure_reasons", []):
		lines.append(f"| {r['split']} | {r['camera']} | {r['method']} | `{r['fail_reason']}` | {r['count']} |")

	return "\n".join(lines) + "\n"


def main(argv=None) -> int:
	p = argparse.ArgumentParser(description=__doc__)
	p.add_argument("--data-dir", "--data", dest="data_dir", type=Path, default=DEFAULT_DATA)
	p.add_argument("--onnx", type=Path, default=DEFAULT_DATA / "model" / "brickseg.onnx")
	p.add_argument("--meta", type=Path, default=None)
	p.add_argument("--aux-meta", type=Path, default=None)
	p.add_argument("--out", type=Path, default=DEFAULT_DATA / "eval.md")
	p.add_argument("--per-frame-out", type=Path, default=None)
	p.add_argument("--json-out", type=Path, default=None)
	p.add_argument("--seed", type=int, default=0)
	p.add_argument("--limit", type=int, default=None)
	args = p.parse_args(argv)
	res = evaluate_shards(
		args.data_dir,
		args.onnx,
		args.meta,
		aux_meta_path=args.aux_meta,
		seed=args.seed,
		limit=args.limit,
		per_frame_out=args.per_frame_out,
	)
	md = render_markdown(res)
	args.out.parent.mkdir(parents=True, exist_ok=True)
	args.out.write_text(md, encoding="utf-8", newline="\n")
	if args.json_out is not None:
		summary_json = {k: v for k, v in res.items() if k != "per_frame"}
		args.json_out.parent.mkdir(parents=True, exist_ok=True)
		args.json_out.write_text(json.dumps(summary_json, indent=2) + "\n", encoding="utf-8")
	print(
		f"wrote {args.out} ({res['n_frames_evaluated']} frames, {len(res['summary'])} summary rows, "
		f"gate={'PASS' if res['prereg_checks']['all_passed'] else 'FAIL'})"
	)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
