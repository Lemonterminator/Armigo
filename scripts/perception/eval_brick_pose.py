#!/usr/bin/env python
"""Offline CPU evaluation of lab vs unet-rect vs unet-pnp on U1 val and ood-appearance shards.

Plan: docs/superpowers/plans/2026-11-24-failure-observability-harness-learning.md Task U4.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
	sys.path.insert(0, str(ROOT / "src"))

from digital_twin_armigo.agent.brick_pose_nn import BrickSegNet, peaks, pose_from_keypoints
from digital_twin_armigo.agent.perception import yaw_error_deg
from digital_twin_armigo.agent.wrist_localize import WristFix, _project, _rays_to_plane, localize_from_wrist

DEFAULT_DATA = Path("E:/Armigo_data_v3/perception/brickseg-v1")
METHODS = ("lab", "unet-rect", "unet-pnp")


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
	from digital_twin_armigo.agent.brick_pose_nn import _nearest_top_component

	axis = w2c[:3, 2]
	tilt_cam = math.degrees(math.acos(float(np.clip(-axis[2], -1.0, 1.0))))
	if tilt_cam > max_tilt_deg:
		return None, "TILTED"
	centre, depth = _project(k, w2c, np.array([prior_xy[0], prior_xy[1], top_z]))
	if depth <= 1e-4:
		return None, "NO_BLOB"
	out = net.predict(rgb)
	sx, sy = out.scale
	gate_px = gate_m * float(k[0, 0]) / depth
	comp_net = _nearest_top_component(out.cls, np.array([centre[0] / sx, centre[1] / sy]), gate_px / max(sx, sy))
	if comp_net is None:
		return None, "NO_BLOB"
	h_img, w_img = rgb.shape[:2]
	mask_full = cv2.resize(comp_net, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
	return _rect_from_mask(mask_full, k, w2c, prior_yaw, size, top_z, tilt_cam)


def _median_p90(vals: list[float]) -> tuple[float | None, float | None]:
	if not vals:
		return None, None
	s = sorted(vals)
	p90 = s[min(len(s) - 1, int(0.9 * (len(s) - 1)))]
	return round(statistics.median(s), 3), round(p90, 3)


def evaluate_shards(
	data_dir: Path,
	onnx_path: Path,
	meta_path: Path | None = None,
	*,
	seed: int = 0,
	limit: int | None = None,
) -> dict:
	data_dir = Path(data_dir)
	net = BrickSegNet(onnx_path, meta_path)
	rng = np.random.default_rng(int(seed))
	shards = sorted(data_dir.rglob("shard_*.npz"))
	if not shards:
		raise FileNotFoundError(f"no shard_*.npz under {data_dir}")

	rows: dict[tuple[str, str, str], dict] = {}
	inter = {c: 0 for c in range(4)}
	union = {c: 0 for c in range(4)}
	pck_hit = 0
	pck_tot = 0
	infer_ms: list[float] = []
	n_eval = 0

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
				flipped = btilt > 30.0
				prior_xy = bpos[:2] + rng.uniform(-0.015, 0.015, 2)
				prior_yaw = byaw + math.radians(float(rng.uniform(-15.0, 15.0)))
				top_z = float(bpos[2] + bsize[2] / 2.0)

				t0 = time.perf_counter()
				net_out = net.predict(rgb)
				infer_ms.append((time.perf_counter() - t0) * 1000.0)
				if split == "val":
					true_c = clss[i]
					for c in range(4):
						inter[c] += int(((net_out.cls == c) & (true_c == c)).sum())
						union[c] += int(((net_out.cls == c) | (true_c == c)).sum())
					for ch, sl in ((0, slice(0, 4)), (1, slice(4, 8))):
						pks = peaks(net_out.heat[ch], thr=0.3, radius=2, scale=net_out.scale)[:, :2]
						for u, v, vis in corners[i, sl]:
							if vis <= 0.5:
								continue
							pck_tot += 1
							if len(pks) and float(np.linalg.norm(pks - [u, v], axis=1).min()) <= 2.0:
								pck_hit += 1

				if camera == "top":
					# Top camera + held pose metrics reported separately only
					sx, sy = net_out.scale
					_ = (sx, sy, pose_from_keypoints)
					continue

				for method in METHODS:
					bucket_key = (split, "flipped" if flipped else "upright", method)
					b = rows.setdefault(bucket_key, {"total": 0, "found": 0, "xy_mm": [], "yaw_deg": []})
					b["total"] += 1
					if method == "lab":
						fix, _ = localize_from_wrist(rgb, k, w2c, prior_xy, prior_yaw, bsize, top_z, segmenter="lab")
					elif method == "unet-rect":
						fix, _ = _unet_rect_localize(net, rgb, k, w2c, prior_xy, prior_yaw, bsize, top_z)
					else:
						fix, _ = localize_from_wrist(
							rgb, k, w2c, prior_xy, prior_yaw, bsize, top_z, segmenter="unet", net=net
						)
					if fix is not None:
						b["found"] += 1
						b["xy_mm"].append(float(np.linalg.norm(fix.pos_xy - bpos[:2]) * 1000.0))
						b["yaw_deg"].append(yaw_error_deg(fix.yaw, byaw, bsize))
		if limit is not None and n_eval >= limit:
			break

	summary = []
	for (split, orient, method), b in sorted(rows.items()):
		med_xy, p90_xy = _median_p90(b["xy_mm"])
		med_y, p90_y = _median_p90(b["yaw_deg"])
		rate = round(100.0 * b["found"] / b["total"], 2) if b["total"] else 0.0
		summary.append(
			{
				"split": split,
				"orientation": orient,
				"method": method,
				"n": b["total"],
				"found": b["found"],
				"rate_pct": rate,
				"xy_med_mm": med_xy,
				"xy_p90_mm": p90_xy,
				"yaw_med_deg": med_y,
				"yaw_p90_deg": p90_y,
			}
		)
	ious = {f"iou_c{c}": round(inter[c] / union[c], 4) if union[c] else 0.0 for c in range(4)}
	pck = round(pck_hit / pck_tot, 4) if pck_tot else 0.0
	cpu_med, cpu_p90 = _median_p90(infer_ms)
	return {"summary": summary, "val_ious": ious, "val_pck_2px": pck, "cpu_ms": {"med": cpu_med, "p90": cpu_p90}}


def render_markdown(res: dict) -> str:
	lines = [
		"# BrickSeg Offline Pose Evaluation",
		"",
		"## Semantic Segmentation & Corner Metrics (`val`)",
		"",
		*[f"- `{k}`: `{v}`" for k, v in res["val_ious"].items()],
		f"- `pck_2px`: `{res['val_pck_2px']}`",
		f"- `cpu_infer_ms`: median `{res['cpu_ms']['med']}` ms, p90 `{res['cpu_ms']['p90']}` ms",
		"",
		"## Pose Accuracy by Split x Method (wrist)",
		"",
		"| split | orient | method | n | found | rate% | xy_med_mm | xy_p90_mm | yaw_med_deg | yaw_p90_deg |",
		"|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
	]
	for r in res["summary"]:
		lines.append(
			f"| {r['split']} | {r['orientation']} | {r['method']} | {r['n']} | {r['found']} | {r['rate_pct']} "
			f"| {r['xy_med_mm']} | {r['xy_p90_mm']} | {r['yaw_med_deg']} | {r['yaw_p90_deg']} |"
		)
	return "\n".join(lines) + "\n"


def main(argv=None) -> int:
	p = argparse.ArgumentParser(description=__doc__)
	p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
	p.add_argument("--onnx", type=Path, default=DEFAULT_DATA / "model" / "brickseg.onnx")
	p.add_argument("--meta", type=Path, default=None)
	p.add_argument("--out", type=Path, default=DEFAULT_DATA / "eval.md")
	p.add_argument("--seed", type=int, default=0)
	p.add_argument("--limit", type=int, default=None)
	args = p.parse_args(argv)
	res = evaluate_shards(args.data_dir, args.onnx, args.meta, seed=args.seed, limit=args.limit)
	md = render_markdown(res)
	args.out.parent.mkdir(parents=True, exist_ok=True)
	args.out.write_text(md, encoding="utf-8", newline="\n")
	print(f"wrote {args.out} ({len(res['summary'])} summary rows)")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
