#!/usr/bin/env python
"""Generate the MuJoCo brick face-segmentation + corner dataset (CPU-only, up to 4 workers).

Plan: docs/superpowers/plans/2026-11-24-failure-observability-harness-learning.md Task U1.
Output shards ``shard_XXXX.npz`` contain:
- ``rgb``: ``(N, 240, 320, 3)`` uint8
- ``cls``: ``(N, 240, 320)`` uint8 (0=bg, 1=top, 2=bottom, 3=side)
- ``corners``: ``(N, 8, 3)`` float32 ``(u, v, visible)`` in 320x240 coords
- ``meta``: ``(N,)`` JSON strings with seed, split, camera, view_kind, brick pose, K, world_from_cam, brick_size
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
	sys.path.insert(0, str(ROOT / "src"))

from digital_twin_armigo.agent.twin_mujoco import INITIAL_Q, sample_case
from digital_twin_armigo.constants import SO101_MJCF_DIR
from digital_twin_armigo.mujoco_cell._mj import mujoco
from digital_twin_armigo.mujoco_cell.cell import RENDER_SIZE, Cell
from digital_twin_armigo.mujoco_cell.grasp_geometry import Kinematics
from digital_twin_armigo.mujoco_cell.handeye import camera_truth, intrinsics
from digital_twin_armigo.mujoco_cell.ik import ik_reach
from digital_twin_armigo.mujoco_cell.physics_expert import ExpertParams
from digital_twin_armigo.perception_nn.labels import brick_corners_world, corner_labels

DEFAULT_OUT = Path("E:/Armigo_data_v3/perception/brickseg-v1")
OUT_HW = (240, 320)
MIN_BRICK_PX = 150
SPLIT_RANGES: dict[str, tuple[int, int]] = {
	"train": (3_000_001, 3_099_999),
	"val": (3_100_001, 3_199_999),
	"ood-appearance": (3_500_001, 3_599_999),
}
CLASS_COLOURS_BGR = {
	1: (0, 255, 0),  # top: green
	2: (255, 0, 0),  # bottom: blue
	3: (0, 165, 255),  # side: orange
}


def _git_commit() -> str:
	try:
		return subprocess.run(
			["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
		).stdout.strip()
	except (OSError, subprocess.CalledProcessError):
		return ""


def _sha256(path: Path) -> str:
	h = hashlib.sha256()
	with path.open("rb") as f:
		for chunk in iter(lambda: f.read(1 << 20), b""):
			h.update(chunk)
	return h.hexdigest()


def _rot_from_euler_xyz(rx: float, ry: float, rz: float) -> np.ndarray:
	cx, sx = math.cos(rx), math.sin(rx)
	cy, sy = math.cos(ry), math.sin(ry)
	cz, sz = math.cos(rz), math.sin(rz)
	rxm = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
	rym = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
	rzm = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
	return rzm @ rym @ rxm


def _mat_to_quat_xyzw(rot: np.ndarray) -> np.ndarray:
	wxyz = np.empty(4, dtype=np.float64)
	mujoco.mju_mat2Quat(wxyz, np.asarray(rot, dtype=np.float64).reshape(-1))
	return np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]], dtype=np.float64)


def _wrist_k(fovy_deg: float, shape: tuple[int, int]) -> np.ndarray:
	h, w = shape[:2]
	f = 0.5 * h / math.tan(math.radians(fovy_deg) / 2.0)
	return np.array([[f, 0.0, w / 2.0 - 0.5], [0.0, f, h / 2.0 - 0.5], [0.0, 0.0, 1.0]], dtype=np.float64)


def _render_sample(
	cell: Cell,
	camera: str,
	view_kind: str,
	seed: int,
	split: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str] | None:
	cls_raw = cell.face_class_map(camera)
	if int(np.count_nonzero(cls_raw >= 0)) < MIN_BRICK_PX:
		return None
	rgb_raw = cell.rgb(camera)
	depth_raw = cell.depth(camera)
	if camera == "top":
		k = intrinsics(cell._cam_poses["top"], RENDER_SIZE)
	else:
		cid = cell.camera_id(camera)
		k = _wrist_k(float(cell.model.cam_fovy[cid]), rgb_raw.shape)
	w2c = camera_truth(cell.model, cell.data, camera)

	brick_pos = cell.brick_pos()
	brick_rot = cell.data.xmat[cell.brick_body].reshape(3, 3).copy()
	size = tuple(float(v) for v in cell.layout.brick_size)
	cw = brick_corners_world(brick_pos, brick_rot, size)
	corners = corner_labels(cw, k, w2c, depth_raw)
	corners[:, :2] *= 0.5

	rgb = cv2.resize(rgb_raw, (OUT_HW[1], OUT_HW[0]), interpolation=cv2.INTER_AREA)
	cls_u8 = (cls_raw.astype(np.int16) + 1).astype(np.uint8)
	cls_res = cv2.resize(cls_u8, (OUT_HW[1], OUT_HW[0]), interpolation=cv2.INTER_NEAREST)

	meta = json.dumps(
		{
			"seed": int(seed),
			"split": split,
			"camera": camera,
			"view_kind": view_kind,
			"brick_pos": [round(float(v), 6) for v in brick_pos],
			"brick_rot": [[round(float(x), 6) for x in row] for row in brick_rot],
			"k": [[round(float(x), 4) for x in row] for row in k],
			"world_from_cam": [[round(float(x), 6) for x in row] for row in w2c],
			"brick_size": [round(float(v), 6) for v in size],
		},
		separators=(",", ":"),
	)
	return rgb, cls_res, corners.astype(np.float32), meta


def _generate_case(seed: int, split: str) -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray, str]], int]:
	rng = np.random.default_rng([int(seed), 0xB51C])
	positive = bool(int(rng.integers(0, 2)))
	face = int(rng.integers(0, 6))
	layout = sample_case(seed, positive=positive, face=face, options={"scene": "v3", "cameras": "wide", "split": split})
	open_g = ExpertParams().open_g
	q_init = np.asarray(INITIAL_Q, dtype=np.float64).copy()
	q_init[5] = open_g

	cell = Cell(layout, SO101_MJCF_DIR, dynamic_clutter=True, assisted_grasp=False, face_probes=True)
	samples: list[tuple[np.ndarray, np.ndarray, np.ndarray, str]] = []
	dropped = 0
	try:
		cell.set_joints(q_init)
		table_pos = cell.brick_pos().copy()
		table_quat = cell.brick_quat_xyzw().copy()
		brick_yaw = math.radians(float(layout.brick_yaw_deg))
		top_z = float(table_pos[2] + layout.brick_size[2] / 2.0)
		kin = Kinematics(cell)

		# 1. Four wrist views above table brick: xy +-3 cm, dz 0.04-0.14 m, roll +-90 deg
		for w_i in range(4):
			cell.set_brick_pose(table_pos, table_quat)
			dx, dy = rng.uniform(-0.03, 0.03, 2)
			dz = float(rng.uniform(0.04, 0.14))
			roll = brick_yaw + float(rng.uniform(-math.pi / 2.0, math.pi / 2.0))
			target = np.array([table_pos[0] + dx, table_pos[1] + dy, top_z + dz], dtype=np.float64)
			q_w = ik_reach(kin, target, q_init, roll)
			q_w[5] = open_g
			cell.set_joints(q_w)
			item = _render_sample(cell, "wrist_cam", f"wrist_{w_i}", seed, split)
			if item is None:
				dropped += 1
			else:
				samples.append(item)

		# 2. One top view with brick on the table
		cell.set_brick_pose(table_pos, table_quat)
		cell.set_joints(q_init)
		item = _render_sample(cell, "top", "top_table", seed, split)
		if item is None:
			dropped += 1
		else:
			samples.append(item)

		# 3. One top view with brick lifted at presentation height (30% flipped/sideways) + occluding arm nearby
		pres_pos = np.array(
			[float(rng.uniform(0.18, 0.30)), float(rng.uniform(-0.08, 0.08)), float(rng.uniform(0.10, 0.18))]
		)
		yaw_p = float(rng.uniform(-math.pi, math.pi))
		if float(rng.random()) < 0.30:
			tilt_opts = ((math.pi, 0.0), (math.pi / 2.0, 0.0), (-math.pi / 2.0, 0.0), (0.0, math.pi / 2.0))
			rx, ry = tilt_opts[int(rng.integers(0, len(tilt_opts)))]
		else:
			rx, ry = 0.0, 0.0
		pres_rot = _rot_from_euler_xyz(rx, ry, yaw_p)
		cell.set_brick_pose(pres_pos, _mat_to_quat_xyzw(pres_rot))

		arm_off = np.array(
			[float(rng.uniform(-0.05, 0.05)), float(rng.uniform(-0.05, 0.05)), float(rng.uniform(0.02, 0.06))]
		)
		q_pres = ik_reach(kin, pres_pos + arm_off, q_init, yaw_p)
		q_pres[5] = float(rng.uniform(0.25, open_g))
		cell.set_joints(q_pres)
		item = _render_sample(cell, "top", "top_present", seed, split)
		if item is None:
			dropped += 1
		else:
			samples.append(item)
	finally:
		cell.close()
	return samples, dropped


def _run_shard_job(job: tuple[int, str, list[int], str]) -> dict:
	shard_idx, split, seeds, shard_path_str = job
	rgbs, clss, corners, metas = [], [], [], []
	dropped = 0
	for s in seeds:
		items, d = _generate_case(int(s), split)
		dropped += d
		for rgb, cls_map, cor, meta in items:
			rgbs.append(rgb)
			clss.append(cls_map)
			corners.append(cor)
			metas.append(meta)
	shard_path = Path(shard_path_str)
	shard_path.parent.mkdir(parents=True, exist_ok=True)
	if rgbs:
		np.savez_compressed(
			shard_path,
			rgb=np.stack(rgbs, axis=0),
			cls=np.stack(clss, axis=0),
			corners=np.stack(corners, axis=0),
			meta=np.asarray(metas),
		)
		sha = _sha256(shard_path)
	else:
		sha = ""
	return {
		"shard_idx": shard_idx,
		"shard_name": shard_path.name if rgbs else "",
		"split": split,
		"n_cases": len(seeds),
		"n_frames": len(rgbs),
		"dropped": dropped,
		"sha256": sha,
	}


def render_preview(shard_path: Path, preview_dir: Path, count: int = 10, seed: int = 0) -> list[Path]:
	preview_dir.mkdir(parents=True, exist_ok=True)
	with np.load(shard_path, allow_pickle=False) as z:
		n = len(z["rgb"])
		if n == 0:
			return []
		rng = np.random.default_rng(seed)
		idxs = rng.choice(n, size=min(count, n), replace=False) if n > count else np.arange(n)
		written = []
		for rank, idx in enumerate(sorted(int(i) for i in idxs)):
			rgb = z["rgb"][idx]
			cls_map = z["cls"][idx]
			cor = z["corners"][idx]
			bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
			overlay = bgr.copy()
			for cval, colour in CLASS_COLOURS_BGR.items():
				overlay[cls_map == cval] = colour
			vis = cv2.addWeighted(bgr, 0.6, overlay, 0.4, 0.0)
			for ci, (u, v, flag) in enumerate(cor):
				if flag > 0.5:
					col = (0, 255, 255) if ci < 4 else (255, 0, 255)
					cv2.circle(vis, (round(float(u)), round(float(v))), 3, col, -1)
			out = preview_dir / f"preview_{rank:02d}_idx{idx}.png"
			cv2.imwrite(str(out), vis)
			written.append(out)
		return written


def generate_dataset(
	out_dir: Path,
	*,
	train_cases: int = 2000,
	val_cases: int = 250,
	ood_cases: int = 200,
	cases_per_shard: int = 50,
	workers: int = 3,
	preview_dir: Path | None = None,
	preview_count: int = 10,
	clean: bool = False,
) -> dict:
	if workers < 1 or workers > 4:
		raise ValueError(f"workers must be in [1, 4] for MuJoCo CPU generation, got {workers}")
	out_dir = Path(out_dir)
	if clean and out_dir.exists():
		shutil.rmtree(out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	split_counts = [("train", train_cases), ("val", val_cases), ("ood-appearance", ood_cases)]
	jobs = []
	shard_idx = 0
	for split, count in split_counts:
		if count <= 0:
			continue
		lo, hi = SPLIT_RANGES[split]
		if lo + count - 1 > hi:
			raise ValueError(f"{split}: {count} cases exceed seed range [{lo}, {hi}]")
		seeds = list(range(lo, lo + count))
		for start in range(0, len(seeds), cases_per_shard):
			chunk = seeds[start : start + cases_per_shard]
			shard_path = out_dir / f"shard_{shard_idx:04d}.npz"
			jobs.append((shard_idx, split, chunk, str(shard_path)))
			shard_idx += 1

	t0 = time.perf_counter()
	results: list[dict] = []
	if workers == 1:
		for job in jobs:
			res = _run_shard_job(job)
			results.append(res)
			print(
				f"[{res['shard_idx'] + 1}/{len(jobs)}] {res['split']} frames={res['n_frames']} dropped={res['dropped']}",
				flush=True,
			)
	else:
		with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
			for res in pool.map(_run_shard_job, jobs):
				results.append(res)
				print(
					f"[{res['shard_idx'] + 1}/{len(jobs)}] {res['split']} frames={res['n_frames']} dropped={res['dropped']}",
					flush=True,
				)

	frames_by_split: dict[str, int] = {}
	dropped_by_split: dict[str, int] = {}
	shards_meta = []
	for r in sorted(results, key=lambda x: x["shard_idx"]):
		sp = r["split"]
		frames_by_split[sp] = frames_by_split.get(sp, 0) + int(r["n_frames"])
		dropped_by_split[sp] = dropped_by_split.get(sp, 0) + int(r["dropped"])
		if r["shard_name"]:
			shards_meta.append(
				{
					"name": r["shard_name"],
					"split": sp,
					"frames": int(r["n_frames"]),
					"dropped": int(r["dropped"]),
					"sha256": r["sha256"],
				}
			)

	manifest = {
		"seed_ranges": {k: list(v) for k, v in SPLIT_RANGES.items()},
		"cases_requested": {"train": train_cases, "val": val_cases, "ood-appearance": ood_cases},
		"frames_by_split": frames_by_split,
		"dropped_by_split": dropped_by_split,
		"total_frames": sum(frames_by_split.values()),
		"total_dropped": sum(dropped_by_split.values()),
		"git_commit": _git_commit(),
		"wall_s": round(time.perf_counter() - t0, 2),
		"shards": shards_meta,
	}
	(out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

	if preview_dir is not None and shards_meta:
		first_shard = out_dir / shards_meta[0]["name"]
		render_preview(first_shard, preview_dir, count=preview_count)

	return manifest


def main(argv=None) -> int:
	p = argparse.ArgumentParser(description=__doc__)
	p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
	p.add_argument("--train-cases", type=int, default=2000)
	p.add_argument("--val-cases", type=int, default=250)
	p.add_argument("--ood-cases", type=int, default=200)
	p.add_argument("--cases-per-shard", type=int, default=50)
	p.add_argument("--workers", type=int, default=3)
	p.add_argument("--preview-dir", type=Path, default=None)
	p.add_argument("--preview-count", type=int, default=10)
	p.add_argument("--clean", action="store_true")
	args = p.parse_args(argv)
	mf = generate_dataset(
		args.out_dir,
		train_cases=args.train_cases,
		val_cases=args.val_cases,
		ood_cases=args.ood_cases,
		cases_per_shard=args.cases_per_shard,
		workers=args.workers,
		preview_dir=args.preview_dir,
		preview_count=args.preview_count,
		clean=args.clean,
	)
	print(json.dumps({k: mf[k] for k in ("total_frames", "total_dropped", "frames_by_split", "dropped_by_split")}))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
