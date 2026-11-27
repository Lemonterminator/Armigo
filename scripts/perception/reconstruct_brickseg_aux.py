#!/usr/bin/env python
"""Lossless auxiliary metadata reconstruction and alignment verification for U1 brickseg-v1 shards.

Reconstructs per-frame ``table_saturation``, ``table_sat_bin``, ``same_color_clutter``, ``held``,
``world_from_gripper``, ``gripper_from_world``, ``brick_in_gripper_pos``, and ``brick_in_gripper_rot``
without modifying any existing ``.npz`` shard or ``manifest.json``.

When ``manifest.json`` records a ``git_commit`` that differs from ``HEAD`` (for example formal ``brickseg-v1``
generated at commit ``230ccde`` prior to WP4's ``visibility.py``/``scene_v3.py`` updates), the reconstructor
deterministically loads ``scene_v3.py`` and ``visibility.py`` from that exact commit so 100% of frames align
with stored ``brick_pos``, ``brick_rot``, and ``world_from_cam``.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
import types
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
	sys.path.insert(0, str(ROOT / "src"))
if str(SCRIPTS_DIR) not in sys.path:
	sys.path.insert(0, str(SCRIPTS_DIR))

from make_brickseg_dataset import (
	_mat_to_quat_xyzw,
	_rot_from_euler_xyz,
	gripper_transforms,
	has_same_color_clutter,
	table_sat_bin,
	table_saturation_from_rgb,
)

from digital_twin_armigo.agent.twin_mujoco import (
	INITIAL_Q,
	_wide_top_pose,
	case_cameras,
	case_wrist_offset,
	sample_case,
)
from digital_twin_armigo.camera import CameraPose
from digital_twin_armigo.constants import SO101_MJCF_DIR
from digital_twin_armigo.mujoco_cell.cell import Cell
from digital_twin_armigo.mujoco_cell.grasp_geometry import Kinematics
from digital_twin_armigo.mujoco_cell.handeye import camera_truth
from digital_twin_armigo.mujoco_cell.ik import ik_reach
from digital_twin_armigo.mujoco_cell.layout import sample_layout
from digital_twin_armigo.mujoco_cell.physics_expert import ExpertParams

DEFAULT_DATA = Path("E:/Armigo_data_v3/perception/brickseg-v1")
_SCENE_V3_REL = "src/digital_twin_armigo/mujoco_cell/scene_v3.py"
_VISIBILITY_REL = "src/digital_twin_armigo/mujoco_cell/visibility.py"
_GIT_LOADER_CACHE: dict[str, tuple[types.ModuleType, types.ModuleType]] = {}
_TOP_CAM_CFG: dict | None = None


def _head_commit() -> str:
	try:
		return subprocess.run(
			["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
		).stdout.strip()
	except (OSError, subprocess.CalledProcessError):
		return ""


def _git_has_commit_files(commit: str) -> bool:
	if not commit:
		return False
	for rel in (_SCENE_V3_REL, _VISIBILITY_REL):
		res = subprocess.run(
			["git", "-C", str(ROOT), "cat-file", "-e", f"{commit}:{rel}"],
			capture_output=True,
			text=True,
			check=False,
		)
		if res.returncode != 0:
			return False
	return True


def _load_module_from_git(commit: str, rel_path: str, package_name: str) -> types.ModuleType:
	src = subprocess.run(
		["git", "-C", str(ROOT), "show", f"{commit}:{rel_path}"],
		capture_output=True,
		text=True,
		check=True,
	).stdout
	safe_rel = rel_path.replace("/", "_").replace(".", "_")
	mod_name = f"_brickseg_hist_{commit[:12]}_{safe_rel}"
	mod = types.ModuleType(mod_name)
	mod.__package__ = package_name
	sys.modules[mod_name] = mod
	# Source is this repository's own file at the commit recorded in the dataset manifest (git show).
	exec(compile(src, f"{commit}:{rel_path}", "exec"), mod.__dict__)  # noqa: S102
	return mod


def _commit_modules(commit: str) -> tuple[types.ModuleType, types.ModuleType]:
	cached = _GIT_LOADER_CACHE.get(commit)
	if cached is not None:
		return cached
	sv3 = _load_module_from_git(commit, _SCENE_V3_REL, "digital_twin_armigo.mujoco_cell")
	vis = _load_module_from_git(commit, _VISIBILITY_REL, "digital_twin_armigo.mujoco_cell")
	_GIT_LOADER_CACHE[commit] = (sv3, vis)
	return sv3, vis


def _sample_case_for_commit(seed: int, positive: bool, face: int, split: str, git_commit: str | None):
	global _TOP_CAM_CFG
	if not git_commit:
		return sample_case(
			seed, positive=positive, face=face, options={"scene": "v3", "cameras": "wide", "split": split}
		)
	sv3, vis = _commit_modules(git_commit)
	if _TOP_CAM_CFG is None:
		_TOP_CAM_CFG = json.loads((ROOT / "config/sim/top_camera_joint.json").read_text(encoding="utf-8"))
	layout = sample_layout(
		np.random.default_rng(seed), CameraPose(**_TOP_CAM_CFG["camera"]), defective=positive, prior=None
	)
	layout.defect_kind = "cross_var"
	layout.defect_face = face
	layout.defect_seed = 19_199 + seed * 100_019
	layout.clutter = []
	layout.distractors = []
	layout.top, layout.front = case_cameras(seed, "fixed")
	scene = sv3.sample_scene_v3(seed, split)
	sv3.apply_scene_v3(layout, scene)
	layout.top, layout.front = case_cameras(seed, "wide", scene_v3=scene)
	layout.wrist_cam_offset = case_wrist_offset(seed, "wide", scene_v3=scene)
	layout.calib_top = None
	resamples = 0
	ok, _ = vis.workspace_visible(layout)
	while not ok:
		resamples += 1
		if resamples > 20:
			raise RuntimeError(
				f"v3 wide camera failed workspace_visible after 20 resamples (seed={seed}, split={split}, commit={git_commit[:7]})"
			)
		rng_cam = np.random.default_rng([int(seed), 0x5C3, sv3.SPLITS.index(split), 0xCA4, resamples])
		cam = sv3.draw_camera_v3(rng_cam, split)
		layout.top = _wide_top_pose(cam)
		ok, _ = vis.workspace_visible(layout)
	return layout


def _reconstruct_case_chunk(
	job: tuple[list[tuple[str, int, list[tuple[str, int, dict]]]], str | None, float, float, float],
) -> tuple[dict[str, dict], dict]:
	cases, git_commit, pos_tol_m, rot_tol, cam_tol = job
	open_g = ExpertParams().open_g
	q_init = np.asarray(INITIAL_Q, dtype=np.float64).copy()
	q_init[5] = open_g

	frames_out: dict[str, dict] = {}
	total_frames = 0
	aligned_frames = 0
	mismatched_frames = 0
	max_pos_diff_m = 0.0
	max_rot_diff = 0.0
	max_cam_diff = 0.0
	sample_mismatches: list[dict] = []

	for split, seed, frame_items in cases:
		rng = np.random.default_rng([int(seed), 0xB51C])
		positive = bool(int(rng.integers(0, 2)))
		face = int(rng.integers(0, 6))
		layout = _sample_case_for_commit(int(seed), positive, face, split, git_commit)

		table_sat = table_saturation_from_rgb(layout.table_rgb)
		sat_bin = table_sat_bin(table_sat)
		same_clutter = has_same_color_clutter(layout)

		cell = Cell(
			layout,
			SO101_MJCF_DIR,
			dynamic_clutter=True,
			assisted_grasp=False,
			face_probes=False,
			render_enabled=False,
		)
		try:
			cell.set_joints(q_init)
			table_pos = cell.brick_pos().copy()
			table_quat = cell.brick_quat_xyzw().copy()
			brick_yaw = math.radians(float(layout.brick_yaw_deg))
			top_z = float(table_pos[2] + layout.brick_size[2] / 2.0)
			kin = Kinematics(cell)

			def _capture(cam_name: str, cell=cell, layout=layout) -> dict:
				bp = cell.brick_pos().copy()
				br = cell.data.xmat[cell.brick_body].reshape(3, 3).copy()
				w2c = camera_truth(cell.model, cell.data, cam_name)
				return {
					"brick_pos": bp,
					"brick_rot": br,
					"world_from_cam": w2c,
					"brick_size": np.asarray(layout.brick_size, dtype=np.float64),
					"gtf": gripper_transforms(cell, bp, br),
				}

			views: dict[str, dict] = {}
			for w_i in range(4):
				cell.set_brick_pose(table_pos, table_quat)
				dx, dy = rng.uniform(-0.03, 0.03, 2)
				dz = float(rng.uniform(0.04, 0.14))
				roll = brick_yaw + float(rng.uniform(-math.pi / 2.0, math.pi / 2.0))
				target = np.array([table_pos[0] + dx, table_pos[1] + dy, top_z + dz], dtype=np.float64)
				q_w = ik_reach(kin, target, q_init, roll)
				q_w[5] = open_g
				cell.set_joints(q_w)
				views[f"wrist_{w_i}"] = _capture("wrist_cam")

			cell.set_brick_pose(table_pos, table_quat)
			cell.set_joints(q_init)
			views["top_table"] = _capture("top")

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
			views["top_present"] = _capture("top")
		finally:
			cell.close()

		for shard_name, frame_idx, stored in frame_items:
			total_frames += 1
			vkind = str(stored["view_kind"])
			rec = views[vkind]
			dp = float(np.max(np.abs(rec["brick_pos"] - np.asarray(stored["brick_pos"], dtype=np.float64))))
			dr = float(np.max(np.abs(rec["brick_rot"] - np.asarray(stored["brick_rot"], dtype=np.float64))))
			dc = float(np.max(np.abs(rec["world_from_cam"] - np.asarray(stored["world_from_cam"], dtype=np.float64))))
			ds = float(np.max(np.abs(rec["brick_size"] - np.asarray(stored["brick_size"], dtype=np.float64))))
			max_pos_diff_m = max(max_pos_diff_m, dp)
			max_rot_diff = max(max_rot_diff, dr)
			max_cam_diff = max(max_cam_diff, dc)
			ok_align = dp <= pos_tol_m and dr <= rot_tol and dc <= cam_tol and ds <= 1e-5
			if ok_align:
				aligned_frames += 1
			else:
				mismatched_frames += 1
				if len(sample_mismatches) < 20:
					sample_mismatches.append(
						{
							"shard": shard_name,
							"index": int(frame_idx),
							"seed": int(seed),
							"split": split,
							"view_kind": vkind,
							"pos_diff_m": round(dp, 7),
							"rot_diff": round(dr, 7),
							"cam_diff": round(dc, 7),
							"size_diff_m": round(ds, 7),
						}
					)
			key = f"{shard_name}:{int(frame_idx)}"
			frames_out[key] = {
				"shard": shard_name,
				"index": int(frame_idx),
				"seed": int(seed),
				"split": split,
				"camera": str(stored.get("camera") or ""),
				"view_kind": vkind,
				"aligned": bool(ok_align),
				"pos_diff_m": round(dp, 7),
				"rot_diff": round(dr, 7),
				"cam_diff": round(dc, 7),
				"table_saturation": round(table_sat, 6),
				"table_sat_bin": sat_bin,
				"same_color_clutter": bool(same_clutter),
				"held": bool(vkind == "top_present"),
				**rec["gtf"],
			}

	stats = {
		"total_frames": total_frames,
		"aligned_frames": aligned_frames,
		"mismatched_frames": mismatched_frames,
		"max_pos_diff_m": max_pos_diff_m,
		"max_rot_diff": max_rot_diff,
		"max_cam_diff": max_cam_diff,
		"mismatches": sample_mismatches,
	}
	return frames_out, stats


def _target_shards(data_dir: Path, splits: tuple[str, ...]) -> tuple[list[Path], str]:
	manifest_path = data_dir / "manifest.json"
	manifest_commit = ""
	if manifest_path.is_file():
		mf = json.loads(manifest_path.read_text(encoding="utf-8"))
		manifest_commit = str(mf.get("git_commit") or "").strip()
		if mf.get("shards"):
			wanted = [data_dir / s["name"] for s in mf["shards"] if s.get("split") in splits]
			if wanted:
				return wanted, manifest_commit
	return sorted(data_dir.rglob("shard_*.npz")), manifest_commit


def reconstruct_aux_metadata(
	data_dir: Path,
	out_path: Path | None = None,
	*,
	splits: tuple[str, ...] = ("val", "ood-appearance"),
	workers: int = 3,
	pos_tol_m: float = 1e-4,
	rot_tol: float = 1e-3,
	cam_tol: float = 1e-4,
) -> dict:
	"""Reconstruct auxiliary metadata for ``splits`` under ``data_dir`` and verify geometric alignment."""
	if workers < 1 or workers > 4:
		raise ValueError(f"workers must be in [1, 4], got {workers}")
	data_dir = Path(data_dir)
	shard_paths, manifest_commit = _target_shards(data_dir, splits)
	if not shard_paths:
		raise FileNotFoundError(f"no target shard_*.npz found under {data_dir} for splits={splits}")

	head = _head_commit()
	use_commit: str | None = None
	if manifest_commit and manifest_commit != head and _git_has_commit_files(manifest_commit):
		use_commit = manifest_commit

	by_case: dict[tuple[str, int], list[tuple[str, int, dict]]] = {}
	view_counts: Counter[tuple[str, str, str, bool]] = Counter()
	for spath in shard_paths:
		with np.load(spath, allow_pickle=False) as z:
			metas = z["meta"]
			for idx, raw in enumerate(metas):
				m = json.loads(str(raw))
				sp = str(m.get("split") or spath.parent.name)
				if sp not in splits:
					continue
				seed = int(m["seed"])
				by_case.setdefault((sp, seed), []).append((spath.name, int(idx), m))
				rot = np.asarray(m["brick_rot"], dtype=np.float64)
				tilt = math.degrees(math.acos(float(np.clip(rot[2, 2], -1.0, 1.0))))
				view_counts[(sp, str(m.get("camera") or ""), str(m.get("view_kind") or ""), tilt > 30.0)] += 1

	ordered_cases = [(sp, seed, items) for (sp, seed), items in sorted(by_case.items())]
	t0 = time.perf_counter()

	all_frames: dict[str, dict] = {}
	total_frames = 0
	aligned_frames = 0
	mismatched_frames = 0
	max_pos_diff_m = 0.0
	max_rot_diff = 0.0
	max_cam_diff = 0.0
	mismatches: list[dict] = []

	n_workers = min(workers, max(1, len(ordered_cases)))
	if n_workers == 1 or len(ordered_cases) < 4:
		chunks = [ordered_cases]
		chunk_results = [_reconstruct_case_chunk((ordered_cases, use_commit, pos_tol_m, rot_tol, cam_tol))]
	else:
		chunk_size = math.ceil(len(ordered_cases) / n_workers)
		chunks = [ordered_cases[i : i + chunk_size] for i in range(0, len(ordered_cases), chunk_size)]
		jobs = [(ch, use_commit, pos_tol_m, rot_tol, cam_tol) for ch in chunks]
		with ProcessPoolExecutor(max_workers=n_workers, mp_context=get_context("spawn")) as pool:
			chunk_results = list(pool.map(_reconstruct_case_chunk, jobs))

	for fdict, st in chunk_results:
		all_frames.update(fdict)
		total_frames += st["total_frames"]
		aligned_frames += st["aligned_frames"]
		mismatched_frames += st["mismatched_frames"]
		max_pos_diff_m = max(max_pos_diff_m, st["max_pos_diff_m"])
		max_rot_diff = max(max_rot_diff, st["max_rot_diff"])
		max_cam_diff = max(max_cam_diff, st["max_cam_diff"])
		if len(mismatches) < 20:
			mismatches.extend(st["mismatches"][: 20 - len(mismatches)])

	sat_counts: Counter[tuple[str, str]] = Counter()
	scc_counts: Counter[tuple[str, str]] = Counter()
	for rec in all_frames.values():
		sat_counts[(rec["split"], rec["table_sat_bin"])] += 1
		scc_counts[(rec["split"], str(rec["same_color_clutter"]))] += 1

	result = {
		"schema_version": 1,
		"source_data_dir": str(data_dir),
		"manifest_git_commit": manifest_commit,
		"used_git_commit_loader": bool(use_commit is not None),
		"splits": list(splits),
		"wall_s": round(time.perf_counter() - t0, 2),
		"alignment": {
			"total_frames": total_frames,
			"aligned_frames": aligned_frames,
			"mismatched_frames": mismatched_frames,
			"max_pos_diff_m": round(max_pos_diff_m, 7),
			"max_rot_diff": round(max_rot_diff, 7),
			"max_cam_diff": round(max_cam_diff, 7),
			"tolerances": {"pos_m": pos_tol_m, "rot": rot_tol, "cam": cam_tol},
			"mismatches": mismatches,
		},
		"counts": {
			"by_split_camera_view_flipped": {
				f"{sp}|{cam}|{vk}|flipped={fl}": cnt for (sp, cam, vk, fl), cnt in sorted(view_counts.items())
			},
			"by_split_sat_bin": {f"{sp}|{sb}": cnt for (sp, sb), cnt in sorted(sat_counts.items())},
			"by_split_same_color_clutter": {
				f"{sp}|same_color={sc}": cnt for (sp, sc), cnt in sorted(scc_counts.items())
			},
		},
		"frames": all_frames,
	}

	if out_path is not None:
		out_path = Path(out_path)
		out_path.parent.mkdir(parents=True, exist_ok=True)
		out_path.write_text(json.dumps(result, separators=(",", ":")) + "\n", encoding="utf-8")

	return result


def main(argv=None) -> int:
	p = argparse.ArgumentParser(description=__doc__)
	p.add_argument("--data-dir", "--data", dest="data_dir", type=Path, default=DEFAULT_DATA)
	p.add_argument("--out", type=Path, required=True)
	p.add_argument("--splits", nargs="+", default=["val", "ood-appearance"])
	p.add_argument("--workers", type=int, default=3)
	p.add_argument("--strict", action="store_true", help="Exit non-zero if any frame fails alignment verification")
	args = p.parse_args(argv)
	res = reconstruct_aux_metadata(args.data_dir, args.out, splits=tuple(args.splits), workers=args.workers)
	al = res["alignment"]
	print(
		json.dumps(
			{
				"out": str(args.out),
				"used_git_commit_loader": res["used_git_commit_loader"],
				"manifest_git_commit": res["manifest_git_commit"][:12],
				"wall_s": res["wall_s"],
				"alignment": {
					k: al[k]
					for k in (
						"total_frames",
						"aligned_frames",
						"mismatched_frames",
						"max_pos_diff_m",
						"max_rot_diff",
						"max_cam_diff",
					)
				},
				"counts": res["counts"],
			},
			indent=2,
		)
	)
	if args.strict and al["mismatched_frames"] > 0:
		return 2
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
