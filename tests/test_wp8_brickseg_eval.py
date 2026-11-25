"""WP8 TDD tests for U4 offline brick pose evaluation and lossless auxiliary metadata reconstruction."""

from __future__ import annotations

import colorsys
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_PERCEPTION = ROOT / "scripts" / "perception"
if str(SCRIPTS_PERCEPTION) not in sys.path:
	sys.path.insert(0, str(SCRIPTS_PERCEPTION))

TINY_ONNX = ROOT / "tests" / "data" / "brickseg_tiny.onnx"
TINY_META = ROOT / "tests" / "data" / "brickseg_tiny.json"
BRICK_SIZE_2X2 = (0.0318, 0.0318, 0.0192)
BRICK_SIZE_2X4 = (0.0636, 0.0318, 0.0192)


def _rot_z(yaw: float) -> np.ndarray:
	c, s = math.cos(yaw), math.sin(yaw)
	return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _rot_x(rx: float) -> np.ndarray:
	c, s = math.cos(rx), math.sin(rx)
	return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _make_cam(pos=(0.30, -0.05, 0.18), tilt_deg=25.0) -> tuple[np.ndarray, np.ndarray]:
	t = math.radians(tilt_deg)
	tilt = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(t), -math.sin(t)], [0.0, math.sin(t), math.cos(t)]])
	w2c = np.eye(4, dtype=np.float64)
	w2c[:3, :3] = np.diag([1.0, -1.0, -1.0]) @ tilt
	w2c[:3, 3] = pos
	k_full = np.array([[420.0, 0.0, 319.5], [0.0, 420.0, 239.5], [0.0, 0.0, 1.0]], dtype=np.float64)
	return k_full, w2c


def _project_pts(k_half: np.ndarray, w2c: np.ndarray, pts_w: np.ndarray) -> np.ndarray:
	c2w = np.linalg.inv(w2c)
	p_c = pts_w @ c2w[:3, :3].T + c2w[:3, 3]
	return (p_c[:, :2] / p_c[:, 2:3]) @ k_half[:2, :2].T + k_half[:2, 2]


def _render_synthetic_brick_frame(
	bpos: np.ndarray,
	brot: np.ndarray,
	size: tuple[float, float, float],
	k_full: np.ndarray,
	w2c: np.ndarray,
	*,
	bg_rgb=(170, 170, 170),
	brick_rgb=(220, 25, 25),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	from digital_twin_armigo.perception_nn.labels import brick_corners_world

	k_half = k_full.copy()
	k_half[0, :] *= 0.5
	k_half[1, :] *= 0.5
	cw = brick_corners_world(bpos, brot, size)
	uv = _project_pts(k_half, w2c, cw)

	rgb = np.full((240, 320, 3), bg_rgb, dtype=np.uint8)
	cls = np.zeros((240, 320), dtype=np.uint8)
	top_poly = np.round(uv[:4]).astype(np.int32)
	bot_poly = np.round(uv[4:]).astype(np.int32)
	cv2.fillPoly(rgb, [bot_poly], (180, 20, 20))
	cv2.fillPoly(rgb, [top_poly], brick_rgb)
	cv2.fillPoly(cls, [bot_poly], 2)
	cv2.fillPoly(cls, [top_poly], 1)

	corners = np.zeros((8, 3), dtype=np.float32)
	corners[:, :2] = uv.astype(np.float32)
	corners[:4, 2] = 1.0
	corners[4:, 2] = 1.0
	return rgb, cls, corners


def test_make_brickseg_dataset_records_native_aux_metadata():
	import make_brickseg_dataset as mbd

	from digital_twin_armigo.agent.twin_mujoco import sample_case

	items, _dropped = mbd._generate_case(3_100_001, "val")
	assert len(items) >= 2
	rng = np.random.default_rng([3_100_001, 0xB51C])
	pos_flag = bool(int(rng.integers(0, 2)))
	face = int(rng.integers(0, 6))
	layout = sample_case(
		3_100_001, positive=pos_flag, face=face, options={"scene": "v3", "cameras": "wide", "split": "val"}
	)
	expected_sat = colorsys.rgb_to_hsv(*layout.table_rgb)[1]

	seen_views = set()
	for _rgb, _cls, _cor, raw_meta in items:
		m = json.loads(raw_meta)
		seen_views.add(m["view_kind"])
		assert "table_saturation" in m
		assert math.isclose(m["table_saturation"], expected_sat, abs_tol=1e-5)
		assert m["table_sat_bin"] in ("low", "mid", "high")
		assert isinstance(m["same_color_clutter"], bool)
		assert m["held"] is (m["view_kind"] == "top_present")
		w2g = np.asarray(m["world_from_gripper"], dtype=np.float64)
		g2w = np.asarray(m["gripper_from_world"], dtype=np.float64)
		assert w2g.shape == (4, 4) and g2w.shape == (4, 4)
		assert np.allclose(g2w @ w2g, np.eye(4), atol=1e-4)
		w2b = np.eye(4, dtype=np.float64)
		w2b[:3, :3] = np.asarray(m["brick_rot"], dtype=np.float64)
		w2b[:3, 3] = np.asarray(m["brick_pos"], dtype=np.float64)
		g2b = g2w @ w2b
		assert np.allclose(m["brick_in_gripper_pos"], g2b[:3, 3], atol=1e-4)
		assert np.allclose(m["brick_in_gripper_rot"], g2b[:3, :3], atol=1e-4)
	assert "top_table" in seen_views


def test_same_color_clutter_respects_hue_and_achromatic_exclusion():
	import make_brickseg_dataset as mbd

	from digital_twin_armigo.mujoco_cell.layout import Layout

	def _layout(brick_rgb, clutter=(), distractors=()):
		return Layout(
			green=(0.2, 0.15),
			pink=(0.2, -0.15),
			brick_xy=(0.25, 0.0),
			brick_yaw_deg=0.0,
			defective=False,
			brick_rgb=brick_rgb,
			clutter=list(clutter),
			distractors=list(distractors),
		)

	red = (0.85, 0.1, 0.1)
	blue = (0.1, 0.45, 0.85)
	orange = (0.95, 0.4, 0.05)  # ~23 deg from red -> within 35 deg
	white = (0.97, 0.97, 0.97)  # achromatic (hue=0 in HSV, sat=0)
	black = (0.09, 0.09, 0.09)

	assert mbd.has_same_color_clutter(_layout(red, clutter=[(0.2, 0.15, 0.0, *red)])) is True
	assert mbd.has_same_color_clutter(_layout(red, clutter=[(0.2, 0.15, 0.0, *orange)])) is True
	assert mbd.has_same_color_clutter(_layout(red, clutter=[(0.2, 0.15, 0.0, *blue)])) is False
	assert (
		mbd.has_same_color_clutter(_layout(red, clutter=[(0.2, 0.15, 0.0, *white), (0.2, -0.15, 0.0, *black)])) is False
	)

	# Distractor whose rendered RGB (0.2+0.6*((x*37)%1), 0.2+0.6*((y*53)%1), 0.5) is blue-ish matches blue brick
	x_blue = 0.0 / 37.0  # r = 0.2
	y_blue = 0.2 / 53.0  # g = 0.32, b = 0.5 -> hue ~ 216 deg, sat = 0.6 -> close to blue (~212 deg)
	assert mbd.has_same_color_clutter(_layout(blue, distractors=[(x_blue, y_blue, 0.02, 0.02, 0.02, 0.0)])) is True
	assert mbd.has_same_color_clutter(_layout(red, distractors=[(x_blue, y_blue, 0.02, 0.02, 0.02, 0.0)])) is False


def test_reconstruct_brickseg_aux_aligns_and_detects_tampering(tmp_path: Path):
	import make_brickseg_dataset as mbd
	import reconstruct_brickseg_aux as rba

	ds_dir = tmp_path / "mini_ds"
	mbd.generate_dataset(ds_dir, train_cases=0, val_cases=1, ood_cases=1, cases_per_shard=1, workers=1)

	# Build a legacy-style copy without aux keys to verify lossless reconstruction
	legacy_dir = tmp_path / "legacy_ds"
	legacy_dir.mkdir()
	legacy_dir.joinpath("manifest.json").write_text(
		(ds_dir / "manifest.json").read_text(encoding="utf-8"), encoding="utf-8"
	)
	native_aux: dict[str, dict] = {}
	for spath in sorted(ds_dir.glob("shard_*.npz")):
		with np.load(spath, allow_pickle=False) as z:
			stripped = []
			for idx, raw in enumerate(z["meta"]):
				m = json.loads(str(raw))
				native_aux[f"{spath.name}:{idx}"] = m
				stripped.append(
					json.dumps(
						{
							k: m[k]
							for k in (
								"seed",
								"split",
								"camera",
								"view_kind",
								"brick_pos",
								"brick_rot",
								"k",
								"world_from_cam",
								"brick_size",
							)
						},
						separators=(",", ":"),
					)
				)
			np.savez_compressed(
				legacy_dir / spath.name,
				rgb=z["rgb"],
				cls=z["cls"],
				corners=z["corners"],
				meta=np.asarray(stripped),
			)

	out_json = tmp_path / "aux_meta.json"
	res = rba.reconstruct_aux_metadata(legacy_dir, out_json, splits=("val", "ood-appearance"), workers=1)
	assert out_json.is_file()
	al = res["alignment"]
	assert al["total_frames"] == len(native_aux) and al["total_frames"] > 0
	assert al["mismatched_frames"] == 0
	assert al["aligned_frames"] == al["total_frames"]
	assert al["max_pos_diff_m"] < 1e-4
	assert al["max_rot_diff"] < 1e-3
	assert al["max_cam_diff"] < 1e-4

	for key, orig in native_aux.items():
		rec = res["frames"][key]
		assert math.isclose(rec["table_saturation"], orig["table_saturation"], abs_tol=1e-5)
		assert rec["table_sat_bin"] == orig["table_sat_bin"]
		assert rec["same_color_clutter"] == orig["same_color_clutter"]
		assert rec["held"] == orig["held"]
		assert np.allclose(rec["gripper_from_world"], orig["gripper_from_world"], atol=1e-4)

	# Tamper with brick_pos in one frame -> must detect mismatch
	tampered_dir = tmp_path / "tampered_ds"
	tampered_dir.mkdir()
	tampered_dir.joinpath("manifest.json").write_text(
		(legacy_dir / "manifest.json").read_text(encoding="utf-8"), encoding="utf-8"
	)
	for idx_s, spath in enumerate(sorted(legacy_dir.glob("shard_*.npz"))):
		with np.load(spath, allow_pickle=False) as z:
			metas = [json.loads(str(raw)) for raw in z["meta"]]
			if idx_s == 0:
				metas[0]["brick_pos"][0] += 0.02  # 20 mm tamper
			np.savez_compressed(
				tampered_dir / spath.name,
				rgb=z["rgb"],
				cls=z["cls"],
				corners=z["corners"],
				meta=np.asarray([json.dumps(m, separators=(",", ":")) for m in metas]),
			)
	tampered_res = rba.reconstruct_aux_metadata(
		tampered_dir, tmp_path / "aux_tampered.json", splits=("val", "ood-appearance"), workers=1
	)
	assert tampered_res["alignment"]["mismatched_frames"] == 1
	assert tampered_res["alignment"]["max_pos_diff_m"] >= 0.019


def test_held_pose_error_in_gripper_frame_matches_hand_calculation():
	import eval_brick_pose as ebp

	# Gripper rotated +90 deg about Z and translated to (0.25, -0.05, 0.14)
	w2g = np.eye(4, dtype=np.float64)
	w2g[:3, :3] = _rot_z(math.pi / 2.0)
	w2g[:3, 3] = [0.25, -0.05, 0.14]
	g2w = np.linalg.inv(w2g)

	true_pos = np.array([0.26, -0.04, 0.12], dtype=np.float64)
	true_rot = _rot_x(math.pi) @ _rot_z(math.radians(30.0))  # flipped upside down
	# Perturb predicted position by (+4 mm in world X, +3 mm in world Y, 0 in world Z)
	# Under g2w = Rz(-90 deg): world +X -> gripper -Y, world +Y -> gripper +X!
	pred_pos = true_pos + np.array([0.004, 0.003, 0.0], dtype=np.float64)
	pred_rot = true_rot @ _rot_z(math.radians(90.0))  # 90 deg z-rotation in brick local frame

	pos_mm_sq, rot_deg_sq, err_vec_sq = ebp.held_pose_error_in_gripper(
		pred_pos, pred_rot, true_pos, true_rot, g2w, BRICK_SIZE_2X2
	)
	assert math.isclose(pos_mm_sq, 5.0, abs_tol=1e-4)
	assert np.allclose(err_vec_sq, [3.0, -4.0, 0.0], atol=1e-4)
	# Square 2x2 brick is 90-deg symmetric around local Z -> 0 deg error!
	assert math.isclose(rot_deg_sq, 0.0, abs_tol=1e-4)

	# Rectangular 2x4 brick is only 180-deg symmetric around local Z -> 90 deg error!
	pos_mm_rect, rot_deg_rect, _ = ebp.held_pose_error_in_gripper(
		pred_pos, pred_rot, true_pos, true_rot, g2w, BRICK_SIZE_2X4
	)
	assert math.isclose(pos_mm_rect, 5.0, abs_tol=1e-4)
	assert math.isclose(rot_deg_rect, 90.0, abs_tol=1e-4)


def test_eval_brick_pose_evaluates_top_flipped_held_strata_shared_priors_and_prereg(tmp_path: Path):
	import eval_brick_pose as ebp

	k_full, w2c_wrist = _make_cam(pos=(0.28, -0.04, 0.15), tilt_deg=20.0)
	_, w2c_top = _make_cam(pos=(0.28, -0.08, 0.35), tilt_deg=25.0)
	w2g = np.eye(4, dtype=np.float64)
	w2g[:3, :3] = _rot_z(math.radians(45.0))
	w2g[:3, 3] = [0.25, 0.02, 0.16]
	g2w = np.linalg.inv(w2g)

	specs = [
		# (shard, split, camera, view_kind, pos, rot, sat, scc, held)
		("shard_0040.npz", "val", "wrist_cam", "wrist_0", [0.28, 0.01, 0.0096], _rot_z(0.2), 0.08, False, False),
		("shard_0040.npz", "val", "wrist_cam", "wrist_1", [0.28, 0.01, 0.0096], _rot_z(0.4), 0.25, True, False),
		("shard_0040.npz", "val", "top", "top_table", [0.28, 0.04, 0.0096], _rot_z(0.1), 0.10, False, False),
		("shard_0040.npz", "val", "top", "top_present", [0.28, 0.04, 0.12], _rot_z(0.3), 0.22, True, True),
		(
			"shard_0040.npz",
			"val",
			"top",
			"top_present",
			[0.28, 0.04, 0.12],
			_rot_x(math.pi) @ _rot_z(0.3),
			0.28,
			False,
			True,
		),
		(
			"shard_0045.npz",
			"ood-appearance",
			"wrist_cam",
			"wrist_0",
			[0.28, 0.01, 0.0096],
			_rot_z(0.2),
			0.42,
			False,
			False,
		),
		(
			"shard_0045.npz",
			"ood-appearance",
			"wrist_cam",
			"wrist_1",
			[0.28, 0.01, 0.0096],
			_rot_z(-0.2),
			0.50,
			True,
			False,
		),
		("shard_0045.npz", "ood-appearance", "top", "top_table", [0.28, 0.04, 0.0096], _rot_z(0.1), 0.45, True, False),
		(
			"shard_0045.npz",
			"ood-appearance",
			"top",
			"top_present",
			[0.28, 0.04, 0.12],
			_rot_x(math.pi / 2.0) @ _rot_z(0.5),
			0.48,
			False,
			True,
		),
	]

	by_shard: dict[str, list] = {}
	for item in specs:
		by_shard.setdefault(item[0], []).append(item)

	aux_frames: dict[str, dict] = {}
	for sname, items in by_shard.items():
		rgbs, clss, cors, metas = [], [], [], []
		for idx, (_, split, cam, vkind, pos, rot, sat, scc, held) in enumerate(items):
			w2c = w2c_wrist if cam == "wrist_cam" else w2c_top
			bpos = np.asarray(pos, dtype=np.float64)
			rgb, cls_m, cor = _render_synthetic_brick_frame(bpos, rot, BRICK_SIZE_2X2, k_full, w2c)
			rgbs.append(rgb)
			clss.append(cls_m)
			cors.append(cor)
			metas.append(
				json.dumps(
					{
						"seed": 3100000 + idx,
						"split": split,
						"camera": cam,
						"view_kind": vkind,
						"brick_pos": bpos.tolist(),
						"brick_rot": rot.tolist(),
						"k": k_full.tolist(),
						"world_from_cam": w2c.tolist(),
						"brick_size": list(BRICK_SIZE_2X2),
					}
				)
			)
			aux_frames[f"{sname}:{idx}"] = {
				"shard": sname,
				"index": idx,
				"seed": 3100000 + idx,
				"split": split,
				"camera": cam,
				"view_kind": vkind,
				"table_saturation": sat,
				"table_sat_bin": "low" if sat < 0.15 else ("mid" if sat <= 0.35 else "high"),
				"same_color_clutter": scc,
				"held": held,
				"world_from_gripper": w2g.tolist(),
				"gripper_from_world": g2w.tolist(),
			}
		np.savez_compressed(
			tmp_path / sname,
			rgb=np.stack(rgbs),
			cls=np.stack(clss),
			corners=np.stack(cors),
			meta=np.asarray(metas),
		)

	aux_path = tmp_path / "aux_meta.json"
	aux_path.write_text(json.dumps({"schema_version": 1, "frames": aux_frames}), encoding="utf-8")
	pf_path = tmp_path / "per_frame.jsonl"

	res = ebp.evaluate_shards(tmp_path, TINY_ONNX, TINY_META, aux_meta_path=aux_path, seed=7, per_frame_out=pf_path)

	# 1. Exact frame counts across camera & orientation buckets (no top skip!)
	sum_map = {(r["split"], r["camera"], r["orientation"], r["method"]): r for r in res["summary"]}
	assert sum_map[("val", "wrist_cam", "upright", "lab")]["n"] == 2
	assert sum_map[("val", "top", "upright", "lab")]["n"] == 2
	assert sum_map[("val", "top", "flipped", "lab")]["n"] == 1
	assert sum_map[("ood-appearance", "wrist_cam", "upright", "unet-pnp")]["n"] == 2
	assert sum_map[("ood-appearance", "top", "upright", "unet-pnp")]["n"] == 1
	assert sum_map[("ood-appearance", "top", "flipped", "unet-pnp")]["n"] == 1

	# 2. Held-in-gripper summary has exact nonzero counts
	held_map = {(r["split"], r["orientation"], r["method"]): r for r in res["held_summary"]}
	assert held_map[("val", "upright", "lab")]["n"] == 1
	assert held_map[("val", "flipped", "lab")]["n"] == 1
	assert held_map[("ood-appearance", "flipped", "unet-pnp")]["n"] == 1

	# 3. Table saturation & same-color clutter strata
	sat_bins_seen = {r["table_sat_bin"] for r in res["by_table_saturation"]}
	assert {"low", "mid", "high"}.issubset(sat_bins_seen)
	scc_seen = {r["same_color_clutter"] for r in res["by_same_color_clutter"]}
	assert {False, True}.issubset(scc_seen)

	# 4. Per-frame records & JSONL output + shared priors + failure reasons
	assert len(res["per_frame"]) == len(specs) * 3
	jsonl_rows = [json.loads(line) for line in pf_path.read_text(encoding="utf-8").splitlines() if line.strip()]
	assert len(jsonl_rows) == len(specs) * 3

	by_frame: dict[tuple[str, int], list[dict]] = {}
	for rec in jsonl_rows:
		by_frame.setdefault((rec["shard"], rec["index"]), []).append(rec)
		if rec["found"]:
			assert rec["fail_reason"] is None
			assert rec["xy_mm"] is not None and rec["yaw_deg"] is not None
			if rec["held"]:
				assert rec["held_pos_mm"] is not None and rec["held_rot_deg"] is not None
				assert rec["held_err_gripper_mm"] is not None and len(rec["held_err_gripper_mm"]) == 3
		else:
			assert isinstance(rec["fail_reason"], str) and len(rec["fail_reason"]) > 0

	for recs in by_frame.values():
		assert {r["method"] for r in recs} == {"lab", "unet-rect", "unet-pnp"}
		# All 3 methods MUST receive the exact same prior_xy and prior_yaw
		assert recs[0]["prior_xy"] == recs[1]["prior_xy"] == recs[2]["prior_xy"]
		assert recs[0]["prior_yaw"] == recs[1]["prior_yaw"] == recs[2]["prior_yaw"]

	# 5. Camera-stratified val IoU & PCK
	assert "wrist_cam" in res["val_metrics_by_camera"] and "top" in res["val_metrics_by_camera"]

	# 6. Preregistered checks & paired OOD comparison
	checks = res["prereg_checks"]
	for key in (
		"sim_val_top_iou",
		"sim_val_pck_2px",
		"sim_val_wrist_xy_med_mm",
		"sim_val_wrist_yaw_med_deg",
		"sim_val_wrist_rate_pct",
		"sim_ood_paired_vs_lab",
		"all_passed",
	):
		assert key in checks
	assert "unet-pnp" in res["ood_paired_vs_lab"] and "unet-rect" in res["ood_paired_vs_lab"]

	# 7. Markdown report includes all sections
	md = ebp.render_markdown(res)
	for heading in (
		"Preregistered Pass Gate",
		"OOD Paired Comparison",
		"Held-in-Gripper",
		"Table Saturation",
		"Same-Color Clutter",
		"Failure Reasons",
	):
		assert heading in md


def test_eval_brick_pose_unet_pnp_recovers_tilted_held_6d_pose_in_gripper_frame(tmp_path: Path, monkeypatch):
	import eval_brick_pose as ebp

	from digital_twin_armigo.agent.brick_pose_nn import NetOut
	from digital_twin_armigo.perception_nn.labels import brick_corners_world

	k_full, w2c_top = _make_cam(pos=(0.28, -0.10, 0.36), tilt_deg=25.0)
	k_half = k_full.copy()
	k_half[0, :] *= 0.5
	k_half[1, :] *= 0.5

	bpos = np.array([0.26, 0.02, 0.13], dtype=np.float64)
	# Tilted 40 deg (> 30 deg -> classified as flipped/tilted!) so both top and bottom corners are visible
	brot = _rot_z(math.radians(20.0)) @ _rot_x(math.radians(40.0))
	cw = brick_corners_world(bpos, brot, BRICK_SIZE_2X4)
	uv = _project_pts(k_half, w2c_top, cw)

	rgb, cls_m, cor = _render_synthetic_brick_frame(bpos, brot, BRICK_SIZE_2X4, k_full, w2c_top)
	w2g = np.eye(4, dtype=np.float64)
	w2g[:3, :3] = _rot_z(math.radians(-90.0))
	w2g[:3, 3] = [0.24, 0.03, 0.15]
	g2w = np.linalg.inv(w2g)

	# Build a fake BrickSegNet that outputs exact cls and sharp Gaussian corner heatmaps
	yy, xx = np.mgrid[0:240, 0:320]
	heat = np.zeros((2, 240, 320), dtype=np.float32)
	for u, v in uv[:4]:
		heat[0] = np.maximum(heat[0], np.exp(-((xx - u) ** 2 + (yy - v) ** 2) / (2 * 1.5**2)))
	for u, v in uv[4:]:
		heat[1] = np.maximum(heat[1], np.exp(-((xx - u) ** 2 + (yy - v) ** 2) / (2 * 1.5**2)))
	fake_out = NetOut(cls=cls_m.astype(np.int64), heat=heat, scale=(1.0, 1.0))

	class _FakeNet:
		def __init__(self, *_a, **_kw):
			pass

		def predict(self, _rgb):
			return fake_out

	monkeypatch.setattr(ebp, "BrickSegNet", _FakeNet)

	meta = {
		"seed": 3100099,
		"split": "val",
		"camera": "top",
		"view_kind": "top_present",
		"brick_pos": bpos.tolist(),
		"brick_rot": brot.tolist(),
		"k": k_full.tolist(),
		"world_from_cam": w2c_top.tolist(),
		"brick_size": list(BRICK_SIZE_2X4),
		"table_saturation": 0.20,
		"table_sat_bin": "mid",
		"same_color_clutter": False,
		"held": True,
		"world_from_gripper": w2g.tolist(),
		"gripper_from_world": g2w.tolist(),
	}
	np.savez_compressed(
		tmp_path / "shard_0040.npz",
		rgb=rgb[None, ...],
		cls=cls_m[None, ...],
		corners=cor[None, ...],
		meta=np.asarray([json.dumps(meta)]),
	)

	res = ebp.evaluate_shards(tmp_path, TINY_ONNX, TINY_META, seed=0)
	pnp_rec = next(r for r in res["per_frame"] if r["method"] == "unet-pnp")
	assert pnp_rec["flipped"] is True
	assert pnp_rec["held"] is True
	assert pnp_rec["found"] is True
	assert pnp_rec["fail_reason"] is None
	assert pnp_rec["held_pos_mm"] < 2.0
	assert pnp_rec["held_rot_deg"] < 3.0
