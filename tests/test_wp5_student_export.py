"""WP5 TDD suite: unified student input contract, K-bowl swap report, legacy 5-camera merge
compatibility, trainable manifest validation, silver/gold video readiness, and verifiable deferred
video re-rendering."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
	sys.path.insert(0, str(REPO / "src"))

LEROBOT_PYTHON = Path("E:/il/lerobot-cuda/Scripts/python.exe")


def _load_script(rel: str, name: str):
	spec = importlib.util.spec_from_file_location(name, REPO / rel)
	assert spec is not None and spec.loader is not None
	mod = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(mod)
	return mod


def _sha256_file(path: Path) -> str:
	h = hashlib.sha256()
	with path.open("rb") as fh:
		for chunk in iter(lambda: fh.read(1 << 20), b""):
			h.update(chunk)
	return h.hexdigest()


def _snapshot_tree(root: Path) -> dict[str, str]:
	out: dict[str, str] = {}
	for p in sorted(root.rglob("*")):
		if p.is_file():
			out[p.relative_to(root).as_posix()] = _sha256_file(p)
	return out


def _no_gpu_env(extra: dict[str, str] | None = None) -> dict[str, str]:
	env = dict(os.environ)
	env["CUDA_VISIBLE_DEVICES"] = ""
	env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
	if extra:
		env.update(extra)
	return env


def _write_minimal_episode(
	ep_dir: Path,
	n: int = 4,
	*,
	noisy: np.ndarray | None = None,
	video: bool = True,
	supervise_action: np.ndarray | None = None,
) -> None:
	from digital_twin_armigo.agent.twin_graph import TASK
	from digital_twin_armigo.agent.twin_record import CAMERAS, Video
	from digital_twin_armigo.constants import CAMERA_SHAPES, FPS, TOP_FULL

	b0 = ep_dir / "branch-0"
	b0.mkdir(parents=True, exist_ok=True)
	seg_ids = np.zeros(n, dtype=np.int32)
	events = [
		{
			"kind": "segment",
			"frame": 0,
			"id": 0,
			"segment_kind": "carry_to_bowl",
			"instruction": "Carry the held brick over the green bowl and lower it into the bowl.",
			"params": {"bowl": "green"},
		},
		{
			"kind": "segment_end",
			"frame": n,
			"id": 0,
			"frames": n,
			"reason": "done",
		},
	]
	(ep_dir / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
	actions = np.linspace(0.0, 0.2, n * 6, dtype=np.float32).reshape(n, 6)
	states = np.linspace(-0.1, 0.1, n * 6, dtype=np.float32).reshape(n, 6)
	npz_data: dict[str, np.ndarray] = {
		"state": states,
		"action": actions,
		"state_f64": states.astype(np.float64),
		"action_f64": actions.astype(np.float64),
		"segment": seg_ids,
		"gripper_cmd": np.full(n, 0.5, dtype=np.float32),
		"supervise_action": np.ones(n, dtype=bool) if supervise_action is None else supervise_action.astype(bool),
	}
	if noisy is not None:
		npz_data["noisy"] = noisy.astype(bool)
	np.savez_compressed(b0 / "actions.npz", **npz_data)

	if video:
		shapes = {
			"front": CAMERA_SHAPES["front"],
			"wrist": CAMERA_SHAPES["wrist"],
			"top_rgb": [TOP_FULL[1], TOP_FULL[0], 3],
			"top_depth": CAMERA_SHAPES["top_depth_v2"],
		}
		for cam in CAMERAS:
			v = Video(b0 / f"{cam}.mkv", shapes[cam], lossless=True)
			for i in range(n):
				v.put(np.full(shapes[cam], (i * 17) % 255, dtype=np.uint8))
			v.close()

	(ep_dir / "episode.json").write_text(
		json.dumps(
			{
				"schema": "armigo-agent-episode/v1",
				"fps": FPS,
				"task": TASK,
				"branches": [{"id": 0, "start_frame": 0, "actions": n}],
				"video": video,
				"provenance": {"sensing": "realistic", "case": [3000101, False, 0]},
				"result": {"verdict": "SUCCESS", "student_eligible": True, "rewinds": 0, "label": "normal"},
			}
		),
		encoding="utf-8",
	)


# ---------------------------------------------------------------------------
# 1. Unified Student Input Contract (vla/contract.py, server.py, dataset.py)
# ---------------------------------------------------------------------------


class _RecordingDummyModel:
	def __init__(self, *, horizon: int = 4, family: str = "smolvla", zero_bowl: bool | None = None) -> None:
		self.horizon = horizon
		self.family = family
		if zero_bowl is not None:
			self.zero_bowl = zero_bowl
		self.seen_states: list[np.ndarray] = []

	def predict_chunk(self, batch: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
		self.seen_states.append(np.asarray(batch["observation.state"], dtype=np.float32)[0].copy())
		actions = np.zeros((1, self.horizon, 6), dtype=np.float32)
		probs = np.zeros((1, self.horizon + 2), dtype=np.float32)
		probs[0, self.horizon + 1] = 1.0
		return actions, probs


def _dummy_images() -> dict[str, np.ndarray]:
	return {
		"top_rgb_v2": np.zeros((416, 512, 3), dtype=np.uint8),
		"top_depth_v2": np.zeros((416, 512, 3), dtype=np.uint8),
		"wrist": np.zeros((480, 640, 3), dtype=np.uint8),
	}


def test_unified_student_input_contract_and_train_serve_parity(tmp_path: Path):
	from digital_twin_armigo.vla.client_policy import RemoteChunkPredictor
	from digital_twin_armigo.vla.contract import (
		INITIAL_Q,
		STATE_DIM,
		STUDENT_CAMERA_NAMES,
		STUDENT_INPUT_CONTRACT_V2,
		STUDENT_INPUT_CONTRACT_V3,
		StudentInputContract,
		policy_inputs_from_segment_dict,
	)
	from digital_twin_armigo.vla.dataset import SegmentChunkDataset
	from digital_twin_armigo.vla.server import (
		LocalChunkPredictor,
		VLAPolicyServer,
		build_parser,
		validate_predict_payload,
	)

	assert STUDENT_INPUT_CONTRACT_V3.version == "v3"
	assert STUDENT_INPUT_CONTRACT_V3.zero_bowl is True
	assert STUDENT_INPUT_CONTRACT_V3.cameras == STUDENT_CAMERA_NAMES
	assert STUDENT_INPUT_CONTRACT_V3.state_dim == STATE_DIM

	assert STUDENT_INPUT_CONTRACT_V2.version == "v2"
	assert STUDENT_INPUT_CONTRACT_V2.zero_bowl is False
	assert STUDENT_INPUT_CONTRACT_V2.cameras == STUDENT_CAMERA_NAMES
	assert STUDENT_INPUT_CONTRACT_V2.state_dim == STATE_DIM
	assert StudentInputContract.for_version("v3") == STUDENT_INPUT_CONTRACT_V3
	assert StudentInputContract.for_version("v2") == STUDENT_INPUT_CONTRACT_V2

	# Four frames: prepare, carry_to_bowl(green), carry_to_bowl(pink), carry_to_bowl(purple)
	q_rows = np.array(
		[
			[0.10, -0.20, 0.30, -0.10, 0.05, 0.80],
			[0.12, -0.18, 0.28, -0.09, 0.06, 0.40],
			[0.14, -0.16, 0.26, -0.08, 0.07, 0.40],
			[0.16, -0.14, 0.24, -0.07, 0.08, 0.40],
		],
		dtype=np.float32,
	)
	act_rows = q_rows + 0.01
	prev_act_rows = np.vstack([INITIAL_Q.astype(np.float32)[None, :], act_rows[:-1]])
	grip_rows = np.array([0.85, 0.42, 0.41, 0.40], dtype=np.float32)

	frame_defs = [
		("prepare", "", 0, 0),
		("carry_to_bowl", "green", 1, 0),
		("carry_to_bowl", "pink", 2, 0),
		("carry_to_bowl", "purple", 3, 0),
	]

	def _build_ep(scene: str | None, defs: list[tuple[str, str, int, int]]) -> dict[str, Any]:
		n = len(defs)
		return {
			"episode_index": 0,
			"scene": scene,
			"frames": {
				"observation.state": q_rows[:n],
				"observation.prev_action": prev_act_rows[:n],
				"observation.gripper_cmd": grip_rows[:n],
				"action": act_rows[:n],
				"segment.index": np.array([d[2] for d in defs], dtype=np.int64),
				"segment.kind": [d[0] for d in defs],
				"segment.chunk_valid_len": np.ones(n, dtype=np.int64),
				"segment.frame": np.array([d[3] for d in defs], dtype=np.int64),
				"segment.bowl": [d[1] for d in defs],
			},
		}

	ds_v3 = SegmentChunkDataset.from_episodes([_build_ep("v3", frame_defs)], horizon=4, return_torch=False)
	ds_v2 = SegmentChunkDataset.from_episodes([_build_ep(None, frame_defs[:3])], horizon=4, return_torch=False)

	imgs = _dummy_images()
	model_v3 = _RecordingDummyModel(zero_bowl=True)
	pred_v3 = LocalChunkPredictor(model_v3)
	assert pred_v3.zero_bowl is True
	h3 = pred_v3.health()
	assert h3["zero_bowl"] is True
	assert h3["input_contract"]["version"] == "v3"
	assert h3["input_contract"]["zero_bowl"] is True

	model_v2 = _RecordingDummyModel()
	pred_v2 = LocalChunkPredictor(model_v2)
	assert pred_v2.zero_bowl is False
	assert pred_v2.health()["zero_bowl"] is False
	assert pred_v2.health()["input_contract"]["version"] == "v2"

	# Verify byte/float-identical states across dataset, validate_predict_payload, LocalChunkPredictor, and HTTP /predict
	with (
		VLAPolicyServer(pred_v3, host="127.0.0.1", port=0) as srv_v3,
		RemoteChunkPredictor(srv_v3.url) as rem_v3,
		VLAPolicyServer(pred_v2, host="127.0.0.1", port=0) as srv_v2,
		RemoteChunkPredictor(srv_v2.url) as rem_v2,
	):
		assert rem_v3.health()["zero_bowl"] is True
		assert rem_v3.health()["input_contract"]["version"] == "v3"
		assert rem_v2.health()["zero_bowl"] is False

		for i, (kind, bowl, _sid, sframe) in enumerate(frame_defs):
			task = policy_inputs_from_segment_dict(kind, {"bowl": bowl} if bowl else {})
			q_prev = q_rows[0] if i == 0 else q_rows[i - 1]
			payload = {
				**imgs,
				"q_t": q_rows[i],
				"q_prev": q_prev,
				"prev_action": prev_act_rows[i],
				"gripper_cmd": float(grip_rows[i]),
				"task": task,
				"segment_frame": sframe,
			}

			# v3: state[31:34] == [0, 0, 0] for ALL four (prepare, green, pink, purple)
			offline_v3 = ds_v3[i]["observation.state"]
			assert offline_v3[31:34].tolist() == [0.0, 0.0, 0.0]

			prep_v3 = validate_predict_payload(payload, zero_bowl=True)
			assert np.array_equal(prep_v3["state"], offline_v3)

			# Payload scene/input_contract hint works when zero_bowl is not forced
			prep_hint = validate_predict_payload({**payload, "scene": "v3"})
			assert np.array_equal(prep_hint["state"], offline_v3)
			prep_ic = validate_predict_payload({**payload, "input_contract": "v3"})
			assert np.array_equal(prep_ic["state"], offline_v3)

			# Explicit zero_bowl=False overrides payload hint
			prep_forced_v2 = validate_predict_payload({**payload, "scene": "v3"}, zero_bowl=False)
			if i < 3:
				offline_v2 = ds_v2[i]["observation.state"]
				expected_onehot = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])[i]
				assert offline_v2[31:34].tolist() == expected_onehot
				assert np.array_equal(prep_forced_v2["state"], offline_v2)

			# HTTP /predict via v3 server matches offline_v3 bit-for-bit
			rem_v3.predict(payload)
			assert np.array_equal(model_v3.seen_states[-1], offline_v3)

			# HTTP /predict via v2 server matches offline_v2 bit-for-bit (and ignores any v3 hint on v2 predictor)
			if i < 3:
				rem_v2.predict(payload)
				assert np.array_equal(model_v2.seen_states[-1], ds_v2[i]["observation.state"])

	# Verify checkpoint contract metadata inference helper + CLI flags
	from digital_twin_armigo.vla.server import _infer_checkpoint_contract

	ckpt_contract = tmp_path / "ckpt_contract"
	ckpt_contract.mkdir()
	(ckpt_contract / "contract.json").write_text(json.dumps({"version": "v3", "zero_bowl": True}), encoding="utf-8")
	assert _infer_checkpoint_contract(ckpt_contract, _RecordingDummyModel()).zero_bowl is True

	ckpt_cfg = tmp_path / "ckpt_cfg"
	ckpt_cfg.mkdir()
	(ckpt_cfg / "vla_config.json").write_text(json.dumps({"extra": {"scene": "v3"}}), encoding="utf-8")
	assert _infer_checkpoint_contract(ckpt_cfg, _RecordingDummyModel()).zero_bowl is True

	parsed_v3 = build_parser().parse_args(["--ckpt", str(ckpt_cfg), "--v3"])
	assert parsed_v3.v3 is True
	parsed_zb = build_parser().parse_args(["--ckpt", str(ckpt_cfg), "--zero-bowl"])
	assert parsed_zb.zero_bowl is True


# ---------------------------------------------------------------------------
# 2. Generalized K-bowl Probe Report (scripts/vla/bowl_swap_report.py)
# ---------------------------------------------------------------------------


def test_bowl_swap_report_k_bowls_and_legacy_2bowl():
	bsr = _load_script("scripts/vla/bowl_swap_report.py", "bowl_swap_report_wp5")

	# Exact 3-bowl counterexample from independent review
	review_file = Path("C:/scratch/v3-dev/independent-review/sol/probe-report-input.json")
	if review_file.is_file():
		k3_row = json.loads(review_file.read_text(encoding="utf-8"))
	else:
		k3_row = {
			"probe": {
				"bowls_true": {"blue": [0.2, 0.12], "black": [0.22, -0.12], "white": [0.15, 0.25]},
				"requested": "blue",
				"arms": [
					{"arm": "req", "bowl": "blue", "tcp_end": [0.2, 0.12, 0.2], "lost": False, "error": None},
					{"arm": "req2", "bowl": "blue", "tcp_end": [0.2, 0.12, 0.2], "lost": False, "error": None},
					{"arm": "swap", "bowl": "black", "tcp_end": [0.22, -0.12, 0.2], "lost": False, "error": None},
					{"arm": "extra:0", "bowl": "white", "tcp_end": [0.15, 0.25, 0.2], "lost": False, "error": None},
				],
				"start": {"tcp": [0.0, 0.0, 0.2]},
			},
			"case": [3100005, True, 0, {"scene": "v3", "split": "val"}],
			"index": 0,
		}

	c_follow = bsr.classify(k3_row["probe"])
	assert c_follow["class"] == "follow"
	assert c_follow["left_bowl"] == "white"
	assert c_follow["bowl_order"] == ["white", "blue", "black"]
	assert c_follow["dist_to_target_m"] == {
		"req": pytest.approx(0.0),
		"req2": pytest.approx(0.0),
		"swap": pytest.approx(0.0),
		"extra:0": pytest.approx(0.0),
	}
	assert c_follow["req_req2_agree"] is True
	assert c_follow["req_req2_tcp_m"] == pytest.approx(0.0)

	s_k3 = bsr.summarize([k3_row])
	assert s_k3["classes"]["follow"] == 1
	assert s_k3["verdict"] == ["D1_follows"]
	assert "extra:0" in s_k3["reached"] and s_k3["reached"]["extra:0"] == 1
	md_k3 = bsr.markdown(s_k3)
	assert "D1_follows" in md_k3

	# K=3 fixed, other, mixed
	k3_fixed = json.loads(json.dumps(k3_row))
	for a in k3_fixed["probe"]["arms"]:
		a["tcp_end"] = [0.15, 0.25, 0.2]  # all go to white
	cf = bsr.classify(k3_fixed["probe"])
	assert cf["class"] == "fixed" and cf["fixed_bowl"] == "white"

	k3_other = json.loads(json.dumps(k3_row))
	k3_other["probe"]["arms"][3]["lost"] = True
	assert bsr.classify(k3_other["probe"])["class"] == "other"

	k3_mixed = json.loads(json.dumps(k3_row))
	k3_mixed["probe"]["arms"][2]["tcp_end"] = [0.15, 0.25, 0.2]  # swap went to white
	k3_mixed["probe"]["arms"][3]["tcp_end"] = [0.22, -0.12, 0.2]  # extra:0 went to black
	assert bsr.classify(k3_mixed["probe"])["class"] in ("mixed", "inverted")

	# Legacy 2-bowl records WITHOUT arm["bowl"] key (green/pink)
	def _legacy_2bowl(r_xy, r2_xy, s_xy, *, req="green", r_lost=False):
		return {
			"case": [101, False, 0],
			"index": 0,
			"probe": {
				"bowls_true": {"green": [0.22, 0.12], "pink": [0.22, -0.12]},
				"requested": req,
				"start": {"tcp": [0.2, 0.0, 0.2]},
				"arms": [
					{"arm": "req", "tcp_end": r_xy, "lost": r_lost, "error": None},
					{"arm": "req2", "tcp_end": r2_xy, "lost": False, "error": None},
					{"arm": "swap", "tcp_end": s_xy, "lost": False, "error": None},
				],
			},
		}

	g_xy, p_xy = [0.22, 0.12, 0.2], [0.22, -0.12, 0.2]
	assert bsr.classify(_legacy_2bowl(g_xy, g_xy, p_xy)["probe"])["class"] == "follow"
	assert bsr.classify(_legacy_2bowl(g_xy, g_xy, g_xy)["probe"])["class"] == "fixed"
	assert bsr.classify(_legacy_2bowl(p_xy, p_xy, g_xy)["probe"])["class"] == "inverted"
	assert bsr.classify(_legacy_2bowl(g_xy, g_xy, p_xy, r_lost=True)["probe"])["class"] == "other"

	s_leg = bsr.summarize(
		[
			_legacy_2bowl(g_xy, g_xy, p_xy),
			_legacy_2bowl(p_xy, p_xy, g_xy),
			_legacy_2bowl(g_xy, g_xy, g_xy),
		]
	)
	assert s_leg["classes"] == {"follow": 1, "inverted": 1, "fixed": 1}
	assert "green/left" in s_leg["req_correct_by_target"]
	assert "pink/right" in s_leg["swap_correct_by_target"]
	assert bsr.markdown(s_leg)


# ---------------------------------------------------------------------------
# 3. Legacy 5-cam Schema Merge Compatibility & Trainable Manifest Check
# ---------------------------------------------------------------------------


def test_legacy_5cam_vs_3cam_noisy_mask_and_real_lerobot_merge(tmp_path: Path):
	pytest.importorskip("av")
	from digital_twin_armigo.agent.twin_record import student_frames

	ep1 = tmp_path / "ep1"
	ep2 = tmp_path / "ep2"
	_write_minimal_episode(ep1, n=4, noisy=None)
	_write_minimal_episode(ep2, n=4, noisy=np.array([False, True, False, False], dtype=bool))

	# Default: 5-camera legacy omits mask.noisy; 3-camera canonical includes mask.noisy
	f_5cam = list(student_frames(ep1, student_cameras_only=False))
	assert "mask.noisy" not in f_5cam[0]
	assert "observation.images.front" in f_5cam[0] and "observation.images.top_full" in f_5cam[0]

	f_3cam = list(student_frames(ep2, student_cameras_only=True))
	assert "mask.noisy" in f_3cam[0]
	assert [bool(f["mask.noisy"][0]) for f in f_3cam] == [False, True, False, False]

	# Explicit overrides work in both directions
	assert "mask.noisy" in next(student_frames(ep1, student_cameras_only=False, include_noisy_mask=True))
	assert "mask.noisy" not in next(student_frames(ep2, student_cameras_only=True, include_noisy_mask=False))

	if not LEROBOT_PYTHON.is_file():
		pytest.skip("lerobot-cuda python not available")

	# Real LeRobot export + merge compatibility check (with CUDA_VISIBLE_DEVICES="")
	legacy_ds = tmp_path / "legacy_ds"
	new_shard = tmp_path / "new_shard"
	merged_ds = tmp_path / "merged_ds"

	# Export two 5-camera shards (default legacy schema without mask.noisy) and merge them via rollout_and_export
	for out_dir, ep in ((legacy_ds, ep1), (new_shard, ep2)):
		subprocess.run(
			[
				str(LEROBOT_PYTHON),
				"-m",
				"digital_twin_armigo.agent.twin_record",
				"--out",
				str(out_dir),
				"--repo-id",
				"local/wp5-legacy-test",
				str(ep),
			],
			check=True,
			cwd=REPO,
			env=_no_gpu_env(),
		)
		info = json.loads((out_dir / "meta/info.json").read_text(encoding="utf-8"))
		assert "mask.noisy" not in info["features"]
		assert "observation.images.front" in info["features"]

	merge_code = (
		"import json, sys; from pathlib import Path; sys.path.insert(0, 'scripts/vla'); "
		"from rollout_and_export import merge_shards_cmd; "
		"from lerobot.datasets.lerobot_dataset import LeRobotDataset; "
		f"merge_shards_cmd(Path(r'{merged_ds}'), 'local/wp5-legacy-test', [Path(r'{legacy_ds}'), Path(r'{new_shard}')]); "
		f"ds = LeRobotDataset('local/wp5-legacy-test', root=r'{merged_ds}', video_backend='pyav'); "
		"ds[0]; ds[len(ds) - 1]; print(json.dumps([ds.num_episodes, ds.num_frames, sorted(ds.meta.features)]))"
	)
	res = subprocess.run(
		[str(LEROBOT_PYTHON), "-c", merge_code],
		check=True,
		cwd=REPO,
		env=_no_gpu_env(),
		capture_output=True,
		text=True,
	)
	n_eps, n_frames, feats = json.loads(res.stdout.strip().splitlines()[-1])
	assert (n_eps, n_frames) == (2, 8)
	assert "mask.noisy" not in feats
	assert "observation.images.front" in feats


def _make_gov_ep(
	root: Path,
	group: str,
	cid: str,
	*,
	seed: int = 3000101,
	verdict: str = "SUCCESS",
	state: str = "done",
	code_commit: str = "abc123",
	has_video_files: bool = False,
	video_flag: bool = False,
	has_motion: bool = False,
) -> Path:
	d = root / "runs" / "b1" / "mujoco" / group / cid
	d.mkdir(parents=True, exist_ok=True)
	(d / "run_status.json").write_text(json.dumps({"state": state}), encoding="utf-8")
	(d / "task.json").write_text(json.dumps({"case": [seed, False, 0]}), encoding="utf-8")
	cam = "fixed" if group.startswith("truth") else group.split("-")[1]
	(d / "episode.json").write_text(
		json.dumps(
			{
				"video": video_flag,
				"result": {"label": "normal" if verdict == "SUCCESS" else "unknown", "verdict": verdict},
				"provenance": {
					"experiment": {
						"batch": "b1",
						"group": group,
						"engine": "mujoco",
						"camera_condition": cam,
						"detector": "v7bot",
						"strategy": "B",
						"case_id": cid,
						"code_commit": code_commit,
					}
				},
			}
		),
		encoding="utf-8",
	)
	(d / "events.jsonl").write_text("", encoding="utf-8")
	if has_video_files:
		b0 = d / "branch-0"
		b0.mkdir(exist_ok=True)
		for name in ("wrist", "top_rgb", "top_depth"):
			(b0 / f"{name}.mkv").write_bytes(b"mkv-placeholder")
	if has_motion:
		rep = d / "replay"
		rep.mkdir(exist_ok=True)
		np.savez_compressed(
			rep / "motion.npz",
			poses=np.zeros((2, 1, 7), dtype=np.float64),
			timestamps=np.array([1 / 30, 2 / 30]),
			node_ids=np.array(["geom_0"]),
		)
	return d


def test_collect_paired_carry_rejects_non_trainable_gold_manifests(tmp_path: Path):
	from digital_twin_armigo.governance import bronze, gold, lake, silver

	cpc = _load_script("scripts/vla/collect_paired_carry.py", "cpc_wp5")

	src = tmp_path / "data"
	freeze = {"code_commit": "abc123", "detector": "v7bot"}
	# Train both_pass (student_train_v3)
	ep_train = _make_gov_ep(src, "realistic-fixed-B-v7bot", "L3000101-N", seed=3000101, verdict="SUCCESS")
	_make_gov_ep(src, "truth-B-v7bot", "L3000101-N", seed=3000101, verdict="SUCCESS")
	# Train truth_only (failure_bank)
	_make_gov_ep(src, "realistic-fixed-B-v7bot", "L3000102-N", seed=3000102, verdict="FAILED")
	_make_gov_ep(src, "truth-B-v7bot", "L3000102-N", seed=3000102, verdict="SUCCESS")
	# Quarantined train (harness_review)
	_make_gov_ep(src, "realistic-fixed-B-v7bot", "L3000103-N", seed=3000103, verdict="FAILED", state="interrupted")
	# Sealed test (eval_test)
	_make_gov_ep(src, "realistic-fixed-B-v7bot", "L3900001-N", seed=3900001, verdict="SUCCESS")
	# Val (eval_val)
	_make_gov_ep(src, "realistic-fixed-B-v7bot", "L3100001-N", seed=3100001, verdict="SUCCESS")

	con = lake.connect(tmp_path / "lake.sqlite")
	bronze.register(con, src, source="v3")
	bid = silver.build_silver(con, freeze=freeze)
	silver.build_pairs(con, bid)
	gov = tmp_path / "gov"

	train_dir = gold.build_product(con, "student_train_v3", silver_build=bid, out_root=gov, freeze=freeze)
	assert cpc.load_manifest_episodes(train_dir / "manifest.jsonl") == [ep_train]

	for non_trainable in ("harness_review", "failure_bank", "eval_test", "eval_val"):
		prod_dir = gold.build_product(con, non_trainable, silver_build=bid, out_root=gov, freeze=freeze)
		with pytest.raises(ValueError, match="trainable|product|split"):
			cpc.load_manifest_episodes(prod_dir / "manifest.jsonl")

	# Also test row-level trainable=False and split="test" and card.md-only detection
	sha_ok = _sha256_file(ep_train / "episode.json")
	m_false = tmp_path / "m_false.jsonl"
	m_false.write_text(
		json.dumps({"path": str(ep_train), "episode_sha256": sha_ok, "trainable": False}) + "\n", encoding="utf-8"
	)
	with pytest.raises(ValueError, match="trainable"):
		cpc.load_manifest_episodes(m_false)

	m_test_split = tmp_path / "m_test.jsonl"
	m_test_split.write_text(
		json.dumps({"path": str(ep_train), "episode_sha256": sha_ok, "split": "test"}) + "\n", encoding="utf-8"
	)
	with pytest.raises(ValueError, match="split|test|trainable"):
		cpc.load_manifest_episodes(m_test_split)

	con.close()


# ---------------------------------------------------------------------------
# 4. Silver/Gold Video Readiness + Verifiable Deferred Video Re-rendering
# ---------------------------------------------------------------------------


def test_silver_and_gold_video_readiness_metadata_and_card(tmp_path: Path):
	from digital_twin_armigo.governance import bronze, gold, lake, silver

	src = tmp_path / "data"
	freeze = {"code_commit": "abc123", "detector": "v7bot"}

	ep_rec = _make_gov_ep(
		src, "realistic-fixed-B-v7bot", "L3000101-N", seed=3000101, video_flag=True, has_video_files=True
	)
	_make_gov_ep(src, "truth-B-v7bot", "L3000101-N", seed=3000101)

	ep_def = _make_gov_ep(src, "realistic-fixed-B-v7bot", "L3000102-N", seed=3000102, video_flag=False, has_motion=True)
	_make_gov_ep(src, "truth-B-v7bot", "L3000102-N", seed=3000102)

	ep_miss = _make_gov_ep(
		src, "realistic-fixed-B-v7bot", "L3000103-N", seed=3000103, video_flag=True, has_video_files=False
	)
	_make_gov_ep(src, "truth-B-v7bot", "L3000103-N", seed=3000103)

	con = lake.connect(tmp_path / "lake.sqlite")
	bronze.register(con, src, source="v3")
	bid = silver.build_silver(con, freeze=freeze)
	silver.build_pairs(con, bid)

	by_path = {
		p: json.loads(rj)
		for p, rj in con.execute("select path, row_json from silver_episode where build_id = ?", (bid,)).fetchall()
	}
	assert by_path[str(ep_rec)]["has_video"] is True
	assert by_path[str(ep_rec)]["video_source"] == "recorded"

	assert by_path[str(ep_def)]["has_video"] is False
	assert by_path[str(ep_def)]["has_motion"] is True
	assert by_path[str(ep_def)]["video_source"] == "deferred_rerender"

	# video=True in episode.json but missing mkv files -> NOT has_video!
	assert by_path[str(ep_miss)]["has_video"] is False
	assert by_path[str(ep_miss)]["has_motion"] is False
	assert by_path[str(ep_miss)]["video_source"] == "missing"

	prod_dir = gold.build_product(con, "student_train_v3", silver_build=bid, out_root=tmp_path / "gov", freeze=freeze)
	items = {
		item["path"]: item
		for item in (
			json.loads(line) for line in (prod_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line
		)
	}
	assert len(items) == 3
	for item in items.values():
		assert item["product"] == "student_train_v3"
		assert item["trainable"] is True

	assert items[str(ep_rec)]["has_video"] is True and items[str(ep_rec)]["video_source"] == "recorded"
	assert (
		items[str(ep_def)]["has_video"] is False
		and items[str(ep_def)]["has_motion"] is True
		and items[str(ep_def)]["video_source"] == "deferred_rerender"
	)
	assert (
		items[str(ep_miss)]["has_video"] is False
		and items[str(ep_miss)]["has_motion"] is False
		and items[str(ep_miss)]["video_source"] == "missing"
	)

	card = (prod_dir / "card.md").read_text(encoding="utf-8")
	assert "Video Readiness" in card
	assert "recorded: 1" in card
	assert "deferred_rerender: 1" in card
	assert "missing: 1" in card
	con.close()


def test_deferred_video_rerender_and_end_to_end_manifest_lerobot_export(tmp_path: Path, monkeypatch):
	pytest.importorskip("mujoco")
	pytest.importorskip("langgraph")
	pytest.importorskip("av")
	pytest.importorskip("cv2")

	from digital_twin_armigo.agent.deferred_video import rerender_student_videos, verify_rerender_dir
	from digital_twin_armigo.agent.student_rollout import run_episode
	from digital_twin_armigo.agent.twin_record import student_frames
	from digital_twin_armigo.governance import bronze, gold, lake, silver
	from digital_twin_armigo.inspection_scorer import FixedScorer

	cpc = _load_script("scripts/vla/collect_paired_carry.py", "cpc_e2e_wp5")

	case = (3000101, True, 0)
	scorer = FixedScorer(lambda _c, _r: 1.0, {"top": 0.0, "wrist_cam": 0.0})
	data_root = tmp_path / "bronze"
	freeze = {"code_commit": "freeze_wp5", "detector": "v7bot"}

	def _slice_tail_episode(ep_dir: Path, tail_frames: int) -> int:
		"""Keep the final ``tail_frames`` real physics frames (preserving the brick settled in the bowl at ``poses[-1]``)."""
		with np.load(ep_dir / "branch-0" / "actions.npz", allow_pickle=False) as z:
			total = len(z["action"])
			start = max(1, total - tail_frames)
			arrs = {k: z[k][start:] for k in z.files}
			n_keep = len(arrs["action"])
		arrs["segment"] = np.zeros(n_keep, dtype=np.int32)
		np.savez_compressed(ep_dir / "branch-0" / "actions.npz", **arrs)

		motion_path = ep_dir / "replay" / "motion.npz"
		if motion_path.is_file():
			with np.load(motion_path, allow_pickle=False) as mz:
				m_arrs = {k: mz[k] for k in mz.files}
			# poses[start - 1] is exact pre-step state for action[start]; poses[-1] is final settled bowl pose (len = n_keep + 1)
			m_arrs["poses"] = m_arrs["poses"][start - 1 :]
			if "timestamps" in m_arrs:
				m_arrs["timestamps"] = m_arrs["timestamps"][start - 1 :]
			np.savez_compressed(motion_path, **m_arrs)

		events = [
			{
				"seq": 0,
				"kind": "segment",
				"node": "place",
				"branch": 0,
				"frame": 0,
				"id": 0,
				"segment_kind": "release_in_bowl",
				"instruction": "Open the gripper to drop the brick into the bowl, lift clear and return to the rest pose.",
				"params": {},
			},
			{
				"seq": 1,
				"kind": "segment_end",
				"node": "place",
				"branch": 0,
				"frame": n_keep,
				"id": 0,
				"segment_kind": "release_in_bowl",
				"frames": n_keep,
				"reason": "done",
			},
		]
		(ep_dir / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
		ep_json = json.loads((ep_dir / "episode.json").read_text(encoding="utf-8"))
		ep_json["branches"][0]["actions"] = n_keep
		ep_json["result"]["sim_frames"] = n_keep
		ep_json["result"]["active_frames"] = n_keep
		(ep_dir / "episode.json").write_text(json.dumps(ep_json, indent=2), encoding="utf-8")
		return n_keep

	def _record_real_ep(group: str, cid: str, *, video: bool, record_motion: bool) -> Path:
		ep_dir = data_root / "runs" / "b1" / "mujoco" / group / cid
		ep_dir.parent.mkdir(parents=True, exist_ok=True)
		if record_motion:
			monkeypatch.setenv("ARMIGO_RERUN", "record")
		else:
			monkeypatch.delenv("ARMIGO_RERUN", raising=False)
		res = run_episode(case, scorer, ep_dir, sensing="realistic", video=video)
		assert res["result"]["verdict"] == "SUCCESS"
		# Add experiment provenance & task/status files like production runner
		(ep_dir / "run_status.json").write_text(json.dumps({"state": "done"}), encoding="utf-8")
		(ep_dir / "task.json").write_text(json.dumps({"case": list(case)}), encoding="utf-8")
		ep_json = json.loads((ep_dir / "episode.json").read_text(encoding="utf-8"))
		ep_json.setdefault("provenance", {})["experiment"] = {
			"batch": "b1",
			"group": group,
			"engine": "mujoco",
			"camera_condition": "fixed",
			"detector": "v7bot",
			"strategy": "B",
			"case_id": cid,
			"code_commit": "freeze_wp5",
		}
		(ep_dir / "episode.json").write_text(json.dumps(ep_json, indent=2), encoding="utf-8")
		return ep_dir

	# 1. Real video-less bronze episode WITH replay/motion.npz
	ep_videoless = _record_real_ep("realistic-fixed-B-v7bot", "L3000101-D0", video=False, record_motion=True)

	# Also verify post-step len(poses) == N indexing on the real prefix (first 6 frames)
	ep_prefix = tmp_path / "prefix_check"
	shutil.copytree(ep_videoless, ep_prefix)
	with np.load(ep_prefix / "branch-0" / "actions.npz", allow_pickle=False) as z:
		p_arrs = {k: z[k][:6] for k in z.files}
	p_arrs["segment"] = np.zeros(6, dtype=np.int32)
	np.savez_compressed(ep_prefix / "branch-0" / "actions.npz", **p_arrs)
	with np.load(ep_prefix / "replay" / "motion.npz", allow_pickle=False) as mz:
		pm_arrs = {k: mz[k] for k in mz.files}
	pm_arrs["poses"] = pm_arrs["poses"][:6]
	pm_arrs["timestamps"] = pm_arrs["timestamps"][:6]
	np.savez_compressed(ep_prefix / "replay" / "motion.npz", **pm_arrs)
	prefix_manifest = rerender_student_videos(ep_prefix, tmp_path / "prefix_rerender", student_cameras_only=True)
	assert prefix_manifest["pose_indexing"] == "post_step_with_warmup_frame0"
	assert prefix_manifest["verified_alignment"] is True
	assert prefix_manifest["num_frames"] == 6

	# Slice ep_videoless to its final 16 real physics frames (pre-step N+1 motion poses, brick settled in pink_bowl)
	n_actions = _slice_tail_episode(ep_videoless, 16)
	assert (ep_videoless / "replay" / "motion.npz").is_file()
	assert not (ep_videoless / "branch-0" / "wrist.mkv").exists()

	# 2. Real recorded-video bronze episode (16 real rendered frames on the same physics trajectory)
	ep_recorded = data_root / "runs" / "b1" / "mujoco" / "realistic-fixed-B-v7bot" / "L3000102-D0"
	shutil.copytree(ep_videoless, ep_recorded)
	rec_vid_tmp = tmp_path / "rec_seed_videos"
	rerender_student_videos(ep_videoless, rec_vid_tmp, student_cameras_only=False)
	for cam in ("front", "wrist", "top_rgb", "top_depth"):
		shutil.copy2(rec_vid_tmp / f"{cam}.mkv", ep_recorded / "branch-0" / f"{cam}.mkv")
	shutil.rmtree(ep_recorded / "replay")
	rec_json = json.loads((ep_recorded / "episode.json").read_text(encoding="utf-8"))
	rec_json["video"] = True
	rec_json["provenance"]["experiment"]["case_id"] = "L3000102-D0"
	(ep_recorded / "episode.json").write_text(json.dumps(rec_json, indent=2), encoding="utf-8")
	(ep_recorded / "task.json").write_text(json.dumps({"case": [3000102, True, 0]}), encoding="utf-8")
	assert (ep_recorded / "branch-0" / "wrist.mkv").is_file()

	# Matching truth episodes so both enter student_train_v3 via both_pass
	_make_gov_ep(data_root, "truth-B-v7bot", "L3000101-D0", seed=3000101, code_commit="freeze_wp5")
	_make_gov_ep(data_root, "truth-B-v7bot", "L3000102-D0", seed=3000102, code_commit="freeze_wp5")
	for cid in ("L3000101-D0", "L3000102-D0"):
		tdir = data_root / "runs" / "b1" / "mujoco" / "truth-B-v7bot" / cid
		(tdir / "task.json").write_text(json.dumps({"case": [int(cid[1:8]), True, 0]}), encoding="utf-8")
		tj = json.loads((tdir / "episode.json").read_text(encoding="utf-8"))
		tj["result"]["label"] = "defective"
		(tdir / "episode.json").write_text(json.dumps(tj), encoding="utf-8")

	snap_videoless_before = _snapshot_tree(ep_videoless)
	snap_recorded_before = _snapshot_tree(ep_recorded)

	# Without video_dir, student_frames must reject video-less episode with clear error
	with pytest.raises(ValueError, match="video|rerender"):
		next(student_frames(ep_videoless, student_cameras_only=True))

	# Cannot write rerender output inside bronze episode_dir
	with pytest.raises(ValueError, match="bronze|inside|episode"):
		rerender_student_videos(ep_videoless, ep_videoless / "bad_out", student_cameras_only=True)

	# Direct verifiable rerender to external cache dir
	cache_dir = tmp_path / "manual_rerender"
	manifest = rerender_student_videos(ep_videoless, cache_dir, student_cameras_only=True)
	assert manifest["verified_alignment"] is True
	assert manifest["cameras"] == ["wrist", "top_rgb", "top_depth"]
	assert manifest["episode_sha256"] == _sha256_file(ep_videoless / "episode.json")
	assert manifest["actions_sha256"] == _sha256_file(ep_videoless / "branch-0" / "actions.npz")
	assert manifest["motion_sha256"] == _sha256_file(ep_videoless / "replay" / "motion.npz")
	assert set(manifest["mkv_sha256"]) == {"wrist", "top_rgb", "top_depth"}

	# Bronze episode must be 100% untouched
	assert _snapshot_tree(ep_videoless) == snap_videoless_before

	# student_frames with verified video_dir yields all frames matching action count
	assert manifest["num_frames"] == n_actions
	frames = list(student_frames(ep_videoless, student_cameras_only=True, video_dir=cache_dir))
	assert len(frames) == n_actions
	assert frames[0]["observation.images.top_rgb_v2"].shape == (416, 512, 3)
	assert frames[0]["observation.images.top_depth_v2"].shape == (416, 512, 3)
	assert frames[0]["observation.images.wrist"].shape == (480, 640, 3)

	# Tamper detection on rerender_manifest / mkv
	bad_cache = tmp_path / "tampered_rerender"
	shutil.copytree(cache_dir, bad_cache)
	(bad_cache / "wrist.mkv").write_bytes((bad_cache / "wrist.mkv").read_bytes() + b"\x00")
	with pytest.raises(ValueError, match="sha256|mismatch|hash"):
		verify_rerender_dir(ep_videoless, bad_cache, required_cameras=("wrist", "top_rgb", "top_depth"))
	with pytest.raises(ValueError, match="camera|front"):
		verify_rerender_dir(ep_videoless, cache_dir, required_cameras=("front", "wrist", "top_rgb", "top_depth"))

	# Build real gold student_train_v3 manifest containing BOTH recorded and video-less episodes
	con = lake.connect(tmp_path / "lake.sqlite")
	bronze.register(con, data_root, source="v3")
	bid = silver.build_silver(con, freeze=freeze)
	silver.build_pairs(con, bid)
	gold_dir = gold.build_product(con, "student_train_v3", silver_build=bid, out_root=tmp_path / "gov", freeze=freeze)
	con.close()

	manifest_rows = [
		json.loads(line) for line in (gold_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line
	]
	assert {r["path"] for r in manifest_rows} == {str(ep_videoless), str(ep_recorded)}

	if not LEROBOT_PYTHON.is_file():
		pytest.skip("lerobot-cuda python not available")

	# End-to-end collect_paired_carry --manifest --rerender-video -> real LeRobot shard + pyav readback (CPU only!)
	export_out = tmp_path / "e2e_export"
	monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
	rc = cpc.main(
		[
			"--manifest",
			str(gold_dir / "manifest.jsonl"),
			"--out",
			str(export_out),
			"--export-jobs",
			"1",
			"--rerender-video",
			"--lerobot-python",
			str(LEROBOT_PYTHON),
		]
	)
	assert rc == 0
	shards = sorted(d for d in (export_out / "shards").glob("shard-*") if d.is_dir())
	assert len(shards) == 1
	assert (export_out / "rerender_cache").is_dir()

	# Read back real LeRobot dataset via pyav (with CUDA_VISIBLE_DEVICES="")
	readback = (
		"import json; from lerobot.datasets.lerobot_dataset import LeRobotDataset; "
		f"ds = LeRobotDataset('local/armigo-six-face', root=r'{shards[0]}', video_backend='pyav'); "
		"s0 = ds[0]; s1 = ds[len(ds) - 1]; "
		"print(json.dumps({'eps': ds.num_episodes, 'frames': ds.num_frames, 'cams': sorted(k for k in s0 if k.startswith('observation.images.'))}))"
	)
	got = subprocess.run(
		[str(LEROBOT_PYTHON), "-c", readback],
		check=True,
		cwd=REPO,
		env=_no_gpu_env(),
		capture_output=True,
		text=True,
	)
	info = json.loads(got.stdout.strip().splitlines()[-1])
	assert info["eps"] == 2
	assert info["frames"] == n_actions * 2
	assert info["cams"] == [
		"observation.images.top_depth_v2",
		"observation.images.top_rgb_v2",
		"observation.images.wrist",
	]

	# Bronze source files (both video-less AND recorded mkv) MUST remain 100% untouched!
	assert _snapshot_tree(ep_videoless) == snap_videoless_before
	assert _snapshot_tree(ep_recorded) == snap_recorded_before
