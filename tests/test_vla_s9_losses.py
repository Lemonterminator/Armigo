"""Unit and integration tests for S9 VLA training package, losses, masking, and CLI.

Python environments:
  - Repo ``.venv`` (numpy, pytest, mujoco, no torch/lerobot):
      ``E:\\digital_twin_armigo\\.venv\\Scripts\\python.exe -m pytest tests/test_vla_s9_losses.py -v``
      Executes all pure-numpy Data Contract v1, 36-dim state encoding, dataset masking / ``let_go`` synthesis /
      takeover cutoff / normalizer, and End-Step F1 metrics tests (torch/lerobot tests skip cleanly).
  - CUDA training env ``E:\\il\\lerobot-cuda`` (torch 2.11+cu130, lerobot 0.6.0, transformers, RTX 5090):
      ``E:\\il\\lerobot-cuda\\Scripts\\python.exe tests/test_vla_s9_losses.py``
      Executes EVERY test including real LeRobot v3 video dataset loading, ACTWithEndHead & SmolVLAWithEndHead
      forward+backward gradient checks, CUDA bf16 AMP, checkpoint round-trip, and ``scripts/vla/train.py`` CLI smoke runs.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

try:
	import pytest
except ImportError:  # pragma: no cover - E:\il\lerobot-cuda runs directly without pytest installed

	class _PytestFallback:
		@staticmethod
		def importorskip(modname: str):
			return importlib.import_module(modname)

		@staticmethod
		@contextlib.contextmanager
		def raises(exc_type, match: str | None = None):
			try:
				yield
			except exc_type as exc:
				if match is not None and not re.search(match, str(exc)):
					raise AssertionError(f"{exc!r} did not match {match!r}") from exc
			else:
				raise AssertionError(f"Expected {exc_type.__name__} to be raised")

	pytest = _PytestFallback()

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
	sys.path.insert(0, str(ROOT / "src"))

from digital_twin_armigo.vla.contract import (
	ACTION_DIM,
	BOWL_VOCAB,
	IGNORE_INDEX,
	INITIAL_Q,
	SEGMENT_KINDS_ORDER,
	SEGMENT_TEMPLATES,
	STATE_DIM,
	STUDENT_CAMERAS,
	ActionNormalizer,
	encode_state,
	encode_state_from_task,
	end_step_target_from_segment,
)
from digital_twin_armigo.vla.dataset import SegmentChunkDataset, StudentImageAugmentor, vla_collate_fn
from digital_twin_armigo.vla.metrics import (
	EvalAccumulator,
	compute_end_step_metrics,
	compute_vla_metrics,
)


def _make_synthetic_episode(
	ep_idx: int,
	segments: list[tuple[str, int, int, str, dict]],
	*,
	is_blind: list[bool] | None = None,
	is_takeover: list[bool] | None = None,
	supervise_action: list[bool] | None = None,
	img_shape: tuple[int, int] = (64, 64),
) -> dict:
	"""Build an in-memory episode adhering strictly to Data Contract v1."""
	total_frames = max(end for _, _, end, _, _ in segments)
	rng = np.random.default_rng(1000 + ep_idx)
	states = (np.arange(total_frames, dtype=np.float32)[:, None] * 0.01) + rng.normal(
		0.0, 0.001, size=(total_frames, ACTION_DIM)
	).astype(np.float32)
	actions = states + 0.05
	prev_actions = np.vstack([INITIAL_Q.astype(np.float32)[None, :], actions[:-1]])
	gripper_cmd = prev_actions[:, 5].copy()

	seg_idx_col = np.full(total_frames, -1, dtype=np.int64)
	seg_kind_col = [""] * total_frames
	seg_done_col = np.zeros(total_frames, dtype=bool)
	cvl_col = np.zeros(total_frames, dtype=np.int64)
	seg_frame_col = np.zeros(total_frames, dtype=np.int64)
	end_reason_col = [""] * total_frames
	bowl_col = [""] * total_frames
	view_col = np.zeros(total_frames, dtype=np.int64)
	views_col = np.zeros(total_frames, dtype=np.int64)
	task_col = [""] * total_frames

	segments_meta = []
	for sid, (kind, start, end, reason, params) in enumerate(segments):
		instruction = SEGMENT_TEMPLATES[kind].format(**params)
		segments_meta.append(
			{
				"id": sid,
				"kind": kind,
				"instruction": instruction,
				"params": dict(params),
				"start": start,
				"end": end,
				"reason": reason,
			}
		)
		if reason == "done" and end < total_frames:
			seg_done_col[end] = True
		for r in range(start, end):
			seg_idx_col[r] = sid
			seg_kind_col[r] = kind
			cvl_col[r] = end - r
			seg_frame_col[r] = r - start
			end_reason_col[r] = reason
			bowl_col[r] = "" if params.get("bowl") is None else str(params["bowl"])
			view_col[r] = 0 if params.get("view") is None else int(params["view"])
			views_col[r] = 0 if params.get("views") is None else int(params["views"])
			task_col[r] = instruction

	assert (seg_idx_col >= 0).all()
	h, w = img_shape
	images = {cam: rng.integers(0, 255, size=(total_frames, h, w, 3), dtype=np.uint8) for cam in STUDENT_CAMERAS}
	return {
		"episode_index": ep_idx,
		"segments": segments_meta,
		"images": images,
		"frames": {
			"observation.state": states,
			"observation.prev_action": prev_actions,
			"observation.gripper_cmd": gripper_cmd[:, None],
			"action": actions,
			"segment.index": seg_idx_col,
			"segment.kind": seg_kind_col,
			"segment.done": seg_done_col,
			"segment.chunk_valid_len": cvl_col,
			"segment.frame": seg_frame_col,
			"segment.end_reason": end_reason_col,
			"segment.bowl": bowl_col,
			"segment.view": view_col,
			"segment.views": views_col,
			"mask.is_blind": np.asarray(is_blind if is_blind is not None else [False] * total_frames, dtype=bool),
			"mask.is_takeover": np.asarray(
				is_takeover if is_takeover is not None else [False] * total_frames, dtype=bool
			),
			"mask.supervise_action": np.asarray(
				supervise_action if supervise_action is not None else [True] * total_frames, dtype=bool
			),
			"task": task_col,
		},
	}


def test_contract_constants_and_repo_sync():
	assert STATE_DIM == 36
	assert ACTION_DIM == 6
	assert len(SEGMENT_KINDS_ORDER) == 12
	assert BOWL_VOCAB == ("", "green", "pink")
	assert STUDENT_CAMERAS == (
		"observation.images.top_rgb_v2",
		"observation.images.top_depth_v2",
		"observation.images.wrist",
	)
	assert all("front" not in c and "top_full" not in c for c in STUDENT_CAMERAS)

	try:
		from digital_twin_armigo.agent.student_rollout import SEGMENT_KINDS
		from digital_twin_armigo.agent.twin_mujoco import INITIAL_Q as TM_INITIAL_Q
	except (ImportError, AttributeError):
		pass
	else:
		assert SEGMENT_KINDS_ORDER == tuple(SEGMENT_KINDS.keys())
		assert np.allclose(INITIAL_Q, TM_INITIAL_Q)


def test_encode_state_layout_and_no_diagnostics():
	q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)
	s0 = encode_state(
		q_t=q,
		q_prev=None,
		prev_action=None,
		gripper_cmd=0.75,
		segment_kind="scan_view",
		bowl="",
		view=2,
		views=5,
		segment_frame=150,
	)
	assert s0.shape == (36,) and s0.dtype == np.float32
	assert np.allclose(s0[0:6], q)
	assert np.allclose(s0[6:12], q)  # repeats q_0
	assert np.allclose(s0[12:18], INITIAL_Q.astype(np.float32))
	assert np.isclose(s0[18], 0.75)
	assert s0[19 + SEGMENT_KINDS_ORDER.index("scan_view")] == 1.0
	assert s0[19:31].sum() == 1.0
	assert s0[31] == 1.0 and s0[31:34].sum() == 1.0
	assert np.isclose(s0[34], 2.0 / 5.0)
	assert np.isclose(s0[35], 150.0 / 300.0)

	# Clamping of segment_frame_norm at 5.0 and view_norm when view=0 / views=0
	s_clamp = encode_state(
		q_t=q,
		q_prev=q - 0.1,
		prev_action=q + 0.1,
		gripper_cmd=0.1,
		segment_kind="carry_to_bowl",
		bowl="pink",
		view=0,
		views=0,
		segment_frame=9999,
	)
	assert s_clamp[31 + 2] == 1.0  # pink
	assert s_clamp[34] == 0.0
	assert s_clamp[35] == 5.0

	# Planner diagnostics MUST be rejected
	for diag_key in ("mode", "faces", "target_q", "via_poses", "bottom_px", "xy"):
		with pytest.raises(ValueError, match="forbidden"):
			encode_state_from_task(
				q,
				q,
				q,
				0.5,
				{"kind": "grasp", "instruction": "Grasp the brick on the table.", diag_key: 123},
			)


def test_action_normalizer_fit_transform_and_serialization(tmp_path):
	rng = np.random.default_rng(7)
	train_actions = rng.normal(loc=[0.2, -0.5, 0.1, 0.0, 0.3, 0.8], scale=[0.5, 0.4, 0.3, 0.2, 0.8, 0.1], size=(200, 6))
	train_actions[:, 3] = 0.42  # constant joint tests zero-std guard
	train_states = rng.normal(size=(200, STATE_DIM)).astype(np.float32)

	norm = ActionNormalizer.fit(train_actions, states=train_states)
	assert norm.action_std[3] == 1.0
	normed = norm.normalize_action(train_actions.astype(np.float32))
	roundtrip = norm.unnormalize_action(normed)
	assert np.allclose(roundtrip, train_actions.astype(np.float32), atol=1e-5)

	st_rt = norm.unnormalize_state(norm.normalize_state(train_states))
	assert np.allclose(st_rt, train_states, atol=1e-5)

	written = norm.save_alongside_checkpoint(tmp_path)
	loaded_json = ActionNormalizer.load(written["json"])
	assert np.allclose(loaded_json.action_mean, norm.action_mean)
	assert np.allclose(loaded_json.action_std, norm.action_std)
	assert np.allclose(loaded_json.state_mean, norm.state_mean)
	if "safetensors" in written:
		loaded_sf = ActionNormalizer.load(written["safetensors"])
		assert np.allclose(loaded_sf.action_mean, norm.action_mean)
		assert np.allclose(loaded_sf.state_std, norm.state_std)


def test_dataset_q_prev_boundary_masking_and_last_valid_fill():
	ep0 = _make_synthetic_episode(
		0,
		[
			("grasp", 0, 3, "done", {}),
			("lift", 3, 10, "done", {}),
		],
	)
	ep1 = _make_synthetic_episode(
		1,
		[
			("prepare", 0, 4, "done", {}),
		],
	)
	ds = SegmentChunkDataset.from_episodes([ep0, ep1], horizon=5, return_torch=False)
	assert len(ds) == 14

	# Episode 0, frame 0: q_{t-1} repeats q_0; frame 1 uses q_0
	s0 = ds[0]
	s1 = ds[1]
	assert np.allclose(s0["observation.state"][0:6], s0["observation.state"][6:12])
	assert np.allclose(s1["observation.state"][6:12], s0["observation.state"][0:6])

	# Episode 1, frame 0: q_{t-1} repeats ep1's q_0 (never ep0's last q!)
	s_ep1_0 = ds[10]
	assert np.allclose(s_ep1_0["observation.state"][0:6], s_ep1_0["observation.state"][6:12])
	assert not np.allclose(s_ep1_0["observation.state"][6:12], ds[9]["observation.state"][0:6])

	# Row 0 of grasp [0, 3): valid len 3 -> m=[T,T,T,F,F], padded with action[2], NEVER action[3] (lift)!
	ep0_act = ep0["frames"]["action"]
	assert s0["action_mask"].tolist() == [True, True, True, False, False]
	assert int(s0["end_step_target"]) == 3
	assert np.allclose(s0["action"][:3], ep0_act[:3])
	assert np.allclose(s0["action"][3], ep0_act[2])
	assert np.allclose(s0["action"][4], ep0_act[2])

	# Row 2 of grasp (last frame): valid len 1 -> m=[T,F,F,F,F], end_step=1
	s2 = ds[2]
	assert s2["action_mask"].tolist() == [True, False, False, False, False]
	assert int(s2["end_step_target"]) == 1
	assert np.allclose(s2["action"][1:], np.repeat(ep0_act[2:3], 4, axis=0))

	# Row 3 of lift [3, 10): cvl=7 > H=5 -> end_step = H+1 = 6
	s3 = ds[3]
	assert s3["action_mask"].tolist() == [True, True, True, True, True]
	assert int(s3["end_step_target"]) == 6

	# Row 5 of lift [3, 10): cvl=5 == H=5 -> end_step = H = 5
	s5 = ds[5]
	assert int(s5["end_step_target"]) == 5

	# Verify numpy vla_collate_fn path works cleanly with return_torch=False (including .venv without torch)
	np_batch = vla_collate_fn([s0, s2, s3])
	assert isinstance(np_batch["observation.state"], np.ndarray)
	assert np_batch["observation.state"].shape == (3, STATE_DIM)
	assert np_batch["action"].shape == (3, 5, ACTION_DIM)
	assert np_batch["action_mask"].shape == (3, 5)
	assert np_batch["end_step_target"].tolist() == [3, 1, 6]


def test_zero_frame_let_go_synthesis_and_segment_done_convention():
	# Includes mid-episode zero-frame let_go AND end-of-episode zero-frame let_go
	ep = _make_synthetic_episode(
		0,
		[
			("grasp", 0, 3, "done", {}),
			("lift", 3, 5, "brick_lost", {}),
			("let_go", 5, 5, "done", {}),
			("return_ready", 5, 7, "done", {}),
			("let_go", 7, 7, "done", {}),
		],
	)
	ds = SegmentChunkDataset.from_episodes([ep], horizon=5, return_torch=False)
	# 7 real frames + 2 synthesized zero-frame let_go samples = 9
	assert len(ds) == 9

	# Row 5 has segment.done=True under SEGMENT_DONE_CONVENTION, yet its OWN segment is return_ready (len 2):
	# end_step_target MUST be 2 (never 0!).
	s_ret = ds[5]
	assert bool(ep["frames"]["segment.done"][5]) is True
	assert s_ret["segment_kind"] == "return_ready"
	assert int(s_ret["end_step_target"]) == 2
	assert not s_ret["is_synthesized_zero_frame"]

	# Synthesized mid-episode let_go sample (index 7): anchored at min(5, 6) = 5
	s_letgo_mid = ds[7]
	assert s_letgo_mid["is_synthesized_zero_frame"] is True
	assert s_letgo_mid["segment_kind"] == "let_go"
	assert s_letgo_mid["task"] == SEGMENT_TEMPLATES["let_go"]
	assert int(s_letgo_mid["end_step_target"]) == 0
	assert s_letgo_mid["action_mask"].tolist() == [False] * 5
	assert np.allclose(s_letgo_mid["observation.state"][0:6], ep["frames"]["observation.state"][5])
	assert np.allclose(s_letgo_mid["observation.state"][6:12], ep["frames"]["observation.state"][4])
	assert s_letgo_mid["observation.state"][35] == 0.0  # segment_frame_norm == 0

	# Synthesized end-of-episode let_go sample (index 8): anchored safely at min(7, 6) = 6
	s_letgo_end = ds[8]
	assert s_letgo_end["is_synthesized_zero_frame"] is True
	assert int(s_letgo_end["end_step_target"]) == 0
	assert np.allclose(s_letgo_end["observation.state"][0:6], ep["frames"]["observation.state"][6])


def test_brick_lost_fault_and_blind_masking():
	ep = _make_synthetic_episode(
		0,
		[
			("lift", 0, 3, "brick_lost", {}),
			("scan_view", 3, 6, "fault", {"view": 1, "views": 4}),
			("prepare", 6, 9, "done", {}),
		],
		is_blind=[False, False, False, False, False, False, True, False, False],
		supervise_action=[True, True, True, True, True, True, True, False, True],
	)
	ds = SegmentChunkDataset.from_episodes([ep], horizon=5, return_torch=False)

	# brick_lost and fault: L_done MUST be masked (-100), while L_act still supervises valid actions!
	for idx in (0, 1, 2, 3, 4, 5):
		assert int(ds[idx]["end_step_target"]) == IGNORE_INDEX
		assert ds[idx]["action_mask"].any()

	# is_blind=True at idx 6: BOTH action_mask (all False) and end_step_target (-100) masked
	assert ds[6]["action_mask"].tolist() == [False] * 5
	assert int(ds[6]["end_step_target"]) == IGNORE_INDEX

	# supervise_action=False at idx 7: BOTH action_mask (all False) and end_step_target (-100) masked
	assert ds[7]["action_mask"].tolist() == [False] * 5
	assert int(ds[7]["end_step_target"]) == IGNORE_INDEX

	# Normal supervised frame at idx 8
	assert ds[8]["action_mask"].tolist() == [True, False, False, False, False]
	assert int(ds[8]["end_step_target"]) == 1


def test_takeover_boundary_cutoff_and_2x_weight():
	ep = _make_synthetic_episode(
		0,
		[("grasp", 0, 5, "done", {})],
		is_takeover=[False, False, True, True, True],
		supervise_action=[True, True, True, True, True],
	)
	ds = SegmentChunkDataset.from_episodes([ep], horizon=5, takeover_weight=2.0, return_torch=False)
	act = ep["frames"]["action"]

	# Frame 0: takeover onset is at frame 2 (offset d=2). Must cut chunk before offset 2!
	s0 = ds[0]
	assert s0["action_mask"].tolist() == [True, True, False, False, False]
	assert np.allclose(s0["action"][:2], act[:2])
	assert np.allclose(s0["action"][2:], np.repeat(act[1:2], 3, axis=0))  # last valid pre-takeover action!
	assert float(s0["sample_weight"]) == 1.0

	# Frame 1: takeover onset at d=1 -> only k=0 valid!
	s1 = ds[1]
	assert s1["action_mask"].tolist() == [True, False, False, False, False]
	assert np.allclose(s1["action"][1:], np.repeat(act[1:2], 4, axis=0))
	assert float(s1["sample_weight"]) == 1.0

	# Frame 2 (takeover active throughout 2,3,4): full remaining segment valid + 2x sample weight!
	s2 = ds[2]
	assert s2["action_mask"].tolist() == [True, True, True, False, False]
	assert float(s2["sample_weight"]) == 2.0
	assert float(ds[3]["sample_weight"]) == 2.0
	assert float(ds[4]["sample_weight"]) == 2.0

	# Check class weights computation (finite, positive, normalized)
	cw = ds.end_step_class_weights()
	assert cw.shape == (7,)
	assert np.isfinite(cw).all() and (cw > 0).all()

	# Check end_step_target_from_segment direct edge cases
	assert end_step_target_from_segment(1, 50, "done") == 1
	assert end_step_target_from_segment(50, 50, "done") == 50
	assert end_step_target_from_segment(51, 50, "done") == 51
	assert end_step_target_from_segment(0, 50, "done", is_zero_frame=True) == 0
	assert end_step_target_from_segment(10, 50, "brick_lost") == IGNORE_INDEX


def test_end_step_f1_tolerance_and_metrics():
	h = 50
	# Samples:
	# 0: true=10, pred=12 -> TP (|2| <= 2)
	# 1: true=10, pred=8  -> TP (|-2| <= 2)
	# 2: true=10, pred=13 -> FP + FN (out_of_tolerance, |3| > 2)
	# 3: true=51, pred=50 -> FP (fp_on_negative! even though |50-51|=1, 51 is negative)
	# 4: true=0,  pred=0  -> TP (let_go immediate end)
	# 5: true=5,  pred=51 -> FN (fn_missed)
	# 6: true=51, pred=51 -> TN
	# 7: true=-100, pred=1 -> ignored!
	y_true = np.array([10, 10, 10, 51, 0, 5, 51, IGNORE_INDEX])
	y_pred = np.array([12, 8, 13, 50, 0, 51, 51, 1])
	res = compute_end_step_metrics(y_pred, y_true, horizon=h, tolerance=2)
	c = res["confusion"]
	assert res["valid_count"] == 7
	assert c["tp"] == 3
	assert c["fp"] == 2  # idx 2 (out_of_tol) + idx 3 (fp_on_neg)
	assert c["fn"] == 2  # idx 2 (out_of_tol) + idx 5 (fn_missed)
	assert c["tn"] == 1
	assert c["out_of_tolerance"] == 1
	assert c["fp_on_negative"] == 1
	assert c["fn_missed"] == 1
	assert np.isclose(res["precision"], 3 / 5)
	assert np.isclose(res["recall"], 3 / 5)
	assert np.isclose(res["f1"], 0.6)
	assert np.isclose(res["end_step_mae_in_chunk"], (2 + 2 + 3 + 0) / 4)

	# Action metrics ignore masked steps and handle let_go (0 valid steps) cleanly
	true_a = np.zeros((3, 4, 6), dtype=np.float32)
	pred_a = np.zeros((3, 4, 6), dtype=np.float32)
	pred_a[0, :2, :] = 0.1  # valid
	pred_a[0, 2:, :] = 999.0  # masked! MUST be ignored
	pred_a[1, :1, :] = 0.2  # valid
	pred_a[2, :, :] = 500.0  # let_go: all masked!
	mask = np.array(
		[
			[True, True, False, False],
			[True, False, False, False],
			[False, False, False, False],
		]
	)
	kinds = ["grasp", "lift", "let_go"]
	m = compute_vla_metrics(pred_a, true_a, mask, np.array([2, 1, 0]), np.array([2, 1, 0]), kinds, horizon=4)
	assert m["valid_action_steps"] == 3
	assert np.isclose(m["action_mae_rad"], (2 * 0.1 + 1 * 0.2) / 3)
	assert np.isclose(m["per_segment_kind"]["grasp"]["mae_rad"], 0.1)
	assert np.isclose(m["per_segment_kind"]["lift"]["mae_rad"], 0.2)
	assert m["per_segment_kind"]["let_go"]["mae_rad"] is None
	assert m["per_segment_kind"]["let_go"]["valid_steps"] == 0

	# Incremental EvalAccumulator must match batch compute_vla_metrics
	acc = EvalAccumulator(horizon=4, tolerance=2)
	acc.update(pred_a[:2], true_a[:2], mask[:2], np.array([2, 1]), np.array([2, 1]), kinds[:2])
	acc.update(pred_a[2:], true_a[2:], mask[2:], np.array([0]), np.array([0]), kinds[2:])
	m2 = acc.compute()
	assert np.isclose(m2["action_mae_rad"], m["action_mae_rad"])
	assert np.isclose(m2["action_rmse_rad"], m["action_rmse_rad"])
	assert m2["end_step"]["f1"] == m["end_step"]["f1"]


def test_depth_augmentation_preserves_color_channels():
	aug = StudentImageAugmentor(
		rgb_crop_prob=0.0,
		depth_crop_prob=0.0,
		brightness=0.3,
		contrast=0.3,
		saturation=0.3,
		hue=0.1,
		rgb_noise_std=0.0,
		depth_noise_std=0.0,
		seed=123,
	)
	depth = np.linspace(0.1, 0.9, 3 * 32 * 32, dtype=np.float32).reshape(3, 32, 32)
	rgb = depth.copy()
	try:
		import torch

		inp = {
			"observation.images.top_rgb_v2": torch.from_numpy(rgb.copy()),
			"observation.images.wrist": torch.from_numpy(rgb.copy()),
			"observation.images.top_depth_v2": torch.from_numpy(depth.copy()),
		}
		out = aug(inp)
		assert torch.equal(out["observation.images.top_depth_v2"], inp["observation.images.top_depth_v2"])
		assert not torch.allclose(out["observation.images.top_rgb_v2"], inp["observation.images.top_rgb_v2"])
		assert not torch.allclose(out["observation.images.wrist"], inp["observation.images.wrist"])
	except ImportError:
		inp = {
			"observation.images.top_rgb_v2": rgb.copy(),
			"observation.images.wrist": rgb.copy(),
			"observation.images.top_depth_v2": depth.copy(),
		}
		out = aug(inp)
		assert np.array_equal(out["observation.images.top_depth_v2"], inp["observation.images.top_depth_v2"])
		assert not np.allclose(out["observation.images.top_rgb_v2"], inp["observation.images.top_rgb_v2"])


def test_act_wrapper_forward_backward_and_predict_chunk(tmp_path):
	torch = pytest.importorskip("torch")
	pytest.importorskip("lerobot")
	from digital_twin_armigo.vla.models import ACTWithEndHead, load_vla_checkpoint

	ep = _make_synthetic_episode(
		0,
		[
			("grasp", 0, 3, "done", {}),
			("lift", 3, 5, "brick_lost", {}),
			("let_go", 5, 5, "done", {}),
			("return_ready", 5, 7, "done", {}),
		],
		is_blind=[False, True, False, False, False, False, False],
		is_takeover=[False, False, False, False, False, True, True],
	)
	ds = SegmentChunkDataset.from_episodes([ep], horizon=6, takeover_weight=2.0, return_torch=True)
	norm = ds.fit_normalizer()
	cw = ds.end_step_class_weights()
	shapes = {c: (3, 64, 64) for c in STUDENT_CAMERAS}

	model = ACTWithEndHead(
		horizon=6,
		normalizer=norm,
		class_weights=cw,
		camera_shapes=shapes,
		tiny_for_test=True,
	)
	model.train()
	batch = vla_collate_fn([ds[i] for i in range(len(ds))])
	loss, loss_dict = model(batch)
	assert torch.isfinite(loss)
	assert set(loss_dict) == {"loss", "l_act", "l1_loss", "kld_loss", "l_done"}
	loss.backward()

	# Verify gradients reach ACT encoder, decoder, action_head, and EndStepHead
	assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.end_head.parameters())
	assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.policy.model.action_head.parameters())
	assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.policy.model.encoder.parameters())

	# Batch with ONLY zero-frame let_go (action_mask all False) -> L_act == 0, L_done trains end_head
	model.zero_grad(set_to_none=True)
	letgo_batch = vla_collate_fn([ds[7], ds[7]])
	loss_lg, ld_lg = model(letgo_batch)
	assert ld_lg["l_act"] == 0.0 and ld_lg["l1_loss"] == 0.0 and ld_lg["kld_loss"] == 0.0
	assert ld_lg["l_done"] > 0.0
	loss_lg.backward()
	assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.end_head.parameters())

	# Batch with ONLY masked end_step_target (-100) -> L_done == 0
	model.zero_grad(set_to_none=True)
	lost_batch = vla_collate_fn([ds[3], ds[4]])
	loss_lost, ld_lost = model(lost_batch)
	assert ld_lost["l_done"] == 0.0 and ld_lost["l_act"] > 0.0
	loss_lost.backward()

	# Predict chunk + save/load checkpoint round-trip
	pred_rad, end_probs = model.predict_chunk(batch)
	assert pred_rad.shape == (len(ds), 6, ACTION_DIM)
	assert end_probs.shape == (len(ds), 8)
	assert torch.allclose(end_probs.sum(dim=-1), torch.ones(len(ds)), atol=1e-5)

	ckpt = model.save_checkpoint(tmp_path / "act_ckpt")
	reloaded = load_vla_checkpoint(ckpt, device="cpu")
	pred_rad_2, end_probs_2 = reloaded.predict_chunk(batch)
	assert torch.allclose(pred_rad, pred_rad_2, atol=1e-5)
	assert torch.allclose(end_probs, end_probs_2, atol=1e-5)


def test_smolvla_wrapper_forward_backward_and_predict_chunk(tmp_path):
	torch = pytest.importorskip("torch")
	pytest.importorskip("lerobot")
	pytest.importorskip("transformers")
	from digital_twin_armigo.vla.models import SmolVLAWithEndHead, load_vla_checkpoint

	ep = _make_synthetic_episode(
		0,
		[
			("scan_view", 0, 3, "done", {"view": 1, "views": 4}),
			("let_go", 3, 3, "done", {}),
			("carry_to_bowl", 3, 6, "done", {"bowl": "pink"}),
		],
		is_takeover=[False, False, False, True, True, True],
	)
	ds = SegmentChunkDataset.from_episodes([ep], horizon=6, takeover_weight=2.0, return_torch=True)
	norm = ds.fit_normalizer()
	cw = ds.end_step_class_weights()
	shapes = {c: (3, 64, 64) for c in STUDENT_CAMERAS}

	model = SmolVLAWithEndHead(
		horizon=6,
		init_from=None,
		normalizer=norm,
		class_weights=cw,
		camera_shapes=shapes,
		tiny_for_test=True,
	)
	model.train()

	# Frozen VLM vs trainable expert/state_proj/end_head
	vlm_params = list(model.policy.model.vlm_with_expert.vlm.parameters())
	assert vlm_params and all(not p.requires_grad for p in vlm_params)
	assert all(p.requires_grad for p in model.policy.model.state_proj.parameters())
	assert all(p.requires_grad for p in model.end_head.parameters())

	batch = vla_collate_fn([ds[i] for i in range(len(ds))])
	loss, loss_dict = model(batch)
	assert torch.isfinite(loss)
	assert set(loss_dict) == {"loss", "l_act", "l_done"}
	loss.backward()

	assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.end_head.parameters())
	assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.policy.model.state_proj.parameters())
	assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.policy.model.action_out_proj.parameters())
	assert any(
		p.grad is not None and p.grad.abs().sum() > 0 for p in model.policy.model.vlm_with_expert.lm_expert.parameters()
	)
	assert all(p.grad is None for p in vlm_params)

	# Zero-frame let_go batch -> l_act == 0.0
	model.zero_grad(set_to_none=True)
	lg_batch = vla_collate_fn([ds[6], ds[6]])
	loss_lg, ld_lg = model(lg_batch)
	assert ld_lg["l_act"] == 0.0 and ld_lg["l_done"] > 0.0
	loss_lg.backward()

	# Predict chunk via 10 Euler steps + save/load checkpoint
	torch.manual_seed(0)
	noise = torch.randn(len(ds), 6, 6)
	pred_rad, end_probs = model.predict_chunk(batch, noise=noise)
	assert pred_rad.shape == (len(ds), 6, ACTION_DIM)
	assert end_probs.shape == (len(ds), 8)
	assert torch.allclose(end_probs.sum(dim=-1), torch.ones(len(ds)), atol=1e-5)

	ckpt = model.save_checkpoint(tmp_path / "smolvla_ckpt")
	reloaded = load_vla_checkpoint(ckpt, device="cpu")
	pred_rad_2, end_probs_2 = reloaded.predict_chunk(batch, noise=noise)
	assert torch.allclose(pred_rad, pred_rad_2, atol=1e-5)
	assert torch.allclose(end_probs, end_probs_2, atol=1e-5)


def test_lerobot_v3_roundtrip_and_train_cli_smoke(tmp_path):
	torch = pytest.importorskip("torch")
	lerobot_ds_mod = pytest.importorskip("lerobot.datasets.lerobot_dataset")
	pytest.importorskip("transformers")
	from lerobot.datasets.lerobot_dataset import LeRobotDataset

	from digital_twin_armigo.constants import FPS, STATE_NAMES

	if str(ROOT) not in sys.path:
		sys.path.insert(0, str(ROOT))
	from scripts.vla.train import main as train_main

	# Build a genuine LeRobot v3 dataset on disk with video cameras (including forbidden front/top_full) + meta/segments.json
	ds_dir = tmp_path / "lerobot_v3"
	features = {
		"observation.state": {"dtype": "float32", "shape": (6,), "names": STATE_NAMES},
		"observation.prev_action": {"dtype": "float32", "shape": (6,), "names": STATE_NAMES},
		"observation.gripper_cmd": {"dtype": "float32", "shape": (1,), "names": ["gripper_cmd"]},
		"action": {"dtype": "float32", "shape": (6,), "names": STATE_NAMES},
		"observation.images.top_rgb_v2": {"dtype": "video", "shape": (64, 64, 3), "names": ["h", "w", "c"]},
		"observation.images.top_depth_v2": {"dtype": "video", "shape": (64, 64, 3), "names": ["h", "w", "c"]},
		"observation.images.wrist": {"dtype": "video", "shape": (64, 64, 3), "names": ["h", "w", "c"]},
		"observation.images.top_full": {"dtype": "video", "shape": (64, 64, 3), "names": ["h", "w", "c"]},
		"observation.images.front": {"dtype": "video", "shape": (64, 64, 3), "names": ["h", "w", "c"]},
		"segment.index": {"dtype": "int64", "shape": (1,), "names": None},
		"segment.kind": {"dtype": "string", "shape": (1,), "names": None},
		"segment.done": {"dtype": "bool", "shape": (1,), "names": None},
		"segment.chunk_valid_len": {"dtype": "int64", "shape": (1,), "names": None},
		"segment.frame": {"dtype": "int64", "shape": (1,), "names": None},
		"segment.end_reason": {"dtype": "string", "shape": (1,), "names": None},
		"segment.bowl": {"dtype": "string", "shape": (1,), "names": None},
		"segment.view": {"dtype": "int64", "shape": (1,), "names": None},
		"segment.views": {"dtype": "int64", "shape": (1,), "names": None},
		"mask.is_blind": {"dtype": "bool", "shape": (1,), "names": None},
		"mask.is_takeover": {"dtype": "bool", "shape": (1,), "names": None},
		"mask.supervise_action": {"dtype": "bool", "shape": (1,), "names": None},
	}

	ep0 = _make_synthetic_episode(
		0,
		[
			("prepare", 0, 4, "done", {}),
			("let_go", 4, 4, "done", {}),
			("return_ready", 4, 7, "done", {}),
		],
	)
	ep1 = _make_synthetic_episode(
		1,
		[
			("carry_to_bowl", 0, 5, "done", {"bowl": "green"}),
		],
	)
	dataset = LeRobotDataset.create(
		repo_id="local/armigo-test",
		fps=FPS,
		root=ds_dir,
		robot_type="SO101_Follower",
		use_videos=True,
		features=features,
	)
	seg_episodes_meta = []
	for ep in (ep0, ep1):
		f = ep["frames"]
		n = len(f["action"])
		for r in range(n):
			frame = {
				"observation.state": f["observation.state"][r],
				"observation.prev_action": f["observation.prev_action"][r],
				"observation.gripper_cmd": f["observation.gripper_cmd"][r],
				"action": f["action"][r],
				"segment.index": np.array([f["segment.index"][r]], dtype=np.int64),
				"segment.kind": f["segment.kind"][r],
				"segment.done": np.array([f["segment.done"][r]], dtype=bool),
				"segment.chunk_valid_len": np.array([f["segment.chunk_valid_len"][r]], dtype=np.int64),
				"segment.frame": np.array([f["segment.frame"][r]], dtype=np.int64),
				"segment.end_reason": f["segment.end_reason"][r],
				"segment.bowl": f["segment.bowl"][r],
				"segment.view": np.array([f["segment.view"][r]], dtype=np.int64),
				"segment.views": np.array([f["segment.views"][r]], dtype=np.int64),
				"mask.is_blind": np.array([f["mask.is_blind"][r]], dtype=bool),
				"mask.is_takeover": np.array([f["mask.is_takeover"][r]], dtype=bool),
				"mask.supervise_action": np.array([f["mask.supervise_action"][r]], dtype=bool),
				"task": f["task"][r],
			}
			for cam in STUDENT_CAMERAS:
				frame[cam] = ep["images"][cam][r]
			frame["observation.images.top_full"] = ep["images"]["observation.images.top_rgb_v2"][r]
			frame["observation.images.front"] = ep["images"]["observation.images.wrist"][r]
			dataset.add_frame(frame)
		dataset.save_episode()
		seg_episodes_meta.append({"episode_index": ep["episode_index"], "source": "synth", "segments": ep["segments"]})
	dataset.finalize()
	(ds_dir / "meta" / "segments.json").write_text(
		json.dumps({"schema": "armigo-student-segments/v1", "episodes": seg_episodes_meta}, indent=2),
		encoding="utf-8",
	)

	# Verify SegmentChunkDataset reads LeRobot v3 + synthesizes let_go + exposes ONLY student cameras
	loaded_ds = SegmentChunkDataset(ds_dir, repo_id="local/armigo-test", horizon=6, video_backend="pyav")
	assert len(loaded_ds) == 7 + 5 + 1  # 12 real frames + 1 synthesized let_go
	sample0 = loaded_ds[0]
	assert "observation.images.front" not in sample0 and "observation.images.top_full" not in sample0
	assert all(cam in sample0 for cam in STUDENT_CAMERAS)
	assert any(loaded_ds[i]["is_synthesized_zero_frame"] for i in range(len(loaded_ds)))

	# Smoke-run train CLI for both ACT and SmolVLA
	for family in ("act", "smolvla"):
		out_dir = tmp_path / f"run_{family}"
		rc = train_main(
			[
				"--family",
				family,
				"--dataset",
				str(ds_dir),
				"--val-dataset",
				str(ds_dir),
				"--repo-id",
				"local/armigo-test",
				"--out",
				str(out_dir),
				"--horizon",
				"6",
				"--smoke-steps",
				"2",
				"--batch-size",
				"4",
				"--eval-every",
				"2",
				"--num-workers",
				"0",
				"--device",
				"cuda" if torch.cuda.is_available() else "cpu",
				"--tiny-for-test",
			]
		)
		assert rc == 0
		assert (out_dir / "last" / "model.safetensors").exists()
		assert (out_dir / "last" / "normalizer.json").exists()
		assert (out_dir / "last" / "normalizer.safetensors").exists()
		assert (out_dir / "last" / "vla_config.json").exists()
		assert (out_dir / "best" / "model.safetensors").exists()
		summary = json.loads((out_dir / "final_summary.json").read_text(encoding="utf-8"))
		assert summary["family"] == family
		assert summary["final_val_metrics"] is not None
	del lerobot_ds_mod


def _run_all_as_script() -> int:
	import inspect
	import time

	tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_") and callable(fn)]
	passed = 0
	for name, fn in tests:
		t0 = time.perf_counter()
		sig = inspect.signature(fn)
		if "tmp_path" in sig.parameters:
			with tempfile.TemporaryDirectory() as td:
				fn(Path(td))
		else:
			fn()
		dt = time.perf_counter() - t0
		print(f"PASS {name} ({dt:.2f}s)")
		passed += 1
	print(f"\n{passed}/{len(tests)} tests PASSED")
	return 0


if __name__ == "__main__":
	raise SystemExit(_run_all_as_script())
