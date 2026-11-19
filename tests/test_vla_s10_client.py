"""Unit and MuJoCo smoke tests for S10 policy server, ChunkClientPolicy, and eval_closed_loop."""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
	sys.path.insert(0, str(ROOT / "src"))

from digital_twin_armigo.agent.student_rollout import (
	POLICY_FIELDS,
	PolicyStep,
	ReplayPolicy,
	Segment,
	StudentObservation,
	policy_inputs,
	run_episode,
)
from digital_twin_armigo.constants import TOP_CROP
from digital_twin_armigo.vla.client_policy import (
	ChunkClientPolicy,
	RemoteChunkPredictor,
	extract_student_images,
)
from digital_twin_armigo.vla.contract import (
	ACTION_DIM,
	INITIAL_Q,
	STATE_DIM,
	STUDENT_CAMERAS,
	encode_state_from_task,
)
from digital_twin_armigo.vla.server import (
	LocalChunkPredictor,
	VLAPolicyServer,
	decode_image,
	encode_image,
)

_EVAL_SPEC = importlib.util.spec_from_file_location("eval_closed_loop", ROOT / "scripts/vla/eval_closed_loop.py")
assert _EVAL_SPEC is not None and _EVAL_SPEC.loader is not None
eval_closed_loop = importlib.util.module_from_spec(_EVAL_SPEC)
_EVAL_SPEC.loader.exec_module(eval_closed_loop)


class _TrapImageDict(Mapping):
	"""Mapping raising AssertionError if any forbidden camera or key enumeration is touched."""

	def __init__(self, data: dict[str, np.ndarray]) -> None:
		self._data = data

	def __getitem__(self, key: str) -> np.ndarray:
		if "front" in key or "top_full" in key:
			raise AssertionError(f"forbidden camera key {key!r} accessed!")
		return self._data[key]

	def __iter__(self):
		raise AssertionError("forbidden full-dict iteration over obs.images()")

	def __len__(self) -> int:
		return len(self._data)

	def __contains__(self, key: object) -> bool:
		if isinstance(key, str) and ("front" in key or "top_full" in key):
			raise AssertionError(f"forbidden camera key {key!r} checked!")
		return key in self._data


class _DummyChunkModel:
	"""Pure-numpy dummy model compatible with ``LocalChunkPredictor``."""

	def __init__(self, horizon: int = 8, family: str = "smolvla") -> None:
		self.horizon = horizon
		self.family = family
		self.calls: list[dict[str, Any]] = []
		self.next_k_hat: int | None = None
		self.next_probs: np.ndarray | None = None
		self.next_action_value: float = 0.25
		self.block_event: threading.Event | None = None
		self.started_event: threading.Event | None = None

	def predict_chunk(self, batch: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
		if self.started_event is not None:
			self.started_event.set()
		if self.block_event is not None:
			self.block_event.wait(timeout=5.0)
		self.calls.append(
			{
				"state": np.asarray(batch["observation.state"]).copy(),
				"task": list(batch["task"]),
				"segment_frame": batch.get("segment_frame"),
				"cam_shapes": {k: np.asarray(batch[k]).shape for k in STUDENT_CAMERAS},
			}
		)
		val = self.next_action_value
		actions = np.full((1, self.horizon, ACTION_DIM), val, dtype=np.float32)
		# Give distinct per-step offset so tests can identify chunk step index.
		actions[0, :, 0] += np.arange(self.horizon, dtype=np.float32) * 0.01
		if self.next_probs is not None:
			probs = self.next_probs.reshape(1, self.horizon + 2).astype(np.float32)
		else:
			probs = np.zeros((1, self.horizon + 2), dtype=np.float32)
			k = self.horizon + 1 if self.next_k_hat is None else int(self.next_k_hat)
			probs[0, k] = 1.0
		return actions, probs


def _make_raw_images(seed: int = 0) -> dict[str, np.ndarray]:
	rng = np.random.default_rng(seed)
	return {
		"front": rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8),
		"top_rgb": rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8),
		"top_depth": rng.integers(0, 255, size=(416, 512, 3), dtype=np.uint8),
		"wrist": rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8),
	}


def _make_obs(
	task: dict[str, Any],
	*,
	segment_id: int,
	segment_frame: int,
	raw_images: Mapping[str, np.ndarray] | None = None,
	frame: int | None = None,
) -> StudentObservation:
	imgs = _make_raw_images() if raw_images is None else raw_images
	q = np.array([0.1, -0.2, 0.3, -0.1, 0.05, 0.7], dtype=np.float64)
	return StudentObservation(
		frame=segment_frame if frame is None else frame,
		task=dict(task),
		segment_id=segment_id,
		segment_frame=segment_frame,
		q=q,
		_images=lambda: imgs,
		q_prev=q - 0.01,
		prev_action=INITIAL_Q.copy(),
		gripper_cmd=0.85,
	)


def test_server_local_and_http_roundtrip():
	imgs = _make_raw_images(11)
	x, y, w, h = TOP_CROP
	crop = imgs["top_rgb"][y : y + h, x : x + w]
	for enc in ("npy", "png", "raw"):
		encoded = encode_image(crop, encoding=enc)
		decoded = decode_image(encoded)
		assert decoded.dtype == np.uint8 and decoded.shape == (416, 512, 3)
		assert np.array_equal(decoded, crop)
		# Raw base64 string auto-detection for npy / png
		if enc in ("npy", "png"):
			assert np.array_equal(decode_image(encoded["data"]), crop)

	model = _DummyChunkModel(horizon=10, family="smolvla")
	model.next_k_hat = 6
	local = LocalChunkPredictor(model)
	task = policy_inputs(Segment("carry_to_bowl", {"bowl": "pink"}))
	payload = {
		"top_rgb_v2": crop,
		"top_depth_v2": imgs["top_depth"],
		"wrist": imgs["wrist"],
		"q_t": np.ones(6, dtype=np.float32) * 0.2,
		"q_prev": np.ones(6, dtype=np.float32) * 0.1,
		"prev_action": INITIAL_Q.astype(np.float32),
		"gripper_cmd": 0.42,
		"task": task,
		"segment_frame": 12,
	}
	res_local = local.predict(payload)
	assert np.asarray(res_local["actions"]).shape == (10, 6)
	assert len(res_local["end_step_probs"]) == 12
	assert res_local["predicted_end_step"] == 6
	assert res_local["latency_ms"] >= 0.0

	expected_state = encode_state_from_task(
		q_t=payload["q_t"],
		q_prev=payload["q_prev"],
		prev_action=payload["prev_action"],
		gripper_cmd=payload["gripper_cmd"],
		task=task,
		segment_frame=12,
	)
	assert np.allclose(model.calls[-1]["state"][0], expected_state)
	assert model.calls[-1]["cam_shapes"] == {
		"observation.images.top_rgb_v2": (1, 3, 416, 512),
		"observation.images.top_depth_v2": (1, 3, 416, 512),
		"observation.images.wrist": (1, 3, 480, 640),
	}

	with VLAPolicyServer(local, host="127.0.0.1", port=0) as srv:
		for enc in ("npy", "png"):
			with RemoteChunkPredictor(srv.url, image_encoding=enc) as remote:
				health = remote.health()
				assert health["status"] == "ok" and health["family"] == "smolvla" and health["horizon"] == 10
				res_http = remote.predict(payload)
				assert np.allclose(res_http["actions"], res_local["actions"])
				assert np.allclose(res_http["end_step_probs"], res_local["end_step_probs"])
				assert res_http["predicted_end_step"] == 6


def test_forbidden_cameras_and_diagnostics_rejected():
	raw = _make_raw_images(22)
	trap = _TrapImageDict({**raw, "top_full": raw["top_rgb"]})
	extracted = extract_student_images(trap)
	x, y, w, h = TOP_CROP
	assert np.array_equal(extracted["top_rgb_v2"], raw["top_rgb"][y : y + h, x : x + w])
	assert np.array_equal(extracted["top_depth_v2"], raw["top_depth"])
	assert np.array_equal(extracted["wrist"], raw["wrist"])

	model = _DummyChunkModel(horizon=8, family="act")
	local = LocalChunkPredictor(model)
	policy = ChunkClientPolicy(local, mode="sync")
	clean_task = policy_inputs(Segment("scan_view", {"mode": "orbit", "view": 1, "views": 3, "target_q": [0.0] * 6}))
	assert set(clean_task) == set(POLICY_FIELDS)

	for diag_key in ("mode", "faces", "target_q", "via_poses", "bottom_px", "xy"):
		bad_task = {**clean_task, diag_key: 123}
		with pytest.raises(ValueError, match="forbidden"):
			policy.start(bad_task, 0)
		with pytest.raises(ValueError, match="forbidden"):
			local.predict(
				{
					"top_rgb_v2": extracted["top_rgb_v2"],
					"top_depth_v2": extracted["top_depth_v2"],
					"wrist": extracted["wrist"],
					"q_t": np.zeros(6),
					"q_prev": np.zeros(6),
					"prev_action": np.zeros(6),
					"gripper_cmd": 0.5,
					"task": bad_task,
					"segment_frame": 0,
				}
			)

	with VLAPolicyServer(local, host="127.0.0.1", port=0) as srv, RemoteChunkPredictor(srv.url) as remote:
		for bad_cam in ("front", "top_full", "observation.images.front", "observation.images.top_full"):
			with pytest.raises(ValueError, match="forbidden"):
				remote.predict(
					{
						"top_rgb_v2": extracted["top_rgb_v2"],
						"top_depth_v2": extracted["top_depth_v2"],
						"wrist": extracted["wrist"],
						bad_cam: raw["front"],
						"q_t": np.zeros(6),
						"q_prev": np.zeros(6),
						"prev_action": np.zeros(6),
						"gripper_cmd": 0.5,
						"task": clean_task,
						"segment_frame": 0,
					}
				)
	policy.close()


def test_queue_reset_on_start_and_k_hat_zero_immediate_stop():
	model = _DummyChunkModel(horizon=10, family="act")
	model.next_k_hat = 11  # H + 1
	local = LocalChunkPredictor(model)
	policy = ChunkClientPolicy(local, mode="sync")

	seg0 = policy_inputs(Segment("prepare"))
	policy.start(seg0, 0)
	step0 = policy.act(_make_obs(seg0, segment_id=0, segment_frame=0))
	assert not step0.done and step0.target is not None
	assert policy.queue_remaining == 9
	assert len(model.calls) == 1

	# Start new segment before queue drains -> MUST clear queue immediately!
	seg1 = policy_inputs(Segment("let_go"))
	model.next_k_hat = 0  # zero-frame segment immediate stop!
	policy.start(seg1, 1)
	assert policy.queue_remaining == 0
	assert policy.planned_stop_step is None

	step_done = policy.act(_make_obs(seg1, segment_id=1, segment_frame=0))
	assert step_done == PolicyStep(done=True)
	assert len(model.calls) == 2
	policy.close()


def test_k_hat_stop_at_base_frame_plus_k_without_extra_call_and_threshold():
	model = _DummyChunkModel(horizon=8, family="act")
	local = LocalChunkPredictor(model)
	policy = ChunkClientPolicy(local, mode="sync")

	seg = policy_inputs(Segment("grasp"))
	policy.start(seg, 0)
	# Chunk 0 (base_frame=0): k_hat = 9 (H+1) -> planned_stop_step = None
	model.next_k_hat = 9
	for f in range(8):
		step = policy.act(_make_obs(seg, segment_id=0, segment_frame=f))
		assert not step.done
	assert len(model.calls) == 1
	assert policy.planned_stop_step is None

	# Chunk 1 (base_frame=8): k_hat = 3 -> planned_stop_step = 8 + 3 = 11
	model.next_k_hat = 3
	for f in (8, 9, 10):
		step = policy.act(_make_obs(seg, segment_id=0, segment_frame=f))
		assert not step.done
	assert len(model.calls) == 2
	assert policy.planned_stop_step == 11

	# Frame 11: MUST stop immediately WITHOUT calling predictor again!
	calls_before = len(model.calls)
	end_step = policy.act(_make_obs(seg, segment_id=0, segment_frame=11))
	assert end_step.done is True
	assert len(model.calls) == calls_before
	policy.close()

	# Verify end_threshold rejects low-confidence end predictions and accepts high-confidence ones.
	model2 = _DummyChunkModel(horizon=6, family="act")
	low_probs = np.array([0.05, 0.05, 0.40, 0.05, 0.05, 0.05, 0.05, 0.30], dtype=np.float32)  # argmax=2, prob=0.40
	model2.next_probs = low_probs
	pol2 = ChunkClientPolicy(LocalChunkPredictor(model2), mode="sync", end_threshold=0.60)
	pol2.start(seg, 0)
	s0 = pol2.act(_make_obs(seg, segment_id=0, segment_frame=0))
	assert not s0.done and pol2.planned_stop_step is None
	s2 = pol2.act(_make_obs(seg, segment_id=0, segment_frame=2))
	assert not s2.done  # ignored low-confidence k_hat=2!
	pol2.close()


def test_async_prefetch_and_stale_future_discard_on_segment_switch():
	model = _DummyChunkModel(horizon=8, family="smolvla")
	local = LocalChunkPredictor(model)

	with VLAPolicyServer(local, host="127.0.0.1", port=0) as srv, RemoteChunkPredictor(srv.url) as remote:
		policy = ChunkClientPolicy(remote, mode="async", replan_interval=4)
		try:
			# Part A: Async prefetch completes within segment 0 and stops cleanly at base_frame + k_hat
			seg0 = policy_inputs(Segment("lift"))
			policy.start(seg0, 0)
			model.next_k_hat = 9  # H+1
			model.next_action_value = 0.1
			for f in range(4):
				assert not policy.act(_make_obs(seg0, segment_id=0, segment_frame=f)).done
			assert len(model.calls) == 1

			# Frame 4 triggers async prefetch anchored at base_frame=4 while returning frame 4 action immediately.
			started = threading.Event()
			release = threading.Event()
			model.started_event = started
			model.block_event = release
			model.next_action_value = 0.5
			model.next_k_hat = 5  # planned_stop_step = 4 + 5 = 9

			step4 = policy.act(_make_obs(seg0, segment_id=0, segment_frame=4))
			assert not step4.done and np.isclose(step4.target[0], 0.1 + 0.04, atol=1e-5)
			assert started.wait(timeout=2.0)
			release.set()

			# Wait briefly for background prefetch to finish, then step 5..8 should use chunk 1!
			while policy.stats.calls < 2:
				threading.Event().wait(0.01)
			model.started_event = None
			model.block_event = None

			step6 = policy.act(_make_obs(seg0, segment_id=0, segment_frame=6))
			assert not step6.done
			# Offset in chunk 1 anchored at base_frame=4 is 6 - 4 = 2 -> value 0.5 + 0.02
			assert np.isclose(step6.target[0], 0.5 + 0.02, atol=1e-5)
			assert policy.planned_stop_step == 9
			assert not policy.act(_make_obs(seg0, segment_id=0, segment_frame=7)).done
			assert not policy.act(_make_obs(seg0, segment_id=0, segment_frame=8)).done
			calls_before_stop = policy.stats.calls
			assert policy.act(_make_obs(seg0, segment_id=0, segment_frame=9)).done is True
			assert policy.stats.calls == calls_before_stop

			# Part B: Segment switch while an async prefetch is STILL in-flight -> stale result MUST be discarded!
			seg1 = policy_inputs(Segment("grasp"))
			policy.start(seg1, 1)
			model.next_action_value = 0.2
			model.next_k_hat = 9
			for f in range(4):
				assert not policy.act(_make_obs(seg1, segment_id=1, segment_frame=f)).done

			stale_started = threading.Event()
			stale_release = threading.Event()
			model.started_event = stale_started
			model.block_event = stale_release
			model.next_action_value = 99.0  # Poison value from stale segment 1 future!
			model.next_k_hat = 1  # Poison stop step!
			assert not policy.act(_make_obs(seg1, segment_id=1, segment_frame=4)).done
			assert stale_started.wait(timeout=2.0)

			# Switch to segment 2 BEFORE stale future finishes!
			model.started_event = None
			model.block_event = None
			model.next_action_value = -0.3
			model.next_k_hat = 9
			seg2 = policy_inputs(Segment("return_ready"))
			policy.start(seg2, 2)
			assert policy.stats.cancelled_prefetches >= 1

			# Segment 2 frame 0 runs cleanly without waiting for stale segment 1 HTTP call
			s2_0 = policy.act(_make_obs(seg2, segment_id=2, segment_frame=0))
			assert not s2_0.done and np.isclose(s2_0.target[0], -0.3, atol=1e-5)

			# Now release the stale future from segment 1 and wait for its worker to finish
			stale_release.set()
			deadline = threading.Event()
			for _ in range(100):
				if policy.stats.stale_discarded >= 1:
					break
				deadline.wait(0.01)
			assert policy.stats.stale_discarded >= 1
			assert policy.planned_stop_step is None

			# Segment 2 continues using its own queue, unaffected by the poison values!
			s2_1 = policy.act(_make_obs(seg2, segment_id=2, segment_frame=1))
			assert not s2_1.done and np.isclose(s2_1.target[0], -0.3 + 0.01, atol=1e-5)
		finally:
			policy.close()

	# Also verify prefetch_ahead triggers async prefetch when remaining steps <= prefetch_ahead
	model_pa = _DummyChunkModel(horizon=8, family="smolvla")
	model_pa.next_k_hat = 9
	with ChunkClientPolicy(LocalChunkPredictor(model_pa), mode="async", prefetch_ahead=2) as pol_pa:
		seg = policy_inputs(Segment("prepare"))
		pol_pa.start(seg, 0)
		for f in range(6):  # frame 6 has remaining = 8 - 6 = 2 <= 2
			assert not pol_pa.act(_make_obs(seg, segment_id=0, segment_frame=f)).done
		assert pol_pa.stats.async_prefetches == 0
		assert not pol_pa.act(_make_obs(seg, segment_id=0, segment_frame=6)).done
		assert pol_pa.stats.async_prefetches == 1


def test_eval_sealed_test_guard_and_mode_resolution(tmp_path):
	sealed_suite = ROOT / "config/vla/test.json"
	with pytest.raises(eval_closed_loop.SealedSuiteError):
		eval_closed_loop.assert_suite_allowed(sealed_suite, allow_sealed=False)
	# Also catches ValueError and CLI invocation
	with pytest.raises(ValueError):
		eval_closed_loop.assert_suite_allowed(sealed_suite, allow_sealed=False)
	with pytest.raises(eval_closed_loop.SealedSuiteError):
		eval_closed_loop.main(
			["--server-url", "http://127.0.0.1:9999", "--suite", str(sealed_suite), "--out", str(tmp_path / "out")]
		)

	custom_sealed = tmp_path / "custom_sealed.json"
	custom_sealed.write_text(json.dumps({"sealed": True, "cases": [[1100001, False, 5]]}), encoding="utf-8")
	with pytest.raises(eval_closed_loop.SealedSuiteError):
		eval_closed_loop.assert_suite_allowed(custom_sealed, allow_sealed=False)

	# Explicitly allowing sealed passes; val.json passes by default
	assert len(eval_closed_loop.assert_suite_allowed(sealed_suite, allow_sealed=True)) == 100
	assert len(eval_closed_loop.assert_suite_allowed(ROOT / "config/vla/val.json", allow_sealed=False)) == 60

	assert eval_closed_loop.resolve_eval_mode("auto", {"family": "act"}) == "sync"
	assert eval_closed_loop.resolve_eval_mode("auto", {"family": "smolvla"}) == "async"
	assert eval_closed_loop.resolve_eval_mode("sync", {"family": "smolvla"}) == "sync"
	assert eval_closed_loop.resolve_eval_mode("async", {"family": "act"}) == "async"


def _write_synthetic_eval_episode(
	ep_dir: Path,
	*,
	case: tuple[int, bool, int],
	verdict: str,
	label: str,
	robot_contact: np.ndarray,
	jaw_env_contact: np.ndarray,
) -> dict[str, Any]:
	ep_dir.mkdir(parents=True, exist_ok=True)
	n = len(robot_contact)
	b0 = ep_dir / "branch-0"
	b0.mkdir(parents=True, exist_ok=True)
	np.savez(
		b0 / "actions.npz",
		action=np.zeros((n, 6), dtype=np.float32),
		state=np.zeros((n, 6), dtype=np.float32),
		phase=np.array(["approach"] * n),
		contact_robot_n=robot_contact.astype(np.float32),
		contact_jaw_env_n=jaw_env_contact.astype(np.float32),
	)
	expected = "defective" if case[1] else "normal"
	manifest = {
		"provenance": {"sensing": "realistic", "case": list(case)},
		"branches": [{"id": 0, "actions": n, "start_frame": 0}],
		"result": {
			"verdict": verdict,
			"label": label,
			"label_correct": label == expected,
			"student_eligible": verdict == "SUCCESS",
			"rewinds": 0,
			"retries": 0,
			"evidence": ["v0"],
			"placement_evidence": "v0" if label == "defective" else None,
		},
	}
	(ep_dir / "episode.json").write_text(json.dumps(manifest), encoding="utf-8")
	(ep_dir / "task.json").write_text(json.dumps({"case": list(case)}), encoding="utf-8")
	(ep_dir / "audit.private.json").write_text(
		json.dumps({"true_label": expected, "true_face": case[2] if case[1] else None}),
		encoding="utf-8",
	)
	events = [
		{
			"seq": 0,
			"kind": "view",
			"observation": {
				"id": "v0",
				"camera": "wrist_cam",
				"score": 1.0 if label == "defective" else -1.0,
				"threshold": 0.0,
				"faces": [0, 1, 2, 3, 4, 5],
				"face_pixels": [1000] * 6,
				"crop_bbox": [0, 0, 256, 256],
			},
		},
		{
			"seq": 1,
			"kind": "view_truth",
			"id": "v0",
			"belief_error_mm": 2.0,
			"true_brick_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
			"belief_brick_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
			"true_face_pixels": [1000] * 6,
			"true_face_pixels_crop_native": [1000] * 6,
			"true_cross_px_crop_native": 50 if case[1] else 0,
			"true_cross_px_crop": 50 if case[1] else 0,
			"truth_crop": {"bbox": [0, 0, 256, 256], "size": 256},
		},
	]
	(ep_dir / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")
	return eval_closed_loop._episode_record_from_dir(ep_dir, case)


def test_paired_bootstrap_ci_and_collisions_over_all_attempts(tmp_path):
	from digital_twin_armigo.agent.vla_grader import DEFAULT_THRESHOLDS, grade

	ci = eval_closed_loop.bootstrap_paired_ci([1, 1, 0, 1], [1, 1, 1, 0], n_boot=2000, seed=42)
	assert ci["n"] == 4
	assert np.isclose(ci["mean_diff"], 0.0)
	assert ci["ci_95"][0] <= 0.0 <= ci["ci_95"][1]
	assert ci["student_wins"] == 1 and ci["teacher_wins"] == 1

	cases = [
		(101, False, 5),
		(102, True, 0),  # defective -> student predicts "unknown" WITH jaw-env collision!
		(103, True, 1),  # defective -> student predicts "normal" (missed defect) + robot collision in FAILED!
		(104, False, 5),
	]
	ref = {"thresholds": DEFAULT_THRESHOLDS}
	records = [
		_write_synthetic_eval_episode(
			tmp_path / "s0",
			case=cases[0],
			verdict="SUCCESS",
			label="normal",
			robot_contact=np.zeros(10),
			jaw_env_contact=np.zeros(10),
		),
		_write_synthetic_eval_episode(
			tmp_path / "s1",
			case=cases[1],
			verdict="UNKNOWN",
			label="unknown",
			robot_contact=np.zeros(10),
			jaw_env_contact=np.array([0.0, 2.5, 3.1] + [0.0] * 7),  # 2 jaw-env collision frames (> 2.0 N)
		),
		_write_synthetic_eval_episode(
			tmp_path / "s2",
			case=cases[2],
			verdict="FAILED",
			label="normal",
			robot_contact=np.array([4.2] + [0.0] * 9),  # 1 robot collision frame (> 3.0 N)
			jaw_env_contact=np.zeros(10),
		),
		_write_synthetic_eval_episode(
			tmp_path / "s3",
			case=cases[3],
			verdict="SUCCESS",
			label="normal",
			robot_contact=np.zeros(10),
			jaw_env_contact=np.zeros(10),
		),
	]
	grades = [grade(Path(r["episode_dir"]), reference=ref) for r in records]
	teacher_by_seed = {c[0]: {"seed": c[0], "verdict": "SUCCESS", "label_correct": True} for c in cases}

	summary = eval_closed_loop.summarize_closed_loop(
		records, grades=grades, teacher_by_seed=teacher_by_seed, n_boot=500
	)
	assert summary["episodes"] == 4
	assert summary["sort_successes"] == 2
	assert summary["student_sort_rate"] == 0.5
	assert summary["defective_episodes"] == 2
	assert summary["missed_defects"] == 2
	assert summary["missed_defect_false_normal"] == 1
	assert summary["missed_defect_unknown"] == 1
	assert summary["collisions"]["jaw_env_contact_frames"] == 2
	assert summary["collisions"]["robot_contact_frames"] == 1
	assert summary["collisions"]["total_collisions"] == 3
	assert summary["collisions"]["collision_episodes"] == 2
	assert summary["collisions"]["by_verdict"]["UNKNOWN"]["jaw_env_contact_frames"] == 2
	assert summary["collisions"]["by_verdict"]["FAILED"]["robot_contact_frames"] == 1
	assert summary["pilot_gate"]["sort_rate_gate_pass"] is False  # 0.5 < 0.7 * 1.0
	assert summary["pilot_gate"]["collision_gate_pass"] is False
	assert summary["pilot_gate"]["pilot_gate_pass"] is False

	# Zero-collision, >= 0.7x teacher passes pilot gate
	records_clean = [
		_write_synthetic_eval_episode(
			tmp_path / f"clean_{i}",
			case=cases[i],
			verdict="SUCCESS" if i < 3 else "FAILED",
			label=("defective" if cases[i][1] else "normal") if i < 3 else "unknown",
			robot_contact=np.zeros(10),
			jaw_env_contact=np.zeros(10),
		)
		for i in range(4)
	]
	grades_clean = [grade(Path(r["episode_dir"]), reference=ref) for r in records_clean]
	sum_clean = eval_closed_loop.summarize_closed_loop(
		records_clean, grades=grades_clean, teacher_by_seed=teacher_by_seed, n_boot=500
	)
	assert sum_clean["student_sort_rate"] == 0.75
	assert sum_clean["collisions"]["total_collisions"] == 0
	assert sum_clean["pilot_gate"]["pilot_gate_pass"] is True


class _ConstantScorer:
	def threshold(self, camera: str) -> float:
		return 0.0

	def score(self, camera: str, image: np.ndarray) -> float:
		return 1.0


class _ScriptedChunkModel:
	"""Dummy model backed by recorded segments; validates D1 inputs and predicts chunks + end_step_probs."""

	family = "smolvla"

	def __init__(self, replay: ReplayPolicy, horizon: int = 50) -> None:
		self.replay = replay
		self.horizon = horizon
		self.seg_cursor = 0
		self.seen_states: list[np.ndarray] = []

	def predict_chunk(self, batch: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
		for cam in STUDENT_CAMERAS:
			arr = np.asarray(batch[cam])
			assert arr.dtype == np.float32
			assert float(arr.min()) >= 0.0 and float(arr.max()) <= 1.0
		state = np.asarray(batch["observation.state"])
		assert state.shape == (1, STATE_DIM)
		self.seen_states.append(state[0].copy())

		task_str = str(batch["task"][0])
		seg_frame = int(batch["segment_frame"])
		# Advance seg_cursor if segment_frame == 0 and instruction matches next segment
		if seg_frame == 0 and self.seg_cursor + 1 < len(self.replay.segments):
			cur_instr = self.replay.segments[self.seg_cursor].segment.instruction
			nxt_instr = self.replay.segments[self.seg_cursor + 1].segment.instruction
			if task_str == nxt_instr or (self.seg_cursor == 0 and task_str != cur_instr and task_str == nxt_instr):
				self.seg_cursor += 1
		# Also match by exact instruction from current position onward if needed
		if task_str != self.replay.segments[self.seg_cursor].segment.instruction:
			for j in range(self.seg_cursor, len(self.replay.segments)):
				if self.replay.segments[j].segment.instruction == task_str:
					self.seg_cursor = j
					break

		rec = self.replay.segments[self.seg_cursor]
		total_len = len(rec.actions)
		remaining = max(0, total_len - seg_frame)
		chunk = np.empty((self.horizon, ACTION_DIM), dtype=np.float32)
		if remaining > 0:
			take = min(self.horizon, remaining)
			chunk[:take] = rec.actions[seg_frame : seg_frame + take].astype(np.float32)
			if take < self.horizon:
				chunk[take:] = chunk[take - 1]
		else:
			chunk[:] = rec.actions[-1].astype(np.float32)

		probs = np.zeros((1, self.horizon + 2), dtype=np.float32)
		k_hat = remaining if remaining <= self.horizon else (self.horizon + 1)
		probs[0, k_hat] = 1.0
		return chunk[None, ...], probs


def test_mujoco_student_twin_smoke_with_http_chunk_client_policy(tmp_path):
	pytest.importorskip("mujoco")
	pytest.importorskip("langgraph")
	pytest.importorskip("av")
	pytest.importorskip("cv2")

	case = (600001, True, 0)
	scorer = _ConstantScorer()
	teacher_dir = tmp_path / "teacher"
	expert = run_episode(case, scorer, teacher_dir / "case-0000", sensing="realistic", video=False)
	(teacher_dir / "case-0000" / "task.json").write_text(json.dumps({"case": list(case)}), encoding="utf-8")
	assert expert["result"]["verdict"] == "SUCCESS"

	replay = ReplayPolicy.from_trace(teacher_dir / "case-0000")
	scripted_model = _ScriptedChunkModel(replay, horizon=50)
	local = LocalChunkPredictor(scripted_model, family="smolvla", horizon=50)

	eval_dir = tmp_path / "eval_out"
	with VLAPolicyServer(local, host="127.0.0.1", port=0) as srv, RemoteChunkPredictor(srv.url) as remote:
		report = eval_closed_loop.evaluate_closed_loop(
			predictor=remote,
			scorer=scorer,
			cases=[case],
			out_dir=eval_dir,
			mode="auto",
			replan_interval=25,
			teacher_runs=teacher_dir,
			run_teacher=False,
			n_boot=200,
		)

	assert report["mode_resolved"] == "async"
	s = report["summary"]
	assert s["episodes"] == 1
	assert s["sort_successes"] == 1
	assert s["student_sort_rate"] == 1.0
	assert s["missed_defects"] == 0
	assert s["collisions"]["total_collisions"] == 0
	assert s["pilot_gate"]["pilot_gate_pass"] is True
	assert (eval_dir / "eval_summary.json").exists()
	assert len(scripted_model.seen_states) > 0


def _write_episode(ep_dir: Path, events: list[dict]) -> None:
	ep_dir.mkdir(parents=True)
	(ep_dir / "episode.json").write_text(json.dumps({"result": {"verdict": "SUCCESS"}}), encoding="utf-8")
	(ep_dir / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")


def test_episode_record_detects_s5_perturbations_incl_nudge(tmp_path: Path) -> None:
	nudge = {"kind": "perturbation", "frame": 100, "perturbation": "post_putdown_nudge", "before_pos_m": [0, 0, 0]}
	seen = {"kind": "brick_perceived", "frame": 140, "moved": True, "position_m": [0.05, 0, 0]}
	_write_episode(tmp_path / "a", [nudge, seen])
	rec = eval_closed_loop._episode_record_from_dir(tmp_path / "a", (1, False, 0))
	assert rec["has_perturbation"] and rec["perturbation_detected"]

	drop = {"kind": "perturbation", "frame": 50, "perturbation": "mid_carry_drop", "before_pos_m": [0, 0, 0]}
	_write_episode(tmp_path / "b", [drop, {"kind": "failure", "frame": 80}])
	assert eval_closed_loop._episode_record_from_dir(tmp_path / "b", (2, True, 1))["perturbation_detected"]

	# A failure BEFORE the perturbation must not count as detecting it.
	_write_episode(tmp_path / "c", [{"kind": "failure", "frame": 10}, drop])
	assert not eval_closed_loop._episode_record_from_dir(tmp_path / "c", (3, True, 1))["perturbation_detected"]

	_write_episode(tmp_path / "d", [{"kind": "failure", "frame": 10}])
	rec = eval_closed_loop._episode_record_from_dir(tmp_path / "d", (4, True, 1))
	assert not rec["has_perturbation"] and not rec["perturbation_detected"]
