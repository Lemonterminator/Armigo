"""Tests for Task 7: OU action-noise regularization and conditional ``noisy`` / ``mask.noisy`` recording."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from digital_twin_armigo.agent import student_rollout, twin_record
from digital_twin_armigo.constants import CAMERA_SHAPES, FPS, TOP_FULL


def _write_minimal_episode(
	ep_dir: Path,
	actions: np.ndarray,
	*,
	noisy: np.ndarray | None = None,
	segments: list[tuple[str, int, int, str]] | None = None,
) -> None:
	from digital_twin_armigo.agent.twin_graph import TASK
	from digital_twin_armigo.agent.twin_record import CAMERAS, Video

	n = len(actions)
	b0 = ep_dir / "branch-0"
	b0.mkdir(parents=True, exist_ok=True)
	if segments is None:
		segments = [("grasp", 0, n, "done")]
	seg_ids = np.zeros(n, dtype=np.int32)
	events = []
	for sid, (kind, start, end, reason) in enumerate(segments):
		seg_ids[start:end] = sid
		events.append(
			{
				"kind": "segment",
				"frame": start,
				"id": sid,
				"segment_kind": kind,
				"instruction": f"Seg {sid}",
				"params": {"bowl": "pink"} if kind == "carry_to_bowl" else {},
			}
		)
		events.append(
			{
				"kind": "segment_end",
				"frame": end,
				"id": sid,
				"frames": end - start,
				"reason": reason,
			}
		)
	(ep_dir / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
	npz_data: dict[str, np.ndarray] = {
		"state": np.zeros((n, 6), dtype=np.float32),
		"action": actions.astype(np.float32),
		"state_f64": np.zeros((n, 6), dtype=np.float64),
		"action_f64": actions.astype(np.float64),
		"segment": seg_ids,
	}
	if noisy is not None:
		npz_data["noisy"] = np.asarray(noisy, dtype=bool)
	np.savez_compressed(b0 / "actions.npz", **npz_data)

	shapes = {
		"front": CAMERA_SHAPES["front"],
		"wrist": CAMERA_SHAPES["wrist"],
		"top_rgb": [TOP_FULL[1], TOP_FULL[0], 3],
		"top_depth": CAMERA_SHAPES["top_depth_v2"],
	}
	for cam in CAMERAS:
		v = Video(b0 / f"{cam}.mkv", shapes[cam], lossless=True)
		for _ in range(n):
			v.put(np.zeros(shapes[cam], dtype=np.uint8))
		v.close()
	(ep_dir / "episode.json").write_text(
		json.dumps(
			{
				"schema": "armigo-agent-episode/v1",
				"fps": FPS,
				"task": TASK,
				"branches": [{"id": 0, "start_frame": 0, "actions": n}],
				"video": True,
				"result": {"verdict": "SUCCESS", "student_eligible": True, "rewinds": 0},
			}
		),
		encoding="utf-8",
	)


def test_student_rollout_reexports_preserved():
	assert hasattr(student_rollout, "POLICY_FIELDS")
	assert hasattr(student_rollout, "POLICY_PARAMS")
	assert hasattr(student_rollout, "SEGMENT_KINDS")
	assert student_rollout.POLICY_FIELDS is twin_record.POLICY_FIELDS
	assert student_rollout.SEGMENT_KINDS is twin_record.SEGMENT_KINDS


def test_action_noise_ou_process_properties():
	from digital_twin_armigo.agent.student_rollout import ActionNoise

	noise = ActionNoise(0.02, 0.3, ("grasp", "lift", "carry_to_bowl", "release_in_bowl", "scan_view"), seed=123)
	dt = 1.0 / FPS
	expected_rho = math.exp(-dt / 0.3)
	assert noise.rho(dt) == pytest.approx(expected_rho, rel=1e-12)
	assert noise.innovation_std(dt) == pytest.approx(0.02 * math.sqrt(1.0 - expected_rho**2), rel=1e-12)

	proc1 = noise.make_process(dt=dt)
	proc2 = noise.make_process(dt=dt)
	seq1 = np.stack([proc1.step() for _ in range(1500)])
	seq2 = np.stack([proc2.step() for _ in range(1500)])

	# Deterministic for a given seed, stateless config across processes
	assert np.array_equal(seq1, seq2)
	# Only arm joints 0..4 are perturbed; gripper joint 5 is always 0
	assert np.all(seq1[:, 5] == 0.0)
	assert np.all(seq1[:, :5].std(axis=0) > 0.01)
	assert 0.5 * 0.02 <= float(seq1[:, :5].std()) <= 1.5 * 0.02

	# Segment reset clears OU state to zero
	proc1.reset()
	assert np.all(proc1.state == 0.0)
	first_after_reset = proc1.step()
	proc_fresh = noise.make_process(dt=dt)
	first_fresh = proc_fresh.step()
	assert first_after_reset.shape == (6,) and first_after_reset[5] == 0.0
	assert np.std(first_after_reset[:5]) < 0.05
	assert np.all(first_fresh[:5] != 0.0)

	# Different seeds yield different draws
	other = ActionNoise(0.02, 0.3, ("grasp",), seed=124).make_process(dt=dt)
	assert not np.array_equal(other.step(), first_fresh)

	# Clipping bound respected when clip_rad is given
	clipped = ActionNoise(0.2, 0.3, ("grasp",), seed=99, clip_rad=0.015).make_process(dt=dt)
	vals = np.stack([clipped.step() for _ in range(300)])
	assert np.max(np.abs(vals[:, :5])) <= 0.015 + 1e-12
	assert np.all(vals[:, 5] == 0.0)

	# Validation
	with pytest.raises(ValueError):
		ActionNoise(-0.01, 0.3, ("grasp",), 0)
	with pytest.raises(ValueError):
		ActionNoise(0.02, 0.0, ("grasp",), 0)
	with pytest.raises(ValueError):
		ActionNoise(0.02, 0.3, ("unknown_segment",), 0)
	with pytest.raises(ValueError):
		ActionNoise(0.02, 0.3, ("grasp",), 0, clip_rad=-0.1)


def test_episode_trace_noisy_conditional_and_segment_features(tmp_path):
	from digital_twin_armigo.agent.twin_record import SEGMENT_FEATURES, EpisodeTrace

	assert "mask.noisy" in SEGMENT_FEATURES
	assert SEGMENT_FEATURES["mask.noisy"] == {"dtype": "bool", "shape": (1,), "names": None}

	# Noise off: noisy list tracks False, actions.npz omits "noisy"
	tr_off = EpisodeTrace(tmp_path / "off", video=False)
	assert tr_off.noisy == []
	tr_off.command(np.zeros(6), np.ones(6), "approach")
	tr_off.command(np.zeros(6), np.ones(6), "grasp", noisy=False)
	assert tr_off.noisy == [False, False]
	tr_off.close()
	with np.load(tmp_path / "off/branch-0/actions.npz", allow_pickle=False) as z:
		assert "noisy" not in z.files

	# Noise on: actions.npz writes bool "noisy" column
	tr_on = EpisodeTrace(tmp_path / "on", video=False)
	tr_on.command(np.zeros(6), np.ones(6), "approach", noisy=False)
	tr_on.command(np.zeros(6), np.ones(6), "grasp", noisy=True)
	assert tr_on.noisy == [False, True]
	tr_on.set_row_masks(noisy=[True, False])
	assert tr_on.noisy == [True, False]
	with pytest.raises(ValueError):
		tr_on.set_row_masks(noisy=[True])
	tr_on.close()
	with np.load(tmp_path / "on/branch-0/actions.npz", allow_pickle=False) as z:
		assert "noisy" in z.files
		assert z["noisy"].dtype == bool and z["noisy"].tolist() == [True, False]


def test_student_frames_mask_noisy_present_and_absent(tmp_path):
	pytest.importorskip("av")
	from digital_twin_armigo.agent.twin_record import student_frames

	actions = np.zeros((3, 6), dtype=np.float32)

	ep_absent = tmp_path / "ep_absent"
	_write_minimal_episode(ep_absent, actions, noisy=None)
	# Legacy 5-camera default omits mask.noisy for merge compatibility; v3 3-camera or explicit flag includes it
	frames_legacy = list(student_frames(ep_absent))
	assert all("mask.noisy" not in f for f in frames_legacy)
	frames_absent = list(student_frames(ep_absent, student_cameras_only=True))
	assert [bool(f["mask.noisy"][0]) for f in frames_absent] == [False, False, False]
	assert all(f["mask.noisy"].dtype == bool and f["mask.noisy"].shape == (1,) for f in frames_absent)

	ep_present = tmp_path / "ep_present"
	_write_minimal_episode(ep_present, actions, noisy=np.array([False, True, True], dtype=bool))
	frames_present = list(student_frames(ep_present, include_noisy_mask=True))
	assert [bool(f["mask.noisy"][0]) for f in frames_present] == [False, True, True]


def test_stitch_perturbation_episode_handles_noisy_conditionally(tmp_path):
	pytest.importorskip("av")
	from digital_twin_armigo.agent.twin_record import stitch_perturbation_episode

	def _setup_pair(name: str, p_noisy: np.ndarray | None, s_noisy: np.ndarray | None) -> Path:
		root = tmp_path / name
		p_dir = root / "parent"
		s_dir = root / "branch"
		_write_minimal_episode(
			p_dir,
			np.zeros((4, 6), dtype=np.float32),
			noisy=p_noisy,
			segments=[("grasp", 0, 2, "done"), ("carry_to_bowl", 2, 4, "done")],
		)
		p_manifest = json.loads((p_dir / "episode.json").read_text(encoding="utf-8"))
		p_manifest["branches"] = [{"id": 0, "start_frame": 0, "actions": 4}]
		(p_dir / "episode.json").write_text(json.dumps(p_manifest), encoding="utf-8")

		_write_minimal_episode(
			s_dir,
			np.ones((2, 6), dtype=np.float32),
			noisy=s_noisy,
			segments=[("carry_to_bowl", 0, 2, "done")],
		)
		s_manifest = json.loads((s_dir / "episode.json").read_text(encoding="utf-8"))
		s_manifest["branches"] = [{"id": 0, "start_frame": 2, "parent_branch": 0, "checkpoint": None}]
		(s_dir / "episode.json").write_text(json.dumps(s_manifest), encoding="utf-8")
		# Rewrite suffix events to start at frame 2 with resumed marker
		s_events = [
			{"kind": "resumed", "frame": 2, "branch": 0},
			{
				"kind": "segment",
				"frame": 2,
				"id": 0,
				"branch": 0,
				"segment_kind": "carry_to_bowl",
				"instruction": "Carry",
				"params": {"bowl": "pink"},
			},
			{"kind": "segment_end", "frame": 4, "id": 0, "branch": 0, "frames": 2, "reason": "done"},
		]
		(s_dir / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in s_events), encoding="utf-8")
		out_dir = root / "stitched"
		stitch_perturbation_episode(p_dir, 0, out_dir, branch_episode=s_dir)
		return out_dir / "branch-0" / "actions.npz"

	# 1. Neither has noisy -> omitted
	npz_none = _setup_pair("neither", None, None)
	with np.load(npz_none, allow_pickle=False) as z:
		assert "noisy" not in z.files

	# 2. Only suffix has noisy -> prefix padded False, suffix preserved
	npz_suffix = _setup_pair("suffix_only", None, np.array([True, False], dtype=bool))
	with np.load(npz_suffix, allow_pickle=False) as z:
		assert z["noisy"].dtype == bool
		assert z["noisy"].tolist() == [False, False, True, False]

	# 3. Only prefix has noisy (with True in prefix[:2]) -> prefix preserved, suffix padded False
	npz_prefix = _setup_pair("prefix_only", np.array([False, True, False, False], dtype=bool), None)
	with np.load(npz_prefix, allow_pickle=False) as z:
		assert z["noisy"].dtype == bool
		assert z["noisy"].tolist() == [False, True, False, False]

	# 4. Prefix noisy only AFTER fork, suffix clean -> no noisy frames remain, so omitted
	npz_after_fork = _setup_pair("after_fork_only", np.array([False, False, True, True], dtype=bool), None)
	with np.load(npz_after_fork, allow_pickle=False) as z:
		assert "noisy" not in z.files

	# 5. Both have noisy -> concatenated
	npz_both = _setup_pair(
		"both",
		np.array([True, False, True, True], dtype=bool),
		np.array([False, True], dtype=bool),
	)
	with np.load(npz_both, allow_pickle=False) as z:
		assert z["noisy"].tolist() == [True, False, False, True]


def test_legacy_600001_actions_npz_bitwise_identical(tmp_path):
	pytest.importorskip("mujoco")
	pytest.importorskip("langgraph")
	pytest.importorskip("av")
	pytest.importorskip("cv2")
	from digital_twin_armigo.agent.student_rollout import run_episode
	from digital_twin_armigo.inspection_scorer import FixedScorer

	fixture = Path(__file__).resolve().parent / "data" / "legacy_actions_600001.npz"
	assert fixture.is_file(), f"missing reference fixture {fixture}"

	scorer = FixedScorer(lambda _c, _r: -1.0, {"top": 0.0, "wrist_cam": 0.0})
	out_dir = tmp_path / "legacy_600001"
	run = run_episode((600001, False, 5), scorer, out_dir, sensing="truth", video=False, action_noise=None)
	assert run["result"]["verdict"] == "SUCCESS"

	with (
		np.load(out_dir / "branch-0" / "actions.npz", allow_pickle=False) as cur,
		np.load(fixture, allow_pickle=False) as ref,
	):
		assert "noisy" not in cur.files
		assert cur.files == ref.files
		for key in ref.files:
			assert cur[key].dtype == ref[key].dtype, key
			assert cur[key].shape == ref[key].shape, key
			assert np.array_equal(cur[key], ref[key]), key


def test_expert_tracer_action_noise_clean_targets_and_determinism(tmp_path):
	pytest.importorskip("mujoco")
	pytest.importorskip("langgraph")
	pytest.importorskip("av")
	pytest.importorskip("cv2")
	from digital_twin_armigo.agent.student_rollout import ActionNoise, ExpertTracer, ReplayPolicy
	from digital_twin_armigo.agent.twin_graph import Policy, build_graph
	from digital_twin_armigo.agent.twin_mujoco import sample_case
	from digital_twin_armigo.agent.twin_record import EpisodeTrace, segment_labels
	from digital_twin_armigo.inspection_scorer import FixedScorer

	case = (600001, False, 5)
	scorer = FixedScorer(lambda _c, _r: -1.0, {"top": 0.0, "wrist_cam": 0.0})
	noise = ActionNoise(0.01, 0.3, ("grasp", "lift", "carry_to_bowl", "release_in_bowl", "scan_view"), seed=1)

	def _run_traced(out_dir: Path) -> tuple[dict, np.ndarray, np.ndarray]:
		trace = EpisodeTrace(out_dir, video=False)
		twin = ExpertTracer(sample_case(*case), scorer, trace, sensing="truth", action_noise=noise)
		teacher_targets: list[np.ndarray] = []
		orig_act = twin.expert.act

		def spy_act(obs):
			target, phase = orig_act(obs)
			if not twin.expert.done:
				teacher_targets.append(np.asarray(target, dtype=np.float64).copy())
			return target, phase

		twin.expert.act = spy_act
		try:
			result = build_graph(twin, Policy()).invoke({}, config={"recursion_limit": 64})["result"]
			trace.finish(result, {"engine": "mujoco", "sensing": "truth", "case": list(case), "student": False})
			executed = np.asarray(twin.targets, dtype=np.float64).copy()
		finally:
			twin.close()
		return result, np.asarray(teacher_targets, dtype=np.float64), executed

	out1 = tmp_path / "run1"
	out2 = tmp_path / "run2"
	res1, teacher1, exec1 = _run_traced(out1)
	res2, teacher2, exec2 = _run_traced(out2)
	assert res1["verdict"] == "SUCCESS" and res1["rewinds"] == 0
	assert res2["verdict"] == "SUCCESS" and res2["rewinds"] == 0

	# Determinism for same seed
	assert np.array_equal(teacher1, teacher2)
	assert np.array_equal(exec1, exec2)
	with (
		np.load(out1 / "branch-0" / "actions.npz", allow_pickle=False) as z1,
		np.load(out2 / "branch-0" / "actions.npz", allow_pickle=False) as z2,
	):
		assert z1.files == z2.files
		for k in z1.files:
			assert np.array_equal(z1[k], z2[k]), k

		# Clean-target supervision: recorded action_f64 == teacher generator targets (and ReplayPolicy matches)
		assert np.array_equal(z1["action_f64"], teacher1)
		replay = ReplayPolicy.from_trace(out1)
		replayed_actions = np.concatenate([s.actions for s in replay.segments], axis=0)
		assert np.array_equal(replayed_actions, teacher1)

		# Conditional noisy column matches noise.kinds
		labels = segment_labels(out1)
		expected_noisy = np.array([k in noise.kinds for k in labels["kind"]], dtype=bool)
		assert "noisy" in z1.files
		assert z1["noisy"].dtype == bool
		assert np.array_equal(z1["noisy"], expected_noisy)
		assert expected_noisy.any() and not expected_noisy.all()

		# Executed targets differ only on noisy frames & arm joints 0..4; gripper (dim 5) diff == 0
		diff = exec1 - z1["action_f64"]
		assert np.all(diff[:, 5] == 0.0)
		assert np.all(diff[~expected_noisy] == 0.0)
		noisy_arm_diff = diff[expected_noisy, :5]
		assert 0.5 * noise.sigma_rad <= float(noisy_arm_diff.std()) <= 1.5 * noise.sigma_rad


def test_run_episode_with_video_and_student_frames_mask_noisy(tmp_path):
	pytest.importorskip("mujoco")
	pytest.importorskip("langgraph")
	pytest.importorskip("av")
	pytest.importorskip("cv2")
	from digital_twin_armigo.agent.student_rollout import ActionNoise, run_episode
	from digital_twin_armigo.agent.twin_record import student_frames
	from digital_twin_armigo.inspection_scorer import FixedScorer

	scorer = FixedScorer(lambda _c, _r: 1.0, {"top": 0.0, "wrist_cam": 0.0})
	noise = ActionNoise(0.01, 0.3, ("grasp", "lift", "carry_to_bowl", "release_in_bowl", "scan_view"), seed=1)
	out_dir = tmp_path / "ep_vid"
	run = run_episode((600001, True, 0), scorer, out_dir, sensing="realistic", video=True, action_noise=noise)
	assert run["result"]["verdict"] == "SUCCESS"
	assert run["result"]["student_eligible"] is True

	with np.load(out_dir / "branch-0" / "actions.npz", allow_pickle=False) as z:
		noisy_col = z["noisy"].tolist()
	frames = list(student_frames(out_dir, student_cameras_only=True))
	assert len(frames) == len(noisy_col)
	assert [bool(f["mask.noisy"][0]) for f in frames] == noisy_col
	assert any(noisy_col) and not all(noisy_col)
