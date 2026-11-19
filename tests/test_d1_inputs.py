"""Tests for D1 input logging, noisy D415 top_depth_v2, S1 Option B, S7 export contract, and StudentObservation parity."""

from __future__ import annotations

import json

import numpy as np
import pytest


def test_episode_trace_command_logs_gripper_cmd_and_effort_frac_compatibly(tmp_path):
	from digital_twin_armigo.agent.twin_record import EpisodeTrace

	# Legacy call without gripper_cmd / effort_frac still works and omits those columns.
	legacy = EpisodeTrace(tmp_path / "legacy", video=False)
	legacy.command(np.zeros(6), np.ones(6), "approach")
	legacy.close()
	with np.load(tmp_path / "legacy" / "branch-0" / "actions.npz") as npz:
		assert "gripper_cmd" not in npz.files and "effort_frac" not in npz.files

	# With keyword args, both are stored as float32 (N,).
	full = EpisodeTrace(tmp_path / "full", video=False)
	full.command(np.zeros(6), np.ones(6), "grasp", gripper_cmd=0.42, effort_frac=0.81)
	full.command(np.zeros(6), np.ones(6), "grasp", gripper_cmd=0.10, effort_frac=0.95)
	full.close()
	with np.load(tmp_path / "full" / "branch-0" / "actions.npz") as npz:
		assert npz["gripper_cmd"].dtype == np.float32 and npz["gripper_cmd"].shape == (2,)
		assert npz["effort_frac"].dtype == np.float32 and npz["effort_frac"].shape == (2,)
		assert np.allclose(npz["gripper_cmd"], [0.42, 0.10], atol=1e-6)
		assert np.allclose(npz["effort_frac"], [0.81, 0.95], atol=1e-6)


def test_s1_option_b_gripper_contact_and_monitor_ignore_effort():
	from digital_twin_armigo.agent.sensing import FORCE_AT_FULL_EFFORT_N, GripperContact
	from digital_twin_armigo.mujoco_cell.gripper_state import (
		GripperMonitor,
		GripperMonitorParams,
		GripperState,
	)

	# GripperContact default keeps effort check; use_effort=False ignores effort_frac.
	g_default = GripperContact()
	assert g_default.update(0.06, 0.2, 0.0).tolist() == [0.0, 0.0]

	g_no_eff = GripperContact(use_effort=False)
	assert g_no_eff.update(0.06, 0.2, 0.0).tolist() == [FORCE_AT_FULL_EFFORT_N, FORCE_AT_FULL_EFFORT_N]
	assert g_no_eff.update(0.06, 0.2, 0.9).tolist() == [FORCE_AT_FULL_EFFORT_N, FORCE_AT_FULL_EFFORT_N]
	assert g_no_eff.update(0.06, 0.07, 0.9).tolist() == [0.0, 0.0]

	# GripperMonitor default requires effort_frac >= 0.70; effort_min=0.0 / use_effort=False ignores it.
	mon_default = GripperMonitor()
	assert mon_default.p.effort_min == pytest.approx(0.70)
	for _ in range(mon_default.p.blocked_frames):
		s, _ = mon_default.update(0.06, 0.76, 0.0, (0.0, 0.0))
	assert s is not GripperState.BLOCKED

	mon_no_eff = GripperMonitor(GripperMonitorParams(effort_min=0.0, use_effort=False))
	for _ in range(mon_no_eff.p.blocked_frames):
		s, d = mon_no_eff.update(0.06, 0.76, 0.0, (0.0, 0.0))
	assert s is GripperState.BLOCKED and d == -1


def test_initial_q_parity_between_twin_record_and_twin_mujoco():
	pytest.importorskip("mujoco")
	from digital_twin_armigo.agent.twin_mujoco import INITIAL_Q as MUJOCO_INITIAL_Q
	from digital_twin_armigo.agent.twin_record import INITIAL_Q as RECORD_INITIAL_Q

	assert np.array_equal(RECORD_INITIAL_Q, MUJOCO_INITIAL_Q)


def test_student_frames_legacy_fallback_without_gripper_cmd(tmp_path):
	import av

	from digital_twin_armigo.agent.twin_record import CAMERAS, INITIAL_Q, Video, student_frames
	from digital_twin_armigo.agent.twin_graph import TASK
	from digital_twin_armigo.constants import CAMERA_SHAPES, FPS, TOP_FULL

	ep = tmp_path / "ep"
	b0 = ep / "branch-0"
	b0.mkdir(parents=True)
	actions = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.2, 0.3, 0.4, 0.5, 0.6, 0.15]], dtype=np.float32)
	states = np.zeros((2, 6), dtype=np.float32)
	np.savez(b0 / "actions.npz", state=states, action=actions, segment=np.array([0, 0], dtype=np.int32))
	events = [
		{"kind": "segment", "frame": 0, "id": 0, "segment_kind": "grasp", "instruction": "Grasp", "params": {}},
		{"kind": "segment_end", "frame": 2, "id": 0, "frames": 2, "reason": "done"},
	]
	(ep / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
	shapes = {
		"front": CAMERA_SHAPES["front"],
		"wrist": CAMERA_SHAPES["wrist"],
		"top_rgb": [TOP_FULL[1], TOP_FULL[0], 3],
		"top_depth": CAMERA_SHAPES["top_depth_v2"],
	}
	for cam in CAMERAS:
		v = Video(b0 / f"{cam}.mkv", shapes[cam], lossless=True)
		for _ in range(2):
			v.put(np.zeros(shapes[cam], dtype=np.uint8))
		v.close()
	(ep / "episode.json").write_text(
		json.dumps(
			{
				"schema": "armigo-agent-episode/v1",
				"fps": FPS,
				"task": TASK,
				"branches": [{"id": 0, "start_frame": 0, "actions": 2}],
				"video": True,
				"result": {"verdict": "SUCCESS", "student_eligible": True, "rewinds": 0},
			}
		),
		encoding="utf-8",
	)
	frames = list(student_frames(ep))
	assert len(frames) == 2
	assert float(frames[0]["observation.gripper_cmd"][0]) == pytest.approx(float(INITIAL_Q[5]), abs=1e-6)
	assert float(frames[1]["observation.gripper_cmd"][0]) == pytest.approx(0.6, abs=1e-6)


class _ClearScorer:
	def threshold(self, camera):
		return 0.0

	def score(self, camera, image):
		return -1.0

	def close(self):
		pass


def test_realistic_top_depth_v2_uses_independent_d415_rng_and_perception_is_unchanged(tmp_path):
	"""Noisy top_depth rendering in realistic mode uses seed + 40_000 and leaves _depth_rng, top_rgb,
	detector keyframes, perception, and actions bit-identical between video=True and video=False."""
	pytest.importorskip("mujoco")
	pytest.importorskip("cv2")
	pytest.importorskip("av")
	pytest.importorskip("langgraph")
	from digital_twin_armigo.agent.student_rollout import run_episode
	from digital_twin_armigo.agent.twin_mujoco import TwinMujoco, sample_case
	from digital_twin_armigo.agent.twin_record import EpisodeTrace
	from digital_twin_armigo.constants import TOP_CROP
	from digital_twin_armigo.depth import layered_height_image

	case = (600001, False, 5)
	no_video = run_episode(case, _ClearScorer(), tmp_path / "novid", sensing="realistic", video=False)
	with_video = run_episode(case, _ClearScorer(), tmp_path / "vid", sensing="realistic", video=True)

	assert no_video["digest"] == with_video["digest"]
	assert no_video["result"]["smoothness"] == with_video["result"]["smoothness"]
	with (
		np.load(tmp_path / "novid/branch-0/actions.npz") as a,
		np.load(tmp_path / "vid/branch-0/actions.npz") as b,
	):
		for key in ("state_f64", "action_f64", "segment", "gripper_cmd", "effort_frac"):
			assert np.array_equal(a[key], b[key]), key

	# Check rendered images: top_rgb is clean RGB, top_depth has D415 noise when realistic vs clean in truth.
	for sensing in ("realistic", "truth"):
		trace = EpisodeTrace(tmp_path / f"render_{sensing}", video=True)
		twin = TwinMujoco(sample_case(*case), _ClearScorer(), trace, sensing=sensing)
		try:
			imgs = trace.images
			clean_rgb = twin.cell.rgb("top")
			z, valid = twin.cell.height("top", TOP_CROP)
			clean_depth = layered_height_image(z, valid)
			assert np.array_equal(imgs["top_rgb"], clean_rgb)
			if sensing == "realistic":
				assert not np.array_equal(imgs["top_depth"], clean_depth)
			else:
				assert np.array_equal(imgs["top_depth"], clean_depth)
		finally:
			trace.close()
			twin.close()


def test_realistic_teacher_is_invariant_to_effort_frac(tmp_path, monkeypatch):
	"""Under S1 Option B, corrupting obs['effort_frac'] in realistic TwinMujoco does not change any action."""
	pytest.importorskip("mujoco")
	pytest.importorskip("cv2")
	pytest.importorskip("av")
	pytest.importorskip("langgraph")
	from digital_twin_armigo.agent import twin_mujoco as tm
	from digital_twin_armigo.agent.student_rollout import run_episode

	case = (600001, False, 5)
	baseline = run_episode(case, _ClearScorer(), tmp_path / "base", sensing="realistic", video=False)

	orig_observe = tm.observe

	def zero_effort_observe(cell, ep):
		obs = orig_observe(cell, ep)
		obs["effort_frac"] = 0.0
		return obs

	monkeypatch.setattr(tm, "observe", zero_effort_observe)
	zeroed = run_episode(case, _ClearScorer(), tmp_path / "zero_effort", sensing="realistic", video=False)

	assert baseline["digest"] == zeroed["digest"]
	with (
		np.load(tmp_path / "base/branch-0/actions.npz") as a,
		np.load(tmp_path / "zero_effort/branch-0/actions.npz") as b,
	):
		assert np.array_equal(a["action_f64"], b["action_f64"])
		assert np.array_equal(a["state_f64"], b["state_f64"])
		assert np.any(a["effort_frac"] > 0.0)
		assert np.all(b["effort_frac"] == 0.0)


def test_student_observation_parity_across_segments_and_boundaries(tmp_path):
	"""StudentTwin provides q_prev, prev_action, and gripper_cmd matching offline actions.npz across segments."""
	pytest.importorskip("mujoco")
	pytest.importorskip("cv2")
	pytest.importorskip("av")
	pytest.importorskip("langgraph")
	from digital_twin_armigo.agent.student_rollout import ReplayPolicy, run_episode
	from digital_twin_armigo.agent.twin_mujoco import INITIAL_Q

	case = (600001, False, 5)
	expert_dir = tmp_path / "expert"
	run_episode(case, _ClearScorer(), expert_dir, sensing="realistic", video=False)
	with np.load(expert_dir / "branch-0" / "actions.npz") as npz:
		exp_state = npz["state_f64"].copy()
		exp_action = npz["action_f64"].copy()
		exp_gcmd = npz["gripper_cmd"].copy()

	observed: list[tuple[int, int, int, np.ndarray, np.ndarray, np.ndarray, float, bool]] = []
	inner = ReplayPolicy.from_trace(expert_dir)

	class RecordingReplay:
		def start(self, task, segment_id):
			inner.start(task, segment_id)

		def act(self, obs):
			step = inner.act(obs)
			assert obs.q_prev is not None and obs.prev_action is not None
			assert obs.q_prev.shape == (6,) and obs.q_prev.dtype == np.float64
			assert obs.prev_action.shape == (6,) and obs.prev_action.dtype == np.float64
			assert isinstance(obs.gripper_cmd, float)
			observed.append(
				(
					obs.frame,
					obs.segment_id,
					obs.segment_frame,
					obs.q.copy(),
					obs.q_prev.copy(),
					obs.prev_action.copy(),
					obs.gripper_cmd,
					step.done,
				)
			)
			return step

	student_dir = tmp_path / "student"
	run_episode(
		case,
		_ClearScorer(),
		student_dir,
		policy=RecordingReplay(),
		segment_monitor=inner.check_segment,
		sensing="realistic",
		video=False,
	)
	assert inner.state_mismatches == 0

	# Every non-done observation matches offline state / q_prev / prev_action / gripper_cmd exactly.
	step_obs = [row for row in observed if not row[7]]
	assert len(step_obs) == len(exp_state)
	for f, _sid, _sf, q, q_prev, prev_action, gcmd, _done in step_obs:
		assert np.array_equal(q, exp_state[f])
		expected_qp = exp_state[0] if f == 0 else exp_state[f - 1]
		expected_pa = INITIAL_Q if f == 0 else exp_action[f - 1]
		assert np.array_equal(q_prev, expected_qp)
		assert np.array_equal(prev_action, expected_pa)
		assert gcmd == pytest.approx(float(exp_gcmd[f]), abs=1e-6)

	# At every segment boundary (done=True before a later segment), the next segment's first obs has identical inputs.
	for i, row in enumerate(observed[:-1]):
		if row[7]:
			nxt = observed[i + 1]
			assert row[0] == nxt[0]  # same global frame
			assert row[1] + 1 == nxt[1] and nxt[2] == 0  # next segment, segment_frame=0
			assert np.array_equal(row[3], nxt[3])  # q
			assert np.array_equal(row[4], nxt[4])  # q_prev
			assert np.array_equal(row[5], nxt[5])  # prev_action
			assert row[6] == nxt[6]  # gripper_cmd


def test_top_depth_rng_persists_across_checkpoint_resume(tmp_path):
	"""Checkpoint/restore preserves _top_depth_rng bit-for-bit so post-resume D415 depth noise stays identical."""
	pytest.importorskip("mujoco")
	pytest.importorskip("cv2")
	pytest.importorskip("av")
	from digital_twin_armigo.agent.twin_mujoco import TwinMujoco, sample_case
	from digital_twin_armigo.agent.twin_record import EpisodeTrace
	from digital_twin_armigo.agent.twin_resume import ResumePlan, live_checkpoint_ids, load_checkpoint, load_history, restore
	from digital_twin_armigo.constants import TOP_CROP, TOP_FULL
	from digital_twin_armigo.depth import layered_height_image
	from digital_twin_armigo.sensor_noise import DEFAULT_D415_NOISE, degrade_depth_height

	case = (600001, False, 5)
	parent_dir = tmp_path / "parent"
	trace = EpisodeTrace(parent_dir, video=False)
	twin = TwinMujoco(sample_case(*case), _ClearScorer(), trace, sensing="realistic")
	try:
		assert twin.prepare() is None
		twin._save([])
		z, valid = twin.cell.height("top", TOP_CROP)
		top_pose = twin.cell._cam_poses["top"]
		z_noisy, v_noisy = degrade_depth_height(
			z, valid, top_pose, twin._top_depth_rng, DEFAULT_D415_NOISE, crop=TOP_CROP, full_shape=(TOP_FULL[1], TOP_FULL[0])
		)
		expected_depth = layered_height_image(z_noisy, v_noisy)
		after_state = twin._top_depth_rng.bit_generator.state
	finally:
		trace.close()
		twin.close()

	ids = live_checkpoint_ids(parent_dir / "events.jsonl")
	ck = load_checkpoint(parent_dir, ids[-1])
	assert "top_depth_rng" in ck.agent
	plan = ResumePlan(parent_dir, [ck], [], None, *load_history(parent_dir, ck))

	resumed_dir = tmp_path / "resumed"
	resumed_trace = EpisodeTrace(resumed_dir, video=False)
	fresh = TwinMujoco(sample_case(*case), _ClearScorer(), resumed_trace, sensing="realistic")
	try:
		restore(fresh, plan)
		assert fresh.record.top_depth_rng is fresh._top_depth_rng
		assert fresh._top_depth_rng.bit_generator.state == ck.agent["top_depth_rng"]
		z_r, v_r = degrade_depth_height(
			z, valid, top_pose, fresh._top_depth_rng, DEFAULT_D415_NOISE, crop=TOP_CROP, full_shape=(TOP_FULL[1], TOP_FULL[0])
		)
		actual_depth = layered_height_image(z_r, v_r)
		assert fresh._top_depth_rng.bit_generator.state == after_state
	finally:
		resumed_trace.close()
		fresh.close()
	assert np.array_equal(actual_depth, expected_depth)
