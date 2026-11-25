import json

import numpy as np
import pytest

from digital_twin_armigo.agent.llm import FakeLLM
from digital_twin_armigo.agent.twin_graph import Policy, build_graph, decision, select_skill


def view(name, faces, score):
	return {"id": name, "faces": faces, "score": score, "threshold": 0.0}


def test_negative_decisions_need_scored_coverage_of_all_six_faces():
	assert decision([view("w", [0, 1, 2, 3, 4], -1), view("b", [5], None)])[0] == "unknown"
	assert decision([view("b", [], 10)])[0] == "unknown"
	assert decision([view("w", [0, 1, 2, 3, 4], -1), view("b", [5], -1)])[0] == "normal"
	assert decision([view("b", [5], 0)])[0] == "defective"


def test_llm_cannot_invent_a_motion_or_override_policy_constraints():
	llm = FakeLLM(['{"skill":"place_in_pink"}'])
	action, source, info = select_skill(llm, ["scan_wrist", "present_bottom"], {})
	assert (action, source, info["reason"]) == ("scan_wrist", "fallback", "illegal_choice")
	with pytest.raises(ValueError):
		Policy(max_rewinds=True)
	with pytest.raises(ValueError):
		Policy(max_bottom_views=4)


def test_rollback_keeps_failure_history_and_requires_fresh_bottom_evidence():
	pytest.importorskip("langgraph")

	class Backend:
		attempt = 0
		failed = False

		def enter(self, node):
			pass

		def prepare(self):
			pass

		def can_scan_wrist(self, policy):
			return True

		def can_present_bottom(self, policy):
			return True

		def choose(self, allowed, views, failures):
			return allowed[0]

		def inspect(self, node, views):
			if node == "scan_wrist":
				return [view("prefix/wrist", [0, 1, 2, 3, 4], -1)], None
			self.attempt += 1
			return [view(f"branch-{self.attempt}/bottom", [5], 1)], None

		def place(self, label):
			assert label == "defective"
			return "failed" if not self.failed else None

		def failure(self, reason, views):
			assert reason == "failed"
			self.failed = True
			return {"discarded": views[1]["id"]}

		def rewind(self, policy, failures):
			assert failures == [{"discarded": "branch-1/bottom"}]
			return [view("prefix/wrist", [0, 1, 2, 3, 4], -1)]

		def evaluate(self, label, views, failures):
			return {"label": label, "views": views, "failures": failures}

	backend = Backend()
	result = build_graph(backend, Policy()).invoke({})["result"]
	assert result["label"] == "defective"
	assert [x["id"] for x in result["views"]] == ["prefix/wrist", "branch-2/bottom"]
	assert result["failures"] == [{"discarded": "branch-1/bottom"}]


def test_recorded_camera_observation_precedes_its_action_and_rewound_trace_is_rejected(tmp_path):
	pytest.importorskip("av")
	pytest.importorskip("cv2")
	from digital_twin_armigo.agent.twin_graph import TASK
	from digital_twin_armigo.agent.twin_record import (
		CAMERAS,
		INPUT_FEATURES,
		SEGMENT_FEATURES,
		SegmentLabelError,
		Video,
		student_frames,
	)
	from digital_twin_armigo.constants import TOP_FULL

	episode = tmp_path / "episode"
	branch = episode / "branch-0"
	branch.mkdir(parents=True)
	# An in-place retry keeps one continuous branch, so it stays exportable (a recovery demonstration).
	result = {"verdict": "SUCCESS", "student_eligible": True, "rewinds": 0, "retries": 1}
	manifest = {
		"schema": "armigo-agent-episode/v1",
		"video": True,
		"result": result,
		"branches": [{"start_frame": 0}],
		"task": TASK,
		"fps": 30,
	}
	(episode / "episode.json").write_text(json.dumps(manifest))
	# wrist_roll 2.8 rad lies outside the Studio [-100, 100] range; radians are exported unchanged.
	action = np.array([[0.1, -0.2, 0.3, -0.4, 2.8, 1.0]] * 2, dtype=np.float32)
	np.savez(branch / "actions.npz", state=np.zeros((2, 6)), action=action, segment=np.array([0, 1], dtype=np.int32))
	events = []
	for sid, (kind, text) in enumerate((("prepare", "Wait at rest."), ("scan_view", "Move the wrist camera."))):
		# Recorded scan_view segments always carry view / views: policy_inputs() formats the instruction from them.
		params = {"view": 1, "views": 1} if kind == "scan_view" else {}
		events += [
			{"kind": "segment", "frame": sid, "id": sid, "segment_kind": kind, "instruction": text, "params": params},
			{"kind": "segment_end", "frame": sid + 1, "id": sid, "frames": 1, "reason": "done"},
		]
	(episode / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
	for name in CAMERAS:
		shape = (TOP_FULL[1], TOP_FULL[0], 3) if name == "top_rgb" else (2, 2, 3)
		video = Video(branch / f"{name}.mkv", shape, lossless=True)
		for value in (25, 37):
			video.put(np.full(shape, value, dtype=np.uint8))
		video.close()
	rows = list(student_frames(episode))
	assert len(rows) == 2
	assert np.array_equal(rows[0]["action"], action[0]) and rows[0]["action"].dtype == np.float32
	assert np.array_equal(rows[1]["observation.state"], np.zeros(6))
	assert np.all(rows[0]["observation.images.top_rgb_v2"] == 25)
	assert np.all(rows[1]["observation.images.wrist"] == 37)
	# The LeRobot task is the frame's segment instruction; the episode goal (TASK) stays in the manifest.
	assert [r["task"] for r in rows] == ["Wait at rest.", "Move the wrist camera."]
	assert [r["segment.kind"] for r in rows] == ["prepare", "scan_view"]
	assert [int(r["segment.index"][0]) for r in rows] == [0, 1]
	assert [bool(r["segment.done"][0]) for r in rows] == [False, True]  # segment 0 ended at observation 1
	assert [int(r["segment.chunk_valid_len"][0]) for r in rows] == [1, 1]
	# The legacy 5-camera export keeps the pre-v3 feature set (no mask.noisy) so old shards still merge (WP5).
	assert set(rows[0]) == {"task", "action", "observation.state"} | {
		"observation.images." + x for x in ("front", "wrist", "top_rgb_v2", "top_depth_v2", "top_full")
	} | set(INPUT_FEATURES) | (set(SEGMENT_FEATURES) - {"mask.noisy"})
	# A recording without segment labels (segment == -1) is not exportable.
	np.savez(branch / "actions.npz", state=np.zeros((2, 6)), action=action, segment=np.array([-1, -1], dtype=np.int32))
	with pytest.raises(SegmentLabelError, match="no segment"):
		next(student_frames(episode))
	result["rewinds"] = 1
	(episode / "episode.json").write_text(json.dumps(manifest))
	with pytest.raises(ValueError, match="not eligible"):
		next(student_frames(episode))


def test_live_preview_retries_after_a_windows_file_lock(tmp_path, monkeypatch):
	pytest.importorskip("av")
	pytest.importorskip("cv2")
	from pathlib import Path

	from digital_twin_armigo.agent.twin_record import EpisodeTrace

	trace = EpisodeTrace(tmp_path / "episode", video=True)
	pixels = np.zeros((20, 20, 3), dtype=np.uint8)
	monkeypatch.setattr(trace, "_composite", lambda *_args: pixels)
	try:
		trace._publish_preview("prepare")
		published = trace.path / "live.jpg"
		previous = published.read_bytes()
		pixels.fill(120)

		def locked(_source, _target):
			raise PermissionError("live preview open in browser")

		with monkeypatch.context() as patch:
			patch.setattr(Path, "replace", locked)
			trace._publish_preview("scan_wrist")
		assert published.read_bytes() == previous
		trace._publish_preview("scan_wrist")
		assert published.read_bytes() != previous
	finally:
		trace.close()


def test_replay_node_strip_fits_the_frame(tmp_path):
	pytest.importorskip("av")
	pytest.importorskip("cv2")
	from digital_twin_armigo.agent.twin_graph import NODES
	from digital_twin_armigo.agent.twin_record import EpisodeTrace

	trace = EpisodeTrace(tmp_path / "episode", video=False)
	trace.images = dict.fromkeys(("front", "wrist", "top_rgb"), np.zeros((20, 20, 3), dtype=np.uint8))
	try:
		frame = trace._composite("evaluate", frame=0)
	finally:
		trace.close()

	background = frame[0, 0]
	strip = frame[690]
	painted = np.flatnonzero(np.any(strip != background, axis=1))
	boxes = np.split(painted, np.flatnonzero(np.diff(painted) > 1) + 1)
	assert len(boxes) == len(NODES)
	assert painted[0] >= 12 and painted[-1] < frame.shape[1] - 12
	assert np.all(strip[boxes[-1]] == (44, 164, 112))
	assert np.all(frame[653:698, np.all(strip == background, axis=1)] == background)


def test_continuous_carry_preserves_the_path_duration_and_gripper_transition():
	pytest.importorskip("mujoco")
	from digital_twin_armigo.agent.twin_mujoco import ContinuousCarryExpert
	from digital_twin_armigo.mujoco_cell.physics_expert import ExpertParams
	from digital_twin_armigo.mujoco_cell.table_relocation import RelocateExpert

	expert = ContinuousCarryExpert.__new__(ContinuousCarryExpert)
	expert.p = ExpertParams()
	expert.last = np.zeros(6)
	expert._lo, expert._hi = np.full(6, -np.pi), np.full(6, np.pi)  # joint range; the target lies inside it
	target = np.array([0.5, -1.0, 0.0, 0.0, 0.0, 0.0])
	smooth = np.array([q for q, _ in expert._move(target, 4.2, "show_move")])
	baseline = np.array([q for q, _ in RelocateExpert._move(expert, target, 4.2, "show_move")])
	assert len(smooth) == len(baseline)
	np.testing.assert_array_equal(smooth[-1], target)
	assert np.all(np.diff(smooth[:, 0]) > 0)
	np.testing.assert_allclose(smooth, (smooth[:, 0] / target[0])[:, None] * target)
	assert expert.smoothed_moves == 1

	target[-1] = 0.5
	transition = np.array([q for q, _ in expert._move(target, 4.2, "place")])
	baseline = np.array([q for q, _ in RelocateExpert._move(expert, target, 4.2, "place")])
	np.testing.assert_array_equal(transition, baseline)


def test_carry_replay_compares_archived_commands_and_measured_end_state():
	from types import SimpleNamespace

	pytest.importorskip("mcp")
	pytest.importorskip("mujoco")
	pytest.importorskip("av")
	pytest.importorskip("cv2")
	from digital_twin_armigo.agent.twin_mcp import CarryReplayCheck

	start = np.zeros((2, 6))
	end_position = np.zeros(6)
	command = np.full(6, 0.75)
	rows = [(None, command, "return"), (None, command, "return")]
	cell = SimpleNamespace(joints=lambda: end_position, data=SimpleNamespace(qvel=start[1]), dofadr=list(range(6)))
	backend = SimpleNamespace(
		cell=cell, record=SimpleNamespace(rows=rows, events=[]), expert=SimpleNamespace(smoothed_moves=1)
	)
	check = CarryReplayCheck()
	result = {"student_eligible": True}
	assert check.finish(backend, start, result) is None
	rows[-1] = (None, np.nextafter(command, np.inf), "return")
	assert check.finish(backend, start, result)["accepted"]

	end_position[0] = 0.02
	rejected = check.finish(backend, start, result)
	assert rejected["boundary_commands_match"]
	assert not rejected["accepted"]
