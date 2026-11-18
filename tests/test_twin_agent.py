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
	from digital_twin_armigo.agent.twin_record import CAMERAS, Video, student_frames
	from digital_twin_armigo.constants import TOP_FULL

	episode = tmp_path / "episode"
	branch = episode / "branch-0"
	branch.mkdir(parents=True)
	result = {"verdict": "SUCCESS", "student_eligible": True, "rewinds": 0}
	manifest = {
		"schema": "armigo-agent-episode/v1",
		"video": True,
		"result": result,
		"branches": [{"start_frame": 0}],
		"task": TASK,
		"fps": 30,
	}
	(episode / "episode.json").write_text(json.dumps(manifest))
	np.savez(branch / "actions.npz", state=np.zeros((2, 6)), action=np.ones((2, 6)))
	for name in CAMERAS:
		shape = (TOP_FULL[1], TOP_FULL[0], 3) if name == "top_rgb" else (2, 2, 3)
		video = Video(branch / f"{name}.mkv", shape, lossless=True)
		for value in (25, 37):
			video.put(np.full(shape, value, dtype=np.uint8))
		video.close()
	rows = list(student_frames(episode))
	assert len(rows) == 2
	assert np.all(rows[0]["observation.images.top_rgb_v2"] == 25)
	assert np.all(rows[1]["observation.images.wrist"] == 37)
	assert rows[0]["task"] == rows[1]["task"] == TASK
	assert set(rows[0]) == {"task", "action", "observation.state"} | {
		"observation.images." + x for x in ("front", "wrist", "top_rgb_v2", "top_depth_v2", "top_full")
	}
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
