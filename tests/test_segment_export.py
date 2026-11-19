"""Segment labels in the production recording path and the S7 export columns (agent/twin_record.py)."""

from __future__ import annotations

import json

import numpy as np
import pytest


def _episode(tmp_path, ids, segments):
	"""A minimal episode directory: the per-frame segment column and its segment / segment_end events."""
	episode = tmp_path / "episode"
	(episode / "branch-0").mkdir(parents=True)
	np.savez(episode / "branch-0" / "actions.npz", segment=np.asarray(ids, dtype=np.int32))
	events = []
	for sid, (kind, start, end, reason) in enumerate(segments):
		events.append(
			{
				"kind": "segment",
				"frame": start,
				"id": sid,
				"segment_kind": kind,
				"instruction": f"do {kind}",
				"params": {},
			}
		)
		events.append({"kind": "segment_end", "frame": end, "id": sid, "frames": end - start, "reason": reason})
	(episode / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
	return episode


def test_segment_labels_done_and_chunk_valid_len(tmp_path):
	from digital_twin_armigo.agent.twin_record import segment_labels

	# lift ends on a lost brick (a boundary, not segment_done); let_go has no frames (done at once).
	segments = [
		("grasp", 0, 3, "done"),
		("lift", 3, 5, "brick_lost"),
		("let_go", 5, 5, "done"),
		("return_ready", 5, 7, "done"),
	]
	labels = segment_labels(_episode(tmp_path, [0, 0, 0, 1, 1, 3, 3], segments))
	assert labels["index"].tolist() == [0, 0, 0, 1, 1, 3, 3]
	assert labels["kind"] == ["grasp"] * 3 + ["lift"] * 2 + ["return_ready"] * 2
	assert labels["instruction"][3] == "do lift"
	# done on the first observation of the next segment; the last segment's done observation has no row
	assert labels["done"].tolist() == [False, False, False, True, False, True, False]
	assert labels["chunk_valid_len"].tolist() == [3, 2, 1, 2, 1, 2, 1]
	assert [s["reason"] for s in labels["segments"]] == ["done", "brick_lost", "done", "done"]


@pytest.mark.parametrize(
	("ids", "segments", "match"),
	[
		([0, -1, -1], [("grasp", 0, 1, "done")], "no segment"),
		([0, 0, 1], [("grasp", 0, 2, "done")], "cover"),  # rows of an unannounced segment
		([0, 1, 0], [("grasp", 0, 1, "done"), ("lift", 1, 3, "done")], "another segment"),
		([0, 0, 0], [("grasp", 0, 2, "done"), ("lift", 1, 3, "done")], "continue"),
	],
)
def test_segment_labels_reject_missing_or_inconsistent_labels(tmp_path, ids, segments, match):
	from digital_twin_armigo.agent.twin_record import SegmentLabelError, segment_labels

	with pytest.raises(SegmentLabelError, match=match):
		segment_labels(_episode(tmp_path, ids, segments))


class ConstantScorer:
	"""Every readable view scores above threshold: defective on the first wrist view (short episode)."""

	def threshold(self, camera):
		return 0.0

	def score(self, camera, image):
		return 1.0

	def close(self):
		pass


def test_production_entry_labels_like_the_replay_gate_and_exports_segment_columns(tmp_path):
	"""``TwinService.run`` (the entry of ``twin_mcp run`` / ``serve`` / resume) records the same segments as the
	replay gate's expert, every frame is labelled, and ``student_frames`` exports the S7 columns."""
	pytest.importorskip("mujoco")
	pytest.importorskip("langgraph")
	pytest.importorskip("av")
	pytest.importorskip("cv2")
	from digital_twin_armigo.agent.student_rollout import _segments_of, run_episode
	from digital_twin_armigo.agent.twin_graph import Policy
	from digital_twin_armigo.agent.twin_mcp import TwinService
	from digital_twin_armigo.agent.twin_record import segment_labels, student_frames

	case = (600001, True, 0)
	detector = tmp_path / "detector.json"
	detector.write_text(json.dumps({"crop_size": 256}), encoding="utf-8")
	service = TwinService(tmp_path / "out", detector, scorer=ConstantScorer(), sensing="realistic")
	episode = tmp_path / "out" / "case-0000"
	try:
		result = service.run(case, Policy(), episode, video=True)
	finally:
		service.close()
	assert result["verdict"] == "SUCCESS" and result["student_eligible"]
	events = [json.loads(line) for line in (episode / "events.jsonl").read_text(encoding="utf-8").splitlines()]
	gate = run_episode(case, ConstantScorer(), tmp_path / "gate")
	assert _segments_of(events) == _segments_of(gate["events"])  # one labelling implementation

	labels = segment_labels(episode)
	with np.load(episode / "branch-0" / "actions.npz") as rows:
		assert len(rows["segment"]) == result["sim_frames"] and (rows["segment"] >= 0).all()
	# Every capture is taken at a segment boundary: at the end of a scan_view / present_bottom segment.
	ends = {s["end"]: s["kind"] for s in labels["segments"]}
	views = [e for e in events if e["kind"] == "view"]
	assert views and all(ends.get(v["frame"]) in ("scan_view", "present_bottom") for v in views)

	frames = list(student_frames(episode))
	assert len(frames) == len(labels["index"])
	n = len(frames)
	assert all(s["end"] > s["start"] and s["reason"] == "done" for s in labels["segments"])
	for i, frame in enumerate(frames):
		sid, valid = int(frame["segment.index"][0]), int(frame["segment.chunk_valid_len"][0])
		seg = labels["segments"][sid]
		assert frame["task"] == seg["instruction"] and frame["segment.kind"] == seg["kind"]
		assert 1 <= valid and i + valid == seg["end"] <= n  # a chunk never reaches past its segment
		assert bool(frame["segment.done"][0]) == (i > 0 and i == seg["start"])  # first observation of a later segment
