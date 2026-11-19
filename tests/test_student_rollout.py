"""Student segment interface (agent/student_rollout.py): segmentation units and one MuJoCo replay end to end."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from digital_twin_armigo.agent.student_rollout import (
	SEGMENT_KINDS,
	ExpertTracer,
	PolicyStep,
	RecordedSegment,
	ReplayMismatch,
	ReplayPolicy,
	Segment,
	SegmentMappingError,
	StudentObservation,
	_scan_stops,
	_SegmentMixin,
	compare_runs,
)


def test_segment_instruction_and_identity():
	seg = Segment("scan_view", {"mode": "orbit", "view": 2, "views": 5, "faces": [1, 2], "target_q": [0.0] * 6})
	assert seg.instruction == "Move the wrist camera to inspection view 2 of 5 around the brick and hold still."
	assert not seg.held and seg.phase == "approach"
	assert Segment("carry_to_bowl", {"bowl": "pink"}).instruction.startswith("Carry the held brick over the pink bowl")
	assert Segment("carry_to_bowl", {"bowl": "pink"}).held
	assert seg.key() == Segment(seg.kind, dict(reversed(seg.params.items()))).key()
	with pytest.raises(ValueError):
		Segment("teleport")
	with pytest.raises(ValueError):
		Segment("lift", {"bad": float("nan")})
	for kind, (text, held, phase, budget) in SEGMENT_KINDS.items():
		assert text and budget > 0 and isinstance(held, bool), kind


def _splitter(name):
	"""ExpertTracer._split on a stub twin: returns the (kind, start, end) segments a frame stream produces."""
	ex = SimpleNamespace(_frame_idx=0, _carry_frame=-1)
	log = []
	stub = SimpleNamespace(expert=ex)
	stub._segment_begin = lambda seg: log.append([seg.kind, ex._frame_idx, None])
	stub._segment_end = lambda reason: log[-1].__setitem__(2, ex._frame_idx)

	def run(frames):
		def gen():
			for phase, carried in frames:
				if carried:
					ex._carry_frame = ex._frame_idx
				yield np.zeros(6), phase

		segs = {
			kind: Segment(kind, {"bowl": "green"} if kind == "carry_to_bowl" else {"xy": [0, 0]})
			for kind in (
				"grasp",
				"lift",
				"return_to_lift",
				"lower_on_table",
				"release_on_table",
				"return_ready",
				"carry_to_bowl",
				"release_in_bowl",
			)
		}
		for _ in ExpertTracer._split(stub, gen(), name, segs):
			ex._frame_idx += 1  # what ClosedLoopExpert.act does after each frame
		if log and log[-1][2] is None:
			log[-1][2] = ex._frame_idx
		return [tuple(x) for x in log]

	return run


def test_composite_generators_split_at_their_decision_points():
	grasp = [("approach", False)] * 3 + [("grasp", False)] * 4 + [("lift", True)] * 2
	assert _splitter("grasp_and_lift")(grasp) == [("grasp", 0, 7), ("lift", 7, 9)]
	# an aborted grasp has no lift frames: one grasp segment
	assert _splitter("grasp_and_lift")(grasp[:7]) == [("grasp", 0, 7)]
	park = [("show_move", True)] * 2 + [("place", True)] * 3 + [("place", False)] * 2 + [("approach", False)] * 2
	assert _splitter("park")(park) == [
		("return_to_lift", 0, 2),
		("lower_on_table", 2, 5),
		("release_on_table", 5, 7),
		("return_ready", 7, 9),
	]
	# release from the lift pose skips the optional return to it
	release = [("place", True)] * 3 + [("place", False)] * 2 + [("return", False)] * 2
	assert _splitter("release")(release) == [("carry_to_bowl", 0, 3), ("release_in_bowl", 3, 7)]
	with pytest.raises(SegmentMappingError):  # carried frames after the release fit no later part
		_splitter("release")(release + [("place", True)])


def test_motion_mapping_by_generator():
	cand = SimpleNamespace(q=np.arange(6.0), faces_px=[900, 0, 1200, 0, 0, 0])
	stub = SimpleNamespace(
		_scan_steps=[(SimpleNamespace(q=np.ones(6), faces_px=[0] * 6), []), (cand, [])],
		_scan_mode="orbit",
		node="scan_wrist",
	)

	def look(path):
		yield path

	def ready():
		yield None

	def mystery():
		yield None

	form, seg = _SegmentMixin._motion(stub, look([np.zeros(6), cand.q]))
	assert form == "simple" and seg.kind == "scan_view"
	assert seg.params["view"] == 2 and seg.params["views"] == 2 and seg.params["faces"] == [0, 2]
	assert seg.params["via_poses"] == 1
	assert _SegmentMixin._motion(stub, ready())[1].kind == "prepare"
	with pytest.raises(SegmentMappingError):
		_SegmentMixin._motion(stub, mystery())


def _obs(seg, i):
	return StudentObservation(frame=i, segment=seg, segment_id=0, segment_frame=i, q=np.zeros(6), _images=dict)


def test_replay_policy_plays_actions_then_segment_done():
	seg = Segment("lift")
	rec = RecordedSegment(seg, np.ones((2, 6)), np.zeros((2, 6)), ["lift", "lift"], "done")
	policy = ReplayPolicy([rec])
	with pytest.raises(ReplayMismatch):
		ReplayPolicy([rec]).start(Segment("grasp"), 0)
	policy.start(seg, 0)
	steps = [policy.act(_obs(seg, i)) for i in range(3)]
	assert [s.done for s in steps] == [False, False, True]
	assert steps[0].phase == "lift" and np.array_equal(steps[1].target, np.ones(6))
	assert policy.state_mismatches == 0
	assert isinstance(steps[2], PolicyStep)


def test_compare_runs_and_scan_stops():
	view = {"id": "0:10:wrist_cam", "camera": "wrist_cam", "frame": 10, "score": -0.5, "threshold": 0.0, "faces": [1]}
	events = [
		{"kind": "node", "node": "scan_wrist", "frame": 0},
		{"kind": "view", "frame": 10, "observation": view, "wall_time_s": 1.0, "detector": {"elapsed_ms": 3}},
		{
			"kind": "view",
			"frame": 20,
			"observation": {**view, "id": "0:20:wrist_cam", "frame": 20, "score": 0.5},
			"detector": {"head_scores": {"cls": 0.5}},
		},
	]
	assert _scan_stops(events) == [(2, True)]
	run = {
		"events": events,
		"result": {"destination": "pink", "smoothness": {"clipped_frames": 1}},
		"digest": "x",
		"crops": {"0-20-wrist_cam-crop.png": "a"},
	}
	other = json.loads(json.dumps(run))
	other["events"][1]["wall_time_s"] = 2.0  # wall clock is not compared
	other["result"]["smoothness"]["clipped_frames"] = 0
	assert all(compare_runs(run, other)[0].values())
	other["events"][2]["observation"]["score"] = 0.4999
	checks, _ = compare_runs(run, other)
	assert not checks["scores"] and not checks["events"] and checks["captures"] and checks["decisions"]
	other["events"][2]["observation"]["score"] = -0.1
	assert not compare_runs(run, other)[0]["decisions"]
	# a crop that differs (render noise): its score is reported, not compared; without images the check fails
	other["events"][2]["observation"]["score"] = 0.4999
	other["crops"]["0-20-wrist_cam-crop.png"] = "b"
	checks, info = compare_runs(run, other)
	assert checks["scores"] and checks["events"] and not checks["images"]
	assert info["render_noise_views"] == ["0:20:wrist_cam"] and info["render_noise_max_score_delta"] > 0


class ConstantScorer:
	"""Every readable view scores above threshold: defective on the first wrist view (short episode)."""

	def threshold(self, camera):
		return 0.0

	def score(self, camera, image):
		return 1.0


def test_mujoco_expert_replays_identically_through_the_student_interface(tmp_path):
	pytest.importorskip("mujoco")
	pytest.importorskip("langgraph")
	pytest.importorskip("av")
	pytest.importorskip("cv2")
	from digital_twin_armigo.agent.student_rollout import run_episode

	case = (600001, True, 0)
	expert = run_episode(case, ConstantScorer(), tmp_path / "expert")
	assert expert["result"]["verdict"] == "SUCCESS" and expert["result"]["label"] == "defective"
	kinds = [e["segment_kind"] for e in expert["events"] if e["kind"] == "segment"]
	assert kinds[:3] == ["prepare", "scan_view", "return_ready"]  # early stop after the first (defective) view
	assert kinds[-3:] == ["lift", "carry_to_bowl", "release_in_bowl"]
	with np.load(tmp_path / "expert" / "branch-0" / "actions.npz") as rows:
		assert (rows["segment"] >= 0).all() and len(rows["segment"]) == expert["frames"]
		assert np.array_equal(rows["action_f64"].astype(np.float32), rows["action"])

	policy = ReplayPolicy.from_trace(tmp_path / "expert")
	student = run_episode(case, ConstantScorer(), tmp_path / "student", policy=policy)
	checks, _ = compare_runs(expert, student, tmp_path / "expert", tmp_path / "student")
	assert all(checks.values()), checks
	assert policy.state_mismatches == 0 and policy.index == len(policy.segments) - 1

	# The gate is sensitive: one target changed by a micro-radian changes the physics.
	tampered = ReplayPolicy.from_trace(tmp_path / "expert")
	tampered.segments[1].actions[5, 0] += 1e-6
	changed = run_episode(case, ConstantScorer(), tmp_path / "tampered", policy=tampered)
	assert not compare_runs(expert, changed)[0]["telemetry"] and tampered.state_mismatches > 0
