"""Student segment interface (agent/student_rollout.py): segmentation units and one MuJoCo replay end to end."""

from __future__ import annotations

import dataclasses
import json
import string
from types import SimpleNamespace

import numpy as np
import pytest

from digital_twin_armigo.agent.student_rollout import (
	POLICY_FIELDS,
	POLICY_PARAMS,
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
	policy_inputs,
)

DIAGNOSTICS = ("mode", "faces", "target_q", "via_poses", "bottom_px", "xy")


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


def test_policy_inputs_hide_planner_diagnostics():
	"""A policy gets the instruction and the fields it is formatted from, never the planner's outputs."""
	seg = Segment(
		"scan_view",
		{"mode": "orbit", "view": 2, "views": 5, "faces": [1, 2], "target_q": [0.1] * 6, "via_poses": 3},
	)
	task = policy_inputs(seg)
	assert tuple(task) == POLICY_FIELDS
	assert task == {"kind": "scan_view", "instruction": seg.instruction, "view": 2, "views": 5, "bowl": None}
	assert set(seg.diagnostics) == {"mode", "faces", "target_q", "via_poses"}
	for diag in (seg, Segment("present_bottom", {"target_q": [0.0] * 6, "bottom_px": 900})):
		assert not set(policy_inputs(diag)) & set(DIAGNOSTICS)
		assert "0.1" not in json.dumps(policy_inputs(diag))
	task["view"] = 99  # a detached copy
	assert seg.params["view"] == 2
	# Every instruction is formatted from policy-visible params only, so the text cannot leak a diagnostic.
	for kind, (text, *_) in SEGMENT_KINDS.items():
		fields = {f for _, f, _, _ in string.Formatter().parse(text) if f}
		assert fields <= set(POLICY_PARAMS), kind
	# The per-frame observation carries the task, not the Segment.
	names = {f.name for f in dataclasses.fields(StudentObservation)}
	assert "segment" not in names and "task" in names


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
	return StudentObservation(
		frame=i, task=policy_inputs(seg), segment_id=0, segment_frame=i, q=np.zeros(6), _images=dict
	)


def test_replay_policy_plays_actions_then_segment_done():
	seg = Segment("lift")
	rec = RecordedSegment(seg, np.ones((2, 6)), np.zeros((2, 6)), ["lift", "lift"], "done")
	policy = ReplayPolicy([rec])
	with pytest.raises(ReplayMismatch):
		ReplayPolicy([rec]).start(policy_inputs(Segment("grasp")), 0)
	policy.start(policy_inputs(seg), 0)
	steps = [policy.act(_obs(seg, i)) for i in range(3)]
	assert [s.done for s in steps] == [False, False, True]
	assert steps[0].phase == "lift" and np.array_equal(steps[1].target, np.ones(6))
	assert policy.state_mismatches == 0
	assert isinstance(steps[2], PolicyStep)
	# Diagnostics are matched through the separate channel only.
	view = Segment("present_bottom", {"target_q": [0.0] * 6, "bottom_px": 900})
	replay = ReplayPolicy([RecordedSegment(view, np.ones((1, 6)), np.zeros((1, 6)), ["show_hold"], "done")])
	other = Segment("present_bottom", {"target_q": [0.0] * 6, "bottom_px": 901})
	replay.start(policy_inputs(other), 0)  # the task alone cannot tell them apart
	replay.check_segment(view, 0)
	with pytest.raises(ReplayMismatch):
		replay.check_segment(other, 0)


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
	assert info["render_noise_crop_views"] == ["0:20:wrist_cam"] and info["render_noise_frame_views"] == []


def _keyframe(path, level=0):
	import cv2

	img = np.full((48, 64, 3), 100, np.uint8)
	img[0, 0, 0] += level
	path.parent.mkdir(parents=True, exist_ok=True)
	assert cv2.imwrite(str(path), img)


def test_full_frame_noise_does_not_exempt_an_identical_crop(tmp_path):
	"""Render noise in a view's full frame only: the crop the detector scored is bitwise equal, so its total and head
	scores must still match exactly (the full-frame tolerance still applies to the image check)."""
	pytest.importorskip("cv2")
	from digital_twin_armigo.agent.student_rollout import crop_digests

	view = {"id": "0:20:wrist_cam", "camera": "wrist_cam", "frame": 20, "score": 0.5, "threshold": 0.0, "faces": [1]}
	events = [
		{"kind": "node", "node": "scan_wrist", "frame": 0},
		{"kind": "view", "frame": 20, "observation": view, "detector": {"head_scores": {"cls": 0.5, "seg": 0.2}}},
	]
	dirs = {}
	for name, level in (("expert", 0), ("student", 1)):  # one full-frame pixel one level off
		dirs[name] = tmp_path / name
		_keyframe(dirs[name] / "keyframes" / "0-20-wrist_cam-frame.png", level)
		_keyframe(dirs[name] / "keyframes" / "0-20-wrist_cam-crop.png")
	run = {"events": events, "result": {"destination": "pink"}, "digest": "x", "crops": crop_digests(dirs["expert"])}
	other = json.loads(json.dumps(run))
	other["crops"] = crop_digests(dirs["student"])
	assert other["crops"]["0-20-wrist_cam-frame.png"] != run["crops"]["0-20-wrist_cam-frame.png"]
	assert other["crops"]["0-20-wrist_cam-crop.png"] == run["crops"]["0-20-wrist_cam-crop.png"]
	checks, info = compare_runs(run, other, dirs["expert"], dirs["student"])
	assert all(checks.values()), checks
	assert info["render_noise_frame_views"] == ["0:20:wrist_cam"] and info["render_noise_crop_views"] == []
	# The auditor's case: the total score moves by 0.5, the decision does not; the identical crop catches it.
	other["events"][1]["observation"]["score"] = 1.0
	checks, _ = compare_runs(run, other, dirs["expert"], dirs["student"])
	assert checks["images"] and checks["decisions"] and not checks["scores"] and not checks["events"]
	# A single detector head that moves is caught as well.
	other["events"][1]["observation"]["score"] = 0.5
	other["events"][1]["detector"]["head_scores"]["seg"] = 0.2001
	checks, _ = compare_runs(run, other, dirs["expert"], dirs["student"])
	assert checks["images"] and not checks["scores"] and not checks["events"]
	# A decision flip is never tolerated, whatever the images.
	other["events"][1]["observation"]["score"] = -0.1
	assert not compare_runs(run, other, dirs["expert"], dirs["student"])[0]["decisions"]


class ConstantScorer:
	"""Every readable view scores above threshold: defective on the first wrist view (short episode)."""

	def threshold(self, camera):
		return 0.0

	def score(self, camera, image):
		return 1.0


class SpyPolicy:
	"""Wraps a policy and records everything it is handed."""

	def __init__(self, inner):
		self.inner, self.tasks = inner, []

	def start(self, task, segment_id):
		assert isinstance(task, dict)
		self.tasks.append(task)
		self.inner.start(task, segment_id)

	def act(self, obs):
		assert not any(isinstance(getattr(obs, f.name), Segment) for f in dataclasses.fields(obs))
		assert obs.task == self.tasks[-1]
		return self.inner.act(obs)


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

	policy = SpyPolicy(ReplayPolicy.from_trace(tmp_path / "expert"))
	student = run_episode(
		case, ConstantScorer(), tmp_path / "student", policy=policy, segment_monitor=policy.inner.check_segment
	)
	checks, _ = compare_runs(expert, student, tmp_path / "expert", tmp_path / "student")
	assert all(checks.values()), checks
	replay = policy.inner
	assert replay.state_mismatches == 0 and replay.index == len(replay.segments) - 1
	# The online policy saw only the policy-visible fields, although the recorded segments carry target_q etc.
	assert any("target_q" in s.segment.params for s in replay.segments)
	assert policy.tasks and all(tuple(t) == POLICY_FIELDS for t in policy.tasks)
	assert not any(d in json.dumps(policy.tasks) for d in DIAGNOSTICS)

	# The gate is sensitive: one target changed by a micro-radian changes the physics.
	tampered = ReplayPolicy.from_trace(tmp_path / "expert")
	tampered.segments[1].actions[5, 0] += 1e-6
	changed = run_episode(case, ConstantScorer(), tmp_path / "tampered", policy=tampered)
	assert not compare_runs(expert, changed)[0]["telemetry"] and tampered.state_mismatches > 0
