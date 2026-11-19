"""Cross-process resume from a persisted checkpoint (VLA plan S2 / S5): a held or on-table checkpoint written to disk
and restored into a newly built twin (new cell, expert, perception and scorer objects) continues exactly as the
uninterrupted episode: every later target, joint reading, telemetry row, event (perception, views, placement check)
and the final ``evaluate`` dict. Captured images are compared within the renderer's documented repeatability."""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("cv2")
pytest.importorskip("av")
pytest.importorskip("langgraph")

from digital_twin_armigo.agent.student_rollout import (
	ExpertTracer,
	ReplayPolicy,
	StudentTwin,
	compare_runs,
	crop_digests,
	telemetry_digest,
)
from digital_twin_armigo.agent.twin_graph import Policy, build_graph
from digital_twin_armigo.agent.twin_mujoco import ROOT, sample_case
from digital_twin_armigo.agent.twin_record import EpisodeTrace
from digital_twin_armigo.agent.twin_resume import (
	ResumePlan,
	live_checkpoint_ids,
	load_checkpoint,
	load_history,
	restore,
)

CASE = (600001, False, 5)  # bench16 case 0, normal: wrist scan, bottom view (brick held), place in the green bowl
DETECTOR = ROOT / "results/agent-bench/sup-v7bot-local.json"


class ClearScorer:
	"""Every readable view clears its faces (normal brick)."""

	def threshold(self, camera):
		return 0.0

	def score(self, camera, image):
		return -1.0

	def close(self):
		pass


class _SaveWhenHeld:
	"""Also checkpoint at the end of a bottom view while the brick is held (no automatic save point exists there)."""

	def inspect(self, node, previous):
		views, error = super().inspect(node, previous)
		if node == "present_bottom" and error is None and self.held is not None:
			self._save([*previous, *views])
		return views, error


class HeldExpert(_SaveWhenHeld, ExpertTracer):
	pass


class HeldStudent(_SaveWhenHeld, StudentTwin):
	pass


def _finish(twin, trace, result, out):
	trace.finish(result, {"engine": "mujoco", "sensing": "realistic", "case": list(CASE)})
	tel = twin.ep.telemetry()
	return {
		"result": json.loads(json.dumps(result)),
		"events": trace.events,
		"digest": telemetry_digest(twin),
		"crops": crop_digests(out),
		"telemetry": tel,
		"targets": np.asarray(twin.targets),
		"phases": np.asarray(twin.phases),
		"dir": out,
	}


def run_full(cls, out, scorer, *args):
	trace = EpisodeTrace(out, video=False)
	twin = cls(sample_case(*CASE), scorer, trace, *args, sensing="realistic")
	try:
		result = build_graph(twin, Policy()).invoke({}, config={"recursion_limit": 64})["result"]
		return _finish(twin, trace, result, out)
	finally:
		twin.close()


def plan_at(parent, checkpoint_id):
	ids = live_checkpoint_ids(parent / "events.jsonl")
	checkpoints = [load_checkpoint(parent, name) for name in ids[: ids.index(checkpoint_id) + 1]]
	return ResumePlan(parent, checkpoints, [], None, *load_history(parent, checkpoints[-1]))


def run_resumed(cls, plan, out, scorer, *args):
	"""Everything new, as in a fresh process: trace, twin (cell, expert, perception), scorer; state from disk only."""
	from langgraph.checkpoint.memory import InMemorySaver

	trace = EpisodeTrace(out, video=False)
	twin = cls(sample_case(*CASE), scorer, trace, *args, sensing="realistic")
	try:
		restore(twin, plan)
		graph = build_graph(twin, Policy(), InMemorySaver())
		config = {"recursion_limit": 64, "configurable": {"thread_id": "resume"}}
		graph.update_state(config, plan.graph_state(), as_node="prepare")
		result = graph.invoke(None, config=config)["result"]
		return _finish(twin, trace, result, out)
	finally:
		twin.close()


def _after(events, index):
	return [{k: v for k, v in e.items() if k != "seq"} for e in events[index + 1 :]]


def assert_same_continuation(full, resumed, checkpoint_id):
	"""The resumed run from ``checkpoint_id`` equals the uninterrupted run from that checkpoint on."""
	start = next(
		i for i, e in enumerate(full["events"]) if e["kind"] == "checkpoint" and e["checkpoint"] == checkpoint_id
	)
	frame = full["events"][start]["frame"]
	resumed_at = next(i for i, e in enumerate(resumed["events"]) if e["kind"] == "resumed")
	assert resumed["events"][resumed_at]["frame"] == frame
	# Final evaluation, complete (health and smoothness included, clipped-target count too).
	assert resumed["result"] == full["result"]
	# Every later policy target, joint reading and telemetry row (the restored prefix is the parent's own).
	for key, rows in full["telemetry"].items():
		assert np.array_equal(resumed["telemetry"][key], rows), key
	assert np.array_equal(resumed["targets"], full["targets"]) and np.array_equal(resumed["phases"], full["phases"])
	with (
		np.load(full["dir"] / "branch-0" / "actions.npz") as a,
		np.load(resumed["dir"] / "branch-0" / "actions.npz") as b,
	):
		assert len(b["action_f64"]) == len(a["action_f64"]) - frame > 0
		for key in ("action_f64", "state_f64", "segment", "phase"):
			assert np.array_equal(b[key], a[key][frame:]), key
	# Every later event (perception, scan plans, views and detector decisions, segments, placement fill counts),
	# frame numbers included; captured images and the scores of their views within the renderer's repeatability.
	checks, info = compare_runs(
		{**full, "events": _after(full["events"], start)},
		{**resumed, "events": _after(resumed["events"], resumed_at)},
		full["dir"],
		resumed["dir"],
	)
	assert all(checks.values()), (checks, info)
	return info


@pytest.fixture(scope="module")
def expert_run(tmp_path_factory):
	out = tmp_path_factory.mktemp("expert") / "ep"
	run = run_full(HeldExpert, out, ClearScorer())
	assert run["result"]["verdict"] == "SUCCESS" and run["result"]["destination"] == "green"
	return run


def _held_checkpoint(parent):
	ids = live_checkpoint_ids(parent / "events.jsonl")
	held = [name for name in ids if load_checkpoint(parent, name).held is not None]
	assert len(held) == 1
	return held[0], ids


def test_expert_resumes_from_disk_exactly_held_and_on_table(expert_run, tmp_path):
	parent = expert_run["dir"]
	held, ids = _held_checkpoint(parent)
	placement = [e for e in expert_run["events"] if e["kind"] == "placement_check"]
	assert len(placement) == 1
	for name in (held, ids[0]):  # mid-carry (then place), and after prepare (scan, perception, grasp, bottom, place)
		ck = load_checkpoint(parent, name)
		agent = ck.agent
		assert agent["brick_est"] is not None or name == ids[0]
		assert (parent / "checkpoints" / agent["empty_bowl_depth"]).is_file()
		plan = plan_at(parent, name)
		assert plan.history is not None and len(plan.history["targets"]) == ck.frame
		resumed = run_resumed(HeldExpert, plan, tmp_path / name, ClearScorer())
		assert_same_continuation(expert_run, resumed, name)
		got = [e for e in resumed["events"] if e["kind"] == "placement_check"]
		assert [e["fill_pixels"] for e in got] == [e["fill_pixels"] for e in placement]
	# The held checkpoint carries the agent's belief and the grasp.
	ck = load_checkpoint(parent, held)
	assert ck.held["roll"] is not None and ck.agent["grip_from_brick"] is not None


def test_student_held_checkpoint_resumes_from_disk_exactly(expert_run, tmp_path):
	"""StudentTwin replaying the expert: save while the student holds the brick, resume in a new twin and policy."""
	trace_dir = expert_run["dir"]
	policy = ReplayPolicy.from_trace(trace_dir)
	full = run_full(HeldStudent, tmp_path / "student", ClearScorer(), policy)
	assert policy.state_mismatches == 0
	assert full["result"]["verdict"] == "SUCCESS" and full["result"]["destination"] == "green"
	held, _ = _held_checkpoint(full["dir"])
	ck = load_checkpoint(full["dir"], held)
	assert ck.held["roll"] == ck.held["q_lift"][4]  # the student's grasp roll: its lift pose's wrist roll
	done = len(ck.agent["segments"]["rows"])
	assert done > 0
	resumed_policy = ReplayPolicy.from_trace(trace_dir)
	resumed_policy.index = done - 1  # the policy resumes at the next segment, as a stateless VLA would
	resumed = run_resumed(HeldStudent, plan_at(full["dir"], held), tmp_path / "resumed", ClearScorer(), resumed_policy)
	assert_same_continuation(full, resumed, held)
	assert resumed_policy.state_mismatches == 0 and resumed_policy.index == len(resumed_policy.segments) - 1


def test_held_resume_from_disk_parks_on_the_belief_identically(tmp_path):
	"""A continuation that reads the brick belief, contact filter and depth RNG: from a held checkpoint, put the brick
	back where it was perceived (abstain / recovery path) and locate it again, in the saving twin and in a new one."""
	from digital_twin_armigo.agent.twin_mujoco import TwinMujoco

	def park_and_look(twin, events_from):
		start = twin.ep.n_frames
		assert twin._park_held() is None and twin._perceive() is None
		events = [{k: v for k, v in e.items() if k not in ("seq", "wall_time_s")} for e in twin.record.events]
		return np.asarray(twin.targets[start:]), twin.ep.telemetry()["qpos"][start:], events[events_from:]

	trace = EpisodeTrace(tmp_path / "parent", video=False)
	twin = TwinMujoco(sample_case(*CASE), ClearScorer(), trace, sensing="realistic")
	try:
		assert twin.prepare() is None
		views = []
		assert twin._bottom(views) is None and twin.held is not None
		twin._save(views)
		first = park_and_look(twin, len(trace.events))
	finally:
		twin.close()
		trace.close()
	ck = twin.checkpoints[-1]
	assert ck.agent["brick_est"] is not None
	plan = plan_at(trace.path, ck.id)
	resumed_trace = EpisodeTrace(tmp_path / "resumed", video=False)
	fresh = TwinMujoco(sample_case(*CASE), ClearScorer(), resumed_trace, sensing="realistic")
	try:
		restore(fresh, plan)
		# The full belief, size class included (realistic planners read the believed brick size).
		assert fresh._believed_brick_size() == tuple(ck.agent["brick_est"]["size"])
		assert fresh._perceived_size == tuple(ck.agent["perceived_size"])
		second = park_and_look(fresh, len(resumed_trace.events))
	finally:
		fresh.close()
		resumed_trace.close()
	assert len(first[0]) > 0 and any(e["kind"] == "brick_perceived" for e in first[2])
	for a, b in zip(first, second, strict=True):
		if isinstance(a, list):
			assert a == b
		else:
			assert np.array_equal(a, b)


def test_real_detector_resume_from_disk_exactly(tmp_path):
	if not DETECTOR.is_file():
		pytest.skip("detector spec not present")
	from digital_twin_armigo.inspection_scorer import default_sup_scorer

	scorer = default_sup_scorer(DETECTOR)
	try:
		full = run_full(HeldExpert, tmp_path / "full", scorer)
		assert full["result"]["verdict"] == "SUCCESS"
		held, ids = _held_checkpoint(full["dir"])
		for name in (held, ids[0]):
			# A new scorer object for the resumed run would load the same detector; the scorer holds no episode state.
			resumed = run_resumed(HeldExpert, plan_at(full["dir"], name), tmp_path / name, scorer)
			assert_same_continuation(full, resumed, name)
	finally:
		scorer.close()
