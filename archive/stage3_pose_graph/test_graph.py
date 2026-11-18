"""LangGraph inspection loop: budgets, repair, stop, checkpoint resume after a crash, and the next_pose adapter."""

from __future__ import annotations

import json
import re

import pytest

pytest.importorskip("langgraph")

from graph import GraphNextPose, RuleProposer, build_inspection_graph, initial_state
from langgraph.checkpoint.memory import InMemorySaver

from digital_twin_armigo.agent import CoveragePolicy, CoverageTable, FakeLLM, LLMProposer, PoseOption
from digital_twin_armigo.agent.policies import pose_id

A = PoseOption(pose_id(0.8, 90), 0.8, 90.0, (0.9, 0.0, 0.0, 0.8, 1.0, 0.0))
B = PoseOption(pose_id(0.0, 150), 0.0, 150.0, (0.1, 0.1, 0.0, 0.0, 0.0, 0.8))
C = PoseOption(pose_id(0.4, -150), 0.4, -150.0, (0.0, 0.9, 0.3, 0.0, 0.0, 0.0))
TABLE = CoverageTable((A, B, C), n_layouts=2, source="test")
BASE = [
	{"flex": 0.65, "roll_deg": r, "status": "UNKNOWN", "visible_faces": [0], "observed_faces": [0, 2, 4], "reason": "x"}
	for r in (-35.0, 0.0, 35.0)
]
CFG = {"recursion_limit": 200}


class FakeExecutor:
	"""Adds ``faces[pose_id]`` to the observed set; KNOWN once all six faces are observed; can crash once."""

	def __init__(self, faces, crash_on_call=None):
		self.faces, self.crash_on_call = faces, crash_on_call
		self.calls: list[str] = []
		self.observed = {0, 2, 4}

	def execute(self, pose, pid):
		self.calls.append(pid)
		if self.crash_on_call == len(self.calls):
			self.crash_on_call = None
			raise RuntimeError("simulated worker crash")
		self.observed |= set(self.faces.get(pid, ()))
		known = len(self.observed) == 6
		return {
			"flex": pose[0],
			"roll_deg": pose[1],
			"status": "KNOWN" if known else "UNKNOWN",
			"visible_faces": sorted(self.faces.get(pid, ())),
			"observed_faces": sorted(self.observed),
			"reason": "done" if known else "uncovered",
			"job_id": f"job-{len(self.calls)}",
		}


def _cfg(tid):
	return {**CFG, "configurable": {"thread_id": tid}}


def _answers(*ids):
	return [json.dumps({"pose_id": i, "rationale": "r"}) for i in ids]


def test_graph_reaches_known_with_rule_policy():
	ex = FakeExecutor({B.pose_id: [5], C.pose_id: [1, 3]})
	g = build_inspection_graph(RuleProposer(CoveragePolicy(TABLE), TABLE), ex, TABLE, max_proposals=4)
	out = g.invoke(initial_state(BASE), _cfg("t1"))
	assert out["done_reason"] == "known"
	assert ex.calls == [B.pose_id, C.pose_id]
	assert out["job_ids"] == ["job-1", "job-2"]


def test_graph_enforces_proposal_budget():
	ex = FakeExecutor({})  # nothing new is ever seen
	llm = FakeLLM(_answers(A.pose_id, B.pose_id, C.pose_id))
	g = build_inspection_graph(LLMProposer(llm), ex, TABLE, max_proposals=2)
	out = g.invoke(initial_state(BASE), _cfg("t2"))
	assert out["done_reason"] == "budget"
	assert ex.calls == [A.pose_id, B.pose_id] and out["n_proposals"] == 2


def test_graph_repairs_once_then_gives_up():
	ex = FakeExecutor({})
	llm = FakeLLM(["garbage", "still garbage"])
	g = build_inspection_graph(LLMProposer(llm), ex, TABLE, max_proposals=4, max_repairs=1)
	out = g.invoke(initial_state(BASE), _cfg("t3"))
	assert out["done_reason"] == "invalid_output"
	assert out["n_invalid"] == 2 and len(llm.calls) == 2 and ex.calls == []


def test_graph_repair_succeeds_and_tried_pose_rejected():
	# second proposal repeats A (already tried) -> invalid; repair answers stop
	llm = FakeLLM(["oops", *_answers(A.pose_id), *_answers(A.pose_id), '{"stop": true}'])
	ex = FakeExecutor({A.pose_id: [1]})
	g = build_inspection_graph(LLMProposer(llm), ex, TABLE, max_proposals=4)
	out = g.invoke(initial_state(BASE), _cfg("t4"))
	assert ex.calls == [A.pose_id]
	assert out["done_reason"] == "policy_stop"
	assert out["n_invalid"] == 2


def test_graph_finishes_immediately_when_already_known():
	ex = FakeExecutor({})
	hist = [*BASE[:2], {**BASE[2], "status": "KNOWN"}]
	out = build_inspection_graph(LLMProposer(FakeLLM(["x"])), ex, TABLE).invoke(initial_state(hist), _cfg("t5"))
	assert out["done_reason"] == "known" and ex.calls == []


def test_checkpoint_resume_after_crash():
	saver = InMemorySaver()
	ex = FakeExecutor({A.pose_id: [3], B.pose_id: [5], C.pose_id: [1]}, crash_on_call=2)
	llm = FakeLLM(_answers(A.pose_id, B.pose_id, C.pose_id))
	g = build_inspection_graph(LLMProposer(llm), ex, TABLE, max_proposals=4, checkpointer=saver)
	cfg = _cfg("crash")
	with pytest.raises(RuntimeError, match="simulated worker crash"):
		g.invoke(initial_state(BASE), cfg)
	snap = g.get_state(cfg)
	assert snap.next == ("execute",)
	assert snap.values["job_ids"] == ["job-1"]  # only the finished job is referenced
	assert len(llm.calls) == 2
	# a fresh graph object on the same checkpointer (a restarted process) resumes from the checkpoint
	g2 = build_inspection_graph(LLMProposer(llm), ex, TABLE, max_proposals=4, checkpointer=saver)
	out = g2.invoke(None, cfg)
	assert out["done_reason"] == "known"
	assert ex.calls == [A.pose_id, B.pose_id, B.pose_id, C.pose_id]  # B retried once, A not re-executed
	assert len(llm.calls) == 3  # the B decision was not re-asked
	assert out["job_ids"] == ["job-1", "job-3", "job-4"]


def test_next_pose_adapter_drives_graph_like_the_expert():
	"""Simulate the expert: ask next_pose while UNKNOWN (max 4), execute the pose itself, append the record."""
	ex = FakeExecutor({B.pose_id: [5], C.pose_id: [1], A.pose_id: [3]})
	llm = FakeLLM(_answers(B.pose_id, C.pose_id, A.pose_id))
	adapter = GraphNextPose(LLMProposer(llm), TABLE, max_proposals=4)
	history = [dict(h, frame=i, label=None) for i, h in enumerate(BASE)]
	for _ in range(4):
		pose = adapter(history)
		if pose is None:
			break
		pid = TABLE.find(pose).pose_id
		rec = ex.execute(pose, pid)
		rec.pop("job_id")
		history.append({**rec, "frame": len(history), "label": "x" if rec["status"] == "KNOWN" else None})
		if rec["status"] == "KNOWN":
			break
	adapter.close(history)
	st = adapter.state()
	assert ex.calls == [B.pose_id, C.pose_id, A.pose_id]
	assert st["done_reason"] == "known"
	assert all(re.fullmatch(r"frame-\d+", j) for j in st["job_ids"])
	assert all("label" not in h and "reason" not in h for h in st["history"])
