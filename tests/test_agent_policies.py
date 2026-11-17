"""Show-pose policies: coverage table, greedy coverage, fixed sequence, LLM validation/repair, prompt leak checks."""

from __future__ import annotations

import json
import re

import pytest

pytest.importorskip("langgraph")

from digital_twin_armigo.agent import (
	CoveragePolicy,
	CoverageTable,
	FakeLLM,
	FixedSequencePolicy,
	LLMShowPosePolicy,
	PoseOption,
	ShowPosePolicy,
	build_context,
	parse_choice,
	render_messages,
	server_reachable,
)
from digital_twin_armigo.agent.llm import LLMError
from digital_twin_armigo.agent.policies import assert_prompt_leak_free, pose_id
from digital_twin_armigo.inspection import DefectLeakageError, assert_no_defect_leakage

LEAK_RE = re.compile(r"defect|pink|green|normal|orange|bowl|label", re.IGNORECASE)

# faces: +x -x +y -y top bottom
A = PoseOption(pose_id(0.8, 90), 0.8, 90.0, (0.9, 0.0, 0.0, 0.8, 1.0, 0.0))
B = PoseOption(pose_id(0.0, 150), 0.0, 150.0, (0.1, 0.1, 0.0, 0.0, 0.0, 0.8))
C = PoseOption(pose_id(0.4, -150), 0.4, -150.0, (0.0, 0.9, 0.3, 0.0, 0.0, 0.0))
TABLE = CoverageTable((A, B, C), n_layouts=2, source="test")


def _rec(flex, roll, observed, status="UNKNOWN", label=None, reason=None, visible=()):
	return {
		"frame": 100,
		"flex": flex,
		"roll_deg": roll,
		"status": status,
		"label": label,
		"visible_faces": list(visible),
		"observed_faces": list(observed),
		"reason": reason or f"uncovered_faces:{[f for f in range(6) if f not in observed]}",
	}


BASE = [_rec(0.65, -35, [0, 4]), _rec(0.65, 0, [0, 2, 4]), _rec(0.65, 35, [0, 2, 4])]


def test_table_from_pose_grid_filters_unsafe_and_roundtrips(tmp_path):
	def row(k, flex, roll, clear, top, wrist):
		r = {"flex": flex, "roll_deg": roll, "clearance_m": clear, "layout_index": k, "defect_face": 1}
		if clear >= 0.02:
			r["top"] = {"faces_px": top, "brick_px": 1}
			r["wrist_cam"] = {"faces_px": wrist, "brick_px": 1}
		return r

	z = [0] * 6
	rows = [
		row(0, 0.0, 90, 0.05, [100, 0, 0, 0, 0, 70], z),
		row(1, 0.0, 90, 0.05, [100, 0, 0, 0, 0, 10], [0, 0, 0, 0, 0, 65]),
		row(0, 0.4, 0, 0.05, z, [0, 61, 0, 0, 0, 0]),
		row(1, 0.4, 0, 0.01, z, z),  # unsafe in one layout -> pose dropped
		row(0, 0.8, 30, 0.05, z, [0, 0, 59, 0, 0, 0]),  # missing in layout 1 -> dropped
	]
	p = tmp_path / "pose_grid.jsonl"
	p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
	t = CoverageTable.from_pose_grid(p)
	assert [o.pose_id for o in t.options] == [pose_id(0.0, 90)]
	assert t.options[0].face_prob == (1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
	assert t.n_layouts == 2
	t.save(tmp_path / "t.json")
	assert CoverageTable.load(tmp_path / "t.json") == t


def test_coverage_policy_greedy_choice_and_stop():
	pol = CoveragePolicy(TABLE)
	assert isinstance(pol, ShowPosePolicy)
	# observed +x,+y,top -> A adds -y (0.8); B adds -x,bottom (0.9); C adds -x (0.9)... tie broken by table order
	assert pol(BASE) == (B.flex, B.roll_deg)
	# after B showed bottom, -x/-y still missing: A gives 0.8, C gives 0.9
	h = [*BASE, _rec(0.0, 150, [0, 2, 4, 5])]
	assert pol(h) == (C.flex, C.roll_deg)
	# everything observed -> no pose adds coverage
	assert pol([*BASE, _rec(0.0, 150, range(6))]) is None
	# all tried -> None
	assert pol([*BASE, _rec(0.0, 150, [0]), _rec(0.8, 90, [0]), _rec(0.4, -150, [0])]) is None


def test_fixed_sequence_policy_skips_shown_poses():
	pol = FixedSequencePolicy([(0.0, 150.0), (0.8, 90.0)])
	assert pol(BASE) == (0.0, 150.0)
	assert pol([*BASE, _rec(0.0, 150, [0])]) == (0.8, 90.0)
	assert pol([*BASE, _rec(0.0, 150, [0]), _rec(0.8, 90, [0])]) is None


def test_parse_choice_validation():
	menu, tried = {A.pose_id, C.pose_id}, {B.pose_id}
	assert parse_choice(f'{{"pose_id": "{A.pose_id}", "rationale": "x"}}', menu, tried)[0]["pose_id"] == A.pose_id
	assert parse_choice(f'```json\n{{"pose_id": "{C.pose_id}"}}\n```', menu, tried)[0]["pose_id"] == C.pose_id
	assert parse_choice('{"stop": true}', menu, tried)[0] == {"stop": True, "rationale": ""}
	assert "already tried" in parse_choice(f'{{"pose_id": "{B.pose_id}"}}', menu, tried)[1]
	assert "not in the menu" in parse_choice('{"pose_id": "f+9.00_r+000"}', menu, tried)[1]
	assert "JSON" in parse_choice("I pick the first one", menu, tried)[1]
	assert parse_choice("[1, 2]", menu, tried)[1]


def test_llm_policy_valid_answer():
	llm = FakeLLM([json.dumps({"pose_id": B.pose_id, "rationale": "bottom"})])
	pol = LLMShowPosePolicy(llm, TABLE)
	assert pol(BASE) == (B.flex, B.roll_deg)
	assert len(pol.log) == 1 and pol.log[0]["raw"] and not pol.failures


def test_llm_policy_repairs_once():
	llm = FakeLLM(["not json", json.dumps({"pose_id": C.pose_id})])
	pol = LLMShowPosePolicy(llm, TABLE)
	assert pol(BASE) == (C.flex, C.roll_deg)
	assert len(llm.calls) == 2
	repair_msgs = llm.calls[1]
	assert repair_msgs[-2] == {"role": "assistant", "content": "not json"}
	assert "Invalid answer" in repair_msgs[-1]["content"]
	assert pol.decisions[-1]["repairs"] == 1


def test_llm_policy_second_failure_returns_none_and_records():
	tried = [*BASE, _rec(0.0, 150, [0, 5])]
	llm = FakeLLM([json.dumps({"pose_id": B.pose_id}), json.dumps({"pose_id": "nope"})])
	pol = LLMShowPosePolicy(llm, TABLE)
	assert pol(tried) is None
	assert len(llm.calls) == 2
	assert len(pol.failures) == 1 and len(pol.failures[0]["errors"]) == 2


def test_llm_policy_stop_and_transport_error():
	assert LLMShowPosePolicy(FakeLLM(['{"stop": true}']), TABLE)(BASE) is None

	class Broken:
		def chat(self, messages):
			raise LLMError("down")

	pol = LLMShowPosePolicy(Broken(), TABLE)
	assert pol(BASE) is None and pol.failures and pol.log[0]["error"]


def test_prompt_is_leak_free_even_with_label_in_history():
	# label and class-specific reasons (UNKNOWN reasons may be worded per class) must never reach the model
	hist = [
		*BASE,
		_rec(0.4, 30, [0, 2, 4], reason="defect_face_not_visible"),
		_rec(0.0, 150, [0, 5], status="KNOWN", label="defective", reason="defect_pixels_visible"),
	]
	ctx = build_context(hist, TABLE, remaining=2)
	assert_no_defect_leakage(ctx)
	assert ctx["history"][3]["faces_not_observed_yet"] == ["-x", "-y", "bottom"]
	text = json.dumps(ctx) + "".join(m["content"] for m in render_messages(ctx))
	assert not LEAK_RE.search(text), LEAK_RE.search(text)
	llm = FakeLLM([json.dumps({"pose_id": C.pose_id})])
	LLMShowPosePolicy(llm, TABLE)(hist)
	for m in llm.calls[0]:
		assert not LEAK_RE.search(m["content"])


def test_prompt_leak_guard_raises():
	with pytest.raises(DefectLeakageError):
		assert_prompt_leak_free([{"role": "user", "content": "put it in the pink bowl"}])
	with pytest.raises(DefectLeakageError):
		assert_prompt_leak_free([{"role": "user", "content": "brick_duplo_2x2_defective.usda"}])


@pytest.mark.skipif(not server_reachable(), reason="local vLLM server not reachable")
def test_live_llm_returns_valid_choice():
	from digital_twin_armigo.agent import OpenAICompatLLM

	pol = LLMShowPosePolicy(OpenAICompatLLM(timeout_s=60, retries=1), TABLE)
	pose = pol(BASE)
	assert pose is None or pose in {(o.flex, o.roll_deg) for o in TABLE.options}
	assert pol.log and pol.log[0]["latency_s"] is not None


def test_conditional_face_probs_follow_similar_layouts():
	base = pose_id(0.65, 0.0)
	ev = (
		{base: (0,), A.pose_id: (5,), B.pose_id: ()},  # layouts that look like "+x seen at the base pose"
		{base: (1,), A.pose_id: (), B.pose_id: (5,)},
	)
	t = CoverageTable((A, B), n_layouts=2, source="t", evidence=ev)
	hist = [{"flex": 0.65, "roll_deg": 0.0, "visible_faces": [0], "observed_faces": [0]}]
	p = t.face_probs(hist, beta=3.0)
	assert p[A.pose_id][5] > 0.9 and p[B.pose_id][5] < 0.1
	assert t.face_probs(hist, conditional=False)[A.pose_id] == A.face_prob
	assert CoveragePolicy(t, conditional=True)(hist) == (A.flex, A.roll_deg)
	assert CoverageTable.from_dict(json.loads(json.dumps(t.to_dict()))) == t
