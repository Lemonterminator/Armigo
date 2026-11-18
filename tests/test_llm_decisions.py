"""Inner-graph LLM decisions: fallback reasons, structured output, repair, token budget and episode statistics."""

import json
from typing import ClassVar

import httpx
import pytest

from digital_twin_armigo.agent.llm import FakeLLM, LLMError, OpenAICompatLLM
from digital_twin_armigo.agent.twin_graph import select_skill, skill_schema
from digital_twin_armigo.agent.twin_mcp import aggregate

CHOICES = ["scan_wrist", "present_bottom"]


def view(i, score=-1.0):
	return {"id": f"0:{i}:wrist_cam", "camera": "wrist_cam", "faces": [0, 1], "score": score, "threshold": 0.0}


@pytest.mark.parametrize(
	("response", "reason"),
	[
		(LLMError("down", "http_error"), "http_error"),
		(LLMError("slow", "timeout"), "timeout"),
		("not json at all", "invalid_json"),
		('{"other": 1}', "invalid_json"),
		('{"skill":"place_in_pink"}', "illegal_choice"),
		("", "empty_output"),
	],
)
def test_every_fallback_reason_is_reported(response, reason):
	llm = FakeLLM([response])
	name, source, info = select_skill(llm, CHOICES, {})
	assert (name, source, info["reason"]) == ("scan_wrist", "fallback", reason)
	assert info["attempts"] == (1 if isinstance(response, LLMError) else 2)  # transport errors are not repaired


def test_context_too_long_when_nothing_is_left_to_trim():
	llm = FakeLLM(['{"skill":"present_bottom"}'], max_model_len=200)
	name, source, info = select_skill(llm, CHOICES, {"notes": "x" * 5000})
	assert (name, source, info["reason"], info["attempts"]) == ("scan_wrist", "fallback", "context_too_long", 0)
	assert llm.calls == []


def test_valid_answer_is_adopted_with_info_and_a_schema_enum():
	llm = FakeLLM(['{"skill":"present_bottom"}'])
	name, source, info = select_skill(llm, CHOICES, {})
	assert (name, source, info["reason"], info["attempts"]) == ("present_bottom", "llm", None, 1)
	assert info["prompt_tokens"] > 0
	assert llm.schemas[0]["properties"]["skill"]["enum"] == CHOICES
	assert skill_schema(["a"])["required"] == ["skill"]


def test_one_repair_attempt_quotes_the_validation_error():
	llm = FakeLLM(['{"skill":"nonsense"}', '{"skill":"present_bottom"}'])
	name, source, info = select_skill(llm, CHOICES, {})
	assert (name, source, info["attempts"], info["repaired"]) == ("present_bottom", "llm", 2, True)
	repair = llm.calls[1]
	assert repair[-2] == {"role": "assistant", "content": '{"skill":"nonsense"}'}
	assert "'nonsense' is not one of the choices" in repair[-1]["content"]


def test_second_bad_answer_falls_back_with_its_own_reason():
	llm = FakeLLM(['{"skill":"nonsense"}', "garbage"])
	_, source, info = select_skill(llm, CHOICES, {})
	assert (source, info["reason"], info["attempts"]) == ("fallback", "invalid_json", 2)


def test_oldest_view_detail_is_trimmed_to_fit_the_window():
	views = [view(i) for i in range(40)]
	llm = FakeLLM(['{"skill":"present_bottom"}'], max_model_len=900)
	name, source, info = select_skill(llm, CHOICES, {"views": views})
	assert (name, source) == ("present_bottom", "llm")
	assert info["trimmed"] > 0
	sent = json.loads(llm.calls[0][1]["content"])
	assert sent["views_omitted"] == info["trimmed"]
	assert sent["views"][-1]["id"] == views[-1]["id"]  # newest evidence is kept
	assert sent["views"][0]["id"] != views[0]["id"]


def client(handler, **kwargs):
	llm = OpenAICompatLLM("http://llm.test/v1", retries=0, **kwargs)
	llm._client = httpx.Client(transport=httpx.MockTransport(handler))
	return llm


def completion(content, tokens=50):
	return {
		"choices": [{"message": {"content": content}, "finish_reason": "stop"}],
		"usage": {"prompt_tokens": tokens, "completion_tokens": 8},
	}


def test_request_body_carries_a_strict_json_schema_whose_skill_is_an_enum():
	bodies = []

	def handler(request):
		if request.url.path.endswith("/chat/completions"):
			bodies.append(json.loads(request.content))
		return httpx.Response(200, json=completion('{"skill":"present_bottom"}'))

	name, source, info = select_skill(client(handler, json_mode=True), CHOICES, {})
	rf = bodies[0]["response_format"]
	assert (name, source, info["format"], info["prompt_tokens"]) == ("present_bottom", "llm", "json_schema", 50)
	assert rf["type"] == "json_schema"
	assert rf["json_schema"]["strict"] is True
	assert rf["json_schema"]["schema"]["properties"]["skill"]["enum"] == CHOICES


def test_server_rejecting_the_schema_degrades_to_json_object_once():
	formats = []

	def handler(request):
		if request.url.path.endswith("/chat/completions"):
			rf = json.loads(request.content).get("response_format", {})
			formats.append(rf.get("type"))
			if rf.get("type") == "json_schema":
				return httpx.Response(400, json={"error": "unsupported"})
		return httpx.Response(200, json=completion('{"skill":"scan_wrist"}'))

	llm = client(handler)
	_, source, info = select_skill(llm, CHOICES, {})
	assert (source, info["format"], llm.schema_degraded) == ("llm", "json_object", True)
	assert formats == ["json_schema", "json_object"]
	select_skill(llm, CHOICES, {})
	assert formats[2:] == ["json_object"]  # no schema retried after degrading


def test_transport_failures_map_to_http_error_and_timeout():
	def boom(request):
		raise httpx.ConnectTimeout("slow", request=request)

	assert select_skill(client(boom), CHOICES, {})[2]["reason"] == "timeout"
	assert select_skill(client(lambda r: httpx.Response(503, text="busy")), CHOICES, {})[2]["reason"] == "http_error"
	calls = []

	def forbidden(request):
		calls.append(request)
		return httpx.Response(403, text="no")

	llm = OpenAICompatLLM("http://llm.test/v1", retries=3, backoff_s=0)
	llm._client = httpx.Client(transport=httpx.MockTransport(forbidden))
	assert select_skill(llm, CHOICES, {})[2]["reason"] == "http_error"
	assert sum(r.url.path.endswith("/chat/completions") for r in calls) == 1  # a 4xx is not retried


def test_tokenizer_is_optional_and_budget_uses_the_server_count_when_available():
	seen = []

	def handler(request):
		if request.url.path == "/tokenize":
			seen.append("tokenize")
			return httpx.Response(200, json={"count": 5000})
		if request.url.path.endswith("/models"):
			return httpx.Response(200, json={"data": [{"id": "m", "max_model_len": 4096}]})
		return httpx.Response(200, json=completion('{"skill":"scan_wrist"}'))

	_, source, info = select_skill(client(handler), CHOICES, {})
	assert (source, info["reason"], seen) == ("fallback", "context_too_long", ["tokenize"])

	def no_tokenizer(request):
		if request.url.path == "/tokenize":
			return httpx.Response(404)
		return handler(request)

	llm = client(no_tokenizer)
	assert llm.count_tokens([{"role": "user", "content": "hi"}]) is None
	assert llm.count_tokens([{"role": "user", "content": "hi"}]) is None  # skipped for a while, no second request
	assert select_skill(llm, CHOICES, {})[1] == "llm"  # falls back to the chars estimate


def test_aggregate_sums_llm_statistics_and_tolerates_old_results():
	base = {"verdict": "SUCCESS", "label_correct": True, "label": "normal", "sim_frames": 1, "views": 1, "rewinds": 0}
	a = {
		**base,
		"failures": [],
		"llm": {
			"consulted": 3,
			"adopted": 2,
			"fallback_reasons": {"timeout": 1},
			"prompt_tokens": [300, 400, 500],
			"latency_s_total": 2.0,
		},
	}
	b = {
		**base,
		"failures": [],
		"llm": {"consulted": 1, "adopted": 0, "fallback_reasons": {"timeout": 1, "empty_output": 0}},
	}
	total = aggregate([a, b, {**base, "failures": []}])
	assert (total["llm_consulted"], total["llm_adopted"], total["llm_adoption_rate"]) == (4, 2, 0.5)
	assert total["fallback_reasons"]["timeout"] == 2
	assert (total["llm_prompt_tokens_median"], total["llm_prompt_tokens_max"]) == (400, 500)
	assert total["llm_latency_s_total"] == 2.0
	assert aggregate([{**base, "failures": []}])["llm_adoption_rate"] is None


def test_decision_context_is_compact_leak_free_and_describes_checkpoints():
	pytest.importorskip("mujoco")
	from digital_twin_armigo.agent.twin_mujoco import TwinMujoco, sample_case
	from digital_twin_armigo.inspection import assert_no_defect_leakage

	class Trace:
		events: ClassVar[list] = []

		def __getattr__(self, name):
			return lambda *args, **kwargs: None

	backend = TwinMujoco(sample_case(442701, True, 4), None, Trace())
	try:
		backend.prepare()
		ctx = backend.decision_context(CHOICES, [], [])
		assert ctx["missing_faces"] == [0, 1, 2, 3, 4, 5] and ctx["grasped"] is False
		assert len(json.dumps(ctx)) < 1200
		assert "geometry" not in ctx and "bodies" not in json.dumps(ctx)
		assert_no_defect_leakage(ctx)
		views = [{"id": "0:5:wrist_cam", "camera": "wrist_cam", "faces": [0, 1], "score": -0.4, "threshold": 0.1}]
		failures = [
			{"code": "PHYSICAL_HEALTH_FAILED", "skill": "place", "reason_codes": ["SLIP"], "prior_views": views}
		]
		ctx = backend.decision_context(
			["checkpoint-0", "stop"], views, failures, {"checkpoint-0": "rewind and carry 40% slower"}
		)
		assert ctx["views"][0]["margin"] == -0.5 and ctx["covered_faces"] == [0, 1]
		assert ctx["failures"] == [{"code": "PHYSICAL_HEALTH_FAILED", "skill": "place", "reason_codes": ["SLIP"]}]
		assert ctx["checkpoints"][0]["id"] == "checkpoint-0"
		assert ctx["choice_notes"]["checkpoint-0"] == "rewind and carry 40% slower"
		assert "stop" in ctx["choice_notes"]
		# choose() logs the exact fields scripts/bench_agent.py reads and keeps the per-episode statistics.
		events = []
		backend.event = lambda kind, **payload: events.append((kind, payload))
		backend.llm = FakeLLM(['{"skill":"present_bottom"}'])
		assert backend.choose(CHOICES, [], []) == "present_bottom"
		payload = events[-1][1]
		assert payload["source"] == "llm" and payload["fallback_reason"] is None
		assert payload["attempts"] == 1 and payload["prompt_tokens"] > 0 and payload["latency_s"] == 0.0
		backend.llm = FakeLLM(["nope"])
		assert backend.choose(CHOICES, [], []) == "scan_wrist"
		assert events[-1][1]["fallback_reason"] == "invalid_json"
		summary = backend._llm_summary()
		assert (summary["consulted"], summary["adopted"], summary["fallback_reasons"]) == (2, 1, {"invalid_json": 1})
	finally:
		backend.close()
