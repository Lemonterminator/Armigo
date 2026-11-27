"""Re-view rule of strategies B and C (agent/review.py)."""

from digital_twin_armigo.agent.review import llm_choice, review_decision, rule_choice
from digital_twin_armigo.agent.twin_graph import decision


def view(vid, faces, score, q=0.0, review_of=None, camera="wrist_cam"):
	v = {"id": vid, "camera": camera, "faces": faces, "score": score, "threshold": 0.0, "q": [q] * 6}
	if review_of:
		v["review_of"] = review_of
	return v


CLEAR_ALL = [
	view("w1", [0, 2, 4], -1.0, 0.0),
	view("w2", [1, 3, 4], -1.0, 0.5),
	view("t1", [5], -1.0, 1.0, camera="top"),
]


def test_single_alarm_is_pending_not_defective():
	views = [view("w1", [0, 2, 4], 1.2, 0.0)]
	label, _covered, pending = review_decision(views)
	assert label == "unknown" and pending[0]["alarm"] == "w1"
	assert decision(views)[0] == "defective"  # strategy A decides at once


def test_second_alarm_from_another_pose_confirms():
	views = [view("w1", [0, 2, 4], 1.2, 0.0), view("r1", [0, 4], 0.4, 0.3, review_of="w1")]
	assert review_decision(views)[0] == "defective"


def test_same_pose_does_not_confirm():
	views = [view("w1", [0, 2, 4], 1.2, 0.0), view("r1", [0, 4], 0.4, 0.05, review_of="w1")]
	assert review_decision(views)[0] == "unknown"


def test_one_clear_review_is_a_conflict():
	views = [view("w1", [0], 1.2, 0.0), view("r1", [0], -1.0, 0.4, review_of="w1")]
	label, _, pending = review_decision(views)
	assert label == "unknown" and pending and pending[0]["faces"] == [0]


def test_refuted_alarm_lets_full_coverage_be_normal():
	views = [*CLEAR_ALL, view("w3", [0], 1.0, 0.8), view("r1", [0], -1.0, 1.2, review_of="w3")]
	label, covered, pending = review_decision(views)
	assert not pending and label == "normal" and covered == set(range(6))


def test_refutation_needs_a_review():
	views = [*CLEAR_ALL, view("w3", [0], 1.0, 0.8), view("w4", [0], -1.0, 1.5)]
	assert review_decision(views)[0] == "unknown"


def test_rule_choice_prefers_pixels_then_separation():
	cands = [
		{"id": "c0", "target_px": 900, "separation_rad": 0.3, "path_rad": 0.1},
		{"id": "c1", "target_px": 2500, "separation_rad": 0.3, "path_rad": 2.0},
		{"id": "c2", "target_px": 2600, "separation_rad": 1.2, "path_rad": 2.0},
	]
	assert rule_choice(cands) == "c2"


def test_rule_choice_uses_priority_over_raw_pixels():
	# c0 is all top face (already cleared, low priority); c1 shows the open side face: priority wins.
	cands = [
		{"id": "c0", "target_px": 26000, "priority_px": 3000, "separation_rad": 0.7, "path_rad": 0.3},
		{"id": "c1", "target_px": 16000, "priority_px": 9000, "separation_rad": 0.6, "path_rad": 0.5},
	]
	assert rule_choice(cands) == "c1"


class _Resp:
	def __init__(self, content):
		self.content, self.latency_s, self.prompt_tokens, self.cached_tokens = content, 0.01, 10, 0


class _FakeLLM:
	structured = True

	def __init__(self, answers):
		self.answers, self.calls = list(answers), []

	def chat(self, messages, json_schema=None):
		self.calls.append(messages[-1])
		return _Resp(self.answers.pop(0))


def test_llm_choice_one_image_per_request_and_fallback():
	import numpy as np

	cands = [
		{"id": "c0", "target_px": 900, "separation_rad": 0.3, "path_rad": 0.1},
		{"id": "c1", "target_px": 3000, "separation_rad": 0.9, "path_rad": 0.2},
	]
	images = [
		{"name": "a", "image": np.zeros((8, 8, 3), np.uint8)},
		{"name": "b", "image": np.zeros((8, 8, 3), np.uint8)},
	]
	llm = _FakeLLM(['{"note":"ok"}', '{"note":"ok"}', '{"candidate":"c0"}'])
	choice, source, info = llm_choice(llm, images, cands, {})
	assert (choice, source, info["requests"]) == ("c0", "llm", 3)
	assert all(sum(p.get("type") == "image_url" for p in m["content"]) == 1 for m in llm.calls[:2])
	bad = _FakeLLM(['{"note":"x"}', '{"note":"x"}', '{"candidate":"c9"}'])
	choice, source, info = llm_choice(bad, images, cands, {})
	assert (choice, source, info["reason"]) == ("c1", "fallback", "illegal_choice")
