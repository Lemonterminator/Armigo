import json
import random

from digital_twin_armigo.agent.llm import FakeLLM
from digital_twin_armigo.agent.twin_graph import Policy
from digital_twin_armigo.agent.twin_search import (
	RESERVED_SEEDS,
	all_policies,
	analyze_failures,
	compare_to_default,
	dev_suite,
	exact_sign_test,
	plan_rounds,
	successive_halving,
)

BEST = Policy("bottom_first", 2, 3, 2)


def make_evaluator(rate, seed, frames=lambda p: 1000):
	"""Synthetic stochastic episodes: independent Bernoulli(rate(policy)) per (policy, case)."""
	rng = random.Random(seed)
	calls = []

	def evaluate_batch(items):
		calls.append(len(items))
		out = []
		for policy, case in items:
			ok = rng.random() < rate(policy)
			out.append(
				{
					"verdict": "SUCCESS" if ok else "FAILED",
					"label_correct": True,
					"sim_frames": int(frames(policy) + rng.gauss(0, 100)),
					"views": 3,
					"failures": [] if ok else [{"code": "PLACE_MISS"}],
				}
			)
		return out

	evaluate_batch.calls = calls
	return evaluate_batch


def test_policy_grid_and_dev_suite_properties():
	policies = all_policies()
	assert len(policies) == len(set(policies)) == 54
	assert policies[0] == Policy()
	suite = dev_suite(48)
	assert suite == dev_suite(60)[:48]
	assert not RESERVED_SEEDS & {case[0] for case in suite}
	assert sum(case[1] for case in suite) == 24
	assert {case[2] for case in suite if case[1]} == set(range(6))
	assert all(abs(sum(c[1] for c in suite[:n]) - n / 2) <= 0.5 for n in range(1, 49))


def test_plan_fits_cap_and_uses_nested_prefixes():
	for cap in (80, 120, 270, 800):
		plan = plan_rounds(54, cap)
		assert plan["cost"] <= cap
		assert plan["rounds"][0]["policies"] == 54
		assert plan["rounds"][-1]["policies"] <= 3
	assert plan_rounds(54, 270)["final_fixtures"] > plan_rounds(54, 80)["final_fixtures"]


def test_exact_sign_test_and_bootstrap_comparison():
	assert exact_sign_test(0, 0) == 1.0
	assert abs(exact_sign_test(5, 0) - 0.0625) < 1e-12
	assert exact_sign_test(3, 3) == 1.0
	row = {"success": True, "label_correct": True, "views": 3, "failures": []}
	best = [{**row, "sim_frames": 900} for _ in range(10)]
	base = [{**row, "sim_frames": 1000, "success": i > 1} for i in range(10)]
	stats = compare_to_default(best, base, 0.05)
	assert stats["candidate_only_success"] == 2 and stats["default_only_success"] == 0
	assert stats["frames_diff_ci"] == [-100.0, -100.0]


def test_halving_selects_the_truly_best_policy_cheaply():
	rate = lambda p: 0.95 if p == BEST else 0.55
	picked, trials = 0, 20
	for seed in range(trials):
		report = successive_halving(make_evaluator(rate, seed), 800)
		picked += report["selected"]["policy"] == BEST.__dict__
		assert report["cost"]["episodes"] <= 800
		# exhaustive evaluation of all 54 policies on the final fixtures would cost >4x more
		assert report["cost"]["episodes"] * 4 < report["cost"]["exhaustive_episodes"]
	assert picked / trials >= 0.85


def test_smooth_landscape_improves_on_the_default_without_regressing():
	def rate(p):
		return 0.45 + 0.1 * (p.max_wrist_views - 1) + 0.1 * p.max_rewinds + (0.05 if p.order == "bottom_first" else 0)

	trials, better = 30, 0
	for seed in range(trials):
		chosen = Policy(**successive_halving(make_evaluator(rate, seed), 800)["selected"]["policy"])
		assert rate(chosen) >= rate(Policy()) - 0.1
		better += rate(chosen) > rate(Policy())
	assert better / trials >= 0.5


def test_default_is_kept_when_differences_are_null():
	for cap in (400, 800):
		kept, trials = 0, 40
		for seed in range(trials):
			kept += successive_halving(make_evaluator(lambda p: 0.6, seed), cap)["selected_is_default"]
		assert kept / trials >= 0.9


def test_report_contains_rounds_statistics_and_cost():
	rate = lambda p: 0.9 if p == BEST else 0.5
	report = successive_halving(make_evaluator(rate, 1), 800)
	json.dumps(report)
	assert report["rounds"][0]["policies"].__len__() == 54
	assert len(report["per_policy"]) == 54
	assert report["cost"]["episodes"] == sum(r["new_episodes"] for r in report["rounds"]) + 2 * report["final_fixtures"]
	# the default is a finalist even if it would have been cut
	assert Policy().__dict__ in [c["policy"] for c in report["candidates"]]
	assert "reason" in report


def test_every_policy_in_a_round_sees_the_same_fixtures():
	seen = {}

	def evaluate_batch(items):
		for policy, case in items:
			seen.setdefault(policy, []).append(case)
		return make_evaluator(lambda p: 0.5, 3)(items)

	report = successive_halving(evaluate_batch, 200)
	first = report["suite"]
	for cases in seen.values():
		assert [list(c) for c in cases] == first[: len(cases)]


def test_llm_analysis_is_validated_and_never_raises():
	report = successive_halving(make_evaluator(lambda p: 0.5, 2), 120)
	good = '{"explanations":[{"code":"PLACE_MISS","explanation":"brick slipped"}],"new_dimensions":[{"name":"grip_force","rationale":"slip"}]}'
	out = analyze_failures(FakeLLM([good]), report)
	assert out["status"] == "ok" and out["new_dimensions"][0]["name"] == "grip_force"
	assert analyze_failures(FakeLLM(["not json"]), report)["status"] == "failed"
	assert analyze_failures(FakeLLM(['{"explanations":5}']), report)["status"] == "failed"
	assert report["selected"]["policy"]  # selection untouched by analysis
