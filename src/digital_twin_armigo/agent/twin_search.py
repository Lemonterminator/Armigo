"""Outer policy search: exhaustive successive halving with paired statistics.

The policy space is tiny (2 x 3 x 3 x 3 = 54), so every policy is measured. Halving spends the episode budget on
the promising ones: each round, all surviving policies run the SAME nested prefix of one dev suite (a paired design),
the top 1/eta survive, and the suite prefix doubles. The default policy is always carried to the final round as the
control. The best finalist replaces the default only if a paired test says it is better. An optional LLM analysis is
advisory and never influences selection.
"""

from __future__ import annotations

import itertools
import json
import math
import random
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict

from .llm import ChatLLM, LLMError
from .twin_graph import Policy

DEV_SEED_START = 700_001
DEFAULT_ALPHA = 0.05
# Seeds already used elsewhere; the generated suite must stay disjoint from them.
RESERVED_SEEDS = frozenset({187271, 187272, 442701, 442702, 442703, *range(600_001, 600_009)})

EvaluateBatch = Callable[[Sequence[tuple[Policy, tuple]]], list[dict]]


def all_policies() -> list[Policy]:
	"""Every legal policy, default first."""
	grid = itertools.product(("wrist_first", "bottom_first"), (1, 2, 3), (1, 2, 3), (0, 1, 2))
	policies = [Policy(*row) for row in grid]
	policies.sort(key=lambda p: p != Policy())
	return policies


def dev_suite(size: int, start_seed: int = DEV_SEED_START) -> list[tuple[int, bool, int]]:
	"""Deterministic balanced fixtures; any prefix is balanced to within one case.

	Case 2j is the normal and 2j+1 the defective brick of seed ``start_seed + j``; the defect face is ``j % 6``."""
	if size < 1:
		raise ValueError("suite size must be positive")
	cases = [(start_seed + i // 2, i % 2 == 1, (i // 2) % 6) for i in range(size)]
	if RESERVED_SEEDS & {case[0] for case in cases}:
		raise ValueError("dev suite overlaps a reserved seed range")
	return cases


def case_row(result: dict) -> dict:
	"""The per-episode facts the search needs."""
	return {
		"success": result["verdict"] == "SUCCESS",
		"label_correct": bool(result["label_correct"]),
		"sim_frames": int(result["sim_frames"]),
		"views": int(result["views"]),
		"failures": [failure["code"] for failure in result.get("failures", ())],
	}


def rank_key(aggregate: dict):
	"""Lower is better: successes, correct labels, then frames and views."""
	return (-aggregate["successes"], -aggregate["labels_correct"], aggregate["sim_frames"], aggregate["views"])


def aggregate_rows(rows: Sequence[dict]) -> dict:
	failures = Counter(code for row in rows for code in row["failures"])
	return {
		"episodes": len(rows),
		"successes": sum(row["success"] for row in rows),
		"labels_correct": sum(row["label_correct"] for row in rows),
		"sim_frames": sum(row["sim_frames"] for row in rows),
		"views": sum(row["views"] for row in rows),
		"failures": [{"code": code, "count": count} for code, count in sorted(failures.items())],
	}


def plan_rounds(n_policies: int, cap: int, *, max_finalists: int = 3) -> dict:
	"""Pick the plan with the most final fixtures whose cost, counting reuse of earlier episodes, fits ``cap``."""
	best = None
	for eta in (2, 3, 4, 6, 9, 18):
		for k0 in range(1, 13):
			survivors, fixtures, cost, rounds, previous = n_policies, k0, 0, [], 0
			while True:
				cost += survivors * (fixtures - previous)
				rounds.append({"policies": survivors, "fixtures": fixtures})
				if survivors <= max_finalists:
					break
				previous = fixtures
				fixtures *= 2
				survivors = max(max_finalists, math.ceil(survivors / eta))
			cost += 2 * fixtures  # worst case: confirm the best finalist and the default on fresh fixtures
			if cost <= cap:
				candidate = {"eta": eta, "k0": k0, "rounds": rounds, "cost": cost, "final_fixtures": fixtures}
				if best is None or (min(k0, 4), fixtures, -cost) > (
					min(best["k0"], 4),
					best["final_fixtures"],
					-best["cost"],
				):
					best = candidate
	if best is None:
		raise ValueError(f"episode cap {cap} cannot evaluate {n_policies} policies once")
	return best


def exact_sign_test(wins: int, losses: int) -> float:
	"""Two-sided exact McNemar / sign test on discordant pairs."""
	n = wins + losses
	if n == 0:
		return 1.0
	tail = sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / 2**n
	return min(1.0, 2 * tail)


def paired_bootstrap_ci(diffs: Sequence[float], alpha: float, *, resamples: int = 4000, seed: int = 0):
	"""Percentile CI of the mean paired difference."""
	rng = random.Random(seed)
	n = len(diffs)
	means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
	lo = means[int(resamples * alpha / 2)]
	hi = means[min(resamples - 1, int(resamples * (1 - alpha / 2)))]
	return lo, hi


def compare_to_default(candidate: Sequence[dict], default: Sequence[dict], alpha: float, seed: int = 0) -> dict:
	"""Paired comparison of per-case rows (same fixtures, same order)."""
	wins = sum(c["success"] and not d["success"] for c, d in zip(candidate, default, strict=True))
	losses = sum(d["success"] and not c["success"] for c, d in zip(candidate, default, strict=True))
	diffs = [c["sim_frames"] - d["sim_frames"] for c, d in zip(candidate, default, strict=True)]
	lo, hi = paired_bootstrap_ci(diffs, alpha, seed=seed)
	return {
		"n_pairs": len(diffs),
		"candidate_only_success": wins,
		"default_only_success": losses,
		"success_p_value": exact_sign_test(wins, losses),
		"frames_diff_mean": sum(diffs) / len(diffs),
		"frames_diff_ci": [lo, hi],
		"alpha": alpha,
	}


def decide(stats: dict) -> tuple[bool, str]:
	"""Replace the default only on a significant, non-regressing improvement."""
	wins, losses, alpha = stats["candidate_only_success"], stats["default_only_success"], stats["alpha"]
	if stats["success_p_value"] < alpha and wins > losses:
		return True, f"success significantly better ({wins} vs {losses} discordant, p={stats['success_p_value']:.4f})"
	if wins >= losses and stats["frames_diff_ci"][1] < 0:
		return True, "success not worse and sim_frames significantly lower (bootstrap CI excludes 0)"
	lo, hi = stats["frames_diff_ci"]
	return False, (
		f"no significant improvement over the default (discordant {wins} vs {losses}, "
		f"p={stats['success_p_value']:.4f}, frames diff CI {lo:.0f}..{hi:.0f})"
	)


def successive_halving(
	evaluate_batch: EvaluateBatch,
	cap: int,
	*,
	policies: Sequence[Policy] | None = None,
	suite: Callable[[int], list] = dev_suite,
	alpha: float = DEFAULT_ALPHA,
	seed: int = 0,
	max_finalists: int = 3,
) -> dict:
	"""Run the search. ``evaluate_batch([(policy, case), ...])`` returns one episode result dict per item."""
	policies = list(policies or all_policies())
	default = Policy()
	if default not in policies:
		policies.insert(0, default)
	plan = plan_rounds(len(policies), cap, max_finalists=max_finalists)
	cases = suite(2 * plan["final_fixtures"])  # second half: fresh confirmation fixtures
	rows: dict[Policy, list[dict]] = {p: [] for p in policies}
	alive, rounds, episodes = list(policies), [], 0
	for number, step in enumerate(plan["rounds"]):
		k = step["fixtures"]
		# Survivors reuse earlier episodes; only the new fixtures of the doubled prefix are run.
		items = [(p, cases[i]) for p in alive for i in range(len(rows[p]), k)]
		if items:
			results = evaluate_batch(items)
			if len(results) != len(items):
				raise RuntimeError("evaluate_batch returned a different number of results")
			for (policy, _), result in zip(items, results, strict=True):
				rows[policy].append(case_row(result))
			episodes += len(items)
		scored = {p: aggregate_rows(rows[p][:k]) for p in alive}
		ordered = sorted(alive, key=lambda p: (rank_key(scored[p]), p != default))
		rounds.append(
			{
				"round": number + 1,
				"fixtures": k,
				"new_episodes": len(items),
				"policies": [{"policy": asdict(p), **scored[p]} for p in ordered],
			}
		)
		if number + 1 < len(plan["rounds"]):
			keep = plan["rounds"][number + 1]["policies"]
			kept = ordered[:keep]
			if default not in kept:
				kept = [*kept[: keep - 1], default]
			alive = kept
		else:
			alive = ordered
	final_k = plan["rounds"][-1]["fixtures"]
	best = alive[0]
	selected, reason, stats, confirmation = default, "the default ranked first among the finalists", None, None
	if best != default:
		# The finalist was picked on these fixtures, so a test on them would be biased (winner's curse). Confirm on
		# fresh fixtures: best finalist and default, paired on the next final_k cases.
		fresh = cases[final_k : 2 * final_k]
		items = [(p, case) for p in (best, default) for case in fresh]
		results = evaluate_batch(items)
		if len(results) != len(items):
			raise RuntimeError("evaluate_batch returned a different number of results")
		episodes += len(items)
		best_rows, default_rows = (
			[case_row(r) for r in results[: len(fresh)]],
			[case_row(r) for r in results[len(fresh) :]],
		)
		stats = compare_to_default(best_rows, default_rows, alpha, seed=seed)
		confirmation = {"cases": [list(c) for c in fresh], "best": best_rows, "default": default_rows}
		replace, reason = decide(stats)
		if replace:
			selected = best
	final = {p: aggregate_rows(rows[p][:final_k]) for p in alive}
	source = "default" if selected == default else "halving"
	return {
		"selected": {"policy": asdict(selected), "source": source, **final[selected]},
		"selected_is_default": selected == default,
		"reason": reason,
		"best_finalist": asdict(best),
		"statistics": stats,
		"confirmation": confirmation,
		"plan": plan,
		"rounds": rounds,
		"final_fixtures": final_k,
		"suite": [list(case) for case in cases],
		"per_policy": [{"policy": asdict(p), "episodes": len(rows[p]), "rows": rows[p]} for p in policies],
		"cost": {"episodes": episodes, "cap": cap, "exhaustive_episodes": len(policies) * final_k},
		"candidates": [{"policy": asdict(p), **final[p]} for p in alive],
	}


def analyze_failures(llm: ChatLLM, report: dict, max_tokens: int = 500) -> dict:
	"""Advisory only: ask the LLM to explain failure codes and suggest new policy dimensions. Never raises."""
	counts: Counter = Counter()
	for entry in report.get("per_policy", ()):
		for row in entry["rows"]:
			counts.update(row["failures"])
	payload = {
		"failure_code_counts": dict(counts.most_common(8)),
		"policy_dimensions": ["order", "max_bottom_views", "max_wrist_views", "max_rewinds"],
		"selected": report["selected"]["policy"],
		"selected_is_default": report["selected_is_default"],
	}
	old = getattr(llm, "max_tokens", None)
	try:
		if old is not None:
			llm.max_tokens = max_tokens
		response = llm.chat(
			[
				{
					"role": "system",
					"content": (
						"You review a robot inspection policy search. Return one JSON object with fields "
						"explanations (list of {code, explanation}) and new_dimensions (list of {name, rationale}): "
						"parameters worth adding to the search. Be brief. Your output is advisory only."
					),
				},
				{"role": "user", "content": json.dumps(payload, separators=(",", ":"))},
			]
		)
		raw = json.loads(response.content)
		return {
			"status": "ok",
			"explanations": [
				{"code": str(item["code"])[:64], "explanation": str(item["explanation"])[:400]}
				for item in raw["explanations"][:8]
			],
			"new_dimensions": [
				{"name": str(item["name"])[:64], "rationale": str(item["rationale"])[:400]}
				for item in raw["new_dimensions"][:6]
			],
		}
	except (KeyError, TypeError, ValueError, AttributeError, LLMError) as exc:
		return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"[:300]}
	finally:
		if old is not None:
			llm.max_tokens = old
