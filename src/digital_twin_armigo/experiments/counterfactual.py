"""Counterfactual oracle re-runs: which belief, replaced by truth, turns a failed realistic episode into a success."""

from __future__ import annotations

VARIANTS = ((), ("camera",), ("brick",), ("bowls",), ("camera", "brick", "bowls"))


def verdict(original_success: bool, variants: dict[tuple, bool | None]) -> dict:
	if any(variants.get(v) is None for v in VARIANTS):
		return {"verdict": "incomplete", "minimal": []}
	if variants[()] != original_success:
		return {"verdict": "repro_mismatch", "minimal": []}
	singles = sorted([list(v) for v in VARIANTS if len(v) == 1 and variants[v]])
	if singles:
		return {"verdict": "confirmed", "minimal": singles}
	if variants[VARIANTS[-1]]:
		return {"verdict": "joint", "minimal": [list(VARIANTS[-1])]}
	return {"verdict": "not_perception", "minimal": []}
