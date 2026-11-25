"""Pairing of truth and realistic trajectories of the same case, and the pair outcome matrix."""

from __future__ import annotations

__all__ = ["OUTCOMES", "flags", "pair_key", "pair_outcome", "sensing_of"]

OUTCOMES = ("both_pass", "truth_only", "realistic_only", "both_fail", "void", "unpaired")


def sensing_of(group: str) -> str:
	"""Return 'truth' or 'realistic' from an experiment group name."""
	head = group.split("-", 1)[0]
	if head in ("truth", "realistic"):
		return head
	raise ValueError(f"cannot tell the sensing of group {group!r}")


def pair_key(row: dict) -> tuple:
	"""Canonical pairing key ignoring sensing and camera condition."""
	return (
		row["engine"],
		row["case_id"],
		row["detector"],
		row.get("strategy") or "A",
		row.get("motion") or "open",
		row.get("evidence") or "hold",
	)


def _void(r: dict) -> bool:
	return bool(r.get("infra_failed") or r.get("quarantined"))


def pair_outcome(truth: dict | None, real: dict | None) -> str:
	"""Classify a (truth, realistic) episode pair into OUTCOMES."""
	if truth is None or real is None:
		return "unpaired"
	if _void(truth) or _void(real):
		return "void"
	t, r = bool(truth.get("physical_success")), bool(real.get("physical_success"))
	return {(True, True): "both_pass", (True, False): "truth_only", (False, True): "realistic_only"}.get(
		(t, r), "both_fail"
	)


def flags(row: dict) -> dict:
	"""Extract boolean governance flags and outcome_source from a scored episode row."""
	return {
		"recovered": bool(row.get("n_recoveries")),
		"review_used": bool(row.get("n_review_views")),
		"llm_used": bool(row.get("n_llm_images")),
		"oracle": bool(row.get("oracle")),
		"outcome_source": (row.get("physical") or {}).get("source") or "verdict",
	}
