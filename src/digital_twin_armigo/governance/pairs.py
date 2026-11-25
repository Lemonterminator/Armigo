"""Pairing of truth and realistic trajectories of the same case, and the pair outcome matrix."""

from __future__ import annotations

__all__ = ["OUTCOMES", "flags", "pair_key", "pair_outcome", "sensing_of", "truth_compatible"]

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


def _row_is_oracle(row: dict) -> bool:
	return bool(row.get("oracle")) or bool((row.get("flags") or {}).get("oracle"))


def _canonical_case(case) -> list | None:
	"""Normalize ``task.case`` by ignoring only the ``cameras`` option (truth runs fixed while realistic may run recal/drift)."""
	if case is None:
		return None
	items = list(case)
	if len(items) >= 4 and isinstance(items[3], dict):
		opts = {k: v for k, v in items[3].items() if k != "cameras"}
		items = [*items[:3], opts] if opts else items[:3]
	return items


def truth_compatible(
	truth: dict,
	real: dict,
	*,
	allow_cross_batch: bool = True,
	allow_suite_subset: bool = False,
) -> tuple[bool, str]:
	"""Check whether ``truth`` is a valid baseline for ``real``; returns ``(ok, scope_or_reason)``."""
	t_pk = truth.get("pair_key")
	r_pk = real.get("pair_key")
	if not t_pk or not r_pk or list(t_pk) != list(r_pk):
		return False, "pair_key_mismatch"
	if _row_is_oracle(truth) or _row_is_oracle(real):
		return False, "oracle"
	t_seg = str(truth.get("segmenter") or "lab")
	r_seg = str(real.get("segmenter") or "lab")
	if t_seg != r_seg:
		return False, "segmenter_mismatch"
	t_case = _canonical_case(truth.get("case_tuple"))
	r_case = _canonical_case(real.get("case_tuple"))
	if t_case is not None and r_case is not None and t_case != r_case:
		return False, "case_mismatch"

	if truth.get("batch") == real.get("batch"):
		return True, "same_batch"

	if not allow_cross_batch:
		return False, "cross_batch_disabled"

	t_commit = truth.get("provenance_commit")
	r_commit = real.get("provenance_commit")
	if (t_commit or r_commit) and t_commit != r_commit:
		return False, "commit_mismatch"

	t_sha = truth.get("suite_sha256")
	r_sha = real.get("suite_sha256")
	if t_sha and r_sha and t_sha != r_sha:
		if allow_suite_subset:
			return True, "cross_batch_subset"
		return False, "suite_sha256_mismatch"
	return True, "cross_batch"
