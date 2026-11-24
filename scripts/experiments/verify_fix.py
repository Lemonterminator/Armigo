"""Verify a fix across three acceptance gates: failure bank, regression suite, and fresh seeds (Task F4).

Usage:
	python scripts/experiments/verify_fix.py \\
		--bank <suite> --regression <suite> --fresh <suite> \\
		--base <old_batch> --batch <new_batch> --criteria <criteria.json>
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path


def _find_repo() -> Path:
	env = os.environ.get("ARMIGO_REPO")
	if env and (Path(env) / "src" / "digital_twin_armigo").is_dir():
		return Path(env)
	here = Path(__file__).resolve()
	for parent in here.parents:
		if (parent / "src" / "digital_twin_armigo").is_dir():
			return parent
	fallback = Path("E:/digital_twin_armigo-v3")
	if (fallback / "src" / "digital_twin_armigo").is_dir():
		return fallback
	return here.parents[2]


REPO = _find_repo()
if str(REPO / "src") not in sys.path:
	sys.path.insert(0, str(REPO / "src"))

from digital_twin_armigo.experiments import runner
from digital_twin_armigo.experiments.contract import (
	CAMERA_CONDITIONS,
	DETECTORS,
	ENGINES,
	SENSING,
	STRATEGIES,
)
from digital_twin_armigo.experiments.contract import (
	case_id as make_case_id,
)
from digital_twin_armigo.experiments.report import scan_batch
from digital_twin_armigo.experiments.stats import METRICS, paired_comparison

DEFAULT_ROOT = Path("E:/Armigo_data_v3")
DEFAULT_CRITERIA = {
	"bank_min_fix_rate": 0.70,
	"regression_max_net_regression": 1,
	"regression_min_mcnemar_p": 0.20,
	"fresh_min_ci_lo": -0.02,
	"bootstrap_b": 2000,
	"seed": 0,
	"metric": "success_rate",
}


def cluster_slug(cluster: str) -> str:
	slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", cluster.strip()).strip("-")
	return slug or "cluster"


def _read_json(path: Path):
	return json.loads(path.read_text(encoding="utf-8-sig"))


def _read_jsonl(path: Path) -> list[dict]:
	rows: list[dict] = []
	with path.open(encoding="utf-8-sig") as fh:
		for line in fh:
			line = line.strip()
			if line:
				rows.append(json.loads(line))
	return rows


def load_score_rows(source: Sequence[dict] | str | Path) -> list[dict]:
	"""Load score rows from an in-memory sequence, a ``scores.jsonl`` file, or a report/batch dir."""
	if not isinstance(source, (str, Path)):
		return list(source)
	path = Path(source)
	if path.is_file():
		return _read_jsonl(path)
	if path.is_dir():
		jsonl = path / "scores.jsonl"
		if jsonl.is_file():
			return _read_jsonl(jsonl)
		rows, _plans = scan_batch(path)
		return rows
	raise FileNotFoundError(f"score source not found: {path}")


def normalize_criteria(raw: dict | None) -> dict:
	raw = dict(raw or {})
	g1 = raw.get("gate1") if isinstance(raw.get("gate1"), dict) else {}
	g2 = raw.get("gate2") if isinstance(raw.get("gate2"), dict) else {}
	g3 = raw.get("gate3") if isinstance(raw.get("gate3"), dict) else {}

	def pick(srcs: tuple[dict, ...], keys: tuple[str, ...], default):
		for src in srcs:
			for k in keys:
				if k in src and src[k] is not None:
					return src[k]
		return default

	return {
		"bank_min_fix_rate": float(
			pick((raw, g1), ("bank_min_fix_rate", "min_fix_rate"), DEFAULT_CRITERIA["bank_min_fix_rate"])
		),
		"regression_max_net_regression": int(
			pick(
				(raw, g2),
				("regression_max_net_regression", "max_net_regression", "max_net_loss", "regression_max_net_loss"),
				DEFAULT_CRITERIA["regression_max_net_regression"],
			)
		),
		"regression_min_mcnemar_p": float(
			pick(
				(raw, g2),
				("regression_min_mcnemar_p", "min_mcnemar_p"),
				DEFAULT_CRITERIA["regression_min_mcnemar_p"],
			)
		),
		"fresh_min_ci_lo": float(
			pick((raw, g3), ("fresh_min_ci_lo", "min_ci_lo"), DEFAULT_CRITERIA["fresh_min_ci_lo"])
		),
		"bootstrap_b": int(pick((raw, g3), ("bootstrap_b", "B"), DEFAULT_CRITERIA["bootstrap_b"])),
		"seed": int(pick((raw, g3), ("seed",), DEFAULT_CRITERIA["seed"])),
		"metric": str(pick((raw,), ("metric",), DEFAULT_CRITERIA["metric"])),
		**({"cluster": str(raw["cluster"])} if raw.get("cluster") else {}),
	}


def load_criteria(path: Path) -> dict:
	data = _read_json(path)
	if not isinstance(data, dict):
		raise TypeError(f"criteria JSON must be an object: {path}")
	return normalize_criteria(data)


def evaluate_gate1_bank(
	bank_base: Sequence[dict] | str | Path,
	bank_new: Sequence[dict] | str | Path,
	*,
	min_fix_rate: float = 0.70,
) -> dict:
	"""Gate 1: fraction of baseline bank failures that succeed after the fix."""
	base_rows = load_score_rows(bank_base)
	new_rows = load_score_rows(bank_new)
	base_map = {r["case_id"]: r for r in base_rows}
	new_map = {r["case_id"]: r for r in new_rows}

	base_failures = [cid for cid, r in sorted(base_map.items()) if not bool(r.get("physical_success"))]
	fixed = sum(1 for cid in base_failures if cid in new_map and bool(new_map[cid].get("physical_success")))
	missing_in_new = sum(1 for cid in base_failures if cid not in new_map)
	regressed_in_bank = sum(
		1
		for cid, r in base_map.items()
		if bool(r.get("physical_success")) and cid in new_map and not bool(new_map[cid].get("physical_success"))
	)
	total_base_failures = len(base_failures)
	fix_rate = (fixed / total_base_failures) if total_base_failures > 0 else 0.0
	passed = total_base_failures > 0 and fix_rate >= min_fix_rate
	return {
		"n_base_rows": len(base_map),
		"n_new_rows": len(new_map),
		"total_base_failures": total_base_failures,
		"fixed": fixed,
		"still_failing": total_base_failures - fixed - missing_in_new,
		"missing_in_new": missing_in_new,
		"regressed_in_bank": regressed_in_bank,
		"fix_rate": fix_rate,
		"min_fix_rate": min_fix_rate,
		"passed": passed,
	}


def evaluate_gate2_regression(
	reg_base: Sequence[dict] | str | Path,
	reg_new: Sequence[dict] | str | Path,
	*,
	max_net_regression: int = 1,
	min_mcnemar_p: float = 0.20,
	bootstrap_b: int = 2000,
	seed: int = 0,
	metric: str = "success_rate",
) -> dict:
	"""Gate 2: paired regression check via exact McNemar test on discordant pairs."""
	base_rows = load_score_rows(reg_base)
	new_rows = load_score_rows(reg_new)
	pc = paired_comparison(base_rows, new_rows, metric=metric, B=bootstrap_b, seed=seed)
	n10 = int(pc["n10"])  # A passed, B failed (new failures)
	n01 = int(pc["n01"])  # A failed, B passed (new successes)
	net_regression = n10 - n01
	p_mcnemar = float(pc["p_mcnemar"])
	# Reject only if net regression exceeds budget or discordant degradation is statistically detectable.
	passed = pc["n_pairs"] > 0 and net_regression <= max_net_regression and (n10 <= n01 or p_mcnemar > min_mcnemar_p)
	return {
		"metric": metric,
		"n_pairs": int(pc["n_pairs"]),
		"n_clusters": int(pc["n_clusters"]),
		"n10": n10,
		"n01": n01,
		"n11": int(pc["n11"]),
		"n00": int(pc["n00"]),
		"net_regression": net_regression,
		"p_mcnemar": p_mcnemar,
		"max_net_regression": max_net_regression,
		"min_mcnemar_p": min_mcnemar_p,
		"unmatched_base": int(pc["unmatched_a"]),
		"unmatched_new": int(pc["unmatched_b"]),
		"passed": passed,
	}


def _matched_rates(base_rows: list[dict], new_rows: list[dict], metric: str) -> tuple[float, float]:
	num, den = METRICS[metric]
	a = {r["case_id"]: r for r in base_rows}
	b = {r["case_id"]: r for r in new_rows}
	shared = [k for k in sorted(set(a) & set(b)) if den(a[k])]
	if not shared:
		return math.nan, math.nan
	base_rate = sum(bool(num(a[k])) for k in shared) / len(shared)
	new_rate = sum(bool(num(b[k])) for k in shared) / len(shared)
	return float(base_rate), float(new_rate)


def evaluate_gate3_fresh(
	fresh_base: Sequence[dict] | str | Path,
	fresh_new: Sequence[dict] | str | Path,
	*,
	min_ci_lo: float = -0.02,
	bootstrap_b: int = 2000,
	seed: int = 0,
	metric: str = "success_rate",
) -> dict:
	"""Gate 3: layout-clustered bootstrap non-inferiority on fresh seeds."""
	base_rows = load_score_rows(fresh_base)
	new_rows = load_score_rows(fresh_new)
	pc = paired_comparison(base_rows, new_rows, metric=metric, B=bootstrap_b, seed=seed)
	base_rate, new_rate = _matched_rates(base_rows, new_rows, metric)
	ci_lo = float(pc["ci_lo"])
	passed = pc["n_pairs"] > 0 and math.isfinite(ci_lo) and ci_lo >= min_ci_lo
	return {
		"metric": metric,
		"n_pairs": int(pc["n_pairs"]),
		"n_clusters": int(pc["n_clusters"]),
		"base_rate": base_rate,
		"new_rate": new_rate,
		"diff": float(pc["diff"]),
		"ci_lo": ci_lo,
		"ci_hi": float(pc["ci_hi"]),
		"min_ci_lo": min_ci_lo,
		"bootstrap_b": bootstrap_b,
		"unmatched_base": int(pc["unmatched_a"]),
		"unmatched_new": int(pc["unmatched_b"]),
		"passed": passed,
	}


def evaluate_three_gates(
	bank_base: Sequence[dict] | str | Path,
	bank_new: Sequence[dict] | str | Path,
	reg_base: Sequence[dict] | str | Path,
	reg_new: Sequence[dict] | str | Path,
	fresh_base: Sequence[dict] | str | Path,
	fresh_new: Sequence[dict] | str | Path,
	criteria: dict | None = None,
) -> dict:
	"""Evaluate Gates 1, 2, 3 and return ``verdict`` in ``{'merge', 'overfit_bank', 'reject'}``."""
	crit = normalize_criteria(criteria)
	g1 = evaluate_gate1_bank(bank_base, bank_new, min_fix_rate=crit["bank_min_fix_rate"])
	g2 = evaluate_gate2_regression(
		reg_base,
		reg_new,
		max_net_regression=crit["regression_max_net_regression"],
		min_mcnemar_p=crit["regression_min_mcnemar_p"],
		bootstrap_b=crit["bootstrap_b"],
		seed=crit["seed"],
		metric=crit["metric"],
	)
	g3 = evaluate_gate3_fresh(
		fresh_base,
		fresh_new,
		min_ci_lo=crit["fresh_min_ci_lo"],
		bootstrap_b=crit["bootstrap_b"],
		seed=crit["seed"],
		metric=crit["metric"],
	)

	if g1["passed"] and g2["passed"] and g3["passed"]:
		verdict = "merge"
	elif g1["passed"] and g2["passed"] and not g3["passed"]:
		verdict = "overfit_bank"
	else:
		verdict = "reject"

	return {
		"criteria": crit,
		"gates": {"bank": g1, "regression": g2, "fresh": g3},
		"verdict": verdict,
	}


def _cases_to_ids(cases: list) -> set[str]:
	return {make_case_id(*c[:3]) for c in cases}


def resolve_suite_case_ids(
	suite_spec: str,
	*,
	roots: Sequence[Path],
	batches: Sequence[str] = (),
) -> set[str]:
	"""Resolve the set of ``case_id`` values belonging to a suite name or JSON path."""
	direct = Path(suite_spec)
	if direct.is_file():
		raw = _read_json(direct)
		cases = raw.get("cases") if isinstance(raw, dict) else raw
		if isinstance(cases, list):
			return _cases_to_ids(cases)

	for r in roots:
		cand = r / "suites" / f"{suite_spec}.json"
		if cand.is_file():
			raw = _read_json(cand)
			cases = raw.get("cases") if isinstance(raw, dict) else raw
			if isinstance(cases, list):
				return _cases_to_ids(cases)

	v2_dir = REPO / "config" / "agent_bench" / "v2"
	manifest_file = v2_dir / "manifest.json"
	if manifest_file.is_file():
		manifest = _read_json(manifest_file)
		meta = (manifest.get("suites") or {}).get(suite_spec)
		if isinstance(meta, dict):
			rel = meta.get("path")
			suite_file = (v2_dir / rel).resolve() if rel else (v2_dir / f"{suite_spec}.json")
			if suite_file.is_file():
				raw = _read_json(suite_file)
				cases = raw.get("cases") if isinstance(raw, dict) else raw
				if isinstance(cases, list):
					return _cases_to_ids(cases)

	for folder in (v2_dir, REPO / "config" / "agent_bench" / "failure_bank", REPO / "config" / "agent_bench"):
		cand = folder / f"{suite_spec}.json"
		if cand.is_file():
			raw = _read_json(cand)
			cases = raw.get("cases") if isinstance(raw, dict) else raw
			if isinstance(cases, list):
				return _cases_to_ids(cases)

	for r in roots:
		for b in batches:
			for plan_file in sorted((r / "runs").glob(f"{b}*/*/*/plan.json")):
				plan = _read_json(plan_file)
				if plan.get("suite") == suite_spec and isinstance(plan.get("cases"), list):
					return set(plan["cases"])

	raise FileNotFoundError(f"cannot resolve suite {suite_spec!r} to case_ids")


def _load_batch_rows(root: Path, batch: str) -> list[dict]:
	scores_file = root / "reports" / batch / "scores.jsonl"
	if scores_file.is_file():
		return _read_jsonl(scores_file)
	batch_dir = root / "runs" / batch
	if batch_dir.is_dir():
		rows, _plans = scan_batch(batch_dir)
		return rows
	return []


def _filter_rows(
	rows: list[dict],
	*,
	case_ids: set[str] | None,
	engine: str | None,
	group: str | None,
) -> list[dict]:
	out = rows
	if engine:
		out = [r for r in out if r.get("engine") == engine]
	if group:
		out = [r for r in out if r.get("group") == group]
	if case_ids is not None:
		out = [r for r in out if r.get("case_id") in case_ids]
	return out


def _select_batch_rows(
	root: Path,
	batch: str,
	role: str,
	*,
	case_ids: set[str],
	engine: str | None = None,
	group: str | None = None,
) -> list[dict]:
	for candidate in (batch, f"{batch}-{role}"):
		rows = _load_batch_rows(root, candidate)
		if not rows:
			continue
		filtered = _filter_rows(rows, case_ids=case_ids, engine=engine, group=group)
		if filtered:
			return filtered
	return []


def _unique_cell(rows: list[dict]) -> tuple[str | None, str | None] | None:
	cells = {(r.get("engine"), r.get("group")) for r in rows if r.get("group")}
	return next(iter(cells)) if len(cells) == 1 else None


def _disambiguate_pair(
	base_rows: list[dict],
	new_rows: list[dict],
	*,
	role: str,
	base_group: str | None,
	new_group: str | None,
) -> tuple[list[dict], list[dict]]:
	if base_group is None:
		new_cell = _unique_cell(new_rows)
		if new_cell is not None:
			matched = [r for r in base_rows if (r.get("engine"), r.get("group")) == new_cell]
			if matched:
				base_rows = matched
	if new_group is None:
		base_cell = _unique_cell(base_rows)
		if base_cell is not None:
			matched = [r for r in new_rows if (r.get("engine"), r.get("group")) == base_cell]
			if matched:
				new_rows = matched

	for label, rows in (("base", base_rows), ("batch", new_rows)):
		groups = sorted({f"{r.get('engine')}/{r.get('group')}" for r in rows if r.get("group")})
		if len(groups) > 1:
			raise SystemExit(
				f"{label} ({role}) matched multiple groups {groups}; pass --group / --base-group to select one"
			)
	return base_rows, new_rows


def _suite_ready(rows: list[dict], case_ids: set[str]) -> bool:
	if not case_ids or not rows:
		return False
	by_id = {r["case_id"]: r for r in rows}
	return all(cid in by_id and bool(by_id[cid].get("has_episode", True)) for cid in case_ids)


def run_suite_batch(args: argparse.Namespace, suite_spec: str, batch_name: str) -> None:
	"""Invoke ``experiments.runner.cmd_run`` resumably for ``suite_spec`` into ``batch_name``."""
	suite_name = Path(suite_spec).stem if Path(suite_spec).is_file() else suite_spec
	ns = argparse.Namespace(
		root=str(args.root),
		sensing=args.sensing,
		camera=args.camera,
		strategy=args.strategy,
		detector=args.detector,
		policy=args.policy,
		max_scored_views=args.max_scored_views,
		boundary=args.boundary,
		batch=batch_name,
		engine=args.engine,
		suite=suite_name,
		workers=args.workers,
		limit=None,
		freeze=args.freeze,
		review_llm_url=args.review_llm_url,
		review_model=args.review_model,
		min_free_gib=args.min_free_gib,
		est_mib=args.est_mib,
	)
	rc = runner.cmd_run(ns)
	if rc != 0:
		raise SystemExit(f"runner failed with exit code {rc} for suite={suite_name!r} batch={batch_name!r}")


def _pct(val: float | None) -> str:
	if val is None or (isinstance(val, float) and math.isnan(val)):
		return "-"
	return f"{100.0 * val:.2f}%"


def _signed_pct(val: float | None) -> str:
	if val is None or (isinstance(val, float) and math.isnan(val)):
		return "-"
	return f"{100.0 * val:+.2f} pp"


def _num(val: float | None, digits: int = 4) -> str:
	if val is None or (isinstance(val, float) and math.isnan(val)):
		return "-"
	return f"{val:.{digits}f}"


def render_verify_md(report: dict) -> str:
	g1 = report["gates"]["bank"]
	g2 = report["gates"]["regression"]
	g3 = report["gates"]["fresh"]
	verdict = report["verdict"]

	lines = [
		f"# Fix Verification: {report['cluster']}",
		"",
		f"- **Verdict**: `{verdict}`",
		f"- **Base batch**: `{report['base_batch']}`",
		f"- **New batch**: `{report['new_batch']}`",
		f"- **Created**: `{report['created']}`",
		"",
		"## Three-Gate Summary",
		"",
		"| Gate | Suite | Metric / Check | Observed | Threshold | Result |",
		"|---|---|---|---|---|---|",
		(
			f"| Gate 1: Failure Bank | `{report['suites']['bank']}` | Fix rate | "
			f"{g1['fixed']}/{g1['total_base_failures']} ({_pct(g1['fix_rate'])}) | "
			f">= {_pct(g1['min_fix_rate'])} | {'PASS' if g1['passed'] else 'FAIL'} |"
		),
		(
			f"| Gate 2: Regression | `{report['suites']['regression']}` | Net regression / McNemar p | "
			f"n10={g2['n10']}, n01={g2['n01']}, net={g2['net_regression']}, p={_num(g2['p_mcnemar'])} | "
			f"net <= {g2['max_net_regression']}, p > {_num(g2['min_mcnemar_p'], 2)} | "
			f"{'PASS' if g2['passed'] else 'FAIL'} |"
		),
		(
			f"| Gate 3: Fresh Seeds | `{report['suites']['fresh']}` | Paired diff & 95% CI | "
			f"{_signed_pct(g3['diff'])} [{_signed_pct(g3['ci_lo'])}, {_signed_pct(g3['ci_hi'])}] | "
			f"ci_lo >= {_signed_pct(g3['min_ci_lo'])} | {'PASS' if g3['passed'] else 'FAIL'} |"
		),
		"",
		"## Gate 1: Failure Bank Fix Rate",
		"",
		f"- Base rows: {g1['n_base_rows']} (base failures: {g1['total_base_failures']})",
		f"- Fixed: {g1['fixed']} / {g1['total_base_failures']} ({_pct(g1['fix_rate'])})",
		(
			f"- Still failing: {g1['still_failing']}, missing in new: {g1['missing_in_new']}, "
			f"regressed in bank: {g1['regressed_in_bank']}"
		),
		"",
		"## Gate 2: Regression Suite (Exact McNemar)",
		"",
		f"- Paired cases: {g2['n_pairs']} across {g2['n_clusters']} layout clusters",
		(
			f"- Discordant pairs: n10 (new failures) = {g2['n10']}, n01 (new successes) = {g2['n01']}, "
			f"net = {g2['net_regression']}"
		),
		f"- Concordant pairs: n11 = {g2['n11']}, n00 = {g2['n00']}",
		f"- Exact McNemar p-value: {_num(g2['p_mcnemar'], 6)}",
		"",
		"## Gate 3: Fresh Seeds (Layout-Clustered Bootstrap)",
		"",
		f"- Paired cases: {g3['n_pairs']} across {g3['n_clusters']} layout clusters (B={g3['bootstrap_b']})",
		f"- Success rate: base {_pct(g3['base_rate'])} -> new {_pct(g3['new_rate'])}",
		f"- Paired delta: {_signed_pct(g3['diff'])}, 95% CI [{_signed_pct(g3['ci_lo'])}, {_signed_pct(g3['ci_hi'])}]",
		"",
	]
	return "\n".join(lines)


def _json_safe(obj):
	if isinstance(obj, float) and not math.isfinite(obj):
		return None
	if isinstance(obj, dict):
		return {k: _json_safe(v) for k, v in obj.items()}
	if isinstance(obj, list):
		return [_json_safe(v) for v in obj]
	return obj


def _build_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	p.add_argument("--bank", required=True, help="Failure bank suite name or JSON path")
	p.add_argument("--regression", required=True, help="Regression suite name or JSON path")
	p.add_argument("--fresh", required=True, help="Fresh-seeds suite name or JSON path")
	p.add_argument("--base", required=True, help="Baseline batch name (read-only)")
	p.add_argument("--batch", required=True, help="New fix batch name")
	p.add_argument("--criteria", required=True, help="Acceptance criteria JSON path")
	p.add_argument("--root", default=str(DEFAULT_ROOT), help="Data root for new batch & outputs")
	p.add_argument("--base-root", help="Optional read-only data root for --base (default: --root)")
	p.add_argument("--out-dir", help="Output directory (default: <root>/reports/<batch>/)")
	p.add_argument("--cluster", help="Cluster label for verify-<cluster>.{json,md}")
	p.add_argument("--skip-run", action="store_true", help="Do not invoke runner; score existing runs/reports only")
	p.add_argument("--engine", choices=ENGINES, help="Optional engine filter (and runner engine, default mujoco)")
	p.add_argument("--group", help="Optional group filter for --batch (and --base unless --base-group is set)")
	p.add_argument("--base-group", help="Optional group filter for --base")
	p.add_argument("--sensing", choices=SENSING, default="realistic")
	p.add_argument("--camera", choices=CAMERA_CONDITIONS, default="fixed")
	p.add_argument("--strategy", choices=STRATEGIES, default="A")
	p.add_argument("--detector", choices=DETECTORS, default="v7bot")
	p.add_argument("--workers", type=int, default=1)
	p.add_argument("--policy", help="Policy JSON path for runner")
	p.add_argument("--max-scored-views", type=int)
	p.add_argument("--boundary", choices=("belief_cell", "legacy"), default="belief_cell")
	p.add_argument("--review-llm-url")
	p.add_argument("--review-model", default="qwen3.8-vllm")
	p.add_argument("--freeze", help="Freeze JSON path if running a sealed suite")
	p.add_argument("--min-free-gib", type=float, default=8.0)
	p.add_argument("--est-mib", type=float, default=25.0)
	p.add_argument("--bootstrap", type=int, help="Override bootstrap iterations B")
	p.add_argument("--seed", type=int, help="Override bootstrap RNG seed")
	return p


def main(argv: list[str] | None = None) -> int:
	parser = _build_parser()
	args = parser.parse_args(argv)
	if args.engine is None:
		args.engine = "mujoco"
		engine_filter = None
	else:
		engine_filter = args.engine

	root = Path(args.root)
	base_root = Path(args.base_root) if args.base_root else root
	out_dir = Path(args.out_dir) if args.out_dir else root / "reports" / args.batch
	criteria = load_criteria(Path(args.criteria))
	if args.bootstrap is not None:
		criteria["bootstrap_b"] = args.bootstrap
	if args.seed is not None:
		criteria["seed"] = args.seed

	cluster_label = args.cluster or criteria.get("cluster") or Path(args.bank).stem
	slug = cluster_slug(cluster_label)

	roots = [root] if base_root == root else [root, base_root]
	batches = (args.batch, args.base)
	bank_ids = resolve_suite_case_ids(args.bank, roots=roots, batches=batches)
	reg_ids = resolve_suite_case_ids(args.regression, roots=roots, batches=batches)
	fresh_ids = resolve_suite_case_ids(args.fresh, roots=roots, batches=batches)

	new_group = args.group
	base_group = args.base_group or args.group

	if not args.skip_run:
		for role, suite_spec, cids in (
			("bank", args.bank, bank_ids),
			("regression", args.regression, reg_ids),
			("fresh", args.fresh, fresh_ids),
		):
			existing = _select_batch_rows(
				root,
				args.batch,
				role,
				case_ids=cids,
				engine=engine_filter,
				group=new_group,
			)
			if not _suite_ready(existing, cids):
				run_suite_batch(args, suite_spec, f"{args.batch}-{role}")

	role_pairs: dict[str, tuple[list[dict], list[dict]]] = {}
	for role, cids in (("bank", bank_ids), ("regression", reg_ids), ("fresh", fresh_ids)):
		b_rows = _select_batch_rows(
			base_root,
			args.base,
			role,
			case_ids=cids,
			engine=engine_filter,
			group=base_group,
		)
		n_rows = _select_batch_rows(
			root,
			args.batch,
			role,
			case_ids=cids,
			engine=engine_filter,
			group=new_group,
		)
		b_rows, n_rows = _disambiguate_pair(
			b_rows,
			n_rows,
			role=role,
			base_group=base_group,
			new_group=new_group,
		)
		role_pairs[role] = (b_rows, n_rows)

	result = evaluate_three_gates(
		role_pairs["bank"][0],
		role_pairs["bank"][1],
		role_pairs["regression"][0],
		role_pairs["regression"][1],
		role_pairs["fresh"][0],
		role_pairs["fresh"][1],
		criteria,
	)

	payload = {
		"cluster": cluster_label,
		"cluster_slug": slug,
		"base_batch": args.base,
		"new_batch": args.batch,
		"suites": {
			"bank": args.bank,
			"regression": args.regression,
			"fresh": args.fresh,
		},
		"created": datetime.now(UTC).isoformat(timespec="seconds"),
		**result,
	}

	out_dir.mkdir(parents=True, exist_ok=True)
	json_path = out_dir / f"verify-{slug}.json"
	md_path = out_dir / f"verify-{slug}.md"
	json_path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
	md_path.write_text(render_verify_md(payload), encoding="utf-8")
	print(f"{md_path}: verdict={payload['verdict']}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
