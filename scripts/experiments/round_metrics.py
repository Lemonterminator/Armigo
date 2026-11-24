"""Compute per-round harness learning metrics and append a summary row to ``rounds.md`` (Task H7).

Usage:
	python scripts/experiments/round_metrics.py \\
		--round-dir E:/Armigo_data_v3/loop/round-00 \\
		--bench config/agent_bench/diagnosis_bench/v1.json \\
		--scores E:/Armigo_data_v3/reports/p2-dev-r2/scores.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path


def _bootstrap_path() -> None:
	repo = os.environ.get("ARMIGO_REPO")
	if repo:
		repo_src = Path(repo) / "src"
		if repo_src.is_dir() and str(repo_src) not in sys.path:
			sys.path.insert(0, str(repo_src))
	src = Path(__file__).resolve().parents[2] / "src"
	if src.is_dir():
		if str(src) in sys.path:
			sys.path.remove(str(src))
		sys.path.insert(0, str(src))


_bootstrap_path()

from digital_twin_armigo.experiments.diagnosis import agreement
from digital_twin_armigo.experiments.stats import cell_metrics, metric_with_ci
from digital_twin_armigo.inventory import query

ROUNDS_HEADER = (
	"| Round | Success (k/n, 95% CI) | Unclassified | Mechanism | CF Confirmed | "
	"Calib Acc (k/n) | Heldout Acc (k/n) | Fix Yield (M/O/R) | Regressions n10 | Diag Cost (calls / s / first) |\n"
	"|---|---|---|---|---|---|---|---|---|---|\n"
)


def _json_safe(obj):
	if isinstance(obj, float) and not math.isfinite(obj):
		return None
	if isinstance(obj, dict):
		return {k: _json_safe(v) for k, v in obj.items()}
	if isinstance(obj, list):
		return [_json_safe(v) for v in obj]
	return obj


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


def _norm_path(path: str | Path) -> str:
	return str(Path(path).resolve()).lower()


def _load_bench(path: Path) -> list[dict]:
	raw = _read_json(path)
	if isinstance(raw, list):
		return raw
	if isinstance(raw, dict):
		for key in ("entries", "items", "bench"):
			if isinstance(raw.get(key), list):
				return raw[key]
	raise TypeError(f"benchmark JSON must be a list of entries: {path}")


def _task_metrics(score_rows: list[dict]) -> dict:
	sr = metric_with_ci(score_rows, "success_rate")
	cm = cell_metrics(score_rows)
	return _json_safe(
		{
			"n": sr["n"],
			"k": sr["k"],
			"success_rate": sr["value"],
			"ci_lo": sr["ci_lo"],
			"ci_hi": sr["ci_hi"],
			"n_clusters": sr["n_clusters"],
			"cell_metrics": cm,
		}
	)


def _attribution_coverage(score_rows: list[dict], counterfactual_path: Path | None) -> dict:
	failed = [r for r in score_rows if r.get("root_cause") != "none"]
	n_failed = len(failed)
	other_n = sum(1 for r in failed if r.get("root_cause") in {"other", "unclassified"})
	mech_n = sum(1 for r in failed if r.get("mechanism") not in (None, ""))

	cf_n: int | None = None
	cf_rate: float | None = None
	if counterfactual_path is not None:
		cf_rows = _read_jsonl(counterfactual_path)
		cf_by_case = {r["case_id"]: r for r in cf_rows if r.get("case_id")}
		cf_by_traj = {r["traj_id"]: r for r in cf_rows if r.get("traj_id")}
		cf_n = 0
		for r in failed:
			cf_row = cf_by_case.get(r.get("case_id")) or cf_by_traj.get(r.get("traj_id"))
			if cf_row and cf_row.get("verdict") == "confirmed":
				cf_n += 1
		cf_rate = cf_n / n_failed if n_failed else None

	return {
		"n_failed": n_failed,
		"other_or_unclassified_n": other_n,
		"other_or_unclassified_rate": other_n / n_failed if n_failed else None,
		"with_mechanism_n": mech_n,
		"with_mechanism_rate": mech_n / n_failed if n_failed else None,
		"counterfactual_confirmed_n": cf_n,
		"counterfactual_confirmed_rate": cf_rate,
	}


def _build_auto_map(
	bench: list[dict],
	score_rows: list[dict],
	inventory_dir: Path | None,
) -> dict[str, dict]:
	auto: dict[str, dict] = {}
	inv_paths: dict[str, Path] = {}

	if inventory_dir is not None:
		with query.connect(inventory_dir) as con:
			cols = {r["name"] for r in con.execute("pragma table_info(trajectories)").fetchall()}
			has_auto = "auto_root_cause" in cols and "auto_mechanism" in cols
			select_cols = "traj_id, path, auto_root_cause, auto_mechanism" if has_auto else "traj_id, path"
			for row in con.execute(f"select {select_cols} from trajectories where record_kind = 'agent_episode'"):
				tid = row["traj_id"]
				if row["path"]:
					inv_paths[tid] = Path(row["path"])
				if has_auto and row["auto_root_cause"] is not None:
					auto[tid] = {
						"root_cause": row["auto_root_cause"],
						"mechanism": row["auto_mechanism"],
					}

	scores_by_path: dict[str, dict] = {}
	for r in score_rows:
		if r.get("path"):
			scores_by_path[_norm_path(r["path"])] = {
				"root_cause": r.get("root_cause"),
				"mechanism": r.get("mechanism"),
			}

	for b in bench:
		tid = b["traj_id"]
		ep_path = b.get("path") or inv_paths.get(tid)
		if ep_path is None:
			continue
		matched = scores_by_path.get(_norm_path(ep_path))
		if matched is not None:
			auto[tid] = matched

	return auto


def _attribution_accuracy(bench: list[dict], auto_by_traj_id: dict[str, dict]) -> dict:
	return _json_safe(
		{
			"calib": agreement(bench, auto_by_traj_id, "calib"),
			"heldout": agreement(bench, auto_by_traj_id, "heldout"),
		}
	)


def _fix_and_regressions(verify_dir: Path) -> tuple[dict, dict]:
	files = sorted(verify_dir.glob("verify-*.json")) if verify_dir.is_dir() else []
	counts = {"merge": 0, "overfit_bank": 0, "reject": 0}
	n10_sum = 0
	for path in files:
		doc = _read_json(path)
		verdict = doc.get("verdict")
		if verdict in counts:
			counts[verdict] += 1
		reg = (doc.get("gates") or {}).get("regression") or {}
		n10_sum += int(reg.get("n10") or 0)
	fix_yield = {"total": len(files), **counts}
	regressions = {"n10_sum": n10_sum, "n_verify_files": len(files)}
	return fix_yield, regressions


def _session_latency_s(s_rows: list[dict]) -> float:
	lat_vals = [float(r["latency_s"]) for r in s_rows if r.get("latency_s") is not None]
	if lat_vals:
		return sum(lat_vals)
	wall_vals = [float(r["wall_time_s"]) for r in s_rows if r.get("wall_time_s") is not None]
	if wall_vals:
		return sum(wall_vals)
	return 0.0


def _diagnosis_cost(diag_log_path: Path | None, bench: list[dict]) -> dict | None:
	if diag_log_path is None:
		return None
	diag_rows = _read_jsonl(diag_log_path)
	bench_by_tid = {b["traj_id"]: b for b in bench}
	sessions: dict[str, list[dict]] = defaultdict(list)
	for row in diag_rows:
		sid = str(row.get("session") or "")
		sessions[sid].append(row)

	call_counts: list[int] = []
	latencies: list[float] = []
	first_correct_calls: list[int] = []

	for s_rows in sessions.values():
		call_counts.append(len(s_rows))
		latencies.append(_session_latency_s(s_rows))
		for idx, row in enumerate(s_rows, start=1):
			if row.get("correct") is True:
				first_correct_calls.append(idx)
				break
			rc = row.get("root_cause")
			tid = row.get("traj_id") or (row.get("args") or {}).get("traj_id")
			if rc and tid in bench_by_tid and rc == bench_by_tid[tid].get("root_cause"):
				first_correct_calls.append(idx)
				break

	return {
		"n_sessions": len(sessions),
		"n_calls": len(diag_rows),
		"median_calls_per_session": float(statistics.median(call_counts)) if call_counts else None,
		"median_latency_s_per_session": float(statistics.median(latencies)) if latencies else None,
		"calls_to_first_correct": float(statistics.median(first_correct_calls)) if first_correct_calls else None,
	}


def _pct(val: float | None) -> str:
	if val is None:
		return "-"
	return f"{100.0 * val:.1f}%"


def _num(val: float | None, digits: int = 2) -> str:
	if val is None:
		return "-"
	return f"{val:.{digits}f}"


def format_rounds_row(metrics: dict) -> str:
	task = metrics["task"]
	cov = metrics["attribution_coverage"]
	acc = metrics["attribution_accuracy"]
	fy = metrics["fix_yield"]
	reg = metrics["regressions"]
	dc = metrics["diagnosis_cost"]

	success_str = (
		f"{_pct(task['success_rate'])} ({task['k']}/{task['n']}, [{_pct(task['ci_lo'])}, {_pct(task['ci_hi'])}])"
	)
	unclass_str = _pct(cov["other_or_unclassified_rate"])
	mech_str = _pct(cov["with_mechanism_rate"])
	cf_str = _pct(cov["counterfactual_confirmed_rate"])

	def _acc_str(split_res: dict) -> str:
		if not split_res["n"]:
			return "0/0 (-)"
		return f"{_pct(split_res['root_cause_acc'])} (n={split_res['n']}, k={_num(split_res['root_cause_kappa'])})"

	fix_str = f"{fy['merge']}/{fy['overfit_bank']}/{fy['reject']}"
	reg_str = str(reg["n10_sum"])
	if dc is None:
		diag_str = "-"
	else:
		diag_str = (
			f"{_num(dc['median_calls_per_session'], 1)} / "
			f"{_num(dc['median_latency_s_per_session'], 2)}s / "
			f"{_num(dc['calls_to_first_correct'], 1)}"
		)

	return (
		f"| {metrics['round']} | {success_str} | {unclass_str} | {mech_str} | {cf_str} | "
		f"{_acc_str(acc['calib'])} | {_acc_str(acc['heldout'])} | {fix_str} | {reg_str} | {diag_str} |\n"
	)


def compute_round_metrics(
	*,
	round_dir: Path,
	bench_path: Path,
	scores_path: Path,
	inventory_dir: Path | None = None,
	verify_dir: Path | None = None,
	counterfactual_path: Path | None = None,
	diag_log_path: Path | None = None,
) -> dict:
	round_dir = Path(round_dir)
	bench = _load_bench(bench_path)
	score_rows = _read_jsonl(scores_path)
	auto_by_tid = _build_auto_map(bench, score_rows, inventory_dir)
	fix_yield, regressions = _fix_and_regressions(verify_dir or round_dir)

	return {
		"round": round_dir.name,
		"created": datetime.now(UTC).isoformat(timespec="seconds"),
		"bench_path": str(bench_path),
		"scores_path": str(scores_path),
		"task": _task_metrics(score_rows),
		"attribution_coverage": _attribution_coverage(score_rows, counterfactual_path),
		"attribution_accuracy": _attribution_accuracy(bench, auto_by_tid),
		"fix_yield": fix_yield,
		"regressions": regressions,
		"diagnosis_cost": _diagnosis_cost(diag_log_path, bench),
	}


def _append_rounds_md(rounds_md: Path, row: str) -> None:
	if not rounds_md.is_file() or rounds_md.stat().st_size == 0:
		rounds_md.write_text("# Harness Learning Rounds\n\n" + ROUNDS_HEADER + row, encoding="utf-8")
		return
	existing = rounds_md.read_text(encoding="utf-8")
	prefix = "" if existing.endswith("\n") else "\n"
	rounds_md.write_text(existing + prefix + row, encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	p.add_argument("--round-dir", required=True, help="Round directory (writes <round-dir>/round_metrics.json)")
	p.add_argument("--bench", required=True, help="Diagnosis benchmark JSON path")
	p.add_argument("--scores", required=True, help="Round scores.jsonl path")
	p.add_argument("--inventory", help="Optional inventory directory for auto_* attribution lookup")
	p.add_argument("--verify-dir", help="Optional directory containing verify-*.json (default: --round-dir)")
	p.add_argument("--counterfactual", help="Optional counterfactual.jsonl path")
	p.add_argument("--diag-log", help="Optional diag_calls.jsonl path")
	return p


def main(argv: list[str] | None = None) -> int:
	args = _build_parser().parse_args(argv)
	round_dir = Path(args.round_dir)
	round_dir.mkdir(parents=True, exist_ok=True)

	metrics = compute_round_metrics(
		round_dir=round_dir,
		bench_path=Path(args.bench),
		scores_path=Path(args.scores),
		inventory_dir=Path(args.inventory) if args.inventory else None,
		verify_dir=Path(args.verify_dir) if args.verify_dir else None,
		counterfactual_path=Path(args.counterfactual) if args.counterfactual else None,
		diag_log_path=Path(args.diag_log) if args.diag_log else None,
	)

	out_json = round_dir / "round_metrics.json"
	out_json.write_text(json.dumps(_json_safe(metrics), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
	rounds_md = round_dir.parent / "rounds.md"
	_append_rounds_md(rounds_md, format_rounds_row(metrics))
	print(f"{out_json} (updated {rounds_md})")
	return 0


if __name__ == "__main__":
	sys.exit(main())
