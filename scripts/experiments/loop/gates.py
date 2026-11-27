"""Three gates for one harness-learning round, driven by a round spec in the repo.

	python gates.py --round config/experiments/loop/round_03.json            # rescore the round's batches, then gate
	python gates.py --round ... --from-reports                               # gate the existing reports/<batch>/scores.jsonl
	python gates.py --round ... --write-bank                                 # also snapshot bank + regression ids into config/
	LOOP_CONFIG=round3_config python gates.py                                # spec from the config module's ROUND_SPEC

PYTHONPATH = candidate checkout src + scripts/experiments (one scorer for both arms).

The round spec (JSON) replaces the per-round gates<N>.py scripts and the criteria.json that lived only in the data root:

- ``tag``: round tag; outputs go to ``<root>/reports/<tag>/``.
- ``root``: data root (override with ``--root`` or ``ARMIGO_LOOP_ROOT``). Any ``base`` source may name its own ``root``.
- ``conditions``: camera conditions to keep (rows outside them are dropped from every arm).
- ``base``: batches (or ``{"batch", "root"}``) whose report rows are the pre-fix state of the bank and regression cases.
- ``rerun``: batches with the same cases on the candidate. ``fresh_base`` / ``fresh_candidate``: the new-seed pair.
- ``bank``: how the failure bank is chosen from the base rows, one of
	``{"ids": [...]}``, ``{"root_causes": [...]}``, or ``{"ledger": <ledger.json>, "cluster": <key>}``
	(the ledger route keeps base failures whose traj_id the cluster holds, via failure_bank.collect_cluster_traj_ids).
- ``criteria``: thresholds for verify_fix.evaluate_three_gates plus optional expected counts (``bank_records``,
	``regression_pairs``, ``rerun_pairs``, ``fresh_pairs``) checked before gating, and free-form provenance.
- ``out``: verify file stem (``verify-<cluster>``); JSON and Markdown are written.

Pairs are keyed on ``<camera>/<case_id>`` and the bootstrap clusters by layout. With more than one condition the pooled
gates are joined by per-condition gates. Regression = base physical successes, bank excluded.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))  # scripts/experiments: verify_fix, failure_bank
from verify_fix import evaluate_three_gates

from digital_twin_armigo.experiments.stats import paired_comparison

GRASPISH = ("grasp_or_path_failure", "recovery_failure")


def rows_of(path: Path) -> list[dict]:
	return [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip()]


def load_spec(path: Path) -> dict:
	spec = json.loads(Path(path).read_text(encoding="utf-8"))
	missing = [
		k
		for k in ("tag", "conditions", "base", "rerun", "fresh_base", "fresh_candidate", "bank", "criteria")
		if k not in spec
	]
	if missing:
		raise SystemExit(f"{path}: round spec lacks {missing}")
	return spec


def _source(item, root: Path) -> tuple[str, Path]:
	if isinstance(item, str):
		return item, root
	return item["batch"], Path(item["root"]) if item.get("root") else root


def keyed(rows: list[dict], conditions) -> list[dict]:
	"""Keep ``conditions``, key on ``<camera>/<case_id>``, pool the groups; incomplete rows are an error."""
	out = []
	for r in rows:
		if r.get("camera_condition") not in conditions:
			continue
		if not r.get("has_episode") or r.get("infra_failed"):
			raise RuntimeError(f"incomplete row {r.get('path')}")
		out.append(
			{
				**r,
				"case_id": f"{r['camera_condition']}/{r['case_id']}",
				"orig_case_id": r["case_id"],
				"orig_group": r["group"],
				"group": "pooled",
			}
		)
	keys = [r["case_id"] for r in out]
	dup = [k for k, n in Counter(keys).items() if n > 1]
	if dup:
		raise RuntimeError(f"{len(dup)} duplicate pair keys, e.g. {dup[:5]}")
	return out


class Loader:
	"""Report rows per batch, either rescored from runs/ (default) or read from reports/<batch>/scores.jsonl."""

	def __init__(self, root: Path, *, from_reports: bool, dry: bool, jobs: int):
		self.root, self.from_reports, self.dry, self.jobs = root, from_reports, dry, jobs

	def prepare(self, batches: list[str]) -> None:
		if self.from_reports:
			return
		check_complete(self.root, batches)
		prescore(self.root, batches, self.jobs)

	def rows(self, batch: str, root: Path | None = None) -> list[dict]:
		root = root or self.root
		if self.from_reports or root != self.root:
			return rows_of(root / "reports" / batch / "scores.jsonl")
		from digital_twin_armigo.experiments import report

		if self.dry:  # through JSON exactly as scores.jsonl would carry the rows
			return [json.loads(json.dumps(r, ensure_ascii=False)) for r in report.scan_batch(root / "runs" / batch)[0]]
		return rows_of(Path(report.write_report(root, batch)) / "scores.jsonl")


def check_complete(root: Path, batches: list[str]) -> None:
	"""Every case in every plan.json has a latest attempt in state done; exit naming the gaps otherwise."""
	from digital_twin_armigo.experiments import runner

	gaps, n = [], 0
	for batch in batches:
		plans = sorted((root / "runs" / batch).glob("*/*/plan.json"))
		if not plans:
			gaps.append(f"{batch}: no plan.json")
		for plan in plans:
			for cid in json.loads(plan.read_text(encoding="utf-8"))["cases"]:
				n += 1
				att = runner._attempts(plan.parent, cid)
				if not att or att[-1][2].get("state") != "done":
					gaps.append(f"{batch}/{plan.parent.name}/{cid}: {att[-1][2].get('state') if att else 'not run'}")
	if gaps:
		raise SystemExit(f"{len(gaps)} of {n} planned episodes not done, e.g. {gaps[:8]}; rerun the driver to resume")
	print(f"complete: {n} planned episodes done", flush=True)


def prescore(root: Path, batches: list[str], jobs: int) -> None:
	"""Score every episode dir of ``batches`` on a process pool; report.scan_batch then reads the cache."""
	from concurrent.futures import ProcessPoolExecutor

	from digital_twin_armigo.experiments import report
	from digital_twin_armigo.experiments.scoring import score_episode

	dirs = [d for b in batches for d in report._episode_dirs(root / "runs" / b)]
	with ProcessPoolExecutor(max_workers=jobs) as pool:
		scored = dict(zip(dirs, pool.map(score_episode, dirs, chunksize=8), strict=True))
	print(f"scored {len(dirs)} episode dirs with {jobs} processes", flush=True)
	report.score_episode = lambda d: dict(scored[d]) if d in scored else score_episode(d)


def select_bank(rule: dict, base: list[dict]) -> set[str]:
	"""Bank pair keys from the base rows by the spec's rule (always base failures)."""
	failed = [r for r in base if not r["physical_success"]]
	if "ids" in rule:
		ids = set(rule["ids"])
		unknown = ids - {r["case_id"] for r in base}
		if unknown:
			raise SystemExit(f"bank ids not among the base rows: {sorted(unknown)[:8]}")
		return ids
	if "root_causes" in rule:
		causes = set(rule["root_causes"])
		return {r["case_id"] for r in failed if r["root_cause"] in causes}
	if "ledger" in rule:
		from failure_bank import collect_cluster_traj_ids, match_cluster

		ledger_path = Path(rule["ledger"])
		cluster = match_cluster(json.loads(ledger_path.read_text(encoding="utf-8")), str(rule["cluster"]))
		tids = set(collect_cluster_traj_ids(ledger_path, cluster))
		return {r["case_id"] for r in failed if r.get("traj_id") in tids}
	raise SystemExit(f"bank rule needs ids, root_causes or ledger: {rule}")


def _check_counts(crit: dict, **seen: int) -> None:
	bad = {k: (crit[k], v) for k, v in seen.items() if k in crit and crit[k] != v}
	if bad:
		raise SystemExit("counts differ from the round spec (expected, seen): " + json.dumps(bad))


def run_gates(spec: dict, loader: Loader) -> dict:
	root, conds, crit = loader.root, tuple(spec["conditions"]), spec["criteria"]
	tag = spec["tag"]
	loader.prepare([*spec["rerun"], spec["fresh_base"], spec["fresh_candidate"]])
	base = keyed([r for s in spec["base"] for r in loader.rows(*_source(s, root))], conds)
	rerun = keyed([r for b in spec["rerun"] for r in loader.rows(b)], conds)
	fresh_b = keyed(loader.rows(spec["fresh_base"]), conds)
	fresh_c = keyed(loader.rows(spec["fresh_candidate"]), conds)
	if {r["case_id"] for r in base} != {r["case_id"] for r in rerun}:
		raise SystemExit("base and rerun rows cover different pair keys")
	if {r["case_id"] for r in fresh_b} != {r["case_id"] for r in fresh_c}:
		raise SystemExit("fresh base and candidate cover different pair keys")
	bank_ids = select_bank(spec["bank"], base)
	reg_ids = {r["case_id"] for r in base if r["physical_success"]} - bank_ids
	_check_counts(
		crit,
		bank_records=len(bank_ids),
		regression_pairs=len(reg_ids),
		rerun_pairs=len(rerun),
		fresh_pairs=len(fresh_b),
	)
	print(
		f"{tag}: bank {len(bank_ids)}, regression {len(reg_ids)}, reruns {len(rerun)}, fresh {len(fresh_b)}", flush=True
	)

	def gate(cond: str | None) -> dict:
		sel = (lambda rows: rows) if cond is None else (lambda rows: [r for r in rows if r["camera_condition"] == cond])
		bi = {i for i in bank_ids if cond is None or i.startswith(cond + "/")}
		ri = {i for i in reg_ids if cond is None or i.startswith(cond + "/")}
		pick = lambda rows, ids: [r for r in rows if r["case_id"] in ids]
		return evaluate_three_gates(
			pick(base, bi),
			pick(rerun, bi),
			pick(base, ri),
			pick(rerun, ri),
			sel(fresh_b),
			sel(fresh_c),
			crit,
			expected_bank_ids=bi,
			expected_reg_ids=ri,
			expected_fresh_ids={r["case_id"] for r in sel(fresh_b)},
			allow_group_mismatch=True,
		)

	result = gate(None)
	bmap = {r["case_id"]: r for r in base}
	fb = {r["case_id"]: r for r in fresh_b}
	fail = lambda rows: Counter(r["root_cause"] for r in rows if not r["physical_success"])
	extra: dict = {
		"all_reruns": paired_comparison(base, rerun, B=2000, seed=0),
		"bank_cases": [
			{"case": r["case_id"], "new_success": r["physical_success"], "new_root_cause": r["root_cause"]}
			for r in sorted((r for r in rerun if r["case_id"] in bank_ids), key=lambda r: r["case_id"])
		],
		"regressions": [
			{"case": r["case_id"], "root_cause": r["root_cause"], "mechanism": r.get("mechanism"), "path": r["path"]}
			for r in rerun
			if r["case_id"] in reg_ids and not r["physical_success"]
		],
		"rerun_changed_outcome": [
			{"case": r["case_id"], "base": bmap[r["case_id"]]["physical_success"], "candidate": r["physical_success"]}
			for r in rerun
			if bmap[r["case_id"]]["physical_success"] != r["physical_success"]
		],
		"fresh_discordant": [
			{
				"case": r["case_id"],
				"base": fb[r["case_id"]]["physical_success"],
				"candidate": r["physical_success"],
				"base_root_cause": fb[r["case_id"]]["root_cause"],
				"candidate_root_cause": r["root_cause"],
			}
			for r in fresh_c
			if fb[r["case_id"]]["physical_success"] != r["physical_success"]
		],
		"root_causes": {
			"base0": fail(base),
			"rerun": fail(rerun),
			"fresh_base": fail(fresh_b),
			"fresh_candidate": fail(fresh_c),
		},
		"fresh_grasp_recovery": {
			"base": sum(r["root_cause"] in GRASPISH for r in fresh_b),
			"candidate": sum(r["root_cause"] in GRASPISH for r in fresh_c),
		},
	}
	if len(conds) > 1:
		extra["gates_by_condition"] = {c: gate(c) for c in conds}
		extra["by_condition"] = {
			c: {
				"reruns": paired_comparison(
					*([r for r in x if r["camera_condition"] == c] for x in (base, rerun)), B=2000, seed=0
				),
				"fresh": paired_comparison(
					*([r for r in x if r["camera_condition"] == c] for x in (fresh_b, fresh_c)), B=2000, seed=0
				),
			}
			for c in conds
		}
	costs = {}
	for name, x, y in (("rerun", base, rerun), ("fresh", fresh_b, fresh_c)):
		xm = {r["case_id"]: r for r in x}
		f = [r["sim_frames"] - xm[r["case_id"]]["sim_frames"] for r in y]
		t = [r["wall_s"] - xm[r["case_id"]]["wall_s"] for r in y]
		costs[name] = {
			"frames_mean": mean(f),
			"frames_median": median(f),
			"wall_mean": mean(t),
			"wall_median": median(t),
			"changed_frames": sum(v != 0 for v in f),
		}
	extra["costs"] = costs
	return {"gates": result, **extra, "_bank_ids": sorted(bank_ids), "_reg_ids": sorted(reg_ids)}


def _pct(x) -> str:
	return "n/a" if x is None else f"{100 * x:.1f}%"


def render_md(spec: dict, doc: dict) -> str:
	g = doc["gates"]["gates"]
	g1, g2, g3 = g["bank"], g["regression"], g["fresh"]
	ok = lambda d: "PASS" if d["passed"] else "FAIL"
	lines = [
		f"# {spec['tag']}: {spec['criteria'].get('cluster', spec.get('out', ''))}",
		"",
		f"Verdict: **{doc['gates']['verdict']}**. Conditions: {', '.join(spec['conditions'])}.",
		"",
		"| Gate | Observed | Threshold | Result |",
		"|---|---|---|---|",
		(
			f"| 1 failure bank | {g1.get('fixed')}/{g1.get('total_base_failures')} fixed ({_pct(g1.get('fix_rate'))}) "
			f"| >= {_pct(g1.get('min_fix_rate'))} | {ok(g1)} |"
		),
		(
			f"| 2 regression | n10={g2.get('n10')}, n01={g2.get('n01')}, net={g2.get('net_regression')}, p={g2.get('p_mcnemar')} "
			f"| net <= {g2.get('max_net_regression')}, p > {g2.get('min_mcnemar_p')} | {ok(g2)} |"
		),
		(
			f"| 3 new seeds | {_pct(g3.get('base_rate'))} -> {_pct(g3.get('new_rate'))}, 95% CI lo {_pct(g3.get('ci_lo'))} "
			f"| >= {_pct(g3.get('min_ci_lo'))} | {ok(g3)} |"
		),
	]
	for c, gc in (doc.get("gates_by_condition") or {}).items():
		lines.append(f"\n- {c}: {gc['verdict']}")
	if doc["regressions"]:
		lines += ["", "Regressions:", *(f"- `{r['case']}` {r['root_cause']}" for r in doc["regressions"])]
	return "\n".join(lines) + "\n"


def write_bank(spec: dict, doc: dict, path: Path) -> None:
	snap = {
		"tag": spec["tag"],
		"bank_rule": spec["bank"],
		"base": spec["base"],
		"conditions": spec["conditions"],
		"bank_ids": doc["_bank_ids"],
		"regression_ids": doc["_reg_ids"],
	}
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(snap, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
	print(f"bank snapshot: {path}", flush=True)


def _spec_path(arg: str | None) -> Path:
	if arg:
		return Path(arg) if Path(arg).is_absolute() else REPO / arg
	mod = os.environ.get("LOOP_CONFIG")
	if mod:
		sys.path.insert(0, str(HERE))
		spec = getattr(importlib.import_module(mod), "ROUND_SPEC", None)
		if spec:
			return Path(spec)
	raise SystemExit("give --round <spec.json> or LOOP_CONFIG naming a module with ROUND_SPEC")


def main(argv: list[str] | None = None) -> int:
	p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
	p.add_argument("--round", help="round spec JSON (repo-relative or absolute)")
	p.add_argument("--root", help="data root (default: ARMIGO_LOOP_ROOT, then the spec's root)")
	p.add_argument("--from-reports", action="store_true", help="read reports/<batch>/scores.jsonl instead of rescoring")
	p.add_argument("--dry", action="store_true", help="rescore without rewriting reports/<batch>")
	p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
	p.add_argument("--out", help="verify JSON path (default <root>/reports/<tag>/<spec out>.json)")
	p.add_argument("--write-bank", action="store_true", help="write config/agent_bench/failure_bank/<tag>.json")
	a = p.parse_args(argv)
	spec_path = _spec_path(a.round)
	spec = load_spec(spec_path)
	root = Path(a.root or os.environ.get("ARMIGO_LOOP_ROOT") or spec["root"])
	doc = run_gates(spec, Loader(root, from_reports=a.from_reports, dry=a.dry, jobs=a.jobs))
	out = Path(a.out) if a.out else root / "reports" / spec["tag"] / f"{spec.get('out', 'verify')}.json"
	out.parent.mkdir(parents=True, exist_ok=True)
	public = {k: v for k, v in doc.items() if not k.startswith("_")}
	out.write_text(json.dumps(public, indent=2, ensure_ascii=False, default=dict), encoding="utf-8")
	out.with_suffix(".md").write_text(render_md(spec, public), encoding="utf-8")
	if a.write_bank:
		write_bank(spec, doc, REPO / "config" / "agent_bench" / "failure_bank" / f"{spec['tag']}.json")
	print(json.dumps({k: v for k, v in doc["gates"].items() if k != "criteria"}, indent=1, default=str)[:3000])
	print(f"wrote {out}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
