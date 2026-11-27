"""Round-3 three gates (drift alarm on ring rejection). PYTHONPATH = candidate checkout src + scripts/experiments.

	LOOP_CONFIG=round3_config python gates3.py [--out verify-drift_missed_alarm.json] [--jobs N] [--dry]

Drift condition only. Base rows for bank and regression are the drift rows of the round-2b candidate arm
(rerun00, rerun01, fresh-candidate: 444); the candidate reruns the same cases. Fresh pairs are round_03_fresh_192 under
drift, base arm at the round-2b candidate code. Pairs keyed on ``<camera>/<case_id>``, CI clusters by layout.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median

os.environ.setdefault("LOOP_CONFIG", "round3_config")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gates2b  # noqa: E402  (check_complete / prescore / keyed, bound to this config below)
import round3_config as C  # noqa: E402
from digital_twin_armigo.experiments import report  # noqa: E402
from digital_twin_armigo.experiments.stats import paired_comparison  # noqa: E402
from verify_fix import evaluate_three_gates  # noqa: E402

gates2b.ROOT = C.ROOT
BATCHES = tuple(b for b, _, _ in C.batches("candidate")) + (f"{C.TAG}-fresh-base",)
GRASPISH = ("grasp_or_path_failure", "recovery_failure")


def rows_of(path):
	return [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s]


def report_rows(batch, dry):
	if dry:
		return [json.loads(json.dumps(r, ensure_ascii=False)) for r in report.scan_batch(C.ROOT / "runs" / batch)[0]]
	return rows_of(Path(report.write_report(C.ROOT, batch)) / "scores.jsonl")


def drift(rows):
	return [r for r in rows if r["camera_condition"] == "drift"]


def main():
	p = argparse.ArgumentParser()
	p.add_argument("--out", default="verify-drift_missed_alarm.json")
	p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
	p.add_argument("--dry", action="store_true")
	a = p.parse_args()
	gates2b.check_complete(BATCHES)
	gates2b.prescore(BATCHES, a.jobs)
	crit = json.loads((C.OUT / "criteria.json").read_text(encoding="utf-8"))
	prev = C.ROOT / "reports"
	base0 = gates2b.keyed(
		drift(
			rows_of(prev / f"{C.PREV}-rerun00/scores.jsonl")
			+ rows_of(prev / f"{C.PREV}-rerun01/scores.jsonl")
			+ rows_of(prev / f"{C.PREV}-fresh-candidate/scores.jsonl")
		)
	)
	rerun = gates2b.keyed(sum((report_rows(f"{C.TAG}-{n}", a.dry) for n in C.RERUNS), []))
	fresh_b = gates2b.keyed(report_rows(f"{C.TAG}-fresh-base", a.dry))
	fresh_c = gates2b.keyed(report_rows(f"{C.TAG}-fresh-candidate", a.dry))
	assert len(base0) == len(rerun) == crit["rerun_pairs"] and len(fresh_b) == len(fresh_c) == crit["fresh_pairs"], (
		len(base0), len(rerun), len(fresh_b), len(fresh_c))
	bank_ids = set(crit["bank_ids"])
	reg_ids = {r["case_id"] for r in base0 if r["physical_success"]}
	assert not bank_ids & reg_ids and bank_ids <= {r["case_id"] for r in base0}
	assert len(reg_ids) == crit["regression_pairs"], len(reg_ids)
	pick = lambda rows, ids: [r for r in rows if r["case_id"] in ids]  # noqa: E731
	result = evaluate_three_gates(
		pick(base0, bank_ids), pick(rerun, bank_ids), pick(base0, reg_ids), pick(rerun, reg_ids), fresh_b, fresh_c, crit,
		expected_bank_ids=bank_ids, expected_reg_ids=reg_ids, expected_fresh_ids={r["case_id"] for r in fresh_b},
		allow_group_mismatch=True,
	)
	bmap = {r["case_id"]: r for r in base0}
	fb = {r["case_id"]: r for r in fresh_b}
	fail = lambda rows: Counter(r["root_cause"] for r in rows if not r["physical_success"])  # noqa: E731
	extra = {
		"all_reruns": paired_comparison(base0, rerun, B=2000, seed=0),
		"bank_cases": [
			{"case": r["case_id"], "new_success": r["physical_success"], "new_root_cause": r["root_cause"]}
			for r in sorted(pick(rerun, bank_ids), key=lambda r: r["case_id"])
		],
		"regressions": [
			{"case": r["case_id"], "root_cause": r["root_cause"], "mechanism": r.get("mechanism"), "path": r["path"]}
			for r in rerun if bmap[r["case_id"]]["physical_success"] and not r["physical_success"]
		],
		"rerun_changed_outcome": [
			{"case": r["case_id"], "base": bmap[r["case_id"]]["physical_success"], "candidate": r["physical_success"]}
			for r in rerun if bmap[r["case_id"]]["physical_success"] != r["physical_success"]
		],
		"fresh_discordant": [
			{"case": r["case_id"], "base": fb[r["case_id"]]["physical_success"], "candidate": r["physical_success"],
				"base_root_cause": fb[r["case_id"]]["root_cause"], "candidate_root_cause": r["root_cause"]}
			for r in fresh_c if fb[r["case_id"]]["physical_success"] != r["physical_success"]
		],
		"root_causes": {"base0": fail(base0), "rerun": fail(rerun), "fresh_base": fail(fresh_b), "fresh_candidate": fail(fresh_c)},
		"fresh_grasp_recovery": {"base": sum(r["root_cause"] in GRASPISH for r in fresh_b),
			"candidate": sum(r["root_cause"] in GRASPISH for r in fresh_c)},
	}
	costs = {}
	for name, x, y in (("rerun", base0, rerun), ("fresh", fresh_b, fresh_c)):
		xm = {r["case_id"]: r for r in x}
		f = [r["sim_frames"] - xm[r["case_id"]]["sim_frames"] for r in y]
		t = [r["wall_s"] - xm[r["case_id"]]["wall_s"] for r in y]
		costs[name] = {"frames_mean": mean(f), "frames_median": median(f), "wall_mean": mean(t), "wall_median": median(t),
			"changed_frames": sum(v != 0 for v in f)}
	extra["costs"] = costs
	out = Path(a.out) if Path(a.out).is_absolute() else C.OUT / a.out
	out.write_text(json.dumps({"gates": result, **extra}, indent=2, ensure_ascii=False, default=dict), encoding="utf-8")
	print(json.dumps({k: result[k] for k in result if k != "criteria"}, indent=1, default=str)[:4000])
	print(json.dumps({k: v for k, v in extra.items() if k not in ("bank_cases", "fresh_discordant")}, indent=1, default=dict)[:6000])


if __name__ == "__main__":
	main()
