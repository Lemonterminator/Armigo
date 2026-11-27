"""Round-2 three gates (superseded by gates.py --round config/experiments/loop/round_02b.json; kept until that
reproduces verify-grasp_arm_occlusion.json, which needs the round-1 rows on D:).

Round-2 three gates. Run with PYTHONPATH = fix checkout src + scripts/experiments (scores both arms with one scorer).

	python gates2b.py [--out verify-grasp_arm_occlusion.json] [--jobs N] [--dry]

Pairs are keyed on ``<camera>/<case_id>`` (the same case runs under three camera conditions); CI clusters by layout.

Compared with the first round-2 gates script: it first checks every planned case has a done attempt (seconds, names what is
missing) instead of failing an assert after scoring; the ~2600 episodes are scored on a process pool, once, and
handed to report.write_report (same rows, same reports).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import mean, median

sys.path.insert(0, str(Path(__file__).resolve().parent))
from digital_twin_armigo.experiments import report, runner  # noqa: E402
from digital_twin_armigo.experiments.scoring import score_episode  # noqa: E402
from digital_twin_armigo.experiments.stats import paired_comparison  # noqa: E402
from round2b_config import OUT, R1, ROOT, ROOT1, TAG  # noqa: E402
from verify_fix import evaluate_three_gates  # noqa: E402

BATCHES = (f"{TAG}-rerun01", f"{TAG}-rerun00", f"{TAG}-fresh-base", f"{TAG}-fresh-candidate")


def rows_of(path):
	return [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s]


def check_complete(batches) -> None:
	"""Every case in every plan.json has a latest attempt in state done; raise naming the gaps otherwise."""
	gaps, n = [], 0
	for batch in batches:
		plans = sorted((ROOT / "runs" / batch).glob("*/*/plan.json"))
		if not plans:
			gaps.append(f"{batch}: no plan.json")
		for plan in plans:
			for cid in json.loads(plan.read_text(encoding="utf-8"))["cases"]:
				n += 1
				att = runner._attempts(plan.parent, cid)
				if not att or att[-1][2].get("state") != "done":
					gaps.append(f"{batch}/{plan.parent.name}/{cid}: {att[-1][2].get('state') if att else 'not run'}")
	if gaps:
		raise SystemExit(f"{len(gaps)} of {n} planned episodes not done, e.g. {gaps[:8]}; rerun driver2b.py / run2b.py")
	print(f"complete: {n} planned episodes done", flush=True)


def prescore(batches, jobs: int) -> None:
	"""Score every episode dir of ``batches`` in parallel; report.scan_batch then reads the cache."""
	dirs = [d for b in batches for d in report._episode_dirs(ROOT / "runs" / b)]
	with ProcessPoolExecutor(max_workers=jobs) as pool:
		scored = dict(zip(dirs, pool.map(score_episode, dirs, chunksize=8), strict=True))
	print(f"scored {len(dirs)} episode dirs with {jobs} processes", flush=True)
	report.score_episode = lambda d: dict(scored[d]) if d in scored else score_episode(d)


def keyed(rows):
	out = []
	for r in rows:
		if not r.get("has_episode") or r.get("infra_failed"):
			raise RuntimeError(f"incomplete row {r.get('path')}")
		# The pooled gates span the three camera groups on purpose (the camera is in the key), so the
		# single-group guard of verify_fix sees one pooled group; per-condition gates are reported too.
		out.append(
			{
				**r,
				"case_id": f"{r['camera_condition']}/{r['case_id']}",
				"orig_case_id": r["case_id"],
				"orig_group": r["group"],
				"group": "pooled-3cond",
			}
		)
	return out


DRY = False  # --dry: score without rewriting ROOT/reports/<batch> (rows go through JSON exactly as from scores.jsonl)


def report_rows(batch):
	if DRY:
		return [json.loads(json.dumps(r, ensure_ascii=False)) for r in report.scan_batch(ROOT / "runs" / batch)[0]]
	path = report.write_report(ROOT, batch)
	return rows_of(Path(path) / "scores.jsonl")


def main():
	p = argparse.ArgumentParser()
	p.add_argument("--out", default="verify-grasp_arm_occlusion.json", help="file name under OUT, or an absolute path")
	p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
	p.add_argument("--dry", action="store_true", help="leave the batch reports alone (use with an --out elsewhere)")
	args = p.parse_args()
	global DRY
	DRY = args.dry
	t0 = time.perf_counter()
	check_complete(BATCHES)
	prescore(BATCHES, args.jobs)
	print(f"checked and scored in {time.perf_counter() - t0:.0f} s", flush=True)
	crit = json.loads((OUT / "criteria.json").read_text(encoding="utf-8"))
	base0 = keyed(rows_of(ROOT1 / "reports" / f"{R1}-fresh-candidate/scores.jsonl")) + keyed(
		rows_of(ROOT1 / "reports" / f"{R1}-rerun/scores.jsonl")
	)
	rerun = keyed(report_rows(f"{TAG}-rerun01")) + keyed(report_rows(f"{TAG}-rerun00"))
	fresh_b = keyed(report_rows(f"{TAG}-fresh-base"))
	fresh_c = keyed(report_rows(f"{TAG}-fresh-candidate"))
	assert len(base0) == len(rerun) == 756 and len(fresh_b) == len(fresh_c) == 576, (
		len(base0),
		len(rerun),
		len(fresh_b),
		len(fresh_c),
	)
	bank_ids = {r["case_id"] for r in base0 if r["root_cause"] in ("grasp_or_path_failure", "recovery_failure")}
	reg_ids = {r["case_id"] for r in base0 if r["physical_success"]}
	assert len(bank_ids) == crit["bank_records"] and len(reg_ids) == crit["regression_pairs"], (len(bank_ids), len(reg_ids))
	pick = lambda rows, ids: [r for r in rows if r["case_id"] in ids]  # noqa: E731
	result = evaluate_three_gates(
		pick(base0, bank_ids),
		pick(rerun, bank_ids),
		pick(base0, reg_ids),
		pick(rerun, reg_ids),
		fresh_b,
		fresh_c,
		crit,
		expected_bank_ids=bank_ids,
		expected_reg_ids=reg_ids,
		expected_fresh_ids={r["case_id"] for r in fresh_b},
		allow_group_mismatch=True,
	)
	per_cond = {}
	for cam in ("fixed", "recal", "drift"):
		c = lambda rows: [r for r in rows if r["camera_condition"] == cam]  # noqa: E731
		bi = {i for i in bank_ids if i.startswith(cam + "/")}
		ri = {i for i in reg_ids if i.startswith(cam + "/")}
		per_cond[cam] = evaluate_three_gates(
			pick(base0, bi),
			pick(rerun, bi),
			pick(base0, ri),
			pick(rerun, ri),
			c(fresh_b),
			c(fresh_c),
			crit,
			expected_bank_ids=bi,
			expected_reg_ids=ri,
			expected_fresh_ids={r["case_id"] for r in c(fresh_b)},
			allow_group_mismatch=True,
		)
	extra = {
		"gates_by_condition": per_cond,
		"all756": paired_comparison(base0, rerun, B=2000, seed=0),
		"by_condition": {},
		"fresh_grasp_recovery": {
			"base": sum(r["root_cause"] in ("grasp_or_path_failure", "recovery_failure") for r in fresh_b),
			"candidate": sum(r["root_cause"] in ("grasp_or_path_failure", "recovery_failure") for r in fresh_c),
		},
		"root_causes": {
			"round1_cand": Counter(r["root_cause"] for r in base0 if not r["physical_success"]),
			"rerun": Counter(r["root_cause"] for r in rerun if not r["physical_success"]),
			"fresh_base": Counter(r["root_cause"] for r in fresh_b if not r["physical_success"]),
			"fresh_candidate": Counter(r["root_cause"] for r in fresh_c if not r["physical_success"]),
		},
	}
	for cam in ("fixed", "recal", "drift"):
		sel = lambda rows: [r for r in rows if r["camera_condition"] == cam]  # noqa: E731
		extra["by_condition"][cam] = {
			"round1_rerun": paired_comparison(sel(base0), sel(rerun), B=2000, seed=0),
			"fresh": paired_comparison(sel(fresh_b), sel(fresh_c), B=2000, seed=0),
		}
	bmap = {r["case_id"]: r for r in base0}
	extra["bank_cases"] = [
		{
			"case": cid,
			"new_success": next(r for r in rerun if r["case_id"] == cid)["physical_success"],
			"new_root_cause": next(r for r in rerun if r["case_id"] == cid)["root_cause"],
		}
		for cid in sorted(bank_ids)
	]
	extra["regressions"] = [
		{"case": r["case_id"], "root_cause": r["root_cause"], "mechanism": r.get("mechanism"), "path": r["path"]}
		for r in rerun
		if bmap[r["case_id"]]["physical_success"] and not r["physical_success"]
	]
	fb = {r["case_id"]: r for r in fresh_b}
	extra["fresh_discordant"] = [
		{
			"case": r["case_id"],
			"base": fb[r["case_id"]]["physical_success"],
			"candidate": r["physical_success"],
			"base_root_cause": fb[r["case_id"]]["root_cause"],
			"candidate_root_cause": r["root_cause"],
		}
		for r in fresh_c
		if fb[r["case_id"]]["physical_success"] != r["physical_success"]
	]
	costs = {}
	for name, a, b in (("rerun", base0, rerun), ("fresh", fresh_b, fresh_c)):
		am = {r["case_id"]: r for r in a}
		f = [r["sim_frames"] - am[r["case_id"]]["sim_frames"] for r in b]
		t = [r["wall_s"] - am[r["case_id"]]["wall_s"] for r in b]
		costs[name] = {"frames_mean": mean(f), "frames_median": median(f), "wall_mean": mean(t), "wall_median": median(t)}
	extra["costs"] = costs
	doc = {"gates": result, **extra}
	(OUT / args.out).write_text(
		json.dumps(doc, indent=2, ensure_ascii=False, default=dict), encoding="utf-8"
	)
	print(json.dumps({k: result[k] for k in result if k != "criteria"}, indent=1, default=str)[:4000])
	print(json.dumps({k: v for k, v in extra.items() if k not in ("bank_cases", "fresh_discordant")}, indent=1, default=dict)[:6000])


if __name__ == "__main__":
	main()
