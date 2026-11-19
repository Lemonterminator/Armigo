"""Count LLM skill choices that differ from the rule default and pair each episode with the rule run.

The rule default of a decision is ``choices[0]``: without an LLM, ``TwinMujoco.choose`` takes it. Reads the local
run folders (not in git) and writes ``summary/llm_disagreement.json``.

    python results/agent-bench/llm_disagreement.py
"""

import json
from collections import Counter
from pathlib import Path

BENCH = Path(__file__).resolve().parent
# (label, LLM run, rule run of the same suite and code version). The old-schema runs have no task.json, so they are
# paired by fixture index; the new runs are paired only when case and policy match.
PAIRS = [
	("before / dev4", "final/before/run-dev4-llm", "final/before/run-dev4-rule"),
	("before / bench16", "final/before/run-bench16-llm", "final/before/run-bench16-rule"),
	("after / dev4 w1", "final/after/run-dev4-llm-oldsrv", "final/after/run-dev4-rule"),
	("after / bench16 w1", "final/after/run-bench16-llm-oldsrv", "final/after/run-bench16-rule"),
	("after / bench16 w4", "final/after/run-bench16-llm-w4", "final/after/run-bench16-rule"),
]


def outcome(case_dir):
	path = case_dir / "episode.json"
	if not path.is_file():
		return None
	r = json.loads(path.read_text(encoding="utf-8"))["result"]
	return {k: r.get(k) for k in ("verdict", "label", "rewinds", "recovery_variants", "sim_frames")}


def decisions(case_dir):
	with (case_dir / "events.jsonl").open(encoding="utf-8") as f:
		return [e for e in map(json.loads, f) if e["kind"] == "decision"]


def paired(case_dir, rule_dir):
	if not rule_dir.is_dir():
		return False
	if not (case_dir / "task.json").is_file():
		return True
	task, rule = (json.loads((d / "task.json").read_text(encoding="utf-8")) for d in (case_dir, rule_dir))
	return task["case"] == rule["case"] and task["policy"] == rule["policy"]


def compare(llm_root, rule_root):
	stats, diffs = Counter(), []
	for case_dir in sorted(p for p in llm_root.iterdir() if p.is_dir()):
		rule_dir = rule_root / case_dir.name
		mine, rule = outcome(case_dir), outcome(rule_dir) if paired(case_dir, rule_dir) else None
		for e in decisions(case_dir):
			if e.get("source") not in ("llm", "fallback"):
				stats["rule_only"] += 1  # a single choice and no failure: the LLM is not consulted
				continue
			stats["consulted"] += 1
			stats[f"consulted_{e['node']}"] += 1
			if e["choice"] != e["choices"][0]:
				stats["disagree"] += 1
				diffs.append({"case": case_dir.name, **{k: e[k] for k in ("node", "branch", "choices", "choice")}})
				diffs[-1].update(llm_outcome=mine, rule_outcome=rule)
		if mine and rule:
			stats["paired_episodes"] += 1
			stats["same_verdict"] += mine["verdict"] == rule["verdict"]
			stats["llm_success"] += mine["verdict"] == "SUCCESS"
			stats["rule_success"] += rule["verdict"] == "SUCCESS"
	return dict(stats), diffs


def main():
	rows = []
	for label, llm_run, rule_run in PAIRS:
		if (BENCH / llm_run).is_dir():
			stats, diffs = compare(BENCH / llm_run, BENCH / rule_run)
			rows.append({"run": label, "llm": llm_run, "rule": rule_run, "stats": stats, "disagreements": diffs})
	out = BENCH / "summary" / "llm_disagreement.json"
	out.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
	for row in rows:
		print(row["run"], row["stats"])


if __name__ == "__main__":
	main()
