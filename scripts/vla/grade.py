"""Grade recorded twin episodes for VLA export (plan S6) and report per-class pass rates.

Usage:
	python scripts/vla/grade.py RUNS_DIR [RUNS_DIR ...] [--reference config/vla/grader_reference.json] [--json OUT]
	python scripts/vla/grade.py --write-reference RUNS_DIR [RUNS_DIR ...]

A runs directory holds episode folders (any depth) with ``episode.json``. ``--write-reference`` rebuilds the expert
smoothness reference from the successful episodes found and writes it, with the current thresholds, to the
reference file.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from digital_twin_armigo.agent.vla_grader import (
	CHECKS,
	REFERENCE_PATH,
	build_smoothness_reference,
	find_episodes,
	grade,
	load_reference,
	summarize,
)

SHORT = {"pass": "ok", "fail": "FAIL", "unavailable": "n/a"}


def _display(path: Path) -> str:
	"""Repository-relative form of a runs path (from its ``results`` folder on) for the reference file."""
	parts = path.resolve().parts
	return "/".join(parts[parts.index("results") :]) if "results" in parts else path.as_posix()


def main():
	p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	p.add_argument("runs", nargs="+", type=Path)
	p.add_argument("--reference", type=Path, default=REFERENCE_PATH)
	p.add_argument("--json", type=Path, help="Write every grade and the summary to this file.")
	p.add_argument("--write-reference", action="store_true")
	a = p.parse_args()
	episodes = [e for run in a.runs for e in find_episodes(run)]
	if not episodes:
		p.error("no episode.json found")
	if a.write_reference:
		ref = load_reference(a.reference) if a.reference.exists() else load_reference()
		ref["smoothness"] = build_smoothness_reference(episodes)
		ref["smoothness"]["source"] = sorted({_display(r) for r in a.runs})
		ref["smoothness"]["built"] = date.today().isoformat()
		a.reference.parent.mkdir(parents=True, exist_ok=True)
		a.reference.write_text(json.dumps(ref, indent=2) + "\n", encoding="utf-8")
		print(f"wrote {a.reference} from {ref['smoothness']['episodes']} successful episodes")
		return
	reference = load_reference(a.reference)
	grades = [grade(e, reference) for e in episodes]
	header = f"{'episode':<34} {'class':<14} {'label':<10} {'verdict':<8} " + " ".join(f"{c[:10]:>10}" for c in CHECKS)
	print(header + "  pass")
	for g in grades:
		name = str(Path(g["episode"]).relative_to(Path(g["episode"]).parents[1]))
		cells = " ".join(f"{SHORT[g['checks'][c]['status']]:>10}" for c in CHECKS)
		print(f"{name:<34} {g['class']:<14} {g['label'] or '-':<10} {g['verdict'] or '-':<8} {cells}  {g['passed']}")
	s = summarize(grades)
	print()
	print(f"{'class':<14} {'n':>3} {'strict':>7} {'avail.':>7}")
	for cls, row in s["classes"].items():
		flags = " ".join(
			f"<0.8x mean ({k.split('_rate')[0]})" for k in ("passed", "passed_available") if row[f"{k}_flag"]
		)
		print(
			f"{cls:<14} {row['episodes']:>3} {row['passed_rate']:>7.2f} {row['passed_available_rate']:>7.2f}  {flags}"
		)
	print(f"{'all':<14} {s['episodes']:>3} {s['passed_rate']:>7.2f} {s['passed_available_rate']:>7.2f}")
	al = s["alignment"]
	print(
		f"\nface alignment: {al['views']} views, raw mismatches {al['raw_mismatches']}, "
		f"after symmetry alignment {al['aligned_mismatches']}, methods {al['methods']}"
	)
	print(f"defective decision crops: {s['defective_decision_crops']}")
	print(f"unavailable per check: {s['unavailable']}")
	print(f"failed per check: {s['failed']}")
	if a.json:
		a.json.write_text(json.dumps({"summary": s, "grades": grades}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
	main()
