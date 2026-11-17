"""CLI runner for bowl collision candidates across robustness gates G0-G5."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
	sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
	sys.path.insert(0, str(ROOT / "scripts"))

from bowl_experiments import candidates, gates

RESULTS_DIR = ROOT / "results" / "bowl-experiments"
SEQUENCE = ("R1", "R2", "R3", "R4")


def _json_default(obj: Any) -> Any:
	import numpy as np

	if isinstance(obj, np.generic):
		return obj.item()
	if isinstance(obj, np.ndarray):
		return obj.tolist()
	raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def run_candidate(cid: str, max_workers: int = 12, stop_on_fail: bool = True) -> dict[str, Any]:
	"""Run gates G0 -> G5 on ``cid``, stopping at the first failure unless ``stop_on_fail=False``."""
	rcid = candidates.resolve_id(cid)
	out_dir = RESULTS_DIR / rcid
	out_dir.mkdir(parents=True, exist_ok=True)
	report_path = out_dir / "gate_report.json"

	t_start = time.time()
	gate_results: list[dict[str, Any]] = []
	builder = candidates.model_builder(rcid)
	failed_gate: str | None = None
	failed_detail: dict[str, Any] | None = None

	gate_calls = [
		("G0", lambda: gates.g0_geometry(builder)),
		("G1", lambda: gates.g1_collider_micro(builder)),
		("G2", lambda: gates.g2_static_soak(rcid)),
		("G3", lambda: gates.g3_two_failing_conditions(rcid, max_workers=max_workers)),
		("G4", lambda: gates.g4_full_regression(rcid, max_workers=max_workers)),
		("G5", lambda: gates.g5_engineering(rcid)),
	]

	for gname, fn in gate_calls:
		t0 = time.time()
		res = fn()
		dt = round(time.time() - t0, 3)
		entry = {"gate": res.name, "passed": res.passed, "wall_s": dt, "detail": res.detail}
		gate_results.append(entry)
		print(f"[{rcid}] {res.name}: {'PASS' if res.passed else 'FAIL'} ({dt:.2f}s) {res.detail}", flush=True)
		if not res.passed and failed_gate is None:
			failed_gate = gname
			failed_detail = res.detail
			if stop_on_fail:
				out = {
					"id": rcid,
					"passed": False,
					"failed_gate": gname,
					"detail": res.detail,
					"wall_s": round(time.time() - t_start, 3),
					"gates": gate_results,
				}
				report_path.write_text(json.dumps(out, indent=2, default=_json_default), encoding="utf-8")
				return out

	out = {
		"id": rcid,
		"passed": failed_gate is None,
		"failed_gate": failed_gate,
		"detail": failed_detail,
		"wall_s": round(time.time() - t_start, 3),
		"gates": gate_results,
	}
	report_path.write_text(json.dumps(out, indent=2, default=_json_default), encoding="utf-8")
	return out


def _write_decision(results: list[dict[str, Any]], winner: dict[str, Any] | None) -> None:
	lines = ["# Bowl Convex Decomposition Experiment Decision", ""]
	if winner is not None:
		lines.append(f"**Adopted candidate:** `{winner['id']}` (passed G0–G5 in {winner['wall_s']:.1f} s).")
	else:
		lines.append("**Result:** No candidate in R1–R4 passed all gates; no `src/` files were modified.")
	lines.extend(["", "## Candidate Summary", ""])
	for r in results:
		status = "PASS (G0–G5)" if r["passed"] else f"FAIL at `{r['failed_gate']}`"
		lines.append(f"- **{r['id']}**: {status}")
		for g in r["gates"]:
			lines.append(
				f"  - `{g['gate']}` ({'PASS' if g['passed'] else 'FAIL'}, {g['wall_s']:.2f}s): `{json.dumps(g['detail'])}`"
			)
	(RESULTS_DIR / "DECISION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
	ap = argparse.ArgumentParser(description=__doc__)
	group = ap.add_mutually_exclusive_group(required=True)
	group.add_argument("--candidate", help="Run a single candidate ID (e.g. R0, R9, R1..R4).")
	group.add_argument("--until-pass", action="store_true", help="Run R1->R2->R3->R4 and stop at the first pass.")
	ap.add_argument("--all-gates", action="store_true", help="Do not stop on the first failed gate.")
	ap.add_argument("--workers", type=int, default=12)
	args = ap.parse_args(argv)

	if args.candidate:
		res = run_candidate(args.candidate, max_workers=args.workers, stop_on_fail=not args.all_gates)
		return 0 if res["passed"] else 1

	results = []
	winner = None
	for cid in SEQUENCE:
		res = run_candidate(cid, max_workers=args.workers)
		results.append(res)
		if res["passed"]:
			winner = res
			break
	_write_decision(results, winner)
	return 0 if winner is not None else 1


if __name__ == "__main__":
	raise SystemExit(main())
