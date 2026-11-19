"""S0 replay gate: replay bench16 expert episodes through the student segment interface and require identity.

For each case the expert runs once through the graph with segment markers (``ExpertTracer``; reused when its
``run.json`` already exists in ``--out``), then a ``StudentTwin`` replays it with ``ReplayPolicy`` (the recorded
float64 targets and segment ends; it sees only the policy inputs, and its ``check_segment`` matches every whole
segment, planner diagnostics included, through the separate ``segment_monitor`` channel). Checks per case (``student_rollout.compare_runs``), all exact: physics telemetry
digest and every observed joint reading, capture frames, detector decisions, early stops of the wrist scan, segment
kinds/params/start/end, destination, the ``evaluate`` result and the whole event stream (without wall-clock fields).
Two documented exceptions: captured images are compared within the MuJoCo renderer's own repeatability (it is not
bitwise repeatable for one state, see ``RENDER_MAX_LEVEL``), and a view's detector scores (total and every head)
are compared exactly whenever its detector crop is bitwise equal, even if its full frame differs; only a differing
crop reports the score delta instead (the decision is always exact);
``smoothness.clipped_frames`` is left out of the result (see ``compare_runs``).

	PYTHONPATH=src .venv/Scripts/python scripts/vla/replay_gate.py --out E:/s0-scratch/gate
	PYTHONPATH=src .venv/Scripts/python scripts/vla/replay_gate.py --out E:/s0-scratch/gate --cases 0 3 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

CHECKS = (
	"captures",
	"images",
	"decisions",
	"scores",
	"early_stops",
	"segments",
	"destination",
	"evaluate",
	"telemetry",
	"events",
)


def gate_case(index, case, scorer, out: Path, sensing="realistic", video=False):
	from digital_twin_armigo.agent.student_rollout import ReplayPolicy, compare_runs, run_episode

	case_dir = out / f"case-{index:02d}"
	expert_dir, student_dir = case_dir / "expert", case_dir / "student"
	if (expert_dir / "run.json").exists():
		expert = json.loads((expert_dir / "run.json").read_text(encoding="utf-8"))
	else:
		case_dir.mkdir(parents=True, exist_ok=True)
		expert = run_episode(case, scorer, expert_dir, sensing=sensing, video=video)
	if student_dir.exists():
		import shutil

		shutil.rmtree(student_dir)
	policy = ReplayPolicy.from_trace(expert_dir)
	error, info = None, {}
	try:
		student = run_episode(
			case, scorer, student_dir, policy=policy, segment_monitor=policy.check_segment, sensing=sensing, video=video
		)
		checks, info = compare_runs(expert, student, expert_dir, student_dir)
		checks["state"] = policy.state_mismatches == 0
		checks["all_segments_used"] = policy.index == len(policy.segments) - 1
	except (
		RuntimeError,
		ValueError,
	) as exc:  # a replay mismatch is a gate failure for this case, not a crash of the gate
		error = f"{type(exc).__name__}: {exc}"
		student, checks = None, dict.fromkeys((*CHECKS, "state", "all_segments_used"), False)
	lift = 0.0
	if student is not None:
		import numpy as np

		pairs = list(zip(expert["lift_poses"], student["lift_poses"], strict=False))
		lift = max((float(np.max(np.abs(np.subtract(a, b)))) for a, b in pairs), default=0.0)
	return {
		"index": index,
		"case": list(case),
		"verdict": expert["result"]["verdict"],
		"label": expert["result"]["label"],
		"destination": expert["result"]["destination"],
		"frames": expert["frames"],
		"views": sum(e["kind"] == "view" for e in expert["events"]),
		"segments": sum(e["kind"] == "segment" for e in expert["events"]),
		"checks": checks,
		"pass": all(checks.values()),
		"error": error,
		**info,
		"max_state_error": policy.max_state_error,
		"lift_pose_max_diff_rad": lift,
		"clipped_frames": [expert["clipped_frames"], None if student is None else student["clipped_frames"]],
	}


def main():
	p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	p.add_argument("--suite", type=Path, default=ROOT / "config/agent_bench/bench16.json")
	p.add_argument("--detector", type=Path, default=ROOT / "results/agent-bench/sup-v7bot-local.json")
	p.add_argument("--out", type=Path, required=True, help="scratch directory (outside the repository)")
	p.add_argument("--cases", type=int, nargs="*", help="case indices (default: all)")
	p.add_argument("--sensing", choices=("realistic", "truth"), default="realistic")
	p.add_argument("--video", action="store_true", help="record lossless video in both runs (slower)")
	a = p.parse_args()
	from digital_twin_armigo.inspection_scorer import default_sup_scorer

	cases = json.loads(a.suite.read_text(encoding="utf-8"))
	indices = a.cases if a.cases else range(len(cases))
	scorer = default_sup_scorer(a.detector)
	rows = []
	try:
		for i in indices:
			row = gate_case(i, tuple(cases[i]), scorer, a.out, a.sensing, a.video)
			rows.append(row)
			failed = [k for k, v in row["checks"].items() if not v]
			print(
				f"case {i:2d} {row['case']!s:24} {row['verdict']:8} {row['label']:9} -> {row['destination']!s:16} "
				f"frames {row['frames']:5d} views {row['views']:2d} segments {row['segments']:3d} "
				f"{'PASS' if row['pass'] else 'FAIL ' + ','.join(failed)}"
				+ (f"  [{row['error']}]" if row["error"] else "")
				+ (
					f"  (render noise: {len(row['render_noise_views'])} views, "
					f"{len(row['render_noise_crop_views'])} with a differing crop, max level "
					f"{row['render_noise_max_level']}, {row['render_noise_max_fraction']:.5f} px, "
					f"max score delta {row['render_noise_max_score_delta']:.4f})"
					if row.get("render_noise_views")
					else ""
				),
				flush=True,
			)
	finally:
		if hasattr(scorer, "close"):
			scorer.close()
	n_pass = sum(r["pass"] for r in rows)
	summary = {"passed": n_pass, "cases": len(rows), "rows": rows}
	a.out.mkdir(parents=True, exist_ok=True)
	(a.out / "gate.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
	print(f"\n{n_pass}/{len(rows)} cases identical -> {'PASS' if n_pass == len(rows) else 'FAIL'}")
	sys.exit(0 if n_pass == len(rows) else 1)


if __name__ == "__main__":
	main()
