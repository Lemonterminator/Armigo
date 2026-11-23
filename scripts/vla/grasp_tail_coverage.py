"""Step 0 of docs/vla/GRASP_TAIL_AB_HANDOFF_2026-11-23.md: offline switch-point coverage of recorded student runs.

For every episode of a closed-loop run, rebuild the first grasp's plan snapshot from the recorded perception (the
last ``brick_perceived`` before the first ``grasp`` segment, the perceived bowls) with the same
``StudentTwin._planned_grasp()``, recover the gripper frame by forward kinematics of the recorded joint readings,
and replay the switch detector of ``agent/grasp_tail.py`` over the segment's observations and executed commands.

Checks of the rebuilt plan: its roll against the segment's recorded ``roll_deg`` (all episodes), and, for runs
recorded with ``--grasp-tail-mode``, its target against the logged snapshot (``grasp_tail.json``).

    python scripts/vla/grasp_tail_coverage.py --run E:/scratch/s11v2-fit30k-eval --out <dir>
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
	sys.path.insert(0, str(ROOT / "src"))

from digital_twin_armigo.agent.grasp_tail import GraspPlan, GraspTail
from digital_twin_armigo.agent.perception import DUPLO_2X2, DUPLO_2X4, BrickEstimate


class _NoPolicy:
	def start(self, task, segment_id):
		raise RuntimeError("offline reconstruction runs no segment")

	def act(self, obs):
		raise RuntimeError("offline reconstruction runs no segment")


def _events(ep_dir: Path) -> list[dict]:
	return [json.loads(x) for x in (ep_dir / "events.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]


def _first_grasp(events):
	start = next((e for e in events if e["kind"] == "segment" and e["segment_kind"] == "grasp"), None)
	if start is None:
		return None, None
	end = next(e for e in events if e["kind"] == "segment_end" and e["id"] == start["id"])
	return start, end


def _perception_before(events, frame):
	found = [e for e in events if e["kind"] == "brick_perceived" and e.get("found") and e["frame"] <= frame]
	return found[-1] if found else None


def _bowls_perceived(events):
	return next((e for e in events if e["kind"] == "bowls_perceived"), None)


def rebuild_plan(case, perceived: dict, bowls_event: dict | None):
	"""``(GraspPlan, twin)`` for the recorded perception on a fresh realistic StudentTwin of ``case``."""
	from digital_twin_armigo.agent.student_rollout import StudentTwin
	from digital_twin_armigo.agent.twin_mujoco import sample_case
	from digital_twin_armigo.agent.twin_record import EpisodeTrace

	trace = EpisodeTrace(Path(tempfile.mkdtemp(prefix="grasp-tail-cov-")) / "trace", video=False)
	twin = StudentTwin(sample_case(*case), None, trace, _NoPolicy(), sensing="realistic")
	bowl_delta_mm = {}
	if bowls_event is not None:
		for name, xy in (bowls_event.get("positions_m") or {}).items():
			if xy is not None and name in twin.bowls:
				bowl_delta_mm[name] = float(np.linalg.norm(np.asarray(xy) - twin.bowls[name]) * 1000)
	long_m = perceived["footprint_mm"][0] / 1000.0
	size = DUPLO_2X4 if long_m > 0.048 else DUPLO_2X2
	twin.brick_est = BrickEstimate(
		np.asarray(perceived["position_m"], dtype=np.float64),
		math.radians(perceived["yaw_deg"]),
		size,
		(perceived["footprint_mm"][0] / 1000.0, perceived["footprint_mm"][1] / 1000.0),
		int(perceived["top_points"]),
		tuple(perceived.get("occlusion_extended_mm", (0.0, 0.0))),
	)
	twin._perceived_size = tuple(size)
	ex = twin.expert
	ex._active_open_g = float(ex.p.open_g)  # TwinMujoco._grasp resets these before the grasp
	ex.grasp_retries = ex.block_replans = 0
	target, roll, open_g = twin._planned_grasp()
	return GraspPlan(target, roll, open_g, int(perceived["frame"]), int(perceived["frame"])), twin, bowl_delta_mm


def replay_switch(plan: GraspPlan, twin, states: np.ndarray, actions: np.ndarray, end_state: np.ndarray | None):
	"""Run the switch detector over observations ``states`` (+ the segment-end one) and executed ``actions``."""
	ex = twin.expert
	tail = GraspTail("act", plan, ex.p.pregrasp_dz_m, 0)
	obs_q = list(states) + ([end_state] if end_state is not None else [])
	for i, q in enumerate(obs_q):
		tcp, _ = ex.kin.site_pose(np.asarray(q, dtype=np.float64))
		obs = {
			"tcp": np.asarray(tcp, dtype=np.float64).copy(),
			"jaw_normal": np.zeros(2),
			"gripper_cmd": float("nan"),
			"effort_frac": float("nan"),
		}
		tail.observe(
			frame=i,
			segment_frame=i,
			obs=obs,
			q=np.asarray(q, dtype=np.float64),
			executed=None if i == 0 else actions[i - 1],
			clamp_run=0,
			gripper_state=None,
			clamp_frames=10**9,
		)
		if i < len(actions):
			tail.command("act", actions[i])
	tail.finish(len(obs_q) - 1, len(obs_q) - 1, "recorded")
	return tail


def analyse_episode(ep_dir: Path, attribution: dict | None) -> dict:
	events = _events(ep_dir)
	manifest = json.loads((ep_dir / "episode.json").read_text(encoding="utf-8"))
	case = tuple((manifest.get("provenance") or {})["case"])
	row: dict = {"case": list(case), "episode_dir": str(ep_dir)}
	if attribution is not None:
		fg = attribution.get("first_grasp") or {}
		row.update(
			first_clamped=fg.get("clamped"),
			lifted=attribution.get("lifted"),
			verdict=attribution.get("verdict"),
			first_failure=attribution.get("first_failure"),
		)
	start, end = _first_grasp(events)
	if start is None:
		row["decision"] = "no_grasp_segment"
		return row
	perceived = _perception_before(events, start["frame"])
	if perceived is None:
		row["decision"] = "no_target"
		return row
	plan, twin, bowl_delta = rebuild_plan(case, perceived, _bowls_perceived(events))
	try:
		with np.load(ep_dir / "branch-0" / "actions.npz", allow_pickle=False) as z:
			seg_ids, states, actions = z["segment"], z["state_f64"], z["action_f64"]
		rows = np.flatnonzero(seg_ids == start["id"])
		end_idx = rows[-1] + 1 if len(rows) else start["frame"]
		end_state = states[end_idx] if end_idx < len(states) else None
		tail = replay_switch(plan, twin, states[rows], actions[rows], end_state)
	finally:
		twin.close()
	recorded_roll = start["params"].get("roll_deg")
	row.update(
		plan=plan.to_json(),
		roll_rebuilt_deg=round(math.degrees(plan.roll), 1),
		roll_recorded_deg=recorded_roll,
		roll_match=recorded_roll is not None and abs(round(math.degrees(plan.roll), 1) - recorded_roll) < 0.051,
		bowl_rebuild_delta_mm=bowl_delta,
		segment_frames=int(end["frames"]),
		segment_end=end["reason"],
		decision=tail.decision,
		switch=tail.switch,
		closing_frame=next((e["segment_frame"] for e in tail.events if e["kind"] == "closing_started"), None),
	)
	logged = ep_dir / "grasp_tail.json"
	if logged.exists():
		first = json.loads(logged.read_text(encoding="utf-8")).get("first_grasp") or {}
		lp = first.get("plan")
		if lp is not None:
			row["logged_plan_delta_m"] = float(np.max(np.abs(np.asarray(lp["target_xyz_m"]) - plan.target)))
			row["logged_roll_delta_rad"] = abs(lp["roll_rad"] - plan.roll)
			row["logged_decision"] = first.get("decision")
			row["logged_switch_frame"] = (first.get("switch") or {}).get("segment_frame")
			row["rebuilt_switch_frame"] = (tail.switch or {}).get("segment_frame")
	return row


def summarize(rows: list[dict]) -> dict:
	def table(sel):
		c = Counter(r["decision"] for r in sel)
		return dict(sorted(c.items()))

	misses = [r for r in rows if r.get("first_clamped") is False]
	hits = [r for r in rows if r.get("first_clamped") is True]

	def errs(sel):
		out = [r["switch"]["err_xy_m"] * 1000 for r in sel if r.get("switch") and "err_xy_m" in r["switch"]]
		rolls = [abs(r["switch"]["err_roll_rad"]) for r in sel if r.get("switch") and "err_roll_rad" in r["switch"]]
		q = lambda a, p: None if not a else float(np.percentile(a, p))
		return {
			"crossed": len(out),
			"xy_mm_median": q(out, 50),
			"xy_mm_p90": q(out, 90),
			"roll_rad_median": q(rolls, 50),
			"roll_rad_p90": q(rolls, 90),
		}

	return {
		"episodes": len(rows),
		"roll_match": sum(bool(r.get("roll_match")) for r in rows),
		"first_grasp_misses": {"n": len(misses), "decisions": table(misses), "at_crossing": errs(misses)},
		"first_grasp_clamped": {"n": len(hits), "decisions": table(hits), "at_crossing": errs(hits)},
		"all": {"decisions": table(rows)},
	}


def markdown(summary: dict, rows: list[dict]) -> str:
	lines = ["# Step 0: first-grasp switch coverage", ""]
	lines.append(
		f"Episodes {summary['episodes']}; rebuilt roll matches the recorded roll_deg in {summary['roll_match']}."
	)
	lines.append("")
	keys = sorted(set(summary["first_grasp_misses"]["decisions"]) | set(summary["first_grasp_clamped"]["decisions"]))
	lines.append("| decision | first-grasp misses | first-grasp clamped |")
	lines.append("| --- | --- | --- |")
	for k in keys:
		lines.append(
			f"| {k} | {summary['first_grasp_misses']['decisions'].get(k, 0)} | "
			f"{summary['first_grasp_clamped']['decisions'].get(k, 0)} |"
		)
	lines.append(f"| total | {summary['first_grasp_misses']['n']} | {summary['first_grasp_clamped']['n']} |")
	lines.append("")
	for name in ("first_grasp_misses", "first_grasp_clamped"):
		e = summary[name]["at_crossing"]
		lines.append(
			f"- {name}: crossed {e['crossed']}, XY at crossing median {e['xy_mm_median']} mm, p90 {e['xy_mm_p90']} mm; |roll| median {e['roll_rad_median']} rad, p90 {e['roll_rad_p90']} rad"
		)
	lines.append("")
	lines.append(
		"| case | first clamped | lifted | verdict | decision | switch frame | XY mm | roll rad | closing frame | seg frames |"
	)
	lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
	for r in rows:
		s = r.get("switch") or {}
		xy = s.get("err_xy_m")
		lines.append(
			f"| {r['case']} | {r.get('first_clamped')} | {r.get('lifted')} | {r.get('verdict')} | {r.get('decision')} | "
			f"{s.get('segment_frame')} | {None if xy is None else round(xy * 1000, 2)} | "
			f"{None if s.get('err_roll_rad') is None else round(s['err_roll_rad'], 3)} | {r.get('closing_frame')} | "
			f"{r.get('segment_frames')} |"
		)
	return "\n".join(lines) + "\n"


def main(argv=None) -> int:
	ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	ap.add_argument("--run", type=Path, required=True, help="Closed-loop eval directory (with student/case-*)")
	ap.add_argument("--attribution", type=Path, default=None, help="failure_attribution.json (default: in --run)")
	ap.add_argument("--out", type=Path, required=True)
	ap.add_argument("--max-episodes", type=int, default=None)
	args = ap.parse_args(argv)
	attr_path = args.attribution or args.run / "failure_attribution.json"
	attribution = {}
	if attr_path.exists():
		for r in json.loads(attr_path.read_text(encoding="utf-8"))["rows"]:
			attribution[Path(r["episode_dir"]).name] = r
	ep_dirs = sorted(p for p in (args.run / "student").glob("case-*") if (p / "episode.json").exists())
	if args.max_episodes is not None:
		ep_dirs = ep_dirs[: args.max_episodes]
	rows = []
	for ep in ep_dirs:
		rows.append(analyse_episode(ep, attribution.get(ep.name)))
		print(ep.name, rows[-1].get("decision"), rows[-1].get("roll_match"), flush=True)
	summary = summarize(rows)
	args.out.mkdir(parents=True, exist_ok=True)
	(args.out / "coverage.json").write_text(
		json.dumps({"summary": summary, "rows": rows}, indent=1) + "\n", encoding="utf-8"
	)
	(args.out / "coverage.md").write_text(markdown(summary, rows), encoding="utf-8")
	print(json.dumps(summary, indent=1))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
