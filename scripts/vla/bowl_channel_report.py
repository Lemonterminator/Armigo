"""Closed-loop paired comparison for the bowl-channel acceptance (docs/vla/BOWL_CHANNEL_VALIDATION_2026-11-23.md 4).

Two independent ``eval_closed_loop.py`` runs on the same suite (baseline and T1), paired by case: first clamp, lift,
sorted, wrong bowl (a placement check that found the brick in the bowl the carry was not told to go to), robot-env and
jaw-env collision layouts. Four-cell tables, net, bootstrap 95% CI and exact McNemar p per endpoint, the probe
verdicts when given, and the pre-registered secondary decision.

    python scripts/vla/bowl_channel_report.py --base <eval dir> --t1 <eval dir> \
        [--probe-base <probe dir>] [--probe-t1 <probe dir>] --out <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "vla"))
sys.path.insert(0, str(ROOT / "src"))

from grasp_tail_report import four_cell, load_arm

from digital_twin_armigo.agent.perception import bowl_filled

FOLLOW_MIN = 0.85
AGREE_MIN = 0.90
FIRST_CLAMP_DROP_MAX = 5
OTHER = {"green": "pink", "pink": "green"}


def wrong_bowl(ep_dir: str) -> bool:
	"""True when any placement check found the brick in the other bowl than the preceding carry's instruction."""
	bowl = None
	for line in (Path(ep_dir) / "events.jsonl").read_text(encoding="utf-8").splitlines():
		e = json.loads(line)
		if e.get("kind") == "segment" and e.get("segment_kind") == "carry_to_bowl":
			bowl = (e.get("params") or {}).get("bowl")
		elif e.get("kind") == "placement_check" and bowl in OTHER:
			fill = e.get("fill_pixels") or {}
			if bowl_filled(fill.get(OTHER[bowl], 0)) and not bowl_filled(fill.get(bowl, 0)):
				return True
	return False


def endpoints(row: dict[str, Any]) -> dict[str, bool | None]:
	col = row["collisions"]
	return {
		"first_clamp": row["first_clamp"],
		"lifted": bool(row.get("lifted")),
		"success": row["success"],
		"wrong_bowl": wrong_bowl(row["dir"]),
		"robot_collision": None if col is None else bool(col["robot_contact_frames"]),
		"jaw_env_collision": None if col is None else bool(col["jaw_env_contact_frames"]),
	}


def probe_verdict(run: Path | None) -> dict[str, Any] | None:
	if run is None:
		return None
	s = json.loads((run / "bowl_swap.json").read_text(encoding="utf-8"))
	a, b = s["req_req2_agree"]
	follow = s["classes"].get("follow", 0)
	return {
		"probed": s["probed"],
		"follow": follow,
		"follow_rate": follow / s["probed"] if s["probed"] else None,
		"req_req2_agree": [a, b],
		"agree_rate": a / b if b else None,
		"classes": s["classes"],
	}


def main(argv=None) -> int:
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--base", type=Path, required=True)
	ap.add_argument("--t1", type=Path, required=True)
	ap.add_argument("--probe-base", type=Path, default=None)
	ap.add_argument("--probe-t1", type=Path, default=None)
	ap.add_argument("--out", type=Path, required=True)
	args = ap.parse_args(argv)

	base, t1 = load_arm(args.base), load_arm(args.t1)
	keys = sorted(set(base) & set(t1))
	ep_b = {k: endpoints(base[k]) for k in keys}
	ep_t = {k: endpoints(t1[k]) for k in keys}
	tables = {}
	for name in ("first_clamp", "lifted", "success", "wrong_bowl", "robot_collision", "jaw_env_collision"):
		ks = [k for k in keys if ep_b[k][name] is not None and ep_t[k][name] is not None]
		tables[name] = four_cell([bool(ep_b[k][name]) for k in ks], [bool(ep_t[k][name]) for k in ks])
	pb, pt = probe_verdict(args.probe_base), probe_verdict(args.probe_t1)
	primary = None
	if pt is not None:
		primary = bool(
			pt["follow_rate"] is not None
			and pt["follow_rate"] >= FOLLOW_MIN
			and pt["agree_rate"] is not None
			and pt["agree_rate"] >= AGREE_MIN
		)
	t = tables
	secondary = {
		"success_not_lower": t["success"]["net_cases"] >= 0,
		"wrong_bowl_lower": t["wrong_bowl"]["second_total"] < t["wrong_bowl"]["first_total"],
		"first_clamp_drop_ok": t["first_clamp"]["net_cases"] >= -FIRST_CLAMP_DROP_MAX,
		"no_new_robot_collision": t["robot_collision"]["only_second"] == 0,
	}
	if primary is None:
		verdict = "probe_missing"
	elif not primary:
		verdict = "not_accepted"
	elif all(secondary.values()):
		verdict = "make_default"
	else:
		verdict = "follow_ok_inspect"
	report = {
		"base": str(args.base),
		"t1": str(args.t1),
		"paired_cases": len(keys),
		"missing": {"base_only": len(set(base) - set(t1)), "t1_only": len(set(t1) - set(base))},
		"probe": {"base": pb, "t1": pt, "primary_pass": primary},
		"tables": tables,
		"secondary": secondary,
		"verdict": verdict,
		"per_case": [{"case": list(k), "base": ep_b[k], "t1": ep_t[k]} for k in keys],
	}
	args.out.mkdir(parents=True, exist_ok=True)
	(args.out / "bowl_channel.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
	lines = [f"# Bowl channel acceptance ({len(keys)} paired cases)", ""]
	for name, p in (("base", pb), ("t1", pt)):
		if p:
			lines.append(
				f"probe {name}: follow {p['follow']}/{p['probed']} ({p['follow_rate']:.2f}), "
				f"req/req2 agree {p['req_req2_agree'][0]}/{p['req_req2_agree'][1]}"
			)
	lines += ["", "| endpoint | base | T1 | base only : T1 only | net | 95% CI | McNemar p |", "| --- " * 7 + "|"]
	for name, c in tables.items():
		ci = c["ci95"] or [None, None]
		ci_s = "" if ci[0] is None else f"[{ci[0]:+.2f}, {ci[1]:+.2f}]"
		lines.append(
			f"| {name} | {c['first_total']} | {c['second_total']} | {c['only_first']} : {c['only_second']} | "
			f"{c['net_cases']:+d} | {ci_s} | {c['mcnemar_p']:.3g} |"
		)
	lines += ["", f"secondary: {secondary}", f"primary (probe) pass: {primary}", "", f"**verdict: {verdict}**", ""]
	(args.out / "bowl_channel.md").write_text("\n".join(lines), encoding="utf-8")
	print("\n".join(lines))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
