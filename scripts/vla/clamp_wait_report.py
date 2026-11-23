"""Paired analysis of the clamp-wait validation (docs/vla/CLAMP_WAIT_VALIDATION_2026-11-23.md, sections 4-5).

Arms: ``act`` (A, records its request RGB), ``act-rep`` (A', pinned to A, hand-off ``clamp-wait``) and ``clamp-wait``
(W, ``--clamp-wait``, pinned to A). Writes ``paired_cases.csv``, ``paired_stats.json`` and ``paired_stats.md``: four-cell
tables, net differences, bootstrap intervals and exact McNemar p for A vs A' and A vs W; lift conversion; collision
details (robot / jaw-environment, peak force) of every collision layout; clamp-wait events (first grasp vs retries);
prefix checks; and the pre-registered decision.

    python scripts/vla/clamp_wait_report.py --run act=<dir> --run act-rep=<dir> --run clamp-wait=<dir> --out <dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "vla"))

from grasp_tail_report import ENDPOINTS, first_state_divergence, four_cell, load_arm, prefix_check

ARMS = ("act", "act-rep", "clamp-wait")
PAIRS = (("act", "act-rep"), ("act", "clamp-wait"))
FIRST_CLAMP_NET = 10  # pre-registered (cases out of 100)
COLLISION_NET_MAX = 2


def clamp_waits(ep_dir: str) -> list[dict[str, Any]]:
	"""The episode's ``clamp_wait`` events, each with the grasp attempt it closed (1 = first grasp)."""
	out, grasps = [], 0
	path = Path(ep_dir) / "events.jsonl"
	if not path.exists():
		return out
	for line in path.read_text(encoding="utf-8").splitlines():
		e = json.loads(line)
		if e.get("kind") == "segment_end" and e.get("segment_kind") == "grasp":
			grasps += 1
		elif e.get("kind") == "clamp_wait":  # logged just before its grasp segment's segment_end
			out.append(
				{"attempt": grasps + 1, "frames": e.get("frames"), "outcome": e.get("outcome"), "frame": e["frame"]}
			)
	return out


def analyse(runs: dict[str, Path]) -> dict[str, Any]:
	arms = {name: load_arm(path) for name, path in runs.items()}
	keys = sorted(set.intersection(*(set(v) for v in arms.values())))
	union = sorted(set.union(*(set(v) for v in arms.values())))
	stats: dict[str, Any] = {
		"runs": {k: str(v) for k, v in runs.items()},
		"cases": len(keys),
		"complete": keys == union and bool(keys),
		"incomplete_cases": [list(k) for k in union if k not in keys],
		"missing": {n: {str(list(k)): a[k]["missing"] for k in keys if a[k]["missing"]} for n, a in arms.items()},
	}
	manifests = {}
	for name, path in runs.items():
		mp = Path(path) / "run_manifest.json"
		m = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else None
		manifests[name] = None if m is None else {"identity": m.get("identity"), "code": m.get("code")}
	stats["manifests"] = manifests
	stats["totals"] = {
		n: {e: sum(bool(a[k][e]) for k in keys if a[k][e] is not None) for e in ENDPOINTS}
		| {
			"collision_graded": sum(a[k]["collision"] is not None for k in keys),
			"robot_env_layouts": sum(bool((a[k]["collisions"] or {}).get("robot_contact_frames")) for k in keys),
			"false_normal": sum(a[k]["outcome"] == "detector_false_normal" for k in keys),
		}
		for n, a in arms.items()
	}
	pairs = {}
	for a, b in PAIRS:
		if a not in arms or b not in arms:
			continue
		res: dict[str, Any] = {}
		for e in ENDPOINTS:
			ks = [k for k in keys if arms[a][k][e] is not None and arms[b][k][e] is not None]
			cell = four_cell([bool(arms[a][k][e]) for k in ks], [bool(arms[b][k][e]) for k in ks])
			cell["only_first_cases"] = [list(k) for k in ks if arms[a][k][e] and not arms[b][k][e]]
			cell["only_second_cases"] = [list(k) for k in ks if arms[b][k][e] and not arms[a][k][e]]
			res[e] = cell
		new_clamp = {tuple(k) for k in res["first_clamp"]["only_second_cases"]}
		new_lift = {tuple(k) for k in res["lifted"]["only_second_cases"]}
		res["lift_conversion"] = {"new_first_clamps": len(new_clamp), "also_new_lifts": len(new_clamp & new_lift)}
		res["new_robot_env_layouts"] = [
			list(k)
			for k in keys
			if (arms[b][k]["collisions"] or {}).get("robot_contact_frames")
			and not (arms[a][k]["collisions"] or {}).get("robot_contact_frames")
		]
		checks = {}
		for k in keys:
			c = prefix_check(arms[a][k], arms[b][k])
			c["first_state_divergence"] = first_state_divergence(arms[a][k], arms[b][k])
			checks[str(list(k))] = c
		res["prefix_checks"] = checks
		res["prefix_summary"] = dict(Counter(c["status"] for c in checks.values()))
		res["handoff_inconsistent_cases"] = [k for k, c in checks.items() if not c["handoff_consistent"]]
		res["pin_mismatch_cases"] = {k: c["pin_mismatch"] for k, c in checks.items() if c["pin_mismatch"]}
		res["identical_episodes"] = sum(c["first_state_divergence"] is None for c in checks.values())
		pairs[f"{a} vs {b}"] = res
	stats["pairs"] = pairs
	stats["collisions"] = {
		str(list(k)): {n: arms[n][k]["collisions"] for n in arms}
		for k in keys
		if any(arms[n][k]["collision"] for n in arms)
	}
	waits = {}
	for n, a in arms.items():
		ev = [dict(w, case=list(k)) for k in keys for w in clamp_waits(a[k]["dir"])]
		waits[n] = {
			"events": len(ev),
			"layouts": len({tuple(w["case"]) for w in ev}),
			"by_attempt": {
				att: dict(Counter(w["outcome"] for w in ev if (w["attempt"] == 1) == (att == "first")))
				for att in ("first", "retry")
			},
			"frames_total": sum(w["frames"] for w in ev),
			"frames_max": max((w["frames"] for w in ev), default=0),
		}
	stats["clamp_waits"] = waits
	stats["pins"] = {
		n: {
			"handoff_reasons": dict(Counter(str((a[k]["pin"] or {}).get("handoff_reason")) for k in keys)),
			"live_from": dict(Counter(str(((a[k]["pin"] or {}).get("live_from") or {}).get("reason")) for k in keys)),
			"pinned": sum((a[k]["pin"] or {}).get("pinned", 0) for k in keys),
		}
		for n, a in arms.items()
		if any((a[k]["pin"] or {}).get("role") == "pin" for k in keys)
	}
	stats["decision"] = decision(stats)
	return {"stats": stats, "arms": arms, "keys": keys}


def decision(stats: dict[str, Any]) -> dict[str, Any]:
	w = stats["pairs"].get("act vs clamp-wait")
	if w is None:
		return {"available": False}
	prefix_ok = all(
		p["prefix_summary"].keys() <= {"ok", "within_tol"} and not p["handoff_inconsistent_cases"]
		for p in stats["pairs"].values()
	)
	net = {e: w[e]["net_cases"] for e in ENDPOINTS}
	clamp_ok = net["first_clamp"] >= FIRST_CLAMP_NET
	downstream_ok = net["lifted"] >= 0 and net["success"] >= 0
	collision_ok = net["collision"] <= COLLISION_NET_MAX and not w["new_robot_env_layouts"]
	if not (stats["complete"] and prefix_ok):
		verdict = "incomplete_or_prefix_failed"
	elif clamp_ok and downstream_ok and collision_ok:
		verdict = "make_default"
	elif clamp_ok:
		verdict = "inspect_costs"
	else:
		verdict = "not_default"
	return {
		"complete": stats["complete"],
		"prefix_ok": prefix_ok,
		"net": net,
		"first_clamp_ok": clamp_ok,
		"downstream_ok": downstream_ok,
		"collision_ok": collision_ok,
		"pin_mismatch_cases": {k: len(p["pin_mismatch_cases"]) for k, p in stats["pairs"].items()},
		"verdict": verdict,
	}


def write_csv(path: Path, arms: dict, keys: list) -> None:
	cols = ["seed", "positive", "face", "arm", "dir", "first_clamp", "lifted", "success", "collision"]
	cols += ["robot_contact_frames", "jaw_env_contact_frames", "max_jaw_env_contact_n", "first_failure", "outcome"]
	cols += ["clamp_waits", "pin_handoff", "pin_live_from", "missing"]
	with path.open("w", newline="", encoding="utf-8") as fh:
		w = csv.writer(fh)
		w.writerow(cols)
		for k in keys:
			for n, a in arms.items():
				r, col = a[k], a[k]["collisions"] or {}
				pin = r["pin"] or {}
				waits = ";".join(f"{x['attempt']}:{x['outcome']}:{x['frames']}" for x in clamp_waits(r["dir"]))
				w.writerow(
					[*k, n, r["dir"], r["first_clamp"], r["lifted"], r["success"], r["collision"]]
					+ [col.get("robot_contact_frames"), col.get("jaw_env_contact_frames")]
					+ [col.get("max_jaw_env_contact_n"), r.get("first_failure"), r.get("outcome"), waits]
					+ [pin.get("handoff_reason"), (pin.get("live_from") or {}).get("reason"), ";".join(r["missing"])]
				)


def markdown(stats: dict[str, Any]) -> str:
	L = ["# Clamp-wait validation: paired statistics", ""]
	L.append(f"Paired layouts: {stats['cases']} (complete: {stats['complete']}).")
	L += ["", "| arm | first clamp | lifted | success | collision eps | robot-env eps | false normal |"]
	L.append("| --- | --- | --- | --- | --- | --- | --- |")
	for n, t in stats["totals"].items():
		L.append(
			f"| {n} | {t['first_clamp']} | {t['lifted']} | {t['success']} | {t['collision']} | "
			f"{t['robot_env_layouts']} | {t['false_normal']} |"
		)
	for name, p in stats["pairs"].items():
		L += ["", f"## {name}", "", "| endpoint | both | only first | only second | neither | net | 95% CI | p |"]
		L.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
		for e in ENDPOINTS:
			c = p[e]
			lo, hi = c["ci95"]
			L.append(
				f"| {e} | {c['both']} | {c['only_first']} | {c['only_second']} | {c['neither']} | "
				f"{c['net_cases']:+d} | [{lo:+.3f}, {hi:+.3f}] | {c['mcnemar_p']:.4f} |"
			)
		lc = p["lift_conversion"]
		L.append("")
		L.append(f"Lift conversion: {lc['also_new_lifts']} of {lc['new_first_clamps']} new first clamps.")
		L.append(
			f"Prefix: {p['prefix_summary']}; hand-off inconsistent {p['handoff_inconsistent_cases']}; "
			f"pin mismatches {len(p['pin_mismatch_cases'])}; identical episodes {p['identical_episodes']}; "
			f"new robot-env layouts {p['new_robot_env_layouts']}."
		)
	L += ["", "## Collision layouts", ""]
	for k, v in stats["collisions"].items():
		L.append(f"- {k}: {json.dumps(v, sort_keys=True)}")
	L += ["", "## Clamp waits", ""]
	for n, v in stats["clamp_waits"].items():
		L.append(f"- {n}: {json.dumps(v, sort_keys=True)}")
	L += ["", "## RGB pin", ""]
	for n, v in stats["pins"].items():
		L.append(f"- {n}: {json.dumps(v, sort_keys=True)}")
	L += ["", "## Decision", "", "```json", json.dumps(stats["decision"], indent=1), "```", ""]
	return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--run", action="append", required=True, help=f"arm=dir (arms: {', '.join(ARMS)})")
	ap.add_argument("--out", type=Path, required=True)
	args = ap.parse_args(argv)
	runs = {}
	for item in args.run:
		name, _, path = item.partition("=")
		if name not in ARMS or not path:
			raise SystemExit(f"--run must be arm=dir with arm in {ARMS}: {item!r}")
		runs[name] = Path(path)
	if "act" not in runs:
		raise SystemExit("--run act=<dir> is required")
	res = analyse(runs)
	args.out.mkdir(parents=True, exist_ok=True)
	write_csv(args.out / "paired_cases.csv", res["arms"], res["keys"])
	(args.out / "paired_stats.json").write_text(json.dumps(res["stats"], indent=1) + "\n", encoding="utf-8")
	(args.out / "paired_stats.md").write_text(markdown(res["stats"]), encoding="utf-8")
	print(markdown(res["stats"]))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
