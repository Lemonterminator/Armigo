"""Paired analysis of the clamp-wait validation (docs/vla/CLAMP_WAIT_VALIDATION_2026-11-23.md, v1 and v2).

Arms: ``act`` (A, records its request RGB), ``act-rep`` (A', pinned to A, hand-off ``clamp-wait``), ``clamp-wait``
(W1, ``--clamp-wait``, pinned to A) and ``clamp-wait-v2`` (W2, ``--clamp-wait --clamp-wait-obstruct``, pinned to A).
Writes ``paired_cases.csv``, ``paired_stats.json`` and ``paired_stats.md``: four-cell tables, net differences, bootstrap
intervals and exact McNemar p per pair; lift conversion; collision details (robot / jaw-environment, peak force, the
segment of the first contact and whether it fell in a clamp-wait hold) of every collision layout; clamp-wait events
(first grasp vs retries) with the jaw-environment contact frames inside each hold; prefix checks; and the
pre-registered decision (v2 when a ``clamp-wait-v2`` arm is given, else v1).

    python scripts/vla/clamp_wait_report.py --run act=<dir> --run act-rep=<dir> --run clamp-wait=<dir> \
        [--run clamp-wait-v2=<dir>] --out <dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from functools import cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "vla"))

import numpy as np
from grasp_tail_report import ENDPOINTS, first_state_divergence, four_cell, load_arm, prefix_check

ARMS = ("act", "act-rep", "clamp-wait", "clamp-wait-v2")
PAIRS = (("act", "act-rep"), ("act", "clamp-wait"), ("act", "clamp-wait-v2"), ("clamp-wait", "clamp-wait-v2"))
FIRST_CLAMP_NET = 10  # pre-registered (cases out of 100)
COLLISION_NET_MAX = 2  # v1 veto; v2 reports it and attributes the excess
HOLD_CONTACT_FRAMES_MAX = 5  # v2: jaw-environment contact frames allowed inside any one clamp-wait hold
JAW_ENV_LIMIT_N = 2.0  # max_jaw_env_contact_n of the frozen grader reference (a contact frame is above it)
ROBOT_LIMIT_N = 3.0  # max_robot_contact_n of the same reference


@cache
def _contacts(ep_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
	"""Per-frame (robot N, jaw-env N, segment id) of branch 0, or None without an action record."""
	path = Path(ep_dir) / "branch-0" / "actions.npz"
	if not path.exists():
		return None
	z = np.load(path)
	if "contact_jaw_env_n" not in z.files:
		return None
	seg = z["segment"] if "segment" in z.files else np.full(len(z["contact_jaw_env_n"]), -1)
	return np.asarray(z["contact_robot_n"]), np.asarray(z["contact_jaw_env_n"]), np.asarray(seg)


def _events(ep_dir: str) -> list[dict[str, Any]]:
	path = Path(ep_dir) / "events.jsonl"
	if not path.exists():
		return []
	return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def clamp_waits(ep_dir: str) -> list[dict[str, Any]]:
	"""The episode's ``clamp_wait`` events, each with the grasp attempt it closed (1 = first grasp), its hold frames
	``[frame - frames, frame)`` and the jaw-environment contact frames (above ``JAW_ENV_LIMIT_N``) inside the hold."""
	out, grasps = [], 0
	con = _contacts(str(ep_dir))
	for e in _events(str(ep_dir)):
		if e.get("kind") == "segment_end" and e.get("segment_kind") == "grasp":
			grasps += 1
		elif e.get("kind") == "clamp_wait":  # logged just before its grasp segment's segment_end
			lo, hi = e["frame"] - e.get("frames", 0), e["frame"]
			jaw = None if con is None else con[1][lo:hi]
			out.append(
				{
					"attempt": grasps + 1,
					"frames": e.get("frames"),
					"outcome": e.get("outcome"),
					"frame": e["frame"],
					"arm_err_max": e.get("arm_err_max"),
					"hold_contact_frames": None if jaw is None else int((jaw > JAW_ENV_LIMIT_N).sum()),
					"hold_contact_max_n": None if jaw is None or not len(jaw) else round(float(jaw.max()), 3),
				}
			)
	return out


def first_contact(ep_dir: str) -> dict[str, Any] | None:
	"""Where the episode's first contact frame (robot above ``ROBOT_LIMIT_N`` or jaw-env above ``JAW_ENV_LIMIT_N``)
	fell: frame, segment kind, and whether inside a clamp-wait hold. None without contact or record."""
	con = _contacts(str(ep_dir))
	if con is None:
		return None
	robot, jaw, seg = con
	hits = np.flatnonzero((robot > ROBOT_LIMIT_N) | (jaw > JAW_ENV_LIMIT_N))
	if not len(hits):
		return None
	i = int(hits[0])
	kinds = {e["id"]: e.get("segment_kind") for e in _events(str(ep_dir)) if e.get("kind") == "segment_end"}
	holds = [(w["frame"] - w["frames"], w["frame"]) for w in clamp_waits(ep_dir)]
	return {
		"frame": i,
		"segment_kind": kinds.get(int(seg[i])),
		"in_hold": any(lo <= i < hi for lo, hi in holds),
		"kind": "robot" if robot[i] > ROBOT_LIMIT_N else "jaw_env",
	}


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
		res["new_collision_attribution"] = {
			str(k): first_contact(arms[b][tuple(k)]["dir"]) for k in res["collision"]["only_second_cases"]
		}
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
			"hold_contact_frames_total": sum(w["hold_contact_frames"] or 0 for w in ev),
			"hold_contact_frames_max": max((w["hold_contact_frames"] or 0 for w in ev), default=0),
			"holds_over_contact_max": [
				[*w["case"], w["frame"], w["hold_contact_frames"]]
				for w in ev
				if (w["hold_contact_frames"] or 0) > HOLD_CONTACT_FRAMES_MAX
			],
			"hold_contact_unavailable": sum(w["hold_contact_frames"] is None for w in ev),
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
	"""v2 (``act vs clamp-wait-v2``) when that pair exists, else v1 (``act vs clamp-wait``). The prefix check covers the
	pairs against A only: W1 vs W2 share no pinned prefix with each other beyond A's."""
	v2 = "act vs clamp-wait-v2" in stats["pairs"]
	w = stats["pairs"].get("act vs clamp-wait-v2" if v2 else "act vs clamp-wait")
	if w is None:
		return {"available": False}
	prefix_ok = all(
		p["prefix_summary"].keys() <= {"ok", "within_tol"} and not p["handoff_inconsistent_cases"]
		for name, p in stats["pairs"].items()
		if name.startswith("act vs ")
	)
	net = {e: w[e]["net_cases"] for e in ENDPOINTS}
	clamp_ok = net["first_clamp"] >= FIRST_CLAMP_NET
	downstream_ok = net["lifted"] >= 0 and net["success"] >= 0
	extra: dict[str, Any] = {"protocol": "v2" if v2 else "v1"}
	if v2:
		waits = stats["clamp_waits"]["clamp-wait-v2"]
		hold_ok = not waits["holds_over_contact_max"] and not waits["hold_contact_unavailable"]
		collision_ok = hold_ok and not w["new_robot_env_layouts"]
		extra |= {
			"hold_contact_ok": hold_ok,
			"collision_net_over_report_cap": net["collision"] > COLLISION_NET_MAX,
			"new_collision_attribution": w["new_collision_attribution"],
		}
	else:
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
	} | extra


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
				waits = ";".join(
					f"{x['attempt']}:{x['outcome']}:{x['frames']}:{x['hold_contact_frames']}"
					for x in clamp_waits(r["dir"])
				)
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
		if p["new_collision_attribution"]:
			L.append("")
			L.append("New collision layouts (first contact):")
			for k, v in p["new_collision_attribution"].items():
				L.append(f"- {k}: {json.dumps(v, sort_keys=True)}")
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
