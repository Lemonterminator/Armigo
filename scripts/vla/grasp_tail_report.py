"""Paired analysis of the first-grasp tail experiment (docs/vla/GRASP_TAIL_AB_HANDOFF_2026-11-23.md, sections 4-5).

From the raw episode files of the four arms (``act``, ``act-rep``, ``act-wait``, ``expert-once``), writes:

- ``paired_cases.csv``: one row per layout and arm (case key, run dir, switch decision and frame, near-field result,
  wait frames, command sources, first clamp / lift / sort / collision, first failure, missing items, prefix check);
- ``paired_stats.json`` and ``paired_stats.md``: four-cell tables, net differences, 95% percentile bootstrap intervals
  (case pairs resampled, 10,000 draws, seed 42) and exact two-sided McNemar p for first clamp, final success,
  collision and lift on A vs A', A vs A+wait, A+wait vs B and A vs B; lift conversion lists; untriggered reasons;
  prefix checks; and the pre-registered action rules.

    python scripts/vla/grasp_tail_report.py --run act=<dir> --run act-rep=<dir> --run act-wait=<dir> \\
        --run expert-once=<dir> --out <dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src", ROOT / "scripts" / "vla"):
	if str(p) not in sys.path:
		sys.path.insert(0, str(p))

from analyze_closed_loop import analyze_episode

ARMS = ("act", "act-rep", "act-wait", "expert-once")
PAIRS = (("act", "act-rep"), ("act", "act-wait"), ("act-wait", "expert-once"), ("act", "expert-once"))
ENDPOINTS = ("first_clamp", "lifted", "success", "collision")
NET_RULE = 6  # pre-registered net gain (cases out of 60)
TOL_RAD = 1e-5
N_BOOT, BOOT_SEED = 10_000, 42


def mcnemar_exact(b: int, c: int) -> float:
	"""Two-sided exact McNemar p from the discordant counts (1 when there are none)."""
	n = b + c
	if n == 0:
		return 1.0
	k = min(b, c)
	tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
	return min(1.0, 2.0 * tail)


def bootstrap_net(x: np.ndarray, y: np.ndarray, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> list[float]:
	"""95% percentile interval of mean(y - x), resampling case pairs."""
	d = y.astype(np.float64) - x.astype(np.float64)
	rng = np.random.default_rng(seed)
	idx = rng.integers(0, len(d), size=(n_boot, len(d)))
	means = d[idx].mean(axis=1)
	return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def four_cell(x: list[bool], y: list[bool]) -> dict[str, Any]:
	both = sum(a and b for a, b in zip(x, y, strict=True))
	only_first = sum(a and not b for a, b in zip(x, y, strict=True))
	only_second = sum(b and not a for a, b in zip(x, y, strict=True))
	neither = len(x) - both - only_first - only_second
	n = len(x)
	return {
		"n": n,
		"both": both,
		"only_first": only_first,
		"only_second": only_second,
		"neither": neither,
		"first_total": both + only_first,
		"second_total": both + only_second,
		"net_cases": only_second - only_first,
		"net": (only_second - only_first) / n if n else None,
		"ci95": bootstrap_net(np.array(x), np.array(y)) if n else None,
		"mcnemar_p": mcnemar_exact(only_first, only_second),
	}


def _key(case) -> tuple:
	return (int(case[0]), bool(case[1]), int(case[2]))


def load_arm(run_dir: Path) -> dict[tuple, dict[str, Any]]:
	run_dir = Path(run_dir)
	grades = {}
	gp = run_dir / "grades.json"
	if gp.exists():
		for g in json.loads(gp.read_text(encoding="utf-8")).get("grades", []):
			grades[Path(g["episode"]).name] = g
	out = {}
	for ep in sorted((run_dir / "student").glob("case-*")):
		if not (ep / "episode.json").exists():
			continue
		row = analyze_episode(ep, grades.get(ep.name))
		missing = []
		col = row["collisions"]
		if col is None:
			missing.append("collisions")
		tail_path = ep / "grasp_tail.json"
		tail = None
		if tail_path.exists():
			tail = json.loads(tail_path.read_text(encoding="utf-8")).get("first_grasp")
		else:
			missing.append("grasp_tail")
		if not (ep / "branch-0" / "actions.npz").exists():
			missing.append("actions")
		pin_path = ep / "rgb_pin.json"
		pin = json.loads(pin_path.read_text(encoding="utf-8")) if pin_path.exists() else None
		row.update(
			first_clamp=bool((row["first_grasp"] or {}).get("clamped")),
			success=row["verdict"] == "SUCCESS",
			collision=None if col is None else bool(col["robot_contact_frames"] + col["jaw_env_contact_frames"]),
			tail=tail,
			pin=pin,
			missing=missing,
			dir=str(ep),
		)
		out[_key(row["case"])] = row
	return out


def _actions(ep_dir: str):
	with np.load(Path(ep_dir) / "branch-0" / "actions.npz", allow_pickle=False) as z:
		return z["action_f64"].copy(), z["state_f64"].copy()


def expected_divergence(tail: dict | None) -> int | None:
	"""Episode frame at which an intervention arm may first issue a different command (None: never)."""
	if not tail:
		return None
	for e in tail.get("events", []):
		if e["kind"] in ("expert_start", "wait_start"):
			return int(e["frame"])
	return None


def allowed_divergence(arm: dict) -> tuple[int | None, str]:
	"""Episode frame from which the arm may differ from A (None: never). r3: the arm's RGB-pin hand-off
	(``vla/rgb_pin.py``); without a pin log, where its own intervention started."""
	pin = arm.get("pin")
	if pin and pin.get("role") == "pin":
		return pin.get("handoff_frame"), "rgb_pin"
	return expected_divergence(arm.get("tail")), "intervention"


def prefix_check(ref: dict, arm: dict) -> dict[str, Any]:
	"""Exact command comparison up to the arm's allowed divergence frame (the whole episode when there is none);
	max abs state / command error before it."""
	try:
		a_act, a_state = _actions(ref["dir"])
		b_act, b_state = _actions(arm["dir"])
	except (OSError, KeyError):
		return {"status": "missing"}
	n = min(len(a_act), len(b_act))
	diff = np.flatnonzero(np.any(a_act[:n] != b_act[:n], axis=1))
	first = int(diff[0]) if len(diff) else (None if len(a_act) == len(b_act) else n)
	allowed, basis = allowed_divergence(arm)
	upto = n if allowed is None else min(n, allowed)
	cmd_err = float(np.max(np.abs(a_act[:upto] - b_act[:upto]))) if upto else 0.0
	state_err = float(np.max(np.abs(a_state[:upto] - b_state[:upto]))) if upto else 0.0
	early = first is not None and (allowed is None or first < allowed)
	ref_t, arm_t = ref.get("tail") or {}, arm.get("tail") or {}
	same_switch = (ref_t.get("decision"), (ref_t.get("switch") or {}).get("frame")) == (
		arm_t.get("decision"),
		(arm_t.get("switch") or {}).get("frame"),
	)
	pin = arm.get("pin") or {}
	intervened = expected_divergence(arm_t or None)
	return {
		"status": "early_divergence" if early and cmd_err > TOL_RAD else ("ok" if not early else "within_tol"),
		"first_command_divergence": first,
		"allowed_divergence": allowed,
		"allowed_basis": basis,
		"intervention_frame": intervened,
		# an intervention must start exactly at the hand-off (none without one)
		"handoff_consistent": basis != "rgb_pin" or intervened is None or intervened == allowed,
		"pin_mismatch": pin.get("live_from") if pin.get("mismatch") else None,
		"max_command_err_before": cmd_err,
		"max_state_err_before": state_err,
		"same_switch_decision": same_switch,
	}


def first_state_divergence(ref: dict, arm: dict) -> int | None:
	try:
		_, a = _actions(ref["dir"])
		_, b = _actions(arm["dir"])
	except (OSError, KeyError):
		return None
	n = min(len(a), len(b))
	d = np.flatnonzero(np.any(a[:n] != b[:n], axis=1))
	return int(d[0]) if len(d) else (None if len(a) == len(b) else n)


def analyse(runs: dict[str, Path]) -> dict[str, Any]:
	arms = {name: load_arm(path) for name, path in runs.items()}
	keys = sorted(set.intersection(*(set(v) for v in arms.values()))) if arms else []
	union = sorted(set.union(*(set(v) for v in arms.values()))) if arms else []
	complete = keys == union and len(keys) > 0
	stats: dict[str, Any] = {
		"runs": {k: str(v) for k, v in runs.items()},
		"cases": len(keys),
		"complete": complete,
		"incomplete_cases": [list(k) for k in union if k not in keys],
	}
	manifests = {}
	for name, path in runs.items():
		mp = Path(path) / "run_manifest.json"
		manifests[name] = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else None
	stats["manifests"] = {
		k: None
		if m is None
		else {"identity": m.get("identity"), "code": m.get("code"), "mode": m.get("grasp_tail_mode")}
		for k, m in manifests.items()
	}
	stats["totals"] = {
		name: {e: sum(bool(arms[name][k][e]) for k in keys if arms[name][k][e] is not None) for e in ENDPOINTS}
		| {
			"collision_graded": sum(arms[name][k]["collision"] is not None for k in keys),
			"false_normal": sum(arms[name][k]["outcome"] == "detector_false_normal" for k in keys),
		}
		for name in arms
	}
	pairs = {}
	for a, b in PAIRS:
		if a not in arms or b not in arms:
			continue
		res = {}
		for e in ENDPOINTS:
			ks = [k for k in keys if arms[a][k][e] is not None and arms[b][k][e] is not None]
			cell = four_cell([bool(arms[a][k][e]) for k in ks], [bool(arms[b][k][e]) for k in ks])
			cell["only_first_cases"] = [list(k) for k in ks if arms[a][k][e] and not arms[b][k][e]]
			cell["only_second_cases"] = [list(k) for k in ks if arms[b][k][e] and not arms[a][k][e]]
			res[e] = cell
		new_clamp = {tuple(k) for k in res["first_clamp"]["only_second_cases"]}
		new_lift = {tuple(k) for k in res["lifted"]["only_second_cases"]}
		res["lift_conversion"] = {
			"new_first_clamps": len(new_clamp),
			"new_first_clamps_also_new_lifts": len(new_clamp & new_lift),
			"cases": [{"case": list(k), "new_lift": k in new_lift} for k in sorted(new_clamp)],
		}
		checks = {}
		for k in keys:
			if a == "act" and b in ("act-rep", "act-wait", "expert-once"):
				checks[str(list(k))] = prefix_check(arms[a][k], arms[b][k])
				checks[str(list(k))]["first_state_divergence"] = first_state_divergence(arms[a][k], arms[b][k])
		if checks:
			res["prefix_checks"] = checks
			res["prefix_summary"] = dict(Counter(c["status"] for c in checks.values()))
			res["switch_mismatch_cases"] = [k for k, c in checks.items() if not c.get("same_switch_decision", True)]
			res["handoff_inconsistent_cases"] = [k for k, c in checks.items() if not c.get("handoff_consistent", True)]
			res["pin_mismatch_cases"] = [k for k, c in checks.items() if c.get("pin_mismatch")]
			res["identical_episodes"] = sum(c["first_state_divergence"] is None for c in checks.values())
		pairs[f"{a} vs {b}"] = res
	stats["pairs"] = pairs
	triggers = {}
	for name in arms:
		dec = Counter(((arms[name][k]["tail"] or {}).get("decision") or "missing") for k in keys)
		trig = [k for k in keys if (arms[name][k]["tail"] or {}).get("decision") == "triggered"]
		outcomes = Counter((arms[name][k]["tail"] or {}).get("outcome") for k in trig)
		waits = [int((arms[name][k]["tail"] or {}).get("wait_frames") or 0) for k in trig]
		triggers[name] = {
			"decisions": dict(dec),
			"triggered": len(trig),
			"triggered_first_clamp": sum(arms[name][k]["first_clamp"] for k in trig),
			"untriggered_first_clamp": sum(arms[name][k]["first_clamp"] for k in keys if k not in trig),
			"untriggered": len(keys) - len(trig),
			"tail_outcomes": {str(k): v for k, v in outcomes.items()},
			"wait_frames": {"n_waited": sum(w > 0 for w in waits), "total": sum(waits), "max": max(waits, default=0)},
		}
	stats["triggers"] = triggers
	pins = {}
	for name in arms:
		logs = [arms[name][k]["pin"] for k in keys if arms[name][k]["pin"]]
		if not logs:
			continue
		pins[name] = {
			"role": sorted({p_["role"] for p_ in logs}),
			"episodes": len(logs),
			"requests": sum(p_["requests"] for p_ in logs),
			"pinned": sum(p_.get("pinned", 0) for p_ in logs),
			"pinned_with_own_render_different": sum(p_.get("rgb_diff_requests", 0) for p_ in logs),
			"handoff_reasons": dict(Counter(str(p_.get("handoff_reason")) for p_ in logs if p_["role"] == "pin")),
			"live_from": dict(
				Counter(str((p_.get("live_from") or {}).get("reason")) for p_ in logs if p_["role"] == "pin")
			),
			"mismatch_cases": [list(k) for k in keys if (arms[name][k]["pin"] or {}).get("mismatch")],
		}
	stats["rgb_pin"] = pins
	stats["rules"] = action_rules(stats)
	return {"stats": stats, "arms": arms, "keys": keys}


def action_rules(stats: dict[str, Any]) -> dict[str, Any]:
	"""The pre-registered action rules of section 5 (net gains in cases out of the paired layouts)."""
	p = stats["pairs"]
	out: dict[str, Any] = {"net_rule_cases": NET_RULE}
	prefix_ok = all(
		(p.get(k) or {}).get("prefix_summary", {}).get("early_divergence", 0) == 0
		and not (p.get(k) or {}).get("handoff_inconsistent_cases")
		for k in ("act vs act-rep", "act vs act-wait", "act vs expert-once")
	)
	out["data_complete"] = stats["complete"]
	out["prefix_ok"] = prefix_ok
	bw = p.get("act-wait vs expert-once")
	if bw:
		gain = bw["first_clamp"]["net_cases"]
		out["B_minus_Await"] = {
			"first_clamp_net": gain,
			"meets_net": gain >= NET_RULE,
			"lift_conversion": bw["lift_conversion"],
			"success_net": bw["success"]["net_cases"],
			"collision_net": bw["collision"]["net_cases"],
			"supports_alignment_demos": bool(
				stats["complete"]
				and prefix_ok
				and gain >= NET_RULE
				and bw["success"]["net_cases"] >= 0
				and bw["collision"]["net_cases"] <= 0
			),
		}
	aw = p.get("act vs act-wait")
	if aw:
		gain = aw["first_clamp"]["net_cases"]
		out["Await_minus_A"] = {
			"first_clamp_net": gain,
			"meets_net": gain >= NET_RULE,
			"lift_net": aw["lifted"]["net_cases"],
			"collision_net": aw["collision"]["net_cases"],
			"supports_wait_rule": bool(
				stats["complete"]
				and prefix_ok
				and gain >= NET_RULE
				and aw["lifted"]["net_cases"] >= 0
				and aw["collision"]["net_cases"] <= 0
			),
		}
	rep = p.get("act vs act-rep")
	if rep:
		out["A_vs_Arep_flips"] = {e: rep[e]["only_first"] + rep[e]["only_second"] for e in ENDPOINTS}
	return out


def write_csv(path: Path, arms: dict, keys: list, stats: dict) -> None:
	checks = {}
	for pair, res in stats["pairs"].items():
		for k, c in (res.get("prefix_checks") or {}).items():
			checks[(pair, k)] = c
	cols = [
		"seed",
		"positive",
		"face",
		"arm",
		"episode_dir",
		"decision",
		"switch_frame",
		"switch_segment_frame",
		"err_xy_mm",
		"err_roll_rad",
		"near_xy_ok",
		"near_roll_ok",
		"tail_outcome",
		"wait_frames",
		"sources",
		"first_clamp",
		"lifted",
		"success",
		"collision",
		"first_failure",
		"outcome",
		"missing",
		"prefix_vs_act",
		"first_divergence_vs_act",
	]
	with open(path, "w", newline="", encoding="utf-8") as f:
		w = csv.writer(f)
		w.writerow(cols)
		for k in keys:
			for name, rows in arms.items():
				r = rows[k]
				t = r["tail"] or {}
				s = t.get("switch") or {}
				pc = checks.get((f"act vs {name}", str(list(k)))) or {}
				w.writerow(
					[
						k[0],
						k[1],
						k[2],
						name,
						r["dir"],
						t.get("decision"),
						s.get("frame"),
						s.get("segment_frame"),
						None if s.get("err_xy_m") is None else round(s["err_xy_m"] * 1000, 3),
						None if s.get("err_roll_rad") is None else round(s["err_roll_rad"], 4),
						s.get("near_xy_ok"),
						s.get("near_roll_ok"),
						t.get("outcome"),
						t.get("wait_frames"),
						json.dumps(t.get("source_counts"), sort_keys=True),
						r["first_clamp"],
						r["lifted"],
						r["success"],
						r["collision"],
						(r["first_failure"] or {}).get("code"),
						r["outcome"],
						";".join(r["missing"]),
						pc.get("status"),
						pc.get("first_command_divergence"),
					]
				)


def markdown(stats: dict[str, Any]) -> str:
	L = ["# First-grasp tail: paired statistics", ""]
	L.append(f"Paired layouts: {stats['cases']} (complete: {stats['complete']}).")
	L.append("")
	L.append("| arm | first clamp | lifted | success | collision eps | graded | false normal | triggered |")
	L.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
	for name, t in stats["totals"].items():
		trig = stats["triggers"].get(name, {}).get("triggered")
		L.append(
			f"| {name} | {t['first_clamp']} | {t['lifted']} | {t['success']} | {t['collision']} | "
			f"{t['collision_graded']} | {t['false_normal']} | {trig} |"
		)
	L.append("")
	for pair, res in stats["pairs"].items():
		L.append(f"## {pair}")
		L.append("")
		L.append("| endpoint | both | only first | only second | neither | net | 95% CI | McNemar p |")
		L.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
		for e in ENDPOINTS:
			c = res[e]
			ci = c["ci95"]
			L.append(
				f"| {e} | {c['both']} | {c['only_first']} | {c['only_second']} | {c['neither']} | "
				f"{c['net_cases']:+d}/{c['n']} | [{ci[0]:+.3f}, {ci[1]:+.3f}] | {c['mcnemar_p']:.4f} |"
			)
		lc = res["lift_conversion"]
		L.append("")
		L.append(
			f"Lift conversion: {lc['new_first_clamps_also_new_lifts']} of {lc['new_first_clamps']} new first clamps "
			f"are also new lifts."
		)
		if "prefix_summary" in res:
			L.append(
				f"Prefix checks: {res['prefix_summary']}; switch mismatches: {res['switch_mismatch_cases']}; "
				f"hand-off inconsistent: {res['handoff_inconsistent_cases']}; pin mismatches: {res['pin_mismatch_cases']}."
			)
			L.append(f"Bitwise identical episodes (state): {res['identical_episodes']} of {stats['cases']}.")
		L.append("")
	if stats.get("rgb_pin"):
		L.append("## RGB pin (shared prefix)")
		L.append("")
		for name, t in stats["rgb_pin"].items():
			L.append(f"- {name}: {json.dumps(t, sort_keys=True)}")
		L.append("")
	L.append("## Switch decisions")
	L.append("")
	for name, t in stats["triggers"].items():
		L.append(f"- {name}: {t['decisions']}; tail outcomes {t['tail_outcomes']}; waits {t['wait_frames']}")
	L.append("")
	L.append("## Action rules")
	L.append("")
	L.append("```json")
	L.append(json.dumps(stats["rules"], indent=1))
	L.append("```")
	return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--run", action="append", required=True, help="arm=dir (arms: act, act-rep, act-wait, expert-once)")
	ap.add_argument("--out", type=Path, required=True)
	args = ap.parse_args(argv)
	runs = {}
	for item in args.run:
		name, _, path = item.partition("=")
		if name not in ARMS or not path:
			raise SystemExit(f"--run must be arm=dir with arm in {ARMS}: {item!r}")
		runs[name] = Path(path)
	res = analyse(runs)
	args.out.mkdir(parents=True, exist_ok=True)
	write_csv(args.out / "paired_cases.csv", res["arms"], res["keys"], res["stats"])
	(args.out / "paired_stats.json").write_text(json.dumps(res["stats"], indent=1) + "\n", encoding="utf-8")
	(args.out / "paired_stats.md").write_text(markdown(res["stats"]), encoding="utf-8")
	print(markdown(res["stats"]))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
