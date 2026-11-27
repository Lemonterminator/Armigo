#!/usr/bin/env python
"""Summarize a bowl_swap_probe.py run (docs/vla/BOWL_SWAP_PROBE_2026-11-23.md section 3).

Supports arbitrary K >= 2 bowls and any bowl colour palette (v2 green/pink and v3 K-bowl scenes).
Writes ``bowl_swap.json`` and ``bowl_swap.md`` next to the probe's ``cases`` directory.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

BOWL_R = 0.10


def _dist(a, b) -> float:
	return math.hypot(a[0] - b[0], a[1] - b[1])


def _angle(xy) -> float:
	return math.atan2(xy[1], xy[0])


def _arm_target(arm: dict, probe: dict) -> str:
	"""Target bowl colour requested for ``arm``, with clean fallback for legacy 2-bowl records."""
	bowl = arm.get("bowl")
	if bowl is not None:
		return str(bowl)
	req = str(probe["requested"])
	name = str(arm["arm"])
	if name in ("req", "req2"):
		return req
	if name == "swap":
		others = [k for k in probe["bowls_true"] if k != req]
		if len(others) == 1:
			return str(others[0])
		raise ValueError(f"legacy swap arm without 'bowl' field is ambiguous for {len(probe['bowls_true'])} bowls")
	raise KeyError(f"arm {name!r} is missing required 'bowl' target field")


def _bowl_sides(bowls: dict) -> tuple[list[str], dict[str, str]]:
	"""Order bowls left-to-right by descending robot-base angle and map each bowl to a side label."""
	ordered = sorted(bowls, key=lambda k: (-_angle(bowls[k]), str(k)))
	n = len(ordered)
	if n == 0:
		raise ValueError("bowls_true cannot be empty")
	if n == 1:
		return ordered, {ordered[0]: "left"}
	if n == 2:
		return ordered, {ordered[0]: "left", ordered[1]: "right"}
	sides: dict[str, str] = {}
	for i, name in enumerate(ordered):
		if i == 0:
			sides[name] = "left"
		elif i == n - 1:
			sides[name] = "right"
		else:
			sides[name] = "middle"
	return ordered, sides


def arm_dest(arm: dict, bowls: dict) -> dict:
	"""Which bowl the arm's segment-end TCP is nearer to; None when the brick was lost or the segment failed."""
	tcp = arm["tcp_end"]
	d = {k: _dist(tcp, v) for k, v in bowls.items()}
	near = min(d, key=d.get)
	ok = not arm["lost"] and arm["error"] is None
	return {"dest": near if ok else None, "reached": ok and d[near] <= BOWL_R, "dist": d, "lost": arm["lost"]}


def classify(probe: dict) -> dict:
	bowls = probe["bowls_true"]
	if len(bowls) < 2:
		raise ValueError(f"expected at least 2 bowls in bowls_true, got {len(bowls)}")
	req = str(probe["requested"])
	arm_list = list(probe["arms"])
	arm_order = [str(a["arm"]) for a in arm_list]
	arms = {str(a["arm"]): a for a in arm_list}
	targets = {str(a["arm"]): _arm_target(a, probe) for a in arm_list}
	dest = {k: arm_dest(a, bowls) for k, a in arms.items()}

	# Directed arms exclude the duplicate noise-control arm ``req2``.
	directed = [name for name in arm_order if name != "req2"]
	directed_dests = [dest[k]["dest"] for k in directed]
	directed_targets = [targets[k] for k in directed]

	if not directed or any(d is None for d in directed_dests):
		cls = "other"
	elif all(d == want for d, want in zip(directed_dests, directed_targets, strict=True)):
		cls = "follow"
	elif len(set(directed_targets)) >= 2 and len(set(directed_dests)) == 1:
		cls = "fixed"
	elif len(directed) == 2 and directed_dests[0] == directed_targets[1] and directed_dests[1] == directed_targets[0]:
		cls = "inverted"
	else:
		cls = "mixed"

	ordered, bowl_side = _bowl_sides(bowls)
	left = ordered[0]
	start = probe["start"]["tcp"]
	nearer_start = min(bowls, key=lambda k: _dist(start, bowls[k]))

	if "req" in arms and "req2" in arms:
		r, r2 = dest["req"]["dest"], dest["req2"]["dest"]
		req_req2_agree: bool | None = None if r is None or r2 is None else (r == r2)
		req_req2_tcp_m: float | None = _dist(arms["req"]["tcp_end"], arms["req2"]["tcp_end"])
	else:
		req_req2_agree = None
		req_req2_tcp_m = None

	return {
		"class": cls,
		"n_bowls": len(bowls),
		"requested": req,
		"arm_targets": targets,
		"directed_arms": directed,
		"left_bowl": left,
		"bowl_order": ordered,
		"bowl_side": bowl_side,
		"nearer_start_bowl": nearer_start,
		"dest": {k: v["dest"] for k, v in dest.items()},
		"reached": {k: v["reached"] for k, v in dest.items()},
		"lost": {k: v["lost"] for k, v in dest.items()},
		"dist_to_target_m": {k: dest[k]["dist"][targets[k]] for k in arm_order},
		"req_req2_agree": req_req2_agree,
		"req_req2_tcp_m": req_req2_tcp_m,
		"fixed_bowl": directed_dests[0] if cls == "fixed" else None,
	}


def _median(xs):
	xs = sorted(xs)
	if not xs:
		return None
	n = len(xs)
	return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def summarize(rows: list[dict]) -> dict:
	probed = [r for r in rows if r["probe"] is not None]
	per = []
	for r in probed:
		c = classify(r["probe"])
		c["case"] = r["case"]
		c["index"] = r["index"]
		per.append(c)
	n = len(per)
	classes = Counter(c["class"] for c in per)
	agree = [c["req_req2_agree"] for c in per if c["req_req2_agree"] is not None]

	arm_names: list[str] = ["req", "req2", "swap"]
	for c in per:
		for name in c["dest"]:
			if name not in arm_names:
				arm_names.append(name)

	# Per-target (colour/side) accuracy for req, swap, extra arms, and all directed arms combined.
	swap_table: dict[str, list[int]] = {}
	req_table: dict[str, list[int]] = {}
	extra_table: dict[str, list[int]] = {}
	directed_table: dict[str, list[int]] = {}
	for c in per:
		for arm_name in c["directed_arms"]:
			d = c["dest"].get(arm_name)
			if d is None:
				continue
			want = c["arm_targets"][arm_name]
			side = c["bowl_side"][want]
			key = f"{want}/{side}"
			ok = int(d == want)
			dt = directed_table.setdefault(key, [0, 0])
			dt[0] += ok
			dt[1] += 1
			if arm_name == "req":
				t = req_table.setdefault(key, [0, 0])
				t[0] += ok
				t[1] += 1
			elif arm_name == "swap":
				t = swap_table.setdefault(key, [0, 0])
				t[0] += ok
				t[1] += 1
			else:
				t = extra_table.setdefault(key, [0, 0])
				t[0] += ok
				t[1] += 1

	fixed = [c for c in per if c["class"] == "fixed"]
	fixed_by = {
		"colour": dict(Counter(c["fixed_bowl"] for c in fixed)),
		"side": dict(Counter(c["bowl_side"][c["fixed_bowl"]] for c in fixed)),
		"nearer_start": dict(Counter(c["fixed_bowl"] == c["nearer_start_bowl"] for c in fixed)),
	}
	wrong_req = [c for c in per if c["dest"].get("req") is not None and c["dest"]["req"] != c["arm_targets"].get("req")]
	wrong_req_swap = dict(
		Counter(
			"none"
			if c["dest"].get("swap") is None
			else ("same_wrong" if c["dest"]["swap"] == c["dest"]["req"] else "moved")
			for c in wrong_req
		)
	)
	class_keys = ("follow", "fixed", "inverted", "mixed", "other")
	frac = {k: classes.get(k, 0) / n if n else None for k in class_keys}
	agree_rate = sum(agree) / len(agree) if agree else None
	verdict = []
	if agree_rate is not None and agree_rate < 0.9:
		verdict.append("noise_high")
	if n:
		if frac["follow"] >= 0.85:
			verdict.append("D1_follows")
		if frac["fixed"] >= 0.30:
			col = max(fixed_by["colour"].values()) / len(fixed)
			side = max(fixed_by["side"].values()) / len(fixed)
			near = max(fixed_by["nearer_start"].values()) / len(fixed)
			verdict.append("D2_ignores")
			if col >= 0.8:
				verdict.append("D2_colour_prior")
			if side >= 0.8 or near >= 0.8:
				verdict.append("D2_spatial_default")
		if frac["inverted"] >= 0.15:
			verdict.append("D3_inverted")
		if frac["mixed"] >= 0.15:
			verdict.append("D3_mixed")
	return {
		"cases": len(rows),
		"probed": n,
		"not_probed": [r["case"] for r in rows if r["probe"] is None],
		"classes": dict(classes),
		"fractions": frac,
		"req_req2_agree": [sum(agree), len(agree)],
		"req_req2_tcp_median_m": _median([c["req_req2_tcp_m"] for c in per if c["req_req2_tcp_m"] is not None]),
		"swap_correct_by_target": swap_table,
		"req_correct_by_target": req_table,
		"extra_correct_by_target": extra_table,
		"directed_correct_by_target": directed_table,
		"fixed_by": fixed_by,
		"req_wrong": len(wrong_req),
		"req_wrong_then_swap": wrong_req_swap,
		"reached": {k: sum(bool(c["reached"].get(k, False)) for c in per) for k in arm_names},
		"lost": {k: sum(bool(c["lost"].get(k, False)) for c in per) for k in arm_names},
		"median_dist_to_target_m": {
			k: _median([c["dist_to_target_m"][k] for c in per if c["dest"].get(k) is not None]) for k in arm_names
		},
		"verdict": verdict or ["undetermined"],
		"per_case": per,
	}


def markdown(s: dict) -> str:
	f = s["fractions"]
	lines = [
		"# Bowl swap probe",
		"",
		f"Probed {s['probed']} / {s['cases']} cases (no first carry: {len(s['not_probed'])}).",
		"",
		"| class | n | fraction |",
		"| --- | --- | --- |",
	]
	for k in ("follow", "fixed", "inverted", "mixed", "other"):
		v = f.get(k)
		lines.append(f"| {k} | {s['classes'].get(k, 0)} | {'' if v is None else f'{v:.2f}'} |")
	a, b = s["req_req2_agree"]
	med_tcp = s["req_req2_tcp_median_m"]
	med_str = "n/a" if med_tcp is None else f"{med_tcp:.3f} m"
	has_extra = bool(s.get("extra_correct_by_target"))
	if has_extra:
		lines += [
			"",
			f"req vs req2 same bowl: {a}/{b}; median end-TCP distance {med_str}",
			"",
			"| target (colour/side) | req to target | swap to target | extra to target | all directed |",
			"| --- | --- | --- | --- | --- |",
		]
		all_keys = sorted(
			set(s["req_correct_by_target"])
			| set(s["swap_correct_by_target"])
			| set(s.get("extra_correct_by_target", {}))
		)
		for key in all_keys:
			r = s["req_correct_by_target"].get(key, [0, 0])
			w = s["swap_correct_by_target"].get(key, [0, 0])
			e = s["extra_correct_by_target"].get(key, [0, 0])
			d = s["directed_correct_by_target"].get(key, [0, 0])
			lines.append(f"| {key} | {r[0]}/{r[1]} | {w[0]}/{w[1]} | {e[0]}/{e[1]} | {d[0]}/{d[1]} |")
	else:
		lines += [
			"",
			f"req vs req2 same bowl: {a}/{b}; median end-TCP distance {med_str}",
			"",
			"| target (colour/side) | req to target | swap to target |",
			"| --- | --- | --- |",
		]
		for key in sorted(set(s["req_correct_by_target"]) | set(s["swap_correct_by_target"])):
			r = s["req_correct_by_target"].get(key, [0, 0])
			w = s["swap_correct_by_target"].get(key, [0, 0])
			lines.append(f"| {key} | {r[0]}/{r[1]} | {w[0]}/{w[1]} |")
	lines += [
		"",
		(
			f"fixed by colour {s['fixed_by']['colour']}, by side {s['fixed_by']['side']}, "
			f"fixed bowl nearer the carry start {s['fixed_by']['nearer_start']}"
		),
		f"req to the wrong bowl: {s['req_wrong']}; swap then {s['req_wrong_then_swap']}",
		f"reached (within BOWL_R): {s['reached']}; lost: {s['lost']}",
		f"median distance to target: {s['median_dist_to_target_m']}",
		"",
		f"**verdict: {', '.join(s['verdict'])}**",
		"",
	]
	return "\n".join(lines)


def main(argv=None) -> int:
	p = argparse.ArgumentParser(description=__doc__)
	p.add_argument("run", type=Path)
	args = p.parse_args(argv)
	rows = [json.loads(f.read_text(encoding="utf-8")) for f in sorted((args.run / "cases").glob("case-*/probe.json"))]
	s = summarize(rows)
	(args.run / "bowl_swap.json").write_text(json.dumps(s, indent=1), encoding="utf-8")
	md = markdown(s)
	(args.run / "bowl_swap.md").write_text(md, encoding="utf-8")
	print(md)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
