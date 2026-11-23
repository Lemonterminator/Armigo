#!/usr/bin/env python
"""Summarize a bowl_swap_probe.py run (docs/vla/BOWL_SWAP_PROBE_2026-11-23.md section 3).

Writes ``bowl_swap.json`` and ``bowl_swap.md`` next to the probe's ``cases`` directory.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

BOWL_R = 0.10
OTHER = {"green": "pink", "pink": "green"}


def _dist(a, b) -> float:
	return math.hypot(a[0] - b[0], a[1] - b[1])


def _angle(xy) -> float:
	return math.atan2(xy[1], xy[0])


def arm_dest(arm: dict, bowls: dict) -> dict:
	"""Which bowl the arm's segment-end TCP is nearer to; None when the brick was lost or the segment failed."""
	tcp = arm["tcp_end"]
	d = {k: _dist(tcp, v) for k, v in bowls.items()}
	near = min(d, key=d.get)
	ok = not arm["lost"] and arm["error"] is None
	return {"dest": near if ok else None, "reached": ok and d[near] <= BOWL_R, "dist": d, "lost": arm["lost"]}


def classify(probe: dict) -> dict:
	bowls = probe["bowls_true"]
	req = probe["requested"]
	arms = {a["arm"]: a for a in probe["arms"]}
	dest = {k: arm_dest(a, bowls) for k, a in arms.items()}
	r, s = dest["req"]["dest"], dest["swap"]["dest"]
	if r is None or s is None:
		cls = "other"
	elif r == req and s == OTHER[req]:
		cls = "follow"
	elif r == s:
		cls = "fixed"
	else:
		cls = "inverted"
	left = "green" if _angle(bowls["green"]) > _angle(bowls["pink"]) else "pink"
	start = probe["start"]["tcp"]
	nearer_start = min(bowls, key=lambda k: _dist(start, bowls[k]))
	r2 = dest["req2"]["dest"]
	return {
		"class": cls,
		"requested": req,
		"left_bowl": left,
		"nearer_start_bowl": nearer_start,
		"dest": {k: v["dest"] for k, v in dest.items()},
		"reached": {k: v["reached"] for k, v in dest.items()},
		"lost": {k: v["lost"] for k, v in dest.items()},
		"dist_to_target_m": {
			"req": dest["req"]["dist"][req],
			"req2": dest["req2"]["dist"][req],
			"swap": dest["swap"]["dist"][OTHER[req]],
		},
		"req_req2_agree": None if r is None or r2 is None else r == r2,
		"req_req2_tcp_m": _dist(arms["req"]["tcp_end"], arms["req2"]["tcp_end"]),
		"fixed_bowl": r if cls == "fixed" else None,
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
	# swap arm to the right bowl, by the colour it was asked for x the side of that bowl
	table = {}
	for c in per:
		if c["dest"]["swap"] is None:
			continue
		want = OTHER[c["requested"]]
		side = "left" if c["left_bowl"] == want else "right"
		key = f"{want}/{side}"
		t = table.setdefault(key, [0, 0])
		t[0] += c["dest"]["swap"] == want
		t[1] += 1
	req_table = {}
	for c in per:
		if c["dest"]["req"] is None:
			continue
		want = c["requested"]
		side = "left" if c["left_bowl"] == want else "right"
		t = req_table.setdefault(f"{want}/{side}", [0, 0])
		t[0] += c["dest"]["req"] == want
		t[1] += 1
	fixed = [c for c in per if c["class"] == "fixed"]
	fixed_by = {
		"colour": dict(Counter(c["fixed_bowl"] for c in fixed)),
		"side": dict(Counter("left" if c["fixed_bowl"] == c["left_bowl"] else "right" for c in fixed)),
		"nearer_start": dict(Counter(c["fixed_bowl"] == c["nearer_start_bowl"] for c in fixed)),
	}
	wrong_req = [c for c in per if c["dest"]["req"] is not None and c["dest"]["req"] != c["requested"]]
	wrong_req_swap = dict(
		Counter(
			"none"
			if c["dest"]["swap"] is None
			else ("same_wrong" if c["dest"]["swap"] == c["dest"]["req"] else "moved")
			for c in wrong_req
		)
	)
	frac = {k: classes.get(k, 0) / n if n else None for k in ("follow", "fixed", "inverted", "other")}
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
	return {
		"cases": len(rows),
		"probed": n,
		"not_probed": [r["case"] for r in rows if r["probe"] is None],
		"classes": dict(classes),
		"fractions": frac,
		"req_req2_agree": [sum(agree), len(agree)],
		"req_req2_tcp_median_m": _median([c["req_req2_tcp_m"] for c in per]),
		"swap_correct_by_target": table,
		"req_correct_by_target": req_table,
		"fixed_by": fixed_by,
		"req_wrong": len(wrong_req),
		"req_wrong_then_swap": wrong_req_swap,
		"reached": {k: sum(c["reached"][k] for c in per) for k in ("req", "req2", "swap")},
		"lost": {k: sum(c["lost"][k] for c in per) for k in ("req", "req2", "swap")},
		"median_dist_to_target_m": {
			k: _median([c["dist_to_target_m"][k] for c in per if c["dest"][k] is not None])
			for k in ("req", "req2", "swap")
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
	for k in ("follow", "fixed", "inverted", "other"):
		v = f[k]
		lines.append(f"| {k} | {s['classes'].get(k, 0)} | {'' if v is None else f'{v:.2f}'} |")
	a, b = s["req_req2_agree"]
	lines += [
		"",
		f"req vs req2 same bowl: {a}/{b}; median end-TCP distance {s['req_req2_tcp_median_m']:.3f} m",
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
