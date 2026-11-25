"""Evaluate wrist re-localisation methods on the checkpoint-stratum set (brickseg_prereg section 4).

Every frame of ``make_wrist_checkpoint_set.py`` output is localised with the closed-loop entry point
``localize_from_wrist`` and the inputs the agent had at that checkpoint (believed camera, believed size, prior from the
top-camera estimate). Methods: ``lab``, ``silhouette`` and, with ``--onnx``, ``unet``.

Reports per method and split: success rate, xy / yaw error (median, p90, p95, max), fixes worse than 3 mm or 3°, the
prior's own error on the same frames, failure reasons; then per checkpoint height and per same-colour confuser bucket
(the most severe bucket among clutter bricks and distractor boxes with rendered ΔE below ``--confuser-de``).

Gates (``checkpoint-val`` / ``checkpoint-ood``): success ≥ 98 %, median xy ≤ 1.5 mm and yaw ≤ 1.5° on val for each
candidate; on ood every ``unet`` candidate must not be worse than ``silhouette`` (paired: success rate, median xy and
yaw error). ``lab`` is reported as a reference. A stress set (``"stress": true`` frames) is reported only.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from digital_twin_armigo.agent.perception import yaw_error_deg
from digital_twin_armigo.agent.wrist_localize import localize_from_wrist

BUCKET_ORDER = ("none", "out_of_view", "in_view_outside_gate", "in_gate", "touching")
GATE_SUCCESS = 0.98
GATE_XY_MM = 1.5
GATE_YAW_DEG = 1.5
WRONG_MM = 3.0
WRONG_DEG = 3.0


def iter_frames(root: Path):
	for shard in sorted(root.glob("shard_*.npz")):
		z = np.load(shard)
		for i in range(len(z["meta"])):
			yield shard.name, i, z["rgb"][i], json.loads(str(z["meta"][i]))


def confuser_bucket(meta: dict, max_de: float) -> str:
	worst = "none"
	for c in meta.get("confusers", []):
		if c["delta_e"] is None and c["bucket"] != "out_of_view":
			continue
		if c["bucket"] != "out_of_view" and c["delta_e"] > max_de:
			continue
		if BUCKET_ORDER.index(c["bucket"]) > BUCKET_ORDER.index(worst):
			worst = c["bucket"]
	return worst


INPUT_MODES = ("closed_loop", "oracle_camera")


def evaluate_frame(rgb, meta: dict, method: str, net=None, mode: str = "closed_loop") -> dict:
	"""``closed_loop``: the agent's inputs at the checkpoint (believed wrist mount, believed size and height).
	``oracle_camera``: the actual wrist camera pose, true size and true height, so only the segmentation and fit
	remain; the difference to ``closed_loop`` is the belief error (wrist mount offset, height, size)."""
	k = np.asarray(meta["k"])
	if mode == "closed_loop":
		w2c, size, top_z = np.asarray(meta["world_from_cam"]), tuple(meta["believed_size"]), float(meta["top_z"])
	else:
		w2c, size, top_z = (
			np.asarray(meta["world_from_cam_actual"]),
			tuple(meta["brick_size"]),
			float(meta["true_top_z"]),
		)
	rot = np.asarray(meta["brick_rot"])
	truth_xy = np.asarray(meta["brick_pos"][:2])
	truth_yaw = math.atan2(rot[1, 0], rot[0, 0])
	t0 = time.perf_counter()
	fix, why = localize_from_wrist(
		rgb,
		k,
		w2c,
		np.asarray(meta["prior_xy"]),
		float(meta["prior_yaw"]),
		size,
		top_z,
		segmenter="unet" if method == "unet" else method,
		net=net,
	)
	rec = {
		"method": method,
		"mode": mode,
		"found": fix is not None,
		"reason": why,
		"ms": (time.perf_counter() - t0) * 1000,
	}
	if fix is not None:
		rec["xy_mm"] = float(np.linalg.norm(fix.pos_xy - truth_xy) * 1000)
		rec["yaw_deg"] = float(yaw_error_deg(fix.yaw, truth_yaw, meta["brick_size"]))
	return rec


def _stats(vals):
	if not vals:
		return {"median": None, "p90": None, "p95": None, "max": None}
	a = np.asarray(vals)
	return {
		k: round(float(v), 3)
		for k, v in zip(("median", "p90", "p95", "max"), (np.median(a), *np.percentile(a, [90, 95]), a.max()))
	}


def summarise(rows: list[dict]) -> dict:
	n = len(rows)
	ok = [r for r in rows if r["found"]]
	return {
		"n": n,
		"found": len(ok),
		"success": round(len(ok) / n, 4) if n else None,
		"xy_mm": _stats([r["xy_mm"] for r in ok]),
		"yaw_deg": _stats([r["yaw_deg"] for r in ok]),
		"wrong_gt_3mm": sum(r["xy_mm"] > WRONG_MM for r in ok),
		"wrong_gt_3deg": sum(r["yaw_deg"] > WRONG_DEG for r in ok),
		"prior_xy_mm": _stats([r["prior_xy_mm"] for r in rows]),
		"reasons": dict(collections.Counter(r["reason"] for r in rows if not r["found"])),
		"ms_median": round(float(np.median([r["ms"] for r in rows])), 1) if rows else None,
	}


def paired_not_worse(cand: list[dict], ref: list[dict]) -> dict:
	by = {(r["shard"], r["index"]): r for r in ref}
	pairs = [(by[(c["shard"], c["index"])], c) for c in cand if (c["shard"], c["index"]) in by]
	rs, cs = summarise([p[0] for p in pairs]), summarise([p[1] for p in pairs])
	both = [(r, c) for r, c in pairs if r["found"] and c["found"]]

	def worse(a, b):
		return a is not None and b is not None and a > b

	out = {
		"n": len(pairs),
		"ref_success": rs["success"],
		"cand_success": cs["success"],
		"both_found": len(both),
		"paired_xy_diff_median_mm": round(float(np.median([c["xy_mm"] - r["xy_mm"] for r, c in both])), 3)
		if both
		else None,
		"paired_yaw_diff_median_deg": round(float(np.median([c["yaw_deg"] - r["yaw_deg"] for r, c in both])), 3)
		if both
		else None,
		"not_worse_rate": (cs["success"] or 0) >= (rs["success"] or 0),
		"not_worse_xy": not worse(cs["xy_mm"]["median"], rs["xy_mm"]["median"]),
		"not_worse_yaw": not worse(cs["yaw_deg"]["median"], rs["yaw_deg"]["median"]),
	}
	out["passed"] = out["not_worse_rate"] and out["not_worse_xy"] and out["not_worse_yaw"]
	return out


def evaluate(roots: list[Path], methods: list[str], onnx: Path | None, max_de: float) -> dict:
	net = None
	if "unet" in methods:
		from digital_twin_armigo.agent.brick_pose_nn import BrickSegNet

		net = BrickSegNet(onnx)
	rows: list[dict] = []
	for root in roots:
		for shard, i, rgb, meta in iter_frames(root):
			prior_err = float(np.linalg.norm(np.asarray(meta["prior_xy"]) - np.asarray(meta["brick_pos"][:2])) * 1000)
			base = {
				"set": root.name,
				"shard": f"{root.name}/{shard}",
				"index": i,
				"seed": meta["seed"],
				"split": meta["split"],
				"stress": bool(meta["stress"]),
				"height_m": meta["height_m"],
				"height_index": meta["height_index"],
				"bucket": confuser_bucket(meta, max_de),
				"prior_xy_mm": prior_err,
			}
			for mode in INPUT_MODES:
				for m in methods:
					rows.append(base | evaluate_frame(rgb, meta, m, net, mode))
	res: dict = {"methods": methods, "confuser_max_delta_e": max_de, "sets": [str(r) for r in roots], "groups": {}}
	keys = sorted({(r["stress"], r["split"], r["mode"]) for r in rows})
	for stress, split, mode in keys:
		name = f"{'stress' if stress else 'checkpoint'}-{'val' if split == 'val' else 'ood'}:{mode}"
		g = {"overall": {}, "by_height": {}, "by_bucket": {}}
		for m in methods:
			sel = [
				r
				for r in rows
				if r["stress"] == stress and r["split"] == split and r["mode"] == mode and r["method"] == m
			]
			g["overall"][m] = summarise(sel)
			for h in sorted({r["height_index"] for r in sel}):
				g["by_height"].setdefault(str(h), {})[m] = summarise([r for r in sel if r["height_index"] == h])
			for b in BUCKET_ORDER:
				bs = [r for r in sel if r["bucket"] == b]
				if bs:
					g["by_bucket"].setdefault(b, {})[m] = summarise(bs)
		res["groups"][name] = g
	gates = {}
	for mode in INPUT_MODES:
		val = res["groups"].get(f"checkpoint-val:{mode}", {}).get("overall", {})
		for m in methods:
			if m == "lab" or m not in val:
				continue
			gates[f"checkpoint-val:{mode}:{m}"] = _gate(val[m])
		if "unet" in methods and "silhouette" in methods:
			ood = [r for r in rows if not r["stress"] and r["split"] != "val" and r["mode"] == mode]
			gates[f"checkpoint-ood:{mode}:unet_vs_silhouette"] = paired_not_worse(
				[r for r in ood if r["method"] == "unet"], [r for r in ood if r["method"] == "silhouette"]
			)
	res["gates"] = gates
	res["rows"] = rows
	return res


def _gate(s: dict) -> dict:
	return {
		"success": s["success"],
		"xy_median_mm": s["xy_mm"]["median"],
		"yaw_median_deg": s["yaw_deg"]["median"],
		"passed": bool(
			s["success"] is not None
			and s["success"] >= GATE_SUCCESS
			and s["xy_mm"]["median"] is not None
			and s["xy_mm"]["median"] <= GATE_XY_MM
			and s["yaw_deg"]["median"] <= GATE_YAW_DEG
		),
	}


def render_markdown(res: dict) -> str:
	lines = ["# Wrist checkpoint-stratum evaluation", ""]
	for name, g in res["groups"].items():
		lines += [
			f"## {name}",
			"",
			"| method | n | success | xy med/p90/p95/max mm | yaw med/p90 ° | >3mm | >3° | prior xy med mm | reasons |",
			"|---|---:|---:|---|---|---:|---:|---:|---|",
		]
		for m, s in g["overall"].items():
			x, y = s["xy_mm"], s["yaw_deg"]
			lines.append(
				f"| {m} | {s['n']} | {s['success']:.3f} | {x['median']}/{x['p90']}/{x['p95']}/{x['max']} | {y['median']}/{y['p90']} "
				f"| {s['wrong_gt_3mm']} | {s['wrong_gt_3deg']} | {s['prior_xy_mm']['median']} | {s['reasons']} |"
			)
		lines += ["", "By checkpoint (0 = hover, then 0.050 / 0.045 / 0.040 / 0.035 / 0.030 m): success, xy median", ""]
		ms = list(g["overall"])
		lines += ["| height | " + " | ".join(ms) + " |", "|---|" + "---|" * len(ms)]
		for h, per in g["by_height"].items():
			lines.append(
				f"| {h} | " + " | ".join(f"{per[m]['success']:.3f} / {per[m]['xy_mm']['median']}" for m in ms) + " |"
			)
		lines += [
			"",
			f"By same-colour confuser bucket (ΔE ≤ {res['confuser_max_delta_e']}): n, success, xy median, >3mm",
			"",
		]
		lines += ["| bucket | " + " | ".join(ms) + " |", "|---|" + "---|" * len(ms)]
		for b, per in g["by_bucket"].items():
			lines.append(
				f"| {b} | "
				+ " | ".join(
					f"{per[m]['n']}, {per[m]['success']:.3f}, {per[m]['xy_mm']['median']}, {per[m]['wrong_gt_3mm']}"
					for m in ms
				)
				+ " |"
			)
		lines.append("")
	lines += ["## Gates", "", "```json", json.dumps(res["gates"], indent=1), "```", ""]
	return "\n".join(lines)


def main(argv=None) -> int:
	p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
	p.add_argument("--sets", nargs="+", type=Path, required=True, help="checkpoint-set directories (shards + manifest)")
	p.add_argument("--methods", nargs="+", default=["lab", "silhouette"], choices=["lab", "silhouette", "unet"])
	p.add_argument("--onnx", type=Path, default=None, help="BrickSegNet ONNX for --methods unet")
	p.add_argument(
		"--confuser-de", type=float, default=35.0, help="ΔE at or below which a confuser counts as same colour"
	)
	p.add_argument("--out", type=Path, required=True)
	a = p.parse_args(argv)
	if "unet" in a.methods and a.onnx is None:
		raise SystemExit("--methods unet needs --onnx")
	res = evaluate(a.sets, a.methods, a.onnx, a.confuser_de)
	a.out.mkdir(parents=True, exist_ok=True)
	(a.out / "rows.jsonl").write_text("\n".join(json.dumps(r) for r in res.pop("rows")) + "\n", encoding="utf-8")
	(a.out / "summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
	md = render_markdown(res)
	(a.out / "report.md").write_text(md, encoding="utf-8")
	sys.stdout.buffer.write(md.encode("utf-8") + b"\n")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
