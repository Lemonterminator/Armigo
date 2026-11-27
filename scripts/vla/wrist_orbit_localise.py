"""Offline check: does a concentric wrist-camera ring over the brick improve on the top-view brick estimate?

For each v3 case (cameras ``drift``, realistic sensing, bowls from the oracle) the twin runs ``prepare`` and the
top-view perception. The arm is then *placed* (kinematically, no physics step) at ring poses: jaws straight down over
the believed brick, the jaw tips ``tip`` above the believed top face, the wrist roll swept over its range so the wrist
camera circles the brick. ``--aim`` shifts the jaw axis per pose so the believed optical axis meets the believed top
face centre. Each wrist image is localised with ``localize_from_wrist`` (camera pose from joints and the *designed*
mount, as the agent sees it). The fused estimate (mean / median of the fixes) is scored against simulator truth next
to the top-view error. Writes ``<out>/results.jsonl`` and ``<out>/summary.json``.

	PYTHONPATH=src python scripts/vla/wrist_orbit_localise.py --out C:/scratch/reverse-act/orbit-loc --n 60
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
	sys.path.insert(0, str(ROOT / "src"))

TIPS_M = (0.03, 0.04, 0.06)
SEGMENTERS = ("lab", "silhouette")


def make_case(seed: int) -> list:
	return [seed, bool(seed % 2), seed % 6, {"scene": "v3", "split": "train", "cameras": "drift", "oracle": "bowls"}]


def run_case(args: tuple) -> dict:
	os.environ["CUDA_VISIBLE_DEVICES"] = ""
	seed, out, n_views = args
	from digital_twin_armigo.agent.perception import yaw_error_deg
	from digital_twin_armigo.agent.student_rollout import ExpertTracer
	from digital_twin_armigo.agent.twin_mujoco import sample_case
	from digital_twin_armigo.agent.twin_record import EpisodeTrace
	from digital_twin_armigo.agent.wrist_localize import believed_wrist_from_cam, localize_from_wrist, wrist_k
	from digital_twin_armigo.agent.wrist_ring import ring_poses
	from digital_twin_armigo.mujoco_cell._mj import mujoco

	case = make_case(seed)
	twin = ExpertTracer(sample_case(*case), None, EpisodeTrace(out / f"case-{seed}", video=False), sensing="realistic")
	try:
		if err := twin.prepare():
			return {"seed": seed, "error": err}
		if err := twin._perceive():
			return {"seed": seed, "error": err}
		cell, ex, est = twin.cell, twin.expert, twin.brick_est
		size = twin._believed_brick_size()
		top_z = float(est.pos[2] + size[2] / 2)
		true_xy = cell.brick_pos()[:2].copy()
		rot = cell.data.xmat[cell.brick_body].reshape(3, 3)
		true_yaw = math.atan2(rot[1, 0], rot[0, 0])
		true_size = tuple(float(v) for v in twin.layout.brick_size)
		row = {
			"seed": seed,
			"size_correct": bool(np.allclose(size, true_size)),
			"top_err_mm": float(np.linalg.norm(est.pos[:2] - true_xy) * 1000),
			"top_yaw_err_deg": float(yaw_error_deg(est.yaw, true_yaw, true_size)),
			"variants": {},
		}
		cid = mujoco.mj_name2id(cell.model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
		q_home = cell.joints()
		for tip in TIPS_M:
			for aim in (False, True):
				poses = ring_poses(ex, cell, est.pos[:2], top_z, tip, n_views, aim)
				shots = []
				for q in poses:
					cell.set_joints(q)
					rgb = cell.rgb("wrist_cam").copy()
					ex.kin.data.qpos[ex.kin.qadr] = q
					shots.append((rgb, believed_wrist_from_cam(cell, ex.kin.data)))
				k = wrist_k(float(cell.model.cam_fovy[cid]), shots[0][0].shape) if shots else None
				for seg in SEGMENTERS:
					fixes, reasons = [], {}
					for rgb, w2c in shots:
						fix, reason = localize_from_wrist(rgb, k, w2c, est.pos[:2], est.yaw, size, top_z, segmenter=seg)
						if fix is None:
							reasons[reason] = reasons.get(reason, 0) + 1
						else:
							fixes.append(fix)
					v = {"poses": len(poses), "fixes": len(fixes), "rejects": reasons}
					if fixes:
						xy = np.array([f.pos_xy for f in fixes])
						v["first_err_mm"] = float(np.linalg.norm(xy[0] - true_xy) * 1000)
						v["single_err_mm_median"] = float(np.median(np.linalg.norm(xy - true_xy, axis=1)) * 1000)
						v["mean_err_mm"] = float(np.linalg.norm(xy.mean(0) - true_xy) * 1000)
						v["median_err_mm"] = float(np.linalg.norm(np.median(xy, axis=0) - true_xy) * 1000)
						v["spread_mm"] = float(np.linalg.norm(xy.std(0)) * 1000)
						v["yaw_err_deg_median"] = float(
							np.median([yaw_error_deg(f.yaw, true_yaw, true_size) for f in fixes])
						)
					row["variants"][f"{seg}/tip{tip * 1000:.0f}/{'aim' if aim else 'plain'}"] = v
		cell.set_joints(q_home)
		return row
	except Exception as e:  # noqa: BLE001 - a crashed case is reported, not fatal
		return {"seed": seed, "error": f"crash:{type(e).__name__}:{e}"}
	finally:
		twin.cell.close()


def summarize(rows: list[dict]) -> dict:
	ok = [r for r in rows if "error" not in r]
	groups = {"size_ok": [r for r in ok if r["size_correct"]], "size_wrong": [r for r in ok if not r["size_correct"]]}
	out: dict = {"n": len(rows), "errors": [r["error"] for r in rows if "error" in r]}
	for g, rs in groups.items():
		if not rs:
			continue
		s = {"n": len(rs), "top_err_mm_median": float(np.median([r["top_err_mm"] for r in rs]))}
		for name in rs[0]["variants"]:
			vs = [r["variants"][name] for r in rs]
			have = [v for v in vs if v["fixes"]]
			s[name] = {
				"cases_with_fix": len(have),
				"fixes_mean": float(np.mean([v["fixes"] for v in vs])),
				"poses_mean": float(np.mean([v["poses"] for v in vs])),
			}
			if have:
				paired = [(r["top_err_mm"], r["variants"][name]) for r in rs if r["variants"][name]["fixes"]]
				for key in ("first_err_mm", "single_err_mm_median", "mean_err_mm", "median_err_mm", "spread_mm"):
					s[name][key] = float(np.median([v[key] for _, v in paired]))
				s[name]["top_err_mm_same_cases"] = float(np.median([t for t, _ in paired]))
				s[name]["median_lt_5mm"] = sum(v["median_err_mm"] < 5 for _, v in paired)
				s[name]["better_than_top"] = sum(v["median_err_mm"] < t for t, v in paired)
		out[g] = s
	return out


def main(argv: list[str] | None = None) -> int:
	ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	ap.add_argument("--out", type=Path, required=True)
	ap.add_argument("--seed0", type=int, default=3001201)
	ap.add_argument("--n", type=int, default=60)
	ap.add_argument("--views", type=int, default=8)
	ap.add_argument("--workers", type=int, default=8)
	a = ap.parse_args(argv)
	a.out.mkdir(parents=True, exist_ok=True)
	seeds = range(a.seed0, a.seed0 + a.n)
	with ProcessPoolExecutor(a.workers) as pool:
		rows = list(pool.map(run_case, [(s, a.out, a.views) for s in seeds]))
	with (a.out / "results.jsonl").open("w", encoding="utf-8") as fh:
		for r in rows:
			fh.write(json.dumps(r) + "\n")
	summary = summarize(rows)
	(a.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
	print(json.dumps(summary, indent=2))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
