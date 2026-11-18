"""Physically present matched bricks, evaluate v6 RGB, and fit a separate bottom operating point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from digital_twin_armigo.agent.twin_mcp import TwinService
from digital_twin_armigo.agent.twin_mujoco import PROFILE, TwinMujoco, sample_case
from digital_twin_armigo.agent.twin_record import EpisodeTrace, write_json
from digital_twin_armigo.inspection_scorer import default_sup_scorer


def trial(service, seed, positive, path, rtx_source=None, split=None):
	trace = EpisodeTrace(path, video=False)
	backend = TwinMujoco(sample_case(seed, positive, 5), service.scorer, trace, crop_size=service.crop_size)
	try:
		backend.enter("prepare")
		error = backend.prepare()
		backend.enter("present_bottom")
		views = []
		if not error:
			views, error = backend.inspect("present_bottom", [])
		view = views[0] if views else {}
		crop = path / "keyframes" / f"{view.get('id', '').replace(':', '-')}-crop.png"
		if rtx_source is not None and view.get("score") is not None:
			save_rtx_source(rtx_source, backend.cell, view, seed, positive, split)
		return {
			"seed": seed,
			"defective": positive,
			"error": error,
			"score": view.get("score"),
			"faces": view.get("faces", []),
			"face_pixels": view.get("face_pixels", []),
			"sim_frames": backend.total_frames,
			"crop": str(crop.relative_to(service.out)) if crop.is_file() else None,
		}
	finally:
		trace.close()
		backend.close()


def save_rtx_source(out, cell, view, seed, positive, split):
	from render_dataset import _camera_fovy, _geom_poses, _scene_nodes

	geoms, nodes = _scene_nodes(cell, out)
	poses, cam = _geom_poses(cell, cell.data, geoms, "top")
	index = seed * 2 + int(positive)
	path = f"png/{index:010d}_top.png"
	np.savez_compressed(
		out / "poses" / f"{index:010d}.npz",
		poses=poses[None, ...],
		cams=cam[None, ...],
		paths=np.array([path]),
		nodes=json.dumps(nodes),
		fovy=json.dumps({"top": _camera_fovy(cell.model, cell.camera_id("top"))}),
	)
	row = {
		"path": path,
		"layout_index": seed,
		"defective": positive,
		"defect_face": 5,
		"split": split,
		"camera": "top",
		"stage": "bottom",
		"task": "cross_var",
		"bbox": view["crop_bbox"],
	}
	with (out / "metadata.jsonl").open("a", encoding="utf-8") as file:
		file.write(json.dumps(row) + "\n")


def counts(rows, threshold):
	positive = [r for r in rows if r["defective"]]
	negative = [r for r in rows if not r["defective"]]
	return {
		"threshold": float(threshold),
		"detected": sum(r["score"] is not None and r["score"] >= threshold for r in positive),
		"defective": len(positive),
		"false_alarms": sum(r["score"] is not None and r["score"] >= threshold for r in negative),
		"normal": len(negative),
		"unreadable_defective": sum(r["score"] is None for r in positive),
		"unreadable_normal": sum(r["score"] is None for r in negative),
	}


def main():
	p = argparse.ArgumentParser(description=__doc__)
	p.add_argument("--detector", type=Path, default=ROOT / "results/anomaly/models-sup/sup.json")
	p.add_argument("--out", type=Path, required=True)
	p.add_argument("--calibration-start", default=661200, type=int)
	p.add_argument("--test-start", default=772400, type=int)
	p.add_argument("--calibration-n", default=12, type=int)
	p.add_argument("--test-n", default=12, type=int)
	p.add_argument("--dump-rtx-source", action="store_true")
	a = p.parse_args()
	cal_seeds = range(a.calibration_start, a.calibration_start + a.calibration_n)
	test_seeds = range(a.test_start, a.test_start + a.test_n)
	if set(cal_seeds) & set(test_seeds) or not cal_seeds or not test_seeds:
		p.error("calibration and test must be nonempty and disjoint")
	service = TwinService(a.out, a.detector)
	rtx_source = service.out / "rtx-source" if a.dump_rtx_source else None
	if rtx_source is not None:
		(rtx_source / "poses").mkdir(parents=True, exist_ok=False)
	try:
		service.scorer = default_sup_scorer(a.detector.resolve())
		data = {}
		for split, seeds, labels in (("calibration", cal_seeds, (False,)), ("test", test_seeds, (False, True))):
			data[split] = []
			for seed in seeds:
				for positive in labels:
					row = trial(
						service, seed, positive, service.out / split / str(seed) / str(int(positive)), rtx_source, split
					)
					data[split].append(row)
					print(split, json.dumps(row), flush=True)
					write_json(service.out / "scores.json", data)
		valid = [r["score"] for r in data["calibration"] if r["score"] is not None]
		if len(valid) < 4:
			raise RuntimeError("fewer than four readable calibration normals")
		# A separate view profile. Never let a candidate lower the existing v6 threshold in this pilot.
		threshold = float(max(0.0, np.quantile(valid, 0.99, method="higher")))
		calibration = {
			"profile": PROFILE,
			"detector_sha256": service.model_hash,
			"threshold": threshold,
			"calibration_requested": len(cal_seeds),
			"calibration_readable": len(valid),
			"scope": "MuJoCo bottom RGB pilot",
			"quantile": 0.99,
			"threshold_floor": 0.0,
		}
		write_json(service.out / "bottom-profile.json", calibration)
		write_json(
			service.out / "summary.json",
			{
				"calibration": calibration,
				"default": counts(data["test"], 0.0),
				"bottom_calibrated": counts(data["test"], threshold),
			},
		)
	finally:
		service.close()


if __name__ == "__main__":
	main()
