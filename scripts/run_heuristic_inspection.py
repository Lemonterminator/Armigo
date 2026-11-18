"""Heuristic inspection strategy vs the show-pose baseline on real MuJoCo episodes (cross_var defects, faces 0..4).

Strategies (all use the same :class:`~digital_twin_armigo.agent.heuristic.HeuristicInspector` decision rules and the
same detector, so they differ only in WHERE they look):

* ``show``          -- no pre-grasp scan; the held wrist view, then the classic show poses (3 base + 4 fixed
  extras) seen by both cameras, until the label is known;
* ``heuristic``     -- pre-grasp wrist scan, grasp chosen to keep the faces of interest free, held wrist view, top-
  camera close-ups of the faces still to check; a wrist detection waits for a top-camera confirmation (budget 4);
* ``heuristic_fast``-- the same without waiting for the top-camera confirmation;
* ``heuristic_x2``  -- ``heuristic`` with every face cleared only by TWO non-detecting views (pre-scan plans double
  coverage, up to 8 views), for a detector that misses often.

Detectors: ``oracle`` (score = visible defect pixels, threshold 25: the strategy's upper bound), ``anomalib`` (the
per-camera PatchCore models through ``scripts/anomaly/score_server.py`` in the anomaly env) or ``sup`` (the supervised
LoRA two-head detector of ``--sup-spec``, written by ``scripts/anomaly/export_sup.py``; crops are 256 px as in its
training set, scores are standardised so the default threshold is 0). Episodes are scored by the
unchanged evaluator against the ground-truth bowl. Writes ``episodes.jsonl`` and ``summary.json`` into ``--out``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import bench_physics_expert as bpe

STRATEGIES = ("show", "heuristic", "heuristic_fast", "heuristic_x2")
SHOW_POSES = ((0.65, -35.0), (0.65, 0.0), (0.65, 35.0), (0.0, 120.0), (0.8, 90.0), (0.4, -150.0), (0.8, -90.0))
ORACLE_MIN_DEFECT_PX = 25
IMAGE_SIZE = 128


def sample_layouts(n: int, seed: int, p_defective: float = 0.5) -> list:
	rng = np.random.default_rng(seed + 30_000)
	out = []
	for k, lay in enumerate(bpe.sample_layouts(n, seed, bpe.load_top_camera())):
		lay.defect_kind = "cross_var"
		lay.defect_face = int(rng.integers(5))
		lay.defective = bool(rng.random() < p_defective)
		lay.defect_seed = int(seed * 100_019 + k)
		out.append(lay)
	return out


class _Capture:
	"""Training-identical crops (``render_dataset.capture``, clamped square crop, full top frame); remembers the last
	view so the oracle detector can read its defect pixels."""

	def __init__(self, image_size: int = IMAGE_SIZE) -> None:
		import anomaly.render_dataset as rd

		self.rd = rd
		self.image_size = int(image_size)
		self.last = None

	def __call__(self, cell, camera: str, data):
		kw = {"clamp": True} if "clamp" in self.rd.capture.__code__.co_varnames else {}
		view, image, _ = self.rd.capture(cell, camera, None, data, self.image_size, **kw)
		if image is not None and view.brick_px < self.rd.MIN_BRICK_PX:
			image = None  # the training set drops such crops too
		self.last = view
		return view, image


class _OracleScorer:
	"""Upper-bound detector: the visible defect pixels of the last capture. A normal brick carries the same cross in
	the hidden probe group, which ``defect_px`` also counts, so the score is 0 unless the brick is defective."""

	def __init__(self, cap: _Capture) -> None:
		self.cap = cap
		self.defective = False  # set per episode by run_one

	def score(self, camera: str, image) -> float:
		return float(self.cap.last.defect_px) if self.defective else 0.0

	def threshold(self, camera: str) -> float:
		return float(ORACLE_MIN_DEFECT_PX)


def run_one(lay, k: int, strategy: str, scorer, cap: _Capture, lib: np.ndarray, args) -> dict[str, Any]:
	from digital_twin_armigo.agent.heuristic import HeuristicInspector
	from digital_twin_armigo.agent.heuristic_episode import HeuristicPlanner
	from digital_twin_armigo.benchmarks import SENSITIVITY_MATRIX
	from digital_twin_armigo.constants import SO101_MJCF_DIR
	from digital_twin_armigo.mujoco_cell import Cell
	from digital_twin_armigo.mujoco_cell.physics_expert import ExpertParams, run_closed_loop_episode
	from digital_twin_armigo.physics import load_physics_spec

	spec = load_physics_spec()
	base = {c.name: c for c in SENSITIVITY_MATRIX}["baseline_2x2"]
	cell = Cell(lay, SO101_MJCF_DIR, physics=spec, dynamic_clutter=True, assisted_grasp=False, face_probes=True)
	min_px = {"wrist_cam": args.min_face_px_wrist, "top": args.min_face_px_top}
	x2 = strategy == "heuristic_x2"
	inspector = HeuristicInspector(
		min_face_px=min_px,
		max_presents=args.max_presents,
		require_top_confirmation=strategy in ("heuristic", "heuristic_x2"),
		min_clear_views=2 if x2 else 1,
	)
	planner = HeuristicPlanner(
		scorer,
		cap,
		lib,
		inspector=inspector,
		thresholds=args.thresholds,
		prescan=strategy != "show",
		show_poses=SHOW_POSES if strategy == "show" else None,
		prescan_min_px=args.min_face_px_wrist,
		prescan_times=2 if x2 else 1,
		max_prescan=8 if x2 else 4,
	)
	gt_bowl = lay.pink if lay.defective else lay.green
	if isinstance(scorer, _OracleScorer):
		scorer.defective = bool(lay.defective)
	t0 = time.perf_counter()
	try:
		tel, phases, info = run_closed_loop_episode(
			cell,
			ExpertParams(max_frames=args.max_frames),
			q_start=bpe._q_start(),
			brick_xy=lay.brick_xy,
			brick_yaw_deg=lay.brick_yaw_deg,
			non_ideal=bpe.bench._mj_non_ideal(base, spec),
			bowls={"normal": lay.green, "defective": lay.pink},
			planner=planner,
		)
	finally:
		cell.close()
	wall = time.perf_counter() - t0
	if info["infeasible_reason"] is not None:
		verdict, reasons = "INFEASIBLE", [info["infeasible_reason"]]
	else:
		health = bpe.bench._evaluate_mj_rollout(tel, phases, spec, tuple(lay.brick_size), gt_bowl)
		verdict, reasons = health["verdict"], list(health["reason_codes"])
	label = info.get("inspection_label")
	summ = planner.summary()
	views = summ.pop("views")
	return {
		"strategy": strategy,
		"layout_index": k,
		"defective": bool(lay.defective),
		"defect_face": int(lay.defect_face),
		"verdict": verdict,
		"reason_codes": reasons,
		"aborted": info["aborted"],
		"label": label,
		"label_correct": label is not None and label == ("defective" if lay.defective else "normal"),
		"n_frames": int(info["n_frames"]),
		"truncated": bool(info.get("truncated")),
		"n_views": sum("score" in v for v in views),
		"n_present": sum(v.get("stage") == "present" and "score" in v for v in views),
		"n_present_skipped": sum("skipped" in v for v in views),
		"inspector": summ,
		"views": views,
		"wall_s": round(wall, 2),
	}


def summarize(eps: list[dict]) -> dict[str, Any]:
	out: dict[str, Any] = {}
	for st in dict.fromkeys(e["strategy"] for e in eps):
		es = [e for e in eps if e["strategy"] == st]
		n = len(es)
		d = [e for e in es if e["defective"]]
		nd = [e for e in es if not e["defective"]]
		known = [e for e in es if e["label"] is not None]
		out[st] = {
			"n": n,
			"success_rate": round(sum(e["verdict"] == "SUCCESS" for e in es) / n, 3),
			"success_with_correct_label_rate": round(
				sum(e["verdict"] == "SUCCESS" and e["label_correct"] for e in es) / n, 3
			),
			"known_rate": round(len(known) / n, 3),
			"accuracy_of_known": round(sum(e["label_correct"] for e in known) / max(len(known), 1), 3),
			"confusion": {
				"defective->defective": sum(e["label"] == "defective" for e in d),
				"defective->normal": sum(e["label"] == "normal" for e in d),
				"defective->unknown": sum(e["label"] is None for e in d),
				"normal->normal": sum(e["label"] == "normal" for e in nd),
				"normal->defective": sum(e["label"] == "defective" for e in nd),
				"normal->unknown": sum(e["label"] is None for e in nd),
			},
			"top_confirmed_of_defective_labels": sum(
				e["inspector"].get("top_confirmed", False) for e in es if e["label"] == "defective"
			),
			"verdicts": dict(Counter(e["verdict"] for e in es)),
			"aborted": dict(Counter(str(e["aborted"]) for e in es)),
			"mean_views": round(statistics.fmean(e["n_views"] for e in es), 2),
			"mean_present": round(statistics.fmean(e["n_present"] for e in es), 2),
			"mean_frames": round(statistics.fmean(e["n_frames"] for e in es), 1),
			"truncated_rate": round(sum(e["truncated"] for e in es) / n, 3),
			"mean_wall_s": round(statistics.fmean(e["wall_s"] for e in es), 2),
		}
	return out


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	ap.add_argument("--n", type=int, default=20)
	ap.add_argument("--seed", type=int, default=11)
	ap.add_argument("--strategies", default=",".join(STRATEGIES))
	ap.add_argument("--detector", choices=("oracle", "anomalib", "sup"), default="oracle")
	ap.add_argument(
		"--sup-spec", type=Path, default=ROOT / "results" / "anomaly" / "models-sup" / "sup.json", help="supervised detector spec (--detector sup)"
	)
	ap.add_argument("--image-size", type=int, default=0, help="crop size; default 256 for sup, else 128")
	ap.add_argument("--models-root", type=Path, default=ROOT / "results" / "anomaly" / "models-v2")
	ap.add_argument("--thresholds", default="", help="per-camera detection thresholds, e.g. top=60,wrist_cam=58")
	ap.add_argument("--closeup-library", type=Path, default=ROOT / "results" / "top-camera-reach" / "poses.jsonl")
	ap.add_argument("--closeup-per-layout", type=int, default=15)
	ap.add_argument("--min-face-px-wrist", type=int, default=800)
	ap.add_argument("--min-face-px-top", type=int, default=250)
	ap.add_argument("--max-presents", type=int, default=4)
	ap.add_argument("--max-frames", type=int, default=2600)
	ap.add_argument("--out", type=Path, default=ROOT / "results" / "heuristic-inspection" / "run")
	args = ap.parse_args()
	args.thresholds = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in args.thresholds.split(",") if "=" in kv}
	strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
	assert all(s in STRATEGIES for s in strategies), strategies

	from digital_twin_armigo.mujoco_cell.inspection_planning import closeup_library

	rows = [json.loads(line) for line in args.closeup_library.open(encoding="utf-8")]
	lib = closeup_library(rows, per_layout=args.closeup_per_layout)
	args.out.mkdir(parents=True, exist_ok=True)
	lays = sample_layouts(args.n, args.seed)
	cap = _Capture(args.image_size or (256 if args.detector == "sup" else IMAGE_SIZE))
	if args.detector == "oracle":
		scorer = _OracleScorer(cap)
	elif args.detector == "sup":
		from digital_twin_armigo.inspection_scorer import default_sup_scorer

		scorer = default_sup_scorer(args.sup_spec)
	else:
		from digital_twin_armigo.inspection_scorer import default_scorer

		scorer = default_scorer(args.models_root)
	eps: list[dict] = []
	t0 = time.perf_counter()
	try:
		for k, lay in enumerate(lays):
			for st in strategies:
				ep = run_one(lay, k, st, scorer, cap, lib, args)
				eps.append(ep)
				print(
					f"[{k:3d} {st:14s}] def={int(ep['defective'])} face={ep['defect_face']} {ep['verdict']:10s} "
					f"label={ep['label']} views={ep['n_views']} present={ep['n_present']} aborted={ep['aborted']} "
					f"reason={ep['inspector'].get('reason')} ({time.perf_counter() - t0:.0f}s)",
					flush=True,
				)
				with (args.out / "episodes.jsonl").open("a" if len(eps) > 1 else "w", encoding="utf-8") as f:
					f.write(json.dumps(ep) + "\n")
	finally:
		if hasattr(scorer, "close"):
			scorer.close()
	summary = {
		"strategies": summarize(eps),
		"config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k not in ("thresholds",)}
		| {
			"thresholds": args.thresholds,
			"closeup_library_size": len(lib),
			"wall_s": round(time.perf_counter() - t0, 1),
		},
	}
	(args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
	print(json.dumps(summary["strategies"], indent=2))


if __name__ == "__main__":
	main()
