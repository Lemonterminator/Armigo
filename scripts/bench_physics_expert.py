"""Acceptance runner for the closed-loop pure-physics expert on the MuJoCo sensitivity matrix.

Runs conditions × seeds with :class:`~digital_twin_armigo.mujoco_cell.physics_expert.ClosedLoopExpert`, evaluates
every episode with the unchanged audited evaluator (``_evaluate_mj_rollout`` in ``bench_cross_engine.py``), prints
one line per condition and writes per-episode JSON.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import bench_cross_engine as bench

from digital_twin_armigo.benchmarks import HELD_OUT_PERTURBATIONS, SENSITIVITY_MATRIX
from digital_twin_armigo.camera import CameraPose
from digital_twin_armigo.constants import REST_U
from digital_twin_armigo.joints import normalized_to_rad
from digital_twin_armigo.metrics import EVALUATOR_VERSION
from digital_twin_armigo.physics import load_physics_spec

DEFAULT_OUT_DIR = ROOT / "results" / "physics-expert-v2"


def load_top_camera() -> CameraPose:
	return CameraPose(**json.loads((ROOT / "config" / "sim" / "top_camera_joint.json").read_text())["camera"])


def _q_start() -> np.ndarray:
	q_start = normalized_to_rad(np.array(REST_U, dtype=np.float64))
	q_start[5] = bench.OPEN_G
	return q_start


def _run_on_cell(cell, spec, non_ideal, params, brick_xy, brick_yaw_deg) -> dict[str, Any]:
	"""Closed-loop episode on an already built pure-physics cell, evaluated by the unchanged audited evaluator.

	A layout the expert declines on kinematic grounds (before any physics step) is reported as ``INFEASIBLE``.
	"""
	from digital_twin_armigo.mujoco_cell.physics_expert import assess_feasibility, run_closed_loop_episode

	if cell.assisted_grasp is not False:
		raise RuntimeError("the closed-loop expert must run with assisted_grasp=False")
	size = tuple(cell.layout.brick_size)
	bowl_xy = cell.layout.pink if cell.layout.defective else cell.layout.green
	q_start = _q_start()
	reason = assess_feasibility(cell, q_start, bowl_xy, params, brick_xy, brick_yaw_deg)
	if reason is not None:
		return {"verdict": "INFEASIBLE", "reason_codes": [reason], "info": {}, "health": {}}
	tel, phases, info = run_closed_loop_episode(
		cell,
		params,
		q_start=q_start,
		bowl_xy=bowl_xy,
		brick_xy=brick_xy,
		brick_yaw_deg=brick_yaw_deg,
		non_ideal=non_ideal,
	)
	health = bench._evaluate_mj_rollout(tel, phases, spec, size, bowl_xy)
	return {
		"verdict": health["verdict"],
		"reason_codes": list(health["reason_codes"]),
		"info": info,
		"health": health,
		"_telemetry": tel,
		"_phases": phases,
	}


def run_episode(cond, spec, top_cam: CameraPose, seed: int, params=None) -> dict[str, Any]:
	"""One closed-loop episode for matrix condition ``cond`` and ``seed``: verdict, reason codes and diagnostics.

	The condition jitter, the ``Cell`` (``assisted_grasp=False``, the condition's mass / friction scales) and the
	non-idealities come from the same helpers the open-loop benchmark uses.
	"""
	from digital_twin_armigo.mujoco_cell.physics_expert import ExpertParams

	params = params or ExpertParams()
	cond_j = bench._mj_jitter_condition(cond, seed)
	cell = bench._mj_make_cell(spec, cond_j, top_cam, seed=seed)
	try:
		out = _run_on_cell(cell, spec, bench._mj_non_ideal(cond_j, spec), params, cond_j.brick_xy, cond_j.brick_yaw_deg)
	finally:
		cell.close()
	return {"condition": cond.name, "seed": seed, **out}


def run_layout_episode(layout, spec, k: int, params=None) -> dict[str, Any]:
	"""One closed-loop episode on a ``sample_layout`` layout (dynamic clutter, pure physics, baseline
	non-idealities)."""
	from digital_twin_armigo.constants import SO101_MJCF_DIR
	from digital_twin_armigo.mujoco_cell import Cell
	from digital_twin_armigo.mujoco_cell.physics_expert import ExpertParams

	params = params or ExpertParams()
	base = {c.name: c for c in SENSITIVITY_MATRIX}["baseline_2x2"]
	cell = Cell(layout, SO101_MJCF_DIR, physics=spec, dynamic_clutter=True, assisted_grasp=False, render_enabled=False)
	try:
		out = _run_on_cell(cell, spec, bench._mj_non_ideal(base, spec), params, layout.brick_xy, layout.brick_yaw_deg)
	finally:
		cell.close()
	lay = {
		"brick_xy": [float(v) for v in layout.brick_xy],
		"brick_yaw_deg": float(layout.brick_yaw_deg),
		"brick_size": [float(v) for v in layout.brick_size],
		"green": [float(v) for v in layout.green],
		"pink": [float(v) for v in layout.pink],
		"defective": bool(layout.defective),
		"n_clutter": len(layout.clutter),
	}
	return {"layout_index": k, "layout": lay, **out}


def sample_layouts(n: int, seed: int, top_cam: CameraPose) -> list:
	"""``n`` layouts from the data pipeline's own sampler (no prior), alternating normal / defective bricks."""
	from digital_twin_armigo.mujoco_cell.layout import sample_layout

	rng = np.random.default_rng(seed)
	out = []
	for k in range(n):
		lay = sample_layout(rng, top_cam, defective=bool(k % 2))
		lay.top = top_cam
		out.append(lay)
	return out


def _jsonable(o: Any) -> Any:
	if isinstance(o, dict):
		return {k: _jsonable(v) for k, v in o.items() if not str(k).startswith("_")}
	if isinstance(o, (list, tuple)):
		return [_jsonable(v) for v in o]
	if isinstance(o, np.ndarray):
		return o.tolist()
	if isinstance(o, (np.floating, float)):
		return None if not math.isfinite(float(o)) else float(o)
	if isinstance(o, (np.integer,)):
		return int(o)
	if isinstance(o, np.bool_):
		return bool(o)
	return o


def _summary(rows: list[dict], key: str) -> dict[str, Any]:
	verdicts = Counter(r["verdict"] for r in rows)
	reasons = Counter(c for r in rows for c in r["reason_codes"])
	jaw_env = [r["info"].get("jaw_env_contact_max_N", 0.0) for r in rows if r["info"]]
	return {
		"successes": verdicts.get("SUCCESS", 0),
		"episodes": len(rows),
		"verdict_counts": dict(verdicts),
		"reason_counts": dict(reasons),
		"failing": [r[key] for r in rows if r["verdict"] != "SUCCESS"],
		"jaw_env_contact_max_N": float(max(jaw_env)) if jaw_env else 0.0,
		"episodes_detail": [_jsonable(r) for r in rows],
	}


def _print_line(name: str, summ: dict[str, Any], what: str) -> None:
	print(
		f"{name:20s} {summ['successes']:3d}/{summ['episodes']}  verdicts={summ['verdict_counts']}  "
		f"reasons={summ['reason_counts']}  jaw_env_max={summ['jaw_env_contact_max_N']:.2f}N  "
		f"failing_{what}={summ['failing']}",
		flush=True,
	)


def main(argv: list[str] | None = None) -> int:
	import mujoco

	from digital_twin_armigo.mujoco_cell.physics_expert import ExpertParams

	all_conds = {c.name: c for c in (*SENSITIVITY_MATRIX, *HELD_OUT_PERTURBATIONS)}
	parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	parser.add_argument("--conditions", nargs="+", default=[c.name for c in SENSITIVITY_MATRIX])
	parser.add_argument("--include-held-out", action="store_true", help="Add the held-out combined perturbations.")
	parser.add_argument("--seed-start", type=int, default=0)
	parser.add_argument("--seeds", type=int, default=30)
	parser.add_argument("--layouts", type=int, default=0, help="Layout mode: run N sample_layout layouts instead.")
	parser.add_argument("--layout-seed", type=int, default=7, help="Generator seed for layout mode.")
	parser.add_argument("--out", type=Path, default=None, help="Output JSON path.")
	args = parser.parse_args(argv)

	spec = load_physics_spec()
	top_cam = load_top_camera()
	params = ExpertParams()
	report: dict[str, Any] = {
		"mujoco_version": mujoco.__version__,
		"evaluator_version": EVALUATOR_VERSION,
		"params": dataclasses.asdict(params),
	}
	t0 = time.time()
	if args.layouts > 0:
		out = args.out or DEFAULT_OUT_DIR / f"physics_expert_layouts_seed{args.layout_seed}_n{args.layouts}.json"
		layouts = sample_layouts(args.layouts, args.layout_seed, top_cam)
		rows = [run_layout_episode(lay, spec, k, params) for k, lay in enumerate(layouts)]
		summ = _summary(rows, "layout_index")
		_print_line(f"layouts seed {args.layout_seed}", summ, "layouts")
		report.update(mode="layouts", layout_seed=args.layout_seed, layouts=summ)
	else:
		names = list(args.conditions)
		if args.include_held_out:
			names += [c.name for c in HELD_OUT_PERTURBATIONS if c.name not in names]
		unknown = [n for n in names if n not in all_conds]
		if unknown:
			parser.error(f"unknown conditions: {unknown}")
		seeds = list(range(args.seed_start, args.seed_start + args.seeds))
		out = args.out or DEFAULT_OUT_DIR / f"physics_expert_seeds{seeds[0]}-{seeds[-1]}.json"
		report.update(mode="matrix", seeds=seeds, conditions={})
		for name in names:
			rows = [run_episode(all_conds[name], spec, top_cam, s, params) for s in seeds]
			summ = _summary(rows, "seed")
			_print_line(name, summ, "seeds")
			report["conditions"][name] = summ
	report["wall_s"] = time.time() - t0
	out.parent.mkdir(parents=True, exist_ok=True)
	out.write_text(json.dumps(_jsonable(report), indent=1), encoding="utf-8")
	print(f"wrote {out}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
