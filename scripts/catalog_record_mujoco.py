"""Record a NEW explicitly requested run, linked to historical evidence without replacing it."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]


def main():
	p = argparse.ArgumentParser()
	p.add_argument("--request", type=Path, required=True)
	p.add_argument("--out", type=Path, required=True)
	a = p.parse_args()
	import bench_cross_engine as b
	import bench_physics_expert as bp
	import mujoco

	from digital_twin_armigo.benchmarks import HELD_OUT_PERTURBATIONS, SENSITIVITY_MATRIX
	from digital_twin_armigo.catalog.capture import MujocoCapture, identity
	from digital_twin_armigo.catalog.store import plain
	from digital_twin_armigo.constants import SO101_MJCF_DIR
	from digital_twin_armigo.mujoco_cell import Cell
	from digital_twin_armigo.mujoco_cell.physics_expert import ExpertParams, jaw_env_contact_force
	from digital_twin_armigo.physics import load_physics_spec, physics_manifest

	request = json.loads(a.request.read_text(encoding="utf-8"))
	e = request["episode"]
	spec = load_physics_spec()
	cam = bp.load_top_camera()
	raw = e["raw"]
	condname = raw.get("condition")
	if condname:
		cond = next(c for c in (*SENSITIVITY_MATRIX, *HELD_OUT_PERTURBATIONS) if c.name == condname)
		seed = int(raw["seed"])
		jitter = b._mj_jitter_condition(cond, seed)
		cell = b._mj_make_cell(spec, jitter, cam, seed=seed)
		nonideal = b._mj_non_ideal(jitter, spec)
		xy, yaw = jitter.brick_xy, jitter.brick_yaw_deg
	else:
		seed = e["provenance"].get("layout_seed")
		if seed is None or "layout_index" not in raw:
			raise ValueError("record has no executable MuJoCo recipe")
		count = e["provenance"].get("layout_count", 60)
		layout = bp.sample_layouts(count, int(seed), cam)[int(raw["layout_index"])]
		cell = Cell(
			layout, SO101_MJCF_DIR, physics=spec, dynamic_clutter=True, assisted_grasp=False, render_enabled=False
		)
		nonideal = b._mj_non_ideal(SENSITIVITY_MATRIX[0], spec)
		xy, yaw = layout.brick_xy, layout.brick_yaw_deg
	params = ExpertParams(**(e["provenance"].get("original", {}).get("params") or {}))
	recorder = MujocoCapture(cell)
	substeps = []
	cell.frame_observer = lambda ep: recorder.frame(cell)

	def substep():
		fn0, ft0, _, fn1, ft1, _ = cell.jaw_force_components()
		substeps.append([fn0, fn1, ft0, ft1, jaw_env_contact_force(cell)])

	cell.substep_observer = substep
	try:
		result = bp._run_on_cell(cell, spec, nonideal, params, xy, yaw)
		tel = result.pop("_telemetry", {})
		phases = result.pop("_phases", None)
		if phases is not None:
			tel["phases"] = phases
		tel["contact_substeps"] = substeps
		if substeps and result.get("health"):
			result["health"]["recorded_substep_pinch_normal_peak_N"] = max(max(r[:2]) for r in substeps)
		provenance = {
			"identity": identity(ROOT, "mujoco", mujoco.__version__),
			"physics_manifest": physics_manifest(spec),
			"params": dataclasses.asdict(params),
			"seed": seed,
			"historical_recipe_used_with_current_code": True,
			"original_comparison": "not_a_claim_of_reproduction",
		}
		manifest = recorder.save_mujoco(a.out, cell, tel, plain(result), provenance, e["alias"] + "_rerun", e["id"])
		print(json.dumps({"episode_manifest": str(manifest)}))
	finally:
		cell.close()


if __name__ == "__main__":
	main()
