"""Full 660-episode regression and bit-identical comparison for Gripper Stage 2+3 against Stage 1 audit."""

from __future__ import annotations

import concurrent.futures
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
	sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
	sys.path.insert(0, str(ROOT / "scripts"))

import audit_gripper_state as ags

from digital_twin_armigo.benchmarks import HELD_OUT_PERTURBATIONS, SENSITIVITY_MATRIX
from digital_twin_armigo.constants import CELL_MARKERS, SO101_MJCF_DIR
from digital_twin_armigo.mujoco_cell.scene import write_marker_textures

STAGE1_AUDIT_PATH = ROOT / "results" / "gripper-state" / "audit.json"
BOWL_POST_PATH = ROOT / "results" / "bowl-experiments" / "baseline" / "SUMMARY_POST_ADOPTION.json"
OUT_PATH = ROOT / "results" / "gripper-state" / "regression.json"


def main() -> int:
	t0 = time.time()
	write_marker_textures(SO101_MJCF_DIR, set(CELL_MARKERS))
	stage1 = json.loads(STAGE1_AUDIT_PATH.read_text(encoding="utf-8"))
	baseline = json.loads(BOWL_POST_PATH.read_text(encoding="utf-8"))
	s1_hashes: dict[str, str] = stage1["telemetry_hashes"]
	s1_blocked = set(stage1["blocked_episode_ids"])

	cond_names = [c.name for c in (*SENSITIVITY_MATRIX, *HELD_OUT_PERTURBATIONS)]
	tasks: list[tuple[str, str, int]] = [
		("matrix", cn, s) for cn in cond_names for s in (*range(30), *range(100, 130))
	] + [("layout", str(ls), idx) for ls in (1000, 1001) for idx in range(60)]

	with concurrent.futures.ProcessPoolExecutor(max_workers=12) as pool:
		rows = list(pool.map(ags._run_audited_episode, tasks))

	matrix_rows = [r for r in rows if r["kind"] == "matrix"]
	layout_rows = [r for r in rows if r["kind"] == "layout"]
	per_cond: dict[str, list[int]] = {}
	for r in matrix_rows:
		ok, n = per_cond.get(r["group"], [0, 0])
		per_cond[r["group"]] = [ok + (1 if r["verdict"] == "SUCCESS" else 0), n + 1]

	matrix_ok = sum(v[0] for v in per_cond.values())
	layouts_ok = sum(1 for r in layout_rows if r["verdict"] == "SUCCESS")
	idx32_row = next(r for r in layout_rows if r["id"] == "layouts_seed1000_idx32")

	bounce_outs = sum(1 for r in rows if "bounce_out" in r["reason_codes"])
	fast_releases = sum(1 for r in rows if (r["release_peak_speed_mps"] or 0.0) > 1.2)
	jaw_over_5n = [r["id"] for r in rows if r["jaw_env_contact_max_N"] > 5.0]
	frames_ge_890 = [r["id"] for r in rows if r["n_frames"] >= 890]
	fp = [r["id"] for r in rows if r["gripper_blocked_frames"] > 0 and r["jaw_env_contact_max_N"] < 0.5]

	# Bit-identical check on episodes that were NOT blocked in Stage 1 and did NOT pre-narrow open_g
	hash_diffs_all = [r["id"] for r in rows if r["telemetry_hash"] != s1_hashes[r["id"]]]
	hash_diffs_unblocked_default = [
		r["id"]
		for r in rows
		if r["id"] not in s1_blocked and r["open_g_used"] == 0.85 and r["telemetry_hash"] != s1_hashes[r["id"]]
	]
	pre_narrowed = [
		{"id": r["id"], "open_g": r["open_g_used"], "verdict": r["verdict"], "jaw_max_N": r["jaw_env_contact_max_N"]}
		for r in rows
		if r["open_g_used"] != 0.85
	]
	infeasible = [r["id"] for r in rows if r["verdict"] == "INFEASIBLE"]

	frames_ge_890_other = [eid for eid in frames_ge_890 if eid != "layouts_seed1000_idx32"]

	passed = (
		matrix_ok >= baseline["matrix_success"]
		and layouts_ok >= baseline["layouts"][0]
		and idx32_row["verdict"] == "SUCCESS"
		and len(jaw_over_5n) < baseline["jaw_env_over_5N"]
		and bounce_outs == 0
		and fast_releases == 0
		and len(frames_ge_890_other) <= baseline["frames_ge_890"]
		and len(hash_diffs_unblocked_default) == 0
		and len(fp) == 0
	)
	report = {
		"passed": passed,
		"wall_s": round(time.time() - t0, 2),
		"matrix_success": matrix_ok,
		"matrix_total": len(matrix_rows),
		"per_condition": per_cond,
		"layouts": [layouts_ok, len(layout_rows)],
		"idx32": idx32_row,
		"infeasible": infeasible,
		"bounce_outs": bounce_outs,
		"fast_releases": fast_releases,
		"jaw_env_over_5N": jaw_over_5n,
		"frames_ge_890": frames_ge_890,
		"false_positives": fp,
		"hash_diffs_all": hash_diffs_all,
		"hash_diffs_unblocked_default": hash_diffs_unblocked_default,
		"pre_narrowed": pre_narrowed,
	}
	OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
	OUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
	print(json.dumps(report, indent=2))
	return 0 if passed else 1


if __name__ == "__main__":
	raise SystemExit(main())
