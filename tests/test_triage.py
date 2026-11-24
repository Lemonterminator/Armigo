"""Unit tests and p2-dev-r2 calibration checks for automatic failure triage rules (R1-R8).

Calibration on ``D:/Armigo_data_v2/runs/p2-dev-r2`` (216 total episodes, read-only):
- Clean ``SUCCESS`` (``scored_verdict == "SUCCESS"`` & ``attribution == "none"``):
  152/152 (100.0%) -> ``stage="ok", root_cause="none"``.
- Non-perfect in ``scores.jsonl`` (50 total):
  - ``evidence_insufficient`` (15): 12 -> R2 (``perception_pose_error``, yaw err 19.8-25.5 deg),
    3 -> R5 (``evidence_insufficient``, missing face 0) [15/15 = 100.0%].
  - ``false_positive_visible`` (2): 2 -> R7 (``false_positive_visible``) [2/2 = 100.0%].
  - ``drift`` + ``no_clamp`` in grasp (27 total across ``grasp_or_path_failure`` and
    ``recovery_failure``): 17 -> R3 (``calibration_stale``), 10 -> R2 (``perception_pose_error``,
    higher-priority rule where upright ``view_truth`` also had yaw err > 10 deg: 4x L700007
    with 19.12 deg initial yaw err, 6x L700006 with 43.76 deg recovery yaw err).
    All 17/17 pure drift grasp failures without >10 deg yaw error map to R3 (100.0%).
  - ``BRICK_LOST`` without ``no_clamp`` or >10 deg yaw error (3x L700005-D4): 3 -> R6
    (``grasp_or_path_failure``) [3/3 = 100.0%].
  - ``target_face_not_in_view`` (3): 2x L700007-D0 (drift-B/C, 19.12 deg yaw err) -> R2,
    1x L700005-N (drift-C, view budget exhausted after review) -> ``unclassified``.
- Physically in right bowl (``attribution == "none"``) but ``verdict == "FAILED"`` due to
  ``health.failure_mode == "torque_stall"`` (14 episodes): all 14 -> ``code="torque_stall"``,
  ``root_cause="unclassified"`` (process torque stall not in R1-R8).
- Overall coverage on R1-R8 targets: 49/50 (98.0%) ``scores.jsonl`` non-perfect episodes triaged
  by R1-R8; 152/152 (100.0%) clean ``SUCCESS`` episodes -> ``none``; 15/15 (100.0%)
  ``evidence_insufficient`` -> R2/R5; 17/17 (100.0%) pure drift grasp failures -> R3.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from digital_twin_armigo.experiments.contract import ROOT_CAUSES
from digital_twin_armigo.experiments.triage import RULES, TriageRow, triage_episode


def write_episode(
	root: Path,
	name: str = "ep",
	*,
	events: list[dict] | None = None,
	result: dict | None = None,
	experiment: dict | None = None,
	task: dict | None = None,
	audit: dict | None = None,
	brick_size: list[float] | None = None,
) -> Path:
	ep_dir = root / name
	ep_dir.mkdir(parents=True, exist_ok=True)
	payload: dict = {"schema": "armigo-agent-episode/v1", "result": result or {}}
	if experiment is not None:
		payload["provenance"] = {"engine": "mujoco", "experiment": experiment}
	(ep_dir / "episode.json").write_text(json.dumps(payload), encoding="utf-8")
	lines = [json.dumps(ev) for ev in (events or [])]
	(ep_dir / "events.jsonl").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
	if task is not None:
		(ep_dir / "task.json").write_text(json.dumps(task), encoding="utf-8")
	if audit is not None:
		(ep_dir / "audit.private.json").write_text(json.dumps(audit), encoding="utf-8")
	if brick_size is not None:
		(ep_dir / "replay").mkdir(exist_ok=True)
		(ep_dir / "replay" / "scene.json").write_text(
			json.dumps({"layout": {"brick_size": brick_size}}), encoding="utf-8"
		)
	return ep_dir


def _yaw_quat(deg: float) -> list[float]:
	rad = math.radians(deg)
	return [0.0, 0.0, round(math.sin(rad / 2), 6), round(math.cos(rad / 2), 6)]


def test_rules_table_and_root_causes_contract():
	assert [r.id for r in RULES] == [f"R{i}" for i in range(1, 9)]
	for cause in (
		"perception_pose_error",
		"calibration_stale",
		"bowl_localization",
		"wrong_bowl_carry",
		"tool_contract",
	):
		assert cause in ROOT_CAUSES
	assert len(ROOT_CAUSES) == len(set(ROOT_CAUSES))


def test_success_episode(tmp_path):
	ep = write_episode(
		tmp_path,
		"ok",
		events=[
			{"kind": "view_truth", "id": "0:1:wrist_cam", "true_face_pixels": [1000, 1000, 1000, 1000, 1000, 0]},
			{
				"kind": "view",
				"observation": {
					"id": "0:1:wrist_cam",
					"camera": "wrist_cam",
					"faces": [0, 1, 2, 3, 4],
					"score": -0.5,
					"threshold": 0.0,
				},
			},
			{"kind": "view_truth", "id": "0:2:top", "true_face_pixels": [0, 0, 0, 0, 0, 800]},
			{
				"kind": "view",
				"observation": {"id": "0:2:top", "camera": "top", "faces": [5], "score": -0.5, "threshold": 0.0},
			},
		],
		result={"verdict": "SUCCESS", "label": "normal", "label_correct": True},
	)
	row = triage_episode(ep)
	assert isinstance(row, TriageRow)
	assert row.stage == "ok"
	assert row.code is None
	assert row.root_cause == "none"
	assert row.rule == "none"
	assert row.to_dict()["root_cause"] == "none"


def test_r1_bowl_not_found(tmp_path):
	ep = write_episode(
		tmp_path,
		"r1",
		events=[
			{"kind": "bowls_perceived", "positions_m": {"green": None, "pink": [0.15, -0.14]}},
			{"kind": "failure", "code": "BOWL_NOT_FOUND", "node": "prepare", "frame": 0},
		],
		result={"verdict": "FAILED", "failures": [{"code": "BOWL_NOT_FOUND", "frame": 0}]},
	)
	row = triage_episode(ep)
	assert row.rule == "R1"
	assert row.root_cause == "bowl_localization"
	assert row.code == "BOWL_NOT_FOUND"
	assert row.stage == "prepare"
	assert row.evidence["code"] == "BOWL_NOT_FOUND"


def test_r2_yaw_error(tmp_path):
	ep = write_episode(
		tmp_path,
		"r2_plan",
		events=[
			{"kind": "brick_perceived", "found": True, "yaw_deg": 30.0, "frame": 10},
			{"kind": "view_truth", "brick_yaw_deg": 5.0, "frame": 10},
			{"kind": "failure", "code": "SCAN_INCOMPLETE", "node": "scan_wrist", "frame": 400},
		],
		result={"verdict": "FAILED", "label": "normal"},
	)
	row = triage_episode(ep)
	assert row.rule == "R2"
	assert row.root_cause == "perception_pose_error"
	assert row.stage == "perception"
	assert row.code == "SCAN_INCOMPLETE"
	assert abs(row.evidence["yaw_err_deg"] - 25.0) < 1e-6


def test_r2_real_quaternion_and_symmetry_modulo(tmp_path):
	# 2x2 square brick (90-deg symmetry): 87 deg diff == 3 deg mod 90 -> NOT R2.
	ep_2x2_ok = write_episode(
		tmp_path,
		"r2_2x2_sym",
		events=[
			{
				"kind": "view_truth",
				"node": "scan_wrist",
				"frame": 326,
				"true_brick_quat_xyzw": _yaw_quat(-46.8),
				"belief_brick_quat_xyzw": _yaw_quat(40.2),
			},
			{"kind": "failure", "code": "OTHER_FAIL", "node": "place", "frame": 500},
		],
		result={"verdict": "FAILED", "label": "normal"},
		brick_size=[0.0318, 0.0318, 0.0192],
	)
	assert triage_episode(ep_2x2_ok).rule != "R2"

	# 2x4 rectangular brick (180-deg symmetry): 87 deg diff == 87 deg mod 180 -> R2!
	ep_2x4_err = write_episode(
		tmp_path,
		"r2_2x4_err",
		events=[
			{
				"kind": "view_truth",
				"node": "scan_wrist",
				"frame": 326,
				"true_brick_quat_xyzw": _yaw_quat(-46.8),
				"belief_brick_quat_xyzw": _yaw_quat(40.2),
			},
		],
		result={"verdict": "FAILED", "label": "unknown"},
		brick_size=[0.0636, 0.0318, 0.0192],
	)
	row_2x4 = triage_episode(ep_2x4_err)
	assert row_2x4.rule == "R2"
	assert row_2x4.root_cause == "perception_pose_error"
	assert row_2x4.evidence["symmetry_deg"] == 180.0
	assert abs(row_2x4.evidence["yaw_err_deg"] - 87.0) < 0.1

	# 2x2 square brick with 70.2 deg raw diff -> 19.8 deg mod 90 -> R2!
	ep_2x2_err = write_episode(
		tmp_path,
		"r2_2x2_err",
		events=[
			{
				"kind": "view_truth",
				"node": "scan_wrist",
				"frame": 337,
				"true_brick_quat_xyzw": _yaw_quat(-76.7),
				"belief_brick_quat_xyzw": _yaw_quat(-6.6),
			},
		],
		result={"verdict": "FAILED", "label": "normal"},
		brick_size=[0.0318, 0.0318, 0.0192],
	)
	row_2x2 = triage_episode(ep_2x2_err)
	assert row_2x2.rule == "R2"
	assert row_2x2.evidence["symmetry_deg"] == 90.0
	assert abs(row_2x2.evidence["yaw_err_deg"] - 19.9) < 0.2


def test_r3_drift_grasp_no_clamp(tmp_path):
	ep = write_episode(
		tmp_path,
		"r3",
		events=[
			{"kind": "camera_calibrated", "calibrated_on": "nominal", "frame": 0},
			{"kind": "segment", "segment_kind": "grasp", "node": "present_bottom", "frame": 227},
			{"kind": "failure", "code": "no_clamp", "skill": "present_bottom", "node": "recover", "frame": 776},
		],
		result={"verdict": "FAILED", "label": "unknown"},
		experiment={"camera_condition": "drift"},
	)
	row = triage_episode(ep)
	assert row.rule == "R3"
	assert row.root_cause == "calibration_stale"
	assert row.stage == "grasp"
	assert row.code == "no_clamp"
	assert row.evidence["camera_condition"] == "drift"


def test_r4_wrong_bowl_carry(tmp_path):
	ep = write_episode(
		tmp_path,
		"r4",
		events=[
			{"kind": "segment", "segment_kind": "carry_to_bowl", "params": {"bowl": "green"}, "frame": 800},
			{"kind": "placement_check", "fill_pixels": {"green": 120, "pink": 1410}, "frame": 900},
			{"kind": "failure", "code": "PLACEMENT_UNVERIFIED", "skill": "place", "node": "recover", "frame": 900},
		],
		result={"verdict": "FAILED", "label": "normal", "label_correct": True, "destination": "pink"},
	)
	row = triage_episode(ep)
	assert row.rule == "R4"
	assert row.root_cause == "wrong_bowl_carry"
	assert row.stage == "place"
	assert row.code == "PLACEMENT_UNVERIFIED"
	assert row.evidence["target_bowl"] == "green"
	assert row.evidence["actual_bowl"] == "pink"


def test_r5_evidence_insufficient(tmp_path):
	ep = write_episode(
		tmp_path,
		"r5",
		events=[
			{"kind": "view_truth", "id": "0:1:wrist_cam", "true_face_pixels": [0, 1200, 1200, 1200, 1200, 0]},
			{
				"kind": "view",
				"observation": {
					"id": "0:1:wrist_cam",
					"camera": "wrist_cam",
					"faces": [0, 1, 2, 3, 4],
					"score": -0.9,
					"threshold": 0.0,
				},
			},
			{"kind": "view_truth", "id": "0:2:top", "true_face_pixels": [0, 0, 0, 0, 0, 1000]},
			{
				"kind": "view",
				"observation": {"id": "0:2:top", "camera": "top", "faces": [5], "score": -0.9, "threshold": 0.0},
			},
		],
		result={
			"verdict": "SUCCESS",
			"label": "normal",
			"label_correct": True,
			"evidence": ["0:1:wrist_cam", "0:2:top"],
		},
	)
	row = triage_episode(ep)
	assert row.rule == "R5"
	assert row.root_cause == "evidence_insufficient"
	assert row.stage == "inspect"
	assert row.evidence["missing_faces"] == [0]


def test_r6_grasp_or_carry_failure(tmp_path):
	# Fixed camera (not drift) with no_clamp -> R6, not R3
	ep_fixed = write_episode(
		tmp_path,
		"r6_noclamp",
		events=[
			{"kind": "segment", "segment_kind": "grasp", "node": "place", "frame": 400},
			{"kind": "failure", "code": "no_clamp", "skill": "place", "node": "recover", "frame": 600},
		],
		result={"verdict": "FAILED", "label": "normal"},
		experiment={"camera_condition": "fixed"},
	)
	row_fixed = triage_episode(ep_fixed)
	assert row_fixed.rule == "R6"
	assert row_fixed.root_cause == "grasp_or_path_failure"
	assert row_fixed.stage == "grasp"
	assert row_fixed.code == "no_clamp"

	# BRICK_LOST during carry_to_bowl -> R6
	ep_lost = write_episode(
		tmp_path,
		"r6_lost",
		events=[
			{"kind": "segment", "segment_kind": "carry_to_bowl", "node": "place", "frame": 500},
			{"kind": "failure", "code": "BRICK_LOST", "skill": "place", "node": "recover", "frame": 627},
		],
		result={"verdict": "FAILED", "label": "defective"},
		experiment={"camera_condition": "drift"},
	)
	row_lost = triage_episode(ep_lost)
	assert row_lost.rule == "R6"
	assert row_lost.root_cause == "grasp_or_path_failure"
	assert row_lost.code == "BRICK_LOST"

	# Collision during carry -> R6
	ep_col = write_episode(
		tmp_path,
		"r6_col",
		events=[
			{"kind": "segment", "segment_kind": "carry_to_bowl", "node": "place", "frame": 500},
			{"kind": "failure", "code": "COLLISION", "skill": "place", "frame": 550},
		],
		result={"verdict": "FAILED", "label": "normal"},
	)
	row_col = triage_episode(ep_col)
	assert row_col.rule == "R6"
	assert row_col.root_cause == "grasp_or_path_failure"
	assert row_col.code == "COLLISION"


def test_r7_false_positive_and_false_negative_visible(tmp_path):
	ep_fp = write_episode(
		tmp_path,
		"r7_fp",
		events=[
			{
				"kind": "view_truth",
				"id": "0:637:wrist_cam",
				"true_face_pixels": [0, 1500, 0, 0, 10000, 0],
				"true_face_pixels_crop_native": [0, 1400, 0, 0, 9000, 0],
			},
			{
				"kind": "view",
				"observation": {
					"id": "0:637:wrist_cam",
					"camera": "wrist_cam",
					"faces": [1, 4],
					"score": 0.25,
					"threshold": 0.0,
				},
			},
		],
		result={
			"verdict": "FAILED",
			"label": "defective",
			"label_correct": False,
			"placement_evidence": "0:637:wrist_cam",
		},
	)
	row_fp = triage_episode(ep_fp)
	assert row_fp.rule == "R7"
	assert row_fp.root_cause == "false_positive_visible"
	assert row_fp.stage == "inspect"
	assert row_fp.code == "wrong_label"

	ep_fn = write_episode(
		tmp_path,
		"r7_fn",
		events=[
			{
				"kind": "view_truth",
				"id": "0:300:wrist_cam",
				"true_face_pixels": [0, 0, 4000, 0, 8000, 0],
				"true_cross_px_crop_native": 120,
			},
			{
				"kind": "view",
				"observation": {
					"id": "0:300:wrist_cam",
					"camera": "wrist_cam",
					"faces": [2, 4],
					"score": -0.4,
					"threshold": 0.0,
				},
			},
		],
		result={"verdict": "FAILED", "label": "normal", "label_correct": False},
	)
	row_fn = triage_episode(ep_fn)
	assert row_fn.rule == "R7"
	assert row_fn.root_cause == "false_negative_visible"
	assert row_fn.stage == "inspect"
	assert row_fn.code == "wrong_label"


def test_r8_tool_contract(tmp_path):
	ep_det = write_episode(
		tmp_path,
		"r8_det",
		events=[{"kind": "detector_error", "node": "scan_wrist", "frame": 120, "reason": "TIMEOUT"}],
		result={"verdict": "FAILED", "execution_error": "DETECTOR_UNAVAILABLE"},
	)
	row_det = triage_episode(ep_det)
	assert row_det.rule == "R8"
	assert row_det.root_cause == "tool_contract"
	assert row_det.stage == "scan_wrist"

	ep_timeout = write_episode(
		tmp_path,
		"r8_timeout",
		events=[{"kind": "failure", "code": "TIMEOUT", "node": "place", "frame": 300}],
		result={"verdict": "FAILED"},
	)
	row_to = triage_episode(ep_timeout)
	assert row_to.rule == "R8"
	assert row_to.root_cause == "tool_contract"
	assert row_to.code == "TIMEOUT"


def test_rule_priority_ordering(tmp_path):
	# R1 + R2 -> R1 wins
	ep_12 = write_episode(
		tmp_path,
		"p12",
		events=[
			{"kind": "brick_perceived", "found": True, "yaw_deg": 40.0, "frame": 10},
			{"kind": "view_truth", "brick_yaw_deg": 5.0, "frame": 10},
			{"kind": "failure", "code": "BOWL_NOT_FOUND", "node": "prepare", "frame": 20},
		],
		result={"verdict": "FAILED"},
	)
	assert triage_episode(ep_12).rule == "R1"

	# R2 + R3 -> R2 wins
	ep_23 = write_episode(
		tmp_path,
		"p23",
		events=[
			{"kind": "brick_perceived", "found": True, "yaw_deg": 35.0, "frame": 10},
			{"kind": "view_truth", "brick_yaw_deg": 5.0, "frame": 10},
			{"kind": "segment", "segment_kind": "grasp", "frame": 200},
			{"kind": "failure", "code": "no_clamp", "frame": 300},
		],
		result={"verdict": "FAILED"},
		experiment={"camera_condition": "drift"},
	)
	assert triage_episode(ep_23).rule == "R2"

	# R3 + R6 -> R3 wins
	ep_36 = write_episode(
		tmp_path,
		"p36",
		events=[
			{"kind": "segment", "segment_kind": "grasp", "frame": 200},
			{"kind": "failure", "code": "no_clamp", "frame": 300},
			{"kind": "segment", "segment_kind": "carry_to_bowl", "frame": 400},
			{"kind": "failure", "code": "BRICK_LOST", "frame": 500},
		],
		result={"verdict": "FAILED"},
		experiment={"camera_condition": "drift"},
	)
	assert triage_episode(ep_36).rule == "R3"


def test_unclassified_fallback(tmp_path):
	ep = write_episode(
		tmp_path,
		"unclass",
		events=[{"kind": "abstain", "node": "place", "frame": 900, "destination": "inspection_table"}],
		result={"verdict": "UNKNOWN", "label": "unknown", "label_correct": False},
	)
	row = triage_episode(ep)
	assert row.root_cause == "unclassified"
	assert row.rule == "unclassified"


P2_DEV_R2 = Path("D:/Armigo_data_v2/runs/p2-dev-r2")
P2_DEV_R2_SCORES = Path("D:/Armigo_data_v2/reports/p2-dev-r2/scores.jsonl")


@pytest.mark.skipif(
	not (P2_DEV_R2.is_dir() and P2_DEV_R2_SCORES.is_file()),
	reason="D:/Armigo_data_v2/runs/p2-dev-r2 dataset not available",
)
def test_calibrate_on_p2_dev_r2():
	scores = [json.loads(line) for line in P2_DEV_R2_SCORES.read_text(encoding="utf-8").splitlines() if line.strip()]
	non_perfect = [r for r in scores if not r["physical_success"] or r["root_cause"] != "none"]
	assert len(non_perfect) == 50

	triaged = {r["path"]: triage_episode(Path(r["path"])) for r in non_perfect}

	# All evidence_insufficient episodes must map to R2 or R5
	ei_rows = [r for r in non_perfect if r["root_cause"] == "evidence_insufficient"]
	assert len(ei_rows) == 15
	assert all(triaged[r["path"]].rule in ("R2", "R5") for r in ei_rows)
	assert sum(triaged[r["path"]].rule == "R2" for r in ei_rows) == 12
	assert sum(triaged[r["path"]].rule == "R5" for r in ei_rows) == 3

	# All false_positive_visible episodes must map to R7
	fp_rows = [r for r in non_perfect if r["root_cause"] == "false_positive_visible"]
	assert len(fp_rows) == 2
	assert all(
		triaged[r["path"]].rule == "R7" and triaged[r["path"]].root_cause == "false_positive_visible" for r in fp_rows
	)

	# All drift grasp failures (no_clamp in grasp) must map to R3 (or higher-priority R2 when yaw > 10 deg)
	drift_grasp_rows = [
		r
		for r in non_perfect
		if r.get("camera_condition") == "drift"
		and any(
			ev.get("kind") == "failure" and ev.get("code") == "no_clamp"
			for ev in (
				json.loads(l)
				for l in (Path(r["path"]) / "events.jsonl").read_text(encoding="utf-8").splitlines()
				if l.strip()
			)
		)
	]
	assert len(drift_grasp_rows) == 27
	assert sum(triaged[r["path"]].rule == "R3" for r in drift_grasp_rows) == 17
	assert all(triaged[r["path"]].rule in ("R2", "R3") for r in drift_grasp_rows)

	# Clean successes (verdict == SUCCESS and attribution == none) map to stage="ok", root_cause="none"
	clean_rows = [
		r for r in scores if r["physical_success"] and r["root_cause"] == "none" and r["scored_verdict"] == "SUCCESS"
	]
	assert len(clean_rows) == 152
	for r in clean_rows[:20]:
		row = triage_episode(Path(r["path"]))
		assert row.stage == "ok" and row.root_cause == "none"

	# Torque-stall episodes (verdict == FAILED despite brick landing in right bowl) surface torque_stall code
	stall_rows = [
		r for r in scores if r["physical_success"] and r["root_cause"] == "none" and r["scored_verdict"] == "FAILED"
	]
	assert len(stall_rows) == 14
	for r in stall_rows[:5]:
		row = triage_episode(Path(r["path"]))
		assert row.code == "torque_stall" and row.root_cause == "unclassified"
