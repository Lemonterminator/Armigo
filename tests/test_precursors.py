import json
from importlib import util
from pathlib import Path

from digital_twin_armigo.experiments.attribution import first_error
from digital_twin_armigo.experiments.precursors import (
	DEFAULT_THRESHOLDS,
	MECHANISMS,
	explain,
	load_thresholds,
	precursors,
)
from digital_twin_armigo.experiments.report import build_summary, mechanism_table, render_markdown
from digital_twin_armigo.experiments.scoring import score_episode

CALIB_DRIFT = {
	"seq": 0,
	"kind": "camera_calibrated",
	"frame": 0,
	"actual_cam_pos_err_mm": 21.4,
	"actual_cam_axis_err_deg": 0.75,
}
CALIB_OK = {
	"seq": 0,
	"kind": "camera_calibrated",
	"frame": 0,
	"actual_cam_pos_err_mm": 0.3,
	"actual_cam_axis_err_deg": 0.1,
}


def brick(seq, xy, yaw):
	return {
		"seq": seq,
		"kind": "brick_perceived",
		"frame": 10 * seq,
		"found": True,
		"error_xy_mm": xy,
		"error_yaw_deg": yaw,
	}


def view(seq, i, faces, px):
	vid = f"0:{i}:wrist_cam"
	base = {"branch": 0, "frame": 100 * seq}
	truth = {
		"seq": seq,
		"kind": "view_truth",
		"id": vid,
		"true_face_pixels": px,
		"true_face_pixels_crop_native": px,
		"true_cross_px": 0,
		"true_cross_px_crop_native": 0,
		"belief_error_mm": 2.0,
		**base,
	}
	obs = {"id": vid, "camera": "wrist_cam", "faces": faces, "score": -0.9, "threshold": 0.0, **base}
	return [truth, {"seq": seq + 1, "kind": "view", "observation": obs, **base}]


def test_precursor_rules_and_thresholds(tmp_path, monkeypatch):
	monkeypatch.setattr("digital_twin_armigo.experiments.precursors.THRESHOLDS_FILE", tmp_path / "missing.json")
	found = precursors(
		[
			CALIB_DRIFT,
			brick(1, 21.0, 2.0),
			{"seq": 2, "kind": "bowls_perceived", "error_xy_mm": {"green": 3.0, "pink": 12.0}},
		]
	)
	assert [p["mechanism"] for p in found] == ["calibration_drift", "brick_pose_error", "bowl_pose_error"]
	assert found[1]["value"] == 21.0 and found[1]["threshold"] == DEFAULT_THRESHOLDS["brick_xy_mm"]
	assert precursors([CALIB_OK, brick(1, 2.0, 1.0)]) == []


def test_drift_grasp_failure_gets_pose_mechanism_and_calibration_upstream():
	events = [CALIB_DRIFT, brick(1, 21.0, 2.0), {"seq": 2, "kind": "failure", "code": "no_clamp", "frame": 1200}]
	episode = {
		"result": {
			"label": "unknown",
			"failures": [{"code": "no_clamp", "frame": 1200}],
			"rewinds": 1,
			"retries": 0,
			"views": 0,
			"evidence": [],
			"cleared_faces": [],
		}
	}
	task = {"case": [7, False, 5], "policy": {"max_rewinds": 1}}
	out = first_error(episode, events, task, {"true_label": "normal"}, {"success": False, "outcome": "table"})
	assert out["root_cause"] == "recovery_failure"
	assert out["mechanism"] == "brick_pose_error" and out["upstream"] == "calibration_drift"
	assert {p["mechanism"] for p in out["precursors"]} == {"calibration_drift", "brick_pose_error"}


def test_evidence_insufficient_gets_yaw_mechanism_without_upstream():
	events = [brick(0, 2.9, 19.8)]
	big = 3000
	for i, f in enumerate([1, 2, 3, 4]):
		events += view(2 * i + 1, i, [f], [big if j == f else 0 for j in range(6)])
	ids = [e["observation"]["id"] for e in events if e["kind"] == "view"]
	episode = {
		"result": {
			"label": "normal",
			"verdict": "SUCCESS",
			"evidence": ids,
			"failures": [],
			"views": 4,
			"cleared_faces": [0, 1, 2, 3, 4, 5],
		}
	}
	out = first_error(episode, events, {"case": [7, False, 5]}, {"true_label": "normal"}, {"success": True})
	assert out["root_cause"] == "evidence_insufficient"
	assert out["mechanism"] == "brick_yaw_error" and out["upstream"] is None


def test_detector_symptoms_and_success_get_no_mechanism():
	found = precursors([CALIB_DRIFT, brick(1, 21.0, 20.0)])
	assert explain("false_positive_visible", 50, found)["mechanism"] is None
	assert explain("none", 10**9, found)["mechanism"] is None


def test_precursor_after_the_symptom_is_ignored():
	found = precursors([brick(5, 21.0, 2.0)])
	assert explain("grasp_or_path_failure", 3, found)["mechanism"] is None


def test_mechanism_table_counts_symptom_by_mechanism():
	rows = [
		{"root_cause": "recovery_failure", "mechanism": "brick_pose_error"},
		{"root_cause": "recovery_failure", "mechanism": "brick_pose_error"},
		{"root_cause": "evidence_insufficient", "mechanism": None},
	]
	assert mechanism_table(rows) == {
		"recovery_failure": {"brick_pose_error": 2},
		"evidence_insufficient": {"(none)": 1},
	}


def test_score_episode_and_report_include_mechanisms(tmp_path):
	import json

	ep = tmp_path / "L7-N"
	ep.mkdir()
	(ep / "task.json").write_text(json.dumps({"case": [7, False, 5], "policy": {"max_rewinds": 1}}))
	(ep / "audit.private.json").write_text(json.dumps({"true_label": "normal", "true_face": None}))
	(ep / "episode.json").write_text(
		json.dumps(
			{
				"provenance": {"engine": "mujoco", "experiment": {"batch": "b", "group": "drift"}},
				"result": {
					"verdict": "FAILED",
					"label": "unknown",
					"failures": [{"code": "no_clamp", "frame": 1200}],
					"rewinds": 1,
					"retries": 0,
					"views": 0,
					"evidence": [],
					"cleared_faces": [],
				},
			}
		)
	)
	events = [CALIB_DRIFT, brick(1, 21.0, 2.0), {"seq": 2, "kind": "failure", "code": "no_clamp", "frame": 1200}]
	(ep / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
	row = score_episode(ep)
	assert row["root_cause"] == "recovery_failure"
	assert row["mechanism"] == "brick_pose_error"
	assert row["upstream"] == "calibration_drift"
	summary = build_summary([row], {}, B=10)
	assert summary["mechanisms"] == {"recovery_failure": {"brick_pose_error": 1}}
	md = render_markdown("b", summary)
	assert "## 症状 × 机制（前兆，未经反事实证实）" in md
	assert "| 症状 | " + " | ".join([*MECHANISMS, "(none)"]) + " |" in md
	assert "| recovery_failure | 0 | 1 | 0 | 0 | 0 | 0 |" in md


_spec = util.spec_from_file_location(
	"calibrate_precursors", Path(__file__).resolve().parents[1] / "scripts/experiments/calibrate_precursors.py"
)
cp = util.module_from_spec(_spec)
_spec.loader.exec_module(cp)


def test_youden_threshold_respects_negative_rate_and_reports_auc():
	r = cp.youden_threshold([20, 22, 25, 30], [1, 2, 3, 2.5, 21], max_neg_rate=0.10)
	assert r["threshold"] == 21 and r["tpr"] == 0.75 and r["fpr"] == 0.0
	assert abs(r["auc"] - 0.95) < 1e-9 and (r["n_pos"], r["n_neg"]) == (4, 5)


def test_youden_threshold_ties_count_half_in_auc():
	assert cp.youden_threshold([5.0], [5.0])["auc"] == 0.5


def test_metric_values_take_first_brick_and_max_bowl():
	events = [
		{"kind": "brick_perceived", "found": True, "error_xy_mm": 3.0, "error_yaw_deg": 1.0},
		{"kind": "brick_perceived", "found": True, "error_xy_mm": 9.0, "error_yaw_deg": 9.0},
		{"kind": "bowls_perceived", "error_xy_mm": {"green": 2.0, "pink": 7.0}},
	]
	assert cp.metric_values(events) == {"brick_xy_mm": 3.0, "brick_yaw_deg": 1.0, "bowl_xy_mm": 7.0}


def test_load_thresholds_reads_calibrated_file(tmp_path, monkeypatch):
	p = tmp_path / "precursor_thresholds.json"
	p.write_text(
		json.dumps({"metrics": {"brick_xy_mm": {"threshold": 14.21}, "cam_pos_mm": {"threshold": 9.5}}}),
		encoding="utf-8",
	)
	monkeypatch.setattr("digital_twin_armigo.experiments.precursors.THRESHOLDS_FILE", p)
	loaded = load_thresholds()
	assert loaded["brick_xy_mm"] == 14.21
	assert loaded["cam_pos_mm"] == 9.5
	assert loaded["view_belief_mm"] == DEFAULT_THRESHOLDS["view_belief_mm"]
