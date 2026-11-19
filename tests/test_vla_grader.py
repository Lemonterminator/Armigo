import json
import math
from pathlib import Path

import numpy as np
import pytest

from digital_twin_armigo.agent.vla_grader import (
	CHECKS,
	DEFAULT_THRESHOLDS,
	face_map,
	grade,
	load_reference,
	smoothness_index,
	summarize,
	symmetry_from_quats,
)

REAL_TRACE = Path(__file__).parent / "data/vla_grader/realistic-600001-face0"
NO_SMOOTHNESS = {"thresholds": DEFAULT_THRESHOLDS}


def _yaw_quat(deg):
	return [0.0, 0.0, math.sin(math.radians(deg) / 2), math.cos(math.radians(deg) / 2)]


def _view(seq, vid, faces, score, true_px, *, belief_yaw=0.0, cross=0, error=2.0, truth=True):
	_, frame, camera = vid.split(":")
	frame = int(frame)
	events = []
	if truth:
		events.append(
			{
				"seq": seq,
				"kind": "view_truth",
				"node": "scan_wrist",
				"branch": 0,
				"frame": frame,
				"id": vid,
				"true_face_pixels": true_px,
				"belief_error_mm": error,
				"true_cross_px": cross,
				"true_brick_quat_xyzw": _yaw_quat(0.0),
				"belief_brick_quat_xyzw": _yaw_quat(belief_yaw),
			}
		)
	events.append(
		{
			"seq": seq + 1,
			"kind": "view",
			"node": "scan_wrist",
			"branch": 0,
			"frame": frame,
			"observation": {
				"id": vid,
				"branch": 0,
				"frame": frame,
				"camera": camera,
				"faces": faces,
				"face_pixels": [0] * 6,
				"score": score,
				"threshold": 0.0,
				"crop_bbox": [0, 0, 1, 1],
			},
		}
	)
	return events


def _episode(tmp_path, name, *, label, true_face, events, result=None, branches=None, sensing="realistic"):
	path = tmp_path / name
	path.mkdir()
	evidence = [e["observation"]["id"] for e in events if e["kind"] == "view"]
	positive = next(
		(e["observation"]["id"] for e in events if e["kind"] == "view" and e["observation"]["score"] >= 0), None
	)
	base = {
		"verdict": "SUCCESS",
		"label": label,
		"rewinds": 0,
		"evidence": evidence,
		"health": {"unintended_collision_count": 0},
		"student_eligible": True,
		"placement_evidence": positive,
	}
	branches = branches or {0: {"contact_robot_n": np.zeros(10), "contact_jaw_env_n": np.zeros(10)}}
	manifest_branches = []
	for bid, arrays in branches.items():
		n = len(next(iter(arrays.values()), np.zeros(10)))
		(path / f"branch-{bid}").mkdir()
		np.savez(
			path / f"branch-{bid}/actions.npz",
			state=np.zeros((n, 6), "float32"),
			action=np.zeros((n, 6), "float32"),
			phase=np.array(["approach"] * n),
			**arrays,
		)
		manifest_branches.append({"id": bid, "start_frame": 0, "actions": n})
	manifest = {
		"schema": "armigo-agent-episode/v1",
		"branches": manifest_branches,
		"provenance": {"sensing": sensing},
		"result": {**base, **(result or {})},
	}
	(path / "episode.json").write_text(json.dumps(manifest), encoding="utf-8")
	(path / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
	true_label = "normal" if true_face is None else "defective"
	(path / "audit.private.json").write_text(json.dumps({"true_label": true_label, "true_face": true_face}))
	return path


def test_face_map_is_the_quarter_turn_about_the_brick_z_axis():
	assert face_map(0) == (0, 1, 2, 3, 4, 5)
	assert face_map(1) == (2, 3, 1, 0, 4, 5)  # +x -> +y, -x -> -y, +y -> -x, -y -> +x
	assert face_map(2) == (1, 0, 3, 2, 4, 5)
	m = list(range(6))
	for _ in range(4):
		m = [face_map(1)[f] for f in m]
	assert m == list(range(6))
	assert symmetry_from_quats(_yaw_quat(10.0), _yaw_quat(100.0 + 3.0))[0] == 1
	k, residual = symmetry_from_quats(_yaw_quat(-80.0), _yaw_quat(12.0))
	assert k == 1 and residual == pytest.approx(2.0, abs=1e-6)


def _rotated_normal_views(belief_yaw):
	# True visible faces {0,2,4} and {1,3,4}; the belief numbers them as a quarter turn: belief 0 is true 2, ...
	return [
		*_view(0, "0:300:wrist_cam", [0, 3, 4], -1.0, [2000, 0, 2000, 0, 9000, 0], belief_yaw=belief_yaw),
		*_view(2, "0:500:wrist_cam", [1, 2, 4], -1.0, [0, 2000, 0, 2000, 9000, 0], belief_yaw=belief_yaw),
		*_view(4, "0:1000:top", [5], -1.0, [0, 0, 0, 0, 0, 6000], belief_yaw=belief_yaw),
	]


def test_symmetry_rotation_of_belief_face_ids_is_aligned_before_comparison(tmp_path):
	path = _episode(tmp_path, "rotated", label="normal", true_face=None, events=_rotated_normal_views(90.0))
	g = grade(path, NO_SMOOTHNESS)
	assert g["checks"]["decision_crop"]["status"] == "pass"
	assert g["checks"]["decision_crop"]["truly_cleared"] == [0, 1, 2, 3, 4, 5]
	s = summarize([g])["alignment"]
	assert (s["views"], s["raw_mismatches"], s["aligned_mismatches"]) == (3, 2, 0)
	assert {a["method"] for a in g["alignment"]} == {"orientation"}


def test_symmetry_comes_from_the_orientation_not_from_the_faces(tmp_path):
	# Same rotated face ids, but the belief orientation says no quarter turn: the claimed faces were not shown.
	path = _episode(tmp_path, "unaligned", label="normal", true_face=None, events=_rotated_normal_views(0.0))
	check = grade(path, NO_SMOOTHNESS)["checks"]["decision_crop"]
	assert check["status"] == "fail"
	assert check["false_claims"]


@pytest.mark.parametrize(
	("face_px", "cross", "status"),
	[(2500, 400, "pass"), (10, 400, "fail"), (2500, 3, "fail"), (2500, None, "unavailable")],
)
def test_defective_decision_view_must_truly_show_the_defect_face_and_cross(tmp_path, face_px, cross, status):
	true_px = [3000, 0, face_px, 0, 9000, 0]
	events = [
		*_view(0, "0:300:wrist_cam", [0, 1, 4], -1.0, [0, 3000, 0, 3000, 9000, 0]),
		*_view(2, "0:500:wrist_cam", [0, 2, 4], 1.5, true_px, cross=cross or 0),
	]
	if cross is None:
		del events[2]["true_cross_px"]
	path = _episode(tmp_path, "defect", label="defective", true_face=2, events=events)
	check = grade(path, NO_SMOOTHNESS)["checks"]["decision_crop"]
	assert check["status"] == status
	assert check["view"] == "0:500:wrist_cam"


def test_unknown_label_collisions_count_jaw_contact_over_all_branches(tmp_path):
	events = _view(0, "0:300:wrist_cam", [0, 2, 4], -1.0, [2000, 0, 2000, 0, 9000, 0])
	unknown = {"verdict": "UNKNOWN", "label": "unknown", "student_eligible": False, "health": {}}
	jaw = np.zeros(20)
	jaw[7] = 4.5  # jaws pressing on the table in the abandoned first attempt
	branches = {
		0: {"contact_robot_n": np.zeros(20), "contact_jaw_env_n": jaw},
		1: {"contact_robot_n": np.zeros(30), "contact_jaw_env_n": np.zeros(30)},
	}
	path = _episode(
		tmp_path, "unknown", label="unknown", true_face=None, events=events, result=unknown, branches=branches
	)
	g = grade(path, NO_SMOOTHNESS)
	collisions = g["checks"]["collisions"]
	assert collisions["status"] == "fail"
	assert (collisions["frames"], collisions["jaw_env_contact_frames"], collisions["robot_contact_frames"]) == (
		50,
		1,
		0,
	)
	assert collisions["first_frames"] == [{"branch": 0, "frame": 7}]
	assert g["checks"]["eligible"]["status"] == "fail"
	assert not g["passed"]


def test_old_trace_without_contact_record_is_unavailable_not_pass(tmp_path):
	events = _view(0, "0:300:wrist_cam", [0, 2, 4], -1.0, [2000, 0, 2000, 0, 9000, 0])
	path = _episode(tmp_path, "old", label="unknown", true_face=None, events=events, branches={0: {}})
	g = grade(path, NO_SMOOTHNESS)
	assert g["checks"]["collisions"]["status"] == "unavailable"
	assert "collisions" in g["unavailable"] and not g["passed"]


def test_eligibility_reads_student_eligible_and_reports_retries(tmp_path):
	events = _rotated_normal_views(90.0)
	path = _episode(tmp_path, "retried", label="normal", true_face=None, events=events, result={"retries": 1})
	check = grade(path, NO_SMOOTHNESS)["checks"]["eligible"]
	assert check["status"] == "pass" and check["retries"] == 1 and check["rewinds"] == 0


def test_belief_error_bound_per_camera(tmp_path):
	events = [
		*_view(0, "0:300:wrist_cam", [0, 2, 4], -1.0, [2000, 0, 2000, 0, 9000, 0], error=7.0),
		*_view(2, "0:1000:top", [5], -1.0, [0, 0, 0, 0, 0, 6000], error=7.0),
	]
	path = _episode(tmp_path, "belief", label="unknown", true_face=None, events=events)
	check = grade(path, NO_SMOOTHNESS)["checks"]["belief_error"]
	assert check["status"] == "fail"
	assert check["over"] == [{"view": "0:300:wrist_cam", "error_mm": 7.0}]  # 7 mm is within the in-hand bound


def test_smoothness_index_normalizes_each_phase_by_the_expert_median():
	medians = {"approach": [2.0, 4.0], "rest": [0.0, 10.0]}
	# 10 frames of approach at twice the median, 10 frames of commanded rest (floored median).
	per_phase = {"approach": (10, 10 * 16.0, 10 * 16.0), "rest": (10, 0.0, 10 * 100.0)}
	index = smoothness_index(per_phase, medians)
	assert index["jerk_index_cmd"] == pytest.approx((10 * 2.0 + 0.0) / 20)
	assert index["jerk_index_meas"] == pytest.approx((10 * 1.0 + 10 * 1.0) / 20)
	assert smoothness_index({"handoff": (5, 1.0, 1.0)}, medians) is None


def test_summary_flags_classes_below_eight_tenths_of_the_mean():
	def g(cls, passed):
		status = "pass" if passed else "fail"
		checks = {c: {"status": status} for c in CHECKS}
		return {
			"class": cls,
			"passed": passed,
			"passed_available": passed,
			"alignment": [],
			"unavailable": [],
			"checks": checks,
		}

	grades = [g("normal", True)] * 4 + [g("defect_face_1", False), g("defect_face_1", True)]
	s = summarize(grades)
	assert s["passed_rate"] == pytest.approx(5 / 6, abs=1e-4)
	assert s["classes"]["defect_face_1"]["passed_flag"]  # 0.5 < 0.8 x 0.833
	assert not s["classes"]["normal"]["passed_flag"]
	assert s["failed"]["eligible"] == 1


def test_real_realistic_trace_passes_with_a_quarter_turn_alignment():
	if not REAL_TRACE.exists():
		pytest.skip("real trace fixture missing")
	g = grade(REAL_TRACE, load_reference(), rebuild_layout=False)
	assert g["class"] == "defect_face_0"
	assert g["passed"], {k: c for k, c in g["checks"].items() if c["status"] != "pass"}
	assert g["checks"]["decision_crop"]["true_cross_px"] >= 25
	assert [a["k"] for a in g["alignment"]] == [3, 3]  # believed yaw -14 deg, true yaw 76 deg
	assert [a["raw_match"] for a in g["alignment"]] == [False, False]
	assert all(a["aligned_match"] for a in g["alignment"])
