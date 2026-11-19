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

REAL_TRACE = Path(__file__).parent / "data/vla_grader/realistic-600001-face0"  # recorded with crop-space truth
LEGACY_TRACE = Path(__file__).parent / "data/vla_grader/realistic-600001-face0-legacy"  # whole-frame truth only
NO_SMOOTHNESS = {"thresholds": DEFAULT_THRESHOLDS}


def _yaw_quat(deg):
	return [0.0, 0.0, math.sin(math.radians(deg) / 2), math.cos(math.radians(deg) / 2)]


BBOX = [286, 144, 206, 198]


def _view(
	seq,
	vid,
	faces,
	score,
	true_px,
	*,
	belief_yaw=0.0,
	cross=0,
	cross_input=None,
	frame_cross=None,
	crop_px=None,
	bbox=BBOX,
	truth_bbox=BBOX,
	error=2.0,
	truth=True,
):
	"""A view and its truth record. ``true_px`` / ``cross`` are the crop-space counts (camera pixels inside the
	detector crop); ``crop_px`` / ``frame_cross`` override the full-frame counts (default: the same)."""
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
				"true_face_pixels": true_px if crop_px is None else crop_px,
				"belief_error_mm": error,
				"true_cross_px": cross if frame_cross is None else frame_cross,
				"true_face_pixels_crop": true_px,
				"true_face_pixels_crop_native": true_px,
				"true_cross_px_crop": cross if cross_input is None else cross_input,
				"true_cross_px_crop_native": cross,
				"truth_crop": {"bbox": truth_bbox, "size": 256, "pad": 0.25, "clamp": True},
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
				"crop_bbox": bbox,
			},
		}
	)
	return events


def _episode(tmp_path, name, *, label, true_face, events, result=None, branches=None, sensing="realistic"):
	path = tmp_path / name
	path.mkdir()
	evidence = [e["observation"]["id"] for e in events if e["kind"] == "view"]
	positive = next(
		(
			e["observation"]["id"]
			for e in events
			if e["kind"] == "view" and e["observation"]["score"] is not None and e["observation"]["score"] >= 0
		),
		None,
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
		arrays = dict(arrays)
		n = arrays.pop("frames", None)  # action count, when the contact columns should not set it
		if n is None:
			n = len(next(iter(arrays.values()), np.zeros(10)))
		(path / f"branch-{bid}").mkdir()
		np.savez(
			path / f"branch-{bid}/actions.npz",
			**{
				"state": np.zeros((n, 6), "float32"),
				"action": np.zeros((n, 6), "float32"),
				"phase": np.array(["approach"] * n),
				**arrays,
			},
		)
		manifest_branches.append({"id": bid, "start_frame": 0, "actions": n})
	manifest = {
		"schema": "armigo-agent-episode/v1",
		"branches": manifest_branches,
		"provenance": {} if sensing is None else {"sensing": sensing},
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
	if cross is None:  # a trace recorded before crop-space truth: only the whole-frame counts
		for key in ("truth_crop", "true_face_pixels_crop", "true_face_pixels_crop_native", "true_cross_px_crop"):
			del events[2][key]
		del events[2]["true_cross_px_crop_native"]
	path = _episode(tmp_path, "defect", label="defective", true_face=2, events=events)
	check = grade(path, NO_SMOOTHNESS)["checks"]["decision_crop"]
	assert check["status"] == status
	assert check["view"] == "0:500:wrist_cam"


def _defect_events(**decision):
	return [
		*_view(0, "0:300:wrist_cam", [0, 1, 4], -1.0, [0, 3000, 0, 3000, 9000, 0]),
		*_view(2, "0:500:wrist_cam", [0, 2, 4], 1.5, [3000, 0, 2500, 0, 9000, 0], **decision),
	]


def test_cross_in_the_frame_but_outside_the_detector_crop_fails(tmp_path):
	events = _defect_events(cross=0, frame_cross=499)
	path = _episode(tmp_path, "outside", label="defective", true_face=2, events=events)
	g = grade(path, NO_SMOOTHNESS)
	check = g["checks"]["decision_crop"]
	assert check["status"] == "fail"
	assert (check["true_cross_px_frame"], check["true_cross_px_crop_native"]) == (499, 0)
	assert summarize([g])["defective_decision_crops"]["cross_in_frame_only"] == 1


@pytest.mark.parametrize(("native", "input_px"), [(12, 20), (30, 5)])
def test_cross_cut_at_the_crop_edge_with_too_few_pixels_fails(tmp_path, native, input_px):
	# 12 camera pixels left inside the crop; or 30 camera pixels shrunk to 5 detector-input pixels.
	events = _defect_events(cross=native, cross_input=input_px, frame_cross=499)
	path = _episode(tmp_path, "edge", label="defective", true_face=2, events=events)
	check = grade(path, NO_SMOOTHNESS)["checks"]["decision_crop"]
	assert check["status"] == "fail"
	assert "cross" in check["reason"]


def test_truth_for_another_crop_than_the_detector_got_fails(tmp_path):
	# The auditor's control: the recorded crop is a 1x1 box at the top-left, the truth was taken elsewhere.
	events = _defect_events(cross=400, bbox=[0, 0, 1, 1])
	path = _episode(tmp_path, "bbox", label="defective", true_face=2, events=events)
	check = grade(path, NO_SMOOTHNESS)["checks"]["decision_crop"]
	assert check["status"] == "fail"
	assert "detector got [0, 0, 1, 1]" in check["reason"]


def test_one_pixel_crop_of_a_visible_cross_holds_no_cross():
	from digital_twin_armigo.mujoco_cell.inspection_views import crop_counts

	face = np.full((480, 640), -1)
	face[200:300, 300:400] = 2
	defect = np.zeros((480, 640), bool)
	defect[240:260, 340:360] = True  # 400 px of cross in the middle of face 2
	real = crop_counts(face, defect, (300, 200, 100, 100), 256, clamp=True)
	assert real["cross_native"] == 400 and real["faces_native"][2] == 10000
	assert real["cross_input"] == pytest.approx(400 * real["input_px_per_native_px"], rel=0.1)
	one = crop_counts(face, defect, (0, 0, 1, 1), 256, clamp=True)
	assert (one["cross_native"], one["cross_input"], one["faces_native"][2]) == (0, 0, 0)
	edge = crop_counts(face, defect, (350, 200, 100, 100), 256, pad=0.0)  # the crop's left edge cuts the cross
	assert 0 < edge["cross_native"] < 400 and edge["region"][0] == 350


def test_normal_view_must_show_the_cleared_faces_inside_its_crop(tmp_path):
	events = _rotated_normal_views(90.0)
	events[0]["true_face_pixels_crop_native"] = [2000, 0, 300, 0, 9000, 0]  # face 2 mostly outside the crop
	path = _episode(tmp_path, "normal-crop", label="normal", true_face=None, events=events)
	check = grade(path, NO_SMOOTHNESS)["checks"]["decision_crop"]
	assert check["status"] == "fail"
	assert {"view": "0:300:wrist_cam", "true_face": 2, "true_px_crop_native": 300} in check["false_claims"]


def test_nan_single_element_contact_columns_are_unavailable(tmp_path):
	# The auditor's control: a 1,035-frame branch whose contact columns hold one NaN.
	events = _rotated_normal_views(90.0)
	nan = {"frames": 1035, "contact_robot_n": np.array([np.nan]), "contact_jaw_env_n": np.array([np.nan])}
	path = _episode(tmp_path, "nan", label="normal", true_face=None, events=events, branches={0: nan})
	g = grade(path, NO_SMOOTHNESS)
	assert g["checks"]["collisions"]["status"] == "unavailable"
	assert not g["passed"]


@pytest.mark.parametrize("bad", ["short", "nan", "negative"])
def test_contact_columns_must_be_full_length_finite_and_non_negative(tmp_path, bad):
	jaw = np.zeros(20)
	if bad == "short":
		jaw = jaw[:19]
	else:
		jaw[5] = np.nan if bad == "nan" else -1.0
	branches = {0: {"frames": 20, "contact_robot_n": np.zeros(20), "contact_jaw_env_n": jaw}}
	path = _episode(
		tmp_path, bad, label="normal", true_face=None, events=_rotated_normal_views(90.0), branches=branches
	)
	assert grade(path, NO_SMOOTHNESS)["checks"]["collisions"]["status"] == "unavailable"


SMOOTH_REF = {
	"thresholds": DEFAULT_THRESHOLDS,
	"smoothness": {
		"phase_median_rms": {"approach": [1.0, 1.0]},
		"p95": {"jerk_index_cmd": 2.0, "jerk_index_meas": 2.0},
	},
}


def test_missing_branch_file_makes_collisions_and_smoothness_unavailable(tmp_path):
	branches = {
		0: {"contact_robot_n": np.zeros(20), "contact_jaw_env_n": np.zeros(20)},
		1: {"contact_robot_n": np.zeros(30), "contact_jaw_env_n": np.zeros(30)},
	}
	events = _rotated_normal_views(90.0)
	path = _episode(tmp_path, "missing", label="normal", true_face=None, events=events, branches=branches)
	assert grade(path, SMOOTH_REF)["checks"]["smoothness"]["status"] == "pass"
	(path / "branch-1/actions.npz").unlink()
	g = grade(path, SMOOTH_REF)
	assert g["checks"]["collisions"]["status"] == "unavailable"
	assert "branch 1: actions.npz missing for 30 actions" in g["checks"]["collisions"]["problems"]
	assert g["checks"]["smoothness"]["status"] == "unavailable"
	assert not g["passed"]


def test_non_finite_actions_make_smoothness_unavailable(tmp_path):
	events = _rotated_normal_views(90.0)
	path = _episode(tmp_path, "nan-actions", label="normal", true_face=None, events=events)
	with np.load(path / "branch-0/actions.npz") as z:
		arrays = {k: z[k] for k in z.files}
	arrays["action"][4, 0] = np.nan
	np.savez(path / "branch-0/actions.npz", **arrays)
	assert grade(path, SMOOTH_REF)["checks"]["smoothness"]["status"] == "unavailable"


def test_belief_error_nan_fails_and_no_scored_view_or_unrecorded_sensing_is_unavailable(tmp_path):
	events = _view(0, "0:300:wrist_cam", [0, 2, 4], -1.0, [2000, 0, 2000, 0, 9000, 0], error=float("nan"))
	path = _episode(tmp_path, "nan-belief", label="unknown", true_face=None, events=events)
	assert grade(path, NO_SMOOTHNESS)["checks"]["belief_error"]["status"] == "fail"
	events = _view(0, "0:300:wrist_cam", [0, 2, 4], None, [2000, 0, 2000, 0, 9000, 0])
	path = _episode(tmp_path, "unscored", label="unknown", true_face=None, events=events)
	assert grade(path, NO_SMOOTHNESS)["checks"]["belief_error"]["status"] == "unavailable"
	events = _view(0, "0:300:wrist_cam", [0, 2, 4], -1.0, [2000, 0, 2000, 0, 9000, 0])
	path = _episode(tmp_path, "no-sensing", label="unknown", true_face=None, events=events, sensing=None)
	assert grade(path, NO_SMOOTHNESS)["checks"]["belief_error"]["status"] == "unavailable"


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
	crop = g["checks"]["decision_crop"]
	assert crop["true_cross_px_crop_native"] == crop["true_cross_px_frame"] >= 25  # the whole cross is in the crop
	assert crop["crop_size"] == 256 and crop["true_cross_px_crop"] >= 10
	assert [a["k"] for a in g["alignment"]] == [3, 3]  # believed yaw -14 deg, true yaw 76 deg
	assert [a["raw_match"] for a in g["alignment"]] == [False, False]
	assert all(a["aligned_match"] for a in g["alignment"])


def _copy_trace(src, dst):
	import shutil

	shutil.copytree(src, dst)
	return dst


def _rewrite_events(path, change):
	lines = (path / "events.jsonl").read_text(encoding="utf-8").splitlines()
	events = [change(json.loads(line)) for line in lines if line]
	(path / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")


def test_real_trace_with_a_one_pixel_crop_bbox_fails(tmp_path):
	# The auditor's negative control: the recorded detector crop becomes a 1x1 box at the top-left.
	if not REAL_TRACE.exists():
		pytest.skip("real trace fixture missing")
	path = _copy_trace(REAL_TRACE, tmp_path / "bbox")

	def one_pixel(e):
		if e["kind"] == "view":
			e["observation"]["crop_bbox"] = [0, 0, 1, 1]
		return e

	_rewrite_events(path, one_pixel)
	g = grade(path, load_reference(), rebuild_layout=False)
	assert g["checks"]["decision_crop"]["status"] == "fail" and not g["passed"]


def test_real_trace_with_truncated_contact_columns_is_unavailable(tmp_path):
	if not REAL_TRACE.exists():
		pytest.skip("real trace fixture missing")
	path = _copy_trace(REAL_TRACE, tmp_path / "nan")
	with np.load(path / "branch-0/actions.npz") as z:
		arrays = {k: z[k] for k in z.files}
	arrays["contact_robot_n"] = arrays["contact_jaw_env_n"] = np.array([np.nan], "float32")
	np.savez(path / "branch-0/actions.npz", **arrays)
	g = grade(path, load_reference(), rebuild_layout=False)
	assert g["checks"]["collisions"]["status"] == "unavailable" and not g["passed"]


def test_legacy_real_trace_without_crop_truth_is_unavailable_not_pass():
	if not LEGACY_TRACE.exists():
		pytest.skip("legacy trace fixture missing")
	g = grade(LEGACY_TRACE, load_reference(), rebuild_layout=False)
	check = g["checks"]["decision_crop"]
	assert check["status"] == "unavailable" and "crop-space truth" in check["reason"]
	assert not g["passed"] and g["passed_available"]
