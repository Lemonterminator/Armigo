import json
from pathlib import Path

import numpy as np
import pytest

from digital_twin_armigo.experiments.contract import ROOT_CAUSES
from digital_twin_armigo.experiments.scoring import classify_pose, score_episode

GREEN = [0.10, 0.20]
PINK = [0.24, -0.10]
BRICK = [0.18, 0.03]
POLICY = {"order": "wrist_first", "max_bottom_views": 2, "max_wrist_views": 2, "max_rewinds": 1}
BIG = 3000


def view_events(seq, vid, camera, faces, score, true_px=None, crop_px=None, cross=0, cross_crop=0, thr=0.0):
	"""``view_truth`` + ``view`` events like ``TwinMujoco._capture``."""
	true_px = true_px or [0] * 6
	crop_px = crop_px if crop_px is not None else true_px
	frame = 100 * (seq + 1)
	base = {"branch": 0, "frame": frame}
	truth = {
		"seq": seq,
		"kind": "view_truth",
		"id": vid,
		"true_face_pixels": true_px,
		"true_face_pixels_crop_native": crop_px,
		"true_cross_px": cross,
		"true_cross_px_crop_native": cross_crop,
		**base,
	}
	obs = {"id": vid, "branch": 0, "frame": frame, "camera": camera, "faces": faces, "score": score, "threshold": thr}
	return [truth, {"seq": seq + 1, "kind": "view", "observation": obs, **base}]


def wrist(i, faces, score, **kw):
	px = kw.pop("true_px", None) or [BIG if f in faces else 0 for f in range(6)]
	return view_events(2 * i, f"0:{i}:wrist_cam", "wrist_cam", faces, score, true_px=px, **kw)


def make(
	tmp_path,
	name,
	case,
	label,
	events,
	final=(*GREEN, 0.0104),
	verdict="SUCCESS",
	replay=True,
	evidence=True,
	**result_extra,
):
	d = tmp_path / name
	d.mkdir()
	positive = case[1]
	(d / "task.json").write_text(json.dumps({"case": list(case), "policy": POLICY}))
	(d / "audit.private.json").write_text(
		json.dumps({"true_label": "defective" if positive else "normal", "true_face": case[2] if positive else None})
	)
	views = [e for e in events if e["kind"] == "view"]
	result = {
		"verdict": verdict,
		"label": label,
		"label_correct": label == ("defective" if positive else "normal"),
		"destination": "green",
		"rewinds": 0,
		"retries": 0,
		"failures": [],
		"views": len(views),
		"sim_frames": 1000,
		"evidence": [v["observation"]["id"] for v in views] if evidence else [],
		"cleared_faces": sorted({f for v in views for f in v["observation"]["faces"]}),
		"health": {},
		**result_extra,
	}
	(d / "episode.json").write_text(json.dumps({"provenance": {"engine": "mujoco"}, "result": result}))
	with (d / "events.jsonl").open("w") as f:
		for i, e in enumerate(events):
			f.write(json.dumps({"wall_time_s": 0.1 * i, **e}) + "\n")
	if replay:
		(d / "replay").mkdir()
		scene = {
			"nodes": [{"name": "table", "id": "geom_0"}, {"name": "brick", "id": "geom_1"}],
			"layout": {"green": GREEN, "pink": PINK, "brick_xy": BRICK},
		}
		(d / "replay" / "scene.json").write_text(json.dumps(scene))
		start = [0, 0, 0, 0, 0, 0, 1.0]
		first = [BRICK[0], BRICK[1], 0.01, 0, 0, 0, 1.0]
		last = [*final, 0, 0, 0, 1.0]
		np.savez(d / "replay" / "motion.npz", poses=np.array([[start, first], [start, last]]), timestamps=[0.0, 1.0])
	return d


NORMAL = (7, False, 5)
DEFECT = (7, True, 2)


def normal_views(covering=range(5), top=True):
	events = []
	for i, f in enumerate(covering):
		events += wrist(i, [f], -0.9)
	if top:
		top_px = [0, 0, 0, 0, 0, BIG]
		events += view_events(20, "0:20:top", "top", [5], -0.8, true_px=top_px)
	return events


def cause(row):
	assert row["root_cause"] in ROOT_CAUSES
	return row["first_error"]["root_cause"]


def test_classify_pose():
	assert classify_pose([*GREEN, 0.0104], GREEN, PINK) == "green_bowl"
	assert classify_pose([PINK[0] + 0.03, PINK[1], 0.02], GREEN, PINK) == "pink_bowl"
	assert classify_pose([0.4, 0.4, 0.01], GREEN, PINK) == "table"
	assert classify_pose([0.4, 0.4, -0.5], GREEN, PINK) == "lost"
	assert classify_pose([2.0, 0.0, 0.01], GREEN, PINK) == "lost"
	assert classify_pose([0.4, 0.4, 0.2], GREEN, PINK) == "unknown"
	assert classify_pose([*GREEN, 0.2], GREEN, PINK) == "unknown"


def test_success_none(tmp_path):
	row = score_episode(make(tmp_path, "ok", NORMAL, "normal", normal_views()))
	assert cause(row) == "none"
	assert row["physical_outcome"] == "green_bowl"
	assert row["physical_success"] and row["label_correct"]
	assert row["n_scored_views"] == 6 and row["n_top_views"] == 1 and row["n_wrist_views"] == 5


def test_outcomes_from_pose(tmp_path):
	pink = score_episode(make(tmp_path, "p", DEFECT, "defective", [], final=(PINK[0], PINK[1], 0.01)))
	assert pink["physical_outcome"] == "pink_bowl" and pink["physical_success"]
	table = score_episode(make(tmp_path, "t", NORMAL, "normal", [], final=(0.4, 0.4, 0.01)))
	assert table["physical_outcome"] == "table" and not table["physical_success"]
	lost = score_episode(make(tmp_path, "l", NORMAL, "normal", [], final=(0.4, 0.4, -0.3)))
	assert lost["physical_outcome"] == "lost"
	air = score_episode(make(tmp_path, "a", NORMAL, "normal", [], final=(0.4, 0.4, 0.3)))
	assert air["physical_outcome"] == "unknown"


def test_no_replay_falls_back_to_result(tmp_path):
	row = score_episode(make(tmp_path, "nr", NORMAL, "normal", normal_views(), replay=False))
	assert row["physical"]["source"] == "result"
	assert row["physical_outcome"] == "green_bowl" and row["physical_success"]


def test_evidence_insufficient_normal(tmp_path):
	row = score_episode(make(tmp_path, "ei", NORMAL, "normal", normal_views(covering=[0, 1, 2, 4])))
	assert cause(row) == "evidence_insufficient"


def test_evidence_insufficient_defective(tmp_path):
	events = wrist(0, [2], 1.5, cross=40, cross_crop=10)
	d = make(
		tmp_path,
		"ed",
		DEFECT,
		"defective",
		events,
		final=(*PINK, 0.01),
		destination="pink",
		placement_evidence="0:0:wrist_cam",
	)
	assert cause(score_episode(d)) == "evidence_insufficient"
	events = wrist(0, [2], 1.5, cross=60, cross_crop=40)
	d = make(
		tmp_path,
		"ed2",
		DEFECT,
		"defective",
		events,
		final=(*PINK, 0.01),
		destination="pink",
		placement_evidence="0:0:wrist_cam",
	)
	assert cause(score_episode(d)) == "none"


def test_false_negative_visible(tmp_path):
	events = wrist(0, [0], -0.9, cross=5, cross_crop=2) + wrist(1, [2], -0.7, cross=80, cross_crop=60)
	row = score_episode(make(tmp_path, "fn", DEFECT, "normal", events))
	err = row["first_error"]
	assert err["root_cause"] == "false_negative_visible" and err["view_id"] == "0:1:wrist_cam"
	assert row["defect_miss"] and not row["physical_success"]


def test_target_face_not_in_view(tmp_path):
	events = wrist(0, [0, 4], -0.9) + wrist(1, [1, 4], -0.9)
	row = score_episode(make(tmp_path, "tf", DEFECT, "normal", events))
	assert cause(row) == "target_face_not_in_view"


def test_occlusion(tmp_path):
	px = [0, 0, 200, 0, BIG, 0]  # the agent listed face 2, only 200 px truly visible
	events = wrist(0, [2, 4], -0.9, true_px=px)
	row = score_episode(make(tmp_path, "oc", DEFECT, "normal", events))
	assert cause(row) == "occlusion"


def test_crop_error_defective(tmp_path):
	px = [0, 0, BIG, 0, BIG, 0]
	crop = [0, 0, 100, 0, BIG, 0]  # the crop cut the target face off
	events = wrist(0, [2, 4], -0.9, true_px=px, crop_px=crop, cross=40, cross_crop=3)
	row = score_episode(make(tmp_path, "ce", DEFECT, "normal", events))
	assert cause(row) == "crop_error"


def test_false_positive_and_crop_error_normal(tmp_path):
	events = wrist(0, [0], 1.2, crop_px=[BIG, 0, 0, 0, 0, 0])
	row = score_episode(
		make(tmp_path, "fp", NORMAL, "defective", events, destination="pink", placement_evidence="0:0:wrist_cam")
	)
	assert cause(row) == "false_positive_visible" and row["false_alarm"]
	events = wrist(0, [0], 1.2, true_px=[BIG, 0, 0, 0, 0, 0], crop_px=[100, 0, 0, 0, 0, 0])
	row = score_episode(
		make(tmp_path, "fp2", NORMAL, "defective", events, destination="pink", placement_evidence="0:0:wrist_cam")
	)
	assert cause(row) == "crop_error"


def test_wrong_label_precedes_physical(tmp_path):
	events = wrist(0, [0], 1.2)
	row = score_episode(
		make(
			tmp_path,
			"wl",
			NORMAL,
			"defective",
			events,
			final=(*PINK, 0.01),
			verdict="FAILED",
			failures=[{"code": "PHYSICAL_HEALTH_FAILED", "reason_codes": ["place_miss"]}],
		)
	)
	assert cause(row) == "false_positive_visible"
	assert row["physical_outcome"] == "pink_bowl" and row["wrong_bowl"]


def test_unknown_branches(tmp_path):
	fail = {"code": "PLAN_FAILED", "frame": 900}
	row = score_episode(
		make(tmp_path, "u1", NORMAL, "unknown", wrist(0, [0], -0.9), verdict="UNKNOWN", rewinds=1, failures=[fail])
	)
	assert cause(row) == "recovery_failure" and row["is_unknown"]
	row = score_episode(make(tmp_path, "u2", NORMAL, "unknown", wrist(0, [0], -0.9), verdict="UNKNOWN"))
	assert cause(row) == "premature_stop"
	events = []
	for i in range(4):
		events += wrist(i, [0, 1], -0.9)
	row = score_episode(make(tmp_path, "u3", NORMAL, "unknown", events, verdict="UNKNOWN", cleared_faces=[0, 1]))
	assert cause(row) == "target_face_not_in_view"


def test_physical_failures(tmp_path):
	table = (0.4, 0.4, 0.01)
	row = score_episode(
		make(
			tmp_path,
			"g",
			NORMAL,
			"normal",
			normal_views(),
			final=table,
			verdict="FAILED",
			health={"failure_mode": "grasp_slip", "reason_codes": ["grasp_slip"]},
		)
	)
	assert cause(row) == "grasp_or_path_failure"
	row = score_episode(
		make(
			tmp_path,
			"p",
			NORMAL,
			"normal",
			normal_views(),
			final=table,
			verdict="FAILED",
			health={"failure_mode": "place_miss", "reason_codes": ["place_miss"]},
		)
	)
	assert cause(row) == "placement_failure"
	row = score_episode(make(tmp_path, "p2", NORMAL, "normal", normal_views(), final=(*PINK, 0.01), verdict="FAILED"))
	assert cause(row) == "placement_failure" and row["wrong_bowl"]
	row = score_episode(
		make(
			tmp_path,
			"path",
			NORMAL,
			"normal",
			normal_views(),
			final=BRICK + [0.01],
			verdict="FAILED",
			failures=[{"code": "PLACE_RETURN_PATH_BLOCKED"}],
		)
	)
	assert cause(row) == "grasp_or_path_failure"


def test_final_cycle_verdict_is_scored(tmp_path):
	row = score_episode(
		make(
			tmp_path,
			"fc",
			NORMAL,
			"normal",
			normal_views(),
			verdict="FAILED",
			verdict_final_cycle="SUCCESS",
			replay=False,
		)
	)
	assert row["scored_verdict"] == "SUCCESS" and row["physical_success"]


def test_infra(tmp_path):
	d = tmp_path / "missing"
	d.mkdir()
	(d / "run_status.json").write_text(json.dumps({"state": "interrupted", "error": "killed"}))
	row = score_episode(d)
	assert cause(row) == "infra_failure" and not row["has_episode"]
	events = [{"seq": 0, "kind": "detector_error", "branch": 0, "frame": 12}]
	row = score_episode(make(tmp_path, "de", NORMAL, "unknown", events, verdict="FAILED"))
	assert cause(row) == "infra_failure" and row["first_error"]["frame"] == 12


def test_counts(tmp_path):
	events = wrist(0, [0], -0.9) + [
		{"seq": 50, "kind": "review_view", "id": "0:0:wrist_cam"},
		{"seq": 51, "kind": "llm_image", "cached": True},
		{"seq": 52, "kind": "llm_image"},
		{"seq": 53, "kind": "decision", "latency_s": 1.5},
		{"seq": 54, "kind": "decision", "latency_s": 0.5},
	]
	row = score_episode(make(tmp_path, "cn", NORMAL, "normal", events, rewinds=1, retries=1))
	assert row["n_review_views"] == 1
	assert row["n_llm_images"] == 2 and row["n_llm_images_cached"] == 1
	assert row["n_recoveries"] == 2
	assert row["decision_latency_s"] == pytest.approx(2.0)


REAL = Path(__file__).resolve().parents[1] / "results/agent-bench/final/after/run-bench16-rule"


@pytest.mark.parametrize("name", ["case-0014", "case-0015"])
def test_real_episodes_do_not_crash(name):
	path = REAL / name
	if not path.is_dir():
		pytest.skip("real episode missing")
	row = score_episode(path)
	assert row["root_cause"] in ROOT_CAUSES
	assert row["case_id"].startswith("L600008")
