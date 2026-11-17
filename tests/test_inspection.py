"""Tests for visibility-constrained sim-oracle inspection and anti-leakage planner sanitization."""

from __future__ import annotations

import numpy as np
import pytest

from digital_twin_armigo.inspection import (
	DefectLeakageError,
	assert_no_defect_leakage,
	inspect_object_sim_oracle,
	sanitize_scene_for_planner,
)


def test_counterexample_planner_context_strips_defect_ground_truth_and_asset_names():
	raw_scene = {
		"episode_id": "ep_0001",
		"brick_xy": [0.26, 0.01],
		"brick_yaw_deg": 20.0,
		"brick_size": [0.0318, 0.0318, 0.0192],
		"asset_usd": "assets/props/duplo_2x2_defective.usda",
		"defective": True,
		"defect_face": 5,
		"defect_px": 140,
		"target_bowl": "pink",
		"task": "Pick up the brick and place it in the pink bowl.",
		"bowls": {"green": [0.22, 0.19], "pink": [0.26, -0.18]},
	}
	# Raw context MUST be rejected by the leakage guard
	with pytest.raises(DefectLeakageError):
		assert_no_defect_leakage(raw_scene)

	clean = sanitize_scene_for_planner(raw_scene)
	assert_no_defect_leakage(clean)
	assert "defective" not in clean
	assert "defect_face" not in clean
	assert "defect_px" not in clean
	assert "target_bowl" not in clean
	assert "defective" not in clean["asset_id"].lower()


def test_counterexample_sim_oracle_inspector_respects_visibility_occlusion_and_face():
	cam_pos = np.array([0.0, 0.0, 0.55])
	cam_fwd = np.array([0.0, 0.0, -1.0])
	brick_pos = np.array([0.26, 0.01, 0.01])
	# Identity rotation: face 4 (+Z) faces camera, face 5 (-Z) faces table (hidden!)
	brick_rot = np.eye(3)

	# 1. Occluded brick -> MUST return UNKNOWN even though ground_truth defective=True
	res_occ = inspect_object_sim_oracle(
		defective_gt=True,
		defect_face_gt=4,
		brick_pos=brick_pos,
		brick_rot=brick_rot,
		cam_pos=cam_pos,
		cam_forward=cam_fwd,
		in_crop=True,
		visible_fraction=0.20,  # < 0.50 unoccluded requirement
	)
	assert res_occ.status == "UNKNOWN"
	assert res_occ.label is None
	assert res_occ.source == "sim_oracle"

	# 2. Visible top face, but defect is on hidden bottom face (5, -Z) -> CANNOT peek through brick!
	res_wrong_face = inspect_object_sim_oracle(
		defective_gt=True,
		defect_face_gt=5,
		brick_pos=brick_pos,
		brick_rot=brick_rot,
		cam_pos=cam_pos,
		cam_forward=cam_fwd,
		in_crop=True,
		visible_fraction=0.95,
	)
	assert res_wrong_face.status == "UNKNOWN"
	assert res_wrong_face.label is None

	# 3. Normal brick viewed from only 1 side -> still UNKNOWN until required faces observed,
	# OR when defect face (4, +Z) IS clearly facing camera and unoccluded -> returns KNOWN defective
	res_seen = inspect_object_sim_oracle(
		defective_gt=True,
		defect_face_gt=4,
		brick_pos=brick_pos,
		brick_rot=brick_rot,
		cam_pos=cam_pos,
		cam_forward=cam_fwd,
		in_crop=True,
		visible_fraction=0.95,
	)
	assert res_seen.status == "KNOWN"
	assert res_seen.label == "defective"
	assert res_seen.source == "sim_oracle"
