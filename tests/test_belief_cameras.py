import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("cv2")

from digital_twin_armigo.agent.twin_mujoco import TwinMujoco, sample_case
from digital_twin_armigo.agent.twin_record import EpisodeTrace
from digital_twin_armigo.experiments.diag_mcp import compare_runs
from digital_twin_armigo.inspection_scorer import FixedScorer
from digital_twin_armigo.mujoco_cell.handeye import camera_truth
from digital_twin_armigo.mujoco_cell.inspection_views import detector_crop


def _twin(tmp_path, name="run", *, seed=3_000_101, cameras="drift", **opts):
	layout = sample_case(seed, positive=False, face=0, options={"cameras": cameras})
	scorer = FixedScorer(lambda _c, _rgb: -1.0, {"top": 0.0, "wrist_cam": 0.0})
	return TwinMujoco(layout, scorer, EpisodeTrace(tmp_path / name, video=False), sensing="realistic", **opts)


def test_camera_override_moves_the_model_camera_and_restores_it():
	from digital_twin_armigo.constants import SO101_MJCF_DIR
	from digital_twin_armigo.mujoco_cell._mj import mujoco
	from digital_twin_armigo.mujoco_cell.cell import Cell

	layout = sample_case(3_000_101, positive=False, face=0, options={"cameras": "drift"})
	cell = Cell(layout, SO101_MJCF_DIR, dynamic_clutter=True, assisted_grasp=False)
	try:
		mujoco.mj_forward(cell.model, cell.data)
		before = camera_truth(cell.model, cell.data, "top")
		moved = before.copy()
		moved[:3, 3] += [0.01, -0.02, 0.0]
		with cell.camera_override("top", moved):
			assert np.allclose(camera_truth(cell.model, cell.data, "top"), moved, atol=1e-9)
		assert np.allclose(camera_truth(cell.model, cell.data, "top"), before, atol=1e-12)
	finally:
		cell.close()


def test_estimated_belief_camera_differs_from_actual_under_drift(tmp_path):
	"""G9: with belief_cameras='estimated', _belief places the top camera through the stale calibration."""
	twin = _twin(tmp_path, "est", belief_cameras="estimated")
	try:
		twin.prepare()
		with twin._belief():
			est = camera_truth(twin.cell.model, twin.cell.data, "top")
		assert np.allclose(est, twin._cam_est, atol=1e-9)
		assert not np.allclose(est, camera_truth(twin.cell.model, twin.cell.data, "top"), atol=1e-4)
	finally:
		twin.close()


def test_center_px_crop_is_centred_on_the_given_point():
	from digital_twin_armigo.constants import SO101_MJCF_DIR
	from digital_twin_armigo.mujoco_cell._mj import mujoco
	from digital_twin_armigo.mujoco_cell.cell import Cell

	layout = sample_case(3_000_101, positive=False, face=0, options={"cameras": "fixed"})
	cell = Cell(layout, SO101_MJCF_DIR, dynamic_clutter=True, assisted_grasp=False)
	try:
		mujoco.mj_forward(cell.model, cell.data)
		_img, bbox = detector_crop(cell, "top", None, cell.data, 96, clamp=True)
		_img2, bbox2 = detector_crop(
			cell,
			"top",
			None,
			cell.data,
			96,
			clamp=True,
			center_px=(bbox[0] + bbox[2] / 2 + 6, bbox[1] + bbox[3] / 2),
		)
		assert bbox2[2:] == bbox[2:] and bbox2[0] == pytest.approx(bbox[0] + 6, abs=1)
	finally:
		cell.close()


def test_default_belief_cameras_and_hold_evidence_match_via_compare_runs_and_img_emits_crop_fields(tmp_path):
	def _run_top(name: str, **opts):
		twin = _twin(tmp_path, name, **opts)
		try:
			twin.prepare()
			twin._perceive()
			view = twin._capture("top")
			res = twin.evaluate("normal", [view], [])
			twin.record.finish(res, {})
			return view, tmp_path / name
		finally:
			twin.record.close()
			twin.close()

	v_def, dir_def = _run_top("def")
	_v_exp, dir_exp = _run_top("exp", belief_cameras="actual", evidence="hold")
	assert "crop_source" not in v_def and "crop_shift_px" not in v_def
	cmp_out = compare_runs(dir_def, dir_exp)
	assert cmp_out["identical_discrete"] and cmp_out["score_deltas"] == [] and cmp_out["verdict_equal"]

	v_img, _ = _run_top("img", evidence="img")
	assert v_img["crop_source"] in ("image", "projection")
	assert "crop_shift_px" in v_img


def test_dense_evidence_captures_in_motion_with_delayed_decision(tmp_path):
	from digital_twin_armigo.constants import FPS

	twin = _twin(tmp_path, "dense", evidence="dense-20")
	try:
		twin.prepare()
		views: list[dict] = []
		assert twin._wrist([], views) is None
		v_events = [e for e in twin.record.events if e["kind"] == "view"]
		in_mot = [e["observation"] for e in v_events if e["observation"].get("in_motion")]
		assert len(in_mot) >= 1
		want_delay = round(0.15 * FPS)
		assert all(obs["decided_after_frames"] == want_delay for obs in in_mot)
	finally:
		twin.close()


def test_hold_evidence_wrist_omits_motion_fields_and_matches_via_compare_runs(tmp_path, monkeypatch):
	from digital_twin_armigo.agent import twin_mujoco as tm

	monkeypatch.setattr(tm.time, "perf_counter", lambda: 0.0)

	def _run_wrist(name: str, **opts):
		twin = _twin(tmp_path, name, **opts)
		try:
			twin.prepare()
			views: list[dict] = []
			assert twin._wrist([], views) is None
			res = twin.evaluate("normal", views, [])
			twin.record.finish(res, {})
			return views, tmp_path / name
		finally:
			twin.record.close()
			twin.close()

	views_a, dir_a = _run_wrist("hold_default")
	views_b, dir_b = _run_wrist("hold_explicit", evidence="hold")
	assert views_a and all("in_motion" not in v and "decided_after_frames" not in v for v in views_a)
	assert len(views_a) == len(views_b)
	cmp_out = compare_runs(dir_a, dir_b)
	assert cmp_out["identical_discrete"] and cmp_out["score_deltas"] == [] and cmp_out["verdict_equal"]
