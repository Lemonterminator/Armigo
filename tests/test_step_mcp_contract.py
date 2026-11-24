import asyncio
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("cv2")
pytest.importorskip("mcp")

from digital_twin_armigo.agent import step_mcp as sm
from digital_twin_armigo.agent.twin_mujoco import ROOT

DETECTOR = ROOT / "results/agent-bench/sup-v7bot-local.json"
LEGACY_NPZ = Path(__file__).resolve().parent / "data" / "legacy_actions_600001.npz"
CLEAN_CASE = (600001, False, 5)
OCCLUDED_CASE = (600004, False, 5)

LEGACY_600001 = {
	"camera_calibration": {
		"camera": "top",
		"intrinsics": [[612.47, 0.0, 319.5], [0.0, 612.47, 239.5], [0.0, 0.0, 1.0]],
		"world_from_camera": [
			[0.00082, 0.90791, -0.41917, 0.48795],
			[1.0, 0.00015, 0.00228, -0.00057],
			[0.00213, -0.41917, -0.9079, 0.51611],
			[0.0, 0.0, 0.0, 1.0],
		],
		"convention": "OpenCV (x right, y down, z forward); robot base frame",
		"reprojection_rms_px": 1.2163,
		"poses_used": 40,
	},
	"locate_bowls": {
		"bowls_m": {"green": [0.1053, 0.19658], "pink": [0.22885, 0.03994]},
		"found_at": "session start",
	},
	"locate_brick": {
		"found": True,
		"position_m": [0.18471, -0.1171, 0.01155],
		"yaw_deg": -13.758,
		"size_class": "2x2",
		"size_m": [0.0318, 0.0318, 0.0192],
		"footprint_mm": [32.9, 32.38],
		"occlusion_correction_mm": [0.0, 0.0],
	},
	"scan_wrist": {
		"new_views": [
			{
				"id": "0:329:wrist_cam",
				"camera": "wrist_cam",
				"faces": [0, 3, 4],
				"face_pixels": [2102, 5, 19, 2336, 11785, 0],
				"score": -1.0,
				"threshold": 0.0,
			},
			{
				"id": "0:535:wrist_cam",
				"camera": "wrist_cam",
				"faces": [1, 2, 4],
				"face_pixels": [23, 2118, 2319, 2, 11793, 0],
				"score": -1.0,
				"threshold": 0.0,
			},
		],
		"error": None,
	},
	"present_bottom": {
		"new_views": [
			{
				"id": "0:1044:top",
				"camera": "top",
				"faces": [5],
				"face_pixels": [466, 0, 58, 51, 0, 5082],
				"score": -1.0,
				"threshold": 0.0,
			},
		],
		"error": None,
	},
	"place": {"error": None},
}


class FakeScorer:
	def threshold(self, camera):
		return 0.0

	def score(self, camera, image):
		return -1.0


def _service(tmp_path, cases, profile="realistic", name="svc"):
	if not DETECTOR.is_file():
		pytest.skip("detector spec not present")
	return sm.StepService(tmp_path / name, DETECTOR, profile, cases, scorer=FakeScorer())


def _payload(content):
	blocks = content[0] if isinstance(content, tuple) else content
	return json.loads(blocks[0].text)


def _assert_face_lists(payload):
	clear = payload["faces_clear"]
	unclear = payload["faces_unclear"]
	assert isinstance(clear, list) and all(isinstance(f, int) and 0 <= f <= 5 for f in clear)
	assert isinstance(unclear, list) and all(isinstance(f, int) and 0 <= f <= 5 for f in unclear)
	assert set(clear).isdisjoint(unclear)


def test_step_mcp_contract_new_fields_and_legacy_compatibility(tmp_path):
	service = _service(tmp_path, [CLEAN_CASE])
	try:
		mcp = sm.create_step_mcp(service)

		async def run():
			case_id = _payload(await mcp.call_tool("list_cases", {}))["case_ids"][0]
			sid = _payload(await mcp.call_tool("start_session", {"case_id": case_id}))["session_id"]
			out = {}
			for tool in ("camera_calibration", "locate_bowls", "locate_brick", "scan_wrist", "present_bottom"):
				out[tool] = _payload(await mcp.call_tool(tool, {"session_id": sid}))
			out["place"] = _payload(await mcp.call_tool("place", {"session_id": sid, "label": "normal"}))
			out["finish"] = _payload(await mcp.call_tool("finish", {"session_id": sid}))
			return out

		out = asyncio.run(run())
	finally:
		service.close()

	for key in ("camera_calibration", "locate_bowls", "locate_brick", "scan_wrist", "present_bottom", "place"):
		sm.assert_sensor_only(out[key])
		for legacy_k, legacy_v in LEGACY_600001[key].items():
			assert out[key][legacy_k] == legacy_v, f"{key}.{legacy_k}"

	cal = out["camera_calibration"]
	assert isinstance(cal["reprojection_px"], float) and cal["reprojection_px"] >= 0.0
	assert cal["reprojection_px"] == pytest.approx(cal["reprojection_rms_px"])

	bowls = out["locate_bowls"]
	assert set(bowls["rim_inliers"]) == set(bowls["bowls_m"])
	assert set(bowls["fit_residual_m"]) == set(bowls["bowls_m"])
	for slot in bowls["bowls_m"]:
		assert isinstance(bowls["rim_inliers"][slot], int) and bowls["rim_inliers"][slot] > 0
		assert isinstance(bowls["fit_residual_m"][slot], float) and bowls["fit_residual_m"][slot] >= 0.0

	brick = out["locate_brick"]
	assert isinstance(brick["yaw_ambiguity_deg"], float) and brick["yaw_ambiguity_deg"] >= 0.0
	assert isinstance(brick["footprint_occluded_frac"], float)
	assert 0.0 <= brick["footprint_occluded_frac"] <= 1.0

	wrist = out["scan_wrist"]
	_assert_face_lists(wrist)
	assert wrist["faces_clear"] == [0, 1, 2, 3, 4]
	assert wrist["faces_unclear"] == []

	bottom = out["present_bottom"]
	_assert_face_lists(bottom)
	assert bottom["faces_clear"] == [5]
	assert bottom["faces_unclear"] == []

	place = out["place"]
	assert isinstance(place["carry_target_dist_m"], float) and 0.0 <= place["carry_target_dist_m"] < 0.05
	assert out["finish"]["result"]["verdict"] == "SUCCESS"


def test_locate_brick_occluded_case_has_higher_ambiguity_than_clean(tmp_path):
	service = _service(tmp_path, [CLEAN_CASE, OCCLUDED_CASE], name="occ")
	try:
		mcp = sm.create_step_mcp(service)

		async def run():
			cids = _payload(await mcp.call_tool("list_cases", {}))["case_ids"]
			out = []
			for cid in cids:
				sid = _payload(await mcp.call_tool("start_session", {"case_id": cid}))["session_id"]
				brick = _payload(await mcp.call_tool("locate_brick", {"session_id": sid}))
				out.append(brick)
			return out

		clean_brick, occ_brick = asyncio.run(run())
	finally:
		service.close()

	sm.assert_sensor_only(clean_brick)
	sm.assert_sensor_only(occ_brick)
	assert clean_brick["found"] and occ_brick["found"]
	assert 0.0 <= clean_brick["footprint_occluded_frac"] <= 1.0
	assert 0.0 <= occ_brick["footprint_occluded_frac"] <= 1.0
	assert occ_brick["footprint_occluded_frac"] > clean_brick["footprint_occluded_frac"] + 0.1
	assert occ_brick["yaw_ambiguity_deg"] > clean_brick["yaw_ambiguity_deg"] + 10.0


def test_place_unknown_and_failed_inspect_uncertainty_cues(tmp_path):
	service = _service(tmp_path, [CLEAN_CASE], name="unknown")
	try:
		mcp = sm.create_step_mcp(service)

		async def run():
			cid = _payload(await mcp.call_tool("list_cases", {}))["case_ids"][0]
			sid = _payload(await mcp.call_tool("start_session", {"case_id": cid}))["session_id"]
			pl = _payload(await mcp.call_tool("place", {"session_id": sid, "label": "unknown"}))
			return sid, pl

		sid, pl = asyncio.run(run())
		session = service.session(sid)
		# Simulate an inspection call whose backend returns an error with no clear views.
		session.placed = False
		session.backend.inspect = lambda node, prev: ([], "NO_CLAMP")
		failed_bottom = session._out(session._inspect("present_bottom"))
	finally:
		service.close()

	sm.assert_sensor_only(pl)
	assert pl["error"] is None
	assert pl["carry_target_dist_m"] is None

	sm.assert_sensor_only(failed_bottom)
	_assert_face_lists(failed_bottom)
	assert failed_bottom["faces_clear"] == []
	assert failed_bottom["faces_unclear"] == [5]


def test_legacy_600001_actions_npz_bitwise_identical_in_stage_f5(tmp_path):
	pytest.importorskip("langgraph")
	pytest.importorskip("av")
	from digital_twin_armigo.agent.student_rollout import run_episode
	from digital_twin_armigo.inspection_scorer import FixedScorer

	if not LEGACY_NPZ.is_file():
		pytest.skip(f"missing reference fixture {LEGACY_NPZ}")
	scorer = FixedScorer(lambda _c, _r: -1.0, {"top": 0.0, "wrist_cam": 0.0})
	out_dir = tmp_path / "legacy_600001"
	run = run_episode(CLEAN_CASE, scorer, out_dir, sensing="truth", video=False, action_noise=None)
	assert run["result"]["verdict"] == "SUCCESS"
	with (
		np.load(out_dir / "branch-0" / "actions.npz", allow_pickle=False) as cur,
		np.load(LEGACY_NPZ, allow_pickle=False) as ref,
	):
		assert cur.files == ref.files
		for key in ref.files:
			assert cur[key].dtype == ref[key].dtype, key
			assert cur[key].shape == ref[key].shape, key
			assert np.array_equal(cur[key], ref[key]), key
