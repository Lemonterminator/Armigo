"""WP4 tests: realistic MCP zero truth leak, TOP_CROP visibility, OOD camera axes, light_k, jittered hue, preflight."""

from __future__ import annotations

import asyncio
import colorsys
import dataclasses
import hashlib
import importlib.util
import inspect
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

from digital_twin_armigo.agent import step_mcp as sm
from digital_twin_armigo.agent.twin_mujoco import FIXED_TOP, ROOT, TwinMujoco, sample_case
from digital_twin_armigo.agent.twin_record import EpisodeTrace
from digital_twin_armigo.camera import project
from digital_twin_armigo.constants import SO101_MJCF_DIR, TOP_CROP, TOP_FULL
from digital_twin_armigo.governance import bronze, lake, silver
from digital_twin_armigo.inspection_scorer import FixedScorer
from digital_twin_armigo.mujoco_cell.cell import Cell
from digital_twin_armigo.mujoco_cell.palette import PALETTE
from digital_twin_armigo.mujoco_cell.scene import scene_xml
from digital_twin_armigo.mujoco_cell.scene_v3 import (
	_CAM_HALF_WIDTHS,
	_OOD_CAM_AXES,
	draw_camera_v3,
	sample_scene_v3,
)
from digital_twin_armigo.mujoco_cell.visibility import WORKSPACE_CORNERS, workspace_visible

REPO_ROOT = Path(__file__).resolve().parents[1]
DETECTOR = ROOT / "results/agent-bench/sup-v7bot-local.json"


def _load_suite_v3():
	name = "make_scene_suite_v3_wp4"
	path = REPO_ROOT / "scripts/vla/make_scene_suite_v3.py"
	spec = importlib.util.spec_from_file_location(name, path)
	mod = importlib.util.module_from_spec(spec)
	sys.modules[name] = mod
	assert spec and spec.loader
	spec.loader.exec_module(mod)
	return mod


def _circular_hue_deg(a_deg: float, b_deg: float) -> float:
	d = abs(a_deg - b_deg) % 360.0
	return min(d, 360.0 - d)


class _FakeScorer:
	def threshold(self, camera):
		return 0.0

	def score(self, camera, image):
		return -1.0


def _payload(content):
	blocks = content[0] if isinstance(content, tuple) else content
	return json.loads(blocks[0].text)


# ---------------------------------------------------------------------------
# 1. Realistic MCP zero truth leak & BRICK_NOT_FOUND terminal recovery
# ---------------------------------------------------------------------------


def test_assert_sensor_only_catches_unfound_bowl_coordinate_leak(tmp_path):
	leaked = {
		"bowls_m": {"green": [0.20621, -0.12927], "pink": [0.14386, 0.18590]},
		"rim_inliers": {"green": 0, "pink": 48},
		"fit_residual_m": {"green": None, "pink": 0.0012},
	}
	with pytest.raises(sm.SimulatorLeak, match=r"unfound bowl.*green.*rim_inliers=0"):
		sm.assert_sensor_only(leaked)

	with pytest.raises(sm.SimulatorLeak, match=r"unfound bowl.*green.*rim_inliers=0"):
		sm.assert_sensor_only({"returned": leaked})

	clean = {
		"bowls_m": {"green": None, "pink": [0.14386, 0.18590]},
		"rim_inliers": {"green": 0, "pink": 48},
		"fit_residual_m": {"green": None, "pink": 0.0012},
		"missing_bowls": ["green"],
		"found": False,
	}
	sm.assert_sensor_only(clean)

	# Governance silver._check_sensor_guard also catches value-level leaks in tool_call logs.
	ep_dir = tmp_path / "data" / "runs" / "b1" / "mujoco" / "realistic-fixed-B-v7bot" / "L3000015-N"
	ep_dir.mkdir(parents=True)
	(ep_dir / "run_status.json").write_text(json.dumps({"state": "done"}), encoding="utf-8")
	(ep_dir / "task.json").write_text(
		json.dumps({"case": [3000015, False, 5, {"scene": "v3", "split": "train"}]}),
		encoding="utf-8",
	)
	(ep_dir / "episode.json").write_text(
		json.dumps(
			{
				"result": {"label": "unknown", "verdict": "FAILED"},
				"provenance": {"experiment": {"batch": "b1", "group": "realistic-fixed-B-v7bot", "engine": "mujoco"}},
			}
		),
		encoding="utf-8",
	)
	(ep_dir / "events.jsonl").write_text(
		json.dumps({"kind": "tool_call", "tool": "locate_bowls", "returned": leaked}) + "\n",
		encoding="utf-8",
	)
	con = lake.connect(tmp_path / "lake.sqlite")
	try:
		bronze.register(con, tmp_path / "data", source="v3")
		bid = silver.build_silver(con)
		row = json.loads(con.execute("select row_json from silver_episode where build_id = ?", (bid,)).fetchone()[0])
		assert row["checks"]["sensor_guard"]["passed"] is False
		assert "SimulatorLeak" in (row["checks"]["sensor_guard"]["detail"] or "")
		assert row["quarantined"] is True
	finally:
		con.close()


def test_realistic_locate_bowls_no_truth_leak_on_seed_3000015(tmp_path):
	pytest.importorskip("mujoco")
	pytest.importorskip("cv2")
	pytest.importorskip("mcp")

	case = [3000015, False, 5, {"scene": "v3", "split": "train"}]
	lay = sample_case(*case)
	truth_green = [round(float(lay.green[0]), 5), round(float(lay.green[1]), 5)]

	det_spec = tmp_path / "det.json"
	det_spec.write_text(json.dumps({"crop_size": 256}), encoding="utf-8")

	service = sm.StepService(tmp_path / "svc", det_spec, "realistic", [case], scorer=_FakeScorer())
	try:
		mcp = sm.create_step_mcp(service)

		async def run():
			cid = _payload(await mcp.call_tool("list_cases", {}))["case_ids"][0]
			start = _payload(await mcp.call_tool("start_session", {"case_id": cid}))
			sid = start["session_id"]
			bowls = _payload(await mcp.call_tool("locate_bowls", {"session_id": sid}))
			return sid, start, bowls

		sid, start, bowls = asyncio.run(run())
		session = service.session(sid)
		assert session.backend.scene_error == "BOWL_NOT_FOUND"
		assert session.backend._bowl_estimates["green"] is None
		assert "green" not in session.backend.bowls

		sm.assert_sensor_only(start)
		sm.assert_sensor_only(bowls)
		assert bowls["bowls_m"]["green"] is None
		assert bowls["rim_inliers"]["green"] == 0
		assert bowls["fit_residual_m"]["green"] is None
		assert bowls["found"] is False
		assert "green" in bowls["missing_bowls"]
		assert bowls["bowls_m"]["pink"] is not None
		assert bowls["rim_inliers"]["pink"] > 0

		# Truth green coordinates must never appear in the tool payload or tool_call event logs.
		payload_str = json.dumps(bowls)
		assert str(truth_green[0]) not in payload_str
		assert str(truth_green[1]) not in payload_str

		service.run_sync(service.finish, sid)
	finally:
		service.close()

	events_files = list((tmp_path / "svc").rglob("events.jsonl"))
	assert events_files
	events = [json.loads(line) for line in events_files[0].read_text(encoding="utf-8").splitlines() if line]
	tool_calls = [e for e in events if e.get("kind") == "tool_call"]
	assert any(e.get("tool") == "locate_bowls" for e in tool_calls)
	for tc in tool_calls:
		if tc.get("returned") is not None:
			sm.assert_sensor_only(tc["returned"])
			tc_str = json.dumps(tc["returned"])
			assert str(truth_green[0]) not in tc_str
			assert str(truth_green[1]) not in tc_str

	# Truth mode and oracle=("bowls",) still populate self.bowls unchanged.
	for mode, oracle in (("truth", ()), ("realistic", ("bowls",))):
		tr = EpisodeTrace(tmp_path / f"check_{mode}_{'_'.join(oracle)}", video=False)
		tw = TwinMujoco(
			lay,
			FixedScorer(lambda _c, _r: -1.0, {"top": 0.0, "wrist_cam": 0.0}),
			tr,
			sensing=mode,
			oracle=oracle,
		)
		try:
			assert "green" in tw.bowls and "pink" in tw.bowls
			assert np.allclose(tw.bowls["green"], lay.green)
		finally:
			tw.close()
			tr.close()


def test_brick_not_found_is_terminal_in_realistic_recovery_choices(tmp_path):
	pytest.importorskip("mujoco")
	pytest.importorskip("cv2")

	lay = sample_case(600001, False, 5)
	tr = EpisodeTrace(tmp_path / "rec", video=False)
	tw = TwinMujoco(
		lay,
		FixedScorer(lambda _c, _r: -1.0, {"top": 0.0, "wrist_cam": 0.0}),
		tr,
		sensing="realistic",
	)
	try:
		for code in ("BOWL_NOT_FOUND", "BRICK_NOT_FOUND", "DETECTOR_UNAVAILABLE"):
			assert tw.recovery_choices(sm.STEP_POLICY, [{"code": code, "skill": "scan_wrist"}]) == []
		# Non-terminal failures still offer recovery choices.
		assert len(tw.recovery_choices(sm.STEP_POLICY, [{"code": "NO_CLAMP", "skill": "present_bottom"}])) > 0
	finally:
		tw.close()
		tr.close()


# ---------------------------------------------------------------------------
# 2. Student top-crop visibility
# ---------------------------------------------------------------------------


def test_visibility_enforces_student_top_crop_bounds():
	x0, y0, cw, ch = TOP_CROP
	u_lo, u_hi = x0 + 0.05 * cw, x0 + 0.95 * cw
	v_lo, v_hi = y0 + 0.05 * ch, y0 + 0.95 * ch

	# Nominal WORKSPACE_CORNERS are inside TOP_CROP with 5% margin under FIXED_TOP.
	uv_ws, front_ws = project(FIXED_TOP, np.asarray(WORKSPACE_CORNERS, dtype=np.float64), TOP_FULL)
	assert bool(front_ws.all())
	assert bool(((uv_ws[:, 0] >= u_lo) & (uv_ws[:, 0] <= u_hi)).all())
	assert bool(((uv_ws[:, 1] >= v_lo) & (uv_ws[:, 1] <= v_hi)).all())

	# Legacy 1100001 has green bowl at u ~ 88.2 < 96 (outside TOP_CROP, though inside TOP_FULL).
	lay_clip = sample_case(1100001, False, 5)
	ok_clip, bad_clip = workspace_visible(lay_clip)
	assert not ok_clip
	assert "green_bowl" in bad_clip

	# Seed 3000001 previously had pink bowl at u=67.80 < 96; after fix, accepted v3 layouts stay inside TOP_CROP.
	lay0 = sample_case(3000001, False, 5, {"scene": "v3", "split": "train"})
	ok0, bad0 = workspace_visible(lay0)
	assert ok0 and bad0 == []

	# Check 120 accepted v3 train layouts: ZERO bowls or bricks outside TOP_CROP 5% margin.
	accepted = 0
	for seed in range(3000001, 3000140):
		try:
			lay = sample_case(seed, True, 2, {"scene": "v3", "split": "train"})
		except RuntimeError:
			continue
		accepted += 1
		pts = [
			(*lay.green, 0.0),
			(*lay.pink, 0.0),
			*[(ex, ey, 0.0) for _, ex, ey, _ in lay.extra_bowls],
			(*lay.brick_xy, lay.brick_size[2] / 2.0),
		]
		uv, front = project(lay.top, np.asarray(pts, dtype=np.float64), TOP_FULL)
		assert bool(front.all()), f"seed {seed} point behind camera"
		assert bool(((uv[:, 0] >= u_lo) & (uv[:, 0] <= u_hi)).all()), f"seed {seed} u outside TOP_CROP: {uv[:, 0]}"
		assert bool(((uv[:, 1] >= v_lo) & (uv[:, 1] <= v_hi)).all()), f"seed {seed} v outside TOP_CROP: {uv[:, 1]}"
	assert accepted >= 120


# ---------------------------------------------------------------------------
# 3. OOD camera axes (all 7)
# ---------------------------------------------------------------------------


def test_ood_cam_samples_all_seven_camera_axes():
	assert set(_OOD_CAM_AXES) == set(_CAM_HALF_WIDTHS.keys())
	assert len(_OOD_CAM_AXES) == 7

	outer_hits = {k: 0 for k in _CAM_HALF_WIDTHS}
	for seed in range(3300001, 3300301):
		cam = draw_camera_v3(np.random.default_rng([seed, 0x00D]), "ood-cam")
		ratios = {k: abs(cam[k]) / hw for k, hw in _CAM_HALF_WIDTHS.items()}
		assert 1.0 - 1e-9 <= max(ratios.values()) <= 1.5 + 1e-9
		for k, r in ratios.items():
			if r >= 1.0 - 1e-9:
				outer_hits[k] += 1
	for k, hits in outer_hits.items():
		assert hits >= 15, f"axis {k!r} under-represented in ood-cam outer ring: {outer_hits}"


# ---------------------------------------------------------------------------
# 4. light_k wired into MJCF lighting (and legacy SHA-256 unchanged)
# ---------------------------------------------------------------------------


def test_light_k_wired_into_v3_mjcf_and_legacy_unchanged():
	pytest.importorskip("mujoco")

	# Legacy 14/14 MJCF SHA-256 hashes must remain byte-for-byte identical.
	data = json.loads((REPO_ROOT / "tests/data/legacy_7cc24e2.json").read_text(encoding="utf-8"))
	assert len(data["cases"]) == 14
	for entry in data["cases"]:
		lay = sample_case(*entry["case"])
		xml = scene_xml(lay).replace(REPO_ROOT.as_posix(), "E:/digital_twin_armigo-v3")
		assert hashlib.sha256(xml.encode("utf-8")).hexdigest() == entry["xml_sha256"], entry["tag"]

	# v3 layout with 3500 K vs 6500 K produces distinct MJCF lighting and warm vs cool renders.
	lay_base = sample_case(3000001, False, 5, {"scene": "v3", "split": "train"})
	lay_warm = dataclasses.replace(lay_base)
	lay_warm._scene_v3 = dataclasses.replace(lay_base._scene_v3, light_k=3500.0)
	lay_cool = dataclasses.replace(lay_base)
	lay_cool._scene_v3 = dataclasses.replace(lay_base._scene_v3, light_k=6500.0)

	xml_warm = scene_xml(lay_warm)
	xml_cool = scene_xml(lay_cool)
	assert xml_warm != xml_cool

	m_warm = re.search(r'<light [^>]*diffuse="([0-9.]+) ([0-9.]+) ([0-9.]+)"', xml_warm)
	m_cool = re.search(r'<light [^>]*diffuse="([0-9.]+) ([0-9.]+) ([0-9.]+)"', xml_cool)
	assert m_warm and m_cool
	rw, gw, bw = (float(x) for x in m_warm.groups())
	rc, gc, bc = (float(x) for x in m_cool.groups())
	assert rw > bw, f"expected 3500K warm light R > B, got {(rw, gw, bw)}"
	assert (rw / bw) > (rc / bc) * 1.05, f"expected 3500K warmer than 6500K: {(rw, gw, bw)} vs {(rc, gc, bc)}"

	from digital_twin_armigo.agent.twin_mujoco import INITIAL_Q

	c_warm = Cell(lay_warm, SO101_MJCF_DIR)
	try:
		c_warm.set_joints(INITIAL_Q)
		rgb_warm = c_warm.rgb("top").astype(np.float64).copy()
	finally:
		c_warm.close()

	c_cool = Cell(lay_cool, SO101_MJCF_DIR)
	try:
		c_cool.set_joints(INITIAL_Q)
		rgb_cool = c_cool.rgb("top").astype(np.float64).copy()
	finally:
		c_cool.close()

	assert not np.array_equal(rgb_warm, rgb_cool)
	ratio_warm = float(rgb_warm[..., 0].mean() / max(rgb_warm[..., 2].mean(), 1e-6))
	ratio_cool = float(rgb_cool[..., 0].mean() / max(rgb_cool[..., 2].mean(), 1e-6))
	assert ratio_warm > ratio_cool + 0.05


# ---------------------------------------------------------------------------
# 5. Jittered bowl colour vs table hue separability
# ---------------------------------------------------------------------------


def test_jittered_bowl_table_hue_separability_eliminates_violations():
	msv3 = _load_suite_v3()

	# Seeds 3000016, 3000289, 3000382, 3000421 (and all 3000001..3000500) must have jittered hue gap >= 35 deg.
	for seed in range(3000001, 3000501):
		sc = sample_scene_v3(seed, "train")
		ht, st, _ = colorsys.rgb_to_hsv(*sc.table_rgb)
		if st <= 0.15:
			continue
		ht_deg = ht * 360.0
		for name, rgb in sc.rgb.items():
			if PALETTE[name].hue_deg is None:
				continue
			hb_deg = colorsys.rgb_to_hsv(*rgb)[0] * 360.0
			gap = _circular_hue_deg(ht_deg, hb_deg)
			assert gap >= 35.0, f"seed {seed} bowl {name!r} jittered hue gap={gap:.2f} < 35 deg"

	# check_colour_separability must reject a layout where nominal hue gap >= 35 deg but jittered hue gap < 35 deg.
	lay = sample_case(3000002, False, 5, {"scene": "v3", "split": "train"})
	lay = dataclasses.replace(lay, extra_bowls=[])
	lay.bowl_names = {"green": "blue", "pink": "yellow"}
	# Nominal blue is ~213 deg; table at 249 deg is 36 deg away from nominal blue,
	# but jittered blue at 218 deg is only 31 deg away from table!
	table_rgb = colorsys.hsv_to_rgb(249.0 / 360.0, 0.25, 0.55)
	jittered_blue = colorsys.hsv_to_rgb(218.0 / 360.0, 0.80, 0.85)
	lay.table_rgb = tuple(float(c) for c in table_rgb)
	lay.bowl_rgb = {"green": tuple(float(c) for c in jittered_blue), "pink": PALETTE["yellow"].rgb}
	ok, reason = msv3.check_colour_separability(lay, "train")
	assert not ok, "expected check_colour_separability to reject jittered bowl hue < 35 deg from table"
	assert reason is not None


# ---------------------------------------------------------------------------
# 6. Preflight strictness: default sensing="realistic", strict perception,
#    strict balance at n=200, and 60-good-brick x 6 hue-bin FP control
# ---------------------------------------------------------------------------


def test_preflight_default_sensing_realistic_and_strict_perception_check():
	pytest.importorskip("mujoco")
	pytest.importorskip("cv2")
	msv3 = _load_suite_v3()

	assert inspect.signature(msv3.build_suite_and_preflight).parameters["sensing"].default == "realistic"
	assert inspect.signature(msv3.run_perception_check).parameters["sensing"].default == "realistic"

	# Evaluate known failure seed 3000016 directly: missed bowls + large brick error MUST fail perception_check.
	case16 = [3000016, False, 5, {"scene": "v3", "split": "train"}]
	lay16 = sample_case(*case16)
	res16 = msv3.run_perception_check([case16], [lay16], perception_sample=1.0, sensing="realistic")
	assert res16["pass"] is False
	assert 3000016 in res16["failed_seeds"]
	assert res16["cases"][0]["passed"] is False
	assert len(res16["cases"][0]["missed_bowls"]) >= 1
	# Any colour with found == 0 (median_error_mm is None) MUST be marked status="failed" and fail the check.
	failed_cols = [c for c, st in res16["per_colour_stats"].items() if st["found"] == 0]
	assert failed_cols, f"expected at least one unfound colour on 3000016, got {res16['per_colour_stats']}"
	for col in failed_cols:
		assert res16["per_colour_stats"][col]["median_error_mm"] is None
		assert res16["per_colour_stats"][col]["status"] == "failed"

	# When build_suite_and_preflight runs with perception filtering enabled on a range whose sample includes 3000016,
	# seed 3000016 must NEVER remain in a pass=True suite.
	suite, pf = msv3.build_suite_and_preflight(
		split="train",
		start_seed=3_000_015,
		end_seed=3_000_018,
		perception_sample=1.0,
		sensing="realistic",
		skip_hue_fp=True,
	)
	seeds_in_suite = [c[0] for c in suite["cases"]]
	if pf["pass"]:
		assert 3000016 not in seeds_in_suite
		assert 3000015 not in seeds_in_suite
		assert pf["perception_check"]["failed_seeds"] == []
	else:
		assert pf["perception_check"]["pass"] is False


def test_preflight_balance_strict_at_200_episodes_and_balanced_generation():
	msv3 = _load_suite_v3()

	assert msv3.BALANCE_STRICT_MIN_EPISODES <= 200

	suite1, pf1 = msv3.build_suite_and_preflight(
		split="train",
		start_seed=3_000_001,
		end_seed=3_000_200,
		perception_sample=0.0,
		max_perception=0,
		skip_hue_fp=True,
	)
	assert len(suite1["cases"]) == 200
	bal = pf1["balance"]
	assert bal["strict_thresholds_enforced"] is True
	assert bal["class_mix_ok"] is True
	assert bal["max_target_deviation_frac"] <= 0.15
	assert bal["max_abs_left_right_corr"] < 0.05
	assert bal["pass"] is True
	assert pf1["pass"] is True

	# Determinism across runs.
	suite2, pf2 = msv3.build_suite_and_preflight(
		split="train",
		start_seed=3_000_001,
		end_seed=3_000_200,
		perception_sample=0.0,
		max_perception=0,
		skip_hue_fp=True,
	)
	assert suite1["cases"] == suite2["cases"]
	assert pf1["balance"] == pf2["balance"]

	# Intentionally imbalanced 200-episode suite MUST fail balance["pass"] at n=200.
	dup_cases = [suite1["cases"][0]] * 200
	dup_lays = [sample_case(*suite1["cases"][0])] * 200
	bad_metrics = msv3._compute_balance_and_factors(dup_cases, dup_lays, "train")
	assert bad_metrics["balance"]["strict_thresholds_enforced"] is True
	assert bad_metrics["balance"]["pass"] is False


def test_hue_fp_control_requires_detector_and_gates_excess_fp():
	pytest.importorskip("mujoco")
	pytest.importorskip("cv2")
	msv3 = _load_suite_v3()

	# 1. Without a scorer (default), preflight MUST NOT silently claim pass=True.
	suite_no_det, pf_no_det = msv3.build_suite_and_preflight(
		split="train",
		start_seed=3_000_001,
		end_seed=3_000_010,
		perception_sample=0.0,
		max_perception=0,
	)
	assert "hue_fp_control" in pf_no_det
	assert pf_no_det["hue_fp_control"]["status"] == "requires_detector"
	assert pf_no_det["hue_fp_control"]["pass"] is False
	assert pf_no_det["pass"] is False

	# Verify default target_cases is 60 (10 per 60-deg bin x 6 bins).
	assert pf_no_det["hue_fp_control"]["target_cases"] == 60
	assert len(pf_no_det["hue_fp_control"]["bins"]) == 6

	# 2. With a clean injected CPU scorer (never alarms), all 6 hue bins + grey baseline pass.
	cases = suite_no_det["cases"]
	lays = [sample_case(*c) for c in cases]
	clean_scorer = FixedScorer(lambda _cam, _img: -1.0, {"top": 0.0, "wrist_cam": 0.0})
	res_clean = msv3.run_hue_fp_control(cases, lays, scorer=clean_scorer, n_per_bin=1, n_views=1)
	assert res_clean["status"] == "passed"
	assert res_clean["pass"] is True
	assert len(res_clean["bins"]) == 6
	assert res_clean["grey_baseline"]["fp_rate"] == 0.0
	assert res_clean["failing_bins"] == []
	for binfo in res_clean["bins"].values():
		assert binfo["cases"] == 1
		assert binfo["fp_rate"] == 0.0
		assert binfo["delta_vs_grey"] == 0.0
		assert binfo["pass"] is True

	# 3. With a hue-sensitive scorer that alarms on green-ish table background pixels,
	#    coloured bin exceeds grey baseline by > 5 percentage points and MUST FAIL.
	def biased_score(_cam, img: np.ndarray) -> float:
		# Inspect border pixels of the wrist crop for greenish tint (G noticeably above R and B).
		border = np.concatenate([img[:12].reshape(-1, 3), img[-12:].reshape(-1, 3)], axis=0).astype(np.float32)
		g_excess = float((border[:, 1] - np.maximum(border[:, 0], border[:, 2])).mean())
		return 1.0 if g_excess > 6.0 else -1.0

	biased_scorer = FixedScorer(biased_score, {"top": 0.0, "wrist_cam": 0.0})
	res_biased = msv3.run_hue_fp_control(cases, lays, scorer=biased_scorer, n_per_bin=1, n_views=1)
	assert res_biased["status"] == "failed"
	assert res_biased["pass"] is False
	assert len(res_biased["failing_bins"]) >= 1
	assert res_biased["grey_baseline"]["fp_rate"] == 0.0

	# Wiring biased scorer into build_suite_and_preflight gates overall preflight["pass"].
	_, pf_biased = msv3.build_suite_and_preflight(
		split="train",
		start_seed=3_000_001,
		end_seed=3_000_010,
		perception_sample=0.0,
		max_perception=0,
		scorer=biased_scorer,
		hue_fp_per_bin=1,
		hue_fp_views=1,
	)
	assert pf_biased["hue_fp_control"]["pass"] is False
	assert pf_biased["pass"] is False
