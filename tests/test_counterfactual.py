import json
from importlib import util
from pathlib import Path

import numpy as np
import pytest

from digital_twin_armigo.experiments.counterfactual import VARIANTS, verdict
from digital_twin_armigo.experiments.runner import group_name

ALL = ("camera", "brick", "bowls")


def test_repro_mismatch_wins():
	v = verdict(False, {(): True, ("camera",): True, ("brick",): False, ("bowls",): False, ALL: True})
	assert v["verdict"] == "repro_mismatch"


def test_confirmed_minimal_sets():
	v = verdict(False, {(): False, ("camera",): True, ("brick",): True, ("bowls",): False, ALL: True})
	assert v == {"verdict": "confirmed", "minimal": [["brick"], ["camera"]]}


def test_joint_and_not_perception_and_incomplete():
	assert (
		verdict(False, {(): False, ("camera",): False, ("brick",): False, ("bowls",): False, ALL: True})["verdict"]
		== "joint"
	)
	assert (
		verdict(False, {(): False, ("camera",): False, ("brick",): False, ("bowls",): False, ALL: False})["verdict"]
		== "not_perception"
	)
	assert (
		verdict(False, {(): False, ("camera",): None, ("brick",): False, ("bowls",): False, ALL: True})["verdict"]
		== "incomplete"
	)


def test_group_name_oracle_suffix():
	assert group_name("realistic", "drift", "A", "v7bot") == "realistic-drift-A-v7bot"
	assert (
		group_name("realistic", "drift", "A", "v7bot", oracle=("brick", "camera"))
		== "realistic-drift-A-v7bot-oracle-brick+camera"
	)


@pytest.mark.slow
def test_true_world_from_cam_matches_hand_eye_on_the_nominal_rig():
	pytest.importorskip("mujoco")
	from digital_twin_armigo.agent.perception import PerceptionCfg
	from digital_twin_armigo.agent.twin_mujoco import sample_case, true_world_from_cam
	from digital_twin_armigo.mujoco_cell.handeye_rig import calibrate_layout

	lay = sample_case(600001, False, 5)
	_, est, _ = calibrate_layout(lay, 0, PerceptionCfg().calibration)
	truth = true_world_from_cam(lay.top)
	assert np.linalg.norm(est[:3, 3] - truth[:3, 3]) < 0.001
	for col in range(3):
		cos = float(np.dot(est[:3, col], truth[:3, col]))
		assert np.degrees(np.arccos(np.clip(cos, -1, 1))) < 0.5


def test_step_service_oracle_replaces_beliefs_and_disqualifies_student(tmp_path):
	pytest.importorskip("mujoco")
	pytest.importorskip("cv2")
	pytest.importorskip("mcp")
	from digital_twin_armigo.agent import step_mcp as sm
	from digital_twin_armigo.agent.twin_mujoco import ROOT, true_world_from_cam

	detector = ROOT / "results/agent-bench/sup-v7bot-local.json"
	if not detector.is_file():
		pytest.skip("detector spec not present")

	class FakeScorer:
		def threshold(self, camera):
			return 0.0

		def score(self, camera, image):
			return -1.0

	case = (600001, False, 5, {"cameras": "drift", "oracle": ["camera", "bowls"]})
	service = sm.StepService(tmp_path / "realistic", detector, "realistic", [case], scorer=FakeScorer())
	try:

		def _check():
			cid = next(iter(service.cases))
			started = service.start(cid)
			sid = started["session_id"]
			sess = service.session(sid)
			b = sess.backend
			assert np.allclose(b._cam_est, true_world_from_cam(b.layout.top), atol=1e-9)
			for name in ("green", "pink"):
				assert np.allclose(b.bowls[name], b._true_bowl_xy(name), atol=1e-9)
			applied = [e for e in sess.trace.events if e["kind"] == "oracle_applied"]
			assert applied and applied[0]["parts"] == ["bowls", "camera"]
			res = service.finish(sid)["result"]
			assert res["student_eligible"] is False

		service.run_sync(_check)
	finally:
		service.close()


def test_step_service_brick_oracle_zeroes_perception_error(tmp_path):
	pytest.importorskip("mujoco")
	pytest.importorskip("cv2")
	pytest.importorskip("mcp")
	from digital_twin_armigo.agent import step_mcp as sm
	from digital_twin_armigo.agent.twin_mujoco import ROOT

	detector = ROOT / "results/agent-bench/sup-v7bot-local.json"
	if not detector.is_file():
		pytest.skip("detector spec not present")

	class FakeScorer:
		def threshold(self, camera):
			return 0.0

		def score(self, camera, image):
			return -1.0

	case = (600001, False, 5, {"cameras": "drift", "oracle": ["brick"]})
	service = sm.StepService(tmp_path / "realistic", detector, "realistic", [case], scorer=FakeScorer())
	try:

		def _check():
			cid = next(iter(service.cases))
			sid = service.start(cid)["session_id"]
			sess = service.session(sid)
			assert sess.backend._perceive() is None
			perceived = [e for e in sess.trace.events if e["kind"] == "brick_perceived"]
			assert perceived and perceived[0]["error_xy_mm"] == 0.0 and perceived[0]["error_yaw_deg"] == 0.0
			assert perceived[0]["size_correct"] is True
			res = service.finish(sid)["result"]
			assert res["student_eligible"] is False

		service.run_sync(_check)
	finally:
		service.close()


def test_counterfactual_cli_plan_and_summarize(tmp_path):
	script = Path(__file__).resolve().parents[1] / "scripts/experiments/counterfactual.py"
	spec = util.spec_from_file_location("cf_script", script)
	cf = util.module_from_spec(spec)
	assert spec.loader is not None
	spec.loader.exec_module(cf)

	ep = tmp_path / "ep"
	ep.mkdir()
	(ep / "task.json").write_text(json.dumps({"case": [700001, True, 2]}), encoding="utf-8")
	base = tmp_path / "base.jsonl"
	base.write_text(
		json.dumps(
			{
				"case_id": "L700001-D2",
				"group": "realistic-drift-B-v7bot",
				"physical_success": False,
				"root_cause": "recovery_failure",
				"mechanism": "brick_pose_error",
				"path": str(ep),
			}
		)
		+ "\n",
		encoding="utf-8",
	)
	suite_out = tmp_path / "cf_suite.json"
	manifest = tmp_path / "manifest.json"
	assert cf.main(["plan", "--scores", str(base), "--out", str(suite_out), "--manifest", str(manifest)]) == 0
	assert json.loads(suite_out.read_text(encoding="utf-8")) == [[700001, True, 2]]
	assert "cf_suite" in json.loads(manifest.read_text(encoding="utf-8"))["suites"]

	vfiles = []
	for i, parts in enumerate(VARIANTS):
		vf = tmp_path / f"v{i}.jsonl"
		suffix = f"-oracle-{'+'.join(parts)}" if parts else ""
		ok = "camera" in parts
		vf.write_text(
			json.dumps({"case_id": "L700001-D2", "group": f"realistic-drift-B-v7bot{suffix}", "physical_success": ok})
			+ "\n",
			encoding="utf-8",
		)
		vfiles.append(vf)
	cf_out = tmp_path / "counterfactual.jsonl"
	assert (
		cf.main(["summarize", "--base", str(base), "--variants", *[str(p) for p in vfiles], "--out", str(cf_out)]) == 0
	)
	summary = [json.loads(l) for l in cf_out.read_text(encoding="utf-8").splitlines() if l.strip()]
	assert summary == [
		{
			"case_id": "L700001-D2",
			"engine": "mujoco",
			"camera_condition": "drift",
			"strategy": "B",
			"motion": "open",
			"evidence": "hold",
			"segmenter": "lab",
			"group": "realistic-drift-B-v7bot",
			"path": str(ep),
			"symptom": "recovery_failure",
			"mechanism": "brick_pose_error",
			"verdict": "confirmed",
			"minimal": [["camera"]],
		}
	]
