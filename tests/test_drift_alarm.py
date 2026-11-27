import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest

from digital_twin_armigo.agent.perception import top_camera_drift
from digital_twin_armigo.agent.twin_graph import RECALIBRATE, RECALIBRATION_FAILED, Policy, build_graph


@pytest.mark.parametrize(
	"correction,accepted,table_z,expected",
	[
		(6.5, True, 0.0, False),
		(6.501, True, 0.0, True),
		(50.0, False, 0.0, False),
		(0.0, True, 2.79, False),
		(0.0, False, 3.0, False),
		(0.0, False, -3.0, False),
		(0.0, False, 3.001, True),
		(0.0, False, -3.001, True),
	],
)
def test_drift_thresholds(correction, accepted, table_z, expected):
	assert top_camera_drift(correction, accepted, table_z) is expected


@pytest.mark.parametrize("calibration_error", [None, RECALIBRATION_FAILED])
def test_graph_recalibrates_once_and_never_rewinds_alarm(calibration_error):
	pytest.importorskip("langgraph")

	class Backend:
		def __init__(self):
			self.nodes = []
			self.calls = 0

		def enter(self, node):
			self.nodes.append(node)

		def prepare(self):
			return None

		def can_localise(self):
			return True

		def localise_brick(self):
			self.calls += 1
			return RECALIBRATE if self.calls == 1 else None

		def recalibrate(self):
			return calibration_error

		def failure(self, error, views):
			return {"code": error}

		def rewind(self, policy, failures):
			pytest.fail("an alarm must not rewind physics")

		def can_scan_wrist(self, policy):
			return False

		def can_present_bottom(self, policy):
			return False

		def place(self, label):
			return None

		def evaluate(self, label, views, failures):
			return {"failures": failures}

	backend = Backend()
	result = build_graph(backend, Policy()).invoke({})
	assert backend.nodes[:3] == ["prepare", "localize", "recalibrate"]
	assert backend.nodes.count("recalibrate") == 1
	if calibration_error:
		assert backend.nodes[3:] == ["recover", "evaluate"]
		assert result["result"]["failures"] == [{"code": RECALIBRATION_FAILED}]
	else:
		assert backend.nodes[3:] == ["localize", "choose_view", "place", "evaluate"]
		assert result["result"]["failures"] == []


def _without_wall_time(value):
	if isinstance(value, dict):
		return {k: _without_wall_time(v) for k, v in value.items() if k != "wall_time_s"}
	if isinstance(value, list):
		return [_without_wall_time(v) for v in value]
	return value


def test_drift_alarm_end_to_end_and_default_equivalence(tmp_path, monkeypatch):
	pytest.importorskip("mujoco")
	import numpy as np

	from digital_twin_armigo.agent.act_grasp import grasp_trial
	from digital_twin_armigo.agent.twin_mujoco import TwinMujoco, sample_case
	from digital_twin_armigo.agent.twin_record import EpisodeTrace
	from digital_twin_armigo.agent.twin_resume import load_plan, restore

	seed = 3001205
	case = [seed, True, seed % 6, {"scene": "v3", "split": "train", "cameras": "drift", "oracle": "bowls"}]
	kwargs = {"expert_overrides": {"descend_close_ratio": 1.3}}
	alarm_dir = tmp_path / "alarm"
	detected = []

	def force_repeated_alarm(correction_mm, ring_accepted, table_z_mm):
		detected.append(top_camera_drift(correction_mm, ring_accepted, table_z_mm))
		return detected[-1] or len(detected) == 2

	with monkeypatch.context() as patch:
		patch.setattr("digital_twin_armigo.agent.twin_mujoco.top_camera_drift", force_repeated_alarm)
		alarm = grasp_trial(case, alarm_dir, localise="vote", drift_alarm=True, **kwargs)

	assert detected == [True, False]
	events = [json.loads(line) for line in (alarm_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
	kinds = [event["kind"] for event in events]
	alarm_events = [event for event in events if event["kind"] == "recalibration_needed"]
	assert [event["already_attempted"] for event in alarm_events] == [False, True]
	assert kinds.count("camera_recalibrated") == 1
	assert alarm_events[0]["seq"] < kinds.index("camera_recalibrated") < alarm_events[1]["seq"]
	assert any(event["kind"] == "brick_perceived" and event.get("source") == "pinned" for event in events)
	assert alarm["localised"]["error_xy_mm"] < 5.0
	assert alarm["lifted"] and alarm["error"] is None
	plan = load_plan(alarm_dir)
	assert plan.graph_state()["recalibration_attempted"] is True
	camera = plan.latest.agent["camera"]
	assert camera != plan.checkpoints[0].agent["camera"]
	trace = EpisodeTrace(tmp_path / "restored", video=False)
	twin = TwinMujoco(sample_case(*case), None, trace, sensing="realistic", localise="vote", drift_alarm=True)
	try:
		restore(twin, plan)
		np.testing.assert_array_equal(twin._cam_k, camera["k"])
		np.testing.assert_array_equal(twin._cam_est, camera["world_from_cam"])
		assert twin.recalibrate() == RECALIBRATE
	finally:
		twin.close()
		trace.close()

	default_dir, false_dir = tmp_path / "default", tmp_path / "false"
	default = grasp_trial(case, default_dir, **kwargs)
	explicit_false = grasp_trial(case, false_dir, localise="top", drift_alarm=False, **kwargs)
	assert _without_wall_time(default) == _without_wall_time(explicit_false)
	default_events = [
		json.loads(line) for line in (default_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
	]
	false_events = [json.loads(line) for line in (false_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
	assert _without_wall_time(default_events) == _without_wall_time(false_events)

	seed = 3001370  # The top camera has no valid shape pick before recalibration.
	missing_case = [seed, False, seed % 6, case[3]]
	missing_dir = tmp_path / "missing"
	missing = grasp_trial(missing_case, missing_dir, localise="vote", drift_alarm=True, **kwargs)
	missing_events = [
		json.loads(line) for line in (missing_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
	]
	assert [event["reason"] for event in missing_events if event["kind"] == "recalibration_requested"] == [
		"BRICK_NOT_FOUND"
	]
	assert sum(event["kind"] == "camera_recalibrated" for event in missing_events) == 1
	assert all(event["kind"] != "recalibration_needed" for event in missing_events)
	assert missing["lifted"] and missing["error"] is None


def test_rejected_ring_requests_one_recalibration(tmp_path, monkeypatch):
	"""Round 3: a rejected ring leaves the drift unmeasured, so the alarm recalibrates (once) instead of trusting
	the top view."""
	pytest.importorskip("mujoco")
	from digital_twin_armigo.agent.act_grasp import grasp_trial
	from digital_twin_armigo.agent.twin_mujoco import TwinMujoco

	seed = 3001205
	case = [seed, True, seed % 6, {"scene": "v3", "split": "train", "cameras": "drift", "oracle": "bowls"}]
	original = TwinMujoco._ring_localise

	def reject(self):
		error, _ = original(self)
		return error, False

	monkeypatch.setattr(TwinMujoco, "_ring_localise", reject)
	out = tmp_path / "rejected"
	result = grasp_trial(case, out, localise="vote", drift_alarm=True, expert_overrides={"descend_close_ratio": 1.3})
	events = [json.loads(line) for line in (out / "events.jsonl").read_text(encoding="utf-8").splitlines()]
	alarms = [e for e in events if e["kind"] == "recalibration_needed"]
	assert alarms and alarms[0]["reason"] in ("drift", "ring_rejected") and alarms[0]["already_attempted"] is False
	assert all(e["reason"] == "ring_rejected" for e in alarms[1:])  # the forced rejection after recalibration
	assert [e["already_attempted"] for e in alarms] == [False, True]
	assert sum(e["kind"] == "camera_recalibrated" for e in events) == 1
	assert result["error"] is None
