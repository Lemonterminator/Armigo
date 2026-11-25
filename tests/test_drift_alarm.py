import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest

from digital_twin_armigo.agent.perception import top_camera_drift
from digital_twin_armigo.agent.twin_graph import RECALIBRATE, Policy, build_graph


@pytest.mark.parametrize(
	"correction,accepted,table_z,expected",
	[
		(6.5, True, 0.0, False),
		(6.501, True, 0.0, True),
		(50.0, False, 0.0, False),
		(0.0, False, 2.5, False),
		(0.0, False, -2.5, False),
		(0.0, False, 2.501, True),
		(0.0, False, -2.501, True),
	],
)
def test_drift_thresholds(correction, accepted, table_z, expected):
	assert top_camera_drift(correction, accepted, table_z) is expected


@pytest.mark.parametrize("second_alarm", [False, True])
def test_graph_recalibrates_once_and_never_rewinds_alarm(second_alarm):
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
			return RECALIBRATE if self.calls == 1 or second_alarm else None

		def recalibrate(self):
			return None

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
	assert backend.nodes[:4] == ["prepare", "localize", "recalibrate", "localize"]
	assert backend.nodes.count("recalibrate") == 1
	if second_alarm:
		assert backend.nodes[4:] == ["recover", "evaluate"]
		assert result["result"]["failures"] == [{"code": RECALIBRATE}]
	else:
		assert backend.nodes[4:] == ["choose_view", "place", "evaluate"]


def _without_wall_time(value):
	if isinstance(value, dict):
		return {k: _without_wall_time(v) for k, v in value.items() if k != "wall_time_s"}
	if isinstance(value, list):
		return [_without_wall_time(v) for v in value]
	return value


def test_drift_alarm_end_to_end_and_default_equivalence(tmp_path):
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
	alarm = grasp_trial(case, alarm_dir, localise="vote", drift_alarm=True, **kwargs)
	events = [json.loads(line) for line in (alarm_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
	kinds = [event["kind"] for event in events]
	assert kinds.count("recalibration_needed") == 1
	assert kinds.count("camera_recalibrated") == 1
	assert kinds.index("recalibration_needed") < kinds.index("camera_recalibrated")
	assert alarm["localised"]["error_xy_mm"] < 5.0
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
