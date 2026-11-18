import asyncio
import json

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("cv2")
pytest.importorskip("mcp")

from digital_twin_armigo.agent import step_mcp as sm
from digital_twin_armigo.agent.twin_mujoco import ROOT

DETECTOR = ROOT / "results/agent-bench/sup-v7bot-local.json"
CASE = (600001, False, 5)


class FakeScorer:
	def threshold(self, camera):
		return 0.0

	def score(self, camera, image):
		return -1.0


def _service(tmp_path, profile):
	if not DETECTOR.is_file():
		pytest.skip("detector spec not present")
	return sm.StepService(tmp_path / profile, DETECTOR, profile, [CASE], scorer=FakeScorer())


def _payload(content):
	blocks = content[0] if isinstance(content, tuple) else content
	return json.loads(blocks[0].text)


def test_sensor_only_guard_rejects_simulator_keys():
	sm.assert_sensor_only({"views": [{"faces": [0], "score": 0.1}]})
	with pytest.raises(sm.SimulatorLeak):
		sm.assert_sensor_only({"a": [{"brick_position_m": [0, 0, 0]}]})
	with pytest.raises(sm.SimulatorLeak):
		sm.assert_sensor_only({"result": {"label_correct": True}})


@pytest.mark.parametrize("profile", sm.PROFILES)
def test_tool_sets_differ_only_in_their_observation_tools(tmp_path, profile):
	service = _service(tmp_path, profile)
	try:
		mcp = sm.create_step_mcp(service)
		names = {t.name for t in asyncio.run(mcp.list_tools())}
		assert names == {*sm.TOOLS_COMMON, *sm.TOOLS_ONLY[profile]}
	finally:
		service.close()


def test_realistic_session_observations_are_sensor_only(tmp_path):
	service = _service(tmp_path, "realistic")
	try:
		mcp = sm.create_step_mcp(service)

		async def run():
			case_id = _payload(await mcp.call_tool("list_cases", {}))["case_ids"][0]
			start = _payload(await mcp.call_tool("start_session", {"case_id": case_id}))
			sid = start["session_id"]
			out = {"start": start}
			for tool in ("robot_state", "camera_calibration", "locate_bowls", "locate_brick", "session_state"):
				out[tool] = _payload(await mcp.call_tool(tool, {"session_id": sid}))
			depth = await mcp.call_tool("capture", {"session_id": sid, "camera": "top", "kind": "depth"})
			out["capture"] = _payload(depth)
			from mcp.server.fastmcp.exceptions import ToolError

			with pytest.raises(ToolError):
				await mcp.call_tool("scene_state", {"session_id": sid})
			return out

		out = asyncio.run(run())
		for payload in out.values():
			sm.assert_sensor_only(payload)
		assert out["locate_brick"]["found"]
		assert out["start"]["session_state"]["budget"]["wrist_scans_left"] == sm.STEP_POLICY.max_wrist_views
		assert "jaw_normal_n" not in out["robot_state"]
		assert out["capture"]["shape"] == [480, 640]
	finally:
		service.close()


def test_scripted_client_sorts_a_normal_brick_through_the_step_tools(tmp_path):
	service = _service(tmp_path, "realistic")
	try:
		(result,) = sm.run_scripted(service)
	finally:
		service.close()
	assert result["label"] == "normal" and result["verdict"] == "SUCCESS", result["failures"]
