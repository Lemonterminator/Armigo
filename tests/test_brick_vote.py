import json
import os
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pytest

from digital_twin_armigo.agent.brick_vote import shape_pick, vlm_pick
from digital_twin_armigo.agent.perception import BrickBlob


def blob(label, long_m, cells=1000):
	return BrickBlob(label, cells, np.zeros(2), 0.02, long_m, 0.03, np.array([[10, 10], [20, 30]]))


def test_shape_pick_skips_long_blobs():
	assert shape_pick([blob(1, 0.18), blob(2, 0.064)]) == 1
	assert shape_pick([blob(1, 0.12)]) is None


class FakeLLM:
	def __init__(self, answer):
		self.answer = answer

	def chat(self, messages, json_schema=None):
		if isinstance(self.answer, Exception):
			raise self.answer
		assert self.answer in json_schema["properties"]["candidate"]["enum"]
		return SimpleNamespace(content=json.dumps({"candidate": self.answer}), latency_s=0.1)


def test_vlm_pick_maps_answers_and_failures():
	rgb = np.zeros((48, 64, 3), np.uint8)
	cands = [blob(1, 0.18), blob(2, 0.064)]
	assert vlm_pick(FakeLLM("2"), rgb, cands)[0] == 1
	assert vlm_pick(FakeLLM("none"), rgb, cands)[0] is None
	idx, info = vlm_pick(FakeLLM(RuntimeError("down")), rgb, cands)
	assert idx is None and "error" in info


@pytest.mark.parametrize("localise", [True, False])
def test_graph_runs_localize_between_prepare_and_choose_view(localise):
	pytest.importorskip("langgraph")
	from digital_twin_armigo.agent.twin_graph import Policy, build_graph

	class Backend:
		def __init__(self):
			self.nodes = []

		def enter(self, node):
			self.nodes.append(node)

		def prepare(self):
			return None

		def can_localise(self):
			return localise

		def localise_brick(self):
			return None

		def can_scan_wrist(self, policy):
			return False

		def can_present_bottom(self, policy):
			return False

		def place(self, label):
			return None

		def evaluate(self, label, views, failures):
			return {"label": label}

	backend = Backend()
	build_graph(backend, Policy()).invoke({})
	expected = ["prepare", "localize", "choose_view"] if localise else ["prepare", "choose_view"]
	assert backend.nodes[: len(expected)] == expected


def test_vote_localise_picks_shape_blob_and_pins_it(tmp_path):
	pytest.importorskip("mujoco")
	from digital_twin_armigo.agent.act_grasp import grasp_trial

	# 3001203: the largest blob is a bowl-rim crescent (oracle bowls under drift); the shape rule takes the brick.
	case = [3001203, True, 3001203 % 6, {"scene": "v3", "split": "train", "cameras": "drift", "oracle": "bowls"}]
	r = grasp_trial(case, tmp_path / "ep", localise="vote")
	loc = r["localised"]
	assert loc["found"] and loc["agree"] and loc["chosen"] == loc["shape"] == 2
	assert loc["error_xy_mm"] < 5.0
	events = [json.loads(line) for line in (tmp_path / "ep" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
	sources = [e.get("source") for e in events if e["kind"] == "brick_perceived"]
	assert sources[:2] == ["vote", "pinned"]
