"""Paired opposite-target carry demonstrations (docs/vla/BOWL_CHANNEL_PAIRED_PLAN_2026-11-23.md Tasks 2-3)."""

from __future__ import annotations

import numpy as np

from digital_twin_armigo.agent.twin_mujoco import TwinMujoco
from digital_twin_armigo.vla.paired_carry import TargetSwapTracer, restrict_supervision


class _Stub:
	target_swap = True


def test_target_bowl_default_mapping():
	assert TwinMujoco.target_bowl(object(), "defective") == "pink"
	assert TwinMujoco.target_bowl(object(), "normal") == "green"


def test_target_swap_flips_the_mapping():
	s = _Stub()
	assert TargetSwapTracer.target_bowl(s, "defective") == "green"
	assert TargetSwapTracer.target_bowl(s, "normal") == "pink"
	s.target_swap = False
	assert TargetSwapTracer.target_bowl(s, "defective") == "pink"
	assert TargetSwapTracer.target_bowl(s, "normal") == "green"


def test_restrict_supervision_keeps_only_carry_and_release():
	arrays = {
		"segment": np.array([0, 0, 1, 1, 2, 2, 3]),
		"supervise_action": np.ones(7, bool),
		"action": np.zeros((7, 6)),
	}
	kinds = ["grasp", "return_to_lift", "carry_to_bowl", "release_in_bowl"]
	out = restrict_supervision(arrays, kinds)
	assert out["supervise_action"].tolist() == [False, False, False, False, True, True, True]
	assert arrays["supervise_action"].all()  # input untouched
	assert out["action"] is not arrays["action"]
	out2 = restrict_supervision(arrays, kinds, from_frame=5)
	assert out2["supervise_action"].tolist() == [False] * 5 + [True, True]


def test_restrict_supervision_respects_existing_false_and_unlabelled_frames():
	arrays = {"segment": np.array([2, 2, -1]), "supervise_action": np.array([False, True, True])}
	out = restrict_supervision(arrays, ["a", "b", "carry_to_bowl"])
	assert out["supervise_action"].tolist() == [False, True, False]
	no_col = restrict_supervision({"segment": np.array([2, 0])}, ["a", "b", "carry_to_bowl"])
	assert no_col["supervise_action"].tolist() == [True, False]
