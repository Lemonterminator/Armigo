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


def test_unsupervised_prefix_end_keeps_one_row_before_supervision(tmp_path):
	from digital_twin_armigo.agent.twin_record import unsupervised_prefix_end

	(tmp_path / "branch-0").mkdir()
	sup = np.zeros(10, dtype=bool)
	sup[6:9] = True
	np.savez(tmp_path / "branch-0" / "actions.npz", supervise_action=sup, action=np.zeros((10, 6)))
	assert unsupervised_prefix_end(tmp_path) == 5
	np.savez(tmp_path / "branch-0" / "actions.npz", supervise_action=np.zeros(10, bool), action=np.zeros((10, 6)))
	assert unsupervised_prefix_end(tmp_path) == 0
	np.savez(tmp_path / "branch-0" / "actions.npz", action=np.zeros((10, 6)))
	assert unsupervised_prefix_end(tmp_path) == 0


def test_trimmed_segments_rebase_and_drop_earlier_ones():
	from digital_twin_armigo.agent.twin_record import _trimmed_segments

	segs = [
		{"id": 0, "start": 0, "end": 4},
		{"id": 1, "start": 4, "end": 4},
		{"id": 2, "start": 4, "end": 9},
		{"id": 3, "start": 9, "end": 12},
		{"id": 4, "start": 12, "end": 12},
	]
	got = _trimmed_segments(segs, 5)
	assert [(s["id"], s["start"], s["end"]) for s in got] == [(2, 0, 4), (3, 4, 7), (4, 7, 7)]
	assert _trimmed_segments(segs, 0) == segs
