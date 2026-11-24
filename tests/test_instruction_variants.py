import pytest

from digital_twin_armigo.agent.student_rollout import with_grasp_roll
from digital_twin_armigo.agent.twin_record import Segment
from digital_twin_armigo.vla.contract import (
	HELD_OUT_VARIANT,
	INSTRUCTION_VARIANTS,
	SEGMENT_TEMPLATES,
	TRAIN_VARIANTS,
	V3_EXTRA_BOWL_COLOURS,
	encode_state,
	format_instruction,
)
from digital_twin_armigo.vla.dataset import SegmentChunkDataset, StudentImageAugmentor


def test_variant_zero_is_the_current_template():
	assert TRAIN_VARIANTS == (0, 1, 2, 3, 4)
	assert HELD_OUT_VARIANT == 5
	for kind, template in SEGMENT_TEMPLATES.items():
		assert kind in INSTRUCTION_VARIANTS
		assert INSTRUCTION_VARIANTS[kind][0] == template
		assert format_instruction(kind, {"view": 2, "views": 4, "bowl": "green"}, variant=0) == template.format(
			view=2, views=4, bowl="green"
		)


def test_six_variants_each_and_bowl_word_present():
	for kind, variants in INSTRUCTION_VARIANTS.items():
		assert len(variants) == 6
		assert len(set(variants)) == 6
		for v in range(6):
			if kind == "scan_view":
				text = format_instruction(kind, {"view": 2, "views": 4}, variant=v)
				assert "2" in text and "4" in text
			if kind == "carry_to_bowl":
				text = format_instruction(kind, {"bowl": "purple"}, variant=v)
				assert "purple" in text
				lower = text.lower()
				assert "other" not in lower
				assert "left" not in lower
				assert "right" not in lower


def test_token_length_under_40():
	transformers = pytest.importorskip("transformers")
	tok = transformers.AutoTokenizer.from_pretrained(
		"HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
		local_files_only=True,
	)
	params = {"view": 4, "views": 4, "bowl": "orange", "roll_deg": -135.0}
	for kind, variants in INSTRUCTION_VARIANTS.items():
		for v in range(len(variants)):
			instr = format_instruction(kind, params, variant=v)
			input_ids = tok(instr)["input_ids"]
			assert len(input_ids) <= 40, (kind, v, len(input_ids), instr)


@pytest.mark.xfail(
	strict=True,
	reason="Plan mismatch: spec §3 keeps hue=0.05 (18.0 deg) while Task 5 asserts < 35.0 / 2 (17.5 deg)",
)
def test_hue_augmentation_cannot_cross_palette_gap():
	assert StudentImageAugmentor().hue * 360.0 < 35.0 / 2


def test_hue_augmentation_under_half_realized_palette_gap():
	from digital_twin_armigo.mujoco_cell.palette import PALETTE, hue_gap

	chromatic = [c for c in PALETTE if c != "white"]
	min_gap = min(hue_gap(a, b) for i, a in enumerate(chromatic) for b in chromatic[i + 1 :])
	assert StudentImageAugmentor().hue * 360.0 < min_gap / 2.0


def test_v3_episode_zeroes_bowl_one_hot_and_applies_variant():
	s_default = encode_state(
		q_t=[0, 0, 0, 0, 0, 0],
		q_prev=None,
		prev_action=None,
		gripper_cmd=0.85,
		segment_kind="grasp",
		bowl="green",
		zero_bowl=False,
	)
	assert s_default[31:34].tolist() == [0.0, 1.0, 0.0]

	s_zero = encode_state(
		q_t=[0, 0, 0, 0, 0, 0],
		q_prev=None,
		prev_action=None,
		gripper_cmd=0.85,
		segment_kind="grasp",
		bowl="green",
		zero_bowl=True,
	)
	assert s_zero[31:34].tolist() == [0.0, 0.0, 0.0]

	for col in V3_EXTRA_BOWL_COLOURS:
		s_extra = encode_state(
			q_t=[0, 0, 0, 0, 0, 0],
			q_prev=None,
			prev_action=None,
			gripper_cmd=0.85,
			segment_kind="grasp",
			bowl=col,
		)
		assert s_extra[31:34].tolist() == [0.0, 0.0, 0.0]

	seg = with_grasp_roll(Segment("grasp", {}, variant=3), -1.0)
	assert seg.variant == 3
	assert seg.instruction == format_instruction("grasp", {"roll_deg": seg.params["roll_deg"]}, variant=3)

	mock_v3 = {
		"episode_index": 0,
		"scene": "v3",
		"variant": 3,
		"frames": [
			{
				"observation.state": [0] * 6,
				"action": [0] * 6,
				"observation.prev_action": [0] * 6,
				"observation.gripper_cmd": 0.85,
				"segment.index": 0,
				"segment.kind": "carry_to_bowl",
				"segment.chunk_valid_len": 1,
				"segment.bowl": "green",
				"segment.view": 0,
				"segment.views": 0,
			}
		],
		"segments": [
			{
				"id": 0,
				"kind": "carry_to_bowl",
				"params": {"bowl": "green"},
				"start": 0,
				"end": 1,
				"reason": "done",
				"variant": 3,
			},
			{
				"id": 1,
				"kind": "let_go",
				"params": {},
				"start": 1,
				"end": 1,
				"reason": "done",
				"variant": 3,
			},
		],
	}
	ds_v3 = SegmentChunkDataset.from_episodes([mock_v3], horizon=10, return_torch=False)
	assert len(ds_v3) == 2
	for sample in ds_v3:
		assert sample["observation.state"][31:34].tolist() == [0.0, 0.0, 0.0]
	assert ds_v3[0]["task"] == format_instruction("carry_to_bowl", {"bowl": "green"}, variant=3)
	assert ds_v3[1]["task"] == format_instruction("let_go", {}, variant=3)

	mock_legacy = {
		"episode_index": 1,
		"frames": mock_v3["frames"],
		"segments": [
			{
				"id": 0,
				"kind": "carry_to_bowl",
				"params": {"bowl": "green"},
				"start": 0,
				"end": 1,
				"reason": "done",
			}
		],
	}
	ds_old = SegmentChunkDataset.from_episodes([mock_legacy], horizon=10, return_torch=False)
	assert ds_old[0]["observation.state"][31:34].tolist() == [0.0, 1.0, 0.0]
	assert ds_old[0]["task"] == format_instruction("carry_to_bowl", {"bowl": "green"}, variant=0)
