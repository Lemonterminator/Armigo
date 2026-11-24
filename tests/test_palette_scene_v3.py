import colorsys
import itertools
import json
import os
import subprocess
import sys

import pytest

from digital_twin_armigo.mujoco_cell.palette import HELD_OUT_PAIRS, PALETTE, hue_gap, jitter_rgb
from digital_twin_armigo.mujoco_cell.scene_v3 import SPLITS, sample_scene_v3


def test_palette_has_eight_colours_and_four_held_out_pairs():
	assert set(PALETTE) == {"green", "pink", "blue", "yellow", "orange", "purple", "white", "black"}
	assert len(HELD_OUT_PAIRS) == 4
	chromatic = [name for name, c in PALETTE.items() if c.hue_deg is not None]
	assert len(chromatic) == 6
	for a, b in itertools.combinations(chromatic, 2):
		assert hue_gap(a, b) >= 35.0


def test_jitter_rgb_respects_colour_bounds():
	import numpy as np

	rng = np.random.default_rng(42)
	for _ in range(100):
		w_rgb = jitter_rgb("white", rng)
		_, w_s, w_v = colorsys.rgb_to_hsv(*w_rgb)
		assert w_s <= 0.03 + 1e-6 and w_v <= PALETTE["white"].rgb[0] + 1e-6

		b_rgb = jitter_rgb("black", rng)
		_, b_s, b_v = colorsys.rgb_to_hsv(*b_rgb)
		assert (
			b_s <= 0.05 + 1e-6 and 0.90 * PALETTE["black"].rgb[0] - 1e-6 <= b_v <= 1.10 * PALETTE["black"].rgb[0] + 1e-6
		)

		g_rgb = jitter_rgb(PALETTE["green"], rng)
		assert all(0.0 <= ch <= 1.0 for ch in g_rgb)
		g_h, g_s, g_v = colorsys.rgb_to_hsv(*g_rgb)
		d = abs(g_h * 360.0 - PALETTE["green"].hue_deg) % 360.0
		assert min(d, 360.0 - d) <= 6.0 + 1e-6
		assert 0.20 - 1e-6 <= g_s <= 1.0 and 0.20 - 1e-6 <= g_v <= 1.0


def test_scene_is_deterministic():
	assert sample_scene_v3(3000001, "train") == sample_scene_v3(3000001, "train")


@pytest.mark.parametrize("split", ["train", "val", "ood-cam", "ood-lang", "test"])
def test_non_colour_splits_never_use_held_out_pairs(split):
	for seed in range(3000001, 3000401):
		s = sample_scene_v3(seed, split)
		names = [s.normal, s.defect] + ([s.extra] if s.extra else [])
		for a, b in itertools.combinations(names, 2):
			assert frozenset((a, b)) not in HELD_OUT_PAIRS
			assert hue_gap(a, b) >= 35.0


def test_ood_colour_split_always_contains_a_held_out_pair():
	for seed in range(3000001, 3000101):
		s = sample_scene_v3(seed, "ood-color")
		assert frozenset((s.normal, s.defect)) in HELD_OUT_PAIRS


def test_train_routing_is_balanced():
	scenes = [sample_scene_v3(seed, "train") for seed in range(3000001, 3002001)]
	counts = {c: sum(s.defect == c for s in scenes) for c in PALETTE}
	mean = len(scenes) / len(PALETTE)
	assert all(abs(n - mean) <= 0.25 * mean for n in counts.values())


def test_table_colour_respects_bowls():
	for seed in range(3000001, 3000401):
		s = sample_scene_v3(seed, "train")
		h, sat, v = colorsys.rgb_to_hsv(*s.table_rgb)
		assert sat <= 0.35 and 0.35 <= v <= 0.80
		if "white" in (s.normal, s.defect, s.extra):
			assert v <= 0.68
		if "black" in (s.normal, s.defect, s.extra):
			assert v >= 0.40
		if sat > 0.15:
			for name in (s.normal, s.defect, s.extra):
				if name and PALETTE[name].hue_deg is not None:
					d = abs(h * 360 - PALETTE[name].hue_deg) % 360
					assert min(d, 360 - d) >= 35.0


def test_camera_ranges():
	for seed in range(3000001, 3000201):
		cam = sample_scene_v3(seed, "train").camera
		assert abs(cam["dx_m"]) <= 0.03 and abs(cam["dyaw_deg"]) <= 8.0
		cam = sample_scene_v3(seed, "ood-cam").camera
		assert max(abs(cam["dx_m"]) / 0.03, abs(cam["dyaw_deg"]) / 8.0, abs(cam["dtilt_deg"]) / 5.0) >= 1.0


def test_scene_cross_process_determinism():
	code = (
		"import dataclasses, json; "
		"from digital_twin_armigo.mujoco_cell.scene_v3 import sample_scene_v3; "
		"print(json.dumps(dataclasses.asdict(sample_scene_v3(3000001, 'train')), sort_keys=True))"
	)
	env = os.environ.copy()
	env["PYTHONPATH"] = "src"
	out1 = subprocess.check_output([sys.executable, "-c", code], env=env, text=True)
	out2 = subprocess.check_output([sys.executable, "-c", code], env=env, text=True)
	assert out1 == out2
	parsed = json.loads(out1)
	assert parsed["normal"] and parsed["defect"]


def test_ood_appearance_and_ood_lang_constraints():
	for seed in range(3000001, 3000201):
		s_app = sample_scene_v3(seed, "ood-appearance")
		_, sat, _ = colorsys.rgb_to_hsv(*s_app.table_rgb)
		assert 0.35 <= sat <= 0.55
		assert s_app.table_texture in {"checker", "plain", "noise"}

		s_lang = sample_scene_v3(seed, "ood-lang")
		assert s_lang.variant == 5
		assert s_lang.table_texture in {"checker", "plain", "noise"}

	for split in SPLITS:
		if split == "ood-lang":
			continue
		for seed in range(3000001, 3000051):
			s = sample_scene_v3(seed, split)
			assert s.variant in range(5)
			assert s.table_texture in {"checker", "plain", "noise"}
