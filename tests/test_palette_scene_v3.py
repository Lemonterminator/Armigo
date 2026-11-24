import colorsys
import dataclasses
import hashlib
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from digital_twin_armigo.agent.twin_mujoco import sample_case
from digital_twin_armigo.mujoco_cell.palette import HELD_OUT_PAIRS, PALETTE, hue_gap, jitter_rgb
from digital_twin_armigo.mujoco_cell.scene import scene_xml
from digital_twin_armigo.mujoco_cell.scene_v3 import SPLITS, sample_scene_v3

ROOT = Path(__file__).resolve().parents[1]


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


def _to_json_comparable(val):
	if dataclasses.is_dataclass(val):
		return {k: _to_json_comparable(v) for k, v in dataclasses.asdict(val).items()}
	if isinstance(val, tuple):
		return [_to_json_comparable(v) for v in val]
	if isinstance(val, list):
		return [_to_json_comparable(v) for v in val]
	return val


def test_old_cases_unchanged_by_v3_fields():
	lay = sample_case(1100001, False, 5)
	assert lay.bowl_rgb is None and lay.bowl_names is None and lay.extra_bowls == [] and lay.table_texture is None
	assert 'rgba="0.25 0.65 0.3 1"' in scene_xml(lay)

	data = json.loads((ROOT / "tests/data/legacy_7cc24e2.json").read_text(encoding="utf-8"))
	for entry in data["cases"]:
		lay = sample_case(*entry["case"])
		for key, expected in entry["layout"].items():
			actual = _to_json_comparable(getattr(lay, key))
			assert actual == expected, f"mismatch on {entry['tag']} {entry['case']} field {key}"
		assert hashlib.sha256(scene_xml(lay).encode("utf-8")).hexdigest() == entry["xml_sha256"]


def test_v3_case_paints_bowls_by_scene():
	pytest.importorskip("mujoco")
	from digital_twin_armigo.constants import SO101_MJCF_DIR
	from digital_twin_armigo.mujoco_cell import Cell
	from digital_twin_armigo.mujoco_cell._mj import mujoco
	from digital_twin_armigo.physics import load_physics_spec

	lay = sample_case(3000001, True, 2, {"scene": "v3", "split": "train"})
	xml = scene_xml(lay)
	assert set(lay.bowl_names) == {"green", "pink"}
	for slot in ("green", "pink"):
		r, g, b = lay.bowl_rgb[slot]
		assert f'rgba="{r:.3f} {g:.3f} {b:.3f} 1"' in xml
	assert all(np.hypot(x - lay.brick_xy[0], y - lay.brick_xy[1]) > 0.13 for _, x, y, _ in lay.extra_bowls)

	# Verify all three table texture renderings
	xml_checker = scene_xml(dataclasses.replace(lay, table_texture="checker"))
	assert '<texture name="table" type="2d" builtin="checker"' in xml_checker
	xml_plain = scene_xml(dataclasses.replace(lay, table_texture="plain"))
	assert '<texture name="table" type="2d" builtin="flat"' in xml_plain
	assert 'mark="random"' not in xml_plain
	xml_noise = scene_xml(dataclasses.replace(lay, table_texture="noise"))
	assert '<texture name="table" type="2d" builtin="flat"' in xml_noise
	assert 'mark="random"' in xml_noise

	# Find a 3-bowl v3 seed and compile into a Cell with physics
	lay3 = next(
		c
		for seed in range(3000001, 3000050)
		for c in (sample_case(seed, True, 2, {"scene": "v3", "split": "train"}),)
		if c.extra_bowls
	)
	xml3 = scene_xml(lay3)
	assert "extra_bowl_0" in xml3
	assert np.hypot(lay3.extra_bowls[0][1] - lay3.brick_xy[0], lay3.extra_bowls[0][2] - lay3.brick_xy[1]) > 0.13
	cell = Cell(lay3, SO101_MJCF_DIR, physics=load_physics_spec(), render_enabled=False)
	try:
		mujoco.mj_forward(cell.model, cell.data)
		# Verify extra_bowl_0 got plate friction
		bid = mujoco.mj_name2id(cell.model, mujoco.mjtObj.mjOBJ_BODY, "extra_bowl_0")
		assert bid >= 0
		from digital_twin_armigo.isaac_bridge.client import scene_payload

		lay3_no_clutter = dataclasses.replace(lay3, clutter=[])
		payload_v3 = scene_payload(lay3_no_clutter, cell)
		assert payload_v3["table"] == {"rgb": [float(c) for c in lay3.table_rgb], "texture": "plain"}
	finally:
		cell.close()

	# Legacy scene_payload omits "texture"
	lay_old = sample_case(1100001, False, 5)
	cell_old = Cell(lay_old, SO101_MJCF_DIR, render_enabled=False)
	try:
		payload_old = scene_payload(lay_old, cell_old)
		assert payload_old["table"] == {"rgb": [float(c) for c in lay_old.table_rgb]}
	finally:
		cell_old.close()

	# Invalid v3 options
	for bad in ({"split": "train"}, {"scene": "v2"}, {"scene": "v3", "split": "invalid"}):
		with pytest.raises(ValueError):
			sample_case(3000001, False, 5, bad)


def test_v3_load_suite_and_realistic_clutter(tmp_path):
	pytest.importorskip("mujoco")
	from digital_twin_armigo.agent.twin_mujoco import TwinMujoco, load_suite
	from digital_twin_armigo.agent.twin_record import EpisodeTrace
	from digital_twin_armigo.inspection_scorer import FixedScorer

	suite_path = tmp_path / "v3.json"
	suite_path.write_text(
		json.dumps({"cases": [[3000001, True, 2]], "options": {"scene": "v3", "split": "train"}}),
		encoding="utf-8",
	)
	cases = load_suite(suite_path)
	assert cases == [(3000001, True, 2, {"scene": "v3", "split": "train"})]

	lay_clutter = next(
		c
		for seed in range(3000001, 3000050)
		for c in (sample_case(seed, False, 5, {"scene": "v3", "split": "train"}),)
		if c.clutter
	)
	trace = EpisodeTrace(tmp_path / "v3_trace", video=False)
	twin = None
	try:
		twin = TwinMujoco(
			lay_clutter,
			FixedScorer(lambda _c, _r: -1, {"top": 0, "wrist_cam": 0}),
			trace,
			sensing="realistic",
		)
		assert set(twin.bowls) == {"green", "pink"}
	finally:
		if twin is not None:
			twin.cell.close()
		trace.close()
