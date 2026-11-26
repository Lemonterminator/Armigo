import json
from pathlib import Path

import numpy as np
import pytest

from digital_twin_armigo.agent import perception as pc
from digital_twin_armigo.agent.twin_mujoco import INITIAL_Q, sample_case
from digital_twin_armigo.constants import SO101_MJCF_DIR, TOP_FULL
from digital_twin_armigo.mujoco_cell import handeye as he
from digital_twin_armigo.mujoco_cell.cell import Cell
from digital_twin_armigo.mujoco_cell.palette import PALETTE
from digital_twin_armigo.mujoco_cell.physics_expert import ExpertParams

ROOT = Path(__file__).resolve().parents[1]


def _render_top_and_locate(layout, *, bands=None, q=INITIAL_Q, depth_noise_seed=None):
	cell = Cell(layout, SO101_MJCF_DIR)
	try:
		cell.set_joints(q)
		r = cell.renderer
		r.enable_depth_rendering()
		r.update_scene(cell.data, "top")
		depth = r.render().astype(np.float64)
		r.disable_depth_rendering()
		if depth_noise_seed is not None:  # the realistic stereo noise of TwinMujoco._top_depth
			sigma = pc.PerceptionCfg().depth_noise_m * (depth / 0.5) ** 2
			depth = depth + np.random.default_rng(depth_noise_seed).normal(0.0, 1.0, depth.shape) * sigma
		k = he.intrinsics(layout.top, TOP_FULL)
		cam = he.camera_truth(cell.model, cell.data, "top")
		return pc.locate_bowls(cell.rgb("top"), depth, k, cam, bands=bands)
	finally:
		cell.close()


def test_legacy_locate_bowls_unchanged():
	pytest.importorskip("mujoco")
	pytest.importorskip("cv2")
	expected = json.loads((ROOT / "tests/data/legacy_7cc24e2.json").read_text(encoding="utf-8"))["perception_600001"]
	layout = sample_case(600001, False, 5)
	found = _render_top_and_locate(layout, bands=None)
	assert set(found) == {"green", "pink"}
	for name in ("green", "pink"):
		est = found[name]
		assert est is not None
		exp = expected[name]
		assert est.xy.tolist() == pytest.approx(exp["xy"], abs=1e-12, rel=0.0)
		assert est.radius_m == pytest.approx(exp["radius_m"], abs=1e-12, rel=0.0)
		assert est.pixels == exp["pixels"]
		assert est.rim_points == exp["rim_points"]


def test_isaac_bands_for_existing_and_new_colours():
	isaac_existing = pc.bands_for({"green": PALETTE["green"].rgb, "pink": PALETTE["pink"].rgb}, engine="isaac")
	assert isaac_existing["green"] == pc.BOWL_HSV["green"]
	assert isaac_existing["pink"] == pc.BOWL_HSV_BY_ENGINE["isaac"]["pink"]
	for colour in ("blue", "yellow", "orange", "purple", "white", "black"):
		with pytest.raises(NotImplementedError):
			pc.bands_for({"green": PALETTE[colour].rgb}, engine="isaac")


COLOUR_SEEDS = [
	("green", 3000009, "pink"),
	("green", 3000010, "pink"),
	("green", 3000013, "extra:0"),
	("pink", 3000002, "pink"),
	("pink", 3000006, "green"),
	("pink", 3000008, "pink"),
	("blue", 3000001, "extra:0"),
	("blue", 3000002, "green"),
	("blue", 3000005, "green"),
	("yellow", 3000001, "pink"),
	("yellow", 3000003, "pink"),
	("yellow", 3000006, "pink"),
	("orange", 3000004, "pink"),
	("orange", 3000023, "green"),
	("orange", 3000033, "extra:0"),
	("purple", 3000004, "extra:0"),
	("purple", 3000007, "pink"),
	("purple", 3000009, "green"),
	("white", 3000003, "green"),
	("white", 3000005, "pink"),
	("white", 3000024, "pink"),
	("black", 3000001, "green"),
	("black", 3000004, "green"),
	("black", 3000007, "green"),
]


@pytest.mark.parametrize(("colour", "seed", "slot"), COLOUR_SEEDS)
def test_locate_bowls_all_eight_colours(colour, seed, slot):
	pytest.importorskip("mujoco")
	pytest.importorskip("cv2")
	layout = sample_case(seed, False, 5, {"scene": "v3", "split": "train"})
	rgb_by_slot = {
		**layout.bowl_rgb,
		**{f"extra:{i}": rgb for i, (_, _, _, rgb) in enumerate(layout.extra_bowls)},
	}
	found = _render_top_and_locate(layout, bands=pc.bands_for(rgb_by_slot, engine="mujoco"))
	est = found[slot]
	assert est is not None, f"{colour} bowl at {slot!r} not found (seed={seed}, light={layout.light_level:.3f})"
	if slot.startswith("extra:"):
		idx = int(slot.split(":", 1)[1])
		truth = np.asarray(layout.extra_bowls[idx][1:3])
	else:
		truth = np.asarray(getattr(layout, slot))
	assert float(np.linalg.norm(est.xy - truth)) <= 0.01


def test_white_bowl_low_light():
	"""Formerly a strict xfail (white vmin=200): under light_level < 0.72 the white bowl renders at V~175."""
	pytest.importorskip("mujoco")
	pytest.importorskip("cv2")
	layout = sample_case(3000015, False, 5, {"scene": "v3", "split": "train"})
	assert layout.bowl_names["green"] == "white"
	assert layout.light_level < 0.70
	rgb_by_slot = {
		**layout.bowl_rgb,
		**{f"extra:{i}": rgb for i, (_, _, _, rgb) in enumerate(layout.extra_bowls)},
	}
	found = _render_top_and_locate(layout, bands=pc.bands_for(rgb_by_slot, engine="mujoco"))
	assert found["green"] is not None
	assert float(np.linalg.norm(found["green"].xy - np.asarray(layout.green))) <= 0.01


READY_Q = np.array([*ExpertParams().ready_q, ExpertParams().open_g])

# Round-1 failure bank (round-0 clutter baseline, seeds 4100101-4100130): every BOWL_NOT_FOUND case.
# (seed, arm pose, slot, colour): white bowls dimmer than V 200, a black bowl whose wall foot was fitted as the rim,
# and yellow bowls merged with the yellow arm when the bowls are re-sensed at the ready pose after recalibration.
ROUND1_BANK = [
	(4100113, "rest", "green", "white"),
	(4100114, "rest", "pink", "white"),
	(4100116, "rest", "pink", "white"),
	(4100126, "rest", "green", "white"),
	(4100128, "rest", "extra:0", "black"),
	(4100104, "ready", "green", "yellow"),
	(4100111, "ready", "green", "black"),
	(4100128, "ready", "green", "yellow"),
	(4100129, "ready", "pink", "black"),
]


@pytest.mark.parametrize(("seed", "pose", "slot", "colour"), ROUND1_BANK)
def test_round1_bowl_bank(seed, pose, slot, colour):
	pytest.importorskip("mujoco")
	pytest.importorskip("cv2")
	layout = sample_case(seed, False, 0, {"scene": "v3", "cameras": "fixed", "split": "train"})
	rgb_by_slot = {**layout.bowl_rgb, **{f"extra:{i}": c for i, (*_, c) in enumerate(layout.extra_bowls)}}
	names = {**layout.bowl_names, **{f"extra:{i}": n for i, (n, *_) in enumerate(layout.extra_bowls)}}
	truth = {
		"green": layout.green,
		"pink": layout.pink,
		**{f"extra:{i}": (x, y) for i, (_, x, y, _) in enumerate(layout.extra_bowls)},
	}
	assert names[slot] == colour
	found = _render_top_and_locate(
		layout, bands=pc.bands_for(rgb_by_slot), q=INITIAL_Q if pose == "rest" else READY_Q, depth_noise_seed=seed
	)
	for name, est in found.items():
		assert est is not None, f"{names[name]} bowl at {name!r} not found"
		assert float(np.linalg.norm(est.xy - np.asarray(truth[name]))) <= 0.006, name


def test_white_band_tracks_v_min_constant():
	band = pc.bands_for({"green": PALETTE["white"].rgb})["green"]
	assert band == (-1, -1, 40, pc.WHITE_V_MIN) and pc.WHITE_V_MIN < 175
