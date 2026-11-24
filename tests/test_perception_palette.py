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

ROOT = Path(__file__).resolve().parents[1]


def _render_top_and_locate(layout, *, bands=None):
	cell = Cell(layout, SO101_MJCF_DIR)
	try:
		cell.set_joints(INITIAL_Q)
		r = cell.renderer
		r.enable_depth_rendering()
		r.update_scene(cell.data, "top")
		depth = r.render().astype(np.float64)
		r.disable_depth_rendering()
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


@pytest.mark.xfail(
	strict=True,
	reason="Task 3 spec fixes white vmin=200; under scene_xml light_level < 0.72 white bowl renders at V~175 <= 200",
)
def test_white_bowl_low_light_known_spec_limit():
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
