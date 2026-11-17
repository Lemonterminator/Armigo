"""Smoke test of the anomaly crop dataset renderer (scripts/anomaly/render_dataset.py) on one layout."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def rd():
	pytest.importorskip("mujoco")
	pytest.importorskip("cv2")
	sys.path.insert(0, str(ROOT / "scripts"))
	sys.path.insert(0, str(ROOT / "scripts" / "anomaly"))
	import render_dataset

	return render_dataset


def test_layout_sampling_is_deterministic_and_covers_kinds(rd):
	from digital_twin_armigo.mujoco_cell.scene import DEFECT_KINDS

	a = rd.sample_dataset_layouts(40, 5)
	b = rd.sample_dataset_layouts(40, 5)
	assert [(x.defect_kind, x.defect_face, x.defective, x.defect_seed, s) for x, s in a] == [
		(x.defect_kind, x.defect_face, x.defective, x.defect_seed, s) for x, s in b
	]
	assert {x.defect_kind for x, _ in a} == set(DEFECT_KINDS)
	assert {s for _, s in a} == {"train", "test"}


def test_layout_sampling_cross_var(rd):
	cross = rd.sample_dataset_layouts(40, 5, task="cross_var")
	assert cross == rd.sample_dataset_layouts(40, 5, task="cross_var")  # deterministic
	assert {x.defect_kind for x, _ in cross} == {"cross_var"}
	assert all(0 <= x.defect_face <= 4 for x, _ in cross)  # the bottom face is never used
	assert {s for _, s in cross} == {"train", "test"}

	def key(lay, split):
		return lay.defect_kind, lay.defect_face, lay.defective, lay.defect_seed, split

	implicit = rd.sample_dataset_layouts(40, 5)
	explicit = rd.sample_dataset_layouts(40, 5, task="mixed")
	assert [key(x, s) for x, s in explicit] == [key(x, s) for x, s in implicit]


def test_square_crop_clamp_has_no_black_border():
	from digital_twin_armigo.mujoco_cell.inspection_views import square_crop

	img = np.full((100, 200, 3), 255, np.uint8)
	bbox = (190, 40, 20, 20)  # the padded 40 px window runs past the right edge
	assert (square_crop(img, bbox, 0.5, 40) == 0).any()
	clamped = square_crop(img, bbox, 0.5, 40, clamp=True)
	assert clamped.shape == (40, 40, 3) and (clamped == 255).all()


def test_square_crop_clamp_is_a_noop_when_already_inside():
	from digital_twin_armigo.mujoco_cell.inspection_views import square_crop

	img = np.arange(100 * 200 * 3, dtype=np.uint8).reshape(100, 200, 3)
	bbox = (80, 40, 20, 20)
	assert np.array_equal(square_crop(img, bbox, 0.5, 40), square_crop(img, bbox, 0.5, 40, clamp=True))


def test_square_crop_clamp_falls_back_when_the_window_does_not_fit():
	from digital_twin_armigo.mujoco_cell.inspection_views import square_crop

	img = np.full((100, 200, 3), 255, np.uint8)
	bbox = (0, 0, 300, 300)  # padded side 600 px: larger than both image dimensions
	assert np.array_equal(square_crop(img, bbox, 0.5, 40), square_crop(img, bbox, 0.5, 40, clamp=True))


def test_one_layout_smoke(rd, tmp_path):
	import cv2

	# a seed whose single layout lands in the test split (train layouts of a defective brick are not rendered)
	seed = next(s for s in range(100) if rd.sample_dataset_layouts(1, s, p_defective=1.0)[0][1] == "test")
	summary = rd.build(1, seed, tmp_path, k=4, image_size=64, workers=1, p_defective=1.0)
	layouts = [json.loads(line) for line in (tmp_path / "layouts.jsonl").read_text().splitlines()]
	rows = [json.loads(line) for line in (tmp_path / "metadata.jsonl").read_text().splitlines()]
	assert len(layouts) == 1 and summary["n_layouts"] == 1
	assert layouts[0]["status"] in ("ok", "no_show_hold")
	assert summary["n_crops"] == len(rows) == layouts[0]["n_crops"]
	if layouts[0]["status"] != "ok":
		pytest.skip(f"the expert did not reach a show hold on this layout ({layouts[0]})")
	assert rows, "a layout that reached its show hold should yield crops"
	assert any(not r["virtual"] for r in rows)  # the real show-hold crops are saved too
	keys = {"path", "layout_index", "split", "camera", "flex", "roll_deg", "virtual", "defective", "defect_kind"}
	keys |= {"defect_face", "defect_px", "faces_px", "brick_px", "label"}
	for r in rows:
		assert keys <= set(r)
		assert r["split"] == "test" and r["layout_index"] == 0 and r["defective"]
		assert r["camera"] in rd.CAMERAS
		assert r["brick_px"] >= rd.MIN_BRICK_PX
		assert len(r["faces_px"]) == 6
		assert r["label"] == ("abnormal" if r["defect_px"] >= rd.MIN_DEFECT_PX else "hidden_defect")
		img = cv2.imread(str(tmp_path / r["path"]))
		assert img is not None and img.shape == (64, 64, 3)
	assert len({r["path"] for r in rows}) == len(rows)
