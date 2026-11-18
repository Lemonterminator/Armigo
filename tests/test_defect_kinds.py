"""Defect kinds on the brick (cross, tape, chip, scuff, discolor): MJCF, physics invisibility and pixel visibility."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _original_cross(layout, stud_top_offset: float) -> str:
	"""The defect cross exactly as ``scene._brick`` emitted it before defect kinds existed (reference copy)."""
	from digital_twin_armigo.mujoco_cell.scene import PROBE_GROUP
	from digital_twin_armigo.presentation import FACES, defect_strokes

	cross = ""
	group = 0 if layout.defective else PROBE_GROUP
	lift = 0.0003 + (stud_top_offset if FACES[layout.defect_face] == (2, 1) else 0.0)
	for k, (centre, axis, side, half) in enumerate(defect_strokes(layout.brick_size, layout.defect_face)):
		normal = np.cross(axis, side)
		pos = " ".join(f"{c:.5f}" for c in centre + lift * normal)
		xy = " ".join(f"{c:.5f}" for c in (*axis, *side))
		cross += (
			f'<geom name="defect_{k}" type="box" size="{half:.4f} 0.0015 0.0003" pos="{pos}" xyaxes="{xy}" '
			f'rgba="0.05 0.05 0.08 1" group="{group}" contype="0" conaffinity="0" mass="0"/>'
		)
	return cross


@pytest.fixture(scope="module")
def bench():
	pytest.importorskip("mujoco")
	sys.path.insert(0, str(ROOT / "scripts"))
	import bench_physics_expert

	return bench_physics_expert


def _layout(bench, idx=1, face=4, defective=True, kind="cross", seed=0):
	from digital_twin_armigo.camera import CameraPose

	top = CameraPose(**json.loads((ROOT / "config" / "sim" / "top_camera_joint.json").read_text())["camera"])
	lay = bench.sample_layouts(idx + 1, 7, top)[idx]
	lay.defective = defective
	lay.defect_face = face
	lay.defect_kind = kind
	lay.defect_seed = seed
	return lay


def _defect_xml(xml: str) -> list[str]:
	return re.findall(r'<geom name="defect_\d+"[^>]*/>', xml)


def test_default_layout_keeps_the_original_cross(bench):
	from digital_twin_armigo.mujoco_cell.layout import Layout
	from digital_twin_armigo.mujoco_cell.scene import scene_xml

	assert Layout((0.2, 0.1), (0.2, -0.1), (0.2, 0.0), 0.0, True).defect_kind == "cross"
	for idx, face, defective in ((0, 4, True), (1, 2, False), (2, 5, True), (3, 0, False)):
		lay = _layout(bench, idx=idx, face=face, defective=defective)
		for meshes, offset in ((True, 0.0035), (False, 0.0044)):
			for probes in (False, True):
				xml = scene_xml(lay, use_visual_meshes=meshes, face_probes=probes)
				assert "".join(_defect_xml(xml)) == _original_cross(lay, offset)
				assert _original_cross(lay, offset) in xml


def test_defect_shape_is_deterministic_and_seeded(bench):
	from digital_twin_armigo.mujoco_cell.scene import DEFECT_KINDS, scene_xml

	for kind in DEFECT_KINDS[1:]:
		a = _defect_xml(scene_xml(_layout(bench, kind=kind, seed=3), use_visual_meshes=False))
		b = _defect_xml(scene_xml(_layout(bench, kind=kind, seed=3), use_visual_meshes=False))
		c = _defect_xml(scene_xml(_layout(bench, kind=kind, seed=4), use_visual_meshes=False))
		assert a == b and a
		assert a != c


def test_unknown_kind_is_rejected(bench):
	from digital_twin_armigo.mujoco_cell.scene import scene_xml

	with pytest.raises(ValueError):
		scene_xml(_layout(bench, kind="dent"))


@pytest.mark.parametrize("kind", ["cross", "tape", "chip", "scuff", "discolor"])
def test_each_kind_compiles_is_visual_only_and_visible_on_top(bench, kind):
	from digital_twin_armigo.constants import SO101_MJCF_DIR, TOP_CROP
	from digital_twin_armigo.mujoco_cell import Cell
	from digital_twin_armigo.mujoco_cell._mj import mujoco
	from digital_twin_armigo.mujoco_cell.inspection_views import capture_view
	from digital_twin_armigo.mujoco_cell.scene import PROBE_GROUP, scene_xml
	from digital_twin_armigo.physics import load_physics_spec

	spec = load_physics_spec()
	for seed in range(3):
		px = {}
		for defective in (True, False):
			lay = _layout(bench, face=4, defective=defective, kind=kind, seed=seed)
			with Cell(lay, SO101_MJCF_DIR, physics=spec, assisted_grasp=False, face_probes=True) as cell:
				m = cell.model
				g = cell.defect_geoms
				assert len(g) >= 1
				assert np.all(m.geom_contype[g] == 0) and np.all(m.geom_conaffinity[g] == 0)
				assert np.all(m.geom_bodyid[g] == cell.brick_body)
				assert np.all(m.geom_group[g] == (0 if defective else PROBE_GROUP))
				mujoco.mj_forward(m, cell.data)
				v = capture_view(cell, "top", crop=TOP_CROP, image_size=64)
				rgb = cell.rgb("top")
			px[defective] = (v.defect_px, rgb)
			geoms = _defect_xml(scene_xml(lay, physics=spec, face_probes=True))
			assert len(geoms) == len(g)
			assert all('contype="0" conaffinity="0" mass="0"' in x for x in geoms)
		assert px[True][0] >= 15, f"{kind} seed {seed}: only {px[True][0]} defect pixels on the top face"
		# segmentation still counts the hidden copy on a normal brick, but it never reaches the colour image
		assert px[False][0] >= 15
		assert not np.array_equal(px[True][1], px[False][1])


@pytest.mark.parametrize("kind", ["tape", "chip", "scuff", "discolor"])
def test_hidden_defect_leaves_the_colour_image_unchanged(bench, kind):
	"""A normal brick's colour image is the same whatever kind of hidden defect it carries."""
	from digital_twin_armigo.constants import SO101_MJCF_DIR
	from digital_twin_armigo.mujoco_cell import Cell
	from digital_twin_armigo.mujoco_cell._mj import mujoco

	imgs = []
	for k in ("cross", kind):
		lay = _layout(bench, face=4, defective=False, kind=k, seed=1)
		with Cell(lay, SO101_MJCF_DIR, assisted_grasp=False, face_probes=True) as cell:
			mujoco.mj_forward(cell.model, cell.data)
			imgs.append(cell.rgb("top").copy())
	assert np.array_equal(imgs[0], imgs[1])
