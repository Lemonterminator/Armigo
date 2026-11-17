"""Randomised ``cross_var`` defect: sampling, containment, determinism, kind ids and MJCF rendering."""

from __future__ import annotations

import math
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from digital_twin_armigo.constants import DUPLO_2X2, DUPLO_2X4

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def bench():
	pytest.importorskip("mujoco")
	sys.path.insert(0, str(ROOT / "scripts"))
	import bench_physics_expert

	return bench_physics_expert


def _layout(*, face=4, seed=0, defective=True, kind="cross_var", size=DUPLO_2X2):
	from digital_twin_armigo.mujoco_cell.layout import Layout

	return Layout(
		green=(0.2, 0.1),
		pink=(0.2, -0.1),
		brick_xy=(0.2, 0.0),
		brick_yaw_deg=0.0,
		defective=defective,
		brick_size=size,
		defect_face=face,
		defect_kind=kind,
		defect_seed=seed,
	)


def _flat(strokes):
	return np.array([[*centre, ang, half_length, half_width] for centre, ang, half_length, half_width in strokes])


def _half_extents(layout):
	from digital_twin_armigo.presentation import FACES, face_frame

	size = np.asarray(layout.brick_size, dtype=float)
	axis = FACES[layout.defect_face][0]
	_, _, _, v = face_frame(layout.brick_size, layout.defect_face)
	return np.array([size[(axis + 1) % 3], abs(v @ size)]) / 2.0


def _corners(stroke):
	(cu, cv), ang, half_length, half_width = stroke
	d = np.array([math.cos(ang), math.sin(ang)])
	s = np.array([-d[1], d[0]])
	c = np.array([cu, cv])
	return np.array(
		[c + along * half_length * d + across * half_width * s for along in (-1.0, 1.0) for across in (-1.0, 1.0)]
	)


def _intersection(strokes):
	(c0, a0, _, _), (c1, a1, _, _) = strokes
	d0 = np.array([math.cos(a0), math.sin(a0)])
	d1 = np.array([math.cos(a1), math.sin(a1)])
	den = d0[0] * d1[1] - d0[1] * d1[0]
	delta = np.asarray(c1) - np.asarray(c0)
	t = (delta[0] * d1[1] - delta[1] * d1[0]) / den
	return np.asarray(c0) + t * d0


def _shapes_equal(a, b):
	if len(a) != len(b):
		return False
	for x, y in zip(a, b, strict=True):
		if not np.array_equal(x[1], y[1]) or not np.array_equal(x[2], y[2]) or not np.array_equal(x[4], y[4]):
			return False
		if x[0] != y[0] or x[3] != y[3] or x[5] != y[5]:
			return False
	return True


def test_cross_var_strokes_are_deterministic_and_seeded():
	from digital_twin_armigo.mujoco_cell.scene import cross_var_strokes

	base = cross_var_strokes(_layout(face=2, seed=11))
	assert np.array_equal(_flat(base), _flat(cross_var_strokes(_layout(face=2, seed=11))))
	assert not np.array_equal(_flat(base), _flat(cross_var_strokes(_layout(face=2, seed=12))))
	assert len(base) == 2


@pytest.mark.parametrize("size", [DUPLO_2X2, DUPLO_2X4], ids=["2x2", "2x4"])
def test_cross_var_geometry_fits_every_face_with_enough_area(size):
	from digital_twin_armigo.mujoco_cell.scene import _cross_var_sample, cross_var_strokes

	fallbacks = 0
	total = 0
	for face in range(6):
		intersections = []
		for seed in range(200):
			layout = _layout(face=face, seed=seed, size=size)
			strokes = cross_var_strokes(layout)
			assert len(strokes) == 2
			half = _half_extents(layout)
			pts = np.concatenate([_corners(stroke) for stroke in strokes])
			assert np.all(np.abs(pts) <= half + 1e-12)
			span = pts.max(axis=0) - pts.min(axis=0)
			assert float(span[0] * span[1]) >= 0.25 * 4.0 * float(half[0] * half[1]) - 1e-15
			skew_deg = math.degrees(abs(strokes[1][1] - strokes[0][1]) % math.pi)
			assert 65.0 - 1e-9 <= skew_deg <= 115.0 + 1e-9
			intersections.append(_intersection(strokes))
			_, _, fallback = _cross_var_sample(layout)
			fallbacks += int(fallback)
			total += 1
		intersections = np.asarray(intersections)
		assert np.ptp(intersections[:, 0]) > 0.0
		assert np.ptp(intersections[:, 1]) > 0.0
	assert fallbacks / total < 0.05
	warnings.warn(f"cross_var fallback rate for {size}: {fallbacks}/{total} = {fallbacks / total:.4f}", stacklevel=2)


def test_cross_var_fallback_is_deterministic_and_fits():
	from digital_twin_armigo.mujoco_cell.scene import _cross_var_strokes

	for face in range(6):
		half = _half_extents(_layout(face=face))
		a, used_a = _cross_var_strokes(np.random.default_rng(1234), half, attempts=0)
		b, used_b = _cross_var_strokes(np.random.default_rng(1234), half, attempts=0)
		assert used_a and used_b
		assert np.array_equal(_flat(a), _flat(b))
		pts = np.concatenate([_corners(stroke) for stroke in a])
		assert np.all(np.abs(pts) <= half + 1e-12)
		span = pts.max(axis=0) - pts.min(axis=0)
		assert float(span[0] * span[1]) >= 0.25 * 4.0 * float(half[0] * half[1]) - 1e-15
		assert a[0][0] == a[1][0] == (0.0, 0.0)
		assert math.degrees(abs(a[1][1] - a[0][1]) % math.pi) == pytest.approx(90.0)


def test_existing_kind_ids_and_tape_stream_are_unchanged(monkeypatch):
	from digital_twin_armigo.mujoco_cell import scene

	assert scene.DEFECT_KINDS[:5] == ("cross", "tape", "chip", "scuff", "discolor")
	assert scene.DEFECT_KINDS[-1] == "cross_var"
	layout = _layout(face=3, seed=7, kind="tape")
	expected = np.random.default_rng([7 & 0xFFFFFFFF, 3, 1])
	assert np.array_equal(scene._defect_rng(layout).random(32), expected.random(32))
	current = scene._defect_shapes(layout)
	monkeypatch.setattr(scene, "DEFECT_KINDS", ("cross", "tape", "chip", "scuff", "discolor"))
	assert _shapes_equal(current, scene._defect_shapes(layout))


def test_scene_xml_emits_two_cross_var_geoms_in_the_right_group(bench):
	from digital_twin_armigo.mujoco_cell.scene import PROBE_GROUP, scene_xml

	for defective, group in ((True, 0), (False, PROBE_GROUP)):
		xml = scene_xml(_layout(face=4, seed=5, defective=defective), use_visual_meshes=False)
		geoms = re.findall(r'<geom name="defect_\d+"[^>]*/>', xml)
		assert len(geoms) == 2
		assert geoms[0].startswith('<geom name="defect_0"')
		assert geoms[1].startswith('<geom name="defect_1"')
		assert all(f'group="{group}"' in geom for geom in geoms)
		assert all('contype="0" conaffinity="0" mass="0"' in geom for geom in geoms)
		rgba = {re.search(r'rgba="([^"]+)"', geom).group(1) for geom in geoms}
		assert rgba <= {"0.05 0.05 0.08 1", "0.08 0.10 0.30 1", "0.30 0.06 0.06 1"}
