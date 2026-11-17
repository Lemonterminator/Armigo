"""Unit tests for bowl convex-decomposition gates and candidate MJCF builders."""

from __future__ import annotations

import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

mujoco = pytest.importorskip("mujoco")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
	sys.path.insert(0, str(ROOT / "scripts"))

from digital_twin_armigo.constants import BOWL_R


def test_g1_flags_the_current_cylinder_floor():
	from bowl_experiments import candidates, gates

	res = gates.g1_collider_micro(candidates.model_builder("R0_current"))
	assert not res.passed
	assert res.detail["worst_dist_mm"] < -50.0  # the 4.0.1 box-vs-cylinder depth bug


def test_g1_passes_and_g0_flags_slab_box_200():
	from bowl_experiments import candidates, gates

	builder = candidates.model_builder("R9_slab_box_200")
	g1 = gates.g1_collider_micro(builder)
	assert g1.passed, g1.detail
	g0 = gates.g0_geometry(builder)
	assert not g0.passed
	assert g0.detail["protrusion_mm"] > 5.0


def test_floor_strips_cover_the_disc_without_gaps():
	from bowl_experiments import candidates

	strips = candidates.strip_params(n=24)
	area = sum(2.0 * (candidates.STRIP_W / 2.0) * 2.0 * hl for _, hl in strips)
	assert abs(area - math.pi * BOWL_R**2) / (math.pi * BOWL_R**2) < 0.01
	assert max(math.hypot(abs(x) + candidates.STRIP_W / 2.0, hl) for x, hl in strips) - BOWL_R < 0.005
	xml = f"<worldbody><body>{candidates.floor_strips((0.0, 0.0), '1 1 1 1', '', '')}</body></worldbody>"
	geoms = ET.fromstring(xml).findall("./body/geom")
	assert len(geoms) == 24
	for g in geoms:
		pos_z = float(g.attrib["pos"].split()[2])
		half_z = float(g.attrib["size"].split()[2])
		assert abs((pos_z + half_z) - 0.003) < 1e-6


def test_wedge_mesh_xml_builds_convex_wedges():
	from bowl_experiments import candidates

	sample_polys = [[(0.062, 0.003), (0.097, 0.022), (0.097, 0.015), (0.065, 0.000)]]
	assets_xml, names = candidates.wedge_mesh_xml(sample_polys, n_wedges=8, prefix="test_w")
	assert len(names) == 8
	xml = (
		f"<mujoco><asset>{assets_xml}</asset><worldbody>"
		+ "".join(f'<geom type="mesh" mesh="{n}"/>' for n in names)
		+ "</worldbody></mujoco>"
	)
	m = mujoco.MjModel.from_xml_string(xml)
	assert m.nmesh == 8


def test_hertel_mehlhorn_polygons_are_convex_and_preserve_area():
	import json

	from bowl_experiments import decompose_plate

	profile = json.loads((ROOT / "results" / "bowl-experiments" / "plate_profile.json").read_text())["profile"]
	loop = [(float(r), float(z)) for r, z in profile]
	polys = decompose_plate.hertel_mehlhorn(loop)
	assert len(polys) >= 2
	for poly in polys:
		assert decompose_plate.is_convex_ccw(poly)
	orig_area = abs(decompose_plate.polygon_area(loop))
	decomp_area = sum(abs(decompose_plate.polygon_area(p)) for p in polys)
	assert abs(decomp_area - orig_area) / orig_area < 0.001
