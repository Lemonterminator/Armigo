"""Tests for the MuJoCo cell scene: visual mesh assets replace the primitive geometry, collision stays hidden."""

from __future__ import annotations

import xml.etree.ElementTree as ET

from digital_twin_armigo import mujoco_cell as sim


def _make_layout(brick_size: tuple[float, float, float] = sim.DUPLO_2X2) -> sim.Layout:
	return sim.Layout(
		green=(0.22, 0.16),
		pink=(0.41, 0.20),
		brick_xy=(0.33, 0.00),
		brick_yaw_deg=20.0,
		defective=True,
		brick_rgb=(0.55, 0.80, 0.15),
		brick_size=brick_size,
		clutter=[(0.17, 0.13, 1.2, 0.95, 0.80, 0.10)],
	)


def test_visual_obj_assets_exist_on_disk():
	expected = ("plate_200mm.obj", "duplo_2x2.obj", "duplo_2x3.obj", "duplo_2x4.obj")
	for name in expected:
		path = sim.SIM_ASSETS_DIR / name
		assert path.is_file(), f"Missing visual mesh asset: {path}"
		assert path.stat().st_size > 0


def test_brick_mesh_name_maps_standard_duplo_sizes():
	assert sim.brick_mesh_name(sim.DUPLO_2X2) == "duplo_2x2"
	assert sim.brick_mesh_name(sim.DUPLO_2X3) == "duplo_2x3"
	assert sim.brick_mesh_name(sim.DUPLO_2X4) == "duplo_2x4"
	assert sim.brick_mesh_name((0.05, 0.05, 0.02)) is None


def test_scene_xml_uses_plate_and_brick_visual_meshes_while_hiding_collision():
	xml_text = sim.scene_xml(_make_layout(sim.DUPLO_2X2))
	root = ET.fromstring(xml_text)  # noqa: S314

	meshes = {el.attrib["name"]: el.attrib["file"] for el in root.findall("./asset/mesh")}
	assert {"plate_200mm", "duplo_2x2", "duplo_2x3", "duplo_2x4"} <= set(meshes)

	for bowl_name in ("green_bowl", "pink_bowl"):
		body = root.find(f"./worldbody/body[@name='{bowl_name}']")
		assert body is not None
		geoms = body.findall("geom")
		visual = [g for g in geoms if g.attrib.get("type") == "mesh"]
		collision = [g for g in geoms if g.attrib.get("type") != "mesh"]
		assert len(visual) == 1
		assert visual[0].attrib["mesh"] == "plate_200mm"
		assert visual[0].attrib.get("contype") == "0"
		assert visual[0].attrib.get("conaffinity") == "0"
		assert visual[0].attrib.get("group", "0") == "0"
		assert len(collision) == 65
		assert all(g.attrib.get("group") == "3" for g in collision)

	brick_body = root.find("./worldbody/body[@name='brick']")
	assert brick_body is not None
	b_geoms = brick_body.findall("geom")
	visual_brick = [g for g in b_geoms if g.attrib.get("name") == "brick"]
	col_brick = [g for g in b_geoms if g.attrib.get("name") == "brick_col"]
	defect_geoms = [g for g in b_geoms if g.attrib.get("name", "").startswith("defect_")]
	assert len(visual_brick) == 1
	assert visual_brick[0].attrib["type"] == "mesh"
	assert visual_brick[0].attrib["mesh"] == "duplo_2x2"
	assert visual_brick[0].attrib.get("contype") == "0"
	assert len(col_brick) == 1
	assert col_brick[0].attrib["type"] == "box"
	assert col_brick[0].attrib.get("group") == "3"
	assert len(defect_geoms) == 2

	# Clutter body uses duplo_2x2 visual mesh + group 3 collision box
	clutter_bodies = [b for b in root.findall("./worldbody/body") if "name" not in b.attrib]
	assert len(clutter_bodies) == 1
	c_geoms = clutter_bodies[0].findall("geom")
	assert [g.attrib.get("mesh") for g in c_geoms if g.attrib.get("type") == "mesh"] == ["duplo_2x2"]
	assert [g.attrib.get("group") for g in c_geoms if g.attrib.get("type") == "box"] == ["3"]


def test_scene_xml_selects_duplo_2x4_mesh_when_layout_is_2x4():
	xml_text = sim.scene_xml(_make_layout(sim.DUPLO_2X4))
	root = ET.fromstring(xml_text)  # noqa: S314
	visual_brick = root.find("./worldbody/body[@name='brick']/geom[@name='brick']")
	assert visual_brick is not None
	assert visual_brick.attrib["mesh"] == "duplo_2x4"
