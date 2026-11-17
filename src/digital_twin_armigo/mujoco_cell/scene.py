"""MJCF generation for one episode of the sorting cell: bowls, brick with defect cross, clutter, markers, cameras."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..constants import (
	BOWL_FLOOR_TOP_M,
	CELL_MARKERS,
	DUPLO_2X2,
	FPS,
	SIM_ASSETS_DIR,
	STUD_PITCH,
	TOP_FULL,
)
from ..physics import PhysicsSpec, brick_spec_for_size
from ..presentation import FACES, defect_strokes
from .layout import Layout

SUBSTEPS = 16  # physics steps per dataset frame
PROBE_GROUP = 5  # hidden geom group carrying the defect cross of a normal brick


VISUAL_MESH_NAMES = ("plate_200mm", "duplo_2x2", "duplo_2x3", "duplo_2x4")

_BRICK_MESH_MAP = {
	(2, 2): "duplo_2x2",
	(3, 2): "duplo_2x3",
	(4, 2): "duplo_2x4",
}


def brick_mesh_name(size: tuple[float, float, float]) -> str | None:
	"""Return the visual mesh name matching ``size``, or ``None`` for non-standard sizes."""
	nx = round(size[0] / STUD_PITCH)
	ny = round(size[1] / STUD_PITCH)
	expected_w = round(nx * 0.0159, 4)
	expected_h = round(ny * 0.0159, 4)
	if abs(size[0] - expected_w) > 0.001 or abs(size[1] - expected_h) > 0.001:
		return None
	return _BRICK_MESH_MAP.get((nx, ny))


BOWL_COLLISION_ID = "R6_box_floor_wedge_walls"
_BOWL_WALL_N_WEDGES = 48
_BOWL_WALL_EPS_A = 1e-4
_BOWL_FLOOR_HALF_XY = 0.070
_BOWL_FLOOR_HALF_Z = 0.00125
# 2D Hertel-Mehlhorn convex decomposition of the plate_200mm.obj wall cross-section (r >= 62 mm), (r_m, z_m).
_BOWL_WALL_POLYS: tuple[tuple[tuple[float, float], ...], ...] = (
	((0.062, 0.003), (0.0625, 0.0005), (0.065, 0.0), (0.073, 0.0), (0.0755, 0.0007), (0.081, 0.0028), (0.0695, 0.0034)),
	(
		(0.081, 0.0028),
		(0.087, 0.0064),
		(0.0928, 0.011),
		(0.0975, 0.0155),
		(0.0998, 0.0182),
		(0.1, 0.0201),
		(0.0989, 0.0216),
		(0.091, 0.0145),
		(0.0865, 0.0102),
	),
	((0.0989, 0.0216), (0.0972, 0.022), (0.0956, 0.0211), (0.0942, 0.0188)),
	((0.0989, 0.0216), (0.0942, 0.0188), (0.091, 0.0145)),
	((0.081, 0.0028), (0.0865, 0.0102), (0.0815, 0.0068)),
	((0.081, 0.0028), (0.0815, 0.0068), (0.076, 0.0046)),
	((0.081, 0.0028), (0.076, 0.0046), (0.0695, 0.0034)),
)


def _bowl_wall_mesh_names() -> list[str]:
	return [f"bowl_wall_{p_idx}_{k}" for p_idx in range(len(_BOWL_WALL_POLYS)) for k in range(_BOWL_WALL_N_WEDGES)]


def _bowl_collision_mesh_assets() -> str:
	"""Emit inline convex-wedge ``<mesh>`` assets for the revolved ``plate_200mm.obj`` wall profile."""
	out: list[str] = []
	for p_idx, poly in enumerate(_BOWL_WALL_POLYS):
		for k in range(_BOWL_WALL_N_WEDGES):
			a0 = 2.0 * math.pi * k / _BOWL_WALL_N_WEDGES - _BOWL_WALL_EPS_A
			a1 = 2.0 * math.pi * (k + 1) / _BOWL_WALL_N_WEDGES + _BOWL_WALL_EPS_A
			c0, s0 = math.cos(a0), math.sin(a0)
			c1, s1 = math.cos(a1), math.sin(a1)
			verts: list[str] = []
			for r, z in poly:
				verts.append(f"{r * c0:.6f} {r * s0:.6f} {z:.6f}")
				if r > 1e-7:
					verts.append(f"{r * c1:.6f} {r * s1:.6f} {z:.6f}")
			out.append(f'<mesh name="bowl_wall_{p_idx}_{k}" vertex="{" ".join(verts)}"/>')
	return "".join(out)


def _visual_mesh_assets(assets_dir: Path | None = None) -> str:
	"""Emit MJCF ``<mesh>`` declarations for the bowl and Duplo brick OBJ assets."""
	root = assets_dir or SIM_ASSETS_DIR
	return "".join(f'<mesh name="{name}" file="{(root / f"{name}.obj").as_posix()}"/>' for name in VISUAL_MESH_NAMES)


def _bowl(name: str, xy: tuple[float, float], rgba: str, use_visual_mesh: bool = True, torsional: bool = False) -> str:
	group = ' group="3"' if use_visual_mesh else ""
	condim = ' condim="4"' if torsional else ""
	bottom_z = BOWL_FLOOR_TOP_M - _BOWL_FLOOR_HALF_Z
	bottom = (
		f'<geom type="box" size="{_BOWL_FLOOR_HALF_XY:.3f} {_BOWL_FLOOR_HALF_XY:.3f} {_BOWL_FLOOR_HALF_Z:.5f}" '
		f'pos="{xy[0]:.4f} {xy[1]:.4f} {bottom_z:.5f}" rgba="{rgba}"{condim}{group}/>'
	)
	walls = "".join(
		f'<geom type="mesh" mesh="{mname}" pos="{xy[0]:.4f} {xy[1]:.4f} 0" rgba="{rgba}"{group}/>'
		for mname in _bowl_wall_mesh_names()
	)
	visual = (
		f'<geom type="mesh" mesh="plate_200mm" pos="{xy[0]:.4f} {xy[1]:.4f} 0" '
		f'contype="0" conaffinity="0" mass="0" rgba="{rgba}"/>'
		if use_visual_mesh
		else ""
	)
	return f'<body name="{name}">{visual}{bottom}{walls}</body>'


def _studs(size: tuple[float, float, float], rgba: str) -> str:
	"""Duplo studs on top of a brick of ``size`` (full extents), as visual-only cylinders."""
	nx, ny = max(1, round(size[0] / STUD_PITCH)), max(1, round(size[1] / STUD_PITCH))
	out = []
	for i in range(nx):
		for j in range(ny):
			x, y = (i - (nx - 1) / 2) * STUD_PITCH, (j - (ny - 1) / 2) * STUD_PITCH
			out.append(
				f'<geom type="cylinder" size="0.0047 0.0022" pos="{x:.4f} {y:.4f} {size[2] / 2 + 0.0022:.4f}" '
				f'rgba="{rgba}" contype="0" conaffinity="0" mass="0"/>'
			)
	return "".join(out)


def _brick(
	layout: Layout,
	use_visual_mesh: bool = True,
	physics: PhysicsSpec | None = None,
	mass_scale: float = 1.0,
	friction_scale: float = 1.0,
) -> str:
	sx, sy, sz = (v / 2 for v in layout.brick_size)
	rgba = " ".join(f"{c:.3f}" for c in layout.brick_rgb) + " 1"
	mesh_name = brick_mesh_name(layout.brick_size) if use_visual_mesh else None
	if physics is not None:
		bspec = brick_spec_for_size(layout.brick_size, physics).scaled(mass_scale, friction_scale)
		mass_kg = bspec.mass_kg
		mu_slide = bspec.dynamic_friction
		solref = "0.009 1"
	else:
		mass_kg = 0.008 * mass_scale
		mu_slide = 1.5 * friction_scale
		solref = "0.004 1"
	# The defect is a dark cross on one face. A normal brick carries the same cross in the hidden geom group 5, so the
	# presentation planner can score views of that face without it appearing in any rendered image.
	cross = ""
	group = 0 if layout.defective else PROBE_GROUP
	stud_top_offset = 0.0035 if mesh_name is not None else 0.0044
	lift = 0.0003 + (stud_top_offset if FACES[layout.defect_face] == (2, 1) else 0.0)
	for k, (centre, axis, side, half) in enumerate(defect_strokes(layout.brick_size, layout.defect_face)):
		normal = np.cross(axis, side)
		pos = " ".join(f"{c:.5f}" for c in centre + lift * normal)
		xy = " ".join(f"{c:.5f}" for c in (*axis, *side))
		cross += (
			f'<geom name="defect_{k}" type="box" size="{half:.4f} 0.0015 0.0003" pos="{pos}" xyaxes="{xy}" '
			f'rgba="0.05 0.05 0.08 1" group="{group}" contype="0" conaffinity="0" mass="0"/>'
		)
	yaw = math.radians(layout.brick_yaw_deg)
	if mesh_name is not None:
		brick_geoms = (
			f'<geom name="brick" type="mesh" mesh="{mesh_name}" contype="0" conaffinity="0" mass="0" rgba="{rgba}"/>'
			f'<geom name="brick_col" type="box" size="{sx:.4f} {sy:.4f} {sz:.4f}" mass="{mass_kg:.5f}" '
			f'friction="{mu_slide:.4f} 0.02 0.002" rgba="{rgba}" solref="{solref}" group="3"/>'
		)
	else:
		studs = _studs(layout.brick_size, rgba)
		brick_geoms = (
			f'<geom name="brick" type="box" size="{sx:.4f} {sy:.4f} {sz:.4f}" mass="{mass_kg:.5f}" '
			f'friction="{mu_slide:.4f} 0.02 0.002" rgba="{rgba}" solref="{solref}"/>{studs}'
		)
	return (
		f'<body name="brick" pos="{layout.brick_xy[0]:.4f} {layout.brick_xy[1]:.4f} {sz + 0.0005:.4f}" '
		f'euler="0 0 {yaw:.4f}"><freejoint name="brick"/>'
		f"{brick_geoms}{cross}</body>"
	)


def _clutter_brick(
	x: float,
	y: float,
	yaw: float,
	r: float,
	g: float,
	b: float,
	use_visual_mesh: bool = True,
	dynamic: bool = False,
	mass_kg: float = 0.007,
	mu_slide: float = 0.35,
	idx: int = 0,
) -> str:
	"""Emit one clutter Duplo 2x2 brick body (mesh visual + group-3 collision box)."""
	rgba = f"{r:.2f} {g:.2f} {b:.2f} 1"
	mass_attr = f' mass="{mass_kg:.5f}" friction="{mu_slide:.4f} 0.02 0.002"' if dynamic else ""
	if use_visual_mesh:
		inner = (
			f'<geom type="mesh" mesh="duplo_2x2" contype="0" conaffinity="0" mass="0" rgba="{rgba}"/>'
			f'<geom type="box" size="{DUPLO_2X2[0] / 2:.4f} {DUPLO_2X2[1] / 2:.4f} {DUPLO_2X2[2] / 2:.4f}" '
			f'rgba="{rgba}"{mass_attr} group="3"/>'
		)
	else:
		inner = (
			f'<geom type="box" size="{DUPLO_2X2[0] / 2:.4f} {DUPLO_2X2[1] / 2:.4f} {DUPLO_2X2[2] / 2:.4f}" '
			f'rgba="{rgba}"{mass_attr}/>{_studs(DUPLO_2X2, rgba)}'
		)
	if dynamic:
		return (
			f'<body name="clutter_{idx}" pos="{x:.4f} {y:.4f} {DUPLO_2X2[2] / 2 + 0.004:.4f}" '
			f'euler="0 0 {yaw:.3f}"><freejoint name="clutter_j_{idx}"/>{inner}</body>'
		)
	return f'<body pos="{x:.4f} {y:.4f} {DUPLO_2X2[2] / 2 + 0.003:.4f}" euler="0 0 {yaw:.3f}">{inner}</body>'


def scene_xml(
	layout: Layout,
	use_visual_meshes: bool = True,
	physics: PhysicsSpec | None = None,
	dynamic_clutter: bool = False,
	mass_scale: float = 1.0,
	friction_scale: float = 1.0,
) -> str:
	"""MJCF for one episode; include it next to Menagerie's ``so101.xml``."""
	table = " ".join(f"{c:.3f}" for c in layout.table_rgb)
	tile2 = " ".join(f"{0.97 * c:.3f}" for c in layout.table_rgb)
	grout = " ".join(f"{0.72 * c:.3f}" for c in layout.table_rgb)
	ld = " ".join(f"{c:.3f}" for c in layout.light_dir)
	lv = layout.light_level
	distractors = "".join(
		f'<geom type="box" size="{sx:.3f} {sy:.3f} {sz:.3f}" pos="{x:.3f} {y:.3f} {sz:.3f}" '
		f'rgba="{0.2 + 0.6 * ((x * 37) % 1):.2f} {0.2 + 0.6 * ((y * 53) % 1):.2f} 0.5 1"/>'
		for x, y, sx, sy, sz, _ in layout.distractors
	)
	clutter_spec = physics.bricks["2x2"].scaled(mass_scale, friction_scale) if physics is not None else None
	clutter = "".join(
		_clutter_brick(
			x,
			y,
			yaw,
			r,
			g,
			b,
			use_visual_mesh=use_visual_meshes,
			dynamic=dynamic_clutter,
			mass_kg=clutter_spec.mass_kg if clutter_spec else 0.007,
			mu_slide=clutter_spec.dynamic_friction if clutter_spec else 0.35,
			idx=k,
		)
		for k, (x, y, yaw, r, g, b) in enumerate(layout.clutter)
	)
	mesh_assets = (_visual_mesh_assets() if use_visual_meshes else "") + _bowl_collision_mesh_assets()
	marker_assets = "".join(
		f'<texture name="m{i}" type="2d" file="_marker_{i}.png"/><material name="m{i}" texture="m{i}" specular="0"/>'
		for i in sorted({m[0] for m in layout.markers})
	)
	marker_geoms = "".join(
		f'<geom type="box" size="{s / 2 * 1.25:.4f} {s / 2 * 1.25:.4f} 0.0004" pos="{x:.4f} {y:.4f} {z + 0.0004:.4f}" '
		f'euler="0 0 {yaw:.3f}" material="m{i}" contype="0" conaffinity="0"/>'
		for i, x, y, z, yaw in layout.markers
		for s in (CELL_MARKERS[i],)
	)
	cams = ""
	if layout.top:
		cams += layout.top.mjcf("top", TOP_FULL)
	if layout.front:
		cams += layout.front.mjcf("front", (640, 480))
	return f"""<mujoco model="sorting_cell">
  <include file="so101.xml"/>
  <option timestep="{1.0 / FPS / SUBSTEPS:.6f}" cone="elliptic" impratio="10"/>
  <visual><headlight diffuse="{0.3 * lv:.2f} {0.3 * lv:.2f} {0.3 * lv:.2f}" ambient="{0.35 * lv:.2f} {0.35 * lv:.2f} {0.35 * lv:.2f}" specular="0 0 0"/>
    <global offwidth="640" offheight="480"/><quality shadowsize="1024"/></visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.75 0.75 0.78" rgb2="0.45 0.45 0.5" width="256" height="1536"/>
    <texture name="table" type="2d" builtin="checker" rgb1="{table}" rgb2="{tile2}" mark="edge" markrgb="{grout}" width="512" height="512"/>
    <material name="table" texture="table" texuniform="true" texrepeat="{1 / (2 * layout.tile_m):.3f} {1 / (2 * layout.tile_m):.3f}" reflectance="0"/>
    {mesh_assets}
    {marker_assets}
  </asset>
  <worldbody>
    <light pos="0.2 0 1.2" dir="{ld}" directional="true" diffuse="{0.6 * lv:.2f} {0.6 * lv:.2f} {0.6 * lv:.2f}" castshadow="true"/>
    <geom name="table" type="plane" size="3 3 0.05" material="table"/>
    {_bowl("green_bowl", layout.green, "0.25 0.65 0.3 1", use_visual_mesh=use_visual_meshes, torsional=physics is not None)}
    {_bowl("pink_bowl", layout.pink, "0.95 0.55 0.72 1", use_visual_mesh=use_visual_meshes, torsional=physics is not None)}
    {_brick(layout, use_visual_mesh=use_visual_meshes, physics=physics, mass_scale=mass_scale, friction_scale=friction_scale)}
    {distractors}
    {clutter}
    {marker_geoms}
    {cams}
  </worldbody>
</mujoco>
"""


def write_marker_textures(directory: Path, ids: set[int]) -> None:
	"""ArUco 4x4_50 images (black square plus a white margin of a quarter side) for the marker geoms."""
	import cv2

	dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
	for i in ids:
		path = directory / f"_marker_{i}.png"
		if not path.exists():
			img = cv2.aruco.generateImageMarker(dictionary, i, 192)
			cv2.imwrite(str(path), cv2.copyMakeBorder(img, 24, 24, 24, 24, cv2.BORDER_CONSTANT, value=255))
