"""Build a realistic 20 cm plastic dining/sorting plate mesh and showcase with random colors.

Reuses the color palette, sRGB->linear conversion, ABS plastic BSDF shader, and random
coloring pipeline from ``../lego_bricks/lego_random_color.py``.

Dimensions match the SO-101 sorting cell bowls in ``scripts/sim/so101_sort_sim.py``:
- Outer diameter: 200 mm (radius 0.100 m)
- Total height:   22 mm  (0.022 m)
- Inner floor z:   3 mm  (0.003 m)
"""

from __future__ import annotations

import argparse
import itertools
import math
import random
import sys
from pathlib import Path

import bmesh
import bpy
import mathutils

# Import random coloring helpers from the LEGO script without duplicating shader logic
_THIS_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path(
	r"c:\Users\Jiang\.gemini\antigravity\scratch\hackjunction\artifacts\plate_bowl"
)
_LEGO_SCRIPT_DIR = _THIS_DIR.parent / "lego_bricks"
if not _LEGO_SCRIPT_DIR.is_dir():
	_LEGO_SCRIPT_DIR = Path(
		r"c:\Users\Jiang\.gemini\antigravity\scratch\hackjunction\artifacts\lego_bricks"
	)
if str(_LEGO_SCRIPT_DIR) not in sys.path:
	sys.path.insert(0, str(_LEGO_SCRIPT_DIR))

import lego_random_color as lrc

# ---------------------------------------------------------------------------
# User-tweakable settings when running directly inside Blender's Text Editor
# ---------------------------------------------------------------------------
RECOLOR_ONLY = False
DEFAULT_MODE = "palette"
DEFAULT_SEED: int | None = None
DEFAULT_PLATE_COUNT = 6
INCLUDE_SAMPLE_BRICKS = True

PLATES_COLLECTION = "Plates_and_Bowls"
BRICKS_COLLECTION = "Sample_Bricks"
STUDIO_COLLECTION = "Studio_Setup"

# Cross-section profile (radius_m, height_m) in CCW order for a revolved plate/shallow bowl.
# Outer radius = 0.100 m (20 cm diameter), total height = 0.022 m, inner floor = 0.003 m.
PLATE_PROFILE_RZ: tuple[tuple[float, float], ...] = (
	# Inner top center -> flat inner floor
	(0.0000, 0.0030),
	(0.0320, 0.0030),
	(0.0620, 0.0030),
	# Smooth concave inner corner -> flared wall
	(0.0695, 0.0034),
	(0.0760, 0.0046),
	(0.0815, 0.0068),
	(0.0865, 0.0102),
	(0.0910, 0.0145),
	(0.0942, 0.0188),
	# Rounded rolled top lip
	(0.0956, 0.0211),
	(0.0972, 0.0220),
	(0.0989, 0.0216),
	(0.1000, 0.0201),
	(0.0998, 0.0182),
	# Outer flared underside
	(0.0975, 0.0155),
	(0.0928, 0.0110),
	(0.0870, 0.0064),
	(0.0810, 0.0028),
	# Foot ring resting flat on z = 0
	(0.0755, 0.0007),
	(0.0730, 0.0000),
	(0.0650, 0.0000),
	(0.0625, 0.0005),
	# Recessed underside back to center
	(0.0560, 0.0008),
	(0.0280, 0.0008),
	(0.0000, 0.0008),
)


def create_plate_mesh(name: str = "Plate_200mm_Mesh", steps: int = 96) -> bpy.types.Mesh:
	"""Revolve ``PLATE_PROFILE_RZ`` 360 degrees around Z into a clean closed manifold mesh."""
	bm = bmesh.new()
	profile_verts = [bm.verts.new((r, 0.0, z)) for r, z in PLATE_PROFILE_RZ]
	for a, b in itertools.pairwise(profile_verts):
		bm.edges.new((a, b))

	center_top = profile_verts[0]
	center_bot = profile_verts[-1]

	bmesh.ops.spin(
		bm,
		geom=list(bm.verts) + list(bm.edges),
		cent=(0.0, 0.0, 0.0),
		axis=(0.0, 0.0, 1.0),
		angle=2.0 * math.pi,
		steps=steps,
		use_merge=True,
		use_normal_flip=False,
	)

	# Remove spin duplicate vertices on the Z axis and close center poles cleanly
	bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
	bm.verts.ensure_lookup_table()
	bmesh.ops.recalc_face_normals(bm, faces=bm.faces)

	# Guard tiny zero-area poles if any remain
	_ = (center_top, center_bot)

	mesh = bpy.data.meshes.new(name)
	bm.to_mesh(mesh)
	bm.free()

	for poly in mesh.polygons:
		poly.use_smooth = True
	mesh.update()
	return mesh


def export_plate_assets(obj: bpy.types.Object, export_dir: Path) -> tuple[Path, Path]:
	"""Export the standalone 20 cm plate mesh to OBJ and STL for reuse or MuJoCo import."""
	export_dir.mkdir(parents=True, exist_ok=True)
	obj_path = export_dir / "plate_200mm.obj"
	stl_path = export_dir / "plate_200mm.stl"

	bpy.ops.object.select_all(action="DESELECT")
	obj.select_set(True)
	bpy.context.view_layer.objects.active = obj

	bpy.ops.wm.obj_export(
		filepath=str(obj_path),
		export_selected_objects=True,
		export_materials=False,
		forward_axis="Y",
		up_axis="Z",
	)
	bpy.ops.wm.stl_export(
		filepath=str(stl_path),
		export_selected_objects=True,
	)
	return obj_path, stl_path


def apply_plate_plastic_finish(objects: list[bpy.types.Object]) -> None:
	"""Tweak the shared ABS/polypropylene shader subtly for semi-matte tableware feel."""
	for obj in objects:
		if not obj.data.materials:
			continue
		mat = obj.data.materials[0]
		bsdf = mat.node_tree.nodes.get("Principled BSDF") if mat.node_tree else None
		if bsdf is not None:
			bsdf.inputs["Roughness"].default_value = 0.24
			bsdf.inputs["Coat Weight"].default_value = 0.18
			bsdf.inputs["Coat Roughness"].default_value = 0.08


def add_studio_lighting_and_camera(center_x: float, center_y: float, span_x: float, span_y: float) -> None:
	"""Create soft studio ground, 3-point area lights, and showcase camera."""
	studio_col = lrc.ensure_collection(STUDIO_COLLECTION)

	bpy.ops.mesh.primitive_plane_add(size=3.0, location=(center_x, center_y, 0.0))
	floor = bpy.context.active_object
	floor.name = "Studio_Floor"
	lrc.move_to_collection(floor, studio_col)
	floor_mat = bpy.data.materials.new("Studio_Floor_Mat")
	floor_mat.use_nodes = True
	floor_bsdf = floor_mat.node_tree.nodes.get("Principled BSDF")
	floor_bsdf.inputs["Base Color"].default_value = (0.14, 0.15, 0.17, 1.0)
	floor_bsdf.inputs["Roughness"].default_value = 0.72
	floor_mat.diffuse_color = (0.14, 0.15, 0.17, 1.0)
	floor.data.materials.append(floor_mat)

	world = bpy.data.worlds.new("Studio_World")
	bpy.context.scene.world = world
	world.use_nodes = True
	bg = world.node_tree.nodes.get("Background")
	if bg is not None:
		bg.inputs["Color"].default_value = (0.18, 0.20, 0.23, 1.0)
		bg.inputs["Strength"].default_value = 0.35

	def add_area_light(name: str, loc: tuple[float, float, float], energy: float, size: float, color: tuple[float, float, float]) -> None:
		ld = bpy.data.lights.new(name=name, type="AREA")
		ld.energy = energy
		ld.size = size
		ld.color = color
		lo = bpy.data.objects.new(name=name, object_data=ld)
		lo.location = loc
		lo.rotation_euler = (mathutils.Vector((center_x, center_y, 0.015)) - lo.location).to_track_quat("-Z", "Y").to_euler()
		studio_col.objects.link(lo)

	add_area_light("Key_Light", (center_x + 0.65, center_y - 0.70, 0.90), energy=14.0, size=0.65, color=(1.0, 0.98, 0.96))
	add_area_light("Fill_Light", (center_x - 0.80, center_y - 0.45, 0.65), energy=5.5, size=0.90, color=(0.88, 0.94, 1.0))
	add_area_light("Rim_Light", (center_x - 0.05, center_y + 0.85, 0.68), energy=9.0, size=0.60, color=(1.0, 1.0, 1.0))

	cam_data = bpy.data.cameras.new("Camera_Showcase")
	cam_data.lens = 50.0
	cam_data.clip_start = 0.001
	cam_data.clip_end = 50.0
	cam_obj = bpy.data.objects.new("Camera_Showcase", cam_data)
	dist = max(span_x, span_y) * 1.38
	cam_obj.location = (center_x + dist * 0.28, center_y - dist * 0.88, dist * 0.62)
	target = mathutils.Vector((center_x, center_y, 0.012))
	cam_obj.rotation_euler = (target - cam_obj.location).to_track_quat("-Z", "Y").to_euler()
	studio_col.objects.link(cam_obj)
	bpy.context.scene.camera = cam_obj


def add_sample_bricks_in_plates(plates: list[bpy.types.Object], rng: random.Random, mode: str, seed: int | None) -> list[bpy.types.Object]:
	"""Optionally drop 2x2, 2x3, and 2x4 LEGO bricks (2x Duplo scale) into the front plates for scale reference."""
	try:
		models_dir = lrc.resolve_models_dir()
	except FileNotFoundError:
		return []

	bricks_col = lrc.ensure_collection(BRICKS_COLLECTION)
	bricks: list[bpy.types.Object] = []
	specs = lrc.BRICK_SPECS
	for idx, plate in enumerate(plates[: len(specs)]):
		label, stl_name = specs[idx]
		brick = lrc.import_and_normalize_brick(models_dir / stl_name, f"Sample_{label}", scale=0.002)
		lrc.move_to_collection(brick, bricks_col)
		# Place flat on the plate's 3 mm inner floor
		brick.location = (
			plate.location.x + rng.uniform(-0.015, 0.015),
			plate.location.y + rng.uniform(-0.015, 0.015),
			0.0030,
		)
		brick.rotation_euler = (0.0, 0.0, math.radians(rng.uniform(-35.0, 35.0)))
		bricks.append(brick)

	if bricks:
		brick_seed = None if seed is None else seed + 101
		lrc.randomize_brick_colors(objects=bricks, mode=mode, seed=brick_seed)
	return bricks


def embed_self_in_blend() -> None:
	"""Embed this script in the saved .blend file so it appears in Blender's Scripting tab."""
	if "__file__" not in globals():
		return
	src = Path(__file__).resolve()
	if not src.exists():
		return
	existing = bpy.data.texts.get(src.name)
	if existing is not None:
		bpy.data.texts.remove(existing)
	txt = bpy.data.texts.new(src.name)
	txt.from_string(src.read_text(encoding="utf-8"))


def build_plate_showcase(
	plate_count: int = DEFAULT_PLATE_COUNT,
	mode: str = DEFAULT_MODE,
	seed: int | None = DEFAULT_SEED,
	include_bricks: bool = INCLUDE_SAMPLE_BRICKS,
	export_dir: Path | None = None,
) -> tuple[list[bpy.types.Object], list[bpy.types.Object]]:
	"""Create colored 20 cm plates arranged in a neat grid, reusing ``lego_random_color``."""
	lrc.clear_scene()
	plates_col = lrc.ensure_collection(PLATES_COLLECTION)
	rng = random.Random(seed)

	base_mesh = create_plate_mesh("Plate_200mm_BaseMesh", steps=96)
	cols = min(3, plate_count)
	rows = math.ceil(plate_count / max(cols, 1))
	spacing = 0.245

	plates: list[bpy.types.Object] = []
	for idx in range(plate_count):
		r = idx // cols
		c = idx % cols
		mesh_copy = base_mesh.copy()
		mesh_copy.name = f"Plate_200mm_{idx + 1:02d}_Mesh"
		plate = bpy.data.objects.new(f"Plate_200mm_{idx + 1:02d}", mesh_copy)
		plate.location = (c * spacing, r * spacing, 0.0)
		plates_col.objects.link(plate)
		plates.append(plate)

	if base_mesh.users == 0:
		bpy.data.meshes.remove(base_mesh)

	# Export standalone OBJ and STL before or after placing at origin cleanly
	if export_dir is not None and plates:
		saved_loc = plates[0].location.copy()
		plates[0].location = (0.0, 0.0, 0.0)
		export_plate_assets(plates[0], export_dir)
		plates[0].location = saved_loc

	# Reuse lego_random_color.randomize_brick_colors directly!
	lrc.randomize_brick_colors(objects=plates, mode=mode, seed=seed)
	apply_plate_plastic_finish(plates)

	bricks = add_sample_bricks_in_plates(plates, rng=rng, mode=mode, seed=seed) if include_bricks else []

	center_x = 0.5 * max(cols - 1, 0) * spacing
	center_y = 0.5 * max(rows - 1, 0) * spacing
	span_x = max(cols * spacing, 0.35)
	span_y = max(rows * spacing, 0.35)

	add_studio_lighting_and_camera(center_x, center_y, span_x, span_y)
	lrc.configure_render_and_viewports()
	embed_self_in_blend()
	return plates, bricks


def recolor_existing_scene(mode: str = DEFAULT_MODE, seed: int | None = DEFAULT_SEED) -> None:
	"""Re-randomize colors on existing plates (and sample bricks if present)."""
	plates_col = bpy.data.collections.get(PLATES_COLLECTION)
	if plates_col is not None:
		plates = [o for o in plates_col.objects if o.type == "MESH"]
		if plates:
			lrc.randomize_brick_colors(objects=plates, mode=mode, seed=seed)
			apply_plate_plastic_finish(plates)
			for o in plates:
				print(f"COLORED: {o.name} -> {o.data.materials[0].name}")

	bricks_col = bpy.data.collections.get(BRICKS_COLLECTION)
	if bricks_col is not None:
		bricks = [o for o in bricks_col.objects if o.type == "MESH"]
		if bricks:
			brick_seed = None if seed is None else seed + 101
			lrc.randomize_brick_colors(objects=bricks, mode=mode, seed=brick_seed)
			for o in bricks:
				print(f"COLORED: {o.name} -> {o.data.materials[0].name}")


def parse_cli_args(argv: list[str]) -> argparse.Namespace:
	user_args = argv[argv.index("--") + 1 :] if "--" in argv else []
	parser = argparse.ArgumentParser(description="Build and randomly color 20 cm plastic plates in Blender.")
	parser.add_argument("--count", type=int, default=DEFAULT_PLATE_COUNT, help="Number of plates.")
	parser.add_argument("--mode", choices=("palette", "hsv"), default=DEFAULT_MODE, help="Color mode.")
	parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Optional RNG seed.")
	parser.add_argument("--no-bricks", action="store_true", help="Do not add sample bricks inside plates.")
	parser.add_argument("--recolor-only", action="store_true", help="Only re-randomize existing scene colors.")
	parser.add_argument("--export-dir", type=Path, default=None, help="Directory to export plate_200mm OBJ/STL.")
	parser.add_argument("--save", type=Path, default=None, help="Output .blend path.")
	parser.add_argument("--render", type=Path, default=None, help="Output preview PNG path.")
	return parser.parse_args(user_args)


def main() -> None:
	args = parse_cli_args(sys.argv)
	recolor = args.recolor_only or (RECOLOR_ONLY and bpy.data.collections.get(PLATES_COLLECTION) is not None)

	if recolor:
		recolor_existing_scene(mode=args.mode, seed=args.seed)
	else:
		export_dir = args.export_dir or (_THIS_DIR / "models")
		plates, bricks = build_plate_showcase(
			plate_count=args.count,
			mode=args.mode,
			seed=args.seed,
			include_bricks=not args.no_bricks,
			export_dir=export_dir,
		)
		for obj in [*plates, *bricks]:
			print(f"COLORED: {obj.name} -> {obj.data.materials[0].name}")

	if args.save is not None:
		args.save.parent.mkdir(parents=True, exist_ok=True)
		bpy.ops.wm.save_as_mainfile(filepath=str(args.save.resolve()))
		print(f"SAVED_BLEND: {args.save.resolve()}")

	if args.render is not None:
		args.render.parent.mkdir(parents=True, exist_ok=True)
		bpy.context.scene.render.filepath = str(args.render.resolve())
		bpy.ops.render.render(write_still=True)
		print(f"RENDERED_PNG: {args.render.resolve()}")


if __name__ == "__main__":
	main()
