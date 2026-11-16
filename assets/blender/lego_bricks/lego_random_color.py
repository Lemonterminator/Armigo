"""Import open-source 2x2, 2x3, and 2x4 LEGO brick STLs and apply random ABS plastic colors.

Works both:
1. Inside Blender GUI (Text Editor -> Run Script):
   - Set ``RECOLOR_ONLY = True`` to re-randomize colors of existing bricks in the scene,
     or keep ``RECOLOR_ONLY = False`` to rebuild the showcase scene from the STL files.
2. From the command line:
   blender --background --python lego_random_color.py -- --seed 42 --mode palette --render preview.png --save lego_showcase.blend
"""

from __future__ import annotations

import argparse
import colorsys
import math
import random
import sys
from pathlib import Path

import bmesh
import bpy
import mathutils

# ---------------------------------------------------------------------------
# User-tweakable settings when running directly inside Blender's Text Editor
# ---------------------------------------------------------------------------
RECOLOR_ONLY = False
DEFAULT_MODE = "palette"  # "palette" (classic LEGO ABS colors) or "hsv" (continuous random)
DEFAULT_SEED: int | None = None  # Set to an int (e.g. 42) for reproducible colors, or None
DEFAULT_COPIES_PER_TYPE = 4
DEFAULT_SCALE = 0.001  # STLs are in mm -> 0.001 gives true 1:1 LEGO dimensions in meters

COLLECTION_NAME = "LEGO_Bricks"
STUDIO_COLLECTION = "Studio_Setup"

BRICK_SPECS = (
	("Brick_2x2", "3003_Brick_2x2.stl"),
	("Brick_2x3", "3002_Brick_2x3.stl"),
	("Brick_2x4", "3001_Brick_2x4.stl"),
)

# Official-style LEGO ABS sRGB palette: (name, (R, G, B) in 0..255 sRGB)
LEGO_PALETTE_SRGB: tuple[tuple[str, tuple[int, int, int]], ...] = (
	("Bright_Red", (201, 26, 9)),
	("Bright_Blue", (0, 85, 191)),
	("Bright_Yellow", (242, 205, 55)),
	("Dark_Green", (35, 120, 65)),
	("Bright_Orange", (254, 138, 24)),
	("Bright_Green", (75, 159, 74)),
	("Medium_Azure", (54, 174, 191)),
	("Magenta", (146, 57, 120)),
	("Coral", (255, 109, 119)),
	("Lime", (187, 233, 11)),
	("Dark_Purple", (63, 54, 145)),
	("Light_Bluish_Gray", (160, 165, 169)),
	("White", (244, 244, 244)),
	("Bright_Pink", (228, 173, 200)),
	("Sand_Blue", (96, 116, 161)),
)


def srgb_channel_to_linear(c: float) -> float:
	"""Convert one sRGB channel in [0, 1] to scene-referred linear RGB."""
	if c <= 0.04045:
		return c / 12.92
	return ((c + 0.055) / 1.055) ** 2.4


def srgb_255_to_linear_rgba(rgb: tuple[int, int, int]) -> tuple[float, float, float, float]:
	"""Convert 8-bit sRGB tuple to Blender linear RGBA."""
	return (
		srgb_channel_to_linear(rgb[0] / 255.0),
		srgb_channel_to_linear(rgb[1] / 255.0),
		srgb_channel_to_linear(rgb[2] / 255.0),
		1.0,
	)


def resolve_models_dir() -> Path:
	"""Locate the directory containing the downloaded STL brick files."""
	candidates: list[Path] = []
	if "__file__" in globals():
		candidates.append(Path(__file__).resolve().parent / "models")
	if bpy.data.filepath:
		blend_dir = Path(bpy.data.filepath).resolve().parent
		candidates.append(blend_dir / "models")
		candidates.append(blend_dir / "lego_bricks" / "models")
	candidates.append(
		Path(r"c:\Users\Jiang\.gemini\antigravity\scratch\hackjunction\artifacts\lego_bricks\models")
	)
	for path in candidates:
		if path.is_dir() and (path / "3001_Brick_2x4.stl").exists():
			return path
	raise FileNotFoundError(f"Could not find LEGO STL models directory. Checked: {candidates}")


def ensure_collection(name: str, parent: bpy.types.Collection | None = None) -> bpy.types.Collection:
	"""Return an existing collection or create and link a new one."""
	col = bpy.data.collections.get(name)
	if col is None:
		col = bpy.data.collections.new(name)
		target_parent = parent or bpy.context.scene.collection
		target_parent.children.link(col)
	return col


def move_to_collection(obj: bpy.types.Object, col: bpy.types.Collection) -> None:
	"""Move object cleanly so it belongs only to ``col``."""
	for existing in list(obj.users_collection):
		existing.objects.unlink(obj)
	col.objects.link(obj)


def clear_scene() -> None:
	"""Reset Blender to an empty scene without leftover data blocks."""
	bpy.ops.wm.read_factory_settings(use_empty=True)


def create_abs_plastic_material(
	mat_name: str,
	linear_rgba: tuple[float, float, float, float],
	roughness: float = 0.17,
) -> bpy.types.Material:
	"""Create a realistic glossy LEGO ABS plastic Principled BSDF material."""
	mat = bpy.data.materials.new(name=mat_name)
	mat.use_nodes = True
	mat.diffuse_color = linear_rgba  # Also visible in Solid viewport mode

	nt = mat.node_tree
	bsdf = nt.nodes.get("Principled BSDF")
	if bsdf is None:
		bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")

	bsdf.inputs["Base Color"].default_value = linear_rgba
	bsdf.inputs["Metallic"].default_value = 0.0
	bsdf.inputs["Roughness"].default_value = roughness
	bsdf.inputs["IOR"].default_value = 1.49  # ABS plastic refractive index
	bsdf.inputs["Specular IOR Level"].default_value = 0.55
	bsdf.inputs["Coat Weight"].default_value = 0.30
	bsdf.inputs["Coat Roughness"].default_value = 0.04
	bsdf.inputs["Coat IOR"].default_value = 1.5
	bsdf.inputs["Subsurface Weight"].default_value = 0.0
	return mat


def pick_random_colors(count: int, mode: str, rng: random.Random) -> list[tuple[str, tuple[float, float, float, float]]]:
	"""Sample ``count`` color entries as (label, linear_rgba)."""
	if mode == "palette":
		pool = list(LEGO_PALETTE_SRGB)
		rng.shuffle(pool)
		picked: list[tuple[str, tuple[float, float, float, float]]] = []
		while len(picked) < count:
			if not pool:
				pool = list(LEGO_PALETTE_SRGB)
				rng.shuffle(pool)
			name, srgb = pool.pop()
			picked.append((name, srgb_255_to_linear_rgba(srgb)))
		return picked

	if mode == "hsv":
		out: list[tuple[str, tuple[float, float, float, float]]] = []
		for i in range(count):
			h = (rng.random() + i / max(count, 1)) % 1.0
			s = rng.uniform(0.68, 0.96)
			v = rng.uniform(0.35, 0.95)
			r, g, b = colorsys.hsv_to_rgb(h, s, v)
			rgba = (
				srgb_channel_to_linear(r),
				srgb_channel_to_linear(g),
				srgb_channel_to_linear(b),
				1.0,
			)
			out.append((f"HSV_{i:02d}", rgba))
		return out

	raise ValueError(f"Unsupported color mode: {mode!r} (expected 'palette' or 'hsv')")


def assign_material(obj: bpy.types.Object, mat: bpy.types.Material) -> None:
	"""Assign a single material slot to an object."""
	if obj.data.materials:
		obj.data.materials[0] = mat
	else:
		obj.data.materials.append(mat)
	obj.color = mat.diffuse_color


def randomize_brick_colors(
	objects: list[bpy.types.Object] | None = None,
	mode: str = DEFAULT_MODE,
	seed: int | None = DEFAULT_SEED,
) -> list[tuple[str, str]]:
	"""Assign random LEGO ABS plastic materials to the target brick objects."""
	rng = random.Random(seed)
	if objects is None:
		col = bpy.data.collections.get(COLLECTION_NAME)
		if col is not None:
			objects = [obj for obj in col.objects if obj.type == "MESH"]
		else:
			objects = [obj for obj in bpy.context.selected_objects if obj.type == "MESH"]

	if not objects:
		raise RuntimeError(f"No brick objects found in collection {COLLECTION_NAME!r} or selection.")

	colors = pick_random_colors(len(objects), mode=mode, rng=rng)
	assignments: list[tuple[str, str]] = []
	for idx, (obj, (color_name, linear_rgba)) in enumerate(zip(objects, colors)):
		roughness = rng.uniform(0.14, 0.21)
		mat_name = f"LEGO_ABS_{color_name}_{idx:02d}"
		mat = create_abs_plastic_material(mat_name, linear_rgba, roughness=roughness)
		assign_material(obj, mat)
		assignments.append((obj.name, color_name))

	# Clean up orphaned materials from prior randomizations
	for mat in list(bpy.data.materials):
		if mat.users == 0 and mat.name.startswith("LEGO_ABS_"):
			bpy.data.materials.remove(mat)

	return assignments


def import_and_normalize_brick(
	stl_path: Path,
	name: str,
	scale: float,
) -> bpy.types.Object:
	"""Import an STL brick, center XY at 0, place bottom at Z=0, smooth studs + crisp corners."""
	bpy.ops.wm.stl_import(filepath=str(stl_path))
	obj = bpy.context.active_object
	obj.name = name
	mesh = obj.data
	mesh.name = f"{name}_Mesh"

	# Clean STL triangle soup: merge coincident vertices, center XY, put Z_min=0, scale to meters
	bm = bmesh.new()
	bm.from_mesh(mesh)
	bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-4)
	bm.verts.ensure_lookup_table()

	xs = [v.co.x for v in bm.verts]
	ys = [v.co.y for v in bm.verts]
	zs = [v.co.z for v in bm.verts]
	cx = 0.5 * (min(xs) + max(xs))
	cy = 0.5 * (min(ys) + max(ys))
	z0 = min(zs)

	for v in bm.verts:
		v.co.x = (v.co.x - cx) * scale
		v.co.y = (v.co.y - cy) * scale
		v.co.z = (v.co.z - z0) * scale

	bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
	bm.to_mesh(mesh)
	bm.free()

	# Smooth cylindrical studs while preserving crisp 90-degree brick edges
	for poly in mesh.polygons:
		poly.use_smooth = True
	mesh.update()

	split = obj.modifiers.new(name="EdgeSplit", type="EDGE_SPLIT")
	split.use_edge_angle = True
	split.split_angle = math.radians(35.0)

	return obj


def duplicate_brick(template: bpy.types.Object, name: str, col: bpy.types.Collection) -> bpy.types.Object:
	"""Create an independent copy sharing mesh data (so each copy holds its own material cleanly)."""
	copy = template.copy()
	copy.name = name
	# Link material per-object so shared mesh instances still support independent materials
	copy.data = template.data.copy()
	copy.data.name = f"{name}_Mesh"
	col.objects.link(copy)
	return copy


def add_studio_lighting_and_camera(
	center_x: float,
	center_y: float,
	span_x: float,
	span_y: float,
	scale: float,
) -> bpy.types.Object:
	"""Build a clean studio backdrop, 3-point lighting, and showcase camera."""
	studio_col = ensure_collection(STUDIO_COLLECTION)
	unit_scale = scale / 0.001

	# Matte studio floor large enough to fill camera frustum cleanly
	bpy.ops.mesh.primitive_plane_add(size=2.0 * unit_scale, location=(center_x, center_y, 0.0))
	floor = bpy.context.active_object
	floor.name = "Studio_Floor"
	move_to_collection(floor, studio_col)
	floor_mat = bpy.data.materials.new("Studio_Floor_Mat")
	floor_mat.use_nodes = True
	floor_bsdf = floor_mat.node_tree.nodes.get("Principled BSDF")
	floor_bsdf.inputs["Base Color"].default_value = (0.14, 0.15, 0.17, 1.0)
	floor_bsdf.inputs["Roughness"].default_value = 0.68
	floor_mat.diffuse_color = (0.14, 0.15, 0.17, 1.0)
	floor.data.materials.append(floor_mat)

	# Balanced studio world environment
	world = bpy.data.worlds.new("Studio_World")
	bpy.context.scene.world = world
	world.use_nodes = True
	bg = world.node_tree.nodes.get("Background")
	if bg is not None:
		bg.inputs["Color"].default_value = (0.18, 0.20, 0.23, 1.0)
		bg.inputs["Strength"].default_value = 0.35

	def add_area_light(name: str, loc: tuple[float, float, float], energy: float, size: float, color: tuple[float, float, float]) -> None:
		light_data = bpy.data.lights.new(name=name, type="AREA")
		light_data.energy = energy * (unit_scale ** 2)
		light_data.size = size * unit_scale
		light_data.color = color
		light_obj = bpy.data.objects.new(name=name, object_data=light_data)
		light_obj.location = loc
		direction = mathutils.Vector((center_x, center_y, 0.006 * unit_scale)) - light_obj.location
		light_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
		studio_col.objects.link(light_obj)

	add_area_light(
		"Key_Light",
		(center_x + 0.18 * unit_scale, center_y - 0.18 * unit_scale, 0.24 * unit_scale),
		energy=0.55,
		size=0.14,
		color=(1.0, 0.98, 0.96),
	)
	add_area_light(
		"Fill_Light",
		(center_x - 0.22 * unit_scale, center_y - 0.12 * unit_scale, 0.18 * unit_scale),
		energy=0.22,
		size=0.22,
		color=(0.88, 0.94, 1.0),
	)
	add_area_light(
		"Rim_Light",
		(center_x - 0.02 * unit_scale, center_y + 0.22 * unit_scale, 0.18 * unit_scale),
		energy=0.40,
		size=0.14,
		color=(1.0, 1.0, 1.0),
	)

	# Showcase camera aimed at grid center
	cam_data = bpy.data.cameras.new("Camera_Showcase")
	cam_data.lens = 50.0
	cam_data.clip_start = 0.001
	cam_data.clip_end = 50.0
	cam_obj = bpy.data.objects.new("Camera_Showcase", cam_data)
	dist = max(span_x, span_y) * 1.42
	cam_obj.location = (center_x + dist * 0.42, center_y - dist * 0.86, dist * 0.62)
	target = mathutils.Vector((center_x, center_y, 0.004 * unit_scale))
	cam_obj.rotation_euler = (target - cam_obj.location).to_track_quat("-Z", "Y").to_euler()
	studio_col.objects.link(cam_obj)
	bpy.context.scene.camera = cam_obj

	return cam_obj


def configure_render_and_viewports() -> None:
	"""Configure EEVEE Next render settings and switch UI viewports to Material Preview."""
	scene = bpy.context.scene
	scene.render.engine = "BLENDER_EEVEE"
	scene.render.resolution_x = 1600
	scene.render.resolution_y = 1000
	scene.render.resolution_percentage = 100

	# Use AgX color management when supported
	try:
		scene.view_settings.view_transform = "AgX"
		scene.view_settings.look = "AgX - Medium High Contrast"
	except Exception:
		pass

	for window in bpy.context.window_manager.windows:
		for area in window.screen.areas:
			if area.type == "VIEW_3D":
				for space in area.spaces:
					if space.type == "VIEW_3D":
						space.clip_start = 0.001
						space.shading.type = "RENDERED"
						space.region_3d.view_perspective = "CAMERA"


def embed_self_in_blend() -> None:
	"""Embed this script in the saved .blend file so it appears in Blender's Scripting tab."""
	if "__file__" not in globals():
		return
	src_path = Path(__file__).resolve()
	if not src_path.exists():
		return
	text_name = src_path.name
	existing = bpy.data.texts.get(text_name)
	if existing is not None:
		bpy.data.texts.remove(existing)
	txt = bpy.data.texts.new(text_name)
	txt.from_string(src_path.read_text(encoding="utf-8"))


def build_showcase_scene(
	models_dir: Path,
	copies_per_type: int = DEFAULT_COPIES_PER_TYPE,
	scale: float = DEFAULT_SCALE,
	mode: str = DEFAULT_MODE,
	seed: int | None = DEFAULT_SEED,
) -> list[bpy.types.Object]:
	"""Build a fresh studio scene populated with 2x2, 2x3, and 2x4 LEGO bricks."""
	clear_scene()
	brick_col = ensure_collection(COLLECTION_NAME)
	rng = random.Random(seed)

	templates: list[tuple[str, bpy.types.Object]] = []
	for label, filename in BRICK_SPECS:
		tmpl = import_and_normalize_brick(models_dir / filename, f"_tmpl_{label}", scale=scale)
		templates.append((label, tmpl))

	step_x = 0.044 * (scale / 0.001)
	step_y = 0.032 * (scale / 0.001)
	all_bricks: list[bpy.types.Object] = []

	for row_idx, (label, tmpl) in enumerate(templates):
		for col_idx in range(copies_per_type):
			brick = duplicate_brick(tmpl, f"{label}_{col_idx + 1:02d}", brick_col)
			brick.location = (col_idx * step_x, row_idx * step_y, 0.0)
			# Keep first column straight for clean size comparison; add subtle natural yaw to others
			yaw_deg = 0.0 if col_idx == 0 else rng.uniform(-10.0, 10.0)
			brick.rotation_euler = (0.0, 0.0, math.radians(yaw_deg))
			all_bricks.append(brick)

	# Remove temporary template objects
	for _, tmpl in templates:
		mesh = tmpl.data
		bpy.data.objects.remove(tmpl, do_unlink=True)
		if mesh.users == 0:
			bpy.data.meshes.remove(mesh)

	center_x = 0.5 * max(copies_per_type - 1, 0) * step_x
	center_y = 0.5 * (len(BRICK_SPECS) - 1) * step_y
	span_x = max(copies_per_type * step_x, 0.08 * (scale / 0.001))
	span_y = max(len(BRICK_SPECS) * step_y, 0.08 * (scale / 0.001))

	randomize_brick_colors(objects=all_bricks, mode=mode, seed=seed)
	add_studio_lighting_and_camera(center_x, center_y, span_x, span_y, scale=scale)
	configure_render_and_viewports()
	embed_self_in_blend()
	return all_bricks


def parse_cli_args(argv: list[str]) -> argparse.Namespace:
	"""Parse optional arguments passed after '--' when launching Blender."""
	user_args = argv[argv.index("--") + 1 :] if "--" in argv else []
	parser = argparse.ArgumentParser(description="Build and randomly color LEGO bricks in Blender.")
	parser.add_argument("--models-dir", type=Path, default=None, help="Folder containing STL brick models.")
	parser.add_argument("--copies", type=int, default=DEFAULT_COPIES_PER_TYPE, help="Copies per brick type.")
	parser.add_argument("--scale", type=float, default=DEFAULT_SCALE, help="Scale factor (0.001 = mm -> m).")
	parser.add_argument("--mode", choices=("palette", "hsv"), default=DEFAULT_MODE, help="Color sampling mode.")
	parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Optional RNG seed.")
	parser.add_argument("--recolor-only", action="store_true", help="Only re-color bricks already in scene.")
	parser.add_argument("--save", type=Path, default=None, help="Output .blend path to save.")
	parser.add_argument("--render", type=Path, default=None, help="Output .png preview render path.")
	return parser.parse_args(user_args)


def main() -> None:
	args = parse_cli_args(sys.argv)
	recolor = args.recolor_only or (RECOLOR_ONLY and bpy.data.collections.get(COLLECTION_NAME) is not None)

	if recolor:
		assignments = randomize_brick_colors(mode=args.mode, seed=args.seed)
	else:
		models_dir = args.models_dir or resolve_models_dir()
		build_showcase_scene(
			models_dir=models_dir,
			copies_per_type=args.copies,
			scale=args.scale,
			mode=args.mode,
			seed=args.seed,
		)
		assignments = [(obj.name, obj.data.materials[0].name) for obj in bpy.data.collections[COLLECTION_NAME].objects]

	for obj_name, mat_name in assignments:
		print(f"COLORED: {obj_name} -> {mat_name}")

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
