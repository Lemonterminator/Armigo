"""Upgrade an exported MuJoCo sorting_cell Blender scene with real plate, Duplo brick, and fixed ArUco textures."""

from __future__ import annotations

import math
from pathlib import Path

import bpy

REPO_ROOT = Path(r"c:\Users\Jiang\.gemini\antigravity\scratch\hackjunction")
SRC_BLEND = Path(r"C:\tmp\hackjunction-sim\artifacts\blender\sorting_cell_sim_0000.blend")
ASSETS_DIR = REPO_ROOT / "scripts" / "sim" / "assets"
OUT_DIR = REPO_ROOT / "artifacts" / "blender_sim"
OUT_BLEND = OUT_DIR / "sorting_cell_sim_0000_upgraded.blend"


def import_obj_mesh(obj_path: Path, name: str, use_edge_split: bool = False) -> tuple[bpy.types.Mesh, bool]:
	"""Import an OBJ file cleanly and return a detached, smooth-shaded Mesh datablock."""
	bpy.ops.wm.obj_import(filepath=str(obj_path), forward_axis="Y", up_axis="Z")
	tmp = bpy.context.active_object
	mesh = tmp.data
	mesh.name = name
	for p in mesh.polygons:
		p.use_smooth = True
	mesh.update()
	bpy.data.objects.remove(tmp, do_unlink=True)
	return mesh, use_edge_split


def enhance_plastic_material(mat: bpy.types.Material, roughness: float = 0.20) -> None:
	"""Tune Principled BSDF for smooth injection-molded plastic highlights."""
	if not mat or not mat.use_nodes:
		return
	bsdf = mat.node_tree.nodes.get("Principled BSDF")
	if bsdf is None:
		return
	bsdf.inputs["Roughness"].default_value = roughness
	bsdf.inputs["IOR"].default_value = 1.49
	if "Coat Weight" in bsdf.inputs:
		bsdf.inputs["Coat Weight"].default_value = 0.22
		bsdf.inputs["Coat Roughness"].default_value = 0.06
	if "Base Color" in bsdf.inputs:
		col = bsdf.inputs["Base Color"].default_value
		mat.diffuse_color = (col[0], col[1], col[2], 1.0)


def replace_bowl(prefix: str, bottom_name: str, new_name: str, plate_mesh: bpy.types.Mesh) -> None:
	"""Replace 65 cylinder+box bowl primitives with one smooth 20 cm plate object."""
	bowls_col = bpy.data.collections["Bowls"]
	bottom = bpy.data.objects[bottom_name]
	loc = (bottom.location.x, bottom.location.y, 0.0)
	mat = bottom.data.materials[0] if bottom.data.materials else None
	enhance_plastic_material(mat, roughness=0.22)

	for obj in list(bowls_col.objects):
		if obj.name.startswith(prefix):
			bpy.data.objects.remove(obj, do_unlink=True)

	mesh_copy = plate_mesh.copy()
	mesh_copy.name = f"{new_name}_Mesh"
	plate_obj = bpy.data.objects.new(new_name, mesh_copy)
	plate_obj.location = loc
	if mat is not None:
		plate_obj.data.materials.append(mat)
	bowls_col.objects.link(plate_obj)


def swap_mesh_preserving_transform(obj: bpy.types.Object, src_mesh: bpy.types.Mesh, add_edge_split: bool = True) -> None:
	"""Replace ``obj.data`` in place so location, rotation, keyframes, and material stay intact."""
	mats = list(obj.data.materials)
	old_mesh = obj.data
	new_mesh = src_mesh.copy()
	new_mesh.name = f"{obj.name}_DuploMesh"
	for m in mats:
		new_mesh.materials.append(m)
		enhance_plastic_material(m, roughness=0.17)
	obj.data = new_mesh
	if old_mesh.users == 0:
		bpy.data.meshes.remove(old_mesh)
	if add_edge_split and "EdgeSplit" not in obj.modifiers:
		mod = obj.modifiers.new(name="EdgeSplit", type="EDGE_SPLIT")
		mod.use_edge_angle = True
		mod.split_angle = math.radians(35.0)


def fix_aruco_uvs() -> None:
	"""Normalize metric-scaled ArUco UVs to [0, 1] and switch sampling to Closest."""
	scales = {
		"Mesh_world__box_1_id1_geom": 1.0 / 0.05,
		"Mesh_world__box_2_id2_geom": 1.0 / 0.05,
		"Mesh_world__box_3_id3_geom": 1.0 / 0.075,
	}
	for obj_name, factor in scales.items():
		obj = bpy.data.objects.get(obj_name)
		if obj is None or obj.type != "MESH":
			continue
		uv_layer = obj.data.uv_layers.active
		if uv_layer is not None:
			for loop_uv in uv_layer.data:
				loop_uv.uv.x *= factor
				loop_uv.uv.y *= factor
		if obj.data.materials and obj.data.materials[0].use_nodes:
			for node in obj.data.materials[0].node_tree.nodes:
				if node.type == "TEX_IMAGE":
					node.interpolation = "Closest"


def main() -> None:
	OUT_DIR.mkdir(parents=True, exist_ok=True)
	bpy.ops.wm.open_mainfile(filepath=str(SRC_BLEND))

	plate_mesh, _ = import_obj_mesh(ASSETS_DIR / "plate_200mm.obj", "Plate_200mm_Shared")
	duplo_mesh, _ = import_obj_mesh(ASSETS_DIR / "duplo_2x2.obj", "Duplo_2x2_Shared")

	# 1. Replace green & pink bowls
	replace_bowl("Mesh_green_bowl", "Mesh_green_bowl__cylinder_1_id54_geom", "Green_Plate_200mm", plate_mesh)
	replace_bowl("Mesh_pink_bowl", "Mesh_pink_bowl__cylinder_1_id119_geom", "Pink_Plate_200mm", plate_mesh)

	# 2. Replace animated main brick body mesh & remove obsolete stud cylinders
	brick = bpy.data.objects["Mesh_brick_id184_geom"]
	swap_mesh_preserving_transform(brick, duplo_mesh, add_edge_split=True)
	brick_col = bpy.data.collections["Brick"]
	for obj in list(brick_col.objects):
		if obj.name.startswith("Mesh_brick__cylinder_"):
			bpy.data.objects.remove(obj, do_unlink=True)

	# 3. Replace clutter boxes & remove clutter stud cylinders
	clutter_col = bpy.data.collections["Clutter_and_markers"]
	for name in ("Mesh_world__box_4_id191_geom", "Mesh_world__box_5_id196_geom", "Mesh_world__box_6_id201_geom"):
		obj = bpy.data.objects.get(name)
		if obj is not None:
			swap_mesh_preserving_transform(obj, duplo_mesh, add_edge_split=True)
	for obj in list(clutter_col.objects):
		if obj.name.startswith("Mesh_world__cylinder_"):
			bpy.data.objects.remove(obj, do_unlink=True)

	# Remove unused base meshes
	if plate_mesh.users == 0:
		bpy.data.meshes.remove(plate_mesh)
	if duplo_mesh.users == 0:
		bpy.data.meshes.remove(duplo_mesh)

	# 4. Fix ArUco marker UVs and pack external PNG textures
	fix_aruco_uvs()
	bpy.ops.file.pack_all()

	# 5. Configure camera & render preview
	scene = bpy.context.scene
	cam = bpy.data.objects.get("Camera_front") or bpy.data.objects.get("Camera_top")
	if cam is not None:
		cam.data.clip_start = 0.001
		scene.camera = cam
	scene.render.engine = "BLENDER_EEVEE"
	scene.render.resolution_x = 960
	scene.render.resolution_y = 540
	scene.render.resolution_percentage = 100

	bpy.ops.wm.save_as_mainfile(filepath=str(OUT_BLEND))
	print(f"SAVED_UPGRADED_BLEND: {OUT_BLEND}")

	# Render 4 key frames across the 398-frame animation for visual verification
	for f in (0, 140, 240, 380):
		scene.frame_set(f)
		out_png = OUT_DIR / f"frame_{f:04d}.png"
		scene.render.filepath = str(out_png)
		bpy.ops.render.render(write_still=True)
		print(f"RENDERED: {out_png}")


if __name__ == "__main__":
	main()
