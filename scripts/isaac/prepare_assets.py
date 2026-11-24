"""Prepare USD assets for Stage 1-3 Isaac Sim / Isaac Lab validation.

Converts OBJ meshes (plate_200mm.obj, duplo_2x2.obj) to OpenUSD props with
materials and PhysX collision schemas, and creates a PhysX-ready SO-101 USD
layer referencing the MuJoCo-converted so101.usda with the wrist camera attached.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import trimesh
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from digital_twin_armigo.physics import load_physics_spec
from digital_twin_armigo.presentation import defect_strokes

SIM_ASSETS = ROOT / "assets" / "meshes"
USD_OUT = ROOT / "assets"
SO101_USDA = USD_OUT / "so101_usd" / "so101.usda"
SO101_PHYSX_USDA = USD_OUT / "so101_usd" / "so101_physx.usda"
PROPS_DIR = USD_OUT / "props"


def _bind_plastic_material(
	stage: Usd.Stage, root_path: str, name: str, rgb: tuple[float, float, float], roughness: float = 0.25
) -> None:
	mat_path = f"{root_path}/Looks/{name}"
	UsdGeom.Scope.Define(stage, f"{root_path}/Looks")
	mat = UsdShade.Material.Define(stage, mat_path)
	shader = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
	shader.CreateIdAttr("UsdPreviewSurface")
	shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
	shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
	shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
	shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.49)
	mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
	UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(root_path)).Bind(mat)


def build_obj_prop_usd(
	obj_path: Path,
	usd_path: Path,
	prim_name: str,
	rgb: tuple[float, float, float],
	mass: float,
	static_friction: float,
	dynamic_friction: float,
	approx: str,
	restitution: float = 0.05,
	roughness: float = 0.25,
	rigid_body: bool = True,
	kinematic: bool = False,
	defect_face: int | None = None,
	brick_size: tuple[float, float, float] = (0.0318, 0.0318, 0.0192),
) -> Path:
	"""Load a Z-up meters OBJ via trimesh and export a clean USD RigidBody/Collider prop."""
	usd_path.parent.mkdir(parents=True, exist_ok=True)
	mesh = trimesh.load(obj_path, force="mesh")
	stage = Usd.Stage.CreateNew(str(usd_path))
	UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
	UsdGeom.SetStageMetersPerUnit(stage, 1.0)

	xform = UsdGeom.Xform.Define(stage, f"/{prim_name}")
	stage.SetDefaultPrim(xform.GetPrim())

	if rigid_body:
		rb = UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim())
		rb.CreateKinematicEnabledAttr(kinematic)
		mass_api = UsdPhysics.MassAPI.Apply(xform.GetPrim())
		mass_api.CreateMassAttr(float(mass))

	geom_prim = UsdGeom.Mesh.Define(stage, f"/{prim_name}/Mesh")
	verts = [Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in mesh.vertices]
	faces = mesh.faces
	geom_prim.CreatePointsAttr(verts)
	geom_prim.CreateFaceVertexCountsAttr([3] * len(faces))
	geom_prim.CreateFaceVertexIndicesAttr([int(i) for f in faces for i in f])
	if mesh.vertex_normals is not None and len(mesh.vertex_normals) == len(mesh.vertices):
		norms = [Gf.Vec3f(float(n[0]), float(n[1]), float(n[2])) for n in mesh.vertex_normals]
		geom_prim.CreateNormalsAttr(norms)
		geom_prim.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
	geom_prim.CreateDoubleSidedAttr(True)

	col_api = UsdPhysics.CollisionAPI.Apply(geom_prim.GetPrim())
	col_api.CreateCollisionEnabledAttr(True)
	if approx:
		mesh_col = UsdPhysics.MeshCollisionAPI.Apply(geom_prim.GetPrim())
		mesh_col.CreateApproximationAttr(approx)

	_bind_plastic_material(stage, f"/{prim_name}", f"{prim_name}_mat", rgb, roughness=roughness)
	# Also bind material directly onto Mesh prim so Hydra/RTX resolves cleanly
	mat_prim = UsdShade.Material(stage.GetPrimAtPath(f"/{prim_name}/Looks/{prim_name}_mat"))
	UsdShade.MaterialBindingAPI.Apply(geom_prim.GetPrim()).Bind(mat_prim)

	# Physics material aligned with PhysicsSpec (max combine mode matches MuJoCo)
	pmat_path = f"/{prim_name}/PhysicsMaterial"
	UsdShade.Material.Define(stage, pmat_path)
	pmat_prim = stage.GetPrimAtPath(pmat_path)
	pmat = UsdPhysics.MaterialAPI.Apply(pmat_prim)
	pmat.CreateStaticFrictionAttr(float(static_friction))
	pmat.CreateDynamicFrictionAttr(float(dynamic_friction))
	pmat.CreateRestitutionAttr(float(restitution))
	pmat_prim.CreateAttribute("physxMaterial:frictionCombineMode", Sdf.ValueTypeNames.Token).Set("max")
	pmat_prim.CreateAttribute("physxMaterial:restitutionCombineMode", Sdf.ValueTypeNames.Token).Set("max")
	UsdShade.MaterialBindingAPI.Apply(geom_prim.GetPrim()).Bind(
		UsdShade.Material(pmat_prim),
		materialPurpose="physics",
	)

	if defect_face is not None:
		import numpy as np
		from scipy.spatial.transform import Rotation

		_bind_plastic_material(stage, f"/{prim_name}", "defect_mat", (0.02, 0.02, 0.02), roughness=0.45)
		dmat = UsdShade.Material(stage.GetPrimAtPath(f"/{prim_name}/Looks/defect_mat"))
		for idx, (centre, axis_u, side_v, half_len) in enumerate(defect_strokes(brick_size, defect_face)):
			normal = np.cross(axis_u, side_v)
			normal = normal / np.linalg.norm(normal)
			# Push slightly past brick body surface (+0.0005m) so cross stays crisp without z-fighting
			pos = centre + normal * 0.0005
			rot_m = np.column_stack([axis_u / np.linalg.norm(axis_u), side_v / np.linalg.norm(side_v), normal])
			q = Rotation.from_matrix(rot_m).as_quat()  # x, y, z, w
			bar = UsdGeom.Cube.Define(stage, f"/{prim_name}/Defect_{idx}")
			bar.CreateSizeAttr(1.0)
			xf = UsdGeom.Xformable(bar.GetPrim())
			xf.AddTranslateOp().Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
			xf.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(
				Gf.Quatf(float(q[3]), float(q[0]), float(q[1]), float(q[2]))
			)
			xf.AddScaleOp().Set(Gf.Vec3f(float(half_len * 2.0), 0.0022, 0.0008))
			UsdShade.MaterialBindingAPI.Apply(bar.GetPrim()).Bind(dmat)

	stage.GetRootLayer().Save()
	return usd_path


def build_so101_physx_usd() -> Path:
	"""Create so101_physx.usda overriding so101.usda with unified PhysicsSpec drives, gripper friction, and wrist_cam."""
	spec = load_physics_spec()
	stage = Usd.Stage.CreateNew(str(SO101_PHYSX_USDA))
	UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
	UsdGeom.SetStageMetersPerUnit(stage, 1.0)
	stage.GetRootLayer().subLayerPaths.append("./so101.usda")

	# Traverse composed stage to discover dynamic mesh colliders and revolute joints
	mesh_colliders: list[str] = []
	joints: list[str] = []
	for prim in stage.Traverse():
		p = str(prim.GetPath())
		if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
			mesh_colliders.append(p)
		if prim.GetTypeName() == "PhysicsRevoluteJoint":
			joints.append(p)

	# Switch dynamic triangle mesh colliders to convexDecomposition so PhysX accepts them cleanly
	stage.SetEditTarget(stage.GetRootLayer())
	for p in mesh_colliders:
		over = stage.OverridePrim(p)
		mcol = UsdPhysics.MeshCollisionAPI.Apply(over)
		mcol.CreateApproximationAttr("convexDecomposition")

	# Add angular drives from unified PhysicsSpec (arm vs gripper separate effort/stiffness)
	for jp in joints:
		over = stage.OverridePrim(jp)
		drv = UsdPhysics.DriveAPI.Apply(over, "angular")
		act = spec.gripper_actuator if jp.endswith("/gripper") else spec.arm_actuator
		drv.CreateTypeAttr("force")
		drv.CreateStiffnessAttr(float(act.stiffness))
		drv.CreateDampingAttr(float(act.damping))
		drv.CreateMaxForceAttr(float(act.effort_limit))

	# Set gripper PhysicsMaterial_1 from unified gripper_pad spec
	gp_mat = stage.OverridePrim("/so101/Physics/PhysicsMaterial_1")
	pm = UsdPhysics.MaterialAPI.Apply(gp_mat)
	pm.CreateStaticFrictionAttr(float(spec.gripper_pad.static_friction))
	pm.CreateDynamicFrictionAttr(float(spec.gripper_pad.dynamic_friction))
	pm.CreateRestitutionAttr(float(spec.gripper_pad.restitution))
	gp_mat.CreateAttribute("physxMaterial:frictionCombineMode", Sdf.ValueTypeNames.Token).Set("max")
	gp_mat.CreateAttribute("physxMaterial:restitutionCombineMode", Sdf.ValueTypeNames.Token).Set("max")

	# Attach wrist_cam matching so101.xml camera_mount pose + fovy=61
	cam_mount = "/so101/Geometry/base/shoulder/upper_arm/lower_arm/wrist/gripper/camera_mount"
	cam_path = f"{cam_mount}/wrist_cam"
	cam = UsdGeom.Camera.Define(stage, cam_path)
	xf = UsdGeom.Xformable(cam.GetPrim())
	xf.AddTranslateOp().Set(Gf.Vec3d(-0.00296, 0.05436, -0.05310))
	xf.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Quatf(0.958280, -0.285792, -0.001556, 0.004547))
	focal_mm = 24.0
	v_ap = 2.0 * focal_mm * math.tan(math.radians(61.0 / 2.0))
	h_ap = v_ap * (640.0 / 480.0)
	cam.CreateFocalLengthAttr(focal_mm)
	cam.CreateVerticalApertureAttr(v_ap)
	cam.CreateHorizontalApertureAttr(h_ap)
	cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 10.0))

	stage.SetDefaultPrim(stage.GetPrimAtPath("/so101"))
	stage.GetRootLayer().Save()
	print("Saved SO101 PhysX USD:", SO101_PHYSX_USDA, "spec:", spec.version)
	print("  mesh_colliders patched:", mesh_colliders)
	print("  joints with angular drive:", joints)
	print("  attached wrist_cam:", cam_path)
	return SO101_PHYSX_USDA


def main() -> None:
	spec = load_physics_spec()
	build_so101_physx_usd()
	# Green plate: dark saturated forest green
	build_obj_prop_usd(
		SIM_ASSETS / "plate_200mm.obj",
		PROPS_DIR / "green_plate.usda",
		"GreenPlate",
		rgb=(0.04, 0.28, 0.11),
		mass=spec.plate.mass_kg,
		static_friction=spec.plate.static_friction,
		dynamic_friction=spec.plate.dynamic_friction,
		restitution=spec.plate.restitution,
		approx="convexDecomposition",
		roughness=0.28,
		rigid_body=True,
		kinematic=True,
	)
	# Pink plate: pastel pink plastic
	build_obj_prop_usd(
		SIM_ASSETS / "plate_200mm.obj",
		PROPS_DIR / "pink_plate.usda",
		"PinkPlate",
		rgb=(0.78, 0.36, 0.45),
		mass=spec.plate.mass_kg,
		static_friction=spec.plate.static_friction,
		dynamic_friction=spec.plate.dynamic_friction,
		restitution=spec.plate.restitution,
		approx="convexDecomposition",
		roughness=0.28,
		rigid_body=True,
		kinematic=True,
	)
	# All Duplo sizes (normal + defective) from unified PhysicsSpec
	for size_key in ("2x2", "2x3", "2x4"):
		bspec = spec.bricks[size_key]
		obj_path = SIM_ASSETS / f"duplo_{size_key}.obj"
		build_obj_prop_usd(
			obj_path,
			PROPS_DIR / f"duplo_{size_key}_orange.usda",
			"Brick",
			rgb=(0.82, 0.24, 0.04),
			mass=bspec.mass_kg,
			static_friction=bspec.static_friction,
			dynamic_friction=bspec.dynamic_friction,
			restitution=bspec.restitution,
			approx="convexHull",
			roughness=0.20,
			rigid_body=True,
			kinematic=False,
			brick_size=bspec.size_m,
		)
		build_obj_prop_usd(
			obj_path,
			PROPS_DIR / f"duplo_{size_key}_defective.usda",
			"Brick",
			rgb=(0.82, 0.24, 0.04),
			mass=bspec.mass_kg,
			static_friction=bspec.static_friction,
			dynamic_friction=bspec.dynamic_friction,
			restitution=bspec.restitution,
			approx="convexHull",
			roughness=0.20,
			rigid_body=True,
			kinematic=False,
			defect_face=0,
			brick_size=bspec.size_m,
		)
	# Dynamic clutter brick prop (2x2 blue) for bowl clutter interactions
	c_spec = spec.bricks["2x2"]
	build_obj_prop_usd(
		SIM_ASSETS / "duplo_2x2.obj",
		PROPS_DIR / "clutter_brick.usda",
		"ClutterBrick",
		rgb=(0.10, 0.42, 0.80),
		mass=c_spec.mass_kg,
		static_friction=c_spec.static_friction,
		dynamic_friction=c_spec.dynamic_friction,
		restitution=c_spec.restitution,
		approx="convexHull",
		roughness=0.22,
		rigid_body=True,
		kinematic=False,
		brick_size=c_spec.size_m,
	)
	print("Props generated under:", PROPS_DIR, "using physics spec:", spec.version)


if __name__ == "__main__":
	main()
