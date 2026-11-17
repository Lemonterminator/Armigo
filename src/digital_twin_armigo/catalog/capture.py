"""Read-only frame capture. Playback transforms are geometry-local to world, metres, xyzw."""

from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path

import numpy as np

from digital_twin_armigo.runtime.canonical import sha256_file
from digital_twin_armigo.runtime.snapshot import hash_tree

from .store import dumps, now, plain


def identity(root: Path, engine: str, version: str) -> dict:
	def git(*args):
		r = subprocess.run(
			["git", "-C", str(root), *args],
			capture_output=True,
			text=True,
			encoding="utf-8",
			errors="replace",
			check=False,
		)
		return r.stdout.strip() if r.returncode == 0 else None

	return {
		"engine": engine,
		"engine_version": version,
		"python": platform.python_version(),
		"numpy": np.__version__,
		"git_commit": git("rev-parse", "HEAD"),
		"dirty": bool(git("status", "--porcelain")),
		"code_sha256": hash_tree(root, ["src/**/*.py", "scripts/**/*.py"]),
		"config_sha256": hash_tree(root, ["config/sim/**/*"]),
		"assets_sha256": hash_tree(
			root,
			["assets/so101_mjcf/**/*", "assets/so101_usd/**/*", "assets/props/**/*", "assets/meshes/**/*"],
			exclude=["**/_sorting_cell_*.xml", "**/_marker_*.png"],
		),
	}


class MotionCapture:
	def __init__(self, scene: dict):
		self.scene = plain(scene)
		self.frames = []
		self.times = []
		self.jaw_env = []
		self.camera_frames = []

	def append(self, poses, time_s, jaw_env=None):
		poses = np.asarray(poses, dtype=np.float64)
		if poses.shape != (len(self.scene["nodes"]), 7) or not np.isfinite(poses).all():
			raise ValueError("invalid complete-scene pose frame")
		if not np.allclose(np.linalg.norm(poses[:, 3:], axis=1), 1, atol=1e-4):
			raise ValueError("non-unit xyzw orientation")
		self.frames.append(poses.copy())
		self.times.append(float(time_s))
		self.jaw_env.append(jaw_env)

	def save(
		self,
		out: Path,
		telemetry: dict,
		result: dict,
		provenance: dict,
		alias: str,
		engine: str,
		parent=None,
		artifacts=None,
	):
		if not self.frames:
			raise ValueError("no recorded motion")
		out = Path(out)
		out.mkdir(parents=True, exist_ok=False)
		(out / "scene.json").write_text(
			dumps({"schema": "catalog-scene/v1", "units": "m", "up_axis": "Z", "quaternion": "xyzw", **self.scene}),
			encoding="utf-8",
		)
		np.savez_compressed(
			out / "motion.npz",
			poses=np.stack(self.frames),
			timestamps=np.asarray(self.times),
			node_ids=np.array([n["id"] for n in self.scene["nodes"]]),
			**({"camera_poses": np.stack(self.camera_frames)} if self.camera_frames else {}),
		)
		np.savez_compressed(out / "telemetry.npz", **{k: v for k, v in telemetry.items() if v is not None})
		manifest = {
			"schema": "catalog-episode/v1",
			"alias": alias,
			"engine": engine,
			"scene": self.scene,
			"result": result,
			"provenance": {
				**provenance,
				"recorded_at": now(),
				"sampling": "post-control-frame",
				"motion_contract": "world geometry poses; xyzw; metres",
			},
			"parent_episode_id": parent,
			"artifacts": {
				"scene": "scene.json",
				"motion": "motion.npz",
				"telemetry": "telemetry.npz",
				**(artifacts or {}),
			},
		}
		manifest["provenance"]["artifact_hashes"] = {
			k: sha256_file(out / v) for k, v in manifest["artifacts"].items() if (out / v).is_file()
		}
		(out / "episode.json").write_text(dumps(manifest), encoding="utf-8")
		return out / "episode.json"


class MujocoCapture(MotionCapture):
	def __init__(self, cell):
		from digital_twin_armigo.mujoco_cell._mj import mujoco

		m = cell.model
		self.geoms = [g for g in range(m.ngeom) if int(m.geom_group[g]) not in (3, 4)]
		nodes = []
		meshes = {}
		for g in self.geoms:
			type_ = mujoco.mjtGeom(int(m.geom_type[g])).name.removeprefix("mjGEOM_").lower()
			mat = int(m.geom_matid[g])
			color = m.mat_rgba[mat] if mat >= 0 else m.geom_rgba[g]
			node = {
				"id": f"geom_{g}",
				"name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom_{g}",
				"type": type_,
				"size": m.geom_size[g].tolist(),
				"color": color.tolist(),
			}
			if type_ == "mesh":
				idx = int(m.geom_dataid[g])
				key = str(idx)
				node["mesh"] = key
				if key not in meshes:
					va = int(m.mesh_vertadr[idx])
					vn = int(m.mesh_vertnum[idx])
					fa = int(m.mesh_faceadr[idx])
					fn = int(m.mesh_facenum[idx])
					meshes[key] = {
						"vertices": m.mesh_vert[va : va + vn].tolist(),
						"faces": m.mesh_face[fa : fa + fn].tolist(),
					}
			nodes.append(node)
		super().__init__(
			{"nodes": nodes, "meshes": meshes, "layout": plain(cell.layout), "fps": 30, "source_engine": "mujoco"}
		)
		self.cameras = [i for i in range(m.ncam) if m.camera(i).name in ("top", "wrist_cam")]
		self.scene["cameras"] = [{"name": m.camera(i).name, "fovy": float(m.cam_fovy[i])} for i in self.cameras]
		self.mj_qpos = []

	def frame(self, cell):
		from digital_twin_armigo.mujoco_cell._mj import mujoco

		poses = np.empty((len(self.geoms), 7))
		for i, g in enumerate(self.geoms):
			poses[i, :3] = cell.data.geom_xpos[g]
			quat = np.empty(4)
			mujoco.mju_mat2Quat(quat, cell.data.geom_xmat[g])
			poses[i, 3:] = quat[[1, 2, 3, 0]]
		self.append(poses, cell.data.time)
		cameras = []
		for i in self.cameras:
			q = np.empty(4)
			mujoco.mju_mat2Quat(q, cell.data.cam_xmat[i])
			cameras.append([*cell.data.cam_xpos[i], *q[[1, 2, 3, 0]]])
		if cameras:
			self.camera_frames.append(np.asarray(cameras))
		self.mj_qpos.append(cell.data.qpos.copy())

	def save_mujoco(self, out, cell, telemetry, result, provenance, alias, parent=None):
		# Compiled MJCF includes the full scene; assets keep absolute paths in the model.
		from digital_twin_armigo.mujoco_cell._mj import mujoco

		out = Path(out)
		manifest = self.save(
			out, {**telemetry, "full_qpos": np.asarray(self.mj_qpos)}, result, provenance, alias, "mujoco", parent
		)
		mujoco.mj_saveLastXML(str(out / "scene.xml"), cell.model)
		# mj_saveLastXML keeps the original mesh directory relative to its old location.
		import xml.etree.ElementTree as ET

		from digital_twin_armigo.constants import SO101_MJCF_DIR

		tree = ET.parse(out / "scene.xml")
		compiler = tree.getroot().find("compiler")
		import shutil

		resources = []
		for tag, attr in (("mesh", "meshdir"), ("texture", "texturedir")):
			dir_ = Path(compiler.get(attr, "")) if compiler is not None else Path()
			if not dir_.is_absolute():
				dir_ = SO101_MJCF_DIR / dir_
			for asset in tree.getroot().findall("./asset/" + tag):
				file = asset.get("file")
				if not file:
					continue
				source = Path(file) if Path(file).is_absolute() else dir_ / file
				name = "resources/" + sha256_file(source) + source.suffix
				(out / "resources").mkdir(exist_ok=True)
				shutil.copyfile(source, out / name)
				asset.set("file", name)
				resources.append(name)
			if compiler is not None:
				compiler.attrib.pop(attr, None)
		tree.write(out / "scene.xml", encoding="unicode")
		d = json.loads(manifest.read_text(encoding="utf-8"))
		d["artifacts"]["mjcf"] = "scene.xml"
		for i, name in enumerate(sorted(set(resources))):
			d["artifacts"][f"resource_{i:03d}"] = name
		d["provenance"]["artifact_hashes"]["mjcf"] = sha256_file(out / "scene.xml")
		manifest.write_text(dumps(d), encoding="utf-8")
		return manifest


class IsaacCapture(MotionCapture):
	"""Capture USD geometry once, then its complete world transforms after each frame."""

	def __init__(self, stage, layout, live_scene):
		from pxr import UsdGeom, UsdShade

		self.prims = []
		nodes = []
		meshes = {}
		for prim in stage.Traverse():
			if not prim.IsA(UsdGeom.Gprim) or "collision" in str(prim.GetPath()).lower():
				continue
			if UsdGeom.Imageable(prim).ComputeVisibility() == "invisible":
				continue
			node = {
				"id": f"geom_{len(nodes)}",
				"name": str(prim.GetPath()),
				"type": prim.GetTypeName().lower(),
				"size": [1, 1, 1],
				"color": [0.6, 0.6, 0.6, 1],
			}
			if prim.IsA(UsdGeom.Mesh):
				mesh = UsdGeom.Mesh(prim)
				counts = list(mesh.GetFaceVertexCountsAttr().Get() or [])
				inds = list(mesh.GetFaceVertexIndicesAttr().Get() or [])
				faces = []
				off = 0
				for count in counts:
					for j in range(1, count - 1):
						faces.append([inds[off], inds[off + j], inds[off + j + 1]])
					off += count
				meshes[node["id"]] = {"vertices": [list(v) for v in mesh.GetPointsAttr().Get()], "faces": faces}
				node["mesh"] = node["id"]
			elif prim.IsA(UsdGeom.Cube):
				node["type"] = "box"
				node["size"] = [float(UsdGeom.Cube(prim).GetSizeAttr().Get()) / 2] * 3
			elif prim.IsA(UsdGeom.Sphere):
				node["size"][0] = float(UsdGeom.Sphere(prim).GetRadiusAttr().Get())
			elif prim.GetTypeName() in ("Cylinder", "Capsule"):
				node["size"] = [
					float(prim.GetAttribute("radius").Get()),
					float(prim.GetAttribute("height").Get()) / 2,
					0,
				]
			else:
				continue
			material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
			if material:
				for shaderprim in material.GetPrim().GetChildren():
					shader = UsdShade.Shader(shaderprim)
					attr = shader.GetInput("diffuseColor")
					if attr and attr.Get() is not None:
						node["color"] = [*map(float, attr.Get()), 1]
			self.prims.append(prim)
			nodes.append(node)
		super().__init__({"nodes": nodes, "meshes": meshes, "layout": layout, "fps": 30, "source_engine": "isaac"})
		self.cameras = [
			p for p in stage.Traverse() if p.IsA(UsdGeom.Camera) and p.GetName() in ("TopCamera", "wrist_cam")
		][:2]
		self.scene["cameras"] = []
		for p in self.cameras:
			cam = UsdGeom.Camera(p)
			f = float(cam.GetFocalLengthAttr().Get())
			aperture = float(cam.GetVerticalApertureAttr().Get())
			self.scene["cameras"].append(
				{"name": p.GetName(), "fovy": float(np.degrees(2 * np.arctan(aperture / (2 * f))))}
			)
		self.scales = None
		# Isaac Lab backends can keep USD transforms stale even with use_fabric=False.
		# Bind each visual/camera to its live rigid body and freeze only the geometry-local offset.
		import re

		from pxr import UsdPhysics

		self.bindings = {}
		bodies = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
		for asset in live_scene.articulations.values():
			for i, name in enumerate(asset.body_names):
				matches = [
					p
					for p in bodies
					if p.GetName() == name and re.fullmatch(asset.cfg.prim_path + "/.*", str(p.GetPath()))
				]
				if len(matches) != 1:
					raise ValueError(f"ambiguous live articulation body {name}")
				self.bindings[str(matches[0].GetPath())] = (asset, i)
		for asset in live_scene.rigid_objects.values():
			matches = [p for p in bodies if re.fullmatch(asset.cfg.prim_path + "(?:/.*)?", str(p.GetPath()))]
			if len(matches) != 1:
				raise ValueError(f"ambiguous live rigid object {asset.cfg.prim_path}")
			self.bindings[str(matches[0].GetPath())] = (asset, None)
		cache = UsdGeom.XformCache()
		self.offsets = {}
		for prim in [*self.prims, *self.cameras]:
			ancestor = prim
			while ancestor and str(ancestor.GetPath()) not in self.bindings:
				ancestor = ancestor.GetParent()
			world = cache.GetLocalToWorldTransform(prim)
			key = str(ancestor.GetPath()) if ancestor else None
			self.offsets[str(prim.GetPath())] = (
				key,
				world * cache.GetLocalToWorldTransform(ancestor).GetInverse() if key else world,
			)
		self.scene["pose_source"] = "live_isaaclab_body_buffers_with_usd_geometry_offsets"
		self.max_reference_error = 0.0

	def frame(self, time_s, expected_brick_pos=None):
		from pxr import Gf

		live = {}
		for key, (asset, index) in self.bindings.items():
			if index is None:
				pos = asset.data.root_pos_w.torch[0].cpu().numpy()
				quat = asset.data.root_quat_w.torch[0].cpu().numpy()
			else:
				pos = asset.data.body_pos_w.torch[0, index].cpu().numpy()
				quat = asset.data.body_quat_w.torch[0, index].cpu().numpy()
			matrix = Gf.Matrix4d().SetRotate(Gf.Quatd(float(quat[3]), Gf.Vec3d(*map(float, quat[:3]))))
			matrix.SetTranslateOnly(Gf.Vec3d(*map(float, pos)))
			live[key] = matrix
			if key.endswith("/Brick") and expected_brick_pos is not None:
				self.max_reference_error = max(
					self.max_reference_error, float(np.max(np.abs(np.asarray(pos) - expected_brick_pos)))
				)
		if self.max_reference_error > 1e-5:
			raise ValueError("recorded rigid body differs from live brick telemetry")

		def world(prim):
			key, offset = self.offsets[str(prim.GetPath())]
			return offset * live[key] if key else offset

		poses = []
		scales = []
		for prim in self.prims:
			matrix = world(prim)

			transform = Gf.Transform(matrix)
			q = transform.GetRotation().GetQuat()
			poses.append([*transform.GetTranslation(), *q.GetImaginary(), q.GetReal()])
			scales.append(list(transform.GetScale()))
		if self.scales is None:
			self.scales = scales
			for n, s in zip(self.scene["nodes"], scales, strict=True):
				n["scale"] = s
		elif not np.allclose(scales, self.scales, atol=1e-6):
			raise ValueError("animated geometry scale is unsupported")
		self.append(poses, time_s)
		cameras = []
		for prim in self.cameras:
			transform = Gf.Transform(world(prim))
			q = transform.GetRotation().GetQuat()
			cameras.append([*transform.GetTranslation(), *q.GetImaginary(), q.GetReal()])
		if cameras:
			self.camera_frames.append(np.asarray(cameras))
