"""Offline RTX render of recorded geometry poses. No physics or policy is executed."""

from __future__ import annotations

import argparse
import json
import os
import sys
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main():
	p = argparse.ArgumentParser()
	p.add_argument("--request", required=True, type=Path)
	p.add_argument("--out", required=True, type=Path)
	p.add_argument("--max-frames", type=int, default=0)
	p.add_argument("--start-frame", type=int, default=0)
	p.add_argument(
		"--stride", type=int, default=1, help="Render every n-th recorded frame; the video plays at fps / n."
	)
	a = p.parse_args()
	request = json.loads(a.request.read_text(encoding="utf-8"))
	e = request["episode"]
	roles = {
		r["role"]: Path(r["path"]) if Path(r["path"]).is_absolute() else a.request.parent / r["path"]
		for r in e["artifacts"]
	}
	import numpy as np

	from digital_twin_armigo.catalog.playback import encode_video, load_motion

	scene, motion, times = load_motion(roles)
	if a.start_frame < 0 or a.start_frame >= len(motion):
		raise ValueError("start frame out of range")
	trace = {}
	if "telemetry" in roles:
		with np.load(roles["telemetry"], allow_pickle=False) as z:
			trace = {k: z[k] for k in ("phases", "jaw_forces", "brick_vel") if k in z}
	captions = json.loads(roles["captions"].read_text(encoding="utf-8")) if "captions" in roles else None
	if captions is not None and len(captions) != len(motion):
		raise ValueError("captions do not match motion")
	quality = request.get("profile") == "quality"
	width, height = (1920, 1080) if quality else (1280, 720)
	spp = 128 if quality else 32
	a.out.mkdir(parents=True, exist_ok=False)
	os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "yes")
	from isaacsim import SimulationApp

	app = SimulationApp({"headless": True, "renderer": "PathTracing", "width": width, "height": height})
	try:
		import carb
		import omni.replicator.core as rep
		import omni.usd
		from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade, Vt

		omni.usd.get_context().new_stage()
		stage = omni.usd.get_context().get_stage()
		UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
		UsdGeom.SetStageMetersPerUnit(stage, 1)
		ops = []
		meshes = scene.get("meshes", {})
		for node in scene["nodes"]:
			path = "/World/" + node["id"]
			kind = node["type"]
			size = node["size"]
			xform = UsdGeom.Xform.Define(stage, path)
			xform.AddTranslateOp()
			xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
			xform.AddScaleOp()
			scale = list(node.get("scale", [1, 1, 1]))
			if kind == "mesh":
				shape = UsdGeom.Mesh.Define(stage, path + "/Shape")
				m = meshes[node["mesh"]]
				shape.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*v) for v in m["vertices"]]))
				shape.CreateFaceVertexCountsAttr([3] * len(m["faces"]))
				shape.CreateFaceVertexIndicesAttr([i for face in m["faces"] for i in face])
				shape.CreateSubdivisionSchemeAttr("none")
				shape.CreateDoubleSidedAttr(True)
			elif kind in ("box", "plane"):
				shape = UsdGeom.Cube.Define(stage, path + "/Shape")
				shape.CreateSizeAttr(2)
				scale = (
					[v * s for v, s in zip(scale, size)] if kind == "box" else [max(size[0], 2), max(size[1], 2), 0.002]
				)
			elif kind == "sphere":
				shape = UsdGeom.Sphere.Define(stage, path + "/Shape")
				shape.CreateRadiusAttr(size[0])
			elif kind in ("cylinder", "capsule"):
				shape = (UsdGeom.Cylinder if kind == "cylinder" else UsdGeom.Capsule).Define(stage, path + "/Shape")
				shape.CreateRadiusAttr(size[0])
				shape.CreateHeightAttr(2 * size[1])
				shape.CreateAxisAttr("Z")
			elif kind == "ellipsoid":
				shape = UsdGeom.Sphere.Define(stage, path + "/Shape")
				shape.CreateRadiusAttr(1)
				scale = [v * s for v, s in zip(scale, size)]
			else:
				raise ValueError(f"unsupported geometry {kind}")
			xform.GetOrderedXformOps()[2].Set(Gf.Vec3d(*scale))
			material = UsdShade.Material.Define(stage, path + "/Material")
			shader = UsdShade.Shader.Define(stage, path + "/Material/Shader")
			shader.CreateIdAttr("UsdPreviewSurface")
			shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*node["color"][:3]))
			shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(
				0.32 if "brick" in node["name"].lower() else 0.48
			)
			material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
			UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(material)
			ops.append(xform.GetOrderedXformOps())
		light = UsdLux.DomeLight.Define(stage, "/World/Dome")
		light.CreateIntensityAttr(650)
		key = UsdLux.RectLight.Define(stage, "/World/Key")
		key.CreateIntensityAttr(3500)
		key.CreateWidthAttr(1.2)
		key.CreateHeightAttr(1.2)
		UsdGeom.Xformable(key).AddTranslateOp().Set(Gf.Vec3d(0.2, -0.3, 1.5))
		# Rect light local -Z points down. Fixed calibrated geometry stays unchanged.
		camera = UsdGeom.Camera.Define(stage, "/World/HeroCamera")
		camera.CreateFocalLengthAttr(27)
		camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100))
		eye = Gf.Vec3d(0.95, -0.85, 0.70)
		target = Gf.Vec3d(0.23, 0, 0.12)
		matrix = Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(0, 0, 1)).GetInverse()
		UsdGeom.Xformable(camera).AddTransformOp().Set(matrix)
		settings = carb.settings.get_settings()
		settings.set("/rtx/rendermode", "PathTracing")
		settings.set("/rtx/pathtracing/spp", 32)
		settings.set("/rtx/pathtracing/totalSpp", spp)
		settings.set("/omni/replicator/captureOnPlay", False)
		rep.orchestrator.set_capture_on_play(False)
		rp = rep.create.render_product(str(camera.GetPath()), (width, height))
		rgb = rep.AnnotatorRegistry.get_annotator("rgb")
		rgb.attach(rp)
		insets = []
		with np.load(roles["motion"], allow_pickle=False) as z:
			camera_poses = z.get("camera_poses", None)
		for i, definition in enumerate(scene.get("cameras", [])[:2]):
			if camera_poses is None:
				break
			cam = UsdGeom.Camera.Define(stage, f"/World/Inset{i}")
			cam.CreateFocalLengthAttr(20.955)
			cam.CreateVerticalApertureAttr(float(2 * 20.955 * np.tan(np.deg2rad(definition["fovy"]) / 2)))
			cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100))
			xf = UsdGeom.Xformable(cam)
			xf.AddTranslateOp()
			xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
			product = rep.create.render_product(str(cam.GetPath()), (320, 240))
			annotator = rep.AnnotatorRegistry.get_annotator("rgb")
			annotator.attach(product)
			insets.append((xf.GetOrderedXformOps(), annotator, definition["name"]))
		for _ in range(20):
			app.update()
		max_error = 0.0
		frames = range(a.start_frame, len(motion), max(1, a.stride))[: a.max_frames or None]
		for output_frame, frame in enumerate(frames):
			poses = motion[frame]
			for transform, pose in zip(ops, poses, strict=True):
				transform[0].Set(Gf.Vec3d(*pose[:3]))
				transform[1].Set(Gf.Quatd(float(pose[6]), Gf.Vec3d(*pose[3:6])))
			for i, (transform, _, _) in enumerate(insets):
				pose = camera_poses[frame, i]
				transform[0].Set(Gf.Vec3d(*pose[:3]))
				transform[1].Set(Gf.Quatd(float(pose[6]), Gf.Vec3d(*pose[3:6])))
			# delta_time=0: sampling never advances the experiment or physics.
			rep.orchestrator.step(rt_subframes=max(1, spp // 32), delta_time=0.0)
			image = np.asarray(rgb.get_data())[:, :, :3]
			import cv2

			image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
			cv2.rectangle(image, (0, 0), (width, 78), (25, 38, 34), -1)
			cv2.putText(
				image,
				f"{e['alias']} | {e['verdict']} | t={times[frame] - times[0]:.3f}s",
				(20, 35),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.65,
				(245, 245, 245),
				2,
			)
			from digital_twin_armigo.constants import PHASES

			caption = []
			if "phases" in trace:
				phase = int(trace["phases"][frame])
				caption.append("phase: " + (PHASES[phase] if 0 <= phase < len(PHASES) else "UNKNOWN"))
			if "brick_vel" in trace:
				caption.append(f"speed: {np.linalg.norm(trace['brick_vel'][frame]):.3f} m/s")
			if "jaw_forces" in trace:
				caption.append("jaw total: " + " / ".join(f"{v:.2f}" for v in trace["jaw_forces"][frame]) + " N")
			cv2.putText(image, " | ".join(caption), (20, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 235, 225), 1)
			if captions is not None:
				cv2.rectangle(image, (0, height - 44), (width - 344, height), (25, 38, 34), -1)
				cv2.putText(
					image, captions[frame], (20, height - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 230, 160), 1
				)
			for i, (_, annotator, name) in enumerate(insets):
				tile = cv2.cvtColor(np.asarray(annotator.get_data())[:, :, :3], cv2.COLOR_RGB2BGR)
				x = width - 332
				y = height - 252 - i * 252
				image[y : y + 240, x : x + 320] = tile
				cv2.rectangle(image, (x, y), (x + 320, y + 26), (25, 38, 34), -1)
				cv2.putText(image, name, (x + 8, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
			if not cv2.imwrite(str(a.out / f"frame_{output_frame:06d}.png"), image):
				raise RuntimeError("PNG write failed")
			cache = UsdGeom.XformCache()
			for node, pose in zip(scene["nodes"], poses, strict=True):
				actual = cache.GetLocalToWorldTransform(
					stage.GetPrimAtPath("/World/" + node["id"])
				).ExtractTranslation()
				max_error = max(max_error, float(np.max(np.abs(np.array(actual) - pose[:3]))))
			print(json.dumps({"frame": frame + 1, "total": len(motion)}), flush=True)
		stage.GetRootLayer().Export(str(a.out / "replay.usda"))
		encode_video(a.out, float(scene.get("fps", 30)) / max(1, a.stride))
		(a.out / "render.json").write_text(
			json.dumps(
				{
					"source_engine": e["engine"],
					"renderer": "Isaac RTX PathTracing",
					"isaac_version": version("isaacsim"),
					"physics_advanced": False,
					"source_frames": len(motion),
					"rendered_frames": len(frames),
					"stride": max(1, a.stride),
					"source_start_frame": a.start_frame,
					"spp": spp,
					"resolution": [width, height],
					"max_position_error_m": max_error,
				},
				indent=2,
			)
		)
	finally:
		app.close()


if __name__ == "__main__":
	main()
