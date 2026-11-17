"""Re-render the crops of a ``render_dataset.py --dump-poses`` dataset with Isaac Sim RTX path tracing.

Run with the Isaac Python (``E:\\il\\v\\Scripts\\python.exe``). For every crop of the MuJoCo dataset the recorded world
poses of the rendered geoms and of the camera are replayed in a USD stage (flat-coloured geometry, no physics), a full
640x480 frame is path traced, and the SAME square crop (bbox, pad, clamp, size) is cut out, so crop ``png/NAME.png`` of
the output is the Isaac twin of crop ``png/NAME.png`` of the input: same layout, pose, camera, label, metadata.

Differences from the MuJoCo render, by design: path-traced lighting and shadows instead of rasterised Phong,
UsdPreviewSurface materials, no textures (the marker and floor textures become flat colours).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

WIDTH, HEIGHT = 640, 480
PAD = 0.25


def square_crop(img: np.ndarray, bbox, pad: float, size: int, clamp: bool) -> np.ndarray:
	"""Identical to ``mujoco_cell.inspection_views.square_crop`` (copied: that module imports MuJoCo)."""
	x, y, w, h = bbox
	side = max(w, h) * (1.0 + 2.0 * pad)
	cx, cy = x + w / 2.0, y + h / 2.0
	xs = np.round(cx - side / 2.0 + (np.arange(size) + 0.5) * side / size).astype(int)
	ys = np.round(cy - side / 2.0 + (np.arange(size) + 0.5) * side / size).astype(int)
	if clamp:
		if side <= img.shape[1]:
			xs = xs + max(0, -int(xs.min())) - max(0, int(xs.max()) - (img.shape[1] - 1))
		if side <= img.shape[0]:
			ys = ys + max(0, -int(ys.min())) - max(0, int(ys.max()) - (img.shape[0] - 1))
	xs = np.clip(xs, -1, img.shape[1])
	ys = np.clip(ys, -1, img.shape[0])
	out = np.zeros((size, size, img.shape[2]), dtype=img.dtype)
	vx = (xs >= 0) & (xs < img.shape[1])
	vy = (ys >= 0) & (ys < img.shape[0])
	out[np.ix_(vy, vx)] = img[np.ix_(ys[vy], xs[vx])]
	return out


def main() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument("--data", type=Path, required=True, help="dataset dir with poses/, meshes/, metadata.jsonl")
	ap.add_argument("--out", type=Path, required=True)
	ap.add_argument("--spp", type=int, default=64)
	ap.add_argument("--limit-layouts", type=int, default=0)
	ap.add_argument("--only-layouts", type=int, nargs="*", default=None)
	ap.add_argument("--debug-frames", action="store_true", help="also save the first full frame of each layout")
	ap.add_argument("--dome", type=float, default=650.0, help="dome light intensity")
	ap.add_argument("--key", type=float, default=3500.0, help="rect key light intensity")
	ap.add_argument("--shard", type=int, nargs=2, default=(0, 1), metavar=("I", "N"), help="render layouts i, i+N, ...")
	ap.add_argument("--image-size", type=int, default=256)
	a = ap.parse_args()
	rows = [json.loads(line) for line in (a.data / "metadata.jsonl").read_text(encoding="utf-8").splitlines() if line]
	by_path = {r["path"]: r for r in rows}
	files = sorted((a.data / "poses").glob("*.npz"))
	if a.only_layouts is not None:
		files = [f for f in files if int(f.stem) in set(a.only_layouts)]
	files = files[a.shard[0] :: a.shard[1]]
	if a.limit_layouts:
		files = files[: a.limit_layouts]
	(a.out / "png").mkdir(parents=True, exist_ok=True)
	os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "yes")
	from isaacsim import SimulationApp

	app = SimulationApp({"headless": True, "renderer": "PathTracing", "width": WIDTH, "height": HEIGHT})
	try:
		import carb
		import cv2
		import omni.replicator.core as rep
		import omni.usd
		from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade, Vt

		settings = carb.settings.get_settings()
		settings.set("/rtx/rendermode", "PathTracing")
		settings.set("/rtx/pathtracing/spp", 32)
		settings.set("/rtx/pathtracing/totalSpp", a.spp)
		settings.set("/omni/replicator/captureOnPlay", False)
		rep.orchestrator.set_capture_on_play(False)
		mesh_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
		t_start = time.perf_counter()
		n_done = 0
		for f in files:
			with np.load(f, allow_pickle=False) as z:
				poses, cams, paths = z["poses"], z["cams"], z["paths"].tolist()
				nodes, fovy = json.loads(str(z["nodes"])), json.loads(str(z["fovy"]))
			todo = [i for i, p in enumerate(paths) if not (a.out / p).exists()]
			if not todo:
				continue
			omni.usd.get_context().new_stage()
			stage = omni.usd.get_context().get_stage()
			UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
			UsdGeom.SetStageMetersPerUnit(stage, 1)
			xforms = []
			for node in nodes:
				path = "/World/" + node["id"]
				kind, size = node["type"], node["size"]
				xform = UsdGeom.Xform.Define(stage, path)
				xform.AddTranslateOp()
				xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
				xform.AddScaleOp()
				scale = [1.0, 1.0, 1.0]
				if kind == "mesh":
					if node["mesh"] not in mesh_cache:
						with np.load(a.data / "meshes" / f"{node['mesh']}.npz") as m:
							mesh_cache[node["mesh"]] = (m["vertices"], m["faces"])
					verts, faces = mesh_cache[node["mesh"]]
					shape = UsdGeom.Mesh.Define(stage, path + "/Shape")
					shape.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(verts))
					shape.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, dtype=np.int32)))
					shape.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.reshape(-1)))
					shape.CreateSubdivisionSchemeAttr("none")
					shape.CreateDoubleSidedAttr(True)
				elif kind in ("box", "plane"):
					shape = UsdGeom.Cube.Define(stage, path + "/Shape")
					shape.CreateSizeAttr(2)
					scale = size[:3] if kind == "box" else [max(size[0], 2), max(size[1], 2), 0.002]
				elif kind == "cylinder":
					shape = UsdGeom.Cylinder.Define(stage, path + "/Shape")
					shape.CreateRadiusAttr(size[0])
					shape.CreateHeightAttr(2 * size[1])
					shape.CreateAxisAttr("Z")
				elif kind == "sphere":
					shape = UsdGeom.Sphere.Define(stage, path + "/Shape")
					shape.CreateRadiusAttr(size[0])
				else:
					raise ValueError(f"unsupported geometry {kind}")
				xform.GetOrderedXformOps()[2].Set(Gf.Vec3d(*scale))
				material = UsdShade.Material.Define(stage, path + "/Material")
				shader = UsdShade.Shader.Define(stage, path + "/Material/Shader")
				shader.CreateIdAttr("UsdPreviewSurface")
				shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*node["color"][:3]))
				shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(
					0.32 if node["name"].startswith(("brick", "defect")) else 0.48
				)
				material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
				UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(material)
				xforms.append(xform.GetOrderedXformOps())
			UsdLux.DomeLight.Define(stage, "/World/Dome").CreateIntensityAttr(a.dome)
			key = UsdLux.RectLight.Define(stage, "/World/Key")
			key.CreateIntensityAttr(a.key)
			key.CreateWidthAttr(1.2)
			key.CreateHeightAttr(1.2)
			UsdGeom.Xformable(key).AddTranslateOp().Set(Gf.Vec3d(0.2, -0.3, 1.5))
			cam = UsdGeom.Camera.Define(stage, "/World/Cam")
			cam.CreateFocalLengthAttr(20.955)
			cam.CreateClippingRangeAttr(Gf.Vec2f(0.005, 100))
			cxf = UsdGeom.Xformable(cam)
			cxf.AddTranslateOp()
			cxf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
			cops = cxf.GetOrderedXformOps()
			product = rep.create.render_product(str(cam.GetPath()), (WIDTH, HEIGHT))
			rgb = rep.AnnotatorRegistry.get_annotator("rgb")
			rgb.attach(product)
			for _ in range(8):
				app.update()
			for i in todo:
				r = by_path[paths[i]]
				for op, p in zip(xforms, poses[i], strict=True):
					op[0].Set(Gf.Vec3d(*p[:3]))
					op[1].Set(Gf.Quatd(float(p[6]), Gf.Vec3d(*p[3:6])))
				cp = cams[i]
				cops[0].Set(Gf.Vec3d(*cp[:3]))
				cops[1].Set(Gf.Quatd(float(cp[6]), Gf.Vec3d(*cp[3:6])))
				vert_ap = float(2 * 20.955 * np.tan(np.deg2rad(fovy[r["camera"]]) / 2))
				cam.CreateVerticalApertureAttr(vert_ap)
				cam.CreateHorizontalApertureAttr(vert_ap * WIDTH / HEIGHT)
				rep.orchestrator.step(rt_subframes=max(1, a.spp // 32), delta_time=0.0)
				frame = np.asarray(rgb.get_data())[:, :, :3]
				crop = square_crop(frame, r["bbox"], PAD, a.image_size, clamp=r["task"] == "cross_var")
				if not cv2.imwrite(str(a.out / paths[i]), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)):
					raise RuntimeError("PNG write failed")
				if a.debug_frames and r["stage"] in ("closeup", "show"):
					dbg = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
					bx, by, bw, bh = r["bbox"]
					cv2.rectangle(dbg, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)
					cv2.imwrite(str(a.out / f"debug_{Path(paths[i]).stem}.png"), dbg)
				n_done += 1
			print(f"layout {f.stem}: {len(todo)} crops ({n_done} total, {time.perf_counter() - t_start:.0f}s)", flush=True)
		for name in ("metadata.jsonl", "layouts.jsonl", "summary.json"):
			if (a.data / name).exists():
				shutil.copyfile(a.data / name, a.out / name)
	except BaseException:
		import traceback

		traceback.print_exc()
		sys.stderr.flush()
		raise
	finally:
		app.close()


if __name__ == "__main__":
	sys.exit(main())
