"""Validated recorded-state playback and portable video encoding."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np


def load_motion(roles):
	scene = json.loads(Path(roles["scene"]).read_text(encoding="utf-8"))
	with np.load(roles["motion"], allow_pickle=False) as data:
		poses = data["poses"]
		times = data["timestamps"]
		names = data["node_ids"].tolist()
	if scene.get("units") != "m" or scene.get("quaternion") != "xyzw" or scene.get("up_axis") != "Z":
		raise ValueError("unsupported coordinate contract")
	if (
		names != [n["id"] for n in scene["nodes"]]
		or poses.shape != (len(times), len(names), 7)
		or not len(times)
		or not np.isfinite(poses).all()
		or not np.isfinite(times).all()
	):
		raise ValueError("invalid motion arrays or object map")
	if len(times) > 1 and np.any(np.diff(times) <= 0):
		raise ValueError("timestamps must increase")
	if not np.allclose(np.linalg.norm(poses[:, :, 3:], axis=2), 1, atol=1e-4):
		raise ValueError("invalid quaternions")
	return scene, poses, times


def encode_video(folder, fps):
	subprocess.run(
		[
			"ffmpeg",
			"-n",
			"-loglevel",
			"error",
			"-framerate",
			str(fps),
			"-i",
			str(folder / "frame_%06d.png"),
			"-c:v",
			"libx264",
			"-pix_fmt",
			"yuv420p",
			"-crf",
			"20",
			str(folder / "animation.mp4"),
		],
		check=True,
	)


def main():
	p = argparse.ArgumentParser()
	p.add_argument("--request", type=Path, required=True)
	p.add_argument("--out", type=Path, required=True)
	a = p.parse_args()
	e = json.loads(a.request.read_text(encoding="utf-8"))["episode"]
	roles = {
		r["role"]: Path(r["path"]) if Path(r["path"]).is_absolute() else a.request.parent / r["path"]
		for r in e["artifacts"]
	}
	scene, poses, _ = load_motion(roles)
	import cv2
	import mujoco

	m = mujoco.MjModel.from_xml_path(str(roles["mjcf"]))
	m.vis.global_.offwidth = 1280
	m.vis.global_.offheight = 720
	d = mujoco.MjData(m)
	with np.load(roles["telemetry"], allow_pickle=False) as z:
		q = z["full_qpos"]
	if q.shape != (len(poses), m.nq):
		raise ValueError("full model state does not match motion")
	a.out.mkdir(parents=True, exist_ok=False)
	cam = mujoco.MjvCamera()
	cam.lookat[:] = (0.23, 0, 0.10)
	cam.distance = 1.1
	cam.azimuth = 135
	cam.elevation = -30
	with mujoco.Renderer(m, 720, 1280) as renderer:
		for i, frame in enumerate(q):
			d.qpos[:] = frame
			mujoco.mj_forward(m, d)
			renderer.update_scene(d, cam)
			cv2.imwrite(str(a.out / f"frame_{i:06d}.png"), cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
	encode_video(a.out, float(scene.get("fps", 30)))


if __name__ == "__main__":
	main()
