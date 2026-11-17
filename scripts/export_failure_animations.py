"""Re-run the failing / flagged closed-loop expert episodes with rendering on and export one animation each.

Episodes are located in ``results/physics-expert-v2/**/*.json`` (the acceptance records). Each one is replayed with the
same code, seed and layout as the benchmark (``bench_physics_expert.run_episode`` / ``run_layout_episode``); rendering
only reads the simulation, it never changes it, and the replay is checked against the recorded verdict, frame count and
release speed. Output per episode: ``<id>.mp4`` (2x2: dataset top camera, overview, chase camera on the brick,
wrist camera, with a telemetry overlay),
``<id>.json`` (the recorded episode plus the replay check) and ``<id>.npz`` (frame-by-frame telemetry).
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import bench_physics_expert as bp

from digital_twin_armigo.benchmarks import HELD_OUT_PERTURBATIONS, SENSITIVITY_MATRIX
from digital_twin_armigo.constants import PHASES
from digital_twin_armigo.mujoco_cell import physics_expert as pe
from digital_twin_armigo.physics import load_physics_spec

RESULTS = ROOT / "results" / "physics-expert-v2"
DEFAULT_OUT = ROOT / "results" / "failure-export"
OVERVIEW = {"lookat": (0.2, 0.0, 0.05), "distance": 0.95, "azimuth": 160.0, "elevation": -35.0}
CHASE_AZIMUTH = 135.0
FFMPEG = "ffmpeg"


def locate(min_jaw_n: float = 5.0, long_frames: int = 890) -> list[dict]:
	"""Every failing or flagged episode in the acceptance JSON files."""
	found = []
	for f in sorted(glob.glob(str(RESULTS / "**" / "*.json"), recursive=True)):
		d = json.loads(Path(f).read_text(encoding="utf-8"))
		src = str(Path(f).relative_to(ROOT)).replace("\\", "/")
		if "conditions" in d:
			eps = [(cn, e) for cn, c in d["conditions"].items() for e in c["episodes_detail"]]
		else:
			eps = [("layout", e) for e in d["layouts"]["episodes_detail"]]
		for cn, e in eps:
			info = e.get("info", {})
			rel = info.get("release_peak_speed_mps") or 0.0
			jaw = info.get("jaw_env_contact_max_N") or 0.0
			flags = []
			if e["verdict"] != "SUCCESS":
				flags.append("FAIL")
			if rel > 1.2:
				flags.append("FAST_RELEASE")
			if jaw > min_jaw_n:
				flags.append("JAW_ENV_OVER_5N")
			if (info.get("n_frames") or 0) >= long_frames:
				flags.append("NEAR_FRAME_CAP")
			if not flags:
				continue
			if cn == "layout":
				key, ident = e["layout_index"], f"layouts_seed{d['layout_seed']}_idx{e['layout_index']:02d}"
			else:
				key, ident = e["seed"], f"{cn}_seed{e['seed']}"
			found.append(
				{
					"id": ident,
					"category": "failure" if "FAIL" in flags else "flagged",
					"flags": flags,
					"source": src,
					"mode": d["mode"],
					"condition": cn,
					"key": key,
					"layout_seed": d.get("layout_seed"),
					"recorded": e,
				}
			)
	# the same episode recorded in two files (e.g. high_mass seed 29) is exported once
	uniq: dict[str, dict] = {}
	for item in found:
		if item["id"] in uniq:
			uniq[item["id"]].setdefault("also_in", []).append(item["source"])
		else:
			uniq[item["id"]] = item
	return list(uniq.values())


class Recorder:
	"""Patches the episode so that every 30 Hz frame is rendered and streamed to a video writer."""

	def __init__(self, path: Path, header: str) -> None:
		self.path, self.header = path, header
		self.writer: cv2.VideoWriter | None = None
		self.phase = "rest"
		self.n = 0
		self._step = pe.PhysicsEpisode.step
		self._act = pe.ClosedLoopExpert.act

	def __enter__(self):
		rec = self

		def act(expert, obs):
			target, phase = rec._act(expert, obs)
			rec.phase = phase
			return target, phase

		def step(episode, target):
			rec._step(episode, target)
			rec.frame(episode)

		pe.ClosedLoopExpert.act = act
		pe.PhysicsEpisode.step = step
		return self

	def __exit__(self, *exc):
		pe.ClosedLoopExpert.act = self._act
		pe.PhysicsEpisode.step = self._step
		if self.writer is not None:
			self.writer.release()

	@staticmethod
	def _free_view(cell, lookat, distance: float, azimuth: float, elevation: float) -> np.ndarray:
		import mujoco

		cam = mujoco.MjvCamera()
		cam.type = mujoco.mjtCamera.mjCAMERA_FREE
		cam.lookat[:] = lookat
		cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
		cell.renderer.disable_depth_rendering()
		cell.renderer.update_scene(cell.data, cam)
		return np.ascontiguousarray(cell.renderer.render())

	def frame(self, episode) -> None:
		cell = episode.cell
		brick = np.asarray(episode.last("brick_pos"), dtype=np.float64)
		tiles = [
			np.ascontiguousarray(cell.rgb("top")),
			self._free_view(cell, OVERVIEW["lookat"], OVERVIEW["distance"], OVERVIEW["azimuth"], OVERVIEW["elevation"]),
			self._free_view(cell, brick, 0.30, CHASE_AZIMUTH, -25.0),
			np.ascontiguousarray(cell.rgb("wrist_cam")),
		]
		for tile, label in zip(tiles, ("top (dataset camera)", "overview", "chase (follows brick)", "wrist"), strict=True):
			cv2.putText(tile, label, (8, tile.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
			cv2.putText(tile, label, (8, tile.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
		img = np.concatenate([np.concatenate(tiles[:2], axis=1), np.concatenate(tiles[2:], axis=1)], axis=0)
		speed = float(np.linalg.norm(episode.last("brick_vel")))
		fn = episode.last("pinch_normal")
		lines = [
			self.header,
			f"frame {self.n:4d}  t={self.n / 30.0:5.2f}s  phase={self.phase}",
			f"brick speed {speed:5.2f} m/s  z={brick[2] * 1000:6.1f} mm  jaw normal {fn[0]:5.1f} / {fn[1]:5.1f} N  "
			f"jaw-env {episode.jaw_env_force[-1]:5.1f} N",
		]
		cv2.rectangle(img, (0, 0), (img.shape[1], 12 + 22 * len(lines)), (0, 0, 0), -1)
		for k, text in enumerate(lines):
			cv2.putText(img, text, (8, 22 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
		if speed > 1.2:
			cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), (255, 0, 0), 6)
		if self.writer is None:
			self.writer = cv2.VideoWriter(
				str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (img.shape[1], img.shape[0])
			)
		self.writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
		self.n += 1


def transcode(src: Path, dst: Path) -> bool:
	"""H.264 / yuv420p copy that plays everywhere; the raw mp4v file is removed on success."""
	cmd = [FFMPEG, "-y", "-loglevel", "error", "-i", str(src), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(dst)]
	if subprocess.run(cmd).returncode == 0:
		src.unlink()
		return True
	src.replace(dst)
	return False


def replay(item: dict, spec, top_cam, out_dir: Path) -> dict:
	import digital_twin_armigo.mujoco_cell as mc

	orig_cell = mc.Cell

	def rendering_cell(*args, **kwargs):
		kwargs["render_enabled"] = True  # rendering never feeds back into the physics
		return orig_cell(*args, **kwargs)

	mc.Cell = rendering_cell
	raw = out_dir / f"{item['id']}.raw.mp4"
	header = f"{item['id']}  recorded: {item['recorded']['verdict']} {item['recorded']['reason_codes']}"
	try:
		with Recorder(raw, header) as rec:
			if item["mode"] == "matrix":
				cond = {c.name: c for c in (*SENSITIVITY_MATRIX, *HELD_OUT_PERTURBATIONS)}[item["condition"]]
				row = bp.run_episode(cond, spec, top_cam, item["key"])
			else:
				layouts = bp.sample_layouts(60, item["layout_seed"], top_cam)
				row = bp.run_layout_episode(layouts[item["key"]], spec, item["key"])
	finally:
		mc.Cell = orig_cell
	tel = row.pop("_telemetry", None)
	phases = row.pop("_phases", None)
	rec_info, new_info = item["recorded"].get("info", {}), row.get("info", {})
	check = {
		"verdict_match": row["verdict"] == item["recorded"]["verdict"],
		"reason_codes_match": list(row["reason_codes"]) == list(item["recorded"]["reason_codes"]),
		"n_frames_recorded": rec_info.get("n_frames"),
		"n_frames_replay": new_info.get("n_frames"),
		"release_speed_recorded": rec_info.get("release_peak_speed_mps"),
		"release_speed_replay": new_info.get("release_peak_speed_mps"),
		"video_frames": rec.n,
	}
	mp4 = out_dir / f"{item['id']}.mp4"
	check["h264"] = transcode(raw, mp4)
	if tel is not None:
		np.savez_compressed(
			out_dir / f"{item['id']}.npz",
			phases=phases,
			phase_names=np.array(PHASES),
			**{k: v for k, v in tel.items()},
		)
	record = {k: v for k, v in item.items() if k != "recorded"}
	record["recorded_episode"] = item["recorded"]
	record["replay_check"] = check
	(out_dir / f"{item['id']}.json").write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
	return check


def main(argv: list[str] | None = None) -> int:
	ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
	ap.add_argument("--category", choices=["failure", "flagged", "all"], default="failure")
	ap.add_argument("--only", nargs="*", help="episode ids to replay")
	ap.add_argument("--list", action="store_true", help="only print the located episodes")
	args = ap.parse_args(argv)

	items = locate()
	if args.category != "all":
		items = [i for i in items if i["category"] == args.category]
	if args.only:
		items = [i for i in items if i["id"] in args.only]
	for i in items:
		r = i["recorded"]
		print(f"{i['category']:8s} {i['id']:32s} {r['verdict']:8s} {r['reason_codes']} {i['flags']} {i['source']}")
	if args.list:
		return 0
	spec = load_physics_spec()
	top_cam = bp.load_top_camera()
	for i in items:
		out_dir = args.out / i["category"]
		out_dir.mkdir(parents=True, exist_ok=True)
		check = replay(i, spec, top_cam, out_dir)
		print("replayed", i["id"], check, flush=True)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
