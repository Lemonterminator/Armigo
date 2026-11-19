"""Frame-aligned camera, robot and LangGraph recordings for the six-face twin."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import av
import cv2
import numpy as np

from ..constants import FPS, STATE_NAMES, TOP_CROP
from ..joints import rad_to_normalized
from .twin_graph import EDGES, NODES, TASK

CAMERAS = ("front", "wrist", "top_rgb", "top_depth")


def write_json(path, value):
	path = Path(path)
	tmp = path.with_suffix(path.suffix + ".tmp")
	tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
	tmp.replace(path)


class Video:
	def __init__(self, path, shape, *, lossless):
		self.output = av.open(str(path), "w")
		self.stream = self.output.add_stream("ffv1" if lossless else "libx264", rate=FPS)
		self.stream.height, self.stream.width = shape[:2]
		self.stream.pix_fmt = "bgr0" if lossless else "yuv420p"
		if not lossless:
			self.stream.options = {"preset": "ultrafast", "crf": "20", "tune": "zerolatency"}
		self.index = 0

	def put(self, rgb):
		frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
		frame.pts = self.index
		for packet in self.stream.encode(frame):
			self.output.mux(packet)
		self.index += 1

	def close(self):
		try:
			for packet in self.stream.encode():
				self.output.mux(packet)
		finally:
			self.output.close()


class EpisodeTrace:
	def __init__(self, path, *, video):
		self.path = Path(path)
		self.path.mkdir(parents=True, exist_ok=False)
		self.video = video
		self.events = []
		self.branch = 0
		self.branches = [{"id": 0, "start_frame": 0}]
		self.streams = {}
		self.rows = []
		self.images = None
		self.started = time.perf_counter()
		self.keyframes = self.path / "keyframes"
		self.keyframes.mkdir()
		self.checkpoints = self.path / "checkpoints"
		self.checkpoints.mkdir()
		self.event_log = (self.path / "events.jsonl").open("w", encoding="utf-8")

	def start(self, cell):
		write_json(self.path / "graph.json", {"nodes": NODES, "edges": EDGES})
		# Scoring only (S6 grader): per-frame contact peaks over the frame's physics substeps, stored next to the
		# actions of every branch, so collisions can be counted over all attempts. The evaluator's own measure
		# excludes the jaws; ``contact_jaw_env_n`` is the jaws against the table, bowls and clutter.
		self.contacts = []
		self._jaw_env_peak = 0.0
		cell.substep_observer = lambda: self._substep_contact(cell)
		cell.frame_observer = self._frame_contact

	def _substep_contact(self, cell):
		from ..mujoco_cell.episode import jaw_env_contact_force

		self._jaw_env_peak = max(self._jaw_env_peak, jaw_env_contact_force(cell))

	def _frame_contact(self, episode):
		robot = float(np.max(episode.last("substep_unintended")))
		self.contacts.append((robot, self._jaw_env_peak))
		self._jaw_env_peak = 0.0

	def event(self, kind, node, branch, frame, **payload):
		row = {
			"seq": len(self.events),
			"kind": kind,
			"node": node,
			"branch": branch,
			"frame": frame,
			"sim_time_s": frame / FPS,
			"wall_time_s": time.perf_counter() - self.started,
			**payload,
		}
		self.events.append(row)
		self.event_log.write(json.dumps(row, allow_nan=False) + "\n")
		self.event_log.flush()

	def checkpoint(self, checkpoint):
		raw = asdict(checkpoint)
		integration = raw.pop("integration")
		np.save(self.checkpoints / f"{checkpoint.id}.npy", integration, allow_pickle=False)
		write_json(self.checkpoints / f"{checkpoint.id}.json", raw)

	def audit(self, data):
		write_json(self.path / "audit.private.json", data)

	def initial(self, cell, q):
		del q
		if self.video:
			self.images = self._render(cell)

	def command(self, state, target, phase):
		self.rows.append((state.copy(), target.copy(), phase))

	def step(self, cell, node):
		if self.video:
			# Video sample i is what the policy would see BEFORE applying command i.
			self._push(node)
			self.images = self._render(cell)
			if len(self.rows) % 15 == 0:
				self._publish_preview(node)

	def _publish_preview(self, node, frame=None):
		preview = self._composite(node, frame)
		ok, buf = cv2.imencode(".jpg", cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
		if not ok:
			return
		tmp = self.path / "live.tmp"
		tmp.write_bytes(buf.tobytes())
		try:
			tmp.replace(self.path / "live.jpg")
		except PermissionError:
			# A browser streaming the previous preview can briefly lock its file on Windows.
			# The next preview can retry. Observation frames and video are recorded separately.
			pass

	def restored(self, cell):
		if self.video:
			self.images = self._render(cell)

	def view(self, evidence, rgb, image):
		if self.video:
			# Use the exact observation the detector cropped at this action boundary.
			# Independent renders of the same state can differ in a few RGB values.
			camera_stream = "top_rgb" if evidence["camera"] == "top" else "wrist"
			self.images[camera_stream] = rgb.copy()
		stem = evidence["id"].replace(":", "-")
		for suffix, pixels in (("frame", rgb), ("crop", image)):
			if pixels is not None:
				encoded = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
				if not cv2.imwrite(str(self.keyframes / f"{stem}-{suffix}.png"), encoded):
					raise OSError("could not store decision camera frame")

	def fork(self, branch, checkpoint):
		self._finish_branch()
		self.branches.append(
			{
				"id": branch,
				"start_frame": checkpoint.frame,
				"parent_branch": checkpoint.branch,
				"checkpoint": checkpoint.id,
			}
		)
		self.branch = branch

	def _render(self, cell):
		top = cell.rgb("top")
		_, depth = cell.top_images()
		return {
			"front": cell.rgb("front"),
			"wrist": cell.rgb("wrist_cam"),
			"top_rgb": top,
			"top_depth": depth,
		}

	def _composite(self, node, frame=None):
		canvas = np.full((720, 1024, 3), (17, 24, 29), dtype=np.uint8)
		for name, (x, y, w, h, title) in {
			"front": (16, 44, 608, 456, "ROBOT / FRONT"),
			"wrist": (648, 44, 352, 264, "WRIST RGB"),
			"top_rgb": (648, 344, 352, 264, "EXTERNAL RGB"),
		}.items():
			canvas[y : y + h, x : x + w] = cv2.resize(self.images[name], (w, h), interpolation=cv2.INTER_AREA)
			cv2.putText(canvas, title, (x, y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (207, 219, 225), 1, cv2.LINE_AA)
		for i, name in enumerate(NODES):
			x, y = 12 + i * 144, 653
			active = name == node
			cv2.rectangle(canvas, (x, y), (x + 136, y + 44), (44, 164, 112) if active else (39, 53, 64), -1)
			cv2.putText(canvas, name, (x + 8, y + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (239, 248, 252), 1, cv2.LINE_AA)
		branch = self.branches[-1]["id"]
		if frame is None:
			frame = self.branches[-1]["start_frame"] + len(self.rows)
		cv2.putText(
			canvas,
			f"Branch {branch}   Frame {frame:04}",
			(16, 548),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.70,
			(201, 219, 223),
			1,
			cv2.LINE_AA,
		)
		return canvas

	def _push(self, node):
		panels = self._composite(node, self.branches[-1]["start_frame"] + len(self.rows) - 1)
		if not self.streams:
			dir_ = self.path / f"branch-{self.branch}"
			dir_.mkdir()
			self.streams = {
				name: Video(dir_ / f"{name}.mkv", pixels.shape, lossless=True) for name, pixels in self.images.items()
			}
			self.streams["replay"] = Video(dir_ / "replay.mp4", panels.shape, lossless=False)
		for name, pixels in self.images.items():
			self.streams[name].put(pixels)
		self.streams["replay"].put(panels)

	def _finish_branch(self):
		if self.rows:
			path = self.path / f"branch-{self.branch}"
			path.mkdir(exist_ok=True)
			state, action, phase = zip(*self.rows, strict=True)
			extra = {}
			contacts = getattr(self, "contacts", None)
			if contacts is not None and len(contacts) == len(self.rows):
				pairs = np.array(contacts, dtype="float32").reshape(-1, 2)
				extra = {"contact_robot_n": pairs[:, 0], "contact_jaw_env_n": pairs[:, 1]}
			np.savez_compressed(
				path / "actions.npz",
				state=np.array(state, dtype="float32"),
				action=np.array(action, dtype="float32"),
				phase=phase,
				**extra,
			)
		self.branches[-1]["actions"] = len(self.rows)
		for stream in self.streams.values():
			stream.close()
		self.streams = {}
		self.rows = []
		if getattr(self, "contacts", None) is not None:
			self.contacts = []

	def finish(self, result, provenance):
		self._finish_branch()
		if self.video and self.images is not None:
			branch = self.branches[-1]
			self._publish_preview("evaluate", branch["start_frame"] + branch["actions"])
		manifest = {
			"schema": "armigo-agent-episode/v1",
			"fps": FPS,
			"task": TASK,
			"state_names": STATE_NAMES,
			"branches": self.branches,
			"video": self.video,
			"provenance": provenance,
			"result": result,
		}
		write_json(self.path / "episode.json", manifest)
		self.event_log.close()

	def close(self):
		self._finish_branch()
		self.event_log.close()


def student_frames(episode: Path):
	"""Yield only contiguous, successful unrewound samples for the LeRobot exporter."""
	from contextlib import ExitStack

	episode = Path(episode)
	manifest = json.loads((episode / "episode.json").read_text(encoding="utf-8"))
	result = manifest["result"]
	if (
		manifest["schema"] != "armigo-agent-episode/v1"
		or not manifest["video"]
		or result["verdict"] != "SUCCESS"
		or not result["student_eligible"]
		or len(manifest["branches"]) != 1
		or manifest["branches"][0]["start_frame"] != 0
		or result["rewinds"]
		or manifest["task"] != TASK
		or manifest["fps"] != FPS
	):
		raise ValueError("episode is not eligible for VLA export")
	root = episode / "branch-0"
	with ExitStack() as stack:
		rows = stack.enter_context(np.load(root / "actions.npz", allow_pickle=False))
		readers = {
			name: iter(stack.enter_context(av.open(str(root / f"{name}.mkv"))).decode(video=0)) for name in CAMERAS
		}
		for row_index, (state, action) in enumerate(zip(rows["state"], rows["action"], strict=True)):
			frame = {
				"observation.state": rad_to_normalized(state),
				"action": rad_to_normalized(action),
				"task": TASK,
			}
			for name, reader in readers.items():
				pixels = next(reader, None)
				if pixels is None or pixels.time is None or abs(pixels.time - row_index / FPS) > 0.001:
					raise ValueError(f"missing or misaligned frame {row_index} in {name} stream")
				image = pixels.to_ndarray(format="rgb24")
				key = name
				if name == "top_rgb":
					frame["observation.images.top_full"] = image
					x, y, w, h = TOP_CROP
					image = image[y : y + h, x : x + w].copy()
					key = "top_rgb_v2"
				elif name == "top_depth":
					key = "top_depth_v2"
				frame[f"observation.images.{key}"] = image
			yield frame
		if any(next(reader, None) is not None for reader in readers.values()):
			raise ValueError("camera stream has frames with no matching action")


def export_lerobot(episodes: list[Path], out: Path, repo_id: str):
	from lerobot.datasets.lerobot_dataset import LeRobotDataset

	from ..constants import CAMERA_SHAPES, TOP_FULL
	from ..lerobot_export import lerobot_features

	# Fail before creating output if an episode ended on a branch with privileged failure context.
	for path in episodes:
		frames = student_frames(path)
		try:
			next(frames)
		finally:
			frames.close()
	dataset = LeRobotDataset.create(
		repo_id=repo_id,
		fps=FPS,
		root=out,
		robot_type="SO101_Follower",
		use_videos=True,
		features=lerobot_features(
			{
				"state_names": STATE_NAMES,
				"cameras": {**CAMERA_SHAPES, "top_full": [TOP_FULL[1], TOP_FULL[0], 3]},
			},
			video=True,
		),
	)
	try:
		for path in episodes:
			for frame in student_frames(path):
				dataset.add_frame(frame)
			dataset.save_episode()
	finally:
		dataset.finalize()
	write_json(
		out / "meta/source.json",
		{
			"schema": "armigo-student-provenance/v1",
			"episodes": [{"episode_index": i, "source": path.name} for i, path in enumerate(episodes)],
			"sampling": "pre-action RGB and proprioception; next policy command; Studio normalized joint units",
			"decision_camera": "top_full or wrist at recorded frame index",
		},
	)


def main():
	import argparse

	parser = argparse.ArgumentParser(description="Export accepted Armigo recordings through LeRobot.")
	parser.add_argument("--out", type=Path, required=True)
	parser.add_argument("--repo-id", default="local/armigo-six-face")
	parser.add_argument("episodes", nargs="+", type=Path)
	args = parser.parse_args()
	export_lerobot(args.episodes, args.out, args.repo_id)


if __name__ == "__main__":
	main()
