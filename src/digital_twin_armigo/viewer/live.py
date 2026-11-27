"""Live 3D streaming of a running MuJoCo agent episode into Rerun (opt-in with ``ARMIGO_RERUN=1``; ``ARMIGO_RERUN=record`` only saves the pose replay, no viewer).

The simulation process hosts a Rerun gRPC server and web viewer (once per process); every episode gets its own Rerun
recording (id = episode directory name) with

* the full scene geometry, logged once (static), and the pose of every geom that moved, logged each control frame,
* the state-machine events, node activity and timing summary (same entities as ``rerun_log.EpisodeLogger``).

Open the viewer URL written to ``<episode>/rerun.json`` (the HTML page does this for you): the 3D view is a normal Rerun
view, so orbit, pan, zoom and fly work freely, and the time slider scrubs back through the run while it is still going.
At the end the poses are saved to ``<episode>/replay/`` (``scene.json`` + ``motion.npz``, the catalog motion contract) so
``armigo-view rerun``/``rrd`` can replay the 3D run after the process is gone.

Streaming never changes the simulation: it only reads poses after each frame, and any failure disables the stream
(with one warning) instead of stopping the episode.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
from pathlib import Path
from urllib.parse import quote

import numpy as np

ENV = "ARMIGO_RERUN"
FPS = 30
POSE_EPS = 1e-7
_hub: dict | None = None
_hub_lock = threading.Lock()


def enabled() -> bool:
	return os.environ.get(ENV, "").strip().lower() not in ("", "0", "false", "off", "no")


def record_only() -> bool:
	"""``ARMIGO_RERUN=record``: write ``replay/`` (poses for a later 3D replay) without a viewer, gRPC server or ``live.rrd``.
	For batch runs (``armigo-inventory rerun``) where nobody watches and hundreds of episodes share one process."""
	return os.environ.get(ENV, "").strip().lower() == "record"


def _free_port() -> int:
	with socket.socket() as s:
		s.bind(("127.0.0.1", 0))
		return s.getsockname()[1]


def ensure_hub() -> dict:
	"""Start (once per process) the Rerun gRPC server and web viewer; returns ``{grpc, web, url}``."""
	global _hub
	with _hub_lock:
		if _hub is not None:
			return _hub
		from digital_twin_armigo.viewer.rerun_log import import_rerun

		rr = import_rerun()
		grpc_port = int(os.environ.get("ARMIGO_RERUN_GRPC_PORT") or _free_port())
		web_port = int(os.environ.get("ARMIGO_RERUN_WEB_PORT") or _free_port())
		hub = rr.RecordingStream("armigo-hub")
		uri = rr.serve_grpc(
			grpc_port=grpc_port, recording=hub, server_memory_limit=os.environ.get("ARMIGO_RERUN_MEMORY", "4GiB")
		)
		rr.serve_web_viewer(web_port=web_port, open_browser=False, connect_to=uri)
		_hub = {
			"grpc": uri,
			"web": web_port,
			"url": f"http://127.0.0.1:{web_port}/?url={quote(uri, safe='')}",
			"recording": hub,
		}
		return _hub


def live_info(episode_dir: Path) -> dict | None:
	"""The viewer info of a live episode, if the process that wrote it is still alive."""
	f = Path(episode_dir) / "rerun.json"
	try:
		info = json.loads(f.read_text(encoding="utf-8"))
	except (OSError, ValueError):
		return None
	pid = int(info.get("pid", 0))
	if not pid or info.get("finished") or not _pid_alive(pid):
		return None
	return info


def _pid_alive(pid: int) -> bool:
	if pid == os.getpid():
		return True
	try:
		if sys.platform == "win32":
			import ctypes

			handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
			if not handle:
				return False
			code = ctypes.c_ulong()
			ok = ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
			ctypes.windll.kernel32.CloseHandle(handle)
			return bool(ok) and code.value == 259  # STILL_ACTIVE
		os.kill(pid, 0)
		return True
	except (OSError, ProcessLookupError):
		return False


class LiveStream:
	def __init__(self, cell, episode_dir: Path, *, started: float | None = None):
		from digital_twin_armigo.catalog.capture import MujocoCapture

		self.dir = Path(episode_dir)
		self.started = time.perf_counter() if started is None else started
		self.recording_only = record_only()
		self.capture = MujocoCapture(cell)
		self.node_ids = [n["id"] for n in self.capture.scene["nodes"]]
		self.last: np.ndarray | None = None
		self.frames: list[np.ndarray] = []
		self.step = 0
		self.lock = threading.Lock()
		if self.recording_only:  # poses only: no Rerun stream at all
			self.rr = self.rec = self.logger = None
			return
		from digital_twin_armigo.viewer import rerun_log as rl

		self.rr = rl.import_rerun()
		hub = ensure_hub()
		self.rec = self.rr.RecordingStream("armigo-live", recording_id=self.dir.name)
		# One stream, two sinks: the live viewer, and ``live.rrd`` next to the episode for replay after the process is gone.
		self.rr.set_sinks(
			self.rr.GrpcSink(hub["grpc"]),
			self.rr.FileSink(str(self.dir / "live.rrd")),
			default_blueprint=rl.blueprint(has_scene=True),
			recording=self.rec,
		)
		rl.log_scene_geometry(self.capture.scene, recording=self.rec)
		self.logger = rl.EpisodeLogger(self.dir, recording=self.rec)
		self.logger.log_graph()
		(self.dir / "rerun.json").write_text(
			json.dumps(
				{"url": hub["url"], "web": hub["web"], "grpc": hub["grpc"], "pid": os.getpid(), "finished": False}
			),
			encoding="utf-8",
		)

	def on_frame(self, episode) -> None:
		"""Log the pose of every geom that moved since the previous control frame."""
		poses = self.capture.poses(episode.cell)
		if self.recording_only:
			with self.lock:
				self.frames.append(poses)
			return
		with self.lock:
			rr, rec = self.rr, self.rec
			rr.set_time("sim_time", duration=episode.n_frames / FPS, recording=rec)
			rr.set_time("wall_time", duration=time.perf_counter() - self.started, recording=rec)
			rr.set_time("step", sequence=self.step, recording=rec)
			moved = (
				range(len(poses))
				if self.last is None
				else np.flatnonzero(np.any(np.abs(poses - self.last) > POSE_EPS, axis=1))
			)
			for i in moved:
				p = poses[i]
				rr.log(
					f"world/{self.node_ids[i]}",
					rr.Transform3D(translation=p[:3], quaternion=rr.Quaternion(xyzw=p[3:])),
					recording=rec,
				)
			self.last = poses
			self.frames.append(poses)
			self.step += 1

	def on_events(self, events: list[dict]) -> None:
		from digital_twin_armigo.viewer import stats as st

		if self.recording_only:
			return
		with self.lock:
			if not self.logger.log_events(events):
				return
			self.logger.log_visit_durations(st.node_visits(events))
			self.logger.log_stats(st.episode_stats(events))

	def finish(self, events: list[dict]) -> None:
		"""Final statistics, the pose replay on disk, and release the viewer entry."""
		from digital_twin_armigo.catalog.store import dumps

		self.on_events(events)
		with self.lock:
			if self.frames:
				out = self.dir / "replay"
				out.mkdir(exist_ok=True)
				(out / "scene.json").write_text(
					dumps(
						{
							"schema": "catalog-scene/v1",
							"units": "m",
							"up_axis": "Z",
							"quaternion": "xyzw",
							**self.capture.scene,
						}
					),
					encoding="utf-8",
				)
				np.savez_compressed(
					out / "motion.npz",
					poses=np.stack(self.frames),
					timestamps=(np.arange(len(self.frames)) + 1) / FPS,
					node_ids=np.array(self.node_ids),
				)
			if self.recording_only:
				return
			info = json.loads((self.dir / "rerun.json").read_text(encoding="utf-8"))
			info["finished"] = True
			(self.dir / "rerun.json").write_text(json.dumps(info), encoding="utf-8")
			self.rec.flush()


def maybe_start(cell, episode_dir: Path, *, started: float | None = None) -> LiveStream | None:
	"""A ``LiveStream`` when ``ARMIGO_RERUN`` is set and Rerun works; otherwise ``None`` (never raises)."""
	if not enabled():
		return None
	try:
		return LiveStream(cell, episode_dir, started=started)
	except Exception as exc:  # noqa: BLE001 - streaming is optional: never stop an episode for it
		print(f"armigo: live Rerun disabled ({exc})", file=sys.stderr)
		return None
