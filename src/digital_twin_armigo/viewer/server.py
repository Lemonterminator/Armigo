"""Loopback-only HTML front end: state machine + timing statistics, and a button that opens the Rerun viewer.

``GET`` routes are read-only. Starting or stopping a Rerun session is a ``POST`` that needs the per-process token from
``/api/token`` (same convention as the catalog web app).
"""

from __future__ import annotations

import json
import os
import secrets
import socket
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np

from digital_twin_armigo.viewer import live
from digital_twin_armigo.viewer import stats as st

SKIP_DIRS = {"branch-0", "keyframes", "checkpoints", "replay", "__pycache__", ".git", "node_modules"}
MAX_DEPTH = 4
LIVE_GRACE_S = 30.0


def discover(root: Path) -> list[dict]:
	"""Agent episodes (events.jsonl + graph.json) and motion recordings (motion.npz + scene.json) under ``root``."""
	root = Path(root).resolve()
	found: list[dict] = []
	for dirpath, dirnames, filenames in os.walk(root):
		depth = len(Path(dirpath).relative_to(root).parts)
		dirnames[:] = (
			[d for d in dirnames if d not in SKIP_DIRS and not d.startswith("branch-")] if depth < MAX_DEPTH else []
		)
		names = set(filenames)
		path = Path(dirpath)
		kind = None
		if {"events.jsonl", "graph.json"} <= names:
			kind = "agent"
		elif {"motion.npz", "scene.json"} <= names:
			kind = "recording"
		if kind:
			marker = path / ("events.jsonl" if kind == "agent" else "motion.npz")
			found.append(
				{
					"id": path.relative_to(root).as_posix() or ".",
					"kind": kind,
					"mtime": marker.stat().st_mtime,
					"finished": (path / "episode.json").is_file() if kind == "agent" else True,
				}
			)
	return sorted(found, key=lambda r: r["mtime"], reverse=True)


def _free_port() -> int:
	with socket.socket() as s:
		s.bind(("127.0.0.1", 0))
		return s.getsockname()[1]


def _wait_port(port: int, timeout: float) -> bool:
	end = time.monotonic() + timeout
	while time.monotonic() < end:
		with socket.socket() as s:
			s.settimeout(0.3)
			if s.connect_ex(("127.0.0.1", port)) == 0:
				return True
		time.sleep(0.2)
	return False


class RerunSession:
	"""At most one `armigo-view rerun` child process, so ports and memory stay bounded."""

	def __init__(self):
		self.proc: subprocess.Popen | None = None
		self.info: dict = {}
		self.lock = threading.Lock()

	def start(self, target: Path, recording: Path | None = None) -> dict:
		with self.lock:
			self._stop()
			grpc, web = _free_port(), _free_port()
			cmd = [
				sys.executable,
				"-m",
				"digital_twin_armigo.viewer.cli",
				"rerun",
				str(target),
				"--grpc-port",
				str(grpc),
				"--web-port",
				str(web),
				"--no-browser",
				"--follow",
			]
			if recording:
				cmd += ["--recording", str(recording)]
			self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
			if not _wait_port(web, 40):
				self._stop()
				raise RuntimeError("Rerun viewer did not start (is rerun-sdk installed? uv sync --extra viewer)")
			self.info = {
				"url": f"http://127.0.0.1:{web}/?url=rerun%2Bhttp%3A%2F%2F127.0.0.1%3A{grpc}%2Fproxy",
				"pid": self.proc.pid,
			}
			return self.info

	def _stop(self) -> None:
		if self.proc and self.proc.poll() is None:
			self.proc.terminate()
			try:
				self.proc.wait(5)
			except subprocess.TimeoutExpired:
				self.proc.kill()
		self.proc, self.info = None, {}

	def stop(self) -> None:
		with self.lock:
			self._stop()

	def status(self) -> dict:
		running = self.proc is not None and self.proc.poll() is None
		return {"running": running, **(self.info if running else {})}


def episode_payload(path: Path, cache: dict) -> dict:
	"""State, visits and statistics of an agent episode; recomputed only when its files change."""
	events_file, end_file = path / "events.jsonl", path / "episode.json"
	key = (
		events_file.stat().st_mtime_ns if events_file.is_file() else 0,
		end_file.stat().st_mtime_ns if end_file.is_file() else 0,
	)
	hit = cache.get(path)
	finished = end_file.is_file()
	if hit and hit[0] == key and finished:
		return hit[1]
	events = st.read_events(events_file)
	meta = json.loads(end_file.read_text(encoding="utf-8")) if finished else {}
	fps = int(meta.get("fps", st.FPS)) if meta else st.FPS
	phases: list[str] = []
	for b in meta.get("branches", []):
		actions = path / f"branch-{b['id']}" / "actions.npz"
		if actions.is_file():
			with np.load(actions, allow_pickle=False) as z:
				if "phase" in z.files:
					phases += [str(x) for x in z["phase"]]
	now_wall = now_sim = None
	if events and not finished:
		quiet = max(0.0, time.time() - events_file.stat().st_mtime)
		if quiet < LIVE_GRACE_S * 10:  # beyond that the run is most likely dead; do not keep growing the open visit
			now_wall = float(events[-1]["wall_time_s"]) + quiet
	payload = {
		"graph": json.loads((path / "graph.json").read_text(encoding="utf-8")),
		"stats": st.episode_stats(
			events, phases or None, fps=fps, now_wall=now_wall, now_sim=now_sim, finished=finished
		),
		"events": events[-200:],
		"event_count": len(events),
		"result": meta.get("result"),
		"live": not finished and bool(events) and (time.time() - events_file.stat().st_mtime) < LIVE_GRACE_S,
		"live_image": (path / "live.jpg").is_file(),
		"replay": (path / "replay" / "motion.npz").is_file(),
	}
	cache[path] = (key, payload)
	return payload


def create_app(root: Path, *, session: RerunSession | None = None):
	from fastapi import FastAPI, HTTPException, Request
	from fastapi.responses import FileResponse, JSONResponse

	root = Path(root).resolve()
	token = secrets.token_urlsafe(32)
	rerun = session or RerunSession()
	cache: dict = {}

	@asynccontextmanager
	async def lifespan(app):
		yield
		rerun.stop()

	app = FastAPI(title="Armigo episode viewer", lifespan=lifespan)

	@app.middleware("http")
	async def local_only(request: Request, call_next):
		host = request.headers.get("host", "").split(":")[0]
		if host not in ("127.0.0.1", "localhost", "testserver"):
			return JSONResponse({"detail": "local requests only"}, status_code=403)
		if request.method not in ("GET", "HEAD") and not secrets.compare_digest(
			request.headers.get("X-Viewer-Token", ""), token
		):
			return JSONResponse({"detail": "viewer token required"}, status_code=403)
		return await call_next(request)

	def resolve(ident: str) -> Path:
		path = (root / ident).resolve()
		if not path.is_relative_to(root) or not path.is_dir():
			raise HTTPException(404, "unknown episode")
		return path

	@app.get("/")
	def index():
		return FileResponse(Path(__file__).with_name("viewer.html"), media_type="text/html")

	@app.get("/api/token")
	def get_token():
		return {"token": token}

	@app.get("/api/episodes")
	def episodes():
		return discover(root)

	@app.get("/api/episode")
	def episode(id: str):
		path = resolve(id)
		if (path / "events.jsonl").is_file() and (path / "graph.json").is_file():
			info = live.live_info(path)
			return {
				"kind": "agent",
				"id": id,
				**episode_payload(path, cache),
				"live_rerun": info["url"] if info else None,
			}
		if (path / "motion.npz").is_file():
			return {"kind": "recording", "id": id}
		raise HTTPException(404, "not an episode directory")

	@app.get("/api/live.jpg")
	def live_image(id: str):
		target = resolve(id) / "live.jpg"
		if not target.is_file():
			raise HTTPException(404)
		return FileResponse(target, headers={"Cache-Control": "no-store"})

	@app.get("/api/rerun")
	def rerun_status():
		return rerun.status()

	@app.post("/api/rerun")
	def rerun_start(id: str):
		path = resolve(id)
		info = live.live_info(path)
		if info:  # the simulation process already serves a viewer with the streamed 3D scene
			return {"url": info["url"], "live": True}
		try:
			return rerun.start(path)
		except RuntimeError as exc:
			return JSONResponse({"detail": str(exc)}, status_code=503)

	@app.delete("/api/rerun")
	def rerun_stop():
		rerun.stop()
		return {"running": False}

	return app


def serve(root: Path, port: int = 8765, open_browser: bool = False) -> None:
	import uvicorn

	if open_browser:
		import webbrowser

		threading.Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{port}/")).start()
	uvicorn.run(create_app(root), host="127.0.0.1", port=port, log_level="warning")
