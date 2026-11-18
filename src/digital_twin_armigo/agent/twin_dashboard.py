"""Loopback-only live monitor and branch-aware episode replay."""

from __future__ import annotations

import json
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

EPISODE_ID = re.compile(r"(?:[0-9a-f]{32}|case-[0-9]{4})")
MEDIA = re.compile(
	r"(?:live\.jpg|branch-\d+/(?:replay\.mp4|front\.mkv|wrist\.mkv|top_rgb\.mkv|top_depth\.mkv)"
	r"|keyframes/\d+-\d+-(?:top|wrist_cam)-(?:frame|crop)\.png)"
)


def create_app(root: Path):
	root = Path(root).resolve()
	app = FastAPI(title="Armigo episode replay")

	def episode(id_):
		if not EPISODE_ID.fullmatch(id_):
			raise HTTPException(404)
		for parent in (root / "episodes", root):
			path = parent / id_
			if path.is_dir() and path.resolve().is_relative_to(root) and (path / "graph.json").is_file():
				return path
		raise HTTPException(404)

	@app.get("/")
	def index():
		return FileResponse(Path(__file__).with_name("twin_dashboard.html"), media_type="text/html")

	@app.get("/api/episodes")
	def episodes():
		found = {}
		for parent in (root / "episodes", root):
			if parent.is_dir():
				for path in parent.iterdir():
					if (
						EPISODE_ID.fullmatch(path.name)
						and path.resolve().is_relative_to(root)
						and (path / "graph.json").is_file()
					):
						found[path.name] = path.stat().st_mtime
		return sorted(found, key=found.get, reverse=True)

	@app.get("/api/episode/{id_}")
	def state(id_: str):
		path = episode(id_)
		events = []
		for line in (path / "events.jsonl").read_text(encoding="utf-8").splitlines():
			try:
				events.append(json.loads(line))
			except json.JSONDecodeError:
				# An episode can be writing its next line while the dashboard polls.
				break
		end = path / "episode.json"
		return {
			"graph": json.loads((path / "graph.json").read_text(encoding="utf-8")),
			"events": events,
			"episode": json.loads(end.read_text(encoding="utf-8")) if end.is_file() else None,
			"thumbnail": (path / "live.jpg").is_file(),
		}

	@app.get("/media/{id_}/{name:path}")
	def media(id_: str, name: str):
		if not MEDIA.fullmatch(name):
			raise HTTPException(404)
		path = episode(id_)
		target = path / name
		if not target.is_file() or not target.resolve().is_relative_to(path.resolve()):
			raise HTTPException(404)
		return FileResponse(target, headers={"Cache-Control": "no-store"} if name == "live.jpg" else None)

	return app


def serve(root, port):
	import uvicorn

	uvicorn.run(create_app(root), host="127.0.0.1", port=port)
