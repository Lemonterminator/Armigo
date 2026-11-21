"""``armigo-view``: HTML front end, Rerun replay/live viewer, `.rrd` export and timing statistics."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from digital_twin_armigo.viewer import stats as st


def _recording_roles(directory: Path | None, catalog_id: str | None, root: Path) -> dict | None:
	"""Locate scene/motion/telemetry either in a directory or through a catalog episode id."""
	if catalog_id:
		from digital_twin_armigo.catalog.store import Catalog

		episode = Catalog(root).episode(catalog_id)
		roles = {a["role"]: Path(a["path"]) for a in episode["artifacts"]}
	elif directory is not None:
		roles = {
			name: directory / f
			for name, f in (("scene", "scene.json"), ("motion", "motion.npz"), ("telemetry", "telemetry.npz"))
		}
	else:
		return None
	return roles if roles.get("scene") and roles.get("motion") and roles["motion"].is_file() else None


def build_recording(
	path: Path, *, recording: Path | None = None, catalog_id: str | None = None, root: Path = Path(".")
):
	"""Open an episode for Rerun: an agent episode directory and/or a motion recording. Returns ``(logger, has_scene)``."""
	from digital_twin_armigo.catalog.playback import load_motion
	from digital_twin_armigo.viewer import rerun_log as rl

	is_agent = (path / "events.jsonl").is_file()
	logger = rl.EpisodeLogger(path, app_id=f"armigo-{path.name}") if is_agent else None
	if logger is None:
		import rerun as rr

		rr.init(f"armigo-{path.name}", spawn=False)
	roles = _recording_roles(recording or (None if is_agent else path), catalog_id, root)
	if roles:
		scene, poses, times = load_motion(roles)
		rl.log_scene(scene, poses, times)
		tele = roles.get("telemetry")
		if tele and Path(tele).is_file():
			with np.load(tele, allow_pickle=False) as z:
				rl.log_telemetry({k: z[k] for k in z.files}, times)
	return logger, bool(roles)


def cmd_rrd(a) -> int:
	from digital_twin_armigo.viewer import rerun_log as rl

	rr = rl.import_rerun()
	out = a.out or Path(f"{a.path.name}.rrd")
	logger, has_scene = build_recording(a.path, recording=a.recording, catalog_id=a.catalog, root=a.root)
	rr.save(out, default_blueprint=rl.blueprint(has_scene=has_scene))
	if logger:
		logger.log_graph()
		logger.refresh()
	print(out)
	return 0


def cmd_rerun(a) -> int:
	from digital_twin_armigo.viewer import rerun_log as rl

	rr = rl.import_rerun()
	logger, has_scene = build_recording(a.path, recording=a.recording, catalog_id=a.catalog, root=a.root)
	uri = rr.serve_grpc(
		grpc_port=a.grpc_port, default_blueprint=rl.blueprint(has_scene=has_scene), server_memory_limit="2GiB"
	)
	rr.serve_web_viewer(web_port=a.web_port, open_browser=not a.no_browser, connect_to=uri)
	print(f"rerun viewer: http://127.0.0.1:{a.web_port or 9090}/  (grpc {uri})", flush=True)
	if logger:
		if a.follow:
			logger.follow(idle_exit_s=None)
		else:
			logger.log_graph()
			logger.refresh()
	try:
		while True:
			time.sleep(1)
	except KeyboardInterrupt:
		return 0


def cmd_serve(a) -> int:
	from digital_twin_armigo.viewer import server

	print(f"armigo-view: http://127.0.0.1:{a.port}/  (episodes under {a.root.resolve()})", flush=True)
	server.serve(a.root, a.port, open_browser=a.open)
	return 0


def cmd_stats(a) -> int:
	events = st.read_events(a.path / "events.jsonl")
	if not events:
		print(f"no events in {a.path}", file=sys.stderr)
		return 1
	meta_file = a.path / "episode.json"
	meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.is_file() else {}
	phases: list[str] = []
	for b in meta.get("branches", []):
		f = a.path / f"branch-{b['id']}" / "actions.npz"
		if f.is_file():
			with np.load(f, allow_pickle=False) as z:
				phases += [str(x) for x in z["phase"]] if "phase" in z.files else []
	result = st.episode_stats(events, phases or None, fps=int(meta.get("fps", st.FPS)), finished=bool(meta))
	if a.json:
		print(json.dumps(result, indent=1, allow_nan=False))
	else:
		from digital_twin_armigo.viewer.rerun_log import stats_markdown

		print(stats_markdown(result))
	return 0


def main(argv: list[str] | None = None) -> int:
	p = argparse.ArgumentParser(prog="armigo-view", description=__doc__.splitlines()[0])
	sub = p.add_subparsers(dest="cmd", required=True)

	s = sub.add_parser("serve", help="HTML page: state machine, timing statistics, Rerun launcher")
	s.add_argument("--root", type=Path, default=Path("data/catalog"), help="directory searched for episodes")
	s.add_argument("--port", type=int, default=8765)
	s.add_argument("--open", action="store_true", help="open the page in the default browser")
	s.set_defaults(func=cmd_serve)

	def source(sp):
		sp.add_argument(
			"path", type=Path, help="agent episode directory (events.jsonl) or recording directory (motion.npz)"
		)
		sp.add_argument("--recording", type=Path, help="recording directory to show in 3D next to an agent episode")
		sp.add_argument("--catalog", help="catalog episode id whose motion/scene/telemetry artifacts are shown in 3D")
		sp.add_argument("--root", type=Path, default=Path("."), help="project root of the catalog (with --catalog)")

	r = sub.add_parser("rerun", help="serve a live Rerun web viewer for an episode")
	source(r)
	r.add_argument("--grpc-port", type=int)
	r.add_argument("--web-port", type=int)
	r.add_argument("--no-browser", action="store_true")
	r.add_argument("--follow", action="store_true", help="keep tailing a running episode until it finishes")
	r.set_defaults(func=cmd_rerun)

	d = sub.add_parser("rrd", help="write a self-contained .rrd file (open with `rerun file.rrd`)")
	source(d)
	d.add_argument("--out", type=Path)
	d.set_defaults(func=cmd_rrd)

	t = sub.add_parser("stats", help="print per-node and per-phase timing of an episode")
	t.add_argument("path", type=Path)
	t.add_argument("--json", action="store_true")
	t.set_defaults(func=cmd_stats)

	a = p.parse_args(argv)
	return a.func(a)


if __name__ == "__main__":
	raise SystemExit(main())
