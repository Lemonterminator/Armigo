"""Stream an agent episode and/or a recorded 3D motion into Rerun.

What is logged (entity paths; two timelines, ``sim_time`` and ``wall_time``, both in seconds):

``state/active/<node>``     1 while the state machine is in the node, 0 otherwise (a Gantt-like step plot)
``state/node_index``        index of the current node (see the graph node order in ``state/graph``)
``state/log``               one text line per event (node enter, decision with source/latency, view, checkpoint ...)
``state/visit_wall_s/<n>``  wall seconds of each finished visit, logged where the visit ends
``phase/index`` ``phase/log``  expert phase per frame of a finished branch (rest, approach, grasp ...)
``robot/state/<j>`` ``robot/action/<j>``  joints of a finished branch
``stats/summary``           markdown table of per-node and per-phase timing, refreshed while the episode is live
``camera/live`` ``camera/<n>``  the live thumbnail and the saved keyframes
``world/<geom>``            recorded scene geometry (static) with a per-frame transform, when a motion recording exists
``telemetry/<name>``        small per-frame telemetry arrays of the recording (forces, velocities ...)

Everything here is read-only on the episode directory, so it can run next to a live episode.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from digital_twin_armigo.viewer import stats as st

DEFAULT_NODES = ("prepare", "choose_view", "scan_wrist", "present_bottom", "review", "recover", "place", "evaluate")
KEYFRAME = re.compile(r"(\d+)-(\d+)-([A-Za-z_]+)-(frame|crop)\.png")
MAX_TELEMETRY_COLUMNS = 8
STEP_EPS_S = 1e-3
DEFECT_MIN_HALF_T_M = 0.0008  # visual half thickness of a defect stroke (the simulated one is 0.1 mm)


def import_rerun():
	try:
		import rerun as rr
	except ImportError as e:  # pragma: no cover - exercised only without the extra
		raise RuntimeError("rerun-sdk is not installed; run `uv sync --extra viewer`") from e
	return rr


def _rgba8(color) -> list[int]:
	c = np.clip(np.asarray(color, dtype=float), 0, 1)
	if c.size == 3:
		c = np.append(c, 1.0)
	return [round(v * 255) for v in c[:4]]


def box_half_sizes(node: dict) -> list[float]:
	"""Half sizes of a box node. The defect strokes are 0.2 mm thick and 0.3 mm off the brick face: at viewer depth
	precision they z-fight with the brick and vanish, so they are drawn thick enough to stand clear of the face."""
	size = np.asarray(node.get("size", [0.01, 0.01, 0.01]), dtype=float)
	scale = np.asarray(node.get("scale", [1.0, 1.0, 1.0]), dtype=float)
	half = (size * scale).tolist()
	if str(node.get("name", "")).startswith("defect_"):
		half[2] = max(half[2], DEFECT_MIN_HALF_T_M)
	return half


def log_scene_geometry(scene: dict, *, prefix: str = "world", recording=None) -> None:
	"""Static geometry of every scene node (meshes, boxes, spheres, capsules); poses are logged separately."""
	rr = import_rerun()
	rr.log(prefix, rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True, recording=recording)
	meshes = scene.get("meshes", {})
	for node in scene["nodes"]:
		path = f"{prefix}/{node['id']}"
		color = _rgba8(node.get("color", [0.7, 0.7, 0.7, 1]))
		scale = np.asarray(node.get("scale", [1.0, 1.0, 1.0]), dtype=float)
		size = np.asarray(node.get("size", [0.01, 0.01, 0.01]), dtype=float)
		kind = node.get("type", "box")
		if kind == "mesh" and node.get("mesh") in meshes:
			m = meshes[node["mesh"]]
			rr.log(
				path,
				rr.Mesh3D(
					vertex_positions=np.asarray(m["vertices"], dtype=np.float32) * scale.astype(np.float32),
					triangle_indices=np.asarray(m["faces"], dtype=np.uint32),
					albedo_factor=color,
				),
				static=True,
				recording=recording,
			)
		elif kind == "sphere":
			r = float(size[0] * scale[0])
			rr.log(path, rr.Ellipsoids3D(half_sizes=[[r, r, r]], colors=[color]), static=True, recording=recording)
		elif kind in ("cylinder", "capsule"):
			rr.log(
				path,
				rr.Capsules3D(
					lengths=[float(2 * size[1] * scale[2])], radii=[float(size[0] * scale[0])], colors=[color]
				),
				static=True,
				recording=recording,
			)
		elif kind == "plane":
			half = [float(size[0]) or 1.0, float(size[1]) or 1.0, 0.001]
			rr.log(path, rr.Boxes3D(half_sizes=[half], colors=[color]), static=True, recording=recording)
		else:  # box, and anything unknown, shown as its bounding box
			rr.log(
				path, rr.Boxes3D(half_sizes=[box_half_sizes(node)], colors=[color]), static=True, recording=recording
			)


def log_scene(scene: dict, poses: np.ndarray, times: np.ndarray, *, prefix: str = "world") -> None:
	"""Static geometry per node plus its pose over time (``poses`` is ``[T, N, 7]``: xyz + xyzw, metres, Z-up)."""
	rr = import_rerun()
	log_scene_geometry(scene, prefix=prefix)
	time_col = rr.TimeColumn("sim_time", duration=np.asarray(times, dtype=float))
	for i, node in enumerate(scene["nodes"]):
		rr.send_columns(
			f"{prefix}/{node['id']}",
			indexes=[time_col],
			columns=rr.Transform3D.columns(translation=poses[:, i, :3], quaternion=poses[:, i, 3:]),
		)


def log_telemetry(telemetry: dict[str, np.ndarray], times: np.ndarray) -> None:
	"""Small per-frame arrays (forces, velocities ...) as scalar series; large or non-numeric arrays are skipped."""
	rr = import_rerun()
	time_col = rr.TimeColumn("sim_time", duration=np.asarray(times, dtype=float))
	for name, arr in telemetry.items():
		arr = np.asarray(arr)
		if (
			name.startswith("full_")
			or arr.dtype.kind not in "fiu"
			or arr.ndim not in (1, 2)
			or arr.shape[0] != len(times)
		):
			continue
		cols = arr[:, None] if arr.ndim == 1 else arr
		if cols.shape[1] > MAX_TELEMETRY_COLUMNS:
			continue
		for j in range(cols.shape[1]):
			suffix = "" if arr.ndim == 1 else f"/{j}"
			rr.send_columns(
				f"telemetry/{name}{suffix}",
				indexes=[time_col],
				columns=rr.Scalars.columns(scalars=cols[:, j].astype(float)),
			)


def stats_markdown(s: dict) -> str:
	"""A compact markdown report of one episode's timing."""

	def table(rows, key, cols, title):
		if not rows:
			return ""
		head = (
			f"| {title} | n | total s | mean s | p50 | p95 | max | share |\n|---|---:|---:|---:|---:|---:|---:|---:|\n"
		)
		body = "".join(
			f"| {r[key]} | {r['count']} | {r['total']:.2f} | {r['mean']:.2f} | {r['p50']:.2f} | {r['p95']:.2f} | "
			f"{r['max']:.2f} | {r['share'] * 100:.0f}% |\n"
			for r in rows
		)
		return f"{head}{body}\n"

	t, d, cur = s["totals"], s["decisions"], s["state"]
	lines = [
		"# Episode timing",
		(
			f"current: **{cur['node']}**{' (finished)' if cur['finished'] else ''} · wall {t['wall_s']:.1f} s · "
			f"sim {t['sim_s']:.1f} s · {t['visits']} visits · {t['recoveries']} recoveries"
		),
		"",
	]
	lines.append("## State machine, wall time\n" + table(s["nodes"]["wall"], "node", None, "node"))
	lines.append("## State machine, sim time\n" + table(s["nodes"]["sim"], "node", None, "node"))
	if s["phases"]["sim"]:
		lines.append("## Expert phases, sim time\n" + table(s["phases"]["sim"], "phase", None, "phase"))
	if d["count"]:
		lat = (
			"n/a"
			if d["latency_mean_s"] is None
			else f"mean {d['latency_mean_s']:.2f} s, max {d['latency_max_s']:.2f} s"
		)
		lines.append(
			f"## Decisions\n{d['count']} decisions, sources {d['sources']}, fallbacks {d['fallbacks']} "
			f"({d['fallback_rate'] * 100:.0f}%), latency {lat}"
		)
	return "\n".join(lines)


def _describe(e: dict) -> str:
	kind = e.get("kind")
	extra = ""
	if kind == "decision":
		extra = f" -> {e.get('choice')} [{e.get('source')}]"
		if e.get("latency_s") is not None:
			extra += f" {float(e['latency_s']):.2f}s"
		if e.get("fallback_reason"):
			extra += f" fallback: {e['fallback_reason']}"
	elif kind == "node":
		extra = f" enter {e.get('node')}"
	elif kind in ("view", "detector_started"):
		extra = f" {e.get('camera', '')}"
	return f"[{e.get('node')}] {kind}{extra}"


class EpisodeLogger:
	"""Incrementally logs an episode directory; safe to call repeatedly while the episode is being written."""

	def __init__(self, path: Path, *, app_id: str = "armigo-episode", fps: int = st.FPS, recording=None):
		self.rr = import_rerun()
		self.path = Path(path)
		self.fps = fps
		self.nodes = list(DEFAULT_NODES)
		graph = self.path / "graph.json"
		if graph.is_file():
			self.nodes = list(json.loads(graph.read_text(encoding="utf-8")).get("nodes", self.nodes))
		self.events_logged = 0
		self.current: str | None = None
		self.current_index = -1.0
		self.branches_logged: set[int] = set()
		self.keyframes_logged: set[str] = set()
		self.visits_logged: set[int] = set()
		self.step_series: set[str] = set()
		self._live_mtime = 0.0
		if recording is None:
			self.rr.init(app_id, spawn=False)
		self.rec = recording

	def _log(self, path, archetype, **kw):
		self.rr.log(path, archetype, recording=self.rec, **kw)

	def _send(self, path, **kw):
		self.rr.send_columns(path, recording=self.rec, **kw)

	def _set_time(self, timeline, **kw):
		self.rr.set_time(timeline, recording=self.rec, **kw)

	def _step_series(self, path: str) -> None:
		"""Draw a state signal as exact steps: the viewer's default plot aggregation averages a step into a ramp."""
		if path in self.step_series:
			return
		self.step_series.add(path)
		self._log(path, self.rr.SeriesLines(aggregation_policy="Off", interpolation_mode="StepAfter"), static=True)

	# -- time ---------------------------------------------------------------------------------------------------------
	def _at(self, sim: float | None, wall: float | None = None) -> None:
		if sim is not None:
			self._set_time("sim_time", duration=float(sim))
		if wall is not None:
			self._set_time("wall_time", duration=float(wall))

	# -- state machine --------------------------------------------------------------------------------------------------
	def log_graph(self) -> None:
		rr = self.rr
		graph = self.path / "graph.json"
		edges = json.loads(graph.read_text(encoding="utf-8")).get("edges", []) if graph.is_file() else []
		lines = [
			"# State machine",
			"nodes (index order): " + ", ".join(f"{i}={n}" for i, n in enumerate(self.nodes)),
			"",
		]
		lines += [f"- {a} -> {b}" for a, b in edges]
		self._log("state/graph", rr.TextDocument("\n".join(lines), media_type=rr.MediaType.MARKDOWN), static=True)

	def log_events(self, events: list[dict]) -> int:
		"""Log events not logged yet; returns how many were new."""
		rr = self.rr
		new = events[self.events_logged :]
		for e in new:
			sim, wall = e.get("sim_time_s"), e.get("wall_time_s")
			self._at(sim, wall)
			self._log("state/log", rr.TextLog(_describe(e)))
			if e.get("kind") == "node":
				node = e["node"]
				self._step_series(f"state/active/{node}")
				self._step_series("state/node_index")
				index = float(self.nodes.index(node)) if node in self.nodes else -1.0
				if self.current is not None and self.current != node:
					# Plots join samples with straight lines; a sample just before the change makes a step instead of a ramp.
					self._at(None if sim is None else sim - STEP_EPS_S, None if wall is None else wall - STEP_EPS_S)
					self._log(f"state/active/{self.current}", rr.Scalars(1.0))
					self._log("state/node_index", rr.Scalars(self.current_index))
					self._at(sim, wall)
					self._log(f"state/active/{self.current}", rr.Scalars(0.0))
				self._log(f"state/active/{node}", rr.Scalars(1.0))
				self._log("state/node_index", rr.Scalars(index))
				self.current, self.current_index = node, index
			elif e.get("kind") == "decision" and e.get("latency_s") is not None:
				self._log("state/decision_latency_s", rr.Scalars(float(e["latency_s"])))
		self.events_logged = len(events)
		return len(new)

	def log_visit_durations(self, visits: list[dict]) -> None:
		rr = self.rr
		for v in visits:
			if v["open"] or v["index"] in self.visits_logged:
				continue
			self.visits_logged.add(v["index"])
			self._at(v["sim0"] + v["sim_s"], v["wall0"] + v["wall_s"])
			self._log(f"state/visit_wall_s/{v['node']}", rr.Scalars(v["wall_s"]))

	def log_stats(self, s: dict) -> None:
		self._log(
			"stats/summary", self.rr.TextDocument(stats_markdown(s), media_type=self.rr.MediaType.MARKDOWN), static=True
		)

	# -- finished branches ----------------------------------------------------------------------------------------------
	def log_branch(
		self, index: int, *, start_frame: int = 0, state_names: list[str] | None = None
	) -> np.ndarray | None:
		"""Joint state/action and expert phases of a finished branch; returns its phase array."""
		rr = self.rr
		actions = self.path / f"branch-{index}" / "actions.npz"
		if index in self.branches_logged or not actions.is_file():
			return None
		with np.load(actions, allow_pickle=False) as z:
			state, action = z["state"], z["action"]
			phase = z["phase"] if "phase" in z.files else None
		n = len(state)
		times = (start_frame + np.arange(n)) / self.fps
		time_col = rr.TimeColumn("sim_time", duration=times)
		names = state_names or [f"j{j}" for j in range(state.shape[1])]
		for j, name in enumerate(names[: state.shape[1]]):
			self._send(
				f"robot/state/{name}", indexes=[time_col], columns=rr.Scalars.columns(scalars=state[:, j].astype(float))
			)
			self._send(
				f"robot/action/{name}",
				indexes=[time_col],
				columns=rr.Scalars.columns(scalars=action[:, j].astype(float)),
			)
		if phase is not None:
			segs = st.phase_segments(phase, fps=self.fps, offset=start_frame)
			order = {}
			idx = np.array([order.setdefault(str(p), len(order)) for p in phase], dtype=float)
			self._send("phase/index", indexes=[time_col], columns=rr.Scalars.columns(scalars=idx))
			self._send(
				"phase/log",
				indexes=[rr.TimeColumn("sim_time", duration=np.array([s["frame0"] / self.fps for s in segs]))],
				columns=rr.TextLog.columns(text=[f"{s['phase']} ({s['sim_s']:.2f} s sim)" for s in segs]),
			)
		self.branches_logged.add(index)
		return phase

	def log_keyframes(self) -> None:
		rr = self.rr
		for p in sorted((self.path / "keyframes").glob("*.png")):
			m = KEYFRAME.fullmatch(p.name)
			if not m or p.name in self.keyframes_logged:
				continue
			_, frame, camera, kind = m.groups()
			self._at(int(frame) / self.fps)
			self._log(f"camera/{camera}/{kind}", rr.EncodedImage(path=str(p)))
			self.keyframes_logged.add(p.name)

	def log_live_image(self, sim: float | None) -> None:
		image = self.path / "live.jpg"
		if not image.is_file() or image.stat().st_mtime <= self._live_mtime:
			return
		self._live_mtime = image.stat().st_mtime
		self._at(sim, None)
		try:
			self._log("camera/live", self.rr.EncodedImage(contents=image.read_bytes(), media_type="image/jpeg"))
		except OSError:
			pass  # being replaced by the recorder; the next poll gets it

	# -- driver ---------------------------------------------------------------------------------------------------------
	def episode_meta(self) -> dict:
		end = self.path / "episode.json"
		return json.loads(end.read_text(encoding="utf-8")) if end.is_file() else {}

	def refresh(self) -> dict:
		"""One pass over everything that exists now; returns the current statistics."""
		events = st.read_events(self.path / "events.jsonl")
		meta = self.episode_meta()
		finished = bool(meta)
		self.log_events(events)
		self.log_keyframes()
		fps = int(meta.get("fps", self.fps)) if meta else self.fps
		self.fps = fps
		phases: list[str] = []
		for b in meta.get("branches", []):
			p = self.log_branch(
				int(b["id"]), start_frame=int(b.get("start_frame", 0)), state_names=meta.get("state_names")
			)
			if p is not None:
				phases += [str(x) for x in p]
		if finished and not phases:
			# Already logged earlier in this process: rebuild the phase list for the statistics only.
			for b in meta.get("branches", []):
				f = self.path / f"branch-{b['id']}" / "actions.npz"
				if f.is_file():
					with np.load(f, allow_pickle=False) as z:
						phases += [str(x) for x in z["phase"]] if "phase" in z.files else []
		sim = events[-1]["sim_time_s"] if events else None
		s = st.episode_stats(events, phases or None, fps=fps, finished=finished)
		self.log_visit_durations(s["visits"])
		self.log_stats(s)
		if not finished:
			self.log_live_image(sim)
		return s

	def follow(
		self, *, poll_s: float = 0.4, stop: Callable[[], bool] = lambda: False, idle_exit_s: float | None = None
	) -> dict:
		"""Poll a live episode until it finishes (``episode.json`` appears), ``stop()`` is true or it goes quiet."""
		self.log_graph()
		last_change = time.monotonic()
		seen = -1
		while True:
			s = self.refresh()
			if s["state"]["finished"] or stop():
				return s
			if self.events_logged != seen:
				seen, last_change = self.events_logged, time.monotonic()
			elif idle_exit_s is not None and time.monotonic() - last_change > idle_exit_s:
				return s
			time.sleep(poll_s)


def blueprint(*, has_scene: bool):
	"""A layout with the state timeline and the statistics next to the 3D view / camera."""
	import rerun.blueprint as rrb

	left = (
		rrb.Spatial3DView(origin="world", name="3D") if has_scene else rrb.Spatial2DView(origin="camera", name="Camera")
	)
	return rrb.Blueprint(
		rrb.Horizontal(
			rrb.Vertical(left, rrb.TextLogView(origin="state/log", name="Events"), row_shares=[3, 1]),
			rrb.Vertical(
				rrb.TimeSeriesView(origin="state/active", name="State machine"),
				rrb.TimeSeriesView(origin="robot/state", name="Joints"),
				rrb.TextDocumentView(origin="stats/summary", name="Timing"),
				row_shares=[2, 2, 3],
			),
			column_shares=[3, 2],
		),
		collapse_panels=True,
	)
