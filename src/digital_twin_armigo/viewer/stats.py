"""Per-state and per-phase timing of an agent episode (pure functions, no Rerun/FastAPI dependency).

Inputs are what an episode already writes: ``events.jsonl`` (one JSON object per line, ``kind == "node"`` marks the state
machine entering a node; ``decision`` events carry the LLM/policy choice and latency) and, once a branch is finished,
``branch-N/actions.npz`` whose per-frame ``phase`` array holds the expert phase. A node visit lasts from its ``node`` event
to the next ``node`` event of the same branch, so a visit still in progress is ``open`` and is measured up to ``now``.
"""

from __future__ import annotations

import bisect
import json
import math
from collections.abc import Iterable, Sequence
from pathlib import Path

FPS = 30


def read_events(path: Path) -> list[dict]:
	"""Parse an events.jsonl file; a half-written trailing line (the episode is live) ends the list."""
	events: list[dict] = []
	try:
		text = Path(path).read_text(encoding="utf-8")
	except FileNotFoundError:
		return events
	for line in text.splitlines():
		if not line.strip():
			continue
		try:
			events.append(json.loads(line))
		except json.JSONDecodeError:
			break
	return events


def node_visits(events: Sequence[dict], *, now_wall: float | None = None, now_sim: float | None = None) -> list[dict]:
	"""One record per state-machine visit, in order, with sim/wall duration and the decisions made inside it.

	``now_*`` close the last visit of a finished episode (default: the last event) or measure a live one so far.
	"""
	if not events:
		return []
	last_wall = max(float(e.get("wall_time_s", 0.0)) for e in events)
	last_sim = max(float(e.get("sim_time_s", 0.0)) for e in events)
	last_frame = max(int(e.get("frame", 0)) for e in events)
	end_wall = last_wall if now_wall is None else max(now_wall, last_wall)
	end_sim = last_sim if now_sim is None else max(now_sim, last_sim)
	starts = [e for e in events if e.get("kind") == "node"]
	visits: list[dict] = []
	for i, e in enumerate(starts):
		nxt = next((n for n in starts[i + 1 :] if n.get("branch", 0) == e.get("branch", 0)), None)
		open_ = nxt is None
		if open_:
			end = {"wall": end_wall, "sim": end_sim, "frame": last_frame}
		else:
			end = {"wall": float(nxt["wall_time_s"]), "sim": float(nxt["sim_time_s"]), "frame": int(nxt["frame"])}
		visits.append(
			{
				"index": i,
				"node": e["node"],
				"branch": int(e.get("branch", 0)),
				"seq": int(e.get("seq", i)),
				"frame0": int(e["frame"]),
				"frame1": end["frame"],
				"wall0": float(e["wall_time_s"]),
				"wall_s": max(0.0, end["wall"] - float(e["wall_time_s"])),
				"sim0": float(e["sim_time_s"]),
				"sim_s": max(0.0, end["sim"] - float(e["sim_time_s"])),
				"open": open_,
				"decisions": 0,
				"llm_latency_s": 0.0,
			}
		)
	by_seq = {v["seq"]: v for v in visits}
	seqs = sorted(by_seq)
	for e in events:
		if e.get("kind") != "decision":
			continue
		pos = bisect.bisect_right(seqs, int(e.get("seq", 0))) - 1
		# Decisions belong to the visit of the same branch that is current at their seq.
		while pos >= 0 and by_seq[seqs[pos]]["branch"] != int(e.get("branch", 0)):
			pos -= 1
		if pos >= 0:
			v = by_seq[seqs[pos]]
			v["decisions"] += 1
			v["llm_latency_s"] += float(e.get("latency_s") or 0.0)
	return visits


def current_state(events: Sequence[dict], *, finished: bool = False) -> dict:
	"""The node the machine is in now (or ended in), for the live display."""
	nodes = [e for e in events if e.get("kind") == "node"]
	if not nodes:
		return {"node": None, "branch": None, "since_wall_s": None, "finished": finished, "last_event": None}
	last = nodes[-1]
	return {
		"node": last["node"],
		"branch": int(last.get("branch", 0)),
		"frame": int(last["frame"]),
		"since_wall_s": float(last["wall_time_s"]),
		"since_sim_s": float(last["sim_time_s"]),
		"finished": finished,
		"last_event": events[-1].get("kind"),
	}


def phase_segments(phases: Iterable[str], *, fps: int = FPS, wall_at=None, offset: int = 0) -> list[dict]:
	"""Run-length encode a per-frame phase array. ``wall_at(frame)`` (optional) maps frame -> wall seconds."""
	segs: list[dict] = []
	for i, name in enumerate(phases):
		name = str(name)
		if segs and segs[-1]["phase"] == name:
			segs[-1]["frames"] += 1
		else:
			segs.append({"phase": name, "frame0": i + offset, "frames": 1})
	for s in segs:
		s["frame1"] = s["frame0"] + s["frames"]
		s["sim_s"] = s["frames"] / fps
		s["wall_s"] = None if wall_at is None else max(0.0, wall_at(s["frame1"]) - wall_at(s["frame0"]))
	return segs


def wall_interpolator(events: Sequence[dict]):
	"""frame -> wall seconds by linear interpolation between events (wall time is only known at events).

	Events sharing a frame happen while the simulation is paused (e.g. waiting for the LLM); the frame is dated by the last
	of them, when the simulation resumed.
	"""
	latest: dict[int, float] = {}
	for e in events:
		if "frame" in e and "wall_time_s" in e:
			latest[int(e["frame"])] = max(latest.get(int(e["frame"]), 0.0), float(e["wall_time_s"]))
	pts = sorted(latest.items())
	if not pts:
		return None
	frames = [p[0] for p in pts]
	walls = [p[1] for p in pts]

	def wall_at(frame: float) -> float:
		if frame <= frames[0]:
			return walls[0]
		if frame >= frames[-1]:
			return walls[-1]
		j = bisect.bisect_right(frames, frame)
		f0, f1, w0, w1 = frames[j - 1], frames[j], walls[j - 1], walls[j]
		return w0 if f1 == f0 else w0 + (w1 - w0) * (frame - f0) / (f1 - f0)

	return wall_at


def percentile(values: Sequence[float], q: float) -> float:
	"""Linear-interpolated percentile, q in [0, 100]."""
	if not values:
		return math.nan
	xs = sorted(values)
	k = (len(xs) - 1) * q / 100
	lo, hi = math.floor(k), math.ceil(k)
	return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarize(rows: Iterable[dict], key: str, value: str) -> list[dict]:
	"""Group ``rows`` by ``key`` and describe ``value``: count/total/mean/p50/p95/max and share of the grand total."""
	groups: dict[str, list[float]] = {}
	for r in rows:
		if r.get(value) is not None:
			groups.setdefault(str(r[key]), []).append(float(r[value]))
	grand = sum(sum(v) for v in groups.values())
	out = [
		{
			key: name,
			"count": len(v),
			"total": sum(v),
			"mean": sum(v) / len(v),
			"p50": percentile(v, 50),
			"p95": percentile(v, 95),
			"max": max(v),
			"share": (sum(v) / grand) if grand else 0.0,
		}
		for name, v in groups.items()
	]
	return sorted(out, key=lambda r: r["total"], reverse=True)


def decision_stats(events: Sequence[dict]) -> dict:
	"""How decisions were made: source mix, fallbacks (LLM failed, policy used) and LLM latency."""
	ds = [e for e in events if e.get("kind") == "decision"]
	lat = [float(e["latency_s"]) for e in ds if e.get("latency_s") is not None]
	sources: dict[str, int] = {}
	for e in ds:
		sources[str(e.get("source"))] = sources.get(str(e.get("source")), 0) + 1
	fallbacks = sum(1 for e in ds if e.get("fallback_reason"))
	return {
		"count": len(ds),
		"sources": sources,
		"fallbacks": fallbacks,
		"fallback_rate": fallbacks / len(ds) if ds else 0.0,
		"latency_mean_s": sum(lat) / len(lat) if lat else None,
		"latency_p95_s": percentile(lat, 95) if lat else None,
		"latency_max_s": max(lat) if lat else None,
		"attempts": sum(int(e.get("attempts") or 0) for e in ds),
		"prompt_tokens": sum(int(e.get("prompt_tokens") or 0) for e in ds),
	}


def episode_stats(
	events: Sequence[dict],
	phases: Sequence[str] | None = None,
	*,
	fps: int = FPS,
	now_wall: float | None = None,
	now_sim: float | None = None,
	finished: bool = False,
) -> dict:
	"""Everything the viewer shows for one episode: current state, visits, and node/phase statistics."""
	visits = node_visits(events, now_wall=now_wall, now_sim=now_sim)
	has_phases = phases is not None and len(phases) > 0
	segs = phase_segments(phases, fps=fps, wall_at=wall_interpolator(events)) if has_phases else []
	return {
		"state": current_state(events, finished=finished),
		"visits": visits,
		"nodes": {
			"wall": summarize(visits, "node", "wall_s"),
			"sim": summarize(visits, "node", "sim_s"),
		},
		"phase_segments": segs,
		"phases": {"sim": summarize(segs, "phase", "sim_s"), "wall": summarize(segs, "phase", "wall_s")},
		"decisions": decision_stats(events),
		"totals": {
			"wall_s": max((float(e.get("wall_time_s", 0.0)) for e in events), default=0.0),
			"sim_s": max((float(e.get("sim_time_s", 0.0)) for e in events), default=0.0),
			"visits": len(visits),
			"recoveries": sum(1 for v in visits if v["node"] == "recover"),
			"branches": len({v["branch"] for v in visits}),
		},
	}
