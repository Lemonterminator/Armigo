"""Episode scoring for the v2 experiments (read-only, engine independent).

``score_episode(path)`` reads one episode directory (``EpisodeTrace`` layout) and returns one flat dict: case and
experiment keys, expected/agent label, the verdict actually scored (``verdict_final_cycle`` when present), the physical
outcome from the recorded brick pose (``replay/motion.npz``), counts, wall/latency and the first error
(``attribution.first_error``). The physical outcome never uses engine health checks: those stay an extra field.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ..constants import BOWL_R
from .attribution import collect_views, first_error
from .contract import PHYSICAL_OUTCOMES, ROOT_CAUSES, case_id, layout_id

RIM_Z = 0.055  # brick centre below this inside the bowl footprint counts as in the bowl (as twin_mujoco)
LOST_Z = -0.02
LOST_RADIUS = 1.0
HELD_Z = 0.06  # on the table footprint but this high: still in the air (unknown)
EXPECTED_BOWL = {"normal": "green_bowl", "defective": "pink_bowl"}


def _json(path: Path):
	try:
		return json.loads(path.read_text(encoding="utf-8-sig"))
	except (OSError, ValueError):
		return None


def _events(path: Path) -> list[dict]:
	out = []
	try:
		with path.open(encoding="utf-8") as f:
			for line in f:
				line = line.strip()
				if line:
					try:
						out.append(json.loads(line))
					except ValueError:
						break  # truncated last line of an interrupted run
	except OSError:
		pass
	return out


def _layout_geometry(scene: dict | None, case) -> dict:
	layout = (scene or {}).get("layout") or {}
	if "green" in layout and "pink" in layout:
		return layout
	if case and len(case) >= 3:
		try:
			from ..agent.twin_mujoco import sample_case

			lay = sample_case(*case)
			return {
				"green": list(lay.green),
				"pink": list(lay.pink),
				"brick_xy": list(lay.brick_xy),
				"brick_size": list(lay.brick_size),
			}
		except (ImportError, OSError, ValueError, TypeError):
			return {}
	return {}


def _brick_index(scene: dict, poses: np.ndarray, layout: dict) -> int:
	nodes = scene.get("nodes") or []
	for i, n in enumerate(nodes):
		if n.get("name") == "brick":
			return i
	xy = layout.get("brick_xy")
	if xy is not None:
		d = np.hypot(poses[0, :, 0] - xy[0], poses[0, :, 1] - xy[1])
		d[0] = np.inf  # the table
		return int(np.argmin(d))
	return poses.shape[1] - 1


def replay_brick_track(path: Path, scene: dict, layout: dict) -> np.ndarray | None:
	"""Read the brick's dense xyz + xyzw replay track."""
	try:
		with np.load(path / "replay" / "motion.npz") as z:
			poses = np.asarray(z["poses"], dtype=np.float64)
	except (OSError, ValueError, KeyError):
		return None
	if poses.ndim != 3 or not poses.shape[0] or not poses.shape[1] or poses.shape[2] < 3:
		return None
	idx = _brick_index(scene, poses, layout)
	return poses[:, idx].copy() if idx < poses.shape[1] else None


def classify_pose(xyz, green, pink, bowl_r: float = BOWL_R) -> str:
	"""Physical outcome of a final brick position: which bowl, table, lost or unknown (still in the air)."""
	x, y, z = (float(v) for v in xyz)
	if not all(math.isfinite(v) for v in (x, y, z)):
		return "unknown"
	if z < LOST_Z or math.hypot(x, y) > LOST_RADIUS:
		return "lost"
	for name, c in (("green_bowl", green), ("pink_bowl", pink)):
		if c is not None and math.hypot(x - c[0], y - c[1]) < bowl_r:
			return name if z < RIM_Z else "unknown"
	return "unknown" if z > HELD_Z else "table"


def physical_outcome(
	path: Path, result: dict, case, expected: str | None, *, scene: dict | None = None, brick_track=None
) -> dict:
	"""``{"outcome", "source", "final_xyz", "moved_m", "dist_green", "dist_pink", "expected_bowl", "success"}``."""
	path = Path(path)
	expected_bowl = EXPECTED_BOWL.get(expected)
	info = {
		"outcome": "unknown",
		"source": "none",
		"final_xyz": None,
		"moved_m": None,
		"dist_green": None,
		"dist_pink": None,
		"expected_bowl": expected_bowl,
		"success": None,
	}
	scene = _json(path / "replay" / "scene.json") if scene is None else scene
	if scene is not None and (brick_track is not None or (path / "replay" / "motion.npz").is_file()):
		layout = _layout_geometry(scene, case)
		replay = replay_brick_track(path, scene, layout) if brick_track is None else brick_track
		if replay is not None:
			track = replay[:, :3]
			finite = np.isfinite(track).all(axis=1)
			if finite.any():
				final = track[np.flatnonzero(finite)[-1]]
				green, pink = layout.get("green"), layout.get("pink")
				info.update(
					outcome=classify_pose(final, green, pink),
					source="replay",
					final_xyz=[round(float(v), 5) for v in final],
					moved_m=round(float(np.linalg.norm(final[:2] - track[np.flatnonzero(finite)[0]][:2])), 5),
					dist_green=None if green is None else round(float(np.hypot(*(final[:2] - green))), 5),
					dist_pink=None if pink is None else round(float(np.hypot(*(final[:2] - pink))), 5),
				)
				info["success"] = None if expected_bowl is None else info["outcome"] == expected_bowl
				return info
	destination = result.get("destination")
	mapped = {"green": "green_bowl", "pink": "pink_bowl", "inspection_table": "table", "table": "table"}
	if destination in mapped:
		info.update(outcome=mapped[destination], source="result")
	return info


def _counts(events: list[dict], result: dict, views) -> dict:
	llm = result.get("llm") or {}
	images = [e for e in events if e.get("kind") == "llm_image"]
	wall = [e["wall_time_s"] for e in events if isinstance(e.get("wall_time_s"), (int, float))]
	latencies = [e["latency_s"] for e in events if e.get("kind") == "decision" and e.get("latency_s") is not None]
	rewinds = int(result.get("rewinds") or 0) or sum(
		e.get("kind") in ("rewound", "restored", "restore") for e in events
	)
	retries = int(result.get("retries") or 0)
	review_ids = {e.get("id") or e.get("view_id") for e in events if e.get("kind") == "review_view"}
	return {
		"n_scored_views": len(views),
		"n_wrist_views": sum(not v.is_top for v in views),
		"n_top_views": sum(v.is_top for v in views),
		"n_review_views": sum(e.get("kind") == "review_view" for e in events)
		+ len({v.id for v in views if v.review} - review_ids),
		"n_llm_images": len(images),
		"n_llm_images_cached": sum(bool(e.get("cached")) for e in images),
		"n_recoveries": rewinds + retries,
		"n_failures": len(result.get("failures") or []),
		"sim_frames": result.get("sim_frames"),
		"wall_s": round(max(wall), 3) if wall else None,
		"decision_latency_s": round(float(sum(latencies)), 3)
		if latencies
		else float(llm.get("latency_s_total") or 0.0),
	}


def score_episode(path) -> dict:
	"""Score one episode directory. Never raises on a missing or partial episode: that is an ``infra_failure`` row."""
	path = Path(path)
	episode = _json(path / "episode.json")
	task = _json(path / "task.json") or {}
	audit = _json(path / "audit.private.json") or {}
	run_status = _json(path / "run_status.json")
	events = _events(path / "events.jsonl")
	result = (episode or {}).get("result") or {}
	exp = ((episode or {}).get("provenance") or {}).get("experiment") or {}
	case = task.get("case") or []

	expected = audit.get("true_label") or (("defective" if case[1] else "normal") if len(case) > 1 else None)
	label = result.get("label")
	label_ok = bool(label in ("normal", "defective") and label == expected)
	verdict = result.get("verdict")
	verdict_final = result.get("verdict_final_cycle")
	scored_verdict = verdict_final or verdict
	physical = physical_outcome(path, result, case, expected)
	if physical["success"] is None:
		physical_success = label_ok and scored_verdict == "SUCCESS"
	else:
		physical_success = label_ok and bool(physical["success"])
	if not result:
		physical_success = False

	views = collect_views(events)
	error = first_error(episode, events, task, audit, physical, run_status)
	if error["root_cause"] not in ROOT_CAUSES:
		error["root_cause"] = "other"
	outcome = physical["outcome"]
	wrong_bowl = outcome in ("green_bowl", "pink_bowl") and physical["expected_bowl"] not in (None, outcome)

	row = {
		"path": str(path),
		"episode_dir": path.name,
		"case_id": exp.get("case_id") or (case_id(case[0], bool(case[1]), case[2]) if len(case) >= 3 else path.name),
		"layout_id": exp.get("layout_id") or (layout_id(case[0]) if case else path.name.split("-")[0]),
		"batch": exp.get("batch"),
		"engine": exp.get("engine") or ((episode or {}).get("provenance") or {}).get("engine"),
		"group": exp.get("group"),
		"detector": exp.get("detector"),
		"strategy": exp.get("strategy"),
		"camera_condition": exp.get("camera_condition"),
		"attempt": exp.get("attempt"),
		"run_state": (run_status or {}).get("state"),
		"has_episode": bool(result),
		"expected_label": expected,
		"label": label,
		"label_correct": label_ok,
		"verdict": verdict,
		"verdict_final_cycle": verdict_final,
		"scored_verdict": scored_verdict,
		"physical_outcome": outcome if outcome in PHYSICAL_OUTCOMES else "unknown",
		"physical": physical,
		"physical_success": physical_success,
		"defect_miss": expected == "defective" and label == "normal",
		"false_alarm": expected == "normal" and label == "defective",
		"is_unknown": bool(result) and label not in ("normal", "defective"),
		"wrong_bowl": bool(wrong_bowl),
		"infra_failed": error["root_cause"] == "infra_failure",
		"first_error": error,
		"root_cause": error["root_cause"],
		"mechanism": error.get("mechanism"),
		"upstream": error.get("upstream"),
		**_counts(events, result, views),
	}
	return row
