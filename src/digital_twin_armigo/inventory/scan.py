"""Walk data roots and describe every synthetic trajectory, scene and derived dataset found there.

Three adapters, all read-only (nothing is copied, renamed or rewritten):

* **agent episodes** — directories holding an ``episode.json`` of schema ``armigo-agent-episode/v1`` (agent runs, expert
  tracer teachers, learned-policy rollouts, DAgger takeovers);
* **catalog records** — the ``episodes`` table of ``catalog.sqlite`` (scripted physics-expert and Isaac stage-2 runs);
* **datasets** — render datasets (``layouts.jsonl`` + ``metadata.jsonl``, one scene per layout) and LeRobot exports
  (``meta/info.json``), registered as derived products, never descended into.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections.abc import Iterator
from pathlib import Path

from digital_twin_armigo.inventory import taxonomy

AGENT_SCHEMA = "armigo-agent-episode/v1"
# Never descend into these: media and arrays of a dataset, or the insides of an episode directory.
PRUNE = {
	".git",
	".venv",
	"__pycache__",
	"node_modules",
	"png",
	"images",
	"videos",
	"meshes",
	"poses",
	"keyframes",
	"checkpoints",
	"mlflow",
}
SPLIT_TOKENS = (
	("train", "train"),
	("val", "val"),
	("calib", "calibration"),
	("held", "held_out"),
	("test", "test"),
)


def sha256_file(path: Path) -> str:
	h = hashlib.sha256()
	with open(path, "rb") as f:
		for block in iter(lambda: f.read(1 << 20), b""):
			h.update(block)
	return h.hexdigest()


def _json(path: Path):
	try:
		return json.loads(path.read_text(encoding="utf-8-sig"))
	except (OSError, ValueError):
		return None


def scene_id_of(case) -> str:
	"""Stable id of a seeded case ``[seed, defective, face(, options)]``."""
	seed, defective, face, *rest = list(case)
	options = (
		f":{hashlib.sha1(json.dumps(rest[0], sort_keys=True).encode()).hexdigest()[:8]}" if rest and rest[0] else ""
	)
	return f"case:{int(seed)}:{int(bool(defective))}:{int(face)}{options}"


def _rounded(obj):
	if isinstance(obj, float):
		return round(obj, 4)
	if isinstance(obj, dict):
		return {k: _rounded(v) for k, v in obj.items()}
	if isinstance(obj, list | tuple):
		return [_rounded(v) for v in obj]
	return obj


def layout_id_of(layout: dict) -> str:
	"""Stable id of a stored layout: its geometry rounded to 0.1 mm."""
	return "layout:" + hashlib.sha1(json.dumps(_rounded(layout), sort_keys=True).encode()).hexdigest()[:12]


def split_hint(path: Path) -> str | None:
	for part in reversed([p.lower() for p in path.parts]):
		for token, split in SPLIT_TOKENS:
			if token in part:
				return split
	return None


def walk(root: Path) -> Iterator[tuple[Path, list[str], list[str]]]:
	"""``os.walk`` that lets the caller prune ``dirs`` in place (like os.walk) and skips unreadable directories."""
	for cur, dirs, files in os.walk(root, onerror=lambda _e: None):
		dirs[:] = [d for d in dirs if d not in PRUNE]
		yield Path(cur), dirs, files


# -- agent episodes -------------------------------------------------------------------------------------------------------
def agent_episode(directory: Path, root_label: str, root: Path) -> dict | None:
	file = directory / "episode.json"
	ep = _json(file)
	if not isinstance(ep, dict) or ep.get("schema") != AGENT_SCHEMA:
		return None
	prov, result = ep.get("provenance") or {}, ep.get("result") or {}
	run = _json(directory / "run.json") or {}
	task = _json(directory / "task.json") or {}
	case = prov.get("case") or run.get("case") or task.get("case")
	written_at = file.stat().st_mtime
	actor = taxonomy.classify_provenance(prov, result, written_at=written_at)
	names = {p.name for p in directory.iterdir()}
	frames = result.get("sim_frames")
	fps = ep.get("fps") or 30
	events = directory / "events.jsonl"
	wall = _last_wall_time(events) if events.is_file() else None
	takeover = ep.get("takeover") or {}
	rel = directory.relative_to(root).as_posix()
	return {
		"traj_id": f"agent:{root_label}:{rel}",
		"record_kind": "agent_episode",
		"root": root_label,
		"path": str(directory),
		"run_group": _run_group(directory, root),
		"category": actor["category"],
		"control": actor["control"],
		"sensing": actor["sensing"],
		"category_basis": actor["basis"],
		"category_note": actor.get("note"),
		"engine": prov.get("engine", "mujoco"),
		"scene_id": scene_id_of(case) if case else None,
		"seed": int(case[0]) if case else None,
		"defective": bool(case[1]) if case else None,
		"defect_face": int(case[2]) if case else None,
		"verdict": result.get("verdict"),
		"label": result.get("label"),
		"label_correct": result.get("label_correct"),
		"sim_frames": frames,
		"fps": fps,
		"sim_s": round(frames / fps, 3) if frames else None,
		"wall_s": wall,
		"views": result.get("views"),
		"rewinds": result.get("rewinds"),
		"retries": result.get("retries"),
		"llm_consulted": (result.get("llm") or {}).get("consulted"),
		"camera_profile": prov.get("camera_profile"),
		"detector_sha256": prov.get("detector_sha256"),
		"takeover_gate": takeover.get("gate") if takeover.get("triggered") else None,
		"takeover_frame": takeover.get("frame") if takeover.get("triggered") else None,
		"has_actions": any((directory / n / "actions.npz").is_file() for n in names if n.startswith("branch-")),
		"has_events": "events.jsonl" in names,
		"has_keyframes": "keyframes" in names,
		"has_video": bool(ep.get("video")),
		"has_motion": (directory / "replay" / "motion.npz").is_file(),
		"split_hint": split_hint(directory.relative_to(root)),
		"written_at": file.stat().st_mtime,
		"episode_sha256": sha256_file(file),
	}


def _last_wall_time(events: Path) -> float | None:
	try:
		with open(events, "rb") as f:
			f.seek(0, os.SEEK_END)
			f.seek(max(0, f.tell() - 4096))
			tail = f.read().decode("utf-8", "replace").splitlines()
	except OSError:
		return None
	for line in reversed(tail):
		try:
			wall = json.loads(line).get("wall_time_s")
		except ValueError:
			continue  # half-written or cut at the seek boundary
		if wall is not None:
			return round(float(wall), 3)
	return None


def _run_group(directory: Path, root: Path) -> str:
	"""The run an episode belongs to: its path up to the ``case-NNNN`` / ``batch`` directory."""
	return "/".join(directory.relative_to(root).parts[:-1]) or root.name


def scan_agent_episodes(root: Path, label: str) -> Iterator[dict]:
	for cur, dirs, files in walk(root):
		if "episode.json" not in files:
			continue
		row = agent_episode(cur, label, root)
		if row is not None:
			dirs[:] = []  # an episode directory holds only its own files
			yield row


# -- datasets and scenes --------------------------------------------------------------------------------------------------
def scan_datasets(root: Path, label: str) -> Iterator[tuple[dict, list[dict]]]:
	"""``(dataset row, scene rows)`` for render datasets and LeRobot exports under ``root``."""
	for cur, dirs, files in walk(root):
		rel = cur.relative_to(root).as_posix() or cur.name
		if "layouts.jsonl" in files and "metadata.jsonl" in files:
			yield _render_dataset(cur, label, rel)
			dirs[:] = []
		elif (cur / "meta" / "info.json").is_file() and "data" in dirs:
			yield _lerobot_dataset(cur, label, rel)
			dirs[:] = []


def _manifest(path: Path) -> dict:
	m = _json(path / "dataset_manifest.json") or {}
	return {"digest": m.get("digest"), "manifest_commit": (m.get("git") or {}).get("commit")}


def _render_dataset(path: Path, label: str, rel: str) -> tuple[dict, list[dict]]:
	summary = _json(path / "summary.json") or {}
	scenes = []
	did = f"dataset:{label}:{rel}"
	for line in (path / "layouts.jsonl").read_text(encoding="utf-8").splitlines():
		if not line.strip():
			continue
		try:
			lay = json.loads(line)
		except ValueError:
			continue  # half-written tail
		scenes.append(
			{
				"scene_id": f"render:{label}:{rel}:{lay.get('layout_index')}",
				"scene_kind": "render_layout",
				"engine": _engine_of_name(rel),
				"dataset_id": did,
				"seed": lay.get("defect_seed"),
				"defective": lay.get("defective"),
				"defect_kind": lay.get("defect_kind"),
				"defect_face": lay.get("defect_face"),
				"brick_size": lay.get("brick_size"),
				"split": lay.get("split"),
				"status": lay.get("status"),
				"layout": lay,
			}
		)
	row = {
		"dataset_id": did,
		"name": rel,
		"kind": "render_dataset",
		"root": label,
		"path": str(path),
		"engine": _engine_of_name(rel),
		"n_scenes": len(scenes),
		"n_items": summary.get("n_crops"),
		"item_unit": "crops",
		"summary": {k: summary.get(k) for k in ("layout_status", "by_stage", "by_split_label") if k in summary},
		**_manifest(path),
	}
	return row, scenes


def _lerobot_dataset(path: Path, label: str, rel: str) -> tuple[dict, list[dict]]:
	info = _json(path / "meta" / "info.json") or {}
	row = {
		"dataset_id": f"dataset:{label}:{rel}",
		"name": rel,
		"kind": "lerobot_export",
		"root": label,
		"path": str(path),
		"engine": None,
		"n_scenes": None,
		"n_items": info.get("total_episodes"),
		"item_unit": "episodes",
		"summary": {k: info.get(k) for k in ("total_frames", "fps", "robot_type", "codebase_version") if k in info},
		**_manifest(path),
	}
	return row, []


def _engine_of_name(name: str) -> str:
	return "isaac" if "isaac" in name.lower() else "mujoco"


# -- catalog --------------------------------------------------------------------------------------------------------------
def scan_catalog(db: Path, label: str) -> Iterator[dict]:
	"""One row per ``episodes`` record of ``catalog.sqlite`` (opened read-only)."""
	con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
	con.row_factory = sqlite3.Row
	try:
		artifacts: dict[str, set[str]] = {}
		for r in con.execute(
			"select ea.episode_id, a.role from episode_artifacts ea join artifacts a on a.id = ea.artifact_id"
		):
			artifacts.setdefault(r["episode_id"], set()).add(r["role"])
		splits: dict[str, list[str]] = {}
		for r in con.execute(
			"select m.episode_id, d.name, m.split from memberships m join datasets d on d.id = m.dataset_id"
		):
			splits.setdefault(r["episode_id"], []).append(f"{r['name']}:{r['split']}")
		sources = {r["id"]: r["path"] for r in con.execute("select id, path from sources")}
		for r in con.execute("select * from episodes"):
			raw = _loads(r["raw"])
			prov = _loads(r["provenance"])
			layout = raw.get("layout") or {}
			info = raw.get("info") or {}
			actor = taxonomy.classify_catalog(r["engine"])
			frames = info.get("n_frames") or raw.get("total_frames")
			roles = artifacts.get(r["id"], set())
			scene = layout_id_of(layout) if layout else None
			yield {
				"traj_id": f"catalog:{r['id'][:16]}",
				"record_kind": "catalog_episode",
				"root": label,
				"path": sources.get(r["source_id"], ""),
				"run_group": f"{Path(sources.get(r['source_id'], '')).parent.as_posix()}",
				"category": actor["category"],
				"control": actor["control"],
				"sensing": actor["sensing"],
				"category_basis": actor["basis"],
				"category_note": actor["note"],
				"engine": r["engine"],
				"scene_id": scene or r["layout_group"],
				"seed": prov.get("layout_seed"),
				"defective": layout.get("defective"),
				"defect_face": None,
				"verdict": r["verdict"],
				"label": None,
				"label_correct": None,
				"sim_frames": frames,
				"fps": raw.get("control_fps") or 30,
				"sim_s": None,
				"wall_s": None,
				"views": None,
				"rewinds": None,
				"retries": info.get("grasp_retries"),
				"llm_consulted": None,
				"camera_profile": None,
				"detector_sha256": None,
				"takeover_gate": None,
				"takeover_frame": None,
				"has_actions": False,
				"has_events": False,
				"has_keyframes": False,
				"has_video": bool(roles & {"video", "rtx_video"}),
				"has_motion": "motion" in roles,
				"split_hint": ";".join(splits.get(r["id"], [])) or r["role"],
				"written_at": None,
				"episode_sha256": None,
				"catalog_role": r["role"],
				"quality_status": r["quality_status"],
				"condition": r["condition"],
				"alias": r["alias"],
				"_layout": layout or None,
			}
	finally:
		con.close()


def _loads(text) -> dict:
	try:
		out = json.loads(text) if text else {}
	except ValueError:
		return {}
	return out if isinstance(out, dict) else {}
