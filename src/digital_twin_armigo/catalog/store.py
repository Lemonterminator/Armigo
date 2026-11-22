"""Metadata-only SQLite catalog with immutable source identities and explicit exports."""

from __future__ import annotations

import json
import math
import re
import sqlite3
from contextlib import closing, contextmanager
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from digital_twin_armigo.runtime.canonical import sha256_file, sha256_json

from .quality import evaluate_quality, layout_group, quality_status, split_for_group

SCHEMA = """
CREATE TABLE IF NOT EXISTS sources(id TEXT PRIMARY KEY,path TEXT NOT NULL,sha256 TEXT NOT NULL,imported_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS episodes(id TEXT PRIMARY KEY,source_id TEXT NOT NULL REFERENCES sources(id),locator TEXT NOT NULL,alias TEXT NOT NULL,engine TEXT NOT NULL,verdict TEXT NOT NULL,condition TEXT,quality_status TEXT NOT NULL,layout_group TEXT,role TEXT NOT NULL,raw TEXT NOT NULL,provenance TEXT NOT NULL,quality TEXT NOT NULL,UNIQUE(source_id,locator));
CREATE INDEX IF NOT EXISTS episode_filters ON episodes(engine,verdict,quality_status);
CREATE TABLE IF NOT EXISTS artifacts(id TEXT PRIMARY KEY,path TEXT NOT NULL,sha256 TEXT NOT NULL,size INTEGER NOT NULL,role TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS episode_artifacts(episode_id TEXT REFERENCES episodes(id),artifact_id TEXT REFERENCES artifacts(id),PRIMARY KEY(episode_id,artifact_id));
CREATE TABLE IF NOT EXISTS derivations(parent TEXT REFERENCES episodes(id),child TEXT REFERENCES episodes(id),relation TEXT NOT NULL,PRIMARY KEY(parent,child,relation));
CREATE TABLE IF NOT EXISTS datasets(id TEXT PRIMARY KEY,name TEXT UNIQUE NOT NULL,seed INTEGER NOT NULL,created_at TEXT NOT NULL,manifest TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS memberships(dataset_id TEXT REFERENCES datasets(id),episode_id TEXT REFERENCES episodes(id),layout_group TEXT NOT NULL,split TEXT NOT NULL,PRIMARY KEY(dataset_id,episode_id));
CREATE TABLE IF NOT EXISTS evaluations(id TEXT PRIMARY KEY,episode_id TEXT REFERENCES episodes(id),created_at TEXT NOT NULL,rules TEXT NOT NULL,checks TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS replay_evidence(episode_id TEXT REFERENCES episodes(id),source_sha256 TEXT NOT NULL,level TEXT NOT NULL,checks TEXT NOT NULL,PRIMARY KEY(episode_id,source_sha256));
CREATE TABLE IF NOT EXISTS capture_validation(episode_id TEXT PRIMARY KEY REFERENCES episodes(id),status TEXT NOT NULL,details TEXT NOT NULL);
"""


def plain(value: Any) -> Any:
	if is_dataclass(value):
		return plain(asdict(value))
	if isinstance(value, dict):
		return {str(k): plain(v) for k, v in value.items()}
	if isinstance(value, (list, tuple)):
		return [plain(v) for v in value]
	if hasattr(value, "tolist"):
		return plain(value.tolist())
	if isinstance(value, float) and not math.isfinite(value):
		return None
	return value


def dumps(value: Any) -> str:
	return json.dumps(plain(value), ensure_ascii=False, sort_keys=True, allow_nan=False)


def now() -> str:
	return datetime.now(UTC).isoformat()


class Catalog:
	def __init__(self, root: Path, home: Path | None = None):
		self.root = Path(root).resolve()
		self.home = Path(home or self.root / "data/catalog").resolve()
		self.home.mkdir(parents=True, exist_ok=True)
		self.path = self.home / "catalog.sqlite"
		with self.connect() as db:
			version = db.execute("PRAGMA user_version").fetchone()[0]
			if version not in (0, 1):
				raise ValueError(f"unsupported catalog schema {version}")
			db.executescript(SCHEMA)
			db.execute("PRAGMA user_version=1")
			for table in ("sources", "artifacts"):
				for row in db.execute(f"SELECT id,path FROM {table}").fetchall():
					path = Path(row["path"])
					if path.is_absolute() and path.is_relative_to(self.root):
						db.execute(
							f"UPDATE {table} SET path=? WHERE id=?", (path.relative_to(self.root).as_posix(), row["id"])
						)

	@contextmanager
	def connect(self):
		db = sqlite3.connect(self.path, timeout=30)
		db.row_factory = sqlite3.Row
		db.execute("PRAGMA foreign_keys=ON")
		db.execute("PRAGMA journal_mode=WAL")
		try:
			with db:
				yield db
		finally:
			db.close()

	def project_path(self, path: str | Path) -> Path:
		p = Path(path)
		p = (p if p.is_absolute() else self.root / p).resolve()
		if (
			not p.is_relative_to(self.root)
			or "agent_logs" in [s.lower() for s in p.parts]
			or p.suffix.lower() == ".jsonl"
		):
			raise ValueError("only project data outside agent_logs can be indexed")
		return p

	def source(self, path: Path) -> str:
		p = self.project_path(path)
		digest = sha256_file(p)
		ident = sha256_json([str(p.relative_to(self.root)), digest])
		with self.connect() as db:
			db.execute(
				"INSERT OR IGNORE INTO sources VALUES(?,?,?,?)",
				(ident, p.relative_to(self.root).as_posix(), digest, now()),
			)
		return ident

	def import_snapshot(self, store_root: Path, ledger_path: Path, snapshot_id: str) -> dict:
		from digital_twin_armigo.runtime.snapshot import SnapshotStore

		root = self.project_path(store_root)
		ledger = self.project_path(ledger_path)
		if not all((root / name).is_dir() for name in ("blobs", "manifests", "staging")):
			raise ValueError("an existing snapshot store is required")
		store = SnapshotStore(root)
		manifest = store.verify(snapshot_id)
		producer = manifest.produced_by or {}
		with closing(sqlite3.connect(ledger.as_uri() + "?mode=ro", uri=True)) as db:
			events = db.execute(
				"SELECT seq,job_id,payload_json FROM job_events WHERE job_id=? AND state='SUCCEEDED' ORDER BY seq DESC",
				(producer.get("job_id"),),
			).fetchall()
			attempts = db.execute(
				"SELECT payload_json FROM job_events WHERE job_id=? AND state='RUNNING' AND fencing_token=?",
				(producer.get("job_id"), producer.get("fencing_token")),
			).fetchall()
		matching = next((r for r in events if json.loads(r[2]).get("output_snapshot_id") == snapshot_id), None)
		if matching is None or not any(json.loads(a[0]).get("output_snapshot_id") == snapshot_id for a in attempts):
			raise ValueError("snapshot is not published by its producer in the supplied ledger")
		payload = json.loads(matching[2])
		result = producer.get("result") or {}
		row = {"verdict": result.get("verdict", "UNKNOWN"), "health": result.get("health", {}), "info": {}}
		provenance = {
			"original_code_identity": manifest.identity,
			"snapshot_id": snapshot_id,
			"original_run_id": manifest.run_id,
			"original_episode_id": manifest.episode_id,
			"contract": manifest.contract,
			"scene_spec_sha256": manifest.scene_sha256,
			"publication_event": {"seq": matching[0], "job_id": matching[1], "payload": payload},
			"publication_evidence_sha256": sha256_json(list(matching)),
			"imported_at": now(),
			"evidence": "verified_manifest_blobs_and_publication",
		}
		ident = self.add_episode(
			root / "manifests" / f"{snapshot_id}.json",
			"/",
			f"snapshot_{snapshot_id}",
			manifest.identity["engine"],
			row,
			provenance,
			manifest.scene_spec,
			"snapshot",
		)
		for name, ref in manifest.blobs.items():
			slug = re.sub(r"[^a-z0-9_]", "_", name.lower())[:40]
			self.add_artifact(ident, root / "blobs" / ref.sha256, "snapshot_" + slug + "_" + sha256_json(name)[:8])
		return {"id": ident, "snapshot_id": snapshot_id, "verified": True}

	def add_artifact(self, episode: str, path: Path, role: str) -> None:
		if not isinstance(role, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", role):
			raise ValueError("invalid artifact role")
		p = self.project_path(path)
		if not p.is_file():
			return
		digest = sha256_file(p)
		reference = p.relative_to(self.root).as_posix()
		ident = sha256_json([reference, digest, role])
		with self.connect() as db:
			old = db.execute(
				"SELECT id FROM artifacts WHERE path=? AND sha256=? AND role=?", (reference, digest, role)
			).fetchone()
			if old:
				ident = old[0]
			db.execute(
				"INSERT OR IGNORE INTO artifacts VALUES(?,?,?,?,?)", (ident, reference, digest, p.stat().st_size, role)
			)
			db.execute("INSERT OR IGNORE INTO episode_artifacts VALUES(?,?)", (episode, ident))

	def add_episode(
		self,
		path: Path,
		locator: str,
		alias: str,
		engine: str,
		row: dict,
		provenance: dict,
		scene: dict | None = None,
		role="historical",
	) -> str:
		source_id = self.source(path)
		ident = sha256_json([source_id, locator])
		checks = evaluate_quality(row.get("health") or {}, row.get("info") or {})
		verdict = row.get("verdict") or (
			"SUCCESS" if row.get("success") is True else "FAILURE" if row.get("success") is False else "UNKNOWN"
		)
		if verdict not in ("SUCCESS", "FAILURE", "UNKNOWN", "INFEASIBLE"):
			verdict = "UNKNOWN"
		with self.connect() as db:
			db.execute(
				"INSERT OR IGNORE INTO episodes VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
				(
					ident,
					source_id,
					locator,
					alias,
					engine,
					verdict,
					row.get("condition"),
					quality_status(checks),
					layout_group(scene or {}),
					role,
					dumps(row),
					dumps(provenance),
					dumps(checks),
				),
			)
		self.add_artifact(ident, path, "source")
		return ident

	def import_path(self, path: str | Path) -> dict:
		p = self.project_path(path)
		if p.is_dir():
			files = []
			for item in sorted(p.rglob("*.json")):
				if "agent_logs" in item.parts or (item.is_relative_to(self.home) and not p.is_relative_to(self.home)):
					continue
				# Explicit data schemas only, never transcripts or configuration files.
				try:
					obj = json.loads(item.read_text(encoding="utf-8"))
				except (ValueError, UnicodeError):
					continue
				if isinstance(obj, dict) and (
					"episodes_detail" in dumps(list(obj.keys()))
					or any(k in obj for k in ("conditions", "layouts", "recorded_episode"))
					or obj.get("schema") == "catalog-episode/v1"
					or item.name in ("stage2_metrics.json", "index.json")
				):
					files.append(item)
			ids = []
			for f in files:
				ids.extend(self.import_file(f))
			return {"files": len(files), "episodes": len(set(ids)), "ids": list(dict.fromkeys(ids))}
		if not p.is_file():
			raise FileNotFoundError(p)
		ids = self.import_file(p)
		return {"files": 1, "episodes": len(ids), "ids": ids}

	def import_file(self, p: Path) -> list[str]:
		d = json.loads(p.read_text(encoding="utf-8"))
		if not isinstance(d, dict):
			return []
		ids = []
		prov = {
			"original": {
				k: d.get(k)
				for k in (
					"mujoco_version",
					"evaluator_version",
					"physics_spec_version",
					"physics_manifest",
					"seed",
					"params",
					"sensor_noise",
					"source",
				)
			},
			"original_code_identity": d.get("identity"),
			"imported_at": now(),
			"source_sha256": sha256_file(p),
			"historical_missing_is_unknown": True,
		}
		if d.get("schema") == "catalog-episode/v1":
			if d.get("parent_episode_id"):
				self.episode(d["parent_episode_id"])
			declared = {
				**d["provenance"],
				"imported_at": now(),
				"manifest_source_sha256": sha256_file(p),
				"identity_evidence": "producer_manifest",
			}
			e = self.add_episode(p, "/", d["alias"], d["engine"], d["result"], declared, d["scene"], "recorded")
			for role, name in d["artifacts"].items():
				self.add_artifact(e, p.parent / name, role)
			parent = d.get("parent_episode_id")
			if parent:
				with self.connect() as db:
					if db.execute("SELECT 1 FROM episodes WHERE id=?", (parent,)).fetchone():
						db.execute("INSERT OR IGNORE INTO derivations VALUES(?,?,?)", (parent, e, "rerun"))
						role = db.execute("SELECT role FROM episodes WHERE id=?", (parent,)).fetchone()[0]
						if role in ("legacy_held_out", "independent_acceptance"):
							db.execute("UPDATE episodes SET role=? WHERE id=?", (role, e))
			return [e]
		if "conditions" in d and isinstance(d["conditions"], dict):
			for name, condition in d["conditions"].items():
				for i, row in enumerate(condition.get("episodes_detail", [])):
					role = (
						"legacy_held_out"
						if name.startswith("combined_")
						else "independent_acceptance"
						if "independent_acceptance" in p.parts
						else "development"
					)
					ids.append(
						self.add_episode(
							p,
							f"/conditions/{name}/episodes_detail/{i}",
							f"{name}_seed{row.get('seed')}",
							"mujoco",
							{**row, "condition": name},
							prov,
							role=role,
						)
					)
		elif "layouts" in d and isinstance(d["layouts"], dict):
			for i, row in enumerate(d["layouts"].get("episodes_detail", [])):
				ids.append(
					self.add_episode(
						p,
						f"/layouts/episodes_detail/{i}",
						f"layouts_seed{d.get('layout_seed')}_idx{row.get('layout_index', i):02d}",
						"mujoco",
						row,
						{
							**prov,
							"layout_seed": d.get("layout_seed"),
							"layout_count": len(d["layouts"].get("episodes_detail", [])),
						},
						scene={"layout": row.get("layout", {})},
						role="independent_acceptance" if "independent_acceptance" in p.parts else "development",
					)
				)
		elif "recorded_episode" in d:
			# Attach old videos to matching source records, not to a fabricated new experiment.
			origin = self.project_path(d["source"])
			candidates = self.import_file(origin) if origin.is_file() else []
			for e in candidates:
				detail = self.episode(e)
				if detail["alias"] == d["id"] and dumps(
					{k: v for k, v in detail["raw"].items() if k != "condition"}
				) == dumps({k: v for k, v in d["recorded_episode"].items() if k != "condition"}):
					ids.append(e)
			if not ids:
				ids = [
					self.add_episode(
						p,
						"/recorded_episode",
						d["id"],
						"mujoco",
						d["recorded_episode"],
						{**prov, "replay_check": d.get("replay_check"), "replay_evidence": "summary_only"},
					)
				]
			for e in ids:
				for ext, role in ((".mp4", "video"), (".npz", "legacy_telemetry"), (".json", "legacy_replay_report")):
					self.add_artifact(e, p.with_suffix(ext), role)
				with self.connect() as db:
					db.execute(
						"INSERT OR IGNORE INTO replay_evidence VALUES(?,?,?,?)",
						(e, sha256_file(p), "summary_only", dumps(d.get("replay_check", {}))),
					)
		elif p.name == "index.json" and isinstance(d.get("episodes"), list):
			for i, row in enumerate(d["episodes"]):
				engine = (
					"isaac"
					if "isaac" in d.get("source", "")
					else "mujoco"
					if "mujoco" in d.get("source", "")
					else "unknown"
				)
				metrics = p.parent.parent / "stage2_metrics.json"
				result = json.loads(metrics.read_text()) if metrics.is_file() else row
				e = self.add_episode(
					p,
					f"/episodes/{i}",
					row["file"],
					engine,
					result,
					{**prov, "recording_index": d, "original_metrics": result if metrics.is_file() else None},
				)
				self.add_artifact(e, p.parent / row["file"], "observations")
				ids.append(e)
		elif p.name == "stage2_metrics.json":
			ids = [self.add_episode(p, "/", str(p.parent.relative_to(self.root)), "isaac", d, prov)]
			for f in p.parent.glob("*.mp4"):
				self.add_artifact(ids[0], f, "video")
		return ids

	def list(self, search="", verdict="", engine="", quality="", limit=50, offset=0) -> dict:
		clauses = []
		args = []
		for key, val in (("verdict", verdict), ("engine", engine), ("quality_status", quality)):
			if val:
				clauses.append(f"e.{key}=?")
				args.append(val)
		if search:
			clauses.append("(e.alias LIKE ? OR e.id LIKE ? OR e.condition LIKE ?)")
			args.extend([f"%{search}%"] * 3)
		where = " WHERE " + " AND ".join(clauses) if clauses else ""
		with self.connect() as db:
			total = db.execute("SELECT COUNT(*) FROM episodes e" + where, args).fetchone()[0]
			rows = db.execute(
				"SELECT e.id,e.alias,e.engine,e.verdict,e.quality_status,e.condition,e.role,e.layout_group,s.path source_path FROM episodes e JOIN sources s ON s.id=e.source_id"
				+ where
				+ " ORDER BY e.rowid DESC LIMIT ? OFFSET ?",
				[*args, min(max(int(limit), 1), 5000), max(int(offset), 0)],
			).fetchall()
		return {"items": [dict(r) for r in rows], "total": total}

	def artifact(self, ident: str) -> dict:
		with self.connect() as db:
			r = db.execute("SELECT * FROM artifacts WHERE id=?", (ident,)).fetchone()
		if r is None:
			raise KeyError(ident)
		return dict(r)

	def episode(self, ident: str) -> dict:
		with self.connect() as db:
			r = db.execute(
				"SELECT e.*,s.path source_path FROM episodes e JOIN sources s ON s.id=e.source_id WHERE e.id=?",
				(ident,),
			).fetchone()
			if r is None:
				raise KeyError(ident)
			out = dict(r)
			out["artifacts"] = [
				{**dict(a), "name": Path(a["path"]).name, "url": f"/api/artifacts/{a['id']}"}
				for a in db.execute(
					"SELECT a.* FROM artifacts a JOIN episode_artifacts ea ON a.id=ea.artifact_id WHERE ea.episode_id=?",
					(ident,),
				)
			]
			out["relations"] = [
				dict(x) for x in db.execute("SELECT * FROM derivations WHERE parent=? OR child=?", (ident, ident))
			]
		for k in ("raw", "provenance", "quality"):
			out[k] = json.loads(out[k])
		for a in out["artifacts"]:
			a["path_ref"] = a["path"]
			a["path"] = str(self.project_path(a["path"]))
		with self.connect() as db:
			out["quality_evaluations"] = [
				dict(r)
				for r in db.execute(
					"SELECT id,created_at,rules FROM evaluations WHERE episode_id=? ORDER BY rowid", (ident,)
				)
			]
			out["replay_evidence"] = [
				dict(r)
				for r in db.execute(
					"SELECT source_sha256,level,checks FROM replay_evidence WHERE episode_id=?", (ident,)
				)
			]
		with self.connect() as db:
			validation = db.execute(
				"SELECT status,details FROM capture_validation WHERE episode_id=?", (ident,)
			).fetchone()
		out["capture_validation"] = dict(validation) if validation else out["provenance"].get("capture_validation", {})
		out["quality_policy"] = (
			json.loads(out["quality_evaluations"][-1]["rules"])
			if out["quality_evaluations"]
			else {
				"version": "catalog-quality/v1",
				"defaults": {
					"unintended_force_N": 3,
					"jaw_environment_N": 5,
					"release_speed_mps": 1.2,
					"near_frame_cap": 890,
					"pinch_force_N": None,
					"torque_saturation_seconds": 1.5,
				},
			}
		)
		roles = {a["role"] for a in out["artifacts"]}
		out["capabilities"] = {
			"rtx": {"motion", "scene"} <= roles,
			"observations": "observations" in roles,
			"mujoco": "mjcf" in roles and "motion" in roles,
		}
		if out["engine"] == "isaac" and "motion" in roles and out["capture_validation"].get("status") != "PASS":
			out["capabilities"]["rtx"] = False
			out["capture_warning"] = "live-body capture validation is absent; observations remain available"
		return out

	def summary(self) -> dict:
		with self.connect() as db:
			return {
				"episodes": db.execute("SELECT COUNT(*) FROM episodes").fetchone()[0],
				"artifacts": db.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0],
				"verdicts": dict(db.execute("SELECT verdict,COUNT(*) FROM episodes GROUP BY verdict")),
				"exports_automatic": False,
			}

	def freeze_split(self, ids: list[str], name: str, seed: int = 0) -> dict:
		if not ids or not name.strip():
			raise ValueError("explicit episode selection and a name are required")
		members = []
		for ident in sorted(set(ids)):
			e = self.episode(ident)
			if not e["layout_group"]:
				raise ValueError(f"{e['alias']}: complete verified layout is required")
			if e["role"] in ("legacy_held_out", "independent_acceptance"):
				raise ValueError("historical evaluation records cannot be assigned to a new training split")
			members.append(
				{
					"episode_id": ident,
					"layout_group": e["layout_group"],
					"split": split_for_group(e["layout_group"], seed),
				}
			)
		manifest = {
			"schema": "catalog-split/v1",
			"name": name,
			"seed": seed,
			"ratios": [0.8, 0.1, 0.1],
			"members": members,
		}
		ident = sha256_json(manifest)
		with self.connect() as db:
			existing = db.execute("SELECT id FROM datasets WHERE name=?", (name,)).fetchone()
			if existing and existing[0] != ident:
				raise ValueError("a frozen dataset name cannot be reused with different members")
			db.execute("INSERT OR IGNORE INTO datasets VALUES(?,?,?,?,?)", (ident, name, seed, now(), dumps(manifest)))
			db.executemany(
				"INSERT OR IGNORE INTO memberships VALUES(?,?,?,?)",
				[(ident, m["episode_id"], m["layout_group"], m["split"]) for m in members],
			)
		return {"id": ident, **manifest}

	def datasets(self) -> dict:
		"""Frozen dataset versions: id (content hash of the manifest), name, seed and per-split member counts."""
		with self.connect() as db:
			rows = db.execute("SELECT id,name,seed,created_at FROM datasets ORDER BY created_at,name").fetchall()
			counts = {
				(r["dataset_id"], r["split"]): r["n"]
				for r in db.execute("SELECT dataset_id,split,COUNT(*) AS n FROM memberships GROUP BY dataset_id,split")
			}
		return {
			"datasets": [
				{**dict(r), "splits": {s: counts.get((r["id"], s), 0) for s in ("train", "validation", "test")}}
				for r in rows
			]
		}

	def memberships(self, episode: str) -> list[dict]:
		with self.connect() as db:
			return [dict(r) for r in db.execute("SELECT * FROM memberships WHERE episode_id=?", (episode,))]

	def evaluate(self, ids: list[str], rules: dict) -> dict:
		if not ids:
			raise ValueError("explicit selection required")
		allowed = {
			"unintended_force_N",
			"jaw_environment_N",
			"release_speed_mps",
			"near_frame_cap",
			"pinch_force_N",
			"torque_saturation_seconds",
		}
		if set(rules) - allowed or any(
			isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or v < 0
			for v in rules.values()
		):
			raise ValueError("rules must be known, finite, nonnegative numerical thresholds")
		results = []
		for ident in sorted(set(ids)):
			e = self.episode(ident)
			checks = evaluate_quality(e["raw"].get("health", {}), e["raw"].get("info", {}), rules)
			policy = {"version": "catalog-quality/v1", "overrides": rules}
			evaluation = sha256_json([ident, policy, checks])
			with self.connect() as db:
				db.execute(
					"INSERT OR IGNORE INTO evaluations VALUES(?,?,?,?,?)",
					(evaluation, ident, now(), dumps(policy), dumps(checks)),
				)
				db.execute(
					"UPDATE episodes SET quality=?,quality_status=? WHERE id=?",
					(dumps(checks), quality_status(checks), ident),
				)
			results.append(
				{"id": ident, "evaluation_id": evaluation, "quality_status": quality_status(checks), "rules": policy}
			)
		return {"evaluations": results}

	def verify(self, ids: list[str]) -> dict:
		errors = []
		seen = set()
		for ident in ids:
			for a in self.episode(ident)["artifacts"]:
				if a["id"] in seen:
					continue
				seen.add(a["id"])
				p = self.project_path(a["path"])
				if not p.is_file() or p.stat().st_size != a["size"] or sha256_file(p) != a["sha256"]:
					errors.append({"artifact": a["id"], "path": str(p), "error": "missing_or_changed"})
		return {"ok": not errors, "checked": len(seen), "errors": errors}

	def backup(self, out: Path) -> dict:
		out = self.project_path(out)
		out.mkdir(parents=True, exist_ok=False)
		for source, name in ((self.path, "catalog.sqlite"), (self.home / "jobs.sqlite", "jobs.sqlite")):
			if source.is_file():
				original = sqlite3.connect(source)
				copy = sqlite3.connect(out / name)
				try:
					original.backup(copy)
				finally:
					copy.close()
					original.close()
		(out / "README.txt").write_text(
			"Metadata backup only. Indexed videos, telemetry and assets remain at their original project-relative locations. Use explicit export to package those files.\n",
			encoding="utf-8",
		)
		return {"metadata_backup": str(out), "copies_artifacts": False}

	def validate_motion(self, ids: list[str]) -> dict:
		import numpy as np

		from .playback import load_motion

		if not ids:
			raise ValueError("explicit selection required")
		results = []
		for ident in ids:
			e = self.episode(ident)
			roles = {a["role"]: a["path"] for a in e["artifacts"]}
			if not self.verify([ident])["ok"]:
				raise ValueError("cannot validate changed files")
			scene, poses, times = load_motion(roles)
			result = {
				"schema": "catalog-capture-validation/v1",
				"frames": len(times),
				"status": "PASS",
				"scope": "coordinate_contract",
			}
			if e["engine"] == "isaac":
				matches = [i for i, n in enumerate(scene["nodes"]) if n["name"].endswith("/Brick/Mesh")]
				if (
					len(matches) != 1
					or scene.get("pose_source") != "live_isaaclab_body_buffers_with_usd_geometry_offsets"
				):
					result.update(status="FAIL", reason="live body geometry binding is absent")
				else:
					with np.load(roles["telemetry"], allow_pickle=False) as z:
						position = z["brick_pos"]
						quaternion = z["brick_quat"]
					if len(position) != len(times):
						raise ValueError("motion/telemetry length differs")

					def rotate(q, v):
						return v + 2 * np.cross(q[:3], np.cross(q[:3], v) + q[3] * v)

					inverse = quaternion[0] * np.array([-1, -1, -1, 1])
					offset = rotate(inverse, poses[0, matches[0], :3] - position[0])
					predicted = position + np.array([rotate(q, offset) for q in quaternion])
					error = float(np.max(np.abs(predicted - poses[:, matches[0], :3])))
					result.update(
						status="PASS" if error <= 1e-5 else "FAIL",
						max_brick_position_error_m=error,
						scope="recorded_geometry_vs_independent_brick_telemetry",
					)
			with self.connect() as db:
				db.execute(
					"INSERT OR REPLACE INTO capture_validation VALUES(?,?,?)", (ident, result["status"], dumps(result))
				)
			results.append({"id": ident, **result})
		return {"validations": results, "ok": all(r["status"] == "PASS" for r in results)}
