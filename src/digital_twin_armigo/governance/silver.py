"""Silver layer: per-episode validation checks, seed split assignment, audit join, and quarantine."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable
from pathlib import Path

from ..experiments.scoring import score_episode
from ..inventory import audit as inv_audit
from .bronze import FILES, _commit
from .pairs import flags, pair_key, sensing_of
from .segments import split_of

__all__ = ["CHECKS", "QUARANTINE_CHECKS", "build_silver"]

QUARANTINE_CHECKS = ("complete", "bronze_intact", "sensor_guard", "no_truth_fallback")


def _read_json(path: Path) -> dict:
	try:
		data = json.loads(path.read_text(encoding="utf-8-sig"))
		return data if isinstance(data, dict) else {}
	except (OSError, ValueError):
		return {}


def _read_events(path: Path) -> list[dict]:
	if not path.is_file():
		return []
	out = []
	for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
		line = line.strip()
		if not line:
			continue
		try:
			obj = json.loads(line)
			if isinstance(obj, dict):
				out.append(obj)
		except ValueError:
			break
	return out


def _seed_of(task: dict, row: dict) -> int | None:
	case = task.get("case") or []
	if isinstance(case, list) and case:
		try:
			return int(case[0])
		except (TypeError, ValueError):
			pass
	lid = str(row.get("layout_id") or "")
	if lid.startswith("L") and lid[1:].isdigit():
		return int(lid[1:])
	return None


def _sensing_of(row: dict, episode: dict) -> str:
	group = row.get("group")
	if group:
		return sensing_of(str(group))
	prov = episode.get("provenance") or {}
	exp = prov.get("experiment") or {}
	s = exp.get("sensing") or prov.get("sensing")
	if s in ("truth", "realistic"):
		return str(s)
	return "unknown"


def _check_complete(row: dict, _path: Path, _ctx: dict) -> tuple[bool, str | None]:
	ok = row.get("run_state") == "done" and bool(row.get("has_episode"))
	return (True, None) if ok else (False, f"run_state={row.get('run_state')},has_episode={row.get('has_episode')}")


def _check_bronze_intact(_row: dict, _path: Path, ctx: dict) -> tuple[bool, str | None]:
	if ctx.get("has_integrity_event"):
		return False, "integrity_event"
	return True, None


def _check_sensor_guard(_row: dict, _path: Path, ctx: dict) -> tuple[bool, str | None]:
	if ctx.get("sensing") != "realistic":
		ctx["sensor_guard_basis"] = "truth"
		return True, None
	calls = ctx.get("tool_calls") or []
	if not calls:
		ctx["sensor_guard_basis"] = "no_tool_calls"
		return True, None
	ctx["sensor_guard_basis"] = "tool_calls"
	from ..agent import step_mcp as sm

	for e in calls:
		ret = e.get("returned")
		if ret is None:
			continue
		try:
			sm.assert_sensor_only(ret)
		except Exception as exc:  # noqa: BLE001 - capture any SimulatorLeak / DefectLeakageError detail
			return False, f"{type(exc).__name__}: {exc}"
	return True, None


def _check_no_truth_fallback(_row: dict, _path: Path, ctx: dict) -> tuple[bool, str | None]:
	if ctx.get("sensing") != "realistic":
		return True, None
	if ctx.get("oracle_events") and not ctx.get("is_oracle"):
		return False, "oracle_applied_without_oracle_flag"
	return True, None


def _check_teacher_frozen(_row: dict, _path: Path, ctx: dict) -> tuple[bool, str | None]:
	freeze = ctx.get("freeze")
	if not freeze:
		return True, None
	ok = ctx.get("provenance_commit") == freeze.get("code_commit")
	return (True, None) if ok else (False, "commit_mismatch")


def _check_latest_attempt(_row: dict, _path: Path, ctx: dict) -> tuple[bool, str | None]:
	ok = not bool(ctx.get("superseded"))
	return (True, None) if ok else (False, "superseded")


def _check_split_known(_row: dict, _path: Path, ctx: dict) -> tuple[bool, str | None]:
	ok = ctx.get("split") != "unknown"
	return (True, None) if ok else (False, "unknown_split")


CHECKS: dict[str, Callable[[dict, Path, dict], tuple[bool, str | None]]] = {
	"complete": _check_complete,
	"bronze_intact": _check_bronze_intact,
	"sensor_guard": _check_sensor_guard,
	"no_truth_fallback": _check_no_truth_fallback,
	"teacher_frozen": _check_teacher_frozen,
	"latest_attempt": _check_latest_attempt,
	"split_known": _check_split_known,
}


def _build_id(bronze_rows: list[tuple], integrity_rows: list[tuple], commit: str | None, freeze: dict | None) -> str:
	h = hashlib.sha256()
	for r in bronze_rows:
		h.update(json.dumps(list(r), separators=(",", ":")).encode("utf-8"))
	for r in integrity_rows:
		h.update(json.dumps(list(r), separators=(",", ":")).encode("utf-8"))
	h.update((commit or "").encode("utf-8"))
	if freeze is not None:
		h.update(json.dumps(freeze, sort_keys=True, separators=(",", ":")).encode("utf-8"))
	return h.hexdigest()[:12]


def _auto_audit_path(ep_path: Path) -> Path | None:
	for parent in ep_path.parents:
		if parent.name == "runs":
			cand = parent.parent / "audit" / "reviews.sqlite"
			if cand.is_file():
				return cand
	return None


def _infer_traj_id(ep_path: Path, source: str) -> str | None:
	for parent in ep_path.parents:
		if parent.name == "runs":
			rel = ep_path.relative_to(parent).as_posix()
			return f"agent:{source}:{rel}"
	return None


def build_silver(
	con: sqlite3.Connection,
	*,
	freeze: dict | None = None,
	audit_path: Path | None = None,
) -> str:
	"""Validate and normalize every episode in ``bronze`` into ``silver_episode``; returns ``build_id``."""
	cols = ["path", "source", *FILES.keys()]
	bronze_rows = con.execute(f"select {', '.join(cols)} from bronze order by path").fetchall()
	integrity_rows = con.execute(
		"select path, field, old_sha, new_sha, seen_at from integrity_event order by id"
	).fetchall()
	commit = _commit()
	bid = _build_id(bronze_rows, integrity_rows, commit, freeze)
	integrity_paths = {r[0] for r in integrity_rows}

	prepared = []
	best_attempt: dict[tuple, tuple] = {}
	for b in bronze_rows:
		p_str, source = b[0], b[1]
		sha_episode = b[2]
		ep_dir = Path(p_str)
		row = score_episode(ep_dir)
		episode = _read_json(ep_dir / "episode.json")
		task = _read_json(ep_dir / "task.json")
		events = _read_events(ep_dir / "events.jsonl")
		prov = episode.get("provenance") or {}
		exp = prov.get("experiment") or {}
		row["motion"] = exp.get("motion") or "open"
		row["evidence"] = exp.get("evidence") or "hold"
		sensing = _sensing_of(row, episode)
		oracle_events = [e for e in events if e.get("kind") == "oracle_applied"]
		is_oracle = bool(exp.get("oracle")) or bool(oracle_events)
		row["oracle"] = is_oracle
		seed = _seed_of(task, row)
		split = split_of(seed)
		pk = pair_key(row) if sensing in ("truth", "realistic") and row.get("engine") and row.get("detector") else None
		attempt_key = (row.get("batch"), row.get("engine"), row.get("group"), row.get("case_id"))
		rank = (int(row.get("attempt") or 0), ep_dir.name, p_str)
		if attempt_key not in best_attempt or rank > best_attempt[attempt_key]:
			best_attempt[attempt_key] = rank
		prepared.append(
			{
				"p_str": p_str,
				"source": source,
				"sha_episode": sha_episode,
				"ep_dir": ep_dir,
				"row": row,
				"events": events,
				"sensing": sensing,
				"seed": seed,
				"split": split,
				"pair_key": pk,
				"attempt_key": attempt_key,
				"rank": rank,
				"oracle_events": oracle_events,
				"is_oracle": is_oracle,
				"provenance_commit": exp.get("code_commit") or (prov.get("git") or {}).get("commit"),
			}
		)

	con.execute("delete from silver_episode where build_id = ?", (bid,))
	con.execute("delete from silver_pair where build_id = ?", (bid,))

	for item in prepared:
		ep_dir = item["ep_dir"]
		row = item["row"]
		superseded = item["rank"] != best_attempt[item["attempt_key"]]
		tool_calls = [e for e in item["events"] if e.get("kind") == "tool_call"]
		ctx = {
			"freeze": freeze,
			"has_integrity_event": item["p_str"] in integrity_paths,
			"sensing": item["sensing"],
			"tool_calls": tool_calls,
			"oracle_events": item["oracle_events"],
			"is_oracle": item["is_oracle"],
			"provenance_commit": item["provenance_commit"],
			"superseded": superseded,
			"split": item["split"],
		}
		check_results = {}
		quarantine_reasons = []
		for name, fn in CHECKS.items():
			passed, detail = fn(row, ep_dir, ctx)
			check_results[name] = {"passed": bool(passed), "detail": detail}
			if not passed and name in QUARANTINE_CHECKS:
				quarantine_reasons.append(name)
		quarantined = bool(quarantine_reasons)

		ap = audit_path or _auto_audit_path(ep_dir)
		latest_audit = None
		if ap is not None and ap.is_file():
			tid = _infer_traj_id(ep_dir, item["source"])
			if tid:
				latest_audit = inv_audit.latest(ap, tid)

		row_flags = flags({**row, "oracle": item["is_oracle"]})
		row["quarantined"] = quarantined
		enriched = {
			**row,
			"sensing": item["sensing"],
			"pair_key": list(item["pair_key"]) if item["pair_key"] is not None else None,
			"seed": item["seed"],
			"split": item["split"],
			"sha_episode": item["sha_episode"],
			"teacher_frozen": check_results["teacher_frozen"]["passed"],
			"superseded": superseded,
			"quarantined": quarantined,
			"quarantine_reasons": quarantine_reasons,
			"sensor_guard_basis": ctx.get("sensor_guard_basis"),
			"checks": check_results,
			"flags": row_flags,
			"audit_state": (latest_audit or {}).get("state"),
			"audit_root_cause": (latest_audit or {}).get("root_cause"),
			"audit_mechanism": (latest_audit or {}).get("mechanism"),
		}
		con.execute(
			"insert into silver_episode (build_id, path, sensing, pair_key, split, quarantined, quarantine_reasons, row_json) "
			"values (?, ?, ?, ?, ?, ?, ?, ?)",
			(
				bid,
				item["p_str"],
				item["sensing"],
				json.dumps(enriched["pair_key"]) if enriched["pair_key"] is not None else None,
				item["split"],
				int(quarantined),
				json.dumps(quarantine_reasons),
				json.dumps(enriched, ensure_ascii=False),
			),
		)
	con.commit()
	return bid
