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
from .pairs import OUTCOMES, flags, pair_key, pair_outcome, sensing_of
from .segments import split_of

__all__ = ["CHECKS", "QUARANTINE_CHECKS", "build_pairs", "build_silver", "pair_report"]

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


def _build_id(
	bronze_rows: list[tuple],
	integrity_rows: list[tuple],
	commit: str | None,
	freeze: dict | None,
	audit_snapshot: list[tuple] | None = None,
) -> str:
	h = hashlib.sha256()
	for r in bronze_rows:
		h.update(json.dumps(list(r), separators=(",", ":")).encode("utf-8"))
	for r in integrity_rows:
		h.update(json.dumps(list(r), separators=(",", ":")).encode("utf-8"))
	h.update((commit or "").encode("utf-8"))
	if freeze is not None:
		h.update(json.dumps(freeze, sort_keys=True, separators=(",", ":")).encode("utf-8"))
	if audit_snapshot:
		for a in audit_snapshot:
			h.update(json.dumps(list(a), separators=(",", ":")).encode("utf-8"))
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
	integrity_paths = {r[0] for r in integrity_rows}

	prepared = []
	best_attempt: dict[tuple, tuple] = {}
	audit_cache: dict[Path, dict[str, dict]] = {}
	audit_snapshot: list[tuple] = []
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

		ap = audit_path or _auto_audit_path(ep_dir)
		latest_audit = None
		if ap is not None and ap.is_file():
			ap_res = Path(ap).resolve()
			if ap_res not in audit_cache:
				audit_cache[ap_res] = inv_audit.latest_all(ap_res)
			tid = _infer_traj_id(ep_dir, source)
			if tid:
				latest_audit = audit_cache[ap_res].get(tid)
		if latest_audit is not None:
			audit_snapshot.append(
				(
					p_str,
					latest_audit.get("id"),
					latest_audit.get("state"),
					latest_audit.get("root_cause"),
					latest_audit.get("mechanism"),
				)
			)

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
				"latest_audit": latest_audit,
			}
		)

	bid = _build_id(bronze_rows, integrity_rows, commit, freeze, audit_snapshot)
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
		latest_audit = item["latest_audit"]

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


def build_pairs(con: sqlite3.Connection, build_id: str) -> int:
	"""Populate ``silver_pair`` for ``build_id`` from latest (non-superseded) episodes; returns pair count."""
	con.execute("delete from silver_pair where build_id = ?", (build_id,))
	rows = [
		json.loads(r[0])
		for r in con.execute("select row_json from silver_episode where build_id = ? order by path", (build_id,))
	]
	truth_by_pk: dict[str, dict] = {}
	real_by_pk_cam: dict[tuple[str, str], dict] = {}
	for r in rows:
		if r.get("superseded") or not r.get("pair_key"):
			continue
		pk_json = json.dumps(r["pair_key"])
		if r.get("sensing") == "truth":
			prev = truth_by_pk.get(pk_json)
			if prev is None or (int(r.get("attempt") or 0), r["path"]) > (int(prev.get("attempt") or 0), prev["path"]):
				truth_by_pk[pk_json] = r
		elif r.get("sensing") == "realistic":
			cam = str(r.get("camera_condition") or "fixed")
			key = (pk_json, cam)
			prev = real_by_pk_cam.get(key)
			if prev is None or (int(r.get("attempt") or 0), r["path"]) > (int(prev.get("attempt") or 0), prev["path"]):
				real_by_pk_cam[key] = r

	paired_truth_pks: set[str] = set()
	inserts: list[tuple] = []
	for (pk_json, cam), r_row in sorted(real_by_pk_cam.items()):
		t_row = truth_by_pk.get(pk_json)
		if t_row is not None:
			paired_truth_pks.add(pk_json)
		outcome = pair_outcome(t_row, r_row)
		inserts.append((build_id, pk_json, cam, t_row["path"] if t_row else None, r_row["path"], outcome))

	for pk_json, t_row in sorted(truth_by_pk.items()):
		if pk_json not in paired_truth_pks:
			inserts.append((build_id, pk_json, "fixed", t_row["path"], None, pair_outcome(t_row, None)))

	con.executemany(
		"insert into silver_pair (build_id, pair_key, camera_condition, truth_path, realistic_path, outcome) "
		"values (?, ?, ?, ?, ?, ?)",
		inserts,
	)
	con.commit()
	return len(inserts)


def pair_report(con: sqlite3.Connection, build_id: str) -> str:
	"""Render Markdown report of silver pair outcomes by batch x camera, realistic_only, truth_only cross-tab and quarantine."""
	ep_rows = {
		r[0]: json.loads(r[1])
		for r in con.execute("select path, row_json from silver_episode where build_id = ?", (build_id,))
	}
	pairs = con.execute(
		"select pair_key, camera_condition, truth_path, realistic_path, outcome "
		"from silver_pair where build_id = ? order by camera_condition, pair_key",
		(build_id,),
	).fetchall()

	by_batch_cam: dict[tuple[str, str], dict[str, int]] = {}
	realistic_only_items: list[dict] = []
	truth_only_cross: dict[tuple[str, str], int] = {}
	for _pk, cam, t_path, r_path, outcome in pairs:
		r_ep = ep_rows.get(r_path) if r_path else None
		t_ep = ep_rows.get(t_path) if t_path else None
		ref = r_ep or t_ep or {}
		batch = str(ref.get("batch") or "-")
		bucket = by_batch_cam.setdefault((batch, cam), {o: 0 for o in OUTCOMES})
		bucket[outcome] = bucket.get(outcome, 0) + 1
		if outcome == "realistic_only":
			realistic_only_items.append(
				{
					"batch": batch,
					"camera": cam,
					"case_id": ref.get("case_id"),
					"truth_path": t_path,
					"realistic_path": r_path,
					"truth_cause": (t_ep or {}).get("root_cause"),
				}
			)
		elif outcome == "truth_only" and r_ep is not None:
			rc = str(r_ep.get("root_cause") or "other")
			mech = str(r_ep.get("mechanism") or "(none)")
			truth_only_cross[(rc, mech)] = truth_only_cross.get((rc, mech), 0) + 1

	quarantine_counts: dict[str, int] = {}
	for ep in ep_rows.values():
		for reason in ep.get("quarantine_reasons") or []:
			quarantine_counts[reason] = quarantine_counts.get(reason, 0) + 1

	lines = [f"# Silver Pair Report (`{build_id}`)", "", "## Batch x Camera Outcomes", ""]
	lines.append("| batch | camera_condition | " + " | ".join(OUTCOMES) + " |")
	lines.append("|---|---|" + "---:|" * len(OUTCOMES))
	for (batch, cam), counts in sorted(by_batch_cam.items()):
		lines.append(f"| {batch} | {cam} | " + " | ".join(str(counts.get(o, 0)) for o in OUTCOMES) + " |")

	lines += ["", "## realistic_only Episodes", ""]
	if not realistic_only_items:
		lines.append("(none)")
	else:
		for item in realistic_only_items:
			lines.append(
				f"- `{item['batch']}` `{item['camera']}` `{item['case_id']}` "
				f"(truth `{item['truth_cause']}`: `{item['truth_path']}`, realistic: `{item['realistic_path']}`)"
			)

	lines += ["", "## truth_only (root_cause x mechanism)", ""]
	if not truth_only_cross:
		lines.append("(none)")
	else:
		mechs = sorted({m for _, m in truth_only_cross})
		rcs = sorted({rc for rc, _ in truth_only_cross})
		lines.append("| root_cause | " + " | ".join(mechs) + " |")
		lines.append("|---|" + "---:|" * len(mechs))
		for rc in rcs:
			lines.append(f"| {rc} | " + " | ".join(str(truth_only_cross.get((rc, m), 0)) for m in mechs) + " |")

	lines += ["", "## Quarantine Reasons", ""]
	if not quarantine_counts:
		lines.append("(none)")
	else:
		lines += ["| reason | count |", "|---|---:|"]
		for reason, n in sorted(quarantine_counts.items()):
			lines.append(f"| {reason} | {n} |")

	return "\n".join(lines) + "\n"
