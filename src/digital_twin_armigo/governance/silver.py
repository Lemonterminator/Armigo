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
from .pairs import OUTCOMES, flags, pair_key, pair_outcome, sensing_of, truth_compatible
from .segments import case_split, is_sealed, seed_of_path_or_case

__all__ = ["CHECKS", "QUARANTINE_CHECKS", "build_pairs", "build_silver", "pair_report", "reconcile_silver"]

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
	if ctx.get("oracle_events") and not ctx.get("declared_oracle"):
		return False, "oracle_applied_without_declared_oracle"
	return True, None


def _check_teacher_frozen(row: dict, _path: Path, ctx: dict) -> tuple[bool, str | None]:
	freeze = ctx.get("freeze")
	if not isinstance(freeze, dict) or not freeze.get("code_commit"):
		return False, "missing_freeze"
	commit = ctx.get("provenance_commit")
	if not commit:
		return False, "missing_provenance_commit"
	if commit != freeze.get("code_commit"):
		return False, "commit_mismatch"
	f_det = freeze.get("detector")
	ep_det = row.get("detector")
	if f_det and ep_det and str(ep_det) != str(f_det):
		return False, "detector_mismatch"
	return True, None


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
		declared_oracle = bool(exp.get("oracle"))
		oracle_events = [e for e in events if e.get("kind") == "oracle_applied"]
		is_oracle = declared_oracle or bool(oracle_events)
		row["oracle"] = is_oracle
		seed = seed_of_path_or_case(ep_dir, task, row)
		case_list = task.get("case")
		case_opts = (
			case_list[3]
			if isinstance(case_list, list) and len(case_list) >= 4 and isinstance(case_list[3], dict)
			else None
		)
		split = "test" if is_sealed(seed, batch=row.get("batch"), options=case_opts) else case_split(seed, case_opts)
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
				"case_tuple": list(case_list) if isinstance(case_list, list) else None,
				"case_options": case_opts,
				"split": split,
				"pair_key": pk,
				"attempt_key": attempt_key,
				"rank": rank,
				"declared_oracle": declared_oracle,
				"oracle_events": oracle_events,
				"is_oracle": is_oracle,
				"provenance_commit": exp.get("code_commit") or (prov.get("git") or {}).get("commit"),
				"suite": exp.get("suite"),
				"suite_sha256": exp.get("suite_sha256"),
				"segmenter": exp.get("segmenter") or "lab",
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
			"declared_oracle": item["declared_oracle"],
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
			"case_tuple": item["case_tuple"],
			"case_options": item["case_options"],
			"split": item["split"],
			"sha_episode": item["sha_episode"],
			"provenance_commit": item["provenance_commit"],
			"suite": item["suite"],
			"suite_sha256": item["suite_sha256"],
			"segmenter": item["segmenter"],
			"declared_oracle": item["declared_oracle"],
			"oracle_events_count": len(item["oracle_events"]),
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


def build_pairs(
	con: sqlite3.Connection,
	build_id: str,
	*,
	allow_cross_batch: bool = True,
	allow_suite_subset: bool = False,
) -> int:
	"""Populate ``silver_pair`` for ``build_id`` with one row per latest realistic episode (preserving provenance across batches)
	plus one row per unmatched latest truth episode; returns pair-row count."""
	con.execute("delete from silver_pair where build_id = ?", (build_id,))
	rows = [
		json.loads(r[0])
		for r in con.execute("select row_json from silver_episode where build_id = ? order by path", (build_id,))
	]

	truth_by_pk: dict[str, list[dict]] = {}
	all_truth: list[dict] = []
	all_real: list[dict] = []
	for r in rows:
		if r.get("superseded") or not r.get("pair_key"):
			continue
		pk_json = json.dumps(r["pair_key"])
		if r.get("sensing") == "truth":
			truth_by_pk.setdefault(pk_json, []).append(r)
			all_truth.append(r)
		elif r.get("sensing") == "realistic":
			all_real.append(r)

	def _rank(ep: dict) -> tuple[int, str]:
		return (int(ep.get("attempt") or 0), str(ep["path"]))

	paired_truth_paths: set[str] = set()
	inserts: list[tuple] = []
	for r_row in sorted(all_real, key=lambda x: str(x["path"])):
		pk_json = json.dumps(r_row["pair_key"])
		cam = str(r_row.get("camera_condition") or "fixed")
		cands = truth_by_pk.get(pk_json, [])

		same_cands: list[tuple[dict, str]] = []
		cross_cands: list[tuple[dict, str]] = []
		reject_reasons: set[str] = set()
		for t_row in cands:
			ok, reason = truth_compatible(
				t_row,
				r_row,
				allow_cross_batch=allow_cross_batch,
				allow_suite_subset=allow_suite_subset,
			)
			if ok:
				if reason == "same_batch":
					same_cands.append((t_row, reason))
				else:
					cross_cands.append((t_row, reason))
			else:
				reject_reasons.add(reason)

		chosen: tuple[dict, str] | None = None
		if same_cands:
			chosen = max(same_cands, key=lambda item: _rank(item[0]))
		elif cross_cands:
			chosen = max(cross_cands, key=lambda item: _rank(item[0]))

		if chosen is not None:
			t_row, scope = chosen
			paired_truth_paths.add(str(t_row["path"]))
			outcome = pair_outcome(t_row, r_row)
			inserts.append(
				(
					build_id,
					str(r_row["path"]),
					r_row.get("batch"),
					pk_json,
					cam,
					str(t_row["path"]),
					t_row.get("batch"),
					str(r_row["path"]),
					outcome,
					scope,
					None,
				)
			)
		else:
			detail = ",".join(sorted(reject_reasons)) if reject_reasons else "no_truth"
			inserts.append(
				(
					build_id,
					str(r_row["path"]),
					r_row.get("batch"),
					pk_json,
					cam,
					None,
					None,
					str(r_row["path"]),
					"unpaired",
					"unpaired",
					detail,
				)
			)

	for t_row in sorted(all_truth, key=lambda x: str(x["path"])):
		if str(t_row["path"]) not in paired_truth_paths:
			pk_json = json.dumps(t_row["pair_key"])
			inserts.append(
				(
					build_id,
					str(t_row["path"]),
					t_row.get("batch"),
					pk_json,
					"fixed",
					str(t_row["path"]),
					t_row.get("batch"),
					None,
					"unpaired",
					"unpaired",
					"no_realistic",
				)
			)

	con.executemany(
		"insert into silver_pair "
		"(build_id, subject_path, batch, pair_key, camera_condition, truth_path, truth_batch, realistic_path, outcome, match_scope, detail) "
		"values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
		inserts,
	)
	con.commit()
	return len(inserts)


def reconcile_silver(con: sqlite3.Connection, build_id: str) -> dict:
	"""Account for every row in ``silver_episode`` against ``silver_pair`` and compute per-batch pairing denominators."""
	rows = [
		json.loads(r[0])
		for r in con.execute("select row_json from silver_episode where build_id = ? order by path", (build_id,))
	]
	pairs = con.execute(
		"select subject_path, batch, truth_path, truth_batch, realistic_path, outcome, match_scope, detail "
		"from silver_pair where build_id = ?",
		(build_id,),
	).fetchall()

	referenced: set[str] = set()
	by_scope: dict[str, int] = {}
	by_outcome: dict[str, int] = {}
	unpaired_reasons: dict[str, int] = {}
	batch_stats: dict[str, dict[str, int | float]] = {}

	for _subj, batch, t_path, _t_batch, r_path, outcome, scope, detail in pairs:
		if t_path:
			referenced.add(t_path)
		if r_path:
			referenced.add(r_path)
		by_scope[scope] = by_scope.get(scope, 0) + 1
		by_outcome[outcome] = by_outcome.get(outcome, 0) + 1
		if outcome == "unpaired" and detail:
			unpaired_reasons[detail] = unpaired_reasons.get(detail, 0) + 1
		bkey = str(batch or "-")
		bs = batch_stats.setdefault(
			bkey, {"valid_paired": 0, "realistic_only": 0, "void": 0, "unpaired": 0, "realistic_only_rate": 1.0}
		)
		if outcome == "void":
			bs["void"] += 1
		elif outcome == "unpaired":
			bs["unpaired"] += 1
		else:
			bs["valid_paired"] += 1
			if outcome == "realistic_only":
				bs["realistic_only"] += 1

	for bs in batch_stats.values():
		denom = int(bs["valid_paired"])
		bs["realistic_only_rate"] = round(int(bs["realistic_only"]) / denom, 6) if denom > 0 else 1.0

	superseded = sum(1 for r in rows if r.get("superseded"))
	missing_pk = sum(
		1
		for r in rows
		if not r.get("superseded") and (not r.get("pair_key") or r.get("sensing") not in ("truth", "realistic"))
	)
	eligible = [
		r for r in rows if not r.get("superseded") and r.get("pair_key") and r.get("sensing") in ("truth", "realistic")
	]
	unrepresented = [r["path"] for r in eligible if r["path"] not in referenced]

	return {
		"build_id": build_id,
		"total_silver_episodes": len(rows),
		"superseded": superseded,
		"excluded_no_pair_key_or_sensing": missing_pk,
		"eligible_latest": len(eligible),
		"represented_in_pairs": len(eligible) - len(unrepresented),
		"unrepresented_latest": len(unrepresented),
		"pairs_total": len(pairs),
		"pairs_by_scope": by_scope,
		"pairs_by_outcome": by_outcome,
		"unpaired_reasons": unpaired_reasons,
		"batch_rates": batch_stats,
	}


def pair_report(con: sqlite3.Connection, build_id: str) -> str:
	"""Render Markdown report of silver pair outcomes by batch x camera, match scopes, reconciliation, realistic_only, truth_only cross-tab and quarantine."""
	ep_rows = {
		r[0]: json.loads(r[1])
		for r in con.execute("select path, row_json from silver_episode where build_id = ?", (build_id,))
	}
	pairs = con.execute(
		"select batch, pair_key, camera_condition, truth_path, realistic_path, outcome, match_scope "
		"from silver_pair where build_id = ? order by batch, camera_condition, subject_path",
		(build_id,),
	).fetchall()

	by_batch_cam: dict[tuple[str, str], dict[str, int]] = {}
	scope_counts: dict[str, int] = {}
	realistic_only_items: list[dict] = []
	truth_only_cross: dict[tuple[str, str], int] = {}
	for batch_col, _pk, cam, t_path, r_path, outcome, scope in pairs:
		r_ep = ep_rows.get(r_path) if r_path else None
		t_ep = ep_rows.get(t_path) if t_path else None
		ref = r_ep or t_ep or {}
		batch = str(batch_col or ref.get("batch") or "-")
		bucket = by_batch_cam.setdefault((batch, cam), {o: 0 for o in OUTCOMES})
		bucket[outcome] = bucket.get(outcome, 0) + 1
		scope_counts[scope] = scope_counts.get(scope, 0) + 1
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

	rec = reconcile_silver(con, build_id)
	quarantine_counts: dict[str, int] = {}
	for ep in ep_rows.values():
		for reason in ep.get("quarantine_reasons") or []:
			quarantine_counts[reason] = quarantine_counts.get(reason, 0) + 1

	lines = [
		f"# Silver Pair Report (`{build_id}`)",
		"",
		"## Reconciliation",
		"",
		f"- total_silver_episodes: {rec['total_silver_episodes']}",
		f"- eligible_latest: {rec['eligible_latest']} (represented={rec['represented_in_pairs']}, unrepresented={rec['unrepresented_latest']})",
		f"- superseded: {rec['superseded']}, excluded_no_pair_key_or_sensing: {rec['excluded_no_pair_key_or_sensing']}",
		f"- pairs_by_scope: {json.dumps(scope_counts, sort_keys=True)}",
		"",
		"## Batch x Camera Outcomes",
		"",
	]
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
