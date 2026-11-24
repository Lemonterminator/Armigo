"""Read-only queries over ``inventory.sqlite`` for the HTML front end (filters, sorting, paging, facet counts).

Every column name that reaches SQL comes from the fixed column tuples in ``store``; values are always bound parameters.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
from pathlib import Path

from digital_twin_armigo.inventory import store, taxonomy

TABLES = {
	"trajectories": (store.TRAJECTORY_COLUMNS, "written_at desc"),
	"scenes": (store.SCENE_COLUMNS, "n_trajectories desc"),
	"datasets": (store.DATASET_COLUMNS, "n_items desc"),
}
# columns a user may filter on with an exact match, per table (a comma list means "any of")
FILTERS = {
	"trajectories": ("category", "engine", "verdict", "verdict_final_cycle", "failure_mode", "failure_mode_final_cycle",
		"run_group", "category_basis", "control", "sensing", "root", "split_hint",
		"batch", "group", "camera_condition", "strategy", "detector", "layout_id", "case_id", "run_id", "run_state"),
	"scenes": ("scene_kind", "engine", "split", "dataset_id"),
	"datasets": ("kind", "engine", "root"),
}  # fmt: skip
SEARCH = {
	"trajectories": ("traj_id", "path", "run_group", "scene_id", "tags", "case_id", "layout_id", "batch"),
	"scenes": ("scene_id", "dataset_id", "defect_kind"),
	"datasets": ("dataset_id", "name", "path"),
}
FACETS = {
	"trajectories": (
		"category",
		"engine",
		"verdict",
		"verdict_final_cycle",
		"failure_mode",
		"run_group",
		"category_basis",
		"root",
		"sensing",
		"batch",
		"group",
		"camera_condition",
		"strategy",
		"detector",
		"run_state",
		"audit_state",
		"root_cause",
	),
	"scenes": ("scene_kind", "engine", "split"),
	"datasets": ("kind", "engine"),
}
# filters answered from the audit database (latest review per trajectory), never stored in inventory.sqlite
AUDIT_FILTERS = ("audit_state", "root_cause")
BOOL_FILTERS = {"trajectories": ("has_motion", "has_video", "has_actions", "has_events", "defective", "llm_consulted")}
MAX_LIMIT = 500


def db_path(inventory_dir: Path) -> Path:
	return Path(inventory_dir) / "inventory.sqlite"


class _Connection(sqlite3.Connection):
	has_audit = False


@contextlib.contextmanager
def connect(inventory_dir: Path, audit: Path | None = None):
	"""Read-only connection, closed on exit (an open handle would block the next scan from replacing the file on Windows).

	With an ``audit`` database that exists, it is attached as ``audit`` (read-only) so the latest review of a trajectory can
	be joined at request time; ``con.has_audit`` tells whether that worked."""
	db = db_path(inventory_dir)
	if not db.is_file():
		raise FileNotFoundError(str(db))
	con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True, factory=_Connection)
	con.row_factory = sqlite3.Row
	if audit is not None and Path(audit).is_file():
		try:
			con.execute("attach database ? as audit", (f"file:{Path(audit).as_posix()}?mode=ro",))
			con.has_audit = bool(con.execute("select 1 from audit.sqlite_master where name = 'reviews'").fetchone())
		except sqlite3.Error:
			con.has_audit = False
	try:
		yield con
	finally:
		con.close()


def _latest(con: _Connection, column: str) -> str:
	"""SQL expression: ``column`` of the latest review of the current trajectory row (``state`` reads as unreviewed)."""
	if con.has_audit:
		sub = f"(select r.{column} from audit.reviews r where r.traj_id = trajectories.traj_id order by r.id desc limit 1)"
	else:
		sub = "null"
	return f"coalesce({sub}, 'unreviewed')" if column == "state" else sub


def _exprs(con: _Connection) -> dict[str, str]:
	return {"audit_state": _latest(con, "state"), "root_cause": _latest(con, "root_cause")}


def _decode(row: sqlite3.Row, json_columns=store.JSON_COLUMNS) -> dict:
	out = {}
	for key, value in dict(row).items():  # (iterating a sqlite3.Row yields values, not keys)
		if key in json_columns and isinstance(value, str):
			try:
				value = json.loads(value)
			except ValueError:
				pass
		out[key] = value
	return out


def _ident(col: str, exprs: dict[str, str] | None = None) -> str:
	"""SQL for a column from the fixed column tuples (quoted: ``group`` is a keyword) or its audit expression."""
	return (exprs or {}).get(col) or f'"{col}"'


def _where(table: str, params: dict[str, str], exprs: dict[str, str] | None = None) -> tuple[str, list]:
	"""WHERE clause for the query-string ``params`` (unknown keys are ignored, so nothing user-supplied becomes SQL).
	``exprs`` maps the audit filters to their SQL expressions (see ``_exprs``)."""
	clauses: list[str] = []
	args: list = []
	cols = FILTERS[table] + (AUDIT_FILTERS if table == "trajectories" else ())
	for col in cols:
		raw = params.get(col)
		if raw:
			values = [v for v in raw.split(",") if v]
			clauses.append(f"{_ident(col, exprs)} in ({','.join('?' * len(values))})")
			args += values
	for col in BOOL_FILTERS.get(table, ()):
		raw = params.get(col)
		if raw in ("0", "1"):
			clauses.append(f"coalesce({col}, 0) = ?")
			args.append(int(raw))
	if table == "trajectories":
		if params.get("distinct", "1") == "1":
			clauses.append("same_as is null")
		if params.get("latest") == "1":  # hide originals that a re-run superseded; the history stays in the table
			clauses.append("superseded_by is null")
		if params.get("is_rerun") in ("0", "1"):
			clauses.append("rerun_of is not null" if params["is_rerun"] == "1" else "rerun_of is null")
	text = (params.get("q") or "").strip()
	if text:
		clauses.append("(" + " or ".join(f"{c} like ? escape '\\'" for c in SEARCH[table]) + ")")
		like = "%" + text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
		args += [like] * len(SEARCH[table])
	return (" where " + " and ".join(clauses)) if clauses else "", args


def _int(value, default: int, lo: int, hi: int) -> int:
	try:
		return max(lo, min(hi, int(value)))
	except (TypeError, ValueError):
		return default


def rows(inventory_dir: Path, table: str, params: dict[str, str], audit: Path | None = None) -> dict:
	"""One page of ``table`` plus the total matching count. ``sort`` is ``column`` or ``-column`` (descending)."""
	if table not in TABLES:
		raise KeyError(table)
	columns, default_sort = TABLES[table]
	sort = params.get("sort") or ""
	order = default_sort
	if sort.lstrip("-") in columns:
		order = f"{_ident(sort.lstrip('-'))} {'desc' if sort.startswith('-') else 'asc'}"
	limit = _int(params.get("limit"), 100, 1, MAX_LIMIT)
	offset = _int(params.get("offset"), 0, 0, 10**9)
	with connect(inventory_dir, audit) as con:
		exprs = _exprs(con)
		where, args = _where(table, params, exprs)
		extra = (
			f", {exprs['audit_state']} as audit_state, {exprs['root_cause']} as root_cause"
			if table == "trajectories"
			else ""
		)
		total = con.execute(f"select count(*) from {table}{where}", args).fetchone()[0]
		found = con.execute(
			f"select {table}.*{extra} from {table}{where} order by {order}, rowid limit ? offset ?",
			[*args, limit, offset],
		).fetchall()
	return {
		"table": table,
		"total": total,
		"limit": limit,
		"offset": offset,
		"rows": [_decode(r) for r in found],
	}


def facets(inventory_dir: Path, table: str, params: dict[str, str], audit: Path | None = None) -> dict[str, list[dict]]:
	"""Value counts per facet column, each computed with every filter except its own (so chips show what a click yields)."""
	if table not in TABLES:
		raise KeyError(table)
	out: dict[str, list[dict]] = {}
	with connect(inventory_dir, audit) as con:
		exprs = _exprs(con)
		present = {r["name"] for r in con.execute(f"pragma table_info({table})")}
		for col in FACETS[table]:
			if col not in present and col not in exprs:  # an inventory scanned before the column existed
				continue
			where, args = _where(table, {k: v for k, v in params.items() if k != col}, exprs)
			found = con.execute(
				f"select {_ident(col, exprs)} v, count(*) n from {table}{where} group by 1 order by n desc, v", args
			).fetchall()
			out[col] = [{"value": r["v"], "count": r["n"]} for r in found if r["v"] is not None]
	return out


def summary(inventory_dir: Path) -> dict:
	"""Scan metadata, category labels, the category × engine matrix (content-distinct trajectories) and table sizes."""
	with connect(inventory_dir) as con:
		meta = {r["key"]: _loads(r["value"]) for r in con.execute("select key, value from meta")}
		matrix = [
			dict(r)
			for r in con.execute(
				"""select category, engine, count(*) n, sum(verdict = 'SUCCESS') success,
				   sum(coalesce(has_motion, 0)) with_motion
				   from trajectories where same_as is null group by 1, 2 order by 1, 2"""
			)
		]
		totals = {
			t: con.execute(f"select count(*) from {t}").fetchone()[0] for t in ("trajectories", "scenes", "datasets")
		}
		duplicates = con.execute("select count(*) from trajectories where same_as is not null").fetchone()[0]
		reruns = dict(
			zip(
				("reruns", "superseded", "rerun_success"),
				con.execute(
					"""select count(rerun_of), count(superseded_by), sum(rerun_of is not null and verdict = 'SUCCESS')
					   from trajectories"""
				).fetchone(),
				strict=True,
			)
		)
	return {
		"meta": meta,
		"categories": taxonomy.CATEGORIES,
		"matrix": matrix,
		"totals": totals,
		"duplicates": duplicates,
		"reruns": {k: v or 0 for k, v in reruns.items()},
	}


def _loads(value):
	try:
		return json.loads(value)
	except (TypeError, ValueError):
		return value


def agent_episode_path(inventory_dir: Path, traj_id: str) -> Path | None:
	"""Directory of an agent episode the inventory knows by ``traj_id`` (never an arbitrary path from the client)."""
	with connect(inventory_dir) as con:
		row = con.execute(
			"select path from trajectories where traj_id = ? and record_kind = 'agent_episode'", (traj_id,)
		).fetchone()
	return Path(row["path"]) if row else None


def trajectory(inventory_dir: Path, traj_id: str, audit: Path | None = None) -> dict | None:
	with connect(inventory_dir, audit) as con:
		exprs = _exprs(con)
		row = con.execute(
			f"select trajectories.*, {exprs['audit_state']} as audit_state, {exprs['root_cause']} as root_cause "
			"from trajectories where traj_id = ?",
			(traj_id,),
		).fetchone()
	return _decode(row) if row else None


PAIR_COLUMNS = ("engine", "strategy", "camera_condition", "detector")


def paired(inventory_dir: Path, traj_id: str, audit: Path | None = None) -> dict | None:
	"""Rows paired with ``traj_id``: ``layout`` = the other brick of the same layout in the same batch, engine and group;
	``case`` = the same case in the same batch under another engine / strategy / camera condition / detector."""
	with connect(inventory_dir, audit) as con:
		me = con.execute("select * from trajectories where traj_id = ?", (traj_id,)).fetchone()
		if me is None:
			return None
		exprs = _exprs(con)
		select = f"select trajectories.*, {exprs['audit_state']} as audit_state, {exprs['root_cause']} as root_cause from trajectories"
		layout: list = []
		case: list = []
		cols = set(me.keys())
		if "layout_id" in cols and me["layout_id"] is not None and me["batch"] is not None:
			layout = con.execute(
				f'{select} where batch = ? and engine is ? and "group" is ? and layout_id = ? and traj_id != ? '
				"and case_id is not ? order by case_id, attempt",
				(me["batch"], me["engine"], me["group"], me["layout_id"], traj_id, me["case_id"]),
			).fetchall()
		if "case_id" in cols and me["case_id"] is not None and me["batch"] is not None:
			same = " and ".join(f"{c} is ?" for c in PAIR_COLUMNS)
			case = con.execute(
				f"{select} where batch = ? and case_id = ? and traj_id != ? and not ({same}) "
				"order by engine, strategy, camera_condition, detector, attempt",
				(me["batch"], me["case_id"], traj_id, *[me[c] for c in PAIR_COLUMNS]),
			).fetchall()
	return {"layout": [_decode(r) for r in layout], "case": [_decode(r) for r in case]}
