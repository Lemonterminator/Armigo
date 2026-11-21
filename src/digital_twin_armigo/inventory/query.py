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
	"trajectories": ("category", "engine", "verdict", "run_group", "category_basis", "control", "sensing", "root", "split_hint"),
	"scenes": ("scene_kind", "engine", "split", "dataset_id"),
	"datasets": ("kind", "engine", "root"),
}  # fmt: skip
SEARCH = {
	"trajectories": ("traj_id", "path", "run_group", "scene_id", "tags"),
	"scenes": ("scene_id", "dataset_id", "defect_kind"),
	"datasets": ("dataset_id", "name", "path"),
}
FACETS = {
	"trajectories": ("category", "engine", "verdict", "run_group", "category_basis", "root"),
	"scenes": ("scene_kind", "engine", "split"),
	"datasets": ("kind", "engine"),
}
BOOL_FILTERS = {"trajectories": ("has_motion", "has_video", "has_actions", "has_events", "defective", "llm_consulted")}
MAX_LIMIT = 500


def db_path(inventory_dir: Path) -> Path:
	return Path(inventory_dir) / "inventory.sqlite"


@contextlib.contextmanager
def connect(inventory_dir: Path):
	"""Read-only connection, closed on exit (an open handle would block the next scan from replacing the file on Windows)."""
	db = db_path(inventory_dir)
	if not db.is_file():
		raise FileNotFoundError(str(db))
	con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
	con.row_factory = sqlite3.Row
	try:
		yield con
	finally:
		con.close()


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


def _where(table: str, params: dict[str, str]) -> tuple[str, list]:
	"""WHERE clause for the query-string ``params`` (unknown keys are ignored, so nothing user-supplied becomes SQL)."""
	clauses: list[str] = []
	args: list = []
	for col in FILTERS[table]:
		raw = params.get(col)
		if raw:
			values = [v for v in raw.split(",") if v]
			clauses.append(f"{col} in ({','.join('?' * len(values))})")
			args += values
	for col in BOOL_FILTERS.get(table, ()):
		raw = params.get(col)
		if raw in ("0", "1"):
			clauses.append(f"coalesce({col}, 0) = ?")
			args.append(int(raw))
	if table == "trajectories" and params.get("distinct", "1") == "1":
		clauses.append("same_as is null")
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


def rows(inventory_dir: Path, table: str, params: dict[str, str]) -> dict:
	"""One page of ``table`` plus the total matching count. ``sort`` is ``column`` or ``-column`` (descending)."""
	if table not in TABLES:
		raise KeyError(table)
	columns, default_sort = TABLES[table]
	where, args = _where(table, params)
	sort = params.get("sort") or ""
	order = default_sort
	if sort.lstrip("-") in columns:
		order = f"{sort.lstrip('-')} {'desc' if sort.startswith('-') else 'asc'}"
	limit = _int(params.get("limit"), 100, 1, MAX_LIMIT)
	offset = _int(params.get("offset"), 0, 0, 10**9)
	with connect(inventory_dir) as con:
		total = con.execute(f"select count(*) from {table}{where}", args).fetchone()[0]
		found = con.execute(
			f"select * from {table}{where} order by {order}, rowid limit ? offset ?", [*args, limit, offset]
		).fetchall()
	return {
		"table": table,
		"total": total,
		"limit": limit,
		"offset": offset,
		"rows": [_decode(r) for r in found],
	}


def facets(inventory_dir: Path, table: str, params: dict[str, str]) -> dict[str, list[dict]]:
	"""Value counts per facet column, each computed with every filter except its own (so chips show what a click yields)."""
	if table not in TABLES:
		raise KeyError(table)
	out: dict[str, list[dict]] = {}
	with connect(inventory_dir) as con:
		for col in FACETS[table]:
			where, args = _where(table, {k: v for k, v in params.items() if k != col})
			found = con.execute(
				f"select {col} v, count(*) n from {table}{where} group by 1 order by n desc, v", args
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
	return {
		"meta": meta,
		"categories": taxonomy.CATEGORIES,
		"matrix": matrix,
		"totals": totals,
		"duplicates": duplicates,
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


def trajectory(inventory_dir: Path, traj_id: str) -> dict | None:
	with connect(inventory_dir) as con:
		row = con.execute("select * from trajectories where traj_id = ?", (traj_id,)).fetchone()
	return _decode(row) if row else None
