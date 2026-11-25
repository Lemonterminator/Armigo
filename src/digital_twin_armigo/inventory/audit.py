"""Human audit verdicts on trajectories: an append-only SQLite log that no inventory scan ever rebuilds.

One row per review. The current verdict of a trajectory is its latest row (highest ``id``); a correction is a new row that
``replaces`` the one it was based on, so the history stays. Writers send ``expected_latest_id`` (``None`` for the first
review of a trajectory); a mismatch means somebody else reviewed it meanwhile and raises :class:`Conflict` carrying the
current latest row.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import json
import sqlite3
from pathlib import Path

from digital_twin_armigo.experiments.contract import AUDIT_STATES, PHYSICAL_OUTCOMES, ROOT_CAUSES
from digital_twin_armigo.experiments.precursors import MECHANISMS

SCHEMA = """
create table if not exists reviews (
	id integer primary key autoincrement,
	traj_id text not null,
	created_at text not null,
	reviewer text not null,
	state text not null,
	root_cause text,
	mechanism text,
	physical_outcome text,
	note text,
	refs text not null default '[]',
	code_version text,
	replaces integer
);
create index if not exists reviews_traj on reviews (traj_id, id);
"""
_CANONICAL_COLUMNS: tuple[str, ...] = (
	"id",
	"traj_id",
	"created_at",
	"reviewer",
	"state",
	"root_cause",
	"mechanism",
	"physical_outcome",
	"note",
	"refs",
	"code_version",
	"replaces",
)
MAX_NOTE = 8000
MAX_REFS = 50


class ValidationError(ValueError):
	"""The review body is malformed (an enum outside its list, a missing reviewer, a bad frame reference)."""


class Conflict(Exception):
	"""``expected_latest_id`` is not the latest review id of the trajectory; ``current`` is that latest row (or None)."""

	def __init__(self, current: dict | None):
		super().__init__("the trajectory was reviewed since this form was loaded")
		self.current = current


def default_path(inventory_dir: Path) -> Path:
	"""``<inventory dir>/../audit/reviews.sqlite``: next to, not inside, the directory a scan rebuilds."""
	return Path(inventory_dir).resolve().parent / "audit" / "reviews.sqlite"


@contextlib.contextmanager
def connect(path: Path):
	"""Connection in WAL mode with the schema ensured; autocommit so that transactions are explicit."""
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	con = sqlite3.connect(path, timeout=15, isolation_level=None)
	con.row_factory = sqlite3.Row
	try:
		con.execute("pragma journal_mode = wal")
		con.executescript(SCHEMA)
		try:
			con.execute("alter table reviews add column mechanism text")
		except sqlite3.OperationalError:
			pass
		yield con
	finally:
		con.close()


def _row(r: sqlite3.Row | None) -> dict | None:
	if r is None:
		return None
	out = dict(r)
	try:
		out["refs"] = json.loads(out["refs"])
	except (TypeError, ValueError):
		out["refs"] = []
	return out


def _enum(value, allowed: tuple[str, ...], name: str, *, nullable: bool) -> str | None:
	if value in (None, ""):
		if nullable:
			return None
		raise ValidationError(f"{name} is required")
	if value not in allowed:
		raise ValidationError(f"{name} {value!r} is not one of {', '.join(allowed)}")
	return value


def _refs(value) -> list[dict]:
	if value in (None, ""):
		return []
	if not isinstance(value, list) or len(value) > MAX_REFS:
		raise ValidationError(f"refs must be a list of at most {MAX_REFS} {{branch, frame, camera?, view_id?}}")
	out = []
	for ref in value:
		if not isinstance(ref, dict) or "branch" not in ref or "frame" not in ref:
			raise ValidationError("every ref needs a branch and a frame")
		try:
			item = {"branch": int(ref["branch"]), "frame": int(ref["frame"])}
		except (TypeError, ValueError):
			raise ValidationError("ref branch and frame must be integers") from None
		for key in ("camera", "view_id"):
			if ref.get(key) not in (None, ""):
				item[key] = str(ref[key])
		out.append(item)
	return out


def validate(body: dict) -> dict:
	"""The cleaned review fields of ``body`` (without ``traj_id`` / ``expected_latest_id``), or ValidationError."""
	if not isinstance(body, dict):
		raise ValidationError("body must be an object")
	reviewer = str(body.get("reviewer") or "").strip()
	if not reviewer or len(reviewer) > 80:
		raise ValidationError("reviewer is required (at most 80 characters)")
	note = body.get("note")
	if note is not None and (not isinstance(note, str) or len(note) > MAX_NOTE):
		raise ValidationError(f"note must be text of at most {MAX_NOTE} characters")
	return {
		"reviewer": reviewer,
		"state": _enum(body.get("state"), AUDIT_STATES, "state", nullable=False),
		"root_cause": _enum(body.get("root_cause"), ROOT_CAUSES, "root_cause", nullable=True),
		"mechanism": _enum(body.get("mechanism"), MECHANISMS, "mechanism", nullable=True),
		"physical_outcome": _enum(body.get("physical_outcome"), PHYSICAL_OUTCOMES, "physical_outcome", nullable=True),
		"note": note or None,
		"refs": _refs(body.get("refs")),
	}


@contextlib.contextmanager
def connect_readonly(path: Path):
	"""Read-only SQLite connection that never mutates the file, switches journal mode, or alters schema."""
	resolved = Path(path).resolve()
	con = sqlite3.connect(f"file:{resolved.as_posix()}?mode=ro", uri=True, timeout=15)
	con.row_factory = sqlite3.Row
	try:
		yield con
	finally:
		con.close()


def _select_columns(con: sqlite3.Connection) -> str | None:
	cols = {r[1] for r in con.execute("pragma table_info(reviews)").fetchall()}
	if not cols:
		return None
	return ", ".join(c if c in cols else f"null as {c}" for c in _CANONICAL_COLUMNS)


def history(path: Path, traj_id: str) -> list[dict]:
	"""All reviews of ``traj_id``, newest first. A missing database reads as no reviews."""
	if not Path(path).is_file():
		return []
	with connect_readonly(path) as con:
		sel = _select_columns(con)
		if sel is None:
			return []
		found = con.execute(f"select {sel} from reviews where traj_id = ? order by id desc", (traj_id,)).fetchall()
	return [_row(r) for r in found]


def latest(path: Path, traj_id: str) -> dict | None:
	if not Path(path).is_file():
		return None
	with connect_readonly(path) as con:
		sel = _select_columns(con)
		if sel is None:
			return None
		found = con.execute(
			f"select {sel} from reviews where traj_id = ? order by id desc limit 1", (traj_id,)
		).fetchone()
	return _row(found)


def latest_all(path: Path) -> dict[str, dict]:
	"""Latest review per ``traj_id`` read in one read-only pass (handles legacy schemas without ``mechanism``)."""
	if not Path(path).is_file():
		return {}
	with connect_readonly(path) as con:
		sel = _select_columns(con)
		if sel is None:
			return {}
		rows = con.execute(
			f"select {sel} from reviews where id in (select max(id) from reviews group by traj_id) order by traj_id"
		).fetchall()
	return {r["traj_id"]: _row(r) for r in rows}


def append(path: Path, traj_id: str, body: dict, *, expected_latest_id: int | None, code_version: str | None) -> dict:
	"""Add a review. ``expected_latest_id`` must be the id of the trajectory's current latest review (None if it has
	none), else :class:`Conflict`; the new row ``replaces`` that one."""
	if not isinstance(traj_id, str) or not traj_id.strip():
		raise ValidationError("traj_id is required")
	if expected_latest_id is not None and (
		not isinstance(expected_latest_id, int) or isinstance(expected_latest_id, bool)
	):
		raise ValidationError("expected_latest_id must be an integer or null")
	fields = validate(body)
	with connect(path) as con:
		con.execute("begin immediate")
		try:
			cur = con.execute("select * from reviews where traj_id = ? order by id desc limit 1", (traj_id,)).fetchone()
			if (cur["id"] if cur else None) != expected_latest_id:
				raise Conflict(_row(cur))
			new_id = con.execute(
				"insert into reviews (traj_id, created_at, reviewer, state, root_cause, mechanism, physical_outcome, note, refs, "
				"code_version, replaces) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
				(
					traj_id,
					dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
					fields["reviewer"],
					fields["state"],
					fields["root_cause"],
					fields["mechanism"],
					fields["physical_outcome"],
					fields["note"],
					json.dumps(fields["refs"], ensure_ascii=False),
					code_version,
					expected_latest_id,
				),
			).lastrowid
			row = _row(con.execute("select * from reviews where id = ?", (new_id,)).fetchone())
			con.execute("commit")
		except BaseException:
			con.execute("rollback")
			raise
	return row


def summary(path: Path) -> dict:
	"""Counts of the *latest* review per trajectory, by state and by root cause."""
	out = {"reviewed": 0, "states": {s: 0 for s in AUDIT_STATES}, "root_causes": {}, "physical_outcomes": {}}
	if not Path(path).is_file():
		return out
	with connect_readonly(path) as con:
		if _select_columns(con) is None:
			return out
		latest_rows = con.execute(
			"select state, root_cause, physical_outcome from reviews "
			"where id in (select max(id) from reviews group by traj_id)"
		).fetchall()
	for r in latest_rows:
		out["reviewed"] += 1
		out["states"][r["state"]] = out["states"].get(r["state"], 0) + 1
		if r["root_cause"]:
			out["root_causes"][r["root_cause"]] = out["root_causes"].get(r["root_cause"], 0) + 1
		if r["physical_outcome"]:
			out["physical_outcomes"][r["physical_outcome"]] = out["physical_outcomes"].get(r["physical_outcome"], 0) + 1
	return out


def enums() -> dict:
	return {
		"states": list(AUDIT_STATES),
		"root_causes": list(ROOT_CAUSES),
		"mechanism": list(MECHANISMS),
		"physical_outcomes": list(PHYSICAL_OUTCOMES),
	}
