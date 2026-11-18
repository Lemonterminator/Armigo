"""Append-only operation ledger: idempotent jobs, fenced leases, stop latch, and budget accounting in SQLite."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Self

from digital_twin_armigo.runtime.canonical import canonical_json, sha256_json
from digital_twin_armigo.runtime.errors import (
	IdempotencyConflictError,
	InvalidTransitionError,
	JobNotFoundError,
	LedgerError,
	StaleWorkerError,
	StopLatchedError,
)

SCHEMA_VERSION = 1
_BUSY_TIMEOUT_MS = 30000


class JobState(StrEnum):
	QUEUED = "QUEUED"
	RUNNING = "RUNNING"
	CANCEL_REQUESTED = "CANCEL_REQUESTED"
	SUCCEEDED = "SUCCEEDED"
	FAILED = "FAILED"
	CANCELLED = "CANCELLED"
	NEEDS_RECOVERY = "NEEDS_RECOVERY"


TERMINAL_STATES = frozenset({JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED})

_S = JobState
# Transitions only a lease-holding worker may make.
_WORKER_TRANSITIONS = frozenset(
	{
		(_S.QUEUED, _S.RUNNING),
		(_S.NEEDS_RECOVERY, _S.RUNNING),
		(_S.RUNNING, _S.SUCCEEDED),
		(_S.RUNNING, _S.FAILED),
		(_S.RUNNING, _S.CANCELLED),
		(_S.CANCEL_REQUESTED, _S.CANCELLED),
		(_S.CANCEL_REQUESTED, _S.SUCCEEDED),
		(_S.CANCEL_REQUESTED, _S.FAILED),
	}
)
# Transitions only the coordinator (no lease) may make.
_COORDINATOR_TRANSITIONS = frozenset(
	{
		(_S.QUEUED, _S.CANCELLED),
		(_S.RUNNING, _S.CANCEL_REQUESTED),
		(_S.RUNNING, _S.NEEDS_RECOVERY),
		(_S.CANCEL_REQUESTED, _S.NEEDS_RECOVERY),
		(_S.NEEDS_RECOVERY, _S.SUCCEEDED),
		(_S.NEEDS_RECOVERY, _S.FAILED),
	}
)
_START_STATES = frozenset({_S.RUNNING})

_TABLES = ("operations", "job_events", "lease_events", "stop_events")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS operations (
	job_id TEXT PRIMARY KEY,
	idempotency_key TEXT UNIQUE NOT NULL,
	op_type TEXT NOT NULL,
	params_json TEXT NOT NULL,
	params_sha256 TEXT NOT NULL,
	run_id TEXT NOT NULL,
	branch_id TEXT NOT NULL,
	created_at_utc TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS operations_run ON operations(run_id);
CREATE TABLE IF NOT EXISTS job_events (
	seq INTEGER PRIMARY KEY AUTOINCREMENT,
	job_id TEXT NOT NULL REFERENCES operations(job_id),
	state TEXT NOT NULL,
	worker_id TEXT,
	fencing_token INTEGER,
	payload_json TEXT NOT NULL,
	created_at_utc TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS job_events_job ON job_events(job_id, seq);
CREATE TABLE IF NOT EXISTS lease_events (
	seq INTEGER PRIMARY KEY AUTOINCREMENT,
	resource TEXT NOT NULL,
	worker_id TEXT NOT NULL,
	fencing_token INTEGER UNIQUE NOT NULL,
	created_at_utc TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS lease_events_resource ON lease_events(resource, fencing_token);
CREATE TABLE IF NOT EXISTS stop_events (
	seq INTEGER PRIMARY KEY AUTOINCREMENT,
	run_id TEXT NOT NULL,
	action TEXT NOT NULL CHECK(action IN ('LATCH','CLEAR')),
	reason TEXT NOT NULL,
	actor TEXT NOT NULL,
	created_at_utc TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS stop_events_run ON stop_events(run_id, seq);
"""


def _trigger_sql() -> str:
	parts = []
	for table in _TABLES:
		for op in ("UPDATE", "DELETE"):
			parts.append(
				f"CREATE TRIGGER IF NOT EXISTS {table}_no_{op.lower()} BEFORE {op} ON {table} "
				"BEGIN SELECT RAISE(ABORT, 'append-only'); END;"
			)
	return "\n".join(parts)


def job_resource(job_id: str) -> str:
	"""Lease resource name guarding one job."""
	return f"job:{job_id}"


def _now() -> str:
	return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class JobRecord:
	job_id: str
	idempotency_key: str
	op_type: str
	params: dict
	params_sha256: str
	run_id: str
	branch_id: str
	created_at_utc: str


@dataclass(frozen=True)
class JobEvent:
	seq: int
	job_id: str
	state: JobState
	worker_id: str | None
	fencing_token: int | None
	payload: dict
	created_at_utc: str


@dataclass(frozen=True)
class Lease:
	resource: str
	worker_id: str
	fencing_token: int


def _record(row: sqlite3.Row | tuple) -> JobRecord:
	return JobRecord(row[0], row[1], row[2], json.loads(row[3]), row[4], row[5], row[6], row[7])


def _event(row: sqlite3.Row | tuple) -> JobEvent:
	return JobEvent(row[0], row[1], JobState(row[2]), row[3], row[4], json.loads(row[5]), row[6])


_OP_COLS = "job_id, idempotency_key, op_type, params_json, params_sha256, run_id, branch_id, created_at_utc"
_EV_COLS = "seq, job_id, state, worker_id, fencing_token, payload_json, created_at_utc"


class OperationLedger:
	"""SQLite-backed append-only ledger; safe across threads and across processes sharing the file."""

	def __init__(self, path: Path) -> None:
		self._path = Path(path)
		self._lock = threading.RLock()
		self._conn: sqlite3.Connection | None = sqlite3.connect(
			self._path, timeout=_BUSY_TIMEOUT_MS / 1000, check_same_thread=False, isolation_level=None
		)
		try:
			self._conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
			self._conn.execute("PRAGMA journal_mode=WAL")
			self._conn.execute("PRAGMA synchronous=FULL")
			self._init_schema()
		except BaseException:
			self._conn.close()
			self._conn = None
			raise

	def _init_schema(self) -> None:
		conn = self._require()
		version = conn.execute("PRAGMA user_version").fetchone()[0]
		if version not in (0, SCHEMA_VERSION):
			raise LedgerError(f"unsupported ledger schema version {version} (expected {SCHEMA_VERSION})")
		with self._tx():
			version = conn.execute("PRAGMA user_version").fetchone()[0]
			if version not in (0, SCHEMA_VERSION):
				raise LedgerError(f"unsupported ledger schema version {version} (expected {SCHEMA_VERSION})")
			self._exec_script(conn, _SCHEMA)
			self._exec_script(conn, _trigger_sql())
			conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

	@staticmethod
	def _exec_script(conn: sqlite3.Connection, script: str) -> None:
		"""Run statements one by one so the surrounding explicit transaction stays open."""
		buf = ""
		for line in script.splitlines():
			buf += line + "\n"
			if sqlite3.complete_statement(buf):
				conn.execute(buf)
				buf = ""

	def _require(self) -> sqlite3.Connection:
		if self._conn is None:
			raise LedgerError("ledger is closed")
		return self._conn

	def close(self) -> None:
		with self._lock:
			if self._conn is not None:
				self._conn.close()
				self._conn = None

	def __enter__(self) -> Self:
		return self

	def __exit__(self, *exc: object) -> None:
		self.close()

	@contextmanager
	def _tx(self) -> Iterator[sqlite3.Connection]:
		with self._lock:
			conn = self._require()
			conn.execute("BEGIN IMMEDIATE")
			try:
				yield conn
			except BaseException:
				conn.execute("ROLLBACK")
				raise
			conn.execute("COMMIT")

	@contextmanager
	def _read(self) -> Iterator[sqlite3.Connection]:
		with self._lock:
			yield self._require()

	# -- registration and lookup

	def register(
		self, *, idempotency_key: str, op_type: str, params: dict, run_id: str, branch_id: str
	) -> tuple[JobRecord, bool]:
		"""Register a job idempotently; returns (record, created)."""
		if not idempotency_key:
			raise LedgerError("idempotency_key must be non-empty")
		try:
			params_json = canonical_json(params).decode("utf-8")
			digest = sha256_json(params)
		except (TypeError, ValueError) as exc:
			raise LedgerError(f"params are not canonical-JSON serialisable: {exc}") from exc
		with self._tx() as conn:
			row = conn.execute(
				f"SELECT {_OP_COLS} FROM operations WHERE idempotency_key=?", (idempotency_key,)
			).fetchone()
			if row is not None:
				existing = _record(row)
				same = (existing.op_type, existing.params_sha256, existing.run_id, existing.branch_id) == (
					op_type,
					digest,
					run_id,
					branch_id,
				)
				if not same:
					raise IdempotencyConflictError(
						f"idempotency key {idempotency_key!r} reused with a different request"
					)
				return existing, False
			if self._latched(conn, run_id):
				raise StopLatchedError(f"run {run_id!r} is stop-latched; new operations are refused")
			job_id = uuid.uuid4().hex
			now = _now()
			conn.execute(
				f"INSERT INTO operations ({_OP_COLS}) VALUES (?,?,?,?,?,?,?,?)",
				(job_id, idempotency_key, op_type, params_json, digest, run_id, branch_id, now),
			)
			conn.execute(
				"INSERT INTO job_events (job_id, state, worker_id, fencing_token, payload_json, created_at_utc) "
				"VALUES (?,?,NULL,NULL,'{}',?)",
				(job_id, JobState.QUEUED.value, now),
			)
			record = JobRecord(
				job_id, idempotency_key, op_type, json.loads(params_json), digest, run_id, branch_id, now
			)
		return record, True

	def job(self, job_id: str) -> JobRecord:
		with self._read() as conn:
			return self._job(conn, job_id)

	@staticmethod
	def _job(conn: sqlite3.Connection, job_id: str) -> JobRecord:
		row = conn.execute(f"SELECT {_OP_COLS} FROM operations WHERE job_id=?", (job_id,)).fetchone()
		if row is None:
			raise JobNotFoundError(f"unknown job {job_id!r}")
		return _record(row)

	def job_by_key(self, idempotency_key: str) -> JobRecord | None:
		with self._read() as conn:
			row = conn.execute(
				f"SELECT {_OP_COLS} FROM operations WHERE idempotency_key=?", (idempotency_key,)
			).fetchone()
		return None if row is None else _record(row)

	def events(self, job_id: str) -> list[JobEvent]:
		with self._read() as conn:
			self._job(conn, job_id)
			rows = conn.execute(f"SELECT {_EV_COLS} FROM job_events WHERE job_id=? ORDER BY seq", (job_id,)).fetchall()
		return [_event(r) for r in rows]

	def latest_event(self, job_id: str) -> JobEvent:
		with self._read() as conn:
			return self._latest(conn, job_id)

	@staticmethod
	def _latest(conn: sqlite3.Connection, job_id: str) -> JobEvent:
		row = conn.execute(
			f"SELECT {_EV_COLS} FROM job_events WHERE job_id=? ORDER BY seq DESC LIMIT 1", (job_id,)
		).fetchone()
		if row is None:
			raise JobNotFoundError(f"unknown job {job_id!r}")
		return _event(row)

	def state(self, job_id: str) -> JobState:
		return self.latest_event(job_id).state

	def jobs(self, run_id: str | None = None) -> list[JobRecord]:
		"""Every registered operation (optionally of one run), oldest first."""
		sql = f"SELECT {_OP_COLS} FROM operations"
		args: tuple[Any, ...] = ()
		if run_id is not None:
			sql += " WHERE run_id=?"
			args = (run_id,)
		with self._read() as conn:
			return [_record(r) for r in conn.execute(sql + " ORDER BY rowid", args).fetchall()]

	def events_in_state(self, state: JobState) -> list[JobEvent]:
		"""Every event that entered ``state``, across all jobs, in ledger order."""
		with self._read() as conn:
			rows = conn.execute(
				f"SELECT {_EV_COLS} FROM job_events WHERE state=? ORDER BY seq", (JobState(state).value,)
			).fetchall()
		return [_event(r) for r in rows]

	def incomplete_jobs(self, run_id: str | None = None) -> list[JobRecord]:
		"""Jobs whose current state is not terminal, oldest first."""
		cols = ", ".join(f"o.{c.strip()}" for c in _OP_COLS.split(","))
		sql = (
			f"SELECT {cols} FROM operations o "
			"JOIN job_events e ON e.seq = (SELECT MAX(seq) FROM job_events WHERE job_id = o.job_id) "
			"WHERE e.state NOT IN ('SUCCEEDED','FAILED','CANCELLED')"
		)
		args: tuple[Any, ...] = ()
		if run_id is not None:
			sql += " AND o.run_id=?"
			args = (run_id,)
		sql += " ORDER BY o.rowid"
		with self._read() as conn:
			return [_record(r) for r in conn.execute(sql, args).fetchall()]

	# -- leases

	def acquire_lease(self, resource: str, worker_id: str) -> Lease:
		"""Grant a new lease with the next global fencing token, superseding any previous holder."""
		if not resource or not worker_id:
			raise LedgerError("resource and worker_id must be non-empty")
		with self._tx() as conn:
			token = conn.execute("SELECT COALESCE(MAX(fencing_token), 0) + 1 FROM lease_events").fetchone()[0]
			conn.execute(
				"INSERT INTO lease_events (resource, worker_id, fencing_token, created_at_utc) VALUES (?,?,?,?)",
				(resource, worker_id, token, _now()),
			)
		return Lease(resource, worker_id, token)

	def current_lease(self, resource: str) -> Lease | None:
		with self._read() as conn:
			return self._current_lease(conn, resource)

	@staticmethod
	def _current_lease(conn: sqlite3.Connection, resource: str) -> Lease | None:
		row = conn.execute(
			"SELECT resource, worker_id, fencing_token FROM lease_events WHERE resource=? "
			"ORDER BY fencing_token DESC LIMIT 1",
			(resource,),
		).fetchone()
		return None if row is None else Lease(row[0], row[1], row[2])

	def assert_current(self, lease: Lease) -> None:
		"""Raise StaleWorkerError unless ``lease`` is exactly the current lease of its resource."""
		with self._read() as conn:
			self._check_lease(conn, lease)

	def _check_lease(self, conn: sqlite3.Connection, lease: Lease) -> None:
		if self._current_lease(conn, lease.resource) != lease:
			raise StaleWorkerError(f"lease {lease} is not current for resource {lease.resource!r}")

	# -- job transitions

	def append_event(
		self,
		job_id: str,
		state: JobState,
		*,
		lease: Lease | None = None,
		payload: dict | None = None,
	) -> JobEvent:
		"""Append a state transition, enforcing the state machine, fencing, and the stop latch atomically."""
		state = JobState(state)
		try:
			payload_json = canonical_json({} if payload is None else payload).decode("utf-8")
		except (TypeError, ValueError) as exc:
			raise LedgerError(f"payload is not canonical-JSON serialisable: {exc}") from exc
		with self._tx() as conn:
			record = self._job(conn, job_id)
			current = self._latest(conn, job_id).state
			pair = (current, state)
			if pair in _WORKER_TRANSITIONS:
				if lease is None or lease.resource != job_resource(job_id):
					raise StaleWorkerError(f"transition {current}->{state} requires the current lease of job {job_id}")
				self._check_lease(conn, lease)
				if state in _START_STATES and self._latched(conn, record.run_id):
					raise StopLatchedError(f"run {record.run_id!r} is stop-latched; job cannot start")
			elif pair in _COORDINATOR_TRANSITIONS:
				if lease is not None:
					raise LedgerError(f"transition {current}->{state} is coordinator-only; do not pass a lease")
			else:
				raise InvalidTransitionError(f"transition {current}->{state} is not allowed for job {job_id}")
			now = _now()
			worker_id = lease.worker_id if lease else None
			token = lease.fencing_token if lease else None
			cur = conn.execute(
				"INSERT INTO job_events (job_id, state, worker_id, fencing_token, payload_json, created_at_utc) "
				"VALUES (?,?,?,?,?,?)",
				(job_id, state.value, worker_id, token, payload_json, now),
			)
			seq = cur.lastrowid
		return JobEvent(seq, job_id, state, worker_id, token, json.loads(payload_json), now)

	def request_cancel(self, job_id: str, *, reason: str = "") -> JobState:
		"""Idempotently request cancellation; returns the resulting current state."""
		payload = {"reason": reason}
		with self._lock:
			current = self.state(job_id)
			if current is JobState.QUEUED:
				target = JobState.CANCELLED
			elif current is JobState.RUNNING:
				target = JobState.CANCEL_REQUESTED
			else:
				return current
			try:
				self.append_event(job_id, target, payload=payload)
			except InvalidTransitionError:
				# Another process moved the job between the read and the write; report what is true now.
				return self.state(job_id)
			return target

	# -- stop latch

	@staticmethod
	def _latched(conn: sqlite3.Connection, run_id: str) -> bool:
		row = conn.execute(
			"SELECT action FROM stop_events WHERE run_id=? ORDER BY seq DESC LIMIT 1", (run_id,)
		).fetchone()
		return row is not None and row[0] == "LATCH"

	def latch_stop(self, run_id: str, *, reason: str, actor: str) -> None:
		with self._tx() as conn:
			conn.execute(
				"INSERT INTO stop_events (run_id, action, reason, actor, created_at_utc) VALUES (?,?,?,?,?)",
				(run_id, "LATCH", reason, actor, _now()),
			)

	def clear_stop(self, run_id: str, *, reason: str, actor: str) -> None:
		"""Explicitly clear a latch; requires a reason and an actor."""
		if not reason or not actor:
			raise LedgerError("clear_stop requires a non-empty reason and actor")
		with self._tx() as conn:
			if not self._latched(conn, run_id):
				raise LedgerError(f"run {run_id!r} is not stop-latched")
			conn.execute(
				"INSERT INTO stop_events (run_id, action, reason, actor, created_at_utc) VALUES (?,?,?,?,?)",
				(run_id, "CLEAR", reason, actor, _now()),
			)

	def stop_latched(self, run_id: str) -> bool:
		with self._read() as conn:
			return self._latched(conn, run_id)

	def stop_history(self, run_id: str) -> list[dict]:
		with self._read() as conn:
			rows = conn.execute(
				"SELECT seq, action, reason, actor, created_at_utc FROM stop_events WHERE run_id=? ORDER BY seq",
				(run_id,),
			).fetchall()
		return [
			{"seq": r[0], "run_id": run_id, "action": r[1], "reason": r[2], "actor": r[3], "created_at_utc": r[4]}
			for r in rows
		]

	# -- budget

	def consumed(self, run_id: str) -> dict[str, float | int]:
		"""Budget consumed by a run (all branches), derived only from append-only history."""
		sim = 0.0
		wall = 0.0
		with self._read() as conn:
			operations = conn.execute("SELECT COUNT(*) FROM operations WHERE run_id=?", (run_id,)).fetchone()[0]
			attempts = conn.execute(
				"SELECT COUNT(*) FROM job_events e JOIN operations o ON o.job_id = e.job_id "
				"WHERE o.run_id=? AND e.state='RUNNING'",
				(run_id,),
			).fetchone()[0]
			rows = conn.execute(
				"SELECT e.payload_json FROM job_events e JOIN operations o ON o.job_id = e.job_id WHERE o.run_id=?",
				(run_id,),
			).fetchall()
		for (payload_json,) in rows:
			payload = json.loads(payload_json)
			sim += _number(payload.get("sim_time_consumed_s"))
			wall += _number(payload.get("wall_time_s"))
		return {"operations": operations, "attempts": attempts, "sim_time_s": sim, "wall_time_s": wall}


def _number(value: object) -> float:
	if isinstance(value, bool) or not isinstance(value, int | float):
		return 0.0
	return float(value)
