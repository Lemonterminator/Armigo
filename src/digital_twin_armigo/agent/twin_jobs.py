"""Durable job registry for the twin service: SQLite (WAL) rows shared by the server and its worker processes.

The store is also the IPC channel between the MCP server process and simulation workers: workers write progress rows
and poll the cancel flag, so the same mechanism serves in-process threads and spawned processes.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path

QUEUED, RUNNING, COMPLETE, FAILED, CANCELLED, INTERRUPTED = (
	"queued",
	"running",
	"complete",
	"failed",
	"cancelled",
	"interrupted",
)
ACTIVE = (QUEUED, RUNNING)
TERMINAL = (COMPLETE, FAILED, CANCELLED, INTERRUPTED)
RESUMABLE = (INTERRUPTED, FAILED, CANCELLED)
CANCEL_REQUESTED, INTERRUPT_REQUESTED = 1, 2  # values of the ``stop`` column

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
	job_id TEXT PRIMARY KEY,
	request_id TEXT NOT NULL UNIQUE,
	kind TEXT NOT NULL,
	arguments TEXT NOT NULL,
	status TEXT NOT NULL,
	stop INTEGER NOT NULL DEFAULT 0,
	parent_job_id TEXT,
	path TEXT,
	node TEXT,
	branch INTEGER,
	frame INTEGER,
	views INTEGER,
	progress_at REAL,
	result TEXT,
	error_code TEXT,
	created_at REAL NOT NULL,
	started_at REAL,
	finished_at REAL,
	updated_at REAL NOT NULL
)
"""


class JobCancelled(Exception):
	"""Raised inside a simulation to stop it cooperatively (``interrupted`` is True for a server shutdown)."""

	def __init__(self, interrupted=False):
		super().__init__("job interrupted" if interrupted else "job cancelled")
		self.interrupted = interrupted

	def __reduce__(self):
		return JobCancelled, (self.interrupted,)


def canonical(value):
	return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class JobStore:
	"""One SQLite file, one connection per thread, safe for several processes (WAL + busy timeout)."""

	def __init__(self, path):
		self.path = Path(path)
		self.path.parent.mkdir(parents=True, exist_ok=True)
		self._local = threading.local()
		self.db.execute(SCHEMA)

	@property
	def db(self):
		db = getattr(self._local, "db", None)
		if db is None:
			db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
			db.row_factory = sqlite3.Row
			db.execute("PRAGMA journal_mode=WAL")
			db.execute("PRAGMA synchronous=NORMAL")
			db.execute("PRAGMA busy_timeout=30000")
			self._local.db = db
		return db

	def close(self):
		db = getattr(self._local, "db", None)
		if db is not None:
			db.close()
			self._local.db = None

	@staticmethod
	def _row(row):
		if row is None:
			return None
		out = dict(row)
		out["arguments"] = json.loads(out["arguments"])
		out["result"] = json.loads(out["result"]) if out["result"] is not None else None
		out["progress"] = {
			"node": out.pop("node"),
			"branch": out.pop("branch"),
			"frame": out.pop("frame"),
			"views": out.pop("views"),
			"updated_at": out.pop("progress_at"),
		}
		return out

	def create(self, request_id, kind, arguments, root=None, parent_job_id=None, max_active=None):
		"""Insert a queued job, or return the existing one for a repeated ``request_id``.

		Returns ``(job_id, created)``. Reusing a request_id with different arguments raises ``ValueError``.
		"""
		text = canonical(arguments)
		db = self.db
		db.execute("BEGIN IMMEDIATE")
		try:
			row = db.execute("SELECT job_id, kind, arguments FROM jobs WHERE request_id=?", (request_id,)).fetchone()
			if row is not None:
				if row["arguments"] != text or row["kind"] != kind:
					raise ValueError("request_id is already bound to different arguments")
				db.execute("COMMIT")
				return row["job_id"], False
			if max_active is not None:
				active = db.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0]
				if active >= max_active:
					raise ValueError("simulation queue is full")
			job_id = uuid.uuid4().hex
			now = time.time()
			path = str(Path(root) / job_id) if root is not None else None
			db.execute(
				"INSERT INTO jobs (job_id, request_id, kind, arguments, status, parent_job_id, path, created_at,"
				" updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
				(job_id, request_id, kind, text, QUEUED, parent_job_id, path, now, now),
			)
			db.execute("COMMIT")
			return job_id, True
		except BaseException:
			if db.in_transaction:
				db.execute("ROLLBACK")
			raise

	def get(self, job_id):
		return self._row(self.db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone())

	def children(self, job_id):
		rows = self.db.execute("SELECT * FROM jobs WHERE parent_job_id=? ORDER BY created_at", (job_id,)).fetchall()
		return [self._row(row) for row in rows]

	def active_ids(self):
		return [r[0] for r in self.db.execute("SELECT job_id FROM jobs WHERE status IN ('queued','running')")]

	def mark_running(self, job_id):
		"""queued -> running. False when the job was cancelled (or otherwise left the queue) first."""
		now = time.time()
		cur = self.db.execute(
			"UPDATE jobs SET status='running', started_at=?, updated_at=? WHERE job_id=? AND status='queued'",
			(now, now, job_id),
		)
		return cur.rowcount == 1

	def progress(self, job_id, node, branch, frame, views):
		now = time.time()
		self.db.execute(
			"UPDATE jobs SET node=?, branch=?, frame=?, views=?, progress_at=?, updated_at=? WHERE job_id=?",
			(node, branch, frame, views, now, now, job_id),
		)

	def finish(self, job_id, status, result=None, error_code=None):
		now = time.time()
		text = json.dumps(result, allow_nan=False) if result is not None else None
		self.db.execute(
			"UPDATE jobs SET status=?, result=?, error_code=?, finished_at=?, updated_at=?"
			" WHERE job_id=? AND status IN ('queued','running')",
			(status, text, error_code, now, now, job_id),
		)

	def cancel(self, job_id):
		"""queued -> cancelled immediately; running -> cooperative stop flag. Returns the resulting status."""
		now = time.time()
		cur = self.db.execute(
			"UPDATE jobs SET status='cancelled', finished_at=?, updated_at=? WHERE job_id=? AND status='queued'",
			(now, now, job_id),
		)
		if cur.rowcount:
			return CANCELLED
		self.db.execute(
			"UPDATE jobs SET stop=?, updated_at=? WHERE job_id=? AND status='running' AND stop=0",
			(CANCEL_REQUESTED, now, job_id),
		)
		row = self.get(job_id)
		if row is None:
			raise ValueError("unknown job")
		return row["status"]

	def request_interrupt(self):
		"""Ask every running job to stop and end as ``interrupted`` (server shutdown)."""
		self.db.execute("UPDATE jobs SET stop=? WHERE status='running' AND stop=0", (INTERRUPT_REQUESTED,))

	def stop_requested(self, job_id):
		row = self.db.execute("SELECT status, stop FROM jobs WHERE job_id=?", (job_id,)).fetchone()
		if row is None or row["status"] == CANCELLED:
			return CANCEL_REQUESTED
		return row["stop"]

	def interrupt_stale(self):
		"""Rows left queued/running by a dead process become ``interrupted``. Returns how many."""
		now = time.time()
		cur = self.db.execute(
			"UPDATE jobs SET status='interrupted', finished_at=?, updated_at=? WHERE status IN ('queued','running')",
			(now, now),
		)
		return cur.rowcount


class JobHooks:
	"""Per-frame callback for ``TwinMujoco``: cooperative cancel plus throttled progress rows."""

	def __init__(self, store, job_id, check_every=10, progress_every=30, report=True):
		self.store = store
		self.job_id = job_id
		self.report = report
		self.check_every = check_every
		self.progress_every = progress_every
		self.count = 0
		self.last_node = None

	def check(self):
		stop = self.store.stop_requested(self.job_id)
		if stop:
			raise JobCancelled(interrupted=stop == INTERRUPT_REQUESTED)

	def __call__(self, backend):
		count = self.count
		self.count += 1
		if count % self.check_every == 0:
			self.check()
		if self.report and (count % self.progress_every == 0 or backend.node != self.last_node):
			self.last_node = backend.node
			self.store.progress(
				self.job_id,
				backend.node,
				backend.branch,
				backend.ep.n_frames,
				backend.wrist_count + backend.bottom_count,
			)


def open_saver(path):
	"""LangGraph SqliteSaver on its own WAL database (a connection per call; safe across processes)."""
	from langgraph.checkpoint.sqlite import SqliteSaver

	db = sqlite3.connect(path, timeout=30, check_same_thread=False)
	db.execute("PRAGMA journal_mode=WAL")
	db.execute("PRAGMA busy_timeout=30000")
	return SqliteSaver(db)
