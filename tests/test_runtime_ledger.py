"""Tests for the append-only operation ledger."""

from __future__ import annotations

import itertools
import sqlite3
import threading
from pathlib import Path

import pytest

from digital_twin_armigo.runtime.errors import (
	IdempotencyConflictError,
	InvalidTransitionError,
	JobNotFoundError,
	LedgerError,
	StaleWorkerError,
	StopLatchedError,
)
from digital_twin_armigo.runtime.ledger import (
	TERMINAL_STATES,
	JobState,
	Lease,
	OperationLedger,
	job_resource,
)

S = JobState
TABLES = ("operations", "job_events", "lease_events", "stop_events")

WORKER_TRANSITIONS = {
	(S.QUEUED, S.RUNNING),
	(S.NEEDS_RECOVERY, S.RUNNING),
	(S.RUNNING, S.SUCCEEDED),
	(S.RUNNING, S.FAILED),
	(S.RUNNING, S.CANCELLED),
	(S.CANCEL_REQUESTED, S.CANCELLED),
	(S.CANCEL_REQUESTED, S.SUCCEEDED),
	(S.CANCEL_REQUESTED, S.FAILED),
}
COORD_TRANSITIONS = {
	(S.QUEUED, S.CANCELLED),
	(S.RUNNING, S.CANCEL_REQUESTED),
	(S.RUNNING, S.NEEDS_RECOVERY),
	(S.CANCEL_REQUESTED, S.NEEDS_RECOVERY),
	(S.NEEDS_RECOVERY, S.SUCCEEDED),
	(S.NEEDS_RECOVERY, S.FAILED),
}


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
	return tmp_path / "ledger.sqlite"


@pytest.fixture
def ledger(db_path: Path):
	with OperationLedger(db_path) as led:
		yield led


def _reg(led: OperationLedger, key: str = "k1", *, run: str = "run", branch: str = "main", params: dict | None = None):
	return led.register(
		idempotency_key=key,
		op_type="rollout",
		params={"a": 1} if params is None else params,
		run_id=run,
		branch_id=branch,
	)[0]


def _counts(path: Path) -> dict[str, int]:
	conn = sqlite3.connect(path)
	try:
		return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in TABLES}
	finally:
		conn.close()


def _drive(led: OperationLedger, state: JobState, key: str = "drive", run: str = "run") -> tuple[str, Lease]:
	"""Create a job and move it to ``state`` through legal transitions; return (job_id, lease)."""
	rec = _reg(led, key, run=run)
	lease = led.acquire_lease(job_resource(rec.job_id), "w1")
	if state is S.QUEUED:
		return rec.job_id, lease
	if state is S.CANCELLED:
		led.append_event(rec.job_id, S.CANCELLED)
		return rec.job_id, lease
	led.append_event(rec.job_id, S.RUNNING, lease=lease)
	if state is S.CANCEL_REQUESTED:
		led.append_event(rec.job_id, S.CANCEL_REQUESTED)
	elif state is S.NEEDS_RECOVERY:
		led.append_event(rec.job_id, S.NEEDS_RECOVERY)
	elif state is not S.RUNNING:
		led.append_event(rec.job_id, state, lease=lease)
	return rec.job_id, lease


# 1. registration / idempotency


def test_register_creates_queued(ledger: OperationLedger) -> None:
	rec, created = ledger.register(idempotency_key="k", op_type="rollout", params={"x": 1}, run_id="r", branch_id="b")
	assert created
	assert ledger.state(rec.job_id) is S.QUEUED
	assert ledger.job(rec.job_id) == rec
	assert ledger.job_by_key("k") == rec
	assert ledger.job_by_key("nope") is None
	events = ledger.events(rec.job_id)
	assert [e.state for e in events] == [S.QUEUED]
	assert events[0].worker_id is None and events[0].fencing_token is None
	assert rec.params == {"x": 1} and len(rec.params_sha256) == 64


def test_register_same_request_is_idempotent(ledger: OperationLedger, db_path: Path) -> None:
	first, c1 = ledger.register(idempotency_key="k", op_type="t", params={"a": 1, "b": 2}, run_id="r", branch_id="b")
	before = _counts(db_path)
	second, c2 = ledger.register(idempotency_key="k", op_type="t", params={"b": 2, "a": 1}, run_id="r", branch_id="b")
	assert (c1, c2) == (True, False)
	assert first.job_id == second.job_id
	assert _counts(db_path) == before


@pytest.mark.parametrize(
	"override",
	[{"params": {"a": 2}}, {"op_type": "other"}, {"run_id": "r2"}, {"branch_id": "b2"}],
)
def test_register_key_reuse_conflict(ledger: OperationLedger, db_path: Path, override: dict) -> None:
	base = {"idempotency_key": "k", "op_type": "t", "params": {"a": 1}, "run_id": "r", "branch_id": "b"}
	ledger.register(**base)
	before = _counts(db_path)
	with pytest.raises(IdempotencyConflictError):
		ledger.register(**{**base, **override})
	assert _counts(db_path) == before


def test_register_rejects_bad_input(ledger: OperationLedger, db_path: Path) -> None:
	with pytest.raises(LedgerError):
		ledger.register(idempotency_key="", op_type="t", params={}, run_id="r", branch_id="b")
	with pytest.raises(LedgerError):
		ledger.register(idempotency_key="k", op_type="t", params={"x": float("nan")}, run_id="r", branch_id="b")
	with pytest.raises(LedgerError):
		ledger.register(idempotency_key="k", op_type="t", params={"x": object()}, run_id="r", branch_id="b")
	assert _counts(db_path)["operations"] == 0


def test_unknown_job(ledger: OperationLedger) -> None:
	for fn in (ledger.job, ledger.events, ledger.latest_event, ledger.state):
		with pytest.raises(JobNotFoundError):
			fn("missing")
	with pytest.raises(JobNotFoundError):
		ledger.append_event("missing", S.CANCELLED)


# 2. happy path


def test_happy_path_events(ledger: OperationLedger) -> None:
	rec = _reg(ledger)
	lease = ledger.acquire_lease(job_resource(rec.job_id), "w1")
	ledger.append_event(rec.job_id, S.RUNNING, lease=lease)
	done = ledger.append_event(rec.job_id, S.SUCCEEDED, lease=lease, payload={"sim_time_consumed_s": 1.5})
	assert ledger.state(rec.job_id) is S.SUCCEEDED
	assert ledger.latest_event(rec.job_id) == done
	events = ledger.events(rec.job_id)
	assert [e.state for e in events] == [S.QUEUED, S.RUNNING, S.SUCCEEDED]
	assert [e.seq for e in events] == sorted(e.seq for e in events)
	assert events[1].worker_id == "w1" and events[1].fencing_token == lease.fencing_token
	assert events[2].payload == {"sim_time_consumed_s": 1.5}
	assert events[1].payload == {}
	assert events[0].created_at_utc


@pytest.mark.parametrize("terminal", sorted(TERMINAL_STATES))
@pytest.mark.parametrize("target", list(JobState))
def test_terminal_states_reject_everything(ledger: OperationLedger, terminal: JobState, target: JobState) -> None:
	job_id, lease = _drive(ledger, terminal)
	before = len(ledger.events(job_id))
	with pytest.raises(InvalidTransitionError):
		ledger.append_event(job_id, target, lease=lease)
	with pytest.raises(InvalidTransitionError):
		ledger.append_event(job_id, target)
	assert len(ledger.events(job_id)) == before


# 3. fencing


def test_tokens_increase_across_resources_and_reopen(db_path: Path) -> None:
	tokens = []
	with OperationLedger(db_path) as led:
		tokens += [led.acquire_lease("a", "w").fencing_token, led.acquire_lease("b", "w").fencing_token]
		tokens.append(led.acquire_lease("a", "w").fencing_token)
	with OperationLedger(db_path) as led:
		tokens.append(led.acquire_lease("c", "w2").fencing_token)
	assert tokens == sorted(set(tokens)) and tokens[0] == 1


def test_superseded_lease_is_fenced(ledger: OperationLedger, db_path: Path) -> None:
	rec = _reg(ledger)
	res = job_resource(rec.job_id)
	old = ledger.acquire_lease(res, "w1")
	ledger.append_event(rec.job_id, S.RUNNING, lease=old)
	new = ledger.acquire_lease(res, "w1")  # same worker id, still supersedes
	assert new.fencing_token > old.fencing_token
	before = _counts(db_path)
	with pytest.raises(StaleWorkerError):
		ledger.append_event(rec.job_id, S.SUCCEEDED, lease=old)
	with pytest.raises(StaleWorkerError):
		ledger.assert_current(old)
	assert _counts(db_path) == before
	ledger.assert_current(new)
	assert ledger.current_lease(res) == new
	assert ledger.current_lease("nothing") is None
	ledger.append_event(rec.job_id, S.SUCCEEDED, lease=new)


def test_lease_bound_to_its_job(ledger: OperationLedger, db_path: Path) -> None:
	a, b = _reg(ledger, "a"), _reg(ledger, "b")
	lease_a = ledger.acquire_lease(job_resource(a.job_id), "w")
	before = _counts(db_path)
	with pytest.raises(StaleWorkerError):
		ledger.append_event(b.job_id, S.RUNNING, lease=lease_a)
	with pytest.raises(StaleWorkerError):
		ledger.append_event(b.job_id, S.RUNNING)
	with pytest.raises(StaleWorkerError):
		ledger.append_event(a.job_id, S.RUNNING, lease=Lease(lease_a.resource, "other", lease_a.fencing_token))
	assert _counts(db_path) == before


# 4. transition table


@pytest.mark.parametrize(("src", "dst"), list(itertools.product(JobState, JobState)))
def test_transition_table(ledger: OperationLedger, src: JobState, dst: JobState) -> None:
	job_id, lease = _drive(ledger, src)
	if (src, dst) in WORKER_TRANSITIONS:
		with pytest.raises(StaleWorkerError):
			ledger.append_event(job_id, dst)
		with pytest.raises(StaleWorkerError):
			ledger.append_event(job_id, dst, lease=Lease(lease.resource, lease.worker_id, lease.fencing_token + 99))
		assert ledger.state(job_id) is src
		ledger.append_event(job_id, dst, lease=lease)
	elif (src, dst) in COORD_TRANSITIONS:
		with pytest.raises(LedgerError):
			ledger.append_event(job_id, dst, lease=lease)
		assert ledger.state(job_id) is src
		ev = ledger.append_event(job_id, dst)
		assert ev.worker_id is None and ev.fencing_token is None
	else:
		with pytest.raises(InvalidTransitionError):
			ledger.append_event(job_id, dst, lease=lease)
		with pytest.raises(InvalidTransitionError):
			ledger.append_event(job_id, dst)
		assert ledger.state(job_id) is src
		return
	assert ledger.state(job_id) is dst


# 5. cancel


def test_cancel_queued(ledger: OperationLedger) -> None:
	rec = _reg(ledger)
	assert ledger.request_cancel(rec.job_id, reason="no") is S.CANCELLED
	assert ledger.latest_event(rec.job_id).payload == {"reason": "no"}


def test_cancel_running_then_worker_cancels(ledger: OperationLedger) -> None:
	job_id, lease = _drive(ledger, S.RUNNING)
	assert ledger.request_cancel(job_id) is S.CANCEL_REQUESTED
	ledger.append_event(job_id, S.CANCELLED, lease=lease)
	assert ledger.state(job_id) is S.CANCELLED


def test_cancel_requested_job_may_still_succeed(ledger: OperationLedger) -> None:
	job_id, lease = _drive(ledger, S.RUNNING)
	ledger.request_cancel(job_id)
	ledger.append_event(job_id, S.SUCCEEDED, lease=lease)
	assert ledger.state(job_id) is S.SUCCEEDED


def test_request_cancel_idempotent(ledger: OperationLedger) -> None:
	job_id, _ = _drive(ledger, S.RUNNING)
	assert ledger.request_cancel(job_id) is S.CANCEL_REQUESTED
	n = len(ledger.events(job_id))
	assert ledger.request_cancel(job_id) is S.CANCEL_REQUESTED
	assert len(ledger.events(job_id)) == n
	done, _ = _drive(ledger, S.SUCCEEDED, key="done")
	n = len(ledger.events(done))
	assert ledger.request_cancel(done) is S.SUCCEEDED
	rec, _ = _drive(ledger, S.NEEDS_RECOVERY, key="rec")
	assert ledger.request_cancel(rec) is S.NEEDS_RECOVERY
	assert len(ledger.events(done)) == n


# 6. recovery


def test_recovery_with_new_lease(ledger: OperationLedger) -> None:
	job_id, old = _drive(ledger, S.RUNNING)
	ledger.append_event(job_id, S.NEEDS_RECOVERY)
	with pytest.raises(StaleWorkerError):
		ledger.append_event(job_id, S.RUNNING)
	new = ledger.acquire_lease(job_resource(job_id), "w2")
	with pytest.raises(StaleWorkerError):
		ledger.append_event(job_id, S.RUNNING, lease=old)
	ledger.append_event(job_id, S.RUNNING, lease=new)
	with pytest.raises(StaleWorkerError):
		ledger.append_event(job_id, S.SUCCEEDED, lease=old)
	ledger.append_event(job_id, S.SUCCEEDED, lease=new)
	assert ledger.state(job_id) is S.SUCCEEDED
	assert [e.state for e in ledger.events(job_id)].count(S.RUNNING) == 2


def test_recovery_adopts_result(ledger: OperationLedger) -> None:
	job_id, _ = _drive(ledger, S.NEEDS_RECOVERY)
	ev = ledger.append_event(job_id, S.SUCCEEDED, payload={"adopted": "snap-1"})
	assert ev.payload == {"adopted": "snap-1"}
	assert ledger.state(job_id) is S.SUCCEEDED


def test_incomplete_jobs(ledger: OperationLedger) -> None:
	a, _ = _drive(ledger, S.RUNNING, key="a", run="r1")
	b, _ = _drive(ledger, S.QUEUED, key="b", run="r2")
	_drive(ledger, S.SUCCEEDED, key="c", run="r1")
	_drive(ledger, S.NEEDS_RECOVERY, key="d", run="r1")
	assert [r.idempotency_key for r in ledger.incomplete_jobs()] == ["a", "b", "d"]
	assert [r.job_id for r in ledger.incomplete_jobs("r2")] == [b]
	assert a in {r.job_id for r in ledger.incomplete_jobs("r1")}


# 7. append-only


@pytest.mark.parametrize("table", TABLES)
def test_append_only_triggers(ledger: OperationLedger, db_path: Path, table: str) -> None:
	rec = _reg(ledger)
	ledger.acquire_lease(job_resource(rec.job_id), "w")
	ledger.latch_stop("run", reason="r", actor="a")
	before = _counts(db_path)
	assert before[table] >= 1
	conn = sqlite3.connect(db_path)
	try:
		with pytest.raises(sqlite3.DatabaseError):
			conn.execute(f"DELETE FROM {table}")
		first_col = conn.execute(f"SELECT name FROM pragma_table_info('{table}') LIMIT 1").fetchone()[0]
		with pytest.raises(sqlite3.DatabaseError):
			conn.execute(f"UPDATE {table} SET {first_col} = {first_col}")
	finally:
		conn.close()
	assert _counts(db_path) == before


# 8. stop latch


def test_latch_blocks_new_but_returns_existing(ledger: OperationLedger, db_path: Path) -> None:
	rec = _reg(ledger, "old")
	ledger.latch_stop("run", reason="halt", actor="op")
	assert ledger.stop_latched("run")
	before = _counts(db_path)
	with pytest.raises(StopLatchedError):
		_reg(ledger, "new")
	again, created = ledger.register(
		idempotency_key="old", op_type="rollout", params={"a": 1}, run_id="run", branch_id="main"
	)
	assert not created and again.job_id == rec.job_id
	assert _counts(db_path) == before


def test_latch_blocks_start_allows_finish(ledger: OperationLedger) -> None:
	queued, qlease = _drive(ledger, S.QUEUED, key="q")
	running, rlease = _drive(ledger, S.RUNNING, key="r")
	recov, clease = _drive(ledger, S.NEEDS_RECOVERY, key="n")
	ledger.latch_stop("run", reason="halt", actor="op")
	with pytest.raises(StopLatchedError):
		ledger.append_event(queued, S.RUNNING, lease=qlease)
	with pytest.raises(StopLatchedError):
		ledger.append_event(recov, S.RUNNING, lease=clease)
	assert ledger.state(queued) is S.QUEUED
	ledger.append_event(running, S.SUCCEEDED, lease=rlease)
	ledger.request_cancel(queued)
	assert ledger.state(running) is S.SUCCEEDED and ledger.state(queued) is S.CANCELLED


def test_latch_survives_reopen_and_clear(db_path: Path) -> None:
	with OperationLedger(db_path) as led:
		led.latch_stop("run", reason="halt", actor="op")
	with OperationLedger(db_path) as led:
		assert led.stop_latched("run")
		led.acquire_lease("x", "w")
		assert led.stop_latched("run")
		with pytest.raises(LedgerError):
			led.clear_stop("run", reason="", actor="op")
		with pytest.raises(LedgerError):
			led.clear_stop("run", reason="ok", actor="")
		assert led.stop_latched("run")
		led.clear_stop("run", reason="all clear", actor="op")
		assert not led.stop_latched("run")
		_rec, created = led.register(idempotency_key="n", op_type="t", params={}, run_id="run", branch_id="b")
		assert created
		with pytest.raises(LedgerError):
			led.clear_stop("run", reason="again", actor="op")


def test_latch_is_per_run_and_history_ordered(ledger: OperationLedger) -> None:
	ledger.latch_stop("A", reason="r1", actor="x")
	assert not ledger.stop_latched("B")
	_reg(ledger, "b-job", run="B")
	ledger.latch_stop("A", reason="r2", actor="y")  # re-latch allowed
	ledger.clear_stop("A", reason="r3", actor="z")
	hist = ledger.stop_history("A")
	assert [h["action"] for h in hist] == ["LATCH", "LATCH", "CLEAR"]
	assert [h["actor"] for h in hist] == ["x", "y", "z"]
	assert hist[0]["reason"] == "r1"
	assert ledger.stop_history("B") == []
	with pytest.raises(LedgerError):
		ledger.clear_stop("B", reason="r", actor="a")


# 9. budget


def test_consumed_budget(db_path: Path) -> None:
	with OperationLedger(db_path) as led:
		a = _reg(led, "a", branch="b1")
		b = _reg(led, "b", branch="b2")
		other = _reg(led, "o", run="other")
		lease_a = led.acquire_lease(job_resource(a.job_id), "w")
		lease_b = led.acquire_lease(job_resource(b.job_id), "w")
		lease_o = led.acquire_lease(job_resource(other.job_id), "w")
		led.append_event(a.job_id, S.RUNNING, lease=lease_a)
		led.append_event(a.job_id, S.NEEDS_RECOVERY, payload={"wall_time_s": 1.0})
		lease_a2 = led.acquire_lease(job_resource(a.job_id), "w2")
		led.append_event(a.job_id, S.RUNNING, lease=lease_a2)
		led.append_event(
			a.job_id, S.SUCCEEDED, lease=lease_a2, payload={"sim_time_consumed_s": 2.0, "wall_time_s": 3.0}
		)
		led.append_event(b.job_id, S.RUNNING, lease=lease_b)
		led.append_event(b.job_id, S.FAILED, lease=lease_b, payload={"sim_time_consumed_s": 0.5})
		led.append_event(other.job_id, S.RUNNING, lease=lease_o)
		led.append_event(other.job_id, S.SUCCEEDED, lease=lease_o, payload={"sim_time_consumed_s": 100.0})
		expected = {"operations": 2, "attempts": 3, "sim_time_s": 2.5, "wall_time_s": 4.0}
		assert led.consumed("run") == expected
		assert led.consumed("other")["operations"] == 1
		assert led.consumed("nothing") == {"operations": 0, "attempts": 0, "sim_time_s": 0.0, "wall_time_s": 0.0}
	with OperationLedger(db_path) as led:
		assert led.consumed("run") == expected


# 10. durability / atomicity


def test_reopen_preserves_everything(db_path: Path) -> None:
	with OperationLedger(db_path) as led:
		job_id, _ = _drive(led, S.SUCCEEDED)
		rec, events, state = led.job(job_id), led.events(job_id), led.state(job_id)
	with OperationLedger(db_path) as led:
		assert (led.job(job_id), led.events(job_id), led.state(job_id)) == (rec, events, state)


def test_failed_transition_leaves_ledger_unchanged(ledger: OperationLedger, db_path: Path) -> None:
	rec = _reg(ledger)
	lease = ledger.acquire_lease(job_resource(rec.job_id), "w")
	before = _counts(db_path)
	with pytest.raises(InvalidTransitionError):
		ledger.append_event(rec.job_id, S.SUCCEEDED, lease=lease)
	with pytest.raises(LedgerError):
		ledger.append_event(rec.job_id, S.RUNNING, lease=lease, payload={"bad": object()})
	assert _counts(db_path) == before
	assert ledger.state(rec.job_id) is S.QUEUED
	ledger.append_event(rec.job_id, S.RUNNING, lease=lease)  # connection still usable after rollback


def test_user_version(db_path: Path) -> None:
	with OperationLedger(db_path):
		pass
	conn = sqlite3.connect(db_path)
	assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
	conn.execute("PRAGMA user_version = 99")
	conn.close()
	with pytest.raises(LedgerError):
		OperationLedger(db_path)


# 11. concurrency


def test_concurrent_register_same_key(ledger: OperationLedger) -> None:
	results: list = []
	barrier = threading.Barrier(8)

	def work() -> None:
		barrier.wait()
		results.append(ledger.register(idempotency_key="k", op_type="t", params={"a": 1}, run_id="r", branch_id="b"))

	threads = [threading.Thread(target=work) for _ in range(8)]
	for t in threads:
		t.start()
	for t in threads:
		t.join()
	assert len(results) == 8
	assert len({rec.job_id for rec, _ in results}) == 1
	assert sum(created for _, created in results) == 1


def test_concurrent_acquire_lease(ledger: OperationLedger) -> None:
	leases: list[Lease] = []
	barrier = threading.Barrier(8)

	def work(i: int) -> None:
		barrier.wait()
		leases.append(ledger.acquire_lease("res", f"w{i}"))

	threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
	for t in threads:
		t.start()
	for t in threads:
		t.join()
	assert len({lease.fencing_token for lease in leases}) == 8
	passing = []
	for lease in leases:
		try:
			ledger.assert_current(lease)
			passing.append(lease)
		except StaleWorkerError:
			pass
	assert len(passing) == 1
	assert passing[0].fencing_token == max(lease.fencing_token for lease in leases)


def test_two_instances_share_state(db_path: Path) -> None:
	with OperationLedger(db_path) as one, OperationLedger(db_path) as two:
		rec = _reg(one)
		assert two.job(rec.job_id) == rec
		lease = two.acquire_lease(job_resource(rec.job_id), "w")
		assert one.current_lease(job_resource(rec.job_id)) == lease
		two.latch_stop("run", reason="r", actor="a")
		assert one.stop_latched("run")
		with pytest.raises(StopLatchedError):
			one.append_event(rec.job_id, S.RUNNING, lease=lease)
		_, created = two.register(
			idempotency_key="k1", op_type="rollout", params={"a": 1}, run_id="run", branch_id="main"
		)
		assert not created


def test_jobs_and_events_in_state_list_history_in_order(tmp_path: Path) -> None:
	with OperationLedger(tmp_path / "ledger.sqlite") as ledger:
		a, _ = ledger.register(idempotency_key="a", op_type="rollout", params={}, run_id="r1", branch_id="main")
		b, _ = ledger.register(idempotency_key="b", op_type="fork", params={}, run_id="r2", branch_id="main")
		assert [j.job_id for j in ledger.jobs()] == [a.job_id, b.job_id]
		assert [j.job_id for j in ledger.jobs("r2")] == [b.job_id]
		for record in (b, a):
			lease = ledger.acquire_lease(job_resource(record.job_id), "w")
			ledger.append_event(record.job_id, JobState.RUNNING, lease=lease)
			ledger.append_event(
				record.job_id, JobState.SUCCEEDED, lease=lease, payload={"output_snapshot_id": record.job_id}
			)
		done = ledger.events_in_state(JobState.SUCCEEDED)
		assert [e.job_id for e in done] == [b.job_id, a.job_id]
		assert [e.payload["output_snapshot_id"] for e in done] == [b.job_id, a.job_id]
		assert ledger.events_in_state(JobState.FAILED) == []
