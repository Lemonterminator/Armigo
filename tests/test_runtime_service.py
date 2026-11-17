"""Execution service: idempotent submission, fenced workers, crash recovery, cancellation, budgets, stop latch.

Runs on the hardware-free toy simulator so every fault can be injected deterministically in milliseconds; the MuJoCo
end-to-end checks live in ``test_runtime_service_mujoco.py``.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from digital_twin_armigo.runtime.errors import (
	BudgetExceededError,
	IdempotencyConflictError,
	IdentityMismatchError,
	LedgerError,
	SnapshotCorruptError,
	SnapshotNotFoundError,
	StaleWorkerError,
	StopLatchedError,
)
from digital_twin_armigo.runtime.ledger import JobState, OperationLedger
from digital_twin_armigo.runtime.service import ExecutionService, FaultHooks, RolloutWorker, RunBudget
from digital_twin_armigo.runtime.simulator import ControlProgram, diff_session_states
from digital_twin_armigo.runtime.snapshot import ExperimentIdentity, SnapshotStore
from digital_twin_armigo.runtime.toy_sim import ToyAdapter, toy_scene_spec

IDENTITY = ExperimentIdentity("c" * 64, "a" * 64, "f" * 64, "toy", "1", "3.12.0", "2.0.0", "test-os")
CONTRACT = {"task": "sort", "evaluator_version": "v2-audited", "thresholds": {"settle_s": 0.5}}
SPEC = toy_scene_spec(seed=11)


class SimulatedCrash(BaseException):
	"""Process death: derives from BaseException so no ``except Exception`` cleanup path can soften it."""


def crash() -> None:
	raise SimulatedCrash


def make_program(n: int = 60, shift: float = 0.0) -> ControlProgram:
	t = np.linspace(0.0, 1.0, n)[:, None]
	targets = np.hstack([np.sin(3.0 * t) + shift, t, -t * t])
	return ControlProgram(30, targets, np.zeros(n, dtype=np.uint8))


def straight(prog: ControlProgram, frames: int | None = None, spec: dict = SPEC):
	sess = ToyAdapter().open(spec)
	sess.load_program(prog)
	sess.step(len(prog) if frames is None else frames)
	return sess.capture()


class Rig:
	"""One store + ledger on disk; ``service()`` opens a fresh service on them, like a restarted process."""

	def __init__(self, tmp_path) -> None:
		self.store_dir = tmp_path / "store"
		self.ledger_path = tmp_path / "ledger.sqlite"
		self._ledgers: list[OperationLedger] = []

	def service(self, *, identity=IDENTITY, adapter=None, **kw) -> ExecutionService:
		ledger = OperationLedger(self.ledger_path)
		self._ledgers.append(ledger)
		return ExecutionService(SnapshotStore(self.store_dir), ledger, adapter or ToyAdapter(), identity, **kw)

	def close(self) -> None:
		for ledger in self._ledgers:
			ledger.close()


@pytest.fixture
def rig(tmp_path):
	r = Rig(tmp_path)
	yield r
	r.close()


def new_root(svc: ExecutionService, key: str = "root", spec: dict = SPEC, run_id: str = "run"):
	return svc.create_root_snapshot(
		idempotency_key=key, run_id=run_id, episode_id="ep0", branch_id="main", scene_spec=spec, contract=CONTRACT
	)


def running_events(svc: ExecutionService, job_id: str) -> int:
	return sum(e.state is JobState.RUNNING for e in svc.ledger.events(job_id))


# -- happy path and resume --------------------------------------------------------------------


def test_rollout_from_root_matches_a_straight_run_and_records_lineage(rig):
	svc = rig.service()
	root = new_root(svc)
	prog = make_program()
	job = svc.submit_rollout(idempotency_key="r1", snapshot_id=root.snapshot_id, program=prog)
	assert svc.get_rollout(job).state is JobState.QUEUED
	status = RolloutWorker(svc, "w1").run(job)
	assert status.state is JobState.SUCCEEDED and status.error is None
	assert status.input_snapshot_id == root.snapshot_id
	out = svc.store.verify(status.output_snapshot_id)
	assert svc.store.lineage(out.snapshot_id) == [out.snapshot_id, root.snapshot_id]
	assert out.produced_by["job_id"] == job and out.produced_by["worker_id"] == "w1"
	assert (out.run_id, out.branch_id, out.contract, out.scene_spec) == ("run", "main", CONTRACT, SPEC)
	result = status.result
	assert result["frames_executed"] == len(prog) and result["program_completed"] is True
	assert result["plan_sha256"] == prog.sha256()
	assert (result["control_step_start"], result["control_step_end"]) == (0, len(prog))
	assert diff_session_states(svc.load_state(out.snapshot_id), straight(prog)) == []
	assert svc.get_rollout(job) == status


def test_chained_rollouts_resume_bit_exactly_across_a_process_restart(rig):
	prog = make_program()
	svc = rig.service()
	root = new_root(svc)
	first = svc.submit_rollout(idempotency_key="a", snapshot_id=root.snapshot_id, program=prog, max_frames=23)
	mid = RolloutWorker(svc, "w1").run(first)
	assert mid.result["frames_executed"] == 23 and mid.result["program_completed"] is False

	restarted = rig.service()  # new process: nothing in memory, only the store and the ledger
	second = restarted.submit_rollout(idempotency_key="b", snapshot_id=mid.output_snapshot_id)
	done = RolloutWorker(restarted, "w2").run(second)
	assert done.result["frames_executed"] == len(prog) - 23 and done.result["program_completed"] is True
	assert diff_session_states(restarted.load_state(done.output_snapshot_id), straight(prog)) == []
	assert restarted.store.lineage(done.output_snapshot_id) == [
		done.output_snapshot_id,
		mid.output_snapshot_id,
		root.snapshot_id,
	]


def test_candidates_forked_from_one_root_are_isolated(rig):
	svc = rig.service()
	root = new_root(svc)
	root_file = svc.store.manifests_dir / f"{root.snapshot_id}.json"
	before = root_file.read_bytes()
	progs = {"a": make_program(), "b": make_program(shift=0.5)}
	outs = {}
	for name, prog in progs.items():
		job = svc.submit_rollout(idempotency_key=name, snapshot_id=root.snapshot_id, program=prog)
		outs[name] = RolloutWorker(svc, f"w-{name}").run(job)
	assert root_file.read_bytes() == before
	for name, prog in progs.items():
		manifest = svc.store.verify(outs[name].output_snapshot_id)
		assert manifest.parent_snapshot_id == root.snapshot_id
		assert diff_session_states(svc.load_state(manifest.snapshot_id), straight(prog)) == []
	assert outs["a"].output_snapshot_id != outs["b"].output_snapshot_id


# -- idempotency ------------------------------------------------------------------------------


def test_duplicate_submission_returns_the_same_job_and_never_executes_twice(rig):
	svc = rig.service()
	root = new_root(svc)
	prog = make_program()
	job = svc.submit_rollout(idempotency_key="k", snapshot_id=root.snapshot_id, program=prog)
	assert svc.submit_rollout(idempotency_key="k", snapshot_id=root.snapshot_id, program=prog) == job
	first = RolloutWorker(svc, "w1").run(job)
	again = RolloutWorker(svc, "w2").run(job)  # a retry after the answer was lost
	assert again == first
	assert svc.submit_rollout(idempotency_key="k", snapshot_id=root.snapshot_id, program=prog) == job
	assert running_events(svc, job) == 1
	assert svc.ledger.consumed("run")["attempts"] == 2  # root creation + one rollout attempt


def test_same_key_with_different_parameters_is_rejected(rig):
	svc = rig.service()
	root = new_root(svc)
	job = svc.submit_rollout(idempotency_key="k", snapshot_id=root.snapshot_id, program=make_program())
	with pytest.raises(IdempotencyConflictError):
		svc.submit_rollout(idempotency_key="k", snapshot_id=root.snapshot_id, program=make_program(shift=0.1))
	with pytest.raises(IdempotencyConflictError):
		svc.submit_rollout(idempotency_key="k", snapshot_id=root.snapshot_id, program=make_program(), max_frames=5)
	with pytest.raises(IdempotencyConflictError):
		new_root(svc, key="k")
	assert svc.get_rollout(job).state is JobState.QUEUED
	assert len(svc.ledger.incomplete_jobs()) == 1


def test_root_and_fork_creation_are_idempotent(rig):
	svc = rig.service()
	root = new_root(svc)
	assert new_root(svc) == root
	fork = svc.fork_snapshot(idempotency_key="f", snapshot_id=root.snapshot_id, branch_id="alt")
	assert svc.fork_snapshot(idempotency_key="f", snapshot_id=root.snapshot_id, branch_id="alt") == fork
	assert fork.parent_snapshot_id == root.snapshot_id and fork.branch_id == "alt" and fork.blobs == root.blobs
	assert sorted(svc.store.list_snapshot_ids()) == sorted([root.snapshot_id, fork.snapshot_id])
	assert diff_session_states(svc.load_state(fork.snapshot_id), svc.load_state(root.snapshot_id)) == []


# -- crash recovery ---------------------------------------------------------------------------


def test_crash_after_snapshot_commit_is_adopted_without_running_again(rig):
	"""Result written, ledger not yet: recovery reconciles manifest and ledger instead of re-executing."""
	svc = rig.service()
	root = new_root(svc)
	prog = make_program()
	job = svc.submit_rollout(idempotency_key="k", snapshot_id=root.snapshot_id, program=prog)
	with pytest.raises(SimulatedCrash):
		RolloutWorker(svc, "w1").run(job, hooks=FaultHooks(after_commit=crash))
	assert svc.get_rollout(job).state is JobState.RUNNING

	restarted = rig.service()
	report = restarted.recover()
	assert report.adopted == [job] and report.needs_rerun == [] and report.orphan_snapshots == []
	status = restarted.get_rollout(job)
	assert status.state is JobState.SUCCEEDED and status.result["recovered"] is True
	assert running_events(restarted, job) == 1
	assert diff_session_states(restarted.load_state(status.output_snapshot_id), straight(prog)) == []
	assert RolloutWorker(restarted, "w2").run(job) == status  # nothing left to do
	assert running_events(restarted, job) == 1
	again = restarted.recover()
	assert (again.adopted, again.needs_rerun, again.orphan_snapshots) == ([], [], [])


def test_crash_before_snapshot_commit_is_rerun_under_a_new_fencing_token(rig):
	svc = rig.service()
	root = new_root(svc)
	prog = make_program()
	job = svc.submit_rollout(idempotency_key="k", snapshot_id=root.snapshot_id, program=prog)
	with pytest.raises(SimulatedCrash):
		RolloutWorker(svc, "w1").run(job, hooks=FaultHooks(before_commit=crash))
	first_token = svc.ledger.latest_event(job).fencing_token

	restarted = rig.service()
	report = restarted.recover()
	assert report.adopted == [] and report.needs_rerun == [job]
	assert restarted.get_rollout(job).state is JobState.NEEDS_RECOVERY
	status = RolloutWorker(restarted, "w2").run(job)
	assert status.state is JobState.SUCCEEDED and status.attempts == 2
	assert restarted.ledger.latest_event(job).fencing_token > first_token
	assert diff_session_states(restarted.load_state(status.output_snapshot_id), straight(prog)) == []
	# The dead attempt left no committed snapshot behind: only the root and the one real output exist.
	assert sorted(restarted.store.list_snapshot_ids()) == sorted([root.snapshot_id, status.output_snapshot_id])


def test_crash_while_creating_the_root_snapshot_is_recovered_on_retry(rig):
	svc = rig.service()
	with pytest.raises(SimulatedCrash):
		svc.create_root_snapshot(
			idempotency_key="root",
			run_id="run",
			episode_id="ep0",
			branch_id="main",
			scene_spec=SPEC,
			contract=CONTRACT,
			hooks=FaultHooks(before_commit=crash),
		)
	restarted = rig.service()
	with pytest.raises(LedgerError, match="recover"):
		new_root(restarted)  # still marked RUNNING: the caller must reconcile first, not start a second copy
	restarted.recover()
	root = new_root(restarted)
	assert restarted.store.list_snapshot_ids() == [root.snapshot_id]


# -- fencing ----------------------------------------------------------------------------------


def test_stale_worker_fenced_before_commit_writes_nothing(rig):
	svc = rig.service()
	root = new_root(svc)
	prog = make_program()
	job = svc.submit_rollout(idempotency_key="k", snapshot_id=root.snapshot_id, program=prog)
	winner: dict = {}

	def take_over() -> None:  # the first worker stalls here; it is presumed dead and replaced
		svc.recover()
		winner["status"] = RolloutWorker(svc, "w2").run(job)

	with pytest.raises(StaleWorkerError):
		RolloutWorker(svc, "w1").run(job, hooks=FaultHooks(before_commit=take_over))
	status = svc.get_rollout(job)
	assert status == winner["status"] and status.state is JobState.SUCCEEDED
	events = svc.ledger.events(job)
	assert [e.state for e in events].count(JobState.SUCCEEDED) == 1
	assert events[-1].worker_id == "w2"
	assert sorted(svc.store.list_snapshot_ids()) == sorted([root.snapshot_id, status.output_snapshot_id])
	assert list(svc.store.staging_dir.iterdir()) == []


def test_stale_worker_that_already_committed_cannot_publish_its_snapshot(rig):
	svc = rig.service()
	root = new_root(svc)
	job = svc.submit_rollout(idempotency_key="k", snapshot_id=root.snapshot_id, program=make_program())
	winner: dict = {}

	def take_over() -> None:
		# Fence the stalled worker and rerun elsewhere without adopting its output.
		svc.ledger.append_event(job, JobState.NEEDS_RECOVERY, payload={"reason": "presumed dead"})
		winner["status"] = RolloutWorker(svc, "w2").run(job)

	with pytest.raises(LedgerError):
		RolloutWorker(svc, "w1").run(job, hooks=FaultHooks(after_commit=take_over))
	status = svc.get_rollout(job)
	assert status == winner["status"] and status.state is JobState.SUCCEEDED
	assert [e.state for e in svc.ledger.events(job)].count(JobState.SUCCEEDED) == 1
	orphans = svc.recover().orphan_snapshots
	assert len(orphans) == 1 and orphans[0] not in (root.snapshot_id, status.output_snapshot_id)
	stale = svc.store.load_manifest(orphans[0])
	assert stale.produced_by["worker_id"] == "w1"
	with pytest.raises(SnapshotNotFoundError):
		svc.submit_rollout(idempotency_key="next", snapshot_id=orphans[0])  # unpublished snapshots are not usable


# -- cancellation -----------------------------------------------------------------------------


def test_cancel_queued_job_never_runs(rig):
	svc = rig.service()
	root = new_root(svc)
	job = svc.submit_rollout(idempotency_key="k", snapshot_id=root.snapshot_id, program=make_program())
	assert svc.cancel_rollout(job, reason="superseded") is JobState.CANCELLED
	status = RolloutWorker(svc, "w1").run(job)
	assert status.state is JobState.CANCELLED and status.output_snapshot_id is None
	assert running_events(svc, job) == 0
	assert svc.cancel_rollout(job) is JobState.CANCELLED


def test_cancel_running_job_stops_at_a_chunk_boundary_and_accounts_for_the_work_done(rig):
	svc = rig.service()
	root = new_root(svc)
	prog = make_program()
	job = svc.submit_rollout(idempotency_key="k", snapshot_id=root.snapshot_id, program=prog)

	def after_chunk(frames_done: int) -> None:
		if frames_done == 20:
			svc.cancel_rollout(job, reason="budget")

	status = RolloutWorker(svc, "w1", chunk_frames=10).run(job, hooks=FaultHooks(after_chunk=after_chunk))
	assert status.state is JobState.CANCELLED
	assert status.output_snapshot_id is None and status.result is None
	last = svc.ledger.latest_event(job)
	assert last.payload["frames_executed"] == 20
	assert svc.ledger.consumed("run")["sim_time_s"] == pytest.approx(20 / 30)
	assert svc.store.list_snapshot_ids() == [root.snapshot_id]


def test_cancel_that_arrives_after_the_work_finished_records_the_real_outcome(rig):
	svc = rig.service()
	root = new_root(svc)
	job = svc.submit_rollout(idempotency_key="k", snapshot_id=root.snapshot_id, program=make_program())
	hooks = FaultHooks(after_commit=lambda: svc.cancel_rollout(job))
	status = RolloutWorker(svc, "w1").run(job, hooks=hooks)
	states = [e.state for e in svc.ledger.events(job)]
	assert states == [JobState.QUEUED, JobState.RUNNING, JobState.CANCEL_REQUESTED, JobState.SUCCEEDED]
	assert status.state is JobState.SUCCEEDED and status.output_snapshot_id is not None


# -- refusal of mismatched or damaged snapshots ------------------------------------------------


@pytest.mark.parametrize("field", ["code_sha256", "assets_sha256", "config_sha256", "engine_version", "numpy_version"])
def test_snapshot_from_a_different_code_asset_or_engine_version_is_refused(rig, field):
	svc = rig.service()
	root = new_root(svc)
	job = svc.submit_rollout(idempotency_key="k", snapshot_id=root.snapshot_id, program=make_program())

	changed = rig.service(identity=dataclasses.replace(IDENTITY, **{field: "changed"}))
	with pytest.raises(IdentityMismatchError, match=field):
		changed.submit_rollout(idempotency_key="k2", snapshot_id=root.snapshot_id, program=make_program())
	status = RolloutWorker(changed, "w1").run(job)
	assert status.state is JobState.FAILED and status.error["reason"] == "identity_mismatch"
	assert field in status.error["detail"]
	assert status.output_snapshot_id is None
	assert changed.store.list_snapshot_ids() == [root.snapshot_id]


def test_model_that_drifted_from_the_scene_specification_is_refused(rig):
	svc = rig.service()
	root = new_root(svc)
	job = svc.submit_rollout(idempotency_key="k", snapshot_id=root.snapshot_id, program=make_program())
	drifted = rig.service(adapter=ToyAdapter(stiffness_drift=0.5))
	status = RolloutWorker(drifted, "w1").run(job)
	assert status.state is JobState.FAILED and status.error["reason"] == "identity_mismatch"
	assert "model_fingerprint" in status.error["detail"]


@pytest.mark.parametrize("target", ["blob_flip", "blob_truncate", "blob_delete", "manifest_edit", "manifest_delete"])
def test_damaged_snapshot_is_refused_at_submit_and_at_execution(rig, target):
	svc = rig.service()
	root = new_root(svc)
	job = svc.submit_rollout(idempotency_key="k", snapshot_id=root.snapshot_id, program=make_program())
	blob = svc.store.blobs_dir / root.blobs["arrays/physics"].sha256
	manifest = svc.store.manifests_dir / f"{root.snapshot_id}.json"
	if target == "blob_flip":
		data = bytearray(blob.read_bytes())
		data[-1] ^= 0x01
		blob.write_bytes(bytes(data))
	elif target == "blob_truncate":
		blob.write_bytes(blob.read_bytes()[:-8])
	elif target == "blob_delete":
		blob.unlink()
	elif target == "manifest_edit":
		doc = json.loads(manifest.read_text("utf-8"))
		doc["control_step"] = 5
		manifest.write_text(json.dumps(doc), "utf-8")
	else:
		manifest.unlink()
	expected = SnapshotNotFoundError if target == "manifest_delete" else SnapshotCorruptError
	with pytest.raises(expected):
		svc.submit_rollout(idempotency_key="k2", snapshot_id=root.snapshot_id, program=make_program())
	status = RolloutWorker(svc, "w1").run(job)
	assert status.state is JobState.FAILED
	assert status.error["reason"] == ("snapshot_missing" if target == "manifest_delete" else "snapshot_corrupt")
	assert status.output_snapshot_id is None


def test_unknown_and_uncommitted_snapshots_are_refused(rig):
	svc = rig.service()
	root = new_root(svc)
	with pytest.raises(SnapshotNotFoundError):
		svc.submit_rollout(idempotency_key="k", snapshot_id="0" * 32, program=make_program())
	# A manifest that reached the store but was never published through the ledger is not a usable snapshot.
	stray = svc.store.fork(root.snapshot_id, branch_id="stray")
	with pytest.raises(SnapshotNotFoundError, match="not published"):
		svc.submit_rollout(idempotency_key="k", snapshot_id=stray.snapshot_id, program=make_program())
	assert svc.recover().orphan_snapshots == [stray.snapshot_id]


def test_engine_failure_is_recorded_as_failed_with_evidence_and_no_snapshot(rig):
	svc = rig.service()
	spec = toy_scene_spec(seed=11, fail_at_step=17)
	root = new_root(svc, spec=spec)
	job = svc.submit_rollout(idempotency_key="k", snapshot_id=root.snapshot_id, program=make_program())
	status = RolloutWorker(svc, "w1", chunk_frames=5).run(job)
	assert status.state is JobState.FAILED
	assert status.error["reason"] == "simulation_error" and "step 17" in status.error["detail"]
	assert svc.ledger.latest_event(job).payload["frames_executed"] == 17
	assert svc.store.list_snapshot_ids() == [root.snapshot_id]


# -- budgets and the stop latch ----------------------------------------------------------------


def test_forking_a_new_branch_does_not_reset_the_run_budget(rig):
	svc = rig.service(budget=RunBudget(max_rollouts=2))
	root = new_root(svc)
	for key in ("a", "b"):
		RolloutWorker(svc, "w").run(
			svc.submit_rollout(idempotency_key=key, snapshot_id=root.snapshot_id, program=make_program())
		)
	fork = svc.fork_snapshot(idempotency_key="f", snapshot_id=root.snapshot_id, branch_id="alt")
	with pytest.raises(BudgetExceededError):
		svc.submit_rollout(idempotency_key="c", snapshot_id=fork.snapshot_id, program=make_program())
	# Retrying an already-accepted request is still answered, on a restarted service too.
	assert (
		svc.get_rollout(
			svc.submit_rollout(idempotency_key="a", snapshot_id=root.snapshot_id, program=make_program())
		).state
		is JobState.SUCCEEDED
	)
	restarted = rig.service(budget=RunBudget(max_rollouts=2))
	with pytest.raises(BudgetExceededError):
		restarted.submit_rollout(idempotency_key="c", snapshot_id=fork.snapshot_id, program=make_program())
	assert restarted.ledger.incomplete_jobs() == []


def test_simulated_time_budget_counts_cancelled_and_failed_work_across_branches(rig):
	svc = rig.service(budget=RunBudget(max_sim_time_s=3.0))
	root = new_root(svc)
	prog = make_program(n=60)  # 2 s each
	RolloutWorker(svc, "w").run(svc.submit_rollout(idempotency_key="a", snapshot_id=root.snapshot_id, program=prog))
	fork = svc.fork_snapshot(idempotency_key="f", snapshot_id=root.snapshot_id, branch_id="alt")
	with pytest.raises(BudgetExceededError):  # 2 s used + 2 s requested > 3 s
		svc.submit_rollout(idempotency_key="b", snapshot_id=fork.snapshot_id, program=prog)
	short = svc.submit_rollout(idempotency_key="c", snapshot_id=fork.snapshot_id, program=prog, max_frames=30)
	RolloutWorker(svc, "w").run(short)
	assert svc.ledger.consumed("run")["sim_time_s"] == pytest.approx(3.0)
	with pytest.raises(BudgetExceededError):
		svc.submit_rollout(idempotency_key="d", snapshot_id=fork.snapshot_id, program=prog, max_frames=1)


def test_single_rollout_longer_than_the_per_candidate_limit_is_rejected(rig):
	svc = rig.service(max_rollout_sim_time_s=1.0)
	root = new_root(svc)
	with pytest.raises(BudgetExceededError):
		svc.submit_rollout(idempotency_key="k", snapshot_id=root.snapshot_id, program=make_program(n=31))
	svc.submit_rollout(idempotency_key="k", snapshot_id=root.snapshot_id, program=make_program(n=30))


def test_stop_latch_cancels_work_refuses_new_work_and_survives_restart(rig):
	svc = rig.service()
	root = new_root(svc)
	queued = svc.submit_rollout(idempotency_key="q", snapshot_id=root.snapshot_id, program=make_program())
	running = svc.submit_rollout(idempotency_key="r", snapshot_id=root.snapshot_id, program=make_program(shift=1.0))

	def after_chunk(frames_done: int) -> None:
		if frames_done == 10:
			svc.stop("run", reason="operator pressed stop", actor="linan")

	status = RolloutWorker(svc, "w1", chunk_frames=10).run(running, hooks=FaultHooks(after_chunk=after_chunk))
	assert status.state is JobState.CANCELLED
	assert svc.get_rollout(queued).state is JobState.CANCELLED

	restarted = rig.service()
	restarted.recover()  # recovery must not clear the latch
	assert restarted.ledger.stop_latched("run")
	with pytest.raises(StopLatchedError):
		restarted.submit_rollout(idempotency_key="n", snapshot_id=root.snapshot_id, program=make_program())
	with pytest.raises(StopLatchedError):
		restarted.fork_snapshot(idempotency_key="f", snapshot_id=root.snapshot_id, branch_id="alt")
	restarted.clear_stop("run", reason="inspected, safe to continue", actor="linan")
	job = restarted.submit_rollout(idempotency_key="n", snapshot_id=root.snapshot_id, program=make_program())
	assert RolloutWorker(restarted, "w2").run(job).state is JobState.SUCCEEDED


def test_stop_latch_blocks_a_queued_job_from_starting(rig):
	svc = rig.service()
	root = new_root(svc)
	job = svc.submit_rollout(idempotency_key="k", snapshot_id=root.snapshot_id, program=make_program())
	svc.ledger.latch_stop("run", reason="manual", actor="linan")  # latched without cancelling the queue
	with pytest.raises(StopLatchedError):
		RolloutWorker(svc, "w1").run(job)
	assert svc.get_rollout(job).state is JobState.QUEUED and running_events(svc, job) == 0


def test_queued_rollouts_reserve_their_simulated_time(rig):
	svc = rig.service(budget=RunBudget(max_sim_time_s=3.0))
	root = new_root(svc)
	svc.submit_rollout(idempotency_key="a", snapshot_id=root.snapshot_id, program=make_program(n=60))  # 2 s, not run
	with pytest.raises(BudgetExceededError):
		svc.submit_rollout(idempotency_key="b", snapshot_id=root.snapshot_id, program=make_program(n=60, shift=1.0))
	assert svc.ledger.consumed("run")["sim_time_s"] == 0.0


def test_fenced_worker_stops_at_the_next_chunk_boundary_instead_of_finishing(rig):
	svc = rig.service()
	root = new_root(svc)
	job = svc.submit_rollout(idempotency_key="k", snapshot_id=root.snapshot_id, program=make_program())
	seen: list[int] = []

	def after_chunk(frames_done: int) -> None:
		seen.append(frames_done)
		if frames_done == 10:
			svc.recover([job])  # the coordinator gives up on this worker mid-run

	with pytest.raises(StaleWorkerError):
		RolloutWorker(svc, "w1", chunk_frames=10).run(job, hooks=FaultHooks(after_chunk=after_chunk))
	assert seen == [10], "a fenced worker must not keep simulating"
	assert svc.get_rollout(job).state is JobState.NEEDS_RECOVERY
	assert svc.store.list_snapshot_ids() == [root.snapshot_id]


def test_recovery_fences_a_stalled_worker_so_it_cannot_commit_afterwards(rig):
	svc = rig.service()
	root = new_root(svc)
	job = svc.submit_rollout(idempotency_key="k", snapshot_id=root.snapshot_id, program=make_program())
	with pytest.raises(StaleWorkerError):
		RolloutWorker(svc, "w1").run(job, hooks=FaultHooks(before_commit=lambda: svc.recover()))
	assert svc.get_rollout(job).state is JobState.NEEDS_RECOVERY
	assert svc.store.list_snapshot_ids() == [root.snapshot_id], "the stalled worker must not leave a manifest"
	assert RolloutWorker(svc, "w2").run(job).state is JobState.SUCCEEDED


def test_recovery_does_not_adopt_a_manifest_this_job_did_not_produce(rig):
	svc = rig.service()
	root = new_root(svc)
	prog = make_program()
	job = svc.submit_rollout(idempotency_key="k", snapshot_id=root.snapshot_id, program=prog)
	with pytest.raises(SimulatedCrash):
		RolloutWorker(svc, "w1").run(job, hooks=FaultHooks(before_commit=crash))
	announced = svc.ledger.latest_event(job).payload["output_snapshot_id"]
	# Something else ends up at the announced id (wrong producer): it must never be taken for this job's result.
	svc.store.fork(root.snapshot_id, branch_id="stray", new_snapshot_id=announced)

	restarted = rig.service()
	report = restarted.recover()
	assert report.adopted == [] and report.needs_rerun == [job] and report.orphan_snapshots == [announced]
	status = RolloutWorker(restarted, "w2").run(job)
	assert status.state is JobState.SUCCEEDED and status.output_snapshot_id != announced
	assert diff_session_states(restarted.load_state(status.output_snapshot_id), straight(prog)) == []
