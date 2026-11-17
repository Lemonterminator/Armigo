"""Execution service: idempotent snapshot and rollout operations on top of the snapshot store and the ledger.

Commit protocol for every operation that produces a snapshot (create, fork, rollout):

1. the worker takes a lease (a new fencing token) and records ``RUNNING`` with the snapshot id it intends to write;
2. it works on a private session, then writes blobs (content-addressed, verified after writing);
3. it re-checks its lease and commits the manifest under the pre-announced id (the atomic commit point of the files);
4. it appends ``SUCCEEDED`` to the ledger. Only this event *publishes* the snapshot: a manifest that no ``SUCCEEDED``
   event names is ignored by every API here, so a later checkpoint may only ever reference ledger-confirmed results.

After a crash :meth:`ExecutionService.recover` reconciles the two stores: a job stuck in ``RUNNING`` whose announced
manifest exists, verifies, and was produced under the job's last fencing token is adopted; otherwise the job is marked
``NEEDS_RECOVERY`` and must be re-executed under a new token. Budgets are derived from the append-only ledger per run,
so forking a branch or rolling back never resets them. The stop latch is only ever cleared by an explicit call.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .errors import (
	BudgetExceededError,
	IdentityMismatchError,
	LedgerError,
	SnapshotCorruptError,
	SnapshotError,
	SnapshotNotFoundError,
	StaleWorkerError,
)
from .ledger import TERMINAL_STATES, JobRecord, JobState, Lease, OperationLedger, job_resource
from .simulator import ControlProgram, SessionState, SimulatorAdapter, load_session_state, save_session_state
from .snapshot import BlobRef, ExperimentIdentity, SnapshotManifest, SnapshotStore

OP_CREATE = "create_snapshot"
OP_FORK = "fork_snapshot"
OP_ROLLOUT = "rollout"
ROLLOUT_RESULT_SCHEMA = "rollout-result/v1"
COORDINATOR = "coordinator"


@dataclass(frozen=True)
class RunBudget:
	"""Limits per run, counted over every branch; ``None`` means unlimited."""

	max_rollouts: int | None = None
	max_sim_time_s: float | None = None


@dataclass(frozen=True)
class FaultHooks:
	"""Test seams called by a worker; production code leaves them unset."""

	after_chunk: Callable[[int], None] | None = None
	before_commit: Callable[[], None] | None = None
	after_commit: Callable[[], None] | None = None


@dataclass(frozen=True)
class RolloutStatus:
	"""What the ledger says about one operation."""

	job_id: str
	op_type: str
	state: JobState
	run_id: str
	branch_id: str
	input_snapshot_id: str | None
	output_snapshot_id: str | None
	result: dict[str, Any] | None
	error: dict[str, Any] | None
	attempts: int


@dataclass
class RecoveryReport:
	"""Outcome of :meth:`ExecutionService.recover`."""

	adopted: list[str] = field(default_factory=list)
	needs_rerun: list[str] = field(default_factory=list)
	queued: list[str] = field(default_factory=list)
	orphan_snapshots: list[str] = field(default_factory=list)


class _Failed(Exception):
	"""Internal: the operation failed for a recordable reason."""

	def __init__(self, reason: str, detail: str) -> None:
		super().__init__(detail)
		self.reason = reason
		self.detail = detail


class ExecutionService:
	"""Coordinator-side API. One coordinator per ledger; workers are separate :class:`RolloutWorker` objects."""

	def __init__(
		self,
		store: SnapshotStore,
		ledger: OperationLedger,
		adapter: SimulatorAdapter,
		identity: ExperimentIdentity,
		*,
		budget: RunBudget | None = None,
		max_rollout_sim_time_s: float = 60.0,
	) -> None:
		self.store = store
		self.ledger = ledger
		self.adapter = adapter
		self.identity = identity
		self.budget = budget or RunBudget()
		self.max_rollout_sim_time_s = float(max_rollout_sim_time_s)

	# -- snapshots -------------------------------------------------------------------------------
	def published_snapshots(self) -> dict[str, str]:
		"""Snapshot id -> job id for every snapshot a ``SUCCEEDED`` ledger event names."""
		out = {}
		for event in self.ledger.events_in_state(JobState.SUCCEEDED):
			sid = event.payload.get("output_snapshot_id")
			if isinstance(sid, str):
				out[sid] = event.job_id
		return out

	def usable_snapshot(self, snapshot_id: str) -> SnapshotManifest:
		"""Verify files, publication and identity; raise instead of returning a snapshot that cannot be trusted."""
		manifest = self.store.verify(snapshot_id)
		if snapshot_id not in self.published_snapshots():
			raise SnapshotNotFoundError(f"snapshot {snapshot_id} is not published in the ledger")
		self.store.check_identity(manifest, self.identity)
		return manifest

	def load_state(self, snapshot_id: str) -> SessionState:
		return load_session_state(self.store, self.usable_snapshot(snapshot_id).blobs)

	def create_root_snapshot(
		self,
		*,
		idempotency_key: str,
		run_id: str,
		episode_id: str,
		branch_id: str,
		scene_spec: Mapping[str, Any],
		contract: Mapping[str, Any],
		hooks: FaultHooks | None = None,
	) -> SnapshotManifest:
		"""Build the scene, capture its initial state and publish it as the run's root snapshot."""
		params = {"scene_spec": dict(scene_spec), "contract": dict(contract), "episode_id": episode_id}
		record, _ = self.ledger.register(
			idempotency_key=idempotency_key, op_type=OP_CREATE, params=params, run_id=run_id, branch_id=branch_id
		)
		return self._run_inline(record, hooks)

	def fork_snapshot(
		self, *, idempotency_key: str, snapshot_id: str, branch_id: str, hooks: FaultHooks | None = None
	) -> SnapshotManifest:
		"""Publish a child of ``snapshot_id`` on a new branch. The parent and the run's budget are untouched."""
		source = self.usable_snapshot(snapshot_id)
		record, _ = self.ledger.register(
			idempotency_key=idempotency_key,
			op_type=OP_FORK,
			params={"input_snapshot_id": snapshot_id},
			run_id=source.run_id,
			branch_id=branch_id,
		)
		return self._run_inline(record, hooks)

	def _run_inline(self, record: JobRecord, hooks: FaultHooks | None) -> SnapshotManifest:
		state = self.ledger.state(record.job_id)
		if state in (JobState.RUNNING, JobState.CANCEL_REQUESTED):
			raise LedgerError(f"{record.op_type} {record.job_id} was left running; call recover() before retrying")
		if state in (JobState.QUEUED, JobState.NEEDS_RECOVERY):
			RolloutWorker(self, COORDINATOR).run(record.job_id, hooks=hooks)
		status = self.get_rollout(record.job_id)
		if status.state is not JobState.SUCCEEDED:
			raise LedgerError(f"{record.op_type} {record.job_id} ended {status.state}: {status.error}")
		return self.store.verify(status.output_snapshot_id)

	# -- rollouts --------------------------------------------------------------------------------
	def submit_rollout(
		self,
		*,
		idempotency_key: str,
		snapshot_id: str,
		program: ControlProgram | None = None,
		max_frames: int | None = None,
	) -> str:
		"""Queue a rollout from ``snapshot_id`` and return its job id.

		``program`` replaces the snapshot's control program; ``None`` continues the stored one. The same key with
		the same request returns the existing job; the same key with a different request is rejected.
		"""
		if max_frames is not None and max_frames <= 0:
			raise ValueError("max_frames must be positive")
		manifest = self.usable_snapshot(snapshot_id)
		if program is not None:
			blob: dict[str, Any] | None = self.store.put_arrays(program.to_arrays()).to_dict()
			plan_sha, available, fps = program.sha256(), len(program), program.fps
		else:
			state = load_session_state(self.store, manifest.blobs)
			try:
				stored = ControlProgram.from_arrays(state.arrays["program"])
				plan_sha, cursor = str(state.meta["program_sha256"]), int(state.meta["program_cursor"])
			except (KeyError, TypeError, ValueError) as exc:
				raise SnapshotCorruptError(f"snapshot {snapshot_id} has no usable control program: {exc}") from exc
			blob, available, fps = None, len(stored) - cursor, stored.fps
		requested = available if max_frames is None else min(max_frames, available)
		params = {
			"input_snapshot_id": snapshot_id,
			"program_blob": blob,
			"plan_sha256": plan_sha,
			"max_frames": max_frames,
			"requested_frames": requested,
			"fps": fps,
			"scene_sha256": manifest.scene_sha256,
			"contract_sha256": manifest.contract_sha256,
		}
		if self.ledger.job_by_key(idempotency_key) is None:
			self._check_budget(manifest.run_id, requested / float(fps))
		record, _ = self.ledger.register(
			idempotency_key=idempotency_key,
			op_type=OP_ROLLOUT,
			params=params,
			run_id=manifest.run_id,
			branch_id=manifest.branch_id,
		)
		return record.job_id

	def _check_budget(self, run_id: str, requested_s: float) -> None:
		if requested_s > self.max_rollout_sim_time_s:
			raise BudgetExceededError(
				f"rollout asks for {requested_s:.3f} s of simulated time; the limit is {self.max_rollout_sim_time_s} s"
			)
		rollouts = [job for job in self.ledger.jobs(run_id) if job.op_type == OP_ROLLOUT]
		if self.budget.max_rollouts is not None and len(rollouts) >= self.budget.max_rollouts:
			raise BudgetExceededError(f"run {run_id!r} has used all {self.budget.max_rollouts} rollouts")
		if self.budget.max_sim_time_s is not None:
			pending = {job.job_id for job in self.ledger.incomplete_jobs(run_id)}
			reserved = sum(
				j.params["requested_frames"] / float(j.params["fps"]) for j in rollouts if j.job_id in pending
			)
			used = float(self.ledger.consumed(run_id)["sim_time_s"]) + reserved
			if used + requested_s > self.budget.max_sim_time_s + 1e-9:
				raise BudgetExceededError(
					f"run {run_id!r} has used or reserved {used:.3f} s of {self.budget.max_sim_time_s} s simulated time"
				)

	def get_rollout(self, job_id: str) -> RolloutStatus:
		record = self.ledger.job(job_id)
		events = self.ledger.events(job_id)
		latest = events[-1]
		done = latest.state is JobState.SUCCEEDED
		return RolloutStatus(
			job_id=job_id,
			op_type=record.op_type,
			state=latest.state,
			run_id=record.run_id,
			branch_id=record.branch_id,
			input_snapshot_id=record.params.get("input_snapshot_id"),
			output_snapshot_id=latest.payload.get("output_snapshot_id") if done else None,
			result=latest.payload.get("result") if done else None,
			error=latest.payload.get("error") if latest.state is JobState.FAILED else None,
			attempts=sum(e.state is JobState.RUNNING for e in events),
		)

	def cancel_rollout(self, job_id: str, *, reason: str = "") -> JobState:
		"""Request cancellation; a queued job is cancelled at once, a running one at its next chunk boundary."""
		return self.ledger.request_cancel(job_id, reason=reason)

	# -- stop latch ------------------------------------------------------------------------------
	def stop(self, run_id: str, *, reason: str, actor: str) -> None:
		"""Latch the run stopped and cancel its outstanding work. Only :meth:`clear_stop` releases the latch."""
		self.ledger.latch_stop(run_id, reason=reason, actor=actor)
		for job in self.ledger.incomplete_jobs(run_id):
			self.ledger.request_cancel(job.job_id, reason=f"stop: {reason}")

	def clear_stop(self, run_id: str, *, reason: str, actor: str) -> None:
		self.ledger.clear_stop(run_id, reason=reason, actor=actor)

	# -- recovery --------------------------------------------------------------------------------
	def recover(self, job_ids: Iterable[str] | None = None) -> RecoveryReport:
		"""Reconcile the ledger with the snapshot store after a crash.

		Every job found ``RUNNING`` (or cancel-requested) is treated as abandoned, so call this at coordinator start
		or pass ``job_ids`` for workers known to be dead. Abandoned workers are fenced first; a snapshot they fully
		committed under their last token is adopted, anything else is queued for re-execution. Never clears a stop
		latch and never deletes files.
		"""
		wanted = None if job_ids is None else set(job_ids)
		report = RecoveryReport()
		for record in self.ledger.incomplete_jobs():
			job_id = record.job_id
			if wanted is not None and job_id not in wanted:
				continue
			state = self.ledger.state(job_id)
			if state is JobState.QUEUED:
				report.queued.append(job_id)
				continue
			if state is JobState.NEEDS_RECOVERY:
				report.needs_rerun.append(job_id)
				continue
			attempt = [e for e in self.ledger.events(job_id) if e.state is JobState.RUNNING][-1]
			self.ledger.acquire_lease(job_resource(job_id), f"{COORDINATOR}-recovery")  # fences the old worker
			self.ledger.append_event(
				job_id,
				JobState.NEEDS_RECOVERY,
				payload={"reason": "worker presumed dead", "superseded_token": attempt.fencing_token},
			)
			manifest = self._committed_output(job_id, attempt.payload.get("output_snapshot_id"), attempt.fencing_token)
			if manifest is None:
				report.needs_rerun.append(job_id)
				continue
			result = dict(manifest.produced_by["result"])
			result["recovered"] = True
			self.ledger.append_event(
				job_id,
				JobState.SUCCEEDED,
				payload={
					"output_snapshot_id": manifest.snapshot_id,
					"result": result,
					"frames_executed": result["frames_executed"],
					"sim_time_consumed_s": result["sim_time_consumed_s"],
				},
			)
			report.adopted.append(job_id)
		published = self.published_snapshots()
		report.orphan_snapshots = [sid for sid in self.store.list_snapshot_ids() if sid not in published]
		return report

	def _committed_output(self, job_id: str, snapshot_id: Any, token: int | None) -> SnapshotManifest | None:
		if not isinstance(snapshot_id, str):
			return None
		try:
			manifest = self.store.verify(snapshot_id)
		except SnapshotError:
			return None
		by = manifest.produced_by or {}
		if by.get("job_id") != job_id or by.get("fencing_token") != token or not isinstance(by.get("result"), dict):
			return None
		return manifest


class RolloutWorker:
	"""Executes one ledger operation at a time under a lease. Each run uses its own simulator session."""

	def __init__(self, service: ExecutionService, worker_id: str, *, chunk_frames: int = 15) -> None:
		if chunk_frames <= 0:
			raise ValueError("chunk_frames must be positive")
		self.service = service
		self.worker_id = worker_id
		self.chunk_frames = chunk_frames

	def run(self, job_id: str, *, hooks: FaultHooks | None = None) -> RolloutStatus:
		"""Execute ``job_id`` if it is waiting; return its status. A finished job is never executed again."""
		svc = self.service
		state = svc.ledger.state(job_id)
		if state in TERMINAL_STATES:
			return svc.get_rollout(job_id)
		if state in (JobState.RUNNING, JobState.CANCEL_REQUESTED):
			raise LedgerError(f"job {job_id} is already running; call recover() if its worker died")
		record = svc.ledger.job(job_id)
		lease = svc.ledger.acquire_lease(job_resource(job_id), self.worker_id)
		output_id = uuid.uuid4().hex
		svc.ledger.append_event(job_id, JobState.RUNNING, lease=lease, payload={"output_snapshot_id": output_id})
		started = time.perf_counter()
		progress = {"frames_executed": 0, "fps": 1}
		try:
			body = {OP_CREATE: self._create, OP_FORK: self._fork, OP_ROLLOUT: self._rollout}.get(record.op_type)
			if body is None:
				raise _Failed("unknown_operation", f"no executor for operation type {record.op_type!r}")
			outcome = body(record, lease, hooks or FaultHooks(), progress)
		except _Failed as failure:
			svc.ledger.append_event(
				job_id,
				JobState.FAILED,
				lease=lease,
				payload={
					"error": {"reason": failure.reason, "detail": failure.detail},
					**self._accounting(progress, started),
				},
			)
			return svc.get_rollout(job_id)
		if outcome is None:  # cancelled at a chunk boundary
			svc.ledger.append_event(
				job_id, JobState.CANCELLED, lease=lease, payload=self._accounting(progress, started)
			)
			return svc.get_rollout(job_id)
		fields, blobs, result = outcome
		result = {**result, "sim_time_consumed_s": progress["frames_executed"] / float(progress["fps"])}
		if hooks and hooks.before_commit:
			hooks.before_commit()
		svc.ledger.assert_current(lease)  # a fenced worker must not write a manifest
		manifest = svc.store.commit(
			snapshot_id=output_id,
			identity=svc.identity,
			blobs=blobs,
			produced_by={
				"job_id": job_id,
				"op_type": record.op_type,
				"worker_id": self.worker_id,
				"fencing_token": lease.fencing_token,
				"result": result,
			},
			**fields,
		)
		if hooks and hooks.after_commit:
			hooks.after_commit()
		svc.ledger.append_event(
			job_id,
			JobState.SUCCEEDED,
			lease=lease,
			payload={
				"output_snapshot_id": manifest.snapshot_id,
				"result": result,
				**self._accounting(progress, started),
			},
		)
		return svc.get_rollout(job_id)

	@staticmethod
	def _accounting(progress: Mapping[str, int], started: float) -> dict[str, Any]:
		return {
			"frames_executed": progress["frames_executed"],
			"sim_time_consumed_s": progress["frames_executed"] / float(progress["fps"]),
			"wall_time_s": time.perf_counter() - started,
		}

	# -- operation bodies: return (manifest fields, blobs, result) or None when cancelled --------
	def _create(self, record: JobRecord, lease: Lease, hooks: FaultHooks, progress: dict) -> tuple:
		svc, params = self.service, record.params
		try:
			session = svc.adapter.open(params["scene_spec"])
		except Exception as exc:
			raise _Failed("scene_error", f"{type(exc).__name__}: {exc}") from exc
		try:
			state = session.capture()
			fingerprint = session.model_fingerprint()
		finally:
			session.close()
		fields = {
			"run_id": record.run_id,
			"episode_id": params["episode_id"],
			"branch_id": record.branch_id,
			"control_step": state.control_step,
			"sim_time_s": state.sim_time_s,
			"model_fingerprint": fingerprint,
			"contract": params["contract"],
			"scene_spec": params["scene_spec"],
			"parent_snapshot_id": None,
		}
		return fields, save_session_state(svc.store, state), {"frames_executed": 0}

	def _load_input(self, snapshot_id: str) -> SnapshotManifest:
		svc = self.service
		try:
			manifest = svc.store.verify(snapshot_id)
			if snapshot_id not in svc.published_snapshots():
				raise SnapshotNotFoundError(f"snapshot {snapshot_id} is not published in the ledger")
			svc.store.check_identity(manifest, svc.identity)
		except SnapshotNotFoundError as exc:
			raise _Failed("snapshot_missing", str(exc)) from exc
		except IdentityMismatchError as exc:
			raise _Failed("identity_mismatch", str(exc)) from exc
		except SnapshotError as exc:
			raise _Failed("snapshot_corrupt", str(exc)) from exc
		return manifest

	@staticmethod
	def _child_fields(source: SnapshotManifest, record: JobRecord, control_step: int, sim_time_s: float) -> dict:
		return {
			"run_id": source.run_id,
			"episode_id": source.episode_id,
			"branch_id": record.branch_id,
			"control_step": control_step,
			"sim_time_s": sim_time_s,
			"model_fingerprint": source.model_fingerprint,
			"contract": source.contract,
			"scene_spec": source.scene_spec,
			"parent_snapshot_id": source.snapshot_id,
		}

	def _fork(self, record: JobRecord, lease: Lease, hooks: FaultHooks, progress: dict) -> tuple:
		source = self._load_input(record.params["input_snapshot_id"])
		if record.branch_id == source.branch_id:
			raise _Failed("same_branch", f"a fork must name a new branch, not {record.branch_id!r}")
		fields = self._child_fields(source, record, source.control_step, source.sim_time_s)
		return fields, dict(source.blobs), {"frames_executed": 0, "forked_from": source.snapshot_id}

	def _rollout(self, record: JobRecord, lease: Lease, hooks: FaultHooks, progress: dict) -> tuple | None:
		svc, params, job_id = self.service, record.params, record.job_id
		progress["fps"] = int(params["fps"])
		source = self._load_input(params["input_snapshot_id"])
		session = None
		try:
			try:
				state = load_session_state(svc.store, source.blobs)
				session = svc.adapter.open(source.scene_spec)
				svc.store.check_identity(source, svc.identity, session.model_fingerprint())
				session.restore(state)
				if params["program_blob"] is not None:
					arrays = svc.store.get_arrays(BlobRef.from_dict(params["program_blob"]))
					session.load_program(ControlProgram.from_arrays(arrays))
			except IdentityMismatchError as exc:
				raise _Failed("identity_mismatch", str(exc)) from exc
			except SnapshotError as exc:
				raise _Failed("snapshot_corrupt", str(exc)) from exc
			start_step, start_time = session.control_step, session.sim_time_s
			target = int(params["requested_frames"])
			cancelled = False
			try:
				while progress["frames_executed"] < target:
					if session.step(min(self.chunk_frames, target - progress["frames_executed"])) == 0:
						break
					progress["frames_executed"] = session.control_step - start_step
					if hooks.after_chunk:
						hooks.after_chunk(progress["frames_executed"])
					svc.ledger.assert_current(lease)
					still_to_do = progress["frames_executed"] < target
					if still_to_do and svc.ledger.state(job_id) is JobState.CANCEL_REQUESTED:
						cancelled = True
						break
			except StaleWorkerError:
				raise
			except Exception as exc:
				progress["frames_executed"] = session.control_step - start_step
				raise _Failed("simulation_error", f"{type(exc).__name__}: {exc}") from exc
			if cancelled:
				return None
			final = session.capture()
			blobs = save_session_state(svc.store, final)
			result = {
				"schema": ROLLOUT_RESULT_SCHEMA,
				"job_id": job_id,
				"input_snapshot_id": source.snapshot_id,
				"plan_sha256": params["plan_sha256"],
				"engine": session.engine,
				"engine_version": session.engine_version,
				"physics_mode": final.meta.get("physics_mode"),
				"fps": int(params["fps"]),
				"frames_executed": progress["frames_executed"],
				"control_step_start": start_step,
				"control_step_end": final.control_step,
				"sim_time_start_s": start_time,
				"sim_time_end_s": final.sim_time_s,
				"program_completed": session.frames_remaining == 0,
				"telemetry_blob": blobs["arrays/telemetry"].to_dict() if "arrays/telemetry" in blobs else None,
			}
			return self._child_fields(source, record, final.control_step, final.sim_time_s), blobs, result
		finally:
			if session is not None:
				session.close()
