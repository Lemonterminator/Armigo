"""Exception hierarchy for snapshots, the operation ledger, and the execution service."""

from __future__ import annotations


class RuntimeStateError(Exception):
	"""Base class for every persistence and recovery error raised by :mod:`digital_twin_armigo.runtime`."""


class SnapshotError(RuntimeStateError):
	"""A snapshot could not be written, found, or trusted."""


class SnapshotNotFoundError(SnapshotError):
	"""No committed manifest exists for the requested snapshot id."""


class SnapshotCorruptError(SnapshotError):
	"""A manifest or blob is unreadable, truncated, or fails its recorded SHA-256 / size / schema check."""


class IdentityMismatchError(SnapshotError):
	"""Code, assets, config, engine, or model fingerprint differ from the snapshot; direct resume is refused."""


class LedgerError(RuntimeStateError):
	"""The operation ledger rejected a request."""


class JobNotFoundError(LedgerError):
	"""The ledger holds no operation with the requested job id."""


class IdempotencyConflictError(LedgerError):
	"""An idempotency key was reused with a different operation type or different parameters."""


class StaleWorkerError(LedgerError):
	"""A worker presented a fencing token that is missing or no longer current for its resource."""


class InvalidTransitionError(LedgerError):
	"""A job state transition is not allowed (for example, leaving a terminal state)."""


class StopLatchedError(LedgerError):
	"""The run's stop latch is set; new work is refused until it is explicitly cleared."""


class BudgetExceededError(LedgerError):
	"""The request would exceed the run's rollout or simulated-time budget, or the per-rollout limit."""
