"""Engine-neutral session interface: control programs, captured session state, and the adapter protocols.

A *session* is one simulator instance advancing in whole control frames. Its complete resumable state is a
:class:`SessionState`: named groups of numeric arrays (physics, controller, program, telemetry) plus a JSON document
(cursors, RNG states, flags). Anything a session needs to continue must be in there; nothing may live only in memory.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .canonical import sha256_bytes, sha256_json
from .errors import SnapshotCorruptError

SESSION_STATE_SCHEMA = "session-state/v1"


@dataclass(frozen=True)
class ControlProgram:
	"""A time-stamped open-loop joint target sequence: one target row and one phase label per control frame."""

	fps: int
	q_targets: np.ndarray
	phases: np.ndarray

	def __post_init__(self) -> None:
		q = np.ascontiguousarray(self.q_targets, dtype=np.float64)
		ph = np.ascontiguousarray(self.phases, dtype=np.uint8)
		if q.ndim != 2 or ph.shape != (q.shape[0],):
			raise ValueError(f"q_targets must be (T, n_joints) and phases (T,); got {q.shape} and {ph.shape}")
		if not np.isfinite(q).all():
			raise ValueError("q_targets holds non-finite values")
		if self.fps <= 0:
			raise ValueError("fps must be positive")
		q.setflags(write=False)
		ph.setflags(write=False)
		object.__setattr__(self, "q_targets", q)
		object.__setattr__(self, "phases", ph)

	def __len__(self) -> int:
		return int(self.q_targets.shape[0])

	def sha256(self) -> str:
		"""Content hash over fps, shapes, targets and phases (the plan hash bound to a rollout)."""
		header = sha256_json({"fps": int(self.fps), "shape": list(self.q_targets.shape)})
		return sha256_bytes(header.encode("ascii") + self.q_targets.tobytes() + self.phases.tobytes())

	def to_arrays(self) -> dict[str, np.ndarray]:
		return {"fps": np.array(self.fps, dtype=np.int64), "q_targets": self.q_targets, "phases": self.phases}

	@classmethod
	def from_arrays(cls, arrays: Mapping[str, np.ndarray]) -> ControlProgram:
		try:
			return cls(int(arrays["fps"]), np.array(arrays["q_targets"]), np.array(arrays["phases"]))
		except (KeyError, ValueError, TypeError) as exc:
			raise SnapshotCorruptError(f"control program blob is malformed: {exc}") from exc


@dataclass(frozen=True)
class SessionState:
	"""Everything one session needs to continue from a control-frame boundary.

	``arrays`` must hold a ``program`` group (see :meth:`ControlProgram.to_arrays`); ``meta`` must carry
	``program_cursor`` and ``program_sha256`` so a service can size and bind a continuation without the engine.
	"""

	control_step: int
	sim_time_s: float
	arrays: dict[str, dict[str, np.ndarray]]
	meta: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class SimulatorSession(Protocol):
	"""One simulator instance. All methods act at control-frame boundaries."""

	engine: str
	engine_version: str
	scene_spec: dict[str, Any]

	@property
	def control_step(self) -> int:
		"""Control frames executed since the episode started (never reset by loading a new program)."""

	@property
	def sim_time_s(self) -> float: ...

	@property
	def frames_remaining(self) -> int:
		"""Frames left in the loaded program."""

	def model_fingerprint(self) -> str:
		"""Hash of the model the engine actually simulates, including parameters mutated after loading."""

	def load_program(self, program: ControlProgram) -> None:
		"""Replace the control program; execution continues from its first frame."""

	def step(self, n_frames: int) -> int:
		"""Advance up to ``n_frames`` control frames; return how many were executed."""

	def capture(self) -> SessionState: ...

	def restore(self, state: SessionState) -> None:
		"""Adopt ``state``; raises ``IdentityMismatchError`` / ``SnapshotCorruptError`` instead of guessing."""

	def telemetry(self) -> dict[str, np.ndarray]: ...

	def close(self) -> None: ...


@runtime_checkable
class SimulatorAdapter(Protocol):
	"""Builds sessions for one engine from a JSON scene specification."""

	engine: str
	engine_version: str

	def open(self, scene_spec: Mapping[str, Any]) -> SimulatorSession:
		"""A fresh session at the scene's initial state, with no program loaded."""


def rng_state_to_json(rng: np.random.Generator) -> dict[str, Any]:
	"""The full bit-generator state (a seed alone cannot resume a stream that has already been drawn from)."""
	return _jsonable(rng.bit_generator.state)


def rng_from_json(state: Mapping[str, Any]) -> np.random.Generator:
	name = state.get("bit_generator")
	if name != "PCG64":
		raise SnapshotCorruptError(f"unsupported RNG bit generator in snapshot: {name!r}")
	bit_gen = np.random.PCG64()
	try:
		bit_gen.state = dict(state)
	except (KeyError, TypeError, ValueError) as exc:
		raise SnapshotCorruptError(f"RNG state in snapshot is malformed: {exc}") from exc
	return np.random.Generator(bit_gen)


def _jsonable(value: Any) -> Any:
	if isinstance(value, Mapping):
		return {str(k): _jsonable(v) for k, v in value.items()}
	if isinstance(value, (list, tuple)):
		return [_jsonable(v) for v in value]
	if isinstance(value, np.generic):
		return value.item()
	return value


def diff_session_states(a: SessionState, b: SessionState) -> list[str]:
	"""Paths at which two session states differ bit-for-bit; an empty list means an exact match."""
	diffs = []
	if a.control_step != b.control_step:
		diffs.append("control_step")
	if a.sim_time_s != b.sim_time_s:
		diffs.append("sim_time_s")
	for key in sorted(set(a.meta) | set(b.meta)):
		if a.meta.get(key) != b.meta.get(key):
			diffs.append(f"meta/{key}")
	for group in sorted(set(a.arrays) | set(b.arrays)):
		left, right = a.arrays.get(group), b.arrays.get(group)
		if left is None or right is None:
			diffs.append(f"arrays/{group}")
			continue
		for name in sorted(set(left) | set(right)):
			x, y = left.get(name), right.get(name)
			if x is None or y is None or x.dtype != y.dtype or x.shape != y.shape or not np.array_equal(x, y):
				diffs.append(f"arrays/{group}/{name}")
	return diffs


_STATE_BLOB = "state"
_ARRAY_PREFIX = "arrays/"


def save_session_state(store: Any, state: SessionState) -> dict[str, Any]:
	"""Write ``state`` into a snapshot store; returns the blob references a manifest should carry.

	One array blob per group (``arrays/<group>``) plus one JSON blob (``state``) with the cursors and metadata.
	"""
	blobs = {f"{_ARRAY_PREFIX}{group}": store.put_arrays(arrays) for group, arrays in state.arrays.items()}
	blobs[_STATE_BLOB] = store.put_json(
		{
			"schema": SESSION_STATE_SCHEMA,
			"control_step": int(state.control_step),
			"sim_time_s": float(state.sim_time_s),
			"groups": sorted(state.arrays),
			"meta": state.meta,
		}
	)
	return blobs


def load_session_state(store: Any, blobs: Mapping[str, Any]) -> SessionState:
	"""Read back what :func:`save_session_state` wrote, verifying every blob on the way."""
	if _STATE_BLOB not in blobs:
		raise SnapshotCorruptError("snapshot has no session state blob")
	doc = store.get_json(blobs[_STATE_BLOB])
	if not isinstance(doc, dict) or doc.get("schema") != SESSION_STATE_SCHEMA:
		raise SnapshotCorruptError("session state blob has an unknown schema")
	try:
		groups = list(doc["groups"])
		arrays = {group: store.get_arrays(blobs[f"{_ARRAY_PREFIX}{group}"]) for group in groups}
		return SessionState(int(doc["control_step"]), float(doc["sim_time_s"]), arrays, dict(doc["meta"]))
	except (KeyError, TypeError, ValueError) as exc:
		raise SnapshotCorruptError(f"session state blob is malformed: {exc}") from exc
