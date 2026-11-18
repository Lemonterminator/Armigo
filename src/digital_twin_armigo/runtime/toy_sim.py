"""A hardware-free reference simulator for exercising snapshots and the execution service.

It is deliberately tiny (a damped spring per joint driven through a delay line, with seeded process noise) but it has
every kind of state a real session has: integrator state, a controller queue, a consumed RNG stream, a program
cursor and telemetry. It makes no physical claim; it exists so recovery logic can be fault-injected in milliseconds.
"""

from __future__ import annotations

import copy
from collections import deque
from collections.abc import Mapping
from typing import Any

import numpy as np

from .canonical import sha256_json
from .errors import IdentityMismatchError, SnapshotCorruptError
from .simulator import SESSION_STATE_SCHEMA, ControlProgram, SessionState, rng_from_json, rng_state_to_json

TOY_SCENE_SCHEMA = "toy-scene/v1"


def toy_scene_spec(
	*,
	n_joints: int = 3,
	fps: int = 30,
	stiffness: float = 40.0,
	damping: float = 6.0,
	noise_std: float = 0.02,
	latency_frames: int = 2,
	seed: int = 0,
	fail_at_step: int | None = None,
) -> dict[str, Any]:
	"""Scene specification for :class:`ToySession`; ``fail_at_step`` makes the engine raise at that control step."""
	return {
		"schema": TOY_SCENE_SCHEMA,
		"n_joints": int(n_joints),
		"fps": int(fps),
		"stiffness": float(stiffness),
		"damping": float(damping),
		"noise_std": float(noise_std),
		"latency_frames": int(latency_frames),
		"seed": int(seed),
		"fail_at_step": fail_at_step,
	}


class ToySession:
	"""Damped-spring joints tracking delayed targets; implements the :class:`SimulatorSession` protocol."""

	engine = "toy"

	def __init__(self, scene_spec: Mapping[str, Any], engine_version: str = "1") -> None:
		if scene_spec.get("schema") != TOY_SCENE_SCHEMA:
			raise ValueError(f"unsupported scene schema: {scene_spec.get('schema')!r}")
		self.engine_version = engine_version
		self.scene_spec: dict[str, Any] = copy.deepcopy(dict(scene_spec))
		self.params = {k: self.scene_spec[k] for k in ("n_joints", "fps", "stiffness", "damping", "noise_std")}
		n = int(self.params["n_joints"])
		self._x = np.zeros(n)
		self._v = np.zeros(n)
		self._time = 0.0
		self._queue: deque[np.ndarray] = deque(
			[np.zeros(n) for _ in range(int(self.scene_spec["latency_frames"]))],
			maxlen=max(1, int(self.scene_spec["latency_frames"])),
		)
		self._rng = np.random.default_rng(int(self.scene_spec["seed"]))
		self._program = ControlProgram(int(self.params["fps"]), np.zeros((0, n)), np.zeros(0, dtype=np.uint8))
		self._cursor = 0
		self._control_step = 0
		self._log: dict[str, list[np.ndarray]] = {"x": [], "q_cmd": []}

	@property
	def control_step(self) -> int:
		return self._control_step

	@property
	def sim_time_s(self) -> float:
		return self._time

	@property
	def frames_remaining(self) -> int:
		return len(self._program) - self._cursor

	def model_fingerprint(self) -> str:
		return sha256_json(self.params)

	def load_program(self, program: ControlProgram) -> None:
		if program.fps != self.params["fps"] or program.q_targets.shape[1] != self.params["n_joints"]:
			raise ValueError("program does not match the toy scene")
		self._program = program
		self._cursor = 0

	def step(self, n_frames: int) -> int:
		done = 0
		dt = 1.0 / float(self.params["fps"])
		while done < n_frames and self._cursor < len(self._program):
			if self.scene_spec["fail_at_step"] is not None and self._control_step == self.scene_spec["fail_at_step"]:
				raise FloatingPointError(f"toy engine diverged at control step {self._control_step}")
			cmd = self._program.q_targets[self._cursor].copy()
			if self.scene_spec["latency_frames"] > 0:
				delayed = self._queue[0].copy()
				self._queue.append(cmd)
				cmd = delayed
			noise = self._rng.normal(0.0, float(self.params["noise_std"]), size=self._x.shape)
			acc = float(self.params["stiffness"]) * (cmd - self._x) - float(self.params["damping"]) * self._v + noise
			self._v = self._v + acc * dt
			self._x = self._x + self._v * dt
			self._time += dt
			self._log["x"].append(self._x.copy())
			self._log["q_cmd"].append(cmd)
			self._cursor += 1
			self._control_step += 1
			done += 1
		return done

	def telemetry(self) -> dict[str, np.ndarray]:
		n = int(self.params["n_joints"])
		return {k: np.array(v, dtype=np.float64).reshape((len(v), n)) for k, v in self._log.items()}

	def capture(self) -> SessionState:
		n = int(self.params["n_joints"])
		meta = {
			"schema": SESSION_STATE_SCHEMA,
			"engine": self.engine,
			"engine_version": self.engine_version,
			"model_fingerprint": self.model_fingerprint(),
			"physics_mode": "toy",
			"program_cursor": self._cursor,
			"program_sha256": self._program.sha256(),
			"rng": {"process": rng_state_to_json(self._rng)},
		}
		arrays = {
			"physics": {"x": self._x.copy(), "v": self._v.copy(), "time": np.array(self._time)},
			"controller": {"queue": np.array(list(self._queue), dtype=np.float64).reshape((-1, n))},
			"program": {k: np.array(v) for k, v in self._program.to_arrays().items()},
			"telemetry": self.telemetry(),
		}
		return SessionState(self._control_step, self._time, arrays, meta)

	def restore(self, state: SessionState) -> None:
		meta = state.meta
		if meta.get("schema") != SESSION_STATE_SCHEMA:
			raise SnapshotCorruptError(f"unknown session state schema: {meta.get('schema')!r}")
		if meta.get("engine") != self.engine or meta.get("engine_version") != self.engine_version:
			raise IdentityMismatchError("snapshot engine or engine version differs from this session")
		if meta.get("model_fingerprint") != self.model_fingerprint():
			raise IdentityMismatchError("model_fingerprint of the snapshot differs from this session's model")
		n = int(self.params["n_joints"])
		try:
			x = np.array(state.arrays["physics"]["x"], dtype=np.float64)
			v = np.array(state.arrays["physics"]["v"], dtype=np.float64)
			time = float(state.arrays["physics"]["time"])
			queue = np.array(state.arrays["controller"]["queue"], dtype=np.float64)
			program = ControlProgram.from_arrays(state.arrays["program"])
			cursor = int(meta["program_cursor"])
			rng = rng_from_json(meta["rng"]["process"])
			log = {k: [row.copy() for row in np.array(state.arrays["telemetry"][k])] for k in self._log}
		except (KeyError, TypeError, ValueError) as exc:
			raise SnapshotCorruptError(f"session state is malformed: {exc}") from exc
		frames = int(state.control_step)
		if x.shape != (n,) or v.shape != (n,) or queue.shape != (int(self.scene_spec["latency_frames"]), n):
			raise SnapshotCorruptError("toy state arrays have the wrong shape")
		if not (np.isfinite(x).all() and np.isfinite(v).all() and np.isfinite(queue).all() and np.isfinite(time)):
			raise SnapshotCorruptError("toy state holds non-finite values")
		if not 0 <= cursor <= len(program) or any(len(rows) != frames for rows in log.values()):
			raise SnapshotCorruptError("cursor or telemetry length is inconsistent")
		if meta.get("program_sha256") != program.sha256():
			raise SnapshotCorruptError("control program does not match its recorded hash")
		self._x, self._v, self._time = x, v, time
		self._queue = deque([row.copy() for row in queue], maxlen=max(1, int(self.scene_spec["latency_frames"])))
		self._rng = rng
		self._program, self._cursor = program, cursor
		self._control_step = frames
		self._log = log

	def close(self) -> None:
		return None


class ToyAdapter:
	"""Builds :class:`ToySession` instances; ``engine_version`` and ``drift`` let tests fake a changed engine/model."""

	engine = "toy"

	def __init__(self, engine_version: str = "1", stiffness_drift: float = 0.0) -> None:
		self.engine_version = engine_version
		self.stiffness_drift = stiffness_drift

	def open(self, scene_spec: Mapping[str, Any]) -> ToySession:
		session = ToySession(scene_spec, self.engine_version)
		session.params["stiffness"] += self.stiffness_drift  # a parameter changed outside the scene specification
		return session
