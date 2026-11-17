"""A resumable MuJoCo rollout session: full-state capture and restore at control-frame boundaries.

What a snapshot must hold for the continuation to be bit-identical (verified by ``tests/test_mujoco_session_resume``):

* MuJoCo's integration state (``mjSTATE_INTEGRATION``: time, qpos, qvel, act, history, ctrl, applied forces, mocap,
  equality flags, userdata, plugin state and the solver warm start) for every joint and free body, not only the arm
  and the target brick;
* the Python-side state MuJoCo knows nothing about: the actuator command filter (delay line, last output, backlash
  position), the assisted-grasp attachment and its release counter, the observation RNG stream, the control program
  with its cursor, and the telemetry recorded so far (the evaluator is a pure function of that telemetry);
* a fingerprint of the compiled model including parameters changed in memory after loading, so a restore into a
  different model is refused instead of silently simulating something else.
"""

from __future__ import annotations

import copy
import hashlib
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from ..camera import CameraPose
from ..constants import FPS, SO101_MJCF_DIR
from ..physics import ActuatorCommandFilter, NonIdealityCfg, PhysicsSpec, physics_spec_from_dict
from ..runtime.errors import IdentityMismatchError, SnapshotCorruptError
from ..runtime.simulator import (
	SESSION_STATE_SCHEMA,
	ControlProgram,
	SessionState,
	rng_from_json,
	rng_state_to_json,
)
from ._mj import mujoco
from .cell import Cell
from .layout import Layout
from .scene import SUBSTEPS as ASSISTED_SUBSTEPS

MUJOCO_SCENE_SCHEMA = "mujoco-scene/v1"
N_JOINTS = 6

_LAYOUT_TUPLES = ("green", "pink", "brick_xy", "brick_rgb", "table_rgb", "light_dir", "brick_size")
_LAYOUT_TUPLE_LISTS = ("distractors", "clutter", "markers")
# Per-frame telemetry: name -> trailing shape ("S" stands for the number of physics substeps per frame).
_TELEMETRY: dict[str, tuple[int | str, ...]] = {
	"t": (),
	"q_target": (N_JOINTS,),
	"q_cmd": (N_JOINTS,),
	"qpos": (N_JOINTS,),
	"q_obs": (N_JOINTS,),
	"tcp_pos": (3,),
	"ee_quat": (4,),
	"brick_pos": (3,),
	"brick_quat": (4,),
	"brick_vel": (3,),
	"brick_ang_vel": (3,),
	"jaw_forces": (2,),
	"pinch_normal": (2,),
	"pinch_tangential": (2,),
	"unintended_forces": (),
	"substep_unintended": ("S",),
	"actuator_torques": (N_JOINTS,),
	"phases": (),
}


def _plain(value: Any) -> Any:
	"""Dataclass / tuple / NumPy values as plain JSON types."""
	if isinstance(value, Mapping):
		return {str(k): _plain(v) for k, v in value.items()}
	if isinstance(value, (list, tuple)):
		return [_plain(v) for v in value]
	if isinstance(value, np.ndarray):
		return value.tolist()
	if isinstance(value, np.generic):
		return value.item()
	return value


def mujoco_scene_spec(
	layout: Layout,
	*,
	physics: PhysicsSpec,
	q_start: Sequence[float],
	brick_pose: tuple[Sequence[float], Sequence[float]] | None = None,
	physics_config_sha256: str | None = None,
	mass_scale: float = 1.0,
	friction_scale: float = 1.0,
	dynamic_clutter: bool = True,
	assisted_grasp: bool = False,
	physics_hz: int = 240,
	non_ideality: NonIdealityCfg | None = None,
	settle_steps: int = 12,
	seed: int = 0,
) -> dict[str, Any]:
	"""The JSON scene specification a :class:`MujocoSession` is built from (and rebuilt from on restore).

	``brick_pose`` is ``(position, quaternion xyzw)``; ``None`` keeps the layout's pose. ``seed`` seeds the
	session's own observation-noise stream. The spec holds simulation truth (including the defect flag): it belongs
	in snapshots, never in a planner's context.
	"""
	if physics_hz % FPS != 0:
		raise ValueError(f"physics_hz must be a multiple of the {FPS} Hz control rate, got {physics_hz}")
	substeps = physics_hz // FPS
	if assisted_grasp and substeps != ASSISTED_SUBSTEPS:
		raise ValueError(
			f"assisted-grasp compatibility mode steps {ASSISTED_SUBSTEPS} substeps per frame; "
			f"use physics_hz={FPS * ASSISTED_SUBSTEPS}"
		)
	q0 = [float(x) for x in q_start]
	if len(q0) != N_JOINTS:
		raise ValueError(f"q_start needs {N_JOINTS} joint angles")
	return {
		"schema": MUJOCO_SCENE_SCHEMA,
		"layout": _plain(asdict(layout)),
		"physics": {"config_sha256": physics_config_sha256, "effective_spec": _plain(asdict(physics))},
		"mass_scale": float(mass_scale),
		"friction_scale": float(friction_scale),
		"dynamic_clutter": bool(dynamic_clutter),
		"assisted_grasp": bool(assisted_grasp),
		"fps": FPS,
		"substeps": int(substeps),
		"timestep_s": 1.0 / float(physics_hz),
		"non_ideality": None if non_ideality is None else _plain(asdict(non_ideality)),
		"q_start": q0,
		"brick_pose": None
		if brick_pose is None
		else {"pos": _plain(brick_pose[0]), "quat_xyzw": _plain(brick_pose[1])},
		"settle_steps": int(settle_steps),
		"seed": int(seed),
	}


def _layout_from_spec(raw: Mapping[str, Any]) -> Layout:
	fields = dict(raw)
	for key in _LAYOUT_TUPLES:
		fields[key] = tuple(fields[key])
	for key in _LAYOUT_TUPLE_LISTS:
		fields[key] = [tuple(item) for item in fields[key]]
	for key in ("top", "front"):
		fields[key] = None if fields[key] is None else CameraPose(**fields[key])
	return Layout(**fields)


def model_fingerprint(model: Any) -> str:
	"""SHA-256 of the compiled model's binary serialisation (covers every parameter changed after compilation)."""
	buf = np.zeros(mujoco.mj_sizeModel(model), dtype=np.uint8)
	mujoco.mj_saveModel(model, None, buf)
	return hashlib.sha256(buf.tobytes()).hexdigest()


class MujocoSession:
	"""One MuJoCo sorting-cell instance driven by a :class:`ControlProgram`, resumable at any control frame."""

	engine = "mujoco"

	def __init__(self, scene_spec: Mapping[str, Any], menagerie: Path = SO101_MJCF_DIR) -> None:
		if mujoco is None:
			raise RuntimeError("the 'mujoco' extra is not installed")
		if scene_spec.get("schema") != MUJOCO_SCENE_SCHEMA:
			raise ValueError(f"unsupported scene schema: {scene_spec.get('schema')!r}")
		self.engine_version: str = mujoco.__version__
		self.scene_spec: dict[str, Any] = copy.deepcopy(dict(scene_spec))
		spec = self.scene_spec
		self.substeps = int(spec["substeps"])
		self.assisted = bool(spec["assisted_grasp"])
		self.cell = Cell(
			_layout_from_spec(spec["layout"]),
			menagerie,
			physics=physics_spec_from_dict(spec["physics"]["effective_spec"]),
			dynamic_clutter=bool(spec["dynamic_clutter"]),
			assisted_grasp=self.assisted,
			mass_scale=float(spec["mass_scale"]),
			friction_scale=float(spec["friction_scale"]),
			render_enabled=False,
		)
		cell = self.cell
		cell.model.opt.timestep = float(spec["timestep_s"])
		q_start = np.array(spec["q_start"], dtype=np.float64)
		cell.set_joints(q_start)
		cell.data.qvel[:] = 0.0
		if spec["brick_pose"] is not None:
			cell.set_brick_pose(np.array(spec["brick_pose"]["pos"]), np.array(spec["brick_pose"]["quat_xyzw"]))
		cell.data.ctrl[:N_JOINTS] = q_start
		for _ in range(int(spec["settle_steps"])):
			mujoco.mj_step(cell.model, cell.data)
		self._filter: ActuatorCommandFilter | None = None
		if spec["non_ideality"] is not None:
			self._filter = ActuatorCommandFilter(NonIdealityCfg(**spec["non_ideality"]), n_joints=N_JOINTS, fps=FPS)
			self._filter.reset(q_start)
		self._rng = {"observation": np.random.default_rng(int(spec["seed"]))}
		self._state_sig = int(mujoco.mjtState.mjSTATE_INTEGRATION)
		self._state_size = int(mujoco.mj_stateSize(cell.model, self._state_sig))
		self._control_step = 0
		self._program = ControlProgram(FPS, np.zeros((0, N_JOINTS)), np.zeros(0, dtype=np.uint8))
		self._cursor = 0
		self._log: dict[str, list[np.ndarray]] = {name: [] for name in _TELEMETRY}

	# -- progress --------------------------------------------------------------------------------
	@property
	def control_step(self) -> int:
		return self._control_step

	@property
	def sim_time_s(self) -> float:
		return float(self.cell.data.time)

	@property
	def frames_remaining(self) -> int:
		return len(self._program) - self._cursor

	@property
	def physics_mode(self) -> str:
		"""``pure_physics`` or ``assisted_compat``; assisted results must never enter a pure-physics ranking."""
		return "assisted_compat" if self.assisted else "pure_physics"

	def model_fingerprint(self) -> str:
		return model_fingerprint(self.cell.model)

	def load_program(self, program: ControlProgram) -> None:
		if program.fps != FPS or program.q_targets.shape[1] != N_JOINTS:
			raise ValueError(f"program must be {FPS} fps with {N_JOINTS} joints")
		self._program = program
		self._cursor = 0

	# -- stepping --------------------------------------------------------------------------------
	def step(self, n_frames: int) -> int:
		done = 0
		while done < n_frames and self._cursor < len(self._program):
			self._step_one()
			done += 1
		return done

	def _step_one(self) -> None:
		cell = self.cell
		target = self._program.q_targets[self._cursor]
		q_cmd = self._filter.step_command(target, cell.joints()) if self._filter is not None else target.copy()
		if self.assisted:
			# Compatibility mode: kinematic attachment and capped release speed; frame-end samples only.
			cell.step(q_cmd)
			cell.update_grasp(float(target[5]))
			fn0, ft0, tot0, fn1, ft1, tot1 = cell.jaw_force_components()
			unintended = np.full(self.substeps, cell.unintended_force())
			jaw_total, jaw_normal, jaw_tan = np.array([tot0, tot1]), np.array([fn0, fn1]), np.array([ft0, ft1])
			torques = cell.actuator_torques()
		else:
			sub = cell.step_with_telemetry(q_cmd, substeps=self.substeps)
			unintended = sub["unintended"]
			jaw_total = sub["jaw_total"][-1]
			jaw_normal = sub["jaw_normal"].mean(axis=0)
			jaw_tan = sub["jaw_tangential"].mean(axis=0)
			torques = sub["torques"][np.argmax(np.abs(sub["torques"]), axis=0), np.arange(N_JOINTS)]
		q_act = cell.joints()
		q_obs = self._filter.observe_joints(q_act, self._rng["observation"]) if self._filter is not None else q_act
		wxyz = cell.data.xquat[cell.grip_body]
		lin, ang = cell.brick_vel()
		frame = {
			"t": self._control_step / float(FPS),
			"q_target": target,
			"q_cmd": q_cmd,
			"qpos": q_act,
			"q_obs": q_obs,
			"tcp_pos": cell.tcp_pos(),
			"ee_quat": (wxyz[1], wxyz[2], wxyz[3], wxyz[0]),
			"brick_pos": cell.brick_pos(),
			"brick_quat": cell.brick_quat_xyzw(),
			"brick_vel": lin,
			"brick_ang_vel": ang,
			"jaw_forces": jaw_total,
			"pinch_normal": jaw_normal,
			"pinch_tangential": jaw_tan,
			"unintended_forces": unintended[-1],
			"substep_unintended": unintended,
			"actuator_torques": torques,
			"phases": self._program.phases[self._cursor],
		}
		for name, value in frame.items():
			self._log[name].append(np.array(value, dtype=np.float64))
		self._cursor += 1
		self._control_step += 1

	def _telemetry_shape(self, name: str) -> tuple[int, ...]:
		return tuple(self.substeps if dim == "S" else int(dim) for dim in _TELEMETRY[name])

	def telemetry(self) -> dict[str, np.ndarray]:
		"""Per-frame logs since the episode started, in the layout ``evaluate_episode_health`` consumes."""
		out = {}
		for name, rows in self._log.items():
			dtype = np.uint8 if name == "phases" else np.float64
			out[name] = np.array(rows, dtype=dtype).reshape((len(rows), *self._telemetry_shape(name)))
		return out

	# -- capture / restore -----------------------------------------------------------------------
	def capture(self) -> SessionState:
		"""Snapshot the complete session at the current control-frame boundary without disturbing it."""
		cell = self.cell
		physics = np.zeros(self._state_size, dtype=np.float64)
		mujoco.mj_getState(cell.model, cell.data, physics, self._state_sig)
		controller: dict[str, np.ndarray] = {}
		if self._filter is not None:
			controller = {
				"queue": np.array(list(self._filter._queue), dtype=np.float64).reshape((-1, N_JOINTS)),
				"prev_out": self._filter._prev_out.copy(),
				"backlash_pos": self._filter._backlash_pos.copy(),
			}
		held = None
		if cell.held is not None:
			held = {"rel_pos": cell.held[0].tolist(), "rel_rot": cell.held[1].tolist()}
		meta = {
			"schema": SESSION_STATE_SCHEMA,
			"engine": self.engine,
			"engine_version": self.engine_version,
			"model_fingerprint": self.model_fingerprint(),
			"physics_mode": self.physics_mode,
			"state_signature": self._state_sig,
			"state_size": self._state_size,
			"program_cursor": self._cursor,
			"program_sha256": self._program.sha256(),
			"cell": {"held": held, "settle_frames": int(cell.settle_frames)},
			"rng": {name: rng_state_to_json(rng) for name, rng in self._rng.items()},
		}
		arrays = {
			"physics": {"state": physics},
			"controller": controller,
			"program": {k: np.array(v) for k, v in self._program.to_arrays().items()},
			"telemetry": self.telemetry(),
		}
		return SessionState(self._control_step, self.sim_time_s, arrays, meta)

	def restore(self, state: SessionState) -> None:
		"""Adopt ``state``. Everything is validated first; on any error this session is left exactly as it was."""
		meta = state.meta
		if meta.get("schema") != SESSION_STATE_SCHEMA:
			raise SnapshotCorruptError(f"unknown session state schema: {meta.get('schema')!r}")
		if meta.get("engine") != self.engine or meta.get("engine_version") != self.engine_version:
			raise IdentityMismatchError(
				f"snapshot was taken with engine {meta.get('engine')} {meta.get('engine_version')}, "
				f"this session runs {self.engine} {self.engine_version}"
			)
		if meta.get("model_fingerprint") != self.model_fingerprint():
			raise IdentityMismatchError("model_fingerprint of the snapshot differs from this session's model")
		try:
			physics = np.array(state.arrays["physics"]["state"], dtype=np.float64)
			program = ControlProgram.from_arrays(state.arrays["program"])
			cursor = int(meta["program_cursor"])
			held_doc = meta["cell"]["held"]
			settle_frames = int(meta["cell"]["settle_frames"])
			rng = {name: rng_from_json(doc) for name, doc in meta["rng"].items()}
			telemetry = self._checked_telemetry(state.arrays["telemetry"], int(state.control_step))
			controller = self._checked_controller(state.arrays["controller"])
			held = None
			if held_doc is not None:
				held = (
					np.array(held_doc["rel_pos"], dtype=np.float64),
					np.array(held_doc["rel_rot"], dtype=np.float64),
				)
				if held[0].shape != (3,) or held[1].shape != (3, 3):
					raise ValueError("assisted-grasp attachment has the wrong shape")
		except (KeyError, TypeError, ValueError) as exc:
			raise SnapshotCorruptError(f"session state is malformed: {exc}") from exc
		if meta.get("state_signature") != self._state_sig or physics.shape != (self._state_size,):
			raise SnapshotCorruptError(
				f"physics state has shape {physics.shape}, expected ({self._state_size},) for this model"
			)
		if not np.isfinite(physics).all():
			raise SnapshotCorruptError("physics state holds non-finite values")
		if set(rng) != set(self._rng):
			raise SnapshotCorruptError(f"snapshot RNG streams {sorted(rng)} do not match {sorted(self._rng)}")
		if not 0 <= cursor <= len(program) or program.fps != FPS or program.q_targets.shape[1] != N_JOINTS:
			raise SnapshotCorruptError("program cursor or program shape is inconsistent")
		if meta.get("program_sha256") != program.sha256():
			raise SnapshotCorruptError("control program does not match its recorded hash")

		# Nothing below can fail: commit the validated state.
		cell = self.cell
		mujoco.mj_setState(cell.model, cell.data, physics, self._state_sig)
		mujoco.mj_forward(cell.model, cell.data)  # refresh derived quantities (positions, contacts) for readers
		mujoco.mj_setState(cell.model, cell.data, physics, self._state_sig)  # and keep the integration state exact
		cell.held = held
		cell.settle_frames = settle_frames
		if self._filter is not None:
			size = max(1, self._filter.cfg.latency_frames)
			self._filter._queue = deque([row.copy() for row in controller["queue"]], maxlen=size)
			self._filter._prev_out = controller["prev_out"].copy()
			self._filter._backlash_pos = controller["backlash_pos"].copy()
		self._rng = rng
		self._program = program
		self._cursor = cursor
		self._control_step = int(state.control_step)
		self._log = telemetry

	def _checked_telemetry(self, arrays: Mapping[str, np.ndarray], frames: int) -> dict[str, list[np.ndarray]]:
		if set(arrays) != set(_TELEMETRY):
			raise ValueError("telemetry channels do not match this session")
		log = {}
		for name, arr in arrays.items():
			if arr.shape != (frames, *self._telemetry_shape(name)):
				raise ValueError(f"telemetry channel {name} has shape {arr.shape} for {frames} frames")
			log[name] = [np.array(row, dtype=np.float64) for row in arr]
		return log

	def _checked_controller(self, arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
		if self._filter is None:
			if arrays:
				raise ValueError("snapshot carries controller state but this scene has no command filter")
			return {}
		out = {name: np.array(arrays[name], dtype=np.float64) for name in ("queue", "prev_out", "backlash_pos")}
		if out["queue"].shape != (max(0, self._filter.cfg.latency_frames), N_JOINTS):
			raise ValueError("delay line length does not match the scene's latency")
		if out["prev_out"].shape != (N_JOINTS,) or out["backlash_pos"].shape != (N_JOINTS,):
			raise ValueError("controller vectors have the wrong shape")
		if not all(np.isfinite(v).all() for v in out.values()):
			raise ValueError("controller state holds non-finite values")
		return out

	def close(self) -> None:
		self.cell.close()


class MujocoAdapter:
	"""Builds :class:`MujocoSession` instances from scene specifications."""

	engine = "mujoco"

	def __init__(self, menagerie: Path = SO101_MJCF_DIR) -> None:
		self.menagerie = menagerie
		self.engine_version: str = mujoco.__version__ if mujoco is not None else "unavailable"

	def open(self, scene_spec: Mapping[str, Any]) -> MujocoSession:
		return MujocoSession(scene_spec, self.menagerie)
