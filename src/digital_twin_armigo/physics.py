"""Unified physics parameters, provenance, and actuator non-ideality filters shared by MuJoCo and Isaac Lab."""

from __future__ import annotations

import hashlib
import json
import math
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .constants import CONFIG_DIR, DUPLO_2X2, DUPLO_2X3, DUPLO_2X4, FPS, STUD_PITCH

PHYSICS_CONFIG_PATH = CONFIG_DIR / "physics_params.json"
PHYSICS_SPEC_VERSION = "v1-nominal-assumed"


@dataclass(frozen=True)
class SurfaceMaterialSpec:
	"""Contact friction and restitution properties for one physical surface."""

	static_friction: float
	dynamic_friction: float
	restitution: float = 0.05


@dataclass(frozen=True)
class BrickSpec:
	"""Physical specification for one DUPLO brick variant."""

	name: str
	size_m: tuple[float, float, float]
	mass_kg: float
	static_friction: float
	dynamic_friction: float
	restitution: float = 0.05

	def scaled(self, mass_scale: float = 1.0, friction_scale: float = 1.0) -> BrickSpec:
		return BrickSpec(
			name=self.name,
			size_m=self.size_m,
			mass_kg=self.mass_kg * mass_scale,
			static_friction=self.static_friction * friction_scale,
			dynamic_friction=self.dynamic_friction * friction_scale,
			restitution=self.restitution,
		)


@dataclass(frozen=True)
class PlateSpec:
	"""Physical specification for the 200 mm sorting plate."""

	mass_kg: float
	static_friction: float
	dynamic_friction: float
	restitution: float = 0.05


@dataclass(frozen=True)
class ActuatorSpec:
	"""Unified PD + motor limits approximating Feetech STS3215 servos."""

	stiffness: float
	damping: float
	effort_limit: float
	velocity_limit: float
	joint_damping: float
	frictionloss: float
	armature: float


@dataclass(frozen=True)
class PhysicsSpec:
	"""Complete versioned physics specification for the SO-101 sorting cell."""

	version: str
	provenance: str
	bricks: dict[str, BrickSpec]
	plate: PlateSpec
	table: SurfaceMaterialSpec
	gripper_pad: SurfaceMaterialSpec
	arm_actuator: ActuatorSpec
	gripper_actuator: ActuatorSpec


DEFAULT_PHYSICS = PhysicsSpec(
	version=PHYSICS_SPEC_VERSION,
	provenance=(
		"Assumed nominal literature/catalog specs (BrickLink ABS DUPLO weights, ABS friction, "
		"Feetech STS3215 1:345 specs). Pending real-hardware measurement."
	),
	bricks={
		"2x2": BrickSpec("2x2", DUPLO_2X2, 0.0070, 0.40, 0.35, 0.05),
		"2x3": BrickSpec("2x3", DUPLO_2X3, 0.0096, 0.40, 0.35, 0.05),
		"2x4": BrickSpec("2x4", DUPLO_2X4, 0.0135, 0.40, 0.35, 0.05),
	},
	plate=PlateSpec(0.12, 0.55, 0.45, 0.05),
	table=SurfaceMaterialSpec(0.45, 0.38, 0.05),
	gripper_pad=SurfaceMaterialSpec(1.45, 1.25, 0.02),
	arm_actuator=ActuatorSpec(180.0, 12.0, 2.94, 5.45, 0.60, 0.052, 0.028),
	gripper_actuator=ActuatorSpec(30.0, 2.5, 0.55, 5.45, 0.30, 0.030, 0.015),
)


def load_physics_spec(path: Path | None = None, *, strict: bool = True) -> PhysicsSpec:
	"""Load the versioned physics specification from JSON.

	When ``strict`` is True (the default for audited evaluations), missing config files raise
	:class:`FileNotFoundError` instead of silently substituting :data:`DEFAULT_PHYSICS`.
	"""
	cfg_path = path or PHYSICS_CONFIG_PATH
	if not cfg_path.is_file():
		if strict:
			raise FileNotFoundError(f"Physics configuration file not found: {cfg_path}")
		return DEFAULT_PHYSICS
	return physics_spec_from_dict(json.loads(cfg_path.read_text(encoding="utf-8")))


def physics_spec_from_dict(raw: dict[str, Any]) -> PhysicsSpec:
	"""Build a :class:`PhysicsSpec` from the JSON config layout (also the layout of ``dataclasses.asdict``)."""
	bricks = {
		k: BrickSpec(
			name=v["name"],
			size_m=tuple(float(x) for x in v["size_m"]),
			mass_kg=float(v["mass_kg"]),
			static_friction=float(v["static_friction"]),
			dynamic_friction=float(v["dynamic_friction"]),
			restitution=float(v.get("restitution", 0.05)),
		)
		for k, v in raw["bricks"].items()
	}
	return PhysicsSpec(
		version=str(raw["version"]),
		provenance=str(raw["provenance"]),
		bricks=bricks,
		plate=PlateSpec(**raw["plate"]),
		table=SurfaceMaterialSpec(**raw["table"]),
		gripper_pad=SurfaceMaterialSpec(**raw["gripper_pad"]),
		arm_actuator=ActuatorSpec(**raw["arm_actuator"]),
		gripper_actuator=ActuatorSpec(**raw["gripper_actuator"]),
	)


def physics_manifest(spec: PhysicsSpec, path: Path | None = None) -> dict[str, Any]:
	"""Return an explicit audit manifest with config SHA-256 and effective physics parameters."""
	cfg_path = path or PHYSICS_CONFIG_PATH
	sha = hashlib.sha256(cfg_path.read_bytes()).hexdigest() if cfg_path.is_file() else None
	return {
		"config_path": str(cfg_path),
		"config_sha256": sha,
		"effective_spec": asdict(spec),
	}


def brick_spec_for_size(size: tuple[float, float, float], spec: PhysicsSpec | None = None) -> BrickSpec:
	"""Look up the :class:`BrickSpec` matching a brick's dimensions, scaling mass linearly if custom."""
	s = spec or DEFAULT_PHYSICS
	nx = round(size[0] / STUD_PITCH)
	ny = round(size[1] / STUD_PITCH)
	a, b = sorted((nx, ny))
	for key in (f"{nx}x{ny}", f"{a}x{b}"):
		if key in s.bricks:
			return s.bricks[key]
	base = s.bricks["2x2"]
	vol_ratio = (size[0] * size[1] * size[2]) / (base.size_m[0] * base.size_m[1] * base.size_m[2])
	return BrickSpec(
		f"{nx}x{ny}", size, base.mass_kg * vol_ratio, base.static_friction, base.dynamic_friction, base.restitution
	)


@dataclass(frozen=True)
class NonIdealityCfg:
	"""Optional actuator and encoder non-idealities for sensitivity testing."""

	latency_frames: int = 0
	backlash_rad: float = 0.0
	velocity_limit_rad_s: float | None = None
	encoder_ticks: int = 0
	joint_noise_std_rad: float = 0.0


@dataclass
class ActuatorCommandFilter:
	"""Stateful command delay, velocity clamp, backlash deadband, and encoder quantization filter."""

	cfg: NonIdealityCfg
	n_joints: int = 6
	fps: int = FPS
	_queue: deque[np.ndarray] = field(init=False, repr=False)
	_prev_out: np.ndarray = field(init=False, repr=False)
	_backlash_pos: np.ndarray = field(init=False, repr=False)

	def __post_init__(self) -> None:
		self.reset(np.zeros(self.n_joints, dtype=np.float64))

	def reset(self, q0: np.ndarray) -> None:
		"""Initialise buffers at starting joint angles ``q0``."""
		q = np.asarray(q0, dtype=np.float64).copy()
		self._queue = deque(
			[q.copy() for _ in range(max(0, self.cfg.latency_frames))], maxlen=max(1, self.cfg.latency_frames)
		)
		self._prev_out = q.copy()
		self._backlash_pos = q.copy()

	def step_command(self, cmd: np.ndarray, q_actual: np.ndarray) -> np.ndarray:
		"""Filter one control-frame target command through latency, velocity limit, and backlash deadband."""
		del q_actual
		c = np.asarray(cmd, dtype=np.float64).copy()
		if self.cfg.latency_frames > 0:
			delayed = self._queue[0].copy()
			self._queue.append(c)
			c = delayed

		if self.cfg.velocity_limit_rad_s is not None and self.cfg.velocity_limit_rad_s > 0.0:
			max_step = self.cfg.velocity_limit_rad_s / float(self.fps)
			c = self._prev_out + np.clip(c - self._prev_out, -max_step, max_step)

		if self.cfg.backlash_rad > 0.0:
			half = 0.5 * self.cfg.backlash_rad
			diff = c - self._backlash_pos
			self._backlash_pos = np.where(
				diff > half,
				c - half,
				np.where(diff < -half, c + half, self._backlash_pos),
			)
			c = self._backlash_pos.copy()

		self._prev_out = c.copy()
		return c

	def observe_joints(self, q_actual: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
		"""Apply optional Gaussian read noise and STS3215 encoder quantization to joint observations."""
		q = np.asarray(q_actual, dtype=np.float64).copy()
		if self.cfg.joint_noise_std_rad > 0.0 and rng is not None:
			q = q + rng.normal(0.0, self.cfg.joint_noise_std_rad, size=q.shape)
		if self.cfg.encoder_ticks > 0:
			step = 2.0 * math.pi / float(self.cfg.encoder_ticks)
			q = np.round(q / step) * step
		return q
