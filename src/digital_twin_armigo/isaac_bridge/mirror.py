"""MuJoCo mirror of a native Isaac episode: the six-face agent's skills on Isaac PhysX (plan 2026-11-24, section 四).

:class:`IsaacMirrorCell` is a :class:`~digital_twin_armigo.mujoco_cell.Cell` whose physics is Isaac. Each control
frame (``PhysicsEpisode.step`` → :meth:`step_with_telemetry`) sends the filtered joint targets to the bridge server,
which advances PhysX one 30 Hz frame; the returned arm joints and brick pose are written into the MuJoCo model
(``mj_forward``), so planning, the expert's observation, the per-frame log, the evaluator and the scoring-only face
visibility all read the Isaac state through the same code as on MuJoCo. Camera images (``rgb`` and the top depth)
are rendered by Isaac on demand, so the detector and the realistic perception see RTX images of the Isaac scene.

What is mirrored and what is not:

- jaw contact forces come from the Isaac contact sensors (``jaw_force`` / ``jaw_normal``); the tangential part and
  the force on other robot links are computed by MuJoCo on the mirrored poses (an approximation);
- actuator torques are Isaac's applied joint efforts, repeated over the 8 substeps;
- native checkpoints (``engine_snapshot`` / ``engine_restore``) exist only where the bridge allows them: gripper
  open and the brick resting on the table. ``TwinMujoco`` offers rewinds to those checkpoints only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..mujoco_cell import Cell
from ..mujoco_cell._mj import mujoco
from .client import BridgeError, IsaacBridge, scene_payload

_BRIDGES: dict[str, IsaacBridge] = {}


def shared_bridge(log_dir=None) -> IsaacBridge:
	"""One Isaac server per process, reused across episodes (building the Isaac app takes about a minute)."""
	bridge = _BRIDGES.get("default")
	if bridge is None:
		bridge = IsaacBridge(headless=True, log_dir=log_dir)
		_BRIDGES["default"] = bridge
	return bridge


class IsaacMirrorCell(Cell):
	engine = "isaac"

	def __init__(self, layout, menagerie, *args, bridge: IsaacBridge, **kwargs):
		super().__init__(layout, menagerie, *args, **kwargs)
		self.bridge = bridge
		self.isaac_frame = 0
		self.branch = 0
		self._effort = np.zeros(6)
		self._jaw = (np.zeros(2), np.zeros(2))
		self._apply(bridge.reset(scene_payload(layout, self)))

	# -- state mirroring --

	def _apply(self, state: dict) -> None:
		m, d = self.model, self.data
		q = np.asarray(state["q"], dtype=np.float64)
		qd = np.asarray(state["qd"], dtype=np.float64)
		d.qpos[self.qadr] = q
		for i, dof in enumerate(self.dofadr):
			d.qvel[dof] = qd[i]
		adr, vadr = self.brick_qadr, self.brick_dofadr
		pos = np.asarray(state["brick_pos"], dtype=np.float64)
		x, y, z, w = (float(v) for v in state["brick_quat_xyzw"])
		d.qpos[adr : adr + 3] = pos
		d.qpos[adr + 3 : adr + 7] = (w, x, y, z)
		d.qvel[vadr : vadr + 3] = np.asarray(state["brick_linvel"], dtype=np.float64)
		# MuJoCo free-joint angular velocity is in the body frame; Isaac reports it in the world frame.
		rot = np.zeros(9)
		mujoco.mju_quat2Mat(rot, np.array([w, x, y, z]))
		d.qvel[vadr + 3 : vadr + 6] = rot.reshape(3, 3).T @ np.asarray(state["brick_angvel"], dtype=np.float64)
		mujoco.mj_forward(m, d)
		self._effort = np.asarray(state.get("effort", np.zeros(6)), dtype=np.float64)
		self._jaw = (
			np.asarray(state.get("jaw_force", np.zeros(2)), dtype=np.float64),
			np.asarray(state.get("jaw_normal", np.zeros(2)), dtype=np.float64),
		)

	def step(self, ctrl: np.ndarray) -> None:
		self.step_with_telemetry(ctrl)

	def step_with_telemetry(self, ctrl: np.ndarray, substeps: int = 8) -> dict[str, np.ndarray]:
		self.data.ctrl[:6] = ctrl
		self._apply(self.bridge.step(np.asarray(ctrl, dtype=np.float64), branch=self.branch, frame=self.isaac_frame))
		self.isaac_frame += 1
		jaw_total, jaw_normal = self._jaw
		tangential = np.sqrt(np.maximum(jaw_total**2 - jaw_normal**2, 0.0))
		unintended = self.unintended_force()
		return {
			"jaw_total": np.tile(jaw_total, (substeps, 1)),
			"jaw_normal": np.tile(jaw_normal, (substeps, 1)),
			"jaw_tangential": np.tile(tangential, (substeps, 1)),
			"unintended": np.full(substeps, unintended),
			"torques": np.tile(self._effort, (substeps, 1)),
		}

	def actuator_torques(self) -> np.ndarray:
		return self._effort.copy()

	# -- sensors: Isaac RTX on demand --

	def rgb(self, camera: str, out: np.ndarray | None = None) -> np.ndarray:
		image = np.asarray(self.bridge.render(camera)["rgb"], dtype=np.uint8)
		if out is not None:
			out[...] = image
			return out
		return image

	def depth(self, camera: str) -> np.ndarray:
		depth = np.asarray(self.bridge.render(camera, depth=True)["depth"], dtype=np.float64)
		# No hit: the far plane, as the MuJoCo depth render reports background (the perception expects finite depth).
		return np.where(np.isfinite(depth) & (depth > 0), depth, 10.0)

	# -- native checkpoints --

	def engine_snapshot(self) -> str | None:
		try:
			return self.bridge.checkpoint()
		except BridgeError:
			return None

	def engine_restore(self, token: str) -> None:
		self.branch += 1
		self._apply(self.bridge.restore(token))

	def save_engine_replay(self, directory: Path) -> None:
		self.bridge.save_replay(directory)


def mirror_cell_factory(*, log_dir=None):
	"""``cell_factory`` for ``TwinMujoco`` / ``TwinService.run``: an Isaac-backed mirror on the shared bridge."""

	def factory(layout, menagerie, *args, **kwargs):
		kwargs = {k: v for k, v in kwargs.items() if k != "dynamic_clutter"}
		return IsaacMirrorCell(layout, menagerie, *args, bridge=shared_bridge(log_dir), dynamic_clutter=False, **kwargs)

	return factory
