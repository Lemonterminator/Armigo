"""One pure-physics episode on the MuJoCo cell: 240 Hz stepping, per-frame telemetry and the observation a policy reads.

``PhysicsEpisode`` owns the stepping and the per-frame telemetry the audited evaluator reads (the open-loop benchmark
in ``scripts/bench_cross_engine.py`` replays its joint targets through it). :func:`observe` builds the per-frame
observation dictionary a closed-loop policy consumes.
"""

from __future__ import annotations

import math

import numpy as np

from ..constants import FPS
from ..physics import DEFAULT_PHYSICS, ActuatorCommandFilter, NonIdealityCfg
from ._mj import mujoco
from .cell import Cell

PHYSICS_HZ = 240
SUBSTEPS = PHYSICS_HZ // FPS  # 8 substeps @ 240 Hz per 30 Hz frame
WARMUP_STEPS = 12


def jaw_env_contact_force(cell: Cell) -> float:
	"""Peak contact force (N) between a jaw geom and anything that is neither the robot nor the target brick
	(table, bowls, clutter). Reads the current contacts only. The evaluator's unintended-collision check excludes
	the jaws, so this is the only record of jaws pressing on the environment."""
	jaws = cell.fixed_jaw_geoms | cell.moving_jaw_geoms
	brick = {int(g) for g in cell.brick_geoms}
	peak = 0.0
	f = np.zeros(6)
	for cid in range(cell.data.ncon):
		con = cell.data.contact[cid]
		g1, g2 = int(con.geom1), int(con.geom2)
		other = g2 if g1 in jaws else (g1 if g2 in jaws else -1)
		if other < 0 or other in cell.robot_all_geoms or other in brick:
			continue
		mujoco.mj_contactForce(cell.model, cell.data, cid, f)
		peak = max(peak, float(np.linalg.norm(f[:3])))
	return peak


class PhysicsEpisode:
	"""Reset the cell for one pure-physics episode, step it one 30 Hz frame at a time and log telemetry.

	``telemetry()`` returns the arrays the audited evaluator consumes (see ``_evaluate_mj_rollout``): one row per
	frame, sampled after the frame's substeps.
	"""

	def __init__(
		self,
		cell: Cell,
		q_start: np.ndarray,
		brick_xy: tuple[float, float] = (0.26, 0.0),
		brick_yaw_deg: float = 0.0,
		non_ideal: NonIdealityCfg | None = None,
	) -> None:
		self.cell = cell
		cell.model.opt.timestep = 1.0 / PHYSICS_HZ
		cell.set_joints(q_start)
		cell.data.qvel[:] = 0.0
		yaw_rad = math.radians(brick_yaw_deg)
		b_quat = (0.0, 0.0, math.sin(yaw_rad / 2.0), math.cos(yaw_rad / 2.0))
		cell.set_brick_pose(
			np.array([brick_xy[0], brick_xy[1], cell.layout.brick_size[2] / 2.0 + 0.0005]), np.array(b_quat)
		)
		cell.data.ctrl[:6] = q_start
		for _ in range(WARMUP_STEPS):
			mujoco.mj_step(cell.model, cell.data)

		self.filt = ActuatorCommandFilter(non_ideal, n_joints=6, fps=FPS) if non_ideal is not None else None
		if self.filt is not None:
			self.filt.reset(q_start)
		# Peak jaw ↔ environment contact force per frame (end-of-frame contacts); kept out of ``telemetry()``,
		# whose keys are fixed by the evaluator contract.
		self.jaw_env_force: list[float] = []
		self._logs: dict[str, list] = {
			k: []
			for k in (
				"q_cmd",
				"qpos",
				"tcp_pos",
				"ee_quat",
				"brick_pos",
				"brick_quat",
				"brick_vel",
				"brick_ang_vel",
				"jaw_forces",
				"pinch_normal",
				"pinch_tangential",
				"unintended_forces",
				"substep_unintended",
				"actuator_torques",
			)
		}

	@property
	def n_frames(self) -> int:
		return len(self._logs["q_cmd"])

	def step(self, target: np.ndarray) -> None:
		"""Advance one 30 Hz frame towards the 6-D joint position ``target`` (through the command filter)."""
		cell = self.cell
		q_cmd = self.filt.step_command(target, cell.joints()) if self.filt is not None else target
		sub = cell.step_with_telemetry(q_cmd, substeps=SUBSTEPS)
		log = self._logs
		log["q_cmd"].append(q_cmd.copy())
		log["qpos"].append(cell.joints())
		log["tcp_pos"].append(cell.tcp_pos())
		wxyz = cell.data.xquat[cell.grip_body]
		log["ee_quat"].append(np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]], dtype=np.float64))
		log["brick_pos"].append(cell.brick_pos())
		log["brick_quat"].append(cell.brick_quat_xyzw())
		lv, av = cell.brick_vel()
		log["brick_vel"].append(lv)
		log["brick_ang_vel"].append(av)
		log["jaw_forces"].append(sub["jaw_total"][-1])
		log["pinch_normal"].append(sub["jaw_normal"].mean(axis=1))
		log["pinch_tangential"].append(sub["jaw_tangential"].mean(axis=1))
		log["unintended_forces"].append(float(sub["unintended"][-1]))
		log["substep_unintended"].append(sub["unintended"])
		log["actuator_torques"].append(sub["torques"][np.argmax(np.abs(sub["torques"]), axis=0), np.arange(6)])
		self.jaw_env_force.append(jaw_env_contact_force(cell))
		observer = getattr(cell, "frame_observer", None)
		if observer is not None:
			observer(self)

	def last(self, key: str):
		"""The most recent logged row of ``key``."""
		return self._logs[key][-1]

	def telemetry(self) -> dict[str, np.ndarray]:
		out = {"t": np.arange(self.n_frames, dtype=np.float64) / float(FPS)}
		out.update({k: np.array(v) for k, v in self._logs.items()})
		return out


def observe(cell: Cell, episode: PhysicsEpisode) -> dict[str, np.ndarray | float]:
	"""The per-frame observation a closed-loop policy reads: joints, gripper frame, brick pose and velocity, per-jaw
	normal force and the gripper command / effort the :class:`~.gripper_state.GripperMonitor` needs."""
	q = cell.joints()
	lv, _ = cell.brick_vel()
	jaw = episode.last("pinch_normal") if episode.n_frames else np.zeros(2)
	effort_limit = (cell.physics or DEFAULT_PHYSICS).gripper_actuator.effort_limit
	gripper_cmd = float(episode.last("q_cmd")[5]) if episode.n_frames else float(q[5])
	effort_frac = abs(float(episode.last("actuator_torques")[5])) / effort_limit if episode.n_frames else 0.0
	return {
		"q": q,
		"tcp": cell.tcp_pos(),
		"brick_pos": cell.brick_pos(),
		"brick_quat": cell.brick_quat_xyzw(),
		"brick_vel": lv,
		"jaw_normal": np.asarray(jaw, dtype=np.float64),
		"gripper_cmd": gripper_cmd,
		"effort_frac": effort_frac,
	}
