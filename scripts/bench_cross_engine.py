"""Audited cross-engine physics benchmark: MuJoCo vs Isaac Sim (PhysX).

Separates two distinct classes of evidence under unified ``PhysicsSpec`` (``v1-nominal-assumed``)
and audited evaluator (``v2-audited``):
1. ``dynamics_comparison``: identical initial states and identical timestamped open-loop control
   sequences across ``joint_tracking``, ``flick_slide``, ``spin_release``, and
   ``shared_pick_place_replay``, comparing full trajectories, time/event alignment, terminal
   position vector diffs, substep contact peaks/impulses, and evaluator verdicts.
2. ``skill_transferability``: engine-specific adapters solving the same high-level tasks across
   sensitivity conditions (search set vs held-out combined perturbations) with verdict and
   reason-code distributions plus binomial confidence intervals.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from digital_twin_armigo.benchmarks import (
	HELD_OUT_PERTURBATIONS,
	SENSITIVITY_MATRIX,
	build_flick_trajectory,
	build_spin_release_trajectory,
	joint_tracking_trajectory,
	summarize_flick_experiment,
	summarize_spin_release_experiment,
	wilson_ci,
)
from digital_twin_armigo.camera import CameraPose
from digital_twin_armigo.constants import BOWL_R, DUPLO_2X2, DUPLO_2X3, DUPLO_2X4, FPS, JOINTS, REST_U, SO101_MJCF_DIR
from digital_twin_armigo.joints import normalized_to_rad
from digital_twin_armigo.metrics import EVALUATOR_VERSION, compare_trajectories, evaluate_episode_health
from digital_twin_armigo.physics import (
	NonIdealityCfg,
	load_physics_spec,
	physics_manifest,
)

PHYSICS_HZ = 240
SUBSTEPS = PHYSICS_HZ // FPS  # 8 substeps @ 240 Hz per 30 Hz frame
TCP_OFFSET_MJCF_SITE = (0.012, -0.000218, -0.098127)
OPEN_G = 0.85
HOLD_G = 0.06
BRICK_SIZES = {"2x2": DUPLO_2X2, "2x3": DUPLO_2X3, "2x4": DUPLO_2X4}
GREEN_XY = (0.22, 0.19)
PINK_XY = (0.26, -0.18)


def _git_head() -> str | None:
	try:
		out = subprocess.run(
			["git", "-C", str(ROOT), "rev-parse", "HEAD"],
			check=True,
			capture_output=True,
			text=True,
		)
		return out.stdout.strip()
	except (subprocess.SubprocessError, OSError):
		return None


def _mj_brick_layout(
	brick_xy: tuple[float, float] = (0.26, 0.0),
	yaw_deg: float = 0.0,
	size: tuple[float, float, float] = DUPLO_2X2,
	clutter_per_bowl: int = 0,
	seed: int = 0,
):
	from digital_twin_armigo.mujoco_cell import Layout

	rng = np.random.default_rng(seed)
	clutter = []
	for bowl in (GREEN_XY, PINK_XY):
		for _ in range(clutter_per_bowl):
			r = BOWL_R * 0.55 * math.sqrt(float(rng.random()))
			a = float(rng.uniform(0.0, 2.0 * math.pi))
			clutter.append(
				(
					bowl[0] + r * math.cos(a),
					bowl[1] + r * math.sin(a),
					float(rng.uniform(0.0, math.pi)),
					0.1,
					0.42,
					0.80,
				)
			)
	return Layout(
		green=GREEN_XY,
		pink=PINK_XY,
		brick_xy=brick_xy,
		brick_yaw_deg=yaw_deg,
		defective=True,
		brick_size=size,
		clutter=clutter,
	)


def _build_mj_pick_place_plan(spec, cond, top_cam: CameraPose, seed: int = 0):
	"""Solve MuJoCo IK plan for a pick-show-place condition (returns trajectory, phases, q_start)."""
	from digital_twin_armigo.mujoco_cell import (
		Cell,
		Grip,
		Plan,
		best_roll,
		expert_handoff,
		ik,
		ik_reach,
		release_point,
	)

	size = BRICK_SIZES[cond.brick_size]
	lay = _mj_brick_layout(
		brick_xy=cond.brick_xy,
		yaw_deg=cond.brick_yaw_deg,
		size=size,
		clutter_per_bowl=cond.clutter_per_bowl,
		seed=seed,
	)
	lay.top = top_cam
	cell = Cell(
		lay,
		SO101_MJCF_DIR,
		physics=spec,
		dynamic_clutter=True,
		assisted_grasp=False,
		mass_scale=cond.mass_scale,
		friction_scale=cond.friction_scale,
		render_enabled=False,
	)
	q_start = normalized_to_rad(np.array(REST_U, dtype=np.float64))
	q_start[5] = OPEN_G
	cell.set_joints(q_start)

	bx, by = lay.brick_xy
	yaw = math.radians(lay.brick_yaw_deg)
	short_axis = np.array([-math.sin(yaw), math.cos(yaw)])
	q_hover = ik(cell, np.array([bx, by, 0.09]), q_start.copy(), 0.0)
	roll, sign = best_roll(cell, q_hover, short_axis)
	grip = Grip(roll, open_g=OPEN_G, closed_g=HOLD_G)
	offset = -sign * short_axis * (size[1] / 2.0 + 0.006)
	gx, gy = bx + offset[0], by + offset[1]
	z_grip = 0.006
	plan = Plan(q_start, 1.0)
	plan.hold(0.3, "rest")
	q = np.array([0.0, -0.35, 0.25, 1.35, 0.0, grip.open_g], dtype=np.float64)
	plan.move(q, 0.8, "approach")
	for x, y, z, g, secs, phase in (
		(gx, gy, 0.09, grip.open_g, 1.4, "approach"),
		(gx, gy, z_grip + 0.02, grip.open_g, 1.2, "approach"),
		(gx, gy, z_grip, grip.open_g, 0.7, "grasp"),
		(gx, gy, z_grip, grip.closed_g, 1.0, "grasp"),
		(gx, gy, 0.12, grip.closed_g, 1.2, "lift"),
	):
		q = ik(cell, np.array([x, y, z]), q, roll)
		q[5] = g
		plan.move(q, secs, phase)
	grip.q_lift = q.copy()

	for wr_deg in (-35.0, 0.0, 35.0):
		q_show = q.copy()
		q_show[3] = 0.65
		q_show[4] = math.radians(wr_deg)
		q_show[5] = HOLD_G
		plan.move(q_show, 0.65, "show_move", settle=0.0)
		plan.hold(0.35, "show_hold")
	expert_handoff(cell, plan, grip, np.random.default_rng(seed))

	bowl_xy = lay.pink if lay.defective else lay.green
	rp = release_point(bowl_xy)
	q_pl = grip.q_lift.copy()
	for z, g, secs in ((0.12, HOLD_G, 1.8), (0.065, HOLD_G, 1.2), (0.065, OPEN_G, 0.6), (0.12, OPEN_G, 0.8)):
		q_pl = ik_reach(cell, np.array([rp[0], rp[1], z]), q_pl, grip.roll)
		q_pl[5] = g
		plan.move(q_pl, secs, "place")
	plan.move(q_start.copy(), 1.2, "return")
	plan.hold(0.9, "return")
	cell.close()
	return np.array(plan.frames, dtype=np.float64), np.array(plan.phases, dtype=np.uint8), q_start, bowl_xy


def _mj_step_trajectory(
	cell,
	q_targets: np.ndarray,
	q_start: np.ndarray,
	brick_xy: tuple[float, float] = (0.26, 0.0),
	brick_yaw_deg: float = 0.0,
	non_ideal: NonIdealityCfg | None = None,
) -> dict[str, np.ndarray]:
	"""Run an open-loop joint target sequence in pure-physics MuJoCo at 240 Hz with substep telemetry."""
	from digital_twin_armigo.mujoco_cell.physics_expert import PhysicsEpisode

	episode = PhysicsEpisode(cell, q_start, brick_xy=brick_xy, brick_yaw_deg=brick_yaw_deg, non_ideal=non_ideal)
	for target in q_targets:
		episode.step(target)
	return episode.telemetry()


def _evaluate_mj_rollout(res: dict[str, np.ndarray], phases: np.ndarray, spec, size, bowl_xy) -> dict[str, Any]:
	torque_lims = np.array([spec.arm_actuator.effort_limit] * 5 + [spec.gripper_actuator.effort_limit])
	health = evaluate_episode_health(
		q_cmd=res["q_cmd"],
		q_act=res["qpos"],
		tcp_pos=res["tcp_pos"],
		brick_pos=res["brick_pos"],
		brick_quat=res["brick_quat"],
		jaw_forces=res["jaw_forces"],
		unintended_forces=res["unintended_forces"],
		phases=phases,
		target_bowl_xy=bowl_xy,
		bowl_r=BOWL_R,
		brick_vel=res["brick_vel"],
		brick_ang_vel=res["brick_ang_vel"],
		ee_quat=res["ee_quat"],
		fps=FPS,
		brick_size=size,
		substep_unintended_forces=res["substep_unintended"],
		pinch_normal_forces=res["pinch_normal"],
		pinch_tangential_forces=res["pinch_tangential"],
		actuator_torques=res["actuator_torques"],
		torque_limits=torque_lims,
		torque_source="solver_measured",
		timestamps=res["t"],
	)
	return health.to_dict()


def _mj_jitter_condition(cond, seed: int):
	"""``cond`` with the small reproducible per-seed brick pose jitter (none for seed 0)."""
	rng = np.random.default_rng(seed)
	# Small reproducible pose jitter per seed when seed > 0 to span search & held-out placement sensitivity
	dx = float(rng.normal(0.0, 0.008)) if seed > 0 else 0.0
	dy = float(rng.normal(0.0, 0.008)) if seed > 0 else 0.0
	dyaw = float(rng.normal(0.0, 6.0)) if seed > 0 else 0.0
	return type(cond)(
		name=cond.name,
		brick_size=cond.brick_size,
		brick_xy=(cond.brick_xy[0] + dx, cond.brick_xy[1] + dy),
		brick_yaw_deg=cond.brick_yaw_deg + dyaw,
		defective=cond.defective,
		clutter_per_bowl=cond.clutter_per_bowl,
		mass_scale=cond.mass_scale,
		friction_scale=cond.friction_scale,
		latency_frames=cond.latency_frames,
		backlash_deg=cond.backlash_deg,
	)


def _mj_make_cell(spec, cond_j, top_cam: CameraPose, seed: int = 0):
	"""Pure-physics ``Cell`` for an already jittered condition (assisted grasp off, the condition's scales)."""
	from digital_twin_armigo.mujoco_cell import Cell

	size = BRICK_SIZES[cond_j.brick_size]
	lay = _mj_brick_layout(
		brick_xy=cond_j.brick_xy,
		yaw_deg=cond_j.brick_yaw_deg,
		size=size,
		clutter_per_bowl=cond_j.clutter_per_bowl,
		seed=seed,
	)
	lay.top = top_cam
	return Cell(
		lay,
		SO101_MJCF_DIR,
		physics=spec,
		dynamic_clutter=True,
		assisted_grasp=False,
		mass_scale=cond_j.mass_scale,
		friction_scale=cond_j.friction_scale,
		render_enabled=False,
	)


def _mj_non_ideal(cond_j, spec) -> NonIdealityCfg:
	"""Actuator non-idealities of a condition (latency, backlash, velocity limit, encoder resolution)."""
	return NonIdealityCfg(
		latency_frames=cond_j.latency_frames,
		backlash_rad=math.radians(cond_j.backlash_deg),
		velocity_limit_rad_s=spec.arm_actuator.velocity_limit,
		encoder_ticks=4096,
	)


def _mj_run_matrix_condition(cond, spec, top_cam: CameraPose, seed: int = 0) -> dict[str, Any]:
	"""Run one pure-physics pick-show-place episode in MuJoCo for ``cond`` and ``seed``."""
	cond_j = _mj_jitter_condition(cond, seed)
	q_plan, phases, q_start, bowl_xy = _build_mj_pick_place_plan(spec, cond_j, top_cam, seed=seed)
	size = BRICK_SIZES[cond_j.brick_size]
	cell = _mj_make_cell(spec, cond_j, top_cam, seed=seed)
	res = _mj_step_trajectory(
		cell,
		q_plan,
		q_start,
		brick_xy=cond_j.brick_xy,
		brick_yaw_deg=cond_j.brick_yaw_deg,
		non_ideal=_mj_non_ideal(cond_j, spec),
	)
	cell.close()
	return _evaluate_mj_rollout(res, phases, spec, size, bowl_xy)


def _quat_yaw(quat_xyzw: np.ndarray) -> np.ndarray:
	from scipy.spatial.transform import Rotation

	return np.array([Rotation.from_quat(q).as_euler("xyz")[2] for q in quat_xyzw], dtype=np.float64)


def _run_isaac_in_context(
	sim_cfg,
	spec,
	q_track: np.ndarray,
	q_flick: np.ndarray,
	q_spin: np.ndarray,
	q_pick_place: np.ndarray,
	q_start: np.ndarray,
) -> tuple[dict[str, dict[str, np.ndarray]], str]:
	import isaaclab.sim as sim_utils
	import isaaclab.utils.math as math_utils
	import torch
	from isaaclab.actuators import ImplicitActuatorCfg
	from isaaclab.assets import ArticulationCfg, RigidObjectCfg
	from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
	from isaaclab.sensors import ContactSensorCfg
	from isaaclab.utils import configclass

	arm_act = spec.arm_actuator
	grip_act = spec.gripper_actuator
	bspec = spec.bricks["2x2"]
	hw_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

	@configclass
	class BenchSceneCfg(InteractiveSceneCfg):
		table = RigidObjectCfg(
			prim_path="{ENV_REGEX_NS}/Table",
			spawn=sim_utils.CuboidCfg(
				size=(1.6, 1.6, 0.04),
				rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
				collision_props=sim_utils.UsdPhysicsCollisionCfg(),
				physics_material=sim_utils.RigidBodyMaterialCfg(
					static_friction=spec.table.static_friction,
					dynamic_friction=spec.table.dynamic_friction,
					restitution=spec.table.restitution,
					friction_combine_mode="max",
					restitution_combine_mode="max",
				),
				visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.58, 0.58, 0.58), roughness=0.9),
			),
			init_state=RigidObjectCfg.InitialStateCfg(pos=(0.25, 0.0, -0.02)),
		)
		green_plate = RigidObjectCfg(
			prim_path="{ENV_REGEX_NS}/GreenPlate",
			spawn=sim_utils.UsdFileCfg(usd_path=str(ROOT / "assets" / "props" / "green_plate.usda")),
			init_state=RigidObjectCfg.InitialStateCfg(pos=(GREEN_XY[0], GREEN_XY[1], 0.0)),
		)
		pink_plate = RigidObjectCfg(
			prim_path="{ENV_REGEX_NS}/PinkPlate",
			spawn=sim_utils.UsdFileCfg(usd_path=str(ROOT / "assets" / "props" / "pink_plate.usda")),
			init_state=RigidObjectCfg.InitialStateCfg(pos=(PINK_XY[0], PINK_XY[1], 0.0)),
		)
		robot = ArticulationCfg(
			prim_path="{ENV_REGEX_NS}/Robot",
			spawn=sim_utils.UsdFileCfg(
				usd_path=str(ROOT / "assets" / "so101_usd" / "so101_physx.usda"),
				activate_contact_sensors=True,
				articulation_props=sim_utils.PhysxArticulationRootPropertiesCfg(
					solver_position_iteration_count=32,
					solver_velocity_iteration_count=8,
				),
			),
			actuators={
				"sts3215_arm": ImplicitActuatorCfg(
					joint_names_expr=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
					stiffness=arm_act.stiffness,
					damping=arm_act.damping,
					effort_limit_sim=arm_act.effort_limit,
					velocity_limit_sim=arm_act.velocity_limit,
					armature=arm_act.armature,
					friction=arm_act.frictionloss,
					dynamic_friction=arm_act.frictionloss,
					viscous_friction=arm_act.joint_damping,
				),
				"sts3215_gripper": ImplicitActuatorCfg(
					joint_names_expr=["gripper"],
					stiffness=grip_act.stiffness,
					damping=grip_act.damping,
					effort_limit_sim=grip_act.effort_limit,
					velocity_limit_sim=grip_act.velocity_limit,
					armature=grip_act.armature,
					friction=grip_act.frictionloss,
					dynamic_friction=grip_act.frictionloss,
					viscous_friction=grip_act.joint_damping,
				),
			},
		)
		brick = RigidObjectCfg(
			prim_path="{ENV_REGEX_NS}/Brick",
			spawn=sim_utils.UsdFileCfg(
				usd_path=str(ROOT / "assets" / "props" / "duplo_2x2_defective.usda"),
				activate_contact_sensors=True,
				mass_props=sim_utils.MassPropertiesCfg(mass=bspec.mass_kg),
				rigid_props=sim_utils.PhysxRigidBodyPropertiesCfg(
					solver_position_iteration_count=32,
					solver_velocity_iteration_count=8,
					linear_damping=0.05,
					angular_damping=0.05,
					max_depenetration_velocity=1.0,
				),
			),
			init_state=RigidObjectCfg.InitialStateCfg(pos=(0.26, 0.0, bspec.size_m[2] / 2.0 + 0.0005)),
		)
		contact_fixed = ContactSensorCfg(
			prim_path="{ENV_REGEX_NS}/Robot/Geometry/base/shoulder/upper_arm/lower_arm/wrist/gripper",
			filter_prim_paths_expr=["{ENV_REGEX_NS}/Brick"],
			update_period=0.0,
			history_length=SUBSTEPS,
			track_friction_forces=True,
		)
		contact_moving = ContactSensorCfg(
			prim_path="{ENV_REGEX_NS}/Robot/Geometry/base/shoulder/upper_arm/lower_arm/wrist/gripper/moving_jaw_so101_v1",
			filter_prim_paths_expr=["{ENV_REGEX_NS}/Brick"],
			update_period=0.0,
			history_length=SUBSTEPS,
			track_friction_forces=True,
		)
		contact_arm_table = ContactSensorCfg(
			prim_path="{ENV_REGEX_NS}/Robot/Geometry/base/shoulder/upper_arm/lower_arm/wrist",
			filter_prim_paths_expr=[
				"{ENV_REGEX_NS}/Table",
				"{ENV_REGEX_NS}/Brick",
				"{ENV_REGEX_NS}/GreenPlate",
				"{ENV_REGEX_NS}/PinkPlate",
			],
			update_period=0.0,
			history_length=SUBSTEPS,
			track_friction_forces=True,
		)

	sim = sim_utils.SimulationContext(sim_cfg)
	scene = InteractiveScene(BenchSceneCfg(num_envs=1, env_spacing=2.0))
	sim.reset()

	robot = scene["robot"]
	brick = scene["brick"]
	c_fixed = scene["contact_fixed"]
	c_moving = scene["contact_moving"]
	c_wrist_tbl = scene["contact_arm_table"]
	joint_names = list(robot.joint_names)
	j_idx = {n: i for i, n in enumerate(joint_names)}
	ee_body_id = robot.find_bodies("gripper")[0][0]
	offset_pos = torch.tensor([TCP_OFFSET_MJCF_SITE], device=robot.device, dtype=torch.float32)
	offset_rot = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=robot.device, dtype=torch.float32)

	def _ordered_to_sim(q_ord: np.ndarray) -> torch.Tensor:
		t = torch.zeros((1, len(joint_names)), device=robot.device, dtype=torch.float32)
		for i, n in enumerate(JOINTS):
			t[0, j_idx[n]] = float(q_ord[i])
		return t

	def _sim_to_ordered(t: torch.Tensor) -> np.ndarray:
		arr = t[0].cpu().numpy()
		return np.array([arr[j_idx[n]] for n in JOINTS], dtype=np.float64)

	def _tcp_pose() -> tuple[np.ndarray, np.ndarray]:
		ee_p = robot.data.body_pos_w.torch[:, ee_body_id]
		ee_q = robot.data.body_quat_w.torch[:, ee_body_id]
		rp = robot.data.root_pos_w.torch
		rq = robot.data.root_quat_w.torch
		eb, eqb = math_utils.subtract_frame_transforms(rp, rq, ee_p, ee_q)
		tp, tq = math_utils.combine_frame_transforms(eb, eqb, offset_pos, offset_rot)
		return tp[0].cpu().numpy().astype(np.float64), tq[0].cpu().numpy().astype(np.float64)

	def _substep_jaw_pair(sensor) -> tuple[np.ndarray, np.ndarray]:
		nh = getattr(sensor.data, "normal_force_matrix_w_history", None)
		fh = getattr(sensor.data, "friction_force_matrix_w_history", None)
		th = getattr(sensor.data, "force_matrix_w_history", None)
		if nh is not None:
			nv = torch.linalg.norm(nh.torch[0, :, 0, 0], dim=-1).cpu().numpy().astype(np.float64)
		elif th is not None:
			nv = torch.linalg.norm(th.torch[0, :, 0, 0], dim=-1).cpu().numpy().astype(np.float64)
		else:
			nv = np.full(SUBSTEPS, float(torch.linalg.norm(sensor.data.force_matrix_w.torch[0, 0, 0]).cpu()))
		if fh is not None:
			tv = torch.linalg.norm(fh.torch[0, :, 0, 0], dim=-1).cpu().numpy().astype(np.float64)
		else:
			tv = np.zeros(SUBSTEPS, dtype=np.float64)
		return nv, tv

	def _substep_unintended() -> tuple[float, np.ndarray]:
		uh = getattr(c_wrist_tbl.data, "force_matrix_w_history", None)
		if uh is not None:
			norms = torch.linalg.norm(uh.torch[0, :, 0, :], dim=-1).max(dim=-1).values
			arr = norms.cpu().numpy().astype(np.float64)
			return float(arr[0]), arr
		val = float(torch.linalg.norm(c_wrist_tbl.data.force_matrix_w.torch[0, 0, :]).cpu())
		return val, np.full(SUBSTEPS, val, dtype=np.float64)

	def _run_one(
		q_targets: np.ndarray,
		brick_xy: tuple[float, float],
		brick_yaw_deg: float = 0.0,
	) -> dict[str, np.ndarray]:
		q0_sim = _ordered_to_sim(q_start)
		robot.write_joint_position_to_sim_index(position=q0_sim)
		robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q0_sim))
		robot.actuators.target_command.set_position_index(value=q0_sim)
		yr = math.radians(brick_yaw_deg)
		b_state = torch.tensor(
			[
				[
					brick_xy[0],
					brick_xy[1],
					bspec.size_m[2] / 2.0 + 0.0005,
					0.0,
					0.0,
					math.sin(yr / 2.0),
					math.cos(yr / 2.0),
					0,
					0,
					0,
					0,
					0,
					0,
				]
			],
			device=robot.device,
			dtype=torch.float32,
		)
		brick.write_root_state_to_sim(b_state)
		for _ in range(12):
			scene.write_data_to_sim()
			sim.step()
			scene.update(sim.get_physics_dt())

		qcmd_log, qpos_log, tcp_log, eeq_log = [], [], [], []
		bpos_log, bquat_log, bvel_log, bang_log = [], [], [], []
		jaw_log, jaw_n_log, jaw_t_log, unint_log, sub_unint_log, tq_log = [], [], [], [], [], []

		for q_cmd in q_targets:
			t_cmd = _ordered_to_sim(q_cmd)
			robot.actuators.target_command.set_position_index(value=t_cmd)
			for _ in range(SUBSTEPS):
				scene.write_data_to_sim()
				sim.step()
				scene.update(sim.get_physics_dt())
			qcmd_log.append(q_cmd.copy())
			qpos_log.append(_sim_to_ordered(robot.data.joint_pos.torch))
			tp, tq = _tcp_pose()
			tcp_log.append(tp)
			eeq_log.append(tq)
			bpos_log.append(brick.data.root_pos_w.torch[0].cpu().numpy().astype(np.float64))
			bquat_log.append(brick.data.root_quat_w.torch[0].cpu().numpy().astype(np.float64))
			bvel_log.append(brick.data.root_lin_vel_w.torch[0].cpu().numpy().astype(np.float64))
			bang_log.append(brick.data.root_ang_vel_w.torch[0].cpu().numpy().astype(np.float64))
			jaw_log.append(
				(
					float(torch.linalg.norm(c_fixed.data.force_matrix_w.torch[0, 0, 0]).cpu()),
					float(torch.linalg.norm(c_moving.data.force_matrix_w.torch[0, 0, 0]).cpu()),
				)
			)
			fn0, ft0 = _substep_jaw_pair(c_fixed)
			fn1, ft1 = _substep_jaw_pair(c_moving)
			jaw_n_log.append(0.5 * (fn0 + fn1))
			jaw_t_log.append(0.5 * (ft0 + ft1))
			u_val, u_sub = _substep_unintended()
			unint_log.append(u_val)
			sub_unint_log.append(u_sub)
			tq_log.append(_sim_to_ordered(robot.data.applied_torque.torch))

		return {
			"t": np.arange(len(q_targets), dtype=np.float64) / float(FPS),
			"q_cmd": np.array(qcmd_log),
			"qpos": np.array(qpos_log),
			"tcp_pos": np.array(tcp_log),
			"ee_quat": np.array(eeq_log),
			"brick_pos": np.array(bpos_log),
			"brick_quat": np.array(bquat_log),
			"brick_vel": np.array(bvel_log),
			"brick_ang_vel": np.array(bang_log),
			"jaw_forces": np.array(jaw_log),
			"pinch_normal": np.array(jaw_n_log),
			"pinch_tangential": np.array(jaw_t_log),
			"unintended_forces": np.array(unint_log),
			"substep_unintended": np.array(sub_unint_log),
			"actuator_torques": np.array(tq_log),
		}

	results = {
		"joint_tracking": _run_one(q_track, (0.26, 0.0), 0.0),
		"flick_slide": _run_one(q_flick, (0.26, 0.0), 0.0),
		"spin_release": _run_one(q_spin, (0.26, 0.0), 0.0),
		"shared_pick_place_replay": _run_one(q_pick_place, (0.26, 0.01), 20.0),
	}
	return results, hw_name


def _summarize_mj_condition_batch(cond, spec, top_cam: CameraPose, seeds: int) -> dict[str, Any]:
	rows = [_mj_run_matrix_condition(cond, spec, top_cam, seed=s) for s in range(seeds)]
	verdict_counts = Counter(r["verdict"] for r in rows)
	reason_counts = Counter(code for r in rows for code in r["reason_codes"])
	n_ok = verdict_counts.get("SUCCESS", 0)
	ci_lo, ci_hi = wilson_ci(n_ok, len(rows))
	mean_slip_mm = float(np.mean([r["max_slip_pos_m"] * 1000.0 for r in rows if np.isfinite(r["max_slip_pos_m"])]))
	mean_pinch_n = float(np.mean([r["mean_pinch_force_N"] for r in rows]))
	max_unint_n = float(np.max([r["max_unintended_force_N"] for r in rows]))
	return {
		"condition": cond.to_dict(),
		"episodes": len(rows),
		"verdict_counts": dict(verdict_counts),
		"reason_counts": dict(reason_counts),
		"success_rate": n_ok / float(len(rows)),
		"wilson_95ci": [ci_lo, ci_hi],
		"mean_slip_mm": mean_slip_mm,
		"mean_pinch_force_N": mean_pinch_n,
		"max_unintended_force_N": max_unint_n,
		"seed0": rows[0],
	}


def main() -> None:
	parser = argparse.ArgumentParser(description="Audited cross-engine MuJoCo vs Isaac Sim verification suite.")
	parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "verification-physics-v2")
	parser.add_argument("--include-matrix", action="store_true", help="Run skill-transfer sensitivity matrix.")
	parser.add_argument("--mujoco-only", action="store_true", help="Run MuJoCo side only.")
	parser.add_argument("--matrix-seeds", type=int, default=30, help="Fixed seeds per MuJoCo matrix condition.")

	if "--mujoco-only" not in sys.argv:
		from isaaclab.app import add_launcher_args, launch_simulation

		add_launcher_args(parser)
		args = parser.parse_args()
		args.visualizer = []
	else:
		args = parser.parse_args()

	args.out_dir.mkdir(parents=True, exist_ok=True)
	spec = load_physics_spec()
	top_cam_raw = json.loads((ROOT / "config" / "sim" / "top_camera_joint.json").read_text())["camera"]
	top_cam = CameraPose(**top_cam_raw)
	q_start = normalized_to_rad(np.array(REST_U, dtype=np.float64))
	q_start[5] = OPEN_G

	def _build_shared_trajs():
		from digital_twin_armigo.mujoco_cell import Cell, ik

		cell_ik = Cell(_mj_brick_layout(), SO101_MJCF_DIR, physics=spec, assisted_grasp=False, render_enabled=False)
		q_track = joint_tracking_trajectory(fps=FPS)
		q_flick, strike_idx = build_flick_trajectory(
			lambda t, q0, r: ik(cell_ik, t, q0, r), q_start, (0.26, 0.0), fps=FPS
		)
		q_spin, release_idx, _ = build_spin_release_trajectory(
			lambda t, q0, r: ik(cell_ik, t, q0, r), q_start, (0.26, 0.0), OPEN_G, HOLD_G, fps=FPS
		)
		cell_ik.close()
		q_pp, phases_pp, _, bowl_pp = _build_mj_pick_place_plan(spec, SENSITIVITY_MATRIX[0], top_cam, seed=0)
		return q_track, q_flick, strike_idx, q_spin, release_idx, q_pp, phases_pp, bowl_pp

	def _finalize(
		q_track: np.ndarray,
		q_flick: np.ndarray,
		strike_idx: int,
		q_spin: np.ndarray,
		release_idx: int,
		q_pp: np.ndarray,
		phases_pp: np.ndarray,
		bowl_pp: tuple[float, float],
		isaac_res: dict[str, dict[str, np.ndarray]],
		isaac_elapsed: float,
		hw_name: str,
	) -> None:
		from digital_twin_armigo.mujoco_cell import Cell
		from digital_twin_armigo.mujoco_cell._mj import mujoco

		t0 = time.perf_counter()
		mj_res: dict[str, dict[str, np.ndarray]] = {}
		for name, q_traj, bxy, byaw in (
			("joint_tracking", q_track, (0.26, 0.0), 0.0),
			("flick_slide", q_flick, (0.26, 0.0), 0.0),
			("spin_release", q_spin, (0.26, 0.0), 0.0),
			("shared_pick_place_replay", q_pp, (0.26, 0.01), 20.0),
		):
			c = Cell(
				_mj_brick_layout(brick_xy=bxy, yaw_deg=byaw),
				SO101_MJCF_DIR,
				physics=spec,
				assisted_grasp=False,
				render_enabled=False,
			)
			mj_res[name] = _mj_step_trajectory(c, q_traj, q_start, brick_xy=bxy, brick_yaw_deg=byaw)
			c.close()
		mj_elapsed = time.perf_counter() - t0

		report: dict[str, Any] = {
			"provenance": {
				"git_head": _git_head(),
				"evaluator_version": EVALUATOR_VERSION,
				"physics_spec_version": spec.version,
				"physics_manifest": physics_manifest(spec),
				"engines": {
					"mujoco": mujoco.__version__,
					"isaac_present": bool(isaac_res),
					"host_os": platform.platform(),
					"hardware": hw_name,
				},
				"simulation_params": {
					"physics_hz": PHYSICS_HZ,
					"control_fps": FPS,
					"substeps": SUBSTEPS,
					"dt_s": 1.0 / PHYSICS_HZ,
					"assisted_grasp": False,
					"sensor_mode": "oracle_state_open_loop_dynamics",
					"seed": 0,
				},
				"claim_scope": (
					"Simulation-only cross-engine audit under nominal assumed physics (v1-nominal-assumed). "
					"Cross-engine agreement does NOT establish real-hardware accuracy or sim-to-real transfer."
				),
			},
			"mujoco_traj_sec": mj_elapsed,
			"isaac_traj_sec": isaac_elapsed,
			"dynamics_comparison": {},
			"skill_transferability": {},
		}

		# Experiment Group 1: Shared-trajectory open-loop dynamics comparison
		exp1: dict[str, Any] = {"frames": len(q_track)}
		if isaac_res:
			exp1["cross_engine"] = compare_trajectories(
				{
					"t": mj_res["joint_tracking"]["t"],
					"qpos": mj_res["joint_tracking"]["qpos"],
					"tcp_pos": mj_res["joint_tracking"]["tcp_pos"],
				},
				{
					"t": isaac_res["joint_tracking"]["t"],
					"qpos": isaac_res["joint_tracking"]["qpos"],
					"tcp_pos": isaac_res["joint_tracking"]["tcp_pos"],
				},
				fps=FPS,
			)
		report["dynamics_comparison"]["joint_tracking"] = exp1

		mj_flick_sum = summarize_flick_experiment(
			mj_res["flick_slide"]["brick_pos"],
			mj_res["flick_slide"]["brick_vel"],
			mj_res["flick_slide"]["brick_quat"],
			strike_idx,
			FPS,
		)
		exp2: dict[str, Any] = {"strike_frame": strike_idx, "mujoco": mj_flick_sum}
		if isaac_res:
			is_flick_sum = summarize_flick_experiment(
				isaac_res["flick_slide"]["brick_pos"],
				isaac_res["flick_slide"]["brick_vel"],
				isaac_res["flick_slide"]["brick_quat"],
				strike_idx,
				FPS,
			)
			exp2["isaac"] = is_flick_sum
			exp2["scalar_displacement_diff_m"] = abs(
				float(mj_flick_sum["final_displacement_m"]) - float(is_flick_sum["final_displacement_m"])
			)
			exp2["cross_engine"] = compare_trajectories(
				{
					"t": mj_res["flick_slide"]["t"][strike_idx:],
					"qpos": mj_res["flick_slide"]["qpos"][strike_idx:],
					"brick_pos": mj_res["flick_slide"]["brick_pos"][strike_idx:],
					"brick_yaw": _quat_yaw(mj_res["flick_slide"]["brick_quat"][strike_idx:]),
					"events": {"stop_time_s": mj_flick_sum["stop_time_s"]},
				},
				{
					"t": isaac_res["flick_slide"]["t"][strike_idx:],
					"qpos": isaac_res["flick_slide"]["qpos"][strike_idx:],
					"brick_pos": isaac_res["flick_slide"]["brick_pos"][strike_idx:],
					"brick_yaw": _quat_yaw(isaac_res["flick_slide"]["brick_quat"][strike_idx:]),
					"events": {"stop_time_s": is_flick_sum["stop_time_s"]},
				},
				fps=FPS,
			)
		report["dynamics_comparison"]["flick_slide"] = exp2

		mj_spin_sum = summarize_spin_release_experiment(
			mj_res["spin_release"]["brick_pos"],
			mj_res["spin_release"]["brick_vel"],
			mj_res["spin_release"]["brick_quat"],
			mj_res["spin_release"]["brick_ang_vel"],
			release_idx,
			FPS,
		)
		exp3: dict[str, Any] = {"release_frame": release_idx, "mujoco": mj_spin_sum}
		if isaac_res:
			is_spin_sum = summarize_spin_release_experiment(
				isaac_res["spin_release"]["brick_pos"],
				isaac_res["spin_release"]["brick_vel"],
				isaac_res["spin_release"]["brick_quat"],
				isaac_res["spin_release"]["brick_ang_vel"],
				release_idx,
				FPS,
			)
			exp3["isaac"] = is_spin_sum
			exp3["cross_engine"] = compare_trajectories(
				{
					"t": mj_res["spin_release"]["t"][release_idx:],
					"qpos": mj_res["spin_release"]["qpos"][release_idx:],
					"brick_pos": mj_res["spin_release"]["brick_pos"][release_idx:],
					"brick_yaw": _quat_yaw(mj_res["spin_release"]["brick_quat"][release_idx:]),
					"events": {"flight_time_s": mj_spin_sum["flight_time_s"]},
				},
				{
					"t": isaac_res["spin_release"]["t"][release_idx:],
					"qpos": isaac_res["spin_release"]["qpos"][release_idx:],
					"brick_pos": isaac_res["spin_release"]["brick_pos"][release_idx:],
					"brick_yaw": _quat_yaw(isaac_res["spin_release"]["brick_quat"][release_idx:]),
					"events": {"flight_time_s": is_spin_sum["flight_time_s"]},
				},
				fps=FPS,
			)
		report["dynamics_comparison"]["spin_release"] = exp3

		# Experiment 4: Exact shared pick-place control replay across BOTH engines
		mj_pp_health = _evaluate_mj_rollout(mj_res["shared_pick_place_replay"], phases_pp, spec, DUPLO_2X2, bowl_pp)
		exp4: dict[str, Any] = {"frames": len(q_pp), "mujoco": mj_pp_health}
		if isaac_res:
			is_r = isaac_res["shared_pick_place_replay"]
			torque_lims = np.array([spec.arm_actuator.effort_limit] * 5 + [spec.gripper_actuator.effort_limit])
			is_pp_health = evaluate_episode_health(
				q_cmd=is_r["q_cmd"],
				q_act=is_r["qpos"],
				tcp_pos=is_r["tcp_pos"],
				brick_pos=is_r["brick_pos"],
				brick_quat=is_r["brick_quat"],
				jaw_forces=is_r["jaw_forces"],
				unintended_forces=is_r["unintended_forces"],
				phases=phases_pp,
				target_bowl_xy=bowl_pp,
				bowl_r=BOWL_R,
				brick_vel=is_r["brick_vel"],
				brick_ang_vel=is_r["brick_ang_vel"],
				ee_quat=is_r["ee_quat"],
				fps=FPS,
				brick_size=DUPLO_2X2,
				substep_unintended_forces=is_r["substep_unintended"],
				pinch_normal_forces=is_r["pinch_normal"],
				pinch_tangential_forces=is_r["pinch_tangential"],
				actuator_torques=is_r["actuator_torques"],
				torque_limits=torque_lims,
				torque_source="estimated_implicit_pd",
				timestamps=is_r["t"],
			).to_dict()
			exp4["isaac"] = is_pp_health
			exp4["cross_engine"] = compare_trajectories(
				{
					"t": mj_res["shared_pick_place_replay"]["t"],
					"qpos": mj_res["shared_pick_place_replay"]["qpos"],
					"tcp_pos": mj_res["shared_pick_place_replay"]["tcp_pos"],
					"brick_pos": mj_res["shared_pick_place_replay"]["brick_pos"],
					"brick_yaw": _quat_yaw(mj_res["shared_pick_place_replay"]["brick_quat"]),
				},
				{
					"t": is_r["t"],
					"qpos": is_r["qpos"],
					"tcp_pos": is_r["tcp_pos"],
					"brick_pos": is_r["brick_pos"],
					"brick_yaw": _quat_yaw(is_r["brick_quat"]),
				},
				fps=FPS,
			)
		report["dynamics_comparison"]["shared_pick_place_replay"] = exp4

		# Experiment Group 2: Skill transferability (engine-specific adapters across search & held-out conditions)
		if args.include_matrix:
			search_rows = []
			heldout_rows = []
			for split_name, cond_list, bucket in (
				("search", SENSITIVITY_MATRIX, search_rows),
				("held_out", HELD_OUT_PERTURBATIONS, heldout_rows),
			):
				for cond in cond_list:
					mj_batch = _summarize_mj_condition_batch(cond, spec, top_cam, seeds=args.matrix_seeds)
					row: dict[str, Any] = {"split": split_name, "condition": cond.to_dict(), "mujoco_batch": mj_batch}
					if not args.mujoco_only:
						sub_out = args.out_dir / f"isaac_matrix_{cond.name}"
						cmd = [
							sys.executable,
							str(ROOT / "scripts" / "isaac" / "stage2_expert.py"),
							"--no-cameras",
							"--brick-size",
							cond.brick_size,
							"--brick-x",
							str(cond.brick_xy[0]),
							"--brick-y",
							str(cond.brick_xy[1]),
							"--brick-yaw",
							str(cond.brick_yaw_deg),
							"--clutter",
							str(cond.clutter_per_bowl),
							"--mass-scale",
							str(cond.mass_scale),
							"--friction-scale",
							str(cond.friction_scale),
							"--latency-frames",
							str(cond.latency_frames),
							"--backlash-deg",
							str(cond.backlash_deg),
							"--out-dir",
							str(sub_out),
						]
						if not cond.defective:
							cmd.append("--no-defective")
						subprocess.run(cmd, check=True)
						is_metrics = json.loads((sub_out / "stage2_metrics.json").read_text(encoding="utf-8"))
						row["isaac_single"] = is_metrics["health"]
					bucket.append(row)
			report["skill_transferability"] = {
				"search_matrix": search_rows,
				"held_out_perturbations": heldout_rows,
			}

		out_json = args.out_dir / "cross_engine_results.json"
		out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
		print("CROSS_ENGINE_SUMMARY:", json.dumps(report, indent=2), flush=True)

	if not args.mujoco_only:
		import isaaclab.sim as sim_utils

		sim_cfg = sim_utils.SimulationCfg(dt=1.0 / PHYSICS_HZ, render_interval=SUBSTEPS, device="cuda:0")
		with launch_simulation(sim_cfg, args):
			q_track, q_flick, strike_idx, q_spin, release_idx, q_pp, phases_pp, bowl_pp = _build_shared_trajs()
			t1 = time.perf_counter()
			isaac_res, hw_name = _run_isaac_in_context(sim_cfg, spec, q_track, q_flick, q_spin, q_pp, q_start)
			isaac_elapsed = time.perf_counter() - t1
			_finalize(
				q_track,
				q_flick,
				strike_idx,
				q_spin,
				release_idx,
				q_pp,
				phases_pp,
				bowl_pp,
				isaac_res,
				isaac_elapsed,
				hw_name,
			)
	else:
		q_track, q_flick, strike_idx, q_spin, release_idx, q_pp, phases_pp, bowl_pp = _build_shared_trajs()
		_finalize(
			q_track,
			q_flick,
			strike_idx,
			q_spin,
			release_idx,
			q_pp,
			phases_pp,
			bowl_pp,
			{},
			0.0,
			platform.processor() or "cpu",
		)


if __name__ == "__main__":
	main()
