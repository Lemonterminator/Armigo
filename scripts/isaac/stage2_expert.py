"""Stage 2 Experiment: Pure-physics DifferentialIK + slow pinch grasp + ContactSensor + Health & LeRobot export.

Reads unified physical parameters from ``digital_twin_armigo.physics`` (``v1-nominal-assumed``), supports
2x2 / 2x3 / 2x4 Duplo bricks, dynamic bowl clutter, mass/friction/latency/backlash sensitivity overrides,
optional RealSense D415 sensor degradation, and full process-health evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Stage 2 Isaac Lab pure-physics pick-inspect-place experiment.")
parser.add_argument("--brick-size", choices=("2x2", "2x3", "2x4"), default="2x2")
parser.add_argument("--brick-x", type=float, default=0.26)
parser.add_argument("--brick-y", type=float, default=0.01)
parser.add_argument("--brick-yaw", type=float, default=20.0, help="Brick yaw in degrees.")
parser.add_argument("--defective", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--clutter", type=int, default=0, help="Dynamic clutter bricks per bowl.")
parser.add_argument("--mass-scale", type=float, default=1.0)
parser.add_argument("--friction-scale", type=float, default=1.0)
parser.add_argument("--latency-frames", type=int, default=0)
parser.add_argument("--backlash-deg", type=float, default=0.0)
parser.add_argument("--sensor-noise", action="store_true", help="Apply D415 depth & RGB degradation.")
parser.add_argument("--no-cameras", action="store_true", help="Disable cameras (uses fast kitless OvPhysX).")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out-dir", type=Path, default=None)
parser.add_argument(
	"--catalog-capture", type=Path, default=None, help="New complete-scene recording directory; must not exist."
)
parser.add_argument("--catalog-parent", default=None, help="Optional parent catalog episode id for this NEW run.")
add_launcher_args(parser)
args_cli = parser.parse_args()
if not args_cli.no_cameras:
	args_cli.enable_cameras = True
args_cli.visualizer = []

import cv2
import isaaclab.sim as sim_utils
import numpy as np
import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.utils import configclass
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from digital_twin_armigo.camera import opengl_axes
from digital_twin_armigo.constants import (
	BOWL_R,
	CAMERA_SHAPES,
	FPS,
	JOINTS,
	PHASES,
	REST_U,
	STATE_NAMES,
	TASKS,
	TOP_CROP,
	TOP_FULL,
)
from digital_twin_armigo.depth import depth_plane_to_height, layered_height_image
from digital_twin_armigo.joints import normalized_to_rad, rad_to_normalized
from digital_twin_armigo.metrics import EVALUATOR_VERSION, evaluate_episode_health
from digital_twin_armigo.physics import ActuatorCommandFilter, NonIdealityCfg, load_physics_spec, physics_manifest
from digital_twin_armigo.presentation import show_points, view_cosine
from digital_twin_armigo.sensor_noise import DEFAULT_D415_NOISE, degrade_depth_plane, degrade_rgb

USD_ASSETS = ROOT / "assets"
DEFAULT_OUT_DIR = ROOT / "results" / "stage2"

PHYSICS_HZ = 240
SUBSTEPS = PHYSICS_HZ // FPS  # 8 physics substeps @ 240 Hz per 30 Hz dataset frame
OPEN_G = 0.85  # ~71 mm jaw gap (> 2.2x 31.8 mm short axis)
HOLD_G = 0.06  # realistic pinch target against ~0.186 rad brick contact (avoids 47 N crush)
TCP_OFFSET_LOCAL = (0.0135, 0.0, -0.0905)


def build_scene_cfg(
	top_cam: dict,
	physics_spec,
	brick_size_key: str,
	brick_xy: tuple[float, float],
	brick_yaw_deg: float,
	defective: bool,
	clutter_count: int,
	mass_scale: float,
	friction_scale: float,
	with_cameras: bool,
	rng: np.random.Generator,
) -> tuple[InteractiveSceneCfg, tuple[float, float], tuple[float, float]]:
	bspec = physics_spec.bricks[brick_size_key].scaled(mass_scale, friction_scale)
	green_xy = (0.22, 0.19)
	pink_xy = (0.26, -0.18)
	suffix = "defective" if defective else "orange"
	brick_usd = USD_ASSETS / "props" / f"duplo_{brick_size_key}_{suffix}.usda"

	arm_act = physics_spec.arm_actuator
	grip_act = physics_spec.gripper_actuator

	@configclass
	class Stage2SceneCfg(InteractiveSceneCfg):
		table = RigidObjectCfg(
			prim_path="{ENV_REGEX_NS}/Table",
			spawn=sim_utils.CuboidCfg(
				size=(1.6, 1.6, 0.04),
				rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
				collision_props=sim_utils.UsdPhysicsCollisionCfg(),
				physics_material=sim_utils.RigidBodyMaterialCfg(
					static_friction=physics_spec.table.static_friction * friction_scale,
					dynamic_friction=physics_spec.table.dynamic_friction * friction_scale,
					restitution=physics_spec.table.restitution,
					friction_combine_mode="max",
					restitution_combine_mode="max",
				),
				visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.58, 0.58, 0.58), roughness=0.9),
			),
			init_state=RigidObjectCfg.InitialStateCfg(pos=(0.25, 0.0, -0.02)),
		)
		robot = ArticulationCfg(
			prim_path="{ENV_REGEX_NS}/Robot",
			spawn=sim_utils.UsdFileCfg(
				usd_path=str(USD_ASSETS / "so101_usd" / "so101_physx.usda"),
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
		green_plate = RigidObjectCfg(
			prim_path="{ENV_REGEX_NS}/GreenPlate",
			spawn=sim_utils.UsdFileCfg(usd_path=str(USD_ASSETS / "props" / "green_plate.usda")),
			init_state=RigidObjectCfg.InitialStateCfg(pos=(green_xy[0], green_xy[1], 0.0)),
		)
		pink_plate = RigidObjectCfg(
			prim_path="{ENV_REGEX_NS}/PinkPlate",
			spawn=sim_utils.UsdFileCfg(usd_path=str(USD_ASSETS / "props" / "pink_plate.usda")),
			init_state=RigidObjectCfg.InitialStateCfg(pos=(pink_xy[0], pink_xy[1], 0.0)),
		)
		brick = RigidObjectCfg(
			prim_path="{ENV_REGEX_NS}/Brick",
			spawn=sim_utils.UsdFileCfg(
				usd_path=str(brick_usd),
				activate_contact_sensors=True,
				mass_props=sim_utils.MassPropertiesCfg(mass=bspec.mass_kg),
				rigid_props=sim_utils.PhysxRigidBodyPropertiesCfg(
					solver_position_iteration_count=32,
					solver_velocity_iteration_count=8,
					linear_damping=0.25,
					angular_damping=0.25,
					max_depenetration_velocity=1.0,
				),
			),
			init_state=RigidObjectCfg.InitialStateCfg(
				pos=(float(brick_xy[0]), float(brick_xy[1]), 0.0096),
				rot=(
					0.0,
					0.0,
					math.sin(math.radians(brick_yaw_deg) / 2.0),
					math.cos(math.radians(brick_yaw_deg) / 2.0),
				),
			),
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

	cfg = Stage2SceneCfg(num_envs=1, env_spacing=2.0)

	# Dynamic clutter bricks inside bowls
	clutter_idx = 0
	for b_xy in (green_xy, pink_xy):
		for _ in range(clutter_count):
			r = BOWL_R * 0.55 * math.sqrt(float(rng.random()))
			a = float(rng.uniform(0.0, 2.0 * math.pi))
			cyaw = float(rng.uniform(0.0, math.pi))
			c_cfg = RigidObjectCfg(
				prim_path=f"{{ENV_REGEX_NS}}/Clutter_{clutter_idx}",
				spawn=sim_utils.UsdFileCfg(
					usd_path=str(USD_ASSETS / "props" / "clutter_brick.usda"),
					mass_props=sim_utils.MassPropertiesCfg(mass=physics_spec.bricks["2x2"].mass_kg * mass_scale),
				),
				init_state=RigidObjectCfg.InitialStateCfg(
					pos=(b_xy[0] + r * math.cos(a), b_xy[1] + r * math.sin(a), 0.015),
					rot=(0.0, 0.0, math.sin(cyaw / 2.0), math.cos(cyaw / 2.0)),
				),
			)
			setattr(cfg, f"clutter_{clutter_idx}", c_cfg)
			clutter_idx += 1

	if with_cameras:
		right, up, z_opengl = opengl_axes(top_cam)
		top_quat = tuple(float(v) for v in Rotation.from_matrix(np.column_stack([right, up, z_opengl])).as_quat())
		eye = np.array([0.62, -0.28, 0.28])
		target = np.array([0.25, -0.02, 0.05])
		fwd = target - eye
		fwd /= np.linalg.norm(fwd)
		r_f = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
		r_f /= np.linalg.norm(r_f)
		u_f = np.cross(r_f, fwd)
		front_quat = tuple(float(v) for v in Rotation.from_matrix(np.column_stack([r_f, u_f, -fwd])).as_quat())

		cfg.dome_light = AssetBaseCfg(
			prim_path="/World/DomeLight",
			spawn=sim_utils.DomeLightCfg(intensity=650.0, color=(0.95, 0.95, 0.95)),
		)
		cfg.sun_light = AssetBaseCfg(
			prim_path="/World/SunLight",
			spawn=sim_utils.DistantLightCfg(intensity=1400.0, color=(1.0, 0.98, 0.94), angle=2.0),
			init_state=AssetBaseCfg.InitialStateCfg(rot=(0.22, 0.12, 0.0, 0.968)),
		)
		cfg.top_cam = CameraCfg(
			prim_path="{ENV_REGEX_NS}/TopCamera",
			update_period=0.0,
			height=TOP_FULL[1],
			width=TOP_FULL[0],
			data_types=["rgb", "distance_to_image_plane"],
			spawn=sim_utils.PinholeCameraCfg(
				focal_length=20.955 * float(top_cam["focal_px"]) / TOP_FULL[0],
				horizontal_aperture=20.955,
				clipping_range=(0.05, 10.0),
			),
			offset=CameraCfg.OffsetCfg(
				pos=(float(top_cam["x"]), float(top_cam["y"]), float(top_cam["height"])),
				rot=top_quat,
				convention="opengl",
			),
		)
		cfg.wrist_cam = CameraCfg(
			prim_path="{ENV_REGEX_NS}/Robot/Geometry/base/shoulder/upper_arm/lower_arm/wrist/gripper/camera_mount/wrist_cam",
			update_period=0.0,
			height=480,
			width=640,
			data_types=["rgb"],
			spawn=None,
		)
		cfg.front_cam = CameraCfg(
			prim_path="{ENV_REGEX_NS}/FrontCamera",
			update_period=0.0,
			height=480,
			width=640,
			data_types=["rgb"],
			spawn=sim_utils.PinholeCameraCfg(
				focal_length=18.0, horizontal_aperture=20.955, clipping_range=(0.05, 10.0)
			),
			offset=CameraCfg.OffsetCfg(pos=tuple(float(v) for v in eye), rot=front_quat, convention="opengl"),
		)

	return cfg, green_xy, pink_xy


def main() -> None:
	if args_cli.catalog_capture is not None and args_cli.catalog_capture.exists():
		raise FileExistsError("catalog capture refuses to replace existing data")
	out_dir = args_cli.out_dir or DEFAULT_OUT_DIR
	demo_dir = out_dir / "demos"
	out_dir.mkdir(parents=True, exist_ok=True)
	if demo_dir.exists():
		for old in demo_dir.glob("*"):
			old.unlink()
	demo_dir.mkdir(parents=True, exist_ok=True)

	physics_spec = load_physics_spec()
	rng = np.random.default_rng(args_cli.seed)
	top_cam = json.loads((ROOT / "config" / "sim" / "top_camera_joint.json").read_text())["camera"]
	with_cameras = not args_cli.no_cameras
	cfg, green_xy, pink_xy = build_scene_cfg(
		top_cam=top_cam,
		physics_spec=physics_spec,
		brick_size_key=args_cli.brick_size,
		brick_xy=(args_cli.brick_x, args_cli.brick_y),
		brick_yaw_deg=args_cli.brick_yaw,
		defective=args_cli.defective,
		clutter_count=args_cli.clutter,
		mass_scale=args_cli.mass_scale,
		friction_scale=args_cli.friction_scale,
		with_cameras=with_cameras,
		rng=rng,
	)

	non_ideal = NonIdealityCfg(
		latency_frames=args_cli.latency_frames,
		backlash_rad=math.radians(args_cli.backlash_deg),
		velocity_limit_rad_s=physics_spec.arm_actuator.velocity_limit,
		encoder_ticks=4096,
	)

	sim_cfg = sim_utils.SimulationCfg(dt=1.0 / PHYSICS_HZ, render_interval=SUBSTEPS, device="cuda:0")
	with launch_simulation(sim_cfg, args_cli):
		import isaaclab.utils.math as math_utils
		from isaaclab.scene import InteractiveScene

		sim = sim_utils.SimulationContext(sim_cfg)
		scene = InteractiveScene(cfg)
		sim.reset()
		catalog_recorder = None
		catalog_identity = None
		if args_cli.catalog_capture is not None:
			import omni.usd

			from digital_twin_armigo.catalog.capture import IsaacCapture, identity

			initial_clutter = []
			for k in range(args_cli.clutter * 2):
				initial = scene.rigid_objects[f"clutter_{k}"].cfg.init_state
				initial_clutter.append(
					[
						float(initial.pos[0]),
						float(initial.pos[1]),
						2 * math.atan2(float(initial.rot[2]), float(initial.rot[3])),
					]
				)
			catalog_recorder = IsaacCapture(
				omni.usd.get_context().get_stage(),
				{
					"green": green_xy,
					"pink": pink_xy,
					"brick_xy": [args_cli.brick_x, args_cli.brick_y],
					"brick_yaw_deg": args_cli.brick_yaw,
					"brick_size": physics_spec.bricks[args_cli.brick_size].size_m,
					"clutter": initial_clutter,
					"distractors": [],
					"markers": [],
				},
				scene,
			)
			catalog_identity = identity(ROOT, "isaac", "6.2.0.0")

		robot = scene["robot"]
		brick = scene["brick"]
		c_fixed = scene["contact_fixed"]
		c_moving = scene["contact_moving"]
		c_wrist_tbl = scene["contact_arm_table"]

		joint_names = list(robot.joint_names)
		j_idx = {n: i for i, n in enumerate(joint_names)}
		arm_names = list(JOINTS[:5])
		arm_ids = [j_idx[n] for n in arm_names]
		grip_id = j_idx["gripper"]

		ee_body_id = robot.find_bodies("gripper")[0][0]
		jacobi_body_idx = ee_body_id - 1 if robot.is_fixed_base else ee_body_id
		jacobi_arm_cols = [j + robot.num_base_dofs for j in arm_ids]

		offset_pos = torch.tensor([TCP_OFFSET_LOCAL], device=robot.device, dtype=torch.float32)
		offset_rot = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=robot.device, dtype=torch.float32)

		ik_cfg = DifferentialIKControllerCfg(
			command_type="pose",
			use_relative_mode=False,
			ik_method="dls",
			ik_params={"lambda_val": 0.01},
			orientation_weight=(0.15, 0.15, 0.05),
		)
		ik_ctrl = DifferentialIKController(ik_cfg, num_envs=1, device=robot.device)

		q_rest = normalized_to_rad(REST_U)
		q_init = torch.zeros((1, len(joint_names)), device=robot.device, dtype=torch.float32)
		for i, n in enumerate(JOINTS):
			q_init[0, j_idx[n]] = float(q_rest[i])
		q_init[0, grip_id] = OPEN_G
		robot.write_joint_position_to_sim_index(position=q_init)
		robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q_init))
		robot.actuators.target_command.set_position_index(value=q_init)
		for _ in range(24):
			scene.write_data_to_sim()
			sim.step()
			scene.update(sim.get_physics_dt())

		cmd_filter = ActuatorCommandFilter(non_ideal, n_joints=len(JOINTS), fps=FPS)
		q0_ordered = np.array([float(robot.data.joint_pos.torch[0, j_idx[n]].cpu()) for n in JOINTS], dtype=np.float64)
		cmd_filter.reset(q0_ordered)

		def _to_sim_order(q_ordered: np.ndarray) -> np.ndarray:
			out = np.zeros(len(joint_names), dtype=np.float64)
			for i, n in enumerate(JOINTS):
				out[j_idx[n]] = q_ordered[i]
			return out

		def _to_joints_order(q_sim: np.ndarray) -> np.ndarray:
			return np.array([q_sim[j_idx[n]] for n in JOINTS], dtype=np.float64)

		def get_tcp_and_jac() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
			ee_pos_w = robot.data.body_pos_w.torch[:, ee_body_id]
			ee_quat_w = robot.data.body_quat_w.torch[:, ee_body_id]
			root_pos_w = robot.data.root_pos_w.torch
			root_quat_w = robot.data.root_quat_w.torch
			ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
			tcp_pos_b, tcp_quat_b = math_utils.combine_frame_transforms(ee_pos_b, ee_quat_b, offset_pos, offset_rot)
			jac_w = robot.data.body_link_jacobian_w.torch[:, jacobi_body_idx, :, :][:, :, jacobi_arm_cols]
			base_rot = math_utils.matrix_from_quat(math_utils.quat_inv(root_quat_w))
			jac_b = jac_w.clone()
			jac_b[:, :3, :] = torch.bmm(base_rot, jac_w[:, :3, :])
			jac_b[:, 3:, :] = torch.bmm(base_rot, jac_w[:, 3:, :])
			body_quat_b = math_utils.quat_mul(math_utils.quat_inv(root_quat_w), ee_quat_w)
			off_b = math_utils.quat_apply(body_quat_b, offset_pos)
			jac_b[:, :3] += torch.linalg.cross(jac_b[:, 3:], off_b[:, :, None], dim=1)
			q_arm = robot.data.joint_pos.torch[:, arm_ids]
			return tcp_pos_b, tcp_quat_b, jac_b, q_arm

		def jaw_forces() -> tuple[float, float]:
			f_fix = float(torch.linalg.norm(c_fixed.data.force_matrix_w.torch[0, 0, 0]).cpu())
			f_mov = float(torch.linalg.norm(c_moving.data.force_matrix_w.torch[0, 0, 0]).cpu())
			return f_fix, f_mov

		def _substep_jaw_components() -> tuple[np.ndarray, np.ndarray]:
			# Extract per-substep (SUBSTEPS,) average jaw normal and tangential forces when available
			def _pair(sensor) -> tuple[np.ndarray, np.ndarray]:
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

			fn0, ft0 = _pair(c_fixed)
			fn1, ft1 = _pair(c_moving)
			return 0.5 * (fn0 + fn1), 0.5 * (ft0 + ft1)

		def _substep_unintended() -> tuple[float, np.ndarray]:
			uh = getattr(c_wrist_tbl.data, "force_matrix_w_history", None)
			if uh is not None:
				# Shape: (env, hist, sensor_body, filter_prims, 3) -> peak across filter prims per substep
				norms = torch.linalg.norm(uh.torch[0, :, 0, :], dim=-1).max(dim=-1).values
				arr = norms.cpu().numpy().astype(np.float64)
				return float(arr[0]), arr
			val = float(torch.linalg.norm(c_wrist_tbl.data.force_matrix_w.torch[0, 0, :]).cpu())
			return val, np.full(SUBSTEPS, val, dtype=np.float64)

		frames_q_cmd: list[np.ndarray] = []
		frames_q_act: list[np.ndarray] = []
		frames_tcp: list[np.ndarray] = []
		frames_ee_quat: list[np.ndarray] = []
		frames_brick_pos: list[np.ndarray] = []
		frames_brick_quat: list[np.ndarray] = []
		frames_brick_vel: list[np.ndarray] = []
		frames_brick_ang_vel: list[np.ndarray] = []
		frames_jaw_f: list[tuple[float, float]] = []
		frames_unintended_f: list[float] = []
		frames_sub_unint: list[np.ndarray] = []
		frames_pinch_norm: list[np.ndarray] = []
		frames_pinch_tan: list[np.ndarray] = []
		frames_torques: list[np.ndarray] = []
		frames_state: list[np.ndarray] = []
		frames_action: list[np.ndarray] = []
		frames_phase: list[int] = []
		frames_top_rgb: list[np.ndarray] = []
		frames_top_depth: list[np.ndarray] = []
		frames_front: list[np.ndarray] = []
		frames_wrist: list[np.ndarray] = []
		frames_defect_px: list[int] = []
		frames_brick_px: list[int] = []
		snapshots: dict[str, np.ndarray] = {}

		max_lift_z = 0.0
		cam_pos_np = np.array([top_cam["x"], top_cam["y"], top_cam["height"]], dtype=np.float64)
		_, _, z_opengl_top = opengl_axes(top_cam)
		cam_fwd_np = -z_opengl_top

		def record_frame(q_cmd_sim: np.ndarray, phase_name: str, snap_tag: str | None = None) -> None:
			nonlocal max_lift_z
			q_curr_sim = robot.data.joint_pos.torch[0].cpu().numpy()
			q_curr_j = _to_joints_order(q_curr_sim)
			q_cmd_j = _to_joints_order(q_cmd_sim)
			q_obs_j = cmd_filter.observe_joints(q_curr_j, rng)
			tq_sim = robot.data.applied_torque.torch[0].cpu().numpy()
			tq_j = _to_joints_order(tq_sim)
			tcp_p, tcp_q, _, _ = get_tcp_and_jac()
			tcp_np = tcp_p[0].cpu().numpy().astype(np.float64)
			ee_q_np = tcp_q[0].cpu().numpy().astype(np.float64)

			b_pos = brick.data.root_pos_w.torch[0].cpu().numpy().astype(np.float64)
			b_quat_xyzw = brick.data.root_quat_w.torch[0].cpu().numpy().astype(np.float64)
			b_vel = brick.data.root_lin_vel_w.torch[0].cpu().numpy().astype(np.float64)
			b_ang_vel = brick.data.root_ang_vel_w.torch[0].cpu().numpy().astype(np.float64)
			if catalog_recorder is not None:
				catalog_recorder.frame(len(frames_state) / FPS, b_pos)
				if len(frames_state) % 60 == 0:
					print("CATALOG_CAPTURE_FRAME:", len(frames_state), flush=True)
			max_lift_z = max(max_lift_z, float(b_pos[2]))

			frames_q_cmd.append(q_cmd_j)
			frames_q_act.append(q_curr_j)
			frames_tcp.append(tcp_np)
			frames_ee_quat.append(ee_q_np)
			frames_brick_pos.append(b_pos)
			frames_brick_quat.append(b_quat_xyzw)
			frames_brick_vel.append(b_vel)
			frames_brick_ang_vel.append(b_ang_vel)
			frames_jaw_f.append(jaw_forces())
			u_val, u_sub = _substep_unintended()
			pn_sub, pt_sub = _substep_jaw_components()
			frames_unintended_f.append(u_val)
			frames_sub_unint.append(u_sub)
			frames_pinch_norm.append(pn_sub)
			frames_pinch_tan.append(pt_sub)
			frames_torques.append(tq_j)
			frames_state.append(rad_to_normalized(q_obs_j))
			frames_action.append(rad_to_normalized(q_cmd_j))
			frames_phase.append(PHASES.index(phase_name))

			if with_cameras:
				top_rgb_full = scene["top_cam"].data.output["rgb"].torch[0, ..., :3].cpu().numpy().astype(np.uint8)
				top_depth_full = (
					scene["top_cam"].data.output["distance_to_image_plane"].torch[0].squeeze(-1).cpu().numpy()
				)
				front_rgb = scene["front_cam"].data.output["rgb"].torch[0, ..., :3].cpu().numpy().astype(np.uint8)
				wrist_rgb = scene["wrist_cam"].data.output["rgb"].torch[0, ..., :3].cpu().numpy().astype(np.uint8)

				x0, y0, w, h = TOP_CROP
				top_rgb_c = top_rgb_full[y0 : y0 + h, x0 : x0 + w]
				if args_cli.sensor_noise:
					top_rgb_c = degrade_rgb(top_rgb_c, rng, DEFAULT_D415_NOISE)
					front_rgb = degrade_rgb(front_rgb, rng, DEFAULT_D415_NOISE)
					wrist_rgb = degrade_rgb(wrist_rgb, rng, DEFAULT_D415_NOISE)
					top_depth_full, d_valid_full = degrade_depth_plane(
						top_depth_full,
						np.isfinite(top_depth_full) & (top_depth_full > 0.05),
						rng,
						DEFAULT_D415_NOISE,
					)
					z_w, valid = depth_plane_to_height(top_cam, top_depth_full)
					valid &= d_valid_full
				else:
					z_w, valid = depth_plane_to_height(top_cam, top_depth_full)
				z_c, valid_c = z_w[y0 : y0 + h, x0 : x0 + w], valid[y0 : y0 + h, x0 : x0 + w]
				top_depth_c = layered_height_image(z_c, valid_c)

				hsv = cv2.cvtColor(top_rgb_c, cv2.COLOR_RGB2HSV)
				orange_mask = (hsv[..., 0] < 22) & (hsv[..., 1] > 110) & (hsv[..., 2] > 90)
				b_px = int(orange_mask.sum())
				rot_b = Rotation.from_quat(b_quat_xyzw).as_matrix()
				cos_d = view_cosine(rot_b[:, 0], b_pos, cam_pos_np)
				d_px = int(max(cos_d, 0.0) * 60) if (args_cli.defective and b_px > 20) else 0

				frames_top_rgb.append(top_rgb_c)
				frames_top_depth.append(top_depth_c)
				frames_front.append(front_rgb)
				frames_wrist.append(wrist_rgb)
				frames_defect_px.append(d_px)
				frames_brick_px.append(b_px)

				if snap_tag is not None:
					panel = np.hstack(
						[
							cv2.resize(top_rgb_c, (480, 390)),
							cv2.resize(front_rgb, (520, 390)),
							cv2.resize(wrist_rgb, (520, 390)),
						]
					)
					cv2.putText(panel, snap_tag, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 30, 30), 2)
					snapshots[snap_tag] = panel

		def step_joint_target(
			target_ordered: np.ndarray, secs: float, phase_name: str, snap_tag: str | None = None
		) -> None:
			n_frames = max(2, round(secs * FPS))
			start_j = robot.data.joint_pos.torch[0].cpu().numpy()
			for k in range(n_frames):
				s = (k + 1) / n_frames
				s_mj = 10 * s**3 - 15 * s**4 + 6 * s**5
				q_raw_sim = start_j + s_mj * (target_ordered - start_j)
				q_act_j = _to_joints_order(robot.data.joint_pos.torch[0].cpu().numpy())
				q_filt_j = cmd_filter.step_command(_to_joints_order(q_raw_sim), q_act_j)
				q_cmd = _to_sim_order(q_filt_j)
				t_cmd = torch.tensor(q_cmd[None, :], device=robot.device, dtype=torch.float32)
				robot.actuators.target_command.set_position_index(value=t_cmd)
				for _ in range(SUBSTEPS):
					scene.write_data_to_sim()
					sim.step()
					scene.update(sim.get_physics_dt())
				tag = snap_tag if (snap_tag and k == n_frames - 1) else None
				record_frame(q_cmd, phase_name, snap_tag=tag)

		def step_ik_pose(
			target_pos: np.ndarray,
			yaw_rad: float,
			gripper_cmd: float,
			max_secs: float,
			phase_name: str,
			tol: float = 0.004,
			max_vel_rad_s: float = 1.2,
			snap_tag: str | None = None,
		) -> float:
			"""Move smoothly via velocity-limited DifferentialIK (integrating ``q_cmd``) until within ``tol``."""
			max_frames = max(4, round(max_secs * FPS))
			quat_des = Rotation.from_euler("z", yaw_rad).as_quat()
			cmd = torch.tensor([[*target_pos, *quat_des]], device=robot.device, dtype=torch.float32)
			target_t = torch.tensor(target_pos, device=robot.device, dtype=torch.float32)
			max_step = max_vel_rad_s / PHYSICS_HZ
			q_cmd_arm = robot.data.joint_pos.torch[:, arm_ids].clone()
			err_m = 1.0
			for k in range(max_frames):
				for _ in range(SUBSTEPS):
					tcp_p, tcp_q, jac_b, q_arm = get_tcp_and_jac()
					ik_ctrl.set_command(cmd, ee_pos=tcp_p, ee_quat=tcp_q)
					q_arm_des = ik_ctrl.compute(tcp_p, tcp_q, jac_b, q_arm)
					q_cmd_arm = q_cmd_arm + torch.clamp(q_arm_des - q_cmd_arm, -max_step, max_step)
					q_full = robot.data.joint_pos.torch.clone()
					q_full[0, arm_ids] = q_cmd_arm[0]
					q_full[0, grip_id] = float(gripper_cmd)
					robot.actuators.target_command.set_position_index(value=q_full)
					scene.write_data_to_sim()
					sim.step()
					scene.update(sim.get_physics_dt())
				q_cmd_filt = _to_sim_order(
					cmd_filter.step_command(
						_to_joints_order(q_full[0].cpu().numpy()),
						_to_joints_order(robot.data.joint_pos.torch[0].cpu().numpy()),
					)
				)
				if args_cli.latency_frames > 0 or args_cli.backlash_deg > 0:
					robot.actuators.target_command.set_position_index(
						value=torch.tensor(q_cmd_filt[None, :], device=robot.device, dtype=torch.float32)
					)
				tcp_p, _, _, _ = get_tcp_and_jac()
				err_m = float(torch.linalg.norm(tcp_p[0] - target_t))
				done = err_m < tol and k >= 10
				tag = snap_tag if (snap_tag and (done or k == max_frames - 1)) else None
				record_frame(q_cmd_filt, phase_name, snap_tag=tag)
				if done:
					break
			return err_m

		# --- Experiment State Machine (pure physics friction, ZERO FixedJoint) ---
		b0 = brick.data.root_pos_w.torch[0].cpu().numpy()
		bx, by = float(b0[0]), float(b0[1])
		brick_yaw = math.radians(args_cli.brick_yaw)
		# For 2x3 and 2x4 bricks (long axis along brick local X), align jaws across the 31.8 mm short axis
		# folded into [-pi/2, pi/2] so wrist_roll stays away from joint stops.
		if args_cli.brick_size != "2x2":
			grasp_yaw = ((brick_yaw + math.pi / 2.0 + math.pi / 2.0) % math.pi) - math.pi / 2.0
		else:
			grasp_yaw = brick_yaw

		step_joint_target(q_init[0].cpu().numpy(), 0.35, "rest")
		q_pre = np.zeros(len(joint_names), dtype=np.float64)
		for n, v in zip(JOINTS, (0.0, -0.35, 0.25, 1.35, 0.0, OPEN_G), strict=True):
			q_pre[j_idx[n]] = v
		step_joint_target(q_pre, 0.8, "approach")

		step_ik_pose(
			np.array([bx, by, 0.095]),
			grasp_yaw,
			OPEN_G,
			2.0,
			"approach",
			tol=0.003,
			max_vel_rad_s=1.8,
			snap_tag="1_above_open",
		)
		step_ik_pose(np.array([bx, by, 0.014]), grasp_yaw, OPEN_G, 2.0, "approach", tol=0.003, max_vel_rad_s=1.2)
		q_hold_open = robot.data.joint_pos.torch[0].cpu().numpy()
		q_hold_open[grip_id] = OPEN_G
		step_joint_target(q_hold_open, 0.4, "grasp", snap_tag="2_descend_pause")

		n_close = 36
		q_arm_locked = robot.data.joint_pos.torch[0].cpu().numpy()
		for k in range(n_close):
			g_t = OPEN_G + (k + 1) / n_close * (HOLD_G - OPEN_G)
			q_cmd = q_arm_locked.copy()
			q_cmd[grip_id] = g_t
			q_filt = _to_sim_order(
				cmd_filter.step_command(
					_to_joints_order(q_cmd), _to_joints_order(robot.data.joint_pos.torch[0].cpu().numpy())
				)
			)
			t_cmd = torch.tensor(q_filt[None, :], device=robot.device, dtype=torch.float32)
			robot.actuators.target_command.set_position_index(value=t_cmd)
			for _ in range(SUBSTEPS):
				scene.write_data_to_sim()
				sim.step()
				scene.update(sim.get_physics_dt())
			tag = "3_closed_pinch" if k == n_close - 1 else None
			record_frame(q_filt, "grasp", snap_tag=tag)

		q_squeeze = q_arm_locked.copy()
		q_squeeze[grip_id] = HOLD_G
		step_joint_target(q_squeeze, 0.3, "grasp")
		f_fix_c, f_mov_c = jaw_forces()
		g_actual_c = float(robot.data.joint_pos.torch[0, grip_id].cpu())
		close_contact_ok = f_fix_c > 0.02 and f_mov_c > 0.02 and 0.02 < g_actual_c < 0.45

		step_ik_pose(
			np.array([bx, by, 0.145]),
			grasp_yaw,
			HOLD_G,
			2.0,
			"lift",
			tol=0.004,
			max_vel_rad_s=1.0,
			snap_tag="4_lifted_physics",
		)

		pts = show_points(cam_pos_np, cam_fwd_np, (0.30, 0.35, 0.40, 0.45), 12.0, (0.14, 0.38), 0.13)
		show_pt = pts[0] if len(pts) else np.array([0.30, 0.05, 0.16])
		step_ik_pose(show_pt, 0.0, HOLD_G, 1.8, "show_move", tol=0.010, max_vel_rad_s=1.0)

		best_cos = -1.0
		for wr_deg in (-40.0, 0.0, 40.0):
			q_show = robot.data.joint_pos.torch[0].cpu().numpy()
			q_show[j_idx["wrist_flex"]] = 0.65
			q_show[j_idx["wrist_roll"]] = math.radians(wr_deg)
			q_show[grip_id] = HOLD_G
			step_joint_target(q_show, 0.65, "show_move")
			b_now = brick.data.root_pos_w.torch[0].cpu().numpy()
			rot_b = Rotation.from_quat(brick.data.root_quat_w.torch[0].cpu().numpy()).as_matrix()
			best_cos = max(best_cos, view_cosine(rot_b[:, 0], b_now, cam_pos_np))
			step_joint_target(q_show, 0.35, "show_hold", snap_tag="5_show_defect" if wr_deg == 0.0 else None)

		q_handoff = np.zeros(len(joint_names), dtype=np.float64)
		for n, v in zip(JOINTS, (-0.45, -0.10, 0.25, 1.35, 0.0, HOLD_G), strict=True):
			q_handoff[j_idx[n]] = v
		step_joint_target(q_handoff, 1.3, "handoff")

		target_bowl_xy = pink_xy if args_cli.defective else green_xy
		step_ik_pose(np.array([target_bowl_xy[0], target_bowl_xy[1], 0.14]), grasp_yaw, HOLD_G, 2.0, "place", tol=0.006)
		step_ik_pose(
			np.array([target_bowl_xy[0], target_bowl_xy[1], 0.065]), grasp_yaw, HOLD_G, 1.6, "place", tol=0.006
		)
		q_open = robot.data.joint_pos.torch[0].cpu().numpy()
		q_open[grip_id] = OPEN_G
		step_joint_target(q_open, 0.7, "place", snap_tag="6_released_in_plate")
		step_joint_target(q_open, 0.5, "place")
		step_ik_pose(np.array([target_bowl_xy[0], target_bowl_xy[1], 0.14]), grasp_yaw, OPEN_G, 0.8, "place", tol=0.015)
		step_joint_target(q_init[0].cpu().numpy(), 0.9, "return")
		step_joint_target(q_init[0].cpu().numpy(), 0.5, "return")

		bspec_run = physics_spec.bricks[args_cli.brick_size]
		torque_lims = np.array(
			[physics_spec.arm_actuator.effort_limit] * 5 + [physics_spec.gripper_actuator.effort_limit]
		)
		health = evaluate_episode_health(
			q_cmd=np.array(frames_q_cmd),
			q_act=np.array(frames_q_act),
			tcp_pos=np.array(frames_tcp),
			brick_pos=np.array(frames_brick_pos),
			brick_quat=np.array(frames_brick_quat),
			jaw_forces=np.array(frames_jaw_f),
			unintended_forces=np.array(frames_unintended_f),
			phases=np.array(frames_phase),
			target_bowl_xy=target_bowl_xy,
			bowl_r=BOWL_R,
			brick_vel=np.array(frames_brick_vel),
			brick_ang_vel=np.array(frames_brick_ang_vel),
			ee_quat=np.array(frames_ee_quat),
			fps=FPS,
			brick_size=bspec_run.size_m,
			substep_unintended_forces=np.array(frames_sub_unint),
			pinch_normal_forces=np.array(frames_pinch_norm),
			pinch_tangential_forces=np.array(frames_pinch_tan),
			actuator_torques=np.array(frames_torques),
			torque_limits=torque_lims,
			torque_source="estimated_implicit_pd",
		)

		if with_cameras:
			seq_order = [
				"1_above_open",
				"2_descend_pause",
				"3_closed_pinch",
				"4_lifted_physics",
				"5_show_defect",
				"6_released_in_plate",
			]
			rows_img = [snapshots[k] for k in seq_order if k in snapshots]
			if rows_img:
				cv2.imwrite(str(out_dir / "stage2_keyframes.png"), cv2.cvtColor(np.vstack(rows_img), cv2.COLOR_RGB2BGR))

			ep_file = "episode_0000.npz"
			np.savez_compressed(
				demo_dir / ep_file,
				state=np.stack(frames_state).astype(np.float32),
				action=np.stack(frames_action).astype(np.float32),
				phase=np.array(frames_phase, dtype=np.uint8),
				defect_px=np.array(frames_defect_px, dtype=np.int32),
				brick_px=np.array(frames_brick_px, dtype=np.int32),
				top_rgb_v2=np.stack(frames_top_rgb),
				top_depth_v2=np.stack(frames_top_depth),
				front=np.stack(frames_front),
				wrist=np.stack(frames_wrist),
			)
			task_key = "pink" if args_cli.defective else "green"
			index = {
				"fps": FPS,
				"state_names": STATE_NAMES,
				"cameras": CAMERA_SHAPES,
				"source": "isaac-sim-pure-physics",
				"physics_spec_version": physics_spec.version,
				"sensor_noise": DEFAULT_D415_NOISE.version if args_cli.sensor_noise else None,
				"episodes": [
					{
						"file": ep_file,
						"length": len(frames_state),
						"success": health.success,
						"failure_mode": health.failure_mode,
						"task": TASKS[task_key],
						"defective": args_cli.defective,
						"defect_face": 0,
					}
				],
			}
			(demo_dir / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")

		summary = {
			"physics_mode": "pure_friction_contact_sensor_no_fixed_joint",
			"evaluator_version": EVALUATOR_VERSION,
			"physics_spec_version": physics_spec.version,
			"physics_manifest": physics_manifest(physics_spec),
			"seed": args_cli.seed,
			"physics_hz": PHYSICS_HZ,
			"control_fps": FPS,
			"backend": "ovphysx_no_cameras" if args_cli.no_cameras else "rtx_physx",
			"brick_size": args_cli.brick_size,
			"brick_pose_init": [args_cli.brick_x, args_cli.brick_y, args_cli.brick_yaw],
			"clutter_per_bowl": args_cli.clutter,
			"mass_scale": args_cli.mass_scale,
			"friction_scale": args_cli.friction_scale,
			"latency_frames": args_cli.latency_frames,
			"backlash_deg": args_cli.backlash_deg,
			"sensor_noise": DEFAULT_D415_NOISE.version if args_cli.sensor_noise else None,
			"total_frames": len(frames_state),
			"close_fixed_force_N": f_fix_c,
			"close_moving_force_N": f_mov_c,
			"close_gripper_rad": g_actual_c,
			"close_contact_ok": close_contact_ok,
			"max_lift_z_m": float(max_lift_z),
			"best_show_cos": float(best_cos),
			"final_brick_xyz": [float(v) for v in frames_brick_pos[-1]],
			"verdict": health.verdict,
			"reason_codes": list(health.reason_codes),
			"health": health.to_dict(),
			"success": health.success,
		}
		(out_dir / "stage2_metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
		if catalog_recorder is not None:
			captured_health = health.to_dict()
			captured_health["recorded_substep_pinch_normal_peak_N"] = float(np.max(frames_pinch_norm))
			catalog_recorder.save(
				args_cli.catalog_capture,
				{
					"qpos": np.array(frames_q_act),
					"q_cmd": np.array(frames_q_cmd),
					"brick_pos": np.array(frames_brick_pos),
					"brick_quat": np.array(frames_brick_quat),
					"brick_vel": np.array(frames_brick_vel),
					"brick_ang_vel": np.array(frames_brick_ang_vel),
					"jaw_forces": np.array(frames_jaw_f),
					"phases": np.array(frames_phase),
					"actuator_torques": np.array(frames_torques),
					"substep_unintended": np.array(frames_sub_unint),
					"pinch_normal_substeps": np.array(frames_pinch_norm),
				},
				{"verdict": health.verdict, "health": captured_health, "info": {"n_frames": len(frames_state)}},
				{
					"identity": catalog_identity,
					"physics_manifest": physics_manifest(physics_spec),
					"seed": args_cli.seed,
					"sensor_noise": summary["sensor_noise"],
					"evaluator_version": EVALUATOR_VERSION,
					"recording_index": index if with_cameras else None,
					"use_fabric": sim_cfg.use_fabric,
					"capture_validation": {
						"status": "PASS",
						"pose_source": catalog_recorder.scene["pose_source"],
						"max_reference_position_error_m": catalog_recorder.max_reference_error,
					},
				},
				"isaac_stage2",
				"isaac",
				args_cli.catalog_parent,
				artifacts={"observations": os.path.relpath(demo_dir / "episode_0000.npz", args_cli.catalog_capture)}
				if with_cameras
				else None,
			)
		print("STAGE2_PHYSICS_SUMMARY:", json.dumps(summary, indent=2))


if __name__ == "__main__":
	main()
