"""Stage 2 Experiment: Pure-physics DifferentialIK + slow pinch grasp + ContactSensor + LeRobot export.

Sequence:
1. Open gripper (0.85 rad) -> hover directly above brick (z = 0.095 m)
2. Descend vertically to brick pinch height (z = 0.011 m, tol < 4 mm)
3. Pause 0.4 s to settle
4. Slowly close gripper onto the 31.8 mm brick under PhysX friction + contact sensing
5. Verify real physical grasp (dual-jaw contact force + gripper blocked by brick width)
6. Lift to z = 0.14 m and verify brick remains held between jaws (no air-gap weld)
7. Present defect face -> transport over pink plate -> open jaws -> drop under gravity -> return
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Stage 2 Isaac Lab pure-physics pick-inspect-place experiment.")
add_launcher_args(parser)
args_cli = parser.parse_args([])
args_cli.enable_cameras = True
args_cli.visualizer = []

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.utils import configclass

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from digital_twin_armigo.presentation import show_points, view_cosine  # noqa: E402

USD_ASSETS = ROOT / "assets"
OUT_DIR = ROOT / "results" / "stage2"
DEMO_DIR = OUT_DIR / "demos"

FPS = 30
PHYSICS_HZ = 240
SUBSTEPS = PHYSICS_HZ // FPS  # 8 physics substeps @ 240 Hz per 30 Hz video frame
BOWL_R = 0.10
PHASES = ("rest", "approach", "grasp", "lift", "show_move", "show_hold", "handoff", "place", "return")
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
STATE_NAMES = [f"{j}.pos" for j in JOINTS]
TASKS = {
	"green": "Pick up the brick and place it in the green bowl.",
	"pink": "Pick up the brick and place it in the pink bowl.",
}
CALIBRATION = {
	"shoulder_pan": (1267, 3200),
	"shoulder_lift": (894, 3251),
	"elbow_flex": (849, 3045),
	"wrist_flex": (788, 3115),
	"wrist_roll": (2040, 3961),
	"gripper": (2041, 3508),
}
OFFSETS_DEG = {"shoulder_lift": 1.5816, "elbow_flex": -4.6851, "wrist_flex": -1.3997}
TICKS_PER_DEG = 4096 / 360
GRIPPER_RAD = (-0.17453, 1.74533)
TOP_FULL = (640, 480)
TOP_CROP = (96, 0, 512, 416)
FINE_MAX_M = 0.04
COARSE_MAX_M = 0.30
CAMERA_SHAPES = {
	"front": [480, 640, 3],
	"top_rgb_v2": [416, 512, 3],
	"top_depth_v2": [416, 512, 3],
	"wrist": [480, 640, 3],
}
HANDOFF_Q = np.array([0.0, 0.2, -0.9, 1.5, 0.0], dtype=np.float64)
REST_U = np.array([4.1, -98.7, 99.3, -98.2, 24.1, 33.7], dtype=np.float64)
OPEN_G = 0.85   # ~71 mm jaw gap (> 2.2x brick width, clears brick cleanly on descent)
HOLD_G = -0.15  # commanded target past 31.8 mm brick contact (~0.17 rad) for firm clamping force
# Biased +6 mm toward moving jaw at open/mid gap so fixed_jaw clears brick -X edge by ~8.4 mm on descent
TCP_OFFSET_LOCAL = (0.0135, 0.0, -0.0905)


def normalized_to_rad(u: np.ndarray) -> np.ndarray:
	u = np.asarray(u, dtype=np.float64)
	out = np.empty(6, dtype=np.float64)
	for i, name in enumerate(JOINTS[:5]):
		lo, hi = CALIBRATION[name]
		deg = u[i] / 100.0 * (hi - lo) / 2.0 / TICKS_PER_DEG + OFFSETS_DEG.get(name, 0.0)
		out[i] = math.radians(deg)
	out[5] = GRIPPER_RAD[0] + np.clip(u[5], 0.0, 100.0) / 100.0 * (GRIPPER_RAD[1] - GRIPPER_RAD[0])
	return out


def rad_to_normalized(q: np.ndarray) -> np.ndarray:
	q = np.asarray(q, dtype=np.float64)
	out = np.empty(6, dtype=np.float64)
	for i, name in enumerate(JOINTS[:5]):
		lo, hi = CALIBRATION[name]
		deg = math.degrees(q[i]) - OFFSETS_DEG.get(name, 0.0)
		out[i] = deg * TICKS_PER_DEG * 2.0 / (hi - lo) * 100.0
	out[5] = (q[5] - GRIPPER_RAD[0]) / (GRIPPER_RAD[1] - GRIPPER_RAD[0]) * 100.0
	return out.astype(np.float32)


def camera_axes_from_pose(cam: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	t, p = math.radians(cam["tilt_deg"]), math.radians(cam["yaw_deg"])
	forward = np.array([math.sin(t) * math.cos(p), math.sin(t) * math.sin(p), -math.cos(t)])
	up = np.array([math.cos(t) * math.cos(p), math.cos(t) * math.sin(p), math.sin(t)])
	right = np.cross(forward, up)
	r = math.radians(cam["roll_deg"])
	right, up = math.cos(r) * right + math.sin(r) * up, -math.sin(r) * right + math.cos(r) * up
	return right, up, -forward


def layered_height_image(height_m: np.ndarray, valid: np.ndarray) -> np.ndarray:
	h = np.clip(np.where(valid, height_m, 0.0), 0.0, None)
	out = np.empty((*h.shape, 3), np.uint8)
	out[..., 0] = np.rint(255.0 * np.minimum(h / FINE_MAX_M, 1.0))
	out[..., 1] = np.rint(255.0 * np.minimum(h / COARSE_MAX_M, 1.0))
	out[..., 2] = np.where(valid, 255, 0)
	out[~valid, :2] = 0
	return out


def depth_plane_to_height(cam: dict, depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	right, up, z_opengl = camera_axes_from_pose(cam)
	forward = -z_opengl
	hgt, wid = depth.shape
	u, v = np.meshgrid(np.arange(wid) + 0.5, np.arange(hgt) + 0.5)
	f = cam["focal_px"]
	ray = ((u - wid / 2.0) / f)[..., None] * right + (-(v - hgt / 2.0) / f)[..., None] * up + forward
	valid = np.isfinite(depth) & (depth > 0.05) & (depth < 5.0)
	world_z = cam["height"] + ray[..., 2] * np.where(valid, depth, 0.0)
	return world_z, valid


@configclass
class Stage2SceneCfg(InteractiveSceneCfg):
	table = AssetBaseCfg(
		prim_path="/World/Table",
		spawn=sim_utils.CuboidCfg(
			size=(1.6, 1.6, 0.04),
			collision_props=sim_utils.UsdPhysicsCollisionCfg(),
			physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.9, restitution=0.0),
			visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.58, 0.58, 0.58), roughness=0.9),
		),
		init_state=AssetBaseCfg.InitialStateCfg(pos=(0.25, 0.0, -0.02)),
	)
	dome_light = AssetBaseCfg(
		prim_path="/World/DomeLight",
		spawn=sim_utils.DomeLightCfg(intensity=650.0, color=(0.95, 0.95, 0.95)),
	)
	sun_light = AssetBaseCfg(
		prim_path="/World/SunLight",
		spawn=sim_utils.DistantLightCfg(intensity=1400.0, color=(1.0, 0.98, 0.94), angle=2.0),
		init_state=AssetBaseCfg.InitialStateCfg(rot=(0.22, 0.12, 0.0, 0.968)),
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
			"sts3215": ImplicitActuatorCfg(
				joint_names_expr=[".*"],
				stiffness=450.0,
				damping=35.0,
				effort_limit_sim=2.94,
			)
		},
	)
	green_plate = RigidObjectCfg(
		prim_path="{ENV_REGEX_NS}/GreenPlate",
		spawn=sim_utils.UsdFileCfg(usd_path=str(USD_ASSETS / "props" / "green_plate.usda")),
		init_state=RigidObjectCfg.InitialStateCfg(pos=(0.22, 0.19, 0.0)),
	)
	pink_plate = RigidObjectCfg(
		prim_path="{ENV_REGEX_NS}/PinkPlate",
		spawn=sim_utils.UsdFileCfg(usd_path=str(USD_ASSETS / "props" / "pink_plate.usda")),
		init_state=RigidObjectCfg.InitialStateCfg(pos=(0.26, -0.18, 0.0)),
	)
	brick = RigidObjectCfg(
		prim_path="{ENV_REGEX_NS}/Brick",
		spawn=sim_utils.UsdFileCfg(
			usd_path=str(USD_ASSETS / "props" / "duplo_2x2_defective.usda"),
			activate_contact_sensors=True,
			rigid_props=sim_utils.PhysxRigidBodyPropertiesCfg(
				solver_position_iteration_count=32,
				solver_velocity_iteration_count=8,
				linear_damping=0.4,
				angular_damping=0.4,
				max_depenetration_velocity=1.0,
			),
		),
		init_state=RigidObjectCfg.InitialStateCfg(
			pos=(0.26, 0.01, 0.0096),
			rot=(0.0, 0.0, math.sin(math.radians(10.0)), math.cos(math.radians(10.0))),
		),
	)
	contact_fixed = ContactSensorCfg(
		prim_path="{ENV_REGEX_NS}/Robot/Geometry/base/shoulder/upper_arm/lower_arm/wrist/gripper",
		filter_prim_paths_expr=["{ENV_REGEX_NS}/Brick"],
		update_period=0.0,
	)
	contact_moving = ContactSensorCfg(
		prim_path="{ENV_REGEX_NS}/Robot/Geometry/base/shoulder/upper_arm/lower_arm/wrist/gripper/moving_jaw_so101_v1",
		filter_prim_paths_expr=["{ENV_REGEX_NS}/Brick"],
		update_period=0.0,
	)
	top_cam = CameraCfg(
		prim_path="{ENV_REGEX_NS}/TopCamera",
		update_period=0.0,
		height=TOP_FULL[1],
		width=TOP_FULL[0],
		data_types=["rgb", "distance_to_image_plane"],
		spawn=sim_utils.PinholeCameraCfg(focal_length=20.0, horizontal_aperture=20.955, clipping_range=(0.05, 10.0)),
		offset=CameraCfg.OffsetCfg(pos=(0.6, 0.2, 0.5), rot=(0.0, 0.0, 0.0, 1.0), convention="opengl"),
	)
	wrist_cam = CameraCfg(
		prim_path="{ENV_REGEX_NS}/Robot/Geometry/base/shoulder/upper_arm/lower_arm/wrist/gripper/camera_mount/wrist_cam",
		update_period=0.0,
		height=480,
		width=640,
		data_types=["rgb"],
		spawn=None,
	)
	front_cam = CameraCfg(
		prim_path="{ENV_REGEX_NS}/FrontCamera",
		update_period=0.0,
		height=480,
		width=640,
		data_types=["rgb"],
		spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, horizontal_aperture=20.955, clipping_range=(0.05, 10.0)),
		offset=CameraCfg.OffsetCfg(pos=(0.62, -0.28, 0.28), rot=(0.0, 0.0, 0.0, 1.0), convention="opengl"),
	)


def configure_cameras(cfg: Stage2SceneCfg, top_cam: dict) -> None:
	right, up, z_opengl = camera_axes_from_pose(top_cam)
	top_quat = tuple(float(v) for v in Rotation.from_matrix(np.column_stack([right, up, z_opengl])).as_quat())
	cfg.top_cam.spawn.focal_length = 20.955 * float(top_cam["focal_px"]) / TOP_FULL[0]
	cfg.top_cam.offset.pos = (float(top_cam["x"]), float(top_cam["y"]), float(top_cam["height"]))
	cfg.top_cam.offset.rot = top_quat

	eye = np.array([0.62, -0.28, 0.28])
	target = np.array([0.25, -0.02, 0.05])
	fwd = target - eye
	fwd /= np.linalg.norm(fwd)
	r_f = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
	r_f /= np.linalg.norm(r_f)
	u_f = np.cross(r_f, fwd)
	cfg.front_cam.offset.pos = tuple(float(v) for v in eye)
	cfg.front_cam.offset.rot = tuple(float(v) for v in Rotation.from_matrix(np.column_stack([r_f, u_f, -fwd])).as_quat())


def main() -> None:
	OUT_DIR.mkdir(parents=True, exist_ok=True)
	if DEMO_DIR.exists():
		for old in DEMO_DIR.glob("*"):
			old.unlink()
	DEMO_DIR.mkdir(parents=True, exist_ok=True)

	top_cam = json.loads((ROOT / "config" / "sim" / "top_camera_joint.json").read_text())["camera"]
	cfg = Stage2SceneCfg(num_envs=1, env_spacing=2.0)
	configure_cameras(cfg, top_cam)

	sim_cfg = sim_utils.SimulationCfg(dt=1.0 / PHYSICS_HZ, render_interval=SUBSTEPS, device="cuda:0")
	with launch_simulation(sim_cfg, args_cli):
		import isaaclab.utils.math as math_utils
		from isaaclab.scene import InteractiveScene

		sim = sim_utils.SimulationContext(sim_cfg)
		scene = InteractiveScene(cfg)
		sim.reset()

		robot = scene["robot"]
		brick = scene["brick"]
		c_fixed = scene["contact_fixed"]
		c_moving = scene["contact_moving"]

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

		# Initialize arm at rest pose
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
		_, _, z_opengl_top = camera_axes_from_pose(top_cam)
		cam_fwd_np = -z_opengl_top

		def record_frame(q_cmd_ordered: np.ndarray, phase_name: str, snap_tag: str | None = None) -> None:
			nonlocal max_lift_z
			q_curr = robot.data.joint_pos.torch[0].cpu().numpy()
			q_curr_j = np.array([q_curr[j_idx[n]] for n in JOINTS], dtype=np.float64)
			q_cmd_j = np.array([q_cmd_ordered[j_idx[n]] for n in JOINTS], dtype=np.float64)
			b_pos = brick.data.root_pos_w.torch[0].cpu().numpy()
			b_quat_xyzw = brick.data.root_quat_w.torch[0].cpu().numpy()
			max_lift_z = max(max_lift_z, float(b_pos[2]))

			top_rgb_full = scene["top_cam"].data.output["rgb"].torch[0, ..., :3].cpu().numpy().astype(np.uint8)
			top_depth_full = scene["top_cam"].data.output["distance_to_image_plane"].torch[0].squeeze(-1).cpu().numpy()
			front_rgb = scene["front_cam"].data.output["rgb"].torch[0, ..., :3].cpu().numpy().astype(np.uint8)
			wrist_rgb = scene["wrist_cam"].data.output["rgb"].torch[0, ..., :3].cpu().numpy().astype(np.uint8)

			x0, y0, w, h = TOP_CROP
			top_rgb_c = top_rgb_full[y0 : y0 + h, x0 : x0 + w]
			z_w, valid = depth_plane_to_height(top_cam, top_depth_full)
			top_depth_c = layered_height_image(z_w[y0 : y0 + h, x0 : x0 + w], valid[y0 : y0 + h, x0 : x0 + w])

			hsv = cv2.cvtColor(top_rgb_c, cv2.COLOR_RGB2HSV)
			orange_mask = (hsv[..., 0] < 22) & (hsv[..., 1] > 110) & (hsv[..., 2] > 90)
			b_px = int(orange_mask.sum())
			rot_b = Rotation.from_quat(b_quat_xyzw).as_matrix()
			cos_d = view_cosine(rot_b[:, 0], b_pos, cam_pos_np)
			d_px = int(max(cos_d, 0.0) * 60) if b_px > 20 else 0

			frames_state.append(rad_to_normalized(q_curr_j))
			frames_action.append(rad_to_normalized(q_cmd_j))
			frames_phase.append(PHASES.index(phase_name))
			frames_top_rgb.append(top_rgb_c)
			frames_top_depth.append(top_depth_c)
			frames_front.append(front_rgb)
			frames_wrist.append(wrist_rgb)
			frames_defect_px.append(d_px)
			frames_brick_px.append(b_px)

			if snap_tag is not None:
				panel = np.hstack([
					cv2.resize(top_rgb_c, (480, 390)),
					cv2.resize(front_rgb, (520, 390)),
					cv2.resize(wrist_rgb, (520, 390)),
				])
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
				q_cmd = start_j + s_mj * (target_ordered - start_j)
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
				tcp_p, _, _, _ = get_tcp_and_jac()
				err_m = float(torch.linalg.norm(tcp_p[0] - target_t))
				done = err_m < tol and k >= 10
				tag = snap_tag if (snap_tag and (done or k == max_frames - 1)) else None
				record_frame(q_full[0].cpu().numpy(), phase_name, snap_tag=tag)
				if done:
					break
			return err_m

		# --- Experiment State Machine (pure physics friction, ZERO FixedJoint) ---
		b0 = brick.data.root_pos_w.torch[0].cpu().numpy()
		bx, by = float(b0[0]), float(b0[1])
		brick_yaw = math.radians(20.0)

		# 0. Rest -> unfold above table with jaws wide open
		step_joint_target(q_init[0].cpu().numpy(), 0.35, "rest")
		q_pre = np.zeros(len(joint_names), dtype=np.float64)
		for n, v in zip(JOINTS, (0.0, -0.35, 0.25, 1.35, 0.0, OPEN_G), strict=True):
			q_pre[j_idx[n]] = v
		step_joint_target(q_pre, 0.8, "approach")

		# 1. Open gripper directly above brick (z = 0.095 m)
		e_above = step_ik_pose(np.array([bx, by, 0.095]), brick_yaw, OPEN_G, 2.0, "approach", tol=0.003, max_vel_rad_s=1.8, snap_tag="1_above_open")
		tcp_a, _, _, _ = get_tcp_and_jac()
		print(f"[1_above_open] tcp={tcp_a[0].cpu().numpy().round(4)} brick={brick.data.root_pos_w.torch[0].cpu().numpy().round(4)} ik_err={e_above:.4f}m")

		# 2. Descend vertically to brick side-wall pinch height (z = 0.014 m, jaw tip 1.5 mm above table)
		e_desc = step_ik_pose(np.array([bx, by, 0.014]), brick_yaw, OPEN_G, 2.0, "approach", tol=0.003, max_vel_rad_s=1.2)
		# 3. Stop / pause (0.4 s) to let jaws settle stationary around brick
		q_hold_open = robot.data.joint_pos.torch[0].cpu().numpy()
		q_hold_open[grip_id] = OPEN_G
		step_joint_target(q_hold_open, 0.4, "grasp", snap_tag="2_descend_pause")
		tcp_p, _, _, _ = get_tcp_and_jac()
		b_pause = brick.data.root_pos_w.torch[0].cpu().numpy()
		print(f"[2_descend_pause] tcp={tcp_p[0].cpu().numpy().round(4)} brick={b_pause.round(4)} ik_err={e_desc:.4f}m")

		# 4. Slowly close gripper onto brick under PhysX friction
		n_close = 36  # 1.2 s slow pinch
		q_arm_locked = robot.data.joint_pos.torch[0].cpu().numpy()
		for k in range(n_close):
			g_t = OPEN_G + (k + 1) / n_close * (HOLD_G - OPEN_G)
			q_cmd = q_arm_locked.copy()
			q_cmd[grip_id] = g_t
			t_cmd = torch.tensor(q_cmd[None, :], device=robot.device, dtype=torch.float32)
			robot.actuators.target_command.set_position_index(value=t_cmd)
			for _ in range(SUBSTEPS):
				scene.write_data_to_sim()
				sim.step()
				scene.update(sim.get_physics_dt())
			tag = "3_closed_pinch" if k == n_close - 1 else None
			record_frame(q_cmd, "grasp", snap_tag=tag)

		# Hold squeeze briefly (0.3 s) and read dual-jaw ContactSensor + gripper width
		q_squeeze = q_arm_locked.copy()
		q_squeeze[grip_id] = HOLD_G
		step_joint_target(q_squeeze, 0.3, "grasp")
		f_fix_c, f_mov_c = jaw_forces()
		g_actual_c = float(robot.data.joint_pos.torch[0, grip_id].cpu())
		tcp_c, _, _, _ = get_tcp_and_jac()
		b_c = brick.data.root_pos_w.torch[0].cpu().numpy()
		close_contact_ok = (f_fix_c > 0.05 and f_mov_c > 0.05 and 0.02 < g_actual_c < 0.45)
		print(
			f"[3_closed_pinch] f_fixed={f_fix_c:.3f}N f_moving={f_mov_c:.3f}N "
			f"gripper_q={g_actual_c:.3f}rad tcp={tcp_c[0].cpu().numpy().round(4)} "
			f"brick={b_c.round(4)} contact_ok={close_contact_ok}"
		)

		# 5. Pure-physics lift (still holding HOLD_G, NO FixedJoint!)
		e_lift = step_ik_pose(np.array([bx, by, 0.145]), brick_yaw, HOLD_G, 2.0, "lift", tol=0.004, max_vel_rad_s=1.0, snap_tag="4_lifted_physics")
		f_fix_l, f_mov_l = jaw_forces()
		g_actual_l = float(robot.data.joint_pos.torch[0, grip_id].cpu())
		tcp_l, _, _, _ = get_tcp_and_jac()
		b_l = brick.data.root_pos_w.torch[0].cpu().numpy()
		dz_tcp = float(abs(b_l[2] - tcp_l[0, 2].cpu()))
		dxy_tcp = float(np.hypot(b_l[0] - float(tcp_l[0, 0].cpu()), b_l[1] - float(tcp_l[0, 1].cpu())))
		grasped = bool(b_l[2] > 0.10 and dz_tcp < 0.018 and dxy_tcp < 0.020 and f_fix_l > 0.02 and f_mov_l > 0.02)
		print(
			f"[4_lifted_physics] ik_err={e_lift:.4f}m brick={b_l.round(4)} tcp={tcp_l[0].cpu().numpy().round(4)} "
			f"|dz|={dz_tcp:.4f}m dxy={dxy_tcp:.4f}m f_fixed={f_fix_l:.3f}N f_moving={f_mov_l:.3f}N "
			f"gripper_q={g_actual_l:.3f}rad grasped={grasped}"
		)

		# 6. Present to top camera (gentle wrist_roll sweep keeping grip)
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
			c = view_cosine(rot_b[:, 0], b_now, cam_pos_np)
			best_cos = max(best_cos, c)
			step_joint_target(q_show, 0.35, "show_hold", snap_tag="5_show_defect" if wr_deg == 0.0 else None)

		# 7. Smooth jaws-down recovery -> Move smoothly over pink plate -> Open jaws -> Drop under gravity -> Return
		q_handoff = np.zeros(len(joint_names), dtype=np.float64)
		for n, v in zip(JOINTS, (-0.45, -0.10, 0.25, 1.35, 0.0, HOLD_G), strict=True):
			q_handoff[j_idx[n]] = v
		step_joint_target(q_handoff, 1.3, "handoff")

		pink_xy = (0.26, -0.18)
		e_p1 = step_ik_pose(np.array([pink_xy[0], pink_xy[1], 0.14]), brick_yaw, HOLD_G, 2.0, "place", tol=0.006, max_vel_rad_s=1.0)
		e_p2 = step_ik_pose(np.array([pink_xy[0], pink_xy[1], 0.065]), brick_yaw, HOLD_G, 1.6, "place", tol=0.006, max_vel_rad_s=1.0)
		print(f"[before_release] brick={brick.data.root_pos_w.torch[0].cpu().numpy().round(4)} place_errs=({e_p1:.4f},{e_p2:.4f})")
		q_open = robot.data.joint_pos.torch[0].cpu().numpy()
		q_open[grip_id] = OPEN_G
		step_joint_target(q_open, 0.7, "place", snap_tag="6_released_in_plate")
		# Hold briefly to let brick settle in plate
		step_joint_target(q_open, 0.35, "place")
		step_ik_pose(np.array([pink_xy[0], pink_xy[1], 0.14]), brick_yaw, OPEN_G, 0.8, "place", tol=0.015)
		step_joint_target(q_init[0].cpu().numpy(), 0.9, "return")

		final_brick = brick.data.root_pos_w.torch[0].cpu().numpy()
		dist_pink = float(np.hypot(final_brick[0] - pink_xy[0], final_brick[1] - pink_xy[1]))
		success = bool(close_contact_ok and grasped and dist_pink < BOWL_R and final_brick[2] < 0.05)

		# Save storyboard checks
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
			cv2.imwrite(str(OUT_DIR / "stage2_keyframes.png"), cv2.cvtColor(np.vstack(rows_img), cv2.COLOR_RGB2BGR))

		ep_file = "episode_0000.npz"
		np.savez_compressed(
			DEMO_DIR / ep_file,
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
		index = {
			"fps": FPS,
			"state_names": STATE_NAMES,
			"cameras": CAMERA_SHAPES,
			"source": "isaac-sim-pure-physics",
			"privileged": {
				"phase": "per-frame expert phase, index into phases",
				"phases": list(PHASES),
				"defect_px": "estimated defect visibility score",
				"brick_px": "orange brick pixels in top_rgb_v2",
			},
			"episodes": [
				{
					"file": ep_file,
					"length": len(frames_state),
					"success": success,
					"task": TASKS["pink"],
					"defective": True,
					"defect_face": 0,
				}
			],
		}
		(DEMO_DIR / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")

		summary = {
			"physics_mode": "pure_friction_contact_sensor_no_fixed_joint",
			"total_frames": len(frames_state),
			"close_fixed_force_N": f_fix_c,
			"close_moving_force_N": f_mov_c,
			"close_gripper_rad": g_actual_c,
			"close_contact_ok": close_contact_ok,
			"lift_fixed_force_N": f_fix_l,
			"lift_moving_force_N": f_mov_l,
			"lift_brick_tcp_dz_m": dz_tcp,
			"lift_brick_tcp_dxy_m": dxy_tcp,
			"grasped": grasped,
			"max_lift_z_m": float(max_lift_z),
			"best_show_cos": float(best_cos),
			"final_brick_xyz": [float(v) for v in final_brick],
			"dist_to_pink_plate_m": dist_pink,
			"success": success,
		}
		(OUT_DIR / "stage2_metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
		print("STAGE2_PHYSICS_SUMMARY:", json.dumps(summary, indent=2))


if __name__ == "__main__":
	main()
