"""Stage 2 Verification: DifferentialIK + FixedJoint grasp + presentation_search + expert_place + LeRobot export."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Stage 2 Isaac Lab expert pick-inspect-place episode generator.")
add_launcher_args(parser)
args_cli = parser.parse_args()
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
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from digital_twin_armigo.presentation import show_points, view_cosine  # noqa: E402

USD_ASSETS = ROOT / "assets"
OUT_DIR = ROOT / "results" / "stage2"
DEMO_DIR = OUT_DIR / "demos"

FPS = 30
PHYSICS_HZ = 120
SUBSTEPS = PHYSICS_HZ // FPS  # 4 steps @ 120 Hz per 30 Hz recorded frame
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
OPEN_G = 1.25
CLOSED_G = -0.17
# Offset from gripper link origin to jaw pinch midpoint in gripper local frame
TCP_OFFSET_LOCAL = (0.004, 0.0, -0.098)


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
		spawn=sim_utils.UsdFileCfg(usd_path=str(USD_ASSETS / "so101_usd" / "so101_physx.usda")),
		actuators={
			"sts3215": ImplicitActuatorCfg(
				joint_names_expr=[".*"],
				stiffness=500.0,
				damping=40.0,
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
		init_state=RigidObjectCfg.InitialStateCfg(pos=(0.27, -0.18, 0.0)),
	)
	brick = RigidObjectCfg(
		prim_path="{ENV_REGEX_NS}/Brick",
		spawn=sim_utils.UsdFileCfg(usd_path=str(USD_ASSETS / "props" / "duplo_2x2_defective.usda")),
		init_state=RigidObjectCfg.InitialStateCfg(
			pos=(0.26, 0.01, 0.0096),
			rot=(0.0, 0.0, math.sin(math.radians(12.5)), math.cos(math.radians(12.5))),
		),
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
		offset=CameraCfg.OffsetCfg(pos=(0.72, -0.32, 0.34), rot=(0.0, 0.0, 0.0, 1.0), convention="opengl"),
	)


def configure_cameras(cfg: Stage2SceneCfg, top_cam: dict) -> None:
	right, up, z_opengl = camera_axes_from_pose(top_cam)
	top_quat = tuple(float(v) for v in Rotation.from_matrix(np.column_stack([right, up, z_opengl])).as_quat())
	cfg.top_cam.spawn.focal_length = 20.955 * float(top_cam["focal_px"]) / TOP_FULL[0]
	cfg.top_cam.offset.pos = (float(top_cam["x"]), float(top_cam["y"]), float(top_cam["height"]))
	cfg.top_cam.offset.rot = top_quat

	eye = np.array([0.72, -0.32, 0.34])
	target = np.array([0.24, 0.02, 0.06])
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

	sim_cfg = sim_utils.SimulationCfg(dt=1.0 / PHYSICS_HZ, device="cuda:0")
	with launch_simulation(sim_cfg, args_cli):
		import isaaclab.utils.math as math_utils
		import omni.usd
		from isaaclab.scene import InteractiveScene
		from pxr import Gf, UsdPhysics

		sim = sim_utils.SimulationContext(sim_cfg)
		scene = InteractiveScene(cfg)
		sim.reset()

		robot = scene["robot"]
		brick = scene["brick"]
		stage = omni.usd.get_context().get_stage()

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
			ik_params={"lambda_val": 0.03},
			orientation_weight=(1.0, 1.0, 0.25),
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
		for _ in range(15):
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

		# Recorders
		frames_state: list[np.ndarray] = []
		frames_action: list[np.ndarray] = []
		frames_phase: list[int] = []
		frames_top_rgb: list[np.ndarray] = []
		frames_top_depth: list[np.ndarray] = []
		frames_front: list[np.ndarray] = []
		frames_wrist: list[np.ndarray] = []
		frames_defect_px: list[int] = []
		frames_brick_px: list[int] = []
		keyframes: dict[str, np.ndarray] = {}

		fj_prim_path = "/World/GraspJoint"
		attached = False
		max_lift_z = 0.0
		cam_pos_np = np.array([top_cam["x"], top_cam["y"], top_cam["height"]], dtype=np.float64)
		_, _, z_opengl_top = camera_axes_from_pose(top_cam)
		cam_fwd_np = -z_opengl_top

		def record_frame(q_cmd_ordered: np.ndarray, phase_name: str) -> None:
			nonlocal max_lift_z
			# Read current joint state & brick state
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

			# Estimate defect normal cosine & visible brick pixels via color segmentation in cropped RGB
			hsv = cv2.cvtColor(top_rgb_c, cv2.COLOR_RGB2HSV)
			orange_mask = (hsv[..., 0] < 22) & (hsv[..., 1] > 110) & (hsv[..., 2] > 90)
			b_px = int(orange_mask.sum())
			rot_b = Rotation.from_quat(b_quat_xyzw).as_matrix()
			defect_normal_w = rot_b[:, 0]  # face 0 (+X)
			cos_d = view_cosine(defect_normal_w, b_pos, cam_pos_np)
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

			if phase_name in ("grasp", "lift", "show_hold", "place"):
				if phase_name == "place" and attached:
					pass
				else:
					panel = np.hstack([
						cv2.resize(top_rgb_c, (480, 390)),
						cv2.resize(front_rgb, (520, 390)),
						cv2.resize(wrist_rgb, (520, 390)),
					])
					keyframes[phase_name] = panel

		def step_joint_target(target_ordered: np.ndarray, secs: float, phase_name: str, settle: float = 0.0) -> None:
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
				record_frame(q_cmd, phase_name)
			for _ in range(round(settle * FPS)):
				t_cmd = torch.tensor(target_ordered[None, :], device=robot.device, dtype=torch.float32)
				robot.actuators.target_command.set_position_index(value=t_cmd)
				for _ in range(SUBSTEPS):
					scene.write_data_to_sim()
					sim.step()
					scene.update(sim.get_physics_dt())
				record_frame(target_ordered, phase_name)

		def step_ik_pose(
			target_pos: np.ndarray, yaw_rad: float, gripper_cmd: float, max_secs: float, phase_name: str, tol: float = 0.008
		) -> float:
			"""Drive TCP via closed-loop DifferentialIKController until within ``tol`` or ``max_secs``."""
			max_frames = max(4, round(max_secs * FPS))
			quat_des = Rotation.from_euler("z", yaw_rad).as_quat()  # xyzw
			cmd = torch.tensor([[*target_pos, *quat_des]], device=robot.device, dtype=torch.float32)
			target_t = torch.tensor(target_pos, device=robot.device, dtype=torch.float32)
			err_m = 1.0
			for k in range(max_frames):
				tcp_p, tcp_q, jac_b, q_arm = get_tcp_and_jac()
				ik_ctrl.set_command(cmd, ee_pos=tcp_p, ee_quat=tcp_q)
				q_arm_des = ik_ctrl.compute(tcp_p, tcp_q, jac_b, q_arm)
				q_step = q_arm + torch.clamp(q_arm_des - q_arm, -0.14, 0.14)
				q_full = robot.data.joint_pos.torch.clone()
				q_full[0, arm_ids] = q_step[0]
				q_full[0, grip_id] = float(gripper_cmd)
				robot.actuators.target_command.set_position_index(value=q_full)
				for _ in range(SUBSTEPS):
					scene.write_data_to_sim()
					sim.step()
					scene.update(sim.get_physics_dt())
				record_frame(q_full[0].cpu().numpy(), phase_name)
				err_m = float(torch.linalg.norm(tcp_p[0] - target_t))
				if err_m < tol and k >= 8:
					break
			return err_m

		def attach_fixed_joint() -> None:
			nonlocal attached
			tcp_p, _, _, _ = get_tcp_and_jac()
			p_g = robot.data.body_pos_w.torch[0, ee_body_id].cpu().numpy()
			q_g = robot.data.body_quat_w.torch[0, ee_body_id].cpu().numpy()  # xyzw
			p_b = brick.data.root_pos_w.torch[0].cpu().numpy()
			q_b = brick.data.root_quat_w.torch[0].cpu().numpy()  # xyzw
			dist = float(np.linalg.norm(tcp_p[0].cpu().numpy() - p_b))
			print(f"Attach check: tcp={tcp_p[0].cpu().numpy().round(4)} brick={p_b.round(4)} dist={dist:.4f} m")
			if dist > 0.055:
				raise RuntimeError(f"TCP missed brick before attach (dist={dist:.4f} m > 0.055 m)")
			R_g = Rotation.from_quat(q_g).as_matrix()
			R_b = Rotation.from_quat(q_b).as_matrix()
			rel_pos = R_g.T @ (p_b - p_g)
			rel_q_xyzw = Rotation.from_matrix(R_g.T @ R_b).as_quat()
			gripper_prim = (
				"/World/envs/env_0/Robot/Geometry/base/shoulder/upper_arm/lower_arm/wrist/gripper"
			)
			fj = UsdPhysics.FixedJoint.Define(stage, fj_prim_path)
			fj.CreateBody0Rel().SetTargets([gripper_prim])
			fj.CreateBody1Rel().SetTargets(["/World/envs/env_0/Brick"])
			fj.CreateLocalPos0Attr().Set(Gf.Vec3f(float(rel_pos[0]), float(rel_pos[1]), float(rel_pos[2])))
			fj.CreateLocalRot0Attr().Set(
				Gf.Quatf(float(rel_q_xyzw[3]), float(rel_q_xyzw[0]), float(rel_q_xyzw[1]), float(rel_q_xyzw[2]))
			)
			fj.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
			fj.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
			attached = True

		def detach_fixed_joint() -> None:
			nonlocal attached
			if stage.GetPrimAtPath(fj_prim_path).IsValid():
				stage.RemovePrim(fj_prim_path)
			attached = False

		# --- Execute Stage 2 State Machine: rest -> rise -> expert_pick -> presentation_search -> handoff -> expert_place -> return ---
		b0 = brick.data.root_pos_w.torch[0].cpu().numpy()
		bx, by = float(b0[0]), float(b0[1])
		brick_yaw = math.radians(25.0)

		# 1. Rest hold + unfold cleanly out of rest singularity into jaws-down workspace hover
		step_joint_target(q_init[0].cpu().numpy(), 0.35, "rest")
		q_pre = np.zeros(len(joint_names), dtype=np.float64)
		for n, v in zip(JOINTS, (0.0, -0.25, 0.45, 1.35, 0.0, OPEN_G), strict=True):
			q_pre[j_idx[n]] = v
		step_joint_target(q_pre, 0.7, "approach")

		# 2. Approach + Descend via DifferentialIKController
		e1 = step_ik_pose(np.array([bx, by, 0.10]), brick_yaw, OPEN_G, 1.6, "approach")
		e2 = step_ik_pose(np.array([bx, by, 0.032]), brick_yaw, OPEN_G, 1.2, "approach")
		e3 = step_ik_pose(np.array([bx, by, 0.014]), brick_yaw, OPEN_G, 0.9, "grasp")
		print(f"IK pick errors (m): hover={e1:.4f} pregrasp={e2:.4f} grasp={e3:.4f}")

		# Close gripper & attach FixedJoint
		q_close = robot.data.joint_pos.torch[0].cpu().numpy()
		q_close[grip_id] = CLOSED_G
		step_joint_target(q_close, 0.5, "grasp")
		attach_fixed_joint()
		q_close[grip_id] = CLOSED_G
		step_joint_target(q_close, 0.2, "grasp")

		# 3. Lift
		e4 = step_ik_pose(np.array([bx, by, 0.13]), brick_yaw, CLOSED_G, 1.3, "lift")
		grasped = float(brick.data.root_pos_w.torch[0, 2].cpu()) > 0.08
		print(f"After lift (ik_err={e4:.4f}m): brick pos={brick.data.root_pos_w.torch[0].cpu().numpy().round(4)} grasped={grasped}")

		# 4. presentation_search: choose reachable show point in front of top_cam + sweep wrist_roll for best defect view cosine
		pts = show_points(cam_pos_np, cam_fwd_np, (0.30, 0.35, 0.40, 0.45), 12.0, (0.14, 0.38), 0.12)
		show_pt = pts[0] if len(pts) else np.array([0.30, 0.05, 0.16])
		step_ik_pose(show_pt + np.array([0.0, 0.0, 0.02]), 0.0, CLOSED_G, 1.3, "show_move")

		best_cos = -1.0
		for wr_deg in (-90.0, 0.0, 90.0):
			q_show = robot.data.joint_pos.torch[0].cpu().numpy()
			q_show[j_idx["wrist_flex"]] = 0.45
			q_show[j_idx["wrist_roll"]] = math.radians(wr_deg)
			q_show[grip_id] = CLOSED_G
			step_joint_target(q_show, 0.45, "show_move")
			b_now = brick.data.root_pos_w.torch[0].cpu().numpy()
			rot_b = Rotation.from_quat(brick.data.root_quat_w.torch[0].cpu().numpy()).as_matrix()
			c = view_cosine(rot_b[:, 0], b_now, cam_pos_np)
			best_cos = max(best_cos, c)
			step_joint_target(q_show, 0.35, "show_hold")

		# 5. Handoff holding pose & jaws-down transition before placing over pink_plate
		q_hand = np.zeros(len(joint_names), dtype=np.float64)
		for i, n in enumerate(JOINTS[:5]):
			q_hand[j_idx[n]] = HANDOFF_Q[i]
		q_hand[grip_id] = CLOSED_G
		step_joint_target(q_hand, 0.8, "handoff")

		# 6. Place in pink bowl via DifferentialIKController -> detach FixedJoint -> open gripper -> drop in plate!
		pink_xy = (0.27, -0.18)
		e5 = step_ik_pose(np.array([pink_xy[0], pink_xy[1], 0.14]), brick_yaw, CLOSED_G, 1.8, "place")
		e6 = step_ik_pose(np.array([pink_xy[0], pink_xy[1], 0.070]), brick_yaw, CLOSED_G, 1.2, "place")
		print(f"IK place errors (m): above_plate={e5:.4f} drop={e6:.4f}")
		detach_fixed_joint()
		q_open = robot.data.joint_pos.torch[0].cpu().numpy()
		q_open[grip_id] = OPEN_G
		step_joint_target(q_open, 0.6, "place", settle=0.45)
		# Capture dropped brick inside plate before lifting!
		panel_drop = np.hstack([
			cv2.resize(frames_top_rgb[-1], (480, 390)),
			cv2.resize(frames_front[-1], (520, 390)),
			cv2.resize(frames_wrist[-1], (520, 390)),
		])
		keyframes["place_drop"] = panel_drop
		step_ik_pose(np.array([pink_xy[0], pink_xy[1], 0.14]), brick_yaw, OPEN_G, 0.7, "place")

		# 7. Return
		step_joint_target(q_init[0].cpu().numpy(), 0.9, "return")

		final_brick = brick.data.root_pos_w.torch[0].cpu().numpy()
		dist_pink = float(np.hypot(final_brick[0] - pink_xy[0], final_brick[1] - pink_xy[1]))
		success = bool(grasped and dist_pink < BOWL_R and final_brick[2] < 0.05)

		# Save offline_pipeline / lerobot_export compatible npz & index.json
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
			"source": "isaac-sim",
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

		# Save vertical 4-row keyframe storyboard
		rows_img = [keyframes[k] for k in ("grasp", "lift", "show_hold", "place_drop") if k in keyframes]
		if rows_img:
			cv2.imwrite(str(OUT_DIR / "stage2_keyframes.png"), cv2.cvtColor(np.vstack(rows_img), cv2.COLOR_RGB2BGR))

		summary = {
			"total_frames": len(frames_state),
			"grasped": grasped,
			"max_lift_z_m": float(max_lift_z),
			"best_show_cos": float(best_cos),
			"final_brick_xyz": [float(v) for v in final_brick],
			"dist_to_pink_plate_m": dist_pink,
			"success": success,
			"show_point": [float(v) for v in show_pt],
		}
		(OUT_DIR / "stage2_metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
		print("STAGE2_SUMMARY:", json.dumps(summary, indent=2))


if __name__ == "__main__":
	main()
