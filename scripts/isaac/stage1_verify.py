"""Stage 1 Verification: Static cell in Isaac Sim/Isaac Lab + camera & reprojection comparison."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Stage 1 Isaac Sim single-env cell and camera verification.")
add_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.visualizer = []

import cv2
import isaaclab.sim as sim_utils
import numpy as np
import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from digital_twin_armigo.camera import as_pose, opengl_axes, project
from digital_twin_armigo.constants import COARSE_MAX_M, JOINTS, TOP_CROP, TOP_FULL
from digital_twin_armigo.depth import depth_plane_to_height, layered_height_image
from digital_twin_armigo.joints import normalized_to_rad
from digital_twin_armigo.physics import load_physics_spec

_PHYS = load_physics_spec()

USD_ASSETS = ROOT / "assets"
OUT_DIR = ROOT / "results" / "stage1"
REAL_DIR = ROOT / "results" / "real_ref"


@configclass
class CellSceneCfg(InteractiveSceneCfg):
	table = AssetBaseCfg(
		prim_path="/World/Table",
		spawn=sim_utils.CuboidCfg(
			size=(1.6, 1.6, 0.04),
			collision_props=sim_utils.CollisionPropertiesCfg(),
			visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.66, 0.66, 0.66), roughness=0.85),
		),
		init_state=AssetBaseCfg.InitialStateCfg(pos=(0.25, 0.0, -0.02)),
	)
	dome_light = AssetBaseCfg(
		prim_path="/World/DomeLight",
		spawn=sim_utils.DomeLightCfg(intensity=1200.0, color=(0.95, 0.95, 0.95)),
	)
	sun_light = AssetBaseCfg(
		prim_path="/World/SunLight",
		spawn=sim_utils.DistantLightCfg(intensity=2200.0, color=(1.0, 0.98, 0.94), angle=2.0),
		init_state=AssetBaseCfg.InitialStateCfg(rot=(0.22, 0.12, 0.0, 0.968)),
	)
	robot = ArticulationCfg(
		prim_path="{ENV_REGEX_NS}/Robot",
		spawn=sim_utils.UsdFileCfg(usd_path=str(USD_ASSETS / "so101_usd" / "so101_physx.usda")),
		actuators={
			"sts3215_arm": ImplicitActuatorCfg(
				joint_names_expr=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
				stiffness=_PHYS.arm_actuator.stiffness,
				damping=_PHYS.arm_actuator.damping,
				effort_limit_sim=_PHYS.arm_actuator.effort_limit,
				velocity_limit_sim=_PHYS.arm_actuator.velocity_limit,
				armature=_PHYS.arm_actuator.armature,
				friction=_PHYS.arm_actuator.frictionloss,
				dynamic_friction=_PHYS.arm_actuator.frictionloss,
				viscous_friction=_PHYS.arm_actuator.joint_damping,
			),
			"sts3215_gripper": ImplicitActuatorCfg(
				joint_names_expr=["gripper"],
				stiffness=_PHYS.gripper_actuator.stiffness,
				damping=_PHYS.gripper_actuator.damping,
				effort_limit_sim=_PHYS.gripper_actuator.effort_limit,
				velocity_limit_sim=_PHYS.gripper_actuator.velocity_limit,
				armature=_PHYS.gripper_actuator.armature,
				friction=_PHYS.gripper_actuator.frictionloss,
				dynamic_friction=_PHYS.gripper_actuator.frictionloss,
				viscous_friction=_PHYS.gripper_actuator.joint_damping,
			),
		},
	)
	green_plate = RigidObjectCfg(
		prim_path="{ENV_REGEX_NS}/GreenPlate",
		spawn=sim_utils.UsdFileCfg(usd_path=str(USD_ASSETS / "props" / "green_plate.usda")),
		init_state=RigidObjectCfg.InitialStateCfg(pos=(0.26, 0.18, 0.0)),
	)
	pink_plate = RigidObjectCfg(
		prim_path="{ENV_REGEX_NS}/PinkPlate",
		spawn=sim_utils.UsdFileCfg(usd_path=str(USD_ASSETS / "props" / "pink_plate.usda")),
		init_state=RigidObjectCfg.InitialStateCfg(pos=(0.45, 0.23, 0.0)),
	)
	brick = RigidObjectCfg(
		prim_path="{ENV_REGEX_NS}/Brick",
		spawn=sim_utils.UsdFileCfg(usd_path=str(USD_ASSETS / "props" / "duplo_2x2_orange.usda")),
		init_state=RigidObjectCfg.InitialStateCfg(pos=(0.35, 0.0, 0.0096)),
	)
	top_cam = CameraCfg(
		prim_path="{ENV_REGEX_NS}/TopCamera",
		update_period=0.0,
		height=TOP_FULL[1],
		width=TOP_FULL[0],
		data_types=["rgb", "distance_to_image_plane"],
		spawn=sim_utils.PinholeCameraCfg(
			focal_length=20.0,
			horizontal_aperture=20.955,
			clipping_range=(0.05, 10.0),
		),
		offset=CameraCfg.OffsetCfg(pos=(0.6, 0.2, 0.5), rot=(0.0, 0.0, 0.0, 1.0), convention="opengl"),
	)
	wrist_cam = CameraCfg(
		prim_path="{ENV_REGEX_NS}/Robot/Geometry/base/shoulder/upper_arm/lower_arm/wrist/gripper/camera_mount/wrist_cam",
		update_period=0.0,
		height=480,
		width=640,
		data_types=["rgb", "distance_to_image_plane"],
		spawn=None,
	)
	front_cam = CameraCfg(
		prim_path="{ENV_REGEX_NS}/FrontCamera",
		update_period=0.0,
		height=480,
		width=640,
		data_types=["rgb"],
		spawn=sim_utils.PinholeCameraCfg(
			focal_length=18.0,
			horizontal_aperture=20.955,
			clipping_range=(0.05, 10.0),
		),
		offset=CameraCfg.OffsetCfg(pos=(0.70, -0.32, 0.35), rot=(0.0, 0.0, 0.0, 1.0), convention="opengl"),
	)


def build_scene_cfg(top_cam: dict, ep2_prior: dict, brick_xy: tuple[float, float]) -> CellSceneCfg:
	right, up, z_opengl = opengl_axes(top_cam)
	rot_mat = np.column_stack([right, up, z_opengl])
	top_quat_xyzw = tuple(float(v) for v in Rotation.from_matrix(rot_mat).as_quat())
	# Isaac PinholeCameraCfg: fx = width * focal_length / horizontal_aperture
	h_ap = 20.955
	top_focal_mm = h_ap * float(top_cam["focal_px"]) / TOP_FULL[0]

	# Front overview camera looking toward workspace center (0.25, 0.05, 0.05) from front-right
	eye = np.array([0.70, -0.32, 0.35])
	target = np.array([0.24, 0.06, 0.04])
	fwd = target - eye
	fwd /= np.linalg.norm(fwd)
	r_f = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
	r_f /= np.linalg.norm(r_f)
	u_f = np.cross(r_f, fwd)
	front_quat_xyzw = tuple(float(v) for v in Rotation.from_matrix(np.column_stack([r_f, u_f, -fwd])).as_quat())

	cfg = CellSceneCfg(num_envs=1, env_spacing=2.0)
	cfg.green_plate.init_state.pos = (float(ep2_prior["green"][0]), float(ep2_prior["green"][1]), 0.0)
	cfg.pink_plate.init_state.pos = (float(ep2_prior["pink"][0]), float(ep2_prior["pink"][1]), 0.0)
	cfg.brick.init_state.pos = (float(brick_xy[0]), float(brick_xy[1]), 0.0096)
	cfg.brick.init_state.rot = (0.0, 0.0, math.sin(math.radians(20.0)), math.cos(math.radians(20.0)))
	cfg.top_cam.spawn.focal_length = top_focal_mm
	cfg.top_cam.offset.pos = (float(top_cam["x"]), float(top_cam["y"]), float(top_cam["height"]))
	cfg.top_cam.offset.rot = top_quat_xyzw
	cfg.front_cam.offset.pos = tuple(float(v) for v in eye)
	cfg.front_cam.offset.rot = front_quat_xyzw
	return cfg


def main() -> None:
	OUT_DIR.mkdir(parents=True, exist_ok=True)
	top_cam = json.loads((ROOT / "config" / "sim" / "top_camera_joint.json").read_text())["camera"]
	prior = json.loads((ROOT / "config" / "sim" / "prior_grasp.json").read_text())
	ep2_prior = next(p for p in prior if p["episode"] == 2)
	grasp_rows = json.loads((ROOT / "config" / "sim" / "grasp_points.json").read_text())
	ep2_grasp = next(r for r in grasp_rows if r["episode"] == 2)
	meta_path = (
		REAL_DIR / "frame_meta.json"
		if (REAL_DIR / "frame_meta.json").is_file()
		else ROOT / "config" / "sim" / "frame_3217_state.json"
	)
	frame_meta = json.loads(meta_path.read_text())
	q_rad = normalized_to_rad(np.array(frame_meta["observation.state"]))

	sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, device="cuda:0")
	with launch_simulation(sim_cfg, args_cli):
		from isaaclab.scene import InteractiveScene

		sim = sim_utils.SimulationContext(sim_cfg)
		scene = InteractiveScene(build_scene_cfg(top_cam, ep2_prior, tuple(ep2_grasp["jaw_centre"][:2])))
		sim.reset()

		robot = scene["robot"]
		joint_names = list(robot.joint_names)
		print("Robot joint names:", joint_names)
		name_to_q = dict(zip(JOINTS, q_rad, strict=True))
		q_ordered = torch.tensor([[name_to_q[n] for n in joint_names]], device=robot.device, dtype=torch.float32)
		q_vel = torch.zeros_like(q_ordered)

		for _ in range(15):
			robot.write_joint_position_to_sim_index(position=q_ordered)
			robot.write_joint_velocity_to_sim_index(velocity=q_vel)
			robot.set_joint_position_target_index(target=q_ordered)
			scene.write_data_to_sim()
			sim.step()
			scene.update(sim.get_physics_dt())

		# Extra render updates to ensure RTX cameras flush current pose cleanly
		for _ in range(4):
			sim.render()
			scene.update(sim.get_physics_dt())

		top_sensor = scene["top_cam"]
		wrist_sensor = scene["wrist_cam"]
		front_sensor = scene["front_cam"]

		top_rgb_full = top_sensor.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
		top_depth_full = top_sensor.data.output["distance_to_image_plane"][0].squeeze(-1).cpu().numpy()
		wrist_rgb = wrist_sensor.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
		front_rgb = front_sensor.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
		K_isaac = top_sensor.data.intrinsic_matrices[0].cpu().numpy()

		# Crop top camera to real depth_bridge 512x416 region
		x0, y0, w, h = TOP_CROP
		top_rgb_crop = top_rgb_full[y0 : y0 + h, x0 : x0 + w]
		z_world, valid = depth_plane_to_height(top_cam, top_depth_full)
		top_depth_layers = layered_height_image(z_world[y0 : y0 + h, x0 : x0 + w], valid[y0 : y0 + h, x0 : x0 + w])

		# Save Isaac rendered frames
		cv2.imwrite(str(OUT_DIR / "isaac_top_rgb_crop.png"), cv2.cvtColor(top_rgb_crop, cv2.COLOR_RGB2BGR))
		cv2.imwrite(str(OUT_DIR / "isaac_top_depth_crop.png"), cv2.cvtColor(top_depth_layers, cv2.COLOR_RGB2BGR))
		cv2.imwrite(str(OUT_DIR / "isaac_wrist_rgb.png"), cv2.cvtColor(wrist_rgb, cv2.COLOR_RGB2BGR))
		cv2.imwrite(str(OUT_DIR / "isaac_front_rgb.png"), cv2.cvtColor(front_rgb, cv2.COLOR_RGB2BGR))

		sim_mask = (z_world[y0 : y0 + h, x0 : x0 + w] > 0.04) & valid[y0 : y0 + h, x0 : x0 + w]
		arm_iou: float | None = None
		real_mask_px: int | None = None
		if (REAL_DIR / "top_rgb_v2_3217.png").is_file() and (REAL_DIR / "top_depth_v2_3217.png").is_file():
			real_top_rgb = cv2.cvtColor(cv2.imread(str(REAL_DIR / "top_rgb_v2_3217.png")), cv2.COLOR_BGR2RGB)
			real_top_depth = cv2.cvtColor(cv2.imread(str(REAL_DIR / "top_depth_v2_3217.png")), cv2.COLOR_BGR2RGB)
			real_mask = real_top_depth[..., 1] > int(255 * 0.04 / COARSE_MAX_M)
			inter = int(np.logical_and(sim_mask, real_mask).sum())
			union = int(np.logical_or(sim_mask, real_mask).sum())
			arm_iou = inter / max(union, 1)
			real_mask_px = int(real_mask.sum())

			top_cmp = np.hstack([real_top_rgb, top_rgb_crop])
			overlay = real_top_rgb.copy()
			overlay[real_mask] = (overlay[real_mask].astype(np.int32) * 0.5 + np.array([255, 0, 0]) * 0.5).astype(
				np.uint8
			)
			overlay[sim_mask] = (overlay[sim_mask].astype(np.int32) * 0.5 + np.array([0, 255, 0]) * 0.5).astype(
				np.uint8
			)
			depth_cmp = np.hstack([real_top_depth, top_depth_layers, overlay])
			cv2.imwrite(str(OUT_DIR / "top_compare_3217.png"), cv2.cvtColor(top_cmp, cv2.COLOR_RGB2BGR))
			cv2.imwrite(str(OUT_DIR / "depth_compare_3217.png"), cv2.cvtColor(depth_cmp, cv2.COLOR_RGB2BGR))
			if (REAL_DIR / "wrist_3217.png").is_file():
				real_wrist_rgb = cv2.cvtColor(cv2.imread(str(REAL_DIR / "wrist_3217.png")), cv2.COLOR_BGR2RGB)
				cv2.imwrite(
					str(OUT_DIR / "wrist_compare_3217.png"),
					cv2.cvtColor(np.hstack([real_wrist_rgb, wrist_rgb]), cv2.COLOR_RGB2BGR),
				)

		# Project all 26 calibrated grasp points using Isaac Camera K & world pose
		cam_pos = top_sensor.data.pos_w[0].cpu().numpy()
		cam_quat_opengl = top_sensor.data.quat_w_opengl[0].cpu().numpy()  # (x, y, z, w)
		R_cam2world = Rotation.from_quat(cam_quat_opengl).as_matrix()
		rows = [r for r in grasp_rows if r["arm_over_brick_m"] > 0.045 and r["px"][0] < 330]
		pts_3d = np.array([[*r["jaw_centre"][:2], r["brick_top"]] for r in rows])
		img_real = np.array([r["px"] for r in rows])

		# Transform world points to Isaac camera OpenGL frame -> right=+X, up=+Y, depth=-Z
		pts_cam = (pts_3d - cam_pos) @ R_cam2world
		z_fwd = -pts_cam[:, 2]
		u_isaac = K_isaac[0, 0] * (pts_cam[:, 0] / z_fwd) + K_isaac[0, 2]
		v_isaac = K_isaac[1, 1] * (-pts_cam[:, 1] / z_fwd) + K_isaac[1, 2]
		uv_isaac = np.stack([u_isaac, v_isaac], axis=-1)
		uv_mj, _ = project(as_pose(top_cam), pts_3d, TOP_FULL)

		err_real_px = np.linalg.norm(uv_isaac - img_real, axis=1)
		err_mj_px = np.linalg.norm(uv_isaac - uv_mj, axis=1)

		metrics = {
			"joint_names": joint_names,
			"q_rad_frame_3217": [float(v) for v in q_rad],
			"K_isaac": K_isaac.tolist(),
			"focal_diff_px": float(abs(K_isaac[0, 0] - top_cam["focal_px"])),
			"arm_silhouette_iou_frame_3217": arm_iou,
			"sim_arm_mask_px": int(sim_mask.sum()),
			"real_arm_mask_px": real_mask_px,
			"grasp_reproj_median_vs_real_px": float(np.median(err_real_px)),
			"grasp_reproj_mean_vs_real_px": float(np.mean(err_real_px)),
			"grasp_reproj_max_vs_mujoco_px": float(np.max(err_mj_px)),
			"wrist_rgb_shape": list(wrist_rgb.shape),
			"wrist_rgb_mean": [float(v) for v in wrist_rgb.mean(axis=(0, 1))],
		}
		(OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
		print("STAGE1_METRICS:", json.dumps(metrics, indent=2))


if __name__ == "__main__":
	main()
