"""Stage 3 Verification: Multi-env GPU parallel ManagerBasedEnv + vectorized RTX Camera + Replicator randomization."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Stage 3 parallel GPU env + RTX Replicator randomization benchmark.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of parallel environments.")
parser.add_argument("--steps", type=int, default=20, help="Number of control steps to benchmark.")
add_launcher_args(parser)
args_cli = parser.parse_args([])
args_cli.enable_cameras = True
args_cli.visualizer = []

import cv2
import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import numpy as np
import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from digital_twin_armigo.camera import opengl_axes
from digital_twin_armigo.constants import TOP_CROP, TOP_FULL
from digital_twin_armigo.physics import load_physics_spec

_PHYS = load_physics_spec()

USD_ASSETS = ROOT / "assets"
OUT_DIR = ROOT / "results" / "stage3"

BRICK_COLORS = [
	(0.75, 0.05, 0.05),
	(0.85, 0.65, 0.02),
	(0.05, 0.22, 0.72),
	(0.85, 0.22, 0.03),
	(0.20, 0.65, 0.08),
	(0.35, 0.08, 0.58),
]


@configclass
class ParallelCellSceneCfg(InteractiveSceneCfg):
	ground = AssetBaseCfg(
		prim_path="/World/Ground",
		spawn=sim_utils.GroundPlaneCfg(),
	)
	table = RigidObjectCfg(
		prim_path="{ENV_REGEX_NS}/Table",
		spawn=sim_utils.CuboidCfg(
			size=(1.1, 1.1, 0.02),
			rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
			collision_props=sim_utils.UsdPhysicsCollisionCfg(),
			physics_material=sim_utils.RigidBodyMaterialCfg(
				static_friction=_PHYS.table.static_friction,
				dynamic_friction=_PHYS.table.dynamic_friction,
				restitution=_PHYS.table.restitution,
				friction_combine_mode="max",
				restitution_combine_mode="max",
			),
			visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.66, 0.66, 0.66), roughness=0.85),
		),
		init_state=RigidObjectCfg.InitialStateCfg(pos=(0.25, 0.0, -0.01)),
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
		init_state=ArticulationCfg.InitialStateCfg(
			joint_pos={
				"shoulder_pan": 0.0,
				"shoulder_lift": -0.25,
				"elbow_flex": 0.45,
				"wrist_flex": 1.35,
				"wrist_roll": 0.0,
				"gripper": 1.1,
			}
		),
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
		init_state=RigidObjectCfg.InitialStateCfg(pos=(0.26, 0.01, 0.0096)),
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


@configclass
class ActionsCfg:
	joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], use_default_offset=True)


@configclass
class ObservationsCfg:
	@configclass
	class PolicyCfg(ObsGroup):
		joint_pos = ObsTerm(func=mdp.joint_pos_rel)

		def __post_init__(self):
			self.concatenate_terms = True

	policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
	reset_brick = EventTerm(
		func=mdp.reset_root_state_uniform,
		mode="reset",
		params={
			"asset_cfg": SceneEntityCfg("brick"),
			"pose_range": {"x": (-0.06, 0.06), "y": (-0.08, 0.08), "yaw": (-3.14, 3.14)},
			"velocity_range": {},
		},
	)
	reset_robot = EventTerm(
		func=mdp.reset_joints_by_offset,
		mode="reset",
		params={
			"asset_cfg": SceneEntityCfg("robot"),
			"position_range": (-0.25, 0.25),
			"velocity_range": (0.0, 0.0),
		},
	)
	brick_color = EventTerm(
		func=mdp.randomize_visual_color,
		mode="reset",
		params={
			"event_name": "brick_color",
			"asset_cfg": SceneEntityCfg("brick"),
			"mesh_name": "Mesh",
			"colors": {"r": (0.05, 0.90), "g": (0.05, 0.85), "b": (0.02, 0.85)},
		},
	)
	table_color = EventTerm(
		func=mdp.randomize_visual_color,
		mode="reset",
		params={
			"event_name": "table_color",
			"asset_cfg": SceneEntityCfg("table"),
			"colors": {"r": (0.45, 0.85), "g": (0.45, 0.85), "b": (0.45, 0.85)},
		},
	)


@configclass
class ParallelCellEnvCfg(ManagerBasedEnvCfg):
	scene: ParallelCellSceneCfg = ParallelCellSceneCfg(num_envs=16, env_spacing=1.6, replicate_physics=False)
	actions: ActionsCfg = ActionsCfg()
	observations: ObservationsCfg = ObservationsCfg()
	events: EventCfg = EventCfg()

	def __post_init__(self):
		self.decimation = 4
		self.sim.dt = 1.0 / 120.0
		self.sim.render_interval = self.decimation


def main() -> None:
	OUT_DIR.mkdir(parents=True, exist_ok=True)
	top_cam = json.loads((ROOT / "config" / "sim" / "top_camera_joint.json").read_text())["camera"]
	right, up, z_opengl = opengl_axes(top_cam)
	top_quat = tuple(float(v) for v in Rotation.from_matrix(np.column_stack([right, up, z_opengl])).as_quat())

	env_cfg = ParallelCellEnvCfg()
	env_cfg.scene.num_envs = args_cli.num_envs
	env_cfg.scene.top_cam.spawn.focal_length = 20.955 * float(top_cam["focal_px"]) / TOP_FULL[0]
	env_cfg.scene.top_cam.offset.pos = (float(top_cam["x"]), float(top_cam["y"]), float(top_cam["height"]))
	env_cfg.scene.top_cam.offset.rot = top_quat

	with launch_simulation(env_cfg.sim, args_cli):
		from isaaclab.envs import ManagerBasedEnv

		env = ManagerBasedEnv(cfg=env_cfg)
		torch.cuda.reset_peak_memory_stats()

		obs, _ = env.reset()
		# Step once to settle + trigger RTX Replicator buffers cleanly, then benchmark
		zero_actions = torch.zeros((args_cli.num_envs, env.action_manager.total_action_dim), device=env.device)
		for _ in range(3):
			obs, _ = env.step(zero_actions)

		t0 = time.perf_counter()
		for k in range(args_cli.steps):
			wave = 0.20 * math.sin(2.0 * math.pi * k / max(args_cli.steps, 1))
			act = zero_actions.clone()
			act[:, 0] = wave
			obs, _ = env.step(act)
		torch.cuda.synchronize()
		elapsed = time.perf_counter() - t0

		top_rgb = env.scene["top_cam"].data.output["rgb"].torch[..., :3].cpu().numpy().astype(np.uint8)
		top_depth = env.scene["top_cam"].data.output["distance_to_image_plane"].torch.squeeze(-1).cpu().numpy()
		wrist_rgb = env.scene["wrist_cam"].data.output["rgb"].torch[..., :3].cpu().numpy().astype(np.uint8)
		peak_vram_mb = float(torch.cuda.max_memory_allocated() / (1024.0**2))

		x0, y0, w, h = TOP_CROP
		top_crop = top_rgb[:, y0 : y0 + h, x0 : x0 + w, :]
		# Check inter-env diversity via std of mean RGB across envs and mean pairwise L1
		per_env_means = top_crop.reshape(args_cli.num_envs, -1).mean(axis=1)
		inter_env_std = float(np.std(top_crop.astype(np.float32), axis=0).mean())

		# Build 4x4 mosaic for first 16 envs (or cols=4)
		cols = 4
		rows = max(1, (args_cli.num_envs + cols - 1) // cols)
		tile_w, tile_h = 320, 260
		mosaic = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
		for idx in range(args_cli.num_envs):
			r, c = divmod(idx, cols)
			tile = cv2.resize(top_crop[idx], (tile_w, tile_h))
			cv2.putText(tile, f"env_{idx}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
			mosaic[r * tile_h : (r + 1) * tile_h, c * tile_w : (c + 1) * tile_w] = tile
		cv2.imwrite(str(OUT_DIR / "stage3_mosaic_16envs.png"), cv2.cvtColor(mosaic, cv2.COLOR_RGB2BGR))

		# 2x2 wrist cam mosaic for envs 0..3
		w_mosaic = np.vstack([
			np.hstack([cv2.resize(wrist_rgb[0], (320, 240)), cv2.resize(wrist_rgb[min(1, args_cli.num_envs - 1)], (320, 240))]),
			np.hstack([cv2.resize(wrist_rgb[min(2, args_cli.num_envs - 1)], (320, 240)), cv2.resize(wrist_rgb[min(3, args_cli.num_envs - 1)], (320, 240))]),
		])
		cv2.imwrite(str(OUT_DIR / "stage3_wrist_4envs.png"), cv2.cvtColor(w_mosaic, cv2.COLOR_RGB2BGR))

		total_env_steps = args_cli.num_envs * args_cli.steps
		total_cam_frames = total_env_steps * 2  # top_cam + wrist_cam per env step
		metrics = {
			"num_envs": args_cli.num_envs,
			"steps_benchmarked": args_cli.steps,
			"elapsed_sec": elapsed,
			"env_steps_per_sec": total_env_steps / elapsed,
			"camera_frames_per_sec": total_cam_frames / elapsed,
			"top_rgb_shape": list(top_rgb.shape),
			"top_depth_shape": list(top_depth.shape),
			"wrist_rgb_shape": list(wrist_rgb.shape),
			"inter_env_pixel_std": inter_env_std,
			"per_env_rgb_means": [float(v) for v in per_env_means],
			"torch_peak_vram_mb": peak_vram_mb,
			"obs_policy_shape": list(obs["policy"].shape),
		}
		(OUT_DIR / "stage3_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
		print("STAGE3_METRICS:", json.dumps(metrics, indent=2))
		env.close()


if __name__ == "__main__":
	main()
