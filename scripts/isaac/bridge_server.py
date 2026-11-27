"""Isaac bridge server: Isaac Lab / PhysX behind a localhost ``multiprocessing.connection`` socket.

Run it with the Isaac Python (``E:\\il\\v\\Scripts\\python.exe``); ``digital_twin_armigo.isaac_bridge.IsaacBridge`` spawns
it. One request at a time. See ``docs/ISAAC_BRIDGE.md`` for the protocol and the conventions.

The scene is the stage 2 cell of ``stage2_expert.py`` (same robot USD, drives, contact sensors, joint-order helpers) built
from the ``scene`` dict of ``IsaacBridge.reset`` instead of command-line options. Physics advances only in ``step``
(8 x 1/240 s per control frame, ``sim.step(render=False)``); ``render`` never advances physics time.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import sys
import time
import traceback
from multiprocessing.connection import Listener
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

parser = argparse.ArgumentParser(description="Isaac bridge server.")
parser.add_argument("--port", type=int, required=True)
parser.add_argument("--authkey-hex", required=True)
parser.add_argument("--headless", action="store_true", help="always headless; accepted for symmetry with the client")
parser.add_argument("--log-dir", type=Path, default=None)
parser.add_argument("--flush-updates", type=int, default=3, help="render-only app updates before a kept capture")
add_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
args_cli.visualizer = []

import isaaclab.sim as sim_utils
import numpy as np
import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.utils import configclass
from scipy.spatial.transform import Rotation

from digital_twin_armigo.constants import JOINTS
from digital_twin_armigo.physics import physics_spec_from_dict


def _load_codec():
	path = ROOT / "src" / "digital_twin_armigo" / "isaac_bridge" / "codec.py"
	spec = importlib.util.spec_from_file_location("bridge_codec", path)
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module


codec = _load_codec()

USD_ASSETS = ROOT / "assets"
PHYSICS_HZ = 240
FPS = 30
SUBSTEPS = PHYSICS_HZ // FPS
GRIPPER_OPEN_MIN = 0.5  # rad; a checkpoint needs the gripper at least this open
SUPPORTED_MAX_Z = 0.03  # m; a checkpointed brick rests on the table or a bowl floor
SUPPORTED_MAX_SPEED = 0.05  # m/s
SUPPORTED_MAX_SPIN = 1.0  # rad/s
JAW_CONTACT_N = 0.02  # N; a jaw touching the brick above this is a grasp
CAM_PRIMS = {
	"top": "top_cam",
	"front": "front_cam",
	"wrist_cam": "wrist_cam",
}
ROBOT_LINKS = "{ENV_REGEX_NS}/Robot/Geometry/base/shoulder/upper_arm/lower_arm/wrist/gripper"
H_APERTURE = 20.955


class BridgeError(RuntimeError):
	pass


def log(*parts) -> None:
	print(time.strftime("%H:%M:%S"), *parts, flush=True)


def _quat_from_rot(rot: np.ndarray) -> tuple[float, ...]:
	return tuple(float(v) for v in Rotation.from_matrix(rot).as_quat())


def _pinhole(cam: dict) -> sim_utils.PinholeCameraCfg:
	focal = H_APERTURE * cam["fx"] / cam["width"]
	return sim_utils.PinholeCameraCfg(
		focal_length=focal,
		horizontal_aperture=H_APERTURE,
		vertical_aperture=focal * cam["height"] / cam["fy"],
		clipping_range=(0.05, 10.0),
	)


def build_scene_cfg(p: dict, spec) -> InteractiveSceneCfg:
	"""The stage 2 scene (``stage2_expert.build_scene_cfg``) from a bridge scene payload."""
	brick = p["brick"]
	brick_usd = USD_ASSETS / "props" / f"duplo_{brick['size_key']}_orange.usda"
	arm_act, grip_act = spec.arm_actuator, spec.gripper_actuator
	green, pink = p["bowls"]["green"]["xy"], p["bowls"]["pink"]["xy"]
	cams = p["cameras"]
	top, front, wrist = cams["top"], cams["front"], cams["wrist_cam"]
	level = float(p["light_level"]) / 0.7

	@configclass
	class BridgeSceneCfg(InteractiveSceneCfg):
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
				visual_material=sim_utils.PreviewSurfaceCfg(
					diffuse_color=tuple(float(c) for c in p["table"]["rgb"]), roughness=0.9
				),
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
			init_state=RigidObjectCfg.InitialStateCfg(pos=(float(green[0]), float(green[1]), 0.0)),
		)
		pink_plate = RigidObjectCfg(
			prim_path="{ENV_REGEX_NS}/PinkPlate",
			spawn=sim_utils.UsdFileCfg(usd_path=str(USD_ASSETS / "props" / "pink_plate.usda")),
			init_state=RigidObjectCfg.InitialStateCfg(pos=(float(pink[0]), float(pink[1]), 0.0)),
		)
		brick = RigidObjectCfg(
			prim_path="{ENV_REGEX_NS}/Brick",
			spawn=sim_utils.UsdFileCfg(
				usd_path=str(brick_usd),
				activate_contact_sensors=True,
				mass_props=sim_utils.MassPropertiesCfg(mass=float(p["brick"]["mass_kg"])),
				rigid_props=sim_utils.PhysxRigidBodyPropertiesCfg(
					solver_position_iteration_count=32,
					solver_velocity_iteration_count=8,
					linear_damping=0.25,
					angular_damping=0.25,
					max_depenetration_velocity=1.0,
				),
			),
			init_state=RigidObjectCfg.InitialStateCfg(
				pos=tuple(float(v) for v in p["brick"]["pos"]), rot=tuple(float(v) for v in p["brick"]["quat_xyzw"])
			),
		)
		contact_fixed = ContactSensorCfg(
			prim_path=ROBOT_LINKS,
			filter_prim_paths_expr=["{ENV_REGEX_NS}/Brick"],
			update_period=0.0,
			history_length=SUBSTEPS,
			track_friction_forces=True,
		)
		contact_moving = ContactSensorCfg(
			prim_path=ROBOT_LINKS + "/moving_jaw_so101_v1",
			filter_prim_paths_expr=["{ENV_REGEX_NS}/Brick"],
			update_period=0.0,
			history_length=SUBSTEPS,
			track_friction_forces=True,
		)
		contact_arm_table = ContactSensorCfg(
			prim_path=ROBOT_LINKS.rsplit("/", 1)[0],
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
		dome_light = AssetBaseCfg(
			prim_path="/World/DomeLight",
			spawn=sim_utils.DomeLightCfg(intensity=650.0 * level, color=(0.95, 0.95, 0.95)),
		)
		sun_light = AssetBaseCfg(
			prim_path="/World/SunLight",
			spawn=sim_utils.DistantLightCfg(intensity=1400.0 * level, color=(1.0, 0.98, 0.94), angle=2.0),
			init_state=AssetBaseCfg.InitialStateCfg(rot=(0.22, 0.12, 0.0, 0.968)),
		)
		top_cam = CameraCfg(
			prim_path="{ENV_REGEX_NS}/TopCamera",
			update_period=0.0,
			height=top["height"],
			width=top["width"],
			data_types=["rgb", "distance_to_image_plane"],
			spawn=_pinhole(top),
			offset=CameraCfg.OffsetCfg(
				pos=tuple(float(v) for v in top["pos"]),
				rot=_quat_from_rot(np.asarray(top["rot"])),
				convention="opengl",
			),
		)
		front_cam = CameraCfg(
			prim_path="{ENV_REGEX_NS}/FrontCamera",
			update_period=0.0,
			height=front["height"],
			width=front["width"],
			data_types=["rgb", "distance_to_image_plane"],
			spawn=_pinhole(front),
			offset=CameraCfg.OffsetCfg(
				pos=tuple(float(v) for v in front["pos"]),
				rot=_quat_from_rot(np.asarray(front["rot"])),
				convention="opengl",
			),
		)
		wrist_cam = CameraCfg(
			prim_path=ROBOT_LINKS + "/camera_mount/wrist_cam",
			update_period=0.0,
			height=wrist["height"],
			width=wrist["width"],
			data_types=["rgb", "distance_to_image_plane"],
			spawn=None,
		)

	return BridgeSceneCfg(num_envs=1, env_spacing=2.0)


def _author_stage(stage, p: dict) -> None:
	"""Edits of the spawned scene that Isaac Lab configs cannot express: brick colour, the defect boxes, the actual
	wrist camera mount and intrinsics."""
	from pxr import Gf, Sdf, UsdGeom, UsdShade

	env = "/World/envs/env_0"
	brick_prim = f"{env}/Brick"
	# brick colour
	shader = None
	for prim in stage.Traverse():
		path = str(prim.GetPath())
		if path.startswith(brick_prim + "/Looks/") and path.endswith("/Shader"):
			shader = UsdShade.Shader(prim)
			break
	if shader is None:
		raise BridgeError("brick material not found")
	shader.GetInput("diffuseColor").Set(Gf.Vec3f(*(float(c) for c in p["brick"]["rgb"])))
	# defect: visible boxes on a defective brick, hidden ones (kept for the record geometry check) otherwise
	d = p["defect"]
	mat = UsdShade.Material.Define(stage, f"{brick_prim}/Looks/defect_ink")
	ink = UsdShade.Shader.Define(stage, f"{brick_prim}/Looks/defect_ink/Shader")
	ink.CreateIdAttr("UsdPreviewSurface")
	ink.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*(float(c) for c in d["rgb"])))
	ink.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.45)
	mat.CreateSurfaceOutput().ConnectToSource(ink.ConnectableAPI(), "surface")
	for k, box in enumerate(d["boxes"]):
		cube = UsdGeom.Cube.Define(stage, f"{brick_prim}/defect_{k}")
		cube.CreateSizeAttr(1.0)
		xf = UsdGeom.Xformable(cube.GetPrim())
		xf.AddTranslateOp().Set(Gf.Vec3d(*(float(v) for v in box["pos"])))
		qx, qy, qz, qw = (float(v) for v in box["quat_xyzw"])
		xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(qw, Gf.Vec3d(qx, qy, qz)))
		xf.AddScaleOp().Set(Gf.Vec3d(*(2.0 * float(h) for h in box["half"])))
		UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(mat)
		if not d["visible"]:
			UsdGeom.Imageable(cube.GetPrim()).MakeInvisible()
	# wrist camera: actual mount (designed mount + layout.wrist_cam_offset) and intrinsics
	w = p["cameras"]["wrist_cam"]
	cam_prim = stage.GetPrimAtPath(
		f"{env}/Robot/Geometry/base/shoulder/upper_arm/lower_arm/wrist/gripper/camera_mount/wrist_cam"
	)
	if not cam_prim.IsValid():
		raise BridgeError("wrist_cam prim not found")
	xf = UsdGeom.Xformable(cam_prim)
	ops = {op.GetOpName().split(":")[-1]: op for op in xf.GetOrderedXformOps()}
	translate = next(op for name, op in ops.items() if name == "translate")
	orient = next(op for name, op in ops.items() if name == "orient")
	mp = w["mount_pos"]
	mq = w["mount_quat_wxyz"]
	translate.Set(Gf.Vec3d(*(float(v) for v in mp)))
	if orient.GetPrecision() == UsdGeom.XformOp.PrecisionFloat:
		orient.Set(Gf.Quatf(float(mq[0]), Gf.Vec3f(float(mq[1]), float(mq[2]), float(mq[3]))))
	else:
		orient.Set(Gf.Quatd(float(mq[0]), Gf.Vec3d(float(mq[1]), float(mq[2]), float(mq[3]))))
	cam = UsdGeom.Camera(cam_prim)
	focal = 24.0
	cam.GetFocalLengthAttr().Set(focal)
	cam.GetHorizontalApertureAttr().Set(focal * w["width"] / w["fx"])
	cam.GetVerticalApertureAttr().Set(focal * w["height"] / w["fy"])
	cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 10.0))


class Bridge:
	def __init__(self, sim_cfg) -> None:
		self.sim_cfg = sim_cfg
		self.sim = None
		self.scene = None
		self.capture = None
		self.signature = None
		self.payload = None
		self.checkpoints: dict[str, dict] = {}
		self.frame = 0
		self.step_count = 0
		self.last_ctrl = None
		self.flush_updates = max(1, int(args_cli.flush_updates))
		self.rows = {"branch": [], "frame": [], "sim_time": []}
		self.render_calls = 0
		self.timing: dict[str, list[float]] = {}
		self.sub_acc = [0.0, 0.0, 0.0]
		self._stale_render = True

	# -- build ---------------------------------------------------------------------------------------------------
	def _teardown(self) -> None:
		self.capture = None
		self.scene = None
		self.sim = None
		self.checkpoints.clear()
		gc.collect()
		if sim_utils.SimulationContext.instance() is not None:
			sim_utils.SimulationContext.clear_instance()
		gc.collect()

	def _build(self, p: dict) -> None:
		import omni.usd
		from isaaclab.scene import InteractiveScene

		t0 = time.perf_counter()
		if self.sim is not None:
			self._teardown()
		spec = physics_spec_from_dict(p["physics"])
		cfg = build_scene_cfg(p, spec)
		self.sim = sim_utils.SimulationContext(self.sim_cfg)
		self.scene = InteractiveScene(cfg)
		stage = omni.usd.get_context().get_stage()
		_author_stage(stage, p)
		self.sim.reset()
		robot = self.scene["robot"]
		self.robot, self.brick = robot, self.scene["brick"]
		self.c_fixed, self.c_moving = self.scene["contact_fixed"], self.scene["contact_moving"]
		self.c_arm = self.scene["contact_arm_table"]
		names = list(robot.joint_names)
		self.j_idx = {n: i for i, n in enumerate(names)}
		self.order = [self.j_idx[n] for n in JOINTS]
		self.nj = len(names)
		self.dt = self.sim.get_physics_dt()
		self.device = robot.device
		self.q_init = np.asarray(p["q_init"], dtype=np.float64)
		layout = json.loads(json.dumps(_plain(p)))
		self.capture = BridgeCapture(stage, layout, self.scene)
		self.capture.add_front_camera(stage)
		self._cam_static = {
			"top": _pose_matrix(p["cameras"]["top"]),
			"front": _pose_matrix(p["cameras"]["front"]),
		}
		log(f"scene built in {time.perf_counter() - t0:.1f} s ({len(self.capture.prims)} recorded nodes)")

	def _to_sim(self, q6: np.ndarray) -> torch.Tensor:
		out = np.zeros((1, self.nj), dtype=np.float32)
		out[0, self.order] = q6
		return torch.tensor(out, device=self.device)

	def _apply_initial(self) -> None:
		p, robot = self.payload, self.robot
		q = self._to_sim(self.q_init)
		robot.write_joint_position_to_sim_index(position=q)
		robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q))
		robot.actuators.target_command.set_position_index(value=q)
		b = p["brick"]
		state = torch.tensor([[*b["pos"], *b["quat_xyzw"], 0, 0, 0, 0, 0, 0]], device=self.device, dtype=torch.float32)
		self.brick.write_root_state_to_sim(state)
		self.scene.write_data_to_sim()
		self.last_ctrl = self.q_init.copy()

	def reset(self, payload: dict) -> dict:
		if payload.get("version") != 1:
			raise BridgeError(f"unsupported scene version {payload.get('version')}")
		signature = hashlib.sha256(codec.dumps(payload)).hexdigest()
		self.payload = payload
		if signature != self.signature or self.sim is None:
			self.signature = None
			self._build(payload)
			self.signature = signature
		else:
			self.scene.reset()
			self.checkpoints.clear()
			log("scene reset in place")
		self._apply_initial()
		for _ in range(int(payload["warmup_steps"])):
			self._substep()
		self.frame = 0
		self.step_count = 0
		self.capture.frames.clear()
		self.capture.times.clear()
		self.capture.jaw_env.clear()
		self.capture.camera_frames.clear()
		for v in self.rows.values():
			v.clear()
		self._stale_render = True
		return self.read_state()

	# -- physics -------------------------------------------------------------------------------------------------
	def _substep(self) -> None:
		t0 = time.perf_counter()
		self.scene.write_data_to_sim()
		t1 = time.perf_counter()
		self.sim.step(render=False)
		t2 = time.perf_counter()
		self.scene.update(self.dt)
		t3 = time.perf_counter()
		acc = self.sub_acc
		acc[0] += t1 - t0
		acc[1] += t2 - t1
		acc[2] += t3 - t2

	def step(self, ctrl, branch: int, frame: int) -> dict:
		self._need_scene()
		robot = self.robot
		target = self._to_sim(np.asarray(ctrl, dtype=np.float64))
		robot.actuators.target_command.set_position_index(value=target)
		t0 = time.perf_counter()
		self.sub_acc = [0.0, 0.0, 0.0]
		for _ in range(SUBSTEPS):
			self._substep()
		t1 = time.perf_counter()
		for k, v in zip(("sub_write_data", "sub_physx_step", "sub_scene_update"), self.sub_acc, strict=True):
			self.timing.setdefault(k, []).append(v * 1e3)
		self.step_count += 1
		self.frame = int(frame)
		self.last_ctrl = np.asarray(ctrl, dtype=np.float64).copy()
		state = self.read_state()
		t2 = time.perf_counter()
		self.capture.frame(self.step_count / FPS, None)
		t3 = time.perf_counter()
		for k, v in (("substeps", t1 - t0), ("state", t2 - t1), ("record", t3 - t2)):
			self.timing.setdefault(k, []).append(v * 1e3)
		self.rows["branch"].append(int(branch))
		self.rows["frame"].append(int(frame))
		self.rows["sim_time"].append(state["sim_time"])
		self._stale_render = True
		return state

	def _need_scene(self) -> None:
		if self.sim is None:
			raise BridgeError("no scene: call reset first")

	def _jaw_hist(self, sensor) -> tuple[float, float]:
		"""(total force norm now, mean over the control frame's substeps of the normal force norm)."""
		data = sensor.data
		total = torch.linalg.norm(data.force_matrix_w.torch[0, 0, 0])
		nh = getattr(data, "normal_force_matrix_w_history", None)
		th = getattr(data, "force_matrix_w_history", None)
		src = nh if nh is not None else th
		if src is not None:
			mean = torch.linalg.norm(src.torch[0, :, 0, 0], dim=-1).mean()
		else:
			mean = total
		return total, mean

	def read_state(self) -> dict:
		robot, brick = self.robot, self.brick
		f_fix, n_fix = self._jaw_hist(self.c_fixed)
		f_mov, n_mov = self._jaw_hist(self.c_moving)
		hist = getattr(self.c_arm.data, "force_matrix_w_history", None)
		if hist is not None:
			unint = torch.linalg.norm(hist.torch[0, :, 0, :], dim=-1).max()
		else:
			unint = torch.linalg.norm(self.c_arm.data.force_matrix_w.torch[0, 0, :], dim=-1).max()
		flat = torch.cat(
			[
				robot.data.joint_pos.torch[0],
				robot.data.joint_vel.torch[0],
				robot.data.applied_torque.torch[0],
				brick.data.root_pos_w.torch[0],
				brick.data.root_quat_w.torch[0],
				brick.data.root_lin_vel_w.torch[0],
				brick.data.root_ang_vel_w.torch[0],
				torch.stack([f_fix, f_mov, n_fix, n_mov, unint]),
			]
		)
		v = flat.cpu().numpy().astype(np.float64)
		nj = self.nj
		o = self.order
		return {
			"q": v[0:nj][o],
			"qd": v[nj : 2 * nj][o],
			"effort": v[2 * nj : 3 * nj][o],
			"brick_pos": v[3 * nj : 3 * nj + 3],
			"brick_quat_xyzw": v[3 * nj + 3 : 3 * nj + 7],
			"brick_linvel": v[3 * nj + 7 : 3 * nj + 10],
			"brick_angvel": v[3 * nj + 10 : 3 * nj + 13],
			"jaw_force": v[3 * nj + 13 : 3 * nj + 15],
			"jaw_normal": v[3 * nj + 15 : 3 * nj + 17],
			"unintended_force": float(v[3 * nj + 17]),
			"frame": int(self.frame),
			"sim_time": self.step_count / FPS,
		}

	# -- checkpoints ---------------------------------------------------------------------------------------------
	def checkpoint(self) -> dict:
		self._need_scene()
		s = self.read_state()
		if s["q"][5] < GRIPPER_OPEN_MIN or (self.last_ctrl is not None and self.last_ctrl[5] < GRIPPER_OPEN_MIN):
			raise BridgeError(f"checkpoint needs the gripper open (q={s['q'][5]:.3f}, cmd={self.last_ctrl[5]:.3f})")
		if float(np.max(s["jaw_force"])) > JAW_CONTACT_N:
			raise BridgeError(f"checkpoint refused: a jaw touches the brick ({s['jaw_force']} N)")
		speed, spin = float(np.linalg.norm(s["brick_linvel"])), float(np.linalg.norm(s["brick_angvel"]))
		if s["brick_pos"][2] > SUPPORTED_MAX_Z or speed > SUPPORTED_MAX_SPEED or spin > SUPPORTED_MAX_SPIN:
			raise BridgeError(
				f"checkpoint refused: brick not table-supported (z={s['brick_pos'][2]:.4f}, v={speed:.3f}, w={spin:.3f})"
			)
		token = f"ckpt-{len(self.checkpoints):04d}-{self.step_count}"
		self.checkpoints[token] = {
			"joint_pos": self.robot.data.joint_pos.torch.clone(),
			"joint_vel": self.robot.data.joint_vel.torch.clone(),
			"rigids": {n: o.data.root_state_w.torch.clone() for n, o in self.scene.rigid_objects.items()},
			"step_count": self.step_count,
			"frame": self.frame,
			"ctrl": self.last_ctrl.copy(),
		}
		return {"token": token}

	def restore(self, token: str) -> dict:
		self._need_scene()
		ck = self.checkpoints.get(token)
		if ck is None:
			raise BridgeError(f"unknown checkpoint {token!r}")
		self.scene.reset()
		robot = self.robot
		robot.write_joint_position_to_sim_index(position=ck["joint_pos"])
		robot.write_joint_velocity_to_sim_index(velocity=ck["joint_vel"])
		target = self._to_sim(ck["ctrl"])
		robot.actuators.target_command.set_position_index(value=target)
		for name, state in ck["rigids"].items():
			self.scene[name].write_root_state_to_sim(state)
		self.scene.write_data_to_sim()
		self.scene.update(0.0)
		self.step_count = ck["step_count"]
		self.frame = ck["frame"]
		self.last_ctrl = ck["ctrl"].copy()
		self._stale_render = True
		return self.read_state()

	# -- cameras -------------------------------------------------------------------------------------------------
	def render(self, camera: str, depth: bool, flush: int | None = None) -> dict:
		self._need_scene()
		from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update

		if camera not in CAM_PRIMS:
			raise BridgeError(f"unknown camera {camera!r}")
		cam = self.scene[CAM_PRIMS[camera]]
		# The RTX annotators return the frame of the PREVIOUS stage update: after the physics state changed, push the
		# current poses through the renderer (render-only app updates, no physics stepping) before the kept capture.
		# An explicit ``flush`` always pumps that many updates.
		n_flush = (self.flush_updates if self._stale_render else 0) if flush is None else int(flush)
		self._stale_render = False
		for _ in range(n_flush):
			self.sim.render()
			ensure_isaac_rtx_render_update(force=True)
		self.sim.render()
		cam.update(self.dt, force_recompute=True)
		out = cam.data.output
		rgb = out["rgb"].torch[0, ..., :3].cpu().numpy().astype(np.uint8)
		dep = None
		if depth:
			dep = out["distance_to_image_plane"].torch[0].squeeze(-1).cpu().numpy().astype(np.float32)
			dep[~np.isfinite(dep)] = 0.0
		self.render_calls += 1
		if camera == "wrist_cam":
			pose = self.capture.camera_matrix("wrist_cam")
		else:
			pose = self._cam_static[camera]
		c = self.payload["cameras"][camera]
		return {
			"camera": camera,
			"rgb": rgb,
			"depth": dep,
			"frame": int(self.frame),
			"sim_time": self.step_count / FPS,
			"cam_pose": pose,
			"intrinsics": {k: c[k] for k in ("fx", "fy", "cx", "cy", "width", "height")},
		}

	# -- replay --------------------------------------------------------------------------------------------------
	def save_replay(self, directory: str) -> dict:
		self._need_scene()
		from digital_twin_armigo.catalog.store import dumps

		if not self.capture.frames:
			raise BridgeError("no recorded frames")
		out = Path(directory) / "replay"
		out.mkdir(parents=True, exist_ok=True)
		cap = self.capture
		(out / "scene.json").write_text(
			dumps({"schema": "catalog-scene/v1", "units": "m", "up_axis": "Z", "quaternion": "xyzw", **cap.scene}),
			encoding="utf-8",
		)
		extra = {k: np.asarray(v) for k, v in self.rows.items()}
		np.savez_compressed(
			out / "motion.npz",
			poses=np.stack(cap.frames),
			timestamps=np.asarray(cap.times),
			node_ids=np.array([n["id"] for n in cap.scene["nodes"]]),
			**({"camera_poses": np.stack(cap.camera_frames)} if cap.camera_frames else {}),
			**extra,
		)
		return {"frames": len(cap.frames), "bytes": (out / "motion.npz").stat().st_size}


def _plain(v):
	if isinstance(v, dict):
		return {str(k): _plain(x) for k, x in v.items()}
	if isinstance(v, (list, tuple)):
		return [_plain(x) for x in v]
	if isinstance(v, np.ndarray):
		return v.tolist()
	if isinstance(v, np.generic):
		return v.item()
	return v


def _pose_matrix(cam: dict) -> np.ndarray:
	m = np.eye(4)
	m[:3, :3] = np.asarray(cam["rot"], dtype=np.float64)
	m[:3, 3] = np.asarray(cam["pos"], dtype=np.float64)
	return m


def _make_capture_class():
	from digital_twin_armigo.catalog.capture import IsaacCapture

	class BridgeCapture(IsaacCapture):
		"""IsaacCapture plus the front camera and a helper that reads a camera's live world pose."""

		def add_front_camera(self, stage) -> None:
			from pxr import UsdGeom

			cams = {p.GetName(): p for p in stage.Traverse() if p.IsA(UsdGeom.Camera)}
			top, front, wrist = cams["TopCamera"], cams["FrontCamera"], cams["wrist_cam"]
			cache = UsdGeom.XformCache()
			for prim in (front,):
				ancestor = prim
				while ancestor and str(ancestor.GetPath()) not in self.bindings:
					ancestor = ancestor.GetParent()
				key = str(ancestor.GetPath()) if ancestor else None
				world = cache.GetLocalToWorldTransform(prim)
				self.offsets[str(prim.GetPath())] = (
					key,
					world * cache.GetLocalToWorldTransform(ancestor).GetInverse() if key else world,
				)
			self.cameras = [top, front, wrist]
			self.camera_names = ["top", "front", "wrist_cam"]
			self.scene["cameras"] = []
			for name, p in zip(self.camera_names, self.cameras, strict=True):
				cam = UsdGeom.Camera(p)
				f = float(cam.GetFocalLengthAttr().Get())
				ap = float(cam.GetVerticalApertureAttr().Get())
				self.scene["cameras"].append({"name": name, "fovy": float(np.degrees(2 * np.arctan(ap / (2 * f))))})

		def camera_matrix(self, name: str) -> np.ndarray:
			"""World-from-camera 4x4 (columns right, up, back) of a camera from the live body buffers."""
			from pxr import Gf

			prim = self.cameras[self.camera_names.index(name)]
			key, offset = self.offsets[str(prim.GetPath())]
			if key is None:
				matrix = offset
			else:
				asset, index = self.bindings[key]
				if index is None:
					pos = asset.data.root_pos_w.torch[0].cpu().numpy()
					quat = asset.data.root_quat_w.torch[0].cpu().numpy()
				else:
					pos = asset.data.body_pos_w.torch[0, index].cpu().numpy()
					quat = asset.data.body_quat_w.torch[0, index].cpu().numpy()
				live = Gf.Matrix4d().SetRotate(Gf.Quatd(float(quat[3]), Gf.Vec3d(*map(float, quat[:3]))))
				live.SetTranslateOnly(Gf.Vec3d(*map(float, pos)))
				matrix = offset * live
			m = np.array(matrix).T  # USD is row-vector: transpose to column-vector form
			rot = m[:3, :3]
			rot = rot / np.linalg.norm(rot, axis=0)
			out = np.eye(4)
			out[:3, :3] = rot
			out[:3, 3] = m[:3, 3]
			return out

	return BridgeCapture


def serve(bridge: Bridge, port: int, authkey: bytes) -> None:
	listener = Listener(("127.0.0.1", port), authkey=authkey)
	log("BRIDGE_READY", port)
	conn = listener.accept()
	log("client connected")
	while True:
		try:
			msg = codec.recv(conn)
		except (EOFError, ConnectionError):
			log("client disconnected")
			break
		cmd = msg.get("cmd")
		try:
			if cmd == "shutdown":
				codec.send(conn, {"ok": True, "result": None})
				break
			if cmd == "hello":
				result = {
					"engine": "isaac-physx",
					"isaaclab": getattr(sys.modules.get("isaaclab"), "__version__", None),
					"device": str(bridge.sim_cfg.device),
					"physics_hz": PHYSICS_HZ,
					"fps": FPS,
					"joints": list(JOINTS),
					"camera_convention": "mujoco/opengl: x right, y up, z back; cam_pose world-from-camera",
				}
			elif cmd == "reset":
				result = bridge.reset(msg["scene"])
			elif cmd == "step":
				result = bridge.step(msg["ctrl"], msg["branch"], msg["frame"])
			elif cmd == "render":
				result = bridge.render(msg["camera"], msg["depth"], msg.get("flush"))
			elif cmd == "state":
				bridge._need_scene()
				result = bridge.read_state()
			elif cmd == "checkpoint":
				result = bridge.checkpoint()
			elif cmd == "restore":
				result = bridge.restore(msg["token"])
			elif cmd == "save_replay":
				result = bridge.save_replay(msg["directory"])
			elif cmd == "stats":
				result = {
					"render_calls": bridge.render_calls,
					"step_count": bridge.step_count,
					"timing_ms_mean": {k: float(np.mean(v)) for k, v in bridge.timing.items()},
				}
			else:
				raise BridgeError(f"unknown command {cmd!r}")
			codec.send(conn, {"ok": True, "result": result})
		except Exception as exc:  # noqa: BLE001 - reported to the client
			log("ERROR", cmd, repr(exc))
			traceback.print_exc()
			codec.send(conn, {"ok": False, "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()})
	conn.close()
	listener.close()


def main() -> None:
	if args_cli.log_dir is not None:
		args_cli.log_dir.mkdir(parents=True, exist_ok=True)
	sim_cfg = sim_utils.SimulationCfg(dt=1.0 / PHYSICS_HZ, render_interval=SUBSTEPS, device="cuda:0")
	global BridgeCapture
	with launch_simulation(sim_cfg, args_cli):
		BridgeCapture = _make_capture_class()
		bridge = Bridge(sim_cfg)
		serve(bridge, args_cli.port, bytes.fromhex(args_cli.authkey_hex))
	os._exit(0)


if __name__ == "__main__":
	main()
