"""One episode's MuJoCo model: stepping with the assisted grasp, and renderers for the dataset cameras."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ..camera import CameraPose
from ..constants import JOINTS, TOP_CROP
from ..depth import layered_height_image
from ..presentation import face_frame
from ._mj import mujoco
from .layout import Layout
from .scene import PROBE_GROUP, SUBSTEPS, scene_xml, write_marker_textures

GRASP_RADIUS_M = 0.03
RELEASE_MAX_SPEED = 0.6  # m/s, brick speed cap for RELEASE_SETTLE_FRAMES frames after the jaws open
RELEASE_SETTLE_FRAMES = 15
RENDER_SIZE = (640, 480)  # width, height of every rendered camera


class Cell:
	"""One episode's MuJoCo model with renderers for the dataset cameras.

	Ids and addresses that the per-frame and per-IK-iteration code needs are resolved once here: looking them up by
	name (``model.joint(name)``, ``mj_name2id``) inside those loops costs more than the kinematics they wrap.
	"""

	def __init__(self, layout: Layout, menagerie: Path) -> None:
		write_marker_textures(menagerie, {m[0] for m in layout.markers})
		# The scene includes so101.xml by relative path, so it is compiled from a file next to it.
		path = menagerie / f"_sorting_cell_{os.getpid()}.xml"
		path.write_text(scene_xml(layout), encoding="utf-8")
		try:
			self.model = mujoco.MjModel.from_xml_path(str(path))
		finally:
			path.unlink(missing_ok=True)
		self.layout = layout
		self.data = mujoco.MjData(self.model)
		self.renderer = mujoco.Renderer(self.model, RENDER_SIZE[1], RENDER_SIZE[0])
		self.held: tuple[np.ndarray, np.ndarray] | None = None
		self.settle_frames = 0
		m = self.model

		def body(name: str) -> int:
			return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)

		self.site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
		self.brick_body = body("brick")
		self.grip_body = body("gripper")
		self.jaw_body = body("moving_jaw_so101_v1")
		# arm joints in JOINTS order: qpos addresses, dof (Jacobian column) addresses and position limits
		self.qadr = [int(m.joint(j).qposadr[0]) for j in JOINTS]
		self.dofadr = [int(m.joint(j).dofadr[0]) for j in JOINTS]
		self.jnt_range = m.jnt_range[[m.joint(j).id for j in JOINTS]].copy()
		self.brick_qadr = int(m.joint("brick").qposadr[0])
		self.brick_dofadr = int(m.joint("brick").dofadr[0])
		self._cam_ids = {m.camera(c).name: c for c in range(m.ncam)}

		robot_root = m.body_rootid[self.grip_body]
		self.robot_geoms = np.array(
			[g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]] == robot_root and m.geom_group[g] == 3]
		)
		# links that move with the shoulder lift (the base and shoulder sit on the table by design)
		lift_body = m.jnt_bodyid[m.joint("shoulder_lift").id]

		def moves(b: int) -> bool:
			while b > 0:
				if b == lift_body:
					return True
				b = m.body_parentid[b]
			return False

		self.link_geoms = np.array([g for g in self.robot_geoms if moves(m.geom_bodyid[g])])
		self.defect_geoms = np.array(
			[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"defect_{k}") for k in range(2)], dtype=int
		)
		self.brick_geoms = np.array([g for g in range(m.ngeom) if m.geom_bodyid[g] == self.brick_body])
		# Segmentation lookups indexed by geom id; the extra last entry stays False and takes index -1 (not a geom).
		self._is_defect = np.zeros(m.ngeom + 1, bool)
		self._is_defect[self.defect_geoms] = True
		self._is_brick = np.zeros(m.ngeom + 1, bool)
		self._is_brick[self.brick_geoms] = True
		# A plain int: comparing an array with the pybind11 enum itself runs a Python call per pixel (about 65 ms
		# for one cropped image against 0.04 ms).
		self._obj_geom = int(mujoco.mjtObj.mjOBJ_GEOM)
		self._all_groups = mujoco.MjvOption()
		self._all_groups.geomgroup[:] = 1
		self._all_groups.geomgroup[3:5] = 0
		self._brick_only = mujoco.MjvOption()
		self._brick_only.geomgroup[:] = 0
		self._brick_only.geomgroup[0] = 1
		self._brick_only.geomgroup[PROBE_GROUP] = 1
		self._ray_z: tuple[tuple, np.ndarray] | None = None

	def close(self) -> None:
		"""Release the renderer's GL context."""
		self.renderer.close()

	def __enter__(self) -> Cell:
		return self

	def __exit__(self, *exc: object) -> None:
		self.close()

	def camera_id(self, name: str) -> int:
		return self._cam_ids[name]

	def set_joints(self, q: np.ndarray) -> None:
		self.data.qpos[self.qadr] = q
		self.data.ctrl[:6] = q
		mujoco.mj_forward(self.model, self.data)

	def set_camera(self, name: str, pose: CameraPose) -> None:
		"""Move an existing camera in place (no recompilation)."""
		cid = self.camera_id(name)
		right, up = pose.axes()
		mat = np.column_stack([right, up, np.cross(right, up)])
		quat = np.zeros(4)
		mujoco.mju_mat2Quat(quat, mat.flatten())
		self.model.cam_pos[cid] = (pose.x, pose.y, pose.height)
		self.model.cam_quat[cid] = quat
		width = self.model.cam_resolution[cid][0]
		self.model.cam_intrinsic[cid][:2] = pose.focal_px * self.model.cam_sensorsize[cid][0] / width
		mujoco.mj_forward(self.model, self.data)

	def joints(self) -> np.ndarray:
		return self.data.qpos[self.qadr].copy()

	def step(self, ctrl: np.ndarray) -> None:
		self.data.ctrl[:6] = ctrl
		if self.settle_frames <= 0:
			mujoco.mj_step(self.model, self.data, nstep=SUBSTEPS)
			return
		# Just after release the brick may still overlap a jaw or the clutter it was held against; cap its speed for
		# a few frames so penetration is resolved as a drop instead of an ejection.
		vadr = self.brick_dofadr
		for _ in range(SUBSTEPS):
			mujoco.mj_step(self.model, self.data)
			v = self.data.qvel[vadr : vadr + 6]
			v[:3] *= min(1.0, RELEASE_MAX_SPEED / max(float(np.linalg.norm(v[:3])), 1e-9))
			v[3:] *= min(1.0, 10.0 / max(float(np.linalg.norm(v[3:])), 1e-9))
		self.settle_frames -= 1

	def update_grasp(self, gripper_cmd: float) -> None:
		"""Assisted grasp: once the jaws are commanded closed with the brick between them, the brick follows the
		gripper rigidly until the jaws are commanded open again. Contact physics still positions the brick while the
		jaws close; holding it is not left to friction, which makes the scripted demonstrations reliable."""
		d = self.data
		if self.held is None:
			closed = gripper_cmd < 0.0 and d.qpos[self.qadr[5]] < 0.35
			near = np.linalg.norm(self.brick_pos() - d.site_xpos[self.site]) < GRASP_RADIUS_M
			if closed and near:
				rot_g = d.xmat[self.grip_body].reshape(3, 3)
				rot_b = d.xmat[self.brick_body].reshape(3, 3)
				self.held = (rot_g.T @ (self.brick_pos() - d.xpos[self.grip_body]), rot_g.T @ rot_b)
			return
		if gripper_cmd > 0.3:
			self.held = None
			self.settle_frames = RELEASE_SETTLE_FRAMES
			return
		d.qvel[self.brick_dofadr : self.brick_dofadr + 6] = 0.0
		self.place_held()

	def brick_pos(self) -> np.ndarray:
		return self.data.xpos[self.brick_body].copy()

	def place_held(self) -> None:
		"""Put the held brick where the stored grasp puts it relative to the gripper (after joints changed)."""
		if self.held is None:
			return
		m, d = self.model, self.data
		mujoco.mj_kinematics(m, d)
		rel_p, rel_r = self.held
		rot_g = d.xmat[self.grip_body].reshape(3, 3)
		adr = self.brick_qadr
		d.qpos[adr : adr + 3] = d.xpos[self.grip_body] + rot_g @ rel_p
		quat = np.zeros(4)
		mujoco.mju_mat2Quat(quat, (rot_g @ rel_r).flatten())
		d.qpos[adr + 3 : adr + 7] = quat
		mujoco.mj_forward(m, d)

	def visibility(
		self, camera: str, crop: tuple[int, int, int, int] | None = None, full: bool = True
	) -> dict[str, int]:
		"""Pixel counts in ``camera``: defect and brick pixels visible in the scene, and the same with the arm
		removed (``*_full``, skipped unless ``full``), from segmentation renders. Defect pixels include the hidden
		probe cross of a normal brick.
		"""
		out = {}
		self.renderer.enable_segmentation_rendering()
		for key, opt in (("", self._all_groups), ("_full", self._brick_only))[: 2 if full else 1]:
			self.renderer.update_scene(self.data, camera, scene_option=opt)
			seg = self.renderer.render()
			if crop is not None:
				x, y, w, h = crop
				seg = seg[y : y + h, x : x + w]
			geom = np.where(seg[..., 1] == self._obj_geom, seg[..., 0], -1)
			out[f"defect{key}"] = int(np.count_nonzero(self._is_defect[geom]))
			out[f"brick{key}"] = int(np.count_nonzero(self._is_brick[geom]))
		self.renderer.disable_segmentation_rendering()
		return out

	def robot_min_z(self) -> float:
		"""Lowest point of the moving links' collision geometry (oriented bounding boxes), metres above the table."""
		g = self.link_geoms
		rot = self.data.geom_xmat[g].reshape(-1, 3, 3)
		aabb = self.model.geom_aabb[g]
		centre_z = self.data.geom_xpos[g, 2] + np.einsum("nj,nj->n", rot[:, 2, :], aabb[:, :3])
		return float(np.min(centre_z - np.einsum("nj,nj->n", np.abs(rot[:, 2, :]), aabb[:, 3:])))

	def defect_normal(self) -> tuple[np.ndarray, np.ndarray]:
		"""World position and outward normal of the defect face centre."""
		centre, normal, _, _ = face_frame(self.layout.brick_size, self.layout.defect_face)
		rot = self.data.xmat[self.brick_body].reshape(3, 3)
		return self.data.xpos[self.brick_body] + rot @ centre, rot @ normal

	def rgb(self, camera: str, out: np.ndarray | None = None) -> np.ndarray:
		"""Colour image of ``camera`` (height, width, 3), written into ``out`` when given."""
		self.renderer.disable_depth_rendering()
		self.renderer.update_scene(self.data, camera)
		return self.renderer.render(out=out)

	def _ray_world_z(self, cid: int, shape: tuple[int, int]) -> np.ndarray:
		"""World z of each pixel's viewing ray at unit depth along the optical axis.

		It depends only on the camera's orientation and focal length, so it is computed once and reused for as long
		as the camera does not move: an episode renders hundreds of frames through one fixed top camera.
		"""
		mat = self.data.cam_xmat[cid]
		f = self.model.cam_intrinsic[cid][0] / self.model.cam_sensorsize[cid][0] * RENDER_SIZE[0]
		key = (cid, shape, float(f), mat.tobytes())
		if self._ray_z is None or self._ray_z[0] != key:
			hgt, wid = shape
			u, v = np.meshgrid(np.arange(wid) + 0.5, np.arange(hgt) + 0.5)
			ray = np.stack([(u - wid / 2) / f, -(v - hgt / 2) / f, -np.ones_like(u)], axis=-1)
			self._ray_z = (key, np.ascontiguousarray((ray @ mat.reshape(3, 3).T)[..., 2]))
		return self._ray_z[1]

	def height(self, camera: str, crop: tuple[int, int, int, int] | None = None) -> tuple[np.ndarray, np.ndarray]:
		"""Height above the table per pixel of ``camera`` and a validity mask, optionally only inside ``crop``."""
		self.renderer.enable_depth_rendering()
		self.renderer.update_scene(self.data, camera)
		depth = self.renderer.render()
		self.renderer.disable_depth_rendering()
		cid = self.camera_id(camera)
		ray_z = self._ray_world_z(cid, depth.shape)
		if crop is not None:
			x, y, w, h = crop
			depth, ray_z = depth[y : y + h, x : x + w], ray_z[y : y + h, x : x + w]
		world_z = self.data.cam_xpos[cid][2] + ray_z * depth.astype(np.float64)
		valid = depth < 5.0
		return world_z, valid

	def top_images(self) -> tuple[np.ndarray, np.ndarray]:
		"""(top_rgb_v2, top_depth_v2) cropped like the depth bridge."""
		x, y, w, h = TOP_CROP
		rgb = self.rgb("top")[y : y + h, x : x + w]
		z, valid = self.height("top", TOP_CROP)
		return rgb, layered_height_image(z, valid)
