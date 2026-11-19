"""One episode's MuJoCo model: stepping with the assisted grasp, and renderers for the dataset cameras."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Self

import numpy as np

from ..camera import CameraPose
from ..constants import JOINTS, TOP_CROP, TOP_FULL
from ..depth import layered_height_image
from ..physics import PhysicsSpec, brick_spec_for_size
from ..presentation import FACES, face_frame
from ..sensor_noise import D415NoiseCfg, degrade_depth_height, degrade_rgb
from ._mj import mujoco
from .layout import Layout
from .scene import PROBE_GROUP, SUBSTEPS, face_probe_name, scene_xml, write_marker_textures

GRASP_RADIUS_M = 0.03
RELEASE_MAX_SPEED = 0.6  # m/s, brick speed cap for RELEASE_SETTLE_FRAMES frames after the jaws open
RELEASE_SETTLE_FRAMES = 15
RENDER_SIZE = (640, 480)  # width, height of every rendered camera


MIN_SPECK_PX = 10  # brick-mask components smaller than this are segmentation decoding errors
MIN_SPECK_FRACTION = 0.1  # ... and so are components smaller than this fraction of the largest one


def _drop_specks(geom: np.ndarray, is_brick: np.ndarray) -> np.ndarray:
	"""``geom`` with brick pixels outside the brick's real blob set to -1.

	Pixels far from the brick (on speckled bowl rims, thin streaks) decode to brick geom ids. A real view of the brick,
	even a thin grazing face, is one connected blob, or a few large pieces where the jaws cut across it; so a component
	is kept only when it has at least :data:`MIN_SPECK_PX` pixels and :data:`MIN_SPECK_FRACTION` of the largest one."""
	import cv2

	mask = is_brick[geom]
	if not mask.any():
		return geom
	_, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
	area = stats[:, cv2.CC_STAT_AREA].copy()
	area[0] = 0  # label 0 is the background
	keep_min = max(MIN_SPECK_PX, MIN_SPECK_FRACTION * area.max())
	small = np.flatnonzero((area > 0) & (area < keep_min))
	if not len(small):
		return geom
	out = geom.copy()
	out[np.isin(labels, small)] = -1
	return out


class Cell:
	"""One episode's MuJoCo model with renderers for the dataset cameras.

	Ids and addresses that the per-frame and per-IK-iteration code needs are resolved once here: looking them up by
	name (``model.joint(name)``, ``mj_name2id``) inside those loops costs more than the kinematics they wrap.
	"""

	def __init__(
		self,
		layout: Layout,
		menagerie: Path,
		physics: PhysicsSpec | None = None,
		dynamic_clutter: bool = False,
		assisted_grasp: bool = True,
		mass_scale: float = 1.0,
		friction_scale: float = 1.0,
		render_enabled: bool = True,
		sensor_noise: D415NoiseCfg | None = None,
		face_probes: bool = False,
		extra_xml: tuple[str, str] = ("", ""),
	) -> None:
		write_marker_textures(menagerie, {m[0] for m in layout.markers})
		# The scene includes so101.xml by relative path, so it is compiled from a file next to it.
		path = menagerie / f"_sorting_cell_{os.getpid()}.xml"
		path.write_text(
			scene_xml(
				layout,
				physics=physics,
				dynamic_clutter=dynamic_clutter,
				mass_scale=mass_scale,
				friction_scale=friction_scale,
				face_probes=face_probes,
				extra_xml=extra_xml,
			),
			encoding="utf-8",
		)
		try:
			self.model = mujoco.MjModel.from_xml_path(str(path))
		finally:
			path.unlink(missing_ok=True)
		self.layout = layout
		self.physics = physics
		self.assisted_grasp = assisted_grasp
		self.sensor_noise = sensor_noise
		self.data = mujoco.MjData(self.model)
		self.renderer = mujoco.Renderer(self.model, RENDER_SIZE[1], RENDER_SIZE[0]) if render_enabled else None
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
		# Every geom named ``defect_<k>`` (any number of them: the cross has two strokes, other defect kinds more).
		self.defect_geoms = np.array(
			[
				g
				for g in range(m.ngeom)
				if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("defect_")
			],
			dtype=int,
		)
		self.brick_geoms = np.array([g for g in range(m.ngeom) if m.geom_bodyid[g] == self.brick_body])
		self.fixed_jaw_geoms = {
			g for g in range(m.ngeom) if m.geom_bodyid[g] == self.grip_body and m.geom_group[g] in (3, 4)
		}
		self.moving_jaw_geoms = {
			g for g in range(m.ngeom) if m.geom_bodyid[g] == self.jaw_body and m.geom_group[g] in (3, 4)
		}
		self.non_jaw_robot_geoms = {
			int(g)
			for g in self.link_geoms
			if int(g) not in self.fixed_jaw_geoms and int(g) not in self.moving_jaw_geoms
		}
		self.robot_all_geoms = {g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]] == robot_root}

		if physics is not None:
			self._apply_physics_spec(physics, mass_scale=mass_scale, friction_scale=friction_scale)

		# Segmentation lookups indexed by geom id; the extra last entry stays False and takes index -1 (not a geom).
		self._is_defect = np.zeros(m.ngeom + 1, bool)
		self._is_defect[self.defect_geoms] = True
		self._is_brick = np.zeros(m.ngeom + 1, bool)
		self._is_brick[self.brick_geoms] = True
		# Face index of each face-probe geom (-1 elsewhere); all -1 unless the cell was built with ``face_probes``.
		self._face_of = np.full(m.ngeom + 1, -1, dtype=np.int64)
		for face in range(len(FACES)):
			gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, face_probe_name(face))
			if gid >= 0:
				self._face_of[gid] = face
		self.has_face_probes = bool((self._face_of >= 0).any())
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
		self._cam_poses: dict[str, CameraPose] = {}
		if layout.top is not None:
			self._cam_poses["top"] = layout.top
		if layout.front is not None:
			self._cam_poses["front"] = layout.front

	def _apply_physics_spec(self, physics: PhysicsSpec, mass_scale: float, friction_scale: float) -> None:
		"""Align MuJoCo actuator gains, joint dynamics, and surface friction with :class:`PhysicsSpec`."""
		m = self.model
		for idx, jname in enumerate(JOINTS):
			act_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, jname)
			spec = physics.gripper_actuator if jname == "gripper" else physics.arm_actuator
			m.actuator_gainprm[act_id, 0] = spec.stiffness
			m.actuator_biasprm[act_id, 1] = -spec.stiffness
			m.actuator_biasprm[act_id, 2] = -spec.damping
			m.actuator_forcerange[act_id] = (-spec.effort_limit, spec.effort_limit)
			dof = self.dofadr[idx]
			m.dof_damping[dof] = spec.joint_damping
			m.dof_frictionloss[dof] = spec.frictionloss
			m.dof_armature[dof] = spec.armature

		# Surface frictions (MuJoCo combines contacts via max by default when priorities match)
		for g in self.fixed_jaw_geoms | self.moving_jaw_geoms:
			m.geom_priority[g] = 0
			m.geom_friction[g] = (physics.gripper_pad.static_friction * friction_scale, 0.005, 0.0005)
		table_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table")
		if table_gid >= 0:
			m.geom_friction[table_gid] = (physics.table.dynamic_friction * friction_scale, 0.005, 0.0005)
		for bname in ("green_bowl", "pink_bowl"):
			bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, bname)
			if bid >= 0:
				for g in range(m.ngeom):
					if m.geom_bodyid[g] == bid and m.geom_contype[g] != 0:
						m.geom_friction[g] = (physics.plate.dynamic_friction * friction_scale, 0.005, 0.0005)
		bspec = brick_spec_for_size(self.layout.brick_size, physics).scaled(mass_scale, friction_scale)
		# Note: body_mass/inertia already set in MJCF _brick when physics is passed, ensure inertia matches scaled box
		sx, sy, sz = self.layout.brick_size
		m.body_mass[self.brick_body] = bspec.mass_kg
		m.body_inertia[self.brick_body] = (
			bspec.mass_kg * (sy**2 + sz**2) / 12.0,
			bspec.mass_kg * (sx**2 + sz**2) / 12.0,
			bspec.mass_kg * (sx**2 + sy**2) / 12.0,
		)

	def close(self) -> None:
		"""Release the renderer's GL context."""
		if self.renderer is not None:
			self.renderer.close()

	def __enter__(self) -> Self:
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
		self._cam_poses[name] = pose
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
		"""Assisted grasp when ``self.assisted_grasp`` is True; pure contact friction otherwise."""
		if not self.assisted_grasp:
			return
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

	def tcp_pos(self) -> np.ndarray:
		return self.data.site_xpos[self.site].copy()

	def brick_pos(self) -> np.ndarray:
		return self.data.xpos[self.brick_body].copy()

	def brick_quat_xyzw(self) -> np.ndarray:
		"""Brick orientation quaternion in (x, y, z, w) convention."""
		wxyz = self.data.xquat[self.brick_body]
		return np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]], dtype=np.float64)

	def brick_vel(self) -> tuple[np.ndarray, np.ndarray]:
		"""World linear (m/s) and angular (rad/s) velocity of the brick."""
		v = self.data.qvel[self.brick_dofadr : self.brick_dofadr + 6]
		return v[:3].copy(), v[3:].copy()

	def set_brick_pose(self, pos: np.ndarray, quat_xyzw: np.ndarray | None = None) -> None:
		"""Move the free-joint brick to ``pos`` and optional ``quat_xyzw``, zeroing its velocity."""
		adr = self.brick_qadr
		self.data.qpos[adr : adr + 3] = pos
		if quat_xyzw is not None:
			x, y, z, w = quat_xyzw
			self.data.qpos[adr + 3 : adr + 7] = (w, x, y, z)
		self.data.qvel[self.brick_dofadr : self.brick_dofadr + 6] = 0.0
		mujoco.mj_forward(self.model, self.data)

	def jaw_force_components(self) -> tuple[float, float, float, float, float, float]:
		"""Return ``(fix_norm, fix_tan, fix_total, mov_norm, mov_tan, mov_total)`` contact forces in N."""
		f_fix_w = np.zeros(3)
		f_mov_w = np.zeros(3)
		fix_n = 0.0
		mov_n = 0.0
		fix_t_vec = np.zeros(2)
		mov_t_vec = np.zeros(2)
		cforce = np.zeros(6)
		brick_set = {int(g) for g in self.brick_geoms}
		for cid in range(self.data.ncon):
			con = self.data.contact[cid]
			g1, g2 = int(con.geom1), int(con.geom2)
			other = g2 if g1 in brick_set else (g1 if g2 in brick_set else -1)
			if other < 0:
				continue
			if other in self.fixed_jaw_geoms or other in self.moving_jaw_geoms:
				mujoco.mj_contactForce(self.model, self.data, cid, cforce)
				world_f = con.frame.reshape(3, 3).T @ cforce[:3]
				if other in self.fixed_jaw_geoms:
					f_fix_w += world_f
					fix_n += abs(float(cforce[0]))
					fix_t_vec += cforce[1:3]
				else:
					f_mov_w += world_f
					mov_n += abs(float(cforce[0]))
					mov_t_vec += cforce[1:3]
		return (
			fix_n,
			float(np.linalg.norm(fix_t_vec)),
			float(np.linalg.norm(f_fix_w)),
			mov_n,
			float(np.linalg.norm(mov_t_vec)),
			float(np.linalg.norm(f_mov_w)),
		)

	def jaw_forces(self) -> tuple[float, float]:
		"""Total contact force magnitude (N) between each jaw and the target brick."""
		_, _, f_fix, _, _, f_mov = self.jaw_force_components()
		return f_fix, f_mov

	def actuator_torques(self) -> np.ndarray:
		"""Solver-applied generalized actuator forces/torques (N or N*m) ordered by :data:`JOINTS`."""
		return np.array([float(self.data.actuator_force[self.model.actuator(j).id]) for j in JOINTS], dtype=np.float64)

	def step_with_telemetry(self, ctrl: np.ndarray, substeps: int = SUBSTEPS) -> dict[str, np.ndarray]:
		"""Advance ``substeps`` MuJoCo steps while sampling substep contact forces and actuator torques."""
		self.data.ctrl[:6] = ctrl
		jaw_total = np.zeros((substeps, 2), dtype=np.float64)
		jaw_normal = np.zeros((substeps, 2), dtype=np.float64)
		jaw_tan = np.zeros((substeps, 2), dtype=np.float64)
		unint = np.zeros(substeps, dtype=np.float64)
		torques = np.zeros((substeps, 6), dtype=np.float64)
		for s in range(substeps):
			mujoco.mj_step(self.model, self.data)
			fn0, ft0, tot0, fn1, ft1, tot1 = self.jaw_force_components()
			jaw_normal[s] = (fn0, fn1)
			jaw_tan[s] = (ft0, ft1)
			jaw_total[s] = (tot0, tot1)
			unint[s] = self.unintended_force()
			torques[s] = self.actuator_torques()
			observer = getattr(self, "substep_observer", None)
			if observer is not None:
				observer()
		return {
			"jaw_total": jaw_total,
			"jaw_normal": jaw_normal,
			"jaw_tangential": jaw_tan,
			"unintended": unint,
			"torques": torques,
		}

	def unintended_force(self) -> float:
		"""Peak external contact force magnitude (N) on non-jaw robot links (ignores internal self-collisions)."""
		peak = 0.0
		cforce = np.zeros(6)
		for cid in range(self.data.ncon):
			con = self.data.contact[cid]
			g1, g2 = int(con.geom1), int(con.geom2)
			if (g1 in self.non_jaw_robot_geoms and g2 not in self.robot_all_geoms) or (
				g2 in self.non_jaw_robot_geoms and g1 not in self.robot_all_geoms
			):
				mujoco.mj_contactForce(self.model, self.data, cid, cforce)
				peak = max(peak, float(np.linalg.norm(cforce[:3])))
		return peak

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

	def face_visibility(
		self, camera: str, crop: tuple[int, int, int, int] | None = None, data=None
	) -> dict[str, np.ndarray | int]:
		"""Visible pixels per brick face (``faces``, shape (6,), :data:`~digital_twin_armigo.presentation.FACES` order)
		and of the defect cross (``defect``) and the brick (``brick``) in ``camera``, occlusion by the arm included,
		plus the brick's pixels with the arm removed (``brick_full``) and its pixel bounding box ``bbox`` (x, y, w, h in
		the crop, or None). Needs a cell built with ``face_probes``. ``data`` renders another ``MjData`` of this model
		(for example a kinematic copy with the arm moved) instead of the simulation state."""
		if not self.has_face_probes:
			raise RuntimeError("face_visibility needs a Cell built with face_probes=True")
		d = self.data if data is None else data
		segs = []
		self.renderer.enable_segmentation_rendering()
		try:
			for opt in (self._all_groups, self._brick_only):
				self.renderer.update_scene(d, camera, scene_option=opt)
				seg = self.renderer.render()
				if crop is not None:
					x, y, w, h = crop
					seg = seg[y : y + h, x : x + w]
				segs.append(np.where(seg[..., 1] == self._obj_geom, seg[..., 0], -1))
		finally:
			self.renderer.disable_segmentation_rendering()
		geom, geom_full = segs
		geom = _drop_specks(geom, self._is_brick)
		geom_full = _drop_specks(geom_full, self._is_brick)
		face = self._face_of[geom]
		faces = np.bincount(face[face >= 0], minlength=len(FACES))
		brick = self._is_brick[geom]
		ys, xs = np.nonzero(brick)
		bbox = None
		if len(xs):
			# 1st-99th percentile extent: stray pixels (the table's reflection of the brick) must not stretch it
			x0, x1 = np.percentile(xs, [1, 99])
			y0, y1 = np.percentile(ys, [1, 99])
			bbox = (int(x0), int(y0), int(x1 - x0) + 1, int(y1 - y0) + 1)
		return {
			"faces": faces,
			"defect": int(np.count_nonzero(self._is_defect[geom])),
			"brick": int(np.count_nonzero(brick)),
			"brick_full": int(np.count_nonzero(self._is_brick[geom_full])),
			"bbox": bbox,
		}

	def visibility_labels(self, camera: str, data=None) -> tuple[np.ndarray, np.ndarray]:
		"""Per-pixel ``(face, defect)`` of ``camera``'s full frame, exactly as :meth:`face_visibility` counts them:
		the face id (-1 off the face probes) and the defect-cross mask, arm occlusion included. Summing them over
		the frame gives ``face_visibility``'s ``faces`` and ``defect``; resampling them like an image crop gives what
		that crop shows."""
		if not self.has_face_probes:
			raise RuntimeError("visibility_labels needs a Cell built with face_probes=True")
		d = self.data if data is None else data
		self.renderer.enable_segmentation_rendering()
		try:
			self.renderer.update_scene(d, camera, scene_option=self._all_groups)
			seg = self.renderer.render()
		finally:
			self.renderer.disable_segmentation_rendering()
		geom = _drop_specks(np.where(seg[..., 1] == self._obj_geom, seg[..., 0], -1), self._is_brick)
		return self._face_of[geom], self._is_defect[geom]

	def robot_min_z(self, data=None) -> float:
		"""Lowest point of the moving links' collision geometry (oriented bounding boxes), metres above the table;
		of ``data`` (another ``MjData`` of this model) when given."""
		d = self.data if data is None else data
		g = self.link_geoms
		rot = d.geom_xmat[g].reshape(-1, 3, 3)
		aabb = self.model.geom_aabb[g]
		centre_z = d.geom_xpos[g, 2] + np.einsum("nj,nj->n", rot[:, 2, :], aabb[:, :3])
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

	def top_images(self, rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
		"""(top_rgb_v2, top_depth_v2) cropped like the depth bridge."""
		x, y, w, h = TOP_CROP
		rgb = self.rgb("top")[y : y + h, x : x + w]
		z, valid = self.height("top", TOP_CROP)
		if self.sensor_noise is not None and rng is not None:
			top_pose = self._cam_poses.get("top")
			if top_pose is None:
				raise ValueError("top CameraPose must be set before applying camera-frame D415 depth noise")
			rgb = degrade_rgb(rgb, rng, self.sensor_noise)
			z, valid = degrade_depth_height(
				z, valid, top_pose, rng, self.sensor_noise, crop=TOP_CROP, full_shape=(TOP_FULL[1], TOP_FULL[0])
			)
		return rgb, layered_height_image(z, valid)
