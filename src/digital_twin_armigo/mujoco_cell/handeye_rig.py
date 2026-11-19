"""A simulated calibration session: the gripper holds the marker card, the arm visits random poses, and
:func:`~.handeye.calibrate` turns what the top camera saw into a camera pose.

The rig is its own :class:`~.cell.Cell` built from the episode's layout, so it shares the episode's top camera and is
done before the episode starts, as a real cell would calibrate before work. Joint readings can be given random
noise and a fixed bias per joint; ground truth is used only for scoring.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..constants import HANDOFF_Q, SO101_MJCF_DIR, TOP_FULL
from . import handeye as he
from ._mj import mujoco
from .cell import Cell
from .layout import Layout

# Arm poses: offsets from the hand-off pose (pan, lift, elbow, wrist flex, wrist roll), radians, before ``spread``.
POSE_SPREAD = np.array([0.6, 0.35, 0.35, 0.5, 0.8])
NOMINAL_CARD = he.transform(np.eye(3), np.array([0.05, 0.0, 0.004]))  # beyond the jaw tips, face along gripper +z
MIN_CARD_MARKERS = 3


def rot_xyz(rx: float, ry: float, rz: float) -> np.ndarray:
	cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
	x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
	y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
	z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
	return z @ y @ x


@dataclass(frozen=True)
class CalibrationCfg:
	poses: int = 40
	spread: float = 2.0
	pixel_noise: float = 2.0  # Gaussian RGB noise, grey levels
	joint_noise_deg: float = 0.0  # random, per pose and joint
	joint_bias_deg: float = 0.0  # fixed per joint (σ of the draw)
	offsets: bool = True  # estimate the identifiable joint zero offsets


class Rig:
	"""A cell with the calibration card; renders the top camera with the card following the gripper."""

	def __init__(self, layout: Layout, seed: int, card_jitter: bool = True):
		self.layout = layout
		he.write_card_texture(SO101_MJCF_DIR)
		self.cell = Cell(layout, SO101_MJCF_DIR, extra_xml=he.card_xml())
		m = self.cell.model
		body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "calib_card")
		self.mocap = int(m.body_mocapid[body])
		self.scratch = mujoco.MjData(m)  # forward kinematics of measured joints, apart from the rendered state
		jitter = np.random.default_rng(seed + 10_000)
		self.grip_from_card = NOMINAL_CARD.copy()
		if card_jitter:  # clamped by hand: a few degrees and millimetres off
			self.grip_from_card = NOMINAL_CARD @ he.transform(
				rot_xyz(*np.radians(jitter.uniform(-5, 5, 3))), jitter.uniform(-0.005, 0.005, 3)
			)
		self.k = he.intrinsics(layout.top, TOP_FULL)
		mujoco.mj_forward(m, self.cell.data)
		self.truth = he.camera_truth(m, self.cell.data, "top")
		pan = m.joint("shoulder_pan").id
		self.pan_anchor = self.cell.data.xanchor[pan].copy()
		self.pan_axis = self.cell.data.xaxis[pan].copy()

	def close(self) -> None:
		self.cell.close()

	def robot_from_world(self, pan_bias: float) -> np.ndarray:
		"""The frame the robot believes in when its pan reading is ``pan_bias`` too high: the world turned about the
		pan axis. A camera found in this frame is what the robot needs, since it also moves in it."""
		import cv2

		rot, _ = cv2.Rodrigues(self.pan_axis * pan_bias)
		return he.transform(rot, self.pan_anchor - rot @ self.pan_anchor)

	def grip(self, q: np.ndarray, data=None) -> np.ndarray:
		"""World-from-gripper-frame for arm joints ``q`` (5 values)."""
		d = self.scratch if data is None else data
		d.qpos[self.cell.qadr[:5]] = q
		mujoco.mj_kinematics(self.cell.model, d)
		return he.transform(d.site_xmat[self.cell.site].reshape(3, 3).copy(), d.site_xpos[self.cell.site].copy())

	def render(self, q: np.ndarray, card: bool = True) -> np.ndarray:
		full = np.array([*q, 0.6])
		self.cell.set_joints(full)
		d = self.cell.data
		if card:
			tf = self.grip(q, d) @ self.grip_from_card
		else:
			tf = he.transform(np.eye(3), np.array([0.0, 0.0, -1.0]))  # under the table
		d.mocap_pos[self.mocap] = tf[:3, 3]
		quat = np.zeros(4)
		mujoco.mju_mat2Quat(quat, tf[:3, :3].flatten())
		d.mocap_quat[self.mocap] = quat
		mujoco.mj_forward(self.cell.model, d)
		return self.cell.rgb("top")

	def collect(self, n: int, rng: np.random.Generator, pixel_noise: float, spread: float = 1.0, max_tries: int = 800):
		"""Random reachable arm poses where the top camera sees at least MIN_CARD_MARKERS card markers."""
		poses = []
		lo, hi = self.cell.jnt_range[:5, 0], self.cell.jnt_range[:5, 1]
		tries = 0
		while len(poses) < n and tries < max_tries:
			tries += 1
			q = np.clip(np.array(HANDOFF_Q) + rng.uniform(-1, 1, 5) * POSE_SPREAD * spread, lo, hi)
			card = self.grip(q) @ self.grip_from_card
			if self.grip(q)[2, 3] < 0.08 or card[2, 3] < 0.06:
				continue
			rgb = self.render(q)
			if pixel_noise > 0:
				rgb = np.clip(rgb + rng.normal(0, pixel_noise, rgb.shape), 0, 255).astype(np.uint8)
			obj, img, seen = he.card_correspondences(he.detect_markers(rgb))
			if seen >= MIN_CARD_MARKERS:
				poses.append((q, obj, img, rgb))
		return poses, tries

	def observations(self, poses, rng: np.random.Generator, joint_noise_deg: float, joint_bias_deg: float):
		"""Observations with the joint readings a real arm would report, and the bias drawn (radians)."""
		bias = np.radians(rng.normal(0, joint_bias_deg, 5)) if joint_bias_deg else np.zeros(5)
		obs = []
		for q, obj, img, _ in poses:
			q_meas = q + bias + np.radians(rng.normal(0, joint_noise_deg, 5))
			obs.append(he.Observation(self.grip(q_meas), obj, img, q=q_meas))
		return obs, bias


def calibrate_layout(layout: Layout, seed: int, cfg: CalibrationCfg) -> tuple[np.ndarray, np.ndarray, dict]:
	"""Calibrate the top camera of ``layout``. Returns the camera matrix, world-from-camera (OpenCV convention) and a
	record with the truth errors for scoring.

	The solver finds the camera in the frame the robot believes in (turned by the pan bias). A real arm keeps that
	bias while it works, so camera and arm stay consistent; the simulated episode arm has no bias, so the estimate is
	turned back into the world frame here."""
	rng = np.random.default_rng(seed + 20_000)
	rig = Rig(layout, seed)
	try:
		poses, tries = rig.collect(cfg.poses, rng, cfg.pixel_noise, cfg.spread)
		if len(poses) < 3:
			raise RuntimeError(f"calibration saw the card in only {len(poses)} of {tries} poses")
		obs, bias = rig.observations(poses, rng, cfg.joint_noise_deg, cfg.joint_bias_deg)
		if cfg.offsets:
			cal = he.calibrate(rig.k, obs, fk=rig.grip, offset_joints=he.OFFSET_JOINTS)
		else:
			cal = he.calibrate(rig.k, obs)
		est = he.inverse(rig.robot_from_world(bias[0])) @ cal.world_from_cam
		truth = rig.truth
		info = {
			"poses": len(poses),
			"tries": tries,
			"rms_px": round(cal.rms_px, 4),
			"cam_pos_err_mm": round(float(np.linalg.norm(est[:3, 3] - truth[:3, 3]) * 1000), 3),
			"cam_rot_err_deg": round(he.rotation_error_deg(est, truth), 4),
			"bias_deg": np.degrees(bias).round(4).tolist(),
		}
		return rig.k, est, info
	finally:
		rig.close()
