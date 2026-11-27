import math

import cv2
import numpy as np
import pytest

from digital_twin_armigo.agent.perception import BrickEstimate
from digital_twin_armigo.agent.wrist_localize import (
	FuseCfg,
	WristFix,
	believed_wrist_from_cam,
	fuse,
	localize_from_wrist,
	wrist_k,
)

SIZE = (0.032, 0.016, 0.0192)  # 2x4
TOP_Z = 0.0192


def _down_camera(x=0.30, y=0.0, z=0.20, tilt_deg=0.0):
	"""Camera above (x, y, z) looking straight down (OpenCV: z forward = world -z), optionally tilted about x."""
	t = math.radians(tilt_deg)
	tilt = np.array([[1, 0, 0], [0, math.cos(t), -math.sin(t)], [0, math.sin(t), math.cos(t)]])
	tf = np.eye(4)
	tf[:3, :3] = np.diag([1.0, -1.0, -1.0]) @ tilt
	tf[:3, 3] = (x, y, z)
	return tf


def _render(world_from_cam, k, bricks, shape=(240, 320), bg=(120, 120, 118)):
	img = np.empty((*shape, 3), np.uint8)
	img[:] = bg
	rng = np.random.default_rng(0)
	img = np.clip(img.astype(int) + rng.integers(-4, 5, img.shape), 0, 255).astype(np.uint8)
	cam_from_world = np.linalg.inv(world_from_cam)
	for cx, cy, yaw, colour in bricks:
		c, s = math.cos(yaw), math.sin(yaw)
		hx, hy = SIZE[0] / 2, SIZE[1] / 2
		corners = np.array(
			[[cx + c * a - s * b, cy + s * a + c * b, TOP_Z] for a, b in ((hx, hy), (-hx, hy), (-hx, -hy), (hx, -hy))]
		)
		p = corners @ cam_from_world[:3, :3].T + cam_from_world[:3, 3]
		uv = (p[:, :2] / p[:, 2:3]) @ k[:2, :2].T + k[:2, 2]
		cv2.fillPoly(img, [np.round(uv).astype(np.int32)], colour)
	return img


def test_recovers_pose_from_a_top_down_view():
	k = wrist_k(60.0, (240, 320))
	w2c = _down_camera()
	img = _render(w2c, k, [(0.305, 0.004, math.radians(20), (200, 30, 30))])
	fix, why = localize_from_wrist(img, k, w2c, np.array([0.300, 0.0]), math.radians(14), SIZE, TOP_Z)
	assert why is None
	assert np.linalg.norm(fix.pos_xy - [0.305, 0.004]) < 1e-3
	assert abs(math.degrees(fix.yaw) - 20) < 1.5


def test_ignores_a_same_colour_brick_outside_the_gate():
	k = wrist_k(60.0, (240, 320))
	w2c = _down_camera()
	img = _render(w2c, k, [(0.302, 0.0, 0.0, (200, 30, 30)), (0.36, 0.05, 0.3, (200, 30, 30))])
	fix, why = localize_from_wrist(img, k, w2c, np.array([0.300, 0.0]), 0.0, SIZE, TOP_Z)
	assert why is None and np.linalg.norm(fix.pos_xy - [0.302, 0.0]) < 1e-3


def test_rejects_tilted_views_and_resolves_yaw_symmetry():
	k = wrist_k(60.0, (240, 320))
	tilted = _down_camera(tilt_deg=60)
	img = _render(tilted, k, [(0.30, 0.0, 0.0, (200, 30, 30))])
	assert localize_from_wrist(img, k, tilted, np.array([0.30, 0.0]), 0.0, SIZE, TOP_Z) == (None, "TILTED")
	w2c = _down_camera()
	img = _render(w2c, k, [(0.30, 0.0, math.radians(5), (200, 30, 30))])
	fix, _ = localize_from_wrist(img, k, w2c, np.array([0.30, 0.0]), math.radians(5 + 180), SIZE, TOP_Z)
	assert abs(math.degrees(fix.yaw) - (5 + 180)) < 1.5  # nearest symmetric yaw to the prior


def test_fuse_gates_large_innovations():
	prior = BrickEstimate(
		pos=np.array([0.30, 0.0, 0.0096]), yaw=0.0, size=SIZE, footprint_m=(0.032, 0.016), top_points=100
	)
	near = WristFix(
		pos_xy=np.array([0.31, 0.0]),
		yaw=math.radians(4),
		area_px=900,
		fill=0.95,
		footprint_m=(0.032, 0.016),
		tilt_deg=0.0,
	)
	fused, info = fuse(prior, near, FuseCfg())
	assert info["accepted"] and info["innovation_xy_mm"] == pytest.approx(10.0, abs=0.01)
	assert fused.pos[0] == pytest.approx(0.31) and fused.pos[2] == pytest.approx(0.0096)
	far = WristFix(
		pos_xy=np.array([0.36, 0.0]), yaw=0.0, area_px=900, fill=0.95, footprint_m=(0.032, 0.016), tilt_deg=0.0
	)
	assert fuse(prior, far, FuseCfg())[0] is None


def test_believed_wrist_from_cam_uses_designed_mount_not_offset():
	pytest.importorskip("mujoco")
	from digital_twin_armigo.agent.twin_mujoco import INITIAL_Q, sample_case
	from digital_twin_armigo.constants import SO101_MJCF_DIR
	from digital_twin_armigo.mujoco_cell import Cell
	from digital_twin_armigo.mujoco_cell._mj import mujoco
	from digital_twin_armigo.mujoco_cell.grasp_geometry import Kinematics

	lay_nom = sample_case(600001, False, 5)
	lay_nom.wrist_cam_offset = None
	lay_off = sample_case(600001, False, 5)
	lay_off.wrist_cam_offset = (0.012, -0.008, 0.006, 5.0, -4.0, 6.0)
	cell_nom = Cell(lay_nom, SO101_MJCF_DIR, render_enabled=False)
	cell_off = Cell(lay_off, SO101_MJCF_DIR, render_enabled=False)
	try:
		q = np.array([*INITIAL_Q[:5], 0.5])
		kin_nom, kin_off = Kinematics(cell_nom), Kinematics(cell_off)
		kin_nom.data.qpos[cell_nom.qadr] = q
		kin_off.data.qpos[cell_off.qadr] = q
		tf_nom = believed_wrist_from_cam(cell_nom, kin_nom.data)
		tf_off = believed_wrist_from_cam(cell_off, kin_off.data)
		assert np.allclose(tf_nom, tf_off, atol=1e-12)
		# Verify actual model.cam_pos really differs under offset
		cid = mujoco.mj_name2id(cell_off.model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
		mujoco.mj_kinematics(cell_off.model, kin_off.data)
		mujoco.mj_camlight(cell_off.model, kin_off.data)
		assert not np.allclose(kin_off.data.cam_xpos[cid], tf_off[:3, 3], atol=1e-3)
	finally:
		cell_nom.close()
		cell_off.close()
