"""Tests for the modules both simulators share: joint units, camera geometry and the layered depth encoding."""

from __future__ import annotations

import math
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from digital_twin_armigo.camera import CameraPose, as_pose, camera_forward, look_from, opengl_axes, project, table_point
from digital_twin_armigo.constants import (
	CALIBRATION,
	COARSE_MAX_M,
	FINE_MAX_M,
	GRIPPER_RAD,
	JOINTS,
	OFFSETS_DEG,
	REST_U,
	TICKS_PER_DEG,
	TOP_FULL,
)
from digital_twin_armigo.depth import decode_height, depth_plane_to_height, layered_height_image
from digital_twin_armigo.joints import normalized_to_rad, rad_to_normalized

REPO = Path(__file__).resolve().parents[1]
TOP = CameraPose(0.63, 0.18, 0.51, yaw_deg=197.5, tilt_deg=26.1, roll_deg=0.4, focal_px=612.5)


def _normalized_to_rad_reference(u: np.ndarray) -> np.ndarray:
	"""The per-joint loop the simulators each carried before the conversion was shared."""
	out = np.empty(6)
	for i, name in enumerate(JOINTS[:5]):
		lo, hi = CALIBRATION[name]
		out[i] = math.radians(u[i] / 100.0 * (hi - lo) / 2.0 / TICKS_PER_DEG + OFFSETS_DEG.get(name, 0.0))
	out[5] = GRIPPER_RAD[0] + np.clip(u[5], 0, 100) / 100.0 * (GRIPPER_RAD[1] - GRIPPER_RAD[0])
	return out


def test_normalized_to_rad_matches_the_per_joint_formula_exactly():
	rng = np.random.default_rng(0)
	for u in [np.array(REST_U), *rng.uniform(-100, 100, (50, 6))]:
		assert np.array_equal(normalized_to_rad(u), _normalized_to_rad_reference(u))


def test_joint_conversion_round_trips_and_clips_the_gripper():
	u = np.array([12.5, -40.0, 63.0, -98.2, 24.1, 33.7])
	assert rad_to_normalized(normalized_to_rad(u)) == pytest.approx(u, abs=1e-4)
	assert rad_to_normalized(normalized_to_rad(u)).dtype == np.float32
	assert normalized_to_rad(np.array([0, 0, 0, 0, 0, 150.0]))[5] == pytest.approx(GRIPPER_RAD[1])
	assert normalized_to_rad(np.array([0, 0, 0, 0, 0, -20.0]))[5] == pytest.approx(GRIPPER_RAD[0])


def test_camera_frame_is_orthonormal_and_looks_down_at_its_tilt():
	right, up, forward = TOP.frame()
	basis = np.stack([right, up, forward])
	assert basis @ basis.T == pytest.approx(np.eye(3), abs=1e-12)
	assert math.degrees(math.acos(-forward[2])) == pytest.approx(TOP.tilt_deg)
	assert camera_forward(TOP) == pytest.approx(forward, abs=1e-12)
	assert np.cross(right, up) == pytest.approx(opengl_axes(TOP)[2], abs=1e-12)


def test_as_pose_accepts_the_camera_mapping_of_a_camera_file():
	mapping = {"x": 0.6, "y": 0.2, "height": 0.5, "yaw_deg": 190.0, "tilt_deg": 25.0, "roll_deg": 1.0, "focal_px": 600.0}
	assert as_pose(mapping) == CameraPose(**mapping)
	assert as_pose(TOP) is TOP
	for a, b in zip(opengl_axes(mapping), opengl_axes(CameraPose(**mapping)), strict=True):
		assert np.array_equal(a, b)


def test_look_from_puts_the_target_on_the_optical_axis():
	pose = look_from((0.2, 0.05), 0.5, 0.3, 160.0, focal_px=520.0)
	uv, front = project(pose, np.array([[0.2, 0.05, 0.0]]), (640, 480))
	assert front.all()
	assert uv[0] == pytest.approx([320.0, 240.0], abs=1e-6)


def test_table_point_inverts_project_on_a_horizontal_plane():
	points = np.array([[0.25, 0.05, 0.02], [0.35, -0.10, 0.02], [0.18, 0.20, 0.02]])
	uv, front = project(TOP, points, TOP_FULL)
	assert front.all()
	assert table_point(TOP, uv[:, 0], uv[:, 1], 0.02) == pytest.approx(points[:, :2], abs=1e-9)


def test_project_flags_points_behind_the_camera():
	_, front = project(TOP, np.array([[0.63, 0.18, 2.0]]), TOP_FULL)
	assert not front.any()


def test_layered_height_image_round_trips_through_decode():
	height = np.array([[0.0, 0.01, 0.039], [0.05, 0.2, 0.29]])
	valid = np.array([[True, True, True], [True, True, False]])
	img = layered_height_image(height, valid)
	assert img.dtype == np.uint8 and img.shape == (2, 3, 3)
	assert np.array_equal(img[..., 2], np.where(valid, 255, 0))
	assert np.array_equal(img[1, 2], [0, 0, 0])
	decoded = decode_height(img)
	assert np.isnan(decoded[1, 2])
	fine = height < 0.035
	assert decoded[valid & fine] == pytest.approx(height[valid & fine], abs=FINE_MAX_M / 255)
	assert decoded[valid & ~fine] == pytest.approx(height[valid & ~fine], abs=COARSE_MAX_M / 255)


def test_layered_height_image_saturates_and_never_goes_negative():
	img = layered_height_image(np.array([[-0.01, 1.0]]), np.array([[True, True]]))
	assert np.array_equal(img[0, 0], [0, 0, 255])
	assert np.array_equal(img[0, 1], [255, 255, 255])


def test_depth_plane_to_height_recovers_a_flat_table():
	wid, hgt = TOP_FULL
	right, up, forward = TOP.frame()
	u, v = np.meshgrid(np.arange(wid) + 0.5, np.arange(hgt) + 0.5)
	ray = ((u - wid / 2) / TOP.focal_px)[..., None] * right + (-(v - hgt / 2) / TOP.focal_px)[..., None] * up + forward
	depth = -TOP.height / ray[..., 2]  # distance along the optical axis at which each ray meets z = 0
	world_z, valid = depth_plane_to_height(TOP, depth)
	assert valid.all()
	assert np.abs(world_z).max() < 1e-12
	depth[0, 0] = np.inf
	assert not depth_plane_to_height(TOP, depth)[1][0, 0]


def test_mujoco_gl_backend_is_only_forced_on_linux():
	"""``MUJOCO_GL=egl`` is invalid on Windows and macOS: importing the cell there must leave the variable alone."""
	env = {k: v for k, v in os.environ.items() if k != "MUJOCO_GL"}
	env["PYTHONPATH"] = str(REPO / "src")
	code = "import os, digital_twin_armigo.mujoco_cell._mj; print(os.environ.get('MUJOCO_GL', ''))"
	out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True).stdout.strip()
	assert out == ("egl" if platform.system() == "Linux" else "")


def test_launcher_script_reexports_the_package():
	sys.path.insert(0, str(REPO / "scripts" / "sim"))
	try:
		import so101_sort_sim as launcher
	finally:
		sys.path.pop(0)
	from digital_twin_armigo import mujoco_cell

	for name in ("Layout", "Cell", "scene_xml", "simulate", "render_views", "main", "DUPLO_2X2", "SIM_ASSETS_DIR"):
		assert getattr(launcher, name) is getattr(mujoco_cell, name)
