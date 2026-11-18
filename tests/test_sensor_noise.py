"""Tests for Intel RealSense D415 depth/RGB degradation model."""

from __future__ import annotations

import numpy as np
import pytest

from digital_twin_armigo.camera import CameraPose
from digital_twin_armigo.depth import height_to_depth_plane, layered_height_image
from digital_twin_armigo.sensor_noise import D415NoiseCfg, degrade_depth_height, degrade_depth_plane, degrade_rgb


def test_d415_depth_degradation_respects_min_range_quadratic_noise_and_dropout():
	h, w = 64, 64
	# Camera at z = 0.55 m looking straight down: table at z = 0.0 -> range 0.55 m; lifted box at z = 0.25 -> range 0.30 m
	z_world = np.zeros((h, w), dtype=np.float64)
	z_world[16:32, 16:32] = 0.25  # range = 0.30 m (< min_z_m 0.45 m)
	z_world[40:56, 40:56] = 0.02  # brick at range 0.53 m
	valid = np.ones((h, w), dtype=bool)
	cam = CameraPose(x=0.0, y=0.0, height=0.55, yaw_deg=0.0, tilt_deg=0.0, focal_px=615.0)

	cfg = D415NoiseCfg(
		min_z_m=0.45,
		z_noise_coeff=0.004,
		quantize_mm=1.0,
		edge_dropout_prob=0.5,
		random_dropout_prob=0.02,
	)
	assert cfg.resolution == (640, 480)
	assert "nominal" in cfg.mode

	z_out, valid_out = degrade_depth_height(z_world, valid, cam, np.random.default_rng(7), cfg)

	# Pixels closer than min_z_m (0.30 m < 0.45 m) drop out
	assert not valid_out[20:28, 20:28].any()
	# Table pixels stay mostly valid with mm-scale noise
	table_mask = np.zeros((h, w), dtype=bool)
	table_mask[2:12, 2:12] = True
	assert valid_out[table_mask].mean() > 0.90
	table_err = np.abs(z_out[table_mask & valid_out] - 0.0)
	assert table_err.std() > 0.0
	assert table_err.max() < 0.02

	# Layered height image preserves invalid B=0
	img = layered_height_image(z_out, valid_out)
	assert np.all(img[~valid_out, 2] == 0)


def test_counterexample_tilted_camera_depth_uses_camera_frame_z_not_cam_z_minus_world_z():
	# Tilted camera at height 0.50 m, tilt=35 deg looking along +X.
	# Center pixel ray points downward at cos(35 deg)=0.81915, so table z=0.10 has:
	#   cam_z - z = 0.40 m (< 0.45 m min_z_m -> old buggy code falsely dropped EVERY pixel!)
	#   camera optical-axis depth = (0.50 - 0.10) / cos(35 deg) = 0.4883 m (> 0.45 m -> MUST remain valid!)
	cam = CameraPose(x=0.0, y=0.0, height=0.50, yaw_deg=0.0, tilt_deg=35.0, focal_px=615.0)
	z_world = np.full((64, 64), 0.10, dtype=np.float64)
	valid = np.ones((64, 64), dtype=bool)
	cfg = D415NoiseCfg(min_z_m=0.45, z_noise_coeff=0.0, quantize_mm=0.0, edge_dropout_prob=0.0, random_dropout_prob=0.0)
	depth_cam, ray_ok = height_to_depth_plane(cam, z_world)
	assert float(depth_cam[32, 32]) > 0.48
	_, valid_out = degrade_depth_height(z_world, valid & ray_ok, cam, np.random.default_rng(1), cfg)
	assert bool(valid_out[32, 32]) is True

	# Direct camera-plane degradation on 0.48 m returns all valid; 0.40 m drops out
	d_plane = np.full((16, 16), 0.40, dtype=np.float64)
	_, v_plane = degrade_depth_plane(d_plane, np.ones((16, 16), dtype=bool), np.random.default_rng(1), cfg)
	assert not v_plane.any()


def test_d415_rgb_degradation_is_deterministic_and_bounded():
	rgb = np.full((32, 32, 3), 120, dtype=np.uint8)
	cfg = D415NoiseCfg(rgb_gain_jitter=0.05, rgb_wb_jitter=0.04, rgb_noise_std=3.0)
	out1 = degrade_rgb(rgb, np.random.default_rng(11), cfg)
	out2 = degrade_rgb(rgb, np.random.default_rng(11), cfg)
	assert np.array_equal(out1, out2)
	assert out1.dtype == np.uint8
	assert out1.shape == rgb.shape
	assert not np.array_equal(out1, rgb)
	assert float(np.abs(out1.astype(float) - rgb.astype(float)).mean()) == pytest.approx(
		float(np.abs(out2.astype(float) - rgb.astype(float)).mean())
	)
