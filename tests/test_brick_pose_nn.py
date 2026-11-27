import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from digital_twin_armigo.agent.brick_pose_nn import BrickSegNet, peaks, pose_from_keypoints
from digital_twin_armigo.agent.wrist_localize import WristFix

DATA_DIR = Path(__file__).resolve().parent / "data"
SIZE = (0.032, 0.016, 0.0192)


def test_brickseg_unet_shape_and_params():
	torch = pytest.importorskip("torch")
	from digital_twin_armigo.perception_nn.model import BrickSegUNet, export_onnx

	model = BrickSegUNet(base=24, classes=4, heat=2)
	n_params = sum(p.numel() for p in model.parameters())
	assert n_params > 1_000_000
	y = model(torch.zeros(2, 3, 240, 320))
	assert tuple(y.shape) == (2, 6, 240, 320)
	_ = export_onnx


def test_tiny_onnx_loads_in_cv2_dnn():
	onnx_path = DATA_DIR / "brickseg_tiny.onnx"
	meta_path = DATA_DIR / "brickseg_tiny.json"
	assert onnx_path.is_file() and meta_path.is_file()
	assert onnx_path.stat().st_size <= 200 * 1024
	net = cv2.dnn.readNetFromONNX(str(onnx_path))
	net.setInput(np.zeros((1, 3, 240, 320), dtype=np.float32))
	out = net.forward()
	assert out.shape == (1, 6, 240, 320)


def _brick_corners_world(pos, rot, size):
	hx, hy, hz = (s / 2 for s in size)
	local = [(sx * hx, sy * hy, hz) for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1))] + [
		(sx * hx, sy * hy, -hz) for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1))
	]
	return np.asarray(local) @ np.asarray(rot).T + np.asarray(pos)


def _cam(tilt_deg=33.0):
	t = math.radians(tilt_deg)
	tilt = np.array([[1, 0, 0], [0, math.cos(t), -math.sin(t)], [0, math.sin(t), math.cos(t)]])
	w2c = np.eye(4)
	w2c[:3, :3] = np.diag([1.0, -1.0, -1.0]) @ tilt
	w2c[:3, 3] = (0.30, -0.09, 0.15)
	return np.array([[407.0, 0, 319.5], [0, 407.0, 239.5], [0, 0, 1]]), w2c


def _project(k, w2c, pts):
	c = np.linalg.inv(w2c)
	p = pts @ c[:3, :3].T + c[:3, 3]
	return (p[:, :2] / p[:, 2:3]) @ k[:2, :2].T + k[:2, 2]


def test_planar_pose_from_four_unordered_top_corners():
	k, w2c = _cam()
	yaw = math.radians(25)
	rot = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
	corners = _brick_corners_world(np.array([0.305, 0.004, 0.0096]), rot, SIZE)
	top = _project(k, w2c, corners[:4]) + np.random.default_rng(0).normal(0, 0.3, (4, 2))
	top = top[[2, 0, 3, 1]]  # unordered
	fit, why = pose_from_keypoints(top, np.zeros((0, 2)), None, k, w2c, SIZE, prior_yaw=math.radians(20))
	assert why is None and fit.mode == "ippe4"
	assert np.linalg.norm(fit.pos[:2] - [0.305, 0.004]) < 1e-3
	assert abs(math.degrees(fit.yaw) - 25) < 1.0


def test_symmetric_twin_resolved_by_prior():
	k, w2c = _cam()
	corners = _brick_corners_world(np.array([0.30, 0.0, 0.0096]), np.eye(3), SIZE)
	top = _project(k, w2c, corners[:4])
	fit, _ = pose_from_keypoints(top, np.zeros((0, 2)), None, k, w2c, SIZE, prior_yaw=math.pi + 0.05)
	assert abs(math.degrees(fit.yaw) - 180) < 1.0


def test_peaks_subpixel():
	heat = np.zeros((2, 240, 320), np.float32)
	yy, xx = np.mgrid[0:240, 0:320]
	heat[0] = np.exp(-((xx - 100.4) ** 2 + (yy - 50.6) ** 2) / (2 * 1.5**2))
	p = peaks(heat[0], scale=(2.0, 2.0))
	assert len(p) == 1 and abs(p[0, 0] - 200.8) < 0.3 and abs(p[0, 1] - 101.2) < 0.3


def test_mask4_fallback_recovers_pose():
	k, w2c = _cam()
	yaw = math.radians(15)
	rot = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
	corners = _brick_corners_world(np.array([0.305, -0.005, 0.0096]), rot, SIZE)
	top_proj = _project(k, w2c, corners[:4])
	mask = np.zeros((480, 640), np.uint8)
	cv2.fillPoly(mask, [np.round(top_proj).astype(np.int32)], 1)
	fit, why = pose_from_keypoints(top_proj[:2], np.zeros((0, 2)), mask, k, w2c, SIZE, prior_yaw=math.radians(10))
	assert why is None and fit.mode == "mask4"
	assert np.linalg.norm(fit.pos[:2] - [0.305, -0.005]) < 0.002


def test_onnx_net_wrapper_runs_tiny_fixture():
	data_dir = Path(__file__).resolve().parent / "data"
	net = BrickSegNet(data_dir / "brickseg_tiny.onnx", data_dir / "brickseg_tiny.json")
	out = net.predict(np.zeros((480, 640, 3), np.uint8))
	assert out.cls.shape == (240, 320)
	assert out.heat.shape == (2, 240, 320)
	assert out.scale == (2.0, 2.0)
	assert float(out.heat.min()) >= 0.0 and float(out.heat.max()) <= 1.0
	assert WristFix(
		pos_xy=np.zeros(2), yaw=0.0, area_px=1, fill=1.0, footprint_m=(0.03, 0.01), tilt_deg=0.0
	).pos_xy.shape == (2,)


def test_localize_from_wrist_dispatches_unet_without_changing_lab():
	from digital_twin_armigo.agent.wrist_localize import localize_from_wrist

	k, w2c = _cam()
	data_dir = Path(__file__).resolve().parent / "data"
	net = BrickSegNet(data_dir / "brickseg_tiny.onnx", data_dir / "brickseg_tiny.json")
	rgb = np.full((480, 640, 3), 120, dtype=np.uint8)
	fix_lab, why_lab = localize_from_wrist(rgb, k, w2c, (0.30, 0.0), 0.0, SIZE, 0.0192)
	assert fix_lab is None and why_lab == "NO_BLOB"
	fix_nn, why_nn = localize_from_wrist(rgb, k, w2c, (0.30, 0.0), 0.0, SIZE, 0.0192, segmenter="unet", net=net)
	assert fix_nn is None or why_nn in (None, "NO_BLOB", "CLIPPED", "OUT_OF_GATE", "TILTED", "NO_POSE")
