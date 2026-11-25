from pathlib import Path

import cv2
import numpy as np
import pytest

DATA_DIR = Path(__file__).resolve().parent / "data"


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
