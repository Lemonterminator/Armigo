"""Small UNet: brick face classes (bg/top/bottom/side) and unordered top/bottom corner heatmaps.

Only ops that OpenCV's dnn module imports from ONNX: Conv, BatchNorm, ReLU, MaxPool, nearest Resize, Concat."""

from __future__ import annotations

import io
import struct
from pathlib import Path

import numpy as np
import torch
from torch import nn

__all__ = ["BrickSegUNet", "block", "export_onnx"]


def block(cin: int, cout: int) -> nn.Sequential:
	return nn.Sequential(
		nn.Conv2d(cin, cout, 3, padding=1, bias=False),
		nn.BatchNorm2d(cout),
		nn.ReLU(inplace=True),
		nn.Conv2d(cout, cout, 3, padding=1, bias=False),
		nn.BatchNorm2d(cout),
		nn.ReLU(inplace=True),
	)


class BrickSegUNet(nn.Module):
	def __init__(self, base: int = 24, classes: int = 4, heat: int = 2):
		super().__init__()
		self.base = int(base)
		self.classes = int(classes)
		self.heat = int(heat)
		c = [base, base * 2, base * 4, base * 8]
		self.enc = nn.ModuleList([block(3, c[0]), block(c[0], c[1]), block(c[1], c[2]), block(c[2], c[3])])
		self.pool = nn.MaxPool2d(2)
		self.up = nn.Upsample(scale_factor=2, mode="nearest")
		self.dec = nn.ModuleList([block(c[3] + c[2], c[2]), block(c[2] + c[1], c[1]), block(c[1] + c[0], c[0])])
		self.head = nn.Conv2d(c[0], classes + heat, 1)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		skips = []
		for i, enc in enumerate(self.enc):
			x = enc(x if i == 0 else self.pool(x))
			skips.append(x)
		x = skips.pop()
		for dec in self.dec:
			x = dec(torch.cat([self.up(x), skips.pop()], dim=1))
		return self.head(x)


# ---- Pure-Python ONNX opset-17 protobuf serializer fallback (used when ``onnx`` package is absent) ----


def _varint(v: int) -> bytes:
	out = bytearray()
	v &= (1 << 64) - 1
	while True:
		b = v & 0x7F
		v >>= 7
		if v:
			out.append(b | 0x80)
		else:
			out.append(b)
			break
	return bytes(out)


def _key(field: int, wire: int) -> bytes:
	return _varint((field << 3) | wire)


def _ld(field: int, data: bytes | bytearray) -> bytes:
	return _key(field, 2) + _varint(len(data)) + bytes(data)


def _str(field: int, text: str) -> bytes:
	return _ld(field, text.encode("utf-8"))


def _i64(field: int, val: int) -> bytes:
	return _key(field, 0) + _varint(int(val))


def _f32(field: int, val: float) -> bytes:
	return _key(field, 5) + struct.pack("<f", float(val))


def _attr_int(name: str, val: int) -> bytes:
	return _str(1, name) + _i64(3, val) + _i64(20, 2)


def _attr_float(name: str, val: float) -> bytes:
	return _str(1, name) + _f32(2, val) + _i64(20, 1)


def _attr_str(name: str, val: str) -> bytes:
	return _str(1, name) + _ld(4, val.encode("utf-8")) + _i64(20, 3)


def _attr_ints(name: str, vals: list[int] | tuple[int, ...]) -> bytes:
	return _str(1, name) + b"".join(_i64(8, int(v)) for v in vals) + _i64(20, 7)


def _tensor_proto(name: str, arr: np.ndarray) -> bytes:
	arr = np.ascontiguousarray(arr, dtype=np.float32)
	buf = bytearray()
	for d in arr.shape:
		buf += _i64(1, int(d))
	buf += _i64(2, 1)  # FLOAT
	buf += _str(8, name)
	buf += _ld(9, arr.tobytes())
	return bytes(buf)


def _value_info(name: str, shape: tuple[int, ...]) -> bytes:
	shape_msg = b"".join(_ld(1, _i64(1, d)) for d in shape)
	tensor_type = _i64(1, 1) + _ld(2, shape_msg)  # FLOAT
	type_proto = _ld(1, tensor_type)
	return _str(1, name) + _ld(2, type_proto)


def _node(op_type: str, inputs: list[str], outputs: list[str], name: str, attrs: list[bytes] = ()) -> bytes:
	buf = bytearray()
	for i in inputs:
		buf += _str(1, i)
	for o in outputs:
		buf += _str(2, o)
	buf += _str(3, name)
	buf += _str(4, op_type)
	for a in attrs:
		buf += _ld(5, a)
	return bytes(buf)


def _serialize_unet_onnx(model: BrickSegUNet, input_hw: tuple[int, int] = (240, 320), opset: int = 17) -> bytes:
	"""Serialize an eval-mode :class:`BrickSegUNet` directly into a standard ONNX opset-17 protobuf."""
	was_training = model.training
	model.eval()
	try:
		nodes: list[bytes] = []
		inits: list[bytes] = []
		idx = 0

		def add_init(name: str, tensor: torch.Tensor | np.ndarray) -> str:
			arr = tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else np.asarray(tensor)
			inits.append(_tensor_proto(name, arr))
			return name

		def add_node(op_type: str, inputs: list[str], attrs: list[bytes] = ()) -> str:
			nonlocal idx
			out = f"t_{idx}"
			nodes.append(_node(op_type, inputs, [out], f"{op_type}_{idx}", attrs))
			idx += 1
			return out

		def emit_block(seq: nn.Sequential, x: str, prefix: str) -> str:
			cur = x
			for step in (0, 3):
				conv: nn.Conv2d = seq[step]
				bn: nn.BatchNorm2d = seq[step + 1]
				w = add_init(f"{prefix}.conv{step}.weight", conv.weight)
				conv_attrs = [
					_attr_ints("dilations", conv.dilation),
					_attr_int("group", conv.groups),
					_attr_ints("kernel_shape", conv.kernel_size),
					_attr_ints("pads", (*conv.padding, *conv.padding)),
					_attr_ints("strides", conv.stride),
				]
				cur = add_node("Conv", [cur, w], conv_attrs)
				scale = add_init(f"{prefix}.bn{step}.weight", bn.weight)
				bias = add_init(f"{prefix}.bn{step}.bias", bn.bias)
				mean = add_init(f"{prefix}.bn{step}.running_mean", bn.running_mean)
				var = add_init(f"{prefix}.bn{step}.running_var", bn.running_var)
				cur = add_node(
					"BatchNormalization",
					[cur, scale, bias, mean, var],
					[_attr_float("epsilon", bn.eps), _attr_float("momentum", 0.9)],
				)
				cur = add_node("Relu", [cur])
			return cur

		x = "input"
		skips: list[str] = []
		for i, enc in enumerate(model.enc):
			if i > 0:
				x = add_node(
					"MaxPool",
					[x],
					[
						_attr_ints("kernel_shape", (2, 2)),
						_attr_ints("pads", (0, 0, 0, 0)),
						_attr_ints("strides", (2, 2)),
					],
				)
			x = emit_block(enc, x, f"enc.{i}")
			skips.append(x)

		scales = add_init("up.scales", np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32))
		x = skips.pop()
		for i, dec in enumerate(model.dec):
			x = add_node(
				"Resize",
				[x, "", scales],
				[
					_attr_str("coordinate_transformation_mode", "asymmetric"),
					_attr_str("mode", "nearest"),
					_attr_str("nearest_mode", "floor"),
				],
			)
			x = add_node("Concat", [x, skips.pop()], [_attr_int("axis", 1)])
			x = emit_block(dec, x, f"dec.{i}")

		hw = add_init("head.weight", model.head.weight)
		hb = add_init("head.bias", model.head.bias)
		head_inputs = [x, hw] + ([hb] if model.head.bias is not None else [])
		out = f"t_{idx}"
		nodes.append(
			_node(
				"Conv",
				head_inputs,
				["output"],
				f"Conv_{idx}",
				[
					_attr_ints("dilations", model.head.dilation),
					_attr_int("group", model.head.groups),
					_attr_ints("kernel_shape", model.head.kernel_size),
					_attr_ints("pads", (*model.head.padding, *model.head.padding)),
					_attr_ints("strides", model.head.stride),
				],
			)
		)
		_ = out

		h, w = int(input_hw[0]), int(input_hw[1])
		graph = bytearray()
		for n in nodes:
			graph += _ld(1, n)
		graph += _str(2, "brickseg_unet")
		for init in inits:
			graph += _ld(5, init)
		graph += _ld(11, _value_info("input", (1, 3, h, w)))
		graph += _ld(12, _value_info("output", (1, model.classes + model.heat, h, w)))

		opset_msg = _str(1, "") + _i64(2, opset)
		model_msg = bytearray()
		model_msg += _i64(1, 8)  # ir_version
		model_msg += _str(5, "digital_twin_armigo")
		model_msg += _ld(7, graph)
		model_msg += _ld(8, opset_msg)
		return bytes(model_msg)
	finally:
		if was_training:
			model.train()


def export_onnx(model: BrickSegUNet, out_path: Path, input_hw: tuple[int, int] = (240, 320), opset: int = 17) -> str:
	"""Export ``model`` to ONNX opset ``opset``; falls back to the pure-Python serializer if ``onnx`` is missing."""
	out_path = Path(out_path)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	model.eval()
	dummy = torch.zeros(1, 3, input_hw[0], input_hw[1], dtype=torch.float32)
	try:
		buf = io.BytesIO()
		torch.onnx.export(
			model,
			(dummy,),
			buf,
			input_names=["input"],
			output_names=["output"],
			opset_version=opset,
			dynamo=False,
		)
		out_path.write_bytes(buf.getvalue())
		return "torch.onnx"
	except (ImportError, ModuleNotFoundError, RuntimeError):
		data = _serialize_unet_onnx(model, input_hw=input_hw, opset=opset)
		out_path.write_bytes(data)
		return "pure_python_onnx"
