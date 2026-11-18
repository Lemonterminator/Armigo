"""RealSense D415 depth and RGB sensor degradation models for synthetic closed-loop testing."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .camera import CameraPose
from .depth import depth_plane_to_height, height_to_depth_plane


@dataclass(frozen=True)
class D415NoiseCfg:
	"""Nominal Intel RealSense D415 stereo-depth and RGB sensor degradation parameters.

	Based on Intel D400-series datasheet specs: ~0.45 m min-Z at 640x480 HD, quadratic Z-error scaling
	in camera optical depth, ~99% baseline fill rate with depth-discontinuity edge dropouts, and
	exposure/WB jitter. This is a nominal assumed model pending real D415 hardware verification.
	"""

	version: str = "d415-nominal-v2"
	mode: str = "hd-nominal-depth-plane"
	resolution: tuple[int, int] = (640, 480)
	min_z_m: float = 0.45
	z_noise_coeff: float = 0.004
	quantize_mm: float = 1.0
	edge_dropout_prob: float = 0.25
	edge_grad_thresh_m: float = 0.010
	random_dropout_prob: float = 0.005
	rgb_gain_jitter: float = 0.04
	rgb_wb_jitter: float = 0.03
	rgb_noise_std: float = 2.5


DEFAULT_D415_NOISE = D415NoiseCfg()


def degrade_depth_plane(
	depth_m: np.ndarray,
	valid: np.ndarray,
	rng: np.random.Generator,
	cfg: D415NoiseCfg = DEFAULT_D415_NOISE,
) -> tuple[np.ndarray, np.ndarray]:
	"""Apply D415 min-Z cutoff, quadratic Z noise, quantization, and dropouts in camera coordinates."""
	d = np.asarray(depth_m, dtype=np.float64).copy()
	v = np.asarray(valid, dtype=bool).copy() & np.isfinite(d) & (d > 0.0)

	# 1. Minimum optical depth distance cutoff in camera coordinates (stereo overlap lost < min_z_m)
	if cfg.min_z_m > 0.0:
		v &= d >= cfg.min_z_m

	# 2. Discontinuity edge dropouts in camera depth (flying pixels / stereo shadow edges)
	if cfg.edge_dropout_prob > 0.0 and d.shape[0] > 1 and d.shape[1] > 1:
		gx = np.abs(np.diff(d, axis=1, prepend=d[:, :1]))
		gy = np.abs(np.diff(d, axis=0, prepend=d[:1, :]))
		edges = (gx > cfg.edge_grad_thresh_m) | (gy > cfg.edge_grad_thresh_m)
		v &= ~(edges & (rng.random(d.shape) < cfg.edge_dropout_prob))

	# 3. Random missing pixels
	if cfg.random_dropout_prob > 0.0:
		v &= rng.random(d.shape) >= cfg.random_dropout_prob

	# 4. Quadratic Z-noise in camera optical depth + quantization
	if cfg.z_noise_coeff > 0.0:
		sigma = cfg.z_noise_coeff * (np.maximum(d, 0.05) ** 2)
		d = np.where(v, np.maximum(0.05, d + rng.normal(0.0, 1.0, size=d.shape) * sigma), d)
	if cfg.quantize_mm > 0.0:
		q_m = cfg.quantize_mm * 1e-3
		d = np.where(v, np.round(d / q_m) * q_m, d)

	return d, v


def degrade_depth_height(
	z_world: np.ndarray,
	valid: np.ndarray,
	cam: CameraPose | Mapping[str, float],
	rng: np.random.Generator,
	cfg: D415NoiseCfg = DEFAULT_D415_NOISE,
	crop: tuple[int, int, int, int] | None = None,
	full_shape: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
	"""Convert world heights to camera depth via ``cam``, apply D415 degradation, and project back to world height."""
	depth_m, ray_valid = height_to_depth_plane(cam, z_world, crop=crop, full_shape=full_shape)
	d_noisy, v_noisy = degrade_depth_plane(depth_m, np.asarray(valid, dtype=bool) & ray_valid, rng, cfg)
	z_noisy, v_out = depth_plane_to_height(cam, d_noisy, crop=crop, full_shape=full_shape)
	v_final = v_noisy & v_out
	return np.where(v_final, z_noisy, np.asarray(z_world, dtype=np.float64)), v_final


def degrade_rgb(
	rgb: np.ndarray,
	rng: np.random.Generator,
	cfg: D415NoiseCfg = DEFAULT_D415_NOISE,
) -> np.ndarray:
	"""Apply mild exposure gain, white-balance channel shift, and sensor read noise to a uint8 RGB image."""
	img = np.asarray(rgb, dtype=np.float32)
	gain = 1.0 + float(rng.uniform(-cfg.rgb_gain_jitter, cfg.rgb_gain_jitter))
	wb = 1.0 + rng.uniform(-cfg.rgb_wb_jitter, cfg.rgb_wb_jitter, size=(1, 1, 3)).astype(np.float32)
	out = img * gain * wb
	if cfg.rgb_noise_std > 0.0:
		out = out + rng.normal(0.0, cfg.rgb_noise_std, size=img.shape).astype(np.float32)
	return np.clip(np.rint(out), 0, 255).astype(np.uint8)
