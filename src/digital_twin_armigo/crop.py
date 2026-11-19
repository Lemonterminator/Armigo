"""Detector crops shared by the MuJoCo renderer and the Isaac RTX re-render (NumPy only, no simulator import)."""

from __future__ import annotations

import numpy as np


def square_crop_indices(
	shape: tuple[int, ...], bbox: tuple[int, int, int, int], pad: float, size: int, clamp: bool = False
) -> tuple[np.ndarray, np.ndarray]:
	"""Source column and row index of every output pixel of :func:`square_crop` on an image of ``shape`` (height,
	width, ...); indices outside the image (-1 or the image size) mark black output pixels."""
	x, y, w, h = bbox
	side = max(w, h) * (1.0 + 2.0 * pad)
	cx, cy = x + w / 2.0, y + h / 2.0
	xs = np.round(cx - side / 2.0 + (np.arange(size) + 0.5) * side / size).astype(int)
	ys = np.round(cy - side / 2.0 + (np.arange(size) + 0.5) * side / size).astype(int)
	if clamp:
		if side <= shape[1]:
			xs = xs + max(0, -int(xs.min())) - max(0, int(xs.max()) - (shape[1] - 1))
		if side <= shape[0]:
			ys = ys + max(0, -int(ys.min())) - max(0, int(ys.max()) - (shape[0] - 1))
	return np.clip(xs, -1, shape[1]), np.clip(ys, -1, shape[0])


def square_crop_region(
	shape: tuple[int, ...], bbox: tuple[int, int, int, int], pad: float, size: int, clamp: bool = False
) -> tuple[int, int, int, int]:
	"""``(x0, y0, x1, y1)``, end exclusive: the image pixels :func:`square_crop` samples from, clipped to the image
	(an empty box when the crop lies fully outside it)."""
	xs, ys = square_crop_indices(shape, bbox, pad, size, clamp)
	xs = xs[(xs >= 0) & (xs < shape[1])]
	ys = ys[(ys >= 0) & (ys < shape[0])]
	if not len(xs) or not len(ys):
		return 0, 0, 0, 0
	return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def square_crop(
	img: np.ndarray, bbox: tuple[int, int, int, int], pad: float, size: int, clamp: bool = False
) -> np.ndarray:
	"""Square crop of ``img`` centred on ``bbox`` with ``pad`` × its larger side added on each side, resized to
	``size`` × ``size`` (nearest neighbour; out-of-image area is black). With ``clamp``, a crop that fits inside the
	image (``side`` ≤ both image dimensions) is shifted so it lies fully inside it, without a black border; a crop
	that does not fit is left as is, and a crop already inside is unchanged."""
	xs, ys = square_crop_indices(img.shape, bbox, pad, size, clamp)
	out = np.zeros((size, size, img.shape[2]), dtype=img.dtype)
	vx = (xs >= 0) & (xs < img.shape[1])
	vy = (ys >= 0) & (ys < img.shape[0])
	out[np.ix_(vy, vx)] = img[np.ix_(ys[vy], xs[vx])]
	return out
