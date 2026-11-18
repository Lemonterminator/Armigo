"""Detector crops shared by the MuJoCo renderer and the Isaac RTX re-render (NumPy only, no simulator import)."""

from __future__ import annotations

import numpy as np


def square_crop(
	img: np.ndarray, bbox: tuple[int, int, int, int], pad: float, size: int, clamp: bool = False
) -> np.ndarray:
	"""Square crop of ``img`` centred on ``bbox`` with ``pad`` × its larger side added on each side, resized to
	``size`` × ``size`` (nearest neighbour; out-of-image area is black). With ``clamp``, a crop that fits inside the
	image (``side`` ≤ both image dimensions) is shifted so it lies fully inside it, without a black border; a crop
	that does not fit is left as is, and a crop already inside is unchanged."""
	x, y, w, h = bbox
	side = max(w, h) * (1.0 + 2.0 * pad)
	cx, cy = x + w / 2.0, y + h / 2.0
	xs = np.round(cx - side / 2.0 + (np.arange(size) + 0.5) * side / size).astype(int)
	ys = np.round(cy - side / 2.0 + (np.arange(size) + 0.5) * side / size).astype(int)
	if clamp:
		if side <= img.shape[1]:
			xs = xs + max(0, -int(xs.min())) - max(0, int(xs.max()) - (img.shape[1] - 1))
		if side <= img.shape[0]:
			ys = ys + max(0, -int(ys.min())) - max(0, int(ys.max()) - (img.shape[0] - 1))
	xs = np.clip(xs, -1, img.shape[1])
	ys = np.clip(ys, -1, img.shape[0])
	out = np.zeros((size, size, img.shape[2]), dtype=img.dtype)
	vx = (xs >= 0) & (xs < img.shape[1])
	vy = (ys >= 0) & (ys < img.shape[0])
	out[np.ix_(vy, vx)] = img[np.ix_(ys[vy], xs[vx])]
	return out
