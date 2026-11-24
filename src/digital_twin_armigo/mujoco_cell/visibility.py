"""Top-camera workspace visibility checks for v3 camera sampling."""

from __future__ import annotations

import numpy as np

from ..camera import project
from ..constants import TOP_FULL
from .layout import Layout

WORKSPACE_CORNERS: tuple[tuple[float, float, float], ...] = (
	(0.10, -0.16, 0.0),
	(0.10, 0.16, 0.0),
	(0.32, 0.16, 0.0),
	(0.32, -0.16, 0.0),
)


def workspace_visible(layout: Layout, margin_frac: float = 0.05) -> tuple[bool, list[str]]:
	"""Check that table corners, bowls, and brick project inside ``TOP_FULL`` beyond ``margin_frac``."""
	if layout.top is None:
		raise ValueError("layout.top must be set to check workspace visibility")

	named_points: list[tuple[str, tuple[float, float, float]]] = [
		(f"table_corner_{i}", corner) for i, corner in enumerate(WORKSPACE_CORNERS)
	]
	named_points.append(("green_bowl", (layout.green[0], layout.green[1], 0.0)))
	named_points.append(("pink_bowl", (layout.pink[0], layout.pink[1], 0.0)))
	for i, (_, x, y, _) in enumerate(layout.extra_bowls):
		named_points.append((f"extra_bowl_{i}", (x, y, 0.0)))
	named_points.append(("brick", (layout.brick_xy[0], layout.brick_xy[1], layout.brick_size[2] / 2.0)))

	names = [name for name, _ in named_points]
	coords = [pt for _, pt in named_points]
	uv, in_front = project(layout.top, np.asarray(coords, dtype=np.float64), TOP_FULL)

	w, h = TOP_FULL
	u_lo, u_hi = margin_frac * w, (1.0 - margin_frac) * w
	v_lo, v_hi = margin_frac * h, (1.0 - margin_frac) * h

	out_of_view: list[str] = []
	for i, name in enumerate(names):
		if not in_front[i] or uv[i, 0] < u_lo or uv[i, 0] > u_hi or uv[i, 1] < v_lo or uv[i, 1] > v_hi:
			out_of_view.append(name)
	return (len(out_of_view) == 0, out_of_view)
