"""Top-camera workspace visibility checks for v3 camera sampling."""

from __future__ import annotations

import numpy as np

from ..camera import project
from ..constants import BOWL_H, BOWL_R, TOP_CROP, TOP_FULL
from .layout import Layout

WORKSPACE_CORNERS: tuple[tuple[float, float, float], ...] = (
	(0.10, -0.16, 0.0),
	(0.10, 0.16, 0.0),
	(0.32, 0.16, 0.0),
	(0.32, -0.16, 0.0),
)

_RIM_VIS_RADIUS_FRAC = 0.80


def workspace_visible(layout: Layout, margin_frac: float = 0.05) -> tuple[bool, list[str]]:
	"""Check that table corners, bowls, and brick project inside ``TOP_CROP`` beyond ``margin_frac``.

	Also verifies each bowl's inner rim stays horizontally within ``TOP_FULL`` so right-edge sensor clipping
	cannot break bowl rim circle-fitting.
	"""
	if layout.top is None:
		raise ValueError("layout.top must be set to check workspace visibility")

	named_points: list[tuple[str, tuple[float, float, float]]] = [
		(f"table_corner_{i}", corner) for i, corner in enumerate(WORKSPACE_CORNERS)
	]
	bowls: list[tuple[str, float, float]] = [
		("green_bowl", layout.green[0], layout.green[1]),
		("pink_bowl", layout.pink[0], layout.pink[1]),
		*((f"extra_bowl_{i}", x, y) for i, (_, x, y, _) in enumerate(layout.extra_bowls)),
	]
	for bname, bx, by in bowls:
		named_points.append((bname, (bx, by, 0.0)))
	named_points.append(("brick", (layout.brick_xy[0], layout.brick_xy[1], layout.brick_size[2] / 2.0)))

	names = [name for name, _ in named_points]
	coords = [pt for _, pt in named_points]
	uv, in_front = project(layout.top, np.asarray(coords, dtype=np.float64), TOP_FULL)

	x0, y0, cw, ch = TOP_CROP
	u_lo, u_hi = x0 + margin_frac * cw, x0 + (1.0 - margin_frac) * cw
	v_lo, v_hi = y0 + margin_frac * ch, y0 + (1.0 - margin_frac) * ch

	out_of_view: list[str] = []
	for i, name in enumerate(names):
		if not in_front[i] or uv[i, 0] < u_lo or uv[i, 0] > u_hi or uv[i, 1] < v_lo or uv[i, 1] > v_hi:
			out_of_view.append(name)

	r_vis = _RIM_VIS_RADIUS_FRAC * BOWL_R
	rim_names: list[str] = []
	rim_coords: list[tuple[float, float, float]] = []
	for bname, bx, by in bowls:
		for dx, dy in ((r_vis, 0.0), (-r_vis, 0.0), (0.0, r_vis), (0.0, -r_vis)):
			rim_names.append(bname)
			rim_coords.append((bx + dx, by + dy, BOWL_H))
	ruv, rin_front = project(layout.top, np.asarray(rim_coords, dtype=np.float64), TOP_FULL)
	full_w = float(TOP_FULL[0])
	for i, bname in enumerate(rim_names):
		if bname not in out_of_view and (not rin_front[i] or ruv[i, 0] < 0.0 or ruv[i, 0] > full_w):
			out_of_view.append(bname)

	return (len(out_of_view) == 0, out_of_view)
