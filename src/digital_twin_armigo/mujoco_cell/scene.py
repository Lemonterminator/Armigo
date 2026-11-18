"""MJCF generation for one episode of the sorting cell: bowls, brick with defect cross, clutter, markers, cameras."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..constants import (
	BOWL_FLOOR_TOP_M,
	CELL_MARKERS,
	DUPLO_2X2,
	FPS,
	SIM_ASSETS_DIR,
	STUD_PITCH,
	TOP_FULL,
)
from ..physics import PhysicsSpec, brick_spec_for_size
from ..presentation import FACES, defect_strokes, face_frame
from .layout import Layout

SUBSTEPS = 16  # physics steps per dataset frame
PROBE_GROUP = 5  # hidden geom group carrying the defect cross of a normal brick


VISUAL_MESH_NAMES = ("plate_200mm", "duplo_2x2", "duplo_2x3", "duplo_2x4")

_BRICK_MESH_MAP = {
	(2, 2): "duplo_2x2",
	(3, 2): "duplo_2x3",
	(4, 2): "duplo_2x4",
}


def brick_mesh_name(size: tuple[float, float, float]) -> str | None:
	"""Return the visual mesh name matching ``size``, or ``None`` for non-standard sizes."""
	nx = round(size[0] / STUD_PITCH)
	ny = round(size[1] / STUD_PITCH)
	expected_w = round(nx * 0.0159, 4)
	expected_h = round(ny * 0.0159, 4)
	if abs(size[0] - expected_w) > 0.001 or abs(size[1] - expected_h) > 0.001:
		return None
	return _BRICK_MESH_MAP.get((nx, ny))


BOWL_COLLISION_ID = "R6_box_floor_wedge_walls"
_BOWL_WALL_N_WEDGES = 48
_BOWL_WALL_EPS_A = 1e-4
_BOWL_FLOOR_HALF_XY = 0.070
_BOWL_FLOOR_HALF_Z = 0.00125
# 2D Hertel-Mehlhorn convex decomposition of the plate_200mm.obj wall cross-section (r >= 62 mm), (r_m, z_m).
_BOWL_WALL_POLYS: tuple[tuple[tuple[float, float], ...], ...] = (
	((0.062, 0.003), (0.0625, 0.0005), (0.065, 0.0), (0.073, 0.0), (0.0755, 0.0007), (0.081, 0.0028), (0.0695, 0.0034)),
	(
		(0.081, 0.0028),
		(0.087, 0.0064),
		(0.0928, 0.011),
		(0.0975, 0.0155),
		(0.0998, 0.0182),
		(0.1, 0.0201),
		(0.0989, 0.0216),
		(0.091, 0.0145),
		(0.0865, 0.0102),
	),
	((0.0989, 0.0216), (0.0972, 0.022), (0.0956, 0.0211), (0.0942, 0.0188)),
	((0.0989, 0.0216), (0.0942, 0.0188), (0.091, 0.0145)),
	((0.081, 0.0028), (0.0865, 0.0102), (0.0815, 0.0068)),
	((0.081, 0.0028), (0.0815, 0.0068), (0.076, 0.0046)),
	((0.081, 0.0028), (0.076, 0.0046), (0.0695, 0.0034)),
)


def _bowl_wall_mesh_names() -> list[str]:
	return [f"bowl_wall_{p_idx}_{k}" for p_idx in range(len(_BOWL_WALL_POLYS)) for k in range(_BOWL_WALL_N_WEDGES)]


def _bowl_collision_mesh_assets() -> str:
	"""Emit inline convex-wedge ``<mesh>`` assets for the revolved ``plate_200mm.obj`` wall profile."""
	out: list[str] = []
	for p_idx, poly in enumerate(_BOWL_WALL_POLYS):
		for k in range(_BOWL_WALL_N_WEDGES):
			a0 = 2.0 * math.pi * k / _BOWL_WALL_N_WEDGES - _BOWL_WALL_EPS_A
			a1 = 2.0 * math.pi * (k + 1) / _BOWL_WALL_N_WEDGES + _BOWL_WALL_EPS_A
			c0, s0 = math.cos(a0), math.sin(a0)
			c1, s1 = math.cos(a1), math.sin(a1)
			verts: list[str] = []
			for r, z in poly:
				verts.append(f"{r * c0:.6f} {r * s0:.6f} {z:.6f}")
				if r > 1e-7:
					verts.append(f"{r * c1:.6f} {r * s1:.6f} {z:.6f}")
			out.append(f'<mesh name="bowl_wall_{p_idx}_{k}" vertex="{" ".join(verts)}"/>')
	return "".join(out)


def _visual_mesh_assets(assets_dir: Path | None = None) -> str:
	"""Emit MJCF ``<mesh>`` declarations for the bowl and Duplo brick OBJ assets."""
	root = assets_dir or SIM_ASSETS_DIR
	return "".join(f'<mesh name="{name}" file="{(root / f"{name}.obj").as_posix()}"/>' for name in VISUAL_MESH_NAMES)


def _bowl(name: str, xy: tuple[float, float], rgba: str, use_visual_mesh: bool = True, torsional: bool = False) -> str:
	group = ' group="3"' if use_visual_mesh else ""
	condim = ' condim="4"' if torsional else ""
	bottom_z = BOWL_FLOOR_TOP_M - _BOWL_FLOOR_HALF_Z
	bottom = (
		f'<geom type="box" size="{_BOWL_FLOOR_HALF_XY:.3f} {_BOWL_FLOOR_HALF_XY:.3f} {_BOWL_FLOOR_HALF_Z:.5f}" '
		f'pos="{xy[0]:.4f} {xy[1]:.4f} {bottom_z:.5f}" rgba="{rgba}"{condim}{group}/>'
	)
	walls = "".join(
		f'<geom type="mesh" mesh="{mname}" pos="{xy[0]:.4f} {xy[1]:.4f} 0" rgba="{rgba}"{group}/>'
		for mname in _bowl_wall_mesh_names()
	)
	visual = (
		f'<geom type="mesh" mesh="plate_200mm" pos="{xy[0]:.4f} {xy[1]:.4f} 0" '
		f'contype="0" conaffinity="0" mass="0" rgba="{rgba}"/>'
		if use_visual_mesh
		else ""
	)
	return f'<body name="{name}">{visual}{bottom}{walls}</body>'


def _studs(size: tuple[float, float, float], rgba: str) -> str:
	"""Duplo studs on top of a brick of ``size`` (full extents), as visual-only cylinders."""
	nx, ny = max(1, round(size[0] / STUD_PITCH)), max(1, round(size[1] / STUD_PITCH))
	out = []
	for i in range(nx):
		for j in range(ny):
			x, y = (i - (nx - 1) / 2) * STUD_PITCH, (j - (ny - 1) / 2) * STUD_PITCH
			out.append(
				f'<geom type="cylinder" size="0.0047 0.0022" pos="{x:.4f} {y:.4f} {size[2] / 2 + 0.0022:.4f}" '
				f'rgba="{rgba}" contype="0" conaffinity="0" mass="0"/>'
			)
	return "".join(out)


def _brick(
	layout: Layout,
	use_visual_mesh: bool = True,
	physics: PhysicsSpec | None = None,
	mass_scale: float = 1.0,
	friction_scale: float = 1.0,
	face_probes: bool = False,
) -> str:
	sx, sy, sz = (v / 2 for v in layout.brick_size)
	rgba = " ".join(f"{c:.3f}" for c in layout.brick_rgb) + " 1"
	mesh_name = brick_mesh_name(layout.brick_size) if use_visual_mesh else None
	if physics is not None:
		bspec = brick_spec_for_size(layout.brick_size, physics).scaled(mass_scale, friction_scale)
		mass_kg = bspec.mass_kg
		mu_slide = bspec.dynamic_friction
		solref = "0.009 1"
	else:
		mass_kg = 0.008 * mass_scale
		mu_slide = 1.5 * friction_scale
		solref = "0.004 1"
	# The defect (see :func:`_defect_geoms`) is drawn on one face. A normal brick carries the same geoms in the hidden
	# geom group 5, so the presentation planner can score views of that face without it appearing in any rendered image.
	stud_top_offset = 0.0035 if mesh_name is not None else 0.0044
	cross = _defect_geoms(layout, stud_top_offset)
	probes = _face_probes(layout.brick_size, stud_top_offset) if face_probes else ""
	yaw = math.radians(layout.brick_yaw_deg)
	if mesh_name is not None:
		brick_geoms = (
			f'<geom name="brick" type="mesh" mesh="{mesh_name}" contype="0" conaffinity="0" mass="0" rgba="{rgba}"/>'
			f'<geom name="brick_col" type="box" size="{sx:.4f} {sy:.4f} {sz:.4f}" mass="{mass_kg:.5f}" '
			f'friction="{mu_slide:.4f} 0.02 0.002" rgba="{rgba}" solref="{solref}" group="3"/>'
		)
	else:
		studs = _studs(layout.brick_size, rgba)
		brick_geoms = (
			f'<geom name="brick" type="box" size="{sx:.4f} {sy:.4f} {sz:.4f}" mass="{mass_kg:.5f}" '
			f'friction="{mu_slide:.4f} 0.02 0.002" rgba="{rgba}" solref="{solref}"/>{studs}'
		)
	return (
		f'<body name="brick" pos="{layout.brick_xy[0]:.4f} {layout.brick_xy[1]:.4f} {sz + 0.0005:.4f}" '
		f'euler="0 0 {yaw:.4f}"><freejoint name="brick"/>'
		f"{brick_geoms}{cross}{probes}</body>"
	)


DEFECT_KINDS = ("cross", "tape", "chip", "scuff", "discolor", "cross_var")
"""Surface defect kinds a brick may carry (``Layout.defect_kind``); all visual only, named ``defect_<k>``."""

_DEFECT_LIFT_M = 0.0003  # defect geoms sit this far off the face, above the face probes (0.15 mm)
_PATCH_HALF_T_M = 0.0001  # half thickness of the flat patches of the non-cross kinds
_CROSS_VAR_INKS = ((0.05, 0.05, 0.08), (0.08, 0.10, 0.30), (0.30, 0.06, 0.06))
_CROSS_VAR_MAX_ATTEMPTS = 200
_MIN_STROKE_HALF_LENGTH_M = 1e-6  # only rejects degenerate strokes, so every previously rendered layout is unchanged


def _defect_rng(layout: Layout) -> np.random.Generator:
	"""Deterministic randomness for the defect shape: a function of the layout's defect fields only."""
	kind_id = DEFECT_KINDS.index(layout.defect_kind)
	return np.random.default_rng([int(layout.defect_seed) & 0xFFFFFFFF, int(layout.defect_face), kind_id])


def _fit_half_length(c: np.ndarray, d: np.ndarray, s: np.ndarray, w: float, half: np.ndarray) -> float:
	"""Largest half length ``L`` such that the rectangle centred at ``c`` (face (u, v) coordinates) with long axis
	``d``, side axis ``s`` and half width ``w`` stays inside the face half extents ``half``."""
	lim = np.inf
	for e in range(2):
		if abs(d[e]) > 1e-9:
			lim = min(lim, (half[e] - abs(c[e]) - w * abs(s[e])) / abs(d[e]))
	return max(float(lim), 0.0)


def _face_half_extents(layout: Layout) -> np.ndarray:
	"""In-plane half extents (u, v) of ``layout.defect_face``, matching the ``half`` of :func:`_defect_shapes`."""
	size = np.asarray(layout.brick_size, dtype=np.float64)
	axis = FACES[layout.defect_face][0]
	_, _, _, v = face_frame(layout.brick_size, layout.defect_face)
	return np.array([size[(axis + 1) % 3], float(np.abs(v) @ size)]) / 2.0


def _shrink_stroke(
	p: np.ndarray, t: float, d: np.ndarray, s: np.ndarray, w: float, half_length: float, bound: np.ndarray
) -> tuple[np.ndarray, float]:
	"""Centre and half length of a stroke through ``p`` at fraction ``t``, shrunk to stay inside ``bound``."""
	offset = 1.0 - 2.0 * t
	length = half_length
	c = p + offset * length * d
	for _ in range(16):
		fit = _fit_half_length(c, d, s, w, bound)
		if fit >= length:
			break
		length = fit
		c = p + offset * length * d
	return c, length


def _stroke_corners(c: np.ndarray, d: np.ndarray, s: np.ndarray, half_length: float, half_width: float) -> np.ndarray:
	"""The four corners (face (u, v) coordinates) of a stroke rectangle."""
	return np.array(
		[c + along * half_length * d + across * half_width * s for along in (-1.0, 1.0) for across in (-1.0, 1.0)]
	)


def _cross_var_fallback(
	rng: np.random.Generator, half: np.ndarray
) -> list[tuple[tuple[float, float], float, float, float]]:
	"""Two face diagonals through the face centre, shortened to fit: the deterministic last resort."""
	bound = 0.95 * half
	width = min(float(rng.uniform(0.0007, 0.0016)), 0.15 * float(half.min()))
	strokes = []
	for ang in (math.pi / 4.0, 3.0 * math.pi / 4.0):
		d = np.array([math.cos(ang), math.sin(ang)])
		s = np.array([-d[1], d[0]])
		length = _fit_half_length(np.zeros(2), d, s, width, bound)
		strokes.append(((0.0, 0.0), ang, float(length), width))
	return strokes


def _cross_var_strokes(
	rng: np.random.Generator, half: np.ndarray, attempts: int = _CROSS_VAR_MAX_ATTEMPTS
) -> tuple[list[tuple[tuple[float, float], float, float, float]], bool]:
	"""Sample the strokes of :func:`cross_var_strokes`; also report whether the diagonal fallback was used."""
	bound = 0.95 * half
	width_cap = 0.15 * float(half.min())
	largest = float(half.max())
	min_area = 0.25 * 4.0 * float(half[0] * half[1])
	for _ in range(attempts):
		theta = float(rng.uniform(0.0, math.pi))
		skew = math.radians(90.0 + float(rng.uniform(-25.0, 25.0)))
		width = min(float(rng.uniform(0.0007, 0.0016)), width_cap)
		point = rng.uniform(-1.0, 1.0, 2) * half
		along = rng.uniform(0.3, 0.7, 2)
		lengths = rng.uniform(0.35, 0.75, 2) * largest
		strokes = []
		corners = []
		for k in range(2):
			ang = theta if k == 0 else theta + skew
			d = np.array([math.cos(ang), math.sin(ang)])
			s = np.array([-d[1], d[0]])
			c, length = _shrink_stroke(point, float(along[k]), d, s, width, float(lengths[k]), bound)
			strokes.append((tuple(float(v) for v in c), ang, length, width))
			corners.append(_stroke_corners(c, d, s, length, width))
		if min(float(st[2]) for st in strokes) < _MIN_STROKE_HALF_LENGTH_M:
			continue  # a stroke squeezed to zero length is an invalid MuJoCo geom (size 0)
		pts = np.concatenate(corners)
		span = pts.max(axis=0) - pts.min(axis=0)
		if np.all(np.abs(pts) <= bound + 1e-12) and float(span[0] * span[1]) >= min_area - 1e-15:
			return strokes, False
	return _cross_var_fallback(rng, half), True


def _cross_var_sample(
	layout: Layout,
) -> tuple[list[tuple[tuple[float, float], float, float, float]], tuple[float, float, float], bool]:
	"""Strokes, ink colour and whether the diagonal fallback was used, for a ``cross_var`` layout."""
	rng = _defect_rng(layout)
	strokes, fallback = _cross_var_strokes(rng, _face_half_extents(layout))
	return strokes, _CROSS_VAR_INKS[int(rng.integers(len(_CROSS_VAR_INKS)))], fallback


def cross_var_strokes(layout: Layout) -> list[tuple[tuple[float, float], float, float, float]]:
	"""Two strokes of a randomised, hand-drawn cross on the brick's defect face.

	Each stroke is ``(centre (u, v), angle from u, half length, half width)`` in metres, in the face frame of
	:func:`~digital_twin_armigo.presentation.face_frame`. The strokes meet at a point uniformly offset from the face
	centre, are skewed by up to 25 degrees from perpendicular, have asymmetric arms and independent lengths, and
	keep every corner inside 95% of the face with a bounding box of at least a quarter of the face area. The
	deterministic fallback is the pair of face diagonals through the centre, shortened to fit. Everything is a
	function of :func:`_defect_rng` alone.
	"""
	return _cross_var_sample(layout)[0]


def _mix(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
	return np.clip((1.0 - t) * np.asarray(a, dtype=np.float64) + t * np.asarray(b, dtype=np.float64), 0.0, 1.0)


def _hue_shift(rgb: np.ndarray, dh: float, sat: float, val: float) -> np.ndarray:
	"""``rgb`` with its HSV hue moved by ``dh`` (turns) and saturation / value scaled."""
	import colorsys

	h, s, v = colorsys.rgb_to_hsv(*(float(c) for c in rgb))
	return np.clip(colorsys.hsv_to_rgb((h + dh) % 1.0, min(s * sat, 1.0), min(v * val, 1.0)), 0.0, 1.0)


def _defect_shapes(
	layout: Layout,
) -> list[tuple[str, tuple[float, float], tuple[float, ...], float, np.ndarray, float]]:
	"""The flat patches of a non-cross defect, in the face's (u, v) coordinates: (geom type, centre (u, v), half
	sizes in-plane, angle from u (rad), rgb, alpha). Boxes have half sizes (along, across); cylinders (radius,)."""
	rng = _defect_rng(layout)
	half = _face_half_extents(layout)
	brick = np.asarray(layout.brick_rgb, dtype=np.float64)
	kind = layout.defect_kind
	out: list = []
	if kind == "tape":
		# One strip of matte, light, semi-transparent tape running across the face.
		tape = (
			np.array([0.86, 0.84, 0.76]),
			np.array([0.90, 0.82, 0.55]),
			np.array([0.78, 0.78, 0.80]),
			np.array([0.80, 0.70, 0.55]),
		)[rng.integers(4)]
		rgb = _mix(tape, brick, rng.uniform(0.1, 0.3))
		w = float(rng.uniform(0.0012, min(0.0032, 0.3 * half.min())))
		ang = float(rng.uniform(-0.6, 0.6)) + (math.pi / 2 if rng.random() < 0.5 else 0.0)
		d = np.array([math.cos(ang), math.sin(ang)])
		s = np.array([-d[1], d[0]])
		c = rng.uniform(-0.3, 0.3, 2) * half
		ln = _fit_half_length(c, d, s, w, 0.97 * half)
		if ln < 2 * w:  # a corner-cut strip would be a stub: run it through the centre instead
			c = np.zeros(2)
			ln = _fit_half_length(c, d, s, w, 0.97 * half)
		out.append(("box", tuple(c), (ln, w), ang, rgb, float(rng.uniform(0.55, 0.8))))
	elif kind == "chip":
		# A broken-off corner or edge: a dark shadowed notch with a lighter fracture rim, at the face boundary.
		su = float(rng.choice([-1.0, 1.0]))
		sv = float(rng.choice([-1.0, 1.0]))
		a = float(rng.uniform(0.18, 0.32)) * half.min()  # notch half size
		b = a * float(rng.uniform(0.6, 1.0))
		if rng.random() < 0.7:  # corner
			c = np.array([su * (half[0] - a), sv * (half[1] - b)])
		else:  # along an edge
			c = np.array([su * (half[0] - a), float(rng.uniform(-0.4, 0.4)) * half[1]])
		dark = brick * float(rng.uniform(0.2, 0.35)) + 0.02
		out.append(("box", tuple(c), (a, b), 0.0, dark, 1.0))
		# fracture rim: lighter, rougher strips along the inner edges of the notch
		rim = _mix(brick * 0.6, np.ones(3) * 0.5, 0.3)
		rw = 0.25 * min(a, b)
		out.append(("box", (c[0] - su * (a + rw), c[1]), (rw, b), 0.0, rim, 1.0))
		out.append(("box", (c[0], c[1] - sv * (b + rw)), (a, rw), 0.0, rim, 1.0))
	elif kind == "scuff":
		# A few short, thin, low-contrast scratches and scrape marks, slightly whitened brick colour.
		for _ in range(int(rng.integers(3, 7))):
			rgb = _mix(brick, np.array([0.75, 0.75, 0.75]), rng.uniform(0.2, 0.4))
			w = float(rng.uniform(0.0003, 0.0007))
			ang = float(rng.uniform(0, math.pi))
			d = np.array([math.cos(ang), math.sin(ang)])
			s = np.array([-d[1], d[0]])
			c = rng.uniform(-0.6, 0.6, 2) * half
			ln = min(float(rng.uniform(0.0015, 0.004)), _fit_half_length(c, d, s, w, 0.95 * half))
			if ln > w:
				out.append(("box", tuple(c), (ln, w), ang, rgb, 1.0))
	elif kind == "discolor":
		# An irregular blotch of slightly different hue: a few overlapping discs.
		rgb = _hue_shift(
			brick,
			float(rng.choice([-1.0, 1.0]) * rng.uniform(0.03, 0.07)),
			float(rng.uniform(0.6, 0.9)),
			float(rng.uniform(0.85, 1.15)),
		)
		r0 = float(rng.uniform(0.3, 0.5)) * half.min()
		c0 = rng.uniform(-1.0, 1.0, 2) * np.maximum(half - r0, 0.0) * 0.6
		for k in range(int(rng.integers(2, 5))):
			r = r0 if k == 0 else r0 * float(rng.uniform(0.4, 0.7))
			c = c0 if k == 0 else c0 + rng.normal(0, 0.6 * r0, 2)
			c = np.clip(c, -(half - r), half - r)
			out.append(("cylinder", tuple(c), (r,), 0.0, rgb, 1.0))
	else:
		raise ValueError(f"unknown defect kind {kind!r}; expected one of {DEFECT_KINDS}")
	return out


def _defect_geoms(layout: Layout, stud_top_offset: float) -> str:
	"""Visual-only geoms ``defect_<k>`` of the brick's defect, ``layout.defect_kind`` on face ``layout.defect_face``:
	in geom group 0 for a defective brick and in the hidden group 5 for a normal one.

	``cross`` is the original dark cross (two strokes along the face diagonals), ``cross_var`` a randomised imperfect
	cross (:func:`cross_var_strokes`). The other kinds are flat patches (:func:`_defect_shapes`) whose shape and
	colour are a deterministic function of ``defect_seed``, ``defect_face``, ``defect_kind`` and ``brick_rgb``."""
	group = 0 if layout.defective else PROBE_GROUP
	lift = _DEFECT_LIFT_M + (stud_top_offset if FACES[layout.defect_face] == (2, 1) else 0.0)
	out = ""
	if layout.defect_kind == "cross":
		for k, (centre, axis, side, half) in enumerate(defect_strokes(layout.brick_size, layout.defect_face)):
			normal = np.cross(axis, side)
			pos = " ".join(f"{c:.5f}" for c in centre + lift * normal)
			xy = " ".join(f"{c:.5f}" for c in (*axis, *side))
			out += (
				f'<geom name="defect_{k}" type="box" size="{half:.4f} 0.0015 0.0003" pos="{pos}" xyaxes="{xy}" '
				f'rgba="0.05 0.05 0.08 1" group="{group}" contype="0" conaffinity="0" mass="0"/>'
			)
		return out
	centre, normal, u, v = face_frame(layout.brick_size, layout.defect_face)
	if layout.defect_kind == "cross_var":
		strokes, ink, _ = _cross_var_sample(layout)
		col = " ".join(f"{c:.2f}" for c in ink)
		for k, ((cu, cv), ang, half_length, half_width) in enumerate(strokes):
			d = math.cos(ang) * u + math.sin(ang) * v
			s = np.cross(normal, d)
			# the second stroke sits a hair higher so the two never z-fight where they cross
			pos = " ".join(f"{c:.5f}" for c in centre + cu * u + cv * v + (lift + 0.00002 * k) * normal)
			xy = " ".join(f"{c:.5f}" for c in (*d, *s))
			out += (
				f'<geom name="defect_{k}" type="box" size="{half_length:.5f} {half_width:.5f} '
				f'{_PATCH_HALF_T_M:.5f}" pos="{pos}" xyaxes="{xy}" rgba="{col} 1" group="{group}" '
				f'contype="0" conaffinity="0" mass="0"/>'
			)
		return out
	for k, (gtype, (cu, cv), halves, ang, rgb, alpha) in enumerate(_defect_shapes(layout)):
		d = math.cos(ang) * u + math.sin(ang) * v
		s = np.cross(normal, d)
		# later patches sit a hair higher so overlapping patches never z-fight
		pos = " ".join(f"{c:.5f}" for c in centre + cu * u + cv * v + (lift + 0.00002 * k) * normal)
		xy = " ".join(f"{c:.5f}" for c in (*d, *s))
		hs = " ".join(f"{h:.5f}" for h in halves)
		col = " ".join(f"{c:.3f}" for c in rgb)
		out += (
			f'<geom name="defect_{k}" type="{gtype}" size="{hs} {_PATCH_HALF_T_M:.5f}" pos="{pos}" xyaxes="{xy}" '
			f'rgba="{col} {alpha:.2f}" group="{group}" contype="0" conaffinity="0" mass="0"/>'
		)
	return out


FACE_PROBE_FILL = 0.8  # fraction of each face's extent a face probe patch covers


def face_probe_name(face: int) -> str:
	return f"face_probe_{face}"


def _face_probes(size: tuple[float, float, float], stud_top_offset: float) -> str:
	"""One thin patch per brick face (see :data:`~digital_twin_armigo.presentation.FACES`), in the hidden probe group:
	never in a colour image, no mass, no contacts. Segmentation renders count each face's visible pixels from them.
	They sit just under the defect cross (0.15 mm off the face against its 0.3 mm), so the cross keeps its pixels."""
	out = ""
	for face, (axis, _sign) in enumerate(FACES):
		centre, normal, u, v = face_frame(size, face)
		lift = 0.00015 + (stud_top_offset if FACES[face] == (2, 1) else 0.0)
		pos = " ".join(f"{c:.5f}" for c in centre + lift * normal)
		xy = " ".join(f"{c:.5f}" for c in (*u, *v))
		hu = FACE_PROBE_FILL * size[(axis + 1) % 3] / 2.0
		hv = FACE_PROBE_FILL * float(np.abs(v) @ np.asarray(size)) / 2.0
		out += (
			f'<geom name="{face_probe_name(face)}" type="box" size="{hu:.5f} {hv:.5f} 0.0001" pos="{pos}" '
			f'xyaxes="{xy}" rgba="1 0 1 1" group="{PROBE_GROUP}" contype="0" conaffinity="0" mass="0"/>'
		)
	return out


def _clutter_brick(
	x: float,
	y: float,
	yaw: float,
	r: float,
	g: float,
	b: float,
	use_visual_mesh: bool = True,
	dynamic: bool = False,
	mass_kg: float = 0.007,
	mu_slide: float = 0.35,
	idx: int = 0,
) -> str:
	"""Emit one clutter Duplo 2x2 brick body (mesh visual + group-3 collision box)."""
	rgba = f"{r:.2f} {g:.2f} {b:.2f} 1"
	mass_attr = f' mass="{mass_kg:.5f}" friction="{mu_slide:.4f} 0.02 0.002"' if dynamic else ""
	if use_visual_mesh:
		inner = (
			f'<geom type="mesh" mesh="duplo_2x2" contype="0" conaffinity="0" mass="0" rgba="{rgba}"/>'
			f'<geom type="box" size="{DUPLO_2X2[0] / 2:.4f} {DUPLO_2X2[1] / 2:.4f} {DUPLO_2X2[2] / 2:.4f}" '
			f'rgba="{rgba}"{mass_attr} group="3"/>'
		)
	else:
		inner = (
			f'<geom type="box" size="{DUPLO_2X2[0] / 2:.4f} {DUPLO_2X2[1] / 2:.4f} {DUPLO_2X2[2] / 2:.4f}" '
			f'rgba="{rgba}"{mass_attr}/>{_studs(DUPLO_2X2, rgba)}'
		)
	if dynamic:
		return (
			f'<body name="clutter_{idx}" pos="{x:.4f} {y:.4f} {DUPLO_2X2[2] / 2 + 0.004:.4f}" '
			f'euler="0 0 {yaw:.3f}"><freejoint name="clutter_j_{idx}"/>{inner}</body>'
		)
	return f'<body pos="{x:.4f} {y:.4f} {DUPLO_2X2[2] / 2 + 0.003:.4f}" euler="0 0 {yaw:.3f}">{inner}</body>'


def scene_xml(
	layout: Layout,
	use_visual_meshes: bool = True,
	physics: PhysicsSpec | None = None,
	dynamic_clutter: bool = False,
	mass_scale: float = 1.0,
	friction_scale: float = 1.0,
	face_probes: bool = False,
) -> str:
	"""MJCF for one episode; include it next to Menagerie's ``so101.xml``. ``face_probes`` adds the hidden per-face
	patches :meth:`~.cell.Cell.face_visibility` counts (no effect on physics or colour images)."""
	table = " ".join(f"{c:.3f}" for c in layout.table_rgb)
	tile2 = " ".join(f"{0.97 * c:.3f}" for c in layout.table_rgb)
	grout = " ".join(f"{0.72 * c:.3f}" for c in layout.table_rgb)
	ld = " ".join(f"{c:.3f}" for c in layout.light_dir)
	lv = layout.light_level
	distractors = "".join(
		f'<geom type="box" size="{sx:.3f} {sy:.3f} {sz:.3f}" pos="{x:.3f} {y:.3f} {sz:.3f}" '
		f'rgba="{0.2 + 0.6 * ((x * 37) % 1):.2f} {0.2 + 0.6 * ((y * 53) % 1):.2f} 0.5 1"/>'
		for x, y, sx, sy, sz, _ in layout.distractors
	)
	clutter_spec = physics.bricks["2x2"].scaled(mass_scale, friction_scale) if physics is not None else None
	clutter = "".join(
		_clutter_brick(
			x,
			y,
			yaw,
			r,
			g,
			b,
			use_visual_mesh=use_visual_meshes,
			dynamic=dynamic_clutter,
			mass_kg=clutter_spec.mass_kg if clutter_spec else 0.007,
			mu_slide=clutter_spec.dynamic_friction if clutter_spec else 0.35,
			idx=k,
		)
		for k, (x, y, yaw, r, g, b) in enumerate(layout.clutter)
	)
	mesh_assets = (_visual_mesh_assets() if use_visual_meshes else "") + _bowl_collision_mesh_assets()
	marker_assets = "".join(
		f'<texture name="m{i}" type="2d" file="_marker_{i}.png"/><material name="m{i}" texture="m{i}" specular="0"/>'
		for i in sorted({m[0] for m in layout.markers})
	)
	marker_geoms = "".join(
		f'<geom type="box" size="{s / 2 * 1.25:.4f} {s / 2 * 1.25:.4f} 0.0004" pos="{x:.4f} {y:.4f} {z + 0.0004:.4f}" '
		f'euler="0 0 {yaw:.3f}" material="m{i}" contype="0" conaffinity="0"/>'
		for i, x, y, z, yaw in layout.markers
		for s in (CELL_MARKERS[i],)
	)
	cams = ""
	if layout.top:
		cams += layout.top.mjcf("top", TOP_FULL)
	if layout.front:
		cams += layout.front.mjcf("front", (640, 480))
	return f"""<mujoco model="sorting_cell">
  <include file="so101.xml"/>
  <option timestep="{1.0 / FPS / SUBSTEPS:.6f}" cone="elliptic" impratio="10"/>
  <visual><headlight diffuse="{0.3 * lv:.2f} {0.3 * lv:.2f} {0.3 * lv:.2f}" ambient="{0.35 * lv:.2f} {0.35 * lv:.2f} {0.35 * lv:.2f}" specular="0 0 0"/>
    <global offwidth="640" offheight="480"/><quality shadowsize="1024"/></visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.75 0.75 0.78" rgb2="0.45 0.45 0.5" width="256" height="1536"/>
    <texture name="table" type="2d" builtin="checker" rgb1="{table}" rgb2="{tile2}" mark="edge" markrgb="{grout}" width="512" height="512"/>
    <material name="table" texture="table" texuniform="true" texrepeat="{1 / (2 * layout.tile_m):.3f} {1 / (2 * layout.tile_m):.3f}" reflectance="0"/>
    {mesh_assets}
    {marker_assets}
  </asset>
  <worldbody>
    <light pos="0.2 0 1.2" dir="{ld}" directional="true" diffuse="{0.6 * lv:.2f} {0.6 * lv:.2f} {0.6 * lv:.2f}" castshadow="true"/>
    <geom name="table" type="plane" size="3 3 0.05" material="table"/>
    {_bowl("green_bowl", layout.green, "0.25 0.65 0.3 1", use_visual_mesh=use_visual_meshes, torsional=physics is not None)}
    {_bowl("pink_bowl", layout.pink, "0.95 0.55 0.72 1", use_visual_mesh=use_visual_meshes, torsional=physics is not None)}
    {_brick(layout, use_visual_mesh=use_visual_meshes, physics=physics, mass_scale=mass_scale, friction_scale=friction_scale, face_probes=face_probes)}
    {distractors}
    {clutter}
    {marker_geoms}
    {cams}
  </worldbody>
</mujoco>
"""


def write_marker_textures(directory: Path, ids: set[int]) -> None:
	"""ArUco 4x4_50 images (black square plus a white margin of a quarter side) for the marker geoms."""
	import cv2

	dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
	for i in ids:
		path = directory / f"_marker_{i}.png"
		if not path.exists():
			img = cv2.aruco.generateImageMarker(dictionary, i, 192)
			cv2.imwrite(str(path), cv2.copyMakeBorder(img, 24, 24, 24, 24, cv2.BORDER_CONSTANT, value=255))
