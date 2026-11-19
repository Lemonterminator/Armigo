"""Episode layout of the sorting cell (bowls, brick, appearance, cameras) and its randomisation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..camera import CameraPose, look_from
from ..constants import BOWL_R, BRICK_COLOURS, DUPLO_2X2, DUPLO_2X4


@dataclass
class Layout:
	"""Bowl centres, brick pose and appearance for one episode (base frame, metres / degrees)."""

	green: tuple[float, float]
	pink: tuple[float, float]
	brick_xy: tuple[float, float]
	brick_yaw_deg: float
	defective: bool
	brick_rgb: tuple[float, float, float] = (0.85, 0.1, 0.1)
	table_rgb: tuple[float, float, float] = (0.55, 0.5, 0.45)
	light_dir: tuple[float, float, float] = (0.0, 0.0, -1.0)
	light_level: float = 0.7
	distractors: list[tuple[float, float, float, float, float, float]] = field(default_factory=list)
	brick_size: tuple[float, float, float] = (0.0318, 0.0318, 0.0192)
	defect_face: int = 4
	clutter: list[tuple[float, float, float, float, float, float]] = field(default_factory=list)
	markers: list[tuple[int, float, float, float, float]] = field(default_factory=list)
	tile_m: float = 0.30
	top: CameraPose | None = None
	front: CameraPose | None = None
	# Wrist camera mount error (dx, dy, dz in metres, rx, ry, rz in degrees) in the camera's own frame; None is the
	# designed mount. The cell renders the offset mount; agents that plan with the nominal mount see the designed one.
	wrist_cam_offset: tuple[float, float, float, float, float, float] | None = None
	defect_kind: str = "cross"  # one of scene.DEFECT_KINDS: cross, tape, chip, scuff, discolor, cross_var
	defect_seed: int = 0  # the defect's shape / colour is a deterministic function of this, the face and the kind


RELEASE_REACH_M = 0.34


def release_point(bowl: tuple[float, float]) -> tuple[float, float]:
	"""Where to open the jaws over a bowl: its centre, or the point on the near side within easy reach (at most
	0.6 bowl radii from the centre), as the operator does for the far bowl in the real recordings."""
	c = np.asarray(bowl, dtype=np.float64)
	r = float(np.hypot(*c))
	shift = min(max(r - RELEASE_REACH_M, 0.0), 0.6 * BOWL_R)
	p = c - shift * c / max(r, 1e-9)
	return float(p[0]), float(p[1])


def sample_layout(
	rng: np.random.Generator, nominal_top: CameraPose, defective: bool, prior: list[dict] | None = None
) -> Layout:
	"""A plausible random cell: bowls and brick inside reach, colours, light, clutter, markers, cameras.

	With ``prior`` (layouts measured on real first frames, see ``layout-prior``), the bowls start from a real
	episode's bowl centres and the brick from a real brick position, each jittered by a few centimetres, so the
	synthetic scenes cover the arrangements the real dataset shows. Without it, positions are uniform in reach.
	"""
	for _ in range(1000):
		if prior:
			ref = prior[rng.integers(len(prior))]
			bowls = [tuple(np.array(ref[c]) + rng.normal(0, 0.015, 2)) for c in ("green", "pink")]
			bricks = [p["brick"] for p in prior if p.get("brick")]
			brick = tuple(np.array(bricks[rng.integers(len(bricks))]) + rng.normal(0, 0.025, 2))
		else:
			r_b, a_b = rng.uniform(0.17, 0.25), rng.uniform(-0.6, 0.6)
			bowls = []
			for _ in range(2):
				r, a = rng.uniform(0.21, 0.27), rng.uniform(-1.1, 1.1)
				bowls.append((r * math.cos(a), r * math.sin(a)))
			brick = (r_b * math.cos(a_b), r_b * math.sin(a_b))
		pts = np.array([*bowls, brick])
		dist = [np.hypot(*(pts[i] - pts[j])) for i in range(3) for j in range(i + 1, 3)]
		reach = np.hypot(pts[:, 0], pts[:, 1])
		if (
			dist[0] > 2 * BOWL_R - 0.01
			and dist[1] > BOWL_R + 0.03
			and dist[2] > BOWL_R + 0.03
			and reach[2] < 0.40
			and np.hypot(*release_point(bowls[0])) < 0.40
			and np.hypot(*release_point(bowls[1])) < 0.40
			and reach.min() > 0.14
		):
			break
	else:
		raise RuntimeError("no layout satisfied the spacing and reach limits")
	size = DUPLO_2X4 if rng.random() < 0.2 else DUPLO_2X2
	distractors = []
	for _ in range(rng.integers(0, 3)):
		r, a = rng.uniform(0.33, 0.45), rng.uniform(-1.4, 1.4)
		distractors.append((r * math.cos(a), r * math.sin(a), *rng.uniform(0.01, 0.03, 3), 0.0))
	clutter = []
	for bowl in bowls:
		for _ in range(rng.integers(0, 5)):
			r, a = BOWL_R * 0.6 * math.sqrt(rng.random()), rng.uniform(0, 2 * math.pi)
			c = BRICK_COLOURS[rng.integers(len(BRICK_COLOURS))]
			clutter.append((bowl[0] + r * math.cos(a), bowl[1] + r * math.sin(a), rng.uniform(0, math.pi), *c))
	markers = []
	for i, bowl in zip((10, 11), bowls, strict=True):
		r, a = BOWL_R * rng.uniform(0.2, 0.55), rng.uniform(0, 2 * math.pi)
		markers.append((i, bowl[0] + r * math.cos(a), bowl[1] + r * math.sin(a), 0.003, rng.uniform(-0.4, 0.4)))
	for i in (16, 17):
		if rng.random() < 0.7:
			r, a = rng.uniform(0.2, 0.32), rng.uniform(-1.3, -0.5) * (1 if i == 16 else -1)
			markers.append((i, r * math.cos(a), r * math.sin(a), 0.0, rng.uniform(-0.3, 0.3)))
	top = CameraPose(
		nominal_top.x + rng.normal(0, 0.01),
		nominal_top.y + rng.normal(0, 0.01),
		nominal_top.height + rng.normal(0, 0.01),
		nominal_top.yaw_deg + rng.normal(0, 2.0),
		nominal_top.tilt_deg + rng.normal(0, 1.5),
		rng.normal(0, 1.0),
		nominal_top.focal_px * rng.uniform(0.98, 1.02),
	)
	front = look_from(
		(0.2, 0.0),
		rng.uniform(0.45, 0.65),
		rng.uniform(0.22, 0.38),
		rng.uniform(150, 210),
		focal_px=rng.uniform(480, 560),
	)
	t = rng.uniform(0.0, 1.0)
	return Layout(
		green=bowls[0],
		pink=bowls[1],
		brick_xy=brick,
		brick_yaw_deg=float(rng.uniform(-90, 90)),
		defective=defective,
		brick_rgb=BRICK_COLOURS[rng.integers(len(BRICK_COLOURS))],
		table_rgb=tuple(np.clip(0.66 + rng.normal(0, 0.05) + rng.normal(0, 0.015, 3), 0.3, 0.85)),
		tile_m=float(rng.uniform(0.25, 0.6)),
		light_dir=(float(rng.normal(0, 0.4)), float(rng.normal(0, 0.4)), -1.0),
		light_level=float(0.6 + 0.4 * t),
		distractors=distractors,
		brick_size=size,
		clutter=clutter,
		markers=markers,
		top=top,
		front=front,
	)
