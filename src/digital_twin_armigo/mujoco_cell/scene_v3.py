"""Deterministic v3 scene factor sampling decoupled from ``sample_layout``."""

from __future__ import annotations

import colorsys
import itertools
import math
from dataclasses import dataclass

import numpy as np

from ..camera import project
from ..constants import BOWL_R, BRICK_COLOURS, TOP_CROP, TOP_FULL
from .layout import Layout, release_point
from .palette import HELD_OUT_PAIRS, PALETTE, hue_gap, jitter_rgb

SPLITS: tuple[str, ...] = ("train", "val", "ood-color", "ood-cam", "ood-lang", "ood-appearance", "test")

_CAM_HALF_WIDTHS: dict[str, float] = {
	"dx_m": 0.03,
	"dy_m": 0.03,
	"dz_m": 0.03,
	"dyaw_deg": 8.0,
	"dtilt_deg": 5.0,
	"droll_deg": 3.0,
	"dfocal_rel": 0.03,
}
_OOD_CAM_AXES: tuple[str, ...] = tuple(_CAM_HALF_WIDTHS.keys())
_TABLE_TEXTURES: tuple[str, ...] = ("checker", "plain", "noise")


@dataclass
class SceneV3:
	"""Decoupled visual, routing, camera, lighting, and phrasing factors for one v3 episode."""

	normal: str
	defect: str
	extra: str | None
	rgb: dict[str, tuple[float, float, float]]
	table_rgb: tuple[float, float, float]
	table_texture: str
	camera: dict[str, float]
	wrist: tuple[float, float, float, float, float, float]
	light_k: float
	variant: int


def draw_camera_v3(rng: np.random.Generator, split: str) -> dict[str, float]:
	"""Sample top-camera extrinsics/focal offsets within train range or ``ood-cam`` outer ring."""
	outer_key = str(rng.choice(_OOD_CAM_AXES)) if split == "ood-cam" else None
	cam: dict[str, float] = {}
	for key, hw in _CAM_HALF_WIDTHS.items():
		if key == outer_key:
			sign = -1.0 if rng.random() < 0.5 else 1.0
			cam[key] = sign * float(rng.uniform(1.0, 1.5)) * hw
		else:
			cam[key] = float(rng.uniform(-1.0, 1.0)) * hw
	return cam


def _sample_bowls(rng: np.random.Generator, split: str) -> tuple[str, str, str | None]:
	n_bowls = 2 if rng.random() < 0.70 else 3
	if split == "ood-color":
		pairs = sorted(HELD_OUT_PAIRS, key=lambda p: tuple(sorted(p)))
		pair = sorted(pairs[int(rng.integers(len(pairs)))])
		if rng.random() < 0.5:
			normal, defect = pair[0], pair[1]
		else:
			normal, defect = pair[1], pair[0]
		extra: str | None = None
		if n_bowls == 3:
			candidates = [
				c
				for c in PALETTE
				if c not in (normal, defect) and hue_gap(c, normal) >= 35.0 and hue_gap(c, defect) >= 35.0
			]
			extra = candidates[int(rng.integers(len(candidates)))]
		return normal, defect, extra

	palette_names = list(PALETTE)
	for _ in range(100):
		picked = [str(x) for x in rng.choice(palette_names, size=n_bowls, replace=False)]
		if all(
			frozenset((a, b)) not in HELD_OUT_PAIRS and hue_gap(a, b) >= 35.0
			for a, b in itertools.combinations(picked, 2)
		):
			return picked[0], picked[1], (picked[2] if n_bowls == 3 else None)
	raise RuntimeError("Failed to sample valid bowl colours after 100 attempts")


def _sample_table_rgb(
	rng: np.random.Generator,
	split: str,
	names: list[str],
	bowl_rgb: dict[str, tuple[float, float, float]] | None = None,
) -> tuple[float, float, float]:
	is_ood_app = split == "ood-appearance"
	sat_lo, sat_hi = (0.35 + 1e-6, 0.55 - 1e-6) if is_ood_app else (0.0, 0.35 - 1e-6)
	has_black = "black" in names
	has_white = "white" in names
	val_lo = 0.40 + 1e-6 if has_black else 0.35 + 1e-6
	val_hi = 0.68 - 1e-6 if has_white else 0.80 - 1e-6
	bowl_hues: list[float] = []
	for n in names:
		if PALETTE[n].hue_deg is None:
			continue
		bowl_hues.append(float(PALETTE[n].hue_deg))
		if bowl_rgb is not None and n in bowl_rgb:
			bowl_hues.append(colorsys.rgb_to_hsv(*bowl_rgb[n])[0] * 360.0)

	for _ in range(200):
		h_deg = float(rng.uniform(0.0, 360.0))
		sat = float(rng.uniform(sat_lo, sat_hi))
		val = float(rng.uniform(val_lo, val_hi))
		r, g, b = colorsys.hsv_to_rgb(h_deg / 360.0, sat, val)
		cand = (
			float(np.clip(r, 0.0, 1.0)),
			float(np.clip(g, 0.0, 1.0)),
			float(np.clip(b, 0.0, 1.0)),
		)
		h_rt, s_rt, v_rt = colorsys.rgb_to_hsv(*cand)
		if is_ood_app:
			if not (0.35 <= s_rt <= 0.55):
				continue
		elif s_rt > 0.35:
			continue
		if not (0.35 <= v_rt <= 0.80):
			continue
		if has_black and v_rt < 0.40:
			continue
		if has_white and v_rt > 0.68:
			continue
		if s_rt > 0.15:
			h_rt_deg = h_rt * 360.0
			if any(min(abs(h_rt_deg - bh) % 360.0, 360.0 - (abs(h_rt_deg - bh) % 360.0)) < 35.0 for bh in bowl_hues):
				continue
		return cand
	raise RuntimeError("Failed to sample valid table colour after 200 attempts")


def _sample_wrist(rng: np.random.Generator, split: str) -> tuple[float, float, float, float, float, float]:
	if split != "ood-cam":
		vals = [float(rng.uniform(-0.003, 0.003)) if i < 3 else float(rng.uniform(-3.0, 3.0)) for i in range(6)]
	else:
		outer_idx = int(rng.integers(6))
		vals = []
		for i in range(6):
			lo_inner, hi_outer = (0.003, 0.005) if i < 3 else (3.0, 5.0)
			if i == outer_idx:
				sign = -1.0 if rng.random() < 0.5 else 1.0
				vals.append(sign * float(rng.uniform(lo_inner, hi_outer)))
			else:
				vals.append(float(rng.uniform(-hi_outer, hi_outer)))
	return (vals[0], vals[1], vals[2], vals[3], vals[4], vals[5])


def sample_scene_v3(seed: int, split: str) -> SceneV3:
	"""Deterministically sample v3 scene factors for ``(seed, split)``."""
	if split not in SPLITS:
		raise ValueError(f"Unknown split {split!r}; expected one of {SPLITS}")

	rng = np.random.default_rng([int(seed), 0x5C3, SPLITS.index(split)])
	normal, defect, extra = _sample_bowls(rng, split)
	names = [normal, defect] + ([extra] if extra is not None else [])
	rgb = {name: jitter_rgb(name, rng) for name in names}
	table_rgb = _sample_table_rgb(rng, split, names, bowl_rgb=rgb)
	table_texture = str(rng.choice(_TABLE_TEXTURES))
	camera = draw_camera_v3(rng, split)
	wrist = _sample_wrist(rng, split)
	light_k = float(rng.uniform(3500.0, 6500.0))
	variant = 5 if split == "ood-lang" else int(rng.integers(0, 5))
	return SceneV3(
		normal=normal,
		defect=defect,
		extra=extra,
		rgb=rgb,
		table_rgb=table_rgb,
		table_texture=table_texture,
		camera=camera,
		wrist=wrist,
		light_k=light_k,
		variant=variant,
	)


def apply_scene_v3(layout: Layout, scene: SceneV3) -> Layout:
	"""Apply decoupled ``SceneV3`` appearance, extra bowl, distractors, and bowl-interior clutter to ``layout``."""
	layout.bowl_names = {"green": scene.normal, "pink": scene.defect}
	layout.bowl_rgb = {"green": scene.rgb[scene.normal], "pink": scene.rgb[scene.defect]}
	layout.table_rgb = scene.table_rgb
	layout.table_texture = scene.table_texture
	layout._scene_v3 = scene

	rng = np.random.default_rng([int(layout.defect_seed) & 0xFFFFFFFF, 0x5C3E])

	if scene.extra is not None:
		cx, cy, cw, ch = TOP_CROP
		u_lo, u_hi = cx + 0.05 * cw, cx + 0.95 * cw
		v_lo, v_hi = cy + 0.05 * ch, cy + 0.95 * ch
		for attempt in range(1500):
			if attempt < 200:
				r = float(rng.uniform(0.21, 0.27))
				a = float(rng.uniform(-1.1, 1.1))
			elif attempt < 1000:
				r = float(rng.uniform(0.21, 0.41))
				a = float(rng.uniform(-1.25, 1.25))
			else:
				r = float(rng.uniform(0.21, 0.42))
				a = float(rng.uniform(-1.45, 1.45))
			x, y = r * math.cos(a), r * math.sin(a)
			if layout.top is not None:
				uv, front = project(layout.top, np.array([[x, y, 0.0]]), TOP_FULL)
				in_view = bool(front[0] and u_lo <= uv[0, 0] <= u_hi and v_lo <= uv[0, 1] <= v_hi)
			else:
				in_view = True
			if (
				in_view
				and math.hypot(x - layout.green[0], y - layout.green[1]) > 2 * BOWL_R - 0.01
				and math.hypot(x - layout.pink[0], y - layout.pink[1]) > 2 * BOWL_R - 0.01
				and math.hypot(x - layout.brick_xy[0], y - layout.brick_xy[1]) > BOWL_R + 0.03
				and math.hypot(*release_point((x, y))) < 0.40
				and math.hypot(x, y) > 0.14
			):
				layout.extra_bowls = [(scene.extra, float(x), float(y), scene.rgb[scene.extra])]
				break
		else:
			raise RuntimeError("no extra bowl position satisfied the spacing and reach limits")
	else:
		layout.extra_bowls = []

	distractors: list[tuple[float, float, float, float, float, float]] = []
	for _ in range(int(rng.integers(0, 4))):
		r = float(rng.uniform(0.33, 0.45))
		a = float(rng.uniform(-1.4, 1.4))
		sx, sy, sz = (float(v) for v in rng.uniform(0.01, 0.03, 3))
		distractors.append((r * math.cos(a), r * math.sin(a), sx, sy, sz, 0.0))
	layout.distractors = distractors

	clutter: list[tuple[float, float, float, float, float, float]] = []
	clutter_palette = list(BRICK_COLOURS) + list(scene.rgb.values())
	bowls_xy = [layout.green, layout.pink, *((x, y) for _, x, y, _ in layout.extra_bowls)]
	for bx, by in bowls_xy:
		n_c = int(rng.integers(0, 5))
		bowl_pts: list[tuple[float, float]] = []
		for _ in range(n_c):
			for _attempt in range(40):
				r = BOWL_R * 0.55 * math.sqrt(float(rng.random()))
				ang = float(rng.uniform(0.0, 2.0 * math.pi))
				cx = float(bx + r * math.cos(ang))
				cy = float(by + r * math.sin(ang))
				if all(math.hypot(cx - px, cy - py) >= 0.048 for px, py in bowl_pts):
					bowl_pts.append((cx, cy))
					c = clutter_palette[int(rng.integers(len(clutter_palette)))]
					yaw = float(rng.uniform(0.0, math.pi))
					clutter.append((cx, cy, yaw, float(c[0]), float(c[1]), float(c[2])))
					break
	layout.clutter = clutter
	return layout
