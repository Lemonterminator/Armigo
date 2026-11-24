"""Eight-colour bowl palette, held-out pairs, and HSV-bounded colour jitter for v3 scenes."""

from __future__ import annotations

import colorsys
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Colour:
	"""Named bowl colour with canonical linear-ish sRGB centre and optional hue angle in degrees."""

	name: str
	rgb: tuple[float, float, float]
	hue_deg: float | None


def _chromatic(name: str, rgb: tuple[float, float, float]) -> Colour:
	h, _, _ = colorsys.rgb_to_hsv(*rgb)
	return Colour(name=name, rgb=rgb, hue_deg=h * 360.0)


PALETTE: dict[str, Colour] = {
	"green": _chromatic("green", (0.25, 0.65, 0.30)),
	"pink": _chromatic("pink", (0.95, 0.55, 0.72)),
	"blue": _chromatic("blue", (0.15, 0.48, 0.90)),
	"yellow": _chromatic("yellow", (0.95, 0.92, 0.14)),
	"orange": _chromatic("orange", (0.95, 0.40, 0.11)),
	"purple": _chromatic("purple", (0.58, 0.25, 0.85)),
	"white": Colour(name="white", rgb=(0.97, 0.97, 0.97), hue_deg=None),
	"black": Colour(name="black", rgb=(0.09, 0.09, 0.09), hue_deg=None),
}

HELD_OUT_PAIRS: frozenset[frozenset[str]] = frozenset(
	{
		frozenset({"blue", "purple"}),
		frozenset({"orange", "yellow"}),
		frozenset({"white", "pink"}),
		frozenset({"green", "black"}),
	}
)


def hue_gap(a: str, b: str) -> float:
	"""Angular hue separation in degrees between two palette colours (180 if either is achromatic)."""
	if a == b:
		return 0.0
	ha = PALETTE[a].hue_deg
	hb = PALETTE[b].hue_deg
	if ha is None or hb is None:
		return 180.0
	d = abs(ha - hb) % 360.0
	return min(d, 360.0 - d)


def jitter_rgb(colour: Colour | str, rng: np.random.Generator) -> tuple[float, float, float]:
	"""Sample a slightly jittered RGB around ``colour`` keeping colour-word identity intact."""
	c = PALETTE[colour] if isinstance(colour, str) else colour
	if c.name == "white":
		v = c.rgb[0] * float(rng.uniform(0.92, 1.0))
		s = float(rng.uniform(0.0, 0.03))
		h = 0.0
	elif c.name == "black":
		v = c.rgb[0] * float(rng.uniform(0.90, 1.10))
		s = float(rng.uniform(0.0, 0.05))
		h = 0.0
	else:
		h, s, v = colorsys.rgb_to_hsv(*c.rgb)
		h_deg = (h * 360.0 + float(rng.uniform(-6.0, 6.0))) % 360.0
		h = h_deg / 360.0
		s = float(np.clip(s * float(rng.uniform(0.90, 1.10)), 0.20, 1.0))
		v = float(np.clip(v * float(rng.uniform(0.90, 1.10)), 0.20, 1.0))
	r, g, b = colorsys.hsv_to_rgb(h, s, v)
	return (
		float(np.clip(r, 0.0, 1.0)),
		float(np.clip(g, 0.0, 1.0)),
		float(np.clip(b, 0.0, 1.0)),
	)
