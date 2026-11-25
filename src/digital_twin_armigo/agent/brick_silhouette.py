"""Known-cuboid silhouette fit for wrist-camera brick re-localisation (``segmenter="silhouette"``).

The brick lies on the table, so its pose has three unknowns (x, y, yaw); its size and the camera (designed wrist
mount, intrinsics) are known. A tilted wrist camera sees the top face *and* one or two side faces, which is why the
top-rectangle fit of the ``lab`` segmenter rejects most views (``BAD_FOOTPRINT``). Here the whole cuboid is projected
(its silhouette is the convex hull of the eight projected corners) and (x, y, yaw) are searched to maximise a region
contrast: the foreground fraction inside the silhouette minus the foreground fraction in a thin band just outside it.

- Foreground is anything that is not the table or a shadow on it (:func:`foreground_mask`): a cast shadow keeps the
  table's chromaticity and only scales it, so it is background here (the arm's shadow otherwise merges with the brick).
- Region contrast tolerates objects touching the brick (a marker, the gripper) because they cover only part of the
  band, and it is near zero for a pose inside a large foreground region. A same-colour object of another size does
  not fit the known cuboid.
- Counts are taken only inside the valid region (inside the image and inside the search gate), so a brick cut by the
  image border or the gate circle is fitted from what is visible.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .wrist_localize import WristFix, _project

__all__ = ["cuboid_corners", "fit_silhouette", "foreground_mask"]

_SHIFT = 4  # fillConvexPoly sub-pixel bits
_STEPS = ((0.004, 6.0), (0.002, 3.0), (0.001, 1.5), (0.0005, 0.75), (0.00025, 0.4))
_BAND_M = 0.004  # outer band width around the cuboid, metres
_MAX_COMPONENTS = 4


def cuboid_corners(xy, yaw: float, size, top_z: float, *, grow: float = 0.0) -> np.ndarray:
	"""(8, 3) world corners of a brick of full ``size`` (x, y, z) centred at ``xy`` with its top face at ``top_z``.

	``grow`` enlarges the footprint by that many metres on every side and raises the top by the same amount."""
	c, s = math.cos(yaw), math.sin(yaw)
	hx, hy = size[0] / 2.0 + grow, size[1] / 2.0 + grow
	pts = []
	for z in (top_z + grow, top_z - size[2]):
		for a, b in ((hx, hy), (-hx, hy), (-hx, -hy), (hx, -hy)):
			pts.append((xy[0] + c * a - s * b, xy[1] + s * a + c * b, z))
	return np.asarray(pts, dtype=np.float64)


def _table_colour(img: np.ndarray, ring: np.ndarray) -> np.ndarray | None:
	"""Lit table colour from a ring: the modal chromaticity (bowls, markers and bricks in the ring are minorities),
	then the median of the brighter half of the pixels with that chromaticity (the darker ones are in shadow)."""
	px = img[ring]
	s = px.sum(axis=1)
	ok = s > 30.0
	px, s = px[ok], s[ok]
	if len(px) < 20:
		return None
	chroma = px[:, :2] / s[:, None]
	bins = np.clip((chroma * 48).astype(int), 0, 47)
	hist = np.zeros((48, 48), np.int64)
	np.add.at(hist, (bins[:, 0], bins[:, 1]), 1)
	hist = cv2.blur(hist.astype(np.float32), (3, 3))
	br, bg = np.unravel_index(int(np.argmax(hist)), hist.shape)
	near = (np.abs(bins[:, 0] - br) <= 1) & (np.abs(bins[:, 1] - bg) <= 1)
	sel, ssel = px[near], s[near]
	if len(sel) < 10:
		return None
	return np.median(sel[ssel >= np.median(ssel)], axis=0)


def foreground_mask(rgb, centre, gate_px, *, resid_min=22.0, alpha_lo=0.3, alpha_hi=1.08):
	"""uint8 mask of non-table pixels inside the gate, treating darker copies of the table colour as background.

	A pixel ``p`` is background when it is within ``resid_min`` (8-bit RGB units) of ``alpha * table`` for some
	``alpha`` in ``[alpha_lo, alpha_hi]``. Returns ``None`` if the table colour cannot be estimated."""
	img = rgb.astype(np.float32)
	h, w = rgb.shape[:2]
	yy, xx = np.mgrid[0:h, 0:w]
	r = np.hypot(xx - centre[0], yy - centre[1])
	ring = (r > 1.6 * gate_px) & (r < 2.4 * gate_px)
	if ring.sum() < 50:
		ring = r > 1.6 * gate_px
	if not ring.any():
		return None
	bg = _table_colour(img, ring)
	if bg is None:
		return None
	bb = float(bg @ bg)
	alpha = np.clip((img @ bg) / bb, alpha_lo, alpha_hi)
	resid = np.linalg.norm(img - alpha[..., None] * bg, axis=-1)
	mask = ((resid > resid_min) & (r < gate_px)).astype(np.uint8)
	return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


class _Scorer:
	"""Region contrast of a cuboid pose against the foreground, inside the valid region, on a (scaled) crop."""

	def __init__(self, fg, valid, k, cam_from_world, size, top_z, max_side=160):
		ys, xs = np.nonzero(valid)
		self.x0, self.y0 = int(xs.min()), int(ys.min())
		x1, y1 = int(xs.max()) + 1, int(ys.max()) + 1
		self.scale = min(1.0, max_side / max(x1 - self.x0, y1 - self.y0))
		w = max(1, round((x1 - self.x0) * self.scale))
		h = max(1, round((y1 - self.y0) * self.scale))
		crop = (slice(self.y0, y1), slice(self.x0, x1))
		self.valid = cv2.resize(valid[crop].astype(np.uint8), (w, h), interpolation=cv2.INTER_AREA) > 0
		self.fg = (
			cv2.resize(fg[crop].astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_AREA) >= 128
		) & self.valid
		self.k, self.cw, self.size, self.top_z = k, cam_from_world, size, top_z
		self.inner = np.zeros((h, w), np.uint8)
		self.outer = np.zeros((h, w), np.uint8)

	def _fill(self, buf, corners) -> float | None:
		p = corners @ self.cw[:3, :3].T + self.cw[:3, 3]
		if np.any(p[:, 2] <= 1e-4):
			return None
		uv = (p[:, :2] / p[:, 2:3]) @ self.k[:2, :2].T + self.k[:2, 2]
		uv = (uv - (self.x0, self.y0)) * self.scale
		hull = cv2.convexHull(np.round(uv * (1 << _SHIFT)).astype(np.int32))
		buf[:] = 0
		cv2.fillConvexPoly(buf, hull, 1, lineType=cv2.LINE_8, shift=_SHIFT)
		return float(cv2.contourArea(hull.astype(np.float32))) / (1 << (2 * _SHIFT))

	def __call__(self, xy, yaw, top_z=None) -> tuple[float, float, float]:
		"""(region contrast, foreground fraction inside, fraction of the full silhouette inside the valid region)."""
		top = self.top_z if top_z is None else top_z
		full = self._fill(self.inner, cuboid_corners(xy, yaw, self.size, top))
		if full is None or self._fill(self.outer, cuboid_corners(xy, yaw, self.size, top, grow=_BAND_M)) is None:
			return -1.0, 0.0, 0.0
		inner = self.inner.astype(bool) & self.valid
		band = self.outer.astype(bool) & ~self.inner.astype(bool) & self.valid
		n_in, n_band = np.count_nonzero(inner), np.count_nonzero(band)
		if n_in == 0:
			return -1.0, 0.0, 0.0
		cov = np.count_nonzero(inner & self.fg) / n_in
		leak = np.count_nonzero(band & self.fg) / n_band if n_band else 0.0
		return cov - leak, cov, min(n_in / max(full, 1.0), 1.0)


def _colour_masks(rgb, fg, *, min_px, chroma_tol=0.035, grey_chroma=0.03, dark_sum=60.0, l_tol=40.0, max_colours=3):
	"""Split the foreground into its dominant colours: one mask per peak of the rg-chromaticity histogram
	(``r / (r + g + b)``, ``g / (r + g + b)``), a neutral peak (grey, black, white, or too dark to have a reliable
	chromaticity) further split by brightness. Chromaticity does not change when a face is darker, so a coloured
	brick's top and side faces stay in one mask while a black-and-white marker card or a differently coloured object
	touching it falls into another."""
	img = rgb.astype(np.float32)
	s = img.sum(axis=-1)
	chroma = img[..., :2] / np.maximum(s, 1.0)[..., None]
	sel = fg > 0
	neutral = (np.hypot(chroma[..., 0] - 1 / 3, chroma[..., 1] - 1 / 3) < grey_chroma) | (s < dark_sum)
	masks = []
	coloured = sel & ~neutral
	hist = np.zeros((64, 64), np.float32)
	c = chroma[coloured]
	np.add.at(hist, (np.clip((c[:, 0] * 64).astype(int), 0, 63), np.clip((c[:, 1] * 64).astype(int), 0, 63)), 1.0)
	hist = cv2.GaussianBlur(hist, (5, 5), 1.0)
	r = math.ceil(chroma_tol * 64)
	for _ in range(max_colours):
		pr, pg = np.unravel_index(int(np.argmax(hist)), hist.shape)
		ref = ((pr + 0.5) / 64, (pg + 0.5) / 64)
		m = coloured & (np.hypot(chroma[..., 0] - ref[0], chroma[..., 1] - ref[1]) < chroma_tol)
		if np.count_nonzero(m) < min_px:
			break
		masks.append(m.astype(np.uint8))
		hist[max(0, pr - r) : pr + r + 1, max(0, pg - r) : pg + r + 1] = 0
	grey = sel & neutral
	lum = s[grey] / 3.0
	lhist, edges = np.histogram(lum, bins=26, range=(0, 260))
	centres = 0.5 * (edges[:-1] + edges[1:])
	for _ in range(2):
		j = int(np.argmax(lhist))
		if lhist[j] < min_px / 2:
			break
		m = grey & (np.abs(s / 3.0 - centres[j]) < l_tol)
		if np.count_nonzero(m) >= min_px:
			masks.append(m.astype(np.uint8))
		lhist[np.abs(centres - centres[j]) < l_tol] = 0
	return masks


def _descend(score, xy, yaw, top, z_lo, z_hi):
	"""Coordinate ascent of the contrast over (x, y, yaw) and, when ``z_lo < z_hi``, the top-face height."""
	best = score(xy, yaw, top)[0]
	free_z = z_hi > z_lo
	for d_m, d_deg in _STEPS:
		d_rad = math.radians(d_deg)
		moves = [(d_m, 0, 0, 0), (-d_m, 0, 0, 0), (0, d_m, 0, 0), (0, -d_m, 0, 0), (0, 0, d_rad, 0), (0, 0, -d_rad, 0)]
		if free_z:
			moves += [(0, 0, 0, d_m), (0, 0, 0, -d_m)]
		improved = True
		while improved:
			improved = False
			for dx, dy, dt, dz in moves:
				cand_top = top + dz
				if not z_lo <= cand_top <= z_hi:
					continue
				cand_xy = (xy[0] + dx, xy[1] + dy)
				val = score(cand_xy, yaw + dt, cand_top)[0]
				if val > best + 1e-6:
					best, xy, yaw, top, improved = val, cand_xy, yaw + dt, cand_top, True
	return best, np.asarray(xy, dtype=np.float64), yaw, top


def _plane_point(k, world_from_cam, u, v, z):
	d = world_from_cam[:3, :3] @ (np.linalg.inv(k) @ np.array([u, v, 1.0]))
	if abs(d[2]) < 1e-9:
		return None
	o = world_from_cam[:3, 3]
	return (o + (z - o[2]) / d[2] * d)[:2]


def _starts(fg, k, world_from_cam, size, top_z, prior_xy, prior_yaw, min_area):
	"""Start poses: each of the largest foreground components (centroid on the mid-height plane, its principal axis,
	the prior yaw and their quarter turns), plus the prior pose."""
	starts = [(np.asarray(prior_xy, dtype=np.float64), float(prior_yaw))]
	n, labels, stats, _ = cv2.connectedComponentsWithStats(fg)
	order = sorted(range(1, n), key=lambda i: -stats[i, cv2.CC_STAT_AREA])[:_MAX_COMPONENTS]
	z_mid = top_z - size[2] / 2.0
	for i in order:
		if stats[i, cv2.CC_STAT_AREA] < min_area:
			continue
		m = cv2.moments((labels == i).astype(np.uint8), binaryImage=True)
		cu, cv = m["m10"] / m["m00"], m["m01"] / m["m00"]
		c_xy = _plane_point(k, world_from_cam, cu, cv, z_mid)
		if c_xy is None:
			continue
		theta = 0.5 * math.atan2(2 * m["mu11"], m["mu20"] - m["mu02"])
		q = _plane_point(k, world_from_cam, cu + 10 * math.cos(theta), cv + 10 * math.sin(theta), z_mid)
		yaws = [prior_yaw, prior_yaw + math.pi / 2]
		if q is not None:
			img_yaw = math.atan2(q[1] - c_xy[1], q[0] - c_xy[0])
			yaws += [img_yaw, img_yaw + math.pi / 2]
		starts += [(c_xy, float(y)) for y in yaws]
	return starts


def fit_silhouette(
	rgb,
	k,
	world_from_cam,
	prior_xy,
	prior_yaw,
	size,
	top_z,
	*,
	gate_m: float = 0.04,
	max_tilt_deg: float = 50.0,
	min_contrast: float = 0.8,
	min_visible: float = 0.9,
	tie: float = 0.05,
	table_z: float | None = 0.0,
	z_range_m: float = 0.0,
):
	"""Return ``(WristFix, None)`` or ``(None, reason)``; reasons TILTED / NO_BLOB / CLIPPED / BAD_FIT / OUTSIDE_GATE.

	``WristFix.fill`` carries the region contrast of the chosen pose; ``footprint_m`` is the known footprint. Among
	poses whose contrast is within ``tie`` of the best, the one nearest the prior is taken.

	Height: a wrong believed height shifts the fitted position along the view direction (about ``tan(view angle)``
	times the height error), and a top-camera estimate under camera drift can be 5-12 mm off. So each start is fitted
	at the believed ``top_z`` and, when ``table_z`` is given, also at ``table_z + size[2]`` (the brick resting flat on
	the table); the better contrast wins. ``z_range_m > 0`` additionally searches the height continuously within
	``± z_range_m`` of each hypothesis (off by default: with a correct height it trades scale against depth and loses
	accuracy). ``WristFix.z`` is the chosen centre height."""
	axis = world_from_cam[:3, 2]
	tilt = math.degrees(math.acos(float(np.clip(-axis[2], -1.0, 1.0))))
	if tilt > max_tilt_deg:
		return None, "TILTED"
	centre, depth = _project(k, world_from_cam, np.array([prior_xy[0], prior_xy[1], top_z]))
	if depth <= 1e-4:
		return None, "NO_BLOB"
	gate_px = gate_m * k[0, 0] / depth
	fg = foreground_mask(rgb, centre, gate_px)
	if fg is None or not fg.any():
		return None, "NO_BLOB"
	h, w = rgb.shape[:2]
	yy, xx = np.mgrid[0:h, 0:w]
	valid = np.hypot(xx - centre[0], yy - centre[1]) < gate_px
	cam_from_world = np.linalg.inv(world_from_cam)
	score = _Scorer(fg.astype(bool), valid, k, cam_from_world, size, top_z)
	# Smallest region worth a start: a quarter of the brick's top face at this depth.
	min_area = 0.25 * size[0] * size[1] * (k[0, 0] / depth) ** 2
	tops = [float(top_z)]
	if table_z is not None and abs(table_z + size[2] - top_z) > 0.001:
		tops.append(float(table_z + size[2]))
	fits = []
	# All foreground, then each dominant colour on its own: an object of another colour touching the brick (a marker
	# card, the gripper) then lies outside the brick's mask and no longer counts as leak in the band.
	for mask in [fg, *_colour_masks(rgb, fg * valid, min_px=min_area)]:
		sc = score if mask is fg else _Scorer(mask.astype(bool), valid, k, cam_from_world, size, top_z)
		for xy0, yaw0 in _starts(mask, k, world_from_cam, size, top_z, prior_xy, prior_yaw, min_area):
			for top0 in tops:
				fits.append(
					_descend(sc, (float(xy0[0]), float(xy0[1])), float(yaw0), top0, top0 - z_range_m, top0 + z_range_m)
				)
	best = max(f[0] for f in fits)
	prior = np.asarray(prior_xy, dtype=np.float64)
	contrast, xy, yaw, top = min(
		(f for f in fits if f[0] >= best - tie), key=lambda f: float(np.linalg.norm(f[1] - prior))
	)
	_, _, visible = score(xy, yaw, top)
	if visible < min_visible:
		return None, "CLIPPED"
	if contrast < min_contrast:
		return None, "BAD_FIT"
	if np.linalg.norm(xy - prior) > gate_m:
		return None, "OUTSIDE_GATE"
	period = math.pi / 2 if abs(size[0] - size[1]) < 1e-6 else math.pi
	yaw += period * round((prior_yaw - yaw) / period)
	footprint = tuple(sorted((float(size[0]), float(size[1])), reverse=True))
	fix = WristFix(
		xy, float(yaw), int(np.count_nonzero(fg)), float(contrast), footprint, tilt, z=float(top - size[2] / 2)
	)
	return fix, None
