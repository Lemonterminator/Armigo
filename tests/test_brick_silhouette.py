"""Known-cuboid silhouette fit (segmenter="silhouette") on tilted wrist views, shadows, touching markers, confusers."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from digital_twin_armigo.agent.brick_silhouette import cuboid_corners, fit_silhouette
from digital_twin_armigo.agent.perception import yaw_error_deg
from digital_twin_armigo.agent.wrist_localize import localize_from_wrist, wrist_k

SIZE = (0.0636, 0.0318, 0.0192)  # 2x4 Duplo
TOP_Z = 0.0192
TABLE = (120, 120, 118)
BLUE = ((40, 100, 210), (28, 72, 150), (20, 52, 110))  # top, lit side, dark side


def _camera(tilt_deg=33.0, x=0.30, y=-0.10, z=0.20):
	"""Camera above (x, y, z) looking down, tilted about its x axis by ``tilt_deg`` (the SO-101 wrist mount is ~33°)."""
	t = math.radians(tilt_deg)
	tilt = np.array([[1, 0, 0], [0, math.cos(t), -math.sin(t)], [0, math.sin(t), math.cos(t)]])
	tf = np.eye(4)
	tf[:3, :3] = np.diag([1.0, -1.0, -1.0]) @ tilt
	tf[:3, 3] = (x, y, z)
	return tf


def _axis_point(w2c, z=TOP_Z):
	"""Where the optical axis meets the plane z (so the brick is centred in view)."""
	o, d = w2c[:3, 3], w2c[:3, 2]
	return (o + (z - o[2]) / d[2] * d)[:2]


def _uv(k, w2c, pts):
	cw = np.linalg.inv(w2c)
	p = pts @ cw[:3, :3].T + cw[:3, 3]
	return (p[:, :2] / p[:, 2:3]) @ k[:2, :2].T + k[:2, 2]


def _table(shape=(240, 320)):
	img = np.empty((*shape, 3), np.uint8)
	img[:] = TABLE
	rng = np.random.default_rng(0)
	return np.clip(img.astype(int) + rng.integers(-4, 5, img.shape), 0, 255).astype(np.uint8)


def _draw_cuboid(img, k, w2c, xy, yaw, size=SIZE, top_z=TOP_Z, colours=BLUE):
	"""Painter's-order render of the visible faces of a cuboid (top face, then the side faces facing the camera)."""
	c = cuboid_corners(xy, yaw, size, top_z)  # 0-3 top, 4-7 bottom
	faces = [(0, 1, 2, 3)] + [(i, (i + 1) % 4, 4 + (i + 1) % 4, 4 + i) for i in range(4)]
	cam = w2c[:3, 3]
	centre = c.mean(axis=0)
	drawn = []
	for n, f in enumerate(faces):
		pts = c[list(f)]
		mid = pts.mean(axis=0)
		normal = mid - centre if n else np.array([0.0, 0.0, 1.0])
		if normal @ (cam - mid) <= 0:
			continue
		colour = colours[0] if n == 0 else colours[1 + (n % 2)]
		drawn.append((np.linalg.norm(cam - mid), pts, colour))
	for _, pts, colour in sorted(drawn, key=lambda t: -t[0]):
		cv2.fillPoly(img, [np.round(_uv(k, w2c, pts)).astype(np.int32)], colour)
	return img


def _setup(yaw_deg=20.0):
	k = wrist_k(60.0, (240, 320))
	w2c = _camera()
	xy = _axis_point(w2c) + [0.004, -0.003]
	return k, w2c, xy, math.radians(yaw_deg)


def _errors(fix, xy, yaw):
	return np.linalg.norm(fix.pos_xy - xy) * 1000, yaw_error_deg(fix.yaw, yaw, SIZE)


def test_fits_a_tilted_view_that_the_top_rectangle_gets_wrong():
	k, w2c, xy, yaw = _setup()
	img = _draw_cuboid(_table(), k, w2c, xy, yaw)
	prior_xy, prior_yaw = xy + [0.009, -0.006], yaw + math.radians(9)
	fix, why = fit_silhouette(img, k, w2c, prior_xy, prior_yaw, SIZE, TOP_Z)
	assert why is None
	e_xy, e_yaw = _errors(fix, xy, yaw)
	assert e_xy < 1.0 and e_yaw < 1.0
	assert fix.fill > 0.9 and fix.footprint_m == (SIZE[0], SIZE[1])
	lab, lab_why = localize_from_wrist(img, k, w2c, prior_xy, prior_yaw, SIZE, TOP_Z)
	# The side face merges with the top: the top-rectangle fit either rejects the footprint or is pulled off by it.
	assert lab_why == "BAD_FOOTPRINT" or _errors(lab, xy, yaw)[0] > 3 * max(e_xy, 0.5)
	via, _ = localize_from_wrist(img, k, w2c, prior_xy, prior_yaw, SIZE, TOP_Z, segmenter="silhouette")
	np.testing.assert_allclose(via.pos_xy, fix.pos_xy)


def test_shadow_touching_the_brick_is_background():
	k, w2c, xy, yaw = _setup()
	img = _table()
	shadow = np.array([[xy[0] - 0.02, xy[1] - 0.06, 0], [xy[0] + 0.06, xy[1] - 0.06, 0], [xy[0] + 0.06, xy[1], 0]])
	shadow = np.vstack([shadow, [xy[0] - 0.02, xy[1], 0]])
	dark = tuple(int(0.5 * v) for v in TABLE)
	cv2.fillPoly(img, [np.round(_uv(k, w2c, shadow)).astype(np.int32)], dark)
	_draw_cuboid(img, k, w2c, xy, yaw)
	fix, why = fit_silhouette(img, k, w2c, xy + [-0.008, 0.008], yaw - math.radians(10), SIZE, TOP_Z)
	assert why is None
	e_xy, e_yaw = _errors(fix, xy, yaw)
	assert e_xy < 1.0 and e_yaw < 1.0


def test_marker_card_touching_the_brick_does_not_pull_the_fit():
	k, w2c, xy, yaw = _setup(yaw_deg=0.0)
	img = _table()
	x0 = xy[0] + SIZE[0] / 2
	card = np.array(
		[[x0, xy[1] - 0.03, 0], [x0 + 0.05, xy[1] - 0.03, 0], [x0 + 0.05, xy[1] + 0.03, 0], [x0, xy[1] + 0.03, 0]]
	)
	cv2.fillPoly(img, [np.round(_uv(k, w2c, card)).astype(np.int32)], (245, 245, 245))
	inner = card.mean(axis=0) + (card - card.mean(axis=0)) * 0.6
	cv2.fillPoly(img, [np.round(_uv(k, w2c, inner)).astype(np.int32)], (10, 10, 10))
	_draw_cuboid(img, k, w2c, xy, yaw)
	fix, why = fit_silhouette(img, k, w2c, xy + [0.008, 0.005], yaw + math.radians(8), SIZE, TOP_Z)
	assert why is None
	e_xy, e_yaw = _errors(fix, xy, yaw)
	assert e_xy < 1.0 and e_yaw < 1.0


def test_a_prior_far_inside_the_gate_does_not_clip_the_brick():
	"""Under camera drift the prior is often 13-18 mm off; the 4 cm gate bounds the fitted centre, not the region the
	silhouette is counted in, so a whole brick in view must not come back CLIPPED (it did when the gate cropped it)."""
	k, w2c, xy, yaw = _setup()
	img = _draw_cuboid(_table(), k, w2c, xy, yaw)
	fix, why = fit_silhouette(img, k, w2c, xy + [0.016, -0.009], yaw + math.radians(4), SIZE, TOP_Z)
	assert why is None
	e_xy, e_yaw = _errors(fix, xy, yaw)
	assert e_xy < 1.0 and e_yaw < 1.0


def test_same_colour_object_of_another_size_is_not_accepted():
	k, w2c, xy, yaw = _setup()
	img = _draw_cuboid(_table(), k, w2c, xy, yaw, size=(0.0318, 0.0318, 0.0192))  # a 2x2 where a 2x4 is expected
	fix, why = fit_silhouette(img, k, w2c, xy, yaw, SIZE, TOP_Z)
	assert fix is None and why == "BAD_FIT"


def test_prefers_the_brick_nearest_the_prior_among_equal_fits():
	k, w2c, xy, yaw = _setup(yaw_deg=0.0)
	other = xy + [0.0, 0.034]
	img = _draw_cuboid(_draw_cuboid(_table(), k, w2c, other, yaw), k, w2c, xy, yaw)
	fix, why = fit_silhouette(img, k, w2c, xy + [0.003, -0.004], yaw, SIZE, TOP_Z, gate_m=0.06)
	if why is None:  # the two bricks touch in the band; the fit may also decline, but must not pick the far brick
		assert np.linalg.norm(fix.pos_xy - xy) < 0.002
	else:
		assert why == "BAD_FIT"


def test_rejects_steep_views_and_unknown_segmenters():
	k = wrist_k(60.0, (240, 320))
	w2c = _camera(tilt_deg=60.0)
	img = _table()
	assert fit_silhouette(img, k, w2c, _axis_point(w2c), 0.0, SIZE, TOP_Z) == (None, "TILTED")
	with pytest.raises(ValueError):
		localize_from_wrist(img, k, w2c, _axis_point(w2c), 0.0, SIZE, TOP_Z, segmenter="nope")


def test_mujoco_wrist_checkpoint_fixes_with_silhouette(tmp_path):
	pytest.importorskip("mujoco")
	from digital_twin_armigo.agent.student_rollout import ExpertTracer
	from digital_twin_armigo.agent.twin_mujoco import sample_case
	from digital_twin_armigo.agent.twin_record import EpisodeTrace
	from digital_twin_armigo.inspection_scorer import FixedScorer

	class _Scorer(FixedScorer):
		def close(self) -> None:
			pass

	trace = EpisodeTrace(tmp_path / "ep", video=False)
	twin = ExpertTracer(
		sample_case(700003, False, 0),
		_Scorer(lambda _cam, _img: -1.0, {"top": 0.0, "wrist_cam": 0.0}),
		trace,
		sensing="realistic",
		motion="w2s",
		segmenter="silhouette",
	)
	try:
		assert twin.prepare() is None
		twin._grasp()
	finally:
		twin.close()
	fixes = [e for e in trace.events if e["kind"] == "wrist_fix"]
	assert fixes and fixes[0]["found"] and fixes[0]["accepted"]
	assert fixes[0]["error_xy_mm"] < 1.5 and fixes[0]["error_yaw_deg"] < 1.5


def test_wrong_believed_height_is_corrected_by_the_table_hypothesis():
	k, w2c, xy, yaw = _setup()
	img = _draw_cuboid(_table(), k, w2c, xy, yaw)
	prior_xy, prior_yaw = xy + [0.006, 0.004], yaw - math.radians(6)
	high = TOP_Z + 0.010  # a top-camera estimate under drift can put the brick 5-12 mm too high
	off, _ = fit_silhouette(img, k, w2c, prior_xy, prior_yaw, SIZE, high, table_z=None, min_contrast=0.0)
	fix, why = fit_silhouette(img, k, w2c, prior_xy, prior_yaw, SIZE, high)
	assert why is None
	assert _errors(fix, xy, yaw)[0] < 1.0 and abs(fix.z - (TOP_Z - SIZE[2] / 2)) < 1e-6
	assert _errors(off, xy, yaw)[0] > 2.0  # without it the wrong height shifts the fit along the view direction
