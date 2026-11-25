import math

import numpy as np

from digital_twin_armigo.perception_nn.labels import brick_corners_world, corner_labels


def test_corners_order_and_extent():
	rot = np.array([[math.cos(0.3), -math.sin(0.3), 0], [math.sin(0.3), math.cos(0.3), 0], [0, 0, 1]])
	c = brick_corners_world(np.array([0.3, 0.0, 0.01]), rot, (0.032, 0.016, 0.0192))
	assert c.shape == (8, 3)
	assert np.allclose(c[:4, 2], 0.01 + 0.0096) and np.allclose(c[4:, 2], 0.01 - 0.0096)
	top = c[:4, :2] - c[:4, :2].mean(axis=0)
	cross = top[0, 0] * top[1, 1] - top[0, 1] * top[1, 0]
	assert cross > 0  # counter-clockwise seen from above
	assert np.isclose(np.linalg.norm(c[0] - c[1]), 0.032) or np.isclose(np.linalg.norm(c[1] - c[2]), 0.032)


def test_corner_visibility_uses_depth():
	k = np.array([[400.0, 0, 319.5], [0, 400.0, 239.5], [0, 0, 1]])
	w2c = np.eye(4)
	w2c[:3, :3] = np.diag([1.0, -1.0, -1.0])
	w2c[:3, 3] = (0.3, 0.0, 0.3)
	c = brick_corners_world(np.array([0.3, 0.0, 0.0096]), np.eye(3), (0.032, 0.016, 0.0192))
	depth = np.full((480, 640), 0.3 - 0.0192)  # a plane at the top face: bottom corners are hidden
	lab = corner_labels(c, k, w2c, depth)
	assert lab[:4, 2].all() and not lab[4:, 2].any()
	assert np.allclose(
		lab[0, :2], [319.5 + 400 * 0.016 / (0.3 - 0.0192), 239.5 - 400 * 0.008 / (0.3 - 0.0192)], atol=0.5
	)


def test_face_class_map_matches_face_visibility():
	"""Pixel counts per class agree with Cell.face_visibility on a rendered v3 case (CPU MuJoCo)."""
	from digital_twin_armigo.perception_nn.labels import _demo_cell

	cell = _demo_cell(3_000_101)
	try:
		vis = cell.face_visibility("top")
		cls = cell.face_class_map("top")
		assert cls.dtype == np.int8
		assert (cls == 0).sum() == vis["faces"][4]
		assert (cls == 1).sum() == vis["faces"][5]
		assert (cls == 2).sum() == vis["faces"][:4].sum()
	finally:
		cell.close()
