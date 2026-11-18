import math

import numpy as np
import pytest

from digital_twin_armigo.agent import perception as pc
from digital_twin_armigo.agent.twin_mujoco import sample_case
from digital_twin_armigo.constants import SO101_MJCF_DIR, TOP_FULL
from digital_twin_armigo.mujoco_cell import handeye as he
from digital_twin_armigo.mujoco_cell.cell import Cell
from digital_twin_armigo.mujoco_cell.physics_expert import ExpertParams


def _locate(case):
	layout = sample_case(*case)
	cell = Cell(layout, SO101_MJCF_DIR)
	try:
		cell.set_joints(np.array([*ExpertParams().ready_q, 0.6]))
		r = cell.renderer
		r.enable_depth_rendering()
		r.update_scene(cell.data, "top")
		depth = r.render().astype(np.float64)
		r.disable_depth_rendering()
		k = he.intrinsics(layout.top, TOP_FULL)
		est = pc.locate_brick(depth, k, he.camera_truth(cell.model, cell.data, "top"), [layout.green, layout.pink])
		rot = cell.data.xmat[cell.brick_body].reshape(3, 3)
		return est, cell.brick_pos(), math.atan2(rot[1, 0], rot[0, 0]), layout.brick_size
	finally:
		cell.close()


@pytest.mark.parametrize("case", [(600001, False, 5), (600003, False, 5)])
def test_locates_a_clear_brick(case):
	est, truth, yaw, size = _locate(case)
	assert np.linalg.norm(est.pos[:2] - truth[:2]) < 0.002
	assert pc.yaw_error_deg(est.yaw, yaw, size) < 2.0
	assert est.size == size
	assert est.extended_mm == (0.0, 0.0)


def test_extends_a_brick_the_arm_partly_hides():
	est, truth, _, size = _locate((600004, False, 5))
	assert est.footprint_m[1] < size[1] - 0.005  # the arm cuts the top face
	assert est.extended_mm[1] > 5
	assert np.linalg.norm(est.pos[:2] - truth[:2]) < 0.002
