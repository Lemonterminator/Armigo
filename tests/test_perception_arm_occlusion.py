"""The arm at the ready pose hides part of a brick near the base from the top camera (round-2 grasp failures,
seed 4100266: a 2x4 brick read as a 2x2 14 mm off). ``arm_occludes_blob`` flags it; at rest the fit is clean."""

import math

import numpy as np
import pytest

from digital_twin_armigo.agent import perception as pc
from digital_twin_armigo.agent.student_rollout import ExpertTracer
from digital_twin_armigo.agent.twin_mujoco import INITIAL_Q, sample_case
from digital_twin_armigo.constants import SO101_MJCF_DIR, TOP_FULL
from digital_twin_armigo.mujoco_cell import handeye as he
from digital_twin_armigo.mujoco_cell.cell import Cell
from digital_twin_armigo.mujoco_cell.physics_expert import ExpertParams

CASE = (4100266, False, 5, {"scene": "v3", "cameras": "fixed", "split": "train"})


def _top(q):
	layout = sample_case(*CASE)
	cell = Cell(layout, SO101_MJCF_DIR)
	try:
		cell.set_joints(q)
		r = cell.renderer
		r.enable_depth_rendering()
		r.update_scene(cell.data, "top")
		depth = r.render().astype(np.float64)
		r.disable_depth_rendering()
		k = he.intrinsics(layout.top, TOP_FULL)
		cam = he.camera_truth(cell.model, cell.data, "top")
		rot = cell.data.xmat[cell.brick_body].reshape(3, 3)
		return depth, k, cam, [layout.green, layout.pink], cell.brick_pos(), math.atan2(rot[1, 0], rot[0, 0]), layout.brick_size
	finally:
		cell.close()


def test_ready_pose_hides_the_brick_and_is_flagged():
	pytest.importorskip("mujoco")
	pytest.importorskip("cv2")
	depth, k, cam, bowls, truth, _, size = _top(np.array([*ExpertParams().ready_q, 0.6]))
	est = pc.locate_brick(depth, k, cam, bowls)
	assert est.size != size  # the visible part of the 2x4 is under the 2x4 length threshold
	assert np.linalg.norm(est.pos[:2] - truth[:2]) > 0.008
	assert pc.arm_occludes_blob(depth, k, cam, bowls)


def test_rest_pose_reads_the_brick_cleanly_and_is_not_flagged():
	pytest.importorskip("mujoco")
	pytest.importorskip("cv2")
	depth, k, cam, bowls, truth, yaw, size = _top(np.asarray(INITIAL_Q, dtype=np.float64))
	est = pc.locate_brick(depth, k, cam, bowls)
	assert est.size == size
	assert np.linalg.norm(est.pos[:2] - truth[:2]) < 0.004
	assert pc.yaw_error_deg(est.yaw, yaw, size) < 2.0
	assert not pc.arm_occludes_blob(depth, k, cam, bowls)


def test_clear_view_motion_is_a_prepare_segment():
	def clear_view():
		yield

	tracer = ExpertTracer.__new__(ExpertTracer)
	tracer._instruction_variant = lambda: 0
	form, seg = tracer._motion(clear_view())
	assert form == "simple" and seg.kind == "prepare" and seg.params == {"mode": "clear_view"}
