"""Wrist camera mount error under ``cameras="jitter"``: the cell renders the offset mount, realistic planners keep
the designed one."""

import numpy as np
import pytest

from digital_twin_armigo.agent.twin_mujoco import WRIST_JITTER, case_cameras, case_wrist_offset, sample_case


def test_wrist_offset_is_bounded_deterministic_and_independent_of_the_other_cameras():
	assert case_wrist_offset(1_100_001, "fixed") is None
	offsets = np.array([case_wrist_offset(s, "jitter") for s in range(1_100_001, 1_100_201)])
	assert offsets.shape == (200, 6)
	assert np.abs(offsets[:, :3]).max() <= WRIST_JITTER["pos_m"]
	assert np.abs(offsets[:, 3:]).max() <= WRIST_JITTER["rot_deg"]
	assert np.abs(offsets[:, :3]).max() > 0.8 * WRIST_JITTER["pos_m"]  # the range is actually used
	assert case_wrist_offset(1_100_001, "jitter") == tuple(offsets[0])
	assert len({tuple(o) for o in offsets}) == 200
	# The wrist draw has its own stream: the top and front cameras are those of case_cameras alone.
	layout = sample_case(1_100_001, False, 5, {"cameras": "jitter"})
	assert (layout.top, layout.front) == case_cameras(1_100_001, "jitter")
	assert layout.wrist_cam_offset == case_wrist_offset(1_100_001, "jitter")


def test_mount_offset_moves_the_camera_in_its_own_frame():
	pytest.importorskip("mujoco")
	from digital_twin_armigo.mujoco_cell._mj import mujoco
	from digital_twin_armigo.mujoco_cell.cell import mount_offset

	pos, quat = np.array([0.1, 0.2, 0.3]), np.array([np.cos(0.4), 0.0, 0.0, np.sin(0.4)])
	p, q = mount_offset(pos, quat, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
	assert np.allclose(p, pos) and np.allclose(q, quat)
	p, q = mount_offset(pos, quat, (0.003, 0.0, 0.0, 0.0, 0.0, 2.0))
	shift = np.zeros(3)
	mujoco.mju_rotVecQuat(shift, np.array([0.003, 0.0, 0.0]), quat)
	assert np.allclose(p - pos, shift) and np.isclose(np.linalg.norm(p - pos), 0.003)
	dq = np.zeros(4)
	mujoco.mju_negQuat(dq, quat)
	rel = np.zeros(4)
	mujoco.mju_mulQuat(rel, dq, q)  # rotation in the camera's own frame
	assert np.isclose(np.degrees(2 * np.arccos(min(1.0, abs(rel[0])))), 2.0)
	assert np.allclose(rel[1:3], 0.0, atol=1e-12)


def test_realistic_planners_see_the_designed_mount(tmp_path):
	pytest.importorskip("mujoco")
	pytest.importorskip("cv2")
	pytest.importorskip("av")
	from digital_twin_armigo.agent.twin_mujoco import TwinMujoco
	from digital_twin_armigo.agent.twin_record import EpisodeTrace
	from digital_twin_armigo.inspection_scorer import FixedScorer

	layout = sample_case(1_100_001, False, 5, {"cameras": "jitter"})
	scorer = FixedScorer(lambda _camera, _rgb: -1, {"top": 0, "wrist_cam": 0})
	backend = TwinMujoco(layout, scorer, EpisodeTrace(tmp_path / "t", video=False), sensing="realistic")
	try:
		cell = backend.cell
		m, d, cid = cell.model, cell.data, cell.camera_id("wrist_cam")
		nominal, actual = cell.wrist_cam_nominal, cell.wrist_cam_actual
		assert not np.allclose(nominal[0], actual[0])
		assert np.allclose(m.cam_pos[cid], actual[0]) and np.allclose(m.cam_quat[cid], actual[1])
		seen = cell.rgb("wrist_cam").copy()
		xpos = d.cam_xpos[cid].copy()
		with backend._belief():
			assert np.allclose(m.cam_pos[cid], nominal[0]) and np.allclose(m.cam_quat[cid], nominal[1])
			assert not np.allclose(d.cam_xpos[cid], xpos)
			planned = cell.rgb("wrist_cam").copy()
		assert np.allclose(m.cam_pos[cid], actual[0]) and np.allclose(m.cam_quat[cid], actual[1])
		assert np.allclose(d.cam_xpos[cid], xpos)
		assert (
			np.array_equal(cell.rgb("wrist_cam"), seen) or np.abs(cell.rgb("wrist_cam").astype(int) - seen).max() <= 1
		)  # renderer repeatability
		# The two mounts give visibly different images (here a ~1 mm, ~1 deg offset moves ~2 % of pixels by > 8 levels).
		assert (np.abs(planned.astype(int) - seen).max(axis=-1) > 8).mean() > 0.005
	finally:
		backend.close()


def test_fixed_cameras_leave_the_wrist_mount_alone(tmp_path):
	pytest.importorskip("mujoco")
	from digital_twin_armigo.constants import SO101_MJCF_DIR
	from digital_twin_armigo.mujoco_cell.cell import Cell

	cell = Cell(sample_case(600001, False, 5), SO101_MJCF_DIR)
	try:
		assert cell.wrist_cam_actual is cell.wrist_cam_nominal
		cid = cell.camera_id("wrist_cam")
		before = cell.model.cam_pos[cid].copy()
		with cell.nominal_wrist_cam():
			assert np.array_equal(cell.model.cam_pos[cid], before)
	finally:
		cell.close()
