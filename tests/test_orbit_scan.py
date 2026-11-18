"""Orbit scan geometry (upward face, ring, eccentric jaw axis), its sweep timing, and the planner on a fixture."""

import math

import numpy as np
import pytest

from digital_twin_armigo.mujoco_cell.orbit_scan import _ordered, jaw_axis_xy, ring_point, top_face

BRICK_2X4 = (0.0636, 0.0318, 0.0192)


def _rot_z(deg):
	c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
	return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot_x(deg):
	c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
	return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def test_upright_brick_uses_its_top_face_with_the_long_side_as_u():
	pos = np.array([0.2, -0.05, 0.0096])
	top = top_face(pos, _rot_z(30), BRICK_2X4)
	assert top.face == 4
	np.testing.assert_allclose(top.centre, pos + [0, 0, 0.0096])
	assert top.half_u == pytest.approx(0.0318)
	assert top.half_v == pytest.approx(0.0159)
	np.testing.assert_allclose(abs(top.u @ [math.cos(math.radians(30)), math.sin(math.radians(30)), 0]), 1.0)
	np.testing.assert_allclose(top.v, [-top.u[1], top.u[0], 0.0])


def test_brick_on_its_side_uses_the_side_face_that_points_up():
	pos = np.array([0.2, 0.0, 0.0159])
	top = top_face(pos, _rot_x(90), BRICK_2X4)  # brick y now points up: face 2 (+y) is on top
	assert top.face == 2
	np.testing.assert_allclose(top.centre, pos + [0, 0, 0.0159], atol=1e-12)
	assert top.half_u == pytest.approx(0.0318)  # brick x stays horizontal
	assert top.half_v == pytest.approx(0.0096)  # brick z is now horizontal


def test_ring_is_a_circle_for_a_square_face_and_the_jaw_axis_goes_eccentric_on_an_ellipse():
	square = top_face(np.array([0.2, 0.0, 0.0096]), np.eye(3), (0.0318, 0.0318, 0.0192))
	radii = [np.linalg.norm(ring_point(square, phi, 0.03) - square.centre[:2]) for phi in np.linspace(0, 6, 7)]
	np.testing.assert_allclose(radii, 0.0159 + 0.03)
	mount = 0.0159 + 0.03
	np.testing.assert_allclose(jaw_axis_xy(square, 1.0, 0.03, mount), square.centre[:2], atol=1e-12)

	long = top_face(np.array([0.2, 0.0, 0.0096]), np.eye(3), BRICK_2X4)
	along_u = jaw_axis_xy(long, 0.0, 0.03, mount) - long.centre[:2]
	along_v = jaw_axis_xy(long, math.pi / 2, 0.03, mount) - long.centre[:2]
	np.testing.assert_allclose(along_u, [0.0318 + 0.03 - mount, 0.0], atol=1e-12)  # out along the long side
	np.testing.assert_allclose(along_v, [0.0, 0.0], atol=1e-12)  # centred across the short side


def test_ordered_plan_sweeps_in_one_roll_direction_through_the_ring():
	from types import SimpleNamespace

	ring = [SimpleNamespace(q=np.array([0, 0, 0, 0, r, 0.0])) for r in (0.9, -0.6, 0.3, -1.2, 0.0, 1.5)]
	plan = _ordered(ring, [ring[0], ring[3]], 0.012, 0.025)
	assert [float(v.q[4]) for v in plan.views] == [-1.2, 0.9]
	assert plan.via[0] == []
	assert [float(q[4]) for q in plan.via[1]] == [-0.6, 0.0, 0.3]
	assert [float(c.q[4]) for c in plan.ring] == sorted(float(c.q[4]) for c in ring)


def test_sweep_matches_a_single_move_and_keeps_each_joint_under_the_peak_speed():
	pytest.importorskip("mujoco")
	from digital_twin_armigo.agent.twin_mujoco import sweep
	from digital_twin_armigo.mujoco_cell.physics_expert import ClosedLoopExpert, ExpertParams

	expert = ClosedLoopExpert.__new__(ClosedLoopExpert)
	expert.p = ExpertParams()
	expert._lo, expert._hi = np.full(6, -np.pi), np.full(6, np.pi)
	expert.last = np.zeros(6)
	target = np.array([0.2, -0.4, 0.1, 0.0, 1.0, 0.3])
	secs = max(0.6, 1.875 * 1.0 / 1.0)
	single = np.array([q for q, _ in sweep(expert, [target], 1.0, 0.6)])
	baseline = np.array([q for q, _ in ClosedLoopExpert._move(expert, target, secs, "approach")])
	np.testing.assert_allclose(single, baseline, atol=1e-5)

	path = [np.array([0, 0, 0, 0, r, 0.3]) for r in (0.5, 1.0, 1.5, 2.0)]
	swept = np.array([q for q, _ in sweep(expert, path, 1.0, 0.6)])
	np.testing.assert_allclose(swept[-1], path[-1])
	speed = np.abs(np.diff(np.vstack([expert.last, swept]), axis=0))[:, :5].max() * 30
	assert speed <= 1.0 + 1e-6
	roll = swept[:, 4]
	assert np.all(np.diff(roll) >= -1e-12)  # never turns back along the ring


def test_orbit_plan_covers_the_side_faces_with_safe_vertical_poses(tmp_path):
	pytest.importorskip("mujoco")
	from digital_twin_armigo.agent.twin_mujoco import WRIST_MIN_PX, TwinMujoco, blind_geometry, sample_case
	from digital_twin_armigo.agent.twin_record import EpisodeTrace
	from digital_twin_armigo.mujoco_cell._mj import mujoco
	from digital_twin_armigo.mujoco_cell.orbit_scan import (
		OrbitParams,
		_body_geoms,
		_brick_geoms,
		_min_distance,
		_robot_geoms,
		plan_orbit_scan,
	)

	backend = TwinMujoco(sample_case(600004, False, 5), None, EpisodeTrace(tmp_path / "e", video=False))
	try:
		assert backend.prepare() is None
		cell, ex = backend.cell, backend.expert
		ex.kin.data.qpos[:] = cell.data.qpos
		params = OrbitParams()
		with blind_geometry(cell):
			plan = plan_orbit_scan(cell, ex.kin, ex.last, gripper=ex.p.open_g, params=params, min_px=WRIST_MIN_PX)
		assert plan.mode == "orbit"
		best = np.max([v.faces_px for v in plan.views], axis=0)
		assert np.all(best[:5] >= WRIST_MIN_PX)
		assert [float(v.q[4]) for v in plan.views] == sorted(float(v.q[4]) for v in plan.views)
		d = mujoco.MjData(cell.model)
		d.qpos[:] = cell.data.qpos
		for view in plan.ring:
			d.qpos[cell.qadr] = view.q
			mujoco.mj_kinematics(cell.model, d)
			assert cell.robot_min_z(d) >= params.min_robot_z_m
			gap = _min_distance(cell.model, d, _robot_geoms(cell), _brick_geoms(cell), 0.05)
			assert gap >= params.brick_clearance_m
			distal = _body_geoms(cell, ("gripper", "camera_mount", "moving_jaw_so101_v1"))
			proximal = _body_geoms(cell, ("base", "shoulder", "upper_arm"))
			assert _min_distance(cell.model, d, distal, proximal, 0.05) >= params.self_clearance_m
			assert -d.site_xmat[cell.site].reshape(3, 3)[2, 0] >= math.cos(math.radians(params.down_tol_deg))
	finally:
		backend.record.close()
		backend.close()
