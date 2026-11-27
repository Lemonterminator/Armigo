import numpy as np
import pytest


def test_3d_context_does_not_encode_the_defect_or_its_face(tmp_path):
	pytest.importorskip("mujoco")
	pytest.importorskip("cv2")
	pytest.importorskip("av")
	from digital_twin_armigo.agent.twin_mujoco import TwinMujoco, sample_case
	from digital_twin_armigo.agent.twin_record import EpisodeTrace
	from digital_twin_armigo.inspection_scorer import FixedScorer

	geometries = []
	observations = []
	for i, (defective, face) in enumerate(((False, 5), (True, 5), (True, 0))):
		trace = EpisodeTrace(tmp_path / str(i), video=False)
		scorer = FixedScorer(lambda _camera, _rgb: -1, {"top": 0, "wrist_cam": 0})
		backend = TwinMujoco(sample_case(187271, defective, face), scorer, trace)
		try:
			geometries.append(backend._geometry())
			joints = backend.cell.joints().copy()
			observations.append(backend._capture("top"))
			assert np.array_equal(backend.cell.joints(), joints)
		finally:
			trace.close()
			backend.close()
	assert geometries[0] == geometries[1] == geometries[2]
	assert observations[0]["face_pixels"] == observations[1]["face_pixels"] == observations[2]["face_pixels"]


def _rim_pose(backend):
	"""Arm pose with the jaws inside the green bowl the backend believes in, low over its floor."""
	ex = backend.expert
	xy = np.asarray(backend.bowls["green"])
	out = xy / np.linalg.norm(xy)  # radially away from the robot base, so the arm reaches over the bowl
	target = np.array([*(xy - 0.06 * out), 0.005])
	return ex._ik(target, np.array([*ex.p.ready_q, ex.p.open_g]), 0.0)


def _move_true_bowl(backend, dx):
	from digital_twin_armigo.mujoco_cell._mj import mujoco

	m = backend.cell.model
	m.body_pos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "green_bowl")] += (dx, 0.0, 0.0)
	mujoco.mj_forward(m, backend.cell.data)


@pytest.mark.parametrize("sensing", ["truth", "realistic"])
def test_realistic_clearance_follows_perceived_bowls_not_true_ones(tmp_path, sensing):
	"""After perception, moving the simulator's bowl must not change a realistic clearance decision (it reads the
	perceived bowls only), while the truth clearance follows the simulator's bowl."""
	pytest.importorskip("mujoco")
	pytest.importorskip("cv2")
	from digital_twin_armigo.agent.twin_mujoco import TwinMujoco, sample_case
	from digital_twin_armigo.agent.twin_record import EpisodeTrace
	from digital_twin_armigo.inspection_scorer import FixedScorer

	trace = EpisodeTrace(tmp_path / sensing, video=False)
	scorer = FixedScorer(lambda _camera, _rgb: -1, {"top": 0, "wrist_cam": 0})
	backend = TwinMujoco(sample_case(600001, False, 5), scorer, trace, sensing=sensing)
	try:
		ex = backend.expert
		q = _rim_pose(backend)
		ex.kin.data.qpos[:] = backend.cell.data.qpos
		before = ex._clearance([q])
		assert before < 0  # the gripper is in the bowl's (proxy) volume
		_move_true_bowl(backend, 0.4)
		after = ex._clearance([q])
		if sensing == "realistic":
			assert after == before
			# No path of the realistic clearance touches the simulator's bowl / clutter geoms.
			ex._env_geoms = lambda: pytest.fail("realistic clearance read the simulator's bowl geoms")
			assert not backend._straight_clear(ex.last, q)
			backend.bowls["green"] = backend.bowls["green"] + (0.4, 0.0)  # a new perception moves the proxy
			assert ex._clearance([q]) > 0
		else:
			assert after > 0
	finally:
		trace.close()
		backend.close()


def test_realistic_refuses_clutter(tmp_path):
	pytest.importorskip("mujoco")
	from digital_twin_armigo.agent.twin_mujoco import TwinMujoco, sample_case
	from digital_twin_armigo.agent.twin_record import EpisodeTrace
	from digital_twin_armigo.inspection_scorer import FixedScorer

	layout = sample_case(600001, False, 5)
	layout.clutter = [(*layout.green, 0.0, 0.5, 0.5, 0.5)]
	trace = EpisodeTrace(tmp_path / "c", video=False)
	try:
		with pytest.raises(ValueError, match="clutter"):
			TwinMujoco(layout, FixedScorer(lambda _c, _r: -1, {"top": 0, "wrist_cam": 0}), trace, sensing="realistic")
	finally:
		trace.close()


def test_perceived_bowl_proxy_geometry():
	from digital_twin_armigo.agent.bowl_clearance import bowl_distance, box_surface_points
	from digital_twin_armigo.mujoco_cell.scene import _BOWL_WALL_POLYS

	pts = box_surface_points(np.zeros(3), np.array([0.01, 0.02, 0.005]), 0.003)
	assert np.allclose(np.abs(pts).max(axis=0), [0.01, 0.02, 0.005])
	c = np.array([0.2, -0.1])

	def at(r, z, angle=0.7):
		return np.array([[c[0] + r * np.cos(angle), c[1] + r * np.sin(angle), z]])

	assert np.isclose(bowl_distance(at(0.3, 0.0201), c), 0.2)  # level with the rim edge
	assert np.isclose(bowl_distance(at(0.0, 0.05), c), 0.028)  # above the centre: rim height 22 mm
	assert bowl_distance(at(0.05, 0.01), c) < 0  # in the (filled) hollow
	assert 0 < bowl_distance(at(0.099, 0.004), c) < 0.01  # under the flared rim, outside the plate
	# The nominal profile contains the simulator's collision wall (every vertex of its convex pieces).
	for poly in _BOWL_WALL_POLYS:
		for r, z in poly:
			assert bowl_distance(at(r, z), c) <= 1e-9


def _plans(backend):
	"""Grasp candidates (roll, pinched faces, target) and the first wrist-scan plan the backend makes now."""
	ex = backend.expert
	backend._belief_obs = True  # the grasp is planned on the perceived pose (TwinMujoco._grasp)
	try:
		ex.obs = backend._observe()
	finally:
		backend._belief_obs = False
	grasps = [(round(c["roll"], 9), c["pinched_faces"], tuple(np.round(c["target"], 9))) for c in ex.grasp_candidates()]
	with backend._belief():
		ex.kin.data.qpos[:] = backend.cell.data.qpos
	steps, mode, _ = backend._scan_plan([], 0)
	views = [(tuple(np.round(c.q, 9)), tuple(c.faces_px)) for c, _ in steps]
	return grasps, mode, views


@pytest.mark.parametrize("sensing", ["truth", "realistic"])
def test_realistic_plans_use_the_perceived_brick_size(tmp_path, sensing, monkeypatch):
	"""The realistic grasp axes, orbit ring and held-brick clearance use the size class perceived from the top depth
	footprint: changing the simulator's brick size behind a fixed perception leaves the realistic plans unchanged,
	while the truth plans follow it."""
	pytest.importorskip("mujoco")
	pytest.importorskip("cv2")
	from digital_twin_armigo.agent.twin_mujoco import INITIAL_Q, TwinMujoco, sample_case
	from digital_twin_armigo.agent.twin_record import EpisodeTrace
	from digital_twin_armigo.constants import DUPLO_2X2, DUPLO_2X4
	from digital_twin_armigo.inspection_scorer import FixedScorer
	from digital_twin_armigo.mujoco_cell.inspection_views import HeldBrickPoser

	monkeypatch.delenv("ARMIGO_BRICK_PERCEPTION", raising=False)
	trace = EpisodeTrace(tmp_path / sensing, video=False)
	scorer = FixedScorer(lambda _camera, _rgb: -1, {"top": 0, "wrist_cam": 0})
	layout = sample_case(600001, False, 5)
	assert tuple(layout.brick_size) == DUPLO_2X2
	backend = TwinMujoco(layout, scorer, trace, sensing=sensing)
	try:
		if sensing == "realistic":
			assert backend._perceive() is None and backend.brick_est.size == DUPLO_2X2
		before = _plans(backend)
		assert {c[1] for c in before[0]} == {(0, 1), (2, 3)}  # a square brick may be pinched across either axis
		backend.layout.brick_size = DUPLO_2X4  # the simulator's brick changes behind the (fixed) perception
		ex = backend.expert  # rebuilt as a retry / restore does, so nothing cached at construction survives
		backend.expert = backend.expert_type(backend.cell, INITIAL_Q, backend._expert_xy, ex.p, ex.rp)
		after = _plans(backend)
		if sensing == "realistic":
			assert after == before
			assert backend._plan_brick_size() == DUPLO_2X2
			assert HeldBrickPoser(backend.cell, backend._plan_brick_size()).size.tolist() == list(DUPLO_2X2)
		else:
			assert {c[1] for c in after[0]} == {(2, 3)}  # a 2x4 brick only across its short axis
			assert after[2] != before[2]  # the orbit ring follows the longer top face
			assert backend._plan_brick_size() is None
	finally:
		trace.close()
		backend.close()
