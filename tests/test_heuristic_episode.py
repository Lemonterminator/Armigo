"""Viewpoint planners and the heuristic EpisodePlanner wired into the closed-loop expert."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _cand(faces, q=0.0):
	from digital_twin_armigo.mujoco_cell.inspection_planning import ViewCandidate

	return ViewCandidate(np.full(6, q), "wrist_cam", tuple(faces), sum(faces), 1.0)


def test_greedy_face_cover_picks_fewest_views():
	pytest.importorskip("mujoco")
	from digital_twin_armigo.mujoco_cell.inspection_planning import greedy_face_cover

	a = _cand((900, 0, 900, 0, 900, 0), 1)
	b = _cand((0, 900, 0, 0, 0, 0), 2)
	c = _cand((0, 900, 0, 900, 0, 0), 3)
	d = _cand((500, 500, 500, 500, 500, 0), 4)  # below the pixel threshold everywhere
	out = greedy_face_cover([b, d, a, c], (0, 1, 2, 3, 4), 800, 4)
	assert out == [a, c]
	assert greedy_face_cover([d], (0, 1), 800, 4) == []
	assert greedy_face_cover([a, b, c], (0, 1, 2, 3, 4), 800, 1) == [a]


def test_best_closeup_and_library():
	pytest.importorskip("mujoco")
	from digital_twin_armigo.mujoco_cell.inspection_planning import best_closeup, closeup_library

	a, b = _cand((100, 0, 0, 0, 0, 0), 1), _cand((300, 0, 0, 0, 0, 0), 2)
	assert best_closeup([a, b], 0) is b
	assert best_closeup([a, b], 3) is None
	rows = [
		{"layout": 0, "source": "random", "visible_fraction": 0.9, "brick_px": p, "q": [p, 0, 0, 0, 0]}
		for p in (10, 30, 20)
	] + [{"layout": 0, "source": "random", "visible_fraction": 0.1, "brick_px": 99, "q": [99, 0, 0, 0, 0]}]
	lib = closeup_library(rows, per_layout=2)
	assert lib.shape == (2, 5) and list(lib[:, 0]) == [30, 20]


class _OracleScorer:
	"""Score = visible defect pixels of the last capture (test only: reads ground truth on purpose)."""

	def __init__(self, defective):
		self.last = None
		self.defective = defective  # the hidden probe cross of a normal brick also counts in defect_px

	def capture(self, cell, camera, data):
		from digital_twin_armigo.mujoco_cell.inspection_views import capture_view

		v = capture_view(cell, camera, data=data, image_size=32)
		self.last = v
		return v, v.image

	def score(self, camera, image):
		return float(self.last.defect_px) if self.defective else 0.0

	def threshold(self, camera):
		return 25.0


@pytest.mark.parametrize("defective", [True, False])
def test_heuristic_planner_runs_stages_and_labels(defective):
	pytest.importorskip("mujoco")
	sys.path.insert(0, str(ROOT / "scripts"))
	import bench_physics_expert as bpe

	from digital_twin_armigo.agent.heuristic import HeuristicInspector
	from digital_twin_armigo.agent.heuristic_episode import HeuristicPlanner
	from digital_twin_armigo.constants import SO101_MJCF_DIR
	from digital_twin_armigo.mujoco_cell import Cell
	from digital_twin_armigo.mujoco_cell.physics_expert import ExpertParams, run_closed_loop_episode
	from digital_twin_armigo.physics import load_physics_spec

	lay = bpe.sample_layouts(1, 3, bpe.load_top_camera())[0]
	lay.defect_kind, lay.defect_face, lay.defective, lay.defect_seed = "cross_var", 0, defective, 5
	spec = load_physics_spec()
	cell = Cell(lay, SO101_MJCF_DIR, physics=spec, dynamic_clutter=True, assisted_grasp=False, face_probes=True)
	sc = _OracleScorer(defective)
	lib = np.array([[0.0, -0.35, 0.25, 1.35, 0.0]])  # a single (cheap) close-up candidate: the ready pose
	planner = HeuristicPlanner(sc, sc.capture, lib, inspector=HeuristicInspector(max_presents=2))
	try:
		_, _, info = run_closed_loop_episode(
			cell,
			ExpertParams(max_frames=2600),
			q_start=bpe._q_start(),
			brick_xy=lay.brick_xy,
			brick_yaw_deg=lay.brick_yaw_deg,
			bowls={"normal": lay.green, "defective": lay.pink},
			planner=planner,
		)
	finally:
		cell.close()
	stages = [v["stage"] for v in planner.log]
	assert stages[0] == "prescan" and "held_wrist" in stages
	assert info["inspection_label"] in (None, "defective" if defective else "normal")  # never the wrong label
	if info["inspection_label"] is not None:
		chosen = lay.pink if defective else lay.green
		assert info["target_bowl_xy"] == [float(v) for v in chosen]
	if defective:
		assert info["inspection_label"] == "defective"
	assert set(planner.inspector.pinched_faces) in ({0, 1}, {2, 3})


def test_planner_mode_is_exclusive():
	pytest.importorskip("mujoco")
	from digital_twin_armigo.mujoco_cell.physics_expert import ClosedLoopExpert, ExpertParams

	class _P:
		pass

	with pytest.raises(ValueError, match="exactly one"):
		ClosedLoopExpert(None, np.zeros(6), (0.0, 0.0), ExpertParams(), bowls={"normal": (0, 0)}, planner=_P())
	with pytest.raises(ValueError, match="needs bowls"):
		ClosedLoopExpert(None, np.zeros(6), None, ExpertParams(), planner=_P())


def test_greedy_face_cover_double_coverage():
	pytest.importorskip("mujoco")
	from digital_twin_armigo.mujoco_cell.inspection_planning import greedy_face_cover

	a = _cand((900, 900, 0, 0, 0, 0), 1)
	b = _cand((900, 0, 0, 0, 0, 0), 2)
	c = _cand((0, 900, 0, 0, 0, 0), 3)
	assert greedy_face_cover([a, b, c], (0, 1), 800, 4) == [a]
	assert greedy_face_cover([a, b, c], (0, 1), 800, 4, times=2) == [a, b, c]


def test_inspector_min_clear_views():
	from digital_twin_armigo.agent.heuristic import HeuristicInspector, ViewEvidence

	ins = HeuristicInspector(eligible_faces=(0,), min_clear_views=2)
	ev = ViewEvidence("prescan", "wrist_cam", (900, 0, 0, 0, 0, 0), 1.0, 10.0)
	ins.observe(ev)
	assert ins.label() is None and not ins.cleared_faces
	ins.observe(ev)
	assert ins.label() == "normal"
