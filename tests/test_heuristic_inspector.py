"""Tests for the pure heuristic inspection decisions (no MuJoCo, no torch)."""

from __future__ import annotations

import json

import pytest

from digital_twin_armigo.agent.heuristic import FACES, HeuristicInspector, ViewEvidence


def _ev(
	stage: str,
	camera: str,
	visible: tuple[int, ...] = (),
	*,
	score: float | None = 0.0,
	threshold: float = 10.0,
	target_face: int | None = None,
	px: int = 1000,
) -> ViewEvidence:
	faces = tuple(px if f in visible else 0 for f in range(FACES))
	return ViewEvidence(stage, camera, faces, score, threshold, target_face)


def test_prescan_alone_clears_every_face_and_labels_normal():
	insp = HeuristicInspector()
	insp.observe(_ev("prescan", "wrist_cam", (0, 1, 2)))
	insp.observe(_ev("prescan", "wrist_cam", (2, 3, 4)))

	assert insp.detected is False
	assert insp.top_confirmed is False
	assert insp.cleared_faces == frozenset({0, 1, 2, 3, 4})
	assert insp.suspect_faces == frozenset()
	assert insp.label() == "normal"
	assert insp.next_present_face() is None
	assert insp.summary()["reason"] == "all_faces_cleared"


def test_detection_in_one_prescan_view_gives_visible_minus_cleared():
	insp = HeuristicInspector()
	insp.observe(_ev("prescan", "wrist_cam", (0, 1, 2), score=20.0))
	insp.observe(_ev("prescan", "wrist_cam", (2,), score=0.0))

	assert insp.detected is True
	assert insp.cleared_faces == frozenset({2})
	assert insp.suspect_faces == frozenset({0, 1})
	assert insp.label() == "defective"
	assert insp.summary()["reason"] == "defect_seen"


def test_suspects_are_the_intersection_of_detecting_views():
	insp = HeuristicInspector()
	insp.observe(_ev("prescan", "wrist_cam", (0, 1), score=20.0))
	insp.observe(_ev("prescan", "wrist_cam", (1, 2), score=20.0))

	assert insp.suspect_faces == frozenset({1})


def test_suspects_fall_back_to_the_union_when_intersection_is_empty():
	insp = HeuristicInspector()
	insp.observe(_ev("prescan", "wrist_cam", (0, 1), score=20.0))
	insp.observe(_ev("prescan", "wrist_cam", (2, 3), score=20.0))

	assert insp.suspect_faces == frozenset({0, 1, 2, 3})


def test_cleared_faces_do_not_empty_the_suspects():
	insp = HeuristicInspector()
	insp.observe(_ev("prescan", "wrist_cam", (0,), score=20.0))
	insp.observe(_ev("prescan", "wrist_cam", (0,), score=0.0))

	assert insp.cleared_faces == frozenset({0})
	assert insp.suspect_faces == frozenset({0})


def test_detecting_view_with_no_clear_face_falls_back_to_any_pixels():
	insp = HeuristicInspector()
	insp.observe(ViewEvidence("prescan", "wrist_cam", (100, 0, 0, 0, 0, 0), 20.0, 10.0))

	assert insp.suspect_faces == frozenset({0})


def test_clear_uses_the_camera_specific_minimum_pixels():
	insp = HeuristicInspector()
	insp.observe(ViewEvidence("present", "top", (200, 200, 0, 0, 0, 0), 0.0, 10.0, target_face=0))
	assert insp.cleared_faces == frozenset()

	insp.observe(ViewEvidence("prescan", "wrist_cam", (799, 0, 0, 0, 0, 0), 0.0, 10.0))
	assert insp.cleared_faces == frozenset()

	insp.observe(ViewEvidence("prescan", "wrist_cam", (800, 0, 0, 0, 0, 0), 0.0, 10.0))
	assert insp.cleared_faces == frozenset({0})


def test_none_score_neither_detects_nor_clears():
	insp = HeuristicInspector()
	insp.observe(ViewEvidence("prescan", "wrist_cam", (1000,) * FACES, None, 10.0))
	insp.observe(ViewEvidence("present", "top", (1000,) * FACES, None, 10.0, target_face=2))

	assert insp.detected is False
	assert insp.cleared_faces == frozenset()
	assert insp.suspect_faces == frozenset()
	assert insp.label() is None
	assert insp.presented_faces == (2,)
	assert insp.summary()["n_views"] == 2


def test_choose_grasp_without_detection_prefers_cleared_pinched_faces():
	insp = HeuristicInspector()
	insp.observe(_ev("prescan", "wrist_cam", (0, 1, 2)))
	candidates = [
		{"pinched_faces": (1, 2), "score": 0.5},
		{"pinched_faces": (0, 1), "score": 0.9},
		{"pinched_faces": (0, 2), "score": 0.9},
		{"pinched_faces": (0, 3), "score": 5.0},
		{"pinched_faces": (3, 4), "score": 3.0},
	]

	assert insp.choose_grasp(candidates) == 1


def test_choose_grasp_with_detection_avoids_suspects_then_takes_best_score():
	insp = HeuristicInspector()
	insp.observe(_ev("prescan", "wrist_cam", (2, 3), score=20.0))
	candidates = [
		{"pinched_faces": (2, 3), "score": 9.0},
		{"pinched_faces": (0, 1), "score": 1.0},
		{"pinched_faces": (0, 1), "score": 5.0},
	]

	assert insp.choose_grasp(candidates) == 2


def test_choose_grasp_requires_a_candidate():
	insp = HeuristicInspector()
	with pytest.raises(ValueError):
		insp.choose_grasp([])


def test_next_present_face_starts_at_top_then_ascending_and_skips_used_faces():
	insp = HeuristicInspector(max_presents=10)
	assert insp.next_present_face() == 4

	insp.observe(_ev("present", "top", (4,), target_face=4))
	assert insp.next_present_face() == 0

	insp.set_grasp((0, 1))
	assert insp.next_present_face() == 2

	insp.observe(_ev("prescan", "wrist_cam", (3,)))
	assert insp.next_present_face() == 2

	insp.observe(_ev("present", "top", (2,), target_face=2))
	assert insp.next_present_face() is None


def test_next_present_face_skips_cleared_and_pinched_faces():
	insp = HeuristicInspector()
	insp.observe(_ev("prescan", "wrist_cam", (4,)))
	assert insp.next_present_face() == 0

	insp.set_grasp((0, 2))
	assert insp.next_present_face() == 1


def test_next_present_face_respects_the_budget():
	insp = HeuristicInspector(max_presents=2)
	assert insp.next_present_face() == 4

	insp.observe(_ev("present", "top", (4,), target_face=4))
	assert insp.next_present_face() == 0

	insp.observe(_ev("present", "top", (0,), target_face=0))
	assert insp.presented_faces == (4, 0)
	assert insp.next_present_face() is None


def test_next_present_face_follows_suspects_while_waiting_for_top():
	insp = HeuristicInspector(require_top_confirmation=True)
	insp.observe(_ev("prescan", "wrist_cam", (0, 2), score=20.0))
	assert insp.suspect_faces == frozenset({0, 2})
	assert insp.next_present_face() == 0

	insp.observe(_ev("present", "top", (), target_face=0))
	assert insp.next_present_face() == 2

	insp.set_grasp((2,))
	assert insp.next_present_face() is None


def test_top_confirmation_is_required_after_a_wrist_detection():
	insp = HeuristicInspector(require_top_confirmation=True)
	insp.observe(_ev("held_wrist", "wrist_cam", (1,), score=20.0))
	assert insp.label() is None
	assert insp.next_present_face() == 1

	insp.observe(_ev("present", "top", (1,), score=20.0, target_face=1))
	assert insp.top_confirmed is True
	assert insp.label() == "defective"
	assert insp.summary()["reason"] == "defect_seen"


def test_wrist_only_detection_is_defective_once_the_budget_is_spent():
	insp = HeuristicInspector(max_presents=1, require_top_confirmation=True)
	insp.observe(_ev("held_wrist", "wrist_cam", (0,), score=20.0))
	assert insp.label() is None

	insp.observe(_ev("present", "top", (0,), target_face=0))
	assert insp.presented_faces == (0,)
	assert insp.label() == "defective"
	assert insp.summary()["reason"] == "wrist_only"


def test_wrist_only_detection_is_defective_when_no_face_can_be_shown():
	insp = HeuristicInspector(require_top_confirmation=True)
	insp.observe(_ev("held_wrist", "wrist_cam", (0,), score=20.0))
	insp.observe(_ev("present", "top", (), target_face=0))

	assert insp.next_present_face() is None
	assert insp.label() == "defective"
	assert insp.summary()["reason"] == "wrist_only"


def test_summary_is_json_serialisable_and_reports_uncovered_faces():
	insp = HeuristicInspector()
	insp.observe(_ev("prescan", "wrist_cam", (0,)))
	insp.set_grasp((1, 2))

	summary = insp.summary()
	loaded = json.loads(json.dumps(summary))

	assert set(loaded) == {
		"detected",
		"top_confirmed",
		"cleared",
		"suspect",
		"presented",
		"pinched",
		"label",
		"n_views",
		"reason",
	}
	assert loaded["cleared"] == [0]
	assert loaded["pinched"] == [1, 2]
	assert loaded["presented"] == []
	assert loaded["reason"] == "uncovered_faces:[1, 2, 3, 4]"
