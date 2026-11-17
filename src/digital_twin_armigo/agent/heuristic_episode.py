"""The heuristic inspection strategy as an :class:`~digital_twin_armigo.mujoco_cell.physics_expert.EpisodePlanner`.

Stages (the decision rules live in :class:`~.heuristic.HeuristicInspector`, this module only connects them to the
simulator, the viewpoint planners and the anomaly detector):

1. **pre-grasp scan** -- up to ``max_prescan`` wrist-camera viewpoints over the brick on the table, chosen to cover
   the eligible faces (:func:`~digital_twin_armigo.mujoco_cell.inspection_planning.greedy_face_cover`); the scan
   stops early once the detector fires;
2. **grasp choice** -- among the expert's grasp candidates, one whose jaws do not cover the suspect faces (after a
   detection) or that covers already-cleared faces (otherwise);
3. **held wrist view** -- one free wrist view of the lifted brick;
4. **top-camera close-ups** -- the arm carries the brick towards the top camera with the next face to check turned to
   it (:func:`~digital_twin_armigo.mujoco_cell.inspection_planning.best_closeup`), up to the inspector's budget.

The detector sees only the colour crop; the per-face pixel counts that steer the stages are geometry (face probes), not
the label. ``log`` keeps one record per view for analysis.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from ..inspection_scorer import AnomalyScorer
from ..mujoco_cell.inspection_planning import (
	ViewCandidate,
	best_closeup,
	closeup_candidates,
	greedy_face_cover,
	path_min_z,
	prescan_candidates,
)
from .heuristic import ELIGIBLE_FACES, HeuristicInspector, ViewEvidence

__all__ = ["HeuristicPlanner", "pinched_faces_now"]

CaptureFn = Callable[[Any, str, Any], "tuple[Any, np.ndarray | None]"]
"""(cell, camera, MjData or None) -> (InspectionView, square colour crop or None), exactly as the training crops."""


def pinched_faces_now(cell) -> tuple[int, int]:
	"""The brick faces the jaws cover right now: the +/- faces of the brick axis best aligned with the jaw opening."""
	d = cell.data
	jaw = d.xipos[cell.jaw_body] - d.site_xpos[cell.site]
	jaw = jaw / max(float(np.linalg.norm(jaw)), 1e-9)
	rot_b = d.xmat[cell.brick_body].reshape(3, 3)
	local = int(np.argmax(np.abs(rot_b[:, :2].T @ jaw)))
	return 2 * local, 2 * local + 1


class HeuristicPlanner:
	"""Pre-grasp scan -> grasp choice -> held view -> top close-ups, with an anomaly detector in the loop.

	``thresholds`` (per camera) override the scorer's own thresholds as the detection threshold."""

	def __init__(
		self,
		scorer: AnomalyScorer,
		capture: CaptureFn,
		closeup_lib: np.ndarray,
		*,
		inspector: HeuristicInspector | None = None,
		thresholds: Mapping[str, float] | None = None,
		max_prescan: int = 4,
		prescan_min_px: int = 800,
		closeup_min_z: float = 0.03,
		closeup_rolls_deg: tuple[float, ...] = (-120.0, -60.0, 0.0, 60.0, 120.0),
		prescan: bool = True,
		prescan_times: int = 1,
		show_poses: tuple[tuple[float, float], ...] | None = None,
	) -> None:
		"""``prescan=False`` skips stage 1. ``show_poses`` (wrist flex rad, wrist roll deg) replaces the close-ups of
		stage 4 by the classic show poses (the lift pose with flex / roll replaced), each looked at by BOTH cameras,
		until the label is known: the baseline with the same detector and decision rules."""
		self.prescan = bool(prescan)
		self.prescan_times = int(prescan_times)
		self.show_poses = None if show_poses is None else tuple(show_poses)
		self._show_i = 0
		self.scorer = scorer
		self.capture = capture
		self.lib = np.asarray(closeup_lib, dtype=np.float64)
		self.inspector = inspector or HeuristicInspector()
		self.thr = dict(thresholds or {})
		self.max_prescan = int(max_prescan)
		self.prescan_min_px = int(prescan_min_px)
		self.closeup_min_z = float(closeup_min_z)
		self.path_clearance_m = 0.005
		self.closeup_rolls_deg = tuple(closeup_rolls_deg)
		self.log: list[dict[str, Any]] = []
		self._closeups: list[ViewCandidate] = []
		self._target: int | None = None

	# -- helpers ---------------------------------------------------------------------------------------------------

	def _threshold(self, camera: str) -> float:
		return float(self.thr[camera]) if camera in self.thr else float(self.scorer.threshold(camera))

	def _observe(self, expert, stage: str, camera: str, target_face: int | None = None) -> ViewEvidence:
		view, image = self.capture(expert.cell, camera, None)
		score = None if image is None else float(self.scorer.score(camera, image))
		ev = ViewEvidence(
			stage, camera, tuple(int(v) for v in view.faces_px), score, self._threshold(camera), target_face
		)
		self.inspector.observe(ev)
		self.log.append(
			{
				"frame": int(expert._frame_idx),
				"stage": stage,
				"camera": camera,
				"faces_px": list(ev.faces_px),
				"brick_px": int(view.brick_px),
				"score": score,
				"threshold": ev.threshold,
				"detected": ev.detected,
				"target_face": target_face,
			}
		)
		return ev

	def _path_ok(self, expert, q_from: np.ndarray, q_to: np.ndarray, steps: int = 8) -> bool:
		"""The straight joint path keeps the arm and the held brick above the table, and the gripper / links clear
		of bowls and clutter (the expert's own clearance check on the interpolated poses)."""
		if path_min_z(expert.cell, q_from, q_to, carried=True) < self.closeup_min_z:
			return False
		qs = [np.asarray(q_from) + s * (np.asarray(q_to) - np.asarray(q_from)) for s in np.linspace(0, 1, steps)]
		expert.kin.data.qpos[:] = expert.cell.data.qpos
		return expert._clearance(qs) >= self.path_clearance_m

	# -- EpisodePlanner --------------------------------------------------------------------------------------------

	def prescan_poses(self, expert) -> list[np.ndarray]:
		p = expert.p
		if not self.prescan:
			return []
		expert.kin.data.qpos[:] = expert.cell.data.qpos  # bowls / clutter as they are now, for the clearance check
		cands = prescan_candidates(
			expert.cell,
			expert._ik,
			np.array([*p.ready_q, expert._active_open_g]),
			clearance=expert._clearance,
			min_clearance=0.005,
		)
		chosen = greedy_face_cover(cands, ELIGIBLE_FACES, self.prescan_min_px, self.max_prescan, self.prescan_times)
		return [c.q for c in chosen]

	def after_prescan_view(self, expert, k: int) -> bool:
		self._observe(expert, "prescan", "wrist_cam")
		return not self.inspector.detected

	def choose_grasp(self, expert, cands: list[dict]) -> int:
		return int(self.inspector.choose_grasp(cands))

	def after_grasp(self, expert) -> None:
		self.inspector.set_grasp(pinched_faces_now(expert.cell))
		self._observe(expert, "held_wrist", "wrist_cam")
		if self.show_poses is not None:
			return
		self._closeups = closeup_candidates(
			expert.cell, self.lib, gripper=expert.p.hold_g, rolls_deg=self.closeup_rolls_deg
		)

	def next_present_pose(self, expert, q_lift: np.ndarray) -> np.ndarray | None:
		if self.show_poses is not None:
			if self.inspector.label() is not None or self._show_i >= len(self.show_poses):
				return None
			flex, roll_deg = self.show_poses[self._show_i]
			self._show_i += 1
			q = np.asarray(q_lift, dtype=np.float64).copy()
			q[3], q[4] = expert._clip_show(flex, roll_deg)[0], np.deg2rad(expert._clip_show(flex, roll_deg)[1])
			return q
		while True:
			face = self.inspector.next_present_face()
			if face is None:
				return None
			pool = list(self._closeups)
			q_now = np.asarray(expert.last, dtype=np.float64)
			while pool:
				c = best_closeup(pool, face)
				if c is None:
					break
				if self._path_ok(expert, q_now, c.q):
					self._target = face
					return c.q
				pool.remove(c)
			# no reachable close-up shows this face: record the attempt so the inspector moves on
			ev = ViewEvidence("present", "top", (0,) * 6, None, self._threshold("top"), face)
			self.inspector.observe(ev)
			self.log.append({"frame": int(expert._frame_idx), "stage": "present", "camera": "top", "skipped": face})

	def after_present(self, expert) -> None:
		if self.show_poses is not None:
			self._observe(expert, "show", "top")
			self._observe(expert, "show", "wrist_cam")
			return
		self._observe(expert, "present", "top", self._target)

	def label(self) -> str | None:
		return self.inspector.label()

	def summary(self) -> dict[str, Any]:
		return {**self.inspector.summary(), "views": list(self.log)}
