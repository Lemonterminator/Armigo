"""ACT grasp skill: the expert runs every skill except the grasp, which an ACT student runs.

:class:`HybridTwin` is a :class:`~.student_rollout.StudentTwin` whose non-grasp motions go to the expert (with segment
markers, as :class:`~.student_rollout.ExpertTracer` writes them). Its grasp is planned on the perceived brick with
the *canonical* wrist roll: of the symmetric grasp solutions, the one nearest the current wrist roll (the rule the
reverse-planned demonstrations of ``scripts/vla/reverse_grasp_demos.py`` are labelled with), instead of the
planner's clearance-scored choice, which a student cannot see.

:func:`grasp_trial` runs one case up to and including the first grasp and lift (``prepare`` -> ``_grasp``) and
checks the outcome on simulator truth: the same trial for the scripted expert (``policy=None``) and for the student
is the ACT-vs-script grasp comparison.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from .student_rollout import ExpertTracer, StudentTwin

LIFTED_M = 0.03  # brick centre this far above its resting height counts as lifted


def canonical_pick(cands: list[dict], wrist_roll: float, min_clear: float) -> tuple[dict | None, bool]:
	"""Nearest symmetric grasp solution to ``wrist_roll`` over every geometric candidate (clearance ignored, so the
	choice depends only on the brick yaw and the wrist roll); when that one is blocked, the nearest clear one.
	Returns (candidate, fell_back)."""
	geo = [c for c in cands if c["gap_ok"]]
	if not geo:
		return None, False

	def nearest(pool: list[dict]) -> dict:
		return pool[int(np.argmin([abs(c["roll"] - wrist_roll) for c in pool]))]

	c = nearest(geo)
	if c["clearance_m"] >= min_clear:
		return c, False
	clear = [c for c in geo if c["clearance_m"] >= min_clear]
	return (nearest(clear), True) if clear else (None, True)


class HybridTwin(StudentTwin, ExpertTracer):
	"""Expert everywhere, the student policy for ``grasp_and_lift`` (grasp + lift segments) only.

	``student_grasp`` switches the student on (default) or off: the step tools leave it off so ``present_bottom``
	and ``place`` keep the scripted grasp, and switch it on for the ``act_grasp`` tool only."""

	student_grasp = True

	def _execute(self, generator):
		form, value = self._motion(generator)
		if self.student_grasp and form == "composite" and value[0] == "grasp_and_lift":
			return StudentTwin._execute(self, generator)
		return ExpertTracer._execute(self, generator)

	def _planned_grasp(self) -> tuple[np.ndarray, float, float]:
		"""The perceived brick's grasp at the canonical roll (nearest the current wrist roll)."""
		ex = self.expert
		obs = dict(ex.obs)
		if self.realistic:
			belief = self._brick_belief()
			if belief is not None:
				obs["brick_pos"], obs["brick_quat"] = belief
		elif self.brick_est is not None:
			obs["brick_pos"], obs["brick_quat"] = self.brick_est.pos.copy(), self.brick_est.quat_xyzw()
		saved = (ex.obs, ex.kin.data.qpos.copy())
		try:
			ex.obs = obs
			cands = ex.grasp_candidates(open_g=ex._active_open_g)
		finally:
			ex.obs, qpos = saved
			ex.kin.data.qpos[:] = qpos
		c, fell_back = canonical_pick(cands, float(ex.last[4]), ex.p.min_grasp_clearance_m)
		if c is None:  # nothing reachable: fall back to the planner (the student will likely fail; it is recorded)
			return super()._planned_grasp()
		self.event("canonical_grasp", roll_deg=round(math.degrees(c["roll"]), 1), fell_back=fell_back)
		return np.asarray(c["target"], dtype=np.float64).copy(), float(c["roll"]), float(c["open_g"])


def grasp_trial(
	case,
	out: Path,
	*,
	policy=None,
	sensing: str = "realistic",
	clamp_wait: bool = True,
	clamp_wait_empty_rad: float | None = 0.03,
	video: bool = False,
) -> dict[str, Any]:
	"""``prepare`` then the first grasp and lift of ``case``; the scripted expert when ``policy`` is None, else the
	hybrid (student grasp). Returns the outcome on truth: ``lifted`` (brick up and between both jaws)."""
	from .twin_mujoco import sample_case
	from .twin_record import EpisodeTrace

	trace = EpisodeTrace(out, video=video)
	layout = sample_case(*case)
	if policy is None:
		twin = ExpertTracer(layout, None, trace, sensing=sensing)
	else:
		twin = HybridTwin(
			layout,
			None,
			trace,
			policy,
			sensing=sensing,
			clamp_wait=clamp_wait,
			clamp_wait_empty_rad=clamp_wait_empty_rad,
		)
	z0 = float(twin.cell.brick_pos()[2])
	error = twin.prepare()
	frames0 = twin.ep.n_frames
	if not error:
		error = twin._grasp()
	tel = twin.ep.telemetry()
	f = tel["pinch_normal"][-1] if len(tel["pinch_normal"]) else np.zeros(2)
	dz = float(twin.cell.brick_pos()[2] - z0)
	lifted = bool(dz > LIFTED_M and min(f) > twin.expert.p.lost_force_n)
	result = {
		"case": list(case),
		"student": policy is not None,
		"lifted": lifted,
		"error": error,
		"aborted": twin.expert.aborted,
		"grasp_retries": int(twin.expert.grasp_retries),
		"grasp_frames": int(twin.ep.n_frames - frames0),
		"brick_dz_m": dz,
		"verdict": "SUCCESS" if lifted else "FAILED",
		"student_eligible": False,
		"rewinds": 0,
	}
	trace.finish(result, {"engine": "mujoco", "sensing": sensing, "case": list(case), "student": policy is not None})
	return result
