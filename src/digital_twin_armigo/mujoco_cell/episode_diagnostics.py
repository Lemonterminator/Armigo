"""Post-hoc diagnostics of a closed-loop episode from its telemetry and phase labels (no simulator)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from ..constants import FPS, PHASES

if TYPE_CHECKING:
	from .physics_expert import ExpertParams

# The evaluator's release rule (``metrics.evaluate_episode_health``: ``opening = place_idx[q_cmd_a[place_idx, 5] >
# hold_baseline + 0.02]``): release starts at the first ``place`` frame whose gripper command exceeds the minimum
# gripper command in ``place`` by this much (rad).
EVALUATOR_RELEASE_DELTA_RAD = 0.02


def show_hold_measured(tel: dict[str, np.ndarray], phases: np.ndarray, params: ExpertParams) -> list[float]:
	"""For each show pose, the longest run (s) of show-phase frames in which the measured wrist flex and roll are
	within tolerance of that pose and every arm joint moves slower than ``show_still_speed_rad_s``."""
	ph = np.asarray(phases)
	q = tel["qpos"]
	speed = np.zeros(len(q))
	if len(q) > 1:
		speed[1:] = np.max(np.abs(np.diff(q[:, :5], axis=0)), axis=1) * FPS
	show = np.isin(ph, [PHASES.index("show_move"), PHASES.index("show_hold")])
	out = []
	for wr in params.show_roll_deg:
		ok = (
			show
			& (np.abs(q[:, 3] - params.show_flex) < params.show_flex_tol)
			& (np.abs(q[:, 4] - math.radians(wr)) < math.radians(params.show_roll_tol_deg))
			& (speed < params.show_still_speed_rad_s)
		)
		best = run = 0
		for flag in ok:
			run = run + 1 if flag else 0
			best = max(best, run)
		out.append(best / float(FPS))
	return out


def episode_diagnostics(tel: dict[str, np.ndarray], phases: np.ndarray, params: ExpertParams) -> dict:
	"""``grasp_push_mm``, ``jaw_normal_at_lift_N``, ``release_peak_speed_mps`` and ``show_hold_measured_s`` from an
	episode's telemetry (``nan`` when the episode never reached the phase a metric needs)."""
	ph = np.asarray(phases)
	g = np.flatnonzero(ph == PHASES.index("grasp"))
	lift = np.flatnonzero(ph == PHASES.index("lift"))
	place = np.flatnonzero(ph == PHASES.index("place"))
	out = {"grasp_push_mm": math.nan, "jaw_normal_at_lift_N": math.nan, "release_peak_speed_mps": math.nan}
	if len(g) and len(lift):
		d = tel["brick_pos"][lift[0], :2] - tel["brick_pos"][g[0], :2]
		out["grasp_push_mm"] = float(np.linalg.norm(d) * 1000.0)
		out["jaw_normal_at_lift_N"] = float(np.min(tel["pinch_normal"][lift[0]]))
	if len(place):
		g_cmd = tel["q_cmd"][place, 5]
		opening = place[g_cmd > g_cmd.min() + EVALUATOR_RELEASE_DELTA_RAD]
		if len(opening):
			out["release_peak_speed_mps"] = float(np.linalg.norm(tel["brick_vel"][opening[0] :], axis=1).max())
	out["show_hold_measured_s"] = show_hold_measured(tel, ph, params)
	return out
