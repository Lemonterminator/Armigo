"""Shared contract of the v2 experiment data (``D:/Armigo_data_v2``).

Layout on disk::

	<data root>/runs/<batch>/<engine>/<group>/<case_id>[-a<attempt>]/   one episode (EpisodeTrace layout)
	<data root>/suites/                                                frozen suites + manifest.json
	<data root>/reports/<batch>/                                       scoring and statistics output
	<data root>/inventory/                                             rebuilt by ``armigo-inventory scan``
	<data root>/audit/reviews.sqlite                                   human verdicts, never rebuilt

Every episode directory carries

- ``episode.json`` whose ``provenance.experiment`` is an :data:`EXPERIMENT_FIELDS` dict, and
- ``run_status.json``: ``{"state": one of RUN_STATES, "started": iso, "finished": iso|None, "error": str|None,
  "host": str, "pid": int}`` written by the runner before the episode starts and rewritten when it ends.

Pairing keys: ``layout_id`` pairs the normal and the defective brick of one layout; ``case_id`` pairs the same case
across engines, detectors, camera conditions and strategies. Neither depends on the run, so they survive reruns.
"""

from __future__ import annotations

ENGINES = ("mujoco", "isaac")
SENSING = ("truth", "realistic")
CAMERA_CONDITIONS = ("fixed", "recal", "drift")
# A: the current text skill selection (first qualifying alarm decides); B: fixed-rule re-view; C: Qwen picks the
# re-view among the same reachable candidates as B. ``None`` on omniscient phase-1 runs (strategy A semantics).
STRATEGIES = ("A", "B", "C")
DETECTORS = ("v6", "v7bot")

RUN_STATES = ("planned", "running", "interrupted", "done", "infra_failed")

EXPERIMENT_FIELDS = (
	"batch",  # e.g. "p1-bench16"
	"engine",  # ENGINES
	"sensing",  # SENSING
	"group",  # directory name under <batch>/<engine>/, e.g. "truth-v7bot" or "realistic-recal-C-v7bot"
	"camera_condition",  # CAMERA_CONDITIONS
	"strategy",  # STRATEGIES
	"detector",  # DETECTORS
	"detector_sha256",
	"suite",  # suite file name
	"suite_sha256",
	"layout_id",  # "L<seed>"
	"case_id",  # "L<seed>-N" (normal) or "L<seed>-D<face>" (defective)
	"run_id",  # uuid4 hex of this attempt
	"attempt",  # 1, 2, ...
	"parent_run",  # run_id of the attempt this one reruns, or None
	"code_commit",
	"budget",  # {"max_scored_views": 6, "max_review_views": 2, ...}
)

# First-error root causes (scoring.attribution and the human audit use the same list).
ROOT_CAUSES = (
	"grasp_or_path_failure",
	"target_face_not_in_view",
	"occlusion",
	"crop_error",
	"false_positive_visible",
	"false_negative_visible",
	"premature_stop",
	"recovery_failure",
	"placement_failure",
	"infra_failure",
	"evidence_insufficient",  # label correct, but no view before the decision could tell the bricks apart
	"none",  # success with sufficient evidence
	"other",
)

# Physical outcome computed offline from replay/motion.npz (engine independent).
PHYSICAL_OUTCOMES = ("green_bowl", "pink_bowl", "table", "lost", "unknown")

AUDIT_STATES = ("unreviewed", "confirmed", "disputed", "needs_rerun")


def layout_id(seed: int) -> str:
	return f"L{int(seed)}"


def case_id(seed: int, positive: bool, face: int) -> str:
	return f"L{int(seed)}-D{int(face)}" if positive else f"L{int(seed)}-N"
