"""Mechanism layer above the first-error symptom: truth diagnostics that crossed a tolerance before the symptom.

A precursor is a risk factor, not a cause: successes can carry them too. ``explain`` attaches the earliest precursor
that can explain the symptom (``RELEVANT``) and happened at or before it; the counterfactual oracle re-runs
(``experiments.counterfactual``) are what confirm it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

MECHANISMS = (
	"calibration_drift",
	"brick_pose_error",
	"brick_yaw_error",
	"bowl_pose_error",
	"view_belief_error",
	"brick_toppled",
)
# Chosen on the open development calibration examples. The failed-grasp window excludes intentional carries.
TOPPLE_TILT_DEG = 45.0
TOPPLE_RISE_MM = 10.0
POSE_MECHANISMS = ("brick_pose_error", "brick_yaw_error", "bowl_pose_error", "view_belief_error")
DEFAULT_THRESHOLDS = {
	"cam_pos_mm": 10.0,
	"cam_axis_deg": 1.0,
	"brick_xy_mm": 8.0,
	"brick_yaw_deg": 10.0,
	"bowl_xy_mm": 8.0,
	"view_belief_mm": 8.0,
}
THRESHOLDS_FILE = Path(__file__).resolve().parents[3] / "config" / "experiments" / "precursor_thresholds.json"
RELEVANT = {
	"grasp_or_path_failure": ("brick_pose_error", "brick_yaw_error"),
	"recovery_failure": ("brick_pose_error", "brick_yaw_error"),
	"placement_failure": ("bowl_pose_error", "brick_pose_error"),
	"evidence_insufficient": ("brick_yaw_error", "view_belief_error", "brick_pose_error"),
	"target_face_not_in_view": ("brick_yaw_error", "view_belief_error", "brick_pose_error"),
	"premature_stop": ("brick_yaw_error", "view_belief_error", "brick_pose_error"),
	"crop_error": ("view_belief_error", "brick_pose_error", "brick_yaw_error"),
	"occlusion": ("view_belief_error", "brick_pose_error", "brick_yaw_error"),
}


def load_thresholds(path: Path | None = None) -> dict[str, float]:
	"""Defaults, overridden by the calibrated file (``calibrate_precursors.py``) when it exists."""
	out = dict(DEFAULT_THRESHOLDS)
	p = Path(path) if path else THRESHOLDS_FILE
	if p.is_file():
		data = json.loads(p.read_text(encoding="utf-8"))
		out.update({k: float(v["threshold"]) for k, v in (data.get("metrics") or {}).items() if k in out})
	return out


def _hit(mechanism: str, event: dict, metric: str, value, threshold: float) -> dict:
	return {
		"mechanism": mechanism,
		"seq": int(event.get("seq", 0)),
		"frame": event.get("frame"),
		"metric": metric,
		"value": float(value),
		"threshold": float(threshold),
	}


def brick_topple(track: np.ndarray | None, start: int, end: int) -> dict | None:
	"""Detect overturning or a bounce during a failed grasp from a dense xyzw replay."""
	if track is None or track.shape[1] < 7 or not 0 <= start < end <= len(track):
		return None
	# Replay index 0 is control frame 1. The frame at grasp start is the pre-contact reference.
	window = track[max(start - 1, 0) : end]
	q = window[:, 3:7]
	lengths = np.linalg.norm(q, axis=1)
	valid = np.isfinite(window[:, :7]).all(axis=1) & (lengths > 1e-8)
	if not valid[0]:
		return None
	x, y, z, w = (q[valid] / lengths[valid, None]).T
	# World up in brick coordinates, independent of the mesh's resting orientation and of yaw.
	up = np.column_stack((2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)))
	tilt = np.degrees(np.arccos(np.clip(up @ up[0], -1, 1)))
	rise = (window[valid, 2] - window[0, 2]) * 1000
	hits = np.flatnonzero((tilt >= TOPPLE_TILT_DEG) | (rise >= TOPPLE_RISE_MM))
	if not len(hits):
		return None
	first = int(hits[0])
	return {
		"mechanism": "brick_toppled",
		"frame": max(start, 1) + int(np.flatnonzero(valid)[first]),
		"grasp_start": start,
		"grasp_end": end,
		"max_tilt_change_deg": round(float(tilt.max()), 3),
		"max_rise_mm": round(float(rise.max()), 3),
		"tilt_threshold_deg": TOPPLE_TILT_DEG,
		"rise_threshold_mm": TOPPLE_RISE_MM,
	}


def precursors(
	events: list[dict], thresholds: dict | None = None, *, camera_condition: str | None = None
) -> list[dict]:
	"""Collect threshold crossings. Calibration drift requires a drift condition or legacy nominal-rig evidence."""
	t = thresholds or load_thresholds()
	out: list[dict] = []
	for e in events:
		kind = e.get("kind")
		if kind == "camera_calibrated":
			condition = camera_condition or e.get("camera_condition")
			is_drift = condition == "drift" if condition else e.get("calibrated_on") == "nominal"
			if not is_drift:
				continue
			pos, axis = e.get("actual_cam_pos_err_mm"), e.get("actual_cam_axis_err_deg")
			if pos is not None and pos > t["cam_pos_mm"]:
				out.append(_hit("calibration_drift", e, "cam_pos_mm", pos, t["cam_pos_mm"]))
			elif axis is not None and axis > t["cam_axis_deg"]:
				out.append(_hit("calibration_drift", e, "cam_axis_deg", axis, t["cam_axis_deg"]))
		elif kind == "brick_perceived" and e.get("found"):
			xy, yaw = e.get("error_xy_mm"), e.get("error_yaw_deg")
			if xy is not None and xy > t["brick_xy_mm"]:
				out.append(_hit("brick_pose_error", e, "brick_xy_mm", xy, t["brick_xy_mm"]))
			if yaw is not None and yaw > t["brick_yaw_deg"]:
				out.append(_hit("brick_yaw_error", e, "brick_yaw_deg", yaw, t["brick_yaw_deg"]))
		elif kind == "bowls_perceived":
			errs = [v for v in (e.get("error_xy_mm") or {}).values() if v is not None]
			if errs and max(errs) > t["bowl_xy_mm"]:
				out.append(_hit("bowl_pose_error", e, "bowl_xy_mm", max(errs), t["bowl_xy_mm"]))
		elif kind == "view_truth":
			belief = e.get("belief_error_mm")
			if belief is not None and belief > t["view_belief_mm"]:
				out.append(_hit("view_belief_error", e, "view_belief_mm", belief, t["view_belief_mm"]))
	return out


def explain(symptom: str, symptom_seq: int, found: list[dict]) -> dict:
	relevant = RELEVANT.get(symptom, ())
	cands = [p for p in found if p["mechanism"] in relevant and p["seq"] <= symptom_seq]
	best = min(cands, key=lambda p: (p["seq"], relevant.index(p["mechanism"]))) if cands else None
	drift = best and any(p["mechanism"] == "calibration_drift" and p["seq"] <= best["seq"] for p in found)
	upstream = "calibration_drift" if best and drift and best["mechanism"] in POSE_MECHANISMS else None
	return {"mechanism": best["mechanism"] if best else None, "mechanism_evidence": best, "upstream": upstream}
