"""Mechanism layer above the first-error symptom: truth diagnostics that crossed a tolerance before the symptom.

A precursor is a risk factor, not a cause: successes can carry them too. ``explain`` attaches the earliest precursor
that can explain the symptom (``RELEVANT``) and happened at or before it; the counterfactual oracle re-runs
(``experiments.counterfactual``) are what confirm it.
"""

from __future__ import annotations

import json
from pathlib import Path

MECHANISMS = ("calibration_drift", "brick_pose_error", "brick_yaw_error", "bowl_pose_error", "view_belief_error")
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


def precursors(events: list[dict], thresholds: dict | None = None) -> list[dict]:
	t = thresholds or load_thresholds()
	out: list[dict] = []
	for e in events:
		kind = e.get("kind")
		if kind == "camera_calibrated":
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
	drift = any(p["mechanism"] == "calibration_drift" for p in found)
	upstream = "calibration_drift" if best and drift and best["mechanism"] in POSE_MECHANISMS else None
	return {"mechanism": best["mechanism"] if best else None, "mechanism_evidence": best, "upstream": upstream}
