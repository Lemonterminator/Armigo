"""Visibility-constrained simulation defect oracle and planner context anti-leakage boundary.

Defect labels returned by :func:`inspect_object_sim_oracle` are strictly labeled ``source="sim_oracle"``
and do not validate real-world vision defect detection. Hidden defect metadata, asset filenames, and
target bowl oracle fields must never leak into planner inputs.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from .presentation import FACES, view_cosine

FORBIDDEN_PLANNER_KEYS = frozenset(
	{
		"defective",
		"defect",
		"defect_face",
		"defect_px",
		"defect_geoms",
		"defect_normal",
		"target_bowl",
		"expected_bowl",
		"oracle_label",
	}
)

_FORBIDDEN_TEXT_PATTERNS = (
	re.compile(r"_defective(?:\.usda)?", re.IGNORECASE),
	re.compile(r"\bdefect_[0-9]+\b", re.IGNORECASE),
	re.compile(r"\bprobe_group\b", re.IGNORECASE),
	re.compile(r"\bplace it in the (?:pink|green) bowl\b", re.IGNORECASE),
)

_NEUTRAL_ASSET_RE = re.compile(r"(duplo_\d+x\d+)_(?:defective|orange)(?:\.usda)?", re.IGNORECASE)


class DefectLeakageError(ValueError):
	"""Raised when hidden simulator defect ground truth leaks into a planner context."""


@dataclass(frozen=True)
class InspectionResult:
	"""Outcome of a single visibility-gated ``sim_oracle`` inspection query."""

	status: str  # "KNOWN" | "UNKNOWN"
	label: str | None  # "defective" | "normal" | None
	source: str  # "sim_oracle"
	visibility_ok: bool
	visible_faces: tuple[int, ...]
	observed_faces: tuple[int, ...]
	reason: str
	disclaimer: str = "sim_oracle_visibility_gated_not_real_vision_detector"


def _neutralize_asset_name(value: str) -> str:
	m = _NEUTRAL_ASSET_RE.search(value)
	if m:
		return m.group(1).lower()
	return value


def assert_no_defect_leakage(payload: Any, *, _path: str = "root") -> None:
	"""Verify recursively that ``payload`` contains no hidden defect metadata or leaked asset identifiers."""
	if isinstance(payload, Mapping):
		for key, val in payload.items():
			key_str = str(key)
			if key_str.lower() in FORBIDDEN_PLANNER_KEYS:
				raise DefectLeakageError(f"Forbidden oracle key {key_str!r} leaked at {_path}.{key_str}")
			if key_str.lower() == "task" and isinstance(val, str):
				for pat in _FORBIDDEN_TEXT_PATTERNS:
					if pat.search(val):
						raise DefectLeakageError(f"Oracle task hint leaked at {_path}.{key_str}: {val!r}")
			assert_no_defect_leakage(val, _path=f"{_path}.{key_str}")
	elif isinstance(payload, (list, tuple)):
		for idx, item in enumerate(payload):
			assert_no_defect_leakage(item, _path=f"{_path}[{idx}]")
	elif isinstance(payload, str):
		for pat in _FORBIDDEN_TEXT_PATTERNS:
			if pat.search(payload):
				raise DefectLeakageError(f"Forbidden oracle identifier matched {pat.pattern!r} at {_path}: {payload!r}")


def sanitize_scene_for_planner(raw_scene: Mapping[str, Any]) -> dict[str, Any]:
	"""Strip hidden defect labels, bowl oracle hints, and raw USD defect filenames from scene observations."""
	clean: dict[str, Any] = {}
	for key, val in raw_scene.items():
		k_low = str(key).lower()
		if k_low in FORBIDDEN_PLANNER_KEYS:
			continue
		if k_low == "task":
			continue
		if k_low in ("asset_usd", "asset_path", "usd_path", "brick_usd"):
			clean["asset_id"] = _neutralize_asset_name(str(val))
			continue
		if isinstance(val, Mapping):
			clean[str(key)] = sanitize_scene_for_planner(val)
		elif isinstance(val, str):
			clean[str(key)] = _neutralize_asset_name(val)
		else:
			clean[str(key)] = val
	assert_no_defect_leakage(clean)
	return clean


def visible_brick_faces(
	brick_pos: np.ndarray,
	brick_rot: np.ndarray,
	cam_pos: np.ndarray,
	cam_forward: np.ndarray,
	min_view_cosine: float = 0.25,
) -> tuple[int, ...]:
	"""Return indices in :data:`FACES` whose outward normal faces ``cam_pos`` within ``min_view_cosine``."""
	pos = np.asarray(brick_pos, dtype=np.float64)
	rot = np.asarray(brick_rot, dtype=np.float64).reshape(3, 3)
	cpos = np.asarray(cam_pos, dtype=np.float64)
	cfwd = np.asarray(cam_forward, dtype=np.float64)
	cfwd = cfwd / max(float(np.linalg.norm(cfwd)), 1e-12)
	to_obj = pos - cpos
	dist = float(np.linalg.norm(to_obj))
	if dist <= 1e-6 or float(np.dot(to_obj / dist, cfwd)) <= 0.15:
		return ()

	seen: list[int] = []
	for face_idx, (axis, sign) in enumerate(FACES):
		local_n = np.zeros(3, dtype=np.float64)
		local_n[axis] = float(sign)
		world_n = rot @ local_n
		if view_cosine(world_n, pos, cpos) >= min_view_cosine:
			seen.append(face_idx)
	return tuple(seen)


def inspect_object_sim_oracle(
	*,
	defective_gt: bool,
	defect_face_gt: int,
	brick_pos: np.ndarray,
	brick_rot: np.ndarray,
	cam_pos: np.ndarray,
	cam_forward: np.ndarray,
	in_crop: bool,
	visible_fraction: float,
	min_visible_fraction: float = 0.50,
	min_dist_m: float = 0.15,
	max_dist_m: float = 0.70,
	min_view_cosine: float = 0.25,
	prior_observed_faces: Iterable[int] = (),
	require_all_faces_for_normal: bool = True,
) -> InspectionResult:
	"""Return a defect label ONLY when visibility, distance, occlusion, and face orientation permit."""
	prior = {int(f) for f in prior_observed_faces}
	dist = float(np.linalg.norm(np.asarray(brick_pos, dtype=np.float64) - np.asarray(cam_pos, dtype=np.float64)))
	if not in_crop or dist < min_dist_m or dist > max_dist_m:
		return InspectionResult(
			status="UNKNOWN",
			label=None,
			source="sim_oracle",
			visibility_ok=False,
			visible_faces=(),
			observed_faces=tuple(sorted(prior)),
			reason="out_of_view_or_range",
		)
	if not np.isfinite(visible_fraction) or visible_fraction < min_visible_fraction:
		return InspectionResult(
			status="UNKNOWN",
			label=None,
			source="sim_oracle",
			visibility_ok=False,
			visible_faces=(),
			observed_faces=tuple(sorted(prior)),
			reason="not_visible_or_occluded",
		)

	vis_faces = visible_brick_faces(brick_pos, brick_rot, cam_pos, cam_forward, min_view_cosine=min_view_cosine)
	obs_all = tuple(sorted(prior | set(vis_faces)))
	if not vis_faces:
		return InspectionResult(
			status="UNKNOWN",
			label=None,
			source="sim_oracle",
			visibility_ok=False,
			visible_faces=(),
			observed_faces=obs_all,
			reason="no_face_oriented_to_camera",
		)

	if defective_gt:
		if int(defect_face_gt) in vis_faces or int(defect_face_gt) in prior:
			return InspectionResult(
				status="KNOWN",
				label="defective",
				source="sim_oracle",
				visibility_ok=True,
				visible_faces=vis_faces,
				observed_faces=obs_all,
				reason="defect_face_visible",
			)
		return InspectionResult(
			status="UNKNOWN",
			label=None,
			source="sim_oracle",
			visibility_ok=True,
			visible_faces=vis_faces,
			observed_faces=obs_all,
			reason="defect_face_not_visible",
		)

	# Normal brick: confirm normal only if all faces (or non-require-all mode) have been visibly inspected
	if not require_all_faces_for_normal or len(obs_all) == len(FACES):
		return InspectionResult(
			status="KNOWN",
			label="normal",
			source="sim_oracle",
			visibility_ok=True,
			visible_faces=vis_faces,
			observed_faces=obs_all,
			reason="all_faces_inspected_normal" if require_all_faces_for_normal else "visible_surface_normal",
		)
	return InspectionResult(
		status="UNKNOWN",
		label=None,
		source="sim_oracle",
		visibility_ok=True,
		visible_faces=vis_faces,
		observed_faces=obs_all,
		reason="uninspected_faces_remaining",
	)
