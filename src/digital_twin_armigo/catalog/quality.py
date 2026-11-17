"""Quality checks, label normalization, and deterministic scene grouping."""

import hashlib
import json
import math
import re
from collections.abc import Mapping

PASS, FAIL, UNKNOWN = "PASS", "FAIL", "UNKNOWN"
_DEFAULTS = {
	"unintended_force_N": 3,
	"jaw_environment_N": 5,
	"release_speed_mps": 1.2,
	"near_frame_cap": 890,
	"pinch_force_N": None,
	"torque_saturation_seconds": 1.5,
}


def _get(obj, key, default=None):
	if obj is None:
		return default
	if isinstance(obj, Mapping):
		return obj.get(key, default)
	return getattr(obj, key, default)


def _num(value):
	if value is None or isinstance(value, bool):
		return None
	if not isinstance(value, (int, float)):
		return None
	if isinstance(value, float) and not math.isfinite(value):
		return None
	return value if value >= 0 else None


def evaluate_quality(health, info, rules=None):
	def limit(key):
		return _num(_get(rules, key, _DEFAULTS[key]))

	def check(name, raw, key, unit, source, strict=False, force_unknown=False):
		value, cap = _num(raw), limit(key)
		ok = value is not None and cap is not None and (value < cap if strict else value <= cap)
		status = UNKNOWN if force_unknown or value is None or cap is None else (PASS if ok else FAIL)
		return {"name": name, "status": status, "value": value, "limit": cap, "unit": unit, "evidence_source": source}

	refs = _get(health, "evidence_refs")
	measured = _get(health, "torque_evidence_source") == "solver_measured"
	return [
		check(
			"max_unintended_force_N",
			_get(health, "max_unintended_force_N"),
			"unintended_force_N",
			"N",
			"health.max_unintended_force_N",
		),
		check(
			"jaw_env_contact_max_N",
			_get(info, "jaw_env_contact_max_N"),
			"jaw_environment_N",
			"N",
			"info.jaw_env_contact_max_N",
		),
		check(
			"release_peak_speed_mps",
			_get(info, "release_peak_speed_mps"),
			"release_speed_mps",
			"m/s",
			"info.release_peak_speed_mps",
		),
		check("n_frames", _get(info, "n_frames"), "near_frame_cap", "frames", "info.n_frames", strict=True),
		check(
			"pinch_normal_peak_N",
			_get(health, "recorded_substep_pinch_normal_peak_N", _get(health, "pinch_normal_peak_N")),
			"pinch_force_N",
			"N",
			"health.recorded_substep_pinch_normal_peak_N"
			if _get(health, "recorded_substep_pinch_normal_peak_N") is not None
			else "health.pinch_normal_peak_N_sampling_unverified",
			force_unknown=_get(health, "recorded_substep_pinch_normal_peak_N") is None,
		),
		check(
			"max_torque_overload_s",
			_get(refs, "max_torque_overload_s", _get(health, "max_stall_duration_s")),
			"torque_saturation_seconds",
			"s",
			"health.evidence_refs.max_torque_overload_s",
			force_unknown=not measured,
		),
	]


def quality_status(checks):
	statuses = {str(_get(c, "status", "")).upper() for c in checks}
	if not statuses:
		return UNKNOWN
	if FAIL in statuses:
		return FAIL
	if UNKNOWN in statuses:
		return UNKNOWN
	return PASS


_LABEL_MAPS = {
	"mujoco": {"defect_px": "defect_visible_pixels", "brick_px": "brick_visible_pixels"},
	"isaac": {"defect_px": "defect_visibility_proxy_score", "brick_px": "brick_color_mask_pixels"},
}
_AMBIGUOUS = ("defect_px", "brick_px")
_LABEL_META = {
	"defect_visible_pixels": ("visible_pixel_count", "px"),
	"brick_visible_pixels": ("visible_pixel_count", "px"),
	"defect_visibility_proxy_score": ("visibility_proxy", "score"),
	"brick_color_mask_pixels": ("color_mask_count", "px"),
}


def label_definitions(source):
	src = str(source).lower()
	if src not in _LABEL_MAPS:
		raise ValueError(f"unknown label source: {source!r}")
	out = {"phase": {"canonical": "phase", "method": "phase_annotation", "units": "label", "privileged": True}}
	for old, new in _LABEL_MAPS[src].items():
		method, units = _LABEL_META[new]
		out[old] = {"canonical": new, "method": method, "units": units, "privileged": True}
		out[old]["source_camera"] = "top_rgb_v2"
	if src == "isaac":
		out["defect_px"].update(
			method="rectified_view_cosine_times_60_truncated_gated_by_orange_pixels_gt20",
			value_range=[0, 60],
			pixel_count=False,
		)
		out["brick_px"]["method"] = "opencv_hsv_h_lt22_s_gt110_v_gt90"
	return out


def _definition_map(definitions):
	if not definitions:
		return {}
	if not isinstance(definitions, Mapping):
		raise TypeError("definitions must be a mapping")
	result = {}
	for label, definition in definitions.items():
		if isinstance(definition, str):
			result[str(label)] = definition
		elif isinstance(definition, Mapping):
			if "source_label" in definition:
				result[str(definition["source_label"])] = str(label)
			target = definition.get("canonical", definition.get("target", definition.get("name")))
			if target:
				result[str(label)] = str(target)
	return result


def _labels_in(arrays):
	if isinstance(arrays, Mapping):
		return list(arrays)
	if isinstance(arrays, (list, tuple)):
		return sorted({k for row in arrays if isinstance(row, Mapping) for k in row})
	raise TypeError("arrays must be a mapping or a sequence of mappings")


def normalize_labels(source, arrays, definitions=None):
	src = str(source).lower()
	mapping = _definition_map(definitions)
	if src in _LABEL_MAPS:
		mapping = {**_LABEL_MAPS[src], **mapping}
	labels = _labels_in(arrays)
	unresolved = [x for x in labels if x in _AMBIGUOUS and x not in mapping]
	if src not in _LABEL_MAPS and unresolved:
		raise ValueError(f"ambiguous labels for unknown source {source!r}: {unresolved}")

	def rename(row):
		result = {}
		for label, value in row.items():
			target = "phase" if label == "phase" else mapping.get(label, label)
			if target in result:
				raise ValueError(f"label collision: {target!r}")
			result[target] = value
		return result

	if isinstance(arrays, Mapping):
		return rename(arrays)
	return [rename(row) if isinstance(row, Mapping) else row for row in arrays]


_REQUIRED = ("green", "pink", "brick_xy", "brick_yaw_deg", "brick_size", "clutter", "distractors", "markers")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_IGNORED = {"color", "colors", "label", "labels"}


def _clean(value):
	if value is None or isinstance(value, (bool, int, str)):
		return value
	if isinstance(value, float):
		if not math.isfinite(value):
			raise ValueError("non-finite float")
		return round(value, 9)
	if isinstance(value, Mapping):
		return {str(k): _clean(v) for k, v in value.items() if str(k).lower() not in _IGNORED}
	if isinstance(value, (list, tuple)):
		return [_clean(v) for v in value]
	raise TypeError("unsupported layout value")


def _entry(value, count, marker=False):
	if isinstance(value, Mapping):
		items = [(k, v) for k, v in value.items() if str(k).lower() not in _IGNORED]
		if marker:
			items = [(k, v) for k, v in items if str(k).lower() != "id"][:4]
		elif count == 3:
			xyaw = [(k, v) for k, v in items if str(k).lower() in {"x", "y", "yaw"}]
			items = xyaw if len(xyaw) == 3 else items[:3]
		else:
			items = items[:count]
		return _clean(dict(items))
	if isinstance(value, (list, tuple)):
		values = list(value)
		return _clean(values[1:5] if marker else values[:count])
	raise TypeError("layout entry must be a mapping or sequence")


def layout_group(scene):
	group_id = _get(scene, "layout_group_id")
	if isinstance(group_id, str) and _HEX64.fullmatch(group_id):
		return group_id
	layout = _get(scene, "layout")
	if not isinstance(layout, Mapping):
		return None
	if any(key not in layout or layout[key] is None for key in _REQUIRED):
		return None
	try:
		canonical = {key: _clean(layout[key]) for key in _REQUIRED[:5]}
		canonical["clutter"] = sorted(
			[_entry(item, 3) for item in layout["clutter"]], key=lambda v: json.dumps(v, sort_keys=True)
		)
		canonical["distractors"] = sorted(
			[_entry(item, 6) for item in layout["distractors"]], key=lambda v: json.dumps(v, sort_keys=True)
		)
		canonical["markers"] = sorted(
			[_entry(item, 4, marker=True) for item in layout["markers"]], key=lambda v: json.dumps(v, sort_keys=True)
		)
		text = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
	except (TypeError, ValueError):
		return None
	return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_for_group(group, seed=0):
	digest = hashlib.sha256(f"{seed}:{group}".encode()).digest()
	point = int.from_bytes(digest[:8], "big") / 2**64
	if point < 0.8:
		return "train"
	return "validation" if point < 0.9 else "test"
