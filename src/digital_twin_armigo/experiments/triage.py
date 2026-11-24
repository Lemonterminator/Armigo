"""Automatic failure triage rules (R1-R8) for agent episodes (read-only, zero simulation).

Each rule in :data:`RULES` is a pure function ``(events, result) -> dict | None`` returning an evidence dict when
matched. :func:`triage_episode` reads ``events.jsonl`` and ``episode.json`` (plus optional sibling metadata for
camera condition, expected label, or brick symmetry) and evaluates R1-R8 in priority order.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from .attribution import (
	BOTTOM_MIN_PX,
	CROP_KEEP_RATIO,
	CROSS_MIN_PX,
	INFRA_CODES,
	LABELS,
	MIN_BRICK_PX,
	WRIST_MIN_PX,
	active_views,
	collect_views,
)
from .scoring import _events, _json

YAW_ERR_THRESHOLD_DEG = 10.0
FILL_MIN_PIXELS = 500
UPRIGHT_Z_DOT_MIN = 0.9

_GRASP_CARRY_STAGES = frozenset(
	{"grasp", "carry", "lift", "carry_to_bowl", "place", "present_bottom", "return_to_lift", "park", "recover"}
)
_CARRY_SEGMENTS = frozenset({"carry", "lift", "carry_to_bowl", "return_to_lift", "park"})
_R6_CODES = frozenset({"no_clamp", "BRICK_LOST", "brick_lost", "grasp_slip", "pick_miss"})
_R8_EVENT_KINDS = frozenset({"detector_error", "tool_error", "tool_exception", "contract_error", "timeout"})
_R8_CODE_SUBSTRINGS = ("TIMEOUT", "EXCEPTION", "CONTRACT", "SCHEMA", "TOOL_ERROR", "TOOL_EXCEPTION", "NONFINITE")


@dataclass(frozen=True)
class Rule:
	id: str
	root_cause: str
	matcher: Callable[[list[dict], dict], dict | None]


@dataclass(frozen=True)
class TriageRow:
	traj_id: str
	stage: str
	code: str | None
	root_cause: str
	evidence: dict
	rule: str

	def to_dict(self) -> dict:
		return asdict(self)


def _quat_to_yaw_deg(q) -> float:
	x, y, z, w = (float(v) for v in q)
	return math.degrees(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _is_upright_quat(q) -> bool:
	x, y, _, _ = (float(v) for v in q)
	return (1.0 - 2.0 * (x * x + y * y)) > UPRIGHT_Z_DOT_MIN


def _sym_yaw_diff_deg(est_deg: float, truth_deg: float, symmetry_deg: float) -> float:
	d = (est_deg - truth_deg) % symmetry_deg
	return min(d, symmetry_deg - d)


def _symmetry_deg(events: list[dict], result: dict) -> float:
	if result.get("symmetry_deg") is not None:
		return float(result["symmetry_deg"])
	for src in (result, *events):
		size = src.get("brick_size") or src.get("brick_size_m")
		if isinstance(size, (list, tuple)) and len(size) >= 2:
			return 90.0 if abs(float(size[0]) - float(size[1])) < 1e-6 else 180.0
		if src.get("symmetry_deg") is not None:
			return float(src["symmetry_deg"])
		sc = src.get("size_class")
		if sc == "2x2":
			return 90.0
		if sc in ("2x3", "2x4"):
			return 180.0
	return 180.0


def _failure_records(events: list[dict], result: dict) -> list[dict]:
	"""Collect failure dicts with inferred ``stage`` from preceding ``segment`` events and ``result.failures``."""
	records: list[dict] = []
	last_seg_by_branch: dict[int | None, str] = {}
	seen: set[tuple[str | None, int | None]] = set()
	for ev in events:
		kind = ev.get("kind")
		branch = ev.get("branch")
		if kind == "segment":
			seg = ev.get("segment_kind")
			if isinstance(seg, str):
				last_seg_by_branch[branch] = seg
		elif kind == "failure":
			seg_kind = last_seg_by_branch.get(branch) or last_seg_by_branch.get(None)
			stage = ev.get("stage")
			if not stage:
				if seg_kind == "grasp":
					stage = "grasp"
				elif seg_kind in _CARRY_SEGMENTS:
					stage = "carry"
				elif ev.get("skill"):
					stage = ev["skill"]
				elif ev.get("node") and ev.get("node") != "recover":
					stage = ev["node"]
				else:
					stage = seg_kind or ev.get("node")
			code = ev.get("code")
			frame = ev.get("frame")
			records.append(
				{
					"code": code,
					"stage": stage,
					"skill": ev.get("skill"),
					"node": ev.get("node"),
					"segment_kind": seg_kind,
					"frame": frame,
				}
			)
			seen.add((code, frame))
	for f in result.get("failures") or []:
		if not isinstance(f, dict):
			continue
		code = f.get("code")
		frame = f.get("frame")
		if (code, frame) in seen:
			continue
		stage = f.get("stage") or f.get("skill") or f.get("node")
		records.append(
			{
				"code": code,
				"stage": stage,
				"skill": f.get("skill"),
				"node": f.get("node"),
				"segment_kind": None,
				"frame": frame,
			}
		)
	return records


def _bowl_name(value: str | None) -> str | None:
	if value in ("green", "green_bowl"):
		return "green"
	if value in ("pink", "pink_bowl"):
		return "pink"
	return None


def _expected_label(result: dict) -> str | None:
	exp = result.get("expected_label")
	if exp in LABELS:
		return exp
	label = result.get("label")
	if result.get("label_correct") is True and label in LABELS:
		return label
	if result.get("label_correct") is False and label in LABELS:
		return "defective" if label == "normal" else "normal"
	return None


def _rule_r1(events: list[dict], result: dict) -> dict | None:
	for rec in _failure_records(events, result):
		if rec["code"] == "BOWL_NOT_FOUND":
			stage = rec["stage"] if rec["stage"] and rec["stage"] != "recover" else "prepare"
			return {"code": "BOWL_NOT_FOUND", "stage": stage, "frame": rec["frame"]}
	for key in ("code", "execution_error", "error"):
		if result.get(key) == "BOWL_NOT_FOUND":
			return {"code": "BOWL_NOT_FOUND", "stage": "prepare", "frame": None}
	return None


def _rule_r2(events: list[dict], result: dict) -> dict | None:
	view_truths = [ev for ev in events if ev.get("kind") == "view_truth"]
	if not view_truths:
		return None
	sym = _symmetry_deg(events, result)
	perceived = [ev for ev in events if ev.get("kind") == "brick_perceived" and ev.get("found") and "yaw_deg" in ev]

	best: dict | None = None
	for vt in view_truths:
		err: float | None = None
		tq = vt.get("true_brick_quat_xyzw")
		bq = vt.get("belief_brick_quat_xyzw")
		if isinstance(tq, (list, tuple)) and len(tq) == 4 and isinstance(bq, (list, tuple)) and len(bq) == 4:
			if _is_upright_quat(tq):
				err = _sym_yaw_diff_deg(_quat_to_yaw_deg(bq), _quat_to_yaw_deg(tq), sym)
		elif "brick_yaw_deg" in vt:
			true_yaw = float(vt["brick_yaw_deg"])
			bel_yaw = vt.get("belief_yaw_deg") if "belief_yaw_deg" in vt else vt.get("yaw_deg")
			if bel_yaw is None and perceived:
				vt_frame = vt.get("frame")
				match = next((p for p in reversed(perceived) if vt_frame is None or p.get("frame") == vt_frame), None)
				if match is None and vt_frame is not None:
					prior = [p for p in perceived if p.get("frame") is not None and p["frame"] <= vt_frame]
					match = prior[-1] if prior else None
				if match is not None:
					bel_yaw = match["yaw_deg"]
			if bel_yaw is not None:
				err = _sym_yaw_diff_deg(float(bel_yaw), true_yaw, sym)
		if err is not None and err > YAW_ERR_THRESHOLD_DEG and (best is None or err > best["yaw_err_deg"]):
			best = {
				"yaw_err_deg": round(err, 3),
				"symmetry_deg": sym,
				"frame": vt.get("frame"),
				"view_id": vt.get("id"),
			}
	return best


def _is_drift_camera(events: list[dict], result: dict) -> bool:
	if result.get("camera_condition") == "drift":
		return True
	for ev in events:
		if ev.get("kind") == "camera_calibrated" and (
			ev.get("camera_condition") == "drift" or ev.get("calibrated_on") == "nominal"
		):
			return True
	return False


def _rule_r3(events: list[dict], result: dict) -> dict | None:
	if not _is_drift_camera(events, result):
		return None
	for rec in _failure_records(events, result):
		if rec["code"] == "no_clamp" and rec["stage"] == "grasp":
			return {
				"camera_condition": "drift",
				"code": "no_clamp",
				"stage": "grasp",
				"frame": rec["frame"],
			}
	return None


def _rule_r4(events: list[dict], result: dict) -> dict | None:
	has_unverified = any(rec["code"] == "PLACEMENT_UNVERIFIED" for rec in _failure_records(events, result)) or any(
		result.get(k) == "PLACEMENT_UNVERIFIED" for k in ("code", "outcome", "execution_error")
	)
	if not has_unverified:
		return None

	target = _bowl_name(result.get("expected_bowl") or result.get("target_bowl"))
	if target is None:
		label = result.get("label")
		if label == "normal":
			target = "green"
		elif label == "defective":
			target = "pink"
	if target is None:
		for ev in reversed(events):
			if ev.get("kind") == "segment" and ev.get("segment_kind") == "carry_to_bowl":
				target = _bowl_name((ev.get("params") or {}).get("bowl"))
				if target is not None:
					break

	actual = _bowl_name(result.get("actual_destination") or result.get("destination") or result.get("physical_outcome"))
	if actual is None:
		for ev in reversed(events):
			if ev.get("kind") == "placement_check":
				fill = ev.get("fill_pixels") or {}
				filled = [b for b in ("green", "pink") if (fill.get(b) or 0) >= FILL_MIN_PIXELS]
				if len(filled) == 1:
					actual = filled[0]
				elif target is not None:
					other = "pink" if target == "green" else "green"
					if (fill.get(other) or 0) >= FILL_MIN_PIXELS:
						actual = other
				break

	if actual in ("green", "pink") and (target is None or actual != target):
		return {
			"code": "PLACEMENT_UNVERIFIED",
			"target_bowl": target,
			"actual_bowl": actual,
		}
	return None


def _rule_r5(events: list[dict], result: dict) -> dict | None:
	verdict = result.get("verdict_final_cycle") or result.get("verdict")
	expected = _expected_label(result)
	label = result.get("label")
	eligible = (
		verdict == "evidence_insufficient"
		or result.get("label_correct") is True
		or (expected in LABELS and label == expected)
	)
	if not eligible:
		return None

	views = active_views(collect_views(events), result)
	target_label = expected or (label if label in LABELS else None)

	if target_label == "defective":
		deciding = next((v for v in views if v.id == result.get("placement_evidence")), None) or next(
			(v for v in views if v.above()), None
		)
		if deciding is not None and deciding.truth:
			cross = (
				deciding.truth.get("true_cross_px_crop_native")
				if "true_cross_px_crop_native" in deciding.truth
				else deciding.truth.get("true_cross_px")
			)
			if cross is not None and int(cross) < CROSS_MIN_PX:
				return {"view_id": deciding.id, "true_cross_px": int(cross)}
		return None

	covered: set[int] = set()
	have_truth = False
	if views:
		for v in views:
			faces = v.true_faces()
			if faces is not None:
				have_truth = True
				covered.update(faces)
	else:
		for ev in events:
			if ev.get("kind") != "view_truth":
				continue
			if ev.get("missing_faces"):
				return {"missing_faces": sorted(int(f) for f in ev["missing_faces"])}
			px = ev.get("true_face_pixels")
			if isinstance(px, list):
				have_truth = True
				is_top = ev.get("camera") == "top" or str(ev.get("id", "")).endswith(":top")
				floor = BOTTOM_MIN_PX if is_top else WRIST_MIN_PX
				covered.update(i for i, n in enumerate(px) if n >= floor and ((i == 5) == is_top))

	if have_truth and len(covered) < 6:
		return {
			"covered_faces": sorted(covered),
			"missing_faces": sorted(set(range(6)) - covered),
		}
	return None


def _is_r6_code(code: str | None) -> bool:
	if not code:
		return False
	if code in _R6_CODES:
		return True
	upper = str(code).upper()
	return "COLLISION" in upper or "PATH_BLOCKED" in upper


def _rule_r6(events: list[dict], result: dict) -> dict | None:
	for rec in _failure_records(events, result):
		code = rec["code"]
		stage = rec["stage"]
		if _is_r6_code(code) and (stage is None or stage in _GRASP_CARRY_STAGES):
			out_stage = "grasp" if stage == "grasp" or code == "no_clamp" else "carry"
			return {"code": code, "stage": out_stage, "frame": rec["frame"]}

	for ev in events:
		if ev.get("kind") == "segment_end":
			reason = ev.get("reason")
			if _is_r6_code(reason):
				return {"code": reason, "stage": "carry", "frame": ev.get("frame")}

	health = result.get("health") or {}
	if not isinstance(health, dict):
		return None
	cycle = health.get("final_cycle") if isinstance(health.get("final_cycle"), dict) else {}
	for src in (cycle, health):
		for code in src.get("reason_codes") or []:
			if _is_r6_code(code):
				out_stage = "grasp" if code in ("pick_miss", "no_clamp") else "carry"
				return {"code": code, "stage": out_stage, "frame": None}
		fm = src.get("failure_mode")
		if _is_r6_code(fm):
			out_stage = "grasp" if fm in ("pick_miss", "no_clamp") else "carry"
			return {"code": fm, "stage": out_stage, "frame": None}
		if (src.get("unintended_collision_count") or 0) > 0 or src.get("unintended_collision_frames"):
			return {"code": "COLLISION", "stage": "carry", "frame": None}
	return None


def _rule_r7(events: list[dict], result: dict) -> dict | None:
	label = result.get("label")
	if label not in LABELS:
		return None
	expected = _expected_label(result)
	wrong = result.get("label_correct") is False or (expected in LABELS and label != expected)
	if not wrong:
		return None

	views = active_views(collect_views(events), result)
	if label == "defective":
		deciding = next((v for v in views if v.id == result.get("placement_evidence")), None) or next(
			(v for v in views if v.above()), None
		)
		if deciding is None or not deciding.truth:
			return None
		truth = deciding.truth
		crop = truth.get("true_face_pixels_crop_native")
		frame_px = truth.get("true_face_pixels")
		visible = False
		if crop is not None and frame_px is not None:
			crop_sum, frame_sum = sum(crop), sum(frame_px)
			visible = crop_sum >= MIN_BRICK_PX and (frame_sum == 0 or crop_sum >= CROP_KEEP_RATIO * frame_sum)
		elif frame_px is not None:
			visible = sum(frame_px) >= MIN_BRICK_PX
		elif truth.get("clear") or truth.get("visible"):
			visible = True
		if visible:
			return {
				"root_cause": "false_positive_visible",
				"view_id": deciding.id,
				"score": deciding.score,
			}
		return None

	# label == "normal" on defective brick -> check if defect cross was clearly visible
	for v in views:
		if not v.truth:
			continue
		cross = (
			v.truth.get("true_cross_px_crop_native")
			if "true_cross_px_crop_native" in v.truth
			else v.truth.get("true_cross_px")
		)
		thr = 0.0 if v.threshold is None else float(v.threshold)
		if cross is not None and int(cross) >= CROSS_MIN_PX and v.score is not None and float(v.score) < thr:
			return {
				"root_cause": "false_negative_visible",
				"view_id": v.id,
				"score": v.score,
				"true_cross_px": int(cross),
			}
		if v.truth.get("defect_face_visible") and v.score is not None and float(v.score) < thr:
			return {
				"root_cause": "false_negative_visible",
				"view_id": v.id,
				"score": v.score,
			}
	return None


def _is_r8_code(code: str | None) -> bool:
	if not code:
		return False
	upper = str(code).upper()
	return any(c in upper for c in INFRA_CODES) or any(sub in upper for sub in _R8_CODE_SUBSTRINGS)


def _rule_r8(events: list[dict], result: dict) -> dict | None:
	if result.get("run_state") == "infra_failed":
		return {"code": "INFRA_FAILED", "stage": "tool", "frame": None}

	for ev in events:
		kind = ev.get("kind")
		if kind in _R8_EVENT_KINDS:
			det = ev.get("detector") if isinstance(ev.get("detector"), dict) else {}
			code = ev.get("code") or ev.get("reason") or det.get("reason") or kind.upper()
			return {"code": str(code), "stage": ev.get("node") or "tool", "frame": ev.get("frame")}
		if kind == "view":
			obs = ev.get("observation") or {}
			score = obs.get("score")
			if score is None or not isinstance(score, (int, float)) or not math.isfinite(float(score)):
				return {"code": "NONFINITE_SCORE", "stage": ev.get("node") or "tool", "frame": ev.get("frame")}

	for rec in _failure_records(events, result):
		if _is_r8_code(rec["code"]):
			return {"code": rec["code"], "stage": rec["stage"] or "tool", "frame": rec["frame"]}

	for key in ("execution_error", "error"):
		val = result.get(key)
		if val:
			return {"code": str(val), "stage": "tool", "frame": None}
	return None


RULES: tuple[Rule, ...] = (
	Rule(id="R1", root_cause="bowl_localization", matcher=_rule_r1),
	Rule(id="R2", root_cause="perception_pose_error", matcher=_rule_r2),
	Rule(id="R3", root_cause="calibration_stale", matcher=_rule_r3),
	Rule(id="R4", root_cause="wrong_bowl_carry", matcher=_rule_r4),
	Rule(id="R5", root_cause="evidence_insufficient", matcher=_rule_r5),
	Rule(id="R6", root_cause="grasp_or_path_failure", matcher=_rule_r6),
	Rule(id="R7", root_cause="false_positive_visible", matcher=_rule_r7),
	Rule(id="R8", root_cause="tool_contract", matcher=_rule_r8),
)


def _load_plan_camera(ep_dir: Path) -> str | None:
	for candidate in (ep_dir / "plan.json", ep_dir.parent / "plan.json"):
		plan = _json(candidate)
		if isinstance(plan, dict) and plan.get("camera_condition"):
			return str(plan["camera_condition"])
	return None


def _derive_traj_id(ep_dir: Path, exp: dict) -> str:
	parts = [exp.get("batch"), exp.get("engine"), exp.get("group")]
	if all(parts):
		return "/".join([*parts, ep_dir.name])
	return str(exp.get("case_id") or ep_dir.name)


def _default_stage_and_code(events: list[dict], result: dict) -> tuple[str, str | None]:
	failures = _failure_records(events, result)
	if failures:
		first = failures[0]
		stage = first["stage"] or first["node"] or "unknown"
		return str(stage), first["code"]
	nodes = [
		ev.get("node")
		for ev in events
		if ev.get("kind") == "node" and ev.get("node") not in (None, "recover", "evaluate")
	]
	stage = str(nodes[-1]) if nodes else "unknown"
	code = result.get("execution_error") or result.get("code")
	if not code and isinstance(result.get("health"), dict):
		health = result["health"]
		cycle = health.get("final_cycle") if isinstance(health.get("final_cycle"), dict) else {}
		fm = cycle.get("failure_mode") or health.get("failure_mode")
		if fm not in (None, "ok"):
			code = str(fm)
	return stage, code


def _stage_and_code_for_rule(rule_id: str, ev: dict, events: list[dict], result: dict) -> tuple[str, str | None]:
	default_stage, default_code = _default_stage_and_code(events, result)
	if rule_id == "R1":
		return str(ev.get("stage") or "prepare"), "BOWL_NOT_FOUND"
	if rule_id == "R2":
		return "perception", default_code
	if rule_id == "R3":
		return "grasp", "no_clamp"
	if rule_id == "R4":
		return "place", "PLACEMENT_UNVERIFIED"
	if rule_id == "R5":
		return "inspect", default_code
	if rule_id == "R6":
		return str(ev.get("stage") or default_stage), ev.get("code") or default_code
	if rule_id == "R7":
		return "inspect", default_code or "wrong_label"
	if rule_id == "R8":
		return str(ev.get("stage") or "tool"), ev.get("code") or default_code
	return default_stage, default_code


def triage_episode(ep_dir: Path, *, traj_id: str | None = None) -> TriageRow:
	"""Triage one episode directory (read-only) using rules R1-R8."""
	ep_dir = Path(ep_dir)
	episode = _json(ep_dir / "episode.json") or {}
	result = dict(episode.get("result") or {})
	exp = ((episode.get("provenance") or {}).get("experiment")) or {}
	task = _json(ep_dir / "task.json") or {}
	audit_priv = _json(ep_dir / "audit.private.json") or {}
	run_status = _json(ep_dir / "run_status.json") or {}
	scene = _json(ep_dir / "replay" / "scene.json") or {}
	events = _events(ep_dir / "events.jsonl")

	resolved_traj_id = traj_id or _derive_traj_id(ep_dir, exp)

	# Enrich a shallow copy of result with optional episode metadata so rules stay pure.
	case = task.get("case") or []
	if "camera_condition" not in result:
		cam = exp.get("camera_condition")
		if cam is None and len(case) > 3 and isinstance(case[3], dict):
			cam = case[3].get("cameras")
		if cam is None:
			cam = _load_plan_camera(ep_dir)
		if cam is not None:
			result["camera_condition"] = cam
	if "brick_size" not in result and isinstance(scene.get("layout"), dict) and "brick_size" in scene["layout"]:
		result["brick_size"] = scene["layout"]["brick_size"]
	if "expected_label" not in result:
		exp_lbl = audit_priv.get("true_label")
		if exp_lbl is None and len(case) > 1:
			exp_lbl = "defective" if case[1] else "normal"
		if exp_lbl is not None:
			result["expected_label"] = exp_lbl
	if "run_state" not in result and run_status.get("state"):
		result["run_state"] = run_status["state"]

	verdict = result.get("verdict_final_cycle") or result.get("verdict")
	if verdict == "SUCCESS" and result.get("label_correct") is not False and _rule_r5(events, result) is None:
		return TriageRow(
			traj_id=resolved_traj_id,
			stage="ok",
			code=None,
			root_cause="none",
			evidence={},
			rule="none",
		)

	for rule in RULES:
		matched = rule.matcher(events, result)
		if matched is None:
			continue
		evidence = dict(matched)
		root_cause = str(evidence.pop("root_cause", rule.root_cause))
		stage, code = _stage_and_code_for_rule(rule.id, evidence, events, result)
		return TriageRow(
			traj_id=resolved_traj_id,
			stage=stage,
			code=code,
			root_cause=root_cause,
			evidence=evidence,
			rule=rule.id,
		)

	stage, code = _default_stage_and_code(events, result)
	return TriageRow(
		traj_id=resolved_traj_id,
		stage=stage,
		code=code,
		root_cause="unclassified",
		evidence={},
		rule="unclassified",
	)
