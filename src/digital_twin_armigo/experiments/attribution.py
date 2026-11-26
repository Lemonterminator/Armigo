"""First-error attribution of one episode (pure: takes the parsed episode files, reads nothing).

``first_error(...)`` returns ``{"root_cause": one of ROOT_CAUSES, "frame", "branch", "view_id", "detail"}``. Each rule
below produces *findings* tagged with the event ``seq`` where the error happened; the reported one is the earliest.
"Active views" are the scored views of the final (active) branch, ``result.evidence`` (stale views of rewound branches
are not evidence). A "crop" is the detector input; ``view_truth`` events carry the simulator's true pixel counts for
the whole frame and inside that crop (see ``twin_mujoco._capture``).

Rules
-----
* ``infra_failure``: no ``episode.json``/result, ``run_status`` state ``infra_failed``, a ``detector_error`` event or a
  failure code in ``INFRA_CODES`` (``DETECTOR_UNAVAILABLE``...) on an episode that did not end in SUCCESS.
* ``bowl_localization``: a failed episode with ``BOWL_NOT_FOUND``. This identifies the sensing failure before
  the resulting unknown label or unused inspection budget.
* Label cases (``wrong_label`` and ``no_label`` precede every physical cause):
  - defective brick labelled normal: the first active view whose crop truly held the cross
    (``true_cross_px_crop_native >= CROSS_MIN_PX``) with ``score < threshold`` is ``false_negative_visible``. Else the
    first view where the agent listed the defect face as seen but its true pixels are below the readable minimum is
    ``occlusion``; else the first view where the cross was visible in the frame (``true_cross_px >= CROSS_MIN_PX``) or
    the face lost most of its pixels inside the crop is ``crop_error``; else ``target_face_not_in_view`` (the defect
    face was never truly visible, or only so small that the cross stayed below ``CROSS_MIN_PX``).
  - normal brick labelled defective: the deciding view is the first above-threshold active view
    (``result.placement_evidence``); few brick pixels in the crop, or most of the brick lost to the crop, is
    ``crop_error``, otherwise ``false_positive_visible``.
  - label ``unknown``/missing: failures present and the recovery budget used (rewinds + retries >= ``max_rewinds``) is
    ``recovery_failure``; face coverage incomplete while the scored-view budget remained is ``premature_stop``; else
    ``target_face_not_in_view``.
* Correct label, physical failure (final-cycle verdict not SUCCESS, or the replay puts the brick in the wrong place):
  ``placement_failure`` for ``place_miss``/``PLACEMENT_UNVERIFIED``/wrong bowl (or a brick that moved and ended on the
  table), ``grasp_or_path_failure`` for every other code (PLAN*, *PATH_BLOCKED*, BRICK_LOST, grasp_slip, pick_miss,
  NO_*PATH, torque_stall...) and for a brick that never left its place.
* Success with correct label checks the evidence: defective: the deciding view's crop must hold the cross
  (``>= CROSS_MIN_PX``); normal: the union of truly visible faces over active views must cover all six. Otherwise
  ``evidence_insufficient``; else ``none``. Without ``view_truth`` events (old runs) the agent's own believed faces
  stand in and the detail says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field

BOTTOM_MIN_PX = 400
WRIST_MIN_PX = 800
MIN_BRICK_PX = 150
CROSS_MIN_PX = 25
CROP_KEEP_RATIO = 0.5  # a crop that keeps less than this share of the true face pixels cut the target off

INFRA_CODES = ("DETECTOR_UNAVAILABLE", "NONFINITE_SCORE", "SCORE_SERVER", "LLM_UNAVAILABLE", "SIM_CRASH", "TIMEOUT")
PLACEMENT_CODES = ("place_miss", "PLACEMENT_UNVERIFIED", "wrong_bowl", "wrong_label")
LABELS = ("normal", "defective")


def _constants() -> tuple[int, int, int]:
	try:
		from ..agent import twin_mujoco as tm

		return int(tm.BOTTOM_MIN_PX), int(tm.WRIST_MIN_PX), int(tm.MIN_BRICK_PX)
	except (ImportError, OSError):  # mujoco missing: the recorded defaults
		return BOTTOM_MIN_PX, WRIST_MIN_PX, MIN_BRICK_PX


@dataclass
class View:
	id: str
	seq: int
	frame: int | None
	branch: int | None
	camera: str
	faces: list
	score: float | None
	threshold: float | None
	truth: dict | None = None
	review: bool = False
	extra: dict = field(default_factory=dict)

	@property
	def is_top(self) -> bool:
		return self.camera == "top"

	def min_px(self) -> int:
		bottom, wrist, _ = _constants()
		return bottom if self.is_top else wrist

	def true_faces(self) -> list[int] | None:
		"""Faces truly readable in the frame (same filter as the agent: top sees face 5 only, wrist sees 0..4)."""
		if not self.truth or "true_face_pixels" not in self.truth:
			return None
		floor = self.min_px()
		return [i for i, n in enumerate(self.truth["true_face_pixels"]) if n >= floor and ((i == 5) == self.is_top)]

	def above(self) -> bool:
		return self.score is not None and self.threshold is not None and self.score >= self.threshold


def collect_views(events: list[dict]) -> list[View]:
	truth = {e["id"]: e for e in events if e.get("kind") == "view_truth" and "id" in e}
	review_ids = {e.get("id") or e.get("view_id") for e in events if e.get("kind") == "review_view"}
	views = []
	for i, e in enumerate(events):
		if e.get("kind") != "view":
			continue
		obs = e.get("observation") or {}
		vid = obs.get("id")
		camera = obs.get("camera") or (str(vid).rsplit(":", 1)[-1] if vid else "")
		views.append(
			View(
				id=vid,
				seq=int(e.get("seq", i)),
				frame=e.get("frame", obs.get("frame")),
				branch=e.get("branch", obs.get("branch")),
				camera="top" if camera == "top" else "wrist_cam",
				faces=list(obs.get("faces") or []),
				score=obs.get("score"),
				threshold=obs.get("threshold"),
				truth=truth.get(vid),
				review=bool(vid in review_ids or obs.get("review") or e.get("review")),
			)
		)
	return views


def active_views(views: list[View], result: dict) -> list[View]:
	evidence = result.get("evidence") or []
	if not evidence:
		return views
	chosen = [v for v in views if v.id in set(evidence)]
	return chosen or views


def _finding(cause: str, detail: str, *, view: View | None = None, event: dict | None = None, seq=None, frame=None):
	src = event or {}
	return {
		"root_cause": cause,
		"frame": view.frame if view else (frame if frame is not None else src.get("frame")),
		"branch": view.branch if view else src.get("branch"),
		"view_id": view.id if view else None,
		"detail": detail,
		"seq": view.seq if view else (seq if seq is not None else src.get("seq", 0)),
	}


def _cross_crop(v: View) -> int | None:
	return (
		None if not v.truth or "true_cross_px_crop_native" not in v.truth else int(v.truth["true_cross_px_crop_native"])
	)


def _infra(events, episode, result, run_status):
	out = []
	if not episode or not result:
		reason = (run_status or {}).get("error") or "episode.json missing"
		state = (run_status or {}).get("state")
		out.append(_finding("infra_failure", f"no episode result (run_status={state}): {reason}", seq=-1, frame=0))
		return out
	if (run_status or {}).get("state") == "infra_failed":
		out.append(_finding("infra_failure", f"run_status infra_failed: {run_status.get('error')}", seq=-1, frame=0))
	if result.get("verdict") == "SUCCESS":
		return out
	for e in events:
		if e.get("kind") == "detector_error":
			out.append(_finding("infra_failure", "detector_error event", event=e))
			break
		if e.get("kind") == "failure" and any(c in str(e.get("code")) for c in INFRA_CODES):
			out.append(_finding("infra_failure", f"failure {e.get('code')}", event=e))
			break
	else:
		for f in result.get("failures") or []:
			if any(c in str(f.get("code")) for c in INFRA_CODES):
				out.append(_finding("infra_failure", f"failure {f.get('code')}", seq=10**9, frame=f.get("frame")))
				break
	return out


def _wrong_label_defective(views: list[View], face, budget_note: str):
	"""Defective brick, label normal."""
	truthful = [v for v in views if v.truth]
	if not truthful:
		seen = {f for v in views for f in v.faces}
		if face is not None and face not in seen:
			return _finding("target_face_not_in_view", f"defect face {face} never among believed faces (no view_truth)")
		last = views[-1] if views else None
		return _finding("other", "defective labelled normal, no view_truth events to attribute", view=last)
	for v in truthful:
		cross = _cross_crop(v)
		if cross is not None and cross >= CROSS_MIN_PX and v.score is not None and v.score < (v.threshold or 0.0):
			return _finding(
				"false_negative_visible", f"cross {cross}px in crop, score {v.score:.3f} < {v.threshold}", view=v
			)
	if face is not None:
		for v in truthful:
			px = v.truth.get("true_face_pixels") or []
			if face in v.faces and face < len(px) and px[face] < v.min_px():
				return _finding("occlusion", f"agent believed face {face} seen, true pixels {px[face]}", view=v)
		for v in truthful:
			px = v.truth.get("true_face_pixels") or []
			crop = v.truth.get("true_face_pixels_crop_native") or []
			if face >= len(px) or px[face] < v.min_px():
				continue
			lost = face < len(crop) and crop[face] < CROP_KEEP_RATIO * px[face]
			if lost or int(v.truth.get("true_cross_px") or 0) >= CROSS_MIN_PX > (_cross_crop(v) or 0):
				return _finding(
					"crop_error", f"face {face}: {px[face]}px in frame, {crop[face] if crop else '?'} in crop", view=v
				)
	best = max((_cross_crop(v) or 0 for v in truthful), default=0)
	return _finding(
		"target_face_not_in_view",
		f"defect face {face} never readable with the cross in the crop (best cross crop {best}px < {CROSS_MIN_PX})"
		+ budget_note,
		view=truthful[-1],
	)


def _wrong_label_normal(views: list[View], result: dict):
	"""Normal brick, label defective."""
	deciding = next((v for v in views if v.id == result.get("placement_evidence")), None) or next(
		(v for v in views if v.above()), None
	)
	if deciding is None:
		return _finding("false_positive_visible", "no above-threshold view recorded", view=views[-1] if views else None)
	truth = deciding.truth or {}
	crop = truth.get("true_face_pixels_crop_native")
	frame_px = truth.get("true_face_pixels")
	if crop is not None and frame_px is not None:
		crop_sum, frame_sum = sum(crop), sum(frame_px)
		if crop_sum < _constants()[2] or (frame_sum and crop_sum < CROP_KEEP_RATIO * frame_sum):
			return _finding("crop_error", f"crop kept {crop_sum} of {frame_sum} brick pixels", view=deciding)
	return _finding("false_positive_visible", f"score {deciding.score:.3f} >= {deciding.threshold}", view=deciding)


def _unknown_label(views, result, task, events):
	failures = result.get("failures") or []
	policy = (task or {}).get("policy") or {}
	spent = int(result.get("rewinds") or 0) + int(result.get("retries") or 0)
	if failures and spent >= max(1, int(policy.get("max_rewinds", 1))):
		f = failures[-1]
		return _finding(
			"recovery_failure",
			f"{len(failures)} failures, recovery used ({spent}); last {f.get('code')}",
			seq=10**9,
			frame=f.get("frame"),
		)
	cleared = set(result.get("cleared_faces") or [])
	budget = _view_budget(task, result)
	n = int(result.get("views") or len(views))
	if len(cleared) < 6 and (budget is None or n < budget):
		return _finding(
			"premature_stop",
			f"{len(cleared)}/6 faces cleared with {n}/{budget} views used",
			view=views[-1] if views else None,
			seq=10**9,
		)
	unavailable = next((e for e in events if e.get("kind") == "view_unavailable"), None)
	return _finding(
		"target_face_not_in_view",
		"label unknown with views exhausted" + (f" ({unavailable.get('reason')})" if unavailable else ""),
		event=unavailable,
		seq=10**9,
	)


def _view_budget(task, result) -> int | None:
	policy = (task or {}).get("policy") or {}
	if "max_bottom_views" in policy or "max_wrist_views" in policy:
		return int(policy.get("max_bottom_views", 0)) + int(policy.get("max_wrist_views", 0))
	return None


def _evidence_check(views: list[View], expected: str, result: dict, face):
	if expected == "defective":
		deciding = next((v for v in views if v.id == result.get("placement_evidence")), None) or next(
			(v for v in views if v.above()), None
		)
		if deciding is None:
			return _finding(
				"evidence_insufficient", "defective label without a deciding view", view=views[-1] if views else None
			)
		cross = _cross_crop(deciding)
		if cross is None:
			return None  # no truth recorded: cannot judge
		if cross < CROSS_MIN_PX:
			return _finding(
				"evidence_insufficient", f"deciding view crop held {cross}px of cross (< {CROSS_MIN_PX})", view=deciding
			)
		return None
	covered, have_truth = set(), False
	for v in views:
		faces = v.true_faces()
		if faces is not None:
			have_truth = True
			covered.update(faces)
	if not have_truth:
		return None
	if len(covered) < 6:
		return _finding(
			"evidence_insufficient",
			f"normal label with faces {sorted(covered)} truly visible",
			view=views[-1] if views else None,
		)
	return None


def _physical_failure(result: dict, events, physical: dict, expected_bowl: str | None):
	codes = []
	health = result.get("health") or {}
	cycle = health.get("final_cycle") or {}
	for src in (cycle, health):
		codes += [c for c in (src.get("reason_codes") or []) if c]
		if src.get("failure_mode") not in (None, "ok"):
			codes.append(src["failure_mode"])
	for f in result.get("failures") or []:
		codes.append(str(f.get("code")))
		codes += [str(c) for c in f.get("reason_codes") or []]
	failure_events = [e for e in events if e.get("kind") == "failure"]
	last = failure_events[-1] if failure_events else (events[-1] if events else {})
	outcome = (physical or {}).get("outcome")
	moved = (physical or {}).get("moved_m")
	wrong_bowl = outcome in ("green_bowl", "pink_bowl") and outcome != expected_bowl
	dropped = not codes and outcome == "table" and moved is not None and moved > 0.02
	placed_badly = any(c in PLACEMENT_CODES for c in codes) or wrong_bowl or dropped
	cause = "placement_failure" if placed_badly else "grasp_or_path_failure"
	return _finding(cause, f"outcome={outcome}, codes={list(dict.fromkeys(codes))[:6]}", event=last)


def _bowl_localization_failure(events: list[dict], result: dict) -> dict | None:
	last_sense = {}
	for event in events:
		kind = event.get("kind")
		if kind == "bowls_perceived":
			last_sense[event.get("branch")] = event
		elif kind == "failure" and event.get("code") == "BOWL_NOT_FOUND":
			sensed = last_sense.get(event.get("branch"), {})
			missing = sorted(slot for slot, pos in (sensed.get("positions_m") or {}).items() if pos is None)
			detail = "BOWL_NOT_FOUND"
			if missing:
				detail += f"; missing bowl slots: {', '.join(missing)}"
			return _finding("bowl_localization", detail, event=event)
	for failure in result.get("failures") or []:
		if failure.get("code") == "BOWL_NOT_FOUND":
			return _finding("bowl_localization", "BOWL_NOT_FOUND", event=failure)
	if "BOWL_NOT_FOUND" in (result.get("code"), result.get("execution_error"), result.get("error")):
		return _finding("bowl_localization", "BOWL_NOT_FOUND")
	return None


def first_error(
	episode: dict | None,
	events: list[dict],
	task: dict | None,
	audit: dict | None,
	physical: dict | None,
	run_status: dict | None = None,
	thresholds: dict | None = None,
) -> dict:
	result = (episode or {}).get("result") or {}
	exp = ((episode or {}).get("provenance") or {}).get("experiment") or {}
	case = (task or {}).get("case") or []
	options = case[3] if len(case) >= 4 and isinstance(case[3], dict) else {}
	camera_condition = (
		exp.get("camera_condition")
		or ((run_status or {}).get("experiment") or {}).get("camera_condition")
		or result.get("camera_condition")
		or options.get("cameras")
	)
	findings = _infra(events, episode, result, run_status)
	if not episode or not result:
		return _with_mechanism(min(findings, key=lambda f: f["seq"]), events, thresholds, camera_condition)

	expected = (audit or {}).get("true_label") or (("defective" if case[1] else "normal") if len(case) > 1 else None)
	face = (audit or {}).get("true_face")
	if face is None and len(case) > 2 and len(case) > 1 and case[1]:
		face = case[2]
	label = result.get("label")
	views = active_views(collect_views(events), result)
	verdict = result.get("verdict_final_cycle") or result.get("verdict")
	expected_bowl = {"normal": "green_bowl", "defective": "pink_bowl"}.get(expected)
	note = f"; scored views {len(views)}"
	physical_ok = (physical or {}).get("success")
	if physical_ok is None:  # no replay: fall back to the engine verdict
		physical_ok = verdict == "SUCCESS"
	if label not in LABELS or (expected in LABELS and label != expected) or not physical_ok:
		bowl_failure = _bowl_localization_failure(events, result)
		if bowl_failure is not None:
			findings.append(bowl_failure)

	if label not in LABELS:
		findings.append(_unknown_label(views, result, task, events))
	elif expected in LABELS and label != expected:
		findings.append(
			_wrong_label_defective(views, face, note) if expected == "defective" else _wrong_label_normal(views, result)
		)
	else:
		if not physical_ok:
			findings.append(_physical_failure(result, events, physical or {}, expected_bowl))
		else:
			bad = _evidence_check(views, expected, result, face)
			if bad:
				findings.append(bad)
	if not findings:
		return _with_mechanism(
			{
				"root_cause": "none",
				"frame": None,
				"branch": None,
				"view_id": None,
				"detail": "sufficient evidence",
				"seq": 10**9,
			},
			events,
			thresholds,
			camera_condition,
		)
	return _with_mechanism(min(findings, key=lambda f: f["seq"]), events, thresholds, camera_condition)


def _with_mechanism(finding: dict, events: list[dict], thresholds, camera_condition: str | None) -> dict:
	from .precursors import explain, precursors

	found = precursors(events, thresholds, camera_condition=camera_condition)
	seq = finding.get("seq", 10**9)
	return {**_strip(finding), **explain(finding["root_cause"], seq, found), "precursors": found}


def _strip(finding: dict) -> dict:
	return {k: v for k, v in finding.items() if k != "seq"}
