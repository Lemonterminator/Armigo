"""WP7 tests: closed-loop wrist recheck persistence and dense in-motion segment mapping under ExpertTracer/TwinService."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("langgraph")
pytest.importorskip("cv2")

from digital_twin_armigo.agent import step_mcp as sm
from digital_twin_armigo.agent.student_rollout import (
	POLICY_FIELDS,
	ExpertTracer,
	Segment,
	_SegmentMixin,
	comparable_events,
	policy_inputs,
)
from digital_twin_armigo.agent.twin_graph import Policy
from digital_twin_armigo.agent.twin_mcp import TwinService
from digital_twin_armigo.agent.twin_mujoco import sample_case
from digital_twin_armigo.agent.twin_record import EpisodeTrace, segment_labels
from digital_twin_armigo.constants import FPS
from digital_twin_armigo.inspection_scorer import FixedScorer
from digital_twin_armigo.vla.contract import assert_no_planner_diagnostics


class _TestScorer(FixedScorer):
	def close(self) -> None:
		pass


def _make_detector_spec(tmp_path: Path) -> Path:
	spec = tmp_path / "detector.json"
	spec.write_text(json.dumps({"crop_size": 256}), encoding="utf-8")
	return spec


def _load_events(ep_dir: Path) -> list[dict]:
	return [json.loads(line) for line in (ep_dir / "events.jsonl").read_text(encoding="utf-8").splitlines() if line]


def test_w2s_recheck_persists_across_retry_and_emits_wrist_fix(tmp_path):
	"""Reproduces motion.json retry audit: ExpertTracer with motion='w2s' must retain recheck across _retry
	and emit wrist_fix events on a subsequent grasp."""
	scorer = _TestScorer(lambda _cam, _img: -1.0, {"top": 0.0, "wrist_cam": 0.0})
	ep_dir = tmp_path / "retry-w2s"
	trace = EpisodeTrace(ep_dir, video=False)
	twin = ExpertTracer(sample_case(700001, False, 0), scorer, trace, sensing="realistic", motion="w2s")
	try:
		assert twin.expert.recheck is not None
		assert twin.expert.recheck.mode == "sync"
		assert twin.prepare() is None

		old_expert = twin.expert
		failures = [{"code": "no_clamp", "skill": "present_bottom"}]
		policy = Policy()
		choices = twin.recovery_choices(policy, failures)
		twin.forced_choice = next(c for c, _ in choices if c != "stop")
		kept = twin._retry(policy, failures)
		assert kept is not None
		assert twin.retries == 1
		assert twin.motion == "w2s"
		assert twin.expert is not old_expert
		assert twin.expert.recheck is not None, "expert.recheck was dropped across _retry"
		assert twin.expert.recheck.mode == "sync"

		retry_frame = twin.ep.n_frames
		err = twin._grasp()
		assert err is None and twin.held is not None
		post_retry_fixes = [e for e in trace.events if e["kind"] == "wrist_fix" and e["frame"] >= retry_frame]
		assert len(post_retry_fixes) >= 1, "subsequent grasp after _retry did not run wrist recheck"
		assert all(e["mode"] == "sync" for e in post_retry_fixes)
	finally:
		trace.close()
		twin.close()


def test_w2s_recheck_persists_across_restore_state(tmp_path):
	"""restore_state must preserve w2s recheck and clear it when restoring an open-loop checkpoint."""
	scorer = _TestScorer(lambda _cam, _img: -1.0, {"top": 0.0, "wrist_cam": 0.0})
	ep_dir = tmp_path / "restore-w2s"
	trace = EpisodeTrace(ep_dir, video=False)
	twin = ExpertTracer(sample_case(700001, False, 0), scorer, trace, sensing="realistic", motion="w2s")
	try:
		assert twin.prepare() is None
		ckpt = twin.checkpoints[0]
		assert twin.expert.recheck is not None

		twin.restore_state(ckpt)
		assert twin.expert.recheck is not None
		assert twin.expert.recheck.mode == "sync"

		# Even if a checkpoint's agent metadata lacks "motion", _make_expert preserves twin.motion.
		legacy_ckpt = type(ckpt)(**{**ckpt.__dict__, "agent": {k: v for k, v in ckpt.agent.items() if k != "motion"}})
		twin.restore_state(legacy_ckpt)
		assert twin.expert.recheck is not None
		assert twin.expert.recheck.mode == "sync"

		# Restoring a checkpoint that explicitly specifies motion="open" must clear recheck.
		open_ckpt = type(ckpt)(**{**ckpt.__dict__, "agent": {**ckpt.agent, "motion": "open"}})
		twin.restore_state(open_ckpt)
		assert twin.motion == "open"
		assert twin.expert.recheck is None
	finally:
		trace.close()
		twin.close()


@pytest.mark.parametrize("evidence", ["dense-10", "dense-20"])
def test_dense_scan_through_twin_service_end_of_sweep_hold_and_valid_segments(tmp_path, evidence):
	"""Reproduces motion.json dense audit through TwinService -> ExpertTracer on (700001, False, 0) with -1.0 scorer:
	walks to end-of-sweep hold fallback, finishes SUCCESS, and passes segment_labels with zero -1 frames."""
	detector = _make_detector_spec(tmp_path)
	scorer = _TestScorer(lambda _cam, _img: -1.0, {"top": 0.0, "wrist_cam": 0.0})
	svc = TwinService(
		tmp_path / f"svc-{evidence}",
		detector,
		sensing="realistic",
		agent_options={"strategy": "B", "boundary": "belief_cell", "evidence": evidence},
		scorer=scorer,
	)
	ep_dir = tmp_path / f"ep-{evidence}"
	try:
		result = svc.run((700001, False, 0), Policy(), ep_dir, video=False, use_llm=False, hook=None)
	finally:
		svc.close()

	assert result["verdict"] == "SUCCESS", result
	assert result["label"] == "normal"

	events = _load_events(ep_dir)
	wrist_views = [
		e["observation"] for e in events if e["kind"] == "view" and e["observation"]["camera"] == "wrist_cam"
	]
	in_motion_views = [v for v in wrist_views if v.get("in_motion") is True]
	static_views = [v for v in wrist_views if v.get("in_motion") is False]
	assert len(in_motion_views) >= 1
	assert len(static_views) >= 1, "negative brick should reach end-of-sweep hold and take static fallback capture"

	expected_delay = max(1, round(0.15 * FPS))
	assert all(v["decided_after_frames"] == expected_delay for v in in_motion_views)
	assert all(v["decided_after_frames"] == 0 for v in static_views)
	for obs in wrist_views:
		sm.assert_sensor_only(obs)

	labels = segment_labels(ep_dir)
	with np.load(ep_dir / "branch-0" / "actions.npz", allow_pickle=False) as rows:
		seg_col = rows["segment"]
		assert len(seg_col) == result["sim_frames"]
		assert (seg_col >= 0).all(), f"found {(seg_col < 0).sum()} frames with segment == -1"

	scan_segs = [s for s in labels["segments"] if s["kind"] == "scan_view"]
	assert len(scan_segs) >= 1
	for seg_row in scan_segs:
		pi = policy_inputs(Segment(seg_row["kind"], seg_row["params"], variant=int(seg_row.get("variant", 0))))
		assert tuple(pi) == POLICY_FIELDS
		assert_no_planner_diagnostics(pi)


def test_dense_scan_through_twin_service_respects_max_scored_views_budget(tmp_path):
	"""Exact motion_probe.py budget-capped configuration (max_scored_views=6) completes without crashing,
	strictly obeys scored_views <= 6, captures in-motion views, and writes valid segment labels."""
	detector = _make_detector_spec(tmp_path)
	scorer = _TestScorer(lambda _cam, _img: -1.0, {"top": 0.0, "wrist_cam": 0.0})
	svc = TwinService(
		tmp_path / "svc-cap",
		detector,
		sensing="realistic",
		agent_options={"strategy": "B", "max_scored_views": 6, "boundary": "belief_cell", "evidence": "dense-10"},
		scorer=scorer,
	)
	ep_dir = tmp_path / "ep-cap"
	try:
		result = svc.run((700001, False, 0), Policy(), ep_dir, video=False, use_llm=False, hook=None)
	finally:
		svc.close()

	assert result["scored_views"] <= 6
	events = _load_events(ep_dir)
	in_motion_views = [e["observation"] for e in events if e["kind"] == "view" and e["observation"].get("in_motion")]
	assert 1 <= len(in_motion_views) <= 6
	labels = segment_labels(ep_dir)
	with np.load(ep_dir / "branch-0" / "actions.npz", allow_pickle=False) as rows:
		assert (rows["segment"] >= 0).all()
		assert len(labels["index"]) == len(rows["segment"])


@pytest.mark.parametrize("evidence", ["dense-10", "dense-20"])
def test_dense_scan_early_stop_defective_through_twin_service(tmp_path, evidence):
	"""Defective case with +1.0 scorer triggers early stop during in-motion dense sweep (no static hold fallback)
	and passes segment_labels with zero -1 frames."""
	detector = _make_detector_spec(tmp_path)
	scorer = _TestScorer(lambda _cam, _img: 1.0, {"top": 0.0, "wrist_cam": 0.0})
	svc = TwinService(
		tmp_path / f"svc-def-{evidence}",
		detector,
		sensing="realistic",
		agent_options={"strategy": "B", "boundary": "belief_cell", "evidence": evidence},
		scorer=scorer,
	)
	ep_dir = tmp_path / f"ep-def-{evidence}"
	try:
		result = svc.run((700001, True, 0), Policy(), ep_dir, video=False, use_llm=False, hook=None)
	finally:
		svc.close()

	assert result["verdict"] == "SUCCESS", result
	assert result["label"] == "defective"
	events = _load_events(ep_dir)
	wrist_views = [
		e["observation"] for e in events if e["kind"] == "view" and e["observation"]["camera"] == "wrist_cam"
	]
	in_motion_views = [v for v in wrist_views if v.get("in_motion") is True]
	static_views = [v for v in wrist_views if v.get("in_motion") is False]
	assert len(in_motion_views) >= 2
	assert len(static_views) == 0, "early-stop defective path should not run end-of-sweep static capture"
	for obs in wrist_views:
		sm.assert_sensor_only(obs)

	labels = segment_labels(ep_dir)
	with np.load(ep_dir / "branch-0" / "actions.npz", allow_pickle=False) as rows:
		assert (rows["segment"] >= 0).all()
		assert len(labels["index"]) == result["sim_frames"]


def test_default_open_hold_expert_tracer_matches_explicit_defaults(tmp_path):
	"""Default motion='open', evidence='hold' through TwinService/ExpertTracer stays bitwise identical."""
	detector = _make_detector_spec(tmp_path)
	scorer = _TestScorer(lambda _cam, _img: 1.0, {"top": 0.0, "wrist_cam": 0.0})

	def _run(name: str, agent_options: dict | None):
		svc = TwinService(
			tmp_path / f"svc-{name}",
			detector,
			sensing="realistic",
			agent_options=agent_options,
			scorer=scorer,
		)
		ep_dir = tmp_path / f"ep-{name}"
		try:
			res = svc.run((600001, True, 0), Policy(), ep_dir, video=False, use_llm=False, hook=None)
		finally:
			svc.close()
		return res, ep_dir

	res_def, dir_def = _run("default", None)
	res_exp, dir_exp = _run("explicit", {"motion": "open", "evidence": "hold"})
	assert res_def["verdict"] == "SUCCESS"
	assert res_def["label"] == res_exp["label"] == "defective"

	with (
		np.load(dir_def / "branch-0" / "actions.npz", allow_pickle=False) as a,
		np.load(dir_exp / "branch-0" / "actions.npz", allow_pickle=False) as b,
	):
		assert a.files == b.files
		for k in a.files:
			assert np.array_equal(a[k], b[k]), k

	assert comparable_events(_load_events(dir_def)) == comparable_events(_load_events(dir_exp))
	assert segment_labels(dir_def)["segments"] == segment_labels(dir_exp)["segments"]


def test_dense_scan_motion_mapping_unit():
	"""_SegmentMixin._motion maps dense_scan(path) to canonical scan_view with schema-compatible params."""
	cand = SimpleNamespace(q=np.array([0.1, 0.2, 0.3, 0.4, 1.5, 0.85]), faces_px=[900, 0, 1200, 0, 950, 0])
	stub = SimpleNamespace(
		_scan_steps=[(cand, [])],
		_scan_mode="orbit",
		evidence="dense-10",
		node="scan_wrist",
	)

	def dense_scan(path):
		yield path[-1], "approach"

	path = [np.zeros(6), cand.q]
	form, seg = _SegmentMixin._motion(stub, dense_scan(path))
	assert form == "simple"
	assert seg.kind == "scan_view"
	assert seg.params["mode"] == "orbit"
	assert seg.params["evidence"] == "dense-10"
	assert seg.params["view"] == 1 and seg.params["views"] == 1
	assert seg.params["via_poses"] == 1
	assert seg.params["faces"] == [0, 2, 4]
	pi = policy_inputs(seg)
	assert tuple(pi) == POLICY_FIELDS
	assert_no_planner_diagnostics(pi)
