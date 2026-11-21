"""Per-state / per-phase timing statistics for the agent viewer."""

from __future__ import annotations

import json
import math

import pytest

from digital_twin_armigo.viewer import stats


def _ev(seq, kind, node, frame, sim, wall, branch=0, **extra):
	return {
		"seq": seq,
		"kind": kind,
		"node": node,
		"branch": branch,
		"frame": frame,
		"sim_time_s": sim,
		"wall_time_s": wall,
		**extra,
	}


EVENTS = [
	_ev(0, "node", "prepare", 0, 0.0, 1.0),
	_ev(1, "node", "choose_view", 150, 5.0, 2.0),
	_ev(2, "decision", "choose_view", 150, 5.0, 2.5, source="llm", latency_s=0.5, fallback_reason=None, attempts=1),
	_ev(3, "node", "scan_wrist", 150, 5.0, 2.6),
	_ev(4, "node", "choose_view", 300, 10.0, 6.0),
	_ev(
		5,
		"decision",
		"choose_view",
		300,
		10.0,
		6.0,
		source="policy",
		latency_s=2.0,
		fallback_reason="timeout",
		attempts=2,
	),
	_ev(6, "node", "place", 300, 10.0, 6.1),
	_ev(7, "node", "evaluate", 450, 15.0, 9.0),
]


def test_node_visits_measure_wall_and_sim_between_node_events():
	v = stats.node_visits(EVENTS)
	assert [x["node"] for x in v] == ["prepare", "choose_view", "scan_wrist", "choose_view", "place", "evaluate"]
	assert v[0]["wall_s"] == pytest.approx(1.0) and v[0]["sim_s"] == pytest.approx(5.0)
	assert v[2]["wall_s"] == pytest.approx(3.4)
	# The last visit ends at the last event: nothing is invented for an episode that already finished.
	assert v[-1]["open"] and v[-1]["wall_s"] == 0.0
	# Decisions are attributed to the visit that was current when they were made.
	assert v[1]["decisions"] == 1 and v[1]["llm_latency_s"] == pytest.approx(0.5)
	assert v[3]["decisions"] == 1 and v[3]["llm_latency_s"] == pytest.approx(2.0)


def test_live_visit_is_measured_up_to_now():
	v = stats.node_visits(EVENTS, now_wall=12.0, now_sim=20.0)
	assert v[-1]["open"] and v[-1]["wall_s"] == pytest.approx(3.0) and v[-1]["sim_s"] == pytest.approx(5.0)


def test_visits_pair_with_next_node_event_of_the_same_branch():
	events = [
		_ev(0, "node", "prepare", 0, 0.0, 0.0, branch=0),
		_ev(1, "node", "recover", 100, 3.0, 1.0, branch=0),
		_ev(2, "node", "choose_view", 100, 3.0, 1.5, branch=1),
		_ev(3, "node", "evaluate", 200, 6.0, 3.0, branch=0),
		_ev(4, "node", "evaluate", 200, 6.0, 4.0, branch=1),
	]
	v = {(x["node"], x["branch"]): x for x in stats.node_visits(events)}
	assert v[("choose_view", 1)]["wall_s"] == pytest.approx(2.5)
	assert v[("recover", 0)]["wall_s"] == pytest.approx(2.0)


def test_current_state_is_last_node_event():
	s = stats.current_state(EVENTS)
	assert s["node"] == "evaluate" and s["frame"] == 450 and not s["finished"]
	assert stats.current_state([])["node"] is None


def test_phase_segments_run_length_encode_with_interpolated_wall_time():
	phases = ["rest"] * 150 + ["approach"] * 150 + ["grasp"] * 150
	segs = stats.phase_segments(phases, wall_at=stats.wall_interpolator(EVENTS))
	assert [(s["phase"], s["frames"]) for s in segs] == [("rest", 150), ("approach", 150), ("grasp", 150)]
	assert segs[0]["sim_s"] == pytest.approx(5.0)
	# Several events share frame 150 (the simulation is paused while the agent decides); the frame is dated by the last
	# one, when the simulation resumed, so LLM thinking time is not charged to the phase that ended there.
	assert segs[0]["wall_s"] == pytest.approx(1.6)  # frame 0 (1.0 s) .. frame 150 (2.6 s)
	assert segs[2]["frame1"] == 450


def test_percentiles_and_summary_shares():
	assert stats.percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)
	assert math.isnan(stats.percentile([], 50))
	rows = [{"node": "a", "x": 1.0}, {"node": "a", "x": 3.0}, {"node": "b", "x": 4.0}]
	out = stats.summarize(rows, "node", "x")
	a = next(r for r in out if r["node"] == "a")
	assert a["count"] == 2 and a["mean"] == pytest.approx(2.0) and a["max"] == 3.0 and a["share"] == pytest.approx(0.5)
	assert out[0]["total"] >= out[1]["total"]


def test_decision_stats_count_fallbacks_and_latency():
	d = stats.decision_stats(EVENTS)
	assert d["count"] == 2 and d["sources"] == {"llm": 1, "policy": 1}
	assert d["fallbacks"] == 1 and d["fallback_rate"] == pytest.approx(0.5)
	assert d["latency_mean_s"] == pytest.approx(1.25) and d["latency_max_s"] == 2.0 and d["attempts"] == 3


def test_episode_stats_bundle():
	s = stats.episode_stats(EVENTS, ["rest"] * 150 + ["approach"] * 150, finished=True)
	assert s["state"]["finished"] and s["totals"]["visits"] == 6 and s["totals"]["recoveries"] == 0
	assert {r["node"] for r in s["nodes"]["wall"]} >= {"prepare", "choose_view"}
	assert s["phases"]["sim"] and s["totals"]["wall_s"] == 9.0
	json.dumps(s, allow_nan=False)  # the HTML API serialises this verbatim


def test_read_events_tolerates_a_half_written_last_line(tmp_path):
	p = tmp_path / "events.jsonl"
	p.write_text(json.dumps(EVENTS[0]) + "\n" + json.dumps(EVENTS[1]) + '\n{"seq": 2, "ki', encoding="utf-8")
	assert len(stats.read_events(p)) == 2
	assert stats.read_events(tmp_path / "missing.jsonl") == []


def test_defect_strokes_are_drawn_thick_enough_to_be_seen():
	from digital_twin_armigo.viewer import rerun_log

	stroke = {"name": "defect_0", "type": "box", "size": [0.0075, 0.0012, 0.0001]}
	assert rerun_log.box_half_sizes(stroke) == [0.0075, 0.0012, rerun_log.DEFECT_MIN_HALF_T_M]
	other = {"name": "brick", "type": "box", "size": [0.01, 0.01, 0.0001], "scale": [1, 1, 2]}
	assert rerun_log.box_half_sizes(other) == [0.01, 0.01, 0.0002]  # only the defect strokes are changed
