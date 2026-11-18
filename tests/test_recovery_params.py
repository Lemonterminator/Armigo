"""Bounded recovery variants for TwinMujoco.rewind(): table bounds, choice ordering, parameter effect."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from digital_twin_armigo.agent import twin_mujoco as tm
from digital_twin_armigo.mujoco_cell.physics_expert import ExpertParams
from digital_twin_armigo.mujoco_cell.table_relocation import RelocateParams

BASE = (ExpertParams(), RelocateParams(do_bottom=False, do_scan=False))


def checkpoints(*frames):
	return [SimpleNamespace(id=f"checkpoint-{i}", frame=f) for i, f in enumerate(frames)]


@pytest.mark.parametrize("name", sorted(tm.RECOVERY_VARIANTS))
def test_variant_inside_bounds_and_never_more_aggressive(name):
	p, rp, speed = tm.apply_variant(name, *BASE)
	for key, value in tm.RECOVERY_VARIANTS[name].items():
		lo, hi = tm.RECOVERY_BOUNDS[key]
		assert lo <= value <= hi
	assert tm.RECOVERY_BOUNDS["scan_peak_speed"][1] == tm.SCAN_PEAK_SPEED
	assert speed <= tm.SCAN_PEAK_SPEED
	assert p.carry_peak_speed_rad_s <= BASE[0].carry_peak_speed_rad_s
	for field in ("show_move_s", "transfer_s", "place_descend_s", "return_s", "retreat_s", "ready_move_s"):
		assert getattr(p, field) >= getattr(BASE[0], field)
	assert rp.scan_move_s >= BASE[1].scan_move_s
	assert p.transfer_z_m >= BASE[0].transfer_z_m


def test_variant_table_is_frozen():
	with pytest.raises(TypeError):
		tm.RECOVERY_VARIANTS["new"] = {}
	with pytest.raises(TypeError):
		tm.RECOVERY_VARIANTS["slow_scan"]["scan_peak_speed"] = 5.0


def test_out_of_bounds_and_unknown_rejected(monkeypatch):
	with pytest.raises(ValueError, match="unknown"):
		tm.apply_variant("fast", *BASE)
	bad = dict(tm.RECOVERY_VARIANTS)
	bad["bad"] = {"carry_peak_speed_rad_s": 2.0}
	monkeypatch.setattr(tm, "RECOVERY_VARIANTS", bad)
	with pytest.raises(ValueError, match="outside"):
		tm.apply_variant("bad", *BASE)


def test_variant_changes_rebuilt_params_only_where_named():
	p, rp, speed = tm.apply_variant("slow_scan", *BASE)
	assert speed == 0.35 and rp.scan_move_s == 1.0 and p.ready_move_s == 1.0
	assert p.carry_peak_speed_rad_s == BASE[0].carry_peak_speed_rad_s
	p2, rp2, speed2 = tm.apply_variant("high_place", *BASE)
	assert p2.transfer_z_m == 0.14 and p2.place_descend_s == 1.8 and speed2 == tm.SCAN_PEAK_SPEED
	assert rp2 == BASE[1]


def test_stall_start_needs_a_run_longer_than_limit():
	limit = 2.94
	torques = np.zeros((200, 6))
	torques[50:90, 4] = limit  # 40 frames at 30 fps = 1.33 s: below the health limit
	assert tm.stall_start(torques, limit) is None
	torques[50:100, 4] = limit  # 50 frames = 1.67 s
	assert tm.stall_start(torques, limit) == 50


def test_node_at_replays_rewinds():
	events = [
		{"kind": "node", "node": "prepare", "frame": 0},
		{"kind": "node", "node": "scan_wrist", "frame": 191},
		{"kind": "node", "node": "present_bottom", "frame": 657},
		{"kind": "node", "node": "place", "frame": 1100},
		{"kind": "node", "node": "recover", "frame": 1700},
		{"kind": "rewound", "node": "recover", "frame": 191},
		{"kind": "node", "node": "scan_wrist", "frame": 191},
	]
	assert tm.node_at(events, 400) == "scan_wrist"
	assert tm.node_at(events, 100) == "prepare"
	assert tm.node_at(events[:5], 1200) == "place"


def test_stall_default_is_checkpoint_before_stall_with_recommended_variant():
	cps = checkpoints(191, 657)
	failure = {"code": "PHYSICAL_HEALTH_FAILED", "reason_codes": ["torque_stall"]}
	options = tm.rewind_options(cps, failure, 554, "scan_wrist")
	choices = [c for c, _ in options]
	assert choices[0] == "checkpoint-0+slow_scan"
	assert "checkpoint-0" in choices and "checkpoint-1" in choices and "checkpoint-1+slow_scan" in choices
	assert choices.index("checkpoint-0") < choices.index("checkpoint-1+slow_scan")
	assert all(isinstance(note, str) and note for _, note in options)
	assert "keeps the stall" in dict(options)["checkpoint-1"]


def test_no_stall_evidence_prefers_latest_checkpoint_plain():
	cps = checkpoints(191, 657)
	choices = [c for c, _ in tm.rewind_options(cps, {"code": "SOMETHING_ELSE"}, None, None)]
	assert choices == ["checkpoint-1", "checkpoint-0"]


def test_code_map_recommends_variant():
	choices = [c for c, _ in tm.rewind_options(checkpoints(191), {"code": "PLACEMENT_UNVERIFIED"}, None, None)]
	assert choices[0] == "checkpoint-0+high_place"
	assert choices[-1] == "checkpoint-0"
	assert all(c.partition("+")[2] in ("", *tm.RECOVERY_VARIANTS) for c in choices)


class FakeScorer:
	def threshold(self, camera):
		return 0.0

	def score(self, camera, image):
		return -1.0


def test_rewind_rebuilds_expert_with_chosen_variant(tmp_path):
	from digital_twin_armigo.agent.twin_graph import Policy
	from digital_twin_armigo.agent.twin_record import EpisodeTrace

	backend = tm.TwinMujoco(tm.sample_case(442701, False, 5), FakeScorer(), EpisodeTrace(tmp_path / "ep", video=False))
	picks = ["checkpoint-0", "checkpoint-0+slow_scan"]
	seen = []

	def choose(allowed, views, failures, notes=None):
		seen.append((allowed, notes))
		return picks[len(seen) - 1]

	try:
		assert backend.prepare() is None
		first_params = backend.expert.p
		backend.choose = choose
		failure = {"code": "TEST_FAILURE"}
		assert backend.rewind(Policy(max_rewinds=2), [failure]) == []
		allowed, notes = seen[0]
		assert allowed == ["checkpoint-0", "stop"] and set(notes) == set(allowed)
		assert backend.expert.p is first_params and backend.scan_peak_speed == 1.0
		assert backend.rewind(Policy(max_rewinds=2), [failure]) == []
		assert backend.expert.p.ready_move_s == 1.0 and first_params.ready_move_s == 0.6
		assert backend.expert.rp.scan_move_s == 1.0
		assert backend.scan_peak_speed == 0.35
		assert backend.recovery_variants == ("none", "slow_scan")
		rewound = [e for e in backend.record.events if e["kind"] == "rewound"]
		assert [e["variant"] for e in rewound] == [None, "slow_scan"]
	finally:
		backend.close()
