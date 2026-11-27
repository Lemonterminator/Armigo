import pytest

from digital_twin_armigo.governance.pairs import OUTCOMES, flags, pair_key, pair_outcome, sensing_of


def row(group, success=True, **kw):
	base = {
		"group": group,
		"engine": "mujoco",
		"case_id": "L3000101-N",
		"detector": "v7bot",
		"strategy": "B",
		"physical_success": success,
		"infra_failed": False,
		"quarantined": False,
		"oracle": False,
		"n_recoveries": 0,
		"n_review_views": 0,
		"n_llm_images": 0,
		"physical": {"source": "replay"},
	}
	return base | kw


def test_sensing_from_group_names():
	assert sensing_of("truth-v7bot") == "truth"
	assert sensing_of("truth-B-v7bot") == "truth"
	assert sensing_of("realistic-drift-B-v7bot-mw2a-eimg") == "realistic"
	with pytest.raises(ValueError):
		sensing_of("weird-group")
	with pytest.raises(ValueError):
		sensing_of("")


def test_pair_key_ignores_sensing_and_camera():
	t = row("truth-B-v7bot")
	r = row("realistic-drift-B-v7bot", camera_condition="drift")
	assert pair_key(t) == pair_key(r) == ("mujoco", "L3000101-N", "v7bot", "B", "open", "hold")
	assert pair_key(row("realistic-drift-B-v7bot", motion="w2a")) != pair_key(t)
	assert pair_key(row("truth-v7bot", strategy=None, motion=None, evidence=None)) == (
		"mujoco",
		"L3000101-N",
		"v7bot",
		"A",
		"open",
		"hold",
	)


@pytest.mark.parametrize(
	("t_ok", "r_ok", "want"),
	[
		(True, True, "both_pass"),
		(True, False, "truth_only"),
		(False, True, "realistic_only"),
		(False, False, "both_fail"),
	],
)
def test_outcome_matrix(t_ok, r_ok, want):
	assert pair_outcome(row("truth-B-v7bot", t_ok), row("realistic-fixed-B-v7bot", r_ok)) == want


def test_void_and_unpaired():
	assert pair_outcome(row("truth-B-v7bot", infra_failed=True), row("realistic-fixed-B-v7bot")) == "void"
	assert pair_outcome(row("truth-B-v7bot"), row("realistic-fixed-B-v7bot", quarantined=True)) == "void"
	assert pair_outcome(None, row("realistic-fixed-B-v7bot")) == "unpaired"
	assert pair_outcome(row("truth-B-v7bot"), None) == "unpaired"
	assert pair_outcome(None, None) == "unpaired"
	assert set(OUTCOMES) == {"both_pass", "truth_only", "realistic_only", "both_fail", "void", "unpaired"}


def test_flags():
	f = flags(row("realistic-fixed-B-v7bot", n_recoveries=2, n_llm_images=1, physical={"source": "verdict"}))
	assert f == {
		"recovered": True,
		"review_used": False,
		"llm_used": True,
		"oracle": False,
		"outcome_source": "verdict",
	}
	f_defaults = flags({"n_review_views": 1, "oracle": True, "physical": None})
	assert f_defaults == {
		"recovered": False,
		"review_used": True,
		"llm_used": False,
		"oracle": True,
		"outcome_source": "verdict",
	}
