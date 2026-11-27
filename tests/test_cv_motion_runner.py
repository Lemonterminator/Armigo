import pytest

from digital_twin_armigo.agent.twin_mujoco import sample_case
from digital_twin_armigo.experiments.contract import EVIDENCE, EXPERIMENT_FIELDS, MOTIONS
from digital_twin_armigo.experiments.runner import group_name


def test_default_group_names_unchanged():
	assert group_name("truth", "fixed", "A", "v7bot") == "truth-v7bot"
	assert group_name("truth", "fixed", "B", "v7bot") == "truth-B-v7bot"
	assert group_name("realistic", "drift", "B", "v7bot") == "realistic-drift-B-v7bot"


def test_variant_suffixes():
	assert (
		group_name("realistic", "drift", "B", "v7bot", motion="w2a", evidence="img")
		== "realistic-drift-B-v7bot-mw2a-eimg"
	)
	assert (
		group_name("realistic", "fixed", "B", "v7bot", motion="open", calib_offset_mm=10)
		== "realistic-fixed-B-v7bot-mopen-c10"
	)
	assert "motion" in EXPERIMENT_FIELDS and "evidence" in EXPERIMENT_FIELDS and "calib_offset_mm" in EXPERIMENT_FIELDS
	assert "w2s" in MOTIONS and "dense-10" in EVIDENCE


def test_calib_offset_is_deterministic_and_sized():
	import numpy as np

	a = sample_case(3_000_101, positive=False, face=0, options={"cameras": "fixed", "calib_offset_mm": 10})
	b = sample_case(3_000_101, positive=False, face=0, options={"cameras": "fixed", "calib_offset_mm": 10})
	assert np.allclose(
		[a.calib_top.x, a.calib_top.y, a.calib_top.yaw_deg], [b.calib_top.x, b.calib_top.y, b.calib_top.yaw_deg]
	)
	assert np.linalg.norm([a.calib_top.x - a.top.x, a.calib_top.y - a.top.y]) * 1000 == pytest.approx(10, abs=0.01)
	assert abs(a.calib_top.yaw_deg - a.top.yaw_deg) == pytest.approx(2.0, abs=1e-6)


def test_drift_and_offset_are_exclusive():
	with pytest.raises(ValueError):
		sample_case(3_000_101, positive=False, face=0, options={"cameras": "drift", "calib_offset_mm": 5})


def test_segmenter_suffix_and_contract():
	from digital_twin_armigo.experiments.contract import SEGMENTERS

	assert SEGMENTERS == ("lab", "unet", "silhouette")
	assert "segmenter" in EXPERIMENT_FIELDS
	assert group_name("realistic", "drift", "B", "v7bot", segmenter="lab") == "realistic-drift-B-v7bot"
	assert (
		group_name("realistic", "drift", "B", "v7bot", motion="w2s", segmenter="unet")
		== "realistic-drift-B-v7bot-mw2s-sunet"
	)
