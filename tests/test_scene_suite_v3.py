"""Tests for v3 scene suite generation and preflight validation (Task 8)."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

from digital_twin_armigo.agent.twin_mujoco import load_suite, sample_case
from digital_twin_armigo.mujoco_cell.palette import HELD_OUT_PAIRS
from digital_twin_armigo.mujoco_cell.scene_v3 import SPLITS
from digital_twin_armigo.vla.contract import HELD_OUT_VARIANT, TRAIN_VARIANTS

ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, rel_path: str):
	if name in sys.modules:
		return sys.modules[name]
	path = ROOT / rel_path
	if not path.exists():
		pytest.fail(f"Expected module {rel_path} to exist")
	spec = importlib.util.spec_from_file_location(name, path)
	mod = importlib.util.module_from_spec(spec)
	sys.modules[name] = mod
	spec.loader.exec_module(mod)
	return mod


def _suite_v3():
	return _load_module("make_scene_suite_v3", "scripts/vla/make_scene_suite_v3.py")


def _legacy_suites():
	return _load_module("make_suites_ref", "scripts/vla/make_suites.py")


def test_train_suite_50_seeds_and_preflight_schema(tmp_path):
	pytest.importorskip("mujoco")
	pytest.importorskip("cv2")
	msv3 = _suite_v3()
	ms = _legacy_suites()

	out_json = tmp_path / "train.json"
	preflight_dir = tmp_path / "preflight"
	rc = msv3.main(
		[
			"--split",
			"train",
			"--seeds",
			"3000001-3000050",
			"--out",
			str(out_json),
			"--preflight-dir",
			str(preflight_dir),
			"--perception-sample",
			"0.1",
			"--max-perception",
			"5",
		]
	)
	assert rc == 0
	assert out_json.exists()
	assert (preflight_dir / "preflight.json").exists()
	assert (preflight_dir / "preflight.md").exists()

	suite = json.loads(out_json.read_text(encoding="utf-8"))
	cases = suite["cases"]
	assert len(cases) == 50
	for c in cases:
		assert isinstance(c, list) and len(c) == 4
		assert isinstance(c[0], int) and isinstance(c[1], bool) and isinstance(c[2], int)
		assert c[3] == {"scene": "v3", "split": "train"}
	assert load_suite(out_json) == [tuple(c) for c in cases]

	mix = (3, 1, 1, 1, 1, 1, 1)
	counts = [sum(not c[1] for c in cases)] + [sum(c[1] and c[2] == f for c in cases) for f in range(6)]
	assert counts == ms.allocate(50, mix)

	preflight = json.loads((preflight_dir / "preflight.json").read_text(encoding="utf-8"))
	assert preflight["pass"] is True
	for key in (
		"balance",
		"colour_pairs",
		"camera_hist",
		"visibility_resamples",
		"perception_check",
		"token_max",
		"factors",
		"rejections",
	):
		assert key in preflight, f"missing preflight key {key!r}"

	assert preflight["perception_check"]["sampled_count"] == 5
	assert len(preflight["perception_check"]["cases"]) == 5
	assert preflight["perception_check"]["crashes"] == 0
	assert isinstance(preflight["token_max"], int) and 1 <= preflight["token_max"] <= 40

	factors = preflight["factors"]
	for fkey in (
		"normal_bowl_colour",
		"defect_bowl_colour",
		"extra_bowl_colour",
		"bowl_count",
		"brick_colour",
		"table_texture",
		"table_v",
		"light_k",
		"instruction_variant",
		"camera_bins",
	):
		assert fkey in factors, f"missing factor {fkey!r}"

	assert preflight["rejections"] == {"reachability": 0, "visibility": 0, "colour_separability": 0}
	assert preflight["colour_pairs"]["held_out_pair_count"] == 0
	assert preflight["colour_pairs"]["held_out_pairs_present"] == []
	assert str(HELD_OUT_VARIANT) not in factors["instruction_variant"]
	assert {int(k) for k in factors["instruction_variant"]}.issubset(set(TRAIN_VARIANTS))

	for seed, pos, face, opts in cases:
		lay = sample_case(seed, pos, face, opts)
		names = [lay.bowl_names["green"], lay.bowl_names["pink"]] + [b[0] for b in lay.extra_bowls]
		assert all(frozenset((a, b)) not in HELD_OUT_PAIRS for i, a in enumerate(names) for b in names[i + 1 :])
		assert lay._scene_v3.variant in TRAIN_VARIANTS


def test_perception_sample_count_without_max_cap(tmp_path):
	pytest.importorskip("mujoco")
	pytest.importorskip("cv2")
	msv3 = _suite_v3()

	out_json = tmp_path / "train25.json"
	preflight_dir = tmp_path / "pf25"
	rc = msv3.main(
		[
			"--split",
			"train",
			"--seeds",
			"3000001-3000025",
			"--out",
			str(out_json),
			"--preflight-dir",
			str(preflight_dir),
			"--perception-sample",
			"0.1",
		]
	)
	assert rc == 0
	preflight = json.loads((preflight_dir / "preflight.json").read_text(encoding="utf-8"))
	assert preflight["perception_check"]["sampled_count"] == math.ceil(0.1 * 25)
	assert len(preflight["perception_check"]["cases"]) == 3


def test_disjoint_seed_ranges_and_out_of_range_rejection(tmp_path):
	msv3 = _suite_v3()
	ms = _legacy_suites()

	ranges = msv3.V3_SPLIT_SEED_RANGES
	assert set(ranges) == set(SPLITS)
	assert ranges["train"] == (3_000_001, 3_099_999)
	assert ranges["val"] == (3_100_001, 3_199_999)
	assert ranges["test"] == (3_900_001, 3_999_999)
	for ood_split in ("ood-color", "ood-cam", "ood-lang", "ood-appearance"):
		lo, hi = ranges[ood_split]
		assert 3_200_001 <= lo <= hi <= 3_899_999

	# Pairwise disjoint across all v3 splits.
	items = list(ranges.items())
	for i, (s1, (lo1, hi1)) in enumerate(items):
		assert lo1 <= hi1
		for s2, (lo2, hi2) in items[i + 1 :]:
			assert hi1 < lo2 or hi2 < lo1, f"v3 seed ranges overlap: {s1} vs {s2}"

	# Zero overlap with legacy RESERVED_RANGES, DEFAULT_SPLITS, and bench_seeds.
	bench = ms.bench_seeds()
	for split, (lo, hi) in ranges.items():
		for rlo, rhi in ms.RESERVED_RANGES:
			assert hi < rlo or lo > rhi, f"{split} overlaps RESERVED_RANGES"
		for dname, (dn, dstart) in ms.DEFAULT_SPLITS.items():
			dlo, dhi = dstart, dstart + dn - 1
			assert hi < dlo or lo > dhi, f"{split} overlaps legacy {dname}"
		assert not any(lo <= b <= hi for b in bench), f"{split} overlaps bench_seeds"

	# Out-of-range --seeds raises ValueError / fails CLI cleanly.
	with pytest.raises(ValueError):
		msv3.parse_seed_range("3100001-3100010", split="train")
	with pytest.raises(ValueError):
		msv3.parse_seed_range("3000001-3000010", split="val")
	with pytest.raises(ValueError):
		msv3.parse_seed_range("500000-500010", split="train")

	with pytest.raises(SystemExit) as exc_info:
		msv3.main(
			[
				"--split",
				"val",
				"--seeds",
				"3000001-3000010",
				"--out",
				str(tmp_path / "bad.json"),
				"--preflight-dir",
				str(tmp_path / "bad_pf"),
			]
		)
	assert exc_info.value.code != 0


def test_ood_color_and_ood_lang_suites(tmp_path):
	pytest.importorskip("mujoco")
	pytest.importorskip("cv2")
	msv3 = _suite_v3()

	color_out = tmp_path / "ood_color.json"
	color_pf = tmp_path / "pf_color"
	assert (
		msv3.main(
			[
				"--split",
				"ood-color",
				"--seeds",
				"3200001-3200012",
				"--out",
				str(color_out),
				"--preflight-dir",
				str(color_pf),
				"--max-perception",
				"2",
			]
		)
		== 0
	)
	color_suite = json.loads(color_out.read_text(encoding="utf-8"))
	color_preflight = json.loads((color_pf / "preflight.json").read_text(encoding="utf-8"))
	assert color_preflight["colour_pairs"]["held_out_pair_count"] == 12
	for seed, pos, face, opts in color_suite["cases"]:
		lay = sample_case(seed, pos, face, opts)
		assert frozenset((lay.bowl_names["green"], lay.bowl_names["pink"])) in HELD_OUT_PAIRS

	lang_out = tmp_path / "ood_lang.json"
	lang_pf = tmp_path / "pf_lang"
	assert (
		msv3.main(
			[
				"--split",
				"ood-lang",
				"--seeds",
				"3400001-3400012",
				"--out",
				str(lang_out),
				"--preflight-dir",
				str(lang_pf),
				"--max-perception",
				"2",
			]
		)
		== 0
	)
	lang_suite = json.loads(lang_out.read_text(encoding="utf-8"))
	lang_preflight = json.loads((lang_pf / "preflight.json").read_text(encoding="utf-8"))
	assert lang_preflight["factors"]["instruction_variant"] == {str(HELD_OUT_VARIANT): 12}
	for seed, pos, face, opts in lang_suite["cases"]:
		lay = sample_case(seed, pos, face, opts)
		assert lay._scene_v3.variant == HELD_OUT_VARIANT


def test_rejection_counting_and_seed_advancement(monkeypatch, tmp_path):
	msv3 = _suite_v3()
	ms = _legacy_suites()

	orig_reach = msv3.check_reachability
	orig_vis = msv3.check_visibility
	orig_col = msv3.check_colour_separability

	def fake_reach(layout):
		if layout.defect_seed == 19_199 + 3000002 * 100_019:
			return False, "mock reachability failure"
		return orig_reach(layout)

	def fake_vis(layout):
		if layout.defect_seed == 19_199 + 3000004 * 100_019:
			return False, "mock visibility failure"
		return orig_vis(layout)

	def fake_col(layout, split):
		if layout.defect_seed == 19_199 + 3000005 * 100_019:
			return False, "mock colour failure"
		return orig_col(layout, split)

	monkeypatch.setattr(msv3, "check_reachability", fake_reach)
	monkeypatch.setattr(msv3, "check_visibility", fake_vis)
	monkeypatch.setattr(msv3, "check_colour_separability", fake_col)

	suite, preflight = msv3.build_suite_and_preflight(
		split="train",
		start_seed=3_000_001,
		end_seed=3_000_005,
		perception_sample=0.0,
		max_perception=0,
	)
	cases = suite["cases"]
	assert [c[0] for c in cases] == [3_000_001, 3_000_003, 3_000_006, 3_000_007, 3_000_008]
	assert preflight["rejections"] == {"reachability": 1, "visibility": 1, "colour_separability": 1}

	mix = (3, 1, 1, 1, 1, 1, 1)
	counts = [sum(not c[1] for c in cases)] + [sum(c[1] and c[2] == f for c in cases) for f in range(6)]
	assert counts == ms.allocate(5, mix)

	# Also verify RuntimeError classification from sample_case itself.
	def raising_sample_case(seed, pos, face, opts):
		if seed == 3_000_002:
			raise RuntimeError("no layout satisfied the spacing and reach limits")
		if seed == 3_000_003:
			raise RuntimeError("v3 wide camera failed workspace_visible after 20 resamples")
		if seed == 3_000_004:
			raise RuntimeError("Failed to sample valid bowl colours after 100 attempts")
		return sample_case(seed, pos, face, opts)

	monkeypatch.setattr(msv3, "check_reachability", orig_reach)
	monkeypatch.setattr(msv3, "check_visibility", orig_vis)
	monkeypatch.setattr(msv3, "check_colour_separability", orig_col)
	monkeypatch.setattr(msv3, "sample_case", raising_sample_case)
	suite2, pf2 = msv3.build_suite_and_preflight(
		split="train",
		start_seed=3_000_001,
		end_seed=3_000_004,
		perception_sample=0.0,
		max_perception=0,
	)
	assert [c[0] for c in suite2["cases"]] == [3_000_001, 3_000_005, 3_000_006, 3_000_007]
	assert pf2["rejections"] == {"reachability": 1, "visibility": 1, "colour_separability": 1}


def test_all_v3_splits_build_cleanly():
	msv3 = _suite_v3()
	for split in SPLITS:
		lo, _ = msv3.V3_SPLIT_SEED_RANGES[split]
		suite, pf = msv3.build_suite_and_preflight(
			split=split,
			start_seed=lo,
			end_seed=lo + 4,
			perception_sample=0.0,
			max_perception=0,
		)
		assert suite["split"] == split
		assert suite["sealed"] is (split == "test")
		assert len(suite["cases"]) == 5
		assert pf["pass"] is True


def test_determinism_across_runs(tmp_path):
	pytest.importorskip("mujoco")
	pytest.importorskip("cv2")
	msv3 = _suite_v3()

	out1 = tmp_path / "run1/train.json"
	pf1 = tmp_path / "run1/pf"
	out2 = tmp_path / "run2/train.json"
	pf2 = tmp_path / "run2/pf"

	args1 = [
		"--split",
		"train",
		"--seeds",
		"3000001-3000020",
		"--out",
		str(out1),
		"--preflight-dir",
		str(pf1),
		"--perception-sample",
		"0.1",
		"--max-perception",
		"2",
	]
	args2 = [
		"--split",
		"train",
		"--seeds",
		"3000001-3000020",
		"--out",
		str(out2),
		"--preflight-dir",
		str(pf2),
		"--perception-sample",
		"0.1",
		"--max-perception",
		"2",
	]
	assert msv3.main(args1) == 0
	assert msv3.main(args2) == 0
	assert out1.read_text(encoding="utf-8") == out2.read_text(encoding="utf-8")
	p1 = json.loads((pf1 / "preflight.json").read_text(encoding="utf-8"))
	p2 = json.loads((pf2 / "preflight.json").read_text(encoding="utf-8"))
	p1.pop("command")
	p2.pop("command")
	assert p1 == p2
