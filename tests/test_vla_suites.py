"""VLA case suites (plan S3): sample_case options, load_suite and scripts/vla/make_suites.py."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from digital_twin_armigo.agent.twin_mujoco import load_suite, sample_case
from digital_twin_armigo.camera import CameraPose, look_from
from digital_twin_armigo.mujoco_cell.layout import sample_layout

ROOT = Path(__file__).resolve().parents[1]
BENCH16 = [tuple(c) for c in json.loads((ROOT / "config/agent_bench/bench16.json").read_text(encoding="utf-8"))]
SPEC = importlib.util.spec_from_file_location("make_suites", ROOT / "scripts/vla/make_suites.py")
ms = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ms)


def _legacy_sample_case(seed, positive, face):
	"""sample_case as it was before the options (commit acd732b), kept verbatim as the reference."""
	raw = json.loads((ROOT / "config/sim/top_camera_joint.json").read_text(encoding="utf-8"))
	layout = sample_layout(np.random.default_rng(seed), CameraPose(**raw["camera"]), defective=positive)
	layout.top = CameraPose(0.4895033468, 0.0, 0.5130680468, 180.0, 25.0712240655, 0.0, 612.47)
	layout.front = look_from((0.20, 0.0), 0.55, 0.35, 35, focal_px=560)
	layout.defect_kind = "cross_var"
	layout.defect_face = face
	layout.defect_seed = 19_199 + seed * 100_019
	layout.clutter = []
	layout.distractors = []
	return layout


@pytest.mark.parametrize("case", BENCH16)
def test_bench16_layouts_unchanged(case):
	ref = repr(_legacy_sample_case(*case))
	assert repr(sample_case(*case)) == ref
	assert repr(sample_case(*case, None)) == ref
	assert repr(sample_case(*case, {})) == ref
	assert repr(sample_case(*case, {"cameras": "fixed"})) == ref


def test_camera_jitter_changes_only_the_cameras():
	for seed in (600001, 1_100_001, 1_100_002):
		fixed = sample_case(seed, True, 2)
		jit = sample_case(seed, True, 2, {"cameras": "jitter"})
		assert jit.top != fixed.top and jit.front != fixed.front
		assert repr(replace(jit, top=fixed.top, front=fixed.front)) == repr(fixed)
		assert repr(sample_case(seed, True, 2, {"cameras": "jitter"})) == repr(jit)  # deterministic per seed
	assert sample_case(1_100_001, False, 5, {"cameras": "jitter"}).top != jit.top  # differs across seeds


def test_prior_option_uses_the_real_layouts():
	prior = json.loads((ROOT / "config/sim/prior_grasp.json").read_text(encoding="utf-8"))
	bricks = np.array([p["brick"] for p in prior])
	lay = sample_case(1_100_001, False, 5, {"prior": "config/sim/prior_grasp.json"})
	assert np.min(np.hypot(*(bricks - lay.brick_xy).T)) < 0.15  # jittered real brick position
	assert lay.top == sample_case(1_100_001, False, 5).top  # cameras unaffected by the prior


@pytest.mark.parametrize("bad", [{"cameras": "orbit"}, {"prior": 3}, {"unknown": 1}, ["jitter"]])
def test_invalid_case_options(bad):
	with pytest.raises(ValueError):
		sample_case(600001, False, 5, bad)


def test_load_suite_formats(tmp_path):
	assert load_suite(ROOT / "config/agent_bench/bench16.json") == BENCH16
	obj = tmp_path / "s.json"
	obj.write_text(json.dumps({"cases": [[1, True, 0]], "options": {"cameras": "fixed"}}), encoding="utf-8")
	assert load_suite(obj) == [(1, True, 0)]  # default options are dropped
	obj.write_text(json.dumps({"cases": [[1, True, 0]], "options": {"cameras": "jitter"}}), encoding="utf-8")
	assert load_suite(obj) == [(1, True, 0, {"cameras": "jitter"})]
	assert sample_case(*load_suite(obj)[0]).top != sample_case(1, True, 0).top
	obj.write_text(json.dumps({"cases": [[1, True, 0, {}]], "options": {"cameras": "jitter"}}), encoding="utf-8")
	with pytest.raises(ValueError):
		load_suite(obj)


def test_allocation_and_bottom_share():
	mix = (3, 1, 1, 1, 1, 1, 1)
	for n in (9, 297, 900):
		counts = ms.allocate(n, mix)
		assert counts == [n // 3] + [n // 9] * 6  # exact mix
		cases = ms.make_cases(n, 1_100_001, mix)
		assert Fraction(sum(ms.uses_bottom(c) for c in cases), n) == Fraction(4, 9)
	for n in (1, 60, 100, 300, 301):
		counts = ms.allocate(n, mix)
		assert sum(counts) == n
		assert all(abs(c - Fraction(n * w, 9)) < 1 for c, w in zip(counts, mix, strict=True))
	order = ms.interleave(ms.allocate(90, mix))
	assert sorted(set(order[:9])) == list(range(7))  # prefixes track the mix


def test_committed_suites(tmp_path):
	suites = {n: json.loads((ROOT / f"config/vla/{n}.json").read_text(encoding="utf-8")) for n in ms.DEFAULT_SPLITS}
	seeds = {n: [c[0] for c in s["cases"]] for n, s in suites.items()}
	all_seeds = [x for v in seeds.values() for x in v]
	assert len(all_seeds) == len(set(all_seeds))  # one case per layout, splits disjoint
	assert not set(all_seeds) & {c[0] for c in BENCH16}
	assert not set(all_seeds) & ms.bench_seeds()
	assert min(all_seeds) >= 1_000_000
	assert [len(v) for v in seeds.values()] == [300, 60, 100]
	assert suites["test"]["sealed"] is True and not suites["train"]["sealed"] and not suites["val"]["sealed"]
	for name, s in suites.items():
		mix = (s["class_mix"]["normal"], *s["class_mix"]["defect_faces_0_to_5"])
		assert mix == (3, 1, 1, 1, 1, 1, 1)
		counts = [sum(not c[1] for c in s["cases"])] + [sum(c[1] and c[2] == f for c in s["cases"]) for f in range(6)]
		assert counts == ms.allocate(len(s["cases"]), mix)
		assert load_suite(ROOT / f"config/vla/{name}.json") == [tuple(c) for c in s["cases"]]
	report = json.loads((ROOT / "config/vla/report.json").read_text(encoding="utf-8"))
	assert report["splits"]["train"]["skills"]["bottom_presentation"] == 133
	# the committed files are exactly what the generator writes
	assert ms.main(["--out", str(tmp_path), "--no-variants"]) == 0
	for name in ms.DEFAULT_SPLITS:
		assert (tmp_path / f"{name}.json").read_text(encoding="utf-8") == (ROOT / f"config/vla/{name}.json").read_text(
			encoding="utf-8"
		)
