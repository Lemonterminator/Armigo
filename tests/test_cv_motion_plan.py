"""CV motion matrix (WP9): single-source plan, group names, loop-01 suite and validation."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

from digital_twin_armigo.agent.grasp_loop import VARIANTS
from digital_twin_armigo.experiments.runner import group_name
from digital_twin_armigo.governance.segments import split_of

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "config/agent_bench/v2/manifest.json"
CAMERAS = ("fixed", "recal", "drift")


def _module():
	spec = importlib.util.spec_from_file_location("cv_motion_plan", ROOT / "scripts/experiments/cv_motion_plan.py")
	assert spec is not None and spec.loader is not None
	mod = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(mod)
	return mod


PLAN = _module()


def _runs(plans: list[dict], stage: str | None = None) -> list[dict]:
	return [p for p in plans if p["kind"] == "run" and (stage is None or p["stage"] == stage)]


def _all_plans() -> list[dict]:
	plans: list[dict] = []
	for stage in PLAN.ALL_STAGES:
		plans.extend(PLAN.plan(stage))
	return plans


def _errors(plans: list[dict]) -> list[str]:
	return PLAN.validate(plans, MANIFEST)


def test_motions_match_grasp_loop_variants():
	assert PLAN.MOTIONS == tuple(VARIANTS)
	assert set(PLAN.MOTIONS) == {"open", "w1s", "w2s", "w3s", "w2a", "w3a", "w4a"}


def test_stage2_compares_existing_groups():
	plans = PLAN.plan("stage2")
	groups = {p["group"] for p in _runs(plans)}
	compares = [p["compare"] for p in plans if p["kind"] == "compare"]
	assert ["realistic-drift-B-v7bot", "realistic-drift-B-v7bot-mw2s"] in compares
	for group_a, group_b in compares:
		assert group_a in groups and group_b in groups
	assert "realistic-drift-B-v7bot-mw2s-ehold" not in groups
	everywhere = {p["group"] for p in _all_plans() if p["kind"] == "run"}
	assert "realistic-drift-B-v7bot-mw2s-ehold" not in everywhere


def test_stage4_uses_registered_loop_01_96_suite():
	manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
	runs = _runs(PLAN.plan("stage4"))
	assert {p["suite"] for p in runs} == {"loop_01_96"}
	meta = manifest["suites"]["loop_01_96"]
	assert meta == {
		"sha256": "972680c0c65269e6ebb35bef10a28500d239292438240cb1c4f2439ea49b766f",
		"cases": 96,
		"layouts": 48,
		"sealed": False,
	}
	path = MANIFEST.parent / "loop_01_96.json"
	assert hashlib.sha256(path.read_bytes()).hexdigest() == meta["sha256"]
	cases = json.loads(path.read_text(encoding="utf-8"))
	seeds = {c[0] for c in cases}
	assert len(cases) == 96 and len(seeds) == 48
	assert all(split_of(seed) == "loop-01" for seed in seeds)
	dev = {c[0] for c in json.loads((MANIFEST.parent / "p2_dev12.json").read_text(encoding="utf-8"))}
	assert not seeds & dev
	assert not any(p["suite"].endswith(".json") for p in _all_plans() if p["kind"] == "run")


def test_stage2_covers_every_motion_and_camera():
	runs = _runs(PLAN.plan("stage2"))
	combos = {(p["motion"], p["camera"]) for p in runs}
	assert combos == {(motion, camera) for motion in VARIANTS for camera in CAMERAS}
	assert len(runs) == len(VARIANTS) * len(CAMERAS)


def test_groups_match_runner_group_name():
	for entry in _runs(PLAN.plan("stage4")) + _runs(PLAN.plan("stage3b")):
		assert entry["group"] == group_name(
			entry["sensing"],
			entry["camera"],
			entry["strategy"],
			entry["detector"],
			motion=entry["motion"],
			evidence=entry["evidence"],
			calib_offset_mm=entry["calib_offset_mm"],
			segmenter=entry["segmenter"],
		)
	offset_runs = [p for p in _runs(PLAN.plan("stage1")) if p["calib_offset_mm"] == 10]
	assert offset_runs and all("-c10" in p["group"] for p in offset_runs)


def test_validate_accepts_default_plans():
	assert _errors(_all_plans()) == []
	assert _errors(PLAN.plan("stage3b")) == []


def test_validate_rejects_suite_path():
	plans = PLAN.plan("stage4")
	for entry in _runs(plans):
		entry["suite"] = "config/agent_bench/v2/loop_01_96.json"
	assert any("registered suite name" in error for error in _errors(plans))


def test_validate_rejects_unknown_suite():
	plans = PLAN.plan("stage4")
	for entry in _runs(plans):
		entry["suite"] = "no_such_suite"
	assert any("unknown suite" in error for error in _errors(plans))


def test_validate_rejects_winner_motion_outside_stage2():
	plans = PLAN.plan("stage4")
	for entry in _runs(plans):
		if not (entry["motion"] == "open" and entry["evidence"] == "hold"):
			entry["motion"] = "w9s"
	assert any("not part of the stage2 matrix" in error for error in _errors(plans))


def test_validate_rejects_missing_open_baseline():
	plans = [
		entry
		for entry in PLAN.plan("stage4")
		if not (entry["kind"] == "run" and entry["motion"] == "open" and entry["evidence"] == "hold")
	]
	assert any("missing the open+hold baseline" in error for error in _errors(plans))


def test_validate_rejects_compare_group_without_run():
	plans = PLAN.plan("stage2")
	plans.append({"kind": "compare", "stage": "stage2", "batch": "cvm-s2", "compare": ["a", "b"]})
	assert any("not produced by a run" in error for error in _errors(plans))


def test_emit_commands_stage2():
	lines = PLAN.emit_commands(PLAN.plan("stage2"))
	assert all(line.startswith('"$PY"') for line in lines)
	assert sum("experiments.runner run" in line for line in lines) == len(VARIANTS) * len(CAMERAS)
	assert any("experiments.report --root" in line for line in lines)
	assert not any("-ehold" in line for line in lines)


def _cli(*args: str) -> subprocess.CompletedProcess:
	env = {
		key: value
		for key, value in os.environ.items()
		if key not in ("STAGE3_MOTIONS", "WINNER_MOTION", "WINNER_EVIDENCE", "WINNER_SEGMENTER", "CONFIRM_SUITE")
	}
	env["PYTHONPATH"] = "src"
	return subprocess.run(
		[sys.executable, "scripts/experiments/cv_motion_plan.py", *args],
		cwd=ROOT,
		env=env,
		capture_output=True,
		text=True,
		check=False,
	)


def test_plan_only_cli_all_stages():
	result = _cli("--plan-only", "--stage", "all")
	assert result.returncode == 0, result.stderr
	summary = json.loads(result.stdout)
	assert set(summary) == {"stage1", "stage2", "stage3", "stage4"}
	assert summary["stage2"]["n_runs"] == len(VARIANTS) * len(CAMERAS)
	assert summary["stage4"]["suite"] == "loop_01_96"
	assert summary["stage4"]["n_cases"] == 96


def test_plan_only_cli_stage3b():
	result = _cli("--plan-only", "--stage", "stage3b")
	assert result.returncode == 0, result.stderr
	assert set(json.loads(result.stdout)) == {"stage3b"}
