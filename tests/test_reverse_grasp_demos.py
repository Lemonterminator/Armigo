"""Smoke test of scripts/vla/reverse_grasp_demos.py: one v3 train case, no video, CPU only."""

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "vla" / "reverse_grasp_demos.py"


def _load():
	spec = importlib.util.spec_from_file_location("reverse_grasp_demos", SCRIPT)
	mod = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(mod)
	return mod


def test_canonical_pick_prefers_nearest_clear_roll():
	mod = _load()
	cands = [
		{"roll": -2.0, "clearance_m": 0.02, "gap_ok": True},
		{"roll": 1.1, "clearance_m": 0.0, "gap_ok": True},
		{"roll": 0.9, "clearance_m": 0.02, "gap_ok": False},
	]
	c, fell_back = mod.canonical_pick(cands, 1.0, 0.002)
	assert fell_back and c["roll"] == -2.0
	cands[1]["clearance_m"] = 0.01
	c, fell_back = mod.canonical_pick(cands, 1.0, 0.002)
	assert not fell_back and c["roll"] == 1.1


def test_one_case_writes_episode(tmp_path):
	pytest.importorskip("mujoco")
	mod = _load()
	rec = mod.run_case(mod.TRAIN_SEED0, tmp_path, False, 2, 0.5, "canonical", "near")
	assert rec["ok"], rec
	ep = tmp_path / "episodes" / f"ep_{mod.TRAIN_SEED0}"
	rows = np.load(ep / "branch-0" / "actions.npz")
	assert rows["action"].shape == rows["state"].shape == (rec["n_rows"], 6)
	assert set(rows["segment"].tolist()) == {0, 1}  # grasp, lift

	from digital_twin_armigo.agent.twin_record import segment_labels

	labels = segment_labels(ep)
	assert [s["kind"] for s in labels["segments"]] == ["grasp", "lift"]
	assert "wrist roll" in labels["segments"][0]["instruction"]
