"""The equivalence gate must reject missing data and numerical regressions."""

import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
	"compare_outputs", Path(__file__).resolve().parents[1] / "scripts/sim/compare_outputs.py"
)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


def test_comparison_tolerance_does_not_hide_state_changes(tmp_path):
	a, b = tmp_path / "a", tmp_path / "b"
	a.mkdir()
	b.mkdir()
	np.savez(a / "sim.npz", front=np.array([0, 255], np.uint8), state=np.array([1.0]))
	np.savez(b / "sim.npz", front=np.array([1, 254], np.uint8), state=np.array([1.0]))
	assert not checker.compare_outputs(a, b)["passed"]
	assert checker.compare_outputs(a, b, rgb_atol=1)["passed"]
	np.savez(b / "sim.npz", front=np.array([1, 254], np.uint8), state=np.array([1.00001]))
	assert not checker.compare_outputs(a, b, rgb_atol=1)["passed"]


def test_comparison_rejects_missing_files_metadata_dtype_and_empty_input(tmp_path):
	a, b = tmp_path / "a", tmp_path / "b"
	a.mkdir()
	b.mkdir()
	assert not checker.compare_outputs(a, b)["passed"]
	np.savez(a / "sim.npz", state=np.array([1], np.int32))
	assert not checker.compare_outputs(a, b)["passed"]
	np.savez(b / "sim.npz", state=np.array([1], np.int64))
	assert not checker.compare_outputs(a, b)["passed"]
	np.savez(b / "sim.npz", state=np.array([1], np.int32))
	assert checker.compare_outputs(a, b)["passed"]
	(a / "sim.json").write_text('{"success": true}')
	(b / "sim.json").write_text('{"success": false}')
	assert not checker.compare_outputs(a, b)["passed"]
