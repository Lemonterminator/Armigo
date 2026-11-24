"""Every truth diagnostic the simulator writes into events must be on the sensor-only deny list."""

import ast
import re
from pathlib import Path

import pytest

pytest.importorskip("mujoco")

from digital_twin_armigo.agent import step_mcp as sm

SRC = Path(sm.__file__).resolve().parents[1]
DIAG = re.compile(r"^(true_|actual_|error_|belief_error)|_err(_|$)|_correct$|truth|^calibrated_on$")


def _event_keys(path: Path) -> set[str]:
	"""Keyword names passed to ``*.event(...)``, ``info["..."] = ...`` keys, and string keys of dict literals."""
	tree = ast.parse(path.read_text(encoding="utf-8"))
	keys: set[str] = set()
	for node in ast.walk(tree):
		if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "event":
			keys |= {k.arg for k in node.keywords if k.arg}
		elif isinstance(node, ast.Assign):
			for t in node.targets:
				if (
					isinstance(t, ast.Subscript)
					and isinstance(t.slice, ast.Constant)
					and isinstance(t.slice.value, str)
				):
					keys.add(t.slice.value)
		elif isinstance(node, ast.Dict):
			keys |= {k.value for k in node.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)}
	return keys


def test_truth_diagnostics_are_denied():
	assert sm.TRUTH_DIAGNOSTIC_KEYS <= sm.SIMULATOR_KEYS


@pytest.mark.parametrize("rel", ["agent/twin_mujoco.py", "mujoco_cell/handeye.py"])
def test_every_diagnostic_key_written_is_registered(rel):
	written = {k for k in _event_keys(SRC / rel) if DIAG.search(k)}
	missing = written - sm.TRUTH_DIAGNOSTIC_KEYS
	assert not missing, f"{rel} writes truth diagnostics not in TRUTH_DIAGNOSTIC_KEYS: {sorted(missing)}"


def test_calibration_drift_fields_are_rejected():
	for key in ("actual_cam_pos_err_mm", "actual_cam_axis_err_deg", "calibrated_on"):
		with pytest.raises(sm.SimulatorLeak):
			sm.assert_sensor_only({"camera": "top", key: 1.0})
