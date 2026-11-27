"""Shared constants for driver2b.py / run2b.py / gates.py (no heavy imports: the launcher reads it too)."""

import os
from pathlib import Path

ROOT = Path(os.environ.get("ROUND2B_ROOT", "E:/Armigo_data_v2_r2"))  # D: is full; round-1 rows stay on D:
ROOT1 = Path("D:/Armigo_data_v2")  # round-1 rows
TAG = os.environ.get("ROUND2B_TAG", "round-02b-grasp-20261127-8902081")  # ROUND2B_*: tests only
R1 = "round-01-bowl-20261126-1a1ca84"
LOG_SUFFIX = "" if "ROUND2B_TAG" not in os.environ else "-" + TAG  # keeps test logs apart
OUT = ROOT / "reports" / TAG
SUITES = OUT / "suite-source"  # a test root needs a copy of the round's suite-source
CAMERAS = ("fixed", "recal", "drift")
HERE = Path(__file__).resolve().parent
PYTHON = Path("E:/digital_twin_armigo/.venv/Scripts/python.exe")
DRIVER_LIB = HERE  # resume_loop.py (copied from E:/scratch/armigo-loop-readiness-driver)

# Code under test per arm. The gates always score with the candidate checkout.
CHECKOUTS = {
	"candidate": Path("E:/digital_twin_armigo-collection"),  # 8902081 + suite registration
	"base": Path("E:/scratch/armigo-bowl-replacement-worker"),  # aa9eb45
}


def pythonpath(arm: str) -> str:
	co = CHECKOUTS[arm]
	return ";".join(str(p) for p in (co / "src", co / "scripts" / "experiments"))

GATES = "gates.py"
ROUND_SPEC = HERE.parents[2] / "config" / "experiments" / "loop" / "round_02b.json"  # bank, criteria, batches
