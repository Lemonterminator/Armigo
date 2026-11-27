"""Round 3 (drift alarm on ring rejection, 5f36960): constants for driver2b.py / run2b.py / gates.py.

	LOOP_CONFIG=round3_config python run2b.py

Only the drift condition runs: the fix lives behind drift_alarm, which is on for drift episodes only, so fixed
and recal episodes take the unchanged code path.
"""

import os
from pathlib import Path

ROOT = Path(os.environ.get("ROUND3_ROOT", "E:/Armigo_data_v2_r2"))
TAG = os.environ.get("ROUND3_TAG", "round-03-drift-20261127-5f36960")
PREV = "round-02b-grasp-20261127-8902081"  # base rows for bank and regression (drift rows of its candidate arm)
LOG_SUFFIX = "" if "ROUND3_TAG" not in os.environ else "-" + TAG
OUT = ROOT / "reports" / TAG
SUITES = OUT / "suite-source"
CAMERAS = ("drift",)
HERE = Path(__file__).resolve().parent
LOG_DIR = Path(os.environ.get("ROUND3_LOGS", "E:/scratch/armigo-round03-drift"))
PYTHON = Path("E:/digital_twin_armigo/.venv/Scripts/python.exe")
DRIVER_LIB = HERE
GATES = "gates.py"
ROUND_SPEC = HERE.parents[2] / "config" / "experiments" / "loop" / "round_03.json"  # bank, criteria, batches
SUITE_NAMES = ("round_00_clutter_60", "round_01_fresh_192", "round_02b_fresh_192", "round_03_fresh_192")
RERUNS = {"rerun00": "round_00_clutter_60", "rerun01": "round_01_fresh_192", "rerun02b": "round_02b_fresh_192"}

CHECKOUTS = {
	"candidate": Path("E:/digital_twin_armigo-collection"),  # 5f36960 + suite registration
	"base": Path("E:/scratch/armigo-bowl-replacement-worker"),  # a57dab3 (src = 8902081, the round-2b candidate)
}


def batches(arm: str) -> list[tuple[str, str, str]]:
	out = [(f"{TAG}-{name}", suite, "drift") for name, suite in RERUNS.items()] if arm == "candidate" else []
	return out + [(f"{TAG}-fresh-{arm}", "round_03_fresh_192", "drift")]


def pythonpath(arm: str) -> str:
	co = CHECKOUTS[arm]
	return ";".join(str(p) for p in (co / "src", co / "scripts" / "experiments"))
