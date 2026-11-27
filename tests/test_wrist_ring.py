import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest

pytest.importorskip("mujoco")

from digital_twin_armigo.agent.act_grasp import grasp_trial


def test_ring_localise_replaces_top_view_xy(tmp_path):
	case = [3001201, True, 3001201 % 6, {"scene": "v3", "split": "train", "cameras": "drift", "oracle": "bowls"}]
	r = grasp_trial(case, tmp_path / "ep", expert_overrides={"descend_close_ratio": 1.3}, ring_localise=True)
	ring = r["ring"]
	assert ring["poses"] >= 6 and ring["fixes"] >= 3 and ring["accepted"]
	assert ring["error_xy_mm"] < 3.0 < ring["error_xy_mm_before"]
	events = (tmp_path / "ep" / "events.jsonl").read_text(encoding="utf-8")
	assert '"kind": "ring_localised"' in events
