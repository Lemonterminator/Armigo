import json

import numpy as np

from digital_twin_armigo.experiments.report import write_report
from digital_twin_armigo.experiments.stats import (
	cell_metrics,
	metric_with_ci,
	minimum_detectable_effect,
	paired_comparison,
)


def rows(success_by_layout, group="g"):
	out = []
	for lay, (s_n, s_d) in success_by_layout.items():
		for kind, ok in (("N", s_n), ("D2", s_d)):
			out.append(
				{
					"layout_id": f"L{lay}",
					"case_id": f"L{lay}-{kind}",
					"group": group,
					"expected_label": "normal" if kind == "N" else "defective",
					"physical_success": bool(ok),
					"defect_miss": kind != "N" and not ok,
					"false_alarm": kind == "N" and not ok,
					"is_unknown": False,
					"wrong_bowl": False,
					"n_review_views": 1,
				}
			)
	return out


def test_ci_contains_truth():
	rng = np.random.default_rng(1)
	data = {i: (rng.random() < 0.8, rng.random() < 0.8) for i in range(60)}
	m = metric_with_ci(rows(data), "success_rate", B=500, seed=3)
	assert m["ci_lo"] <= 0.8 <= m["ci_hi"]
	assert m["n"] == 120 and m["n_clusters"] == 60
	again = metric_with_ci(rows(data), "success_rate", B=500, seed=3)
	assert again == m


def test_cell_metrics_denominators():
	data = {1: (True, False), 2: (False, False), 3: (True, True)}
	m = cell_metrics(rows(data), B=200)
	assert m["success_rate"]["value"] == 3 / 6
	assert m["defect_miss_rate"]["n"] == 3 and m["defect_miss_rate"]["k"] == 2
	assert m["false_alarm_rate"]["n"] == 3 and m["false_alarm_rate"]["k"] == 1
	assert m["extra_views"] == 1


def test_paired_difference_sign_and_counts():
	a = rows({i: (i % 2 == 0, False) for i in range(20)}, "A")
	b = rows({i: (True, i < 10) for i in range(20)}, "B")
	pc = paired_comparison(a, b, B=500)
	assert pc["diff"] > 0 and pc["ci_lo"] > 0
	assert pc["n_pairs"] == 40
	assert pc["n10"] == 0 and pc["n01"] == 20
	assert pc["n11"] + pc["n00"] + pc["n10"] + pc["n01"] == 40
	rev = paired_comparison(b, a, B=500)
	assert rev["diff"] < 0 and rev["n10"] == pc["n01"]
	assert pc["p_mcnemar"] < 0.05


def test_paired_ignores_unmatched():
	a = rows({1: (True, True), 2: (True, True)}, "A")
	b = rows({1: (True, True)}, "B")
	pc = paired_comparison(a, b, B=100)
	assert pc["n_pairs"] == 2 and pc["unmatched_a"] == 2 and pc["diff"] == 0


def test_mde_monotonic():
	assert minimum_detectable_effect(0.2, 24) > minimum_detectable_effect(0.1, 24)
	assert minimum_detectable_effect(0.2, 24) > minimum_detectable_effect(0.2, 48)
	assert minimum_detectable_effect(0.2, 24, power=0.9) > minimum_detectable_effect(0.2, 24, power=0.8)
	assert minimum_detectable_effect(0.0, 24) == 0.0


def test_report_end_to_end(tmp_path):
	for group, ok in (("A", False), ("B", True)):
		for lay in (1, 2, 3):
			d = tmp_path / "runs" / "bt" / "mujoco" / group / f"L{lay}-N"
			d.mkdir(parents=True)
			(d / "task.json").write_text(json.dumps({"case": [lay, False, 0]}))
			result = {"verdict": "SUCCESS" if ok else "FAILED", "label": "normal", "destination": "green", "views": 0}
			(d / "episode.json").write_text(json.dumps({"result": result}))
		(tmp_path / "runs" / "bt" / "mujoco" / group / "plan.json").write_text(
			json.dumps({"cases": ["L1-N", "L2-N", "L3-N", "L4-N"]})
		)
	broken = tmp_path / "runs" / "bt" / "mujoco" / "B" / "L4-N"
	broken.mkdir()
	(broken / "run_status.json").write_text(json.dumps({"state": "interrupted"}))
	out = write_report(tmp_path, "bt", ("A", "B"), B=100)
	summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
	assert summary["cells"]["mujoco/B"]["coverage"] == {
		"planned": 4,
		"completed": 3,
		"infra_failed": 1,
		"missing": [],
	}
	assert summary["cells"]["mujoco/A"]["coverage"]["missing"] == ["L4-N"]
	assert summary["paired"]["mujoco"]["diff"] > 0
	assert (out / "scores.jsonl").read_text(encoding="utf-8").count("\n") == 7
	text = (out / "SUMMARY.md").read_text(encoding="utf-8")
	assert "http://127.0.0.1:8767/inventory?q=L1-N" in text
