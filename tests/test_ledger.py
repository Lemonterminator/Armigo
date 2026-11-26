"""Tests for failure ledger clustering, human review overrides, factor skew, and CLI (Task F2)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from digital_twin_armigo.experiments.ledger import (
	build_ledger,
	extract_episode_factors,
	render_ledger_md,
)
from digital_twin_armigo.experiments.triage import TriageRow
from digital_twin_armigo.inventory import audit


def _load_cli():
	path = Path(__file__).resolve().parents[1] / "scripts" / "experiments" / "failure_ledger.py"
	spec = importlib.util.spec_from_file_location("failure_ledger", path)
	assert spec is not None and spec.loader is not None
	mod = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(mod)
	return mod.main


def _row(
	traj_id: str,
	*,
	stage: str = "grasp",
	code: str | None = "no_clamp",
	root_cause: str = "calibration_stale",
	rule: str = "R3",
	evidence: dict | None = None,
) -> TriageRow:
	return TriageRow(
		traj_id=traj_id,
		stage=stage,
		code=code,
		root_cause=root_cause,
		evidence=evidence or {},
		rule=rule,
	)


def _write_synth_episode(
	ep_dir: Path,
	*,
	case: list,
	result: dict,
	events: list[dict],
	experiment: dict,
	scene_layout: dict | None = None,
) -> None:
	ep_dir.mkdir(parents=True, exist_ok=True)
	(ep_dir / "task.json").write_text(json.dumps({"case": case}), encoding="utf-8")
	(ep_dir / "episode.json").write_text(
		json.dumps({"schema": "armigo-agent-episode/v1", "provenance": {"experiment": experiment}, "result": result}),
		encoding="utf-8",
	)
	(ep_dir / "events.jsonl").write_text(
		"\n".join(json.dumps(ev) for ev in events) + ("\n" if events else ""),
		encoding="utf-8",
	)
	(ep_dir / "run_status.json").write_text(json.dumps({"state": "done"}), encoding="utf-8")
	if scene_layout is not None:
		(ep_dir / "replay").mkdir(exist_ok=True)
		(ep_dir / "replay" / "scene.json").write_text(json.dumps({"layout": scene_layout}), encoding="utf-8")


def test_build_ledger_clusters_counts_impact_share_and_sort():
	rows = [
		_row("ok-1", stage="ok", code=None, root_cause="none", rule="none"),
		_row("ok-2", stage="ok", code=None, root_cause="none", rule="none"),
		# 3x calibration_stale (loss 1.0 each -> impact 3.0)
		_row("c-1", stage="grasp", code="no_clamp", root_cause="calibration_stale", rule="R3"),
		_row("c-2", stage="grasp", code="no_clamp", root_cause="calibration_stale", rule="R3"),
		_row("c-3", stage="grasp", code="no_clamp", root_cause="calibration_stale", rule="R3"),
		# 5x evidence_insufficient (loss 0.3 each -> impact 1.5)
		*[_row(f"ei-{i}", stage="inspect", code=None, root_cause="evidence_insufficient", rule="R5") for i in range(5)],
		# 2x perception_pose_error (loss 1.0 each -> impact 2.0)
		_row("p-1", stage="perception", code="SCAN_INCOMPLETE", root_cause="perception_pose_error", rule="R2"),
		_row("p-2", stage="perception", code="SCAN_INCOMPLETE", root_cause="perception_pose_error", rule="R2"),
	]
	ledger = build_ledger(rows, reviews={}, factors={})

	assert ledger["total_episodes"] == 12
	assert ledger["ok_episodes"] == 2
	assert ledger["failure_episodes"] == 10
	assert len(ledger["clusters"]) == 3

	# Clean ok/none rows are excluded from failure clusters.
	assert all(c["key"][2] != "none" for c in ledger["clusters"])

	# Sorted descending by impact: calibration_stale (3.0) > perception_pose_error (2.0) > evidence_insufficient (1.5)
	c0, c1, c2 = ledger["clusters"]
	assert c0["key"] == ["grasp", "no_clamp", "calibration_stale"]
	assert c0["n"] == 3
	assert c0["impact"] == pytest.approx(3.0)
	assert c0["share"] == pytest.approx(0.3)
	assert c0["review_agreement"] is None

	assert c1["key"] == ["perception", "SCAN_INCOMPLETE", "perception_pose_error"]
	assert c1["n"] == 2
	assert c1["impact"] == pytest.approx(2.0)
	assert c1["share"] == pytest.approx(0.2)

	assert c2["key"] == ["inspect", None, "evidence_insufficient"]
	assert c2["n"] == 5
	assert c2["impact"] == pytest.approx(1.5)
	assert c2["share"] == pytest.approx(0.5)

	assert sum(c["n"] for c in ledger["clusters"]) == 10
	assert sum(c["share"] for c in ledger["clusters"]) == pytest.approx(1.0)


def test_build_ledger_all_ok_returns_empty_clusters():
	rows = [_row(f"ok-{i}", stage="ok", code=None, root_cause="none", rule="none") for i in range(4)]
	ledger = build_ledger(rows, reviews={}, factors={})
	assert ledger["total_episodes"] == 4
	assert ledger["ok_episodes"] == 4
	assert ledger["failure_episodes"] == 0
	assert ledger["clusters"] == []


def test_build_ledger_review_override_and_agreement():
	rows = [
		_row("ep-1", stage="grasp", code="no_clamp", root_cause="grasp_or_path_failure", rule="R6"),
		_row("ep-2", stage="grasp", code="no_clamp", root_cause="calibration_stale", rule="R3"),
		_row("ep-3", stage="grasp", code="no_clamp", root_cause="calibration_stale", rule="R3"),
	]
	reviews = {
		# Human overrides ep-1 from grasp_or_path_failure -> calibration_stale (disagrees with rule)
		"ep-1": {"state": "disputed", "root_cause": "calibration_stale"},
		# Human confirms ep-2 as calibration_stale (agrees with rule)
		"ep-2": {"state": "confirmed", "root_cause": "calibration_stale"},
		# ep-3 is unreviewed
	}
	ledger = build_ledger(rows, reviews=reviews, factors={})

	assert len(ledger["clusters"]) == 1
	cluster = ledger["clusters"][0]
	assert cluster["key"] == ["grasp", "no_clamp", "calibration_stale"]
	assert cluster["n"] == 3
	# 1 agreed (ep-2), 1 disagreed (ep-1), 1 unreviewed (ep-3) -> 0.5
	assert cluster["review_agreement"] == pytest.approx(0.5)
	assert ledger["review_agreement"] == pytest.approx(0.5)


def test_build_ledger_factors_skew_and_examples_cap():
	rows = [_row(f"ep-{i:02d}", stage="grasp", code="no_clamp", root_cause="calibration_stale") for i in range(15)]
	factors = {
		f"ep-{i:02d}": {
			"camera_condition": "drift" if i < 12 else "calibrated",
			"brick_size": "2x4" if i % 3 == 0 else "2x2",
			"clutter": 0 if i < 10 else 2,
		}
		for i in range(15)
	}
	ledger = build_ledger(rows, reviews={}, factors=factors)
	assert len(ledger["clusters"]) == 1
	c = ledger["clusters"][0]
	assert c["n"] == 15
	assert len(c["examples"]) == 10
	assert c["examples"] == [f"ep-{i:02d}" for i in range(10)]
	assert c["factors"]["camera_condition"] == {"drift": 12, "calibrated": 3}
	assert c["factors"]["brick_size"] == {"2x2": 10, "2x4": 5}
	assert c["factors"]["clutter"] == {"0": 10, "2": 5}


def test_render_ledger_md_contains_viewer_links_and_details():
	rows = [
		_row("b/mujoco/g/L1-N", stage="grasp", code="no_clamp", root_cause="calibration_stale"),
		_row("b/mujoco/g/L2-D0", stage="perception", code="SCAN_INCOMPLETE", root_cause="perception_pose_error"),
	]
	reviews = {"b/mujoco/g/L1-N": {"state": "confirmed", "root_cause": "calibration_stale"}}
	factors = {
		"b/mujoco/g/L1-N": {"camera_condition": "drift", "bowl_colour": "green", "brick_size": "2x2", "clutter": 0},
		"b/mujoco/g/L2-D0": {"camera_condition": "recal", "bowl_colour": "pink", "brick_size": "2x4", "clutter": 2},
	}
	ledger = build_ledger(rows, reviews=reviews, factors=factors)
	md = render_ledger_md(ledger, batch="b", round_name="round-01")

	assert "http://127.0.0.1:8767/#b/mujoco/g/L1-N" in md
	assert "http://127.0.0.1:8767/#b/mujoco/g/L2-D0" in md
	assert "calibration_stale" in md
	assert "perception_pose_error" in md
	assert "camera_condition" in md
	assert "drift" in md
	assert "100.0%" in md


def test_extract_episode_factors_from_plan_task_and_scene(tmp_path):
	group_dir = tmp_path / "runs" / "b1" / "mujoco" / "realistic-drift-A-v7bot"
	group_dir.mkdir(parents=True)
	plan = {"batch": "b1", "engine": "mujoco", "group": "realistic-drift-A-v7bot", "camera_condition": "drift"}
	(group_dir / "plan.json").write_text(json.dumps(plan), encoding="utf-8")

	ep_dir = group_dir / "L700001-D0"
	_write_synth_episode(
		ep_dir,
		case=[700001, True, 0, {"cameras": "drift", "scene": "v3", "split": "train"}],
		result={"verdict": "FAILED", "label": "unknown"},
		events=[{"kind": "segment", "segment_kind": "grasp", "frame": 100}, {"kind": "failure", "code": "no_clamp"}],
		experiment={"batch": "b1", "engine": "mujoco", "group": "realistic-drift-A-v7bot", "case_id": "L700001-D0"},
		scene_layout={
			"brick_size": [0.0636, 0.0318, 0.0192],
			"bowl_names": {"green": "cyan", "pink": "magenta"},
			"clutter": [[0.1, 0.1, 0.0, 1.0, 0.0, 0.0], [0.2, 0.1, 0.0, 0.0, 1.0, 0.0]],
		},
	)
	fac = extract_episode_factors(ep_dir, plan=plan)
	assert fac["camera_condition"] == "drift"
	assert fac["bowl_colour"] == "magenta"
	assert fac["brick_size"] == "2x4"
	assert fac["clutter"] == 2
	assert fac["split"] == "train"


def test_failure_ledger_cli_synthetic_batch(tmp_path):
	cli_main = _load_cli()

	root = tmp_path / "data"
	batch = "b-synth"
	group = "realistic-drift-A-v7bot"
	group_dir = root / "runs" / batch / "mujoco" / group
	group_dir.mkdir(parents=True)
	(group_dir / "plan.json").write_text(
		json.dumps({"batch": batch, "engine": "mujoco", "group": group, "camera_condition": "drift"}),
		encoding="utf-8",
	)

	# 1 clean episode, 2 drift grasp failures (1 overridden in audit DB)
	_write_synth_episode(
		group_dir / "L1-N",
		case=[1, False, 0],
		result={"verdict": "SUCCESS", "label": "normal", "label_correct": True},
		events=[],
		experiment={"batch": batch, "engine": "mujoco", "group": group, "case_id": "L1-N", "attempt": 1},
		scene_layout={"brick_size": [0.0318, 0.0318, 0.0192], "clutter": []},
	)
	for cid in ("L2-D0", "L3-D1"):
		_write_synth_episode(
			group_dir / cid,
			case=[int(cid[1]), True, int(cid[-1]), {"cameras": "drift"}],
			result={"verdict": "FAILED", "label": "defective"},
			events=[
				{"kind": "segment", "segment_kind": "grasp", "frame": 100},
				{"kind": "failure", "code": "no_clamp", "frame": 200},
			],
			experiment={"batch": batch, "engine": "mujoco", "group": group, "case_id": cid, "attempt": 1},
			scene_layout={"brick_size": [0.0318, 0.0318, 0.0192], "clutter": []},
		)

	audit_db = root / "audit" / "reviews.sqlite"
	audit.append(
		audit_db,
		f"{batch}/mujoco/{group}/L2-D0",
		{"reviewer": "tester", "state": "disputed", "root_cause": "perception_pose_error"},
		expected_latest_id=None,
		code_version="test",
	)

	rc = cli_main(["--root", str(root), "--batch", batch, "--round", "01"])
	assert rc == 0

	out_dir = root / "reports" / batch / "round-01"
	assert (out_dir / "triage.jsonl").is_file()
	assert (out_dir / "ledger.json").is_file()
	assert (out_dir / "ledger.md").is_file()

	triage_lines = [
		json.loads(line) for line in (out_dir / "triage.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()
	]
	assert len(triage_lines) == 3

	ledger = json.loads((out_dir / "ledger.json").read_text(encoding="utf-8"))
	assert ledger["total_episodes"] == 3
	assert ledger["ok_episodes"] == 1
	assert ledger["failure_episodes"] == 2
	causes = {tuple(c["key"]) for c in ledger["clusters"]}
	assert ("grasp", "no_clamp", "calibration_stale") in causes
	assert ("grasp", "no_clamp", "perception_pose_error") in causes

	md = (out_dir / "ledger.md").read_text(encoding="utf-8")
	assert f"http://127.0.0.1:8767/#{batch}/mujoco/{group}/L2-D0" in md


P2_DEV_R2_ROOT = Path("D:/Armigo_data_v2")


@pytest.mark.skipif(
	not (P2_DEV_R2_ROOT / "runs" / "p2-dev-r2").is_dir(),
	reason="D:/Armigo_data_v2/runs/p2-dev-r2 dataset not available",
)
def test_failure_ledger_smoke_p2_dev_r2(tmp_path):
	cli_main = _load_cli()

	out_dir = tmp_path / "p2-dev-r2-round-01"
	rc = cli_main(
		[
			"--root",
			str(P2_DEV_R2_ROOT),
			"--batch",
			"p2-dev-r2",
			"--round",
			"01",
			"--out-dir",
			str(out_dir),
		]
	)
	assert rc == 0

	triage_rows = [
		json.loads(line) for line in (out_dir / "triage.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()
	]
	assert len(triage_rows) == 216

	ledger = json.loads((out_dir / "ledger.json").read_text(encoding="utf-8"))
	assert ledger["total_episodes"] == 216
	assert sum(c["n"] for c in ledger["clusters"]) == ledger["failure_episodes"]
	assert ledger["ok_episodes"] + ledger["failure_episodes"] == 216

	cluster_causes = {c["key"][2] for c in ledger["clusters"]}
	assert "calibration_stale" in cluster_causes
	assert "perception_pose_error" in cluster_causes
	top_keys = [tuple(c["key"]) for c in ledger["clusters"][:5]]
	assert any(k[2] == "calibration_stale" and k[1] == "no_clamp" for k in top_keys)

	md = (out_dir / "ledger.md").read_text(encoding="utf-8")
	assert "http://127.0.0.1:8767/#p2-dev-r2/" in md
