"""WP6 TDD tests: counterfactual composite keys (H6), strict compare_runs (H8),
failure_bank engine/cluster verification (F3), verify_fix completeness/isolation/R5 (F4),
and triage R1/R4 + ROOT_CAUSE_TO_MECHANISM_COMPAT + ledger impact_share (F1/F2).
"""

from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import pytest

from digital_twin_armigo.experiments import diag_mcp
from digital_twin_armigo.experiments.contract import ROOT_CAUSES
from digital_twin_armigo.experiments.counterfactual import VARIANTS
from digital_twin_armigo.experiments.ledger import build_ledger, render_ledger_md
from digital_twin_armigo.experiments.precursors import MECHANISMS
from digital_twin_armigo.experiments.triage import TriageRow, triage_episode

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts" / "experiments"


def _load_script(name: str):
	path = SCRIPTS_DIR / f"{name}.py"
	spec = importlib.util.spec_from_file_location(f"wp6_{name}", path)
	assert spec is not None and spec.loader is not None
	mod = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(mod)
	return mod


def _write_ep(
	ep_dir: Path,
	*,
	events: list[dict] | None = None,
	result: dict | None = None,
	provenance: dict | None = None,
	task: dict | None = None,
	actions_state: float | None = None,
	actions_action: float | None = None,
	branch_name: str = "branch-0",
	n_frames: int = 6,
) -> Path:
	ep_dir.mkdir(parents=True, exist_ok=True)
	if events is not None:
		(ep_dir / "events.jsonl").write_text(
			"".join(json.dumps(e) + "\n" for e in events),
			encoding="utf-8",
		)
	if result is not None or provenance is not None:
		payload: dict = {"schema": "armigo-agent-episode/v1"}
		if provenance is not None:
			payload["provenance"] = provenance
		if result is not None:
			payload["result"] = result
		(ep_dir / "episode.json").write_text(json.dumps(payload), encoding="utf-8")
	if task is not None:
		(ep_dir / "task.json").write_text(json.dumps(task), encoding="utf-8")
	if actions_state is not None or actions_action is not None:
		bdir = ep_dir / branch_name
		bdir.mkdir(parents=True, exist_ok=True)
		s_val = 0.25 if actions_state is None else actions_state
		a_val = 0.35 if actions_action is None else actions_action
		state64 = np.full((n_frames, 6), s_val, dtype=np.float64)
		action64 = np.full((n_frames, 6), a_val, dtype=np.float64)
		np.savez_compressed(
			bdir / "actions.npz",
			state=state64.astype(np.float32),
			action=action64.astype(np.float32),
			state_f64=state64,
			action_f64=action64,
			segment=np.arange(n_frames, dtype=np.int32),
			phase=np.full(n_frames, "approach"),
		)
	return ep_dir


# ---------------------------------------------------------------------------
# 1. H6: Counterfactual composite keys & 4th-element options preservation
# ---------------------------------------------------------------------------


def test_h6_case_from_row_preserves_4th_element_options(tmp_path):
	cf = _load_script("counterfactual")
	opts = {"scene": "v3", "split": "train", "cameras": "drift"}
	ep = tmp_path / "ep1"
	ep.mkdir()
	(ep / "task.json").write_text(json.dumps({"case": [700001, True, 2, opts]}), encoding="utf-8")
	got = cf._case_from_row({"case_id": "L700001-D2", "path": str(ep)})
	assert got == [700001, True, 2, opts]


def test_h6_cmd_plan_preserves_options_and_deduplicates_full_case(tmp_path):
	cf = _load_script("counterfactual")
	ep_fixed = tmp_path / "ep_fixed"
	ep_drift = tmp_path / "ep_drift"
	ep_fixed.mkdir()
	ep_drift.mkdir()
	opts_fixed = {"scene": "v3", "split": "train", "cameras": "fixed"}
	opts_drift = {"scene": "v3", "split": "train", "cameras": "drift"}
	(ep_fixed / "task.json").write_text(json.dumps({"case": [700001, True, 2, opts_fixed]}), encoding="utf-8")
	(ep_drift / "task.json").write_text(json.dumps({"case": [700001, True, 2, opts_drift]}), encoding="utf-8")

	scores = tmp_path / "scores.jsonl"
	rows = [
		{
			"case_id": "L700001-D2",
			"group": "realistic-fixed-A-v7bot",
			"camera_condition": "fixed",
			"strategy": "A",
			"physical_success": False,
			"path": str(ep_fixed),
		},
		{
			"case_id": "L700001-D2",
			"group": "realistic-drift-A-v7bot",
			"camera_condition": "drift",
			"strategy": "A",
			"physical_success": False,
			"path": str(ep_drift),
		},
		# Exact duplicate of first row should be deduplicated
		{
			"case_id": "L700001-D2",
			"group": "realistic-fixed-A-v7bot",
			"camera_condition": "fixed",
			"strategy": "A",
			"physical_success": False,
			"path": str(ep_fixed),
		},
	]
	scores.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
	out = tmp_path / "suite.json"
	manifest = tmp_path / "manifest.json"
	assert cf.main(["plan", "--scores", str(scores), "--out", str(out), "--manifest", str(manifest)]) == 0
	cases = json.loads(out.read_text(encoding="utf-8"))
	assert len(cases) == 2
	assert [700001, True, 2, opts_fixed] in cases
	assert [700001, True, 2, opts_drift] in cases


def test_h6_cmd_summarize_composite_keys_no_collision_and_order_invariant(tmp_path):
	cf = _load_script("counterfactual")

	specs = [
		{
			"case_id": "L700001-D2",
			"engine": "mujoco",
			"camera_condition": "fixed",
			"strategy": "A",
			"motion": "open",
			"evidence": "hold",
			"segmenter": "lab",
			"group": "realistic-fixed-A-v7bot",
			"path": str(tmp_path / "fixed_A" / "L700001-D2"),
			"traj_id": "b/mujoco/realistic-fixed-A-v7bot/L700001-D2",
			"physical_success": False,
			"root_cause": "grasp_or_path_failure",
			"mechanism": "brick_pose_error",
			"expect_verdict": "confirmed",
			"expect_minimal": [["brick"]],
			"ok_parts": lambda parts: "brick" in parts,
		},
		{
			"case_id": "L700001-D2",
			"engine": "mujoco",
			"camera_condition": "drift",
			"strategy": "A",
			"motion": "open",
			"evidence": "hold",
			"segmenter": "lab",
			"group": "realistic-drift-A-v7bot",
			"path": str(tmp_path / "drift_A" / "L700001-D2"),
			"traj_id": "b/mujoco/realistic-drift-A-v7bot/L700001-D2",
			"physical_success": False,
			"root_cause": "calibration_stale",
			"mechanism": "calibration_drift",
			"expect_verdict": "confirmed",
			"expect_minimal": [["camera"]],
			"ok_parts": lambda parts: "camera" in parts,
		},
		{
			"case_id": "L700001-D2",
			"engine": "mujoco",
			"camera_condition": "drift",
			"strategy": "B",
			"motion": "w2s",
			"evidence": "dense-10",
			"segmenter": "unet",
			"group": "realistic-drift-B-v7bot-mw2s-edense-10-sunet",
			"path": str(tmp_path / "drift_B" / "L700001-D2"),
			"traj_id": "b/mujoco/realistic-drift-B-v7bot-mw2s-edense-10-sunet/L700001-D2",
			"physical_success": False,
			"root_cause": "placement_failure",
			"mechanism": None,
			"expect_verdict": "not_perception",
			"expect_minimal": [],
			"ok_parts": lambda _parts: False,
		},
	]

	def _run_with_order(subdir: Path, perm: list[int]) -> str:
		subdir.mkdir(parents=True, exist_ok=True)
		base_file = subdir / "base.jsonl"
		base_rows = [
			{
				k: specs[idx][k]
				for k in (
					"case_id",
					"engine",
					"camera_condition",
					"strategy",
					"motion",
					"evidence",
					"segmenter",
					"group",
					"path",
					"traj_id",
					"physical_success",
					"root_cause",
					"mechanism",
				)
			}
			for idx in perm
		]
		base_file.write_text("".join(json.dumps(r) + "\n" for r in base_rows), encoding="utf-8")

		vfiles = []
		for vi, parts in enumerate(VARIANTS):
			vf = subdir / f"v{vi}.jsonl"
			suffix = f"-oracle-{'+'.join(parts)}" if parts else ""
			vrows = []
			for idx in reversed(perm):
				s = specs[idx]
				vrows.append(
					{
						"case_id": s["case_id"],
						"engine": s["engine"],
						"camera_condition": s["camera_condition"],
						"strategy": s["strategy"],
						"motion": s["motion"],
						"evidence": s["evidence"],
						"segmenter": s["segmenter"],
						"group": f"realistic-{s['camera_condition']}-{s['strategy']}-v7bot{suffix}",
						"physical_success": bool(s["ok_parts"](parts)),
					}
				)
			vf.write_text("".join(json.dumps(r) + "\n" for r in vrows), encoding="utf-8")
			vfiles.append(vf)

		out_file = subdir / "cf.jsonl"
		rc = cf.main(
			["summarize", "--base", str(base_file), "--variants", *[str(p) for p in vfiles], "--out", str(out_file)]
		)
		assert rc == 0
		return out_file.read_text(encoding="utf-8")

	out_a = _run_with_order(tmp_path / "run_a", [0, 1, 2])
	rng = random.Random(42)
	perm = [0, 1, 2]
	rng.shuffle(perm)
	if perm == [0, 1, 2]:
		perm = [2, 0, 1]
	out_b = _run_with_order(tmp_path / "run_b", perm)

	assert out_a == out_b
	parsed = [json.loads(line) for line in out_a.splitlines() if line.strip()]
	assert len(parsed) == 3

	by_cam_strat = {(r["camera_condition"], r["strategy"]): r for r in parsed}
	assert by_cam_strat[("fixed", "A")]["verdict"] == "confirmed"
	assert by_cam_strat[("fixed", "A")]["minimal"] == [["brick"]]
	assert by_cam_strat[("fixed", "A")]["traj_id"] == specs[0]["traj_id"]
	assert by_cam_strat[("drift", "A")]["verdict"] == "confirmed"
	assert by_cam_strat[("drift", "A")]["minimal"] == [["camera"]]
	assert by_cam_strat[("drift", "B")]["verdict"] == "not_perception"
	assert by_cam_strat[("drift", "B")]["motion"] == "w2s"
	assert by_cam_strat[("drift", "B")]["evidence"] == "dense-10"
	assert by_cam_strat[("drift", "B")]["segmenter"] == "unet"
	assert by_cam_strat[("drift", "B")]["group"] == specs[2]["group"]
	assert by_cam_strat[("drift", "B")]["path"] == specs[2]["path"]


def test_h6_cmd_summarize_raises_on_duplicate_composite_keys(tmp_path):
	cf = _load_script("counterfactual")
	dup_base = tmp_path / "dup_base.jsonl"
	row = {
		"case_id": "L700001-D2",
		"engine": "mujoco",
		"camera_condition": "drift",
		"strategy": "A",
		"physical_success": False,
	}
	dup_base.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
	vfiles = []
	for i, parts in enumerate(VARIANTS):
		vf = tmp_path / f"v{i}.jsonl"
		suffix = f"-oracle-{'+'.join(parts)}" if parts else ""
		vf.write_text(
			json.dumps({**row, "group": f"realistic-drift-A-v7bot{suffix}", "physical_success": False}) + "\n",
			encoding="utf-8",
		)
		vfiles.append(vf)
	with pytest.raises(ValueError, match="duplicate"):
		cf.main(
			[
				"summarize",
				"--base",
				str(dup_base),
				"--variants",
				*[str(p) for p in vfiles],
				"--out",
				str(tmp_path / "out.jsonl"),
			]
		)


def test_h6_round_metrics_no_cross_condition_counterfactual_leakage(tmp_path):
	rm = _load_script("round_metrics")
	score_rows = [
		{
			"case_id": "L700001-D2",
			"engine": "mujoco",
			"camera_condition": "drift",
			"strategy": "B",
			"motion": "open",
			"evidence": "hold",
			"group": "realistic-drift-B-v7bot",
			"path": str(tmp_path / "drift" / "L700001-D2"),
			"root_cause": "calibration_stale",
			"mechanism": "calibration_drift",
		},
		{
			"case_id": "L700001-D2",
			"engine": "mujoco",
			"camera_condition": "fixed",
			"strategy": "A",
			"motion": "open",
			"evidence": "hold",
			"group": "realistic-fixed-A-v7bot",
			"path": str(tmp_path / "fixed" / "L700001-D2"),
			"root_cause": "grasp_or_path_failure",
			"mechanism": "brick_pose_error",
		},
	]
	cf_path = tmp_path / "counterfactual.jsonl"
	# Only drift/B has a confirmed counterfactual result — MUST NOT leak onto fixed/A!
	cf_path.write_text(
		json.dumps(
			{
				"case_id": "L700001-D2",
				"engine": "mujoco",
				"camera_condition": "drift",
				"strategy": "B",
				"motion": "open",
				"evidence": "hold",
				"verdict": "confirmed",
			}
		)
		+ "\n",
		encoding="utf-8",
	)
	cov = rm._attribution_coverage(score_rows, cf_path)
	assert cov["n_failed"] == 2
	assert cov["counterfactual_confirmed_n"] == 1
	assert cov["counterfactual_confirmed_rate"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 2. H8: diag_mcp.compare_runs completeness, timing whitelist, action diff
# ---------------------------------------------------------------------------


def test_h8_compare_runs_completeness_checks(tmp_path):
	# Non-existent directories must NEVER report identical_discrete=True or verdict_equal=True
	r_missing = diag_mcp.compare_runs(tmp_path / "does_not_exist_a", tmp_path / "does_not_exist_b")
	assert r_missing["complete"] is False
	assert r_missing["identical"] is False
	assert r_missing["identical_discrete"] is False
	assert r_missing["verdict_equal"] is False
	assert len(r_missing["errors"]) >= 2

	# Missing verdict in episode.json must yield complete=False, verdict_equal=False, identical_discrete=False
	ev = [{"seq": 0, "kind": "node", "node": "prepare", "branch": 0}]
	ep_a = _write_ep(tmp_path / "no_verdict_a", events=ev, result={"label": "normal"})
	ep_b = _write_ep(tmp_path / "no_verdict_b", events=ev, result={"label": "normal"})
	r_nv = diag_mcp.compare_runs(ep_a, ep_b)
	assert r_nv["complete"] is False
	assert r_nv["identical"] is False
	assert r_nv["identical_discrete"] is False
	assert r_nv["verdict_equal"] is False
	assert any("verdict" in err.lower() for err in r_nv["errors"])


def test_h8_compare_runs_ignores_nested_timing_and_preserves_discrete_sensitivity(tmp_path):
	ev_a = [
		{
			"seq": 0,
			"kind": "view",
			"node": "scan_wrist",
			"branch": 0,
			"wall_time_s": 1.0,
			"ts": "2026-11-25T10:00:00Z",
			"observation": {"id": "0:1:wrist_cam", "score": -0.5, "alarm": False},
			"detector": {"name": "v7bot", "elapsed_ms": 12.3, "inference_ms": 10.1, "score_ms": 11.0},
		}
	]
	ev_b = [
		{
			"seq": 5,
			"kind": "view",
			"node": "scan_wrist",
			"branch": 0,
			"wall_time_s": 4.2,
			"ts": "2026-11-25T10:05:00Z",
			"observation": {"id": "0:1:wrist_cam", "score": -0.5, "alarm": False},
			"detector": {"name": "v7bot", "elapsed_ms": 28.9, "inference_ms": 25.0, "score_ms": 26.4},
		}
	]
	ep_a = _write_ep(tmp_path / "t_a", events=ev_a, result={"verdict": "SUCCESS", "label": "normal"})
	ep_b = _write_ep(tmp_path / "t_b", events=ev_b, result={"verdict": "SUCCESS", "label": "normal"})
	r = diag_mcp.compare_runs(ep_a, ep_b)
	assert r["complete"] is True
	assert r["identical"] is True
	assert r["identical_discrete"] is True
	assert r["verdict_equal"] is True
	assert r["first_divergence"] is None
	assert r["errors"] == []
	timing_fields = {td["field"] for td in r["timing_deltas"]}
	assert "detector.elapsed_ms" in timing_fields
	assert "wall_time_s" in timing_fields

	# Discrete change (alarm False -> True) MUST still fail identical_discrete
	ev_c = [
		{
			**ev_b[0],
			"observation": {"id": "0:1:wrist_cam", "score": -0.5, "alarm": True},
		}
	]
	ep_c = _write_ep(tmp_path / "t_c", events=ev_c, result={"verdict": "SUCCESS", "label": "normal"})
	r_diff = diag_mcp.compare_runs(ep_a, ep_c)
	assert r_diff["complete"] is True
	assert r_diff["identical_discrete"] is False
	assert r_diff["identical"] is False
	assert r_diff["first_divergence"]["index"] == 0


def test_h8_compare_runs_compares_actions_npz(tmp_path):
	ev = [{"seq": 0, "kind": "node", "node": "place", "branch": 0}]
	ep_a = _write_ep(
		tmp_path / "act_a",
		events=ev,
		result={"verdict": "SUCCESS", "label": "normal"},
		actions_state=0.1,
		actions_action=0.2,
	)
	ep_b = _write_ep(
		tmp_path / "act_b",
		events=ev,
		result={"verdict": "SUCCESS", "label": "normal"},
		actions_state=0.1,
		actions_action=0.2,
	)
	r_eq = diag_mcp.compare_runs(ep_a, ep_b)
	assert r_eq["complete"] is True
	assert r_eq["actions_equal"] is True
	assert r_eq["action_diff"] is None
	assert r_eq["identical"] is True

	ep_c = _write_ep(
		tmp_path / "act_c",
		events=ev,
		result={"verdict": "SUCCESS", "label": "normal"},
		actions_state=0.1,
		actions_action=0.25,
	)
	r_neq = diag_mcp.compare_runs(ep_a, ep_c)
	assert r_neq["complete"] is True
	assert r_neq["identical_discrete"] is True
	assert r_neq["verdict_equal"] is True
	assert r_neq["actions_equal"] is False
	assert r_neq["identical"] is False
	assert r_neq["action_diff"]["branch"] == "branch-0"
	assert r_neq["action_diff"]["first_frame"] == 0
	assert r_neq["action_diff"]["max_abs_diff"] == pytest.approx(0.05, abs=1e-5)


# ---------------------------------------------------------------------------
# 3. F3: failure_bank engine guard, dirty commit flag, target cluster & H8
# ---------------------------------------------------------------------------


def _setup_ledger_and_episode(
	tmp_path: Path,
	*,
	engine: str = "mujoco",
	code_commit: str = "abc1234",
	cluster_key: list,
	case: list | None = None,
) -> tuple[Path, Path, str]:
	root = tmp_path / "data"
	batch = "b-test"
	group = "realistic-drift-A-v7bot"
	case = case or [700001, False, 0, {"cameras": "drift"}]
	cid = f"L{case[0]}-D{case[2]}" if case[1] else f"L{case[0]}-N"
	tid = f"{batch}/{engine}/{group}/{cid}"
	ep_dir = root / "runs" / tid
	_write_ep(
		ep_dir,
		events=[
			{"kind": "segment", "segment_kind": "grasp", "frame": 100},
			{"kind": "failure", "code": "no_clamp", "frame": 200},
		],
		result={"verdict": "FAILED", "label": "unknown"},
		provenance={
			"engine": engine,
			"sensing": "realistic",
			"experiment": {
				"batch": batch,
				"engine": engine,
				"group": group,
				"case_id": cid,
				"camera_condition": "drift",
				"code_commit": code_commit,
			},
		},
		task={"case": case},
	)
	report_dir = root / "reports" / batch / "round-01"
	report_dir.mkdir(parents=True)
	ledger_path = report_dir / "ledger.json"
	ledger_path.write_text(
		json.dumps(
			{
				"batch": batch,
				"round": "round-01",
				"clusters": [
					{
						"key": cluster_key,
						"n": 1,
						"impact": 1.0,
						"share": 1.0,
						"factors": {},
						"examples": [tid],
					}
				],
			}
		),
		encoding="utf-8",
	)
	return root, ledger_path, tid


def test_f3_rejects_isaac_engine_without_invoking_repro(tmp_path, monkeypatch):
	fb = _load_script("failure_bank")
	root, ledger_path, _ = _setup_ledger_and_episode(
		tmp_path,
		engine="isaac",
		cluster_key=["grasp", "no_clamp", "calibration_stale"],
	)

	called = []

	def _fail_if_called(spec, out_dir):
		called.append((spec, out_dir))
		raise AssertionError("run_repro_subprocess MUST NOT be called for non-mujoco engines")

	monkeypatch.setattr(fb, "run_repro_subprocess", _fail_if_called)

	payload, _ = fb.build_failure_bank(
		ledger_path=ledger_path,
		cluster_spec="grasp:no_clamp:calibration_stale",
		round_label="01",
		out_path=tmp_path / "bank.json",
		episodes_root=root,
		scratch_dir=tmp_path / "scratch",
		update_manifest=False,
	)
	assert called == []
	assert payload["cases"] == []
	assert len(payload["repro_checked"]["dropped"]) == 1
	assert payload["repro_checked"]["dropped"][0]["reason"] == "unsupported_engine: isaac"


def test_f3_drops_when_target_cluster_not_reproduced(tmp_path, monkeypatch):
	fb = _load_script("failure_bank")
	root, ledger_path, _ = _setup_ledger_and_episode(
		tmp_path,
		engine="mujoco",
		cluster_key=["grasp", "no_clamp", "calibration_stale"],
	)

	def _repro_that_passes(spec: dict, out_dir: Path) -> None:
		_write_ep(
			out_dir,
			events=[{"seq": 0, "kind": "node", "node": "place", "branch": 0}],
			result={"verdict": "SUCCESS", "label": "normal", "label_correct": True, "destination": "green"},
			provenance={"engine": "mujoco", "experiment": {"camera_condition": "drift"}},
			task={"case": spec["case"]},
			actions_state=0.2,
			actions_action=0.3,
		)

	monkeypatch.setattr(fb, "run_repro_subprocess", _repro_that_passes)

	payload, _ = fb.build_failure_bank(
		ledger_path=ledger_path,
		cluster_spec="grasp:no_clamp:calibration_stale",
		round_label="01",
		out_path=tmp_path / "bank.json",
		episodes_root=root,
		scratch_dir=tmp_path / "scratch",
		update_manifest=False,
	)
	assert payload["cases"] == []
	assert len(payload["repro_checked"]["dropped"]) == 1
	assert payload["repro_checked"]["dropped"][0]["reason"] == "target_cluster_not_reproduced"


def test_f3_uses_compare_runs_timing_tolerance_and_flags_dirty_commit(tmp_path, monkeypatch):
	fb = _load_script("failure_bank")
	root, ledger_path, _ = _setup_ledger_and_episode(
		tmp_path,
		engine="mujoco",
		code_commit="deadbeef-dirty",
		cluster_key=["grasp", "no_clamp", "calibration_stale"],
	)

	def _repro_matching_cluster_with_timing_jitter(spec: dict, out_dir: Path) -> None:
		idx = int(spec["run_index"])
		_write_ep(
			out_dir,
			events=[
				{
					"seq": 0,
					"kind": "segment",
					"segment_kind": "grasp",
					"frame": 100,
					"detector": {"elapsed_ms": 10.0 * idx},
				},
				{"seq": 1, "kind": "failure", "code": "no_clamp", "frame": 200, "wall_time_s": float(idx)},
			],
			result={"verdict": "FAILED", "label": "unknown", "destination": "table"},
			provenance={"engine": "mujoco", "experiment": {"camera_condition": "drift"}},
			task={"case": spec["case"]},
			actions_state=0.2,
			actions_action=0.3,
		)

	monkeypatch.setattr(fb, "run_repro_subprocess", _repro_matching_cluster_with_timing_jitter)

	payload, _ = fb.build_failure_bank(
		ledger_path=ledger_path,
		cluster_spec="grasp:no_clamp:calibration_stale",
		round_label="01",
		out_path=tmp_path / "bank.json",
		episodes_root=root,
		scratch_dir=tmp_path / "scratch",
		update_manifest=False,
	)
	assert len(payload["cases"]) == 1
	kept = payload["repro_checked"]["kept"][0]
	assert kept["engine"] == "mujoco"
	assert kept["source_commit"] == "deadbeef-dirty"
	assert kept["source_commit_dirty"] is True
	assert any("dirty" in w.lower() for w in payload["repro_checked"].get("warnings", []))


# ---------------------------------------------------------------------------
# 4. F4: verify_fix completeness, group consistency, seed isolation, R5 metric
# ---------------------------------------------------------------------------


def _vf_row(
	case_id: str,
	*,
	success: bool,
	expected_label: str = "normal",
	group: str = "realistic-drift-A-v7bot",
	engine: str = "mujoco",
	root_cause: str | None = None,
	label_correct: bool | None = None,
) -> dict:
	lid = case_id.split("-")[0]
	rc = root_cause if root_cause is not None else ("none" if success else "calibration_stale")
	return {
		"case_id": case_id,
		"layout_id": lid,
		"engine": engine,
		"group": group,
		"expected_label": expected_label,
		"physical_success": bool(success),
		"label_correct": bool(success) if label_correct is None else label_correct,
		"defect_miss": expected_label == "defective" and not success,
		"false_alarm": expected_label == "normal" and not success,
		"is_unknown": False,
		"wrong_bowl": False,
		"infra_failed": False,
		"root_cause": rc,
	}


def test_f4_incomplete_pairing_and_suite_coverage_never_merge():
	vf = _load_script("verify_fix")

	bank_base = [_vf_row(f"L{100 + i}-D0", success=False, expected_label="defective") for i in range(5)]
	bank_new = [_vf_row(f"L{100 + i}-D0", success=True, expected_label="defective") for i in range(5)]

	# Regression has 100 base rows but only 1 new row (99 unmatched!) -> MUST FAIL Gate 2!
	reg_base = [_vf_row(f"L{200 + i}-N", success=True) for i in range(100)]
	reg_new_partial = [_vf_row("L200-N", success=True)]

	fresh_base = [_vf_row(f"L{500 + i}-N", success=True) for i in range(20)]
	fresh_new = [_vf_row(f"L{500 + i}-N", success=True) for i in range(20)]

	res_reg = vf.evaluate_three_gates(
		bank_base,
		bank_new,
		reg_base,
		reg_new_partial,
		fresh_base,
		fresh_new,
		{"bootstrap_b": 200},
	)
	assert res_reg["gates"]["regression"]["passed"] is False
	assert res_reg["gates"]["regression"]["fail_reason"] == "incomplete_pairing"
	assert res_reg["verdict"] == "reject"

	# Fresh has 200 base rows but only 1 new row (199 unmatched!) -> MUST FAIL Gate 3 and NOT merge!
	reg_full = [_vf_row(f"L{200 + i}-N", success=True) for i in range(10)]
	fresh_base_200 = [_vf_row(f"L{500 + i}-N", success=True) for i in range(200)]
	fresh_new_1 = [_vf_row("L500-N", success=True)]
	res_fresh = vf.evaluate_three_gates(
		bank_base,
		bank_new,
		reg_full,
		reg_full,
		fresh_base_200,
		fresh_new_1,
		{"bootstrap_b": 200},
	)
	assert res_fresh["gates"]["fresh"]["passed"] is False
	assert res_fresh["gates"]["fresh"]["fail_reason"] == "incomplete_pairing"
	assert res_fresh["verdict"] != "merge"

	# Expected suite IDs exceed matched rows even if base & new have identical subset -> incomplete_suite_coverage!
	expected_reg_ids = {f"L{200 + i}-N" for i in range(15)}
	res_cov = vf.evaluate_three_gates(
		bank_base,
		bank_new,
		reg_full,
		reg_full,
		fresh_base,
		fresh_new,
		{"bootstrap_b": 200},
		expected_reg_ids=expected_reg_ids,
	)
	assert res_cov["gates"]["regression"]["passed"] is False
	assert res_cov["gates"]["regression"]["fail_reason"] == "incomplete_suite_coverage"
	assert res_cov["verdict"] == "reject"


def test_f4_group_consistency_rejects_mixed_or_mismatched_groups():
	vf = _load_script("verify_fix")

	bank_base = [_vf_row(f"L{100 + i}-D0", success=False, expected_label="defective") for i in range(4)]
	bank_new = [_vf_row(f"L{100 + i}-D0", success=True, expected_label="defective") for i in range(4)]
	reg_mixed = [
		_vf_row("L201-N", success=True, group="realistic-fixed-A-v7bot"),
		_vf_row("L202-N", success=True, group="realistic-drift-A-v7bot"),
	]
	fresh = [_vf_row(f"L{300 + i}-N", success=True) for i in range(10)]

	res_mixed = vf.evaluate_three_gates(
		bank_base,
		bank_new,
		reg_mixed,
		reg_mixed,
		fresh,
		fresh,
		{"bootstrap_b": 200},
	)
	assert res_mixed["gates"]["regression"]["passed"] is False
	assert res_mixed["verdict"] == "reject"

	# Mismatched groups between fresh_base and fresh_new
	reg_clean = [_vf_row("L201-N", success=True)]
	fresh_fixed = [_vf_row(f"L{300 + i}-N", success=True, group="realistic-fixed-A-v7bot") for i in range(10)]
	fresh_drift = [_vf_row(f"L{300 + i}-N", success=True, group="realistic-drift-A-v7bot") for i in range(10)]
	res_mismatch = vf.evaluate_three_gates(
		bank_base,
		bank_new,
		reg_clean,
		reg_clean,
		fresh_fixed,
		fresh_drift,
		{"bootstrap_b": 200},
	)
	assert res_mismatch["gates"]["fresh"]["passed"] is False
	assert res_mismatch["verdict"] != "merge"


def test_f4_seed_isolation_fails_gate3_on_bank_or_reg_overlap():
	vf = _load_script("verify_fix")

	bank_base = [_vf_row("L101-D0", success=False, expected_label="defective")]
	bank_new = [_vf_row("L101-D0", success=True, expected_label="defective")]
	reg_rows = [_vf_row("L201-N", success=True)]
	# Fresh illegally includes L101 (from bank) and L201 (from regression)!
	fresh_leaky = [
		_vf_row("L101-N", success=True),
		_vf_row("L201-D0", success=True, expected_label="defective"),
		_vf_row("L301-N", success=True),
	]
	res = vf.evaluate_three_gates(
		bank_base,
		bank_new,
		reg_rows,
		reg_rows,
		fresh_leaky,
		fresh_leaky,
		{"bootstrap_b": 200},
	)
	g3 = res["gates"]["fresh"]
	assert g3["passed"] is False
	assert set(g3["seed_overlap_with_bank_or_reg"]) == {"L101", "L201"}
	assert res["verdict"] != "merge"


def test_f4_gate1_r5_cluster_specific_metric():
	vf = _load_script("verify_fix")

	# All 10 baseline R5 rows ALREADY have physical_success=True!
	r5_base = [
		_vf_row(f"L{100 + i}-N", success=True, label_correct=True, root_cause="evidence_insufficient")
		for i in range(10)
	]
	# Candidate resolves R5 on 8/10, leaves 1 still R5, and breaks physics on 1 -> 8/10 = 0.80 >= 0.70 -> PASS
	r5_new_pass = (
		[_vf_row(f"L{100 + i}-N", success=True, label_correct=True, root_cause="none") for i in range(8)]
		+ [_vf_row("L108-N", success=True, label_correct=True, root_cause="evidence_insufficient")]
		+ [_vf_row("L109-N", success=False, label_correct=False, root_cause="grasp_or_path_failure")]
	)

	reg = [_vf_row(f"L{200 + i}-N", success=True) for i in range(10)]
	fresh = [_vf_row(f"L{300 + i}-N", success=True) for i in range(10)]

	res_pass = vf.evaluate_three_gates(
		r5_base,
		r5_new_pass,
		reg,
		reg,
		fresh,
		fresh,
		{"bank_metric": "evidence_sufficient", "bank_min_fix_rate": 0.70, "bootstrap_b": 200},
	)
	assert res_pass["gates"]["bank"]["total_base_failures"] == 10
	assert res_pass["gates"]["bank"]["fixed"] == 8
	assert res_pass["gates"]["bank"]["fix_rate"] == pytest.approx(0.80)
	assert res_pass["gates"]["bank"]["passed"] is True
	assert res_pass["verdict"] == "merge"

	# Also works via target_cluster="R5"
	r5_new_fail = [
		_vf_row(
			f"L{100 + i}-N", success=True, label_correct=True, root_cause="none" if i < 6 else "evidence_insufficient"
		)
		for i in range(10)
	]
	res_fail = vf.evaluate_three_gates(
		r5_base,
		r5_new_fail,
		reg,
		reg,
		fresh,
		fresh,
		{"target_cluster": "R5", "bank_min_fix_rate": 0.70, "bootstrap_b": 200},
	)
	assert res_fail["gates"]["bank"]["total_base_failures"] == 10
	assert res_fail["gates"]["bank"]["fixed"] == 6
	assert res_fail["gates"]["bank"]["passed"] is False
	assert res_fail["verdict"] == "reject"


# ---------------------------------------------------------------------------
# 5. F1 R1/R4 + ROOT_CAUSE_TO_MECHANISM_COMPAT + F2 impact_share
# ---------------------------------------------------------------------------


def test_f1_r1_matches_brick_not_found_and_returns_valid_root_cause(tmp_path):
	ep = _write_ep(
		tmp_path / "r1_brick",
		events=[{"kind": "failure", "code": "BRICK_NOT_FOUND", "node": "prepare", "frame": 0}],
		result={"verdict": "FAILED", "failures": [{"code": "BRICK_NOT_FOUND", "frame": 0}]},
	)
	row = triage_episode(ep)
	assert row.rule == "R1"
	assert row.code == "BRICK_NOT_FOUND"
	assert row.stage == "prepare"
	assert row.root_cause in ROOT_CAUSES
	assert row.evidence["code"] == "BRICK_NOT_FOUND"

	# Also via result.execution_error without failure event
	ep2 = _write_ep(
		tmp_path / "r1_brick_res",
		events=[],
		result={"verdict": "FAILED", "execution_error": "BRICK_NOT_FOUND"},
	)
	row2 = triage_episode(ep2)
	assert row2.rule == "R1"
	assert row2.code == "BRICK_NOT_FOUND"
	assert row2.root_cause in ROOT_CAUSES


def test_f1_r4_supports_extra_distractor_bowls(tmp_path):
	# Detected via placement_check fill_pixels containing extra:0
	ep1 = _write_ep(
		tmp_path / "r4_extra_fill",
		events=[
			{"kind": "segment", "segment_kind": "carry_to_bowl", "params": {"bowl": "green"}, "frame": 800},
			{
				"kind": "placement_check",
				"fill_pixels": {"green": 80, "pink": 40, "extra:0": 1350},
				"frame": 900,
			},
			{"kind": "failure", "code": "PLACEMENT_UNVERIFIED", "skill": "place", "node": "recover", "frame": 900},
		],
		result={"verdict": "FAILED", "label": "normal", "label_correct": True},
	)
	row1 = triage_episode(ep1)
	assert row1.rule == "R4"
	assert row1.root_cause == "wrong_bowl_carry"
	assert row1.evidence["target_bowl"] == "green"
	assert row1.evidence["actual_bowl"] == "extra:0"

	# Detected via result.actual_destination == "extra:1"
	ep2 = _write_ep(
		tmp_path / "r4_extra_dest",
		events=[
			{"kind": "failure", "code": "PLACEMENT_UNVERIFIED", "skill": "place", "frame": 900},
		],
		result={"verdict": "FAILED", "label": "defective", "actual_destination": "extra:1"},
	)
	row2 = triage_episode(ep2)
	assert row2.rule == "R4"
	assert row2.root_cause == "wrong_bowl_carry"
	assert row2.evidence["target_bowl"] == "pink"
	assert row2.evidence["actual_bowl"] == "extra:1"


def test_f1_root_cause_to_mechanism_compat_complete_and_readonly():
	from digital_twin_armigo.experiments.triage import ROOT_CAUSE_TO_MECHANISM_COMPAT

	assert set(ROOT_CAUSE_TO_MECHANISM_COMPAT) == set(ROOT_CAUSES)
	for rc, mechs in ROOT_CAUSE_TO_MECHANISM_COMPAT.items():
		assert isinstance(mechs, tuple), rc
		for m in mechs:
			assert m in MECHANISMS, (rc, m)

	# v3's 5 added root causes are explicitly mapped
	assert "calibration_drift" in ROOT_CAUSE_TO_MECHANISM_COMPAT["calibration_stale"]
	assert "brick_yaw_error" in ROOT_CAUSE_TO_MECHANISM_COMPAT["perception_pose_error"]
	assert "bowl_pose_error" in ROOT_CAUSE_TO_MECHANISM_COMPAT["bowl_localization"]
	assert "bowl_pose_error" in ROOT_CAUSE_TO_MECHANISM_COMPAT["wrong_bowl_carry"]
	assert ROOT_CAUSE_TO_MECHANISM_COMPAT["tool_contract"] == ()

	# Must be read-only
	with pytest.raises(TypeError):
		ROOT_CAUSE_TO_MECHANISM_COMPAT["new_cause"] = ()  # type: ignore[index]


def test_f2_ledger_freq_share_and_impact_share_and_stop_criterion():
	from digital_twin_armigo.experiments.ledger import check_stop_criterion

	rows = [
		TriageRow("c-1", "grasp", "no_clamp", "calibration_stale", {}, "R3"),
		TriageRow("c-2", "grasp", "no_clamp", "calibration_stale", {}, "R3"),
		TriageRow("c-3", "grasp", "no_clamp", "calibration_stale", {}, "R3"),
		*[TriageRow(f"ei-{i}", "inspect", None, "evidence_insufficient", {}, "R5") for i in range(5)],
		TriageRow("p-1", "perception", "SCAN_INCOMPLETE", "perception_pose_error", {}, "R2"),
		TriageRow("p-2", "perception", "SCAN_INCOMPLETE", "perception_pose_error", {}, "R2"),
	]
	ledger = build_ledger(rows)
	c0, c1, c2 = ledger["clusters"]
	# c0: n=3/10=0.3, impact=3.0/6.5=0.461538
	assert c0["freq_share"] == pytest.approx(0.3)
	assert c0["impact_share"] == pytest.approx(3.0 / 6.5, rel=1e-5)
	assert c0["share"] == pytest.approx(0.3)
	# c1: n=2/10=0.2, impact=2.0/6.5=0.307692
	assert c1["freq_share"] == pytest.approx(0.2)
	assert c1["impact_share"] == pytest.approx(2.0 / 6.5, rel=1e-5)
	# c2: n=5/10=0.5, impact=1.5/6.5=0.230769
	assert c2["freq_share"] == pytest.approx(0.5)
	assert c2["impact_share"] == pytest.approx(1.5 / 6.5, rel=1e-5)
	assert sum(c["freq_share"] for c in ledger["clusters"]) == pytest.approx(1.0)
	assert sum(c["impact_share"] for c in ledger["clusters"]) == pytest.approx(1.0)

	# Stopping criterion MUST inspect impact_share, not freq_share!
	# Case where evidence_insufficient is 50% by count (5/10) but top cluster impact_share is 0.4615
	assert check_stop_criterion(ledger, max_top_impact_share=0.40)["stop"] is False
	assert check_stop_criterion(ledger, max_top_impact_share=0.50)["stop"] is True

	# Markdown renders impact share too
	md = render_ledger_md(ledger)
	assert "Impact Share" in md
