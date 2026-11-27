"""Tests for deterministic failure bank builder, runner integration, and three-gate fix verification (Tasks F3 & F4)."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from digital_twin_armigo.agent.twin_mujoco import load_suite
from digital_twin_armigo.experiments import runner
from digital_twin_armigo.experiments.contract import case_id
from digital_twin_armigo.experiments.stats import _exact_mcnemar, paired_comparison

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts" / "experiments"


def _load_script(name: str):
	path = SCRIPTS_DIR / f"{name}.py"
	spec = importlib.util.spec_from_file_location(name, path)
	assert spec is not None and spec.loader is not None
	mod = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(mod)
	return mod


def _write_synth_episode(
	ep_dir: Path,
	*,
	case: list,
	result: dict | None = None,
	events: list[dict] | None = None,
	experiment: dict | None = None,
	policy: dict | None = None,
	sensing: str = "realistic",
	strategy: str = "A",
) -> Path:
	ep_dir.mkdir(parents=True, exist_ok=True)
	pol = policy or {"order": "wrist_first", "max_bottom_views": 2, "max_wrist_views": 2, "max_rewinds": 1}
	(ep_dir / "task.json").write_text(
		json.dumps({"case": case, "policy": pol, "video": False, "use_llm": False}),
		encoding="utf-8",
	)
	(ep_dir / "episode.json").write_text(
		json.dumps(
			{
				"schema": "armigo-agent-episode/v1",
				"provenance": {
					"engine": "mujoco",
					"sensing": sensing,
					"policy": pol,
					"agent_options": {"strategy": strategy, "max_scored_views": 6, "boundary": "belief_cell"},
					"experiment": experiment or {},
				},
				"result": result or {"verdict": "FAILED", "label": "unknown"},
			}
		),
		encoding="utf-8",
	)
	evs = events or []
	(ep_dir / "events.jsonl").write_text(
		"\n".join(json.dumps(ev) for ev in evs) + ("\n" if evs else ""),
		encoding="utf-8",
	)
	(ep_dir / "run_status.json").write_text(json.dumps({"state": "done"}), encoding="utf-8")
	return ep_dir


def _write_fake_npz(ep_dir: Path, state_val: float, *, n_frames: int = 8) -> None:
	branch = ep_dir / "branch-0"
	branch.mkdir(parents=True, exist_ok=True)
	state64 = np.full((n_frames, 6), state_val, dtype=np.float64)
	action64 = np.full((n_frames, 6), state_val + 0.1, dtype=np.float64)
	np.savez_compressed(
		branch / "actions.npz",
		state=state64.astype(np.float32),
		action=action64.astype(np.float32),
		state_f64=state64,
		action_f64=action64,
		segment=np.arange(n_frames, dtype=np.int32),
		phase=np.full(n_frames, "approach"),
	)
	(ep_dir / "events.jsonl").write_text(
		json.dumps({"seq": 0, "kind": "segment", "segment_kind": "grasp", "frame": 100})
		+ "\n"
		+ json.dumps({"seq": 1, "kind": "failure", "code": "no_clamp", "frame": 200})
		+ "\n",
		encoding="utf-8",
	)
	(ep_dir / "episode.json").write_text(
		json.dumps(
			{
				"provenance": {"engine": "mujoco", "experiment": {"camera_condition": "drift"}},
				"result": {"verdict": "FAILED", "label": "normal", "destination": "table"},
			}
		),
		encoding="utf-8",
	)


def test_cluster_matching_and_triage_expansion(tmp_path):
	fb = _load_script("failure_bank")
	ledger = {
		"batch": "b1",
		"round": "round-01",
		"clusters": [
			{
				"key": ["grasp", "no_clamp", "calibration_stale"],
				"n": 15,
				"impact": 15.0,
				"share": 0.75,
				"factors": {},
				"examples": [f"b1/mujoco/g/L{700001 + i}-D0" for i in range(10)],
			},
			{
				"key": ["inspect", None, "evidence_insufficient"],
				"n": 5,
				"impact": 1.5,
				"share": 0.25,
				"factors": {},
				"examples": ["b1/mujoco/g/L700020-N"],
			},
		],
	}

	assert fb.match_cluster(ledger, "grasp:no_clamp:calibration_stale")["n"] == 15
	assert fb.match_cluster(ledger, "grasp/no_clamp/calibration_stale")["n"] == 15
	assert fb.match_cluster(ledger, '["grasp", "no_clamp", "calibration_stale"]')["n"] == 15
	assert fb.match_cluster(ledger, "grasp-no_clamp-calibration_stale")["n"] == 15
	assert fb.match_cluster(ledger, "inspect:none:evidence_insufficient")["n"] == 5
	assert fb.match_cluster(ledger, "inspect::evidence_insufficient")["n"] == 5
	assert fb.match_cluster(ledger, '["inspect", null, "evidence_insufficient"]')["n"] == 5

	report_dir = tmp_path / "reports" / "b1" / "round-01"
	report_dir.mkdir(parents=True)
	ledger_path = report_dir / "ledger.json"
	ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
	triage_rows = [
		{
			"traj_id": f"b1/mujoco/g/L{700001 + i}-D0",
			"stage": "grasp",
			"code": "no_clamp",
			"root_cause": "calibration_stale",
			"evidence": {},
			"rule": "R3",
		}
		for i in range(15)
	]
	(report_dir / "triage.jsonl").write_text(
		"\n".join(json.dumps(r) for r in triage_rows) + "\n",
		encoding="utf-8",
	)
	cluster = fb.match_cluster(ledger, "grasp:no_clamp:calibration_stale")
	traj_ids = fb.collect_cluster_traj_ids(ledger_path, cluster)
	assert len(traj_ids) == 15
	assert traj_ids[-1] == "b1/mujoco/g/L700015-D0"


def test_failure_bank_deterministic_two_subprocess_repro(tmp_path):
	pytest.importorskip("mujoco")
	fb = _load_script("failure_bank")

	dev_cases = load_suite(ROOT / "config" / "agent_bench" / "v2" / "dev_suite_700001.json")
	# Use defective case [700001, True, 0] with fixed scorer (-1.0 < 0.0) so MuJoCo genuinely reproduces R7 FN.
	first_case = list(dev_cases[1])
	assert first_case == [700001, True, 0]

	root = tmp_path / "data"
	batch = "b-repro"
	group = "realistic-fixed-A-fixed"
	cid = case_id(*first_case[:3])
	traj_id = f"{batch}/mujoco/{group}/{cid}"
	_write_synth_episode(
		root / "runs" / traj_id,
		case=first_case,
		result={"verdict": "FAILED", "label": "normal", "label_correct": False},
		events=[],
		experiment={"batch": batch, "engine": "mujoco", "group": group, "case_id": cid, "sensing": "realistic"},
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
						"key": ["inspect", "place_miss", "false_negative_visible"],
						"n": 1,
						"impact": 1.0,
						"share": 1.0,
						"factors": {},
						"examples": [traj_id],
					}
				],
			}
		),
		encoding="utf-8",
	)

	out_suite = tmp_path / "failure_bank" / "round-01-inspect-place_miss-false_negative_visible.json"
	manifest_path = tmp_path / "v2" / "manifest.json"
	manifest_path.parent.mkdir(parents=True)
	manifest_path.write_text((ROOT / "config" / "agent_bench" / "v2" / "manifest.json").read_text(encoding="utf-8"))
	scratch_dir = tmp_path / "scratch"

	rc = fb.main(
		[
			"--ledger",
			str(ledger_path),
			"--cluster",
			"inspect:place_miss:false_negative_visible",
			"--round",
			"01",
			"--max",
			"30",
			"--out",
			str(out_suite),
			"--manifest",
			str(manifest_path),
			"--episodes-root",
			str(root),
			"--scratch-dir",
			str(scratch_dir),
			"--scorer",
			"fixed",
		]
	)
	assert rc == 0
	assert out_suite.is_file()

	suite = json.loads(out_suite.read_text(encoding="utf-8"))
	assert suite["name"] == "round-01-inspect-place_miss-false_negative_visible"
	assert suite["sealed"] is False
	assert suite["source_round"] == "round-01"
	assert suite["cluster"] == ["inspect", "place_miss", "false_negative_visible"]
	assert isinstance(suite["commit"], str) and suite["commit"]
	assert suite["cases"] == [first_case]
	assert load_suite(out_suite) == [tuple(first_case)]

	repro = suite["repro_checked"]
	assert repro["commit"] == suite["commit"]
	assert repro["runs_per_case"] == 2
	assert len(repro["kept"]) == 1
	assert repro["kept"][0]["case_id"] == cid
	assert repro["kept"][0]["bitwise_equal"] is True
	assert repro["dropped"] == []

	# Both fresh subprocess outputs exist under scratch_dir and are bitwise equal.
	run_dirs = [p for p in sorted((scratch_dir / cid).glob("run-*")) if p.is_dir()]
	assert len(run_dirs) == 2
	with (
		np.load(run_dirs[0] / "branch-0" / "actions.npz", allow_pickle=False) as a,
		np.load(run_dirs[1] / "branch-0" / "actions.npz", allow_pickle=False) as b,
	):
		for key in ("state_f64", "action_f64", "state", "action", "segment", "phase"):
			assert np.array_equal(a[key], b[key]), key

	manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
	assert suite["name"] in manifest["suites"]
	assert manifest["suites"][suite["name"]]["cases"] == 1
	assert manifest["suites"][suite["name"]]["layouts"] == 1


def test_failure_bank_drops_nondeterministic_case(tmp_path, monkeypatch):
	fb = _load_script("failure_bank")

	root = tmp_path / "data"
	batch = "b-nondet"
	group = "realistic-drift-A-v7bot"
	c1 = [700001, False, 0, {"cameras": "drift"}]
	c2 = [700002, True, 1, {"cameras": "drift"}]
	t1 = f"{batch}/mujoco/{group}/{case_id(*c1[:3])}"
	t2 = f"{batch}/mujoco/{group}/{case_id(*c2[:3])}"
	_write_synth_episode(root / "runs" / t1, case=c1)
	_write_synth_episode(root / "runs" / t2, case=c2)

	report_dir = root / "reports" / batch / "round-02"
	report_dir.mkdir(parents=True)
	ledger_path = report_dir / "ledger.json"
	ledger_path.write_text(
		json.dumps(
			{
				"batch": batch,
				"round": "round-02",
				"clusters": [
					{
						"key": ["grasp", "no_clamp", "calibration_stale"],
						"n": 2,
						"impact": 2.0,
						"share": 1.0,
						"factors": {},
						"examples": [t1, t2],
					}
				],
			}
		),
		encoding="utf-8",
	)

	def fake_subprocess_runner(spec: dict, out_dir: Path) -> None:
		run_idx = int(spec["run_index"])
		cid = spec["case_id"]
		if cid == "L700001-N":
			_write_fake_npz(out_dir, 0.25)
		else:
			_write_fake_npz(out_dir, 0.25 if run_idx == 1 else 0.2501)

	monkeypatch.setattr(fb, "run_repro_subprocess", fake_subprocess_runner)

	out_suite = tmp_path / "failure_bank" / "round-02-grasp-no_clamp-calibration_stale.json"
	rc = fb.main(
		[
			"--ledger",
			str(ledger_path),
			"--cluster",
			"grasp:no_clamp:calibration_stale",
			"--round",
			"02",
			"--out",
			str(out_suite),
			"--episodes-root",
			str(root),
			"--scratch-dir",
			str(tmp_path / "scratch"),
			"--no-manifest",
		]
	)
	assert rc == 0

	suite = json.loads(out_suite.read_text(encoding="utf-8"))
	assert suite["cases"] == [c1]
	assert len(suite["repro_checked"]["kept"]) == 1
	assert suite["repro_checked"]["kept"][0]["case_id"] == "L700001-N"
	assert len(suite["repro_checked"]["dropped"]) == 1
	assert suite["repro_checked"]["dropped"][0]["case_id"] == "L700002-D1"
	assert "state_f64" in suite["repro_checked"]["dropped"][0]["reason"]


def test_failure_bank_checkpoint_resume_late_stage(tmp_path):
	mujoco = pytest.importorskip("mujoco")
	del mujoco
	fb = _load_script("failure_bank")

	from digital_twin_armigo.agent.student_rollout import ExpertTracer
	from digital_twin_armigo.agent.twin_graph import Policy, build_graph
	from digital_twin_armigo.agent.twin_mujoco import sample_case
	from digital_twin_armigo.agent.twin_record import EpisodeTrace, write_json
	from digital_twin_armigo.agent.twin_resume import live_checkpoint_ids
	from digital_twin_armigo.inspection_scorer import FixedScorer

	case = (700001, False, 0)
	root = tmp_path / "data"
	batch = "b-late"
	group = "realistic-fixed-A-fixed"
	cid = case_id(*case)
	traj_id = f"{batch}/mujoco/{group}/{cid}"
	ep_dir = root / "runs" / traj_id

	scorer = FixedScorer(lambda _c, _r: -1.0, {"top": 0.0, "wrist_cam": 0.0})
	trace = EpisodeTrace(ep_dir, video=False)
	policy = Policy()
	twin = ExpertTracer(sample_case(*case), scorer, trace, sensing="truth", strategy="A")
	try:
		result = build_graph(twin, policy).invoke({}, config={"recursion_limit": 64})["result"]
		trace.finish(
			result,
			{
				"engine": "mujoco",
				"sensing": "truth",
				"agent_options": {"strategy": "A", "max_scored_views": 6, "boundary": "belief_cell"},
				"experiment": {"batch": batch, "engine": "mujoco", "group": group, "case_id": cid},
			},
		)
	finally:
		twin.close()

	write_json(
		ep_dir / "task.json",
		{"case": list(case), "policy": policy.__dict__, "video": False, "use_llm": False},
	)
	live_ids = live_checkpoint_ids(ep_dir / "events.jsonl")
	assert "checkpoint-1" in live_ids

	# Inject a wrong-bowl coordinate into the latest checkpoint so resuming from it genuinely reproduces
	# place:PLACEMENT_UNVERIFIED:wrong_bowl_carry in MuJoCo.
	latest_ck = live_ids[-1]
	ck_path = ep_dir / "checkpoints" / f"{latest_ck}.json"
	ck_doc = json.loads(ck_path.read_text(encoding="utf-8"))
	ck_doc["agent"]["bowls"]["green"] = list(ck_doc["agent"]["bowls"]["pink"])
	write_json(ck_path, ck_doc)

	# Annotate episode as a late-stage place failure (after checkpoint-1) so failure_bank triggers resume_from.
	with (ep_dir / "events.jsonl").open("a", encoding="utf-8") as f:
		f.write(
			json.dumps(
				{
					"kind": "failure",
					"code": "PLACEMENT_UNVERIFIED",
					"node": "place",
					"stage": "place",
					"frame": 999999,
				}
			)
			+ "\n"
		)
	ep_json = json.loads((ep_dir / "episode.json").read_text(encoding="utf-8"))
	ep_json["result"].update({"verdict": "FAILED", "destination": "pink", "label": "normal"})
	write_json(ep_dir / "episode.json", ep_json)

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
						"key": ["place", "PLACEMENT_UNVERIFIED", "wrong_bowl_carry"],
						"n": 1,
						"impact": 1.0,
						"share": 1.0,
						"factors": {},
						"examples": [traj_id],
					}
				],
			}
		),
		encoding="utf-8",
	)

	out_suite = tmp_path / "failure_bank" / "round-01-place-PLACEMENT_UNVERIFIED-wrong_bowl_carry.json"
	scratch_dir = tmp_path / "scratch"
	rc = fb.main(
		[
			"--ledger",
			str(ledger_path),
			"--cluster",
			"place:PLACEMENT_UNVERIFIED:wrong_bowl_carry",
			"--round",
			"01",
			"--out",
			str(out_suite),
			"--episodes-root",
			str(root),
			"--scratch-dir",
			str(scratch_dir),
			"--scorer",
			"fixed",
			"--no-manifest",
		]
	)
	assert rc == 0

	suite = json.loads(out_suite.read_text(encoding="utf-8"))
	assert suite["cases"] == [list(case)]
	expected_ck = live_ids[-1]
	assert suite["repro_checked"]["kept"][0]["resume_from"] == expected_ck
	assert suite["resume_from"][cid] == expected_ck

	run_dirs = [p for p in sorted((scratch_dir / cid).glob("run-*")) if p.is_dir()]
	assert len(run_dirs) == 2
	for rd in run_dirs:
		resume_meta = json.loads((rd / "resume.json").read_text(encoding="utf-8"))
		assert resume_meta["checkpoint"] == expected_ck
	with (
		np.load(ep_dir / "branch-0" / "actions.npz", allow_pickle=False) as full_npz,
		np.load(run_dirs[0] / "branch-0" / "actions.npz", allow_pickle=False) as a,
		np.load(run_dirs[1] / "branch-0" / "actions.npz", allow_pickle=False) as b,
	):
		assert 0 < len(a["state_f64"]) < len(full_npz["state_f64"])
		assert np.array_equal(a["state_f64"], b["state_f64"])
		assert np.array_equal(a["action_f64"], b["action_f64"])


def test_failure_bank_manifest_and_runner_integration(tmp_path, monkeypatch):
	ms = _load_script("make_suites")
	v2_dir = tmp_path / "config" / "agent_bench" / "v2"
	v2_dir.mkdir(parents=True)
	manifest_path = v2_dir / "manifest.json"
	manifest_path.write_text((ROOT / "config" / "agent_bench" / "v2" / "manifest.json").read_text(encoding="utf-8"))

	bank_dir = tmp_path / "config" / "agent_bench" / "failure_bank"
	bank_dir.mkdir(parents=True)
	suite_name = "round-01-grasp-no_clamp-calibration_stale"
	suite_path = bank_dir / f"{suite_name}.json"
	cases = [[700001, True, 0, {"cameras": "drift"}], [700002, False, 1, {"cameras": "drift"}]]
	suite_payload = {
		"name": suite_name,
		"sealed": False,
		"source_round": "round-01",
		"cluster": ["grasp", "no_clamp", "calibration_stale"],
		"commit": "deadbeef",
		"cases": cases,
		"repro_checked": {"commit": "deadbeef", "runs_per_case": 2, "kept": [], "dropped": []},
	}
	suite_path.write_text(json.dumps(suite_payload, indent=2) + "\n", encoding="utf-8")

	entry = ms.register_suite(manifest_path, suite_name, suite_path, cases, sealed=False)
	assert entry["cases"] == 2
	assert entry["layouts"] == 2
	assert entry["sealed"] is False
	assert entry["path"] == f"../failure_bank/{suite_name}.json"

	monkeypatch.setattr(runner, "SUITES", v2_dir)
	data_root = tmp_path / "runner_data"
	frozen_path, frozen_sha, sealed = runner._freeze_suite(data_root, suite_name)
	assert frozen_sha == entry["sha256"]
	assert sealed is False
	assert load_suite(frozen_path) == [tuple(c) for c in cases]

	args = argparse.Namespace(
		root=str(data_root),
		sensing="realistic",
		camera="drift",
		strategy="A",
		detector="v7bot",
		policy=None,
		max_scored_views=None,
		boundary="belief_cell",
		batch="bank-dryrun",
		engine="mujoco",
		suite=suite_name,
		freeze=None,
		review_llm_url=None,
		review_model="qwen3.8-vllm",
	)
	_config, jobs, group_dir = runner.plan_group(args)
	assert len(jobs) == 2
	assert [j["experiment"]["case_id"] for j in jobs] == ["L700001-D0", "L700002-N"]
	assert (group_dir / "plan.json").is_file()


def _load_verify_fix():
	return _load_script("verify_fix")


def _score_row(
	case_id: str,
	*,
	layout_id: str | None = None,
	success: bool,
	expected_label: str = "normal",
	group: str = "realistic-drift-A-v7bot",
	engine: str = "mujoco",
) -> dict:
	lid = layout_id or case_id.split("-")[0]
	return {
		"case_id": case_id,
		"layout_id": lid,
		"engine": engine,
		"group": group,
		"expected_label": expected_label,
		"physical_success": bool(success),
		"defect_miss": expected_label == "defective" and not success,
		"false_alarm": expected_label == "normal" and not success,
		"is_unknown": False,
		"wrong_bowl": False,
		"infra_failed": False,
		"root_cause": "none" if success else "calibration_stale",
	}


def _paired_layout_rows(layout_outcomes: dict[int, tuple[bool, bool]], *, group: str = "g") -> list[dict]:
	out: list[dict] = []
	for seed, (ok_n, ok_d) in sorted(layout_outcomes.items()):
		lid = f"L{seed}"
		out.append(_score_row(f"{lid}-N", layout_id=lid, success=ok_n, expected_label="normal", group=group))
		out.append(_score_row(f"{lid}-D0", layout_id=lid, success=ok_d, expected_label="defective", group=group))
	return out


def _write_episode_dir(ep_dir: Path, *, case: list, ok: bool, experiment: dict) -> None:
	ep_dir.mkdir(parents=True, exist_ok=True)
	expected = "defective" if case[1] else "normal"
	dest = "green" if (expected == "normal") == ok else "pink"
	label = expected if ok else ("defective" if expected == "normal" else "normal")
	verdict = "SUCCESS" if ok else "FAILED"
	(ep_dir / "task.json").write_text(json.dumps({"case": case}), encoding="utf-8")
	(ep_dir / "episode.json").write_text(
		json.dumps(
			{
				"schema": "armigo-agent-episode/v1",
				"provenance": {"engine": "mujoco", "experiment": experiment},
				"result": {"verdict": verdict, "label": label, "destination": dest},
			}
		),
		encoding="utf-8",
	)
	(ep_dir / "run_status.json").write_text(json.dumps({"state": "done"}), encoding="utf-8")


def test_f4_gate1_bank_fix_rate_dict_and_jsonl(tmp_path):
	vf = _load_verify_fix()

	base_rows = [_score_row(f"L{100 + i}-D0", success=False, expected_label="defective") for i in range(10)]
	# Include one already-passing row in base; denominator must remain total bank base failures (10).
	base_rows.append(_score_row("L999-N", success=True, expected_label="normal"))

	new_pass = [_score_row(f"L{100 + i}-D0", success=(i < 7), expected_label="defective") for i in range(10)] + [
		_score_row("L999-N", success=True, expected_label="normal")
	]
	g1_pass = vf.evaluate_gate1_bank(base_rows, new_pass, min_fix_rate=0.70)
	assert g1_pass["total_base_failures"] == 10
	assert g1_pass["fixed"] == 7
	assert g1_pass["fix_rate"] == pytest.approx(0.70)
	assert g1_pass["passed"] is True

	new_fail = [_score_row(f"L{100 + i}-D0", success=(i < 6), expected_label="defective") for i in range(10)] + [
		_score_row("L999-N", success=True, expected_label="normal")
	]
	g1_fail = vf.evaluate_gate1_bank(base_rows, new_fail, min_fix_rate=0.70)
	assert g1_fail["total_base_failures"] == 10
	assert g1_fail["fixed"] == 6
	assert g1_fail["fix_rate"] == pytest.approx(0.60)
	assert g1_fail["passed"] is False

	base_jsonl = tmp_path / "base_scores.jsonl"
	new_jsonl = tmp_path / "new_scores.jsonl"
	base_jsonl.write_text("\n".join(json.dumps(r) for r in base_rows) + "\n", encoding="utf-8")
	new_jsonl.write_text("\n".join(json.dumps(r) for r in new_pass) + "\n", encoding="utf-8")
	g1_file = vf.evaluate_gate1_bank(vf.load_score_rows(base_jsonl), vf.load_score_rows(new_jsonl), min_fix_rate=0.70)
	assert g1_file["fixed"] == 7
	assert g1_file["total_base_failures"] == 10
	assert g1_file["passed"] is True


def test_f4_gate2_regression_mcnemar_reuses_stats():
	vf = _load_verify_fix()

	reg_base = [_score_row(f"L{200 + i}-N", success=True) for i in range(20)]

	# 1 new failure -> net_regression = 1 <= 1, p_mcnemar = 1.0 > 0.2 -> passes.
	reg_one_loss = [_score_row(f"L{200 + i}-N", success=(i != 0)) for i in range(20)]
	g2_ok = vf.evaluate_gate2_regression(reg_base, reg_one_loss, max_net_regression=1, min_mcnemar_p=0.2)
	expected_ok = paired_comparison(reg_base, reg_one_loss, metric="success_rate", B=200, seed=0)
	assert g2_ok["n10"] == 1 and g2_ok["n01"] == 0
	assert g2_ok["net_regression"] == 1
	assert g2_ok["p_mcnemar"] == pytest.approx(_exact_mcnemar(1, 0))
	assert g2_ok["p_mcnemar"] == pytest.approx(expected_ok["p_mcnemar"])
	assert g2_ok["passed"] is True

	# 6 new failures -> net_regression = 6 > 1 and p_mcnemar = 0.03125 <= 0.2 -> fails.
	reg_six_losses = [_score_row(f"L{200 + i}-N", success=(i >= 6)) for i in range(20)]
	g2_fail = vf.evaluate_gate2_regression(reg_base, reg_six_losses, max_net_regression=1, min_mcnemar_p=0.2)
	assert g2_fail["n10"] == 6 and g2_fail["n01"] == 0
	assert g2_fail["net_regression"] == 6
	assert g2_fail["p_mcnemar"] == pytest.approx(_exact_mcnemar(6, 0))
	assert g2_fail["p_mcnemar"] < 0.2
	assert g2_fail["passed"] is False

	# 2 new failures -> p_mcnemar = 0.5 > 0.2, but net_regression = 2 > 1 -> fails.
	reg_two_losses = [_score_row(f"L{200 + i}-N", success=(i >= 2)) for i in range(20)]
	g2_net_fail = vf.evaluate_gate2_regression(reg_base, reg_two_losses, max_net_regression=1, min_mcnemar_p=0.2)
	assert g2_net_fail["n10"] == 2 and g2_net_fail["n01"] == 0
	assert g2_net_fail["passed"] is False

	# Pure improvement on previously failing regression rows (n10=0, n01=8) -> never treated as regression.
	mixed_base = [_score_row(f"L{300 + i}-N", success=(i >= 8)) for i in range(20)]
	all_pass = [_score_row(f"L{300 + i}-N", success=True) for i in range(20)]
	g2_improved = vf.evaluate_gate2_regression(mixed_base, all_pass, max_net_regression=1, min_mcnemar_p=0.2)
	assert g2_improved["n10"] == 0 and g2_improved["n01"] == 8
	assert g2_improved["passed"] is True


def test_f4_gate3_fresh_clustered_bootstrap_ci():
	vf = _load_verify_fix()

	# 30 layouts, 2 cases each = 60 pairs.
	base_map = {seed: (True, seed % 5 != 0) for seed in range(1, 31)}
	# Slight improvement on 3 layouts -> CI lower bound >= -0.02.
	new_ok_map = {seed: (True, True if seed in (5, 10, 15) else ok_d) for seed, (_ok_n, ok_d) in base_map.items()}

	fresh_base = _paired_layout_rows(base_map)
	fresh_new_ok = _paired_layout_rows(new_ok_map)
	g3_ok = vf.evaluate_gate3_fresh(fresh_base, fresh_new_ok, min_ci_lo=-0.02, bootstrap_b=1000, seed=7)
	pc_ok = paired_comparison(fresh_base, fresh_new_ok, metric="success_rate", B=1000, seed=7)
	assert g3_ok["n_pairs"] == 60
	assert g3_ok["n_clusters"] == 30
	assert g3_ok["diff"] == pytest.approx(pc_ok["diff"])
	assert g3_ok["ci_lo"] == pytest.approx(pc_ok["ci_lo"])
	assert g3_ok["ci_hi"] == pytest.approx(pc_ok["ci_hi"])
	assert g3_ok["ci_lo"] >= -0.02
	assert g3_ok["passed"] is True

	# Drop both cases in 4 layouts -> mean diff = -8/60 = -13.3%, ci_lo < -0.02 -> fails.
	new_bad_map = {seed: (False, False) if seed <= 4 else vals for seed, vals in base_map.items()}
	fresh_new_bad = _paired_layout_rows(new_bad_map)
	g3_bad = vf.evaluate_gate3_fresh(fresh_base, fresh_new_bad, min_ci_lo=-0.02, bootstrap_b=1000, seed=7)
	assert g3_bad["diff"] < -0.05
	assert g3_bad["ci_lo"] < -0.02
	assert g3_bad["passed"] is False


def test_f4_evaluate_three_gates_all_verdicts():
	vf = _load_verify_fix()

	bank_base = [_score_row(f"B{i}-D0", success=False, expected_label="defective") for i in range(10)]
	bank_pass = [_score_row(f"B{i}-D0", success=(i < 8), expected_label="defective") for i in range(10)]
	bank_fail = [_score_row(f"B{i}-D0", success=(i < 5), expected_label="defective") for i in range(10)]

	reg_base = [_score_row(f"R{i}-N", success=True) for i in range(20)]
	reg_pass = [_score_row(f"R{i}-N", success=True) for i in range(20)]
	reg_fail = [_score_row(f"R{i}-N", success=(i >= 6)) for i in range(20)]

	fresh_base = _paired_layout_rows({s: (True, True) for s in range(1, 21)})
	fresh_pass = _paired_layout_rows({s: (True, True) for s in range(1, 21)})
	fresh_fail = _paired_layout_rows({s: (False, False) if s <= 5 else (True, True) for s in range(1, 21)})

	criteria = {
		"bank_min_fix_rate": 0.70,
		"regression_max_net_regression": 1,
		"regression_min_mcnemar_p": 0.20,
		"fresh_min_ci_lo": -0.02,
		"bootstrap_b": 500,
		"seed": 0,
	}

	res_merge = vf.evaluate_three_gates(bank_pass, bank_pass, reg_base, reg_pass, fresh_base, fresh_pass, criteria)
	# Note: bank_pass vs bank_pass has 0 base failures -> use bank_base vs bank_pass!
	assert res_merge["verdict"] == "reject"

	res_merge = vf.evaluate_three_gates(bank_base, bank_pass, reg_base, reg_pass, fresh_base, fresh_pass, criteria)
	assert res_merge["gates"]["bank"]["passed"] is True
	assert res_merge["gates"]["regression"]["passed"] is True
	assert res_merge["gates"]["fresh"]["passed"] is True
	assert res_merge["verdict"] == "merge"

	res_overfit = vf.evaluate_three_gates(bank_base, bank_pass, reg_base, reg_pass, fresh_base, fresh_fail, criteria)
	assert res_overfit["gates"]["bank"]["passed"] is True
	assert res_overfit["gates"]["regression"]["passed"] is True
	assert res_overfit["gates"]["fresh"]["passed"] is False
	assert res_overfit["verdict"] == "overfit_bank"

	res_reject_g1 = vf.evaluate_three_gates(bank_base, bank_fail, reg_base, reg_pass, fresh_base, fresh_pass, criteria)
	assert res_reject_g1["gates"]["bank"]["passed"] is False
	assert res_reject_g1["verdict"] == "reject"

	res_reject_g2 = vf.evaluate_three_gates(bank_base, bank_pass, reg_base, reg_fail, fresh_base, fresh_pass, criteria)
	assert res_reject_g2["gates"]["regression"]["passed"] is False
	assert res_reject_g2["verdict"] == "reject"

	res_reject_g1_g3 = vf.evaluate_three_gates(
		bank_base, bank_fail, reg_base, reg_pass, fresh_base, fresh_fail, criteria
	)
	assert res_reject_g1_g3["gates"]["bank"]["passed"] is False
	assert res_reject_g1_g3["gates"]["fresh"]["passed"] is False
	assert res_reject_g1_g3["verdict"] == "reject"


def test_f4_cli_skip_run_reads_base_and_writes_reports(tmp_path):
	vf = _load_verify_fix()

	base_root = tmp_path / "base_ro"
	new_root = tmp_path / "new_rw"
	suites_dir = new_root / "suites"
	suites_dir.mkdir(parents=True)

	bank_cases = [[101 + i, True, 0] for i in range(4)]
	reg_cases = [[201 + i, False, 0] for i in range(6)]
	fresh_cases = []
	for s in range(301, 311):
		fresh_cases.extend([[s, False, 0], [s, True, 0]])

	bank_suite = suites_dir / "bank_s.json"
	reg_suite = suites_dir / "reg_s.json"
	fresh_suite = suites_dir / "fresh_s.json"
	bank_suite.write_text(json.dumps({"name": "bank_s", "cases": bank_cases}), encoding="utf-8")
	reg_suite.write_text(json.dumps({"name": "reg_s", "cases": reg_cases}), encoding="utf-8")
	fresh_suite.write_text(json.dumps({"name": "fresh_s", "cases": fresh_cases}), encoding="utf-8")

	# Populate base_root with pre-existing scores.jsonl (must remain untouched).
	group = "realistic-drift-A-v7bot"
	base_rows = (
		[_score_row(f"L{101 + i}-D0", success=False, expected_label="defective", group=group) for i in range(4)]
		+ [_score_row(f"L{201 + i}-N", success=True, expected_label="normal", group=group) for i in range(6)]
		+ [_score_row(f"L{s}-N", success=True, expected_label="normal", group=group) for s in range(301, 311)]
		+ [_score_row(f"L{s}-D0", success=True, expected_label="defective", group=group) for s in range(301, 311)]
	)
	base_rep = base_root / "reports" / "base-b"
	base_rep.mkdir(parents=True)
	base_scores_file = base_rep / "scores.jsonl"
	base_scores_text = "\n".join(json.dumps(r) for r in base_rows) + "\n"
	base_scores_file.write_text(base_scores_text, encoding="utf-8")

	# Populate new_root with episode dirs under runs/new-b/mujoco/<group>/<case_id>/
	batch_dir = new_root / "runs" / "new-b" / "mujoco" / group
	for c in bank_cases:
		cid = f"L{c[0]}-D0"
		_write_episode_dir(
			batch_dir / cid, case=c, ok=True, experiment={"batch": "new-b", "group": group, "case_id": cid}
		)
	for c in reg_cases:
		cid = f"L{c[0]}-N"
		_write_episode_dir(
			batch_dir / cid, case=c, ok=True, experiment={"batch": "new-b", "group": group, "case_id": cid}
		)
	for c in fresh_cases:
		cid = f"L{c[0]}-D0" if c[1] else f"L{c[0]}-N"
		_write_episode_dir(
			batch_dir / cid, case=c, ok=True, experiment={"batch": "new-b", "group": group, "case_id": cid}
		)

	criteria_path = tmp_path / "criteria.json"
	criteria_path.write_text(
		json.dumps(
			{
				"bank_min_fix_rate": 0.70,
				"regression_max_net_regression": 1,
				"regression_min_mcnemar_p": 0.20,
				"fresh_min_ci_lo": -0.02,
				"bootstrap_b": 400,
				"seed": 0,
			}
		),
		encoding="utf-8",
	)
	out_dir = tmp_path / "verify_out"

	rc = vf.main(
		[
			"--bank",
			str(bank_suite),
			"--regression",
			str(reg_suite),
			"--fresh",
			str(fresh_suite),
			"--base",
			"base-b",
			"--batch",
			"new-b",
			"--base-root",
			str(base_root),
			"--root",
			str(new_root),
			"--criteria",
			str(criteria_path),
			"--out-dir",
			str(out_dir),
			"--cluster",
			"grasp:no_clamp:calibration_stale",
			"--skip-run",
		]
	)
	assert rc == 0

	# Base root remained untouched.
	assert sorted(p.relative_to(base_root).as_posix() for p in base_root.rglob("*") if p.is_file()) == [
		"reports/base-b/scores.jsonl"
	]
	assert base_scores_file.read_text(encoding="utf-8") == base_scores_text

	out_json = out_dir / "verify-grasp-no_clamp-calibration_stale.json"
	out_md = out_dir / "verify-grasp-no_clamp-calibration_stale.md"
	assert out_json.is_file()
	assert out_md.is_file()

	payload = json.loads(out_json.read_text(encoding="utf-8"))
	assert payload["verdict"] == "merge"
	assert payload["cluster"] == "grasp:no_clamp:calibration_stale"
	assert payload["gates"]["bank"]["fix_rate"] == pytest.approx(1.0)
	assert payload["gates"]["regression"]["net_regression"] == 0
	assert payload["gates"]["fresh"]["n_pairs"] == 20
	assert payload["gates"]["fresh"]["n_clusters"] == 10

	md_text = out_md.read_text(encoding="utf-8")
	assert "merge" in md_text
	assert "Gate 1" in md_text and "Gate 2" in md_text and "Gate 3" in md_text


def test_f4_cli_run_invokes_runner_unless_skip_run(tmp_path, monkeypatch):
	vf = _load_verify_fix()

	root = tmp_path / "data"
	suites_dir = root / "suites"
	suites_dir.mkdir(parents=True)
	bank_suite = suites_dir / "b_suite.json"
	reg_suite = suites_dir / "r_suite.json"
	fresh_suite = suites_dir / "f_suite.json"
	bank_suite.write_text(json.dumps([[101, True, 0]]), encoding="utf-8")
	reg_suite.write_text(json.dumps([[201, False, 0]]), encoding="utf-8")
	fresh_suite.write_text(json.dumps([[301, False, 0], [301, True, 0]]), encoding="utf-8")

	group = "realistic-fixed-A-v7bot"
	base_rows = [
		_score_row("L101-D0", success=False, expected_label="defective", group=group),
		_score_row("L201-N", success=True, expected_label="normal", group=group),
		_score_row("L301-N", success=True, expected_label="normal", group=group),
		_score_row("L301-D0", success=True, expected_label="defective", group=group),
	]
	base_rep = root / "reports" / "base-b"
	base_rep.mkdir(parents=True)
	(base_rep / "scores.jsonl").write_text("\n".join(json.dumps(r) for r in base_rows) + "\n", encoding="utf-8")

	called: list[tuple[str, str]] = []

	def fake_run_suite(args, suite_spec: str, batch_name: str) -> None:
		called.append((suite_spec, batch_name))
		rep = root / "reports" / batch_name
		rep.mkdir(parents=True, exist_ok=True)
		rows = [
			_score_row("L101-D0", success=True, expected_label="defective", group=group),
			_score_row("L201-N", success=True, expected_label="normal", group=group),
			_score_row("L301-N", success=True, expected_label="normal", group=group),
			_score_row("L301-D0", success=True, expected_label="defective", group=group),
		]
		(rep / "scores.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

	monkeypatch.setattr(vf, "run_suite_batch", fake_run_suite)

	criteria_path = tmp_path / "criteria.json"
	criteria_path.write_text(json.dumps({"bootstrap_b": 200}), encoding="utf-8")

	rc = vf.main(
		[
			"--bank",
			"b_suite",
			"--regression",
			"r_suite",
			"--fresh",
			"f_suite",
			"--base",
			"base-b",
			"--batch",
			"new-b",
			"--root",
			str(root),
			"--criteria",
			str(criteria_path),
			"--cluster",
			"test_cluster",
		]
	)
	assert rc == 0
	assert called == [
		("b_suite", "new-b-bank"),
		("r_suite", "new-b-regression"),
		("f_suite", "new-b-fresh"),
	]
	assert (root / "reports" / "new-b" / "verify-test_cluster.json").is_file()
