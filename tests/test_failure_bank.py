"""Tests for deterministic failure bank builder and runner integration (Task F3)."""

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
	(ep_dir / "episode.json").write_text(
		json.dumps({"result": {"verdict": "FAILED", "label": "normal", "destination": "pink"}}),
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
	first_case = list(dev_cases[0])
	assert first_case == [700001, False, 0]

	root = tmp_path / "data"
	batch = "b-repro"
	group = "realistic-fixed-A-fixed"
	cid = case_id(*first_case[:3])
	traj_id = f"{batch}/mujoco/{group}/{cid}"
	_write_synth_episode(
		root / "runs" / traj_id,
		case=first_case,
		result={"verdict": "FAILED", "label": "normal"},
		events=[{"kind": "segment", "segment_kind": "grasp", "frame": 50}, {"kind": "failure", "code": "no_clamp"}],
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
						"key": ["grasp", "no_clamp", "grasp_or_path_failure"],
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

	out_suite = tmp_path / "failure_bank" / "round-01-grasp-no_clamp-grasp_or_path_failure.json"
	manifest_path = tmp_path / "v2" / "manifest.json"
	manifest_path.parent.mkdir(parents=True)
	manifest_path.write_text((ROOT / "config" / "agent_bench" / "v2" / "manifest.json").read_text(encoding="utf-8"))
	scratch_dir = tmp_path / "scratch"

	rc = fb.main(
		[
			"--ledger",
			str(ledger_path),
			"--cluster",
			"grasp:no_clamp:grasp_or_path_failure",
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
	assert suite["name"] == "round-01-grasp-no_clamp-grasp_or_path_failure"
	assert suite["sealed"] is False
	assert suite["source_round"] == "round-01"
	assert suite["cluster"] == ["grasp", "no_clamp", "grasp_or_path_failure"]
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
	twin = ExpertTracer(sample_case(*case), scorer, trace, sensing="realistic", strategy="A")
	try:
		result = build_graph(twin, policy).invoke({}, config={"recursion_limit": 64})["result"]
		trace.finish(
			result,
			{
				"engine": "mujoco",
				"sensing": "realistic",
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
