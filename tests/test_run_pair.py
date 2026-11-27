from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from digital_twin_armigo.experiments import runner
from digital_twin_armigo.experiments.runner import pair_jobs_order

REPO = Path(__file__).resolve().parents[1]


def _load_lake_cli():
	path = REPO / "scripts" / "governance" / "lake.py"
	spec = importlib.util.spec_from_file_location("gov_lake_cli", path)
	mod = importlib.util.module_from_spec(spec)
	assert spec.loader is not None
	spec.loader.exec_module(mod)
	return mod


def test_pair_jobs_are_interleaved_by_case():
	jobs = [
		{"group": "truth-B-v7bot", "case": ["L1"]},
		{"group": "truth-B-v7bot", "case": ["L2"]},
		{"group": "realistic-drift-B-v7bot", "case": ["L1"]},
		{"group": "realistic-drift-B-v7bot", "case": ["L2"]},
	]
	order = [(j["case"][0], j["group"].split("-")[0]) for j in pair_jobs_order(jobs)]
	assert order == [("L1", "truth"), ("L1", "realistic"), ("L2", "truth"), ("L2", "realistic")]


def test_collect_paired_carry_manifest_rejects_hash_mismatch(tmp_path):
	sys.path.insert(0, str(REPO / "scripts" / "vla"))
	import collect_paired_carry as cpc

	ep_dir = tmp_path / "ep1"
	ep_dir.mkdir()
	ep_file = ep_dir / "episode.json"
	ep_file.write_text('{"ok": true}\n', encoding="utf-8")

	manifest = tmp_path / "manifest.jsonl"
	manifest.write_text(
		json.dumps({"path": str(ep_dir), "episode_sha256": "0" * 64}) + "\n",
		encoding="utf-8",
	)
	out_dir = tmp_path / "out_export"
	with pytest.raises(ValueError, match="sha256|hash"):
		cpc.load_manifest_episodes(manifest)
	with pytest.raises((ValueError, SystemExit)):
		cpc.main(["--manifest", str(manifest), "--out", str(out_dir), "--export-only"])
	assert not (out_dir / "shards").exists()


def test_lake_cli_db_flag_before_and_after_subcommand(tmp_path, monkeypatch):
	lake_cli = _load_lake_cli()
	default_guard = tmp_path / "default_should_not_exist.sqlite"
	monkeypatch.setattr(lake_cli, "DEFAULT_DB", default_guard)

	runs = tmp_path / "runs"
	runs.mkdir()
	db_before = tmp_path / "before" / "lake.sqlite"
	db_after = tmp_path / "after" / "lake.sqlite"

	assert lake_cli.main(["--db", str(db_before), "register", "--root", str(runs), "--source", "t"]) == 0
	assert db_before.is_file()
	assert not default_guard.exists()

	assert lake_cli.main(["register", "--db", str(db_after), "--root", str(runs), "--source", "t"]) == 0
	assert db_after.is_file()
	assert not default_guard.exists()


def test_paired_freeze_runs_both_sensing_modes_and_rejects_changed_configuration(tmp_path, monkeypatch):
	suites_dir = tmp_path / "suites-source"
	suites_dir.mkdir()
	suite = suites_dir / "test_suite.json"
	suite.write_text(json.dumps([[600008, False, 5], [600008, True, 1]]), encoding="utf-8")
	(suites_dir / "manifest.json").write_text(
		json.dumps({"suites": {"test_suite": {"sha256": runner._sha256(suite), "sealed": True}}}),
		encoding="utf-8",
	)
	checkpoint = tmp_path / "checkpoint.bin"
	checkpoint.write_bytes(b"checkpoint")
	detector = tmp_path / "detector.json"
	detector.write_text(json.dumps({"checkpoint": checkpoint.name}), encoding="utf-8")
	monkeypatch.setattr(runner, "SUITES", suites_dir)
	monkeypatch.setattr(runner, "code_commit", lambda: "frozen-commit")
	items_seen = []

	def record_jobs(items, *_args):
		items_seen.extend(items)
		return 0

	monkeypatch.setattr(runner, "_execute_jobs", record_jobs)
	freeze = tmp_path / "freeze.json"
	root = tmp_path / "runs-root"
	common = [
		"--detector-spec",
		str(detector),
		"--suite",
		"test_suite",
		"--strategy",
		"B",
		"--real-cameras",
		"fixed,drift",
	]
	assert runner.main(["freeze-pair", *common, "--out", str(freeze)]) == 0
	args = ["run-pair", *common, "--root", str(root), "--batch", "paired", "--case-limit", "1", "--freeze", str(freeze)]
	assert runner.main(args) == 0
	assert len(items_seen) == 3
	configs, jobs = zip(*items_seen, strict=True)
	assert [c["sensing"] for c in configs] == ["truth", "realistic", "realistic"]
	assert [c["agent_options"]["drift_alarm"] for c in configs] == [False, False, True]
	assert [j["experiment"]["localise"] for j in jobs] == ["top", "top", "vote"]
	assert [j["experiment"]["drift_alarm"] for j in jobs] == [False, False, True]
	assert all(j["experiment"]["sealed"] for j in jobs)
	assert len({j["experiment"]["freeze_sha256"] for j in jobs}) == 1
	assert [j["experiment"]["camera_condition"] for j in jobs] == ["fixed", "fixed", "drift"]
	assert all(j["experiment"]["detector_checkpoint_sha256"] == runner._sha256(checkpoint) for j in jobs)

	with pytest.raises(SystemExit, match=r"groups\.realistic-drift-B-v7bot-loc-vote"):
		runner.main([*args, "--no-drift-alarm"])
	assert len(items_seen) == 3
	checkpoint.write_bytes(b"different checkpoint")
	with pytest.raises(SystemExit, match="detector_checkpoint_sha256"):
		runner.main(args)


def test_drift_alarm_ablation_leaves_vote_enabled_and_allows_oracle(monkeypatch):
	items_seen = []
	monkeypatch.setattr(runner, "cmd_run", lambda a: items_seen.append(runner._group_setup(a)) or 0)
	args = ["run", "--sensing", "realistic", "--camera", "drift", "--suite", "smoke_600008", "--batch", "foo"]
	for part in ("camera", "brick"):
		with pytest.raises(SystemExit, match="requires.*measured brick and camera"):
			runner.main([*args, "--oracle", part])
		assert runner.main([*args, "--no-drift-alarm", "--oracle", part]) == 0
	group, setup = items_seen[0]
	assert group == "realistic-drift-A-v7bot-oracle-camera-loc-vote"
	assert setup["agent_options"]["localise"] == "vote"
	assert setup["agent_options"]["drift_alarm"] is False


def test_dev_resume_refuses_a_different_commit(tmp_path, monkeypatch):
	detector = tmp_path / "detector.json"
	detector.write_text("{}", encoding="utf-8")
	monkeypatch.setattr(runner, "_execute_jobs", lambda *_args: 0)
	monkeypatch.setattr(runner, "code_commit", lambda: "commit-a")
	args = [
		"run-pair",
		"--suite",
		"smoke_600008",
		"--root",
		str(tmp_path),
		"--batch",
		"dev",
		"--detector-spec",
		str(detector),
		"--real-cameras",
		"fixed",
	]
	assert runner.main(args) == 0
	monkeypatch.setattr(runner, "code_commit", lambda: "commit-b")
	with pytest.raises(SystemExit, match="planned differently.*code_commit"):
		runner.main(args)
