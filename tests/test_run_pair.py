from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

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
