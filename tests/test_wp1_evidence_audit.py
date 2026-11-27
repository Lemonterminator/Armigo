from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

import pytest

from digital_twin_armigo.experiments import diag_mcp
from digital_twin_armigo.governance import bronze, lake, silver
from digital_twin_armigo.inventory import audit

REPO = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
	return hashlib.sha256(path.read_bytes()).hexdigest()


def _create_legacy_audit_db(db_path: Path, traj_id: str) -> None:
	"""Create an old rollback-journal reviews.sqlite WITHOUT the ``mechanism`` column."""
	db_path.parent.mkdir(parents=True, exist_ok=True)
	con = sqlite3.connect(db_path)
	try:
		con.executescript(
			"""
			create table reviews (
				id integer primary key autoincrement,
				traj_id text not null,
				created_at text not null,
				reviewer text not null,
				state text not null,
				root_cause text,
				physical_outcome text,
				note text,
				refs text not null default '[]',
				code_version text,
				replaces integer
			);
			create index reviews_traj on reviews (traj_id, id);
			"""
		)
		con.execute(
			"insert into reviews (traj_id, created_at, reviewer, state, root_cause, physical_outcome, note, refs, code_version, replaces) "
			"values (?, '2026-11-24T00:00:00+00:00', 'auditor', 'confirmed', 'recovery_failure', 'dropped', 'legacy row', '[]', 'abc', null)",
			(traj_id,),
		)
		con.commit()
	finally:
		con.close()


def _make_episode(root: Path, batch: str, group: str, cid: str, *, seed: int = 3000101) -> Path:
	d = root / "runs" / batch / "mujoco" / group / cid
	d.mkdir(parents=True, exist_ok=True)
	(d / "run_status.json").write_text(json.dumps({"state": "done"}), encoding="utf-8")
	(d / "task.json").write_text(json.dumps({"case": [seed, False, 0]}), encoding="utf-8")
	(d / "episode.json").write_text(
		json.dumps(
			{
				"result": {"label": "normal", "verdict": "SUCCESS"},
				"provenance": {
					"experiment": {
						"batch": batch,
						"group": group,
						"engine": "mujoco",
						"camera_condition": "fixed",
						"detector": "v7bot",
						"strategy": "B",
						"case_id": cid,
					}
				},
			}
		),
		encoding="utf-8",
	)
	(d / "events.jsonl").write_text("", encoding="utf-8")
	return d


def _load_script(rel: str, name: str):
	spec = importlib.util.spec_from_file_location(name, REPO / rel)
	mod = importlib.util.module_from_spec(spec)
	assert spec and spec.loader
	spec.loader.exec_module(mod)
	return mod


def test_readonly_audit_does_not_mutate_legacy_db_or_create_sidecars(tmp_path):
	ep = _make_episode(tmp_path, "b1", "realistic-fixed-B-v7bot", "L3000101-N")
	tid = "agent:v3:b1/mujoco/realistic-fixed-B-v7bot/L3000101-N"
	legacy_db = tmp_path / "legacy_audit" / "reviews.sqlite"
	_create_legacy_audit_db(legacy_db, tid)

	before_sha = _sha256(legacy_db)
	before_files = sorted(p.name for p in legacy_db.parent.iterdir())

	# 1. Direct read-only queries on legacy schema without mechanism column
	rev = audit.latest(legacy_db, tid)
	assert rev is not None
	assert rev["state"] == "confirmed"
	assert rev["root_cause"] == "recovery_failure"
	assert rev["mechanism"] is None
	assert len(audit.history(legacy_db, tid)) == 1
	assert audit.summary(legacy_db)["states"]["confirmed"] == 1

	# 2. H7 make_diagnosis_bench reading legacy DB
	inv_dir = tmp_path / "inv"
	inv_dir.mkdir()
	inv_con = sqlite3.connect(inv_dir / "inventory.sqlite")
	inv_con.execute(
		"create table trajectories (traj_id text primary key, path text not null, record_kind text not null)"
	)
	inv_con.execute("insert into trajectories values (?, ?, 'agent_episode')", (tid, str(ep)))
	inv_con.commit()
	inv_con.close()

	mdb = _load_script("scripts/experiments/make_diagnosis_bench.py", "mdb_wp1")
	entries = mdb.build_benchmark(legacy_db, inv_dir)
	assert len(entries) == 1
	assert entries[0]["root_cause"] == "recovery_failure"
	assert entries[0]["mechanism"] is None

	# 3. P3 silver build reading legacy DB
	lcon = lake.connect(tmp_path / "lake.sqlite")
	bronze.register(lcon, tmp_path, source="v3")
	bid = silver.build_silver(lcon, audit_path=legacy_db)
	row = json.loads(
		lcon.execute("select row_json from silver_episode where build_id = ? and path = ?", (bid, str(ep))).fetchone()[
			0
		]
	)
	assert row["audit_state"] == "confirmed"
	assert row["audit_root_cause"] == "recovery_failure"
	assert row["audit_mechanism"] is None
	lcon.close()

	# Legacy DB must remain byte-for-byte identical, without WAL/SHM sidecars or added columns
	assert _sha256(legacy_db) == before_sha
	assert sorted(p.name for p in legacy_db.parent.iterdir()) == before_files
	ro = sqlite3.connect(f"file:{legacy_db.resolve().as_posix()}?mode=ro", uri=True)
	cols = {r[1] for r in ro.execute("pragma table_info(reviews)").fetchall()}
	ro.close()
	assert "mechanism" not in cols


def test_silver_build_id_versions_human_audit_and_preserves_history(tmp_path):
	ep = _make_episode(tmp_path, "b1", "realistic-fixed-B-v7bot", "L3000101-N")
	tid = "agent:v3:b1/mujoco/realistic-fixed-B-v7bot/L3000101-N"
	audit_db = tmp_path / "audit" / "reviews.sqlite"

	lcon = lake.connect(tmp_path / "lake.sqlite")
	bronze.register(lcon, tmp_path, source="v3")

	bid_before = silver.build_silver(lcon, audit_path=audit_db)
	row_before = json.loads(
		lcon.execute(
			"select row_json from silver_episode where build_id = ? and path = ?", (bid_before, str(ep))
		).fetchone()[0]
	)
	assert row_before["audit_state"] is None
	assert row_before["audit_mechanism"] is None

	# Human appends an asynchronous review
	audit.append(
		audit_db,
		tid,
		{
			"reviewer": "human",
			"state": "confirmed",
			"root_cause": "recovery_failure",
			"mechanism": "brick_pose_error",
		},
		expected_latest_id=None,
		code_version="abc",
	)

	bid_after = silver.build_silver(lcon, audit_path=audit_db)
	assert bid_after != bid_before

	# Old build must still retain its original pre-edit audit state
	old_row = json.loads(
		lcon.execute(
			"select row_json from silver_episode where build_id = ? and path = ?", (bid_before, str(ep))
		).fetchone()[0]
	)
	assert old_row["audit_state"] is None
	assert old_row["audit_mechanism"] is None

	# New build reflects the appended verdict
	new_row = json.loads(
		lcon.execute(
			"select row_json from silver_episode where build_id = ? and path = ?", (bid_after, str(ep))
		).fetchone()[0]
	)
	assert new_row["audit_state"] == "confirmed"
	assert new_row["audit_mechanism"] == "brick_pose_error"

	# Idempotent rebuild without changes keeps the same build_id
	bid_again = silver.build_silver(lcon, audit_path=audit_db)
	assert bid_again == bid_after
	lcon.close()


def test_manifest_export_preserves_source_mkvs_and_staging_cleanup_logs_events(tmp_path):
	sys.path.insert(0, str(REPO / "scripts" / "vla"))
	import collect_paired_carry as pc

	bronze_ep = tmp_path / "bronze" / "ep1"
	(bronze_ep / "branch-0").mkdir(parents=True)
	(bronze_ep / "episode.json").write_text('{"ok": true}\n', encoding="utf-8")
	src_mkv = bronze_ep / "branch-0" / "top_rgb.mkv"
	src_mkv.write_bytes(b"bronze source sentinel")

	out_dir = tmp_path / "export_out"
	out_dir.mkdir()

	def fake_run_ok(args, **kwargs):
		if "-m" in args:
			shard_dir = Path(args[args.index("--out") + 1])
			shard_dir.mkdir(parents=True, exist_ok=True)
		return CompletedProcess(args, 0, stdout="[1, 10]\n")

	def fake_run_bad_readback(args, **kwargs):
		if "-m" in args:
			shard_dir = Path(args[args.index("--out") + 1])
			shard_dir.mkdir(parents=True, exist_ok=True)
		return CompletedProcess(args, 0, stdout="[1, 999]\n")

	# 1. Default _export_shard must NOT delete source MKVs
	with (
		patch.object(pc.subprocess, "run", side_effect=fake_run_ok),
		patch.object(pc, "_expected_frames", return_value=10),
	):
		pc._export_shard(out_dir / "shard-direct", [bronze_ep], Path("unused"))
	assert src_mkv.is_file() and src_mkv.read_bytes() == b"bronze source sentinel"

	# 2. Readback failure leaves source MKVs intact
	with (
		patch.object(pc.subprocess, "run", side_effect=fake_run_bad_readback),
		patch.object(pc, "_expected_frames", return_value=10),
		pytest.raises(RuntimeError, match="read back"),
	):
		pc._export_shard(out_dir / "shard-fail", [bronze_ep], Path("unused"))
	assert src_mkv.is_file()

	# 3. Manifest export_pending (success + retry) preserves bronze source MKVs
	manifest_out = tmp_path / "manifest_export"
	manifest_out.mkdir()
	with (
		patch.object(pc.subprocess, "run", side_effect=fake_run_ok),
		patch.object(pc, "_expected_frames", return_value=10),
	):
		shards1 = pc.export_pending(manifest_out, Path("unused"), jobs=1, eps_override=[bronze_ep])
		shards2 = pc.export_pending(manifest_out, Path("unused"), jobs=1, eps_override=[bronze_ep])
	assert len(shards1) == 1 and shards2 == []
	assert src_mkv.is_file() and src_mkv.read_bytes() == b"bronze source sentinel"

	# 4. Attempting delete_mkv on a path outside staging_root raises and preserves source
	with (
		patch.object(pc.subprocess, "run", side_effect=fake_run_ok),
		patch.object(pc, "_expected_frames", return_value=10),
		pytest.raises(ValueError, match="staging"),
	):
		pc._export_shard(
			manifest_out / "shard-bad-del",
			[bronze_ep],
			Path("unused"),
			delete_mkv=True,
			staging_root=manifest_out,
		)
	assert src_mkv.is_file()

	# 5. Explicit staging cleanup inside staging_root deletes staging MKV and logs event
	staging_dir = tmp_path / "staging"
	staged_ep = staging_dir / "rollouts" / "ep1"
	(staged_ep / "branch-0").mkdir(parents=True)
	staged_mkv = staged_ep / "branch-0" / "top_rgb.mkv"
	staged_mkv.write_bytes(b"staging temp")
	with (
		patch.object(pc.subprocess, "run", side_effect=fake_run_ok),
		patch.object(pc, "_expected_frames", return_value=10),
	):
		pc._export_shard(
			staging_dir / "shards" / "shard-000",
			[staged_ep],
			Path("unused"),
			delete_mkv=True,
			staging_root=staging_dir,
		)
	assert not staged_mkv.exists()
	log_file = staging_dir / "cleanup_events.jsonl"
	assert log_file.is_file()
	events = [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines() if line.strip()]
	assert len(events) == 1
	assert str(staged_mkv) in events[0]["deleted_mkvs"]


def test_diag_mcp_requires_round_dir_and_never_writes_to_episode_parent(tmp_path):
	from mcp.server.fastmcp.exceptions import ToolError

	ep = _make_episode(tmp_path, "b1", "realistic-fixed-B-v7bot", "L3000101-N")
	mcp = diag_mcp.create_diag_mcp()

	with pytest.raises((ToolError, ValueError), match="round_dir"):
		asyncio.run(mcp.call_tool("episode_summary", {"path": str(ep), "session": "s1"}))
	assert not (ep.parent / "diag_calls.jsonl").exists()

	with pytest.raises((ToolError, ValueError), match="round_dir"):
		asyncio.run(mcp.call_tool("compare_runs", {"a": str(ep), "b": str(ep), "session": "s1"}))
	assert not (ep.parent / "diag_calls.jsonl").exists()

	rdir = tmp_path / "round-01"
	rdir.mkdir()
	asyncio.run(mcp.call_tool("episode_summary", {"path": str(ep), "session": "s1", "round_dir": str(rdir)}))
	assert (rdir / "diag_calls.jsonl").is_file()
	assert not (ep.parent / "diag_calls.jsonl").exists()


def test_inventory_html_supports_mechanism_input_and_display():
	html = (REPO / "src" / "digital_twin_armigo" / "viewer" / "inventory.html").read_text(encoding="utf-8")
	assert "ENUMS.mechanism" in html
	assert "mechanism:" in html
	assert "r.mechanism" in html
