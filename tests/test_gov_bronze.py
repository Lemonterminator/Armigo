import json

from digital_twin_armigo.governance import bronze, lake


def make_ep(root, name, events="{}\n"):
	d = root / "runs" / "b1" / "mujoco" / "truth-v7bot" / name
	d.mkdir(parents=True)
	(d / "run_status.json").write_text(json.dumps({"state": "done"}))
	(d / "events.jsonl").write_text(events)
	return d


def test_register_is_append_only_and_detects_changes(tmp_path):
	con = lake.connect(tmp_path / "lake.sqlite")
	d = make_ep(tmp_path, "L1-N")
	assert bronze.register(con, tmp_path, source="t") == {"new": 1, "unchanged": 0, "changed": 0}
	assert bronze.register(con, tmp_path, source="t") == {"new": 0, "unchanged": 1, "changed": 0}
	(d / "events.jsonl").write_text('{"kind": "edited"}\n')
	assert bronze.register(con, tmp_path, source="t") == {"new": 0, "unchanged": 0, "changed": 1}
	rows = con.execute("select count(*) from bronze").fetchone()[0]
	events = con.execute("select path, field from integrity_event").fetchall()
	assert rows == 1  # the original registration is kept, not overwritten
	assert events == [(str(d), "events.jsonl")]


def test_missing_files_are_null(tmp_path):
	con = lake.connect(tmp_path / "lake.sqlite")
	make_ep(tmp_path, "L2-N")
	bronze.register(con, tmp_path, source="t")
	sha = con.execute("select sha_motion from bronze").fetchone()[0]
	assert sha is None


def test_both_markers_deduplicated_and_skips_git_and_non_episode_trees(tmp_path):
	con = lake.connect(tmp_path / "lake.sqlite")
	d = make_ep(tmp_path, "L3-N")
	(d / "episode.json").write_text(json.dumps({"result": {"verdict": "SUCCESS"}}))
	(d / "replay").mkdir()
	(d / "replay" / "motion.npz").write_bytes(b"npz-bytes")
	# .git tree and non-episode trees must be skipped cleanly
	git_ep = tmp_path / ".git" / "objects" / "fake"
	git_ep.mkdir(parents=True)
	(git_ep / "run_status.json").write_text("{}")
	(tmp_path / "reports" / "misc").mkdir(parents=True)
	(tmp_path / "reports" / "misc" / "notes.txt").write_text("hi")

	assert bronze.register(con, tmp_path, source="t") == {"new": 1, "unchanged": 0, "changed": 0}
	sha_ep, sha_motion = con.execute("select sha_episode, sha_motion from bronze").fetchone()
	assert isinstance(sha_ep, str) and len(sha_ep) == 64
	assert isinstance(sha_motion, str) and len(sha_motion) == 64

	# Deleting motion.npz records an integrity event on motion.npz
	(d / "replay" / "motion.npz").unlink()
	assert bronze.register(con, tmp_path, source="t") == {"new": 0, "unchanged": 0, "changed": 1}
	events = con.execute("select path, field, new_sha from integrity_event").fetchall()
	assert events == [(str(d), "motion.npz", None)]


def test_lake_schema_and_empty_or_missing_root(tmp_path):
	con = lake.connect(tmp_path / "db" / "lake.sqlite")
	tables = {r[0] for r in con.execute("select name from sqlite_master where type='table'").fetchall()}
	assert {"bronze", "integrity_event", "silver_episode", "silver_pair", "gold_build"} <= tables
	assert bronze.register(con, tmp_path / "nonexistent", source="t") == {"new": 0, "unchanged": 0, "changed": 0}
