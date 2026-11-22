"""Inventory of synthetic trajectories and scenes: provenance categories, adapters, overrides, tables."""

from __future__ import annotations

import csv
import json
import os
import sqlite3

import pytest

from digital_twin_armigo.inventory import cli, scan, store, taxonomy

OLD = 1_700_000_000.0  # before realistic sensing existed
NEW = taxonomy.SENSING_MODE_INTRODUCED + 1000


def _result(verdict="SUCCESS", consulted=0):
	return {
		"verdict": verdict,
		"sim_frames": 300,
		"views": 3,
		"rewinds": 0,
		"retries": 0,
		"llm": {"consulted": consulted},
	}


# -- taxonomy -------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
	("prov", "result", "written", "category", "control", "sensing", "basis"),
	[
		({"engine": "mujoco", "sensing": "truth"}, _result(), NEW, "agent_omniscient", "agent_rule", "truth", "inferred"),
		({"engine": "mujoco", "sensing": "realistic"}, _result(consulted=2), NEW, "agent_realistic", "agent_llm", "realistic", "inferred"),
		({"engine": "mujoco"}, _result(), OLD, "agent_omniscient", "agent_rule", "truth", "inferred"),  # predates the field
		({"engine": "mujoco"}, _result(), NEW, "agent_unknown", "agent_rule", "unknown", "inferred"),
		({"sensing": "realistic", "student": True, "case": [1, False, 0]}, _result(), NEW, "model", "model_policy", "realistic", "inferred"),
		({"sensing": "realistic", "student": True, "dagger_takeover": True}, _result(), NEW, "model_takeover", "model_policy", "realistic", "inferred"),
		({"sensing": "realistic", "student": False}, _result(), NEW, "scripted", "script_expert_tracer", "realistic", "inferred"),
		({"sensing": "truth", "actor": {"category": "model", "control": "x", "sensing": "truth"}}, _result(), NEW, "model", "x", "truth", "explicit"),
	],
)  # fmt: skip
def test_classification_rules(prov, result, written, category, control, sensing, basis):
	got = taxonomy.classify_provenance(prov, result, written_at=written)
	assert (got["category"], got["control"], got["sensing"], got["basis"]) == (category, control, sensing, basis)


def test_catalog_records_are_scripted_with_privileged_state():
	got = taxonomy.classify_catalog("isaac")
	assert got["category"] == "scripted" and got["sensing"] == "privileged" and got["control"].endswith("isaac")


# -- fixtures -------------------------------------------------------------------------------------------------------------
def _episode(directory, provenance, result=None, *, case=(7, True, 2), extra=None, age=NEW):
	directory.mkdir(parents=True)
	body = {
		"schema": scan.AGENT_SCHEMA,
		"fps": 30,
		"branches": [{"id": 0, "start_frame": 0, "actions": 300}],
		"video": False,
		"provenance": {**provenance, "case": list(case)} if case else provenance,
		"result": result or _result(),
		**(extra or {}),
	}
	f = directory / "episode.json"
	f.write_text(json.dumps(body), encoding="utf-8")
	(directory / "events.jsonl").write_text(
		json.dumps({"seq": 0, "kind": "node", "wall_time_s": 1.5})
		+ "\n"
		+ json.dumps({"seq": 1, "kind": "node", "wall_time_s": 4.25})
		+ "\n",
		encoding="utf-8",
	)
	(directory / "branch-0").mkdir()
	(directory / "branch-0" / "actions.npz").write_bytes(b"x")
	(directory / "keyframes").mkdir()
	os.utime(f, (age, age))
	return f


def _catalog_db(path):
	con = sqlite3.connect(path)
	con.executescript(
		"""create table sources(id text primary key, path text, sha256 text, imported_at text);
		create table episodes(id text primary key, source_id text, locator text, alias text, engine text, verdict text,
			condition text, quality_status text, layout_group text, role text, raw text, provenance text);
		create table artifacts(id text primary key, path text, sha256 text, size integer, role text);
		create table episode_artifacts(episode_id text, artifact_id text);
		create table datasets(id text primary key, name text, seed integer, created_at text, manifest text);
		create table memberships(dataset_id text, episode_id text, layout_group text, split text);"""
	)
	layout = {"brick_size": [0.03, 0.03, 0.02], "brick_xy": [0.1, 0.2], "defective": False, "n_clutter": 4}
	con.execute("insert into sources values ('s1', 'results/physics-expert-v2/a.json', 'h', 't')")
	for i, engine in enumerate(("mujoco", "isaac")):
		con.execute(
			"insert into episodes values (?, 's1', ?, ?, ?, 'SUCCESS', null, 'PASS', null, 'development', ?, ?)",
			(
				f"e{i}" * 8,
				f"#{i}",
				f"alias{i}",
				engine,
				json.dumps({"layout": layout, "info": {"n_frames": 840}}),
				json.dumps({"layout_seed": 1000}),
			),
		)
	con.execute("insert into artifacts values ('a1', 'm.npz', 'h', 1, 'motion')")
	con.execute("insert into episode_artifacts values ('e0e0e0e0e0e0e0e0', 'a1')")
	con.execute("insert into datasets values ('d1', 'dataset-v1', 0, 't', '{}')")
	con.execute("insert into memberships values ('d1', 'e0e0e0e0e0e0e0e0', 'g', 'train')")
	con.commit()
	con.close()


@pytest.fixture
def tree(tmp_path):
	root = tmp_path / "data"
	_episode(root / "agent/run-a/case-0000", {"engine": "mujoco", "sensing": "truth"})
	_episode(root / "agent/run-a/case-0001", {"engine": "mujoco", "sensing": "truth"}, case=(8, False, 1))
	_episode(
		root / "agent/run-b/case-0000", {"engine": "mujoco", "sensing": "realistic"}, _result("FAILURE", consulted=1)
	)
	_episode(
		root / "roll/student/case-0000",
		{"sensing": "realistic", "student": True},
		_result("FAILED"),
		case=(9, False, 0),
	)
	_episode(root / "roll/takeover/case-0000", {"sensing": "realistic", "student": True, "dagger_takeover": True},
		extra={"takeover": {"triggered": True, "frame": 120, "gate": "joint_deviation"}})  # fmt: skip
	_episode(root / "old/case-0000", {"engine": "mujoco"}, age=OLD)
	_episode(
		root / "old/case-0001", {"engine": "mujoco"}, case=(11, False, 3), age=NEW
	)  # same content as case-0000 modulo nothing: unknown sensing
	# identical bytes to agent/run-a/case-0000, as a copy would be
	copy = root / "copy/case-0000"
	copy.mkdir(parents=True)
	(copy / "episode.json").write_bytes((root / "agent/run-a/case-0000/episode.json").read_bytes())
	# not an agent episode: ignored
	(root / "other").mkdir()
	(root / "other/episode.json").write_text(json.dumps({"schema": "something-else"}), encoding="utf-8")
	# a render dataset and a LeRobot export
	ds = root / "anomaly/dataset-x"
	ds.mkdir(parents=True)
	(ds / "layouts.jsonl").write_text(
		json.dumps({"layout_index": 0, "split": "train", "defective": True, "defect_kind": "cross_var", "defect_face": 2, "defect_seed": 5, "brick_size": [1, 1, 1], "status": "ok"})
		+ "\n"
		+ json.dumps({"layout_index": 1, "split": "test", "defective": False, "status": "ok"})
		+ "\n",
		encoding="utf-8",
	)  # fmt: skip
	(ds / "metadata.jsonl").write_text("", encoding="utf-8")
	(ds / "summary.json").write_text(json.dumps({"n_crops": 12}), encoding="utf-8")
	lr = root / "lerobot-x"
	(lr / "meta").mkdir(parents=True)
	(lr / "data").mkdir()
	(lr / "meta/info.json").write_text(
		json.dumps({"total_episodes": 3, "total_frames": 900, "fps": 30}), encoding="utf-8"
	)
	_catalog_db(root / "catalog.sqlite")
	return root


def _rows(out, sql, *params):
	con = sqlite3.connect(out / "inventory.sqlite")
	con.row_factory = sqlite3.Row
	try:
		return [dict(r) for r in con.execute(sql, params)]
	finally:
		con.close()


# -- scan -----------------------------------------------------------------------------------------------------------------
def test_scan_categorises_every_source_and_writes_the_tables(tree, tmp_path):
	out = tmp_path / "inv"
	info = store.build([("d", tree)], out, overrides=tmp_path / "none.json")
	assert info["trajectories"] == 8 + 2  # 8 agent episode dirs (incl. the byte copy) + 2 catalog rows
	by_path = {
		r["path"].replace("\\", "/").split("/data/")[-1]: r
		for r in _rows(out, "select * from trajectories where record_kind='agent_episode'")
	}
	assert by_path["agent/run-a/case-0000"]["category"] == "agent_omniscient"
	assert (
		by_path["agent/run-b/case-0000"]["category"] == "agent_realistic"
		and by_path["agent/run-b/case-0000"]["llm_consulted"] == 1
	)
	assert by_path["roll/student/case-0000"]["category"] == "model"
	take = by_path["roll/takeover/case-0000"]
	assert (
		take["category"] == "model_takeover"
		and take["takeover_gate"] == "joint_deviation"
		and take["takeover_frame"] == 120
	)
	assert by_path["old/case-0000"]["category"] == "agent_omniscient"  # predates the sensing field
	assert by_path["old/case-0001"]["category"] == "agent_unknown"
	run = by_path["agent/run-a/case-0000"]
	assert run["scene_id"] == "case:7:1:2" and run["defective"] == 1 and run["defect_face"] == 2 and run["seed"] == 7
	assert (
		run["sim_s"] == 10.0
		and run["wall_s"] == 4.25
		and run["has_actions"] == 1
		and run["has_keyframes"] == 1
		and run["has_motion"] == 0
	)
	assert by_path["copy/case-0000"]["same_as"] == run["traj_id"] and info["duplicates"] == 1
	assert "other" not in " ".join(by_path)

	catalog = _rows(out, "select * from trajectories where record_kind='catalog_episode' order by engine")
	assert [r["category"] for r in catalog] == ["scripted", "scripted"]
	isaac, mujoco = catalog
	assert (
		mujoco["has_motion"] == 1
		and isaac["has_motion"] == 0
		and mujoco["sim_frames"] == 840
		and mujoco["split_hint"] == "dataset-v1:train"
	)
	assert (
		mujoco["scene_id"].startswith("layout:") and mujoco["scene_id"] == isaac["scene_id"]
	)  # same geometry, same scene


def test_scenes_aggregate_cases_layouts_and_render_datasets(tree, tmp_path):
	out = tmp_path / "inv"
	store.build([("d", tree)], out, overrides=tmp_path / "none.json")
	scenes = {r["scene_id"]: r for r in _rows(out, "select * from scenes")}
	case = scenes["case:7:1:2"]
	assert case["scene_kind"] == "case" and case["n_trajectories"] == 5  # run-a, run-b, takeover, old/0 and the copy
	assert json.loads(case["categories"]) == {"agent_omniscient": 3, "agent_realistic": 1, "model_takeover": 1}
	layout = next(s for s in scenes.values() if s["scene_kind"] == "layout")
	assert layout["n_trajectories"] == 2 and json.loads(layout["layout"])["n_clutter"] == 4
	render = [s for s in scenes.values() if s["scene_kind"] == "render_layout"]
	assert (
		len(render) == 2
		and {s["split"] for s in render} == {"train", "test"}
		and all(s["dataset_id"].endswith("anomaly/dataset-x") for s in render)
	)
	datasets = {d["kind"]: d for d in _rows(out, "select * from datasets")}
	assert datasets["render_dataset"]["n_scenes"] == 2 and datasets["render_dataset"]["n_items"] == 12
	assert datasets["lerobot_export"]["n_items"] == 3


def test_csv_exports_have_the_same_rows_and_open_in_excel(tree, tmp_path):
	out = tmp_path / "inv"
	info = store.build([("d", tree)], out, overrides=tmp_path / "none.json")
	raw = (out / "trajectories.csv").read_bytes()
	assert raw.startswith(b"\xef\xbb\xbf")  # BOM so Excel reads UTF-8
	with open(out / "trajectories.csv", encoding="utf-8-sig", newline="") as f:
		rows = list(csv.DictReader(f))
	assert len(rows) == info["trajectories"] and set(rows[0]) == set(store.TRAJECTORY_COLUMNS)
	summary = (out / "SUMMARY.md").read_text(encoding="utf-8")
	assert "纯脚本" in summary and "全知 agent" in summary and "需人工确认" in summary


def test_overrides_relabel_by_path_glob_and_are_marked(tree, tmp_path):
	rules = tmp_path / "rules.json"
	rules.write_text(
		json.dumps(
			[
				{"match": "*/old/case-0001", "set": {"category": "agent_realistic", "sensing": "realistic", "note": "from the run log", "tags": ["manual"]}},
			]
		),
		encoding="utf-8",
	)  # fmt: skip
	out = tmp_path / "inv"
	info = store.build([("d", tree)], out, overrides=rules)
	assert info["overrides_applied"] == 1
	row = _rows(out, "select * from trajectories where path like '%old%case-0001'")[0]
	assert (row["category"], row["sensing"], row["category_basis"], row["category_note"]) == (
		"agent_realistic",
		"realistic",
		"override",
		"from the run log",
	)
	assert json.loads(row["tags"]) == ["manual"]
	rules.write_text(json.dumps([{"match": "*", "set": {"category": "nonsense"}}]), encoding="utf-8")
	with pytest.raises(ValueError, match="unknown category"):
		store.build([("d", tree)], out, overrides=rules)


def test_scan_never_modifies_the_scanned_data(tree, tmp_path):
	before = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in tree.rglob("*") if p.is_file()}
	store.build([("d", tree)], tmp_path / "inv", overrides=tmp_path / "none.json")
	assert before == {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in tree.rglob("*") if p.is_file()}


def test_missing_root_is_an_error(tmp_path):
	with pytest.raises(FileNotFoundError):
		store.build([("x", tmp_path / "nope")], tmp_path / "inv")


# -- cli and freezing ---------------------------------------------------------------------------------------------------------
def test_cli_scan_summary_show_label(tree, tmp_path, capsys):
	out, rules = tmp_path / "inv", tmp_path / "rules.json"
	assert cli.main(["--out", str(out), "scan", "--root", f"d={tree}", "--overrides", str(rules)]) == 0
	capsys.readouterr()
	assert cli.main(["--out", str(out), "summary"]) == 0
	text = capsys.readouterr().out
	assert "agent_omniscient" in text and "scripted" in text and "scenes:" in text
	assert cli.main(["--out", str(out), "show", "old/case-0001"]) == 0
	assert "agent_unknown" in capsys.readouterr().out
	assert (
		cli.main(
			[
				"--out",
				str(out),
				"label",
				"--match",
				"*/old/case-0001",
				"--category",
				"agent_realistic",
				"--overrides",
				str(rules),
			]
		)
		== 0
	)
	assert cli.main(["--out", str(out), "label", "--match", "*", "--category", "bogus", "--overrides", str(rules)]) == 2
	capsys.readouterr()
	assert cli.main(["--out", str(out), "scan", "--root", f"d={tree}", "--overrides", str(rules)]) == 0
	assert json.loads(capsys.readouterr().out)["overrides_applied"] == 1


def test_trace_finish_freezes_the_actor_into_provenance(tmp_path):
	from digital_twin_armigo.agent.twin_record import EpisodeTrace

	trace = EpisodeTrace(tmp_path / "ep", video=False)
	trace.finish(_result("SUCCESS", consulted=3), {"engine": "mujoco", "sensing": "realistic", "case": [1, False, 0]})
	ep = json.loads((tmp_path / "ep" / "episode.json").read_text(encoding="utf-8"))
	actor = ep["provenance"]["actor"]
	assert (actor["category"], actor["control"], actor["sensing"]) == ("agent_realistic", "agent_llm", "realistic")
	assert "basis" not in actor and actor["rules_version"] == taxonomy.RULES_VERSION
	row = scan.agent_episode(tmp_path / "ep", "t", tmp_path)
	assert row["category"] == "agent_realistic" and row["category_basis"] == "explicit"
