from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest

from digital_twin_armigo.governance import bronze, lake, silver
from digital_twin_armigo.governance.gold import PRODUCTS, GovernanceError, assert_product_invariants, build_product
from digital_twin_armigo.inventory import audit
from digital_twin_armigo.vla.dagger import assert_train_layouts_only

REPO = Path(__file__).resolve().parents[1]


def _load_script(rel: str, name: str):
	spec = importlib.util.spec_from_file_location(name, REPO / rel)
	mod = importlib.util.module_from_spec(spec)
	assert spec and spec.loader
	spec.loader.exec_module(mod)
	return mod


def _make_ep(
	root: Path,
	batch: str,
	group: str,
	cid: str,
	*,
	seed: int = 3000101,
	verdict: str = "SUCCESS",
	state: str = "done",
	events: list[dict] | None = None,
	code_commit: str | None = None,
	oracle: list[str] | tuple[str, ...] | None = None,
	case_opts: dict | None = None,
) -> Path:
	d = root / "runs" / batch / "mujoco" / group / cid
	d.mkdir(parents=True, exist_ok=True)
	(d / "run_status.json").write_text(json.dumps({"state": state}), encoding="utf-8")
	case = [seed, False, 0] if case_opts is None else [seed, False, 0, case_opts]
	(d / "task.json").write_text(json.dumps({"case": case}), encoding="utf-8")
	cam = "fixed" if group.startswith("truth") else group.split("-")[1]
	exp = {
		"batch": batch,
		"group": group,
		"engine": "mujoco",
		"camera_condition": cam,
		"detector": "v7bot",
		"strategy": "B",
		"case_id": cid,
	}
	if code_commit is not None:
		exp["code_commit"] = code_commit
	if oracle:
		exp["oracle"] = list(oracle)
	(d / "episode.json").write_text(
		json.dumps(
			{
				"result": {"label": "normal" if verdict == "SUCCESS" else "unknown", "verdict": verdict},
				"provenance": {"experiment": exp},
			}
		),
		encoding="utf-8",
	)
	(d / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in (events or [])), encoding="utf-8")
	return d


def test_dagger_guard_rejects_v3_val_ood_and_sealed_in_all_input_shapes(tmp_path):
	# Valid v3 train row is accepted both in-memory and in a train-labeled suite file
	good_case = [3000001, False, 0, {"scene": "v3", "split": "train"}]
	good_suite = tmp_path / "good_train.json"
	good_suite.write_text(json.dumps({"split": "train", "sealed": False, "cases": [good_case]}), encoding="utf-8")
	assert_train_layouts_only(cases=[good_case])
	assert_train_layouts_only(suite_path=good_suite)

	bad_cases = [
		("row-val-options", [3100001, True, 0, {"scene": "v3", "split": "val"}]),
		("row-test-seed-relabeled", [3900001, False, 0, {"scene": "v3", "split": "train"}]),
		("row-test-seed-bare", [3900001, False, 0]),
		("row-ood-options-train-seed", [3000001, False, 0, {"scene": "v3", "split": "ood-color"}]),
		("row-ood-appearance-seed", [3500001, False, 0, {"scene": "v3", "split": "ood-appearance"}]),
		("row-sealed-option-flag", [3000002, False, 0, {"scene": "v3", "split": "train", "sealed": True}]),
	]
	for label, case in bad_cases:
		with pytest.raises(ValueError, match="forbidden|non-train|sealed|split"):
			assert_train_layouts_only(cases=[case])

		disguised = tmp_path / f"disguised_{label}.json"
		disguised.write_text(
			json.dumps({"split": "train", "sealed": False, "cases": [good_case, case]}),
			encoding="utf-8",
		)
		with pytest.raises(ValueError, match="forbidden|non-train|sealed|split"):
			assert_train_layouts_only(suite_path=disguised)


def test_missing_or_mismatched_freeze_blocks_student_train_and_eval_test(tmp_path):
	src = tmp_path / "data"
	# Episodes WITHOUT provenance commit
	_make_ep(src, "b1", "truth-B-v7bot", "L3000101-N", seed=3000101, code_commit=None)
	_make_ep(src, "b1", "realistic-fixed-B-v7bot", "L3000101-N", seed=3000101, code_commit=None)
	# Episodes WITH commit "aaa111"
	_make_ep(src, "b1", "truth-B-v7bot", "L3000102-N", seed=3000102, code_commit="aaa111")
	_make_ep(src, "b1", "realistic-fixed-B-v7bot", "L3000102-N", seed=3000102, code_commit="aaa111")
	# Sealed test episode with commit "aaa111"
	_make_ep(src, "b1", "realistic-fixed-B-v7bot", "L3900001-N", seed=3900001, code_commit="aaa111")

	con = lake.connect(tmp_path / "lake.sqlite")
	bronze.register(con, src, source="v3")

	# 1. No freeze -> teacher_frozen must be False, student_train_v3 must be empty, eval_test must be empty
	bid_nofreeze = silver.build_silver(con, freeze=None)
	silver.build_pairs(con, bid_nofreeze)
	rows_nf = [
		json.loads(r[0])
		for r in con.execute("select row_json from silver_episode where build_id = ?", (bid_nofreeze,)).fetchall()
	]
	assert all(r["teacher_frozen"] is False for r in rows_nf)

	out_nf = build_product(
		con, "student_train_v3", silver_build=bid_nofreeze, out_root=tmp_path / "gov_nf", freeze=None
	)
	assert (out_nf / "manifest.jsonl").read_text(encoding="utf-8").strip() == ""

	ev_nf = build_product(con, "eval_test", silver_build=bid_nofreeze, out_root=tmp_path / "gov_nf", freeze=None)
	assert (ev_nf / "manifest.jsonl").read_text(encoding="utf-8").strip() == ""

	# 2. Wrong freeze -> still empty for both student_train_v3 and eval_test
	wrong_freeze = {"code_commit": "bbb222", "detector": "v7bot"}
	bid_wrong = silver.build_silver(con, freeze=wrong_freeze)
	silver.build_pairs(con, bid_wrong)
	out_wrong = build_product(
		con, "student_train_v3", silver_build=bid_wrong, out_root=tmp_path / "gov_w", freeze=wrong_freeze
	)
	assert (out_wrong / "manifest.jsonl").read_text(encoding="utf-8").strip() == ""
	ev_wrong = build_product(con, "eval_test", silver_build=bid_wrong, out_root=tmp_path / "gov_w", freeze=wrong_freeze)
	assert (ev_wrong / "manifest.jsonl").read_text(encoding="utf-8").strip() == ""

	# 3. Matching freeze -> ONLY L3000102-N enters student_train_v3, and L3900001-N enters eval_test
	good_freeze = {"code_commit": "aaa111", "detector": "v7bot"}
	bid_ok = silver.build_silver(con, freeze=good_freeze)
	silver.build_pairs(con, bid_ok)
	out_ok = build_product(
		con, "student_train_v3", silver_build=bid_ok, out_root=tmp_path / "gov_ok", freeze=good_freeze
	)
	train_rows = [
		json.loads(line) for line in (out_ok / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line
	]
	assert len(train_rows) == 1
	assert train_rows[0]["path"].endswith("L3000102-N")

	# Passing mismatched freeze to build_product even with bid_ok must produce 0 rows
	out_cross = build_product(
		con, "student_train_v3", silver_build=bid_ok, out_root=tmp_path / "gov_cross", freeze=wrong_freeze
	)
	assert (out_cross / "manifest.jsonl").read_text(encoding="utf-8").strip() == ""

	ev_ok = build_product(con, "eval_test", silver_build=bid_ok, out_root=tmp_path / "gov_ok", freeze=good_freeze)
	ev_rows = [json.loads(line) for line in (ev_ok / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line]
	assert len(ev_rows) == 1
	assert ev_rows[0]["path"].endswith("L3900001-N")
	con.close()


def test_undeclared_oracle_is_quarantined_and_declared_oracle_is_excluded_from_train(tmp_path):
	src = tmp_path / "data"
	freeze = {"code_commit": "aaa111", "detector": "v7bot"}
	# 1. Undeclared oracle_applied event -> must fail no_truth_fallback AND be quarantined
	ep_undecl = _make_ep(
		src,
		"b1",
		"realistic-fixed-B-v7bot",
		"L3000101-N",
		seed=3000101,
		code_commit="aaa111",
		oracle=None,
		events=[{"kind": "oracle_applied", "parts": ["brick"]}],
	)
	# 2. Declared oracle -> NOT quarantined, but flags.oracle=True so excluded from student_train_v3
	ep_decl = _make_ep(
		src,
		"b1",
		"realistic-fixed-B-v7bot",
		"L3000102-N",
		seed=3000102,
		code_commit="aaa111",
		oracle=["brick"],
		events=[{"kind": "oracle_applied", "parts": ["brick"]}],
	)
	_make_ep(src, "b1", "truth-B-v7bot", "L3000101-N", seed=3000101, code_commit="aaa111")
	_make_ep(src, "b1", "truth-B-v7bot", "L3000102-N", seed=3000102, code_commit="aaa111")

	con = lake.connect(tmp_path / "lake.sqlite")
	bronze.register(con, src, source="v3")
	bid = silver.build_silver(con, freeze=freeze)
	silver.build_pairs(con, bid)

	r_undecl = json.loads(
		con.execute(
			"select row_json from silver_episode where build_id = ? and path = ?", (bid, str(ep_undecl))
		).fetchone()[0]
	)
	assert r_undecl["quarantined"] is True
	assert "no_truth_fallback" in r_undecl["quarantine_reasons"]
	assert r_undecl["checks"]["no_truth_fallback"]["passed"] is False
	assert r_undecl["flags"]["oracle"] is True

	r_decl = json.loads(
		con.execute(
			"select row_json from silver_episode where build_id = ? and path = ?", (bid, str(ep_decl))
		).fetchone()[0]
	)
	assert r_decl["quarantined"] is False
	assert r_decl["checks"]["no_truth_fallback"]["passed"] is True
	assert r_decl["flags"]["oracle"] is True

	out = build_product(con, "student_train_v3", silver_build=bid, out_root=tmp_path / "gov", freeze=freeze)
	assert (out / "manifest.jsonl").read_text(encoding="utf-8").strip() == ""
	con.close()


def test_sealed_seed_blocked_from_harness_review_and_diagnosis_bench(tmp_path):
	src = tmp_path / "data"
	# Quarantined sealed episode under a normal batch name ("round-x")
	ep_sealed = _make_ep(
		src,
		"round-x",
		"realistic-fixed-B-v7bot",
		"L3900001-N",
		seed=3900001,
		state="interrupted",
		verdict="FAILED",
	)
	ep_train = _make_ep(
		src,
		"round-x",
		"realistic-fixed-B-v7bot",
		"L3000101-N",
		seed=3000101,
		state="interrupted",
		verdict="FAILED",
	)

	# 1. harness_review must exclude sealed 3900001, keep normal quarantined 3000101
	con = lake.connect(tmp_path / "lake.sqlite")
	bronze.register(con, src, source="v3")
	bid = silver.build_silver(con)
	silver.build_pairs(con, bid)
	hr = build_product(con, "harness_review", silver_build=bid, out_root=tmp_path / "gov")
	hr_rows = [json.loads(line) for line in (hr / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line]
	assert len(hr_rows) == 1
	assert hr_rows[0]["path"] == str(ep_train)
	assert all(r["split"] != "test" for r in hr_rows)

	with pytest.raises(GovernanceError):
		assert_product_invariants(
			[{"path": str(ep_sealed), "split": "test", "seed": 3900001}], PRODUCTS["harness_review"]
		)
	con.close()

	# 2. H7 make_diagnosis_bench must exclude 3900001 even under batch "round-x"
	inv_dir = tmp_path / "inv"
	inv_dir.mkdir()
	inv_con = sqlite3.connect(inv_dir / "inventory.sqlite")
	inv_con.execute(
		"create table trajectories (traj_id text primary key, path text not null, record_kind text not null)"
	)
	inv_con.execute("insert into trajectories values ('t_sealed', ?, 'agent_episode')", (str(ep_sealed),))
	inv_con.execute("insert into trajectories values ('t_train', ?, 'agent_episode')", (str(ep_train),))
	inv_con.commit()
	inv_con.close()

	audit_db = tmp_path / "audit" / "reviews.sqlite"
	audit.append(
		audit_db,
		"t_sealed",
		{"reviewer": "h", "state": "confirmed", "root_cause": "crop_error"},
		expected_latest_id=None,
		code_version="abc",
	)
	audit.append(
		audit_db,
		"t_train",
		{"reviewer": "h", "state": "confirmed", "root_cause": "crop_error"},
		expected_latest_id=None,
		code_version="abc",
	)

	mdb = _load_script("scripts/experiments/make_diagnosis_bench.py", "mdb_wp2")
	entries = mdb.build_benchmark(audit_db, inv_dir)
	assert [e["traj_id"] for e in entries] == ["t_train"]
