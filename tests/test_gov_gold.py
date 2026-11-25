import json

import pytest

from digital_twin_armigo.governance import bronze, lake, silver
from digital_twin_armigo.governance.gold import PRODUCTS, GovernanceError, assert_product_invariants, build_product


def r(**kw):
	base = {
		"path": "p",
		"batch": "b",
		"split": "train",
		"physical_success": True,
		"teacher_frozen": True,
		"quarantined": False,
		"superseded": False,
		"pair_outcome": "both_pass",
		"pair_key": "k",
		"flags": {"oracle": False, "recovered": False},
	}
	return base | kw


CTX = {"realistic_only_rate": {"b": 0.0}, "void_streak": {}, "freeze": None}


def test_student_train_predicate():
	p = PRODUCTS["student_train_v3"].predicate
	assert p(r(), CTX)
	assert not p(r(pair_outcome="realistic_only"), CTX)
	assert not p(r(split="val"), CTX)
	assert not p(r(flags={"oracle": True}), CTX)
	assert p(r(pair_outcome="unpaired"), CTX)
	assert not p(r(pair_outcome="unpaired"), CTX | {"realistic_only_rate": {"b": 0.02}})
	assert p(r(flags={"oracle": False, "recovered": True}), CTX)  # recovery is a column, not a tier


def test_invariants_reject_leaks_into_trainable_products():
	prod = PRODUCTS["student_train_v3"]
	for bad in (r(split="test"), r(quarantined=True), r(flags={"oracle": True}), r(pair_outcome="realistic_only")):
		with pytest.raises(GovernanceError):
			assert_product_invariants([bad], prod)
	assert_product_invariants([r()], prod)


def test_failure_bank_never_holds_sealed_cases():
	assert not PRODUCTS["failure_bank"].predicate(r(split="test", pair_outcome="truth_only"), CTX)


def _make_ep(root, group, name, *, seed=3000101, verdict="SUCCESS"):
	d = root / "runs" / "b1" / "mujoco" / group / name
	d.mkdir(parents=True)
	(d / "run_status.json").write_text(json.dumps({"state": "done"}), encoding="utf-8")
	(d / "task.json").write_text(json.dumps({"case": [seed, False, 0]}), encoding="utf-8")
	cam = "fixed" if group.startswith("truth") else group.split("-")[1]
	label = "normal" if verdict == "SUCCESS" else "unknown"
	(d / "episode.json").write_text(
		json.dumps(
			{
				"result": {"label": label, "verdict": verdict},
				"provenance": {
					"experiment": {
						"batch": "b1",
						"group": group,
						"engine": "mujoco",
						"camera_condition": cam,
						"detector": "v7bot",
						"strategy": "B",
						"case_id": name,
					}
				},
			}
		),
		encoding="utf-8",
	)
	(d / "events.jsonl").write_text("", encoding="utf-8")
	return d


def test_build_product_is_byte_identical_across_rebuilds(tmp_path):
	src = tmp_path / "data"
	_make_ep(src, "truth-B-v7bot", "L3000101-N", verdict="SUCCESS")
	_make_ep(src, "realistic-fixed-B-v7bot", "L3000101-N", verdict="SUCCESS")
	_make_ep(src, "realistic-drift-B-v7bot", "L3000101-N", verdict="FAILED")
	con = lake.connect(tmp_path / "lake.sqlite")
	bronze.register(con, src, source="t")
	bid = silver.build_silver(con, freeze=None)
	silver.build_pairs(con, bid)

	out1 = build_product(con, "student_train_v3", silver_build=bid, out_root=tmp_path / "gov", freeze=None)
	b1 = (out1 / "manifest.jsonl").read_bytes()
	out2 = build_product(con, "student_train_v3", silver_build=bid, out_root=tmp_path / "gov", freeze=None)
	b2 = (out2 / "manifest.jsonl").read_bytes()
	assert out1 == out2
	assert b1 == b2
	rows = [json.loads(line) for line in b1.decode("utf-8").splitlines() if line]
	assert len(rows) == 1
	assert (out1 / "card.md").is_file()

	fb = build_product(con, "failure_bank", silver_build=bid, out_root=tmp_path / "gov", freeze=None)
	fb_rows = [json.loads(line) for line in (fb / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line]
	assert len(fb_rows) == 1 and fb_rows[0]["pair_outcome"] == "truth_only"


def test_harness_review_handles_list_pair_key_and_void_streak(tmp_path):
	list_pk = ["b1", "mujoco", "v7bot", "B", "open", "hold", "L3000101-N"]
	p = PRODUCTS["harness_review"].predicate
	assert not p(r(pair_key=list_pk), CTX)
	assert p(r(pair_key=list_pk), CTX | {"void_streak": {json.dumps(list_pk): 2}})

	src = tmp_path / "data"
	_make_ep(src, "truth-B-v7bot", "L3000101-N", verdict="SUCCESS")
	_make_ep(src, "realistic-fixed-B-v7bot", "L3000101-N", verdict="SUCCESS")
	q_dir = _make_ep(src, "realistic-fixed-B-v7bot", "L3000102-N", verdict="SUCCESS")
	(q_dir / "run_status.json").unlink()

	con = lake.connect(tmp_path / "lake.sqlite")
	bronze.register(con, src, source="t")
	bid = silver.build_silver(con, freeze=None)
	silver.build_pairs(con, bid)

	hr1 = build_product(con, "harness_review", silver_build=bid, out_root=tmp_path / "gov", freeze=None)
	hr2 = build_product(con, "harness_review", silver_build=bid, out_root=tmp_path / "gov", freeze=None)
	assert hr1 == hr2
	b1 = (hr1 / "manifest.jsonl").read_bytes()
	b2 = (hr2 / "manifest.jsonl").read_bytes()
	assert b1 == b2
	rows = [json.loads(line) for line in b1.decode("utf-8").splitlines() if line]
	assert len(rows) == 1
	assert rows[0]["path"].endswith("L3000102-N")
