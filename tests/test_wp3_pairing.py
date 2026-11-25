from __future__ import annotations

import json
from pathlib import Path

import pytest

from digital_twin_armigo.governance import bronze, gold, lake, silver
from digital_twin_armigo.governance.gold import GovernanceError, build_product


def _make_ep(
	root: Path,
	batch: str,
	group: str,
	cid: str,
	*,
	seed: int = 3000101,
	verdict: str = "SUCCESS",
	code_commit: str = "aaa111",
	suite: str = "v3_train",
	suite_sha256: str = "sha_v3_train",
	segmenter: str = "lab",
	oracle: list[str] | None = None,
) -> Path:
	d = root / "runs" / batch / "mujoco" / group / cid
	d.mkdir(parents=True, exist_ok=True)
	(d / "run_status.json").write_text(json.dumps({"state": "done"}), encoding="utf-8")
	(d / "task.json").write_text(json.dumps({"case": [seed, False, 0]}), encoding="utf-8")
	cam = "fixed" if group.startswith("truth") else group.split("-")[1]
	exp = {
		"batch": batch,
		"group": group,
		"engine": "mujoco",
		"camera_condition": cam,
		"detector": "v7bot",
		"strategy": "B",
		"case_id": cid,
		"code_commit": code_commit,
		"suite": suite,
		"suite_sha256": suite_sha256,
		"segmenter": segmenter,
	}
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
	(d / "events.jsonl").write_text("", encoding="utf-8")
	return d


def test_two_batch_realistic_only_is_not_masked_as_unpaired_or_exported(tmp_path):
	"""Audit counterexample: Batch A has truth=FAIL, realistic=PASS; Batch B also runs the same case.
	Batch A realistic MUST be classified realistic_only and MUST NEVER enter student_train_v3."""
	src = tmp_path / "data"
	freeze = {"code_commit": "aaa111", "detector": "v7bot"}

	# Batch A: truth fails, realistic passes (same case L3000101-N) -> realistic_only!
	# Also add one clean both_pass pair in Batch A and one unpaired realistic in Batch A to verify rate gate (>=1%).
	ep_a_t = _make_ep(src, "batch-a", "truth-B-v7bot", "L3000101-N", seed=3000101, verdict="FAILED")
	ep_a_r = _make_ep(src, "batch-a", "realistic-fixed-B-v7bot", "L3000101-N", seed=3000101, verdict="SUCCESS")
	_make_ep(src, "batch-a", "realistic-fixed-B-v7bot", "L3000102-N", seed=3000102, verdict="SUCCESS")

	# Batch B: both pass for the exact same case L3000101-N (previously overwrote Batch A in silver_pair!)
	ep_b_t = _make_ep(src, "batch-b", "truth-B-v7bot", "L3000101-N", seed=3000101, verdict="SUCCESS")
	ep_b_r = _make_ep(src, "batch-b", "realistic-fixed-B-v7bot", "L3000101-N", seed=3000101, verdict="SUCCESS")

	con = lake.connect(tmp_path / "lake.sqlite")
	bronze.register(con, src, source="v3")
	bid = silver.build_silver(con, freeze=freeze)
	silver.build_pairs(con, bid)

	pairs = con.execute(
		"select truth_path, realistic_path, outcome, match_scope from silver_pair where build_id = ?", (bid,)
	).fetchall()
	by_real = {r_path: (t_path, outcome, scope) for t_path, r_path, outcome, scope in pairs if r_path}

	# Batch A realistic must be paired against Batch A truth -> realistic_only (same_batch)
	assert by_real[str(ep_a_r)] == (str(ep_a_t), "realistic_only", "same_batch")
	# Batch B realistic must be paired against Batch B truth -> both_pass (same_batch)
	assert by_real[str(ep_b_r)] == (str(ep_b_t), "both_pass", "same_batch")

	# Gold student_train_v3: Batch A's realistic_only AND Batch A's unpaired (rate=100% >= 1%) must be excluded;
	# only Batch B's both_pass enters.
	out1 = build_product(con, "student_train_v3", silver_build=bid, out_root=tmp_path / "gov", freeze=freeze)
	out2 = build_product(con, "student_train_v3", silver_build=bid, out_root=tmp_path / "gov", freeze=freeze)
	assert out1 == out2
	assert (out1 / "manifest.jsonl").read_bytes() == (out2 / "manifest.jsonl").read_bytes()

	rows = [json.loads(line) for line in (out1 / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line]
	assert [r["path"] for r in rows] == [str(ep_b_r)]
	assert all("batch-a" not in r["path"] for r in rows)

	# Every latest episode must be represented in reconciliation with zero orphans
	rec = silver.reconcile_silver(con, bid)
	assert rec["unrepresented_latest"] == 0
	assert rec["total_silver_episodes"] == 5
	con.close()


def test_cross_batch_truth_reuse_requires_explicit_compatibility(tmp_path):
	"""Cross-batch truth reuse is only allowed when suite_sha256, code_commit, detector, segmenter, and options match."""
	src = tmp_path / "data"
	freeze = {"code_commit": "aaa111", "detector": "v7bot"}

	# Historical truth in batch-t
	ep_t = _make_ep(
		src,
		"batch-t",
		"truth-B-v7bot",
		"L3000101-N",
		seed=3000101,
		verdict="SUCCESS",
		code_commit="aaa111",
		suite_sha256="sha_A",
		segmenter="lab",
	)
	# Compatible realistic in batch-r-compat (same suite_sha256, commit, segmenter) -> cross_batch both_pass
	ep_r_ok = _make_ep(
		src,
		"batch-r-compat",
		"realistic-fixed-B-v7bot",
		"L3000101-N",
		seed=3000101,
		verdict="SUCCESS",
		code_commit="aaa111",
		suite_sha256="sha_A",
		segmenter="lab",
	)
	# Incompatible realistic in batch-r-diffsuite (different suite_sha256) -> unpaired
	ep_r_suite = _make_ep(
		src,
		"batch-r-diffsuite",
		"realistic-fixed-B-v7bot",
		"L3000101-N",
		seed=3000101,
		verdict="SUCCESS",
		code_commit="aaa111",
		suite_sha256="sha_B",
		segmenter="lab",
	)
	# Incompatible realistic in batch-r-diffcommit (different code_commit) -> unpaired
	ep_r_commit = _make_ep(
		src,
		"batch-r-diffcommit",
		"realistic-fixed-B-v7bot",
		"L3000101-N",
		seed=3000101,
		verdict="SUCCESS",
		code_commit="bbb222",
		suite_sha256="sha_A",
		segmenter="lab",
	)
	# Incompatible realistic in batch-r-diffseg (different segmenter) -> unpaired
	ep_r_seg = _make_ep(
		src,
		"batch-r-diffseg",
		"realistic-fixed-B-v7bot",
		"L3000101-N",
		seed=3000101,
		verdict="SUCCESS",
		code_commit="aaa111",
		suite_sha256="sha_A",
		segmenter="unet",
	)

	con = lake.connect(tmp_path / "lake.sqlite")
	bronze.register(con, src, source="v3")
	bid = silver.build_silver(con, freeze=freeze)
	silver.build_pairs(con, bid)

	rows = con.execute(
		"select truth_path, realistic_path, outcome, match_scope from silver_pair where build_id = ?", (bid,)
	).fetchall()
	by_real = {r_path: (t_path, outcome, scope) for t_path, r_path, outcome, scope in rows if r_path}

	assert by_real[str(ep_r_ok)] == (str(ep_t), "both_pass", "cross_batch")
	assert by_real[str(ep_r_suite)] == (None, "unpaired", "unpaired")
	assert by_real[str(ep_r_commit)] == (None, "unpaired", "unpaired")
	assert by_real[str(ep_r_seg)] == (None, "unpaired", "unpaired")
	assert silver.reconcile_silver(con, bid)["unrepresented_latest"] == 0
	con.close()


def test_gold_refuses_missing_silver_pair_instead_of_defaulting_to_unpaired(tmp_path):
	"""If a latest realistic episode is absent from silver_pair, gold MUST raise GovernanceError (never default to unpaired)."""
	src = tmp_path / "data"
	freeze = {"code_commit": "aaa111", "detector": "v7bot"}
	ep_r = _make_ep(src, "batch-a", "realistic-fixed-B-v7bot", "L3000101-N", seed=3000101, verdict="SUCCESS")
	_make_ep(src, "batch-a", "truth-B-v7bot", "L3000101-N", seed=3000101, verdict="SUCCESS")

	con = lake.connect(tmp_path / "lake.sqlite")
	bronze.register(con, src, source="v3")
	bid = silver.build_silver(con, freeze=freeze)
	# Deliberately do NOT call silver.build_pairs (or delete the realistic's pair row)
	with pytest.raises(GovernanceError, match="silver_pair|missing"):
		gold.build_product(con, "student_train_v3", silver_build=bid, out_root=tmp_path / "gov", freeze=freeze)

	silver.build_pairs(con, bid)
	con.execute("delete from silver_pair where build_id = ? and realistic_path = ?", (bid, str(ep_r)))
	con.commit()
	with pytest.raises(GovernanceError, match="silver_pair|missing"):
		gold.build_product(con, "student_train_v3", silver_build=bid, out_root=tmp_path / "gov", freeze=freeze)
	con.close()


def test_truth_pairs_with_drift_and_recal_ignoring_cameras_option(tmp_path):
	"""Truth runs fixed while realistic drift/recal records cameras in task.case[3]; same-batch pairing must succeed."""
	src = tmp_path / "data"
	freeze = {"code_commit": "aaa111", "detector": "v7bot"}
	ep_t = _make_ep(src, "batch-a", "truth-B-v7bot", "L3000101-N", seed=3000101, verdict="SUCCESS")
	ep_r_drift = _make_ep(src, "batch-a", "realistic-drift-B-v7bot", "L3000101-N", seed=3000101, verdict="FAILED")
	(ep_t / "task.json").write_text(json.dumps({"case": [3000101, False, 0, {"scene": "v3", "split": "train"}]}))
	(ep_r_drift / "task.json").write_text(
		json.dumps({"case": [3000101, False, 0, {"scene": "v3", "split": "train", "cameras": "drift"}]})
	)

	con = lake.connect(tmp_path / "lake.sqlite")
	bronze.register(con, src, source="v3")
	bid = silver.build_silver(con, freeze=freeze)
	silver.build_pairs(con, bid)

	row = con.execute(
		"select truth_path, outcome, match_scope from silver_pair where build_id = ? and realistic_path = ?",
		(bid, str(ep_r_drift)),
	).fetchone()
	assert row == (str(ep_t), "truth_only", "same_batch")
	con.close()
