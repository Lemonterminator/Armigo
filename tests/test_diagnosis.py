import importlib.util
import json
import math
import sqlite3
from pathlib import Path

from digital_twin_armigo.experiments.diagnosis import agreement, cohen_kappa, split_of
from digital_twin_armigo.inventory import audit, store


def _load_script(name: str):
	path = Path(__file__).resolve().parents[1] / "scripts/experiments" / f"{name}.py"
	spec = importlib.util.spec_from_file_location(name, path)
	assert spec is not None and spec.loader is not None
	mod = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(mod)
	return mod


make_diagnosis_bench = _load_script("make_diagnosis_bench")
round_metrics = _load_script("round_metrics")


def test_split_is_deterministic_and_roughly_balanced():
	ids = [f"traj-{i}" for i in range(1000)]
	splits = [split_of(t) for t in ids]
	assert splits == [split_of(t) for t in ids]
	assert 400 < splits.count("calib") < 600


def test_cohen_kappa_perfect_and_chance():
	assert cohen_kappa(["a", "b", "a"], ["a", "b", "a"]) == 1.0
	assert abs(cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"])) < 1e-9


def test_agreement_uses_only_the_requested_split_and_labelled_mechanisms():
	bench = [
		{"traj_id": "x", "root_cause": "recovery_failure", "mechanism": "brick_pose_error", "split": "heldout"},
		{"traj_id": "y", "root_cause": "evidence_insufficient", "mechanism": None, "split": "heldout"},
		{"traj_id": "z", "root_cause": "occlusion", "mechanism": None, "split": "calib"},
	]
	auto = {
		"x": {"root_cause": "recovery_failure", "mechanism": "brick_pose_error"},
		"y": {"root_cause": "premature_stop", "mechanism": "brick_yaw_error"},
		"z": {"root_cause": "occlusion", "mechanism": None},
	}
	r = agreement(bench, auto, "heldout")
	assert r["n"] == 2 and r["root_cause_acc"] == 0.5 and r["mechanism_n"] == 1 and r["mechanism_acc"] == 1.0


def _create_inventory_db(inv_dir: Path, rows: list[dict], *, with_auto_cols: bool = True) -> None:
	inv_dir.mkdir(parents=True, exist_ok=True)
	db = inv_dir / "inventory.sqlite"
	auto_cols = ("auto_root_cause", "auto_mechanism", "auto_upstream")
	cols = [c for c in store.TRAJECTORY_COLUMNS if c != "traj_id" and (with_auto_cols or c not in auto_cols)]
	if with_auto_cols:
		for c in auto_cols:
			if c not in cols:
				cols.append(c)
	cols_sql = ", ".join(f'"{c}" text' for c in cols)
	con = sqlite3.connect(db)
	con.execute(f"create table trajectories (traj_id text primary key, {cols_sql})")
	for r in rows:
		keys = list(r.keys())
		placeholders = ", ".join("?" * len(keys))
		quoted = ", ".join(f'"{k}"' for k in keys)
		con.execute(f"insert into trajectories ({quoted}) values ({placeholders})", [r[k] for k in keys])
	con.commit()
	con.close()


def _create_audit_db(db_path: Path, reviews: list[dict]) -> None:
	with audit.connect(db_path) as con:
		cols = {r[1] for r in con.execute("pragma table_info(reviews)").fetchall()}
		if "mechanism" not in cols:
			con.execute("alter table reviews add column mechanism text")
		for idx, rev in enumerate(reviews, start=1):
			con.execute(
				"insert into reviews (traj_id, created_at, reviewer, state, root_cause, mechanism, physical_outcome, note, refs) "
				"values (?, ?, ?, ?, ?, ?, ?, ?, ?)",
				(
					rev["traj_id"],
					f"2026-11-24T00:00:{idx:02d}+00:00",
					"reviewer",
					rev["state"],
					rev.get("root_cause"),
					rev.get("mechanism"),
					rev.get("physical_outcome"),
					None,
					"[]",
				),
			)


def test_make_diagnosis_bench_filters_and_excludes_sealed(tmp_path):
	audit_db = tmp_path / "audit" / "reviews.sqlite"
	inv_dir = tmp_path / "inventory"

	ep_keep_b = tmp_path / "runs" / "p2-dev-r2" / "mujoco" / "g" / "L1-N"
	ep_keep_a = tmp_path / "runs" / "p2-dev-r2" / "mujoco" / "g" / "L2-D0"
	ep_sealed = tmp_path / "runs" / "p2_sealed_eval" / "mujoco" / "g" / "L3-N"
	ep_disputed = tmp_path / "runs" / "p2-dev-r2" / "mujoco" / "g" / "L4-N"
	ep_empty_rc = tmp_path / "runs" / "p2-dev-r2" / "mujoco" / "g" / "L5-N"

	_create_inventory_db(
		inv_dir,
		[
			{"traj_id": "t_keep_b", "record_kind": "agent_episode", "path": str(ep_keep_b)},
			{"traj_id": "t_keep_a", "record_kind": "agent_episode", "path": str(ep_keep_a)},
			{"traj_id": "t_sealed", "record_kind": "agent_episode", "path": str(ep_sealed)},
			{"traj_id": "t_disputed_latest", "record_kind": "agent_episode", "path": str(ep_disputed)},
			{"traj_id": "t_empty_rc", "record_kind": "agent_episode", "path": str(ep_empty_rc)},
		],
	)
	_create_audit_db(
		audit_db,
		[
			# Earlier disputed, then latest confirmed -> kept
			{"traj_id": "t_keep_b", "state": "disputed", "root_cause": "other"},
			{
				"traj_id": "t_keep_b",
				"state": "confirmed",
				"root_cause": "recovery_failure",
				"mechanism": "brick_pose_error",
			},
			{"traj_id": "t_keep_a", "state": "confirmed", "root_cause": "occlusion", "mechanism": None},
			# Sealed batch -> excluded
			{"traj_id": "t_sealed", "state": "confirmed", "root_cause": "crop_error"},
			# Latest review is disputed (even though earlier was confirmed) -> excluded
			{"traj_id": "t_disputed_latest", "state": "confirmed", "root_cause": "crop_error"},
			{"traj_id": "t_disputed_latest", "state": "disputed", "root_cause": "crop_error"},
			# Empty root_cause -> excluded
			{"traj_id": "t_empty_rc", "state": "confirmed", "root_cause": None},
			# Missing from inventory -> skipped cleanly
			{"traj_id": "t_missing_inv", "state": "confirmed", "root_cause": "crop_error"},
		],
	)

	out_file = tmp_path / "bench" / "v1.json"
	rc = make_diagnosis_bench.main(["--audit", str(audit_db), "--inventory", str(inv_dir), "--out", str(out_file)])
	assert rc == 0

	data = json.loads(out_file.read_text(encoding="utf-8"))
	assert [d["traj_id"] for d in data] == ["t_keep_a", "t_keep_b"]
	assert data[0] == {
		"traj_id": "t_keep_a",
		"path": str(ep_keep_a),
		"root_cause": "occlusion",
		"mechanism": None,
		"split": split_of("t_keep_a"),
	}
	assert data[1] == {
		"traj_id": "t_keep_b",
		"path": str(ep_keep_b),
		"root_cause": "recovery_failure",
		"mechanism": "brick_pose_error",
		"split": split_of("t_keep_b"),
	}


def _find_ids_for_split(target_split: str, count: int) -> list[str]:
	found: list[str] = []
	i = 0
	while len(found) < count:
		tid = f"traj-{i}"
		if split_of(tid) == target_split:
			found.append(tid)
		i += 1
	return found


def test_round_metrics_counterfactual_does_not_leak_across_drift_arms(tmp_path):
	base = {
		"case_id": "L7-N",
		"root_cause": "recovery_failure",
		"group": "realistic-drift-B-v7bot",
		"path": str(tmp_path / "top"),
	}
	vote = {**base, "group": "realistic-drift-B-v7bot-loc-vote", "path": str(tmp_path / "vote")}
	path = tmp_path / "counterfactual.jsonl"
	path.write_text(json.dumps({**vote, "verdict": "confirmed"}) + "\n", encoding="utf-8")
	coverage = round_metrics._attribution_coverage([base, vote], path)
	assert coverage["counterfactual_confirmed_n"] == 1
	assert coverage["counterfactual_confirmed_rate"] == 0.5


def test_round_metrics_computes_all_six_groups_and_appends_rounds_md(tmp_path):
	calib_ids = _find_ids_for_split("calib", 2)
	heldout_ids = _find_ids_for_split("heldout", 2)

	ep_c0 = tmp_path / "runs" / "r1" / "mujoco" / "g" / "L1-N"
	ep_c1 = tmp_path / "runs" / "r1" / "mujoco" / "g" / "L1-D0"
	ep_h0 = tmp_path / "runs" / "r1" / "mujoco" / "g" / "L2-N"
	ep_h1 = tmp_path / "runs" / "r1" / "mujoco" / "g" / "L2-D1"

	bench = [
		{
			"traj_id": calib_ids[0],
			"path": str(ep_c0),
			"root_cause": "recovery_failure",
			"mechanism": "brick_pose_error",
			"split": "calib",
		},
		{
			"traj_id": calib_ids[1],
			"path": str(ep_c1),
			"root_cause": "occlusion",
			"mechanism": None,
			"split": "calib",
		},
		{
			"traj_id": heldout_ids[0],
			"path": str(ep_h0),
			"root_cause": "recovery_failure",
			"mechanism": "brick_pose_error",
			"split": "heldout",
		},
		{
			"traj_id": heldout_ids[1],
			"path": str(ep_h1),
			"root_cause": "evidence_insufficient",
			"mechanism": "brick_yaw_error",
			"split": "heldout",
		},
	]
	bench_path = tmp_path / "bench.json"
	bench_path.write_text(json.dumps(bench), encoding="utf-8")

	# 4 score rows: 1 success, 3 failed (1 unclassified, 1 other, 1 recovery_failure; 2 with mechanism).
	scores = [
		{
			"path": str(ep_c0),
			"case_id": "L1-N",
			"layout_id": "L1",
			"physical_success": True,
			"expected_label": "normal",
			"label": "normal",
			"defect_miss": False,
			"false_alarm": False,
			"is_unknown": False,
			"wrong_bowl": False,
			"root_cause": "none",
			"mechanism": None,
		},
		{
			"path": str(ep_c1),
			"case_id": "L1-D0",
			"layout_id": "L1",
			"physical_success": False,
			"expected_label": "defective",
			"label": "normal",
			"defect_miss": True,
			"false_alarm": False,
			"is_unknown": False,
			"wrong_bowl": False,
			"root_cause": "unclassified",
			"mechanism": None,
		},
		{
			"path": str(ep_h0),
			"case_id": "L2-N",
			"layout_id": "L2",
			"physical_success": False,
			"expected_label": "normal",
			"label": "unknown",
			"defect_miss": False,
			"false_alarm": False,
			"is_unknown": True,
			"wrong_bowl": False,
			"root_cause": "recovery_failure",
			"mechanism": "brick_pose_error",
		},
		{
			"path": str(ep_h1),
			"case_id": "L2-D1",
			"layout_id": "L2",
			"physical_success": False,
			"expected_label": "defective",
			"label": "normal",
			"defect_miss": True,
			"false_alarm": False,
			"is_unknown": False,
			"wrong_bowl": False,
			"root_cause": "other",
			"mechanism": "calibration_drift",
		},
	]
	scores_path = tmp_path / "scores.jsonl"
	scores_path.write_text("".join(json.dumps(r) + "\n" for r in scores), encoding="utf-8")

	cf_rows = [
		{"case_id": "L2-N", "verdict": "confirmed"},
		{"case_id": "L2-D1", "verdict": "not_perception"},
	]
	cf_path = tmp_path / "counterfactual.jsonl"
	cf_path.write_text("".join(json.dumps(r) + "\n" for r in cf_rows), encoding="utf-8")

	loop_dir = tmp_path / "loop"
	round_dir = loop_dir / "round-01"
	round_dir.mkdir(parents=True)
	(round_dir / "verify-a.json").write_text(
		json.dumps({"verdict": "merge", "gates": {"regression": {"n10": 0}}}), encoding="utf-8"
	)
	(round_dir / "verify-b.json").write_text(
		json.dumps({"verdict": "overfit_bank", "gates": {"regression": {"n10": 1}}}), encoding="utf-8"
	)
	(round_dir / "verify-c.json").write_text(
		json.dumps({"verdict": "reject", "gates": {"regression": {"n10": 2}}}), encoding="utf-8"
	)

	diag_rows = [
		{"session": "s1", "tool": "ledger_clusters", "latency_s": 0.2},
		{"session": "s1", "tool": "episode_summary", "latency_s": 0.8, "correct": True},
		{"session": "s2", "tool": "episode_summary", "latency_s": 1.0},
		{"session": "s2", "tool": "compare_runs", "latency_s": 1.5},
		{
			"session": "s2",
			"tool": "counterfactual_result",
			"latency_s": 0.5,
			"traj_id": heldout_ids[0],
			"root_cause": "recovery_failure",
		},
		{"session": "s3", "tool": "episode_summary", "latency_s": 0.4, "correct": True},
	]
	diag_path = tmp_path / "diag_calls.jsonl"
	diag_path.write_text("".join(json.dumps(r) + "\n" for r in diag_rows), encoding="utf-8")

	rc = round_metrics.main(
		[
			"--round-dir",
			str(round_dir),
			"--bench",
			str(bench_path),
			"--scores",
			str(scores_path),
			"--counterfactual",
			str(cf_path),
			"--diag-log",
			str(diag_path),
		]
	)
	assert rc == 0

	out = json.loads((round_dir / "round_metrics.json").read_text(encoding="utf-8"))
	assert set(out) >= {
		"round",
		"task",
		"attribution_coverage",
		"attribution_accuracy",
		"fix_yield",
		"regressions",
		"diagnosis_cost",
	}
	assert out["round"] == "round-01"

	# 1. Task: 1/4 = 0.25 across 2 layouts
	assert out["task"]["k"] == 1
	assert out["task"]["n"] == 4
	assert math.isclose(out["task"]["success_rate"], 0.25)
	assert out["task"]["n_clusters"] == 2
	assert "cell_metrics" in out["task"]

	# 2. Attribution coverage: 3 failed rows (L1-D0 unclassified, L2-N recovery_failure, L2-D1 other)
	cov = out["attribution_coverage"]
	assert cov["n_failed"] == 3
	assert cov["other_or_unclassified_n"] == 2
	assert math.isclose(cov["other_or_unclassified_rate"], 2 / 3)
	assert cov["with_mechanism_n"] == 2
	assert math.isclose(cov["with_mechanism_rate"], 2 / 3)
	assert cov["counterfactual_confirmed_n"] == 1
	assert math.isclose(cov["counterfactual_confirmed_rate"], 1 / 3)

	# 3. Attribution accuracy:
	# calib: c0 (human recovery_failure vs auto none), c1 (human occlusion vs auto unclassified) -> 0/2
	# heldout: h0 (human recovery_failure/brick_pose_error vs auto recovery_failure/brick_pose_error -> hit),
	#          h1 (human evidence_insufficient/brick_yaw_error vs auto other/calibration_drift -> miss)
	acc = out["attribution_accuracy"]
	assert acc["calib"]["n"] == 2
	assert acc["calib"]["root_cause_acc"] == 0.0
	assert acc["calib"]["mechanism_n"] == 1
	assert acc["calib"]["mechanism_acc"] == 0.0
	assert acc["heldout"]["n"] == 2
	assert math.isclose(acc["heldout"]["root_cause_acc"], 0.5)
	assert acc["heldout"]["mechanism_n"] == 2
	assert math.isclose(acc["heldout"]["mechanism_acc"], 0.5)

	# 4 & 5. Fix yield & regressions:
	assert out["fix_yield"] == {"total": 3, "merge": 1, "overfit_bank": 1, "reject": 1}
	assert out["regressions"] == {"n10_sum": 3, "n_verify_files": 3}

	# 6. Diagnosis cost: sessions s1 (2 calls, 1.0s, first=2), s2 (3 calls, 3.0s, first=3), s3 (1 call, 0.4s, first=1)
	dc = out["diagnosis_cost"]
	assert dc["n_sessions"] == 3
	assert dc["n_calls"] == 6
	assert dc["median_calls_per_session"] == 2.0
	assert math.isclose(dc["median_latency_s_per_session"], 1.0)
	assert dc["calls_to_first_correct"] == 2.0

	rounds_md = loop_dir / "rounds.md"
	assert rounds_md.is_file()
	lines = rounds_md.read_text(encoding="utf-8").splitlines()
	assert len(lines) == 5
	assert lines[-1].startswith("| round-01 |")

	# Run round-00 without optional counterfactual/diag-log, using --inventory auto_* lookup
	inv_dir = tmp_path / "inventory"
	_create_inventory_db(
		inv_dir,
		[
			{
				"traj_id": heldout_ids[0],
				"record_kind": "agent_episode",
				"path": str(tmp_path / "other_batch" / "ep0"),
				"auto_root_cause": "recovery_failure",
				"auto_mechanism": "brick_pose_error",
			}
		],
	)
	round0_dir = loop_dir / "round-00"
	round0_scores = tmp_path / "scores_r0.jsonl"
	round0_scores.write_text(json.dumps(scores[0]) + "\n", encoding="utf-8")
	rc0 = round_metrics.main(
		[
			"--round-dir",
			str(round0_dir),
			"--bench",
			str(bench_path),
			"--scores",
			str(round0_scores),
			"--inventory",
			str(inv_dir),
		]
	)
	assert rc0 == 0
	out0 = json.loads((round0_dir / "round_metrics.json").read_text(encoding="utf-8"))
	assert out0["attribution_coverage"]["counterfactual_confirmed_n"] is None
	assert out0["attribution_coverage"]["counterfactual_confirmed_rate"] is None
	assert out0["diagnosis_cost"] is None
	assert out0["attribution_accuracy"]["heldout"]["n"] == 1
	assert out0["attribution_accuracy"]["heldout"]["root_cause_acc"] == 1.0
	assert len(rounds_md.read_text(encoding="utf-8").splitlines()) == 6
