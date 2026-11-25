import json

from digital_twin_armigo.governance import bronze, lake, silver
from digital_twin_armigo.governance.segments import split_of


def test_split_segments():
	assert split_of(3_000_101) == "train"
	assert split_of(3_100_005) == "val"
	assert split_of(3_900_001) == "test"
	assert split_of(3_600_001) == "loop-01"
	assert split_of(700_001) == "legacy"
	assert split_of(3_500_010) == "ood-appearance"
	assert split_of(9_999_999) == "unknown"


def _episode(root, group, name, *, events, state="done", seed=3000101, verdict="SUCCESS", positive=False, face=0):
	d = root / "runs" / "b1" / "mujoco" / group / name
	d.mkdir(parents=True)
	(d / "run_status.json").write_text(json.dumps({"state": state}), encoding="utf-8")
	(d / "task.json").write_text(json.dumps({"case": [seed, positive, face]}), encoding="utf-8")
	cam = "fixed" if group.startswith("truth") else group.split("-")[1]
	label = (
		"normal"
		if verdict == "SUCCESS" and not positive
		else ("defective" if positive and verdict == "SUCCESS" else "unknown")
	)
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
	(d / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
	return d


def test_leaky_tool_return_is_quarantined(tmp_path):
	leak = {"kind": "tool_call", "tool": "observe", "returned": {"brick_position_m": [0, 0, 0]}, "ok": True}
	d = _episode(tmp_path, "realistic-fixed-B-v7bot", "L3000101-N", events=[leak])
	con = lake.connect(tmp_path / "lake.sqlite")
	bronze.register(con, tmp_path, source="t")
	build = silver.build_silver(con, freeze=None)
	q, reasons = con.execute(
		"select quarantined, quarantine_reasons from silver_episode where build_id = ? and path = ?", (build, str(d))
	).fetchone()
	assert q == 1 and "sensor_guard" in json.loads(reasons)


def test_clean_episode_passes_and_changed_bronze_is_quarantined(tmp_path):
	d = _episode(tmp_path, "truth-B-v7bot", "L3000101-N", events=[])
	con = lake.connect(tmp_path / "lake.sqlite")
	bronze.register(con, tmp_path, source="t")
	b1 = silver.build_silver(con, freeze=None)
	assert con.execute("select quarantined from silver_episode where build_id = ?", (b1,)).fetchone()[0] == 0
	(d / "events.jsonl").write_text('{"kind": "x"}\n', encoding="utf-8")
	bronze.register(con, tmp_path, source="t")
	b2 = silver.build_silver(con, freeze=None)
	assert b2 != b1
	assert con.execute("select quarantined from silver_episode where build_id = ?", (b2,)).fetchone()[0] == 1


def test_build_pairs_and_pair_report(tmp_path):
	_episode(tmp_path, "truth-B-v7bot", "L3000101-N", events=[], verdict="SUCCESS")
	_episode(tmp_path, "realistic-fixed-B-v7bot", "L3000101-N", events=[], verdict="SUCCESS")
	_episode(tmp_path, "realistic-drift-B-v7bot", "L3000101-N", events=[], verdict="FAILED")
	con = lake.connect(tmp_path / "lake.sqlite")
	bronze.register(con, tmp_path, source="t")
	bid = silver.build_silver(con, freeze=None)
	n = silver.build_pairs(con, bid)
	assert n == 2
	rows = dict(con.execute("select camera_condition, outcome from silver_pair where build_id = ?", (bid,)).fetchall())
	assert rows == {"fixed": "both_pass", "drift": "truth_only"}
	md = silver.pair_report(con, bid)
	assert "| drift | 0 | 1 |" in md
	assert "truth_only" in md
