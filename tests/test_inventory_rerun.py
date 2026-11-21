"""Re-running failed agent episodes: planning, lineage, linking to the originals (which stay in the table)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import ClassVar

import pytest
from test_inventory import NEW, _episode, _result

from digital_twin_armigo.inventory import cli, query, rerun, scan, store


def _task(directory, case=(7, True, 2), *, use_llm=False):
	(directory / "task.json").write_text(
		json.dumps({"case": list(case), "policy": {"order": "wrist_first"}, "video": False, "use_llm": use_llm}),
		encoding="utf-8",
	)


@pytest.fixture
def failed(tmp_path):
	root = tmp_path / "data"
	_episode(
		root / "run/case-0000", {"engine": "mujoco", "sensing": "truth", "detector_sha256": "old"}, _result("FAILED")
	)
	_episode(
		root / "run/case-0001", {"engine": "mujoco", "sensing": "realistic"}, _result("FAILED"), case=(8, False, 1)
	)
	_episode(root / "run/case-0002", {"engine": "mujoco", "sensing": "truth"}, _result("FAILED"), case=(9, False, 1))
	_episode(root / "run/case-0003", {"engine": "mujoco", "sensing": "truth"}, _result("FAILED"), case=(10, False, 1))
	_episode(root / "run/case-0004", {"engine": "mujoco", "sensing": "truth"}, _result("SUCCESS"), case=(11, False, 1))
	_episode(root / "run/student", {"sensing": "realistic", "student": True}, _result("FAILED"), case=(12, False, 1))
	for name, case, llm in (
		("case-0000", (7, True, 2), False),
		("case-0001", (8, False, 1), False),
		("case-0002", (9, False, 1), True),
	):
		_task(root / "run" / name, case, use_llm=llm)  # case-0003 has no task.json: cannot be rebuilt
	return root


def _inv(failed, tmp_path):
	out = tmp_path / "inv"
	store.build([("d", failed)], out, overrides=tmp_path / "none.json")
	return out


def test_plan_takes_failed_rebuildable_agent_episodes_and_says_why_others_are_skipped(failed, tmp_path):
	inv = _inv(failed, tmp_path)
	items, skipped = rerun.plan(inv)
	assert sorted(i["case"][0] for i in items) == [7, 8]
	assert {i["sensing"] for i in items} == {"truth", "realistic"} and all(i["attempt"] == 1 for i in items)
	assert skipped["LLM agent (needs the model server)"] == 1
	assert skipped["no task.json (cannot rebuild the run)"] == 1  # the SUCCESS and the model rollout never qualify
	assert [i["case"][0] for i in rerun.plan(inv, limit=1)[0]] == [7]
	assert rerun.plan(inv, run_group="nothing")[0] == []


class FakeService:
	"""Stands in for TwinService: writes the episode files a real run would, with the verdict the test wants."""

	verdicts: ClassVar[dict] = {}
	started: ClassVar[list] = []

	def __init__(self, out, detector, bottom_profile=None, *, workers=1, sensing="truth"):
		self.sensing = sensing

	def run(self, case, policy, path, *, video, use_llm=False):
		assert os.environ["ARMIGO_RERUN"] == "record" and not use_llm and not video
		FakeService.started.append((tuple(case), self.sensing))
		_episode(
			path,
			{"engine": "mujoco", "sensing": self.sensing},
			_result(FakeService.verdicts.get(case[0], "FAILED")),
			case=tuple(case),
			age=NEW + 5,
		)
		_task(path, tuple(case))

	def close(self):
		pass


def test_rerun_keeps_originals_links_them_and_marks_the_current_result(failed, tmp_path, monkeypatch):
	monkeypatch.setenv("ARMIGO_RERUN", "")  # restored afterwards: execute() sets it for its own process
	from digital_twin_armigo.agent import twin_mcp

	monkeypatch.setattr(twin_mcp, "TwinService", FakeService)
	FakeService.verdicts, FakeService.started = {7: "SUCCESS"}, []
	inv = _inv(failed, tmp_path)
	detector = tmp_path / "det.json"
	detector.write_text("{}", encoding="utf-8")
	items, _ = rerun.plan(inv)
	out_root = inv / "reruns"
	results = rerun.execute(items, out_root, detector, progress=lambda _: None)
	assert sorted(r["verdict"] for r in results) == ["FAILED", "SUCCESS"]
	assert sorted(FakeService.started) == [((7, True, 2), "truth"), ((8, False, 1), "realistic")]
	dest = rerun.destination(out_root, items[0])
	lineage = json.loads((dest / scan.LINEAGE_FILE).read_text(encoding="utf-8"))
	assert lineage["rerun_of"] == items[0]["traj_id"] and lineage["attempt"] == 1 and lineage["supersedes"] is True
	assert lineage["original_verdict"] == "FAILED" and lineage["detector_changed"] is True
	again = rerun.execute(items, out_root, detector, progress=lambda _: None)  # resuming runs nothing twice
	assert all(r.get("skipped") == "already done" for r in again) and len(FakeService.started) == 2

	# the next scan picks up <inventory>/reruns; originals stay with their old verdict
	assert (
		cli.main(["--out", str(inv), "scan", "--root", f"d={failed}", "--overrides", str(tmp_path / "none.json")]) == 0
	)
	rows = {r["traj_id"]: r for r in query.rows(inv, "trajectories", {"distinct": "0", "limit": "500"})["rows"]}
	original = rows[items[0]["traj_id"]]
	assert original["verdict"] == "FAILED" and original["superseded_by"]
	new = rows[original["superseded_by"]]
	assert new["rerun_of"] == original["traj_id"] and new["attempt"] == 1 and new["same_as"] is None
	assert new["rerun_info"]["original_verdict"] == "FAILED"
	latest = {r["traj_id"] for r in query.rows(inv, "trajectories", {"latest": "1", "limit": "500"})["rows"]}
	assert original["traj_id"] not in latest and new["traj_id"] in latest
	assert query.rows(inv, "trajectories", {"is_rerun": "1"})["total"] == 2
	assert query.summary(inv)["reruns"] == {"reruns": 2, "superseded": 2, "rerun_success": 1}
	# everything is re-run now; --redo plans a second attempt
	items2, skipped2 = rerun.plan(inv)
	assert items2 == [] and skipped2["already re-run"] == 2
	assert sorted(i["attempt"] for i in rerun.plan(inv, redo=True)[0]) == [2, 2]


def test_link_reruns_tags_exact_reproductions_and_ignores_non_superseding_reruns():
	base = {"episode_sha256": "a", "written_at": 1.0, "tags": None, "superseded_by": None, "rerun_of": None}
	rows = [
		{**base, "traj_id": "orig"},
		{
			**base,
			"traj_id": "r1",
			"rerun_of": "orig",
			"attempt": 1,
			"written_at": 2.0,
			"rerun_info": {"supersedes": True},
		},
		{
			**base,
			"traj_id": "r2",
			"rerun_of": "orig",
			"attempt": 2,
			"episode_sha256": "b",
			"written_at": 3.0,
			"rerun_info": {"supersedes": True},
		},
		{**base, "traj_id": "m", "episode_sha256": "c"},
		{
			**base,
			"traj_id": "m1",
			"rerun_of": "m",
			"attempt": 1,
			"episode_sha256": "d",
			"rerun_info": {"supersedes": False},
		},
	]
	store.link_reruns(rows)
	by = {r["traj_id"]: r for r in rows}
	assert by["orig"]["superseded_by"] == "r2"  # the latest attempt wins
	assert by["r1"]["tags"] == ["reproduced"] and by["r2"]["tags"] is None
	assert by["m"]["superseded_by"] is None  # a measurement re-run never replaces the original
	assert store.mark_duplicates(rows) == 0  # a re-run is never folded into the original as a duplicate


def test_cli_rerun_dry_run_and_errors(failed, tmp_path, capsys):
	inv = _inv(failed, tmp_path)
	assert cli.main(["--out", str(inv), "rerun", "--dry-run"]) == 0
	text = capsys.readouterr().out
	assert "2" in text and not (inv / "reruns").exists()  # a dry run writes nothing
	assert cli.main(["--out", str(inv), "rerun", "--category", "nope", "--dry-run"]) != 0


def test_scan_keeps_the_old_verdict_and_reads_the_final_cycle_verdict_and_failure_modes(tmp_path):
	root = tmp_path / "data"
	cycle_ok = {"verdict": "SUCCESS", "failure_mode": "ok", "start_frame": 1552}
	cycle_bad = {"verdict": "FAILURE", "failure_mode": "grasp_slip", "start_frame": 1552}
	for name, final, cycle in (("a", "SUCCESS", cycle_ok), ("b", "FAILED", cycle_bad)):
		result = {
			**_result("FAILED"),
			"verdict_final_cycle": final,
			"health": {"failure_mode": "place_miss", "final_cycle": cycle},
		}
		_episode(root / f"run/{name}", {"engine": "mujoco", "sensing": "truth"}, result, case=(ord(name), False, 1))
	_episode(root / "run/plain", {"engine": "mujoco", "sensing": "truth"}, _result("SUCCESS"), case=(3, False, 1))
	inv = tmp_path / "inv"
	store.build([("d", root)], inv, overrides=tmp_path / "none.json")
	rows = {Path(r["path"]).name: r for r in query.rows(inv, "trajectories", {})["rows"]}
	assert rows["a"]["verdict"] == "FAILED" and rows["a"]["verdict_final_cycle"] == "SUCCESS"
	assert rows["a"]["failure_mode"] == "place_miss" and rows["a"]["failure_mode_final_cycle"] is None
	assert rows["b"]["failure_mode_final_cycle"] == "grasp_slip"
	assert rows["plain"]["failure_mode"] is None and rows["plain"]["verdict_final_cycle"] is None  # SUCCESS: no mode
	only = query.rows(inv, "trajectories", {"verdict": "FAILED", "verdict_final_cycle": "SUCCESS"})
	assert [Path(r["path"]).name for r in only["rows"]] == ["a"]
	assert {f["value"] for f in query.facets(inv, "trajectories", {})["failure_mode"]} == {"place_miss"}


def test_final_cycle_verdict_needs_a_correct_label_and_a_passing_cycle():
	pytest.importorskip("mujoco")
	from digital_twin_armigo.agent.twin_mujoco import TwinMujoco

	f = TwinMujoco._final_cycle_verdict
	assert f(True, {"final_cycle": {"verdict": "SUCCESS"}}) == {"verdict_final_cycle": "SUCCESS"}
	assert f(False, {"final_cycle": {"verdict": "SUCCESS"}}) == {"verdict_final_cycle": "FAILED"}  # wrong label or bowl
	assert f(True, {"final_cycle": {"verdict": "FAILURE"}}) == {"verdict_final_cycle": "FAILED"}
	assert f(True, {"verdict": "SUCCESS"}) == {}  # no park or retry: nothing to add
