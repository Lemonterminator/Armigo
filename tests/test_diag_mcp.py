import ast
import asyncio
import json
from pathlib import Path

import pytest

from digital_twin_armigo.experiments import diag_mcp


def _ep(d: Path, events, verdict="SUCCESS"):
	d.mkdir(parents=True)
	(d / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
	(d / "episode.json").write_text(json.dumps({"result": {"verdict": verdict, "label": "normal"}}), encoding="utf-8")
	return d


BASE = [
	{"seq": 0, "kind": "node", "node": "scan_wrist", "branch": 0, "wall_time_s": 1.0},
	{"seq": 1, "kind": "view", "node": "scan_wrist", "branch": 0, "observation": {"score": -0.5}, "wall_time_s": 2.0},
	{"seq": 2, "kind": "node", "node": "place", "branch": 0, "wall_time_s": 3.0},
]


def test_compare_ignores_wall_time_and_lists_score_deltas(tmp_path):
	other = [dict(e, wall_time_s=e["wall_time_s"] + 9) for e in BASE]
	other[1] = {**other[1], "observation": {"score": -0.5 + 1e-3}}
	r = diag_mcp.compare_runs(_ep(tmp_path / "a", BASE), _ep(tmp_path / "b", other))
	assert r["identical_discrete"] and r["verdict_equal"] and r["first_divergence"] is None
	assert len(r["score_deltas"]) == 1


def test_compare_finds_first_divergence(tmp_path):
	other = BASE[:2] + [{"seq": 2, "kind": "failure", "node": "scan_wrist", "branch": 0, "code": "no_clamp"}]
	r = diag_mcp.compare_runs(_ep(tmp_path / "a", BASE), _ep(tmp_path / "b", other, verdict="FAILED"))
	assert not r["identical_discrete"] and r["first_divergence"]["index"] == 2 and not r["verdict_equal"]


def test_compare_detects_length_difference_and_ignores_seq_latency(tmp_path):
	other = [dict(e, seq=e["seq"] + 10, latency_s=0.42) for e in BASE[:2]]
	r = diag_mcp.compare_runs(_ep(tmp_path / "a", BASE), _ep(tmp_path / "b", other))
	assert not r["identical_discrete"]
	assert r["first_divergence"]["index"] == 2
	assert r["first_divergence"]["a"]["node"] == "place"
	assert r["first_divergence"]["b"] is None


def test_diag_tools_are_disjoint_from_task_tools_and_never_imported_by_step_mcp():
	pytest.importorskip("mujoco")
	from digital_twin_armigo.agent import step_mcp as sm

	task_tools = set(sm.TOOLS_COMMON) | {t for v in sm.TOOLS_ONLY.values() for t in v}
	assert not set(diag_mcp.TOOLS) & task_tools
	tree = ast.parse(Path(sm.__file__).read_text(encoding="utf-8"))
	imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom | ast.Import) for a in n.names}
	imported |= {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
	assert not any("diag_mcp" in name for name in imported)


def test_calls_are_logged_per_session(tmp_path):
	ep = _ep(tmp_path / "a", BASE)
	diag_mcp.logged(tmp_path, "s1", "episode_summary", {"path": str(ep)}, lambda: diag_mcp.episode_summary(ep))
	rows = [json.loads(line) for line in (tmp_path / "diag_calls.jsonl").read_text(encoding="utf-8").splitlines()]
	assert rows[0]["session"] == "s1" and rows[0]["tool"] == "episode_summary" and rows[0]["latency_s"] >= 0


def test_episode_summary_extracts_fields_and_truncates_timeline(tmp_path):
	events = [
		{"seq": 0, "kind": "node", "node": "prepare", "branch": 0},
		{"seq": 1, "kind": "bowls_perceived", "node": "prepare", "branch": 0, "error_xy_mm": {"green": 1.2}},
		{
			"seq": 2,
			"kind": "brick_perceived",
			"node": "prepare",
			"branch": 0,
			"found": True,
			"error_xy_mm": 12.5,
			"error_yaw_deg": 2.1,
		},
		{"seq": 3, "kind": "decision", "node": "prepare", "branch": 0, "latency_s": 0.1},
		{"seq": 4, "kind": "tool_call", "node": "scan_wrist", "branch": 0, "tool": "capture"},
		{"seq": 5, "kind": "view", "node": "scan_wrist", "branch": 0, "observation": {"id": "v0", "score": -0.7}},
		{"seq": 6, "kind": "failure", "node": "place", "branch": 0, "code": "place_miss", "frame": 42},
	]
	ep = _ep(tmp_path / "ep1", events, verdict="FAILED")
	(ep / "episode.json").write_text(
		json.dumps(
			{"result": {"verdict": "FAILED", "label": "normal", "failures": [{"code": "place_miss", "frame": 42}]}}
		),
		encoding="utf-8",
	)
	(ep / "task.json").write_text(json.dumps({"case": [100, False, 0]}), encoding="utf-8")

	summary = diag_mcp.episode_summary(ep)
	assert summary["root_cause"] == "placement_failure"
	assert summary["scored_verdict"] == "FAILED"
	assert summary["physical_outcome"] == "unknown"
	assert summary["physical_success"] is False
	assert summary["frame"] == 42
	assert "place_miss" in summary["detail"]
	assert "mechanism" in summary and "upstream" in summary and "precursors" in summary

	timeline = summary["timeline"]
	assert len(timeline) == 6  # decision event excluded
	assert timeline[0] == "0 node prepare"
	assert timeline[1] == "1 bowls_perceived prepare error_xy_mm={'green': 1.2}"
	assert timeline[2] == "2 brick_perceived prepare error_xy_mm=12.5 error_yaw_deg=2.1"
	assert timeline[3] == "4 tool_call scan_wrist tool=capture"
	assert timeline[4] == "5 view scan_wrist score=-0.7"
	assert timeline[5] == "6 failure place code=place_miss"

	many = [{"seq": i, "kind": "node", "node": f"n{i}", "branch": 0} for i in range(250)]
	long_summary = diag_mcp.episode_summary(_ep(tmp_path / "ep_long", many))
	tl = long_summary["timeline"]
	assert len(tl) == 201
	assert tl[0] == "0 node n0"
	assert tl[99] == "99 node n99"
	assert tl[100] == "... (truncated)"
	assert tl[101] == "150 node n150"
	assert tl[-1] == "249 node n249"


def test_ledger_clusters_and_counterfactual_result(tmp_path):
	r1 = tmp_path / "r1"
	r1.mkdir()
	clusters = [{"key": ["grasp", "pick_miss", "grasp_or_path_failure"], "n": 3, "impact": 3.0}]
	(r1 / "ledger.json").write_text(json.dumps({"batch": "b1", "clusters": clusters}), encoding="utf-8")
	assert diag_mcp.ledger_clusters(r1) == clusters

	r2 = tmp_path / "r2"
	r2.mkdir()
	(r2 / "ledger.json").write_text(json.dumps(clusters), encoding="utf-8")
	assert diag_mcp.ledger_clusters(r2) == clusters

	assert diag_mcp.counterfactual_result(r1, "c1") is None
	(r1 / "counterfactual.jsonl").write_text(
		json.dumps({"case_id": "c1", "verdict": "confirmed", "minimal": [["brick"]]}) + "\n",
		encoding="utf-8",
	)
	assert diag_mcp.counterfactual_result(r1, "c1")["verdict"] == "confirmed"
	assert diag_mcp.counterfactual_result(r1, "missing") is None


def test_create_diag_mcp_registers_tools_and_logs_calls(tmp_path):
	rdir = tmp_path / "round-01"
	rdir.mkdir()
	(rdir / "ledger.json").write_text(json.dumps({"clusters": [{"key": ["a", "b", "c"], "n": 1}]}), encoding="utf-8")

	mcp = diag_mcp.create_diag_mcp()
	names = [t.name for t in asyncio.run(mcp.list_tools())]
	assert tuple(names) == diag_mcp.TOOLS

	asyncio.run(mcp.call_tool("ledger_clusters", {"round_dir": str(rdir), "session": "s-mcp"}))
	rows = [json.loads(line) for line in (rdir / "diag_calls.jsonl").read_text(encoding="utf-8").splitlines()]
	assert len(rows) == 1
	assert rows[0]["session"] == "s-mcp" and rows[0]["tool"] == "ledger_clusters"
