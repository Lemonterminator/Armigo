"""HTML viewer API and Rerun logging, on small synthetic episodes."""

from __future__ import annotations

import json
import os
import time

import numpy as np
import pytest

from digital_twin_armigo.viewer import stats as st

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from digital_twin_armigo.viewer import server

NODES = ["prepare", "choose_view", "scan_wrist", "evaluate"]
EDGES = [["prepare", "choose_view"], ["choose_view", "scan_wrist"], ["scan_wrist", "evaluate"]]


def _events():
	rows = [
		("node", "prepare", 0, 0.0, 0.5),
		("node", "choose_view", 30, 1.0, 1.0),
		("decision", "choose_view", 30, 1.0, 1.4),
		("node", "scan_wrist", 30, 1.0, 1.5),
		("node", "evaluate", 90, 3.0, 4.0),
	]
	out = []
	for seq, (kind, node, frame, sim, wall) in enumerate(rows):
		e = {
			"seq": seq,
			"kind": kind,
			"node": node,
			"branch": 0,
			"frame": frame,
			"sim_time_s": sim,
			"wall_time_s": wall,
		}
		if kind == "decision":
			e.update(
				choice="scan_wrist", source="llm", latency_s=0.4, fallback_reason=None, attempts=1, prompt_tokens=10
			)
		out.append(e)
	return out


def make_episode(path, *, finished=True):
	path.mkdir(parents=True)
	(path / "graph.json").write_text(json.dumps({"nodes": NODES, "edges": EDGES}), encoding="utf-8")
	(path / "events.jsonl").write_text("\n".join(json.dumps(e) for e in _events()) + "\n", encoding="utf-8")
	if finished:
		(path / "branch-0").mkdir()
		phase = np.array(["rest"] * 30 + ["approach"] * 40 + ["grasp"] * 20)
		np.savez(
			path / "branch-0" / "actions.npz",
			state=np.zeros((90, 2), np.float32),
			action=np.ones((90, 2), np.float32),
			phase=phase,
		)
		(path / "episode.json").write_text(
			json.dumps(
				{
					"fps": 30,
					"state_names": ["shoulder", "elbow"],
					"branches": [{"id": 0, "start_frame": 0, "actions": 90}],
					"result": {"verdict": "SUCCESS", "destination": "pink"},
				}
			),
			encoding="utf-8",
		)
	return path


def make_recording(path):
	path.mkdir(parents=True)
	scene = {
		"schema": "catalog-scene/v1",
		"units": "m",
		"up_axis": "Z",
		"quaternion": "xyzw",
		"fps": 30,
		"nodes": [
			{"id": "geom_0", "type": "box", "size": [0.1, 0.1, 0.01], "color": [0.5, 0.5, 0.5, 1]},
			{"id": "geom_1", "type": "sphere", "size": [0.02, 0, 0], "color": [1, 0, 0, 1]},
			{"id": "geom_2", "type": "mesh", "mesh": "0", "size": [1, 1, 1], "color": [0, 1, 0, 1]},
		],
		"meshes": {"0": {"vertices": [[0, 0, 0], [1, 0, 0], [0, 1, 0]], "faces": [[0, 1, 2]]}},
	}
	(path / "scene.json").write_text(json.dumps(scene), encoding="utf-8")
	poses = np.zeros((5, 3, 7))
	poses[:, :, 6] = 1.0
	poses[:, 1, 2] = np.linspace(0, 0.2, 5)
	np.savez(
		path / "motion.npz",
		poses=poses,
		timestamps=np.arange(5) / 30,
		node_ids=np.array(["geom_0", "geom_1", "geom_2"]),
	)
	np.savez(path / "telemetry.npz", jaw_forces=np.arange(10, dtype=float).reshape(5, 2), full_qpos=np.zeros((5, 20)))
	return path


@pytest.fixture
def root(tmp_path):
	make_episode(tmp_path / "agent" / "case-0000")
	make_episode(tmp_path / "agent" / "case-live", finished=False)
	make_recording(tmp_path / "recordings" / "rec-1")
	return tmp_path


class FakeSession:
	def __init__(self):
		self.started = None

	def start(self, target, recording=None):
		self.started = target
		return {"url": "http://127.0.0.1:1/?x"}

	def stop(self):
		self.started = None

	def status(self):
		return {"running": self.started is not None}


def test_discover_finds_agent_episodes_and_recordings(root):
	kinds = {r["id"]: r["kind"] for r in server.discover(root)}
	assert kinds == {"agent/case-0000": "agent", "agent/case-live": "agent", "recordings/rec-1": "recording"}
	finished = {r["id"]: r["finished"] for r in server.discover(root)}
	assert finished["agent/case-0000"] and not finished["agent/case-live"]


def test_episode_api_returns_state_and_statistics(root):
	client = TestClient(server.create_app(root, session=FakeSession()))
	d = client.get("/api/episode", params={"id": "agent/case-0000"}).json()
	assert d["kind"] == "agent" and d["result"]["verdict"] == "SUCCESS" and not d["live"]
	s = d["stats"]
	assert s["state"] == {**s["state"], "node": "evaluate", "finished": True}
	by_node = {r["node"]: r for r in s["nodes"]["wall"]}
	assert by_node["scan_wrist"]["total"] == pytest.approx(2.5)
	assert {r["phase"]: r["total"] for r in s["phases"]["sim"]} == pytest.approx(
		{"rest": 1.0, "approach": 40 / 30, "grasp": 20 / 30}
	)
	assert d["graph"]["nodes"] == NODES and len(d["events"]) == 5
	assert client.get("/api/episode", params={"id": "recordings/rec-1"}).json()["kind"] == "recording"


def test_live_episode_reports_open_visit_growing_with_the_clock(root):
	live = root / "agent" / "case-live"
	client = TestClient(server.create_app(root, session=FakeSession()))
	d = client.get("/api/episode", params={"id": "agent/case-live"}).json()
	assert d["live"] and d["stats"]["state"]["node"] == "evaluate" and not d["stats"]["state"]["finished"]
	# New events appended after the first poll show up on the next one (no stale cache for an unfinished episode).
	with (live / "events.jsonl").open("a", encoding="utf-8") as f:
		f.write(
			json.dumps(
				{
					"seq": 5,
					"kind": "node",
					"node": "prepare",
					"branch": 0,
					"frame": 100,
					"sim_time_s": 3.3,
					"wall_time_s": 5.0,
				}
			)
			+ "\n"
		)
	d2 = client.get("/api/episode", params={"id": "agent/case-live"}).json()
	assert d2["stats"]["state"]["node"] == "prepare" and d2["event_count"] == 6
	# An untouched events file older than the grace period is a dead run: not live, and its open visit stops growing.
	old = time.time() - server.LIVE_GRACE_S * 20
	os.utime(live / "events.jsonl", (old, old))
	d3 = client.get("/api/episode", params={"id": "agent/case-live"}).json()
	assert not d3["live"] and d3["stats"]["visits"][-1]["wall_s"] == pytest.approx(0.0)


def test_paths_cannot_escape_the_root(root):
	client = TestClient(server.create_app(root, session=FakeSession()))
	assert client.get("/api/episode", params={"id": "../"}).status_code == 404
	assert client.get("/api/episode", params={"id": "agent/nope"}).status_code == 404
	assert client.get("/api/live.jpg", params={"id": "agent/case-0000"}).status_code == 404


def test_rerun_control_needs_the_token_and_loopback_host(root):
	fake = FakeSession()
	client = TestClient(server.create_app(root, session=fake))
	assert client.post("/api/rerun", params={"id": "agent/case-0000"}).status_code == 403
	token = client.get("/api/token").json()["token"]
	r = client.post("/api/rerun", params={"id": "agent/case-0000"}, headers={"X-Viewer-Token": token})
	assert r.status_code == 200 and fake.started == root / "agent" / "case-0000"
	assert client.get("/api/rerun").json()["running"]
	assert client.delete("/api/rerun", headers={"X-Viewer-Token": token}).json() == {"running": False}
	assert client.get("/api/episodes", headers={"host": "evil.example"}).status_code == 403


def test_page_is_served(root):
	client = TestClient(server.create_app(root, session=FakeSession()))
	r = client.get("/")
	assert r.status_code == 200 and "Rerun" in r.text


def test_stats_cli_prints_markdown_and_json(root, capsys):
	from digital_twin_armigo.viewer import cli

	assert cli.main(["stats", str(root / "agent" / "case-0000")]) == 0
	assert "Episode timing" in capsys.readouterr().out
	assert cli.main(["stats", str(root / "agent" / "case-0000"), "--json"]) == 0
	assert json.loads(capsys.readouterr().out)["totals"]["visits"] == 4


def test_rerun_export_contains_state_robot_and_scene_entities(root, tmp_path):
	pytest.importorskip("rerun")
	import subprocess
	import sys

	from digital_twin_armigo.viewer import cli

	out = tmp_path / "ep.rrd"
	rc = cli.main(
		["rrd", str(root / "agent" / "case-0000"), "--recording", str(root / "recordings" / "rec-1"), "--out", str(out)]
	)
	assert rc == 0
	# The file sink is flushed when the recording stream is dropped; `rerun rrd print` also proves the file is readable.
	import rerun as rr

	rr.disconnect()
	text = subprocess.run(
		[sys.executable, "-m", "rerun", "rrd", "print", str(out)],
		capture_output=True,
		text=True,
		check=True,
		timeout=120,
	).stdout
	for entity in (
		"/state/log",
		"/state/active/prepare",
		"/stats/summary",
		"/robot/state/shoulder",
		"/phase/index",
		"/world/geom_0",
		"/world/geom_1",
		"/world/geom_2",
		"/telemetry/jaw_forces/0",
	):
		assert entity in text, entity
	assert "/telemetry/full_qpos" not in text


def test_wall_interpolator_matches_stats_module():
	assert st.wall_interpolator(_events())(30) == pytest.approx(1.5)


def test_logger_follows_a_growing_episode_until_it_finishes(tmp_path):
	pytest.importorskip("rerun")
	import threading

	from digital_twin_armigo.viewer import rerun_log

	ep = make_episode(tmp_path / "live", finished=False)
	events = (ep / "events.jsonl").read_text(encoding="utf-8").splitlines()
	(ep / "events.jsonl").write_text("\n".join(events[:2]) + "\n", encoding="utf-8")
	logger = rerun_log.EpisodeLogger(ep)
	result = {}
	t = threading.Thread(target=lambda: result.update(logger.follow(poll_s=0.05)))
	t.start()
	deadline = time.time() + 10
	while logger.events_logged < 2 and time.time() < deadline:
		time.sleep(0.05)
	assert logger.events_logged == 2 and t.is_alive()
	with (ep / "events.jsonl").open("a", encoding="utf-8") as f:
		f.write("\n".join(events[2:]) + "\n")
	(ep / "branch-0").mkdir()
	np.savez(
		ep / "branch-0" / "actions.npz",
		state=np.zeros((90, 2), np.float32),
		action=np.zeros((90, 2), np.float32),
		phase=np.array(["rest"] * 90),
	)
	(ep / "episode.json").write_text(
		json.dumps({"fps": 30, "branches": [{"id": 0, "start_frame": 0}], "state_names": ["a", "b"]}), encoding="utf-8"
	)
	t.join(10)
	assert not t.is_alive() and logger.events_logged == 5
	assert result["state"]["finished"] and result["phases"]["sim"][0]["phase"] == "rest"
	assert logger.branches_logged == {0} and len(logger.visits_logged) == 3
