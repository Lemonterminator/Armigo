"""v2 experiment columns in the inventory scan, and the human audit store / API that survives rescans."""

from __future__ import annotations

import json

import pytest

from digital_twin_armigo.experiments.contract import AUDIT_STATES, PHYSICAL_OUTCOMES, ROOT_CAUSES
from digital_twin_armigo.inventory import audit, scan, store

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from test_inventory import NEW, _episode

from digital_twin_armigo.viewer import server


def _exp(case_id, layout, **kw):
	return {
		"batch": "p1",
		"engine": "mujoco",
		"sensing": "realistic",
		"group": "realistic-recal-C-v7bot",
		"camera_condition": "recal",
		"strategy": "C",
		"detector": "v7bot",
		"detector_sha256": "abc",
		"layout_id": layout,
		"case_id": case_id,
		"run_id": "r-" + case_id,
		"attempt": 1,
		"parent_run": None,
		**kw,
	}


def _run(root, rel, case_id, layout, state="done", **kw):
	d = root / rel
	seed = int(layout[1:])
	if state == "done":
		_episode(
			d,
			{"engine": kw.get("engine", "mujoco"), "sensing": "realistic", "experiment": _exp(case_id, layout, **kw)},
			case=(seed, case_id.endswith(tuple("0123456789")), 0),
		)
	else:
		d.mkdir(parents=True)
	(d / "run_status.json").write_text(json.dumps({"state": state, "started": "t", "finished": None}), encoding="utf-8")
	return d


@pytest.fixture
def v2(tmp_path):
	root = tmp_path / "runs"
	g = "p1/mujoco/realistic-recal-C-v7bot"
	_run(root, f"{g}/L5-N", "L5-N", "L5")
	_run(root, f"{g}/L5-D2", "L5-D2", "L5")
	other = _run(root, "p1/isaac/realistic-recal-C-v7bot/L5-D2", "L5-D2", "L5", engine="isaac")
	assert other.is_dir()
	_run(root, f"{g}/L6-D1", "L6-D1", "L6", state="running")
	_run(root, f"{g}/L7-D3-a2", "L7-D3", "L7", state="interrupted")
	return root


@pytest.fixture
def inv(v2, tmp_path):
	out = tmp_path / "data" / "inventory"
	store.build([("v2", v2)], out, overrides=tmp_path / "none.json")
	return out


@pytest.fixture
def client(inv, v2):
	return TestClient(server.create_app(v2, inventory=inv))


def _rows(client, **params):
	return client.get("/api/inventory/trajectories", params={"distinct": "0", **params}).json()["rows"]


def _tid(client, case, engine="mujoco"):
	return next(r["traj_id"] for r in _rows(client, case_id=case, engine=engine))


# -- scan ---------------------------------------------------------------------------------------------------------------
def test_scan_fills_experiment_columns(client):
	row = _rows(client, case_id="L5-D2", engine="mujoco")[0]
	assert row["traj_id"] == "agent:v2:p1/mujoco/realistic-recal-C-v7bot/L5-D2"
	for key, value in {
		"batch": "p1",
		"group": "realistic-recal-C-v7bot",
		"camera_condition": "recal",
		"strategy": "C",
		"detector": "v7bot",
		"detector_sha256": "abc",
		"layout_id": "L5",
		"run_id": "r-L5-D2",
		"attempt": 1,
		"run_state": "done",
		"sensing": "realistic",
		"verdict": "SUCCESS",
	}.items():
		assert row[key] == value, key
	assert row["parent_run"] is None and row["audit_state"] == "unreviewed"


def test_running_only_directory_is_a_row_without_verdict(client):
	run = _rows(client, run_state="running")
	assert [r["case_id"] for r in run] == ["L6-D1"]
	assert run[0]["verdict"] is None and run[0]["batch"] == "p1" and run[0]["layout_id"] == "L6"
	assert run[0]["defective"] == 1 and run[0]["defect_face"] == 1 and run[0]["sensing"] == "realistic"
	stopped = _rows(client, run_state="interrupted")[0]
	assert stopped["case_id"] == "L7-D3" and stopped["attempt"] == 2
	facets = client.get("/api/inventory/trajectories/facets").json()
	assert {i["value"] for i in facets["run_state"]} == {"done", "running", "interrupted"}
	assert {i["value"] for i in facets["strategy"]} == {"C"}


def test_old_episodes_keep_null_experiment_columns(tmp_path):
	root = tmp_path / "old"
	_episode(root / "run/case-0000", {"engine": "mujoco", "sensing": "truth"}, age=NEW)
	rows = list(scan.scan_agent_episodes(root, "d"))
	assert rows[0]["batch"] is None and rows[0]["run_state"] is None and rows[0]["case_id"] is None


def test_pairing(client):
	me = _tid(client, "L5-D2")
	pair = client.get("/api/inventory/paired", params={"id": me}).json()
	assert [r["case_id"] for r in pair["layout"]] == ["L5-N"]  # the other brick, same batch/engine/group
	assert [(r["engine"], r["case_id"]) for r in pair["case"]] == [("isaac", "L5-D2")]
	assert pair["layout"][0]["viewer_id"] is not None
	assert client.get("/api/inventory/paired", params={"id": "nope"}).status_code == 404


# -- audit store --------------------------------------------------------------------------------------------------------
def test_append_latest_and_conflict(tmp_path):
	db = tmp_path / "a" / "reviews.sqlite"
	body = {"reviewer": "linan", "state": "confirmed", "root_cause": "none", "physical_outcome": "green_bowl"}
	first = audit.append(db, "t1", body, expected_latest_id=None, code_version="abc")
	assert first["replaces"] is None and first["refs"] == [] and first["code_version"] == "abc"
	with pytest.raises(audit.Conflict) as exc:  # a second writer that still thinks there is no review
		audit.append(db, "t1", body, expected_latest_id=None, code_version=None)
	assert exc.value.current["id"] == first["id"]
	refs = [{"branch": "1", "frame": "40", "camera": "wrist"}]
	second = audit.append(
		db,
		"t1",
		{**body, "state": "disputed", "root_cause": "occlusion", "refs": refs},
		expected_latest_id=first["id"],
		code_version=None,
	)
	assert second["replaces"] == first["id"] and second["refs"] == [{"branch": 1, "frame": 40, "camera": "wrist"}]
	assert [r["id"] for r in audit.history(db, "t1")] == [second["id"], first["id"]]
	assert audit.latest(db, "t1")["state"] == "disputed" and audit.latest(db, "other") is None
	s = audit.summary(db)
	assert s["states"]["disputed"] == 1 and s["states"]["confirmed"] == 0 and s["root_causes"] == {"occlusion": 1}


@pytest.mark.parametrize(
	"patch",
	[
		{"state": "maybe"},
		{"state": None},
		{"root_cause": "gremlins"},
		{"physical_outcome": "moon"},
		{"reviewer": " "},
		{"refs": [{"frame": 3}]},
		{"refs": "0:3"},
	],
)
def test_enum_and_shape_validation(tmp_path, patch):
	body = {"reviewer": "x", "state": "confirmed", **patch}
	with pytest.raises(audit.ValidationError):
		audit.append(tmp_path / "r.sqlite", "t", body, expected_latest_id=None, code_version=None)
	assert audit.history(tmp_path / "r.sqlite", "t") == []


def test_enums_match_the_contract():
	e = audit.enums()
	assert (e["states"], e["root_causes"], e["physical_outcomes"]) == (
		list(AUDIT_STATES),
		list(ROOT_CAUSES),
		list(PHYSICAL_OUTCOMES),
	)


# -- API ----------------------------------------------------------------------------------------------------------------
def _post(client, **body):
	token = client.get("/api/token").json()["token"]
	return client.post("/api/audit/reviews", json=body, headers={"X-Viewer-Token": token})


def test_post_requires_the_token(client):
	body = {"traj_id": "t", "reviewer": "x", "state": "confirmed", "expected_latest_id": None}
	assert client.post("/api/audit/reviews", json=body).status_code == 403
	assert client.post("/api/audit/reviews", json=body, headers={"X-Viewer-Token": "wrong"}).status_code == 403
	assert client.get("/api/audit/reviews", params={"traj": "t"}).json()["reviews"] == []


def test_api_review_flow_conflict_and_validation(client):
	tid = _tid(client, "L5-D2")
	assert client.get("/api/audit/enums").json()["states"] == list(AUDIT_STATES)
	ok = _post(client, traj_id=tid, reviewer="linan", state="confirmed", root_cause="none", expected_latest_id=None)
	assert ok.status_code == 201 and ok.json()["id"] == 1 and ok.json()["reviewer"] == "linan"
	stale = _post(client, traj_id=tid, reviewer="b", state="disputed", expected_latest_id=None)
	assert stale.status_code == 409 and stale.json()["current"]["reviewer"] == "linan"
	bad = _post(client, traj_id=tid, reviewer="b", state="nope", expected_latest_id=1)
	assert bad.status_code == 422
	bad = _post(client, traj_id=tid, reviewer="b", state="disputed", root_cause="x", expected_latest_id=1)
	assert bad.status_code == 422
	again = _post(client, traj_id=tid, reviewer="b", state="disputed", root_cause="occlusion", expected_latest_id=1)
	assert again.status_code == 201 and again.json()["replaces"] == 1
	hist = client.get("/api/audit/reviews", params={"traj": tid}).json()["reviews"]
	assert [r["id"] for r in hist] == [2, 1]
	summary = client.get("/api/audit/summary").json()
	assert summary["states"]["disputed"] == 1 and summary["root_causes"] == {"occlusion": 1}


def test_audit_filters_survive_a_rescan(client, inv, v2, tmp_path):
	tid = _tid(client, "L5-D2")
	assert (
		_post(
			client, traj_id=tid, reviewer="a", state="disputed", root_cause="occlusion", expected_latest_id=None
		).status_code
		== 201
	)
	total = len(_rows(client))
	assert [r["traj_id"] for r in _rows(client, audit_state="disputed")] == [tid]
	assert len(_rows(client, audit_state="unreviewed")) == total - 1
	store.build([("v2", v2)], inv, overrides=tmp_path / "none.json")  # inventory.sqlite is rebuilt from scratch
	store.build([("v2", v2)], inv, overrides=tmp_path / "none.json")
	row = _rows(client, root_cause="occlusion")[0]
	assert row["traj_id"] == tid and row["audit_state"] == "disputed" and row["root_cause"] == "occlusion"
	facets = client.get("/api/inventory/trajectories/facets").json()
	assert {i["value"]: i["count"] for i in facets["audit_state"]}["disputed"] == 1
	assert [i["value"] for i in facets["root_cause"]] == ["occlusion"]
	detail = client.get("/api/inventory/trajectory", params={"id": tid}).json()
	assert detail["audit_state"] == "disputed"


def test_default_audit_path_is_next_to_the_inventory(tmp_path):
	assert (
		audit.default_path(tmp_path / "data" / "inventory")
		== (tmp_path / "data" / "audit" / "reviews.sqlite").resolve()
	)
