"""The /inventory page and its read-only API, on the small inventory the scan tests build."""

from __future__ import annotations

import json
import shutil

import pytest

from digital_twin_armigo.inventory import query, store, taxonomy

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from test_inventory import tree  # noqa: F401  (fixture)
from test_viewer_app import EDGES, NODES, _events

from digital_twin_armigo.viewer import server


@pytest.fixture
def inv(tree, tmp_path):  # noqa: F811
	out = tmp_path / "inv"
	store.build([("d", tree)], out, overrides=tmp_path / "none.json")
	return out


@pytest.fixture
def client(inv, tree, tmp_path):  # noqa: F811
	return TestClient(server.create_app(tree, inventory=inv))


def test_page_is_served_and_linked_from_the_viewer(client):
	page = client.get("/inventory")
	assert page.status_code == 200 and "轨迹与场景清单" in page.text
	assert 'href="/inventory"' in client.get("/").text


def test_summary_has_labels_matrix_and_scan_meta(client):
	s = client.get("/api/inventory/summary").json()
	assert s["categories"] == taxonomy.CATEGORIES
	assert s["meta"]["rules_version"] == taxonomy.RULES_VERSION
	assert s["duplicates"] == 1 and s["totals"]["trajectories"] == 10
	cats = {r["category"] for r in s["matrix"]}
	assert {"agent_omniscient", "agent_realistic", "model", "model_takeover", "scripted"} <= cats
	# the matrix counts content-distinct trajectories only
	assert sum(r["n"] for r in s["matrix"]) == 10 - 1


def test_trajectory_filters_search_sort_and_paging(client):
	def get(**p):
		return client.get("/api/inventory/trajectories", params=p).json()

	assert get(distinct="0")["total"] == 10
	assert get()["total"] == 9  # byte copy hidden by default
	only = get(category="model,model_takeover")
	assert {r["category"] for r in only["rows"]} == {"model", "model_takeover"}
	assert get(q="run-b")["total"] == 1
	assert get(q="%")["total"] == 0  # LIKE wildcards in the search box are literal
	assert get(has_motion="1")["total"] == 1 and get(has_motion="0")["total"] == 8  # the catalog row with motion.npz
	asc = [r["seed"] for r in get(category="agent_omniscient", sort="seed")["rows"]]
	desc = [r["seed"] for r in get(category="agent_omniscient", sort="-seed")["rows"]]
	assert asc == sorted(asc) and desc == sorted(desc, reverse=True) and len(asc) == 3
	page = get(distinct="0", limit="3", offset="3")
	assert page["total"] == 10 and len(page["rows"]) == 3 and page["offset"] == 3
	assert get(limit="99999")["limit"] == query.MAX_LIMIT


def test_json_columns_are_decoded_and_unknown_params_are_not_sql(client):
	scenes = client.get("/api/inventory/scenes").json()["rows"]
	assert all(isinstance(r["categories"], dict) for r in scenes if r["categories"] is not None)
	# a hostile sort / filter name must neither error nor change the result
	assert (
		client.get("/api/inventory/trajectories", params={"sort": "seed; drop table trajectories"}).status_code == 200
	)
	assert client.get("/api/inventory/trajectories", params={"nope": "1"}).json()["total"] == 9
	assert client.get("/api/inventory/trajectories").json()["total"] == 9


def test_facets_ignore_their_own_filter(client):
	f = client.get("/api/inventory/trajectories/facets", params={"category": "model"}).json()
	cats = {i["value"]: i["count"] for i in f["category"]}
	assert cats["model"] == 1 and cats["agent_realistic"] >= 1  # other categories still offered
	engines = {i["value"]: i["count"] for i in f["engine"]}
	assert sum(engines.values()) == 1  # but every other facet respects the category filter


def test_trajectory_detail_links_to_the_viewer_when_under_root(client):
	rows = client.get("/api/inventory/trajectories", params={"q": "run-b"}).json()["rows"]
	detail = client.get("/api/inventory/trajectory", params={"id": rows[0]["traj_id"]}).json()
	assert detail["viewer_id"] == "agent/run-b/case-0000"
	hidden = client.get("/api/inventory/trajectories", params={"record_kind": "catalog_episode"}).json()
	assert hidden["total"] == 9  # record_kind is not a public filter
	assert client.get("/api/inventory/trajectory", params={"id": "nope"}).status_code == 404


def test_episodes_outside_root_replay_through_the_inventory(inv, tree, tmp_path):  # noqa: F811
	elsewhere = tmp_path / "elsewhere"
	elsewhere.mkdir()
	other = TestClient(server.create_app(elsewhere, inventory=inv))  # the served root contains none of the episodes
	rows = other.get("/api/inventory/trajectories", params={"q": "run-b"}).json()["rows"]
	detail = other.get("/api/inventory/trajectory", params={"id": rows[0]["traj_id"]}).json()
	assert detail["viewer_id"] == "inv:" + rows[0]["traj_id"]
	episode_dir = tree / "agent/run-b/case-0000"
	assert other.get("/api/episode", params={"id": detail["viewer_id"]}).status_code == 404  # no events.jsonl yet
	shutil.rmtree(episode_dir / "branch-0", ignore_errors=True)  # the scan fixture's arrays are not viewer arrays
	(episode_dir / "graph.json").write_text(json.dumps({"nodes": NODES, "edges": EDGES}), encoding="utf-8")
	(episode_dir / "events.jsonl").write_text("\n".join(json.dumps(e) for e in _events()) + "\n", encoding="utf-8")
	replay = other.get("/api/episode", params={"id": detail["viewer_id"]})
	assert replay.status_code == 200 and replay.json()["kind"] == "agent"
	# only ids the inventory knows are resolved: no client-supplied paths
	assert other.get("/api/episode", params={"id": "inv:../../etc"}).status_code == 404
	assert other.get("/api/episode", params={"id": "inv:" + str(episode_dir)}).status_code == 404


def test_unknown_table_and_missing_inventory(tree, tmp_path, client):  # noqa: F811
	assert client.get("/api/inventory/nope").status_code == 404
	empty = TestClient(server.create_app(tree, inventory=tmp_path / "absent"))
	r = empty.get("/api/inventory/summary")
	assert r.status_code == 404 and "armigo-inventory scan" in r.json()["detail"]
	assert empty.get("/inventory").status_code == 200  # the page itself still loads and shows the message


def test_inventory_api_is_loopback_only(client):
	assert client.get("/api/inventory/summary", headers={"host": "evil.example"}).status_code == 403
