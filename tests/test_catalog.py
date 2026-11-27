"""Governance invariants tested with synthetic sources; no GPU required."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from digital_twin_armigo.catalog.jobs import Jobs
from digital_twin_armigo.catalog.playback import load_motion
from digital_twin_armigo.catalog.quality import (
	evaluate_quality,
	label_definitions,
	layout_group,
	normalize_labels,
	quality_status,
	split_for_group,
)
from digital_twin_armigo.catalog.store import Catalog


@pytest.fixture
def catalog(tmp_path):
	return Catalog(tmp_path, tmp_path / "data/catalog")


def report(catalog, name="run.json", verdict="SUCCESS", seed=1):
	p = catalog.root / name
	p.write_text(
		json.dumps(
			{
				"mode": "matrix",
				"conditions": {
					"baseline_2x2": {
						"episodes_detail": [
							{
								"seed": seed,
								"verdict": verdict,
								"info": {"n_frames": 500, "jaw_env_contact_max_N": 0},
								"health": {},
							}
						]
					}
				},
			}
		),
		encoding="utf-8",
	)
	return catalog.import_file(p)[0]


def geometry():
	return {
		"layout": {
			"green": [0.2, 0.2],
			"pink": [0.2, -0.2],
			"brick_xy": [0.25, 0],
			"brick_yaw_deg": 20,
			"brick_size": [0.03, 0.03, 0.02],
			"clutter": [[0.2, 0.2, 0, 1, 0, 0]],
			"distractors": [],
			"markers": [],
		}
	}


def recorded(catalog, name="new", parent=None):
	dir_ = catalog.root / name
	dir_.mkdir()
	scene = geometry()
	scene.update({"units": "m", "up_axis": "Z", "quaternion": "xyzw", "nodes": [{"id": "brick"}]})
	(dir_ / "scene.json").write_text(json.dumps(scene))
	np.savez(
		dir_ / "motion.npz",
		poses=np.array([[[0, 0, 0, 0, 0, 0, 1]], [[0.1, 0, 0, 0, 0, 0, 1]]]),
		timestamps=np.array([0, 1 / 30]),
		node_ids=np.array(["brick"]),
	)
	(dir_ / "episode.json").write_text(
		json.dumps(
			{
				"schema": "catalog-episode/v1",
				"alias": name,
				"engine": "mujoco",
				"scene": scene,
				"result": {"verdict": "SUCCESS"},
				"provenance": {"identity": {"code_sha256": "a" * 64}},
				"parent_episode_id": parent,
				"artifacts": {"scene": "scene.json", "motion": "motion.npz"},
			}
		)
	)
	return catalog.import_file(dir_ / "episode.json")[0]


@pytest.mark.parametrize("value", [None, True, float("nan"), float("inf")])
def test_missing_is_unknown(value):
	checks = evaluate_quality({"max_unintended_force_N": value}, {})
	assert checks[0]["status"] == "UNKNOWN"
	assert quality_status(checks) == "UNKNOWN"


def test_contact_rules_independent_and_estimated_torque_unknown():
	checks = evaluate_quality(
		{
			"max_unintended_force_N": 4,
			"torque_evidence_source": "estimated_implicit_pd",
			"evidence_refs": {"max_torque_overload_s": 0},
		},
		{"jaw_env_contact_max_N": 4},
	)
	assert checks[0]["status"] == "FAIL"
	assert checks[1]["status"] == "PASS"
	assert checks[-1]["status"] == "UNKNOWN"


def test_labels_cannot_be_silently_mixed():
	a = normalize_labels("mujoco", {"defect_px": [12]})
	b = normalize_labels("isaac", {"defect_px": [12]})
	assert a == {"defect_visible_pixels": [12]}
	assert b == {"defect_visibility_proxy_score": [12]}
	assert all(x["privileged"] for x in label_definitions("isaac").values())
	with pytest.raises(ValueError):
		normalize_labels("unknown", {"defect_px": [12]})


def test_layout_groups_ignore_appearance_but_not_geometry():
	a = geometry()
	b = copy.deepcopy(a)
	b["layout"]["clutter"][0][3:] = [0, 1, 0]
	b["layout"]["defective"] = True
	assert layout_group(a) == layout_group(b)
	b["layout"]["clutter"][0][0] += 0.01
	assert layout_group(a) != layout_group(b)
	assert layout_group({"layout": {"green": [1, 2]}}) is None
	assert split_for_group(layout_group(a)) == split_for_group(layout_group(a))


@pytest.mark.parametrize("verdict", ["SUCCESS", "FAILURE", "UNKNOWN", "INFEASIBLE"])
def test_import_idempotence_and_verdict(catalog, verdict):
	ident = report(catalog, verdict=verdict)
	assert catalog.import_file(catalog.root / "run.json") == [ident]
	assert catalog.summary()["episodes"] == 1
	assert catalog.episode(ident)["verdict"] == verdict
	assert Jobs(catalog).list() == {"items": []}


def test_same_alias_different_evidence_remains_distinct(catalog):
	a = report(catalog, "one.json")
	b = report(catalog, "two.json", verdict="FAILURE")
	assert a != b
	assert catalog.episode(a)["alias"] == catalog.episode(b)["alias"]
	assert catalog.episode(a)["provenance"]["original_code_identity"] is None


def test_changed_file_blocks_export(catalog):
	ident = report(catalog)
	jobs = Jobs(catalog)
	job = jobs.submit([ident])
	(catalog.root / "run.json").write_text("{}")
	result = jobs.execute(job["id"])
	assert result["state"] == "FAILED" and result["download_url"] is None
	assert not catalog.verify([ident])["ok"]


def test_export_explicit_frozen_and_idempotent(catalog):
	ident = report(catalog)
	jobs = Jobs(catalog)
	a = jobs.submit([ident], request_id="same")
	b = jobs.submit([ident], request_id="same")
	assert a["id"] == b["id"]
	catalog.evaluate([ident], {"pinch_force_N": 12})
	result = jobs.execute(a["id"])
	assert result["state"] == "SUCCEEDED"
	manifest = json.loads((Path(result["result"]["bundle"]).parent / "manifest.json").read_text(encoding="utf-8"))
	assert manifest["episodes"][0]["quality"][4]["limit"] is None
	assert len(manifest["files"]) == 1


def test_missing_scene_prevents_rtx(catalog):
	ident = report(catalog)
	with pytest.raises(ValueError, match="complete motion"):
		Jobs(catalog).submit([ident], "isaac-rtx")


def test_frozen_split_group_and_members_do_not_drift(catalog):
	a = recorded(catalog, "first")
	b = recorded(catalog, "second", a)
	result = catalog.freeze_split([a, b], "v1")
	assert len({m["split"] for m in result["members"]}) == 1
	with pytest.raises(ValueError, match="frozen"):
		catalog.freeze_split([a], "v1")
	assert len(catalog.memberships(a)) == 1
	with pytest.raises(ValueError, match="layout"):
		catalog.freeze_split([report(catalog)], "invalid")


def test_datasets_lists_frozen_versions_with_split_counts(catalog):
	assert catalog.datasets() == {"datasets": []}
	a = recorded(catalog, "first")
	b = recorded(catalog, "second", a)
	frozen = catalog.freeze_split([a, b], "v1")
	(listed,) = catalog.datasets()["datasets"]
	assert listed["id"] == frozen["id"] and listed["name"] == "v1" and listed["seed"] == 0
	assert sum(listed["splits"].values()) == 2 and set(listed["splits"]) == {"train", "validation", "test"}


def test_no_paths_outside_project_or_logs(catalog, tmp_path):
	with pytest.raises(ValueError):
		catalog.project_path(tmp_path.parent / "private.json")
	with pytest.raises(ValueError):
		catalog.project_path("results/agent_logs/private.md")


def test_motion_shape_and_coordinates(catalog):
	e = catalog.episode(recorded(catalog))
	roles = {a["role"]: a["path"] for a in e["artifacts"]}
	assert load_motion(roles)[1].shape == (2, 1, 7)
	scene = json.loads(Path(roles["scene"]).read_text())
	scene["quaternion"] = "wxyz"
	Path(roles["scene"]).write_text(json.dumps(scene))
	with pytest.raises(ValueError, match="coordinate"):
		load_motion(roles)


def test_web_read_no_export_and_write_token(catalog):
	pytest.importorskip("fastapi")
	pytest.importorskip("httpx")
	from fastapi.testclient import TestClient

	from digital_twin_armigo.catalog.web import create_app

	ident = report(catalog)
	app = create_app(catalog)
	with TestClient(app) as client:
		assert client.get("/").status_code == 200
		assert client.get("/api/episodes").json()["total"] == 1
		assert client.get(f"/api/episodes/{ident}").json()["id"] == ident
		assert client.get("/api/jobs").json()["items"] == []
		assert client.post("/api/import", json={"path": "run.json"}).status_code == 403
		token = client.get("/api/token").json()["token"]
		assert (
			client.post("/api/import", json={"path": "run.json"}, headers={"X-Catalog-Token": token}).status_code == 200
		)


def test_held_out_role_propagates_to_rerun(catalog):
	a = report(catalog)
	with catalog.connect() as db:
		db.execute("UPDATE episodes SET role='legacy_held_out' WHERE id=?", (a,))
	b = recorded(catalog, parent=a)
	assert catalog.episode(b)["role"] == "legacy_held_out"
	with pytest.raises(ValueError, match="evaluation"):
		catalog.freeze_split([b], "bad")


def test_producer_claims_separated_from_import_identity(catalog):
	ident = recorded(catalog)
	provenance = catalog.episode(ident)["provenance"]
	assert provenance["identity_evidence"] == "producer_manifest"
	assert len(provenance["manifest_source_sha256"]) == 64
	assert "imported_at" in provenance


def test_missing_declared_parent_does_not_become_fresh_training_data(catalog):
	with pytest.raises(KeyError):
		recorded(catalog, parent="missing-parent")
	assert catalog.summary()["episodes"] == 0


def test_memberships_frozen_at_submission(catalog):
	ident = recorded(catalog)
	jobs = Jobs(catalog)
	job = jobs.submit([ident])
	catalog.freeze_split([ident], "after-submission")
	result = jobs.execute(job["id"])
	manifest = json.loads((Path(result["result"]["bundle"]).parent / "manifest.json").read_text(encoding="utf-8"))
	assert manifest["memberships"][ident] == []


def test_metadata_can_relocate_with_the_project(catalog):
	import shutil

	ident = report(catalog)
	destination = catalog.root.parent / (catalog.root.name + "_moved")
	shutil.copytree(catalog.root, destination)
	relocated = Catalog(destination)
	assert relocated.verify([ident])["ok"]
	assert relocated.import_file(destination / "run.json") == [ident]


def test_unconfigured_or_unverified_pinch_peak_is_unknown():
	checks = evaluate_quality({"pinch_normal_peak_N": 10}, {}, {"pinch_force_N": 20})
	assert checks[4]["status"] == "UNKNOWN"
	checks = evaluate_quality({"recorded_substep_pinch_normal_peak_N": 10}, {}, {"pinch_force_N": 20})
	assert checks[4]["status"] == "PASS"


def test_snapshot_requires_verified_publication(catalog):
	from digital_twin_armigo.runtime.ledger import JobState, OperationLedger, job_resource
	from digital_twin_armigo.runtime.snapshot import ExperimentIdentity, SnapshotStore

	store = SnapshotStore(catalog.root / "snapshots")
	ledger = OperationLedger(catalog.root / "runtime.sqlite")
	job, _ = ledger.register(
		idempotency_key="snapshot-test", op_type="create_snapshot", params={}, run_id="run", branch_id="main"
	)
	lease = ledger.acquire_lease(job_resource(job.job_id), "test")
	sid = "1" * 32
	ledger.append_event(job.job_id, JobState.RUNNING, lease=lease, payload={"output_snapshot_id": sid})
	identity = ExperimentIdentity(
		code_sha256="a" * 64,
		assets_sha256="b" * 64,
		config_sha256="c" * 64,
		engine="mujoco",
		engine_version="4",
		python_version="3.14",
		numpy_version="2",
		platform="test",
	)
	store.commit(
		snapshot_id=sid,
		run_id="run",
		episode_id="episode",
		branch_id="main",
		control_step=0,
		sim_time_s=0,
		identity=identity,
		model_fingerprint="f" * 64,
		contract={},
		scene_spec=geometry(),
		blobs={"arrays/integration": store.put_arrays({"qpos": np.zeros(6)})},
		produced_by={"job_id": job.job_id, "fencing_token": lease.fencing_token, "result": {}},
	)
	with pytest.raises(ValueError, match="published"):
		catalog.import_snapshot(store.root, catalog.root / "runtime.sqlite", sid)
	ledger.append_event(job.job_id, JobState.SUCCEEDED, lease=lease, payload={"output_snapshot_id": sid})
	result = catalog.import_snapshot(store.root, catalog.root / "runtime.sqlite", sid)
	assert result["verified"]
	assert catalog.episode(result["id"])["provenance"]["original_code_identity"]["code_sha256"] == "a" * 64
	ledger.close()


def test_cancel_before_execution_never_publishes(catalog):
	jobs = Jobs(catalog)
	job = jobs.submit([report(catalog)])
	jobs.ledger.request_cancel(job["id"])
	assert jobs.execute(job["id"])["state"] == "CANCELLED"
	assert not (catalog.home / "exports" / job["id"] / "bundle.zip").exists()
