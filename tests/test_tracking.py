"""Dataset manifests and MLflow run helpers."""

from __future__ import annotations

import json

import pytest

from digital_twin_armigo.tracking import lineage, runs


def _make_dataset(root):
	(root / "sub").mkdir(parents=True)
	(root / "a.txt").write_text("alpha", encoding="utf-8")
	(root / "sub" / "b.txt").write_text("beta", encoding="utf-8")
	return root


def test_manifest_digest_tracks_content(tmp_path):
	ds = _make_dataset(tmp_path / "ds")
	first = lineage.build_manifest(ds, name="toy", kind="test", params={"seed": 1})
	assert first["file_count"] == 2 and first["total_bytes"] == 9
	assert (ds / lineage.MANIFEST_NAME).is_file()
	# Rebuilding unchanged data is stable, keeps the creation stamp and the recorded parameters.
	again = lineage.build_manifest(ds)
	assert again["digest"] == first["digest"] and again["created_at"] == first["created_at"]
	assert again["params"] == {"seed": 1} and again["name"] == "toy" and again["kind"] == "test"
	# A same-size edit changes the digest even though size is unchanged.
	(ds / "a.txt").write_text("ALPHA", encoding="utf-8")
	assert lineage.build_manifest(ds)["digest"] != first["digest"]


def test_verify_reports_changed_missing_and_unlisted(tmp_path):
	ds = _make_dataset(tmp_path / "ds")
	lineage.build_manifest(ds)
	assert lineage.verify_manifest(ds)["ok"]
	(ds / "a.txt").write_text("changed!", encoding="utf-8")
	(ds / "sub" / "b.txt").unlink()
	(ds / "new.txt").write_text("x", encoding="utf-8")
	result = lineage.verify_manifest(ds)
	assert not result["ok"]
	assert result["changed"] == ["a.txt"] and result["missing"] == ["sub/b.txt"] and result["unlisted"] == ["new.txt"]


def test_dataset_ref_refreshes_stale_manifest(tmp_path):
	ds = _make_dataset(tmp_path / "ds")
	old = lineage.dataset_ref(ds)
	(ds / "sub" / "b.txt").write_text("BETA!", encoding="utf-8")
	assert lineage.dataset_ref(ds)["digest"] != old["digest"]


def test_manifest_requires_existing_directory(tmp_path):
	with pytest.raises(FileNotFoundError):
		lineage.build_manifest(tmp_path / "missing")


def test_flatten_numbers_keeps_only_finite_numbers():
	flat = runs.flatten_numbers(
		{"auroc": 0.9, "at": {"recall": 1, "bad": float("nan")}, "ok": True, "name": "x", "xs": [1, 2], "a=b": 2}
	)
	assert flat == {"auroc": 0.9, "at.recall": 1.0, "a_b": 2.0}
	assert runs.flatten_numbers({"m": 1}, "val") == {"val.m": 1.0}


def test_disabled_tracking_is_a_noop(monkeypatch):
	monkeypatch.setattr(runs, "_import_mlflow", lambda: pytest.fail("mlflow must not be touched"))
	with runs.tracked_run("exp", enabled=False) as run:
		run.log_params({"a": 1})
		run.log_report({"x": 1})
		run.register_model("m", "model")
		assert run.run_id is None


def test_env_switch_disables_tracking(monkeypatch):
	monkeypatch.setenv("ARMIGO_MLFLOW", "0")
	monkeypatch.setattr(runs, "_import_mlflow", lambda: pytest.fail("mlflow must not be touched"))
	with runs.tracked_run("exp") as run:
		assert run.run_id is None


def test_store_location_is_overridable(monkeypatch, tmp_path):
	monkeypatch.setenv("ARMIGO_MLFLOW_DIR", str(tmp_path / "store"))
	monkeypatch.delenv("ARMIGO_MLFLOW_URI", raising=False)
	assert runs.tracking_uri() == f"sqlite:///{(tmp_path / 'store' / 'mlflow.db').as_posix()}"
	monkeypatch.setenv("ARMIGO_MLFLOW_URI", "http://tracking.example:5000")
	assert runs.tracking_uri() == "http://tracking.example:5000"


def test_real_run_records_lineage_and_registers_model(monkeypatch, tmp_path):
	mlflow = pytest.importorskip("mlflow")
	if not hasattr(mlflow, "start_run"):
		pytest.skip("mlflow package not installed (only the ./mlflow store directory resolved)")
	monkeypatch.setenv("ARMIGO_MLFLOW_DIR", str(tmp_path / "store"))
	monkeypatch.delenv("ARMIGO_MLFLOW_URI", raising=False)
	monkeypatch.delenv("ARMIGO_MLFLOW", raising=False)
	ds = _make_dataset(tmp_path / "ds")
	weights = tmp_path / "weights"
	weights.mkdir()
	(weights / "head.pt").write_bytes(b"\x00\x01")
	with runs.tracked_run(
		"unit", "r1", params={"lr": 0.1, "none": None}, tags={"task": "t"}, datasets={"train": ds}
	) as run:
		run.log_report({"val": {"f1": 0.5}})
		run.log_artifact(weights, "model")
		run.register_model("unit-model", "model", alias="candidate")
		run_id = run.run_id
	client = mlflow.MlflowClient(runs.tracking_uri())
	got = client.get_run(run_id)
	assert got.data.params == {"lr": "0.1"}
	assert got.data.metrics["val.f1"] == 0.5
	assert got.data.tags["dataset.train.digest"] == lineage.dataset_ref(ds)["digest"]
	assert "git.commit" in got.data.tags and got.data.tags["task"] == "t"
	version = client.get_model_version_by_alias("unit-model", "candidate")
	assert version.run_id == run_id
	json.dumps(got.data.tags)  # tags are plain strings
