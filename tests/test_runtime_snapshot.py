"""Tests for the content-addressed snapshot store."""

from __future__ import annotations

import dataclasses
import json
import os
import re
from pathlib import Path

import numpy as np
import pytest

from digital_twin_armigo.runtime import snapshot as snap
from digital_twin_armigo.runtime.errors import (
	IdentityMismatchError,
	SnapshotCorruptError,
	SnapshotError,
	SnapshotNotFoundError,
)
from digital_twin_armigo.runtime.snapshot import (
	BlobRef,
	ExperimentIdentity,
	SnapshotManifest,
	SnapshotStore,
	current_identity,
	hash_tree,
)

HEX64 = re.compile(r"^[0-9a-f]{64}$")


def make_identity(**over: str) -> ExperimentIdentity:
	base = {
		"code_sha256": "a" * 64,
		"assets_sha256": "b" * 64,
		"config_sha256": "c" * 64,
		"engine": "mujoco",
		"engine_version": "3.0",
		"python_version": "3.12.0",
		"numpy_version": "2.0.0",
		"platform": "test-os",
	}
	base.update(over)
	return ExperimentIdentity(**base)


@pytest.fixture
def store(tmp_path: Path) -> SnapshotStore:
	return SnapshotStore(tmp_path / "store")


def do_commit(store: SnapshotStore, **over):
	arrays = store.put_arrays({"qpos": np.arange(4.0), "qvel": np.zeros(3)})
	meta = store.put_json({"a": 1})
	kwargs = {
		"run_id": "run1",
		"episode_id": "ep1",
		"branch_id": "main",
		"control_step": 7,
		"sim_time_s": 0.7,
		"identity": make_identity(),
		"model_fingerprint": "f" * 64,
		"contract": {"dt": 0.1},
		"scene_spec": {"objects": ["brick"]},
		"blobs": {"state": arrays, "meta": meta},
	}
	kwargs.update(over)
	return store.commit(**kwargs)


def manifest_path(store: SnapshotStore, sid: str) -> Path:
	return store.root / "manifests" / f"{sid}.json"


def blob_path(store: SnapshotStore, ref: BlobRef) -> Path:
	return store.root / "blobs" / ref.sha256


# 1. arrays
def test_arrays_round_trip(store: SnapshotStore) -> None:
	arrays = {
		"f": np.linspace(0, 1, 6).reshape(2, 3),
		"i": np.arange(5, dtype=np.int64),
		"u": np.array([0, 255, 7], dtype=np.uint8),
		"b": np.array([True, False, True]),
		"z": np.array(3.5),
		"e": np.zeros((0, 3)),
	}
	ref = store.put_arrays(arrays)
	assert ref.kind == "arrays"
	out = store.get_arrays(ref)
	assert set(out) == set(arrays)
	for k, v in arrays.items():
		assert out[k].dtype == v.dtype
		assert out[k].shape == v.shape
		assert np.array_equal(out[k], v)
	ref2 = store.put_arrays(arrays)
	assert ref2 == ref
	assert len(list((store.root / "blobs").iterdir())) == 1
	assert list((store.root / "staging").iterdir()) == []


def test_arrays_rejects_bad_input(store: SnapshotStore) -> None:
	with pytest.raises(SnapshotError):
		store.put_arrays({"o": np.array([object()], dtype=object)})
	with pytest.raises(SnapshotError):
		store.put_arrays({"": np.zeros(1)})
	with pytest.raises(SnapshotError):
		store.put_arrays({1: np.zeros(1)})  # type: ignore[dict-item]


# 2. determinism
def test_arrays_deterministic_and_np_loadable(tmp_path: Path) -> None:
	arrays = {"b": np.arange(3.0), "a": np.ones((2, 2), dtype=np.int32)}
	s1 = SnapshotStore(tmp_path / "s1")
	s2 = SnapshotStore(tmp_path / "s2")
	r1 = s1.put_arrays(arrays)
	r2 = s2.put_arrays(dict(reversed(list(arrays.items()))))
	assert r1 == r2
	assert blob_path(s1, r1).read_bytes() == blob_path(s2, r2).read_bytes()
	with np.load(blob_path(s1, r1), allow_pickle=False) as npz:
		assert sorted(npz.files) == ["a", "b"]
		assert np.array_equal(npz["b"], arrays["b"])


# 3. json
def test_json_round_trip_and_nan(store: SnapshotStore) -> None:
	obj = {"x": [1, 2, {"y": "z"}], "n": None}
	ref = store.put_json(obj)
	assert ref.kind == "json"
	assert store.get_json(ref) == obj
	with pytest.raises(SnapshotError):
		store.put_json({"v": float("nan")})
	with pytest.raises(SnapshotError):
		store.put_json({"v": float("inf")})


# 4. commit / load / verify
def test_commit_load_verify(store: SnapshotStore) -> None:
	m = do_commit(store)
	assert re.fullmatch(r"[0-9a-f]{32}", m.snapshot_id)
	assert m.schema == snap.SNAPSHOT_SCHEMA
	loaded = store.load_manifest(m.snapshot_id)
	assert loaded == m
	assert store.verify(m.snapshot_id) == m
	assert m.manifest_sha256 == snap.sha256_json(m.body())
	assert m.contract_sha256 == snap.sha256_json(m.contract)
	assert m.scene_sha256 == snap.sha256_json(m.scene_spec)
	assert "manifest_sha256" not in m.body()
	assert m.to_dict()["manifest_sha256"] == m.manifest_sha256
	assert SnapshotManifest.from_dict(m.to_dict()) == m
	assert store.list_snapshot_ids() == [m.snapshot_id]
	assert m.parent_snapshot_id is None
	assert m.produced_by is None


def test_commit_rejects_bad_id(store: SnapshotStore) -> None:
	with pytest.raises(SnapshotError):
		do_commit(store, snapshot_id="NOT-HEX")


def test_identity_roundtrip_and_strictness() -> None:
	ident = make_identity()
	assert ExperimentIdentity.from_dict(ident.to_dict()) == ident
	d = ident.to_dict()
	d["extra"] = "x"
	with pytest.raises(SnapshotCorruptError):
		ExperimentIdentity.from_dict(d)
	d = ident.to_dict()
	del d["engine"]
	with pytest.raises(SnapshotCorruptError):
		ExperimentIdentity.from_dict(d)
	d = ident.to_dict()
	d["engine"] = 3  # type: ignore[assignment]
	with pytest.raises(SnapshotCorruptError):
		ExperimentIdentity.from_dict(d)
	assert ident.mismatches(make_identity(engine="x", code_sha256="0" * 64)) == ["code_sha256", "engine"]


# 5. corruption
def test_corrupt_flipped_byte(store: SnapshotStore) -> None:
	m = do_commit(store)
	p = blob_path(store, m.blobs["state"])
	data = bytearray(p.read_bytes())
	data[len(data) // 2] ^= 0xFF
	p.write_bytes(bytes(data))
	with pytest.raises(SnapshotCorruptError):
		store.get_arrays(m.blobs["state"])
	with pytest.raises(SnapshotCorruptError):
		store.verify(m.snapshot_id)


def test_corrupt_truncated(store: SnapshotStore) -> None:
	m = do_commit(store)
	p = blob_path(store, m.blobs["meta"])
	p.write_bytes(p.read_bytes()[:-2])
	with pytest.raises(SnapshotCorruptError):
		store.get_json(m.blobs["meta"])
	with pytest.raises(SnapshotCorruptError):
		store.verify(m.snapshot_id)


def test_corrupt_deleted_blob(store: SnapshotStore) -> None:
	m = do_commit(store)
	blob_path(store, m.blobs["state"]).unlink()
	with pytest.raises(SnapshotCorruptError):
		store.get_arrays(m.blobs["state"])
	with pytest.raises(SnapshotCorruptError):
		store.verify(m.snapshot_id)


def test_corrupt_manifest_edited(store: SnapshotStore) -> None:
	m = do_commit(store)
	p = manifest_path(store, m.snapshot_id)
	d = json.loads(p.read_text("utf-8"))
	d["control_step"] = 999
	p.write_text(json.dumps(d), "utf-8")
	with pytest.raises(SnapshotCorruptError):
		store.load_manifest(m.snapshot_id)


def test_corrupt_manifest_renamed(store: SnapshotStore) -> None:
	m = do_commit(store)
	other = "0" * 32
	os.replace(manifest_path(store, m.snapshot_id), manifest_path(store, other))
	with pytest.raises(SnapshotCorruptError):
		store.load_manifest(other)


def test_corrupt_manifest_not_json(store: SnapshotStore) -> None:
	m = do_commit(store)
	manifest_path(store, m.snapshot_id).write_text("{not json", "utf-8")
	with pytest.raises(SnapshotCorruptError):
		store.load_manifest(m.snapshot_id)


def test_corrupt_unknown_schema(store: SnapshotStore) -> None:
	m = do_commit(store)
	d = m.to_dict()
	d["schema"] = "snapshot-manifest/v999"
	with pytest.raises(SnapshotCorruptError):
		SnapshotManifest.from_dict(d)
	d = m.to_dict()
	d["unknown"] = 1
	with pytest.raises(SnapshotCorruptError):
		SnapshotManifest.from_dict(d)
	d = m.to_dict()
	del d["run_id"]
	with pytest.raises(SnapshotCorruptError):
		SnapshotManifest.from_dict(d)
	d = m.to_dict()
	d["contract"] = {"dt": 0.2}
	with pytest.raises(SnapshotCorruptError):
		SnapshotManifest.from_dict(d)


def test_wrong_kind_rejected(store: SnapshotStore) -> None:
	ref = store.put_json({"a": 1})
	with pytest.raises(SnapshotCorruptError):
		store.get_arrays(ref)
	arr = store.put_arrays({"a": np.zeros(2)})
	with pytest.raises(SnapshotCorruptError):
		store.get_json(arr)
	with pytest.raises(SnapshotCorruptError):
		store.get_json(dataclasses.replace(arr, kind="json"))


# 6. uncommitted invisible
def test_staging_manifest_invisible(store: SnapshotStore) -> None:
	m = do_commit(store)
	sid = "1" * 32
	(store.root / "staging" / f"{sid}.json").write_text(manifest_path(store, m.snapshot_id).read_text("utf-8"), "utf-8")
	assert store.list_snapshot_ids() == [m.snapshot_id]
	with pytest.raises(SnapshotNotFoundError):
		store.load_manifest(sid)


def test_crash_before_rename(store: SnapshotStore, monkeypatch: pytest.MonkeyPatch) -> None:
	sid = "2" * 32
	real = os.link

	def boom(src, dst, *a, **kw):
		if Path(dst).parent.name == "manifests":
			raise OSError("simulated crash")
		return real(src, dst, *a, **kw)

	monkeypatch.setattr(os, "link", boom)
	with pytest.raises(OSError):
		do_commit(store, snapshot_id=sid)
	monkeypatch.setattr(os, "link", real)
	assert store.list_snapshot_ids() == []
	with pytest.raises(SnapshotNotFoundError):
		store.load_manifest(sid)
	m = do_commit(store, snapshot_id=sid)
	assert store.load_manifest(sid) == m
	assert store.list_snapshot_ids() == [sid]


def test_commit_is_exclusive_when_another_writer_wins_the_race(
	store: SnapshotStore, monkeypatch: pytest.MonkeyPatch
) -> None:
	"""Two writers pass the existence check for one id; the commit point itself must refuse the second."""
	sid = "3" * 32
	first = do_commit(store, snapshot_id=sid)
	before = manifest_path(store, sid).read_bytes()
	monkeypatch.setattr(Path, "exists", lambda self: False)  # the loser's stale "not there yet" view
	with pytest.raises(SnapshotError):
		do_commit(store, snapshot_id=sid, control_step=999)
	monkeypatch.undo()
	assert manifest_path(store, sid).read_bytes() == before
	assert store.load_manifest(sid) == first
	assert list(store.staging_dir.iterdir()) == []


# 7. immutability
def test_commit_existing_id_raises(store: SnapshotStore) -> None:
	m = do_commit(store)
	p = manifest_path(store, m.snapshot_id)
	before = p.read_bytes()
	with pytest.raises(SnapshotError):
		do_commit(store, snapshot_id=m.snapshot_id, control_step=8)
	assert p.read_bytes() == before


def test_commit_missing_blob_or_parent(store: SnapshotStore) -> None:
	ghost = BlobRef(sha256="e" * 64, size=3, kind="json")
	with pytest.raises(SnapshotCorruptError):
		do_commit(store, blobs={"x": ghost})
	with pytest.raises(SnapshotNotFoundError):
		do_commit(store, parent_snapshot_id="3" * 32)
	assert store.list_snapshot_ids() == []


def test_commit_verifies_blob_hash(store: SnapshotStore) -> None:
	ref = store.put_json({"a": 1})
	p = blob_path(store, ref)
	data = bytearray(p.read_bytes())
	data[0] ^= 0x01
	p.write_bytes(bytes(data))
	with pytest.raises(SnapshotCorruptError):
		do_commit(store, blobs={"x": ref})


# 8. fork
def test_fork(store: SnapshotStore) -> None:
	src = do_commit(store, produced_by={"op": "step"})
	p = manifest_path(store, src.snapshot_id)
	before = p.read_bytes()
	fk = store.fork(src.snapshot_id, branch_id="alt")
	assert fk.snapshot_id != src.snapshot_id
	assert fk.parent_snapshot_id == src.snapshot_id
	assert fk.branch_id == "alt"
	assert fk.blobs == src.blobs
	assert fk.produced_by is None
	assert fk.control_step == src.control_step
	assert fk.identity == src.identity
	assert fk.contract == src.contract
	assert p.read_bytes() == before
	assert store.lineage(fk.snapshot_id) == [fk.snapshot_id, src.snapshot_id]
	assert store.lineage(src.snapshot_id) == [src.snapshot_id]
	assert store.verify(fk.snapshot_id) == fk
	assert store.list_snapshot_ids() == sorted([fk.snapshot_id, src.snapshot_id])
	with pytest.raises(SnapshotError):
		store.fork(src.snapshot_id, branch_id="main")
	with pytest.raises(SnapshotNotFoundError):
		store.fork("4" * 32, branch_id="x")


# 9. identity check
def test_check_identity(store: SnapshotStore) -> None:
	m = do_commit(store)
	store.check_identity(m, make_identity())
	store.check_identity(m, make_identity(), "f" * 64)
	with pytest.raises(IdentityMismatchError, match="code_sha256"):
		store.check_identity(m, make_identity(code_sha256="9" * 64))
	with pytest.raises(IdentityMismatchError, match="engine_version"):
		store.check_identity(m, make_identity(engine_version="4.0"))
	with pytest.raises(IdentityMismatchError, match="model_fingerprint"):
		store.check_identity(m, make_identity(), "0" * 64)
	with pytest.raises(IdentityMismatchError) as ei:
		store.check_identity(m, make_identity(engine="x", config_sha256="0" * 64))
	assert "config_sha256" in str(ei.value)
	assert "engine" in str(ei.value)


# 10. hash_tree
def _tree(root: Path) -> None:
	(root / "sub").mkdir(parents=True)
	(root / "a.py").write_text("a", "utf-8")
	(root / "sub" / "b.py").write_text("b", "utf-8")


def test_hash_tree(tmp_path: Path) -> None:
	root = tmp_path / "t"
	_tree(root)
	h = hash_tree(root, ["**/*.py"])
	assert HEX64.match(h)
	assert hash_tree(root, ["**/*.py"]) == h
	# content change
	(root / "a.py").write_text("changed", "utf-8")
	h2 = hash_tree(root, ["**/*.py"])
	assert h2 != h
	# add
	(root / "c.py").write_text("c", "utf-8")
	h3 = hash_tree(root, ["**/*.py"])
	assert h3 != h2
	# rename
	(root / "c.py").rename(root / "d.py")
	assert hash_tree(root, ["**/*.py"]) != h3


def test_hash_tree_ignores(tmp_path: Path) -> None:
	root = tmp_path / "t"
	_tree(root)
	base = hash_tree(root, ["**/*"])
	(root / "__pycache__").mkdir()
	(root / "__pycache__" / "x.pyc").write_bytes(b"1")
	(root / "sub" / "__pycache__").mkdir()
	(root / "sub" / "__pycache__" / "y.pyc").write_bytes(b"2")
	assert hash_tree(root, ["**/*"]) == base
	(root / "_gen_1.xml").write_text("x", "utf-8")
	assert hash_tree(root, ["**/*"]) != base
	assert hash_tree(root, ["**/*"], exclude=["_gen_*.xml"]) == base
	with pytest.raises(FileNotFoundError):
		hash_tree(tmp_path / "missing", ["**/*"])


def test_current_identity_real_repo() -> None:
	ident = current_identity("mujoco", "x")
	assert ident.engine == "mujoco"
	assert ident.engine_version == "x"
	for name in ("code_sha256", "assets_sha256", "config_sha256"):
		assert HEX64.match(getattr(ident, name))
	assert ident.numpy_version == np.__version__
