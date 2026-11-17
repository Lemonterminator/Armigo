"""Content-addressed, immutable snapshot store: one committed manifest plus SHA-256 addressed blobs."""

from __future__ import annotations

import dataclasses
import fnmatch
import io
import json
import os
import platform as _platform
import re
import uuid
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from digital_twin_armigo.constants import REPO_ROOT
from digital_twin_armigo.runtime.canonical import canonical_json, sha256_bytes, sha256_file, sha256_json
from digital_twin_armigo.runtime.errors import (
	IdentityMismatchError,
	SnapshotCorruptError,
	SnapshotError,
	SnapshotNotFoundError,
)

SNAPSHOT_SCHEMA = "snapshot-manifest/v1"
_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_BLOB_KINDS = ("arrays", "json")


@dataclass(frozen=True)
class ExperimentIdentity:
	"""Everything that must match for a snapshot to be resumed directly."""

	code_sha256: str
	assets_sha256: str
	config_sha256: str
	engine: str
	engine_version: str
	python_version: str
	numpy_version: str
	platform: str

	def to_dict(self) -> dict[str, str]:
		return dataclasses.asdict(self)

	@classmethod
	def from_dict(cls, d: Mapping[str, Any]) -> ExperimentIdentity:
		names = {f.name for f in dataclasses.fields(cls)}
		if not isinstance(d, Mapping) or set(d) != names:
			raise SnapshotCorruptError("identity has missing or unexpected fields")
		if not all(isinstance(v, str) for v in d.values()):
			raise SnapshotCorruptError("identity fields must be strings")
		return cls(**{k: d[k] for k in names})

	def mismatches(self, other: ExperimentIdentity) -> list[str]:
		"""Sorted names of the fields that differ from ``other``."""
		a, b = self.to_dict(), other.to_dict()
		return sorted(k for k in a if a[k] != b[k])


def hash_tree(root: Path, patterns: Sequence[str], exclude: Sequence[str] = ()) -> str:
	"""Deterministic SHA-256 over the files under ``root`` matching ``patterns``."""
	root = Path(root)
	if not root.is_dir():
		raise FileNotFoundError(f"hash_tree root not found: {root}")
	found: dict[str, Path] = {}
	for pattern in patterns:
		for path in root.glob(pattern):
			if not path.is_file():
				continue
			rel = path.relative_to(root).as_posix()
			if "__pycache__" in rel.split("/"):
				continue
			if any(fnmatch.fnmatch(rel, ex) for ex in exclude):
				continue
			found[rel] = path
	return sha256_json([[rel, sha256_file(found[rel])] for rel in sorted(found)])


def current_identity(engine: str, engine_version: str, *, repo_root: Path = REPO_ROOT) -> ExperimentIdentity:
	"""Identity of the code, assets, config, and environment in the working tree."""
	assets = {
		"so101_mjcf": hash_tree(
			repo_root / "assets" / "so101_mjcf", ["**/*"], exclude=["_sorting_cell_*.xml", "_marker_*.png"]
		),
		"meshes": hash_tree(repo_root / "assets" / "meshes", ["**/*"]),
	}
	return ExperimentIdentity(
		code_sha256=hash_tree(repo_root / "src" / "digital_twin_armigo", ["**/*.py"]),
		assets_sha256=sha256_json(assets),
		config_sha256=hash_tree(repo_root / "config" / "sim", ["**/*"]),
		engine=engine,
		engine_version=engine_version,
		python_version=_platform.python_version(),
		numpy_version=np.__version__,
		platform=_platform.platform(),
	)


@dataclass(frozen=True)
class BlobRef:
	"""Pointer to an immutable blob."""

	sha256: str
	size: int
	kind: str

	def to_dict(self) -> dict[str, Any]:
		return {"sha256": self.sha256, "size": self.size, "kind": self.kind}

	@classmethod
	def from_dict(cls, d: Mapping[str, Any]) -> BlobRef:
		if not isinstance(d, Mapping) or set(d) != {"sha256", "size", "kind"}:
			raise SnapshotCorruptError("blob ref has missing or unexpected fields")
		sha, size, kind = d["sha256"], d["size"], d["kind"]
		if not (isinstance(sha, str) and _SHA_RE.match(sha)):
			raise SnapshotCorruptError("blob ref sha256 is malformed")
		if not isinstance(size, int) or isinstance(size, bool) or size < 0:
			raise SnapshotCorruptError("blob ref size is malformed")
		if kind not in _BLOB_KINDS:
			raise SnapshotCorruptError(f"blob ref kind is unknown: {kind!r}")
		return cls(sha256=sha, size=size, kind=kind)


def _is_int(v: Any) -> bool:
	return isinstance(v, int) and not isinstance(v, bool)


@dataclass(frozen=True)
class SnapshotManifest:
	"""Committed description of one snapshot; self-checking via three SHA-256 fields."""

	schema: str
	snapshot_id: str
	parent_snapshot_id: str | None
	run_id: str
	episode_id: str
	branch_id: str
	created_at_utc: str
	control_step: int
	sim_time_s: float
	identity: dict[str, str]
	model_fingerprint: str
	contract: dict[str, Any]
	contract_sha256: str
	scene_spec: dict[str, Any]
	scene_sha256: str
	blobs: dict[str, BlobRef]
	produced_by: dict[str, Any] | None
	manifest_sha256: str

	def body(self) -> dict[str, Any]:
		"""JSON-able dict of every field except ``manifest_sha256``."""
		out: dict[str, Any] = {}
		for f in dataclasses.fields(self):
			if f.name == "manifest_sha256":
				continue
			value = getattr(self, f.name)
			out[f.name] = {k: v.to_dict() for k, v in value.items()} if f.name == "blobs" else value
		return out

	def to_dict(self) -> dict[str, Any]:
		return {**self.body(), "manifest_sha256": self.manifest_sha256}

	@classmethod
	def build(cls, **fields: Any) -> SnapshotManifest:
		"""Construct a manifest, computing the three hashes from the other fields."""
		fields["contract_sha256"] = sha256_json(fields["contract"])
		fields["scene_sha256"] = sha256_json(fields["scene_spec"])
		draft = cls(manifest_sha256="", **fields)
		return dataclasses.replace(draft, manifest_sha256=sha256_json(draft.body()))

	@classmethod
	def from_dict(cls, d: Mapping[str, Any]) -> SnapshotManifest:
		"""Strict parse; any structural or hash inconsistency raises ``SnapshotCorruptError``."""
		names = {f.name for f in dataclasses.fields(cls)}
		if not isinstance(d, Mapping) or set(d) != names:
			raise SnapshotCorruptError("manifest has missing or unexpected keys")
		if d["schema"] != SNAPSHOT_SCHEMA:
			raise SnapshotCorruptError(f"unknown manifest schema: {d['schema']!r}")
		for key in ("snapshot_id", "run_id", "episode_id", "branch_id", "created_at_utc", "model_fingerprint"):
			if not isinstance(d[key], str):
				raise SnapshotCorruptError(f"manifest field {key} must be a string")
		for key in ("contract_sha256", "scene_sha256", "manifest_sha256"):
			if not isinstance(d[key], str):
				raise SnapshotCorruptError(f"manifest field {key} must be a string")
		parent = d["parent_snapshot_id"]
		if parent is not None and not isinstance(parent, str):
			raise SnapshotCorruptError("parent_snapshot_id must be a string or null")
		if not _is_int(d["control_step"]):
			raise SnapshotCorruptError("control_step must be an integer")
		sim_time = d["sim_time_s"]
		if isinstance(sim_time, bool) or not isinstance(sim_time, (int, float)):
			raise SnapshotCorruptError("sim_time_s must be a number")
		for key in ("identity", "contract", "scene_spec", "blobs"):
			if not isinstance(d[key], dict):
				raise SnapshotCorruptError(f"manifest field {key} must be an object")
		produced_by = d["produced_by"]
		if produced_by is not None and not isinstance(produced_by, dict):
			raise SnapshotCorruptError("produced_by must be an object or null")
		identity = ExperimentIdentity.from_dict(d["identity"]).to_dict()
		blobs = {str(k): BlobRef.from_dict(v) for k, v in d["blobs"].items()}
		manifest = cls(
			schema=d["schema"],
			snapshot_id=d["snapshot_id"],
			parent_snapshot_id=parent,
			run_id=d["run_id"],
			episode_id=d["episode_id"],
			branch_id=d["branch_id"],
			created_at_utc=d["created_at_utc"],
			control_step=d["control_step"],
			sim_time_s=sim_time,
			identity=identity,
			model_fingerprint=d["model_fingerprint"],
			contract=d["contract"],
			contract_sha256=d["contract_sha256"],
			scene_spec=d["scene_spec"],
			scene_sha256=d["scene_sha256"],
			blobs=blobs,
			produced_by=produced_by,
			manifest_sha256=d["manifest_sha256"],
		)
		try:
			if manifest.contract_sha256 != sha256_json(manifest.contract):
				raise SnapshotCorruptError("contract_sha256 does not match contract")
			if manifest.scene_sha256 != sha256_json(manifest.scene_spec):
				raise SnapshotCorruptError("scene_sha256 does not match scene_spec")
			if manifest.manifest_sha256 != sha256_json(manifest.body()):
				raise SnapshotCorruptError("manifest_sha256 does not match manifest body")
		except ValueError as exc:
			raise SnapshotCorruptError(f"manifest is not canonically encodable: {exc}") from exc
		return manifest


def _fsync_write(path: Path, data: bytes) -> None:
	with path.open("wb") as fh:
		fh.write(data)
		fh.flush()
		os.fsync(fh.fileno())


class SnapshotStore:
	"""Local-filesystem store of immutable blobs and committed manifests."""

	def __init__(self, root: Path) -> None:
		self.root = Path(root)
		self.blobs_dir = self.root / "blobs"
		self.manifests_dir = self.root / "manifests"
		self.staging_dir = self.root / "staging"
		for d in (self.blobs_dir, self.manifests_dir, self.staging_dir):
			d.mkdir(parents=True, exist_ok=True)

	# -- blobs ---------------------------------------------------------------------------------
	def _put_bytes(self, data: bytes, kind: str) -> BlobRef:
		ref = BlobRef(sha256=sha256_bytes(data), size=len(data), kind=kind)
		dest = self.blobs_dir / ref.sha256
		tmp = self.staging_dir / f"blob-{uuid.uuid4().hex}.tmp"
		try:
			_fsync_write(tmp, data)
			if sha256_file(tmp) != ref.sha256:
				raise SnapshotError("blob changed between write and re-read")
			if dest.is_file() and dest.stat().st_size == ref.size:
				return ref
			os.replace(tmp, dest)
		finally:
			tmp.unlink(missing_ok=True)
		return ref

	def put_arrays(self, arrays: Mapping[str, np.ndarray]) -> BlobRef:
		"""Store named arrays as a deterministic, pickle-free ZIP of ``.npy`` entries."""
		for key, arr in arrays.items():
			if not isinstance(key, str) or not key:
				raise SnapshotError(f"array keys must be non-empty strings, got {key!r}")
			if np.asarray(arr).dtype.hasobject:
				raise SnapshotError(f"array {key!r} has object dtype")
		buf = io.BytesIO()
		with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
			for key in sorted(arrays):
				entry = io.BytesIO()
				np.lib.format.write_array(entry, np.asarray(arrays[key], order="C"), allow_pickle=False)
				info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
				info.compress_type = zipfile.ZIP_STORED
				zf.writestr(info, entry.getvalue())
		return self._put_bytes(buf.getvalue(), "arrays")

	def put_json(self, obj: Any) -> BlobRef:
		"""Store ``obj`` as canonical JSON."""
		try:
			data = canonical_json(obj)
		except (ValueError, TypeError) as exc:
			raise SnapshotError(f"object is not canonically JSON-encodable: {exc}") from exc
		return self._put_bytes(data, "json")

	def _read_verified(self, ref: BlobRef) -> bytes:
		path = self.blobs_dir / ref.sha256
		try:
			data = path.read_bytes()
		except OSError as exc:
			raise SnapshotCorruptError(f"blob {ref.sha256} is missing or unreadable") from exc
		if len(data) != ref.size:
			raise SnapshotCorruptError(f"blob {ref.sha256} has size {len(data)}, expected {ref.size}")
		if sha256_bytes(data) != ref.sha256:
			raise SnapshotCorruptError(f"blob {ref.sha256} fails its SHA-256 check")
		return data

	def get_arrays(self, ref: BlobRef) -> dict[str, np.ndarray]:
		if ref.kind != "arrays":
			raise SnapshotCorruptError(f"blob {ref.sha256} is kind {ref.kind!r}, not 'arrays'")
		data = self._read_verified(ref)
		try:
			with np.load(io.BytesIO(data), allow_pickle=False) as npz:
				return {name: npz[name] for name in npz.files}
		except Exception as exc:
			raise SnapshotCorruptError(f"blob {ref.sha256} cannot be decoded as arrays: {exc}") from exc

	def get_json(self, ref: BlobRef) -> Any:
		if ref.kind != "json":
			raise SnapshotCorruptError(f"blob {ref.sha256} is kind {ref.kind!r}, not 'json'")
		data = self._read_verified(ref)
		try:
			return json.loads(data.decode("utf-8"))
		except (UnicodeDecodeError, ValueError) as exc:
			raise SnapshotCorruptError(f"blob {ref.sha256} cannot be decoded as JSON: {exc}") from exc

	# -- manifests -----------------------------------------------------------------------------
	def _manifest_path(self, snapshot_id: str) -> Path:
		if not _ID_RE.match(snapshot_id):
			raise SnapshotError(f"invalid snapshot id: {snapshot_id!r}")
		return self.manifests_dir / f"{snapshot_id}.json"

	def commit(
		self,
		*,
		run_id: str,
		episode_id: str,
		branch_id: str,
		control_step: int,
		sim_time_s: float,
		identity: ExperimentIdentity,
		model_fingerprint: str,
		contract: dict[str, Any],
		scene_spec: dict[str, Any],
		blobs: Mapping[str, BlobRef],
		parent_snapshot_id: str | None = None,
		produced_by: Mapping[str, Any] | None = None,
		snapshot_id: str | None = None,
	) -> SnapshotManifest:
		"""Atomically commit a new immutable manifest; the final rename is the commit point."""
		sid = uuid.uuid4().hex if snapshot_id is None else snapshot_id
		path = self._manifest_path(sid)
		if path.exists():
			raise SnapshotError(f"snapshot {sid} already exists; manifests are immutable")
		for ref in blobs.values():
			self._read_verified(ref)
		if parent_snapshot_id is not None:
			self.load_manifest(parent_snapshot_id)
		try:
			manifest = SnapshotManifest.build(
				schema=SNAPSHOT_SCHEMA,
				snapshot_id=sid,
				parent_snapshot_id=parent_snapshot_id,
				run_id=run_id,
				episode_id=episode_id,
				branch_id=branch_id,
				created_at_utc=datetime.now(UTC).isoformat(),
				control_step=int(control_step),
				sim_time_s=float(sim_time_s),
				identity=identity.to_dict(),
				model_fingerprint=model_fingerprint,
				contract=contract,
				scene_spec=scene_spec,
				blobs=dict(blobs),
				produced_by=None if produced_by is None else dict(produced_by),
			)
			text = json.dumps(manifest.to_dict(), indent=2, sort_keys=True, allow_nan=False)
		except (ValueError, TypeError) as exc:
			raise SnapshotError(f"manifest is not JSON-encodable: {exc}") from exc
		tmp = self.staging_dir / f"manifest-{sid}-{uuid.uuid4().hex}.tmp"
		try:
			_fsync_write(tmp, text.encode("utf-8"))
			# A hard link is the commit point: unlike os.replace it fails when the id already exists, so two
			# writers that both passed the existence check above cannot overwrite each other.
			try:
				os.link(tmp, path)
			except FileExistsError as exc:
				raise SnapshotError(f"snapshot {sid} already exists; manifests are immutable") from exc
		finally:
			tmp.unlink(missing_ok=True)
		return manifest

	def load_manifest(self, snapshot_id: str) -> SnapshotManifest:
		path = self._manifest_path(snapshot_id)
		if not path.is_file():
			raise SnapshotNotFoundError(f"no committed snapshot {snapshot_id}")
		try:
			raw = json.loads(path.read_text("utf-8"))
		except (OSError, ValueError) as exc:
			raise SnapshotCorruptError(f"manifest {snapshot_id} is unreadable: {exc}") from exc
		manifest = SnapshotManifest.from_dict(raw)
		if manifest.snapshot_id != snapshot_id:
			raise SnapshotCorruptError(f"manifest file {snapshot_id} embeds id {manifest.snapshot_id}")
		return manifest

	def verify(self, snapshot_id: str) -> SnapshotManifest:
		"""Load the manifest and verify size and SHA-256 of every referenced blob."""
		manifest = self.load_manifest(snapshot_id)
		for ref in manifest.blobs.values():
			self._read_verified(ref)
		return manifest

	def fork(self, snapshot_id: str, *, branch_id: str, new_snapshot_id: str | None = None) -> SnapshotManifest:
		"""Commit a child of ``snapshot_id`` on a new branch; the source is never touched."""
		src = self.verify(snapshot_id)
		if branch_id == src.branch_id:
			raise SnapshotError(f"fork must use a new branch, not {branch_id!r}")
		return self.commit(
			run_id=src.run_id,
			episode_id=src.episode_id,
			branch_id=branch_id,
			control_step=src.control_step,
			sim_time_s=src.sim_time_s,
			identity=ExperimentIdentity.from_dict(src.identity),
			model_fingerprint=src.model_fingerprint,
			contract=src.contract,
			scene_spec=src.scene_spec,
			blobs=src.blobs,
			parent_snapshot_id=snapshot_id,
			produced_by=None,
			snapshot_id=new_snapshot_id,
		)

	def list_snapshot_ids(self) -> list[str]:
		"""Sorted ids of committed manifests only."""
		return sorted(p.stem for p in self.manifests_dir.glob("*.json") if _ID_RE.match(p.stem))

	def lineage(self, snapshot_id: str) -> list[str]:
		"""Ids from ``snapshot_id`` back to its root."""
		chain: list[str] = []
		current: str | None = snapshot_id
		while current is not None:
			if current in chain:
				raise SnapshotCorruptError(f"lineage cycle at {current}")
			chain.append(current)
			current = self.load_manifest(current).parent_snapshot_id
		return chain

	def check_identity(
		self, manifest: SnapshotManifest, current: ExperimentIdentity, model_fingerprint: str | None = None
	) -> None:
		"""Raise ``IdentityMismatchError`` naming every differing field."""
		bad = ExperimentIdentity.from_dict(manifest.identity).mismatches(current)
		if model_fingerprint is not None and model_fingerprint != manifest.model_fingerprint:
			bad.append("model_fingerprint")
		if bad:
			raise IdentityMismatchError(f"snapshot {manifest.snapshot_id} identity mismatch: {', '.join(bad)}")
