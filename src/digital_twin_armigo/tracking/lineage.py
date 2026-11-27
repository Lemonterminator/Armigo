"""Code and dataset lineage for experiment tracking. Standard library only, so every venv can import it.

A *dataset version* is a directory plus a ``dataset_manifest.json`` written next to its files. The manifest pins every
file by size and SHA-256 and folds them into one ``digest``; changing, adding or removing a file changes the digest.
Training and evaluation runs record that digest (and the git commit) so a result can be traced to exact inputs.
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from digital_twin_armigo.runtime.canonical import sha256_file, sha256_json

ROOT = Path(__file__).resolve().parents[3]
MANIFEST_NAME = "dataset_manifest.json"
MANIFEST_SCHEMA = "dataset-manifest/v1"


def _git(root: Path, *args: str) -> str | None:
	try:
		done = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=15, check=True)
	except (OSError, subprocess.SubprocessError):
		return None
	return done.stdout.strip()


def git_state(root: Path = ROOT) -> dict[str, Any]:
	"""Commit, branch and whether tracked files differ from it; ``None`` values when ``root`` is not a git checkout."""
	commit = _git(root, "rev-parse", "HEAD")
	status = _git(root, "status", "--porcelain", "--untracked-files=no") if commit else None
	return {
		"commit": commit,
		"branch": _git(root, "rev-parse", "--abbrev-ref", "HEAD") if commit else None,
		"dirty": None if status is None else bool(status),
	}


def _read_manifest(path: Path) -> dict | None:
	try:
		data = json.loads(path.read_text(encoding="utf-8"))
	except (OSError, ValueError):
		return None
	return data if isinstance(data, dict) and data.get("schema") == MANIFEST_SCHEMA else None


def _files_digest(files: dict[str, dict]) -> str:
	return sha256_json({rel: [f["size"], f["sha256"]] for rel, f in sorted(files.items())})


def build_manifest(
	directory: Path,
	*,
	name: str | None = None,
	kind: str | None = None,
	params: dict[str, Any] | None = None,
	write: bool = True,
	root: Path = ROOT,
) -> dict[str, Any]:
	"""Hash every file under ``directory`` and (re)write ``dataset_manifest.json`` there.

	Files whose size and mtime match the previous manifest reuse its hash, so refreshing a multi-GB dataset after a
	small edit only re-reads what changed. ``name``, ``kind`` and ``params`` are kept from the previous manifest when
	not given again.
	"""
	directory = Path(directory).resolve()
	if not directory.is_dir():
		raise FileNotFoundError(f"dataset directory not found: {directory}")
	manifest_path = directory / MANIFEST_NAME
	previous = _read_manifest(manifest_path) or {}
	old_files = previous.get("files", {})
	files: dict[str, dict] = {}
	for path in sorted(p for p in directory.rglob("*") if p.is_file() and p.name != MANIFEST_NAME):
		rel = path.relative_to(directory).as_posix()
		st = path.stat()
		old = old_files.get(rel)
		if old and old.get("size") == st.st_size and old.get("mtime_ns") == st.st_mtime_ns:
			files[rel] = old
		else:
			files[rel] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": sha256_file(path)}
	digest = _files_digest(files)
	manifest = {
		"schema": MANIFEST_SCHEMA,
		"name": name or previous.get("name") or directory.name,
		"kind": kind or previous.get("kind") or "dataset",
		"digest": digest,
		"file_count": len(files),
		"total_bytes": sum(f["size"] for f in files.values()),
		"params": params if params is not None else previous.get("params", {}),
		# Keep the original creation stamp while the content is unchanged; a new digest is a new version.
		"created_at": previous["created_at"] if previous.get("digest") == digest else datetime.now(UTC).isoformat(),
		"git": git_state(root),
		"files": files,
	}
	if write:
		manifest_path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
	return manifest


def verify_manifest(directory: Path) -> dict[str, Any]:
	"""Re-hash ``directory`` against its stored manifest; reports changed, missing and unlisted files."""
	directory = Path(directory).resolve()
	manifest = _read_manifest(directory / MANIFEST_NAME)
	if manifest is None:
		raise FileNotFoundError(f"no {MANIFEST_NAME} in {directory}")
	changed, missing = [], []
	for rel, f in manifest["files"].items():
		path = directory / rel
		if not path.is_file():
			missing.append(rel)
		elif path.stat().st_size != f["size"] or sha256_file(path) != f["sha256"]:
			changed.append(rel)
	listed = set(manifest["files"])
	unlisted = sorted(
		p.relative_to(directory).as_posix()
		for p in directory.rglob("*")
		if p.is_file() and p.name != MANIFEST_NAME and p.relative_to(directory).as_posix() not in listed
	)
	return {
		"ok": not (changed or missing or unlisted),
		"digest": manifest["digest"],
		"changed": changed,
		"missing": missing,
		"unlisted": unlisted,
	}


def dataset_ref(directory: Path) -> dict[str, Any]:
	"""Compact identity of a dataset directory for run tags: name, digest, counts.

	Always re-checks the files (cheap: unchanged size+mtime reuse the stored hash), so a modified dataset never
	reports the digest of an older version. A read-only directory is hashed without writing the manifest.
	"""
	directory = Path(directory)
	try:
		manifest = build_manifest(directory)
	except PermissionError:
		manifest = build_manifest(directory, write=False)
	return {k: manifest[k] for k in ("name", "kind", "digest", "file_count", "total_bytes", "created_at")} | {
		"path": str(directory)
	}
