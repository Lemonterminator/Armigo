"""One manifest per evaluation output directory: the frozen inputs, code state and resolved arguments of the run.

A resumed run (S10 reuses existing ``episode.json`` files) must have the same identity: :func:`bind_manifest` writes
``run_manifest.json`` on the first start and refuses a later start whose identity differs.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any

MANIFEST_NAME = "run_manifest.json"
PACKAGES = ("digital-twin-armigo", "numpy", "mujoco", "opencv-python", "opencv-python-headless", "scipy", "torch")


def sha256_path(path: Path) -> str:
	h = hashlib.sha256()
	with open(path, "rb") as f:
		for block in iter(lambda: f.read(1 << 20), b""):
			h.update(block)
	return h.hexdigest()


def describe_file(path: Path | None) -> dict[str, Any] | None:
	if path is None:
		return None
	path = Path(path)
	return {"path": str(path.resolve()), "sha256": sha256_path(path) if path.is_file() else None}


def describe_dir(path: Path | None) -> dict[str, Any] | None:
	"""Every file of a checkpoint directory with its SHA-256 (sorted by relative path)."""
	if path is None:
		return None
	path = Path(path)
	files = {p.relative_to(path).as_posix(): sha256_path(p) for p in sorted(path.rglob("*")) if p.is_file()}
	digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
	return {"path": str(path.resolve()), "files": files, "sha256": digest}


def describe_detector(path: Path | None) -> dict[str, Any] | None:
	"""The detector config and every existing file it names (the LoRA weights among them)."""
	out = describe_file(path)
	if out is None or not Path(path).is_file():
		return out
	referenced: dict[str, Any] = {}

	def walk(v):
		if isinstance(v, dict):
			for x in v.values():
				walk(x)
		elif isinstance(v, list):
			for x in v:
				walk(x)
		elif isinstance(v, str) and len(v) < 400:
			for cand in (Path(v), Path(path).parent / v):
				try:
					if cand.is_file():
						referenced[v] = sha256_path(cand)
						return
					if cand.is_dir():
						referenced[v] = describe_dir(cand)["sha256"]
						return
				except OSError:
					return

	try:
		walk(json.loads(Path(path).read_text(encoding="utf-8")))
	except (ValueError, OSError):
		pass
	out["referenced"] = referenced
	return out


def git_state(root: Path) -> dict[str, Any]:
	def run(*args):
		return subprocess.run(["git", *args], cwd=root, capture_output=True, check=False).stdout

	diff = run("diff", "HEAD", "--binary")
	status = run("status", "--porcelain").decode("utf-8", "replace").splitlines()
	untracked = sorted(line[3:] for line in status if line.startswith("??"))
	untracked_src = {}
	for rel in untracked:
		if not rel.startswith(("src/", "scripts/", "config/")):
			continue
		p = root / rel
		files = [p] if p.is_file() else sorted(p.rglob("*.py")) if p.is_dir() else []
		for f in files:
			if f.suffix == ".py":
				untracked_src[f.relative_to(root).as_posix()] = sha256_path(f)
	return {
		"commit": run("rev-parse", "HEAD").decode().strip(),
		"branch": run("rev-parse", "--abbrev-ref", "HEAD").decode().strip(),
		"diff_sha256": hashlib.sha256(diff).hexdigest(),
		"dirty": bool(diff.strip()),
		"untracked_py": untracked_src,
	}


def environment() -> dict[str, Any]:
	versions = {}
	for name in PACKAGES:
		try:
			versions[name] = metadata.version(name)
		except metadata.PackageNotFoundError:
			continue
	return {"python": sys.executable, "python_version": platform.python_version(), "packages": versions}


def identity(manifest: dict[str, Any]) -> str:
	"""Digest of everything that must match for a resume: inputs, code and arguments (not times or paths)."""
	keys = ("protocol", "grasp_tail_mode", "args", "inputs_sha256", "code")
	payload = {k: manifest.get(k) for k in keys}
	return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def build_manifest(
	*, root: Path, args: dict[str, Any], inputs: dict[str, dict | None], protocol: str, grasp_tail_mode: str
) -> dict[str, Any]:
	git = git_state(root)
	manifest = {
		"protocol": protocol,
		"grasp_tail_mode": grasp_tail_mode,
		"args": {k: (str(v) if isinstance(v, Path) else v) for k, v in sorted(args.items())},
		"inputs": inputs,
		"inputs_sha256": {k: (None if v is None else v.get("sha256")) for k, v in inputs.items()},
		"code": {k: git[k] for k in ("commit", "diff_sha256", "untracked_py")},
		"git": git,
		"environment": environment(),
		"created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
	}
	manifest["identity"] = identity(manifest)
	return manifest


def bind_manifest(out_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
	"""Write ``run_manifest.json`` or check a resume against it; returns the bound manifest."""
	out_dir = Path(out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)
	path = out_dir / MANIFEST_NAME
	if path.exists():
		old = json.loads(path.read_text(encoding="utf-8"))
		if old.get("identity") != manifest["identity"]:
			diffs = [
				k
				for k in ("protocol", "grasp_tail_mode", "args", "inputs_sha256", "code")
				if old.get(k) != manifest.get(k)
			]
			raise ValueError(f"{path} belongs to a different run (differs in {diffs}): use a new --out / run ID")
		old.setdefault("resumed_at", []).append(manifest["created_at"])
		path.write_text(json.dumps(old, indent=2) + "\n", encoding="utf-8")
		return old
	if any(out_dir.glob("student/case-*/episode.json")):
		raise ValueError(f"{out_dir} has episodes but no {MANIFEST_NAME}: use a new --out / run ID")
	path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
	return manifest
