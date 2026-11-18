"""Canonical JSON encoding and SHA-256 helpers shared by snapshots, the ledger, and the execution service."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

_CHUNK = 1 << 20


def canonical_json(obj: Any) -> bytes:
	"""Encode ``obj`` deterministically (sorted keys, no whitespace, UTF-8); NaN and infinities are rejected."""
	return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
	return hashlib.sha256(data).hexdigest()


def sha256_json(obj: Any) -> str:
	"""SHA-256 of the canonical JSON encoding of ``obj``."""
	return sha256_bytes(canonical_json(obj))


def sha256_file(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open("rb") as fh:
		while chunk := fh.read(_CHUNK):
			digest.update(chunk)
	return digest.hexdigest()
