"""Bronze: register raw episode directories by content hash. Append-only; a changed hash is an integrity event."""

from __future__ import annotations

import datetime as dt
import hashlib
import os
import sqlite3
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
MARKERS = ("run_status.json", "episode.json")
FILES = {
	"sha_episode": "episode.json",
	"sha_events": "events.jsonl",
	"sha_status": "run_status.json",
	"sha_task": "task.json",
	"sha_motion": "replay/motion.npz",
}


def _sha(p: Path) -> str | None:
	if not p.is_file():
		return None
	h = hashlib.sha256()
	with p.open("rb") as f:
		for chunk in iter(lambda: f.read(1 << 20), b""):
			h.update(chunk)
	return h.hexdigest()


def _commit() -> str | None:
	try:
		out = subprocess.run(
			["git", "-C", str(REPO), "rev-parse", "HEAD"],
			capture_output=True,
			text=True,
			check=True,
		).stdout.strip()
	except (OSError, subprocess.CalledProcessError):
		return None
	return out or None


def episode_dirs(root: Path | str):
	root_path = Path(root)
	if not root_path.is_dir():
		return
	for dirpath, dirnames, _filenames in os.walk(root_path, followlinks=False, onerror=lambda _err: None):
		dirnames[:] = sorted(d for d in dirnames if d != ".git")
		d = Path(dirpath)
		if any((d / m).is_file() for m in MARKERS):
			yield d


def register(con: sqlite3.Connection, root: Path | str, *, source: str) -> dict[str, int]:
	now = dt.datetime.now(dt.UTC).isoformat()
	commit = _commit()
	counts = {"new": 0, "unchanged": 0, "changed": 0}
	cols = list(FILES)
	col_sql = ", ".join(cols)
	placeholders = ", ".join("?" * len(cols))
	for d in dict.fromkeys(episode_dirs(root)):
		path_str = str(d)
		shas = {c: _sha(d / f) for c, f in FILES.items()}
		old = con.execute(f"select {col_sql} from bronze where path = ?", (path_str,)).fetchone()
		if old is None:
			con.execute(
				f"insert into bronze (path, source, registered_at, git_commit, {col_sql}) "
				f"values (?, ?, ?, ?, {placeholders})",
				(path_str, source, now, commit, *shas.values()),
			)
			counts["new"] += 1
			continue
		diffs = [(c, o, shas[c]) for c, o in zip(cols, old, strict=True) if o != shas[c]]
		for c, o, n in diffs:
			con.execute(
				"insert into integrity_event (path, field, old_sha, new_sha, seen_at) values (?, ?, ?, ?, ?)",
				(path_str, FILES[c].split("/")[-1], o, n, now),
			)
		counts["changed" if diffs else "unchanged"] += 1
	con.commit()
	return counts
