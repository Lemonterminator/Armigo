"""The governance store: bronze registrations, integrity events, silver tables and gold builds (one SQLite file)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
create table if not exists bronze (
	path text primary key,
	source text not null,
	registered_at text not null,
	git_commit text,
	sha_episode text,
	sha_events text,
	sha_status text,
	sha_task text,
	sha_motion text
);
create table if not exists integrity_event (
	id integer primary key,
	path text not null,
	field text not null,
	old_sha text,
	new_sha text,
	seen_at text not null
);
create table if not exists silver_episode (
	build_id text not null,
	path text not null,
	sensing text,
	pair_key text,
	split text,
	quarantined integer,
	quarantine_reasons text,
	row_json text not null,
	primary key (build_id, path)
);
create table if not exists silver_pair (
	build_id text not null,
	pair_key text not null,
	camera_condition text not null,
	truth_path text,
	realistic_path text,
	outcome text not null,
	primary key (build_id, pair_key, camera_condition)
);
create table if not exists gold_build (
	product text not null,
	version text not null,
	silver_build text not null,
	built_at text not null,
	git_commit text,
	predicate_sha text not null,
	n_items integer not null,
	primary key (product, version)
);
"""


def connect(path: Path | str) -> sqlite3.Connection:
	db_path = Path(path)
	db_path.parent.mkdir(parents=True, exist_ok=True)
	con = sqlite3.connect(db_path)
	con.executescript(SCHEMA)
	return con
