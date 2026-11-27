"""Build the inventory tables (SQLite + CSV + a Markdown summary) from the scanners, and apply operator overrides."""

from __future__ import annotations

import csv
import datetime as dt
import fnmatch
import json
import os
import sqlite3
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from digital_twin_armigo.inventory import scan, taxonomy

TRAJECTORY_COLUMNS = (
	"traj_id", "record_kind", "root", "path", "run_group",
	"category", "control", "sensing", "category_basis", "category_note",
	"engine", "scene_id", "seed", "defective", "defect_face",
	"verdict", "verdict_final_cycle", "failure_mode", "failure_mode_final_cycle",
	"label", "label_correct", "sim_frames", "fps", "sim_s", "wall_s",
	"views", "rewinds", "retries", "llm_consulted", "camera_profile", "detector_sha256",
	"takeover_gate", "takeover_frame",
	"has_actions", "has_events", "has_keyframes", "has_video", "has_motion",
	"split_hint", "catalog_role", "quality_status", "condition", "alias",
	"rerun_of", "attempt", "rerun_code", "rerun_info", "superseded_by",
	"written_at", "episode_sha256", "same_as", "tags",
	"batch", "group", "camera_condition", "strategy", "detector", "layout_id", "case_id", "run_id", "parent_run",
	"run_state",
	"auto_root_cause", "auto_mechanism", "auto_upstream",
)  # fmt: skip
SCENE_COLUMNS = (
	"scene_id", "scene_kind", "engine", "dataset_id", "seed", "defective", "defect_kind", "defect_face",
	"brick_size", "split", "status", "n_trajectories", "categories", "layout",
)  # fmt: skip
DATASET_COLUMNS = (
	"dataset_id", "name", "kind", "root", "path", "engine", "n_scenes", "n_items", "item_unit", "digest", "manifest_commit", "summary",
)  # fmt: skip
JSON_COLUMNS = {"brick_size", "layout", "summary", "tags", "categories", "rerun_info"}


def project_root() -> Path:
	return Path(__file__).resolve().parents[3]


def load_overrides(path: Path) -> list[dict]:
	"""``[{"match": glob, "set": {category, control, sensing, note, tags}}]``; the glob tests the path and the id."""
	try:
		rules = json.loads(path.read_text(encoding="utf-8"))
	except (OSError, ValueError):
		return []
	return [r for r in rules if isinstance(r, dict) and r.get("match") and isinstance(r.get("set"), dict)]


def apply_overrides(rows: list[dict], rules: list[dict]) -> int:
	"""Later rules win. Returns the number of rows touched."""
	touched = 0
	for row in rows:
		hit = False
		for rule in rules:
			pattern = rule["match"]
			if fnmatch.fnmatch(Path(row["path"]).as_posix(), pattern) or fnmatch.fnmatch(row["traj_id"], pattern):
				new = rule["set"]
				if new.get("category") and new["category"] not in taxonomy.CATEGORIES:
					raise ValueError(f"override {pattern!r}: unknown category {new['category']!r}")
				for key in ("category", "control", "sensing"):
					if new.get(key):
						row[key] = new[key]
				if "note" in new:
					row["category_note"] = new["note"]
				if "tags" in new:
					row["tags"] = sorted(set(row.get("tags") or []) | set(new["tags"]))
				hit = True
		if hit:
			row["category_basis"] = "override"
			touched += 1
	return touched


def mark_duplicates(rows: list[dict]) -> int:
	"""Rows whose ``episode.json`` is byte-identical to an earlier row's point at it (``same_as``): copies, or a deterministic
	re-run of the same case and configuration."""
	first: dict[str, str] = {}
	n = 0
	for row in rows:
		sha = row.get("episode_sha256")
		if not sha or row.get(
			"rerun_of"
		):  # a re-run is a record of its own, even when it reproduced the original exactly
			continue
		if sha in first:
			row["same_as"] = first[sha]
			n += 1
		else:
			first[sha] = row["traj_id"]
	return n


def link_reruns(rows: list[dict]) -> int:
	"""Fill ``superseded_by`` on originals whose latest *superseding* re-run exists, and tag re-runs that reproduced the
	original byte for byte. Originals are never changed otherwise: the old verdict stays in the table. Returns the number of
	superseded originals."""
	by_id = {r["traj_id"]: r for r in rows}
	latest: dict[str, dict] = {}
	for row in rows:
		orig = by_id.get(row.get("rerun_of") or "")
		if orig is None:
			continue
		if row.get("episode_sha256") and row["episode_sha256"] == orig.get("episode_sha256"):
			row["tags"] = sorted(set(row.get("tags") or []) | {"reproduced"})
		if not (row.get("rerun_info") or {}).get("supersedes"):
			continue
		best = latest.get(orig["traj_id"])
		if best is None or (row.get("attempt") or 0, row["written_at"]) > (
			best.get("attempt") or 0,
			best["written_at"],
		):
			latest[orig["traj_id"]] = row
	for tid, row in latest.items():
		by_id[tid]["superseded_by"] = row["traj_id"]
	return len(latest)


def collect_scenes(trajectories: list[dict], render_scenes: list[dict]) -> list[dict]:
	"""One scene per distinct seeded case / stored layout (counted over trajectories), plus every render-dataset layout."""
	groups: dict[str, list[dict]] = defaultdict(list)
	for row in trajectories:
		if row["scene_id"]:
			groups[row["scene_id"]].append(row)
	scenes = []
	for sid, rows in groups.items():
		first = rows[0]
		kind = "case" if sid.startswith("case:") else "layout" if sid.startswith("layout:") else "layout_group"
		layout = next((r.get("_layout") for r in rows if r.get("_layout")), None)
		scenes.append(
			{
				"scene_id": sid,
				"scene_kind": kind,
				"engine": "/".join(sorted({r["engine"] for r in rows if r["engine"]})),
				"dataset_id": None,
				"seed": first["seed"],
				"defective": first["defective"],
				"defect_kind": "cross_var" if first["defective"] else None,
				"defect_face": first["defect_face"],
				"brick_size": (layout or {}).get("brick_size"),
				"split": None,
				"status": None,
				"n_trajectories": len(rows),
				"categories": dict(Counter(r["category"] for r in rows)),
				"layout": layout,
			}
		)
	for s in render_scenes:
		scenes.append({**s, "n_trajectories": 0, "categories": {}})
	return scenes


def _cell(column: str, value):
	if column in JSON_COLUMNS and value is not None:
		return json.dumps(value, ensure_ascii=False, sort_keys=True)
	if isinstance(value, bool):
		return int(value)
	return value


def _write_sqlite(path: Path, tables: dict[str, tuple[tuple[str, ...], list[dict], str]], meta: dict) -> None:
	tmp = path.with_suffix(".tmp")
	tmp.unlink(missing_ok=True)
	con = sqlite3.connect(tmp)
	try:
		for name, (columns, rows, key) in tables.items():
			defs = ", ".join(f'"{c}"' + (" PRIMARY KEY" if c == key else "") for c in columns)
			con.execute(f"create table {name} ({defs})")
			con.executemany(
				f"insert into {name} values ({', '.join('?' * len(columns))})",
				[[_cell(c, r.get(c)) for c in columns] for r in rows],
			)
		con.execute("create table meta (key text primary key, value text)")
		con.executemany("insert into meta values (?, ?)", [(k, json.dumps(v, default=str)) for k, v in meta.items()])
		con.commit()
	finally:
		con.close()
	os.replace(tmp, path)


def _write_csv(path: Path, columns: tuple[str, ...], rows: list[dict]) -> None:
	with open(path, "w", encoding="utf-8-sig", newline="") as f:  # BOM: Excel opens Chinese text correctly
		w = csv.writer(f)
		w.writerow(columns)
		for r in rows:
			w.writerow(["" if (v := _cell(c, r.get(c))) is None else v for c in columns])


def _git_commit() -> str | None:
	try:
		return (
			subprocess.run(
				["git", "rev-parse", "--short", "HEAD"],
				check=False,
				cwd=project_root(),
				capture_output=True,
				text=True,
				timeout=10,
			).stdout.strip()
			or None
		)
	except (OSError, subprocess.SubprocessError):
		return None


def default_overrides() -> Path:
	return project_root() / "config" / "inventory_overrides.json"


def build(
	roots: list[tuple[str, Path]],
	out: Path,
	*,
	catalogs: list[tuple[str, Path]] | None = None,
	overrides: Path | None = None,
) -> dict:
	"""Scan ``roots`` (label, path) and write ``inventory.sqlite``, ``trajectories.csv``, ``scenes.csv``, ``datasets.csv``
	and ``SUMMARY.md`` into ``out``. The overrides file (``config/inventory_overrides.json``, operator-written and versioned) is
	applied last and never rewritten here."""
	out.mkdir(parents=True, exist_ok=True)
	trajectories: list[dict] = []
	datasets: list[dict] = []
	render_scenes: list[dict] = []
	found: dict[str, dict] = {}
	catalog_dbs = list(catalogs or [])
	for label, root in roots:
		if not root.is_dir():
			raise FileNotFoundError(f"scan root {label}={root} is not a directory")
		if (root / "catalog.sqlite").is_file():
			catalog_dbs.append((label, root / "catalog.sqlite"))
		before = len(trajectories)
		trajectories += scan.scan_agent_episodes(root, label)
		n_agent = len(trajectories) - before
		n_data = 0
		for ds, scenes in scan.scan_datasets(root, label):
			datasets.append(ds)
			render_scenes += scenes
			n_data += 1
		found[label] = {"path": str(root), "agent_episodes": n_agent, "datasets": n_data}
	for label, db in catalog_dbs:
		rows = list(scan.scan_catalog(db, label))
		trajectories += rows
		found[f"{label}:catalog"] = {"path": str(db), "catalog_episodes": len(rows)}
	overrides = overrides or default_overrides()
	rules = load_overrides(overrides)
	touched = apply_overrides(trajectories, rules)
	dups = mark_duplicates(trajectories)
	superseded = link_reruns(trajectories)
	for row in trajectories:
		if row.get("sim_frames") and row.get("fps") and row.get("sim_s") is None:
			row["sim_s"] = round(row["sim_frames"] / row["fps"], 3)
	scenes = collect_scenes(trajectories, render_scenes)
	for table_rows in (trajectories, scenes, datasets):
		table_rows.sort(key=lambda r: str(r.get("traj_id") or r.get("scene_id") or r.get("dataset_id")))
	meta = {
		"scanned_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
		"rules_version": taxonomy.RULES_VERSION,
		"git_commit": _git_commit(),
		"roots": found,
		"overrides_file": str(overrides),
		"overrides_applied": touched,
		"duplicates": dups,
		"superseded": superseded,
	}
	_write_sqlite(
		out / "inventory.sqlite",
		{
			"trajectories": (TRAJECTORY_COLUMNS, trajectories, "traj_id"),
			"scenes": (SCENE_COLUMNS, scenes, "scene_id"),
			"datasets": (DATASET_COLUMNS, datasets, "dataset_id"),
		},
		meta,
	)
	_write_csv(out / "trajectories.csv", TRAJECTORY_COLUMNS, trajectories)
	_write_csv(out / "scenes.csv", SCENE_COLUMNS, scenes)
	_write_csv(out / "datasets.csv", DATASET_COLUMNS, datasets)
	summary = summarize(trajectories, scenes, datasets, meta)
	(out / "SUMMARY.md").write_text(summary, encoding="utf-8")
	return {
		"out": str(out),
		"trajectories": len(trajectories),
		"scenes": len(scenes),
		"datasets": len(datasets),
		**meta,
	}


# -- summary --------------------------------------------------------------------------------------------------------------
def _pct(n: int, d: int) -> str:
	return f"{100 * n / d:.0f}%" if d else "-"


def summarize(trajectories: list[dict], scenes: list[dict], datasets: list[dict], meta: dict) -> str:
	unique = [r for r in trajectories if not r.get("same_as")]
	lines = [
		"# 合成轨迹与场景清单",
		"",
		f"扫描时间 {meta['scanned_at']} · 分类规则 v{meta['rules_version']} · 代码 {meta['git_commit']}",
		"",
		f"轨迹 {len(trajectories)} 条，其中 {meta['duplicates']} 条的 episode.json 与更早的某条字节相同"
		+ f"（拷贝，或同案例同配置的确定性重跑），内容不重复的 {len(unique)} 条；场景 {len(scenes)} 个，数据集 {len(datasets)} 个。",
		"",
		"## 轨迹：按来源类别",
		"",
		"| 类别 | 条数 | 内容不重复 | MuJoCo | Isaac | 成功 | 失败 | 其他 | 平均 sim 秒 |",
		"| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
	]
	for cat in taxonomy.CATEGORIES:
		rows = [r for r in unique if r["category"] == cat]
		total = sum(r["category"] == cat for r in trajectories)
		if not total:
			continue
		ok = sum(r["verdict"] == "SUCCESS" for r in rows)
		bad = sum(r["verdict"] in ("FAILURE", "FAILED") for r in rows)
		secs = [r["sim_s"] for r in rows if r.get("sim_s")]
		mean = f"{sum(secs) / len(secs):.1f}" if secs else "-"
		lines.append(
			f"| {taxonomy.CATEGORIES[cat]} (`{cat}`) | {total} | {len(rows)} | {sum(r['engine'] == 'mujoco' for r in rows)} | "
			f"{sum(r['engine'] == 'isaac' for r in rows)} | {ok} ({_pct(ok, len(rows))}) | {bad} | {len(rows) - ok - bad} | {mean} |"
		)
	basis = Counter((r["category"], r["category_basis"]) for r in unique)
	lines += [
		"",
		"## 分类依据（可信度）",
		"",
		"`explicit` 写入时冻结；`inferred` 由字段推断；`override` 人工指定。",
		"",
	]
	lines += ["| 类别 | explicit | inferred | override |", "| --- | ---: | ---: | ---: |"]
	for cat in taxonomy.CATEGORIES:
		if any(c == cat for c, _ in basis):
			lines.append(
				f"| `{cat}` | {basis[cat, 'explicit']} | {basis[cat, 'inferred']} | {basis[cat, 'override']} |"
			)
	unknown = [r for r in unique if r["category"] == "agent_unknown"]
	if unknown:
		lines += [
			"",
			f"**需人工确认：{len(unknown)} 条 agent 回合没有 sensing 记录且无法推断**（用 `armigo-inventory label` 补）。",
		]
	groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
	for r in trajectories:
		groups[r["root"], r["run_group"]].append(r)
	lines += [
		"",
		"## 轨迹：按运行批次",
		"",
		"| 批次 | 类别 | 条数 | 重复 | 成功率 | 引擎 |",
		"| --- | --- | ---: | ---: | ---: | --- |",
	]
	for (root, group), rows in sorted(groups.items(), key=lambda kv: -len(kv[1])):
		ok = sum(r["verdict"] == "SUCCESS" for r in rows)
		cats = ", ".join(sorted({r["category"] for r in rows}))
		engines = "/".join(sorted({r["engine"] for r in rows if r["engine"]}))
		dup = sum(bool(r.get("same_as")) for r in rows)
		lines.append(f"| `{root}:{group}` | {cats} | {len(rows)} | {dup or ''} | {_pct(ok, len(rows))} | {engines} |")
	kinds = Counter(s["scene_kind"] for s in scenes)
	lines += ["", "## 场景", "", "| 类型 | 个数 |", "| --- | ---: |"]
	lines += [f"| `{k}` | {n} |" for k, n in kinds.most_common()]
	lines += ["", "## 数据集", "", "| 数据集 | 类型 | 场景 | 条目 | 摘要 |", "| --- | --- | ---: | ---: | --- |"]
	for d in datasets:
		digest = f"`{d['digest'][:10]}`" if d.get("digest") else "无清单"
		lines.append(
			f"| `{d['root']}:{d['name']}` | {d['kind']} | {d['n_scenes'] or ''} | {d['n_items'] or ''} {d['item_unit']} | {digest} |"
		)
	return "\n".join(lines) + "\n"
