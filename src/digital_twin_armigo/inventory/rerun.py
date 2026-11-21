"""Re-run failed agent episodes from the inventory, keeping the originals.

Every item is a fresh full run of the recorded case with the recorded policy and sensing mode (``task.json`` next to the
original ``episode.json``), written to a new directory with

* the usual episode files plus ``replay/`` (poses of every moving geom, saved by ``ARMIGO_RERUN=record``), so a failed
  re-run can still be rebuilt in 3D without storing video, and
* ``lineage.json``: which trajectory it re-runs, the attempt number, the code commit, whether the detector changed, and
  the old and new verdict.

Nothing about the original changes. The next ``armigo-inventory scan`` links the two rows (``rerun_of`` /
``superseded_by``), so the table shows the current result per case and the whole history.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from digital_twin_armigo.inventory import query, scan, store

RERUNNABLE = ("agent_omniscient", "agent_realistic")
FAILED = ("FAILED", "FAILURE", "UNKNOWN")
SENSING = ("truth", "realistic")


def plan(
	inventory_dir: Path,
	*,
	categories=RERUNNABLE,
	verdicts=FAILED,
	run_group: str | None = None,
	limit: int | None = None,
	redo: bool = False,
) -> tuple[list[dict], Counter]:
	"""``(items, skipped)``: the failed agent episodes that can be re-run, and counts of why others cannot.

	``redo`` also takes originals that already have a superseding re-run (for another attempt)."""
	sql = f"""select traj_id, path, run_group, category, sensing, verdict, detector_sha256, superseded_by
		from trajectories
		where record_kind = 'agent_episode' and same_as is null and rerun_of is null
		  and verdict in ({",".join("?" * len(verdicts))}) and category in ({",".join("?" * len(categories))})"""
	args: list = [*verdicts, *categories]
	if run_group:
		sql += " and run_group like ?"
		args.append(f"%{run_group}%")
	sql += " order by run_group, traj_id"
	items: list[dict] = []
	skipped: Counter = Counter()
	with query.connect(inventory_dir) as con:
		rows = [dict(r) for r in con.execute(sql, args)]
		for row in rows:
			if row["superseded_by"] and not redo:
				skipped["already re-run"] += 1
				continue
			task = scan._json(Path(row["path"]) / "task.json")
			if not isinstance(task, dict) or not task.get("case") or not isinstance(task.get("policy"), dict):
				skipped["no task.json (cannot rebuild the run)"] += 1
				continue
			if task.get("use_llm"):
				skipped["LLM agent (needs the model server)"] += 1
				continue
			if row["sensing"] not in SENSING:
				skipped[f"sensing {row['sensing']}"] += 1
				continue
			n_prev = con.execute("select count(*) from trajectories where rerun_of = ?", (row["traj_id"],)).fetchone()[
				0
			]
			items.append(
				{
					**row,
					"case": task["case"],
					"policy": task["policy"],
					"attempt": n_prev + 1,
				}
			)
			if limit and len(items) >= limit:
				break
	return items, skipped


def destination(out_root: Path, item: dict) -> Path:
	return Path(out_root) / item["run_group"] / f"{Path(item['path']).name}-r{item['attempt']}"


def _commit() -> str | None:
	try:
		out = subprocess.run(
			["git", "rev-parse", "--short", "HEAD"],
			capture_output=True,
			text=True,
			cwd=store.project_root(),
			check=False,
			timeout=10,
		)
	except (OSError, subprocess.SubprocessError):
		return None
	return out.stdout.strip() or None if out.returncode == 0 else None


def execute(
	items: list[dict], out_root: Path, detector: Path, *, progress: Callable[[str], None] = print
) -> list[dict]:
	"""Run every item serially in this process and write ``lineage.json`` beside each new episode. A crashed run leaves
	an ``error`` entry in the returned list (and no lineage); finished destinations are skipped, so the command resumes."""
	os.environ["ARMIGO_RERUN"] = "record"  # poses to replay/, no viewer
	from digital_twin_armigo.agent.twin_graph import Policy
	from digital_twin_armigo.agent.twin_mcp import TwinService

	out_root = Path(out_root).resolve()
	detector_sha = hashlib.sha256(Path(detector).read_bytes()).hexdigest()
	commit = _commit()
	results: list[dict] = []
	done = 0
	for sensing in sorted({i["sensing"] for i in items}):
		service = TwinService(out_root / "_service" / sensing, detector, None, workers=1, sensing=sensing)
		try:
			for item in (i for i in items if i["sensing"] == sensing):
				done += 1
				dest = destination(out_root, item)
				if (dest / scan.LINEAGE_FILE).is_file():
					results.append({"traj_id": item["traj_id"], "skipped": "already done", "dest": str(dest)})
					continue
				if dest.exists():  # a crashed earlier attempt of ours
					if out_root not in dest.resolve().parents:
						raise RuntimeError(f"refusing to clear {dest}")
					shutil.rmtree(dest)
				dest.parent.mkdir(parents=True, exist_ok=True)
				started = time.perf_counter()
				try:
					service.run(tuple(item["case"]), Policy(**item["policy"]), dest, video=False, use_llm=False)
					episode = json.loads((dest / "episode.json").read_text(encoding="utf-8"))
				except Exception as exc:  # noqa: BLE001 - one broken case must not stop the batch
					results.append({"traj_id": item["traj_id"], "error": f"{type(exc).__name__}: {exc}"})
					progress(f"[{done}/{len(items)}] {item['traj_id']}: ERROR {type(exc).__name__}: {exc}")
					continue
				verdict = (episode.get("result") or {}).get("verdict")
				wall = round(time.perf_counter() - started, 1)
				lineage = {
					"schema": scan.LINEAGE_SCHEMA,
					"rerun_of": item["traj_id"],
					"attempt": item["attempt"],
					"ran_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
					"code_commit": commit,
					"supersedes": True,
					"original_path": item["path"],
					"original_verdict": item["verdict"],
					"original_detector_sha256": item["detector_sha256"],
					"detector_sha256": detector_sha,
					"detector_changed": item["detector_sha256"] != detector_sha,
					"verdict": verdict,
					"wall_s": wall,
				}
				(dest / scan.LINEAGE_FILE).write_text(json.dumps(lineage, indent=1), encoding="utf-8")
				results.append({"traj_id": item["traj_id"], "verdict": verdict, "dest": str(dest), "wall_s": wall})
				progress(f"[{done}/{len(items)}] {item['traj_id']}: {item['verdict']} -> {verdict} ({wall}s)")
		finally:
			service.close()
	return results
