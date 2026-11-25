"""Build a diagnosis benchmark JSON from human-confirmed audit verdicts (Task H7).

Usage:
	python scripts/experiments/make_diagnosis_bench.py \\
		--audit D:/Armigo_data_v2/audit/reviews.sqlite \\
		--inventory D:/Armigo_data_v2/inventory \\
		--out config/agent_bench/diagnosis_bench/v1.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _bootstrap_path() -> None:
	repo = os.environ.get("ARMIGO_REPO")
	if repo:
		repo_src = Path(repo) / "src"
		if repo_src.is_dir() and str(repo_src) not in sys.path:
			sys.path.insert(0, str(repo_src))
	src = Path(__file__).resolve().parents[2] / "src"
	if src.is_dir():
		if str(src) in sys.path:
			sys.path.remove(str(src))
		sys.path.insert(0, str(src))


_bootstrap_path()

from digital_twin_armigo.experiments.diagnosis import split_of
from digital_twin_armigo.governance.segments import is_sealed, seed_of_path_or_case
from digital_twin_armigo.inventory import audit, query


def _batch_from_path(path: Path) -> str | None:
	parts = path.parts
	for i in range(len(parts) - 2, -1, -1):
		if parts[i] == "runs":
			return parts[i + 1]
	return None


def _read_task_json(ep_path: Path) -> dict | None:
	tf = ep_path / "task.json"
	if not tf.is_file():
		return None
	try:
		data = json.loads(tf.read_text(encoding="utf-8-sig"))
		return data if isinstance(data, dict) else None
	except (OSError, ValueError):
		return None


def build_benchmark(audit_path: Path, inventory_dir: Path) -> list[dict]:
	"""Extract human-confirmed latest verdicts, attach episode paths, exclude sealed batches and sealed seeds."""
	audit_path = Path(audit_path)
	inventory_dir = Path(inventory_dir)
	if not audit_path.is_file():
		raise FileNotFoundError(f"audit database not found: {audit_path}")

	latest_by_tid = audit.latest_all(audit_path)

	entries: list[dict] = []
	for tid, rev in sorted(latest_by_tid.items()):
		if rev is None or rev.get("state") != "confirmed" or not rev.get("root_cause"):
			continue
		ep_path = query.agent_episode_path(inventory_dir, tid)
		if ep_path is None:
			continue
		batch = _batch_from_path(ep_path)
		task = _read_task_json(ep_path)
		case_list = (task or {}).get("case")
		case_opts = (
			case_list[3]
			if isinstance(case_list, list) and len(case_list) >= 4 and isinstance(case_list[3], dict)
			else None
		)
		seed = seed_of_path_or_case(ep_path, task, None)
		if is_sealed(seed, batch=batch, options=case_opts):
			continue
		entries.append(
			{
				"traj_id": tid,
				"path": str(ep_path),
				"root_cause": rev["root_cause"],
				"mechanism": rev.get("mechanism"),
				"split": split_of(tid),
			}
		)

	entries.sort(key=lambda r: r["traj_id"])
	return entries


def _build_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	p.add_argument("--audit", required=True, help="Path to audit reviews.sqlite")
	p.add_argument("--inventory", required=True, help="Path to inventory directory containing inventory.sqlite")
	p.add_argument("--out", required=True, help="Output diagnosis benchmark JSON path")
	return p


def main(argv: list[str] | None = None) -> int:
	args = _build_parser().parse_args(argv)
	entries = build_benchmark(Path(args.audit), Path(args.inventory))
	out_path = Path(args.out)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	out_path.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

	n_calib = sum(1 for e in entries if e["split"] == "calib")
	n_heldout = sum(1 for e in entries if e["split"] == "heldout")
	print(f"{out_path}: {len(entries)} entries (calib={n_calib}, heldout={n_heldout})")
	if len(entries) < 40:
		print(
			f"WARNING: only {len(entries)} confirmed entries (< 40); collect more human audits before drawing conclusions.",
			file=sys.stderr,
		)
	return 0


if __name__ == "__main__":
	sys.exit(main())
