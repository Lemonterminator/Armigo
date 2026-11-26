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


def _read_episode_file(ep_path: Path, name: str) -> dict | None:
	tf = ep_path / name
	if not tf.is_file():
		return None
	try:
		data = json.loads(tf.read_text(encoding="utf-8-sig"))
		return data if isinstance(data, dict) else None
	except (OSError, ValueError):
		return None


def _path_from_runs(runs_root: Path | None, tid: str) -> Path | None:
	if runs_root is None:
		return None
	root = runs_root.resolve()
	parts = Path(tid).parts
	if len(parts) != 4 or any(part in (".", "..") for part in parts):
		return None
	candidate = (root / tid).resolve()
	return candidate if candidate.is_relative_to(root) and candidate.is_dir() else None


def build_benchmark(
	audit_path: Path,
	inventory_dir: Path,
	*,
	runs_root: Path | None = None,
	review_ids: list[int] | None = None,
) -> list[dict]:
	"""Use confirmed reviews by default, or an explicitly selected snapshot of human adjudications."""
	audit_path = Path(audit_path)
	inventory_dir = Path(inventory_dir)
	if not audit_path.is_file():
		raise FileNotFoundError(f"audit database not found: {audit_path}")

	if review_ids is None:
		reviews = list(audit.latest_all(audit_path).values())
	else:
		reviews = audit.reviews_by_ids(audit_path, review_ids)
		missing = set(review_ids) - {rev["id"] for rev in reviews}
		if missing:
			raise ValueError(f"unknown audit review IDs: {sorted(missing)}")
		if len({rev["traj_id"] for rev in reviews}) != len(reviews):
			raise ValueError("select at most one review per trajectory")

	entries: list[dict] = []
	states = {"confirmed"} if review_ids is None else {"confirmed", "disputed"}
	for rev in reviews:
		if rev.get("state") not in states or not rev.get("root_cause"):
			continue
		tid = rev["traj_id"]
		ep_path = query.agent_episode_path(inventory_dir, tid) or _path_from_runs(runs_root, tid)
		if ep_path is None:
			continue
		batch = _batch_from_path(ep_path)
		if is_sealed(seed_of_path_or_case(ep_path), batch=batch):
			continue
		task = _read_episode_file(ep_path, "task.json")
		episode = _read_episode_file(ep_path, "episode.json") or {}
		if ((episode.get("provenance") or {}).get("experiment") or {}).get("sealed") is True:
			continue
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
				**({"review_id": rev["id"], "review_state": rev["state"]} if review_ids is not None else {}),
			}
		)

	entries.sort(key=lambda r: r["traj_id"])
	return entries


def _build_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	p.add_argument("--audit", required=True, help="Path to audit reviews.sqlite")
	p.add_argument("--inventory", required=True, help="Path to inventory directory containing inventory.sqlite")
	p.add_argument("--runs-root", type=Path, help="Fallback root for batch/engine/group/episode trajectory IDs")
	p.add_argument(
		"--review-ids",
		type=int,
		nargs="+",
		help="Explicit human adjudications, including disputes of an automatic label",
	)
	p.add_argument("--out", required=True, help="Output diagnosis benchmark JSON path")
	return p


def main(argv: list[str] | None = None) -> int:
	args = _build_parser().parse_args(argv)
	entries = build_benchmark(
		Path(args.audit), Path(args.inventory), runs_root=args.runs_root, review_ids=args.review_ids
	)
	out_path = Path(args.out)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	out_path.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

	n_calib = sum(1 for e in entries if e["split"] == "calib")
	n_heldout = sum(1 for e in entries if e["split"] == "heldout")
	print(f"{out_path}: {len(entries)} entries (calib={n_calib}, heldout={n_heldout})")
	if len(entries) < 40:
		print(
			f"WARNING: only {len(entries)} human adjudications (< 40); collect more before drawing conclusions.",
			file=sys.stderr,
		)
	return 0


if __name__ == "__main__":
	sys.exit(main())
