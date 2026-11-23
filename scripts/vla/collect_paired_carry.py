#!/usr/bin/env python
"""Collect paired opposite-target carry demonstrations on train layouts (vla/paired_carry.py).

Per case: ``<out>/cases/case-XXXX/pair.json`` plus the raw and exportable episodes; existing ``pair.json`` files are
skipped (resume). Rollouts run in this interpreter (the repo ``.venv``). ``--export`` then exports every accepted
episode through LeRobot in ``--lerobot-python`` (one shard per ``--shard-size`` episodes) and merges the shards into
``<out>/lerobot``. Plan: docs/vla/BOWL_CHANNEL_PAIRED_PLAN_2026-11-23.md Tasks 4-5.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
	sys.path.insert(0, str(ROOT / "src"))

DEFAULT_DETECTOR = ROOT / "results/agent-bench/sup-v7bot-local.json"
LEROBOT_PYTHON = Path("E:/il/lerobot-cuda/Scripts/python.exe")
REPO_ID = "local/armigo-six-face"

_SCORER = None
_CROP = None


def _init_worker(detector: str) -> None:
	global _SCORER, _CROP
	if str(ROOT / "src") not in sys.path:
		sys.path.insert(0, str(ROOT / "src"))
	from digital_twin_armigo.inspection_scorer import default_sup_scorer

	_SCORER = default_sup_scorer(Path(detector))
	_CROP = int(json.loads(Path(detector).read_text(encoding="utf-8"))["crop_size"])


def _run_one(task: tuple[int, list, str]) -> dict:
	from digital_twin_armigo.vla.paired_carry import run_pair

	idx, case, case_dir = task
	t0 = time.perf_counter()
	try:
		rec = run_pair(case, _SCORER, Path(case_dir), crop_size=_CROP)
	except Exception as exc:  # noqa: BLE001 - one failing layout must not stop the batch
		rec = {"case": list(case), "skip": "error", "error": f"{type(exc).__name__}: {exc}"}
	rec["index"] = idx
	rec["wall_s"] = round(time.perf_counter() - t0, 1)
	(Path(case_dir) / "pair.json").write_text(json.dumps(rec, indent=1, allow_nan=False), encoding="utf-8")
	return rec


def collect(cases: list, out: Path, detector: Path, workers: int) -> None:
	tasks = []
	for idx, case in enumerate(cases):
		case_dir = out / "cases" / f"case-{idx:04d}"
		if (case_dir / "pair.json").exists():
			continue
		case_dir.mkdir(parents=True, exist_ok=True)
		tasks.append((idx, list(case), str(case_dir)))
	print(f"{len(tasks)} cases to run ({len(cases) - len(tasks)} already done)", flush=True)
	if not tasks:
		return
	if workers <= 1:
		_init_worker(str(detector))
		try:
			for t in tasks:
				_report(_run_one(t), len(cases))
		finally:
			if hasattr(_SCORER, "close"):
				_SCORER.close()
		return
	with ProcessPoolExecutor(
		max_workers=workers, mp_context=get_context("spawn"), initializer=_init_worker, initargs=(str(detector),)
	) as pool:
		for rec in pool.map(_run_one, tasks):
			_report(rec, len(cases))


def _report(rec: dict, n: int) -> None:
	status = rec.get("skip") or f"ok {rec['bowls']['parent']}->{rec['bowls']['swap']}"
	print(f"[{rec['index'] + 1}/{n}] case {rec['case'][0]}: {status} ({rec['wall_s']} s)", flush=True)


def accepted_episodes(out: Path) -> list[Path]:
	eps = []
	for f in sorted((out / "cases").glob("case-*/pair.json")):
		rec = json.loads(f.read_text(encoding="utf-8"))
		if rec.get("skip") is None and rec["episodes"]["parent"] and rec["episodes"]["swap"]:
			eps += [Path(rec["episodes"]["parent"]), Path(rec["episodes"]["swap"])]
	return eps


def export(out: Path, lerobot_python: Path, shard_size: int) -> None:
	eps = accepted_episodes(out)
	print(f"exporting {len(eps)} episodes", flush=True)
	shards = []
	for i in range(0, len(eps), shard_size):
		shard = out / "shards" / f"shard-{i // shard_size:03d}"
		shards.append(shard)
		if (shard / "meta" / "info.json").exists():
			continue
		cmd = [str(lerobot_python), "-m", "digital_twin_armigo.agent.twin_record", "--out", str(shard), "--repo-id"]
		cmd += [REPO_ID, *map(str, eps[i : i + shard_size])]
		subprocess.run(cmd, check=True, cwd=ROOT, env=_env())
	merge = (
		"import sys; from pathlib import Path; sys.path.insert(0, 'scripts/vla'); "
		"from rollout_and_export import merge_shards_cmd; "
		f"merge_shards_cmd(Path(r'{out / 'lerobot'}'), '{REPO_ID}', [Path(p) for p in {[str(s) for s in shards]!r}])"
	)
	subprocess.run([str(lerobot_python), "-c", merge], check=True, cwd=ROOT, env=_env())


def _env() -> dict:
	import os

	env = dict(os.environ)
	env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
	return env


def main(argv=None) -> int:
	p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	p.add_argument("--suite", type=Path, default=ROOT / "config/vla/train.json")
	p.add_argument("--out", type=Path, required=True)
	p.add_argument("--detector", type=Path, default=DEFAULT_DETECTOR)
	p.add_argument("--max-episodes", type=int, default=None)
	p.add_argument("--workers", type=int, default=1)
	p.add_argument("--export", action="store_true", help="Export accepted episodes and merge into <out>/lerobot")
	p.add_argument("--export-only", action="store_true", help="Skip rollouts; export what is there")
	p.add_argument("--lerobot-python", type=Path, default=LEROBOT_PYTHON)
	p.add_argument("--shard-size", type=int, default=40)
	args = p.parse_args(argv)

	from digital_twin_armigo.vla.dagger import DEFAULT_FORBID_SUITES, assert_train_layouts_only

	raw = json.loads(args.suite.read_text(encoding="utf-8-sig"))
	cases = list(raw["cases"] if isinstance(raw, dict) else raw)
	extra = [f"config/vla/val{i}.json" for i in range(2, 10) if (ROOT / f"config/vla/val{i}.json").exists()]
	assert_train_layouts_only(args.suite, cases, forbid_suites=(*DEFAULT_FORBID_SUITES, *extra))
	if args.max_episodes is not None:
		cases = cases[: args.max_episodes]
	args.out.mkdir(parents=True, exist_ok=True)
	if not args.export_only:
		collect(cases, args.out, args.detector, args.workers)
	if args.export or args.export_only:
		export(args.out, args.lerobot_python, args.shard_size)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
