#!/usr/bin/env python
"""Collect paired opposite-target carry demonstrations on train layouts (vla/paired_carry.py).

Per case: ``<out>/cases/case-XXXX/pair.json`` plus the two exportable episodes; existing ``pair.json`` files are
skipped (resume) unless they record an ``error``. Rollouts run in this interpreter (the repo ``.venv``).

With ``--export`` the rollouts run in batches of ``--batch`` cases; after each batch the newly accepted episodes are
exported through LeRobot in ``--lerobot-python`` into ``<out>/shards/shard-NNN`` (its ``episodes.json`` lists them),
the shard is read back (episode and frame counts), and only then are those episodes' lossless ``*.mkv`` deleted (a
pair is ~1 GB of video; 300 layouts do not fit on disk otherwise). Rejected cases keep only ``pair.json``.
``--merge-out`` merges the ``--merge-with`` datasets (in order) and all shards into one training dataset.
Plan: docs/vla/BOWL_CHANNEL_PAIRED_PLAN_2026-11-23.md Tasks 4-5.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

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
	if not (rec.get("episodes") or {}).get("parent"):
		for sub in ("raw", "parent", "swap"):
			shutil.rmtree(Path(case_dir) / sub, ignore_errors=True)
	rec["index"] = idx
	rec["wall_s"] = round(time.perf_counter() - t0, 1)
	(Path(case_dir) / "pair.json").write_text(json.dumps(rec, indent=1, allow_nan=False), encoding="utf-8")
	return rec


def _pending(out: Path, idx: int) -> bool:
	f = out / "cases" / f"case-{idx:04d}" / "pair.json"
	if not f.exists():
		return True
	return json.loads(f.read_text(encoding="utf-8")).get("skip") == "error"


def collect(cases: list, out: Path, detector: Path, workers: int, batch: int | None = None, after_batch=None) -> None:
	tasks = []
	for idx, case in enumerate(cases):
		if not _pending(out, idx):
			continue
		case_dir = out / "cases" / f"case-{idx:04d}"
		if case_dir.exists():
			shutil.rmtree(case_dir)
		case_dir.mkdir(parents=True)
		tasks.append((idx, list(case), str(case_dir)))
	print(f"{len(tasks)} cases to run ({len(cases) - len(tasks)} already done)", flush=True)
	if not tasks:
		return
	step = batch or len(tasks)
	if workers <= 1:
		_init_worker(str(detector))
		try:
			for i in range(0, len(tasks), step):
				for t in tasks[i : i + step]:
					_report(_run_one(t), len(cases))
				if after_batch is not None:
					after_batch()
		finally:
			if hasattr(_SCORER, "close"):
				_SCORER.close()
		return
	with ProcessPoolExecutor(
		max_workers=workers, mp_context=get_context("spawn"), initializer=_init_worker, initargs=(str(detector),)
	) as pool:
		for i in range(0, len(tasks), step):
			for rec in pool.map(_run_one, tasks[i : i + step]):
				_report(rec, len(cases))
			if after_batch is not None:
				after_batch()


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


def _shards(out: Path) -> list[Path]:
	return sorted(d for d in (out / "shards").glob("shard-*") if (d / "episodes.json").exists())


def _expected_frames(eps: list[Path]) -> int:
	from digital_twin_armigo.agent.twin_record import unsupervised_prefix_end

	frames = 0
	for e in eps:
		with np.load(e / "branch-0" / "actions.npz", allow_pickle=False) as z:
			frames += len(z["action"]) - unsupervised_prefix_end(e)
	return frames


def _export_shard(shard: Path, eps: list[Path], lerobot_python: Path, student_cameras_only: bool = True) -> None:
	"""Export ``eps`` (trimmed to their supervised part) into ``shard``, read it back, then delete their ``*.mkv``."""
	if shard.exists():
		shutil.rmtree(shard)  # an unfinished export (no episodes.json)
	log = shard.with_suffix(".log")
	cmd = [str(lerobot_python), "-m", "digital_twin_armigo.agent.twin_record", "--out", str(shard)]
	cmd += ["--repo-id", REPO_ID, "--trim-unsupervised-prefix"]
	if student_cameras_only:
		cmd.append("--student-cameras-only")
	cmd += map(str, eps)
	with log.open("w", encoding="utf-8") as fh:
		subprocess.run(cmd, check=True, cwd=ROOT, env=_env(), stdout=fh, stderr=subprocess.STDOUT)
	check = (
		"import json; from lerobot.datasets.lerobot_dataset import LeRobotDataset; "
		f"ds = LeRobotDataset('{REPO_ID}', root=r'{shard}', video_backend='pyav'); "
		"ds[0]; ds[len(ds) - 1]; print(json.dumps([ds.num_episodes, ds.num_frames]))"
	)
	got = subprocess.run(
		[str(lerobot_python), "-c", check], check=True, cwd=ROOT, env=_env(), capture_output=True, text=True
	)
	n_eps, n_frames = json.loads(got.stdout.strip().splitlines()[-1])
	want = (len(eps), _expected_frames(eps))
	if (n_eps, n_frames) != want:
		raise RuntimeError(f"{shard}: read back {n_eps} episodes / {n_frames} frames, expected {want}")
	(shard / "episodes.json").write_text(json.dumps([str(e) for e in eps], indent=1), encoding="utf-8")
	mkvs = [m for e in eps for m in e.rglob("*.mkv")]
	for m in mkvs:
		m.unlink()
	print(f"[export] {shard.name}: {n_eps} episodes / {n_frames} frames verified; deleted {len(mkvs)} mkv", flush=True)


def export_pending(out: Path, lerobot_python: Path, jobs: int = 1, student_cameras_only: bool = True) -> list[Path]:
	"""Export accepted episodes not yet in a shard into new shards (``jobs`` exporters at once, pairs kept together)."""
	from concurrent.futures import ThreadPoolExecutor

	done = {e for d in _shards(out) for e in json.loads((d / "episodes.json").read_text(encoding="utf-8"))}
	eps = [e for e in accepted_episodes(out) if str(e) not in done]
	if not eps:
		return []
	pairs = [eps[i : i + 2] for i in range(0, len(eps), 2)]
	parts = [p for p in (pairs[k::jobs] for k in range(jobs)) if p]
	first = len(_shards(out))
	shards = [out / "shards" / f"shard-{first + k:03d}" for k in range(len(parts))]
	print(f"[export] {len(eps)} episodes -> {len(parts)} shards from {shards[0].name}", flush=True)
	shards[0].parent.mkdir(parents=True, exist_ok=True)
	with ThreadPoolExecutor(len(parts)) as pool:
		futs = [
			pool.submit(_export_shard, s, [e for pair in part for e in pair], lerobot_python, student_cameras_only)
			for s, part in zip(shards, parts, strict=True)
		]
		for f in futs:
			f.result()
	return shards


class BackgroundExport:
	"""Runs :func:`export_pending` after each rollout batch without blocking the next batch (one export at a time)."""

	def __init__(self, out: Path, lerobot_python: Path, jobs: int, student_cameras_only: bool = True):
		from concurrent.futures import ThreadPoolExecutor

		self.out, self.lerobot_python, self.jobs, self.student_cameras_only = (
			out,
			lerobot_python,
			jobs,
			student_cameras_only,
		)
		self.pool = ThreadPoolExecutor(1)
		self.pending = None

	def __call__(self) -> None:
		self.wait()
		self.pending = self.pool.submit(
			export_pending, self.out, self.lerobot_python, self.jobs, self.student_cameras_only
		)

	def wait(self) -> None:
		if self.pending is not None:
			self.pending.result()
			self.pending = None


def merge(out_dir: Path, inputs: list[Path], lerobot_python: Path) -> None:
	code = (
		"import sys; from pathlib import Path; sys.path.insert(0, 'scripts/vla'); "
		"from rollout_and_export import merge_shards_cmd; "
		f"merge_shards_cmd(Path(r'{out_dir}'), '{REPO_ID}', [Path(p) for p in {[str(s) for s in inputs]!r}])"
	)
	print(f"[merge] {len(inputs)} datasets -> {out_dir}", flush=True)
	subprocess.run([str(lerobot_python), "-c", code], check=True, cwd=ROOT, env=_env())


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
	p.add_argument("--export", action="store_true", help="Export and verify each batch, then delete its *.mkv")
	p.add_argument("--export-only", action="store_true", help="Skip rollouts; export what is there")
	p.add_argument("--batch", type=int, default=24, help="Cases per rollout batch when exporting")
	p.add_argument("--export-jobs", type=int, default=6, help="Concurrent LeRobot exporters per export round")
	p.add_argument("--merge-out", type=Path, default=None, help="Merge --merge-with datasets and all shards here")
	p.add_argument("--merge-with", type=Path, nargs="*", default=[], help="Datasets merged before the shards")
	p.add_argument("--lerobot-python", type=Path, default=LEROBOT_PYTHON)
	p.add_argument(
		"--student-cameras-only",
		action=argparse.BooleanOptionalAction,
		default=True,
		help="Export only canonical student cameras (omit front and top_full)",
	)
	args = p.parse_args(argv)

	from digital_twin_armigo.vla.dagger import DEFAULT_FORBID_SUITES, assert_train_layouts_only

	raw = json.loads(args.suite.read_text(encoding="utf-8-sig"))
	cases = list(raw["cases"] if isinstance(raw, dict) else raw)
	extra = [f"config/vla/val{i}.json" for i in range(2, 10) if (ROOT / f"config/vla/val{i}.json").exists()]
	assert_train_layouts_only(args.suite, cases, forbid_suites=(*DEFAULT_FORBID_SUITES, *extra))
	if args.max_episodes is not None:
		cases = cases[: args.max_episodes]
	args.out.mkdir(parents=True, exist_ok=True)
	exporting = args.export or args.export_only
	hook = (
		BackgroundExport(args.out, args.lerobot_python, args.export_jobs, args.student_cameras_only)
		if exporting
		else None
	)
	if not args.export_only:
		collect(cases, args.out, args.detector, args.workers, args.batch if exporting else None, hook)
	if exporting:
		hook.wait()
		export_pending(args.out, args.lerobot_python, args.export_jobs, args.student_cameras_only)
	if args.merge_out is not None:
		merge(args.merge_out, [*args.merge_with, *_shards(args.out)], args.lerobot_python)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
