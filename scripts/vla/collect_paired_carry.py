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
import hashlib
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


def _export_shard(
	shard: Path,
	eps: list[Path],
	lerobot_python: Path,
	student_cameras_only: bool = True,
	*,
	delete_mkv: bool = False,
	staging_root: Path | None = None,
	cleanup_log: Path | None = None,
	include_noisy_mask: bool | None = None,
	video_dir_map: dict[str, str] | None = None,
) -> None:
	"""Export ``eps`` (trimmed to their supervised part) into ``shard``, read it back, and optionally clean staging ``*.mkv``."""
	from datetime import UTC, datetime

	if delete_mkv:
		if staging_root is None:
			raise ValueError("staging_root is required when delete_mkv=True")
		sroot = Path(staging_root).resolve()
		for e in eps:
			if not Path(e).resolve().is_relative_to(sroot):
				raise ValueError(f"refusing to delete mkv outside staging_root {sroot}: {e}")
	if shard.exists():
		shutil.rmtree(shard)  # an unfinished export (no episodes.json)
	shard.parent.mkdir(parents=True, exist_ok=True)
	log = shard.with_suffix(".log")
	cmd = [str(lerobot_python), "-m", "digital_twin_armigo.agent.twin_record", "--out", str(shard)]
	cmd += ["--repo-id", REPO_ID, "--trim-unsupervised-prefix"]
	if student_cameras_only:
		cmd.append("--student-cameras-only")
	effective_noisy = student_cameras_only if include_noisy_mask is None else bool(include_noisy_mask)
	if not effective_noisy:
		cmd.append("--no-include-noisy-mask")
	elif include_noisy_mask is True:
		cmd.append("--include-noisy-mask")
	if video_dir_map:
		sub_map = {str(e): video_dir_map[str(e)] for e in eps if str(e) in video_dir_map}
		if sub_map:
			vmap_path = shard.with_suffix(".video_dirs.json")
			vmap_path.write_text(json.dumps(sub_map, indent=2) + "\n", encoding="utf-8")
			cmd += ["--video-dir-map", str(vmap_path)]
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
	if delete_mkv:
		sroot = Path(staging_root).resolve()
		mkvs = [m for e in eps for m in e.rglob("*.mkv")]
		for m in mkvs:
			if not m.resolve().is_relative_to(sroot):
				raise ValueError(f"refusing to delete mkv outside staging_root {sroot}: {m}")
		for m in mkvs:
			m.unlink()
		ev_path = cleanup_log or (sroot / "cleanup_events.jsonl")
		ev = {
			"ts": datetime.now(UTC).isoformat(timespec="seconds"),
			"policy": "staging_post_readback",
			"shard": str(shard),
			"episodes": [str(e) for e in eps],
			"deleted_mkvs": [str(m) for m in mkvs],
		}
		ev_path.parent.mkdir(parents=True, exist_ok=True)
		with ev_path.open("a", encoding="utf-8") as fh:
			fh.write(json.dumps(ev, ensure_ascii=False) + "\n")
		print(
			f"[export] {shard.name}: {n_eps} episodes / {n_frames} frames verified; deleted {len(mkvs)} staging mkv",
			flush=True,
		)
	else:
		print(
			f"[export] {shard.name}: {n_eps} episodes / {n_frames} frames verified; source mkv preserved",
			flush=True,
		)


def _is_trainable_product_name(name: str) -> bool:
	from digital_twin_armigo.governance.gold import PRODUCTS

	if name in PRODUCTS:
		return bool(PRODUCTS[name].trainable)
	if name.startswith("eval_") or name in ("harness_review", "failure_bank", "eval_test"):
		return False
	return False


def _validate_manifest_product_eligibility(manifest_path: Path) -> None:
	"""Reject non-trainable gold manifests using sibling metadata, card.md, or gold/<product>/<ver> directory."""
	import re

	mpath = Path(manifest_path).resolve()
	meta_file = mpath.parent / "manifest_meta.json"
	if meta_file.is_file():
		meta = json.loads(meta_file.read_text(encoding="utf-8-sig"))
		if meta.get("trainable") is False:
			raise ValueError(f"manifest {mpath} is marked non-trainable in manifest_meta.json")
		prod = meta.get("product")
		if prod is not None and not _is_trainable_product_name(str(prod)):
			raise ValueError(f"manifest product {prod!r} in manifest_meta.json is not trainable")

	card_file = mpath.parent / "card.md"
	if card_file.is_file():
		card_text = card_file.read_text(encoding="utf-8-sig")
		m_hdr = re.search(r"^#\s+Gold Product\s+`([^`]+)`", card_text, flags=re.MULTILINE)
		if m_hdr:
			prod = m_hdr.group(1).strip()
			if not _is_trainable_product_name(prod):
				raise ValueError(f"manifest product {prod!r} (from card.md) is not trainable")
		if re.search(r"^-\s+trainable:\s+`False`", card_text, flags=re.MULTILINE):
			raise ValueError(f"manifest {mpath} is marked trainable=False in card.md")

	# Canonical gold layout: <root>/gold/<product>/<version>/manifest.jsonl
	parts = mpath.parts
	if len(parts) >= 4 and parts[-4] == "gold":
		dir_prod = parts[-3]
		if not _is_trainable_product_name(dir_prod):
			raise ValueError(f"manifest product {dir_prod!r} (from path {mpath}) is not trainable")


def load_manifest_episodes(manifest_path: Path) -> list[Path]:
	"""Load, verify product trainability, and SHA256-verify every episode listed in a gold ``manifest.jsonl``."""
	mpath = Path(manifest_path)
	_validate_manifest_product_eligibility(mpath)
	eps: list[Path] = []
	for line_no, raw_line in enumerate(mpath.read_text(encoding="utf-8-sig").splitlines(), 1):
		line = raw_line.strip()
		if not line:
			continue
		item = json.loads(line)
		if "trainable" in item and item["trainable"] is not True:
			raise ValueError(f"manifest line {line_no}: item is not trainable (trainable={item['trainable']!r})")
		if "product" in item and item["product"] is not None:
			prod = str(item["product"])
			if not _is_trainable_product_name(prod):
				raise ValueError(f"manifest line {line_no}: product {prod!r} is not trainable")
		if "split" in item and item["split"] is not None and str(item["split"]) != "train":
			raise ValueError(
				f"manifest line {line_no}: non-train split {item['split']!r} is forbidden for training export"
			)
		ep_dir = Path(item["path"])
		expected = item.get("episode_sha256")
		ep_file = ep_dir / "episode.json"
		if not ep_file.is_file():
			raise ValueError(f"manifest line {line_no}: missing {ep_file}")
		h = hashlib.sha256()
		with ep_file.open("rb") as f:
			for chunk in iter(lambda: f.read(1 << 20), b""):
				h.update(chunk)
		actual = h.hexdigest()
		if not expected or actual != expected:
			raise ValueError(
				f"manifest line {line_no}: episode_sha256 mismatch for {ep_dir}: expected {expected!r}, got {actual!r}"
			)
		eps.append(ep_dir)
	return eps


def _prepare_deferred_videos(
	eps: list[Path],
	out: Path,
	*,
	student_cameras_only: bool,
) -> dict[str, str]:
	"""Verifiably rerender missing student videos into ``<out>/rerender_cache/<ep_key>`` (bronze stays read-only)."""
	from digital_twin_armigo.agent.deferred_video import (
		ALL_RENDER_CAMERAS,
		STUDENT_REQUIRED_CAMERAS,
		rerender_student_videos,
		verify_rerender_dir,
	)

	req_cams = STUDENT_REQUIRED_CAMERAS if student_cameras_only else ALL_RENDER_CAMERAS
	cache_root = out / "rerender_cache"
	video_dir_map: dict[str, str] = {}
	for ep in eps:
		ep_json = json.loads((ep / "episode.json").read_text(encoding="utf-8"))
		has_recorded = bool(ep_json.get("video")) and all((ep / "branch-0" / f"{c}.mkv").is_file() for c in req_cams)
		if has_recorded:
			continue
		ep_key = f"{ep.name}-{hashlib.sha256(str(ep.resolve()).encode('utf-8')).hexdigest()[:12]}"
		cache_dir = cache_root / ep_key
		reusable = False
		if (cache_dir / "rerender_manifest.json").is_file():
			try:
				verify_rerender_dir(ep, cache_dir, required_cameras=req_cams)
				reusable = True
			except ValueError:
				shutil.rmtree(cache_dir, ignore_errors=True)
		if not reusable:
			if cache_dir.exists():
				shutil.rmtree(cache_dir)
			rerender_student_videos(ep, cache_dir, student_cameras_only=student_cameras_only)
			verify_rerender_dir(ep, cache_dir, required_cameras=req_cams)
		video_dir_map[str(ep)] = str(cache_dir)
	return video_dir_map


def export_pending(
	out: Path,
	lerobot_python: Path,
	jobs: int = 1,
	student_cameras_only: bool = True,
	eps_override: list[Path] | None = None,
	*,
	delete_mkv: bool | None = None,
	include_noisy_mask: bool | None = None,
	rerender_video: bool = False,
) -> list[Path]:
	"""Export accepted episodes not yet in a shard into new shards (``jobs`` exporters at once, pairs kept together)."""
	from concurrent.futures import ThreadPoolExecutor

	if eps_override is not None and delete_mkv is True:
		raise ValueError("cannot delete source mkv when exporting from a manifest")
	effective_delete = (eps_override is None) if delete_mkv is None else bool(delete_mkv)

	done = {e for d in _shards(out) for e in json.loads((d / "episodes.json").read_text(encoding="utf-8"))}
	source_eps = list(eps_override) if eps_override is not None else accepted_episodes(out)
	eps = [e for e in source_eps if str(e) not in done]
	if not eps:
		return []
	video_dir_map = (
		_prepare_deferred_videos(eps, out, student_cameras_only=student_cameras_only) if rerender_video else {}
	)
	pairs = [eps[i : i + 2] for i in range(0, len(eps), 2)]
	parts = [p for p in (pairs[k::jobs] for k in range(jobs)) if p]
	first = len(_shards(out))
	shards = [out / "shards" / f"shard-{first + k:03d}" for k in range(len(parts))]
	print(f"[export] {len(eps)} episodes -> {len(parts)} shards from {shards[0].name}", flush=True)
	shards[0].parent.mkdir(parents=True, exist_ok=True)
	with ThreadPoolExecutor(len(parts)) as pool:
		futs = [
			pool.submit(
				_export_shard,
				s,
				[e for pair in part for e in pair],
				lerobot_python,
				student_cameras_only,
				delete_mkv=effective_delete,
				staging_root=out if effective_delete else None,
				include_noisy_mask=include_noisy_mask,
				video_dir_map=video_dir_map or None,
			)
			for s, part in zip(shards, parts, strict=True)
		]
		for f in futs:
			f.result()
	return shards


class BackgroundExport:
	"""Runs :func:`export_pending` after each rollout batch without blocking the next batch (one export at a time)."""

	def __init__(
		self,
		out: Path,
		lerobot_python: Path,
		jobs: int,
		student_cameras_only: bool = True,
		include_noisy_mask: bool | None = None,
	):
		from concurrent.futures import ThreadPoolExecutor

		self.out, self.lerobot_python, self.jobs, self.student_cameras_only, self.include_noisy_mask = (
			out,
			lerobot_python,
			jobs,
			student_cameras_only,
			include_noisy_mask,
		)
		self.pool = ThreadPoolExecutor(1)
		self.pending = None

	def __call__(self) -> None:
		self.wait()
		extra = {"include_noisy_mask": self.include_noisy_mask} if self.include_noisy_mask is not None else {}
		self.pending = self.pool.submit(
			export_pending, self.out, self.lerobot_python, self.jobs, self.student_cameras_only, **extra
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
	p.add_argument(
		"--manifest",
		type=Path,
		default=None,
		help="Gold manifest.jsonl: export only listed episodes after verifying every episode_sha256",
	)
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
	p.add_argument(
		"--include-noisy-mask",
		action=argparse.BooleanOptionalAction,
		default=None,
		help="Include mask.noisy column (default: True for student-cameras-only, False for legacy 5-camera export)",
	)
	p.add_argument(
		"--rerender-video",
		action="store_true",
		help="Verifiably rerender missing student videos from replay/motion.npz into <out>/rerender_cache (bronze read-only)",
	)
	args = p.parse_args(argv)

	if args.manifest is not None:
		manifest_eps = load_manifest_episodes(args.manifest)
		if args.max_episodes is not None:
			manifest_eps = manifest_eps[: args.max_episodes]
		args.out.mkdir(parents=True, exist_ok=True)
		export_pending(
			args.out,
			args.lerobot_python,
			args.export_jobs,
			args.student_cameras_only,
			eps_override=manifest_eps,
			include_noisy_mask=args.include_noisy_mask,
			rerender_video=args.rerender_video,
		)
		if args.merge_out is not None:
			merge(args.merge_out, [*args.merge_with, *_shards(args.out)], args.lerobot_python)
		return 0

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
		BackgroundExport(
			args.out,
			args.lerobot_python,
			args.export_jobs,
			args.student_cameras_only,
			include_noisy_mask=args.include_noisy_mask,
		)
		if exporting
		else None
	)
	if not args.export_only:
		collect(cases, args.out, args.detector, args.workers, args.batch if exporting else None, hook)
	if exporting:
		hook.wait()
		extra_kw = {}
		if args.include_noisy_mask is not None:
			extra_kw["include_noisy_mask"] = args.include_noisy_mask
		if args.rerender_video:
			extra_kw["rerender_video"] = True
		export_pending(args.out, args.lerobot_python, args.export_jobs, args.student_cameras_only, **extra_kw)
	if args.merge_out is not None:
		merge(args.merge_out, [*args.merge_with, *_shards(args.out)], args.lerobot_python)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
