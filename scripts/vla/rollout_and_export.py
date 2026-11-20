"""Batched S4 + S6 + S7 rollout, grading, and LeRobot export pipeline.

Executes a suite JSON in batches:
  1. Slices remaining cases into batch suite JSONs (skipping completed batches on ``--resume``).
  2. Runs ``python -m digital_twin_armigo.agent.twin_mcp run --sensing realistic`` (video ON).
  3. Grades every episode in the batch via ``digital_twin_armigo.agent.vla_grader.grade``.
  4. Computes batch and cumulative S4 (>= 95% expert success) & S6 (min class >= 0.8x mean) gates
     and wall-clock timings (written to ``timing.json`` and ``summary.json``).
  5. Exports passing episodes per batch in the LeRobot environment, verifies pyav readback
     (frame counts and segment labels), and only then deletes raw ``*.mkv`` files (unless ``--keep-raw``).
  6. Merges all batch LeRobot shards into a single unified v3 dataset at ``<out>/lerobot`` and
     verifies the merged dataset with ``video_backend="pyav"``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
	sys.path.insert(0, str(ROOT / "src"))

from digital_twin_armigo.agent.vla_grader import grade, load_reference, summarize

LEROBOT_CUDA_PYTHON = Path("E:/il/lerobot-cuda/Scripts/python.exe")
LEROBOT_CPU_PYTHON = Path("E:/il/lerobot/Scripts/python.exe")
DEFAULT_DETECTOR = ROOT / "results/agent-bench/sup-v7bot-local.json"
DEFAULT_REPO_ID = "local/armigo-six-face"
S4_SUCCESS_GATE = 0.95
S6_CLASS_RATIO_GATE = 0.80


def default_lerobot_python() -> Path:
	"""Return ``E:\\il\\lerobot-cuda\\Scripts\\python.exe`` if present, else ``E:\\il\\lerobot\\Scripts\\python.exe``."""
	return LEROBOT_CUDA_PYTHON if LEROBOT_CUDA_PYTHON.exists() else LEROBOT_CPU_PYTHON


def load_suite_cases(suite_path: Path) -> tuple[list, dict]:
	"""Load cases and suite metadata from a suite JSON file."""
	raw = json.loads(suite_path.read_text(encoding="utf-8-sig"))
	if isinstance(raw, dict):
		cases = list(raw["cases"])
		meta = {k: v for k, v in raw.items() if k != "cases"}
	else:
		cases = list(raw)
		meta = {}
	return cases, meta


def slice_batches(cases: list, batch_size: int, max_episodes: int | None = None) -> list[list]:
	"""Cap ``cases`` at ``max_episodes`` (if set) and split into batches of ``batch_size``."""
	if batch_size <= 0:
		raise ValueError(f"batch_size must be positive, got {batch_size}")
	if max_episodes is not None:
		if max_episodes < 0:
			raise ValueError(f"max_episodes must be non-negative, got {max_episodes}")
		cases = cases[:max_episodes]
	return [cases[i : i + batch_size] for i in range(0, len(cases), batch_size)]


def episode_expected_frames(episode_dir: Path) -> int:
	"""Read the exported branch's action frame count from ``episode.json``."""
	manifest = json.loads((episode_dir / "episode.json").read_text(encoding="utf-8"))
	branches = [b for b in manifest.get("branches", []) if (b.get("actions") or 0) > 0]
	if not branches:
		return 0
	return int(branches[-1]["actions"])


def evaluate_gates(
	grades: list[dict],
	success_gate: float = S4_SUCCESS_GATE,
	class_ratio_gate: float = S6_CLASS_RATIO_GATE,
) -> dict:
	"""Evaluate S4 (expert success >= 95%) and S6 (min class >= 0.8x mean) gates over ``grades``."""
	n = len(grades)
	expert_ok = sum(1 for g in grades if g.get("verdict") == "SUCCESS" and g["checks"]["eligible"]["status"] == "pass")
	expert_rate = (expert_ok / n) if n else 0.0
	s4_pass = n > 0 and expert_rate >= success_gate

	summary = summarize(grades)
	classes = summary["classes"]
	overall_pass = summary["passed_rate"]
	if classes:
		min_class_rate = min(row["passed_rate"] for row in classes.values())
		flagged = [
			cls
			for cls, row in classes.items()
			if overall_pass > 0 and row["passed_rate"] < class_ratio_gate * overall_pass
		]
		s6_pass = len(flagged) == 0 and overall_pass > 0
	else:
		min_class_rate = 0.0
		flagged = []
		s6_pass = False

	return {
		"episodes": n,
		"expert_success_count": expert_ok,
		"expert_success_rate": round(expert_rate, 4),
		"s4_threshold": success_gate,
		"s4_gate_pass": s4_pass,
		"overall_pass_rate": overall_pass,
		"min_class_pass_rate": round(min_class_rate, 4),
		"s6_threshold_ratio": class_ratio_gate,
		"flagged_classes": flagged,
		"s6_gate_pass": s6_pass,
		"summary": summary,
	}


def delete_raw_videos(raw_dir: Path) -> int:
	"""Delete all raw ``*.mkv`` files under ``raw_dir`` and return the count removed."""
	removed = 0
	if not raw_dir.exists():
		return 0
	for mkv in raw_dir.rglob("*.mkv"):
		mkv.unlink()
		removed += 1
	return removed


def verify_dataset(root: Path, manifest_path: Path) -> dict:
	"""LeRobot-env verification: open dataset with ``pyav``, check frame counts, segment labels, and sample decode."""
	import numpy as np
	from lerobot.datasets.lerobot_dataset import LeRobotDataset

	manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
	repo_id = manifest["repo_id"]
	expected_eps = manifest["episodes"]

	ds = LeRobotDataset(repo_id, root=root, video_backend="pyav")
	if ds.meta.total_episodes != len(expected_eps):
		raise AssertionError(f"episode count mismatch: expected {len(expected_eps)}, got {ds.meta.total_episodes}")
	expected_frames = sum(int(ep["frames"]) for ep in expected_eps)
	if len(ds) != expected_frames:
		raise AssertionError(f"frame count mismatch: expected {expected_frames}, got {len(ds)}")

	hf = ds.hf_dataset.with_format(None)
	required_cols = (
		"episode_index",
		"frame_index",
		"task_index",
		"segment.index",
		"segment.kind",
		"segment.done",
		"segment.chunk_valid_len",
	)
	for col in required_cols:
		if col not in hf.column_names:
			raise AssertionError(f"missing column {col!r} in exported dataset")

	seg_idx = np.asarray(hf["segment.index"]).reshape(-1)
	if len(seg_idx) > 0 and int(seg_idx.min()) < 0:
		raise AssertionError("negative segment.index found in exported dataset")

	ep_idx = np.asarray(hf["episode_index"]).reshape(-1)
	for i, ep in enumerate(expected_eps):
		actual_n = int((ep_idx == i).sum())
		if actual_n != int(ep["frames"]):
			raise AssertionError(f"episode {i} frame count mismatch: expected {ep['frames']}, got {actual_n}")

	seg_meta_path = root / "meta" / "segments.json"
	if not seg_meta_path.exists():
		raise AssertionError(f"missing {seg_meta_path}")
	seg_meta = json.loads(seg_meta_path.read_text(encoding="utf-8"))
	if len(seg_meta.get("episodes", [])) != len(expected_eps):
		raise AssertionError("meta/segments.json episode count mismatch")

	decoded = []
	if len(ds) > 0:
		for idx in sorted({0, len(ds) - 1}):
			sample = ds[idx]
			img_keys = [k for k in sample if k.startswith("observation.images.")]
			if not img_keys:
				raise AssertionError(f"no observation.images.* decoded at index {idx}")
			decoded.append({"index": idx, "task": sample["task"], "cameras": sorted(img_keys)})

	return {
		"verified": True,
		"episodes": ds.meta.total_episodes,
		"frames": len(ds),
		"decoded_samples": decoded,
	}


def _merge_auxiliary_meta(out_dir: Path, shard_dirs: list[Path]) -> None:
	"""Merge ``meta/source.json`` and ``meta/segments.json`` across shard directories."""
	source_eps = []
	segment_eps = []
	base_source = None
	base_segments = None
	ep_offset = 0

	for shard in shard_dirs:
		src_path = shard / "meta" / "source.json"
		seg_path = shard / "meta" / "segments.json"
		if src_path.exists():
			src_data = json.loads(src_path.read_text(encoding="utf-8"))
			if base_source is None:
				base_source = {k: v for k, v in src_data.items() if k != "episodes"}
			for row in src_data.get("episodes", []):
				source_eps.append({**row, "episode_index": ep_offset + len(source_eps) - ep_offset})
		if seg_path.exists():
			seg_data = json.loads(seg_path.read_text(encoding="utf-8"))
			if base_segments is None:
				base_segments = {k: v for k, v in seg_data.items() if k != "episodes"}
			for row in seg_data.get("episodes", []):
				segment_eps.append({**row, "episode_index": len(segment_eps)})
		ep_offset = len(segment_eps)

	# Re-index source_eps cleanly 0..N-1
	for i, row in enumerate(source_eps):
		row["episode_index"] = i

	if base_source is not None:
		(out_dir / "meta" / "source.json").write_text(
			json.dumps({**base_source, "episodes": source_eps}, indent=2) + "\n",
			encoding="utf-8",
		)
	if base_segments is not None:
		(out_dir / "meta" / "segments.json").write_text(
			json.dumps({**base_segments, "episodes": segment_eps}, indent=2) + "\n",
			encoding="utf-8",
		)


def merge_shards_cmd(out_dir: Path, repo_id: str, shard_dirs: list[Path]) -> None:
	"""LeRobot-env merge: combine per-batch LeRobot shards into ``out_dir``."""
	from lerobot.datasets.dataset_tools import merge_datasets
	from lerobot.datasets.lerobot_dataset import LeRobotDataset

	if out_dir.exists():
		shutil.rmtree(out_dir)

	if len(shard_dirs) == 1:
		shutil.copytree(shard_dirs[0], out_dir)
		return

	datasets = [LeRobotDataset(repo_id, root=s, video_backend="pyav") for s in shard_dirs]
	merge_datasets(datasets, output_repo_id=repo_id, output_dir=out_dir)
	_merge_auxiliary_meta(out_dir, shard_dirs)


def _write_json(path: Path, payload: dict) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _run_subprocess(cmd: list[str], env: dict[str, str], label: str) -> None:
	print(f"[{label}] {' '.join(cmd)}", flush=True)
	subprocess.run(cmd, env=env, check=True)


def split_contiguous(items: list, parts: int) -> list[list]:
	"""Split ``items`` into at most ``parts`` contiguous, near-equal, non-empty chunks (order preserved)."""
	parts = max(1, min(parts, len(items)))
	base, extra = divmod(len(items), parts)
	chunks, start = [], 0
	for k in range(parts):
		end = start + base + (1 if k < extra else 0)
		chunks.append(items[start:end])
		start = end
	return [c for c in chunks if c]


def _batch_shard_dirs(batch_dir: Path, done_info: dict) -> list[Path]:
	"""Shard dirs recorded in a batch's done.json (old single-shard batches have no ``shard_dirs`` key)."""
	names = done_info.get("shard_dirs")
	if names is None:
		names = ["lerobot"] if done_info.get("exported_episodes", 0) > 0 else []
	return [batch_dir / n for n in names if (batch_dir / n).exists()]


def export_batch(
	*,
	b_idx: int,
	batch_dir: Path,
	raw_dir: Path,
	passing_eps: list[Path],
	lerobot_py: Path,
	repo_id: str,
	env: dict[str, str],
	keep_raw: bool,
	export_jobs: int = 1,
) -> dict:
	"""Export passing episodes into one or more LeRobot shards in parallel, verify each, then delete raw mkv.

	With ``export_jobs > 1`` the passing episodes are split into contiguous chunks, each exported by its own
	LeRobot process into ``lerobot-NN``; concatenating the shards in order reproduces the single-shard
	episode order, so the final merge and manifest are unchanged. Raw ``*.mkv`` files are deleted only after
	every shard verified (unless ``keep_raw``).
	"""
	from concurrent.futures import ThreadPoolExecutor

	for stale in [batch_dir / "lerobot", *batch_dir.glob("lerobot-*")]:
		if stale.is_dir():
			shutil.rmtree(stale)
	for stale in batch_dir.glob("verify_manifest*.json"):
		stale.unlink()

	t0 = time.perf_counter()
	chunks = split_contiguous(passing_eps, export_jobs)
	single = len(chunks) == 1
	shard_names = ["lerobot"] if single else [f"lerobot-{k:02d}" for k in range(len(chunks))]

	def export_and_verify(k: int) -> list[dict]:
		shard_dir = batch_dir / shard_names[k]
		tag = "" if single else f" shard {k}"
		_run_subprocess(
			[
				str(lerobot_py),
				"-m",
				"digital_twin_armigo.agent.twin_record",
				"--out",
				str(shard_dir),
				"--repo-id",
				repo_id,
				*[str(p) for p in chunks[k]],
			],
			env=env,
			label=f"batch-{b_idx:04d} export{tag}",
		)
		manifest = [{"path": str(ep), "frames": episode_expected_frames(ep)} for ep in chunks[k]]
		manifest_path = batch_dir / ("verify_manifest.json" if single else f"verify_manifest-{k:02d}.json")
		_write_json(manifest_path, {"repo_id": repo_id, "episodes": manifest})
		_run_subprocess(
			[str(lerobot_py), str(Path(__file__).resolve()), "--verify-dataset", str(shard_dir), str(manifest_path)],
			env=env,
			label=f"batch-{b_idx:04d} verify{tag}",
		)
		return manifest

	if single:
		manifests = [export_and_verify(0)]
	else:
		with ThreadPoolExecutor(max_workers=len(chunks)) as pool:
			manifests = list(pool.map(export_and_verify, range(len(chunks))))
	export_s = time.perf_counter() - t0
	raw_deleted = 0 if keep_raw else delete_raw_videos(raw_dir)
	return {
		"export_s": export_s,
		"manifest_eps": [m for chunk in manifests for m in chunk],
		"shard_names": shard_names,
		"raw_deleted": raw_deleted,
	}


def run_pipeline(
	suite_path: Path,
	out_dir: Path,
	detector: Path = DEFAULT_DETECTOR,
	workers: int = 4,
	batch_size: int = 20,
	lerobot_python: Path | None = None,
	repo_id: str = DEFAULT_REPO_ID,
	grader_ref_path: Path | None = None,
	keep_raw: bool = False,
	resume: bool = False,
	max_episodes: int | None = None,
	rollout_python: Path | None = None,
	export_jobs: int = 1,
	overlap_export: bool = False,
) -> dict:
	"""Execute the batched rollout, grading, and LeRobot export pipeline.

	``export_jobs`` runs that many LeRobot export processes per batch. ``overlap_export`` exports batch N in
	the background while batch N+1 rolls out (at most one batch export in flight; batches still finish, and
	their ``done.json`` is still written, in order).
	"""
	from concurrent.futures import Future, ThreadPoolExecutor

	out_dir = out_dir.resolve()
	if not resume and (out_dir / "batches").exists():
		raise FileExistsError(f"{out_dir} already contains batches; pass --resume to continue")
	out_dir.mkdir(parents=True, exist_ok=True)

	lerobot_py = lerobot_python or default_lerobot_python()
	rollout_py = rollout_python or Path(sys.executable)
	env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
	reference = load_reference(grader_ref_path)

	cases, suite_meta = load_suite_cases(suite_path)
	batches = slice_batches(cases, batch_size=batch_size, max_episodes=max_episodes)

	all_grades: list[dict] = []
	shard_dirs: list[Path] = []
	exported_manifest_eps: list[dict] = []
	batch_timings: list[dict] = []
	pipeline_t0 = time.perf_counter()

	executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="export") if overlap_export else None
	pending: tuple[Future, dict] | None = None

	def finish(job: tuple[Future, dict]) -> None:
		"""Wait for a batch's export, then record it in order (shards, manifest, done.json, cumulative reports)."""
		future, ctx = job
		result = future.result()
		if result["manifest_eps"]:
			shard_dirs.extend(ctx["batch_dir"] / n for n in result["shard_names"])
			exported_manifest_eps.extend(result["manifest_eps"])
		b_timing = {
			"batch_index": ctx["b_idx"],
			"episodes": ctx["episodes"],
			"passing_episodes": len(result["manifest_eps"]),
			"rollout_s": round(ctx["rollout_s"], 3),
			"grade_s": round(ctx["grade_s"], 3),
			"export_s": round(result["export_s"], 3),
			"per_episode_mean_s": round(
				(ctx["rollout_s"] + ctx["grade_s"] + result["export_s"]) / max(ctx["episodes"], 1), 3
			),
		}
		batch_timings.append(b_timing)
		_write_json(
			ctx["batch_dir"] / "done.json",
			{
				"status": "done",
				"verified": True,
				"exported_episodes": len(result["manifest_eps"]),
				"raw_mkv_deleted": result["raw_deleted"],
				"manifest_episodes": result["manifest_eps"],
				"shard_dirs": result["shard_names"] if result["manifest_eps"] else [],
				"timing": b_timing,
			},
		)
		_write_cumulative_reports(
			out_dir, all_grades, batch_timings, time.perf_counter() - pipeline_t0, len(exported_manifest_eps)
		)

	def run_export(b_idx: int, batch_dir: Path, raw_dir: Path, passing_eps: list[Path]) -> dict:
		if not passing_eps:
			# No passing episodes; raw *.mkv can still be cleaned up, only after grading completed cleanly.
			return {
				"export_s": 0.0,
				"manifest_eps": [],
				"shard_names": [],
				"raw_deleted": 0 if keep_raw else delete_raw_videos(raw_dir),
			}
		return export_batch(
			b_idx=b_idx,
			batch_dir=batch_dir,
			raw_dir=raw_dir,
			passing_eps=passing_eps,
			lerobot_py=lerobot_py,
			repo_id=repo_id,
			env=env,
			keep_raw=keep_raw,
			export_jobs=export_jobs,
		)

	try:
		for b_idx, batch_cases in enumerate(batches):
			batch_dir = out_dir / "batches" / f"batch-{b_idx:04d}"
			done_file = batch_dir / "done.json"
			grades_file = batch_dir / "grades.json"
			raw_dir = batch_dir / "raw"
			batch_suite = batch_dir / "suite.json"

			if resume and done_file.exists():
				done_info = json.loads(done_file.read_text(encoding="utf-8"))
				if done_info.get("status") == "done" and grades_file.exists():
					if pending is not None:
						job, pending = pending, None
						finish(job)
					b_grades = json.loads(grades_file.read_text(encoding="utf-8"))["grades"]
					all_grades.extend(b_grades)
					batch_timings.append(done_info["timing"])
					batch_shards = _batch_shard_dirs(batch_dir, done_info)
					if done_info.get("exported_episodes", 0) > 0 and batch_shards:
						shard_dirs.extend(batch_shards)
						exported_manifest_eps.extend(done_info["manifest_episodes"])
					print(f"[batch-{b_idx:04d}] skipped (already completed on --resume)", flush=True)
					continue

			if raw_dir.exists():
				shutil.rmtree(raw_dir)
			batch_dir.mkdir(parents=True, exist_ok=True)
			suite_payload = {**suite_meta, "cases": batch_cases, "batch_index": b_idx}
			_write_json(batch_suite, suite_payload)

			# Step b: rollout with video ON (overlaps the previous batch's export when overlap_export)
			t0 = time.perf_counter()
			run_cmd = [
				str(rollout_py),
				"-m",
				"digital_twin_armigo.agent.twin_mcp",
				"run",
				"--sensing",
				"realistic",
				"--suite",
				str(batch_suite),
				"--detector",
				str(detector),
				"--workers",
				str(workers),
				"--out",
				str(raw_dir),
			]
			_run_subprocess(run_cmd, env=env, label=f"batch-{b_idx:04d} rollout")
			rollout_s = time.perf_counter() - t0

			# Step c: grade all batch episodes
			t0 = time.perf_counter()
			episode_dirs = [raw_dir / f"case-{i:04d}" for i in range(len(batch_cases))]
			b_grades = [grade(ep, reference=reference) for ep in episode_dirs]
			grade_s = time.perf_counter() - t0
			_write_json(grades_file, {"grades": b_grades, "summary": summarize(b_grades)})
			all_grades.extend(b_grades)

			# Step e/f: export passing episodes, verify readback, then delete raw *.mkv (unless --keep-raw)
			if pending is not None:
				job, pending = pending, None
				finish(job)
			passing_eps = [Path(g["episode"]) for g in b_grades if g["passed"]]
			ctx = {
				"b_idx": b_idx,
				"batch_dir": batch_dir,
				"episodes": len(batch_cases),
				"rollout_s": rollout_s,
				"grade_s": grade_s,
			}
			if executor is not None:
				pending = (executor.submit(run_export, b_idx, batch_dir, raw_dir, passing_eps), ctx)
			else:
				done: Future = Future()
				done.set_result(run_export(b_idx, batch_dir, raw_dir, passing_eps))
				finish((done, ctx))

		if pending is not None:
			job, pending = pending, None
			finish(job)
	except BaseException:
		# Let an in-flight export finish and record its batch so --resume does not redo it; keep the original error.
		if pending is not None:
			job, pending = pending, None
			try:
				finish(job)
			except Exception as exc:  # noqa: BLE001
				print(f"[export] in-flight batch export also failed: {exc!r}", flush=True)
		raise
	finally:
		if executor is not None:
			executor.shutdown(wait=True)

	# Merge all batch LeRobot shards into unified <out>/lerobot
	final_lerobot = out_dir / "lerobot"
	merge_s = 0.0
	if shard_dirs:
		t0 = time.perf_counter()
		merge_cmd = [
			str(lerobot_py),
			str(Path(__file__).resolve()),
			"--merge-shards",
			str(final_lerobot),
			repo_id,
			*[str(s) for s in shard_dirs],
		]
		_run_subprocess(merge_cmd, env=env, label="merge shards")
		final_manifest = out_dir / "verify_manifest.json"
		_write_json(final_manifest, {"repo_id": repo_id, "episodes": exported_manifest_eps})
		verify_final_cmd = [
			str(lerobot_py),
			str(Path(__file__).resolve()),
			"--verify-dataset",
			str(final_lerobot),
			str(final_manifest),
		]
		_run_subprocess(verify_final_cmd, env=env, label="verify final lerobot")
		merge_s = time.perf_counter() - t0

	total_s = time.perf_counter() - pipeline_t0
	summary_payload = _write_cumulative_reports(
		out_dir,
		all_grades,
		batch_timings,
		total_s,
		len(exported_manifest_eps),
		merge_s=merge_s,
	)
	return summary_payload


def _write_cumulative_reports(
	out_dir: Path,
	all_grades: list[dict],
	batch_timings: list[dict],
	total_s: float,
	exported_episodes: int,
	merge_s: float = 0.0,
) -> dict:
	rollout_s = round(sum(t["rollout_s"] for t in batch_timings), 3)
	grade_s = round(sum(t["grade_s"] for t in batch_timings), 3)
	export_s = round(sum(t["export_s"] for t in batch_timings) + merge_s, 3)
	n_ep = len(all_grades)
	timing_payload = {
		"episodes": n_ep,
		"rollout_s": rollout_s,
		"grade_s": grade_s,
		"export_s": export_s,
		"merge_s": round(merge_s, 3),
		"total_s": round(total_s, 3),
		"per_episode_mean_s": round((rollout_s + grade_s + export_s) / n_ep, 3) if n_ep else 0.0,
		"batches": batch_timings,
	}
	_write_json(out_dir / "timing.json", timing_payload)

	gates = evaluate_gates(all_grades)
	summary_payload = {
		**gates,
		"exported_episodes": exported_episodes,
		"timing": timing_payload,
	}
	_write_json(out_dir / "summary.json", summary_payload)
	return summary_payload


def build_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	p.add_argument("--suite", type=Path, help="Suite JSON path (e.g. config/vla/train.json).")
	p.add_argument("--out", type=Path, help="Output directory (supports --resume).")
	p.add_argument("--detector", type=Path, default=DEFAULT_DETECTOR)
	p.add_argument("--workers", type=int, default=4)
	p.add_argument("--batch-size", type=int, default=20)
	p.add_argument("--lerobot-python", type=Path, default=None)
	p.add_argument("--repo-id", default=DEFAULT_REPO_ID)
	p.add_argument("--grader-ref", type=Path, default=None)
	p.add_argument("--keep-raw", action="store_true", help="Keep raw lossless *.mkv files after verified export.")
	p.add_argument("--resume", action="store_true", help="Skip already-completed batches in --out.")
	p.add_argument("--max-episodes", type=int, default=None, help="Optional cap on total episodes.")
	p.add_argument("--export-jobs", type=int, default=1, help="Parallel LeRobot export processes per batch.")
	p.add_argument(
		"--overlap-export",
		action="store_true",
		help="Export batch N in the background while batch N+1 rolls out.",
	)
	p.add_argument("--verify-dataset", nargs=2, type=Path, metavar=("ROOT", "MANIFEST"), help=argparse.SUPPRESS)
	p.add_argument("--merge-shards", nargs="+", type=str, help=argparse.SUPPRESS)
	return p


def main():
	p = build_parser()
	a = p.parse_args()
	if a.verify_dataset:
		report = verify_dataset(a.verify_dataset[0], a.verify_dataset[1])
		print(json.dumps(report))
		return
	if a.merge_shards:
		if len(a.merge_shards) < 3:
			p.error("--merge-shards requires OUT_DIR REPO_ID SHARD...")
		out_dir = Path(a.merge_shards[0])
		repo_id = a.merge_shards[1]
		shards = [Path(s) for s in a.merge_shards[2:]]
		merge_shards_cmd(out_dir, repo_id, shards)
		return
	if not a.suite or not a.out:
		p.error("--suite and --out are required")

	summary = run_pipeline(
		suite_path=a.suite,
		out_dir=a.out,
		detector=a.detector,
		workers=a.workers,
		batch_size=a.batch_size,
		lerobot_python=a.lerobot_python,
		repo_id=a.repo_id,
		grader_ref_path=a.grader_ref,
		keep_raw=a.keep_raw,
		resume=a.resume,
		max_episodes=a.max_episodes,
		export_jobs=a.export_jobs,
		overlap_export=a.overlap_export,
	)
	print(
		f"Done: {summary['episodes']} episodes, "
		f"expert_success={summary['expert_success_rate']:.3f} (S4={'PASS' if summary['s4_gate_pass'] else 'FAIL'}), "
		f"overall_pass={summary['overall_pass_rate']:.3f} (S6={'PASS' if summary['s6_gate_pass'] else 'FAIL'}), "
		f"exported={summary['exported_episodes']}"
	)


if __name__ == "__main__":
	main()
