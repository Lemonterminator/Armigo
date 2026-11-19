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

from digital_twin_armigo.agent.vla_grader import grade, load_reference, summarize  # noqa: E402

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
	expert_ok = sum(
		1
		for g in grades
		if g.get("verdict") == "SUCCESS" and g["checks"]["eligible"]["status"] == "pass"
	)
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
		raise AssertionError(
			f"episode count mismatch: expected {len(expected_eps)}, got {ds.meta.total_episodes}"
		)
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
			raise AssertionError(
				f"episode {i} frame count mismatch: expected {ep['frames']}, got {actual_n}"
			)

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
) -> dict:
	"""Execute the batched rollout, grading, and LeRobot export pipeline."""
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

	for b_idx, batch_cases in enumerate(batches):
		batch_dir = out_dir / "batches" / f"batch-{b_idx:04d}"
		done_file = batch_dir / "done.json"
		grades_file = batch_dir / "grades.json"
		raw_dir = batch_dir / "raw"
		shard_dir = batch_dir / "lerobot"
		batch_suite = batch_dir / "suite.json"

		if resume and done_file.exists():
			done_info = json.loads(done_file.read_text(encoding="utf-8"))
			if done_info.get("status") == "done" and grades_file.exists():
				b_grades = json.loads(grades_file.read_text(encoding="utf-8"))["grades"]
				all_grades.extend(b_grades)
				batch_timings.append(done_info["timing"])
				if done_info.get("exported_episodes", 0) > 0 and shard_dir.exists():
					shard_dirs.append(shard_dir)
					exported_manifest_eps.extend(done_info["manifest_episodes"])
				print(f"[batch-{b_idx:04d}] skipped (already completed on --resume)", flush=True)
				continue

		batch_dir.mkdir(parents=True, exist_ok=True)
		suite_payload = {**suite_meta, "cases": batch_cases, "batch_index": b_idx}
		_write_json(batch_suite, suite_payload)

		# Step b: rollout with video ON
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

		# Step e: export passing episodes & verify readback
		passing_eps = [Path(g["episode"]) for g in b_grades if g["passed"]]
		export_s = 0.0
		manifest_eps: list[dict] = []
		raw_deleted = 0

		if passing_eps:
			t0 = time.perf_counter()
			if shard_dir.exists():
				shutil.rmtree(shard_dir)
			export_cmd = [
				str(lerobot_py),
				"-m",
				"digital_twin_armigo.agent.twin_record",
				"--out",
				str(shard_dir),
				"--repo-id",
				repo_id,
				*[str(p) for p in passing_eps],
			]
			_run_subprocess(export_cmd, env=env, label=f"batch-{b_idx:04d} export")

			manifest_eps = [
				{"path": str(ep), "frames": episode_expected_frames(ep)}
				for ep in passing_eps
			]
			verify_manifest = batch_dir / "verify_manifest.json"
			_write_json(verify_manifest, {"repo_id": repo_id, "episodes": manifest_eps})
			verify_cmd = [
				str(lerobot_py),
				str(Path(__file__).resolve()),
				"--verify-dataset",
				str(shard_dir),
				str(verify_manifest),
			]
			_run_subprocess(verify_cmd, env=env, label=f"batch-{b_idx:04d} verify")
			export_s = time.perf_counter() - t0

			shard_dirs.append(shard_dir)
			exported_manifest_eps.extend(manifest_eps)

			# Step f: delete raw *.mkv only after verified readback (unless --keep-raw)
			if not keep_raw:
				raw_deleted = delete_raw_videos(raw_dir)
		elif not keep_raw:
			# No passing episodes in this batch; raw *.mkv can still be cleaned up if desired,
			# only after grading completed cleanly.
			raw_deleted = delete_raw_videos(raw_dir)

		b_timing = {
			"batch_index": b_idx,
			"episodes": len(batch_cases),
			"passing_episodes": len(passing_eps),
			"rollout_s": round(rollout_s, 3),
			"grade_s": round(grade_s, 3),
			"export_s": round(export_s, 3),
			"per_episode_mean_s": round((rollout_s + grade_s + export_s) / max(len(batch_cases), 1), 3),
		}
		batch_timings.append(b_timing)
		_write_json(
			done_file,
			{
				"status": "done",
				"verified": True,
				"exported_episodes": len(passing_eps),
				"raw_mkv_deleted": raw_deleted,
				"manifest_episodes": manifest_eps,
				"timing": b_timing,
			},
		)

		# Update cumulative timing & summary after each batch
		_write_cumulative_reports(
			out_dir, all_grades, batch_timings, time.perf_counter() - pipeline_t0, len(exported_manifest_eps)
		)

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
	)
	print(
		f"Done: {summary['episodes']} episodes, "
		f"expert_success={summary['expert_success_rate']:.3f} (S4={'PASS' if summary['s4_gate_pass'] else 'FAIL'}), "
		f"overall_pass={summary['overall_pass_rate']:.3f} (S6={'PASS' if summary['s6_gate_pass'] else 'FAIL'}), "
		f"exported={summary['exported_episodes']}"
	)


if __name__ == "__main__":
	main()
