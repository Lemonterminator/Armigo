"""Iterative Takeover HG-DAgger orchestration loop (plan S11).

Implements the 5-step S11 DAgger loop for up to ``--iters`` iterations:
  1. Roll out the current student policy closed-loop on **training** layouts ONLY (``assert_train_layouts_only``
     enforced strictly before every iteration).
  2. Realistic-only takeover gates hand off to the realistic expert in place without rewinding physics; keep episodes
     with ``verdict == "SUCCESS"`` and ``student_eligible == True``.
  3. Grade with ``vla_grader`` and export passing takeover trajectories ``D_takeover`` in ``E:\\il\\lerobot-cuda``.
  4. Aggregate ``D_union = D_base ∪ D_takeover`` via ``rollout_and_export.merge_shards_cmd`` and fine-tune from
     ``--init-from <prev_ckpt>`` with ``--takeover-weight 2.0`` (``scripts/vla/train.py``).
  5. Evaluate on ``config/vla/val.json`` (``scripts/vla/eval_closed_loop.py``), record validation sort rate history,
     and stop automatically when :func:`should_stop_dagger` returns True (< 2 percentage points gain over two
     consecutive iterations).

Supports dependency-injected hooks (``rollout_fn``, ``grade_fn``, ``export_fn``, ``merge_fn``, ``train_fn``,
``eval_fn``) and ``--dry-run`` for deterministic CPU testing without GPU contention.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
	sys.path.insert(0, str(ROOT / "src"))

from digital_twin_armigo.agent.vla_grader import grade, load_reference
from digital_twin_armigo.vla.dagger import (
	TakeoverConfig,
	assert_train_layouts_only,
	load_suite_cases,
	run_takeover_episode,
	should_stop_dagger,
)

LEROBOT_CUDA_PYTHON = Path("E:/il/lerobot-cuda/Scripts/python.exe")
LEROBOT_CPU_PYTHON = Path("E:/il/lerobot/Scripts/python.exe")
DEFAULT_TRAIN_SUITE = ROOT / "config/vla/train.json"
DEFAULT_VAL_SUITE = ROOT / "config/vla/val.json"
DEFAULT_DETECTOR = ROOT / "results/agent-bench/sup-v7bot-local.json"
DEFAULT_REPO_ID = "local/armigo-six-face"


RolloutFn = Callable[[int, Path | None, list[tuple], Path, TakeoverConfig], list[dict[str, Any]]]
GradeFn = Callable[[Path], dict[str, Any]]
ExportFn = Callable[[int, list[Path], Path], Path]
MergeFn = Callable[[int, list[Path], Path], Path]
TrainFn = Callable[[int, Path, Path | None, Path], Path]
EvalFn = Callable[[int, Path, Path, Path], dict[str, Any]]


def default_lerobot_python() -> Path:
	return LEROBOT_CUDA_PYTHON if LEROBOT_CUDA_PYTHON.exists() else LEROBOT_CPU_PYTHON


def _write_json(path: Path, payload: Any) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	tmp = path.with_suffix(path.suffix + ".tmp")
	tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
	tmp.replace(path)


def _assert_val_not_sealed(val_suite: Path) -> None:
	_, meta = load_suite_cases(val_suite)
	if meta.get("sealed") is True or meta.get("split") == "test":
		raise ValueError(f"DAgger validation/stopping must NEVER use sealed test suite ({val_suite})")


def _select_iter_cases(cases: list[tuple], iter_idx: int, episodes_per_iter: int | None) -> list[tuple]:
	if episodes_per_iter is None or episodes_per_iter >= len(cases):
		return list(cases)
	if episodes_per_iter <= 0:
		raise ValueError(f"episodes_per_iter must be positive, got {episodes_per_iter}")
	start = ((iter_idx - 1) * episodes_per_iter) % len(cases)
	return [cases[(start + offset) % len(cases)] for offset in range(episodes_per_iter)]


def _free_tcp_port() -> int:
	with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
		s.bind(("127.0.0.1", 0))
		return int(s.getsockname()[1])


def _wait_server_health(url: str, timeout_s: float = 120.0) -> None:
	import urllib.error
	import urllib.request

	deadline = time.monotonic() + timeout_s
	health_url = url.rstrip("/") + "/health"
	while time.monotonic() < deadline:
		try:
			with urllib.request.urlopen(health_url, timeout=2.0) as resp:
				if resp.status == 200:
					return
		except (urllib.error.URLError, TimeoutError, ConnectionError):
			time.sleep(0.5)
	raise TimeoutError(f"VLA policy server failed to become healthy at {health_url} within {timeout_s}s")


def _default_rollout(
	iter_idx: int,
	ckpt_path: Path | None,
	iter_cases: list[tuple],
	raw_dir: Path,
	takeover_cfg: TakeoverConfig,
	*,
	detector: Path,
	lerobot_python: Path,
	mode: str,
	video: bool,
	server_url: str | None = None,
) -> list[dict[str, Any]]:
	from digital_twin_armigo.inspection_scorer import default_sup_scorer
	from digital_twin_armigo.vla.client_policy import ChunkClientPolicy, RemoteChunkPredictor

	del iter_idx
	env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
	proc: subprocess.Popen | None = None
	url = server_url
	if url is None:
		if ckpt_path is None:
			raise ValueError("--init-ckpt (or --server-url) is required for live DAgger rollouts")
		port = _free_tcp_port()
		url = f"http://127.0.0.1:{port}"
		cmd = [
			str(lerobot_python),
			"-m",
			"digital_twin_armigo.vla.server",
			"--ckpt",
			str(ckpt_path),
			"--host",
			"127.0.0.1",
			"--port",
			str(port),
		]
		proc = subprocess.Popen(cmd, env=env)
		try:
			_wait_server_health(url)
		except BaseException:
			proc.kill()  # never leak the spawned server when it fails to become healthy
			proc.wait(timeout=10)
			raise

	scorer = default_sup_scorer(detector)
	predictor = RemoteChunkPredictor(url)
	resolved_mode = "sync" if mode == "auto" else mode
	if mode == "auto" and ckpt_path is not None and (ckpt_path / "vla_config.json").is_file():
		cfg = json.loads((ckpt_path / "vla_config.json").read_text(encoding="utf-8"))
		resolved_mode = "async" if cfg.get("family") == "smolvla" else "sync"

	results: list[dict[str, Any]] = []
	try:
		raw_dir.mkdir(parents=True, exist_ok=True)
		for i, case in enumerate(iter_cases):
			ep_dir = raw_dir / f"case-{i:04d}"
			policy = ChunkClientPolicy(predictor, mode=resolved_mode)
			try:
				run = run_takeover_episode(
					case,
					scorer,
					ep_dir,
					policy=policy,
					takeover_config=takeover_cfg,
					video=video,
					enforce_train_only=True,
				)
			finally:
				policy.close()
			results.append({"episode": ep_dir, "result": run["result"], "takeover": run["takeover"]})
	finally:
		if hasattr(scorer, "close"):
			scorer.close()
		if proc is not None:
			proc.terminate()
			try:
				proc.wait(timeout=10)
			except subprocess.TimeoutExpired:
				proc.kill()
	return results


def _default_export(
	iter_idx: int,
	episode_dirs: list[Path],
	d_takeover_dir: Path,
	*,
	lerobot_python: Path,
	repo_id: str,
) -> Path:
	del iter_idx
	if d_takeover_dir.exists():
		shutil.rmtree(d_takeover_dir)
	env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
	cmd = [
		str(lerobot_python),
		"-m",
		"digital_twin_armigo.agent.twin_record",
		"--out",
		str(d_takeover_dir),
		"--repo-id",
		repo_id,
		*[str(ep) for ep in episode_dirs],
	]
	subprocess.run(cmd, env=env, check=True)
	return d_takeover_dir


def _default_merge(
	iter_idx: int,
	shard_dirs: list[Path],
	d_union_dir: Path,
	*,
	lerobot_python: Path,
	repo_id: str,
) -> Path:
	del iter_idx
	env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
	cmd = [
		str(lerobot_python),
		str(ROOT / "scripts/vla/rollout_and_export.py"),
		"--merge-shards",
		str(d_union_dir),
		repo_id,
		*[str(s) for s in shard_dirs],
	]
	subprocess.run(cmd, env=env, check=True)
	return d_union_dir


def _default_train(
	iter_idx: int,
	d_union_dir: Path,
	prev_ckpt: Path | None,
	ckpt_out_dir: Path,
	*,
	lerobot_python: Path,
	family: str,
	repo_id: str,
	val_dataset: Path | None,
	takeover_weight: float,
	steps: int,
	batch_size: int,
	lr: float | None,
	num_workers: int,
) -> Path:
	del iter_idx
	env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
	cmd = [
		str(lerobot_python),
		str(ROOT / "scripts/vla/train.py"),
		"--family",
		family,
		"--dataset",
		str(d_union_dir),
		"--repo-id",
		repo_id,
		"--out",
		str(ckpt_out_dir),
		"--takeover-weight",
		str(takeover_weight),
		"--steps",
		str(steps),
		"--batch-size",
		str(batch_size),
		"--num-workers",
		str(num_workers),
	]
	if prev_ckpt is not None:
		cmd.extend(["--init-from", str(prev_ckpt)])
	if val_dataset is not None:
		cmd.extend(["--val-dataset", str(val_dataset)])
	if lr is not None:
		cmd.extend(["--lr", str(lr)])
	subprocess.run(cmd, env=env, check=True)
	best = ckpt_out_dir / "best"
	return best if best.exists() else (ckpt_out_dir / "last")


def _default_eval(
	iter_idx: int,
	ckpt_path: Path,
	val_suite: Path,
	eval_out_dir: Path,
	*,
	detector: Path,
	mode: str,
	max_episodes: int | None,
) -> dict[str, Any]:
	del iter_idx
	_assert_val_not_sealed(val_suite)
	env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
	eval_out_dir.mkdir(parents=True, exist_ok=True)
	cmd = [
		sys.executable,
		str(ROOT / "scripts/vla/eval_closed_loop.py"),
		"--ckpt",
		str(ckpt_path),
		"--suite",
		str(val_suite),
		"--out",
		str(eval_out_dir),
		"--detector",
		str(detector),
		"--mode",
		mode,
		"--no-teacher",  # the stopping rule only needs the student sort rate; skip re-running the teacher each iter
	]
	if max_episodes is not None:
		cmd.extend(["--max-episodes", str(max_episodes)])
	subprocess.run(cmd, env=env, check=True)
	# eval_closed_loop.py (S10) writes eval_summary.json = {"summary": {...}, ...}; also accept a flat summary.json.
	summary_path = eval_out_dir / "eval_summary.json"
	if not summary_path.is_file():
		summary_path = eval_out_dir / "summary.json"
	if not summary_path.is_file():
		raise FileNotFoundError(f"eval_closed_loop.py did not produce eval_summary.json in {eval_out_dir}")
	payload = json.loads(summary_path.read_text(encoding="utf-8"))
	if isinstance(payload.get("summary"), dict):
		payload = payload["summary"]
	if "student_sort_rate" not in payload:
		for alt in ("sort_rate", "success_rate"):
			if alt in payload:
				payload["student_sort_rate"] = float(payload[alt])
				break
	if "student_sort_rate" not in payload:
		raise KeyError(f"validation summary {summary_path} missing 'student_sort_rate'")
	return payload


def _delete_raw_videos(raw_dir: Path) -> int:
	removed = 0
	if raw_dir.exists():
		for mkv in raw_dir.rglob("*.mkv"):
			mkv.unlink()
			removed += 1
	return removed


def run_dagger_loop(
	*,
	train_suite: Path = DEFAULT_TRAIN_SUITE,
	val_suite: Path = DEFAULT_VAL_SUITE,
	out_dir: Path,
	init_ckpt: Path | None = None,
	base_dataset: Path | None = None,
	val_dataset: Path | None = None,
	family: str = "smolvla",
	iters: int = 5,
	episodes_per_iter: int | None = None,
	takeover_weight: float = 2.0,
	train_steps: int = 5000,
	batch_size: int = 16,
	lr: float | None = None,
	num_workers: int = 4,
	detector: Path = DEFAULT_DETECTOR,
	grader_ref_path: Path | None = None,
	lerobot_python: Path | None = None,
	repo_id: str = DEFAULT_REPO_ID,
	mode: str = "auto",
	eval_max_episodes: int | None = None,
	takeover_config: TakeoverConfig | None = None,
	min_val_gain: float = 0.02,
	stop_patience: int = 2,
	baseline_val_rate: float | None = None,
	keep_raw: bool = False,
	video: bool = True,
	dry_run: bool = False,
	rollout_fn: RolloutFn | None = None,
	grade_fn: GradeFn | None = None,
	export_fn: ExportFn | None = None,
	merge_fn: MergeFn | None = None,
	train_fn: TrainFn | None = None,
	eval_fn: EvalFn | None = None,
) -> dict[str, Any]:
	"""Execute the S11 5-step iterative Takeover HG-DAgger loop with train-only guard & early stopping."""
	if iters < 1:
		raise ValueError(f"iters must be >= 1, got {iters}")
	if takeover_weight <= 0.0:
		raise ValueError(f"takeover_weight must be > 0, got {takeover_weight}")

	train_suite = Path(train_suite)
	val_suite = Path(val_suite)
	out_dir = Path(out_dir).resolve()
	out_dir.mkdir(parents=True, exist_ok=True)

	# Step 1a: Enforce strict split isolation before doing any work.
	all_train_cases, _ = load_suite_cases(train_suite)
	assert_train_layouts_only(train_suite, all_train_cases)
	_assert_val_not_sealed(val_suite)

	takeover_cfg = takeover_config or TakeoverConfig()
	lerobot_py = lerobot_python or default_lerobot_python()
	reference = load_reference(grader_ref_path) if grade_fn is None and not dry_run else None

	# Wire default vs dry-run hooks cleanly.
	if dry_run:
		dry_rates = [0.60 + 0.04 * (k == 0) + 0.01 * k for k in range(iters)]

		def _dry_rollout(it, ckpt, cases, rdir, cfg):
			del ckpt, cfg
			rdir.mkdir(parents=True, exist_ok=True)
			rows = []
			for idx, _ in enumerate(cases):
				ep = rdir / f"case-{idx:04d}"
				ep.mkdir(parents=True, exist_ok=True)
				rows.append(
					{
						"episode": ep,
						"result": {"verdict": "SUCCESS", "student_eligible": True, "rewinds": 0, "retries": 0},
						"takeover": {"triggered": True, "frame": 10 + it, "gate": "joint_deviation"},
					}
				)
			return rows

		def _dry_grade(ep):
			return {"episode": str(ep), "passed": True, "verdict": "SUCCESS"}

		def _dry_export(it, eps, d_to):
			del it, eps
			d_to.mkdir(parents=True, exist_ok=True)
			return d_to

		def _dry_merge(it, shards, d_un):
			del it, shards
			d_un.mkdir(parents=True, exist_ok=True)
			return d_un

		def _dry_train(it, d_un, prev_c, ckpt_out):
			del it, d_un, prev_c
			ckpt_out.mkdir(parents=True, exist_ok=True)
			best = ckpt_out / "best"
			best.mkdir(parents=True, exist_ok=True)
			return best

		def _dry_eval(it, ckpt, vs, eout):
			del ckpt, vs
			eout.mkdir(parents=True, exist_ok=True)
			rate = dry_rates[min(it - 1, len(dry_rates) - 1)]
			return {"student_sort_rate": float(rate), "total_collisions": 0}

		rollout_cb = rollout_fn or _dry_rollout
		grade_cb = grade_fn or _dry_grade
		export_cb = export_fn or _dry_export
		merge_cb = merge_fn or _dry_merge
		train_cb = train_fn or _dry_train
		eval_cb = eval_fn or _dry_eval
		base_ds_path = base_dataset or (out_dir / "d_base_dryrun")
		base_ds_path.mkdir(parents=True, exist_ok=True)
	else:
		if init_ckpt is None and (rollout_fn is None or train_fn is None):
			raise ValueError("--init-ckpt is required when running without injected rollout_fn/train_fn")
		if base_dataset is None and merge_fn is None:
			raise ValueError("--base-dataset is required when running without injected merge_fn")
		base_ds_path = Path(base_dataset) if base_dataset is not None else out_dir / "d_base"
		rollout_cb = rollout_fn or (
			lambda it, ckpt, cases, rdir, cfg: _default_rollout(
				it,
				ckpt,
				cases,
				rdir,
				cfg,
				detector=detector,
				lerobot_python=lerobot_py,
				mode=mode,
				video=video,
			)
		)
		grade_cb = grade_fn or (lambda ep: grade(ep, reference=reference))
		export_cb = export_fn or (
			lambda it, eps, d_to: _default_export(it, eps, d_to, lerobot_python=lerobot_py, repo_id=repo_id)
		)
		merge_cb = merge_fn or (
			lambda it, shards, d_un: _default_merge(it, shards, d_un, lerobot_python=lerobot_py, repo_id=repo_id)
		)
		train_cb = train_fn or (
			lambda it, d_un, prev_c, ckpt_out: _default_train(
				it,
				d_un,
				prev_c,
				ckpt_out,
				lerobot_python=lerobot_py,
				family=family,
				repo_id=repo_id,
				val_dataset=val_dataset,
				takeover_weight=takeover_weight,
				steps=train_steps,
				batch_size=batch_size,
				lr=lr,
				num_workers=num_workers,
			)
		)
		eval_cb = eval_fn or (
			lambda it, ckpt, vs, eout: _default_eval(
				it, ckpt, vs, eout, detector=detector, mode=mode, max_episodes=eval_max_episodes
			)
		)

	prev_ckpt: Path | None = Path(init_ckpt) if init_ckpt is not None else None
	shards: list[Path] = [base_ds_path]
	current_union: Path = base_ds_path
	val_rates: list[float] = [] if baseline_val_rate is None else [float(baseline_val_rate)]
	iterations_log: list[dict[str, Any]] = []
	stop_reason = "max_iters"

	for it in range(1, iters + 1):
		iter_dir = out_dir / f"iter-{it:02d}"
		raw_dir = iter_dir / "raw"
		d_takeover_dir = iter_dir / "d_takeover"
		d_union_dir = iter_dir / "d_union"
		ckpt_out_dir = iter_dir / "ckpt"
		eval_out_dir = iter_dir / "val_eval"

		iter_cases = _select_iter_cases(all_train_cases, it, episodes_per_iter)
		# Re-enforce train-only suite & seed guard on this iteration's exact cases.
		assert_train_layouts_only(train_suite, iter_cases)

		# Steps 1 & 2: Roll out student with realistic-only takeover gates, keep SUCCESS & student_eligible.
		rollout_rows = rollout_cb(it, prev_ckpt, iter_cases, raw_dir, takeover_cfg)
		eligible_eps: list[Path] = []
		takeover_triggered_count = 0
		for row in rollout_rows:
			res = row["result"]
			to_info = row.get("takeover") or {}
			if to_info.get("triggered"):
				takeover_triggered_count += 1
			if res.get("verdict") == "SUCCESS" and bool(res.get("student_eligible")):
				eligible_eps.append(Path(row["episode"]))

		# Step 3: Grade & export D_takeover.
		grades = [grade_cb(ep) for ep in eligible_eps]
		passing_eps = [Path(g["episode"]) for g in grades if g.get("passed")]
		d_takeover_path: Path | None = None
		if passing_eps:
			d_takeover_path = export_cb(it, passing_eps, d_takeover_dir)
			shards.append(d_takeover_path)
			if not keep_raw and not dry_run:
				_delete_raw_videos(raw_dir)

		# Step 4: Aggregate D_union = D_base U D_takeover and fine-tune from prev_ckpt with 2x takeover weight.
		if len(shards) > 1:
			current_union = merge_cb(it, list(shards), d_union_dir)
		else:
			current_union = shards[0]

		next_ckpt = train_cb(it, current_union, prev_ckpt, ckpt_out_dir)

		# Step 5: Evaluate on val.json and check S11 stopping criterion.
		eval_payload = eval_cb(it, next_ckpt, val_suite, eval_out_dir)
		val_rate = float(eval_payload["student_sort_rate"])
		val_rates.append(val_rate)
		converged = should_stop_dagger(val_rates, min_delta=min_val_gain, patience=stop_patience)

		iter_entry = {
			"iter": it,
			"cases": len(iter_cases),
			"rolled_out": len(rollout_rows),
			"takeovers_triggered": takeover_triggered_count,
			"eligible_episodes": len(eligible_eps),
			"graded_passed_episodes": len(passing_eps),
			"d_takeover": str(d_takeover_path) if d_takeover_path is not None else None,
			"d_union": str(current_union),
			"init_from": str(prev_ckpt) if prev_ckpt is not None else None,
			"checkpoint": str(next_ckpt),
			"takeover_weight": float(takeover_weight),
			"val_sort_rate": val_rate,
			"eval": eval_payload,
			"converged": converged,
		}
		iterations_log.append(iter_entry)
		prev_ckpt = next_ckpt

		history_payload = {
			"schema": "armigo-dagger-history/v1",
			"family": family,
			"train_suite": str(train_suite),
			"val_suite": str(val_suite),
			"base_dataset": str(base_ds_path),
			"takeover_weight": float(takeover_weight),
			"min_val_gain": float(min_val_gain),
			"stop_patience": int(stop_patience),
			"baseline_val_rate": baseline_val_rate,
			"val_rates": val_rates,
			"iterations": iterations_log,
			"stopped_early": converged,
			"stop_reason": "converged" if converged else "running",
			"final_checkpoint": str(prev_ckpt),
		}
		_write_json(out_dir / "dagger_history.json", history_payload)

		if converged:
			stop_reason = "converged"
			break

	final_payload = {
		"schema": "armigo-dagger-history/v1",
		"family": family,
		"train_suite": str(train_suite),
		"val_suite": str(val_suite),
		"base_dataset": str(base_ds_path),
		"takeover_weight": float(takeover_weight),
		"min_val_gain": float(min_val_gain),
		"stop_patience": int(stop_patience),
		"baseline_val_rate": baseline_val_rate,
		"val_rates": val_rates,
		"iterations": iterations_log,
		"iters_completed": len(iterations_log),
		"stopped_early": stop_reason == "converged",
		"stop_reason": stop_reason,
		"final_checkpoint": str(prev_ckpt) if prev_ckpt is not None else None,
		"dry_run": bool(dry_run),
	}
	_write_json(out_dir / "dagger_history.json", final_payload)
	return final_payload


def build_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	p.add_argument("--init-ckpt", type=Path, default=None, help="Initial BC checkpoint directory")
	p.add_argument("--base-dataset", type=Path, default=None, help="Base LeRobot v3 training dataset (D_base)")
	p.add_argument("--val-dataset", type=Path, default=None, help="Validation LeRobot v3 dataset for train.py")
	p.add_argument("--train-suite", type=Path, default=DEFAULT_TRAIN_SUITE)
	p.add_argument("--val-suite", type=Path, default=DEFAULT_VAL_SUITE)
	p.add_argument("--out", type=Path, required=True, help="Output directory for DAgger iterations & checkpoints")
	p.add_argument("--family", choices=("smolvla", "act"), default="smolvla")
	p.add_argument("--iters", type=int, default=5)
	p.add_argument("--episodes-per-iter", type=int, default=None)
	p.add_argument("--takeover-weight", type=float, default=2.0)
	p.add_argument("--train-steps", type=int, default=5000)
	p.add_argument("--batch-size", type=int, default=16)
	p.add_argument("--lr", type=float, default=None)
	p.add_argument("--num-workers", type=int, default=4)
	p.add_argument("--detector", type=Path, default=DEFAULT_DETECTOR)
	p.add_argument("--grader-ref", type=Path, default=None)
	p.add_argument("--lerobot-python", type=Path, default=None)
	p.add_argument("--repo-id", default=DEFAULT_REPO_ID)
	p.add_argument("--mode", choices=("auto", "sync", "async"), default="auto")
	p.add_argument("--eval-max-episodes", type=int, default=None)
	p.add_argument("--deviation-thresh-rad", type=float, default=0.25)
	p.add_argument("--soft-deviation-thresh-rad", type=float, default=0.10)
	p.add_argument("--stall-frames", type=int, default=30)
	p.add_argument("--stall-eps-rad", type=float, default=0.002)
	p.add_argument("--min-val-gain", type=float, default=0.02)
	p.add_argument("--stop-patience", type=int, default=2)
	p.add_argument("--baseline-val-rate", type=float, default=None)
	p.add_argument("--keep-raw", action="store_true")
	p.add_argument("--no-video", action="store_true")
	p.add_argument("--dry-run", action="store_true", help="Run deterministic CPU dry-run without GPU/MuJoCo")
	return p


def main(argv: Sequence[str] | None = None) -> int:
	args = build_parser().parse_args(argv)
	cfg = TakeoverConfig(
		deviation_thresh_rad=args.deviation_thresh_rad,
		soft_deviation_thresh_rad=args.soft_deviation_thresh_rad,
		stall_frames=args.stall_frames,
		stall_eps_rad=args.stall_eps_rad,
	)
	summary = run_dagger_loop(
		train_suite=args.train_suite,
		val_suite=args.val_suite,
		out_dir=args.out,
		init_ckpt=args.init_ckpt,
		base_dataset=args.base_dataset,
		val_dataset=args.val_dataset,
		family=args.family,
		iters=args.iters,
		episodes_per_iter=args.episodes_per_iter,
		takeover_weight=args.takeover_weight,
		train_steps=args.train_steps,
		batch_size=args.batch_size,
		lr=args.lr,
		num_workers=args.num_workers,
		detector=args.detector,
		grader_ref_path=args.grader_ref,
		lerobot_python=args.lerobot_python,
		repo_id=args.repo_id,
		mode=args.mode,
		eval_max_episodes=args.eval_max_episodes,
		takeover_config=cfg,
		min_val_gain=args.min_val_gain,
		stop_patience=args.stop_patience,
		baseline_val_rate=args.baseline_val_rate,
		keep_raw=args.keep_raw,
		video=not args.no_video,
		dry_run=args.dry_run,
	)
	print(json.dumps(summary, indent=2))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
