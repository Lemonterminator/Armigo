"""Closed-loop VLA student evaluation in MuJoCo (plan S10).

Connects ``StudentTwin`` (via ``ChunkClientPolicy``) to either a running policy server
(``--server-url``) or an auto-spawned ``vla/server.py`` subprocess in ``E:\\il\\lerobot-cuda``
(``--ckpt``). Evaluates episodes sequentially using ``evaluate`` + S6 collision/crop rules
over ALL attempts (including ``UNKNOWN`` / ``FAILED``), computes missed defects, paired
teacher-student difference with 95% bootstrap CI across layouts, and checks the S10 pilot
gate (``student_sort_rate >= 0.7 * teacher_sort_rate`` and ``total_collisions == 0``).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
	sys.path.insert(0, str(ROOT / "src"))

from digital_twin_armigo.agent.student_rollout import run_episode
from digital_twin_armigo.agent.twin_mujoco import load_suite
from digital_twin_armigo.agent.vla_grader import find_episodes, grade, load_reference
from digital_twin_armigo.agent.vla_grader import summarize as summarize_grades
from digital_twin_armigo.vla.client_policy import ChunkClientPolicy, RemoteChunkPredictor
from digital_twin_armigo.vla.server import ChunkPredictor

LEROBOT_CUDA_PYTHON = Path("E:/il/lerobot-cuda/Scripts/python.exe")
DEFAULT_SUITE = ROOT / "config/vla/val.json"
DEFAULT_DETECTOR = ROOT / "results/agent-bench/sup-v7bot-local.json"
PILOT_TEACHER_RATIO_GATE = 0.70


class SealedSuiteError(ValueError):
	"""Raised when evaluating the sealed test suite without ``--allow-sealed``."""


def assert_suite_allowed(suite_path: Path, *, allow_sealed: bool = False) -> list[tuple]:
	"""Load ``suite_path`` and refuse sealed suites (e.g. ``config/vla/test.json``) unless allowed."""
	path = Path(suite_path)
	raw = json.loads(path.read_text(encoding="utf-8-sig"))
	is_sealed_flag = isinstance(raw, dict) and bool(raw.get("sealed", False))
	sealed_test_path = (ROOT / "config/vla/test.json").resolve()
	is_test_file = path.name == "test.json" or path.resolve() == sealed_test_path
	if (is_sealed_flag or is_test_file) and not allow_sealed:
		raise SealedSuiteError(
			f"Refusing to run evaluation on sealed suite {path}; pass --allow-sealed only for final frozen evaluation."
		)
	return load_suite(path)


def resolve_eval_mode(mode: str, health: dict[str, Any] | None = None) -> str:
	"""Resolve ``--mode auto|sync|async`` using server ``/health`` family metadata."""
	if mode in ("sync", "async"):
		return mode
	if mode != "auto":
		raise ValueError(f"unsupported mode {mode!r}")
	family = str((health or {}).get("family", "")).lower()
	if family == "smolvla":
		return "async"
	if family == "act":
		return "sync"
	return "sync"


def bootstrap_paired_ci(
	student_outcomes: Sequence[float],
	teacher_outcomes: Sequence[float],
	*,
	n_boot: int = 10_000,
	alpha: float = 0.05,
	seed: int = 0,
) -> dict[str, Any]:
	"""Compute paired difference (student - teacher) and percentile bootstrap CI across layouts."""
	stu = np.asarray(student_outcomes, dtype=np.float64)
	tea = np.asarray(teacher_outcomes, dtype=np.float64)
	if stu.shape != tea.shape or stu.ndim != 1 or len(stu) == 0:
		raise ValueError(
			f"student and teacher outcomes must be non-empty 1D arrays of equal length, got {stu.shape} vs {tea.shape}"
		)

	diffs = stu - tea
	n = len(diffs)
	mean_diff = float(diffs.mean())
	rng = np.random.default_rng(seed)

	boot_chunks: list[np.ndarray] = []
	chunk_size = 2000
	remaining = int(n_boot)
	while remaining > 0:
		cur = min(chunk_size, remaining)
		idx = rng.integers(0, n, size=(cur, n))
		boot_chunks.append(diffs[idx].mean(axis=1))
		remaining -= cur
	boot_means = np.concatenate(boot_chunks) if boot_chunks else np.array([mean_diff], dtype=np.float64)

	lo, hi = np.percentile(boot_means, [100.0 * (alpha / 2.0), 100.0 * (1.0 - alpha / 2.0)])
	return {
		"n": int(n),
		"student_rate": float(stu.mean()),
		"teacher_rate": float(tea.mean()),
		"mean_diff": mean_diff,
		"ci_95": [float(lo), float(hi)],
		"n_boot": int(n_boot),
		"student_wins": int(((stu > 0.5) & (tea < 0.5)).sum()),
		"teacher_wins": int(((stu < 0.5) & (tea > 0.5)).sum()),
		"both_pass": int(((stu > 0.5) & (tea > 0.5)).sum()),
		"both_fail": int(((stu < 0.5) & (tea < 0.5)).sum()),
	}


def _episode_case(ep_dir: Path) -> list | None:
	"""The full ``[seed, positive, face, ...]`` case recorded with an episode, or None."""
	for rel in ("task.json", "run.json"):
		p = ep_dir / rel
		if p.exists():
			data = json.loads(p.read_text(encoding="utf-8"))
			if isinstance(data.get("case"), list) and data["case"]:
				return list(data["case"])
	ep_json = ep_dir / "episode.json"
	if ep_json.exists():
		manifest = json.loads(ep_json.read_text(encoding="utf-8"))
		case = (manifest.get("provenance") or {}).get("case")
		if isinstance(case, list) and case:
			return list(case)
	return None


def _episode_case_seed(ep_dir: Path) -> int | None:
	for rel, key in (("task.json", "case"), ("run.json", "case")):
		p = ep_dir / rel
		if p.exists():
			data = json.loads(p.read_text(encoding="utf-8"))
			if isinstance(data.get(key), list) and data[key]:
				return int(data[key][0])
	ep_json = ep_dir / "episode.json"
	if ep_json.exists():
		manifest = json.loads(ep_json.read_text(encoding="utf-8"))
		case = (manifest.get("provenance") or {}).get("case")
		if isinstance(case, list) and case:
			return int(case[0])
	return None


def load_teacher_baseline(
	teacher_dir: Path,
	cases: Sequence[tuple] | None = None,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any] | None]:
	"""Load per-layout teacher outcomes and/or ``summary.json`` from ``teacher_dir``."""
	teacher_dir = Path(teacher_dir)
	summary_path = teacher_dir / "summary.json"
	teacher_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None

	by_seed: dict[int, dict[str, Any]] = {}
	for ep_dir in find_episodes(teacher_dir):
		manifest = json.loads((ep_dir / "episode.json").read_text(encoding="utf-8"))
		result = manifest.get("result") or {}
		seed = _episode_case_seed(ep_dir)
		if seed is None and cases is not None and ep_dir.name.startswith("case-"):
			with contextlib.suppress(ValueError):
				idx = int(ep_dir.name.split("-", 1)[1])
				if 0 <= idx < len(cases):
					seed = int(cases[idx][0])
		if seed is None:
			continue
		if seed in by_seed:
			raise ValueError(
				f"teacher seed {seed} appears twice ({by_seed[seed]['episode_dir']} and {ep_dir}); "
				"pairing would be ambiguous"
			)
		by_seed[seed] = {
			"seed": seed,
			"case": _episode_case(ep_dir),
			"verdict": result.get("verdict"),
			"label": result.get("label"),
			"label_correct": bool(result.get("label_correct", False)),
			"episode_dir": str(ep_dir),
		}
	return by_seed, teacher_summary


def _episode_record_from_dir(ep_dir: Path, case: tuple) -> dict[str, Any]:
	manifest = json.loads((ep_dir / "episode.json").read_text(encoding="utf-8"))
	result = manifest.get("result") or {}
	events_path = ep_dir / "events.jsonl"
	events = (
		[json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line]
		if events_path.exists()
		else []
	)
	from digital_twin_armigo.agent.perturbations import find_perturbation_detection

	has_perturb = any(e.get("kind") == "perturbation" for e in events)
	# S5's matcher: per-type detection events at/after the perturbation frame (covers the nudge's
	# brick_perceived(moved) path, which emits no failure/retry).
	has_detect = has_perturb and find_perturbation_detection(events) is not None
	return {
		"seed": int(case[0]),
		"case": list(case),
		"episode_dir": str(ep_dir),
		"true_label": "defective" if bool(case[1]) else "normal",
		"true_face": int(case[2]) if bool(case[1]) else None,
		"verdict": result.get("verdict"),
		"label": result.get("label"),
		"label_correct": bool(result.get("label_correct", False)),
		"destination": result.get("destination"),
		"retries": int(result.get("retries", 0)),
		"rewinds": int(result.get("rewinds", 0)),
		"sim_frames": int(result.get("sim_frames", 0)),
		"has_perturbation": has_perturb,
		"perturbation_detected": has_perturb and has_detect,
	}


def summarize_closed_loop(
	student_records: Sequence[dict[str, Any]],
	*,
	grades: Sequence[dict[str, Any]],
	teacher_by_seed: dict[int, dict[str, Any]] | None = None,
	teacher_summary: dict[str, Any] | None = None,
	n_boot: int = 10_000,
	boot_seed: int = 0,
) -> dict[str, Any]:
	"""Aggregate S10 closed-loop outcomes, S6 collision/crop stats, paired CI, and pilot gate."""
	if len(student_records) != len(grades):
		raise ValueError(f"student_records ({len(student_records)}) != grades ({len(grades)})")
	n = len(student_records)
	verdicts = {"SUCCESS": 0, "UNKNOWN": 0, "FAILED": 0}
	for r in student_records:
		v = str(r.get("verdict") or "FAILED")
		verdicts[v] = verdicts.get(v, 0) + 1

	sort_successes = verdicts["SUCCESS"]
	student_sort_rate = (sort_successes / n) if n else 0.0
	label_correct_count = sum(1 for r in student_records if r.get("label_correct"))
	label_accuracy = (label_correct_count / n) if n else 0.0

	defective_recs = [r for r in student_records if r.get("true_label") == "defective"]
	normal_recs = [r for r in student_records if r.get("true_label") == "normal"]
	missed_defects = [r for r in defective_recs if r.get("label") != "defective"]
	missed_false_normal = sum(1 for r in missed_defects if r.get("label") == "normal")
	missed_unknown = len(missed_defects) - missed_false_normal
	false_positives = sum(1 for r in normal_recs if r.get("label") == "defective")

	# Collisions over ALL attempts (including UNKNOWN and FAILED) and jaw-env contacts.
	robot_contact_frames = 0
	jaw_env_contact_frames = 0
	collision_episodes = 0
	collision_unavailable = 0
	max_robot_n = 0.0
	max_jaw_env_n = 0.0
	collisions_by_verdict: dict[str, dict[str, int]] = {}

	for rec, g in zip(student_records, grades, strict=True):
		v = str(rec.get("verdict") or "FAILED")
		bucket = collisions_by_verdict.setdefault(
			v, {"episodes": 0, "collision_episodes": 0, "robot_contact_frames": 0, "jaw_env_contact_frames": 0}
		)
		bucket["episodes"] += 1
		col = (g.get("checks") or {}).get("collisions") or {}
		status = col.get("status")
		if status == "unavailable":
			collision_unavailable += 1
			continue
		r_frames = int(col.get("robot_contact_frames", 0))
		j_frames = int(col.get("jaw_env_contact_frames", 0))
		robot_contact_frames += r_frames
		jaw_env_contact_frames += j_frames
		bucket["robot_contact_frames"] += r_frames
		bucket["jaw_env_contact_frames"] += j_frames
		if status == "fail" or r_frames > 0 or j_frames > 0:
			collision_episodes += 1
			bucket["collision_episodes"] += 1
		max_robot_n = max(max_robot_n, float(col.get("max_robot_contact_n", 0.0)))
		max_jaw_env_n = max(max_jaw_env_n, float(col.get("max_jaw_env_contact_n", 0.0)))

	total_collisions = robot_contact_frames + jaw_env_contact_frames
	grader_sum = summarize_grades(list(grades))
	crop_pass = sum(1 for g in grades if g["checks"]["decision_crop"]["status"] == "pass")
	belief_pass = sum(1 for g in grades if g["checks"]["belief_error"]["status"] == "pass")
	smooth_pass = sum(1 for g in grades if g["checks"]["smoothness"]["status"] == "pass")

	# Perturbation / retry breakdown
	perturb_recs = [r for r in student_records if r.get("has_perturbation")]
	retry_recs = [r for r in student_records if r.get("retries", 0) > 0]
	perturbation_stats = {
		"perturbation_attempted": len(perturb_recs),
		"perturbation_detected": sum(1 for r in perturb_recs if r.get("perturbation_detected")),
		"perturbation_sorted": sum(1 for r in perturb_recs if r.get("verdict") == "SUCCESS"),
		"episodes_with_retry": len(retry_recs),
		"retries_total": sum(int(r.get("retries", 0)) for r in student_records),
		"retry_sorted": sum(1 for r in retry_recs if r.get("verdict") == "SUCCESS"),
	}

	# Teacher paired comparison
	paired_diff: dict[str, Any] | None = None
	teacher_sort_rate: float | None = None
	pairing: dict[str, Any] | None = None
	if teacher_by_seed:
		stu_vec: list[float] = []
		tea_vec: list[float] = []
		unpaired: list[int] = []
		case_mismatch: list[int] = []
		for r in student_records:
			t_row = teacher_by_seed.get(int(r["seed"]))
			if t_row is None:
				unpaired.append(int(r["seed"]))
				continue
			t_case = t_row.get("case")
			if t_case is not None and list(t_case)[:3] != list(r["case"])[:3]:
				case_mismatch.append(int(r["seed"]))
				continue
			stu_vec.append(1.0 if r.get("verdict") == "SUCCESS" else 0.0)
			tea_vec.append(1.0 if t_row.get("verdict") == "SUCCESS" else 0.0)
		# The gate compares rates over ONE set of layouts: every student layout must have exactly its teacher.
		pairing = {
			"students": n,
			"paired": len(stu_vec),
			"unpaired_seeds": unpaired,
			"case_mismatch_seeds": case_mismatch,
			"complete": bool(n > 0 and len(stu_vec) == n),
		}
		if stu_vec:
			paired_diff = bootstrap_paired_ci(stu_vec, tea_vec, n_boot=n_boot, seed=boot_seed)
			teacher_sort_rate = float(np.mean(tea_vec))

	if teacher_sort_rate is None and teacher_summary is not None:
		if "expert_success_rate" in teacher_summary:
			teacher_sort_rate = float(teacher_summary["expert_success_rate"])
		elif "student_sort_rate" in teacher_summary:
			teacher_sort_rate = float(teacher_summary["student_sort_rate"])

	rate_diff = (student_sort_rate - teacher_sort_rate) if teacher_sort_rate is not None else None
	sort_ratio = (
		(student_sort_rate / teacher_sort_rate) if (teacher_sort_rate is not None and teacher_sort_rate > 0.0) else None
	)

	collision_gate_pass = (total_collisions == 0) and (collision_unavailable == 0) and (n > 0)
	if pairing is not None and not pairing["complete"]:
		# Partial pairing gives a student rate and a teacher rate over different layouts: no gate.
		sort_gate_threshold = None
		sort_rate_gate_pass = None
		pilot_gate_pass = None
	elif teacher_sort_rate is not None:
		sort_gate_threshold = PILOT_TEACHER_RATIO_GATE * teacher_sort_rate
		sort_rate_gate_pass: bool | None = student_sort_rate >= sort_gate_threshold
		pilot_gate_pass: bool | None = bool(sort_rate_gate_pass and collision_gate_pass)
	else:
		sort_gate_threshold = None
		sort_rate_gate_pass = None
		pilot_gate_pass = None

	return {
		"episodes": n,
		"verdicts": verdicts,
		"sort_successes": sort_successes,
		"student_sort_rate": round(student_sort_rate, 4),
		"label_accuracy": round(label_accuracy, 4),
		"defective_episodes": len(defective_recs),
		"missed_defects": len(missed_defects),
		"missed_defect_false_normal": missed_false_normal,
		"missed_defect_unknown": missed_unknown,
		"missed_defect_rate": round(len(missed_defects) / len(defective_recs), 4) if defective_recs else 0.0,
		"false_positive_defects": false_positives,
		"collisions": {
			"total_collisions": total_collisions,
			"collision_episodes": collision_episodes,
			"collision_unavailable_episodes": collision_unavailable,
			"robot_contact_frames": robot_contact_frames,
			"jaw_env_contact_frames": jaw_env_contact_frames,
			"max_robot_contact_n": round(max_robot_n, 3),
			"max_jaw_env_contact_n": round(max_jaw_env_n, 3),
			"by_verdict": collisions_by_verdict,
		},
		"s6_checks": {
			"decision_crop_pass_rate": round(crop_pass / n, 4) if n else 0.0,
			"belief_error_pass_rate": round(belief_pass / n, 4) if n else 0.0,
			"smoothness_pass_rate": round(smooth_pass / n, 4) if n else 0.0,
			"grader_summary": grader_sum,
		},
		"perturbations": perturbation_stats,
		"teacher_sort_rate": round(teacher_sort_rate, 4) if teacher_sort_rate is not None else None,
		"rate_diff": round(rate_diff, 4) if rate_diff is not None else None,
		"sort_rate_ratio": round(sort_ratio, 4) if sort_ratio is not None else None,
		"paired_teacher_student": paired_diff,
		"teacher_pairing": pairing,
		"pilot_gate": {
			"ratio_threshold": PILOT_TEACHER_RATIO_GATE,
			"sort_rate_threshold": round(sort_gate_threshold, 4) if sort_gate_threshold is not None else None,
			"sort_rate_gate_pass": sort_rate_gate_pass,
			"collision_gate_pass": collision_gate_pass,
			"pilot_gate_pass": pilot_gate_pass,
		},
	}


@contextlib.contextmanager
def managed_policy_server(
	*,
	ckpt: Path | None = None,
	server_url: str | None = None,
	lerobot_python: Path = LEROBOT_CUDA_PYTHON,
	host: str = "127.0.0.1",
	port: int = 0,
	device: str = "auto",
	allow_download: bool = False,
	startup_timeout_s: float = 120.0,
) -> Iterator[str]:
	"""Yield a live policy server URL, spawning ``vla/server.py`` when ``--ckpt`` is given."""
	if bool(ckpt is not None) == bool(server_url is not None):
		raise ValueError("specify exactly one of --ckpt or --server-url")
	if server_url is not None:
		yield str(server_url).rstrip("/")
		return

	assert ckpt is not None
	if not lerobot_python.exists():
		raise FileNotFoundError(f"CUDA/LeRobot Python executable not found: {lerobot_python}")

	with tempfile.TemporaryDirectory(prefix="armigo-vla-server-", ignore_cleanup_errors=True) as tmp_dir:
		ready_file = Path(tmp_dir) / "ready.json"
		log_file = Path(tmp_dir) / "server.log"
		env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
		cmd = [
			str(lerobot_python),
			"-m",
			"digital_twin_armigo.vla.server",
			"--ckpt",
			str(ckpt),
			"--host",
			host,
			"--port",
			str(port),
			"--device",
			device,
			"--ready-file",
			str(ready_file),
			"--quiet",
		]
		if allow_download:
			cmd.append("--allow-download")

		with open(log_file, "wb") as log_handle:
			proc = subprocess.Popen(cmd, env=env, stdout=log_handle, stderr=subprocess.STDOUT)
		url: str | None = None
		try:
			deadline = time.monotonic() + float(startup_timeout_s)
			while time.monotonic() < deadline:
				if proc.poll() is not None:
					tail = log_file.read_text(encoding="utf-8", errors="replace")[-3000:]
					raise RuntimeError(f"VLA policy server exited early (code {proc.returncode}):\n{tail}")
				if ready_file.exists():
					info = json.loads(ready_file.read_text(encoding="utf-8"))
					url = str(info["url"]).rstrip("/")
					break
				time.sleep(0.25)
			if url is None:
				tail = log_file.read_text(encoding="utf-8", errors="replace")[-3000:]
				raise TimeoutError(f"VLA policy server failed to start within {startup_timeout_s}s:\n{tail}")

			# Verify /health responds 200 OK
			with RemoteChunkPredictor(url, timeout_s=10.0) as probe:
				while time.monotonic() < deadline:
					if proc.poll() is not None:
						tail = log_file.read_text(encoding="utf-8", errors="replace")[-3000:]
						raise RuntimeError(
							f"VLA policy server crashed before /health (code {proc.returncode}):\n{tail}"
						)
					try:
						probe.health()
						break
					except Exception:  # noqa: BLE001
						time.sleep(0.25)
			yield url
		finally:
			if proc.poll() is None and url is not None:
				with contextlib.suppress(Exception):
					with RemoteChunkPredictor(url, timeout_s=2.0) as killer:
						killer._request_json("POST", "/shutdown")
					proc.wait(timeout=5.0)
			if proc.poll() is None:
				if os.name == "nt":
					with contextlib.suppress(Exception):
						subprocess.run(
							["taskkill", "/F", "/T", "/PID", str(proc.pid)],
							stdout=subprocess.DEVNULL,
							stderr=subprocess.DEVNULL,
							timeout=5.0,
							check=False,
						)
				else:
					proc.terminate()
				try:
					proc.wait(timeout=5.0)
				except subprocess.TimeoutExpired:
					proc.kill()
					proc.wait(timeout=5.0)


def _write_task_json_if_missing(ep_dir: Path, case: tuple) -> None:
	if ep_dir.exists() and not (ep_dir / "task.json").exists():
		(ep_dir / "task.json").write_text(json.dumps({"case": list(case)}, indent=2) + "\n", encoding="utf-8")


def evaluate_closed_loop(
	*,
	predictor: ChunkPredictor,
	scorer: Any,
	cases: Sequence[tuple],
	out_dir: Path,
	mode: str = "auto",
	replan_interval: int | None = None,
	prefetch_ahead: int | None = None,
	end_threshold: float | None = None,
	temporal_ensemble: float | None = None,
	stop_policy: str = "latest",
	teacher_runs: Path | None = None,
	run_teacher: bool = True,
	grader_ref: Path | None = None,
	video: bool = False,
	n_boot: int = 10_000,
) -> dict[str, Any]:
	"""Run sequential ``StudentTwin`` episodes, grade with S6 rules, and compare with teacher."""
	out_dir = Path(out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)
	health = predictor.health()
	resolved_mode = resolve_eval_mode(mode, health)
	reference = load_reference(grader_ref)

	student_records: list[dict[str, Any]] = []
	student_grades: list[dict[str, Any]] = []
	client_stats_list: list[dict[str, Any]] = []

	for idx, case in enumerate(cases):
		ep_dir = out_dir / "student" / f"case-{idx:04d}"
		if (ep_dir / "episode.json").exists():
			recorded = _episode_case(ep_dir)
			if recorded is None or list(recorded)[:3] != list(case)[:3]:
				raise ValueError(
					f"{ep_dir} holds case {recorded}, not suite case {list(case)}: use a fresh --out per checkpoint/suite"
				)
			rec = _episode_record_from_dir(ep_dir, case)
			g = grade(ep_dir, reference=reference)
			student_records.append(rec)
			student_grades.append(g)
			continue

		policy = ChunkClientPolicy(
			predictor,
			mode=resolved_mode,
			replan_interval=replan_interval,
			prefetch_ahead=prefetch_ahead,
			end_threshold=end_threshold,
			temporal_ensemble=temporal_ensemble,
			stop_policy=stop_policy,
		)
		try:
			run_episode(case, scorer, ep_dir, policy=policy, sensing="realistic", video=video)
		finally:
			_write_task_json_if_missing(ep_dir, case)
			policy.close()

		rec = _episode_record_from_dir(ep_dir, case)
		g = grade(ep_dir, reference=reference)
		rec["client_stats"] = {
			"calls": policy.stats.calls,
			"sync_calls": policy.stats.sync_calls,
			"async_prefetches": policy.stats.async_prefetches,
			"cancelled_prefetches": policy.stats.cancelled_prefetches,
			"stale_discarded": policy.stats.stale_discarded,
			"stop_events": policy.stats.stop_events,
			"mean_latency_ms": policy.stats.mean_latency_ms,
		}
		client_stats_list.append(rec["client_stats"])
		student_records.append(rec)
		student_grades.append(g)

	teacher_by_seed: dict[int, dict[str, Any]] = {}
	teacher_summary: dict[str, Any] | None = None
	if teacher_runs is not None:
		teacher_by_seed, teacher_summary = load_teacher_baseline(teacher_runs, cases=cases)
	elif run_teacher:
		teacher_root = out_dir / "teacher"
		for idx, case in enumerate(cases):
			t_dir = teacher_root / f"case-{idx:04d}"
			if not (t_dir / "episode.json").exists():
				try:
					run_episode(case, scorer, t_dir, policy=None, sensing="realistic", video=video)
				finally:
					_write_task_json_if_missing(t_dir, case)
		teacher_by_seed, teacher_summary = load_teacher_baseline(teacher_root, cases=cases)

	summary = summarize_closed_loop(
		student_records,
		grades=student_grades,
		teacher_by_seed=teacher_by_seed,
		teacher_summary=teacher_summary,
		n_boot=n_boot,
	)
	# One row per layout: the case, the student's and the teacher's outcome (plan B paired table).
	paired_rows = []
	for r in student_records:
		t_row = teacher_by_seed.get(int(r["seed"])) if teacher_by_seed else None
		paired_rows.append(
			{
				"case": r["case"],
				"student_verdict": r.get("verdict"),
				"student_label": r.get("label"),
				"teacher_verdict": None if t_row is None else t_row.get("verdict"),
				"teacher_label": None if t_row is None else t_row.get("label"),
				"student_dir": r.get("episode_dir"),
				"teacher_dir": None if t_row is None else t_row.get("episode_dir"),
			}
		)
	report = {
		"mode_requested": mode,
		"mode_resolved": resolved_mode,
		"server_health": health,
		"summary": summary,
		"paired_rows": paired_rows,
		"episodes": student_records,
		"grades": student_grades,
	}
	(out_dir / "grades.json").write_text(
		json.dumps({"grades": student_grades, "summary": summary["s6_checks"]["grader_summary"]}, indent=2) + "\n",
		encoding="utf-8",
	)
	(out_dir / "eval_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
	return report


def build_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	src = p.add_mutually_exclusive_group(required=True)
	src.add_argument("--ckpt", type=Path, help="Trained checkpoint directory (auto-spawns vla/server.py)")
	src.add_argument("--server-url", type=str, help="Running VLA policy server URL (e.g. http://127.0.0.1:8000)")
	p.add_argument("--suite", type=Path, default=DEFAULT_SUITE, help="Case suite JSON (default: config/vla/val.json)")
	p.add_argument("--out", type=Path, required=True, help="Output directory for student traces and eval_summary.json")
	p.add_argument("--detector", type=Path, default=DEFAULT_DETECTOR)
	p.add_argument("--mode", choices=("auto", "sync", "async"), default="auto")
	p.add_argument("--replan-interval", type=int, default=None)
	p.add_argument("--prefetch-ahead", type=int, default=None)
	p.add_argument("--end-threshold", type=float, default=None)
	p.add_argument(
		"--temporal-ensemble",
		type=float,
		default=None,
		help="ACT temporal ensembling coefficient m (e.g. 0.01); pair with a small --replan-interval",
	)
	p.add_argument(
		"--stop-policy",
		choices=("latest", "keep", "earliest"),
		default="latest",
		help="How a replan combines its end-step with a planned stop (see ChunkClientPolicy)",
	)
	p.add_argument("--teacher-runs", type=Path, default=None, help="Optional dir of expert runs on the same suite")
	p.add_argument("--no-teacher", action="store_true", help="Do not run teacher baseline if --teacher-runs is absent")
	p.add_argument("--max-episodes", type=int, default=None)
	p.add_argument("--grader-ref", type=Path, default=None)
	p.add_argument("--video", action="store_true", help="Record video during student evaluation")
	p.add_argument("--allow-sealed", action="store_true", help="Allow evaluation on sealed test.json (plan R5)")
	p.add_argument("--lerobot-python", type=Path, default=LEROBOT_CUDA_PYTHON)
	p.add_argument("--device", default="auto", help="Server device when --ckpt is given (auto|cpu|cuda)")
	p.add_argument("--allow-download", action="store_true")
	p.add_argument("--timeout-s", type=float, default=30.0)
	p.add_argument("--n-boot", type=int, default=10_000)
	return p


def main(argv: list[str] | None = None) -> int:
	args = build_parser().parse_args(argv)
	cases = assert_suite_allowed(args.suite, allow_sealed=args.allow_sealed)
	if args.max_episodes is not None:
		if args.max_episodes < 0:
			raise ValueError(f"--max-episodes must be >= 0, got {args.max_episodes}")
		cases = cases[: args.max_episodes]
	if not cases:
		raise ValueError("suite contains no episodes to evaluate")

	from digital_twin_armigo.inspection_scorer import default_sup_scorer

	scorer = default_sup_scorer(args.detector)
	try:
		with (
			managed_policy_server(
				ckpt=args.ckpt,
				server_url=args.server_url,
				lerobot_python=args.lerobot_python,
				device=args.device,
				allow_download=args.allow_download,
			) as url,
			RemoteChunkPredictor(url, timeout_s=args.timeout_s) as predictor,
		):
			report = evaluate_closed_loop(
				predictor=predictor,
				scorer=scorer,
				cases=cases,
				out_dir=args.out,
				mode=args.mode,
				replan_interval=args.replan_interval,
				prefetch_ahead=args.prefetch_ahead,
				end_threshold=args.end_threshold,
				temporal_ensemble=args.temporal_ensemble,
				stop_policy=args.stop_policy,
				teacher_runs=args.teacher_runs,
				run_teacher=not args.no_teacher,
				grader_ref=args.grader_ref,
				video=args.video,
				n_boot=args.n_boot,
			)
	finally:
		if hasattr(scorer, "close"):
			scorer.close()

	s = report["summary"]
	pg = s["pilot_gate"]
	print(
		f"Closed-loop eval ({report['mode_resolved']}): {s['sort_successes']}/{s['episodes']} sorted "
		f"({s['student_sort_rate']:.3f}), missed_defects={s['missed_defects']}/{s['defective_episodes']}, "
		f"collisions={s['collisions']['total_collisions']} ({s['collisions']['collision_episodes']} eps), "
		f"teacher={s['teacher_sort_rate']}, pilot_gate={pg['pilot_gate_pass']}"
	)
	if s["paired_teacher_student"] is not None:
		p = s["paired_teacher_student"]
		print(
			f"Paired teacher-student diff (n={p['n']}): {p['mean_diff']:+.4f} "
			f"95% CI [{p['ci_95'][0]:+.4f}, {p['ci_95'][1]:+.4f}]"
		)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
