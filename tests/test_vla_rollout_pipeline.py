"""Tests for scripts/vla/calibrate_grader.py and scripts/vla/rollout_and_export.py."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "vla"))

import calibrate_grader
import rollout_and_export

from digital_twin_armigo.agent import vla_grader


def _make_synthetic_episode(
	ep_dir: Path,
	*,
	cmd_scale: float,
	meas_scale: float,
	belief_err_mm: float = 2.0,
	verdict: str = "SUCCESS",
	with_mkv: bool = False,
) -> Path:
	"""Create a minimal synthetic episode directory valid for vla_grader."""
	ep_dir.mkdir(parents=True, exist_ok=True)
	n_frames = 20
	t = np.linspace(0.0, 1.0, n_frames)
	# Smooth cubic + controlled high-frequency oscillation for deterministic jerk
	osc = np.array([(-1.0) ** i for i in range(n_frames)], dtype=np.float64)
	action = np.zeros((n_frames, 6), dtype=np.float32)
	state = np.zeros((n_frames, 6), dtype=np.float32)
	for j in range(5):
		action[:, j] = (t**2 + cmd_scale * 0.01 * osc).astype(np.float32)
		state[:, j] = (t**2 + meas_scale * 0.01 * osc).astype(np.float32)
	phase = np.array(["approach"] * 10 + ["grasp"] * 10)
	contact_r = np.zeros(n_frames, dtype=np.float32)
	contact_j = np.zeros(n_frames, dtype=np.float32)

	branch_dir = ep_dir / "branch-0"
	branch_dir.mkdir(parents=True, exist_ok=True)
	np.savez(
		branch_dir / "actions.npz",
		action=action,
		state=state,
		phase=phase,
		contact_robot_n=contact_r,
		contact_jaw_env_n=contact_j,
	)
	if with_mkv:
		(branch_dir / "wrist_cam.mkv").write_bytes(b"fake-mkv-data")
		(branch_dir / "top_full.mkv").write_bytes(b"fake-mkv-data")

	manifest = {
		"provenance": {"sensing": "realistic"},
		"branches": [{"id": 0, "actions": n_frames, "start_frame": 0}],
		"result": {
			"verdict": verdict,
			"student_eligible": verdict == "SUCCESS",
			"label": "normal",
			"evidence": ["v0"],
			"rewinds": 0,
		},
	}
	(ep_dir / "episode.json").write_text(json.dumps(manifest), encoding="utf-8")
	(ep_dir / "audit.private.json").write_text(
		json.dumps({"true_label": "normal", "true_face": None}), encoding="utf-8"
	)

	events = [
		{
			"seq": 0,
			"kind": "view",
			"observation": {
				"id": "v0",
				"camera": "wrist_cam",
				"score": 0.01,
				"threshold": 0.5,
				"faces": [0, 1, 2, 3, 4, 5],
				"face_pixels": [1000] * 6,
				"crop_bbox": [0, 0, 256, 256],
			},
		},
		{
			"seq": 1,
			"kind": "view_truth",
			"id": "v0",
			"belief_error_mm": belief_err_mm,
			"true_brick_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
			"belief_brick_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
			"true_face_pixels": [1000] * 6,
			"true_face_pixels_crop_native": [1000] * 6,
			"true_cross_px_crop_native": 0,
			"true_cross_px_crop": 0,
			"truth_crop": {"bbox": [0, 0, 256, 256], "size": 256},
		},
	]
	(ep_dir / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")
	return ep_dir


def test_batch_slicing_and_suite_loading(tmp_path: Path):
	suite_path = tmp_path / "suite.json"
	cases = [[1000 + i, bool(i % 2), i % 6] for i in range(45)]
	suite_path.write_text(
		json.dumps({"name": "test-suite", "options": {"seed": 1}, "cases": cases}),
		encoding="utf-8",
	)
	loaded_cases, meta = rollout_and_export.load_suite_cases(suite_path)
	assert len(loaded_cases) == 45
	assert meta["options"] == {"seed": 1}

	batches = rollout_and_export.slice_batches(loaded_cases, batch_size=20)
	assert [len(b) for b in batches] == [20, 20, 5]

	capped = rollout_and_export.slice_batches(loaded_cases, batch_size=10, max_episodes=25)
	assert [len(b) for b in capped] == [10, 10, 5]

	with pytest.raises(ValueError):
		rollout_and_export.slice_batches(loaded_cases, batch_size=0)


def test_s4_and_s6_gate_checks():
	def _fake_grade(cls: str, *, verdict: str = "SUCCESS", eligible: bool = True, passed: bool = True):
		return {
			"episode": f"/tmp/{cls}",
			"class": cls,
			"true_label": "normal" if cls == "normal" else "defective",
			"true_face": None,
			"label": "normal",
			"verdict": verdict,
			"sensing": "realistic",
			"passed": passed,
			"passed_available": passed,
			"unavailable": [],
			"checks": {
				c: {"status": "pass" if (c != "eligible" or (verdict == "SUCCESS" and eligible)) else "fail"}
				for c in vla_grader.CHECKS
			},
			"alignment": [],
		}

	# 20 episodes, all expert-eligible and all classes >= 0.8x mean -> S4 PASS, S6 PASS
	grades_ok = [_fake_grade("normal") for _ in range(10)] + [_fake_grade("defect_face_0") for _ in range(10)]
	res_ok = rollout_and_export.evaluate_gates(grades_ok)
	assert res_ok["s4_gate_pass"] is True
	assert res_ok["s6_gate_pass"] is True
	assert res_ok["expert_success_rate"] == 1.0

	# 20 episodes, 2 fail expert eligibility -> 0.90 < 0.95 -> S4 FAIL
	grades_s4_fail = grades_ok[:18] + [
		_fake_grade("normal", verdict="FAIL", eligible=False, passed=False) for _ in range(2)
	]
	res_s4_fail = rollout_and_export.evaluate_gates(grades_s4_fail)
	assert res_s4_fail["s4_gate_pass"] is False
	assert res_s4_fail["expert_success_rate"] == 0.9

	# S6 fail: normal=1.0 (10/10), defect_face_5=0.5 (5/10), overall=0.75, 0.8*0.75=0.60 -> 0.50 < 0.60
	grades_s6_fail = [_fake_grade("normal") for _ in range(10)] + [
		_fake_grade("defect_face_5", passed=(i < 5)) for i in range(10)
	]
	res_s6_fail = rollout_and_export.evaluate_gates(grades_s6_fail)
	assert res_s6_fail["s4_gate_pass"] is True
	assert res_s6_fail["s6_gate_pass"] is False
	assert res_s6_fail["flagged_classes"] == ["defect_face_5"]


def test_calibrate_grader_joint_rejection(tmp_path: Path):
	cal_dir = tmp_path / "cal_runs"
	# 20 episodes: episode 0 has high cmd jerk, episode 1 has high meas jerk (anti-correlated outliers)
	# Marginal p95 rejects ~2/20 (10%), while calibrated joint multiplier rejects 1/20 (5%).
	for i in range(20):
		cmd_s = 5.0 if i == 0 else 1.0 + 0.02 * i
		meas_s = 5.0 if i == 1 else 1.0 + 0.02 * (19 - i)
		_make_synthetic_episode(
			cal_dir / f"case-{i:04d}",
			cmd_scale=cmd_s,
			meas_scale=meas_s,
			belief_err_mm=1.5 + 0.1 * i,
		)

	out_ref = tmp_path / "frozen_ref.json"
	frozen = calibrate_grader.calibrate_reference(
		episode_dirs=[cal_dir],
		out_path=out_ref,
		target_rejection_rate=0.05,
		base_reference_path=ROOT / "config/vla/grader_reference.json",
	)

	loaded = vla_grader.load_reference(out_ref)
	assert "thresholds" in loaded
	assert "smoothness" in loaded
	assert frozen["smoothness"]["actual_joint_rejection_rate"] == pytest.approx(0.05, abs=0.01)

	# Verify vla_grader.check_smoothness reproduces the 5% joint rejection rate
	episodes = vla_grader.find_episodes(cal_dir)
	rejected = 0
	for ep in episodes:
		manifest = json.loads((ep / "episode.json").read_text(encoding="utf-8"))
		chk = vla_grader.check_smoothness(ep, manifest, loaded)
		assert chk["status"] in ("pass", "fail")
		if chk["status"] == "fail":
			rejected += 1
	assert rejected == 1
	assert frozen["calibration"]["belief_error"]["overall"]["views"] == 20


def _fake_grade_from_ep(ep: Path, passed: bool = True) -> dict:
	return {
		"episode": str(ep),
		"class": "normal",
		"true_label": "normal",
		"true_face": None,
		"label": "normal",
		"verdict": "SUCCESS",
		"sensing": "realistic",
		"passed": passed,
		"passed_available": passed,
		"unavailable": [],
		"checks": {c: {"status": "pass" if passed else "fail"} for c in vla_grader.CHECKS},
		"alignment": [],
	}


def test_resume_and_mkv_deletion_safety(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
	suite_path = tmp_path / "suite.json"
	cases = [[2000 + i, False, 5] for i in range(4)]
	suite_path.write_text(json.dumps({"cases": cases}), encoding="utf-8")

	out_dir = tmp_path / "run_out"
	calls: list[list[str]] = []
	fail_verify_on_batch_1 = True

	def fake_subprocess_run(cmd, env=None, check=True):
		calls.append(cmd)
		# Rollout command
		if "-m" in cmd and "digital_twin_armigo.agent.twin_mcp" in cmd:
			raw_dir = Path(cmd[cmd.index("--out") + 1])
			# twin_mcp / EpisodeTrace requires raw_dir/case-XXXX to not exist beforehand
			assert not raw_dir.exists(), f"stale raw_dir was not cleaned before resume: {raw_dir}"
			batch_suite = json.loads(Path(cmd[cmd.index("--suite") + 1]).read_text(encoding="utf-8"))
			for i in range(len(batch_suite["cases"])):
				_make_synthetic_episode(
					raw_dir / f"case-{i:04d}",
					cmd_scale=1.0,
					meas_scale=1.0,
					with_mkv=True,
				)
			return subprocess.CompletedProcess(cmd, 0)
		# Export command
		if "-m" in cmd and "digital_twin_armigo.agent.twin_record" in cmd:
			shard_dir = Path(cmd[cmd.index("--out") + 1])
			(shard_dir / "meta").mkdir(parents=True, exist_ok=True)
			return subprocess.CompletedProcess(cmd, 0)
		# Verify command
		if "--verify-dataset" in cmd:
			if fail_verify_on_batch_1 and "batch-0001" in cmd[cmd.index("--verify-dataset") + 1]:
				raise subprocess.CalledProcessError(1, cmd)
			return subprocess.CompletedProcess(cmd, 0)
		# Merge command
		if "--merge-shards" in cmd:
			merged_dir = Path(cmd[cmd.index("--merge-shards") + 1])
			(merged_dir / "meta").mkdir(parents=True, exist_ok=True)
			return subprocess.CompletedProcess(cmd, 0)
		return subprocess.CompletedProcess(cmd, 0)

	monkeypatch.setattr(subprocess, "run", fake_subprocess_run)
	monkeypatch.setattr(rollout_and_export, "grade", lambda ep, reference=None: _fake_grade_from_ep(ep))

	# First run: batch-0000 succeeds (mkv deleted), batch-0001 verify FAILS (mkv MUST be preserved!)
	with pytest.raises(subprocess.CalledProcessError):
		rollout_and_export.run_pipeline(
			suite_path=suite_path,
			out_dir=out_dir,
			batch_size=2,
			keep_raw=False,
			resume=False,
			lerobot_python=Path(sys.executable),
		)

	assert (out_dir / "batches/batch-0000/done.json").exists()
	assert list((out_dir / "batches/batch-0000/raw").rglob("*.mkv")) == []
	assert not (out_dir / "batches/batch-0001/done.json").exists()
	# Safety check: raw .mkv files preserved when verify failed!
	assert len(list((out_dir / "batches/batch-0001/raw").rglob("*.mkv"))) == 4

	# Resume run: batch-0000 must be skipped, batch-0001 succeeds and deletes mkv
	calls.clear()
	fail_verify_on_batch_1 = False
	summary = rollout_and_export.run_pipeline(
		suite_path=suite_path,
		out_dir=out_dir,
		batch_size=2,
		keep_raw=False,
		resume=True,
		lerobot_python=Path(sys.executable),
	)

	assert summary["episodes"] == 4
	assert summary["exported_episodes"] == 4
	assert summary["s4_gate_pass"] is True
	assert (out_dir / "timing.json").exists()
	assert (out_dir / "summary.json").exists()
	# Batch 0 was NOT re-run on resume
	assert not any("batch-0000" in " ".join(c) and "twin_mcp" in " ".join(c) for c in calls)
	assert list((out_dir / "batches/batch-0001/raw").rglob("*.mkv")) == []


def test_keep_raw_preserves_mkv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
	suite_path = tmp_path / "suite.json"
	suite_path.write_text(json.dumps({"cases": [[3000, False, 5]]}), encoding="utf-8")
	out_dir = tmp_path / "keep_raw_out"

	def fake_subprocess_run(cmd, env=None, check=True):
		if "-m" in cmd and "digital_twin_armigo.agent.twin_mcp" in cmd:
			raw_dir = Path(cmd[cmd.index("--out") + 1])
			_make_synthetic_episode(raw_dir / "case-0000", cmd_scale=1.0, meas_scale=1.0, with_mkv=True)
		elif "-m" in cmd and "digital_twin_armigo.agent.twin_record" in cmd:
			shard_dir = Path(cmd[cmd.index("--out") + 1])
			(shard_dir / "meta").mkdir(parents=True, exist_ok=True)
		elif "--merge-shards" in cmd:
			merged_dir = Path(cmd[cmd.index("--merge-shards") + 1])
			(merged_dir / "meta").mkdir(parents=True, exist_ok=True)
		return subprocess.CompletedProcess(cmd, 0)

	monkeypatch.setattr(subprocess, "run", fake_subprocess_run)
	monkeypatch.setattr(rollout_and_export, "grade", lambda ep, reference=None: _fake_grade_from_ep(ep))

	rollout_and_export.run_pipeline(
		suite_path=suite_path,
		out_dir=out_dir,
		batch_size=1,
		keep_raw=True,
		lerobot_python=Path(sys.executable),
	)
	assert len(list((out_dir / "batches/batch-0000/raw").rglob("*.mkv"))) == 2


def test_split_contiguous_preserves_order_and_balances():
	items = list(range(7))
	chunks = rollout_and_export.split_contiguous(items, 3)
	assert chunks == [[0, 1, 2], [3, 4], [5, 6]]
	assert rollout_and_export.split_contiguous(items[:2], 4) == [[0], [1]]
	assert rollout_and_export.split_contiguous(items, 1) == [items]


def test_parallel_overlapped_export_shards_resume_and_merge_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
	import threading

	suite_path = tmp_path / "suite.json"
	suite_path.write_text(json.dumps({"cases": [[4000 + i, False, 5] for i in range(9)]}), encoding="utf-8")
	out_dir = tmp_path / "par_out"
	lock = threading.Lock()
	calls: list[list[str]] = []
	merged_shards: list[str] = []
	fail_verify_batch = "batch-0001"

	def fake_subprocess_run(cmd, env=None, check=True):
		with lock:
			calls.append(cmd)
		if "-m" in cmd and "digital_twin_armigo.agent.twin_mcp" in cmd:
			raw_dir = Path(cmd[cmd.index("--out") + 1])
			assert not raw_dir.exists()
			batch_suite = json.loads(Path(cmd[cmd.index("--suite") + 1]).read_text(encoding="utf-8"))
			for i in range(len(batch_suite["cases"])):
				_make_synthetic_episode(raw_dir / f"case-{i:04d}", cmd_scale=1.0, meas_scale=1.0, with_mkv=True)
		elif "-m" in cmd and "digital_twin_armigo.agent.twin_record" in cmd:
			shard_dir = Path(cmd[cmd.index("--out") + 1])
			(shard_dir / "meta").mkdir(parents=True, exist_ok=True)
			eps = cmd[cmd.index("--repo-id") + 2 :]
			(shard_dir / "eps.json").write_text(json.dumps(eps), encoding="utf-8")
		elif "--verify-dataset" in cmd:
			root = cmd[cmd.index("--verify-dataset") + 1]
			if fail_verify_batch and fail_verify_batch in root:
				raise subprocess.CalledProcessError(1, cmd)
		elif "--merge-shards" in cmd:
			merged_shards[:] = cmd[cmd.index("--merge-shards") + 3 :]
			(Path(cmd[cmd.index("--merge-shards") + 1]) / "meta").mkdir(parents=True, exist_ok=True)
		return subprocess.CompletedProcess(cmd, 0)

	monkeypatch.setattr(subprocess, "run", fake_subprocess_run)
	monkeypatch.setattr(rollout_and_export, "grade", lambda ep, reference=None: _fake_grade_from_ep(ep))
	kwargs = {
		"suite_path": suite_path,
		"out_dir": out_dir,
		"batch_size": 4,
		"keep_raw": False,
		"lerobot_python": Path(sys.executable),
		"export_jobs": 3,
		"overlap_export": True,
	}

	# Batch 1's export fails in the background while batch 2 rolls out: batch 0 is recorded, batch 1 is not,
	# and batch 1's raw mkv survive.
	with pytest.raises(subprocess.CalledProcessError):
		rollout_and_export.run_pipeline(resume=False, **kwargs)
	b0 = json.loads((out_dir / "batches/batch-0000/done.json").read_text(encoding="utf-8"))
	assert b0["shard_dirs"] == ["lerobot-00", "lerobot-01", "lerobot-02"]
	assert [len(json.loads((out_dir / f"batches/batch-0000/{n}/eps.json").read_text())) for n in b0["shard_dirs"]] == [
		2,
		1,
		1,
	]
	assert not (out_dir / "batches/batch-0001/done.json").exists()
	assert len(list((out_dir / "batches/batch-0001/raw").rglob("*.mkv"))) == 8

	# Rewrite batch 0 as an old single-shard done.json (no shard_dirs key) to check backward-compatible resume.
	for n in b0["shard_dirs"]:
		shutil.rmtree(out_dir / "batches/batch-0000" / n)
	(out_dir / "batches/batch-0000/lerobot/meta").mkdir(parents=True)
	del b0["shard_dirs"]
	(out_dir / "batches/batch-0000/done.json").write_text(json.dumps(b0), encoding="utf-8")

	calls.clear()
	fail_verify_batch = None
	summary = rollout_and_export.run_pipeline(resume=True, **kwargs)
	assert summary["episodes"] == 9
	assert summary["exported_episodes"] == 9
	assert not any("batch-0000" in " ".join(c) and "twin_mcp" in " ".join(c) for c in calls)
	expected = [str(out_dir.resolve() / "batches/batch-0000/lerobot")]
	expected += [str(out_dir.resolve() / f"batches/batch-0001/lerobot-{k:02d}") for k in range(3)]
	expected += [str(out_dir.resolve() / "batches/batch-0002/lerobot")]  # 1 passing episode -> single shard
	assert merged_shards == expected
	# Manifest order matches shard concatenation order.
	manifest = json.loads((out_dir / "verify_manifest.json").read_text(encoding="utf-8"))["episodes"]
	shard_eps = [json.loads((Path(s) / "eps.json").read_text()) for s in expected[1:]]
	assert [m["path"] for m in manifest[4:]] == [e for chunk in shard_eps for e in chunk]
	assert list(out_dir.rglob("*.mkv")) == []
