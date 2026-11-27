"""Tests for student-cameras-only export in twin_record and collect_paired_carry."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from digital_twin_armigo.constants import FPS, TOP_FULL
from digital_twin_armigo.vla.contract import FORBIDDEN_CAMERA_NAMES, STUDENT_CAMERA_NAMES


def _write_minimal_episode(
	ep_dir: Path,
	actions: np.ndarray,
	*,
	segments: list[tuple[str, int, int, str]] | None = None,
) -> None:
	from digital_twin_armigo.agent.twin_graph import TASK
	from digital_twin_armigo.agent.twin_record import CAMERAS, Video
	from digital_twin_armigo.constants import CAMERA_SHAPES

	n = len(actions)
	b0 = ep_dir / "branch-0"
	b0.mkdir(parents=True, exist_ok=True)
	if segments is None:
		segments = [("grasp", 0, n, "done")]
	seg_ids = np.zeros(n, dtype=np.int32)
	events = []
	for sid, (kind, start, end, reason) in enumerate(segments):
		seg_ids[start:end] = sid
		events.append(
			{
				"kind": "segment",
				"frame": start,
				"id": sid,
				"segment_kind": kind,
				"instruction": f"Seg {sid}",
				"params": {},
			}
		)
		events.append(
			{
				"kind": "segment_end",
				"frame": end,
				"id": sid,
				"frames": end - start,
				"reason": reason,
			}
		)
	(ep_dir / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
	npz_data: dict[str, np.ndarray] = {
		"state": np.zeros((n, 6), dtype=np.float32),
		"action": actions.astype(np.float32),
		"state_f64": np.zeros((n, 6), dtype=np.float64),
		"action_f64": actions.astype(np.float64),
		"segment": seg_ids,
	}
	np.savez_compressed(b0 / "actions.npz", **npz_data)

	shapes = {
		"front": CAMERA_SHAPES["front"],
		"wrist": CAMERA_SHAPES["wrist"],
		"top_rgb": [TOP_FULL[1], TOP_FULL[0], 3],
		"top_depth": CAMERA_SHAPES["top_depth_v2"],
	}
	for cam in CAMERAS:
		v = Video(b0 / f"{cam}.mkv", shapes[cam], lossless=True)
		for _ in range(n):
			v.put(np.zeros(shapes[cam], dtype=np.uint8))
		v.close()
	(ep_dir / "episode.json").write_text(
		json.dumps(
			{
				"schema": "armigo-agent-episode/v1",
				"fps": FPS,
				"task": TASK,
				"branches": [{"id": 0, "start_frame": 0, "actions": n}],
				"video": True,
				"result": {"verdict": "SUCCESS", "student_eligible": True, "rewinds": 0},
			}
		),
		encoding="utf-8",
	)


def test_student_frames_cameras_flag(tmp_path: Path):
	from digital_twin_armigo.agent.twin_record import student_frames

	ep = tmp_path / "ep"
	_write_minimal_episode(ep, np.zeros((5, 6), dtype=np.float32))

	# Default: student_cameras_only=False -> yields all cameras (front, top_full, top_rgb_v2, top_depth_v2, wrist)
	frames_all = list(student_frames(ep, student_cameras_only=False))
	assert len(frames_all) == 5
	f0 = frames_all[0]
	assert "observation.images.front" in f0
	assert "observation.images.top_full" in f0
	for cam in STUDENT_CAMERA_NAMES:
		assert f"observation.images.{cam}" in f0

	# student_cameras_only=True -> skips FORBIDDEN_CAMERA_NAMES, yields ONLY STUDENT_CAMERA_NAMES
	frames_student = list(student_frames(ep, student_cameras_only=True))
	assert len(frames_student) == 5
	fs0 = frames_student[0]
	for forbidden in FORBIDDEN_CAMERA_NAMES:
		assert f"observation.images.{forbidden}" not in fs0
	for cam in STUDENT_CAMERA_NAMES:
		assert f"observation.images.{cam}" in fs0
		# Bit-for-bit check on non-image values compared to all
		assert np.array_equal(fs0[f"observation.images.{cam}"], f0[f"observation.images.{cam}"])
	assert np.array_equal(fs0["observation.state"], f0["observation.state"])
	assert np.array_equal(fs0["action"], f0["action"])


def _check_export_lerobot_cameras(root_dir: Path) -> dict:
	from digital_twin_armigo.agent.twin_record import export_lerobot
	from digital_twin_armigo.vla.dataset import SegmentChunkDataset

	ep = root_dir / "ep"
	if not (ep / "episode.json").exists():
		_write_minimal_episode(ep, np.zeros((10, 6), dtype=np.float32))

	out_true = root_dir / "dataset_student"
	export_lerobot([ep], out_true, "local/test_student", student_cameras_only=True)
	info_true = json.loads((out_true / "meta/info.json").read_text(encoding="utf-8"))
	features_true = info_true["features"]
	for forbidden in FORBIDDEN_CAMERA_NAMES:
		assert f"observation.images.{forbidden}" not in features_true
	for cam in STUDENT_CAMERA_NAMES:
		assert f"observation.images.{cam}" in features_true

	ds_true = SegmentChunkDataset(out_true, repo_id="local/test_student", horizon=4, video_backend="pyav")
	assert len(ds_true) > 0
	item_true = ds_true[0]
	assert isinstance(item_true, dict)

	out_false = root_dir / "dataset_all"
	export_lerobot([ep], out_false, "local/test_all", student_cameras_only=False)
	info_false = json.loads((out_false / "meta/info.json").read_text(encoding="utf-8"))
	features_false = info_false["features"]
	for cam in (*FORBIDDEN_CAMERA_NAMES, *STUDENT_CAMERA_NAMES):
		assert f"observation.images.{cam}" in features_false

	return {
		"student_only_features": sorted(k for k in features_true if k.startswith("observation.images.")),
		"legacy_features": sorted(k for k in features_false if k.startswith("observation.images.")),
		"dataset_len": len(ds_true),
	}


def test_export_lerobot_cameras_only(tmp_path: Path):
	pytest.importorskip("lerobot")
	_check_export_lerobot_cameras(tmp_path)


def test_twin_record_main_args(monkeypatch, tmp_path: Path):
	from digital_twin_armigo.agent import twin_record

	called_with = {}

	def fake_export(episodes, out_path, repo_id, trim_unsupervised_prefix=False, student_cameras_only=False):
		called_with["episodes"] = list(episodes)
		called_with["out_path"] = out_path
		called_with["repo_id"] = repo_id
		called_with["trim_unsupervised_prefix"] = trim_unsupervised_prefix
		called_with["student_cameras_only"] = student_cameras_only

	monkeypatch.setattr(twin_record, "export_lerobot", fake_export)
	ep = tmp_path / "ep"
	out = tmp_path / "ds"

	monkeypatch.setattr("sys.argv", ["twin_record", "--out", str(out), "--student-cameras-only", str(ep)])
	twin_record.main()
	assert called_with["student_cameras_only"] is True

	monkeypatch.setattr("sys.argv", ["twin_record", "--out", str(out), str(ep)])
	twin_record.main()
	assert called_with["student_cameras_only"] is False


def test_collect_paired_carry_args(monkeypatch, tmp_path: Path):
	import importlib.util
	import types

	script_path = Path(__file__).resolve().parents[1] / "scripts" / "vla" / "collect_paired_carry.py"
	spec = importlib.util.spec_from_file_location("_collect_paired_carry_test", script_path)
	mod = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(mod)

	ep = tmp_path / "ep"
	_write_minimal_episode(ep, np.zeros((4, 6), dtype=np.float32))
	recorded_cmds: list[list[str]] = []

	def fake_run(cmd, **kwargs):
		if isinstance(cmd, list) and len(cmd) >= 3 and cmd[2] == "digital_twin_armigo.agent.twin_record":
			recorded_cmds.append(cmd)
			Path(cmd[cmd.index("--out") + 1]).mkdir(parents=True, exist_ok=True)
			return types.SimpleNamespace(returncode=0, stdout="")
		return types.SimpleNamespace(returncode=0, stdout="[1, 4]\n")

	monkeypatch.setattr(mod.subprocess, "run", fake_run)

	mod._export_shard(tmp_path / "shard_true", [ep], Path("python"), student_cameras_only=True)
	assert "--student-cameras-only" in recorded_cmds[-1]

	mod._export_shard(tmp_path / "shard_false", [ep], Path("python"), student_cameras_only=False)
	assert "--student-cameras-only" not in recorded_cmds[-1]

	captured: list[bool] = []
	monkeypatch.setattr("digital_twin_armigo.vla.dagger.assert_train_layouts_only", lambda *a, **kw: None)
	monkeypatch.setattr(mod, "collect", lambda *a, **kw: None)

	def fake_export_pending(out, lerobot_python, jobs=1, student_cameras_only=True):
		captured.append(student_cameras_only)
		return []

	monkeypatch.setattr(mod, "export_pending", fake_export_pending)
	suite = tmp_path / "suite.json"
	suite.write_text("[]", encoding="utf-8")
	mod.main(["--suite", str(suite), "--out", str(tmp_path / "out_def"), "--export-only"])
	mod.main(["--suite", str(suite), "--out", str(tmp_path / "out_no"), "--export-only", "--no-student-cameras-only"])
	assert captured == [True, False]
