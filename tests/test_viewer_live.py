"""Live 3D streaming of a real MuJoCo agent episode into Rerun (ARMIGO_RERUN=1)."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("cv2")
pytest.importorskip("av")
pytest.importorskip("rerun")

from digital_twin_armigo.agent.twin_mujoco import TwinMujoco, sample_case
from digital_twin_armigo.agent.twin_record import EpisodeTrace
from digital_twin_armigo.viewer import live

CASE = (600001, False, 5)


class FakeScorer:
	def threshold(self, camera):
		return 0.0

	def score(self, camera, image):
		return -1.0

	def close(self):
		pass


def test_live_stream_is_off_without_the_environment_switch(tmp_path, monkeypatch):
	monkeypatch.delenv(live.ENV, raising=False)
	trace = EpisodeTrace(tmp_path / "ep", video=False)
	backend = TwinMujoco(sample_case(*CASE), FakeScorer(), trace, sensing="truth")
	try:
		assert trace.live is None
	finally:
		backend.close()
	assert not (tmp_path / "ep" / "rerun.json").exists()


def test_real_episode_streams_poses_and_saves_a_replay(tmp_path, monkeypatch):
	monkeypatch.setenv(live.ENV, "1")
	trace = EpisodeTrace(tmp_path / "ep", video=False)
	backend = TwinMujoco(sample_case(*CASE), FakeScorer(), trace, sensing="truth")
	try:
		assert trace.live is not None
		info = json.loads((tmp_path / "ep" / "rerun.json").read_text(encoding="utf-8"))
		assert info["pid"] > 0 and info["url"].startswith("http://127.0.0.1:") and not info["finished"]
		assert live.live_info(tmp_path / "ep")["url"] == info["url"]
		assert backend.prepare() is None  # drives the arm to the ready pose: many control frames
		n = backend.ep.n_frames
		assert n > 0 and len(trace.live.frames) == n
		scene_nodes = len(trace.live.node_ids)
		moved = np.stack(trace.live.frames)
		assert moved.shape == (n, scene_nodes, 7)
		assert np.ptp(moved[:, :, :3], axis=0).max() > 0.01  # something (the arm) actually moved
	finally:
		trace.close()
		backend.close()
	replay = tmp_path / "ep" / "replay"
	assert (replay / "scene.json").is_file() and (replay / "motion.npz").is_file()
	assert live.live_info(tmp_path / "ep") is None  # finished: the viewer entry is released
	from digital_twin_armigo.catalog.playback import load_motion

	scene, poses, times = load_motion({"scene": replay / "scene.json", "motion": replay / "motion.npz"})
	assert poses.shape == (n, scene_nodes, 7) and len(scene["nodes"]) == scene_nodes and times[0] > 0
	# The saved replay and the events render into one .rrd for later viewing (3D + state machine in one file).
	from digital_twin_armigo.viewer import cli

	out = tmp_path / "ep.rrd"
	assert cli.main(["rrd", str(tmp_path / "ep"), "--out", str(out)]) == 0
	import rerun as rr

	rr.disconnect()
	text = subprocess.run(
		[sys.executable, "-m", "rerun", "rrd", "print", str(out)],
		capture_output=True,
		text=True,
		encoding="utf-8",
		errors="replace",
		check=True,
		timeout=180,
	).stdout
	assert "/world/geom_" in text and "/state/log" in text
