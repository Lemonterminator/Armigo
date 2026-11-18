"""Tests for the main-venv client of the anomaly score server (no torch needed)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from digital_twin_armigo.inspection_scorer import FixedScorer, SubprocessAnomalyScorer

FAKE_SERVER = Path(__file__).resolve().with_name("fake_score_server.py")
IMAGE = np.full((16, 16, 3), 255, dtype=np.uint8)


@pytest.fixture
def scorer():
	with SubprocessAnomalyScorer(sys.executable, FAKE_SERVER, {"top": "60", "wrist_cam": "20"}, timeout_s=30.0) as s:
		yield s


def test_ready_handshake_and_scores(scorer):
	assert set(scorer.cameras) == {"top", "wrist_cam"}
	assert scorer.threshold("top") == 60.0
	assert scorer.threshold("wrist_cam") == 20.0
	assert scorer.score("top", IMAGE) == pytest.approx(100.0)
	assert scorer.score("wrist_cam", IMAGE) == pytest.approx(100.0)


def test_scores_depend_on_pixels(scorer):
	image = np.full((8, 12, 3), 51, dtype=np.uint8)  # mean 51 -> 20.0
	assert scorer.score("top", image) == pytest.approx(20.0, abs=1e-6)


def test_unknown_camera_error_does_not_kill_the_server(scorer):
	with pytest.raises(RuntimeError, match="unknown camera"):
		scorer.score("side", IMAGE)
	assert scorer.score("top", IMAGE) == pytest.approx(100.0)


def test_close_terminates_the_process(scorer):
	proc = scorer._proc
	assert proc is not None and proc.poll() is None
	scorer.close()
	assert proc.poll() is not None


def test_server_that_exits_immediately_raises(tmp_path):
	script = tmp_path / "crash.py"
	script.write_text("import sys\nsys.stderr.write('boom\\n')\nsys.exit(3)\n", encoding="utf-8")
	with pytest.raises(RuntimeError, match="boom"):
		SubprocessAnomalyScorer(sys.executable, script, {"top": "60"}, timeout_s=10.0)


def test_fixed_scorer():
	calls = []

	def score_fn(camera, image):
		calls.append((camera, image.shape))
		return 12.5

	scorer = FixedScorer(score_fn, {"top": 10.0})
	assert scorer.threshold("top") == 10.0
	assert scorer.score("top", IMAGE) == 12.5
	assert calls == [("top", IMAGE.shape)]
	with pytest.raises(KeyError):
		scorer.score("wrist_cam", IMAGE)


def test_score_rejects_non_uint8(scorer):
	with pytest.raises(TypeError):
		scorer.score("top", IMAGE.astype(np.float32))


def test_score_details_support_optional_heads(scorer):
	assert scorer.score_with_details("top", IMAGE) == (100.0, {})
	with SubprocessAnomalyScorer(
		sys.executable, FAKE_SERVER, {"top": "60"}, extra_args=("--head-scores",), timeout_s=30.0
	) as detailed:
		score, heads = detailed.score_with_details("top", IMAGE)
		assert score == 100.0
		assert heads == {"cls": 100.0, "seg": 98.75}
		assert detailed.score("top", IMAGE) == score
