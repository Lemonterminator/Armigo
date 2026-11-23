"""Shared pre-intervention prefix (vla/rgb_pin.py, docs/vla/GRASP_TAIL_AB_HANDOFF_2026-11-23.md r3)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from digital_twin_armigo.agent.student_rollout import StudentObservation
from digital_twin_armigo.agent.twin_mujoco import INITIAL_Q
from digital_twin_armigo.agent.twin_record import Segment, policy_inputs
from digital_twin_armigo.vla import rgb_pin

H = 10


class FakePredictor:
	"""Records every request; the chunk does not matter here."""

	def __init__(self):
		self.payloads = []

	def predict(self, payload):
		self.payloads.append(payload)
		probs = np.zeros(H + 2)
		probs[H + 1] = 1.0  # no end inside the horizon
		return {"actions": np.zeros((H, 6)), "end_step_probs": probs}


def _images(frame, noise):
	"""Deterministic scene per frame; ``noise`` flips one RGB level (render noise), depth never changes."""
	rng = np.random.default_rng(frame)
	top = rng.integers(0, 255, (416, 512, 3), dtype=np.uint8)
	wrist = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
	if noise:
		top[noise % 416, 7, 1] += 1
		wrist[3, noise % 640, 0] += 1
	depth = np.full((416, 512, 3), frame % 251, dtype=np.uint8)
	return {"top_rgb_v2": top, "top_depth": depth, "wrist": wrist}


def _episode(policy, frames, noise=0, q_shift=None):
	"""One prepare segment of ``frames`` frames; sync, replan every 2 frames -> requests at even frames."""
	task = policy_inputs(Segment("prepare"))
	policy.start(task, 0)
	for f in range(frames):
		q = np.full(6, 0.01 * f)
		if q_shift is not None and f == q_shift:
			q = q + 1e-6
		imgs = _images(f, noise and noise + f)
		policy.act(
			StudentObservation(
				frame=f,
				task=dict(task),
				segment_id=0,
				segment_frame=f,
				q=q,
				_images=lambda imgs=imgs: imgs,
				q_prev=q,
				prev_action=INITIAL_Q.copy(),
				gripper_cmd=0.85,
			)
		)


def _first_grasp(switch=4, done=5, eligible=True, decision="triggered"):
	return {
		"decision": decision,
		"switch": {"frame": switch} if decision == "triggered" else None,
		"events": [{"kind": "act_done", "frame": done, "wait_eligible": eligible}],
	}


def _record(tmp_path, frames=6, first_grasp=None):
	pred = FakePredictor()
	pol = rgb_pin.RgbPinClientPolicy(pred, recorder=rgb_pin.RgbRecorder(), mode="sync", replan_interval=2)
	_episode(pol, frames)
	ep = tmp_path / "a"
	ep.mkdir()
	pol.save(ep)
	(ep / "grasp_tail.json").write_text(json.dumps({"first_grasp": first_grasp or _first_grasp()}))
	return ep, pred


def _pin(src, mode, frames=6, **kw):
	pred = FakePredictor()
	pin = rgb_pin.RgbPin(src, mode)
	pol = rgb_pin.RgbPinClientPolicy(pred, pin=pin, mode="sync", replan_interval=2)
	_episode(pol, frames, **kw)
	return pin, pred


def test_pin_replaces_only_rgb_until_the_handoff(tmp_path):
	src, a = _record(tmp_path)
	assert [p["segment_frame"] for p in a.payloads] == [0, 2, 4]
	assert json.loads((src / rgb_pin.LOG_FILE).read_text())["requests"] == 3
	pin, b = _pin(src, "expert-once", noise=1)  # B hands off at A's switch frame 4
	for i in (0, 1):  # before the switch: A's RGB, own everything else
		for k in rgb_pin.PIN_RGB_KEYS:
			assert np.array_equal(b.payloads[i][k], a.payloads[i][k])
		assert np.array_equal(b.payloads[i]["top_depth_v2"], a.payloads[i]["top_depth_v2"])
		assert rgb_pin.meta_digest(b.payloads[i]) == rgb_pin.meta_digest(a.payloads[i])
	own = _images(4, 1 + 4)
	assert np.array_equal(b.payloads[2]["wrist"], own["wrist"])  # live from the switch on
	assert not np.array_equal(b.payloads[2]["wrist"], a.payloads[2]["wrist"])
	log = pin.log()
	assert log["pinned"] == 2 and log["rgb_diff_requests"] == 2 and not log["mismatch"]
	assert log["live_from"]["reason"] == "handoff" and log["live_from"]["frame"] == 4


def test_untriggered_arm_pins_the_whole_episode(tmp_path):
	src, a = _record(tmp_path, first_grasp=_first_grasp(decision="near_field_xy"))
	pin, b = _pin(src, "expert-once", noise=1)
	assert pin.handoff is None and pin.log()["pinned"] == 3 and pin.live_from is None
	for pa, pb in zip(a.payloads, b.payloads, strict=True):
		assert all(np.array_equal(pa[k], pb[k]) for k in rgb_pin.PIN_RGB_KEYS)
	pin, _ = _pin(src, "act", frames=8)  # more requests than A recorded
	assert pin.live_from["reason"] == "source_exhausted" and pin.live_from["request"] == 3


def test_a_request_that_differs_beyond_rgb_ends_the_pin(tmp_path):
	src, _ = _record(tmp_path)
	pin, _ = _pin(src, "act", q_shift=2)
	log = pin.log()
	assert log["mismatch"] and log["live_from"]["reason"] == "mismatch_request"
	assert log["live_from"]["request"] == 1 and log["pinned"] == 1
	assert log["live_from"]["q_t_max_abs"] > 0


@pytest.mark.parametrize(
	("mode", "first_grasp", "frame", "reason"),
	[
		("act", _first_grasp(), 4, "switch"),
		("expert-once", _first_grasp(), 4, "switch"),
		("act-wait", _first_grasp(), 5, "act_done_unclamped"),
		("act-wait", _first_grasp(eligible=False), None, "clamped_at_done"),
		("act-wait", {**_first_grasp(), "events": []}, None, "no_act_done"),
		("expert-once", _first_grasp(decision="closed_early"), None, "not_triggered:closed_early"),
		("act", None, None, "no_first_grasp"),
	],
)
def test_handoff_frame(mode, first_grasp, frame, reason):
	assert rgb_pin.handoff_frame(mode, first_grasp) == (frame, reason)


def test_act_wait_pins_through_the_unclamped_done(tmp_path):
	src, _ = _record(tmp_path)
	pin, _ = _pin(src, "act-wait", frames=6)  # A's done at frame 5: all three requests are before it
	assert pin.cut == 3 and pin.log()["pinned"] == 3 and pin.live_from is None


def test_pin_policy_refuses_async_and_needs_one_role():
	with pytest.raises(ValueError):
		rgb_pin.RgbPinClientPolicy(FakePredictor(), recorder=rgb_pin.RgbRecorder(), mode="async")
	with pytest.raises(ValueError):
		rgb_pin.RgbPinClientPolicy(FakePredictor(), mode="sync")


def test_source_manifest_must_match_but_for_the_arm():
	base = {
		"protocol": "r3",
		"code": {"commit": "x"},
		"inputs_sha256": {"ckpt": "c", "pin_source": None},
		"args": {"record_act_rgb": True, "pin_rgb_from": None, "grasp_tail_mode": "act", "suite": "val"},
	}
	own = {
		**base,
		"inputs_sha256": {"ckpt": "c", "pin_source": "id"},
		"args": {"record_act_rgb": False, "pin_rgb_from": "A", "grasp_tail_mode": "expert-once", "suite": "val"},
	}
	rgb_pin.check_source_manifest(base, own)
	for bad in (
		{**base, "args": {**base["args"], "record_act_rgb": False}},
		{**base, "code": {"commit": "y"}},
		{**base, "inputs_sha256": {"ckpt": "d"}},
		{**base, "args": {**base["args"], "suite": "train"}},
	):
		with pytest.raises(ValueError):
			rgb_pin.check_source_manifest(bad, own)
