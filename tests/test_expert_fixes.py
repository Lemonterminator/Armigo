"""VLA distillation step S2: separate retry/rewind counters, held checkpoints, radians as the canonical unit."""

from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("cv2")
pytest.importorskip("av")

from digital_twin_armigo.agent.twin_graph import Policy
from digital_twin_armigo.agent.twin_mujoco import ROOT, TwinMujoco, sample_case
from digital_twin_armigo.agent.twin_record import EpisodeTrace
from digital_twin_armigo.joints import JOINT_RANGE_RAD, rad_to_unit, unit_to_rad

DETECTOR = ROOT / "results/agent-bench/sup-v7bot-local.json"
CASE = (600001, False, 5)  # bench16 case 0, a normal brick: a detector that clears every face sorts it correctly


class FakeScorer:
	def threshold(self, camera):
		return 0.0

	def score(self, camera, image):
		return -1.0

	def close(self):
		pass


def test_fault_at_frame_192_recovered_in_place_is_student_eligible(tmp_path):
	"""The audit's case: a motion fault at frame 192 (start of the first wrist scan), in-place recovery, correct sort.
	Before S2 it ended ``SUCCESS, rewinds=1, student_eligible=False``."""
	from digital_twin_armigo.agent.twin_mcp import TwinService

	if not DETECTOR.is_file():
		pytest.skip("detector spec not present")
	service = TwinService(tmp_path, DETECTOR, scorer=FakeScorer(), sensing="realistic")
	try:
		result = service.run(
			CASE, Policy(), tmp_path / "ep", video=False, fault=lambda node, branch, frame: frame == 192, hook=None
		)
	finally:
		service.close()
	assert [f["code"] for f in result["failures"]] == ["INJECTED_MOTION_FAILURE"]
	assert result["verdict"] == "SUCCESS" and result["label"] == "normal"
	assert (result["rewinds"], result["retries"], result["active_branch"]) == (0, 1, 0)
	assert result["student_eligible"] is True
	# The recovery stays on the one physical branch the exporter accepts.
	actions = np.load(tmp_path / "ep/branch-0/actions.npz")
	assert not (tmp_path / "ep/branch-1").exists()
	assert len(actions["action"]) == result["active_frames"] == result["sim_frames"]
	# Radians are canonical: the normalisation covers every recorded command and inverts exactly.
	for key in ("action", "state"):
		q = actions[key].astype(np.float64)
		assert np.allclose(unit_to_rad(rad_to_unit(q)), q, rtol=0, atol=1e-12)
		assert np.allclose(rad_to_unit(unit_to_rad(rad_to_unit(q))), rad_to_unit(q), rtol=0, atol=1e-12)
	unit = rad_to_unit(actions["action"])
	assert unit.min() >= -1 - 1e-6 and unit.max() <= 1 + 1e-6


def _snapshot(cell):
	d = cell.data
	return d.qpos.copy(), d.qvel.copy(), d.ctrl.copy(), d.act.copy()


@pytest.mark.parametrize("sensing", ["realistic", "truth"])
def test_held_checkpoint_restores_the_carry_identically(tmp_path, sensing):
	from digital_twin_armigo.agent.twin_resume import load_checkpoint

	trace = EpisodeTrace(tmp_path / "ep", video=False)
	backend = TwinMujoco(sample_case(*CASE), FakeScorer(), trace, sensing=sensing)
	try:
		assert np.array_equal(backend.cell.jnt_range, JOINT_RANGE_RAD)  # the normalisation range is the model's
		assert backend.prepare() is None
		views = []
		assert backend._bottom(views) is None and len(views) == 1  # grasped, lifted, bottom shown to the top camera
		assert backend.held is not None
		backend._save(views)
		ck = backend.checkpoints[-1]
		assert ck.held is not None and ck.frame == backend.ep.n_frames
		before = _snapshot(backend.cell)
		held = (backend.held[0].copy(), backend.held[1])
		grip = None if backend._grip_from_brick is None else backend._grip_from_brick.copy()
		contact = backend._contact.prev if backend.realistic else None
		q_lift = held[0]

		def carry_on():
			"""Carry back to the lift pose and hold it there, with the lost-brick watch on."""
			ex = backend.expert
			yield from ex._carry(ex._move(q_lift, ex._carry_secs(q_lift, ex.p.show_move_s), "show_move"))
			yield from ex._carry(ex._hold_servo(q_lift, 0.5, "show_hold"))

		def run():
			start = backend.ep.n_frames
			assert backend._execute(carry_on()) is None
			tel = backend.ep.telemetry()
			return (
				np.array(backend.targets[start:]),
				tel["qpos"][start:],
				tel["brick_pos"][start:],
				backend.cell.data.qvel.copy(),
				backend.expert.lost,
			)

		first = run()
		assert not first[4], "the brick must stay in the gripper"
		assert not np.array_equal(_snapshot(backend.cell)[0], before[0])

		assert backend.restore_checkpoint(ck) == views
		assert (backend.rewinds, backend.branch) == (1, 1)
		for a, b in zip(_snapshot(backend.cell), before, strict=True):
			assert np.array_equal(a, b)
		assert np.array_equal(backend.held[0], held[0]) and backend.held[1] == held[1]
		if grip is None:
			assert backend._grip_from_brick is None
		else:
			assert np.array_equal(backend._grip_from_brick, grip)
		if backend.realistic:
			assert backend._contact.prev == contact

		second = run()
		for a, b in zip(first, second, strict=True):
			assert np.array_equal(a, b)

		# The persisted checkpoint (JSON + npy) carries the same held state.
		loaded = load_checkpoint(trace.path, ck.id)
		assert np.array_equal(loaded.integration, ck.integration)
		raw, raw_loaded = asdict(ck), asdict(loaded)
		raw.pop("integration"), raw_loaded.pop("integration")
		assert raw_loaded == json.loads(json.dumps(raw))  # JSON turns tuples (crop boxes) into lists
		assert raw_loaded["held"] == raw["held"] and raw_loaded["agent"] == raw["agent"]
	finally:
		backend.close()
		trace.close()
