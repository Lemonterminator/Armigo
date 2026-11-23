"""First-grasp tail experiment (agent/grasp_tail.py, docs/vla/GRASP_TAIL_AB_HANDOFF_2026-11-23.md).

The ACT policy is stood in for by a replay of the expert's own recorded episode, which descends, crosses the switch
line in the near field and closes like a good student. Episodes stop at the first segment after the first grasp.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from digital_twin_armigo.agent import grasp_tail as gt
from digital_twin_armigo.agent.grasp_tail import GraspPlan, GraspTail

CASE = (600001, True, 0)


# -- the switch detector on synthetic observations --------------------------------------------------------------------


def _plan(target=(0.2, 0.0, 0.02), roll=0.3, valid=True):
	t = np.array(target, dtype=np.float64) if valid else np.full(3, np.nan)
	return GraspPlan(t, roll, 0.85, 0, 0)


def _feed(tail, zs, *, xy=(0.2, 0.0), roll=0.3, grip=None):
	"""Observations at heights ``zs``; executed gripper commands ``grip`` (default: open)."""
	fired = []
	for i, z in enumerate(zs):
		q = np.array([0, 0, 0, 0, roll, 0.85])
		obs = {"tcp": np.array([xy[0], xy[1], z]), "jaw_normal": np.zeros(2), "gripper_cmd": 0.85, "effort_frac": 0.0}
		executed = None if i == 0 else np.array([0, 0, 0, 0, roll, 0.85 if grip is None else grip[i - 1]])
		fired.append(
			tail.observe(
				frame=i,
				segment_frame=i,
				obs=obs,
				q=q,
				executed=executed,
				clamp_run=0,
				gripper_state=None,
				clamp_frames=4,
			)
		)
		tail.command("act", np.zeros(6))
	return fired


LINE = 0.02 + 0.02 + gt.CROSS_MARGIN_M  # target z + pregrasp_dz + margin


def test_switch_fires_once_at_the_first_downward_crossing():
	tail = GraspTail("act", _plan(), 0.02, 0)
	fired = _feed(tail, [0.10, 0.07, LINE + 1e-4, LINE - 1e-4, 0.03, 0.02])
	assert fired == [False, False, False, True, False, False]
	assert tail.decision == "triggered" and tail.switch["segment_frame"] == 3


@pytest.mark.parametrize(
	("kwargs", "zs", "reason"),
	[
		({"xy": (0.2 + gt.NEAR_XY_M + 1e-3, 0.0)}, [0.1, 0.03], "near_field_xy"),
		({"roll": 0.3 + gt.NEAR_ROLL_RAD + 0.01}, [0.1, 0.03], "near_field_roll"),
		({"xy": (0.25, 0.0), "roll": 1.0}, [0.1, 0.03], "near_field_xy_roll"),
		({}, [0.03, 0.02], "never_above"),
		({}, [0.10, 0.08, 0.07], "never_crossed"),
		({"grip": [0.85, 0.70, 0.60]}, [0.10, 0.08, 0.07, 0.03], "closed_early"),
	],
)
def test_switch_reasons(kwargs, zs, reason):
	tail = GraspTail("expert-once", _plan(roll=0.3), 0.02, 0)
	fired = _feed(tail, zs, **kwargs)
	tail.finish(len(zs), len(zs), "done")
	assert not any(fired) and tail.decision == reason


def test_missing_target_never_switches():
	tail = GraspTail("expert-once", _plan(valid=False), 0.02, 0)
	assert not any(_feed(tail, [0.1, 0.03])) and tail.decision == "no_target"


def test_closing_latch_uses_the_largest_executed_command():
	tail = GraspTail("act", _plan(), 0.02, 0)
	# 0.85 -> 0.78 is 0.07 rad (not closing); 0.76 is 0.09 below the maximum: closing, before the crossing.
	_feed(tail, [0.10, 0.09, 0.08, 0.03], grip=[0.85, 0.78, 0.76])
	assert tail.decision == "closed_early"


# -- episodes ---------------------------------------------------------------------------------------------------------


class ConstantScorer:
	def threshold(self, camera):
		return 0.0

	def score(self, camera, image):
		return 1.0


class _Stop(Exception):
	pass


class FirstGraspPolicy:
	"""Replays the expert recording until the first grasp segment is over, then stops the episode. ``done_when``:
	end the first grasp early, at the first frame whose recorded command satisfies it."""

	def __init__(self, replay, done_when=None):
		self.replay, self.done_when = replay, done_when
		self.grasps = 0
		self.in_grasp = False
		self.act_calls_after_switch = 0

	def start(self, task, segment_id):
		if self.grasps:
			raise _Stop
		self.in_grasp = task["kind"] == "grasp"
		self.grasps += self.in_grasp
		self.replay.start(task, segment_id)

	def act(self, obs):
		step = self.replay.act(obs)
		if self.in_grasp and self.done_when is not None and step.target is not None and self.done_when(step.target):
			from digital_twin_armigo.agent.student_rollout import PolicyStep

			return PolicyStep(done=True)
		return step


@pytest.fixture(scope="module")
def expert_trace(tmp_path_factory):
	pytest.importorskip("mujoco")
	pytest.importorskip("langgraph")
	pytest.importorskip("cv2")
	from digital_twin_armigo.agent.student_rollout import run_episode

	out = tmp_path_factory.mktemp("grasp-tail") / "expert"
	run_episode(CASE, ConstantScorer(), out)
	return out


def _run(expert_trace, tmp_path, mode, done_when=None, name=None):
	"""Run the student up to the first segment after the first grasp; returns (twin, policy, actions array)."""
	from digital_twin_armigo.agent.student_rollout import ReplayPolicy, StudentTwin
	from digital_twin_armigo.agent.twin_graph import Policy, build_graph
	from digital_twin_armigo.agent.twin_mujoco import sample_case
	from digital_twin_armigo.agent.twin_record import EpisodeTrace

	trace = EpisodeTrace(tmp_path / (name or str(mode)), video=False)
	policy = FirstGraspPolicy(ReplayPolicy.from_trace(expert_trace), done_when)
	twin = StudentTwin(sample_case(*CASE), ConstantScorer(), trace, policy, sensing="realistic", grasp_tail=mode)
	with pytest.raises(_Stop):
		build_graph(twin, Policy()).invoke({}, config={"recursion_limit": 64})
	return twin, policy, np.array(twin.targets)


def _grasp_row(twin):
	return next(r for r in twin.segments if r["kind"] == "grasp")


def test_act_mode_logs_without_changing_a_single_command(expert_trace, tmp_path):
	plain, _, a0 = _run(expert_trace, tmp_path, None)
	logged, _, a1 = _run(expert_trace, tmp_path, "act")
	try:
		assert np.array_equal(a0, a1)  # bitwise: the logger draws no extra observation, RNG or filter update
		assert np.array_equal(plain.ep.telemetry()["qpos"], logged.ep.telemetry()["qpos"])
		log = logged.grasp_tail_log.to_json()
		assert log["decision"] == "triggered" and log["source_counts"] == {
			"act": _grasp_row(logged)["end"] - _grasp_row(logged)["start"]
		}
		row = _grasp_row(logged)
		assert len(log["rows"]["frame"]) == row["end"] - row["start"] + 1  # + the segment-end observation
		assert log["rows"]["frame"][0] == row["start"]
		done = [e for e in log["events"] if e["kind"] == "act_done"]
		assert len(done) == 1 and done[0]["wait_eligible"] is False  # the expert's segment ends clamped
		# The plan snapshot is the one the grasp segment's roll came from.

		assert row["params"]["roll_deg"] == round(math.degrees(log["plan"]["roll_rad"]), 1)
		json.dumps(log)  # serialisable
	finally:
		plain.close()
		logged.close()


def test_expert_once_takes_over_at_the_switch_and_hands_a_clamp_to_lift(expert_trace, tmp_path):
	a_twin, _, a = _run(expert_trace, tmp_path, "act")
	b_twin, _, b = _run(expert_trace, tmp_path, "expert-once")
	try:
		log = b_twin.grasp_tail_log.to_json()
		k = log["switch"]["segment_frame"]
		row = _grasp_row(b_twin)
		start = row["start"]
		assert log["decision"] == "triggered" and log["outcome"] == "clamped"
		# Identical prefix up to the switch observation, then only expert-tail commands until the segment ends.
		assert np.array_equal(a[: start + k], b[: start + k])
		src = log["rows"]["source"]
		assert set(src[:k]) == {"act"} and set(src[k:-1]) == {"expert_tail"} and src[-1] == ""
		assert row["reason"] == "done" and b_twin._clamped and b_twin.expert.aborted is None
		assert b_twin.segments[-1]["kind"] == "lift"  # ACT resumes with a new lift segment
		assert b_twin.expert.obs is not None and "tcp" not in (b_twin.expert.obs or {})  # restored
		# Every expert command lies inside the first grasp's frame window.
		assert len(src) - 1 == row["end"] - row["start"]
	finally:
		a_twin.close()
		b_twin.close()


def test_expert_once_without_a_switch_is_arm_a(expert_trace, tmp_path, monkeypatch):
	monkeypatch.setattr(gt, "NEAR_XY_M", -1.0)  # no crossing passes the near-field gate
	a_twin, _, a = _run(expert_trace, tmp_path, "act", name="a")
	b_twin, _, b = _run(expert_trace, tmp_path, "expert-once", name="b")
	try:
		assert b_twin.grasp_tail_log.decision == "near_field_xy"
		assert np.array_equal(a, b)
	finally:
		a_twin.close()
		b_twin.close()


def _closed(target):
	return target[5] <= 0.06 + 1e-9  # the expert's close ramp has reached hold_g


def _closing(target):
	return target[5] <= 0.85 - 0.15  # the ramp has just started: the hold keeps the jaws open


def test_act_wait_confirms_a_clamp_that_act_ended_too_early(expert_trace, tmp_path):
	a_twin, _, _ = _run(expert_trace, tmp_path, "act", done_when=_closed, name="a")
	w_twin, _, _ = _run(expert_trace, tmp_path, "act-wait", done_when=_closed, name="w")
	try:
		# Arm A ended before the clamp was confirmed: no lift (the retry path clears ``aborted`` before we look).
		assert not a_twin._clamped and a_twin.segments[-1]["kind"] != "lift"
		log = w_twin.grasp_tail_log.to_json()
		assert log["outcome"] == "clamped" and 0 < log["wait_frames"] <= 45
		# Arm A logs where A+wait starts waiting: the RGB-pin hand-off of A+wait (vla/rgb_pin.py).
		from digital_twin_armigo.vla.rgb_pin import handoff_frame

		a_log = a_twin.grasp_tail_log.to_json()
		wait_start = next(e["frame"] for e in log["events"] if e["kind"] == "wait_start")
		assert handoff_frame("act-wait", a_log) == (wait_start, "act_done_unclamped")
		assert handoff_frame("expert-once", a_log) == (a_log["switch"]["frame"], "switch")
		assert log["source_counts"]["act_wait_hold"] == log["wait_frames"]
		holds = [c for c, s in zip(log["rows"]["command"], log["rows"]["source"], strict=True) if s == "act_wait_hold"]
		assert all(h == holds[0] for h in holds)  # the last executed ACT command, unchanged
		assert w_twin._clamped and w_twin.segments[-1]["kind"] == "lift"
	finally:
		a_twin.close()
		w_twin.close()


def test_act_wait_times_out_into_the_original_failure(expert_trace, tmp_path):
	w_twin, _, _ = _run(expert_trace, tmp_path, "act-wait", done_when=_closing)
	try:
		log = w_twin.grasp_tail_log.to_json()
		assert log["outcome"] == "timeout" and log["wait_frames"] == 45
		assert not w_twin._clamped and w_twin.segments[-1]["kind"] != "lift"
	finally:
		w_twin.close()


@pytest.mark.parametrize("outcome", ["blocked", "timeout"])
def test_expert_tail_failures_hand_back_to_the_failure_branch(expert_trace, tmp_path, monkeypatch, outcome):
	from digital_twin_armigo.agent import student_rollout

	def fake_tail(ex, plan):
		for _ in range(3):
			yield ex.last.copy(), "grasp"
		return outcome

	monkeypatch.setattr(student_rollout, "expert_tail", fake_tail)
	twin, _, _ = _run(expert_trace, tmp_path, "expert-once")
	try:
		log = twin.grasp_tail_log.to_json()
		assert log["outcome"] == outcome and log["source_counts"]["expert_tail"] == 3
		assert not twin._clamped and (twin._blocked or outcome == "timeout")
		assert not any(r["kind"] == "lift" for r in twin.segments)
		assert _grasp_row(twin)["reason"] == "done"
	finally:
		twin.close()


# -- no ground truth reaches the tail ----------------------------------------------------------------------------------


def test_tail_target_switch_and_commands_ignore_hidden_object_poses(tmp_path):
	"""Fixed joints and perception inputs; moving the true brick and the true bowls changes neither the plan
	snapshot, the observation fields the switch reads, nor the expert-tail commands."""
	pytest.importorskip("mujoco")
	pytest.importorskip("cv2")
	from digital_twin_armigo.agent.perception import BrickEstimate
	from digital_twin_armigo.agent.student_rollout import StudentTwin
	from digital_twin_armigo.agent.twin_mujoco import sample_case
	from digital_twin_armigo.agent.twin_record import EpisodeTrace
	from digital_twin_armigo.constants import DUPLO_2X4
	from digital_twin_armigo.mujoco_cell._mj import mujoco

	trace = EpisodeTrace(tmp_path / "leak", video=False)
	twin = StudentTwin(sample_case(*CASE), ConstantScorer(), trace, None, sensing="realistic", grasp_tail="expert-once")
	try:
		layout = twin.layout
		twin.brick_est = BrickEstimate(
			np.array([layout.brick_xy[0] + 0.003, layout.brick_xy[1] - 0.002, 0.0096]),
			math.radians(layout.brick_yaw_deg + 2.0),
			DUPLO_2X4,
			(0.0638, 0.0318),
			1200,
		)
		twin._perceived_size = tuple(DUPLO_2X4)

		def snapshot():
			target, roll, open_g = twin._planned_grasp()
			plan = GraspPlan(target, roll, open_g, 0, 0)
			ex = twin.expert
			saved_prev = twin._contact.prev
			obs = twin._observe()
			twin._contact.prev = saved_prev
			fields = {
				k: np.asarray(obs[k], dtype=np.float64).copy()
				for k in ("q", "tcp", "jaw_normal", "brick_vel", "brick_pos")
			}
			# Drive the expert tail for a few frames from a fixed state 25 mm above the target.
			start = plan.target + np.array([0.004, -0.003, 0.025])
			last0 = ex.last.copy()
			ex.last = ex._ik(start, np.array([*ex.p.ready_q, plan.open_g]), plan.roll)
			ex.last[5] = plan.open_g
			ex.obs = {**obs, "tcp": start}
			ex.gripper_state = gt_state_free()
			gen = gt.expert_tail(ex, plan)
			cmds = []
			for _ in range(25):
				cmd, _phase = next(gen)
				cmds.append(np.asarray(cmd, dtype=np.float64).copy())
				ex.last = cmd.copy()
			ex.last = last0
			tail = GraspTail("expert-once", plan, ex.p.pregrasp_dz_m, 0)
			q = np.asarray(obs["q"], dtype=np.float64).copy()
			q[4], q[5] = plan.roll, plan.open_g
			for i, dz in enumerate((0.06, 0.024)):
				tail.observe(
					frame=i,
					segment_frame=i,
					obs={**obs, "tcp": plan.target + np.array([0.004, 0.0, dz])},
					q=q,
					executed=None if i == 0 else q,
					clamp_run=0,
					gripper_state=None,
					clamp_frames=4,
				)
			return plan, fields, np.array(cmds), tail.decision, tail.switch

		m, d = twin.cell.model, twin.cell.data
		mujoco.mj_forward(m, d)  # after mj_step the derived site positions lag qpos by a substep
		before = snapshot()
		adr = twin.cell.brick_qadr
		d.qpos[adr : adr + 3] += (0.03, -0.02, 0.0)
		for body in ("green_bowl", "pink_bowl"):
			m.body_pos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)] += (0.05, 0.04, 0.0)
		mujoco.mj_forward(m, d)
		after = snapshot()
		assert np.array_equal(before[0].target, after[0].target) and before[0].roll == after[0].roll
		assert before[0].open_g == after[0].open_g
		for k in before[1]:
			assert np.array_equal(before[1][k], after[1][k]), k
		assert np.array_equal(before[2], after[2])
		assert before[3] == after[3] == "triggered" and before[4] == after[4]
	finally:
		trace.close()
		twin.close()


def gt_state_free():
	from digital_twin_armigo.mujoco_cell.gripper_state import GripperState

	return GripperState.FREE


# -- run manifest -----------------------------------------------------------------------------------------------------


def test_manifest_binds_an_output_directory(tmp_path):
	from pathlib import Path

	from digital_twin_armigo.vla import run_manifest

	root = Path(__file__).resolve().parents[1]
	m = run_manifest.build_manifest(
		root=root, args={"mode": "sync"}, inputs={"suite": None}, protocol="p", grasp_tail_mode="act"
	)
	run_manifest.bind_manifest(tmp_path, m)
	run_manifest.bind_manifest(tmp_path, m)  # resume with the same identity
	assert json.loads((tmp_path / "run_manifest.json").read_text())["resumed_at"]
	other = run_manifest.build_manifest(
		root=root, args={"mode": "sync"}, inputs={"suite": None}, protocol="p", grasp_tail_mode="expert-once"
	)
	with pytest.raises(ValueError, match="different run"):
		run_manifest.bind_manifest(tmp_path, other)
	fresh = tmp_path / "fresh"
	(fresh / "student" / "case-0000").mkdir(parents=True)
	(fresh / "student" / "case-0000" / "episode.json").write_text("{}")
	with pytest.raises(ValueError, match="no run_manifest"):
		run_manifest.bind_manifest(fresh, m)


# -- paired statistics ------------------------------------------------------------------------------------------------


def test_exact_mcnemar_matches_the_protocol_numbers():
	import sys
	from pathlib import Path

	sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "vla"))
	from grasp_tail_report import four_cell, mcnemar_exact

	assert mcnemar_exact(0, 0) == 1.0
	assert mcnemar_exact(0, 5) == pytest.approx(0.0625)
	assert mcnemar_exact(0, 6) == pytest.approx(0.03125)
	assert mcnemar_exact(1, 7) == pytest.approx(0.0703125)
	assert mcnemar_exact(2, 8) == pytest.approx(0.109375)
	assert mcnemar_exact(5, 15) == pytest.approx(0.041389, abs=1e-6)
	c = four_cell([True, True, False, False, False], [True, False, True, True, False])
	assert (c["both"], c["only_first"], c["only_second"], c["neither"], c["net_cases"]) == (1, 1, 2, 1, 1)
	assert c["ci95"][0] <= c["net"] <= c["ci95"][1]
