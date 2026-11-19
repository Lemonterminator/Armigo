"""Tests for S11 Takeover HG-DAgger: train-only guard, realistic-only gates, no-backfill supervision,
in-place MuJoCo recovery + SegmentChunkDataset integration, and run_dagger_iter orchestration."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
	sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts" / "vla") not in sys.path:
	sys.path.insert(0, str(ROOT / "scripts" / "vla"))

import run_dagger_iter

from digital_twin_armigo.agent.student_rollout import (
	PolicyStep,
	ReplayPolicy,
	Segment,
	StudentObservation,
	run_episode,
)
from digital_twin_armigo.agent.twin_mujoco import INITIAL_Q, sample_case
from digital_twin_armigo.agent.twin_record import EpisodeTrace, segment_labels
from digital_twin_armigo.agent.vla_grader import grade
from digital_twin_armigo.vla.contract import IGNORE_INDEX
from digital_twin_armigo.vla.dagger import (
	TakeoverConfig,
	TakeoverGate,
	TakeoverTwin,
	assert_train_layouts_only,
	episode_to_dataset_dict,
	load_suite_cases,
	run_takeover_episode,
	should_stop_dagger,
)
from digital_twin_armigo.vla.dataset import SegmentChunkDataset

TRAIN_CASE_DEFECTIVE = (1_100_001, True, 0)  # train split, side-face defect (fast)
TRAIN_CASE_NORMAL = (1_100_002, False, 5)  # train split, normal (exercises bottom + bowl release)


class ConstantScorer:
	"""Headless constant scorer so MuJoCo tests run on CPU without GPU contention."""

	def __init__(self, score_val: float = 1.0) -> None:
		self._score = float(score_val)

	def threshold(self, camera: str) -> float:
		del camera
		return 0.0

	def score(self, camera: str, image: np.ndarray) -> float:
		del camera, image
		return self._score


class DriftingScanPolicy:
	"""Replays ``prepare`` cleanly (or holds ready), then drifts on joint 0 during ``scan_view``."""

	def __init__(self, *, drift_kind: str = "scan_view", drift_per_frame: float = 0.03) -> None:
		self.drift_kind = drift_kind
		self.drift_per_frame = float(drift_per_frame)
		self.kind = ""
		self.q_start: np.ndarray | None = None

	def start(self, task: dict, segment_id: int) -> None:
		del segment_id
		self.kind = str(task["kind"])
		self.q_start = None

	def act(self, obs: StudentObservation) -> PolicyStep:
		if self.q_start is None:
			self.q_start = obs.q.copy()
		if self.kind == "prepare":
			# Hold rest for 9 frames, then interpolate to ready_q over 18 frames, then finish cleanly.
			ready_q = np.array([0.0, -0.35, 0.25, 1.35, 0.0, 0.85], dtype=np.float64)
			if obs.segment_frame < 9:
				return PolicyStep(target=INITIAL_Q.copy(), phase="rest")
			if obs.segment_frame < 27:
				s = float(obs.segment_frame - 8) / 18.0
				w = 10 * s**3 - 15 * s**4 + 6 * s**5
				return PolicyStep(target=INITIAL_Q + w * (ready_q - INITIAL_Q), phase="approach")
			return PolicyStep(done=True)

		if self.kind == self.drift_kind:
			tgt = self.q_start.copy()
			tgt[0] += self.drift_per_frame * float(obs.segment_frame + 1)
			return PolicyStep(target=tgt, phase="approach")

		return PolicyStep(target=obs.q.copy(), phase="approach")


class StallingScanPolicy(DriftingScanPolicy):
	"""Runs ``prepare`` cleanly, then freezes still on ``scan_view`` without emitting ``done``."""

	def act(self, obs: StudentObservation) -> PolicyStep:
		if self.kind == "prepare":
			return super().act(obs)
		return PolicyStep(target=self.q_start.copy() if self.q_start is not None else obs.q.copy(), phase="approach")


# -----------------------------------------------------------------------------------------------------------
# 1. Train-only suite guard tests
# -----------------------------------------------------------------------------------------------------------


def test_assert_train_layouts_only_rejects_val_test_and_bench16(tmp_path: Path):
	# Valid training suite must pass cleanly.
	train_cases, _ = load_suite_cases(ROOT / "config/vla/train.json")
	assert_train_layouts_only(ROOT / "config/vla/train.json", train_cases)

	# Direct paths to val, sealed test, and bench16 MUST fail.
	for bad_suite in (
		"config/vla/val.json",
		"config/vla/test.json",
		"config/agent_bench/bench16.json",
	):
		with pytest.raises(ValueError, match="forbidden"):
			assert_train_layouts_only(ROOT / bad_suite)

	# Disguised suite copying even ONE seed from val, test, or bench16 MUST fail.
	val_cases, _ = load_suite_cases(ROOT / "config/vla/val.json")
	test_cases, _ = load_suite_cases(ROOT / "config/vla/test.json")
	bench_cases, _ = load_suite_cases(ROOT / "config/agent_bench/bench16.json")

	for label, forbidden_case in (("val", val_cases[0]), ("test", test_cases[0]), ("bench16", bench_cases[0])):
		disguised = tmp_path / f"disguised_{label}.json"
		mixed_cases = [list(train_cases[0]), list(forbidden_case)]
		disguised.write_text(
			json.dumps({"name": "fake-train", "split": "train", "sealed": False, "cases": mixed_cases}),
			encoding="utf-8",
		)
		with pytest.raises(ValueError, match="forbidden"):
			assert_train_layouts_only(disguised, mixed_cases)
		# Even without suite_path, checking cases alone catches seed overlap.
		with pytest.raises(ValueError, match="forbidden"):
			assert_train_layouts_only(None, [forbidden_case])

	# run_takeover_episode also enforces train-only guard on single cases.
	with pytest.raises(ValueError, match="forbidden"):
		run_takeover_episode(val_cases[0], ConstantScorer(), tmp_path / "bad_ep", policy=DriftingScanPolicy())


# -----------------------------------------------------------------------------------------------------------
# 2. Zero ground-truth leakage & all 4 takeover gates
# -----------------------------------------------------------------------------------------------------------


def test_takeover_gates_zero_ground_truth_leakage(tmp_path: Path):
	pytest.importorskip("mujoco")
	pytest.importorskip("cv2")
	from digital_twin_armigo.constants import DUPLO_2X4

	layout = sample_case(*TRAIN_CASE_DEFECTIVE)
	trace = EpisodeTrace(tmp_path / "leak", video=False)
	twin = TakeoverTwin(layout, ConstantScorer(), trace, DriftingScanPolicy(), sensing="realistic")
	try:
		assert twin.prepare() is None
		assert twin._perceive() is None
		assert twin.brick_est is not None

		# Poison simulator ground-truth hooks: TakeoverGate MUST NEVER call or read any of them!
		def _boom(*_args, **_kwargs):
			raise AssertionError("ground-truth leakage: TakeoverGate read simulator truth!")

		twin.cell.brick_pos = _boom
		twin.cell.brick_quat_xyzw = _boom
		twin.cell.brick_vel = _boom
		twin.layout.brick_xy = (9.99, 9.99)
		twin.layout.brick_size = DUPLO_2X4

		raw_obs = {
			"q": twin.cell.joints(),
			"tcp": twin.cell.tcp_pos(),
			"jaw_normal": np.zeros(2),
			"gripper_cmd": float(twin.cell.joints()[5]),
		}
		belief = twin._brick_belief()
		assert belief is not None
		from digital_twin_armigo.vla.dagger import _sanitized_realistic_obs

		safe_obs = _sanitized_realistic_obs(raw_obs, belief)
		gate = TakeoverGate(TakeoverConfig(deviation_thresh_rad=0.25, soft_deviation_thresh_rad=0.10))
		seg = Segment("grasp")
		with twin._belief():
			gate.start_segment(
				seg,
				obs=safe_obs,
				realistic_state=twin._realistic_state_for_gate(seg),
				segment_start_frame=twin.ep.n_frames,
				shadow_expert_factory=twin._make_shadow_expert_for_gate,
			)

		# Record shadow plan with poisoned truth vs moved simulator brick qpos -> must be bitwise identical!
		decisions_a = []
		for f in range(5):
			decisions_a.append(
				gate.evaluate(frame=twin.ep.n_frames + f, segment_frame=f, obs=safe_obs, blocked=False, lost=False)
			)
		plan_a = gate._q_expected.copy()

		# Move simulator brick physically far away without re-perceiving -> realistic shadow plan is unchanged!
		adr = twin.cell.brick_qadr
		twin.cell.data.qpos[adr : adr + 3] = np.array([0.40, 0.25, 0.02])
		gate_b = TakeoverGate(TakeoverConfig(deviation_thresh_rad=0.25, soft_deviation_thresh_rad=0.10))
		with twin._belief():
			gate_b.start_segment(
				seg,
				obs=safe_obs,
				realistic_state=twin._realistic_state_for_gate(seg),
				segment_start_frame=twin.ep.n_frames,
				shadow_expert_factory=twin._make_shadow_expert_for_gate,
			)
		for f in range(5):
			gate_b.evaluate(frame=twin.ep.n_frames + f, segment_frame=f, obs=safe_obs, blocked=False, lost=False)
		assert np.array_equal(plan_a, gate_b._q_expected)
	finally:
		trace.close()
		twin.close()


def test_takeover_gate_unit_conditions():
	q0 = np.zeros(6, dtype=np.float64)
	targets = [q0.copy() for _ in range(40)]
	obs0 = {"q": q0.copy(), "tcp": np.zeros(3), "jaw_normal": np.zeros(2), "gripper_cmd": 0.0}

	# a) joint_deviation + soft warning onset t_dev + reset when returning inside band
	gate = TakeoverGate(TakeoverConfig(deviation_thresh_rad=0.25, soft_deviation_thresh_rad=0.10))
	gate.start_segment(Segment("prepare"), obs=obs0, realistic_state={}, segment_start_frame=10, shadow_targets=targets)
	assert not gate.evaluate(frame=10, segment_frame=0, obs=obs0).trigger
	# Soft warning crossed at frame 11, then cleared at 12, crossed again at 13, hard trigger at 15
	assert not gate.evaluate(frame=11, segment_frame=1, obs={**obs0, "q": np.array([0.12, 0, 0, 0, 0, 0])}).trigger
	assert gate.deviation_onset_frame == 11
	assert not gate.evaluate(frame=12, segment_frame=2, obs={**obs0, "q": np.array([0.05, 0, 0, 0, 0, 0])}).trigger
	assert gate.deviation_onset_frame is None
	assert not gate.evaluate(frame=13, segment_frame=3, obs={**obs0, "q": np.array([0.15, 0, 0, 0, 0, 0])}).trigger
	assert not gate.evaluate(frame=14, segment_frame=4, obs={**obs0, "q": np.array([0.22, 0, 0, 0, 0, 0])}).trigger
	dec = gate.evaluate(frame=15, segment_frame=5, obs={**obs0, "q": np.array([0.28, 0, 0, 0, 0, 0])})
	assert dec.trigger and dec.gate == "joint_deviation" and dec.blind_start_frame == 13

	# b) gripper_blocked
	gate_blk = TakeoverGate(TakeoverConfig())
	gate_blk.start_segment(
		Segment("grasp"), obs=obs0, realistic_state={}, segment_start_frame=20, shadow_targets=targets
	)
	dec_blk = gate_blk.evaluate(frame=25, segment_frame=5, obs=obs0, blocked=True)
	assert dec_blk.trigger and dec_blk.gate == "gripper_blocked" and dec_blk.blind_start_frame == 22

	# c) brick_lost (only on held segments)
	gate_lost = TakeoverGate(TakeoverConfig())
	gate_lost.start_segment(
		Segment("scan_view", {"view": 1, "views": 2}), obs=obs0, realistic_state={}, shadow_targets=targets
	)
	assert not gate_lost.evaluate(frame=0, segment_frame=0, obs=obs0, lost=True, no_dual_frames=10).trigger
	gate_lost.start_segment(
		Segment("lift"), obs=obs0, realistic_state={}, segment_start_frame=30, shadow_targets=targets
	)
	dec_lost = gate_lost.evaluate(frame=45, segment_frame=15, obs=obs0, lost=True, no_dual_frames=10)
	assert dec_lost.trigger and dec_lost.gate == "brick_lost" and dec_lost.blind_start_frame == 35

	# d) budget_or_stall: hold frames do NOT trigger stall; non-hold moving shadow + still student DOES trigger
	moving_targets = [np.array([0.05 * i, 0, 0, 0, 0, 0]) for i in range(15)]
	gate_hold = TakeoverGate(TakeoverConfig(stall_frames=4, max_segment_frames=10, deviation_thresh_rad=5.0))
	gate_hold.start_segment(
		Segment("prepare"), obs=obs0, realistic_state={}, segment_start_frame=0, shadow_targets=targets
	)
	for f in range(8):
		assert not gate_hold.evaluate(frame=f, segment_frame=f, obs=obs0).trigger  # in hold (targets constant)
	# At frame 10 budget triggers
	dec_bud = gate_hold.evaluate(frame=10, segment_frame=10, obs=obs0)
	assert dec_bud.trigger and dec_bud.gate == "budget_or_stall" and dec_bud.reason == "segment_frame_budget"

	gate_stall = TakeoverGate(TakeoverConfig(stall_frames=4, deviation_thresh_rad=5.0))
	gate_stall.start_segment(
		Segment("scan_view", {"view": 1, "views": 2}),
		obs=obs0,
		realistic_state={},
		segment_start_frame=0,
		shadow_targets=moving_targets,
	)
	stalled = [gate_stall.evaluate(frame=f, segment_frame=f, obs=obs0) for f in range(6)]
	trig_idx = next(i for i, d in enumerate(stalled) if d.trigger)
	assert stalled[trig_idx].gate == "budget_or_stall" and stalled[trig_idx].reason == "stalled_non_hold"


# -----------------------------------------------------------------------------------------------------------
# 3. Real MuJoCo TakeoverTwin rollouts + zero back-fill + SegmentChunkDataset
# -----------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
	("case", "score_val"),
	[
		(TRAIN_CASE_DEFECTIVE, 1.0),
		(TRAIN_CASE_NORMAL, -1.0),
	],
)
def test_mujoco_replay_policy_has_zero_false_takeovers(tmp_path: Path, case: tuple, score_val: float):
	pytest.importorskip("mujoco")
	pytest.importorskip("langgraph")
	pytest.importorskip("av")
	pytest.importorskip("cv2")

	expert_dir = tmp_path / "expert"
	expert_run = run_episode(case, ConstantScorer(score_val), expert_dir, sensing="realistic", video=False)
	assert expert_run["result"]["verdict"] == "SUCCESS"

	replay = ReplayPolicy.from_trace(expert_dir)
	takeover_dir = tmp_path / "takeover_replay"
	run = run_takeover_episode(
		case,
		ConstantScorer(score_val),
		takeover_dir,
		policy=replay,
		segment_monitor=replay.check_segment,
	)
	assert run["result"]["verdict"] == "SUCCESS"
	assert run["result"]["student_eligible"] is True
	assert run["takeover"]["triggered"] is False
	with np.load(takeover_dir / "branch-0" / "actions.npz", allow_pickle=False) as rows:
		assert not rows["is_takeover"].any()
		assert not rows["is_blind"].any()
		assert not rows["supervise_action"].any()  # student frames never get expert supervision


def test_mujoco_takeover_joint_deviation_no_backfill_and_dataset(tmp_path: Path):
	pytest.importorskip("mujoco")
	pytest.importorskip("langgraph")
	pytest.importorskip("av")
	pytest.importorskip("cv2")

	ep_dir = tmp_path / "ep_drift"
	cfg = TakeoverConfig(deviation_thresh_rad=0.22, soft_deviation_thresh_rad=0.08)
	run = run_takeover_episode(
		TRAIN_CASE_DEFECTIVE,
		ConstantScorer(1.0),
		ep_dir,
		policy=DriftingScanPolicy(drift_kind="scan_view", drift_per_frame=0.03),
		takeover_config=cfg,
	)

	res = run["result"]
	to = run["takeover"]
	assert res["verdict"] == "SUCCESS"
	assert res["rewinds"] == 0
	assert res["student_eligible"] is True
	assert to["triggered"] is True
	assert to["gate"] == "joint_deviation"

	t_to = int(to["frame"])
	t_dev = int(to["deviation_onset_frame"])
	assert 0 < t_dev < t_to < run["frames"]

	# Verify explicit takeover event and takeover segment end in events.jsonl.
	events = [json.loads(line) for line in (ep_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
	to_events = [e for e in events if e["kind"] == "takeover"]
	assert len(to_events) == 1
	assert to_events[0]["frame"] == t_to and to_events[0]["gate"] == "joint_deviation"
	end_events = [e for e in events if e["kind"] == "segment_end" and e["reason"] == "takeover"]
	assert len(end_events) == 1 and end_events[0]["frame"] == t_to

	# Verify exact boolean mask columns in branch-0/actions.npz (matching w2-s5-perturb student_frames schema).
	with np.load(ep_dir / "branch-0" / "actions.npz", allow_pickle=False) as rows:
		n = len(rows["action"])
		assert n == run["frames"]
		for col in ("is_blind", "is_takeover", "supervise_action"):
			assert col in rows.files
			assert rows[col].dtype == np.bool_
			assert rows[col].shape == (n,)
		is_blind = rows["is_blind"]
		is_takeover = rows["is_takeover"]
		supervise_action = rows["supervise_action"]

		# ZERO back-fill before takeover; full takeover supervision from t_takeover onward; blind in [t_dev, t_takeover).
		assert not supervise_action[:t_to].any()
		assert supervise_action[t_to:].all()
		assert not is_takeover[:t_to].any()
		assert is_takeover[t_to:].all()
		assert not is_blind[:t_dev].any()
		assert is_blind[t_dev:t_to].all()
		assert not is_blind[t_to:].any()

	# S7 segment_labels validates cleanly across the student-to-expert handoff.
	labels = segment_labels(ep_dir)
	assert len(labels["index"]) == n
	assert labels["end_reason"][t_to - 1] == "takeover"
	assert labels["end_reason"][t_to] == "done"

	# S6 vla_grader checks (eligible, decision_crop, belief_error, collisions) pass cleanly.
	g = grade(ep_dir)
	assert g["checks"]["eligible"]["status"] == "pass"
	assert g["checks"]["decision_crop"]["status"] == "pass"
	assert g["checks"]["belief_error"]["status"] == "pass"
	assert g["checks"]["collisions"]["status"] == "pass"

	# SegmentChunkDataset loads it cleanly: student frames masked (no back-fill), blind masked, chunks cut, 2x weight.
	ep_dict = episode_to_dataset_dict(ep_dir, episode_index=0)
	ds = SegmentChunkDataset.from_episodes([ep_dict], horizon=20, takeover_weight=2.0, return_torch=False)
	assert len(ds) >= n

	for idx in range(t_to):
		sample = ds[idx]
		assert not sample["action_mask"].any()  # zero back-fill & blind frames masked
		assert int(sample["end_step_target"]) == IGNORE_INDEX
		assert float(sample["sample_weight"]) == 1.0

	for idx in range(t_to, n):
		sample = ds[idx]
		assert float(sample["sample_weight"]) == 2.0
		assert sample["action_mask"].any()
		assert int(sample["end_step_target"]) >= 0

	# Verify SegmentChunkDataset cuts chunks at takeover onset even when pre-takeover rows are hypothetically supervised.
	ep_dict_cut = episode_to_dataset_dict(ep_dir, episode_index=1)
	ep_dict_cut["frames"]["mask.supervise_action"][:t_to] = True
	ep_dict_cut["frames"]["mask.is_blind"][:] = False
	ds_cut = SegmentChunkDataset.from_episodes([ep_dict_cut], horizon=20, takeover_weight=2.0, return_torch=False)
	pre = ds_cut[t_to - 3]
	assert pre["action_mask"].tolist()[:5] == [True, True, True, False, False]


def test_mujoco_takeover_budget_or_stall_recovers_to_success(tmp_path: Path):
	pytest.importorskip("mujoco")
	pytest.importorskip("langgraph")
	pytest.importorskip("av")
	pytest.importorskip("cv2")

	ep_dir = tmp_path / "ep_stall"
	cfg = TakeoverConfig(stall_frames=12, stall_eps_rad=0.002, deviation_thresh_rad=5.0, soft_deviation_thresh_rad=4.0)
	run = run_takeover_episode(
		TRAIN_CASE_DEFECTIVE,
		ConstantScorer(1.0),
		ep_dir,
		policy=StallingScanPolicy(),
		takeover_config=cfg,
	)
	assert run["result"]["verdict"] == "SUCCESS"
	assert run["result"]["rewinds"] == 0
	assert run["result"]["student_eligible"] is True
	assert run["takeover"]["triggered"] is True
	assert run["takeover"]["gate"] == "budget_or_stall"
	labels = segment_labels(ep_dir)
	assert len(labels["index"]) == run["frames"]


# -----------------------------------------------------------------------------------------------------------
# 4. S11 stopping rule & run_dagger_iter.py 5-step orchestration
# -----------------------------------------------------------------------------------------------------------


def test_should_stop_dagger_rule():
	assert should_stop_dagger([]) is False
	assert should_stop_dagger([0.60]) is False
	assert should_stop_dagger([0.60, 0.61]) is False
	# Two consecutive gains of exactly 0.02 (2.0 pts) must NOT stop (< 0.02 required).
	assert should_stop_dagger([0.70, 0.72, 0.74]) is False
	assert should_stop_dagger([0.50, 0.55, 0.56]) is False
	assert should_stop_dagger([0.50, 0.55, 0.56, 0.57]) is True
	assert should_stop_dagger([0.50, 0.55, 0.56, 0.55]) is True


def test_run_dagger_loop_5step_orchestration_and_stopping(tmp_path: Path):
	calls: list[tuple[str, int]] = []
	eval_seq = iter(
		[0.60, 0.65, 0.66, 0.665, 0.80]
	)  # stops at iter 4? Wait: 65->66 (+0.01), 66->665 (+0.005) => iter 4!

	base_ds = tmp_path / "d_base"
	base_ds.mkdir()
	init_ckpt = tmp_path / "init_ckpt"
	init_ckpt.mkdir()

	def fake_rollout(it, ckpt, cases, raw_dir, cfg):
		calls.append(("rollout", it))
		assert_train_layouts_only(ROOT / "config/vla/train.json", cases)
		raw_dir.mkdir(parents=True, exist_ok=True)
		rows = []
		# Episode 0: SUCCESS + student_eligible; Episode 1: FAILED (filtered out); Episode 2: SUCCESS + ineligible
		for idx, (verdict, elig) in enumerate([("SUCCESS", True), ("FAILED", False), ("SUCCESS", False)]):
			ep = raw_dir / f"case-{idx:04d}"
			ep.mkdir()
			rows.append(
				{
					"episode": ep,
					"result": {"verdict": verdict, "student_eligible": elig},
					"takeover": {"triggered": True, "frame": 10},
				}
			)
		return rows

	def fake_grade(ep: Path):
		calls.append(("grade", int(ep.parent.parent.name.split("-")[1])))
		return {"episode": str(ep), "passed": True}

	def fake_export(it, eps, d_takeover):
		calls.append(("export", it))
		assert len(eps) == 1  # only episode 0 kept!
		d_takeover.mkdir(parents=True, exist_ok=True)
		return d_takeover

	def fake_merge(it, shards, d_union):
		calls.append(("merge", it))
		assert shards[0] == base_ds
		assert len(shards) == 1 + it
		d_union.mkdir(parents=True, exist_ok=True)
		return d_union

	def fake_train(it, d_union, prev_ckpt, ckpt_out):
		calls.append(("train", it))
		assert d_union.exists()
		if it == 1:
			assert prev_ckpt == init_ckpt
		else:
			assert f"iter-{it - 1:02d}" in str(prev_ckpt)
		best = ckpt_out / "best"
		best.mkdir(parents=True, exist_ok=True)
		return best

	def fake_eval(it, ckpt, val_suite, eval_out):
		calls.append(("eval", it))
		assert val_suite == ROOT / "config/vla/val.json"
		assert ckpt.exists()
		eval_out.mkdir(parents=True, exist_ok=True)
		return {"student_sort_rate": next(eval_seq)}

	out_dir = tmp_path / "dagger_out"
	summary = run_dagger_iter.run_dagger_loop(
		train_suite=ROOT / "config/vla/train.json",
		val_suite=ROOT / "config/vla/val.json",
		out_dir=out_dir,
		init_ckpt=init_ckpt,
		base_dataset=base_ds,
		iters=5,
		episodes_per_iter=3,
		takeover_weight=2.0,
		rollout_fn=fake_rollout,
		grade_fn=fake_grade,
		export_fn=fake_export,
		merge_fn=fake_merge,
		train_fn=fake_train,
		eval_fn=fake_eval,
	)

	# 0.60 -> 0.65 (+0.05) -> 0.66 (+0.01) -> 0.665 (+0.005): stops automatically at iter 4!
	assert summary["iters_completed"] == 4
	assert summary["stopped_early"] is True
	assert summary["stop_reason"] == "converged"
	assert summary["val_rates"] == [0.60, 0.65, 0.66, 0.665]
	assert (out_dir / "dagger_history.json").is_file()

	# Verify exact 5-step ordering on every iteration
	for it in range(1, 5):
		iter_ops = [op for op, i in calls if i == it]
		assert iter_ops == ["rollout", "grade", "export", "merge", "train", "eval"]

	# Reject forbidden train/val suites
	with pytest.raises(ValueError, match="forbidden"):
		run_dagger_iter.run_dagger_loop(
			train_suite=ROOT / "config/vla/val.json",
			val_suite=ROOT / "config/vla/val.json",
			out_dir=tmp_path / "bad_train",
			dry_run=True,
		)
	with pytest.raises(ValueError, match="sealed"):
		run_dagger_iter.run_dagger_loop(
			train_suite=ROOT / "config/vla/train.json",
			val_suite=ROOT / "config/vla/test.json",
			out_dir=tmp_path / "bad_val",
			dry_run=True,
		)


def test_run_dagger_iter_cli_dry_run(tmp_path: Path):
	out_dir = tmp_path / "cli_dry"
	rc = run_dagger_iter.main(
		[
			"--out",
			str(out_dir),
			"--iters",
			"4",
			"--episodes-per-iter",
			"2",
			"--takeover-weight",
			"2.0",
			"--dry-run",
		]
	)
	assert rc == 0
	hist = json.loads((out_dir / "dagger_history.json").read_text(encoding="utf-8"))
	assert hist["dry_run"] is True
	assert hist["stopped_early"] is True
	assert hist["iters_completed"] == 3  # dry_rates: [0.64, 0.61, 0.62] -> -0.03, +0.01 -> stops at iter 3
