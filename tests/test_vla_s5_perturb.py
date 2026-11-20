"""S5 perturbation branches: hook units, detection/metric logic, mask columns, no GT leak, unperturbed parity.

The MuJoCo tests (second half) run real MuJoCo branches (CPU, ~20-50 s each); the rest are pure logic.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

pytest.importorskip("mujoco")

from digital_twin_armigo.agent import perturbations as P
from digital_twin_armigo.agent.student_rollout import POLICY_FIELDS, StudentObservation


class ConstantScorer:
	"""Every readable view scores above threshold: defective on the first wrist view (short episode)."""

	def threshold(self, camera):
		return 0.0

	def score(self, camera, image):
		return 1.0


class ZeroScorer:
	"""Every view is clean: the brick is a normal one (needed for the park / put-down path)."""

	def threshold(self, camera):
		return 0.5

	def score(self, camera, image):
		return 0.0


# kind -> (case, scorer class, may fail to recover: the 80 % gate is statistical, one seed proves only the mechanics)
SCENARIOS = {
	"pre_grasp_displace": ((600001, True, 0), ConstantScorer, False),
	"in_jaw_offset": ((600001, True, 0), ConstantScorer, True),
	"mid_carry_drop": ((600001, True, 0), ConstantScorer, False),
	"post_putdown_nudge": ((600002, False, 0), ZeroScorer, False),
}


# ---- pure logic ---------------------------------------------------------------------------------------------------


def test_types_constructor_and_clone():
	assert P.PERTURBATION_TYPES == ("pre_grasp_displace", "in_jaw_offset", "mid_carry_drop", "post_putdown_nudge")
	with pytest.raises(ValueError):
		P.make_perturbation("teleport")
	for kind in P.PERTURBATION_TYPES:
		hook = P.make_perturbation(kind, after_frame=10, offset_m=0.01)
		hook.triggered = True
		fresh = hook.clone(after_frame=20)
		assert (fresh.kind, fresh.after_frame, fresh.offset_m, fresh.triggered) == (kind, 20, 0.01, False)


def test_hook_waits_for_after_frame_and_fires_once():
	hook = P.make_perturbation("pre_grasp_displace", after_frame=100)
	twin = type("T", (), {"ep": type("E", (), {"n_frames": 5})()})()
	assert hook(twin, {}, "approach") is None  # before after_frame: touches nothing else of the twin
	hook.triggered = True
	assert hook(twin, {}, "approach") is None


def test_quaternion_product():
	ident = np.array([0.0, 0.0, 0.0, 1.0])
	q = np.array([0.1, -0.2, 0.3, 0.9])
	q /= np.linalg.norm(q)
	assert np.allclose(P._quat_mul_xyzw(ident, q), q) and np.allclose(P._quat_mul_xyzw(q, ident), q)
	z90 = np.array([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)])
	assert np.allclose(P._quat_mul_xyzw(z90, z90), [0.0, 0.0, 1.0, 0.0])


def _ev(kind, frame, **kw):
	return {"kind": kind, "frame": frame, "branch": 1, **kw}


def test_detection_matching_per_type():
	def span(kind, *after, **pkw):
		events = [_ev("segment", 5, segment_kind="grasp"), _ev("perturbation", 10, perturbation=kind, **pkw), *after]
		return P.infer_blind_span(events), P.find_perturbation_detection(events)

	# an event before the perturbation never counts; undetected -> no span
	assert span("pre_grasp_displace", _ev("view", 20))[0] is None
	events = [_ev("failure", 3), _ev("perturbation", 10, perturbation="mid_carry_drop"), _ev("failure", 40)]
	assert P.infer_blind_span(events) == (10, 40)
	assert span("pre_grasp_displace", _ev("retry", 33))[0] == (10, 33)
	assert span("mid_carry_drop", _ev("segment", 25, segment_kind="let_go"))[0] == (10, 25)
	assert span("in_jaw_offset", _ev("placement_check", 30))[0] == (10, 30)
	assert span("mid_carry_drop", _ev("placement_check", 30))[0] is None
	# nudge: only a re-perceived brick that moved (flag, missing, or far from where it was) is a detection
	nudge = {"before_pos_m": [0.2, 0.0, 0.01]}
	assert (
		span("post_putdown_nudge", _ev("brick_perceived", 30, moved=False, position_m=[0.2, 0.001, 0.01]), **nudge)[0]
		is None
	)
	assert span("post_putdown_nudge", _ev("brick_perceived", 31, moved=True), **nudge)[0] == (10, 31)
	assert span("post_putdown_nudge", _ev("brick_perceived", 32, found=False), **nudge)[0] == (10, 32)
	far = _ev("brick_perceived", 33, position_m=[0.25, 0.0, 0.01])
	assert span("post_putdown_nudge", far, **nudge)[0] == (10, 33)
	assert P.infer_blind_span([_ev("failure", 3)]) is None  # no perturbation at all


def test_recovery_started_after_detection():
	events = [
		_ev("perturbation", 10, perturbation="mid_carry_drop"),
		_ev("failure", 20),
		_ev("retry", 20),
	]
	assert P.recovery_started_after_detection(events)
	assert not P.recovery_started_after_detection(events[:2])
	assert not P.recovery_started_after_detection([_ev("failure", 20)])


def test_metrics_split_and_gate():
	def row(kind, *, detected=True, started=True, sorted_=True):
		return {
			"perturbation_type": kind,
			"detected": detected,
			"recovery_started": started,
			"sorted": sorted_,
		}

	rows = [row("mid_carry_drop") for _ in range(4)] + [row("mid_carry_drop", sorted_=False)]
	rows += [row("pre_grasp_displace"), row("pre_grasp_displace", detected=False, started=False, sorted_=False)]
	m = P.compute_perturbation_metrics(rows)
	drop = m["by_type"]["mid_carry_drop"]
	assert (drop["attempted"], drop["sorted"], drop["recovery_rate"], drop["gate_pass"]) == (5, 4, 0.8, True)
	pre = m["by_type"]["pre_grasp_displace"]
	assert (pre["attempted"], pre["detected"], pre["recovery_rate"], pre["gate_pass"]) == (2, 1, 0.5, False)
	assert m["overall"]["attempted"] == 7 and m["gate_pass"] is False  # one failing type fails the gate
	assert P.compute_perturbation_metrics([])["gate_pass"] is False


def test_checkpoint_choice():
	cks = [type("C", (), {"held": h, "frame": i})() for i, h in enumerate((None, None, {"q_lift": []}, None))]
	assert P.choose_checkpoint_for_perturbation(cks, "post_putdown_nudge") is cks[0]
	assert P.choose_checkpoint_for_perturbation(cks, "mid_carry_drop") is cks[2]
	assert P.choose_checkpoint_for_perturbation(cks, "pre_grasp_displace") is cks[3]
	with pytest.raises(ValueError):
		P.choose_checkpoint_for_perturbation([], "mid_carry_drop")


# ---- recorder columns ---------------------------------------------------------------------------------------------


def test_recorder_mask_defaults_and_overrides(tmp_path):
	from digital_twin_armigo.agent.twin_record import EpisodeTrace

	trace = EpisodeTrace(tmp_path / "t", video=False)
	state = np.zeros(6)
	trace.command(state, state, "approach")
	trace.command(state, state, "approach", is_blind=True, is_takeover=True, supervise_action=False)
	assert trace._is_blind == [False, True]
	assert trace._is_takeover == [False, True]
	assert trace._supervise_action == [True, False]
	trace.set_row_masks(is_blind=[True, True], supervise_action=[False, True])
	assert trace._is_blind == [True, True] and trace._is_takeover == [False, True]
	assert trace._supervise_action == [False, True]
	with pytest.raises(ValueError):
		trace.set_row_masks(is_blind=[True])


def test_lerobot_features_keep_the_s9_mask_columns_and_no_gt():
	from digital_twin_armigo.agent import twin_record

	feats = {**twin_record.INPUT_FEATURES, **twin_record.SEGMENT_FEATURES}
	for name in ("mask.is_blind", "mask.is_takeover", "mask.supervise_action"):
		assert feats[name]["dtype"] == "bool" and tuple(feats[name]["shape"]) == (1,)
	assert not [k for k in feats if any(w in k for w in ("perturb", "truth", "brick_pos", "true_", "before_pos"))]


def test_student_observation_has_no_perturbation_or_gt_fields():
	names = {f.name for f in dataclasses.fields(StudentObservation)}
	assert names == {
		"frame", "task", "segment_id", "segment_frame", "q", "_images", "q_prev", "prev_action", "gripper_cmd",
	}  # fmt: skip
	assert not any(w in " ".join(POLICY_FIELDS) for w in ("perturb", "truth", "brick_pos", "blind"))


# ---- MuJoCo integration -------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
	pytest.importorskip("langgraph")
	pytest.importorskip("av")
	pytest.importorskip("cv2")
	cache = {}

	def get(kind):
		if kind not in cache:
			case, scorer, _ = SCENARIOS[kind]
			root = tmp_path_factory.mktemp(kind)
			attempt = P.run_perturbation_branch(case, scorer(), root / "raw", root / "st", kind)
			cache[kind] = (attempt, root)
		return cache[kind]

	return get


def _events(path):
	return [json.loads(line) for line in (path / "events.jsonl").read_text(encoding="utf-8").splitlines() if line]


@pytest.mark.parametrize("kind", P.PERTURBATION_TYPES)
def test_branch_perturbs_visibly_and_the_expert_detects_it(runs, kind):
	attempt, root = runs(kind)
	assert attempt["error"] is None, attempt["error"]
	assert attempt["triggered"], "perturbation never fired"
	assert attempt["perturbation_rewinds"] == 1 and attempt["rewinds"] == 0  # online recovery budget untouched
	events = _events(root / "raw")
	branch = attempt["branch_id"]
	assert any(e["kind"] == "perturbation_branch" and e["branch"] == branch for e in events)
	p = next(e for e in events if e["kind"] == "perturbation")
	assert p["perturbation"] == kind and p["branch"] == branch
	assert np.linalg.norm(np.subtract(p["after_pos_m"], p["before_pos_m"])) > 0.005  # the brick really moved
	assert attempt["detected"], "no detection event after the perturbation"
	lo, hi = attempt["blind_span"]
	assert lo == p["frame"] and hi >= lo
	if not SCENARIOS[kind][2]:
		assert attempt["sorted"] and attempt["verdict"] == "SUCCESS" and attempt["recovery_started"]


@pytest.mark.parametrize("kind", P.PERTURBATION_TYPES)
def test_stitched_columns_match_the_s9_s11_contract(runs, kind):
	attempt, root = runs(kind)
	if not attempt["sorted"]:
		pytest.skip("no stitched episode for an unrecovered branch")
	st = root / "st"
	with np.load(st / "branch-0" / "actions.npz") as z:
		n = len(z["action"])
		for col in ("is_blind", "is_takeover", "supervise_action"):
			assert z[col].dtype == bool and z[col].shape == (n,), col
		assert not z["is_takeover"].any() and z["supervise_action"].all()  # the expert acts on every frame
		lo, hi = attempt["blind_span"]
		blind = z["is_blind"]
		assert blind[lo:hi].all() and not blind[:lo].any() and not blind[hi:].any()  # half-open [perturb, detect)
		assert int(blind.sum()) == hi - lo
		assert (z["segment"] >= 0).all() and len(z["segment"]) == n
		assert set(z.files) >= {"action", "state", "segment", "is_blind"}
		assert not [k for k in z.files if any(w in k for w in ("perturb", "truth", "brick", "true_"))]
	manifest = json.loads((st / "episode.json").read_text(encoding="utf-8"))
	assert manifest["branches"] == [{"id": 0, "start_frame": 0, "actions": n}]
	assert manifest["result"]["verdict"] == "SUCCESS" and manifest["result"]["active_branch"] == 0
	assert "perturbation" in {e["kind"] for e in _events(st)}


def test_stitched_branch_passes_the_s0_replay_gate_and_policy_sees_no_gt(runs):
	from digital_twin_armigo.agent import student_rollout as sr

	kind = "mid_carry_drop"
	attempt, root = runs(kind)
	assert attempt["sorted"]
	seen = []
	orig = sr.ReplayPolicy.start

	def spy(self, task, segment_id):
		seen.append(dict(task))
		return orig(self, task, segment_id)

	sr.ReplayPolicy.start = spy
	try:
		_, checks, _ = P.replay_perturbation_episode(
			root / "st",
			ConstantScorer(),
			root / "replay",
			perturb=P.make_perturbation(kind, after_frame=attempt["start_frame"]),
		)
	finally:
		sr.ReplayPolicy.start = orig
	assert all(checks.values()), checks
	assert seen and all(tuple(t) == POLICY_FIELDS for t in seen)
	assert not any(w in json.dumps(seen) for w in ("perturb", "before_pos", "after_pos", "drop_m"))


def test_unperturbed_episode_is_unchanged_by_the_s5_hooks(tmp_path):
	from digital_twin_armigo.agent.student_rollout import run_episode

	run = run_episode((600001, True, 0), ConstantScorer(), tmp_path / "e")
	assert run["result"]["verdict"] == "SUCCESS"
	assert "perturbation_rewinds" not in run["result"]  # result schema identical to pre-S5 runs
	assert {"perturbation", "perturbation_branch"}.isdisjoint(e["kind"] for e in run["events"])
	assert not [e for e in run["events"] if e["kind"] == "brick_perceived" and "moved_mm" in e]
	with np.load(tmp_path / "e" / "branch-0" / "actions.npz") as z:
		assert not z["is_blind"].any() and not z["is_takeover"].any() and z["supervise_action"].all()
