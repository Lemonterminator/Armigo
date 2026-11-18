"""Paired "run straight through" vs "save mid-way, restore in a new instance" tests for the MuJoCo session.

Every paired test asserts its own precondition (queue not drained, deadband engaged, noise drawn, brick pinched, ...)
so a passing comparison cannot come from a snapshot taken in a trivially quiet state.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from digital_twin_armigo.constants import FPS, PHASES, REST_U
from digital_twin_armigo.joints import normalized_to_rad
from digital_twin_armigo.mujoco_cell import Layout, Plan, best_roll, ik
from digital_twin_armigo.mujoco_cell.session import MujocoAdapter, MujocoSession, mujoco_scene_spec
from digital_twin_armigo.physics import NonIdealityCfg, load_physics_spec
from digital_twin_armigo.runtime.errors import IdentityMismatchError, SnapshotCorruptError
from digital_twin_armigo.runtime.simulator import (
	ControlProgram,
	SessionState,
	load_session_state,
	rng_state_to_json,
	save_session_state,
)
from digital_twin_armigo.runtime.snapshot import SnapshotStore

PHYSICS = load_physics_spec()
OPEN_G, HOLD_G = 0.85, 0.06
BRICK_XY = (0.26, 0.0)
Q_START = normalized_to_rad(np.array(REST_U, dtype=np.float64))
Q_START[5] = OPEN_G
Q_READY = np.array([0.0, -0.35, 0.25, 1.35, 0.0, OPEN_G])
ADAPTER = MujocoAdapter()


def layout() -> Layout:
	return Layout(green=(0.26, 0.18), pink=(0.26, -0.18), brick_xy=BRICK_XY, brick_yaw_deg=0.0, defective=True)


def scene(**kw) -> dict:
	lay = layout()
	pose = ((BRICK_XY[0], BRICK_XY[1], lay.brick_size[2] / 2.0 + 0.0005), (0.0, 0.0, 0.0, 1.0))
	return mujoco_scene_spec(lay, physics=PHYSICS, q_start=Q_START, brick_pose=pose, **kw)


def program(frames: np.ndarray, phase: str = "approach") -> ControlProgram:
	return ControlProgram(FPS, frames, np.full(len(frames), PHASES.index(phase), dtype=np.uint8))


def _ramp(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
	return a + np.linspace(0.0, 1.0, n + 1)[1:, None] * (b - a)


def push_program() -> ControlProgram:
	"""Move to a ready pose, then sweep the open gripper down through the brick: contact-rich arm/brick motion."""
	low = Q_READY + np.array([0.0, 0.9, 0.2, -0.3, 0.3, -0.9])
	return program(np.vstack([_ramp(Q_START, Q_READY, 20), _ramp(Q_READY, low, 60), np.repeat(low[None], 10, 0)]))


def reversal_program() -> ControlProgram:
	"""Sweep down, then straight back up, so every moving joint reverses direction at frame 50."""
	low = Q_READY + np.array([0.3, 0.5, 0.2, -0.3, 0.3, -0.4])
	return program(np.vstack([_ramp(Q_START, Q_READY, 20), _ramp(Q_READY, low, 30), _ramp(low, Q_READY, 30)]))


def open_with(spec: dict, prog: ControlProgram) -> MujocoSession:
	sess = ADAPTER.open(spec)
	sess.load_program(prog)
	return sess


def persisted(tmp_path, state: SessionState) -> SessionState:
	"""Write ``state`` to a snapshot store and read it back, so only what reached the disk can be restored."""
	store = SnapshotStore(tmp_path / "store")
	return load_session_state(store, save_session_state(store, state))


def assert_same_state(a: SessionState, b: SessionState) -> None:
	assert a.control_step == b.control_step
	assert a.sim_time_s == b.sim_time_s
	assert a.meta == b.meta
	assert a.arrays.keys() == b.arrays.keys()
	for group in a.arrays:
		assert a.arrays[group].keys() == b.arrays[group].keys(), group
		for name, arr in a.arrays[group].items():
			other = b.arrays[group][name]
			assert arr.dtype == other.dtype and arr.shape == other.shape, f"{group}/{name}"
			assert np.array_equal(arr, other), f"{group}/{name} differs after resume"


def paired(tmp_path, spec: dict, prog: ControlProgram, split: int) -> tuple[SessionState, SessionState, SessionState]:
	"""Return (state at the split, final state of the uninterrupted run, final state of the resumed run)."""
	ref = open_with(spec, prog)
	assert ref.step(len(prog)) == len(prog)
	first = open_with(spec, prog)
	first.step(split)
	mid = first.capture()
	first.close()
	resumed = ADAPTER.open(spec)
	resumed.restore(persisted(tmp_path, mid))
	assert resumed.control_step == split
	assert resumed.step(len(prog)) == len(prog) - split
	out = ref.capture(), resumed.capture()
	ref.close()
	resumed.close()
	return mid, *out


def test_capture_does_not_perturb_the_run_and_resume_is_bit_exact(tmp_path):
	prog = push_program()
	ref = open_with(scene(), prog)
	ref.step(len(prog))
	probed = open_with(scene(), prog)
	for _ in range(len(prog) // 6):
		probed.step(6)
		probed.capture()
	assert_same_state(ref.capture(), probed.capture())

	mid, straight, resumed = paired(tmp_path, scene(), prog, split=45)
	tele = straight.arrays["telemetry"]
	assert np.linalg.norm(tele["brick_pos"][-1][:2] - np.array(BRICK_XY)) > 0.005, "the sweep must move the brick"
	assert tele["jaw_forces"].max() > 0.0 or tele["unintended_forces"].max() > 0.0
	assert mid.control_step == 45 and straight.control_step == len(prog)
	assert math.isclose(straight.sim_time_s - mid.sim_time_s, (len(prog) - 45) / FPS, rel_tol=0, abs_tol=1e-9)
	assert_same_state(straight, resumed)


def test_resume_with_a_latency_queue_that_has_not_drained(tmp_path):
	spec = scene(non_ideality=NonIdealityCfg(latency_frames=4))
	mid, straight, resumed = paired(tmp_path, spec, push_program(), split=40)
	queue = mid.arrays["controller"]["queue"]
	assert queue.shape == (4, 6)
	assert len({row.tobytes() for row in queue}) == 4, "the delay line must hold four distinct in-flight commands"
	assert_same_state(straight, resumed)
	# The delayed command stream really is delayed: the logged command lags the target by four frames.
	tele = straight.arrays["telemetry"]
	assert np.array_equal(tele["q_cmd"][44], tele["q_target"][40])


def test_resume_while_the_backlash_deadband_is_engaged_after_a_reversal(tmp_path):
	backlash = math.radians(2.0)
	spec = scene(non_ideality=NonIdealityCfg(backlash_rad=backlash))
	prog = reversal_program()
	split = 52  # two frames after the reversal at frame 50
	mid, straight, resumed = paired(tmp_path, spec, prog, split)
	held_pos = mid.arrays["controller"]["backlash_pos"]
	gap = np.abs(prog.q_targets[split - 1] - held_pos)
	moving = np.abs(prog.q_targets[split - 1] - prog.q_targets[split - 2]) > 1e-6
	assert moving.any()
	assert np.all(gap[moving] < 0.5 * backlash), "moving joints must still be inside the deadband at the split"
	assert np.all(gap[moving] > 0.0), "the output must be lagging the command, not sitting on it"
	assert_same_state(straight, resumed)


def test_resume_while_the_velocity_limiter_is_still_catching_up(tmp_path):
	spec = scene(non_ideality=NonIdealityCfg(velocity_limit_rad_s=0.5))
	low = Q_READY + np.array([0.0, 0.9, 0.2, -0.3, 0.3, -0.9])
	prog = program(np.vstack([_ramp(Q_START, Q_READY, 20), np.repeat(low[None], 50, 0)]))  # a step command
	split = 30
	mid, straight, resumed = paired(tmp_path, spec, prog, split)
	lag = np.abs(mid.arrays["controller"]["prev_out"] - prog.q_targets[split - 1])
	assert lag.max() > 0.2, "the limited output must still be far from the commanded step at the split"
	tele = straight.arrays["telemetry"]
	step_per_frame = np.abs(np.diff(tele["q_cmd"], axis=0)).max()
	assert step_per_frame <= 0.5 / FPS + 1e-12
	assert_same_state(straight, resumed)


def test_resume_after_observation_noise_has_been_drawn_and_seed_alone_is_not_enough(tmp_path):
	spec = scene(non_ideality=NonIdealityCfg(joint_noise_std_rad=0.01, encoder_ticks=4096), seed=7)
	prog = push_program()
	mid, straight, resumed = paired(tmp_path, spec, prog, split=45)
	tele = straight.arrays["telemetry"]
	assert not np.array_equal(tele["q_obs"], tele["qpos"]), "observations must actually be noisy"
	assert mid.meta["rng"]["observation"] != rng_state_to_json(np.random.default_rng(7)), "noise must be consumed"
	assert_same_state(straight, resumed)

	# Negative control: restoring the stream from its seed instead of its state replays already-used draws.
	reseeded_state = copy.deepcopy(persisted(tmp_path, mid))
	reseeded_state.meta["rng"]["observation"] = rng_state_to_json(np.random.default_rng(7))
	reseeded = ADAPTER.open(spec)
	reseeded.restore(reseeded_state)
	reseeded.step(len(prog))
	bad = reseeded.capture().arrays["telemetry"]
	assert np.array_equal(bad["qpos"], tele["qpos"]), "physics does not depend on observation noise"
	assert not np.array_equal(bad["q_obs"][45:], tele["q_obs"][45:])


@pytest.fixture(scope="module")
def pick() -> dict:
	"""A pure-physics pick, lift and mid-air release, with the frame indices the tests split at."""
	spec = scene()
	sess = ADAPTER.open(spec)
	cell = sess.cell
	size = cell.layout.brick_size
	short_axis = np.array([0.0, 1.0])
	q_hover = ik(cell, np.array([*BRICK_XY, 0.09]), Q_START.copy(), 0.0)
	roll, sign = best_roll(cell, q_hover, short_axis)
	gx, gy = np.array(BRICK_XY) - sign * short_axis * 0.006
	del size
	plan = Plan(Q_START, 1.0)
	plan.move(Q_READY, 0.6, "approach")
	q = Q_READY
	for z, g, secs, phase in (
		(0.09, OPEN_G, 1.0, "approach"),
		(0.026, OPEN_G, 0.8, "approach"),
		(0.006, OPEN_G, 0.5, "grasp"),
		(0.006, HOLD_G, 0.8, "grasp"),
		(0.12, HOLD_G, 1.0, "lift"),
	):
		q = ik(cell, np.array([gx, gy, z]), q, roll)
		q[5] = g
		plan.move(q, secs, phase)
	lifted = len(plan.frames)
	q_open = q.copy()
	q_open[5] = OPEN_G
	plan.move(q_open, 0.3, "place", settle=0.0)
	opened = len(plan.frames)
	plan.hold(0.7, "place")
	sess.close()
	prog = ControlProgram(FPS, np.array(plan.frames), np.array(plan.phases, dtype=np.uint8))
	return {"spec": spec, "program": prog, "lifted": lifted, "opened": opened}


def test_resume_while_the_brick_is_pinched_by_friction_only(tmp_path, pick):
	split = pick["lifted"] - 15  # mid-lift, brick carried by the jaws
	_, straight, resumed = paired(tmp_path, pick["spec"], pick["program"], split)
	tele = straight.arrays["telemetry"]
	assert pick["spec"]["assisted_grasp"] is False
	assert tele["jaw_forces"][split - 1].min() > 1.0, "both jaws must be loading the brick at the split"
	assert tele["brick_pos"][split - 1][2] > 0.03, "the brick must be off the table at the split"
	assert tele["brick_pos"][pick["lifted"] - 1][2] > 0.10, "the grasp must survive the lift"
	assert_same_state(straight, resumed)


def test_resume_after_release_while_the_brick_is_falling(tmp_path, pick):
	tele_ref = open_with(pick["spec"], pick["program"])
	tele_ref.step(len(pick["program"]))
	speed = np.linalg.norm(tele_ref.telemetry()["brick_vel"], axis=1)
	jaws = tele_ref.telemetry()["jaw_forces"].max(axis=1)
	falling = np.flatnonzero((np.arange(len(speed)) >= pick["lifted"]) & (jaws == 0.0) & (speed > 0.3))
	tele_ref.close()
	assert len(falling), "the released brick must be in free fall for at least one frame"
	split = int(falling[0]) + 1
	_, straight, resumed = paired(tmp_path, pick["spec"], pick["program"], split)
	assert straight.arrays["telemetry"]["brick_pos"][-1][2] < 0.03, "the brick must have landed by the end"
	assert_same_state(straight, resumed)


@pytest.fixture(scope="module")
def assisted_pick(pick) -> dict:
	"""The same pick in assisted-grasp compatibility mode (kinematic attachment, capped release speed)."""
	spec = scene(assisted_grasp=True, physics_hz=FPS * 16)
	frames = pick["program"].q_targets.copy()
	frames[frames[:, 5] <= HOLD_G + 1e-9, 5] = -0.17  # the assisted grasp triggers on a negative gripper command
	prog = ControlProgram(FPS, frames, pick["program"].phases)
	sess = open_with(spec, prog)
	held, settle = [], []
	for _ in range(len(prog)):
		sess.step(1)
		held.append(sess.cell.held is not None)
		settle.append(sess.cell.settle_frames)
	sess.close()
	return {"spec": spec, "program": prog, "held": np.array(held), "settle": np.array(settle)}


def test_assisted_mode_resume_while_attached_and_inside_the_release_settle_window(tmp_path, assisted_pick):
	held, settle = assisted_pick["held"], assisted_pick["settle"]
	assert held.any(), "the assisted grasp must attach the brick"
	attached = int(np.flatnonzero(held)[len(np.flatnonzero(held)) // 2]) + 1
	mid, straight, resumed = paired(tmp_path, assisted_pick["spec"], assisted_pick["program"], attached)
	assert mid.meta["cell"]["held"] is not None
	assert mid.meta["physics_mode"] == "assisted_compat"
	assert_same_state(straight, resumed)

	in_window = np.flatnonzero((settle > 2) & (settle < 13) & ~held)
	assert len(in_window), "the release speed cap must be active for several frames"
	split = int(in_window[0]) + 1
	mid, straight, resumed = paired(tmp_path, assisted_pick["spec"], assisted_pick["program"], split)
	assert 2 < mid.meta["cell"]["settle_frames"] < 13
	assert mid.meta["cell"]["held"] is None
	assert_same_state(straight, resumed)


def test_two_branches_from_one_snapshot_do_not_contaminate_each_other(tmp_path):
	prog = push_program()
	root_sess = open_with(scene(), prog)
	root_sess.step(30)
	root = persisted(tmp_path, root_sess.capture())
	root_copy = copy.deepcopy(root)
	root_sess.close()

	tail_a = prog.q_targets[30:]
	tail_b = tail_a + np.array([0.25, 0.0, 0.0, 0.0, 0.0, 0.0])  # a different candidate: pan while sweeping
	finals = {}
	for name, tail in (("a", tail_a), ("b", tail_b)):
		branch = ADAPTER.open(scene())
		branch.restore(root)
		branch.load_program(program(tail))
		branch.step(len(tail))
		finals[name] = branch.capture()
		branch.close()
	assert_same_state(root, root_copy)  # restoring and running branches never writes into the parent state
	assert finals["a"].control_step == finals["b"].control_step == len(prog)
	assert not np.array_equal(finals["a"].arrays["physics"]["state"], finals["b"].arrays["physics"]["state"])

	straight = open_with(scene(), prog)
	straight.step(len(prog))
	ref = straight.capture()
	assert np.array_equal(ref.arrays["physics"]["state"], finals["a"].arrays["physics"]["state"])
	for key in ("qpos", "brick_pos", "q_cmd", "jaw_forces"):
		assert np.array_equal(ref.arrays["telemetry"][key], finals["a"].arrays["telemetry"][key]), key
	# Branch b keeps the shared prefix as evidence and then diverges.
	assert np.array_equal(finals["b"].arrays["telemetry"]["qpos"][:30], ref.arrays["telemetry"]["qpos"][:30])
	assert not np.array_equal(finals["b"].arrays["telemetry"]["qpos"][30:], ref.arrays["telemetry"]["qpos"][30:])


def _mid_state(tmp_path) -> SessionState:
	sess = open_with(scene(), push_program())
	sess.step(20)
	state = persisted(tmp_path, sess.capture())
	sess.close()
	return state


def test_restore_refuses_a_model_that_differs_from_the_snapshot(tmp_path):
	state = _mid_state(tmp_path)
	heavier = ADAPTER.open(scene(mass_scale=1.5))
	with pytest.raises(IdentityMismatchError, match="model_fingerprint"):
		heavier.restore(state)

	mutated = ADAPTER.open(scene())
	mutated.cell.model.opt.timestep *= 0.5  # a parameter changed in memory after the model was built
	with pytest.raises(IdentityMismatchError, match="model_fingerprint"):
		mutated.restore(state)

	other_engine = copy.deepcopy(state)
	other_engine.meta["engine_version"] = "0.0.0"
	with pytest.raises(IdentityMismatchError, match="engine"):
		ADAPTER.open(scene()).restore(other_engine)


@pytest.mark.parametrize(
	"damage",
	["nan_physics", "short_physics", "bad_schema", "bad_rng", "missing_group", "cursor_past_end", "ragged_telemetry"],
)
def test_restore_refuses_damaged_or_inconsistent_state_and_leaves_the_session_untouched(tmp_path, damage):
	state = copy.deepcopy(_mid_state(tmp_path))
	if damage == "nan_physics":
		state.arrays["physics"]["state"][3] = np.nan
	elif damage == "short_physics":
		state.arrays["physics"]["state"] = state.arrays["physics"]["state"][:-1]
	elif damage == "bad_schema":
		state.meta["schema"] = "session-state/v0"
	elif damage == "bad_rng":
		state.meta["rng"]["observation"]["bit_generator"] = "MT19937"
	elif damage == "missing_group":
		del state.arrays["controller"]
	elif damage == "cursor_past_end":
		state.meta["program_cursor"] = 10_000
	elif damage == "ragged_telemetry":
		state.arrays["telemetry"]["qpos"] = state.arrays["telemetry"]["qpos"][:-1]
	sess = open_with(scene(), push_program())
	sess.step(5)
	before = sess.capture()
	with pytest.raises(SnapshotCorruptError):
		sess.restore(state)
	assert_same_state(before, sess.capture())


def test_scene_spec_is_json_and_rebuilds_the_same_model():
	import json

	spec = scene(non_ideality=NonIdealityCfg(latency_frames=2, backlash_rad=0.01), mass_scale=1.2, seed=3)
	again = json.loads(json.dumps(spec))
	assert again == spec
	a, b = ADAPTER.open(spec), ADAPTER.open(again)
	assert a.model_fingerprint() == b.model_fingerprint()
	assert a.model_fingerprint() != ADAPTER.open(scene()).model_fingerprint()
	assert a.engine == "mujoco" and a.engine_version == mujoco.__version__


def test_pure_physics_scene_rejects_substep_rates_the_assisted_step_cannot_honour():
	with pytest.raises(ValueError):
		scene(assisted_grasp=True, physics_hz=240)
	with pytest.raises(ValueError):
		scene(physics_hz=250)


# -- the execution service on the real engine ---------------------------------------------------


def _service(tmp_path):
	from digital_twin_armigo.runtime.ledger import OperationLedger
	from digital_twin_armigo.runtime.service import ExecutionService
	from digital_twin_armigo.runtime.snapshot import current_identity

	identity = current_identity("mujoco", mujoco.__version__)
	ledger = OperationLedger(tmp_path / "ledger.sqlite")
	return ExecutionService(SnapshotStore(tmp_path / "store"), ledger, ADAPTER, identity)


def test_service_resumes_a_mujoco_rollout_across_restart_and_crash_and_the_evaluator_agrees(tmp_path):
	from digital_twin_armigo.constants import BOWL_R
	from digital_twin_armigo.metrics import EVALUATOR_VERSION, evaluate_episode_health
	from digital_twin_armigo.runtime.ledger import JobState
	from digital_twin_armigo.runtime.service import FaultHooks, RolloutWorker

	class Crash(BaseException):
		pass

	def crash() -> None:
		raise Crash

	cfg = NonIdealityCfg(
		latency_frames=3, backlash_rad=math.radians(1.0), joint_noise_std_rad=0.005, encoder_ticks=4096
	)
	spec = scene(non_ideality=cfg, seed=5)
	prog = push_program()
	contract = {"task": "sort_place", "evaluator_version": EVALUATOR_VERSION}

	svc = _service(tmp_path)
	root = svc.create_root_snapshot(
		idempotency_key="root", run_id="run", episode_id="ep0", branch_id="main", scene_spec=spec, contract=contract
	)
	first = svc.submit_rollout(idempotency_key="a", snapshot_id=root.snapshot_id, program=prog, max_frames=37)
	mid = RolloutWorker(svc, "w1").run(first)
	assert mid.result["physics_mode"] == "pure_physics" and mid.result["frames_executed"] == 37
	second = svc.submit_rollout(idempotency_key="b", snapshot_id=mid.output_snapshot_id)
	with pytest.raises(Crash):  # snapshot committed, ledger not yet updated
		RolloutWorker(svc, "w1").run(second, hooks=FaultHooks(after_commit=crash))
	svc.ledger.close()

	restarted = _service(tmp_path)
	assert restarted.recover().adopted == [second]
	done = restarted.get_rollout(second)
	assert done.state is JobState.SUCCEEDED and done.attempts == 1
	final = restarted.load_state(done.output_snapshot_id)
	reference = open_with(spec, prog)
	reference.step(len(prog))
	assert_same_state(reference.capture(), final)

	def verdict(tele: dict) -> dict:
		return evaluate_episode_health(
			q_cmd=tele["q_cmd"],
			q_act=tele["qpos"],
			tcp_pos=tele["tcp_pos"],
			brick_pos=tele["brick_pos"],
			brick_quat=tele["brick_quat"],
			jaw_forces=tele["jaw_forces"],
			unintended_forces=tele["unintended_forces"],
			phases=tele["phases"],
			target_bowl_xy=(0.26, -0.18),
			bowl_r=BOWL_R,
			brick_vel=tele["brick_vel"],
			brick_ang_vel=tele["brick_ang_vel"],
			ee_quat=tele["ee_quat"],
			fps=FPS,
			brick_size=layout().brick_size,
			substep_unintended_forces=tele["substep_unintended"],
			pinch_normal_forces=tele["pinch_normal"],
			pinch_tangential_forces=tele["pinch_tangential"],
			actuator_torques=tele["actuator_torques"],
			torque_source="solver_measured",
			timestamps=tele["t"],
		).to_dict()

	from_snapshot = restarted.store.get_arrays(
		type(root.blobs["arrays/telemetry"]).from_dict(done.result["telemetry_blob"])
	)
	assert verdict(from_snapshot) == verdict(reference.telemetry())
	assert verdict(from_snapshot)["evaluator_version"] == contract["evaluator_version"]
	reference.close()
	restarted.ledger.close()


def test_service_refuses_a_mujoco_snapshot_after_the_scene_model_changes(tmp_path):
	from digital_twin_armigo.runtime.service import RolloutWorker

	svc = _service(tmp_path)
	root = svc.create_root_snapshot(
		idempotency_key="root", run_id="run", episode_id="ep0", branch_id="main", scene_spec=scene(), contract={}
	)
	job = svc.submit_rollout(idempotency_key="a", snapshot_id=root.snapshot_id, program=push_program())

	class DriftingAdapter(MujocoAdapter):
		def open(self, scene_spec):
			session = super().open(scene_spec)
			session.cell.model.geom_friction[:, 0] *= 0.9  # e.g. a code change in how friction is applied
			return session

	svc.adapter = DriftingAdapter()
	status = RolloutWorker(svc, "w1").run(job)
	assert status.state.value == "FAILED" and status.error["reason"] == "identity_mismatch"
	assert "model_fingerprint" in status.error["detail"]
	svc.ledger.close()
