import numpy as np

from digital_twin_armigo.agent.grasp_loop import VARIANTS, GraspLoop


class FakeExpert:
	"""Records commanded targets; _move yields `round(secs * 30)` frames ending at the goal."""

	class P:
		hover_z_m, pregrasp_dz_m, descend_s = 0.06, 0.02, 0.5

	p, z_grasp, _active_open_g = P(), 0.01, 0.9

	def __init__(self):
		self.last = np.zeros(6)
		self.frames = []

	def _ik(self, pos, q0, roll):
		q = np.zeros(6)
		q[:3] = pos
		return q

	def _move(self, q, secs, phase):
		for _ in range(max(1, round(secs * 30))):
			self.last = q.copy()
			self.frames.append(q[:3].copy())
			yield q.copy(), phase

	def _hold_servo(self, q, secs, phase):
		yield from self._move(q, secs, phase)


def _run(gen):
	try:
		while True:
			next(gen)
	except StopIteration as stop:
		return stop.value


def test_variants_table():
	assert VARIANTS["open"] is None
	assert VARIANTS["w2s"] == (("hover", 0.035), "sync")
	assert VARIANTS["w4a"][1] == "async"


def test_sync_looks_at_each_checkpoint_and_replans():
	ex, looks = FakeExpert(), []
	new_target = np.array([0.31, 0.0, 0.01])

	def look(h):
		looks.append((round(h, 4), ex.last[:3].copy()))
		return (new_target, 0.0, np.zeros(6)) if len(looks) == 1 else None

	loop = GraspLoop(("hover", 0.035), "sync", look, fps=30)
	target, _roll, _q = _run(loop.descend(ex, "approach", np.array([0.30, 0.0, 0.01]), 0.0, np.zeros(6)))
	assert [h for h, _ in looks] == [0.05, 0.035]
	assert np.allclose(looks[1][1][:2], new_target[:2])  # the second look is above the replanned target
	assert np.allclose(target, new_target)


def test_async_applies_the_result_after_the_latency():
	ex, looks = FakeExpert(), []
	new_target = np.array([0.31, 0.0, 0.01])

	def look(h):
		looks.append(len(ex.frames))
		return (new_target, 0.0, np.zeros(6)) if len(looks) == 1 else None

	loop = GraspLoop(("hover", 0.035), "async", look, latency_s=0.1, segment_s=0.5, fps=30)
	target, _, _ = _run(loop.descend(ex, "approach", np.array([0.30, 0.0, 0.01]), 0.0, np.zeros(6)))
	first = looks[0]
	# the next 3 frames still head for the old target, then the arm turns to the new one
	assert np.allclose(ex.frames[first + 2][:2], [0.30, 0.0])
	assert np.allclose(ex.frames[-1][:2], new_target[:2]) and np.allclose(target, new_target)
	assert len(ex.frames) < 30 * (0.5 + 0.15) * 3  # no settle holds


def test_recheck_none_is_default_and_matches_open_loop(tmp_path):
	import pytest

	pytest.importorskip("mujoco")
	from digital_twin_armigo.agent.twin_mujoco import TwinMujoco, sample_case
	from digital_twin_armigo.agent.twin_record import EpisodeTrace

	class _StubScorer:
		def score(self, *a, **kw):
			return {}

	def _run_pick(subdir: str, *, set_none: bool):
		trace = EpisodeTrace(tmp_path / subdir, video=False)
		twin = TwinMujoco(sample_case(600001, False, 5), _StubScorer(), trace, sensing="truth")
		try:
			assert twin.expert.recheck is None
			if set_none:
				twin.expert.recheck = None
			assert twin.prepare() is None
			assert twin._grasp() is None
			return twin.ep.telemetry()["qpos"].copy()
		finally:
			trace.close()
			twin.close()

	assert np.array_equal(_run_pick("default", set_none=False), _run_pick("explicit_none", set_none=True))


def test_motion_open_default_matches_via_compare_runs(tmp_path):
	import pytest

	pytest.importorskip("mujoco")
	pytest.importorskip("cv2")
	from digital_twin_armigo.agent.twin_mujoco import TwinMujoco, sample_case
	from digital_twin_armigo.agent.twin_record import EpisodeTrace
	from digital_twin_armigo.experiments.diag_mcp import compare_runs
	from digital_twin_armigo.inspection_scorer import FixedScorer

	def _run_ep(name: str, **kw):
		ep_dir = tmp_path / name
		trace = EpisodeTrace(ep_dir, video=False)
		scorer = FixedScorer(lambda _c, _rgb: -1.0, {"top": 0.0, "wrist_cam": 0.0})
		twin = TwinMujoco(sample_case(600001, False, 5), scorer, trace, sensing="realistic", **kw)
		try:
			assert twin.motion == "open"
			assert twin.prepare() is None
			assert twin._grasp() is None
			assert twin.place("normal") is None
			res = twin.evaluate("normal", [], [])
			trace.finish(res, {})
			return ep_dir
		finally:
			trace.close()
			twin.close()

	a = _run_ep("default")
	b = _run_ep("explicit_open", motion="open")
	cmp_out = compare_runs(a, b)
	assert cmp_out["identical_discrete"] and cmp_out["score_deltas"] == [] and cmp_out["verdict_equal"]


def test_closed_loop_w2s_emits_wrist_fix_and_rejects_truth_mode(tmp_path):
	import pytest

	pytest.importorskip("mujoco")
	pytest.importorskip("cv2")
	from digital_twin_armigo.agent.twin_mujoco import TwinMujoco, sample_case
	from digital_twin_armigo.agent.twin_record import EpisodeTrace
	from digital_twin_armigo.inspection_scorer import FixedScorer

	scorer = FixedScorer(lambda _c, _rgb: -1.0, {"top": 0.0, "wrist_cam": 0.0})
	trace_t = EpisodeTrace(tmp_path / "truth", video=False)
	try:
		with pytest.raises(ValueError, match="realistic sensing"):
			TwinMujoco(sample_case(600001, False, 5), scorer, trace_t, sensing="truth", motion="w2s")
	finally:
		trace_t.close()

	trace = EpisodeTrace(tmp_path / "w2s", video=False)
	twin = TwinMujoco(sample_case(600001, False, 5), scorer, trace, sensing="realistic", motion="w2s")
	try:
		assert twin.prepare() is None
		assert twin._grasp() is None
		fixes = [e for e in trace.events if e["kind"] == "wrist_fix"]
		assert len(fixes) >= 1
		assert all(f["mode"] == "sync" and "error_xy_mm" in f and "error_yaw_deg" in f for f in fixes)
	finally:
		trace.close()
		twin.close()


def test_step_mcp_last_wrist_fix_strips_truth_keys(tmp_path):
	import pytest

	pytest.importorskip("mujoco")
	pytest.importorskip("cv2")
	from digital_twin_armigo.agent import step_mcp as sm
	from digital_twin_armigo.agent.twin_mujoco import ROOT
	from digital_twin_armigo.inspection_scorer import FixedScorer

	det = ROOT / "results/agent-bench/sup-v7bot-local.json"
	if not det.is_file():
		pytest.skip("detector spec not present")
	scorer = FixedScorer(lambda _c, _rgb: -1.0, {"top": 0.0, "wrist_cam": 0.0})
	svc = sm.StepService(tmp_path / "svc", det, "realistic", [(600001, False, 5)], scorer=scorer)
	try:

		def _check():
			started = svc.start(next(iter(svc.cases)))
			sess = svc.session(started["session_id"])
			assert sess.last_wrist_fix() is None
			sess.backend.event(
				"wrist_fix",
				height_m=0.05,
				found=True,
				accepted=True,
				mode="sync",
				error_xy_mm=1.23,
				error_yaw_deg=0.45,
			)
			fix = sess.last_wrist_fix()
			assert fix is not None
			assert fix["height_m"] == 0.05 and fix["found"] is True
			assert "error_xy_mm" not in fix and "error_yaw_deg" not in fix
			sm.assert_sensor_only(fix)

		svc.run_sync(_check)
	finally:
		svc.close()
