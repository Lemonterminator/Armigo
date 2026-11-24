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
