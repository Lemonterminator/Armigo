"""The printed cell markers are taken off the table after the top camera is calibrated (2026-11-25)."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("mujoco")

from digital_twin_armigo.agent.student_rollout import ExpertTracer
from digital_twin_armigo.agent.twin_mujoco import sample_case
from digital_twin_armigo.agent.twin_record import EpisodeTrace
from digital_twin_armigo.inspection_scorer import FixedScorer
from digital_twin_armigo.mujoco_cell._mj import mujoco


class _Scorer(FixedScorer):
	def close(self) -> None:
		pass


def _marker_geoms(cell) -> int:
	m = cell.model
	return sum(
		1
		for g in range(m.ngeom)
		if m.geom_matid[g] >= 0
		and mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, m.geom_matid[g]) in {"m10", "m11", "m16", "m17"}
	)


def _twin(tmp_path, sensing, **kw):
	layout = sample_case(700001, False, 0)
	assert layout.markers, "the sampled layout carries printed cell markers"
	trace = EpisodeTrace(tmp_path / f"ep-{sensing}-{len(kw)}", video=False)
	twin = ExpertTracer(
		layout, _Scorer(lambda _c, _i: -1.0, {"top": 0.0, "wrist_cam": 0.0}), trace, sensing=sensing, **kw
	)
	return layout, trace, twin


@pytest.mark.parametrize("sensing", ["truth", "realistic"])
def test_episode_cell_has_no_cell_markers_but_the_case_layout_keeps_them(tmp_path, sensing):
	layout, trace, twin = _twin(tmp_path, sensing)
	try:
		assert _marker_geoms(twin.cell) == 0
		assert twin.layout.markers == [] and layout.markers  # the caller's layout is not mutated
		assert trace.scene == {"cell_markers": False}
		if sensing == "realistic":
			calib = next(e for e in trace.events if e["kind"] == "camera_calibrated")
			assert calib["cell_markers_removed"] == len(layout.markers)
			assert twin._calib_layout is layout  # the calibration rig still sees the cell as it was
	finally:
		twin.close()
	trace.finish({"verdict": "UNKNOWN"}, {})
	assert json.loads((trace.path / "episode.json").read_text(encoding="utf-8"))["scene"] == {"cell_markers": False}


def test_cell_markers_true_keeps_them_for_old_episodes(tmp_path):
	layout, trace, twin = _twin(tmp_path, "truth", cell_markers=True)
	try:
		assert _marker_geoms(twin.cell) == len(layout.markers) > 0
		assert trace.scene == {"cell_markers": True}
	finally:
		twin.close()


def test_calibration_is_unchanged_by_removing_the_markers(tmp_path):
	_, t1, a = _twin(tmp_path, "realistic")
	_, t2, b = _twin(tmp_path, "realistic", cell_markers=True)
	try:
		ca = next(e for e in t1.events if e["kind"] == "camera_calibrated")
		cb = next(e for e in t2.events if e["kind"] == "camera_calibrated")
		assert {k: v for k, v in ca.items() if k not in ("seq", "wall_time_s", "cell_markers_removed")} == {
			k: v for k, v in cb.items() if k not in ("seq", "wall_time_s", "cell_markers_removed")
		}
		assert (a._cam_est == b._cam_est).all()
	finally:
		a.close()
		b.close()
