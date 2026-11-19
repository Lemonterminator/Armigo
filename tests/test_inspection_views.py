"""Face-probe visibility, the pixel oracle inspector, and the closed-loop expert's inspection decision point."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from digital_twin_armigo.inspection import assert_no_defect_leakage

ROOT = Path(__file__).resolve().parents[1]


def _view(faces=(0, 0, 0, 0, 0, 0), defect=0, brick=100, full=100):
	from digital_twin_armigo.mujoco_cell.inspection_views import InspectionView

	return InspectionView("top", tuple(faces), defect, brick, full, (0, 0, 10, 10))


# --- pure: the pixel oracle --------------------------------------------------------------------------------------


def test_oracle_defective_known_only_once_defect_pixels_seen():
	pytest.importorskip("mujoco")
	from digital_twin_armigo.mujoco_cell.inspection_views import PixelOracleInspector

	insp = PixelOracleInspector(True, min_defect_px=25)
	assert insp.observe(_view(faces=(100, 0, 0, 0, 100, 0), defect=10)).status == "UNKNOWN"
	res = insp.observe(_view(faces=(0, 0, 100, 0, 0, 0), defect=30))
	assert (res.status, res.label) == ("KNOWN", "defective")


def test_oracle_normal_needs_every_eligible_face():
	pytest.importorskip("mujoco")
	from digital_twin_armigo.mujoco_cell.inspection_views import PixelOracleInspector

	insp = PixelOracleInspector(False, min_face_px=60, eligible_faces=(0, 1, 4))
	assert insp.observe(_view(faces=(100, 0, 0, 0, 100, 0))).status == "UNKNOWN"
	assert insp.observe(_view(faces=(0, 59, 0, 0, 0, 0))).status == "UNKNOWN"  # below the pixel threshold
	res = insp.observe(_view(faces=(0, 60, 0, 0, 0, 0)))
	assert (res.status, res.label) == ("KNOWN", "normal")
	assert res.observed_faces == (0, 1, 4)


def test_oracle_unknown_reason_does_not_depend_on_ground_truth():
	pytest.importorskip("mujoco")
	from digital_twin_armigo.mujoco_cell.inspection_views import PixelOracleInspector

	v = _view(faces=(100, 0, 0, 0, 100, 0), defect=0)
	a = PixelOracleInspector(True).observe(v)
	b = PixelOracleInspector(False).observe(v)
	assert a.status == b.status == "UNKNOWN"
	assert a.reason == b.reason
	assert_no_defect_leakage({"reason": a.reason, "observed": list(a.observed_faces)})


def test_square_crop_centres_and_pads():
	pytest.importorskip("mujoco")
	from digital_twin_armigo.mujoco_cell.inspection_views import square_crop

	img = np.zeros((100, 200, 3), np.uint8)
	img[40:60, 90:110] = 255
	out = square_crop(img, (90, 40, 20, 20), pad=0.5, size=40)
	assert out.shape == (40, 40, 3)
	assert out[20, 20].tolist() == [255, 255, 255]  # brick at the centre
	assert out[0, 0].tolist() == [0, 0, 0]  # padding around it
	edge = square_crop(img, (0, 0, 20, 20), pad=0.5, size=40)  # partly outside the image: black, no error
	assert edge.shape == (40, 40, 3)


# --- MuJoCo -----------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bench():
	pytest.importorskip("mujoco")
	sys.path.insert(0, str(ROOT / "scripts"))
	import bench_physics_expert

	return bench_physics_expert


def _layout(bench, idx=1, face=4, defective=True):
	from digital_twin_armigo.camera import CameraPose

	top = CameraPose(**json.loads((ROOT / "config" / "sim" / "top_camera_joint.json").read_text())["camera"])
	lay = bench.sample_layouts(idx + 1, 7, top)[idx]
	lay.defective = defective
	lay.defect_face = face
	return lay


def test_face_probes_count_only_faces_in_view(bench):
	from digital_twin_armigo.constants import SO101_MJCF_DIR, TOP_CROP
	from digital_twin_armigo.mujoco_cell import Cell
	from digital_twin_armigo.mujoco_cell._mj import mujoco
	from digital_twin_armigo.mujoco_cell.inspection_views import capture_view
	from digital_twin_armigo.physics import load_physics_spec

	lay = _layout(bench, face=4, defective=True)
	with Cell(lay, SO101_MJCF_DIR, physics=load_physics_spec(), assisted_grasp=False, face_probes=True) as cell:
		mujoco.mj_forward(cell.model, cell.data)
		v = capture_view(cell, "top", crop=TOP_CROP, image_size=64)
	assert v.faces_px[4] > 100  # the top face of a brick on the table faces the top camera
	assert v.faces_px[5] < 60  # the bottom face lies on the table: below the observed-face threshold
	assert v.defect_px > 25  # the cross is on the top face
	assert v.image is not None and v.image.shape == (64, 64, 3)
	_, _, w, h = v.bbox
	assert w < 80 and h < 80  # a 3 cm brick, not stretched by stray (reflection) pixels


def test_crop_truth_counts_only_what_the_detector_crop_shows(bench):
	from digital_twin_armigo.constants import SO101_MJCF_DIR
	from digital_twin_armigo.mujoco_cell import Cell
	from digital_twin_armigo.mujoco_cell._mj import mujoco
	from digital_twin_armigo.mujoco_cell.inspection_views import crop_counts, detector_crop
	from digital_twin_armigo.physics import load_physics_spec

	lay = _layout(bench, face=4, defective=True)
	with Cell(lay, SO101_MJCF_DIR, physics=load_physics_spec(), assisted_grasp=False, face_probes=True) as cell:
		mujoco.mj_forward(cell.model, cell.data)
		full = cell.face_visibility("top")
		face, defect = cell.visibility_labels("top")
		image, bbox = detector_crop(cell, "top", None, cell.data, 64, clamp=True)
	# The labels are exactly what face_visibility counts over the frame.
	assert np.bincount(face[face >= 0], minlength=6).tolist() == full["faces"].tolist()
	assert int(defect.sum()) == full["defect"] > 25
	real = crop_counts(face, defect, bbox, 64, clamp=True)
	assert real["cross_native"] == full["defect"]  # the whole cross lies inside the crop around the brick
	assert real["cross_input"] > 0 and image.shape == (64, 64, 3)
	one = crop_counts(face, defect, (0, 0, 1, 1), 64, clamp=True)  # a 1x1 box at the top-left: no cross
	assert one["cross_native"] == one["cross_input"] == 0


def test_face_probes_do_not_change_the_physics(bench):
	from digital_twin_armigo.benchmarks import SENSITIVITY_MATRIX
	from digital_twin_armigo.constants import SO101_MJCF_DIR
	from digital_twin_armigo.mujoco_cell import Cell
	from digital_twin_armigo.mujoco_cell.physics_expert import ExpertParams, run_closed_loop_episode
	from digital_twin_armigo.physics import load_physics_spec

	spec = load_physics_spec()
	base = {c.name: c for c in SENSITIVITY_MATRIX}["baseline_2x2"]
	lay = _layout(bench)
	params = ExpertParams(max_frames=150)  # the approach and the start of the grasp
	runs = []
	for probes in (False, True):
		with Cell(
			lay,
			SO101_MJCF_DIR,
			physics=spec,
			dynamic_clutter=True,
			assisted_grasp=False,
			render_enabled=False,
			face_probes=probes,
		) as cell:
			tel, _, info = run_closed_loop_episode(
				cell,
				params,
				q_start=bench._q_start(),
				bowl_xy=lay.green,
				brick_xy=lay.brick_xy,
				brick_yaw_deg=lay.brick_yaw_deg,
				non_ideal=bench.bench._mj_non_ideal(base, spec),
			)
		runs.append((info["_targets"], tel["brick_pos"]))
	assert np.array_equal(runs[0][0], runs[1][0])
	assert np.array_equal(runs[0][1], runs[1][1])


def test_expert_needs_exactly_one_of_bowl_or_inspector(bench):
	from digital_twin_armigo.constants import SO101_MJCF_DIR
	from digital_twin_armigo.mujoco_cell import Cell
	from digital_twin_armigo.mujoco_cell.physics_expert import ClosedLoopExpert, ExpertParams

	lay = _layout(bench)
	with Cell(lay, SO101_MJCF_DIR, assisted_grasp=False, render_enabled=False) as cell:
		q = bench._q_start()
		with pytest.raises(ValueError):
			ClosedLoopExpert(cell, q, None, ExpertParams())
		with pytest.raises(ValueError):
			ClosedLoopExpert(cell, q, lay.green, ExpertParams(), inspect=lambda *a: None, bowls={"normal": lay.green})
		with pytest.raises(ValueError):
			ClosedLoopExpert(cell, q, None, ExpertParams(), inspect=lambda *a: None)


@pytest.mark.parametrize("label", ["defective", "normal"])
def test_inspection_label_picks_the_bowl(bench, label):
	"""A scripted inspector that answers KNOWN on its second call: the expert stops showing there and places into
	the bowl the label names, whatever the ground truth."""
	from digital_twin_armigo.benchmarks import SENSITIVITY_MATRIX
	from digital_twin_armigo.constants import SO101_MJCF_DIR
	from digital_twin_armigo.inspection import InspectionResult
	from digital_twin_armigo.mujoco_cell import Cell
	from digital_twin_armigo.mujoco_cell.physics_expert import ExpertParams, run_closed_loop_episode
	from digital_twin_armigo.physics import load_physics_spec

	spec = load_physics_spec()
	base = {c.name: c for c in SENSITIVITY_MATRIX}["baseline_2x2"]
	lay = _layout(bench, defective=False)
	calls = []

	def inspect(flex, roll_deg, frame):
		calls.append((flex, roll_deg, frame))
		if len(calls) < 2:
			return InspectionResult("UNKNOWN", None, "test", True, (), (), "undecided")
		return InspectionResult("KNOWN", label, "test", True, (), (), "scripted")

	bowls = {"normal": lay.green, "defective": lay.pink}
	with Cell(lay, SO101_MJCF_DIR, physics=spec, dynamic_clutter=True, assisted_grasp=False, render_enabled=False) as c:
		tel, phases, info = run_closed_loop_episode(
			c,
			ExpertParams(),
			q_start=bench._q_start(),
			brick_xy=lay.brick_xy,
			brick_yaw_deg=lay.brick_yaw_deg,
			non_ideal=bench.bench._mj_non_ideal(base, spec),
			bowls=bowls,
			inspect=inspect,
		)
	assert len(calls) == 2  # stop_when_known: the third show pose is skipped
	assert info["inspection_label"] == label
	assert info["target_bowl_xy"] == pytest.approx(list(bowls[label]))
	assert [r["status"] for r in info["inspections"]] == ["UNKNOWN", "KNOWN"]
	health = bench.bench._evaluate_mj_rollout(tel, phases, spec, tuple(lay.brick_size), bowls[label])
	assert health["verdict"] == "SUCCESS", health["reason_codes"]


def test_inspection_unknown_aborts_without_placing(bench):
	from digital_twin_armigo.benchmarks import SENSITIVITY_MATRIX
	from digital_twin_armigo.constants import PHASES, SO101_MJCF_DIR
	from digital_twin_armigo.inspection import InspectionResult
	from digital_twin_armigo.mujoco_cell import Cell
	from digital_twin_armigo.mujoco_cell.physics_expert import ExpertParams, run_closed_loop_episode
	from digital_twin_armigo.physics import load_physics_spec

	spec = load_physics_spec()
	base = {c.name: c for c in SENSITIVITY_MATRIX}["baseline_2x2"]
	lay = _layout(bench)
	params = ExpertParams(extra_show_poses=((0.9, 90.0),))
	with Cell(lay, SO101_MJCF_DIR, physics=spec, dynamic_clutter=True, assisted_grasp=False, render_enabled=False) as c:
		_, phases, info = run_closed_loop_episode(
			c,
			params,
			q_start=bench._q_start(),
			brick_xy=lay.brick_xy,
			brick_yaw_deg=lay.brick_yaw_deg,
			non_ideal=bench.bench._mj_non_ideal(base, spec),
			bowls={"normal": lay.green, "defective": lay.pink},
			inspect=lambda *a: InspectionResult("UNKNOWN", None, "test", True, (), (), "undecided"),
		)
	assert info["aborted"] == "inspection_unknown"
	assert len(info["inspections"]) == 4  # three base poses and the extra one
	assert info["inspections"][-1]["flex"] == pytest.approx(0.9)
	assert info["target_bowl_xy"] is None
	assert PHASES.index("place") not in set(phases.tolist())
	assert PHASES.index("handoff") not in set(phases.tolist())


def test_next_pose_policy_sees_history_and_is_capped(bench):
	"""While UNKNOWN after the base poses the expert asks the policy for poses, at most ``max_extra_shows`` times,
	passing every inspection so far; a pose outside the wrist roll range is clipped to it."""
	from digital_twin_armigo.benchmarks import SENSITIVITY_MATRIX
	from digital_twin_armigo.constants import SO101_MJCF_DIR
	from digital_twin_armigo.inspection import InspectionResult
	from digital_twin_armigo.mujoco_cell import Cell
	from digital_twin_armigo.mujoco_cell.physics_expert import ExpertParams, run_closed_loop_episode
	from digital_twin_armigo.physics import load_physics_spec

	spec = load_physics_spec()
	base = {c.name: c for c in SENSITIVITY_MATRIX}["baseline_2x2"]
	lay = _layout(bench)
	seen_lengths = []

	def policy(history):
		seen_lengths.append(len(history))
		return (0.8, 400.0)  # roll beyond the joint range

	with Cell(lay, SO101_MJCF_DIR, physics=spec, dynamic_clutter=True, assisted_grasp=False, render_enabled=False) as c:
		_, _, info = run_closed_loop_episode(
			c,
			ExpertParams(max_extra_shows=2),
			q_start=bench._q_start(),
			brick_xy=lay.brick_xy,
			brick_yaw_deg=lay.brick_yaw_deg,
			non_ideal=bench.bench._mj_non_ideal(base, spec),
			bowls={"normal": lay.green, "defective": lay.pink},
			inspect=lambda *a: InspectionResult("UNKNOWN", None, "test", True, (), (), "undecided"),
			next_pose=policy,
		)
		roll_hi = float(np.degrees(c.jnt_range[4, 1]))
	assert seen_lengths == [3, 4]
	assert len(info["inspections"]) == 5
	assert info["inspections"][-1]["roll_deg"] == pytest.approx(roll_hi)
	assert info["aborted"] == "inspection_unknown"


def test_next_pose_none_stops_and_known_skips_policy(bench):
	from digital_twin_armigo.benchmarks import SENSITIVITY_MATRIX
	from digital_twin_armigo.constants import SO101_MJCF_DIR
	from digital_twin_armigo.inspection import InspectionResult
	from digital_twin_armigo.mujoco_cell import Cell
	from digital_twin_armigo.mujoco_cell.physics_expert import ExpertParams, run_closed_loop_episode
	from digital_twin_armigo.physics import load_physics_spec

	spec = load_physics_spec()
	base = {c.name: c for c in SENSITIVITY_MATRIX}["baseline_2x2"]
	lay = _layout(bench)
	calls = []
	known = InspectionResult("KNOWN", "normal", "test", True, (), (), "scripted")
	unknown = InspectionResult("UNKNOWN", None, "test", True, (), (), "undecided")
	for answer, expect_calls in ((unknown, 1), (known, 0)):
		calls.clear()
		with Cell(
			lay, SO101_MJCF_DIR, physics=spec, dynamic_clutter=True, assisted_grasp=False, render_enabled=False
		) as c:
			_, _, info = run_closed_loop_episode(
				c,
				ExpertParams(),
				q_start=bench._q_start(),
				brick_xy=lay.brick_xy,
				brick_yaw_deg=lay.brick_yaw_deg,
				non_ideal=bench.bench._mj_non_ideal(base, spec),
				bowls={"normal": lay.green, "defective": lay.pink},
				inspect=lambda *a, r=answer: r,
				next_pose=lambda h: calls.append(len(h)),  # returns None: stop
			)
		assert len(calls) == expect_calls
		assert info["aborted"] == (None if answer is known else "inspection_unknown")


# --- regressions from the adversarial review ---------------------------------------------------------------------


def test_drop_specks_keeps_brick_pieces_and_drops_streaks():
	pytest.importorskip("cv2")
	from digital_twin_armigo.mujoco_cell.cell import _drop_specks

	is_brick = np.array([False, True, False])  # geom 1 is the brick; index -1 (last) is "no geom"
	geom = np.full((60, 200), -1, dtype=np.int64)
	geom[10:30, 10:30] = 1  # the brick, 400 px
	geom[10:30, 32:40] = 1  # a piece cut off by a jaw, 160 px: kept
	geom[50, 100:130] = 1  # a far thin streak, 30 px (>= 10 px, < 10 % of 400): dropped
	geom[2, 150] = 1  # a single speck: dropped
	out = _drop_specks(geom, is_brick)
	assert (out[10:30, 10:30] == 1).all() and (out[10:30, 32:40] == 1).all()
	assert (out[50, 100:130] == -1).all() and out[2, 150] == -1


def test_held_brick_poser_moves_the_wrist_camera(bench):
	from digital_twin_armigo.constants import SO101_MJCF_DIR
	from digital_twin_armigo.mujoco_cell import Cell
	from digital_twin_armigo.mujoco_cell._mj import mujoco
	from digital_twin_armigo.mujoco_cell.inspection_views import HeldBrickPoser

	lay = _layout(bench)
	with Cell(lay, SO101_MJCF_DIR, assisted_grasp=False, render_enabled=False) as cell:
		mujoco.mj_forward(cell.model, cell.data)
		live = cell.data.qpos.copy()
		cid = mujoco.mj_name2id(cell.model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
		poser = HeldBrickPoser(cell)
		q = cell.joints()
		cam0 = poser.at(q).cam_xpos[cid].copy()
		q[3] += 0.5
		d = poser.at(q)
		# the wrist camera rides on the arm; the live simulation state is untouched
		assert np.linalg.norm(d.cam_xpos[cid] - cam0) > 0.01
		assert np.array_equal(cell.data.qpos, live)


def _run_inspect(bench, params, inspect, next_pose=None, lay=None):
	from digital_twin_armigo.benchmarks import SENSITIVITY_MATRIX
	from digital_twin_armigo.constants import SO101_MJCF_DIR
	from digital_twin_armigo.mujoco_cell import Cell
	from digital_twin_armigo.mujoco_cell.physics_expert import run_closed_loop_episode
	from digital_twin_armigo.physics import load_physics_spec

	spec = load_physics_spec()
	base = {c.name: c for c in SENSITIVITY_MATRIX}["baseline_2x2"]
	lay = lay or _layout(bench)
	with Cell(lay, SO101_MJCF_DIR, physics=spec, dynamic_clutter=True, assisted_grasp=False, render_enabled=False) as c:
		qs = []

		def spy(flex, roll_deg, frame):
			qs.append(c.joints().copy())
			return inspect(flex, roll_deg, frame)

		_, _, info = run_closed_loop_episode(
			c,
			params,
			q_start=bench._q_start(),
			brick_xy=lay.brick_xy,
			brick_yaw_deg=lay.brick_yaw_deg,
			non_ideal=bench.bench._mj_non_ideal(base, spec),
			bowls={"normal": lay.green, "defective": lay.pink},
			inspect=spy,
			next_pose=next_pose,
		)
	return info, qs


def test_policy_never_sees_inspector_reason_or_label(bench):
	from digital_twin_armigo.inspection import InspectionResult
	from digital_twin_armigo.mujoco_cell.physics_expert import POLICY_HISTORY_KEYS, ExpertParams

	seen = []
	unknown = InspectionResult("UNKNOWN", None, "test", True, (4,), (4,), "defect_face_not_visible")
	info, _ = _run_inspect(
		bench, ExpertParams(max_extra_shows=1), lambda *a: unknown, next_pose=lambda h: seen.extend(h)
	)
	assert seen and all(set(r) == set(POLICY_HISTORY_KEYS) for r in seen)
	assert all("defect" not in json.dumps(r) for r in seen)
	assert info["inspections"][0]["reason"] == "defect_face_not_visible"  # kept in the episode record


def test_inspection_happens_at_the_measured_show_pose(bench):
	"""The inspector is called once the arm has settled at the commanded show pose, not before."""
	from digital_twin_armigo.inspection import InspectionResult
	from digital_twin_armigo.mujoco_cell.physics_expert import ExpertParams

	p = ExpertParams(stop_when_known=False)
	unknown = InspectionResult("UNKNOWN", None, "test", True, (), (), "undecided")
	info, qs = _run_inspect(bench, p, lambda *a: unknown)
	assert [r["roll_deg"] for r in info["inspections"]] == list(p.show_roll_deg)  # all base poses, in order
	for q, rec in zip(qs, info["inspections"], strict=True):
		assert abs(q[3] - rec["flex"]) < p.show_flex_tol
		assert abs(np.degrees(q[4]) - rec["roll_deg"]) < p.show_roll_tol_deg


def test_stop_when_known_false_shows_every_base_pose_but_no_extras(bench):
	from digital_twin_armigo.inspection import InspectionResult
	from digital_twin_armigo.mujoco_cell.physics_expert import ExpertParams

	known = InspectionResult("KNOWN", "normal", "test", True, (), (), "scripted")
	info, _ = _run_inspect(
		bench, ExpertParams(stop_when_known=False, extra_show_poses=((0.0, 120.0),)), lambda *a: known
	)
	assert len(info["inspections"]) == 3
	assert info["inspection_label"] == "normal" and info["aborted"] is None


def test_static_extra_pose_is_clipped_and_recorded_clipped(bench):
	from digital_twin_armigo.inspection import InspectionResult
	from digital_twin_armigo.mujoco_cell.physics_expert import ExpertParams

	unknown = InspectionResult("UNKNOWN", None, "test", True, (), (), "undecided")
	info, qs = _run_inspect(bench, ExpertParams(extra_show_poses=((5.0, 400.0),)), lambda *a: unknown)
	last = info["inspections"][-1]
	assert last["flex"] < 5.0 and last["roll_deg"] < 400.0  # clipped to the joint limits
	assert abs(qs[-1][3] - last["flex"]) < 0.06  # and actually reached


def test_known_label_without_a_bowl_is_an_error(bench):
	from digital_twin_armigo.inspection import InspectionResult
	from digital_twin_armigo.mujoco_cell.physics_expert import ExpertParams

	odd = InspectionResult("KNOWN", "abnormal", "test", True, (), (), "scripted")
	with pytest.raises(ValueError, match="abnormal"):
		_run_inspect(bench, ExpertParams(), lambda *a: odd)
