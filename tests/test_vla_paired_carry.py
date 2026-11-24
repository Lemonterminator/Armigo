"""Paired opposite-target carry demonstrations (docs/vla/BOWL_CHANNEL_PAIRED_PLAN_2026-11-23.md Tasks 2-3)."""

from __future__ import annotations

import numpy as np
import pytest

from digital_twin_armigo.agent.twin_mujoco import TwinMujoco
from digital_twin_armigo.vla.paired_carry import TargetSwapTracer, restrict_supervision


class _Stub:
	target_swap = True


def test_target_bowl_default_mapping():
	assert TwinMujoco.target_bowl(object(), "defective") == "pink"
	assert TwinMujoco.target_bowl(object(), "normal") == "green"


def test_target_swap_flips_the_mapping():
	s = _Stub()
	assert TargetSwapTracer.target_bowl(s, "defective") == "green"
	assert TargetSwapTracer.target_bowl(s, "normal") == "pink"
	s.target_swap = False
	assert TargetSwapTracer.target_bowl(s, "defective") == "pink"
	assert TargetSwapTracer.target_bowl(s, "normal") == "green"


def test_restrict_supervision_keeps_only_carry_and_release():
	arrays = {
		"segment": np.array([0, 0, 1, 1, 2, 2, 3]),
		"supervise_action": np.ones(7, bool),
		"action": np.zeros((7, 6)),
	}
	kinds = ["grasp", "return_to_lift", "carry_to_bowl", "release_in_bowl"]
	out = restrict_supervision(arrays, kinds)
	assert out["supervise_action"].tolist() == [False, False, False, False, True, True, True]
	assert arrays["supervise_action"].all()  # input untouched
	assert out["action"] is not arrays["action"]
	out2 = restrict_supervision(arrays, kinds, from_frame=5)
	assert out2["supervise_action"].tolist() == [False] * 5 + [True, True]


def test_restrict_supervision_respects_existing_false_and_unlabelled_frames():
	arrays = {"segment": np.array([2, 2, -1]), "supervise_action": np.array([False, True, True])}
	out = restrict_supervision(arrays, ["a", "b", "carry_to_bowl"])
	assert out["supervise_action"].tolist() == [False, True, False]
	no_col = restrict_supervision({"segment": np.array([2, 0])}, ["a", "b", "carry_to_bowl"])
	assert no_col["supervise_action"].tolist() == [True, False]


def test_unsupervised_prefix_end_keeps_one_row_before_supervision(tmp_path):
	from digital_twin_armigo.agent.twin_record import unsupervised_prefix_end

	(tmp_path / "branch-0").mkdir()
	sup = np.zeros(10, dtype=bool)
	sup[6:9] = True
	np.savez(tmp_path / "branch-0" / "actions.npz", supervise_action=sup, action=np.zeros((10, 6)))
	assert unsupervised_prefix_end(tmp_path) == 5
	np.savez(tmp_path / "branch-0" / "actions.npz", supervise_action=np.zeros(10, bool), action=np.zeros((10, 6)))
	assert unsupervised_prefix_end(tmp_path) == 0
	np.savez(tmp_path / "branch-0" / "actions.npz", action=np.zeros((10, 6)))
	assert unsupervised_prefix_end(tmp_path) == 0


def test_trimmed_segments_rebase_and_drop_earlier_ones():
	from digital_twin_armigo.agent.twin_record import _trimmed_segments

	segs = [
		{"id": 0, "start": 0, "end": 4},
		{"id": 1, "start": 4, "end": 4},
		{"id": 2, "start": 4, "end": 9},
		{"id": 3, "start": 9, "end": 12},
		{"id": 4, "start": 12, "end": 12},
	]
	got = _trimmed_segments(segs, 5)
	assert [(s["id"], s["start"], s["end"]) for s in got] == [(2, 0, 4), (3, 4, 7), (4, 7, 7)]
	assert _trimmed_segments(segs, 0) == segs


def test_bowl_colour_and_true_bowl_xy_v3_and_legacy():
	from types import SimpleNamespace

	import pytest

	v3_layout = SimpleNamespace(
		green=(0.21, 0.11),
		pink=(0.22, -0.12),
		bowl_names={"green": "blue", "pink": "yellow"},
		extra_bowls=[
			("purple", 0.15, 0.23, (0.5, 0.2, 0.7)),
			("white", -0.12, 0.24, (0.9, 0.9, 0.9)),
		],
	)
	v3_twin = SimpleNamespace(layout=v3_layout)
	assert TwinMujoco.bowl_colour(v3_twin, "green") == "blue"
	assert TwinMujoco.bowl_colour(v3_twin, "pink") == "yellow"
	assert TwinMujoco.bowl_colour(v3_twin, "extra:0") == "purple"
	assert TwinMujoco.bowl_colour(v3_twin, "extra:1") == "white"
	assert TwinMujoco._true_bowl_xy(v3_twin, "green") == (0.21, 0.11)
	assert TwinMujoco._true_bowl_xy(v3_twin, "pink") == (0.22, -0.12)
	assert TwinMujoco._true_bowl_xy(v3_twin, "extra:0") == (0.15, 0.23)
	assert TwinMujoco._true_bowl_xy(v3_twin, "extra:1") == (-0.12, 0.24)
	with pytest.raises(KeyError):
		TwinMujoco.bowl_colour(v3_twin, "extra:2")
	with pytest.raises(KeyError):
		TwinMujoco._true_bowl_xy(v3_twin, "extra:2")
	with pytest.raises(KeyError):
		TwinMujoco._true_bowl_xy(v3_twin, "orange")

	legacy_layout = SimpleNamespace(green=(0.20, 0.10), pink=(0.20, -0.10), bowl_names=None, extra_bowls=[])
	legacy_twin = SimpleNamespace(layout=legacy_layout)
	assert TwinMujoco.bowl_colour(legacy_twin, "green") == "green"
	assert TwinMujoco.bowl_colour(legacy_twin, "pink") == "pink"
	assert TwinMujoco._true_bowl_xy(legacy_twin, "green") == (0.20, 0.10)
	assert TwinMujoco._true_bowl_xy(legacy_twin, "pink") == (0.20, -0.10)


def test_place_and_evaluate_route_extra_bowl_and_emit_v3_metadata():
	from types import SimpleNamespace

	extra_xy = (0.14, 0.25)

	def _make_stub(*, v3: bool, target_slot: str):
		events = []
		layout = SimpleNamespace(
			green=(0.22, 0.12),
			pink=(0.22, -0.12),
			defective=False,
			defect_face=2,
			brick_size=(0.0318, 0.0318, 0.0192),
			bowl_names={"green": "blue", "pink": "yellow"} if v3 else None,
			extra_bowls=[("purple", *extra_xy, (0.5, 0.2, 0.7))] if v3 else [],
			**({"_scene_v3": SimpleNamespace(variant=4)} if v3 else {}),
		)
		ex = SimpleNamespace(last=np.zeros(6), lost=False, bowl_xy=None, p=SimpleNamespace(show_move_s=0.1))
		ex._carry = lambda g: g
		ex._move = lambda *a, **kw: iter(())
		ex._carry_secs = lambda *a: 0.1
		ex._place = lambda *a: iter(())
		ex._go_home = lambda: iter(())
		cell = SimpleNamespace(brick_pos=lambda: np.array([extra_xy[0], extra_xy[1], 0.01]))
		stub = SimpleNamespace(
			layout=layout,
			bowls={
				"green": np.array([0.22, 0.12]),
				"pink": np.array([0.22, -0.12]),
				**({"extra:0": np.array(extra_xy)} if v3 else {}),
			},
			held=(np.zeros(6), 0.0),
			expert=ex,
			cell=cell,
			realistic=False,
			actual_destination=None,
			execution_error=None,
			branch=0,
			rewinds=0,
			perturbation_rewinds=0,
			retries=0,
			recovery_variants=(),
			total_frames=5,
			ep=SimpleNamespace(n_frames=5),
			record=SimpleNamespace(events=[], audit=lambda d: None),
			strategy="A",
			scored_count=0,
			review_count=0,
			_decide=lambda views: ("normal", set(range(6))),
			_llm_summary=dict,
			_smoothness=dict,
			_straight_clear=lambda *a, **kw: True,
			_execute=lambda g: None,
			_health=lambda target: {"verdict": "SUCCESS", "target_seen": tuple(target)},
			event=lambda kind, **kw: events.append((kind, kw)),
			target_bowl=lambda label: target_slot,
		)
		stub._bowl_slots = lambda: TwinMujoco._bowl_slots(stub)
		stub._true_bowl_xy = lambda slot: TwinMujoco._true_bowl_xy(stub, slot)
		stub._final_cycle_verdict = TwinMujoco._final_cycle_verdict
		return stub

	v3_stub = _make_stub(v3=True, target_slot="extra:0")
	assert TwinMujoco.place(v3_stub, "normal") is None
	assert np.allclose(v3_stub.expert.bowl_xy, extra_xy)
	assert v3_stub.actual_destination == "extra:0"
	res_v3 = TwinMujoco.evaluate(v3_stub, "normal", [], [])
	assert res_v3["verdict"] == "SUCCESS"
	assert res_v3["destination"] == "extra:0"
	assert res_v3["health"]["target_seen"] == extra_xy
	assert res_v3["scene"] == "v3"
	assert res_v3["variant"] == 4

	old_stub = _make_stub(v3=False, target_slot="green")
	old_stub.cell = SimpleNamespace(brick_pos=lambda: np.array([0.22, 0.12, 0.01]))
	assert TwinMujoco.place(old_stub, "normal") is None
	res_old = TwinMujoco.evaluate(old_stub, "normal", [], [])
	assert res_old["verdict"] == "SUCCESS"
	assert "scene" not in res_old
	assert "variant" not in res_old


def test_segment_mixin_bowl_name_uses_bowl_colour_with_stub_fallback():
	from types import SimpleNamespace

	from digital_twin_armigo.agent.student_rollout import _SegmentMixin

	v3_stub = SimpleNamespace(
		expert=SimpleNamespace(bowl_xy=np.array([0.15, 0.23])),
		bowls={"green": np.array([0.2, 0.1]), "pink": np.array([0.2, -0.1]), "extra:0": np.array([0.15, 0.23])},
		layout=SimpleNamespace(
			bowl_names={"green": "blue", "pink": "yellow"},
			extra_bowls=[("purple", 0.15, 0.23, (0.5, 0.2, 0.7))],
		),
	)
	v3_stub.bowl_colour = lambda slot: TwinMujoco.bowl_colour(v3_stub, slot)
	assert _SegmentMixin._bowl_name(v3_stub) == "purple"

	v3_stub.expert.bowl_xy = np.array([0.2, 0.1])
	assert _SegmentMixin._bowl_name(v3_stub) == "blue"

	minimal_stub = SimpleNamespace(
		expert=SimpleNamespace(bowl_xy=np.array([0.2, -0.1])),
		bowls={"green": np.array([0.2, 0.1]), "pink": np.array([0.2, -0.1])},
	)
	assert _SegmentMixin._bowl_name(minimal_stub) == "pink"


def test_target_swap_tracer_forced_target_and_run_branches_slot_selection(monkeypatch, tmp_path):
	import json
	from types import SimpleNamespace

	from digital_twin_armigo.vla import paired_carry

	forced_stub = SimpleNamespace(target_swap=False, forced_target="extra:0")
	assert TargetSwapTracer.target_bowl(forced_stub, "normal") == "extra:0"
	assert TargetSwapTracer.target_bowl(forced_stub, "defective") == "extra:0"

	tracer_calls = []

	class FakeTracer:
		def __init__(self, layout, scorer, record, *, crop_size, sensing, target_swap=True, forced_target=None, **kw):
			self.layout = layout
			self.record = record
			self.target_swap = target_swap
			self.forced_target = forced_target
			self.kw = kw
			tracer_calls.append({"target_swap": target_swap, "forced_target": forced_target, "kw": kw})

		def target_bowl(self, label):
			return TargetSwapTracer.target_bowl(self, label)

		def bowl_colour(self, slot):
			return TwinMujoco.bowl_colour(SimpleNamespace(layout=self.layout), slot)

		def close(self):
			pass

	class FakeGraph:
		def __init__(self, twin):
			self.twin = twin

		def update_state(self, *a, **kw):
			pass

		def invoke(self, *a, **kw):
			slot = self.twin.target_bowl("normal")
			col = self.twin.bowl_colour(slot)
			branch_dir = self.twin.record.path / "branch-0"
			branch_dir.mkdir(parents=True, exist_ok=True)
			np.savez(
				branch_dir / "actions.npz",
				state_f64=np.zeros((4, 6)),
				action=np.zeros((4, 6)),
				segment=np.array([0, 0, 1, 1], dtype=np.int32),
				supervise_action=np.ones(4, dtype=bool),
			)
			result = {
				"verdict": "SUCCESS",
				"student_eligible": True,
				"label": "normal",
				"destination": slot,
				"colour": col,
			}
			self.twin.record.last_result = result
			return {"result": result}

	class FakeTrace:
		def __init__(self, path, *, video=True):
			self.path = path
			self.path.mkdir(parents=True, exist_ok=True)
			self.last_result = None

		def finish(self, result, provenance):
			(self.path / "episode.json").write_text(
				json.dumps({"result": result, "provenance": provenance}), encoding="utf-8"
			)

	def fake_load_plan(_parent_dir):
		ck = SimpleNamespace(frame=2, branch=0, held=True)
		return SimpleNamespace(checkpoints=[ck], latest=ck, graph_state=dict)

	def fake_stitch(parent, branch_id, out_dir, *, branch_episode):
		out_dir.mkdir(parents=True, exist_ok=True)
		(out_dir / "branch-0").mkdir(exist_ok=True)
		np.savez(
			out_dir / "branch-0" / "actions.npz",
			state_f64=np.zeros((6, 6)),
			action=np.zeros((6, 6)),
			segment=np.array([0, 0, 1, 1, 2, 2], dtype=np.int32),
			supervise_action=np.ones(6, dtype=bool),
		)
		ep = json.loads((branch_episode / "episode.json").read_text(encoding="utf-8"))
		(out_dir / "episode.json").write_text(json.dumps(ep), encoding="utf-8")

	def fake_carry_only(ep_dir, from_frame):
		ep = json.loads((ep_dir / "episode.json").read_text(encoding="utf-8"))
		col = ep["result"]["colour"]
		return {"supervised_frames": 3, "carry_bowls": [col]}

	from digital_twin_armigo.agent import twin_graph, twin_mujoco, twin_record, twin_resume

	def make_4bowl_layout(seed, *_a, **_kw):
		return SimpleNamespace(
			defective=False,
			defect_seed=seed,
			green=(0.2, 0.1),
			pink=(0.2, -0.1),
			bowl_names={"green": "blue", "pink": "yellow"},
			extra_bowls=[
				("purple", 0.1, 0.2, (0.5, 0.2, 0.7)),
				("orange", -0.1, 0.2, (0.9, 0.5, 0.1)),
			],
		)

	monkeypatch.setattr(paired_carry, "TargetSwapTracer", FakeTracer)
	monkeypatch.setattr(paired_carry, "_carry_only", fake_carry_only)
	monkeypatch.setattr(twin_mujoco, "sample_case", make_4bowl_layout)
	monkeypatch.setattr(twin_graph, "build_graph", lambda twin, *a, **kw: FakeGraph(twin))
	monkeypatch.setattr(twin_record, "EpisodeTrace", FakeTrace)
	monkeypatch.setattr(twin_record, "stitch_perturbation_episode", fake_stitch)
	monkeypatch.setattr(twin_record, "student_frames", lambda ep: iter([{"frame": 0}]))
	monkeypatch.setattr(twin_resume, "load_plan", fake_load_plan)
	monkeypatch.setattr(twin_resume, "load_history", lambda _d, _c: (None, None))
	monkeypatch.setattr(twin_resume, "restore", lambda _t, _p: {"checkpoint": "cp0"})

	scorer = SimpleNamespace()
	rec1 = paired_carry.run_branches((3000010, False, 0), tmp_path / "r1", scorer, crop_size=256)
	rec2 = paired_carry.run_branches((3000010, False, 0), tmp_path / "r2", scorer, crop_size=256)
	assert rec1["fork_frame"] == 2
	assert rec1["fork_state_match"] is True
	assert len(rec1["branches"]) == 3
	targets1 = [b["target"] for b in rec1["branches"]]
	targets2 = [b["target"] for b in rec2["branches"]]
	assert targets1 == targets2
	assert targets1[0] == "green"
	assert len(set(targets1)) == 3
	assert [b["colour"] for b in rec1["branches"]] == [
		{"green": "blue", "pink": "yellow", "extra:0": "purple", "extra:1": "orange"}[t] for t in targets1
	]
	assert all("action_noise" not in c["kw"] for c in tracer_calls)

	tracer_calls.clear()
	sentinel_noise = object()
	paired_carry.run_branches((3000010, False, 0), tmp_path / "r3", scorer, crop_size=256, action_noise=sentinel_noise)
	assert all(c["kw"].get("action_noise") is sentinel_noise for c in tracer_calls)


def test_bowl_swap_probe_k_bowl_routing():
	import importlib.util
	from pathlib import Path
	from types import SimpleNamespace

	import pytest

	probe_path = Path(__file__).resolve().parents[1] / "scripts" / "vla" / "bowl_swap_probe.py"
	spec = importlib.util.spec_from_file_location("bowl_swap_probe", probe_path)
	mod = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(mod)

	from digital_twin_armigo.agent.twin_record import Segment

	def _run_probe(layout, bowls, req_colour, variant):
		seen_segs = []
		cp = SimpleNamespace(views=[])
		stub = SimpleNamespace(
			expert=SimpleNamespace(last=np.zeros(6), lost=False),
			held=(np.zeros(6), 0.0),
			checkpoints=[cp],
			total_frames=10,
			ep=SimpleNamespace(n_frames=10),
			cell=SimpleNamespace(
				tcp_pos=lambda: np.array([0.2, 0.0, 0.1]),
				brick_pos=lambda: np.array([0.2, 0.0, 0.09]),
			),
			retries=0,
			on_frame=None,
			segments=[],
			layout=layout,
			bowls=bowls,
			probe=None,
		)
		stub.bowl_colour = lambda slot: TwinMujoco.bowl_colour(stub, slot)
		stub._true_bowl_xy = lambda slot: TwinMujoco._true_bowl_xy(stub, slot)
		stub._save = lambda views: None
		stub.restore_checkpoint = lambda *a, **kw: None

		def fake_run_segment(seg):
			seen_segs.append(seg)
			stub.segments.append({"start": 10, "end": 15, "reason": "done"})

		stub.run_segment = fake_run_segment
		segs = {"carry_to_bowl": Segment("carry_to_bowl", {"bowl": req_colour}, variant=variant)}
		with pytest.raises(mod.ProbeDone):
			mod.BowlSwapTwin._student_release(stub, segs)
		return stub.probe, seen_segs

	v3_layout = SimpleNamespace(
		green=(0.22, 0.12),
		pink=(0.22, -0.12),
		brick_xy=(0.2, 0.0),
		bowl_names={"green": "blue", "pink": "yellow"},
		extra_bowls=[("purple", 0.14, 0.25, (0.5, 0.2, 0.7))],
	)
	probe_v3, segs_v3 = _run_probe(
		v3_layout,
		{"green": np.array([0.22, 0.12]), "pink": np.array([0.22, -0.12]), "extra:0": np.array([0.14, 0.25])},
		"blue",
		variant=3,
	)
	assert [(a["arm"], a["bowl"]) for a in probe_v3["arms"]] == [
		("req", "blue"),
		("req2", "blue"),
		("swap", "yellow"),
		("extra:0", "purple"),
	]
	assert all(s.variant == 3 for s in segs_v3)
	assert probe_v3["bowls_true"] == {"blue": [0.22, 0.12], "yellow": [0.22, -0.12], "purple": [0.14, 0.25]}

	v2_layout = SimpleNamespace(
		green=(0.22, 0.12),
		pink=(0.22, -0.12),
		brick_xy=(0.2, 0.0),
		bowl_names=None,
		extra_bowls=[],
	)
	probe_v2, segs_v2 = _run_probe(
		v2_layout,
		{"green": np.array([0.22, 0.12]), "pink": np.array([0.22, -0.12])},
		"green",
		variant=0,
	)
	assert [(a["arm"], a["bowl"]) for a in probe_v2["arms"]] == [
		("req", "green"),
		("req2", "green"),
		("swap", "pink"),
	]
	assert all(s.variant == 0 for s in segs_v2)
	assert probe_v2["bowls_true"] == {"green": [0.22, 0.12], "pink": [0.22, -0.12]}


@pytest.mark.slow
def test_v3_three_bowl_run_branches_smoke(tmp_path):
	import json

	import pytest

	pytest.importorskip("mujoco")
	from digital_twin_armigo.agent.twin_mujoco import sample_case
	from digital_twin_armigo.agent.twin_record import segment_labels, student_frames
	from digital_twin_armigo.inspection_scorer import FixedScorer
	from digital_twin_armigo.vla.paired_carry import run_branches

	case = next(
		(seed, True, 2, {"scene": "v3", "split": "train"})
		for seed in range(3000001, 3000050)
		if sample_case(seed, True, 2, {"scene": "v3", "split": "train"}).extra_bowls
	)
	scorer = FixedScorer(lambda _c, _r: 1.0, {"top": 0.0, "wrist_cam": 0.0})
	rec = run_branches(case, tmp_path / "branches", scorer, crop_size=256, sensing="realistic")

	assert rec["fork_state_match"] is True
	assert len(rec["branches"]) == 3
	assert len({b["target"] for b in rec["branches"]}) == 3
	assert any(b["target"].startswith("extra:") for b in rec["branches"])
	ok_branches = [b for b in rec["branches"] if b["verdict"] == "SUCCESS"]
	assert len(ok_branches) >= 2
	assert any(b["target"].startswith("extra:") for b in ok_branches)

	for b in ok_branches:
		ep_dir = tmp_path / "branches"
		from pathlib import Path

		b_dir = Path(b["dir"])
		assert b_dir.is_dir() and ep_dir in b_dir.parents
		assert b["supervised_frames"] > 0
		labels = segment_labels(b_dir)
		segs = labels["segments"]
		assert all("extra:" not in s["instruction"] for s in segs)
		carries = [s for s in segs if s["kind"] == "carry_to_bowl" and s["start"] >= rec["fork_frame"]]
		assert carries
		assert all(s["params"]["bowl"] == b["colour"] and b["colour"] in s["instruction"] for s in carries)
		with np.load(b_dir / "branch-0" / "actions.npz", allow_pickle=False) as z:
			sup = np.asarray(z["supervise_action"], dtype=bool)
			assert not sup[: rec["fork_frame"]].any()
			assert int(sup.sum()) == b["supervised_frames"]
			for idx in np.flatnonzero(sup):
				assert labels["kind"][idx] in ("carry_to_bowl", "release_in_bowl")
		manifest = json.loads((b_dir / "episode.json").read_text(encoding="utf-8"))
		assert manifest["result"]["scene"] == "v3"
		assert isinstance(manifest["result"]["variant"], int)
		frames = student_frames(b_dir)
		try:
			next(frames)
		finally:
			frames.close()
