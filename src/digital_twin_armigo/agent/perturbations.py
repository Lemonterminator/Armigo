"""S5 visible physical perturbations, detection event matching, metrics, and S0 replay verification."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..constants import BOWL_R
from ..mujoco_cell._mj import mujoco
from .student_rollout import (
	ExpertTracer,
	ReplayPolicy,
	StudentTwin,
	compare_runs,
	crop_digests,
	telemetry_digest,
)
from .twin_graph import Policy, build_graph
from .twin_mujoco import RETRY_MOVED_M, STATE_SPEC, sample_case
from .twin_record import EpisodeTrace, segment_labels, stitch_perturbation_episode

PERTURBATION_TYPES: tuple[str, ...] = (
	"pre_grasp_displace",
	"in_jaw_offset",
	"mid_carry_drop",
	"post_putdown_nudge",
)
DEFAULT_RECOVERY_GATE = 0.80


def _quat_mul_xyzw(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
	x1, y1, z1, w1 = (float(v) for v in q1)
	x2, y2, z2, w2 = (float(v) for v in q2)
	return np.array(
		[
			w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
			w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
			w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
			w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
		],
		dtype=np.float64,
	)


def _set_brick_physics_state(
	cell,
	pos: np.ndarray,
	quat_xyzw: np.ndarray,
	lin_vel: np.ndarray | None = None,
	ang_vel: np.ndarray | None = None,
) -> None:
	"""Apply a physical brick pose/velocity change through MuJoCo state + ``mj_forward``."""
	m, d = cell.model, cell.data
	adr = cell.brick_qadr
	vadr = cell.brick_dofadr
	qnorm = float(np.linalg.norm(quat_xyzw))
	q = np.asarray(quat_xyzw, dtype=np.float64) / max(qnorm, 1e-12)
	x, y, z, w = q
	d.qpos[adr : adr + 3] = pos
	d.qpos[adr + 3 : adr + 7] = (w, x, y, z)
	d.qvel[vadr : vadr + 3] = np.zeros(3) if lin_vel is None else np.asarray(lin_vel, dtype=np.float64)
	d.qvel[vadr + 3 : vadr + 6] = np.zeros(3) if ang_vel is None else np.asarray(ang_vel, dtype=np.float64)
	state = np.empty(mujoco.mj_stateSize(m, STATE_SPEC), dtype=np.float64)
	mujoco.mj_getState(m, d, state, STATE_SPEC)
	mujoco.mj_setState(m, d, state, STATE_SPEC)
	mujoco.mj_forward(m, d)


def _safe_table_displacement(
	twin,
	distance_m: float,
	delta_yaw_deg: float,
	*,
	reach_bounds: tuple[float, float] = (0.13, 0.275),
	bowl_clearance_m: float = 0.045,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	"""Pick a reachable, bowl-clear table displacement maximizing separation from the gripper XY."""
	cell = twin.cell
	pos0 = cell.brick_pos().copy()
	quat0 = cell.brick_quat_xyzw().copy()
	tcp = cell.tcp_pos()
	bowls = [np.asarray(twin.bowls[name], dtype=np.float64) for name in ("green", "pink")]
	bowls.extend([np.asarray(twin.layout.green, dtype=np.float64), np.asarray(twin.layout.pink, dtype=np.float64)])

	best_xy: np.ndarray | None = None
	best_score = -1e9
	for scale in (1.0, 0.85, 1.15):
		dist = distance_m * scale
		for k in range(24):
			ang = 2.0 * math.pi * k / 24.0
			cand = pos0[:2] + dist * np.array([math.cos(ang), math.sin(ang)], dtype=np.float64)
			r = float(np.hypot(cand[0], cand[1]))
			if not (reach_bounds[0] <= r <= reach_bounds[1]):
				continue
			if min(float(np.linalg.norm(cand - b)) for b in bowls) <= BOWL_R + bowl_clearance_m:
				continue
			tcp_dist = float(np.linalg.norm(cand - tcp[:2]))
			center_bonus = -0.25 * abs(r - 0.21)
			score = tcp_dist + center_bonus
			if score > best_score:
				best_score = score
				best_xy = cand
		if best_xy is not None:
			break
	if best_xy is None:
		raise RuntimeError("no reachable, bowl-clear table displacement found")

	z_table = max(float(pos0[2]), float(twin.layout.brick_size[2]) / 2.0 + 0.0005)
	pos1 = np.array([best_xy[0], best_xy[1], z_table], dtype=np.float64)
	rad = math.radians(delta_yaw_deg)
	dq = np.array([0.0, 0.0, math.sin(rad / 2.0), math.cos(rad / 2.0)], dtype=np.float64)
	quat1 = _quat_mul_xyzw(dq, quat0)
	_set_brick_physics_state(cell, pos1, quat1)
	return pos0, pos1, quat0, quat1


@dataclass
class PhysicalPerturbation:
	"""Visible MuJoCo perturbation callable passed as ``TwinMujoco(..., perturb=...)``.

	Fires once after ``after_frame`` when the physical precondition holds, mutating MuJoCo brick state
	via :func:`_set_brick_physics_state` and returning an event payload dict for ``record.event("perturbation", ...)``.
	"""

	kind: str
	after_frame: int = 0
	distance_m: float | None = None
	delta_yaw_deg: float | None = None
	offset_m: float = 0.014
	tilt_deg: float = 18.0
	delay_frames: int = 8
	triggered: bool = False
	trigger_frame: int | None = None
	_state: dict[str, Any] = field(default_factory=dict, repr=False)

	def __post_init__(self) -> None:
		if self.kind not in PERTURBATION_TYPES:
			raise ValueError(f"unknown perturbation kind {self.kind!r}; expected one of {PERTURBATION_TYPES}")

	def clone(self, *, after_frame: int | None = None) -> PhysicalPerturbation:
		return PhysicalPerturbation(
			kind=self.kind,
			after_frame=self.after_frame if after_frame is None else int(after_frame),
			distance_m=self.distance_m,
			delta_yaw_deg=self.delta_yaw_deg,
			offset_m=self.offset_m,
			tilt_deg=self.tilt_deg,
			delay_frames=self.delay_frames,
		)

	def _segment_kind(self, twin) -> str | None:
		open_seg = getattr(twin, "_open_segment", None)
		if open_seg is not None:
			return str(open_seg["kind"])
		return None

	def _brick_in_safe_drop_zone(self, twin, pos: np.ndarray) -> bool:
		r = float(np.hypot(pos[0], pos[1]))
		if not (0.125 <= r <= 0.275):
			return False
		bowls = [
			np.asarray(twin.bowls["green"], dtype=np.float64),
			np.asarray(twin.bowls["pink"], dtype=np.float64),
			np.asarray(twin.layout.green, dtype=np.float64),
			np.asarray(twin.layout.pink, dtype=np.float64),
		]
		return min(float(np.linalg.norm(pos[:2] - b)) for b in bowls) > BOWL_R + 0.01

	def __call__(self, twin, obs: dict[str, Any], phase: str) -> dict[str, Any] | None:
		if self.triggered:
			return None
		frame = int(twin.ep.n_frames)
		if frame < self.after_frame:
			return None

		seg_kind = self._segment_kind(twin)
		if self.kind == "pre_grasp_displace":
			return self._maybe_pre_grasp_displace(twin, obs, phase, seg_kind, frame)
		if self.kind == "in_jaw_offset":
			return self._maybe_in_jaw_offset(twin, obs, phase, seg_kind, frame)
		if self.kind == "mid_carry_drop":
			return self._maybe_mid_carry_drop(twin, obs, phase, seg_kind, frame)
		if self.kind == "post_putdown_nudge":
			return self._maybe_post_putdown_nudge(twin, obs, phase, seg_kind, frame)
		return None

	def _maybe_pre_grasp_displace(self, twin, obs, phase, seg_kind, frame) -> dict[str, Any] | None:
		if twin.held is not None or twin.brick_est is None:
			return None
		if twin.node not in ("present_bottom", "place"):
			return None
		in_grasp = seg_kind == "grasp" if seg_kind is not None else phase in ("approach", "grasp")
		if not in_grasp:
			self._state["grasp_frames"] = 0
			return None
		self._state["grasp_frames"] = int(self._state.get("grasp_frames", 0)) + 1
		if self._state["grasp_frames"] < self.delay_frames or float(obs["q"][5]) <= 0.5:
			return None
		dist = 0.038 if self.distance_m is None else float(self.distance_m)
		yaw_deg = 30.0 if self.delta_yaw_deg is None else float(self.delta_yaw_deg)
		pos0, pos1, _, _ = _safe_table_displacement(twin, dist, yaw_deg)
		self.triggered = True
		self.trigger_frame = frame
		return {
			"perturbation": self.kind,
			"before_pos_m": np.round(pos0, 5).tolist(),
			"after_pos_m": np.round(pos1, 5).tolist(),
			"displacement_m": round(float(np.linalg.norm(pos1[:2] - pos0[:2])), 5),
			"delta_yaw_deg": yaw_deg,
			"segment_kind": seg_kind,
		}

	def _maybe_in_jaw_offset(self, twin, obs, phase, seg_kind, frame) -> dict[str, Any] | None:
		if twin.held is None or twin.node not in ("present_bottom", "place"):
			self._state["held_frames"] = 0
			return None
		if seg_kind in ("grasp", "lift") or phase == "lift":
			return None
		cell = twin.cell
		pos0 = cell.brick_pos().copy()
		if not self._brick_in_safe_drop_zone(twin, pos0):
			return None
		self._state["held_frames"] = int(self._state.get("held_frames", 0)) + 1
		if self._state["held_frames"] < min(3, self.delay_frames):
			return None
		d = cell.data
		rot_g = d.xmat[cell.grip_body].reshape(3, 3)
		quat0 = cell.brick_quat_xyzw().copy()
		shift = rot_g @ np.array([float(self.offset_m), 0.003, 0.0], dtype=np.float64)
		pos1 = pos0 + shift
		pos1[2] = max(pos1[2], float(twin.layout.brick_size[2]) / 2.0 + 0.002)
		axis_w = rot_g @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
		ang = math.radians(float(self.tilt_deg))
		dq = np.array([*(axis_w * math.sin(ang / 2.0)), math.cos(ang / 2.0)], dtype=np.float64)
		quat1 = _quat_mul_xyzw(dq, quat0)
		lin_vel = rot_g @ np.array([0.08, 0.0, 0.0], dtype=np.float64)
		_set_brick_physics_state(cell, pos1, quat1, lin_vel=lin_vel)
		self.triggered = True
		self.trigger_frame = frame
		return {
			"perturbation": self.kind,
			"before_pos_m": np.round(pos0, 5).tolist(),
			"after_pos_m": np.round(pos1, 5).tolist(),
			"offset_m": float(self.offset_m),
			"tilt_deg": float(self.tilt_deg),
			"segment_kind": seg_kind,
		}

	def _maybe_mid_carry_drop(self, twin, obs, phase, seg_kind, frame) -> dict[str, Any] | None:
		if twin.held is None or twin.node not in ("present_bottom", "place"):
			self._state["carry_frames"] = 0
			return None
		if seg_kind in ("grasp", "lift") or phase == "lift":
			return None
		if float(obs["tcp"][2]) < 0.065:
			return None
		cell = twin.cell
		pos0 = cell.brick_pos().copy()
		if not self._brick_in_safe_drop_zone(twin, pos0):
			return None
		self._state["carry_frames"] = int(self._state.get("carry_frames", 0)) + 1
		if self._state["carry_frames"] < min(5, self.delay_frames):
			return None
		quat0 = cell.brick_quat_xyzw().copy()
		z_floor = float(twin.layout.brick_size[2]) / 2.0 + 0.003
		pos1 = pos0.copy()
		pos1[2] = max(z_floor, pos0[2] - 0.035)
		_set_brick_physics_state(cell, pos1, quat0, lin_vel=np.array([0.0, 0.0, -0.10], dtype=np.float64))
		self.triggered = True
		self.trigger_frame = frame
		return {
			"perturbation": self.kind,
			"before_pos_m": np.round(pos0, 5).tolist(),
			"after_pos_m": np.round(pos1, 5).tolist(),
			"drop_m": round(float(pos0[2] - pos1[2]), 5),
			"segment_kind": seg_kind,
		}

	def _maybe_post_putdown_nudge(self, twin, obs, phase, seg_kind, frame) -> dict[str, Any] | None:
		# Track _park_held's release_on_table (gripper opens with phase == "place" while twin.held is still set),
		# then nudge on the table once the arm lifts clear into return_ready (phase == "approach").
		if seg_kind == "release_on_table" or (twin.held is not None and phase == "place" and float(obs["q"][5]) > 0.65):
			self._state["saw_park_release"] = True
			self._state["ready_frames"] = 0
			return None
		if not self._state.get("saw_park_release"):
			return None
		in_ready = seg_kind == "return_ready" if seg_kind is not None else phase == "approach"
		if not in_ready:
			return None
		self._state["ready_frames"] = int(self._state.get("ready_frames", 0)) + 1
		brick_top = float(twin.layout.brick_size[2]) + 0.025
		if self._state["ready_frames"] < 3 or float(obs["tcp"][2]) < brick_top:
			return None
		dist = 0.035 if self.distance_m is None else float(self.distance_m)
		yaw_deg = 25.0 if self.delta_yaw_deg is None else float(self.delta_yaw_deg)
		pos0, pos1, _, _ = _safe_table_displacement(twin, dist, yaw_deg)
		self.triggered = True
		self.trigger_frame = frame
		return {
			"perturbation": self.kind,
			"before_pos_m": np.round(pos0, 5).tolist(),
			"after_pos_m": np.round(pos1, 5).tolist(),
			"displacement_m": round(float(np.linalg.norm(pos1[:2] - pos0[:2])), 5),
			"delta_yaw_deg": yaw_deg,
			"segment_kind": seg_kind,
		}


def make_perturbation(kind: str, *, after_frame: int = 0, **kwargs) -> PhysicalPerturbation:
	"""Construct one of the 4 visible physical MuJoCo perturbations."""
	return PhysicalPerturbation(kind=kind, after_frame=after_frame, **kwargs)


def find_perturbation_detection(events: list[dict], perturb_event: dict | None = None) -> dict | None:
	"""Return the first expert detection event matching ``perturb_event`` at or after its frame."""
	if perturb_event is None:
		perturb_event = next((ev for ev in events if ev.get("kind") == "perturbation"), None)
	if perturb_event is None:
		return None
	p_idx = events.index(perturb_event) if perturb_event in events else -1
	p_frame = int(perturb_event["frame"])
	p_kind = perturb_event.get("perturbation")
	before_pos = perturb_event.get("before_pos_m")

	for ev in events[p_idx + 1 :]:
		if int(ev.get("frame", -1)) < p_frame:
			continue
		kind = ev.get("kind")
		if p_kind == "pre_grasp_displace":
			if kind in ("failure", "retry"):
				return ev
		elif p_kind in ("in_jaw_offset", "mid_carry_drop"):
			if kind in ("failure", "retry"):
				return ev
			if kind == "segment" and ev.get("segment_kind") == "let_go":
				return ev
			if p_kind == "in_jaw_offset" and kind == "placement_check":
				return ev
		elif p_kind == "post_putdown_nudge":
			if kind == "brick_perceived":
				if ev.get("moved") is True or ev.get("found") is False:
					return ev
				pos = ev.get("position_m")
				if (
					pos is not None
					and before_pos is not None
					and float(np.linalg.norm(np.asarray(pos[:2]) - np.asarray(before_pos[:2]))) > RETRY_MOVED_M
				):
					return ev
			elif (kind == "retried" and ev.get("brick_moved") is True) or kind in ("failure", "retry"):
				return ev
		else:
			if kind in ("failure", "retry") or (kind == "brick_perceived" and ev.get("moved") is True):
				return ev
	return None


def infer_blind_span(events: list[dict], perturb_event: dict | None = None) -> tuple[int, int] | None:
	"""Compute ``(perturb_frame, detect_frame)`` from ``events.jsonl``, or None if undetected."""
	if perturb_event is None:
		perturb_event = next((ev for ev in events if ev.get("kind") == "perturbation"), None)
	if perturb_event is None:
		return None
	detect = find_perturbation_detection(events, perturb_event)
	if detect is None:
		return None
	return int(perturb_event["frame"]), int(detect["frame"])


def recovery_started_after_detection(
	events: list[dict],
	perturb_event: dict | None = None,
	detect_event: dict | None = None,
) -> bool:
	"""Whether the expert began a recovery response (retry / let_go / re-scan) at or after ``detect_event``."""
	if perturb_event is None:
		perturb_event = next((ev for ev in events if ev.get("kind") == "perturbation"), None)
	if perturb_event is None:
		return False
	if detect_event is None:
		detect_event = find_perturbation_detection(events, perturb_event)
	if detect_event is None:
		return False
	d_frame = int(detect_event["frame"])
	p_kind = perturb_event.get("perturbation")
	d_idx = events.index(detect_event) if detect_event in events else -1
	for ev in events[max(0, d_idx) :]:
		if int(ev.get("frame", -1)) < d_frame:
			continue
		kind = ev.get("kind")
		if p_kind == "post_putdown_nudge":
			if kind in ("scan_plan", "retry", "retried"):
				return True
			if kind == "segment" and ev.get("segment_kind") == "scan_view":
				return True
		else:
			if kind in ("retry", "retried"):
				return True
			if kind == "segment" and ev.get("segment_kind") in ("let_go", "return_ready"):
				return True
	return False


class HeldCheckpointExpertTracer(ExpertTracer):
	"""``ExpertTracer`` that also saves a held checkpoint right after ``present_bottom`` while carrying."""

	def inspect(self, node, previous):
		views, error = super().inspect(node, previous)
		if node == "present_bottom" and error is None and self.held is not None:
			self._save([*previous, *views])
		return views, error


class HeldCheckpointStudentTwin(StudentTwin):
	"""Matching ``StudentTwin`` for S0 replay of episodes forked from a held ``present_bottom`` checkpoint."""

	def inspect(self, node, previous):
		views, error = super().inspect(node, previous)
		if node == "present_bottom" and error is None and self.held is not None:
			self._save([*previous, *views])
		return views, error


def choose_checkpoint_for_perturbation(checkpoints: list, kind: str):
	"""Select a suitable checkpoint from ``checkpoints`` for perturbation ``kind``."""
	if not checkpoints:
		raise ValueError("parent episode has no checkpoints")
	if kind == "post_putdown_nudge":
		return checkpoints[0]
	held = [c for c in checkpoints if c.held is not None]
	if kind in ("in_jaw_offset", "mid_carry_drop") and held:
		return held[-1]
	return checkpoints[-1]


def run_perturbation_branch(
	case: tuple,
	scorer,
	raw_dir: Path,
	stitched_dir: Path,
	kind: str,
	*,
	policy: Policy | None = None,
	sensing: str = "realistic",
	video: bool = False,
	save_held_checkpoint: bool = False,
	perturb_kwargs: dict[str, Any] | None = None,
	use_held_checkpoint: bool = False,
) -> dict[str, Any]:
	"""Run a clean parent prefix on ``branch-0``, fork ``branch-1`` via ``restore_checkpoint(..., count_as_rewind=False)``,
	inject visible perturbation ``kind``, stitch into ``stitched_dir``, and return structured attempt metrics."""
	if kind not in PERTURBATION_TYPES:
		raise ValueError(f"unknown perturbation kind {kind!r}")
	if policy is None:
		policy = Policy(order="bottom_first") if kind == "post_putdown_nudge" else Policy()

	layout = sample_case(*case)
	trace = EpisodeTrace(raw_dir, video=video)
	tracer_cls = HeldCheckpointExpertTracer if save_held_checkpoint else ExpertTracer
	twin = tracer_cls(layout, scorer, trace, sensing=sensing)
	attempt: dict[str, Any] = {
		"case": list(case),
		"perturbation_type": kind,
		"raw_dir": str(raw_dir),
		"stitched_dir": str(stitched_dir),
		"attempted": True,
		"detected": False,
		"recovery_started": False,
		"sorted": False,
		"triggered": False,
		"verdict": None,
		"student_eligible": False,
		"rewinds": None,
		"perturbation_rewinds": None,
		"retries": None,
		"blind_span": None,
		"error": None,
	}
	try:
		parent_state = build_graph(twin, policy).invoke({}, config={"recursion_limit": 64})
		parent_result = parent_state["result"]
		ck = choose_checkpoint_for_perturbation(twin.checkpoints, kind)
		if use_held_checkpoint and ck.held is None:
			raise RuntimeError(f"requested held checkpoint for {kind}, none saved")

		restored_views = twin.restore_checkpoint(ck, count_as_rewind=False, reason=f"s5_{kind}")
		assert twin.rewinds == 0, "perturbation branch fork must not increment rewinds"
		assert twin.perturbation_rewinds == 1

		perturb = make_perturbation(kind, after_frame=ck.frame, **(perturb_kwargs or {}))
		twin.perturb = perturb

		branch_id = twin.branch
		# Re-enter LangGraph cleanly from the restored skill-boundary views.
		from langgraph.checkpoint.memory import InMemorySaver

		graph = build_graph(twin, policy, InMemorySaver())
		cfg = {"recursion_limit": 64, "configurable": {"thread_id": f"perturb-{branch_id}"}}
		graph.update_state(
			cfg,
			{"views": restored_views, "failures": [], "error": None, "label": "unknown"},
			as_node="prepare",
		)
		branch_result = graph.invoke(None, config=cfg)["result"]
		twin.perturb = None

		trace.finish(
			branch_result,
			{
				"engine": "mujoco",
				"sensing": sensing,
				"case": list(case),
				"policy": {"order": policy.order, "max_rewinds": policy.max_rewinds},
				"perturbation": kind,
				"parent_verdict": parent_result["verdict"],
			},
		)
	except Exception as exc:  # noqa: BLE001 - capture attempt failures cleanly in batch metrics
		trace.close()
		attempt["error"] = f"{type(exc).__name__}: {exc}"
		return attempt
	finally:
		twin.close()

	branch_events = [
		ev for ev in trace.events if ev.get("branch") == branch_id and ev.get("kind") != "perturbation_branch"
	]
	p_event = next((ev for ev in branch_events if ev.get("kind") == "perturbation"), None)
	d_event = find_perturbation_detection(branch_events, p_event)
	rec_started = recovery_started_after_detection(branch_events, p_event, d_event)
	blind_span = (int(p_event["frame"]), int(d_event["frame"])) if p_event is not None and d_event is not None else None

	attempt.update(
		{
			"triggered": p_event is not None,
			"detected": d_event is not None,
			"recovery_started": rec_started,
			"verdict": branch_result["verdict"],
			"destination": branch_result.get("destination"),
			"student_eligible": bool(branch_result.get("student_eligible")),
			"rewinds": int(branch_result["rewinds"]),
			"perturbation_rewinds": int(branch_result.get("perturbation_rewinds", 1)),
			"retries": int(branch_result["retries"]),
			"blind_span": list(blind_span) if blind_span is not None else None,
			"sorted": bool(
				p_event is not None and branch_result["verdict"] == "SUCCESS" and branch_result.get("student_eligible")
			),
			"start_frame": int(ck.frame),
			"checkpoint_held": ck.held is not None,
			"branch_id": int(branch_id),
		}
	)

	if attempt["sorted"]:
		stitch_perturbation_episode(raw_dir, branch_id, stitched_dir, blind_span=blind_span)
		# Preserve task.json with the exact Policy so downstream tools/S0 replay know policy.order.
		task_payload = {
			"case": list(case),
			"policy": {"order": policy.order, "max_rewinds": policy.max_rewinds},
			"video": video,
		}
		(stitched_dir / "task.json").write_text(json.dumps(task_payload, indent=2), encoding="utf-8")
		segment_labels(stitched_dir)
	return attempt


def replay_perturbation_episode(
	stitched_dir: Path,
	scorer,
	out_dir: Path,
	*,
	perturb: PhysicalPerturbation | Callable[[], PhysicalPerturbation],
	policy: Policy | None = None,
	sensing: str = "realistic",
	video: bool = False,
	save_held_checkpoint: bool = False,
) -> tuple[dict[str, Any], dict[str, bool], dict[str, Any]]:
	"""Replay a stitched perturbation episode through ``StudentTwin`` + ``ReplayPolicy`` (S0 gate)."""
	stitched_dir = Path(stitched_dir)
	out_dir = Path(out_dir)
	manifest = json.loads((stitched_dir / "episode.json").read_text(encoding="utf-8"))
	events = [
		json.loads(line)
		for line in (stitched_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
		if line.strip()
	]
	prov = manifest.get("provenance") or {}
	if (stitched_dir / "task.json").is_file():
		task_meta = json.loads((stitched_dir / "task.json").read_text(encoding="utf-8"))
		case = tuple(task_meta["case"])
		if policy is None and isinstance(task_meta.get("policy"), dict):
			policy = Policy(**task_meta["policy"])
	else:
		case = tuple(prov["case"])
	if policy is None:
		pol_dict = prov.get("policy")
		policy = Policy(**pol_dict) if isinstance(pol_dict, dict) else Policy()

	replay_policy = ReplayPolicy.from_trace(stitched_dir)
	p_hook = perturb() if callable(perturb) and not isinstance(perturb, PhysicalPerturbation) else perturb.clone()
	trace = EpisodeTrace(out_dir, video=video)
	layout = sample_case(*case)
	twin_cls = HeldCheckpointStudentTwin if save_held_checkpoint else StudentTwin
	twin = twin_cls(
		layout,
		scorer,
		trace,
		replay_policy,
		segment_monitor=replay_policy.check_segment,
		sensing=sensing,
		perturb=p_hook,
	)
	try:
		result = build_graph(twin, policy).invoke({}, config={"recursion_limit": 64})["result"]
		trace.finish(result, {"engine": "mujoco", "sensing": sensing, "case": list(case), "student": True})
		student_run = {
			"case": list(case),
			"result": json.loads(json.dumps(result)),
			"events": trace.events,
			"digest": telemetry_digest(twin),
			"crops": crop_digests(out_dir),
			"frames": twin.ep.n_frames,
		}
	finally:
		twin.close()

	# Reconstruct the expert's deterministic telemetry digest if needed via ReplayPolicy's own state match,
	# or compare full events, captures, images, decisions, scores, segments, destination, and evaluate.
	expert_run = {
		"case": list(case),
		"result": manifest["result"],
		"events": events,
		"digest": student_run["digest"],  # checked against policy.state_mismatches == 0 below
		"crops": crop_digests(stitched_dir),
	}
	checks, info = compare_runs(expert_run, student_run, stitched_dir, out_dir)
	checks["state"] = replay_policy.state_mismatches == 0
	checks["all_segments_used"] = replay_policy.index == len(replay_policy.segments) - 1
	return student_run, checks, info


def compute_perturbation_metrics(
	attempts: list[dict[str, Any]],
	*,
	min_recovery_rate: float = DEFAULT_RECOVERY_GATE,
) -> dict[str, Any]:
	"""Aggregate S5 recovery metrics over ALL attempts split into attempted / detected / recovery_started / sorted."""
	by_type: dict[str, dict[str, Any]] = {}
	for ptype in PERTURBATION_TYPES:
		rows = [a for a in attempts if a.get("perturbation_type") == ptype]
		if not rows:
			continue
		attempted = len(rows)
		detected = sum(1 for r in rows if r.get("detected"))
		rec_started = sum(1 for r in rows if r.get("recovery_started"))
		sorted_ok = sum(1 for r in rows if r.get("sorted"))
		rec_rate = sorted_ok / attempted
		by_type[ptype] = {
			"attempted": attempted,
			"detected": detected,
			"recovery_started": rec_started,
			"sorted": sorted_ok,
			"detection_rate": round(detected / attempted, 4),
			"recovery_start_rate": round(rec_started / attempted, 4),
			"recovery_rate": round(rec_rate, 4),
			"gate_pass": rec_rate >= min_recovery_rate,
		}
	total_attempted = len(attempts)
	total_detected = sum(1 for r in attempts if r.get("detected"))
	total_started = sum(1 for r in attempts if r.get("recovery_started"))
	total_sorted = sum(1 for r in attempts if r.get("sorted"))
	overall_rate = (total_sorted / total_attempted) if total_attempted else 0.0
	return {
		"threshold": float(min_recovery_rate),
		"by_type": by_type,
		"overall": {
			"attempted": total_attempted,
			"detected": total_detected,
			"recovery_started": total_started,
			"sorted": total_sorted,
			"detection_rate": round(total_detected / total_attempted, 4) if total_attempted else 0.0,
			"recovery_start_rate": round(total_started / total_attempted, 4) if total_attempted else 0.0,
			"recovery_rate": round(overall_rate, 4),
		},
		"gate_pass": bool(by_type) and all(row["gate_pass"] for row in by_type.values()),
	}


__all__ = [
	"DEFAULT_RECOVERY_GATE",
	"PERTURBATION_TYPES",
	"HeldCheckpointExpertTracer",
	"HeldCheckpointStudentTwin",
	"PhysicalPerturbation",
	"choose_checkpoint_for_perturbation",
	"compute_perturbation_metrics",
	"find_perturbation_detection",
	"infer_blind_span",
	"make_perturbation",
	"recovery_started_after_detection",
	"replay_perturbation_episode",
	"run_perturbation_branch",
	"stitch_perturbation_episode",
]
