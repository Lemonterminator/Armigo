"""Trusted MuJoCo skills and skill-boundary checkpoints for the six-face graph."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import ClassVar

import numpy as np

from ..benchmarks import SENSITIVITY_MATRIX
from ..camera import CameraPose, look_from
from ..constants import BOWL_R, FPS, PHASES, REST_U, SO101_MJCF_DIR
from ..inspection import assert_no_defect_leakage
from ..joints import normalized_to_rad
from ..metrics import evaluate_episode_health
from ..mujoco_cell import Cell
from ..mujoco_cell._mj import mujoco
from ..mujoco_cell.episode import PhysicsEpisode, observe
from ..mujoco_cell.inspection_planning import path_min_z
from ..mujoco_cell.inspection_views import capture_view, detector_crop
from ..mujoco_cell.layout import sample_layout
from ..mujoco_cell.physics_expert import ExpertParams
from ..mujoco_cell.table_relocation import RelocateExpert, RelocateParams, plan_bottom_view, plan_table_scan
from ..physics import NonIdealityCfg, load_physics_spec
from .twin_graph import Policy, decision, select_skill

ROOT = Path(__file__).resolve().parents[3]
INITIAL_Q = np.array([*normalized_to_rad(np.array(REST_U, dtype=np.float64))[:5], ExpertParams().open_g])
STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION

PROFILE = "mujoco-close-top-1.35"
BOTTOM_MIN_PX = 400
WRIST_MIN_PX = 800
MIN_BRICK_PX = 150
MAX_TOTAL_FRAMES = 6300


def sample_case(seed: int, positive: bool, face: int):
	if type(seed) is not int or not 0 <= seed <= 2**32 - 1 or type(positive) is not bool:
		raise ValueError("invalid fixture seed or class")
	if type(face) is not int or not 0 <= face < 6:
		raise ValueError("face must be in [0, 5]")
	raw = json.loads((ROOT / "config/sim/top_camera_joint.json").read_text(encoding="utf-8"))
	layout = sample_layout(np.random.default_rng(seed), CameraPose(**raw["camera"]), defective=positive)
	layout.top = CameraPose(0.4895033468, 0.0, 0.5130680468, 180.0, 25.0712240655, 0.0, 612.47)
	layout.front = look_from((0.20, 0.0), 0.55, 0.35, 35, focal_px=560)
	layout.defect_kind = "cross_var"
	layout.defect_face = face
	layout.defect_seed = 19_199 + seed * 100_019
	layout.clutter = []
	layout.distractors = []
	return layout


@contextmanager
def blind_geometry(cell):
	# Group 4 is disabled in both visibility render options. The model is session-local.
	original = cell.model.geom_group[cell.defect_geoms].copy()
	cell.model.geom_group[cell.defect_geoms] = 4
	try:
		yield
	finally:
		cell.model.geom_group[cell.defect_geoms] = original


@dataclass
class Checkpoint:
	id: str
	branch: int
	frame: int
	integration: np.ndarray
	expert: dict
	filter: dict
	wrist_count: int
	bottom_count: int
	views: list[dict]


# Bounded recovery variants. Each is a fixed, named, deterministic transform of the default expert parameters. The
# agent (or LLM) only ever picks a name; no number crosses that boundary. Every value must lie in RECOVERY_BOUNDS and
# every variant only slows motion down or raises the transfer height, so a variant cannot be more aggressive than the
# defaults. Keys of "expert" / "relocate" are ExpertParams / RelocateParams fields; "scan_peak_speed" is the peak
# joint speed (rad/s) of the wrist-scan moves (default SCAN_PEAK_SPEED, the value the scan used before variants).
SCAN_PEAK_SPEED = 1.0
RECOVERY_BOUNDS = MappingProxyType(
	{
		"scan_peak_speed": (0.25, SCAN_PEAK_SPEED),
		"scan_move_s": (0.6, 1.5),
		"ready_move_s": (0.6, 1.2),
		"carry_peak_speed_rad_s": (0.2, 0.5),
		"show_move_s": (0.9, 2.0),
		"transfer_s": (1.8, 3.0),
		"transfer_z_m": (0.12, 0.15),
		"place_descend_s": (1.2, 2.0),
		"retreat_s": (0.8, 1.5),
		"return_s": (1.2, 2.0),
	}
)
_SLOW_SCAN = {"scan_peak_speed": 0.35, "scan_move_s": 1.0, "ready_move_s": 1.0}
_SLOW_CARRY = {"carry_peak_speed_rad_s": 0.3, "show_move_s": 1.4, "transfer_s": 2.6}
_GENTLE_PLACE = {"place_descend_s": 1.8, "retreat_s": 1.2, "return_s": 1.8}
RECOVERY_VARIANTS = MappingProxyType(
	{
		"slow_scan": MappingProxyType(_SLOW_SCAN),
		"slow_carry": MappingProxyType(_SLOW_CARRY),
		"gentle_place": MappingProxyType(_GENTLE_PLACE),
		"high_place": MappingProxyType({**_GENTLE_PLACE, "transfer_z_m": 0.14}),
		"cautious": MappingProxyType({**_SLOW_SCAN, **_SLOW_CARRY, **_GENTLE_PLACE}),
	}
)
VARIANT_NOTES = MappingProxyType(
	{
		"slow_scan": "slower wrist-scan and ready moves, so joints stop saturating",
		"slow_carry": "slower carry and show moves",
		"gentle_place": "slower place descent, retreat and return",
		"high_place": "slower place moves with a higher approach height over the bowl",
		"cautious": "slow scan, carry and place together",
	}
)
# Failure evidence -> recommended variant. A torque stall is attributed to the skill that was running when the
# saturated run started; other physical or placement failures map by code.
STALL_VARIANT_BY_NODE = MappingProxyType(
	{"scan_wrist": "slow_scan", "present_bottom": "slow_carry", "place": "gentle_place"}
)
VARIANT_BY_CODE = MappingProxyType(
	{
		"BRICK_LOST": "slow_carry",
		"PLACEMENT_UNVERIFIED": "high_place",
		"PLACE_RETURN_PATH_BLOCKED": "high_place",
		"PARK_PATH_BLOCKED": "slow_carry",
	}
)
RELOCATE_KEYS = frozenset({"scan_move_s"})  # RelocateParams fields a variant may set
STALL_REASONS = ("torque_stall", "tracking_lag")
STALL_SECONDS = 1.5  # metrics.evaluate_episode_health max_stall_secs
STALL_FRACTION = 0.95  # fraction of the torque limit that counts as saturated


def apply_variant(name, params, relocate):
	"""Return ``(ExpertParams, RelocateParams, scan_peak_speed)`` for variant ``name`` applied to the given defaults.
	Raises ValueError for an unknown name or a value outside RECOVERY_BOUNDS."""
	if name not in RECOVERY_VARIANTS:
		raise ValueError(f"unknown recovery variant: {name}")
	values = RECOVERY_VARIANTS[name]
	for key, value in values.items():
		lo, hi = RECOVERY_BOUNDS[key]
		if not lo <= value <= hi:
			raise ValueError(f"variant {name}: {key}={value} outside [{lo}, {hi}]")
	# transfer_z_m exists in both parameter classes: only the ExpertParams one (the bowl approach) is a variant key.
	rel = {k: v for k, v in values.items() if k in RELOCATE_KEYS}
	expert = {k: v for k, v in values.items() if k not in RELOCATE_KEYS and k != "scan_peak_speed"}
	return (
		replace(params, **expert),
		replace(relocate, **rel),
		float(values.get("scan_peak_speed", SCAN_PEAK_SPEED)),
	)


def stall_start(torques, limit, fps=FPS):
	"""First frame of a continuous run of saturated arm-joint torque longer than STALL_SECONDS, else None."""
	saturated = (np.abs(np.asarray(torques)[:, :5]) >= STALL_FRACTION * limit).any(axis=1)
	run = 0
	for i, flag in enumerate(saturated):
		run = run + 1 if flag else 0
		if run / fps > STALL_SECONDS:
			return i - run + 1
	return None


def node_at(events, frame):
	"""Skill node running at active-branch ``frame``, replaying node and rewound events of the trace."""
	timeline = []
	for e in events:
		if e["kind"] == "rewound":
			timeline = [t for t in timeline if t[0] <= e["frame"]]
		elif e["kind"] == "node" and e["node"] != "recover":
			timeline.append((e["frame"], e["node"]))
	return next((node for f, node in reversed(timeline) if f <= frame), None)


def rewind_options(checkpoints, failure, stall_frame, stall_node):
	"""Ordered ``(choice, note)`` pairs for a rewind. The first entry is the deterministic default: the latest
	checkpoint taken before the stall began (a checkpoint after it keeps the stall in the health history, so it can
	never clear it) with the variant recommended for the failure, then the plain rewind of that checkpoint."""
	reasons = failure.get("reason_codes") or []
	recommended = VARIANT_BY_CODE.get(failure.get("code"))
	if stall_frame is not None and any(r in STALL_REASONS for r in reasons):
		recommended = STALL_VARIANT_BY_NODE.get(stall_node, "cautious")
	latest = list(reversed(checkpoints))
	if stall_frame is not None:
		clean = [c for c in latest if c.frame <= stall_frame]
		latest = clean + [c for c in latest if c not in clean]
	options = []
	for c in latest:
		hint = ""
		if stall_frame is not None and c.frame > stall_frame:
			hint = f"; keeps the stall that began at frame {stall_frame}"
		variants = [recommended] if recommended else []
		variants += ["cautious"] if recommended not in (None, "cautious") else []
		for v in variants:
			options.append((f"{c.id}+{v}", f"rewind to {c.id} (frame {c.frame}) with {VARIANT_NOTES[v]}{hint}"))
		options.append((c.id, f"rewind to {c.id} (frame {c.frame}) and retry with unchanged parameters{hint}"))
	return options


class TwinMujoco:
	scan_peak_speed = SCAN_PEAK_SPEED
	recovery_variants: tuple = ()

	def __init__(self, layout, scorer, record, *, llm=None, bottom_threshold=None, crop_size=256, fault=None):
		self.layout = layout
		self.scorer = scorer
		self.record = record
		self.llm = llm
		self.bottom_threshold = bottom_threshold
		self.crop_size = crop_size
		self.fault = fault  # Internal acceptance hook. It is absent from the public MCP API.
		self.spec = load_physics_spec()
		base = next(c for c in SENSITIVITY_MATRIX if c.name == "baseline_2x2")
		cfg = NonIdealityCfg(
			latency_frames=base.latency_frames,
			backlash_rad=math.radians(base.backlash_deg),
			velocity_limit_rad_s=self.spec.arm_actuator.velocity_limit,
			encoder_ticks=4096,
		)
		self.cell = Cell(
			layout, SO101_MJCF_DIR, physics=self.spec, dynamic_clutter=True, assisted_grasp=False, face_probes=True
		)
		self.ep = PhysicsEpisode(self.cell, INITIAL_Q, layout.brick_xy, layout.brick_yaw_deg, cfg)
		self.expert = RelocateExpert(
			self.cell, INITIAL_Q, layout.brick_xy, ExpertParams(), RelocateParams(do_bottom=False, do_scan=False)
		)
		self.node = ""
		self.phases: list[int] = []
		self.checkpoints: list[Checkpoint] = []
		self.branch = 0
		self.rewinds = 0
		self.total_frames = 0
		self.wrist_count = 0
		self.bottom_count = 0
		self.held: tuple[np.ndarray, float] | None = None
		self.actual_destination = None
		self.execution_error = None
		self.record.start(self.cell)
		self.record.initial(self.cell, INITIAL_Q)

	def close(self):
		self.cell.close()

	def event(self, kind, **payload):
		self.record.event(kind, self.node, self.branch, self.ep.n_frames, **payload)

	def enter(self, node):
		self.node = node
		self.event("node")

	def _execute(self, generator):
		expert = self.expert
		expert._gen = generator
		expert.done = False
		while self.total_frames < MAX_TOTAL_FRAMES:
			obs = observe(self.cell, self.ep)
			target, phase = expert.act(obs)
			if expert.done:
				return None
			# Observation at t is paired with the next unfiltered policy command.
			self.record.command(obs["q"], target, phase)
			self.ep.step(target)
			self.total_frames += 1
			self.phases.append(PHASES.index(phase))
			self.record.step(self.cell, self.node)
			if self.fault and self.fault(self.node, self.branch, self.ep.n_frames):
				expert._gen.close()
				return "INJECTED_MOTION_FAILURE"
		expert._gen.close()
		return "ACTION_BUDGET_EXHAUSTED"

	def _save(self, views):
		if self.held is not None:
			return
		integration = np.empty(mujoco.mj_stateSize(self.cell.model, STATE_SPEC))
		mujoco.mj_getState(self.cell.model, self.cell.data, integration, STATE_SPEC)
		ex, filt = self.expert, self.ep.filt
		skill_state = {
			"last": ex.last.tolist(),
			"active_open": ex._active_open_g,
			"frame_idx": ex._frame_idx,
			"monitor_position": ex.gripper_monitor._prev_meas,
			"monitor_count": ex.gripper_monitor._run,
		}
		filter_state = {
			"queue": [q.tolist() for q in filt._queue],
			"prev": filt._prev_out.tolist(),
			"backlash": filt._backlash_pos.tolist(),
		}
		checkpoint = Checkpoint(
			f"checkpoint-{len(self.checkpoints)}",
			self.branch,
			self.ep.n_frames,
			integration,
			skill_state,
			filter_state,
			self.wrist_count,
			self.bottom_count,
			copy.deepcopy(views),
		)
		self.record.checkpoint(checkpoint)
		self.checkpoints.append(checkpoint)
		self.event("checkpoint", checkpoint=checkpoint.id)

	def prepare(self):
		def ready():
			yield from self.expert._hold(self.expert.p.rest_s, "rest")
			yield from self.expert._go_ready()

		error = self._execute(ready())
		if not error:
			self._save([])
		return error

	def can_scan_wrist(self, policy):
		self.policy = policy  # read by decision_context for the remaining-budget fields
		return self.wrist_count < policy.max_wrist_views

	def can_present_bottom(self, policy):
		self.policy = policy
		return self.bottom_count < policy.max_bottom_views

	def _geometry(self):
		cell = self.cell
		m, d = cell.model, cell.data

		def rounded(a):
			return np.asarray(a).round(5).tolist()

		# World poses of all simulator bodies. Repeated meshes such as the bowl-wall wedges share
		# a parent, so their 336 constant local transforms need not be retransmitted at every decision.
		bodies = [
			[m.body(i).name, i, int(m.body_parentid[i]), *rounded(d.xpos[i]), *rounded(d.xquat[i][[1, 2, 3, 0]])]
			for i in range(m.nbody)
		]

		return {
			"brick_size_m": self.layout.brick_size,
			"brick_position_m": rounded(cell.brick_pos()),
			"brick_rotation_xyzw": rounded(cell.brick_quat_xyzw()),
			"joints_rad": rounded(cell.joints()),
			"tcp_position_m": rounded(cell.tcp_pos()),
			"grasped": self.held is not None,
			"joint_limits_rad": rounded(cell.jnt_range),
			"body_columns": ["name", "body_id", "parent_id", "xyz_m", "quaternion_xyzw"],
			"bodies": bodies,
			"bowl_radius_m": BOWL_R,
			"collision_model": "SO-101 / R6_box_floor_wedge_walls",
			"bowl_positions_m": {"green": self.layout.green, "pink": self.layout.pink},
			"cameras": {
				m.camera(i).name: {
					"position_m": rounded(d.cam_xpos[i]),
					"rotation": rounded(d.cam_xmat[i]),
					"intrinsic": rounded(m.cam_intrinsic[i]),
					"fovy_deg": float(m.cam_fovy[i]),
				}
				for i in range(m.ncam)
			},
			"pinch_normal_n": rounded(self.ep.last("pinch_normal")) if self.ep.n_frames else [0.0, 0.0],
		}

	SKILL_NOTES: ClassVar[dict[str, str]] = {
		"scan_wrist": "move the wrist camera over the brick to read the side and top faces (0-4) not yet cleared",
		"present_bottom": "grasp the brick and show its bottom face (5) to the top camera",
		"stop": "give up: the episode ends as a failure with the brick unsorted",
	}

	def decision_context(self, allowed, views, failures, notes=None):
		"""Compact, leak-checked state for the LLM: coverage, per-view scores, failures, budgets, checkpoints."""
		covered = decision(views)[1]
		cell = self.cell
		brick = cell.brick_pos()
		policy = getattr(self, "policy", None)  # set by whoever owns the Policy; absent means limits are unknown
		budget = {
			"wrist_views_used": self.wrist_count,
			"bottom_views_used": self.bottom_count,
			"rewinds_used": self.rewinds,
			"frames_left": MAX_TOTAL_FRAMES - self.total_frames,
		}
		if policy is not None:
			budget["wrist_views_left"] = policy.max_wrist_views - self.wrist_count
			budget["bottom_views_left"] = policy.max_bottom_views - self.bottom_count
			budget["rewinds_left"] = policy.max_rewinds - self.rewinds
		context = {
			"covered_faces": sorted(covered),
			"missing_faces": [f for f in range(6) if f not in covered],
			"views": [
				{
					"id": v["id"],
					"camera": v["camera"],
					"faces": v["faces"],
					"score": None if v["score"] is None else round(v["score"], 3),
					"threshold": round(v["threshold"], 3),
					"margin": None if v["score"] is None else round(v["score"] - v["threshold"], 3),
				}
				for v in views
			],
			"failures": [{k: f[k] for k in ("code", "skill", "reason_codes") if k in f} for f in failures],
			"budget": budget,
			"grasped": self.held is not None,
			"brick": {"xy_m": np.asarray(brick[:2]).round(3).tolist(), "z_m": round(float(brick[2]), 3)},
		}
		described = {k: self.SKILL_NOTES[k] for k in allowed if k in self.SKILL_NOTES}
		checkpoints = []
		for c in self.checkpoints:
			if c.id in allowed:
				faces = sorted(decision(c.views)[1])
				checkpoints.append({"id": c.id, "frame": c.frame, "faces_covered": faces})
				described[c.id] = f"rewind to frame {c.frame} with faces {faces} covered"
		if checkpoints:
			context["checkpoints"] = checkpoints
		context["choice_notes"] = {**described, **{k: str(v) for k, v in (notes or {}).items() if k in allowed}}
		# Nothing about the defect, its face or the label may reach the model.
		assert_no_defect_leakage(context)
		return context

	def _note_llm(self, source, info):
		stats = getattr(self, "_llm_stats", None)
		if stats is None:
			stats = self._llm_stats = {
				"consulted": 0,
				"adopted": 0,
				"fallback_reasons": {},
				"prompt_tokens": [],
				"latency_s_total": 0.0,
				"attempts": 0,
			}
		stats["consulted"] += 1
		stats["adopted"] += source == "llm"
		if info.get("reason"):
			stats["fallback_reasons"][info["reason"]] = stats["fallback_reasons"].get(info["reason"], 0) + 1
		stats["prompt_tokens"].append(int(info.get("prompt_tokens") or 0))
		stats["latency_s_total"] = round(stats["latency_s_total"] + float(info.get("latency_s") or 0.0), 4)
		stats["attempts"] += int(info.get("attempts") or 0)

	def _llm_summary(self):
		stats = getattr(self, "_llm_stats", None) or {
			"consulted": 0,
			"adopted": 0,
			"fallback_reasons": {},
			"prompt_tokens": [],
			"latency_s_total": 0.0,
			"attempts": 0,
		}
		tokens = stats["prompt_tokens"]
		return {
			**stats,
			"prompt_tokens_median": float(np.median(tokens)) if tokens else None,
			"prompt_tokens_max": max(tokens) if tokens else None,
		}

	def choose(self, allowed, views, failures, notes=None):
		info = {}
		if self.llm is None or len(allowed) == 1 and not failures:
			action, source = allowed[0], "policy"
		else:
			context = self.decision_context(allowed, views, failures, notes)
			action, source, info = select_skill(self.llm, allowed, context)
			self._note_llm(source, info)
		self.event(
			"decision",
			choices=allowed,
			choice=action,
			source=source,
			evidence=[v["id"] for v in views],
			prior_failure=bool(failures),
			fallback_reason=info.get("reason"),
			prompt_tokens=info.get("prompt_tokens"),
			latency_s=info.get("latency_s"),
			attempts=info.get("attempts"),
		)
		return action

	def _capture(self, camera):
		cell = self.cell
		with blind_geometry(cell):
			geometry = capture_view(cell, camera)
		rgb = cell.rgb(camera)
		image, bbox = detector_crop(cell, camera, None, cell.data, self.crop_size, clamp=True, rgb=rgb)
		min_px = BOTTOM_MIN_PX if camera == "top" else WRIST_MIN_PX
		faces = [i for i, n in enumerate(geometry.faces_px) if n >= min_px and ((i == 5) == (camera == "top"))]
		threshold = (
			self.bottom_threshold
			if camera == "top" and self.bottom_threshold is not None
			else self.scorer.threshold(camera)
		)
		score = elapsed_ms = None
		heads = {}
		error = None
		reason = None
		if image is None:
			reason = "NO_CROP"
		elif geometry.brick_px < MIN_BRICK_PX:
			reason = "TOO_FEW_BRICK_PIXELS"
		elif not faces:
			reason = "NO_READABLE_FACE"
		else:
			from ..inspection_scorer import ScoreServerError

			self.event("detector_started", camera=camera)
			started = time.perf_counter()
			try:
				score_details = getattr(self.scorer, "score_with_details", None)
				if score_details is None:
					score = float(self.scorer.score(camera, image))
				else:
					score, heads = score_details(camera, image)
				if not math.isfinite(score):
					score = None
					reason = "NONFINITE_SCORE"
			except ScoreServerError as exc:
				reason = "DETECTOR_UNAVAILABLE"
				error = exc
			elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
		view = {
			"id": f"{self.branch}:{self.ep.n_frames}:{camera}",
			"branch": self.branch,
			"frame": self.ep.n_frames,
			"camera": camera,
			"faces": faces,
			"face_pixels": list(geometry.faces_px),
			"score": score,
			"threshold": float(threshold),
			"crop_bbox": bbox,
		}
		self.record.view(view, rgb, image)
		# Keep diagnostics in the trace, separate from the decision evidence supplied to the agent.
		self.event(
			"view" if error is None else "detector_error",
			observation=view,
			detector={"head_scores": heads, "elapsed_ms": elapsed_ms, "reason": reason},
		)
		if error is not None:
			raise error
		return view

	def _straight_clear(self, first, last, *, held=False):
		self.expert.kin.data.qpos[:] = self.cell.data.qpos
		# Sample the swept arm and bowl/clutter clearance in addition to table height.
		qs = [first + a * (last - first) for a in np.linspace(0, 1, 10)]
		return path_min_z(self.cell, first, last, carried=held) >= 0.015 and self.expert._clearance(qs) >= 0

	def _park_held(self):
		q_lift, roll = self.held
		ex = self.expert
		if not self._straight_clear(ex.last, q_lift, held=True):
			return "PARK_PATH_BLOCKED"

		def park():
			yield from ex._carry(ex._move(q_lift, ex._carry_secs(q_lift, ex.p.show_move_s), "show_move"))
			if not ex.lost:
				yield from ex._put_down(np.asarray(self.layout.brick_xy), roll, q_lift)
			if not ex.lost:
				yield from ex._go_ready()

		error = self._execute(park())
		if error or ex.lost:
			return error or "BRICK_LOST"
		self.held = None
		return None

	def _wrist(self, previous, views):
		self.wrist_count += 1
		if self.held is not None:
			error = self._park_held()
			if error:
				return error
		ex, cell = self.expert, self.cell
		ex.kin.data.qpos[:] = cell.data.qpos
		modes = (
			{"heights": (0.06, 0.075, 0.09), "offset_m": 0.03},
			{"heights": (0.07, 0.085, 0.10), "offset_m": 0.02},
			{"heights": (0.055, 0.08, 0.095), "offset_m": 0.04},
		)
		with blind_geometry(cell):
			candidates, mode = plan_table_scan(
				cell,
				ex.kin,
				ex.last,
				gripper=ex.p.open_g,
				min_px=WRIST_MIN_PX,
				**modes[min(self.wrist_count - 1 + self.rewinds, 2)],
			)
		self.event("scan_plan", mode=mode, candidates=len(candidates))
		for candidate in candidates:
			_, covered = decision([*previous, *views])
			if not any(f not in covered and candidate.faces_px[f] >= WRIST_MIN_PX for f in range(5)):
				continue
			if not self._straight_clear(ex.last, candidate.q):
				continue

			def look(q):
				move_time = max(
					ex.rp.scan_move_s, 1.875 * float(np.max(np.abs(q[:5] - ex.last[:5]))) / self.scan_peak_speed
				)
				yield from ex._move(q, move_time, "approach")
				yield from ex._hold_servo(q, ex.rp.scan_hold_s, "approach")

			if error := self._execute(look(candidate.q)):
				return error
			views.append(self._capture("wrist_cam"))
			if decision(views)[0] == "defective":
				break
		if error := self._execute(ex._go_ready()):
			return error
		self._save([*previous, *views])
		return None

	def _grasp(self):
		ex = self.expert
		ex.grasp_retries = ex.block_replans = 0
		ex._active_open_g = float(ex.p.open_g)

		def grasp_and_lift():
			grasp = yield from ex._grasp()
			if grasp is not None:
				q_lift = yield from ex._lift(grasp)
				self.held = (q_lift, grasp.roll)

		error = self._execute(grasp_and_lift())
		return error or ("BRICK_LOST" if ex.lost else ex.aborted)

	def _bottom(self, views):
		self.bottom_count += 1
		if self.held is None and (error := self._grasp()):
			return error
		ex, cell = self.expert, self.cell
		with blind_geometry(cell):
			candidate = plan_bottom_view(
				cell,
				ex.last,
				gripper=ex.p.hold_g,
				seed=self.rewinds * 71 + self.bottom_count - 1,
				clearance=lambda q: self._straight_clear(ex.last, q, held=True),
			)
		if candidate is None or candidate.faces_px[5] < BOTTOM_MIN_PX:
			self.event("view_unavailable", camera="top", reason="NO_BOTTOM_PATH")
			return None

		def look():
			pose = candidate.q.copy()
			yield from ex._carry(ex._move(pose, ex._carry_secs(pose, ex.p.show_move_s), "show_move"))
			yield from ex._carry(ex._hold_servo(pose, ex.rp.bottom_hold_s, "show_hold"))

		if error := self._execute(look()):
			return error
		if ex.lost:
			return "BRICK_LOST"
		views.append(self._capture("top"))
		return None

	def inspect(self, node, previous):
		views = []
		try:
			error = self._wrist(previous, views) if node == "scan_wrist" else self._bottom(views)
		except RuntimeError as exc:
			# The scoring process can stop independently of simulation. Keep the physical state and trace.
			from ..inspection_scorer import ScoreServerError

			if not isinstance(exc, ScoreServerError):
				raise
			error = "DETECTOR_UNAVAILABLE"
		return views, error

	def failure(self, reason, views):
		self.execution_error = reason
		failed_node = next(
			(e["node"] for e in reversed(self.record.events) if e["kind"] == "node" and e["node"] != "recover"), None
		)

		payload = {
			"code": reason,
			"skill": failed_node,
			"frame": self.ep.n_frames,
			"branch": self.branch,
			"prior_views": [{"id": v["id"], "faces": v["faces"], "score": v["score"]} for v in views],
		}
		if reason == "PHYSICAL_HEALTH_FAILED":
			payload["reason_codes"] = next(
				event["reason_codes"]
				for event in reversed(self.record.events)
				if event["kind"] == "motion_health" and event["branch"] == self.branch
			)
		self.event("failure", **{k: v for k, v in payload.items() if k not in ("frame", "branch")})
		return payload

	def rewind(self, policy: Policy, failures):
		self.policy = policy
		if not self.checkpoints or self.rewinds >= policy.max_rewinds or self.total_frames >= MAX_TOTAL_FRAMES:
			return None
		stall = stall_start(self.ep.telemetry()["actuator_torques"], self.spec.arm_actuator.effort_limit)
		stall_node = None if stall is None else node_at(self.record.events, stall)
		options = rewind_options(self.checkpoints, failures[-1], stall, stall_node)
		actions = [choice for choice, _ in options] + ["stop"]
		notes = {**dict(options), "stop": "give up and report the failure"}
		choice = self.choose(actions, self.checkpoints[-1].views, failures, notes=notes)
		if choice == "stop":
			return None
		checkpoint_id, _, variant = choice.partition("+")
		checkpoint = next(c for c in self.checkpoints if c.id == checkpoint_id)
		self.rewinds += 1
		self.branch = self.rewinds
		self.record.fork(self.branch, checkpoint)
		m, d = self.cell.model, self.cell.data
		mujoco.mj_setState(m, d, checkpoint.integration, STATE_SPEC)
		mujoco.mj_forward(m, d)
		mujoco.mj_setState(m, d, checkpoint.integration, STATE_SPEC)
		for rows in self.ep._logs.values():
			del rows[checkpoint.frame :]
		del self.ep.jaw_env_force[checkpoint.frame :]
		del self.phases[checkpoint.frame :]
		filt = self.ep.filt
		filt._queue = deque((np.array(q) for q in checkpoint.filter["queue"]), maxlen=filt._queue.maxlen)
		filt._prev_out = np.array(checkpoint.filter["prev"])
		filt._backlash_pos = np.array(checkpoint.filter["backlash"])
		params, relocate = self.expert.p, self.expert.rp
		if variant:
			# Variants transform the first attempt's defaults, so they never stack on an earlier variant.
			self._base_params = getattr(self, "_base_params", (params, relocate))
			params, relocate, self.scan_peak_speed = apply_variant(variant, *self._base_params)
		self.recovery_variants = (*self.recovery_variants, variant or "none")
		self.expert = ex = RelocateExpert(self.cell, INITIAL_Q, self.layout.brick_xy, params, relocate)
		ex.last = np.array(checkpoint.expert["last"])
		ex._active_open_g = checkpoint.expert["active_open"]
		ex._frame_idx = checkpoint.expert["frame_idx"]
		ex.gripper_monitor._prev_meas = checkpoint.expert["monitor_position"]
		ex.gripper_monitor._run = checkpoint.expert["monitor_count"]
		self.held = None
		self.actual_destination = None
		self.execution_error = None
		self.wrist_count = checkpoint.wrist_count
		self.bottom_count = checkpoint.bottom_count
		del self.checkpoints[self.checkpoints.index(checkpoint) + 1 :]
		self.record.restored(self.cell)
		self.event(
			"rewound",
			parent_branch=checkpoint.branch,
			checkpoint=checkpoint.id,
			reason=failures[-1]["code"],
			variant=variant or None,
		)
		return copy.deepcopy(checkpoint.views)

	def place(self, label):
		if label == "unknown":
			self.event("abstain", destination="inspection_table")
			error = self._park_held() if self.held is not None else None
			if error:
				return error
			position = self.cell.brick_pos()
			if (
				np.linalg.norm(position[:2] - self.layout.brick_xy) > 0.03
				or position[2] > self.layout.brick_size[2] / 2 + 0.01
			):
				return "INSPECTION_TABLE_PARK_UNVERIFIED"
			self.actual_destination = "inspection_table"
			return None
		if self.held is None and (error := self._grasp()):
			return error
		ex = self.expert
		q_lift, _ = self.held
		ex.bowl_xy = np.array(self.layout.pink if label == "defective" else self.layout.green)
		if not self._straight_clear(ex.last, q_lift, held=True):
			return "PLACE_RETURN_PATH_BLOCKED"

		def release():
			if np.max(np.abs(ex.last - q_lift)) > 0.01:
				yield from ex._carry(ex._move(q_lift, ex._carry_secs(q_lift, ex.p.show_move_s), "show_move"))
			if not ex.lost:
				yield from ex._place(q_lift)
			if not ex.lost:
				yield from ex._go_home()

		error = self._execute(release())
		if error or ex.lost:
			return error or "BRICK_LOST"
		self.held = None
		position = self.cell.brick_pos()
		for name, xy in (("green", self.layout.green), ("pink", self.layout.pink)):
			if np.linalg.norm(position[:2] - xy) < BOWL_R and position[2] < 0.055:
				self.actual_destination = name
				break
		if self.actual_destination != ("pink" if label == "defective" else "green"):
			return "PLACEMENT_UNVERIFIED"
		health = self._health(ex.bowl_xy)
		if health["verdict"] != "SUCCESS":
			self.event("motion_health", reason_codes=health["reason_codes"])
			return "PHYSICAL_HEALTH_FAILED"
		return None

	def evaluate(self, label, views, failures):
		# Ground truth enters only after the graph has selected and executed its final action.
		expected = "defective" if self.layout.defective else "normal"
		target = self.layout.pink if self.layout.defective else self.layout.green
		health = self._health(target) if self.ep.n_frames and label != "unknown" else {}
		ok = label == expected and self.actual_destination == ("pink" if self.layout.defective else "green")
		result = {
			"verdict": "SUCCESS"
			if ok and health.get("verdict") == "SUCCESS"
			else "UNKNOWN"
			if label == "unknown" and not self.execution_error
			else "FAILED",
			"label": label,
			"label_correct": label == expected,
			"destination": self.actual_destination,
			"active_branch": self.branch,
			"rewinds": self.rewinds,
			"recovery_variants": list(self.recovery_variants),
			"failures": failures,
			"active_views": len(views),
			"views": sum(e["kind"] == "view" for e in self.record.events),
			"sim_frames": self.total_frames,
			"active_frames": self.ep.n_frames,
			"evidence": [v["id"] for v in views],
			"cleared_faces": sorted(decision(views)[1]),
			"health": health,
			"llm": self._llm_summary(),
			"student_eligible": bool(ok and health.get("verdict") == "SUCCESS" and not self.rewinds),
			"placement_evidence": next(
				(v["id"] for v in views if v["score"] is not None and v["score"] >= v["threshold"]), None
			),
		}
		self.record.audit(
			{"true_label": expected, "true_face": self.layout.defect_face if self.layout.defective else None}
		)
		return result

	def _health(self, target):
		tel = self.ep.telemetry()
		return evaluate_episode_health(
			q_cmd=tel["q_cmd"],
			q_act=tel["qpos"],
			tcp_pos=tel["tcp_pos"],
			brick_pos=tel["brick_pos"],
			brick_quat=tel["brick_quat"],
			jaw_forces=tel["jaw_forces"],
			unintended_forces=tel["unintended_forces"],
			phases=np.asarray(self.phases),
			target_bowl_xy=target,
			bowl_r=BOWL_R,
			brick_vel=tel["brick_vel"],
			brick_ang_vel=tel["brick_ang_vel"],
			ee_quat=tel["ee_quat"],
			fps=FPS,
			brick_size=self.layout.brick_size,
			substep_unintended_forces=tel["substep_unintended"],
			pinch_normal_forces=tel["pinch_normal"],
			pinch_tangential_forces=tel["pinch_tangential"],
			actuator_torques=tel["actuator_torques"],
			torque_limits=np.array(
				[self.spec.arm_actuator.effort_limit] * 5 + [self.spec.gripper_actuator.effort_limit]
			),
			torque_source="solver_measured",
			timestamps=tel["t"],
		).to_dict()


def file_digest(path):
	return hashlib.sha256(Path(path).read_bytes()).hexdigest()
