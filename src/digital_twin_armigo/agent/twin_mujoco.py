"""Trusted MuJoCo skills and skill-boundary checkpoints for the six-face graph."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..benchmarks import SENSITIVITY_MATRIX
from ..camera import CameraPose, look_from
from ..constants import BOWL_R, FPS, PHASES, REST_U, SO101_MJCF_DIR
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


class TwinMujoco:
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
		return self.wrist_count < policy.max_wrist_views

	def can_present_bottom(self, policy):
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

	def choose(self, allowed, views, failures):
		context = {
			"geometry": self._geometry(),
			"views": views,
			"failures": failures,
			"spent_sim_frames": self.total_frames,
		}
		if self.llm is None or len(allowed) == 1 and not failures:
			action, source = allowed[0], "policy"
		else:
			action, source = select_skill(self.llm, allowed, context)
		self.event(
			"decision",
			choices=allowed,
			choice=action,
			source=source,
			evidence=[v["id"] for v in views],
			prior_failure=bool(failures),
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
				move_time = max(ex.rp.scan_move_s, 1.875 * float(np.max(np.abs(q[:5] - ex.last[:5]))))
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
		if not self.checkpoints or self.rewinds >= policy.max_rewinds or self.total_frames >= MAX_TOTAL_FRAMES:
			return None
		actions = [c.id for c in reversed(self.checkpoints)] + ["stop"]
		choice = self.choose(actions, self.checkpoints[-1].views, failures)
		if choice == "stop":
			return None
		checkpoint = next(c for c in self.checkpoints if c.id == choice)
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
		self.expert = ex = RelocateExpert(
			self.cell,
			INITIAL_Q,
			self.layout.brick_xy,
			self.expert.p,
			self.expert.rp,
		)
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
		self.event("rewound", parent_branch=checkpoint.branch, checkpoint=checkpoint.id, reason=failures[-1]["code"])
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
			"failures": failures,
			"active_views": len(views),
			"views": sum(e["kind"] == "view" for e in self.record.events),
			"sim_frames": self.total_frames,
			"active_frames": self.ep.n_frames,
			"evidence": [v["id"] for v in views],
			"cleared_faces": sorted(decision(views)[1]),
			"health": health,
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
