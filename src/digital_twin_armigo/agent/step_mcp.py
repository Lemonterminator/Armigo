"""Step-by-step MCP tools: an agent drives one simulated six-face episode itself, skill by skill.

``armigo-mcp serve`` runs whole episodes from a policy. Here the client is the orchestrator: it looks, chooses the
next skill, recovers from failures and decides the label. Two tool sets share the same skills and grading:

``omniscient``
	The skills read the simulator, as in ``armigo-mcp run``. ``scene_state`` returns the brick and bowl poses, the
	contact forces and the face visibility. An upper bound, and the teacher for comparisons.
``realistic``
	What a real cell offers. The skills run with ``TwinMujoco(sensing="realistic")``; the observation tools return
	camera images, arm readings, the hand-eye calibration and what perception derives from them. Every payload is
	checked against :data:`SIMULATOR_KEYS` before it leaves the server.

In both, the defect, its face and the true label stay private until ``finish`` grades the episode. Observation
payloads use the dataset's names (joint vector ``observation.state`` in radians, ``top``/``wrist_cam`` images), so
a session trace exports like any other episode.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import math
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from ..inspection import assert_no_defect_leakage
from ..mujoco_cell.inspection_views import capture_view
from .student_rollout import ExpertTracer
from .twin_graph import Policy, decision
from .twin_mujoco import PROFILE, blind_geometry, sample_case
from .twin_record import EpisodeTrace

PROFILES = ("omniscient", "realistic")
# Generous fixed budgets: the agent decides how to use them.
STEP_POLICY = Policy(max_bottom_views=3, max_wrist_views=3, max_rewinds=2)
MAX_SESSIONS = 4
# Keys that only the simulator knows. A realistic payload carrying one of them is a bug, not a feature.
SIMULATOR_KEYS = frozenset(
	{
		"bodies",
		"brick_position_m",
		"brick_rotation_xyzw",
		"bowl_positions_m",
		"jaw_normal_n",
		"pinch_normal_n",
		"true_face_pixels",
		"true_cross_px",
		"true_face_pixels_crop",
		"true_face_pixels_crop_native",
		"true_cross_px_crop",
		"true_cross_px_crop_native",
		"truth_crop",
		"true_brick_quat_xyzw",
		"belief_brick_quat_xyzw",
		"belief_error_mm",
		"error_xy_mm",
		"error_yaw_deg",
		"cam_pos_err_mm",
		"cam_rot_err_deg",
		"bias_deg",
		"size_correct",
		"label_correct",
		"true_label",
		"true_face",
		"destination",
	}
)


class SimulatorLeak(AssertionError):
	pass


def assert_sensor_only(payload: Any, path: str = "root") -> None:
	"""Raise :class:`SimulatorLeak` if ``payload`` holds a simulator-only key, and check it for the defect."""
	if isinstance(payload, dict):
		for key, value in payload.items():
			if str(key) in SIMULATOR_KEYS:
				raise SimulatorLeak(f"simulator key {key!r} at {path}")
			assert_sensor_only(value, f"{path}.{key}")
	elif isinstance(payload, (list, tuple)):
		for i, value in enumerate(payload):
			assert_sensor_only(value, f"{path}[{i}]")
	if path == "root":
		assert_no_defect_leakage(payload)


def _round(a, digits=5):
	return np.asarray(a, dtype=np.float64).round(digits).tolist()


def _png(pixels: np.ndarray) -> bytes:
	import cv2

	img = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR) if pixels.ndim == 3 else pixels
	ok, buf = cv2.imencode(".png", img)
	if not ok:
		raise OSError("PNG encoding failed")
	return buf.tobytes()


class StepSession:
	"""One episode driven step by step. Not thread-safe: the server runs every session on one simulation thread."""

	def __init__(self, case, profile: str, scorer, path: Path, *, crop_size=256, bottom_threshold=None):
		if profile not in PROFILES:
			raise ValueError(f"profile must be one of {PROFILES}")
		self.profile = profile
		self.realistic = profile == "realistic"
		self.policy = STEP_POLICY
		self.trace = EpisodeTrace(path, video=False)
		self.case = case
		# Segment-labelling expert (student_rollout.ExpertTracer): the recorded frames carry their motion segment.
		self.backend = ExpertTracer(
			sample_case(*case),
			scorer,
			self.trace,
			crop_size=crop_size,
			bottom_threshold=bottom_threshold,
			sensing="realistic" if self.realistic else "truth",
		)
		self.views: list[dict] = []
		self.failures: list[dict] = []
		self.pending: dict | None = None  # the failure waiting for recover()
		self.label: str | None = None
		self.placed = False
		self.stopped = False
		self.result: dict | None = None
		self.steps = 0
		self._run("prepare", self.backend.prepare)

	# -- helpers ----------------------------------------------------------------------------------------------------

	def _check_open(self, *, allow_pending=False):
		if self.result is not None:
			raise ValueError("the episode is finished")
		if self.pending is not None and not allow_pending:
			raise ValueError("a failure is pending: call recover with one of its choices first")
		if self.stopped:
			raise ValueError("recovery was stopped: only finish is left")

	def _run(self, node: str, fn, *args):
		self.steps += 1
		self.backend.enter(node)
		error = fn(*args)
		if error:
			self.pending = self.backend.failure(error, self.views)
			self.failures.append(self.pending)
			if not self.backend.recovery_choices(self.policy, self.failures):
				self.pending, self.stopped = None, True
		return error

	def _view(self, v: dict) -> dict:
		return {
			"id": v["id"],
			"camera": v["camera"],
			"faces": v["faces"],
			"face_pixels": v["face_pixels"],
			"score": None if v["score"] is None else round(v["score"], 4),
			"threshold": round(v["threshold"], 4),
		}

	def _out(self, payload: dict) -> dict:
		payload = {"session_state": self.state(), **payload}
		if self.realistic:
			assert_sensor_only(payload)
		else:
			assert_no_defect_leakage(payload)
		return payload

	# -- observations shared by both profiles ----------------------------------------------------------------------

	def state(self) -> dict:
		"""Skill-level progress: views and their scores, cleared faces, budgets, pending failure and its choices."""
		b = self.backend
		label, covered = decision(self.views)
		out = {
			"profile": self.profile,
			"frame": b.ep.n_frames,
			"node": b.node,
			"views": [self._view(v) for v in self.views],
			"cleared_faces": sorted(covered),
			"evidence_label": label,  # what the views alone support: defective / normal / unknown
			"budget": {
				"wrist_scans_left": self.policy.max_wrist_views - b.wrist_count,
				"bottom_views_left": self.policy.max_bottom_views - b.bottom_count,
				"recoveries_left": self.policy.max_rewinds - b.recoveries,
			},
			"holding_brick": b.held is not None,
			"placed": self.placed,
			"finished": self.result is not None,
		}
		if self.pending is not None:
			out["pending_failure"] = {k: self.pending[k] for k in ("code", "skill") if k in self.pending}
			out["recovery_choices"] = dict(b.recovery_choices(self.policy, self.failures))
		if self.stopped:
			out["recovery_stopped"] = True
		return out

	def robot_state(self) -> dict:
		"""Joint readings (rad), gripper position and effort, and the gripper frame from forward kinematics."""
		b = self.backend
		q = b.cell.joints()
		ep = b.ep
		effort = 0.0
		if ep.n_frames:
			effort = abs(float(ep.last("actuator_torques")[5])) / b.spec.gripper_actuator.effort_limit
		grip = b._grip_pose()
		out = {
			"observation.state": _round(q),
			"joint_names": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"],
			"gripper_position_rad": round(float(q[5]), 5),
			"gripper_command_rad": round(float(ep.last("q_cmd")[5]) if ep.n_frames else float(q[5]), 5),
			"gripper_effort_frac": round(effort, 4),
			"tcp_position_m": _round(grip[:3, 3]),
			"tcp_rotation": _round(grip[:3, :3]),
		}
		if not self.realistic:
			out["jaw_normal_n"] = _round(ep.last("pinch_normal")) if ep.n_frames else [0.0, 0.0]
		return out

	def capture(self, camera: str, kind: str) -> tuple[dict, bytes]:
		"""A camera image: ``rgb`` (8-bit PNG) or ``depth`` (16-bit PNG, millimetres along the optical axis, 0 =
		invalid; noisy like a D415 in ``realistic``)."""
		b = self.backend
		if camera not in ("top", "wrist_cam") or kind not in ("rgb", "depth"):
			raise ValueError("camera is top or wrist_cam; kind is rgb or depth")
		if kind == "rgb":
			pixels = b.cell.rgb(camera)
		else:
			if self.realistic and camera == "top":
				depth = b._top_depth()
			else:
				r = b.cell.renderer
				r.enable_depth_rendering()
				try:
					r.update_scene(b.cell.data, camera)
					depth = r.render().astype(np.float64)
				finally:
					r.disable_depth_rendering()
			mm = np.where(np.isfinite(depth) & (depth > 0) & (depth < 5.0), depth * 1000.0, 0.0)
			pixels = np.clip(np.rint(mm), 0, 65535).astype(np.uint16)
		cid = b.cell.camera_id(camera)
		meta = {"camera": camera, "kind": kind, "shape": list(pixels.shape), "frame": b.ep.n_frames}
		meta["intrinsics"] = _round(b._cam_k if self.realistic and camera == "top" else b.cell.model.cam_intrinsic[cid])
		return meta, _png(pixels)

	def view_image(self, view_id: str, crop: bool) -> bytes:
		if view_id not in {v["id"] for v in self.views} | {
			e["observation"]["id"] for e in self.trace.events if e["kind"] == "view"
		}:
			raise ValueError("unknown view id")
		path = self.trace.keyframes / f"{view_id.replace(':', '-')}-{'crop' if crop else 'frame'}.png"
		if not path.is_file():
			raise ValueError("no image was stored for this view")
		return path.read_bytes()

	# -- skills ----------------------------------------------------------------------------------------------------

	def scan_wrist(self) -> dict:
		self._check_open()
		if self.placed or not self.backend.can_scan_wrist(self.policy):
			raise ValueError("no wrist scan is left, or the brick is already placed")
		return self._inspect("scan_wrist")

	def present_bottom(self) -> dict:
		self._check_open()
		if self.placed or not self.backend.can_present_bottom(self.policy):
			raise ValueError("no bottom view is left, or the brick is already placed")
		return self._inspect("present_bottom")

	def _inspect(self, node: str) -> dict:
		new: list[dict] = []

		def run():
			views, error = self.backend.inspect(node, self.views)
			new.extend(views)
			return error

		error = self._run(node, run)
		self.views.extend(new)
		return {"new_views": [self._view(v) for v in new], "error": error}

	def place(self, label: str) -> dict:
		self._check_open()
		if label not in ("defective", "normal", "unknown"):
			raise ValueError("label is defective, normal or unknown")
		if self.placed:
			raise ValueError("the brick is already placed")
		self.label = label
		error = self._run("place", self.backend.place, label)
		self.placed = not error
		return {"error": error}

	def recover(self, choice: str) -> dict:
		self._check_open(allow_pending=True)
		if self.pending is None:
			raise ValueError("there is no failure to recover from")
		choices = dict(self.backend.recovery_choices(self.policy, self.failures))
		if choice not in choices:
			raise ValueError(f"choice must be one of {sorted(choices)}")
		self.backend.enter("recover")
		self.backend.forced_choice = choice
		restored = self.backend.rewind(self.policy, self.failures)
		self.pending = None
		if restored is None:
			self.stopped = True
			return {"recovered": False}
		self.views = restored
		self.placed = False
		self.label = None
		return {"recovered": True}

	def finish(self, service_provenance: dict) -> dict:
		if self.result is not None:
			return self.result
		b = self.backend
		label = self.label if self.placed else "unknown"
		b.enter("evaluate")
		self.result = b.evaluate(label, self.views, self.failures)
		self.result["agent_steps"] = self.steps
		self.trace.finish(
			self.result,
			{"engine": "mujoco", "camera_profile": PROFILE, "step_profile": self.profile, **service_provenance},
		)
		b.close()
		return self.result

	def close(self):
		if self.result is None:
			self.trace.close()
			self.backend.close()

	# -- profile-specific observations ----------------------------------------------------------------------------

	def scene_state(self) -> dict:
		"""omniscient: the simulator's brick, bowls, contact forces and face visibility in both cameras."""
		if self.realistic:
			raise ValueError("scene_state is not part of the realistic tool set")
		b, cell = self.backend, self.backend.cell
		with blind_geometry(cell):
			faces = {cam: list(capture_view(cell, cam).faces_px) for cam in ("top", "wrist_cam")}
		return {
			"brick_size_m": list(b.layout.brick_size),
			"brick_position_m": _round(cell.brick_pos()),
			"brick_rotation_xyzw": _round(cell.brick_quat_xyzw()),
			"bowl_positions_m": {"green": list(b.layout.green), "pink": list(b.layout.pink)},
			"bowl_radius_m": 0.10,
			"face_pixels": faces,
			"jaw_normal_n": _round(b.ep.last("pinch_normal")) if b.ep.n_frames else [0.0, 0.0],
		}

	def camera_calibration(self) -> dict:
		"""realistic: the top camera's intrinsics and the hand-eye estimate of its pose in the robot frame."""
		b = self.backend
		event = next(e for e in self.trace.events if e["kind"] == "camera_calibrated")
		return {
			"camera": "top",
			"intrinsics": _round(b._cam_k),
			"world_from_camera": _round(b._cam_est),
			"convention": "OpenCV (x right, y down, z forward); robot base frame",
			"reprojection_rms_px": event["rms_px"],
			"poses_used": event["poses"],
		}

	def locate_brick(self) -> dict:
		"""realistic: locate the brick on the table in a fresh top depth image."""
		b = self.backend
		if b.held is not None:
			return {"found": False, "reason": "the brick is in the gripper"}
		if b._perceive():
			return {"found": False, "reason": "no brick-sized blob on the table"}
		est = b.brick_est
		return {
			"found": True,
			"position_m": _round(est.pos),
			"yaw_deg": round(math.degrees(est.yaw), 3),
			"size_class": "2x4" if est.size[0] != est.size[1] else "2x2",
			"size_m": list(est.size),
			"footprint_mm": [round(v * 1000, 2) for v in est.footprint_m],
			"occlusion_correction_mm": list(est.extended_mm),
		}

	def locate_bowls(self) -> dict:
		"""realistic: the bowl centres found in the top image at the start of the session."""
		return {"bowls_m": {n: _round(xy) for n, xy in self.backend.bowls.items()}, "found_at": "session start"}


class StepService:
	def __init__(self, out: Path, detector: Path, profile: str, cases, *, scorer=None):
		if profile not in PROFILES:
			raise ValueError(f"profile must be one of {PROFILES}")
		self.out = Path(out).resolve()
		self.detector = Path(detector).resolve()
		spec = json.loads(self.detector.read_text(encoding="utf-8"))
		self.crop_size = int(spec["crop_size"])
		self.profile = profile
		self.cases = {uuid.uuid4().hex: tuple(c) for c in cases}
		self.scorer = scorer
		self.sessions: dict[str, StepSession] = {}
		self.sim = ThreadPoolExecutor(max_workers=1, thread_name_prefix="armigo-step")  # one GL context, one thread

	async def call(self, fn, *args):
		return await asyncio.wrap_future(self.sim.submit(fn, *args))

	def session(self, session_id: str) -> StepSession:
		if session_id not in self.sessions:
			raise ValueError("unknown session")
		return self.sessions[session_id]

	def start(self, case_id: str) -> dict:
		if case_id not in self.cases:
			raise ValueError("unknown case")
		if sum(s.result is None for s in self.sessions.values()) >= MAX_SESSIONS:
			raise ValueError(f"at most {MAX_SESSIONS} open sessions: finish one first")
		if self.scorer is None:
			from ..inspection_scorer import default_sup_scorer

			self.scorer = default_sup_scorer(self.detector)
		sid = uuid.uuid4().hex
		session = StepSession(
			self.cases[case_id], self.profile, self.scorer, self.out / "sessions" / sid, crop_size=self.crop_size
		)
		self.sessions[sid] = session
		return {"session_id": sid, **session._out({})}

	def finish(self, session_id: str) -> dict:
		session = self.session(session_id)
		result = session.finish({"detector": str(self.detector), "policy": asdict(session.policy)})
		return {"session_id": session_id, "result": result}

	def run_sync(self, fn, *args):
		"""Run ``fn`` on the simulation thread and wait: every renderer must be created, used and closed there."""
		return self.sim.submit(fn, *args).result()

	def close(self):
		def close_all():
			for s in self.sessions.values():
				s.close()

		self.run_sync(close_all)
		self.sim.shutdown(wait=True)
		if self.scorer is not None and hasattr(self.scorer, "close"):
			self.scorer.close()


TOOLS_COMMON = (
	"list_cases",
	"start_session",
	"session_state",
	"robot_state",
	"capture",
	"view_image",
	"scan_wrist",
	"present_bottom",
	"recover",
	"place",
	"finish",
)
TOOLS_ONLY = {"omniscient": ("scene_state",), "realistic": ("camera_calibration", "locate_brick", "locate_bowls")}


def create_step_mcp(service: StepService, host="127.0.0.1", port=8766):
	from mcp.server.fastmcp import FastMCP, Image

	profile = service.profile
	mcp = FastMCP(f"Armigo Six-Face Steps ({profile})", host=host, port=port)

	def session_tool(method_name: str):
		async def run(session_id: str):
			session = service.session(session_id)
			return session._out(await service.call(getattr(session, method_name)))

		return run

	@mcp.tool()
	def list_cases() -> dict[str, Any]:
		"""Opaque scene ids. The task: inspect the brick on the table for a cross defect on any of its six faces,
		then sort it, defective into the pink bowl and normal into the green bowl (or leave it, as unknown)."""
		return {"case_ids": list(service.cases), "profile": profile, "tools": [*TOOLS_COMMON, *TOOLS_ONLY[profile]]}

	@mcp.tool()
	async def start_session(case_id: str) -> dict[str, Any]:
		"""Open an episode on a case: the arm moves to its ready pose (in realistic, after calibrating the top camera
		and finding the bowls). Returns the session id and its state."""
		return await service.call(service.start, case_id)

	@mcp.tool()
	async def session_state(session_id: str) -> dict[str, Any]:
		"""Views so far with detector scores, cleared faces, the label the evidence supports, budgets, and a pending
		failure with its recovery choices."""
		session = service.session(session_id)
		return session._out({})

	@mcp.tool()
	async def robot_state(session_id: str) -> dict[str, Any]:
		"""Joint readings (rad), gripper position, command and effort, and the gripper frame by forward kinematics."""
		return await session_tool("robot_state")(session_id)

	@mcp.tool()
	async def capture(session_id: str, camera: str = "top", kind: str = "rgb") -> list:
		"""A camera image now: camera top or wrist_cam; kind rgb (PNG) or depth (16-bit PNG, millimetres)."""
		session = service.session(session_id)
		meta, png = await service.call(session.capture, camera, kind)
		meta = session._out(meta)
		return [json.dumps(meta, allow_nan=False), Image(data=png, format="png")]

	@mcp.tool()
	async def view_image(session_id: str, view_id: str, crop: bool = True) -> list:
		"""The image a scored view was taken from: the detector crop, or the full frame."""
		session = service.session(session_id)
		return [Image(data=await service.call(session.view_image, view_id, crop), format="png")]

	@mcp.tool()
	async def scan_wrist(session_id: str) -> dict[str, Any]:
		"""Move the wrist camera around the brick on the table and score its side and top faces (0-4) not yet
		cleared. Parks a carried brick first. Returns the new views and an error code, if any."""
		return await session_tool("scan_wrist")(session_id)

	@mcp.tool()
	async def present_bottom(session_id: str) -> dict[str, Any]:
		"""Grasp the brick (if not held) and show its bottom face (5) to the top camera, then score it."""
		return await session_tool("present_bottom")(session_id)

	@mcp.tool()
	async def recover(session_id: str, choice: str) -> dict[str, Any]:
		"""After a failed skill, apply one of session_state.recovery_choices (stop ends the episode unsorted)."""
		session = service.session(session_id)
		return session._out(await service.call(session.recover, choice))

	@mcp.tool()
	async def place(session_id: str, label: str) -> dict[str, Any]:
		"""Sort the brick: defective (pink bowl), normal (green bowl) or unknown (leave it on the table). The skill
		checks the placement; it does not tell you whether the label was right."""
		session = service.session(session_id)
		return session._out(await service.call(session.place, label))

	@mcp.tool()
	async def finish(session_id: str) -> dict[str, Any]:
		"""End the episode and grade it against the hidden truth (label, destination, motion health)."""
		return await service.call(service.finish, session_id)

	if profile == "omniscient":

		@mcp.tool()
		async def scene_state(session_id: str) -> dict[str, Any]:
			"""Simulator state: brick pose and size, bowl positions, jaw contact forces, visible face pixels."""
			return await session_tool("scene_state")(session_id)

	else:

		@mcp.tool()
		async def camera_calibration(session_id: str) -> dict[str, Any]:
			"""Top camera intrinsics and its hand-eye-calibrated pose in the robot base frame."""
			return await session_tool("camera_calibration")(session_id)

		@mcp.tool()
		async def locate_brick(session_id: str) -> dict[str, Any]:
			"""Locate the brick on the table in a fresh top depth image: position, yaw, size class."""
			return await session_tool("locate_brick")(session_id)

		@mcp.tool()
		async def locate_bowls(session_id: str) -> dict[str, Any]:
			"""Bowl centres found in the top image when the session started."""
			return await session_tool("locate_bowls")(session_id)

	return mcp


def run_scripted(service: StepService, case_ids=None) -> list[dict]:
	"""A scripted client of the step tools (the graph's default policy, wrist scans first), calling the same
	session methods the MCP tools call. Shows the tool set is enough to do the task, and gives the baseline an LLM
	client is compared to."""
	return service.run_sync(_scripted, service, case_ids)


def _scripted(service: StepService, case_ids) -> list[dict]:
	results = []
	for case_id in case_ids or list(service.cases):
		sid = service.start(case_id)["session_id"]
		s = service.session(sid)
		for _ in itertools.count():
			st = s.state()
			if st["finished"] or s.stopped or s.placed:
				break
			if s.pending is not None:
				s.recover(next(iter(st["recovery_choices"])))
				continue
			label = st["evidence_label"]
			wrist = bool(set(range(5)) - set(st["cleared_faces"])) and st["budget"]["wrist_scans_left"] > 0
			bottom = 5 not in st["cleared_faces"] and st["budget"]["bottom_views_left"] > 0
			if label != "unknown" or not (wrist or bottom):
				s.place(label)
			elif wrist:
				s.scan_wrist()
			else:
				s.present_bottom()
		results.append({"case": list(s.case), **service.finish(sid)["result"]})
	return results


def main_step(a, cases):
	service = StepService(a.out, a.detector, a.profile, cases)
	try:
		if a.scripted:
			results = run_scripted(service)
			report = {
				"profile": a.profile,
				"episodes": len(results),
				"successes": sum(r["verdict"] == "SUCCESS" for r in results),
				"labels_correct": sum(r["label_correct"] for r in results),
				"episodes_detail": [
					{
						k: r[k]
						for k in (
							"case",
							"verdict",
							"label",
							"label_correct",
							"rewinds",
							"retries",
							"agent_steps",
							"sim_frames",
						)
					}
					| {"failures": [f["code"] for f in r["failures"]]}
					for r in results
				],
			}
			(Path(a.out) / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
			print(json.dumps({k: v for k, v in report.items() if k != "episodes_detail"}))
			return
		create_step_mcp(service, a.host, a.port).run(transport=a.transport)
	finally:
		service.close()
