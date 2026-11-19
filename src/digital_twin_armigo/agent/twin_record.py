"""Frame-aligned camera, robot and LangGraph recordings for the six-face twin."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np

from ..constants import FPS, REST_U, STATE_NAMES, TOP_CROP, TOP_FULL
from ..depth import layered_height_image
from ..joints import joint_normalization, normalized_to_rad
from ..sensor_noise import DEFAULT_D415_NOISE, degrade_depth_height
from .twin_graph import EDGES, NODES, TASK

CAMERAS = ("front", "wrist", "top_rgb", "top_depth")
# Matches twin_mujoco.INITIAL_Q ([rest_u[:5] in rad, ExpertParams().open_g]) without importing mujoco.
INITIAL_Q = np.array([*normalized_to_rad(np.array(REST_U, dtype=np.float64))[:5], 0.85], dtype=np.float64)


def write_json(path, value):
	path = Path(path)
	tmp = path.with_suffix(path.suffix + ".tmp")
	tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
	tmp.replace(path)


class Video:
	def __init__(self, path, shape, *, lossless):
		self.output = av.open(str(path), "w")
		self.stream = self.output.add_stream("ffv1" if lossless else "libx264", rate=FPS)
		self.stream.height, self.stream.width = shape[:2]
		self.stream.pix_fmt = "bgr0" if lossless else "yuv420p"
		if not lossless:
			self.stream.options = {"preset": "ultrafast", "crf": "20", "tune": "zerolatency"}
		self.index = 0

	def put(self, rgb):
		frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
		frame.pts = self.index
		for packet in self.stream.encode(frame):
			self.output.mux(packet)
		self.index += 1

	def close(self):
		try:
			for packet in self.stream.encode():
				self.output.mux(packet)
		finally:
			self.output.close()


class EpisodeTrace:
	def __init__(self, path, *, video):
		self.path = Path(path)
		self.path.mkdir(parents=True, exist_ok=False)
		self.video = video
		self.events = []
		self.branch = 0
		self.branches = [{"id": 0, "start_frame": 0}]
		self.streams = {}
		self.rows = []
		# Additive per-frame columns (agent/student_rollout.py): the motion segment of each command, set by whoever
		# announces segments (-1: none), and the exact float64 state and command for bitwise replay.
		self.segment = -1
		self.exact = []
		self._gripper_cmd: list[float] = []
		self._effort_frac: list[float] = []
		self.top_depth_rng: np.random.Generator | None = None
		self.images = None
		self.started = time.perf_counter()
		self.keyframes = self.path / "keyframes"
		self.keyframes.mkdir()
		self.checkpoints = self.path / "checkpoints"
		self.checkpoints.mkdir()
		self.event_log = (self.path / "events.jsonl").open("w", encoding="utf-8")

	def start(self, cell):
		write_json(self.path / "graph.json", {"nodes": NODES, "edges": EDGES})
		# Scoring only (S6 grader): per-frame contact peaks over the frame's physics substeps, stored next to the
		# actions of every branch, so collisions can be counted over all attempts. The evaluator's own measure
		# excludes the jaws; ``contact_jaw_env_n`` is the jaws against the table, bowls and clutter.
		self.contacts = []
		self._jaw_env_peak = 0.0
		# Chain any observers already installed on the cell (e.g. a catalog recorder) instead of replacing them.
		prior_substep = getattr(cell, "substep_observer", None)
		prior_frame = getattr(cell, "frame_observer", None)

		def substep():
			if prior_substep is not None:
				prior_substep()
			self._substep_contact(cell)

		def frame(episode):
			if prior_frame is not None:
				prior_frame(episode)
			self._frame_contact(episode)

		cell.substep_observer = substep
		cell.frame_observer = frame

	def _substep_contact(self, cell):
		from ..mujoco_cell.episode import jaw_env_contact_force

		self._jaw_env_peak = max(self._jaw_env_peak, jaw_env_contact_force(cell))

	def _frame_contact(self, episode):
		robot = float(np.max(episode.last("substep_unintended")))
		self.contacts.append((robot, self._jaw_env_peak))
		self._jaw_env_peak = 0.0

	def event(self, kind, node, branch, frame, **payload):
		row = {
			"seq": len(self.events),
			"kind": kind,
			"node": node,
			"branch": branch,
			"frame": frame,
			"sim_time_s": frame / FPS,
			"wall_time_s": time.perf_counter() - self.started,
			**payload,
		}
		self.events.append(row)
		self.event_log.write(json.dumps(row, allow_nan=False) + "\n")
		self.event_log.flush()

	def checkpoint(self, checkpoint):
		raw = asdict(checkpoint)
		integration = raw.pop("integration")
		np.save(self.checkpoints / f"{checkpoint.id}.npy", integration, allow_pickle=False)
		write_json(self.checkpoints / f"{checkpoint.id}.json", raw)

	def audit(self, data):
		write_json(self.path / "audit.private.json", data)

	def initial(self, cell, q):
		del q
		if self.video:
			self.images = self._render(cell)

	def command(self, state, target, phase, *, gripper_cmd=None, effort_frac=None):
		self.rows.append((state.copy(), target.copy(), phase))
		self.exact.append(
			(np.asarray(state, dtype=np.float64).copy(), np.asarray(target, dtype=np.float64).copy(), self.segment)
		)
		if gripper_cmd is not None:
			self._gripper_cmd.append(float(gripper_cmd))
		if effort_frac is not None:
			self._effort_frac.append(float(effort_frac))

	def step(self, cell, node):
		if self.video:
			# Video sample i is what the policy would see BEFORE applying command i.
			self._push(node)
			self.images = self._render(cell)
			if len(self.rows) % 15 == 0:
				self._publish_preview(node)

	def _publish_preview(self, node, frame=None):
		preview = self._composite(node, frame)
		ok, buf = cv2.imencode(".jpg", cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
		if not ok:
			return
		tmp = self.path / "live.tmp"
		tmp.write_bytes(buf.tobytes())
		try:
			tmp.replace(self.path / "live.jpg")
		except PermissionError:
			# A browser streaming the previous preview can briefly lock its file on Windows.
			# The next preview can retry. Observation frames and video are recorded separately.
			pass

	def restored(self, cell):
		if self.video:
			self.images = self._render(cell)

	def view(self, evidence, rgb, image):
		if self.video:
			# Use the exact observation the detector cropped at this action boundary.
			# Independent renders of the same state can differ in a few RGB values.
			camera_stream = "top_rgb" if evidence["camera"] == "top" else "wrist"
			self.images[camera_stream] = rgb.copy()
		stem = evidence["id"].replace(":", "-")
		for suffix, pixels in (("frame", rgb), ("crop", image)):
			if pixels is not None:
				encoded = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
				if not cv2.imwrite(str(self.keyframes / f"{stem}-{suffix}.png"), encoded):
					raise OSError("could not store decision camera frame")

	def fork(self, branch, checkpoint):
		self._finish_branch()
		self.branches.append(
			{
				"id": branch,
				"start_frame": checkpoint.frame,
				"parent_branch": checkpoint.branch,
				"checkpoint": checkpoint.id,
			}
		)
		self.branch = branch

	def _render(self, cell):
		top = cell.rgb("top")
		z, valid = cell.height("top", TOP_CROP)
		if self.top_depth_rng is not None:
			top_pose = cell._cam_poses.get("top")
			if top_pose is None:
				raise ValueError("top CameraPose must be set before applying camera-frame D415 depth noise")
			z, valid = degrade_depth_height(
				z,
				valid,
				top_pose,
				self.top_depth_rng,
				DEFAULT_D415_NOISE,
				crop=TOP_CROP,
				full_shape=(TOP_FULL[1], TOP_FULL[0]),
			)
		depth = layered_height_image(z, valid)
		return {
			"front": cell.rgb("front"),
			"wrist": cell.rgb("wrist_cam"),
			"top_rgb": top,
			"top_depth": depth,
		}

	def _composite(self, node, frame=None):
		canvas = np.full((720, 1024, 3), (17, 24, 29), dtype=np.uint8)
		for name, (x, y, w, h, title) in {
			"front": (16, 44, 608, 456, "ROBOT / FRONT"),
			"wrist": (648, 44, 352, 264, "WRIST RGB"),
			"top_rgb": (648, 344, 352, 264, "EXTERNAL RGB"),
		}.items():
			canvas[y : y + h, x : x + w] = cv2.resize(self.images[name], (w, h), interpolation=cv2.INTER_AREA)
			cv2.putText(canvas, title, (x, y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (207, 219, 225), 1, cv2.LINE_AA)
		for i, name in enumerate(NODES):
			x, y = 12 + i * 144, 653
			active = name == node
			cv2.rectangle(canvas, (x, y), (x + 136, y + 44), (44, 164, 112) if active else (39, 53, 64), -1)
			cv2.putText(canvas, name, (x + 8, y + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (239, 248, 252), 1, cv2.LINE_AA)
		branch = self.branches[-1]["id"]
		if frame is None:
			frame = self.branches[-1]["start_frame"] + len(self.rows)
		cv2.putText(
			canvas,
			f"Branch {branch}   Frame {frame:04}",
			(16, 548),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.70,
			(201, 219, 223),
			1,
			cv2.LINE_AA,
		)
		return canvas

	def _push(self, node):
		panels = self._composite(node, self.branches[-1]["start_frame"] + len(self.rows) - 1)
		if not self.streams:
			dir_ = self.path / f"branch-{self.branch}"
			dir_.mkdir()
			self.streams = {
				name: Video(dir_ / f"{name}.mkv", pixels.shape, lossless=True) for name, pixels in self.images.items()
			}
			self.streams["replay"] = Video(dir_ / "replay.mp4", panels.shape, lossless=False)
		for name, pixels in self.images.items():
			self.streams[name].put(pixels)
		self.streams["replay"].put(panels)

	def _finish_branch(self):
		if self.rows:
			path = self.path / f"branch-{self.branch}"
			path.mkdir(exist_ok=True)
			state, action, phase = zip(*self.rows, strict=True)
			extra = {}
			contacts = getattr(self, "contacts", None)
			if contacts is not None and len(contacts) == len(self.rows):
				pairs = np.array(contacts, dtype="float32").reshape(-1, 2)
				extra = {"contact_robot_n": pairs[:, 0], "contact_jaw_env_n": pairs[:, 1]}
			if len(self._gripper_cmd) == len(self.rows):
				extra["gripper_cmd"] = np.array(self._gripper_cmd, dtype="float32")
			if len(self._effort_frac) == len(self.rows):
				extra["effort_frac"] = np.array(self._effort_frac, dtype="float32")
			state64, action64, segment = zip(*self.exact, strict=True)
			np.savez_compressed(
				path / "actions.npz",
				state=np.array(state, dtype="float32"),
				action=np.array(action, dtype="float32"),
				phase=phase,
				**extra,
				state_f64=np.array(state64, dtype=np.float64),
				action_f64=np.array(action64, dtype=np.float64),
				segment=np.array(segment, dtype=np.int32),
			)
		self.branches[-1]["actions"] = len(self.rows)
		for stream in self.streams.values():
			stream.close()
		self.streams = {}
		self.rows = []
		if getattr(self, "contacts", None) is not None:
			self.contacts = []
		self.exact = []
		self._gripper_cmd = []
		self._effort_frac = []

	def finish(self, result, provenance):
		self._finish_branch()
		if self.video and self.images is not None:
			branch = self.branches[-1]
			self._publish_preview("evaluate", branch["start_frame"] + branch["actions"])
		manifest = {
			"schema": "armigo-agent-episode/v1",
			"fps": FPS,
			"task": TASK,
			"state_names": STATE_NAMES,
			"branches": self.branches,
			"video": self.video,
			"provenance": provenance,
			"result": result,
		}
		write_json(self.path / "episode.json", manifest)
		self.event_log.close()

	def close(self):
		self._finish_branch()
		self.event_log.close()


# kind -> (instruction template, held: the brick is carried (lost-brick monitor on), default phase, frame budget)
SEGMENT_KINDS: dict[str, tuple[str, bool, str, int]] = {
	"prepare": ("Wait at rest, then move the arm to the ready pose with the gripper open.", False, "approach", 600),
	"scan_view": (
		"Move the wrist camera to inspection view {view} of {views} around the brick and hold still.",
		False,
		"approach",
		900,
	),
	"return_ready": ("Return the arm to the ready pose with the gripper open.", False, "approach", 600),
	"grasp": ("Grasp the brick on the table.", False, "grasp", 1500),
	"lift": ("Lift the grasped brick straight up.", True, "lift", 600),
	"present_bottom": (
		"Show the bottom face of the held brick to the top camera and hold still.",
		True,
		"show_hold",
		900,
	),
	"return_to_lift": ("Move the held brick back to the lift pose.", True, "show_move", 600),
	"lower_on_table": (
		"Carry the held brick back to its spot on the table and lower it onto the table.",
		True,
		"place",
		1200,
	),
	"release_on_table": (
		"Open the gripper, let the brick settle on the table and lift the gripper clear.",
		False,
		"place",
		600,
	),
	"carry_to_bowl": ("Carry the held brick over the {bowl} bowl and lower it into the bowl.", True, "place", 1200),
	"release_in_bowl": (
		"Open the gripper to drop the brick into the bowl, lift clear and return to the rest pose.",
		False,
		"place",
		900,
	),
	# Recovery (TwinMujoco._retry): let go of a brick that cannot be put down, then return_ready.
	"let_go": ("Open the gripper where it is and rise.", False, "place", 300),
}

# Segment params a policy may read: exactly the fields the instruction texts are formatted from (a test checks
# this). The other params (``mode``, ``faces``, ``target_q``, ``via_poses``, ``bottom_px``, ``xy``) are planner
# outputs: a real-rig student under the D1 default (docs/vla/STUDENT_INPUTS.md) gets the instruction, not the plan.
POLICY_PARAMS: tuple[str, ...] = ("view", "views", "bowl")
# Keys of policy_inputs(): the one definition shared by the online policy and the offline training export.
POLICY_FIELDS: tuple[str, ...] = ("kind", "instruction", *POLICY_PARAMS)


@dataclass(frozen=True)
class Segment:
	kind: str
	params: dict[str, Any] = field(default_factory=dict)

	def __post_init__(self):
		if self.kind not in SEGMENT_KINDS:
			raise ValueError(f"unknown segment kind {self.kind!r}")
		json.dumps(self.params, allow_nan=False)  # params must be plain JSON

	@property
	def instruction(self) -> str:
		return SEGMENT_KINDS[self.kind][0].format(**self.params)

	@property
	def held(self) -> bool:
		return SEGMENT_KINDS[self.kind][1]

	@property
	def phase(self) -> str:
		return SEGMENT_KINDS[self.kind][2]

	@property
	def max_frames(self) -> int:
		return SEGMENT_KINDS[self.kind][3]

	@property
	def diagnostics(self) -> dict[str, Any]:
		"""Planner outputs carried for replay and grading only (never shown to a policy)."""
		return {k: v for k, v in self.params.items() if k not in POLICY_PARAMS}

	def key(self) -> str:
		"""Canonical identity used to match a student's segment with a recorded one (includes the diagnostics)."""
		return json.dumps({"kind": self.kind, "params": self.params}, sort_keys=True, allow_nan=False)


def policy_inputs(segment: Segment) -> dict[str, Any]:
	"""The segment as a policy (online) or a training example (offline export) sees it: ``POLICY_FIELDS`` only, with
	None for a param the segment kind does not have."""
	out = {"kind": segment.kind, "instruction": segment.instruction}
	out.update({k: segment.params.get(k) for k in POLICY_PARAMS})
	return json.loads(json.dumps(out))  # a detached copy


# Per-frame vector inputs added in Data Contract v1 beyond observation.state / action.
INPUT_FEATURES = {
	"observation.prev_action": {"dtype": "float32", "shape": (len(STATE_NAMES),), "names": list(STATE_NAMES)},
	"observation.gripper_cmd": {"dtype": "float32", "shape": (1,), "names": ["gripper_cmd"]},
}

# Per-frame segment and mask columns of the student export (plan S7 / Data Contract v1). Segments are announced by
# the expert backend of every production entry (agent/student_rollout.ExpertTracer) as ``segment`` / ``segment_end``
# events and the per-frame ``segment`` column of actions.npz. Row r is (observation r, the command applied after it).
SEGMENT_FEATURES = {
	"segment.index": {"dtype": "int64", "shape": (1,), "names": None},
	"segment.kind": {"dtype": "string", "shape": (1,), "names": None},
	"segment.done": {"dtype": "bool", "shape": (1,), "names": None},
	"segment.chunk_valid_len": {"dtype": "int64", "shape": (1,), "names": None},
	"segment.frame": {"dtype": "int64", "shape": (1,), "names": None},
	"segment.end_reason": {"dtype": "string", "shape": (1,), "names": None},
	"segment.bowl": {"dtype": "string", "shape": (1,), "names": None},
	"segment.view": {"dtype": "int64", "shape": (1,), "names": None},
	"segment.views": {"dtype": "int64", "shape": (1,), "names": None},
	"mask.is_blind": {"dtype": "bool", "shape": (1,), "names": None},
	"mask.is_takeover": {"dtype": "bool", "shape": (1,), "names": None},
	"mask.supervise_action": {"dtype": "bool", "shape": (1,), "names": None},
}
SEGMENT_DONE_CONVENTION = (
	"segment.done is True on row r when the segment before it ended at observation r with segment_done (end reason"
	" 'done'). Observation r is then both the last observation of the ended segment (the policy answered done under"
	" its instruction) and the first observation of the next one (row r's task and action): no step happens between"
	" them. The ended segment is meta/segments.json's row with end == r (normally row r-1's segment). The last"
	" segment's done observation follows the last action and has no row (the episode ends). Ends by a monitor"
	" (brick_lost) or a fault are boundaries but not segment_done."
)
CHUNK_VALID_LEN_CONVENTION = (
	"segment.chunk_valid_len on row r = rows from r to the end of row r's segment (>= 1): an action chunk starting at"
	" row r may use its steps 0 .. chunk_valid_len-1; later steps belong to the next segment and must be masked."
)


class SegmentLabelError(ValueError):
	"""The recording's segment labels are missing or inconsistent: the episode cannot be exported."""


def segment_labels(episode: Path) -> dict:
	"""Per-row segment labels of an episode's single branch, checked against its events.

	Returns ``segments`` (one dict per segment in id order: id, kind, instruction, params, start, end, reason) and
	per-row arrays ``index``, ``kind``, ``instruction``, ``done``, ``chunk_valid_len``, ``frame``, ``end_reason``,
	``bowl``, ``view`` and ``views`` (see SEGMENT_DONE_CONVENTION and CHUNK_VALID_LEN_CONVENTION). Raises
	:class:`SegmentLabelError` when a row has no segment (``-1``: recorded without segment labels), a segment's rows
	do not match its events, or the segments do not tile the branch."""
	episode = Path(episode)
	with np.load(episode / "branch-0" / "actions.npz", allow_pickle=False) as rows:
		if "segment" not in rows.files:
			raise SegmentLabelError(f"{episode.name}: actions.npz has no segment column")
		ids = rows["segment"].astype(np.int64)
	n = len(ids)
	if n == 0:
		raise SegmentLabelError(f"{episode.name}: no actions")
	if (ids < 0).any():
		raise SegmentLabelError(
			f"{episode.name}: {int((ids < 0).sum())} of {n} frames have no segment (segment == -1); record with the"
			" segment-labelling expert backend (student_rollout.ExpertTracer)"
		)
	events = [json.loads(line) for line in (episode / "events.jsonl").read_text(encoding="utf-8").splitlines()]
	starts = {e["id"]: e for e in events if e["kind"] == "segment"}
	ends = {e["id"]: e for e in events if e["kind"] == "segment_end"}
	if starts.keys() != ends.keys() or sorted(starts) != list(range(len(starts))):
		raise SegmentLabelError(f"{episode.name}: segment start / end events do not pair up")
	segments, cursor = [], 0
	policy_by_id: dict[int, dict[str, Any]] = {}
	for sid in sorted(starts):
		start, end = starts[sid], ends[sid]
		first, last = start["frame"], end["frame"]
		if first != cursor or last - first != end["frames"] or last > n:
			raise SegmentLabelError(f"{episode.name}: segment {sid} [{first}, {last}) does not continue at {cursor}")
		if not (ids[first:last] == sid).all():
			raise SegmentLabelError(f"{episode.name}: frames of segment {sid} carry another segment id")
		# Pull policy-visible conditioning strictly from policy_inputs(Segment(kind, params)), never diagnostics.
		pi = policy_inputs(Segment(start["segment_kind"], start["params"]))
		policy_by_id[sid] = {
			"bowl": "" if pi["bowl"] is None else str(pi["bowl"]),
			"view": 0 if pi["view"] is None else int(pi["view"]),
			"views": 0 if pi["views"] is None else int(pi["views"]),
		}
		segments.append(
			{
				"id": sid,
				"kind": start["segment_kind"],
				"instruction": start["instruction"],
				"params": start["params"],
				"start": first,
				"end": last,
				"reason": end["reason"],
			}
		)
		cursor = last
	if cursor != n:
		raise SegmentLabelError(f"{episode.name}: segments cover {cursor} of {n} frames")
	by_id = {s["id"]: s for s in segments}
	row = np.arange(n)
	done = np.zeros(n, dtype=bool)
	for s in segments:
		if s["reason"] == "done" and s["end"] < n:
			done[s["end"]] = True
	return {
		"segments": segments,
		"index": ids,
		"kind": [by_id[i]["kind"] for i in ids],
		"instruction": [by_id[i]["instruction"] for i in ids],
		"done": done,
		"chunk_valid_len": np.array([by_id[i]["end"] for i in ids], dtype=np.int64) - row,
		"frame": row - np.array([by_id[i]["start"] for i in ids], dtype=np.int64),
		"end_reason": [by_id[i]["reason"] for i in ids],
		"bowl": [policy_by_id[i]["bowl"] for i in ids],
		"view": np.array([policy_by_id[i]["view"] for i in ids], dtype=np.int64),
		"views": np.array([policy_by_id[i]["views"] for i in ids], dtype=np.int64),
	}


def student_frames(episode: Path):
	"""Yield only contiguous, successful unrewound samples for the LeRobot exporter. In-place retries are allowed
	(one continuous branch: a recovery demonstration); a physics rewind is not. Joints are in radians. Each frame
	carries its segment and mask columns (SEGMENT_FEATURES), D1 input vectors (INPUT_FEATURES) and the segment
	instruction as ``task``; an episode with frames outside any segment is rejected (:class:`SegmentLabelError`)."""
	from contextlib import ExitStack

	episode = Path(episode)
	manifest = json.loads((episode / "episode.json").read_text(encoding="utf-8"))
	result = manifest["result"]
	if (
		manifest["schema"] != "armigo-agent-episode/v1"
		or not manifest["video"]
		or result["verdict"] != "SUCCESS"
		or not result["student_eligible"]
		or len(manifest["branches"]) != 1
		or manifest["branches"][0]["start_frame"] != 0
		or result["rewinds"]
		or manifest["task"] != TASK
		or manifest["fps"] != FPS
	):
		raise ValueError("episode is not eligible for VLA export")
	root = episode / "branch-0"
	labels = segment_labels(episode)
	with ExitStack() as stack:
		rows = stack.enter_context(np.load(root / "actions.npz", allow_pickle=False))
		gripper_cmds = rows["gripper_cmd"] if "gripper_cmd" in rows.files else None
		actions = rows["action"]
		states = rows["state"]
		readers = {
			name: iter(stack.enter_context(av.open(str(root / f"{name}.mkv"))).decode(video=0)) for name in CAMERAS
		}
		for row_index, (state, action) in enumerate(zip(states, actions, strict=True)):
			prev_action = INITIAL_Q if row_index == 0 else actions[row_index - 1]
			gripper_cmd = float(gripper_cmds[row_index]) if gripper_cmds is not None else float(prev_action[5])
			frame = {
				"observation.state": np.asarray(state, dtype=np.float32),
				"observation.prev_action": np.asarray(prev_action, dtype=np.float32),
				"observation.gripper_cmd": np.array([gripper_cmd], dtype=np.float32),
				"action": np.asarray(action, dtype=np.float32),
				"task": labels["instruction"][row_index],
				"segment.index": np.array([labels["index"][row_index]], dtype=np.int64),
				"segment.kind": labels["kind"][row_index],
				"segment.done": np.array([labels["done"][row_index]], dtype=bool),
				"segment.chunk_valid_len": np.array([labels["chunk_valid_len"][row_index]], dtype=np.int64),
				"segment.frame": np.array([labels["frame"][row_index]], dtype=np.int64),
				"segment.end_reason": labels["end_reason"][row_index],
				"segment.bowl": labels["bowl"][row_index],
				"segment.view": np.array([labels["view"][row_index]], dtype=np.int64),
				"segment.views": np.array([labels["views"][row_index]], dtype=np.int64),
				"mask.is_blind": np.array([False], dtype=bool),
				"mask.is_takeover": np.array([False], dtype=bool),
				"mask.supervise_action": np.array([True], dtype=bool),
			}
			for name, reader in readers.items():
				pixels = next(reader, None)
				if pixels is None or pixels.time is None or abs(pixels.time - row_index / FPS) > 0.001:
					raise ValueError(f"missing or misaligned frame {row_index} in {name} stream")
				image = pixels.to_ndarray(format="rgb24")
				key = name
				if name == "top_rgb":
					frame["observation.images.top_full"] = image
					x, y, w, h = TOP_CROP
					image = image[y : y + h, x : x + w].copy()
					key = "top_rgb_v2"
				elif name == "top_depth":
					key = "top_depth_v2"
				frame[f"observation.images.{key}"] = image
			yield frame
		if any(next(reader, None) is not None for reader in readers.values()):
			raise ValueError("camera stream has frames with no matching action")


def export_lerobot(episodes: list[Path], out: Path, repo_id: str):
	from lerobot.datasets.lerobot_dataset import LeRobotDataset

	from ..constants import CAMERA_SHAPES
	from ..lerobot_export import lerobot_features

	# Fail before creating output if an episode ended on a branch with privileged failure context or lacks segment
	# labels (student_frames checks both on its first frame).
	for path in episodes:
		frames = student_frames(path)
		try:
			next(frames)
		finally:
			frames.close()
	dataset = LeRobotDataset.create(
		repo_id=repo_id,
		fps=FPS,
		root=out,
		robot_type="SO101_Follower",
		use_videos=True,
		features=lerobot_features(
			{
				"state_names": STATE_NAMES,
				"cameras": {**CAMERA_SHAPES, "top_full": [TOP_FULL[1], TOP_FULL[0], 3]},
			},
			video=True,
		)
		| {key: dict(spec) for key, spec in INPUT_FEATURES.items()}
		| {key: dict(spec) for key, spec in SEGMENT_FEATURES.items()},
	)
	try:
		for path in episodes:
			for frame in student_frames(path):
				dataset.add_frame(frame)
			dataset.save_episode()
	finally:
		dataset.finalize()
	write_json(
		out / "meta/source.json",
		{
			"schema": "armigo-student-provenance/v1",
			"episodes": [{"episode_index": i, "source": path.name} for i, path in enumerate(episodes)],
			"sampling": "pre-action RGB and proprioception; next policy command; joint angles in radians",
			"joint_units": joint_normalization(STATE_NAMES),
			"decision_camera": "top_full or wrist at recorded frame index",
			"task": "the instruction of the frame's segment (meta/segments.json); the episode goal is goal_task",
			"goal_task": TASK,
		},
	)
	write_json(
		out / "meta/segments.json",
		{
			"schema": "armigo-student-segments/v1",
			"features": sorted(SEGMENT_FEATURES),
			"segment_done": SEGMENT_DONE_CONVENTION,
			"chunk_valid_len": CHUNK_VALID_LEN_CONVENTION,
			"episodes": [
				{"episode_index": i, "source": str(path), "segments": segment_labels(path)["segments"]}
				for i, path in enumerate(episodes)
			],
		},
	)


def main():
	import argparse

	parser = argparse.ArgumentParser(description="Export accepted Armigo recordings through LeRobot.")
	parser.add_argument("--out", type=Path, required=True)
	parser.add_argument("--repo-id", default="local/armigo-six-face")
	parser.add_argument("episodes", nargs="+", type=Path)
	args = parser.parse_args()
	export_lerobot(args.episodes, args.out, args.repo_id)


if __name__ == "__main__":
	main()
