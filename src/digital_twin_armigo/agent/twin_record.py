"""Frame-aligned camera, robot and LangGraph recordings for the six-face twin."""

from __future__ import annotations

import json
import shutil
import sys
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
from ..vla.contract import format_instruction
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
		# A fresh directory; the experiment runner may have put only its run_status.json there already.
		if not (self.path.is_dir() and {p.name for p in self.path.iterdir()} <= {"run_status.json"}):
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
		self._is_blind: list[bool] = []
		self._is_takeover: list[bool] = []
		self._supervise_action: list[bool] = []
		self.noisy: list[bool] = []
		self.top_depth_rng: np.random.Generator | None = None
		self.images = None
		self.started = time.perf_counter()
		self.keyframes = self.path / "keyframes"
		self.keyframes.mkdir()
		self.checkpoints = self.path / "checkpoints"
		self.checkpoints.mkdir()
		self.event_log = (self.path / "events.jsonl").open("w", encoding="utf-8")
		self.live = None  # viewer.live.LiveStream when ARMIGO_RERUN is set (see docs/VIEWER.md)

	def _live(self, call, *args):
		"""Run a live-viewer hook; a failure turns the stream off instead of interrupting the episode."""
		if self.live is None:
			return
		try:
			call(*args)
		except Exception as exc:  # noqa: BLE001 - optional streaming must never stop an episode
			print(f"armigo: live Rerun stream stopped ({exc})", file=sys.stderr)
			self.live = None

	def _end_live(self):
		live, self.live = self.live, None
		if live is not None:
			try:
				live.finish(self.events)
			except Exception as exc:  # noqa: BLE001 - optional streaming must never stop an episode
				print(f"armigo: live Rerun stream not finalised ({exc})", file=sys.stderr)

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
			if self.live is not None:
				self._live(self.live.on_frame, episode)

		cell.substep_observer = substep
		cell.frame_observer = frame
		from digital_twin_armigo.viewer import live

		self.live = live.maybe_start(cell, self.path, started=self.started)

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

	def command(
		self,
		state,
		target,
		phase,
		*,
		gripper_cmd=None,
		effort_frac=None,
		is_blind=False,
		is_takeover=False,
		supervise_action=True,
		noisy=False,
	):
		self.rows.append((state.copy(), target.copy(), phase))
		self.exact.append(
			(np.asarray(state, dtype=np.float64).copy(), np.asarray(target, dtype=np.float64).copy(), self.segment)
		)
		if gripper_cmd is not None:
			self._gripper_cmd.append(float(gripper_cmd))
		if effort_frac is not None:
			self._effort_frac.append(float(effort_frac))
		self._is_blind.append(bool(is_blind))
		self._is_takeover.append(bool(is_takeover))
		self._supervise_action.append(bool(supervise_action))
		self.noisy.append(bool(noisy))

	def set_row_masks(self, *, is_blind=None, is_takeover=None, supervise_action=None, noisy=None):
		"""Overwrite the per-row mask columns of the rows recorded so far (the active branch). Each argument is a bool
		sequence of length ``len(self.rows)`` (or None to keep it); rows appended later use ``command``'s flags.
		Hook for callers that only know a mask after the fact (S11 takeover onset, S5 detection frame)."""
		for attr, label, values in (
			("_is_blind", "is_blind", is_blind),
			("_is_takeover", "is_takeover", is_takeover),
			("_supervise_action", "supervise_action", supervise_action),
			("noisy", "noisy", noisy),
		):
			if values is None:
				continue
			values = [bool(v) for v in values]
			if len(values) != len(self.rows):
				raise ValueError(f"{label} has {len(values)} entries for {len(self.rows)} rows")
			setattr(self, attr, values)

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
			if len(self._is_blind) == len(self.rows):
				extra["is_blind"] = np.array(self._is_blind, dtype=bool)
			if len(self._is_takeover) == len(self.rows):
				extra["is_takeover"] = np.array(self._is_takeover, dtype=bool)
			if len(self._supervise_action) == len(self.rows):
				extra["supervise_action"] = np.array(self._supervise_action, dtype=bool)
			if len(self.noisy) == len(self.rows) and any(self.noisy):
				extra["noisy"] = np.array(self.noisy, dtype=bool)
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
			self._jaw_env_peak = 0.0
		self.exact = []
		self._gripper_cmd = []
		self._effort_frac = []
		self._is_blind = []
		self._is_takeover = []
		self._supervise_action = []
		self.noisy = []

	def finish(self, result, provenance):
		from digital_twin_armigo.inventory.taxonomy import classify_provenance

		# Freeze who drove this episode (see inventory.taxonomy) so later rule changes never relabel written data.
		actor = classify_provenance(provenance, result)
		provenance = {**provenance, "actor": {k: v for k, v in actor.items() if k != "basis"}}
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
		self._end_live()

	def close(self):
		self._finish_branch()
		self.event_log.close()
		self._end_live()


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
#
# ``roll_deg`` is the wrist roll the controller chose for the segment: for ``grasp`` the candidate whose pinched faces
# suit the detector's cleared / suspect faces (``choose_grasp``), for ``scan_view`` / ``present_bottom`` the planned
# view pose's roll. These are segment-level decisions (R3): the student cannot infer them from its observation, so
# without them the target is multimodal (S10 ACT baseline 02: 0/60, every episode failing its first grasp).
POLICY_PARAMS: tuple[str, ...] = ("view", "views", "bowl", "roll_deg")
ROLL_SUFFIX = " Use a wrist roll of {roll_deg} degrees."
# Keys of policy_inputs(): the one definition shared by the online policy and the offline training export.
POLICY_FIELDS: tuple[str, ...] = ("kind", "instruction", *POLICY_PARAMS)


@dataclass(frozen=True)
class Segment:
	kind: str
	params: dict[str, Any] = field(default_factory=dict)
	variant: int = 0

	def __post_init__(self):
		if self.kind not in SEGMENT_KINDS:
			raise ValueError(f"unknown segment kind {self.kind!r}")
		json.dumps(self.params, allow_nan=False)  # params must be plain JSON

	@property
	def instruction(self) -> str:
		return format_instruction(self.kind, self.params, variant=self.variant)

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
		d = {"kind": self.kind, "params": self.params}
		if self.variant != 0:
			d["variant"] = self.variant
		return json.dumps(d, sort_keys=True, allow_nan=False)


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
	"mask.noisy": {"dtype": "bool", "shape": (1,), "names": None},
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
		variant = int(start.get("variant", 0))
		pi = policy_inputs(Segment(start["segment_kind"], start["params"], variant=variant))
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
				**({"variant": variant} if variant else {}),
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


def student_frames(
	episode: Path,
	start_frame: int = 0,
	student_cameras_only: bool = False,
	*,
	include_noisy_mask: bool | None = None,
	video_dir: Path | None = None,
):
	"""Yield only contiguous, successful unrewound samples for the LeRobot exporter. In-place retries are allowed
	(one continuous branch: a recovery demonstration); a physics rewind is not. Joints are in radians. Each frame
	carries its segment and mask columns (SEGMENT_FEATURES), D1 input vectors (INPUT_FEATURES) and the segment
	instruction as ``task``; an episode with frames outside any segment is rejected (:class:`SegmentLabelError`).
	Rows before ``start_frame`` are checked and decoded but not yielded (see :func:`unsupervised_prefix_end`).

	``include_noisy_mask`` defaults to ``True`` when ``student_cameras_only=True`` (canonical v3 3-camera export)
	and ``False`` when ``student_cameras_only=False`` (legacy 5-camera export without ``mask.noisy`` for merge
	compatibility). When ``video_dir`` is provided, ``.mkv`` streams are verified against ``rerender_manifest.json``
	and read from ``video_dir`` even if the bronze episode was recorded with ``video=False``."""
	from contextlib import ExitStack

	episode = Path(episode)
	manifest = json.loads((episode / "episode.json").read_text(encoding="utf-8"))
	result = manifest["result"]
	if (
		manifest["schema"] != "armigo-agent-episode/v1"
		or result["verdict"] != "SUCCESS"
		or not result["student_eligible"]
		or len(manifest["branches"]) != 1
		or manifest["branches"][0]["start_frame"] != 0
		or result["rewinds"]
		or manifest["task"] != TASK
		or manifest["fps"] != FPS
	):
		raise ValueError("episode is not eligible for VLA export")
	if video_dir is None and not manifest["video"]:
		raise ValueError(
			f"{episode.name}: episode has no recorded video (video=False); "
			"deferred rerender is required (pass verified video_dir or --rerender-video)"
		)
	use_noisy_mask = bool(student_cameras_only) if include_noisy_mask is None else bool(include_noisy_mask)
	root = episode / "branch-0"
	labels = segment_labels(episode)
	active_cameras = [c for c in CAMERAS if not (student_cameras_only and c in ("front", "top_full"))]
	if video_dir is not None:
		from .deferred_video import verify_rerender_dir

		mkv_root = Path(video_dir)
		verify_rerender_dir(episode, mkv_root, required_cameras=active_cameras)
	else:
		mkv_root = root
	with ExitStack() as stack:
		rows = stack.enter_context(np.load(root / "actions.npz", allow_pickle=False))
		gripper_cmds = rows["gripper_cmd"] if "gripper_cmd" in rows.files else None
		is_blind_col = rows["is_blind"] if "is_blind" in rows.files else None
		is_takeover_col = rows["is_takeover"] if "is_takeover" in rows.files else None
		supervise_col = rows["supervise_action"] if "supervise_action" in rows.files else None
		noisy_col = rows["noisy"] if "noisy" in rows.files else None
		actions = rows["action"]
		states = rows["state"]
		for col_name, arr in (
			("is_blind", is_blind_col),
			("is_takeover", is_takeover_col),
			("supervise_action", supervise_col),
			("noisy", noisy_col),
		):
			if arr is not None and len(arr) != len(actions):
				raise ValueError(f"{col_name} length {len(arr)} != actions length {len(actions)}")
		readers = {
			name: iter(stack.enter_context(av.open(str(mkv_root / f"{name}.mkv"))).decode(video=0))
			for name in active_cameras
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
				"mask.is_blind": np.array(
					[bool(is_blind_col[row_index]) if is_blind_col is not None else False], dtype=bool
				),
				"mask.is_takeover": np.array(
					[bool(is_takeover_col[row_index]) if is_takeover_col is not None else False], dtype=bool
				),
				"mask.supervise_action": np.array(
					[bool(supervise_col[row_index]) if supervise_col is not None else True], dtype=bool
				),
			}
			if use_noisy_mask:
				frame["mask.noisy"] = np.array(
					[bool(noisy_col[row_index]) if noisy_col is not None else False], dtype=bool
				)
			for name, reader in readers.items():
				pixels = next(reader, None)
				if pixels is None or pixels.time is None or abs(pixels.time - row_index / FPS) > 0.001:
					raise ValueError(f"missing or misaligned frame {row_index} in {name} stream")
				image = pixels.to_ndarray(format="rgb24")
				key = name
				if name == "top_rgb":
					if not student_cameras_only:
						frame["observation.images.top_full"] = image
					x, y, w, h = TOP_CROP
					image = image[y : y + h, x : x + w].copy()
					key = "top_rgb_v2"
				elif name == "top_depth":
					key = "top_depth_v2"
				frame[f"observation.images.{key}"] = image
			if row_index >= start_frame:
				yield frame
		if any(next(reader, None) is not None for reader in readers.values()):
			raise ValueError("camera stream has frames with no matching action")


def unsupervised_prefix_end(episode: Path) -> int:
	"""First row worth exporting when only the supervised part of an episode is wanted: one row before the first
	``supervise_action`` row (that row's previous state is still in the data), 0 without the column or such rows."""
	with np.load(Path(episode) / "branch-0" / "actions.npz", allow_pickle=False) as rows:
		if "supervise_action" not in rows.files:
			return 0
		sup = np.flatnonzero(rows["supervise_action"])
	return max(int(sup[0]) - 1, 0) if sup.size else 0


def _trimmed_segments(segments: list[dict], start: int) -> list[dict]:
	"""Segments re-based to an export that starts at row ``start``; segments wholly before it are dropped."""
	out = []
	for seg in segments:
		if seg["end"] > start or (seg["start"] == seg["end"] and seg["start"] >= start):
			out.append({**seg, "start": max(seg["start"] - start, 0), "end": seg["end"] - start})
	return out


def _resolve_video_dir(video_dirs: Any, idx: int, path: Path) -> Path | None:
	if video_dirs is None:
		return None
	if isinstance(video_dirs, dict):
		for key in (path, str(path), path.resolve(), str(path.resolve())):
			if key in video_dirs and video_dirs[key] is not None:
				return Path(video_dirs[key])
		return None
	val = video_dirs[idx]
	return None if val is None else Path(val)


def export_lerobot(
	episodes: list[Path],
	out: Path,
	repo_id: str,
	trim_unsupervised_prefix: bool = False,
	student_cameras_only: bool = False,
	*,
	include_noisy_mask: bool | None = None,
	video_dirs: Any = None,
):
	"""Export ``episodes`` as a LeRobot dataset. With ``trim_unsupervised_prefix`` each episode starts at
	:func:`unsupervised_prefix_end` (paired carry episodes: the shared prefix carries no supervision)."""
	from lerobot.datasets.lerobot_dataset import LeRobotDataset

	from ..constants import CAMERA_SHAPES
	from ..lerobot_export import lerobot_features

	use_noisy_mask = bool(student_cameras_only) if include_noisy_mask is None else bool(include_noisy_mask)
	resolved_vdirs = [_resolve_video_dir(video_dirs, i, Path(p)) for i, p in enumerate(episodes)]

	# Fail before creating output if an episode ended on a branch with privileged failure context or lacks segment
	# labels (student_frames checks both on its first frame).
	for path, vdir in zip(episodes, resolved_vdirs, strict=True):
		frames = student_frames(
			path,
			student_cameras_only=student_cameras_only,
			include_noisy_mask=use_noisy_mask,
			video_dir=vdir,
		)
		try:
			next(frames)
		finally:
			frames.close()
	cam_shapes = dict(CAMERA_SHAPES)
	if student_cameras_only:
		cam_shapes.pop("front", None)
	else:
		cam_shapes["top_full"] = [TOP_FULL[1], TOP_FULL[0], 3]

	seg_features = {key: dict(spec) for key, spec in SEGMENT_FEATURES.items() if use_noisy_mask or key != "mask.noisy"}
	dataset = LeRobotDataset.create(
		repo_id=repo_id,
		fps=FPS,
		root=out,
		robot_type="SO101_Follower",
		use_videos=True,
		features=lerobot_features(
			{
				"state_names": STATE_NAMES,
				"cameras": cam_shapes,
			},
			video=True,
		)
		| {key: dict(spec) for key, spec in INPUT_FEATURES.items()}
		| seg_features,
	)
	starts = [unsupervised_prefix_end(p) if trim_unsupervised_prefix else 0 for p in episodes]
	try:
		for path, start, vdir in zip(episodes, starts, resolved_vdirs, strict=True):
			for frame in student_frames(
				path,
				start,
				student_cameras_only=student_cameras_only,
				include_noisy_mask=use_noisy_mask,
				video_dir=vdir,
			):
				dataset.add_frame(frame)
			dataset.save_episode()
	finally:
		dataset.finalize()
	write_json(
		out / "meta/source.json",
		{
			"schema": "armigo-student-provenance/v1",
			"episodes": [
				{"episode_index": i, "source": path.name, **({"start_frame": st} if st else {})}
				for i, (path, st) in enumerate(zip(episodes, starts, strict=True))
			],
			"sampling": "pre-action RGB and proprioception; next policy command; joint angles in radians",
			"joint_units": joint_normalization(STATE_NAMES),
			"decision_camera": "top_full or wrist at recorded frame index",
			"task": "the instruction of the frame's segment (meta/segments.json); the episode goal is goal_task",
			"goal_task": TASK,
		},
	)
	episodes_meta = []
	for i, (path, st) in enumerate(zip(episodes, starts, strict=True)):
		ep_json_path = path / "episode.json"
		ep_variant = 0
		ep_scene = None
		if ep_json_path.exists():
			ep_manifest = json.loads(ep_json_path.read_text(encoding="utf-8"))
			prov = ep_manifest.get("provenance") or {}
			res = ep_manifest.get("result") or {}
			case = prov.get("case") or []
			case_opts = case[3] if len(case) == 4 and isinstance(case[3], dict) else {}
			if prov.get("scene") == "v3" or res.get("scene") == "v3" or case_opts.get("scene") == "v3":
				ep_scene = "v3"
			if "variant" in prov:
				ep_variant = int(prov["variant"])
			elif "variant" in res:
				ep_variant = int(res["variant"])
		# Also check segments for any variant
		segs_list = _trimmed_segments(segment_labels(path)["segments"], st)
		for s_item in segs_list:
			if s_item.get("variant", 0) != 0:
				ep_variant = int(s_item["variant"])
		ep_dict: dict[str, Any] = {
			"episode_index": i,
			"source": str(path),
			"segments": segs_list,
		}
		if ep_scene == "v3":
			ep_dict["scene"] = "v3"
		if ep_variant != 0:
			ep_dict["variant"] = ep_variant
		if st:
			ep_dict["start_frame"] = st
		episodes_meta.append(ep_dict)

	write_json(
		out / "meta/segments.json",
		{
			"schema": "armigo-student-segments/v1",
			"features": sorted(seg_features),
			"segment_done": SEGMENT_DONE_CONVENTION,
			"chunk_valid_len": CHUNK_VALID_LEN_CONVENTION,
			"episodes": episodes_meta,
		},
	)


def _rewrite_view_id(view_id: str | None, src_branch: int, dst_branch: int = 0) -> str | None:
	if not view_id or src_branch == dst_branch:
		return view_id
	prefix = f"{src_branch}:"
	if view_id.startswith(prefix):
		return f"{dst_branch}:{view_id[len(prefix) :]}"
	return view_id


def _extract_prefix_events(events: list[dict], start_frame: int, ck_id: str | None, parent_branch: int) -> list[dict]:
	if ck_id is not None:
		for i, ev in enumerate(events):
			if (
				ev.get("kind") == "checkpoint"
				and ev.get("checkpoint") == ck_id
				and ev.get("branch", 0) == parent_branch
				and not ev.get("restored")
			):
				return [ev_i for ev_i in events[: i + 1] if ev_i.get("branch", 0) == parent_branch]
	out = []
	for ev in events:
		if ev.get("branch", 0) == parent_branch and ev.get("kind") in ("rewound", "perturbation_branch"):
			break
		if ev.get("branch", 0) != parent_branch:
			continue
		f = ev.get("frame", 0)
		if f < start_frame or (f == start_frame and ev.get("kind") in ("segment_end", "checkpoint")):
			out.append(ev)
	return out


def _extract_suffix_events(
	events: list[dict], branch_id: int, src_branch_id: int, *, cross_process: bool
) -> list[dict]:
	if cross_process:
		resumed_idx = next((i for i, ev in enumerate(events) if ev.get("kind") == "resumed"), -1)
		return [
			ev
			for ev in events[resumed_idx + 1 :]
			if ev.get("branch", src_branch_id) == src_branch_id
			and ev.get("kind") not in ("rewound", "perturbation_branch", "resumed")
		]
	fork_idx = -1
	for i, ev in enumerate(events):
		if ev.get("branch") == branch_id and ev.get("kind") in ("perturbation_branch", "rewound"):
			fork_idx = i
	return [
		ev
		for ev in events[fork_idx + 1 :]
		if ev.get("branch") == branch_id and ev.get("kind") not in ("perturbation_branch", "rewound", "resumed")
	]


def _concat_lossless_videos(
	prefix_dir: Path,
	prefix_frames: int,
	suffix_dir: Path,
	suffix_frames: int,
	out_branch_dir: Path,
) -> None:
	"""Concatenate prefix[0:prefix_frames] + suffix[0:suffix_frames] for every lossless camera stream."""
	for name in CAMERAS:
		prefix_mkv = prefix_dir / f"{name}.mkv"
		suffix_mkv = suffix_dir / f"{name}.mkv"
		if not prefix_mkv.is_file() or not suffix_mkv.is_file():
			raise FileNotFoundError(f"missing {name}.mkv in {prefix_dir} or {suffix_dir}")
		frames_written = 0
		writer = None
		try:
			with av.open(str(prefix_mkv)) as pin:
				for idx, frame in enumerate(pin.decode(video=0)):
					if idx >= prefix_frames:
						break
					rgb = frame.to_ndarray(format="rgb24")
					if writer is None:
						writer = Video(out_branch_dir / f"{name}.mkv", rgb.shape, lossless=True)
					writer.put(rgb)
					frames_written += 1
			if frames_written != prefix_frames:
				raise ValueError(f"{prefix_mkv}: expected {prefix_frames} prefix frames, got {frames_written}")
			suffix_written = 0
			with av.open(str(suffix_mkv)) as sin:
				for frame in sin.decode(video=0):
					rgb = frame.to_ndarray(format="rgb24")
					if writer is None:
						writer = Video(out_branch_dir / f"{name}.mkv", rgb.shape, lossless=True)
					writer.put(rgb)
					suffix_written += 1
			if suffix_written != suffix_frames:
				raise ValueError(f"{suffix_mkv}: expected {suffix_frames} suffix frames, got {suffix_written}")
		finally:
			if writer is not None:
				writer.close()


def stitch_perturbation_episode(
	parent_episode: Path,
	branch_id: int,
	out_dir: Path,
	*,
	blind_span: tuple[int, int] | None = None,
	branch_episode: Path | None = None,
) -> dict[str, Any]:
	"""Combine ``parent_episode/branch-0[0:start_frame]`` with ``branch-{branch_id}`` into one contiguous
	``branch-0`` episode directory at ``out_dir``, renumbering segments ``0..K-1`` and marking ``is_blind=True``
	on ``[perturb_frame, detect_frame)``.

	Validates with :func:`segment_labels` before returning the stitched ``episode.json`` manifest."""
	parent_episode = Path(parent_episode)
	branch_src = parent_episode if branch_episode is None else Path(branch_episode)
	cross_process = branch_episode is not None
	out_dir = Path(out_dir)
	if out_dir.resolve() == parent_episode.resolve() or out_dir.resolve() == branch_src.resolve():
		raise ValueError("out_dir must be distinct from source episode directories")
	if (out_dir / "episode.json").exists():
		raise FileExistsError(f"stitched output directory already has episode.json: {out_dir}")

	parent_manifest = json.loads((parent_episode / "episode.json").read_text(encoding="utf-8"))
	branch_manifest = (
		parent_manifest
		if branch_episode is None
		else json.loads((branch_src / "episode.json").read_text(encoding="utf-8"))
	)
	branch_meta = next((b for b in branch_manifest.get("branches", []) if int(b["id"]) == branch_id), None)
	if branch_meta is None and cross_process and len(branch_manifest.get("branches", [])) == 1:
		branch_meta = branch_manifest["branches"][0]
	if branch_meta is None:
		raise ValueError(f"branch {branch_id} not found in {branch_src / 'episode.json'}")
	src_branch_id = int(branch_meta["id"])
	start_frame = int(branch_meta["start_frame"])
	parent_branch = int(branch_meta.get("parent_branch", 0))
	if parent_branch != 0:
		raise ValueError(f"stitching requires parent_branch == 0, got {parent_branch}")
	ck_id = branch_meta.get("checkpoint")

	prefix_dir = parent_episode / "branch-0"
	suffix_dir = branch_src / f"branch-{src_branch_id}"
	with (
		np.load(prefix_dir / "actions.npz", allow_pickle=False) as pz,
		np.load(suffix_dir / "actions.npz", allow_pickle=False) as sz,
	):
		p_arrs = {k: pz[k] for k in pz.files}
		s_arrs = {k: sz[k] for k in sz.files}

	n_prefix_avail = len(p_arrs["action"])
	if start_frame < 0 or start_frame > n_prefix_avail:
		raise ValueError(f"invalid start_frame={start_frame} for prefix with {n_prefix_avail} actions")
	n_suffix = len(s_arrs["action"])
	n_total = start_frame + n_suffix
	if n_total == 0:
		raise ValueError("stitched episode has 0 actions")

	parent_events = [
		json.loads(line)
		for line in (parent_episode / "events.jsonl").read_text(encoding="utf-8").splitlines()
		if line.strip()
	]
	branch_events = (
		parent_events
		if not cross_process
		else [
			json.loads(line)
			for line in (branch_src / "events.jsonl").read_text(encoding="utf-8").splitlines()
			if line.strip()
		]
	)
	prefix_raw = _extract_prefix_events(parent_events, start_frame, ck_id, parent_branch)
	suffix_raw = _extract_suffix_events(branch_events, branch_id, src_branch_id, cross_process=cross_process)

	prefix_starts = [ev for ev in prefix_raw if ev.get("kind") == "segment"]
	prefix_ends = {int(ev["id"]): ev for ev in prefix_raw if ev.get("kind") == "segment_end"}
	suffix_starts = [ev for ev in suffix_raw if ev.get("kind") == "segment"]
	suffix_ends = {int(ev["id"]): ev for ev in suffix_raw if ev.get("kind") == "segment_end"}

	prefix_id_map: dict[int, int] = {}
	suffix_id_map: dict[int, int] = {}
	next_sid = 0
	for ev in prefix_starts:
		old_id = int(ev["id"])
		if old_id not in prefix_ends:
			raise SegmentLabelError(f"prefix segment {old_id} missing segment_end")
		prefix_id_map[old_id] = next_sid
		next_sid += 1
	for ev in suffix_starts:
		old_id = int(ev["id"])
		if old_id not in suffix_ends:
			raise SegmentLabelError(f"suffix segment {old_id} missing segment_end")
		suffix_id_map[old_id] = next_sid
		next_sid += 1

	p_seg_raw = p_arrs["segment"][:start_frame].astype(np.int64)
	s_seg_raw = s_arrs["segment"].astype(np.int64)
	if (p_seg_raw < 0).any() or (s_seg_raw < 0).any():
		raise SegmentLabelError("cannot stitch episode with unlabelled (segment == -1) frames")
	p_seg = np.array([prefix_id_map[int(sid)] for sid in p_seg_raw], dtype=np.int32)
	s_seg = np.array([suffix_id_map[int(sid)] for sid in s_seg_raw], dtype=np.int32)

	stitched_events: list[dict] = []
	view_copies: list[tuple[bool, str, str]] = []
	for is_suffix, ev in [*((False, e) for e in prefix_raw), *((True, e) for e in suffix_raw)]:
		new_ev = json.loads(json.dumps(ev))
		new_ev["seq"] = len(stitched_events)
		new_ev["branch"] = 0
		src_b = src_branch_id if is_suffix else 0
		kind = new_ev.get("kind")
		if kind in ("segment", "segment_end"):
			id_map = suffix_id_map if is_suffix else prefix_id_map
			new_ev["id"] = id_map[int(new_ev["id"])]
		elif kind in ("view", "detector_error"):
			obs = new_ev["observation"]
			orig_vid = obs["id"]
			new_vid = _rewrite_view_id(orig_vid, src_b, 0)
			obs["id"] = new_vid
			obs["branch"] = 0
			if kind == "view":
				view_copies.append((is_suffix, orig_vid, new_vid))
		elif kind == "view_truth":
			new_ev["id"] = _rewrite_view_id(new_ev["id"], src_b, 0)
		elif kind == "decision" and "evidence" in new_ev:
			new_ev["evidence"] = [_rewrite_view_id(v, src_b, 0) for v in new_ev["evidence"]]
		elif kind == "failure" and "prior_views" in new_ev:
			new_ev["prior_views"] = [
				{**pv, "id": _rewrite_view_id(pv.get("id"), src_b, 0)} for pv in new_ev["prior_views"]
			]
		stitched_events.append(new_ev)

	if blind_span is None and any(ev.get("kind") == "perturbation" for ev in stitched_events):
		from .perturbations import infer_blind_span

		blind_span = infer_blind_span(stitched_events)

	mask_defaults = {"is_blind": False, "is_takeover": False, "supervise_action": True}
	stitched_arrays: dict[str, np.ndarray] = {"segment": np.concatenate([p_seg, s_seg], axis=0)}
	skip_keys = {"segment", "noisy", *mask_defaults}
	for key in p_arrs:
		if key in skip_keys or key not in s_arrs:
			continue
		stitched_arrays[key] = np.concatenate([p_arrs[key][:start_frame], s_arrs[key]], axis=0)

	for mask_key, default_val in mask_defaults.items():
		p_mask = (
			np.asarray(p_arrs[mask_key][:start_frame], dtype=bool)
			if mask_key in p_arrs
			else np.full(start_frame, default_val, dtype=bool)
		)
		s_mask = (
			np.asarray(s_arrs[mask_key], dtype=bool)
			if mask_key in s_arrs
			else np.full(n_suffix, default_val, dtype=bool)
		)
		stitched_arrays[mask_key] = np.concatenate([p_mask, s_mask], axis=0)

	if "noisy" in p_arrs or "noisy" in s_arrs:
		p_noisy = (
			np.asarray(p_arrs["noisy"][:start_frame], dtype=bool)
			if "noisy" in p_arrs
			else np.zeros(start_frame, dtype=bool)
		)
		s_noisy = np.asarray(s_arrs["noisy"], dtype=bool) if "noisy" in s_arrs else np.zeros(n_suffix, dtype=bool)
		noisy_cat = np.concatenate([p_noisy, s_noisy], axis=0)
		if bool(noisy_cat.any()):
			stitched_arrays["noisy"] = noisy_cat

	if blind_span is not None:
		lo = max(0, min(n_total, int(blind_span[0])))
		hi = max(lo, min(n_total, int(blind_span[1])))
		stitched_arrays["is_blind"][lo:hi] = True

	out_dir.mkdir(parents=True, exist_ok=True)
	out_branch = out_dir / "branch-0"
	out_branch.mkdir(exist_ok=True)
	(out_dir / "keyframes").mkdir(exist_ok=True)
	(out_dir / "checkpoints").mkdir(exist_ok=True)

	np.savez_compressed(out_branch / "actions.npz", **stitched_arrays)

	has_video = bool(parent_manifest.get("video")) and bool(branch_manifest.get("video", True))
	if has_video:
		_concat_lossless_videos(prefix_dir, start_frame, suffix_dir, n_suffix, out_branch)

	for is_suffix, orig_vid, new_vid in view_copies:
		src_kf_dir = (branch_src if is_suffix else parent_episode) / "keyframes"
		src_stem = orig_vid.replace(":", "-")
		dst_stem = new_vid.replace(":", "-")
		for suffix_name in ("frame", "crop"):
			src_png = src_kf_dir / f"{src_stem}-{suffix_name}.png"
			if src_png.is_file():
				shutil.copy2(src_png, out_dir / "keyframes" / f"{dst_stem}-{suffix_name}.png")

	graph_src = parent_episode / "graph.json"
	if graph_src.is_file():
		shutil.copy2(graph_src, out_dir / "graph.json")
	else:
		write_json(out_dir / "graph.json", {"nodes": NODES, "edges": EDGES})

	for meta_name in ("audit.private.json", "task.json"):
		src_meta = branch_src / meta_name if (branch_src / meta_name).is_file() else parent_episode / meta_name
		if src_meta.is_file():
			shutil.copy2(src_meta, out_dir / meta_name)

	(out_dir / "events.jsonl").write_text(
		"".join(json.dumps(ev, allow_nan=False) + "\n" for ev in stitched_events),
		encoding="utf-8",
	)

	branch_result_file = suffix_dir / "result.json"
	if branch_result_file.is_file():
		raw_result = json.loads(branch_result_file.read_text(encoding="utf-8"))
	elif (
		isinstance(branch_manifest.get("branch_results"), dict) and str(branch_id) in branch_manifest["branch_results"]
	):
		raw_result = branch_manifest["branch_results"][str(branch_id)]
	else:
		raw_result = branch_manifest["result"]
	result = json.loads(json.dumps(raw_result))
	result["active_branch"] = 0
	result["rewinds"] = 0
	result.pop("perturbation_rewinds", None)  # the stitched episode is one branch; replays never fork
	result["sim_frames"] = n_total
	result["active_frames"] = n_total
	result["views"] = sum(1 for ev in stitched_events if ev.get("kind") == "view")
	result["evidence"] = [_rewrite_view_id(v, src_branch_id, 0) for v in result.get("evidence", [])]
	if result.get("placement_evidence"):
		result["placement_evidence"] = _rewrite_view_id(result["placement_evidence"], src_branch_id, 0)
	for fail in result.get("failures", []):
		if "branch" in fail:
			fail["branch"] = 0
		if "prior_views" in fail:
			fail["prior_views"] = [
				{**pv, "id": _rewrite_view_id(pv.get("id"), src_branch_id, 0)} for pv in fail["prior_views"]
			]
	oracle_used = bool(
		((parent_manifest.get("provenance") or {}).get("experiment") or {}).get("oracle")
		or ((branch_manifest.get("provenance") or {}).get("experiment") or {}).get("oracle")
		or any(ev.get("kind") == "oracle_applied" for ev in stitched_events)
	)
	result["student_eligible"] = bool(result.get("verdict") == "SUCCESS" and not result["rewinds"] and not oracle_used)

	provenance = {
		**parent_manifest.get("provenance", {}),
		**branch_manifest.get("provenance", {}),
		"stitched_from": {
			"parent_episode": str(parent_episode),
			"branch_id": int(branch_id),
			"start_frame": start_frame,
			"checkpoint": ck_id,
			"blind_span": list(blind_span) if blind_span is not None else None,
		},
	}
	manifest = {
		"schema": "armigo-agent-episode/v1",
		"fps": FPS,
		"task": TASK,
		"state_names": STATE_NAMES,
		"branches": [{"id": 0, "start_frame": 0, "actions": n_total}],
		"video": has_video,
		"provenance": provenance,
		"result": result,
	}
	write_json(out_dir / "episode.json", manifest)
	segment_labels(out_dir)
	return manifest


def main():
	import argparse

	parser = argparse.ArgumentParser(description="Export accepted Armigo recordings through LeRobot.")
	parser.add_argument("--out", type=Path, required=True)
	parser.add_argument("--repo-id", default="local/armigo-six-face")
	parser.add_argument(
		"--trim-unsupervised-prefix",
		action="store_true",
		help="Start each episode one row before its first supervised row (paired carry episodes)",
	)
	parser.add_argument(
		"--student-cameras-only",
		action="store_true",
		help="Export only canonical student cameras (omit front and top_full)",
	)
	parser.add_argument(
		"--include-noisy-mask",
		action=argparse.BooleanOptionalAction,
		default=None,
		help="Include mask.noisy feature (default: True for --student-cameras-only, False for legacy 5-camera export)",
	)
	parser.add_argument(
		"--video-dir-map",
		type=Path,
		default=None,
		help="Optional JSON mapping episode_dir -> verified deferred rerender video_dir",
	)
	parser.add_argument("episodes", nargs="+", type=Path)
	args = parser.parse_args()
	extra_kw: dict[str, Any] = {}
	if args.include_noisy_mask is not None:
		extra_kw["include_noisy_mask"] = args.include_noisy_mask
	if args.video_dir_map is not None:
		extra_kw["video_dirs"] = json.loads(args.video_dir_map.read_text(encoding="utf-8"))
	export_lerobot(
		args.episodes,
		args.out,
		args.repo_id,
		trim_unsupervised_prefix=args.trim_unsupervised_prefix,
		student_cameras_only=args.student_cameras_only,
		**extra_kw,
	)


if __name__ == "__main__":
	main()
