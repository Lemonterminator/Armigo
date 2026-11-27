"""Data Contract v1 constants, 38-dim state encoding, and normalizer (pure numpy/optional torch/safetensors).

Must remain free of ``mujoco`` and planner diagnostic dependencies so both the repo ``.venv`` and the CUDA
training environment (``E:\\il\\lerobot-cuda``) can import and execute it identically.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..constants import JOINTS, REST_U
from ..joints import normalized_to_rad

CONTRACT_VERSION = "armigo-vla-contract/v1"

ACTION_DIM = 6
STATE_DIM = 38
IGNORE_INDEX = -100

# Canonical student cameras (current frame only; never include front or top_full).
STUDENT_CAMERA_NAMES: tuple[str, ...] = ("top_rgb_v2", "top_depth_v2", "wrist")
STUDENT_CAMERAS: tuple[str, ...] = tuple(f"observation.images.{name}" for name in STUDENT_CAMERA_NAMES)
FORBIDDEN_CAMERA_NAMES: tuple[str, ...] = ("front", "top_full")

# Canonical segment kind order matching tuple(student_rollout.SEGMENT_KINDS.keys()).
SEGMENT_KINDS_ORDER: tuple[str, ...] = (
	"prepare",
	"scan_view",
	"return_ready",
	"grasp",
	"lift",
	"present_bottom",
	"return_to_lift",
	"lower_on_table",
	"release_on_table",
	"carry_to_bowl",
	"release_in_bowl",
	"let_go",
)
SEGMENT_KIND_TO_INDEX: dict[str, int] = {kind: idx for idx, kind in enumerate(SEGMENT_KINDS_ORDER)}

# Segment templates copied from student_rollout.SEGMENT_KINDS so zero-frame segment synthesis requires no mujoco.
SEGMENT_TEMPLATES: dict[str, str] = {
	"prepare": "Wait at rest, then move the arm to the ready pose with the gripper open.",
	"scan_view": "Move the wrist camera to inspection view {view} of {views} around the brick and hold still.",
	"return_ready": "Return the arm to the ready pose with the gripper open.",
	"grasp": "Grasp the brick on the table.",
	"lift": "Lift the grasped brick straight up.",
	"present_bottom": "Show the bottom face of the held brick to the top camera and hold still.",
	"return_to_lift": "Move the held brick back to the lift pose.",
	"lower_on_table": "Carry the held brick back to its spot on the table and lower it onto the table.",
	"release_on_table": "Open the gripper, let the brick settle on the table and lift the gripper clear.",
	"carry_to_bowl": "Carry the held brick over the {bowl} bowl and lower it into the bowl.",
	"release_in_bowl": "Open the gripper to drop the brick into the bowl, lift clear and return to the rest pose.",
	"let_go": "Open the gripper where it is and rise.",
}

TRAIN_VARIANTS: tuple[int, ...] = (0, 1, 2, 3, 4)
HELD_OUT_VARIANT: int = 5
V3_EXTRA_BOWL_COLOURS: tuple[str, ...] = ("blue", "yellow", "orange", "purple", "white", "black")

INSTRUCTION_VARIANTS: dict[str, tuple[str, ...]] = {
	"prepare": (
		"Wait at rest, then move the arm to the ready pose with the gripper open.",
		"Hold at rest, then drive the arm to ready with the gripper wide.",
		"Pause at rest, then shift the arm into the ready pose with open jaws.",
		"Rest briefly, then raise the arm to the ready pose, gripper unlatched.",
		"Stand by at rest, then command the arm to ready, gripper fully open.",
		"Wait quietly at rest, then transition the arm to ready position, gripper open.",
	),
	"scan_view": (
		"Move the wrist camera to inspection view {view} of {views} around the brick and hold still.",
		"Position the wrist camera at view {view} out of {views} near the brick and pause.",
		"Drive the wrist camera to inspection frame {view} of {views} surrounding the brick and wait.",
		"Set the wrist camera to view {view} of {views} facing the brick and stay still.",
		"Direct the wrist camera toward view {view} of {views} by the brick and hold position.",
		"Shift the wrist camera to inspection slot {view} of {views} next to the brick and halt.",
	),
	"return_ready": (
		"Return the arm to the ready pose with the gripper open.",
		"Drive the arm back to the ready position, gripper unlatched.",
		"Bring the arm straight back to ready pose, gripper wide open.",
		"Retract the arm to the ready stance with open jaws.",
		"Move the arm back into ready posture, gripper open.",
		"Head the arm back toward the ready pose, gripper released.",
	),
	"grasp": (
		"Grasp the brick on the table.",
		"Clamp the brick resting on the table.",
		"Pick up the brick located on the table surface.",
		"Secure the brick sitting upon the table.",
		"Catch the brick resting down on the table.",
		"Grip the target brick positioned on the table.",
	),
	"lift": (
		"Lift the grasped brick straight up.",
		"Raise the held brick vertically upward.",
		"Elevate the gripped brick straight into the air.",
		"Hoist the captured brick directly upward.",
		"Take the secured brick straight up.",
		"Pull the held brick upward off the table.",
	),
	"present_bottom": (
		"Show the bottom face of the held brick to the top camera and hold still.",
		"Present the underside of the gripped brick toward the top camera and pause.",
		"Display the bottom surface of the held brick to the overhead camera and halt.",
		"Expose the underside of the secured brick to the top camera and hold position.",
		"Turn the bottom face of the caught brick upward to the top camera and wait.",
		"Reveal the lower face of the held brick to the upper camera and stay still.",
	),
	"return_to_lift": (
		"Move the held brick back to the lift pose.",
		"Bring the gripped brick back to the lifting posture.",
		"Shift the carried brick to the lift position.",
		"Return the secured brick back to the lift spot.",
		"Drive the held brick backward into lift stance.",
		"Recollect the carried brick to the lift pose.",
	),
	"lower_on_table": (
		"Carry the held brick back to its spot on the table and lower it onto the table.",
		"Transport the gripped brick to its table location and set it down.",
		"Move the carried brick back over its place and lower onto the table.",
		"Take the secured brick back to its table position and descend.",
		"Transfer the held brick back to its home spot and lower to the table.",
		"Bring the carried brick back to its original table point and place down.",
	),
	"release_on_table": (
		"Open the gripper, let the brick settle on the table and lift the gripper clear.",
		"Unlatch the jaws, allow the brick to rest on the table and raise the gripper.",
		"Release the gripper, let the brick settle on table surface and retract clear.",
		"Open the fingers, let the brick drop to the table and lift gripper away.",
		"Drop the grip, permit the brick to rest on the table and elevate clear.",
		"Disengage the gripper, let the brick settle on the table and rise up.",
	),
	"carry_to_bowl": (
		"Carry the held brick over the {bowl} bowl and lower it into the bowl.",
		"Transport the gripped brick above the {bowl} bowl and descend into it.",
		"Move the carried brick over the {bowl} bowl and lower inside.",
		"Take the secured brick across the {bowl} bowl and lower down into the bowl.",
		"Transfer the held brick above the {bowl} bowl and lower it into position.",
		"Bring the carried brick over to the {bowl} bowl and descend into the container.",
	),
	"release_in_bowl": (
		"Open the gripper to drop the brick into the bowl, lift clear and return to the rest pose.",
		"Unlatch the jaws to release the brick into the bowl, retract and go to rest.",
		"Drop the grip to deposit the brick into the bowl, rise and return to rest posture.",
		"Open fingers to dump the brick inside the bowl, lift clear and return to rest.",
		"Release the brick into the bowl container, elevate clear and move to rest pose.",
		"Disengage jaws to drop the brick into the bowl, pull back and return to rest.",
	),
	"let_go": (
		"Open the gripper where it is and rise.",
		"Unlatch the fingers in place and elevate.",
		"Release the grip right here and lift up.",
		"Open the jaws where situated and rise.",
		"Let go of the grip on the spot and ascend.",
		"Disengage the gripper in position and pull up.",
	),
}

BOWL_VOCAB: tuple[str, ...] = ("", "green", "pink")
BOWL_TO_INDEX: dict[str, int] = {bowl: idx for idx, bowl in enumerate(BOWL_VOCAB)}

# ``roll_deg``: the wrist roll the controller chose for this segment (grasp: which brick faces the jaws pinch, picked
# from the detector's cleared / suspect faces; scan_view / present_bottom: the planned view pose). A segment-level
# decision (R3), so it is part of the instruction; None for kinds without one.
POLICY_PARAMS: tuple[str, ...] = ("view", "views", "bowl", "roll_deg")
ROLL_SUFFIX = " Use a wrist roll of {roll_deg} degrees."
POLICY_FIELDS: tuple[str, ...] = ("kind", "instruction", *POLICY_PARAMS)
FORBIDDEN_DIAGNOSTIC_KEYS: tuple[str, ...] = ("mode", "faces", "target_q", "via_poses", "bottom_px", "xy")

SEGMENT_FRAME_NORM_SCALE = 300.0
SEGMENT_FRAME_NORM_MAX = 5.0

DEFAULT_HORIZON: dict[str, int] = {"smolvla": 50, "act": 100}


@dataclass(frozen=True)
class StudentInputContract:
	"""Explicit versioned contract shared by offline dataset, replay, and online ``/predict`` serving."""

	version: str
	zero_bowl: bool
	cameras: tuple[str, ...] = STUDENT_CAMERA_NAMES
	state_dim: int = STATE_DIM
	instruction_mode: str = "template"

	def __post_init__(self) -> None:
		if self.version not in ("v2", "v3"):
			raise ValueError(f"unsupported StudentInputContract version {self.version!r}; expected 'v2' or 'v3'")
		if tuple(self.cameras) != STUDENT_CAMERA_NAMES:
			raise ValueError(f"cameras must be {STUDENT_CAMERA_NAMES}, got {self.cameras!r}")
		if int(self.state_dim) != STATE_DIM:
			raise ValueError(f"state_dim must be {STATE_DIM}, got {self.state_dim!r}")

	def to_dict(self) -> dict[str, Any]:
		return {
			"schema": CONTRACT_VERSION,
			"version": self.version,
			"zero_bowl": bool(self.zero_bowl),
			"cameras": list(self.cameras),
			"state_dim": int(self.state_dim),
			"instruction_mode": str(self.instruction_mode),
		}

	@classmethod
	def for_version(cls, version: str) -> StudentInputContract:
		v = str(version).strip().lower()
		if v == "v3":
			return STUDENT_INPUT_CONTRACT_V3
		if v == "v2":
			return STUDENT_INPUT_CONTRACT_V2
		raise ValueError(f"unknown student input contract version {version!r}; expected 'v2' or 'v3'")

	@classmethod
	def from_zero_bowl(cls, zero_bowl: bool) -> StudentInputContract:
		return STUDENT_INPUT_CONTRACT_V3 if bool(zero_bowl) else STUDENT_INPUT_CONTRACT_V2

	@classmethod
	def from_dict(cls, data: Mapping[str, Any]) -> StudentInputContract:
		version = data.get("version")
		if version is None:
			version = "v3" if bool(data.get("zero_bowl", False)) else "v2"
		base = cls.for_version(str(version))
		zero_bowl = bool(data["zero_bowl"]) if "zero_bowl" in data and data["zero_bowl"] is not None else base.zero_bowl
		cameras = tuple(data.get("cameras", base.cameras))
		state_dim = int(data.get("state_dim", base.state_dim))
		instruction_mode = str(data.get("instruction_mode", base.instruction_mode))
		return cls(
			version=base.version,
			zero_bowl=zero_bowl,
			cameras=cameras,
			state_dim=state_dim,
			instruction_mode=instruction_mode,
		)


STUDENT_INPUT_CONTRACT_V2 = StudentInputContract(
	version="v2",
	zero_bowl=False,
	cameras=STUDENT_CAMERA_NAMES,
	state_dim=STATE_DIM,
	instruction_mode="template",
)

STUDENT_INPUT_CONTRACT_V3 = StudentInputContract(
	version="v3",
	zero_bowl=True,
	cameras=STUDENT_CAMERA_NAMES,
	state_dim=STATE_DIM,
	instruction_mode="paraphrase",
)


def resolve_student_input_contract(
	*,
	scene: str | None = None,
	bowls: Sequence[str] | None = None,
	input_contract: StudentInputContract | Mapping[str, Any] | str | None = None,
	zero_bowl: bool | None = None,
) -> StudentInputContract:
	"""Resolve a :class:`StudentInputContract` from explicit overrides or episode/payload metadata."""
	if input_contract is not None:
		if isinstance(input_contract, StudentInputContract):
			resolved = input_contract
		elif isinstance(input_contract, str):
			resolved = StudentInputContract.for_version(input_contract)
		elif isinstance(input_contract, Mapping):
			resolved = StudentInputContract.from_dict(input_contract)
		else:
			raise TypeError(f"unsupported input_contract {input_contract!r}")
		if zero_bowl is not None and bool(zero_bowl) != resolved.zero_bowl:
			raise ValueError(f"conflicting zero_bowl={zero_bowl!r} and input_contract.zero_bowl={resolved.zero_bowl!r}")
		return resolved
	if zero_bowl is not None:
		return StudentInputContract.from_zero_bowl(bool(zero_bowl))
	is_v3 = (scene == "v3") or (bowls is not None and any(str(b) in V3_EXTRA_BOWL_COLOURS for b in bowls))
	return STUDENT_INPUT_CONTRACT_V3 if is_v3 else STUDENT_INPUT_CONTRACT_V2


# Exact match to twin_mujoco.INITIAL_Q (ExpertParams.open_g = 0.85).
DEFAULT_OPEN_G = 0.85
INITIAL_Q = np.array([*normalized_to_rad(np.array(REST_U, dtype=np.float64))[:5], DEFAULT_OPEN_G], dtype=np.float64)


def assert_no_planner_diagnostics(mapping: Mapping[str, Any]) -> None:
	"""Raise ``ValueError`` if ``mapping`` contains any planner diagnostics or unexpected keys."""
	bad = set(mapping) & set(FORBIDDEN_DIAGNOSTIC_KEYS)
	if bad:
		raise ValueError(f"planner diagnostic key(s) forbidden in student inputs: {sorted(bad)}")
	extra = set(mapping) - set(POLICY_FIELDS)
	if extra:
		raise ValueError(f"non-policy key(s) forbidden in student task dict: {sorted(extra)}")


def policy_inputs_from_segment_dict(
	kind: str, params: Mapping[str, Any] | None = None, *, variant: int = 0
) -> dict[str, Any]:
	"""Build sanitized ``POLICY_FIELDS`` dict from a segment row without exposing planner diagnostics."""
	if kind not in SEGMENT_TEMPLATES:
		raise ValueError(f"unknown segment kind {kind!r}")
	p = dict(params or {})
	instruction = format_instruction(kind, p, variant=variant)
	out: dict[str, Any] = {"kind": kind, "instruction": instruction}
	for key in POLICY_PARAMS:
		val = p.get(key)
		if key in ("view", "views") and val is not None:
			val = int(val)
		elif key == "bowl" and val is not None:
			val = str(val)
		elif key == "roll_deg" and val is not None:
			val = float(val)
		out[key] = val
	return out


def format_instruction(kind: str, params: Mapping[str, Any] | None = None, variant: int = 0) -> str:
	"""The segment instruction: the kind's template, plus :data:`ROLL_SUFFIX` when the segment carries a roll."""
	if kind not in INSTRUCTION_VARIANTS:
		raise ValueError(f"unknown segment kind {kind!r}")
	variants = INSTRUCTION_VARIANTS[kind]
	v_idx = int(variant)
	if not (0 <= v_idx < len(variants)):
		raise ValueError(f"invalid variant {v_idx} for kind {kind!r} (expected 0..{len(variants) - 1})")
	p = dict(params or {})
	text = variants[v_idx].format(**p)
	if p.get("roll_deg") is not None:
		text += ROLL_SUFFIX.format(roll_deg=p["roll_deg"])
	return text


def roll_features(roll_deg: Any | None) -> tuple[float, float]:
	"""State slots ``[36:38]`` for a segment roll: ``(roll in rad, 1.0)``, or ``(0.0, 0.0)`` without one."""
	if roll_deg is None:
		return 0.0, 0.0
	val = float(roll_deg)
	if not np.isfinite(val):
		raise ValueError(f"roll_deg must be finite, got {roll_deg!r}")
	return float(np.radians(val)), 1.0


def _as_vec6(values: Any, name: str) -> np.ndarray:
	arr = np.asarray(values, dtype=np.float32).reshape(-1)
	if arr.shape != (ACTION_DIM,):
		raise ValueError(f"{name} must have shape ({ACTION_DIM},), got {np.asarray(values).shape}")
	if not np.isfinite(arr).all():
		raise ValueError(f"{name} contains non-finite values")
	return arr


def encode_state(
	q_t: Any,
	q_prev: Any | None,
	prev_action: Any | None,
	gripper_cmd: Any,
	segment_kind: str,
	bowl: str | None = "",
	view: int | None = 0,
	views: int | None = 0,
	segment_frame: int = 0,
	roll_deg: float | None = None,
	*,
	zero_bowl: bool = False,
) -> np.ndarray:
	"""Encode the canonical 38-dim student state + conditioning vector (float32).

	Layout (``STATE_DIM = 38``):
	  - ``[0:6]``   ``q_t`` (rad)
	  - ``[6:12]``  ``q_{t-1}`` (rad; repeats ``q_0`` when ``q_prev`` is None)
	  - ``[12:18]`` ``a_{t-1}`` (rad; ``INITIAL_Q`` when ``prev_action`` is None)
	  - ``[18:19]`` ``gripper_cmd`` (scalar filtered gripper command, rad)
	  - ``[19:31]`` ``segment_kind`` one-hot (12, canonical ``SEGMENT_KINDS_ORDER``)
	  - ``[31:34]`` ``bowl`` one-hot (3, ``("", "green", "pink")``)
	  - ``[34:35]`` ``view_norm`` = ``view / max(views, 1) if view > 0 else 0.0``
	  - ``[35:36]`` ``segment_frame_norm`` = ``min(segment_frame / 300.0, 5.0)``
	  - ``[36:38]`` segment wrist roll: ``(radians(roll_deg), 1.0)``, or ``(0.0, 0.0)`` when the segment has none
	"""
	q_cur = _as_vec6(q_t, "q_t")
	q_prv = q_cur if q_prev is None else _as_vec6(q_prev, "q_prev")
	a_prv = INITIAL_Q.astype(np.float32) if prev_action is None else _as_vec6(prev_action, "prev_action")

	grip_arr = np.asarray(gripper_cmd, dtype=np.float32).reshape(-1)
	if grip_arr.size != 1 or not np.isfinite(grip_arr).all():
		raise ValueError(f"gripper_cmd must be a finite scalar, got {gripper_cmd!r}")
	grip_val = float(grip_arr[0])

	if segment_kind not in SEGMENT_KIND_TO_INDEX:
		raise ValueError(f"unknown segment_kind {segment_kind!r}")
	kind_idx = SEGMENT_KIND_TO_INDEX[segment_kind]

	bowl_str = "" if bowl is None else str(bowl)
	if bowl_str not in BOWL_TO_INDEX and bowl_str not in V3_EXTRA_BOWL_COLOURS:
		raise ValueError(f"unknown bowl {bowl!r}; expected one of {BOWL_VOCAB} or {V3_EXTRA_BOWL_COLOURS}")

	view_i = 0 if view is None else int(view)
	views_i = 0 if views is None else int(views)
	if view_i < 0 or views_i < 0:
		raise ValueError(f"view and views must be non-negative, got view={view_i}, views={views_i}")
	view_norm = float(view_i) / max(views_i, 1) if view_i > 0 else 0.0

	seg_frame_i = int(segment_frame)
	if seg_frame_i < 0:
		raise ValueError(f"segment_frame must be >= 0, got {seg_frame_i}")
	seg_frame_norm = min(float(seg_frame_i) / SEGMENT_FRAME_NORM_SCALE, SEGMENT_FRAME_NORM_MAX)

	out = np.zeros(STATE_DIM, dtype=np.float32)
	out[0:6] = q_cur
	out[6:12] = q_prv
	out[12:18] = a_prv
	out[18] = grip_val
	out[19 + kind_idx] = 1.0
	if not zero_bowl and bowl_str in BOWL_TO_INDEX:
		bowl_idx = BOWL_TO_INDEX[bowl_str]
		out[31 + bowl_idx] = 1.0
	out[34] = view_norm
	out[35] = seg_frame_norm
	out[36], out[37] = roll_features(roll_deg)
	return out


def encode_state_from_task(
	q_t: Any,
	q_prev: Any | None,
	prev_action: Any | None,
	gripper_cmd: Any,
	task: Mapping[str, Any],
	segment_frame: int = 0,
	*,
	zero_bowl: bool = False,
) -> np.ndarray:
	"""Encode the 38-dim vector from a ``policy_inputs`` task dict after checking for zero planner diagnostics."""
	assert_no_planner_diagnostics(task)
	return encode_state(
		q_t=q_t,
		q_prev=q_prev,
		prev_action=prev_action,
		gripper_cmd=gripper_cmd,
		segment_kind=str(task["kind"]),
		bowl=task.get("bowl"),
		view=task.get("view"),
		views=task.get("views"),
		segment_frame=segment_frame,
		roll_deg=task.get("roll_deg"),
		zero_bowl=zero_bowl,
	)


def encode_student_state(
	q_t: Any,
	q_prev: Any | None,
	prev_action: Any | None,
	gripper_cmd: Any,
	task: Mapping[str, Any],
	segment_frame: int = 0,
	*,
	input_contract: StudentInputContract | Mapping[str, Any] | str | None = None,
	zero_bowl: bool | None = None,
) -> np.ndarray:
	"""Single canonical state encoder shared by offline dataset, replay, and online ``/predict`` serving."""
	contract = resolve_student_input_contract(input_contract=input_contract, zero_bowl=zero_bowl)
	return encode_state_from_task(
		q_t=q_t,
		q_prev=q_prev,
		prev_action=prev_action,
		gripper_cmd=gripper_cmd,
		task=task,
		segment_frame=segment_frame,
		zero_bowl=contract.zero_bowl,
	)


def end_step_target_from_segment(
	chunk_valid_len: int,
	horizon: int,
	end_reason: str,
	*,
	is_blind: bool = False,
	supervise_action: bool = True,
	is_zero_frame: bool = False,
) -> int:
	"""Compute categorical end-step target in ``0..H+1`` or ``IGNORE_INDEX (-100)``.

	Classes:
	  - ``0``: ended immediately (synthesized zero-frame segments such as ``let_go``)
	  - ``1..H``: execute ``k`` actions ``a_0..a_{k-1}``, then stop at step ``k``
	  - ``H + 1``: segment end is not within this ``H``-step chunk (``chunk_valid_len > H``)

	Returns ``-100`` whenever ``end_reason != "done"`` OR ``is_blind`` OR ``not supervise_action``.
	"""
	if end_reason != "done" or is_blind or not supervise_action:
		return IGNORE_INDEX
	if is_zero_frame:
		return 0
	cvl = int(chunk_valid_len)
	if cvl < 1:
		raise ValueError(f"chunk_valid_len must be >= 1 for non-zero-frame segments, got {cvl}")
	return cvl if cvl <= horizon else horizon + 1


def _is_torch_tensor(x: Any) -> bool:
	return type(x).__module__.startswith("torch")


@dataclass
class ActionNormalizer:
	"""Per-joint mean/std normalizer for 6-DoF radian actions (and optional 38-dim student state).

	Stats MUST be fitted exclusively on the training split. Supports both numpy arrays and torch tensors,
	and serializes cleanly to JSON and/or safetensors alongside checkpoints.
	"""

	action_mean: np.ndarray
	action_std: np.ndarray
	state_mean: np.ndarray | None = None
	state_std: np.ndarray | None = None
	eps: float = 1e-6
	joint_names: tuple[str, ...] = JOINTS

	def __post_init__(self) -> None:
		self.action_mean = np.asarray(self.action_mean, dtype=np.float32).reshape(ACTION_DIM)
		self.action_std = np.asarray(self.action_std, dtype=np.float32).reshape(ACTION_DIM)
		self.action_std = np.where(self.action_std < self.eps, np.float32(1.0), self.action_std).astype(np.float32)
		if self.state_mean is not None:
			self.state_mean = np.asarray(self.state_mean, dtype=np.float32).reshape(STATE_DIM)
		if self.state_std is not None:
			self.state_std = np.asarray(self.state_std, dtype=np.float32).reshape(STATE_DIM)
			self.state_std = np.where(self.state_std < self.eps, np.float32(1.0), self.state_std).astype(np.float32)
		self.joint_names = tuple(self.joint_names)

	@classmethod
	def identity(cls) -> ActionNormalizer:
		return cls(
			action_mean=np.zeros(ACTION_DIM, dtype=np.float32),
			action_std=np.ones(ACTION_DIM, dtype=np.float32),
			state_mean=np.zeros(STATE_DIM, dtype=np.float32),
			state_std=np.ones(STATE_DIM, dtype=np.float32),
		)

	@classmethod
	def fit(
		cls,
		actions: np.ndarray,
		states: np.ndarray | None = None,
		*,
		eps: float = 1e-6,
	) -> ActionNormalizer:
		"""Fit mean and std on training-split arrays only."""
		act = np.asarray(actions, dtype=np.float64)
		if act.ndim == 3:
			act = act.reshape(-1, ACTION_DIM)
		elif act.ndim != 2 or act.shape[-1] != ACTION_DIM:
			raise ValueError(f"actions must have shape (N, {ACTION_DIM}) or (N, H, {ACTION_DIM}), got {act.shape}")
		if len(act) == 0:
			raise ValueError("cannot fit ActionNormalizer on zero samples")
		a_mean = act.mean(axis=0).astype(np.float32)
		a_std = act.std(axis=0).astype(np.float32)
		a_std = np.where(a_std < eps, np.float32(1.0), a_std).astype(np.float32)

		s_mean = s_std = None
		if states is not None:
			st = np.asarray(states, dtype=np.float64)
			if st.ndim != 2 or st.shape[-1] != STATE_DIM:
				raise ValueError(f"states must have shape (N, {STATE_DIM}), got {st.shape}")
			if len(st) == 0:
				raise ValueError("cannot fit state stats on zero samples")
			s_mean = st.mean(axis=0).astype(np.float32)
			s_std = st.std(axis=0).astype(np.float32)
			s_std = np.where(s_std < eps, np.float32(1.0), s_std).astype(np.float32)
		return cls(action_mean=a_mean, action_std=a_std, state_mean=s_mean, state_std=s_std, eps=eps)

	def normalize_action(self, action: Any) -> Any:
		if _is_torch_tensor(action):
			import torch

			mean = torch.as_tensor(self.action_mean, device=action.device, dtype=action.dtype)
			std = torch.as_tensor(self.action_std, device=action.device, dtype=action.dtype)
			return (action - mean) / std
		arr = np.asarray(action, dtype=np.float32)
		return (arr - self.action_mean) / self.action_std

	def unnormalize_action(self, action_norm: Any) -> Any:
		if _is_torch_tensor(action_norm):
			import torch

			mean = torch.as_tensor(self.action_mean, device=action_norm.device, dtype=action_norm.dtype)
			std = torch.as_tensor(self.action_std, device=action_norm.device, dtype=action_norm.dtype)
			return action_norm * std + mean
		arr = np.asarray(action_norm, dtype=np.float32)
		return arr * self.action_std + self.action_mean

	# Convenient aliases
	normalize = normalize_action
	unnormalize = unnormalize_action

	def normalize_state(self, state: Any) -> Any:
		if self.state_mean is None or self.state_std is None:
			return state
		if _is_torch_tensor(state):
			import torch

			mean = torch.as_tensor(self.state_mean, device=state.device, dtype=state.dtype)
			std = torch.as_tensor(self.state_std, device=state.device, dtype=state.dtype)
			return (state - mean) / std
		arr = np.asarray(state, dtype=np.float32)
		return (arr - self.state_mean) / self.state_std

	def unnormalize_state(self, state_norm: Any) -> Any:
		if self.state_mean is None or self.state_std is None:
			return state_norm
		if _is_torch_tensor(state_norm):
			import torch

			mean = torch.as_tensor(self.state_mean, device=state_norm.device, dtype=state_norm.dtype)
			std = torch.as_tensor(self.state_std, device=state_norm.device, dtype=state_norm.dtype)
			return state_norm * std + mean
		arr = np.asarray(state_norm, dtype=np.float32)
		return arr * self.state_std + self.state_mean

	def to_dict(self) -> dict[str, Any]:
		payload: dict[str, Any] = {
			"schema": CONTRACT_VERSION,
			"unit": "rad",
			"joint_names": list(self.joint_names),
			"eps": float(self.eps),
			"action_mean": self.action_mean.tolist(),
			"action_std": self.action_std.tolist(),
		}
		if self.state_mean is not None and self.state_std is not None:
			payload["state_mean"] = self.state_mean.tolist()
			payload["state_std"] = self.state_std.tolist()
		return payload

	@classmethod
	def from_dict(cls, payload: Mapping[str, Any]) -> ActionNormalizer:
		return cls(
			action_mean=np.asarray(payload["action_mean"], dtype=np.float32),
			action_std=np.asarray(payload["action_std"], dtype=np.float32),
			state_mean=np.asarray(payload["state_mean"], dtype=np.float32)
			if payload.get("state_mean") is not None
			else None,
			state_std=np.asarray(payload["state_std"], dtype=np.float32)
			if payload.get("state_std") is not None
			else None,
			eps=float(payload.get("eps", 1e-6)),
			joint_names=tuple(payload.get("joint_names", JOINTS)),
		)

	def save(self, path: str | Path) -> Path:
		"""Save normalizer to ``.json`` or ``.safetensors``."""
		target = Path(path)
		target.parent.mkdir(parents=True, exist_ok=True)
		if target.suffix == ".safetensors":
			from safetensors.numpy import save_file

			tensors: dict[str, np.ndarray] = {
				"action_mean": self.action_mean.astype(np.float32),
				"action_std": self.action_std.astype(np.float32),
			}
			if self.state_mean is not None and self.state_std is not None:
				tensors["state_mean"] = self.state_mean.astype(np.float32)
				tensors["state_std"] = self.state_std.astype(np.float32)
			metadata = {"schema": CONTRACT_VERSION, "joint_names": json.dumps(list(self.joint_names))}
			save_file(tensors, str(target), metadata=metadata)
		else:
			target.write_text(json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n", encoding="utf-8")
		return target

	@classmethod
	def load(cls, path: str | Path) -> ActionNormalizer:
		"""Load normalizer from ``.json`` or ``.safetensors``."""
		src = Path(path)
		if src.suffix == ".safetensors":
			from safetensors.numpy import load_file

			tensors = load_file(str(src))
			return cls(
				action_mean=tensors["action_mean"],
				action_std=tensors["action_std"],
				state_mean=tensors.get("state_mean"),
				state_std=tensors.get("state_std"),
			)
		return cls.from_dict(json.loads(src.read_text(encoding="utf-8")))

	def save_alongside_checkpoint(self, out_dir: str | Path) -> dict[str, Path]:
		"""Write ``normalizer.json`` (always) and ``normalizer.safetensors`` (when safetensors is available)."""
		out = Path(out_dir)
		out.mkdir(parents=True, exist_ok=True)
		written = {"json": self.save(out / "normalizer.json")}
		try:
			import safetensors.numpy  # noqa: F401
		except ImportError:
			return written
		written["safetensors"] = self.save(out / "normalizer.safetensors")
		return written
