"""Data Contract v1 constants, 36-dim state encoding, and normalizer (pure numpy/optional torch/safetensors).

Must remain free of ``mujoco`` and planner diagnostic dependencies so both the repo ``.venv`` and the CUDA
training environment (``E:\\il\\lerobot-cuda``) can import and execute it identically.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..constants import JOINTS, REST_U
from ..joints import normalized_to_rad

CONTRACT_VERSION = "armigo-vla-contract/v1"

ACTION_DIM = 6
STATE_DIM = 36
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

BOWL_VOCAB: tuple[str, ...] = ("", "green", "pink")
BOWL_TO_INDEX: dict[str, int] = {bowl: idx for idx, bowl in enumerate(BOWL_VOCAB)}

POLICY_PARAMS: tuple[str, ...] = ("view", "views", "bowl")
POLICY_FIELDS: tuple[str, ...] = ("kind", "instruction", *POLICY_PARAMS)
FORBIDDEN_DIAGNOSTIC_KEYS: tuple[str, ...] = ("mode", "faces", "target_q", "via_poses", "bottom_px", "xy")

SEGMENT_FRAME_NORM_SCALE = 300.0
SEGMENT_FRAME_NORM_MAX = 5.0

DEFAULT_HORIZON: dict[str, int] = {"smolvla": 50, "act": 100}

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


def policy_inputs_from_segment_dict(kind: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
	"""Build sanitized ``POLICY_FIELDS`` dict from a segment row without exposing planner diagnostics."""
	if kind not in SEGMENT_TEMPLATES:
		raise ValueError(f"unknown segment kind {kind!r}")
	p = dict(params or {})
	instruction = SEGMENT_TEMPLATES[kind].format(**p)
	out: dict[str, Any] = {"kind": kind, "instruction": instruction}
	for key in POLICY_PARAMS:
		val = p.get(key)
		if key in ("view", "views") and val is not None:
			val = int(val)
		elif key == "bowl" and val is not None:
			val = str(val)
		out[key] = val
	return out


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
) -> np.ndarray:
	"""Encode the canonical 36-dim student state + conditioning vector (float32).

	Layout (``STATE_DIM = 36``):
	  - ``[0:6]``   ``q_t`` (rad)
	  - ``[6:12]``  ``q_{t-1}`` (rad; repeats ``q_0`` when ``q_prev`` is None)
	  - ``[12:18]`` ``a_{t-1}`` (rad; ``INITIAL_Q`` when ``prev_action`` is None)
	  - ``[18:19]`` ``gripper_cmd`` (scalar filtered gripper command, rad)
	  - ``[19:31]`` ``segment_kind`` one-hot (12, canonical ``SEGMENT_KINDS_ORDER``)
	  - ``[31:34]`` ``bowl`` one-hot (3, ``("", "green", "pink")``)
	  - ``[34:35]`` ``view_norm`` = ``view / max(views, 1) if view > 0 else 0.0``
	  - ``[35:36]`` ``segment_frame_norm`` = ``min(segment_frame / 300.0, 5.0)``
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
	if bowl_str not in BOWL_TO_INDEX:
		raise ValueError(f"unknown bowl {bowl!r}; expected one of {BOWL_VOCAB}")
	bowl_idx = BOWL_TO_INDEX[bowl_str]

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
	out[31 + bowl_idx] = 1.0
	out[34] = view_norm
	out[35] = seg_frame_norm
	return out


def encode_state_from_task(
	q_t: Any,
	q_prev: Any | None,
	prev_action: Any | None,
	gripper_cmd: Any,
	task: Mapping[str, Any],
	segment_frame: int = 0,
) -> np.ndarray:
	"""Encode the 36-dim vector from a ``policy_inputs`` task dict after checking for zero planner diagnostics."""
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
	"""Per-joint mean/std normalizer for 6-DoF radian actions (and optional 36-dim student state).

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
