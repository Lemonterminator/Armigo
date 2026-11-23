"""Segment-chunk dataset over LeRobot v3 and in-memory episodes (plan S7/S9, Data Contract v1)."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .contract import (
	ACTION_DIM,
	BOWL_VOCAB,
	IGNORE_INDEX,
	INITIAL_Q,
	SEGMENT_KINDS_ORDER,
	STATE_DIM,
	STUDENT_CAMERAS,
	ActionNormalizer,
	encode_state,
	encode_state_from_task,
	end_step_target_from_segment,
	policy_inputs_from_segment_dict,
)

try:
	import torch
	from torch.utils.data import Dataset as _BaseDataset
except ImportError:  # pragma: no cover - repo .venv lacks torch
	torch = None
	_BaseDataset = object


_PYAV_CACHE_SIZE = 16  # open containers kept per process (3 student cameras x a few chunk files)
_pyav_containers: dict[str, Any] = {}  # per process: each DataLoader worker builds its own


def _decode_frame_pyav_cached(video_path: str, ts: float, tolerance_s: float) -> Any:
	"""One RGB frame at ``ts`` as float32 CHW in [0, 1], decoded like LeRobot's ``decode_video_frames_pyav``.

	Same seek (nearest keyframe at or before ``ts`` with a one-frame margin), decode-forward and nearest-frame
	tolerance rule, but the container stays open between calls. LeRobot opens the chunk file (~180 MB, many
	episodes) for every frame, which made the training loader CPU-bound (S9 ACT run 03).

	Only DataLoader worker processes keep containers open (their handles go away with the worker); in the main
	process the file is opened per call, so nothing holds a lock on it (Windows) after the dataset is used."""
	import av
	from torch.utils.data import get_worker_info

	if get_worker_info() is None:
		with av.open(video_path) as container:
			return _decode_nearest(container, container.streams.video[0], video_path, ts, tolerance_s)
	entry = _pyav_containers.pop(video_path, None)
	if entry is None:
		container = av.open(video_path)
		entry = (container, container.streams.video[0])
		while len(_pyav_containers) >= _PYAV_CACHE_SIZE:
			_pyav_containers.pop(next(iter(_pyav_containers)))[0].close()
	_pyav_containers[video_path] = entry  # most recently used last
	return _decode_nearest(*entry, video_path, ts, tolerance_s)


def _decode_nearest(container: Any, stream: Any, video_path: str, ts: float, tolerance_s: float) -> Any:
	container.seek(round(ts / stream.time_base) - 1, backward=True, any_frame=False, stream=stream)
	best, best_dist = None, float("inf")
	for frame in container.decode(stream):
		if frame.pts is None:
			continue
		cur = float(frame.pts * stream.time_base)
		dist = abs(cur - ts)
		if dist < best_dist:
			best, best_dist = frame, dist
		if cur >= ts:
			break
	if best is None or not best_dist < tolerance_s:
		raise RuntimeError(f"no frame within {tolerance_s}s of t={ts:.4f} in {video_path} (closest {best_dist:.4f}s)")
	arr = best.to_ndarray(format="rgb24")
	return torch.from_numpy(arr).permute(2, 0, 1).to(dtype=torch.float32).div_(255.0).contiguous()


def _to_float_chw(img: Any) -> np.ndarray:
	"""Convert HWC uint8/float or CHW float/uint8 image into contiguous float32 CHW in [0, 1]."""
	if torch is not None and isinstance(img, torch.Tensor):
		arr = img.detach().cpu().numpy()
	else:
		arr = np.asarray(img)
	if arr.ndim != 3:
		raise ValueError(f"expected 3D image tensor, got shape {arr.shape}")
	if arr.shape[0] != 3 and arr.shape[-1] == 3:
		arr = np.transpose(arr, (2, 0, 1))
	if arr.shape[0] != 3:
		raise ValueError(f"expected 3-channel image, got shape {arr.shape}")
	if arr.dtype == np.uint8:
		return np.ascontiguousarray(arr, dtype=np.float32) / 255.0
	arr_f = np.ascontiguousarray(arr, dtype=np.float32)
	if arr_f.size and arr_f.max() > 1.0:
		arr_f = arr_f / 255.0
	return np.clip(arr_f, 0.0, 1.0)


class StudentImageAugmentor:
	"""Training-time image augmentations honoring Data Contract v1.

	- ``top_rgb_v2`` & ``wrist``: random crop-resize, color jitter (brightness/contrast/saturation/hue), Gaussian noise.
	- ``top_depth_v2``: mild spatial crop-resize and mild Gaussian noise ONLY (NO color/hue/brightness/contrast jitter,
	  because ``top_depth_v2`` encodes fine/coarse table-relative height in R/G and validity in B).
	"""

	def __init__(
		self,
		*,
		rgb_crop_prob: float = 0.8,
		rgb_crop_scale: tuple[float, float] = (0.90, 1.0),
		rgb_crop_ratio: tuple[float, float] = (0.95, 1.05),
		brightness: float = 0.2,
		contrast: float = 0.2,
		saturation: float = 0.2,
		hue: float = 0.05,
		rgb_noise_std: float = 0.015,
		depth_crop_prob: float = 0.5,
		depth_crop_scale: tuple[float, float] = (0.96, 1.0),
		depth_crop_ratio: tuple[float, float] = (0.97, 1.03),
		depth_noise_std: float = 0.008,
		seed: int | None = None,
	) -> None:
		self.rgb_crop_prob = float(rgb_crop_prob)
		self.rgb_crop_scale = rgb_crop_scale
		self.rgb_crop_ratio = rgb_crop_ratio
		self.brightness = float(brightness)
		self.contrast = float(contrast)
		self.saturation = float(saturation)
		self.hue = float(hue)
		self.rgb_noise_std = float(rgb_noise_std)
		self.depth_crop_prob = float(depth_crop_prob)
		self.depth_crop_scale = depth_crop_scale
		self.depth_crop_ratio = depth_crop_ratio
		self.depth_noise_std = float(depth_noise_std)
		self._rng = np.random.default_rng(seed)

	def _random_resized_crop_torch(self, img: Any, scale: tuple[float, float], ratio: tuple[float, float]) -> Any:
		import torch.nn.functional as F
		import torchvision.transforms.functional as TF

		_, h, w = img.shape
		area = h * w
		target_area = area * float(self._rng.uniform(scale[0], scale[1]))
		log_ratio = (np.log(ratio[0]), np.log(ratio[1]))
		aspect = float(np.exp(self._rng.uniform(log_ratio[0], log_ratio[1])))
		crop_w = max(1, min(w, round(np.sqrt(target_area * aspect))))
		crop_h = max(1, min(h, round(np.sqrt(target_area / aspect))))
		top = int(self._rng.integers(0, h - crop_h + 1))
		left = int(self._rng.integers(0, w - crop_w + 1))
		cropped = TF.crop(img, top, left, crop_h, crop_w)
		resized = F.interpolate(cropped.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False).squeeze(0)
		return resized.clamp(0.0, 1.0)

	def _augment_rgb_torch(self, img: Any) -> Any:
		import torchvision.transforms.functional as TF

		if self._rng.random() < self.rgb_crop_prob:
			img = self._random_resized_crop_torch(img, self.rgb_crop_scale, self.rgb_crop_ratio)
		if self.brightness > 0:
			img = TF.adjust_brightness(
				img, float(self._rng.uniform(max(0.0, 1.0 - self.brightness), 1.0 + self.brightness))
			)
		if self.contrast > 0:
			img = TF.adjust_contrast(img, float(self._rng.uniform(max(0.0, 1.0 - self.contrast), 1.0 + self.contrast)))
		if self.saturation > 0:
			img = TF.adjust_saturation(
				img, float(self._rng.uniform(max(0.0, 1.0 - self.saturation), 1.0 + self.saturation))
			)
		if self.hue > 0:
			img = TF.adjust_hue(img, float(self._rng.uniform(-self.hue, self.hue)))
		if self.rgb_noise_std > 0:
			noise = torch.randn_like(img) * self.rgb_noise_std
			img = img + noise
		return img.clamp(0.0, 1.0)

	def _augment_depth_torch(self, img: Any) -> Any:
		# Strictly NO color/hue/brightness/contrast jitter on top_depth_v2!
		if self._rng.random() < self.depth_crop_prob:
			img = self._random_resized_crop_torch(img, self.depth_crop_scale, self.depth_crop_ratio)
		if self.depth_noise_std > 0:
			noise = torch.randn_like(img) * self.depth_noise_std
			img = img + noise
		return img.clamp(0.0, 1.0)

	def _augment_numpy(self, img: np.ndarray, *, is_depth: bool) -> np.ndarray:
		out = img.copy()
		if not is_depth and self.brightness > 0:
			factor = float(self._rng.uniform(max(0.0, 1.0 - self.brightness), 1.0 + self.brightness))
			out = out * factor
		std = self.depth_noise_std if is_depth else self.rgb_noise_std
		if std > 0:
			out = out + self._rng.normal(0.0, std, size=out.shape).astype(np.float32)
		return np.clip(out, 0.0, 1.0).astype(np.float32)

	def __call__(self, images: dict[str, Any]) -> dict[str, Any]:
		out: dict[str, Any] = {}
		for key, img in images.items():
			is_depth = key.endswith("top_depth_v2")
			if torch is not None and isinstance(img, torch.Tensor):
				out[key] = self._augment_depth_torch(img) if is_depth else self._augment_rgb_torch(img)
			else:
				out[key] = self._augment_numpy(np.asarray(img, dtype=np.float32), is_depth=is_depth)
		return out


def _segment_roll_rows(ep: _EpisodeData) -> tuple[list[float | None], list[str]]:
	"""Per-row segment ``roll_deg`` (None without one) and instruction, from ``meta/segments.json``'s params.

	The roll is a segment param (not a per-frame column), so rows take it from their segment (``segment.index`` =
	the segment row's ``id``). A row whose segment carries a roll gets the instruction formatted with it."""
	params_by_id = {int(m["id"]): dict(m.get("params") or {}) for m in ep.segments_meta if "id" in m}
	rolls: list[float | None] = []
	tasks: list[str] = []
	for t in range(ep.length):
		params = params_by_id.get(int(ep.segment_index[t]), {})
		roll = params.get("roll_deg")
		rolls.append(None if roll is None else float(roll))
		tasks.append(
			ep.task[t] if roll is None else policy_inputs_from_segment_dict(ep.segment_kind[t], params)["instruction"]
		)
	return rolls, tasks


@dataclass
class _EpisodeData:
	episode_index: int
	length: int
	state: np.ndarray  # (T, 6) float32
	prev_action: np.ndarray  # (T, 6) float32
	gripper_cmd: np.ndarray  # (T,) float32
	action: np.ndarray  # (T, 6) float32
	segment_index: np.ndarray  # (T,) int64
	segment_kind: list[str]
	segment_done: np.ndarray  # (T,) bool
	chunk_valid_len: np.ndarray  # (T,) int64
	segment_frame: np.ndarray  # (T,) int64
	end_reason: list[str]
	bowl: list[str]
	view: np.ndarray  # (T,) int64
	views: np.ndarray  # (T,) int64
	is_blind: np.ndarray  # (T,) bool
	is_takeover: np.ndarray  # (T,) bool
	supervise_action: np.ndarray  # (T,) bool
	task: list[str]
	segments_meta: list[dict[str, Any]]
	timestamps: np.ndarray | None = None
	rel_row_indices: np.ndarray | None = None
	images: dict[str, np.ndarray] | Callable[[int], dict[str, np.ndarray]] | None = None


@dataclass(frozen=True)
class _SampleSpec:
	ep_pos: int
	anchor_row: int
	state_vec: np.ndarray  # (36,) float32
	action_chunk: np.ndarray  # (H, 6) float32
	action_mask: np.ndarray  # (H,) bool
	end_step_target: int
	sample_weight: float
	task: str
	segment_kind: str
	is_synthesized_zero_frame: bool


def _unwrap_str_col(values: Sequence[Any]) -> list[str]:
	out: list[str] = []
	for v in values:
		if isinstance(v, (list, tuple, np.ndarray)):
			v = v[0] if len(v) else ""
		out.append("" if v is None else str(v))
	return out


def _normalize_in_memory_episode(ep_idx: int, raw: Mapping[str, Any]) -> _EpisodeData:
	"""Convert an in-memory episode dict (used in unit tests / mocks) into ``_EpisodeData``."""
	if "frames" in raw and isinstance(raw["frames"], Sequence) and not isinstance(raw["frames"], Mapping):
		rows = list(raw["frames"])
		if not rows:
			raise ValueError(f"episode {ep_idx} has 0 frames")
		cols: dict[str, list[Any]] = {}
		for row in rows:
			for k, v in row.items():
				cols.setdefault(k, []).append(v)
	else:
		cols = dict(raw.get("frames", raw))

	state = np.asarray(cols["observation.state"], dtype=np.float32).reshape(-1, ACTION_DIM)
	n = len(state)
	if n == 0:
		raise ValueError(f"episode {ep_idx} has 0 frames")
	action = np.asarray(cols["action"], dtype=np.float32).reshape(n, ACTION_DIM)

	if "observation.prev_action" in cols:
		prev_action = np.asarray(cols["observation.prev_action"], dtype=np.float32).reshape(n, ACTION_DIM)
	else:
		prev_action = np.vstack([INITIAL_Q.astype(np.float32)[None, :], action[:-1]])

	if "observation.gripper_cmd" in cols:
		gripper_cmd = np.asarray(cols["observation.gripper_cmd"], dtype=np.float32).reshape(n)
	else:
		gripper_cmd = prev_action[:, 5].astype(np.float32)

	seg_index = np.asarray(cols["segment.index"], dtype=np.int64).reshape(n)
	seg_kind = _unwrap_str_col(cols["segment.kind"])
	seg_done = (
		np.asarray(cols["segment.done"], dtype=bool).reshape(n) if "segment.done" in cols else np.zeros(n, dtype=bool)
	)
	chunk_valid_len = np.asarray(cols["segment.chunk_valid_len"], dtype=np.int64).reshape(n)
	if "segment.frame" in cols:
		seg_frame = np.asarray(cols["segment.frame"], dtype=np.int64).reshape(n)
	else:
		seg_frame = np.zeros(n, dtype=np.int64)
		for t in range(1, n):
			seg_frame[t] = seg_frame[t - 1] + 1 if seg_index[t] == seg_index[t - 1] else 0

	end_reason = _unwrap_str_col(cols["segment.end_reason"]) if "segment.end_reason" in cols else ["done"] * n
	bowl = _unwrap_str_col(cols["segment.bowl"]) if "segment.bowl" in cols else [""] * n
	view = (
		np.asarray(cols["segment.view"], dtype=np.int64).reshape(n) if "segment.view" in cols else np.zeros(n, np.int64)
	)
	views = (
		np.asarray(cols["segment.views"], dtype=np.int64).reshape(n)
		if "segment.views" in cols
		else np.zeros(n, np.int64)
	)
	is_blind = (
		np.asarray(cols["mask.is_blind"], dtype=bool).reshape(n) if "mask.is_blind" in cols else np.zeros(n, bool)
	)
	is_takeover = (
		np.asarray(cols["mask.is_takeover"], dtype=bool).reshape(n) if "mask.is_takeover" in cols else np.zeros(n, bool)
	)
	supervise_action = (
		np.asarray(cols["mask.supervise_action"], dtype=bool).reshape(n)
		if "mask.supervise_action" in cols
		else np.ones(n, bool)
	)
	if "task" in cols:
		task = _unwrap_str_col(cols["task"])
	else:
		task = [
			policy_inputs_from_segment_dict(
				seg_kind[t],
				{
					"bowl": bowl[t] if bowl[t] else None,
					"view": int(view[t]) if view[t] > 0 else None,
					"views": int(views[t]) if views[t] > 0 else None,
				},
			)["instruction"]
			for t in range(n)
		]

	segments_meta = list(raw.get("segments", []))
	images_payload = raw.get("images")
	if images_payload is None:
		# Check if camera arrays exist directly inside cols
		cam_cols = {k: cols[k] for k in STUDENT_CAMERAS if k in cols}
		if cam_cols:
			images_payload = cam_cols

	return _EpisodeData(
		episode_index=int(raw.get("episode_index", ep_idx)),
		length=n,
		state=state,
		prev_action=prev_action,
		gripper_cmd=gripper_cmd,
		action=action,
		segment_index=seg_index,
		segment_kind=seg_kind,
		segment_done=seg_done,
		chunk_valid_len=chunk_valid_len,
		segment_frame=seg_frame,
		end_reason=end_reason,
		bowl=bowl,
		view=view,
		views=views,
		is_blind=is_blind,
		is_takeover=is_takeover,
		supervise_action=supervise_action,
		task=task,
		segments_meta=segments_meta,
		images=images_payload,
	)


class SegmentChunkDataset(_BaseDataset):
	"""Dataset yielding Data Contract v1 samples with H-step masked chunks and H+2 end-step targets.

	Supports both LeRobot v3 disk datasets (reading ``meta/segments.json`` and decoding ONLY the 3
	``STUDENT_CAMERAS``, never ``front`` or ``top_full``) and lightweight in-memory episode dicts for tests.
	"""

	def __init__(
		self,
		root: str | Path | None = None,
		*,
		repo_id: str = "local/armigo-six-face",
		episodes: list[int] | None = None,
		episodes_data: Sequence[Mapping[str, Any]] | None = None,
		horizon: int = 50,
		takeover_weight: float = 2.0,
		augment: bool = False,
		augmentor: StudentImageAugmentor | None = None,
		video_backend: str = "pyav",
		default_image_shape: tuple[int, int] = (64, 64),
		return_torch: bool | None = None,
		drop_unsupervised: bool = False,
	) -> None:
		"""``drop_unsupervised``: leave out samples that carry no loss at all (every action step masked and no end-step
		label), e.g. the unsupervised prefix of paired-carry or takeover episodes; otherwise they fill batch slots at
		zero loss."""
		if horizon < 1:
			raise ValueError(f"horizon must be >= 1, got {horizon}")
		if takeover_weight <= 0:
			raise ValueError(f"takeover_weight must be > 0, got {takeover_weight}")
		if (root is None) == (episodes_data is None):
			raise ValueError("specify exactly one of `root` (LeRobot v3 path) or `episodes_data` (in-memory)")

		self.root = Path(root) if root is not None else None
		self.repo_id = repo_id
		self.horizon = int(horizon)
		self.takeover_weight = float(takeover_weight)
		self.augment = bool(augment)
		self.augmentor = augmentor if augmentor is not None else (StudentImageAugmentor() if augment else None)
		self.video_backend = video_backend
		self.default_image_shape = tuple(default_image_shape)
		self.return_torch = (torch is not None) if return_torch is None else bool(return_torch)
		if self.return_torch and torch is None:
			raise RuntimeError("return_torch=True requested, but PyTorch is not installed in this environment")

		self._lerobot_ds = None
		self._is_video_camera: dict[str, bool] = {}
		self._video_lookup: dict[tuple[int, str], tuple[str, float]] = {}
		self._tolerance_s: float = 1e-4

		if episodes_data is not None:
			self._episodes = [_normalize_in_memory_episode(i, ep) for i, ep in enumerate(episodes_data)]
		else:
			self._episodes = self._load_lerobot(self.root, repo_id=repo_id, episodes=episodes)

		self._samples: list[_SampleSpec] = []
		self._build_samples()
		self.dropped_unsupervised = 0
		if drop_unsupervised:
			kept = [s for s in self._samples if s.action_mask.any() or s.end_step_target >= 0]
			self.dropped_unsupervised = len(self._samples) - len(kept)
			self._samples = kept

	@classmethod
	def from_episodes(
		cls,
		episodes_data: Sequence[Mapping[str, Any]],
		*,
		horizon: int = 50,
		takeover_weight: float = 2.0,
		augment: bool = False,
		augmentor: StudentImageAugmentor | None = None,
		default_image_shape: tuple[int, int] = (64, 64),
		return_torch: bool | None = None,
		drop_unsupervised: bool = False,
	) -> SegmentChunkDataset:
		return cls(
			root=None,
			episodes_data=episodes_data,
			horizon=horizon,
			takeover_weight=takeover_weight,
			augment=augment,
			augmentor=augmentor,
			default_image_shape=default_image_shape,
			return_torch=return_torch,
			drop_unsupervised=drop_unsupervised,
		)

	def _load_lerobot(self, root: Path, *, repo_id: str, episodes: list[int] | None) -> list[_EpisodeData]:
		from lerobot.datasets.lerobot_dataset import LeRobotDataset

		# Infer repo_id from meta/info.json if default doesn't matter (LeRobotDataset reads local root directly)
		ds = LeRobotDataset(repo_id=repo_id, root=root, episodes=episodes, video_backend=self.video_backend)
		self._tolerance_s = float(ds.tolerance_s)
		all_video = True
		for cam in STUDENT_CAMERAS:
			if cam not in ds.meta.features:
				raise KeyError(
					f"required student camera {cam!r} missing from dataset features {list(ds.meta.features)}"
				)
			is_vid = cam in ds.meta.video_keys
			self._is_video_camera[cam] = is_vid
			all_video = all_video and is_vid
		if not all_video:
			self._lerobot_ds = ds

		seg_meta_by_ep: dict[int, list[dict[str, Any]]] = {}
		seg_meta_path = root / "meta" / "segments.json"
		if seg_meta_path.exists():
			seg_meta = json.loads(seg_meta_path.read_text(encoding="utf-8"))
			for row in seg_meta.get("episodes", []):
				seg_meta_by_ep[int(row["episode_index"])] = list(row.get("segments", []))

		hf = ds.hf_dataset.with_format(None)
		required_cols = (
			"episode_index",
			"frame_index",
			"timestamp",
			"task_index",
			"observation.state",
			"observation.prev_action",
			"observation.gripper_cmd",
			"action",
			"segment.index",
			"segment.kind",
			"segment.done",
			"segment.chunk_valid_len",
			"segment.frame",
			"segment.end_reason",
			"segment.bowl",
			"segment.view",
			"segment.views",
			"mask.is_blind",
			"mask.is_takeover",
			"mask.supervise_action",
		)
		missing_cols = [c for c in required_cols if c not in hf.column_names]
		if missing_cols:
			raise KeyError(f"dataset at {root} is missing Data Contract v1 column(s): {missing_cols}")

		task_map = {int(row.task_index): str(task) for task, row in ds.meta.tasks.iterrows()}
		ep_indices_all = np.asarray(hf["episode_index"], dtype=np.int64).reshape(-1)
		frame_indices_all = np.asarray(hf["frame_index"], dtype=np.int64).reshape(-1)
		timestamps_all = np.asarray(hf["timestamp"], dtype=np.float64).reshape(-1)
		task_indices_all = np.asarray(hf["task_index"], dtype=np.int64).reshape(-1)
		states_all = np.asarray(hf["observation.state"], dtype=np.float32)
		prev_actions_all = np.asarray(hf["observation.prev_action"], dtype=np.float32)
		gripper_cmds_all = np.asarray(hf["observation.gripper_cmd"], dtype=np.float32).reshape(-1)
		actions_all = np.asarray(hf["action"], dtype=np.float32)
		seg_index_all = np.asarray(hf["segment.index"], dtype=np.int64).reshape(-1)
		seg_kind_all = _unwrap_str_col(hf["segment.kind"])
		seg_done_all = np.asarray(hf["segment.done"], dtype=bool).reshape(-1)
		cvl_all = np.asarray(hf["segment.chunk_valid_len"], dtype=np.int64).reshape(-1)
		seg_frame_all = np.asarray(hf["segment.frame"], dtype=np.int64).reshape(-1)
		end_reason_all = _unwrap_str_col(hf["segment.end_reason"])
		bowl_all = _unwrap_str_col(hf["segment.bowl"])
		view_all = np.asarray(hf["segment.view"], dtype=np.int64).reshape(-1)
		views_all = np.asarray(hf["segment.views"], dtype=np.int64).reshape(-1)
		is_blind_all = np.asarray(hf["mask.is_blind"], dtype=bool).reshape(-1)
		is_takeover_all = np.asarray(hf["mask.is_takeover"], dtype=bool).reshape(-1)
		supervise_all = np.asarray(hf["mask.supervise_action"], dtype=bool).reshape(-1)

		# Group row positions by episode_index in order of frame_index
		by_ep: dict[int, list[int]] = {}
		for row_pos, ep_i in enumerate(ep_indices_all.tolist()):
			by_ep.setdefault(ep_i, []).append(row_pos)

		out: list[_EpisodeData] = []
		for ep_i in sorted(by_ep):
			rows = by_ep[ep_i]
			rows.sort(key=lambda r: int(frame_indices_all[r]))
			idx_arr = np.asarray(rows, dtype=np.int64)
			expected_frames = np.arange(len(rows), dtype=np.int64)
			if not np.array_equal(frame_indices_all[idx_arr], expected_frames):
				raise ValueError(f"episode {ep_i} has non-contiguous frame_index values")
			ep_meta = ds.meta.episodes[ep_i]
			for cam in STUDENT_CAMERAS:
				if self._is_video_camera[cam]:
					from_ts = float(ep_meta[f"videos/{cam}/from_timestamp"])
					video_path = str(ds.root / ds.meta.get_video_file_path(ep_i, cam))
					self._video_lookup[(ep_i, cam)] = (video_path, from_ts)
			out.append(
				_EpisodeData(
					episode_index=ep_i,
					length=len(rows),
					state=states_all[idx_arr],
					prev_action=prev_actions_all[idx_arr],
					gripper_cmd=gripper_cmds_all[idx_arr],
					action=actions_all[idx_arr],
					segment_index=seg_index_all[idx_arr],
					segment_kind=[seg_kind_all[r] for r in rows],
					segment_done=seg_done_all[idx_arr],
					chunk_valid_len=cvl_all[idx_arr],
					segment_frame=seg_frame_all[idx_arr],
					end_reason=[end_reason_all[r] for r in rows],
					bowl=[bowl_all[r] for r in rows],
					view=view_all[idx_arr],
					views=views_all[idx_arr],
					is_blind=is_blind_all[idx_arr],
					is_takeover=is_takeover_all[idx_arr],
					supervise_action=supervise_all[idx_arr],
					task=[task_map[int(task_indices_all[r])] for r in rows],
					segments_meta=seg_meta_by_ep.get(ep_i, []),
					timestamps=timestamps_all[idx_arr],
					rel_row_indices=idx_arr,
				)
			)
		return out

	def _build_samples(self) -> None:
		h = self.horizon
		for ep_pos, ep in enumerate(self._episodes):
			n = ep.length
			roll_rows, task_rows = _segment_roll_rows(ep)
			# q_{t-1}: frame 0 repeats q_0; frame t > 0 uses previous frame in the SAME episode.
			q_prev = np.empty_like(ep.state)
			q_prev[0] = ep.state[0]
			if n > 1:
				q_prev[1:] = ep.state[:-1]

			for t in range(n):
				cvl = int(ep.chunk_valid_len[t])
				if cvl < 1:
					raise ValueError(f"episode {ep.episode_index} frame {t}: chunk_valid_len must be >= 1, got {cvl}")
				seg_rem = min(cvl, n - t)

				# Stop before crossing a mask.is_takeover onset (False -> True transition strictly after t).
				takeover_cutoff = seg_rem
				for d in range(1, seg_rem):
					if bool(ep.is_takeover[t + d]) and not bool(ep.is_takeover[t + d - 1]):
						takeover_cutoff = d
						break

				# Stop before the first FUTURE frame whose action may not be supervised (blind or
				# supervise_action=False). Checking only the anchor would let a chunk that starts before a
				# perturbation put blind-zone actions into the loss of an earlier prediction.
				unsup_cutoff = seg_rem
				for d in range(1, seg_rem):
					if bool(ep.is_blind[t + d]) or not bool(ep.supervise_action[t + d]):
						unsup_cutoff = d
						break

				valid_action_horizon = min(seg_rem, takeover_cutoff, unsup_cutoff)
				supervised_here = bool(ep.supervise_action[t]) and (not bool(ep.is_blind[t]))
				num_unmasked = min(h, valid_action_horizon) if supervised_here else 0

				action_mask = np.zeros(h, dtype=bool)
				if num_unmasked > 0:
					action_mask[:num_unmasked] = True

				copy_len = min(h, valid_action_horizon)
				action_chunk = np.empty((h, ACTION_DIM), dtype=np.float32)
				action_chunk[:copy_len] = ep.action[t : t + copy_len]
				if copy_len < h:
					# Masked chunk steps MUST be filled with the segment's LAST VALID ACTION (never next segment!).
					action_chunk[copy_len:] = action_chunk[copy_len - 1]

				end_target = end_step_target_from_segment(
					cvl,
					h,
					ep.end_reason[t],
					is_blind=bool(ep.is_blind[t]),
					supervise_action=bool(ep.supervise_action[t]),
					is_zero_frame=False,
				)
				# If an unsupervisable frame lies inside this chunk window before the segment end, the segment's
				# end timing reflects what happened in that zone (e.g. a perturbation the anchor cannot see):
				# conservatively drop the end-step label rather than teach it.
				if unsup_cutoff < min(h, seg_rem):
					end_target = IGNORE_INDEX
				state_vec = encode_state(
					q_t=ep.state[t],
					q_prev=q_prev[t],
					prev_action=ep.prev_action[t],
					gripper_cmd=ep.gripper_cmd[t],
					segment_kind=ep.segment_kind[t],
					bowl=ep.bowl[t],
					view=int(ep.view[t]),
					views=int(ep.views[t]),
					segment_frame=int(ep.segment_frame[t]),
					roll_deg=roll_rows[t],
				)
				weight = self.takeover_weight if bool(ep.is_takeover[t]) else 1.0
				self._samples.append(
					_SampleSpec(
						ep_pos=ep_pos,
						anchor_row=t,
						state_vec=state_vec,
						action_chunk=action_chunk,
						action_mask=action_mask,
						end_step_target=end_target,
						sample_weight=float(weight),
						task=task_rows[t],
						segment_kind=ep.segment_kind[t],
						is_synthesized_zero_frame=False,
					)
				)

			# Synthesize zero-frame segment samples (start == end and reason == "done", e.g., let_go).
			for seg in ep.segments_meta:
				if int(seg["start"]) == int(seg["end"]) and str(seg.get("reason", "")) == "done":
					anchor_row = min(int(seg["start"]), n - 1)
					pi = policy_inputs_from_segment_dict(str(seg["kind"]), seg.get("params"))
					is_blind = bool(ep.is_blind[anchor_row])
					is_takeover = bool(ep.is_takeover[anchor_row])
					supervise_action = bool(ep.supervise_action[anchor_row])
					state_vec = encode_state_from_task(
						q_t=ep.state[anchor_row],
						q_prev=q_prev[anchor_row],
						prev_action=ep.prev_action[anchor_row],
						gripper_cmd=ep.gripper_cmd[anchor_row],
						task=pi,
						segment_frame=0,
					)
					action_mask = np.zeros(h, dtype=bool)
					action_chunk = np.repeat(ep.prev_action[anchor_row][None, :], h, axis=0).astype(np.float32)
					end_target = end_step_target_from_segment(
						0,
						h,
						"done",
						is_blind=is_blind,
						supervise_action=supervise_action,
						is_zero_frame=True,
					)
					weight = self.takeover_weight if is_takeover else 1.0
					self._samples.append(
						_SampleSpec(
							ep_pos=ep_pos,
							anchor_row=anchor_row,
							state_vec=state_vec,
							action_chunk=action_chunk,
							action_mask=action_mask,
							end_step_target=end_target,
							sample_weight=float(weight),
							task=str(pi["instruction"]),
							segment_kind=str(pi["kind"]),
							is_synthesized_zero_frame=True,
						)
					)

	def __len__(self) -> int:
		return len(self._samples)

	def _load_student_images(self, ep: _EpisodeData, row: int) -> dict[str, Any]:
		if self._video_lookup or self._lerobot_ds is not None:
			from lerobot.datasets.video_utils import decode_video_frames

			ts = float(ep.timestamps[row])
			rel_row = int(ep.rel_row_indices[row])
			images: dict[str, Any] = {}
			raw_row = None
			for cam in STUDENT_CAMERAS:
				if self._is_video_camera.get(cam, False):
					video_path, from_ts = self._video_lookup[(ep.episode_index, cam)]
					if self.video_backend == "pyav":
						images[cam] = _decode_frame_pyav_cached(video_path, from_ts + ts, self._tolerance_s)
					else:
						frame = decode_video_frames(video_path, [from_ts + ts], self._tolerance_s, self.video_backend)
						images[cam] = frame.squeeze(0).to(dtype=torch.float32)
				else:
					if raw_row is None:
						raw_row = self._lerobot_ds.hf_dataset[rel_row]
					images[cam] = torch.from_numpy(_to_float_chw(raw_row[cam]))
			return images

		# In-memory / mock episode backend
		images = {}
		if callable(ep.images):
			raw_imgs = ep.images(row)
			for cam in STUDENT_CAMERAS:
				arr = _to_float_chw(raw_imgs[cam])
				images[cam] = torch.from_numpy(arr) if self.return_torch else arr
		elif isinstance(ep.images, Mapping):
			for cam in STUDENT_CAMERAS:
				if cam in ep.images:
					arr = _to_float_chw(ep.images[cam][row])
				else:
					arr = np.zeros((3, *self.default_image_shape), dtype=np.float32)
				images[cam] = torch.from_numpy(arr) if self.return_torch else arr
		else:
			for cam in STUDENT_CAMERAS:
				arr = np.zeros((3, *self.default_image_shape), dtype=np.float32)
				images[cam] = torch.from_numpy(arr) if self.return_torch else arr
		return images

	def __getitem__(self, idx: int) -> dict[str, Any]:
		spec = self._samples[idx]
		ep = self._episodes[spec.ep_pos]
		images = self._load_student_images(ep, spec.anchor_row)
		if self.augment and self.augmentor is not None:
			images = self.augmentor(images)

		if self.return_torch:
			sample: dict[str, Any] = {
				**images,
				"observation.state": torch.from_numpy(spec.state_vec.copy()),
				"action": torch.from_numpy(spec.action_chunk.copy()),
				"action_mask": torch.from_numpy(spec.action_mask.copy()),
				"end_step_target": torch.tensor(spec.end_step_target, dtype=torch.int64),
				"sample_weight": torch.tensor(spec.sample_weight, dtype=torch.float32),
				"task": spec.task,
				"segment_kind": spec.segment_kind,
				"is_synthesized_zero_frame": spec.is_synthesized_zero_frame,
			}
			return sample

		return {
			**images,
			"observation.state": spec.state_vec.copy(),
			"action": spec.action_chunk.copy(),
			"action_mask": spec.action_mask.copy(),
			"end_step_target": np.int64(spec.end_step_target),
			"sample_weight": np.float32(spec.sample_weight),
			"task": spec.task,
			"segment_kind": spec.segment_kind,
			"is_synthesized_zero_frame": spec.is_synthesized_zero_frame,
		}

	def fit_normalizer(self, *, eps: float = 1e-6) -> ActionNormalizer:
		"""Compute per-joint action and 36-dim state mean/std ONLY over this dataset (training split)."""
		action_chunks: list[np.ndarray] = []
		for ep in self._episodes:
			valid = ep.supervise_action & (~ep.is_blind)
			if valid.any():
				action_chunks.append(ep.action[valid])
		if not action_chunks:
			action_chunks = [ep.action for ep in self._episodes]
		actions = np.concatenate(action_chunks, axis=0)
		states = np.stack([s.state_vec for s in self._samples], axis=0)
		return ActionNormalizer.fit(actions, states=states, eps=eps)

	def end_step_class_weights(self, *, max_weight: float = 15.0) -> np.ndarray:
		"""Inverse-frequency bounded weights over valid (non-ignored) ``end_step_target`` values."""
		num_classes = self.horizon + 2
		counts = np.zeros(num_classes, dtype=np.float64)
		for s in self._samples:
			if 0 <= s.end_step_target < num_classes:
				counts[s.end_step_target] += 1.0
		total = float(counts.sum())
		if total <= 0.0:
			return np.ones(num_classes, dtype=np.float32)
		max_c = float(counts.max())
		weights = np.ones(num_classes, dtype=np.float64)
		present = counts > 0
		weights[present] = np.clip(max_c / counts[present], 1.0, float(max_weight))
		mean_sample_weight = float((weights * counts).sum() / total)
		if mean_sample_weight > 0.0:
			weights = weights / mean_sample_weight
		return weights.astype(np.float32)

	def summary(self) -> dict[str, Any]:
		"""Dataset statistics without loading video frames."""
		kind_counts = {k: 0 for k in SEGMENT_KINDS_ORDER}
		takeover_count = 0
		synth_count = 0
		valid_end_count = 0
		for s in self._samples:
			kind_counts[s.segment_kind] = kind_counts.get(s.segment_kind, 0) + 1
			if s.sample_weight > 1.0:
				takeover_count += 1
			if s.is_synthesized_zero_frame:
				synth_count += 1
			if s.end_step_target >= 0:
				valid_end_count += 1
		return {
			"episodes": len(self._episodes),
			"samples": len(self._samples),
			"synthesized_zero_frame_samples": synth_count,
			"takeover_samples": takeover_count,
			"supervised_end_step_samples": valid_end_count,
			"dropped_unsupervised_samples": int(getattr(self, "dropped_unsupervised", 0)),
			"segment_kinds": kind_counts,
			"bowls": list(BOWL_VOCAB),
			"state_dim": STATE_DIM,
			"horizon": self.horizon,
		}


def vla_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
	"""Collate ``SegmentChunkDataset`` items into a batch dict of tensors + metadata lists."""
	if not batch:
		raise ValueError("cannot collate empty batch")
	out: dict[str, Any] = {}
	tensor_keys = (
		*STUDENT_CAMERAS,
		"observation.state",
		"action",
		"action_mask",
		"end_step_target",
		"sample_weight",
	)
	for key in tensor_keys:
		vals = [item[key] for item in batch]
		if torch is not None and isinstance(vals[0], torch.Tensor):
			out[key] = torch.stack(vals, dim=0)
		else:
			out[key] = np.stack(vals, axis=0)
	for key in ("task", "segment_kind", "is_synthesized_zero_frame"):
		out[key] = [item[key] for item in batch]
	return out
