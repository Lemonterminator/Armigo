"""Masked Flow-Matching SmolVLA and ACT student wrappers with H+2 End-Step Head (plan S9)."""

from __future__ import annotations

import contextlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, pad_vector
from lerobot.utils.constants import (
	ACTION,
	IMAGENET_STATS,
	OBS_ENV_STATE,
	OBS_IMAGES,
	OBS_LANGUAGE_ATTENTION_MASK,
	OBS_LANGUAGE_TOKENS,
	OBS_STATE,
)
from torch import Tensor, nn

from .contract import (
	ACTION_DIM,
	BOWL_VOCAB,
	CONTRACT_VERSION,
	DEFAULT_HORIZON,
	IGNORE_INDEX,
	SEGMENT_KINDS_ORDER,
	STATE_DIM,
	STUDENT_CAMERAS,
	ActionNormalizer,
)

DEFAULT_CAMERA_SHAPES: dict[str, tuple[int, int, int]] = {
	"observation.images.top_rgb_v2": (3, 416, 512),
	"observation.images.top_depth_v2": (3, 416, 512),
	"observation.images.wrist": (3, 480, 640),
}
# Target-bowl one-hot inside the 38-dim student state (contract.encode_state [31:34]). With ``bowl_token`` the ACT
# wrapper moves it to its own encoder token (LeRobot's environment-state input) and zeroes it in the state token
# (docs/vla/BOWL_CHANNEL_PAIRED_PLAN_2026-11-23.md).
BOWL_SLICE = slice(31, 34)
POS_EMBED_KEY = "policy.model.encoder_1d_feature_pos_embed.weight"
DEFAULT_SMOLVLM_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
DEFAULT_SMOLVLA_BASE_ID = "lerobot/smolvla_base"


class EndStepHead(nn.Module):
	"""Small 2-layer MLP predicting H+2 categorical end-step logits from pooled prefix features."""

	def __init__(self, in_dim: int, num_classes: int, hidden_dim: int = 256, dropout: float = 0.1) -> None:
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(in_dim, hidden_dim),
			nn.SiLU(),
			nn.Dropout(dropout),
			nn.Linear(hidden_dim, num_classes),
		)

	def forward(self, x: Tensor) -> Tensor:
		return self.net(x)


def compute_end_step_loss(
	logits: Tensor,
	targets: Tensor,
	sample_weight: Tensor | None = None,
	class_weights: Tensor | None = None,
	*,
	ignore_index: int = IGNORE_INDEX,
) -> Tensor:
	"""Masked and optionally class-weighted / sample-weighted cross-entropy for H+2 end-step targets."""
	logits_f = logits.float()
	valid = targets != ignore_index
	if not valid.any():
		return logits_f.sum() * 0.0

	cw = class_weights.to(device=logits_f.device, dtype=torch.float32) if class_weights is not None else None
	per_sample = F.cross_entropy(logits_f[valid], targets[valid].long(), weight=cw, reduction="none")
	if sample_weight is None:
		return per_sample.mean()
	w = sample_weight[valid].to(device=logits_f.device, dtype=torch.float32).clamp_min(0.0)
	denom = w.sum()
	if float(denom.item()) <= 0.0:
		return logits_f.sum() * 0.0
	return (per_sample * w).sum() / denom


class _BaseVLAWrapper(nn.Module):
	"""Shared normalizer buffers, end-step loss weighting, and checkpoint save/load helpers."""

	family: str = ""

	def __init__(
		self,
		*,
		horizon: int,
		state_dim: int = STATE_DIM,
		action_dim: int = ACTION_DIM,
		lambda_done: float = 0.1,
		normalizer: ActionNormalizer | None = None,
		class_weights: Tensor | np.ndarray | None = None,
	) -> None:
		super().__init__()
		self.horizon = int(horizon)
		self.state_dim = int(state_dim)
		self.action_dim = int(action_dim)
		self.num_end_classes = self.horizon + 2
		self.lambda_done = float(lambda_done)

		norm = normalizer or ActionNormalizer.identity()
		self.register_buffer("action_mean", torch.as_tensor(norm.action_mean, dtype=torch.float32))
		self.register_buffer("action_std", torch.as_tensor(norm.action_std, dtype=torch.float32))
		s_mean = np.zeros(self.state_dim, dtype=np.float32) if norm.state_mean is None else norm.state_mean
		s_std = np.ones(self.state_dim, dtype=np.float32) if norm.state_std is None else norm.state_std
		self.register_buffer("state_mean", torch.as_tensor(s_mean, dtype=torch.float32))
		self.register_buffer("state_std", torch.as_tensor(s_std, dtype=torch.float32))

		if class_weights is not None:
			cw = torch.as_tensor(class_weights, dtype=torch.float32).reshape(self.num_end_classes)
		else:
			cw = torch.ones(self.num_end_classes, dtype=torch.float32)
		self.register_buffer("end_class_weights", cw)

	def set_normalizer(self, normalizer: ActionNormalizer) -> None:
		device = self.action_mean.device
		self.action_mean.copy_(torch.as_tensor(normalizer.action_mean, dtype=torch.float32, device=device))
		self.action_std.copy_(torch.as_tensor(normalizer.action_std, dtype=torch.float32, device=device))
		if normalizer.state_mean is not None and normalizer.state_std is not None:
			self.state_mean.copy_(torch.as_tensor(normalizer.state_mean, dtype=torch.float32, device=device))
			self.state_std.copy_(torch.as_tensor(normalizer.state_std, dtype=torch.float32, device=device))

	def export_normalizer(self) -> ActionNormalizer:
		return ActionNormalizer(
			action_mean=self.action_mean.detach().cpu().numpy(),
			action_std=self.action_std.detach().cpu().numpy(),
			state_mean=self.state_mean.detach().cpu().numpy(),
			state_std=self.state_std.detach().cpu().numpy(),
		)

	def set_class_weights(self, class_weights: Tensor | np.ndarray) -> None:
		cw = torch.as_tensor(class_weights, dtype=torch.float32, device=self.end_class_weights.device)
		if cw.shape != (self.num_end_classes,):
			raise ValueError(f"expected class_weights shape ({self.num_end_classes},), got {tuple(cw.shape)}")
		self.end_class_weights.copy_(cw)

	def _normalize_state(self, state: Tensor) -> Tensor:
		mean = self.state_mean.to(device=state.device, dtype=state.dtype)
		std = self.state_std.to(device=state.device, dtype=state.dtype)
		return (state - mean) / std

	def _normalize_action(self, action: Tensor) -> Tensor:
		mean = self.action_mean.to(device=action.device, dtype=action.dtype)
		std = self.action_std.to(device=action.device, dtype=action.dtype)
		return (action - mean) / std

	def _unnormalize_action(self, action_norm: Tensor) -> Tensor:
		mean = self.action_mean.to(device=action_norm.device, dtype=torch.float32)
		std = self.action_std.to(device=action_norm.device, dtype=torch.float32)
		return action_norm.float() * std + mean

	def _checkpoint_meta(self) -> dict[str, Any]:
		raise NotImplementedError

	def get_optim_param_groups(
		self, lr: float, *, weight_decay: float = 1e-4, end_head_lr: float | None = None
	) -> list[dict[str, Any]]:
		head = [p for n, p in self.named_parameters() if n.startswith("end_head.") and p.requires_grad]
		rest = [p for n, p in self.named_parameters() if not n.startswith("end_head.") and p.requires_grad]
		groups = [{"params": rest, "lr": float(lr), "weight_decay": float(weight_decay)}]
		if head:
			head_lr = float(end_head_lr) if end_head_lr is not None else float(lr)
			groups.append({"params": head, "lr": head_lr, "weight_decay": float(weight_decay)})
		return groups

	def save_checkpoint(self, save_dir: str | Path, extra_meta: Mapping[str, Any] | None = None) -> Path:
		out = Path(save_dir)
		out.mkdir(parents=True, exist_ok=True)
		from safetensors.torch import save_file

		sd = {k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()}
		save_file(sd, str(out / "model.safetensors"))
		self.export_normalizer().save_alongside_checkpoint(out)

		meta = {
			"schema": CONTRACT_VERSION,
			"wrapper": type(self).__name__,
			"family": self.family,
			"horizon": self.horizon,
			"state_dim": self.state_dim,
			"action_dim": self.action_dim,
			"num_end_classes": self.num_end_classes,
			"lambda_done": self.lambda_done,
			"student_cameras": list(STUDENT_CAMERAS),
			"segment_kinds_order": list(SEGMENT_KINDS_ORDER),
			"bowl_vocab": list(BOWL_VOCAB),
			"model_config": self._checkpoint_meta(),
		}
		if extra_meta:
			meta["extra"] = dict(extra_meta)
		(out / "vla_config.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
		return out


class _TinyTokenizer:
	"""Minimal fast tokenizer for in-memory SmolVLA unit tests without HuggingFace network/cache calls."""

	fake_image_token_id = 10
	global_image_token_id = 11
	pad_token_id = 2
	padding_side = "right"

	def __call__(
		self,
		texts: list[str],
		*,
		padding: str = "max_length",
		max_length: int = 16,
		truncation: bool = True,
		return_tensors: str = "pt",
	) -> dict[str, Tensor]:
		del padding, truncation, return_tensors
		ids = torch.full((len(texts), max_length), self.pad_token_id, dtype=torch.long)
		mask = torch.zeros((len(texts), max_length), dtype=torch.long)
		for row, txt in enumerate(texts):
			tokens = [20 + (ord(ch) % 200) for ch in txt[:max_length]]
			if not tokens:
				tokens = [20]
			ids[row, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
			mask[row, : len(tokens)] = 1
		return {"input_ids": ids, "attention_mask": mask}


class _TinyProcessor:
	def __init__(self) -> None:
		self.tokenizer = _TinyTokenizer()


@contextlib.contextmanager
def _tiny_smolvlm_context():
	"""Temporarily supply an in-memory tiny SmolVLMConfig/Processor to SmolVLMWithExpertModel for fast tests."""
	import lerobot.policies.smolvla.smolvlm_with_expert as swe
	from transformers import SmolVLMConfig, SmolVLMForConditionalGeneration

	tiny_cfg = SmolVLMConfig(
		use_cache=False,
		scale_factor=2,
		image_token_id=12,
		vision_config={
			"hidden_size": 64,
			"intermediate_size": 128,
			"num_hidden_layers": 1,
			"num_attention_heads": 4,
			"image_size": 64,
			"patch_size": 16,
			"num_channels": 3,
		},
		text_config={
			"vocab_size": 512,
			"hidden_size": 64,
			"intermediate_size": 256,
			"num_hidden_layers": 2,
			"num_attention_heads": 4,
			"num_key_value_heads": 2,
			"head_dim": 16,
			"max_position_embeddings": 512,
			"pad_token_id": 2,
			"tie_word_embeddings": False,
		},
	)

	class _FakeAutoConfig:
		@staticmethod
		def from_pretrained(_model_id: str, **_kwargs):
			return tiny_cfg

	class _FakeAutoProcessor:
		@staticmethod
		def from_pretrained(_model_id: str, **_kwargs):
			return _TinyProcessor()

	orig_auto_cfg = swe.AutoConfig
	orig_auto_proc = swe.AutoProcessor
	orig_smol_cls = swe.SmolVLMForConditionalGeneration
	swe.AutoConfig = _FakeAutoConfig
	swe.AutoProcessor = _FakeAutoProcessor
	swe.SmolVLMForConditionalGeneration = SmolVLMForConditionalGeneration
	try:
		yield
	finally:
		swe.AutoConfig = orig_auto_cfg
		swe.AutoProcessor = orig_auto_proc
		swe.SmolVLMForConditionalGeneration = orig_smol_cls


def _adapt_smolvla_base_state_dict(base_sd: dict[str, Tensor], target_model: SmolVLAPolicy) -> dict[str, Tensor]:
	"""Load ``lerobot/smolvla_base`` weights into a 36-dim-state ``SmolVLAPolicy`` cleanly."""
	target_sd = target_model.state_dict()
	adapted: dict[str, Tensor] = {}
	for key, val in base_sd.items():
		if key not in target_sd:
			continue
		tgt = target_sd[key]
		if val.shape == tgt.shape:
			adapted[key] = val.to(dtype=tgt.dtype)
		elif key == "model.state_proj.weight" and val.shape[0] == tgt.shape[0] and val.shape[1] < tgt.shape[1]:
			new_w = tgt.detach().clone()
			new_w[:, : val.shape[1]] = val.to(dtype=tgt.dtype)
			nn.init.normal_(new_w[:, val.shape[1] :], mean=0.0, std=0.02)
			adapted[key] = new_w
		else:
			raise ValueError(
				f"incompatible tensor shape for {key}: checkpoint {tuple(val.shape)} vs {tuple(tgt.shape)}"
			)
	return adapted


class SmolVLAWithEndHead(_BaseVLAWrapper):
	"""LeRobot 0.6 ``SmolVLAPolicy`` wrapped with masked flow matching and H+2 End-Step Head."""

	family = "smolvla"

	def __init__(
		self,
		*,
		horizon: int = DEFAULT_HORIZON["smolvla"],
		vlm_model_name: str = DEFAULT_SMOLVLM_ID,
		init_from: str | None = DEFAULT_SMOLVLA_BASE_ID,
		freeze_vision_encoder: bool = True,
		train_expert_only: bool = True,
		train_state_proj: bool = True,
		resize_imgs_with_padding: tuple[int, int] = (512, 512),
		num_steps: int = 10,
		lambda_done: float = 0.1,
		normalizer: ActionNormalizer | None = None,
		class_weights: Tensor | np.ndarray | None = None,
		camera_shapes: Mapping[str, tuple[int, int, int]] | None = None,
		end_head_hidden_dim: int = 256,
		end_head_dropout: float = 0.1,
		tiny_for_test: bool = False,
		local_files_only: bool = False,
	) -> None:
		super().__init__(
			horizon=horizon,
			state_dim=STATE_DIM,
			action_dim=ACTION_DIM,
			lambda_done=lambda_done,
			normalizer=normalizer,
			class_weights=class_weights,
		)
		self.vlm_model_name = vlm_model_name
		self.init_from = init_from
		self.freeze_vision_encoder = bool(freeze_vision_encoder)
		self.train_expert_only = bool(train_expert_only)
		self.train_state_proj = bool(train_state_proj)
		self.resize_imgs_with_padding = tuple(resize_imgs_with_padding)
		self.num_steps = int(num_steps)
		self.camera_shapes = {k: tuple(v) for k, v in (camera_shapes or DEFAULT_CAMERA_SHAPES).items()}
		self.end_head_hidden_dim = int(end_head_hidden_dim)
		self.end_head_dropout = float(end_head_dropout)
		self.tiny_for_test = bool(tiny_for_test)

		input_features = {
			cam: PolicyFeature(type=FeatureType.VISUAL, shape=self.camera_shapes[cam]) for cam in STUDENT_CAMERAS
		}
		input_features[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,))
		output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))}

		pending_end_head_sd: dict[str, Any] | None = None
		if self.tiny_for_test:
			cfg = SmolVLAConfig(
				chunk_size=self.horizon,
				n_action_steps=self.horizon,
				max_state_dim=max(STATE_DIM, 32),
				max_action_dim=16,
				resize_imgs_with_padding=(64, 64),
				tokenizer_max_length=16,
				num_steps=self.num_steps,
				freeze_vision_encoder=self.freeze_vision_encoder,
				train_expert_only=self.train_expert_only,
				train_state_proj=self.train_state_proj,
				vlm_model_name="tiny-smolvlm-test",
				load_vlm_weights=False,
				pad_language_to="max_length",
				num_expert_layers=0,
				num_vlm_layers=2,
				self_attn_every_n_layers=2,
				expert_width_multiplier=0.75,
				device="cpu",
			)
			cfg.input_features = input_features
			cfg.output_features = output_features
			with _tiny_smolvlm_context():
				self.policy = SmolVLAPolicy(cfg)
			# Tiny models can only continue from a local Armigo checkpoint (continue-training tests).
			if init_from not in (None, "none", "scratch", "vlm") and Path(str(init_from)).exists():
				pending_end_head_sd = self._load_pretrained_backbone(str(init_from), local_files_only=True)
		else:
			init_str = None if init_from in (None, "none", "scratch", "vlm") else str(init_from)
			use_vlm_init = init_from == "vlm"
			cfg = SmolVLAConfig(
				chunk_size=self.horizon,
				n_action_steps=self.horizon,
				max_state_dim=max(STATE_DIM, 32),
				max_action_dim=32,
				resize_imgs_with_padding=self.resize_imgs_with_padding,
				tokenizer_max_length=48,
				num_steps=self.num_steps,
				freeze_vision_encoder=self.freeze_vision_encoder,
				train_expert_only=self.train_expert_only,
				train_state_proj=self.train_state_proj,
				vlm_model_name=self.vlm_model_name,
				load_vlm_weights=use_vlm_init,
				pad_language_to="max_length",
				num_expert_layers=0,
				num_vlm_layers=16,
				self_attn_every_n_layers=2,
				expert_width_multiplier=0.75,
				device="cpu",
			)
			cfg.input_features = input_features
			cfg.output_features = output_features
			self.policy = SmolVLAPolicy(cfg)
			if init_str is not None:
				pending_end_head_sd = self._load_pretrained_backbone(init_str, local_files_only=local_files_only)

		vlm_hidden = int(self.policy.model.vlm_with_expert.config.text_config.hidden_size)
		self.end_head = EndStepHead(
			in_dim=vlm_hidden * 2,
			num_classes=self.num_end_classes,
			hidden_dim=self.end_head_hidden_dim,
			dropout=self.end_head_dropout,
		)
		# Continuing from an Armigo checkpoint (e.g. S11 DAgger fine-tuning) must keep the trained End-Step Head;
		# only a fresh upstream backbone gets a newly initialised head. Normalizer / class-weight buffers are NOT
		# restored here: the trainer passes statistics fitted on the current training split (same as ACT).
		if pending_end_head_sd:
			self.end_head.load_state_dict(pending_end_head_sd, strict=True)
		self._token_cache: dict[str, tuple[Tensor, Tensor]] = {}
		self._captured_prefix_pad_masks: Tensor | None = None
		self._captured_prefix_out: Tensor | None = None
		self._register_prefix_hooks()

	def _load_pretrained_backbone(self, init_from: str, *, local_files_only: bool = False) -> dict[str, Any] | None:
		"""Load backbone weights; return the ``end_head.*`` sub-state of an Armigo checkpoint (None otherwise)."""
		from huggingface_hub import hf_hub_download
		from safetensors.torch import load_file

		p = Path(init_from)
		if p.is_dir():
			sf_path = p / "model.safetensors"
		elif p.is_file():
			sf_path = p
		else:
			sf_path = Path(hf_hub_download(init_from, "model.safetensors", local_files_only=local_files_only))
		sd = load_file(str(sf_path), device="cpu")
		# Check if this is an Armigo wrapper checkpoint (`policy.*`) vs LeRobot SmolVLAPolicy (`model.*`)
		if any(k.startswith("policy.") for k in sd):
			sub_sd = {k.removeprefix("policy."): v for k, v in sd.items() if k.startswith("policy.")}
			self.policy.load_state_dict(sub_sd, strict=True)
			end_sd = {k.removeprefix("end_head."): v for k, v in sd.items() if k.startswith("end_head.")}
			if not end_sd:
				raise KeyError(f"Armigo checkpoint {sf_path} has policy.* weights but no end_head.* weights")
			return end_sd
		adapted = _adapt_smolvla_base_state_dict(sd, self.policy)
		self.policy.load_state_dict(adapted, strict=True)
		return None

	def _register_prefix_hooks(self) -> None:
		orig_embed_prefix = self.policy.model.embed_prefix
		orig_vlm_forward = self.policy.model.vlm_with_expert.forward

		def _wrapped_embed_prefix(*args, **kwargs):
			out = orig_embed_prefix(*args, **kwargs)
			self._captured_prefix_pad_masks = out[1]
			return out

		def _wrapped_vlm_forward(*args, **kwargs):
			out = orig_vlm_forward(*args, **kwargs)
			outputs_embeds = out[0]
			if outputs_embeds is not None and len(outputs_embeds) > 0 and outputs_embeds[0] is not None:
				self._captured_prefix_out = outputs_embeds[0]
			return out

		self.policy.model.embed_prefix = _wrapped_embed_prefix
		self.policy.model.vlm_with_expert.forward = _wrapped_vlm_forward

	def _pool_prefix_features(self) -> Tensor:
		if self._captured_prefix_out is None or self._captured_prefix_pad_masks is None:
			raise RuntimeError("SmolVLA prefix hidden states were not captured during forward pass")
		prefix_out = self._captured_prefix_out
		pad_masks = self._captured_prefix_pad_masks
		self._captured_prefix_out = None
		self._captured_prefix_pad_masks = None

		mask = pad_masks.to(device=prefix_out.device, dtype=torch.float32).unsqueeze(-1)
		prefix_f = prefix_out.float()
		mean_pooled = (prefix_f * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
		positions = torch.arange(pad_masks.shape[1], device=pad_masks.device).unsqueeze(0)
		last_valid_idx = (positions * pad_masks.long()).argmax(dim=1)
		batch_idx = torch.arange(prefix_f.shape[0], device=prefix_f.device)
		state_tok = prefix_f[batch_idx, last_valid_idx]
		return torch.cat([mean_pooled, state_tok], dim=-1)

	def tokenize_tasks(self, tasks: list[str], device: torch.device) -> tuple[Tensor, Tensor]:
		tokenizer = self.policy.model.vlm_with_expert.processor.tokenizer
		max_len = int(self.policy.config.tokenizer_max_length)
		toks_list: list[Tensor] = []
		masks_list: list[Tensor] = []
		uncached = [t for t in tasks if t not in self._token_cache]
		if uncached:
			formatted = [t if t.endswith("\n") else f"{t}\n" for t in uncached]
			enc = tokenizer(
				formatted,
				padding="max_length",
				max_length=max_len,
				truncation=True,
				return_tensors="pt",
			)
			for i, raw_t in enumerate(uncached):
				self._token_cache[raw_t] = (
					enc["input_ids"][i].clone(),
					enc["attention_mask"][i].bool().clone(),
				)
		for t in tasks:
			ids, m = self._token_cache[t]
			toks_list.append(ids)
			masks_list.append(m)
		return torch.stack(toks_list, dim=0).to(device), torch.stack(masks_list, dim=0).to(device)

	def _prepare_model_batch(self, batch: Mapping[str, Any], *, include_action: bool) -> dict[str, Tensor]:
		device = self.action_mean.device
		state = batch[OBS_STATE].to(device=device, dtype=torch.float32)
		model_batch: dict[str, Tensor] = {
			OBS_STATE: self._normalize_state(state),
		}
		for cam in STUDENT_CAMERAS:
			model_batch[cam] = batch[cam].to(device=device, dtype=torch.float32)

		if OBS_LANGUAGE_TOKENS in batch and OBS_LANGUAGE_ATTENTION_MASK in batch:
			model_batch[OBS_LANGUAGE_TOKENS] = batch[OBS_LANGUAGE_TOKENS].to(device=device, dtype=torch.long)
			model_batch[OBS_LANGUAGE_ATTENTION_MASK] = batch[OBS_LANGUAGE_ATTENTION_MASK].to(
				device=device, dtype=torch.bool
			)
		else:
			tasks = list(batch["task"])
			lang_toks, lang_mask = self.tokenize_tasks(tasks, device=device)
			model_batch[OBS_LANGUAGE_TOKENS] = lang_toks
			model_batch[OBS_LANGUAGE_ATTENTION_MASK] = lang_mask

		if include_action:
			actions = batch[ACTION].to(device=device, dtype=torch.float32)
			model_batch[ACTION] = self._normalize_action(actions)
		return model_batch

	def forward(
		self,
		batch: Mapping[str, Any],
		*,
		noise: Tensor | None = None,
		time: Tensor | None = None,
	) -> tuple[Tensor, dict[str, float]]:
		"""Compute masked flow-matching ``L_act`` + weighted ``L_done`` and return ``(total_loss, loss_dict)``."""
		device = self.action_mean.device
		model_batch = self._prepare_model_batch(batch, include_action=True)
		action_mask = batch["action_mask"].to(device=device, dtype=torch.bool)
		end_target = batch["end_step_target"].to(device=device, dtype=torch.long)
		sample_weight = (
			batch["sample_weight"].to(device=device, dtype=torch.float32)
			if "sample_weight" in batch
			else torch.ones(end_target.shape[0], device=device, dtype=torch.float32)
		)

		self._captured_prefix_out = None
		self._captured_prefix_pad_masks = None

		images, img_masks = self.policy.prepare_images(model_batch)
		state = self.policy.prepare_state(model_batch)
		lang_tokens = model_batch[OBS_LANGUAGE_TOKENS]
		lang_masks = model_batch[OBS_LANGUAGE_ATTENTION_MASK]
		actions = self.policy.prepare_action(model_batch)
		if noise is not None:
			noise = pad_vector(noise.to(device=device, dtype=actions.dtype), self.policy.config.max_action_dim)
		if time is not None:
			time = time.to(device=device, dtype=torch.float32)

		use_cuda_bf16 = device.type == "cuda"
		with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_cuda_bf16):
			losses = self.policy.model.forward(
				images, img_masks, lang_tokens, lang_masks, state, actions, noise=noise, time=time
			)
		losses = losses[:, :, : self.action_dim].float()

		if action_mask.any():
			w_3d = action_mask.unsqueeze(-1).float() * sample_weight.view(-1, 1, 1)
			l_act = (losses * w_3d).sum() / w_3d.sum().clamp_min(1.0)
		else:
			l_act = losses.sum() * 0.0

		pooled_prefix = self._pool_prefix_features()
		end_logits = self.end_head(pooled_prefix)
		l_done = compute_end_step_loss(end_logits, end_target, sample_weight, self.end_class_weights)

		total_loss = l_act + self.lambda_done * l_done
		loss_dict = {
			"loss": float(total_loss.detach().item()),
			"l_act": float(l_act.detach().item()),
			"l_done": float(l_done.detach().item()),
		}
		return total_loss, loss_dict

	@torch.no_grad()
	def predict_chunk(
		self,
		batch: Mapping[str, Any],
		*,
		noise: Tensor | None = None,
	) -> tuple[Tensor, Tensor]:
		"""Integrate ``num_steps`` Euler steps from noise and return ``(action_chunk_rad, end_step_probs)``."""
		was_training = self.training
		self.eval()
		try:
			device = self.action_mean.device
			model_batch = self._prepare_model_batch(batch, include_action=False)
			self._captured_prefix_out = None
			self._captured_prefix_pad_masks = None

			images, img_masks = self.policy.prepare_images(model_batch)
			state = self.policy.prepare_state(model_batch)
			lang_tokens = model_batch[OBS_LANGUAGE_TOKENS]
			lang_masks = model_batch[OBS_LANGUAGE_ATTENTION_MASK]
			if noise is not None:
				noise = pad_vector(noise.to(device=device, dtype=torch.float32), self.policy.config.max_action_dim)

			use_cuda_bf16 = device.type == "cuda"
			with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_cuda_bf16):
				actions_norm = self.policy.model.sample_actions(
					images, img_masks, lang_tokens, lang_masks, state, noise=noise
				)
			actions_rad = self._unnormalize_action(actions_norm[:, :, : self.action_dim])
			pooled_prefix = self._pool_prefix_features()
			end_logits = self.end_head(pooled_prefix)
			end_probs = F.softmax(end_logits.float(), dim=-1)
			return actions_rad, end_probs
		finally:
			self.train(was_training)

	def _checkpoint_meta(self) -> dict[str, Any]:
		return {
			"vlm_model_name": self.vlm_model_name,
			"init_from": self.init_from,
			"freeze_vision_encoder": self.freeze_vision_encoder,
			"train_expert_only": self.train_expert_only,
			"train_state_proj": self.train_state_proj,
			"resize_imgs_with_padding": list(self.resize_imgs_with_padding),
			"num_steps": self.num_steps,
			"camera_shapes": {k: list(v) for k, v in self.camera_shapes.items()},
			"end_head_hidden_dim": self.end_head_hidden_dim,
			"end_head_dropout": self.end_head_dropout,
			"tiny_for_test": self.tiny_for_test,
		}

	@classmethod
	def load_checkpoint(
		cls,
		ckpt_dir: str | Path,
		*,
		device: str | torch.device = "cpu",
		local_files_only: bool = True,
	) -> SmolVLAWithEndHead:
		from safetensors.torch import load_file

		src = Path(ckpt_dir)
		meta = json.loads((src / "vla_config.json").read_text(encoding="utf-8"))
		mc = meta["model_config"]
		normalizer = ActionNormalizer.load(src / "normalizer.json")
		model = cls(
			horizon=int(meta["horizon"]),
			vlm_model_name=mc["vlm_model_name"],
			init_from=None,
			freeze_vision_encoder=bool(mc["freeze_vision_encoder"]),
			train_expert_only=bool(mc["train_expert_only"]),
			train_state_proj=bool(mc["train_state_proj"]),
			resize_imgs_with_padding=tuple(mc["resize_imgs_with_padding"]),
			num_steps=int(mc["num_steps"]),
			lambda_done=float(meta["lambda_done"]),
			normalizer=normalizer,
			camera_shapes={k: tuple(v) for k, v in mc["camera_shapes"].items()},
			end_head_hidden_dim=int(mc["end_head_hidden_dim"]),
			end_head_dropout=float(mc["end_head_dropout"]),
			tiny_for_test=bool(mc.get("tiny_for_test", False)),
			local_files_only=local_files_only,
		)
		sd = load_file(str(src / "model.safetensors"), device="cpu")
		model.load_state_dict(sd, strict=True)
		model.to(device)
		model.eval()
		return model


class ACTWithEndHead(_BaseVLAWrapper):
	"""LeRobot 0.6 ``ACTPolicy`` wrapped with masked L1 + 10*KL and H+2 End-Step Head."""

	family = "act"

	def __init__(
		self,
		*,
		horizon: int = DEFAULT_HORIZON["act"],
		vision_backbone: str = "resnet18",
		pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1",
		dim_model: int = 512,
		n_heads: int = 8,
		dim_feedforward: int = 3200,
		n_encoder_layers: int = 4,
		n_decoder_layers: int = 1,
		n_vae_encoder_layers: int = 4,
		latent_dim: int = 32,
		kl_weight: float = 10.0,
		dropout: float = 0.1,
		lambda_done: float = 0.1,
		normalizer: ActionNormalizer | None = None,
		class_weights: Tensor | np.ndarray | None = None,
		camera_shapes: Mapping[str, tuple[int, int, int]] | None = None,
		end_head_hidden_dim: int = 256,
		end_head_dropout: float = 0.1,
		tiny_for_test: bool = False,
		bowl_token: bool = False,
	) -> None:
		"""``bowl_token``: feed the target-bowl one-hot (state ``[31:34]``) as its own encoder token instead of inside
		the state token, which then sees those three slots as zero."""
		super().__init__(
			horizon=horizon,
			state_dim=STATE_DIM,
			action_dim=ACTION_DIM,
			lambda_done=lambda_done,
			normalizer=normalizer,
			class_weights=class_weights,
		)
		if tiny_for_test:
			pretrained_backbone_weights = None
			dim_model = 64
			n_heads = 4
			dim_feedforward = 128
			n_encoder_layers = 1
			n_decoder_layers = 1
			n_vae_encoder_layers = 1
			latent_dim = 16
			end_head_hidden_dim = 64

		self.vision_backbone = vision_backbone
		self.pretrained_backbone_weights = pretrained_backbone_weights
		self.dim_model = int(dim_model)
		self.n_heads = int(n_heads)
		self.dim_feedforward = int(dim_feedforward)
		self.n_encoder_layers = int(n_encoder_layers)
		self.n_decoder_layers = int(n_decoder_layers)
		self.n_vae_encoder_layers = int(n_vae_encoder_layers)
		self.latent_dim = int(latent_dim)
		self.kl_weight = float(kl_weight)
		self.dropout = float(dropout)
		self.camera_shapes = {k: tuple(v) for k, v in (camera_shapes or DEFAULT_CAMERA_SHAPES).items()}
		self.end_head_hidden_dim = int(end_head_hidden_dim)
		self.end_head_dropout = float(end_head_dropout)
		self.tiny_for_test = bool(tiny_for_test)
		self.bowl_token = bool(bowl_token)

		input_features = {
			cam: PolicyFeature(type=FeatureType.VISUAL, shape=self.camera_shapes[cam]) for cam in STUDENT_CAMERAS
		}
		input_features[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,))
		if self.bowl_token:
			input_features[OBS_ENV_STATE] = PolicyFeature(
				type=FeatureType.ENV, shape=(BOWL_SLICE.stop - BOWL_SLICE.start,)
			)
		output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))}

		cfg = ACTConfig(
			chunk_size=self.horizon,
			n_action_steps=self.horizon,
			vision_backbone=self.vision_backbone,
			pretrained_backbone_weights=self.pretrained_backbone_weights,
			dim_model=self.dim_model,
			n_heads=self.n_heads,
			dim_feedforward=self.dim_feedforward,
			n_encoder_layers=self.n_encoder_layers,
			n_decoder_layers=self.n_decoder_layers,
			use_vae=True,
			latent_dim=self.latent_dim,
			n_vae_encoder_layers=self.n_vae_encoder_layers,
			kl_weight=self.kl_weight,
			dropout=self.dropout,
			device="cpu",
		)
		cfg.input_features = input_features
		cfg.output_features = output_features
		self.policy = ACTPolicy(cfg)

		# Register ImageNet normalization buffers for ACT ResNet18 inputs.
		img_mean = torch.tensor(IMAGENET_STATS["mean"], dtype=torch.float32).view(1, 3, 1, 1)
		img_std = torch.tensor(IMAGENET_STATS["std"], dtype=torch.float32).view(1, 3, 1, 1)
		self.register_buffer("_imagenet_mean", img_mean)
		self.register_buffer("_imagenet_std", img_std)

		self.end_head = EndStepHead(
			in_dim=self.dim_model * 2,
			num_classes=self.num_end_classes,
			hidden_dim=self.end_head_hidden_dim,
			dropout=self.end_head_dropout,
		)
		self._captured_encoder_out: Tensor | None = None
		self.policy.model.encoder.register_forward_hook(self._hook_encoder)

	def _hook_encoder(self, _module, _args, output: Tensor) -> None:
		# ACT encoder outputs (S_enc, B, dim_model).
		self._captured_encoder_out = output

	def _pool_encoder_features(self) -> Tensor:
		if self._captured_encoder_out is None:
			raise RuntimeError("ACT encoder output was not captured during forward pass")
		enc = self._captured_encoder_out.transpose(0, 1).float()  # (B, S_enc, dim_model)
		self._captured_encoder_out = None
		mean_pooled = enc.mean(dim=1)
		# Token 0 is latent, token 1 is robot_state projection (since OBS_STATE is configured).
		state_tok = enc[:, 1, :]
		return torch.cat([mean_pooled, state_tok], dim=-1)

	def _prepare_act_batch(self, batch: Mapping[str, Any], *, include_action: bool) -> dict[str, Any]:
		device = self.action_mean.device
		state = batch[OBS_STATE].to(device=device, dtype=torch.float32)
		mean = self._imagenet_mean.to(device=device, dtype=torch.float32)
		std = self._imagenet_std.to(device=device, dtype=torch.float32)

		imgs = []
		for cam in STUDENT_CAMERAS:
			img = batch[cam].to(device=device, dtype=torch.float32)
			if img.ndim == 5:
				img = img[:, -1]
			imgs.append((img - mean) / std)

		state_norm = self._normalize_state(state)
		if self.bowl_token:
			state_norm = state_norm.clone()
			state_norm[:, BOWL_SLICE] = 0.0
		act_batch: dict[str, Any] = {OBS_STATE: state_norm, OBS_IMAGES: imgs}
		if self.bowl_token:
			act_batch[OBS_ENV_STATE] = state[:, BOWL_SLICE]  # raw one-hot ("", green, pink)
		if include_action:
			actions = batch[ACTION].to(device=device, dtype=torch.float32)
			act_batch[ACTION] = self._normalize_action(actions)
			act_batch["action_is_pad"] = ~batch["action_mask"].to(device=device, dtype=torch.bool)
		return act_batch

	def forward(self, batch: Mapping[str, Any]) -> tuple[Tensor, dict[str, float]]:
		"""Compute masked L1 + kl_weight*KL + lambda_done*L_done and return ``(total_loss, loss_dict)``."""
		device = self.action_mean.device
		act_batch = self._prepare_act_batch(batch, include_action=True)
		action_mask = batch["action_mask"].to(device=device, dtype=torch.bool)
		end_target = batch["end_step_target"].to(device=device, dtype=torch.long)
		sample_weight = (
			batch["sample_weight"].to(device=device, dtype=torch.float32)
			if "sample_weight" in batch
			else torch.ones(end_target.shape[0], device=device, dtype=torch.float32)
		)

		self._captured_encoder_out = None
		use_cuda_bf16 = device.type == "cuda"
		with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_cuda_bf16):
			actions_hat, (mu_hat, log_sigma_x2_hat) = self.policy.model(act_batch)

		actions_hat_f = actions_hat.float()
		abs_err = F.l1_loss(act_batch[ACTION].float(), actions_hat_f, reduction="none")
		if action_mask.any():
			w_3d = action_mask.unsqueeze(-1).float() * sample_weight.view(-1, 1, 1)
			l1_loss = (abs_err * w_3d).sum() / w_3d.sum().clamp_min(1.0)
		else:
			l1_loss = actions_hat_f.sum() * 0.0

		has_valid_actions = action_mask.any(dim=1)
		if self.policy.config.use_vae and log_sigma_x2_hat is not None and has_valid_actions.any():
			per_sample_kld = (
				-0.5 * (1.0 + log_sigma_x2_hat.float() - mu_hat.float().pow(2) - log_sigma_x2_hat.float().exp())
			).sum(dim=-1)
			w_kld = sample_weight[has_valid_actions]
			kld_loss = (per_sample_kld[has_valid_actions] * w_kld).sum() / w_kld.sum().clamp_min(1e-6)
		else:
			kld_loss = actions_hat_f.sum() * 0.0

		l_act = l1_loss + self.kl_weight * kld_loss
		pooled_enc = self._pool_encoder_features()
		end_logits = self.end_head(pooled_enc)
		l_done = compute_end_step_loss(end_logits, end_target, sample_weight, self.end_class_weights)

		total_loss = l_act + self.lambda_done * l_done
		loss_dict = {
			"loss": float(total_loss.detach().item()),
			"l_act": float(l_act.detach().item()),
			"l1_loss": float(l1_loss.detach().item()),
			"kld_loss": float(kld_loss.detach().item()),
			"l_done": float(l_done.detach().item()),
		}
		return total_loss, loss_dict

	@torch.no_grad()
	def predict_chunk(self, batch: Mapping[str, Any]) -> tuple[Tensor, Tensor]:
		"""Predict chunk with zero latent and return ``(action_chunk_rad, end_step_probs)``."""
		was_training = self.training
		self.eval()
		try:
			device = self.action_mean.device
			act_batch = self._prepare_act_batch(batch, include_action=False)
			self._captured_encoder_out = None
			use_cuda_bf16 = device.type == "cuda"
			with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_cuda_bf16):
				actions_hat, _ = self.policy.model(act_batch)
			actions_rad = self._unnormalize_action(actions_hat)
			pooled_enc = self._pool_encoder_features()
			end_logits = self.end_head(pooled_enc)
			end_probs = F.softmax(end_logits.float(), dim=-1)
			return actions_rad, end_probs
		finally:
			self.train(was_training)

	def _checkpoint_meta(self) -> dict[str, Any]:
		return {
			"vision_backbone": self.vision_backbone,
			"pretrained_backbone_weights": self.pretrained_backbone_weights,
			"dim_model": self.dim_model,
			"n_heads": self.n_heads,
			"dim_feedforward": self.dim_feedforward,
			"n_encoder_layers": self.n_encoder_layers,
			"n_decoder_layers": self.n_decoder_layers,
			"n_vae_encoder_layers": self.n_vae_encoder_layers,
			"latent_dim": self.latent_dim,
			"kl_weight": self.kl_weight,
			"dropout": self.dropout,
			"camera_shapes": {k: list(v) for k, v in self.camera_shapes.items()},
			"end_head_hidden_dim": self.end_head_hidden_dim,
			"end_head_dropout": self.end_head_dropout,
			"tiny_for_test": self.tiny_for_test,
			"bowl_token": self.bowl_token,
		}

	def get_optim_param_groups(
		self,
		lr: float,
		*,
		weight_decay: float = 1e-4,
		backbone_lr: float | None = None,
		end_head_lr: float | None = None,
	) -> list[dict[str, Any]]:
		"""Backbone, End-Step Head and the rest of ACT as separate groups.

		The End-Step Head is initialised from scratch, so the ACT default LR (1e-5, tuned for 100k+ steps) leaves it
		near its initial prior within a 20k-step budget; ``end_head_lr`` lets it train at its own rate."""
		bb_lr = float(backbone_lr) if backbone_lr is not None else float(lr)
		head_lr = float(end_head_lr) if end_head_lr is not None else float(lr)
		rest, bb, head = [], [], []
		for n, p in self.named_parameters():
			if not p.requires_grad:
				continue
			if n.startswith("policy.model.backbone."):
				bb.append(p)
			elif n.startswith("end_head."):
				head.append(p)
			else:
				rest.append(p)
		groups = [{"params": rest, "lr": float(lr), "weight_decay": float(weight_decay)}]
		if bb:
			groups.append({"params": bb, "lr": bb_lr, "weight_decay": float(weight_decay)})
		if head:
			groups.append({"params": head, "lr": head_lr, "weight_decay": float(weight_decay)})
		return groups

	@classmethod
	def load_checkpoint(
		cls,
		ckpt_dir: str | Path,
		*,
		device: str | torch.device = "cpu",
	) -> ACTWithEndHead:
		from safetensors.torch import load_file

		src = Path(ckpt_dir)
		meta = json.loads((src / "vla_config.json").read_text(encoding="utf-8"))
		mc = meta["model_config"]
		normalizer = ActionNormalizer.load(src / "normalizer.json")
		model = cls(
			horizon=int(meta["horizon"]),
			vision_backbone=mc["vision_backbone"],
			pretrained_backbone_weights=None,  # weights loaded from checkpoint below
			dim_model=int(mc["dim_model"]),
			n_heads=int(mc["n_heads"]),
			dim_feedforward=int(mc["dim_feedforward"]),
			n_encoder_layers=int(mc["n_encoder_layers"]),
			n_decoder_layers=int(mc["n_decoder_layers"]),
			n_vae_encoder_layers=int(mc["n_vae_encoder_layers"]),
			latent_dim=int(mc["latent_dim"]),
			kl_weight=float(mc["kl_weight"]),
			dropout=float(mc["dropout"]),
			lambda_done=float(meta["lambda_done"]),
			normalizer=normalizer,
			camera_shapes={k: tuple(v) for k, v in mc["camera_shapes"].items()},
			end_head_hidden_dim=int(mc["end_head_hidden_dim"]),
			end_head_dropout=float(mc["end_head_dropout"]),
			tiny_for_test=bool(mc.get("tiny_for_test", False)),
			bowl_token=bool(mc.get("bowl_token", False)),
		)
		sd = load_file(str(src / "model.safetensors"), device="cpu")
		model.load_state_dict(sd, strict=True)
		model.to(device)
		model.eval()
		return model


def load_act_weights_expanding(model: ACTWithEndHead, src_state_dict: Mapping[str, Tensor]) -> list[str]:
	"""Load an ACT state dict without the bowl token into ``model`` (which has it): every shared tensor is copied, the
	1-d positional embedding keeps its old rows and gets one new row (left at its fresh init), and the new env-state
	projection stays freshly initialised. Returns the names of the freshly initialised tensors."""
	if not model.bowl_token:
		raise ValueError("load_act_weights_expanding needs a model with bowl_token=True")
	if any("encoder_env_state_input_proj" in k for k in src_state_dict):
		raise ValueError("source already has a bowl token; load it with load_state_dict(strict=True)")
	own = model.state_dict()
	fresh = sorted(set(own) - set(src_state_dict))
	merged = dict(own)
	for k, v in src_state_dict.items():
		if k not in own:
			raise ValueError(f"unexpected tensor {k} in source state dict")
		if k == POS_EMBED_KEY:
			if own[k].shape[0] != v.shape[0] + 1 or own[k].shape[1:] != v.shape[1:]:
				raise ValueError(f"{k}: cannot expand {tuple(v.shape)} to {tuple(own[k].shape)}")
			merged[k] = torch.cat([v.to(own[k].dtype), own[k][v.shape[0] :]], dim=0)
			fresh.append(f"{k}[{v.shape[0]}]")
		elif own[k].shape != v.shape:
			raise ValueError(f"{k}: shape {tuple(v.shape)} != {tuple(own[k].shape)}")
		else:
			merged[k] = v
	model.load_state_dict(merged, strict=True)
	return fresh


def load_vla_checkpoint(
	ckpt_dir: str | Path,
	*,
	device: str | torch.device = "cpu",
	local_files_only: bool = True,
) -> SmolVLAWithEndHead | ACTWithEndHead:
	"""Load either ``SmolVLAWithEndHead`` or ``ACTWithEndHead`` from a saved checkpoint directory."""
	src = Path(ckpt_dir)
	meta = json.loads((src / "vla_config.json").read_text(encoding="utf-8"))
	family = meta["family"]
	if family == "smolvla":
		return SmolVLAWithEndHead.load_checkpoint(src, device=device, local_files_only=local_files_only)
	if family == "act":
		return ACTWithEndHead.load_checkpoint(src, device=device)
	raise ValueError(f"unknown VLA checkpoint family {family!r}")
