"""Verify the E:\\il\\lerobot-cuda environment on RTX 5090 (Blackwell, sm_120).

Checks:
1. PyTorch CUDA availability, device info, and fp32 / bf16 matmul on GPU.
2. LeRobotDataset import and pyav video decoding via decode_video_frames(..., video_backend="pyav").
3. ACTPolicy synthetic forward + backward pass on CUDA.
4. SmolVLAPolicy HuggingFace availability/cache status + bf16 forward + backward pass on CUDA.

Usage:
    E:\\il\\lerobot-cuda\\Scripts\\python.exe scripts/vla/check_cuda_env.py
    E:\\il\\lerobot-cuda\\Scripts\\python.exe scripts/vla/check_cuda_env.py --quick
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import av
import numpy as np
import torch


def _mib(num_bytes: int) -> float:
	return num_bytes / (1024**2)


def check_cuda_matmul() -> None:
	print("[1/4] Checking CUDA & matmul...")
	if not torch.cuda.is_available():
		raise RuntimeError("torch.cuda.is_available() is False")
	dev_idx = torch.cuda.current_device()
	props = torch.cuda.get_device_properties(dev_idx)
	cap = torch.cuda.get_device_capability(dev_idx)
	print(f"  torch={torch.__version__}, cuda={torch.version.cuda}, device={props.name}")
	print(f"  compute_capability=sm_{cap[0]}{cap[1]}, total_vram={_mib(props.total_memory):.0f} MiB")

	for dtype in (torch.float32, torch.bfloat16):
		a = torch.randn(512, 512, device="cuda", dtype=dtype)
		b = torch.randn(512, 512, device="cuda", dtype=dtype)
		c = a @ b
		torch.cuda.synchronize()
		if not torch.isfinite(c).all():
			raise RuntimeError(f"Non-finite matmul output for {dtype}")
		print(f"  matmul {dtype}: ok (shape={tuple(c.shape)})")


def check_lerobot_pyav() -> None:
	print("[2/4] Checking LeRobotDataset & pyav decoding...")
	from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: F401
	from lerobot.datasets.video_utils import decode_video_frames

	with tempfile.TemporaryDirectory() as tmpdir:
		mp4_path = Path(tmpdir) / "synthetic.mp4"
		with av.open(str(mp4_path), mode="w") as container:
			stream = container.add_stream("libx264", rate=10)
			stream.width = 64
			stream.height = 64
			stream.pix_fmt = "yuv420p"
			for i in range(4):
				arr = np.full((64, 64, 3), fill_value=(i * 50) % 255, dtype=np.uint8)
				frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
				for pkt in stream.encode(frame):
					container.mux(pkt)
			for pkt in stream.encode():
				container.mux(pkt)

		decoded = decode_video_frames(mp4_path, [0.0, 0.1], tolerance_s=0.05, backend="pyav")
		print(f"  LeRobotDataset import: ok, av={av.__version__}")
		print(f"  pyav decoded tensor: shape={tuple(decoded.shape)}, dtype={decoded.dtype}")


def check_act_policy() -> None:
	print("[3/4] Checking ACTPolicy GPU forward + backward...")
	from lerobot.configs.types import FeatureType, PolicyFeature
	from lerobot.policies.act.configuration_act import ACTConfig
	from lerobot.policies.act.modeling_act import ACTPolicy

	torch.cuda.empty_cache()
	torch.cuda.reset_peak_memory_stats()

	cfg = ACTConfig(
		chunk_size=16,
		n_action_steps=16,
		pretrained_backbone_weights=None,
		input_features={
			"observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
			"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
		},
		output_features={
			"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
		},
	)
	policy = ACTPolicy(cfg).to("cuda")
	policy.train()

	batch = {
		"observation.images.top": torch.rand(2, 3, 256, 256, device="cuda", dtype=torch.float32),
		"observation.state": torch.randn(2, 6, device="cuda", dtype=torch.float32),
		"action": torch.randn(2, 16, 6, device="cuda", dtype=torch.float32),
		"action_is_pad": torch.zeros(2, 16, device="cuda", dtype=torch.bool),
	}
	loss, loss_dict = policy.forward(batch)
	loss.backward()
	torch.cuda.synchronize()
	peak_mib = _mib(torch.cuda.max_memory_allocated())
	print(f"  ACTPolicy forward+backward: ok (loss={loss.item():.4f}, peak_vram={peak_mib:.1f} MiB)")
	del policy, batch, loss, loss_dict
	torch.cuda.empty_cache()


def _hf_status(repo_id: str, filename: str = "config.json") -> tuple[bool, str]:
	from huggingface_hub import HfApi, try_to_load_from_cache

	cached = try_to_load_from_cache(repo_id, filename)
	cached_str = "cached" if isinstance(cached, str) else "not_cached"
	try:
		info = HfApi().model_info(repo_id)
		return True, f"accessible (sha={info.sha[:8]}, cache={cached_str})"
	except Exception as exc:
		return False, f"unreachable ({type(exc).__name__}: {exc}, cache={cached_str})"


def check_smolvla_policy(quick: bool) -> None:
	print("[4/4] Checking SmolVLAPolicy (HF availability + GPU bf16 forward + backward)...")
	from lerobot.configs.types import FeatureType, PolicyFeature
	from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
	from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

	vlm_repo = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
	base_repo = "lerobot/smolvla_base"
	vlm_ok, vlm_msg = _hf_status(vlm_repo)
	base_ok, base_msg = _hf_status(base_repo)
	print(f"  HF {vlm_repo}: {vlm_msg}")
	print(f"  HF {base_repo}: {base_msg}")

	if not quick and not vlm_ok:
		raise RuntimeError(f"HF repo {vlm_repo} is not accessible: {vlm_msg}")

	torch.cuda.empty_cache()
	torch.cuda.reset_peak_memory_stats()

	load_vlm = not quick
	cfg = SmolVLAConfig(
		chunk_size=16,
		n_action_steps=16,
		load_vlm_weights=load_vlm,
		input_features={
			"observation.images.camera1": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
			"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
		},
		output_features={
			"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
		},
	)
	policy = SmolVLAPolicy(cfg).to(device="cuda", dtype=torch.bfloat16)
	policy.train()

	batch = {
		"observation.images.camera1": torch.rand(1, 3, 256, 256, device="cuda", dtype=torch.bfloat16),
		"observation.state": torch.randn(1, 6, device="cuda", dtype=torch.bfloat16),
		"observation.language.tokens": torch.ones(1, 16, device="cuda", dtype=torch.long),
		"observation.language.attention_mask": torch.ones(1, 16, device="cuda", dtype=torch.bool),
		"action": torch.randn(1, 16, 6, device="cuda", dtype=torch.bfloat16),
		"action_is_pad": torch.zeros(1, 16, device="cuda", dtype=torch.bool),
	}
	with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
		loss, loss_dict = policy.forward(batch)
	loss.backward()
	torch.cuda.synchronize()
	peak_mib = _mib(torch.cuda.max_memory_allocated())
	mode = "quick (random VLM init)" if quick else "full (pretrained VLM weights)"
	print(f"  SmolVLAPolicy [{mode}] bf16 forward+backward: ok (loss={loss.item():.4f}, peak_vram={peak_mib:.1f} MiB)")
	del policy, batch, loss, loss_dict
	torch.cuda.empty_cache()


def main() -> int:
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument(
		"--quick",
		action="store_true",
		help="Skip downloading full VLM weights if not testing full HF weight instantiation.",
	)
	args = parser.parse_args()

	check_cuda_matmul()
	check_lerobot_pyav()
	check_act_policy()
	check_smolvla_policy(quick=args.quick)
	print("ALL CHECKS PASSED")
	return 0


if __name__ == "__main__":
	sys.exit(main())
