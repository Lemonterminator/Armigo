#!/usr/bin/env python
"""Generate the tiny CPU ONNX fixture (``base=4``, seed 0) used to test ``BrickSegNet`` in ``.venv`` without torch."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import torch

from digital_twin_armigo.perception_nn.model import BrickSegUNet, export_onnx

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "tests" / "data"


def _git_commit() -> str:
	try:
		return subprocess.run(
			["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
		).stdout.strip()
	except (OSError, subprocess.CalledProcessError):
		return ""


def make_fixture(out_dir: Path, base: int = 4, seed: int = 0) -> tuple[Path, Path, str]:
	out_dir = Path(out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)
	torch.manual_seed(int(seed))
	model = BrickSegUNet(base=int(base), classes=4, heat=2)
	model.eval()
	onnx_path = out_dir / "brickseg_tiny.onnx"
	meta_path = out_dir / "brickseg_tiny.json"
	backend = export_onnx(model, onnx_path, input_hw=(240, 320), opset=17)
	meta = {
		"input_hw": [240, 320],
		"mean": [0.485, 0.456, 0.406],
		"std": [0.229, 0.224, 0.225],
		"classes": ["bg", "top", "bottom", "side"],
		"heat_channels": 2,
		"data_manifest_sha": "",
		"git_commit": _git_commit(),
		"val_metrics": {},
		"base": int(base),
		"seed": int(seed),
		"onnx_backend": backend,
	}
	meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
	return onnx_path, meta_path, backend


def main(argv=None) -> int:
	p = argparse.ArgumentParser(description=__doc__)
	p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
	p.add_argument("--base", type=int, default=4)
	p.add_argument("--seed", type=int, default=0)
	args = p.parse_args(argv)
	onnx_path, meta_path, backend = make_fixture(args.out_dir, base=args.base, seed=args.seed)
	size_kb = onnx_path.stat().st_size / 1024.0
	print(f"wrote {onnx_path} ({size_kb:.1f} KB via {backend}) and {meta_path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
