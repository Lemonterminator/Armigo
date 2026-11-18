"""Score brick crops with the exported anomaly model (run in the anomaly env, see ``requirements.txt``).

Usage::

    E:/venvs/armigo-anomaly/Scripts/python scripts/anomaly/score.py crop1.png crop2.png some_dir/ [--model-dir DIR]

Inputs are square RGB crops around the brick as ``render_dataset.py`` writes them (any size; resized to the model's
input). Prints one JSON object: per image its anomaly ``score``, the ``threshold`` chosen on validation normals and
``anomalous`` (score above threshold).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_eval import DEFAULT_REPORT, build_model, score


def load(model_dir: Path, device: torch.device):
	meta = json.loads((model_dir / "best.json").read_text(encoding="utf-8"))
	model = build_model(meta["model"], pre_trained=False)
	state = torch.load(model_dir / "best.pt", map_location="cpu", weights_only=True)
	if meta["model"] == "padim":  # buffers start empty: give them the saved shapes before loading
		model.gaussian.mean = torch.empty_like(state["gaussian.mean"])
		model.gaussian.inv_covariance = torch.empty_like(state["gaussian.inv_covariance"])
	else:
		model.memory_bank = torch.empty_like(state["memory_bank"])
	model.load_state_dict(state)
	return model.to(device).eval(), meta


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	ap.add_argument("inputs", nargs="+", type=Path, help="PNG files or directories of PNGs")
	ap.add_argument("--model-dir", type=Path, default=DEFAULT_REPORT / "models")
	ap.add_argument("--batch", type=int, default=8)
	ap.add_argument("--cpu", action="store_true")
	args = ap.parse_args()
	paths: list[Path] = []
	for p in args.inputs:
		paths += sorted(p.glob("*.png")) if p.is_dir() else [p]
	device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
	model, meta = load(args.model_dir, device)
	imgs, ok = [], []
	for p in paths:
		im = cv2.imread(str(p))
		if im is None:
			continue
		side = int(meta.get("crop_size", 128))  # the training crops' size; the model input resize happens in score()
		if im.shape[:2] != (side, side):
			im = cv2.resize(im, (side, side), interpolation=cv2.INTER_AREA)
		imgs.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
		ok.append(p)
	scores = score(model, np.stack(imgs), meta["input_size"], device, args.batch) if imgs else np.zeros(0)
	thr = float(meta["threshold"])
	out = {
		"model": meta["model"],
		"threshold": thr,
		"scores": [
			{"path": str(p), "score": round(float(s), 6), "anomalous": bool(s > thr)}
			for p, s in zip(ok, scores, strict=True)
		],
		"unreadable": [str(p) for p in paths if p not in ok],
	}
	print(json.dumps(out, indent=2))


if __name__ == "__main__":
	main()
