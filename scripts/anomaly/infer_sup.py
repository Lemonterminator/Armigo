"""Score every crop of a dataset with a trained supervised cross detector (``train_sup.py`` checkpoint).

Run with the anomaly venv's python::

    E:/venvs/armigo-anomaly/Scripts/python scripts/anomaly/infer_sup.py \
        --ckpt results/anomaly/sup-lora/head-lora.pt --data results/anomaly/dataset-v4-isaac \
        --split-from results/anomaly/dataset-v4 --out results/anomaly/sup-lora/scores-isaac.npz

The output ``.npz`` has one entry per crop of ``--data`` (after the real-top-crop filter used in training): ``paths``,
``layout_index``, ``camera``, ``stage``, ``label``, ``defective``, ``split``, ``role``, ``cls`` (image-head logit gap) and ``seg``
(max patch-head cell logit). ``role`` is ``test`` for the test layouts and ``val_normal`` / ``fit_normal`` for normal
train layouts, split exactly as ``train_sup.py`` split them (same seed and fraction), so the validation bricks are the
ones the checkpoint never trained on.

* ``--holdout``: the dataset was rendered with a fresh seed, so no brick was seen in training; every crop has role
  ``holdout`` (select ``split`` / ``label`` in ``fuse_views.py`` to pick calibration and evaluation bricks).
* ``--split-from DIR``: take the layout split from another dataset with the same layout indices (the MuJoCo set when
  scoring its Isaac re-render, so both domains use the same validation bricks).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from train_sup import CrossNet, read_rows, score, split_layouts


def load_net(ckpt: Path, mode: str, rank: int, alpha: float, device: torch.device) -> CrossNet:
	net = CrossNet(mode, rank, alpha, pretrained=True)
	state = torch.load(ckpt, map_location="cpu", weights_only=True)
	missing, unexpected = net.load_state_dict(state, strict=False)
	if unexpected:
		raise RuntimeError(f"unexpected keys in {ckpt}: {unexpected[:5]}")
	trained_missing = [k for k in missing if "backbone" not in k or ".a." in k or ".b." in k]
	if trained_missing:
		raise RuntimeError(f"checkpoint lacks trained weights: {trained_missing[:5]}")
	return net.to(device).to(memory_format=torch.channels_last).eval()


def validation_layouts(split_root: Path, real_top: bool, frac: float, seed: int) -> set[int]:
	"""Normal train layouts held out of ``train_sup.py`` training (identical call, same seed)."""
	rows = read_rows(split_root, False, real_top)
	idx_norm = [i for i, r in enumerate(rows) if r["split"] == "train" and r["label"] == "normal"]
	_, val = split_layouts(rows, idx_norm, frac, seed)
	return {int(rows[i]["layout_index"]) for i in val}


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	ap.add_argument("--ckpt", type=Path, required=True)
	ap.add_argument("--data", type=Path, required=True)
	ap.add_argument("--split-from", type=Path, default=None, help="dataset whose layout split defines val bricks")
	ap.add_argument("--out", type=Path, required=True)
	ap.add_argument("--mode", choices=["head", "lora", "full"], default="lora")
	ap.add_argument("--rank", type=int, default=8)
	ap.add_argument("--alpha", type=float, default=16.0)
	ap.add_argument("--input-size", type=int, default=256)
	ap.add_argument("--val-fraction", type=float, default=0.15)
	ap.add_argument("--seed", type=int, default=0)
	ap.add_argument("--no-real-top-crop", action="store_true")
	ap.add_argument("--holdout", action="store_true", help="every layout is unseen by the checkpoint: role 'holdout'")
	ap.add_argument("--batch", type=int, default=64)
	a = ap.parse_args()
	real_top = not a.no_real_top_crop
	device = torch.device("cuda")
	rows = read_rows(a.data, False, real_top)
	if a.holdout:
		role = np.array(["holdout"] * len(rows))
	else:
		val_layouts = validation_layouts(a.split_from or a.data, real_top, a.val_fraction, a.seed)
		role = np.array(
			[
				"test"
				if r["split"] == "test"
				else ("val_normal" if int(r["layout_index"]) in val_layouts else "fit_normal")
				for r in rows
			]
		)
	net = load_net(a.ckpt, a.mode, a.rank, a.alpha, device)
	size = a.input_size
	imgs = np.empty((len(rows), size, size, 3), np.uint8)
	for k, r in enumerate(rows):
		im = cv2.cvtColor(cv2.imread(str(a.data / r["path"])), cv2.COLOR_BGR2RGB)
		imgs[k] = im if im.shape[:2] == (size, size) else cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
	s_cls, s_seg = score(net, torch.from_numpy(imgs).to(device), a.input_size, a.batch)
	a.out.parent.mkdir(parents=True, exist_ok=True)
	np.savez_compressed(
		a.out,
		paths=np.array([r["path"] for r in rows]),
		layout_index=np.array([r["layout_index"] for r in rows]),
		camera=np.array([r["camera"] for r in rows]),
		stage=np.array([r["stage"] for r in rows]),
		label=np.array([r["label"] for r in rows]),
		defective=np.array([bool(r["defective"]) for r in rows]),
		role=role,
		split=np.array([r["split"] for r in rows]),
		cls=s_cls,
		seg=s_seg,
	)
	vals, counts = np.unique(role, return_counts=True)
	print(f"{len(rows)} crops from {a.data} -> {a.out}; roles {dict(zip(vals.tolist(), counts.tolist(), strict=True))}")


if __name__ == "__main__":
	main()
