"""Write the scorer spec of a supervised cross detector (``sup_scorer.py`` / ``score_server.py --sup``).

Run with any python that has numpy (``fuse_views`` is numpy only)::

    python scripts/anomaly/export_sup.py --ckpt results/anomaly/sup-lora/head-lora.pt \
        --calib "results/anomaly/sup-lora/scores-v6.npz:split=train" --out results/anomaly/models-sup/sup.json

``--calib`` selects the NORMAL crops the per-camera standardisation is estimated on (same selector syntax as
``fuse_views.py``); use bricks the checkpoint never trained on, otherwise the view threshold is too low and the
alarm rate on new bricks is higher than ``1 - view_q``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from fuse_views import calibration_stats, load


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	ap.add_argument("--ckpt", type=Path, required=True)
	ap.add_argument("--calib", required=True)
	ap.add_argument("--out", type=Path, required=True)
	ap.add_argument("--head", choices=("cls", "seg", "both"), default="both")
	ap.add_argument("--view-q", type=float, default=0.99)
	ap.add_argument("--mode", default="lora")
	ap.add_argument("--rank", type=int, default=8)
	ap.add_argument("--alpha", type=float, default=16.0)
	ap.add_argument("--input-size", type=int, default=256)
	a = ap.parse_args()
	cal = load(a.calib)
	normal = ~cal["defective"]
	cal = {k: v[normal] for k, v in cal.items()}
	spec = {
		"kind": "sup",
		"checkpoint": os.path.relpath(a.ckpt.resolve(), a.out.resolve().parent).replace("\\", "/"),
		"mode": a.mode,
		"rank": a.rank,
		"alpha": a.alpha,
		"input_size": a.input_size,
		"crop_size": a.input_size,
		"head": a.head,
		"view_q": a.view_q,
		"calibration": {"selector": a.calib, "n_crops": len(cal["paths"]), "n_bricks": len(np.unique(cal["layout"]))},
		"cameras": calibration_stats(cal, a.head, a.view_q),
	}
	a.out.parent.mkdir(parents=True, exist_ok=True)
	a.out.write_text(json.dumps(spec, indent=2), encoding="utf-8")
	print(json.dumps(spec["cameras"], indent=2))


if __name__ == "__main__":
	main()
