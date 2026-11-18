"""Supervised cross detector as a per-view scorer (anomaly env): one ``train_sup.py`` checkpoint, both cameras.

The scorer reads a spec written by ``export_sup.py``::

    {"kind": "sup", "checkpoint": "head-lora.pt", "mode": "lora", "rank": 8, "alpha": 16.0, "input_size": 256,
     "crop_size": 256, "head": "both", "view_q": 0.99,
     "cameras": {"top": {"cls": {"t": .., "med": ..}, "seg": {..}}, "wrist_cam": {..}}}

``score`` returns ``u`` = max over the heads of ``(raw - t) / (t - median)`` (see ``fuse_views.standardise_value``),
so every camera shares one scale and ``u >= 0`` is the single-view alarm at the calibration quantile; the natural
detection threshold of every camera is therefore ``0.0`` (raise it to trade recall for fewer false alarms).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from fuse_views import standardise_value
from infer_sup import load_net
from train_sup import score as raw_scores


class SupScorer:
	def __init__(self, spec_path: Path, device: torch.device) -> None:
		self.spec_path = Path(spec_path)
		self.spec = json.loads(self.spec_path.read_text(encoding="utf-8"))
		if self.spec.get("kind") != "sup":
			raise ValueError(f"{spec_path} is not a supervised detector spec")
		ckpt = Path(self.spec["checkpoint"])
		if not ckpt.is_absolute():
			ckpt = self.spec_path.parent / ckpt
		self.device = device
		self.input_size = int(self.spec["input_size"])
		self.crop_size = int(self.spec["crop_size"])
		self.net = load_net(ckpt, self.spec["mode"], int(self.spec["rank"]), float(self.spec["alpha"]), device)
		self.cameras: dict[str, dict] = self.spec["cameras"]

	def score(self, camera: str, rgb: np.ndarray) -> float:
		"""``u`` of one square RGB uint8 crop (resize to ``crop_size`` is the caller's job; ``train_sup`` resizes to
		``input_size``)."""
		stats = self.cameras[camera]
		x = torch.from_numpy(np.ascontiguousarray(rgb)[None]).to(self.device)
		s_cls, s_seg = raw_scores(self.net, x, self.input_size, 1)
		raw = {"cls": float(s_cls[0]), "seg": float(s_seg[0])}
		return max(standardise_value(raw[h], st["t"], st["med"]) for h, st in stats.items())
