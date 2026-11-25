#!/usr/bin/env python
"""Train BrickSegUNet on MuJoCo shards and export ONNX + metadata JSON (``lerobot-cuda`` env)."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from digital_twin_armigo.perception_nn.model import BrickSegUNet, export_onnx

REPO = Path(__file__).resolve().parents[2]
DEFAULT_DATA = Path("E:/Armigo_data_v3/perception/brickseg-v1")
INPUT_HW = (240, 320)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
CLASSES = ["bg", "top", "bottom", "side"]


def _git_commit() -> str:
	try:
		return subprocess.run(
			["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
		).stdout.strip()
	except (OSError, subprocess.CalledProcessError):
		return ""


def _manifest_sha(data_dir: Path) -> str:
	mf = data_dir / "manifest.json"
	if not mf.is_file():
		return ""
	return hashlib.sha256(mf.read_bytes()).hexdigest()


def render_corner_heatmaps(
	corners: np.ndarray,
	hw: tuple[int, int] = INPUT_HW,
	sigma: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
	"""Render ``(2, H, W)`` float32 Gaussian targets and ``(2, H, W)`` bool peak masks for visible corners."""
	h, w = hw
	heat = np.zeros((2, h, w), dtype=np.float32)
	pos_mask = np.zeros((2, h, w), dtype=bool)
	radius = int(np.ceil(3.0 * sigma))
	inv_two_sig2 = 1.0 / (2.0 * sigma * sigma)
	for ch, sl in ((0, slice(0, 4)), (1, slice(4, 8))):
		for u, v, vis in corners[sl]:
			if float(vis) <= 0.5:
				continue
			ui, vi = round(float(u)), round(float(v))
			if not (0 <= ui < w and 0 <= vi < h):
				continue
			pos_mask[ch, vi, ui] = True
			x0, x1 = max(0, ui - radius), min(w, ui + radius + 1)
			y0, y1 = max(0, vi - radius), min(h, vi + radius + 1)
			yy, xx = np.mgrid[y0:y1, x0:x1]
			g = np.exp(-((xx - float(u)) ** 2 + (yy - float(v)) ** 2) * inv_two_sig2).astype(np.float32)
			heat[ch, y0:y1, x0:x1] = np.maximum(heat[ch, y0:y1, x0:x1], g)
	return heat, pos_mask


def _augment(
	rgb: np.ndarray,
	cls_map: np.ndarray,
	corners: np.ndarray,
	rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	img = rgb.copy()
	cls_out = cls_map.copy()
	c_out = corners.copy()
	h, w = img.shape[:2]

	# Horizontal flip only (geometry invariant for symmetric brick + corners)
	if rng.random() < 0.5:
		img = img[:, ::-1].copy()
		cls_out = cls_out[:, ::-1].copy()
		c_out[:, 0] = (w - 1.0) - c_out[:, 0]

	# Brightness & contrast (+-20%)
	alpha = float(rng.uniform(0.8, 1.2))
	beta = float(rng.uniform(-0.2, 0.2)) * 255.0
	f = np.clip(img.astype(np.float32) * alpha + beta, 0.0, 255.0).astype(np.uint8)

	# Hue shift (+-0.03 turns ~= +-5.4 in OpenCV [0, 180) H channel)
	hue_shift = float(rng.uniform(-0.03, 0.03)) * 180.0
	if abs(hue_shift) > 0.1:
		hsv = cv2.cvtColor(f, cv2.COLOR_RGB2HSV).astype(np.float32)
		hsv[..., 0] = (hsv[..., 0] + hue_shift) % 180.0
		f = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)

	# Gaussian noise
	noise_std = float(rng.uniform(0.0, 8.0))
	if noise_std > 0.0:
		f = np.clip(f.astype(np.float32) + rng.normal(0.0, noise_std, f.shape), 0.0, 255.0).astype(np.uint8)

	# MJPEG compression simulation (quality 60-95)
	if rng.random() < 0.7:
		q = int(rng.integers(60, 96))
		ok, enc = cv2.imencode(".jpg", cv2.cvtColor(f, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), q])
		if ok:
			dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
			if dec is not None and dec.shape[:2] == (h, w):
				f = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)

	return f, cls_out, c_out


def load_samples(data_dir: Path, limit: int | None = None) -> tuple[list[dict], list[dict]]:
	shards = sorted(Path(data_dir).rglob("shard_*.npz"))
	if not shards:
		raise FileNotFoundError(f"no shard_*.npz found under {data_dir}")
	by_split: dict[str, list[dict]] = {"train": [], "val": []}
	for spath in shards:
		with np.load(spath, allow_pickle=False) as z:
			rgbs = z["rgb"]
			clss = z["cls"]
			corners = z["corners"]
			metas = z["meta"] if "meta" in z.files else None
			fallback_split = "val" if spath.parent.name == "val" else "train"
			for i in range(len(rgbs)):
				m = json.loads(str(metas[i])) if metas is not None else {}
				split = str(m.get("split") or fallback_split)
				if split not in by_split:
					continue
				if limit is not None and split == "train" and len(by_split["train"]) >= limit:
					if len(by_split["val"]) >= max(1, limit // 5):
						break
					continue
				by_split[split].append(
					{
						"rgb": rgbs[i].copy(),
						"cls": clss[i].copy(),
						"corners": corners[i].astype(np.float32).copy(),
					}
				)
				if limit is not None and len(by_split["train"]) >= limit and len(by_split["val"]) >= max(1, limit // 5):
					break
		if limit is not None and len(by_split["train"]) >= limit and len(by_split["val"]) >= max(1, limit // 5):
			break

	train_s = by_split["train"]
	val_s = by_split["val"]
	if not train_s:
		raise RuntimeError(f"no training samples found in {data_dir}")
	if not val_s:
		cut = max(1, len(train_s) // 5) if len(train_s) > 1 else 0
		val_s = train_s[-cut:] if cut else train_s[:1]
		if cut:
			train_s = train_s[:-cut]
	if limit is not None:
		n_val = min(len(val_s), max(1, limit // 5))
		n_train = max(1, limit - n_val)
		train_s = train_s[:n_train]
		val_s = val_s[:n_val]
	return train_s, val_s


def compute_class_weights(samples: list[dict], n_classes: int = 4, cap: float = 10.0) -> torch.Tensor:
	counts = np.zeros(n_classes, dtype=np.float64)
	for s in samples:
		counts += np.bincount(s["cls"].ravel(), minlength=n_classes)[:n_classes]
	freq = counts / max(counts.sum(), 1.0)
	inv = 1.0 / np.maximum(freq, 1e-6)
	inv = inv / max(float(inv.min()), 1e-6)
	weights = np.minimum(inv, cap)
	return torch.tensor(weights, dtype=torch.float32)


class BrickSegDataset(Dataset):
	def __init__(self, samples: list[dict], augment: bool = False, seed: int = 0) -> None:
		self.samples = samples
		self.augment = augment
		self.seed = int(seed)
		self._mean = np.asarray(MEAN, dtype=np.float32)
		self._std = np.asarray(STD, dtype=np.float32)

	def __len__(self) -> int:
		return len(self.samples)

	def __getitem__(self, idx: int):
		s = self.samples[idx]
		rgb, cls_map, corners = s["rgb"], s["cls"], s["corners"]
		if self.augment:
			rng = np.random.default_rng(np.random.SeedSequence([self.seed, idx]))
			rgb, cls_map, corners = _augment(rgb, cls_map, corners, rng)
		x = (rgb.astype(np.float32) / 255.0 - self._mean) / self._std
		x_t = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))
		cls_t = torch.from_numpy(cls_map.astype(np.int64))
		heat, pos = render_corner_heatmaps(corners, hw=rgb.shape[:2])
		return (
			x_t,
			cls_t,
			torch.from_numpy(heat),
			torch.from_numpy(pos),
			torch.from_numpy(corners.astype(np.float32)),
		)


def centernet_focal_loss(
	logits: torch.Tensor,
	target: torch.Tensor,
	pos_mask: torch.Tensor,
	alpha: float = 2.0,
	beta: float = 4.0,
) -> torch.Tensor:
	pred = torch.sigmoid(logits.float()).clamp(1e-6, 1.0 - 1e-6)
	pos_f = pos_mask.float()
	neg_f = 1.0 - pos_f
	pos_loss = -((1.0 - pred) ** alpha) * torch.log(pred) * pos_f
	neg_loss = -((1.0 - target.float()) ** beta) * (pred**alpha) * torch.log(1.0 - pred) * neg_f
	num_pos = pos_f.sum().clamp(min=1.0)
	return (pos_loss.sum() + neg_loss.sum()) / num_pos


def _extract_peaks_np(heat_2hw: np.ndarray, thr: float = 0.3, radius: int = 2) -> list[np.ndarray]:
	out = []
	k = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
	for ch in range(heat_2hw.shape[0]):
		h = heat_2hw[ch]
		is_pk = (h >= thr) & (h == cv2.dilate(h, k)) & (h > 0)
		ys, xs = np.nonzero(is_pk)
		out.append(np.column_stack([xs, ys]).astype(np.float32) if len(xs) else np.zeros((0, 2), dtype=np.float32))
	return out


@torch.no_grad()
def evaluate(
	model: BrickSegUNet,
	loader: DataLoader,
	class_weights: torch.Tensor,
	device: torch.device,
) -> dict[str, float]:
	model.eval()
	inter = 0
	union = 0
	correct_kpts = 0
	total_kpts = 0
	losses = []
	use_bf16 = device.type == "cuda"
	for x, cls_t, heat_t, pos_t, corners_t in loader:
		x = x.to(device)
		cls_t = cls_t.to(device)
		heat_t = heat_t.to(device)
		pos_t = pos_t.to(device)
		with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
			out = model(x)
		ce = F.cross_entropy(out[:, :4].float(), cls_t, weight=class_weights)
		fl = centernet_focal_loss(out[:, 4:6], heat_t, pos_t)
		losses.append(float((ce + fl).item()))
		pred_cls = out[:, :4].argmax(dim=1).cpu().numpy()
		true_cls = cls_t.cpu().numpy()
		inter += int(((pred_cls == 1) & (true_cls == 1)).sum())
		union += int(((pred_cls == 1) | (true_cls == 1)).sum())
		pred_heat = torch.sigmoid(out[:, 4:6].float()).cpu().numpy()
		corners_np = corners_t.numpy()
		for b in range(len(pred_heat)):
			pks = _extract_peaks_np(pred_heat[b])
			for ch, sl in ((0, slice(0, 4)), (1, slice(4, 8))):
				pts = pks[ch]
				for u, v, vis in corners_np[b, sl]:
					if vis <= 0.5:
						continue
					total_kpts += 1
					if len(pts) and float(np.linalg.norm(pts - [u, v], axis=1).min()) <= 2.0:
						correct_kpts += 1
	top_iou = float(inter / union) if union > 0 else 0.0
	pck_2px = float(correct_kpts / total_kpts) if total_kpts > 0 else 0.0
	return {
		"val_loss": float(np.mean(losses)) if losses else 0.0,
		"top_iou": round(top_iou, 6),
		"pck_2px": round(pck_2px, 6),
		"score": round(top_iou + pck_2px, 6),
	}


def train(args: argparse.Namespace) -> dict:
	torch.manual_seed(int(args.seed))
	np.random.seed(int(args.seed))
	device = torch.device(args.device)
	train_s, val_s = load_samples(args.data_dir, limit=args.limit)
	cw = compute_class_weights(train_s).to(device)
	train_ds = BrickSegDataset(train_s, augment=True, seed=args.seed)
	val_ds = BrickSegDataset(val_s, augment=False, seed=args.seed)
	train_loader = DataLoader(train_ds, batch_size=min(args.batch_size, len(train_ds)), shuffle=True, drop_last=False)
	val_loader = DataLoader(val_ds, batch_size=min(args.batch_size, len(val_ds)), shuffle=False, drop_last=False)

	model = BrickSegUNet(base=args.base, classes=4, heat=2).to(device)
	opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
	sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
	use_bf16 = device.type == "cuda"

	best_score = -1.0
	best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
	best_metrics: dict[str, float] = {}
	history = []
	t0 = time.perf_counter()

	for epoch in range(1, args.epochs + 1):
		model.train()
		train_ds.seed = int(args.seed) + epoch * 1009
		ep_losses = []
		for x, cls_t, heat_t, pos_t, _corners in train_loader:
			x = x.to(device)
			cls_t = cls_t.to(device)
			heat_t = heat_t.to(device)
			pos_t = pos_t.to(device)
			opt.zero_grad(set_to_none=True)
			with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
				out = model(x)
			loss = F.cross_entropy(out[:, :4].float(), cls_t, weight=cw) + centernet_focal_loss(
				out[:, 4:6], heat_t, pos_t
			)
			loss.backward()
			opt.step()
			ep_losses.append(float(loss.item()))
		sched.step()
		val_m = evaluate(model, val_loader, cw, device)
		row = {"epoch": epoch, "train_loss": round(float(np.mean(ep_losses)), 6), **val_m}
		history.append(row)
		if val_m["score"] >= best_score:
			best_score = val_m["score"]
			best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
			best_metrics = dict(val_m)

	args.out_dir.mkdir(parents=True, exist_ok=True)
	best_model = BrickSegUNet(base=args.base, classes=4, heat=2)
	best_model.load_state_dict(best_state)
	onnx_path = args.out_dir / "brickseg.onnx"
	backend = export_onnx(best_model, onnx_path, input_hw=INPUT_HW, opset=17)
	meta = {
		"input_hw": list(INPUT_HW),
		"mean": list(MEAN),
		"std": list(STD),
		"classes": list(CLASSES),
		"heat_channels": 2,
		"data_manifest_sha": _manifest_sha(args.data_dir),
		"git_commit": _git_commit(),
		"val_metrics": best_metrics,
		"base": int(args.base),
		"onnx_backend": backend,
	}
	(args.out_dir / "brickseg.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
	report = {
		"train_samples": len(train_s),
		"val_samples": len(val_s),
		"epochs": int(args.epochs),
		"wall_s": round(time.perf_counter() - t0, 2),
		"best_val": best_metrics,
		"history": history,
	}
	(args.out_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
	return report


def main(argv=None) -> int:
	p = argparse.ArgumentParser(description=__doc__)
	p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
	p.add_argument("--out-dir", type=Path, default=DEFAULT_DATA / "model")
	p.add_argument("--epochs", type=int, default=30)
	p.add_argument("--batch-size", type=int, default=32)
	p.add_argument("--lr", type=float, default=1e-3)
	p.add_argument("--base", type=int, default=24)
	p.add_argument("--limit", type=int, default=None)
	p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
	p.add_argument("--seed", type=int, default=0)
	args = p.parse_args(argv)
	rep = train(args)
	print(json.dumps(rep["best_val"]))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
