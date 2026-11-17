"""Supervised cross detector on the heuristic-inspection crops: wide_resnet50_2 (layer2-3) + two heads.

Run with the anomaly venv's python. The backbone is PatchCore's feature extractor (``timm`` wide_resnet50_2, layer2 and
layer3, ImageNet weights); ``--mode`` chooses what is trained on top of it:

* ``head``: both heads only (frozen features),
* ``lora``: LoRA adapters (rank ``--rank``) on every convolution of layer2 and layer3 plus the heads,
* ``full``: layer2 and layer3 weights plus the heads.

The two heads read the concatenated layer2 + layer3 feature map (stride 8):

* image head: global max + mean pooling -> linear -> 2-way cross-entropy (normal / defective),
* patch head: 1x1 convolution -> per-cell cross logit, supervised with the rendered defect masks (BCE + Dice).

Data: ``--data`` (normal and defective TEST layouts, crops of ``render_dataset.py``) and ``--data-defect`` (the
defective TRAIN layouts of ``render_dataset.py --only-defective-train``, with ``masks/``). Training uses normal train
crops and ``abnormal`` crops (>= 25 visible defect pixels) of the defective train layouts; ``hidden_defect`` crops are
never used. Validation and the threshold follow the PatchCore protocol: layouts held out of training, per camera the
``--val-quantile`` quantile of the validation normal scores. Test metrics are ``train_eval.evaluate`` on the same test
crops, so reports compare one to one with PatchCore.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import timm
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import nn
from train_eval import IMAGENET_MEAN, IMAGENET_STD, evaluate, in_real_top_crop, load_rows

LAYERS = ("layer2", "layer3")


class LoRAConv2d(nn.Module):
	"""Frozen convolution plus a trainable low-rank update ``scale * B(A(x))`` (A: same kernel / stride, B: 1x1, zero init)."""

	def __init__(self, base: nn.Conv2d, rank: int, alpha: float) -> None:
		super().__init__()
		if base.groups != 1:
			raise ValueError("LoRAConv2d supports groups=1 only")
		self.base = base
		for p in base.parameters():
			p.requires_grad = False
		self.a = nn.Conv2d(
			base.in_channels, rank, base.kernel_size, base.stride, base.padding, base.dilation, bias=False
		)
		self.b = nn.Conv2d(rank, base.out_channels, 1, bias=False)
		nn.init.kaiming_uniform_(self.a.weight, a=5**0.5)
		nn.init.zeros_(self.b.weight)
		self.scale = alpha / rank

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.base(x) + self.scale * self.b(self.a(x))


def add_lora(module: nn.Module, rank: int, alpha: float) -> int:
	"""Replace every Conv2d below ``module`` by a LoRAConv2d; returns how many."""
	n = 0
	for name, child in list(module.named_children()):
		if isinstance(child, nn.Conv2d):
			setattr(module, name, LoRAConv2d(child, rank, alpha))
			n += 1
		else:
			n += add_lora(child, rank, alpha)
	return n


class CrossNet(nn.Module):
	def __init__(self, mode: str, rank: int, alpha: float, pretrained: bool = True) -> None:
		super().__init__()
		self.backbone = timm.create_model(
			"wide_resnet50_2", pretrained=pretrained, features_only=True, out_indices=(2, 3)
		)
		for p in self.backbone.parameters():
			p.requires_grad = False
		self.n_lora = 0
		if mode == "lora":
			for name in LAYERS:
				self.n_lora += add_lora(getattr(self.backbone, name), rank, alpha)
		elif mode == "full":
			for name in LAYERS:
				for p in getattr(self.backbone, name).parameters():
					p.requires_grad = True
		elif mode != "head":
			raise ValueError(mode)
		self.reduce = nn.Sequential(nn.Conv2d(1536, 256, 1), nn.GroupNorm(8, 256), nn.ReLU(inplace=True))
		self.cls = nn.Linear(512, 2)
		self.seg = nn.Conv2d(256, 1, 1)

	def train(self, mode: bool = True) -> CrossNet:
		super().train(mode)
		for m in self.backbone.modules():  # frozen batch-norm statistics
			if isinstance(m, nn.BatchNorm2d):
				m.eval()
		return self

	def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		f2, f3 = self.backbone(x)
		f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
		z = self.reduce(torch.cat([f2, f3], dim=1))
		pooled = torch.cat([z.amax(dim=(2, 3)), z.mean(dim=(2, 3))], dim=1)
		return self.cls(pooled), self.seg(z)[:, 0]


def read_rows(root: Path, defect: bool, real_top: bool) -> list[dict]:
	rows = load_rows(root)
	if real_top:
		rows = [r for r in rows if in_real_top_crop(r)]
	for r in rows:
		r["_root"] = str(root)
		r["_has_mask"] = defect
	return rows


def split_layouts(rows: list[dict], idx: list[int], frac: float, seed: int) -> tuple[list[int], list[int]]:
	layouts = sorted({rows[i]["layout_index"] for i in idx})
	rng = np.random.default_rng(seed)
	val_l = {int(v) for v in rng.choice(layouts, size=max(1, int(len(layouts) * frac)), replace=False)}
	return [i for i in idx if rows[i]["layout_index"] not in val_l], [
		i for i in idx if rows[i]["layout_index"] in val_l
	]


def load_tensors(rows: list[dict], size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
	imgs = np.empty((len(rows), size, size, 3), np.uint8)
	masks = np.zeros((len(rows), size, size), np.uint8)
	for k, r in enumerate(rows):
		path = Path(r["_root"]) / r["path"]
		imgs[k] = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
		if r["_has_mask"] and r["label"] == "abnormal":
			masks[k] = cv2.imread(str(Path(r["_root"]) / "masks" / path.name), cv2.IMREAD_GRAYSCALE)
	return torch.from_numpy(imgs).to(device), torch.from_numpy(masks).to(device)


def prepare(x: torch.Tensor, input_size: int) -> torch.Tensor:
	"""uint8 (B, H, W, 3) -> normalised float (B, 3, S, S)."""
	x = x.permute(0, 3, 1, 2).float() / 255.0
	if x.shape[-1] != input_size:
		x = F.interpolate(x, size=(input_size, input_size), mode="bilinear", align_corners=False)
	mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
	std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
	return (x - mean) / std


def augment(x: torch.Tensor, m: torch.Tensor, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
	"""Flips, 90-degree rotations, brightness / contrast jitter, applied identically to image and mask."""
	out_x, out_m = [], []
	for i in range(x.shape[0]):
		xi, mi = x[i], m[i]
		k = int(torch.randint(0, 4, (1,), generator=gen))
		xi, mi = torch.rot90(xi, k, (0, 1)), torch.rot90(mi, k, (0, 1))
		if torch.rand(1, generator=gen) < 0.5:
			xi, mi = xi.flip(1), mi.flip(1)
		out_x.append(xi)
		out_m.append(mi)
	x, m = torch.stack(out_x), torch.stack(out_m)
	xf = x.float()
	b = 1.0 + 0.4 * (torch.rand(x.shape[0], 1, 1, 1, generator=gen).to(x.device) - 0.5)
	c = 1.0 + 0.4 * (torch.rand(x.shape[0], 1, 1, 1, generator=gen).to(x.device) - 0.5)
	mu = xf.mean(dim=(1, 2, 3), keepdim=True)
	return ((xf - mu) * c + mu * b).clamp(0, 255).to(torch.uint8), m


def losses(
	net: CrossNet, x: torch.Tensor, y: torch.Tensor, m: torch.Tensor, input_size: int, lam: float
) -> tuple[torch.Tensor, dict[str, float]]:
	logits, seg = net(prepare(x, input_size))
	ce = F.cross_entropy(logits.float(), y)
	target = F.max_pool2d(F.interpolate(m[:, None].float() / 255.0, size=(input_size, input_size)), 8)[:, 0]
	target = F.max_pool2d(target[:, None], 3, 1, 1)[:, 0].clamp(0, 1)
	seg = seg.float()
	bce = F.binary_cross_entropy_with_logits(seg, target, pos_weight=torch.tensor(8.0, device=seg.device))
	p = torch.sigmoid(seg)
	dice = 1 - (2 * (p * target).sum() + 1) / (p.sum() + target.sum() + 1)
	return ce + lam * (bce + dice), {"ce": float(ce), "bce": float(bce), "dice": float(dice)}


@torch.no_grad()
def score(net: CrossNet, imgs: torch.Tensor, input_size: int, batch: int) -> tuple[np.ndarray, np.ndarray]:
	"""(image-head score = logit gap, patch-head score = max cell logit) per image."""
	net.eval()
	s_cls, s_seg = [], []
	for k in range(0, len(imgs), batch):
		with torch.autocast("cuda", dtype=torch.bfloat16):
			logits, seg = net(prepare(imgs[k : k + batch], input_size))
		logits, seg = logits.float(), seg.float()
		s_cls.append((logits[:, 1] - logits[:, 0]).cpu().numpy())
		s_seg.append(seg.amax(dim=(1, 2)).cpu().numpy())
	return np.concatenate(s_cls), np.concatenate(s_seg)


def auroc(neg: np.ndarray, pos: np.ndarray) -> float:
	if len(neg) == 0 or len(pos) == 0:
		return float("nan")
	return float(roc_auc_score(np.r_[np.zeros(len(neg)), np.ones(len(pos))], np.r_[neg, pos]))


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	ap.add_argument("--data", type=Path, required=True, help="normal + test crops (dataset-v4 or dataset-v4-isaac)")
	ap.add_argument("--data-defect", type=Path, required=True, help="defective train layouts, with masks/")
	ap.add_argument("--mode", choices=["head", "lora", "full"], default="lora")
	ap.add_argument("--rank", type=int, default=8)
	ap.add_argument("--alpha", type=float, default=16.0)
	ap.add_argument("--input-size", type=int, default=256)
	ap.add_argument("--steps", type=int, default=3000)
	ap.add_argument("--eval-every", type=int, default=500)
	ap.add_argument("--batch", type=int, default=32)
	ap.add_argument("--lr", type=float, default=3e-4)
	ap.add_argument("--lam", type=float, default=1.0, help="weight of the patch head loss")
	ap.add_argument("--val-fraction", type=float, default=0.15)
	ap.add_argument("--val-quantile", type=float, default=0.99)
	ap.add_argument("--real-top-crop", action="store_true")
	ap.add_argument("--seed", type=int, default=0)
	ap.add_argument("--tag", required=True)
	ap.add_argument("--report-dir", type=Path, required=True)
	a = ap.parse_args()
	torch.manual_seed(a.seed)
	device = torch.device("cuda")
	t0 = time.perf_counter()
	normal_rows = read_rows(a.data, False, a.real_top_crop)
	defect_rows = [r for r in read_rows(a.data_defect, True, a.real_top_crop) if r["label"] == "abnormal"]
	rows = normal_rows + defect_rows
	n_norm = len(normal_rows)
	idx_norm = [i for i in range(n_norm) if rows[i]["split"] == "train" and rows[i]["label"] == "normal"]
	idx_def = list(range(n_norm, len(rows)))
	fit_n, val_n = split_layouts(rows, idx_norm, a.val_fraction, a.seed)
	fit_d, val_d = split_layouts(rows, idx_def, a.val_fraction, a.seed + 1)
	test_idx = [i for i in range(n_norm) if rows[i]["split"] == "test"]
	need = sorted(set(fit_n + fit_d + val_n + val_d + test_idx))
	rows = [rows[i] for i in need]
	remap = {old: new for new, old in enumerate(need)}
	fit_n, fit_d, val_n, val_d, test_idx = (
		[remap[i] for i in group] for group in (fit_n, fit_d, val_n, val_d, test_idx)
	)
	imgs, masks = load_tensors(rows, 256, device)
	print(
		f"loaded {len(rows)} crops ({time.perf_counter() - t0:.0f}s): fit normal {len(fit_n)} defective {len(fit_d)}, "
		f"val {len(val_n)}/{len(val_d)}, test {len(test_idx)}",
		flush=True,
	)
	net = CrossNet(a.mode, a.rank, a.alpha).to(device).to(memory_format=torch.channels_last)
	params = [p for p in net.parameters() if p.requires_grad]
	print(f"mode {a.mode}: {sum(p.numel() for p in params) / 1e6:.2f} M trainable, {net.n_lora} LoRA convs", flush=True)
	opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4)
	sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.1)
	gen = torch.Generator().manual_seed(a.seed)
	fit_n_t, fit_d_t = torch.tensor(fit_n, device=device), torch.tensor(fit_d, device=device)
	val_idx = val_n + val_d
	val_y = np.array([0] * len(val_n) + [1] * len(val_d))
	best = (-1.0, None)
	half = a.batch // 2
	for step in range(1, a.steps + 1):
		net.train()
		ids = torch.cat(
			[
				fit_n_t[torch.randint(len(fit_n_t), (half,), generator=gen).to(device)],
				fit_d_t[torch.randint(len(fit_d_t), (a.batch - half,), generator=gen).to(device)],
			]
		)
		x, m = augment(imgs[ids], masks[ids], gen)
		y = torch.cat([torch.zeros(half), torch.ones(a.batch - half)]).long().to(device)
		with torch.autocast("cuda", dtype=torch.bfloat16):
			loss, parts = losses(net, x, y, m, a.input_size, a.lam)
		opt.zero_grad(set_to_none=True)
		loss.backward()
		opt.step()
		sched.step()
		if step % a.eval_every == 0 or step == a.steps:
			s_cls, s_seg = score(net, imgs[torch.tensor(val_idx, device=device)], a.input_size, 64)
			v = (auroc(s_cls[val_y == 0], s_cls[val_y == 1]) + auroc(s_seg[val_y == 0], s_seg[val_y == 1])) / 2
			print(f"step {step} loss {float(loss):.3f} {parts} val AUROC cls/seg mean {v:.4f}", flush=True)
			if v > best[0]:
				best = (v, {k: t.detach().clone() for k, t in net.state_dict().items()})
	net.load_state_dict(best[1])
	s_cls_val, s_seg_val = score(net, imgs[torch.tensor(val_n, device=device)], a.input_size, 64)
	s_cls_all, s_seg_all = score(net, imgs, a.input_size, 64)
	cams = np.array([r["camera"] for r in rows])
	report: dict[str, Any] = {
		"mode": a.mode,
		"rank": a.rank,
		"input_size": a.input_size,
		"steps": a.steps,
		"real_top_crop": a.real_top_crop,
		"val_quantile": a.val_quantile,
		"best_val_auroc": best[0],
		"data": {"normal": str(a.data), "defect": str(a.data_defect)},
		"n": {"fit_normal": len(fit_n), "fit_defective": len(fit_d), "val": len(val_idx), "test": len(test_idx)},
		"models": {},
	}
	val_cam = cams[np.array(val_n)]
	for head, all_scores, val_scores in (("cls", s_cls_all, s_cls_val), ("seg", s_seg_all, s_seg_val)):
		for scope in ("wrist_cam", "top", "pooled"):
			sel = [i for i in test_idx if scope == "pooled" or rows[i]["camera"] == scope]
			vsel = val_scores if scope == "pooled" else val_scores[val_cam == scope]
			thr = float(np.quantile(vsel, a.val_quantile))
			res = evaluate(rows, all_scores[np.array(sel)], sel, thr)
			res["threshold"] = thr
			report["models"][f"{head}/{scope}"] = res
			t = res["at_threshold"]
			print(
				f"{head}/{scope:9s} AUROC {res['auroc']:.3f} recall {t['recall']:.3f} fpr {t['fpr']:.3f} "
				f"bestF1 {res['best_f1_oracle']:.3f}",
				flush=True,
			)
	a.report_dir.mkdir(parents=True, exist_ok=True)
	(a.report_dir / f"report-{a.tag}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
	np.savez_compressed(
		a.report_dir / f"scores-{a.tag}.npz",
		paths=np.array([r["path"] for r in rows]),
		test_idx=np.array(test_idx),
		cls=s_cls_all,
		seg=s_seg_all,
	)
	torch.save(
		{k: v for k, v in net.state_dict().items() if "backbone" not in k or "lora" in k or ".a." in k or ".b." in k},
		a.report_dir / f"head-{a.tag}.pt",
	)
	print(f"done in {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
	main()
