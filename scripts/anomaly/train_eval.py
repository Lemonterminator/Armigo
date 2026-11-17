"""Train and evaluate image anomaly detectors (anomalib PaDiM and PatchCore) on the rendered brick crops.

Runs in the separate anomaly environment (Python 3.12, torch + anomalib; see ``requirements.txt``), not in the
repository's main venv. Reads ``metadata.jsonl`` written by ``render_dataset.py``.

* Training data: the ``normal`` crops of the train layouts, split again BY LAYOUT into a fit part and a validation
  part (``--val-fraction``). Detectors see normal images only.
* Threshold: the ``--val-quantile`` quantile of the validation normal scores (a target false-positive rate).
* Test: the test layouts' ``normal`` vs ``abnormal`` crops (image AUROC, and precision / recall / F1 / FPR at the
  threshold). ``hidden_defect`` crops (defective brick, defect not in view) are scored separately: they should
  look normal.
* Models: anomalib's ``PadimModel`` (resnet18, layer1-3, 100 features) and ``PatchcoreModel`` (wide_resnet50_2,
  layer2-3, k-center-greedy coreset), each trained per camera (top, wrist_cam) and pooled over both. The PaDiM
  Gaussian is fitted with streaming sums and the PatchCore memory bank from a random subset of training patches, so
  the GPU footprint stays small; anomalib's own scoring (anomaly map max / nearest-neighbour score) is unchanged.

Writes ``report-v1.json`` and ``REPORT.md`` next to the dataset (``--report-dir``) and the best pooled model to
``models/best.pt`` + ``models/best.json`` for ``score.py``.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "results" / "anomaly" / "dataset-v1"
DEFAULT_REPORT = ROOT / "results" / "anomaly"
CAMERAS = ("top", "wrist_cam")
PX_BINS = ((25, 60), (60, 150), (150, None))
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

MODEL_CONFIGS: dict[str, dict[str, Any]] = {
	"padim": {"backbone": "resnet18", "layers": ["layer1", "layer2", "layer3"], "n_features": 100},
	"patchcore": {"backbone": "wide_resnet50_2", "layers": ["layer2", "layer3"], "num_neighbors": 9},
	# finer patches (stride 4 instead of 8) for the small defects
	"patchcore_fine": {"backbone": "wide_resnet50_2", "layers": ["layer1", "layer2"], "num_neighbors": 9},
}


# --- data --------------------------------------------------------------------------------------------------------


def load_rows(data_dir: Path) -> list[dict]:
	return [json.loads(line) for line in (data_dir / "metadata.jsonl").read_text(encoding="utf-8").splitlines() if line]


REAL_TOP_CROP = (96, 0, 512, 416)  # x, y, w, h of the recorded D415 colour view inside the 640x480 frame


def in_real_top_crop(row: dict) -> bool:
	"""True for wrist crops, and for top crops whose brick bbox (in the full 640x480 frame) lies entirely inside the
	512x416 region the real top camera records."""
	if row["camera"] != "top":
		return True
	cx, cy, cw, ch = REAL_TOP_CROP
	x, y, w, h = row["bbox"]
	return x >= cx and y >= cy and x + w <= cx + cw and y + h <= cy + ch


def load_images(data_dir: Path, rows: list[dict]) -> np.ndarray:
	"""(N, H, W, 3) uint8 RGB."""
	return np.stack([cv2.cvtColor(cv2.imread(str(data_dir / r["path"])), cv2.COLOR_BGR2RGB) for r in rows])


def to_tensor(images: np.ndarray, input_size: int, device: torch.device) -> torch.Tensor:
	"""uint8 RGB (N, H, W, 3) -> normalised float (N, 3, S, S) on ``device`` (bilinear resize)."""
	x = torch.from_numpy(images).to(device).permute(0, 3, 1, 2).float() / 255.0
	if x.shape[-1] != input_size:
		x = F.interpolate(x, size=(input_size, input_size), mode="bilinear", align_corners=False)
	mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
	std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
	return (x - mean) / std


def split_train_val(rows: list[dict], val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
	"""Indices of train-split normal rows, divided by layout into fit and validation parts."""
	idx = [i for i, r in enumerate(rows) if r["split"] == "train" and r["label"] == "normal"]
	layouts = sorted({rows[i]["layout_index"] for i in idx})
	rng = np.random.default_rng(seed)
	val_layouts = set(rng.choice(layouts, size=max(1, round(val_fraction * len(layouts))), replace=False).tolist())
	fit = [i for i in idx if rows[i]["layout_index"] not in val_layouts]
	val = [i for i in idx if rows[i]["layout_index"] in val_layouts]
	return fit, val


# --- models ------------------------------------------------------------------------------------------------------


def build_model(name: str, pre_trained: bool = True) -> torch.nn.Module:
	cfg = MODEL_CONFIGS[name]
	if name == "padim":
		from anomalib.models.image.padim.torch_model import PadimModel

		return PadimModel(
			backbone=cfg["backbone"], layers=cfg["layers"], pre_trained=pre_trained, n_features=cfg["n_features"]
		)
	from anomalib.models.image.patchcore.torch_model import PatchcoreModel  # "patchcore" and "patchcore_fine"

	return PatchcoreModel(
		layers=cfg["layers"], backbone=cfg["backbone"], pre_trained=pre_trained, num_neighbors=cfg["num_neighbors"]
	)


@torch.no_grad()
def fit_padim(model, images: np.ndarray, input_size: int, device: torch.device, batch: int) -> None:
	"""PaDiM's per-position Gaussian from streaming sums (same estimate as anomalib's ``MultiVariateGaussian``:
	unbiased covariance + 0.01 I + 1e-5 I), without holding every embedding in memory."""
	model.eval()
	s1 = s2 = None
	n = 0
	for k in range(0, len(images), batch):
		feats = model.feature_extractor(to_tensor(images[k : k + batch], input_size, device))
		emb = model.generate_embedding(feats)  # (B, C, H, W)
		b, c, h, w = emb.shape
		e = emb.reshape(b, c, h * w).permute(2, 0, 1).double()  # (HW, B, C)
		if s1 is None:
			s1 = torch.zeros(h * w, c, dtype=torch.float64, device=device)
			s2 = torch.zeros(h * w, c, c, dtype=torch.float64, device=device)
		s1 += e.sum(1)
		s2 += torch.bmm(e.transpose(1, 2), e)
		n += b
	mean = s1 / n  # (HW, C)
	cov = (s2 - n * mean.unsqueeze(2) * mean.unsqueeze(1)) / (n - 1)
	eye = torch.eye(cov.shape[-1], dtype=torch.float64, device=device)
	cov = cov + (0.01 + 1e-5) * eye
	model.gaussian.mean = mean.T.float().contiguous()  # (C, HW)
	model.gaussian.inv_covariance = torch.linalg.inv(cov).float()  # (HW, C, C)


@torch.no_grad()
def fit_patchcore(
	model, images: np.ndarray, input_size: int, device: torch.device, batch: int, max_patches: int, bank: int, seed: int
) -> None:
	"""PatchCore memory bank: a random subset of at most ``max_patches`` training patch embeddings, reduced to about
	``bank`` entries with anomalib's k-center-greedy coreset."""
	gen = torch.Generator().manual_seed(seed)
	model.train()
	per_img = None
	keep: list[torch.Tensor] = []
	for k in range(0, len(images), batch):
		x = to_tensor(images[k : k + batch], input_size, device)
		emb = model(x)  # (B * H * W, C), also appended to model.embedding_store
		model.embedding_store.clear()
		b = x.shape[0]
		if per_img is None:
			per_img = emb.shape[0] // b
		take = max(1, min(per_img, max_patches // max(len(images), 1)))
		e = emb.reshape(b, per_img, -1)
		sel = torch.stack([torch.randperm(per_img, generator=gen)[:take] for _ in range(b)]).to(device)
		keep.append(torch.gather(e, 1, sel.unsqueeze(-1).expand(-1, -1, e.shape[-1])).reshape(-1, e.shape[-1]).cpu())
	store = torch.cat(keep)
	model.embedding_store = [store.to(device)]
	model.subsample_embedding(sampling_ratio=min(1.0, bank / store.shape[0]))
	model.eval()


@torch.no_grad()
def score(model, images: np.ndarray, input_size: int, device: torch.device, batch: int) -> np.ndarray:
	model.eval()
	out = []
	for k in range(0, len(images), batch):
		res = model(to_tensor(images[k : k + batch], input_size, device))
		out.append(res.pred_score.reshape(-1).float().cpu().numpy())
	return np.concatenate(out) if out else np.zeros(0)


# --- metrics -----------------------------------------------------------------------------------------------------


def _auroc(neg: np.ndarray, pos: np.ndarray) -> float | None:
	if len(neg) == 0 or len(pos) == 0:
		return None
	y = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
	return round(float(roc_auc_score(y, np.r_[neg, pos])), 4)


def _at_threshold(neg: np.ndarray, pos: np.ndarray, thr: float) -> dict[str, float]:
	tp = int((pos > thr).sum())
	fp = int((neg > thr).sum())
	fn = len(pos) - tp
	prec = tp / max(tp + fp, 1)
	rec = tp / max(len(pos), 1)
	f1 = 2 * prec * rec / max(prec + rec, 1e-12)
	return {
		"precision": round(prec, 4),
		"recall": round(rec, 4),
		"f1": round(f1, 4),
		"fpr": round(fp / max(len(neg), 1), 4),
		"tp": tp,
		"fp": fp,
		"fn": fn,
	}


def _best_f1(neg: np.ndarray, pos: np.ndarray) -> float | None:
	if len(neg) == 0 or len(pos) == 0:
		return None
	return max(_at_threshold(neg, pos, t)["f1"] for t in np.unique(np.r_[neg, pos]) - 1e-9)


def evaluate(rows: list[dict], scores: np.ndarray, test_idx: list[int], thr: float) -> dict[str, Any]:
	"""Metrics over the test crops ``test_idx`` (rows and scores indexed alike)."""
	sc = {i: float(scores[k]) for k, i in enumerate(test_idx)}
	normal = [i for i in test_idx if rows[i]["label"] == "normal"]
	abnormal = [i for i in test_idx if rows[i]["label"] == "abnormal"]
	hidden = [i for i in test_idx if rows[i]["label"] == "hidden_defect"]
	neg = np.array([sc[i] for i in normal])
	pos = np.array([sc[i] for i in abnormal])

	def strat(key, values) -> dict[str, Any]:
		out = {}
		for v in values:
			p = np.array([sc[i] for i in abnormal if key(rows[i]) == v])
			out[str(v)] = {
				"n_abnormal": len(p),
				"auroc": _auroc(neg, p),
				"recall": _at_threshold(neg, p, thr)["recall"],
			}
		return out

	def bin_of(r) -> str:
		for lo, hi in PX_BINS:
			if r["defect_px"] >= lo and (hi is None or r["defect_px"] < hi):
				return f"{lo}-{hi}" if hi else f"{lo}+"
		return "<25"

	by_cam = {}
	for cam in CAMERAS:
		n_c = np.array([sc[i] for i in normal if rows[i]["camera"] == cam])
		p_c = np.array([sc[i] for i in abnormal if rows[i]["camera"] == cam])
		if len(n_c) or len(p_c):
			by_cam[cam] = {"n_normal": len(n_c), "n_abnormal": len(p_c), "auroc": _auroc(n_c, p_c)}
	h = np.array([sc[i] for i in hidden])
	by_stage = {}
	for stage in sorted({rows[i]["stage"] for i in test_idx if "stage" in rows[i]}):
		n_s = np.array([sc[i] for i in normal if rows[i].get("stage") == stage])
		p_s = np.array([sc[i] for i in abnormal if rows[i].get("stage") == stage])
		by_stage[stage] = {
			"n_normal": len(n_s),
			"n_abnormal": len(p_s),
			"auroc": _auroc(n_s, p_s),
			"recall": _at_threshold(n_s, p_s, thr)["recall"],
		}
	out = {
		"n_normal": len(neg),
		"n_abnormal": len(pos),
		"n_hidden_defect": len(h),
		"auroc": _auroc(neg, pos),
		"threshold": round(thr, 6),
		"at_threshold": _at_threshold(neg, pos, thr),
		"best_f1_oracle": _best_f1(neg, pos),
		"by_kind": strat(lambda r: r["defect_kind"], sorted({rows[i]["defect_kind"] for i in abnormal})),
		"by_face": strat(lambda r: r["defect_face"], sorted({rows[i]["defect_face"] for i in abnormal})),
		"by_defect_px": strat(bin_of, [f"{lo}-{hi}" if hi else f"{lo}+" for lo, hi in PX_BINS]),
		"by_camera": by_cam,
		"by_virtual": {
			str(v): {
				"auroc": _auroc(
					np.array([sc[i] for i in normal if rows[i]["virtual"] == v]),
					np.array([sc[i] for i in abnormal if rows[i]["virtual"] == v]),
				)
			}
			for v in (False, True)
		},
		"hidden_defect": {
			"auroc_vs_normal": _auroc(neg, h),
			"frac_above_threshold": round(float((h > thr).mean()), 4) if len(h) else None,
			"normal_frac_above_threshold": round(float((neg > thr).mean()), 4) if len(neg) else None,
			"quantiles": [round(float(q), 4) for q in np.quantile(h, [0.05, 0.25, 0.5, 0.75, 0.95])] if len(h) else [],
			"normal_quantiles": [round(float(q), 4) for q in np.quantile(neg, [0.05, 0.25, 0.5, 0.75, 0.95])],
		},
	}
	if by_stage:
		out["by_stage"] = by_stage
	return out


# --- main --------------------------------------------------------------------------------------------------------


def export_model(
	model, name: str, scope: str, args, crop_size: int, threshold: float, res: dict[str, Any], out_dir: Path
) -> None:
	"""Save ``best.pt`` + ``best.json`` for one trained model to ``out_dir`` (format read by ``score.py``)."""
	out_dir.mkdir(parents=True, exist_ok=True)
	torch.save(model.state_dict(), out_dir / "best.pt")
	meta = {
		"model": name,
		"config": MODEL_CONFIGS[name],
		"scope": scope,
		"input_size": args.input_size,
		"real_top_crop": bool(args.real_top_crop),
		"crop_size": crop_size,
		"threshold": threshold,
		"val_quantile": args.val_quantile,
		"test_auroc": res["auroc"],
		"dataset": str(args.data),
	}
	(out_dir / "best.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def run(args) -> dict[str, Any]:
	device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
	rows = load_rows(args.data)
	n_all = len(rows)
	if args.real_top_crop:
		rows = [r for r in rows if in_real_top_crop(r)]
		print(f"real top-camera crop: kept {len(rows)} of {n_all} crops", flush=True)
	t0 = time.perf_counter()
	images = load_images(args.data, rows)
	print(f"loaded {len(rows)} crops in {time.perf_counter() - t0:.1f}s on {device}", flush=True)
	fit_all, val_all = split_train_val(rows, args.val_fraction, args.seed)
	test_all = [i for i, r in enumerate(rows) if r["split"] == "test"]
	report: dict[str, Any] = {
		"device": str(device),
		"torch": torch.__version__,
		"input_size": args.input_size,
		"real_top_crop": bool(args.real_top_crop),
		"n_crops_before_filter": n_all,
		"val_quantile": args.val_quantile,
		"data": {
			"n_fit": len(fit_all),
			"n_val": len(val_all),
			"n_test": len(test_all),
			"fit_layouts": len({rows[i]["layout_index"] for i in fit_all}),
			"val_layouts": len({rows[i]["layout_index"] for i in val_all}),
			"test_layouts": len({rows[i]["layout_index"] for i in test_all}),
		},
		"models": {},
		"configs": MODEL_CONFIGS,
	}
	import anomalib

	report["anomalib"] = anomalib.__version__
	scopes = {"top": ("top",), "wrist_cam": ("wrist_cam",), "pooled": CAMERAS}
	per_crop: dict[str, dict[str, float]] = defaultdict(dict)
	best = None
	args.models_dir.mkdir(parents=True, exist_ok=True)
	export_scopes = {key.strip() for key in args.export_scopes.split(",") if key.strip()}
	for name in args.models:
		for scope, cams in scopes.items():
			fit = [i for i in fit_all if rows[i]["camera"] in cams]
			val = [i for i in val_all if rows[i]["camera"] in cams]
			test = [i for i in test_all if rows[i]["camera"] in cams]
			t1 = time.perf_counter()
			torch.manual_seed(args.seed)
			model = build_model(name).to(device)
			if name == "padim":
				fit_padim(model, images[fit], args.input_size, device, args.batch)
			else:
				fit_patchcore(
					model, images[fit], args.input_size, device, args.batch, args.max_patches, args.bank, args.seed
				)
			t_fit = time.perf_counter() - t1
			# PatchCore scores every patch against the whole bank: keep the (patches x bank) distance matrix small
			sb = args.batch if name == "padim" else max(1, args.batch // (4 if name == "patchcore" else 16))
			val_s = score(model, images[val], args.input_size, device, sb)
			thr = float(np.quantile(val_s, args.val_quantile))
			test_s = score(model, images[test], args.input_size, device, sb)
			res = evaluate(rows, test_s, test, thr)
			for i, v in zip(test, test_s, strict=True):
				per_crop[rows[i]["path"]][f"{name}/{scope}"] = round(float(v), 5)
			res.update(n_fit=len(fit), n_val=len(val), fit_seconds=round(t_fit, 1))
			res["seconds"] = round(time.perf_counter() - t1, 1)
			if device.type == "cuda":
				res["gpu_peak_mb"] = round(torch.cuda.max_memory_allocated() / 2**20)
				torch.cuda.reset_peak_memory_stats()
			key = f"{name}/{scope}"
			report["models"][key] = res
			print(
				f"{key:20s} AUROC={res['auroc']} F1@thr={res['at_threshold']['f1']} "
				f"kinds={ {k: v['auroc'] for k, v in res['by_kind'].items()} } ({res['seconds']}s)",
				flush=True,
			)
			crop_size = int(images.shape[1])
			if scope == "pooled" and (best is None or (res["auroc"] or 0) > best[1]):
				best = (key, res["auroc"] or 0)
				export_model(model, name, scope, args, crop_size, thr, res, args.models_dir)
			if key in export_scopes:
				export_model(model, name, scope, args, crop_size, thr, res, args.models_dir / f"{name}_{scope}")
			del model
			if device.type == "cuda":
				torch.cuda.empty_cache()
	report["best_pooled_model"] = best[0] if best else None
	report["_per_crop_scores"] = per_crop
	return report


def _fmt(v) -> str:
	return "-" if v is None else f"{v:.3f}"


def write_markdown(report: dict[str, Any], dataset_summary: dict[str, Any] | None, path: Path, notes: str = "") -> None:
	models = report["models"]
	keys = list(models)
	lines = [
		"# Brick surface anomaly detection on simulated crops (v1)",
		"",
		(
			"Simulation only. Crops of the held brick at the expert's first show hold and at random virtual show "
			"poses (MuJoCo, `scripts/anomaly/render_dataset.py`), detectors from anomalib trained on normal crops "
			"only (`scripts/anomaly/train_eval.py`). Full numbers: `report-v1.json`."
		),
		"",
		(
			f"Device: {report['device']}, torch {report['torch']}, anomalib {report.get('anomalib', '?')}; "
			f"input {report['input_size']} px; threshold = {report['val_quantile']:.2f} quantile of validation "
			"normal scores."
		),
		"",
		(
			f"Data (crops): fit {report['data']['n_fit']} / val {report['data']['n_val']} normal "
			f"({report['data']['fit_layouts']} / {report['data']['val_layouts']} layouts); "
			f"test {report['data']['n_test']} ({report['data']['test_layouts']} layouts)."
		),
		"",
	]
	if dataset_summary:
		lines += [
			"Test crops by label: " + ", ".join(f"{k}={v}" for k, v in dataset_summary["by_split_label"].items()),
			"",
		]
	lines += [
		"## Overall (test: normal vs abnormal)",
		"",
		"| model / scope | n normal | n abnormal | AUROC | F1 @thr | precision | recall | FPR | best F1 (oracle thr) |",
		"|---|---|---|---|---|---|---|---|---|",
	]
	for k in keys:
		m = models[k]
		t = m["at_threshold"]
		lines.append(
			f"| {k} | {m['n_normal']} | {m['n_abnormal']} | {_fmt(m['auroc'])} | {_fmt(t['f1'])} | "
			f"{_fmt(t['precision'])} | {_fmt(t['recall'])} | {_fmt(t['fpr'])} | {_fmt(m['best_f1_oracle'])} |"
		)

	def table(title: str, field: str) -> None:
		cols = sorted({c for k in keys for c in models[k][field]})
		lines.extend(["", f"## AUROC by {title} (abnormal of that stratum vs all test normals)", ""])
		lines.append("| model / scope | " + " | ".join(cols) + " |")
		lines.append("|---" * (len(cols) + 1) + "|")
		for k in keys:
			cells = []
			for c in cols:
				s = models[k][field].get(c)
				cells.append("-" if s is None else f"{_fmt(s.get('auroc'))} (n={s.get('n_abnormal', '?')})")
			lines.append(f"| {k} | " + " | ".join(cells) + " |")

	table("defect kind", "by_kind")
	table("defect face (0 +x, 1 -x, 2 +y, 3 -y, 4 top, 5 bottom)", "by_face")
	table("visible defect pixels", "by_defect_px")
	table("camera", "by_camera")
	stages = sorted({c for k in keys for c in models[k].get("by_stage", {})})
	if stages:
		lines.extend(["", "## AUROC by stage (abnormal of that stage vs normals of that stage)", ""])
		lines.append("| model / scope | " + " | ".join(stages) + " |")
		lines.append("|---" * (len(stages) + 1) + "|")
		for k in keys:
			cells = []
			for s in stages:
				st = models[k].get("by_stage", {}).get(s)
				if st is None:
					cells.append("-")
				else:
					cells.append(
						f"{_fmt(st.get('auroc'))} (n_abn={st.get('n_abnormal', '?')}, n_norm={st.get('n_normal', '?')})"
					)
			lines.append(f"| {k} | " + " | ".join(cells) + " |")
	lines += [
		"",
		"## Hidden defects (defective brick, < 25 defect pixels in view)",
		"",
		"| model / scope | n | AUROC vs normal | frac above thr | normal frac above thr | median (normal median) |",
		"|---|---|---|---|---|---|",
	]
	for k in keys:
		h = models[k]["hidden_defect"]
		med = h["quantiles"][2] if h["quantiles"] else None
		lines.append(
			f"| {k} | {models[k]['n_hidden_defect']} | {_fmt(h['auroc_vs_normal'])} | {_fmt(h['frac_above_threshold'])} | "
			f"{_fmt(h['normal_frac_above_threshold'])} | {_fmt(med)} ({_fmt(h['normal_quantiles'][2])}) |"
		)
	lines += ["", f"Exported model: `{report.get('best_pooled_model')}` -> `models/best.pt` (see `score.py`).", ""]
	if notes:
		lines += ["", notes.strip(), ""]
	path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
	ap.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT)
	ap.add_argument("--models-dir", type=Path, default=DEFAULT_REPORT / "models")
	ap.add_argument(
		"--models", nargs="+", default=["padim", "patchcore", "patchcore_fine"], choices=sorted(MODEL_CONFIGS)
	)
	ap.add_argument("--input-size", type=int, default=224)
	ap.add_argument("--batch", type=int, default=32)
	ap.add_argument("--val-fraction", type=float, default=0.15)
	ap.add_argument("--val-quantile", type=float, default=0.95)
	ap.add_argument("--max-patches", type=int, default=200_000, help="PatchCore training patches kept before coreset")
	ap.add_argument("--bank", type=int, default=20_000, help="PatchCore memory bank size after coreset")
	ap.add_argument("--seed", type=int, default=0)
	ap.add_argument("--cpu", action="store_true")
	ap.add_argument(
		"--export-scopes",
		default="",
		help="comma list of model/scope keys to export individually, e.g. patchcore/top,patchcore/wrist_cam",
	)
	ap.add_argument(
		"--real-top-crop",
		action="store_true",
		help="drop top crops whose brick is not fully inside the real 512x416 D415 region (train and test)",
	)
	ap.add_argument("--tag", default="v1")
	ap.add_argument("--notes", type=Path, default=None, help="markdown appended to REPORT.md (caveats)")
	args = ap.parse_args()
	report = run(args)
	per_crop = report.pop("_per_crop_scores")
	summary_path = args.data / "summary.json"
	ds = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None
	report["dataset_summary"] = ds
	args.report_dir.mkdir(parents=True, exist_ok=True)
	(args.report_dir / f"report-{args.tag}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
	(args.report_dir / f"scores-{args.tag}.jsonl").write_text(
		"".join(json.dumps({"path": p, **v}) + "\n" for p, v in per_crop.items()), encoding="utf-8"
	)
	notes = args.notes.read_text(encoding="utf-8") if args.notes else ""
	write_markdown(report, ds, args.report_dir / "REPORT.md", notes)
	print(f"wrote {args.report_dir / f'report-{args.tag}.json'}")


if __name__ == "__main__":
	main()
