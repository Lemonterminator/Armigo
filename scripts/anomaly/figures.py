"""Illustrative figures for the anomaly v1 dataset: example crops with PatchCore scores, and score distributions.

Run with the anomaly env:  E:/venvs/armigo-anomaly/Scripts/python.exe scripts/anomaly/figures.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[2] / "results" / "anomaly"
DATA = ROOT / "dataset-v1"
OUT = ROOT / "figures"
KINDS = ("chip", "cross", "discolor", "scuff", "tape")
CAMS = ("wrist_cam", "top")
CAM_CN = {"wrist_cam": "wrist camera", "top": "top camera"}


def load():
	meta = {r["path"]: r for r in map(json.loads, (DATA / "metadata.jsonl").read_text(encoding="utf-8").splitlines())}
	scores = {
		r["path"]: r for r in map(json.loads, (ROOT / "scores-v1.jsonl").read_text(encoding="utf-8").splitlines())
	}
	report = json.loads((ROOT / "report-v1.json").read_text(encoding="utf-8"))
	thr = {c: report["models"][f"patchcore/{c}"]["threshold"] for c in CAMS}
	rows = []
	for p, s in scores.items():
		m = meta.get(p)
		if m is None or m["split"] != "test":
			continue
		sc = s.get(f"patchcore/{m['camera']}")
		if sc is not None:
			rows.append({**m, "score": sc})
	return rows, thr


def examples(rows, thr, n=3, seed=0):
	rng = random.Random(seed)
	groups = [("normal", None)] + [("abnormal", k) for k in KINDS] + [("hidden_defect", None)]
	fig, axes = plt.subplots(len(groups), 2 * n, figsize=(2 * n * 1.7, len(groups) * 1.9))
	for i, (label, kind) in enumerate(groups):
		for j, cam in enumerate(CAMS):
			pool = [
				r
				for r in rows
				if r["label"] == label and r["camera"] == cam and (kind is None or r["defect_kind"] == kind)
			]
			pick = rng.sample(pool, min(n, len(pool)))
			for k in range(n):
				ax = axes[i, j * n + k]
				ax.set_xticks([])
				ax.set_yticks([])
				if k >= len(pick):
					ax.axis("off")
					continue
				r = pick[k]
				ax.imshow(Image.open(DATA / r["path"]))
				flagged = r["score"] >= thr[cam]
				truth_bad = label == "abnormal"
				ok = flagged == truth_bad if label != "hidden_defect" else None
				color = "gray" if ok is None else ("tab:green" if ok else "tab:red")
				for s in ax.spines.values():
					s.set_edgecolor(color)
					s.set_linewidth(3)
				ax.set_title(f"{r['score']:.0f} {'FLAG' if flagged else 'ok'}", fontsize=8, color=color)
				if i == 0 and k == 0:
					ax.text(0, -0.35, CAM_CN[cam], transform=ax.transAxes, fontsize=10, weight="bold")
			name = kind if kind else label.replace("_", " ")
			axes[i, 0].set_ylabel(name, fontsize=10)
	fig.suptitle(
		"Test crops, PatchCore per-camera score (title) vs threshold\n"
		"green = correct, red = wrong, gray = defect on a hidden face (no visible evidence)",
		fontsize=10,
	)
	fig.tight_layout()
	fig.savefig(OUT / "examples.png", dpi=130)
	plt.close(fig)


def distributions(rows, thr):
	fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
	for ax, cam in zip(axes, CAMS):
		sub = [r for r in rows if r["camera"] == cam]
		lo = min(r["score"] for r in sub)
		hi = max(r["score"] for r in sub)
		bins = [lo + (hi - lo) * t / 50 for t in range(51)]
		for label, col in (("normal", "tab:blue"), ("hidden_defect", "tab:gray"), ("abnormal", "tab:red")):
			v = [r["score"] for r in sub if r["label"] == label]
			ax.hist(v, bins=bins, density=True, alpha=0.45, color=col, label=f"{label} (n={len(v)})")
		ax.axvline(thr[cam], color="k", ls="--", lw=1, label="threshold (val normal q95)")
		ax.set_title(f"{CAM_CN[cam]} — PatchCore anomaly score")
		ax.set_xlabel("score")
		ax.legend(fontsize=8)
	fig.tight_layout()
	fig.savefig(OUT / "score_distributions.png", dpi=130)
	plt.close(fig)


def recall_by_kind(rows, thr):
	fig, ax = plt.subplots(figsize=(7, 3.4))
	w = 0.38
	for j, cam in enumerate(CAMS):
		rec = []
		for k in KINDS:
			v = [r for r in rows if r["camera"] == cam and r["label"] == "abnormal" and r["defect_kind"] == k]
			rec.append(sum(r["score"] >= thr[cam] for r in v) / len(v))
		xs = [i + (j - 0.5) * w for i in range(len(KINDS))]
		bars = ax.bar(xs, rec, w, label=CAM_CN[cam])
		ax.bar_label(bars, fmt="%.2f", fontsize=8)
	ax.set_xticks(range(len(KINDS)), KINDS)
	ax.set_ylim(0, 1.05)
	ax.set_ylabel("recall @ threshold")
	ax.set_title("Detection rate per defect kind (FPR ≈ 5% by construction)")
	ax.legend(fontsize=8)
	fig.tight_layout()
	fig.savefig(OUT / "recall_by_kind.png", dpi=130)
	plt.close(fig)


if __name__ == "__main__":
	OUT.mkdir(exist_ok=True)
	rows, thr = load()
	examples(rows, thr)
	distributions(rows, thr)
	recall_by_kind(rows, thr)
	print("wrote", sorted(p.name for p in OUT.iterdir()))
