"""Agent-view alarm rates of a scored bottom-view set (``render_bottom.py`` + ``infer_sup.py --holdout``).

Each crop's agent score is ``u`` = max over heads of ``(raw - t) / (t - median)`` with the top-camera statistics of a
scorer spec (``export_sup.py``), the score the six-face agent compares with 0. Prints, per brick size and label, the
crops at or above 0 and the AUROC of ``u``; only the ``--split`` layouts (default ``test``) are counted. Numpy only::

    python scripts/anomaly/eval_bottom_scores.py results/anomaly/dataset-v7bot SCORES.npz SPEC.json [--json OUT]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def auroc(neg: np.ndarray, pos: np.ndarray) -> float:
	if not len(neg) or not len(pos):
		return float("nan")
	allv = np.concatenate([neg, pos])
	ranks = allv.argsort().argsort() + 1.0
	return float((ranks[len(neg) :].sum() - len(pos) * (len(pos) + 1) / 2) / (len(neg) * len(pos)))


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	ap.add_argument("data", type=Path)
	ap.add_argument("scores", type=Path)
	ap.add_argument("spec", type=Path)
	ap.add_argument("--split", default="test")
	ap.add_argument("--json", type=Path, default=None)
	a = ap.parse_args()
	meta = {
		r["path"]: r for r in map(json.loads, (a.data / "metadata.jsonl").read_text(encoding="utf-8").splitlines()) if r
	}
	z = np.load(a.scores)
	stats = json.loads(a.spec.read_text(encoding="utf-8"))["cameras"]["top"]
	u = np.max([(z[h] - st["t"]) / (st["t"] - st["med"]) for h, st in stats.items()], axis=0)
	rows = [meta[str(p)] for p in z["paths"]]
	keep = np.array([r["split"] == a.split and r["label"] in ("normal", "abnormal") for r in rows])
	out = {}
	for size in ("2x2", "2x4", "all"):
		sel = keep & np.array([size == "all" or (r["brick_2x4"] == (size == "2x4")) for r in rows])
		y = np.array([r["label"] == "abnormal" for r in rows])
		pos, neg = u[sel & y], u[sel & ~y]
		out[size] = {
			"auroc": auroc(neg, pos),
			"recall": [int((pos >= 0).sum()), len(pos)],
			"false_alarms": [int((neg >= 0).sum()), len(neg)],
			"normal_median": float(np.median(neg)) if len(neg) else None,
			"defective_median": float(np.median(pos)) if len(pos) else None,
		}
		o = out[size]
		print(
			f"{size}: AUROC {o['auroc']:.3f}  recall {o['recall'][0]}/{o['recall'][1]}  "
			f"false alarms {o['false_alarms'][0]}/{o['false_alarms'][1]}  "
			f"median normal {o['normal_median']:.3f} defective {o['defective_median']:.3f}"
		)
	if a.json is not None:
		a.json.write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
	main()
