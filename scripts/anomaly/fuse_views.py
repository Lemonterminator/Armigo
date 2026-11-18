"""Brick-level fusion of per-view detector scores (numpy only, no torch).

A brick is inspected from several views (``prescan`` wrist crops, ``show`` holds, top ``closeup``s, optionally the random
``virtual_show`` poses). ``infer_sup.py`` gives each crop a score per head (``cls``, ``seg``); this script turns the
views of one brick into ONE decision and measures the system-level recall / false-alarm rate with exact binomial
intervals, which is what a "99 % of defective bricks found" requirement has to be judged by.

Pipeline:

1. **Per-view standardisation.** Per camera and head, ``u = (score - t) / (t - median)`` with ``t`` the ``--view-q``
   quantile (default 0.99) and the median of the *calibration* normal crops, so ``u >= 0`` is the old single-view alarm.
   ``--head both`` takes ``max(u_cls, u_seg)`` per view.
2. **Brick statistic** over the views of the brick in the chosen view set: ``max`` (any view alarms), ``k2`` / ``k3``
   (the 2nd / 3rd largest ``u``: at least 2 / 3 views agree), ``top2`` (mean of the two largest).
3. **Brick threshold** = the ``1 - alpha`` quantile of the statistic over the *calibration* normal bricks, so ``alpha``
   is the false-alarm rate budget per brick; recall and false-alarm rate are then read on the *evaluation* bricks.

Calibration and evaluation sets are separate bricks (``--calib`` / ``--eval`` take ``FILE[:key=v1|v2,key=v]``
selectors over the ``.npz`` arrays; several files are joined with ``+``). ``coverage`` is the share of defective
bricks having at least one ``abnormal`` view (>= 25 visible defect pixels) in the view set: no detector can exceed it.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from math import sqrt
from pathlib import Path

import numpy as np

VIEW_SETS = {
	"pipeline": ("prescan", "show", "closeup"),  # what the heuristic inspector actually does
	"all": ("prescan", "show", "closeup", "virtual_show"),
	"wrist": ("prescan", "show", "virtual_show"),
}
STATS = ("max", "k2", "k3", "top2")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
	"""Wilson score interval of a binomial proportion (95 % by default)."""
	if n == 0:
		return float("nan"), float("nan")
	p = k / n
	d = 1 + z * z / n
	c = (p + z * z / (2 * n)) / d
	h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
	return c - h, c + h


def upper_zero(n: int, conf: float = 0.95) -> float:
	"""One-sided upper bound of the miss rate when 0 of ``n`` were missed (exact: 1 - (1 - conf)^(1/n))."""
	return 1 - (1 - conf) ** (1 / n) if n else float("nan")


def load(spec: str) -> dict[str, np.ndarray]:
	"""``a.npz[:key=v|v,key=v]+b.npz[:...]`` -> arrays of the rows matching the selectors (layouts made unique)."""
	parts, offset = [], 0
	for item in spec.split("+"):
		path, _, sel = item.rpartition(":")  # a Windows drive letter also contains ':', selectors always contain '='
		if "=" not in sel:
			path, sel = item, ""
		z = np.load(path)
		keep = np.ones(len(z["paths"]), bool)
		for cond in filter(None, sel.split(",")):
			key, _, vals = cond.partition("=")
			keep &= np.isin(z[key].astype(str), vals.split("|"))
		part = {k: z[k][keep] for k in z.files}
		part["layout"] = part["layout_index"].astype(np.int64) + offset
		offset += int(z["layout_index"].max()) + 1
		parts.append(part)
	return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def calibration_stats(cal: dict[str, np.ndarray], head: str, view_q: float) -> dict[str, dict[str, dict[str, float]]]:
	"""``{camera: {head: {"t": view_q quantile, "med": median}}}`` of the calibration normal crops."""
	heads = ("cls", "seg") if head == "both" else (head,)
	return {
		str(cam): {
			h: {
				"t": float(np.quantile(cal[h][cal["camera"] == cam], view_q)),
				"med": float(np.median(cal[h][cal["camera"] == cam])),
			}
			for h in heads
		}
		for cam in np.unique(cal["camera"])
	}


def standardise_value(score: float, t: float, med: float) -> float:
	"""``u = (score - t) / (t - median)``: 0 at the view-level threshold, 1 one (t - median) above it."""
	return (score - t) / max(t - med, 1e-6)


def standardise(z: dict[str, np.ndarray], stats: dict[str, dict[str, dict[str, float]]]) -> np.ndarray:
	"""Per-view ``u`` (>= 0: single-view alarm at the calibration quantile); ``max`` over the heads in ``stats``."""
	out = np.full(len(z["paths"]), -np.inf)
	for cam, per_head in stats.items():
		m = z["camera"] == cam
		for h, st in per_head.items():
			out[m] = np.maximum(out[m], standardise_value(z[h][m], st["t"], st["med"]))
	return out


def brick_table(z: dict[str, np.ndarray], u: np.ndarray, views: Sequence[str]) -> dict[str, np.ndarray]:
	"""One row per brick: label, per-statistic score, and whether it has any ``abnormal`` view in the set."""
	rows: dict[str, list] = {"layout": [], "defective": [], "n_views": [], "covered": [], **{s: [] for s in STATS}}
	for lay in np.unique(z["layout"]):
		m = (z["layout"] == lay) & np.isin(z["stage"], views)
		if not m.any():
			continue
		v = np.sort(u[m])[::-1]
		rows["layout"].append(int(lay))
		rows["defective"].append(bool(z["defective"][m][0]))
		rows["n_views"].append(int(m.sum()))
		rows["covered"].append(bool((z["label"][m] == "abnormal").any()))
		rows["max"].append(v[0])
		rows["k2"].append(v[1] if len(v) > 1 else -np.inf)
		rows["k3"].append(v[2] if len(v) > 2 else -np.inf)
		rows["top2"].append(v[:2].mean())
	return {k: np.array(v) for k, v in rows.items()}


def brick_auroc(neg: np.ndarray, pos: np.ndarray) -> float:
	if not len(neg) or not len(pos):
		return float("nan")
	s = np.r_[neg, pos]
	ranks = s.argsort(kind="stable").argsort() + 1.0
	for val in np.unique(s):  # average ranks of ties
		m = s == val
		ranks[m] = ranks[m].mean()
	return float((ranks[len(neg) :].sum() - len(pos) * (len(pos) + 1) / 2) / (len(neg) * len(pos)))


def operating_point(cal_scores: np.ndarray, ev: dict[str, np.ndarray], stat: str, alpha: float) -> dict:
	"""Threshold = (1 - alpha) quantile of the calibration normal bricks' statistic (``higher``: never under-shoots)."""
	thr = float(np.quantile(cal_scores, 1 - alpha, method="higher"))
	pos, neg = ev["defective"], ~ev["defective"]
	tp, fp = int((ev[stat][pos] >= thr).sum()), int((ev[stat][neg] >= thr).sum())
	n_pos, n_neg = int(pos.sum()), int(neg.sum())
	return {
		"alpha": alpha,
		"threshold": thr,
		"recall": tp / n_pos if n_pos else float("nan"),
		"recall_ci95": wilson(tp, n_pos),
		"missed": n_pos - tp,
		"n_defective": n_pos,
		"fpr": fp / n_neg if n_neg else float("nan"),
		"fpr_ci95": wilson(fp, n_neg),
		"false_alarms": fp,
		"n_normal": n_neg,
	}


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	ap.add_argument("--calib", required=True, help="normal calibration bricks, FILE[:key=v|v,...][+FILE...]")
	ap.add_argument("--eval", required=True, help="evaluation bricks (both classes), same syntax")
	ap.add_argument("--head", choices=("cls", "seg", "both"), default="both")
	ap.add_argument("--view-q", type=float, default=0.99)
	ap.add_argument("--views", choices=tuple(VIEW_SETS), default="pipeline")
	ap.add_argument("--alphas", type=float, nargs="+", default=[0.05, 0.02, 0.01])
	ap.add_argument("--out", type=Path, default=None, help="write the table as JSON")
	a = ap.parse_args()
	cal, ev = load(a.calib), load(a.eval)
	cal = {k: v[~cal["defective"]] for k, v in cal.items()}  # normal bricks only (defective calibration rows dropped)
	views = VIEW_SETS[a.views]
	stats = calibration_stats(cal, a.head, a.view_q)
	u_cal, u_ev = standardise(cal, stats), standardise(ev, stats)
	b_cal, b_ev = brick_table(cal, u_cal, views), brick_table(ev, u_ev, views)
	n_def, n_nor = int(b_ev["defective"].sum()), int((~b_ev["defective"]).sum())
	cov = int(b_ev["covered"][b_ev["defective"]].sum())
	print(
		f"views={a.views} head={a.head} view_q={a.view_q}: calibration {len(b_cal['layout'])} normal bricks; "
		f"eval {n_def} defective / {n_nor} normal; coverage {cov}/{n_def} = {cov / max(n_def, 1):.3f}"
	)
	result: dict = {"config": vars(a) | {"out": None}, "coverage": [cov, n_def], "stats": {}}
	for stat in STATS:
		auc = brick_auroc(b_ev[stat][~b_ev["defective"]], b_ev[stat][b_ev["defective"]])
		pts = [operating_point(b_cal[stat], b_ev, stat, al) for al in a.alphas]
		result["stats"][stat] = {"auroc": auc, "points": pts}
		line = "  ".join(
			f"a={p['alpha']:.2f}: rec {p['recall']:.3f} ({p['missed']} missed) fpr {p['fpr']:.3f}" for p in pts
		)
		print(f"  {stat:5s} AUROC {auc:.4f}  {line}")
	if a.out:
		a.out.parent.mkdir(parents=True, exist_ok=True)
		a.out.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")


if __name__ == "__main__":
	main()
