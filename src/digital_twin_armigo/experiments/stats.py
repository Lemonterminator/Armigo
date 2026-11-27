"""Per-cell metrics, layout-clustered bootstrap intervals and paired comparisons for score rows.

Rows are the dicts of ``scoring.score_episode``. Episodes of one layout share the layout (and its normal/defective
pair), so every interval resamples *layouts* (``layout_id``), never single episodes. All randomness is seeded.
"""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import NormalDist

import numpy as np

# metric -> (numerator predicate, denominator predicate); denominators include "unknown" outcomes.
METRICS = {
	"success_rate": (lambda r: bool(r.get("physical_success")), lambda r: True),
	"defect_miss_rate": (lambda r: bool(r.get("defect_miss")), lambda r: r.get("expected_label") == "defective"),
	"false_alarm_rate": (lambda r: bool(r.get("false_alarm")), lambda r: r.get("expected_label") == "normal"),
	"unknown_rate": (lambda r: bool(r.get("is_unknown")), lambda r: True),
	"wrong_bowl_rate": (lambda r: bool(r.get("wrong_bowl")), lambda r: True),
}


def _clusters(rows, num, den):
	by = defaultdict(lambda: [0.0, 0.0])
	for r in rows:
		if den(r):
			cell = by[r.get("layout_id") or r.get("case_id")]
			cell[0] += float(num(r))
			cell[1] += 1.0
	return np.array(list(by.values()), dtype=np.float64).reshape(-1, 2)


def bootstrap_ratio(clusters: np.ndarray, B: int = 2000, seed: int = 0, alpha: float = 0.05):
	"""``(point, lo, hi)`` of sum(num)/sum(den) with a percentile interval over resampled clusters (rows: num, den)."""
	if len(clusters) == 0 or clusters[:, 1].sum() == 0:
		return math.nan, math.nan, math.nan
	point = clusters[:, 0].sum() / clusters[:, 1].sum()
	rng = np.random.default_rng(seed)
	idx = rng.integers(0, len(clusters), size=(B, len(clusters)))
	num, den = clusters[idx, 0].sum(axis=1), clusters[idx, 1].sum(axis=1)
	ok = den > 0
	draws = num[ok] / den[ok]
	lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
	return float(point), float(lo), float(hi)


def metric_with_ci(rows, metric: str, B: int = 2000, seed: int = 0) -> dict:
	num, den = METRICS[metric]
	clusters = _clusters(rows, num, den)
	point, lo, hi = bootstrap_ratio(clusters, B=B, seed=seed)
	return {
		"value": point,
		"ci_lo": lo,
		"ci_hi": hi,
		"n": int(clusters[:, 1].sum()) if len(clusters) else 0,
		"k": int(sum(num(r) for r in rows if den(r))),
		"n_clusters": len(clusters),
	}


def cell_metrics(rows, B: int = 2000, seed: int = 0) -> dict:
	"""All headline metrics of one (engine, group) cell, plus mean extra views and latency."""
	rows = list(rows)
	out = {"n_episodes": len(rows)}
	for name in METRICS:
		out[name] = metric_with_ci(rows, name, B=B, seed=seed)

	def mean(key):
		vals = [r[key] for r in rows if r.get(key) is not None]
		return float(np.mean(vals)) if vals else None

	out["extra_views"] = mean("n_review_views")
	out["scored_views"] = mean("n_scored_views")
	out["wall_s"] = mean("wall_s")
	out["decision_latency_s"] = mean("decision_latency_s")
	return out


def _exact_mcnemar(n10: int, n01: int) -> float:
	n = n10 + n01
	if n == 0:
		return 1.0
	k = min(n10, n01)
	p = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
	return float(min(1.0, 2 * p))


def paired_comparison(
	rows_a, rows_b, metric: str = "success_rate", B: int = 2000, seed: int = 0, key="case_id"
) -> dict:
	"""``B - A`` of a rate over cases present in both groups, matched on ``key``; CI clustered by layout.

	``n10``: A hit and B miss; ``n01``: A miss and B hit (discordant pairs). A pair counts where the metric's
	denominator holds for the A row. ``diff > 0`` means B is better for success-like metrics.
	"""
	num, den = METRICS[metric]
	a = {r[key]: r for r in rows_a}
	b = {r[key]: r for r in rows_b}
	shared = sorted(set(a) & set(b))
	by = defaultdict(lambda: [0.0, 0.0])
	n10 = n01 = n11 = n00 = 0
	for k in shared:
		if not den(a[k]):
			continue
		x, y = bool(num(a[k])), bool(num(b[k]))
		n10 += x and not y
		n01 += y and not x
		n11 += x and y
		n00 += not x and not y
		cell = by[a[k].get("layout_id") or k]
		cell[0] += float(y) - float(x)
		cell[1] += 1.0
	clusters = np.array(list(by.values()), dtype=np.float64).reshape(-1, 2)
	diff, lo, hi = bootstrap_ratio(clusters, B=B, seed=seed)
	n = n10 + n01 + n11 + n00
	return {
		"metric": metric,
		"n_pairs": n,
		"n_clusters": len(clusters),
		"diff": diff,
		"ci_lo": lo,
		"ci_hi": hi,
		"n10": n10,
		"n01": n01,
		"n11": n11,
		"n00": n00,
		"discordance_rate": (n10 + n01) / n if n else math.nan,
		"p_mcnemar": _exact_mcnemar(n10, n01),
		"unmatched_a": len(set(a) - set(b)),
		"unmatched_b": len(set(b) - set(a)),
	}


def minimum_detectable_effect(
	discordance_rate: float,
	n_clusters: int,
	alpha: float = 0.05,
	power: float = 0.8,
	pairs_per_cluster: float = 1.0,
	design_effect: float = 1.0,
) -> float:
	"""Smallest true paired rate difference detected with ``power`` at two-sided ``alpha``.

	A pair's difference d in {-1, 0, 1} has E[d^2] = discordance_rate (when the effect is small), so the standard
	error of the mean over ``N = n_clusters * pairs_per_cluster / design_effect`` effective pairs is
	``sqrt(discordance_rate / N)`` and the MDE is ``(z_{1-alpha/2} + z_power) * SE``. Pass the layout-level design
	effect of the dev data if pairs of one layout are correlated.
	"""
	if n_clusters <= 0 or discordance_rate < 0:
		return math.nan
	n_eff = n_clusters * pairs_per_cluster / max(design_effect, 1e-9)
	z = NormalDist().inv_cdf
	return float((z(1 - alpha / 2) + z(power)) * math.sqrt(discordance_rate / n_eff))
