"""Brick-level fusion of per-view scores (``scripts/anomaly/fuse_views.py``, ``export_sup.py``): numpy only."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "anomaly"))

import fuse_views as fv


def _views(rng: np.random.Generator, n_bricks: int, defective: bool, first_layout: int) -> dict[str, np.ndarray]:
	"""Synthetic crops: 6 views per brick (3 wrist prescan, 1 wrist show, 2 top closeup)."""
	rows: dict[str, list] = {
		k: [] for k in ("paths", "layout_index", "camera", "stage", "label", "defective", "cls", "seg")
	}
	for b in range(n_bricks):
		for i, (cam, stage) in enumerate(
			[("wrist_cam", "prescan")] * 3 + [("wrist_cam", "show")] + [("top", "closeup")] * 2
		):
			hit = defective and i in (0, 4)  # the defect is seen in two views
			rows["paths"].append(f"png/{first_layout + b}_{i}.png")
			rows["layout_index"].append(first_layout + b)
			rows["camera"].append(cam)
			rows["stage"].append(stage)
			rows["label"].append("abnormal" if hit else "normal")
			rows["defective"].append(defective)
			rows["cls"].append(rng.normal(8.0 if hit else -3.0, 1.0))
			rows["seg"].append(rng.normal(12.0 if hit else -4.0, 1.5))
	return {k: np.array(v) for k, v in rows.items()}


@pytest.fixture
def scores(tmp_path: Path) -> tuple[Path, Path]:
	rng = np.random.default_rng(0)
	normal = _views(rng, 40, False, 0)
	test_n, test_d = _views(rng, 20, False, 100), _views(rng, 20, True, 200)
	calib = tmp_path / "calib.npz"
	test = tmp_path / "test.npz"
	np.savez(calib, **normal)
	np.savez(test, **{k: np.concatenate([test_n[k], test_d[k]]) for k in test_n})
	return calib, test


def test_wilson_interval():
	lo, hi = fv.wilson(0, 10)
	assert lo == pytest.approx(0.0, abs=1e-9) and 0.25 < hi < 0.31
	lo, hi = fv.wilson(50, 100)
	assert lo < 0.5 < hi and hi - lo < 0.21
	assert np.isnan(fv.wilson(0, 0)[0])


def test_upper_zero_is_rule_of_three():
	assert fv.upper_zero(300) == pytest.approx(3 / 300, rel=0.02)
	assert fv.upper_zero(64) > 0.04  # 64 defective bricks cannot demonstrate a 99 % recall


def test_standardise_value_is_zero_at_threshold_and_one_a_scale_above():
	assert fv.standardise_value(2.0, t=2.0, med=-3.0) == 0.0
	assert fv.standardise_value(7.0, t=2.0, med=-3.0) == pytest.approx(1.0)
	assert np.isfinite(fv.standardise_value(5.0, t=1.0, med=1.0))  # degenerate scale does not divide by zero


def test_load_selector_joins_files_with_unique_layouts(scores):
	calib, test = scores
	z = fv.load(f"{test}:defective=True")
	assert z["defective"].all() and len(z["paths"]) == 20 * 6
	both = fv.load(f"{calib}+{test}")
	# bricks of the second file must not collide with the first file's layout ids
	assert len(np.unique(both["layout"])) == 40 + 40


def test_brick_table_statistics_and_coverage(scores):
	_, test = scores
	z = fv.load(str(test))
	stats = fv.calibration_stats(z, "both", 0.99)
	u = fv.standardise(z, stats)
	table = fv.brick_table(z, u, fv.VIEW_SETS["pipeline"])
	assert len(table["layout"]) == 40
	assert table["covered"][table["defective"]].all() and not table["covered"][~table["defective"]].any()
	assert (table["max"] >= table["k2"]).all() and (table["k2"] >= table["k3"]).all()
	only_prescan = fv.brick_table(z, u, ("prescan",))
	assert (only_prescan["n_views"] == 3).all()


def test_separable_scores_give_perfect_brick_auroc_and_calibrated_fpr(scores):
	calib, test = scores
	cal, ev = fv.load(str(calib)), fv.load(str(test))
	stats = fv.calibration_stats(cal, "both", 0.99)
	b_cal = fv.brick_table(cal, fv.standardise(cal, stats), fv.VIEW_SETS["pipeline"])
	b_ev = fv.brick_table(ev, fv.standardise(ev, stats), fv.VIEW_SETS["pipeline"])
	assert fv.brick_auroc(b_ev["max"][~b_ev["defective"]], b_ev["max"][b_ev["defective"]]) == 1.0
	point = fv.operating_point(b_cal["max"], b_ev, "max", 0.05)
	assert point["recall"] == 1.0 and point["fpr"] == 0.0 and point["n_defective"] == 20


def test_brick_auroc_handles_ties_and_empty():
	assert fv.brick_auroc(np.array([0.0, 0.0]), np.array([0.0, 0.0])) == 0.5
	assert np.isnan(fv.brick_auroc(np.array([]), np.array([1.0])))


def test_export_sup_spec_roundtrip(scores, tmp_path):
	calib, _ = scores
	out = tmp_path / "models" / "sup.json"
	ckpt = tmp_path / "ckpt" / "head.pt"
	ckpt.parent.mkdir()
	ckpt.write_bytes(b"")
	subprocess.run(
		[
			sys.executable,
			str(ROOT / "scripts" / "anomaly" / "export_sup.py"),
			"--ckpt",
			str(ckpt),
			"--calib",
			str(calib),
			"--out",
			str(out),
		],
		check=True,
		capture_output=True,
	)
	spec = json.loads(out.read_text(encoding="utf-8"))
	assert spec["kind"] == "sup" and (out.parent / spec["checkpoint"]).resolve() == ckpt.resolve()
	assert set(spec["cameras"]) == {"top", "wrist_cam"}
	assert set(spec["cameras"]["top"]) == {"cls", "seg"}
	z = fv.load(str(calib))
	expected = fv.calibration_stats(z, "both", 0.99)
	assert spec["cameras"]["wrist_cam"]["cls"]["t"] == pytest.approx(expected["wrist_cam"]["cls"]["t"])
