"""Score a bottom-view RTX replay under separate normal-only calibration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from evaluate_bottom import counts

from digital_twin_armigo.inspection_scorer import default_sup_scorer


def main():
	p = argparse.ArgumentParser(description=__doc__)
	p.add_argument("data", type=Path)
	p.add_argument("--detector", type=Path, required=True)
	p.add_argument("--out", type=Path, required=True)
	a = p.parse_args()
	rows = [json.loads(r) for r in (a.data / "metadata.jsonl").read_text(encoding="utf-8").splitlines()]
	with default_sup_scorer(a.detector.resolve()) as scorer:
		for row in rows:
			image = cv2.imread(str(a.data / row["path"]))
			if image is None:
				raise FileNotFoundError(row["path"])
			row["score"] = scorer.score("top", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
	normal_scores = [r["score"] for r in rows if r["split"] == "calibration" and not r["defective"]]
	if len(normal_scores) < 4:
		raise ValueError("fewer than four independent calibration normals")
	threshold = max(0.0, float(np.quantile(normal_scores, 0.99, method="higher")))
	test = [r for r in rows if r["split"] == "test"]
	report = {
		"engine": "isaac-rtx",
		"physics": "mujoco-recorded",
		"camera": "bottom",
		"calibration_normals": len(normal_scores),
		"default": counts(test, 0.0),
		"local_calibration": counts(test, threshold),
		"scores": rows,
	}
	a.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
	print(json.dumps({k: v for k, v in report.items() if k != "scores"}, indent=2))


if __name__ == "__main__":
	main()
