"""Calibrate precursor thresholds on a dev batch (never on sealed or held-out data).

    PYTHONPATH=src python scripts/experiments/calibrate_precursors.py \
        --scores D:/Armigo_data_v2/reports/p2-dev-r2/scores.jsonl --out config/experiments/precursor_thresholds.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from digital_twin_armigo.experiments.precursors import RELEVANT

CALIBRATED = {
	"brick_xy_mm": "brick_pose_error",
	"brick_yaw_deg": "brick_yaw_error",
	"bowl_xy_mm": "bowl_pose_error",
	"view_belief_mm": "view_belief_error",
}


def _auc(pos, neg) -> float:
	pairs = [(1.0 if p > n else 0.5 if p == n else 0.0) for p in pos for n in neg]
	return sum(pairs) / len(pairs) if pairs else float("nan")


def youden_threshold(pos, neg, max_neg_rate: float = 0.10) -> dict:
	best = None
	for c in sorted(set(pos) | set(neg)):
		tpr = sum(v > c for v in pos) / len(pos)
		fpr = sum(v > c for v in neg) / len(neg)
		if fpr > max_neg_rate:
			continue
		if best is None or tpr - fpr > best["j"]:
			best = {"threshold": c, "tpr": tpr, "fpr": fpr, "j": tpr - fpr}
	best = best or {"threshold": max([*pos, *neg]), "tpr": 0.0, "fpr": 0.0, "j": 0.0}
	return {**best, "auc": _auc(pos, neg), "n_pos": len(pos), "n_neg": len(neg)}


def metric_values(events: list[dict]) -> dict[str, float]:
	out: dict[str, float] = {}
	for e in events:
		if e.get("kind") == "brick_perceived" and e.get("found") and "brick_xy_mm" not in out:
			out["brick_xy_mm"], out["brick_yaw_deg"] = float(e["error_xy_mm"]), float(e["error_yaw_deg"])
		elif e.get("kind") == "bowls_perceived":
			errs = [v for v in (e.get("error_xy_mm") or {}).values() if v is not None]
			if errs:
				out["bowl_xy_mm"] = max(out.get("bowl_xy_mm", 0.0), float(max(errs)))
		elif e.get("kind") == "view_truth" and e.get("belief_error_mm") is not None:
			out["view_belief_mm"] = max(out.get("view_belief_mm", 0.0), float(e["belief_error_mm"]))
	return out


def _events(path: Path) -> list[dict]:
	f = path / "events.jsonl"
	return [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line] if f.is_file() else []


def main(argv=None) -> int:
	p = argparse.ArgumentParser()
	p.add_argument("--scores", type=Path, required=True)
	p.add_argument("--out", type=Path, required=True)
	p.add_argument("--max-neg-rate", type=float, default=0.10)
	a = p.parse_args(argv)
	rows = [json.loads(line) for line in a.scores.read_text(encoding="utf-8").splitlines() if line]
	pos = {m: [] for m in CALIBRATED}
	neg = {m: [] for m in CALIBRATED}
	for r in rows:
		values = metric_values(_events(Path(r["path"])))
		for metric, mech in CALIBRATED.items():
			if metric not in values:
				continue
			if r["root_cause"] == "none":
				neg[metric].append(values[metric])
			elif mech in RELEVANT.get(r["root_cause"], ()):
				pos[metric].append(values[metric])
	metrics = {m: youden_threshold(pos[m], neg[m], a.max_neg_rate) for m in CALIBRATED if pos[m] and neg[m]}
	repo = Path(__file__).resolve().parents[2]
	commit = subprocess.run(
		["git", "-C", str(repo), "rev-parse", "HEAD"], check=False, capture_output=True, text=True
	).stdout.strip()
	a.out.parent.mkdir(parents=True, exist_ok=True)
	a.out.write_text(
		json.dumps({"source": str(a.scores), "commit": commit, "metrics": metrics}, indent=2) + "\n", encoding="utf-8"
	)
	for m, r in metrics.items():
		print(
			f"{m}: threshold {r['threshold']:.2f} auc {r['auc']:.3f} "
			f"tpr {r['tpr']:.2f} fpr {r['fpr']:.2f} ({r['n_pos']}+/{r['n_neg']}-)"
		)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
