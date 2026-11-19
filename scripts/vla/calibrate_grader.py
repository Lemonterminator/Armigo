"""Calibrate a frozen VLA grader reference JSON from a batch of calibration episodes.

Computes smoothness and belief-error statistics using ``digital_twin_armigo.agent.vla_grader``
helpers, selects a smoothness multiplier on the marginal p95 thresholds so the joint smoothness
rejection rate (failing either ``jerk_index_cmd`` or ``jerk_index_meas``) matches the target
rejection rate (default 0.05), and writes a frozen reference JSON compatible with
``vla_grader.load_reference()``.

Usage:
	PYTHONPATH=src python scripts/vla/calibrate_grader.py EPISODES_DIR [EPISODES_DIR ...] \\
		--out config/vla/grader_reference_calibrated.json --target-rejection-rate 0.05
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
	sys.path.insert(0, str(ROOT / "src"))

from digital_twin_armigo.agent.vla_grader import (  # noqa: E402
	REFERENCE_PATH,
	build_smoothness_reference,
	find_episodes,
	grade,
	load_reference,
	phase_jerk,
	smoothness_index,
)

JERK_KEYS = ("jerk_index_cmd", "jerk_index_meas")


def _collect_belief_error_stats(episodes: list[Path], base_ref: dict) -> dict:
	"""Compute belief-error summary statistics across all scored views in ``episodes``."""
	by_camera: dict[str, list[float]] = {}
	all_errors: list[float] = []
	episode_statuses: list[str] = []

	for ep in episodes:
		g = grade(ep, reference=base_ref, rebuild_layout=False)
		episode_statuses.append(g["checks"]["belief_error"]["status"])
		events = [
			json.loads(line)
			for line in (ep / "events.jsonl").read_text(encoding="utf-8").splitlines()
			if line
		]
		views = {
			e["observation"]["id"]: e["observation"]
			for e in events
			if e["kind"] == "view" and e["observation"].get("score") is not None
		}
		truths = {e["id"]: e for e in events if e["kind"] == "view_truth"}
		for vid, view in views.items():
			truth = truths.get(vid)
			if truth is None:
				continue
			err = truth.get("belief_error_mm")
			if isinstance(err, (int, float)) and not isinstance(err, bool) and math.isfinite(err):
				val = float(err)
				all_errors.append(val)
				by_camera.setdefault(view["camera"], []).append(val)

	def _summary(vals: list[float]) -> dict:
		if not vals:
			return {"views": 0, "median_mm": None, "p95_mm": None, "max_mm": None}
		arr = np.asarray(vals, dtype=np.float64)
		return {
			"views": len(vals),
			"median_mm": round(float(np.median(arr)), 4),
			"p95_mm": round(float(np.percentile(arr, 95)), 4),
			"max_mm": round(float(np.max(arr)), 4),
		}

	n_ep = len(episode_statuses)
	n_fail = sum(1 for s in episode_statuses if s == "fail")
	return {
		"episodes": n_ep,
		"episode_rejection_rate": round(n_fail / n_ep, 4) if n_ep else 0.0,
		"overall": _summary(all_errors),
		"by_camera": {cam: _summary(vals) for cam, vals in sorted(by_camera.items())},
	}


def _choose_joint_multiplier(
	indices: list[dict[str, float]],
	base_p95: dict[str, float],
	target_rejection_rate: float,
) -> tuple[float, dict[str, float], float, float]:
	"""Choose a scalar multiplier ``alpha`` on ``base_p95`` so the joint rejection rate matches ``target_rejection_rate``."""
	ratios = [
		max(idx[k] / max(float(base_p95[k]), 1e-9) for k in JERK_KEYS)
		for idx in indices
	]
	marginal_rej = sum(
		1 for idx in indices if any(idx[k] > float(base_p95[k]) for k in JERK_KEYS)
	) / len(indices)

	if target_rejection_rate <= 0.0:
		alpha = max(ratios) * 1.0001
		calibrated_p95 = {
			k: math.ceil(alpha * float(base_p95[k]) * 10000.0) / 10000.0
			for k in JERK_KEYS
		}
	elif target_rejection_rate >= 1.0:
		alpha = 0.0
		calibrated_p95 = {k: 0.0 for k in JERK_KEYS}
	else:
		alpha = float(np.quantile(ratios, 1.0 - target_rejection_rate))
		calibrated_p95 = {k: round(alpha * float(base_p95[k]), 4) for k in JERK_KEYS}

	joint_rej = sum(
		1 for idx in indices if any(idx[k] > calibrated_p95[k] for k in JERK_KEYS)
	) / len(indices)
	return alpha, calibrated_p95, joint_rej, marginal_rej


def calibrate_reference(
	episode_dirs: list[Path],
	out_path: Path,
	target_rejection_rate: float = 0.05,
	base_reference_path: Path | None = None,
) -> dict:
	"""Compute smoothness & belief-error stats and write a frozen grader reference JSON."""
	if not 0.0 <= target_rejection_rate <= 1.0:
		raise ValueError(f"target_rejection_rate must be in [0, 1], got {target_rejection_rate}")

	episodes = [ep for d in episode_dirs for ep in find_episodes(d)]
	if not episodes:
		raise FileNotFoundError("No episode.json found in the provided directories")

	base_ref = load_reference(base_reference_path)
	smooth = build_smoothness_reference(episodes)
	if smooth["episodes"] == 0:
		raise ValueError("No successful episodes found for smoothness calibration")

	indices: list[dict[str, float]] = []
	for ep in episodes:
		manifest = json.loads((ep / "episode.json").read_text(encoding="utf-8"))
		if manifest["result"].get("verdict") != "SUCCESS":
			continue
		per_phase = phase_jerk(ep, manifest)
		idx = smoothness_index(per_phase, smooth["phase_median_rms"])
		if idx is not None and all(math.isfinite(v) for v in idx.values()):
			indices.append(idx)

	if not indices:
		raise ValueError("No valid smoothness indices could be computed from calibration episodes")

	marginal_p95 = dict(smooth["p95"])
	alpha, calibrated_p95, joint_rej, marginal_rej = _choose_joint_multiplier(
		indices, marginal_p95, target_rejection_rate
	)
	belief_stats = _collect_belief_error_stats(episodes, base_ref)

	frozen = {
		"thresholds": dict(base_ref["thresholds"]),
		"notes": dict(base_ref.get("notes", {})),
		"smoothness": {
			"episodes": smooth["episodes"],
			"phase_median_rms": smooth["phase_median_rms"],
			"p95": calibrated_p95,
			"median": smooth["median"],
			"marginal_p95": marginal_p95,
			"multiplier": round(alpha, 6),
			"target_joint_rejection_rate": target_rejection_rate,
			"actual_joint_rejection_rate": round(joint_rej, 4),
			"marginal_rejection_rate": round(marginal_rej, 4),
			"source": [d.as_posix() for d in episode_dirs],
			"built": date.today().isoformat(),
		},
		"calibration": {
			"total_episodes": len(episodes),
			"smoothness_episodes": len(indices),
			"belief_error": belief_stats,
		},
	}

	out_path.parent.mkdir(parents=True, exist_ok=True)
	out_path.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
	return frozen


def main():
	p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	p.add_argument("episodes", nargs="+", type=Path, help="Calibration episode directory (or parent run dirs).")
	p.add_argument("--out", "-o", type=Path, required=True, help="Output frozen grader reference JSON path.")
	p.add_argument(
		"--target-rejection-rate",
		type=float,
		default=0.05,
		help="Target joint smoothness rejection rate (default: 0.05).",
	)
	p.add_argument(
		"--base-reference",
		type=Path,
		default=REFERENCE_PATH,
		help="Base grader reference JSON (default: config/vla/grader_reference.json).",
	)
	args = p.parse_args()

	frozen = calibrate_reference(
		episode_dirs=args.episodes,
		out_path=args.out,
		target_rejection_rate=args.target_rejection_rate,
		base_reference_path=args.base_reference,
	)
	sm = frozen["smoothness"]
	be = frozen["calibration"]["belief_error"]["overall"]
	print(
		f"Calibrated {args.out} from {sm['episodes']} successful episodes "
		f"(multiplier={sm['multiplier']:.4f}, marginal_rej={sm['marginal_rejection_rate']:.3f}, "
		f"joint_rej={sm['actual_joint_rejection_rate']:.3f}, p95={sm['p95']})"
	)
	print(
		f"Belief error ({be['views']} views): median={be['median_mm']} mm, "
		f"p95={be['p95_mm']} mm, max={be['max_mm']} mm"
	)


if __name__ == "__main__":
	main()
