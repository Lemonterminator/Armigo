"""Log cross-detector runs (``train_sup.py`` supervised heads and ``train_eval.py`` PatchCore baselines) to MLflow.

Run with the anomaly venv's python. The store is ``mlflow/mlflow.db`` + ``mlflow/artifacts`` in the repo root (git-ignored,
see ``digital_twin_armigo.tracking.runs``); browse it with ``armigo-track ui``.

  log_mlflow.py --run-dir results/anomaly/sup-lora --tag lora            # one run
  log_mlflow.py --backfill                                               # every historical report, idempotent
  log_mlflow.py --run-dir ... --tag lora --register-as cross-detector --alias candidate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
	sys.path.insert(0, str(ROOT / "src"))

from digital_twin_armigo.tracking.runs import tracked_run

PARAM_KEYS = ("mode", "rank", "input_size", "steps", "real_top_crop", "val_quantile", "device", "torch", "anomalib")


def _dataset_dirs(report: dict, base: Path) -> dict[str, Path]:
	"""Dataset directories named in a supervised report's ``data`` block (PatchCore reports hold counts only)."""
	out: dict[str, Path] = {}
	for role, value in (report.get("data") or {}).items():
		for i, raw in enumerate(value if isinstance(value, list) else [value]):
			if isinstance(raw, str):
				path = Path(raw)
				path = path if path.is_absolute() else (base / path).resolve()
				if path.is_dir():
					out[role if i == 0 else f"{role}{i}"] = path
	return out


def log_report(
	run_dir: Path,
	tag: str,
	*,
	experiment: str = "cross-detector",
	run_name: str | None = None,
	note: str = "",
	best: bool = False,
	hash_datasets: bool = True,
	register_as: str | None = None,
	alias: str | None = None,
) -> str | None:
	report_path = run_dir / f"report-{tag}.json"
	report = json.loads(report_path.read_text(encoding="utf-8"))
	try:
		source = report_path.resolve().relative_to(ROOT).as_posix()
	except ValueError:
		source = report_path.as_posix()
	params = {k: report[k] for k in PARAM_KEYS if k in report and not isinstance(report[k], (dict, list))}
	params.update({f"n_{k}": v for k, v in (report.get("n") or {}).items()})
	params.update({f"data_{k}": v for k, v in (report.get("data") or {}).items() if isinstance(v, int)})
	metrics = {}
	if "best_val_auroc" in report:
		metrics["best_val_auroc"] = report["best_val_auroc"]
	for name, res in report["models"].items():
		head, scope = name.split("/")
		at = res["at_threshold"]
		metrics.update(
			{
				f"{head}_{scope}_auroc": res["auroc"],
				f"{head}_{scope}_recall": at["recall"],
				f"{head}_{scope}_fpr": at["fpr"],
				f"{head}_{scope}_best_f1": res["best_f1_oracle"],
			}
		)
	datasets = _dataset_dirs(report, run_dir) if hash_datasets else {}
	with tracked_run(
		experiment,
		run_name or tag,
		params=params,
		tags={
			"task": "cross_var",
			"detector": "patchcore" if "patchcore/pooled" in report["models"] else "supervised",
			"source_report": source,
			"best": str(best).lower(),
		},
		datasets=datasets,
		note=note or None,
	) as run:
		run.log_metrics(metrics)
		run.log_dict(report, "report.json")
		for f in sorted(run_dir.glob(f"*-{tag}.*")):
			if f.suffix != ".npz" or f.stat().st_size < 50 * 2**20:
				run.log_artifact(f)
		head_weights = run_dir / f"head-{tag}.pt"
		if register_as and head_weights.is_file():
			run.log_artifact(head_weights, "model")
			run.register_model(register_as, "model", alias)
		print("run", run.run_id, source)
		return run.run_id


def _already_logged(experiment: str, source: str, name: str) -> bool:
	import mlflow

	exp = mlflow.get_experiment_by_name(experiment)
	if exp is None:
		return False
	# MLflow filter strings have no OR: match on either the source report or the run name with two queries.
	return any(
		mlflow.search_runs([exp.experiment_id], filter_string=f, max_results=1, output_format="list")
		for f in (f"tags.source_report = '{source}'", f"tags.`mlflow.runName` = '{name}'")
	)


def backfill(experiment: str, hash_datasets: bool) -> None:
	import mlflow

	from digital_twin_armigo.tracking.runs import tracking_uri

	mlflow.set_tracking_uri(tracking_uri())
	reports = sorted((ROOT / "results" / "anomaly").glob("*/report-*.json"))
	for path in reports:
		tag = path.stem.removeprefix("report-")
		source = path.relative_to(ROOT).as_posix()
		if _already_logged(experiment, source, tag):
			print("skip (already logged)", source)
			continue
		log_report(path.parent, tag, experiment=experiment, hash_datasets=hash_datasets)


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	ap.add_argument("--run-dir", type=Path, help="train_sup.py --report-dir")
	ap.add_argument("--tag")
	ap.add_argument(
		"--backfill", action="store_true", help="log every results/anomaly/*/report-*.json not yet in the store"
	)
	ap.add_argument("--no-hash-datasets", action="store_true", help="skip dataset manifests (hashing multi-GB renders)")
	ap.add_argument("--experiment", default="cross-detector")
	ap.add_argument("--run-name", default=None)
	ap.add_argument("--note", default="")
	ap.add_argument("--best", action="store_true", help="tag the run best=true")
	ap.add_argument("--register-as", help="register head-<tag>.pt as a new version of this registered model")
	ap.add_argument("--alias", help="alias to point at the new model version, e.g. candidate")
	a = ap.parse_args()
	if a.backfill:
		backfill(a.experiment, not a.no_hash_datasets)
		return
	if not (a.run_dir and a.tag):
		ap.error("--run-dir and --tag are required unless --backfill is given")
	log_report(
		a.run_dir,
		a.tag,
		experiment=a.experiment,
		run_name=a.run_name,
		note=a.note,
		best=a.best,
		hash_datasets=not a.no_hash_datasets,
		register_as=a.register_as,
		alias=a.alias,
	)


if __name__ == "__main__":
	main()
