"""Log a ``train_sup.py`` run (report, scores, adapter + head weights) to the local MLflow store.

Run with the anomaly venv's python. The store is ``mlflow/mlflow.db`` + ``mlflow/artifacts`` in the repo root (git-ignored);
browse it with ``mlflow server --backend-store-uri sqlite:///mlflow/mlflow.db --default-artifact-root mlflow/artifacts``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	ap.add_argument("--run-dir", type=Path, required=True, help="train_sup.py --report-dir")
	ap.add_argument("--tag", required=True)
	ap.add_argument("--experiment", default="cross-detector")
	ap.add_argument("--run-name", default=None)
	ap.add_argument("--note", default="")
	a = ap.parse_args()
	report = json.loads((a.run_dir / f"report-{a.tag}.json").read_text(encoding="utf-8"))
	store = ROOT / "mlflow"
	(store / "artifacts").mkdir(parents=True, exist_ok=True)
	mlflow.set_tracking_uri(f"sqlite:///{(store / 'mlflow.db').as_posix()}")
	if mlflow.get_experiment_by_name(a.experiment) is None:
		mlflow.create_experiment(a.experiment, artifact_location=(store / "artifacts").as_uri())
	mlflow.set_experiment(a.experiment)
	with mlflow.start_run(run_name=a.run_name or a.tag):
		mlflow.log_params({k: report[k] for k in ("mode", "rank", "input_size", "steps", "real_top_crop", "val_quantile")})
		mlflow.log_params({f"n_{k}": v for k, v in report["n"].items()})
		mlflow.log_params({f"data_{k}": v for k, v in report["data"].items()})
		mlflow.log_metric("best_val_auroc", report["best_val_auroc"])
		for name, res in report["models"].items():
			head, scope = name.split("/")
			at = res["at_threshold"]
			mlflow.log_metrics(
				{
					f"{head}_{scope}_auroc": res["auroc"],
					f"{head}_{scope}_recall": at["recall"],
					f"{head}_{scope}_fpr": at["fpr"],
					f"{head}_{scope}_best_f1": res["best_f1_oracle"],
				}
			)
		for f in sorted(a.run_dir.glob(f"*-{a.tag}.*")):
			mlflow.log_artifact(str(f))
		if a.note:
			mlflow.set_tag("mlflow.note.content", a.note)
		mlflow.set_tag("task", "cross_var")
		mlflow.set_tag("best", "true")
		print("run", mlflow.active_run().info.run_id)


if __name__ == "__main__":
	main()
