"""MLflow run helpers shared by every training / evaluation script.

Scripts stay usable without MLflow: tracking is opt-in (``--mlflow``), and ``tracked_run(enabled=False)`` yields a
no-op handle with the same methods. Requesting tracking without MLflow installed fails loudly rather than silently
dropping the record.

Store location (git-ignored): ``mlflow/mlflow.db`` + ``mlflow/artifacts`` under the repo root, overridable with
``ARMIGO_MLFLOW_DIR`` (a directory) or ``ARMIGO_MLFLOW_URI`` (any tracking URI, e.g. a shared server).
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .lineage import ROOT, dataset_ref, git_state

_PARAM_BATCH = 90  # MLflow accepts at most 100 params per call.
_PARAM_MAXLEN = 500


def store_dir() -> Path:
	return Path(os.environ.get("ARMIGO_MLFLOW_DIR") or ROOT / "mlflow")


def tracking_uri() -> str:
	return os.environ.get("ARMIGO_MLFLOW_URI") or f"sqlite:///{(store_dir() / 'mlflow.db').as_posix()}"


def flatten_numbers(obj: Any, prefix: str = "", sep: str = ".") -> dict[str, float]:
	"""Finite numeric leaves of a nested report as ``{"a.b.c": value}``; bools, strings and lists are skipped."""
	out: dict[str, float] = {}
	if isinstance(obj, Mapping):
		for k, v in obj.items():
			out.update(flatten_numbers(v, f"{prefix}{sep}{k}" if prefix else str(k), sep))
	elif isinstance(obj, (int, float)) and not isinstance(obj, bool) and math.isfinite(obj):
		out[re.sub(r"[^\w\-. /]", "_", prefix)] = float(obj)  # MLflow rejects other characters in metric names
	return out


def _clip(value: Any) -> str:
	return str(value)[:_PARAM_MAXLEN]


class NullRun:
	"""Stand-in used when tracking is off; every method is a no-op."""

	run_id: str | None = None

	def log_params(self, params: Mapping[str, Any]) -> None: ...
	def log_metrics(self, metrics: Mapping[str, Any], step: int | None = None) -> None: ...
	def log_report(self, report: Mapping[str, Any], prefix: str = "") -> None: ...
	def log_artifact(self, path: Path | str, artifact_path: str | None = None) -> None: ...
	def log_dict(self, obj: Any, artifact_file: str) -> None: ...
	def set_tags(self, tags: Mapping[str, Any]) -> None: ...
	def register_model(self, name: str, artifact_path: str, alias: str | None = None) -> None: ...


class MlflowRun(NullRun):
	def __init__(self, mlflow: Any, run: Any):
		self._mlflow = mlflow
		self.run_id = run.info.run_id

	def log_params(self, params: Mapping[str, Any]) -> None:
		items = [(str(k), _clip(v)) for k, v in params.items() if v is not None]
		for i in range(0, len(items), _PARAM_BATCH):
			self._mlflow.log_params(dict(items[i : i + _PARAM_BATCH]))

	def log_metrics(self, metrics: Mapping[str, Any], step: int | None = None) -> None:
		clean = flatten_numbers(metrics)
		if clean:
			self._mlflow.log_metrics(clean, step=step)

	def log_report(self, report: Mapping[str, Any], prefix: str = "") -> None:
		"""Log every finite numeric leaf of a (possibly nested) report as a metric."""
		self.log_metrics(flatten_numbers(report, prefix))

	def log_artifact(self, path: Path | str, artifact_path: str | None = None) -> None:
		p = Path(path)
		if p.is_dir():
			self._mlflow.log_artifacts(str(p), artifact_path)
		elif p.is_file():
			self._mlflow.log_artifact(str(p), artifact_path)

	def log_dict(self, obj: Any, artifact_file: str) -> None:
		self._mlflow.log_dict(obj, artifact_file)

	def set_tags(self, tags: Mapping[str, Any]) -> None:
		self._mlflow.set_tags({str(k): _clip(v) for k, v in tags.items() if v is not None})

	def register_model(self, name: str, artifact_path: str, alias: str | None = None) -> None:
		"""Register a run artifact (weights directory/file) as a new version of ``name``, optionally setting an alias.

		Goes through the client API with the artifact URI as source: ``mlflow.register_model("runs:/...")`` only
		accepts MLflow-flavoured logged models, and our checkpoints (LoRA adapters, heads) are plain files.
		"""
		client = self._mlflow.MlflowClient()
		try:
			client.get_registered_model(name)
		except self._mlflow.exceptions.MlflowException:
			client.create_registered_model(name)
		version = client.create_model_version(
			name,
			self._mlflow.get_artifact_uri(artifact_path),
			run_id=self.run_id,
			tags={"git.commit": git_state()["commit"] or ""},
		)
		if alias:
			client.set_registered_model_alias(name, alias, version.version)


def _import_mlflow() -> Any:
	try:
		import mlflow
	except ImportError as exc:  # pragma: no cover - depends on the venv
		raise RuntimeError("tracking was requested but mlflow is not installed (uv sync --extra tracking)") from exc
	if not hasattr(mlflow, "start_run"):
		# Only the ./mlflow store directory (a namespace package) was found: the real package is not installed here.
		raise RuntimeError("tracking was requested but mlflow is not installed (uv sync --extra tracking)")
	return mlflow


@contextmanager
def tracked_run(
	experiment: str,
	run_name: str | None = None,
	*,
	enabled: bool = True,
	params: Mapping[str, Any] | None = None,
	tags: Mapping[str, Any] | None = None,
	datasets: Mapping[str, Path | str] | None = None,
	note: str | None = None,
) -> Iterator[NullRun]:
	"""Open an MLflow run recording code and data lineage; yield a no-op handle when ``enabled`` is false.

	``datasets`` maps a role (``train``, ``val``, ``suite`` ...) to a dataset directory; each one's manifest digest is
	tagged as ``dataset.<role>.digest`` so runs can be grouped by exact data version. The run is ended FAILED if the
	body raises.
	"""
	if not enabled or os.environ.get("ARMIGO_MLFLOW") == "0":
		yield NullRun()
		return
	mlflow = _import_mlflow()
	if "ARMIGO_MLFLOW_URI" not in os.environ:
		(store_dir() / "artifacts").mkdir(parents=True, exist_ok=True)
	mlflow.set_tracking_uri(tracking_uri())
	if "ARMIGO_MLFLOW_URI" not in os.environ and mlflow.get_experiment_by_name(experiment) is None:
		mlflow.create_experiment(experiment, artifact_location=(store_dir() / "artifacts").as_uri())
	mlflow.set_experiment(experiment)
	with mlflow.start_run(run_name=run_name) as run:
		handle = MlflowRun(mlflow, run)
		git = git_state()
		handle.set_tags({"git.commit": git["commit"], "git.branch": git["branch"], "git.dirty": git["dirty"]})
		for role, directory in (datasets or {}).items():
			ref = dataset_ref(Path(directory))
			handle.set_tags(
				{
					f"dataset.{role}.name": ref["name"],
					f"dataset.{role}.digest": ref["digest"],
					f"dataset.{role}.files": ref["file_count"],
					f"dataset.{role}.path": ref["path"],
				}
			)
		if params:
			handle.log_params(params)
		if tags:
			handle.set_tags(tags)
		if note:
			handle.set_tags({"mlflow.note.content": note})
		yield handle
