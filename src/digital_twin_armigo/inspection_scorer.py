"""Anomaly scorers for the main simulator venv (numpy only, no torch).

The image detector lives in a separate environment (``E:/venvs/armigo-anomaly``). :class:`SubprocessAnomalyScorer`
keeps ``scripts/anomaly/score_server.py`` running and exchanges one line-delimited JSON request/response per crop.
:class:`FixedScorer` is the in-process stand-in for tests and offline replays, and :class:`AnomalyScorer` is the
small interface the inspection pipeline depends on.
"""

from __future__ import annotations

import base64
import contextlib
import json
import math
import os
import queue
import subprocess
import tempfile
import threading
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Protocol, Self

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ANOMALY_PYTHON = Path("E:/venvs/armigo-anomaly/Scripts/python.exe")
DEFAULT_SERVER_SCRIPT = REPO_ROOT / "scripts" / "anomaly" / "score_server.py"


class ScoreServerError(RuntimeError):
	"""A protocol or read failure of the anomaly score server (still a ``RuntimeError`` for callers)."""


class AnomalyScorer(Protocol):
	"""What the inspection pipeline needs from an anomaly detector."""

	def score(self, camera: str, image: np.ndarray) -> float:
		"""Anomaly score of one square RGB uint8 crop (higher is worse)."""

	def threshold(self, camera: str) -> float:
		"""Score threshold chosen for ``camera`` on validation normals."""


class SubprocessAnomalyScorer:
	"""Score crops through ``score_server.py`` in the anomaly environment (one request per crop).

	The server is started lazily in :meth:`__init__` and stopped by :meth:`close` (also on context-manager exit);
	reads use a reader thread plus a queue so a wedged server cannot block forever on Windows pipes.
	"""

	def __init__(
		self,
		python_exe: str | Path,
		server_script: str | Path,
		models: Mapping[str, str | Path],
		*,
		extra_args: Iterable[str] = (),
		timeout_s: float = 120.0,
	) -> None:
		self._timeout = float(timeout_s)
		self._lock = threading.Lock()
		self._next_id = 0
		self._closed = False
		self._proc: subprocess.Popen[str] | None = None
		self._cameras: dict[str, dict] = {}
		self._lines: queue.Queue[object] = queue.Queue()
		fd, log_name = tempfile.mkstemp(prefix="armigo-anomaly-", suffix=".log")
		os.close(fd)
		self._log_path = Path(log_name)
		cmd = [str(python_exe), str(server_script)]
		for camera, model_dir in models.items():
			cmd += ["--model", f"{camera}={model_dir}"]
		cmd += list(extra_args)
		try:
			# The child inherits its own handle; the parent can close its copy once Popen has returned.
			with open(self._log_path, "wb") as log_file:
				self._proc = subprocess.Popen(
					cmd,
					stdin=subprocess.PIPE,
					stdout=subprocess.PIPE,
					stderr=log_file,
					text=True,
					encoding="utf-8",
					errors="replace",
					bufsize=1,
				)
			reader = threading.Thread(target=self._read_lines, name="anomaly-score-reader", daemon=True)
			reader.start()
			ready = self._read_message(self._timeout, context="waiting for the ready line")
			if not isinstance(ready, dict) or not ready.get("ready"):
				raise ScoreServerError(f"anomaly score server did not send a ready line: {ready!r}")
			self._cameras = dict(ready.get("cameras") or {})
		except Exception:
			self.close()
			raise

	# --- process plumbing -------------------------------------------------------------------------------------

	def _read_lines(self) -> None:
		proc = self._proc
		assert proc is not None and proc.stdout is not None
		try:
			for line in proc.stdout:
				self._lines.put(line)
		except (OSError, ValueError) as exc:  # surfaced by _read_message
			self._lines.put(exc)
		finally:
			self._lines.put(None)

	def _stderr_tail(self, limit: int = 4000) -> str:
		try:
			with open(self._log_path, "rb") as log:
				log.seek(0, os.SEEK_END)
				log.seek(max(0, log.tell() - limit))
				text = log.read().decode("utf-8", "replace").strip()
		except OSError:
			return ""
		return f"\n--- anomaly score server stderr ---\n{text}" if text else ""

	def _read_message(self, timeout: float, context: str = "") -> object:
		where = f" ({context})" if context else ""
		try:
			item = self._lines.get(timeout=timeout)
		except queue.Empty:
			raise ScoreServerError(
				f"anomaly score server timed out after {timeout:.1f}s{where}{self._stderr_tail()}"
			) from None
		if item is None:
			code = None
			if self._proc is not None:
				try:
					code = self._proc.wait(timeout=1.0)
				except subprocess.TimeoutExpired:
					code = self._proc.poll()
			raise ScoreServerError(f"anomaly score server exited (code {code}){where}{self._stderr_tail()}")
		if isinstance(item, Exception):
			raise ScoreServerError(f"anomaly score server read failed: {item}{where}{self._stderr_tail()}") from None
		try:
			return json.loads(item)
		except json.JSONDecodeError as exc:
			raise ScoreServerError(f"anomaly score server sent invalid JSON {item!r}: {exc}{where}") from None

	def _send(self, obj: dict) -> None:
		assert self._proc is not None and self._proc.stdin is not None
		try:
			self._proc.stdin.write(json.dumps(obj) + "\n")
			self._proc.stdin.flush()
		except (OSError, ValueError) as exc:
			raise ScoreServerError(
				f"anomaly score server is not accepting requests: {exc}{self._stderr_tail()}"
			) from None

	def _camera(self, camera: str) -> dict:
		try:
			return self._cameras[camera]
		except KeyError:
			known = ", ".join(sorted(self._cameras)) or "none"
			raise KeyError(f"unknown camera {camera!r} (loaded: {known})") from None

	# --- protocol ---------------------------------------------------------------------------------------------

	@property
	def cameras(self) -> dict[str, dict]:
		"""Per-camera metadata sent in the server's ready line."""
		return {camera: dict(entry) for camera, entry in self._cameras.items()}

	def score(self, camera: str, image: np.ndarray) -> float:
		return self.score_with_details(camera, image)[0]

	def score_with_details(self, camera: str, image: np.ndarray) -> tuple[float, dict[str, float]]:
		"""Read the fused score and any calibrated head scores from one inference request."""
		img = np.asarray(image)
		if img.dtype != np.uint8:
			raise TypeError(f"expected a uint8 image, got {img.dtype}")
		if img.ndim != 3 or img.shape[2] != 3:
			raise ValueError(f"expected an (H, W, 3) RGB image, got {img.shape}")
		img = np.ascontiguousarray(img)
		with self._lock:
			if self._closed or self._proc is None:
				raise ScoreServerError("anomaly score server is closed")
			self._next_id += 1
			request_id = self._next_id
			self._send(
				{
					"id": request_id,
					"camera": camera,
					"shape": list(img.shape),
					"rgb_b64": base64.b64encode(img.tobytes()).decode("ascii"),
				}
			)
			response = self._read_message(self._timeout, context=f"for request {request_id}")
		if not isinstance(response, dict):
			raise ScoreServerError(f"anomaly score server sent {response!r}, expected a JSON object")
		if response.get("id") != request_id:
			raise ScoreServerError(
				f"anomaly score server answered request {response.get('id')!r}, expected {request_id!r}"
			)
		if "error" in response:
			raise ScoreServerError(f"anomaly score server error: {response['error']}")
		if "score" not in response:
			raise ScoreServerError(f"anomaly score server response has no score: {response!r}")
		details = response.get("head_scores")
		heads = (
			{
				key: float(value)
				for key, value in details.items()
				if key in ("cls", "seg") and type(value) in (int, float) and math.isfinite(value)
			}
			if isinstance(details, dict)
			else {}
		)
		return float(response["score"]), heads

	def threshold(self, camera: str) -> float:
		return float(self._camera(camera)["threshold"])

	# --- lifecycle --------------------------------------------------------------------------------------------

	def close(self) -> None:
		if self._closed:
			return
		self._closed = True
		proc = self._proc
		if proc is not None:
			try:
				if proc.poll() is None and proc.stdin is not None:
					try:
						proc.stdin.write('{"cmd": "quit"}\n')
						proc.stdin.flush()
					except (OSError, ValueError):
						pass
				try:
					proc.wait(timeout=5.0)
				except subprocess.TimeoutExpired:
					proc.kill()
					try:
						proc.wait(timeout=5.0)
					except subprocess.TimeoutExpired:
						pass
			finally:
				for stream in (proc.stdin, proc.stdout):
					if stream is not None:
						try:
							stream.close()
						except OSError:
							pass
		self._cleanup_log()

	def _cleanup_log(self) -> None:
		log_path = getattr(self, "_log_path", None)
		if log_path is not None:
			try:
				log_path.unlink()
			except OSError:
				pass

	def __enter__(self) -> Self:
		return self

	def __exit__(self, *exc_info: object) -> None:
		self.close()

	def __del__(self) -> None:
		with contextlib.suppress(Exception):
			self.close()


class FixedScorer:
	"""Scorer from a plain callable, for tests and offline replays."""

	def __init__(self, score_fn: Callable[[str, np.ndarray], float], thresholds: Mapping[str, float]) -> None:
		self._score_fn = score_fn
		self._thresholds = {camera: float(value) for camera, value in thresholds.items()}

	def _check(self, camera: str) -> None:
		if camera not in self._thresholds:
			known = ", ".join(sorted(self._thresholds)) or "none"
			raise KeyError(f"unknown camera {camera!r} (loaded: {known})")

	def score(self, camera: str, image: np.ndarray) -> float:
		self._check(camera)
		return float(self._score_fn(camera, image))

	def threshold(self, camera: str) -> float:
		self._check(camera)
		return self._thresholds[camera]


def default_scorer(models_root: str | Path) -> SubprocessAnomalyScorer:
	"""Scorer for per-camera models exported by ``train_eval.py --export-scopes patchcore/top,...``."""
	root = Path(models_root)
	return SubprocessAnomalyScorer(
		DEFAULT_ANOMALY_PYTHON,
		DEFAULT_SERVER_SCRIPT,
		{"top": root / "patchcore_top", "wrist_cam": root / "patchcore_wrist_cam"},
	)


def default_sup_scorer(spec_path: str | Path) -> SubprocessAnomalyScorer:
	"""Scorer for the supervised cross detector written by ``scripts/anomaly/export_sup.py``.

	Scores are standardised views ``u`` (see ``scripts/anomaly/sup_scorer.py``) shared by both cameras, so the detection
	threshold of every camera is ``0.0`` unless the caller overrides it (``HeuristicPlanner(thresholds=...)``).
	"""
	return SubprocessAnomalyScorer(
		DEFAULT_ANOMALY_PYTHON, DEFAULT_SERVER_SCRIPT, {}, extra_args=("--sup", str(spec_path))
	)
