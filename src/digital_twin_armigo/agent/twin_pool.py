"""Process-based simulation workers that share one detector subprocess owned by the parent.

The RTX GPU is mostly held by the LLM server, so spawning one detector per worker does not fit. Workers send crops to
the parent through a queue; a single broker thread feeds the one real scorer and answers on per-worker queues.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import threading
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import numpy as np

LOG = logging.getLogger(__name__)
SCORE_TIMEOUT_S = 300.0


class ScorerBroker:
	"""Parent side: serialises scoring requests from all workers onto one real scorer."""

	def __init__(self, scorer_factory, clients, ctx):
		self.factory = scorer_factory
		self.requests = ctx.Queue()
		self.responses = [ctx.Queue() for _ in range(clients)]
		self.slot = ctx.Value("i", 0)
		self.scorer = None
		self.thread = threading.Thread(target=self._serve, name="armigo-scorer-broker", daemon=True)
		self.thread.start()

	def _serve(self):
		try:
			self.scorer = self.factory()  # start the detector (and load its weights) before the first episode asks
		except Exception:
			LOG.exception("could not start the shared detector; retrying on first request")
		while (item := self.requests.get()) is not None:
			client, request_id, op, camera, image = item
			try:
				if self.scorer is None:
					self.scorer = self.factory()
				if op == "threshold":
					payload = self.scorer.threshold(camera)
				else:
					payload = self.scorer.score_with_details(camera, image)
				self.responses[client].put((request_id, True, payload))
			except Exception as exc:  # noqa: BLE001
				# Re-raised in the worker as ScoreServerError so the skill reports DETECTOR_UNAVAILABLE.
				self.responses[client].put((request_id, False, f"{type(exc).__name__}: {exc}"))

	def close(self):
		self.requests.put(None)
		self.thread.join(timeout=10)
		if self.scorer is not None:
			self.scorer.close()


class ProxyScorer:
	"""Worker side: the scorer interface ``TwinMujoco`` uses, answered by the parent's broker."""

	def __init__(self, requests, responses, client):
		self.requests = requests
		self.responses = responses
		self.client = client
		self.next_id = 0
		self.thresholds = {}

	def _call(self, op, camera, image=None):
		from ..inspection_scorer import ScoreServerError

		self.next_id += 1
		request_id = self.next_id
		self.requests.put((self.client, request_id, op, camera, image))
		try:
			answer = self.responses.get(timeout=SCORE_TIMEOUT_S)
		except queue.Empty:
			raise ScoreServerError("shared detector did not answer in time") from None
		if answer[0] != request_id:
			raise ScoreServerError("shared detector answered a different request")
		if not answer[1]:
			raise ScoreServerError(answer[2])
		return answer[2]

	def threshold(self, camera):
		if camera not in self.thresholds:
			self.thresholds[camera] = float(self._call("threshold", camera))
		return self.thresholds[camera]

	def score_with_details(self, camera, image):
		score, heads = self._call("score", camera, np.ascontiguousarray(image))
		return float(score), heads

	def score(self, camera, image):
		return self.score_with_details(camera, image)[0]

	def close(self):
		pass


_WORKER = {}


def _init_worker(config, requests, responses, slot):
	from .twin_mcp import TwinService

	with slot.get_lock():
		client = slot.value
		slot.value += 1
	_WORKER["service"] = TwinService(**config, scorer=ProxyScorer(requests, responses[client], client), workers=1)


def _worker_task(task):
	return _WORKER["service"].execute(task)


class EpisodePool:
	"""Spawned worker processes executing ``TwinService.execute`` tasks; rebuilt if a worker dies."""

	def __init__(self, workers, config, scorer_factory):
		self.workers = workers
		self.config = config
		self.ctx = mp.get_context("spawn")
		self.broker = ScorerBroker(scorer_factory, workers * 2, self.ctx)
		self.lock = threading.Lock()
		self.executor = self._build()

	def _build(self):
		return ProcessPoolExecutor(
			self.workers,
			mp_context=self.ctx,
			initializer=_init_worker,
			initargs=(self.config, self.broker.requests, self.broker.responses, self.broker.slot),
		)

	def submit(self, task):
		"""Future for ``task``; a crashed pool is replaced once (the failed task still fails)."""
		with self.lock:
			try:
				return self.executor.submit(_worker_task, task)
			except BrokenProcessPool:
				LOG.warning("worker pool broke; starting a new one")
				self.executor.shutdown(wait=False, cancel_futures=True)
				self.broker.slot.value = 0
				self.executor = self._build()
				return self.executor.submit(_worker_task, task)

	def close(self):
		self.executor.shutdown(wait=True, cancel_futures=True)
		self.broker.close()
