"""Local MCP entry point for six-face simulation, policy search, and episode export."""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import re
import statistics
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import anyio
import numpy as np
from langgraph.checkpoint.memory import InMemorySaver
from mcp.server.fastmcp import Context, FastMCP, Image

from ..inspection_scorer import default_sup_scorer
from .llm import DEFAULT_MODEL, OpenAICompatLLM
from .twin_graph import Policy, build_graph, search_policies
from .twin_jobs import (
	ACTIVE,
	CANCELLED,
	COMPLETE,
	FAILED,
	INTERRUPTED,
	RESUMABLE,
	TERMINAL,
	JobCancelled,
	JobHooks,
	JobStore,
	open_saver,
)
from .twin_mujoco import PROFILE, ROOT, TwinMujoco, file_digest, load_suite, sample_case
from .twin_pool import EpisodePool
from .twin_record import EpisodeTrace, export_lerobot, write_json
from .twin_resume import live_checkpoint_ids, load_plan, restore
from .twin_search import analyze_failures, dev_suite

LOG = logging.getLogger(__name__)
# First 8 cases of the generated dev suite (seeds from 700001, disjoint from DEMO/dev4/bench16).
DEV_CASES = tuple(dev_suite(8))
# Policy-search episode cap = SEARCH_EPISODES_PER_BUDGET * (budget + 1); the 54-policy grid needs at least 54.
SEARCH_EPISODES_PER_BUDGET = 50
_JOB = object()  # default ``hook``: use the calling job's cancel flag, if any
DEMO_CASES = ((187271, False, 5), (187271, True, 5), (187272, False, 5), (187272, True, 5))
END_POSITION_TOL_RAD = 0.01
END_SPEED_TOL_RAD_S = 0.05


def _robot_state(cell):
	return np.stack((cell.joints(), cell.data.qvel[cell.dofadr]))


@dataclass(frozen=True)
class ReplayEndpoints:
	start: np.ndarray
	end: np.ndarray
	first_command: np.ndarray
	last_command: np.ndarray
	views: tuple[tuple[str, tuple[int, ...]], ...]


class CarryReplayCheck:
	def __init__(self):
		self.reference: ReplayEndpoints | None = None

	def finish(self, backend, start, result):
		if not result["student_eligible"]:
			return {"accepted": False, "reason": "episode_ineligible"} if self.reference is not None else None
		rows = backend.record.rows
		views = tuple(
			(obs["camera"], tuple(obs["faces"]))
			for event in backend.record.events
			if event["kind"] == "view"
			for obs in (event["observation"],)
		)
		# Compare the targets at the precision of the action archive.
		current = ReplayEndpoints(
			start, _robot_state(backend.cell), rows[0][1].astype(np.float32), rows[-1][1].astype(np.float32), views
		)
		reference = self.reference
		if reference is None:
			self.reference = current
			return None
		pos_error, speed_error = np.max(np.abs(current.end - reference.end), axis=1)
		if not (math.isfinite(pos_error) and math.isfinite(speed_error)):
			return {"accepted": False, "reason": "nonfinite_endpoint"}
		checks = {
			"smoothed_moves": backend.expert.smoothed_moves,
			"start_state_matches": bool(np.array_equal(current.start, reference.start)),
			"boundary_commands_match": bool(
				np.array_equal(current.first_command, reference.first_command)
				and np.array_equal(current.last_command, reference.last_command)
			),
			"views_match": current.views == reference.views,
			"final_position_error_rad": float(pos_error),
			"final_speed_error_rad_s": float(speed_error),
		}
		checks["accepted"] = bool(
			checks["smoothed_moves"]
			and checks["start_state_matches"]
			and checks["boundary_commands_match"]
			and checks["views_match"]
			and pos_error <= END_POSITION_TOL_RAD
			and speed_error <= END_SPEED_TOL_RAD_S
		)
		return checks


class TwinService:
	def __init__(
		self,
		out,
		detector,
		bottom_profile=None,
		llm=None,
		*,
		workers=1,
		scorer=None,
		llm_spec=None,
		owner=True,
		max_queue=None,
		sensing="truth",
	):
		self.sensing = sensing  # TwinMujoco sensing mode of every episode this service runs
		self.out = Path(out).resolve()
		self.detector = Path(detector).resolve()
		spec = json.loads(self.detector.read_text(encoding="utf-8"))
		self.model_hash = file_digest(self.detector)
		self.crop_size = int(spec["crop_size"])
		self.bottom_threshold = None
		self.bottom_profile = Path(bottom_profile) if bottom_profile else None
		if bottom_profile:
			raw = json.loads(Path(bottom_profile).read_text(encoding="utf-8"))
			if raw["profile"] != PROFILE or raw["detector_sha256"] != self.model_hash:
				raise ValueError("bottom calibration belongs to a different detector or camera profile")
			self.bottom_threshold = float(raw["threshold"])
			if not math.isfinite(self.bottom_threshold):
				raise ValueError("invalid bottom threshold")
		self.llm_spec = llm_spec
		if llm is None and llm_spec:
			llm = OpenAICompatLLM(**llm_spec)
		self.llm = llm
		self.scorer = scorer
		self.workers = max(1, int(workers))
		self.owner = owner  # Only the server process recovers stale job rows; spawned workers never do.
		self.max_queue = max_queue or max(8, 4 * self.workers)
		# Job threads: one in-process MuJoCo thread, or one dispatcher per worker process.
		self.worker = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="armigo-job")
		self._store = None
		self._pool = None
		self._pool_lock = threading.Lock()
		self._ctx = threading.local()

	@property
	def store(self):
		if self._store is None:
			self._store = JobStore(self.out / "jobs.sqlite")
			if self.owner:
				recovered = self._store.interrupt_stale()
				if recovered:
					LOG.warning("%d job(s) were left running by a previous server and are now interrupted", recovered)
		return self._store

	def recover(self):
		return self.store

	@property
	def pool(self):
		"""Process workers for ``workers > 1`` (``None`` means everything runs on this process's job thread)."""
		if self.workers == 1:
			return None
		with self._pool_lock:
			if self._pool is None:
				config = {
					"out": str(self.out),
					"detector": str(self.detector),
					"bottom_profile": str(self.bottom_profile) if self.bottom_profile else None,
					"llm_spec": self.llm_spec,
					"owner": False,
					"sensing": self.sensing,
				}
				self._pool = EpisodePool(self.workers, config, lambda: default_sup_scorer(self.detector))
			return self._pool

	def close(self):
		if self._store is not None:
			self._store.request_interrupt()
		self.worker.shutdown(wait=True, cancel_futures=True)
		if self._pool is not None:
			self._pool.close()
		if self._store is not None:
			self._store.interrupt_stale()
			self._store.close()
		if self.scorer:
			self.scorer.close()
		if self.llm:
			self.llm.close()

	def run(
		self,
		case,
		policy,
		path,
		*,
		video,
		use_llm=False,
		fault=None,
		hook=_JOB,
		thread_id=None,
		resume=None,
		carry_check=None,
	):
		"""Run one episode in this process. ``hook`` is called per frame and may raise ``JobCancelled``.

		``thread_id`` persists the inner LangGraph state in ``graph.sqlite``; ``resume`` is a ``ResumePlan``.
		"""
		if self.scorer is None:
			self.scorer = default_sup_scorer(self.detector)
		if hook is _JOB:
			hook = self._job_hook(report=False)
		layout = sample_case(*case)
		trace = EpisodeTrace(path, video=video)
		backend = saver = None
		try:
			write_json(
				trace.path / "task.json",
				{"case": list(case), "policy": asdict(policy), "video": video, "use_llm": use_llm},
			)
			backend = TwinMujoco(
				layout,
				self.scorer,
				trace,
				llm=self.llm if use_llm else None,
				bottom_threshold=self.bottom_threshold,
				crop_size=self.crop_size,
				fault=fault,
				on_frame=hook,
				smooth_carry=carry_check is not None and carry_check.reference is not None,
				sensing=self.sensing,
			)
			start = _robot_state(backend.cell) if carry_check is not None else None
			link = None
			if resume is not None:
				link = restore(backend, resume)
				write_json(trace.path / "resume.json", link)
			if thread_id:
				saver = open_saver(self.out / "graph.sqlite")
			elif resume is not None:
				saver = InMemorySaver()
			graph = build_graph(backend, policy, saver)
			config = {"recursion_limit": 64}
			if saver is not None:
				config["configurable"] = {"thread_id": thread_id or "resume"}
			if resume is None:
				result = graph.invoke({}, config=config)["result"]
			else:
				graph.update_state(config, resume.graph_state(), as_node="prepare")
				result = graph.invoke(None, config=config)["result"]
				result["student_eligible"] = False
			if carry_check is not None and (checked := carry_check.finish(backend, start, result)) is not None:
				result["carry_smoothing"] = checked
				result["student_eligible"] = checked["accepted"]
			trace.finish(
				result,
				{
					"engine": "mujoco",
					"sensing": self.sensing,
					"camera_profile": PROFILE,
					"detector_sha256": self.model_hash,
					"policy": asdict(policy),
					"llm_enabled": use_llm,
					"bottom_threshold": self.bottom_threshold,
					"model": self.llm.model if use_llm else None,
					**({"resumed_from": link} if link else {}),
				},
			)
			return result
		except BaseException:
			trace.close()
			raise
		finally:
			if backend:
				backend.close()
			if saver is not None and hasattr(saver, "conn"):
				saver.conn.close()

	def run_for_vla(self, case, policy, path, *, video, use_llm):
		check = CarryReplayCheck()
		baseline = self.run(case, policy, path, video=video, use_llm=use_llm, carry_check=check)
		selection = {"baseline": path.name, "selected": None}
		if check.reference is not None:
			target = path.parent / "smoothed" / path.name
			candidate = self.run(case, policy, target, video=video, use_llm=use_llm, carry_check=check)
			relative = target.relative_to(path.parent).as_posix()
			selection.update(
				candidate=relative,
				selected=relative if candidate["student_eligible"] else path.name,
				verification=candidate["carry_smoothing"],
			)
		return baseline, selection

	def _job_hook(self, report=True):
		job_id = getattr(self._ctx, "job_id", None)
		return JobHooks(self.store, job_id, report=report) if job_id else None

	def _task(self, case, policy, path, *, video, use_llm=False, progress=False, resume=None, thread_id=None):
		return {
			"case": list(case),
			"policy": asdict(policy),
			"path": str(path),
			"video": video,
			"use_llm": use_llm,
			"job_id": getattr(self._ctx, "job_id", None),
			"progress": progress,
			"thread_id": thread_id,
			"resume": resume,
		}

	def execute(self, task):
		"""Run a picklable task dict in this process (the entry point used by worker processes)."""
		hook = None
		if task["job_id"]:
			hook = JobHooks(self.store, task["job_id"], report=task["progress"])
			hook.check()  # cancelled while waiting for a free worker
		resume = None
		if task["resume"]:
			saver = open_saver(self.out / "graph.sqlite")
			try:
				resume = load_plan(
					task["resume"]["parent_dir"], saver=saver, parent_job_id=task["resume"]["parent_job_id"]
				)
			finally:
				saver.conn.close()
		return self.run(
			tuple(task["case"]),
			Policy(**task["policy"]),
			Path(task["path"]),
			video=task["video"],
			use_llm=task["use_llm"],
			hook=hook,
			thread_id=task["thread_id"],
			resume=resume,
		)

	def _dispatch(self, task):
		pool = self.pool
		return self.execute(task) if pool is None else pool.submit(task).result()

	def run_batch(self, items, *, video=False, use_llm=False):
		"""Evaluate ``(case, Policy, out_path)`` items in parallel (serially when ``workers == 1``).

		Results keep the order of ``items``. The first failure (including a cancelled job) is re-raised.
		"""
		tasks = [self._task(case, policy, path, video=video, use_llm=use_llm) for case, policy, path in items]
		pool = self.pool
		if pool is None:
			return [self.execute(task) for task in tasks]
		futures = [pool.submit(task) for task in tasks]
		try:
			results = []
			for done, future in enumerate(futures, 1):
				results.append(future.result())
				if tasks[0]["job_id"]:
					self.store.progress(tasks[0]["job_id"], "batch", None, done, len(futures))
			return results
		finally:
			for future in futures:
				future.cancel()

	def _submit(self, request_id, arguments, operation, folder, parent=None):
		"""Persist a job row and run ``operation(path)`` on a job thread. Repeats of a request_id return its job."""
		if not request_id or len(request_id) > 128:
			raise ValueError("request_id must contain 1 to 128 characters")
		kind = arguments["kind"] if isinstance(arguments, dict) else arguments[0]
		job_id, created = self.store.create(
			request_id, kind, arguments, root=self.out / folder, parent_job_id=parent, max_active=self.max_queue
		)
		if created:
			self.worker.submit(self._run_job, job_id, operation)
		return job_id

	def _run_job(self, job_id, operation):
		store = self.store
		if not store.mark_running(job_id):
			return  # cancelled while queued
		self._ctx.job_id = job_id
		try:
			store.finish(job_id, COMPLETE, operation(Path(store.get(job_id)["path"])))
		except JobCancelled as exc:
			store.finish(job_id, INTERRUPTED if exc.interrupted else CANCELLED)
		except Exception:
			LOG.exception("Twin job %s failed", job_id)
			store.finish(job_id, FAILED, error_code="JOB_EXECUTION_FAILED")
		finally:
			self._ctx.job_id = None

	def start(self, request_id, case, policy, record, use_llm):
		if use_llm and self.llm is None:
			raise ValueError("start armigo-mcp with --llm-url to enable LLM skill selection")
		arguments = {
			"kind": "episode",
			"case": list(case),
			"policy": asdict(policy),
			"record": record,
			"use_llm": use_llm,
		}
		return self._submit(request_id, arguments, self._episode_operation(case, policy, record, use_llm), "episodes")

	def _episode_operation(self, case, policy, record, use_llm, resume=None):
		def job(path):
			task = self._task(
				case,
				policy,
				path,
				video=record,
				use_llm=use_llm,
				progress=True,
				resume=resume,
				thread_id=self._ctx.job_id,
			)
			return self._dispatch(task)

		return job

	def resume(self, job_id):
		"""Restart an interrupted/failed/cancelled episode from its latest persisted skill-boundary checkpoint."""
		row = self.store.get(job_id)
		if row is None:
			raise ValueError("unknown job")
		if row["kind"] not in ("episode", "resume"):
			raise ValueError("only episode jobs can be resumed")
		if row["status"] not in RESUMABLE:
			raise ValueError(f"job is {row['status']}; only interrupted, failed or cancelled jobs can be resumed")
		children = self.store.children(job_id)
		for child in children:
			if child["status"] in (*ACTIVE, COMPLETE):
				return child["job_id"]
		args = row["arguments"]["origin"] if row["kind"] == "resume" else row["arguments"]
		case, policy = tuple(args["case"]), Policy(**args["policy"])
		source, plan_dir = row, None
		while source is not None:
			path = Path(source["path"])
			if (path / "events.jsonl").is_file() and live_checkpoint_ids(path / "events.jsonl"):
				plan_dir = path
				break
			parent = source["arguments"].get("resume_of") if source["kind"] == "resume" else None
			source = self.store.get(parent) if parent else None
		resume = {"parent_dir": str(plan_dir), "parent_job_id": source["job_id"]} if plan_dir else None
		arguments = {
			"kind": "resume",
			"resume_of": job_id,
			"attempt": len(children),
			"origin": {k: args[k] for k in ("case", "policy", "record", "use_llm")},
			"from_checkpoint": bool(resume),
		}
		operation = self._episode_operation(case, policy, args["record"], args["use_llm"], resume)
		return self._submit(f"resume:{job_id}:{len(children)}", arguments, operation, "episodes", parent=job_id)

	def cancel(self, job_id):
		if self.store.get(job_id) is None:
			raise ValueError("unknown job")
		self.store.cancel(job_id)
		return self.status(job_id)

	def optimize(self, request_id, budget):
		if not 1 <= budget <= 8:
			raise ValueError("policy budget must be in [1, 8]")
		cap = SEARCH_EPISODES_PER_BUDGET * (budget + 1)

		def job(path):
			counter = itertools.count()

			def evaluate_batch(items):
				jobs = [(case, policy, path / "episodes" / f"{next(counter):04d}") for policy, case in items]
				return self.run_batch(jobs)  # fans out over the worker pool when --workers > 1

			report = search_policies(evaluate_batch, cap)
			report["llm_analysis"] = (
				analyze_failures(self.llm, report) if self.llm is not None else {"status": "skipped", "error": "no llm"}
			)
			write_json(path / "report.json", report)
			write_json(path / "policy.json", report["selected"]["policy"])
			return report

		return self._submit(request_id, ("optimize", budget), job, "search")

	def status(self, job_id):
		row = self.store.get(job_id)
		if row is None:
			raise ValueError("unknown job")
		state = row["status"]
		out = {
			"job_id": job_id,
			"status": state,
			"kind": row["kind"],
			"progress": row["progress"],
			"created_at": row["created_at"],
			"updated_at": row["updated_at"],
		}
		if row["arguments"].get("resume_of") if isinstance(row["arguments"], dict) else None:
			out["resume_of"] = row["arguments"]["resume_of"]
		if state in ACTIVE:
			out["cancel_requested"] = bool(row["stop"])
		elif state in RESUMABLE:
			out["resumable"] = row["kind"] in ("episode", "resume")
			if state == FAILED:
				out["code"] = row["error_code"]
		elif row["kind"] == "optimize":
			out["search"] = row["result"]
		else:
			keys = (
				"label",
				"destination",
				"active_branch",
				"rewinds",
				"retries",
				"failures",
				"active_views",
				"views",
				"sim_frames",
				"evidence",
				"cleared_faces",
				"placement_evidence",
				"smoothness",
			)
			out["episode"] = {key: row["result"][key] for key in keys if key in row["result"]}
		if row["kind"] in ("episode", "resume") and (resumed := [c["job_id"] for c in self.store.children(job_id)]):
			out["resumed_by"] = resumed
		return out


def aggregate(results):
	from collections import Counter

	failures = Counter(failure["code"] for result in results for failure in result["failures"])
	llm = [x.get("llm") or {} for x in results]
	consulted = sum(x.get("consulted", 0) for x in llm)
	adopted = sum(x.get("adopted", 0) for x in llm)
	reasons = Counter()
	tokens = []
	for x in llm:
		reasons.update(x.get("fallback_reasons") or {})
		tokens.extend(x.get("prompt_tokens") or [])
	return {
		"llm_consulted": consulted,
		"llm_adopted": adopted,
		"llm_adoption_rate": round(adopted / consulted, 4) if consulted else None,
		"fallback_reasons": dict(sorted(reasons.items())),
		"llm_prompt_tokens_median": statistics.median(tokens) if tokens else None,
		"llm_prompt_tokens_max": max(tokens) if tokens else None,
		"llm_latency_s_total": round(sum(x.get("latency_s_total", 0.0) for x in llm), 3),
		"episodes": len(results),
		"successes": sum(x["verdict"] == "SUCCESS" for x in results),
		"labels_correct": sum(x["label_correct"] for x in results),
		"unknown": sum(x["label"] == "unknown" for x in results),
		"sim_frames": sum(x["sim_frames"] for x in results),
		"views": sum(x["views"] for x in results),
		"rewinds": sum(x["rewinds"] for x in results),
		"retries": sum(x.get("retries", 0) for x in results),
		"failures": [{"code": code, "count": count} for code, count in sorted(failures.items())],
		"smoothness": aggregate_smoothness([x.get("smoothness") or {} for x in results]),
	}


def aggregate_smoothness(rows):
	"""Suite view of the per-episode ``smoothness`` blocks: medians of the rates, worst peak, totals of counts."""
	rows = [r for r in rows if "jerk_rms_meas" in r]
	if not rows:
		return {}

	def median(key):
		return round(statistics.median(r[key] for r in rows), 3)

	peak = max(rows, key=lambda r: r["jerk_peak_meas"])
	return {
		"episodes": len(rows),
		"jerk_rms_cmd_median": median("jerk_rms_cmd"),
		"jerk_rms_meas_median": median("jerk_rms_meas"),
		"jerk_peak_meas_max": peak["jerk_peak_meas"],
		"jerk_peak_phase": peak.get("jerk_peak_phase"),
		"stops_median": median("stops"),
		"stops_total": sum(r["stops"] for r in rows),
		"still_fraction_median": median("still_fraction"),
		"clipped_frames_total": sum(r.get("clipped_frames", 0) for r in rows),
	}


def create_mcp(service: TwinService, cases, host="127.0.0.1", port=8765):
	mcp = FastMCP("Armigo Six-Face Twin", host=host, port=port)
	opaque_cases = {uuid.uuid4().hex: case for case in cases}

	@mcp.tool()
	def list_cases() -> dict[str, Any]:
		"""List opaque scene IDs configured by the evaluator. Geometry is observed at execution time."""
		return {"case_ids": list(opaque_cases), "engine": "mujoco", "camera_profile": PROFILE}

	@mcp.tool()
	def start_episode(
		request_id: str,
		case_id: str,
		order: str = "wrist_first",
		max_wrist_views: int = 2,
		max_bottom_views: int = 2,
		max_rewinds: int = 1,
		record: bool = True,
		use_llm: bool = False,
	) -> dict[str, Any]:
		"""Run one inspected classification and physical sort from a bounded generic policy."""
		if case_id not in opaque_cases:
			raise ValueError("unknown case")
		policy = Policy(order, max_bottom_views, max_wrist_views, max_rewinds)
		return {"job_id": service.start(request_id, opaque_cases[case_id], policy, record, use_llm)}

	@mcp.tool()
	def start_policy_search(request_id: str, budget: int = 3) -> dict[str, Any]:
		"""Successive-halving search over all 54 generic policies on a generated development suite.

		``budget`` 1-8 caps the search at 50 * (budget + 1) episodes (default 3 = 200). The default policy is
		replaced only after a significant paired confirmation on fresh fixtures. Qwen's failure analysis is advisory."""
		return {"job_id": service.optimize(request_id, budget)}

	@mcp.tool()
	def get_observation(job_id: str, observation_id: str, crop: bool = True) -> list:
		"""Return a recorded decision PNG with its camera, frame, visible faces and detector score."""
		if not re.fullmatch(r"[0-9a-f]{32}", job_id) or not re.fullmatch(r"\d+:\d+:(top|wrist_cam)", observation_id):
			raise ValueError("invalid episode or observation ID")
		path = service.out / "episodes" / job_id
		if not path.resolve().is_relative_to(service.out) or not (path / "episode.json").is_file():
			raise ValueError("episode has not been published")
		with (path / "events.jsonl").open(encoding="utf-8") as events:
			for line in events:
				event = json.loads(line)
				if event["kind"] == "view" and event["observation"]["id"] == observation_id:
					metadata = event["observation"]
					break
			else:
				raise ValueError("unknown observation")
		name = observation_id.replace(":", "-") + ("-crop.png" if crop else "-frame.png")
		image = path / "keyframes" / name
		if not image.resolve().is_relative_to(path.resolve()) or not image.is_file():
			raise ValueError("an RGB crop was unavailable for this view")
		return [json.dumps(metadata, allow_nan=False), Image(data=image.read_bytes(), format="png")]

	@mcp.tool()
	def job_status(job_id: str) -> dict[str, Any]:
		"""Read job status, progress (node, branch, frame, views) or evidence-based outcome. Simulator truth is private."""
		return service.status(job_id)

	@mcp.tool()
	async def wait_job(job_id: str, ctx: Context, timeout_s: float = 30.0) -> dict[str, Any]:
		"""Block up to timeout_s (max 600) for a job to finish, sending progress notifications; returns its status."""
		deadline = time.monotonic() + min(max(timeout_s, 0.0), 600.0)
		last, tick = None, 0
		while True:
			status = service.status(job_id)
			progress = status["progress"]
			key = (status["status"], progress["node"], progress["branch"], progress["frame"], progress["views"])
			if key != last:
				last, tick = key, tick + 1
				await ctx.report_progress(
					tick,
					None,
					f"{status['status']} node={progress['node']} branch={progress['branch']} "
					f"frame={progress['frame']} views={progress['views']}",
				)
			if status["status"] in TERMINAL or time.monotonic() >= deadline:
				return status
			await anyio.sleep(0.5)

	@mcp.tool()
	def cancel_job(job_id: str) -> dict[str, Any]:
		"""Cancel a queued job at once, or ask a running job to stop at its next frame (status then ends cancelled)."""
		return service.cancel(job_id)

	@mcp.tool()
	def resume_job(job_id: str) -> dict[str, Any]:
		"""Restart an interrupted, failed or cancelled episode from its last persisted skill-boundary checkpoint.

		Returns a new job_id; the resumed episode is a linked episode directory. Repeating the call returns the
		same job while it is active or complete.
		"""
		return {"job_id": service.resume(job_id), "resume_of": job_id}

	return mcp


def options(parser):
	parser.add_argument("--detector", type=Path, default=ROOT / "results/anomaly/models-sup/sup.json")
	parser.add_argument("--bottom-profile", type=Path)
	parser.add_argument("--llm-url")
	parser.add_argument("--model", default=DEFAULT_MODEL)
	parser.add_argument("--workers", type=int, default=1, help="Simulation worker processes (1 = in-process thread).")
	parser.add_argument(
		"--sensing",
		choices=("truth", "realistic"),
		default="truth",
		help="realistic: skills see only camera images and arm readings (see TwinMujoco).",
	)


def _service(a):
	llm_spec = (
		{"base_url": a.llm_url, "model": a.model, "max_tokens": 96, "timeout_s": 30, "retries": 0, "json_mode": True}
		if a.llm_url
		else None
	)
	return TwinService(a.out, a.detector, a.bottom_profile, workers=a.workers, llm_spec=llm_spec, sensing=a.sensing)


def main():
	p = argparse.ArgumentParser(description=__doc__)
	sub = p.add_subparsers(dest="command")
	server = sub.add_parser("serve")
	options(server)
	server.add_argument("--out", type=Path, default=ROOT / "data/catalog/agent")
	server.add_argument("--suite", type=Path, help="Operator-controlled suite (twin_mujoco.load_suite).")
	server.add_argument("--transport", choices=("stdio", "streamable-http"), default="stdio")
	server.add_argument("--host", default="127.0.0.1")
	server.add_argument("--port", type=int, default=8765)
	run = sub.add_parser("run")
	options(run)
	run.add_argument("--out", type=Path, required=True)
	run.add_argument("--suite", type=Path)
	run.add_argument("--policy", type=Path)
	run.add_argument("--headless", action="store_true")
	run.add_argument("--resume", type=Path, help="Interrupted episode directory to restart from its last checkpoint.")
	run.add_argument("--use-llm", action="store_true")
	run.add_argument("--smooth-for-vla", action="store_true", help="Rerun successful demonstrations with smooth carry.")
	step = sub.add_parser("step", help="Step-by-step tools: the client drives the episode (see step_mcp).")
	step.add_argument("--profile", choices=("omniscient", "realistic"), required=True)
	step.add_argument("--detector", type=Path, default=ROOT / "results/anomaly/models-sup/sup.json")
	step.add_argument("--out", type=Path, default=ROOT / "data/catalog/agent-step")
	step.add_argument("--suite", type=Path)
	step.add_argument("--transport", choices=("stdio", "streamable-http"), default="stdio")
	step.add_argument("--host", default="127.0.0.1")
	step.add_argument("--port", type=int, default=8766)
	step.add_argument("--scripted", action="store_true", help="Run the scripted client over every case instead.")
	dashboard = sub.add_parser("dashboard")
	dashboard.add_argument("--out", type=Path, default=ROOT / "data/catalog/agent")
	dashboard.add_argument("--port", type=int, default=8095)
	export = sub.add_parser("export")
	export.add_argument("--out", type=Path, required=True)
	export.add_argument("--repo-id", default="local/armigo-six-face")
	export.add_argument("episodes", nargs="+", type=Path)
	a = p.parse_args()
	if a.command is None:
		a = p.parse_args(["serve"])
	if a.command == "dashboard":
		from .twin_dashboard import serve

		serve(a.out, a.port)
		return
	if a.command == "export":
		export_lerobot(a.episodes, a.out, a.repo_id)
		return
	cases = load_suite(a.suite) if a.suite else DEMO_CASES
	if not cases:
		p.error("episode suite must be nonempty")
	if a.command == "step":
		from .step_mcp import main_step

		main_step(a, cases)
		return
	service = _service(a)
	try:
		if a.command == "serve":
			service.recover()  # jobs a previous server left running become interrupted before any request
			create_mcp(service, cases, a.host, a.port).run(transport=a.transport)
		else:
			if a.use_llm and service.llm is None:
				p.error("--use-llm requires --llm-url")
			policy = Policy(**json.loads(a.policy.read_text(encoding="utf-8"))) if a.policy else Policy()
			if a.resume:
				task = json.loads((a.resume / "task.json").read_text(encoding="utf-8"))
				plan = load_plan(a.resume)
				if plan is None:
					p.error("episode has no persisted checkpoint to resume from")
				result = service.run(
					tuple(task["case"]),
					Policy(**task["policy"]),
					a.out,
					video=task["video"],
					use_llm=task["use_llm"],
					resume=plan,
				)
				print(json.dumps(result, indent=2))
				return
			started = time.perf_counter()
			if a.smooth_for_vla:
				# Serial and in-process: the replay check holds the baseline endpoints between the two runs.
				reports, selection = [], []
				for i, case in enumerate(cases):
					baseline, selected = service.run_for_vla(
						case, policy, a.out / f"case-{i:04d}", video=not a.headless, use_llm=a.use_llm
					)
					reports.append(baseline)
					selection.append(selected)
			else:
				reports = service.run_batch(
					[(case, policy, a.out / f"case-{i:04d}") for i, case in enumerate(cases)],
					video=not a.headless,
					use_llm=a.use_llm,
				)
			report = {
				**aggregate(reports),
				"median_sim_frames": statistics.median(r["sim_frames"] for r in reports),
				"wall_seconds": round(time.perf_counter() - started, 3),
				"policy": asdict(policy),
			}
			if a.smooth_for_vla:
				report["carry_smoothing"] = selection
			for is_defective, expected in ((True, "defective"), (False, "normal")):
				group = [row for case, row in zip(cases, reports, strict=True) if case[1] == is_defective]
				report[expected] = {
					"episodes": len(group),
					"correct": sum(row["label_correct"] for row in group),
					"unknown": sum(row["label"] == "unknown" for row in group),
				}
			write_json(a.out / "report.json", report)
			print(json.dumps(report, indent=2))
	finally:
		service.close()


if __name__ == "__main__":
	main()
