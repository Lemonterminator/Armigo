"""Local MCP entry point for six-face simulation, policy search, and episode export."""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import re
import statistics
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP, Image

from ..inspection_scorer import default_sup_scorer
from .llm import DEFAULT_MODEL, OpenAICompatLLM
from .twin_graph import Policy, build_graph, search_policies
from .twin_mujoco import PROFILE, ROOT, TwinMujoco, file_digest, sample_case
from .twin_record import EpisodeTrace, export_lerobot, write_json
from .twin_search import analyze_failures, dev_suite

LOG = logging.getLogger(__name__)
# First 8 cases of the generated dev suite (seeds from 700001, disjoint from DEMO/dev4/bench16).
DEV_CASES = tuple(dev_suite(8))
# Policy-search episode cap = SEARCH_EPISODES_PER_BUDGET * (budget + 1); the 54-policy grid needs at least 54.
SEARCH_EPISODES_PER_BUDGET = 50
DEMO_CASES = ((187271, False, 5), (187271, True, 5), (187272, False, 5), (187272, True, 5))


class TwinService:
	def __init__(self, out, detector, bottom_profile=None, llm=None):
		self.out = Path(out).resolve()
		self.detector = Path(detector).resolve()
		spec = json.loads(self.detector.read_text(encoding="utf-8"))
		self.model_hash = file_digest(self.detector)
		self.crop_size = int(spec["crop_size"])
		self.bottom_threshold = None
		if bottom_profile:
			raw = json.loads(Path(bottom_profile).read_text(encoding="utf-8"))
			if raw["profile"] != PROFILE or raw["detector_sha256"] != self.model_hash:
				raise ValueError("bottom calibration belongs to a different detector or camera profile")
			self.bottom_threshold = float(raw["threshold"])
			if not math.isfinite(self.bottom_threshold):
				raise ValueError("invalid bottom threshold")
		self.llm = llm
		self.scorer = None
		self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="armigo-mujoco")
		self.jobs: dict[str, tuple[tuple, Future, str]] = {}

	def close(self):
		self.worker.shutdown(wait=True, cancel_futures=True)
		if self.scorer:
			self.scorer.close()
		if self.llm:
			self.llm.close()

	def run(self, case, policy, path, *, video, use_llm=False, fault=None):
		if self.scorer is None:
			self.scorer = default_sup_scorer(self.detector)
		layout = sample_case(*case)
		trace = EpisodeTrace(path, video=video)
		backend = None
		try:
			backend = TwinMujoco(
				layout,
				self.scorer,
				trace,
				llm=self.llm if use_llm else None,
				bottom_threshold=self.bottom_threshold,
				crop_size=self.crop_size,
				fault=fault,
			)
			result = build_graph(backend, policy).invoke({}, config={"recursion_limit": 64})["result"]
			trace.finish(
				result,
				{
					"engine": "mujoco",
					"camera_profile": PROFILE,
					"detector_sha256": self.model_hash,
					"policy": asdict(policy),
					"llm_enabled": use_llm,
					"bottom_threshold": self.bottom_threshold,
					"model": self.llm.model if use_llm else None,
				},
			)
			return result
		except BaseException:
			trace.close()
			raise
		finally:
			if backend:
				backend.close()

	def _submit(self, request_id, arguments, operation, folder):
		if not request_id or len(request_id) > 128:
			raise ValueError("request_id must contain 1 to 128 characters")
		old = self.jobs.get(request_id)
		if old is not None:
			if arguments != old[0]:
				raise ValueError("request_id is already bound to different arguments")
			return old[2]
		if sum(not item[1].done() for item in self.jobs.values()) >= 8:
			raise ValueError("simulation queue is full")
		job_id = uuid.uuid4().hex
		self.jobs[request_id] = (arguments, self.worker.submit(operation, self.out / folder / job_id), job_id)
		return job_id

	def start(self, request_id, case, policy, record, use_llm):
		if use_llm and self.llm is None:
			raise ValueError("start armigo-mcp with --llm-url to enable LLM skill selection")

		def job(path):
			return self.run(case, policy, path, video=record, use_llm=use_llm)

		return self._submit(request_id, ("episode", case, policy, record, use_llm), job, "episodes")

	def optimize(self, request_id, budget):
		if not 1 <= budget <= 8:
			raise ValueError("policy budget must be in [1, 8]")
		cap = SEARCH_EPISODES_PER_BUDGET * (budget + 1)

		def job(path):
			counter = itertools.count()

			def evaluate_batch(items):
				jobs = [(case, policy, path / "episodes" / f"{next(counter):04d}") for policy, case in items]
				if hasattr(self, "run_batch"):
					return self.run_batch(jobs)
				return [self.run(case, policy, out, video=False) for case, policy, out in jobs]

			report = search_policies(evaluate_batch, cap)
			report["llm_analysis"] = (
				analyze_failures(self.llm, report) if self.llm is not None else {"status": "skipped", "error": "no llm"}
			)
			write_json(path / "report.json", report)
			write_json(path / "policy.json", report["selected"]["policy"])
			return report

		return self._submit(request_id, ("optimize", budget), job, "search")

	def status(self, job_id):
		item = next((item for item in self.jobs.values() if item[2] == job_id), None)
		if item is None:
			raise ValueError("unknown job")
		arguments, future, _ = item
		if not future.done():
			return {"job_id": job_id, "status": "running" if future.running() else "queued"}
		try:
			result = future.result()
		except Exception:
			LOG.exception("Twin job %s failed", job_id)
			return {"job_id": job_id, "status": "failed", "code": "JOB_EXECUTION_FAILED"}
		if arguments[0] == "optimize":
			return {"job_id": job_id, "status": "complete", "search": result}
		keys = (
			"label",
			"destination",
			"active_branch",
			"rewinds",
			"failures",
			"active_views",
			"views",
			"sim_frames",
			"evidence",
			"cleared_faces",
			"placement_evidence",
		)
		return {"job_id": job_id, "status": "complete", "episode": {key: result[key] for key in keys}}


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
		"failures": [{"code": code, "count": count} for code, count in sorted(failures.items())],
	}


def create_mcp(service: TwinService, cases):
	mcp = FastMCP("Armigo Six-Face Twin")
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
		"""Search generic policies using only aggregates on the fixed development suite."""
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
		"""Read job progress or evidence-based outcome. Simulator truth is private."""
		return service.status(job_id)

	return mcp


def options(parser):
	parser.add_argument("--detector", type=Path, default=ROOT / "results/anomaly/models-sup/sup.json")
	parser.add_argument("--bottom-profile", type=Path)
	parser.add_argument("--llm-url")
	parser.add_argument("--model", default=DEFAULT_MODEL)


def _service(a):
	llm = (
		OpenAICompatLLM(a.llm_url, a.model, max_tokens=96, timeout_s=30, retries=0, json_mode=True)
		if a.llm_url
		else None
	)
	return TwinService(a.out, a.detector, a.bottom_profile, llm)


def main():
	p = argparse.ArgumentParser(description=__doc__)
	sub = p.add_subparsers(dest="command")
	server = sub.add_parser("serve")
	options(server)
	server.add_argument("--out", type=Path, default=ROOT / "data/catalog/agent")
	server.add_argument("--suite", type=Path, help="Operator-controlled JSON array of [seed, defective, face].")
	run = sub.add_parser("run")
	options(run)
	run.add_argument("--out", type=Path, required=True)
	run.add_argument("--suite", type=Path)
	run.add_argument("--policy", type=Path)
	run.add_argument("--headless", action="store_true")
	run.add_argument("--use-llm", action="store_true")
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
	cases = [tuple(row) for row in json.loads(a.suite.read_text(encoding="utf-8-sig"))] if a.suite else DEMO_CASES
	if not cases:
		p.error("episode suite must be nonempty")
	service = _service(a)
	try:
		if a.command == "serve":
			create_mcp(service, cases).run(transport="stdio")
		else:
			if a.use_llm and service.llm is None:
				p.error("--use-llm requires --llm-url")
			policy = Policy(**json.loads(a.policy.read_text(encoding="utf-8"))) if a.policy else Policy()
			started = time.perf_counter()
			reports = [
				service.run(case, policy, a.out / f"case-{i:04d}", video=not a.headless, use_llm=a.use_llm)
				for i, case in enumerate(cases)
			]
			report = {
				**aggregate(reports),
				"median_sim_frames": statistics.median(r["sim_frames"] for r in reports),
				"wall_seconds": round(time.perf_counter() - started, 3),
				"policy": asdict(policy),
			}
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
