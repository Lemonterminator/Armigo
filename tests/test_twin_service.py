"""Job store, idempotency across restarts, cancel, progress and resume plumbing with fake operations (no MuJoCo)."""

import asyncio
import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from digital_twin_armigo.agent.twin_graph import Policy
from digital_twin_armigo.agent.twin_jobs import JobCancelled, JobHooks, JobStore
from digital_twin_armigo.agent.twin_mcp import TwinService, create_mcp
from digital_twin_armigo.agent.twin_resume import live_checkpoint_ids, load_plan


@pytest.fixture
def detector(tmp_path):
	path = tmp_path / "detector.json"
	path.write_text(json.dumps({"crop_size": 256}), encoding="utf-8")
	return path


def wait_for(service, job_id, states, timeout=10.0):
	deadline = time.monotonic() + timeout
	while time.monotonic() < deadline:
		status = service.status(job_id)
		if status["status"] in states:
			return status
		time.sleep(0.02)
	raise AssertionError(f"job stayed {service.status(job_id)['status']}")


def fake_backend(node="scan_wrist", frame=1):
	return SimpleNamespace(node=node, branch=0, wrist_count=1, bottom_count=0, ep=SimpleNamespace(n_frames=frame))


def test_store_is_idempotent_and_survives_a_restart(tmp_path):
	store = JobStore(tmp_path / "jobs.sqlite")
	job, created = store.create("req", "episode", {"kind": "episode", "case": [1, True, 2]}, root=tmp_path / "episodes")
	assert created
	assert store.create("req", "episode", {"kind": "episode", "case": [1, True, 2]}, root=tmp_path) == (job, False)
	with pytest.raises(ValueError, match="different arguments"):
		store.create("req", "episode", {"kind": "episode", "case": [9, True, 2]})
	store.close()
	reopened = JobStore(tmp_path / "jobs.sqlite")
	assert reopened.create("req", "episode", {"kind": "episode", "case": [1, True, 2]}) == (job, False)
	assert reopened.get(job)["path"].endswith(job)


def test_queue_limit_counts_only_active_jobs(tmp_path):
	store = JobStore(tmp_path / "jobs.sqlite")
	first, _ = store.create("a", "episode", {"n": 1}, max_active=1)
	with pytest.raises(ValueError, match="queue is full"):
		store.create("b", "episode", {"n": 2}, max_active=1)
	store.mark_running(first)
	store.finish(first, "complete", {"ok": True})
	assert store.create("b", "episode", {"n": 2}, max_active=1)[1]


def test_rows_left_running_become_interrupted_on_startup(tmp_path, detector):
	service = TwinService(tmp_path / "out", detector)
	gate = threading.Event()
	job = service._submit("r1", {"kind": "episode", "x": 1}, lambda path: gate.wait(10), "episodes")
	wait_for(service, job, {"running"})
	queued = service._submit("r2", {"kind": "episode", "x": 2}, lambda path: {}, "episodes")
	assert service.status(queued)["status"] == "queued"
	# Simulate a hard kill: nothing is closed, a new server opens the same output directory.
	restarted = TwinService(tmp_path / "out", detector)
	restarted.recover()
	assert restarted.status(job)["status"] == "interrupted"
	assert restarted.status(queued)["status"] == "interrupted"
	assert restarted.status(job)["resumable"]
	# The request id is still bound to the original job, so a retry does not start a second simulation.
	assert restarted._submit("r1", {"kind": "episode", "x": 1}, lambda path: {}, "episodes") == job
	with pytest.raises(ValueError, match="different arguments"):
		restarted._submit("r1", {"kind": "episode", "x": 2}, lambda path: {}, "episodes")
	gate.set()
	service.close()
	restarted.close()


def test_cancel_queued_and_running_jobs(tmp_path, detector):
	service = TwinService(tmp_path / "out", detector)
	started = threading.Event()

	def operation(path):
		hook = JobHooks(service.store, service._ctx.job_id, check_every=1)
		started.set()
		while True:
			hook(fake_backend())
			time.sleep(0.01)

	running = service._submit("r1", {"kind": "episode", "x": 1}, operation, "episodes")
	queued = service._submit("r2", {"kind": "episode", "x": 2}, lambda path: {"ran": True}, "episodes")
	assert started.wait(5)
	assert service.cancel(queued)["status"] == "cancelled"
	assert service.cancel(running)["cancel_requested"] is True
	assert wait_for(service, running, {"cancelled"})["status"] == "cancelled"
	time.sleep(0.1)
	assert service.status(queued)["status"] == "cancelled"  # the queued job never started
	service.close()


def test_shutdown_interrupts_running_jobs_instead_of_cancelling(tmp_path, detector):
	service = TwinService(tmp_path / "out", detector)
	started = threading.Event()

	def operation(path):
		hook = JobHooks(service.store, service._ctx.job_id, check_every=1)
		started.set()
		while True:
			hook(fake_backend())
			time.sleep(0.01)

	job = service._submit("r1", {"kind": "episode", "x": 1}, operation, "episodes")
	assert started.wait(5)
	service.close()
	assert JobStore(tmp_path / "out" / "jobs.sqlite").get(job)["status"] == "interrupted"


def test_progress_is_throttled_and_visible_in_status(tmp_path, detector):
	service = TwinService(tmp_path / "out", detector)
	seen = threading.Event()
	release = threading.Event()

	def operation(path):
		hook = JobHooks(service.store, service._ctx.job_id, progress_every=5)
		for frame in range(12):
			hook(fake_backend(frame=frame))
		seen.set()
		release.wait(5)
		return {"label": "normal"}

	job = service._submit("r1", {"kind": "optimize", "budget": 1}, operation, "search")
	assert seen.wait(5)
	progress = service.status(job)["progress"]
	assert progress["node"] == "scan_wrist" and progress["frame"] == 10 and progress["views"] == 1
	release.set()
	status = wait_for(service, job, {"complete"})
	assert status["search"] == {"label": "normal"}
	service.close()


def test_failed_jobs_report_a_code_and_cancelled_exception_pickles(tmp_path, detector):
	import pickle

	service = TwinService(tmp_path / "out", detector)

	def boom(path):
		raise RuntimeError("secret detail")

	job = service._submit("r1", {"kind": "episode", "x": 1}, boom, "episodes")
	status = wait_for(service, job, {"failed"})
	assert status["code"] == "JOB_EXECUTION_FAILED" and "secret" not in json.dumps(status)
	assert pickle.loads(pickle.dumps(JobCancelled(True))).interrupted is True
	assert pickle.loads(pickle.dumps(JobCancelled(False))).interrupted is False
	service.close()


def test_run_batch_runs_items_in_order_without_a_pool(tmp_path, detector):
	service = TwinService(tmp_path / "out", detector)
	service.execute = lambda task: {"path": task["path"], "case": task["case"]}
	items = [((1, True, 0), Policy(), tmp_path / "a"), ((2, False, 5), Policy(), tmp_path / "b")]
	results = service.run_batch(items)
	assert [r["case"] for r in results] == [[1, True, 0], [2, False, 5]]
	service.close()


def write_checkpoint(directory, index, frame, views=()):
	(directory / "checkpoints").mkdir(parents=True, exist_ok=True)
	np.save(directory / "checkpoints" / f"checkpoint-{index}.npy", np.zeros(3))
	(directory / "checkpoints" / f"checkpoint-{index}.json").write_text(
		json.dumps(
			{
				"id": f"checkpoint-{index}",
				"branch": 0,
				"frame": frame,
				"expert": {},
				"filter": {},
				"wrist_count": 0,
				"bottom_count": 0,
				"views": list(views),
			}
		),
		encoding="utf-8",
	)


def test_live_checkpoints_follow_rewinds_and_ignore_a_torn_last_line(tmp_path):
	events = [
		{"kind": "checkpoint", "checkpoint": "checkpoint-0"},
		{"kind": "checkpoint", "checkpoint": "checkpoint-1"},
		{"kind": "checkpoint", "checkpoint": "checkpoint-2"},
		{"kind": "rewound", "checkpoint": "checkpoint-1"},
		{"kind": "checkpoint", "checkpoint": "checkpoint-2"},
	]
	log = tmp_path / "events.jsonl"
	log.write_text("".join(json.dumps(e) + "\n" for e in events) + '{"kind": "checkp', encoding="utf-8")
	assert live_checkpoint_ids(log) == ["checkpoint-0", "checkpoint-1", "checkpoint-2"]
	for i in range(3):
		write_checkpoint(tmp_path, i, 100 * i, [{"id": f"0:{i}:wrist_cam"}])
	plan = load_plan(tmp_path)
	assert plan.latest.id == "checkpoint-2" and plan.latest.frame == 200
	assert plan.graph_state()["views"] == [{"id": "0:2:wrist_cam"}]
	assert load_plan(tmp_path / "missing") is None


def test_resume_requires_a_resumable_episode_and_is_idempotent(tmp_path, detector):
	service = TwinService(tmp_path / "out", detector)
	ran = []
	service._episode_operation = lambda *args, **kwargs: lambda path: ran.append(path) or {"label": "x"}
	with pytest.raises(ValueError, match="unknown job"):
		service.resume("0" * 32)
	args = {
		"kind": "episode",
		"case": [1, True, 0],
		"policy": {"order": "wrist_first"},
		"record": False,
		"use_llm": False,
	}
	args["policy"] = {"order": "wrist_first", "max_bottom_views": 2, "max_wrist_views": 2, "max_rewinds": 1}
	job = service._submit("r1", args, lambda path: {"label": "x"}, "episodes")
	wait_for(service, job, {"complete"})
	with pytest.raises(ValueError, match="only interrupted"):
		service.resume(job)
	service.store.db.execute("UPDATE jobs SET status='interrupted' WHERE job_id=?", (job,))
	child = service.resume(job)
	assert wait_for(service, child, {"complete"})["resume_of"] == job
	assert service.resume(job) == child  # a completed resume is returned again instead of re-running
	assert service.status(job)["resumed_by"] == [child]
	assert len(ran) == 1
	service.close()


def test_mcp_exposes_the_job_control_tools(tmp_path, detector):
	service = TwinService(tmp_path / "out", detector)
	mcp = create_mcp(service, [(1, True, 0)], port=9999)
	names = {tool.name for tool in asyncio.run(mcp.list_tools())}
	assert {"job_status", "wait_job", "cancel_job", "resume_job", "start_episode"} <= names
	assert mcp.settings.port == 9999
	service.close()


def test_shared_scorer_broker_answers_worker_proxies_and_maps_errors():
	import multiprocessing as mp

	from digital_twin_armigo.agent.twin_pool import ProxyScorer, ScorerBroker
	from digital_twin_armigo.inspection_scorer import ScoreServerError

	class Scorer:
		def threshold(self, camera):
			return 0.5

		def score_with_details(self, camera, image):
			if camera == "bad":
				raise ScoreServerError("down")
			return float(image.mean()), {"cls": 1.0}

		def close(self):
			pass

	broker = ScorerBroker(Scorer, 2, mp.get_context("spawn"))
	proxies = [ProxyScorer(broker.requests, broker.responses[i], i) for i in range(2)]
	assert proxies[1].threshold("top") == 0.5
	assert proxies[0].score_with_details("top", np.full((2, 2, 3), 4, np.uint8)) == (4.0, {"cls": 1.0})
	with pytest.raises(ScoreServerError, match="down"):
		proxies[1].score("bad", np.zeros((2, 2, 3), np.uint8))
	broker.close()
