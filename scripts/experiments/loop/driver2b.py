"""Round-2b gate runs, one arm per process.

	python driver2b.py candidate   # PYTHONPATH = candidate checkout (8902081): rerun round_00 + round_01, fresh round_02b
	python driver2b.py base        # PYTHONPATH = aa9eb45 worker: fresh round_02b control arm
	python run2b.py                # both arms under one CPU budget, then the gates (preferred)

Options (env): W = worker cap when OUT/workers-<arm>.txt is absent (default 8, two arms by hand = 16); ROUNDS (default 4);
JOB_TIMEOUT_S (default 1200; longest episode so far 284 s); ARMIGO_THREADS (BLAS/OpenMP threads per worker,
0 = library default). SPAWN_GAP_S (default 3) between new workers.
Tests only: ROUND2B_ROOT / ROUND2B_TAG (another data root / batch names), CASE_LIMIT (first N cases per group).

Compared with the first round-2 driver (ProcessPoolExecutor via resume_loop.execute):
- Workers are plain processes, one pipe each. A worker that crashes or hangs is killed with its process tree
  (the detector server too) and replaced; only its own episode is lost. ProcessPoolExecutor broke the whole pool
  and the run on any child death (BrokenProcessPool).
- Rounds re-plan from disk: done cases are skipped, dead/failed attempts get a new attempt dir, and
  runner.MAX_ATTEMPTS still caps each case. A rerun after a crash continues where it stopped.
- The worker cap is re-read during the run, so run2b.py can hand a finished arm's CPUs to the other arm.
- New workers start 3 s apart, and each worker death lowers this round's cap by one. Both the 01:43 and 02:11
  BrokenProcessPool crashes were the NVIDIA OpenGL driver refusing a context (Application log, "not able to
  initialize a connection with the GPU"); a 24-worker test hit it again. 16 workers in all has been clean.
- BLAS/OpenMP threads default to 1 per worker (default was ~31 threads each): with 16 workers the first episode
  per worker went 98 -> 63 s (start-up contention) and later episodes 50 -> 47 s; replay/motion.npz bit-identical
  on 32 episodes (bench_threads.py, benchA vs benchB).
- Progress lines carry rate and ETA; state-<arm>.json carries them too.
"""

from __future__ import annotations

import os

_THREADS = os.environ.get("ARMIGO_THREADS", "1")
if _THREADS != "0":  # set before numpy/mujoco load, and inherited by the workers and their detector servers
	for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
		os.environ.setdefault(_k, _THREADS)

import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import deque  # noqa: E402
from multiprocessing.connection import wait  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib  # noqa: E402

# LOOP_CONFIG names the round's config module (default round2b_config): paths, tag, suites and run batches.
cfg_mod = importlib.import_module(os.environ.get("LOOP_CONFIG", "round2b_config"))
CAMERAS, DRIVER_LIB, OUT, ROOT, SUITES, TAG = (cfg_mod.CAMERAS, cfg_mod.DRIVER_LIB, cfg_mod.OUT, cfg_mod.ROOT,
	cfg_mod.SUITES, cfg_mod.TAG)

sys.path.insert(0, str(DRIVER_LIB))
import resume_loop as driver  # noqa: E402
from digital_twin_armigo.experiments import runner  # noqa: E402
from make_suites import register_suite  # noqa: E402

GIB, MIB = 1024**3, 1024**2
MIN_FREE_GIB = 12.0
EST_MIB = 40.0  # per episode on disk; observed mean 24.5, max 36.1
SPAWN_GAP_S = float(os.environ.get("SPAWN_GAP_S", "3"))
SUITE_NAMES = getattr(cfg_mod, "SUITE_NAMES", ("round_00_clutter_60", "round_01_fresh_192", "round_02b_fresh_192"))


# ---- worker side ----


def _worker_main(conn) -> None:
	runner._init_worker()
	while True:
		try:
			msg = conn.recv()
		except (EOFError, OSError):
			return
		if msg is None:
			return
		cfg, job = msg
		try:
			row = driver.run_worker_job(cfg, job)
		except BaseException as exc:  # noqa: BLE001 - run_job records episode errors itself; this is the rest
			row = {"case_id": job["experiment"]["case_id"], "state": f"worker error {exc!r}"}
		conn.send(row)


def _kill_tree(pid: int) -> None:
	subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)], capture_output=True, check=False)


class Worker:
	def __init__(self, ctx):
		self.conn, child = ctx.Pipe()
		self.proc = ctx.Process(target=_worker_main, args=(child,), daemon=True)
		self.proc.start()
		child.close()
		self.job = None
		self.t0 = 0.0

	def send(self, cfg, job) -> None:
		self.job, self.t0 = job, time.monotonic()
		self.conn.send((cfg, job))

	def stop(self) -> None:
		try:
			self.conn.send(None)
		except OSError:
			pass
		self.proc.join(30)
		if self.proc.is_alive():
			self.kill()

	def kill(self) -> None:
		_kill_tree(self.proc.pid)
		self.proc.join(10)


# ---- coordinator side ----


def worker_cap(arm: str) -> int:
	path = OUT / f"workers-{arm}.txt"
	try:
		return max(0, int(path.read_text().strip()))
	except (OSError, ValueError):
		return int(os.environ.get("W", "8"))


def room(inflight: int) -> bool:
	return shutil.disk_usage(ROOT).free > MIN_FREE_GIB * GIB + (inflight + 1) * EST_MIB * MIB


def mark_dead(job: dict, state: str, error: str) -> dict:
	"""A worker died or hung mid-episode: say so in the attempt's status; the next round re-plans it."""
	path = Path(job["path"]) / "run_status.json"
	try:
		status = json.loads(path.read_text(encoding="utf-8"))
		status.update(state=state, error=error, finished=runner._now())
		runner._write_json(path, status)
	except (OSError, ValueError):
		pass
	return {"case_id": job["experiment"]["case_id"], "state": state}


class Progress:
	def __init__(self, arm: str, total: int, rnd: int):
		self.arm, self.total, self.rnd = arm, total, rnd
		self.done = self.bad = 0
		self.times: deque[float] = deque(maxlen=64)
		self.last_beat = time.monotonic()

	def eta(self) -> str:
		if len(self.times) < 2:
			return "?"
		rate = (len(self.times) - 1) / max(1e-9, self.times[-1] - self.times[0])  # episodes / s
		return f"{rate * 60:.1f}/min, ETA {(self.total - self.done) / rate / 60:.0f} min"

	def record(self, row: dict) -> None:
		self.done += 1
		self.bad += int(row["state"] != "done")
		self.times.append(time.monotonic())
		eta = f"  ({self.eta()})" if self.done % 25 == 0 or self.done == self.total else ""
		print(f"  [{self.done}/{self.total}] {row['case_id']}: {row['state']} {row.get('verdict') or ''}{eta}", flush=True)
		if self.done % 25 == 0:
			self.write("running")

	def heartbeat(self, busy: int, cap: int) -> None:
		if time.monotonic() - self.last_beat > 300:
			self.last_beat = time.monotonic()
			print(f"  .. {self.done}/{self.total} done, {busy} running, cap {cap}, {self.eta()}", flush=True)
			self.write("running")

	def write(self, state: str, **extra) -> None:
		doc = {"state": state, "round": self.rnd, "pending": self.total, "done": self.done, "not_done": self.bad}
		driver.write_json(OUT / f"state-{self.arm}.json", {**doc, "eta": self.eta(), "updated": runner._now(), **extra})


def run_round(arm: str, items: list, progress: Progress, timeout_s: float) -> bool:
	"""Run ``items``; returns False when it stopped at the disk boundary."""
	ctx = mp.get_context("spawn")
	queue = deque(items)
	idle: list[Worker] = []
	busy: dict = {}
	disk_ok = True
	last_spawn = 0.0
	penalty = 0
	try:
		while queue or busy:
			cap = worker_cap(arm)
			cap = max(1, cap - penalty) if cap else 0
			while idle and len(idle) + len(busy) > cap:
				idle.pop().stop()
			while queue and len(busy) < cap:
				if not room(len(busy)):
					print(f"  disk: under {MIN_FREE_GIB} GiB + in-flight left on {ROOT}; finishing running episodes", flush=True)
					disk_ok = False
					queue.clear()
					break
				if not idle:
					# New workers start one at a time: a burst of MuJoCo GL contexts made the NVIDIA OpenGL driver
					# fail ("not able to initialize a connection with the GPU", Application log 01:43 and 02:11)
					# and killed the worker, which took the whole ProcessPoolExecutor down.
					if time.monotonic() - last_spawn < SPAWN_GAP_S:
						break
					idle.append(Worker(ctx))
					last_spawn = time.monotonic()
				w = idle.pop()
				w.send(*queue.popleft())
				busy[w.conn] = w
			if not busy:
				if queue:  # cap 0: the launcher holds this arm back for now; or waiting out the spawn gap
					time.sleep(min(5.0, SPAWN_GAP_S))
					continue
				break
			ramping = bool(queue) and len(busy) < cap
			for conn in wait(list(busy), timeout=SPAWN_GAP_S if ramping else 10):
				w = busy.pop(conn)
				try:
					row = conn.recv()
				except (EOFError, OSError):
					w.proc.join(5)
					code = w.proc.exitcode
					w.kill()
					row = mark_dead(w.job, "worker_died", f"worker exit code {code}")
					# Most deaths so far were the GL driver refusing one more context: run one fewer for this round.
					penalty += 1
					print(f"  worker died (exit {code}); this round now runs at most cap - {penalty}", flush=True)
				else:
					idle.append(w)
				progress.record(row)
			now = time.monotonic()
			for conn, w in list(busy.items()):
				if now - w.t0 > timeout_s:
					busy.pop(conn)
					w.kill()
					progress.record(mark_dead(w.job, "timeout", f"no result after {timeout_s:.0f} s"))
			progress.heartbeat(len(busy), cap)
	finally:
		for w in list(busy.values()):
			w.kill()
		for w in idle:
			w.stop()
	return disk_ok


def plan_jobs(arm: str) -> tuple[list, list]:
	"""(jobs still to run, group dirs) for this arm, re-read from disk."""
	saved = ROOT / "suites/manifest.json"
	before = driver.read_json(saved) if saved.exists() else {"suites": {}}
	planned, groups = [], []
	if hasattr(cfg_mod, "batches"):
		batches = cfg_mod.batches(arm)  # [(batch, suite, camera), ...]
	else:
		batches = []
		if arm == "candidate":
			for cam in CAMERAS:
				batches += [(f"{TAG}-rerun00", "round_00_clutter_60", cam), (f"{TAG}-rerun01", "round_01_fresh_192", cam)]
		batches += [(f"{TAG}-fresh-{arm}", "round_02b_fresh_192", cam) for cam in CAMERAS]
	original = runner.cmd_run
	try:
		for batch, suite_name, cam in batches:
			got = []
			runner.cmd_run = lambda a, got=got: got.append(runner.plan_group(a)) or 0
			runner.main(["run", "--root", str(ROOT), "--batch", batch, "--suite", suite_name, "--engine", "mujoco",
				"--sensing", "realistic", "--camera", cam, "--strategy", "B", "--detector-spec", driver.DETECTOR,
				*(["--case-limit", os.environ["CASE_LIMIT"]] if os.environ.get("CASE_LIMIT") else [])])
			cfg, jobs, group_dir = got[0]
			planned += [(cfg, j) for j in jobs]
			groups.append(group_dir)
	finally:
		runner.cmd_run = original
	after = driver.read_json(saved)
	after["suites"] = {**before["suites"], **after["suites"]}
	driver.write_json(saved, after)
	planned.sort(key=lambda v: (v[1]["experiment"]["layout_id"], v[1]["experiment"]["case_id"]))
	return planned, groups


def missing(groups: list) -> list[str]:
	"""Planned cases whose latest attempt is not done (e.g. MAX_ATTEMPTS used up)."""
	out = []
	for gd in groups:
		for cid in json.loads((gd / "plan.json").read_text(encoding="utf-8"))["cases"]:
			att = runner._attempts(gd, cid)
			if not att or att[-1][2].get("state") != "done":
				out.append(f"{gd.parent.parent.name}/{gd.name}/{cid}")
	return out


def main(arm: str) -> int:
	if arm not in ("candidate", "base"):
		raise SystemExit("usage: driver2b.py candidate|base")
	driver.ROOT, driver.OUT, driver.STATE_NAME = ROOT, OUT, f"state-{arm}.json"
	runner.SUITES = SUITES
	for name in SUITE_NAMES:
		path = SUITES / f"{name}.json"
		register_suite(SUITES / "manifest.json", name, path, json.loads(path.read_text(encoding="utf-8")))
	timeout_s = float(os.environ.get("JOB_TIMEOUT_S", "1200"))
	rounds = int(os.environ.get("ROUNDS", "4"))
	print(f"{arm}: threads/worker {os.environ.get('OMP_NUM_THREADS', 'default')}, timeout {timeout_s:.0f} s", flush=True)
	groups = []
	for rnd in range(1, rounds + 1):
		plan, groups = plan_jobs(arm)
		print(f"{arm}_runs round {rnd}: {len(plan)} to run", flush=True)
		progress = Progress(arm, len(plan), rnd)
		if not plan:
			break
		need = MIN_FREE_GIB + len(plan) * EST_MIB / 1024
		free = shutil.disk_usage(ROOT).free / GIB
		if free < need:
			print(f"  warning: {free:.0f} GiB free, this round may need {need:.0f} GiB; it stops cleanly at the boundary", flush=True)
		progress.write("running")
		if not run_round(arm, plan, progress, timeout_s):
			progress.write("stopped_disk")
			print("stopped at the disk boundary; free space and rerun to continue", flush=True)
			return 3
		progress.write("round_done")
	left = missing(groups)
	if left:
		driver.write_json(OUT / f"state-{arm}.json", {"state": "incomplete", "missing": left, "updated": runner._now()})
		print(f"{arm}: {len(left)} case(s) not done after {rounds} rounds: {left[:10]}", flush=True)
		return 2
	n = sum(len(json.loads((gd / "plan.json").read_text(encoding="utf-8"))["cases"]) for gd in groups)
	driver.write_json(OUT / f"state-{arm}.json", {"state": "collected", "runs": n, "updated": runner._now()})
	print(json.dumps({"done": arm, "runs": n}), flush=True)
	return 0


if __name__ == "__main__":
	os.environ["ARMIGO_RERUN"] = "off"
	try:
		sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else ""))
	except BaseException as exc:
		if not isinstance(exc, SystemExit | KeyboardInterrupt):
			driver.write_json(OUT / f"state-{sys.argv[1]}.json", {"state": "error", "error": repr(exc)})
		raise
