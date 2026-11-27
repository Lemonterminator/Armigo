"""Run both arms under one worker budget, then the gates.

	python run2b.py                 # TOTAL workers (default 16) split by pending episodes; gates after both finish
	python run2b.py --total 20 --no-gates
	python run2b.py --arms base     # one arm only

Each arm is its own process (driver2b.py) on its own checkout. The split is proportional to the work each arm
reports after planning, so both finish together; an arm that finishes early hands its workers to the other
(driver2b re-reads OUT/workers-<arm>.txt). Logs: run2b-<arm>.log, gates2b.log, run2b.log (this script).
Rerunning resumes: done episodes are skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import importlib

cfg_mod = importlib.import_module(os.environ.get("LOOP_CONFIG", "round2b_config"))  # see driver2b.py
HERE, LOG_SUFFIX, OUT, PYTHON, pythonpath = cfg_mod.HERE, cfg_mod.LOG_SUFFIX, cfg_mod.OUT, cfg_mod.PYTHON, cfg_mod.pythonpath
GATES = getattr(cfg_mod, "GATES", "gates2b.py")
ROUND_SPEC = getattr(cfg_mod, "ROUND_SPEC", None)  # gates.py reads the round spec; gates2b.py takes no arguments
LOG_DIR = getattr(cfg_mod, "LOG_DIR", HERE)


def log(msg: str) -> None:
	print(time.strftime("%H:%M:%S"), msg, flush=True)


def write_cap(arm: str, n: int) -> None:
	OUT.mkdir(parents=True, exist_ok=True)
	(OUT / f"workers-{arm}.txt").write_text(str(n))


def state(arm: str) -> dict:
	try:
		return json.loads((OUT / f"state-{arm}.json").read_text(encoding="utf-8-sig"))
	except (OSError, ValueError):
		return {}


def start(arm: str) -> subprocess.Popen:
	env = {**os.environ, "PYTHONPATH": pythonpath(arm), "PYTHONUTF8": "1"}
	out = open(LOG_DIR / f"run2b-{arm}{LOG_SUFFIX}.log", "a", encoding="utf-8")  # noqa: SIM115 - lives as long as the child
	out.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} start {arm}\n")
	out.flush()
	return subprocess.Popen([str(PYTHON), "-u", str(HERE / "driver2b.py"), arm], env=env, stdout=out,
		stderr=subprocess.STDOUT, cwd=HERE)


def split(total: int, work: dict[str, int]) -> dict[str, int]:
	live = {a: n for a, n in work.items() if n > 0}
	if not live:
		return {a: 0 for a in work}
	caps = {a: max(1, round(total * n / sum(live.values()))) for a, n in live.items()}
	while sum(caps.values()) > total:  # rounding
		caps[max(caps, key=caps.get)] -= 1
	return {a: caps.get(a, 0) for a in work}


def main() -> int:
	p = argparse.ArgumentParser()
	p.add_argument("--total", type=int, default=int(os.environ.get("TOTAL", "16")))
	p.add_argument("--arms", default="candidate,base")
	p.add_argument("--no-gates", action="store_true")
	a = p.parse_args()
	arms = a.arms.split(",")
	for arm in arms:  # hold every arm until both have planned, so neither grabs the whole machine first
		write_cap(arm, 0)
		(OUT / f"state-{arm}.json").write_text(json.dumps({"state": "launching"}), encoding="utf-8")
	procs = {arm: start(arm) for arm in arms}
	log(f"started {', '.join(f'{k} pid {v.pid}' for k, v in procs.items())}; total workers {a.total}")
	split_done = False
	codes: dict[str, int] = {}
	while len(codes) < len(procs):
		for arm, proc in procs.items():
			if arm not in codes and proc.poll() is not None:
				codes[arm] = proc.returncode
				log(f"{arm} exited {proc.returncode}: {state(arm).get('state')}")
				rest = [x for x in procs if x not in codes]
				if split_done and rest:
					for x in rest:
						write_cap(x, a.total // len(rest))
					log(f"workers handed to {rest}")
		if not split_done:
			work = {}
			for arm in arms:
				st = state(arm)
				if arm in codes:
					work[arm] = 0
				elif st.get("state") == "running" and "pending" in st:
					work[arm] = int(st["pending"])
			if len(work) == len(arms):
				caps = split(a.total, work)
				for arm, n in caps.items():
					write_cap(arm, n)
				split_done = True
				log(f"pending {work} -> workers {caps}")
		time.sleep(5)
	if any(codes.values()):
		log(f"not running the gates: exit codes {codes}; rerun run2b.py to resume after fixing the cause")
		return 1
	if a.no_gates:
		return 0
	log(f"both arms collected; running {GATES}")
	with open(LOG_DIR / f"{GATES[:-3]}{LOG_SUFFIX}.log", "w", encoding="utf-8") as out:
		spec_args = ["--round", str(ROUND_SPEC), "--root", str(cfg_mod.ROOT)] if ROUND_SPEC else []
		code = subprocess.call([str(PYTHON), "-u", str(HERE / GATES), *spec_args], cwd=HERE, stdout=out,
			stderr=subprocess.STDOUT, env={**os.environ, "PYTHONPATH": pythonpath("candidate"), "PYTHONUTF8": "1"})
	log(f"{GATES} exited {code}; see {GATES[:-3]}.log")
	return code


if __name__ == "__main__":
	sys.exit(main())
