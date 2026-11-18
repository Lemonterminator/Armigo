"""Before/after benchmarks for the Armigo agent stack.

Subcommands
-----------
``episodes``  Run ``armigo-mcp run`` on a fixture suite (extra args are passed through, e.g. ``--use-llm`` or
              ``--workers 4``), then summarise wall time, outcomes and every graph ``decision`` event
              (source llm / fallback / policy, fallback reasons, prompt tokens and latency when logged).
``vllm``      Fire N chat requests at an OpenAI-compatible server with C concurrent clients. Every request shares
              one fixed system prefix (to expose prefix caching) and has a short distinct user turn.
``context``   Build the inner-graph LLM context after ``prepare`` for one fixture and count its prompt tokens
              through the server's ``/tokenize`` endpoint (the pre-change ``select_skill`` message shape).

Every subcommand prints one JSON object and can write it with ``--json``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _emit(result: dict, path: Path | None) -> None:
	text = json.dumps(result, indent=2, ensure_ascii=False)
	if path:
		path.parent.mkdir(parents=True, exist_ok=True)
		path.write_text(text, encoding="utf-8")
	print(text)


def summarise_decisions(out: Path) -> dict:
	sources: Counter = Counter()
	reasons: Counter = Counter()
	tokens, latency = [], []
	llm_calls = 0
	for log in sorted(out.rglob("events.jsonl")):
		for line in log.read_text(encoding="utf-8").splitlines():
			event = json.loads(line)
			if event.get("kind") != "decision":
				continue
			source = event.get("source", "?")
			sources[source] += 1
			if source != "policy":
				llm_calls += 1
			if event.get("fallback_reason") or event.get("reason"):
				reasons[event.get("fallback_reason") or event.get("reason")] += 1
			if event.get("prompt_tokens"):
				tokens.append(int(event["prompt_tokens"]))
			if event.get("latency_s") is not None:
				latency.append(float(event["latency_s"]))
	llm = sources.get("llm", 0)
	return {
		"decisions": dict(sources),
		"llm_consulted": llm_calls,
		"llm_adoption_rate": round(llm / llm_calls, 3) if llm_calls else None,
		"fallback_reasons": dict(reasons),
		"prompt_tokens_median": statistics.median(tokens) if tokens else None,
		"prompt_tokens_max": max(tokens) if tokens else None,
		"llm_latency_median_s": round(statistics.median(latency), 3) if latency else None,
	}


def cmd_episodes(a, passthrough: list[str]) -> dict:
	out = a.out.resolve()
	cmd = [
		sys.executable,
		"-m",
		"digital_twin_armigo.agent.twin_mcp",
		"run",
		"--out",
		str(out),
		"--suite",
		str(a.suite),
	]
	cmd += ["--headless", *passthrough]
	t0 = time.perf_counter()
	proc = subprocess.run(
		cmd, check=False, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace"
	)
	wall = time.perf_counter() - t0
	if proc.returncode:
		sys.stderr.write(proc.stdout[-4000:] + proc.stderr[-4000:])
		raise SystemExit(f"armigo-mcp run failed with {proc.returncode}")
	report = json.loads((out / "report.json").read_text(encoding="utf-8"))
	n = report["episodes"]
	return {
		"suite": str(a.suite),
		"args": passthrough,
		"episodes": n,
		"wall_s": round(wall, 1),
		"episodes_per_min": round(60 * n / wall, 2),
		"successes": report["successes"],
		"labels_correct": report["labels_correct"],
		"unknown": report["unknown"],
		"rewinds": report["rewinds"],
		"sim_frames": report["sim_frames"],
		"failures": report["failures"],
		**summarise_decisions(out),
		"smoothness": report.get("smoothness"),
		"report_extra": {k: v for k, v in report.items() if k.startswith("llm") or k == "fallback_reasons"},
	}


def cmd_vllm(a) -> dict:
	import httpx

	prefix = "You select legal inspection or recovery skills for a sorting robot. " + " ".join(
		f"Rule {i}: the detector and motion controller own safety and classification." for i in range(a.prefix_rules)
	)
	client = httpx.Client(timeout=300)

	def one(i: int) -> tuple[float, int]:
		body = {
			"model": a.model,
			"messages": [
				{"role": "system", "content": prefix},
				{"role": "user", "content": json.dumps({"choices": ["scan_wrist", "present_bottom"], "request": i})},
			],
			"temperature": 0,
			"max_tokens": a.max_tokens,
			"chat_template_kwargs": {"enable_thinking": False},
		}
		t0 = time.perf_counter()
		r = client.post(f"{a.url.rstrip('/')}/chat/completions", json=body)
		r.raise_for_status()
		usage = r.json().get("usage") or {}
		return time.perf_counter() - t0, int(usage.get("prompt_tokens") or 0)

	one(-1)  # warm-up (and prefix-cache fill when enabled)
	t0 = time.perf_counter()
	with ThreadPoolExecutor(a.concurrency) as pool:
		rows = list(pool.map(one, range(a.requests)))
	wall = time.perf_counter() - t0
	lat = sorted(r[0] for r in rows)
	return {
		"url": a.url,
		"requests": a.requests,
		"concurrency": a.concurrency,
		"prompt_tokens": rows[0][1],
		"wall_s": round(wall, 2),
		"requests_per_s": round(a.requests / wall, 3),
		"latency_p50_s": round(lat[len(lat) // 2], 3),
		"latency_p95_s": round(lat[min(len(lat) - 1, int(0.95 * len(lat)))], 3),
	}


def cmd_context(a) -> dict:
	import httpx

	from digital_twin_armigo.agent.twin_mujoco import TwinMujoco, sample_case

	class _NullTrace:
		def __init__(self):
			self.events = []

		def __getattr__(self, name):
			return lambda *args, **kwargs: None

	backend = TwinMujoco(sample_case(*a.case), None, _NullTrace())
	try:
		backend.prepare()
		build = getattr(backend, "decision_context", None)
		context = (
			build(["scan_wrist", "present_bottom"], [], [])
			if build
			else {
				"geometry": backend._geometry(),
				"views": [],
				"failures": [],
				"spent_sim_frames": backend.total_frames,
			}
		)
	finally:
		backend.close()
	user = json.dumps({"choices": ["scan_wrist", "present_bottom"], **context}, separators=(",", ":"))
	r = httpx.post(
		f"{a.url.rstrip('/').removesuffix('/v1')}/tokenize",
		json={"model": a.model, "messages": [{"role": "user", "content": user}]},
		timeout=60,
	)
	r.raise_for_status()
	return {"case": a.case, "context_chars": len(user), "prompt_tokens": r.json()["count"], "max_model_len": a.max_len}


def main() -> None:
	p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	p.add_argument("--json", type=Path)
	sub = p.add_subparsers(dest="cmd", required=True)
	ep = sub.add_parser("episodes")
	ep.add_argument("--suite", type=Path, required=True)
	ep.add_argument("--out", type=Path, required=True)
	vl = sub.add_parser("vllm")
	vl.add_argument("--url", default="http://127.0.0.1:8038/v1")
	vl.add_argument("--model", default="qwen3.8-vllm")
	vl.add_argument("--requests", type=int, default=16)
	vl.add_argument("--concurrency", type=int, default=4)
	vl.add_argument("--max-tokens", type=int, default=32)
	vl.add_argument("--prefix-rules", type=int, default=60)
	cx = sub.add_parser("context")
	cx.add_argument("--url", default="http://127.0.0.1:8038/v1")
	cx.add_argument("--model", default="qwen3.8-vllm")
	cx.add_argument("--case", type=json.loads, default=[442701, False, 5])
	cx.add_argument("--max-len", type=int, default=4096)
	a, rest = p.parse_known_args()
	if a.cmd == "episodes":
		result = cmd_episodes(a, rest)
	elif rest:
		p.error(f"unrecognised arguments: {rest}")
	else:
		result = cmd_vllm(a) if a.cmd == "vllm" else cmd_context(a)
	_emit(result, a.json)


if __name__ == "__main__":
	main()
