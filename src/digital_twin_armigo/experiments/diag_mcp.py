"""Read-only failure diagnosis MCP for tool-fixing agents and human reviewers (Task H8).

Exposes episode summaries, discrete/score run comparisons, failure ledger clusters, and counterfactual oracle verdicts,
with per-session call accounting to ``<round_dir>/diag_calls.jsonl``. Never connect this server to task agents:
``episode_summary`` reads ``audit.private.json`` and simulator truth diagnostics.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .scoring import _events, _json, score_episode

TOOLS = ("episode_summary", "compare_runs", "ledger_clusters", "counterfactual_result")
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8770

_TIMELINE_KINDS = frozenset({"node", "failure", "tool_call", "view"})
_TIMELINE_LIMIT = 200
_TIMELINE_HALF = 100
_IGNORED_COMPARE_KEYS = frozenset({"wall_time_s", "latency_s", "seq"})


def _is_timeline_event(event: dict[str, Any]) -> bool:
	kind = str(event.get("kind") or "")
	return kind in _TIMELINE_KINDS or kind.endswith("_perceived")


def _format_timeline_line(idx: int, event: dict[str, Any]) -> str:
	seq = event.get("seq")
	seq_str = idx if seq is None else seq
	kind = str(event.get("kind") or "")
	node = event.get("node")
	node_str = "-" if node is None or node == "" else str(node)
	parts = [f"{seq_str} {kind} {node_str}"]
	for key in ("code", "tool", "error_xy_mm", "error_yaw_deg", "score"):
		val = event.get(key)
		if val is None and key == "score" and isinstance(event.get("observation"), dict):
			val = event["observation"].get("score")
		if val is not None:
			parts.append(f"{key}={val}")
	return " ".join(parts)


def _build_timeline(events: list[dict[str, Any]]) -> list[str]:
	indexed = []
	for idx, event in enumerate(events):
		if not _is_timeline_event(event):
			continue
		raw_seq = event.get("seq")
		seq_key = idx if raw_seq is None else int(raw_seq)
		indexed.append((seq_key, idx, event))
	indexed.sort(key=lambda item: (item[0], item[1]))
	lines = [_format_timeline_line(idx, event) for _, idx, event in indexed]
	if len(lines) > _TIMELINE_LIMIT:
		return [*lines[:_TIMELINE_HALF], "... (truncated)", *lines[-_TIMELINE_HALF:]]
	return lines


def episode_summary(path: Path) -> dict[str, Any]:
	"""Summarize one episode: ``first_error`` fields, physical outcome, and compressed seq-ordered timeline."""
	ep_dir = Path(path)
	scored = score_episode(ep_dir)
	err = scored.get("first_error") or {}
	events = _events(ep_dir / "events.jsonl")
	return {
		"path": scored["path"],
		"case_id": scored["case_id"],
		"expected_label": scored["expected_label"],
		"label": scored["label"],
		"label_correct": scored["label_correct"],
		"root_cause": err.get("root_cause"),
		"mechanism": err.get("mechanism"),
		"upstream": err.get("upstream"),
		"precursors": err.get("precursors", []),
		"detail": err.get("detail"),
		"frame": err.get("frame"),
		"physical": scored["physical"],
		"physical_success": scored["physical_success"],
		"physical_outcome": scored["physical_outcome"],
		"scored_verdict": scored["scored_verdict"],
		"timeline": _build_timeline(events),
	}


def _is_number(val: Any) -> bool:
	return isinstance(val, (int, float)) and not isinstance(val, bool)


def _numbers_equal(va: float, vb: float, tol: float) -> bool:
	fa, fb = float(va), float(vb)
	if math.isnan(fa) and math.isnan(fb):
		return True
	if math.isfinite(fa) and math.isfinite(fb):
		return abs(fa - fb) <= tol
	return fa == fb


def _values_equal(va: Any, vb: Any, tol: float) -> bool:
	if isinstance(va, bool) or isinstance(vb, bool):
		return type(va) is type(vb) and va == vb
	if _is_number(va) and _is_number(vb):
		return _numbers_equal(va, vb, tol)
	if isinstance(va, dict) and isinstance(vb, dict):
		if va.keys() != vb.keys():
			return False
		return all(_values_equal(va[k], vb[k], tol) for k in va)
	if isinstance(va, list) and isinstance(vb, list):
		return len(va) == len(vb) and all(_values_equal(x, y, tol) for x, y in zip(va, vb, strict=True))
	return type(va) is type(vb) and va == vb


def _observations_discrete_equal(oa: dict[str, Any], ob: dict[str, Any], tol: float) -> bool:
	if oa.keys() != ob.keys():
		return False
	for k, val_a in oa.items():
		val_b = ob[k]
		if k == "score" and _is_number(val_a) and _is_number(val_b):
			continue  # detector score deltas are reported separately without breaking discrete equality
		if not _values_equal(val_a, val_b, tol):
			return False
	return True


def _discrete_event_equal(ea: dict[str, Any], eb: dict[str, Any], tol: float) -> bool:
	if (ea.get("kind"), ea.get("node"), ea.get("branch")) != (eb.get("kind"), eb.get("node"), eb.get("branch")):
		return False
	keys_a = set(ea) - _IGNORED_COMPARE_KEYS
	keys_b = set(eb) - _IGNORED_COMPARE_KEYS
	if keys_a != keys_b:
		return False
	for k in keys_a:
		va, vb = ea[k], eb[k]
		if k == "observation" and isinstance(va, dict) and isinstance(vb, dict):
			if not _observations_discrete_equal(va, vb, tol):
				return False
		elif not _values_equal(va, vb, tol):
			return False
	return True


def _score_delta(idx: int, ea: dict[str, Any], eb: dict[str, Any], tol: float) -> dict[str, Any] | None:
	oa, ob = ea.get("observation"), eb.get("observation")
	if not (isinstance(oa, dict) and isinstance(ob, dict)):
		return None
	sa, sb = oa.get("score"), ob.get("score")
	if not (_is_number(sa) and _is_number(sb)):
		return None
	if _numbers_equal(sa, sb, tol):
		return None
	fa, fb = float(sa), float(sb)
	return {
		"index": idx,
		"seq_a": ea.get("seq"),
		"seq_b": eb.get("seq"),
		"node": ea.get("node"),
		"view_id": oa.get("id"),
		"a": fa,
		"b": fb,
		"delta": (fb - fa) if (math.isfinite(fa) and math.isfinite(fb)) else None,
	}


def compare_runs(a: Path, b: Path, *, tol: float = 1e-6) -> dict[str, Any]:
	"""Align ``events.jsonl`` of ``a`` and ``b`` positionally, compare discrete events, and check verdict equality."""
	dir_a, dir_b = Path(a), Path(b)
	events_a = _events(dir_a / "events.jsonl")
	events_b = _events(dir_b / "events.jsonl")

	first_divergence: dict[str, Any] | None = None
	score_deltas: list[dict[str, Any]] = []
	n = min(len(events_a), len(events_b))
	for i in range(n):
		ea, eb = events_a[i], events_b[i]
		if not _discrete_event_equal(ea, eb, tol):
			first_divergence = {"index": i, "a": ea, "b": eb}
			break
		delta = _score_delta(i, ea, eb, tol)
		if delta is not None:
			score_deltas.append(delta)

	if first_divergence is None and len(events_a) != len(events_b):
		first_divergence = {
			"index": n,
			"a": events_a[n] if n < len(events_a) else None,
			"b": events_b[n] if n < len(events_b) else None,
		}

	ep_a = _json(dir_a / "episode.json") or {}
	ep_b = _json(dir_b / "episode.json") or {}
	res_a = ep_a.get("result") or {}
	res_b = ep_b.get("result") or {}
	verdict_a = res_a.get("verdict_final_cycle") or res_a.get("verdict")
	verdict_b = res_b.get("verdict_final_cycle") or res_b.get("verdict")

	return {
		"identical_discrete": first_divergence is None,
		"first_divergence": first_divergence,
		"score_deltas": score_deltas,
		"verdict_equal": verdict_a == verdict_b,
	}


def ledger_clusters(round_dir: Path) -> list[dict[str, Any]]:
	"""Read failure clusters from ``<round_dir>/ledger.json`` (supports list or ``{"clusters": [...]}``)."""
	path = Path(round_dir) / "ledger.json"
	data = json.loads(path.read_text(encoding="utf-8-sig"))
	if isinstance(data, list):
		return data
	if isinstance(data, dict):
		return list(data.get("clusters") or [])
	raise ValueError(f"unexpected ledger.json structure in {path}")


def counterfactual_result(round_dir: Path, case_id: str) -> dict[str, Any] | None:
	"""Search ``<round_dir>/counterfactual.jsonl`` for ``case_id``."""
	path = Path(round_dir) / "counterfactual.jsonl"
	if not path.is_file():
		return None
	with path.open(encoding="utf-8-sig") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			row = json.loads(line)
			if row.get("case_id") == case_id:
				return row
	return None


def logged[T](round_dir: Path, session: str, tool: str, args: dict[str, Any], fn: Callable[[], T]) -> T:
	"""Time ``fn()``, append ``{"ts", "session", "tool", "args", "latency_s"}`` to ``diag_calls.jsonl``, return result."""
	started = time.perf_counter()
	try:
		return fn()
	finally:
		latency_s = round(time.perf_counter() - started, 6)
		log_dir = Path(round_dir)
		log_dir.mkdir(parents=True, exist_ok=True)
		entry = {
			"ts": datetime.now(UTC).isoformat(timespec="seconds"),
			"session": str(session),
			"tool": str(tool),
			"args": args,
			"latency_s": latency_s,
		}
		with (log_dir / "diag_calls.jsonl").open("a", encoding="utf-8") as f:
			f.write(json.dumps(entry, default=str, allow_nan=False) + "\n")


def create_diag_mcp(
	host: str = DEFAULT_HOST,
	port: int = DEFAULT_PORT,
	*,
	default_round_dir: Path | None = None,
):
	"""Create a FastMCP server wrapping the four read-only diagnostic tools with per-session logging."""
	from mcp.server.fastmcp import FastMCP

	mcp = FastMCP("Armigo Failure Diagnosis", host=host, port=port)
	fallback_dir = Path(default_round_dir) if default_round_dir is not None else None

	def _log_dir(round_dir: str | None) -> Path:
		if round_dir:
			return Path(round_dir)
		if fallback_dir is not None:
			return fallback_dir
		raise ValueError("round_dir is required; pass round_dir or configure --round-dir")

	@mcp.tool(name="episode_summary")
	def tool_episode_summary(path: str, session: str, round_dir: str | None = None) -> dict[str, Any]:
		"""Summarize an episode's first_error attribution, physical outcome, and compressed event timeline."""
		ep_path = Path(path)
		log_dir = _log_dir(round_dir)
		return logged(
			log_dir,
			session,
			"episode_summary",
			{"path": str(ep_path), **({"round_dir": round_dir} if round_dir else {})},
			lambda: episode_summary(ep_path),
		)

	@mcp.tool(name="compare_runs")
	def tool_compare_runs(
		a: str,
		b: str,
		session: str,
		tol: float = 1e-6,
		round_dir: str | None = None,
	) -> dict[str, Any]:
		"""Compare two episode runs positionally for discrete equivalence, score deltas, and verdict equality."""
		path_a, path_b = Path(a), Path(b)
		log_dir = _log_dir(round_dir)
		return logged(
			log_dir,
			session,
			"compare_runs",
			{"a": str(path_a), "b": str(path_b), "tol": tol, **({"round_dir": round_dir} if round_dir else {})},
			lambda: compare_runs(path_a, path_b, tol=tol),
		)

	@mcp.tool(name="ledger_clusters")
	def tool_ledger_clusters(round_dir: str, session: str) -> list[dict[str, Any]]:
		"""Read failure clusters from ``<round_dir>/ledger.json``."""
		rdir = Path(round_dir)
		return logged(
			rdir,
			session,
			"ledger_clusters",
			{"round_dir": str(rdir)},
			lambda: ledger_clusters(rdir),
		)

	@mcp.tool(name="counterfactual_result")
	def tool_counterfactual_result(round_dir: str, case_id: str, session: str) -> dict[str, Any] | None:
		"""Look up one case_id in ``<round_dir>/counterfactual.jsonl``."""
		rdir = Path(round_dir)
		return logged(
			rdir,
			session,
			"counterfactual_result",
			{"round_dir": str(rdir), "case_id": case_id},
			lambda: counterfactual_result(rdir, case_id),
		)

	return mcp


def main(argv: list[str] | None = None) -> None:
	parser = argparse.ArgumentParser(description="Read-only failure diagnosis MCP server.")
	parser.add_argument("--transport", choices=("stdio", "streamable-http"), default="stdio")
	parser.add_argument("--host", default=DEFAULT_HOST)
	parser.add_argument("--port", type=int, default=DEFAULT_PORT)
	parser.add_argument("--round-dir", type=Path, default=None, help="Default directory for diag_calls.jsonl.")
	args = parser.parse_args(argv)
	create_diag_mcp(host=args.host, port=args.port, default_round_dir=args.round_dir).run(transport=args.transport)


if __name__ == "__main__":
	main()
