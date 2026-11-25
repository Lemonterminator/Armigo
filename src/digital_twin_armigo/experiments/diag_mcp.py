"""Read-only failure diagnosis MCP for tool-fixing agents and human reviewers (Task H8).

Exposes episode summaries, discrete/score run comparisons, failure ledger clusters, and counterfactual oracle verdicts,
with per-session call accounting to ``<round_dir>/diag_calls.jsonl``. Never connect this server to task agents:
``episode_summary`` reads ``audit.private.json`` and simulator truth diagnostics.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .scoring import _events, score_episode

TOOLS = ("episode_summary", "compare_runs", "ledger_clusters", "counterfactual_result")
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8770

_TIMELINE_KINDS = frozenset({"node", "failure", "tool_call", "view"})
_TIMELINE_LIMIT = 200
_TIMELINE_HALF = 100
_IGNORED_NESTED_TIMING_KEYS = frozenset(
	{
		"wall_time_s",
		"latency_s",
		"elapsed_ms",
		"elapsed_s",
		"inference_ms",
		"score_ms",
		"duration_s",
		"ts",
		"timestamp",
	}
)
_IGNORED_COMPARE_KEYS = _IGNORED_NESTED_TIMING_KEYS | frozenset({"seq"})
_BRANCH_RE = re.compile(r"^branch-(\d+)$")
_ACTION_COMPARE_KEYS = ("state_f64", "action_f64", "state", "action", "segment", "phase")


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
		keys_a = set(va) - _IGNORED_NESTED_TIMING_KEYS
		keys_b = set(vb) - _IGNORED_NESTED_TIMING_KEYS
		if keys_a != keys_b:
			return False
		return all(_values_equal(va[k], vb[k], tol) for k in keys_a)
	if isinstance(va, list) and isinstance(vb, list):
		return len(va) == len(vb) and all(_values_equal(x, y, tol) for x, y in zip(va, vb, strict=True))
	return type(va) is type(vb) and va == vb


def _observations_discrete_equal(oa: dict[str, Any], ob: dict[str, Any], tol: float) -> bool:
	keys_a = set(oa) - _IGNORED_NESTED_TIMING_KEYS
	keys_b = set(ob) - _IGNORED_NESTED_TIMING_KEYS
	if keys_a != keys_b:
		return False
	for k in keys_a:
		val_a = oa[k]
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


def _collect_timing_deltas(
	idx: int,
	node: Any,
	va: Any,
	vb: Any,
	tol: float,
	prefix: str = "",
) -> list[dict[str, Any]]:
	out: list[dict[str, Any]] = []
	if isinstance(va, dict) and isinstance(vb, dict):
		all_keys = sorted(set(va) | set(vb))
		for k in all_keys:
			field = f"{prefix}.{k}" if prefix else str(k)
			if k in _IGNORED_NESTED_TIMING_KEYS:
				ta, tb = va.get(k), vb.get(k)
				if _is_number(ta) and _is_number(tb):
					if not _numbers_equal(ta, tb, tol):
						fa, fb = float(ta), float(tb)
						out.append(
							{
								"index": idx,
								"field": field,
								"node": node,
								"a": fa,
								"b": fb,
								"delta": (fb - fa) if (math.isfinite(fa) and math.isfinite(fb)) else None,
							}
						)
				elif ta != tb:
					out.append({"index": idx, "field": field, "node": node, "a": ta, "b": tb, "delta": None})
			elif k in va and k in vb:
				out.extend(_collect_timing_deltas(idx, node, va[k], vb[k], tol, field))
	elif isinstance(va, list) and isinstance(vb, list) and len(va) == len(vb):
		for j, (xa, xb) in enumerate(zip(va, vb, strict=True)):
			out.extend(_collect_timing_deltas(idx, node, xa, xb, tol, f"{prefix}[{j}]"))
	return out


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


def _load_strict_events(path: Path, side: str, errors: list[str]) -> list[dict[str, Any]] | None:
	if not path.is_file():
		errors.append(f"{side}: missing events.jsonl ({path})")
		return None
	out: list[dict[str, Any]] = []
	try:
		with path.open(encoding="utf-8-sig") as fh:
			for lineno, raw in enumerate(fh, start=1):
				line = raw.strip()
				if not line:
					continue
				try:
					ev = json.loads(line)
				except ValueError as exc:
					errors.append(f"{side}: invalid events.jsonl line {lineno}: {exc}")
					return None
				if not isinstance(ev, dict):
					errors.append(f"{side}: non-object event at events.jsonl line {lineno}")
					return None
				out.append(ev)
	except OSError as exc:
		errors.append(f"{side}: unreadable events.jsonl: {exc}")
		return None
	return out


def _load_strict_episode(path: Path, side: str, errors: list[str]) -> tuple[dict[str, Any] | None, str | None]:
	if not path.is_file():
		errors.append(f"{side}: missing episode.json ({path})")
		return None, None
	try:
		ep = json.loads(path.read_text(encoding="utf-8-sig"))
	except (OSError, ValueError) as exc:
		errors.append(f"{side}: invalid episode.json: {exc}")
		return None, None
	if not isinstance(ep, dict):
		errors.append(f"{side}: episode.json root is not an object")
		return None, None
	res = ep.get("result")
	if not isinstance(res, dict) or not res:
		errors.append(f"{side}: missing result in episode.json")
		return ep, None
	verdict = res.get("verdict_final_cycle")
	if not isinstance(verdict, str) or not verdict.strip():
		verdict = res.get("verdict")
	if not isinstance(verdict, str) or not verdict.strip():
		errors.append(f"{side}: missing verdict in episode.json")
		return ep, None
	return ep, verdict.strip()


def _branch_dirs(ep_dir: Path) -> list[str]:
	if not ep_dir.is_dir():
		return []
	found: list[tuple[int, str]] = []
	for entry in ep_dir.iterdir():
		if entry.is_dir():
			m = _BRANCH_RE.match(entry.name)
			if m:
				found.append((int(m.group(1)), entry.name))
	found.sort()
	return [name for _, name in found]


def _first_array_diff(arr_a: np.ndarray, arr_b: np.ndarray, action_tol: float) -> tuple[int, float | None] | None:
	if action_tol == 0.0 or not (np.issubdtype(arr_a.dtype, np.floating) and np.issubdtype(arr_b.dtype, np.floating)):
		if np.array_equal(arr_a, arr_b):
			return None
		neq = arr_a != arr_b
		row_bad = neq.any(axis=tuple(range(1, neq.ndim))) if neq.ndim > 1 else neq
		first_frame = int(np.flatnonzero(row_bad)[0]) if np.any(row_bad) else 0
		max_abs: float | None = None
		if np.issubdtype(arr_a.dtype, np.number) and np.issubdtype(arr_b.dtype, np.number):
			diff = np.abs(arr_a.astype(np.float64) - arr_b.astype(np.float64))
			finite = diff[np.isfinite(diff)]
			max_abs = float(finite.max()) if finite.size else None
		return first_frame, max_abs

	fa = arr_a.astype(np.float64)
	fb = arr_b.astype(np.float64)
	diff = np.abs(fa - fb)
	both_nan = np.isnan(fa) & np.isnan(fb)
	nonfinite_mismatch = (~np.isfinite(fa) | ~np.isfinite(fb)) & ~both_nan & (fa != fb)
	exceeds = nonfinite_mismatch | (~both_nan & (diff > action_tol))
	if not np.any(exceeds):
		return None
	row_bad = exceeds.any(axis=tuple(range(1, exceeds.ndim))) if exceeds.ndim > 1 else exceeds
	first_frame = int(np.flatnonzero(row_bad)[0]) if np.any(row_bad) else 0
	finite = diff[np.isfinite(diff)]
	max_abs = float(finite.max()) if finite.size else None
	return first_frame, max_abs


def _compare_actions(dir_a: Path, dir_b: Path, *, action_tol: float) -> tuple[bool | None, dict[str, Any] | None]:
	branches_a = _branch_dirs(dir_a)
	branches_b = _branch_dirs(dir_b)
	if not branches_a and not branches_b:
		return None, None
	if branches_a != branches_b:
		return False, {"reason": "branch_mismatch", "branches_a": branches_a, "branches_b": branches_b}

	compared_any = False
	for bname in branches_a:
		npz_a = dir_a / bname / "actions.npz"
		npz_b = dir_b / bname / "actions.npz"
		if npz_a.is_file() != npz_b.is_file():
			return False, {
				"branch": bname,
				"reason": "missing_actions_npz",
				"in_a": npz_a.is_file(),
				"in_b": npz_b.is_file(),
			}
		if not npz_a.is_file():
			continue
		compared_any = True
		with np.load(npz_a, allow_pickle=False) as za, np.load(npz_b, allow_pickle=False) as zb:
			for key in _ACTION_COMPARE_KEYS:
				in_a, in_b = key in za.files, key in zb.files
				if in_a != in_b:
					return False, {
						"branch": bname,
						"key": key,
						"reason": "key_presence_mismatch",
						"in_a": in_a,
						"in_b": in_b,
					}
				if not in_a:
					continue
				arr_a, arr_b = za[key], zb[key]
				if arr_a.shape != arr_b.shape:
					return False, {
						"branch": bname,
						"key": key,
						"reason": "shape_mismatch",
						"shape_a": list(arr_a.shape),
						"shape_b": list(arr_b.shape),
					}
				diff = _first_array_diff(arr_a, arr_b, action_tol)
				if diff is not None:
					first_frame, max_abs = diff
					return False, {
						"branch": bname,
						"key": key,
						"reason": "array_diff",
						"first_frame": first_frame,
						"max_abs_diff": max_abs,
					}
	return (True, None) if compared_any else (None, None)


def compare_runs(a: Path, b: Path, *, tol: float = 1e-6, action_tol: float = 1e-9) -> dict[str, Any]:
	"""Validate completeness, compare discrete events, timing/score deltas, verdicts, and ``branch-*/actions.npz``."""
	dir_a, dir_b = Path(a), Path(b)
	errors: list[str] = []
	events_a = _load_strict_events(dir_a / "events.jsonl", "a", errors)
	_ep_a, verdict_a = _load_strict_episode(dir_a / "episode.json", "a", errors)
	events_b = _load_strict_events(dir_b / "events.jsonl", "b", errors)
	_ep_b, verdict_b = _load_strict_episode(dir_b / "episode.json", "b", errors)

	complete = len(errors) == 0
	verdict_equal = bool(verdict_a and verdict_b and verdict_a == verdict_b)

	first_divergence: dict[str, Any] | None = None
	score_deltas: list[dict[str, Any]] = []
	timing_deltas: list[dict[str, Any]] = []

	if events_a is not None and events_b is not None:
		n = min(len(events_a), len(events_b))
		for i in range(n):
			ea, eb = events_a[i], events_b[i]
			if not _discrete_event_equal(ea, eb, tol):
				first_divergence = {"index": i, "a": ea, "b": eb}
				break
			delta = _score_delta(i, ea, eb, tol)
			if delta is not None:
				score_deltas.append(delta)
			timing_deltas.extend(_collect_timing_deltas(i, ea.get("node"), ea, eb, tol))

		if first_divergence is None and len(events_a) != len(events_b):
			first_divergence = {
				"index": n,
				"a": events_a[n] if n < len(events_a) else None,
				"b": events_b[n] if n < len(events_b) else None,
			}

	actions_equal, action_diff = _compare_actions(dir_a, dir_b, action_tol=action_tol)
	identical_discrete = bool(complete and first_divergence is None)
	identical = bool(complete and identical_discrete and verdict_equal and (actions_equal is not False))

	return {
		"complete": complete,
		"identical": identical,
		"identical_discrete": identical_discrete,
		"first_divergence": first_divergence,
		"score_deltas": score_deltas,
		"timing_deltas": timing_deltas,
		"verdict_equal": verdict_equal,
		"actions_equal": actions_equal,
		"action_diff": action_diff,
		"errors": errors,
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
