"""Build a deterministic failure-bank suite from a failure ledger cluster (Task F3).

Usage:
	python scripts/experiments/failure_bank.py --ledger <ledger.json> --cluster <key> --round 01 --max 30
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
	sys.path.insert(0, str(REPO / "src"))

from digital_twin_armigo.experiments.contract import case_id as make_case_id
from digital_twin_armigo.experiments.ledger import _resolve_review
from digital_twin_armigo.experiments.runner import code_commit
from digital_twin_armigo.experiments.triage import TriageRow
from digital_twin_armigo.inventory import audit

DEFAULT_BANK_DIR = REPO / "config" / "agent_bench" / "failure_bank"
DEFAULT_MANIFEST = REPO / "config" / "agent_bench" / "v2" / "manifest.json"
DEFAULT_MAX_CASES = 30
LATE_FAILURE_STAGES = frozenset({"place", "carry", "carry_to_bowl", "park"})
LATE_FAILURE_ROOT_CAUSES = frozenset({"wrong_bowl_carry", "placement_failure"})
TRAJ_COMPARE_KEYS = ("state_f64", "action_f64", "state", "action", "segment", "phase")
_BRANCH_RE = re.compile(r"^branch-(\d+)$")
_NONE_TOKENS = frozenset({"", "none", "null", "-"})


def _load_make_suites():
	path = Path(__file__).resolve().parent / "make_suites.py"
	spec = importlib.util.spec_from_file_location("make_suites", path)
	if spec is None or spec.loader is None:
		raise RuntimeError(f"cannot load {path}")
	mod = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(mod)
	return mod


def _format_round(value: str) -> str:
	raw = value.strip()
	if raw.isdigit():
		return f"round-{int(raw):02d}"
	return raw if raw.startswith("round-") else f"round-{raw}"


def cluster_slug(key: Sequence[str | None]) -> str:
	parts = []
	for item in key:
		token = "none" if item is None or item == "" else str(item)
		parts.append(re.sub(r"[^A-Za-z0-9_.-]+", "_", token))
	return "-".join(parts)


def _normalize_code(token: str | None) -> str | None:
	if token is None:
		return None
	if token.strip().lower() in _NONE_TOKENS:
		return None
	return token.strip()


def match_cluster(ledger: dict, cluster_spec: str) -> dict:
	"""Resolve a ``--cluster`` specification against ``ledger['clusters']``."""
	clusters = ledger.get("clusters") or []
	if not clusters:
		raise ValueError("ledger has no clusters")
	spec = cluster_spec.strip()

	if spec.startswith("["):
		try:
			parsed = json.loads(spec)
		except ValueError as exc:
			raise ValueError(f"invalid JSON cluster key {cluster_spec!r}: {exc}") from exc
		if not isinstance(parsed, list) or len(parsed) != 3:
			raise ValueError(f"JSON cluster key must have 3 elements, got {parsed!r}")
		target = [str(parsed[0]), None if parsed[1] is None else str(parsed[1]), str(parsed[2])]
		for c in clusters:
			if c.get("key") == target:
				return c

	for sep in (":", "/"):
		if sep in spec:
			parts = spec.split(sep)
			if len(parts) == 3:
				stage, raw_code, cause = (p.strip() for p in parts)
				norm_code = _normalize_code(raw_code)
				for c in clusters:
					ck = c.get("key") or []
					if len(ck) == 3 and ck[0] == stage and ck[2] == cause and ck[1] in (norm_code, raw_code):
						return c

	for c in clusters:
		ck = c.get("key") or []
		if len(ck) == 3 and cluster_slug(ck) == spec:
			return c

	by_cause = [c for c in clusters if len(c.get("key") or []) == 3 and c["key"][2] == spec]
	if len(by_cause) == 1:
		return by_cause[0]
	if len(by_cause) > 1:
		keys = [c["key"] for c in by_cause]
		raise ValueError(f"ambiguous cluster {cluster_spec!r}; matching keys: {keys}")

	available = [c.get("key") for c in clusters]
	raise ValueError(f"cluster {cluster_spec!r} not found in ledger; available: {available}")


def collect_cluster_traj_ids(
	ledger_path: Path,
	cluster: dict,
	*,
	audit_db: Path | None = None,
) -> list[str]:
	"""Gather all episode traj_ids in ``cluster``, expanding beyond ``examples[:10]`` via sibling ``triage.jsonl``."""
	seen: set[str] = set()
	ordered: list[str] = []
	for tid in cluster.get("examples") or []:
		if tid not in seen:
			seen.add(tid)
			ordered.append(tid)

	triage_path = Path(ledger_path).parent / "triage.jsonl"
	target_key = tuple(cluster["key"])
	if triage_path.is_file():
		has_audit = audit_db is not None and Path(audit_db).is_file()
		with triage_path.open(encoding="utf-8") as fh:
			for line in fh:
				line = line.strip()
				if not line:
					continue
				raw = json.loads(line)
				tid = str(raw["traj_id"])
				row = TriageRow(
					traj_id=tid,
					stage=str(raw.get("stage") or "unknown"),
					code=raw.get("code"),
					root_cause=str(raw.get("root_cause") or "unclassified"),
					evidence=dict(raw.get("evidence") or {}),
					rule=str(raw.get("rule") or "unclassified"),
				)
				effective_cause = row.root_cause
				if has_audit:
					effective_cause, _ = _resolve_review(row, audit.latest(audit_db, tid))
				if (row.stage, row.code, effective_cause) == target_key and tid not in seen:
					seen.add(tid)
					ordered.append(tid)
	return ordered


def _infer_search_roots(ledger_path: Path, episodes_root: Path | None) -> list[Path]:
	if episodes_root is not None:
		return [Path(episodes_root)]
	roots: list[Path] = []
	for parent in (ledger_path.parent, *ledger_path.parents):
		if (parent / "runs").is_dir():
			roots.append(parent)
			break
	roots.append(ledger_path.parent)
	return roots


def resolve_episode_dir(traj_id: str, search_roots: Sequence[Path]) -> Path | None:
	name = Path(traj_id).name
	for root in search_roots:
		for candidate in (root / traj_id, root / "runs" / traj_id, root / name):
			if candidate.is_dir() and any(
				(candidate / marker).is_file() for marker in ("task.json", "episode.json", "events.jsonl")
			):
				return candidate
	return None


def _normalize_case(case_raw: Sequence) -> list:
	if len(case_raw) not in (3, 4):
		raise ValueError(f"invalid case length: {case_raw!r}")
	base = [int(case_raw[0]), bool(case_raw[1]), int(case_raw[2])]
	if len(case_raw) == 4 and isinstance(case_raw[3], dict) and case_raw[3]:
		return [*base, dict(case_raw[3])]
	return base


def _read_json(path: Path) -> dict:
	if not path.is_file():
		return {}
	data = json.loads(path.read_text(encoding="utf-8-sig"))
	return data if isinstance(data, dict) else {}


def _late_failure_frame(events_path: Path) -> int | None:
	failure_frames: list[int] = []
	late_frames: list[int] = []
	place_frames: list[int] = []
	with events_path.open(encoding="utf-8") as fh:
		for line in fh:
			if not line.endswith("\n"):
				break
			ev = json.loads(line)
			kind = ev.get("kind")
			frame = ev.get("frame")
			if not isinstance(frame, int):
				continue
			if kind == "node" and ev.get("node") == "place":
				place_frames.append(frame)
			elif kind == "failure":
				failure_frames.append(frame)
				if ev.get("stage") in LATE_FAILURE_STAGES or ev.get("node") in LATE_FAILURE_STAGES:
					late_frames.append(frame)
	if late_frames:
		return late_frames[0]
	if failure_frames:
		return failure_frames[-1]
	if place_frames:
		return place_frames[0]
	return None


def select_resume_checkpoint(ep_dir: Path, cluster_key: Sequence[str | None]) -> str | None:
	"""Pick the latest valid saved checkpoint before a late-stage failure (``place`` / ``carry``)."""
	stage = cluster_key[0] if len(cluster_key) > 0 else None
	cause = cluster_key[2] if len(cluster_key) > 2 else None
	if stage not in LATE_FAILURE_STAGES and cause not in LATE_FAILURE_ROOT_CAUSES:
		return None

	events_path = ep_dir / "events.jsonl"
	ck_dir = ep_dir / "checkpoints"
	if not events_path.is_file() or not ck_dir.is_dir():
		return None

	from digital_twin_armigo.agent.twin_resume import live_checkpoint_ids, load_checkpoint

	live_ids = live_checkpoint_ids(events_path)
	if not live_ids:
		return None

	fail_frame = _late_failure_frame(events_path)
	for cid in reversed(live_ids):
		if not ((ck_dir / f"{cid}.json").is_file() and (ck_dir / f"{cid}.npy").is_file()):
			continue
		try:
			ck = load_checkpoint(ep_dir, cid)
		except (OSError, ValueError, KeyError, TypeError):
			continue
		depth_name = (ck.agent or {}).get("empty_bowl_depth")
		if depth_name and not (ck_dir / depth_name).is_file():
			continue
		if fail_frame is not None and ck.frame > fail_frame:
			continue
		return cid
	return None


def _resolve_scorer_spec(
	scorer_arg: str | None,
	detector_name: str | None,
	manifest_path: Path | None,
) -> dict:
	if scorer_arg == "fixed":
		return {"kind": "fixed", "score": -1.0, "thresholds": {"top": 0.0, "wrist_cam": 0.0}}
	if scorer_arg:
		raw = scorer_arg.removeprefix("sup:")
		p = Path(raw)
		if not p.is_absolute():
			p = (REPO / p).resolve()
		return {"kind": "sup", "path": str(p)}
	if detector_name:
		det_path = None
		if manifest_path is not None and manifest_path.is_file():
			manifest = _read_json(manifest_path)
			rel = (manifest.get("detectors") or {}).get(detector_name)
			if rel:
				det_path = (REPO / rel).resolve()
		if det_path is None:
			candidate = Path(detector_name)
			det_path = candidate if candidate.is_absolute() else (REPO / candidate).resolve()
		if det_path.is_file():
			return {"kind": "sup", "path": str(det_path)}
	return {"kind": "fixed", "score": -1.0, "thresholds": {"top": 0.0, "wrist_cam": 0.0}}


def extract_case_spec(
	ep_dir: Path,
	traj_id: str,
	cluster_key: Sequence[str | None],
	*,
	scorer_arg: str | None = None,
	manifest_path: Path | None = None,
) -> dict:
	"""Extract a self-contained reproduction spec from an episode directory."""
	task = _read_json(ep_dir / "task.json")
	episode = _read_json(ep_dir / "episode.json")
	plan = _read_json(ep_dir / "plan.json") or _read_json(ep_dir.parent / "plan.json")

	case_raw = task.get("case")
	if not isinstance(case_raw, list):
		raise TypeError(f"{ep_dir}: missing task.json 'case'")
	case = _normalize_case(case_raw)
	cid = make_case_id(*case[:3])

	prov = episode.get("provenance") or {}
	exp = prov.get("experiment") or {}
	policy = task.get("policy") or prov.get("policy") or plan.get("policy") or {}
	sensing = exp.get("sensing") or prov.get("sensing") or plan.get("sensing") or "realistic"
	agent_options = dict(prov.get("agent_options") or plan.get("agent_options") or {})
	strategy = exp.get("strategy") or plan.get("strategy")
	if strategy and "strategy" not in agent_options:
		agent_options["strategy"] = strategy
	detector_name = exp.get("detector") or plan.get("detector")

	resume_from = select_resume_checkpoint(ep_dir, cluster_key)
	scorer_spec = _resolve_scorer_spec(scorer_arg, detector_name, manifest_path)

	return {
		"traj_id": traj_id,
		"case_id": cid,
		"case": case,
		"policy": policy,
		"sensing": sensing,
		"agent_options": agent_options,
		"scorer": scorer_spec,
		"episode_dir": str(ep_dir),
		"resume_from": resume_from,
	}


def _build_scorer(scorer_spec: dict):
	from digital_twin_armigo.inspection_scorer import FixedScorer, default_sup_scorer

	if scorer_spec.get("kind") == "sup":
		return default_sup_scorer(scorer_spec["path"])
	score_val = float(scorer_spec.get("score", -1.0))
	thresholds = scorer_spec.get("thresholds") or {"top": 0.0, "wrist_cam": 0.0}
	return FixedScorer(lambda _cam, _img, v=score_val: v, thresholds)


def _run_one_episode(spec: dict, out_dir: Path) -> None:
	"""Execute one episode (full or resumed from ``resume_from``) in the current process."""
	from langgraph.checkpoint.memory import InMemorySaver

	from digital_twin_armigo.agent.student_rollout import ExpertTracer
	from digital_twin_armigo.agent.twin_graph import Policy, build_graph
	from digital_twin_armigo.agent.twin_mujoco import sample_case
	from digital_twin_armigo.agent.twin_record import EpisodeTrace, write_json
	from digital_twin_armigo.agent.twin_resume import (
		ResumePlan,
		live_checkpoint_ids,
		load_checkpoint,
		load_history,
		restore,
	)

	case = tuple(spec["case"])
	policy = Policy(**(spec.get("policy") or {}))
	sensing = spec.get("sensing", "realistic")
	agent_options = dict(spec.get("agent_options") or {})
	scorer = _build_scorer(spec.get("scorer") or {"kind": "fixed"})
	trace = EpisodeTrace(out_dir, video=False)
	backend = None
	try:
		write_json(
			trace.path / "task.json",
			{"case": list(case), "policy": policy.__dict__, "video": False, "use_llm": False},
		)
		backend = ExpertTracer(sample_case(*case), scorer, trace, sensing=sensing, **agent_options)
		resume_id = spec.get("resume_from")
		link = None
		saver = None
		if resume_id:
			parent_dir = Path(spec["episode_dir"])
			live_ids = live_checkpoint_ids(parent_dir / "events.jsonl")
			if resume_id not in live_ids:
				raise ValueError(f"checkpoint {resume_id!r} not in live checkpoints {live_ids}")
			checkpoints = [load_checkpoint(parent_dir, name) for name in live_ids[: live_ids.index(resume_id) + 1]]
			history, depth = load_history(parent_dir, checkpoints[-1])
			plan = ResumePlan(parent_dir, checkpoints, [], None, history, depth)
			link = restore(backend, plan)
			write_json(trace.path / "resume.json", link)
			saver = InMemorySaver()
		graph = build_graph(backend, policy, saver)
		config: dict = {"recursion_limit": 64}
		if saver is not None:
			config["configurable"] = {"thread_id": "failure-bank-repro"}
		if resume_id:
			graph.update_state(config, plan.graph_state(), as_node="prepare")
			result = graph.invoke(None, config=config)["result"]
			result["student_eligible"] = False
		else:
			result = graph.invoke({}, config=config)["result"]
		trace.finish(
			result,
			{
				"engine": getattr(backend, "engine", "mujoco"),
				"sensing": sensing,
				"agent_options": agent_options,
				**({"resumed_from": link} if link else {}),
			},
		)
	except BaseException:
		trace.close()
		raise
	finally:
		if backend is not None:
			backend.close()
		if hasattr(scorer, "close"):
			scorer.close()


def run_repro_subprocess(spec: dict, out_dir: Path) -> None:
	"""Spawn a fresh Python subprocess on the current commit to reproduce one episode."""
	out_dir.parent.mkdir(parents=True, exist_ok=True)
	spec_path = out_dir.parent / f".{out_dir.name}.spec.json"
	spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
	env = {
		**os.environ,
		"PYTHONPATH": str(REPO / "src"),
		"PYTHONHASHSEED": "0",
	}
	subprocess.run(
		[sys.executable, str(Path(__file__).resolve()), "--run-one", str(spec_path), "--out", str(out_dir)],
		cwd=str(REPO),
		env=env,
		check=True,
	)


def _branch_dirs(ep_dir: Path) -> list[str]:
	found: list[tuple[int, str]] = []
	for entry in ep_dir.iterdir():
		if entry.is_dir():
			m = _BRANCH_RE.match(entry.name)
			if m:
				found.append((int(m.group(1)), entry.name))
	found.sort()
	return [name for _, name in found]


def compare_episode_trajectories(dir_a: Path, dir_b: Path) -> tuple[bool, str | None]:
	"""Check bitwise equality of ``actions.npz`` trajectories and outcomes across two runs."""
	branches_a = _branch_dirs(dir_a)
	branches_b = _branch_dirs(dir_b)
	if not branches_a:
		return False, "missing branch-* output in run 1"
	if branches_a != branches_b:
		return False, f"branch mismatch: {branches_a} vs {branches_b}"

	for bname in branches_a:
		npz_a = dir_a / bname / "actions.npz"
		npz_b = dir_b / bname / "actions.npz"
		if not npz_a.is_file() or not npz_b.is_file():
			return False, f"missing actions.npz in {bname}"
		with np.load(npz_a, allow_pickle=False) as za, np.load(npz_b, allow_pickle=False) as zb:
			if "state_f64" not in za.files and "state" not in za.files:
				return False, f"{bname}: missing state arrays"
			for key in TRAJ_COMPARE_KEYS:
				in_a, in_b = key in za.files, key in zb.files
				if in_a != in_b:
					return False, f"{bname}: array {key} presence mismatch ({in_a} vs {in_b})"
				if not in_a:
					continue
				if za[key].shape != zb[key].shape:
					return False, f"{bname}: {key} shape mismatch {za[key].shape} vs {zb[key].shape}"
				if not np.array_equal(za[key], zb[key]):
					return False, f"{bname}: {key} differs bitwise"

	res_a = _read_json(dir_a / "episode.json").get("result") or {}
	res_b = _read_json(dir_b / "episode.json").get("result") or {}
	for field in ("verdict", "label", "destination"):
		if res_a.get(field) != res_b.get(field):
			return False, f"result.{field} mismatch: {res_a.get(field)!r} vs {res_b.get(field)!r}"
	return True, None


def build_failure_bank(
	*,
	ledger_path: Path,
	cluster_spec: str,
	round_label: str,
	max_cases: int = DEFAULT_MAX_CASES,
	out_path: Path | None = None,
	manifest_path: Path | None = DEFAULT_MANIFEST,
	episodes_root: Path | None = None,
	scratch_dir: Path | None = None,
	scorer_arg: str | None = None,
	update_manifest: bool = True,
) -> tuple[dict, Path]:
	"""Verify reproducibility across 2 fresh subprocesses and write a failure bank suite."""
	ledger_path = Path(ledger_path)
	ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
	cluster = match_cluster(ledger, cluster_spec)
	cluster_key = list(cluster["key"])
	round_name = _format_round(round_label)
	slug = cluster_slug(cluster_key)

	if out_path is None:
		out_path = DEFAULT_BANK_DIR / f"{round_name}-{slug}.json"
	else:
		out_path = Path(out_path)
	suite_name = out_path.stem

	search_roots = _infer_search_roots(ledger_path, episodes_root)
	audit_db = None
	for root in search_roots:
		cand = root / "audit" / "reviews.sqlite"
		if cand.is_file():
			audit_db = cand
			break
	traj_ids = collect_cluster_traj_ids(ledger_path, cluster, audit_db=audit_db)

	commit = code_commit()
	kept_cases: list[list] = []
	kept_meta: list[dict] = []
	dropped_meta: list[dict] = []
	resume_map: dict[str, str] = {}
	seen_case_ids: set[str] = set()

	def _check_all(work_root: Path) -> None:
		for tid in traj_ids:
			if len(kept_cases) >= max_cases:
				break
			ep_dir = resolve_episode_dir(tid, search_roots)
			if ep_dir is None:
				dropped_meta.append({"traj_id": tid, "case_id": None, "case": None, "reason": "episode_dir_not_found"})
				continue
			spec = extract_case_spec(
				ep_dir,
				tid,
				cluster_key,
				scorer_arg=scorer_arg,
				manifest_path=manifest_path,
			)
			cid = spec["case_id"]
			if cid in seen_case_ids:
				continue
			seen_case_ids.add(cid)

			case_work = work_root / cid
			run1_dir = case_work / "run-1"
			run2_dir = case_work / "run-2"
			try:
				run_repro_subprocess({**spec, "run_index": 1}, run1_dir)
				run_repro_subprocess({**spec, "run_index": 2}, run2_dir)
			except Exception as exc:  # noqa: BLE001 - record failed reproduction cleanly
				dropped_meta.append(
					{
						"case_id": cid,
						"traj_id": tid,
						"case": spec["case"],
						"resume_from": spec["resume_from"],
						"reason": f"repro_error: {type(exc).__name__}: {exc}",
					}
				)
				continue

			ok, reason = compare_episode_trajectories(run1_dir, run2_dir)
			if not ok:
				dropped_meta.append(
					{
						"case_id": cid,
						"traj_id": tid,
						"case": spec["case"],
						"resume_from": spec["resume_from"],
						"reason": reason or "trajectory_mismatch",
					}
				)
				continue

			kept_cases.append(spec["case"])
			branches = _branch_dirs(run1_dir)
			entry = {
				"case_id": cid,
				"traj_id": tid,
				"case": spec["case"],
				"bitwise_equal": True,
				"branches": branches,
				"resume_from": spec["resume_from"],
			}
			kept_meta.append(entry)
			if spec["resume_from"]:
				resume_map[cid] = spec["resume_from"]

	if scratch_dir is not None:
		work_root = Path(scratch_dir)
		work_root.mkdir(parents=True, exist_ok=True)
		_check_all(work_root)
	else:
		with tempfile.TemporaryDirectory(prefix="armigo-failure-bank-") as tmp_dir:
			_check_all(Path(tmp_dir))

	payload: dict = {
		"name": suite_name,
		"sealed": False,
		"source_round": round_name,
		"cluster": cluster_key,
		"commit": commit,
		"cases": kept_cases,
		"repro_checked": {
			"commit": commit,
			"runs_per_case": 2,
			"checked_at": datetime.now(UTC).isoformat(timespec="seconds"),
			"kept": kept_meta,
			"dropped": dropped_meta,
		},
	}
	if resume_map:
		payload["resume_from"] = resume_map

	out_path.parent.mkdir(parents=True, exist_ok=True)
	out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")

	if update_manifest and manifest_path is not None:
		ms = _load_make_suites()
		ms.register_suite(Path(manifest_path), suite_name, out_path, kept_cases, sealed=False)

	return payload, out_path


def main(argv: list[str] | None = None) -> int:
	p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	p.add_argument("--ledger", help="Path to ledger.json")
	p.add_argument("--cluster", help="Cluster key (stage:code:root_cause, stage/code/root_cause, JSON, or slug)")
	p.add_argument("--round", help="Loop round number or label (e.g. 01)")
	p.add_argument("--max", type=int, default=DEFAULT_MAX_CASES, help="Max reproducible cases to keep (default: 30)")
	p.add_argument("--out", help="Output suite JSON path")
	p.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Manifest JSON path to register the suite")
	p.add_argument("--episodes-root", help="Root directory containing runs/ or episode directories")
	p.add_argument("--scratch-dir", help="Scratch directory for 2-run reproduction traces (default: temp dir)")
	p.add_argument("--scorer", help="Scorer override ('fixed' or detector spec JSON path)")
	p.add_argument("--no-manifest", action="store_true", help="Do not update manifest.json")
	p.add_argument("--run-one", help=argparse.SUPPRESS)
	a = p.parse_args(argv)

	if a.run_one:
		if not a.out:
			raise SystemExit("--run-one requires --out")
		spec = json.loads(Path(a.run_one).read_text(encoding="utf-8"))
		_run_one_episode(spec, Path(a.out))
		return 0

	if not (a.ledger and a.cluster and a.round):
		p.error("--ledger, --cluster, and --round are required")

	payload, out_path = build_failure_bank(
		ledger_path=Path(a.ledger),
		cluster_spec=a.cluster,
		round_label=a.round,
		max_cases=a.max,
		out_path=Path(a.out) if a.out else None,
		manifest_path=Path(a.manifest) if a.manifest else None,
		episodes_root=Path(a.episodes_root) if a.episodes_root else None,
		scratch_dir=Path(a.scratch_dir) if a.scratch_dir else None,
		scorer_arg=a.scorer,
		update_manifest=not a.no_manifest,
	)
	repro = payload["repro_checked"]
	print(
		f"{out_path}: kept {len(repro['kept'])}/{len(repro['kept']) + len(repro['dropped'])} cases "
		f"(dropped {len(repro['dropped'])})"
	)
	return 0


if __name__ == "__main__":
	sys.exit(main())
