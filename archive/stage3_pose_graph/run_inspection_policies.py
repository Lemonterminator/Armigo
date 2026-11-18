"""Compare show-pose policies on real MuJoCo episodes at the same extra-pose budget.

Policies: ``none`` (base show poses only), ``fixed`` (a hand-picked list from the pose grid), ``coverage`` (greedy
expected-face-coverage over the grid menu), ``coverage_cond`` (the same with face probabilities conditioned on this
episode's views, :meth:`CoverageTable.face_probs`), ``llm`` / ``llm_cond`` (the local vLLM picks from the same menu
with marginal / conditioned probabilities, through the LangGraph loop via :class:`GraphNextPose`) and ``llm_direct``
(the plain :class:`LLMShowPosePolicy`, no graph). Every policy runs
on the same seeded layouts (defect face uniform over the six faces, as ``inspection_visibility.py``); the label comes
from a :class:`PixelOracleInspector` and each episode is scored by the unchanged evaluator against the ground-truth bowl.

The menu table is built from ``--grid`` (``pose_grid.jsonl`` of an ``inspection_visibility.py`` run); use a different
``--seed`` than that run so the evaluation layouts are not the ones the table was measured on.

Writes ``<out>/episodes.jsonl``, ``<out>/llm_log.jsonl`` (raw LLM exchanges), ``<out>/coverage_table.json`` and
``<out>/summary.json``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import inspection_visibility as iv
from graph import GraphNextPose

from digital_twin_armigo.agent import CoveragePolicy, CoverageTable, FixedSequencePolicy, LLMShowPosePolicy
from digital_twin_armigo.agent.llm import DEFAULT_BASE_URL, OpenAICompatLLM
from digital_twin_armigo.agent.policies import LLMProposer

bpe = iv.bpe
N_BASE = 3
# base poses miss bottom always, -y ~47 %, -x ~32 %, +y ~22 % of layouts: one good grid pose for each, in that order
FIXED_POSES = ((0.0, 120.0), (0.8, 90.0), (0.4, -150.0), (0.8, -90.0))
ALL_POLICIES = ("none", "fixed", "coverage", "coverage_cond", "llm", "llm_cond", "llm_direct")

_W: dict[str, Any] = {}


def _init_worker(table: dict, llm_cfg: dict) -> None:
	from digital_twin_armigo.physics import load_physics_spec

	_W["spec"] = load_physics_spec()
	_W["table"] = CoverageTable.from_dict(table)
	_W["llm_cfg"] = llm_cfg


def _make_policy(name: str, max_extra: int):
	table = _W["table"]
	if name == "none":
		return None, None
	if name == "fixed":
		return FixedSequencePolicy(FIXED_POSES), None
	if name == "coverage":
		return CoveragePolicy(table), None
	if name == "coverage_cond":
		return CoveragePolicy(table, conditional=True, name="coverage_cond"), None
	client = OpenAICompatLLM(**_W["llm_cfg"])
	if name == "llm":
		prop = LLMProposer(client)
		return GraphNextPose(prop, table, max_proposals=max_extra, name="llm"), prop.log
	if name == "llm_cond":
		prop = LLMProposer(client)
		return GraphNextPose(prop, table, max_proposals=max_extra, conditional=True, name="llm_cond"), prop.log
	if name == "llm_direct":
		pol = LLMShowPosePolicy(client, table, budget=max_extra, name="llm_direct")
		return pol, pol.log
	raise ValueError(name)


def run_one(lay, k: int, policy_name: str, max_extra: int, max_frames: int) -> dict[str, Any]:
	"""``iv.run_one`` without the pose grid, with a ``next_pose`` policy."""
	from digital_twin_armigo.benchmarks import SENSITIVITY_MATRIX
	from digital_twin_armigo.constants import SO101_MJCF_DIR
	from digital_twin_armigo.mujoco_cell import Cell
	from digital_twin_armigo.mujoco_cell.inspection_views import PixelOracleInspector, capture_view
	from digital_twin_armigo.mujoco_cell.physics_expert import ExpertParams, run_closed_loop_episode

	spec = _W["spec"]
	params = ExpertParams(max_extra_shows=max_extra, max_frames=max_frames)
	policy, llm_log = _make_policy(policy_name, max_extra)
	base = {c.name: c for c in SENSITIVITY_MATRIX}["baseline_2x2"]
	cell = Cell(lay, SO101_MJCF_DIR, physics=spec, dynamic_clutter=True, assisted_grasp=False, face_probes=True)
	oracle = PixelOracleInspector(lay.defective)

	def inspect(flex: float, roll_deg: float, frame: int):
		# one oracle over both cameras; visible_faces of the record = faces shown in either camera at this pose
		res, seen = None, set()
		for cam, crop in iv.CAMERAS.items():
			res = oracle.observe(capture_view(cell, cam, crop=crop, frame=frame))
			seen.update(res.visible_faces)
		return dataclasses.replace(res, visible_faces=tuple(sorted(seen)))

	bowls = {"normal": lay.green, "defective": lay.pink}
	gt_bowl = lay.pink if lay.defective else lay.green
	t0 = time.perf_counter()
	try:
		tel, phases, info = run_closed_loop_episode(
			cell,
			params,
			q_start=bpe._q_start(),
			brick_xy=lay.brick_xy,
			brick_yaw_deg=lay.brick_yaw_deg,
			non_ideal=bpe.bench._mj_non_ideal(base, spec),
			bowls=bowls,
			inspect=inspect,
			next_pose=policy,
		)
	finally:
		cell.close()
	wall = time.perf_counter() - t0
	if info["infeasible_reason"] is not None:
		verdict, reasons = "INFEASIBLE", [info["infeasible_reason"]]
	else:
		health = bpe.bench._evaluate_mj_rollout(tel, phases, spec, tuple(lay.brick_size), gt_bowl)
		verdict, reasons = health["verdict"], list(health["reason_codes"])
	insp = info.get("inspections", [])
	label = info.get("inspection_label")
	ep: dict[str, Any] = {
		"policy": policy_name,
		"layout_index": k,
		"defective": bool(lay.defective),
		"defect_face": int(lay.defect_face),
		"verdict": verdict,
		"reason_codes": reasons,
		"aborted": info["aborted"],
		"label": label,
		"label_correct": label is not None and label == ("defective" if lay.defective else "normal"),
		"n_frames": int(info["n_frames"]),
		"truncated": bool(info.get("truncated")),
		"n_inspections": len(insp),
		"n_extra_poses": max(0, len(insp) - N_BASE),
		"extra_poses": [[h["flex"], round(h["roll_deg"], 1)] for h in insp[N_BASE:]],
		"observed_faces": sorted(set().union(*(h["observed_faces"] for h in insp))) if insp else [],
		"wall_s": round(wall, 2),
	}
	if llm_log is not None:
		calls = [
			{kk: r.get(kk) for kk in ("repair", "raw", "latency_s", "prompt_tokens", "completion_tokens", "error")}
			for r in llm_log
		]
		ep["llm_calls"] = calls
		if isinstance(policy, GraphNextPose):
			st = policy.state() if policy._started else {}
			ep["llm_invalid"] = int(st.get("n_invalid", 0))
			ep["llm_decision_failures"] = int(st.get("done_reason") == "invalid_output")
			ep["graph_done_reason"] = st.get("done_reason") or ("expert_stopped" if policy._started else "not_called")
			ep["graph_events"] = st.get("events", [])
		else:
			ep["llm_invalid"] = sum(len(f["errors"]) for f in policy.failures) + sum(
				d["repairs"] for d in policy.decisions if d["decision"] is not None
			)
			ep["llm_decision_failures"] = len(policy.failures)
			ep["llm_decisions"] = policy.decisions
		ep["_llm_log"] = [{kk: v for kk, v in r.items() if kk != "t"} for r in llm_log]
	return ep


def _pct(xs: list[float], q: float) -> float | None:
	if not xs:
		return None
	xs = sorted(xs)
	return round(xs[min(len(xs) - 1, round(q * (len(xs) - 1)))], 3)


def summarize(eps: list[dict]) -> dict[str, Any]:
	out: dict[str, Any] = {}
	for pol in dict.fromkeys(e["policy"] for e in eps):
		es = [e for e in eps if e["policy"] == pol]
		n = len(es)
		known = [e for e in es if e["label"] is not None]
		row: dict[str, Any] = {
			"n": n,
			"known_rate": round(len(known) / n, 3),
			"known_rate_defective": round(
				sum(e["label"] is not None for e in es if e["defective"]) / max(sum(e["defective"] for e in es), 1), 3
			),
			"known_rate_normal": round(
				sum(e["label"] is not None for e in es if not e["defective"])
				/ max(sum(not e["defective"] for e in es), 1),
				3,
			),
			"accuracy_of_known": round(sum(e["label_correct"] for e in known) / max(len(known), 1), 3),
			"inspection_unknown_rate": round(sum(e["aborted"] == "inspection_unknown" for e in es) / n, 3),
			"success_rate": round(sum(e["verdict"] == "SUCCESS" for e in es) / n, 3),
			# evaluator SUCCESS on an episode the expert aborted (no label) is a scoring anomaly; count it separately
			"success_with_correct_label_rate": round(
				sum(e["verdict"] == "SUCCESS" and e["label_correct"] for e in es) / n, 3
			),
			"success_but_aborted": sum(e["verdict"] == "SUCCESS" and e["aborted"] is not None for e in es),
			"verdicts": dict(Counter(e["verdict"] for e in es)),
			"aborted": dict(Counter(str(e["aborted"]) for e in es)),
			"mean_extra_poses": round(statistics.fmean(e["n_extra_poses"] for e in es), 3),
			"truncated_rate": round(sum(e["truncated"] for e in es) / n, 3),
			"mean_frames": round(statistics.fmean(e["n_frames"] for e in es), 1),
			"mean_wall_s": round(statistics.fmean(e["wall_s"] for e in es), 2),
		}
		if any("llm_calls" in e for e in es):
			calls = [c for e in es for c in e.get("llm_calls", [])]
			lat = [c["latency_s"] for c in calls if c["latency_s"] is not None]
			row["llm"] = {
				"calls": len(calls),
				"calls_per_episode": round(len(calls) / n, 2),
				"transport_errors": sum(c["error"] is not None for c in calls),
				"invalid_outputs": sum(e["llm_invalid"] for e in es),
				"invalid_output_rate": round(sum(e["llm_invalid"] for e in es) / max(len(calls), 1), 3),
				"decision_failures_after_repair": sum(e["llm_decision_failures"] for e in es),
				"latency_s_mean": round(statistics.fmean(lat), 3) if lat else None,
				"latency_s_p50": _pct(lat, 0.5),
				"latency_s_p95": _pct(lat, 0.95),
				"prompt_tokens_mean": round(statistics.fmean(c["prompt_tokens"] for c in calls), 1) if calls else None,
				"completion_tokens_mean": round(statistics.fmean(c["completion_tokens"] for c in calls), 1)
				if calls
				else None,
			}
		out[pol] = row
	return out


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	ap.add_argument("--n", type=int, default=10)
	ap.add_argument("--seed", type=int, default=7, help="layout seed (keep it different from the --grid run's)")
	ap.add_argument("--policies", default="none,fixed,coverage,llm")
	ap.add_argument("--max-extra", type=int, default=4, help="ExpertParams.max_extra_shows (same for all policies)")
	ap.add_argument(
		"--grid", type=Path, default=ROOT / "results" / "inspection-visibility" / "base-poses" / "pose_grid.jsonl"
	)
	ap.add_argument(
		"--max-frames",
		type=int,
		default=1600,
		help="ExpertParams.max_frames for every policy (900 truncates extras; 1300 still truncated 4 slow extras)",
	)
	ap.add_argument("--workers", type=int, default=1, help="parallel episodes (<= 4: also caps LLM concurrency)")
	ap.add_argument("--llm-url", default=DEFAULT_BASE_URL)
	ap.add_argument("--llm-model", default="qwen3.8-vllm")
	ap.add_argument("--llm-thinking", action="store_true", help="enable Qwen thinking (default off)")
	ap.add_argument("--out", type=Path, default=ROOT / "results" / "inspection-policies" / "run")
	args = ap.parse_args()
	pols = [p.strip() for p in args.policies.split(",") if p.strip()]
	assert all(p in ALL_POLICIES for p in pols), pols
	workers = max(1, min(args.workers, 4))

	table = CoverageTable.from_pose_grid(args.grid)
	args.out.mkdir(parents=True, exist_ok=True)
	table.save(args.out / "coverage_table.json")
	llm_cfg = {
		"base_url": args.llm_url,
		"model": args.llm_model,
		"enable_thinking": bool(args.llm_thinking),
		"max_tokens": 4096 if args.llm_thinking else 512,
		"timeout_s": 300.0 if args.llm_thinking else 120.0,
	}
	lays = iv._layouts(args.n, args.seed, bpe.load_top_camera())
	jobs = [(p, k) for k in range(len(lays)) for p in pols]
	eps: list[dict] = []
	t0 = time.perf_counter()
	with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(table.to_dict(), llm_cfg)) as pool:
		futs = {pool.submit(run_one, lays[k], k, p, args.max_extra, args.max_frames): (p, k) for p, k in jobs}
		for fut in as_completed(futs):
			ep = fut.result()
			eps.append(ep)
			print(
				f"[{ep['layout_index']:3d} {ep['policy']:10s}] def={int(ep['defective'])} "
				f"face={iv.FACE_NAMES[ep['defect_face']]:6s} {ep['verdict']:10s} label={ep['label']} "
				f"extra={ep['n_extra_poses']} aborted={ep['aborted']} ({time.perf_counter() - t0:.0f}s, "
				f"{len(eps)}/{len(jobs)})",
				flush=True,
			)
	eps.sort(key=lambda e: (e["layout_index"], pols.index(e["policy"])))
	with (args.out / "llm_log.jsonl").open("w", encoding="utf-8") as f:
		for e in eps:
			for r in e.pop("_llm_log", []):
				f.write(json.dumps({"policy": e["policy"], "layout_index": e["layout_index"], **r}) + "\n")
	(args.out / "episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in eps), encoding="utf-8")
	summary = {
		"policies": summarize(eps),
		"config": {
			"n": args.n,
			"seed": args.seed,
			"max_extra_shows": args.max_extra,
			"max_frames": args.max_frames,
			"grid": str(args.grid),
			"menu_size": len(table.options),
			"table_layouts": table.n_layouts,
			"fixed_poses": [list(p) for p in FIXED_POSES],
			"llm": {k: v for k, v in llm_cfg.items() if k != "timeout_s"},
			"workers": workers,
			"wall_s": round(time.perf_counter() - t0, 1),
		},
	}
	(args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
	print(json.dumps(summary, indent=2))


if __name__ == "__main__":
	main()
