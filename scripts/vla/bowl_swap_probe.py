#!/usr/bin/env python
"""Target-bowl swap probe: does the frozen ACT student carry the brick to the bowl it is told to?

Runs the student episode of each case with the evaluation default (clamp-wait, closed-empty exit 0.03) up to its
first ``carry_to_bowl`` segment, checkpoints the simulator there and runs that one segment three times from the same
state, closed loop, with only the bowl instruction changed:

- ``req``: the bowl the episode asked for (its own classification);
- ``req2``: the same bowl again (render / bf16 inference noise control);
- ``swap``: the other bowl.

Each arm records the TCP path, the TCP and brick at segment end, whether the brick was lost, and the true bowl
centres. The episode stops after the probe (no release, no grading). Analysis: ``bowl_swap_report.py``.
Protocol: docs/vla/BOWL_SWAP_PROBE_2026-11-23.md.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
	sys.path.insert(0, str(ROOT / "src"))

from digital_twin_armigo.agent.student_rollout import StudentTwin
from digital_twin_armigo.agent.twin_record import Segment
from digital_twin_armigo.inspection_scorer import default_sup_scorer
from digital_twin_armigo.vla.client_policy import ChunkClientPolicy, RemoteChunkPredictor

PROBE_VERSION = "bowl-swap-probe-v1"
OTHER = {"green": "pink", "pink": "green"}
CLAMP_WAIT_EMPTY_RAD = 0.03


def _eval_module():
	spec = importlib.util.spec_from_file_location("eval_closed_loop", ROOT / "scripts/vla/eval_closed_loop.py")
	mod = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(mod)
	return mod


class ProbeDone(Exception):
	"""Raised after the probe to stop the episode."""


class BowlSwapTwin(StudentTwin):
	"""StudentTwin whose first carry_to_bowl is run once per probe arm from one checkpoint."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.probe: dict | None = None

	def _student_release(self, segs):
		ex = self.expert
		q_lift, _ = self.held
		if np.max(np.abs(ex.last - q_lift)) > 0.01 and (error := self.run_segment(segs["return_to_lift"])):
			return error
		if ex.lost:
			return None
		seg = segs["carry_to_bowl"]
		req = seg.params["bowl"]
		views = copy.deepcopy(self.checkpoints[-1].views) if self.checkpoints else []
		self._save(views)
		cp = self.checkpoints[-1]
		total = self.total_frames
		start = {
			"frame": int(self.ep.n_frames),
			"tcp": self.cell.tcp_pos().tolist(),
			"brick": self.cell.brick_pos().tolist(),
			"q": np.asarray(ex.last).tolist(),
			"retries": int(self.retries),
		}
		arms = []
		for name, bowl in (("req", req), ("req2", req), ("swap", OTHER[req])):
			self.restore_checkpoint(cp, count_as_rewind=False, reason=f"bowl_swap_probe:{name}")
			self.total_frames = total
			path: list[list[float]] = []
			prev_hook = self.on_frame
			self.on_frame = lambda twin, path=path: path.append(twin.cell.tcp_pos().tolist())
			t0 = time.perf_counter()
			try:
				error = self.run_segment(Segment("carry_to_bowl", {**seg.params, "bowl": bowl}))
			finally:
				self.on_frame = prev_hook
			row = self.segments[-1]
			arms.append(
				{
					"arm": name,
					"bowl": bowl,
					"error": error,
					"reason": row.get("reason"),
					"frames": int(row["end"] - row["start"]),
					"lost": bool(self.expert.lost),
					"tcp_end": self.cell.tcp_pos().tolist(),
					"brick_end": self.cell.brick_pos().tolist(),
					"q_end": np.asarray(self.expert.last).tolist(),
					"path": path,
					"wall_s": round(time.perf_counter() - t0, 2),
				}
			)
		lay = self.layout
		self.probe = {
			"version": PROBE_VERSION,
			"requested": req,
			"bowls_true": {"green": list(lay.green), "pink": list(lay.pink)},
			"bowls_perceived": {k: np.asarray(v).tolist() for k, v in self.bowls.items()},
			"brick_xy_initial": list(lay.brick_xy),
			"start": start,
			"arms": arms,
		}
		raise ProbeDone


def run_case(case, scorer, out: Path, policy) -> dict:
	from digital_twin_armigo.agent.twin_graph import Policy, build_graph
	from digital_twin_armigo.agent.twin_mujoco import sample_case
	from digital_twin_armigo.agent.twin_record import EpisodeTrace

	trace = EpisodeTrace(out, video=False)
	twin = BowlSwapTwin(
		sample_case(*case),
		scorer,
		trace,
		policy,
		sensing="realistic",
		clamp_wait=True,
		clamp_wait_empty_rad=CLAMP_WAIT_EMPTY_RAD,
	)
	try:
		status, result = "no_carry", None
		try:
			result = build_graph(twin, Policy()).invoke({}, config={"recursion_limit": 64})["result"]
		except ProbeDone:
			status = "probed"
		except Exception as exc:  # the graph may wrap the exception
			if twin.probe is None:
				raise
			status = f"probed ({type(exc).__name__})"
		return {
			"case": list(case),
			"status": status,
			"probe": twin.probe,
			"result": None if result is None else json.loads(json.dumps(result, default=str)),
		}
	finally:
		twin.close()


def main(argv=None) -> int:
	ev = _eval_module()
	p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	p.add_argument("--suite", type=Path, required=True)
	p.add_argument("--ckpt", type=Path, required=True)
	p.add_argument("--out", type=Path, required=True)
	p.add_argument("--detector", type=Path, default=ev.DEFAULT_DETECTOR)
	p.add_argument("--lerobot-python", type=Path, default=ev.LEROBOT_CUDA_PYTHON)
	p.add_argument("--device", default="cuda")
	p.add_argument("--max-episodes", type=int, default=None)
	p.add_argument("--replan-interval", type=int, default=50)
	p.add_argument("--timeout-s", type=float, default=30.0)
	args = p.parse_args(argv)

	cases = ev.assert_suite_allowed(args.suite)
	if args.max_episodes is not None:
		cases = cases[: args.max_episodes]
	args.out.mkdir(parents=True, exist_ok=True)
	meta = {
		"version": PROBE_VERSION,
		"suite": str(args.suite),
		"suite_sha256": ev.sha256_file(args.suite),
		"ckpt": str(args.ckpt),
		"clamp_wait_empty_rad": CLAMP_WAIT_EMPTY_RAD,
		"replan_interval": args.replan_interval,
		"mode": "sync",
	}
	(args.out / "probe_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
	scorer = default_sup_scorer(args.detector)
	try:
		with (
			ev.managed_policy_server(ckpt=args.ckpt, lerobot_python=args.lerobot_python, device=args.device) as url,
			RemoteChunkPredictor(url, timeout_s=args.timeout_s) as predictor,
		):
			for idx, case in enumerate(cases):
				ep_dir = args.out / "cases" / f"case-{idx:04d}"
				done = ep_dir / "probe.json"
				if done.exists():
					continue
				policy = ChunkClientPolicy(
					predictor, mode="sync", replan_interval=args.replan_interval, stop_policy="latest"
				)
				t0 = time.perf_counter()
				try:
					rec = run_case(case, scorer, ep_dir, policy)
				finally:
					policy.close()
				rec["index"] = idx
				rec["wall_s"] = round(time.perf_counter() - t0, 1)
				done.write_text(json.dumps(rec, allow_nan=False), encoding="utf-8")
				print(f"[{idx + 1}/{len(cases)}] case {case[0]}: {rec['status']} ({rec['wall_s']} s)", flush=True)
	finally:
		if hasattr(scorer, "close"):
			scorer.close()
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
