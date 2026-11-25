"""ACT grasp vs scripted grasp: the first grasp + lift of v3 cases under realistic sensing (agent/act_grasp.py).

Both arms run ``prepare`` then the twin's grasp skill on the same cases (cameras ``drift``, bowls from the oracle:
the comparison is about the grasp only). ``--ckpt`` adds the student arm through a spawned policy server (GPU, the
LeRobot CUDA environment); without it only the scripted arm runs (CPU). Writes ``<out>/<arm>/results.jsonl`` and
``<out>/summary.json``.

	PYTHONPATH=src python scripts/vla/eval_act_grasp.py --seed0 3001201 --n 60 --out C:/scratch/reverse-act/eval \
		[--ckpt C:/scratch/reverse-act/train-01/best]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
	sys.path.insert(0, str(ROOT / "src"))


def make_case(seed: int, split: str = "train") -> list:
	return [seed, bool(seed % 2), seed % 6, {"scene": "v3", "split": split, "cameras": "drift", "oracle": "bowls"}]


def _scripted(args: tuple) -> dict:
	os.environ["CUDA_VISIBLE_DEVICES"] = ""
	from digital_twin_armigo.agent.act_grasp import grasp_trial

	seed, split, out, overrides, ring, localise, vlm_url = args
	llm = None
	if vlm_url:
		from digital_twin_armigo.agent.llm import OpenAICompatLLM

		llm = OpenAICompatLLM(vlm_url, max_tokens=32, timeout_s=60, retries=1)
	try:
		return grasp_trial(
			make_case(seed, split),
			out / f"case-{seed}",
			expert_overrides=overrides,
			ring_localise=ring,
			localise=localise,
			perception_llm=llm,
		)
	except Exception as e:  # noqa: BLE001 - a crashed trial is a reported failure
		return {"case": make_case(seed, split), "lifted": False, "error": f"crash:{type(e).__name__}:{e}"}


def _student(args: tuple) -> dict:
	from digital_twin_armigo.agent.act_grasp import CloseBeforeDoneGuard, grasp_trial
	from digital_twin_armigo.vla.client_policy import ChunkClientPolicy, RemoteChunkPredictor

	seed, split, out, url, policy_kw, guard, ring = args
	with RemoteChunkPredictor(url, timeout_s=30.0) as predictor:
		policy = ChunkClientPolicy(predictor, **policy_kw)
		if guard:
			policy = CloseBeforeDoneGuard(policy)
		try:
			r = grasp_trial(make_case(seed, split), out / f"case-{seed}", policy=policy, ring_localise=ring)
			r["done_overrides"] = getattr(policy, "overrides", 0)
			return r
		except Exception as e:  # noqa: BLE001
			return {"case": make_case(seed, split), "lifted": False, "error": f"crash:{type(e).__name__}:{e}"}
		finally:
			policy.close()


def summarize(rows: list[dict]) -> dict:
	n = len(rows)
	ok = sum(r["lifted"] for r in rows)
	first = sum(r["lifted"] and not r.get("grasp_retries") for r in rows)
	fails = Counter(str(r.get("aborted") or r.get("error")) for r in rows if not r["lifted"])
	return {"n": n, "lifted": ok, "rate": ok / n if n else None, "lifted_first_attempt": first, "failures": dict(fails)}


def main(argv: list[str] | None = None) -> int:
	ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	ap.add_argument("--out", type=Path, required=True)
	ap.add_argument("--seed0", type=int, default=3001201)
	ap.add_argument("--n", type=int, default=60)
	ap.add_argument("--split", default="train")
	ap.add_argument("--workers", type=int, default=8)
	ap.add_argument("--ckpt", type=Path, default=None, help="student checkpoint: adds the ACT arm")
	ap.add_argument("--skip-scripted", action="store_true")
	ap.add_argument("--replan-interval", type=int, default=None)
	ap.add_argument("--temporal-ensemble", type=float, default=None)
	ap.add_argument("--guard", action="store_true", help="ignore a grasp end-step before the gripper closes")
	ap.add_argument(
		"--descend-close-ratio",
		type=float,
		default=None,
		help="scripted arm: close the jaws while descending to this x the believed brick width (ExpertParams)",
	)
	ap.add_argument(
		"--ring", action="store_true", help="re-localise the brick from a ring of wrist views before the grasp"
	)
	ap.add_argument(
		"--localise",
		choices=("top", "vote"),
		default="top",
		help="vote: the localize node (shape rule + VLM, wrist confirmation, ring, pin) after prepare (scripted arm)",
	)
	ap.add_argument("--vlm-url", default=None, help="OpenAI-compatible VLM for --localise vote (none: shape rule only)")
	a = ap.parse_args(argv)
	ring_tag = ("-ring" if a.ring else "") + ("-vote" if a.localise == "vote" else "")
	vlm_url = a.vlm_url if a.localise == "vote" else None
	seeds = list(range(a.seed0, a.seed0 + a.n))
	summary_path = a.out / "summary.json"
	summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
	arms: list[tuple[str, list[dict]]] = []
	if not a.skip_scripted:
		overrides = {"descend_close_ratio": a.descend_close_ratio} if a.descend_close_ratio else None
		name = "scripted" + (f"-close{a.descend_close_ratio:g}" if overrides else "") + ring_tag
		out = a.out / name
		out.mkdir(parents=True, exist_ok=True)
		with ProcessPoolExecutor(a.workers) as pool:
			rows = list(pool.map(_scripted, [(s, a.split, out, overrides, a.ring, a.localise, vlm_url) for s in seeds]))
		arms.append((name, rows))
	if a.ckpt is not None:
		sys.path.insert(0, str(ROOT / "scripts" / "vla"))
		from eval_closed_loop import managed_policy_server

		out = a.out / f"act-{a.ckpt.parent.name}-{a.ckpt.name}{'-guard' if a.guard else ''}{ring_tag}"
		out.mkdir(parents=True, exist_ok=True)
		policy_kw = {"replan_interval": a.replan_interval, "temporal_ensemble": a.temporal_ensemble}
		with managed_policy_server(ckpt=a.ckpt) as url, ProcessPoolExecutor(min(a.workers, 4)) as pool:
			rows = list(pool.map(_student, [(s, a.split, out, url, policy_kw, a.guard, a.ring) for s in seeds]))
		arms.append((out.name, rows))
	for name, rows in arms:
		with (a.out / name / "results.jsonl").open("w", encoding="utf-8") as fh:
			for r in rows:
				fh.write(json.dumps(r, default=str) + "\n")
		summary[name] = summarize(rows)
	summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
	print(json.dumps(summary, indent=2))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
