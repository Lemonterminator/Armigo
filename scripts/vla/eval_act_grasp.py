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

	seed, split, out = args
	try:
		return grasp_trial(make_case(seed, split), out / f"case-{seed}")
	except Exception as e:  # noqa: BLE001 - a crashed trial is a reported failure
		return {"case": make_case(seed, split), "lifted": False, "error": f"crash:{type(e).__name__}:{e}"}


def _student(args: tuple) -> dict:
	from digital_twin_armigo.agent.act_grasp import CloseBeforeDoneGuard, grasp_trial
	from digital_twin_armigo.vla.client_policy import ChunkClientPolicy, RemoteChunkPredictor

	seed, split, out, url, policy_kw, guard = args
	with RemoteChunkPredictor(url, timeout_s=30.0) as predictor:
		policy = ChunkClientPolicy(predictor, **policy_kw)
		if guard:
			policy = CloseBeforeDoneGuard(policy)
		try:
			r = grasp_trial(make_case(seed, split), out / f"case-{seed}", policy=policy)
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
	a = ap.parse_args(argv)
	seeds = list(range(a.seed0, a.seed0 + a.n))
	summary_path = a.out / "summary.json"
	summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
	arms: list[tuple[str, list[dict]]] = []
	if not a.skip_scripted:
		out = a.out / "scripted"
		out.mkdir(parents=True, exist_ok=True)
		with ProcessPoolExecutor(a.workers) as pool:
			rows = list(pool.map(_scripted, [(s, a.split, out) for s in seeds]))
		arms.append(("scripted", rows))
	if a.ckpt is not None:
		sys.path.insert(0, str(ROOT / "scripts" / "vla"))
		from eval_closed_loop import managed_policy_server

		out = a.out / f"act-{a.ckpt.parent.name}-{a.ckpt.name}{'-guard' if a.guard else ''}"
		out.mkdir(parents=True, exist_ok=True)
		policy_kw = {"replan_interval": a.replan_interval, "temporal_ensemble": a.temporal_ensemble}
		with managed_policy_server(ckpt=a.ckpt) as url, ProcessPoolExecutor(min(a.workers, 4)) as pool:
			rows = list(pool.map(_student, [(s, a.split, out, url, policy_kw, a.guard) for s in seeds]))
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
