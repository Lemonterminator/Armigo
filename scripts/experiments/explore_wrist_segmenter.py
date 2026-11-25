"""Exploratory CPU closed-loop check of wrist re-localisation segmenters (brickseg_prereg section 4 item 6).

NOT a preregistered stage: no GPU detector, and the inspection scorer is replaced by an oracle-sign ``FixedScorer``
(+1 on every view of a defective case, -1 on a normal one), so labels are correct by construction and the outcome
isolates the motion physics. Groups (realistic sensing, cameras ``drift``, strategy B, suite ``p2_dev12``):
``open`` (lab, no wrist checkpoints), ``w2s`` with ``lab``, ``w2s`` with ``silhouette``. Episodes go through the
experiment runner's own job path (``runner.plan_group`` / ``runner.run_job``) under a scratch root; never the formal
cv-motion root.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from argparse import Namespace
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

GROUPS = {"open": ("open", "lab"), "w2s-lab": ("w2s", "lab"), "w2s-silhouette": ("w2s", "silhouette")}


def _args(root: Path, batch: str, motion: str, segmenter: str, camera: str) -> Namespace:
	return Namespace(
		root=str(root),
		sensing="realistic",
		camera=camera,
		strategy="B",
		detector="v7bot",
		policy=None,
		max_scored_views=None,
		boundary="belief_cell",
		oracle="",
		motion=motion,
		evidence="hold",
		calib_offset_mm=0.0,
		segmenter=segmenter,
		batch=batch,
		engine="mujoco",
		suite="p2_dev12",
		workers=1,
		limit=None,
		freeze=None,
		review_llm_url=None,
		review_model="qwen3.8-vllm",
	)


def _work(item):
	config, job = item
	os.environ["CUDA_VISIBLE_DEVICES"] = ""
	from digital_twin_armigo.experiments import runner
	from digital_twin_armigo.inspection_scorer import FixedScorer

	svc = runner._service(config)
	value = 1.0 if job["case"][1] else -1.0
	if svc.scorer is not None:
		svc.scorer.close()
	svc.scorer = FixedScorer(lambda _cam, _img, v=value: v, {"top": 0.0, "wrist_cam": 0.0})
	return runner.run_job(config, job)


def _summary(group_dir: Path) -> dict:
	from digital_twin_armigo.experiments.scoring import score_episode

	rows, fixes = [], collections.Counter()
	errs = []
	for ep in sorted(p for p in group_dir.iterdir() if p.is_dir()):
		if not (ep / "run_status.json").exists():
			continue
		rows.append(score_episode(ep))
		ev = ep / "events.jsonl"
		if ev.exists():
			for line in ev.read_text(encoding="utf-8").splitlines():
				e = json.loads(line)
				if e.get("kind") == "wrist_fix":
					fixes["looks"] += 1
					fixes["found"] += bool(e.get("found"))
					fixes["accepted"] += bool(e.get("accepted"))
					if e.get("reason"):
						fixes[f"reason:{e['reason']}"] += 1
					if e.get("accepted") and e.get("error_xy_mm") is not None:
						errs.append(e["error_xy_mm"])
	succ = sum(bool(r.get("physical_success")) for r in rows)
	return {
		"episodes": len(rows),
		"success": succ,
		"label_correct": sum(bool(r.get("label_correct")) for r in rows),
		"outcomes": dict(collections.Counter(r.get("physical_outcome") for r in rows)),
		"root_causes": dict(collections.Counter(r.get("root_cause") for r in rows if not r.get("physical_success"))),
		"infra_failed": sum(bool(r.get("infra_failed")) for r in rows),
		"wrist_fix": dict(fixes),
		"accepted_fix_error_xy_mm_median": None if not errs else round(sorted(errs)[len(errs) // 2], 3),
	}


def main(argv=None) -> int:
	p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
	p.add_argument("--root", type=Path, required=True, help="scratch experiment root (not the formal cv-motion root)")
	p.add_argument("--batch", default="explore-wseg")
	p.add_argument("--camera", default="drift", choices=["fixed", "recal", "drift"])
	p.add_argument("--groups", nargs="+", default=list(GROUPS), choices=list(GROUPS))
	p.add_argument("--workers", type=int, default=8)
	p.add_argument("--limit", type=int, help="episodes per group (smoke)")
	a = p.parse_args(argv)
	os.environ["CUDA_VISIBLE_DEVICES"] = ""
	if "Armigo_data_v3" in str(a.root.resolve()):
		raise SystemExit("exploratory runs must not write under the formal data root")
	from digital_twin_armigo.experiments import runner

	items, dirs = [], {}
	for name in a.groups:
		motion, seg = GROUPS[name]
		config, jobs, group_dir = runner.plan_group(_args(a.root, a.batch, motion, seg, a.camera))
		dirs[name] = group_dir
		items += [(config, j) for j in jobs[: a.limit]]
	print(f"{len(items)} episodes in {len(dirs)} groups", flush=True)
	with ProcessPoolExecutor(a.workers) as pool:
		for n, res in enumerate(pool.map(_work, items), 1):
			print(f"[{n}/{len(items)}] {res['case_id']} {res['state']} {res.get('verdict')}", flush=True)
	out = {name: _summary(d) for name, d in dirs.items()}
	(a.root / f"{a.batch}-{a.camera}-summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
	print(json.dumps(out, indent=1))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
