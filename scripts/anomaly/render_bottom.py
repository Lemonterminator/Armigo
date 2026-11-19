"""Render top-camera crops of the held brick's bottom face at the poses the six-face agent uses (``present_bottom``).

The cross_var datasets (``render_dataset.py``) never put the defect on the bottom face (5), and their random show
poses rarely face the bottom squarely to the top camera. The agent does exactly that (``plan_bottom_view``), so
normal 2x4 bottoms, with their three tubes, raised false alarms (``results/agent-bench/REPORT.md``, bottom48).

For each layout this runs the agent backend (``TwinMujoco``, agent fixtures from ``sample_case``) up to the grasp and
lift, then poses the held brick kinematically (:class:`HeldBrickPoser`) at ``--plan-seeds`` bottom views of
``plan_bottom_view`` (the agent's seeds are 0, 1, 71, ...) and at ``--jitter`` random perturbations of them, and saves
the crop as the agent makes it (``detector_crop``, 256 px, clamped, no fixed crop). Defective layouts carry a
``cross_var`` defect on face 5. Brick sizes are drawn with ``--p-2x4`` (the agent sampler draws 2x4 with p = 0.2).

The output has the ``render_dataset.py`` layout (``png/``, ``masks/``, ``metadata.jsonl``, ``layouts.jsonl``,
``summary.json``) with train and test layouts and both classes in one directory; masks are written for abnormal crops.
Rows carry ``stage`` ``bottom`` (planned pose) or ``bottom_jitter``. Run with the main venv (MuJoCo)::

    python scripts/anomaly/render_bottom.py --n 240 --workers 8 --out results/anomaly/dataset-v7bot
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "anomaly"))
sys.path.insert(0, str(ROOT / "scripts"))

TRAIN_FRACTION = 0.7
BOTTOM_MIN_PX = 400  # twin_mujoco.BOTTOM_MIN_PX: the agent scores a bottom view only from this many face-5 pixels
JITTER_RAD = 0.15
CLEARANCE_M = 0.04


def sample_jobs(n: int, seed_start: int, seed: int, p_2x4: float, p_defective: float) -> list[tuple]:
	"""``n`` (index, fixture seed, defective, split) with the requested brick-size mix; fixture seeds are consumed in
	order from ``seed_start``, skipping a seed whose brick size is not the one drawn for the layout."""
	from digital_twin_armigo.agent.twin_mujoco import sample_case
	from digital_twin_armigo.constants import DUPLO_2X4

	rng = np.random.default_rng(seed)
	jobs, s = [], seed_start
	for k in range(n):
		want_2x4 = bool(rng.random() < p_2x4)
		defective = bool(rng.random() < p_defective)
		split = "train" if rng.random() < TRAIN_FRACTION else "test"
		while (tuple(sample_case(s, defective, 5).brick_size) == tuple(DUPLO_2X4)) != want_2x4:
			s += 1
		jobs.append((k, s, defective, split))
		s += 1
	return jobs


def render_layout(job: tuple) -> dict[str, Any]:
	import cv2
	from render_dataset import MIN_BRICK_PX, _label, capture

	from digital_twin_armigo.agent.twin_mujoco import TwinMujoco, blind_geometry, sample_case
	from digital_twin_armigo.agent.twin_record import EpisodeTrace
	from digital_twin_armigo.mujoco_cell.inspection_planning import path_min_z
	from digital_twin_armigo.mujoco_cell.inspection_views import HeldBrickPoser, capture_view
	from digital_twin_armigo.mujoco_cell.table_relocation import plan_bottom_view

	k, fixture, defective, split, out_dir, plan_seeds, n_jitter, image_size = job
	out = Path(out_dir)
	case = sample_case(fixture, defective, 5)
	rec: dict[str, Any] = {
		"layout_index": k,
		"fixture_seed": fixture,
		"split": split,
		"defective": defective,
		"defect_kind": case.defect_kind,
		"defect_face": 5,
		"brick_size": [float(v) for v in case.brick_size],
		"brick_rgb": [float(v) for v in case.brick_rgb],
	}
	rows: list[dict] = []
	t0 = time.perf_counter()
	tmp = Path(tempfile.mkdtemp(dir=out / "tmp"))
	backend = TwinMujoco(case, None, EpisodeTrace(tmp / "e", video=False))
	try:
		error = backend.prepare() or backend._grasp()
		if error or backend.held is None:
			rec["status"] = f"no_grasp:{error}"
			return {"layout": rec, "rows": rows}
		cell, ex = backend.cell, backend.expert
		rng = np.random.default_rng([fixture, 11])
		poses: list[tuple[str, np.ndarray]] = []
		with blind_geometry(cell):
			for s in plan_seeds:
				cand = plan_bottom_view(cell, ex.last, gripper=ex.p.hold_g, seed=s)
				if cand is not None and cand.faces_px[5] >= BOTTOM_MIN_PX:
					poses.append(("bottom", cand.q.copy()))
			planned = [q for _, q in poses]
			poser = HeldBrickPoser(cell)
			tries = 0
			while planned and sum(st == "bottom_jitter" for st, _ in poses) < n_jitter and tries < 20 * n_jitter:
				tries += 1
				q = planned[int(rng.integers(len(planned)))].copy()
				q[:5] += rng.normal(0.0, JITTER_RAD, 5)
				q[:5] = np.clip(q[:5], cell.jnt_range[:5, 0], cell.jnt_range[:5, 1])
				d = poser.at(q)
				if min(cell.robot_min_z(d), poser.brick_min_z()) < CLEARANCE_M:
					continue
				if capture_view(cell, "top", data=d).faces_px[5] < BOTTOM_MIN_PX:
					continue
				if path_min_z(cell, ex.last, q, carried=True) < CLEARANCE_M * 0.5:
					continue
				poses.append(("bottom_jitter", q))
		poser = HeldBrickPoser(cell)
		for stage, q in poses:
			d = poser.at(q)
			mask: list = []
			view, image, bbox = capture(cell, "top", None, d, image_size, clamp=True, mask_out=mask)
			if image is None or view.brick_px < MIN_BRICK_PX:
				continue
			label = _label(defective, view.defect_px)
			name = f"{k:04d}_{len(rows):02d}_top.png"
			cv2.imwrite(str(out / "png" / name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
			if label == "abnormal":
				cv2.imwrite(str(out / "masks" / name), mask[0])
			rows.append(
				{
					"path": f"png/{name}",
					"layout_index": k,
					"fixture_seed": fixture,
					"split": split,
					"camera": "top",
					"stage": stage,
					"task": "bottom",
					"flex": round(float(q[3]), 4),
					"roll_deg": round(float(np.degrees(q[4])), 2),
					"virtual": True,
					"defective": defective,
					"defect_kind": case.defect_kind,
					"defect_face": 5,
					"defect_px": int(view.defect_px),
					"faces_px": [int(v) for v in view.faces_px],
					"brick_px": int(view.brick_px),
					"brick_2x4": bool(case.brick_size[0] > 0.05),
					"bbox": list(bbox),
					"label": label,
				}
			)
		rec["status"] = "ok"
	finally:
		backend.record.close()
		backend.close()
		shutil.rmtree(tmp, ignore_errors=True)
	rec["n_crops"] = len(rows)
	rec["seconds"] = round(time.perf_counter() - t0, 2)
	return {"layout": rec, "rows": rows}


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	ap.add_argument("--n", type=int, default=240)
	ap.add_argument("--seed", type=int, default=7001, help="draws of size, class and split")
	ap.add_argument(
		"--seed-start",
		type=int,
		default=800_001,
		help="first agent fixture seed (bench suites use 6xxxxx, the search dev suite 7xxxxx)",
	)
	ap.add_argument("--p-2x4", type=float, default=0.5)
	ap.add_argument("--p-defective", type=float, default=0.5)
	ap.add_argument("--plan-seeds", type=int, nargs="+", default=[0, 1, 2, 71])
	ap.add_argument("--jitter", type=int, default=4, help="random perturbations of the planned poses per layout")
	ap.add_argument("--image-size", type=int, default=256)
	ap.add_argument("--workers", type=int, default=8)
	ap.add_argument("--out", type=Path, required=True)
	a = ap.parse_args()
	for sub in ("png", "masks", "tmp"):
		(a.out / sub).mkdir(parents=True, exist_ok=True)
	jobs = [
		(*j, str(a.out), tuple(a.plan_seeds), a.jitter, a.image_size)
		for j in sample_jobs(a.n, a.seed_start, a.seed, a.p_2x4, a.p_defective)
	]
	t0 = time.perf_counter()
	results = []
	with ProcessPoolExecutor(max_workers=a.workers) as pool:
		for res in pool.map(render_layout, jobs):
			results.append(res)
			lay = res["layout"]
			print(
				f"[{lay['layout_index']:4d}] {lay['split']:5s} def={int(lay['defective'])} "
				f"2x4={int(lay['brick_size'][0] > 0.05)} {lay['status']:12s} crops={len(res['rows']):2d} "
				f"({time.perf_counter() - t0:.0f}s)",
				flush=True,
			)
	shutil.rmtree(a.out / "tmp", ignore_errors=True)
	rows = [r for res in results for r in res["rows"]]
	layouts = [res["layout"] for res in results]
	(a.out / "metadata.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
	(a.out / "layouts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in layouts), encoding="utf-8")
	summary = {
		"n_layouts": len(layouts),
		"layout_status": dict(Counter(lay["status"] for lay in layouts)),
		"n_crops": len(rows),
		"by_split_size_label": dict(
			sorted(Counter(f"{r['split']}/{'2x4' if r['brick_2x4'] else '2x2'}/{r['label']}" for r in rows).items())
		),
		"by_stage": dict(Counter(r["stage"] for r in rows)),
		"config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
		"seconds": round(time.perf_counter() - t0, 1),
	}
	(a.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
	print(json.dumps(summary, indent=2))


if __name__ == "__main__":
	main()
