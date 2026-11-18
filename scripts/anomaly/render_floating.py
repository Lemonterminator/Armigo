"""Floating-brick twin of a ``render_dataset.py --dump-poses`` dataset (MuJoCo): the brick and its defect, nothing else.

Run with the main ``.venv`` python::

    .venv/Scripts/python scripts/anomaly/render_floating.py --data results/anomaly/dataset-v4 \
        --out results/anomaly/dataset-v4-float --workers 2

For every crop of ``--data`` the recorded world poses of the brick geoms and of the camera are replayed in the SAME
MuJoCo scene (same lights, same brick material and defect), but every other geom (arm, gripper, bowls, clutter, table,
markers) is hidden, so the brick floats in the sky background. The crop box is the one stored in ``metadata.jsonl``:
``png/NAME.png`` of the output is the paired twin of ``png/NAME.png`` of the input (same layout, pose, camera, split,
bbox), with the defect mask of the floating view written to ``masks/NAME.png`` for defective bricks.

The label is recomputed on the floating view: with no arm in front of the brick, a defect is ``abnormal`` as soon as
``MIN_DEFECT_PX`` of its pixels are in the frame, so some ``hidden_defect`` crops of the original become ``abnormal``
here (the original label is kept as ``label_scene``). The layouts come from ``summary.json``'s ``config`` (same seed, so
the same bricks and defects as ``--data``).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

MIN_DEFECT_PX = 25
PAD = 0.25
HIDDEN_GROUP = (
	4  # not drawn by the default MjvOption (groups 0-2 only), neither in the colour nor the segmentation render
)


def _render_layout(job: tuple[Path, Any, Path, int, list[dict]]) -> dict[str, Any]:
	import cv2

	from digital_twin_armigo.constants import SO101_MJCF_DIR
	from digital_twin_armigo.mujoco_cell import Cell
	from digital_twin_armigo.mujoco_cell._mj import mujoco
	from digital_twin_armigo.mujoco_cell.inspection_views import square_crop
	from digital_twin_armigo.physics import load_physics_spec

	pose_file, lay, out, image_size, rows = job
	t0 = time.perf_counter()
	with np.load(pose_file, allow_pickle=False) as z:
		poses, cams, paths = z["poses"], z["cams"], z["paths"].tolist()
		nodes, fovy = json.loads(str(z["nodes"])), json.loads(str(z["fovy"]))
	by_path = {r["path"]: r for r in rows}
	cell = Cell(
		lay, SO101_MJCF_DIR, physics=load_physics_spec(), dynamic_clutter=True, assisted_grasp=False, face_probes=True
	)
	m = cell.model
	geoms = [g for g in range(m.ngeom) if int(m.geom_group[g]) in (0, 1, 2)]
	names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom_{g}" for g in geoms]
	if names != [n["name"] for n in nodes]:
		raise RuntimeError(
			f"{pose_file.name}: the rebuilt scene does not match the recorded nodes (wrong layout / seed?)"
		)
	keep = [i for i, n in enumerate(names) if n.startswith(("brick", "defect"))]
	kept = {geoms[i] for i in keep}
	for g in geoms:
		if g not in kept:
			m.geom_group[g] = HIDDEN_GROUP
	defect_ids = np.array([g for g in kept if names[geoms.index(g)].startswith("defect")], dtype=np.int64)
	brick_ids = np.array([g for g in kept if g not in set(defect_ids.tolist())], dtype=np.int64)
	# A fresh cell has not run kinematics yet: the directional light's world direction is still 0 (no diffuse light).
	# The light is fixed in the world, so one pass is enough; the poses below are then overwritten per crop.
	mujoco.mj_kinematics(m, cell.data)
	mujoco.mj_camlight(m, cell.data)
	r = cell.renderer
	out_rows: list[dict] = []
	for i, path in enumerate(paths):
		row = by_path[path]
		cam = row["camera"]
		cid = cell.camera_id(cam)
		m.cam_sensorsize[cid] = 0.0  # render with fovy, as the dumped field of view is the effective one
		m.cam_fovy[cid] = fovy[cam]
		d = cell.data
		for k in keep:
			p = poses[i][k]
			d.geom_xpos[geoms[k]] = p[:3]
			mujoco.mju_quat2Mat(d.geom_xmat[geoms[k]], np.array([p[6], p[3], p[4], p[5]]))
		cp = cams[i]
		d.cam_xpos[cid] = cp[:3]
		mujoco.mju_quat2Mat(d.cam_xmat[cid], np.array([cp[6], cp[3], cp[4], cp[5]]))
		r.disable_depth_rendering()
		r.update_scene(d, cam)
		rgb = r.render().copy()
		r.enable_segmentation_rendering()
		try:
			r.update_scene(d, cam, scene_option=mujoco.MjvOption())
			seg = r.render()
		finally:
			r.disable_segmentation_rendering()
		geom = np.where(seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), seg[..., 0], -1)
		defect = np.isin(geom, defect_ids)
		brick_px = int(np.isin(geom, brick_ids).sum() + defect.sum())
		defect_px = int(defect.sum())
		bbox = tuple(row["bbox"])
		image = square_crop(rgb, bbox, PAD, image_size, clamp=row["task"] == "cross_var")
		name = Path(path).name
		cv2.imwrite(str(out / "png" / name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
		if row["defective"]:
			mask = square_crop(
				defect.astype(np.uint8)[..., None] * 255, bbox, PAD, image_size, clamp=row["task"] == "cross_var"
			)
			cv2.imwrite(str(out / "masks" / name), mask[..., 0])
		label = "normal" if not row["defective"] else ("abnormal" if defect_px >= MIN_DEFECT_PX else "hidden_defect")
		out_rows.append(
			{
				**row,
				"label": label,
				"label_scene": row["label"],
				"defect_px": defect_px,
				"brick_px": brick_px,
				"floating": True,
			}
		)
	return {"layout": int(pose_file.stem), "rows": out_rows, "seconds": time.perf_counter() - t0}


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	ap.add_argument(
		"--data", type=Path, required=True, help="dataset dir with poses/, metadata.jsonl, layouts.jsonl, summary.json"
	)
	ap.add_argument("--out", type=Path, required=True)
	ap.add_argument("--workers", type=int, default=2)
	ap.add_argument("--image-size", type=int, default=256)
	ap.add_argument("--limit-layouts", type=int, default=0)
	ap.add_argument("--only-layouts", type=int, nargs="*", default=None)
	a = ap.parse_args()
	import render_dataset as rd

	summary = json.loads((a.data / "summary.json").read_text(encoding="utf-8"))
	cfg = summary["config"]
	layouts = rd.sample_dataset_layouts(cfg["n"], cfg["seed"], cfg["p_defective"], cfg["task"])
	rows = [json.loads(line) for line in (a.data / "metadata.jsonl").read_text(encoding="utf-8").splitlines() if line]
	recs = {
		int(r["layout_index"]): r
		for r in (
			json.loads(line) for line in (a.data / "layouts.jsonl").read_text(encoding="utf-8").splitlines() if line
		)
	}
	for k, (lay, split) in enumerate(layouts):  # the rebuilt layouts must be the recorded ones
		rec = recs[k]
		if (rec["split"], bool(rec["defective"]), int(rec["defect_seed"])) != (
			split,
			bool(lay.defective),
			int(lay.defect_seed),
		):
			raise RuntimeError(f"layout {k} differs from layouts.jsonl: wrong seed / n in summary.json?")
	files = sorted((a.data / "poses").glob("*.npz"))
	if a.only_layouts is not None:
		files = [f for f in files if int(f.stem) in set(a.only_layouts)]
	if a.limit_layouts:
		files = files[: a.limit_layouts]
	(a.out / "png").mkdir(parents=True, exist_ok=True)
	(a.out / "masks").mkdir(parents=True, exist_ok=True)
	by_layout: dict[int, list[dict]] = {}
	for r in rows:
		by_layout.setdefault(int(r["layout_index"]), []).append(r)
	jobs = [(f, layouts[int(f.stem)][0], a.out, a.image_size, by_layout[int(f.stem)]) for f in files]
	t0 = time.perf_counter()
	done: list[dict] = []
	with ProcessPoolExecutor(a.workers) as ex:
		for res in ex.map(_render_layout, jobs):
			done.append(res)
			print(
				f"[{res['layout']:4d}] {len(res['rows']):3d} crops ({res['seconds']:.0f}s, total {time.perf_counter() - t0:.0f}s)",
				flush=True,
			)
	out_rows = [r for res in done for r in res["rows"]]
	with (a.out / "metadata.jsonl").open("w", encoding="utf-8") as f:
		for r in out_rows:
			f.write(json.dumps(r) + "\n")
	for name in ("layouts.jsonl",):
		(a.out / name).write_text((a.data / name).read_text(encoding="utf-8"), encoding="utf-8")
	changed = sum(r["label"] != r["label_scene"] for r in out_rows)
	summary_out = {
		"source": str(a.data),
		"n_crops": len(out_rows),
		"labels_changed_vs_scene": changed,
		"by_label": {k: sum(r["label"] == k for r in out_rows) for k in ("normal", "abnormal", "hidden_defect")},
		"config": cfg,
		"seconds": time.perf_counter() - t0,
	}
	(a.out / "summary.json").write_text(json.dumps(summary_out, indent=2), encoding="utf-8")
	print(json.dumps(summary_out))


if __name__ == "__main__":
	main()
