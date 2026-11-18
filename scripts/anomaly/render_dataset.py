"""Render a crop dataset for image anomaly detection of the held brick (normal vs surface defect).

For ``--n`` random layouts (the benchmark sampler, with the defect face drawn uniformly over the six faces, the defect
kind uniformly over :data:`~digital_twin_armigo.mujoco_cell.scene.DEFECT_KINDS` and the brick defective with
probability ``--p-defective``) it runs the closed-loop expert in inspection mode up to its FIRST show hold. There it
saves the crop around the brick of both inspection cameras ("top", "wrist_cam") at the real pose, and the crops of
``--k`` virtual show poses (random wrist flex x wrist roll, the brick carried rigidly by the gripper, see
:class:`~digital_twin_armigo.mujoco_cell.inspection_views.HeldBrickPoser`) that keep the arm and the brick at least
2 cm above the table; then it stops the episode.

Each crop is labelled ``normal`` (normal brick), ``abnormal`` (defective, at least 25 defect pixels visible) or
``hidden_defect`` (defective but the defect is not in view: excluded from training and from AUROC, kept for
analysis). The split is by LAYOUT (~70 % train / 30 % test); a train layout keeps only normal crops, since the
detector is trained on normal images only (defective train layouts are not rendered at all). Writes ``png/*.png``,
``metadata.jsonl`` (one row per crop), ``layouts.jsonl`` (one row per layout) and ``summary.json`` into ``--out``.

``--task cross_var`` instead renders the views a heuristic inspection takes: wrist-camera prescan views of the brick
lying on the table (:func:`~digital_twin_armigo.mujoco_cell.inspection_planning.prescan_candidates`), the real show
pose, virtual show poses, and top-camera close-ups of the held brick from the close-up pose library
(:func:`~digital_twin_armigo.mujoco_cell.inspection_planning.closeup_candidates`). The top camera is then used
uncropped, square crops are clamped to the image, layouts carry the ``cross_var`` defect kind on a non-bottom face,
and the default output directory is ``dataset-v2``.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

CAMERAS = ("top", "wrist_cam")
FLEX_RANGE = (-1.2, 1.6)  # rad
ROLL_RANGE_DEG = (-150.0, 150.0)
TABLE_CLEARANCE_M = 0.02
MIN_DEFECT_PX = 25
MIN_BRICK_PX = 150
TRAIN_FRACTION = 0.7
TASKS = ("mixed", "cross_var")
DEFAULT_OUTS = {
	"mixed": ROOT / "results" / "anomaly" / "dataset-v1",
	"cross_var": ROOT / "results" / "anomaly" / "dataset-v2",
}
DEFAULT_OUT = DEFAULT_OUTS["mixed"]
DEFAULT_CLOSEUP_LIBRARY = ROOT / "results" / "top-camera-reach" / "poses.jsonl"
PRESCAN_Q_SEED = np.array([0.0, -0.35, 0.25, 1.35, 0.0, 0.85])
_CLOSEUP_LIBRARIES: dict[str, np.ndarray] = {}


class _ShowReached(Exception):
	"""Raised from the inspection callback once the first show hold is captured: ends the episode early."""


def sample_dataset_layouts(n: int, seed: int, p_defective: float = 0.5, task: str = "mixed") -> list[tuple[Any, str]]:
	"""``n`` (layout, split) pairs: the benchmark sampler's layouts with a random defect face, kind, seed and class,
	and a train / test split drawn per layout. ``task="cross_var"`` fixes the kind to ``cross_var`` and draws the
	face over the five non-bottom faces; ``task="mixed"`` keeps the original sampling unchanged."""
	if task not in TASKS:
		raise ValueError(f"unknown task {task!r}; expected one of {TASKS}")
	import bench_physics_expert as bpe

	from digital_twin_armigo.mujoco_cell.scene import DEFECT_KINDS

	rng = np.random.default_rng(seed + 20_000)
	out = []
	for k, lay in enumerate(bpe.sample_layouts(n, seed, bpe.load_top_camera())):
		if task == "cross_var":
			lay.defect_face = int(rng.integers(5))  # never the bottom face (5)
			lay.defect_kind = "cross_var"
		else:
			lay.defect_face = int(rng.integers(6))
			lay.defect_kind = DEFECT_KINDS[int(rng.integers(len(DEFECT_KINDS)))]
		lay.defective = bool(rng.random() < p_defective)
		lay.defect_seed = int(seed * 100_003 + k)
		split = "train" if rng.random() < TRAIN_FRACTION else "test"
		out.append((lay, split))
	return out


def _label(defective: bool, defect_px: int) -> str:
	if not defective:
		return "normal"
	return "abnormal" if defect_px >= MIN_DEFECT_PX else "hidden_defect"


def _virtual_poses(cell, k: int, rng: np.random.Generator) -> list[tuple[float, float]]:
	lo3, hi3 = cell.jnt_range[3]
	lo4, hi4 = cell.jnt_range[4]
	flex = rng.uniform(max(FLEX_RANGE[0], lo3), min(FLEX_RANGE[1], hi3), k)
	roll = rng.uniform(max(ROLL_RANGE_DEG[0], math.degrees(lo4)), min(ROLL_RANGE_DEG[1], math.degrees(hi4)), k)
	return [(float(f), float(r)) for f, r in zip(flex, roll, strict=True)]


def _defect_mask(cell, camera: str, crop, data) -> np.ndarray:
	"""Boolean mask (full frame, or inside ``crop``) of the visible defect cross as the colour render shows it: arm
	occlusion included, and the hidden probe cross of a normal brick (geom group 5) excluded, as in the RGB image."""
	from digital_twin_armigo.mujoco_cell._mj import mujoco

	r = cell.renderer
	r.enable_segmentation_rendering()
	try:
		r.update_scene(data, camera, scene_option=mujoco.MjvOption())
		seg = r.render()
	finally:
		r.disable_segmentation_rendering()
	if crop is not None:
		x, y, w, h = crop
		seg = seg[y : y + h, x : x + w]
	geom = np.where(seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), seg[..., 0], -1)
	return np.isin(geom, cell.defect_geoms)


def capture(
	cell, camera: str, crop, data, image_size: int, pad: float = 0.25, clamp: bool = False, mask_out: list | None = None
):
	"""Pixel counts of ``camera``'s view (:func:`capture_view`) and a square colour crop around the brick's main blob
	(None when the brick is not in view); returns (view, image, bbox). With ``clamp`` the square crop is shifted to
	stay inside the image (see :func:`~digital_twin_armigo.mujoco_cell.inspection_views.square_crop`)."""
	from digital_twin_armigo.mujoco_cell.inspection_views import capture_view, detector_crop, square_crop

	d = cell.data if data is None else data
	view = capture_view(cell, camera, crop=crop, data=d)
	if view.bbox is None:
		return view, None, None
	image, bbox = detector_crop(cell, camera, crop, d, image_size, pad, clamp)
	if bbox is None:
		return view, None, None
	if mask_out is not None:
		m = _defect_mask(cell, camera, crop, d).astype(np.uint8)[..., None] * 255
		mask_out.append(square_crop(m, bbox, pad, image_size, clamp=clamp)[..., 0])
	return view, image, bbox


def _kinematic_copy(cell, q: np.ndarray):
	"""A copy of the cell state with the arm joints set to ``q``; kinematics and camera / light update only."""
	from digital_twin_armigo.mujoco_cell._mj import mujoco

	d = copy.copy(cell.data)
	d.qpos[cell.qadr] = q
	mujoco.mj_kinematics(cell.model, d)
	mujoco.mj_camlight(cell.model, d)
	return d


def _closeup_poses(path: str | Path) -> np.ndarray:
	"""The top-camera close-up pose library, parsed and filtered once per process."""
	key = str(path)
	if key not in _CLOSEUP_LIBRARIES:
		from digital_twin_armigo.mujoco_cell.inspection_planning import closeup_library

		rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
		_CLOSEUP_LIBRARIES[key] = closeup_library(rows)
	return _CLOSEUP_LIBRARIES[key]


def _scene_nodes(cell, out: Path) -> tuple[list[int], list[dict]]:
	"""The rendered (geom group 0-2) geoms of ``cell`` as scene nodes for the Isaac replay; meshes go to the shared
	content-addressed ``meshes/`` directory (the robot meshes are identical in every layout)."""
	import hashlib
	import os

	from digital_twin_armigo.mujoco_cell._mj import mujoco

	m = cell.model
	geoms = [g for g in range(m.ngeom) if int(m.geom_group[g]) in (0, 1, 2)]
	nodes = []
	(out / "meshes").mkdir(exist_ok=True)
	for g in geoms:
		kind = mujoco.mjtGeom(int(m.geom_type[g])).name.removeprefix("mjGEOM_").lower()
		mat = int(m.geom_matid[g])
		rgba = np.array(m.mat_rgba[mat] if mat >= 0 else m.geom_rgba[g], dtype=float)
		tex = int(m.mat_texid[mat][1]) if mat >= 0 else -1  # RGB-role texture: flat Isaac colour = its mean colour
		if tex >= 0:
			adr, n = int(m.tex_adr[tex]), int(m.tex_width[tex]) * int(m.tex_height[tex]) * int(m.tex_nchannel[tex])
			pix = m.tex_data[adr : adr + n].reshape(-1, int(m.tex_nchannel[tex]))[:, :3].astype(float) / 255.0
			rgba[:3] *= pix.mean(axis=0)
		node = {
			"id": f"geom_{g}",
			"name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom_{g}",
			"type": kind,
			"size": [float(v) for v in m.geom_size[g]],
			"color": [float(v) for v in rgba],
		}
		if kind == "mesh":
			i = int(m.geom_dataid[g])
			va, vn, fa, fn = (int(v[i]) for v in (m.mesh_vertadr, m.mesh_vertnum, m.mesh_faceadr, m.mesh_facenum))
			verts = np.ascontiguousarray(m.mesh_vert[va : va + vn], dtype=np.float32)
			faces = np.ascontiguousarray(m.mesh_face[fa : fa + fn], dtype=np.int32)
			key = hashlib.sha1(verts.tobytes() + faces.tobytes()).hexdigest()[:16]
			path = out / "meshes" / f"{key}.npz"
			if not path.exists():
				tmp = out / "meshes" / f"{key}.{os.getpid()}.tmp.npz"
				np.savez_compressed(tmp, vertices=verts, faces=faces)
				try:
					os.replace(tmp, path)
				except PermissionError:  # another worker wrote the same (identical) mesh first
					tmp.unlink(missing_ok=True)
					if not path.exists():
						raise
			node["mesh"] = key
		nodes.append(node)
	return geoms, nodes


def _camera_fovy(m, cid: int) -> float:
	"""Vertical field of view (degrees) a camera actually renders with: a camera with a sensor size uses its
	intrinsics (``Cell.set_camera`` updates those, not ``cam_fovy``)."""
	sensor, intr = m.cam_sensorsize[cid], m.cam_intrinsic[cid]
	if sensor[1] > 0 and intr[1] > 0:
		return float(np.degrees(2.0 * np.arctan(sensor[1] / (2.0 * intr[1]))))
	return float(m.cam_fovy[cid])


def _geom_poses(cell, d, geoms: list[int], camera: str) -> tuple[np.ndarray, np.ndarray]:
	"""World poses (x y z qx qy qz qw) of ``geoms`` and of ``camera`` in the state ``d``."""
	from digital_twin_armigo.mujoco_cell._mj import mujoco

	poses = np.empty((len(geoms), 7))
	for i, g in enumerate(geoms):
		quat = np.empty(4)
		mujoco.mju_mat2Quat(quat, d.geom_xmat[g])
		poses[i] = (*d.geom_xpos[g], *quat[[1, 2, 3, 0]])
	cid = cell.camera_id(camera)
	quat = np.empty(4)
	mujoco.mju_mat2Quat(quat, d.cam_xmat[cid])
	return poses, np.array([*d.cam_xpos[cid], *quat[[1, 2, 3, 0]]])


def render_layout(job: tuple[int, Any, str, int, int, str, str, int, int, str]) -> dict[str, Any]:
	"""Run one layout to its first show hold and write its crops; returns the layout record and the crop rows."""
	import bench_physics_expert as bpe
	import cv2

	from digital_twin_armigo.benchmarks import SENSITIVITY_MATRIX
	from digital_twin_armigo.constants import SO101_MJCF_DIR, TOP_CROP
	from digital_twin_armigo.mujoco_cell import Cell
	from digital_twin_armigo.mujoco_cell._mj import mujoco
	from digital_twin_armigo.mujoco_cell.inspection_views import HeldBrickPoser
	from digital_twin_armigo.mujoco_cell.physics_expert import ExpertParams, run_closed_loop_episode
	from digital_twin_armigo.physics import load_physics_spec

	k, lay, split, n_virtual, image_size, out_dir, task, k_prescan, k_closeup, library_path = job
	out = Path(out_dir)
	clamp = task == "cross_var"
	crops = {"top": None if clamp else TOP_CROP, "wrist_cam": None}
	rec: dict[str, Any] = {
		"layout_index": k,
		"split": split,
		"defective": bool(lay.defective),
		"defect_kind": lay.defect_kind,
		"defect_face": int(lay.defect_face),
		"defect_seed": int(lay.defect_seed),
		"brick_size": [float(v) for v in lay.brick_size],
		"brick_rgb": [float(v) for v in lay.brick_rgb],
	}
	rows: list[dict] = []
	only_def = (out / "masks").is_dir()  # --only-defective-train: defective train layouts only, with defect masks
	is_def_train = split == "train" and bool(lay.defective)
	if only_def and not is_def_train:
		rec["status"] = "skipped_not_defective_train"
		return {"layout": rec, "rows": rows}
	if not only_def and is_def_train:
		rec["status"] = "skipped_defective_train"
		return {"layout": rec, "rows": rows}
	t0 = time.perf_counter()
	spec = load_physics_spec()
	base = {c.name: c for c in SENSITIVITY_MATRIX}["baseline_2x2"]
	rng = np.random.default_rng([lay.defect_seed & 0xFFFFFFFF, 7])
	cell = Cell(lay, SO101_MJCF_DIR, physics=spec, dynamic_clutter=True, assisted_grasp=False, face_probes=True)
	dump = (out / "poses").is_dir()  # --dump-poses: also record what Isaac needs to replay each crop
	dump_geoms, dump_nodes = _scene_nodes(cell, out) if dump else ([], [])
	dump_poses: list[np.ndarray] = []
	dump_cams: list[np.ndarray] = []

	def save(cam: str, data, flex: float, roll_deg: float, virtual: bool, stage: str) -> None:
		mask: list = []
		view, image, bbox = capture(
			cell, cam, crops[cam], data, image_size, clamp=clamp, mask_out=mask if only_def else None
		)
		if image is None or view.brick_px < MIN_BRICK_PX:
			return
		label = _label(lay.defective, view.defect_px)
		if split == "train" and label != "normal" and not only_def:
			return
		name = f"{k:04d}_{len(rows):02d}_{cam}.png"
		cv2.imwrite(str(out / "png" / name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
		if mask:
			cv2.imwrite(str(out / "masks" / name), mask[0])
		if dump:
			gp, cp = _geom_poses(cell, cell.data if data is None else data, dump_geoms, cam)
			dump_poses.append(gp)
			dump_cams.append(cp)
		rows.append(
			{
				"path": f"png/{name}",
				"layout_index": k,
				"split": split,
				"camera": cam,
				"stage": stage,
				"task": task,
				"flex": round(float(flex), 4),
				"roll_deg": round(float(roll_deg), 2),
				"virtual": virtual,
				"defective": bool(lay.defective),
				"defect_kind": lay.defect_kind,
				"defect_face": int(lay.defect_face),
				"defect_px": int(view.defect_px),
				"faces_px": [int(v) for v in view.faces_px],
				"brick_px": int(view.brick_px),
				"bbox": list(bbox),
				"bbox_stretched": view.bbox is not None and view.bbox[2] * view.bbox[3] > 2 * bbox[2] * bbox[3],
				"label": label,
			}
		)

	if task == "cross_var":
		# Wrist views of the brick still lying on the table, before the grasp: the only time the wrist camera can
		# survey the side faces (kinematic copies, the simulation state is untouched).
		from digital_twin_armigo.mujoco_cell.grasp_geometry import Kinematics
		from digital_twin_armigo.mujoco_cell.ik import ik_reach
		from digital_twin_armigo.mujoco_cell.inspection_planning import prescan_candidates

		mujoco.mj_forward(cell.model, cell.data)
		kin = Kinematics(cell)
		cands = prescan_candidates(cell, partial(ik_reach, kin), PRESCAN_Q_SEED)
		n_pick = min(k_prescan, len(cands))
		picks = rng.choice(len(cands), size=n_pick, replace=False) if n_pick else []
		for i in picks:
			c = cands[int(i)]
			save("wrist_cam", _kinematic_copy(cell, c.q), float(c.q[3]), float(np.degrees(c.q[4])), False, "prescan")
		rec["n_prescan_candidates"] = len(cands)

	def inspect(flex: float, roll_deg: float, frame: int):
		for cam in CAMERAS:
			save(cam, None, flex, roll_deg, False, "show")
		poser = HeldBrickPoser(cell)
		q_now = cell.joints()
		n_safe = 0
		for fx, rd in _virtual_poses(cell, n_virtual, rng):
			q = q_now.copy()
			q[3], q[4] = fx, math.radians(rd)
			d = poser.at(q)
			if min(cell.robot_min_z(d), poser.brick_min_z()) < TABLE_CLEARANCE_M:
				continue
			n_safe += 1
			for cam in CAMERAS:
				save(cam, d, fx, rd, True, "virtual_show")
		rec["virtual_poses_safe"] = n_safe
		if task == "cross_var":
			from digital_twin_armigo.mujoco_cell.inspection_planning import closeup_candidates

			cands = closeup_candidates(cell, _closeup_poses(library_path), gripper=cell.joints()[5])
			n_pick = min(k_closeup, len(cands))
			picks = rng.choice(len(cands), size=n_pick, replace=False) if n_pick else []
			closeup_poser = HeldBrickPoser(cell)
			for i in picks:
				c = cands[int(i)]
				d = closeup_poser.at(c.q)
				save("top", d, float(c.q[3]), float(np.degrees(c.q[4])), True, "closeup")
			rec["n_closeup_candidates"] = len(cands)
		raise _ShowReached

	try:
		_, _, info = run_closed_loop_episode(
			cell,
			ExpertParams(),
			q_start=bpe._q_start(),
			brick_xy=lay.brick_xy,
			brick_yaw_deg=lay.brick_yaw_deg,
			non_ideal=bpe.bench._mj_non_ideal(base, spec),
			bowls={"normal": lay.green, "defective": lay.pink},
			inspect=inspect,
		)
		rec["status"] = "no_show_hold"
		rec["aborted"] = info.get("aborted")
		rec["infeasible_reason"] = info.get("infeasible_reason")
	except _ShowReached:
		rec["status"] = "ok"
	finally:
		if dump and dump_poses:
			m = cell.model
			np.savez_compressed(
				out / "poses" / f"{k:04d}.npz",
				poses=np.stack(dump_poses),
				cams=np.stack(dump_cams),
				paths=np.array([r["path"] for r in rows]),
				nodes=json.dumps(dump_nodes),
				fovy=json.dumps({c: _camera_fovy(m, cell.camera_id(c)) for c in ("top", "wrist_cam")}),
			)
		cell.close()
	rec["n_crops"] = len(rows)
	rec["seconds"] = round(time.perf_counter() - t0, 2)
	return {"layout": rec, "rows": rows}


def summarize(rows: list[dict], layouts: list[dict]) -> dict[str, Any]:
	def count(key) -> dict[str, int]:
		return dict(sorted(Counter(key(r) for r in rows).items()))

	return {
		"n_layouts": len(layouts),
		"layout_status": dict(Counter(lay["status"] for lay in layouts)),
		"layouts_with_crops_by_split": count_layouts(rows),
		"n_crops": len(rows),
		"by_stage": count(lambda r: r["stage"]),
		"by_split_label": count(lambda r: f"{r['split']}/{r['label']}"),
		"by_split_camera_label": count(lambda r: f"{r['split']}/{r['camera']}/{r['label']}"),
		"test_by_kind_label": count(lambda r: f"{r['defect_kind']}/{r['label']}" if r["split"] == "test" else "train"),
		"test_by_face_label": count(lambda r: f"{r['defect_face']}/{r['label']}" if r["split"] == "test" else "train"),
		"virtual_fraction": round(sum(r["virtual"] for r in rows) / max(len(rows), 1), 3),
	}


def count_layouts(rows: list[dict]) -> dict[str, int]:
	seen = {(r["split"], r["layout_index"]) for r in rows}
	return dict(Counter(s for s, _ in seen))


def build(
	n: int,
	seed: int,
	out: Path,
	k: int,
	image_size: int,
	workers: int,
	p_defective: float,
	task: str = "mixed",
	k_prescan: int = 8,
	k_closeup: int = 8,
	library_path: str | Path = DEFAULT_CLOSEUP_LIBRARY,
	dump_poses: bool = False,
	only_defective_train: bool = False,
) -> dict[str, Any]:
	if task not in TASKS:
		raise ValueError(f"unknown task {task!r}; expected one of {TASKS}")
	(out / "png").mkdir(parents=True, exist_ok=True)
	if dump_poses:
		(out / "poses").mkdir(exist_ok=True)
	if only_defective_train:
		(out / "masks").mkdir(exist_ok=True)
	jobs = [
		(i, lay, split, k, image_size, str(out), task, k_prescan, k_closeup, str(library_path))
		for i, (lay, split) in enumerate(sample_dataset_layouts(n, seed, p_defective, task))
	]
	t0 = time.perf_counter()
	results: list[dict] = []
	if workers <= 1:
		it = map(render_layout, jobs)
		for res in it:
			results.append(res)
			_progress(res, t0)
	else:
		with ProcessPoolExecutor(max_workers=workers) as ex:
			for res in ex.map(render_layout, jobs):
				results.append(res)
				_progress(res, t0)
	rows = [r for res in results for r in res["rows"]]
	layouts = [res["layout"] for res in results]
	(out / "metadata.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
	(out / "layouts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in layouts), encoding="utf-8")
	summary = summarize(rows, layouts)
	summary["config"] = {
		"task": task,
		"n": n,
		"seed": seed,
		"k_virtual": k,
		"k_prescan": k_prescan,
		"k_closeup": k_closeup,
		"image_size": image_size,
		"p_defective": p_defective,
		"train_fraction": TRAIN_FRACTION,
		"min_defect_px": MIN_DEFECT_PX,
		"min_brick_px": MIN_BRICK_PX,
		"table_clearance_m": TABLE_CLEARANCE_M,
		"flex_range": list(FLEX_RANGE),
		"roll_range_deg": list(ROLL_RANGE_DEG),
	}
	summary["seconds"] = round(time.perf_counter() - t0, 1)
	(out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
	return summary


def _progress(res: dict, t0: float) -> None:
	lay = res["layout"]
	print(
		f"[{lay['layout_index']:4d}] {lay['split']:5s} def={int(lay['defective'])} {lay['defect_kind']:8s} "
		f"face={lay['defect_face']} {lay['status']:24s} crops={len(res['rows']):3d} ({time.perf_counter() - t0:.0f}s)",
		flush=True,
	)


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	ap.add_argument("--n", type=int, default=300)
	ap.add_argument("--seed", type=int, default=2027)
	ap.add_argument("--k", type=int, default=12, help="virtual show poses sampled per layout")
	ap.add_argument("--image-size", type=int, default=128)
	ap.add_argument("--workers", type=int, default=1)
	ap.add_argument("--p-defective", type=float, default=0.5)
	ap.add_argument(
		"--task", choices=TASKS, default="mixed", help="'mixed' (v1) or the heuristic 'cross_var' views (v2)"
	)
	ap.add_argument("--k-prescan", type=int, default=8, help="cross_var: wrist prescan crops sampled per layout")
	ap.add_argument("--k-closeup", type=int, default=8, help="cross_var: top-camera close-up crops sampled per layout")
	ap.add_argument("--closeup-library", type=Path, default=DEFAULT_CLOSEUP_LIBRARY)
	ap.add_argument(
		"--only-defective-train",
		action="store_true",
		help="render only the defective TRAIN layouts (labelled abnormal / hidden_defect) with defect masks/",
	)
	ap.add_argument("--dump-poses", action="store_true", help="also write poses/ and meshes/ for the Isaac replay")
	ap.add_argument("--out", type=Path, default=None, help="default: dataset-v1 for mixed, dataset-v2 for cross_var")
	args = ap.parse_args()
	out = args.out if args.out is not None else DEFAULT_OUTS[args.task]
	summary = build(
		args.n,
		args.seed,
		out,
		args.k,
		args.image_size,
		args.workers,
		args.p_defective,
		task=args.task,
		k_prescan=args.k_prescan,
		k_closeup=args.k_closeup,
		library_path=args.closeup_library,
		dump_poses=args.dump_poses,
		only_defective_train=args.only_defective_train,
	)
	print(json.dumps(summary, indent=2))


if __name__ == "__main__":
	main()
