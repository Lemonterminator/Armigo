"""Episode generation: simulate once (physics, full 3D state per frame), then render many views of each episode."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..camera import CameraPose, camera_forward, project
from ..constants import BOWL_H, BOWL_R, CAMERA_SHAPES, FPS, PHASES, STATE_NAMES, TASKS, TOP_CROP
from ..joints import normalized_to_rad, rad_to_normalized
from ..presentation import FACES, camera_view_ok, show_points
from ._mj import mujoco
from .cell import Cell
from .expert import (
	SHOW_DISTANCES_M,
	SHOW_OFF_AXIS_DEG,
	SHOW_REACH_M,
	SHOW_Z_MIN_M,
	Plan,
	expert_handoff,
	expert_pick,
	expert_place,
	expert_present,
	presentation_search,
)
from .layout import Layout, sample_layout
from .views import perturb_main_camera, sample_front_camera, sample_side_camera, sample_top_camera, top_view_ok

logger = logging.getLogger(__name__)


def _simulate_one(job: tuple) -> dict:
	menagerie, nominal, prior, out, e, seed, rest_u = job
	rng = np.random.default_rng([seed, e])
	rest = normalized_to_rad(rest_u)
	defective = bool(rng.random() < 0.35)
	for _ in range(500):
		layout = sample_layout(rng, nominal, defective, prior)
		if top_view_ok(nominal, asdict(layout), np.zeros((0, 3)), margin_px=30.0)[0]:
			break
	else:
		raise RuntimeError("no layout fits the nominal top camera; check the camera file")
	# The main (inspection) camera is fixed before simulating: the presentation poses are planned against it, and
	# rendering only perturbs it slightly. A camera without a reachable show point is rejected.
	layout.defect_face = int(rng.integers(len(FACES)))
	for _ in range(50):
		main = sample_top_camera(rng, nominal)
		cam = np.array([main.x, main.y, main.height])
		pts = show_points(cam, camera_forward(main), SHOW_DISTANCES_M, SHOW_OFF_AXIS_DEG, SHOW_REACH_M, SHOW_Z_MIN_M)
		if len(pts) and top_view_ok(main, asdict(layout), np.zeros((0, 3)), margin_px=20.0)[0]:
			break
	else:
		raise RuntimeError("no main camera near the nominal one has a reachable show point")
	layout.top = main
	start = rest + np.concatenate([rng.normal(0, 0.05, 5), [0.0]])
	cell = Cell(layout, menagerie)
	cell.set_joints(start)
	qpos, state, action, tcp, phase = [], [], [], [], []

	def run(plan: Plan, first: int) -> None:
		for target, ph in zip(plan.frames[first:], plan.phases[first:], strict=True):
			noisy = target + np.concatenate([rng.normal(0, 0.004, 5), [0.0]])
			qpos.append(cell.data.qpos.copy())
			state.append(rad_to_normalized(cell.joints()))
			action.append(rad_to_normalized(noisy))
			tcp.append(cell.data.site_xpos[cell.site].copy())
			phase.append(ph)
			cell.step(noisy)
			cell.update_grasp(float(target[5]))

	plan, grip = expert_pick(cell, start, rng)
	run(plan, 0)
	grasped = cell.held is not None
	shown: list[dict] = []
	search: dict = {"show_point": None, "poses": [], "points_tried": 0}
	if grasped:
		n = len(plan.frames)
		search = presentation_search(cell, cell.joints(), main)
		shown = expert_present(plan, search, rng, grip.closed_g)
		expert_handoff(cell, plan, grip, rng)
		run(plan, n)
	n = len(plan.frames)
	expert_place(cell, plan, grip, start)
	run(plan, n)
	bowl = np.array(layout.pink if defective else layout.green)
	brick = cell.brick_pos()
	success = bool(np.hypot(*(brick[:2] - bowl)) < BOWL_R and brick[2] < 0.05)
	name = f"sim_{e:04d}"
	np.savez_compressed(
		out / f"{name}.npz",
		qpos=np.array(qpos),
		state=np.array(state, np.float32),
		action=np.array(action, np.float32),
		tcp=np.array(tcp),
		phase=np.array(phase, np.uint8),
	)
	entry = {
		"name": name,
		"length": len(plan.frames),
		"success": success,
		"grasped": grasped,
		"presented": bool(shown),
		"final_brick": brick.tolist(),
		"task": TASKS["pink" if defective else "green"],
		"defective": defective,
		"defect_face": layout.defect_face,
		"phases": list(PHASES),
		"top_camera": asdict(main),
		"presentation": {
			"show_point": search["show_point"],
			"points_tried": search["points_tried"],
			"poses_scored": len(search["poses"]),
			"shown": shown,
		},
		"layout": {k: v for k, v in asdict(layout).items() if k not in ("top", "front")},
	}
	(out / f"{name}.json").write_text(json.dumps(entry, indent=2) + "\n", encoding="utf-8")
	cell.renderer.close()
	logger.info(
		"simulated %s: %d frames, %s bowl, face %d, grasped=%s, shown=%d poses (best %.0f px), success=%s",
		name,
		len(plan.frames),
		"pink" if defective else "green",
		layout.defect_face,
		grasped,
		len(shown),
		shown[0]["defect"] if shown else 0,
		success,
	)
	return entry


def _log_setup() -> None:
	logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _pool(workers: int):
	import multiprocessing
	from concurrent.futures import ProcessPoolExecutor

	return ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"), initializer=_log_setup)


def simulate(
	menagerie: Path,
	camera_file: Path,
	out: Path,
	episodes: int,
	seed: int,
	rest_u: np.ndarray,
	prior_file: Path | None = None,
	workers: int = 1,
) -> list[dict]:
	"""Run scripted episodes with physics grasping; store full 3D state per frame (qpos), actions and the layout.

	Episode ``e`` draws everything from ``default_rng([seed, e])``, so results do not depend on ``workers``.
	"""
	nominal = CameraPose(**json.loads(camera_file.read_text())["camera"])
	prior = [p for p in json.loads(prior_file.read_text()) if "green" in p and "pink" in p] if prior_file else None
	out.mkdir(parents=True, exist_ok=False)
	jobs = [(menagerie, nominal, prior, out, e, seed, rest_u) for e in range(episodes)]
	if workers <= 1:
		return [_simulate_one(j) for j in jobs]
	with _pool(workers) as pool:
		return list(pool.map(_simulate_one, jobs))


def _show_frames_ok(cell: Cell, qpos: np.ndarray, frames: np.ndarray, defective: bool) -> tuple[bool, str]:
	"""In the sampled show-hold frames the brick is inside the crop and at least half unoccluded, and a defective
	brick shows its defect in at least one of them."""
	if len(frames) == 0:
		return True, ""
	seen = 0
	for t in frames:
		cell.data.qpos[:] = qpos[t]
		mujoco.mj_forward(cell.model, cell.data)
		vis = cell.visibility("top", TOP_CROP)
		if vis["brick_full"] < 50 or vis["brick"] < 0.5 * vis["brick_full"]:
			return False, f"brick occluded or outside the crop at frame {t}"
		seen += vis["defect"]
	if defective and seen == 0:
		return False, "defect never visible in the show frames"
	return True, ""


def _render_one(job: tuple) -> dict | None:
	menagerie, nominal, meta_path, out, k, seed, stride = job
	meta = json.loads(meta_path.read_text())
	name = f"{meta['name']}_v{k:02d}.npz"
	if (out / f"{name}.json").exists():  # finished by an earlier, interrupted run
		return json.loads((out / f"{name}.json").read_text())
	rng = np.random.default_rng([seed, int(meta["name"].split("_")[1]), k])
	traj = np.load(meta_path.with_suffix(".npz"))
	qpos = traj["qpos"]
	phase = traj["phase"] if "phase" in traj.files else np.zeros(len(qpos), np.uint8)
	lay = dict(meta["layout"])
	# the gripper path close to the table (grasp, release) must stay in view; higher up it may leave the frame at the
	# top edge as in the real recordings, and the show frames are checked separately below
	showing = np.isin(phase, [PHASES.index("show_move"), PHASES.index("show_hold")])
	low = traj["tcp"][(traj["tcp"][:, 2] < 0.06) & ~showing]
	main = CameraPose(**meta["top_camera"]) if "top_camera" in meta else None
	hold = np.flatnonzero(phase == PHASES.index("show_hold"))[::5]
	layout = Layout(**{**lay, "top": main or nominal, "front": sample_front_camera(rng)})
	layout.table_rgb = tuple(np.clip(np.array(lay["table_rgb"]) + rng.normal(0, 0.03, 3), 0.3, 0.85))
	layout.light_level = float(rng.uniform(0.6, 1.0))
	layout.light_dir = (float(rng.normal(0, 0.4)), float(rng.normal(0, 0.4)), -1.0)
	cell = Cell(layout, menagerie)
	# arm swept volume: collision geometry every 5th frame, as spheres of the geoms' bounding radii
	swept, radii = [], []
	for t in range(0, len(qpos), 5):
		cell.data.qpos[:] = qpos[t]
		mujoco.mj_kinematics(cell.model, cell.data)
		swept.append(cell.data.geom_xpos[cell.robot_geoms].copy())
		radii.append(cell.model.geom_rbound[cell.robot_geoms])
	swept_pts, swept_r = np.vstack(swept), np.concatenate(radii)
	rejected, why = 0, ""
	top = None
	for attempt in range(200):
		cand = perturb_main_camera(rng, main) if main else sample_top_camera(rng, nominal)
		if main and attempt >= 20:
			cand = main  # the simulated camera passed these checks when the show poses were planned
		ok, why = top_view_ok(cand, lay, low[:: max(1, len(low) // 40)])
		if ok:
			cell.set_camera("top", cand)
			ok, why = _show_frames_ok(cell, qpos, hold, meta["defective"])
		if ok:
			top = cand
			break
		rejected += 1
		if main and attempt >= 20:
			break
	if top is None:
		logger.warning("%s view %d: no plausible main camera (%s)", meta["name"], k, why)
		cell.renderer.close()
		return None
	targets = np.array([[*lay["brick_xy"], 0.02], [*lay["green"], BOWL_H], [*lay["pink"], BOWL_H]])
	for _ in range(500):
		side = sample_side_camera(rng)
		ok, why = camera_view_ok(
			np.array([side.x, side.y, side.height]), camera_forward(side), swept_pts, 0.05, radii=swept_r
		)
		uv, front = project(side, targets, (640, 480))
		if ok and front.all() and np.all((uv > 20) & (uv < np.array([620, 460]))):
			break
		rejected += 1
	else:
		logger.warning("%s view %d: no side camera clears the arm (%s)", meta["name"], k, why)
		cell.renderer.close()
		return None
	cell.set_camera("front", side)
	frames = list(range(0, meta["length"], stride))
	# preallocated: an episode is about 2 GB of frames, so a list plus a stacked copy would double the peak memory
	cams = {c: np.empty((len(frames), *shape), np.uint8) for c, shape in CAMERA_SHAPES.items()}
	defect_px = np.zeros(len(frames), np.int32)
	brick_px = np.zeros(len(frames), np.int32)
	for i, t in enumerate(frames):
		cell.data.qpos[:] = qpos[t]
		mujoco.mj_forward(cell.model, cell.data)
		cams["top_rgb_v2"][i], cams["top_depth_v2"][i] = cell.top_images()
		cams["front"][i] = cell.rgb("front")
		cams["wrist"][i] = cell.rgb("wrist_cam")
		vis = cell.visibility("top", TOP_CROP, full=False)
		defect_px[i] = vis["defect"] if meta["defective"] else 0
		brick_px[i] = vis["brick"]
	cell.renderer.close()
	np.savez_compressed(
		out / name,
		state=traj["state"][frames],
		action=traj["action"][frames],
		phase=phase[frames],
		defect_px=defect_px,
		brick_px=brick_px,
		**cams,
	)
	logger.info("rendered %s (%d frames)", name, len(frames))
	entry = {
		"file": name,
		"length": len(frames),
		"success": True,
		"task": meta["task"],
		"defective": meta["defective"],
		"defect_face": meta.get("defect_face"),
		"presented": meta.get("presented", False),
		"sim_episode": meta["name"],
		"top_camera": asdict(top),
		"front_camera": asdict(side),
		"rejected_cameras": rejected,
	}
	(out / f"{name}.json").write_text(json.dumps(entry, indent=2) + "\n", encoding="utf-8")
	return entry


def render_views(
	menagerie: Path,
	camera_file: Path,
	sim_dir: Path,
	out: Path,
	views: int,
	seed: int,
	stride: int = 1,
	workers: int = 1,
) -> dict:
	"""Replay every successful simulated episode under ``views`` random plausible cameras and appearances.

	Writes the ``offline_pipeline`` format (``index.json`` + one ``.npz`` per rendered episode), which
	``lerobot_export.py --video --robot-type SO101_Follower`` turns into a dataset shaped like sort-depth-v2.
	Each finished episode also gets a ``.json`` entry, so rerunning into the same ``out`` resumes an interrupted run.
	An episode holds about 2 GB of frames in memory while it renders: keep ``workers`` times 2 GB below the RAM.
	"""
	nominal = CameraPose(**json.loads(camera_file.read_text())["camera"])
	out.mkdir(parents=True, exist_ok=True)
	metas = [p for p in sorted(sim_dir.glob("sim_*.json")) if json.loads(p.read_text())["success"]]
	jobs = [(menagerie, nominal, p, out, k, seed, stride) for p in metas for k in range(views)]
	if workers <= 1:
		results = [_render_one(j) for j in jobs]
	else:
		with _pool(workers) as pool:
			results = list(pool.map(_render_one, jobs))
	episodes = [r for r in results if r is not None]
	index = {
		"fps": FPS // stride,
		"state_names": STATE_NAMES,
		"cameras": CAMERA_SHAPES,
		"source": "mujoco-sim",
		"privileged": {
			"phase": "per-frame expert phase, index into phases",
			"phases": list(PHASES),
			"defect_px": "defect pixels visible in top_rgb_v2 (segmentation render; 0 for a normal brick)",
			"brick_px": "brick pixels visible in top_rgb_v2",
		},
		"menagerie_commit": "370c43965323c0944b433b202aac5058b5f907a9",
		"episodes": episodes,
		"rejected_cameras": sum(r["rejected_cameras"] for r in episodes),
	}
	(out / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
	return index
