"""M2 acceptance probe of the Isaac bridge (run in the main environment, spawns the Isaac server itself).

Per layout case: (1) physics time is unchanged by render calls, (2) projected brick corners land on the rendered brick
(top, front, wrist), (3) top depth at a free table point equals the expected optical-axis distance, (4) a wrist render
after moving the arm shows the new pose (no previous-frame lag), (5) three replays of one joint-target sequence from
reset agree, (6) peak VRAM and per-render / per-step latency, (7) save_replay writes a motion.npz with the right frame
count and bytes per frame. Extras: (8) the wrist camera pose agrees with MuJoCo's at the same joints, (9) checkpoint /
restore replays identically and is refused while the gripper is closed.

    .venv/Scripts/python scripts/isaac/bridge_probe.py --out results/isaac-bridge/probe-600008
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from digital_twin_armigo.agent.twin_mujoco import INITIAL_Q, sample_case
from digital_twin_armigo.constants import SO101_MJCF_DIR
from digital_twin_armigo.isaac_bridge import BridgeError, IsaacBridge, scene_payload
from digital_twin_armigo.mujoco_cell._mj import mujoco
from digital_twin_armigo.mujoco_cell.cell import Cell
from digital_twin_armigo.physics import load_physics_spec

CAMERAS = ("top", "front", "wrist_cam")
GRIPPER_OPEN, GRIPPER_HOLD = 0.85, 0.06
Q_PRE = np.array([0.0, -0.35, 0.25, 1.35, 0.0, GRIPPER_OPEN])


# -- small numpy helpers ---------------------------------------------------------------------------------------------
def quat_to_rot(q) -> np.ndarray:
	x, y, z, w = q / np.linalg.norm(q)
	return np.array(
		[
			[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
			[2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
			[2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
		]
	)


def project(cam_pose: np.ndarray, intr: dict, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	"""Pixels (u right, v down) and depth along the optical axis of world points; MuJoCo/OpenGL camera frame."""
	c = (np.linalg.inv(cam_pose) @ np.c_[pts, np.ones(len(pts))].T).T[:, :3]
	z = -c[:, 2]
	zz = np.where(z > 1e-6, z, np.nan)
	return np.stack([intr["cx"] + intr["fx"] * c[:, 0] / zz, intr["cy"] - intr["fy"] * c[:, 1] / zz], axis=1), z


def brick_corners(pos, quat_xyzw, size) -> np.ndarray:
	signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=np.float64)
	return pos + (signs * (np.asarray(size) / 2.0)) @ quat_to_rot(np.asarray(quat_xyzw)).T


def hull_mask(uv: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
	mask = np.zeros(shape, np.uint8)
	if np.isnan(uv).any():
		return mask.astype(bool)
	hull = cv2.convexHull(np.round(uv).astype(np.int32))
	cv2.fillConvexPoly(mask, hull, 1)
	return mask.astype(bool)


def colour_mask(rgb: np.ndarray, brick_rgb) -> np.ndarray:
	"""Pixels with the brick's hue and enough saturation (hue is kept by the lighting, brightness is not)."""
	hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV_FULL).astype(np.float64)
	ref = cv2.cvtColor(np.uint8([[np.clip(np.asarray(brick_rgb) * 255, 0, 255)]]), cv2.COLOR_RGB2HSV_FULL)[0, 0]
	dh = np.abs(hsv[..., 0] - float(ref[0]))
	dh = np.minimum(dh, 256 - dh) * 360.0 / 256.0
	return (dh < 14) & (hsv[..., 1] > 0.28 * 255) & (hsv[..., 2] > 0.25 * 255)


def projection_check(r: dict, state: dict, size, brick_rgb, out_png: Path) -> dict:
	rgb = r["rgb"]
	uv, z = project(r["cam_pose"], r["intrinsics"], brick_corners(state["brick_pos"], state["brick_quat_xyzw"], size))
	hull = hull_mask(uv, rgb.shape[:2])
	seen = colour_mask(rgb, brick_rgb)
	grown = cv2.dilate(hull.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
	n_seen, n_hull = int(seen.sum()), int(hull.sum())
	# only the part of the image the hull can explain (the hull may leave the frame)
	precision = float((seen & grown).sum() / max(n_seen, 1))
	recall = float((seen & hull).sum() / max(n_hull, 1))
	ys, xs = np.nonzero(seen)
	centroid = [float(xs.mean()), float(ys.mean())] if n_seen else None
	hc = hull_mask(uv, rgb.shape[:2])
	hy, hx = np.nonzero(hc)
	hull_centre = [float(hx.mean()), float(hy.mean())] if n_hull else None
	dbg = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
	if not np.isnan(uv).any():
		for a, b in ((0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)):
			cv2.line(dbg, tuple(int(v) for v in uv[a]), tuple(int(v) for v in uv[b]), (0, 255, 0), 1)
	dbg[seen] = (0.6 * dbg[seen] + 0.4 * np.array([255, 0, 255])).astype(np.uint8)
	out_png.parent.mkdir(parents=True, exist_ok=True)
	cv2.imwrite(str(out_png), dbg)
	return {
		"hull_px": n_hull,
		"colour_px": n_seen,
		"precision": precision,
		"recall": recall,
		"centre_err_px": float(np.hypot(centroid[0] - hull_centre[0], centroid[1] - hull_centre[1]))
		if centroid and hull_centre
		else None,
		"min_depth_m": float(np.nanmin(z)),
	}


# -- kinematics in the MuJoCo cell (only to find reachable joint targets) -------------------------------------------
def ik_position(cell: Cell, target: np.ndarray, q0: np.ndarray, iters: int = 300) -> np.ndarray:
	m, d = cell.model, cell.data
	q = np.asarray(q0, dtype=np.float64).copy()
	jacp = np.zeros((3, m.nv))
	for _ in range(iters):
		cell.set_joints(q)
		err = target - cell.tcp_pos()
		if np.linalg.norm(err) < 5e-4:
			break
		mujoco.mj_jacSite(m, d, jacp, None, cell.site)
		j = jacp[:, cell.dofadr[:5]]
		dq = j.T @ np.linalg.solve(j @ j.T + 1e-4 * np.eye(3), err)
		q[:5] = np.clip(q[:5] + np.clip(dq, -0.1, 0.1), cell.jnt_range[:5, 0], cell.jnt_range[:5, 1])
	return q


def wrist_pose_mj(cell: Cell, q: np.ndarray) -> np.ndarray:
	cell.set_joints(q)
	cid = cell.camera_id("wrist_cam")
	out = np.eye(4)
	out[:3, :3] = cell.data.cam_xmat[cid].reshape(3, 3)
	out[:3, 3] = cell.data.cam_xpos[cid]
	return out


def find_wrist_views(cell: Cell, brick: np.ndarray, rng: np.random.Generator) -> list[np.ndarray]:
	"""Joint targets from which the wrist camera looks at the brick from 0.12 to 0.3 m, arm clear of the table."""
	intr = {"fx": 407.4, "fy": 407.4, "cx": 320.0, "cy": 240.0}
	found = []
	for _ in range(6000):
		q = Q_PRE + np.r_[rng.normal(0, 0.45, 5), 0.0]
		q[:5] = np.clip(q[:5], cell.jnt_range[:5, 0], cell.jnt_range[:5, 1])
		pose = wrist_pose_mj(cell, q)
		uv, z = project(pose, intr, brick[None])
		if not (0.12 < z[0] < 0.3 and 120 < uv[0, 0] < 520 and 90 < uv[0, 1] < 390):
			continue
		if pose[2, 3] < 0.06 or cell.robot_min_z() < 0.02:
			continue
		found.append((float(np.linalg.norm(uv[0] - [320, 240])), q))
	found.sort(key=lambda t: t[0])
	if len(found) < 2:
		raise RuntimeError("no wrist view of the brick found")
	a = found[0][1]
	b = max((q for _, q in found[:200]), key=lambda q: float(np.linalg.norm(q - a)))
	return [a, b]


def build_sequence(cell: Cell, brick_xy) -> np.ndarray:
	"""Fixed joint-target sequence: approach, descend around the brick, close, lift (contact included)."""
	bx, by = float(brick_xy[0]), float(brick_xy[1])
	q_above = ik_position(cell, np.array([bx, by, 0.08]), Q_PRE)
	q_at = ik_position(cell, np.array([bx, by, 0.014]), q_above)
	waypoints = [(INITIAL_Q, 40), (Q_PRE, 60), (q_above, 60), (q_at, 50)]
	frames = []
	prev = INITIAL_Q.copy()
	for target, n in waypoints:
		for k in range(1, n + 1):
			s = k / n
			s = 10 * s**3 - 15 * s**4 + 6 * s**5
			frames.append(prev + s * (np.asarray(target) - prev))
		prev = np.asarray(target, dtype=np.float64)
	for k in range(1, 41):  # close the gripper on the brick
		q = prev.copy()
		q[5] = GRIPPER_OPEN + k / 40 * (GRIPPER_HOLD - GRIPPER_OPEN)
		frames.append(q)
	closed = frames[-1]
	q_up = ik_position(cell, np.array([bx, by, 0.12]), closed)
	q_up[5] = GRIPPER_HOLD
	for k in range(1, 61):
		s = k / 60
		s = 10 * s**3 - 15 * s**4 + 6 * s**5
		frames.append(closed + s * (q_up - closed))
	return np.asarray(frames)


# -- VRAM sampler --------------------------------------------------------------------------------------------------
class VramSampler(threading.Thread):
	def __init__(self) -> None:
		super().__init__(daemon=True)
		self.samples: list[tuple[float, int]] = []
		self.stop = threading.Event()

	def run(self) -> None:
		while not self.stop.is_set():
			try:
				out = (
					subprocess.run(
						["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
						capture_output=True,
						text=True,
						check=False,
						timeout=5,
					)
					.stdout.strip()
					.splitlines()[0]
				)
				self.samples.append((time.time(), int(out)))
			except Exception:  # noqa: BLE001, S110 - sampling is best effort
				pass
			self.stop.wait(0.5)


def lat(values: list[float]) -> dict:
	a = np.asarray(values) * 1e3
	return {
		"n": len(a),
		"median_ms": float(np.median(a)),
		"p95_ms": float(np.percentile(a, 95)),
		"max_ms": float(a.max()),
	}


def quat_angle(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
	dot = np.abs(np.sum(q1 * q2, axis=-1)) / (np.linalg.norm(q1, axis=-1) * np.linalg.norm(q2, axis=-1))
	return 2 * np.arccos(np.clip(dot, 0, 1))


# -- the probe ---------------------------------------------------------------------------------------------------
def run_case(bridge: IsaacBridge, case: list, out: Path, timings: dict) -> dict:
	seed, positive, face = case[:3]
	name = f"{seed}_{'pos' if positive else 'neg'}_f{face}"
	out = out / name
	out.mkdir(parents=True, exist_ok=True)
	layout = sample_case(seed, positive, face)
	cell = Cell(
		layout,
		SO101_MJCF_DIR,
		physics=load_physics_spec(),
		dynamic_clutter=True,
		assisted_grasp=False,
		face_probes=True,
	)
	payload = scene_payload(layout, cell)
	size, rgb_brick = layout.brick_size, layout.brick_rgb
	res: dict = {"case": case, "brick_size": list(size)}
	rng = np.random.default_rng(1)

	t0 = time.perf_counter()
	state = bridge.reset(payload)
	res["reset_s"] = time.perf_counter() - t0
	res["state_after_reset"] = {k: np.asarray(v).tolist() for k, v in state.items()}

	# warm renders (first RTX frames are slow and not representative), then step the arm a little
	for cam in CAMERAS:
		bridge.render(cam, depth=True)
	for i in range(10):
		state = bridge.step(INITIAL_Q, branch=0, frame=i)
	for cam in CAMERAS:
		bridge.render(cam, depth=True)

	# (1) physics time unchanged by renders
	before = bridge.state()
	n_before = bridge.stats()["step_count"]
	for _ in range(2):
		for cam in CAMERAS:
			t0 = time.perf_counter()
			bridge.render(cam, depth=True)
			timings.setdefault(f"render_{cam}", []).append(time.perf_counter() - t0)
	after = bridge.state()
	n_after = bridge.stats()["step_count"]
	dq = float(np.max(np.abs(before["q"] - after["q"])))
	db = float(np.max(np.abs(before["brick_pos"] - after["brick_pos"])))
	res["check1_time_unchanged"] = {
		"sim_time_before": before["sim_time"],
		"sim_time_after": after["sim_time"],
		"server_step_count": [n_before, n_after],
		"max_q_change": dq,
		"max_brick_change": db,
		"pass": before["sim_time"] == after["sim_time"] and n_before == n_after and dq == 0.0 and db == 0.0,
	}

	for i in range(10):
		bridge.step(INITIAL_Q, branch=0, frame=100 + i)
		for k, cam in enumerate(CAMERAS):
			t0 = time.perf_counter()
			bridge.render(cam, depth=(cam == "top"))
			timings.setdefault("render_after_step_first" if k == 0 else "render_same_state", []).append(
				time.perf_counter() - t0
			)
	after = bridge.state()

	# (2) projection of the brick corners on the rendered brick
	c2 = {}
	for cam in ("top", "front"):
		r = bridge.render(cam)
		c2[cam] = projection_check(r, after, size, rgb_brick, out / f"proj_{cam}.png")
		cv2.imwrite(str(out / f"{cam}.png"), cv2.cvtColor(r["rgb"], cv2.COLOR_RGB2BGR))
	for cam, c in c2.items():
		c["pass"] = (
			c["colour_px"] > 80 and c["precision"] > 0.85 and c["recall"] > 0.5 and (c["centre_err_px"] or 99) < 6
		)
	res["check2_projection"] = {**c2, "pass": all(c["pass"] for c in c2.values())}

	# (3) top depth at a free table point
	r = bridge.render("top", depth=True)
	intr = r["intrinsics"]
	best = None
	obstacles = [np.asarray(layout.green), np.asarray(layout.pink)]
	for x in np.linspace(0.1, 0.5, 21):
		for y in np.linspace(-0.3, 0.3, 21):
			p = np.array([x, y, 0.0])
			if (
				min(np.linalg.norm(p[:2] - o) for o in obstacles) < BOWL_CLEAR
				or np.linalg.norm(p[:2] - after["brick_pos"][:2]) < 0.08
			):
				continue
			uv, z = project(r["cam_pose"], intr, p[None])
			if np.isnan(uv).any() or not (40 < uv[0, 0] < 600 and 40 < uv[0, 1] < 440) or np.hypot(x, y) < 0.12:
				continue
			score = abs(uv[0, 0] - 320) + abs(uv[0, 1] - 240)
			if best is None or score < best[0]:
				best = (score, p, uv[0], float(z[0]))
	_, p, uv, z_exp = best
	ui, vi = round(uv[0]), round(uv[1])
	patch = r["depth"][vi - 2 : vi + 3, ui - 2 : ui + 3]
	z_got = float(np.median(patch))
	res["check3_top_depth"] = {
		"table_point": p.tolist(),
		"pixel": [ui, vi],
		"expected_m": z_exp,
		"rendered_m": z_got,
		"error_m": z_got - z_exp,
		"pass": abs(z_got - z_exp) < 0.005,
	}

	# (4) wrist render after moving the arm: pose A, pose B, back to A; flush sensitivity
	views = find_wrist_views(cell, np.asarray(state["brick_pos"]), rng)

	def go(q_target: np.ndarray, frames: int = 45, settle: int = 25) -> dict:
		start = np.asarray(bridge.state()["q"])
		s = None
		for k in range(1, frames + 1):
			a = k / frames
			a = 10 * a**3 - 15 * a**4 + 6 * a**5
			s = bridge.step(start + a * (q_target - start), branch=1, frame=100 + k)
		for k in range(settle):
			s = bridge.step(q_target, branch=1, frame=200 + k)
		return s

	c4 = {"views": []}
	ok = True
	for label, q_view in (("A", views[0]), ("B", views[1]), ("A2", views[0])):
		st = go(q_view)
		# one big jump so the pose really changes in the very last step before the capture
		q_jump = views[1] if label != "B" else views[0]
		st_j = bridge.step(q_jump, branch=1, frame=300)
		lag0 = bridge.render("wrist_cam", flush=0)  # no flush: shows how stale the annotator is
		r = bridge.render("wrist_cam", flush=3)  # the server default
		ref = bridge.render("wrist_cam", flush=12)
		pj = projection_check(r, st_j, size, rgb_brick, out / f"proj_wrist_{label}.png")
		pj0 = projection_check(lag0, st_j, size, rgb_brick, out / f"proj_wrist_{label}_flush0.png")
		pose_err = float(np.linalg.norm(r["cam_pose"][:3, 3] - wrist_pose_mj(cell, st_j["q"])[:3, 3]))
		diff = float(np.mean(np.abs(r["rgb"].astype(np.float32) - ref["rgb"].astype(np.float32))))
		stale = float(np.mean(np.abs(lag0["rgb"].astype(np.float32) - ref["rgb"].astype(np.float32))))
		cv2.imwrite(str(out / f"wrist_{label}.png"), cv2.cvtColor(r["rgb"], cv2.COLOR_RGB2BGR))
		good = diff < 1.0 and pj["colour_px"] > 60 and pj["precision"] > 0.8 and pj["recall"] > 0.3
		ok &= good
		c4["views"].append(
			{
				"view": label,
				"settled_at_then_jumped_to": [label, "B" if label != "B" else "A"],
				"projection_flush3": pj,
				"projection_flush0": pj0,
				"mean_abs_diff_flush3_vs_flush12": diff,
				"mean_abs_diff_flush0_vs_flush12": stale,
				"pose_err_vs_mujoco_m": pose_err,
				"pass": good,
			}
		)
	a_img = cv2.imread(str(out / "wrist_A.png")).astype(np.float32)
	b_img = cv2.imread(str(out / "wrist_B.png")).astype(np.float32)
	c4["mean_abs_diff_A_vs_B"] = float(np.mean(np.abs(a_img - b_img)))
	c4["pass"] = bool(ok and c4["mean_abs_diff_A_vs_B"] > 3.0)
	res["check4_wrist_no_lag"] = c4

	# (8) wrist camera pose vs MuJoCo at identical joints (position m, rotation deg) -- both poses at the last state
	st = bridge.state()
	r = bridge.render("wrist_cam")
	pm = wrist_pose_mj(cell, st["q"])
	rot_err = math.degrees(math.acos(np.clip((np.trace(r["cam_pose"][:3, :3].T @ pm[:3, :3]) - 1) / 2, -1, 1)))
	res["check8_wrist_pose_vs_mujoco"] = {
		"position_err_m": float(np.linalg.norm(r["cam_pose"][:3, 3] - pm[:3, 3])),
		"rotation_err_deg": rot_err,
		"pass": float(np.linalg.norm(r["cam_pose"][:3, 3] - pm[:3, 3])) < 0.002 and rot_err < 0.5,
	}

	# (5) determinism: three replays of one sequence from reset
	seq = build_sequence(cell, payload["brick"]["pos"][:2])
	runs = []
	for k in range(3):
		bridge.reset(payload)
		traj = []
		for i, q in enumerate(seq):
			t0 = time.perf_counter()
			s = bridge.step(q, branch=k, frame=i)
			timings.setdefault("step", []).append(time.perf_counter() - t0)
			traj.append(np.concatenate([s["q"], s["brick_pos"], s["brick_quat_xyzw"], s["jaw_force"]]))
		runs.append(np.asarray(traj))
		if k == 2:
			replay_frames = len(seq)
	jq = max(float(np.max(np.abs(runs[0][:, :6] - runs[i][:, :6]))) for i in (1, 2))
	bp = max(float(np.max(np.abs(runs[0][:, 6:9] - runs[i][:, 6:9]))) for i in (1, 2))
	ba = max(float(np.max(quat_angle(runs[0][:, 9:13], runs[i][:, 9:13]))) for i in (1, 2))
	same = np.all(runs[0] == runs[1], axis=1) & np.all(runs[0] == runs[2], axis=1)
	first_diff = int(np.argmin(same)) if not same.all() else None
	moved = float(np.linalg.norm(runs[0][-1, 6:9] - np.asarray(payload["brick"]["pos"])))
	res["check5_determinism"] = {
		"frames": len(seq),
		"max_joint_diff_rad": jq,
		"max_brick_pos_diff_m": bp,
		"max_brick_angle_diff_rad": ba,
		"first_differing_frame": first_diff,
		"bit_identical": bool(np.array_equal(runs[0], runs[1]) and np.array_equal(runs[0], runs[2])),
		"max_jaw_force_N": float(np.max(runs[0][:, 13:15])),
		"brick_displacement_m": moved,
		"brick_final_z": float(runs[0][-1, 8]),
		"pass": jq < 0.02 and bp < 0.002 and ba < 0.05,
	}

	# (7) replay of the last run
	stats = bridge.stats()
	res["server_timing_ms_mean"] = stats["timing_ms_mean"]
	bridge.save_replay(out)
	motion = np.load(out / "replay" / "motion.npz")
	scene_json = json.loads((out / "replay" / "scene.json").read_text(encoding="utf-8"))
	size_bytes = (out / "replay" / "motion.npz").stat().st_size
	scene_bytes = (out / "replay" / "scene.json").stat().st_size
	n = len(motion["poses"])
	res["check7_replay"] = {
		"frames_in_motion": n,
		"frames_sent": replay_frames,
		"nodes": len(scene_json["nodes"]),
		"cameras": [c["name"] for c in scene_json["cameras"]],
		"branch_unique": np.unique(motion["branch"]).tolist(),
		"frame_range": [int(motion["frame"].min()), int(motion["frame"].max())],
		"motion_npz_bytes": size_bytes,
		"scene_json_bytes": scene_bytes,
		"bytes_per_frame": size_bytes / n,
		"est_MiB_per_2500_frames": size_bytes / n * 2500 / 2**20,
		"pass": n == replay_frames and list(motion["branch"]) == [2] * n and list(motion["frame"]) == list(range(n)),
	}

	# (9) checkpoint / restore
	bridge.reset(payload)
	ok9 = True
	for i in range(30):
		bridge.step(Q_PRE if i > 5 else INITIAL_Q, branch=0, frame=i)
	token = bridge.checkpoint()
	tail = []
	for i in range(30, 60):
		s = bridge.step(seq[min(i, len(seq) - 1)], branch=0, frame=i)
		tail.append(np.concatenate([s["q"], s["brick_pos"]]))
	bridge.restore(token)
	tail2 = []
	for i in range(30, 60):
		s = bridge.step(seq[min(i, len(seq) - 1)], branch=1, frame=i)
		tail2.append(np.concatenate([s["q"], s["brick_pos"]]))
	err = float(np.max(np.abs(np.asarray(tail) - np.asarray(tail2))))
	refused = False
	for i in range(len(seq)):
		s = bridge.step(seq[i], branch=1, frame=i)
	try:
		bridge.checkpoint()
	except BridgeError as exc:
		refused = "gripper" in str(exc) or "brick" in str(exc)
	res["check9_checkpoint"] = {
		"replay_max_abs_diff": err,
		"refused_when_closed": refused,
		"pass": err < 0.02 and refused,
	}
	del ok9
	cell.close()
	return res


BOWL_CLEAR = 0.14


def main() -> int:
	ap = argparse.ArgumentParser()
	ap.add_argument("--out", type=Path, default=ROOT / "results" / "isaac-bridge" / "probe-600008")
	ap.add_argument("--cases", default="[[600008, false, 5], [600008, true, 1]]")
	ap.add_argument("--python", default=r"E:\il\v\Scripts\python.exe")
	a = ap.parse_args()
	a.out.mkdir(parents=True, exist_ok=True)
	sampler = VramSampler()
	sampler.start()
	time.sleep(1.0)
	baseline = sampler.samples[-1][1] if sampler.samples else None
	timings: dict[str, list[float]] = {}
	report: dict = {"vram_baseline_MiB": baseline, "cases": []}
	t0 = time.perf_counter()
	bridge = IsaacBridge(python=a.python, log_dir=a.out / "logs")
	report["server_start_s"] = time.perf_counter() - t0
	report["server_info"] = bridge.info
	try:
		for case in json.loads(a.cases):
			report["cases"].append(run_case(bridge, case, a.out, timings))
	finally:
		bridge.close()
		sampler.stop.set()
	peak = max((v for _, v in sampler.samples), default=None)
	report["check6_vram_latency"] = {
		"vram_peak_total_MiB": peak,
		"vram_peak_delta_MiB": None if peak is None or baseline is None else peak - baseline,
		"latency": {k: lat(v) for k, v in timings.items()},
		"note": "total includes other GPU users (a local vLLM server); delta is relative to the baseline before the bridge started",
	}
	passes = {}
	for c in report["cases"]:
		for k, v in c.items():
			if isinstance(v, dict) and "pass" in v:
				passes.setdefault(k, []).append(bool(v["pass"]))
	report["summary"] = {k: all(v) for k, v in passes.items()}
	(a.out / "probe.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
	print(json.dumps(report["summary"], indent=2))
	print(json.dumps(report["check6_vram_latency"], indent=2))
	return 0 if all(report["summary"].values()) else 1


if __name__ == "__main__":
	sys.exit(main())
