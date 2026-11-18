"""Simulated eye-to-hand calibration of the top camera, scored against the simulator's ground truth.

Per fixture seed: a random cell (the top camera pose and focal length are randomised as in the datasets), a card
clamped on the gripper at an unknown offset, random arm poses kept only when the top camera detects the card.
The solver gets the rendered images, the joint readings (optionally noisy or biased) and the intrinsics. It then
uses its own camera estimate to measure the printed cell markers on the table, which is how a real cell would
learn where the stickers are without a tape measure.

	uv run python scripts/calib/handeye_sim.py --seeds 20 --out results/handeye-sim
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from digital_twin_armigo.camera import CameraPose
from digital_twin_armigo.constants import CELL_MARKERS, HANDOFF_Q, REST_U, SO101_MJCF_DIR, TOP_FULL
from digital_twin_armigo.joints import normalized_to_rad
from digital_twin_armigo.mujoco_cell import handeye as he
from digital_twin_armigo.mujoco_cell._mj import mujoco
from digital_twin_armigo.mujoco_cell.cell import Cell
from digital_twin_armigo.mujoco_cell.layout import sample_layout

NOMINAL_CAMERA = Path("config/sim/top_camera_grasp.json")
# Arm poses: offsets from the hand-off pose (pan, lift, elbow, wrist flex, wrist roll), radians.
POSE_SPREAD = np.array([0.6, 0.35, 0.35, 0.5, 0.8])
NOMINAL_CARD = he.transform(np.eye(3), np.array([0.05, 0.0, 0.004]))  # beyond the jaw tips, face along gripper +z
MIN_CARD_MARKERS = 3
MARKER_Z = 0.0008  # top of the cell marker geoms above their layout z


def rot_xyz(rx: float, ry: float, rz: float) -> np.ndarray:
	cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
	x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
	y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
	z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
	return z @ y @ x


class Rig:
	"""A cell with the calibration card; renders the top camera with the card following the gripper."""

	def __init__(self, seed: int, card_jitter: bool):
		raw = json.loads(NOMINAL_CAMERA.read_text(encoding="utf-8"))["camera"]
		rng = np.random.default_rng(seed)
		self.layout = sample_layout(rng, CameraPose(**raw), defective=False)
		he.write_card_texture(SO101_MJCF_DIR)
		self.cell = Cell(self.layout, SO101_MJCF_DIR, extra_xml=he.card_xml())
		m = self.cell.model
		body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "calib_card")
		self.mocap = int(m.body_mocapid[body])
		self.scratch = mujoco.MjData(m)  # forward kinematics of measured joints, apart from the rendered state
		jitter = np.random.default_rng(seed + 10_000)
		self.grip_from_card = NOMINAL_CARD.copy()
		if card_jitter:  # clamped by hand: a few degrees and millimetres off
			self.grip_from_card = NOMINAL_CARD @ he.transform(
				rot_xyz(*np.radians(jitter.uniform(-5, 5, 3))), jitter.uniform(-0.005, 0.005, 3)
			)
		self.k = he.intrinsics(self.layout.top, TOP_FULL)
		mujoco.mj_forward(m, self.cell.data)
		self.truth = he.camera_truth(m, self.cell.data, "top")
		pan = m.joint("shoulder_pan").id
		self.pan_anchor = self.cell.data.xanchor[pan].copy()
		self.pan_axis = self.cell.data.xaxis[pan].copy()

	def robot_from_world(self, pan_bias: float) -> np.ndarray:
		"""The frame the robot believes in when its pan reading is ``pan_bias`` too high: the world turned about the
		pan axis. A camera found in this frame is what the robot needs, since it also moves in it."""
		rot, _ = cv2.Rodrigues(self.pan_axis * pan_bias)
		return he.transform(rot, self.pan_anchor - rot @ self.pan_anchor)

	def grip(self, q: np.ndarray, data=None) -> np.ndarray:
		"""World-from-gripper-frame for arm joints ``q`` (5 values)."""
		d = self.scratch if data is None else data
		d.qpos[self.cell.qadr[:5]] = q
		mujoco.mj_kinematics(self.cell.model, d)
		return he.transform(d.site_xmat[self.cell.site].reshape(3, 3).copy(), d.site_xpos[self.cell.site].copy())

	def render(self, q: np.ndarray, card: bool = True) -> np.ndarray:
		full = np.array([*q, 0.6])
		self.cell.set_joints(full)
		d = self.cell.data
		if card:
			tf = self.grip(q, d) @ self.grip_from_card
		else:
			tf = he.transform(np.eye(3), np.array([0.0, 0.0, -1.0]))  # under the table
		d.mocap_pos[self.mocap] = tf[:3, 3]
		quat = np.zeros(4)
		mujoco.mju_mat2Quat(quat, tf[:3, :3].flatten())
		d.mocap_quat[self.mocap] = quat
		mujoco.mj_forward(self.cell.model, d)
		return self.cell.rgb("top")


def collect(
	rig: Rig, n: int, rng: np.random.Generator, pixel_noise: float, spread: float = 1.0, max_tries: int = 800
):
	"""Random reachable arm poses where the top camera sees at least MIN_CARD_MARKERS card markers."""
	poses = []
	lo, hi = rig.cell.jnt_range[:5, 0], rig.cell.jnt_range[:5, 1]
	tries = 0
	while len(poses) < n and tries < max_tries:
		tries += 1
		q = np.clip(np.array(HANDOFF_Q) + rng.uniform(-1, 1, 5) * POSE_SPREAD * spread, lo, hi)
		card = rig.grip(q) @ rig.grip_from_card
		if rig.grip(q)[2, 3] < 0.08 or card[2, 3] < 0.06:
			continue
		rgb = rig.render(q)
		if pixel_noise > 0:
			rgb = np.clip(rgb + rng.normal(0, pixel_noise, rgb.shape), 0, 255).astype(np.uint8)
		obj, img, seen = he.card_correspondences(he.detect_markers(rgb))
		if seen >= MIN_CARD_MARKERS:
			poses.append((q, obj, img, rgb))
	return poses, tries


def workspace_error_mm(k: np.ndarray, truth: np.ndarray, est: np.ndarray, z: float = 0.0096) -> dict:
	"""Error of locating table points (brick-centre height) through the estimated camera, over the reachable area."""
	xs, ys = np.meshgrid(np.arange(0.12, 0.41, 0.02), np.arange(-0.26, 0.27, 0.02))
	pts = np.column_stack([xs.ravel(), ys.ravel(), np.full(xs.size, z)])
	uv = he.project(k, he.inverse(truth), pts)
	inside = (uv[:, 0] >= 0) & (uv[:, 0] < TOP_FULL[0]) & (uv[:, 1] >= 0) & (uv[:, 1] < TOP_FULL[1])
	err = np.linalg.norm(he.table_points(k, est, uv[inside], z)[:, :2] - pts[inside, :2], axis=1) * 1000
	return {
		"points": int(inside.sum()),
		"mean": float(err.mean()),
		"p95": float(np.percentile(err, 95)),
		"max": float(err.max()),
	}


def measure_markers(rig: Rig, est: np.ndarray, frame: np.ndarray, found: dict) -> list[dict]:
	"""Locate the printed cell markers with the estimated camera; truth is expressed in ``frame``."""
	out = []
	for mid, x, y, z, _ in rig.layout.markers:
		uv = he.project(rig.k, he.inverse(rig.truth), np.array([[x, y, z + MARKER_Z]]))[0]
		in_view = bool(0 <= uv[0] < TOP_FULL[0] and 0 <= uv[1] < TOP_FULL[1])
		if mid not in found:
			out.append({"id": mid, "in_view": in_view, "detected": False})
			continue
		cam_from_marker = he.marker_pose(rig.k, found[mid], CELL_MARKERS[mid])
		centre = (est @ cam_from_marker)[:3, 3]
		ray = he.table_points(rig.k, est, found[mid].mean(axis=0, keepdims=True), z + MARKER_Z)[0]
		truth = (frame @ [x, y, z + MARKER_Z, 1.0])[:3]
		out.append(
			{
				"id": mid,
				"in_view": in_view,
				"detected": True,
				"pnp_error_mm": float(np.linalg.norm(centre - truth) * 1000),
				"plane_error_mm": float(np.linalg.norm(ray[:2] - truth[:2]) * 1000),
			}
		)
	return out


def score(rig: Rig, cal: he.Calibration, frame: np.ndarray, bias: np.ndarray, found: dict) -> dict:
	est, truth = cal.world_from_cam, frame @ rig.truth
	out = {
		"rms_px": cal.rms_px,
		"iterations": cal.iterations,
		"cam_pos_err_mm": float(np.linalg.norm(est[:3, 3] - truth[:3, 3]) * 1000),
		"cam_rot_err_deg": he.rotation_error_deg(est, truth),
		"workspace_mm": workspace_error_mm(rig.k, truth, est),
		"markers": measure_markers(rig, est, frame, found),
	}
	if len(cal.joint_offsets):
		# the solver's offsets are added to the readings, so the right answer is minus the bias
		joints = list(he.OFFSET_JOINTS)
		out["offset_err_deg"] = np.degrees(cal.joint_offsets[joints] + bias[joints]).round(4).tolist()
	return out


def run_seed(seed: int, args, joint_noise_deg: float, joint_bias_deg: float, keep_images: Path | None):
	rng = np.random.default_rng(seed + 20_000)
	rig = Rig(seed, card_jitter=True)
	try:
		poses, tries = collect(rig, args.poses, rng, args.pixel_noise, args.spread)
		if len(poses) < 3:
			return {"seed": seed, "ok": False, "poses": len(poses), "tries": tries}
		bias = np.radians(rng.normal(0, joint_bias_deg, 5)) if joint_bias_deg else np.zeros(5)
		obs = []
		for q, obj, img, _ in poses:
			q_meas = q + bias + np.radians(rng.normal(0, joint_noise_deg, 5))
			obs.append(he.Observation(rig.grip(q_meas), obj, img, q=q_meas))
		frame = rig.robot_from_world(bias[0])
		found = he.detect_markers(rig.render(np.array(normalized_to_rad(np.array(REST_U)))[:5], card=False))
		result = {
			"seed": seed,
			"ok": True,
			"poses": len(poses),
			"tries": tries,
			"corners": int(sum(len(o.obj) for o in obs)),
			"bias_deg": np.degrees(bias).round(4).tolist(),
			"fixed": score(rig, he.calibrate(rig.k, obs), frame, bias, found),
			"offsets": score(
				rig, he.calibrate(rig.k, obs, fk=rig.grip, offset_joints=he.OFFSET_JOINTS), frame, bias, found
			),
		}
		if keep_images is not None:
			keep_images.mkdir(parents=True, exist_ok=True)
			tiles = []
			for q, _, img, rgb in poses[:6]:
				tile = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
				for u, v in img:
					cv2.circle(tile, (round(u), round(v)), 3, (0, 255, 0), -1)
				tiles.append(tile)
			while len(tiles) < 6:
				tiles.append(np.zeros_like(tiles[0]))
			grid = np.vstack([np.hstack(tiles[0:3]), np.hstack(tiles[3:6])])
			cv2.imwrite(str(keep_images / f"poses-seed{seed}.jpg"), cv2.resize(grid, None, fx=0.5, fy=0.5))
		return result
	finally:
		rig.cell.close()


def summarise(rows: list[dict]) -> dict:
	ok = [r for r in rows if r["ok"]]

	def stats(values):
		v = np.array(values, dtype=float)
		return {"median": float(np.median(v)), "p90": float(np.percentile(v, 90)), "max": float(v.max())}

	out = {"seeds": len(rows), "solved": len(ok), "poses": stats([r["poses"] for r in ok])}
	for solver in ("fixed", "offsets"):
		res = [r[solver] for r in ok]
		markers = [m for r in res for m in r["markers"] if m["detected"]]
		out[solver] = {
			"rms_px": stats([r["rms_px"] for r in res]),
			"cam_pos_err_mm": stats([r["cam_pos_err_mm"] for r in res]),
			"cam_rot_err_deg": stats([r["cam_rot_err_deg"] for r in res]),
			"workspace_mean_mm": stats([r["workspace_mm"]["mean"] for r in res]),
			"workspace_max_mm": stats([r["workspace_mm"]["max"] for r in res]),
			"marker_pnp_err_mm": stats([m["pnp_error_mm"] for m in markers]) if markers else None,
			"marker_plane_err_mm": stats([m["plane_error_mm"] for m in markers]) if markers else None,
			"markers_detected": f"{len(markers)}/{sum(len(r['markers']) for r in res)}",
			"markers_in_view": sum(m["in_view"] for r in res for m in r["markers"]),
		}
		if solver == "offsets":
			out[solver]["offset_abs_err_deg"] = stats([abs(e) for r in res for e in r["offset_err_deg"]])
	return out


def main(argv: list[str] | None = None) -> int:
	parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	parser.add_argument("--seeds", type=int, default=20)
	parser.add_argument("--first-seed", type=int, default=900001)
	parser.add_argument("--poses", type=int, default=15)
	parser.add_argument("--spread", type=float, default=1.0, help="scale of the random arm pose offsets")
	parser.add_argument("--pixel-noise", type=float, default=2.0, help="Gaussian RGB noise, grey levels")
	parser.add_argument(
		"--conditions",
		default="0:0,0.3:0,1.0:0,0:0.5",
		help="comma-separated joint_noise_deg:joint_bias_deg pairs",
	)
	parser.add_argument("--out", type=Path, default=Path("results/handeye-sim"))
	args = parser.parse_args(argv)
	args.out.mkdir(parents=True, exist_ok=True)
	report = {"config": {k: str(v) for k, v in vars(args).items()}, "conditions": {}}
	for cond in args.conditions.split(","):
		noise, bias = (float(x) for x in cond.split(":"))
		rows = []
		for i in range(args.seeds):
			seed = args.first_seed + i
			images = args.out / "images" if (i == 0 and noise == 0 and bias == 0) else None
			row = run_seed(seed, args, noise, bias, images)
			rows.append(row)
			if row["ok"]:
				fx, of = row["fixed"], row["offsets"]
				print(
					f"noise={noise} bias={bias} seed={seed} poses={row['poses']} "
					f"ws_max fixed={fx['workspace_mm']['max']:.2f}mm offsets={of['workspace_mm']['max']:.2f}mm "
					f"offset_err={of['offset_err_deg']}",
					flush=True,
				)
			else:
				print(f"noise={noise} bias={bias} seed={seed} FAILED poses={row['poses']}", flush=True)
		key = f"noise{noise}_bias{bias}"
		report["conditions"][key] = {"summary": summarise(rows), "rows": rows}
		print(key, json.dumps(report["conditions"][key]["summary"], indent=1), flush=True)
	(args.out / "summary.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
