"""Eye-to-hand calibration of the fixed top camera from a marker card the gripper holds.

The solver sees only what a real cell offers: camera images, joint readings with forward kinematics, the camera
intrinsics and the printed marker sizes. Where the card sits on the gripper is unknown and is estimated together with
the camera pose, so the card can be clamped any way. The simulator places the card with a mocap body that follows the
gripper frame; ground truth is used only to score the result.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..camera import CameraPose
from ._mj import mujoco

CARD_IDS = (20, 21, 22, 23)  # clear of the cell markers (10, 11, 16, 17)
CARD_MARKER_M = 0.028  # black-square side of each card marker
CARD_GAP_M = 0.006
CARD_MARGIN_M = 0.006  # white border around the 2x2 grid
CARD_TEXTURE = "_calib_card.png"
CARD_PX_PER_M = 8000


def _cv2():
	import cv2

	return cv2


def card_side_m() -> float:
	return 2 * CARD_MARKER_M + CARD_GAP_M + 2 * CARD_MARGIN_M


def card_board():
	cv2 = _cv2()
	dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
	return cv2.aruco.GridBoard((2, 2), CARD_MARKER_M, CARD_GAP_M, dictionary, np.array(CARD_IDS))


def write_card_texture(directory: Path) -> None:
	cv2 = _cv2()
	side = round(card_side_m() * CARD_PX_PER_M)
	margin = round(CARD_MARGIN_M * CARD_PX_PER_M)
	img = card_board().generateImage((side, side), marginSize=margin, borderBits=1)
	if not cv2.imwrite(str(directory / CARD_TEXTURE), img):
		raise OSError("could not write the calibration card texture")


def card_xml() -> tuple[str, str]:
	"""(assets, worldbody) for :class:`~.cell.Cell` ``extra_xml``: a thin textured card on a mocap body."""
	h = card_side_m() / 2
	assets = (
		f'<texture name="calib_card" type="2d" file="{CARD_TEXTURE}"/>'
		'<material name="calib_card" texture="calib_card" specular="0"/>'
	)
	body = (
		'<body name="calib_card" mocap="true">'
		f'<geom type="box" size="{h:.4f} {h:.4f} 0.0005" material="calib_card" contype="0" conaffinity="0"/>'
		"</body>"
	)
	return assets, body


# --- rigid transforms (4x4) -------------------------------------------------------------------------------------


def transform(rot: np.ndarray, t: np.ndarray) -> np.ndarray:
	out = np.eye(4)
	out[:3, :3] = rot
	out[:3, 3] = t
	return out


def inverse(tf: np.ndarray) -> np.ndarray:
	rot, t = tf[:3, :3], tf[:3, 3]
	return transform(rot.T, -rot.T @ t)


def from_rvec(rvec: np.ndarray, t: np.ndarray) -> np.ndarray:
	rot, _ = _cv2().Rodrigues(np.asarray(rvec, np.float64).reshape(3))
	return transform(rot, np.asarray(t, np.float64).reshape(3))


def to_rvec(tf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	rvec, _ = _cv2().Rodrigues(tf[:3, :3])
	return rvec.reshape(3), tf[:3, 3].copy()


def rotation_error_deg(a: np.ndarray, b: np.ndarray) -> float:
	c = (np.trace(a[:3, :3].T @ b[:3, :3]) - 1) / 2
	return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


# --- camera model ------------------------------------------------------------------------------------------------


def intrinsics(pose: CameraPose, resolution: tuple[int, int]) -> np.ndarray:
	"""OpenCV camera matrix of a rendered camera. MuJoCo puts the principal point at the image centre in continuous
	coordinates, which is half a pixel up-left of it in OpenCV's pixel-centre convention."""
	w, h = resolution
	f = pose.focal_px
	return np.array([[f, 0.0, w / 2 - 0.5], [0.0, f, h / 2 - 0.5], [0.0, 0.0, 1.0]])


def camera_truth(model, data, name: str) -> np.ndarray:
	"""Pose of a MuJoCo camera in the world, in the OpenCV convention (x right, y down, z forward)."""
	cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
	rot = data.cam_xmat[cid].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
	return transform(rot, data.cam_xpos[cid])


def project(k: np.ndarray, cam_from_world: np.ndarray, points: np.ndarray) -> np.ndarray:
	p = points @ cam_from_world[:3, :3].T + cam_from_world[:3, 3]
	uv = p[:, :2] / p[:, 2:3]
	return uv * [k[0, 0], k[1, 1]] + [k[0, 2], k[1, 2]]


def table_points(k: np.ndarray, world_from_cam: np.ndarray, uv: np.ndarray, z: float) -> np.ndarray:
	"""Where the rays through pixels ``uv`` meet the horizontal plane at height ``z`` (world frame)."""
	rays = np.column_stack([(uv[:, 0] - k[0, 2]) / k[0, 0], (uv[:, 1] - k[1, 2]) / k[1, 1], np.ones(len(uv))])
	rays = rays @ world_from_cam[:3, :3].T
	origin = world_from_cam[:3, 3]
	s = (z - origin[2]) / rays[:, 2]
	return origin + s[:, None] * rays


# --- detection ---------------------------------------------------------------------------------------------------


def _detector():
	cv2 = _cv2()
	params = cv2.aruco.DetectorParameters()
	params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
	dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
	return cv2.aruco.ArucoDetector(dictionary, params)


def detect_markers(rgb: np.ndarray) -> dict[int, np.ndarray]:
	"""Corner pixels (4, 2) of every detected 4x4_50 marker, by id."""
	cv2 = _cv2()
	gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
	corners, ids, _ = _detector().detectMarkers(gray)
	if ids is None:
		return {}
	return {int(i): c.reshape(4, 2).astype(np.float64) for i, c in zip(ids.ravel(), corners, strict=True)}


def card_correspondences(markers: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray, int]:
	"""Card-frame 3D corners and their pixels for the card markers seen, and how many card markers that is."""
	board = card_board()
	ids = [i for i in CARD_IDS if i in markers]
	if not ids:
		return np.zeros((0, 3)), np.zeros((0, 2)), 0
	obj = np.concatenate([np.asarray(board.getObjPoints()[CARD_IDS.index(i)], np.float64) for i in ids])
	img = np.concatenate([markers[i] for i in ids])
	return obj.reshape(-1, 3), img.reshape(-1, 2), len(ids)


def marker_object_points(side: float) -> np.ndarray:
	"""Corners of a single square marker in its own frame, in ArUco order (TL, TR, BR, BL), z = 0."""
	s = side / 2
	return np.array([[-s, s, 0.0], [s, s, 0.0], [s, -s, 0.0], [-s, -s, 0.0]])


def marker_pose(k: np.ndarray, corners: np.ndarray, side: float) -> np.ndarray:
	"""Camera-from-marker transform of one square marker of known side."""
	cv2 = _cv2()
	ok, rvec, tvec = cv2.solvePnP(marker_object_points(side), corners, k, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
	if not ok:
		raise ValueError("marker PnP failed")
	return from_rvec(rvec, tvec)


# --- solver ------------------------------------------------------------------------------------------------------


# Arm joints whose zero offsets the solver can estimate. shoulder_pan (0) is excluded: a pan offset turns the whole arm
# about the vertical pan axis, which a camera turned about the same axis matches exactly, so the robot keeps working in
# its own (turned) frame. wrist_roll (4) is excluded: its offset is absorbed by the card pose on the gripper.
OFFSET_JOINTS = (1, 2, 3)
OFFSET_PRIOR_RAD = np.radians(10.0)  # very weak prior (1 px per 10 degrees), only against a joint that never moves


@dataclass
class Observation:
	world_from_grip: np.ndarray  # from the measured joints through forward kinematics
	obj: np.ndarray  # (n, 3) card-frame corners
	img: np.ndarray  # (n, 2) detected pixels
	q: np.ndarray | None = None  # measured arm joints, needed when joint offsets are estimated


@dataclass
class Calibration:
	world_from_cam: np.ndarray
	grip_from_card: np.ndarray
	rms_px: float
	per_pose_rms_px: list[float]
	iterations: int
	joint_offsets: np.ndarray  # radians, added to the measured joints (zeros unless estimated)


def solve_ax_xb(a: list[np.ndarray], b: list[np.ndarray]) -> np.ndarray:
	"""X with A_i X = X B_i (Park and Martin 1994): rotation from the axis-angle vectors, then translation by least
	squares. Needs at least two motions with non-parallel rotation axes."""
	cv2 = _cv2()
	alpha = [cv2.Rodrigues(ai[:3, :3])[0].reshape(3) for ai in a]
	beta = [cv2.Rodrigues(bi[:3, :3])[0].reshape(3) for bi in b]
	m = sum(np.outer(be, al) for al, be in zip(alpha, beta, strict=True))
	w, v = np.linalg.eigh(m.T @ m)
	rot = v @ np.diag(1 / np.sqrt(np.maximum(w, 1e-18))) @ v.T @ m.T
	u, _, vt = np.linalg.svd(rot)  # re-orthonormalise
	rot = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
	lhs = np.concatenate([ai[:3, :3] - np.eye(3) for ai in a])
	rhs = np.concatenate([rot @ bi[:3, 3] - ai[:3, 3] for ai, bi in zip(a, b, strict=True)])
	t, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
	return transform(rot, t)


def _initial(k: np.ndarray, obs: list[Observation]) -> tuple[np.ndarray, np.ndarray]:
	"""Closed-form start: per-pose PnP of the card, then the eye-to-hand AX = XB solution.

	With X = world-from-camera and Y = gripper-from-card, every pose gives G_i Y = X C_i (G from kinematics, C from
	PnP), so for each pair of poses (G_j G_i^-1) X = X (C_j C_i^-1)."""
	cv2 = _cv2()
	cam_from_card = []
	for o in obs:
		ok, rvec, tvec = cv2.solvePnP(o.obj, o.img, k, None, flags=cv2.SOLVEPNP_IPPE)
		if not ok:
			raise ValueError("card PnP failed")
		cam_from_card.append(from_rvec(rvec, tvec))
	grips = [o.world_from_grip for o in obs]
	pairs = [(i, j) for i in range(len(obs)) for j in range(i + 1, len(obs))]
	world_from_cam = solve_ax_xb(
		[grips[j] @ inverse(grips[i]) for i, j in pairs],
		[cam_from_card[j] @ inverse(cam_from_card[i]) for i, j in pairs],
	)
	guesses = [inverse(g) @ world_from_cam @ c for g, c in zip(grips, cam_from_card, strict=True)]
	t_med = np.median([g[:3, 3] for g in guesses], axis=0)
	grip_from_card = min(guesses, key=lambda g: np.linalg.norm(g[:3, 3] - t_med))
	return world_from_cam, grip_from_card


def _unpack(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	return from_rvec(p[0:3], p[3:6]), from_rvec(p[6:9], p[9:12])


def _offsets(p: np.ndarray, joints: tuple[int, ...], n_joints: int) -> np.ndarray:
	out = np.zeros(n_joints)
	out[list(joints)] = p[12:]
	return out


def calibrate(
	k: np.ndarray,
	obs: list[Observation],
	iterations: int = 100,
	fk=None,
	offset_joints: tuple[int, ...] = (),
) -> Calibration:
	"""Camera pose in the world and card pose on the gripper, by Levenberg-Marquardt on the reprojection error of
	every detected card corner, started from :func:`_initial`. Needs at least three poses with distinct rotations.

	With ``offset_joints`` (see :data:`OFFSET_JOINTS`) the zero offsets of those joints are estimated too: every
	observation must carry ``q``, and ``fk(q)`` gives world-from-gripper for corrected joints."""
	if len(obs) < 3:
		raise ValueError("hand-eye calibration needs at least three card poses")
	if offset_joints and (fk is None or any(o.q is None for o in obs)):
		raise ValueError("estimating joint offsets needs fk and the measured joints of every observation")
	n_joints = len(obs[0].q) if offset_joints else 0
	n_px = sum(2 * len(o.obj) for o in obs)

	def residuals(p: np.ndarray) -> np.ndarray:
		cam_from_world, grip_from_card = _unpack(p)
		if offset_joints:
			delta = _offsets(p, offset_joints, n_joints)
			grips = [fk(o.q + delta) for o in obs]
		else:
			grips = [o.world_from_grip for o in obs]
		out = [project(k, cam_from_world @ g @ grip_from_card, o.obj) - o.img for g, o in zip(grips, obs, strict=True)]
		return np.concatenate([np.concatenate(out).ravel(), p[12:] / OFFSET_PRIOR_RAD])

	world_from_cam, grip_from_card = _initial(k, obs)
	p = np.concatenate([*to_rvec(inverse(world_from_cam)), *to_rvec(grip_from_card), np.zeros(len(offset_joints))])
	r = residuals(p)
	cost, lam, it = r @ r, 1e-3, 0
	for it in range(1, iterations + 1):
		jac = np.empty((len(r), len(p)))
		for j in range(len(p)):
			dp = np.zeros_like(p)
			dp[j] = 1e-7
			jac[:, j] = (residuals(p + dp) - r) / 1e-7
		jtj, jtr = jac.T @ jac, jac.T @ r
		improved = False
		while lam < 1e8:
			step = np.linalg.solve(jtj + lam * np.diag(np.diag(jtj) + 1e-12), -jtr)
			r_new = residuals(p + step)
			if r_new @ r_new < cost:
				p, r, improved = p + step, r_new, True
				lam = max(lam / 10, 1e-9)
				break
			lam *= 10
		new_cost = r @ r
		if not improved or cost - new_cost < 1e-12 * max(cost, 1e-12):
			cost = new_cost
			break
		cost = new_cost
	cam_from_world, grip_from_card = _unpack(p)
	px = r[:n_px]
	per_pose, start = [], 0
	for o in obs:
		n = 2 * len(o.obj)
		per_pose.append(float(np.sqrt(np.mean(px[start : start + n] ** 2) * 2)))
		start += n
	rms = float(np.sqrt(px @ px / (n_px / 2)))
	offsets = _offsets(p, offset_joints, n_joints) if offset_joints else np.zeros(0)
	return Calibration(inverse(cam_from_world), grip_from_card, rms, per_pose, it, offsets)
