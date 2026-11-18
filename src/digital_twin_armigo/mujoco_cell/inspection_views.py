"""What an inspection camera sees of the held brick, measured from segmentation renders of a ``face_probes`` cell.

:func:`capture_view` turns the current simulator state (or a kinematic copy, see :class:`HeldBrickPoser`) into an
:class:`InspectionView`: visible pixels per brick face, of the defect cross and of the brick, plus an optional colour
crop around the brick for an image-based detector. :class:`PixelOracleInspector` turns a sequence of views into a
visibility-gated label. Both read hidden ground truth (face probes, defect geoms) and are for generating and auditing
teacher data only; the colour crop is the only part a deployable detector may consume.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from ..inspection import InspectionResult
from ..presentation import FACES
from ._mj import mujoco
from .cell import Cell


@dataclass(frozen=True)
class ShowPose:
	"""A presentation pose: the lift pose with wrist flex (rad) and wrist roll (degrees) replaced."""

	flex: float
	roll_deg: float


@dataclass(frozen=True)
class InspectionView:
	"""One camera's view of the brick. Pixel counts are inside ``crop`` when one was given."""

	camera: str
	faces_px: tuple[int, ...]  # visible pixels per face, FACES order (occlusion by the arm included)
	defect_px: int  # visible pixels of the defect cross (the hidden probe cross on a normal brick)
	brick_px: int
	brick_full_px: int  # brick pixels with the arm removed
	bbox: tuple[int, int, int, int] | None  # brick pixel bounding box (x, y, w, h)
	pose: ShowPose | None = None
	frame: int = -1
	image: np.ndarray | None = field(default=None, repr=False, compare=False)  # square RGB crop around the brick

	@property
	def visible_fraction(self) -> float:
		return self.brick_px / self.brick_full_px if self.brick_full_px else 0.0


def _render_rgb(cell: Cell, camera: str, data) -> np.ndarray:
	cell.renderer.disable_depth_rendering()
	cell.renderer.update_scene(data, camera)
	return cell.renderer.render()


def square_crop(
	img: np.ndarray, bbox: tuple[int, int, int, int], pad: float, size: int, clamp: bool = False
) -> np.ndarray:
	"""Square crop of ``img`` centred on ``bbox`` with ``pad`` × its larger side added on each side, resized to
	``size`` × ``size`` (nearest neighbour; out-of-image area is black). With ``clamp``, a crop that fits inside the
	image (``side`` ≤ both image dimensions) is shifted so it lies fully inside it, without a black border; a crop
	that does not fit is left as is, and a crop already inside is unchanged."""
	x, y, w, h = bbox
	side = max(w, h) * (1.0 + 2.0 * pad)
	cx, cy = x + w / 2.0, y + h / 2.0
	xs = np.round(cx - side / 2.0 + (np.arange(size) + 0.5) * side / size).astype(int)
	ys = np.round(cy - side / 2.0 + (np.arange(size) + 0.5) * side / size).astype(int)
	if clamp:
		if side <= img.shape[1]:
			xs = xs + max(0, -int(xs.min())) - max(0, int(xs.max()) - (img.shape[1] - 1))
		if side <= img.shape[0]:
			ys = ys + max(0, -int(ys.min())) - max(0, int(ys.max()) - (img.shape[0] - 1))
	xs = np.clip(xs, -1, img.shape[1])
	ys = np.clip(ys, -1, img.shape[0])
	out = np.zeros((size, size, img.shape[2]), dtype=img.dtype)
	vx = (xs >= 0) & (xs < img.shape[1])
	vy = (ys >= 0) & (ys < img.shape[0])
	out[np.ix_(vy, vx)] = img[np.ix_(ys[vy], xs[vx])]
	return out


def _brick_bbox(cell, camera: str, crop, data) -> tuple[int, int, int, int] | None:
	"""Pixel bounding box (x, y, w, h, in the crop) of the brick's main blob in ``camera``.

	``Cell.face_visibility`` takes the 1st-99th percentile extent of every brick pixel, which a few stray thin streaks
	of brick-mesh pixels far from the brick (segmentation artefacts seen in the top camera) can stretch to most of the
	image. Here the brick mask is closed with a small kernel and only connected components with at least a fifth of the
	largest one's area are kept."""
	import cv2

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
	mask = np.isin(geom, cell.brick_geoms).astype(np.uint8)
	if not mask.any():
		return None
	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
	_, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
	areas = stats[1:, cv2.CC_STAT_AREA]
	keep = stats[1:][areas >= 0.2 * areas.max()]
	x0 = int(keep[:, cv2.CC_STAT_LEFT].min())
	y0 = int(keep[:, cv2.CC_STAT_TOP].min())
	x1 = int((keep[:, cv2.CC_STAT_LEFT] + keep[:, cv2.CC_STAT_WIDTH]).max())
	y1 = int((keep[:, cv2.CC_STAT_TOP] + keep[:, cv2.CC_STAT_HEIGHT]).max())
	return x0, y0, x1 - x0, y1 - y0


def detector_crop(cell, camera, crop, data, image_size, pad=0.25, clamp=False, *, rgb=None):
	"""Crop around the main brick blob, identically for detector training and inference."""
	bbox = _brick_bbox(cell, camera, crop, data)
	if bbox is None:
		return None, None
	if rgb is None:
		rgb = _render_rgb(cell, camera, data)
	if crop is not None:
		x, y, w, h = crop
		rgb = rgb[y : y + h, x : x + w]
	return square_crop(rgb, bbox, pad, image_size, clamp=clamp), bbox


def capture_view(
	cell: Cell,
	camera: str,
	*,
	crop: tuple[int, int, int, int] | None = None,
	pose: ShowPose | None = None,
	frame: int = -1,
	data=None,
	image_size: int | None = None,
	image_pad: float = 0.25,
) -> InspectionView:
	"""Measure ``camera``'s view of the brick. With ``image_size``, also render the colour image and keep a square
	crop around the brick (None when the brick is not in view)."""
	vis = cell.face_visibility(camera, crop=crop, data=data)
	image = None
	if image_size is not None and vis["bbox"] is not None:
		rgb = _render_rgb(cell, camera, cell.data if data is None else data)
		if crop is not None:
			x, y, w, h = crop
			rgb = rgb[y : y + h, x : x + w]
		image = square_crop(rgb, vis["bbox"], image_pad, image_size)
	return InspectionView(
		camera=camera,
		faces_px=tuple(int(v) for v in vis["faces"]),
		defect_px=int(vis["defect"]),
		brick_px=int(vis["brick"]),
		brick_full_px=int(vis["brick_full"]),
		bbox=vis["bbox"],
		pose=pose,
		frame=frame,
		image=image,
	)


class HeldBrickPoser:
	"""Kinematic copies of the current state with the arm at other joint angles and the held brick moved rigidly
	with the gripper, for scoring candidate show poses without stepping the physics."""

	def __init__(self, cell: Cell) -> None:
		self.cell = cell
		self.data = copy.copy(cell.data)
		m, d = cell.model, cell.data
		rot_g = d.xmat[cell.grip_body].reshape(3, 3)
		rot_b = d.xmat[cell.brick_body].reshape(3, 3)
		self.rel_p = rot_g.T @ (d.xpos[cell.brick_body] - d.xpos[cell.grip_body])
		self.rel_r = rot_g.T @ rot_b
		self._m = m

	def at(self, q: np.ndarray):
		"""The kinematic copy with arm joints ``q`` (6,) and the brick carried along; returns its ``MjData``."""
		cell, d = self.cell, self.data
		d.qpos[cell.qadr] = q
		mujoco.mj_kinematics(self._m, d)
		rot_g = d.xmat[cell.grip_body].reshape(3, 3)
		adr = cell.brick_qadr
		d.qpos[adr : adr + 3] = d.xpos[cell.grip_body] + rot_g @ self.rel_p
		quat = np.zeros(4)
		mujoco.mju_mat2Quat(quat, (rot_g @ self.rel_r).flatten())
		d.qpos[adr + 3 : adr + 7] = quat
		mujoco.mj_kinematics(self._m, d)
		# mj_kinematics leaves cameras and lights where they were; the wrist camera rides on the arm
		mujoco.mj_camlight(self._m, d)
		return d

	def brick_min_z(self) -> float:
		"""Lowest brick corner above the table in the current copy (metres)."""
		cell, d = self.cell, self.data
		half = np.asarray(cell.layout.brick_size) / 2.0
		corners = np.array([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * half
		rot = d.xmat[cell.brick_body].reshape(3, 3)
		return float((d.xpos[cell.brick_body] + corners @ rot.T)[:, 2].min())


class PixelOracleInspector:
	"""Visibility-gated defect label from exact pixel counts (ground truth, for teacher data and audits only).

	A brick is ``defective`` once some view shows at least ``min_defect_px`` pixels of the defect cross, and
	``normal`` once every face in ``eligible_faces`` has shown at least ``min_face_px`` pixels in some view; until
	then the result is ``UNKNOWN``. ``eligible_faces`` are the faces a defect may be on.
	"""

	def __init__(
		self,
		defective_gt: bool,
		*,
		min_defect_px: int = 25,
		min_face_px: int = 60,
		eligible_faces: Iterable[int] = range(len(FACES)),
	) -> None:
		self.defective_gt = bool(defective_gt)
		self.min_defect_px = int(min_defect_px)
		self.min_face_px = int(min_face_px)
		self.eligible = tuple(sorted({int(f) for f in eligible_faces}))
		self.reset()

	def reset(self) -> None:
		self.observed: set[int] = set()
		self.defect_seen = False

	def observe(self, view: InspectionView) -> InspectionResult:
		seen = tuple(f for f, n in enumerate(view.faces_px) if n >= self.min_face_px)
		self.observed.update(seen)
		if self.defective_gt and view.defect_px >= self.min_defect_px:
			self.defect_seen = True
		observed = tuple(sorted(self.observed))
		ok = view.brick_px > 0
		if self.defective_gt and self.defect_seen:
			return InspectionResult("KNOWN", "defective", "sim_oracle", ok, seen, observed, "defect_pixels_visible")
		if not self.defective_gt and set(self.eligible) <= self.observed:
			return InspectionResult("KNOWN", "normal", "sim_oracle", ok, seen, observed, "all_eligible_faces_seen")
		# The same wording for both classes: the reason may reach a planner and must not reveal the label.
		missing = [f for f in self.eligible if f not in self.observed]
		return InspectionResult("UNKNOWN", None, "sim_oracle", ok, seen, observed, f"uncovered_faces:{missing}")
