"""Pick a brick, put it back down on the TABLE at a chosen point, then inspect it there (wrist scan + bottom view).

:class:`RelocateExpert` reuses the closed-loop expert's grasp, lift and carry machinery and adds a table put-down
(release a few millimetres above the table, wait for the brick to settle, retreat) and two inspection stages run
on the live physics state:

* **wrist scan** -- the gripper hovers over the brick with the jaws straight down (falling back to the tilted,
  outside sweep where straight down is out of reach) and the wrist roll orbits the off-axis wrist camera around it;
  the hover poses are chosen kinematically to cover the top and four side faces (:func:`plan_table_scan`);
* **bottom view** -- the brick is grasped again, lifted, and the arm pose that shows its bottom face largest to the
  top camera (searched kinematically, :func:`plan_bottom_view`) is held; then the brick is put down again.

Every view is recorded by the ``on_view(stage, expert)`` callback after its hold. The relocation target is the
caller's choice (see :func:`relocation_target`).
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from ..constants import FPS, TOP_FULL
from ._mj import mujoco
from .cell import Cell
from .ik import ik, ik_reach
from .inspection_planning import ViewCandidate, greedy_face_cover, path_min_z
from .inspection_views import HeldBrickPoser, capture_view
from .physics_expert import ClosedLoopExpert, ExpertParams

__all__ = [
	"RelocateExpert",
	"RelocateParams",
	"max_tip_reach",
	"plan_bottom_view",
	"plan_table_scan",
	"relocation_target",
	"shoulder_anchor",
]

SCAN_FACES = (0, 1, 2, 3, 4)


def shoulder_anchor(cell: Cell) -> np.ndarray:
	"""World position of the shoulder-pan joint anchor (x, y, z)."""
	jid = mujoco.mj_name2id(cell.model, mujoco.mjtObj.mjOBJ_JOINT, "shoulder_pan")
	return cell.data.xanchor[jid].copy()


def max_tip_reach(cell: Cell, samples: int = 4000, seed: int = 0) -> float:
	"""Largest distance (m) from the shoulder-pan anchor to the gripper frame (jaw tip region) over random joint
	vectors within the joint limits (kinematic)."""
	rng = np.random.default_rng(seed)
	d = copy.copy(cell.data)
	anchor = shoulder_anchor(cell)
	lo, hi = cell.jnt_range[:, 0], cell.jnt_range[:, 1]
	best = 0.0
	for _ in range(samples):
		d.qpos[cell.qadr] = rng.uniform(lo, hi)
		mujoco.mj_kinematics(cell.model, d)
		best = max(best, float(np.linalg.norm(d.site_xpos[cell.site] - anchor)))
	return best


def relocation_target(
	brick_xy: Sequence[float], anchor_xy: Sequence[float], zone_outer_m: float, margin_m: float = 0.04
) -> tuple[np.ndarray, float]:
	"""The point to put a brick down so it can be scanned straight down: same azimuth around the shoulder axis, at
	``zone_outer_m - margin_m`` from it (or where it already is, if that is closer). Returns (xy, carry distance)."""
	b, a = np.asarray(brick_xy, dtype=np.float64), np.asarray(anchor_xy, dtype=np.float64)[:2]
	r = float(np.linalg.norm(b - a))
	r_new = min(r, zone_outer_m - margin_m)
	u = (b - a) / max(r, 1e-9)
	return a + r_new * u, r - r_new


def site_err_down(cell, d, q, target) -> tuple[float, float]:
	"""Gripper-frame position error to ``target`` and cosine of the approach axis with straight down."""
	d.qpos[cell.qadr] = q
	mujoco.mj_kinematics(cell.model, d)
	return float(np.linalg.norm(d.site_xpos[cell.site] - target)), float(-d.site_xmat[cell.site].reshape(3, 3)[2, 0])


def plan_table_scan(
	cell: Cell,
	kin,
	q_seed: np.ndarray,
	*,
	gripper: float,
	heights: Sequence[float] = (0.06, 0.075, 0.09),
	offset_m: float = 0.03,
	rolls: int = 16,
	min_px: int = 800,
	max_views: int = 5,
	ik_tol: float = 0.004,
	down_tol_deg: float = 8.0,
	min_z: float = 0.02,
	restarts: int = 12,
) -> tuple[list[ViewCandidate], str]:
	"""Hover poses over the brick (at its CURRENT simulated pose) whose wrist views cover the top and four side faces:
	jaws straight down if that covers them ("vertical"), else the tilted sweep ("outside"); returns (views, mode),
	the views in roll order. ``kin`` is the expert's private :class:`Kinematics` (IK runs there)."""
	m = cell.model
	d = copy.copy(cell.data)
	bpos = cell.data.xpos[cell.brick_body].copy()
	anchor = shoulder_anchor(cell)[:2]
	radial = (bpos[:2] - anchor) / max(float(np.linalg.norm(bpos[:2] - anchor)), 1e-9)
	tang = np.array([-radial[1], radial[0]])
	offsets = [np.zeros(2), -offset_m * radial, offset_m * radial, offset_m * tang, -offset_m * tang]
	lo, hi = cell.jnt_range[4]
	roll_grid = np.linspace(lo, hi, rolls)
	cos_tol = math.cos(math.radians(down_tol_deg))
	seeds = np.random.default_rng(0).uniform(cell.jnt_range[:, 0], cell.jnt_range[:, 1], (restarts, 6))

	def sweep(target: np.ndarray, vertical: bool) -> list[ViewCandidate]:
		out: list[ViewCandidate] = []
		q_prev = None
		if vertical:
			for s in [q_seed, *seeds]:
				q0 = np.asarray(s, dtype=np.float64).copy()
				q0[4] = 0.0
				q = ik(kin, target, q0, 0.0, iters=600)
				err, down = site_err_down(cell, d, q, target)
				if err <= ik_tol and down >= cos_tol:
					q_prev = q
					break
			if q_prev is None:
				return out
		else:
			q_prev = np.asarray(q_seed, dtype=np.float64).copy()
		for roll in roll_grid:
			q0 = q_prev.copy()
			q0[4] = roll
			q = (
				ik(kin, target, q0, float(roll), iters=400)
				if vertical
				else ik_reach(kin, target, q0, float(roll), ik_tol)
			)
			err, down = site_err_down(cell, d, q, target)
			if err > ik_tol or (vertical and down < cos_tol):
				continue
			q_prev = q
			q = q.copy()
			q[5] = gripper
			d.qpos[cell.qadr] = q
			mujoco.mj_kinematics(m, d)
			mujoco.mj_camlight(m, d)
			if cell.robot_min_z(d) < min_z:
				continue
			v = capture_view(cell, "wrist_cam", data=d)
			out.append(ViewCandidate(q.copy(), "wrist_cam", v.faces_px, v.brick_px, v.visible_fraction))
		return out

	for mode, hs in (("vertical", heights), ("outside", (0.07, 0.10, 0.14))):
		cands: list[ViewCandidate] = []
		for h in hs:
			for off in offsets:
				cands += sweep(np.array([bpos[0] + off[0], bpos[1] + off[1], h]), mode == "vertical")
		chosen = greedy_face_cover(cands, SCAN_FACES, min_px, max_views)
		covered = {f for c in chosen for f in SCAN_FACES if c.faces_px[f] >= min_px}
		if covered == set(SCAN_FACES):
			chosen.sort(key=lambda c: c.q[4])
			return chosen, mode
	return [], "none"


def _inside(bbox, resolution, border_px: int) -> bool:
	if bbox is None:
		return False
	x, y, w, h = bbox
	width, height = resolution
	return x >= border_px and y >= border_px and x + w <= width - border_px and y + h <= height - border_px


def plan_bottom_view(
	cell: Cell,
	q_from: np.ndarray,
	*,
	gripper: float,
	samples: int = 600,
	clearance_m: float = 0.04,
	seed: int = 0,
	clearance: Callable[[np.ndarray], bool] | None = None,
	border_px: int = 0,
) -> ViewCandidate | None:
	"""Arm pose (held brick carried rigidly) showing the brick's bottom face largest to the top camera, among random
	joint vectors that keep arm and brick ``clearance_m`` above the table along the straight path from ``q_from``.
	With ``border_px``, the brick's image box must also keep that far from the image edges (room for an uncertain
	in-hand pose)."""
	rng = np.random.default_rng(seed)
	poser = HeldBrickPoser(cell)
	lo, hi = cell.jnt_range[:, 0], cell.jnt_range[:, 1]
	best: ViewCandidate | None = None
	for _ in range(samples):
		q = rng.uniform(lo, hi)
		q[5] = gripper
		dd = poser.at(q)
		if min(cell.robot_min_z(dd), poser.brick_min_z()) < clearance_m:
			continue
		v = capture_view(cell, "top", data=dd)
		if v.visible_fraction < 0.5 or (best is not None and v.faces_px[5] <= best.faces_px[5]):
			continue
		if border_px and not _inside(v.bbox, TOP_FULL, border_px):
			continue
		if path_min_z(cell, q_from, q, carried=True) < clearance_m * 0.5:
			continue
		if clearance is not None and not clearance(q):
			continue
		best = ViewCandidate(q.copy(), "top", v.faces_px, v.brick_px, v.visible_fraction)
	return best


@dataclass(frozen=True)
class RelocateParams:
	transfer_z_m: float = 0.08  # gripper-frame height while carrying over the table
	release_clearance_m: float = 0.004  # brick bottom above the table at release
	cartesian_carry: bool = True  # drag in a straight line at the lifted height (False: one joint-space move)
	carry_step_m: float = 0.01
	carry_speed_mps: float = 0.04
	carry_min_z_m: float = 0.03  # gripper-frame height of the straight-line carry at least this
	scan_move_s: float = 0.6
	scan_hold_s: float = 0.3
	bottom_hold_s: float = 0.6
	min_scan_px: int = 800
	do_scan: bool = True
	do_bottom: bool = True


class RelocateExpert(ClosedLoopExpert):
	"""Pick -> put down on the table at ``target_xy`` -> wrist scan there -> regrasp, show the bottom, put down.

	``on_view(stage, expert)`` is called after every inspection hold (stages ``"start_top"``, ``"scan"``,
	``"placed_top"``, ``"bottom"``). Results land in ``self.result``."""

	def __init__(
		self,
		cell: Cell,
		q_start: np.ndarray,
		target_xy: Sequence[float],
		params: ExpertParams,
		rp: RelocateParams | None = None,
		on_view: Callable[[str, ClosedLoopExpert], None] | None = None,
	) -> None:
		self.target_xy = np.asarray(target_xy, dtype=np.float64)
		self.rp = rp or RelocateParams()
		self.on_view = on_view or (lambda stage, ex: None)
		self.result: dict = {"stages": []}
		super().__init__(cell, q_start, tuple(self.target_xy), params)

	def _stage(self, name: str) -> None:
		self.result["stages"].append(name)

	def _script(self):
		p = self.p
		yield from self._hold(p.rest_s, "rest")
		self.on_view("start_top", self)
		yield from self._move(np.array([*p.ready_q, p.open_g]), p.ready_move_s, "approach")
		grasp = yield from self._grasp()
		if grasp is None:
			self.result["failed"] = f"grasp:{self.aborted}"
			yield from self._go_home()
			return
		self._stage("grasped")
		q_lift = yield from self._lift(grasp)
		ok = yield from self._put_down(self.target_xy, grasp.roll, q_lift)
		if not ok:
			yield from self._go_home()
			return
		self._stage("placed")
		self.result["placed_brick_pos"] = [float(v) for v in self.obs["brick_pos"]]
		yield from self._go_ready()
		self.on_view("placed_top", self)
		if self.rp.do_scan:
			yield from self._scan()
		if self.rp.do_bottom:
			yield from self._bottom()
		yield from self._go_home()

	def _go_ready(self):
		q = np.array([*self.p.ready_q, self.p.open_g])
		yield from self._move(q, self._carry_secs(q, self.p.ready_move_s, 1.0), "approach")

	def _put_down(self, xy: np.ndarray, roll: float, q_lift: np.ndarray):
		"""Carry the held brick over ``xy`` at transfer height, lower it to just above the table, release, settle and
		retreat. Returns False (``result["failed"]`` set) when the brick is lost."""
		p, rp = self.p, self.rp
		if rp.cartesian_carry:
			# straight line at the height the lift actually reached: IK every ``carry_step_m`` keeps the jaw
			# orientation smooth, so a far, low, tilted grasp is dragged in without twisting the wrist
			start = self.obs["tcp"].copy()
			z = max(float(start[2]), rp.carry_min_z_m)
			end = np.array([xy[0], xy[1], z])
			n = max(1, math.ceil(float(np.linalg.norm(end[:2] - start[:2])) / rp.carry_step_m))
			q = self.last.copy()
			frames_per = max(1, self._frames(rp.carry_step_m / rp.carry_speed_mps))
			for k in range(1, n + 1):
				q_next = self._ik(start + (end - start) * (k / n), q, roll)
				q_next[5] = p.hold_g
				yield from self._carry(self._move(q_next, frames_per / FPS, "place"))
				q = q_next
				if self.lost:
					break
		else:
			q = self._ik(np.array([xy[0], xy[1], rp.transfer_z_m]), q_lift, roll)
			q[5] = p.hold_g
			yield from self._carry(self._move(q, self._carry_secs(q, p.transfer_s), "place"))
		if self.lost:
			self.result["failed"] = "lost_in_transfer"
			return False
		dz = self.obs["brick_pos"][2] - self.obs["tcp"][2]
		z_tcp = rp.release_clearance_m + self.size[2] / 2.0 - dz
		q = self._ik(np.array([xy[0], xy[1], z_tcp]), q, roll)
		q[5] = p.hold_g
		yield from self._carry(self._move(q, self._carry_secs(q, p.place_descend_s), "place"))
		if not self.lost:
			yield from self._carry(self._servo(np.array([xy[0], xy[1], z_tcp]), roll, "place", p.place_servo_tol_m))
		if self.lost:
			self.result["failed"] = "lost_in_descent"
			return False
		q_open = self.last.copy()
		q_open[5] = p.open_g
		yield from self._move(q_open, p.open_ramp_s, "place")
		still = 0
		for _ in range(self._frames(p.settle_timeout_s)):
			v = float(np.linalg.norm(self.obs["brick_vel"]))
			still = still + 1 if v < p.settle_speed_mps else 0
			if still >= p.settle_frames:
				break
			yield self.last.copy(), "place"
		q_up = self._ik(self.obs["tcp"] + np.array([0.0, 0.0, p.retreat_dz_m]), self.last, roll)
		q_up[5] = p.open_g
		yield from self._move(q_up, p.retreat_s, "place")
		return True

	def _scan(self):
		p, rp = self.p, self.rp
		self.kin.data.qpos[:] = self.cell.data.qpos
		views, mode = plan_table_scan(
			self.cell, self.kin, np.array([*p.ready_q, p.open_g]), gripper=p.open_g, min_px=rp.min_scan_px
		)
		self.result["scan_mode"] = mode
		self.result["scan_planned_faces_px"] = [list(map(int, v.faces_px)) for v in views]
		for v in views:
			q = v.q.copy()
			dq = float(np.max(np.abs(q[:5] - self.last[:5])))
			yield from self._move(q, max(rp.scan_move_s, 1.875 * dq / 1.0), "approach")
			yield from self._hold_servo(q, rp.scan_hold_s, "approach")
			self.on_view("scan", self)
		if views:
			self._stage("scanned")
			yield from self._go_ready()

	def _bottom(self):
		p, rp = self.p, self.rp
		self.grasp_retries = 0
		self.block_replans = 0
		self._active_open_g = float(p.open_g)
		grasp = yield from self._grasp()
		if grasp is None:
			self.result["bottom_failed"] = f"regrasp:{self.aborted}"
			self.aborted = None
			return
		q_lift = yield from self._lift(grasp)
		best = plan_bottom_view(self.cell, self.last, gripper=p.hold_g)
		if best is None:
			self.result["bottom_failed"] = "no_pose"
		else:
			self.result["bottom_planned_px"] = int(best.faces_px[5])
			q = best.q.copy()
			q[5] = p.hold_g
			yield from self._carry(self._move(q, self._carry_secs(q, p.show_move_s), "show_move"))
			yield from self._carry(self._hold_servo(q, rp.bottom_hold_s, "show_hold"))
			if self.lost:
				self.result["bottom_failed"] = "lost"
				return
			self.on_view("bottom", self)
			yield from self._carry(self._move(q_lift, self._carry_secs(q_lift, p.show_move_s), "show_move"))
		if self.lost:
			self.result["bottom_failed"] = "lost"
			return
		self._stage("bottom_shown")
		yield from self._put_down(self.target_xy, grasp.roll, q_lift)
