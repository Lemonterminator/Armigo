"""Wrist-camera ring over the brick: brick localisation from several wrist views at different wrist rolls.

With the jaws straight down over the believed brick, the wrist roll swings the wrist camera (about 57 mm off the jaw
axis, looking inwards and down) around the brick, so its views see the top face from every side. Each view is
localised with :func:`~.wrist_localize.localize_from_wrist` (camera pose from the joint readings and the *designed*
mount). The wrist mount error moves each fix by a vector that turns with the roll, so the median over a ring cancels
most of it: offline (``scripts/vla/wrist_orbit_localise.py``, 43 v3 drift cases) the fused xy error was 0.6 mm
median against 14.9 mm for the top view; a single view was about 3 mm.
"""

from __future__ import annotations

import math

import numpy as np

from ..mujoco_cell._mj import mujoco

RING_TIP_M = 0.04  # jaw tips above the believed top face (30-60 mm localise alike; 40 mm aims the axis nearest)
RING_VIEWS = 8
RING_ENTRY_DZ_M = 0.06  # the first ring pose is entered from this much higher, straight down
RING_GATE_M = 0.04  # a fused fix farther than this from the prior is rejected (as FuseCfg.gate_xy_m)
RING_MIN_FIXES = 3


def _axis_hit(cell, kin_data, top_z: float) -> np.ndarray:
	"""Where the believed wrist optical axis meets the plane z = ``top_z`` (x, y)."""
	from .wrist_localize import believed_wrist_from_cam

	w2c = believed_wrist_from_cam(cell, kin_data)
	o, a = w2c[:3, 3], w2c[:3, 2]
	t = (top_z - o[2]) / a[2] if abs(a[2]) > 1e-6 else 0.0
	return (o + t * a)[:2]


def ring_poses(
	ex, cell, centre_xy, top_z: float, tip: float = RING_TIP_M, n: int = RING_VIEWS, aim: bool = False
) -> list[np.ndarray]:
	"""Jaws-down joint poses (IK in ``ex.kin``) over ``centre_xy`` with the tips ``tip`` above ``top_z``, at ``n``
	wrist rolls spread over the roll range (one turn at most), in roll order. ``aim`` shifts the jaw axis so the
	believed optical axis meets ``centre_xy``. Poses the IK cannot reach (jaws off vertical, height off, roll
	clipped) are left out."""
	lo, hi = (float(v) for v in cell.jnt_range[4])
	span = min(hi - lo, 2 * math.pi * (n - 1) / n)
	mid = 0.5 * (lo + hi)
	centre_xy = np.asarray(centre_xy, dtype=np.float64)
	q_seed = cell.joints()
	out = []
	for roll in np.linspace(mid - span / 2, mid + span / 2, n):
		xy = centre_xy.copy()
		q = None
		for _ in range(4 if aim else 1):
			q = ex._ik(np.array([*xy, top_z + tip]), q_seed if q is None else q, float(roll))
			if aim:
				ex.kin.data.qpos[ex.kin.qadr] = q
				xy = xy - (_axis_hit(cell, ex.kin.data, top_z) - centre_xy)
		ex.kin.data.qpos[ex.kin.qadr] = q
		mujoco.mj_kinematics(ex.kin.model, ex.kin.data)
		site = ex.kin.data.site_xpos[ex.kin.site]
		down = float(-ex.kin.data.site_xmat[ex.kin.site].reshape(3, 3)[2, 0])
		if down < math.cos(math.radians(8)) or abs(site[2] - (top_z + tip)) > 0.006 or abs(float(q[4]) - roll) > 0.05:
			continue
		out.append(q)
	return out


def fuse_ring(fixes) -> np.ndarray:
	"""Fused brick centre (x, y) of the ring's fixes: the per-axis median (robust to a stray view)."""
	return np.median(np.array([f.pos_xy for f in fixes]), axis=0)
