"""Closed-loop grasp descent: re-localise the brick with the wrist camera at checkpoints between hover and pre-grasp.

``sync`` settles at every checkpoint and waits for the result; ``async`` looks on arrival without settling, keeps
descending towards the old target, and turns to the replanned one ``latency_s`` later. The brick lies still on the table,
so a world-frame estimate does not go stale during the latency."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..constants import FPS

__all__ = ["VARIANTS", "GraspLoop"]

VARIANTS: dict[str, tuple[tuple, str] | None] = {
	"open": None,
	"w1s": (("hover",), "sync"),
	"w2s": (("hover", 0.035), "sync"),
	"w3s": (("hover", 0.045, 0.030), "sync"),
	"w2a": (("hover", 0.035), "async"),
	"w3a": (("hover", 0.045, 0.030), "async"),
	"w4a": (("hover", 0.050, 0.040, 0.030), "async"),
}


@dataclass
class GraspLoop:
	heights: tuple
	mode: str
	look: Callable
	settle_s: float = 0.15
	latency_s: float = 0.10
	segment_s: float = 0.6
	fps: int = FPS

	def _heights(self, ex):
		hover = ex.p.hover_z_m - ex.z_grasp
		return [hover if h == "hover" else float(h) for h in self.heights]

	def _goal(self, ex, target, h, roll, q):
		goal = ex._ik(np.asarray(target) + np.array([0.0, 0.0, h]), q, roll)
		goal[5] = ex._active_open_g
		return goal

	def descend(self, ex, phase, target, roll, q):
		if self.mode == "sync":
			for h in self._heights(ex):
				goal = self._goal(ex, target, h, roll, ex.last)
				yield from ex._move(goal, self.segment_s, phase)
				yield from ex._hold_servo(goal, self.settle_s, phase)
				replan = self.look(h)
				if replan is not None:
					target, roll, _ = replan
			return target, roll, ex.last.copy()
		lat = max(1, round(self.latency_s * self.fps))
		pending = None  # [frames until usable, replan or None]
		stops = [*self._heights(ex), ex.p.pregrasp_dz_m]
		for i, h in enumerate(stops):
			while True:
				turned = False
				for frame in ex._move(self._goal(ex, target, h, roll, ex.last), self.segment_s, phase):
					yield frame
					if pending is not None:
						pending[0] -= 1
						if pending[0] == 0:
							replan, pending = pending[1], None
							if replan is not None:
								target, roll, _ = replan
								turned = True
								break
				if not turned:
					break
			if i < len(stops) - 1:
				pending = [lat, self.look(h)]
		if pending is not None and pending[1] is not None:
			target, roll, _ = pending[1]
		return target, roll, ex.last.copy()
