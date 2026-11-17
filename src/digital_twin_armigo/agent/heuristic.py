"""Pure heuristic decision logic for brick inspection (no simulator, MuJoCo or torch).

Faces are indexed ``0 +x``, ``1 -x``, ``2 +y``, ``3 -y``, ``4 top``, ``5 bottom``; the bottom face never carries the
cross in this task, hence :data:`ELIGIBLE_FACES`. A view's :class:`ViewEvidence` gives the geometric visible pixels
of every face (known from geometry, never from the label) and the anomaly score of one camera with its threshold.

The inspector only consumes the evidence passed to :meth:`HeuristicInspector.observe`: it reads no global state and
uses no randomness. See :class:`HeuristicInspector` for the exact decision rules.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

__all__ = ["ELIGIBLE_FACES", "FACES", "HeuristicInspector", "ViewEvidence"]

FACES = 6
ELIGIBLE_FACES = (0, 1, 2, 3, 4)


@dataclass(frozen=True)
class ViewEvidence:
	"""One camera view of the brick as seen by the inspector.

	``faces_px`` holds the geometric visible pixel count of each face (index 0..5 = +x, -x, +y, -y, top, bottom),
	known from geometry and never from the label. ``score`` is the anomaly-detector output (higher means more
	anomalous) and ``threshold`` the detection threshold of that camera. ``score is None`` means the brick was not in
	view / there was no crop, so the view can neither detect nor clear any face. For ``present`` views
	``target_face`` is the face the arm tried to show to the top camera.
	"""

	stage: str
	camera: str
	faces_px: tuple[int, ...]
	score: float | None
	threshold: float
	target_face: int | None = None

	@property
	def detected(self) -> bool:
		"""True when there is a score and it reaches this camera's threshold."""
		return self.score is not None and self.score >= self.threshold


class HeuristicInspector:
	"""Accumulate view evidence and decide the label, grasp and next presentation.

	Semantics (all state is derived from the observed evidence only):

	* ``detected``: at least one observed view detected.
	* ``top_confirmed``: at least one ``top`` view detected.
	* ``cleared_faces``: eligible faces with ``>= min_face_px[camera]`` pixels in at least ``min_clear_views``
	  (default 1) NON-detecting views (more views for a weak detector: one miss is not proof of absence). A view
	  with ``score is None`` neither detects nor clears.
	* ``suspect_faces``: for every detecting view, the eligible faces with ``>= min_face_px[camera]`` pixels (or,
	  if that set is empty, the eligible faces with ``px > 0``). Intersect these sets over detecting views; when the
	  intersection is empty use their union. Then remove cleared faces unless that would leave the set empty. Empty
	  when nothing was detected.
	* ``presented_faces``: ``target_face`` of ``present`` views, in observation order.
	* ``pinched_faces``: set by :meth:`set_grasp`; the jaws hide these faces.

	:meth:`label` returns ``"defective"`` when a detection is accepted (with ``require_top_confirmation`` that means
	a top-camera detection, or a wrist-only detection once no useful presentation is left), ``"normal"`` when nothing
	was detected and every eligible face was cleared, and ``None`` (unknown) otherwise. :meth:`choose_grasp` ranks
	candidates by avoiding suspects (when detected) or by covering already-cleared faces (when not detected), then by
	descending score and ascending index. :meth:`next_present_face` picks the next face for the top camera, skipping
	presented, pinched or cleared faces and respecting the presentation budget.
	"""

	def __init__(
		self,
		*,
		eligible_faces: Iterable[int] = ELIGIBLE_FACES,
		min_face_px: Mapping[str, int] = {"wrist_cam": 800, "top": 250},
		max_presents: int = 4,
		require_top_confirmation: bool = False,
		min_clear_views: int = 1,
	) -> None:
		self._eligible = tuple(sorted({int(f) for f in eligible_faces}))
		self._min_face_px = dict(min_face_px)
		self._max_presents = int(max_presents)
		self._require_top_confirmation = bool(require_top_confirmation)
		self._detected = False
		self._top_confirmed = False
		self._cleared: set[int] = set()
		self._min_clear_views = max(1, int(min_clear_views))
		self._clear_counts: dict[int, int] = {}
		self._detecting_faces: list[set[int]] = []
		self._presented: list[int] = []
		self._pinched: set[int] = set()
		self._n_views = 0

	# --- evidence ---------------------------------------------------------------------------------------------------

	def observe(self, ev: ViewEvidence) -> None:
		"""Record one view.

		A scored view either detects (and contributes its visible face set to the suspects) or clears its eligible
		faces with ``>= min_face_px[camera]`` pixels. ``score is None`` views are only counted and, for ``present``
		views, still record their ``target_face`` (the arm spent that presentation).
		"""
		self._n_views += 1
		if ev.stage == "present" and ev.target_face is not None:
			self._presented.append(int(ev.target_face))
		if ev.detected:
			self._detected = True
			if ev.camera == "top":
				self._top_confirmed = True
			self._detecting_faces.append(self._visible_faces(ev))
		elif ev.score is not None:
			for f in self._faces_at_least_min(ev):
				self._clear_counts[f] = self._clear_counts.get(f, 0) + 1
				if self._clear_counts[f] >= self._min_clear_views:
					self._cleared.add(f)

	def _faces_at_least_min(self, ev: ViewEvidence) -> set[int]:
		min_px = self._min_face_px[ev.camera]
		return {f for f in self._eligible if 0 <= f < len(ev.faces_px) and ev.faces_px[f] >= min_px}

	def _visible_faces(self, ev: ViewEvidence) -> set[int]:
		faces = self._faces_at_least_min(ev)
		if faces:
			return faces
		return {f for f in self._eligible if 0 <= f < len(ev.faces_px) and ev.faces_px[f] > 0}

	# --- derived state ----------------------------------------------------------------------------------------------

	@property
	def detected(self) -> bool:
		"""At least one observed view detected."""
		return self._detected

	@property
	def top_confirmed(self) -> bool:
		"""At least one ``top`` view detected."""
		return self._top_confirmed

	@property
	def cleared_faces(self) -> frozenset[int]:
		"""Eligible faces seen clearly in some non-detecting, scored view."""
		return frozenset(self._cleared)

	@property
	def suspect_faces(self) -> frozenset[int]:
		"""Faces visible in the detecting views (intersection, else union), minus cleared unless that empties it."""
		if not self._detecting_faces:
			return frozenset()
		intersection = set(self._detecting_faces[0])
		union = set(self._detecting_faces[0])
		for faces in self._detecting_faces[1:]:
			intersection &= faces
			union |= faces
		suspect = intersection or union
		reduced = suspect - self._cleared
		return frozenset(reduced or suspect)

	@property
	def presented_faces(self) -> tuple[int, ...]:
		"""``target_face`` of the ``present`` views, in order."""
		return tuple(self._presented)

	@property
	def pinched_faces(self) -> frozenset[int]:
		"""Faces hidden by the current grasp's jaws."""
		return frozenset(self._pinched)

	def _all_cleared(self) -> bool:
		return all(f in self._cleared for f in self._eligible)

	# --- decisions --------------------------------------------------------------------------------------------------

	def label(self) -> str | None:
		"""``"defective"``, ``"normal"`` or ``None`` (unknown).

		Defective requires a detection; with ``require_top_confirmation`` a wrist-only detection stays unknown until
		the top camera confirms it, unless the presentation budget is spent or no useful presentation is left (then a
		wrist detection alone is enough). Normal requires no detection and every eligible face cleared.
		"""
		if self._detected and (not self._require_top_confirmation or self._top_confirmed):
			return "defective"
		if not self._detected and self._all_cleared():
			return "normal"
		if (
			self._detected
			and self._require_top_confirmation
			and not self._top_confirmed
			and (len(self._presented) >= self._max_presents or self._next_present_face() is None)
		):
			return "defective"
		return None

	def choose_grasp(self, candidates: Sequence[Mapping]) -> int:
		"""Index of the grasp candidate to use (higher ``score`` is better; the list may be unsorted).

		When detected, candidates whose ``pinched_faces`` do not intersect :attr:`suspect_faces` come first; when not
		detected, candidates with both pinched faces already cleared come first, then those with exactly one cleared
		pinched face, then the rest. Ties go to the highest ``score``, then the lowest index. Raises ``ValueError`` on
		an empty candidate list.
		"""
		if not candidates:
			raise ValueError("choose_grasp requires at least one candidate")

		def key(item: tuple[int, Mapping]) -> tuple[int, float, int]:
			index, candidate = item
			pinched = tuple(int(f) for f in candidate["pinched_faces"])
			if self._detected:
				tier = 1 if any(f in self.suspect_faces for f in pinched) else 0
			elif all(f in self._cleared for f in pinched):
				tier = 0
			elif sum(f in self._cleared for f in pinched) == 1:
				tier = 1
			else:
				tier = 2
			return tier, -float(candidate["score"]), index

		return min(enumerate(candidates), key=key)[0]

	def set_grasp(self, pinched: Iterable[int]) -> None:
		"""Remember the faces pinched by the jaws; they are never presented."""
		self._pinched = {int(f) for f in pinched}

	def _next_present_face(self) -> int | None:
		"""Face selection without the label/budget guards of :meth:`next_present_face`."""
		if self._detected and self._require_top_confirmation and not self._top_confirmed:
			for face in sorted(self.suspect_faces):
				if face not in self._presented and face not in self._pinched:
					return face
			return None
		if not self._detected:
			order = ([4] if 4 in self._eligible else []) + [f for f in self._eligible if f != 4]
			for face in order:
				if face not in self._cleared and face not in self._presented and face not in self._pinched:
					return face
		return None

	def next_present_face(self) -> int | None:
		"""Face to show the top camera next, or ``None`` when nothing useful or affordable is left.

		With a wrist detection awaiting top confirmation, suspects are shown in ascending order. Otherwise uncovered
		eligible faces are shown top (4) first, then 0, 1, 2, 3. Presented, pinched and cleared faces are skipped. The
		budget is ``max_presents`` presented views.
		"""
		if len(self._presented) >= self._max_presents:
			return None
		if self.label() is not None and not (
			self._detected and self._require_top_confirmation and not self._top_confirmed
		):
			return None
		return self._next_present_face()

	# --- reporting --------------------------------------------------------------------------------------------------

	def _reason(self) -> str:
		lab = self.label()
		if lab == "defective":
			if self._detected and self._require_top_confirmation and not self._top_confirmed:
				return "wrist_only"
			return "defect_seen"
		if lab == "normal":
			return "all_faces_cleared"
		if not self._detected:
			return f"uncovered_faces:{sorted(f for f in self._eligible if f not in self._cleared)}"
		return "unknown"

	def summary(self) -> dict:
		"""JSON-serialisable snapshot of the derived state and the decision reason."""
		return {
			"detected": self.detected,
			"top_confirmed": self.top_confirmed,
			"cleared": sorted(self.cleared_faces),
			"suspect": sorted(self.suspect_faces),
			"presented": list(self.presented_faces),
			"pinched": sorted(self.pinched_faces),
			"label": self.label(),
			"n_views": self._n_views,
			"reason": self._reason(),
		}
