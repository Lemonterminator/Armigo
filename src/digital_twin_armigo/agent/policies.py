"""Show-pose policies for the inspection phase: ``(history) -> (wrist flex rad, wrist roll deg) | None``.

All policies match :data:`digital_twin_armigo.mujoco_cell.physics_expert.NextPoseFn`: the expert asks them for an
extra show pose while the inspection status is UNKNOWN after its base poses, up to ``ExpertParams.max_extra_shows``
times. History records carry ``physics_expert.POLICY_HISTORY_KEYS`` (frame, flex, roll_deg, status, visible_faces,
observed_faces) and no ground truth. Nothing here reads ``label`` or ``reason`` even if present (an inspector may word
its UNKNOWN reason per class); the LLM only ever sees the copy built by :func:`sanitize_history`, whose missing-face
list is derived from ``observed_faces``.

* :class:`FixedSequencePolicy` -- a hand-picked list (baseline, equivalent to static extra poses).
* :class:`CoveragePolicy` -- greedy: the untried menu pose with the largest expected number of newly covered faces,
  from a :class:`CoverageTable` measured on the pose grid of ``scripts/inspection_visibility.py``.
* :class:`LLMShowPosePolicy` -- an LLM picks a ``pose_id`` from the same menu; output is strict JSON, validated
  deterministically (:func:`parse_choice`) with one repair attempt; on a second failure it stops (returns None).
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ..inspection import DefectLeakageError, assert_no_defect_leakage
from .llm import ChatLLM, LLMError

__all__ = [
	"FACE_NAMES",
	"CoveragePolicy",
	"CoverageTable",
	"FixedSequencePolicy",
	"LLMProposer",
	"LLMShowPosePolicy",
	"PoseOption",
	"ShowPosePolicy",
	"assert_prompt_leak_free",
	"build_context",
	"parse_choice",
	"render_messages",
	"sanitize_history",
]

Pose = tuple[float, float]
FACE_NAMES = ("+x", "-x", "+y", "-y", "top", "bottom")  # presentation.FACES order
CAMERAS = ("top", "wrist_cam")
N_BASE_POSES = 3  # ExpertParams.show_roll_deg

# Words that would name or hint at the class of the held brick (or the bowl it goes to). The rendered prompt must not
# contain any of them; the task text describes the check without naming its outcomes.
_PROMPT_LEAK_RE = re.compile(
	r"\b(defect\w*|pink|green|normal|orange|label\w*|bowl\w*|faulty|broken|damage\w*)\b", re.IGNORECASE
)


@runtime_checkable
class ShowPosePolicy(Protocol):
	"""A ``NextPoseFn`` with a name."""

	name: str

	def __call__(self, history: Sequence[Mapping]) -> Pose | None: ...


def pose_id(flex: float, roll_deg: float) -> str:
	return f"f{flex:+.2f}_r{round(roll_deg):+04d}"


def _same_pose(a: Pose, b: Pose, flex_tol: float = 1e-3, roll_tol_deg: float = 0.5) -> bool:
	return abs(a[0] - b[0]) <= flex_tol and abs(a[1] - b[1]) <= roll_tol_deg


def observed_faces(history: Sequence[Mapping]) -> set[int]:
	out: set[int] = set()
	for h in history:
		out.update(int(f) for f in h.get("observed_faces", ()))
	return out


@dataclass(frozen=True)
class PoseOption:
	"""One menu pose: wrist flex (rad), wrist roll (deg), P(face shown with >= min_face_px in either camera)."""

	pose_id: str
	flex: float
	roll_deg: float
	face_prob: tuple[float, ...]

	def gain(self, observed: set[int]) -> float:
		return sum(p for f, p in enumerate(self.face_prob) if f not in observed)


@dataclass
class CoverageTable:
	"""Pose menu with expected face coverage, measured on the show-pose grid of ``inspection_visibility.py``."""

	options: tuple[PoseOption, ...]
	min_face_px: int = 60
	min_clearance_m: float = 0.02
	n_layouts: int = 0
	source: str = ""
	# per measured layout: pose_id -> faces shown (>= min_face_px, either camera) at that pose, for the base show poses
	# (from the run's episodes.jsonl) and the menu poses; drives :meth:`face_probs` conditioned on an episode's views
	evidence: tuple[dict[str, tuple[int, ...]], ...] = ()

	@classmethod
	def from_pose_grid(
		cls,
		path: str | Path,
		*,
		min_face_px: int = 60,
		min_clearance_m: float = 0.02,
		cameras: Sequence[str] = CAMERAS,
		episodes_path: str | Path | None = "auto",
	) -> CoverageTable:
		"""For each grid pose: the fraction of layouts in which each face got ``>= min_face_px`` in either camera;
		only poses whose clearance is ``>= min_clearance_m`` in every layout are kept. With ``episodes_path`` (default:
		``episodes.jsonl`` next to ``path`` if present) the per-layout views are kept as :attr:`evidence`."""
		path = Path(path)
		rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
		layouts = {r["layout_index"] for r in rows}
		by_pose: dict[Pose, list[dict]] = defaultdict(list)
		for r in rows:
			by_pose[(float(r["flex"]), float(r["roll_deg"]))].append(r)
		opts = []
		for (flex, roll), rs in sorted(by_pose.items()):
			if len({r["layout_index"] for r in rs}) < len(layouts):
				continue
			if any(r["clearance_m"] < min_clearance_m or any(c not in r for c in cameras) for r in rs):
				continue
			prob = []
			for f in range(len(FACE_NAMES)):
				hits = sum(any(r[c]["faces_px"][f] >= min_face_px for c in cameras) for r in rs)
				prob.append(round(hits / len(rs), 3))
			opts.append(PoseOption(pose_id(flex, roll), flex, roll, tuple(prob)))
		if episodes_path == "auto":
			episodes_path = path.with_name("episodes.jsonl")
			episodes_path = episodes_path if episodes_path.exists() else None
		evidence: tuple[dict[str, tuple[int, ...]], ...] = ()
		if episodes_path is not None:
			menu = {o.pose_id for o in opts}

			def shown(v: Mapping) -> tuple[int, ...]:
				return tuple(
					f for f in range(len(FACE_NAMES)) if any(v[c]["faces_px"][f] >= min_face_px for c in cameras)
				)

			ev: dict[int, dict[str, tuple[int, ...]]] = {k: {} for k in sorted(layouts)}
			for r in rows:
				pid = pose_id(float(r["flex"]), float(r["roll_deg"]))
				if pid in menu:
					ev[r["layout_index"]][pid] = shown(r)
			for line in Path(episodes_path).read_text(encoding="utf-8").splitlines():
				e = json.loads(line) if line.strip() else None
				if e is None or e["layout_index"] not in ev:
					continue
				for v in e.get("views", []):
					ev[e["layout_index"]].setdefault(pose_id(float(v["flex"]), float(v["roll_deg"])), shown(v))
			evidence = tuple(ev.values())
		return cls(tuple(opts), min_face_px, min_clearance_m, len(layouts), str(path), evidence)

	def to_dict(self) -> dict[str, Any]:
		d = asdict(self)
		d["face_names"] = list(FACE_NAMES)
		return d

	@classmethod
	def from_dict(cls, d: Mapping[str, Any]) -> CoverageTable:
		opts = tuple(
			PoseOption(o["pose_id"], float(o["flex"]), float(o["roll_deg"]), tuple(float(p) for p in o["face_prob"]))
			for o in d["options"]
		)
		evidence = tuple({k: tuple(int(f) for f in v) for k, v in e.items()} for e in d.get("evidence", ()))
		return cls(
			opts, int(d["min_face_px"]), float(d["min_clearance_m"]), int(d["n_layouts"]), str(d["source"]), evidence
		)

	def save(self, path: str | Path) -> None:
		Path(path).write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")

	@classmethod
	def load(cls, path: str | Path) -> CoverageTable:
		return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

	def get(self, pid: str) -> PoseOption | None:
		return next((o for o in self.options if o.pose_id == pid), None)

	def find(self, pose: Pose) -> PoseOption | None:
		return next((o for o in self.options if _same_pose((o.flex, o.roll_deg), pose)), None)

	def face_probs(
		self, history: Sequence[Mapping] = (), *, conditional: bool = True, beta: float = 1.0
	) -> dict[str, tuple[float, ...]]:
		"""P(face shown) per menu pose. Marginal over the measured layouts, or (``conditional`` and evidence present)
		weighted per layout by ``exp(-beta * sum_h |visible_faces_h xor faces the layout showed at pose_h|)`` over the
		history poses the layout has a view at: layouts whose views look like this episode's (similar grasp) count."""
		if not (conditional and self.evidence and history):
			return {o.pose_id: o.face_prob for o in self.options}
		logw = []
		for ev in self.evidence:
			d = 0
			for h in history:
				seen = ev.get(pose_id(float(h["flex"]), float(h["roll_deg"])))
				if seen is not None:
					d += len(set(seen) ^ {int(f) for f in h.get("visible_faces", ())})
			logw.append(-beta * d)
		m = max(logw)
		w = [math.exp(x - m) for x in logw]
		tot = sum(w)
		return {
			o.pose_id: tuple(
				round(sum(wi for wi, ev in zip(w, self.evidence, strict=True) if f in ev.get(o.pose_id, ())) / tot, 3)
				for f in range(len(FACE_NAMES))
			)
			for o in self.options
		}

	def tried_ids(self, history: Sequence[Mapping]) -> set[str]:
		out = set()
		for h in history:
			o = self.find((float(h["flex"]), float(h["roll_deg"])))
			if o is not None:
				out.add(o.pose_id)
		return out


# --------------------------------------------------------------------------------------------------- rule policies


@dataclass
class FixedSequencePolicy:
	"""The first pose of ``poses`` not shown yet; None when all were shown."""

	poses: Sequence[Pose]
	name: str = "fixed"

	def __call__(self, history: Sequence[Mapping]) -> Pose | None:
		done = [(float(h["flex"]), float(h["roll_deg"])) for h in history]
		for p in self.poses:
			if not any(_same_pose(p, d) for d in done):
				return float(p[0]), float(p[1])
		return None


@dataclass
class CoveragePolicy:
	"""Greedy: the untried pose maximising the expected number of newly covered faces; None below ``min_gain``."""

	table: CoverageTable
	min_gain: float = 0.05
	conditional: bool = False  # condition the face probabilities on this episode's views (CoverageTable.face_probs)
	name: str = "coverage"

	def choose(self, history: Sequence[Mapping]) -> PoseOption | None:
		obs = observed_faces(history)
		tried = self.table.tried_ids(history)
		probs = self.table.face_probs(history, conditional=self.conditional)
		best, best_gain = None, self.min_gain
		for o in self.table.options:
			if o.pose_id in tried:
				continue
			g = sum(p for f, p in enumerate(probs[o.pose_id]) if f not in obs)
			if g > best_gain + 1e-9:
				best, best_gain = o, g
		return best

	def __call__(self, history: Sequence[Mapping]) -> Pose | None:
		o = self.choose(history)
		return None if o is None else (o.flex, o.roll_deg)


# ------------------------------------------------------------------------------------------------- LLM context/IO

TASK_TEXT = (
	"You control the wrist of a robot arm. The gripper holds a small rectangular brick and shows it to two fixed "
	"cameras (an overhead camera and a camera on the wrist). An automatic visual check accumulates which of the "
	"brick's six faces (+x, -x, +y, -y, top, bottom, in the brick's own frame) have been seen clearly. Its status "
	"stays UNKNOWN while it still lacks evidence and becomes KNOWN once it can reach a verdict; seeing every face "
	"always suffices. Choose the next show pose from the menu so that faces not observed yet become visible, so the "
	"check can finish within the remaining pose budget. Each menu entry gives the wrist flex (rad), the wrist roll "
	"(deg) and, per face, the estimated probability that the pose shows the face clearly, from past trials "
	"(possibly weighted towards trials whose views resembled this episode's so far; values below 0.05 are omitted). What a pose shows varies with how the brick sits in the gripper, so use this episode's history "
	"too. Never choose a pose already tried. Stop if no menu pose is likely to show an unobserved face."
)
ANSWER_FORMAT = (
	'Answer with ONLY a JSON object and nothing else: {"pose_id": "<an id from the menu>", "rationale": "<one '
	'short sentence>"} or {"stop": true, "rationale": "<one short sentence>"}.'
)


def sanitize_history(history: Sequence[Mapping], table: CoverageTable | None = None) -> list[dict[str, Any]]:
	"""The leak-free view of the inspection records: only pose, status and faces; never ``label`` or ``reason``."""
	out = []
	for i, h in enumerate(history):
		known = h.get("status") != "UNKNOWN"
		o = table.find((float(h["flex"]), float(h["roll_deg"]))) if table is not None else None
		out.append(
			{
				"step": i + 1,
				"pose_id": o.pose_id if o is not None else ("base" if i < N_BASE_POSES else "off_menu"),
				"flex": round(float(h["flex"]), 3),
				"roll_deg": round(float(h["roll_deg"]), 1),
				"status": "KNOWN" if known else "UNKNOWN",
				"faces_seen_in_view": [FACE_NAMES[int(f)] for f in h.get("visible_faces", ())],
				"faces_observed_so_far": [FACE_NAMES[int(f)] for f in h.get("observed_faces", ())],
				"faces_not_observed_yet": [
					FACE_NAMES[f]
					for f in range(len(FACE_NAMES))
					if f not in {int(x) for x in h.get("observed_faces", ())}
				],
			}
		)
	return out


def build_context(
	history: Sequence[Mapping], table: CoverageTable, *, remaining: int, conditional: bool = False
) -> dict[str, Any]:
	"""The structured, leak-checked decision context (task, untried menu, sanitised history, budget); with
	``conditional`` the menu probabilities are conditioned on this episode's views (:meth:`CoverageTable.face_probs`)."""
	obs = observed_faces(history)
	tried = table.tried_ids(history)
	probs = table.face_probs(history, conditional=conditional)
	menu = [
		{
			"pose_id": o.pose_id,
			"flex": round(o.flex, 3),
			"roll_deg": round(o.roll_deg, 1),
			"p_face": {FACE_NAMES[f]: p for f, p in enumerate(probs[o.pose_id]) if p >= 0.05},
		}
		for o in table.options
		if o.pose_id not in tried
	]
	ctx = {
		"task": TASK_TEXT,
		"faces": list(FACE_NAMES),
		"faces_observed": [FACE_NAMES[f] for f in sorted(obs)],
		"faces_not_observed": [FACE_NAMES[f] for f in range(len(FACE_NAMES)) if f not in obs],
		"remaining_pose_budget": int(remaining),
		"tried_pose_ids": sorted(tried),
		"history": sanitize_history(history, table),
		"menu": menu,
	}
	assert_no_defect_leakage(ctx)
	return ctx


def render_messages(ctx: Mapping[str, Any]) -> list[dict[str, str]]:
	menu_lines = [
		f"{m['pose_id']}: flex={m['flex']:+.2f} roll={m['roll_deg']:+.0f} "
		+ " ".join(f"{k}={v:.2f}" for k, v in m["p_face"].items())
		for m in ctx["menu"]
	]
	hist_lines = [json.dumps(h, separators=(",", ":")) for h in ctx["history"]]
	user = "\n".join(
		[
			f"Faces observed so far: {ctx['faces_observed']}",
			f"Faces not observed yet: {ctx['faces_not_observed']}",
			f"Remaining pose budget: {ctx['remaining_pose_budget']}",
			f"Pose ids already tried (not allowed): {ctx['tried_pose_ids']}",
			"History (one show pose per line):",
			*hist_lines,
			"Menu (pose_id: flex, roll, P(face shown)):",
			*menu_lines,
			ANSWER_FORMAT,
		]
	)
	msgs = [{"role": "system", "content": ctx["task"]}, {"role": "user", "content": user}]
	assert_prompt_leak_free(msgs)
	return msgs


def assert_prompt_leak_free(messages: Sequence[Mapping[str, str]]) -> None:
	"""Raise :class:`DefectLeakageError` if a rendered message names a class or a bowl."""
	for m in messages:
		assert_no_defect_leakage({"task": m["content"]})
		hit = _PROMPT_LEAK_RE.search(m["content"])
		if hit:
			raise DefectLeakageError(f"prompt word {hit.group(0)!r} may reveal the brick class")


def _extract_json(text: str) -> Any:
	t = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
	t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t).strip()
	try:
		return json.loads(t)
	except ValueError:
		pass
	start, end = t.find("{"), t.rfind("}")
	if start < 0 or end <= start:
		raise ValueError("no JSON object found")
	return json.loads(t[start : end + 1])


def parse_choice(text: str, menu_ids: set[str], tried_ids: set[str]) -> tuple[dict[str, Any] | None, str | None]:
	"""Deterministic validation of one answer -> (decision, None) or (None, error).

	A decision is ``{"stop": True, ...}`` or ``{"pose_id": <id in the menu, not tried>, "rationale": str}``."""
	try:
		obj = _extract_json(text)
	except ValueError as exc:
		return None, f"not valid JSON ({exc})"
	if not isinstance(obj, dict):
		return None, "the answer must be a JSON object"
	rationale = str(obj.get("rationale", ""))[:500]
	if obj.get("stop") is True:
		return {"stop": True, "rationale": rationale}, None
	pid = obj.get("pose_id")
	if not isinstance(pid, str):
		return None, 'missing "pose_id" (or {"stop": true})'
	pid = pid.strip()
	if pid in tried_ids:
		return None, f"pose_id {pid!r} was already tried"
	if pid not in menu_ids:
		return None, f"pose_id {pid!r} is not in the menu"
	return {"pose_id": pid, "rationale": rationale}, None


@dataclass
class LLMProposer:
	"""Asks the LLM for a pose choice; keeps one log record (latency, tokens, raw exchange) per call."""

	client: ChatLLM
	log: list[dict[str, Any]] = field(default_factory=list)

	def propose(self, ctx: Mapping[str, Any], repair: tuple[str, str] | None = None) -> str:
		"""Raw answer text; ``repair = (previous answer, validation error)`` asks for a corrected answer."""
		msgs = render_messages(ctx)
		if repair is not None:
			prev, err = repair
			msgs += [
				{"role": "assistant", "content": prev},
				{"role": "user", "content": f"Invalid answer: {err}. {ANSWER_FORMAT}"},
			]
			assert_prompt_leak_free(msgs[-1:])
		rec: dict[str, Any] = {"t": time.time(), "repair": repair is not None, "messages": msgs}
		try:
			res = self.client.chat(msgs)
		except LLMError as exc:
			rec.update(error=str(exc), raw=None, latency_s=None, prompt_tokens=0, completion_tokens=0)
			self.log.append(rec)
			raise
		rec.update(
			raw=res.content,
			latency_s=res.latency_s,
			prompt_tokens=res.prompt_tokens,
			completion_tokens=res.completion_tokens,
			reasoning_tokens=res.reasoning_tokens,
			finish_reason=res.finish_reason,
		)
		self.log.append(rec)
		return res.content


@dataclass
class LLMShowPosePolicy:
	"""LLM picks the next pose from the :class:`CoverageTable` menu (see module doc); None stops showing."""

	client: ChatLLM
	table: CoverageTable
	budget: int = 4  # ExpertParams.max_extra_shows
	max_repairs: int = 1
	conditional: bool = False
	name: str = "llm"
	log: list[dict[str, Any]] = field(default_factory=list)  # one record per LLM call
	decisions: list[dict[str, Any]] = field(default_factory=list)  # one record per __call__
	failures: list[dict[str, Any]] = field(default_factory=list)

	def __post_init__(self) -> None:
		self._proposer = LLMProposer(self.client, self.log)

	def __call__(self, history: Sequence[Mapping]) -> Pose | None:
		used = max(0, len(history) - N_BASE_POSES)
		ctx = build_context(history, self.table, remaining=self.budget - used, conditional=self.conditional)
		menu_ids = {m["pose_id"] for m in ctx["menu"]}
		tried = set(ctx["tried_pose_ids"])
		repair = None
		errors: list[str] = []
		for _ in range(self.max_repairs + 1):
			try:
				raw = self._proposer.propose(ctx, repair)
			except LLMError as exc:
				errors.append(f"llm_error: {exc}")
				break
			dec, err = parse_choice(raw, menu_ids, tried)
			if dec is not None:
				self.decisions.append({"step": len(history), "decision": dec, "repairs": len(errors)})
				if dec.get("stop"):
					return None
				o = self.table.get(dec["pose_id"])
				return o.flex, o.roll_deg
			errors.append(err)
			repair = (raw, err)
		self.failures.append({"step": len(history), "errors": errors})
		self.decisions.append({"step": len(history), "decision": None, "repairs": len(errors), "errors": errors})
		return None
