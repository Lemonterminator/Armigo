"""LangGraph loop for the inspection show-pose decisions of one episode, independent of the physics.

::

    START -> evaluate --(UNKNOWN and budget left)--> build_context -> propose -> validate --(pose)--> execute
                 ^                                                      ^          |  |                  |
                 |                                                      +-(repair)-+  +-(stop/invalid)-> finish
                 +--------------------------------------------------------------------------------------+

* ``build_context`` -- the leak-checked decision context (:func:`policies.build_context`).
* ``propose`` -- a :class:`Proposer` (the LLM via :class:`policies.LLMProposer`, or a rule policy via
  :class:`RuleProposer`) returns a raw JSON answer.
* ``validate`` -- deterministic (:func:`policies.parse_choice`); an invalid answer goes back to ``propose`` with the
  error, at most ``max_repairs`` times per decision, then the loop finishes with ``invalid_output``.
* ``execute`` -- an injected :class:`PoseExecutor` shows the pose and returns the inspection record (a fake in unit
  tests). Only the job ids of finished executions enter the state, so a checkpoint never references a job that may
  not have happened.
* ``evaluate`` -- KNOWN -> ``known``; ``max_proposals`` used -> ``budget``; otherwise loop.

Relation to the in-episode ``next_pose`` callback. The MuJoCo expert is a frame generator that calls
``next_pose(history)`` synchronously and executes the pose itself, so the episode cannot be a node. The sync adapter
:class:`GraphNextPose` runs this same graph with an executor that *interrupts* (``langgraph.types.interrupt``) at
``execute``: the pose is handed back to the expert as the callback's return value, and the next ``next_pose`` call
resumes the graph with the new inspection record (``Command(resume=record)``). Hence propose/validate (and the
budgets and checkpoints) of the graph back the callback, and the expert's own ``max_extra_shows`` is the outer cap.
"""

from __future__ import annotations

import itertools
import operator
import uuid
from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Protocol, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from .llm import LLMError
from .policies import FACE_NAMES, CoverageTable, Pose, ShowPosePolicy, build_context, parse_choice

__all__ = [
	"GraphNextPose",
	"InspectionState",
	"PoseExecutor",
	"Proposer",
	"RuleProposer",
	"build_inspection_graph",
	"initial_state",
]


class Proposer(Protocol):
	def propose(self, ctx: Mapping[str, Any], repair: tuple[str, str] | None = None) -> str: ...


class PoseExecutor(Protocol):
	"""Show ``pose`` (wrist flex rad, wrist roll deg) and return the inspection record: ``flex, roll_deg, status,
	visible_faces, observed_faces`` and a ``job_id`` of the finished execution."""

	def execute(self, pose: Pose, pose_id: str) -> dict[str, Any]: ...


class RuleProposer:
	"""Wraps a rule :class:`ShowPosePolicy` as a :class:`Proposer` (it answers in the LLM's JSON format)."""

	def __init__(self, policy: ShowPosePolicy, table: CoverageTable) -> None:
		self.policy, self.table = policy, table

	def propose(self, ctx: Mapping[str, Any], repair: tuple[str, str] | None = None) -> str:
		import json

		hist = [
			{
				"flex": h["flex"],
				"roll_deg": h["roll_deg"],
				"visible_faces": [FACE_NAMES.index(n) for n in h["faces_seen_in_view"]],
				"observed_faces": [FACE_NAMES.index(n) for n in h["faces_observed_so_far"]],
			}
			for h in ctx["history"]
		]
		pose = self.policy(hist)
		if pose is None:
			return json.dumps({"stop": True, "rationale": f"{self.policy.name}: nothing to add"})
		o = self.table.find(pose)
		return json.dumps({"pose_id": o.pose_id if o else f"off_menu:{pose}", "rationale": self.policy.name})


class InspectionState(TypedDict, total=False):
	history: list[dict[str, Any]]  # inspection records (no ground truth)
	context: dict[str, Any]
	raw: str | None
	decision: dict[str, Any] | None
	error: str | None
	repairs: int  # repairs used for the current decision
	n_proposals: int
	n_invalid: int
	job_ids: list[str]  # finished executions only
	done_reason: str | None
	events: Annotated[list[dict[str, Any]], operator.add]


def initial_state(history: Sequence[Mapping[str, Any]] = ()) -> InspectionState:
	return {
		"history": [dict(h) for h in history],
		"raw": None,
		"decision": None,
		"error": None,
		"repairs": 0,
		"n_proposals": 0,
		"n_invalid": 0,
		"job_ids": [],
		"done_reason": None,
		"events": [],
	}


def build_inspection_graph(
	proposer: Proposer,
	executor: PoseExecutor,
	table: CoverageTable,
	*,
	max_proposals: int = 4,
	max_repairs: int = 1,
	checkpointer: Any = None,
	conditional: bool = False,
):
	"""Compile the loop (see module doc). ``checkpointer=None`` uses an :class:`InMemorySaver`; pass ``False`` for none."""

	def evaluate(s: InspectionState) -> dict:
		hist = s.get("history", [])
		if hist and hist[-1].get("status") == "KNOWN":
			return {"done_reason": "known"}
		if s.get("n_proposals", 0) >= max_proposals:
			return {"done_reason": "budget"}
		return {}

	def build_ctx(s: InspectionState) -> dict:
		ctx = build_context(
			s["history"], table, remaining=max_proposals - s.get("n_proposals", 0), conditional=conditional
		)
		return {"context": ctx, "raw": None, "decision": None, "error": None, "repairs": 0}

	def propose(s: InspectionState) -> dict:
		repair = (s["raw"] or "", s["error"]) if s.get("error") else None
		upd: dict[str, Any] = {} if repair else {"n_proposals": s.get("n_proposals", 0) + 1}
		try:
			raw = proposer.propose(s["context"], repair)
		except LLMError as exc:
			raw = ""
			upd["events"] = [{"node": "propose", "llm_error": str(exc)}]
		upd["raw"] = raw
		return upd

	def validate(s: InspectionState) -> dict:
		ctx = s["context"]
		dec, err = parse_choice(s.get("raw") or "", {m["pose_id"] for m in ctx["menu"]}, set(ctx["tried_pose_ids"]))
		if dec is not None:
			ev = {"node": "validate", "decision": dec, "repairs": s.get("repairs", 0)}
			upd: dict[str, Any] = {"decision": dec, "error": None, "events": [ev]}
			if dec.get("stop"):
				upd["done_reason"] = "policy_stop"
			return upd
		upd = {"error": err, "n_invalid": s.get("n_invalid", 0) + 1, "events": [{"node": "validate", "error": err}]}
		if s.get("repairs", 0) < max_repairs:
			upd["repairs"] = s.get("repairs", 0) + 1
		else:
			upd["done_reason"] = "invalid_output"
		return upd

	def execute(s: InspectionState) -> dict:
		o = table.get(s["decision"]["pose_id"])
		rec = dict(executor.execute((o.flex, o.roll_deg), o.pose_id))
		job = str(rec.get("job_id") or uuid.uuid4().hex)
		return {
			"history": [*s["history"], rec],
			"job_ids": [*s.get("job_ids", []), job],
			"events": [{"node": "execute", "pose_id": o.pose_id, "job_id": job, "status": rec.get("status")}],
		}

	def finish(s: InspectionState) -> dict:
		return {"events": [{"node": "finish", "done_reason": s.get("done_reason")}]}

	g = StateGraph(InspectionState)
	g.add_node("evaluate", evaluate)
	g.add_node("build_context", build_ctx)
	g.add_node("propose", propose)
	g.add_node("validate", validate)
	g.add_node("execute", execute)
	g.add_node("finish", finish)
	g.add_edge(START, "evaluate")
	g.add_conditional_edges(
		"evaluate", lambda s: "finish" if s.get("done_reason") else "build_context", ["finish", "build_context"]
	)
	g.add_edge("build_context", "propose")
	g.add_edge("propose", "validate")

	def after_validate(s: InspectionState) -> str:
		if s.get("done_reason"):
			return "finish"
		return "execute" if s.get("decision") else "propose"

	g.add_conditional_edges("validate", after_validate, ["finish", "execute", "propose"])
	g.add_edge("execute", "evaluate")
	g.add_edge("finish", END)
	return g.compile(checkpointer=InMemorySaver() if checkpointer is None else (checkpointer or None))


class _InterruptExecutor:
	"""Hands the pose to whoever drives the graph and waits for the inspection record (``Command(resume=...)``)."""

	def execute(self, pose: Pose, pose_id: str) -> dict[str, Any]:
		return interrupt({"pose": list(pose), "pose_id": pose_id})


class GraphNextPose:
	"""Sync adapter: a ``NextPoseFn`` backed by the graph (one instance and one checkpoint thread per episode)."""

	_ids = itertools.count()

	def __init__(
		self,
		proposer: Proposer,
		table: CoverageTable,
		*,
		max_proposals: int = 4,
		max_repairs: int = 1,
		conditional: bool = False,
		name: str = "",
	) -> None:
		self.name = name or "graph"
		self.graph = build_inspection_graph(
			proposer,
			_InterruptExecutor(),
			table,
			max_proposals=max_proposals,
			max_repairs=max_repairs,
			conditional=conditional,
		)
		self.config = {"configurable": {"thread_id": f"episode-{next(self._ids)}"}, "recursion_limit": 200}
		self._started = False
		self._pending = False

	def _pose_from(self, out: Mapping[str, Any]) -> Pose | None:
		intr = out.get("__interrupt__") if isinstance(out, Mapping) else None
		if not intr:
			self._pending = False
			return None
		self._pending = True
		v = intr[0].value
		return float(v["pose"][0]), float(v["pose"][1])

	@staticmethod
	def _record(h: Mapping[str, Any]) -> dict[str, Any]:
		rec = {k: h[k] for k in ("flex", "roll_deg", "status", "visible_faces", "observed_faces") if k in h}
		rec["job_id"] = f"frame-{h.get('frame', 'na')}"
		return rec

	def __call__(self, history: Sequence[Mapping]) -> Pose | None:
		if not self._started:
			self._started = True
			hist = [self._record(h) for h in history]
			return self._pose_from(self.graph.invoke(initial_state(hist), self.config))
		if not self._pending:
			return None
		return self._pose_from(self.graph.invoke(Command(resume=self._record(history[-1])), self.config))

	def close(self, history: Sequence[Mapping]) -> None:
		"""Feed the last inspection record after the episode so the graph reaches ``finish`` (for its log)."""
		if self._pending and history:
			self._pose_from(self.graph.invoke(Command(resume=self._record(history[-1])), self.config))

	def state(self) -> dict[str, Any]:
		return dict(self.graph.get_state(self.config).values)
