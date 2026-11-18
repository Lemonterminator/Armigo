"""Six-face inspection and bounded strategy search over trusted simulator skills."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal, TypedDict

from .llm import ChatLLM, LLMError

TASK = "Inspect the brick for a cross. Sort defective bricks into the pink bowl and normal bricks into the green bowl."
NODES = ("prepare", "choose_view", "scan_wrist", "present_bottom", "recover", "place", "evaluate")
EDGES = (
	("prepare", "recover"),
	("prepare", "choose_view"),
	("choose_view", "scan_wrist"),
	("choose_view", "present_bottom"),
	("choose_view", "place"),
	("scan_wrist", "choose_view"),
	("present_bottom", "choose_view"),
	("scan_wrist", "recover"),
	("present_bottom", "recover"),
	("place", "recover"),
	("recover", "choose_view"),
	("recover", "evaluate"),
	("place", "evaluate"),
)


@dataclass(frozen=True)
class Policy:
	order: Literal["wrist_first", "bottom_first"] = "wrist_first"
	max_bottom_views: int = 2
	max_wrist_views: int = 2
	max_rewinds: int = 1

	def __post_init__(self):
		if self.order not in ("wrist_first", "bottom_first"):
			raise ValueError("invalid inspection order")
		for name, lo, hi in (("max_bottom_views", 1, 3), ("max_wrist_views", 1, 3), ("max_rewinds", 0, 2)):
			value = getattr(self, name)
			if type(value) is not int or not lo <= value <= hi:
				raise ValueError(f"{name} must be an integer in [{lo}, {hi}]")


class InspectionState(TypedDict, total=False):
	views: list[dict]
	failures: list[dict]
	action: str
	error: str | None
	label: str
	result: dict


def decision(views: list[dict]) -> tuple[str, set[int]]:
	"""Determine a brick label from scored views and physical face visibility."""
	covered: set[int] = set()
	for view in views:
		if view["score"] is None or not view["faces"]:
			continue
		if view["score"] >= view["threshold"]:
			return "defective", covered
		covered.update(view["faces"])
	return ("normal" if len(covered) == 6 else "unknown"), covered


def build_graph(backend, policy: Policy):
	from langgraph.graph import END, START, StateGraph

	def prepare(state: InspectionState):
		return {"views": [], "failures": [], "error": backend.prepare()}

	def choose_view(state: InspectionState):
		views = state["views"]
		label, covered = decision(views)
		if label != "unknown":
			return {"label": label, "action": "place"}
		wrist = bool(set(range(5)) - covered) and backend.can_scan_wrist(policy)
		bottom = 5 not in covered and backend.can_present_bottom(policy)
		order = ("scan_wrist", "present_bottom") if policy.order == "wrist_first" else ("present_bottom", "scan_wrist")
		allowed = [name for name in order if (wrist if name == "scan_wrist" else bottom)]
		if not allowed:
			return {"label": "unknown", "action": "place"}
		return {"label": "unknown", "action": backend.choose(allowed, views, state["failures"])}

	def observe(node: str):
		def run(state: InspectionState):
			views, error = backend.inspect(node, state["views"])
			return {"views": [*state["views"], *views], "error": error}

		return run

	def recover(state: InspectionState):
		failure = backend.failure(state["error"], state["views"])
		failures = [*state["failures"], failure]
		restored = backend.rewind(policy, failures)
		if restored is None:
			return {"action": "evaluate", "label": state.get("label", "unknown"), "failures": failures}
		return {"action": "choose_view", "views": restored, "error": None, "label": "unknown", "failures": failures}

	def place(state: InspectionState):
		return {"error": backend.place(state["label"])}

	def evaluate(state: InspectionState):
		return {"result": backend.evaluate(state["label"], state["views"], state["failures"])}

	graph = StateGraph(InspectionState)
	for name, operation in (
		("prepare", prepare),
		("choose_view", choose_view),
		("scan_wrist", observe("scan_wrist")),
		("present_bottom", observe("present_bottom")),
		("recover", recover),
		("place", place),
		("evaluate", evaluate),
	):

		def node(state, name=name, operation=operation):
			backend.enter(name)
			return operation(state)

		graph.add_node(name, node)
	graph.add_edge(START, "prepare")
	graph.add_conditional_edges(
		"prepare",
		lambda state: "recover" if state["error"] else "choose_view",
		["recover", "choose_view"],
	)
	graph.add_conditional_edges("choose_view", lambda s: s["action"], ["scan_wrist", "present_bottom", "place"])
	for name in ("scan_wrist", "present_bottom", "place"):
		next_node = "evaluate" if name == "place" else "choose_view"
		graph.add_conditional_edges(
			name,
			lambda s, target=next_node: "recover" if s["error"] else target,
			["recover", next_node],
		)
	graph.add_conditional_edges("recover", lambda s: s["action"], ["choose_view", "evaluate"])
	graph.add_edge("evaluate", END)
	return graph.compile()


def select_skill(llm: ChatLLM, choices: list[str], distinct_context: dict) -> tuple[str, str]:
	try:
		response = llm.chat(
			[
				{
					"role": "system",
					"content": (
						"Select a legal inspection or recovery skill for a sorting robot. Return a JSON object with one "
						"field, skill, set to exactly one of choices. The detector and motion controller own safety and "
						"classification. Choose the shortest evidence-gathering path."
					),
				},
				{
					"role": "user",
					"content": json.dumps(
						{"choices": choices, **distinct_context}, allow_nan=False, separators=(",", ":")
					),
				},
			]
		)
		name = json.loads(response.content)["skill"]
		if name in choices:
			return name, "llm"
	except (KeyError, TypeError, ValueError, LLMError):
		pass
	return choices[0], "fallback"


def search_policies(evaluate_batch, cap: int, **options) -> dict:
	"""Exhaustive successive halving with paired statistics; see ``twin_search``."""
	from .twin_search import successive_halving

	return successive_halving(evaluate_batch, cap, **options)
