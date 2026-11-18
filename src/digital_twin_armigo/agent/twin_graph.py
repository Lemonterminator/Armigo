"""Six-face inspection and bounded strategy search over trusted simulator skills."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
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


def suggest_policy(llm: ChatLLM, reports: list[dict]) -> Policy:
	"""Propose a reusable, validated policy without per-brick labels or simulator access."""
	response = llm.chat(
		[
			{
				"role": "system",
				"content": (
					"Choose a generic six-face robot inspection policy. Optimize successful sorts first, then simulation "
					"time and view count, using only development aggregates. Propose a new candidate. Do not repeat any measured policy. Return one JSON object with exactly these "
					"fields: order (wrist_first or bottom_first), max_bottom_views (1ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Å“3), max_wrist_views (1ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Å“3), "
					"max_rewinds (0ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Å“2). The detector, coverage requirements and destinations are fixed."
				),
			},
			{"role": "user", "content": json.dumps({"task": TASK, "development_results": reports})},
		]
	)
	return Policy(**json.loads(response.content))


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


class SearchState(TypedDict):
	source: str
	reports: list[dict]
	pending: Policy
	remaining: int


def search_policies(llm: ChatLLM, evaluate, budget: int) -> dict:
	"""Evaluate bounded proposals on one fixed development set and rank successes before cost."""
	from langgraph.graph import END, START, StateGraph

	if not 1 <= budget <= 8:
		raise ValueError("budget must be in [1, 8]")

	def measure(state: SearchState):
		policy = state["pending"]
		return {
			"reports": [*state["reports"], {"policy": asdict(policy), "source": state["source"], **evaluate(policy)}],
			"remaining": state["remaining"] - 1,
		}

	def propose(state: SearchState):
		try:
			next_policy = suggest_policy(llm, state["reports"])
			source = "llm"
		except (TypeError, ValueError, LLMError):
			next_policy = Policy(order="bottom_first")
			source = "fallback_invalid_proposal"
		seen = [report["policy"] for report in state["reports"]]
		if asdict(next_policy) in seen:
			source = "fallback_duplicate_proposal"
			next_policy = next(
				(
					candidate
					for candidate in (
						Policy(order="bottom_first"),
						Policy(max_bottom_views=3),
						Policy(max_wrist_views=3),
					)
					if asdict(candidate) not in seen
				),
				None,
			)
			if next_policy is None:
				return {"remaining": 0}
		return {"pending": next_policy, "source": source}

	graph = StateGraph(SearchState)
	graph.add_node("measure", measure)
	graph.add_node("propose", propose)
	graph.add_edge(START, "measure")
	graph.add_conditional_edges("measure", lambda s: "propose" if s["remaining"] else END, ["propose", END])
	graph.add_conditional_edges("propose", lambda s: "measure" if s["remaining"] else END, ["measure", END])
	result = graph.compile().invoke({"pending": Policy(), "reports": [], "remaining": budget, "source": "baseline"})

	def rank(report):
		# Compare total cost only after success and label correctness on the same development set.
		return (-report["successes"], -report["labels_correct"], report["sim_frames"], report["views"])

	return {"selected": min(result["reports"], key=rank), "candidates": result["reports"]}
