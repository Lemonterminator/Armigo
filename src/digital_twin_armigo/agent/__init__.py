"""Stage-3 inspection agent: show-pose policies (rule baselines and an LLM over a typed pose menu) and a LangGraph
decision loop.

The LLM never writes code or free-form joint targets: it picks a ``pose_id`` from a menu built from measured data,
every answer is validated deterministically (one repair attempt), budgets are bounded, and everything sent to the
model passes :func:`digital_twin_armigo.inspection.assert_no_defect_leakage`.
"""

from .llm import ChatResult, FakeLLM, LLMError, OpenAICompatLLM, server_reachable
from .policies import (
	FACE_NAMES,
	CoveragePolicy,
	CoverageTable,
	FixedSequencePolicy,
	LLMProposer,
	LLMShowPosePolicy,
	PoseOption,
	ShowPosePolicy,
	build_context,
	parse_choice,
	render_messages,
)

__all__ = [
	"FACE_NAMES",
	"ChatResult",
	"CoveragePolicy",
	"CoverageTable",
	"FakeLLM",
	"FixedSequencePolicy",
	"LLMError",
	"LLMProposer",
	"LLMShowPosePolicy",
	"OpenAICompatLLM",
	"PoseOption",
	"ShowPosePolicy",
	"build_context",
	"parse_choice",
	"render_messages",
	"server_reachable",
]
