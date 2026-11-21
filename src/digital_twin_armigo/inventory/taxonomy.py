"""Who produced a synthetic trajectory: the five-way provenance category and how it is decided.

``category`` answers "what actually drove the arm" and "what did it get to see":

=====================  ==========================================================================================
``scripted``           a fixed script (physics expert, expert tracer, heuristic): no decision loop, privileged state
``agent_omniscient``   the state-machine agent (rule or LLM) driven by ground truth (``sensing="truth"``)
``agent_realistic``    the state-machine agent driven by what a real cell senses (cameras, joints, effort)
``model``              a learned policy (ACT / SmolVLA ...) produced every action
``model_takeover``     a learned policy, until a safety gate handed control to the scripted expert (DAgger)
``agent_unknown``      an agent episode whose sensing mode was never recorded and cannot be inferred
=====================  ==========================================================================================

The rules read only fields the writers already record, so they can run over old data. ``EpisodeTrace.finish`` freezes the
decision into ``provenance["actor"]`` at write time (``basis = "explicit"``); everything else is ``"inferred"`` and an
operator override is ``"override"``.
"""

from __future__ import annotations

RULES_VERSION = "1"

CATEGORIES: dict[str, str] = {
	"scripted": "纯脚本",
	"agent_omniscient": "全知 agent",
	"agent_realistic": "实时感知 agent",
	"model": "纯模型输出",
	"model_takeover": "模型 + 专家接管",
	"agent_unknown": "agent（感知模式未知）",
}

# Realistic sensing (b218d60) and the ``sensing`` provenance field (2229ebf) came later than the first agent runs: an
# episode written before the former can only have been run with ground truth, which was the only mode then.
SENSING_MODE_INTRODUCED = 1_795_037_041.0  # 2026-11-18T23:24:01+02:00 as a Unix timestamp


def classify_provenance(provenance: dict, result: dict | None = None, *, written_at: float | None = None) -> dict:
	"""Category of one agent episode from its ``provenance`` (and ``result``), as ``{category, control, sensing, ...}``.

	``written_at`` (file time) is only used for episodes without a ``sensing`` field."""
	explicit = provenance.get("actor")
	if isinstance(explicit, dict) and explicit.get("category") in CATEGORIES:
		return {**explicit, "basis": "explicit"}
	result = result or {}
	sensing = provenance.get("sensing")
	note = None
	basis = "inferred"
	if sensing not in ("truth", "realistic"):
		if written_at is not None and written_at < SENSING_MODE_INTRODUCED:
			sensing, note = "truth", "no sensing field; written before realistic sensing existed, so ground truth"
		else:
			sensing, note = "unknown", "no sensing field and written after realistic sensing was introduced"
	if "student" in provenance:  # student_rollout: a learned policy, or (student false) the expert it imitates
		if not provenance["student"]:
			return _out("scripted", "script_expert_tracer", sensing, basis, note)
		taken = bool(provenance.get("dagger_takeover"))
		return _out("model_takeover" if taken else "model", "model_policy", sensing, basis, note)
	llm = (result.get("llm") or {}).get("consulted") or 0
	control = "agent_llm" if llm else "agent_rule"
	if sensing == "truth":
		return _out("agent_omniscient", control, sensing, basis, note)
	if sensing == "realistic":
		return _out("agent_realistic", control, sensing, basis, note)
	return _out("agent_unknown", control, sensing, basis, note)


def classify_catalog(engine: str) -> dict:
	"""Catalog records are imports of physics-expert / Isaac stage-2 reports and recordings: scripted, privileged state."""
	return _out("scripted", f"script_physics_expert_{engine}", "privileged", "inferred", "catalog import")


def _out(category: str, control: str, sensing: str, basis: str, note: str | None) -> dict:
	return {
		"category": category,
		"control": control,
		"sensing": sensing,
		"basis": basis,
		"note": note,
		"rules_version": RULES_VERSION,
	}
