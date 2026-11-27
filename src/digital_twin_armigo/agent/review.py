"""Re-view of detector alarms (plan 2026-11-24, section 5.3): strategies B (fixed rule) and C (LLM looks at images).

Strategy A keeps :func:`.twin_graph.decision`: the first scored view at or above its threshold labels the brick
defective. Under B and C an alarm is held as *pending* and checked from another angle:

- an alarm is **confirmed** when a second alarming view, taken from a different arm pose, shares one of its faces;
- an alarm is **refuted** when every face of the alarming view has at least two clear scored views and at least one
  of those is a re-view taken for this alarm;
- otherwise it stays **unresolved**. While the re-view budget lasts the agent takes another re-view; when the
  budget is spent with an unresolved alarm the label is ``unknown``.

A brick is ``normal`` only when all six faces are cleared and no alarm is unresolved or confirmed.

B and C share the trigger, the confirmation rule, the reachable candidates and the budget; only the chooser of the
next view differs. The LLM (C) returns a candidate id; the label always comes from the frozen detector and this rule.
"""

from __future__ import annotations

import base64
import io
import json
import math
import time

import numpy as np

# Two arm poses closer than this (L2 over the six joints) count as the same view: a re-view must move.
MIN_POSE_SEPARATION_RAD = 0.2
MAX_CANDIDATES = 6
MAX_LLM_IMAGES = 5


def _alarm(view) -> bool:
	return view.get("score") is not None and bool(view.get("faces")) and view["score"] >= view["threshold"]


def _clear(view) -> bool:
	return view.get("score") is not None and bool(view.get("faces")) and view["score"] < view["threshold"]


def _pose_distance(a, b) -> float:
	if a is None or b is None:
		return math.inf
	return float(np.linalg.norm(np.asarray(a, dtype=np.float64)[:6] - np.asarray(b, dtype=np.float64)[:6]))


def review_decision(views: list[dict]) -> tuple[str, set[int], list[dict]]:
	"""``(label, covered faces, pending alarms)`` under strategies B and C.

	``pending`` lists unresolved alarms oldest first: ``{"alarm": view id, "camera", "faces": faces still needing a
	clear view, "q": arm pose of the alarm}``."""
	scored = [v for v in views if v.get("score") is not None and v.get("faces")]
	alarms = [v for v in scored if _alarm(v)]
	confirmed = False
	pending = []
	for a in alarms:
		others = [
			b
			for b in alarms
			if b is not a
			and set(b["faces"]) & set(a["faces"])
			and _pose_distance(a.get("q"), b.get("q")) >= MIN_POSE_SEPARATION_RAD
		]
		if others:
			confirmed = True
			continue
		clears = [v for v in scored if _clear(v)]
		reviews = [v for v in clears if v.get("review_of") == a["id"]]
		missing = [f for f in a["faces"] if sum(f in v["faces"] for v in clears) < 2]
		if missing or not reviews:
			need = missing or list(a["faces"])
			pending.append({"alarm": a["id"], "camera": a["camera"], "faces": sorted(need), "q": a.get("q")})
	covered = set()
	for v in scored:
		if _clear(v):
			covered.update(v["faces"])
	if confirmed:
		return "defective", covered, []
	covered -= {f for p in pending for f in p["faces"]}
	if pending:
		return "unknown", covered, pending
	return ("normal" if len(covered) == 6 else "unknown"), covered, []


def rule_choice(candidates: list[dict]) -> str:
	"""Strategy B: most predicted pixels of the target faces (``priority_px``: capped per face and weighted to the
	faces with the fewest clear views; ``target_px`` when absent), then the larger pose change, then the shorter path."""
	best = max(
		candidates,
		key=lambda c: (
			round(c.get("priority_px", c["target_px"]) / 400.0),
			round(min(c["separation_rad"], 1.5), 1),
			-c["path_rad"],
		),
	)
	return best["id"]


def _png_b64(image: np.ndarray, max_side: int = 448) -> str:
	from PIL import Image

	img = Image.fromarray(np.asarray(image, dtype=np.uint8))
	scale = max_side / max(img.size)
	if scale < 1.0:
		img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))))
	buf = io.BytesIO()
	img.save(buf, format="PNG")
	return base64.b64encode(buf.getvalue()).decode("ascii")


REVIEW_SYSTEM = (
	"You help a sorting robot decide where to look next. A detector raised an alarm on a brick view; the alarm must "
	"be checked from another angle. Each request shows one camera image; reply with JSON "
	'{"note": "<at most 20 words on what helps choosing a view: occlusion, glare, blur, which brick sides are '
	'visible>"}. Finally you get the reachable candidate views and reply with JSON {"candidate": "<id>"} choosing '
	"the view most likely to show the alarmed area clearly from a new angle. You never decide the label."
)


def llm_choice(llm, images: list[dict], candidates: list[dict], context: dict) -> tuple[str, str, dict]:
	"""Strategy C: show ``images`` (``{"name", "image", "cached"}``) one per request, then ask for a candidate id.

	Returns ``(candidate id, source, info)``; ``source`` is ``"llm"`` or ``"fallback"`` (rule choice, with
	``info["reason"]``)."""
	ids = [c["id"] for c in candidates]
	info = {"reason": None, "requests": 0, "latency_s": 0.0, "prompt_tokens": 0, "cached_tokens": 0, "notes": []}
	messages = [{"role": "system", "content": REVIEW_SYSTEM}]
	note_schema = {
		"type": "object",
		"properties": {"note": {"type": "string"}},
		"required": ["note"],
		"additionalProperties": False,
	}
	choice_schema = {
		"type": "object",
		"properties": {"candidate": {"type": "string", "enum": ids}},
		"required": ["candidate"],
		"additionalProperties": False,
	}
	structured = getattr(llm, "structured", False)

	def ask(content, schema):
		# One image per request (the Armigo vLLM server allows one): every request is the shared system prefix
		# (prefix-cached) plus one user turn; the final choice request carries the per-image notes as text.
		request = [messages[0], {"role": "user", "content": content}]
		started = time.perf_counter()
		response = llm.chat(request, json_schema=schema) if structured else llm.chat(request)
		info["requests"] += 1
		info["latency_s"] = round(info["latency_s"] + (response.latency_s or (time.perf_counter() - started)), 4)
		info["prompt_tokens"] += int(response.prompt_tokens or 0)
		info["cached_tokens"] += int(getattr(response, "cached_tokens", 0) or 0)
		return response.content or ""

	try:
		for item in images[:MAX_LLM_IMAGES]:
			text = ask(
				[
					{"type": "text", "text": f"Image: {item['name']}"},
					{"type": "image_url", "image_url": {"url": "data:image/png;base64," + _png_b64(item["image"])}},
				],
				note_schema,
			)
			try:
				info["notes"].append(str(json.loads(text).get("note", ""))[:200])
			except (ValueError, AttributeError):
				info["notes"].append(text[:200])
		notes = [{"image": item["name"], "note": note} for item, note in zip(images, info["notes"], strict=False)]
		payload = {"context": context, "image_notes": notes, "candidates": candidates}
		text = ask(json.dumps(payload, separators=(",", ":"), allow_nan=False), choice_schema)
	except Exception as exc:  # noqa: BLE001 - any transport or model failure falls back to the rule
		info["reason"] = getattr(exc, "kind", type(exc).__name__)
		return rule_choice(candidates), "fallback", info
	try:
		data = json.loads(text[text.find("{") : text.rfind("}") + 1])
		choice = data.get("candidate") if isinstance(data, dict) else None
	except ValueError:
		choice = None
	if choice in ids:
		return choice, "llm", info
	info["reason"] = "illegal_choice" if choice is not None else "invalid_json"
	return rule_choice(candidates), "fallback", info
