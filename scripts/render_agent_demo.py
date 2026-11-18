"""Record six-face agent episodes for an Isaac RTX demo video.

Each case runs once through the LangGraph graph (default policy) with ``TwinMujoco``, in the chosen sensing mode,
while :class:`~digital_twin_armigo.catalog.capture.MujocoCapture` records every geometry pose per simulated frame.
A caption per frame names the skill, the last scored view and, at the end, the label. The recording is written in
the catalog's scene/motion format, together with a ``request.json`` for ``scripts/isaac/catalog_render.py``:

	.venv/Scripts/python scripts/render_agent_demo.py --out results/demos/realistic 11 1 0
	E:/il/v/Scripts/python.exe scripts/isaac/catalog_render.py --request results/demos/realistic/case-0011/request.json \\
	  --out results/demos/realistic/case-0011/render

Recording only reads the simulation; the episode is the same as in ``armigo-mcp run``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

NODE_TEXT = {
	"prepare": "moving to the ready pose",
	"scan_wrist": "scan_wrist: reading faces 0-4 with the wrist camera",
	"present_bottom": "present_bottom: showing face 5 to the top camera",
	"choose_view": "choosing the next view",
	"recover": "recovering",
	"place": "place",
	"evaluate": "done",
}


def record(case, sensing, out: Path, scorer):
	from digital_twin_armigo.agent.twin_graph import Policy, build_graph
	from digital_twin_armigo.agent.twin_mujoco import TwinMujoco, sample_case
	from digital_twin_armigo.agent.twin_record import EpisodeTrace
	from digital_twin_armigo.catalog.capture import MujocoCapture
	from digital_twin_armigo.catalog.store import plain
	from digital_twin_armigo.constants import PHASES

	out.mkdir(parents=True, exist_ok=False)
	trace = EpisodeTrace(out / "trace", video=False)
	captions: list[str] = []
	state = {"seen": 0, "last": "", "label": None}
	box: dict = {}

	def on_frame(backend):
		rec = box.get("recorder")
		if rec is None:
			rec = box["recorder"] = MujocoCapture(backend.cell)
		for event in trace.events[state["seen"] :]:
			if event["kind"] == "view":
				o = event["observation"]
				verdict = "CROSS" if o["score"] is not None and o["score"] >= o["threshold"] else "clear"
				faces = ",".join(map(str, o["faces"])) or "-"
				score = "n/a" if o["score"] is None else f"{o['score']:+.2f}"
				state["last"] = f"last view: {o['camera']} faces [{faces}] score {score} -> {verdict}"
		state["seen"] = len(trace.events)
		node = backend.node
		text = NODE_TEXT.get(node, node)
		if node == "place" and state["label"]:
			bowl = {"defective": "pink bowl", "normal": "green bowl"}.get(state["label"], "table")
			text = f"place: {state['label']} -> {bowl}"
		caption = f"[{sensing}] {text}" + (f"  |  {state['last']}" if state["last"] else "")
		if rec.times and backend.cell.data.time <= rec.times[-1]:
			# Skill boundaries call on_frame without stepping: keep one frame per simulated time.
			captions[-1] = caption
			return
		captions.append(caption)
		rec.frame(backend.cell)

	backend = TwinMujoco(sample_case(*case), scorer, trace, sensing=sensing, on_frame=on_frame)
	original_place = backend.place

	def place(label):
		state["label"] = label
		return original_place(label)

	backend.place = place
	try:
		result = build_graph(backend, Policy()).invoke({}, config={"recursion_limit": 64})["result"]
		trace.finish(result, {"engine": "mujoco", "sensing": sensing, "case": list(case)})
		rec = box["recorder"]
		if len(captions) != len(rec.frames):
			raise RuntimeError("captions and frames are out of step")
		phases = np.asarray(backend.phases[: len(rec.frames)], dtype=np.int16)
		if len(phases) < len(rec.frames):
			phases = np.concatenate([phases, np.full(len(rec.frames) - len(phases), phases[-1] if len(phases) else 0)])
		truth = "defective" if case[1] else "normal"
		alias = f"case {case[0]} ({truth}, defect face {case[2]})" if case[1] else f"case {case[0]} (normal)"
		manifest = rec.save(
			out / "recording",
			{"phases": phases},
			plain(result),
			{"sensing": sensing, "case": list(case), "phase_names": list(PHASES)},
			alias,
			"mujoco",
		)
		(out / "recording" / "captions.json").write_text(json.dumps(captions), encoding="utf-8")
		verdict = f"{result['verdict']}: labelled {result['label']}"
		request = {
			"episode": {
				"alias": alias,
				"verdict": verdict,
				"engine": "mujoco",
				"artifacts": [
					{"role": role, "path": f"recording/{name}"}
					for role, name in (
						("scene", "scene.json"),
						("motion", "motion.npz"),
						("telemetry", "telemetry.npz"),
						("captions", "captions.json"),
					)
				],
			},
			"profile": "default",
		}
		(out / "request.json").write_text(json.dumps(request, indent=1), encoding="utf-8")
		return {
			"case": list(case),
			"verdict": result["verdict"],
			"label": result["label"],
			"frames": len(captions),
		}, manifest
	finally:
		backend.close()


def main():
	p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	p.add_argument("indices", nargs="+", type=int, help="Case indices in --suite.")
	p.add_argument("--suite", type=Path, default=ROOT / "config/agent_bench/bench16.json")
	p.add_argument("--sensing", choices=("truth", "realistic"), default="realistic")
	p.add_argument("--detector", type=Path, default=ROOT / "results/agent-bench/sup-v7bot-local.json")
	p.add_argument("--out", type=Path, required=True)
	a = p.parse_args()
	from digital_twin_armigo.inspection_scorer import default_sup_scorer

	cases = json.loads(a.suite.read_text(encoding="utf-8"))
	scorer = default_sup_scorer(a.detector)
	try:
		for i in a.indices:
			summary, _ = record(tuple(cases[i]), a.sensing, a.out / f"case-{i:04d}", scorer)
			print(json.dumps(summary), flush=True)
	finally:
		if hasattr(scorer, "close"):
			scorer.close()


if __name__ == "__main__":
	main()
