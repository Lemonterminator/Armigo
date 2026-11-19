"""Compare wrist scans between six-face agent runs of the same suite: face exposure, views, frames, outcomes.

    python results/orbit-scan/compare.py NAME=RUN_DIR [NAME=RUN_DIR ...]

Episodes are matched by fixture (seed, class, face). For each episode, the best pixel count of every side and top
face over the wrist views of the active branch is taken; "weakest side" is the smallest of the four side faces.
Writes ``summary.json`` next to this script.
"""

import json
import statistics
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent / "summary.json"


def load(run: Path) -> dict:
	out = {}
	for task in sorted(run.rglob("task.json")):
		ep = task.parent
		if not (ep / "episode.json").is_file():
			continue
		case = tuple(json.loads(task.read_text(encoding="utf-8"))["case"])
		result = json.loads((ep / "episode.json").read_text(encoding="utf-8"))["result"]
		active = result["active_branch"]
		wrist, modes, alarms = [], [], 0
		with (ep / "events.jsonl").open(encoding="utf-8") as f:
			for e in map(json.loads, f):
				if e["kind"] == "scan_plan" and e["branch"] == active:
					modes.append(e["mode"])
				if e["kind"] == "view" and e["observation"]["camera"] == "wrist_cam":
					o = e["observation"]
					alarms += o["score"] is not None and o["score"] >= o["threshold"]
					if o["branch"] == active:
						wrist.append(o["face_pixels"])
		best = [max((w[f] for w in wrist), default=0) for f in range(5)]
		out[case] = {
			"verdict": result["verdict"],
			"label": result["label"],
			"rewinds": result["rewinds"],
			"sim_frames": result["sim_frames"],
			"wrist_views": len(wrist),
			"best_face_px": best,
			"weakest_side_px": min(best[:4]),
			"modes": modes,
			"wrist_alarms": alarms,
			"failures": [x.get("code") for x in result.get("failures", [])],
		}
	return out


def describe(name: str, rows: dict) -> dict:
	vals = list(rows.values())
	if not vals:
		return {"name": name, "episodes": 0}
	weak = [r["weakest_side_px"] for r in vals]
	return {
		"name": name,
		"episodes": len(vals),
		"success": sum(r["verdict"] == "SUCCESS" for r in vals),
		"rewinds": sum(r["rewinds"] for r in vals),
		"failure_codes": sorted({c for r in vals for c in r["failures"]}),
		"wrist_alarms": sum(r["wrist_alarms"] for r in vals),
		"wrist_views_mean": statistics.mean(r["wrist_views"] for r in vals),
		"sim_frames_total": sum(r["sim_frames"] for r in vals),
		"weakest_side_px_median": statistics.median(weak),
		"weakest_side_px_min": min(weak),
		"episodes_weakest_side_below_1500": sum(w < 1500 for w in weak),
		"top_px_median": statistics.median(r["best_face_px"][4] for r in vals),
		"modes": {m: sum(m in r["modes"] for r in vals) for m in sorted({m for r in vals for m in r["modes"]})},
	}


def main():
	runs = {}
	for arg in sys.argv[1:]:
		name, path = arg.split("=", 1)
		runs[name] = load(Path(path))
	common = set.intersection(*(set(r) for r in runs.values()))
	report = {"common_episodes": len(common), "runs": []}
	for name, rows in runs.items():
		report["runs"].append(describe(name, {k: v for k, v in rows.items() if k in common}))
	names = list(runs)
	if len(names) == 2:
		a, b = (runs[n] for n in names)
		pairs = [(a[k]["weakest_side_px"], b[k]["weakest_side_px"]) for k in sorted(common)]
		report["paired"] = {
			"weakest_side_ratio_median": statistics.median(y / max(x, 1) for x, y in pairs),
			f"{names[1]}_higher": sum(y > x for x, y in pairs),
			"verdict_changes": [
				{"case": list(k), names[0]: a[k]["verdict"], names[1]: b[k]["verdict"]}
				for k in sorted(common)
				if a[k]["verdict"] != b[k]["verdict"]
			],
		}
	OUT.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
	print(json.dumps(report, indent=1))


if __name__ == "__main__":
	main()
