"""Bottom-face (face 5) detection from six-face agent runs: per-episode outcome and every bottom-view score.

    python results/agent-bench/bottom_views.py data/catalog/agent/bottom48 [more run folders]

Prints a summary and writes ``summary/bottom_views.json`` (or ``--out=FILE``, given first). A run folder holds ``case-*`` episode folders (``run``) or
``episodes/<id>`` folders (the MCP service); both are searched.
"""

import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent / "summary" / "bottom_views.json"


def episodes(root: Path):
	for task in sorted(root.rglob("task.json")):
		if (task.parent / "episode.json").is_file():
			yield task.parent


def main():
	args = sys.argv[1:]
	out = OUT
	if args and args[0].startswith("--out="):
		out = Path(args.pop(0).removeprefix("--out="))
	rows = []
	for root in map(Path, args):
		for ep in episodes(root):
			seed, defective, face = json.loads((ep / "task.json").read_text(encoding="utf-8"))["case"]
			result = json.loads((ep / "episode.json").read_text(encoding="utf-8"))["result"]
			views = []
			with (ep / "events.jsonl").open(encoding="utf-8") as f:
				for event in map(json.loads, f):
					if event["kind"] == "view":
						o = event["observation"]
						views.append(
							{
								"id": o["id"],
								"camera": o["camera"],
								"faces": o["faces"],
								"score": o["score"],
								"threshold": o["threshold"],
								"face5_px": o["face_pixels"][5],
							}
						)
			rows.append(
				{
					"run": str(root),
					"episode": ep.name,
					"seed": seed,
					"defective": defective,
					"face": face,
					"verdict": result["verdict"],
					"label": result["label"],
					"failures": [f.get("code") for f in result.get("failures", [])],
					"evidence": result.get("evidence", []),
					"views": views,
				}
			)
	out.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")

	def bottom(row, active_only=True):
		ids = set(row["evidence"])
		return [v for v in row["views"] if v["camera"] == "top" and (not active_only or v["id"] in ids)]

	for cls in (True, False):
		group = [r for r in rows if r["defective"] == cls]
		name = "defective, face 5" if cls else "normal"
		print(f"\n{name}: {len(group)} episodes")
		labels = {}
		for r in group:
			labels[r["label"]] = labels.get(r["label"], 0) + 1
		print("  labels", labels, "| verdicts", {v: sum(r["verdict"] == v for r in group) for v in ("SUCCESS", "FAILED", "UNKNOWN")})
		wrist_alarm = sum(
			any(v["camera"] == "wrist_cam" and v["score"] is not None and v["score"] >= v["threshold"] for v in r["views"])
			for r in group
		)
		print("  episodes with a wrist alarm", wrist_alarm)
		scores = sorted(v["score"] for r in group for v in bottom(r, False) if v["score"] is not None)
		no_bottom = sum(not bottom(r, False) for r in group)
		print("  bottom views", len(scores), "| episodes without a bottom view", no_bottom)
		if scores:
			alarms = sum(s >= 0.0 for s in scores)
			print(f"  bottom scores min {scores[0]:.3f} median {scores[len(scores) // 2]:.3f} max {scores[-1]:.3f}")
			print(f"  bottom views at or above 0: {alarms}/{len(scores)}")
		failures = {}
		for r in group:
			for code in r["failures"]:
				failures[code] = failures.get(code, 0) + 1
		print("  failure codes", failures)


if __name__ == "__main__":
	main()
