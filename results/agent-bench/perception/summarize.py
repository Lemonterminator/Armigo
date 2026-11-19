"""Summarise six-face agent runs with and without brick perception, matched by fixture.

	python results/agent-bench/perception/summarize.py NAME=RUN_DIR [NAME=RUN_DIR ...]

Per run: sorted episodes, correct labels, rewinds, failure codes, wrist views, simulated frames, and (with
perception) the camera calibration error and the error of every brick localisation. Writes ``summary.json`` next
to this script.
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
		calib, located, wrist = None, [], 0
		with (ep / "events.jsonl").open(encoding="utf-8") as f:
			for e in map(json.loads, f):
				if e["kind"] == "camera_calibrated":
					calib = {k: e[k] for k in ("cam_pos_err_mm", "cam_rot_err_deg", "rms_px", "poses", "tries")}
				elif e["kind"] == "brick_perceived":
					located.append(
						{k: e.get(k) for k in ("found", "node", "error_xy_mm", "error_yaw_deg", "size_correct")}
						| {"extended": any(v > 0 for v in e.get("occlusion_extended_mm", []))}
					)
				elif e["kind"] == "view" and e["observation"]["camera"] == "wrist_cam":
					wrist += e["observation"]["branch"] == result["active_branch"]
		out[case] = {
			"verdict": result["verdict"],
			"label": result["label"],
			"label_correct": result["label_correct"],
			"rewinds": result["rewinds"],
			"failures": [x.get("code") for x in result.get("failures", [])],
			"sim_frames": result["sim_frames"],
			"wrist_views": wrist,
			"calibration": calib,
			"localisations": located,
		}
	return out


def describe(name: str, rows: dict) -> dict:
	vals = list(rows.values())
	locs = [x for r in vals for x in r["localisations"] if x["found"]]
	calib = [r["calibration"] for r in vals if r["calibration"]]

	def spread(xs):
		return {"median": round(statistics.median(xs), 3), "max": round(max(xs), 3)} if xs else None

	return {
		"name": name,
		"episodes": len(vals),
		"sorted": sum(r["verdict"] == "SUCCESS" for r in vals),
		"labels_correct": sum(r["label_correct"] for r in vals),
		"rewinds": sum(r["rewinds"] for r in vals),
		"failure_codes": sorted({c for r in vals for c in r["failures"] if c}),
		"wrist_views": sum(r["wrist_views"] for r in vals),
		"sim_frames": sum(r["sim_frames"] for r in vals),
		"calibration_pos_err_mm": spread([c["cam_pos_err_mm"] for c in calib]),
		"calibration_rot_err_deg": spread([c["cam_rot_err_deg"] for c in calib]),
		"localisations": len(locs),
		"not_found": sum(not x["found"] for r in vals for x in r["localisations"]),
		"localisation_xy_err_mm": spread([x["error_xy_mm"] for x in locs]),
		"localisation_yaw_err_deg": spread([x["error_yaw_deg"] for x in locs]),
		"size_wrong": sum(not x["size_correct"] for x in locs),
		"occlusion_extended": sum(x["extended"] for x in locs),
	}


def main(argv: list[str]) -> int:
	runs = dict(arg.split("=", 1) for arg in argv)
	loaded = {name: load(Path(path)) for name, path in runs.items()}
	report = {"runs": [describe(n, rows) for n, rows in loaded.items()], "episodes": {}}
	cases = sorted({c for rows in loaded.values() for c in rows})
	for case in cases:
		report["episodes"][json.dumps(list(case))] = {n: rows.get(case) for n, rows in loaded.items()}
	OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
	for row in report["runs"]:
		print(json.dumps(row))
	for case in cases:
		cells = []
		for n, rows in loaded.items():
			r = rows.get(case)
			cells.append(f"{n}:{'-' if r is None else r['verdict'][:4]}/{r and r['label'][:3]}/rw{r and r['rewinds']}")
		print(case, *cells)
	return 0


if __name__ == "__main__":
	raise SystemExit(main(sys.argv[1:]))
