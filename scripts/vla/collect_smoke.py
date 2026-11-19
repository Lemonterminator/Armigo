"""Collection smoke test: record a small batch on the pilot's command path, export it to LeRobot and check the labels.

The path is exactly the one the S4 pilot uses:

1. rollouts: ``python -m digital_twin_armigo.agent.twin_mcp run --sensing realistic --suite <suite> --workers N
   --detector <detector> --out <dir>/rollouts`` (video on), in this repository's ``.venv``;
2. export: ``python -m digital_twin_armigo.agent.twin_record --out <dir>/lerobot <episodes>`` in the LeRobot env;
3. read back: ``LeRobotDataset(..., video_backend="pyav")`` in the LeRobot env (this script, ``--read-dataset``).

Default batch (``config/vla/train.json``): the first normal case, the first side-face defect and the first face-5
defect. Checks per episode: every frame has a segment; every event other than the segment markers (captures,
perceptions, placement checks, failures, node entries, ...) falls on a segment boundary, and each capture (``view``)
ends a ``scan_view`` / ``present_bottom`` segment; ``segment.chunk_valid_len`` reaches exactly to the next boundary,
so no chunk crosses it; ``task`` equals the segment instruction of the events; ``segment.done`` marks the first
observation of the segment after one that ended with segment_done; frame counts match ``actions.npz``.

	PYTHONPATH=src .venv/Scripts/python scripts/vla/collect_smoke.py --out E:/scratch/collect-smoke
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
LEROBOT_PYTHON = Path("E:/il/lerobot/Scripts/python.exe")
REPO_ID = "local/armigo-collect-smoke"
SEGMENT_EVENTS = ("segment", "segment_end")
CAPTURE_SEGMENTS = ("scan_view", "present_bottom")


def pick_cases(suite: Path, pick: int = 0) -> dict:
	"""The ``pick``-th (0: first) normal case, side-face (0-4) defect and face-5 defect of a suite file."""
	raw = json.loads(suite.read_text(encoding="utf-8-sig"))
	cases = raw["cases"] if isinstance(raw, dict) else raw
	wanted = {
		"normal": lambda c: not c[1],
		"side-face defect": lambda c: c[1] and c[2] != 5,
		"face-5 defect": lambda c: c[1] and c[2] == 5,
	}
	picked = [[c for c in cases if test(c)][pick] for test in wanted.values()]
	out = {"cases": picked, "labels": list(wanted)}
	if isinstance(raw, dict) and raw.get("options"):
		out["options"] = raw["options"]
	return out


def read_dataset(root: Path, out: Path):
	"""LeRobot env side: load the dataset as a trainer would and dump its non-image columns plus a few decoded
	samples. Imports only LeRobot."""
	import numpy as np
	from lerobot.datasets.lerobot_dataset import LeRobotDataset

	ds = LeRobotDataset(REPO_ID, root=root, video_backend="pyav")
	hf = ds.hf_dataset.with_format(None)
	keys = (
		"episode_index",
		"frame_index",
		"task_index",
		"segment.index",
		"segment.kind",
		"segment.done",
		"segment.chunk_valid_len",
	)
	cols = {}
	for key in keys:
		values = hf[key]
		cols[key] = values if key == "segment.kind" else np.asarray(values).reshape(-1).tolist()
	cols["segment.kind"] = [v[0] if isinstance(v, list) else v for v in cols["segment.kind"]]
	tasks = {int(row.task_index): str(task) for task, row in ds.meta.tasks.iterrows()}
	done_rows = [i for i, d in enumerate(cols["segment.done"]) if d]
	samples = []
	for i in sorted({0, *done_rows[:2], len(ds) - 1}):
		x = ds[i]
		samples.append(
			{
				"index": i,
				"task": x["task"],
				"segment.kind": x["segment.kind"],
				"images": {k: list(v.shape) for k, v in x.items() if k.startswith("observation.images.")},
			}
		)
	payload = {"frames": len(ds), "episodes": ds.meta.total_episodes, "columns": cols, "tasks": tasks}
	payload["samples"] = samples
	out.write_text(json.dumps(payload), encoding="utf-8")


def check_recording(episode: Path) -> tuple[dict, dict]:
	"""Label checks on the recording itself (any verdict): every frame has a segment, the segments tile the branch
	(``twin_record.segment_labels``), all other events fall on segment boundaries and each capture ends a capture
	segment."""
	from digital_twin_armigo.agent.twin_record import SegmentLabelError, segment_labels

	events = [json.loads(line) for line in (episode / "events.jsonl").read_text(encoding="utf-8").splitlines()]
	try:
		labels = segment_labels(episode)
	except SegmentLabelError as exc:
		return {"labels": False}, {"error": str(exc), "segments": 0, "captures": 0, "kinds": []}
	segs, n = labels["segments"], len(labels["index"])
	boundaries = {0, n} | {s["start"] for s in segs} | {s["end"] for s in segs}
	end_kind = {s["end"]: s["kind"] for s in segs if s["end"] > s["start"]}
	views = [e for e in events if e["kind"] == "view" and not e.get("inherited")]
	off = sorted({e["kind"] for e in events if e["kind"] not in SEGMENT_EVENTS and e["frame"] not in boundaries})
	checks = {
		"labels": True,
		"events_on_boundaries": not off,
		"captures_end_segments": all(end_kind.get(v["frame"]) in CAPTURE_SEGMENTS for v in views),
	}
	info = {"frames": n, "segments": len(segs), "captures": len(views), "kinds": [s["kind"] for s in segs]}
	return checks, info


def check_episode(episode: Path, rows: dict, offset: int, n: int) -> tuple[dict, dict]:
	"""Compare one exported episode (rows ``offset .. offset+n`` of the dataset columns) with its source events."""
	import numpy as np

	events = [json.loads(line) for line in (episode / "events.jsonl").read_text(encoding="utf-8").splitlines()]
	with np.load(episode / "branch-0" / "actions.npz", allow_pickle=False) as npz:
		npz_segment = npz["segment"]
	starts = {e["id"]: e for e in events if e["kind"] == "segment"}
	ends = {e["id"]: e for e in events if e["kind"] == "segment_end"}
	segs = [(starts[i], ends[i]) for i in sorted(starts)]
	boundaries = {0, n} | {s["frame"] for s, _ in segs} | {e["frame"] for _, e in segs}
	seg_end_kind = {e["frame"]: s["segment_kind"] for s, e in segs if e["frames"] > 0}

	def col(key):
		return rows[key][offset : offset + n]

	index, kind, done, valid = (
		col("segment.index"),
		col("segment.kind"),
		col("segment.done"),
		col("segment.chunk_valid_len"),
	)
	task = [rows["tasks"][t] for t in col("task_index")]
	views = [e for e in events if e["kind"] == "view" and not e.get("inherited")]
	off_boundary = sorted(
		{e["kind"] for e in events if e["kind"] not in SEGMENT_EVENTS and e["frame"] not in boundaries}
	)
	# next boundary of each row: the first later row with another segment id, or the episode end
	next_boundary, nxt = [0] * n, n
	for i in range(n - 1, -1, -1):
		if i + 1 < n and index[i + 1] != index[i]:
			nxt = i + 1
		next_boundary[i] = nxt
	expected_done = [False] * n
	for s, e in segs:
		if e["reason"] == "done" and e["frame"] < n:
			expected_done[e["frame"]] = True
	checks = {
		"frames": n == len(npz_segment) and col("frame_index") == list(range(n)),
		"valid_segment": all(i >= 0 for i in index) and bool((npz_segment >= 0).all()),
		"ids_match_npz": index == npz_segment.tolist(),
		"events_on_boundaries": not off_boundary,
		"captures_end_segments": bool(views) and all(seg_end_kind.get(v["frame"]) in CAPTURE_SEGMENTS for v in views),
		"chunks_within_segment": all(v >= 1 and i + v == next_boundary[i] for i, v in enumerate(valid)),
		"tasks_are_instructions": all(task[i] == starts[index[i]]["instruction"] for i in range(n)),
		"kinds_match": all(kind[i] == starts[index[i]]["segment_kind"] for i in range(n)),
		"done_labels": list(map(bool, done)) == expected_done,
	}
	info = {
		"segments": len(segs),
		"captures": len(views),
		"done_rows": int(sum(map(bool, done))),
		"min_chunk_valid_len": min(valid),
		"off_boundary_kinds": off_boundary,
		"kinds": [s["segment_kind"] for s, _ in segs],
	}
	return checks, info


def main():
	p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	p.add_argument("--out", type=Path, help="scratch directory (outside the repository); must not exist")
	p.add_argument("--suite", type=Path, default=ROOT / "config/vla/train.json")
	p.add_argument("--detector", type=Path, default=ROOT / "results/agent-bench/sup-v7bot-local.json")
	p.add_argument("--workers", type=int, default=2, help="concurrent episodes (the GPU is shared)")
	p.add_argument("--lerobot-python", type=Path, default=LEROBOT_PYTHON)
	p.add_argument("--pick", type=int, default=0, help="take the k-th case of each class (0: the first)")
	p.add_argument("--read-dataset", nargs=2, type=Path, metavar=("ROOT", "JSON"), help=argparse.SUPPRESS)
	a = p.parse_args()
	if a.read_dataset:
		read_dataset(*a.read_dataset)
		return
	if a.out is None:
		p.error("--out is required")
	out = a.out.resolve()
	out.mkdir(parents=True, exist_ok=False)
	env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
	batch = pick_cases(a.suite, a.pick)
	suite = out / "suite.json"
	suite.write_text(json.dumps({k: v for k, v in batch.items() if k != "labels"}, indent=1), encoding="utf-8")

	timing = {}
	started = time.perf_counter()
	run_cmd = [sys.executable, "-m", "digital_twin_armigo.agent.twin_mcp", "run", "--sensing", "realistic"]
	run_cmd += ["--suite", str(suite), "--detector", str(a.detector), "--workers", str(a.workers)]
	run_cmd += ["--out", str(out / "rollouts")]
	print("rollouts:", " ".join(run_cmd), flush=True)
	subprocess.run(run_cmd, env=env, check=True, stdout=subprocess.DEVNULL)
	timing["rollouts_s"] = time.perf_counter() - started

	episodes, rows_out = [], []
	for i, (case, label) in enumerate(zip(batch["cases"], batch["labels"], strict=True)):
		episode = out / "rollouts" / f"case-{i:04d}"
		result = json.loads((episode / "episode.json").read_text(encoding="utf-8"))["result"]
		rows_out.append({"case": case, "class": label, "result": result, "episode": episode})
		if result["verdict"] == "SUCCESS" and result["student_eligible"]:
			episodes.append(episode)

	started = time.perf_counter()
	export_cmd = [str(a.lerobot_python), "-m", "digital_twin_armigo.agent.twin_record", "--out", str(out / "lerobot")]
	export_cmd += ["--repo-id", REPO_ID, *map(str, episodes)]
	print(f"export ({len(episodes)} eligible episodes) in the LeRobot env", flush=True)
	subprocess.run(export_cmd, env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
	timing["export_s"] = time.perf_counter() - started
	dump = out / "readback.json"
	read_cmd = [str(a.lerobot_python), str(Path(__file__).resolve()), "--read-dataset", str(out / "lerobot"), str(dump)]
	subprocess.run(read_cmd, env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
	data = json.loads(dump.read_text(encoding="utf-8"))
	cols = {**data["columns"], "tasks": {int(k): v for k, v in data["tasks"].items()}}
	segments_meta = json.loads((out / "lerobot" / "meta" / "segments.json").read_text(encoding="utf-8"))

	header = (
		f"{'case':22} {'class':17} {'verdict':8} {'label':9} {'frames':>6} {'segs':>4} {'caps':>4} {'done':>4}  checks"
	)
	print("\n" + header + "\n" + "-" * len(header))
	ep_index = cols["episode_index"]
	labels_pass = data["episodes"] == len(episodes)
	for row in rows_out:
		r = row["result"]
		checks, info = check_recording(row["episode"])
		if row["episode"] in episodes:
			e = episodes.index(row["episode"])
			positions = [i for i, x in enumerate(ep_index) if x == e]
			offset, n = positions[0], len(positions)
			exported, info = check_episode(row["episode"], cols, offset, n)
			exported["meta_segments"] = len(segments_meta["episodes"][e]["segments"]) == info["segments"]
			checks |= exported
			done = f"{info['done_rows']:4d}"
		else:
			n, done = info.get("frames", 0), "   -"
		ok = all(checks.values())
		labels_pass &= ok
		failed = [k for k, v in checks.items() if not v]
		verdict = "PASS" if ok else "FAIL " + ",".join(failed)
		if row["episode"] not in episodes:
			verdict += " (recording only: not exported, ineligible)"
		print(
			f"{row['case']!s:22} {row['class']:17} {r['verdict']:8} {r['label']:9} {n:6d} {info['segments']:4d} "
			f"{info['captures']:4d} {done}  {verdict}"
		)
		print(f"{'':22} segments: {' '.join(info['kinds'])}")
	print(
		f"\nLeRobot read-back: {data['frames']} frames, {data['episodes']} episodes, {len(cols['tasks'])} distinct tasks;"
		f" decoded samples {[s['index'] for s in data['samples']]} -> tasks match: "
		f"{all(s['task'] == cols['tasks'][cols['task_index'][s['index']]] for s in data['samples'])}"
	)
	print(f"timing: rollouts {timing['rollouts_s']:.0f} s ({a.workers} workers), export {timing['export_s']:.0f} s")
	experts_ok = len(episodes) == len(rows_out)
	all_pass = labels_pass and experts_ok
	summary = {
		"pass": all_pass,
		"labels_pass": labels_pass,
		"expert_eligible": [len(episodes), len(rows_out)],
		"timing": timing,
		"episodes": [str(e) for e in episodes],
	}
	(out / "smoke.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
	print(f"labels and export: {'PASS' if labels_pass else 'FAIL'}; expert eligible {len(episodes)}/{len(rows_out)}")
	print("PASS" if all_pass else "FAIL")
	sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
	main()
