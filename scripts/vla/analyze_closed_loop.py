"""Per-layout failure attribution for S10 closed-loop runs (plan B "片段失败" table).

Reads every episode under ``--runs`` (a student eval dir's ``student/`` or a teacher root) and writes one row per
layout: the case, verdict, predicted vs true label, an outcome class, the FIRST failure (code, skill, frame and the
segment kind active at that frame), the last segment the episode reached, retries and segment end reasons.

Milestone counts per layout (the S10 report's first look): reached a grasp, the FIRST grasp clamped (the next segment
is ``lift``, which StudentTwin only starts after a verified clamp), a lift completed, sorted, and collisions (from the
eval's ``grades.json``, the same per-frame contact limits as the gate) with the segment of the first contact frame.

Outcome classes (mutually exclusive, in this order):
  sorted                 verdict SUCCESS
  detector_false_normal  defective brick labelled normal (detector/controller misjudgement)
  detector_false_defect  normal brick labelled defective
  mechanical             a recorded skill failure (no clamp, brick lost, fault, frame budget, ...) or other non-success
  abstain_unknown        label unknown with no recorded failure (controller abstained)

	.venv/Scripts/python scripts/vla/analyze_closed_loop.py --runs E:/scratch/s10-act-01/student --out E:/scratch/s10-act-01
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
	sys.path.insert(0, str(ROOT / "src"))


def _episode_dirs(root: Path) -> list[Path]:
	return sorted(p.parent for p in root.rglob("episode.json"))


def _case_of(ep: Path) -> list | None:
	for name in ("task.json", "run.json"):
		p = ep / name
		if p.exists():
			case = json.loads(p.read_text(encoding="utf-8")).get("case")
			if isinstance(case, list) and case:
				return case
	return None


def _segment_at(segments: list[dict[str, Any]], frame: int) -> str | None:
	"""Kind of the segment whose actions led to ``frame``: the last one that started strictly before it.

	A failure is detected at a segment boundary (e.g. the clamp check after ``grasp``), where the recovery segment
	starts at the same frame; that failure belongs to the segment that just ended, not to the new one."""
	kind = None
	for s in segments:
		if int(s["frame"]) < frame or (frame == 0 and int(s["frame"]) == 0):
			kind = s["segment_kind"]
	return kind


def _milestones(events: list[dict[str, Any]]) -> dict[str, Any]:
	"""First-grasp clamp and lift completion from the segment events."""
	seq = [e for e in events if e.get("kind") in ("segment", "segment_end")]
	first_grasp = None
	for i, e in enumerate(seq):
		if e["kind"] == "segment_end" and e.get("segment_kind") == "grasp":
			nxt = next((x for x in seq[i + 1 :] if x["kind"] == "segment"), None)
			first_grasp = {
				"reason": e.get("reason"),
				"frames": e.get("frames"),
				"clamped": e.get("reason") == "done" and nxt is not None and nxt.get("segment_kind") == "lift",
			}
			break
	lifted = any(
		e["kind"] == "segment_end" and e.get("segment_kind") == "lift" and e.get("reason") == "done" for e in seq
	)
	return {"reached_grasp": first_grasp is not None, "first_grasp": first_grasp, "lifted": lifted}


def _collisions(grade: dict[str, Any] | None, segments: list[dict[str, Any]]) -> dict[str, Any] | None:
	col = ((grade or {}).get("checks") or {}).get("collisions")
	if not col or col.get("status") == "unavailable":
		return None
	firsts = col.get("first_frames") or []
	frame = int(firsts[0]["frame"]) if firsts else None
	return {
		"robot_contact_frames": int(col.get("robot_contact_frames", 0)),
		"jaw_env_contact_frames": int(col.get("jaw_env_contact_frames", 0)),
		"max_jaw_env_contact_n": col.get("max_jaw_env_contact_n"),
		"first_frame": frame,
		"first_segment_kind": None if frame is None else _segment_at(segments, frame + 1),
	}


def analyze_episode(ep: Path, grade: dict[str, Any] | None = None) -> dict[str, Any]:
	result = json.loads((ep / "episode.json").read_text(encoding="utf-8"))["result"]
	events_path = ep / "events.jsonl"
	events = (
		[json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line]
		if events_path.exists()
		else []
	)
	case = _case_of(ep)
	true_label = None if case is None else ("defective" if bool(case[1]) else "normal")
	label = result.get("label")
	verdict = result.get("verdict")
	if verdict == "SUCCESS":
		outcome = "sorted"
	elif true_label == "defective" and label == "normal":
		outcome = "detector_false_normal"
	elif true_label == "normal" and label == "defective":
		outcome = "detector_false_defect"
	elif result.get("failures"):
		# A recorded motion/skill failure (no clamp, frame budget, fault, ...) ends the episode before any verdict:
		# its "unknown" label is a consequence, not a controller abstention.
		outcome = "mechanical"
	elif label in (None, "unknown"):
		outcome = "abstain_unknown"
	else:
		outcome = "mechanical"
	segments = [e for e in events if e.get("kind") == "segment"]
	ends = Counter(e.get("reason") for e in events if e.get("kind") == "segment_end")
	failures = result.get("failures") or []
	first = None
	if failures:
		f = min(failures, key=lambda x: int(x.get("frame", 0)))
		first = {
			"code": f.get("code"),
			"skill": f.get("skill"),
			"frame": f.get("frame"),
			"segment_kind": _segment_at(segments, int(f.get("frame", 0))),
		}
	return {
		"case": case,
		"episode_dir": str(ep),
		"verdict": verdict,
		"true_label": true_label,
		"label": label,
		"outcome": outcome,
		"first_failure": first,
		"last_segment": segments[-1]["segment_kind"] if segments else None,
		"segments": len(segments),
		"segment_end_reasons": dict(ends),
		"retries": int(result.get("retries", 0) or 0),
		"rewinds": int(result.get("rewinds", 0) or 0),
		**_milestones(events),
		"collisions": _collisions(grade, segments),
	}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
	fails = [r for r in rows if r["outcome"] != "sorted"]
	graded = [r for r in rows if r["collisions"] is not None]
	hits = [r for r in graded if r["collisions"]["robot_contact_frames"] + r["collisions"]["jaw_env_contact_frames"]]
	return {
		"episodes": len(rows),
		"milestones": {
			"reached_grasp": sum(r["reached_grasp"] for r in rows),
			"first_grasp_clamped": sum(bool((r["first_grasp"] or {}).get("clamped")) for r in rows),
			"lifted": sum(r["lifted"] for r in rows),
			"sorted": sum(r["outcome"] == "sorted" for r in rows),
			"collision_episodes": len(hits) if graded else None,
			"collision_graded_episodes": len(graded),
		},
		"collision_first_segment": dict(Counter(r["collisions"]["first_segment_kind"] for r in hits)),
		"outcomes": dict(Counter(r["outcome"] for r in rows)),
		"first_failure_segment": dict(
			Counter((r["first_failure"] or {}).get("segment_kind") or r["last_segment"] for r in fails)
		),
		"first_failure_code": dict(Counter((r["first_failure"] or {}).get("code") for r in fails)),
		"segment_end_reasons": dict(sum((Counter(r["segment_end_reasons"]) for r in rows), Counter())),
		"episodes_with_retry": sum(1 for r in rows if r["retries"] > 0),
	}


def main(argv: list[str] | None = None) -> int:
	p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	p.add_argument("--runs", type=Path, required=True)
	p.add_argument("--out", type=Path, required=True)
	p.add_argument("--name", default="failure_attribution.json")
	p.add_argument("--grades", type=Path, default=None, help="eval grades.json (default: <out>/grades.json if present)")
	args = p.parse_args(argv)
	grades_path = args.grades or (args.out / "grades.json")
	by_dir: dict[str, dict[str, Any]] = {}
	if grades_path.exists():
		for g in json.loads(grades_path.read_text(encoding="utf-8")).get("grades", []):
			by_dir[str(Path(g["episode"]).resolve())] = g
	rows = [analyze_episode(ep, by_dir.get(str(ep.resolve()))) for ep in _episode_dirs(args.runs)]
	rows.sort(key=lambda r: (r["case"] or [0])[0])
	report = {"runs": str(args.runs), "summary": summarize(rows), "rows": rows}
	args.out.mkdir(parents=True, exist_ok=True)
	(args.out / args.name).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
	print(json.dumps(report["summary"], indent=2))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
