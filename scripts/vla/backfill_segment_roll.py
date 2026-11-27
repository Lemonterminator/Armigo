"""Add the controller's wrist roll (``roll_deg``) to the segments of an existing student LeRobot export.

Exports made before ``roll_deg`` became a policy param (S10 ACT baseline 02 report) lack it. The value is recoverable
from what was recorded, so no re-collection or video re-encode is needed:

  scan_view, present_bottom   ``target_q[4]`` of the segment params (the planned view pose)
  grasp                       the wrist-roll command at the segment's first ``grasp``-phase frame of the raw trace:
                              ``ClosedLoopExpert._approach_and_descend`` moves over the brick and down at the
                              planned roll, then starts the final descent (phase ``grasp``) at that roll

Only ``meta/segments.json`` changes (a ``.pre-roll.json`` copy is kept); ``SegmentChunkDataset`` reads segment params
from there. ``--check DIR`` instead compares the rule against the ``roll_deg`` a current recorder wrote into a raw
trace's segment events (run it on a fresh expert episode to confirm the backfill matches the live params).

	.venv/Scripts/python scripts/vla/backfill_segment_roll.py E:/scratch/vla-pilot-train/lerobot
	.venv/Scripts/python scripts/vla/backfill_segment_roll.py --check E:/scratch/parity-roll/expert
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
	sys.path.insert(0, str(ROOT / "src"))

ROLL_KINDS = ("scan_view", "present_bottom", "grasp")


def _roll_deg(rad: float) -> float:
	return round(float(np.degrees(float(rad))), 1)


def segment_roll(kind: str, params: dict, seg_id: int, actions: dict[str, np.ndarray]) -> float | None:
	"""The ``roll_deg`` a current recorder would give this segment, from its params and the raw trace."""
	if kind in ("scan_view", "present_bottom"):
		return _roll_deg(params["target_q"][4])
	if kind != "grasp":
		return None
	rows = np.flatnonzero((actions["segment"] == seg_id) & (actions["phase"] == "grasp"))
	if rows.size == 0:
		raise ValueError(f"grasp segment {seg_id} has no grasp-phase frame")
	return _roll_deg(actions["action_f64"][rows[0], 4] if "action_f64" in actions else actions["action"][rows[0], 4])


def _load_actions(raw_dir: Path) -> dict[str, np.ndarray]:
	branch = json.loads((raw_dir / "episode.json").read_text(encoding="utf-8"))["result"].get("active_branch", 0)
	with np.load(raw_dir / f"branch-{branch}" / "actions.npz") as z:
		return {k: z[k] for k in ("segment", "phase", "action", *(["action_f64"] if "action_f64" in z.files else []))}


def backfill(root: Path) -> dict:
	path = root / "meta" / "segments.json"
	meta = json.loads(path.read_text(encoding="utf-8"))
	counts = {k: 0 for k in ROLL_KINDS}
	for ep in meta["episodes"]:
		actions = _load_actions(Path(ep["source"]))
		for seg in ep["segments"]:
			roll = segment_roll(seg["kind"], seg.get("params") or {}, int(seg["id"]), actions)
			if roll is None:
				continue
			sel = np.flatnonzero(actions["segment"] == int(seg["id"]))
			if sel.size and (int(sel[0]), int(sel[-1]) + 1) != (int(seg["start"]), int(seg["end"])):
				raise ValueError(f"{ep['source']}: segment {seg['id']} rows do not match the raw trace")
			old = (seg.get("params") or {}).get("roll_deg")
			if old is not None and abs(float(old) - roll) > 0.05:
				raise ValueError(f"{ep['source']}: segment {seg['id']} already has roll_deg {old} != {roll}")
			seg["params"] = {**(seg.get("params") or {}), "roll_deg": roll}
			counts[seg["kind"]] += 1
	backup = path.with_name("segments.pre-roll.json")
	if not backup.exists():
		shutil.copyfile(path, backup)
	path.write_text(json.dumps(meta) + "\n", encoding="utf-8")
	return {"root": str(root), "episodes": len(meta["episodes"]), "segments_with_roll": counts}


def check(raw_dir: Path) -> dict:
	"""Compare the backfill rule with the roll_deg the recorder put in each segment event of a raw trace."""
	actions = _load_actions(raw_dir)
	rows = []
	for line in (raw_dir / "events.jsonl").read_text(encoding="utf-8").splitlines():
		e = json.loads(line) if line else {}
		if e.get("kind") != "segment" or e.get("segment_kind") not in ROLL_KINDS:
			continue
		live = (e.get("params") or {}).get("roll_deg")
		rule = segment_roll(e["segment_kind"], e.get("params") or {}, int(e["id"]), actions)
		rows.append({"id": e["id"], "kind": e["segment_kind"], "live": live, "backfill": rule})
	bad = [r for r in rows if r["live"] is None or abs(float(r["live"]) - r["backfill"]) > 0.05]
	return {"raw": str(raw_dir), "segments": rows, "mismatches": bad, "pass": bool(rows) and not bad}


def main(argv: list[str] | None = None) -> int:
	p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	p.add_argument("roots", nargs="*", type=Path, help="LeRobot export roots to backfill")
	p.add_argument("--check", type=Path, action="append", default=[], help="raw trace dir to check the rule on")
	args = p.parse_args(argv)
	ok = True
	for raw in args.check:
		rep = check(raw)
		ok &= rep["pass"]
		print(json.dumps(rep, indent=2))
	for root in args.roots:
		print(json.dumps(backfill(root), indent=2))
	return 0 if ok else 1


if __name__ == "__main__":
	raise SystemExit(main())
