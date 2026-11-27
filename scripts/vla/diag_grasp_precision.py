"""Grasp precision diagnostic: open-loop (teacher states) vs closed-loop (student rollouts) error at jaw close.

The question: is the remaining ``grasp`` ``no_clamp`` failure a representation/precision problem (the model is
inaccurate even on teacher states) or a covariate-shift problem (accurate on teacher states, drifts in closed loop)?

Reference frame for both: the *close frame* of the first ``grasp`` segment, i.e. the first frame where the gripper
target has dropped half-way from its value at segment start to its minimum within the segment.

``closed`` (repo .venv or any numpy python): student eval episode vs the paired teacher episode on the same layout,
comparing measured joint state at each side's own close frame.

	python scripts/vla/diag_grasp_precision.py closed --eval E:/scratch/s11-long-03-eval

``open`` (LeRobot CUDA env): the model predicts from teacher frames of the dataset; compares the predicted joint target
at the close frame with the teacher's target, from anchors at segment start and at segment start + 50 (replan 50).

	E:/il/lerobot-cuda/Scripts/python.exe scripts/vla/diag_grasp_precision.py open \\
		--ckpt E:/scratch/s11-long-03/best --dataset E:/scratch/vla-pilot-val/lerobot --out E:/scratch/diag-val.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
	sys.path.insert(0, str(ROOT / "src"))

JOINTS = ["pan", "lift", "elbow", "wflex", "wroll"]


def close_frame(grip_target: np.ndarray, start: int, end: int) -> int | None:
	"""First frame in [start, end) where the gripper target is half-way from its start value to its segment min."""
	seg = np.asarray(grip_target[start:end], dtype=np.float64)
	if seg.size < 2:
		return None
	g0, gmin = float(seg[0]), float(seg.min())
	if g0 - gmin < 1e-3:
		return None
	idx = np.nonzero(seg <= g0 - 0.5 * (g0 - gmin))[0]
	return None if idx.size == 0 else start + int(idx[0])


def _stats(err: np.ndarray) -> dict[str, Any]:
	"""err: (N, 5) absolute joint errors in rad."""
	if err.size == 0:
		return {"n": 0}
	pos = err[:, :3].mean(axis=1)
	return {
		"n": int(err.shape[0]),
		"j012_mean_abs_median": round(float(np.median(pos)), 4),
		"j012_mean_abs_p90": round(float(np.percentile(pos, 90)), 4),
		"per_joint_median": {j: round(float(np.median(err[:, i])), 4) for i, j in enumerate(JOINTS)},
	}


# --------------------------------------------------------------------------------------------- closed loop


def _first_grasp(ep: Path) -> dict[str, Any] | None:
	events = [json.loads(l) for l in (ep / "events.jsonl").read_text(encoding="utf-8").splitlines() if l]
	seq = [e for e in events if e.get("kind") in ("segment", "segment_end")]
	for i, e in enumerate(seq):
		if e["kind"] == "segment" and e.get("segment_kind") == "grasp":
			end = next((x for x in seq[i + 1 :] if x["kind"] == "segment_end" and x.get("id") == e.get("id")), None)
			if end is None:
				return None
			nxt = next((x for x in seq[seq.index(end) + 1 :] if x["kind"] == "segment"), None)
			clamped = end.get("reason") == "done" and nxt is not None and nxt.get("segment_kind") == "lift"
			return {"start": int(e["frame"]), "end": int(end["frame"]), "clamped": bool(clamped)}
	return None


def _close_state(ep: Path) -> tuple[np.ndarray, dict[str, Any]] | None:
	g = _first_grasp(ep)
	if g is None:
		return None
	a = np.load(ep / "branch-0" / "actions.npz")
	c = close_frame(a["action"][:, 5], g["start"], g["end"])
	if c is None:
		return None
	return a["state"][c, :5].astype(np.float64), {**g, "close": c, "close_seg_frame": c - g["start"]}


def run_closed(eval_dir: Path) -> dict[str, Any]:
	summary = json.loads((eval_dir / "eval_summary.json").read_text(encoding="utf-8"))
	rows = []
	for r in summary["paired_rows"]:
		s = _close_state(Path(r["student_dir"]))
		t = _close_state(Path(r["teacher_dir"]))
		if s is None or t is None:
			continue
		err = np.abs(s[0] - t[0])
		rows.append({"case": r["case"], "err": err.tolist(), "student": s[1], "teacher": t[1]})
	err = np.array([r["err"] for r in rows]).reshape(-1, 5)
	clamped = np.array([r["student"]["clamped"] for r in rows], dtype=bool)
	return {
		"eval": str(eval_dir),
		"all": _stats(err),
		"student_clamped": _stats(err[clamped]),
		"student_no_clamp": _stats(err[~clamped]),
		"close_seg_frame_median": {
			"student": float(np.median([r["student"]["close_seg_frame"] for r in rows])),
			"teacher": float(np.median([r["teacher"]["close_seg_frame"] for r in rows])),
		},
		"rows": rows,
	}


# --------------------------------------------------------------------------------------------- open loop


def run_open(ckpt: Path, dataset: Path, max_episodes: int | None, offsets: list[int], device: str) -> dict[str, Any]:
	import torch

	from digital_twin_armigo.vla.dataset import SegmentChunkDataset, vla_collate_fn
	from digital_twin_armigo.vla.models import ACTWithEndHead

	episodes = list(range(max_episodes)) if max_episodes else None
	ds = SegmentChunkDataset(dataset, horizon=100, augment=False, episodes=episodes)
	model = ACTWithEndHead.load_checkpoint(ckpt, device=device)
	model.to(device)
	model.eval()
	h = int(model.horizon)

	index = {(s.ep_pos, s.anchor_row): i for i, s in enumerate(ds._samples) if not s.is_synthesized_zero_frame}
	queries: list[tuple[int, int, int, int]] = []  # (sample_idx, offset, k, ep_pos)
	targets: list[np.ndarray] = []
	for ep_pos, ep in enumerate(ds._episodes):
		kinds = ep.segment_kind
		seg_idx = ep.segment_index
		starts = [t for t in range(ep.length) if kinds[t] == "grasp" and (t == 0 or seg_idx[t] != seg_idx[t - 1])]
		if not starts:
			continue
		s = starts[0]  # first grasp only, matching the closed-loop comparison
		e = s
		while e < ep.length and seg_idx[e] == seg_idx[s]:
			e += 1
		c = close_frame(ep.action[:, 5], s, e)
		if c is None:
			continue
		for off in offsets:
			anchor = s + off
			k = c - anchor
			if anchor >= c or k >= h or (ep_pos, anchor) not in index:
				continue
			queries.append((index[(ep_pos, anchor)], off, k, ep_pos))
			targets.append(ep.action[c, :5].astype(np.float64))

	preds: list[np.ndarray] = []
	bs = 16
	with torch.no_grad():
		for i in range(0, len(queries), bs):
			batch = vla_collate_fn([ds[q[0]] for q in queries[i : i + bs]])
			batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
			pred, _ = model.predict_chunk(batch)
			pred = pred.float().cpu().numpy()
			for j, q in enumerate(queries[i : i + bs]):
				preds.append(pred[j, q[2], :5].astype(np.float64))

	out: dict[str, Any] = {"ckpt": str(ckpt), "dataset": str(dataset), "episodes": len(ds._episodes)}
	for off in offsets:
		sel = [i for i, q in enumerate(queries) if q[1] == off]
		err = np.abs(np.array([preds[i] for i in sel]) - np.array([targets[i] for i in sel])).reshape(-1, 5)
		out[f"anchor_seg_frame_{off}"] = _stats(err)
	return out


def main(argv: list[str] | None = None) -> int:
	p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	sub = p.add_subparsers(dest="cmd", required=True)
	c = sub.add_parser("closed")
	c.add_argument("--eval", type=Path, required=True)
	c.add_argument("--out", type=Path, default=None)
	o = sub.add_parser("open")
	o.add_argument("--ckpt", type=Path, required=True)
	o.add_argument("--dataset", type=Path, required=True)
	o.add_argument("--max-episodes", type=int, default=None)
	o.add_argument("--offsets", type=int, nargs="+", default=[0, 50])
	o.add_argument("--device", default="cuda")
	o.add_argument("--out", type=Path, default=None)
	args = p.parse_args(argv)

	if args.cmd == "closed":
		res = run_closed(args.eval)
	else:
		res = run_open(args.ckpt, args.dataset, args.max_episodes, args.offsets, args.device)
	printable = {k: v for k, v in res.items() if k != "rows"}
	print(json.dumps(printable, indent=1))
	if args.out is not None:
		args.out.write_text(json.dumps(res, indent=1), encoding="utf-8")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
