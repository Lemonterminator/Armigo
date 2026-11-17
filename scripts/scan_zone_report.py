"""Decision map from a ``scan_zone.py`` run: what the pre-grasp inspection should do at every brick position.

Per position (worst case over the brick yaws tested):

* ``scan_vertical``   -- scan top + 4 sides in place, gripper straight down, wrist roll orbiting the camera;
* ``scan_outside``    -- not possible straight down, but the tilted (outside) sweep scans all five faces in place;
* ``relocate``        -- graspable, but no in-place scan: pick the brick and put it down at the nearest point of the
  vertical-scan zone (same azimuth), then scan there;
* ``unreachable``     -- not graspable: report an error;
* ``over_bowl``       -- the position lies in a bowl of this cell (not a table position; excluded).

The minimum carry distance for ``relocate`` is the radial distance to the vertical-scan zone's outer edge at that
azimuth (minus a margin). The bottom face is checked after the grasp in every case (top-camera close-up).
Writes ``decision_map.png`` and ``decision.json`` next to the input.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CATS = ("scan_vertical", "scan_outside", "relocate", "unreachable", "over_bowl")
COLORS = {
	"scan_vertical": "tab:green",
	"scan_outside": "tab:olive",
	"relocate": "tab:orange",
	"unreachable": "tab:red",
	"over_bowl": "lightgray",
}


def decide(rows: list[dict]) -> str:
	"""The action for one position given its rows (one per yaw): the worst case over yaws."""
	if any(r.get("over_bowl") for r in rows):
		return "over_bowl"
	if all(r["scan_vertical"] for r in rows):
		return "scan_vertical"
	if all(r["scan_vertical"] or r["scan_outside"] for r in rows):
		return "scan_outside"
	if all(r["graspable"] for r in rows):
		return "relocate"
	return "unreachable"


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	ap.add_argument("--run", type=Path, default=ROOT / "results" / "scan-zone")
	ap.add_argument("--margin", type=float, default=0.02, help="put a relocated brick this far inside the zone edge")
	a = ap.parse_args()
	rows = [json.loads(line) for line in (a.run / "positions.jsonl").read_text(encoding="utf-8").splitlines() if line]
	by_pos: dict[tuple[float, float], list[dict]] = defaultdict(list)
	for r in rows:
		by_pos[(r["radius"], r["azimuth_deg"])].append(r)
	dec = {k: decide(v) for k, v in by_pos.items()}
	# vertical-scan zone outer edge per azimuth
	edge: dict[float, float | None] = {}
	for az in sorted({k[1] for k in by_pos}):
		ok = [rad for (rad, a2), c in dec.items() if a2 == az and c == "scan_vertical"]
		edge[az] = max(ok) if ok else None
	moves = []
	for (rad, az), c in dec.items():
		if c == "relocate" and edge[az] is not None:
			moves.append(max(0.0, rad - (edge[az] - a.margin)))
	out = {
		"counts": dict(Counter(dec.values())),
		"vertical_zone_outer_edge_m_by_azimuth": {f"{k:.0f}": v for k, v in edge.items()},
		"vertical_zone_radius_band_m": [
			min(r for (r, _), c in dec.items() if c == "scan_vertical"),
			max(r for (r, _), c in dec.items() if c == "scan_vertical"),
		]
		if any(c == "scan_vertical" for c in dec.values())
		else None,
		"relocate_carry_m": {
			"min": round(float(np.min(moves)), 3),
			"median": round(float(np.median(moves)), 3),
			"max": round(float(np.max(moves)), 3),
		}
		if moves
		else None,
		"graspable_outer_radius_m": max((r for (r, _), c in dec.items() if c != "unreachable"), default=None),
		"over_bowl_positions": sum(any(x.get("over_bowl") for x in v) for v in by_pos.values()),
	}
	(a.run / "decision.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
	print(json.dumps(out, indent=2))

	fig, ax = plt.subplots(figsize=(8.5, 6))
	for cat in CATS:
		pts = [by_pos[k][0]["xy"] for k, c in dec.items() if c == cat]
		if pts and cat != "over_bowl":
			p = np.array(pts)
			ax.scatter(p[:, 0], p[:, 1], s=28, c=COLORS[cat], label=f"{cat} ({len(pts)})")
	bowl = [by_pos[k][0]["xy"] for k in by_pos if any(x.get("over_bowl") for x in by_pos[k])]
	if bowl:
		b = np.array(bowl)
		ax.scatter(b[:, 0], b[:, 1], s=60, facecolors="none", edgecolors="k", lw=0.6, label="over a bowl")
	ax.plot(0.0388, 0.0, "k^", ms=10, label="robot base (pan axis)")
	ax.set_aspect("equal")
	ax.set_xlabel("x (m)")
	ax.set_ylabel("y (m)")
	ax.set_title("Pre-grasp inspection: what to do at each brick position\n(worst case over brick yaws)")
	ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
	ax.grid(alpha=0.3)
	fig.tight_layout()
	fig.savefig(a.run / "decision_map.png", dpi=130)


if __name__ == "__main__":
	main()
