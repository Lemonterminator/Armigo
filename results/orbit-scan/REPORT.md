# Orbit wrist scan, 2026-11-18

The six-face agent's wrist scan (`scan_wrist`) now orbits the brick instead of hovering at fixed offsets.
Code: `src/digital_twin_armigo/mujoco_cell/orbit_scan.py`, used by `TwinMujoco._scan_plan`; the old hover scan
(`plan_table_scan`) stays as the fallback when the orbit cannot cover the faces still needed.

## How it works

- **Axis.** The vertical line through the x-y centre of the brick face that points most upwards (the top face of an
  upright brick, a side face of a brick lying down).
- **Camera.** The SO-101 wrist camera sits 56.6 mm off the jaw axis and 45 mm above the jaw tips, and looks inwards
  56.8° below the horizon. With the jaws straight down, the wrist roll swings it around the jaw axis, on the
  outside, looking back towards the arm.
- **Ring.** The camera follows a horizontal ellipse around the axis: a circle for a square face, stretched along the
  long side of a 2x4. Where the ellipse is not the camera's natural 56.6 mm circle, the jaw axis moves off the brick
  axis (eccentric orbit), so the stand-off from each side face stays about the same.
- **Knobs.** `OrbitParams.tip_heights_m` (jaw tips above the upward face; lower shows more of the sides) and
  `standoffs_m` (ring beyond the face edges). Later attempts (repeat scans, scans after a rewind) use higher, wider
  rings (`ORBIT_ATTEMPTS`).
- **Why not aim the camera freely.** A camera position plus a viewing direction is five constraints for five joints,
  but the approach axis of this arm only tilts within its vertical plane. A free look-at solver reached a handful of
  azimuths on a 2x2 fixture (errors up to 25 mm and 30°), so the orbit keeps the jaws vertical.
- **Safety.** Every ring pose and three poses between neighbouring ring poses must keep the robot 12 mm above the
  table, 6 mm from the brick, and the gripper and wrist camera 10 mm from the base, shoulder and upper arm. The ring
  is split into arcs where a check fails, and one scan sweeps inside one arc. The agent's path check against bowls
  and clutter still runs on every segment.
- **Motion.** The chosen views are visited in wrist-roll order, sweeping through the arc poses between them
  (`sweep`: min-jerk along the joint path, chunks of at most 1.4 s, as `_move` does for one segment).

The self-collision check was added after the first benchmark: on 2x4 brick 610015, close to the base, the far side of
the ring swung the camera mount into the shoulder link. The wrist roll then stayed saturated for 2.6 s, the health
check failed the episode with `torque_stall`, and the same stall came back after the rewind.

## Results

Rule-based skill choices, supervised detector (`sup-local.json`), 4 workers. Old scan on the same fixtures for
comparison: `bottom48` from the current `main`, `bench16` from `results/agent-bench/final/after/run-bench16-rule`.
"Weakest side" is the smallest of the four side faces' best pixel counts over the wrist views of an episode; a face
counts as readable from 800 px. `compare.py` writes `bottom48.json` and `bench16.json`.

| | bottom48, old | bottom48, orbit | bench16, old | bench16, orbit |
|---|---|---|---|---|
| Sorted | 41/48 | 41/48 | 14/16 | 14/16 |
| Rewinds, failure codes | 0 | 0 | 2, `PHYSICAL_HEALTH_FAILED` | 0 |
| Weakest side, median (normal bricks for bench16) | 1000 px | 2111 px | 897 px | 2114 px |
| Weakest side, minimum (bottom48) | 813 px | 1554 px | | |
| Episodes with weakest side under 1500 px (bottom48) | 48 | 0 | | |
| Top face, median | 12,525 px | 11,824 px | 12,296 px | 11,818 px |
| Wrist views per episode | 2.00 | 2.17 | 1.69 | 1.50 |
| Simulated frames | 82,345 | 81,735 | 26,375 | 20,864 |

- On bottom48, every episode's weakest side got more pixels (median ratio 2.1) and no outcome changed.
- In all 64 scans of both suites the planner chose 25 mm tips; the 12 mm rings lost arcs to the clearance checks
  with the jaws open. Defective bricks stop scanning at the first alarm, so the weakest-side row
  uses normal bricks only.
- bench16 case 600001 (normal): the old oblique view raised a 0.885 wrist false alarm; the orbit views do not.
- bench16 case 600008 (normal): the old run's bottom false alarm (0.216) was undone by a rewind for a torque stall.
  The orbit run has no stall; its first, close wrist view scored 0.141 (class head; segmentation head -0.59), just
  over the threshold of 0, so the brick went to pink. The gripper shades the brick in that view
  (`case-0014/keyframes/0-225-wrist_cam-crop.png` of the local run). The wrist thresholds were calibrated on other
  views and need recalibrating on orbit views.

## Not done

- Bricks lying on a side: the planner finds the upward face, but the agent still assumes faces 0-4 come from the
  wrist and face 5 from the top camera. No episode in these suites has a brick on its side.
- Closing the jaws during the scan would make the gripper smaller and could let the 12 mm rings win.
- Wrist thresholds for orbit views (see case 600008). With the detector retrained on bottom views
  (`results/agent-bench/REPORT.md`, "Bottom-view training data") bench16 sorts 16/16 and 600008 is correct, but
  one wrist false alarm is left in bottom2x4 (610138, 0.061), so the recalibration is still open.

Run folders are local (`data/catalog/agent/bottom48`, `orbit2-bottom48`, `orbit2-bench16`). Rebuild with:

```bash
python -m digital_twin_armigo.agent.twin_mcp run --detector results/agent-bench/sup-local.json --workers 4 --headless --out <run> --suite config/agent_bench/bottom48.json
```

```bash
python results/orbit-scan/compare.py old=<old run> orbit=<orbit run>
```
