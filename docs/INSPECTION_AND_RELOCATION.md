# Brick inspection, scan zone and table relocation (round of 2026-11-17)

Scope: MuJoCo SO-101 sorting cell, simulation only. Nothing here has been run on hardware. Numbers are taken from the
result files named in each section. The large datasets and trained models are gitignored; only metadata, summaries and
reports are versioned.

## 1. What changed

| Area | Files | What it does |
|---|---|---|
| Refactor | `mujoco_cell/episode.py`, `episode_diagnostics.py`, `grasp_geometry.py`, `physics_expert.py`, `scene.py`, `cell.py` | `PhysicsEpisode` loop and diagnostics are split out of the expert; grasp geometry gets its own module. `cell.py` adds per-face segmentation probes, which make per-face visible pixels measurable. |
| Inspection views | `mujoco_cell/inspection_views.py`, `inspection_planning.py` | `HeldBrickPoser` (kinematic copy of the held brick), per-face visibility, `greedy_face_cover`, pre-scan and close-up candidate libraries, `best_closeup`, `path_min_z`. |
| Expert hooks | `physics_expert.py` (`EpisodePlanner`) | Optional planner hooks: pre-scan poses, grasp choice, after-grasp, present poses, label. With no planner, the default path is unchanged; the target-state hash still matches. |
| Anomaly detection | `scripts/anomaly/*`, `inspection_scorer.py`, `tests/fake_score_server.py` | Renders a dataset of brick crops, trains and evaluates anomalib models (PaDiM / PatchCore), and runs a scoring client and server. v2 uses only the `cross_var` defect: a cross that may be skewed, off-centre or asymmetric and covers at least 1/4 of the face. |
| Agent | `agent/graph.py`, `policies.py`, `llm.py`, `heuristic.py`, `heuristic_episode.py` | LangGraph inspection graph, baseline policies (none / fixed / LLM), and the heuristic inspector. The inspector chains: pre-scan, grasp avoiding the suspect faces, held view, then top-camera close-ups of the next face to check. |
| Scan zone | `scripts/scan_zone.py`, `scripts/scan_zone_report.py` | Kinematic map of where a brick on the table can be scanned in place, and how. |
| Relocation | `mujoco_cell/table_relocation.py`, `scripts/relocate_experiment.py` | `RelocateExpert`, which picks a far brick, puts it down at a target point on the table, scans it there and shows its bottom to the top camera. |
| Studies | `scripts/inspection_visibility.py`, `top_camera_reach.py`, `run_inspection_policies.py` | Face visibility per camera / pose, top-camera placement vs reach, and the policy comparison. |

## 2. Anomaly detector (cross defect only)

From `results/anomaly/v2/REPORT.md`. The model is PatchCore; the threshold is the 0.99 quantile of validation normals.

| camera | AUROC | F1 @thr | precision | recall | FPR |
|---|---|---|---|---|---|
| wrist_cam | 0.815 | 0.584 | 0.933 | 0.424 | 0.016 |
| top | 0.696 | 0.171 | 0.812 | 0.096 | 0.008 |

The wrist camera is the useful detector. The top camera at its default distance sees too few pixels per face, which
motivates the close-up and the closer top camera below.

## 3. Scan zone: what to do at each brick position

`scan_zone.py` sweeps radius 0.08 to 0.46 m (step 0.02) from the shoulder-pan axis, azimuth -90 to 90 deg (step 15),
and brick yaw 0 and 30 deg. Each position is tested three ways:

- **graspable**: some grasp candidate has an IK error below 5 mm and non-negative clearance.
- **scan_vertical**: jaws straight down, with the wrist roll orbiting the wrist camera. Strict vertical IK uses
  multistart seeds (the ready pose plus 12 random restarts). The position counts as covered when every face 0-4 shows at
  least 800 px.
- **scan_outside**: a tilted sweep from slightly outside the brick (`ik_reach`).

`scan_zone_report.py` takes the worst case over yaws and writes `decision.json` and `decision_map.png`. Positions inside
a bowl are flagged `over_bowl` rather than skipped.

| decision | positions | meaning |
|---|---|---|
| scan_vertical | 56 | scan top + 4 sides in place, jaws down; radius band 0.08 to 0.26 m |
| scan_outside | 32 | in-place tilted sweep; up to about 0.34 m |
| relocate | 30 | graspable (up to 0.38 m) but no in-place scan: carry it in first |
| unreachable | 54 | not graspable (r of 0.40 m or more): report an error |
| over_bowl | 88 | inside a bowl of this layout |

**Minimum carry** for a relocation is the radial distance to the vertical zone's outer edge at the same azimuth, minus a
margin: min 0.04, median 0.13, max 0.30 m. `relocation_target()` computes it.

## 4. Relocation experiment

### Setup

`relocate_experiment.py` uses the following setup:

- **Bowls:** moved behind the robot, out of the work area.
- **Clutter:** cleared.
- **Defect:** on the bottom face (5).
- **Top camera:** at 1.2, 1.35 or 1.5 times the maximum tip reach (0.472 m), measured from the shoulder anchor. It sits
  at 45 deg elevation, looking at the zone edge.
- **Starts:** uniform over radius 0.30 to 0.38 m, azimuth ±75 deg and yaw ±45 deg.
- **Target:** inside the vertical zone (outer edge 0.26 m, margin 0.04 m).

### `RelocateExpert` stages

1. Ready, then grasp and lift.
2. **Cartesian carry:** a straight line at constant height (at least 3 cm), with IK every 1 cm, to the target xy.
3. Descend so the release clearance is 4 mm, then open, settle and retreat.
4. Plan a vertical scan with `plan_table_scan` (vertical first, outside as fallback) and visit up to 5 views.
5. Regrasp and lift.
6. `plan_bottom_view`: sample whole-arm poses and keep the one that maximizes the bottom face's pixels in the top
   camera. The held brick must stay at least 50% visible, and the path must stay above the table.
7. Hold, put the brick back at the target, and go home.

**Fixed during the round:**

- **Fault:** at r of 0.35 m or more, 12 of 40 trials lost the brick in transfer. The lift reached only z ≈ 0.036 m,
  and a single joint-space move to the target swung the wrist through a large rotation.
- **Fix:** the straight-line Cartesian carry. Losses fell to 5 of 40.

### Results

n = 40 per camera distance, `results/relocate/cam*/summary.json`.

| top camera | distance from anchor | full pipeline | placement error | tilt | bottom face px (median) |
|---|---|---|---|---|---|
| 1.2x | 0.567 m | 85% | ~5 mm | 0 | ~7,650 |
| 1.35x | 0.637 m | 82.5% | 5.2 mm | 0 | 3,726 |
| 1.5x | 0.708 m | 85% | ~5 mm | 0 | ~3,230 |

At 1.35x, every placed brick was scanned with the vertical scan on all five faces, and its bottom was shown. The median
carry was 0.11 m.

**Failures at 1.35x:** 5 lost in transfer and 2 grasp failures, all from starts beyond 0.35 m.

| start radius | success |
|---|---|
| 0.30–0.34 m | 22 / 22 |
| 0.34–0.36 m | 8 / 9 |
| 0.36–0.38 m | 3 / 9 |

**Recommended limits:**

- Treat about 0.36 m as the practical grasp limit; beyond it, report the brick as out of reach.
- Relocate bricks between 0.26 and 0.36 m into the vertical zone before scanning.
- Put the top camera at about 1.2x reach for the bottom check; it gives twice the pixels of 1.35x.

## 5. Reproduce

```bash
.venv/Scripts/python scripts/scan_zone.py                                         # -> results/scan-zone/positions.jsonl
E:/venvs/armigo-anomaly/Scripts/python.exe scripts/scan_zone_report.py              # needs matplotlib
.venv/Scripts/python scripts/relocate_experiment.py --cam-factor 1.35 --n 40       # -> results/relocate/cam1.35
```

Check `--help` on each script for the exact flags. The anomaly scripts need the separate anomalib environment, which has
torch and anomalib.

## 6. Not covered / open

- Simulation only, with an idealised wrist camera and no motion blur.
- The relocation expert does not yet call the anomaly detector. It measures geometry (face pixels), not detections.
- Joining relocation and the heuristic inspector into one episode is the next step.
- Transfer losses beyond 0.35 m remain. A higher lift, or a regrasp closer to the base, may help.
