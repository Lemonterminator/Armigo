# Hand-eye calibration of the top camera in simulation

Question: can the agent find the top camera's pose relative to the robot base from sensor data alone, without
measured marker positions? It gets the rendered top images, the joint readings, the camera intrinsics and the printed
marker sizes, and nothing else.

## Setup

- `src/digital_twin_armigo/mujoco_cell/handeye.py`: card detection, Park-Martin AX = XB start, then
  Levenberg-Marquardt on the reprojection error of every card corner. The 12 unknowns are the camera pose in the
  base frame and the card pose on the gripper, so the card can be clamped any way.
- `scripts/calib/handeye_sim.py`: per fixture seed (900001 to 900020):
  - a random cell, with the top camera pose and focal length randomised as in the datasets;
  - a 74 mm card with a 2x2 grid of 28 mm ArUco markers (ids 20 to 23), clamped about 5 cm beyond the jaw tips
    with an unknown offset of up to ±5 mm and ±5°;
  - random arm poses around the hand-off pose, kept only when the top camera sees at least 3 card markers;
  - Gaussian RGB noise of σ = 2 grey levels;
  - joint readings perturbed by random noise per pose, or by a fixed bias per joint and seed.
- Scoring against MuJoCo ground truth:
  - camera position and rotation error;
  - **workspace error**: table points at brick-centre height over the reachable area (x 0.12 to 0.40 m, |y| ≤ 0.26 m,
    inside the image), projected with the true camera and located with the estimated one. This is the error a
    top-camera brick localisation would inherit.
- Intrinsics are taken as known (true focal length of each seed).

## Results (20 seeds each, medians; ws = workspace error)

| Joint error | Poses | Reproj. RMS (px) | Camera pos (mm) | Camera rot (°) | ws mean (mm) | ws max, median / p90 (mm) |
|---|---|---|---|---|---|---|
| none | 15 | 0.47 | 1.4 | 0.05 | 0.9 | 1.1 / 1.5 |
| none | 40 | 0.48 | 1.0 | 0.03 | 0.7 | 0.9 / 1.1 |
| noise σ 0.1° | 15 | 0.92 | 3.8 | 0.27 | 1.7 | 2.4 / 4.6 |
| noise σ 0.1° | 40 | 1.02 | 2.2 | 0.17 | 1.1 | 1.7 / 2.5 |
| noise σ 0.3° | 15 | 2.40 | 10.3 | 0.74 | 4.5 | 7.0 / 12.4 |
| noise σ 0.3° | 40 | 2.75 | 5.8 | 0.53 | 2.4 | 3.9 / 7.3 |
| noise σ 1.0° | 15 | 7.79 | 32.8 | 2.52 | 16.0 | 24.3 / 38.2 |
| noise σ 1.0° | 40 | 9.09 | 19.3 | 1.69 | 7.3 | 11.9 / 23.2 |
| bias σ 0.5° | 15 | 0.61 | 8.8 | 0.83 | 2.8 | 5.5 / 8.8 |
| bias σ 0.5° | 40 | 0.60 | 10.1 | 0.92 | 3.2 | 6.0 / 9.6 |

Every run converged (20/20 in all conditions). Each seed takes under a second.

- **The image side is not the limit.** With exact joints, 15 poses give about 1 mm over the workspace. That is
  well inside the tolerance of a 32 mm brick.
- **Joint error is the limit.** Random joint noise propagates through forward kinematics into the card pose, and
  the camera error grows about linearly with it. More poses help roughly as √N.
- **A joint bias does not average out.** A fixed per-joint offset (an encoder zero error) gives about 3 mm mean and
  6 mm max with 15 or 40 poses alike, and the reprojection RMS stays low (0.6 px), so the residual does not reveal
  it. The fix is to add the five joint offsets as unknowns to the same least-squares problem.
- **Coverage.** Random poses put the card in the upper third of the image (near the robot), so the far workspace is
  extrapolated. Choosing poses to cover the image is a planning task for the agent.

### Measuring the cell markers with the calibrated camera

With the arm at rest and the card removed, each detected cell marker was located through the estimated camera
(exact-joint condition):

| Method | Error, median (mm) |
|---|---|
| single-marker PnP (marker size only) | 6.1 to 6.3 |
| ray through the marker centre meets the table plane | 0.9 to 1.1 |

Single-marker PnP has a depth error of several millimetres at 0.6 m. Using the known table plane removes it.
Stickers should therefore be located by ray-plane intersection, and a drift check should compare plane positions.

Only 12 of the 51 markers inside the image were detected. 16 and 17 sit next to the base, mostly outside the top
camera's view. 10 and 11 lie in the bowls, where the rim, clutter or the image edge cut them. Stickers meant for
drift checks must be placed where the top camera sees them whole.

## Next steps

1. Add per-joint offsets to the solver and repeat the bias condition.
2. Choose calibration poses for image coverage instead of at random.
3. Measure the real SO-101's repeatability (random) and zero error (bias). Those two numbers decide whether this
   reaches the 1 to 3 mm range on hardware.
4. Replace the expert's true brick pose with one located through the calibrated camera, and rerun bench16.

Raw numbers: `poses15/summary.json`, `poses40/summary.json`; detected card corners: `poses*/images/`.
