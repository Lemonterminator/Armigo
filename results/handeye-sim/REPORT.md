# Hand-eye calibration of the top camera in simulation

Question: can the agent find the top camera's pose relative to the robot base from sensor data alone, without
measured marker positions? It gets the rendered top images, the joint readings, the camera intrinsics and the printed
marker sizes, and nothing else.

## Setup

- `src/digital_twin_armigo/mujoco_cell/handeye.py`: card detection, then a Park-Martin AX = XB start, then
  Levenberg-Marquardt on the reprojection error of every card corner.
  - **fixed** solver, 12 unknowns: the camera pose in the base frame and the card pose on the gripper, so the card
    can be clamped any way.
  - **offsets** solver: the same, plus the zero offsets of shoulder_lift, elbow_flex and wrist_flex, with a very weak
    prior of 1 px per 10°. Two offsets are left out because they cannot be identified:
    - shoulder_pan: a pan offset turns the whole arm about the vertical pan axis, and a camera turned the same way
      matches it exactly. The robot then works consistently in its own turned frame.
    - wrist_roll: its offset is absorbed by the card pose on the gripper.
- `scripts/calib/handeye_sim.py`: per fixture seed (900001 to 900020):
  - a random cell, with the top camera pose and focal length randomised as in the datasets;
  - a 74 mm card with a 2x2 grid of 28 mm ArUco markers (ids 20 to 23), clamped about 5 cm beyond the jaw tips
    with an unknown offset of up to ±5 mm and ±5°;
  - random arm poses around the hand-off pose (`--spread` scales their range), kept only when the top camera sees
    at least 3 card markers;
  - Gaussian RGB noise of σ = 2 grey levels;
  - joint readings perturbed by random noise per pose, or by a fixed bias per joint and seed;
  - both solvers run on the same observations.
- Scoring is against MuJoCo ground truth, expressed in the frame the robot believes in (the world turned by the pan
  bias, as explained above):
  - camera position and rotation error;
  - **workspace error ("ws")**: table points at brick-centre height over the reachable area (x 0.12 to 0.40 m,
    |y| ≤ 0.26 m, inside the image), projected with the true camera and located with the estimated one. This is the
    error a top-camera brick localisation would inherit.
- Intrinsics are taken as known (true focal length of each seed).

## Results

20 seeds per row, all solved. Medians, except where marked. "ws" is the workspace error in mm, given as mean /
max / max at p90.

| Joint error | Poses | Fixed: camera | Fixed: ws | Offsets: camera | Offsets: ws | Offset error (°) |
|---|---|---|---|---|---|---|
| none | 15 | 1.4 mm, 0.05° | 0.9 / 1.1 / 1.5 | 1.9 mm, 0.09° | 1.3 / 1.5 / 2.0 | 0.28 |
| none | 40 | 1.0 mm, 0.03° | 0.7 / 0.9 / 1.1 | 1.5 mm, 0.05° | 1.1 / 1.3 / 1.5 | 0.22 |
| none | 40, spread ×2 | 0.6 mm, 0.01° | 0.5 / 0.6 / 0.7 | 0.9 mm, 0.06° | 0.7 / 0.8 / 1.0 | 0.03 |
| bias σ 0.5° | 15 | 6.7 mm, 0.58° | 1.7 / 3.0 / 6.3 | 1.9 mm, 0.09° | 1.3 / 1.5 / 2.0 | 0.28 |
| bias σ 0.5° | 40 | 7.8 mm, 0.66° | 1.3 / 3.1 / 6.0 | 1.5 mm, 0.05° | 1.1 / 1.3 / 1.5 | 0.22 |
| bias σ 0.5° | 40, spread ×2 | 5.4 mm, 0.44° | 1.2 / 2.4 / 4.6 | 0.9 mm, 0.06° | 0.7 / 0.8 / 1.0 | 0.03 |
| bias σ 1.0° | 15 | 12.9 mm, 1.11° | 3.0 / 5.9 / 12.3 | 1.9 mm, 0.09° | 1.3 / 1.5 / 2.0 | 0.28 |
| bias σ 1.0° | 40 | 15.6 mm, 1.31° | 2.6 / 6.2 / 12.2 | 1.5 mm, 0.05° | 1.1 / 1.3 / 1.5 | 0.22 |
| bias σ 1.0° | 40, spread ×2 | 10.8 mm, 0.87° | 2.4 / 5.2 / 8.3 | 0.9 mm, 0.06° | 0.7 / 0.8 / 1.0 | 0.03 |
| noise σ 0.3° | 15 | 10.3 mm, 0.74° | 4.5 / 7.0 / 12.4 | 11.3 mm, 1.00° | 4.4 / 7.3 / 13.9 | 2.48 |
| noise σ 0.3° | 40 | 5.8 mm, 0.53° | 2.4 / 3.9 / 7.3 | 9.6 mm, 0.73° | 2.7 / 4.7 / 9.0 | 1.19 |
| noise σ 0.3° | 40, spread ×2 | 5.1 mm, 0.43° | 4.0 / 5.6 / 7.7 | 7.5 mm, 0.49° | 4.2 / 5.7 / 7.8 | 0.65 |
| noise 0.3° + bias 0.5° | 15 | 9.8 mm, 0.88° | 5.3 / 8.4 / 12.1 | 21.3 mm, 1.98° | 6.5 / 11.4 / 19.4 | 2.69 |
| noise 0.3° + bias 0.5° | 40 | 9.5 mm, 0.76° | 3.7 / 5.3 / 7.3 | 8.5 mm, 0.78° | 3.5 / 6.1 / 7.9 | 1.68 |
| noise 0.3° + bias 0.5° | 40, spread ×2 | 6.6 mm, 0.55° | 2.2 / 4.0 / 8.0 | 8.6 mm, 0.81° | 2.7 / 4.5 / 6.5 | 0.61 |

Earlier versions of this report scored the bias rows in the true world frame, which counted the harmless pan bias
as error. The fixed solver's bias numbers are lower here for that reason.

### Findings

- **The image side is not the limit.** With exact joints, the workspace error is about 1 mm with 15 random poses
  and 0.6 mm with 40 wider ones. That is well inside the tolerance of a 32 mm brick.
- **Estimating the offsets removes a joint bias completely.** The bias rows of the offsets solver equal its no-bias
  rows to the digit: any zero error goes into the offsets. The fixed solver leaves 3 to 6 mm (median max) and up to
  12 mm at p90, and its residual gives little warning: 0.6 to 1.1 px against 0.5 px without bias.
- **The offsets cost a little when there is no bias**: about 0.3 mm on the workspace max.
- **Random joint noise is the floor, and offsets do not help with it.** At σ 0.3° the workspace max is 4 to 7 mm
  with either solver. More poses help roughly as √N. With few poses the extra unknowns overfit the noise: 15 poses
  with noise and bias give 11.4 mm with offsets against 8.4 mm without.
- **Pose diversity decides whether the offsets can be identified.** With the default spread, the lift and elbow
  offset errors come out with opposite signs: only their combination is constrained. At twice the spread the offset
  error falls from 0.22° to 0.03° with exact joints, and from 1.68° to 0.61° with noise and bias. Random wide poses
  are expensive, though: only about 1 in 10 shows the card (40 poses took 180 to 531 tries). Choosing poses for
  coverage and rotation diversity is planning work for the agent.

### Measuring the cell markers with the calibrated camera

With the arm at rest and the card removed, each detected cell marker was located through the estimated camera:

| Method | Error, median, exact joints (mm) |
|---|---|
| single-marker PnP (marker size only) | 5.3 to 6.5 |
| ray through the marker centre meets the table plane | 0.9 to 1.5 |

Single-marker PnP has a depth error of several millimetres at 0.6 m. Using the known table plane removes it.
Stickers should therefore be located by ray-plane intersection, and a drift check should compare plane positions.

Only 12 of the 51 markers inside the image were detected. 16 and 17 sit next to the base, mostly outside the top
camera's view. 10 and 11 lie in the bowls, where the rim, clutter or the image edge cut them. Stickers meant for
drift checks must be placed where the top camera sees them whole.

## What this means for the real cell

- Two numbers of the real SO-101 decide the outcome:
  - **zero error** (bias): the offsets solver handles it, given diverse poses;
  - **repeatability** (random error): this sets the floor.
- To turn backlash from noise into bias, approach every calibration pose from the same direction and let the arm
  settle before reading the joints. The offsets then absorb it.
- Use the offsets solver only with enough diverse poses (40 or more here). Otherwise use the fixed solver. Held-out
  reprojection error can make this choice.

## Next steps

1. Choose calibration poses for image coverage and rotation diversity instead of at random.
2. Measure the real arm's repeatability and zero error.
3. Replace the expert's true brick pose with one located through the calibrated camera, and rerun bench16.

Raw numbers: `poses15/`, `poses40/`, `poses40-spread2/` (`summary.json`); detected card corners: `*/images/`.
