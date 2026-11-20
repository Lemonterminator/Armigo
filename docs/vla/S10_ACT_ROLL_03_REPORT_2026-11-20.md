# S10 report: ACT run 03, wrist roll as conditioning (2026-11-20)

Checkpoint: `E:/scratch/s9-act-roll-03/best` (step 60000, the last step; no checkpoint passed the S9 gate, selection
fell back to highest F1). Change vs run 02: the controller's chosen wrist roll (`roll_deg`) is in the segment
instruction and in state `[36:38]` for `grasp`, `scan_view` and `present_bottom` (STATE_DIM 38), backfilled into the
existing exports; 60k steps (about 3 epochs) instead of 20k. Closed loop: 60 val layouts, sync, four replan settings.

## Result

Gate: student ≥ 40/60 sorted. Not met in any setting.

| S10 setting | first grasp clamped | lifted | sorted | collision episodes | first collision segment |
|---|---|---|---|---|---|
| teacher | 55 | 58 | 57 | - | - |
| no replan (`s10-act-roll-03`) | 6 | 4 | 0 | 49 | grasp 48 |
| replan every 50 frames (`-replan50`) | **12** | **17** | **2** | 33 | grasp 32 |
| replan every 25 frames (`-replan25`) | 5 | 5 | 1 | 54 | grasp 54 |
| replan every 10 frames (`-replan10`) | 0 (stalls in `prepare`) | 0 | 0 | 0 | - |

"lifted" counts lifts after a retry too, so it can exceed "first grasp clamped". Robot-body contact is 0 everywhere;
all contact frames are jaw-environment. First failure with replan 50: `grasp` `no_clamp` 46, `BRICK_LOST` 6,
`PLACEMENT_UNVERIFIED` 5.

S9 val at 60k: action MAE 0.038 rad (run 02: 0.059), end-step F1 0.69 (0.52); grasp segment MAE 0.041, grasp end F1
0.87. Both still improving slowly at the last step.

## What the roll fixed

| open-loop endpoint error at grasp start, wrist roll (rad) | run 02 | run 03 |
|---|---|---|
| train | 0.519 | 0.041 |
| val | 0.623 | 0.058 |

In closed loop (no replan) the student's wrist roll at the end of the first grasp is a median 3° from the commanded
roll (50/60 within 10°). 48/60 layouts get the same roll as the teacher picked on that layout; the rest differ because
the controller plans from the student's belief, not the teacher's.

## What is still wrong

**1. Grasp position.** At the frame the jaws start closing, joints 0-2 differ from the teacher's on the same layout by
a median 0.08 rad (p90 0.15), even where the roll matches. Open-loop at teacher states the error is 0.045-0.05 rad on
train and val alike, so this is precision (training length / visual localisation), not a generalisation gap; closed
loop roughly doubles it.

**2. Each segment runs open loop by default.** `eval_closed_loop.py` with no `--replan-interval` predicts one
100-frame chunk at segment start and replans only when it runs out. A `grasp` segment is about 80-90 frames, so the
whole descent and close run without visual feedback, and the stop frame is fixed at segment start (it cannot react to
jaw contact).

**3. Frequent replanning drifts out of distribution.** Fed teacher states, the model predicts mid-segment motion at
the right speed (`prepare` frame 60: 0.28 rad per 10 frames vs teacher 0.27). Near a motion onset it predicts about
half the speed (frame 10: 0.056 vs 0.117), because the onset time is ambiguous and L1 averages it. With replan 10 the
arm executes only that slow start, falls behind schedule, sees a state the teacher never visited at that segment
frame, predicts another slow start, and stalls (wrist flex moves 0.1 rad in 600 frames; the teacher finishes
`prepare` in 191). Replan 50 is the best of the four: the first chunk carries the motion, one correction lands before
the close.

**Downstream, not yet reached often:** `carry_to_bowl` base joint open-loop endpoint error is 0.63 rad on val vs 0.17
on train (bowl position generalises poorly).

## Proposed next steps (need a decision)

1. **S11 DAgger on run 03 with replan 50.** The failure is covariate shift at grasp; DAgger labels the states the
   student actually visits. `run_dagger_iter.py` has no `--replan-interval` yet; add it so S11 rollouts and S10 use
   the same execution. Gate S11 on correction frames (already in).
2. **Make the student robust to running behind schedule** without new rollouts: noise-injected teacher demonstrations
   (DART-style) or ACT temporal ensembling at inference. Cheaper than (1) to try, but does not label new states.
3. **Controller passes the grasp target pose** (as it now passes the roll). The student would then only need to
   follow the plan; this changes what the student is asked to learn (STUDENT_INPUTS.md), so it is a design decision.
4. Train longer only alongside one of the above; open-loop grasp precision is still improving but closed loop
   doubles the error.
