# S10 report: ACT baseline run 02 (2026-11-20)

Checkpoint: `E:/scratch/s9-act-baseline-02/best` (step 20000, the last step; selection rule fell back to highest F1
because no checkpoint passed the S9 gate). Closed loop: `E:/scratch/s10-act-baseline-02` (sync, 60 val layouts).

## The three questions

**1. How many of the 60 val layouts are sorted?** 0/60. Teacher 57/60; gate 40/60. Paired diff -0.95, 95% CI
[-1.00, -0.88]. Jaw-environment contact in 59/60 episodes (1553 frames, max 131 N); robot-body contact 0.

**2. Where do failures first occur?** All 60 at the first `grasp` segment, code `no_clamp`. The student gets through
`prepare`, the scan views and `return_ready`, then closes the jaws at a wrong wrist roll. After the retry the
`return_ready` and second `grasp` segments end after 1-4 frames (end head fires at once off-distribution) and the
episode fails again.

**3. Where should the next budget go?** Not into more episodes of the same kind. Fix the student's conditioning first.

## Diagnosis

| | S9 val | Target |
|---|---|---|
| End-step F1 (±2) | 0.52 (still rising: 0.41 @10k, 0.49 @16k) | ≥ 0.90 |
| Action MAE | 0.059 rad | - |

Open-loop endpoint error of the predicted chunk at segment start (60 samples per row, rad, joints 0-5):

| split / kind | base | shoulder | elbow | flex | wrist roll | gripper |
|---|---|---|---|---|---|---|
| train grasp | 0.076 | 0.061 | 0.058 | 0.040 | **0.519** | 0.232 |
| val grasp | 0.088 | 0.065 | 0.071 | 0.050 | **0.623** | 0.249 |
| train scan_view | 0.061 | 0.048 | 0.052 | 0.022 | **0.307** | 0.017 |
| val carry_to_bowl | 0.633 | 0.087 | 0.156 | 0.059 | 0.075 | 0.015 |

Closed loop, grasp-end wrist roll vs the teacher on the same layout: mean |error| 0.76 rad, not explained by a π or
π/2 symmetry.

The wrist-roll error is as large on **train** as on val, so this is not a generalisation gap. Cause: the teacher
picks the grasp roll in `ClosedLoopExpert._grasp_plan` from 2-4 discrete candidates (`grasp_candidates`: the jaws
across the brick's short axis, or either axis of a square brick, from either side), ranked by clearance to bowls and
clutter (capped) minus a roll penalty. The twin's expert has no `planner`, so `choose_grasp` is not involved. The
choice is a controller-level plan selection (STUDENT_INPUTS.md row 17) and an argmax: near ties it flips by π/2 or π
on small changes in brick pose or bowl position. The student sees neither the candidates nor the scores, and the
`grasp` instruction ("Grasp the brick on the table.") carries no params. L1 regression averages the modes and the
jaws land misaligned. Scan views and bottom views follow the same pattern, milder: their pose comes from the planner
(`target_q` sits in the segment params, which the student never sees).

Secondary: the run is about one epoch (20k steps × batch 16 ≈ 320k samples over 317k frames) and both metrics were
still improving; the end head is far from the 0.9 gate; `carry_to_bowl` base error is 3× higher on val than train.

## Proposed next steps (need a decision)

1. **Condition the student on the planner's choice.** For `grasp`, pass the chosen roll in the instruction and
   state; for `scan_view` and `present_bottom`, the planned view roll. Both can be read from the existing raw traces (grasp
   roll = wrist angle at grasp end; scan `target_q` is already in the segment params), so no re-collection is needed:
   re-export, retrain. StudentTwin knows the planner state at run time and can pass the same values.
2. **Train longer** (≥ 3-5 epochs) once (1) is in; F1 had not plateaued.
3. Hold SmolVLA until (1) is in; it would hit the same missing input.
