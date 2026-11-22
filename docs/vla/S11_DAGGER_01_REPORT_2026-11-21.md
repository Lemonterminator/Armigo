# S11 Takeover DAgger 01 report (2026-11-21)

Run: `E:/scratch/s11-act-roll-03`, starting from `s9-act-roll-03/best`. 2 iterations. Each iteration rolls out 60 train
layouts, fine-tunes for 5000 steps on D_base + D_takeover (takeover weight 2.0, no augmentation), then runs S10 val on
60 layouts. All rollouts and evals are sync with replan every 50 frames.

Gate settings:

- Joint-deviation gate: 0.5 rad hard, 0.2 rad soft. `prepare` and `return_ready` are exempt. `grasp` uses 0.08 rad.
- No-clamp takeover is on.
- The expert starts from the student's last command.
- Smoothness is graded on takeover rows only.

The control (`E:/scratch/s11-control-01`) continues run 03 on D_base only for the same 5000 steps with the same flags.
Its eval uses the same S10 settings.

## Result

| model | first grasp clamped | lifted | sorted | collision episodes |
|---|---|---|---|---|
| run 03, replan 50 (baseline) | 12 | 17 | 2 | 33 |
| control: +5000 steps on D_base | 12 | 20 | **7** | 27 |
| S11 iter 1 | 13 | 21 | 3 | 28 |
| S11 iter 2 | 15 | 19 | 2 | 39 |

S11 does not beat the control. The loop stopped as "converged" because iteration 2 did not improve on iteration 1. The
first failure is still `grasp` `no_clamp` in about 45/60 episodes for every model. None of the differences between
2 and 7 out of 60 are significant.

## Takeover data

| iter | takeovers | eligible | passed grading (used) | correction frames | passed, by takeover segment |
|---|---|---|---|---|---|
| 1 | 60 (scan_view 29, grasp 31) | 56 | 28 | 20947 | grasp 23, scan_view 5 |
| 2 | 60 (scan_view 34, grasp 26) | 55 | 29 | 24611 | grasp 17, scan_view 12 |

All takeovers come from `joint_deviation`. The no-clamp gate never fires, because the 0.08 rad grasp gate fires first.

## Why it did not help

1. **Grasp takeovers fire at the start of the segment.** The median is 2-3 frames into the grasp segment; 52 of 57
   fire within 10 frames, and the latest fires at frame 21. A grasp segment is about 80-90 frames. The student never
   gets near the brick.
2. **The recovery is a full reset.** It opens the gripper, returns to ready, re-perceives, and runs the expert grasp
   and lift. The expert then finishes the episode, about 750 takeover frames per used episode.
3. **So D_takeover is teacher demos.** It is ordinary demonstrations that start from ready. It contains no labels for
   the states that fail, where the jaws are near the brick but misaligned. That is why it performs like extra
   teacher data. The control, which simply trains longer, does at least as well.

## Takeaways

- Run 03 was undertrained. 5000 more steps on D_base took sorted from 2 to 7 and lifted from 17 to 20. S9 val was
  still falling at 60k.
- DAgger in this form adds no signal for the grasp failure. To label the states that fail, the takeover must happen
  near the brick and recover in place: from the current pose, move to pre-grasp, servo, close, with no return_ready
  and no re-perceive. The grasp gate must also stop firing on the start-of-segment offset, for example by measuring
  the error only after the approach, or near the end of the chunk.

## Proposed next steps

1. **Train run 03 longer on D_base** (for example +40k steps), then run S10 at replan 50. No code changes. This shows
   how much of the gap is just undertraining.
2. **In-place grasp recovery for S11:** a new expert recovery routine, plus a grasp gate that measures position error
   in the last part of the approach. Rerun S11 on top of the longer-trained checkpoint.
