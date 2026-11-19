# VLA distillation: implementation report 1 (for independent review)

Status date: 2026-11-19. Branch `calibrated-brick-pose`, local only (not pushed). The plan under review is
`docs/VLA_DISTILLATION_PLAN.md` revision 2 (commit `acd732b`). This report covers (A) how revision 2 changed
revision 1 after the first audit, and (B) what has been implemented since, with evidence and known gaps.

How the work was done: five implementation agents worked in parallel, each in its own git worktree, one per plan
step (S0, S1, S2, S3, S6). The coordinating agent (author of this report) reviewed each diff, re-ran the tests,
merged the branches and fixed integration issues. Evidence below is labelled **[verified]** (re-run or read by the
coordinator after merge) or **[reported]** (measured by the implementing agent on its branch, not yet re-run on the
merged tree).

## A. Changes from plan revision 1 to revision 2

The first audit refuted five claims of revision 1. All were checked against the code and bench16 data and are now
corrected in revision 2:

| Revision 1 claim | Fact (checked) | Revision 2 response |
|---|---|---|
| In-place recoveries are exportable | `_retry` incremented `rewinds`, so `student_eligible` was false | S2: separate `retries` / `rewinds` counters |
| Capture can be keyed on the predicted phase (`show_hold`) | 28 wrist captures at phase `approach`, 9 top captures at the `show_hold`->`show_move` boundary; capture is triggered by skill code | R3 / S0: student executes motion **segments**; the harness captures when a segment ends |
| Cameras are randomized per seed | `sample_case` overwrote top and front cameras with constants; no layout prior | S3: explicit options `prior` and `cameras`, realized distribution reported |
| Bottom presentation share is 1/3 | 4/9 with the proposed mix (normal and face-5 defects reach the bottom view) | S3 reports and tests 4/9 |
| 2000 episodes in about 3.5 h | `--smooth-for-vla` runs serially; linear estimate about 6.7 h | Pilot of 300 first; scale only if the student reaches >= 0.7x teacher success |

Further additions in revision 2: segment-boundary masks for action chunks (S7), teacher restricted to
student-observable signals (S1, R1), a symmetry-aligned grader that checks the decision crop (S6), radians as the
canonical action unit (R4), a sealed test split (R5), equal-budget 2x2 ablations plus an ACT vs SmolVLA backbone
comparison (S12), and the S0 replay gate before any data collection.

## B. Implementation status

| Step | Status | Commit(s) |
|---|---|---|
| S0 student interface replay gate | merged (bench16 part); perturbation part needs S5 | `ea64833`, `685ccc8`, merge (see B.4) |
| S1 student inputs / teacher signals | merged (code part); D1 awaits the owner | `c269569`, merge (see B.4) |
| S2 expert fixes | merged | `597cb87`, merge `2bdb6ea` |
| S3 case generator and splits | merged | `2ac77da` |
| S6 grader | merged | `703e674`, merge `98d893b`, integration fix (see B.4) |
| S4, S5, S7-S13 | not started | - |

### B.1 S2: expert fixes before collection

Files: `agent/twin_mujoco.py`, `agent/twin_record.py`, `agent/twin_resume.py`, `agent/step_mcp.py`,
`agent/twin_mcp.py`, `agent/twin_graph.py`, `joints.py`, `scripts/bench_agent.py`; tests `tests/test_expert_fixes.py`,
`tests/test_twin_agent.py`.

1. **Counters.** `_retry` increments `self.retries`; only `restore_checkpoint()` (the physics rewind, now its own
   method for S5) increments `rewinds` and opens a new branch. `evaluate()` reports both. `student_eligible` and the
   exporter (`student_frames`) require `rewinds == 0` only, so an episode with in-place retries is one continuous
   branch and exportable as a recovery demonstration.
2. **Held checkpoints.** `_save` no longer returns early while the brick is held. The grasp is physical (no weld), so
   the MuJoCo integration state already carries it; `Checkpoint` gains defaulted fields (`retries`, `held` = lift pose
   and grasp roll, `agent` = believed grip-from-brick, contact-filter state, depth-noise RNG state, park spot,
   expected yaw) plus three expert scalars (previous joint reading, lost-brick counter, blocked-gripper direction).
   Older checkpoint files still load. No new automatic save points were added; today `_save` is called only after
   `prepare` and after `scan_wrist` (brick on the table), so existing flows are unchanged **[verified by reading the
   call sites]**.
3. **Units.** The exporter writes `observation.state` and `action` in radians (float32). `joints.JOINT_RANGE_RAD` is
   the SO-101 MJCF joint range, which the expert clips every command to, mapped affinely onto [-1, 1]
   (`rad_to_unit` / `unit_to_rad`); the range and formulas are written to the dataset's `meta/source.json` under
   `joint_units`.

Gate evidence:
- Frame-192 fault case (realistic, case `[600001, false, 5]`, placeholder detector): before `SUCCESS, rewinds=1,
  student_eligible=False`; after `SUCCESS, rewinds=0, retries=1, student_eligible=True`, branch 0. **[reported;
  covered by a test in `tests/test_expert_fixes.py`, which passes after merge: verified]**
- Held checkpoint: save after the bottom view while holding, continue, restore, re-step: qpos, qvel, ctrl, act,
  targets and trajectories bit-identical in realistic and truth. **[reported; test passes after merge: verified]**
- Round trip rad -> unit -> rad over all 22,828 bench16 expert commands: max error 2.2e-16 rad, all values in
  [-1, 1]. **[reported]**
- bench16 realistic 16/16 SUCCESS, 16 eligible, rewinds 0, retries 0. **[reported; 16/16 re-verified on the
  merged tree, see B.6]**

Deviations and choices the reviewer should judge:
- **One shared recovery budget.** `Policy.max_rewinds` bounds `rewinds + retries` (new `recoveries` property).
  Rationale: `truth` only rewinds and `realistic` only retries, so bench16 behaviour is unchanged. Consequence: the
  name `max_rewinds` is now a misnomer, and the decision-context budget keys were renamed to
  `recoveries_used` / `recoveries_left`, which changes the LLM decision prompt slightly.
- **Normalization range = MJCF joint limits**, not the empirical expert span. It covers every expert command by
  construction (commands are clipped to it), but measured joints can sit slightly outside the soft limits, so
  normalized values just beyond +-1 are possible.
- **Known gap (corrected after review 1):** the earlier claim that a cross-process restore (`twin_resume`) differs
  only on its first frame is **withdrawn**. The reviewer showed that a restore from disk into a fresh instance loses
  the brick belief and changes the empty-bowl reference depth (145,596 pixels), and the difference persists to the
  placement check (green-bowl fill count 1,411 vs 1,261); `_save()` also raises for a student-held brick
  (`held.roll=None`). Both are being fixed (see section E). In-process restore is exact. `ContinuousCarryExpert.smoothed_moves` still resets on a rewind (unchanged behaviour).

### B.2 S3: case generator and splits

Files: `agent/twin_mujoco.py` (`sample_case`, `case_cameras`, `load_suite`, `_layout_prior`), `agent/twin_mcp.py`
(suite loading), new `scripts/vla/make_suites.py`, `config/vla/{train,val,test}.json`, `config/vla/SUITES.md`,
`config/vla/report.json`, `tests/test_vla_suites.py`.

- `sample_case(seed, positive, face, options=None)`; options `prior` (real-layout prior JSON, e.g.
  `config/sim/prior_grasp.json`, 35 real first frames) and `cameras` (`"fixed"` default or `"jitter"`, decision D3:
  top camera +-1 cm, yaw 2 deg, tilt 1.5 deg, roll 1 deg, focal +-2 %, plus the front camera, drawn from a separate
  random stream so the rest of the layout is unchanged). No options reproduces today's layouts exactly; a test
  checks all 16 bench16 cases. `load_suite` reads the old array format and a suite object with suite-level options.
- Default suites (no prior, fixed cameras): train 300 (seeds 1,100,001-1,100,300), val 60 (1,200,001-...), sealed
  test 100 (1,300,001-..., `sealed: true`). Disjoint; none of the 83 seeds under `config/agent_bench`.
- Bottom presentations **expected from the class labels**: 133/300, 26/60, 45/100 (4/9; the actual count depends on
  misses, early stops and failures and will be taken from the pilot's event logs; sizes are not multiples of 9, classes
  allocated by largest remainder).
- Realized train layouts: brick x 0.148-0.249 m, y -0.130-0.121 m, yaw roughly uniform over +-90 deg, 59/300 bricks
  2x4, five colours about 60 each, cameras constant.

Gate evidence: 25 new tests plus the agent tests pass **[verified: 37 passed in the worktree, and again after
merge]**.

Open owner decisions surfaced by S3:
- **D3 camera jitter** is implemented but off. In `realistic`, `TwinMujoco.__init__` already hand-eye calibrates
  every episode on its own layout; a 6-seed spot check gave 0.9-2.1 mm and 0.08-0.25 deg calibration error
  **[reported]**. The detector and bottom-view threshold were tuned on the fixed cameras, so a jittered suite needs
  an expert run before collection.
- **Real-layout prior** is off. With it, bricks sit at x of about 0.27-0.38 m versus at most 0.25 m uniform, and the
  pink bowl up to 0.46 m away; the expert has never been run there.

### B.3 S6: grader

Files: new `agent/vla_grader.py`, `scripts/vla/grade.py`, `config/vla/grader_reference.json`,
`tests/test_vla_grader.py`, `tests/data/vla_grader/realistic-600001-face0/` (50 KB real trace); changed
`agent/twin_record.py`, `agent/twin_mujoco.py` (`_capture` only), `agent/step_mcp.py` (`SIMULATOR_KEYS`),
`docs/ARMIGO_AGENT.md`.

Checks (each `pass`, `fail` or `unavailable`; `unavailable` never counts as a pass):
1. **Eligible:** SUCCESS and `student_eligible`; `rewinds` / `retries` reported.
2. **Decision crop with symmetry alignment.** Belief face ids are mapped to true ids by the footprint symmetry
   (90 deg for 2x2, 180 deg for 2x4) nearest the believed vs true brick orientation (old traces: layout yaw, then
   face pixels). Defective label: the `placement_evidence` view must show >= 800 px (wrist) / >= 400 px (top) of
   the true defect face and >= 25 px of the cross. Normal label: every true face must be truly visible in a clear
   view that claims it.
3. **Belief error** per scored view: <= 6 mm (wrist, brick on table), <= 12 mm (top, brick in hand) = 1.5x the
   bench16 maxima (3.4 / 7.9 mm).
4. **Collisions** on every frame of every branch, including `unknown` episodes and recoveries, no phase windows:
   robot links <= 3 N, jaws against table / bowls / clutter <= 2 N.
5. **Smoothness:** per-phase jerk RMS divided by the expert's per-phase median, frame-weighted, for commanded and
   measured joints; must be <= the expert p95 (1.32 / 1.225 from 95 bench16 episodes). A whole-episode jerk was
   rejected because it is class-biased (defective episodes skip the smooth bottom presentation).

Recording changes (additive, scoring only): `view_truth` gains `true_cross_px` and the true and believed brick
quaternions and is emitted in all sensing modes; each branch's `actions.npz` gains per-frame substep peaks
`contact_robot_n` and `contact_jaw_env_n` from the cell's observer hooks. The three new view keys were added to
`step_mcp.SIMULATOR_KEYS`, so the realistic step-MCP leak check strips them **[verified by reading the diff]**.

Gate evidence **[reported]**:
- Old bench16 traces: strict pass rate 0 (collisions and cross visibility unavailable). Symmetry alignment explains
  the audit's 12/37 raw face-set mismatches: **0 remain after alignment**.
- New bench16 realistic traces (S6 branch, before S2 was merged): 16/16 sorted, all checks available, overall pass
  0.94; every defect class 1.0, normal 0.88. The single failure is the expert's own measured smoothness on case
  600006 (1.2265 vs p95 1.225).
- Contact recorder sanity: bench16 expert contact peaks 0.0 N; pushing the jaws into the table logged 12 N.

Reviewer notes: a p95 threshold rejects the expert's own top 5 % by construction, and with bench16's 8 layouts that
tail lands on one or two layouts; the 0.8x per-class gate is only meaningful on the 300-episode pilot. Each
`_capture` now renders one extra segmentation view for `true_cross_px` (cost not yet measured).

### B.0 S0: student interface replay gate

Files: new `agent/student_rollout.py`, `scripts/vla/replay_gate.py`, `tests/test_student_rollout.py`; hooks in
`agent/twin_mujoco.py`; additive columns in `agent/twin_record.py`.

- **Segments.** `Segment(kind, params, instruction)`; the instruction is the VLA task string. 12 kinds: `prepare`,
  `scan_view` (orbit / hover view k of n, planned faces, target pose), `return_ready`, `grasp`, `lift`,
  `present_bottom`, `return_to_lift`, `lower_on_table`, `release_on_table`, `carry_to_bowl` (with the bowl),
  `release_in_bowl`, `let_go` (recovery). Five are carried-brick segments with a lost-brick monitor.
- **`StudentTwin(TwinMujoco)`** overrides only `_execute(generator)`: each expert motion generator is mapped to its
  segment by function name and arguments, closed unrun, and replaced by `run_segment` (observe -> `policy.act` ->
  joint-range clip -> `_apply_target`, i.e. record, command filter, `Episode.step`), until `segment_done`, a frame
  budget, or a lost brick. All graph-side skill code runs unchanged (scan planning and skipping, clearance checks,
  `_capture`, early stop, perception, placement checks, `_retry`), so S1 / S2 changes reach the student
  automatically. Three generators contain decision points (grasp-and-lift, park, release); the controller reproduces
  that logic between segments (clamp check, carried-brick belief at the grasp, lost-brick count, put-down yaw).
- **Hooks in `twin_mujoco.py`:** `_execute` now calls `_observe()` and `_apply_target()`, a pure extract-method of
  the existing loop.
- **`ExpertTracer(TwinMujoco)`** labels the unchanged expert with the same mapping (composite generators split by
  phase changes and carry-watch frames), so expert and student instructions match by construction.
- **`ReplayPolicy`** replays a trace's float64 actions and segment ends and checks every joint reading bitwise.
- **Recording:** `segment` / `segment_end` events; `actions.npz` gains `segment`, `state_f64`, `action_f64`. The
  `segment_done` label is attached to the first observation of the next segment.

Gate **[reported on the S0 branch]**: bench16 realistic 16/16 PASS (8 normal -> green, 8 defective -> pink, 897-1856
frames, 7-10 segments each). Exact: physics telemetry digest and every joint reading, capture frames, detector
decisions, early stops, segment kinds / params / starts / ends, destination, the `evaluate` dict, the whole event
stream. One case also passes with video on. Re-run on the fully merged tree: 16/16 PASS **[verified]**, see B.6.

Documented exceptions the reviewer should judge:
- **Renderer non-determinism.** The MuJoCo renderer is not bitwise repeatable for one state (shadows +
  multisampling): 12 renders of one state gave 8 distinct wrist images, off by 1 grey level on up to 46 of 307,200
  pixels; physics stays bitwise identical, and two plain expert runs show the same effect. Captured images are
  therefore compared within <= 1 level on <= 0.1 % of pixels; detector scores are compared exactly only when the crop
  is bitwise equal, otherwise the score delta is reported and the **decision** is still checked exactly. 9/16 cases
  had one or two such views; max score delta 0.0032; no decision flipped. A view scoring near the threshold could
  flip; removing the noise means changing render settings, which also changes the detector's input distribution.
- `smoothness.clipped_frames` is excluded from the `evaluate` comparison (expert 1, replay 0): it counts raw generator
  targets the expert clipped, which a replay of already-clipped targets cannot see.
- Not yet covered: the perturbation / in-place-recovery half of the gate (needs S5; `ReplayPolicy` rejects
  multi-branch traces); the `let_go` path and the lost-brick / clamp-failure monitors are untested in MuJoCo because
  bench16 has no recoveries; a learned student emits no phase label, so each segment kind's default phase is used,
  which shifts the health-scoring windows (to be handled in S6 / S10).

### B.3a S1: student inputs and teacher signals

Files: new `agent/bowl_clearance.py`; `agent/twin_mujoco.py` (clearance-related lines only); `tests/test_twin_leakage.py`
(3 tests); new `docs/vla/STUDENT_INPUTS.md`.

- **Bowl truth removed from the `realistic` clearance checks** (corrected after review 1: this was not the last
  truth read; the grasp-axis choice and wrist view planning still read the true brick size, see section E). The grasp-candidate and path clearance checks
  (`expert._clearance`, `_straight_clear`) now measure the robot against the nominal bowl cross-section revolved
  around the **perceived** bowl centres, dilated by 5 mm (about 2x the largest bench16 bowl localisation error,
  2.6 mm). Robot collision geoms are sampled every 2 mm and the sampling slack (about 1.4 mm) is subtracted. In
  `realistic` the expert class is wrapped by a subclass whose `_clearance` uses the proxy, so restore / retry paths
  that rebuild the expert keep it. `truth` keeps the simulator geometry. `realistic` now refuses layouts with clutter
  (clutter is not perceived; bench16 and the S3 suites have none).
- Proxy check **[reported]**: on 267 random poses near the bowls of three layouts, no pose where the true geometry
  collides and the proxy says clear; on average 1-3.5 mm more conservative. A plain cylinder + bounding boxes was
  15-20 mm too conservative and changed 4 grasp choices, so it was replaced.
- Leak test: after perception the simulator's green bowl is moved 0.4 m; the realistic clearance value does not
  change, and `_env_geoms` is patched to raise if anything reads simulator bowl geometry. Moving the perceived bowl
  does change the decision; the truth variant follows the moved simulator bowl.
- bench16 realistic **[reported]**: before and after both 16/16, 0 unintended collisions, 22,828 total frames,
  identical frame counts per episode, 0 recoveries.
- `docs/vla/STUDENT_INPUTS.md`: inventory of every signal the realistic expert reads (file:line), whether an
  RGB + joint-state student can infer it, and D1 options A (add as student inputs), B (restrict the teacher), C
  (hybrid). Default recommendation (hybrid): add the previous action (6 values) as a student input; add gripper
  effort only if the real SO-101 driver reads gripper load at 30 Hz, otherwise drop effort from the teacher's contact
  estimate (record `gripper_cmd` and `effort_frac` in `actions.npz` now either way); keep the student RGB-only and
  measure the depth gap; 2 observation frames plus the S7 per-segment frame index. The export is not yet changed.
- Known gaps: joint state is the exact simulated position (real encoders are quantised); the exported `top_depth` is
  a clean height image, not the noisy depth the teacher reads; the proxy ignores the held brick (as the old clearance
  did).

### B.4 Integration by the coordinator

- Merge order S3 -> S2 -> S6 -> S1 -> S0. One textual conflict: `EpisodeTrace._finish_branch`, where S6 (contact
  columns) and S0 (`segment`, `state_f64`, `action_f64`) both add `actions.npz` columns; resolved by keeping both.
- **Fix:** `EpisodeTrace.start` replaced any `substep_observer` / `frame_observer` already installed on the cell
  (`scripts/catalog_record_mujoco.py` installs its own on a separate cell, so nothing broke today). It now chains
  existing observers.
- Tests on the merged tree: `test_vla_grader, test_expert_fixes, test_vla_suites, test_twin_agent, test_step_mcp,
  test_mujoco_session_resume, test_twin_leakage, test_twin_service, test_defect_kinds`: **101 passed [verified]**.
- End-to-end acceptance on the merged tree (bench16 realistic, then the grader): 16/16 sorted, 15/16 graded pass;
  see B.6.

### B.5 Not started

S4 (rollouts), S5 (perturbation branches), S7 (segment-masked export), S8 (CUDA env), S9-S13.

### B.6 End-to-end acceptance on the merged tree

**[verified]** bench16 realistic (16 cases, 2 workers, headless) on the merged tree (S2 + S3 + S6), then
`scripts/vla/grade.py` on the fresh traces:

- 16/16 SUCCESS, all labels correct.
- Grader: overall strict pass 15/16 (0.94); every defect class 1.00, normal 0.88 (7/8). No check unavailable.
- Failures per check: eligible 0, decision crop 0, belief error 0, collisions 0, smoothness 1 (case index 10, a
  normal brick: the same expert smoothness tail the S6 agent reported on its own branch).
- Face alignment: 37 views, 12 raw mismatches, 0 after symmetry alignment (all 37 aligned by orientation).

So merging S2 did not change bench16 outcomes or grades relative to the S6 branch run. **[verified]** After also
merging S1 (perceived-bowl clearance) the same run gives identical results: 16/16 SUCCESS, 15/16 graded pass, the
same single smoothness failure, 12 -> 0 face mismatches; 124 tests passed.

**[verified]** S0 replay gate on the fully merged tree (S0 + S1 + S2 + S3 + S6, commit `10f1933`):
`scripts/vla/replay_gate.py` on all 16 bench16 realistic cases (two processes of 8): **16/16 PASS**, 887-1856
frames, 7-10 segments per episode. Six cases had render noise on 1-3 views (max 1 grey level, at most 0.026 % of
pixels); max detector score delta 0.0017 (case 8); no decision changed. 130 tests passed on the merged tree
(`test_student_rollout` included).

## C. Questions for the reviewer

1. Is one shared recovery budget (`rewinds + retries <= max_rewinds`) acceptable, or should retries get their own
   budget before S5 generates recovery demonstrations?
2. Is the MJCF joint range the right normalization for training, given that most of it is unused by the expert
   (e.g. a per-joint empirical range would give finer resolution but needs a fixed reference set)?
3. Grader thresholds: 1.5x the bench16 maxima for belief error and the expert's p95 for smoothness are derived from
   8 layouts. Should they be re-derived on the pilot before they gate the dataset, and from which split?
4. The decision-crop check requires the true cross to be visible (>= 25 px) in the triggering crop. Is a pixel count
   on the segmentation the right criterion, or should it be cross visibility inside the detector crop bbox
   specifically?
5. Collisions use substep peaks per frame. Is a per-frame peak threshold sufficient, or should impulse / duration
   also gate?
6. S1: is a fixed 5 mm margin around the perceived bowls justified by 16 episodes of localisation error, or should
   the margin scale with the localisation uncertainty per episode? Is the hybrid D1 recommendation sound?
7. S0: is the renderer tolerance (<= 1 grey level on <= 0.1 % of pixels, decisions exact, scores exact only for
   bitwise-equal crops) acceptable for a gate, or should rendering be made deterministic (shadows / multisampling
   off) at the cost of changing the detector's input images? Is matching the expert's generators by function name a
   sound basis for segment labels, or too brittle as skills evolve?
8. Anything in revision 2's ordering (S0/S1/S2 before a 300-episode pilot) that the implemented pieces change?

## D. Reproduce

	.venv/Scripts/python -m pytest -q tests/test_student_rollout.py tests/test_vla_grader.py tests/test_expert_fixes.py tests/test_vla_suites.py \
	  tests/test_twin_agent.py tests/test_step_mcp.py tests/test_mujoco_session_resume.py tests/test_twin_leakage.py
	.venv/Scripts/python scripts/vla/make_suites.py --out <dir>      # regenerate suites and report
	.venv/Scripts/python -m digital_twin_armigo.agent.twin_mcp run --sensing realistic \
	  --suite config/agent_bench/bench16.json --detector results/agent-bench/sup-v7bot-local.json \
	  --headless --workers 2 --out <dir>/runs
	.venv/Scripts/python scripts/vla/grade.py <dir>/runs
	.venv/Scripts/python scripts/vla/replay_gate.py --out <dir>/gate --cases 0 1 2 3 4 5 6 7   # and 8-15

## E. Review 1 response (in progress)

The independent review of this report (commit `c654838` / `10f1933`) confirmed the core numbers (bench16 16/16 sorted,
15/16 graded pass, face mismatches 12 -> 0 after alignment, replay gate 16/16) and found the following. Each is being
fixed with a failing test first; this section will be updated with the fixes and evidence.

| Priority | Finding | Fix under way |
|---|---|---|
| P1 before collection | The production entry (`twin_mcp run`) records no segment labels: all 22,828 frames have `segment == -1`; only the replay gate's `ExpertTracer` labels segments | One labelling implementation used by every recording entry; segment id / kind / instruction / `segment_done` / `chunk_valid_len` in the LeRobot export; a smoke batch through the exact pilot command path, read back and checked |
| P1 before grading | `true_cross_px` is counted on the full frame, not in the detector crop; a 1x1 crop still graded 499 cross px and passed | Crop-space truth masks with the same region and transform as the RGB crop; threshold resolution stated; negative tests |
| P1 before grading | Truncated or NaN contact columns (and missing branches) can pass the collision check | Length, finiteness and branch-presence checks; otherwise `unavailable` |
| P1 before teacher ablations | `realistic` grasp-axis choice and wrist planning read the true brick size; the online student policy can read planner diagnostics in `Segment.params` (`target_q`, predicted faces, bottom pixels) | Believed brick size in realistic planning; one shared policy-visible field list for the online policy and the export |
| P1 before recovery branches | Fresh-instance restore from disk is not faithful; `_save()` crashes for a student-held brick | Persist all state later behaviour depends on; fresh-instance restore tests for expert- and student-held bricks |
| P2 replay criterion | Full-image render noise exempted byte-identical crops from exact score comparison | Separate full-image and crop flags; exact scores (all heads) for identical crops |

Adopted recommendations without code yet:
- Grader thresholds will be re-derived on a calibration subset of the training layouts at a chosen joint rejection
  rate and frozen before validation and the sealed test (the current p95 pair rejects 9/95 = 9.5 % jointly).
- S5 perturbation branches will keep their own accounting, separate from the online recovery budget, so a
  `restore_checkpoint()` used to create a perturbation start cannot consume the single in-place retry; recovery
  success will be reported over all perturbation attempts, split into detected / recovery started / sorted.
- Next acceptance step before the 300-episode pilot: one small batch (normal, side-face defect, face-5 defect)
  through the exact collection path, checked for segment labels, crop-space truth and complete contact records; and
  one full "visible perturbation -> detection -> in-place recovery -> replay -> export" trajectory.
