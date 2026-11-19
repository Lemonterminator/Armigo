# Student inputs and teacher signals (VLA plan step S1)

Inventory for decision D1 of `docs/VLA_DISTILLATION_PLAN.md` (principle R1: the teacher should act only on what
the student can observe or infer). It lists every signal the `realistic` expert reads when it chooses an action or
detects a failure. For each one it says whether a student that sees **[top RGB, wrist RGB, optional front RGB, joint
state]** could infer it, and what each D1 option would change. Line numbers refer to branch `calibrated-brick-pose`
after the S1 clearance change.

## 0. What S1 already changed

The grasp and path clearance checks no longer read the simulator's bowl and clutter collision geoms in
`realistic`. They use the nominal plate cross-section, revolved around each **perceived** bowl centre and dilated
by a 5 mm margin. It is measured against the robot's moving collision geoms: mesh hulls, and boxes for the rest
(`agent/bowl_clearance.py`; the expert class is wrapped at `agent/twin_mujoco.py:368-369`; `_straight_clear` is at
`:868`). A `realistic` twin now refuses layouts with clutter, because clutter is not perceived. `truth` behaves as
before. On bench16 `realistic` the result stays 16/16, with frame counts identical to the code before the change in
every episode and 0 unintended collisions. A plain-cylinder proxy with box bounds also passed 16/16, but it was
15-20 mm more conservative next to bowls and changed 4 grasp choices. Test: `tests/test_twin_leakage.py`.

## 1. Two levels of signals

Under R3 (hierarchical student), the graph and a small controller keep the **segment-level** decisions: which view
comes next, the detector label, early stop, the placement check, and recovery. The student executes the motion
**inside** a segment. A segment-level signal therefore never needs to become a student input. It only has to be
computable on the real rig by the controller. The column "Level" marks this: **M** = read inside a motion segment
(the student must reproduce its effect); **C** = read by the controller between segments.

"Inferable" means:
- **yes**: it is a student input, or a deterministic function of the student's observation history;
- **approx.**: it can be estimated from the images, but not exactly as the teacher sees it;
- **history**: it needs the student's own previous action(s) or more than one frame of state;
- **no**: the student has no access to it.

## 2. Inventory

| # | Signal | Read at (file:line) | Used for | Level | Inferable from [top RGB, wrist RGB, front RGB, joint state] |
|---|---|---|---|---|---|
| 1 | Arm and gripper joint positions `obs["q"]` (exact `qpos`; the encoder quantisation in `physics.py:239` acts on commands, not readings) | `mujoco_cell/episode.py:141`; `physics_expert.py:282, 348, 636, 639, 832, 851` | servo integrators, blocked-gripper back-off, show poses | M | **yes** (joint state). The real reading is quantised (4096 ticks); the sim reading is not |
| 2 | Gripper-frame position `obs["tcp"]` (forward kinematics of `q`) | `episode.py:149`; `physics_expert.py:552, 627, 899, 922`; `table_relocation.py:293, 313, 333` | Cartesian servo, rise / retreat / carry lines | M | **yes** (FK of joint state) |
| 3 | Gripper frame for the in-hand belief, `_grip_pose` (FK, `site_xpos/xmat`) | `agent/twin_mujoco.py:428-434, 995` | carried brick = FK x grasp transform | M | **yes** (FK of joint state) |
| 4 | Previous gripper command `obs["gripper_cmd"]` (= `q_cmd[5]` after the command filter: latency, backlash, quantisation) | `episode.py:145`; `agent/sensing.py:29-33`; `mujoco_cell/gripper_state.py:48-73` | jaw-contact estimate (clamp, lost brick), BLOCKED detection | M | **history**: a deterministic function of the policy's own previous actions and the known filter. Without previous-action input, only approx. (from wrist RGB and the gripper reading) |
| 5 | Gripper effort `obs["effort_frac"]` (gripper actuator torque / limit; the servo "load" register) | `episode.py:146`; `sensing.py:33` (`effort_frac >= 0.5`); `gripper_state.py:64-68` | jaw-contact estimate, BLOCKED detection | M | **no** (approx. from `q[5]` vs. command: a position servo's load grows with the error) |
| 6 | Gripper speed (difference of successive `q[5]`) | `sensing.py:31`; `gripper_state.py:57` | "settled" condition of contact / BLOCKED | M | **history** (two frames of joint state) |
| 7 | Estimated jaw forces `obs["jaw_normal"]` = GripperContact(4, 1, 5, 6) | `twin_mujoco.py:550-552`; `physics_expert.py:362` (`_dual`), `616-619` (clamp), `863-865` (lost after 10 frames) | clamp verified, brick lost while carrying | M (clamp, lost), C (lost aborts the skill) | as 4 + 5 + 6 |
| 8 | Brick velocity `obs["brick_vel"]` | `twin_mujoco.py:549` (forced to zero); `physics_expert.py:617, 917`; `table_relocation.py:328` | clamp "still", settle after release | M | **yes**: a constant (zero), so the settle and clamp loops reduce to fixed frame counts |
| 9 | Brick pose on the table: `locate_brick` on **noisy top depth** through the calibrated camera | `twin_mujoco.py:456-465` (noise), `471`; `perception.py:128` | grasp target and roll, orbit / hover scan plans, bottom-view planning, detector crop | M (grasp approach), C (scan plan) | **approx.**: top RGB shows the brick on a known table plane with a fixed camera. The student sees RGB, not the teacher's metric depth. The exported `top_depth_v2` is a clean layered height image (`mujoco_cell/cell.py:543-556`), not the noisy depth the teacher reads |
| 10 | Yaw continuity after a put-down (`_expected_yaw`) | `twin_mujoco.py:396, 476-480, 892` | keeps the face numbering across a park | C | history (the controller's memory) |
| 11 | In-hand brick belief (`_grip_from_brick`, fixed at grasp time) | `twin_mujoco.py:436-445, 995` | carry and placement targets, bottom-view border, detector crop | M | **history**: needs the brick pose at the grasp (9) plus FK (3). Visible in wrist RGB (approx.) |
| 12 | Perceived bowl centres `self.bowls` (colour + 3D rim fit in top RGB + depth) | `twin_mujoco.py:409-425` (`locate_bowls` at 412); `perception.py:218` | placement target `ex.bowl_xy` (`:1240`), brick-mask exclusion (`:471`), **clearance proxy** (`bowl_clearance.py`, after S1), placement check (`:1267`) | M (carry to bowl, clearance), C (check) | **approx. / yes**: bowls are large coloured objects in fixed top RGB, also visible in front RGB |
| 13 | Empty-bowl baseline depth, and the fill test after a release | `twin_mujoco.py:413, 616, 1264-1271`; `perception.py:265` | `PLACEMENT_UNVERIFIED` | C | not needed by the student (the controller check runs on the rig's own sensor) |
| 14 | Hand-eye calibration of the top camera (`_cam_k`, `_cam_est`) | `twin_mujoco.py:402`; `mujoco_cell/handeye_rig.py:132` | every top-camera localisation (9, 12, 13) | C (static per episode) | **implicit** when cameras are fixed (D3 = fixed): learned once into the weights. With camera randomisation it is **no** |
| 15 | Detector score on the believed crop | `twin_mujoco.py:826-830`; graph `decision` at `:943` | label, early stop of the scan | C | not a student input under R3 (segment instruction comes from the graph) |
| 16 | Believed visible faces / crop box (from the believed brick) | `twin_mujoco.py:795-806` | which faces a view clears | C | not needed under R3 |
| 17 | Clearance: robot vs. table (`path_min_z`, FK and believed brick) and vs. bowls (12) | `twin_mujoco.py:868-874`; `mujoco_cell/inspection_planning.py:180` | path rejection in scan, park, bottom view, place | C (plan selection) | the plan is selected by the controller. The motion the student imitates follows it, and its inputs (FK, perceived bowls) are inferable |
| 18 | Previous expert target `ex.last` (start of every min-jerk move; the servo integrators' base) | `physics_expert.py:304, 316, 340`; `twin_mujoco.py:203` (`sweep`) | trajectory shape | M | **history** (exact with previous-action input; approx. from joint state, since the tracking lag is about 0.1-0.27 rad at peak) |
| 19 | Servo integrator state (`corr` in `_hold_servo`, `_servo`) | `physics_expert.py:342-352, 547-557` | holds that remove sag / backlash | M | **history** (integral of joint error: needs previous actions, or several frames of state) |
| 20 | Time inside a segment (hold lengths, chunk pauses, `servo_max_frames`, `clamp_still_frames`, `settle_frames`, `lost_frames`) | `physics_expert.py:116, 129, 156-166, 327, 338-341, 551`; `twin_mujoco.py:938, 1024` | when a hold or wait ends | M | **history / approx.**: action chunks reproduce fixed holds open-loop; the `segment_done` head (S9) learns the end from images and stillness |
| 21 | Budgets: `wrist_count`, `bottom_count`, `rewinds`, `total_frames` vs. `MAX_TOTAL_FRAMES` | `twin_mujoco.py:539, 622, 626, 687, 1103` | which skill is allowed, recovery allowed | C | not needed under R3 |
| 22 | Retry "brick moved" test (re-localised pose vs. the checkpoint belief, 5 mm / 3 deg) | `twin_mujoco.py:1207-1212` | keep or discard earlier views | C | not needed under R3 |

Not read by the `realistic` expert any more: true bowl positions and collision geoms (removed by S1), true brick
pose (only through the belief), true contact forces, true brick velocity. Ground truth still appears only in scoring
events (`view_truth`, `error_*_mm`), in `actual_destination` (read by `evaluate` only) and in the final `evaluate`.
The only remaining read of `layout.green/pink` before motion is the initial value of `self.bowls`. Perception
overwrites it in `_sense_bowls`, or the episode stops with `BOWL_NOT_FOUND` before any motion.

## 3. Signals that break R1 for an RGB + joint-state student

Only the motion-level signals (M) matter for the student:

1. **Gripper effort (5)**: not observable.
2. **Previous command / previous target (4, 18, 19)**: observable only with action history.
3. **Gripper speed (6) and time in segment (20)**: need more than one frame of state, or action chunking.
4. **Noisy metric depth for the brick pose (9) and the in-hand belief (11)**: approximately inferable from RGB.
   The teacher's localisation (3.0 mm median on the table) is not guaranteed to be reproducible from RGB.

## 4. D1 options

### Option A: add the signals as student inputs (the teacher stays as it is)

| Signal | Export change (`agent/twin_record.py`, `lerobot_export.py`) |
|---|---|
| Previous action (4, 18, 19) | none in recording (it is `action` shifted by one frame; frame 0 = the initial command `INITIAL_Q`). `student_frames` (`twin_record.py:249-296`) adds `observation.prev_action` (6, same units as `action`); `export_lerobot` (`:299-330`) declares it in the features passed to `lerobot_features` (`lerobot_export.py:29-43`, a second vector feature next to `observation.state`) |
| Gripper command after the filter (4) | `TwinMujoco._execute` must pass `obs["gripper_cmd"]` to `record.command` (`twin_mujoco.py:560`; S2 also touches this area). `EpisodeTrace.command` (`twin_record.py:103-104`) and the `actions.npz` writer (`:213-216`) store a `gripper_cmd` column. `student_frames` emits it, e.g. inside `observation.gripper` (2: cmd, effort) |
| Gripper effort (5) | as the gripper command: an `effort_frac` column in `actions.npz`, emitted as `observation.gripper[1]`. Needs the real SO-101 driver to read the gripper servo's load register at 30 Hz |
| Gripper speed (6), time in segment (20) | no new column. Use `n_obs_steps >= 2` (or a velocity feature computed from `state`) in training, plus a per-frame `segment_frame` index (already planned with the S7 segment ids) |
| Teacher's noisy depth (9) | record the depth the teacher actually used (`_top_depth`, float metres plus noise). Today `top_depth` is the clean layered height image from `Cell.top_images` (`twin_record.py:160-165`). Export it as a lossless single-channel uint16 (millimetres) **image** feature (LeRobot video features are 8-bit RGB), e.g. `observation.images.top_depth_mm`. About +40-60 MB per exported episode; the student network needs a depth encoder |

### Option B: restrict the teacher to what an RGB + joint-state student sees

| Signal | Teacher change | Export change |
|---|---|---|
| Gripper effort (5) | `GripperContact` drops the `effort_frac >= EFFORT_MIN` term (contact = gripper reading held above its command by more than `GAP_RAD` and still); `GripperMonitor` likewise. Re-measure the agreement with true contact (today 98-99 %) and bench16 | none |
| Previous command (4, 18, 19) | replace `gripper_cmd` by the segment's nominal gripper target (open / `hold_g`), known from the segment instruction; keep the servo integrators (their effect on the target is bounded by `carry_joint_max_corr`) and accept them as approx. | none (or a segment-level `gripper_target` scalar with the S7 instruction) |
| Depth (9, 11) | localise the brick from top RGB (colour segmentation on the known table plane) instead of depth; keep depth only for the controller's placement check (13), which the student never imitates | none |
| Time in segment (20) | no change; rely on action chunks and `segment_done` | `segment_frame` with the S7 segment ids |

### Option C (hybrid)

Add the signals a real SO-101 provides for free, and restrict the teacher where the real rig cannot provide the
signal.

## 5. Recommendation (default unless the owner decides otherwise)

**Option C:**

1. **Add previous action** as `observation.prev_action` (6). Every real policy loop knows the command it sent. This
   makes 4, 18 and 19 functions of the student's history. The export change is local to `student_frames` and
   `export_lerobot`; there is no recording change.
2. **Gripper effort: add it if the real driver reads the gripper load at 30 Hz, otherwise restrict the teacher**
   (Option B row 5). Either way, record `gripper_cmd` and `effort_frac` in `actions.npz` now (cheap; S2 is already
   editing `_execute` / the recorder), so the decision does not require recollection.
3. **Depth: keep the student RGB-only for the pilot and do not add depth.** The teacher's depth is used at segment
   boundaries for the table localisation (9), where the brick is unoccluded in top RGB. Treat it as "approx.
   inferable" and measure it: localise the brick from top RGB on bench16 and compare with the depth belief error
   (3.0 mm median). If RGB stays within the grasp tolerance (about 5 mm), the gap is acceptable. Otherwise switch the
   teacher's table localisation to RGB (Option B row 9). Add depth as a student input only if D1 says the real top
   camera is a D415 **and** the chosen student family (D2) can take a depth image.
4. **Time in segment:** `n_obs_steps = 2` (gripper speed) and the S7 `segment_frame` / `chunk_valid_len`. No teacher
   change.
5. Keep all C-level signals (13-16, 21, 22) with the controller. They are computed on the real rig from the same
   sensors and never enter the student's inputs or its loss.

With this default, every motion-level teacher signal is a student input or a deterministic function of the
student's history, except the depth-based brick localisation (9, 11). That one is approximately inferable, and point
3 measures the gap explicitly. The LeRobot export is **not** changed in S1; the changes above wait for the owner's
D1 decision.
