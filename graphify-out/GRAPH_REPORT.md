# Graph Report - digital_twin_armigo  (2026-11-18)

## Corpus Check
- 172 files · ~464,334 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 2911 nodes · 7209 edges · 154 communities (135 shown, 19 thin omitted)
- Extraction: 87% EXTRACTED · 13% INFERRED · 0% AMBIGUOUS · INFERRED: 943 edges (avg confidence: 0.92)
- Token cost: 269,993 input · 0 output

## Community Hubs (Navigation)
- Archived Physics Diagnostics
- MuJoCo Session And Toy Sim
- Snapshot Tests
- Episode Metrics
- Execution Service
- MuJoCo Cell API
- Isaac Stage 3 Parallel
- Session Resume Tests
- Physics Expert Tests
- Execution Service Tests
- Cross-Engine Benchmarks
- Cameras And Views
- IK Pick-Place Expert
- Catalog Playback And Rendering
- Inspection View Tests
- Snapshot Store
- Anomaly Dataset Rendering
- Experiment Utility Scripts
- Inspection View Capture
- Supervised Detector Training
- Plate Bowl Blender Scene
- Show-Pose Policies
- Twin MuJoCo Skills
- Depth And Sensor Noise
- Gripper Blocked State
- Table Relocation Planner
- Heuristic Inspector
- Closed-Loop Expert Core
- Anomaly Train Eval
- Inspection LangGraph
- Detector Reports And Fakes
- Scene Layout
- Ledger Tests
- Lego Brick Blender Script
- Physics Expert Benchmark
- Catalog Store
- Operation Ledger
- Grasp Geometry Diagnostics
- Presentation Scoring
- Simulator Adapter
- Catalog Tests
- Sim Blend Upgrade
- Inspection Labels
- Brick-Level View Fusion
- Inverse Kinematics
- Catalog Capture
- Cross Defect Variants
- LLM Policy Prompting
- Episode Trace Recorder
- Ledger Leases
- Heuristic Episode Runner
- Expert Grasp Phases
- Plate Decomposition
- Runtime Errors
- Agent Graph Tests
- Armigo MCP Server
- Catalog Quality Checks
- Expert Place Phases
- Failure Animation Export
- Bowl Collider Candidates
- Catalog Web App
- Defect Kind Tests
- Gripper State Audit
- Detector Crop Utilities
- Inspection Planning
- Six-Face Twin Graph
- Anomaly Score Client
- Lego Random Color Detail
- Ledger Detail
- Score Server Detail
- Scene Detail
- Six-Face Agent Docs
- Data Catalog Docs
- Bowl Gate Checks
- Test Fuse Views Detail
- Graph Detail
- Bowl Collider Decision
- Physics Audit Reports
- Joint Calibration
- Catalog Episodes
- Local LLM Client
- Test Twin Agent Detail
- Twin Record Detail
- Physics Detail
- Scan Zone Detail
- Gripper Blocked Docs
- Inspection Relocation Docs
- Stage2 Expert Detail
- Test Agent Policies Detail
- Session Detail
- LeRobot Export
- Index Detail
- Run Detail
- Camera Detail
- Test Inspection Scorer Detail
- Recovery Protocol Docs
- Gates Detail
- Replay Dashboard
- Prepare Assets Detail
- Report Detail
- Failure Handoff Notes
- Report Detail 2
- Export Agent Logs Detail
- Jobs Detail
- Inspection Scorer Detail
- Inspection Views Detail
- Supervised Scorer
- Isaac Checkpoint Probe
- Run Heuristic Inspection Detail
- Twin Graph Detail
- Test Heuristic Episode Detail
- Ledger Detail 2
- Report Detail 3
- Heuristic Detail
- Heuristic Detail 2
- Physics Expert Detail
- Train Sup Detail
- Bowl Collision Docs
- Inspection Scorer Detail 2
- Test Runtime Service Detail
- Train Eval Detail
- Run Inspection Policies Detail
- Test Bowl Experiments Detail
- 2026-11-17-Bowl-Convex-Decomposition-Experiments Detail
- Report Detail 4
- Candidates Detail
- Serve Armigo Qwen Detail
- Top Camera Reach Detail
- Gripper Blocked State Detail
- Index Detail 2
- Inspection Scorer Detail 3
- Preview Detail
- Stage2 Keyframes Detail
- Cross Engine Report Detail
- Physics Expert Detail 2
- Scene Detail 2
- Test Shared Modules Detail
- Test Runtime Service Detail 2
- Heuristic Detail 3
- Heuristic Detail 4
- Heuristic Detail 5
- Heuristic Detail 6
- Heuristic Detail 7
-   Init   Detail
-   Init   Detail 2
- Inspection Scorer Detail 4
- Test Shared Modules Detail 2
- Shared Imports Detail
- Shared Imports Detail 2
- Shared Imports Detail 3
- Pyproject Detail
- Shared Imports Detail 4
- Frame 0000 Detail
- Shared Imports Detail 5

## God Nodes (most connected - your core abstractions)
1. `Cell` - 123 edges
2. `OperationLedger` - 83 edges
3. `ClosedLoopExpert` - 60 edges
4. `CameraPose` - 59 edges
5. `SnapshotStore` - 58 edges
6. `RolloutWorker` - 49 edges
7. `HeuristicInspector` - 45 edges
8. `ExpertParams` - 43 edges
9. `SnapshotCorruptError` - 43 edges
10. `JobState` - 40 edges

## Surprising Connections (you probably didn't know these)
- `Task 0: 用真实数据确定阈值` --references--> `ClosedLoopExpert`  [INFERRED]
  docs/superpowers/plans/2026-11-17-gripper-blocked-state.md → src/digital_twin_armigo/mujoco_cell/physics_expert.py
- `5. 采纳后新基线与待夹爪计划处理的清单` --references--> `GripperMonitor`  [INFERRED]
  docs/BOWL_COLLISION.md → src/digital_twin_armigo/mujoco_cell/gripper_state.py
- `1. 状态定义与输入信号` --references--> `GripperMonitor`  [INFERRED]
  docs/GRIPPER_BLOCKED_STATE.md → src/digital_twin_armigo/mujoco_cell/gripper_state.py
- `Global Constraints` --references--> `test_expert_acts_only_through_the_joint_target()`  [INFERRED]
  docs/superpowers/plans/2026-11-17-gripper-blocked-state.md → tests/test_physics_expert.py
- `What a snapshot holds` --references--> `evaluate_episode_health()`  [INFERRED]
  docs/RECOVERY.md → src/digital_twin_armigo/metrics.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Gripper BLOCKED staged detect-recover-prenarrow strategy** — docs_gripper_blocked_state_gripper_monitor, docs_gripper_blocked_state_gripper_states, docs_gripper_blocked_state_reactive_recovery, docs_gripper_blocked_state_open_ladder_prenarrowing, docs_superpowers_plans_2026_11_17_gripper_blocked_state_bit_exact_unblocked_invariance [EXTRACTED 1.00]
- **Recoverable MuJoCo execution stack** — docs_recovery_snapshot_store, docs_recovery_operation_ledger, docs_recovery_execution_service, docs_recovery_mujoco_session, docs_recovery_commit_protocol [EXTRACTED 1.00]
- **Closed-loop expert failures and their physics root causes** — results_failure_export_index_high_mass_seed29, results_failure_export_index_low_friction_seed111, results_failure_export_index_layouts_seed1000_idx32, results_failure_export_physics_diagnosis_report_box_cylinder_ccd_bug, results_failure_export_physics_diagnosis_report_torsional_spin_instability, results_failure_export_physics_diagnosis_report_jaw_blocked_by_rim [EXTRACTED 1.00]
- **Phase 1 audit fixes forming the v2-audited evaluator** — results_verification_physics_v2_audit_report_v2_audited_evaluator, results_verification_physics_v2_audit_report_bounce_out_vs_settle_timeout, results_verification_physics_v2_audit_report_tracking_vs_torque_split, results_verification_physics_v2_audit_report_substep_contact_sampling, results_verification_physics_v2_audit_report_strict_physics_config, results_verification_physics_v2_audit_report_compare_trajectories_resampling [EXTRACTED 1.00]
- **Six-face agent inspect-decide-sort flow** — docs_armigo_agent_armigo_mcp_server, docs_armigo_agent_inner_llm_graph, docs_armigo_agent_outer_policy_search, results_anomaly_report_supervised_v6_detector, docs_armigo_agent_six_face_label_rule, docs_armigo_agent_time_travel_checkpoints [INFERRED 0.85]
- **Isaac Sim Stage 1 verification camera views** — results_stage1_isaac_front_rgb, results_stage1_isaac_top_rgb_crop, results_stage1_isaac_top_depth_crop, results_stage1_isaac_wrist_rgb [INFERRED 0.85]
- **Pick-and-place keyframe sequence** — assets_blender_sim_scene_frame_0000, assets_blender_sim_scene_frame_0140, assets_blender_sim_scene_frame_0240, assets_blender_sim_scene_frame_0380 [INFERRED 0.85]
- **PatchCore simulated-crop anomaly report series (v2 -> v4-isaac)** — results_anomaly_v2_report, results_anomaly_v3_256_report, results_anomaly_v3_320_report, results_anomaly_v3_320_realtop_report, results_anomaly_v4_320_report, results_anomaly_v4_isaac_320_report [INFERRED 0.95]

## Communities (154 total, 19 thin omitted)

### Community 0 - "Archived Physics Diagnostics"
Cohesion: 0.08
Nodes (38): av, collections, dataclasses, hashlib, math, mujoco, numpy, platform (+30 more)

### Community 1 - "MuJoCo Session And Toy Sim"
Cohesion: 0.05
Nodes (29): MujocoSession, ndarray, One MuJoCo sorting-cell instance driven by a :class:`ControlProgram`, resumable…, ``pure_physics`` or ``assisted_compat``; assisted results must never enter a…, Per-frame logs since the episode started, in the layout…, Snapshot the complete session at the current control-frame boundary without…, Adopt ``state``. Everything is validated first; on any error this session is…, sha256_bytes() (+21 more)

### Community 2 - "Snapshot Tests"
Cohesion: 0.08
Nodes (52): MonkeyPatch, A snapshot could not be written, found, or trusted., No committed manifest exists for the requested snapshot id., SnapshotError, SnapshotNotFoundError, Recoverable execution: content-addressed snapshots, an append-only operation…, current_identity(), _fsync_write() (+44 more)

### Community 3 - "Episode Metrics"
Cohesion: 0.07
Nodes (49): 1. 审计问题、确定性反例与最小修复清单, 历史报告 `flick_slide` 文案勘误（【源码检查】）, aggregate_substep_contacts(), _brick_contained_in_bowl(), compare_trajectories(), EpisodeHealthMetrics, _estimate_time_shift_s(), evaluate_episode_health() (+41 more)

### Community 4 - "Execution Service"
Cohesion: 0.06
Nodes (30): Pieces, BudgetExceededError, The request would exceed the run's rollout or simulated-time budget, or the…, JobRecord, ExecutionService, _Failed, Any, Exception (+22 more)

### Community 5 - "MuJoCo Cell API"
Cohesion: 0.06
Nodes (26): Cell, _drop_specks(), ndarray, Self, Release the renderer's GL context., Move an existing camera in place (no recompilation)., Assisted grasp when ``self.assisted_grasp`` is True; pure contact friction…, Brick orientation quaternion in (x, y, z, w) convention. (+18 more)

### Community 6 - "Isaac Stage 3 Parallel"
Cohesion: 0.07
Nodes (42): cv2, glob, isaaclab_actuators, isaaclab_app, isaaclab_assets, isaaclab_controllers, isaaclab_envs, isaaclab_managers (+34 more)

### Community 7 - "Session Resume Tests"
Cohesion: 0.10
Nodes (46): digital_twin_armigo, 4. 受碗几何改变而更新的测试金标值, 1. Where the failures are recorded, NonIdealityCfg, Optional actuator and encoder non-idealities for sensitivity testing., assert_same_state(), assisted_pick(), layout() (+38 more)

### Community 8 - "Physics Expert Tests"
Cohesion: 0.07
Nodes (48): 1. 采纳方案：`R6_box_floor_wedge_walls`, 2. 候选与对照实验证据, 3. 与 Isaac 凸分解的差异, 5. 采纳后新基线与待夹爪计划处理的清单, 6. 未覆盖项, 7.1 本轮踩坑与 3D 建模经验, 7.2 MuJoCo vs. Isaac Sim (PhysX) 在本项目的工程差异, 7. 3D 碰撞建模经验与 MuJoCo / Isaac Sim (PhysX) 对比 (+40 more)

### Community 9 - "Execution Service Tests"
Cohesion: 0.14
Nodes (48): JobState, FaultHooks, Executes one ledger operation at a time under a lease. Each run uses its own…, Test seams called by a worker; production code leaves them unset., RolloutWorker, diff_session_states(), Paths at which two session states differ bit-for-bit; an empty list means an…, StrEnum (+40 more)

### Community 10 - "Cross-Engine Benchmarks"
Cohesion: 0.06
Nodes (46): _evaluate_mj_rollout(), _git_head(), _build_shared_trajs(), _finalize(), _mj_brick_layout(), _mj_jitter_condition(), _mj_make_cell(), _mj_run_matrix_condition() (+38 more)

### Community 11 - "Cameras And Views"
Cohesion: 0.08
Nodes (38): camera_pose(), ndarray, Top camera at ``factor * reach`` from the shoulder anchor, in front of the…, sample_case(), CameraPose, look_from(), project(), ndarray (+30 more)

### Community 12 - "IK Pick-Place Expert"
Cohesion: 0.11
Nodes (36): camera_forward(), main(), Command line of the MuJoCo sorting cell. Camera images follow dataset sort-…, expert_handoff(), expert_pick(), expert_place(), expert_present(), Grip (+28 more)

### Community 13 - "Catalog Playback And Rendering"
Cohesion: 0.09
Nodes (29): contextlib, datetime, fnmatch, importlib_metadata, io, os, pathlib, queue (+21 more)

### Community 14 - "Inspection View Tests"
Cohesion: 0.11
Nodes (40): main(), ndarray, run_one(), InspectionResult, Outcome of a single visibility-gated ``sim_oracle`` inspection query., assess_feasibility(), ExpertParams, Kinematic-only check before an episode: a reason string when the layout is… (+32 more)

### Community 15 - "Snapshot Store"
Cohesion: 0.09
Nodes (19): BlobRef, ExperimentIdentity, _is_int(), Any, ndarray, Pointer to an immutable blob., Committed description of one snapshot; self-checking via three SHA-256 fields., JSON-able dict of every field except ``manifest_sha256``. (+11 more)

### Community 16 - "Anomaly Dataset Rendering"
Cohesion: 0.08
Nodes (34): functools, counts(), main(), Physically present matched bricks, evaluate v6 RGB, and fit a separate bottom…, save_rtx_source(), trial(), build(), _camera_fovy() (+26 more)

### Community 17 - "Experiment Utility Scripts"
Cohesion: 0.08
Nodes (23): argparse, collections_abc, concurrent_futures, json, mcp_server_fastmcp, mlflow, Log a ``train_sup.py`` run (report, scores, adapter + head weights) to the…, Floating-brick twin of a ``render_dataset.py --dump-poses`` dataset (MuJoCo):… (+15 more)

### Community 18 - "Inspection View Capture"
Cohesion: 0.08
Nodes (28): capture(), Pixel counts of ``camera``'s view (:func:`capture_view`) and a square colour…, Run one layout to its first show hold and write its crops; returns the layout…, render_layout(), inspect(), save(), _pose_grid(), ndarray (+20 more)

### Community 19 - "Supervised Detector Training"
Cohesion: 0.12
Nodes (29): load_net(), main(), device, Path, Score every crop of a dataset with a trained supervised cross detector…, Normal train layouts held out of ``train_sup.py`` training (identical call,…, validation_layouts(), augment() (+21 more)

### Community 20 - "Plate Bowl Blender Scene"
Cohesion: 0.09
Nodes (30): embed_self_in_blend(), Embed this script in the saved .blend file so it appears in Blender's Scripting…, add_sample_bricks_in_plates(), add_studio_lighting_and_camera(), apply_plate_plastic_finish(), build_plate_showcase(), create_plate_mesh(), embed_self_in_blend() (+22 more)

### Community 21 - "Show-Pose Policies"
Cohesion: 0.10
Nodes (24): _make_policy(), build_ctx(), Stage-3 inspection agent: show-pose policies (rule baselines and an LLM over a…, True if ``GET {base_url}/models`` answers., server_reachable(), build_context(), CoveragePolicy, _extract_json() (+16 more)

### Community 22 - "Twin MuJoCo Skills"
Cohesion: 0.13
Nodes (3): blind_geometry(), Checkpoint, TwinMujoco

### Community 23 - "Depth And Sensor Noise"
Cohesion: 0.11
Nodes (29): decode_height(), depth_plane_to_height(), height_to_depth_plane(), layered_height_image(), ndarray, Height above the table → RGB uint8 in the depth bridge ``layers`` encoding (R…, Height above the table (m, NaN where invalid) from a ``layers``-encoded depth…, Compute world Z component of camera viewing rays per pixel using full-frame… (+21 more)

### Community 24 - "Gripper Blocked State"
Cohesion: 0.12
Nodes (26): File Structure, Global Constraints, Self-Review, Task 0: 用真实数据确定阈值, Task 1: `GripperMonitor`（纯函数式，先写测试）, Task 6: 全量回归与文档, 先后顺序与依赖, 夹爪「被卡住」显式状态与恢复策略 Implementation Plan (+18 more)

### Community 25 - "Table Relocation Planner"
Cohesion: 0.12
Nodes (20): main(), run_trial(), on_view(), summarize(), max_tip_reach(), plan_bottom_view(), plan_table_scan(), ndarray (+12 more)

### Community 26 - "Heuristic Inspector"
Cohesion: 0.14
Nodes (26): HeuristicInspector, At least one observed view detected., At least one ``top`` view detected., Eligible faces seen clearly in some non-detecting, scored view., Accumulate view evidence and decide the label, grasp and next presentation.…, _ev(), Tests for the pure heuristic inspection decisions (no MuJoCo, no torch)., test_choose_grasp_requires_a_candidate() (+18 more)

### Community 27 - "Closed-Loop Expert Core"
Cohesion: 0.12
Nodes (16): Task 4: 阶段 2 —— 被卡住时的反应式恢复, Task 5: 阶段 3 —— 仅在需要时收窄预张开度 + 可达性预检, _min_jerk(), ndarray, Min-jerk move from the last target to ``q_to``. Moves longer than…, Collision geoms that are neither the robot nor the target brick, nor the table…, Smallest distance (m) between the gripper / moving links and bowls or clutter…, Grasp target, joints and clearance for wrist roll ``roll`` pinching across… (+8 more)

### Community 28 - "Anomaly Train Eval"
Cohesion: 0.14
Nodes (27): export_model(), fit_padim(), fit_patchcore(), _fmt(), in_real_top_crop(), load_images(), load_rows(), main() (+19 more)

### Community 29 - "Inspection LangGraph"
Cohesion: 0.11
Nodes (14): build_inspection_graph(), validate(), GraphNextPose, _InterruptExecutor, PoseExecutor, Proposer, Any, Pose (+6 more)

### Community 30 - "Detector Reports And Fakes"
Cohesion: 0.09
Nodes (18): base64, binascii, httpx, matplotlib, matplotlib_pyplot, pil, random, Offline, blinded Qwen cross classification on paired bottom RGB crops. (+10 more)

### Community 31 - "Scene Layout"
Cohesion: 0.15
Nodes (26): Layout, Bowl centres, brick pose and appearance for one episode (base frame, metres /…, _cross_var_fallback(), _cross_var_sample(), _cross_var_strokes(), _defect_geoms(), _defect_rng(), _defect_shapes() (+18 more)

### Community 32 - "Ledger Tests"
Cohesion: 0.19
Nodes (26): job_resource(), Lease resource name guarding one job., _counts(), db_path(), ledger(), fixture, Path, Tests for the append-only operation ledger. (+18 more)

### Community 33 - "Lego Brick Blender Script"
Cohesion: 0.12
Nodes (25): add_studio_lighting_and_camera(), assign_material(), build_showcase_scene(), create_abs_plastic_material(), duplicate_brick(), ensure_collection(), import_and_normalize_brick(), main() (+17 more)

### Community 34 - "Physics Expert Benchmark"
Cohesion: 0.11
Nodes (26): Task 3: 检测器审计（660 回合）, _mj_non_ideal(), Actuator non-idealities of a condition (latency, backlash, velocity limit,…, _jsonable(), load_top_camera(), main(), _print_line(), Any (+18 more)

### Community 35 - "Catalog Store"
Cohesion: 0.24
Nodes (6): Catalog, now(), Path, Any, SHA-256 of the canonical JSON encoding of ``obj``., sha256_json()

### Community 36 - "Operation Ledger"
Cohesion: 0.14
Nodes (15): Connection, canonical_json(), Encode ``obj`` deterministically (sorted keys, no whitespace, UTF-8); NaN and…, IdempotencyConflictError, LedgerError, The operation ledger rejected a request., An idempotency key was reused with a different operation type or different…, The run's stop latch is set; new work is refused until it is explicitly cleared. (+7 more)

### Community 37 - "Grasp Geometry Diagnostics"
Cohesion: 0.13
Nodes (20): episode_diagnostics(), ndarray, Post-hoc diagnostics of a closed-loop episode from its telemetry and phase…, For each show pose, the longest run (s) of show-phase frames in which the…, ``grasp_push_mm``, ``jaw_normal_at_lift_N``, ``release_peak_speed_mps`` and…, show_hold_measured(), geom_points(), jaw_pad_faces() (+12 more)

### Community 38 - "Presentation Scoring"
Cohesion: 0.11
Nodes (23): camera_view_ok(), defect_strokes(), face_frame(), order_show_poses(), ndarray, Indices of up to ``k`` poses to present, best score first. A pose is skipped…, Whether a camera is above the table, looks down and keeps ``clearance`` from…, Centre, outward normal and in-plane axes (u, v with u x v = normal) of one face… (+15 more)

### Community 39 - "Simulator Adapter"
Cohesion: 0.08
Nodes (14): Limits per run, counted over every branch; ``None`` means unlimited., RunBudget, Protocol, Replace the control program; execution continues from its first frame., Advance up to ``n_frames`` control frames; return how many were executed., Adopt ``state``; raises ``IdentityMismatchError`` / ``SnapshotCorruptError``…, Builds sessions for one engine from a JSON scene specification., A fresh session at the scene's initial state, with no program loaded. (+6 more)

### Community 40 - "Catalog Tests"
Cohesion: 0.14
Nodes (23): catalog(), geometry(), fixture, parametrize, Governance invariants tested with synthetic sources; no GPU required., recorded(), report(), test_cancel_before_execution_never_publishes() (+15 more)

### Community 41 - "Sim Blend Upgrade"
Cohesion: 0.12
Nodes (23): Frame 0000: arm upright at rest, start of sort, Frame 0140: arm folded down, gripper at olive brick (grasp phase), Frame 0240: arm extended carrying olive brick toward pink plate (transport phase), Frame 0380: arm returned upright, olive brick placed on pink plate (sort complete), MuJoCo Cell Verification Frame, SO-101 yellow arm with bricks and green/pink plates in MuJoCo cell, enhance_plastic_material(), fix_aruco_uvs() (+15 more)

### Community 42 - "Inspection Labels"
Cohesion: 0.13
Nodes (22): pytest, assert_prompt_leak_free(), Raise :class:`DefectLeakageError` if a rendered message names a class or a bowl., assert_no_defect_leakage(), DefectLeakageError, inspect_object_sim_oracle(), _neutralize_asset_name(), Any (+14 more)

### Community 43 - "Brick-Level View Fusion"
Cohesion: 0.14
Nodes (22): main(), Write the scorer spec of a supervised cross detector (``sup_scorer.py`` /…, brick_auroc(), brick_table(), calibration_stats(), load(), main(), operating_point() (+14 more)

### Community 44 - "Inverse Kinematics"
Cohesion: 0.15
Nodes (22): _build_mj_pick_place_plan(), Solve MuJoCo IK plan for a pick-show-place condition (returns trajectory,…, best_roll(), ik(), ik_reach(), jaw_direction(), _jaw_direction(), ndarray (+14 more)

### Community 45 - "Catalog Capture"
Cohesion: 0.13
Nodes (14): IsaacCapture, MotionCapture, MujocoCapture, Capture USD geometry once, then its complete world transforms after each frame., main(), convert(), main(), Path (+6 more)

### Community 46 - "Cross Defect Variants"
Cohesion: 0.15
Nodes (22): cross_var_strokes(), Sample the strokes of :func:`cross_var_strokes`; also report whether the…, Two strokes of a randomised, hand-drawn cross on the brick's defect face. Each…, bench(), _corners(), _flat(), _half_extents(), _intersection() (+14 more)

### Community 47 - "LLM Policy Prompting"
Cohesion: 0.15
Nodes (10): CoverageTable, pose_id(), PoseOption, Path, For each grid pose: the fraction of layouts in which each face got ``>=…, P(face shown) per menu pose. Marginal over the measured layouts, or…, One menu pose: wrist flex (rad), wrist roll (deg), P(face shown with >=…, Pose menu with expected face coverage, measured on the show-pose grid of… (+2 more)

### Community 48 - "Episode Trace Recorder"
Cohesion: 0.15
Nodes (3): EpisodeTrace, write_json(), test_3d_context_does_not_encode_the_defect_or_its_face()

### Community 49 - "Ledger Leases"
Cohesion: 0.13
Nodes (16): OperationLedger, Path, Self, SQLite-backed append-only ledger; safe across threads and across processes…, Idempotently request cancellation; returns the resulting current state., _drive(), Create a job and move it to ``state`` through legal transitions; return…, test_cancel_requested_job_may_still_succeed() (+8 more)

### Community 50 - "Heuristic Episode Runner"
Cohesion: 0.14
Nodes (9): CaptureFn, HeuristicPlanner, pinched_faces_now(), Any, ndarray, The straight joint path keeps the arm and the held brick above the table, and…, The brick faces the jaws cover right now: the +/- faces of the brick axis best…, Pre-grasp scan -> grasp choice -> held view -> top close-ups, with an anomaly… (+1 more)

### Community 51 - "Expert Grasp Phases"
Cohesion: 0.13
Nodes (10): InspectFn, NextPoseFn, ClosedLoopExpert, EpisodePlanner, Protocol, Per-frame pick → show → handoff → place policy on privileged simulator state.…, ``bowl_xy`` fixes the target bowl up front (the label read from ground truth).…, A whole-episode inspection strategy driving the expert through extra stages… (+2 more)

### Community 52 - "Plate Decomposition"
Cohesion: 0.17
Nodes (19): _chain_segments_to_loop(), _clip_wall_polygon(), _cross(), ear_clip_triangulate(), hertel_mehlhorn(), is_convex_ccw(), main(), _point_in_triangle() (+11 more)

### Community 53 - "Runtime Errors"
Cohesion: 0.11
Nodes (18): sqlite3, InvalidTransitionError, JobNotFoundError, Exception, Exception hierarchy for snapshots, the operation ledger, and the execution…, The ledger holds no operation with the requested job id., A job state transition is not allowed (for example, leaving a terminal state)., Base class for every persistence and recovery error raised by… (+10 more)

### Community 54 - "Agent Graph Tests"
Cohesion: 0.25
Nodes (18): initial_state(), FakeLLM, Scripted answers in order (a string, or a callable of the messages); repeats…, LLMProposer, Asks the LLM for a pose choice; keeps one log record (latency, tokens, raw…, _answers(), _cfg(), FakeExecutor (+10 more)

### Community 55 - "Armigo MCP Server"
Cohesion: 0.13
Nodes (10): aggregate(), create_mcp(), start_episode(), main(), options(), _service(), TwinService, job() (+2 more)

### Community 56 - "Catalog Quality Checks"
Cohesion: 0.16
Nodes (19): _clean(), _definition_map(), _entry(), evaluate_quality(), check(), limit(), _get(), label_definitions() (+11 more)

### Community 57 - "Expert Place Phases"
Cohesion: 0.14
Nodes (9): _Grasp, A grasp the expert executed: gripper-frame target, wrist roll and the servo…, Duration of a carry move from the last target to ``q_to``: at least ``min_s``…, Hold arm pose ``q_des`` while integrating the measured joint error into the…, The episode as a sequence of phases. Each phase is a generator of…, Lift the clamped brick straight up; returns the lift pose (the base of the show…, Planner stage 1: visit the planner's pre-grasp viewpoints with the open…, Planner stages 3-4: the held-brick view at the lift pose, then the planner's… (+1 more)

### Community 58 - "Failure Animation Export"
Cohesion: 0.12
Nodes (11): Task 0: 冻结基线与测量成本, locate(), main(), ndarray, Path, H.264 / yuv420p copy that plays everywhere; the raw mp4v file is removed on…, Every failing or flagged episode in the acceptance JSON files., Patches the episode so that every 30 Hz frame is rendered and streamed to a… (+3 more)

### Community 59 - "Bowl Collider Candidates"
Cohesion: 0.15
Nodes (17): build_candidate(), Candidate, _current_wall_boxes(), floor_strips(), _load_profile_polys(), Path, _r1_hull_files(), Candidate bowl collision builders and monkeypatch install/uninstall helpers. (+9 more)

### Community 61 - "Defect Kind Tests"
Cohesion: 0.18
Nodes (18): _bowl_collision_mesh_assets(), MJCF for one episode; include it next to Menagerie's ``so101.xml``.…, Emit inline convex-wedge ``<mesh>`` assets for the revolved ``plate_200mm.obj``…, scene_xml(), bench(), _defect_xml(), _layout(), _original_cross() (+10 more)

### Community 62 - "Gripper State Audit"
Cohesion: 0.11
Nodes (16): Global Constraints, MuJoCo 碗碰撞体：凸分解实验与采纳 Implementation Plan, 候选方案（按保真度从高到低，依次尝试）, 背景（为什么要做）, 鲁棒性闸门（每道的阈值都写死在 `gates.py` 常量里）, main(), Any, _run_audited_episode() (+8 more)

### Community 63 - "Detector Crop Utilities"
Cohesion: 0.12
Nodes (14): main(), main(), Any, Path, _render_layout(), ndarray, Square crop of ``img`` centred on ``bbox`` with ``pad`` × its larger side added…, square_crop() (+6 more)

### Community 64 - "Inspection Planning"
Cohesion: 0.21
Nodes (16): The heuristic inspection strategy as an…, best_closeup(), closeup_candidates(), closeup_library(), path_min_z(), prescan_candidates(), ndarray, Viewpoint planning for the heuristic inspection: where to put the arm so a… (+8 more)

### Community 65 - "Six-Face Twin Graph"
Cohesion: 0.17
Nodes (15): ChatLLM, Protocol, Anything that answers a list of chat messages., InspectionState, Policy, TypedDict, Six-face inspection and bounded strategy search over trusted simulator skills., Propose a reusable, validated policy without per-brick labels or simulator… (+7 more)

### Community 66 - "Anomaly Score Client"
Cohesion: 0.18
Nodes (6): RuntimeError, Self, A protocol or read failure of the anomaly score server (still a…, Score crops through ``score_server.py`` in the anomaly environment (one request…, ScoreServerError, SubprocessAnomalyScorer

### Community 67 - "Lego Random Color Detail"
Cohesion: 0.15
Nodes (16): clear_scene(), configure_render_and_viewports(), parse_cli_args(), pick_random_colors(), Namespace, Random, Import open-source 2x2, 2x3, and 2x4 LEGO brick STLs and apply random ABS…, Reset Blender to an empty scene without leftover data blocks. (+8 more)

### Community 68 - "Ledger Detail"
Cohesion: 0.18
Nodes (7): Row, _event(), JobEvent, Every registered operation (optionally of one run), oldest first., Every event that entered ``state``, across all jobs, in ledger order., Jobs whose current state is not terminal, oldest first., _record()

### Community 69 - "Score Server Detail"
Cohesion: 0.13
Nodes (17): load(), main(), device, Path, _decode_image(), _handle(), main(), _parse_model_spec() (+9 more)

### Community 70 - "Scene Detail"
Cohesion: 0.15
Nodes (13): Path, Align MuJoCo actuator gains, joint dynamics, and surface friction with…, _brick(), brick_mesh_name(), face_probe_name(), _face_probes(), Return the visual mesh name matching ``size``, or ``None`` for non-standard…, One thin patch per brick face (see… (+5 more)

### Community 71 - "Six-Face Agent Docs"
Cohesion: 0.18
Nodes (15): armigo-mcp stdio MCP server (list_cases, start_episode, job_status), Inner LLM skill-selection graph, Outer bounded JSON policy search (LangGraph), Local Qwen3.8-27B NVFP4 vLLM service, Armigo six-face inspection agent, Six-face label rule (UNKNOWN on exhausted budget), Time-travel checkpoints and branch rewind, twin_record LeRobot 0.6 exporter (+7 more)

### Community 72 - "Data Catalog Docs"
Cohesion: 0.14
Nodes (15): armigo-data local SQLite experiment catalog, Armigo 本地实验资料库, Layout-group frozen 80/10/10 split, Privileged labels in meta/privileged.parquet, Provenance and lineage (source/import/rerun identities), Versioned quality checks (PASS/FAIL/UNKNOWN), Recorded-state Isaac RTX playback export, 动画记录与 RTX (+7 more)

### Community 73 - "Bowl Gate Checks"
Cohesion: 0.23
Nodes (16): Task 1: 闸门库（先写闸门，再写候选）, bowl_geom_ids(), _brick_body_xml(), _contacts_vs_bowl(), g0_geometry(), g1_collider_micro(), _max_bowl_radius_mm(), MjData (+8 more)

### Community 74 - "Test Fuse Views Detail"
Cohesion: 0.13
Nodes (7): fixture, ndarray, Path, Brick-level fusion of per-view scores (``scripts/anomaly/fuse_views.py``,…, Synthetic crops: 6 views per brick (3 wrist prescan, 1 wrist show, 2 top…, scores(), _views()

### Community 75 - "Graph Detail"
Cohesion: 0.15
Nodes (12): langgraph_checkpoint_memory, langgraph_graph, langgraph_types, operator, InspectionState, TypedDict, LangGraph loop for the inspection show-pose decisions of one episode,…, Wraps a rule :class:`ShowPosePolicy` as a :class:`Proposer` (it answers in the… (+4 more)

### Community 76 - "Bowl Collider Decision"
Cohesion: 0.13
Nodes (13): Bowl collider convex decomposition (no candidate passed; src/ unchanged), Bowl Convex Decomposition Experiment Decision, Candidate Summary, R1_coacd_full (CoACD, fails G0 wall deviation 7.4 mm), R2_wedge_full (fails G1, 417 bad normals), R3_strips_floor_wedge_walls (fails G1), R4_strips_floor_box_walls (fails G0, wall miss), R5_sdf (fails G0; settle_timeout, 2.43x slower) (+5 more)

### Community 77 - "Physics Audit Reports"
Cohesion: 0.16
Nodes (13): d415-nominal-v1 sensor degradation check (2.64% invalid depth), physics_params.json v1-nominal-assumed (DUPLO masses, frictions, STS3215 gains), MuJoCo vs Isaac PhysX trajectory equivalence (joint_tracking, flick_slide, spin_release), 2. 实验一：动力学对照（相同初始条件 + 相同带时间戳控制序列，【仿真实跑】）, 3. 实验二：技能可迁移性与敏感性矩阵（搜索集 vs 留出组合扰动，【仿真实跑】）, 4. 工具链与未验证边界, compare_trajectories timestamp resampling (time shift, length mismatch, final vector diff), D415 camera-frame depth noise (d415-nominal-v2) (+5 more)

### Community 78 - "Joint Calibration"
Cohesion: 0.17
Nodes (15): main(), normalized_to_rad(), ndarray, rad_to_normalized(), Studio normalised joint values (6,) → model joint angles in radians., Inverse of :func:`normalized_to_rad` (float32, as stored in the datasets)., fit_camera(), loss() (+7 more)

### Community 79 - "Catalog Episodes"
Cohesion: 0.16
Nodes (10): substep(), jaw_env_contact_force(), observe(), PhysicsEpisode, ndarray, Advance one 30 Hz frame towards the 6-D joint position ``target`` (through the…, The most recent logged row of ``key``., The per-frame observation a closed-loop policy reads: joints, gripper frame,… (+2 more)

### Community 80 - "Local LLM Client"
Cohesion: 0.16
Nodes (9): ChatResult, LLMError, OpenAICompatLLM, Any, RuntimeError, The chat endpoint failed after all retries., ``POST {base_url}/chat/completions`` with temperature 0 and (Qwen) thinking off…, test_llm_policy_stop_and_transport_error() (+1 more)

### Community 81 - "Test Twin Agent Detail"
Cohesion: 0.14
Nodes (7): decision(), Determine a brick label from scored views and physical face visibility., test_negative_decisions_need_scored_coverage_of_all_six_faces(), test_rollback_keeps_failure_history_and_requires_fresh_bottom_evidence(), inspect(), rewind(), view()

### Community 82 - "Twin Record Detail"
Cohesion: 0.14
Nodes (9): export_lerobot(), main(), Path, Yield only contiguous, successful unrewound samples for the LeRobot exporter., student_frames(), Video, test_live_preview_retries_after_a_windows_file_lock(), test_outer_search_prefers_success_then_cost_and_reports_model_provenance() (+1 more)

### Community 83 - "Physics Detail"
Cohesion: 0.13
Nodes (14): ActuatorSpec, BrickSpec, physics_manifest(), physics_spec_from_dict(), PlateSpec, Any, Path, Build a :class:`PhysicsSpec` from the JSON config layout (also the layout of… (+6 more)

### Community 84 - "Scan Zone Detail"
Cohesion: 0.22
Nodes (12): copy, check_position(), _init(), main(), pan_axis_xy(), ndarray, Where on the table can the wrist camera scan a brick's top + four side faces…, A jaws-down IK solution at ``target`` (roll 0) from the ready pose or from… (+4 more)

### Community 85 - "Gripper Blocked Docs"
Cohesion: 0.22
Nodes (13): GripperMonitor (robot-readable signals only), Gripper states FREE/MOVING/CLAMPED/BLOCKED, layouts_seed1000_idx32 grasp-blocked failure case, Stage 3 open_ladder pre-narrowing and feasibility precheck, Stage 2 reactive back-off and replan (no retry consumption), Gripper blocked-state plan, Bit-exact invariance for unblocked episodes, README (+5 more)

### Community 86 - "Inspection Relocation Docs"
Cohesion: 0.23
Nodes (13): Straight-line Cartesian carry fix, EpisodePlanner optional hooks in physics_expert, HeldBrickPoser kinematic copy and per-face visibility, Heuristic inspector (pre-scan, held view, close-ups), RelocateExpert table relocation, Scan zone decision map (scan_vertical/outside/relocate/unreachable), Anomaly detection report, Brick-level max fusion (304/304 detected) (+5 more)

### Community 87 - "Stage2 Expert Detail"
Cohesion: 0.27
Nodes (12): main(), get_tcp_and_jac(), jaw_forces(), record_frame(), step_ik_pose(), step_joint_target(), _substep_jaw_components(), _substep_unintended() (+4 more)

### Community 88 - "Test Agent Policies Detail"
Cohesion: 0.23
Nodes (12): LLMShowPosePolicy, LLM picks the next pose from the :class:`CoverageTable` menu (see module doc);…, skipif, Show-pose policies: coverage table, greedy coverage, fixed sequence, LLM…, _rec(), test_coverage_policy_greedy_choice_and_stop(), test_fixed_sequence_policy_skips_shown_poses(), test_live_llm_returns_valid_choice() (+4 more)

### Community 89 - "Session Detail"
Cohesion: 0.18
Nodes (11): _layout_from_spec(), model_fingerprint(), mujoco_scene_spec(), MujocoAdapter, _plain(), Any, Path, The JSON scene specification a :class:`MujocoSession` is built from (and… (+3 more)

### Community 90 - "LeRobot Export"
Cohesion: 0.21
Nodes (12): logging, export(), lerobot_features(), main(), privileged_rows(), Any, Path, Convert recorded episodes (``offline_pipeline record`` output) into a LeRobot… (+4 more)

### Community 91 - "Index Detail"
Cohesion: 0.18
Nodes (12): Closed-loop physics expert (physics_expert.py, ClosedLoopExpert), Flagged episodes (JAW_ENV_OVER_5N, NEAR_FRAME_CAP), high_mass_seed29 (bounce_out, release 2.66 m/s), low_friction_seed111 (bounce_out, release 5.00 m/s), Resting brick spin-up/hop on bowl-floor cylinder (condim 4 torsional contact instability), MuJoCo false bounce_out from bowl-bottom micro-rocking, Pure-physics pick-inspect-place sensitivity matrix v1, bounce_out vs settle_timeout split (+4 more)

### Community 92 - "Run Detail"
Cohesion: 0.19
Nodes (12): model_builder(), MjData, MjModel, Return ``builder(extra_body_xml) -> (model, data)`` for a standalone bowl model., g2_static_soak(), G2: 4 s @ 240 Hz brick drop soak under extreme mass/friction combinations., _json_default(), main() (+4 more)

### Community 93 - "Camera Detail"
Cohesion: 0.18
Nodes (13): build_scene_cfg(), CellSceneCfg, main(), configclass, InteractiveSceneCfg, build_scene_cfg(), InteractiveSceneCfg, as_pose() (+5 more)

### Community 94 - "Test Inspection Scorer Detail"
Cohesion: 0.15
Nodes (6): fixture, Tests for the main-venv client of the anomaly score server (no torch needed)., scorer(), test_fixed_scorer(), test_score_details_support_optional_heads(), test_server_that_exits_immediately_raises()

### Community 95 - "Recovery Protocol Docs"
Cohesion: 0.30
Nodes (11): Commit protocol, ExecutionService and RolloutWorker, MujocoSession full-state capture/restore, Not covered, Append-only SQLite operation ledger with fencing tokens, Recovery rules, Content-addressed snapshot store (runtime/snapshot.py), Snapshots, execution service and recovery (stage two) (+3 more)

### Community 96 - "Gates Detail"
Cohesion: 0.23
Nodes (12): Self-Review, _ensure_markers(), g3_two_failing_conditions(), g4_full_regression(), GateResult, Any, G3: high_mass (0-29, 200-229) and low_friction (100-129, 200-229) -> 120/120…, G4: full 540 matrix/held-out + 120 layouts regression against baseline… (+4 more)

### Community 97 - "Replay Dashboard"
Cohesion: 0.21
Nodes (9): fastapi, fastapi_responses, create_app(), episode(), media(), state(), Path, Loopback-only live monitor and branch-aware episode replay. (+1 more)

### Community 98 - "Prepare Assets Detail"
Cohesion: 0.23
Nodes (11): pxr, _bind_plastic_material(), build_obj_prop_usd(), build_so101_physx_usd(), main(), Path, Prepare USD assets for Stage 1-3 Isaac Sim / Isaac Lab validation. Converts OBJ…, Create so101_physx.usda overriding so101.usda with unified PhysicsSpec drives,… (+3 more)

### Community 99 - "Report Detail"
Cohesion: 0.27
Nodes (12): Anomaly report v2 (PatchCore, 224 px), Exported model patchcore/pooled -> models/best.pt (score.py), Hidden defects (<25 defect pixels in view, AUROC ~0.5), PatchCore anomaly detector (anomalib, normal-only training), Simulated held-brick crops (show hold + virtual show poses, MuJoCo render_dataset.py), Threshold = 0.99 quantile of validation normal scores, wrist_cam scope outperforms top camera scope, Anomaly report v3-256 (PatchCore, 256 px) (+4 more)

### Community 100 - "Failure Handoff Notes"
Cohesion: 0.18
Nodes (11): 2. How the animations were made, 3. Agent logs, 4. Physics diagnosis (follow-up), 5. Suggested governance work, Bit-exact episode replay check (13/13 matched), export_agent_logs.py (raw jsonl + readable md), export_failure_animations.py (replay + 2x2 camera mp4/json/npz export), Failure/flagged selection rule (verdict != SUCCESS; release >1.2 m/s, jaw-env >5 N, frames >=890) (+3 more)

### Community 101 - "Report Detail 2"
Cohesion: 0.20
Nodes (11): layouts_seed1000_idx32 (pick_miss, no_clamp, jaw-env 23.1 N), No reachability pre-check for layouts, 1. `high_mass_seed29` (bounce_out, 2.66 m/s): collision detection returns a 115 mm penetration, 2. `low_friction_seed111` (bounce_out, 5.00 m/s): a resting brick spins up and hops with no external force, 3. What fixes them (experiment, not applied), 4. `layouts_seed1000_idx32` (pick_miss): the moving jaw is blocked by the green bowl rim, MuJoCo 4.0.1 native box-vs-cylinder collision bug (-115 mm penetration, horizontal normal), Box bowl floor instead of cylinder (30/30 on high_mass and low_friction; experiment, not applied) (+3 more)

### Community 102 - "Export Agent Logs Detail"
Cohesion: 0.21
Nodes (9): Re-render the crops of a ``render_dataset.py --dump-poses`` dataset with Isaac…, _blocks(), main(), Path, Copy Claude Code session / subagent transcripts and render each as a readable…, _result_text(), to_markdown(), shutil (+1 more)

### Community 103 - "Jobs Detail"
Cohesion: 0.27
Nodes (5): Jobs, loop(), Path, An OS-released process lock, including Windows; serializes GPU work across…, worker_lock()

### Community 104 - "Inspection Scorer Detail"
Cohesion: 0.23
Nodes (5): FixedScorer, ndarray, Read the fused score and any calibrated head scores from one inference request., Scorer from a plain callable, for tests and offline replays., Anomaly score of one square RGB uint8 crop (higher is worse).

### Community 105 - "Inspection Views Detail"
Cohesion: 0.23
Nodes (8): InspectionView, PixelOracleInspector, Visibility-gated defect label from exact pixel counts (ground truth, for…, One camera's view of the brick. Pixel counts are inside ``crop`` when one was…, test_oracle_defective_known_only_once_defect_pixels_seen(), test_oracle_normal_needs_every_eligible_face(), test_oracle_unknown_reason_does_not_depend_on_ground_truth(), _view()

### Community 106 - "Supervised Scorer"
Cohesion: 0.24
Nodes (7): device, ndarray, Path, Max over the finite head scores; NaN if none is finite. Plain ``max`` is order-…, Fuse the calibrated scores of the enabled heads., Calibrated u per enabled head, computed from one pass over the RGB crop., SupScorer

### Community 107 - "Isaac Checkpoint Probe"
Cohesion: 0.24
Nodes (8): capture(), IsaacOpenCheckpoint, probe(), Path, Verify direct Isaac state restoration at a table-supported, open-gripper skill…, Restore observable articulation, targets and rigid roots. PhysX solver caches…, Run from the active Stage 2 cell. Compare native restore continuations without…, restore()

### Community 108 - "Run Heuristic Inspection Detail"
Cohesion: 0.18
Nodes (6): _Capture, _OracleScorer, Training-identical crops (``render_dataset.capture``, clamped square crop, full…, Upper-bound detector: the visible defect pixels of the last capture. A normal…, fixture, rd()

### Community 109 - "Twin Graph Detail"
Cohesion: 0.22
Nodes (9): build_graph(), choose_view(), evaluate(), node(), observe(), run(), recover(), measure() (+1 more)

### Community 110 - "Test Heuristic Episode Detail"
Cohesion: 0.25
Nodes (9): greedy_face_cover(), At most ``max_views`` candidates covering ``faces`` (a face is covered by a…, _cand(), Viewpoint planners and the heuristic EpisodePlanner wired into the closed-loop…, test_best_closeup_and_library(), test_greedy_face_cover_double_coverage(), test_greedy_face_cover_picks_fewest_views(), test_inspector_min_clear_views() (+1 more)

### Community 111 - "Ledger Detail 2"
Cohesion: 0.22
Nodes (7): A worker presented a fencing token that is missing or no longer current for its…, StaleWorkerError, Lease, Raise StaleWorkerError unless ``lease`` is exactly the current lease of its…, test_concurrent_acquire_lease(), test_concurrent_register_same_key(), work()

### Community 112 - "Report Detail 3"
Cohesion: 0.20
Nodes (10): gname(), mj_step(), Hot paths and C++ decision, Isaac, Knowledge graph and limits, Refactor and performance verification — 2026-11-17, RGB control experiment, Scope and provenance (+2 more)

### Community 113 - "Heuristic Detail"
Cohesion: 0.27
Nodes (5): Pure heuristic decision logic for brick inspection (no simulator, MuJoCo or…, One camera view of the brick as seen by the inspector. ``faces_px`` holds the…, True when there is a score and it reaches this camera's threshold., Record one view. A scored view either detects (and contributes its visible face…, ViewEvidence

### Community 114 - "Heuristic Detail 2"
Cohesion: 0.24
Nodes (4): ``"defective"``, ``"normal"`` or ``None`` (unknown). Defective requires a…, Face selection without the label/budget guards of :meth:`next_present_face`., Face to show the top camera next, or ``None`` when nothing useful or affordable…, JSON-serialisable snapshot of the derived state and the decision reason.

### Community 115 - "Physics Expert Detail"
Cohesion: 0.20
Nodes (4): Show the brick to the camera at each show pose in turn (see…, One inspection of the current state at show pose (``flex``, ``roll_deg``); the…, The measured show-pose criterion: wrist flex and roll within tolerance and the…, Present the brick: hold ``q_show`` (with the joint-error integral) until the…

### Community 116 - "Train Sup Detail"
Cohesion: 0.25
Nodes (6): Conv2d, add_lora(), LoRAConv2d, Module, Frozen convolution plus a trainable low-rank update ``scale * B(A(x))`` (A:…, Replace every Conv2d below ``module`` by a LoRAConv2d; returns how many.

### Community 117 - "Bowl Collision Docs"
Cohesion: 0.39
Nodes (8): MuJoCo 4.0.1 box-vs-cylinder CCD defect (-115.31 mm), MuJoCo vs Isaac Sim PhysX collision comparison, R6_box_floor_wedge_walls bowl collider, Rejected candidates R0, R1 CoACD, R2, R3, R4, R5 SDF, R9, Bowl robustness gates G0-G5, Split-floor seams produce horizontal normals, Bowl convex decomposition plan, Monkeypatch candidate injection before adoption

### Community 118 - "Inspection Scorer Detail 2"
Cohesion: 0.25
Nodes (9): main(), main(), Any, summarize(), default_scorer(), default_sup_scorer(), Path, Scorer for per-camera models exported by ``train_eval.py --export-scopes… (+1 more)

### Community 119 - "Test Runtime Service Detail"
Cohesion: 0.25
Nodes (5): Builds :class:`ToySession` instances; ``engine_version`` and ``drift`` let…, ToyAdapter, fixture, One store + ledger on disk; ``service()`` opens a fresh service on them, like a…, Rig

### Community 120 - "Train Eval Detail"
Cohesion: 0.46
Nodes (7): _at_threshold(), _auroc(), _best_f1(), evaluate(), strat(), ndarray, Metrics over the test crops ``test_idx`` (rows and scores indexed alike).

### Community 121 - "Run Inspection Policies Detail"
Cohesion: 0.29
Nodes (8): _init_worker(), main(), _pct(), Any, ``iv.run_one`` without the pose grid, with a ``next_pose`` policy., run_one(), inspect(), summarize()

### Community 123 - "2026-11-17-Bowl-Convex-Decomposition-Experiments Detail"
Cohesion: 0.38
Nodes (7): File Structure, Task 2: 候选构造器与几何单元测试, Task 3: 离线分解工具与参考剖面, Task 4: 运行器（遇到第一个通过者即停）, Task 6: 收尾与报告, Restore monkeypatched ``scene`` and ``physics_expert`` functions., uninstall()

### Community 124 - "Report Detail 4"
Cohesion: 0.29
Nodes (6): compare_outputs.py (exit 1 on mismatch), Isaac stages 1-3 verification (camera reprojection, friction-contact pick, 16-env benchmark), 1,600-line MuJoCo script split into mujoco_cell modules (compat launcher, shared Isaac camera geometry), Decision: no C++ extension, RGB render variability (max 1/255; --rgb-atol 1 only for uint8 RGB), Visibility pybind11 enum comparison hot path (fixed by int conversion; 6.70x sim, 2.94x render)

### Community 125 - "Candidates Detail"
Cohesion: 0.29
Nodes (4): install(), Monkeypatch ``scene._bowl``, ``scene._visual_mesh_assets``, and…, g5_engineering(), G5: bit-identical determinism (3 episodes x 2), wall-clock <= 1.5x baseline,…

### Community 126 - "Serve Armigo Qwen Detail"
Cohesion: 0.29
Nodes (6): CUDA_HOME, HF_HUB_OFFLINE, LD_LIBRARY_PATH, LIBRARY_PATH, PATH, serve_armigo_qwen.sh script

### Community 127 - "Top Camera Reach Detail"
Cohesion: 0.33
Nodes (5): main(), Exception, How large can the held brick get in the top camera? Sweep whole-arm poses with…, _Reached, summarize()

### Community 128 - "Gripper Blocked State Detail"
Cohesion: 0.33
Nodes (6): 1. 状态定义与输入信号, 2. 阈值来源（Task 0 实测分布）, 3. 阶段 1 检测器审计（R6 碗采纳后，660 回合）, 4. 阶段 2 与阶段 3 策略及全量回归, 5. 已知局限, 夹爪「被卡住」显式状态与恢复策略

### Community 129 - "Index Detail 2"
Cohesion: 0.40
Nodes (5): Closed-loop physics expert: failure export, Failures (verdict != SUCCESS), Flagged (verdict SUCCESS, but over a threshold), Layout, Reproduce

### Community 130 - "Inspection Scorer Detail 3"
Cohesion: 0.40
Nodes (4): AnomalyScorer, Protocol, What the inspection pipeline needs from an anomaly detector., Score threshold chosen for ``camera`` on validation normals.

### Community 131 - "Preview Detail"
Cohesion: 0.50
Nodes (4): LEGO Bricks Preview Render, Random-colored 2x2 / 2x3 / 2x4 LEGO brick set on grey ground, Plates Preview Render, Six glossy colored sorting plates, three holding LEGO bricks

### Community 132 - "Stage2 Keyframes Detail"
Cohesion: 0.50
Nodes (4): Stage 2 keyframe storyboard (6 phases x 3 camera views), Pure-physics friction pinch grasp lifting orange brick (no fixed joint), Closed-loop expert pick/present-defect/place sequence (above_open, descend_pause, closed_pinch, lifted_physics, show_defect, released_in_plate), Sorting scene: yellow SO-101 arm, green and pink plates, orange brick with defect X mark

### Community 133 - "Cross Engine Report Detail"
Cohesion: 0.50
Nodes (4): 1. Cross-Engine Trajectory Equivalence (MuJoCo vs Isaac Sim PhysX), 2. Pure-Physics Pick-Inspect-Place Sensitivity Matrix, 3. Sensor Degradation Check (`d415-nominal-v1` + RTX Cameras + Dynamic Clutter), Cross-Engine Physics Verification & Sensitivity Report (`v1-nominal-assumed`)

### Community 135 - "Scene Detail 2"
Cohesion: 0.50
Nodes (4): _clutter_brick(), Duplo studs on top of a brick of ``size`` (full extents), as visual-only…, Emit one clutter Duplo 2x2 brick body (mesh visual + group-3 collision box)., _studs()

### Community 136 - "Test Shared Modules Detail"
Cohesion: 0.50
Nodes (4): _normalized_to_rad_reference(), ndarray, The per-joint loop the simulators each carried before the conversion was shared., test_normalized_to_rad_matches_the_per_joint_formula_exactly()

### Community 137 - "Test Runtime Service Detail 2"
Cohesion: 0.67
Nodes (3): BaseException, Process death: derives from BaseException so no ``except Exception`` cleanup…, SimulatedCrash

## Knowledge Gaps
- **88 isolated node(s):** `Task 3: 离线分解工具与参考剖面`, `Task 6: 收尾与报告`, `Candidate Summary`, `Baseline Summary (`results/physics-expert-v2/independent_acceptance`)`, `Global Constraints` (+83 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 1041 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **19 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Cell` connect `MuJoCo Cell API` to `Archived Physics Diagnostics`, `MuJoCo Session And Toy Sim`, `Cross-Engine Benchmarks`, `Cameras And Views`, `IK Pick-Place Expert`, `Inspection View Tests`, `Inspection View Capture`, `Depth And Sensor Noise`, `Table Relocation Planner`, `Scene Layout`, `Physics Expert Benchmark`, `Grasp Geometry Diagnostics`, `Inverse Kinematics`, `Expert Grasp Phases`, `Defect Kind Tests`, `Detector Crop Utilities`, `Inspection Planning`, `Scene Detail`, `Joint Calibration`, `Catalog Episodes`, `Scan Zone Detail`, `Session Detail`, `Recovery Protocol Docs`, `Failure Handoff Notes`, `Run Inspection Policies Detail`?**
  _High betweenness centrality (0.128) - this node is a cross-community bridge._
- **Why does `ClosedLoopExpert` connect `Expert Grasp Phases` to `MuJoCo Cell API`, `Grasp Geometry Diagnostics`, `Physics Expert Detail 2`, `Inspection View Tests`, `Test Heuristic Episode Detail`, `Physics Expert Detail`, `Scan Zone Detail`, `Gripper Blocked State`, `Expert Place Phases`, `Closed-Loop Expert Core`, `Table Relocation Planner`?**
  _High betweenness centrality (0.060) - this node is a cross-community bridge._
- **Why does `HeuristicInspector` connect `Heuristic Inspector` to `Inspection Planning`, `Heuristic Detail 3`, `Heuristic Detail 4`, `Heuristic Detail 5`, `Heuristic Detail 6`, `Inspection View Tests`, `Heuristic Detail 7`, `Test Heuristic Episode Detail`, `Heuristic Detail`, `Heuristic Episode Runner`, `Heuristic Detail 2`?**
  _High betweenness centrality (0.033) - this node is a cross-community bridge._
- **Are the 69 inferred relationships involving `Cell` (e.g. with `Not covered` and `2. How the animations were made`) actually correct?**
  _`Cell` has 69 INFERRED edges - model-reasoned connections that need verification._
- **Are the 45 inferred relationships involving `OperationLedger` (e.g. with `Jobs` and `IdempotencyConflictError`) actually correct?**
  _`OperationLedger` has 45 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Task 3: 离线分解工具与参考剖面`, `Task 6: 收尾与报告`, `Candidate Summary` to the rest of the system?**
  _88 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Archived Physics Diagnostics` be split into smaller, more focused modules?**
  _Cohesion score 0.07981220657276995 - nodes in this community are weakly interconnected._