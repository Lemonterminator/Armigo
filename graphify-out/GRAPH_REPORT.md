# Graph Report - digital_twin_armigo  (2026-11-17)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 478 nodes · 1067 edges · 19 communities (17 shown, 2 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 56 edges (avg confidence: 0.9)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `6f85feb4`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- test_shared_modules.py
- stage3_parallel.py
- lego_random_color.py
- scene.py
- Armigo Digital Twin (SO-101 brick sorting cell)
- presentation.py
- upgrade_sim_blend.py
- plate_bowl_showcase.py
- fit_camera
- mujoco_cell/__init__.py
- ndarray
- depth.py
- calibration.py
- CameraPose
- project
- look_from
- SO-101 pick-and-place keyframe sequence (rest, grasp, transport, place)
- pathlib
- digital-twin-armigo

## God Nodes (most connected - your core abstractions)
1. `Cell` - 44 edges
2. `CameraPose` - 37 edges
3. `_simulate_one()` - 17 edges
4. `Layout` - 16 edges
5. `project()` - 14 edges
6. `fit_camera()` - 14 edges
7. `_render_one()` - 14 edges
8. `build_showcase_scene()` - 13 edges
9. `normalized_to_rad()` - 13 edges
10. `Armigo Digital Twin (SO-101 brick sorting cell)` - 13 edges

## Surprising Connections (you probably didn't know these)
- `build_obj_prop_usd()` --calls--> `defect_strokes()`  [INFERRED]
  scripts/isaac/prepare_assets.py → src/digital_twin_armigo/presentation.py
- `build_scene_cfg()` --calls--> `opengl_axes()`  [EXTRACTED]
  scripts/isaac/stage1_verify.py → src/digital_twin_armigo/camera.py
- `main()` --calls--> `as_pose()`  [EXTRACTED]
  scripts/isaac/stage1_verify.py → src/digital_twin_armigo/camera.py
- `main()` --calls--> `project()`  [EXTRACTED]
  scripts/isaac/stage1_verify.py → src/digital_twin_armigo/camera.py
- `main()` --calls--> `normalized_to_rad()`  [EXTRACTED]
  scripts/isaac/stage1_verify.py → src/digital_twin_armigo/joints.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Isaac Lab verification stages pipeline** — scripts_isaac_prepare_assets, scripts_isaac_stage1_verify, scripts_isaac_stage2_expert, scripts_isaac_stage3_parallel [EXTRACTED 1.00]
- **SO-101 robot model representations** — readme_assets_so101_mjcf, readme_assets_so101_usd, readme_assets_meshes [INFERRED 0.75]
- **Isaac Sim Stage 1 verification camera views** — results_stage1_isaac_front_rgb, results_stage1_isaac_top_rgb_crop, results_stage1_isaac_top_depth_crop, results_stage1_isaac_wrist_rgb [INFERRED 0.85]
- **Pick-and-place keyframe sequence** — assets_blender_sim_scene_frame_0000, assets_blender_sim_scene_frame_0140, assets_blender_sim_scene_frame_0240, assets_blender_sim_scene_frame_0380 [INFERRED 0.85]

## Communities (19 total, 2 thin omitted)

### Community 0 - "test_shared_modules.py"
Cohesion: 0.18
Nodes (9): mujoco, os, platform, One episode's MuJoCo model: stepping with the assisted grasp, and renderers for…, The optional ``mujoco`` import, with a headless GL backend selected where one…, subprocess, Tests for the modules both simulators share: joint units, camera geometry and…, ``MUJOCO_GL=egl`` is invalid on Windows and macOS: importing the cell there… (+1 more)

### Community 1 - "stage3_parallel.py"
Cohesion: 0.07
Nodes (42): cv2, isaaclab_actuators, isaaclab_app, isaaclab_assets, isaaclab_controllers, isaaclab_envs, isaaclab_envs_mdp, isaaclab_managers (+34 more)

### Community 2 - "lego_random_color.py"
Cohesion: 0.08
Nodes (43): add_studio_lighting_and_camera(), assign_material(), build_showcase_scene(), clear_scene(), configure_render_and_viewports(), create_abs_plastic_material(), duplicate_brick(), embed_self_in_blend() (+35 more)

### Community 3 - "scene.py"
Cohesion: 0.08
Nodes (26): tests (hardware-free unit tests), Digital twin geometry, presentation scoring, and dataset export utilities., Path, Layout, Bowl centres, brick pose and appearance for one episode (base frame, metres /…, _bowl(), _brick(), brick_mesh_name() (+18 more)

### Community 4 - "Armigo Digital Twin (SO-101 brick sorting cell)"
Cohesion: 0.06
Nodes (43): Any, argparse, MuJoCo Cell Verification Frame, SO-101 yellow arm with bricks and green/pink plates in MuJoCo cell, json, pxr, Armigo Digital Twin README, Armigo Digital Twin (SO-101 brick sorting cell) (+35 more)

### Community 5 - "presentation.py"
Cohesion: 0.08
Nodes (35): parametrize, pytest, Defect presentation_search, configure_cameras(), main(), get_tcp_and_jac(), record_frame(), step_ik_pose() (+27 more)

### Community 6 - "upgrade_sim_blend.py"
Cohesion: 0.12
Nodes (23): Frame 0000: arm upright at rest, start of sort, Frame 0140: arm folded down, gripper at olive brick (grasp phase), Frame 0240: arm extended carrying olive brick toward pink plate (transport phase), Frame 0380: arm returned upright, olive brick placed on pink plate (sort complete), enhance_plastic_material(), fix_aruco_uvs(), import_obj_mesh(), main() (+15 more)

### Community 7 - "plate_bowl_showcase.py"
Cohesion: 0.09
Nodes (31): LEGO Bricks Preview Render, Random-colored 2x2 / 2x3 / 2x4 LEGO brick set on grey ground, add_sample_bricks_in_plates(), add_studio_lighting_and_camera(), apply_plate_plastic_finish(), build_plate_showcase(), create_plate_mesh(), embed_self_in_blend() (+23 more)

### Community 8 - "fit_camera"
Cohesion: 0.10
Nodes (29): normalized_to_rad(), ndarray, rad_to_normalized(), Studio normalised joint values (6,) → model joint angles in radians., Inverse of :func:`normalized_to_rad` (float32, as stored in the datasets)., fit_camera(), loss(), score() (+21 more)

### Community 9 - "mujoco_cell/__init__.py"
Cohesion: 0.07
Nodes (58): contextlib, logging, camera_forward(), Cell, Pixel counts in ``camera``: defect and brick pixels visible in the scene, and…, Lowest point of the moving links' collision geometry (oriented bounding boxes),…, One episode's MuJoCo model with renderers for the dataset cameras. Ids and…, Release the renderer's GL context. (+50 more)

### Community 10 - "ndarray"
Cohesion: 0.10
Nodes (9): ndarray, Move an existing camera in place (no recompilation)., Assisted grasp: once the jaws are commanded closed with the brick between them,…, Put the held brick where the stored grasp puts it relative to the gripper…, World position and outward normal of the defect face centre., Colour image of ``camera`` (height, width, 3), written into ``out`` when given., World z of each pixel's viewing ray at unit depth along the optical axis. It…, Height above the table per pixel of ``camera`` and a validity mask, optionally… (+1 more)

### Community 11 - "depth.py"
Cohesion: 0.14
Nodes (17): collections_abc, build_scene_cfg(), CellSceneCfg, main(), configclass, InteractiveSceneCfg, decode_height(), depth_plane_to_height() (+9 more)

### Community 12 - "calibration.py"
Cohesion: 0.28
Nodes (9): dataclasses, math, numpy, Pinhole camera above the table: pose parametrisation, projection and ray…, Shared constants of the SO-101 sorting cell: joints, calibration, camera crop,…, Joint unit conversion between Studio's normalised readings and model joint…, Fitting the simulated cell to the real recordings: top camera pose and the…, Episode layout of the sorting cell (bowls, brick, appearance, cameras) and its… (+1 more)

### Community 13 - "CameraPose"
Cohesion: 0.21
Nodes (11): as_pose(), CameraPose, opengl_axes(), ndarray, A camera above the table: horizontal position, height, view yaw, tilt from…, Image right, image up and viewing direction of the camera, as unit vectors in…, MuJoCo camera x (image right) and y (image up) axes in the world., A :class:`CameraPose` from itself or from the ``camera`` mapping of a camera… (+3 more)

### Community 14 - "project"
Cohesion: 0.22
Nodes (11): project(), Where the rays through full-frame pixels (u, v) meet the horizontal plane at…, Pixel coordinates (u right, v down) of world points, and whether each lies in…, table_point(), bowl_rim_points(), ndarray, Whether a top camera is plausible: above the table, sees the brick, the gripper…, top_view_ok() (+3 more)

### Community 15 - "look_from"
Cohesion: 0.29
Nodes (7): look_from(), A camera ``distance_xy`` behind ``target`` (along -yaw) at ``height``, tilted…, A low side camera looking across the workspace from the side opposite the robot., A fully random camera above the table looking down at the workspace from any…, sample_front_camera(), sample_side_camera(), test_look_from_puts_the_target_on_the_optical_axis()

### Community 17 - "pathlib"
Cohesion: 0.33
Nodes (3): importlib_util, pathlib, The equivalence gate must reject missing data and numerical regressions.

## Knowledge Gaps
- **12 isolated node(s):** `digital-twin-armigo`, `Per-env domain randomization of brick color/position/orientation and lighting tint`, `Wrist-camera RGB observations showing gripper jaw, brick and plate with randomized appearance`, `Isaac Stage1 Wrist Camera RGB (gripper jaw tip with shadow on table)`, `SO-101 pick-and-place keyframe sequence (rest, grasp, transport, place)` (+7 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 195 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Cell` connect `mujoco_cell/__init__.py` to `test_shared_modules.py`, `scene.py`, `fit_camera`, `ndarray`, `calibration.py`, `CameraPose`?**
  _High betweenness centrality (0.130) - this node is a cross-community bridge._
- **Why does `CameraPose` connect `CameraPose` to `test_shared_modules.py`, `scene.py`, `fit_camera`, `mujoco_cell/__init__.py`, `ndarray`, `depth.py`, `calibration.py`, `project`, `look_from`?**
  _High betweenness centrality (0.059) - this node is a cross-community bridge._
- **Why does `Layout` connect `scene.py` to `test_shared_modules.py`, `fit_camera`, `mujoco_cell/__init__.py`, `calibration.py`, `CameraPose`?**
  _High betweenness centrality (0.037) - this node is a cross-community bridge._
- **Are the 15 inferred relationships involving `Cell` (e.g. with `CameraPose` and `Layout`) actually correct?**
  _`Cell` has 15 INFERRED edges - model-reasoned connections that need verification._
- **Are the 8 inferred relationships involving `CameraPose` (e.g. with `depth_plane_to_height()` and `Cell`) actually correct?**
  _`CameraPose` has 8 INFERRED edges - model-reasoned connections that need verification._
- **What connects `digital-twin-armigo`, `Per-env domain randomization of brick color/position/orientation and lighting tint`, `Wrist-camera RGB observations showing gripper jaw, brick and plate with randomized appearance` to the rest of the system?**
  _12 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `stage3_parallel.py` be split into smaller, more focused modules?**
  _Cohesion score 0.06845513413506013 - nodes in this community are weakly interconnected._