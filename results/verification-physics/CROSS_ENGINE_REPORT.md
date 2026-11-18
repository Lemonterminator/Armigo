# Cross-Engine Physics Verification & Sensitivity Report (`v1-nominal-assumed`)

**Scope & Provenance:** All numbers below are **simulation-only** under the unified assumed physical specification [`config/sim/physics_params.json`](../../config/sim/physics_params.json) (`v1-nominal-assumed`: DUPLO 2x2 = 7.0 g, 2x3 = 9.6 g, 2x4 = 13.5 g; ABS friction 0.40/0.35; table 0.45/0.38; plate 0.55/0.45; gripper elastomer pad 1.45/1.25; STS3215 arm `kp=180, kd=12, effort=2.94`; gripper `kp=30, kd=2.5, effort=0.55`; 240 Hz physics, 30 Hz control). Real hardware calibration, real brick weights, and sim-to-real transfer remain unverified until hardware arrives.

## 1. Cross-Engine Trajectory Equivalence (MuJoCo vs Isaac Sim PhysX)

Source: `results/verification-physics/latest_traj/cross_engine_results.json`

| Experiment | Metric | MuJoCo | Isaac Sim | Cross-Engine Diff |
|---|---|---:|---:|---|
| `joint_tracking` (120 frames, 6 joints) | Trajectory tracking | — | — | `qpos_rmse = 0.00726 rad` (0.42°), `tcp_pos_rmse = 0.00396 m` (3.96 mm) |
| `flick_slide` (strike @ frame 63) | Peak launch speed / final disp. | `0.875 m/s` / `0.0815 m` | `0.649 m/s` / `0.0771 m` | `brick_pos_rmse = 0.0352 m`, `final_disp_diff = 4.4 mm` |
| `spin_release` (release @ frame 158) | Release Z / flight / landing | `z=0.2738 m` / `0.300 s` / `x=0.345 m` | `z=0.2786 m` / `0.433 s` / `x=0.413 m` | `qpos_rmse = 0.0130 rad`, `brick_pos_rmse = 0.0676 m` (Isaac landed on side `z=0.0159 m`) |

## 2. Pure-Physics Pick-Inspect-Place Sensitivity Matrix

Sources:
- MuJoCo (`assisted_grasp=False`, `solref="0.009 1"`, torsional bowl contact): `results/verification-physics/mj_matrix_check/cross_engine_results.json`
- Isaac Sim (`stage2_expert.py --no-cameras`): `results/verification-physics/isaac_matrix_*/stage2_metrics.json`

| Condition | MuJoCo Mode / Dist / Pinch | Isaac Mode / Dist / Slip / Pinch |
|---|---|---|
| `baseline_2x2` (`20°`) | `ok` (`0.0131 m`, `7.19 N`, `7.8 mm` slip) | `ok` (`0.0257 m`, `0.013 mm` slip, `9.29 N`) |
| `brick_2x3` (`35°`) | `bounce_out` (`0.0087 m`, `0.011 m/s` micro-rocking, `7.19 N`) | `ok` (`0.0292 m`, `0.093 mm` slip, `9.49 N`) |
| `brick_2x4` (`-25°`) | `ok` (`0.0075 m`, `7.29 N`, `11.3 mm` slip) | `ok` (`0.0180 m`, `1.60 mm` slip, `8.63 N`) |
| `dynamic_clutter` (3/bowl) | `ok` (`0.0201 m`, `7.24 N`, `7.4 mm` slip) | `ok` (`0.0154 m`, `7.68 mm` slip, `8.96 N`) |
| `low_friction` (`0.65x`) | `bounce_out` (`0.0075 m`, `0.0012 m/s`, `settle=0`, `7.17 N`) | `ok` (`0.0258 m`, `0.013 mm` slip, `9.29 N`) |
| `high_mass` (`1.6x`) | `bounce_out` (`0.0120 m`, `0.0346 m/s` at end, `7.27 N`) | `ok` (`0.0257 m`, `1.41 mm` slip, `8.52 N`) |
| `latency_backlash` (`2 frames`, `0.5°`) | `bounce_out` (`0.0155 m`, `0.0137 m/s`, `7.13 N`) | `ok` (`0.0259 m`, `0.014 mm` slip, `9.27 N`) |

*Note on `bounce_out` in MuJoCo:* In all 4 MuJoCo `bounce_out` cases, the brick actually landed cleanly inside the bowl center (`0.0075 - 0.0155 m` from center, `grasped_ok=True`, `0` unintended collisions), and was flagged by the strict post-release calm-velocity gate (`< 0.03 m/s` and `< 0.35 rad/s` for 8 consecutive frames) due to residual plate-bottom micro-rocking before the 1.8 s return/hold ended.

## 3. Sensor Degradation Check (`d415-nominal-v1` + RTX Cameras + Dynamic Clutter)

Source: `results/stage2_d415_check/stage2_metrics.json` & `demos/`
- 556 frames, `success=true`, `failure_mode="ok"`, `final_dist_to_bowl_m=0.0101 m`, `mean_pinch_force_N=9.29 N`
- `top_depth_v2` invalid pixel ratio (`B == 0` from min-Z 0.45 m + discontinuity edge + random dropouts): `2.64%`
