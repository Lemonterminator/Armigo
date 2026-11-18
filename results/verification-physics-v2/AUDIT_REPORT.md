# Phase 1 物理审计与判定修复报告 (`v2-audited`)

> **结论边界与证据标注规则**：本报告严格区分 **【源码检查】**、**【单元验证】**、**【仿真实跑】** 与 **【未验证 / 待实机验证】**。  
> 所有物理参数（[`config/sim/physics_params.json`](../../config/sim/physics_params.json)，`v1-nominal-assumed`，SHA-256 `f45b5200933282fd452d15e334f44bf51ae3e4e38470232ad3dc21aa5e272eea`）与 D415 退化参数（`d415-nominal-v2`）均为名义假设值；**双引擎一致与仿真验证通过不能代替真实机械臂、真实相机标定或 sim-to-real 迁移验证。**  
> 历史报告（`results/verification-2026-11-17/REPORT.md` 及 `results/verification-physics/`）保留未覆盖。

---

## 1. 审计问题、确定性反例与最小修复清单

所有反例测试先在未修复/契约缺失阶段执行并确认失败（TDD Red），随后实施最小修复并在 [`tests/`](../../tests/) 通过验证（TDD Green）：

| 审计项 | 原问题（【源码检查】） | 确定性反例（【单元验证】） | 修复方案与结果 |
|---|---|---|---|
| **假入碗 / 错分 `bounce_out`** | `metrics.py` 仅检查质心 XY `< bowl_r` 且 `z < 0.055`，导致边缘悬空、高挂碗沿通过，且碗内微晃未静置被误记为 `bounce_out` | [`test_counterexamples_false_bowl_containment_and_settle_timeout_vs_bounce_out`](../../tests/test_physics_and_metrics.py)：2x4 中心 `r=0.085m` 但端角探出碗外 `0.1168m > 0.10m`；质心高悬 `z=0.050m`；碗内未静置微晃 `0.045m/s` | 新增 8 角点几何包络判定；释放后独立观察最多 `3.0s`，区分 `place_miss` / `bounce_out` / `settle_timeout` |
| **先静止后运动** | `metrics.py` 仅取历史最长连续 calm 窗口且末帧未查角速度 | [`test_counterexamples_settled_early_then_moved_again_or_spun`](../../tests/test_physics_and_metrics.py)：先静止 `0.80s`，末帧前旋转 `1.2rad/s`、脉冲速度 `0.08m/s` 或末帧二次夹持 | 强制校验**观察结束前连续 `0.5s`**（`v < 0.03m/s`、`ω < 0.35rad/s`、全部 8 角点在碗内且未再夹持） |
| **缺角速度** | `brick_ang_vel is None` 时静默填零，将缺证据当静止 | [`test_counterexamples_missing_telemetry_empty_nan_and_bad_quaternions_return_unknown`](../../tests/test_physics_and_metrics.py)：传入 `brick_ang_vel=None` | 禁止填零；直接判定 `verdict="UNKNOWN"`、`reason_codes=["missing_brick_ang_vel"]` |
| **空轨迹 / NaN / 错误四元数** | 空轨迹崩溃；部分 NaN 或零模/非单位四元数未拦截 | 同上：`n=0` 空轨迹、第 50 帧 `brick_pos=NaN`、四元数 `[0,0,0,0]` 与模 `3.0` | 完整完整性检查；任一异常返回 `verdict="UNKNOWN"` 与明细 `reason_codes` |
| **`torque_saturation_ratio` 失真** | 用 `|q_cmd - q_act|` 伪造力矩饱和与 `overload_stall` | [`test_episode_health_detects_ok_drop_bounce_and_tracking_vs_torque_stall`](../../tests/test_physics_and_metrics.py) & `test_measured_torque_saturation_reported_separately_from_tracking_lag` | 跟踪滞后与力矩饱和彻底拆分为 `tracking_error_ratio` / `tracking_lag` 与 `torque_saturation_ratio` / `torque_stall`；无实测力矩数据时 `torque_saturation_ratio=None`（标注 `torque_evidence_source`） |
| **早期滑移漏检** | 从 `lift` 中点 `lift_idx[len//2]` 取夹持参考，忽略 `lift` 前半段滑移；`show_drop` 误判在 `place` 阶段 | [`test_counterexample_early_lift_slip_is_detected`](../../tests/test_physics_and_metrics.py)：第 42~88 帧在 `lift` 初段下移 `35mm` 后稳定 | 改从夹持闭合/开始抬升首帧取参考并全程追踪；按首次超限帧准确区分 `grasp_slip` 与 `show_drop` |
| **子步碰撞漏检 & 静态桌面过滤失效** | MuJoCo/Isaac 仅在 30Hz 帧末采样漏掉帧内冲击；【仿真实跑】暴露 `stage2_expert.py` 中 `/World/Table` 为静态 `AssetBaseCfg` 导致 PhysX GPU contact filter 报不支持警告且读数恒 0 | [`test_counterexample_substep_collision_even_when_terminal_bowl_succeeds`](../../tests/test_physics_and_metrics.py)：substep 3 出现 `18.0N` 碰撞而帧末 `0.0N`，砖块最终仍入碗 | 新增 240Hz 子步峰值+冲量采集；`cell.py` 与 `stage2_expert.py` 改为 8 子步历史聚合；将 Isaac `Table` 改为 `kinematic_enabled=True` 的 `RigidObjectCfg`，消除 GPU contact filter 警告；实现终态/过程约束分离与多原因返回 |
| **时间错位与指标混淆** | `compare_trajectories` 按 `min(len)` 截断、无时间戳重采样、漏末段矢量差 | [`test_counterexample_compare_trajectories_detects_time_shift_length_mismatch_and_final_vector_diff`](../../tests/test_physics_and_metrics.py) | 基于时间戳 `t` 重采样对齐，显式报告 `length_mismatch`、`duration_diff_s`、`estimated_time_shift_s`、`event_time_diff_s` 与未截断 `brick_final_pos_diff_m` / `brick_final_xy_diff_m` |
| **D415 相机坐标深度误差** | 用 `cam_z - z_world` 近似深度，在倾斜相机下触发误裁剪 | [`test_counterexample_tilted_camera_depth_uses_camera_frame_z_not_cam_z_minus_world_z`](../../tests/test_sensor_noise.py)：`tilt=35°`、`height=0.50m`、`z=0.10m` 时垂直差 `0.40m < 0.45m` 而光轴深度 `0.488m >= 0.45m` | 改为先在相机坐标系 `distance_to_image_plane` 上执行截断、边缘缺失与 $z^2$ 噪声（`d415-nominal-v2`），再反算世界高度；`D415NoiseCfg` 显式包含 `resolution` 与 `mode` |
| **缺陷信息泄露 & 无遮挡透视** | 原始场景/资产命名含 `_defective`、`defect_face` 等字段且缺乏视野/遮挡隔离 | [`test_counterexample_planner_context_strips_defect_ground_truth_and_asset_names`](../../tests/test_inspection.py) & `test_counterexample_sim_oracle_inspector_respects_visibility_occlusion_and_face` | 新增 `inspection.py`：`sanitize_scene_for_planner` + `assert_no_defect_leakage` 强制脱敏并拦截泄露；`inspect_object_sim_oracle` 仅在视野、距离、无遮挡且缺陷面朝向相机时返回带 `source="sim_oracle"` 标记的标签，否则返回 `UNKNOWN` |
| **配置静默回退** | `physics_params.json` 缺失时静默返回 `DEFAULT_PHYSICS` | [`test_missing_physics_config_raises_by_default`](../../tests/test_physics_and_metrics.py) | `load_physics_spec(..., strict=True)` 默认缺文件抛 `FileNotFoundError`，并新增 `physics_manifest` 记录文件路径与 SHA-256 |

### 历史报告 `flick_slide` 文案勘误（【源码检查】）

原 [`results/verification-physics/latest_traj/cross_engine_results.json`](../verification-physics/latest_traj/cross_engine_results.json) 中：
- MuJoCo `final_displacement_m = 0.081494 m`、`final_xy = [0.273969, 0.080288]`
- Isaac `final_displacement_m = 0.077141 m`、`final_xy = [0.260301, 0.077140]`
- 原报告写的 **4.4 mm** 是**标量滑行距离之差**（`81.49 - 77.14 = 4.35 mm`）；两个最终 XY 坐标的**欧氏矢量距离约为 14.0 mm**（`hypot(0.273969 - 0.260301, 0.080288 - 0.077140) = 0.01403 m`）。新版本 `compare_trajectories` 与报告同时分别报告两者，不再混淆。

---

## 2. 实验一：动力学对照（相同初始条件 + 相同带时间戳控制序列，【仿真实跑】）

数据源：[`results/verification-physics-v2/cross_engine_results.json`](./cross_engine_results.json)  
运行条件：`240 Hz` 物理、`dt = 0.004167 s`、`30 Hz` 控制（8 子步）、`assisted_grasp = False`、Kinematic `Table`、评估器 `v2-audited`、硬件 `NVIDIA GeForce RTX 5090` / `Windows-11`。

| 实验 | 长度与时长 | MuJoCo (`3.12.0`) | Isaac Sim (`OvPhysX`) | 跨引擎对齐与差异 |
|---|---|---|---|---|
| `joint_tracking` | `120 frames` (`3.967 s`) | `duration_diff = 0.0 s` | `duration_diff = 0.0 s` | `qpos_rmse = 0.00726 rad`，`tcp_pos_rmse = 0.00396 m`，`time_shift = 0.0 s` |
| `flick_slide` (strike @ 63) | `54 frames` (`1.767 s`) | `peak = 0.875 m/s`，`disp = 0.0815 m`，`stop = 0.10 s` | `peak = 0.837 m/s`，`disp = 0.0807 m`，`stop = 0.10 s` | 标量位移差 `0.80 mm`；最终 XY 矢量差 `6.31 mm`；`brick_pos_rmse = 0.0265 m`，`estimated_time_shift = -0.233 s`，`stop_time_diff = 0.0 s` |
| `spin_release` (release @ 158) | `55 frames` (`1.800 s`) | `flight = 0.300 s`，`final_x = 0.345 m` (`z=0.0095 m`) | `flight = 0.300 s`，`final_x = 0.284 m` (`z=0.0159 m`) | `flight_time_diff = 0.0 s`，`qpos_rmse = 0.00910 rad`，`brick_final_xy_diff = 69.3 mm`（Isaac 释放角速度 `12.83 rad/s` vs `2.05 rad/s` 并侧立落地） |
| **`shared_pick_place_replay`** | `597 frames` (`19.867 s`) | **`FAILURE ["settle_timeout"]`** (`dist = 0.0145 m`, `v = 0.009 m/s`, `ω = 2.142 rad/s`, `pinch_mean = 7.19 N`, `norm_peak = 13.50 N`, `tan_peak = 1.35 N`, `imp = 69.64 N·s`) | **`SUCCESS []`** (`dist = 0.0122 m`, `v = 0.0001 m/s`, `ω = 0.0084 rad/s`, `pinch_mean = 8.33 N`, `norm_peak = 47.08 N`, `tan_peak = 14.22 N`, `imp = 77.31 N·s`) | `qpos_rmse = 0.01385 rad`，`tcp_pos_rmse = 2.30 mm`，`brick_pos_rmse = 9.66 mm`，`brick_final_xy_diff = 3.20 mm`，`brick_final_pos_diff = 5.79 mm`，`time_shift = 0.0 s` |

> **关键发现**：在完全相同的 597 帧控制轨迹上，MuJoCo 与 Isaac 的空间轨迹已高度接近（TCP RMSE `2.30 mm`、最终 XY 矢差 `3.20 mm`），但判定暴露了两个此前被掩盖的引擎细节差异：  
> 1. MuJoCo 碗底接触存在残余角速度抖动（`ω = 2.142 rad/s` 尽管 `v = 0.009 m/s`），因此被真实判定为 `settle_timeout`（而非过去的假 `bounce_out`）；  
> 2. 回放快速闭合阶段，Isaac 240Hz 子步瞬时夹持法向峰值达 `47.08 N`（无阶段二 36 帧慢闭合斜坡时）。

---

## 3. 实验二：技能可迁移性与敏感性矩阵（搜索集 vs 留出组合扰动，【仿真实跑】）

数据源：[`results/verification-physics-v2/cross_engine_results.json`](./cross_engine_results.json) 与 [`results/verification-physics-v2/stage2_d415_v2/stage2_metrics.json`](./stage2_d415_v2/stage2_metrics.json)

| 划分 | 条件 | MuJoCo (30 固定种子，开环 IK + 自由物理，95% Wilson CI) | Isaac (`stage2_expert.py` 闭环 IK，单种子，240 Hz 子步采集) |
|---|---|---|---|
| `search` | `baseline_2x2` (`20°`) | `4/30` (`13.3%`, CI `[5.3%, 29.7%]`) — `settle_timeout:13, show_drop:9, place_miss:9, grasp_slip:2, pick_miss:2` | `SUCCESS` (`dist=0.0258m`, `norm_peak=11.68N`, `tan_peak=2.97N`, `unint=0`) |
| `search` | `brick_2x3` (`35°`) | `7/30` (`23.3%`, CI `[11.8%, 40.9%]`) — `settle_timeout:10, show_drop:8, place_miss:8, grasp_slip:6, pick_miss:3` | `SUCCESS` (`dist=0.0306m`, `norm_peak=11.38N`, `tan_peak=3.28N`, `unint=0`) |
| `search` | `brick_2x4` (`-25°`) | `5/30` (`16.7%`, CI `[7.3%, 33.6%]`) — `settle_timeout:19, grasp_slip:5, show_drop:3, place_miss:1, pick_miss:1` | `SUCCESS` (`dist=0.0170m`, `norm_peak=11.71N`, `tan_peak=3.76N`, `unint=0`) |
| `search` | `dynamic_clutter` (3/碗) | `11/30` (`36.7%`, CI `[21.9%, 54.5%]`) — `place_miss:13, show_drop:9, grasp_slip:5, settle_timeout:4, pick_miss:1` | `SUCCESS` (`dist=0.0265m`, `norm_peak=17.87N`, `tan_peak=11.27N`, `unint=0`) |
| `search` | `low_friction` (`0.65x`) | `1/30` (`3.3%`, CI `[0.6%, 16.7%]`) — `settle_timeout:21, grasp_slip:7, place_miss:5, show_drop:3, pick_miss:1` | `SUCCESS` (`dist=0.0265m`, `norm_peak=11.67N`, `tan_peak=2.95N`, `unint=0`) |
| `search` | `high_mass` (`1.60x`) | `8/30` (`26.7%`, CI `[14.2%, 44.4%]`) — `settle_timeout:12, place_miss:8, show_drop:7, grasp_slip:2, pick_miss:2` | `SUCCESS` (`dist=0.0322m`, `norm_peak=12.34N`, `tan_peak=3.06N`, `unint=0`) |
| `search` | `latency_backlash` (`2f`, `0.5°`) | `6/30` (`20.0%`, CI `[9.5%, 37.3%]`) — `settle_timeout:19, grasp_slip:6, place_miss:2, show_drop:1` | `SUCCESS` (`dist=0.0259m`, `norm_peak=11.32N`, `tan_peak=2.49N`, `unint=0`) |
| `held_out` | `combined_friction_mass` (`2x3`, `1.4x`, `0.7x`) | `4/30` (`13.3%`, CI `[5.3%, 29.7%]`) — `settle_timeout:20, grasp_slip:7, pick_miss:3, place_miss:2, show_drop:1` | **`FAILURE ["grasp_slip"]`** (`dist=0.0251m`, `norm_peak=66.95N`, `tan_peak=23.81N`, `unint=0`) |
| `held_out` | `combined_all_nominal_stress` | `8/30` (`26.7%`, CI `[14.2%, 44.4%]`) — `settle_timeout:19, grasp_slip:5, show_drop:3, place_miss:3` | `SUCCESS` (`dist=0.0076m`, `norm_peak=13.06N`, `tan_peak=2.88N`, `unint=0`) |

- **RTX 相机 + 相机坐标系 D415 退化验证（`stage2_d415_v2`，`571 frames`，【仿真实跑】）**：`verdict="SUCCESS"`，`trailing_settle_frames=83` (`2.77 s`)，`final_dist_to_bowl_m = 0.01018 m`，`unintended_collision_count = 0`，裁剪后 `top_depth_v2` 无效像素比例 **`1.89%`**（优于旧近似 `cam_z - z` 造成的 `2.64%` 误剔除）。

---

## 4. 工具链与未验证边界

- **依赖固定与工程验证**：已恢复安装 `uv 0.13.8` 并生成 [`uv.lock`](../../uv.lock)；Phase 1 全套硬件无关单元测试（`55 passed`）与相关 Ruff 静态/格式检查均于最终修改后通过。
- **仍为【待实机验证】的项目**：实机电机力矩/背隙标定、真实 DUPLO 质量与 ABS/硅胶摩擦系数、真实 D415 噪声分布、以及合成策略的 sim-to-real 迁移效果。
