# 夹爪「被卡住」显式状态与恢复策略

> 仿真名义参数（`config/sim/physics_params.json`，`v1-nominal-assumed`），结论不外推真实硬件。
> 本文档涵盖完整 **Stage 1（只观测）→ Stage 2（遇卡停压退回重规划）→ Stage 3（必要时预收窄 + 可达性预检）**，均基于已采纳的 `R6_box_floor_wedge_walls` 碗碰撞体实测验证。

## 1. 状态定义与输入信号

监视器 `GripperMonitor`（[`gripper_state.py`](file:///e:/digital_twin_armigo/src/digital_twin_armigo/mujoco_cell/gripper_state.py)）仅消费真机可读的四个信号，不读取 MuJoCo 接触表：

- `cmd` (`obs["gripper_cmd"]`)：上一帧夹爪关节指令（rad）
- `meas` (`obs["q"][5]`)：当前测得夹爪关节角（rad）
- `effort_frac` (`obs["effort_frac"]`)：上一帧夹爪执行器峰值力矩绝对值与 `gripper_actuator.effort_limit` 之比
- `jaw_normal` (`obs["jaw_normal"]`)：两爪对目标积木的法向接触力（N）

状态优先级：

1. `CLAMPED`：`min(jaw_normal) >= clamp_force_n`（1.5 N）
2. `BLOCKED`：非 `CLAMPED`，且连续 `blocked_frames`（4）帧同时满足：
   - `|cmd - meas| > err_rad`
   - `|vel| < still_rad_s`（基于相邻两帧 `meas` 差分 × 30 Hz）
   - `max(jaw_normal) < brick_force_n`（0.5 N）
   - `effort_frac >= effort_min`
   同时返回方向 `sign(cmd - meas)`（`-1` 为闭合方向被卡，`+1` 为张开方向被卡）。
3. `MOVING`：非上两者，且 `|cmd - meas| > err_rad` 或 `|vel| >= still_rad_s`
4. `FREE`：其余

## 2. 阈值来源（Task 0 实测分布）

采样范围：`layouts_seed1000_idx32`（[`signals_idx32.txt`](file:///e:/digital_twin_armigo/results/gripper-state/signals_idx32.txt)）与 30 个正常矩阵回合（`baseline_2x2`、`brick_2x3`、`brick_2x4`、`dynamic_clutter`、`low_friction`、`high_mass` 各 seed 0–4，[`signal_distributions.json`](file:///e:/digital_twin_armigo/results/gripper-state/signal_distributions.json)）。

| 帧类别 | 帧数 | `|cmd - meas|` (p1 / p50 / p99) | `|vel|` rad/s (p1 / p50 / p99) | `effort_frac` (p1 / p50 / p99) |
|---|---:|---|---|---|
| `closing_ramp`（正常闭合爬坡） | 444 | 0.002 / 0.183 / 0.308 | 0.009 / 1.611 / 1.735 | 0.112 / 1.000 / 1.000 |
| `clamped`（正常夹紧） | 13,452 | 0.094 / 0.139 / 0.148 | 1.2e-6 / 1.7e-4 / 0.012 | 1.000 / 1.000 / 1.000 |
| `open_or_other`（张开/保持） | 6,006 | 4.6e-6 / 2.8e-4 / 0.111 | 1.8e-6 / 4.2e-4 / 1.362 | 0.001 / 0.016 / 1.000 |
| `idx32_stalled`（`jaw_env > 2 N` 且无积木力） | 187 | 0.062 / 0.698 / 0.702 | 0.0028 / 0.0036 / 0.481 | 0.924 / 1.000 / 1.000 |

联合可分性检验：在 30 个正常回合的全部 6,450 帧非夹紧帧中，满足 `|vel| < 0.10` 且 `effort_frac >= 0.50` 且 `max(jaw_normal) < 0.5 N` 的帧数为 **0**。因此取间隙中段作为默认阈值：
- `err_rad = 0.05`、`still_rad_s = 0.05`、`effort_min = 0.70`
- `brick_force_n = 0.5`、`clamp_force_n = 1.5`、`blocked_frames = 4`

## 3. 阶段 1 检测器审计（R6 碗采纳后，660 回合）

运行脚本 [`audit_gripper_state.py`](file:///e:/digital_twin_armigo/scripts/audit_gripper_state.py)（结果保存在 [`audit.json`](file:///e:/digital_twin_armigo/results/gripper-state/audit.json)）：

- **误报（`FP`）**：`0 / 660`
- **抓取期漏报（`grasp_FN`）**：`0 / 660`
- **真值事件检出**：`3 / 3`（中位延迟 **6.0 帧**）：
  - `layouts_seed1000_idx32`（`17.0 N`，`grasp` 闭合被卡）：`truth_start=101 → first=107`（`direction=-1`）
  - `layouts_seed1000_idx15`（`12.5 N`，`place` 张开被卡）：`truth_start=720 → first=726`（`direction=+1`）
  - `layouts_seed1001_idx11`（`4.2 N`，`place` 张开被卡）：`truth_start=638 → first=644`（`direction=+1`）

## 4. 阶段 2 与阶段 3 策略及全量回归

实现在 [`physics_expert.py`](file:///e:/digital_twin_armigo/src/digital_twin_armigo/mujoco_cell/physics_expert.py)：
- **阶段 2（反应式恢复）**：在 `grasp` 闭合爬坡或夹紧等待期间检测到 `BLOCKED` 时，立即指令 `meas - dir * block_backoff_rad` 停压退回、抬至 `hover_z_m`、记录被卡姿态并在 `open_ladder`（`(0.85, 0.70, 0.55, 0.40)`）中选择下一档满足间隙与避障的更窄 `open_g` 重规划；不占用 `max_grasp_retries`，超限或无可行档位时记 `aborted = "gripper_blocked"` 并直接回 home（不抬起）。
- **阶段 3（必要时预收窄 + 可达性预检）**：默认 `open_g = 0.85` 的最优候选 `clearance_m < min_grasp_clearance_m`（2 mm）时，才沿 `open_ladder` 试算满足夹爪净间隙（`>= 积木宽 + pad_clearance_m + 2*servo_tol_m + 0.004 m`）的候选并选最高得分；`assess_feasibility` 在所有 `(roll, open_g)` 候选均存在几何干涉（`clearance_m < 0`）时提前返回 `"grasp_blocked_by_environment"`。

660 回合全量回归结果（[`regress_gripper_state.py`](file:///e:/digital_twin_armigo/scripts/regress_gripper_state.py)、[`regression.json`](file:///e:/digital_twin_armigo/results/gripper-state/regression.json)）：
- **矩阵 + 留出条件**：**`540 / 540 SUCCESS`**（无退化）
- **随机布局（seed 1000 & 1001）**：**`120 / 120 SUCCESS`**（`layouts_seed1000_idx32` 从 `pick_miss` 变为 `SUCCESS`，`open_g = 0.40`，`jaw_env_contact_max_N = 0.0 N`）
- **被预收窄的布局**（仅 3 个）：`idx32`（`0.40`, `0.0 N`）、`idx48`（`0.70`, `0.0 N`）、`idx25`（`0.70`, `0.0 N`）
- **逐位不变性**：其余 **`657 / 657`** 个回合的 telemetry 与 phases SHA-256 哈希与 Stage 1 **逐位一致**（`hash_diffs_unblocked_default = []`）
- **`jaw_env > 5 N`**：从 `R0` 基线 `6` 个 / `R6` Stage 1 `2` 个降至 **`1` 个**（仅剩 `layouts_seed1000_idx15` 的 `place` 阶段张开碰撞）
- **`bounce_outs` / `fast_releases` / 误报 `FP`**：均为 **`0`**；既有成功回合无新增 `>= 890` 帧（`idx37` 仍为 1 个；`idx32` 因完整走完 `pick→show→handoff→place→return` 而在 `return` 末帧达到 900 帧并判定 `SUCCESS`）。

## 5. 已知局限

- 监视器仅覆盖**夹爪关节自由度被环境顶住**的情况，不检测砖块与爪垫间滑移卡滞；
- `effort_frac` 在真实硬件上的 STS3215 舵机电流/负载读数映射与阈值裕度仍需实机标定；
- `place` 阶段放砖后张开夹爪时被碗壁阻挡（如 `layouts_seed1000_idx15`）不在本轮抓取恢复范围内。
