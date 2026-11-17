# 夹爪「被卡住」显式状态与恢复策略 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让智能体仅凭真机上也能读到的信号（夹爪指令、夹爪测量角、舵机负载、积木法向力）得到一个显式的 `GripperState`，并在 `BLOCKED` 时停止顶压、收窄张开度、重新规划，而不是原样重试。

**Architecture:** 一个纯函数式监视器 `GripperMonitor`（无 MuJoCo 依赖）每帧输出状态；`ClosedLoopExpert` 只在状态为 `BLOCKED` 时才改变行为，所以从不被卡住的回合轨迹逐位不变。分三阶段：只观测 → 反应式恢复 → 仅在需要时收窄预张开度。

**Tech Stack:** Python 3.14（`.venv`）、MuJoCo 4.0.1、numpy、pytest。

**Spec:** 诊断见 `results/failure-export/physics_diagnosis/REPORT.md` 第 4 节（`layouts_seed1000_idx32`）。

## Global Constraints

- 审计评估器 `v2-audited` 与 `PhysicsEpisode.telemetry()` 的键不改；新增信息只放 `info` 字典（与 `jaw_env_force` 同样的做法）。
- 智能体只经 `PhysicsEpisode.step(target)` 的关节目标行动（`test_expert_acts_only_through_the_joint_target` 约束）；监视器不得读 MuJoCo 接触表。`jaw_env_contact_force` 只能用作审计真值，不能进监视器输入。
- 输入信号限定为：`cmd`（夹爪指令 rad）、`meas`（夹爪测量角 rad）、`effort_frac`（舵机力矩 / 力矩上限，真机对应负载读数）、`jaw_normal`（两爪法向力，沿用现有 `pinch_normal`）。
- 不被卡住的回合：轨迹、telemetry、判定逐位不变（阶段 1、2 的硬性回归条件）。
- 默认 `open_g = 0.85`、`hold_g = 0.06`、`clamp_force_n = 1.5`、`max_grasp_retries = 2`、`clearance_sweep_samples = 6` 不改默认值，只加新字段。
- 源码用 tab 缩进，行宽 120；`uv` 不在 PATH，用 `.venv\Scripts\python -m pytest`；MuJoCo 测试需 `uv sync --extra mujoco`。
- 不提交、不推送，除非用户明确要求。
- 仿真名义参数，结论不外推真实硬件。

## 先后顺序与依赖

- 建议在碗碰撞体计划（`2026-11-17-bowl-convex-decomposition-experiments.md`）采纳之后再做阶段 2、3，因为壁几何一变，idx32 和 jaw-env >5 N 的回合清单可能变。
- 阶段 1（只观测，轨迹逐位不变）与碗计划互不干扰，可以先做。
- 已知事实：idx32 里 `green_bowl/g76`（碗壁盒子）卡住动爪，卡住时测量角停在 0.76 rad、指令降到 0.06 rad，积木受力 0 N，两次重试完全重复。`open_g=0.40` 单个布局成功，但 0.55/0.70 反而更糟，所以不能一刀切改默认值。

## 状态定义

| 状态 | 判据（全部来自上面四个信号） |
|---|---|
| `CLAMPED` | `min(jaw_normal) ≥ clamp_force_n` |
| `BLOCKED` | 非 CLAMPED，且 `|cmd−meas| > err_rad`、测得夹爪角速度 `< still_rad_s`、`max(jaw_normal) < brick_force_n`、`effort_frac ≥ effort_min`，上述条件连续 `blocked_frames` 帧 |
| `MOVING` | 非上两者，且 `|cmd−meas| > err_rad` 或测得角速度 ≥ `still_rad_s` |
| `FREE` | 其余 |

方向（`sign(cmd−meas)`）随状态一起返回：`>0` 是张开方向被挡，`<0` 是闭合方向被挡。idx32 的挡法（是在张开途中就顶住，还是闭合时才顶住）先由任务 0 的数据确认，不预设。

阈值不拍脑袋：任务 0 用现有回合的真实分布确定，写进 `GripperMonitorParams` 默认值。

## File Structure

- Create `src/digital_twin_armigo/mujoco_cell/gripper_state.py` — `GripperState`、`GripperMonitorParams`、`GripperMonitor`（纯 numpy/标准库）。
- Modify `src/digital_twin_armigo/mujoco_cell/physics_expert.py` — `_observe`（加 `gripper_cmd`、`effort_frac`）、`ClosedLoopExpert`（持有监视器、阶段 2/3 的行为）、`ExpertParams`（新字段）、`assess_feasibility`、`run_closed_loop_episode`（`info` 新字段）。
- Create `tests/test_gripper_state.py` — 合成信号单元测试。
- Modify `tests/test_physics_expert.py` — idx32、逐位不变、恢复策略。
- Create `scripts/audit_gripper_state.py` — 在 660 个回合上审计检测器的误报/漏报。
- Create `docs/GRIPPER_BLOCKED_STATE.md` — 阈值来源、审计结果。

---

### Task 0: 用真实数据确定阈值

**Files:**
- Create: `results/gripper-state/signals_idx32.txt`、`results/gripper-state/signal_distributions.json`
- Reuse: `results/failure-export/physics_diagnosis/idx32.py`（它已经 patch 了 `PhysicsEpisode.step` 记录每帧）

**Interfaces:**
- Produces: `signal_distributions.json` = 在「正常闭合爬坡」「成功夹紧」「张开」三类帧上，`|cmd−meas|`、夹爪角速度、`effort_frac` 的 p1/p50/p99/max；以及 idx32 卡住区间的同样统计。

- [ ] **Step 1: 记录 idx32 的每帧信号**：在 `idx32.py` 的 `step` 钩子里再记 `cmd=target[5]`、`meas=cell.joints()[5]`、`effort_frac=|actuator_torques[5]| / physics.gripper_actuator.effort_limit`、`pinch_normal`。输出帧 80–130 和三次尝试的表格。Expected: 卡住区间 `meas≈0.76`、`cmd` 下降、`pinch_normal=0`、`effort_frac` 偏高。
- [ ] **Step 2: 记录正常回合的同样信号**：矩阵里 `baseline_2x2` 等 6 个条件各取 5 个 seed。把帧按 `phase` 和 `clamped` 分类（用 `ClosedLoopExpert` 的相位标签：`grasp` 里闭合爬坡 / 夹紧；其余相位的夹爪张开保持）。
- [ ] **Step 3: 算分布并选阈值**：`err_rad` 取「正常闭合爬坡 `|cmd−meas|` 的 p99」之上、「idx32 卡住 `|cmd−meas|` 的 p1」之下的值；`still_rad_s`、`effort_min` 同理。若两类分布重叠，停下来报告，不硬凑阈值（这说明只靠这四个信号分不开，需要先和用户讨论）。
- [ ] **Step 4: 写入 `docs/GRIPPER_BLOCKED_STATE.md` 的「阈值来源」一节。**

### Task 1: `GripperMonitor`（纯函数式，先写测试）

**Files:**
- Create: `src/digital_twin_armigo/mujoco_cell/gripper_state.py`
- Create: `tests/test_gripper_state.py`

**Interfaces:**
- Produces: 

```python
class GripperState(enum.IntEnum):
	FREE = 0
	MOVING = 1
	CLAMPED = 2
	BLOCKED = 3

@dataclass(frozen=True)
class GripperMonitorParams:
	err_rad: float          # Task 0
	still_rad_s: float      # Task 0
	effort_min: float       # Task 0
	brick_force_n: float = 0.5
	clamp_force_n: float = 1.5
	blocked_frames: int = 4
	fps: float = 30.0

class GripperMonitor:
	def __init__(self, params: GripperMonitorParams) -> None: ...
	def reset(self) -> None: ...
	def update(self, cmd: float, meas: float, effort_frac: float, jaw_normal: Sequence[float]) -> tuple[GripperState, int]:
		"""Return (state, direction) with direction = sign(cmd - meas) when BLOCKED, else 0."""
```

- [ ] **Step 1: 写失败测试**（合成轨迹，不依赖 MuJoCo；数值用 Task 0 的阈值替换下面的占位阈值常量 `P`，该常量在测试顶部由 `GripperMonitorParams(...)` 实例化）。

```python
def feed(mon, rows):
	return [mon.update(*r) for r in rows]


def test_blocked_needs_consecutive_stalled_frames():
	mon = GripperMonitor(P)
	rows = [(0.06, 0.76, 0.9, (0.0, 0.0))] * (P.blocked_frames - 1)
	assert all(s is not GripperState.BLOCKED for s, _ in feed(mon, rows))
	state, direction = mon.update(0.06, 0.76, 0.9, (0.0, 0.0))
	assert state is GripperState.BLOCKED and direction == -1


def test_clamped_on_brick_is_never_blocked():
	mon = GripperMonitor(P)
	rows = [(0.06, 0.20, 0.9, (3.0, 3.0))] * 20
	assert all(s is GripperState.CLAMPED for s, _ in feed(mon, rows))


def test_normal_closing_ramp_is_moving_not_blocked():
	mon = GripperMonitor(P)
	rows = [(0.85 - 0.053 * k, 0.85 - 0.053 * k + P.err_rad * 1.5, 0.4, (0.0, 0.0)) for k in range(15)]
	assert all(s is not GripperState.BLOCKED for s, _ in feed(mon, rows))


def test_brick_force_on_one_jaw_prevents_blocked():
	mon = GripperMonitor(P)
	rows = [(0.06, 0.40, 0.9, (2.0, 0.0))] * 10  # fixed jaw touches the brick: that is "no clamp", not "blocked"
	assert all(s is not GripperState.BLOCKED for s, _ in feed(mon, rows))
```

- [ ] **Step 2: 运行确认失败**：`.venv\Scripts\python -m pytest tests/test_gripper_state.py -q`，Expected: FAIL（模块不存在）。
- [ ] **Step 3: 实现**。

```python
class GripperMonitor:
	def __init__(self, params: GripperMonitorParams) -> None:
		self.p = params
		self.reset()

	def reset(self) -> None:
		self._prev_meas: float | None = None
		self._run = 0

	def update(self, cmd, meas, effort_frac, jaw_normal):
		p = self.p
		vel = 0.0 if self._prev_meas is None else (meas - self._prev_meas) * p.fps
		self._prev_meas = meas
		fn = [float(x) for x in jaw_normal]
		if min(fn) >= p.clamp_force_n:
			self._run = 0
			return GripperState.CLAMPED, 0
		err = cmd - meas
		stalled = (
			abs(err) > p.err_rad
			and abs(vel) < p.still_rad_s
			and max(fn) < p.brick_force_n
			and effort_frac >= p.effort_min
		)
		self._run = self._run + 1 if stalled else 0
		if self._run >= p.blocked_frames:
			return GripperState.BLOCKED, int(np.sign(err))
		if abs(err) > p.err_rad or abs(vel) >= p.still_rad_s:
			return GripperState.MOVING, 0
		return GripperState.FREE, 0
```

- [ ] **Step 4: 运行确认通过。**
- [ ] **Step 5: `ruff check src tests`；暂不提交。**

### Task 2: 阶段 1 —— 只观测，轨迹逐位不变

**Files:**
- Modify: `src/digital_twin_armigo/mujoco_cell/physics_expert.py`（`_observe` 约 770 行、`ClosedLoopExpert.act`、`run_closed_loop_episode` 约 835–870 行）
- Modify: `tests/test_physics_expert.py`

**Interfaces:**
- Consumes: `GripperMonitor.update(...)`。
- Produces: `obs["gripper_cmd"]: float`、`obs["effort_frac"]: float`；`ClosedLoopExpert.gripper_state: GripperState`、`ClosedLoopExpert.blocked_frames: list[int]`；`info["gripper_blocked_frames"]: int`、`info["gripper_first_blocked_frame"]: int | None`、`info["gripper_block_direction"]: int`。

- [ ] **Step 1: 先录金标哈希（在改任何代码之前）**：对 `baseline_2x2` seed 1、`high_mass` seed 3、`dynamic_clutter` seed 2、layouts seed 1000 idx 0 四个回合，保存 `hashlib.sha256` 对 telemetry 各数组 `tobytes()` 的汇总到 `tests/data/gripper_state_golden.json`。
- [ ] **Step 2: 写失败测试**。

```python
def test_observing_gripper_state_does_not_change_the_trajectory():
	golden = json.loads(GOLDEN.read_text())
	for key, expected in golden.items():
		assert telemetry_hash(run_case(key)) == expected, key


def test_idx32_reports_blocked_within_the_closing_window():
	info = run_layout_case(1000, 32)["info"]
	assert info["gripper_first_blocked_frame"] is not None
	assert 95 <= info["gripper_first_blocked_frame"] <= 125  # closing ramp starts near frame 101


def test_clean_episode_never_reports_blocked():
	assert _closed_loop("baseline_2x2", 1)["info"]["gripper_blocked_frames"] == 0
```

- [ ] **Step 3: 运行确认失败**（字段不存在）。
- [ ] **Step 4: 实现**：`_observe` 加 `gripper_cmd`（上一帧的 `q_cmd[5]`，第一帧用当前测量）和 `effort_frac`（上一帧 `actuator_torques[5]` 的绝对值除以 `physics.gripper_actuator.effort_limit`）；`ClosedLoopExpert.act` 开头调用监视器并记录；`run_closed_loop_episode` 把统计写进 `info`。不改任何目标关节输出。
- [ ] **Step 5: 运行** `.venv\Scripts\python -m pytest tests/test_physics_expert.py -q`，Expected: PASS（含原有 `test_expert_acts_only_through_the_joint_target`）。
- [ ] **Step 6: 暂不提交。**

### Task 3: 检测器审计（660 回合）

**Files:**
- Create: `scripts/audit_gripper_state.py`
- Create: `results/gripper-state/audit.json`

**Interfaces:**
- Consumes: `info["gripper_blocked_frames"]`、`info["jaw_env_contact_max_N"]`、回合逐帧 `jaw_env_force`（`PhysicsEpisode.jaw_env_force`，仅审计用）。
- Produces: `audit.json` = `{"false_positive_episodes": [...], "false_negative_episodes": [...], "detection_latency_frames": {...}, "other_cause": [...]}`。

- [ ] **Step 1: 实现审计脚本**：复用 `bench_physics_expert.run_episode` / `run_layout_episode`，在每个回合结束后取 `episode.jaw_env_force`（逐帧真值）和监视器输出。真值事件定义：连续 ≥5 帧 `jaw_env_force ≥ 2 N`，且接触的是爪 geom。
- [ ] **Step 2: 判定**：
  - 误报：监视器报 `BLOCKED` 但该回合 `jaw_env_contact_max_N < 0.5` → 必须为 0。
  - 漏报：存在真值事件但没有 `BLOCKED` → 逐个列出，并记录接触对象（碗壁、杂物、桌面）。接触对象不是夹爪卡住而是别的 link 蹭到的，记入 `other_cause`，不算检测器缺陷。
  - 延迟：真值事件起点到 `BLOCKED` 的帧数，要求中位数 ≤6 帧。
- [ ] **Step 3: 后台运行**（约 660 回合，耗时按任务 0 实测外推），日志写 `results/gripper-state/audit.run.log`。
- [ ] **Step 4: 闸门**：误报 0；之前标记的 8 个 jaw-env >5 N 回合里，由夹爪被卡住造成的全部被检出；不通过则回到任务 0 调阈值或报告「四个信号分不开」。
- [ ] **Step 5: 把结果写入 `docs/GRIPPER_BLOCKED_STATE.md`。**

### Task 4: 阶段 2 —— 被卡住时的反应式恢复

**Files:**
- Modify: `src/digital_twin_armigo/mujoco_cell/physics_expert.py`（`ExpertParams`、`_script` 约 584–630 行）
- Modify: `tests/test_physics_expert.py`

**Interfaces:**
- Produces（`ExpertParams` 新字段，默认值使旧行为不变）：`block_backoff_rad: float = 0.10`、`max_block_replans: int = 2`、`open_ladder: tuple[float, ...] = (0.85, 0.70, 0.55, 0.40)`；`ClosedLoopExpert.block_replans: int`；`aborted` 新取值 `"gripper_blocked"`。
- Consumes: `GripperState.BLOCKED` 和方向。

行为：`_script` 的夹紧等待循环（`for _ in range(p.clamp_timeout_frames)`）和闭合爬坡 `self._move(q_closed, p.close_s, "grasp")` 期间，监视器报 `BLOCKED` 时：

1. 立刻把夹爪指令改为 `meas ∓ block_backoff_rad`（沿卡住的反方向退一点），不再顶压；
2. 抬到 `hover_z_m`（与现有重试一致，避开碗沿）；
3. 把这次的 `(q_arm, meas)` 记为「已知被卡姿态」，下一次抓取规划从 `open_ladder` 里取下一档更窄的 `open_g`，且该档下 `_clearance` 必须 ≥ `min_grasp_clearance_m`；
4. 重新 `_grasp_plan()`；`block_replans` 超过 `max_block_replans` 或阶梯用尽则 `aborted = "gripper_blocked"`，回家，不抬起。

这条路径与现有 `no_clamp` 重试分开计数：卡住不消耗 `max_grasp_retries`。

- [ ] **Step 1: 写失败测试**。

```python
def test_idx32_recovers_by_narrowing_the_opening():
	row = run_layout_case(1000, 32)
	info = row["info"]
	assert row["verdict"] == "SUCCESS"
	assert info["block_replans"] >= 1
	assert info["jaw_env_contact_max_N"] < 5.0


def test_unrecoverable_block_aborts_without_lifting():
	row = run_forced({"open_ladder": (0.85,), "max_block_replans": 1}, layout=(1000, 32))
	assert row["info"]["aborted"] == "gripper_blocked"
	assert PHASES.index("lift") not in row["_phases"]
```

- [ ] **Step 2: 运行确认失败。**
- [ ] **Step 3: 实现**上面四步；复用 `_move`、`_ik`、`_grasp_plan`，不新增 MuJoCo 接触读取。
- [ ] **Step 4: 运行测试**，`.venv\Scripts\python -m pytest tests/test_physics_expert.py tests/test_gripper_state.py -q`，Expected: PASS。
- [ ] **Step 5: 逐位不变回归**：任务 2 的金标哈希测试仍须通过（这四个回合不应触发 `BLOCKED`）。再扩展到矩阵 540 + 布局 120：对任务 3 中「`gripper_blocked_frames == 0`」的回合，telemetry 哈希必须与阶段 1 完全相同；只允许被检出卡住的回合改变。
- [ ] **Step 6: 暂不提交。**

### Task 5: 阶段 3 —— 仅在需要时收窄预张开度 + 可达性预检

**Files:**
- Modify: `src/digital_twin_armigo/mujoco_cell/physics_expert.py`（`_grasp_candidate` 约 471 行、`_grasp_plan` 约 526 行、`assess_feasibility` 约 758 行）
- Modify: `tests/test_physics_expert.py`

**Interfaces:**
- Produces: `_grasp_candidate(roll, axis, half_w, q_hover, open_g=None)`（`None` 表示 `p.open_g`）；`assess_feasibility(...) -> str | None` 在所有 (roll, open_g) 候选的 `clearance_m < min_grasp_clearance_m` 时返回 `"grasp_blocked_by_environment"`，否则 `None`。
- 规则：先用默认 `open_g` 评分；**只有**最优候选 `clearance_m < min_grasp_clearance_m` 时，才沿 `open_ladder` 往下试；每一档额外要求夹爪间隙 ≥ 积木宽 + `pad_clearance_m` + 2×`servo_tol_m` + 0.004 m（间隙由 `jaw_pad_faces` 在该 `open_g` 下算出）。最终选 `score = min(clearance, cap) − roll_penalty` 最高的 (roll, open_g)。

- [ ] **Step 1: 写失败测试**：idx32 的首次规划就选出 `open_g < 0.85`（不必等卡住后才重试）；`baseline_2x2` seed 1 仍选 `open_g == 0.85`；`assess_feasibility` 对一个人为把碗放到积木旁的布局返回 `"grasp_blocked_by_environment"`。
- [ ] **Step 2: 运行确认失败。**
- [ ] **Step 3: 实现**；`run_closed_loop_episode` 开头调用 `assess_feasibility`，非 `None` 时回合直接记为不可达（不执行脚本），原因写入 `info["infeasible_reason"]`。
- [ ] **Step 4: 回归（逐位不变条件）**：默认 `open_g` 候选已满足间隙要求的回合，telemetry 哈希与阶段 2 相同。
- [ ] **Step 5: 暂不提交。**

### Task 6: 全量回归与文档

**Files:**
- Create: `docs/GRIPPER_BLOCKED_STATE.md`
- Create: `results/gripper-state/regression.json`

- [ ] **Step 1: 跑审计矩阵 540 + 留出条件 + 120 布局**，与碗计划采纳后的基线对比。通过条件：
  - 矩阵成功数 ≥ 基线；布局 ≥ 119/120，且 `layouts_seed1000_idx32` 成功；
  - jaw-env >5 N 回合数 < 基线（8），无新增 bounce_out / FAST_RELEASE；
  - 回合 ≥890 帧的数量不增加；
  - 「`gripper_blocked_frames == 0`」的回合 telemetry 与阶段 1 逐位相同；
  - 检测器误报 0（任务 3 重跑）。
- [ ] **Step 2: 全套测试** `.venv\Scripts\python -m pytest -q`，`ruff check src tests`。
- [ ] **Step 3: 写文档**：状态定义、阈值来源、审计结果、已知局限（只覆盖夹爪被环境顶住；爪与积木的卡滞、舵机真实负载读数与仿真力矩的对应关系要在真机上标定）。
- [ ] **Step 4: 向用户报告，等待是否提交。**

## Self-Review

- 目标覆盖：显式状态（任务 1–2）、真机可用信号约束（Global Constraints）、停止顶压并重规划（任务 4）、预张开度收窄（任务 5）、可达性预检（任务 5）、回归与逐位不变（任务 2、4、5、6）。
- 空白：阈值依赖任务 0 的真实分布，若分布重叠计划会停下来报告，不假设可分；idx32 的挡法方向未预设，由任务 0 数据确认。
- 类型一致：`GripperState`、`GripperMonitor.update` 的返回值、`ExpertParams.open_ladder`、`aborted="gripper_blocked"`、`info` 字段名在各任务一致。
- 不在范围：长回合（接近 900 帧）和 3.12.0 碗底自旋，另行处理。
