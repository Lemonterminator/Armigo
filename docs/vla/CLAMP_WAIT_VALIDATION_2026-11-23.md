# 等待夹稳规则：独立验证协议 v1（2026-11-23）

写于运行之前，没有看过 val2 上的任何结果。

## 1. 背景与改动

首次 grasp 末段 A/B 实验的结果见 `E:/scratch/grasp-tail-ab/20261123-075031/REPORT.md`，协议见 [GRASP_TAIL_AB_HANDOFF_2026-11-23.md](GRASP_TAIL_AB_HANDOFF_2026-11-23.md) r3。在 val 的 60 个布局上，A+wait 相对 A：

- 首次夹稳 +22/60，22:0；
- 抬升 +9；
- 分拣 +8，8:0；
- 碰撞布局 +1：一次抬升时夹爪与环境接触 2 帧，峰值 3.4 N。

按那份协议第 5 节的规则，碰撞布局增加，因此要求把等待夹稳作为独立的工程改动，在新的布局集上另行验证。本协议就是这次验证。

**改动内容。** `StudentTwin(clamp_wait=True)`，评测参数为 `--clamp-wait`，代码在 [student_rollout.py](../../src/digital_twin_armigo/agent/student_rollout.py)。

- 只要 ACT 在**任一** grasp 段（包括重试）给出 `done` 时夹持尚未确认（连续 4 帧双爪受力 ≥ 1.5 N 且砖速 < 5 mm/s），就保持最后一条已执行命令，直到出现以下任一情况：
  - 夹稳且静止；
  - 夹爪监测为 BLOCKED；
  - 保持满 45 帧（超时）。
- 判据和帧数沿用专家的 `_close_and_clamp`。不增加其他动作。
- 每次等待在 `events.jsonl` 里写一条 `clamp_wait` 事件，记录帧数和结果。

**与 A/B 实验中 A+wait 的区别。**

- A+wait 只在首次 grasp 触发切换时等待。
- 本规则不看切换判定，并且覆盖重试的 grasp 段。
- 首次 grasp 触发切换时，两者的命令逐字节相同（`tests/test_grasp_tail.py`）。

`TakeoverTwin`（DAgger 采集）暂不支持此规则，传入会直接报错。

## 2. 布局、模型与参数

- **布局：** `config/vla/val2.json`，100 个布局，种子 1400001–1400100。
  - 生成命令：`make_suites.py --val 100 --val-start 1400001`。
  - 与 train、val、test 以及 agent_bench 的种子都不重叠。
  - 类别比例与其他套件相同：正常 34，缺陷面 0–5 各 11。
  - 在此之前从未用于任何模型或规则开发。
- **模型：** `E:/scratch/s11v2-fit30k/best`，与 A/B 实验相同。
- **参数：** 与 A/B 实验相同。
  - `--device cuda --mode sync --replan-interval 50 --no-teacher`
  - 判别器 `results/agent-bench/sup-v7bot-local.json`
  - 评分参考 `E:/scratch/vla-pilot-grader_reference_frozen.json`
  - `sensing=realistic`

## 3. 三次运行

三次运行都带 `--grasp-tail-mode act`（只插桩，不改动作）和 `--protocol clamp-wait-v1`。

| 组 | 额外参数 | 含义 |
| --- | --- | --- |
| **A** | `--record-act-rgb` | 原始 ACT，记录每次 ACT 请求的 RGB |
| **W** | `--clamp-wait --pin-rgb-from <A>` | 等待夹稳规则 |
| **A′** | `--pin-rgb-from <A> --pin-handoff clamp-wait` | 重复 A，作为噪声基线 |

先跑 A，然后跑 W 和 A′。

**共享前缀。** 沿用 r3 的 RGB 固定，见 [rgb_pin.py](../../src/digital_twin_armigo/vla/rgb_pin.py)。

- W 和 A′ 的交接帧都定在 A 首次 grasp 中 ACT 给出 `done` 且尚未夹稳的那一帧（`handoff_frame("clamp-wait")`），不看切换判定。
- A 在 `done` 时已经夹稳的布局，整回合都固定。之后如果在重试的 grasp 上才出现差异，会记为请求失配并从那里转为实时，逐例列出。
- 因此，A′ 衡量的是与 W 同一前缀、同一交接点之后实时运行的波动。

## 4. 终点与统计

**终点：**

- 主要终点：首次 grasp 夹稳的布局数 / 100。
- 次要终点：
  - 完成过抬升 / 100；
  - 最终分拣成功 / 100；
  - 发生碰撞的布局数 / 100，并拆分为机械臂与环境、夹爪与环境两类，报告峰值力；
  - 抬升转化：新增的首次夹稳中，同时新增抬升的布局数。
- 规则遥测：等待次数、各结果（夹稳 / 阻挡 / 超时）的计数、等待帧数，首次 grasp 与重试分开统计。

**统计方法：** 两组配对，A vs A′、A vs W。每组给出四格表、净差、95% bootstrap 区间（10,000 次，种子 42）和双侧精确 McNemar p 值，方法与 A/B 实验相同。

**前缀核验：** 从开头到交接帧，命令须逐字节一致。请求失配的布局逐例列出。

**命令：**

```
python scripts/vla/clamp_wait_report.py --out <root> --run act=<A> --run act-rep=<A′> --run clamp-wait=<W>
```

## 5. 预先约定的判定

| 条件 | 判定 |
| --- | --- |
| 数据完整，前缀核验通过；W − A 首次夹稳净增 ≥ 10/100；抬升和分拣的净差都 ≥ 0；碰撞布局净增 ≤ 2/100；没有新增机械臂与环境接触的布局 | 把 `clamp_wait` 设为学生评测的默认行为，并在 `TakeoverTwin` 中接入同一规则 |
| 首次夹稳达到阈值，但抬升或分拣减少，或碰撞条件不满足 | 逐例检查新增的碰撞和下游失败，报告收益与代价，暂不设为默认 |
| 首次夹稳净增 < 10/100，或与 A′ 噪声同量级 | 不设为默认；报告差值、区间和超时布局 |

关于阈值的说明：

- 10/100 约为 A/B 实验中 +22/60（37 个百分点）的四分之一，是本轮的工程取舍。
- 碰撞允许 +2，是因为 A/B 实验中，新增的碰撞来自原本没有走到的抬升，属于短时的夹爪轻碰。阈值在运行前定下，不按结果调整。
- 验证只覆盖这 100 个布局。碰撞事件稀少，没有新增碰撞也不能证明风险相同。
