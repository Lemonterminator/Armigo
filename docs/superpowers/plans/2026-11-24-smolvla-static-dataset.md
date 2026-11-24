# 大规模静态 SmolVLA 教师数据集 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在跑任何教师回合之前，先把场景分布设计好并逐项校验；再用冻结的 realistic 教师一次性采出一个静态数据集，让 SmolVLA 学会按语言指令选碗，并对相机外参、颜色和外观的变化保持稳健。

**Architecture:**
- **场景规格 v3**：每个案例由种子确定性地生成，碗颜色、相机、外观、杂物、指令改写和路由规则都写成显式字段。
- **预检**：生成套件时就算各因子的分布表，并做可达性、可见性、颜色可分性检查，不合格的场景在采集前剔除。
- **采集**：realistic 教师加动作噪声注入；在持砖检查点向场景里的每一只碗各分出一支搬运。
- **导出**：只导出学生能看到的三路相机。

**Tech Stack:** Python 3.14，MuJoCo，LeRobot 0.6（SmolVLA，`E:/il/lerobot-cuda`），PyTorch bf16，pytest，ruff（tab 缩进）。

**Spec:** 本文第 0–7 节就是规格。依据：[BOWL_SWAP_PROBE_2026-11-23.md](../../vla/BOWL_SWAP_PROBE_2026-11-23.md)（ACT 跟随 0/237）、[SMOLVLA_BOWL_PROBE_2026-11-24.md](../../vla/SMOLVLA_BOWL_PROBE_2026-11-24.md)（SmolVLA 探针）、[BOWL_CHANNEL_PAIRED_PLAN_2026-11-23.md](../../vla/BOWL_CHANNEL_PAIRED_PLAN_2026-11-23.md)（配对搬运的实现）。失败收集和工具改进见 [2026-11-24-agent-failure-loop.md](2026-11-24-agent-failure-loop.md)。

## Global Constraints

- **旧案例逐位不变。** 三元组和现有四元组案例（`options` 不含 `scene`）生成的 Layout、MJCF、相机和回合必须与现在完全相同。新功能只在 `options["scene"] == "v3"` 时生效。
- **槽位语义不变。** 代码里的 `"green"` / `"pink"` 继续表示“良品箱” / “缺陷箱”这两个槽位。v3 只改变它们的显示颜色和颜色名，不改字段名。
- **学生输入不得包含特权信息。** 遵守 `FORBIDDEN_DIAGNOSTIC_KEYS` 和 `FORBIDDEN_CAMERA_NAMES`。碗的坐标、颜色 RGB、路由规则都不能进模型；颜色只能以指令里的颜色词出现。
- **数据契约。** `STATE_DIM = 38` 不变。v3 数据的碗 one-hot（`state[31:34]`）全部置零，碗只通过语言给出。旧数据保持原样。
- **指令长度。** 用 `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` 的分词器计，每条指令不超过 40 个 token（上限 48，留余量）。
- **切分隔离。** 训练只用 v3 训练种子段。验证、OOD 测试和封存测试的种子段互不重叠，用 `dagger.assert_train_layouts_only` 守护。
- **教师冻结。** 正式采集必须在教师代码提交并写出 freeze 文件之后进行（见失败闭环计划的 F7）。每个回合记录教师的提交号。
- **教师不用真值兜底（已定）。** 教师找不到碗（`BOWL_NOT_FOUND`）或找不到积木时，回合按失败收录并立即终止，不退回真值。这样的回合不进训练集，进失败台账（失败闭环 F1，规则 R1），并计入试采判据。现有代码已经满足：realistic 下 `BOWL_NOT_FOUND` 没有恢复选项（`twin_mujoco.py` 约 2013 行），`BRICK_NOT_FOUND` 只会重新感知、不用真值；`collect_static_v3.py` 只需按终局把这类回合分流。
- **评测可比。** 旧的固定相机验收集 val2–val5 继续保留，用来和 ACT、fit30k 对比。
- **代码风格。** tab 缩进；`ruff check` 和 `ruff format` 通过。测试用仓库的 `.venv/Scripts/python -m pytest`；需要 torch 或 transformers 的测试用 `pytest.importorskip` 跳过，再在 `E:/il/lerobot-cuda` 里单独跑。
- **提交。** 只在本地分支提交，不推送。提交信息以 `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` 结尾。

---

## 0. 为什么要先设计分布

ACT 没学会按碗的颜色选碗。原因不在模型能不能读颜色，而在训练分布本身给了捷径：

- 目标碗完全由缺陷标签决定，同一个持砖画面只对应一种去向；
- 碗的颜色永远是绿和粉，71% 的示范去粉碗；
- 相机永远在同一个位置。

在这样的分布里，模型用多数先验和固定像素位置就能把损失降下去，不需要真正理解指令。所以这次的原则是：**每一个希望模型泛化的因素，都要在训练集里独立变化，并且和其他因素解耦；每一个希望测量的泛化，都要留出训练中没有出现过的组合。**这些在采集前就要决定，因为采集之后无法再补救。

## 1. 场景因子与分布

每个因子独立采样。“OOD 留出”一列的取值只出现在测试集里。

| 因子 | 训练分布 | OOD 留出（只在测试集） | 现状 |
|---|---|---|---|
| 碗颜色 | 调色板 8 色里无放回抽 2–3 色，两两色相差 ≥ 35° | 若干颜色对组合只在测试集出现（见第 2 节） | 固定绿/粉 |
| 碗数量 | 2 只（70%）或 3 只（30%，第三只是干扰碗，不承担分拣） | 无 | 2 只 |
| 碗位置 | 现有 `sample_layout` 的可达范围；颜色与位置独立 | 无 | 同 |
| 路由规则 | 每回合随机：良品→颜色 X，缺陷→颜色 Y | 无 | 缺陷固定去粉碗 |
| 积木 | 2x2 占 80%、2x4 占 20%，5 种颜色 | 无 | 同 |
| 杂物 | 每只碗 0–4 块，远处干扰物 0–3 个；**允许和某只碗同色** | 无 | 0–4、0–2 |
| 顶相机外参 | xy ±3 cm，高 ±3 cm，yaw ±8°，tilt ±5°，roll ±3°（均匀分布）；焦距 ±3% | 外环：每个分量 1.0–1.5 倍训练范围 | 固定 |
| 腕相机安装 | 位置 ±3 mm，角度 ±3° | 外环 ±5 mm / ±5° | 固定 |
| 光照 | 现有亮度和方向，加色温 3500–6500 K | 无 | 亮度、方向 |
| 桌面 | 颜色按 HSV 采样：色相 0–360°，饱和度 0–0.35，明度 0.35–0.80；纹理三选一：棋盘格（现有）、纯色、细噪声；格子尺寸沿用现有 | 高饱和桌面（饱和度 0.35–0.55）只在 v3-ood-appearance | 近中性灰 0.66±0.05，棋盘格 |
| 渲染器 | 前 1,000 个回合只用 MuJoCo；第 1,001 个回合起加入 Isaac RTX（比例见第 3 节） | 前 1,000 个回合期间，Isaac RTX 只做测试集 | MuJoCo |
| 指令改写 | 每类动作段 6 种说法，训练用其中 5 种 | 第 6 种只在测试集出现 | 1 种 |

**与 agent 实验的相机条件的关系。** agent 实验的 `jitter`（xy ±1 cm、yaw 2°）是“标定后的小误差”。这里的范围是“相机被重新装过一次”，大得多。教师在 realistic 模式下每回合都会对顶相机做手眼标定，所以外参变大也能正常工作。腕相机的安装误差教师并不知道，范围只放到 ±3 mm / ±3°：先用预检回合（任务 10）确认教师在这个范围内的成功率不下降，再决定是否放宽。

## 2. 语言与颜色的绑定

1. **调色板（8 色，已定包括白和黑）**：green、pink、blue、yellow、orange、purple、white、black。每色给定一个 RGB 中心值和一个抖动范围（色相 ±6°，饱和度 ±10%，明度 ±10%），这样同一个颜色词对应的是一段颜色，而不是一个固定的像素值。白和黑没有色相，教师感知按明度和饱和度分割。
   - **MuJoCo 渲染实测（2026-11-24，`C:/scratch/v3-colour-check/`）**：白碗和黑碗的轮廓、碗沿都清楚，没有糊成一团。实际风险有两个：一是浅色桌面（灰度约 0.73）上白碗过曝，边缘有锯齿亮点；二是黑碗和机械臂的黑色部件、阴影、ArUco 标记块的黑色部分颜色接近。
   - **因此加两条约束**：有白碗的场景，桌面灰度不超过 0.68，白碗明度抖动只往下走；黑碗的分割加碗沿高度过滤（只保留高于桌面、位于碗沿高度带的像素），排除机械臂和阴影。
   - **桌面与碗的冲突**：桌面饱和度 > 0.15 时，桌面色相和每只碗的色相差 ≥ 35°（否则教师按色相分割会把桌面当成碗）；有白碗时桌面明度 ≤ 0.68，有黑碗时 ≥ 0.40。不满足就重抽桌面，不重抽碗。
   - **Isaac RTX**：白碗和黑碗同样要在 Isaac 上标定色带（Task 3b），Isaac 的曝光和 MuJoCo 不同（粉碗饱和度只有约 20）。
2. **颜色对留出**：从 28 种颜色对里固定选 4 对（例如 blue–purple、orange–yellow、white–pink、green–black），只出现在测试集。训练时每种颜色都单独出现过，测试时考的是没见过的组合，测的是组合泛化，不是新词。
3. **路由与标签解耦**：每回合随机抽“良品→X、缺陷→Y”。搬运指令写颜色词，例如 “Carry the held brick over the purple bowl…”。“缺陷该去哪只碗”是上层 agent 的决定，VLA 只负责照指令执行。这样训练集里就不再有“缺陷 = 粉碗”这条捷径。
4. **多支搬运**：在 `place()` 开始搬运前的持砖检查点（现有 `TargetSwapTracer` 的分叉点），向场景里的**每一只**碗各分出一支搬运和释放（2 只碗就是 2 支，3 只就是 3 支，原目标那支即主回合）。每支只监督搬运和释放帧。同一画面、不同指令、不同去向，正是打破捷径需要的对比样本，而且成本很低：每支只多约 15 秒仿真。
5. **碗 one-hot 置零**：v3 数据里 `state[31:34]` 全为 0，碗只能从语言里得到。
6. **同色干扰**：杂物积木可以和某只碗同色。这迫使模型把“颜色词”和“碗的形状”结合起来，不能只找那个颜色的像素。教师感知在这种情况下靠“最大连通块 + 碗沿圆拟合”区分碗和积木。预检（任务 8）会渲染首帧并检查教师能否定位，定位失败的场景剔除，并记录剔除率。
7. **指令改写**：每类动作段 6 种说法，颜色词和视角编号作为占位符。训练从前 5 种里随机抽（每个回合抽一次，整回合统一），第 6 种只用于测试。所有改写都要经过分词器长度检查。

## 3. 相机与外观

- **顶相机**：按第 1 节的范围采样。预检把桌面工作区的四角、两只碗中心和积木投影到图像里，任何一点出画或落在图像边缘 5% 以内，就重抽该案例的相机（重抽次数有上限，并会记录）。
- **前相机**：学生看不到它，只用于回放查看，维持现有的抖动范围。
- **腕相机**：见第 1 节。
- **渲染器混合（已定）**：
  - 第 1–1,000 个回合（含试采的 200 个）只用 MuJoCo 训练；这期间 Isaac RTX 只做测试集 v3-isaac，用来测渲染器差异。
  - 第 1,001 个回合起，Isaac RTX 回合加入训练集。比例在采第 1,001 个回合之前，根据 v3-isaac 和 v3-val 的跟随率差距来定：差距 ≥ 10 个百分点取 40%，5–10 个百分点取 25%，小于 5 个百分点取 15%。这条规则先写进第 1,000 回合的中期报告，再看数据。
  - Isaac 回合的吞吐量要单独实测。不跑 Qwen 时显存有余，可能能开 2 个 Isaac 进程。参考值：单进程 3–5 分钟/回合，约 15 回合/小时。
  - 依据记忆 [act-rtx-zeroshot-result]：RTX 不能提升抓取精度，加入它是为了外观上的稳健性。
- **图像增强**：沿用 `StudentImageAugmentor`，训练时用 `--gpu-augment`。色相抖动保持在 ±0.05（约 ±18°），小于调色板最小色相间隔的一半，免得增强把一种颜色变成另一种。任务 5 里有测试守住这一点。

## 4. 对抗协变量偏移：静态数据里的纠偏样本

ACT 的主要瓶颈是闭环漂移：闭环误差约是开环的 3.5 倍。纯示范数据会重演这个问题。因此在静态数据里加入纠偏样本：

- **动作噪声注入**：抽 40% 的回合，在 `grasp`、`lift`、`carry_to_bowl`、`release_in_bowl`、`scan_view` 这几类动作段里，给**执行的**关节目标加时间相关噪声（OU 过程，σ 取 0.01–0.03 rad，相关时间 0.3 s，夹爪不加）；**记录的标签**仍是教师的原始目标。教师给的是绝对关节目标，所以下一帧的原始目标本身就是把偏离拉回来的纠偏动作。
- **与 DAgger 的区别**：不需要学生在线推理。噪声帧在数据里用 `mask.noisy` 标出，训练时可以单独加权。
- **原地抓取恢复片段**：现有 `TakeoverTwin` 的原地恢复生成器可以在教师执行时被主动触发（人为让抓取偏 3–8 mm 再纠正），作为第二种纠偏样本。由任务 7 的开关控制，默认关闭，试采后再决定是否打开。

## 5. 采集前的场景预检

`make_scene_suite_v3.py` 生成套件时，在跑任何教师回合之前就输出：

1. **分布表**：每个因子的直方图或计数；碗颜色 × 是否为目标 × 碗位置（左/右/中）的交叉表；路由规则计数；颜色对计数（训练集里必须不含留出的颜色对）。
2. **静态检查**：现有的可达性和间距检查；第 3 节的相机可见性检查；碗之间、碗与桌面之间的色相间隔 ≥ 35°；指令长度。
   - **检测器对桌面颜色的敏感性**：缺陷检测器 v7bot 是在灰色桌面的数据上训的，彩色桌面可能在腕部裁剪里引入误报。预检从训练套件里抽 60 个良品案例（覆盖桌面色相的 6 个区间），用教师的腕扫视图跑检测器，按色相区间报告良品误报率。任何一个区间的误报率比灰色桌面高出 5 个百分点以上，就收窄该区间的桌面饱和度上限，或者把它移进 OOD 集，并在报告里写明。
3. **感知检查**：随机抽 10% 的案例渲染首帧，用 realistic 感知定位碗和积木，报告定位误差和失败率。各颜色的碗定位误差中位数 ≤ 1 cm 才算通过。
4. **平衡阈值**：每种颜色当目标的次数，与均值相差不超过 ±15%；碗左右与目标颜色的相关系数绝对值 < 0.05。不满足就换种子段重新生成，不手调单个案例。

全部输出写到 `E:/Armigo_data_v3/suites/<suite>/preflight.md` 和 `preflight.json`，人工看过之后才开始采集。

## 6. 数据量与预算

| 项 | 估计 | 依据 |
|---|---|---|
| 教师主回合 | 2,000 个（第 1–1,000 个只用 MuJoCo，第 1,001 个起混入 Isaac RTX） | ACT 用了 310 个回合；SmolVLA 冻结 VLM、只训动作专家，需要更多样化的场景而非更多重复 |
| 搬运分支 | 每回合多 1–2 支，约 2,600 支，只含搬运和释放帧 | 第 2 节第 4 点 |
| 仿真时间 | 主回合 3–5 分钟、分支约 15 秒；8 个进程并行约 100 回合/小时 → 约 20 小时 | 配对采集的实测 |
| 导出 | 只导出学生的 3 路相机，约 37 MB/回合（d_union 实测：5 路共 23 GB / 386 回合，其中 front 和 top_full 占一半） | 任务 9 |
| 磁盘 | 主回合约 75 GB，分支约 15 GB，原始 mkv 分批导出后删除；数据根目录 `E:/Armigo_data_v3` | E 盘剩 231 GB（D 盘只剩 43 GB，不用） |
| 训练 | 批 32、0.42 s/步，6 万步约 7 小时 | 本轮实测 |

**分阶段（已定）**：
1. 试采 200 个回合（MuJoCo，Task 10），按判据验收，并跑失败闭环第 1 轮；
2. 第 201–1,000 个回合（MuJoCo），之后出中期报告：v3-val 和 v3-isaac 的跟随率，以及 Isaac 的比例；
3. 第 1,001–2,000 个回合，MuJoCo 加 Isaac RTX。

## 7. 评测集与判据

| 集合 | 内容 | 规模 | 用途 |
|---|---|---|---|
| v3-val | 训练分布内的新种子 | 200 | 选权重、调参 |
| v3-ood-cam | 相机外环 | 100 | 外参泛化 |
| v3-ood-color | 留出的颜色对 | 100 | 颜色组合泛化 |
| v3-ood-lang | 留出的第 6 种改写 | 100 | 改写泛化 |
| v3-ood-appearance | 高饱和桌面 | 100 | 外观泛化 |
| v3-isaac | Isaac RTX 渲染，训练分布内的新种子，含白碗和黑碗 | 60 | 渲染器差异；前 1,000 个回合期间只做测试 |
| v3-test（封存） | 训练分布内，开发结束后才开 | 200 | 最终报告 |
| val2–val5 | 现有固定相机布局 | 各 100 | 与 ACT 和 fit30k 可比 |

**指标**：

- **碗跟随率**：`bowl_swap_probe.py --prefix teacher` 推广到 K 只碗，从同一持砖检查点对每只碗各跑一次搬运，统计段末离指令碗最近的比例。
- **闭环分拣成功率**：`eval_closed_loop.py` 默认设置。
- **首次夹稳率、碰撞**：同现有报告。

**判据**（训练前写入验收文档，运行后不改）：

- v3-val 上跟随率 ≥ 90%；
- 三个 OOD 集上跟随率比 v3-val 低不超过 10 个百分点；
- val2–val5 上闭环分拣成功率不低于 fit30k（配对 McNemar）。

---

## 文件结构

| 文件 | 改动 | 职责 |
|---|---|---|
| `src/digital_twin_armigo/mujoco_cell/palette.py` | 新 | 调色板、颜色抖动、HSV 色带、颜色对留出表 |
| `src/digital_twin_armigo/mujoco_cell/scene_v3.py` | 新 | `SceneV3` 规格、`sample_scene_v3(seed, split)`、`apply_scene_v3(layout, scene)` |
| `src/digital_twin_armigo/mujoco_cell/layout.py` | 改 | `Layout.bowl_rgb`、`Layout.bowl_names`、`Layout.extra_bowls`（缺省时与现在相同） |
| `src/digital_twin_armigo/mujoco_cell/scene.py` | 改 | 碗颜色从 Layout 读；干扰碗 |
| `src/digital_twin_armigo/agent/twin_mujoco.py` | 改 | `sample_case` 的 `scene` 选项；`case_cameras` 的 `wide` 模式；路由钩子 `target_bowl` 读路由规则 |
| `src/digital_twin_armigo/agent/perception.py` | 改 | `locate_bowls(..., bands=None)`：按场景颜色生成色带，按槽位返回 |
| `src/digital_twin_armigo/vla/contract.py` | 改 | `INSTRUCTION_VARIANTS`、`format_instruction(kind, params, variant=0)`、v3 碗 one-hot 置零 |
| `src/digital_twin_armigo/vla/paired_carry.py` | 改 | `run_branches()`：向每只碗各分一支 |
| `src/digital_twin_armigo/agent/student_rollout.py` | 改 | `ExpertTracer(action_noise=...)` 噪声注入，`mask.noisy` |
| `src/digital_twin_armigo/agent/twin_record.py` | 改 | 导出时去掉 `FORBIDDEN_CAMERA_NAMES` 的相机；`mask.noisy` 列 |
| `scripts/vla/make_scene_suite_v3.py` | 新 | 生成套件和预检报告 |
| `scripts/vla/collect_static_v3.py` | 新 | 批量采集：主回合 + 分支 + 分批导出校验（复用 `collect_paired_carry.py` 的导出和合并） |
| `scripts/vla/bowl_swap_probe.py` | 改 | K 只碗的跟随探针 |
| `tests/test_palette_scene_v3.py`、`tests/test_perception_palette.py`、`tests/test_instruction_variants.py`、`tests/test_action_noise.py`、`tests/test_export_cameras.py`、`tests/test_scene_suite_v3.py` | 新 | 各任务的测试 |

---

### Task 1：调色板与场景规格 v3

**Files:**
- Create: `src/digital_twin_armigo/mujoco_cell/palette.py`、`src/digital_twin_armigo/mujoco_cell/scene_v3.py`
- Test: `tests/test_palette_scene_v3.py`

**Interfaces:**
- Produces:
  - `PALETTE: dict[str, Colour]`，其中 `Colour(name: str, rgb: tuple[float, float, float], hue_deg: float | None)`；
  - `jitter_rgb(colour, rng) -> tuple[float, float, float]`；
  - `hue_gap(a: str, b: str) -> float`（无色相的颜色与任何颜色的间隔记为 180）；
  - `HELD_OUT_PAIRS: frozenset[frozenset[str]]`；
  - `@dataclass SceneV3(normal: str, defect: str, extra: str | None, rgb: dict[str, tuple], table_rgb: tuple[float, float, float], table_texture: str, camera: dict[str, float], wrist: tuple[float, ...], light_k: float, variant: int)`；`table_texture` 取 `"checker" | "plain" | "noise"`；
  - `sample_scene_v3(seed: int, split: str) -> SceneV3`，其中 `split` 取 `"train" | "val" | "ood-color" | "ood-cam" | "ood-lang" | "ood-appearance" | "test"`。

- [ ] **Step 1：写失败的测试**

```python
# tests/test_palette_scene_v3.py
import itertools

import pytest

from digital_twin_armigo.mujoco_cell.palette import HELD_OUT_PAIRS, PALETTE, hue_gap
from digital_twin_armigo.mujoco_cell.scene_v3 import sample_scene_v3


def test_palette_has_eight_colours_and_four_held_out_pairs():
	assert set(PALETTE) == {"green", "pink", "blue", "yellow", "orange", "purple", "white", "black"}
	assert len(HELD_OUT_PAIRS) == 4


def test_scene_is_deterministic():
	assert sample_scene_v3(3000001, "train") == sample_scene_v3(3000001, "train")


@pytest.mark.parametrize("split", ["train", "val", "ood-cam", "ood-lang", "test"])
def test_non_colour_splits_never_use_held_out_pairs(split):
	for seed in range(3000001, 3000401):
		s = sample_scene_v3(seed, split)
		names = [s.normal, s.defect] + ([s.extra] if s.extra else [])
		for a, b in itertools.combinations(names, 2):
			assert frozenset((a, b)) not in HELD_OUT_PAIRS
			assert hue_gap(a, b) >= 35.0


def test_ood_colour_split_always_contains_a_held_out_pair():
	for seed in range(3000001, 3000101):
		s = sample_scene_v3(seed, "ood-color")
		assert frozenset((s.normal, s.defect)) in HELD_OUT_PAIRS


def test_train_routing_is_balanced():
	scenes = [sample_scene_v3(seed, "train") for seed in range(3000001, 3002001)]
	counts = {c: sum(s.defect == c for s in scenes) for c in PALETTE}
	mean = len(scenes) / len(PALETTE)
	assert all(abs(n - mean) <= 0.25 * mean for n in counts.values())


def test_table_colour_respects_bowls():
	import colorsys

	for seed in range(3000001, 3000401):
		s = sample_scene_v3(seed, "train")
		h, sat, v = colorsys.rgb_to_hsv(*s.table_rgb)
		assert sat <= 0.35 and 0.35 <= v <= 0.80
		if "white" in (s.normal, s.defect, s.extra):
			assert v <= 0.68
		if sat > 0.15:
			for name in (s.normal, s.defect, s.extra):
				if name and PALETTE[name].hue_deg is not None:
					d = abs(h * 360 - PALETTE[name].hue_deg) % 360
					assert min(d, 360 - d) >= 35.0


def test_camera_ranges():
	for seed in range(3000001, 3000201):
		cam = sample_scene_v3(seed, "train").camera
		assert abs(cam["dx_m"]) <= 0.03 and abs(cam["dyaw_deg"]) <= 8.0
		cam = sample_scene_v3(seed, "ood-cam").camera
		assert max(abs(cam["dx_m"]) / 0.03, abs(cam["dyaw_deg"]) / 8.0, abs(cam["dtilt_deg"]) / 5.0) >= 1.0
```

- [ ] **Step 2：运行测试，确认失败**（`ModuleNotFoundError`）

Run: `.venv/Scripts/python -m pytest tests/test_palette_scene_v3.py -v`

- [ ] **Step 3：实现**

要点：
- `sample_scene_v3` 用 `np.random.default_rng([seed, 0x5C3, split_index])`，与 `sample_layout` 的随机流分开，所以不会改变旧布局。
- 颜色先无放回抽，不满足间隔或命中留出组合就重抽（最多 100 次，超过则抛错）。
- `ood-cam` 至少有一个分量落在 1.0–1.5 倍区间：先均匀选出这个分量，再从外环区间采样；其余分量照训练范围采样。

- [ ] **Step 4：运行测试，确认通过**

- [ ] **Step 5：Commit** `git add src/digital_twin_armigo/mujoco_cell/palette.py src/digital_twin_armigo/mujoco_cell/scene_v3.py tests/test_palette_scene_v3.py`

### Task 2：Layout 与 MJCF 支持显示颜色和干扰碗

**Files:**
- Modify: `src/digital_twin_armigo/mujoco_cell/layout.py`（Layout 数据类）、`src/digital_twin_armigo/mujoco_cell/scene.py:583-584`（碗）、`src/digital_twin_armigo/agent/twin_mujoco.py:160-190`（`sample_case`、`_case_options`）
- Test: `tests/test_palette_scene_v3.py`（追加）

**Interfaces:**
- Consumes: `sample_scene_v3`、`SceneV3`
- Produces:
  - `Layout.bowl_rgb: dict[str, tuple] | None`；
  - `Layout.bowl_names: dict[str, str] | None`（例如 `{"green": "purple", "pink": "blue"}`）；
  - `Layout.extra_bowls: list[tuple[str, float, float, tuple]]`；
  - `sample_case(seed, positive, face, {"scene": "v3", "split": "train"})`。

- [ ] **Step 1：写失败的测试**

```python
from digital_twin_armigo.agent.twin_mujoco import sample_case
from digital_twin_armigo.mujoco_cell.scene import scene_xml


def test_old_cases_unchanged_by_v3_fields():
	lay = sample_case(1100001, False, 5)
	assert lay.bowl_rgb is None and lay.bowl_names is None and lay.extra_bowls == []
	assert 'rgba="0.25 0.65 0.3 1"' in scene_xml(lay)  # green bowl colour as before


def test_v3_case_paints_bowls_by_scene():
	lay = sample_case(3000001, True, 2, {"scene": "v3", "split": "train"})
	xml = scene_xml(lay)
	assert set(lay.bowl_names) == {"green", "pink"}
	for slot in ("green", "pink"):
		r, g, b = lay.bowl_rgb[slot]
		assert f'rgba="{r:.3f} {g:.3f} {b:.3f} 1"' in xml
	assert all(np.hypot(x - lay.brick_xy[0], y - lay.brick_xy[1]) > 0.13 for _, x, y, _ in lay.extra_bowls)
```

（`scene_xml(layout, ...)` 在 `scene.py:511`，返回完整 MJCF 字符串。测试文件顶部需要 `import numpy as np`。）

- [ ] **Step 2：运行，确认失败**
- [ ] **Step 3（桌面）：** `scene.py:524-574` 的桌面纹理按 `layout.table_texture` 选择：`checker` 保持现有写法；`plain` 用 `builtin="flat"`；`noise` 用 `builtin="flat" mark="random" random="0.02" markrgb=<桌面色 × 0.85>`。`table_texture` 为 None 时输出与现在逐字相同。Isaac 桥（`isaac_bridge/client.py:342`）目前只传桌面 RGB，所以 Isaac 场景统一用纯色桌面，并在场景记录里标明。
- [ ] **Step 3：实现。** 干扰碗的位置沿用 `sample_layout` 的可达和间距规则，在主布局生成后单独采样，使用 v3 的随机流。干扰碗用同一个碗网格，带碰撞（积木可能被放进去）。
- [ ] **Step 4：跑新测试，以及完整测试集里与布局和场景相关的部分**：`.venv/Scripts/python -m pytest tests -k "layout or scene or twin" -q`。旧案例的测试必须全部通过。
- [ ] **Step 5：Commit**

### Task 3：教师感知按场景颜色定位碗

**Files:**
- Modify: `src/digital_twin_armigo/agent/perception.py:221-260`（`locate_bowls`）、`src/digital_twin_armigo/agent/twin_mujoco.py`（`_sense_bowls` 把 `layout.bowl_rgb` 转成色带传入）
- Test: `tests/test_perception_palette.py`

**Interfaces:**
- Produces:
  - `bands_for(rgb_by_slot: dict[str, tuple]) -> dict[str, tuple[int, int, int, int]]`：有色相的颜色给出色相区间、饱和度下限和明度下限；白和黑给出特殊色带，编码为 `(-1, -1, smin, vmin)` 或 `(-2, -2, smax, vmax)`；
  - `locate_bowls(rgb, depth, k, world_from_cam, *, engine="mujoco", bands=None)`：`bands=None` 时行为与现在逐位相同。

- [ ] **Step 1：写失败的测试。** 对 8 种颜色各取 3 个种子：渲染 v3 场景的顶相机首帧，用真值相机位姿调用 `locate_bowls`，断言每个槽位都能找到，且误差 ≤ 0.01 m。另加一个回归测试：一个旧案例在 `bands=None` 时，结果与改动前保存下来的数值完全相同（先在未改动的代码上算出数值，写进测试）。
- [ ] **Step 2：运行，确认失败**
- [ ] **Step 3：实现。**
  - 色相区间取 `hue ± (6° 抖动 + 8° 余量)`，换算成 OpenCV 的 0–179 刻度。
  - 白色用“饱和度 < 40 且明度 > 200”，黑色用“明度 < 50”。
  - 同色积木的干扰靠现有的碗沿圆拟合内点筛选来排除；连通块面积下限 `BOWL_MIN_PIXELS` 不变。
- [ ] **Step 4：运行测试。** 若某种颜色有失败，查清是色带问题还是遮挡造成的，在测试报告里记录；不要为了过测试去放宽阈值。
- [ ] **Step 5：Commit**

### Task 3b：Isaac RTX 的调色板色带

**Files:**
- Modify: `src/digital_twin_armigo/agent/perception.py`（`BOWL_HSV_BY_ENGINE["isaac"]` 推广为 `bands_for(rgb_by_slot, engine="isaac")`：每种颜色给一组 Isaac 专用的色带偏移）
- Create: `scripts/isaac/calibrate_palette.py`（在 Isaac 上为 8 种颜色各渲染 5 个开发种子的顶相机首帧，测出每种颜色碗的 HSV 分布，写入 `config/sim/isaac_palette_bands.json`）
- Test: `tests/test_perception_palette.py`（追加：读取 JSON，断言 8 种颜色都有色带，并且色带两两不重叠）

- [ ] **Step 1：** 用开发种子（不和任何测试集重叠）在 Isaac 上渲染并测量；标定只能用开发帧。
- [ ] **Step 2：** 写测试并实现 `bands_for(..., engine="isaac")`。
- [ ] **Step 3：** 在 10 个 Isaac 开发种子上用 realistic 感知定位碗，各颜色的定位误差中位数 ≤ 1.5 cm（Isaac 的手眼标定精度较低，2.5 mm 的积木误差来自同一原因，这里放宽）。
- [ ] **Step 4：Commit**

### Task 4：相机外参的大范围采样与可见性检查

**Files:**
- Modify: `src/digital_twin_armigo/agent/twin_mujoco.py:78-160`（`CAMERA_MODES` 加入 `"wide"`，`case_cameras` / `case_wrist_offset` 在 v3 下从 `SceneV3.camera` / `SceneV3.wrist` 取值）
- Create: `src/digital_twin_armigo/mujoco_cell/visibility.py`
- Test: `tests/test_palette_scene_v3.py`（追加）

**Interfaces:**
- Produces: `workspace_visible(layout, margin_frac=0.05) -> tuple[bool, list[str]]`，返回是否全部可见，以及出画的点名列表（`"table_corner_0"`、`"green_bowl"`、`"brick"` 等）。

- [ ] **Step 1：测试。**
  - `FIXED_TOP` 下 `workspace_visible` 为真；
  - 人为把 yaw 设成 +40° 时为假，并且列表里有点名；
  - v3 训练种子 3000001–3000500 里，`sample_case` 重抽相机后全部可见，重抽次数中位数为 0。
- [ ] **Step 2–4：实现与验证。** 投影使用 `CameraPose` 的现有内参和外参。重抽用同一种子的后续随机数（确定性），最多 20 次，超过则抛 `RuntimeError`。
- [ ] **Step 5：Commit**

### Task 5：指令改写与 v3 碗 one-hot 置零

**Files:**
- Modify: `src/digital_twin_armigo/vla/contract.py:49-120`（`SEGMENT_TEMPLATES` 保留为第 0 种说法；新增 `INSTRUCTION_VARIANTS: dict[str, tuple[str, ...]]`，每类 6 种）
- Modify: `src/digital_twin_armigo/vla/dataset.py`（读取 `meta/segments.json` 里的 `variant` 和 `scene` 字段；v3 回合把 `state[31:34]` 置零）
- Test: `tests/test_instruction_variants.py`

**Interfaces:**
- Produces:
  - `format_instruction(kind, params, variant=0) -> str`：`variant=0` 时输出与现在逐字相同；
  - `TRAIN_VARIANTS = (0, 1, 2, 3, 4)`、`HELD_OUT_VARIANT = 5`；
  - `encode_state(..., bowl=None)`：`bowl=None` 时碗槽位为 0（现有行为）。v3 导出时始终传 `bowl=None`。

- [ ] **Step 1：测试**

```python
import pytest

from digital_twin_armigo.vla.contract import INSTRUCTION_VARIANTS, SEGMENT_TEMPLATES, format_instruction


def test_variant_zero_is_the_current_template():
	for kind, text in SEGMENT_TEMPLATES.items():
		assert INSTRUCTION_VARIANTS[kind][0] == text


def test_six_variants_each_and_bowl_word_present():
	for kind, variants in INSTRUCTION_VARIANTS.items():
		assert len(variants) == 6 and len(set(variants)) == 6
	for v in range(6):
		assert "purple" in format_instruction("carry_to_bowl", {"bowl": "purple"}, variant=v)


def test_token_length_under_40():
	transformers = pytest.importorskip("transformers")
	tok = transformers.AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct", local_files_only=True)
	params = {"view": 4, "views": 4, "bowl": "orange", "roll_deg": -135.0}
	for kind, variants in INSTRUCTION_VARIANTS.items():
		for v in range(len(variants)):
			assert len(tok(format_instruction(kind, params, variant=v))["input_ids"]) <= 40, (kind, v)


def test_hue_augmentation_cannot_cross_palette_gap():
	from digital_twin_armigo.vla.dataset import StudentImageAugmentor

	assert StudentImageAugmentor().hue * 360.0 < 35.0 / 2
```

- [ ] **Step 2–4：实现与验证。** 改写要人工逐条读一遍，不能有歧义：不能出现 “the other bowl”，也不能用和颜色无关的方位词替代颜色。
- [ ] **Step 5：Commit**

### Task 6：路由规则与多支搬运

**Files:**
- Modify: `src/digital_twin_armigo/agent/twin_mujoco.py`（`target_bowl(label)` 在 v3 下返回槽位，指令里的颜色词取 `layout.bowl_names[slot]`；干扰碗也可以作为分支目标，用 `"extra:<i>"` 表示）
- Modify: `src/digital_twin_armigo/vla/paired_carry.py`（新增 `run_branches(case, out, detector) -> dict`，向每只碗各分一支；保留 `run_pair` 作为兼容入口）
- Test: `tests/test_vla_paired_carry.py`（追加）

**Interfaces:**
- Produces: `run_branches` 返回 `{"fork_frame": int, "branches": [{"target": str, "colour": str, "verdict": str, "supervised_frames": int, "dir": str}], "fork_state_match": bool}`。

- [ ] **Step 1：测试。**
  - 冒烟一个 3 只碗的 v3 案例（标记为慢测试，加 `@pytest.mark.slow`）：分出 3 支，三支在分叉点的状态误差都为 0，各支的目标两两不同，指令颜色词与目标碗颜色一致，每支的受监督帧只出现在分叉之后。
  - 单元测试：给出两个槽位加一只干扰碗，`target_bowl` 和颜色词的映射符合路由规则。
- [ ] **Step 2–4：实现与验证**
- [ ] **Step 5：Commit**

### Task 7：动作噪声注入

**Files:**
- Modify: `src/digital_twin_armigo/agent/student_rollout.py`（`ExpertTracer.__init__(..., action_noise: ActionNoise | None = None)`；在 `_execute` 包装生成器：执行 `target + noise`，记录 `target`）
- Modify: `src/digital_twin_armigo/agent/twin_record.py`（`mask.noisy` 列，导出到 LeRobot）
- Test: `tests/test_action_noise.py`

**Interfaces:**
- Produces: `@dataclass ActionNoise(sigma_rad: float, tau_s: float, kinds: tuple[str, ...], seed: int)`；记录器提供 `record.noisy: list[bool]`，导出为 `mask.noisy`。

- [ ] **Step 1：测试。** 在一个固定案例上：
  - 关闭噪声时，`actions.npz` 与改动前逐位相同；
  - 打开噪声时，记录的 `action` 等于教师生成器给出的目标（用 `ReplayPolicy` 回放生成器序列比对），执行的目标与记录值之差的标准差在 σ 的 0.5–1.5 倍之间；
  - 夹爪维（第 5 维）差为 0；
  - 同一种子两次运行逐位相同。
- [ ] **Step 2–4：实现与验证。** OU 离散化：`n[t+1] = n[t] * exp(-dt/tau) + sigma * sqrt(1 - exp(-2 dt/tau)) * N(0, 1)`，`dt = 1/30`。每段开始时噪声重置为 0。
- [ ] **Step 5：Commit**

### Task 8：套件生成与预检报告

**Files:**
- Create: `scripts/vla/make_scene_suite_v3.py`
- Test: `tests/test_scene_suite_v3.py`

**Interfaces:**
- Consumes: Task 1–5 的全部接口
- Produces:
  - 命令行：`make_scene_suite_v3.py --split train --seeds 3000001-3002000 --out config/vla/v3/train.json --preflight-dir E:/Armigo_data_v3/suites/train`；
  - 套件 JSON 的案例为 `[seed, positive, face, {"scene": "v3", "split": "train"}]`，类别配比沿用现有规则（良品 3 份 : 每个缺陷面 1 份）；
  - `preflight.json` 包含 `balance`、`colour_pairs`、`camera_hist`、`visibility_resamples`、`perception_check`、`token_max` 和 `pass: bool`。

- [ ] **Step 1：测试。** 用 50 个种子的小套件：
  - 生成成功，`pass` 字段存在；
  - 训练套件不含留出的颜色对；
  - 各切分的种子段互不重叠（val 3100001–，ood 3200001–，test 3900001–）；
  - `perception_check` 的抽样数等于案例数的 10%（向上取整）。
- [ ] **Step 2–4：实现与验证。** 感知检查要渲染，测试里用 `--perception-sample 0.1 --max-perception 5` 限制时间。
- [ ] **Step 5：Commit**

### Task 9：导出只保留学生相机

**Files:**
- Modify: `src/digital_twin_armigo/agent/twin_record.py`（LeRobot 导出：`--student-cameras-only` 开关，打开时跳过 `FORBIDDEN_CAMERA_NAMES`）
- Modify: `scripts/vla/collect_paired_carry.py` 的 `_export_shard`（传入该开关）
- Test: `tests/test_export_cameras.py`

- [ ] **Step 1：测试。** 用一个小的内存回合导出：打开开关时 `meta/info.json` 的 features 不含 `observation.images.front` 和 `observation.images.top_full`；`SegmentChunkDataset` 能正常读取；关闭开关时与现在相同。
- [ ] **Step 2–4：实现与验证**
- [ ] **Step 5：Commit**

### Task 10：试采 200 回合与验收

**Files:**
- Create: `scripts/vla/collect_static_v3.py`（复用 `collect_paired_carry.collect / export_pending / merge`，每个案例调用 `run_branches`，40% 的案例打开 `ActionNoise(0.02, 0.3, ("grasp","lift","carry_to_bowl","release_in_bowl","scan_view"), seed)`）
- Create: `docs/vla/STATIC_V3_PILOT_<日期>.md`（试采前写好判据）

- [ ] **Step 1：写判据（采集前提交）：**
  - 教师主回合的分拣成功率 ≥ fit30k 训练集教师成功率减 5 个百分点；
  - `BOWL_NOT_FOUND` ≤ 2%；
  - 按颜色分组的教师成功率，最低组不低于均值减 10 个百分点；
  - 噪声回合的教师成功率不低于无噪声回合减 5 个百分点；
  - 各分支的受监督帧数中位数在 200–450；
  - 导出读回校验全部通过。
- [ ] **Step 2：** 生成 `config/vla/v3/pilot.json`（200 个训练种子），跑预检，人工检查 `preflight.md`。
- [ ] **Step 3：** 采集：`collect_static_v3.py --suite config/vla/v3/pilot.json --out E:/Armigo_data_v3/pilot --workers 8 --export --student-cameras-only`。
- [ ] **Step 4：** 把失败回合交给失败闭环（[2026-11-24-agent-failure-loop.md](2026-11-24-agent-failure-loop.md) F1–F2）出失败台账。按判据给出 `pass` 或 `fix_then_repilot`。
- [ ] **Step 5：** 冒烟训练：SmolVLA 用 `--gpu-augment`、批 32、5k 步，从探针轮的最佳权重热启动，在 v3-val 的 50 个案例上跑 K 碗跟随探针。这一步只验证流程，不作为结论。
- [ ] **Step 6：** 记录结果并提交文档。

### Task 11：正式采集、训练与评测

- [ ] **Step 1：** 教师冻结（失败闭环 F7）；生成训练、验证和全部 OOD 套件，跑预检。
- [ ] **Step 2：** 采集 2,000 个主回合和全部分支。每 200 个回合导出、合并一次，并更新数据记录。
- [ ] **Step 3：** 写训练前的验收文档：判据照抄第 7 节，另写明步数、批大小、学习率和选权重规则。
- [ ] **Step 4：** 训练 SmolVLA：从探针轮的最佳权重热启动，6 万步，批 32，`--gpu-augment --drop-unsupervised`，噪声帧权重为 1（不额外加权）。
- [ ] **Step 5：** 在 v3-val、三个 OOD 集、v3-isaac 和 val2–val5 上评测。最后开封存的 v3-test。
- [ ] **Step 6：** 写报告，更新记忆。

---

## 已定事项（2026-11-24）

1. **调色板包括白和黑。** MuJoCo 实测能清楚渲染，加了桌面亮度和碗沿高度两条约束；Isaac 单独标定色带（Task 3b）。
2. **教师不用真值兜底。** 找不到碗就按失败收录并终止回合，进失败台账，不进训练集。
3. **腕相机外参先取 ±3 mm / ±3°**，试采后再看是否放宽。
4. **先试采 200 个回合。**
5. **前 1,000 个回合（含试采）Isaac 只做测试集**，第 1,001 个回合起加入训练，比例按第 3 节的规则定。
