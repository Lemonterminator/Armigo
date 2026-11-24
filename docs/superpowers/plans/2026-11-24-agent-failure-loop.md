# Agent 失败收集与 MCP 工具递归改进 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把“跑一批 → 找失败 → 修工具 → 重跑”做成一个有固定输入输出、可以重复多轮的闭环。每一轮都能说清楚修好了什么、有没有带来回退、是否在新布局上也成立；教师工具的改进收敛之后再冻结，用来采集静态数据集。

**Architecture:** 每一轮分六步：

1. **采集**：用 v2 runner 跑一批，再做 inventory 扫描。
2. **自动分诊**：按规则从事件里给每个失败回合判一个暂定根因。
3. **台账**：按失败阶段、失败码、根因和场景因子聚类，按“频次 × 影响”排序。
4. **复现库**：每个要修的簇冻结一组可以确定性复现的案例。
5. **修复**：每个修复都带一条从复现案例提取出来的单元测试。
6. **验证**：三道关，分别是复现库修复率、旧成功案例的配对回退检查、全新种子上的非劣检验。

人工审计（viewer 加 audit 数据库）抽查自动分诊的结果，校准分诊规则。

**Tech Stack:** Python 3.14，MuJoCo，LangGraph，FastMCP（`step_mcp.py` / `twin_mcp.py`），SQLite（inventory、audit），pytest，ruff（tab 缩进）。

**Spec:** 本文第 0–3 节。相关的现有组件：
- [EXPERIMENTS_V2.md](../../EXPERIMENTS_V2.md)：runner、report、可续跑和尝试目录；
- `inventory/`：`scan`、`query`、`audit`；
- `viewer/`：8767 端口，配对链接、3D 回放；
- `agent/twin_resume.py`：从检查点恢复；
- `agent/twin_search.py`：成对策略搜索；
- `experiments/contract.py`：`ROOT_CAUSES`、`AUDIT_STATES`。

## Global Constraints

- **开发和验证分开。** 修复只能参考 dev 和复现库里的失败；判断修复是否成立要用全新种子段（每轮新开一段，不复用）。封存套件（`p1_sealed`、`p2_sealed`、v3-test）不参与闭环。
- **判据先写。** 每个修复在重跑之前，先在本轮台账里写明目标簇、预期的修复率和三道验收关的阈值；运行之后不改。
- **工具只给传感器信息。** `step_mcp` / `twin_mcp` 工具的返回值必须通过 `assert_sensor_only`，新增字段（置信度、失败原因）也一样。
- **审计只追加。** 人工裁定写进 `reviews.sqlite`，不覆盖旧裁定（`audit.append` 的 `expected_latest_id` 机制）。
- **可复现。** 复现库的案例在同一提交上新开进程运行两次，结果必须逐位相同，否则不能进库（Isaac 只要求终局相同）。
- **一次修一个簇。** 一个修复对应一个提交和一份验收记录；不把多个修复混在一次验证里。
- **代码风格与提交。** 同数据集计划：tab 缩进，ruff 通过；只在本地分支提交，不推送；提交信息以 `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` 结尾。

---

## 0. 和数据集计划的先后关系

```
v3 场景设计（数据集 Task 1–9）
   └─> 试采 200（数据集 Task 10）= 闭环第 1 轮的输入
          └─> 台账 → 复现库 → 修复 → 三道关
                 └─> 第 2 轮（全新种子） … 直到满足停止条件
                        └─> 教师冻结（F7）
                               └─> 正式采集（数据集 Task 11）
```

教师的系统性错误会被学生原样模仿。例如顶视 yaw 错 20–25° 导致有些面没看清，这种示范一旦进了数据集就改不掉了。所以**先在 v3 场景上把教师修到收敛，再冻结、再采集**。闭环也可以继续用在 MuJoCo 和 Isaac 的 agent 实验上，流程相同，只是套件不同。

## 1. 一轮闭环的输入和输出

| 步骤 | 输入 | 输出（都在 `E:/Armigo_data_v3/loop/round-NN/`） |
|---|---|---|
| 采集 | 套件、提交号、freeze 文件（可选） | `runs/…` 回合目录，inventory 数据库 |
| 自动分诊 | 回合的 `events.jsonl`、`result`、`episode.json` | `triage.jsonl`：每回合一行，含阶段、失败码、暂定根因、证据和规则编号 |
| 人工抽查 | 每个簇随机抽 5 例（不足 5 例全看） | `reviews.sqlite` 里的裁定；`triage_agreement.json`：规则与人工的一致率 |
| 台账 | 分诊结果加人工裁定 | `ledger.md` / `ledger.json`：簇、频次、影响、例子和 viewer 链接 |
| 复现库 | 台账里选中要修的簇 | `config/agent_bench/failure_bank/round-NN.json`：案例加复现检查结果 |
| 修复 | 一个簇 | 一个提交，内含从复现案例提取的单元测试 |
| 验证 | 修复后的提交 | `verify-<簇>.md`：三道关的数字和判定 |

**影响**的定义：这个簇的回合数 × 每回合的损失。损失按物理结果算：放错碗或丢失记 1，`unknown` 但砖在桌上记 0.5，标签对但证据不足记 0.3。

## 2. 三道验收关（每个修复都要过）

1. **复现库修复率**：修复后，目标簇复现案例的成功比例 ≥ 判据（默认 70%）。
2. **回退检查**：修复前成功的案例里固定抽一组回退集（默认 100 个，跨簇和跨场景因子分层抽样），配对比较，新增失败数 − 新增成功数 ≤ 1，精确 McNemar p > 0.2（即没有可测的退化）。
3. **新种子非劣**：一段全新种子（默认 200 个），新旧提交配对，分拣成功率差的 95% 自助区间下限 ≥ −2 个百分点。

三关全过才合并。只过 1、2 两关时记为“过拟合到复现库”，回到修复步骤。

## 3. 停止条件（满足任一即停，然后冻结教师）

- 台账里没有任何一个簇的影响份额超过 5%；
- 连续两轮在新种子上的分拣成功率提升都小于 1 个百分点；
- 剩下的簇都被判为“需要新能力”（例如需要一个新工具，而不是修现有工具），转成单独的计划。

---

## 文件结构

| 文件 | 改动 | 职责 |
|---|---|---|
| `src/digital_twin_armigo/experiments/triage.py` | 新 | 分诊规则：`triage_episode(ep_dir) -> TriageRow` |
| `src/digital_twin_armigo/experiments/ledger.py` | 新 | 聚类和排序：`build_ledger(rows, reviews) -> Ledger` |
| `scripts/experiments/failure_ledger.py` | 新 | 命令行：扫描 → 分诊 → 台账 |
| `scripts/experiments/failure_bank.py` | 新 | 从台账簇生成复现库，并做两次复现检查 |
| `scripts/experiments/verify_fix.py` | 新 | 三道关：跑复现库、回退集和新种子，出 `verify-<簇>.md` |
| `src/digital_twin_armigo/experiments/contract.py` | 改 | `ROOT_CAUSES` 增补（见 F1） |
| `src/digital_twin_armigo/agent/step_mcp.py` | 改 | 工具返回置信度和失败原因（F5） |
| `tests/test_triage.py`、`tests/test_ledger.py`、`tests/test_failure_bank.py` | 新 | 测试 |

---

### Task F1：自动分诊规则

**Files:**
- Create: `src/digital_twin_armigo/experiments/triage.py`
- Modify: `src/digital_twin_armigo/experiments/contract.py:54-70`（`ROOT_CAUSES` 增加 `"perception_pose_error"`、`"calibration_stale"`、`"bowl_localization"`、`"wrong_bowl_carry"`、`"evidence_insufficient"`、`"tool_contract"`；保留原有项）
- Test: `tests/test_triage.py`

**Interfaces:**
- Produces:
  - `@dataclass TriageRow(traj_id: str, stage: str, code: str | None, root_cause: str, evidence: dict, rule: str)`；
  - `triage_episode(ep_dir: Path) -> TriageRow`；
  - `RULES: tuple[Rule, ...]`，按顺序匹配，第一条命中的生效，都不命中时 `root_cause="unclassified"`。

**规则表（按顺序，全部只读事件和结果，不重跑仿真）：**

| 编号 | 条件 | 根因 |
|---|---|---|
| R1 | `failure.code == "BOWL_NOT_FOUND"` | `bowl_localization` |
| R2 | 有 `view_truth` 事件，且感知 yaw 与真值 yaw 的差（按积木对称性取模）> 10° | `perception_pose_error` |
| R3 | 相机条件是 `drift`，且失败阶段为 `grasp` 并带 `no_clamp` | `calibration_stale` |
| R4 | 结局是 `PLACEMENT_UNVERIFIED`，砖进了另一只碗 | `wrong_bowl_carry` |
| R5 | 标签正确，但按 `view_truth` 有面没看清 | `evidence_insufficient` |
| R6 | 失败阶段为 `grasp` 或搬运，带 `no_clamp`、`BRICK_LOST` 或碰撞 | `grasp_or_path_failure` |
| R7 | 标签错误，且报警视图里该面清晰可见 | `false_positive_visible` / `false_negative_visible` |
| R8 | 工具返回异常、超时或不符合契约 | `tool_contract` |

- [ ] **Step 1：写失败的测试。** 每条规则各一个最小事件序列夹具，写成内存里的 `events` 列表加 `result` 字典，再写到临时目录，断言命中正确的规则编号和根因。外加：
  - 成功回合返回 `stage="ok"`；
  - 两条规则同时满足时取编号小的那条。

```python
def test_r2_yaw_error(tmp_path):
	ep = write_episode(tmp_path, events=[
		{"kind": "brick_perceived", "found": True, "yaw_deg": 30.0, "frame": 10},
		{"kind": "view_truth", "brick_yaw_deg": 5.0, "frame": 10},
		{"kind": "failure", "code": "SCAN_INCOMPLETE", "node": "scan_wrist", "frame": 400},
	], result={"verdict": "FAILED", "label": "normal"})
	row = triage_episode(ep)
	assert row.rule == "R2" and row.root_cause == "perception_pose_error"
	assert abs(row.evidence["yaw_err_deg"] - 25.0) < 1e-6
```

（`write_episode` 是测试文件里的辅助函数：写 `events.jsonl` 和 `episode.json`。事件字段名以 `twin_record` 实际写出的为准，先用 `grep -o '"kind": "[a-z_]*"' <某回合>/events.jsonl | sort | uniq -c` 查清楚，再写夹具。）

- [ ] **Step 2：运行，确认失败**
- [ ] **Step 3：实现规则。** 每条规则是一个纯函数 `(events, result) -> dict | None`，返回证据字典。
- [ ] **Step 4：用真实数据校准。** 在 `D:/Armigo_data_v2/runs/p2-dev-r2` 上跑一遍，把规则结果和上一轮人工分析的结论对照：drift 失败应命中 R3，evidence_insufficient 的几例应命中 R2 或 R5。把一致率写进测试注释。
- [ ] **Step 5：Commit**

### Task F2：失败台账

**Files:**
- Create: `src/digital_twin_armigo/experiments/ledger.py`、`scripts/experiments/failure_ledger.py`
- Test: `tests/test_ledger.py`

**Interfaces:**
- Consumes: `TriageRow`、`audit.latest(path, traj_id)`
- Produces:
  - `build_ledger(rows: list[TriageRow], reviews: dict[str, dict], factors: dict[str, dict]) -> dict`，返回值含 `clusters`，每个簇为 `{"key": [stage, code, root_cause], "n": int, "impact": float, "share": float, "factors": {因子: 分布}, "examples": [traj_id, …最多 10 个], "review_agreement": float | None}`，按 `impact` 降序；
  - 命令行：`failure_ledger.py --root E:/Armigo_data_v3 --batch <批次> --round NN`，写出 `ledger.md` / `ledger.json` / `triage.jsonl`。

- [ ] **Step 1：测试。** 用合成的分诊行和裁定：
  - 簇的计数、影响和排序正确；
  - 人工裁定的根因与规则不一致时，以人工为准，并在 `review_agreement` 里反映出来；
  - `factors` 能揭示偏斜，例如某个簇 80% 出在 `drift`。
- [ ] **Step 2–4：实现与验证。**
  - 场景因子从 `plan.json` 加案例选项读取，包括相机条件、碗颜色、积木型号和杂物数。
  - `ledger.md` 的每个簇附 viewer 链接：`http://127.0.0.1:8767/?traj=<id>`。
- [ ] **Step 5：Commit**

### Task F3：复现库

**Files:**
- Create: `scripts/experiments/failure_bank.py`
- Test: `tests/test_failure_bank.py`

**Interfaces:**
- Produces:
  - `failure_bank.py --ledger <ledger.json> --cluster <key> --round NN --max 30`；
  - 生成 `config/agent_bench/failure_bank/round-NN-<簇名>.json`，格式与 v2 套件相同，并带上 `source_round`、`cluster`、`commit`、`repro_checked` 字段；
  - 复现检查：每个案例在当前提交上新开进程跑两次（MuJoCo 要求状态轨迹逐位相同），不一致的剔除并记录。

- [ ] **Step 1：测试。**
  - 用一个已知确定的小案例（例如 `dev_suite_700001` 的第一个），跑两次结果一致，入库；
  - 用 monkeypatch 让第二次的结果不同，验证该案例被剔除；
  - 生成的套件能被 `experiments.runner run --suite` 直接读取（需要先把它登记进 manifest，复用 `make_suites.py` 的 manifest 写入函数）。
- [ ] **Step 2–4：实现与验证。** 只能按失败所在阶段从检查点恢复的簇（例如复现只需要放碗阶段），记录 `resume_from` 检查点编号，复现时用 `twin_resume.restore`，以缩短运行时间。
- [ ] **Step 5：Commit**

### Task F4：修复验证脚本（三道关）

**Files:**
- Create: `scripts/experiments/verify_fix.py`
- Test: `tests/test_failure_bank.py`（追加统计部分）

**Interfaces:**
- Produces: `verify_fix.py --bank <套件> --regression <套件> --fresh <套件> --base <旧提交的批次> --batch <新批次> --criteria <判据 JSON>`，写出 `verify-<簇>.md` 和 `verify-<簇>.json`。判定为 `merge`、`overfit_bank` 或 `reject`。

- [ ] **Step 1：测试统计部分。** 用合成的配对结果：
  - 复现库修复率；
  - McNemar 的 n10/n01 和 p 值（复用 `experiments/stats.py` 现有函数）；
  - 新种子差的聚类自助区间；
  - 判定逻辑覆盖三关的各种组合。
- [ ] **Step 2–4：实现与验证。** 运行部分直接调用 `experiments.runner run`（可续跑）；旧提交的结果从基线批次读取，不重跑。
- [ ] **Step 5：Commit**

### Task F5：工具返回值改进（MCP 工具面）

失败分析常常发现：工具其实“知道”自己不确定，但没告诉调用方。agent 因此没法换策略。这一步不改工具的行为，只把已有的内部量暴露成传感器层面的置信度和失败原因。

**Files:**
- Modify: `src/digital_twin_armigo/agent/step_mcp.py`（`locate_brick`、`locate_bowls`、`camera_calibration`、`scan_wrist`、`present_bottom`、`place`）
- Test: `tests/test_step_mcp_contract.py`（新建；已有同类测试就追加到那里）

**Interfaces:**
- Produces（新增字段，旧字段不变）：
  - `locate_brick`：`yaw_ambiguity_deg`（足迹拟合的残差换算出来的 yaw 不确定度）、`footprint_occluded_frac`；
  - `locate_bowls`：每只碗的 `rim_inliers`、`fit_residual_m`；
  - `camera_calibration`：`reprojection_px`；
  - `scan_wrist` / `present_bottom`：`faces_clear`、`faces_unclear`；
  - `place`：`carry_target_dist_m`（段末 TCP 到所选碗心的感知距离）。

- [ ] **Step 1：测试。** 每个新字段都通过 `assert_sensor_only`；旧客户端只读旧字段，结果不变；`yaw_ambiguity_deg` 在一个遮挡案例上明显大于一个干净案例（两个固定案例对比）。
- [ ] **Step 2–4：实现与验证**
- [ ] **Step 5：Commit**

### Task F6：第 1 轮（在 v3 试采数据上）

已知候选簇来自上一轮的实验记录；最终以台账为准：

| 候选簇 | 证据 | 修复方向（每个单独过三道关） |
|---|---|---|
| 顶视 yaw 错误 → 证据不足 | p2-dev-r2：每种取景策略都有 1–3 例，yaw 误差 20–25° | 用腕扫的第一张视图复核 yaw（已有图，不加动作）；或在 `yaw_ambiguity_deg` 大时多加一个视图 |
| 标定过期 → 抓取失败 | drift 下 A/B/C 都只有约 50% | 抓取失败后用腕相机重新定位积木再抓 |
| 多色碗定位 | v3 新增 | 色带和碗沿拟合（数据集 Task 3）；以台账为准 |
| 同色杂物干扰 | v3 新增 | 碗沿拟合的内点阈值；以台账为准 |

- [ ] **Step 1：** 跑 `failure_ledger.py` 生成第 1 轮台账；每个簇抽 5 例人工审计（viewer 8767），写进 `reviews.sqlite`。
- [ ] **Step 2：** 选影响最大的簇，在 `ledger.md` 里写判据：目标修复率、回退集和新种子的种子段。
- [ ] **Step 3：** 生成复现库（F3）。
- [ ] **Step 4：** 修复，并加上从复现案例提取的单元测试。例如保存失败帧的 RGB 和深度，断言修复后的 yaw 误差 < 5°。
- [ ] **Step 5：** 跑 `verify_fix.py`，按判定合并或返工。
- [ ] **Step 6：** 对下一个簇重复 Step 2–5，直到本轮选定的簇都处理完；然后开第 2 轮（新种子段）。

### Task F7：教师冻结

**Files:**
- Modify: `src/digital_twin_armigo/experiments/runner.py`（`freeze` 子命令已经存在；新增 `--purpose teacher-v3`，在 freeze 文件里额外记录 v3 场景生成器版本、调色板哈希和指令改写表哈希）

- [ ] **Step 1：** 满足第 3 节的停止条件后，提交代码，运行 `runner freeze --purpose teacher-v3`。
- [ ] **Step 2：** `collect_static_v3.py` 启动时校验 freeze 文件：提交号与 HEAD 一致、工作区干净、哈希一致，否则拒绝运行。
- [ ] **Step 3：** 测试：哈希不一致或工作区有改动时拒绝运行（用临时 git 仓库做夹具）。
- [ ] **Step 4：Commit**

---

## 节奏估计

- 一轮：采集 200 个回合约 2 小时；分诊和台账几分钟；人工抽查约 30 分钟；修一个簇 0.5–1 天；验证（复现库 30 + 回退集 100 + 新种子 200 个回合）约 3 小时。
- 预期 2–3 轮收敛，之后冻结、正式采集。
