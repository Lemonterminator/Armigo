# 配对轨迹数据治理（Paired Trajectory Governance）

本仓库通过 `bronze -> silver -> gold` 三层 SQLite 数据湖（代码位于 `src/digital_twin_armigo/governance/`，CLI 为 `scripts/governance/lake.py`）管理 twin 采集到的 truth / realistic 配对轨迹。

## 1. 三层结构

| 层 | 内容 | 存放 | 可变性 |
|---|---|---|---|
| `bronze` | 每个回合目录一条记录：路径、各核心文件 SHA256（`episode.json` / `events.jsonl` / `run_status.json` / `task.json` / `replay/motion.npz`）、登记时间、git commit、来源批次 | `lake.sqlite` 表 `bronze` | **只追加**；重扫发现哈希变化时保留原行，写入 `integrity_event` |
| `silver` | `silver_episode`（一回合一行：评分、`root_cause`、`mechanism`、种子段、审计结论、5 项校验与隔离标记）；`silver_pair`（按配对键和相机条件聚合） | `lake.sqlite` 表 `silver_episode`、`silver_pair` | 可从 `bronze` + 当前代码完全重建；每次构建记录可复现的 12 位 `build_id` |
| `gold` | 数据产品 manifest 与数据卡 | `gold/<product>/<version>/manifest.jsonl` 与 `card.md` | 版本号为 `manifest.jsonl` 内容的 SHA256 前 12 位；生成后不可原地修改 |

## 2. 配对键与配对结果矩阵

- **配对键**（`governance/pairs.py:pair_key`）：`pair_key = (engine, case_id, detector, strategy, motion, evidence)`。
  - `truth` 一侧固定使用 `fixed` 相机，其 `group_name` 不含相机字段；
  - `realistic` 一侧每种相机条件（`fixed` / `recal` / `drift`）各自与对应的 `truth` 回合配对。

| `realistic` \ `truth` | 成功 | 失败 |
|---|---|---|
| 成功 | `both_pass` | `realistic_only`（**异常**：必须隔离复查，禁止进可训练产品） |
| 失败 | `truth_only`（感知/观测缺口） | `both_fail`（规划、物理或场景本身问题） |

另有两类特殊状态：

- `void`：任一侧 `infra_failed`，或任一侧被隔离（不算结果，应重跑）。
- `unpaired`：仅有 `truth` 或仅有 `realistic` 一侧。

## 3. Gold 产品与谓词

所有产品均在 `src/digital_twin_armigo/governance/gold.py` 中定义，并共用前置过滤 `_base(r)`（未隔离、`oracle` 为假、`superseded` 为假——仅最新一次尝试）：

| 产品 | 传感侧 | 谓词（中文复述） | 代码位置 |
|---|---|---|---|
| `student_train_v3` | `realistic` | 可训练产物。要求处于 `train` 种子段、`physical_success` 为真、教师提交号与 `freeze` 一致（`teacher_frozen`）；且要么配对结果为 `both_pass`，要么为 `unpaired` 但其所在批次的 `realistic_only` 率低于 `1%`。 | `governance/gold.py:PRODUCTS["student_train_v3"]` |
| `privileged_demos_v3` | `truth` | 可训练产物。要求处于 `train` 种子段且配对结果为 `truth_only`（记录真实侧卡住的难场景特权示范）。 | `governance/gold.py:PRODUCTS["privileged_demos_v3"]` |
| `failure_bank` | `realistic` | 不可训练产物。要求不在封存 `test` 段，且配对结果为 `truth_only` 或 `both_fail`（供 F2 台账、F3 复现库、H7 诊断基准使用）。 | `governance/gold.py:PRODUCTS["failure_bank"]` |
| `harness_review` | 任意 | 不可训练产物（人工复查队列）。命中任一条件即可：被隔离、`pair_outcome == "realistic_only"`、或同一 `pair_key` 连续至少 `2` 次尝试为 `void`。 | `governance/gold.py:PRODUCTS["harness_review"]` |
| `eval_<suite>` | `realistic` | 不可训练产物。要求属于对应评测种子段（如 `val` / `ood-appearance` / `ood-pose` / `test`）；当段为 `test` 时必须同时提供 `freeze` 文件。 | `governance/gold.py:PRODUCTS["eval_<suite>"]` |

## 4. 隔离与守护硬约束

1. **逐回合强制隔离项**（`governance/silver.py:QUARANTINE_CHECKS`）：
   - `complete`：`run_state == "done"` 且 `episode.json` 存在；
   - `bronze_intact`：该回合不存在 `integrity_event` 篡改记录；
   - `sensor_guard`：对 `realistic` 回合检查工具返回与传感器载荷不得包含 `step_mcp.SIMULATOR_KEYS` / `TRUTH_DIAGNOSTIC_KEYS`；
   - `no_truth_fallback`：`truth_fallback` 标记必须为假。
2. **Gold 构建不变量**（`governance/gold.py:assert_product_invariants`）：
   - 对可训练产品（`trainable=True`），强制拒绝：非 `train` 段、任何已隔离回合、任何 `oracle` 回合、任何 `realistic_only` 回合，并调用 `vla.dagger.assert_train_layouts_only` 二次核验；
   - `failure_bank` 显式禁止包含封存 `test` 段回合；
   - 构建 `version` 仅依赖排序后的 manifest 内容哈希，相同输入构建两次产物逐字节一致。
3. **导出校验**（`scripts/vla/collect_paired_carry.py --manifest`）：
   - 从 manifest 导出前逐条重新计算 `episode.json` 的 `sha256` 并与 `episode_sha256` 核对；任一条不匹配立即报错并中止，不写任何输出文件。

## 5. 标准操作顺序

1. **采集**：用 `python -m digital_twin_armigo.experiments.runner run-pair ...`（同 `case_id` 的 truth 与 realistic 交错执行，支持随时恢复后大部分仍成对）。
2. **bronze 登记**：`PYTHONPATH=src python scripts/governance/lake.py register --root <runs_root> --source <name>`。
3. **构建 silver**：`PYTHONPATH=src python scripts/governance/lake.py silver --freeze <freeze.json>`。
4. **生成 gold 产品与审阅报告**：
   - `PYTHONPATH=src python scripts/governance/lake.py report`
   - `PYTHONPATH=src python scripts/governance/lake.py gold student_train_v3 --freeze <freeze.json>`
   - `PYTHONPATH=src python scripts/governance/lake.py gold failure_bank`
   - `PYTHONPATH=src python scripts/governance/lake.py gold harness_review`
5. **导出 LeRobot 分片**：`PYTHONPATH=src python scripts/vla/collect_paired_carry.py --manifest <gold/student_train_v3/<ver>/manifest.jsonl> --out <export_dir> --export`。
6. **VLA 训练**（需用户确认排期后执行）。

## 6. 待定事项（引自治理计划 第 2.4 节）

- 已定：试采 `200` 个回合全配对；此后按 `25%` 抽样配对（若某批次 `realistic_only >= 1%`，该批次未配对回合暂不进 `student_train_v3`）。
- 已定：当前优先产线为 `failure_bank`、`harness_review` 与 `eval_*`；`student_train_v3` 与 `privileged_demos_v3` 按谓词正常产成并保留。
- **待定 1**：发生过恢复（`recovered=True`）的回合在 VLA 训练中的采样/损失权重。
- **待定 2**：`privileged_demos_v3`（`truth_only` 特权示范）在重启 VLA 训练时是否以及按何比例混入训练集（需消融决定）。
