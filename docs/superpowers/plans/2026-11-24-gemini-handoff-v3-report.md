# Gemini 交接执行报告：SmolVLA Static v3 (Tasks 1–9，跳过 3b) 与 Agent Failure Loop (F1–F5)

- **工作树**：`E:/digital_twin_armigo-v3`
- **分支**：`feat/smolvla-static-v3`（基于 `35d0d8c` / `7cc24e2` 之上，共新增 14 个任务提交 + 本报告提交；未合并、未推送）
- **隔离状态**：主仓库 `E:/digital_twin_armigo` 保持在 `main`（`0b28091`），无任何文件被写入或改动；未安装/升级任何包，未改动 `pyproject.toml`，未启动任何 GPU 任务；`D:/Armigo_data_v2` 仅只读使用；所有冒烟与预检产物写入 `C:/scratch/v3-dev/`（未向 `E:/Armigo_data_v3` 或 `C:/scratch/smolvla-probe` 写入）。

---

## 1. 任务清单总表

> 新增测试数按对应提交中新增 `def test_` 函数数量统计（不含参数化展开实例与复用既有套件的回归用例）。

| 任务 | 提交号 | 新增 / 修改文件 | 新增测试数 (`def test_`) |
|---|---|---|---|
| **v3 Task 1** 调色板与确定性 `SceneV3` | `b69c2e0` | 新增 `src/digital_twin_armigo/mujoco_cell/palette.py`, `src/digital_twin_armigo/mujoco_cell/scene_v3.py`, `tests/test_palette_scene_v3.py` | 10（14 条参数化用例） |
| **v3 Task 2** Layout/MJCF 碗色、桌面材质与多碗 | `68f7a1d` | 新增 `tests/data/legacy_7cc24e2.json`；修改 `src/digital_twin_armigo/mujoco_cell/layout.py`, `scene.py`, `scene_v3.py`, `cell.py`, `src/digital_twin_armigo/agent/twin_mujoco.py`, `src/digital_twin_armigo/isaac_bridge/client.py`, `tests/test_palette_scene_v3.py` | 3 |
| **v3 Task 3** 8 色 HSV 带与碗感知 | `eb57a84` | 新增 `tests/test_perception_palette.py`；修改 `src/digital_twin_armigo/agent/perception.py`, `src/digital_twin_armigo/agent/twin_mujoco.py`, `tests/test_palette_scene_v3.py` | 4（含 24 色×种子参数化 + 1 条 strict xfail） |
| **v3 Task 3b** Isaac 新颜色支持 | *跳过* | 无（按交接要求跳过，保留 `NotImplementedError` 桩） | 0 |
| **v3 Task 4** 宽相机采样与工作区可见性 | `76475a5` | 新增 `src/digital_twin_armigo/mujoco_cell/visibility.py`；修改 `src/digital_twin_armigo/agent/twin_mujoco.py`, `tests/test_palette_scene_v3.py` | 2 |
| **v3 Task 5** 每类 6 种指令改写与 v3 碗状态置零 | `f67000b` | 新增 `tests/test_instruction_variants.py`；修改 `src/digital_twin_armigo/vla/contract.py`, `src/digital_twin_armigo/vla/dataset.py`, `src/digital_twin_armigo/agent/student_rollout.py`, `src/digital_twin_armigo/agent/twin_record.py` | 6（含 1 条需 `transformers` 的 skip 与 1 条 strict xfail） |
| **v3 Task 6** 多碗路由、`run_branches` 与 K 碗探针 | `32021e2` | 新增 `tests/conftest.py`；修改 `src/digital_twin_armigo/agent/twin_mujoco.py`, `src/digital_twin_armigo/agent/student_rollout.py`, `src/digital_twin_armigo/vla/paired_carry.py`, `scripts/vla/bowl_swap_probe.py`, `tests/test_vla_paired_carry.py` | 6（含 1 条 `@pytest.mark.slow` 3 碗端到端冒烟） |
| **v3 Task 7** OU 动作噪声正则化与 `mask.noisy` | `c61b97e` | 新增 `tests/data/legacy_actions_600001.npz`, `tests/test_action_noise.py`；修改 `src/digital_twin_armigo/agent/student_rollout.py`, `src/digital_twin_armigo/agent/twin_mujoco.py`, `src/digital_twin_armigo/agent/twin_record.py` | 8 |
| **v3 Task 8** v3 场景套件生成器与预检报告 | `18fcd06` | 新增 `scripts/vla/make_scene_suite_v3.py`, `tests/test_scene_suite_v3.py` | 7 |
| **v3 Task 9** `--student-cameras-only` LeRobot 导出 | `8cf48e0` | 新增 `tests/test_export_cameras.py`；修改 `src/digital_twin_armigo/agent/twin_record.py`, `scripts/vla/collect_paired_carry.py` | 4（含 1 条需 `lerobot` 的 `.venv` skip，另在 CPU `lerobot-cuda` 上完整跑通导出与回读） |
| **loop F1** 自动分诊规则 R1–R8 | `bc7375c` | 新增 `src/digital_twin_armigo/experiments/triage.py`, `tests/test_triage.py`；修改 `src/digital_twin_armigo/experiments/contract.py` | 14 |
| **loop F2** 聚类失败台账与 CLI | `20538ce` | 新增 `src/digital_twin_armigo/experiments/ledger.py`, `scripts/experiments/failure_ledger.py`, `tests/test_ledger.py` | 8 |
| **loop F3** 双进程逐位复现库与 manifest 集成 | `027c50a` | 新增 `scripts/experiments/failure_bank.py`, `tests/test_failure_bank.py`；修改 `scripts/experiments/make_suites.py`, `src/digital_twin_armigo/experiments/runner.py` | 5 |
| **loop F4** 三道关修复验证脚本与统计 | `f5329c1` | 新增 `scripts/experiments/verify_fix.py`；修改 `tests/test_failure_bank.py` | 6 |
| **loop F5** MCP 工具传感器不确定度与线索 | `f2c93d0` | 新增 `tests/test_step_mcp_contract.py`；修改 `src/digital_twin_armigo/agent/perception.py`, `src/digital_twin_armigo/agent/step_mcp.py`, `src/digital_twin_armigo/agent/student_rollout.py`, `src/digital_twin_armigo/agent/twin_mujoco.py` | 4 |

合计新增测试函数：**87** 条 `def test_`；全部代码修改共涉及 **38** 个 `.py` 文件。

---

## 2. 全量测试结果、跳过/XFAIL 清单与代码风格检查

### 2.1 主 `.venv` 全量测试输出

在完成全部代码修改（对应 F5 `f2c93d0`）后执行命令：

```powershell
Set-Location E:/digital_twin_armigo-v3
$env:PYTHONPATH = "src"
E:/digital_twin_armigo/.venv/Scripts/python.exe -m pytest -q -rsx
```

最终汇总输出：

- `963 passed, 14 skipped, 2 xfailed, 16 warnings in 1438.44s (0:23:58)`（0 failed）

逐条跳过的测试与原因（直接引自 `short test summary info`）：

| 状态 | 数量 | 测试文件与行号 | 原因 |
|---|---|---|---|
| `SKIPPED` | 1 | `tests/test_vla_augment_batch.py:6` | `could not import 'torch': No module named 'torch'`（主 `.venv` 不含 `torch`） |
| `SKIPPED` | 1 | `tests/test_vla_bowl_token.py:23` | `needs torch and lerobot (run under E:/il/lerobot-cuda)` |
| `SKIPPED` | 1 | `tests/test_agent_policies.py:175` | `local vLLM server not reachable` |
| `SKIPPED` | 2 | `tests/test_experiments_scoring.py:350` | `real episode missing`（仓库 `results/agent-bench/...` 真实回合未随 worktree 检出） |
| `SKIPPED` | 1 | `tests/test_export_cameras.py:155` | `could not import 'lerobot': No module named 'lerobot'`（已在 CPU `lerobot-cuda` 实测覆盖） |
| `SKIPPED` | 1 | `tests/test_instruction_variants.py:46` | `could not import 'transformers': No module named 'transformers'`（已在 CPU `lerobot-cuda` 实测覆盖） |
| `SKIPPED` | 1 | `tests/test_tracking.py:90` | `could not import 'mlflow': No module named 'mlflow'` |
| `SKIPPED` | 1 | `tests/test_vla_s5_perturb.py:235` | `no stitched episode for an unrecovered branch` |
| `SKIPPED` | 5 | `tests/test_vla_s9_losses.py:545, 666, 736, 759, 856` | `could not import 'torch': No module named 'torch'` |
| `XFAIL` (strict) | 1 | `tests/test_instruction_variants.py::test_hue_augmentation_cannot_cross_palette_gap` | Plan mismatch: spec §3 keeps `hue=0.05` (18.0 deg) while Task 5 asserts `< 35.0 / 2` (17.5 deg)（见第 3 节第 4 条） |
| `XFAIL` (strict) | 1 | `tests/test_perception_palette.py::test_white_bowl_low_light_known_spec_limit` | Task 3 spec fixes white `vmin=200`; under `scene_xml` `light_level < 0.72` white bowl renders at `V~175 <= 200`（见第 3 节第 3 条） |

> 历史全量回归注记：在 **Task 9**（`8cf48e0`）提交前的全量 `pytest -q` 中，除该次自身无失败外，曾出现 1 次既有并发竞态失败：`tests/test_viewer_app.py::test_logger_follows_a_growing_episode_until_it_finishes` 在读取写入中的 `episode.json` 时抛出 `JSONDecodeError`（与 Task 9 无关）；立即单独执行 `E:/digital_twin_armigo/.venv/Scripts/python.exe -m pytest tests/test_viewer_app.py -v` 输出 `11 passed, 13 warnings in 1.34s`，且此后 F1 (`bc7375c`)、F2 (`20538ce`)、F3 (`027c50a`)、F4 (`f5329c1`)、F5 (`f2c93d0`) 五次全量 `pytest -q` 均 0 失败通过。

### 2.2 历史兼容与防泄漏检查（F5 HEAD `f2c93d0`）

1. `tests/test_palette_scene_v3.py`（含 `test_old_cases_unchanged_by_v3_fields` 对 `tests/data/legacy_7cc24e2.json` 的 14 项 `Layout` 与 MJCF XML SHA-256 校验）：
   ```text
   E:/digital_twin_armigo/.venv/Scripts/python.exe -m pytest tests/test_palette_scene_v3.py -v
   -> 19 passed, 13 warnings in 5.75s
   ```
2. `tests/test_action_noise.py -k legacy`（噪声关闭时 `(600001, False, 5)` 的 `actions.npz` 与 `tests/data/legacy_actions_600001.npz` 逐键、逐 dtype、逐数组相等）：
   ```text
   E:/digital_twin_armigo/.venv/Scripts/python.exe -m pytest tests/test_action_noise.py -k legacy -v
   -> 1 passed, 7 deselected, 13 warnings in 13.70s
   ```
3. `tests/test_twin_leakage.py` 在全量 `pytest -q -rsx` 中全部通过（0 failed）。

### 2.3 代码风格检查（全部改动 Python 文件）

对 `35d0d8c..f2c93d0` 间全部 38 个新增/修改的 `.py` 文件执行：

```powershell
Set-Location E:/digital_twin_armigo-v3
$changed = git diff --name-only 35d0d8c..HEAD | Where-Object { $_ -like '*.py' }
E:/digital_twin_armigo/.venv/Scripts/ruff.exe check @changed
E:/digital_twin_armigo/.venv/Scripts/ruff.exe format --check @changed
```

输出：`All checks passed!` 与 `38 files already formatted`。

### 2.4 提交签名与工作区隔离校验

- `35d0d8c..f2c93d0` 全部 14 个任务提交均带有 `Co-Authored-By: Gemini <noreply@google.com>`，不含 `Claude` 署名（`COMMIT_TRAILER_CHECK: 14/14 clean`）。
- 主仓库核查：`git -C E:/digital_twin_armigo branch --show-current` -> `main`；`git -C E:/digital_twin_armigo log -n 1 --oneline` -> `0b28091 Merge viewer/inventory branch: Rerun replay, inventory, failed-episode reruns, final-cycle verdict`；`git -C E:/digital_twin_armigo status -sb` -> 仅原有未跟踪草稿，无任何变更。

---

## 3. 与原计划不一致之处

1. **Task 2：`extra_bowls` 放置范围放宽**
   - **计划原文**：从 `r in [0.21, 0.27]`、`a in [-1.1, 1.1]` 采样第 3 只碗。
   - **实际做法**：在 `src/digital_twin_armigo/mujoco_cell/scene_v3.py` 的 `apply_scene_v3` 中，前 200 次尝试使用窄范围，若间距校验不通过则依次放宽到 `r in [0.21, 0.41], a in [-1.25, 1.25]` 与 `r in [0.21, 0.42], a in [-1.45, 1.45]`。
   - **原因**：3 碗时窄扇区内满足圆心距与工作空间约束的比例不足（约 78% 失败）；放宽后在 2000 粒种子上全部成功放置，且 500 粒种子下在 `FIXED_TOP` 视野内超 5% 边距越界数为 0/145。
2. **Task 2：`TwinMujoco.__init__` 对 `layout.clutter` 的校验条件**
   - **计划原文**：v3 场景允许 `extra_clutter`（1–3 个碗内杂物），同时不破坏历史行为。
   - **实际做法**：`src/digital_twin_armigo/agent/twin_mujoco.py` 将原检查调整为仅当 `layout.clutter and layout.bowl_names is None` 时抛出 `ValueError`。
   - **原因**：既让 v3 碗内杂物正常运行，又保证既有回归测试 `tests/test_twin_leakage.py::test_realistic_refuses_clutter` 继续按原契约抛出 `ValueError`。
3. **Task 3：白碗高亮度阈值 `vmin = 200` 与低光照下的识别边界**
   - **计划原文**：白色带条件设为 `V > 200, S <= 45`，同时场景采样允许 `light_level in [0.60, 1.00]`；Task 3 Step 4 明确要求若白碗失效不放宽阈值。
   - **实际做法**：保留 `vmin = 200` 未改；将已知低光照触发例（如 `seed=3000015`, `light_level=0.650`）在 `tests/test_perception_palette.py::test_white_bowl_low_light_known_spec_limit` 中以 `@pytest.mark.xfail(strict=True)` 固定记录。
   - **原因**：遵守“不放宽白碗阈值、暴露真实物理边界”的交接要求；在 `light_level < ~0.81` 且白色抖动偏暗（如 `0.904`）时，桌面渲染灰白碗像素亮度 `V` 低于 200 属于光照与固定绝对阈值的物理冲突。
4. **Task 5：颜色抖动 `hue` 算术冲突与 `encode_state(..., zero_bowl=False)`**
   - **计划原文**：§3 保持 `StudentImageAugmentor.hue = 0.05`（±18.0°），而 Task 5 测试步骤要求断言 `hue * 360.0 < 35.0 / 2`（`< 17.5°`）；同时要求 v3 数据集中 `state[31:34]` 碗位标志全零。
   - **实际做法**：保持 `StudentImageAugmentor.hue = 0.05` 规格未改，将原断言标为 `@pytest.mark.xfail(strict=True)`，并新增 `test_hue_augmentation_under_half_realized_palette_gap`（实测 `PALETTE` 中 `orange` 与 `yellow` 最小彩色色相间隔为 `37.07°`，半宽 `18.535° > 18.0°`，通过）；同时 `encode_state(..., zero_bowl=False)` 默认维持旧契约（`bowl=None` 时 `s[31]=1.0`，兼容 `tests/test_vla_s9_losses.py`），由 `src/digital_twin_armigo/vla/dataset.py` 在 `ep.scene == "v3"` 时显式传入 `zero_bowl=True`。
   - **原因**：`0.05 * 360 = 18.0 < 17.5` 在数学上为假；不擅自修改规格常数与既有回归契约。
5. **Task 6：`@pytest.mark.slow` 注册位置与 Windows 分支目录名**
   - **计划原文**：为慢速端到端多碗冒烟注册 `slow` marker；分支数据落盘到分支目录。
   - **实际做法**：遵守不修改 `pyproject.toml` 约束，在新建 `tests/conftest.py` 中注册 `slow` marker；`vla/paired_carry.py::run_branches` 将 `extra:0` 的子目录名转为 `extra_0`，同时返回结果与元数据中的 `"target"` 仍保留 `"extra:0"`。
   - **原因**：Windows NTFS 不允许文件名含 `:` 字符，且禁止改动 `pyproject.toml`。
6. **Task 7：`ActionNoise` 默认 `sigma_rad = 0.02` 在 `(600001, False, 5)` 上的表现**
   - **计划原文**：`ActionNoise` 默认 `sigma_rad = 0.02, tau_s = 0.3`，覆盖 `grasp, lift, carry_to_bowl, release_in_bowl, scan_view`；Task 7 测试未固定具体 `sigma_rad` 数值，仅要求偏差标准差落在 `0.5–1.5 * sigma`。
   - **实际做法**：保留 `ActionNoise.sigma_rad = 0.02` 类默认值不变；`tests/test_action_noise.py` 的端到端测试采用 `sigma_rad = 0.01`（位于文档 `0.01–0.03` 建议区间内）。
   - **原因**：在 `(600001, False, 5)` 上实测 `sigma_rad = 0.02` 时，6/6 种子（`truth`）与 2/2 种子（`realistic`）因 `grasp`/持握段 5 自由度关节扰动发生 `no_clamp` / `BRICK_LOST`；而 `sigma_rad = 0.01` 时 3/3 `truth` 与 2/2 `realistic` 全部 `SUCCESS`。
7. **Task 8 & Loop F1–F4：只读数据保护与 `.venv` 无 `transformers`/`lerobot` 的适配**
   - **计划原文**：`failure_ledger.py --root ...` 写出报告到数据集目录下；`make_scene_suite_v3.py` 计算 `token_max`。
   - **实际做法**：为 `failure_ledger.py`、`failure_bank.py` 与 `verify_fix.py` 增加只读隔离参数（`--out-dir`、`--audit-db`、`--episodes-root`、`--manifest`、`--scratch-dir`、`--base-root`、`--skip-run`），使对 `D:/Armigo_data_v2` 的端到端验证全部落盘至 `C:/scratch/v3-dev/loop-v2-check/` 且不改写仓库 `config/agent_bench/v2/manifest.json`；`make_scene_suite_v3.py` 计算 `token_max` 时自动探测并调用 CPU 子进程 `E:/il/lerobot-cuda/Scripts/python.exe` 执行真实 SmolVLM2 分词（若无 `transformers` 则降级为子词估算并标注 `backend`），并在 `episodes < 400` 的小规模预检套件中记录左右相关性指标而不触发明确针对 `400+` 大套件的 `0.05` 硬阈值阻断。

---

## 4. 未做完或跳过的步骤

- **v3 Task 3b（Isaac 新增颜色支持）**：按交接指令跳过，仅保留 `NotImplementedError` 桩。
- **v3 Task 10–11**：200 回合试采验收与 SmolVLA 静态 v3 训练评估（不在本次交接范围）。
- **Loop F6–F7**：在 v3 试采数据上的第 1–3 轮失败闭环迭代（不在本次交接范围）。
- **大规模采集与 GPU 任务**：未生成全量 `train/val/ood/test` 套件到 `E:/Armigo_data_v3`，未启动任何 GPU 训练/推理或 Isaac 服务。

---

## 5. 需要用户或 Claude 决定的问题

1. **Task 7 `ActionNoise` 参数取舍**：在 Task 10 200 回合试采前，需决定是否将默认 `sigma_rad` 从 `0.02` 调整为 `0.01`，或从 `ActionNoise.kinds` 中移除 `grasp` 段（仅在 `lift, carry_to_bowl, release_in_bowl, scan_view` 注入扰动），以免抓取阶段成功率大幅下降。
2. **Task 5 `hue` 阈值裁决**：§3 与 Task 5 测试的算术不一致（`hue=0.05 -> 18.0°` vs `< 17.5°`）：需决定是将 `StudentImageAugmentor.hue` 调小为 `<= 0.045`（`< 16.2°`），还是接受调色板实测最小彩色间隔 `37.07°`（半宽 `18.535° > 18.0°`）并移除旧 `35/2` 断言。
3. **Task 3 白碗低光照阈值策略**：当 `light_level < ~0.81` 且白色抖动偏暗时 `V > 200` 无法命中白碗，需决定是否在 `perception.py` 中改用相对桌面亮度/色彩对比，还是维持 `vmin=200` 由 R1 (`BOWL_NOT_FOUND`) 在低光极限样本上暴露为预期失败。
4. **Loop F1/F2 `p2-dev-r2` 中的 `torque_stall` 簇（14 例）**：在 `D:/Armigo_data_v2/runs/p2-dev-r2` 校准中，有 14 个回合砖已物理入对碗但 `health.failure_mode == "torque_stall"` 导致 `verdict == "FAILED"`，目前全部落入 `["place", "torque_stall", "unclassified"]`；同时 `D:/Armigo_data_v2/audit/reviews.sqlite` 中无 `p2-dev-r2` 人工审核记录（仅含 1 条 `m0-smoke`）。需决定是否为 `torque_stall` 补充单独规则 R9，或在 `p2-dev-r2` 后补录审核记录验证 `review_agreement`。
5. **既有 `test_viewer_app.py` 并发读写竞争**：`test_logger_follows_a_growing_episode_until_it_finishes` 在高负载下读到写入中的半截 `episode.json` 触发 `JSONDecodeError`（单独重跑 1.34s 通过），建议后续将 `episode.json` 写入改为临时文件原子替换（`replace`）。

---

## 6. 冒烟、预检与只读验证产物及实测输出

所有冒烟与验证产物均位于 `C:/scratch/v3-dev/`：

1. **基线快照与黄金回归对照**
   - `C:/scratch/v3-dev/golden_7cc24e2.json`（已同步固化至 `tests/data/legacy_7cc24e2.json`）：Task 2 前采集的 14 个历史案例 `Layout`、`camera` 与 MJCF SHA-256。
   - `C:/scratch/v3-dev/legacy_actions_600001.npz`（已固化至 `tests/data/legacy_actions_600001.npz`）：Task 7 前采集的 `(600001, False, 5)` `sensing="truth", video=False` 动作轨迹。
2. **Task 5 SmolVLM2 CPU 分词长度实测**（`E:/il/lerobot-cuda/Scripts/python.exe`，`CUDA_VISIBLE_DEVICES=""`，`PYTHONPATH="src"`）：
   - 覆盖 12 种 segment × 6 种变体共 **72** 条指令：输出 `count 72 max (34, 'release_in_bowl', 0)`（最大 34 token，<= 40）。
3. **Task 8 小规模套件预检冒烟**（`C:/scratch/v3-dev/preflight-train50/`）
   - 命令：`PYTHONPATH=src E:/digital_twin_armigo/.venv/Scripts/python.exe scripts/vla/make_scene_suite_v3.py --split train --seeds 3000001-3000050 --out C:/scratch/v3-dev/preflight-train50/train.json --preflight-dir C:/scratch/v3-dev/preflight-train50/ --perception-sample 0.1 --max-perception 5`
   - 输出：`v3 suite train: 50 cases, rejections={'reachability': 0, 'visibility': 0, 'colour_separability': 0}, pass=True`；`preflight.json` 中 `held_out_pair_count: 0`、`visibility_resamples.total: 0`、`bowl_localization_rate: 0.8462`（11/13）、`bowl_median_error_mm: 0.934`、`brick_median_error_mm: 0.127`、`token_max: 32`（`backend: "lerobot-cuda"`）、`pass: true`。
4. **Task 9 纯学生相机 LeRobot 导出与回读验证**（`C:/scratch/v3-dev/export-cam-check/`）
   - 在 CPU `E:/il/lerobot-cuda/Scripts/python.exe`（`CUDA_VISIBLE_DEVICES=""`）对 10 帧合成回合分别以 `student_cameras_only=True` 和 `student_cameras_only=False` 调用 `export_lerobot`，并用 `SegmentChunkDataset(..., repo_id='local/test_student', horizon=4, video_backend='pyav')` 回读：
   - 输出：`student_imgs: ['observation.images.top_depth_v2', 'observation.images.top_rgb_v2', 'observation.images.wrist'] legacy_imgs: ['observation.images.front', 'observation.images.top_depth_v2', 'observation.images.top_full', 'observation.images.top_rgb_v2', 'observation.images.wrist'] ds_len: 10`。
5. **Loop F1–F4 在 `D:/Armigo_data_v2/runs/p2-dev-r2` 上的只读端到端检查**（`C:/scratch/v3-dev/loop-v2-check/`）
   - **F1/F2 分诊与台账**（`round-01/{triage.jsonl,ledger.json,ledger.md}`）：共 **216** 回合（**152** clean `ok`、**64** failures，共 **9** 个簇）。在前 50 条 `scores.jsonl` 归因非完美回合中，R1–R8 命中 **49/50（98.0%）**：15/15 `evidence_insufficient` 全部命中 R2（12）或 R5（3），17/17 纯 `drift` 抓取无误朝向案例命中 R3（其余 10 例伴随 `> 10°` 偏航误差先命中更小编号 R2），3/3 `BRICK_LOST` 命中 R6，2/2 `false_positive_visible` 命中 R7。台账前 3 簇：`['grasp', 'no_clamp', 'calibration_stale']` (`n=17, impact=17.0`)、`['perception', None, 'perception_pose_error']` (`n=14, impact=14.0`)、`['place', 'torque_stall', 'unclassified']` (`n=14, impact=14.0`)。
   - **F3 双进程逐位复现建库**（`round-01-bank.json`, `manifest.json`, `repro-tmp/`）：选取簇 `perception:null:perception_pose_error`（`--max 1`，`--episodes-root D:/Armigo_data_v2`）在两独立子进程跑真实案例 `[700007, True, 0, {'cameras': 'drift'}]`，输出 `kept=1, dropped=0`。
   - **F4 三道关自比冒烟**（`verify-perception-null-perception_pose_error.json` / `.md`）：在 `p2-dev-r2` `realistic-drift-B-v7bot` 与 `p2_dev12` 上 `--skip-run` 验证统计链路：Gate 1 (`0/1 = 0.0%, FAIL`)、Gate 2 (`n10=0, n01=0, p=1.0, PASS`)、Gate 3 (`24 pairs / 12 clusters, diff=0.0, ci=[0.0, 0.0], PASS`)，总判定 `reject`。
6. **Loop F5 MCP 工具新增传感器字段实测**（`C:/scratch/v3-dev/f5-check/`，F5 提交 `f2c93d0` 后实测）：
   - 干净案例 `(600001, False, 5)` `locate_brick`：`yaw_ambiguity_deg = 4.341`, `footprint_occluded_frac = 0.0`, `footprint_mm = [32.9, 32.38]`。
   - 遮挡案例 `(600004, False, 5)` `locate_brick`：`yaw_ambiguity_deg = 32.38`, `footprint_occluded_frac = 0.3594`, `footprint_mm = [33.54, 23.0]`, `occlusion_correction_mm = [0.0, 8.8]`。
   - 全部 MCP 真实剖面返回字段均通过 `assert_sensor_only`（无任何 `SIMULATOR_KEYS` 泄漏），且所有既有一致性字段维持不变。
7. **辅助草稿文件**：`C:/scratch/v3-dev/stage-f4/`、`C:/scratch/v3-dev/stage-f5/`、`C:/scratch/v3-dev/report-draft.md`（并行实作与起草时的临时目录，正式交接文件以仓库 git 提交为准）。
