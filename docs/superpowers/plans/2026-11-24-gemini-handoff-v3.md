# 交接：v3 静态数据集与失败闭环的代码实现（给 Gemini）

写于 2026-11-24。实现完成后由 Claude 验收。

## 1. 位置

- **分支：`feat/smolvla-static-v3`**（从 `7cc24e2` 切出）。
- **工作目录：`E:/digital_twin_armigo-v3`**（git worktree）。所有修改、测试和提交都在这里做。
- **不要碰 `E:/digital_twin_armigo`。** 那里在跑 SmolVLA 训练和探针，分支是 `feat/dual-engine-audit`。不要在那里切分支、改文件、提交或删除任何东西。
- 不要推送，不要合并，不要改 `main`。

## 2. 要读的文件（按顺序）

1. `docs/superpowers/plans/2026-11-24-smolvla-static-dataset.md`：数据集计划，第 0–7 节是规格，"已定事项"一节是用户已做的决定。
2. `docs/superpowers/plans/2026-11-24-agent-failure-loop.md`：失败闭环计划。
3. `docs/vla/BOWL_CHANNEL_PAIRED_PLAN_2026-11-23.md`：配对搬运（`TargetSwapTracer`）的现有实现，Task 6 在它上面扩展。
4. `docs/EXPERIMENTS_V2.md`：v2 runner、inventory、audit，F1–F4 用到。

两份计划的 Global Constraints 适用于每个任务。和本文冲突时以本文为准。

## 3. 范围

按这个顺序做，一个任务一个提交：

1. 数据集计划 **Task 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9**
2. 失败闭环计划 **Task F1 → F2 → F3 → F4 → F5**

**不做：**

- **Task 3b**（Isaac 色带）。它要在 Isaac 上渲染，需要 GPU 和 Isaac 服务器，由 Claude 做。Task 3 的代码要给 `bands_for(..., engine="isaac")` 留好入口：Isaac 先沿用现有 `BOWL_HSV_BY_ENGINE` 的行为，新颜色在 Isaac 下抛出明确的 `NotImplementedError`。
- **Task 10、11、F6、F7。** 这些是试采、正式采集、训练、闭环第 1 轮和教师冻结，要和 GPU 排期，由用户和 Claude 决定何时运行。
- 计划之外的重构、改名和"顺手"清理。

做到某个任务发现计划本身有错或缺信息时：不要自己改规格，在报告里写明，跳过这一步，继续做不依赖它的任务。

## 4. 环境

- worktree 里没有自己的 `.venv`，用主仓库的解释器：
  - 测试：在 `E:/digital_twin_armigo-v3` 下运行 `E:/digital_twin_armigo/.venv/Scripts/python -m pytest ...`。`pyproject.toml` 的 `pythonpath=["src"]` 会让测试导入 worktree 的代码。
  - 脚本：必须加 `PYTHONPATH=src`。主 `.venv` 的 editable 安装指向主仓库的 `src`，不加就会跑到主仓库的旧代码。先确认：
    `PYTHONPATH=src E:/digital_twin_armigo/.venv/Scripts/python -c "import digital_twin_armigo as m; print(m.__file__)"`
    输出必须是 `E:\digital_twin_armigo-v3\src\...`。
- 主 `.venv` 没有 torch。需要 torch 或 transformers 的测试用 `pytest.importorskip` 跳过；要实际跑的时候用 `E:/il/lerobot-cuda/Scripts/python.exe`（它没有 pytest，可以写一个调用测试函数的小脚本）。Task 5 的 token 长度测试就属于这种。
- **不要安装、升级或卸载任何包**，不要改 `pyproject.toml` 的依赖。
- **GPU 正被训练占用。** 不要启动任何用 GPU 的任务（训练、Isaac、vLLM、CUDA 推理）。MuJoCo 的 CPU 冒烟可以跑，同时最多 4 个进程。
- 缺陷检测器：`results/anomaly/models-sup/sup-v7bot.json`，权重已硬链接到 worktree，不要移动或覆盖。
- 数据输出：冒烟和预检产物写到 `C:/scratch/v3-dev/` 下。不要往 `E:/Armigo_data_v3`、`D:/Armigo_data_v2` 或 `C:/scratch/smolvla-probe` 写东西。
- 代码风格：tab 缩进，`ruff check` 和 `ruff format --check` 对改动的文件通过。ruff 的自动修复可能删掉被再导出的 import（例如 `student_rollout.py` 里的 `POLICY_FIELDS`），修完要跑全量测试确认。

## 5. 每个任务的做法

1. 先写失败的测试，运行，确认失败的原因是对的。
2. 写最少的实现，让测试通过。
3. 跑该任务的测试，再跑全量测试：`E:/digital_twin_armigo/.venv/Scripts/python -m pytest -q`。已知 `test_d1_inputs` 在 GPU 负载下偶尔失败，单独重跑通过即可，在报告里写明。
4. `ruff check` 和 `ruff format --check`。
5. 提交。提交信息第一行写 `v3 Task N: <内容>` 或 `loop F<N>: <内容>`。计划里写的 Claude 署名行不要用，换成你自己的署名行，例如 `Co-Authored-By: Gemini <noreply@google.com>`。

**旧案例逐位不变**是最重要的约束。Task 2、4、5、9 动到 Layout、MJCF、相机或导出时，都要有一条测试对旧的三元组和四元组案例比较改动前后的输出（MJCF 字符串、相机位姿、Layout 字段逐个相等）。参考值从 `7cc24e2` 上生成，写成测试数据，不要在测试里现算。

## 6. 完成后的报告

在 `docs/superpowers/plans/2026-11-24-gemini-handoff-v3-report.md` 写报告并提交，内容：

- 每个任务一行：任务号、提交号、新增和修改的文件、新增测试数。
- 全量测试的最后一行输出（通过、失败、跳过的数目），以及跳过的测试是哪些、为什么跳过。
- 和计划不一致的地方：每条写明计划原文、实际做法、原因。
- 没做完或跳过的步骤。
- 需要用户或 Claude 决定的问题。
- 冒烟和预检跑了什么、产物在哪个目录。

不要在报告里写"已验证"之类的结论，除非给出对应的命令和输出。

## 7. 验收清单（Claude 会逐条检查）

- [ ] 所有提交都在 `feat/smolvla-static-v3`，没有推送；`E:/digital_twin_armigo` 的工作区和分支没有变化。
- [ ] 每个任务一个提交，提交前全量测试通过。
- [ ] 旧案例逐位不变：Claude 会在 `7cc24e2` 和分支头上分别为 val2 的 5 个案例和 train 的 5 个案例生成 MJCF、相机位姿和 Layout，逐位比较。
- [ ] v3 案例由种子确定：同一种子在两个新进程里生成的规格和 MJCF 完全相同。
- [ ] 调色板 8 色，含白和黑；4 个留出的颜色组合不出现在训练种子段；白碗只配 V ≤ 0.68 的桌面。
- [ ] 学生输入没有特权信息：`FORBIDDEN_DIAGNOSTIC_KEYS`、`FORBIDDEN_CAMERA_NAMES` 的守护测试通过；v3 数据的 `state[31:34]` 全为零。
- [ ] 指令改写有 6 种，第 5 种只出现在 ood-lang；每条 ≤ 40 token（Claude 用 lerobot-cuda 的分词器复核）。
- [ ] realistic 教师找不到碗或积木时不退回真值，回合按失败终止并能被 F1 的规则 R1 分诊。
- [ ] Task 8 的预检报告列出每个因子的分布表，以及可达性、可见性、颜色可分性三项检查剔除的案例数。
- [ ] Task 9 导出只含 `top_rgb_v2`、`top_depth_v2`、`wrist` 三路相机。
- [ ] F1–F4 能在现有 v2 数据（`D:/Armigo_data_v2`，只读）上跑通：分诊、台账、复现库（同一提交两次运行逐位相同）、三道关的统计。
- [ ] F5 新增的工具字段全部通过 `assert_sensor_only`，旧字段不变。
- [ ] `ruff check`、`ruff format --check` 通过，tab 缩进。
