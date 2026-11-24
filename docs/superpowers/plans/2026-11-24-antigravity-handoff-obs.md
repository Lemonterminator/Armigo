# 交接：失败可观测性、CV 闭环动作、积木分割网络与配对数据治理（给 Antigravity）

写于 2026-11-24。实现完成后由 Claude 验收。

## 0. 开始条件

**v3 交接（`2026-11-24-gemini-handoff-v3.md`）的 F1–F5 必须已经做完、报告已经提交，并且用户明确说"开始 obs"，才能开始。**

如果 Claude 验收 F1–F5 时提出了修改，先在 `feat/smolvla-static-v3` 上把修改做完，再从它的头切出本分支。

## 1. 位置

- 新建 worktree 和分支（只改 git 元数据，不碰主仓库的工作区）：

  ```powershell
  git -C E:/digital_twin_armigo-v3 worktree add -b feat/failure-observability E:/digital_twin_armigo-obs feat/smolvla-static-v3
  ```

- **第一个提交只放文档。** 把下面三个文件从 `E:/digital_twin_armigo/docs/superpowers/plans/` 复制到新 worktree 的同一路径，提交 `obs docs: plans and handoff`：
  - `2026-11-24-failure-observability-harness-learning.md`
  - `2026-11-24-paired-trajectory-governance.md`
  - `2026-11-24-antigravity-handoff-obs.md`（本文）
  - 这些计划在主仓库里还没有提交，所以复制之后不要回头修改主仓库里的副本。
- 所有修改、测试和提交都在 `E:/digital_twin_armigo-obs` 里做。
- **不要碰 `E:/digital_twin_armigo`**（训练在跑），也不要动 `E:/digital_twin_armigo-v3` 的分支和工作区。
- 不推送，不合并，不改 `main`。

## 2. 要读的文件（按顺序）

1. 本文。
2. `2026-11-24-failure-observability-harness-learning.md`：观测计划，分三部分（H1–H8、C1–C7、U1–U5）。以第三部分末尾的"执行总顺序"为准。
3. `2026-11-24-paired-trajectory-governance.md`：配对运行与数据治理（P1–P6）。2.4 节是用户已经做的决定。
4. 作为参考：
   - `2026-11-24-agent-failure-loop.md`（F1–F7）；
   - `2026-11-24-smolvla-static-dataset.md`（种子段、Task 9 导出）；
   - `2026-11-24-gemini-handoff-v3-report.md`（你上一轮的报告）；
   - `docs/EXPERIMENTS_V2.md`。

各计划的 Global Constraints 适用于其中每个任务。和本文冲突时，以本文为准。

## 3. 范围

按这个顺序做，一个任务一个提交：

1. 观测计划 **H1 → H8**。
2. **C1 → C6**，以及 **C7 的 Step 1–2**（预注册文档和矩阵脚本）。
3. **U1**（包括用 CPU 生成正式数据集）、**U2 的 Step 1–4**、**U3**、**U4 的 Step 1–3**、**U5 的 Step 1**。
4. 治理计划 **P1 → P6**，P6 的 Step 6 除外。

**不做**（全部要用 GPU 或者要用户决定，由用户和 Claude 安排）：

- H6 Step 8；C7 Step 3–5；U2 Step 5；U4 Step 4；U5 Step 2（要等 U2 正式模型）；P6 Step 6；
- 数据集计划 Task 10、11，以及 F6、F7；
- 修改任何默认值，使教师行为发生变化（`motion`、`evidence`、`belief_cameras`、`segmenter` 的默认值都保持旧行为）；
- 计划之外的重构、改名和"顺手"清理。

做到某一步发现计划有错、缺信息，或者和你在 F1–F5 里的实现冲突（例如 F5 已经加了 `reprojection_px`）时：不要自己改规格，也不要删掉已有的实现。在报告里写明，跳过这一步，继续做不依赖它的任务。

## 4. 环境

- **解释器**：用主仓库的 `.venv`：`E:/digital_twin_armigo/.venv/Scripts/python`。
  - 测试在 `E:/digital_twin_armigo-obs` 下运行 `... -m pytest`。
  - 脚本必须加 `PYTHONPATH=src`，先确认：
    `PYTHONPATH=src E:/digital_twin_armigo/.venv/Scripts/python -c "import digital_twin_armigo as m; print(m.__file__)"`
    输出必须是 `E:\digital_twin_armigo-obs\src\...`。
- **torch**：主 `.venv` 没有 torch。U2 的 torch 部分用 `E:/il/lerobot-cuda/Scripts/python.exe`，这个环境没有 pytest，可以写小脚本调用测试函数。在本交接范围内，**一律先设 `CUDA_VISIBLE_DEVICES=""` 只用 CPU**：生成微型 ONNX fixture、跑 1 个 epoch 的冒烟。
- **不要安装、升级或卸载任何包**，不要改 `pyproject.toml` 的依赖。ONNX 导出用 `dynamo=False`；如果仍然缺包，写进报告。
- **GPU 正被训练占用**，不要启动任何用 GPU 的任务。MuJoCo 的 CPU 任务最多同时开 4 个进程，U1 的正式数据生成也算在内。
- **缺陷检测器**：只有 GPU 步骤会用到，本交接范围内不需要。
- **数据输出**：
  - 冒烟和开发：`C:/scratch/obs-dev/`、`C:/scratch/brickseg-dev/`、`C:/scratch/gov-dry/`；
  - 正式产物只有一个：U1 的数据集，写到 `E:/Armigo_data_v3/perception/brickseg-v1/`；
  - `D:/Armigo_data_v2` 和 `C:/Users/Jiang/.gemini/antigravity/scratch/hackjunction`（U5 读真机帧）都只读。
- **代码风格**：tab 缩进，`ruff check` 和 `ruff format --check` 对改动的文件通过。ruff 自动修复后要跑全量测试，它可能删掉被再导出的 import。

## 5. 每个任务的做法

1. 先写失败的测试，运行，确认失败的原因是对的。
2. 写最少的实现，让测试通过。
3. 跑该任务的测试，再跑全量测试 `E:/digital_twin_armigo/.venv/Scripts/python -m pytest -q`。
4. `ruff check`、`ruff format --check`。
5. 提交。第一行写 `obs H<N>:`、`obs C<N>:`、`obs U<N>:` 或 `gov P<N>:`。计划里写的 Claude 署名行不要用，换成你自己的，例如 `Co-Authored-By: Gemini <noreply@google.com>`。

**默认选项下旧行为逐位不变**是最重要的约束。涉及 C2、C3、C4、C5、U3、U4 的改动，每个都要有一条测试，证明默认选项下的事件序列或输出和改动前完全相同。H8 完成之后，用 `compare_runs` 判定。

计划里需要照抄现有构造方式的地方，例如 `TwinMujoco` 怎样构造、`Cell` 的参数：先读现有测试怎么做，再照抄。断言的含义不能改。

## 6. 报告

每做完一部分（H、C、U、P），就在 `docs/superpowers/plans/2026-11-24-antigravity-handoff-obs-report.md` 追加一节并提交，这样中途停下也有记录。全部做完后补上总结，提交，然后**停下等验收**。报告内容：

- 每个任务一行：任务号、提交号、新增和修改的文件、新增测试数。
- 全量测试的最后一行输出；跳过的测试是哪些、为什么跳过。
- 和计划不一致的地方：每条写明计划原文、实际做法、原因。
- 没做完或跳过的步骤。
- 需要用户或 Claude 决定的问题。
- 跑了哪些冒烟和集成检查、产物在哪里。下面几项必须给出数字：
  - C1 Step 5：误差、各失败原因的计数、实测倾角；
  - U1：帧数、丢弃数、预览图目录；
  - P6 Step 5：v2 只读演练的配对结果计数。
- 不要写"已验证"之类的结论，除非给出对应的命令和输出。

## 7. 验收清单（Claude 会逐条检查）

- [ ] 所有提交都在 `feat/failure-observability`，没有推送；`E:/digital_twin_armigo` 和 `E:/digital_twin_armigo-v3` 的工作区和分支没有变化。
- [ ] 每个任务一个提交，提交前全量测试通过。
- [ ] `ROOT_CAUSES` 没变；用 p2-dev-r2 重新评分后，`root_cause` 计数和观测计划第 0 节相同。
- [ ] 真值诊断键都已登记；新事件的传感器字段通过 `assert_sensor_only`；oracle 回合 `student_eligible` 为假。
- [ ] 默认选项下逐位不变：Claude 会在分支起点和分支头上，对 p2_dev12 的 4 个案例跑 realistic B，用 `compare_runs` 比较。
- [ ] `wrist_localize` 只用设计安装位（G10）：有一条测试在腕相机安装误差非零时检查 `believed_wrist_from_cam` 等于设计值。
- [ ] 微型 ONNX 能被 `cv2.dnn` 加载；真机图片和正式模型权重不在仓库里；U5 没有计算任何真机指标，hackjunction 目录没有被写入。
- [ ] 可训练的 gold 产品拒绝 test 段、隔离、oracle 和 `realistic_only` 四类回合；同样的输入构建两次，manifest 逐字节相同。
- [ ] 默认值没有改；预注册文档的提交早于任何相关运行。
