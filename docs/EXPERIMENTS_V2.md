# 双引擎六面检验实验（v2 数据）

计划：[2026-11-24-dual-engine-six-face-agent-audit.md](superpowers/plans/2026-11-24-dual-engine-six-face-agent-audit.md)。
数据根目录 `D:/Armigo_data_v2`，布局与字段见 [`experiments/contract.py`](../src/digital_twin_armigo/experiments/contract.py)。

## 跑一个实验组

```powershell
# 全知（阶段一），MuJoCo，可续跑
.venv/Scripts/python -m digital_twin_armigo.experiments.runner run --batch p1-dev --engine mujoco --sensing truth --suite dev_suite_700001 --detector v7bot --workers 4
# realistic（阶段二）：相机条件 fixed / recal / drift，取景 A / B / C
.venv/Scripts/python -m digital_twin_armigo.experiments.runner run --batch p2-dev --sensing realistic --camera recal --strategy B --suite p2_dev12 --detector v7bot
.venv/Scripts/python -m digital_twin_armigo.experiments.runner run --batch p2-dev --sensing realistic --camera recal --strategy C --suite p2_dev12 --detector v7bot --review-llm-url http://127.0.0.1:8038/v1
# Isaac 原生闭环（单进程，一个 Isaac 服务复用于整组）
.venv/Scripts/python -m digital_twin_armigo.experiments.runner run --batch p1-dev --engine isaac --sensing truth --suite dev_suite_700001 --detector v7bot --est-mib 40
.venv/Scripts/python -m digital_twin_armigo.experiments.runner status --batch p1-dev
.venv/Scripts/python -m digital_twin_armigo.experiments.report --root D:/Armigo_data_v2 --batch p1-dev --compare truth-v6 truth-v7bot
```

- 回合写到 `runs/<批次>/<引擎>/<组>/<case_id>[-a<n>]/`；组名 `truth-<检测器>` 或 `realistic-<相机>-<策略>-<检测器>`。
- 同一命令再跑一次就是续跑：已完成的跳过，中断、崩溃、基础设施失败的开新尝试目录（`-a2`），`parent_run` 指向上一次，最多 3 次。
- 每组第一次运行写 `plan.json`（套件、检测器摘要、选项、案例列表）；以后选项不一致会拒绝运行，避免一个组里混两种设置。
- 磁盘：每提交一个回合前检查剩余空间，低于 `--min-free-gib`（默认 8）加在途回合的估计体积就停在批次边界，返回码 3。
- 套件在 `config/agent_bench/v2/`（`scripts/experiments/make_suites.py` 生成，`manifest.json` 记 sha256）；首次使用时复制到 `D:/Armigo_data_v2/suites/` 并校验。
- **封存套件**（`p1_sealed`、`p2_sealed`）只有带 `--freeze <freeze.json>` 才能跑；单组的 freeze 文件由 `runner freeze` 在开发结束、代码已提交时写出，记录提交号、检测器摘要、Agent 选项和 Policy，运行时逐项比对，工作区有未提交改动也拒绝。
- 检测器：`v6` = `results/agent-bench/sup-local.json`（`models-sup/sup.json` 的检查点相对路径失效），`v7bot` = `results/anomaly/models-sup/sup-v7bot.json`。

## 配对采集和封存

`run-pair` 用同一套件在同一批次写入全知和 realistic 回合。默认包含 fixed、recal、drift。用 `--case-limit` 固定每组的试采案例数，续跑会跳过已完成的回合。每组的计划固定提交和检测器权重哈希。换代码或模型后开新批次。模拟器和检测器等基础设施失败返回 2，任务本身的失败会正常保存轨迹。

新工作树可通过 `--detector-spec` 指向已有模型 JSON。以下命令假定 PowerShell，当前路径为仓库根目录。封存注明套件、引擎、检测器和各组配置，并将 freeze 的内容哈希写入每个回合。两条命令使用相同的组与模型选项。

```powershell
$armigoDetectorSpec = "E:/digital_twin_armigo/results/anomaly/models-sup/sup-v7bot.json"
$armigoFreezePath = "D:/Armigo_data_v2/freezes/p2-mujoco-B.json"
python -m digital_twin_armigo.experiments.runner freeze-pair --engine mujoco --suite p2_sealed --strategy B --detector-spec $armigoDetectorSpec --out $armigoFreezePath
python -m digital_twin_armigo.experiments.runner run-pair --engine mujoco --suite p2_sealed --strategy B --detector-spec $armigoDetectorSpec --batch p2-sealed-mujoco-B --freeze $armigoFreezePath
```

`--localise vote` 启用形状投票和腕环定位，`--drift-alarm` 再启用一次顶相机重校准。配对命令只在 realistic 侧应用这两个参数。对照组和告警组有独立的组名。执行封存实验时沿用对应的预注册配置。

## Agent 侧的新选项（`TwinMujoco`）

| 选项 | 含义 |
|---|---|
| `strategy="A"` | 现有路径：第一个达到阈值的视图直接判缺陷。 |
| `strategy="B"/"C"` | 报警先挂起，换角度复核（[`agent/review.py`](../src/digital_twin_armigo/agent/review.py)）：另一位姿的第二次报警且共享一个面 → 确认缺陷；报警视图的每个面都有 ≥2 个通过视图、其中至少一个是本次复核 → 撤销；否则未决，预算用完返回 `unknown`。B 按规则选视角（`priority_px` → 位姿变化 → 路径短；`priority_px` 把每个待查面的预测像素截到 6000，再除以 1+该面已有的通过视图数，避免顶面这种大而早已看清的面压过真正待查的侧面），C 让 Qwen 逐张看图（每请求一张：报警裁剪、同面旧裁剪、当前相机、侧视、深度预览）后从同一组候选中选 ID，失败回退到 B 的选择。两个决策点：腕扫内部的报警当场复核；放碗前还有未决报警时进入 `review` 节点。 |
| `max_scored_views` | 每回合评分视图上限（realistic 默认 6，含最多 2 张复核；全知默认不限，与历史一致）。被回滚分支上拍的图也计数。 |
| `boundary="belief_cell"` | realistic 的裁剪框、可见面和取景规划改在一个“信念单元”上计算：用感知到的砖型号尺寸、感知到的碗心、手眼标定估计的顶相机位姿和标称腕相机安装，按关节读数和砖的估计位姿同步。`legacy` 保留旧做法（把仿真砖挪到估计位姿）作开发对照。 |
| 相机模式 `jitter_extrinsic` / `drift` | 与 `jitter` 同一份外参采样，但焦距保持标称；`drift` 的手眼标定在标称顶相机上完成，实体顶相机再移到扰动位姿。`camera_calibrated` 事件多记 `actual_cam_pos_err_mm` / `actual_cam_axis_err_deg`（相对实际安装位姿）。runner 的 `--camera recal` / `drift` 对应这两种模式。 |
| `cell_factory` | 换物理引擎：`isaac_bridge.mirror.mirror_cell_factory()` 返回 Isaac 镜像单元（见下）。 |

新事件：`review_decision`（触发点、候选、选择、来源、回退原因、延迟、token）、`review_view`、`review_unavailable`、`llm_image`（`cached` 标记已发过的图，原图存 `llm_images/`）、`belief_cell`。结果新增 `strategy`、`scored_views`、`review_views`；`placement_evidence` 只在标签为缺陷时给出。

## Isaac 原生闭环

[`isaac_bridge/mirror.py`](../src/digital_twin_armigo/isaac_bridge/mirror.py) 的 `IsaacMirrorCell` 继承 MuJoCo `Cell`：每个控制帧把滤波后的关节目标发给 Isaac 桥（[ISAAC_BRIDGE.md](ISAAC_BRIDGE.md)），Isaac PhysX 推进一帧，返回的关节和砖位姿写回 MuJoCo 模型。规划、专家观测、逐帧日志、评估器和 3D 回放（`replay/`）都读这份镜像状态；相机图像（RGB 和顶视深度）由 Isaac 按需渲染。

- 夹爪接触力来自 Isaac 接触传感器；切向分量和其他连杆受力由 MuJoCo 在镜像位姿上计算（近似）。
- 原生检查点只在张爪且砖静置桌面时存在；全知模式只回滚到这类检查点。
- realistic：Isaac RTX 把粉碗渲染成浅粉（饱和度约 20，MuJoCo 约 106），`perception.BOWL_HSV_BY_ENGINE` 给 Isaac 单独一条粉碗色带（饱和度下限 12，只在开发帧上标定）；MuJoCo 不变。Isaac 上碗和砖的定位误差约 2.5 mm（MuJoCo 0.3–0.8 mm），来自下面的标定限制。
- 显存同驻（RTX 5090 32 GB）：Isaac + 检测器 + Armigo Qwen（`gpu_util` 0.76，16k，`max_num_seqs` 2）实测峰值 31.9 GB，C 的复核延迟约 11 s/次。Qwen 权重 19.8 GiB，`gpu_util` 低于约 0.72 时 KV 缓存为负、起不来，所以没有更多余量；同驻时 Isaac 只能单进程。
- 已知限制：手眼标定仍在 MuJoCo 的标定场景里完成（未经 Isaac 渲染）；确定性：新进程从头跑同一案例逐位可复现（`600008` 两例各 3 次，状态轨迹、评分、终局完全相同），但同一进程内从检查点恢复后重放不是逐位的（桥探针：约 0.4 mm / 7 mrad），所以回滚后的分支只当作新尝试，不当作复现；每帧约 60 ms，一回合 3–5 分钟、约 20–23 MiB。
