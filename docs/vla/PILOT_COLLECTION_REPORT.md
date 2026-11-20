# Pilot 采集实验报告（train 300 / val 60）

日期：2026-11-20。分支：采集代码来自 `pilot-parallel-export`（提交 `d793142`），检测修复来自 `wave2-integration`（`fdf03b9`），两者已合入 main。
计划依据：`docs/VLA_DISTILLATION_PLAN.md`（Pilot D1-D4 默认值已在 `dd381f3` 确认）。

## 1. 结论

两个数据集的 S4、S6 门槛均通过，导出数据合并回读验证通过，可进入 S9 训练。

| 数据集 | 条数 | S4 专家成功率（门槛 ≥95%） | S6 最差类别通过率（门槛 ≥ 总体 × 0.8） | 总体通过率 | 导出条数 | 帧数 | 数据大小 |
|---|---|---|---|---|---|---|---|
| train | 300 | 96.7%（290/300）通过 | 0.72（总体 0.827，门槛 0.661）通过 | 0.827 | 248 | 316,715 | 18 GB |
| val | 60 | 95.0%（57/60）通过 | 0.714（总体 0.867，门槛 0.693）通过 | 0.867 | 52 | 69,380 | 4.1 GB |

- 导出条数 = 评分通过的条数（train 248 = 300 × 0.827，val 52 = 60 × 0.867）；未通过的不导出。
- 单条帧数：train 737–2784，均值 1277；val 802–1875，均值 1334。
- 合并后的 LeRobot 数据集（`repo_id = local/armigo-six-face`）：`E:\scratch\vla-pilot-train\lerobot`、`E:\scratch\vla-pilot-val\lerobot`。两边都通过 `--verify-dataset`（条数、总帧数、逐条帧数与 `verify_manifest.json` 一致，抽样解码成功）。
- val 的 S4 恰好踩线（57/60 = 0.95），统计上没有余量，S4 在 val 上只能说“没有低于门槛”。

## 2. 类别通过率

| 类别 | train 条数 | train 通过率 | val 条数 | val 通过率 |
|---|---|---|---|---|
| normal | 100 | **0.720** | 20 | 0.900 |
| defect_face_0 | 34 | 0.941 | 7 | 0.857 |
| defect_face_1 | 34 | 0.912 | 7 | 0.857 |
| defect_face_2 | 33 | 0.909 | 7 | 1.000 |
| defect_face_3 | 33 | 0.818 | 7 | 0.714 |
| defect_face_4 | 33 | 0.879 | 6 | 0.833 |
| defect_face_5 | 33 | 0.818 | 6 | 0.833 |

- train 的 normal 类最差（0.72），val 的 normal 反而是 0.90，说明 normal 的差异来自样本量与种子，不是系统性差。val 每类只有 6–7 条（normal 20 条），单类通过率噪声很大，不要当成类别结论。
- 评分各项的失败条数（一条可同时失败多项）：

| 检查项 | train 失败 | val 失败 |
|---|---|---|
| eligible（专家是否成功且可导出） | 10 | 3 |
| decision_crop（决策裁剪） | 22 | 4 |
| belief_error（信念误差） | 25 | 4 |
| collisions（碰撞） | 3 | 0 |
| smoothness（平滑度） | 25 | 3 |

- 有缺陷样本的决策裁剪：train 198 条全部含十字标记，195 条检查通过；val 39 条全部含，38 条通过。
- 视角对齐：train 673 个视角里，对齐前不匹配 273，对齐后 92；val 136 个里 52 → 17。对齐方法全部为 `orientation`。

## 3. 采集配置

- 套件：`config/vla/train.json`（种子 1100001–1100300，normal : 每个缺陷面 = 3 : 1）、`config/vla/val.json`（60 条）。
- 命令：`scripts/vla/rollout_and_export.py --suite … --workers 8 --batch-size 20 --grader-ref grader_reference_frozen.json --resume --export-jobs 2 --overlap-export`（评分参考已冻结：`E:\scratch\vla-pilot-grader_reference_frozen.json`）。
- 每批 20 条：先 rollout（带视频），评分，再把通过的条数导出为 LeRobot 分片，验证后删除原始 mkv，最后合并所有批次。
- 完整的运行日志在 `E:\scratch\vla-pilot-train.log`、`vla-pilot-val.log`，状态记录在 `vla-pilot-run.status`。

## 4. 并行导出的效果

导出（mkv → LeRobot）是原来的瓶颈。本次新增 `--export-jobs N`：把一批里通过的条数按顺序切成 N 段，各由一个 LeRobot 进程写入 `lerobot-NN` 分片，每个分片验证通过后才删除原始 mkv；按顺序拼接分片得到的条数顺序与单分片一致，所以最终合并和清单没有变化。

train 每批导出耗时（约 17 条通过/批）：

| 批次 | export-jobs | 平均导出耗时 |
|---|---|---|
| 0–2 | 1（串行） | 1330 s |
| 3–9 | 3 | 731 s |
| 10–14 | 2 | 858 s |

- 3 个并行进程比串行快约 45%；换成 2 个后慢一些，是因为 3 个进程曾导致内存耗尽（见下）。
- 这些耗时是在 overlap 模式下测得的（导出与下一批 rollout 同时进行，互相争抢 CPU 和内存），不是干净基准。不要拿它当精确加速比。
- `timing.json` 的 `total_s`（train 5228 s、val 2267 s）只是最后一次 `--resume` 调用的墙钟时间，不是整个采集的总耗时；各批 rollout 之和为 8156 s（train）/ 1647 s（val），导出之和为 13670 s / 2468 s（串行累加，overlap 下实际更短）。

## 5. 运行中的问题

| 时间（本机，+02:00） | 现象 | 处理 |
|---|---|---|
| 11-19 16:36 | 3 个导出进程 + overlap，train 退出码 1，val 随后 exit=1（内存耗尽） | 降到 2 个导出进程（`par2`），加上按套件重试 `--resume` |
| 11-19 23:37 | 并行 rollout 时标定卡 PNG 被读到半截文件 | 修复：`handeye.write_card_texture` 改为临时文件 + `os.replace` 原子写，Windows 下遇到 `PermissionError` 重试 |
| 11-20 01:30 | 控制台窗口被关闭，进程被杀 | 在 `conhost --headless` 下重新启动，`--resume` 续跑 |

- 另外把 `scene.write_marker_textures` 也改成原子写（多进程同时首次生成 ArUco 贴图时会读到空文件）。
- 全程 `--resume` 只跳过已完成的批次，没有重做和丢失已导出的数据；终端没有人工介入。train 在 02:57 结束，val 在 03:35 结束。
- 日志里没有 `MemoryError` / `OOM` 字样（内存耗尽是由退出码和系统行为判断的，不是日志里的明确报错）。

## 6. 代码改动（已提交到 main）

- `d793142`（`pilot-parallel-export`）：并行导出、两处 PNG 原子写、回归测试（`tests/test_handeye.py` 两个并发测试，`tests/test_vla_rollout_pipeline.py` 的 `test_split_contiguous_…` 与 `test_parallel_overlapped_export_shards_resume_and_merge_order`）。
- `fdf03b9`（`wave2-integration`）：S10 评测用 S5 的逐类型匹配器检测扰动。合并提交里同时带入 S5 扰动、S10 评测与策略服务、S11 接管 HG-DAgger 及其测试。
- 测试编写时发现并修复一处缺陷：原子写测试的读线程在 Windows 上会因 `PermissionError`（文件正被替换）提前退出，测试虽通过但没有真正并发读；已改为读线程遇到该错误就重试。

## 7. 已知问题与风险

1. `tests/test_twin_agent.py::test_recorded_camera_observation_precedes_its_action_and_rewound_trace_is_rejected` 失败（`scan_view` 段缺 `view` 参数，`KeyError: 'view'`）。在合并前的 main（`a13c647`）上同样失败，与本次改动无关，未处理。
2. 合并后的 main 全量测试：657 通过、5 跳过、2 失败（耗时 11 分钟）。其中 `tests/test_d1_inputs.py::test_realistic_top_depth_v2_uses_independent_d415_rng_and_perception_is_unchanged` 只在全量运行时失败（当时我同时在读大数据目录），之后单独跑 3 次、空闲时跑 `test_[a-d]*.py` 都通过，判断为负载下的偶发失败，原因未查明。该测试单个要跑约 3 分钟。
   ruff 在 `src tests scripts/vla` 上报 10 条，都不在本次改动的文件里（`student_rollout.py` 的未使用导入、`test_d1_inputs.py` 的导入排序、`check_cuda_env.py` 等），合并前的 main 上更多（13 条可自动修复）。
3. train 评分失败的 52 条被整体丢弃，没有逐条归类失败原因；`belief_error` 与 `smoothness` 各 25 条是最大的两类，若要提高导出率，应从这两项入手。
4. val 只有 60 条，类别通过率的置信区间很宽；S6 对 val 的结论只能作为粗检。
5. 数据质量只验证了结构（条数、帧数、可解码），训练效果要到 S9 才能看到。

## 8. 下一步

1. 用 `E:\scratch\vla-pilot-train\lerobot` 训练 S9 学生模型，用 val 做离线评估。
2. 对 S10 闭环评测使用已合入的 S5 扰动与逐类型检测。
3. 修复上面的 `scan_view` 旧测试。
