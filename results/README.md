# results/ 索引

已提交的是**报告和汇总数字**（JSON／JSONL／Markdown）；大体量数据不进 Git：渲染数据集、录像、动画、模型权重、日志见根目录 `.gitignore`。数据的版本用 `dataset_manifest.json` 标识，模型和实验用 MLflow 记录，见 [docs/MLOPS.md](../docs/MLOPS.md)。

| 目录 | 来自 | 报告 |
| --- | --- | --- |
| `verification-2026-11-17/` `verification-physics/` `verification-physics-v2/` | `scripts/bench_cross_engine.py`、`scripts/isaac/` | [REPORT](verification-2026-11-17/REPORT.md)、[CROSS_ENGINE_REPORT](verification-physics/CROSS_ENGINE_REPORT.md)、[AUDIT_REPORT](verification-physics-v2/AUDIT_REPORT.md) |
| `stage1/` `stage2/` `stage2_d415_check/` `stage2_test_2x2/` `stage3/` | `scripts/isaac/stage*.py` | — |
| `physics-expert-v2/` | `scripts/bench_physics_expert.py` | — |
| `gripper-state/` | `scripts/audit_gripper_state.py`、`regress_gripper_state.py` | — |
| `bowl-experiments/` | `scripts/bowl_experiments/` | — |
| `failure-export/` | `scripts/export_failure_animations.py` 等 | [physics_diagnosis](failure-export/physics_diagnosis/REPORT.md) |
| `heuristic-inspection/` `inspection-policies/` `inspection-visibility/` `scan-zone/` `top-camera-reach/` `relocate/` `orbit-scan/` | `scripts/` 下对应的研究脚本 | [orbit-scan](orbit-scan/REPORT.md) |
| `handeye-sim/` | `scripts/calib/handeye_sim.py` | [REPORT](handeye-sim/REPORT.md) |
| `anomaly/` | `scripts/anomaly/`（各子目录是一次训练／评测，`sup-*` 为监督头，`v*-*` 为 PatchCore 基线） | [REPORT](anomaly/REPORT.md) 及各版本目录下的 REPORT |
| `agent-bench/` `agent-twin/` | `scripts/bench_agent.py`、agent 验收 | [agent-bench](agent-bench/REPORT.md)、[agent-twin](agent-twin/REPORT.md) |

VLA 的实验报告在 [docs/vla/](../docs/vla/)，套件与评分参照在 [config/vla/](../config/vla/)。

约定：新一轮实验写到带日期或版本的新目录，不覆盖旧结果；目录里放一个 `REPORT.md` 说明命令、数据版本和结论，并在上表登记一行。
