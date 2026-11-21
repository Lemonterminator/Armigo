# scripts/ 索引

脚本是各实验的入口；可复用的逻辑在 `src/digital_twin_armigo/`。多数脚本把结果写到 `results/<同名主题>/`，索引见 [results/README.md](../results/README.md)。带 `--mlflow` 的脚本可把运行记入 MLflow，见 [docs/MLOPS.md](../docs/MLOPS.md)。

| 目录／脚本 | 用途 | 环境 |
| --- | --- | --- |
| `sim/so101_sort_sim.py` | MuJoCo 分拣单元命令行（`fit-camera` / `layout-prior` / `simulate` / `render`）的兼容启动器；包内入口是 `armigo-sim` | 主 venv `--extra mujoco` |
| `sim/compare_outputs.py` | 比对两个仿真／渲染目录，缺文件或数据变化时失败 | 主 venv |
| `calib/handeye_sim.py` | 顶部相机的仿真手眼标定 | 主 venv |
| `isaac/` | Isaac Sim／Lab 阶段 1–3 验证、资产准备、RTX 回放渲染、状态恢复探针 | Isaac Python（`ARMIGO_ISAAC_PYTHON`） |
| `bench_cross_engine.py` | MuJoCo 对 Isaac(PhysX) 的跨引擎物理基准 | 主 venv / Isaac |
| `bench_physics_expert.py` | 纯物理专家在敏感性矩阵上的验收（`--mlflow`） | 主 venv |
| `bench_agent.py` | agent 栈前后对比基准（`--mlflow`） | 主 venv |
| `audit_gripper_state.py` `regress_gripper_state.py` | 夹爪 `BLOCKED` 状态在 660 个实验上的审计与回归 | 主 venv |
| `bowl_experiments/` | 碗碰撞凸分解候选与 G0–G5 门槛 | 主 venv |
| `inspection_visibility.py` `scan_zone.py` `scan_zone_report.py` `top_camera_reach.py` `relocate_experiment.py` `run_heuristic_inspection.py` | 检查姿态、扫描区、重放置与启发式检查的研究脚本 | 主 venv |
| `catalog_record_mujoco.py` | 为资料库新录制一次明确请求的实验 | 主 venv `--extra catalog` |
| `export_failure_animations.py` `render_agent_demo.py` `compare_smoothness_videos.py` | 失败动画、演示视频、平滑度对比 | 主 venv / Isaac |
| `export_agent_logs.py` | 把 Claude Code 会话转录成可读 Markdown（默认不入库） | 主 venv |
| `serve_armigo_qwen.sh` | 启动本地 Qwen 的 vLLM 服务 | shell（LF 换行） |
| `anomaly/` | 异常检测：渲染数据集 → 训练（PatchCore／监督头）→ 评测 → 打分服务；`log_mlflow.py` 记录 | 异常检测 venv（`anomaly/requirements.txt`） |
| `vla/` | VLA 学生策略：套件 → 采集 → 评分 → 训练 → 闭环评测 → DAgger（`train.py`、`eval_closed_loop.py` 支持 `--mlflow`） | LeRobot CUDA venv |

约定：脚本入口里自行把 `src/` 放进 `sys.path`；不要在脚本之间互相 import（`bench_physics_expert.py` 复用 `bench_cross_engine.py` 是历史例外）。`archive/` 里的东西不再维护。
