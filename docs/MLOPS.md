# 数据与模型版本治理（数据工程 + MLflow）

本地优先、不新增服务：**数据版本**由目录清单（`dataset_manifest.json`）和 `armigo-data` 的冻结划分表示，**实验与模型版本**由 MLflow 记录。每次训练或评测运行同时记下代码提交和输入数据的摘要，结果能追到确切的代码与数据。

## 对象与身份

| 对象 | 版本标识 | 存放 |
| --- | --- | --- |
| 渲染数据集、LeRobot 导出、录制目录 | `dataset_manifest.json` 里的 `digest`（逐文件大小 + SHA-256 的合并哈希） | 数据目录内，随数据走 |
| 实验资料库的冻结划分 | `armigo-data datasets` 列出的 `id`（成员清单的哈希） | `data/catalog/catalog.sqlite` |
| 一次训练／评测 | MLflow run（参数、指标、产物、`git.commit`、`dataset.<角色>.digest`） | `mlflow/`（Git 忽略） |
| 可部署的模型 | MLflow Model Registry 的版本和别名（如 `cross-detector@candidate`） | 同上 |

清单规则：文件的大小或内容任一变化、增删，`digest` 就变；重建时大小和修改时间不变的文件复用旧哈希，大数据集刷新只重读改动的文件。`dataset_manifest.json` 自身不参与哈希。

## 日常用法

数据集：

```powershell
armigo-track dataset-manifest results/anomaly/dataset-v4 --kind anomaly-render --param seed=2027
armigo-track verify results/anomaly/dataset-v4          # 数据被改动、缺文件或多文件时返回非零
armigo-data datasets                                    # 冻结划分及各划分条目数
```

训练与评测脚本都有**可选**的 `--mlflow`（默认关闭，不加时行为不变）：

```powershell
# 异常检测：训练后记录；也可补录历史报告（幂等，已记录的跳过）
E:\venvs\armigo-anomaly\Scripts\python.exe scripts/anomaly/log_mlflow.py --run-dir results/anomaly/sup-lora --tag lora
E:\venvs\armigo-anomaly\Scripts\python.exe scripts/anomaly/log_mlflow.py --backfill --no-hash-datasets

# VLA 训练 / 闭环评测
python scripts/vla/train.py ... --mlflow --register-model armigo-vla
python scripts/vla/eval_closed_loop.py ... --mlflow

# 基准
python scripts/bench_physics_expert.py --mlflow
python scripts/bench_agent.py --mlflow episodes --suite ... --out ...
```

查看：`armigo-track ui`（默认 <http://127.0.0.1:5000>）。

## 实验与模型注册

| 实验名 | 来源 | 关键指标 |
| --- | --- | --- |
| `cross-detector` | `scripts/anomaly/log_mlflow.py`（监督头与 PatchCore 基线） | `<head>_<scope>_auroc / recall / fpr / best_f1` |
| `vla-train` | `scripts/vla/train.py --mlflow` | `val.*`、`best_end_step_f1`，标签 `s9_gate_pass` |
| `vla-eval` | `scripts/vla/eval_closed_loop.py --mlflow` | 闭环汇总里所有数值（排序成功率、碰撞、漏检等），标签 `suite.sha256` |
| `physics-expert-bench` | `scripts/bench_physics_expert.py --mlflow` | 各条件成功数、接触力等 |
| `agent-bench` | `scripts/bench_agent.py --mlflow` | 子命令输出里的数值 |

模型注册：`--register-model <名称>`（异常检测用 `--register-as`）把权重目录登记为该名称的新版本，并把别名 `candidate` 指向它。**别名由人决定何时晋升**（例如 `champion`），脚本不会自动晋升。检查点是普通文件而非 MLflow flavor，所以走客户端 `create_model_version`（源为 run 的产物地址），不使用 `runs:/` 形式的 `register_model`。

## 约定与限制

- 存储位置：默认 `<仓库>/mlflow/mlflow.db` 与 `mlflow/artifacts`（Git 忽略）。用 `ARMIGO_MLFLOW_DIR` 指向别的目录（多个 worktree 共享同一个库时用它），或 `ARMIGO_MLFLOW_URI` 指向远程服务。`ARMIGO_MLFLOW=0` 一律关闭记录。
- 运行环境：MLflow 只装在需要的 venv 里（异常检测 venv 已有；主 venv 用 `uv sync --extra tracking`）。请求了 `--mlflow` 而没有装时会明确报错，不会静默丢记录。
- 清单只证明“数据与记录时一致”，不证明数据正确；历史数据（清单之前生成的）从第一次计算清单起才有版本。
- 补录的历史运行只含当时报告里的数字和文件，没有当时的 Git 提交（`git.commit` 是补录时的提交，仅作参考）；运行名和 `source_report` 标签指向原报告。
- `--backfill` 默认会为报告里引用的数据集算清单（数 GB 的渲染图，首次较慢）；`--no-hash-datasets` 跳过。
- 与 `armigo-data` 的分工不变：资料库管**实验条目**（来源、质量、划分、导出），本页的清单管**成品数据目录**；LeRobot 导出包用 `armigo-track dataset-manifest` 登记后，训练脚本的 `--dataset` 会自动带上其摘要。

## 验证

```powershell
uv run pytest tests/test_tracking.py tests/test_catalog.py -q
```

`tests/test_tracking.py` 的真实 MLflow 用例在没有安装 MLflow 的环境里会跳过；它覆盖运行、数据摘要标签、模型注册和别名。
