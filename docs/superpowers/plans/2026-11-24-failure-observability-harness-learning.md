# 失败可观测性、因果归因与 harness 学习 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 现在的失败闭环只能回答“在哪一步失败”。这份计划要让它进一步回答“为什么失败”，并能用反事实复跑来证实；同时把“改工具”这件事本身变成可以度量的对象，用数据说明 harness learning 有没有在逐轮变好、有没有出现二阶改进（RSI 的必要条件）。

**Architecture:** 在现有 v2 管线（runner → scoring/attribution → report → inventory/viewer/audit）上加四层，不新建平行系统：

1. **观测**：给 MCP 工具调用记录 `tool_call` 事件；所有真值诊断字段集中登记，并纳入传感器守护。
2. **归因**：保留现有的“症状”归因（`attribution.first_error` 输出的 `root_cause`），在它上面加一层“机制”（`mechanism`、`upstream`），由感知误差前兆推出，阈值用 dev 数据校准。
3. **复跑**：realistic 回合可以把相机、积木、碗三类估计中的任意子集换成真值（oracle），从而反事实地确认机制；重复运行判定改为“离散轨迹一致加数值容差”。
4. **学习度量**：建立诊断基准（人工确认过的失败，分校准和留出两半）、每轮指标表，再给改工具的人或 agent 提供一个只读的诊断 MCP，并记录诊断成本。

**Tech Stack:** Python 3.14，MuJoCo，FastMCP，SQLite，numpy，pytest，ruff（tab 缩进）。

**Spec:** 本文第 0–3 节。上游计划：[2026-11-24-agent-failure-loop.md](2026-11-24-agent-failure-loop.md)（F1–F7）、[2026-11-24-smolvla-static-dataset.md](2026-11-24-smolvla-static-dataset.md)、[2026-11-24-dual-engine-six-face-agent-audit.md](2026-11-24-dual-engine-six-face-agent-audit.md)、交接文档 [2026-11-24-gemini-handoff-v3.md](2026-11-24-gemini-handoff-v3.md)。

## Global Constraints

- **起点。** 等 Gemini 的 F1–F5 验收通过后，从 `feat/smolvla-static-v3` 的头切出分支 `feat/failure-observability`，新建 worktree `E:/digital_twin_armigo-obs`。在 `E:/digital_twin_armigo` 训练结束之前，不在那里改文件、切分支或提交。
- **环境。** 测试用 `E:/digital_twin_armigo/.venv/Scripts/python -m pytest`；脚本一律加 `PYTHONPATH=src`。不安装、不升级包，不改 `pyproject.toml` 的依赖。
- **GPU。** 标【GPU】的步骤要占检测器或 Isaac，必须由用户或 Claude 排期后再跑。其余步骤只用 CPU，可以同时开最多 4 个 MuJoCo 进程。
- **只读数据。** `D:/Armigo_data_v2` 只读。新产物写到 `E:/Armigo_data_v3/loop/` 或 `C:/scratch/obs-dev/`。
- **向后兼容。** `ROOT_CAUSES` 的取值和含义不变，p1/p2 报告的“第一处错误”表逐项可比。新信息只放进新字段（`mechanism`、`upstream`、`precursors`）。
- **真值只给评估器。** 真值诊断字段只能写进 `events.jsonl` 和评分输出，不能进 realistic agent 的工具返回值，也不能进学生数据。oracle 回合永远 `student_eligible = False`，不进任何训练集。
- **判据先写，开发和验证分开。** 沿用失败闭环计划的规定：阈值只在 dev 批次上校准；诊断基准的留出一半只用来评测，不用来调规则。
- **提交。** 一个任务一个提交，先写测试。提交信息第一行写 `obs H<N>: <内容>`，以 `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` 结尾。不推送，不合并。

---

## 0. 现状（2026-11-24 核实）

| 能力 | 现有实现 | 局限 |
|---|---|---|
| 失败检验 | `experiments/scoring.py`：从 `replay/motion.npz` 离线算物理结局；放桌重扫的回合只评 `verdict_final_cycle` | — |
| 归因 | `experiments/attribution.py::first_error`：13 类 `ROOT_CAUSES`，基于 `view_truth` 的像素真值，取最早的一条发现 | 只到症状层；结果只写进 `reports/<batch>/scores.jsonl` |
| 复跑 | `inventory/rerun.py`（全新重跑，写 `lineage.json`）；`agent/twin_resume.py`（从检查点恢复）；runner 的 `-a<n>` 尝试目录 | 不能做“只改一个因素”的反事实复跑 |
| 审计 | `inventory/audit.py`：只追加；viewer 8767；inventory 的 `root_cause` 分面**只来自人工裁定** | 自动归因进不了 inventory 和 viewer |
| 策略搜索 | `agent/twin_search.py`：54 个策略做逐轮减半 | 搜的是策略参数，不是工具 |
| MCP 工具 | `agent/step_mcp.py`：11 个公共工具加 4 个 realistic 专用工具；`assert_sensor_only` 按 `SIMULATOR_KEYS` 黑名单检查 | 不记录工具调用；黑名单不全（见 G6） |

**p2-dev-r2 实测**（`D:/Armigo_data_v2/reports/p2-dev-r2/scores.jsonl`，216 个回合）：

- 第一处错误：`none` 166，`recovery_failure` 23（**全部是 drift 条件**，其中 22 例最后一个码是 `no_clamp`），`evidence_insufficient` 15，`grasp_or_path_failure` 5，`target_face_not_in_view` 3，`placement_failure` 2，`false_positive_visible` 2。
- 首次 `brick_perceived` 的误差（真值诊断，已经写在事件里）：

| 症状 | 回合 | 位置误差中位数 / 最大（mm） | yaw 误差中位数 / 最大（°） |
|---|---:|---|---|
| `recovery_failure` | 23 | **21.2** / 30.8 | 3.4 / 19.1 |
| `evidence_insufficient` | 15 | 2.9 / 15.8 | **19.8** / 25.5 |
| `none` | 166 | 2.0 / 16.9 | 1.2 / 25.5 |

结论：机制信号已经记录在事件里，但归因没有用它。漂移失败被记成了 `recovery_failure`（症状），真正的原因是标定过期导致积木定位偏差约 2 cm。证据不足的回合背后是顶视 yaw 偏差约 20°。注意 `none` 里也有 yaw 误差到 25° 的回合，所以前兆只是风险因素，不能单独定因，需要第 3 层的反事实复跑来证实。

## 1. 发现的缺口（也作为 Gemini F1–F5 验收的附加检查项）

| 编号 | 缺口 | 对 F1–F5 的影响 | 本计划的处理 |
|---|---|---|---|
| G1 | 归因只到症状层，看不到上游机制 | — | H2、H3、H6 |
| G2 | F1 的 `triage.py` 重新实现了 `first_error` 已有的大部分规则 | F1 的 R2 读 `view_truth.brick_yaw_deg`，这个字段不存在；yaw 误差实际在 `brick_perceived.error_yaw_deg`。F1 往 `ROOT_CAUSES` 加入机制类取值（`perception_pose_error`、`calibration_stale` 等），会把症状和机制混在一个枚举里，破坏和 p1/p2 报告的可比性 | 验收时检查 R2 用的字段名；机制放进单独的 `mechanism` 字段（H2）；F1 的分诊改成包一层 `first_error` 加 `precursors`，不维护第二套规则 |
| G3 | 自动归因不进 inventory 和 viewer | F2 台账里的 viewer 链接没法按根因筛选 | H4 |
| G4 | 没有工具调用记录：agent 问了什么、工具返回了什么、耗时多少，都无法复原 | — | H5 |
| G5 | F5 计划新增 `camera_calibration.reprojection_px`，但现有返回值里已经有 `reprojection_rms_px`；而且它是标定时算的，对标定之后的漂移完全看不见，而漂移恰恰是最主要的失败来源 | F5 这一项是重复的，而且对漂移无效 | 验收时检查有没有重复字段；能在线发现漂移的传感器信号属于新能力，转为独立计划（第 3.4 节） |
| G6 | `SIMULATOR_KEYS` 缺 `actual_cam_pos_err_mm`、`actual_cam_axis_err_deg`、`calibrated_on`。目前没有泄漏（`camera_calibration` 只返回白名单字段），但 F5 会从事件里取新字段，有泄漏风险 | F5 的守护测试漏掉了这三个键 | H1 |
| G7 | F3 要求“两次运行逐位相同”，没有说明检测器分数（GPU）和墙钟时间字段怎么处理 | F3 的复现检查可能误剔案例，也可能误收 | H8 的 `compare_runs` 给出明确的判定口径 |
| G8 | 闭环过程本身没有被度量：分诊准不准、修一个簇要多少成本、诊断能力有没有变强，都没有数字，“harness learning / RSI”这个说法因此无法证伪 | — | H7、H8 |

## 2. 设计

### 2.1 归因分三层

| 层 | 字段 | 来源 | 是否定因 |
|---|---|---|---|
| 症状 | `root_cause`（不变） | `first_error` | 是“在哪失败” |
| 机制 | `mechanism`、`upstream`、`precursors` | `precursors(events)`：真值诊断超过阈值，且发生在症状之前，并且和症状相关 | 是假设 |
| 证实 | `counterfactual.verdict` | oracle 复跑：换掉哪一类估计就能翻成成功 | 是因果结论 |

机制与症状的相关表（`RELEVANT`）：

| 症状 | 可解释它的机制（按优先级） |
|---|---|
| `grasp_or_path_failure`、`recovery_failure` | `brick_pose_error`、`brick_yaw_error` |
| `placement_failure` | `bowl_pose_error`、`brick_pose_error` |
| `evidence_insufficient`、`target_face_not_in_view`、`premature_stop` | `brick_yaw_error`、`view_belief_error`、`brick_pose_error` |
| `crop_error`、`occlusion` | `view_belief_error`、`brick_pose_error`、`brick_yaw_error` |
| `false_*_visible`、`infra_failure`、`none`、`other` | 无（检测器、基础设施问题，或成功） |

`upstream = "calibration_drift"`：当机制是某种位姿误差，并且同一回合里顶相机的实际外参误差超过阈值时填写。

### 2.2 反事实复跑（oracle 替换）

对一个失败的 realistic 回合，用同一案例、同一种子跑 5 个变体：`∅`（原样重跑，用来检查能否复现）、`camera`、`brick`、`bowls`、`camera+brick+bowls`。判定：

- `∅` 的物理结局和原回合不同 → `repro_mismatch`，这个案例不进复现库；
- 有单项变体成功 → `confirmed:<最小集合>`（可能不止一个）；
- 只有全替换才成功 → `joint`；
- 全替换仍然失败 → `not_perception`：问题在工具、策略或检测器，不在感知。

oracle 只能用于评估，回合不进训练、不进台账的频次统计；它的结果单独写进 `counterfactual.jsonl`。

### 2.3 harness learning 的度量，以及什么时候能谈 RSI

| 层级 | 定义 | 怎么证明 |
|---|---|---|
| L0 测量 | harness 固定，只出报告 | 现状 |
| L1 harness learning | 台账驱动修工具，过三道关才合并 | 新种子上的成功率逐轮上升；`overfit_bank` 的比例下降 |
| L2 二阶改进 | 诊断工具本身被改进，并且让下一轮的诊断更准、更便宜 | 在**同一份留出诊断基准**上，新旧诊断工具做 A/B：准确率不降，每例诊断成本（工具调用次数、墙钟时间）下降，并且至少连续两轮成立 |
| L3 RSI | 提出、验证、合并都不经人工把关，并且改进速度本身在增长 | 本计划不做。前提是 L2 成立，并且三道关加封存集足以替代人工把关；到那时另写计划 |

**每轮指标**（`round_metrics.json`，H7）：

1. 任务：新种子上的分拣成功率及区间（三道关的第 3 关）。
2. 归因覆盖：失败回合里 `root_cause ∈ {other, unclassified}` 的比例；有 `mechanism` 的比例；有反事实证实的比例。
3. 归因准确：诊断基准留出一半上，自动症状和机制与人工的一致率（Cohen κ）。
4. 修复产出：本轮尝试的修复中 `merge`、`overfit_bank`、`reject` 各几个。
5. 回退：本轮所有修复的回退集 n10 总和。
6. 诊断成本：诊断 MCP 的每例工具调用次数和墙钟时间中位数，以及首次正确定因所需的调用次数。

**防 Goodhart。** 诊断基准分 `calib` 和 `heldout` 两半，规则和阈值只看 `calib`；`heldout` 每轮只评测一次，结果写进指标表之后不再回头改规则。封存套件永远不进诊断基准。

---

## 文件结构

| 文件 | 改动 | 职责 |
|---|---|---|
| `src/digital_twin_armigo/agent/step_mcp.py` | 改 | `TRUTH_DIAGNOSTIC_KEYS` 并入 `SIMULATOR_KEYS`（H1）；工具调用记录（H5） |
| `src/digital_twin_armigo/experiments/precursors.py` | 新 | `MECHANISMS`、`precursors()`、`explain()`、`load_thresholds()` |
| `src/digital_twin_armigo/experiments/attribution.py` | 改 | `first_error` 的输出加 `mechanism`、`upstream`、`precursors` |
| `src/digital_twin_armigo/experiments/scoring.py`、`report.py` | 改 | 评分行加机制列；报告加“症状 × 机制”表 |
| `scripts/experiments/calibrate_precursors.py` | 新 | 在 dev 批次上校准阈值，写 `config/experiments/precursor_thresholds.json` |
| `src/digital_twin_armigo/inventory/scan.py`、`store.py`、`query.py` | 改 | `auto_root_cause`、`auto_mechanism`、`auto_upstream` 三列及其筛选和分面 |
| `src/digital_twin_armigo/inventory/audit.py` | 改 | 可选的 `mechanism` 字段（只追加，自动迁移） |
| `src/digital_twin_armigo/agent/twin_mujoco.py` | 改 | `oracle` 选项、`true_world_from_cam()` |
| `src/digital_twin_armigo/experiments/runner.py`、`contract.py` | 改 | `--oracle`、实验字段 `oracle` |
| `src/digital_twin_armigo/experiments/counterfactual.py`、`scripts/experiments/counterfactual.py` | 新 | 反事实变体的判定和命令行 |
| `src/digital_twin_armigo/experiments/diagnosis.py`、`scripts/experiments/make_diagnosis_bench.py`、`scripts/experiments/round_metrics.py` | 新 | 诊断基准和每轮指标 |
| `src/digital_twin_armigo/experiments/diag_mcp.py` | 新 | 给改工具的人或 agent 用的只读诊断 MCP，含 `compare_runs` |
| `tests/test_truth_diagnostics.py`、`tests/test_precursors.py`、`tests/test_counterfactual.py`、`tests/test_diagnosis.py`、`tests/test_diag_mcp.py` | 新 | 测试 |

依赖：H1 → H2 → H3 → H4；H5 独立；H6 依赖 H1；H7 依赖 H2 和 H4；H8 依赖 H2、H5、H6、H7。

---

### Task H1：真值诊断键登记与守护加固

**Files:**
- Modify: `src/digital_twin_armigo/agent/step_mcp.py:45-72`（`SIMULATOR_KEYS`）
- Test: `tests/test_truth_diagnostics.py`

**Interfaces:**
- Produces: `step_mcp.TRUTH_DIAGNOSTIC_KEYS: frozenset[str]`，并保证 `TRUTH_DIAGNOSTIC_KEYS <= SIMULATOR_KEYS`。

- [ ] **Step 1：写失败的测试**

```python
# tests/test_truth_diagnostics.py
"""Every truth diagnostic the simulator writes into events must be on the sensor-only deny list."""

import ast
import re
from pathlib import Path

import pytest

pytest.importorskip("mujoco")

from digital_twin_armigo.agent import step_mcp as sm

SRC = Path(sm.__file__).resolve().parents[1]
DIAG = re.compile(r"^(true_|actual_|error_|belief_error)|_err(_|$)|_correct$|truth|^calibrated_on$")


def _event_keys(path: Path) -> set[str]:
	"""Keyword names passed to ``*.event(...)``, ``info["..."] = ...`` keys, and string keys of dict literals."""
	tree = ast.parse(path.read_text(encoding="utf-8"))
	keys: set[str] = set()
	for node in ast.walk(tree):
		if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "event":
			keys |= {k.arg for k in node.keywords if k.arg}
		elif isinstance(node, ast.Assign):
			for t in node.targets:
				if isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant) and isinstance(t.slice.value, str):
					keys.add(t.slice.value)
		elif isinstance(node, ast.Dict):
			keys |= {k.value for k in node.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)}
	return keys


def test_truth_diagnostics_are_denied():
	assert sm.TRUTH_DIAGNOSTIC_KEYS <= sm.SIMULATOR_KEYS


@pytest.mark.parametrize("rel", ["agent/twin_mujoco.py", "mujoco_cell/handeye.py"])
def test_every_diagnostic_key_written_is_registered(rel):
	written = {k for k in _event_keys(SRC / rel) if DIAG.search(k)}
	missing = written - sm.TRUTH_DIAGNOSTIC_KEYS
	assert not missing, f"{rel} writes truth diagnostics not in TRUTH_DIAGNOSTIC_KEYS: {sorted(missing)}"


def test_calibration_drift_fields_are_rejected():
	for key in ("actual_cam_pos_err_mm", "actual_cam_axis_err_deg", "calibrated_on"):
		with pytest.raises(sm.SimulatorLeak):
			sm.assert_sensor_only({"camera": "top", key: 1.0})
```

- [ ] **Step 2：运行，确认失败**

Run: `E:/digital_twin_armigo/.venv/Scripts/python -m pytest tests/test_truth_diagnostics.py -v`
Expected: FAIL，报 `AttributeError: ... TRUTH_DIAGNOSTIC_KEYS`。

- [ ] **Step 3：实现。** 在 `step_mcp.py` 的 `SIMULATOR_KEYS` 前面加：

```python
# Simulator truth written into events for scoring only (attribution, precursors). Never part of a tool result.
TRUTH_DIAGNOSTIC_KEYS = frozenset(
	{
		"true_face_pixels",
		"true_cross_px",
		"true_face_pixels_crop",
		"true_face_pixels_crop_native",
		"true_cross_px_crop",
		"true_cross_px_crop_native",
		"truth_crop",
		"true_brick_quat_xyzw",
		"belief_error_mm",
		"error_xy_mm",
		"error_yaw_deg",
		"cam_pos_err_mm",
		"cam_rot_err_deg",
		"size_correct",
		"label_correct",
		"actual_cam_pos_err_mm",
		"actual_cam_axis_err_deg",
		"calibrated_on",
	}
)
```

再把 `SIMULATOR_KEYS = frozenset({...})` 改成 `SIMULATOR_KEYS = frozenset({...}) | TRUTH_DIAGNOSTIC_KEYS`。原集合里重复的键可以保留，不影响结果。如果第 2 个测试报出列表之外的键（例如 Gemini 的 Task 3、4 新加的诊断字段），逐个确认它是不是真值：是就加进 `TRUTH_DIAGNOSTIC_KEYS`；不是就在测试的 `DIAG` 正则旁边加一条带注释的排除，并写进报告。

- [ ] **Step 4：运行本任务测试和全量测试**

Run: `E:/digital_twin_armigo/.venv/Scripts/python -m pytest tests/test_truth_diagnostics.py tests/test_step_mcp.py -v`，然后 `... -m pytest -q`
Expected: 全部 PASS。

- [ ] **Step 5：Commit**

```bash
git add src/digital_twin_armigo/agent/step_mcp.py tests/test_truth_diagnostics.py
git commit -m "obs H1: register truth diagnostics and deny them in sensor-only tool results"
```

### Task H2：机制层（前兆）归因

**Files:**
- Create: `src/digital_twin_armigo/experiments/precursors.py`
- Modify: `src/digital_twin_armigo/experiments/attribution.py`（`first_error` 末尾）、`src/digital_twin_armigo/experiments/scoring.py:199-235`（评分行）、`src/digital_twin_armigo/experiments/report.py`（新表）
- Test: `tests/test_precursors.py`

**Interfaces:**
- Produces:
  - `MECHANISMS = ("calibration_drift", "brick_pose_error", "brick_yaw_error", "bowl_pose_error", "view_belief_error")`；
  - `precursors(events: list[dict], thresholds: dict | None = None) -> list[dict]`，每项为 `{"mechanism", "seq", "frame", "metric", "value", "threshold"}`；
  - `explain(symptom: str, symptom_seq: int, found: list[dict]) -> dict`，返回 `{"mechanism": str | None, "mechanism_evidence": dict | None, "upstream": str | None}`；
  - `load_thresholds(path: Path | None = None) -> dict[str, float]`；
  - `first_error(..., thresholds=None)` 的返回值多出 `mechanism`、`mechanism_evidence`、`upstream`、`precursors` 四个键；
  - 评分行多出 `mechanism`、`upstream`；
  - `report.mechanism_table(rows) -> dict[str, dict[str, int]]`。

- [ ] **Step 1：写失败的测试**

```python
# tests/test_precursors.py
from digital_twin_armigo.experiments.attribution import first_error
from digital_twin_armigo.experiments.precursors import DEFAULT_THRESHOLDS, explain, precursors
from digital_twin_armigo.experiments.report import mechanism_table

CALIB_DRIFT = {"seq": 0, "kind": "camera_calibrated", "frame": 0, "actual_cam_pos_err_mm": 21.4, "actual_cam_axis_err_deg": 0.75}
CALIB_OK = {"seq": 0, "kind": "camera_calibrated", "frame": 0, "actual_cam_pos_err_mm": 0.3, "actual_cam_axis_err_deg": 0.1}


def brick(seq, xy, yaw):
	return {"seq": seq, "kind": "brick_perceived", "frame": 10 * seq, "found": True, "error_xy_mm": xy, "error_yaw_deg": yaw}


def view(seq, i, faces, px):
	vid = f"0:{i}:wrist_cam"
	base = {"branch": 0, "frame": 100 * seq}
	truth = {"seq": seq, "kind": "view_truth", "id": vid, "true_face_pixels": px, "true_face_pixels_crop_native": px,
		"true_cross_px": 0, "true_cross_px_crop_native": 0, "belief_error_mm": 2.0, **base}  # fmt: skip
	obs = {"id": vid, "camera": "wrist_cam", "faces": faces, "score": -0.9, "threshold": 0.0, **base}
	return [truth, {"seq": seq + 1, "kind": "view", "observation": obs, **base}]


def test_precursor_rules_and_thresholds():
	found = precursors([CALIB_DRIFT, brick(1, 21.0, 2.0), {"seq": 2, "kind": "bowls_perceived", "error_xy_mm": {"green": 3.0, "pink": 12.0}}])
	assert [p["mechanism"] for p in found] == ["calibration_drift", "brick_pose_error", "bowl_pose_error"]
	assert found[1]["value"] == 21.0 and found[1]["threshold"] == DEFAULT_THRESHOLDS["brick_xy_mm"]
	assert precursors([CALIB_OK, brick(1, 2.0, 1.0)]) == []


def test_drift_grasp_failure_gets_pose_mechanism_and_calibration_upstream():
	events = [CALIB_DRIFT, brick(1, 21.0, 2.0), {"seq": 2, "kind": "failure", "code": "no_clamp", "frame": 1200}]
	episode = {"result": {"label": "unknown", "failures": [{"code": "no_clamp", "frame": 1200}], "rewinds": 1,
		"retries": 0, "views": 0, "evidence": [], "cleared_faces": []}}  # fmt: skip
	task = {"case": [7, False, 5], "policy": {"max_rewinds": 1}}
	out = first_error(episode, events, task, {"true_label": "normal"}, {"success": False, "outcome": "table"})
	assert out["root_cause"] == "recovery_failure"
	assert out["mechanism"] == "brick_pose_error" and out["upstream"] == "calibration_drift"
	assert {p["mechanism"] for p in out["precursors"]} == {"calibration_drift", "brick_pose_error"}


def test_evidence_insufficient_gets_yaw_mechanism_without_upstream():
	events = [brick(0, 2.9, 19.8)]
	big = 3000
	for i, f in enumerate([1, 2, 3, 4]):
		events += view(2 * i + 1, i, [f], [big if j == f else 0 for j in range(6)])
	ids = [e["observation"]["id"] for e in events if e["kind"] == "view"]
	episode = {"result": {"label": "normal", "verdict": "SUCCESS", "evidence": ids, "failures": [], "views": 4,
		"cleared_faces": [0, 1, 2, 3, 4, 5]}}  # fmt: skip
	out = first_error(episode, events, {"case": [7, False, 5]}, {"true_label": "normal"}, {"success": True})
	assert out["root_cause"] == "evidence_insufficient"
	assert out["mechanism"] == "brick_yaw_error" and out["upstream"] is None


def test_detector_symptoms_and_success_get_no_mechanism():
	found = precursors([CALIB_DRIFT, brick(1, 21.0, 20.0)])
	assert explain("false_positive_visible", 50, found)["mechanism"] is None
	assert explain("none", 10**9, found)["mechanism"] is None


def test_precursor_after_the_symptom_is_ignored():
	found = precursors([brick(5, 21.0, 2.0)])
	assert explain("grasp_or_path_failure", 3, found)["mechanism"] is None


def test_mechanism_table_counts_symptom_by_mechanism():
	rows = [{"root_cause": "recovery_failure", "mechanism": "brick_pose_error"},
		{"root_cause": "recovery_failure", "mechanism": "brick_pose_error"},
		{"root_cause": "evidence_insufficient", "mechanism": None}]  # fmt: skip
	assert mechanism_table(rows) == {"recovery_failure": {"brick_pose_error": 2}, "evidence_insufficient": {"(none)": 1}}
```

- [ ] **Step 2：运行，确认失败**

Run: `E:/digital_twin_armigo/.venv/Scripts/python -m pytest tests/test_precursors.py -v`
Expected: FAIL，`ModuleNotFoundError: digital_twin_armigo.experiments.precursors`。

- [ ] **Step 3：实现 `precursors.py`**

```python
"""Mechanism layer above the first-error symptom: truth diagnostics that crossed a tolerance before the symptom.

A precursor is a risk factor, not a cause: successes can carry them too. ``explain`` attaches the earliest precursor
that can explain the symptom (``RELEVANT``) and happened at or before it; the counterfactual oracle re-runs
(``experiments.counterfactual``) are what confirm it.
"""

from __future__ import annotations

import json
from pathlib import Path

MECHANISMS = ("calibration_drift", "brick_pose_error", "brick_yaw_error", "bowl_pose_error", "view_belief_error")
POSE_MECHANISMS = ("brick_pose_error", "brick_yaw_error", "bowl_pose_error", "view_belief_error")
DEFAULT_THRESHOLDS = {
	"cam_pos_mm": 10.0,
	"cam_axis_deg": 1.0,
	"brick_xy_mm": 8.0,
	"brick_yaw_deg": 10.0,
	"bowl_xy_mm": 8.0,
	"view_belief_mm": 8.0,
}
THRESHOLDS_FILE = Path(__file__).resolve().parents[3] / "config" / "experiments" / "precursor_thresholds.json"
RELEVANT = {
	"grasp_or_path_failure": ("brick_pose_error", "brick_yaw_error"),
	"recovery_failure": ("brick_pose_error", "brick_yaw_error"),
	"placement_failure": ("bowl_pose_error", "brick_pose_error"),
	"evidence_insufficient": ("brick_yaw_error", "view_belief_error", "brick_pose_error"),
	"target_face_not_in_view": ("brick_yaw_error", "view_belief_error", "brick_pose_error"),
	"premature_stop": ("brick_yaw_error", "view_belief_error", "brick_pose_error"),
	"crop_error": ("view_belief_error", "brick_pose_error", "brick_yaw_error"),
	"occlusion": ("view_belief_error", "brick_pose_error", "brick_yaw_error"),
}


def load_thresholds(path: Path | None = None) -> dict[str, float]:
	"""Defaults, overridden by the calibrated file (``calibrate_precursors.py``) when it exists."""
	out = dict(DEFAULT_THRESHOLDS)
	p = Path(path) if path else THRESHOLDS_FILE
	if p.is_file():
		data = json.loads(p.read_text(encoding="utf-8"))
		out.update({k: float(v["threshold"]) for k, v in (data.get("metrics") or {}).items() if k in out})
	return out


def _hit(mechanism: str, event: dict, metric: str, value, threshold: float) -> dict:
	return {
		"mechanism": mechanism,
		"seq": int(event.get("seq", 0)),
		"frame": event.get("frame"),
		"metric": metric,
		"value": float(value),
		"threshold": float(threshold),
	}


def precursors(events: list[dict], thresholds: dict | None = None) -> list[dict]:
	t = thresholds or load_thresholds()
	out: list[dict] = []
	for e in events:
		kind = e.get("kind")
		if kind == "camera_calibrated":
			pos, axis = e.get("actual_cam_pos_err_mm"), e.get("actual_cam_axis_err_deg")
			if pos is not None and pos > t["cam_pos_mm"]:
				out.append(_hit("calibration_drift", e, "cam_pos_mm", pos, t["cam_pos_mm"]))
			elif axis is not None and axis > t["cam_axis_deg"]:
				out.append(_hit("calibration_drift", e, "cam_axis_deg", axis, t["cam_axis_deg"]))
		elif kind == "brick_perceived" and e.get("found"):
			xy, yaw = e.get("error_xy_mm"), e.get("error_yaw_deg")
			if xy is not None and xy > t["brick_xy_mm"]:
				out.append(_hit("brick_pose_error", e, "brick_xy_mm", xy, t["brick_xy_mm"]))
			if yaw is not None and yaw > t["brick_yaw_deg"]:
				out.append(_hit("brick_yaw_error", e, "brick_yaw_deg", yaw, t["brick_yaw_deg"]))
		elif kind == "bowls_perceived":
			errs = [v for v in (e.get("error_xy_mm") or {}).values() if v is not None]
			if errs and max(errs) > t["bowl_xy_mm"]:
				out.append(_hit("bowl_pose_error", e, "bowl_xy_mm", max(errs), t["bowl_xy_mm"]))
		elif kind == "view_truth" and e.get("belief_error_mm") is not None:
			if e["belief_error_mm"] > t["view_belief_mm"]:
				out.append(_hit("view_belief_error", e, "view_belief_mm", e["belief_error_mm"], t["view_belief_mm"]))
	return out


def explain(symptom: str, symptom_seq: int, found: list[dict]) -> dict:
	relevant = RELEVANT.get(symptom, ())
	cands = [p for p in found if p["mechanism"] in relevant and p["seq"] <= symptom_seq]
	best = min(cands, key=lambda p: (p["seq"], relevant.index(p["mechanism"]))) if cands else None
	drift = any(p["mechanism"] == "calibration_drift" for p in found)
	upstream = "calibration_drift" if best and drift and best["mechanism"] in POSE_MECHANISMS else None
	return {"mechanism": best["mechanism"] if best else None, "mechanism_evidence": best, "upstream": upstream}
```

- [ ] **Step 4：接进 `first_error`。** 给 `first_error` 加关键字参数 `thresholds: dict | None = None`。把末尾三处返回改成经过同一个函数：

```python
def _with_mechanism(finding: dict, events: list[dict], thresholds) -> dict:
	from .precursors import explain, precursors

	found = precursors(events, thresholds)
	seq = finding.get("seq", 10**9)
	return {**_strip(finding), **explain(finding["root_cause"], seq, found), "precursors": found}
```

- 无结果的提前返回：`return _with_mechanism(min(findings, key=lambda f: f["seq"]), events, thresholds)`；
- `none` 分支：`return _with_mechanism({"root_cause": "none", "frame": None, "branch": None, "view_id": None, "detail": "sufficient evidence", "seq": 10**9}, events, thresholds)`；
- 正常返回：`return _with_mechanism(min(findings, key=lambda f: f["seq"]), events, thresholds)`。

- [ ] **Step 5：评分和报告。**
  - `scoring.score_episode` 的行里，在 `"root_cause"` 后面加 `"mechanism": error.get("mechanism"), "upstream": error.get("upstream"),`。
  - `report.py` 加：

```python
def mechanism_table(rows: list[dict]) -> dict[str, dict[str, int]]:
	"""Failed (root_cause != none) episodes counted by symptom x mechanism; ``(none)`` = no precursor explains it."""
	out: dict[str, dict[str, int]] = {}
	for r in rows:
		if r.get("root_cause") in (None, "none"):
			continue
		cell = out.setdefault(r["root_cause"], {})
		key = r.get("mechanism") or "(none)"
		cell[key] = cell.get(key, 0) + 1
	return out
```

  - 在 `build_summary` 的返回字典里加 `"mechanisms": mechanism_table(rows)`；`render_markdown` 在“第一处错误分布”之后加一节 `## 症状 × 机制（前兆，未经反事实证实）`，每个症状一行，列是 `MECHANISMS` 加 `(none)`。

- [ ] **Step 6：运行测试（含旧的评分测试，确认 `root_cause` 不变）**

Run: `E:/digital_twin_armigo/.venv/Scripts/python -m pytest tests/test_precursors.py tests/test_experiments_scoring.py -v`，然后全量测试。
Expected: 全部 PASS。

- [ ] **Step 7：在真实数据上核对（只读，CPU）**

```bash
PYTHONPATH=src E:/digital_twin_armigo/.venv/Scripts/python -c "
import json, collections
from pathlib import Path
from digital_twin_armigo.experiments.scoring import score_episode
rows = [score_episode(Path(json.loads(l)['path'])) for l in open('D:/Armigo_data_v2/reports/p2-dev-r2/scores.jsonl')]
print(collections.Counter((r['root_cause'], r['mechanism'], r['upstream']) for r in rows if r['root_cause'] != 'none'))
"
```

Expected：`root_cause` 的计数和第 0 节完全一致；drift 下的 `recovery_failure` 大多数为 `(brick_pose_error, calibration_drift)`。把输出贴进提交说明。

- [ ] **Step 8：Commit**

```bash
git add src/digital_twin_armigo/experiments/precursors.py src/digital_twin_armigo/experiments/attribution.py src/digital_twin_armigo/experiments/scoring.py src/digital_twin_armigo/experiments/report.py tests/test_precursors.py
git commit -m "obs H2: mechanism layer (precursors) above first-error symptoms"
```

### Task H3：在 dev 数据上校准前兆阈值

**Files:**
- Create: `scripts/experiments/calibrate_precursors.py`
- Create: `config/experiments/precursor_thresholds.json`（脚本的输出，一起提交）
- Test: `tests/test_precursors.py`（追加）

**Interfaces:**
- Consumes: `precursors.RELEVANT`、`scoring.score_episode`
- Produces:
  - `youden_threshold(pos: list[float], neg: list[float], max_neg_rate: float = 0.10) -> dict`，返回 `{"threshold", "tpr", "fpr", "j", "auc", "n_pos", "n_neg"}`；
  - 命令行：`calibrate_precursors.py --scores <scores.jsonl> --out config/experiments/precursor_thresholds.json`；
  - 输出文件：`{"source": <scores 路径>, "commit": <HEAD>, "metrics": {metric: youden_threshold 的结果}}`。

规则：`brick_xy_mm`、`brick_yaw_deg`、`bowl_xy_mm`、`view_belief_mm` 参与校准；`cam_pos_mm`、`cam_axis_deg` **不校准**，固定为 10 mm / 1°。原因：漂移本身就是实验条件，成功回合里也普遍有，用它做预测阈值没有意义。正例是症状和该指标的机制相关的失败回合，负例是 `root_cause == "none"` 的回合。

- [ ] **Step 1：写失败的测试**

```python
from importlib import util
from pathlib import Path

_spec = util.spec_from_file_location("calibrate_precursors", Path(__file__).resolve().parents[1] / "scripts/experiments/calibrate_precursors.py")
cp = util.module_from_spec(_spec)
_spec.loader.exec_module(cp)


def test_youden_threshold_respects_negative_rate_and_reports_auc():
	r = cp.youden_threshold([20, 22, 25, 30], [1, 2, 3, 2.5, 21], max_neg_rate=0.10)
	assert r["threshold"] == 21 and r["tpr"] == 0.75 and r["fpr"] == 0.0
	assert abs(r["auc"] - 0.95) < 1e-9 and (r["n_pos"], r["n_neg"]) == (4, 5)


def test_youden_threshold_ties_count_half_in_auc():
	assert cp.youden_threshold([5.0], [5.0])["auc"] == 0.5


def test_metric_values_take_first_brick_and_max_bowl():
	events = [{"kind": "brick_perceived", "found": True, "error_xy_mm": 3.0, "error_yaw_deg": 1.0},
		{"kind": "brick_perceived", "found": True, "error_xy_mm": 9.0, "error_yaw_deg": 9.0},
		{"kind": "bowls_perceived", "error_xy_mm": {"green": 2.0, "pink": 7.0}}]  # fmt: skip
	assert cp.metric_values(events) == {"brick_xy_mm": 3.0, "brick_yaw_deg": 1.0, "bowl_xy_mm": 7.0}
```

- [ ] **Step 2：运行，确认失败**（文件不存在）

- [ ] **Step 3：实现**

```python
"""Calibrate precursor thresholds on a dev batch (never on sealed or held-out data).

    PYTHONPATH=src python scripts/experiments/calibrate_precursors.py \
        --scores D:/Armigo_data_v2/reports/p2-dev-r2/scores.jsonl --out config/experiments/precursor_thresholds.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from digital_twin_armigo.experiments.precursors import RELEVANT

CALIBRATED = {"brick_xy_mm": "brick_pose_error", "brick_yaw_deg": "brick_yaw_error", "bowl_xy_mm": "bowl_pose_error",
	"view_belief_mm": "view_belief_error"}  # fmt: skip


def _auc(pos, neg) -> float:
	pairs = [(1.0 if p > n else 0.5 if p == n else 0.0) for p in pos for n in neg]
	return sum(pairs) / len(pairs) if pairs else float("nan")


def youden_threshold(pos, neg, max_neg_rate: float = 0.10) -> dict:
	best = None
	for c in sorted(set(pos) | set(neg)):
		tpr = sum(v > c for v in pos) / len(pos)
		fpr = sum(v > c for v in neg) / len(neg)
		if fpr > max_neg_rate:
			continue
		if best is None or tpr - fpr > best["j"]:
			best = {"threshold": c, "tpr": tpr, "fpr": fpr, "j": tpr - fpr}
	best = best or {"threshold": max([*pos, *neg]), "tpr": 0.0, "fpr": 0.0, "j": 0.0}
	return {**best, "auc": _auc(pos, neg), "n_pos": len(pos), "n_neg": len(neg)}


def metric_values(events: list[dict]) -> dict[str, float]:
	out: dict[str, float] = {}
	for e in events:
		if e.get("kind") == "brick_perceived" and e.get("found") and "brick_xy_mm" not in out:
			out["brick_xy_mm"], out["brick_yaw_deg"] = float(e["error_xy_mm"]), float(e["error_yaw_deg"])
		elif e.get("kind") == "bowls_perceived":
			errs = [v for v in (e.get("error_xy_mm") or {}).values() if v is not None]
			if errs:
				out["bowl_xy_mm"] = max(out.get("bowl_xy_mm", 0.0), float(max(errs)))
		elif e.get("kind") == "view_truth" and e.get("belief_error_mm") is not None:
			out["view_belief_mm"] = max(out.get("view_belief_mm", 0.0), float(e["belief_error_mm"]))
	return out


def _events(path: Path) -> list[dict]:
	f = path / "events.jsonl"
	return [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line] if f.is_file() else []


def main(argv=None) -> int:
	p = argparse.ArgumentParser()
	p.add_argument("--scores", type=Path, required=True)
	p.add_argument("--out", type=Path, required=True)
	p.add_argument("--max-neg-rate", type=float, default=0.10)
	a = p.parse_args(argv)
	rows = [json.loads(line) for line in a.scores.read_text(encoding="utf-8").splitlines() if line]
	pos = {m: [] for m in CALIBRATED}
	neg = {m: [] for m in CALIBRATED}
	for r in rows:
		values = metric_values(_events(Path(r["path"])))
		for metric, mech in CALIBRATED.items():
			if metric not in values:
				continue
			if r["root_cause"] == "none":
				neg[metric].append(values[metric])
			elif mech in RELEVANT.get(r["root_cause"], ()):
				pos[metric].append(values[metric])
	metrics = {m: youden_threshold(pos[m], neg[m], a.max_neg_rate) for m in CALIBRATED if pos[m] and neg[m]}
	commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
	a.out.parent.mkdir(parents=True, exist_ok=True)
	a.out.write_text(json.dumps({"source": str(a.scores), "commit": commit, "metrics": metrics}, indent=2) + "\n")
	for m, r in metrics.items():
		print(f"{m}: threshold {r['threshold']:.2f} auc {r['auc']:.3f} tpr {r['tpr']:.2f} fpr {r['fpr']:.2f} ({r['n_pos']}+/{r['n_neg']}-)")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
```

- [ ] **Step 4：运行测试，确认通过**

- [ ] **Step 5：在 p2-dev-r2 上校准（只读，CPU）**

```bash
PYTHONPATH=src E:/digital_twin_armigo/.venv/Scripts/python scripts/experiments/calibrate_precursors.py --scores D:/Armigo_data_v2/reports/p2-dev-r2/scores.jsonl --out config/experiments/precursor_thresholds.json
```

Expected：`brick_xy_mm` 的 AUC 应明显高于 0.75（第 0 节的中位数是 21.2 对 2.0 mm）。任何指标的 AUC < 0.75 时**不要手调**，写进提交说明，并把它在报告里标为“弱前兆”。

- [ ] **Step 6：用校准后的阈值重跑 H2 Step 7 的核对命令，输出贴进提交说明。**

- [ ] **Step 7：Commit**

```bash
git add scripts/experiments/calibrate_precursors.py config/experiments/precursor_thresholds.json tests/test_precursors.py
git commit -m "obs H3: calibrate precursor thresholds on p2-dev-r2 (dev only)"
```

### Task H4：自动归因进入 inventory 和 viewer，审计可写机制

**Files:**
- Modify: `src/digital_twin_armigo/inventory/scan.py`（新函数 `load_auto_attribution`，在 `scan_agent_episodes` 里合并）
- Modify: `src/digital_twin_armigo/inventory/store.py:17`（`TRAJECTORY_COLUMNS` 末尾加三列）
- Modify: `src/digital_twin_armigo/inventory/query.py:21-55`（`FILTERS`、`FACETS`）
- Modify: `src/digital_twin_armigo/inventory/audit.py`（`mechanism` 列和校验）
- Test: `tests/test_inventory.py`（追加）

**Interfaces:**
- Produces:
  - `scan.AUTO_COLUMNS = ("auto_root_cause", "auto_mechanism", "auto_upstream")`；
  - `scan.load_auto_attribution(root: Path) -> dict[str, dict]`：键是 `str(Path(path).resolve()).lower()`；
  - 审计正文可选 `mechanism`（取 `MECHANISMS` 之一或 null），存进 `reviews.mechanism`。

- [ ] **Step 1：写失败的测试**

```python
def test_auto_attribution_is_joined_from_reports(tmp_path):
	from digital_twin_armigo.inventory import query, scan, store

	runs = tmp_path / "runs"
	ep = runs / "b1" / "mujoco" / "g" / "L7-N"
	ep.mkdir(parents=True)
	rep = tmp_path / "reports" / "b1"
	rep.mkdir(parents=True)
	row = {"path": str(ep), "root_cause": "recovery_failure", "mechanism": "brick_pose_error", "upstream": "calibration_drift"}
	(rep / "scores.jsonl").write_text(json.dumps(row) + "\n")
	auto = scan.load_auto_attribution(runs)
	assert auto[str(ep.resolve()).lower()] == {"auto_root_cause": "recovery_failure", "auto_mechanism": "brick_pose_error",
		"auto_upstream": "calibration_drift"}  # fmt: skip
	assert set(scan.AUTO_COLUMNS) <= set(store.TRAJECTORY_COLUMNS)
	assert set(scan.AUTO_COLUMNS) <= set(query.FILTERS["trajectories"]) and set(scan.AUTO_COLUMNS) <= set(query.FACETS["trajectories"])


def test_audit_mechanism_is_optional_and_validated(tmp_path):
	from digital_twin_armigo.inventory import audit

	db = tmp_path / "reviews.sqlite"
	audit.append(db, "t1", {"reviewer": "x", "state": "confirmed", "root_cause": "recovery_failure",
		"mechanism": "brick_pose_error"}, expected_latest_id=None, code_version=None)  # fmt: skip
	assert audit.latest(db, "t1")["mechanism"] == "brick_pose_error"
	with pytest.raises(audit.ValidationError):
		audit.append(db, "t2", {"reviewer": "x", "state": "confirmed", "mechanism": "bogus"}, expected_latest_id=None, code_version=None)
	assert audit.append(db, "t3", {"reviewer": "x", "state": "confirmed"}, expected_latest_id=None, code_version=None)["mechanism"] is None
```

（`tests/test_inventory.py` 顶部没有 `json` 或 `pytest` 的导入时补上。）另外复制该文件里现有的一个“扫描单个 agent 回合”的测试，复制品里在 `<tmp>/reports/<batch>/scores.jsonl` 写一行指向那个回合，断言扫描结果的 `auto_root_cause` 被填上。

- [ ] **Step 2：运行，确认失败**

- [ ] **Step 3：实现。**
  - `scan.py`：

```python
AUTO_COLUMNS = ("auto_root_cause", "auto_mechanism", "auto_upstream")


def load_auto_attribution(root: Path) -> dict[str, dict]:
	"""Rows of ``<root>/../reports/*/scores.jsonl`` (v2 layout) by resolved episode path; newer files win."""
	reports = Path(root).resolve().parent / "reports"
	out: dict[str, dict] = {}
	if not reports.is_dir():
		return out
	for f in sorted(reports.glob("*/scores.jsonl"), key=lambda p: p.stat().st_mtime):
		for line in f.read_text(encoding="utf-8").splitlines():
			if not line.strip():
				continue
			r = json.loads(line)
			out[str(Path(r["path"]).resolve()).lower()] = {
				"auto_root_cause": r.get("root_cause"),
				"auto_mechanism": r.get("mechanism"),
				"auto_upstream": r.get("upstream"),
			}
	return out
```

    在 `scan_agent_episodes(root, label)` 的循环前调用一次 `auto = load_auto_attribution(root)`；每产出一行之前执行 `row |= auto.get(str(directory.resolve()).lower(), dict.fromkeys(AUTO_COLUMNS))`（`directory` 是该行的回合目录，变量名以函数里的实际名字为准）。
  - `store.TRAJECTORY_COLUMNS` 末尾加三列；`query.FILTERS["trajectories"]` 和 `query.FACETS["trajectories"]` 末尾各加三列。`viewer/inventory.html` 的分面是按 API 返回值通用渲染的（`#facets`），不需要改；扫描后在 8767 上肉眼确认三组分面出现。
  - `audit.py`：`SCHEMA` 的建表语句里加 `mechanism text`；在 `connect` 里 `executescript(SCHEMA)` 之后，对旧库执行 `alter table reviews add column mechanism text`，捕获 `sqlite3.OperationalError`（列已存在）后忽略；`validate` 的返回加 `"mechanism": _enum(body.get("mechanism"), MECHANISMS, "mechanism", nullable=True)`（从 `experiments.precursors` 导入 `MECHANISMS`）；`append` 的 insert 语句加上这一列；`enums()` 的返回加 `"mechanism": list(MECHANISMS)`。
- [ ] **Step 4：运行 inventory 全部测试和全量测试**

Run: `E:/digital_twin_armigo/.venv/Scripts/python -m pytest tests/test_inventory.py tests/test_inventory_view.py tests/test_inventory_rerun.py -v`

- [ ] **Step 5：Commit**

```bash
git add src/digital_twin_armigo/inventory/ tests/test_inventory.py
git commit -m "obs H4: auto attribution columns in inventory; optional mechanism in audit verdicts"
```

### Task H5：MCP 工具调用记录

**Files:**
- Modify: `src/digital_twin_armigo/agent/step_mcp.py`（`StepService.call`、`StepSession.log_tool`）
- Test: `tests/test_step_mcp.py`（追加）

**Interfaces:**
- Produces: 每次会话工具调用写一条事件 `{"kind": "tool_call", "tool": str, "args": dict, "latency_s": float, "ok": bool, "error": str | None, "returned": dict | None}`。图像类返回值记成 `{"n_images": int}`。

- [ ] **Step 1：先读代码。** 读 `StepService.call`（约 410 行）和 `_scripted`（约 584 行），确认脚本客户端是否经过 `call`。如果 `_scripted` 直接调用 `StepSession` 的方法，把这些调用改成经过 `service.call`（行为不变，只是多一层），这样记录对 MCP 和脚本两条路径都生效。

- [ ] **Step 2：写失败的测试**

```python
def test_tool_calls_are_recorded_in_order_and_sensor_only(tmp_path):
	service = _service(tmp_path, "realistic")
	sm.run_scripted(service)
	events_files = list((tmp_path / "realistic").rglob("events.jsonl"))
	assert events_files
	events = [json.loads(line) for line in events_files[0].read_text().splitlines()]
	calls = [e for e in events if e["kind"] == "tool_call"]
	assert calls, "no tool_call events"
	assert all(isinstance(e["latency_s"], float) and e["latency_s"] >= 0 for e in calls)
	assert [e["seq"] for e in calls] == sorted(e["seq"] for e in calls)
	assert {"scan_wrist", "place"} <= {e["tool"] for e in calls}
	for e in calls:
		if e["returned"] is not None:
			sm.assert_sensor_only(e["returned"])
```

- [ ] **Step 3：运行，确认失败**

- [ ] **Step 4：实现。**

```python
# StepSession
	def log_tool(self, tool: str, latency_s: float, *, args=None, returned=None, error: str | None = None) -> None:
		if isinstance(returned, (list, tuple)):  # image content blocks
			returned = {"n_images": len(returned)}
		self.backend.event(
			"tool_call",
			tool=tool,
			args=dict(args or {}),
			latency_s=round(float(latency_s), 4),
			ok=error is None,
			error=error,
			returned=returned,
		)
```

```python
# StepService
	async def call(self, fn, *args):
		session = getattr(fn, "__self__", None)
		t0 = time.perf_counter()
		try:
			out = await asyncio.wrap_future(self.sim.submit(fn, *args))
		except Exception as exc:
			if isinstance(session, StepSession):
				session.log_tool(fn.__name__, time.perf_counter() - t0, args={"args": list(args)}, error=f"{type(exc).__name__}: {exc}")
			raise
		if isinstance(session, StepSession):
			session.log_tool(fn.__name__, time.perf_counter() - t0, args={"args": list(args)}, returned=out)
		return out
```

  （`time` 在文件顶部没有导入就补上。记录发生在 `_out` 之前，所以 `returned` 是未加 `session_state` 的原始返回值；realistic 下仍然要过测试里的 `assert_sensor_only`。如果某个原始返回值没过，说明 `_out` 之前就带了真值，按泄漏处理：修返回值，不修测试。）

- [ ] **Step 5：运行 `tests/test_step_mcp.py` 和全量测试**

- [ ] **Step 6：Commit**

```bash
git add src/digital_twin_armigo/agent/step_mcp.py tests/test_step_mcp.py
git commit -m "obs H5: record every step-MCP tool call (args, latency, result) as a tool_call event"
```

### Task H6：oracle 替换与反事实复跑

**Files:**
- Modify: `src/digital_twin_armigo/agent/twin_mujoco.py`（`true_world_from_cam`；`oracle` 选项在标定、`_perceive`、`_sense_bowls` 三处生效；`student_eligible` 加条件）
- Modify: `src/digital_twin_armigo/experiments/contract.py`（`EXPERIMENT_FIELDS` 加 `"oracle"`；`ORACLE_PARTS = ("camera", "brick", "bowls")`）
- Modify: `src/digital_twin_armigo/experiments/runner.py`（`--oracle`，组名后缀）
- Create: `src/digital_twin_armigo/experiments/counterfactual.py`、`scripts/experiments/counterfactual.py`
- Test: `tests/test_counterfactual.py`

**Interfaces:**
- Produces:
  - `true_world_from_cam(top) -> np.ndarray`（4×4，OpenCV 轴向）；
  - 案例选项 `{"oracle": ["camera" | "brick" | "bowls", ...]}`，只在 realistic 下生效；生效时写事件 `oracle_applied {"parts": [...]}`；
  - `runner.group_name(sensing, camera, strategy, detector, oracle=())`：oracle 非空时组名末尾加 `-oracle-<parts 用 + 连接>`；
  - `counterfactual.VARIANTS = ((), ("camera",), ("brick",), ("bowls",), ("camera", "brick", "bowls"))`；
  - `counterfactual.verdict(original_success: bool, variants: dict[tuple, bool | None]) -> dict`，返回 `{"verdict": "repro_mismatch" | "confirmed" | "joint" | "not_perception" | "incomplete", "minimal": [list[str], ...]}`。

- [ ] **Step 1：写失败的测试（纯逻辑部分）**

```python
# tests/test_counterfactual.py
from digital_twin_armigo.experiments.counterfactual import verdict
from digital_twin_armigo.experiments.runner import group_name

ALL = ("camera", "brick", "bowls")


def test_repro_mismatch_wins():
	v = verdict(False, {(): True, ("camera",): True, ("brick",): False, ("bowls",): False, ALL: True})
	assert v["verdict"] == "repro_mismatch"


def test_confirmed_minimal_sets():
	v = verdict(False, {(): False, ("camera",): True, ("brick",): True, ("bowls",): False, ALL: True})
	assert v == {"verdict": "confirmed", "minimal": [["brick"], ["camera"]]}


def test_joint_and_not_perception_and_incomplete():
	assert verdict(False, {(): False, ("camera",): False, ("brick",): False, ("bowls",): False, ALL: True})["verdict"] == "joint"
	assert verdict(False, {(): False, ("camera",): False, ("brick",): False, ("bowls",): False, ALL: False})["verdict"] == "not_perception"
	assert verdict(False, {(): False, ("camera",): None, ("brick",): False, ("bowls",): False, ALL: True})["verdict"] == "incomplete"


def test_group_name_oracle_suffix():
	assert group_name("realistic", "drift", "A", "v7bot") == "realistic-drift-A-v7bot"
	assert group_name("realistic", "drift", "A", "v7bot", oracle=("brick", "camera")) == "realistic-drift-A-v7bot-oracle-brick+camera"
```

- [ ] **Step 2：写失败的测试（MuJoCo 部分，CPU，标 `@pytest.mark.slow`）**

```python
import numpy as np
import pytest


@pytest.mark.slow
def test_true_world_from_cam_matches_hand_eye_on_the_nominal_rig():
	pytest.importorskip("mujoco")
	from digital_twin_armigo.agent.twin_mujoco import sample_case, true_world_from_cam
	from digital_twin_armigo.mujoco_cell.handeye import calibrate_layout  # 以实际所在模块为准

	lay = sample_case(600001, False, 5)
	_, est, _ = calibrate_layout(lay, 0, None)
	truth = true_world_from_cam(lay.top)
	assert np.linalg.norm(est[:3, 3] - truth[:3, 3]) < 0.001
	for col in range(3):
		cos = float(np.dot(est[:3, col], truth[:3, col]))
		assert np.degrees(np.arccos(np.clip(cos, -1, 1))) < 0.5
```

  （先用 `grep -rn "def calibrate_layout" src/` 确认它在哪个模块、第三个参数的含义；`twin_mujoco.py` 约 752 行调用它时传的是 `self.perception.calibration`，测试里传同样的值。）

  再写一个 oracle 生效的冒烟测试：用 `tests/test_step_mcp.py` 的 `FakeScorer` 和 `StepService`，profile 取 `"realistic"`，案例为 `(600001, False, 5, {"cameras": "drift", "oracle": ["camera", "bowls"]})`；开会话后断言 `backend._cam_est` 与 `true_world_from_cam(backend.layout.top)` 相等（`np.allclose`），每只碗的 `backend.bowls[name]` 等于真值，并且事件里有一条 `oracle_applied`。如果 `StepService` 只接受三元组案例，先改它，让四元组的第四项原样传给 `sample_case`（兼容旧的三元组），这一改动和测试一起提交。

- [ ] **Step 3：运行，确认失败**

- [ ] **Step 4：实现 `twin_mujoco.py`。**

```python
def true_world_from_cam(top) -> np.ndarray:
	"""World-from-camera (OpenCV axes: x right, y down, z forward) of the top camera as mounted: the oracle that the
	hand-eye estimate approximates. Same axes as the ``actual_cam_*`` diagnostics in ``camera_calibrated``."""
	right, up = top.axes()
	forward = -np.cross(right, up)
	m = np.eye(4)
	m[:3, 0] = right / np.linalg.norm(right)
	m[:3, 1] = -up / np.linalg.norm(up)
	m[:3, 2] = forward / np.linalg.norm(forward)
	m[:3, 3] = (top.x, top.y, top.height)
	return m
```

  - `self.oracle = frozenset(options.get("oracle") or ()) if self.realistic else frozenset()`，在读取案例选项的地方设置。值不在 `ORACLE_PARTS` 里时抛 `ValueError`。
  - 标定之后（约 752 行，`self.event("camera_calibrated", ...)` 之后）：`if "camera" in self.oracle: self._cam_est = true_world_from_cam(layout.top)`。
  - `_sense_bowls`（约 800 行）：`"bowls" in self.oracle` 时，对每个槽位用 `self._true_bowl_xy(name)`，并且不设 `BOWL_NOT_FOUND`。照常写 `bowls_perceived` 事件，此时误差为 0。
  - `_perceive`（约 880 行）：`"brick" in self.oracle` 时，把 `est` 的位置、yaw 和尺寸换成真值（`self.cell.brick_pos()`、上面已经算好的 `true_yaw`、`self.layout.brick_size`）。如果 `est` 是 dataclass，用 `dataclasses.replace`；不是的话，按它的构造函数新建一个。
  - 第一次生效时写 `self.event("oracle_applied", parts=sorted(self.oracle))`。
  - 约 2383 行的 `student_eligible` 加 `and not self.oracle`。`twin_record.py:1204` 也加同样的条件（如果它能读到 provenance 里的 oracle），保证 oracle 回合进不了任何学生数据。
- [ ] **Step 5：实现 runner 和 contract。** `EXPERIMENT_FIELDS` 加 `"oracle"`；`runner._common` 加 `p.add_argument("--oracle", default="", help="comma list of camera,brick,bowls (evaluation only)")`；`group_name` 加参数 `oracle=()`，按测试实现后缀，parts 排序；`plan_group` 把 oracle 写进每个作业的案例选项和 provenance 的 experiment 字段。oracle 非空且 sensing 为 `truth` 时报错退出。
- [ ] **Step 6：实现 `counterfactual.py`。**

```python
"""Counterfactual oracle re-runs: which belief, replaced by truth, turns a failed realistic episode into a success."""

from __future__ import annotations

VARIANTS = ((), ("camera",), ("brick",), ("bowls",), ("camera", "brick", "bowls"))


def verdict(original_success: bool, variants: dict[tuple, bool | None]) -> dict:
	if any(variants.get(v) is None for v in VARIANTS):
		return {"verdict": "incomplete", "minimal": []}
	if variants[()] != original_success:
		return {"verdict": "repro_mismatch", "minimal": []}
	singles = sorted([list(v) for v in VARIANTS if len(v) == 1 and variants[v]])
	if singles:
		return {"verdict": "confirmed", "minimal": singles}
	if variants[VARIANTS[-1]]:
		return {"verdict": "joint", "minimal": [list(VARIANTS[-1])]}
	return {"verdict": "not_perception", "minimal": []}
```

  `scripts/experiments/counterfactual.py` 有两个子命令：
  - `plan --scores <scores.jsonl> --out config/agent_bench/counterfactual/<batch>.json`：选 realistic 且 `physical_success` 为假的案例，写成 v2 套件格式。格式照抄 `D:/Armigo_data_v2/suites/` 里任意一个套件文件；manifest 用 `scripts/experiments/make_suites.py` 的写入函数登记。
  - `summarize --base <scores.jsonl> --variants <每个变体批次的 scores.jsonl，共 5 个> --out <counterfactual.jsonl>`：按 `case_id` 配对，调用 `verdict`，每个案例写一行，含 `case_id`、`symptom`、`mechanism`、`verdict`、`minimal`，并打印“机制假设 vs 反事实结论”的交叉表。
- [ ] **Step 7：运行测试和全量测试，提交**

```bash
git add src/digital_twin_armigo/agent/twin_mujoco.py src/digital_twin_armigo/agent/step_mcp.py src/digital_twin_armigo/experiments/ scripts/experiments/counterfactual.py tests/test_counterfactual.py
git commit -m "obs H6: oracle substitution for realistic beliefs and counterfactual verdicts"
```

- [ ] **Step 8【GPU】：在 p2-dev-r2 的失败案例上跑反事实。** 用 runner 跑 5 个变体批次 `p2-dev-r2-cf-{none,camera,brick,bowls,all}`，配置和 p2-dev-r2 相同（同一检测器、相机条件、策略），再跑 `summarize`。结果写进 `E:/Armigo_data_v3/loop/round-00/counterfactual.md`：drift 下的 `recovery_failure` 有多少是 `confirmed:[camera]` 或 `[brick]`；`evidence_insufficient` 有多少是 `confirmed:[brick]`；`repro_mismatch` 的比例。这一步的结论决定第 3.4 节的独立计划要不要做。

### Task H7：诊断基准与每轮指标

**Files:**
- Create: `src/digital_twin_armigo/experiments/diagnosis.py`、`scripts/experiments/make_diagnosis_bench.py`、`scripts/experiments/round_metrics.py`
- Test: `tests/test_diagnosis.py`

**Interfaces:**
- Produces:
  - `diagnosis.split_of(traj_id: str) -> str`：`sha256(traj_id)` 第一个字节为偶数取 `"calib"`，奇数取 `"heldout"`（确定、与时间无关）；
  - `diagnosis.cohen_kappa(a: list[str], b: list[str]) -> float`；
  - `diagnosis.agreement(bench: list[dict], auto: dict[str, dict], split: str) -> dict`：返回 `{"n", "root_cause_acc", "root_cause_kappa", "mechanism_acc", "mechanism_n"}`；`mechanism` 只在人工给了机制的条目上计算；
  - `make_diagnosis_bench.py --audit <reviews.sqlite> --inventory <inventory dir> --out config/agent_bench/diagnosis_bench/v<N>.json`：取每条轨迹最新的裁定，只要 `state == "confirmed"` 且 `root_cause` 非空的；条目为 `{"traj_id", "path", "root_cause", "mechanism", "split"}`；排除批次名含 `sealed` 的轨迹；
  - `round_metrics.py --round-dir E:/Armigo_data_v3/loop/round-NN --bench <bench.json> --scores <本轮 scores.jsonl> [--verify-dir ...] [--counterfactual ...] [--diag-log ...]`：写 `round_metrics.json`，并在 `E:/Armigo_data_v3/loop/rounds.md` 里追加一行（第 2.3 节的 6 类指标）。

- [ ] **Step 1：写失败的测试**

```python
# tests/test_diagnosis.py
from digital_twin_armigo.experiments.diagnosis import agreement, cohen_kappa, split_of


def test_split_is_deterministic_and_roughly_balanced():
	ids = [f"traj-{i}" for i in range(1000)]
	splits = [split_of(t) for t in ids]
	assert splits == [split_of(t) for t in ids]
	assert 400 < splits.count("calib") < 600


def test_cohen_kappa_perfect_and_chance():
	assert cohen_kappa(["a", "b", "a"], ["a", "b", "a"]) == 1.0
	assert abs(cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"])) < 1e-9


def test_agreement_uses_only_the_requested_split_and_labelled_mechanisms():
	bench = [{"traj_id": "x", "root_cause": "recovery_failure", "mechanism": "brick_pose_error", "split": "heldout"},
		{"traj_id": "y", "root_cause": "evidence_insufficient", "mechanism": None, "split": "heldout"},
		{"traj_id": "z", "root_cause": "occlusion", "mechanism": None, "split": "calib"}]  # fmt: skip
	auto = {"x": {"root_cause": "recovery_failure", "mechanism": "brick_pose_error"},
		"y": {"root_cause": "premature_stop", "mechanism": "brick_yaw_error"}, "z": {"root_cause": "occlusion", "mechanism": None}}  # fmt: skip
	r = agreement(bench, auto, "heldout")
	assert r["n"] == 2 and r["root_cause_acc"] == 0.5 and r["mechanism_n"] == 1 and r["mechanism_acc"] == 1.0
```

- [ ] **Step 2：运行，确认失败**

- [ ] **Step 3：实现 `diagnosis.py`**

```python
"""Diagnosis benchmark: human-confirmed failures, split by a hash of the trajectory id into calib / heldout.

Rules and thresholds may only be tuned on ``calib``; ``heldout`` is evaluated once per round (plan section 2.3)."""

from __future__ import annotations

import hashlib
from collections import Counter


def split_of(traj_id: str) -> str:
	return "calib" if hashlib.sha256(traj_id.encode()).digest()[0] % 2 == 0 else "heldout"


def cohen_kappa(a: list[str], b: list[str]) -> float:
	n = len(a)
	if n == 0:
		return float("nan")
	po = sum(x == y for x, y in zip(a, b, strict=True)) / n
	ca, cb = Counter(a), Counter(b)
	pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
	return 1.0 if pe == 1.0 else (po - pe) / (1.0 - pe)


def agreement(bench: list[dict], auto: dict[str, dict], split: str) -> dict:
	items = [b for b in bench if b["split"] == split and b["traj_id"] in auto]
	human = [b["root_cause"] for b in items]
	machine = [auto[b["traj_id"]].get("root_cause") or "other" for b in items]
	mech = [b for b in items if b.get("mechanism")]
	mech_hits = sum(auto[b["traj_id"]].get("mechanism") == b["mechanism"] for b in mech)
	return {
		"n": len(items),
		"root_cause_acc": sum(h == m for h, m in zip(human, machine, strict=True)) / len(items) if items else float("nan"),
		"root_cause_kappa": cohen_kappa(human, machine),
		"mechanism_acc": mech_hits / len(mech) if mech else float("nan"),
		"mechanism_n": len(mech),
	}
```

- [ ] **Step 4：实现两个脚本。** `make_diagnosis_bench.py` 用 `audit.connect` 执行 `select traj_id from reviews group by traj_id`，再对每个 id 调 `audit.latest`；路径用 `inventory.query.agent_episode_path(inventory_dir, traj_id)`；批次从路径里的 `runs/<batch>/` 段读。`round_metrics.py` 读入：本轮 scores（用 `scoring` 字段算归因覆盖率）、基准加本轮自动归因（`auto` 按 `traj_id` 映射，从 inventory 的 `auto_*` 列取）、`verify-*.json`（F4 的判定计数和 n10）、`counterfactual.jsonl`（证实比例）、诊断日志（H8，没有时这一项写 null）。两个脚本的测试各用一个最小临时目录夹具：断言输出 JSON 的键齐全、数值和手算一致。
- [ ] **Step 5：在现有审计库上建 v1 基准（只读，CPU）。** 把条目数和两半的数目写进提交说明。如果 confirmed 的条目少于 40，在报告里写明，按第 2.3 节规定：先补人工审计，再用这个基准下结论。
- [ ] **Step 6：运行测试和全量测试，提交**

```bash
git add src/digital_twin_armigo/experiments/diagnosis.py scripts/experiments/make_diagnosis_bench.py scripts/experiments/round_metrics.py config/agent_bench/diagnosis_bench/ tests/test_diagnosis.py
git commit -m "obs H7: diagnosis benchmark (calib/heldout) and per-round loop metrics"
```

### Task H8：给改工具的人或 agent 用的诊断 MCP

这一步把闭环接上：负责诊断和修工具的 agent（Claude 或人）通过这里读台账、看回合、比较两次运行。每次调用都会记账，诊断成本因此可以度量。诊断工具本身被改进时，在 H7 的留出基准上做新旧 A/B，这就是第 2.3 节 L2 的证据。

**Files:**
- Create: `src/digital_twin_armigo/experiments/diag_mcp.py`
- Test: `tests/test_diag_mcp.py`

**Interfaces:**
- Produces（纯函数，FastMCP 只是一层薄包装；端口默认 8770，不和 8766、8767 冲突）：
  - `episode_summary(path: Path) -> dict`：症状、机制、上游、前兆、按 seq 排列的时间线（`node`、`failure`、`tool_call`、`*_perceived` 事件压缩成一行一条）、物理结局；
  - `compare_runs(a: Path, b: Path, *, tol: float = 1e-6) -> dict`：按 `(kind, node, branch)` 对齐两个事件序列，忽略 `wall_time_s`、`latency_s` 和检测器 `score` 的数值差（分数差超过 `tol` 的单独列进 `score_deltas`），返回 `{"identical_discrete": bool, "first_divergence": {"index", "a", "b"} | None, "score_deltas": [...], "verdict_equal": bool}`；
  - `ledger_clusters(round_dir: Path) -> list[dict]`、`counterfactual_result(round_dir: Path, case_id: str) -> dict | None`：直接读 F2 和 H6 的文件；
  - 每次工具调用往 `<round_dir>/diag_calls.jsonl` 追加 `{"ts", "session", "tool", "args", "latency_s"}`；`session` 由调用方传入（一个诊断任务一个 id）；
  - 守护：`diag_mcp` 的工具名和 `step_mcp.TOOLS_COMMON`、`step_mcp.TOOLS_ONLY` 没有交集；`step_mcp.py` 不导入 `diag_mcp`（测试用 `ast` 检查）。诊断 MCP 可以看到真值，所以绝不能接给任务 agent。

**F3 复现判定口径（G7）：** 复现库的“两次运行一致”等价于 `compare_runs(...)["identical_discrete"] and verdict_equal`；`score_deltas` 非空时记录进复现库条目，但不剔除。MuJoCo 状态轨迹的逐位比较（F3 原文的要求）照旧做，作为额外字段写进条目。

- [ ] **Step 1：写失败的测试**

```python
# tests/test_diag_mcp.py
import ast
import json
from pathlib import Path

from digital_twin_armigo.experiments import diag_mcp


def _ep(d: Path, events, verdict="SUCCESS"):
	d.mkdir(parents=True)
	(d / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
	(d / "episode.json").write_text(json.dumps({"result": {"verdict": verdict, "label": "normal"}}))
	return d


BASE = [{"seq": 0, "kind": "node", "node": "scan_wrist", "branch": 0, "wall_time_s": 1.0},
	{"seq": 1, "kind": "view", "node": "scan_wrist", "branch": 0, "observation": {"score": -0.5}, "wall_time_s": 2.0},
	{"seq": 2, "kind": "node", "node": "place", "branch": 0, "wall_time_s": 3.0}]  # fmt: skip


def test_compare_ignores_wall_time_and_lists_score_deltas(tmp_path):
	other = [dict(e, wall_time_s=e["wall_time_s"] + 9) for e in BASE]
	other[1] = {**other[1], "observation": {"score": -0.5 + 1e-3}}
	r = diag_mcp.compare_runs(_ep(tmp_path / "a", BASE), _ep(tmp_path / "b", other))
	assert r["identical_discrete"] and r["verdict_equal"] and r["first_divergence"] is None
	assert len(r["score_deltas"]) == 1


def test_compare_finds_first_divergence(tmp_path):
	other = BASE[:2] + [{"seq": 2, "kind": "failure", "node": "scan_wrist", "branch": 0, "code": "no_clamp"}]
	r = diag_mcp.compare_runs(_ep(tmp_path / "a", BASE), _ep(tmp_path / "b", other, verdict="FAILED"))
	assert not r["identical_discrete"] and r["first_divergence"]["index"] == 2 and not r["verdict_equal"]


def test_diag_tools_are_disjoint_from_task_tools_and_never_imported_by_step_mcp():
	from digital_twin_armigo.agent import step_mcp as sm

	task_tools = set(sm.TOOLS_COMMON) | {t for v in sm.TOOLS_ONLY.values() for t in v}
	assert not set(diag_mcp.TOOLS) & task_tools
	tree = ast.parse(Path(sm.__file__).read_text(encoding="utf-8"))
	imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom | ast.Import) for a in n.names}
	imported |= {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
	assert not any("diag_mcp" in name for name in imported)


def test_calls_are_logged_per_session(tmp_path):
	ep = _ep(tmp_path / "a", BASE)
	diag_mcp.logged(tmp_path, "s1", "episode_summary", {"path": str(ep)}, lambda: diag_mcp.episode_summary(ep))
	rows = [json.loads(line) for line in (tmp_path / "diag_calls.jsonl").read_text().splitlines()]
	assert rows[0]["session"] == "s1" and rows[0]["tool"] == "episode_summary" and rows[0]["latency_s"] >= 0
```

  （测试导入 `step_mcp` 需要 mujoco；顶部加 `pytest.importorskip("mujoco")`。如果 `TOOLS_ONLY` 的实际结构不是 `dict[str, list]`，按实际结构改这一行，并写进报告。）

- [ ] **Step 2：运行，确认失败**

- [ ] **Step 3：实现。** `episode_summary` 调用 `scoring.score_episode(path)`，取 `first_error` 的各字段，再拼时间线：每个事件一行，格式为 `f"{seq} {kind} {node}"`，后面跟上关键字段（`code`、`tool`、`error_xy_mm`、`error_yaw_deg`、`score`），总长度上限 200 行，超出时保留首尾各 100 行。`compare_runs` 的对齐：按位置逐个比较两个事件序列；第 i 个事件先比 `(kind, node, branch)`，再比去掉忽略键（`wall_time_s`、`latency_s`、`seq`）之后的其余字段，浮点用 `tol`；`observation.score` 的差不算离散差异，只进 `score_deltas`。长度不同时，较短序列结束的位置就是第一处分歧。`logged(round_dir, session, tool, args, fn)` 负责计时、追加日志、返回 `fn()`。`TOOLS = ("episode_summary", "compare_runs", "ledger_clusters", "counterfactual_result")`。`main()` 用 FastMCP 注册这四个工具，每个都带 `session: str` 参数，调用都经过 `logged`。
- [ ] **Step 4：运行测试和全量测试，提交**

```bash
git add src/digital_twin_armigo/experiments/diag_mcp.py tests/test_diag_mcp.py
git commit -m "obs H8: read-only diagnosis MCP with call accounting and run comparison"
```

---

## 3. 运行协议（每轮）

### 3.1 一轮的顺序

1. 采集（F 计划）→ `report.py` 出 scores（这时已经带机制列）→ `armigo-inventory scan`（带 `auto_*` 列）。
2. `failure_ledger.py`（F2）：簇键沿用 F2 的 `[stage, code, root_cause]`，**另加** `mechanism` 作为第四维，用来拆分簇。
3. 每个簇抽 5 例人工审计，裁定里填 `mechanism`（H4）。
4. 选影响最大的簇，写判据（F 计划第 2 节的三道关）。机制只是前兆的簇，先做反事实（H6 Step 8 的流程）；`confirmed` 之后再修。修复方向直接由 `minimal` 集合决定：`[camera]` 就修标定或漂移处理，`[brick]` 就修积木定位。
5. 复现库（F3，按 H8 的口径判定）→ 修复 → `verify_fix.py`（F4）。
6. `round_metrics.py` → `rounds.md` 追加一行。
7. 诊断工具（`diag_mcp`、`precursors`、阈值）在本轮有改动的，先在留出基准上做新旧 A/B（同一批诊断任务，新旧工具各跑一遍，记成本和准确率），再合并。

### 3.2 种子段

闭环每轮的新种子段：`3_600_001 + 1000 × (NN − 1)` 起，取 1000 个。它不和 v3 的 train/val/ood/test 段（3000001–、3100001–、3200001–、3900001–）重叠，用 `make_suites.py` 的 manifest 登记。

### 3.3 停止与冻结

沿用失败闭环计划第 3 节，另加一条：**最大的簇在反事实中被判为 `not_perception` 并且需要新工具时**，转成独立计划，不在闭环里硬修。

### 3.4 转独立计划的候选（本计划不做）

- **在线漂移检测**（G5）：p2-dev-r2 里漂移是最大的失败来源，现有工具返回值看不出漂移，标定时的重投影误差对它无效。候选信号是传感器层面的一致性残差：例如腕相机在预抓取位拍到的积木位置，与顶视估计投影过去的位置之差。是否立项由 H6 Step 8 决定：如果 drift 的 `recovery_failure` 大多是 `confirmed:[camera]`，就立项。**更新**：已并入本文第二部分（C1 至 C3 的 `innovation_xy_mm`，以及腕部闭环）；H6 Step 8 的结果改为用来解释 C7。
- **L3 无人工把关的自动修复**：前提是 L2 连续两轮成立。

## 4. 验收清单

- [ ] H1：守护测试能抓到新写入但未登记的真值字段（例如临时在 `twin_mujoco` 的某个 `event` 调用里加一个 `true_foo=1`，测试应失败，验证后删除）。
- [ ] H2：p2-dev-r2 重新评分后，`root_cause` 的计数和第 0 节完全相同；drift 下的 `recovery_failure` 带机制和 `upstream`。
- [ ] H3：阈值文件记录了来源批次和提交号；AUC 低于 0.75 的指标在报告里标明。
- [ ] H4：8767 上出现 `auto_root_cause`、`auto_mechanism`、`auto_upstream` 三组分面；旧审计库打开后自动加列，旧裁定不变。
- [ ] H5：脚本客户端跑完的每个回合都有按顺序排列的 `tool_call` 事件，realistic 下的 `returned` 全部通过 `assert_sensor_only`。
- [ ] H6：oracle 回合 `student_eligible` 为假；`true_world_from_cam` 和手眼标定在标称机位上误差小于 1 mm / 0.5°；【GPU】反事实结果写入 `round-00/counterfactual.md`。
- [ ] H7：基准两半的条目数确定、可重算；`rounds.md` 有第 0 轮的基线一行。
- [ ] H8：`compare_runs` 用作 F3 的复现判定；诊断调用日志能按会话算出每例成本。
- [ ] 全量测试、`ruff check`、`ruff format --check` 通过；每个任务一个提交；没有推送。

---

# 第二部分：抓取与取证动作改为 CV 闭环脚本（C1–C7）

> 第一部分（H1–H8）全部验收后再开始。仍在 `feat/failure-observability` 分支和 `E:/digital_twin_armigo-obs` worktree 上做，提交信息第一行写 `obs C<N>: <内容>`。第一部分的 Global Constraints 全部继续有效。

## 5. 为什么改、改什么

**现状（2026-11-24 核实）**

- 判定之前的机器人动作现在本来就是脚本：`TwinMujoco._grasp` 调专家的 `_approach_and_descend`，`_wrist` 和 `_bottom` 负责取证视角。实验里策略 A、B 不调用 LLM（runner 里 `use_llm=False`），策略 C 只在复核时让 Qwen 挑视角。p2-dev-r2 在 fixed、recal 两种相机条件下 A=B=C=24/24，C 和 B 的不一致只有 1/72。ACT 不在这条路径上。
- 问题在于**开环**：`physics_expert.py:592` 的 `_approach_and_descend` 只在开始时按顶相机估计规划一次（hover → pre → target → servo），下降过程中不再看。drift 条件下约 50% 失败，失败回合的积木定位误差中位数 21 mm，成功回合是 2 mm。
- 腕相机的位姿由关节读数加**设计安装位**算出，和顶相机外参无关。所以顶相机标定过期（drift）时，腕相机重新定位不受影响。这是闭环能修正 drift 的物理依据。

**新发现（列入验收附加项）**

- **G9：realistic 的检测裁剪用了真实顶相机。** `_capture` 在 `_belief()` 里调 `detector_crop`。`_brick_bbox`（`inspection_views.py:60`）会把*信念中的*积木渲染成分割图，渲染用的是模型里的相机。`_belief()` 只把腕相机换成了设计安装位，顶相机仍是真实位姿。结果是：顶相机视图的裁剪框和 `faces_px` 都是"用真实相机投影信念积木"，真机做不到。drift 下仿真因此偏乐观；同时裁剪框跟着信念走，不跟着图像里真实的积木走，信念偏了，裁剪也跟着偏。
- **G10：运动学模型带着真实腕相机安装位。** `Kinematics(cell)` 和仿真共用 `cell.model`，而 `cell.py:204` 把真实安装位（设计值加 `wrist_cam_offset`）写进了 `model.cam_pos`。任何感知代码直接读 `kin.data.cam_xpos` 都会拿到特权信息。必须在 `cell.nominal_wrist_cam(kin.data)` 里计算。

**改动**

1. **抓取闭环（"缩小取证间距"）**：在 hover 和 pre 之间设 K 个检查点，每到一个检查点就用腕相机 RGB 重新定位积木（分割、拟合顶面矩形、射线求交顶面平面），按门限融合进信念，再重新规划剩余的下降。K=0 就是现在的开环。
2. **同步与异步**：同步是在检查点停稳（`settle_s`）、拍照、等结果、再走。异步是到点即拍、不停，按旧目标继续下降，结果在 `latency_s` 后才可用；可用时从当前位置改道去新目标。积木在桌上静止，世界坐标下的估计不会因延迟而过时，所以异步不需要做运动补偿。
3. **取证闭环**：
   - (a) realistic 的信念相机改为*估计*的相机（顶相机用标定结果 `_cam_est`），修 G9；
   - (b) 裁剪中心改由图像里的积木色块决定，投影只用来给出搜索窗口；
   - (c) 可选的"密集在途取证"：扫描时不在每个视角停稳，而是在路径上按间距 `spacing_deg` 在途拍照；检测器结果按 `detector_latency_s` 延迟生效，一报警就提前结束。
4. **在线漂移信号**：每回合第一次腕部定位的新息（腕部估计与顶视估计之差）就是传感器层面的漂移残差。记为 `innovation_xy_mm`，可以进工具返回值。它取代第 3.4 节里"在线漂移检测"这个候选。

**不变的东西**：默认变体（`motion="open"`、`evidence="hold"`、`belief_cameras="actual"`）下，所有旧回合逐位不变。学生、ACT 和 SmolVLA 的路径不动。教师要不要换成新变体，由 C7 的实验决定，再由用户批准。

## 6. 实验与选优（判据先写）

**变体**

| 名称 | 抓取检查点（相对抓取点的高度，m） | 模式 |
|---|---|---|
| `open` | 无 | 开环（现状） |
| `w1s` | hover | 同步 |
| `w2s` | hover, 0.035 | 同步 |
| `w3s` | hover, 0.045, 0.030 | 同步 |
| `w2a` | hover, 0.035 | 异步 |
| `w3a` | hover, 0.045, 0.030 | 异步 |
| `w4a` | hover, 0.050, 0.040, 0.030 | 异步 |

取证变体：`hold`（现状），`img`（估计相机加图像裁剪），`dense`（`img` 加密集在途取证，`spacing_deg` ∈ {10, 20}）。

**阶段**（全部标【GPU】，因为要跑检测器）

1. **容差曲线**：`open` 和 `w2s` 两组，顶相机标定误差 0、2、5、10、20 mm（C6 的 `calib_offset_mm`），套件 p2_dev12，策略 B。目的是找出开环成功率开始下降的拐点，也就是真机标定要多准。
2. **抓取变体初筛**：7 个抓取变体 × {fixed, drift} × p2_dev12，取证固定为 `hold`。按下面的判据选出前 2 名。
3. **取证变体**：前 2 名 × {`hold`, `img`, `dense-10`, `dense-20`} × {fixed, recal, drift}。
4. **确认**：最终胜者对 `open`，在闭环新种子段（第 3.2 节规则，取本轮之后未用过的 96 个案例）上跑 {fixed, recal, drift}。

**选优判据**：在跑阶段 2 之前写进 `docs/experiments/cv_motion_prereg.md`，并单独提交。

- 主指标：drift 下的物理成功率（`scoring` 的 `outcome == "success"`）。
- 约束：
  - fixed 和 recal 下，成功的回合数不比 `open` 少 1 个以上；
  - 标签准确率不低于 `open`；
  - 不新增 `infra_failed`。
- 在满足约束、且 drift 成功率与最好者相差不超过 1 个回合的变体里，取机器人时间（`sim_s` 中位数）最短的；再打平，取检查点更少的。
- 确认阶段：胜者对 `open`，drift 下精确 McNemar 单侧 p < 0.05，并且另外两种条件没有显著变差，才算成立。不成立就保留 `open`，并在报告里写明。
- 墙钟时间只报告，不参与选择，因为仿真里检测器的耗时不占机器人时间。

**风险**（仿真看不出来，写进报告）：

- 在途拍照有运动模糊。仿真里没有模糊，所以 `dense` 的结果对真机偏乐观。
- 腕相机离积木很近时可能裁边，C1 用 `CLIPPED` 拒绝这类视图。
- 真机的腕相机安装误差需要单独标定一次。

## 第二部分文件结构

| 文件 | 改动 | 职责 |
|---|---|---|
| `src/digital_twin_armigo/agent/wrist_localize.py` | 新 | 纯 CV：腕相机位姿（设计安装位）、内参、色块分割、矩形拟合、射线求交、融合门限 |
| `src/digital_twin_armigo/agent/grasp_loop.py` | 新 | `GraspLoop`：检查点、同步/异步下降、`VARIANTS` |
| `src/digital_twin_armigo/mujoco_cell/physics_expert.py` | 改 | `_approach_and_descend` 的 `recheck` 钩子（为 None 时逐位不变） |
| `src/digital_twin_armigo/mujoco_cell/cell.py` | 改 | `camera_override(name, world_from_cam)` 上下文 |
| `src/digital_twin_armigo/mujoco_cell/inspection_views.py` | 改 | `detector_crop(..., center_px=None)`：给定中心时按中心裁剪 |
| `src/digital_twin_armigo/agent/twin_mujoco.py` | 改 | `_wrist_look`、`wrist_fix` 事件、`belief_cameras`、`evidence` 选项、`calib_offset_mm` |
| `src/digital_twin_armigo/agent/step_mcp.py` | 改 | `SIMULATOR_KEYS` 不变；`wrist_fix` 的传感器字段可以进工具返回值 |
| `src/digital_twin_armigo/experiments/runner.py`、`contract.py` | 改 | `--motion`、`--evidence`、`--calib-offset-mm`；组名后缀；结果字段 |
| `scripts/experiments/cv_motion_matrix.sh` | 新 | 第 6 节的四个阶段 |
| `docs/experiments/cv_motion_prereg.md` | 新 | 选优判据 |
| `tests/test_wrist_localize.py`、`tests/test_grasp_loop.py`、`tests/test_belief_cameras.py`、`tests/test_cv_motion_runner.py` | 新 | 各任务的测试 |

---

### Task C1：腕相机 RGB 定位积木（纯 CV）

**Files:**
- Create: `src/digital_twin_armigo/agent/wrist_localize.py`
- Test: `tests/test_wrist_localize.py`

**Interfaces:**
- Produces:
  - `believed_wrist_from_cam(cell, kin_data) -> np.ndarray(4,4)`：OpenCV 约定，设计安装位，按关节读数计算。
  - `wrist_k(fovy_deg: float, shape: tuple) -> np.ndarray(3,3)`：腕相机内参，`shape` 取图像的形状。
  - `localize_from_wrist(rgb, k, world_from_cam, prior_xy, prior_yaw, size, top_z, *, gate_m=0.04, max_tilt_deg=50.0) -> tuple[WristFix | None, str | None]`，第二项是失败原因：`TILTED`、`NO_BLOB`、`CLIPPED`、`BAD_FOOTPRINT`。
  - `WristFix(pos_xy, yaw, area_px, fill, footprint_m, tilt_deg)`。
  - `fuse(prior: BrickEstimate, fix: WristFix, cfg: FuseCfg) -> tuple[BrickEstimate | None, dict]`。返回的 `dict` 字段：`innovation_xy_mm`、`innovation_yaw_deg`、`accepted`。
  - `FuseCfg(gate_xy_m=0.04, gate_yaw_deg=30.0, w_wrist=1.0)`。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_wrist_localize.py
import math

import cv2
import numpy as np
import pytest

from digital_twin_armigo.agent.perception import BrickEstimate
from digital_twin_armigo.agent.wrist_localize import FuseCfg, WristFix, fuse, localize_from_wrist, wrist_k

SIZE = (0.032, 0.016, 0.0192)  # 2x4
TOP_Z = 0.0192


def _down_camera(x=0.30, y=0.0, z=0.20, tilt_deg=0.0):
	"""Camera above (x, y, z) looking straight down (OpenCV: z forward = world -z), optionally tilted about x."""
	t = math.radians(tilt_deg)
	tilt = np.array([[1, 0, 0], [0, math.cos(t), -math.sin(t)], [0, math.sin(t), math.cos(t)]])
	tf = np.eye(4)
	tf[:3, :3] = np.diag([1.0, -1.0, -1.0]) @ tilt
	tf[:3, 3] = (x, y, z)
	return tf


def _render(world_from_cam, k, bricks, shape=(240, 320), bg=(120, 120, 118)):
	img = np.empty((*shape, 3), np.uint8)
	img[:] = bg
	rng = np.random.default_rng(0)
	img = np.clip(img.astype(int) + rng.integers(-4, 5, img.shape), 0, 255).astype(np.uint8)
	cam_from_world = np.linalg.inv(world_from_cam)
	for (cx, cy, yaw, colour) in bricks:
		c, s = math.cos(yaw), math.sin(yaw)
		hx, hy = SIZE[0] / 2, SIZE[1] / 2
		corners = np.array([[cx + c * a - s * b, cy + s * a + c * b, TOP_Z] for a, b in ((hx, hy), (-hx, hy), (-hx, -hy), (hx, -hy))])
		p = corners @ cam_from_world[:3, :3].T + cam_from_world[:3, 3]
		uv = (p[:, :2] / p[:, 2:3]) @ k[:2, :2].T + k[:2, 2]
		cv2.fillPoly(img, [np.round(uv).astype(np.int32)], colour)
	return img


def test_recovers_pose_from_a_top_down_view():
	k = wrist_k(60.0, (240, 320))
	w2c = _down_camera()
	img = _render(w2c, k, [(0.305, 0.004, math.radians(20), (200, 30, 30))])
	fix, why = localize_from_wrist(img, k, w2c, np.array([0.300, 0.0]), math.radians(14), SIZE, TOP_Z)
	assert why is None
	assert np.linalg.norm(fix.pos_xy - [0.305, 0.004]) < 1e-3
	assert abs(math.degrees(fix.yaw) - 20) < 1.5


def test_ignores_a_same_colour_brick_outside_the_gate():
	k = wrist_k(60.0, (240, 320))
	w2c = _down_camera()
	img = _render(w2c, k, [(0.302, 0.0, 0.0, (200, 30, 30)), (0.36, 0.05, 0.3, (200, 30, 30))])
	fix, why = localize_from_wrist(img, k, w2c, np.array([0.300, 0.0]), 0.0, SIZE, TOP_Z)
	assert why is None and np.linalg.norm(fix.pos_xy - [0.302, 0.0]) < 1e-3


def test_rejects_tilted_views_and_resolves_yaw_symmetry():
	k = wrist_k(60.0, (240, 320))
	tilted = _down_camera(tilt_deg=60)
	img = _render(tilted, k, [(0.30, 0.0, 0.0, (200, 30, 30))])
	assert localize_from_wrist(img, k, tilted, np.array([0.30, 0.0]), 0.0, SIZE, TOP_Z) == (None, "TILTED")
	w2c = _down_camera()
	img = _render(w2c, k, [(0.30, 0.0, math.radians(5), (200, 30, 30))])
	fix, _ = localize_from_wrist(img, k, w2c, np.array([0.30, 0.0]), math.radians(5 + 180), SIZE, TOP_Z)
	assert abs(math.degrees(fix.yaw) - (5 + 180)) < 1.5  # nearest symmetric yaw to the prior


def test_fuse_gates_large_innovations():
	prior = BrickEstimate(pos=np.array([0.30, 0.0, 0.0096]), yaw=0.0, size=SIZE, footprint_m=(0.032, 0.016), top_points=100)
	near = WristFix(pos_xy=np.array([0.31, 0.0]), yaw=math.radians(4), area_px=900, fill=0.95, footprint_m=(0.032, 0.016), tilt_deg=0.0)
	fused, info = fuse(prior, near, FuseCfg())
	assert info["accepted"] and info["innovation_xy_mm"] == pytest.approx(10.0, abs=0.01)
	assert fused.pos[0] == pytest.approx(0.31) and fused.pos[2] == pytest.approx(0.0096)
	far = WristFix(pos_xy=np.array([0.36, 0.0]), yaw=0.0, area_px=900, fill=0.95, footprint_m=(0.032, 0.016), tilt_deg=0.0)
	assert fuse(prior, far, FuseCfg())[0] is None
```

- [ ] **Step 2：运行，确认失败**（`ModuleNotFoundError: wrist_localize`）

Run: `E:/digital_twin_armigo/.venv/Scripts/python -m pytest tests/test_wrist_localize.py -v`

- [ ] **Step 3：实现**

```python
# src/digital_twin_armigo/agent/wrist_localize.py
"""Wrist-camera brick localisation from RGB only (sensor side, no simulator truth).

The camera pose comes from the joint readings and the *designed* wrist mount (``Cell.nominal_wrist_cam``), never from
``model.cam_pos`` of the simulation, which holds the actual, possibly offset, mount. The brick is the colour blob nearest
to its predicted image position; the top-face rectangle's corners are intersected with the top-face plane."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import cv2
import mujoco
import numpy as np

from .perception import BrickEstimate, yaw_error_deg

__all__ = ["FuseCfg", "WristFix", "believed_wrist_from_cam", "fuse", "localize_from_wrist", "wrist_k"]


@dataclass(frozen=True)
class WristFix:
	pos_xy: np.ndarray
	yaw: float
	area_px: int
	fill: float
	footprint_m: tuple[float, float]
	tilt_deg: float


@dataclass(frozen=True)
class FuseCfg:
	gate_xy_m: float = 0.04
	gate_yaw_deg: float = 30.0
	w_wrist: float = 1.0


def believed_wrist_from_cam(cell, kin_data) -> np.ndarray:
	"""World-from-camera (OpenCV) of the wrist camera on its designed mount at ``kin_data.qpos``."""
	m = cell.model
	cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
	mujoco.mj_kinematics(m, kin_data)
	with cell.nominal_wrist_cam(kin_data):
		mujoco.mj_camlight(m, kin_data)
		tf = np.eye(4)
		tf[:3, :3] = kin_data.cam_xmat[cid].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
		tf[:3, 3] = kin_data.cam_xpos[cid]
	return tf


def wrist_k(fovy_deg: float, shape) -> np.ndarray:
	h, w = shape[:2]
	f = 0.5 * h / math.tan(math.radians(fovy_deg) / 2)
	return np.array([[f, 0.0, w / 2 - 0.5], [0.0, f, h / 2 - 0.5], [0.0, 0.0, 1.0]])


def _project(k, world_from_cam, p):
	c = np.linalg.inv(world_from_cam)
	q = c[:3, :3] @ p + c[:3, 3]
	return (k @ (q / q[2]))[:2], float(q[2])


def _rays_to_plane(px, k, world_from_cam, z):
	d = np.c_[px, np.ones(len(px))] @ np.linalg.inv(k).T @ world_from_cam[:3, :3].T
	o = world_from_cam[:3, 3]
	return o + ((z - o[2]) / d[:, 2])[:, None] * d


def _blob(rgb, centre, gate_px, min_delta=35.0):
	lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
	h, w = rgb.shape[:2]
	yy, xx = np.mgrid[0:h, 0:w]
	r = np.hypot(xx - centre[0], yy - centre[1])
	ring = (r > 1.6 * gate_px) & (r < 2.4 * gate_px)
	if ring.sum() < 50:
		ring = r > 1.6 * gate_px
	bg = np.median(lab[ring], axis=0)
	mask = ((np.linalg.norm(lab - bg, axis=-1) > min_delta) & (r < gate_px)).astype(np.uint8)
	mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
	n, labels, _, cent = cv2.connectedComponentsWithStats(mask)
	if n <= 1:
		return None
	i = 1 + int(np.argmin(np.hypot(cent[1:, 0] - centre[0], cent[1:, 1] - centre[1])))
	return (labels == i).astype(np.uint8)


def localize_from_wrist(rgb, k, world_from_cam, prior_xy, prior_yaw, size, top_z, *, gate_m=0.04, max_tilt_deg=50.0):
	axis = world_from_cam[:3, 2]
	tilt = math.degrees(math.acos(float(np.clip(-axis[2], -1.0, 1.0))))
	if tilt > max_tilt_deg:
		return None, "TILTED"
	centre, depth = _project(k, world_from_cam, np.array([prior_xy[0], prior_xy[1], top_z]))
	mask = _blob(rgb, centre, gate_m * k[0, 0] / depth)
	if mask is None:
		return None, "NO_BLOB"
	if mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
		return None, "CLIPPED"
	contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	contour = max(contours, key=cv2.contourArea)
	rect = cv2.minAreaRect(contour)
	corners = _rays_to_plane(cv2.boxPoints(rect), k, world_from_cam, top_z)
	edges = np.roll(corners, -1, axis=0) - corners
	lengths = np.linalg.norm(edges[:, :2], axis=1)
	long_i = int(np.argmax(lengths[:2]))
	footprint = (float(lengths[long_i]), float(lengths[1 - long_i]))
	want = sorted(size[:2], reverse=True)
	if abs(footprint[0] - want[0]) > 0.3 * want[0] or abs(footprint[1] - want[1]) > 0.3 * want[1]:
		return None, "BAD_FOOTPRINT"
	yaw = math.atan2(edges[long_i, 1], edges[long_i, 0])
	period = math.pi / 2 if abs(size[0] - size[1]) < 1e-6 else math.pi
	yaw += period * round((prior_yaw - yaw) / period)
	area = float(cv2.contourArea(contour))
	box = max(rect[1][0] * rect[1][1], 1.0)
	return WristFix(corners.mean(axis=0)[:2], yaw, int(mask.sum()), area / box, footprint, tilt), None


def fuse(prior: BrickEstimate, fix: WristFix, cfg: FuseCfg):
	d_xy = fix.pos_xy - prior.pos[:2]
	d_yaw_deg = yaw_error_deg(fix.yaw, prior.yaw, prior.size)
	info = {
		"innovation_xy_mm": round(float(np.linalg.norm(d_xy) * 1000), 3),
		"innovation_yaw_deg": round(d_yaw_deg, 3),
		"accepted": False,
	}
	if np.linalg.norm(d_xy) > cfg.gate_xy_m or d_yaw_deg > cfg.gate_yaw_deg:
		return None, info
	info["accepted"] = True
	pos = prior.pos.copy()
	pos[:2] += cfg.w_wrist * d_xy
	yaw = prior.yaw + cfg.w_wrist * (fix.yaw - prior.yaw)
	return replace(prior, pos=pos, yaw=yaw), info
```

- [ ] **Step 4：运行测试，确认通过；`ruff check`、`ruff format` 通过**

- [ ] **Step 5：用 MuJoCo 渲染做一次集成检查（CPU，不提交结果）**

用 `twin_mujoco.sample_case(seed, positive=False, face=0, options={"cameras": "drift"})` 建 4 个案例。各让专家走到 hover，取 `cell.rgb("wrist_cam")`、`believed_wrist_from_cam` 和 `wrist_k`；先验用 `brick_perceived` 的顶视估计。把 `localize_from_wrist` 的结果和 `cell.brick_pos()` 比较，验收线是 xy < 3 mm、yaw < 3°。结果记进提交信息正文。

如果达不到：先调 `min_delta` 和 hover 高度，再看是不是侧面混进了色块。**不改**测试里的合成阈值，未通过的情况写进报告。

**SO-101 腕相机是斜装的。** `assets/so101_mjcf/so101.xml` 里 `wrist_cam` 的四元数约合 33° 倾角，`fovy` 为 61，渲染尺寸 640×480（`RENDER_SIZE`）。所以在 hover 位，视轴偏离竖直方向可能有 30° 上下，侧面会进入画面。`max_tilt_deg` 的默认值因此取 50。侧面像素混进色块，会让拟合出的矩形偏大或偏移，`BAD_FOOTPRINT` 会拒掉一部分视图。Step 5 要报告各种失败原因的计数和实测倾角。如果侧面污染是主要的失败原因，就把结果记下来，交给第三部分（U3 的顶面分割）解决，不在 C1 里加启发式规则。

- [ ] **Step 6：提交**

```bash
git add src/digital_twin_armigo/agent/wrist_localize.py tests/test_wrist_localize.py
git commit -m "obs C1: RGB wrist-camera brick localisation on the designed mount, with fusion gate"
```

---

### Task C2：专家的 `recheck` 钩子与 `GraspLoop`（同步/异步）

**Files:**
- Create: `src/digital_twin_armigo/agent/grasp_loop.py`
- Modify: `src/digital_twin_armigo/mujoco_cell/physics_expert.py:592-606`
- Test: `tests/test_grasp_loop.py`

**Interfaces:**
- Consumes: 专家的 `_move(q, secs, phase)`、`_hold_servo(q, secs, phase)`、`_ik(pos, q0, roll)`、`last`、`p`、`_active_open_g`。
- Produces:
  - `GraspLoop(heights, mode, look, settle_s=0.15, latency_s=0.10, segment_s=0.6, fps=FPS)`；
  - `GraspLoop.descend(ex, phase, target, roll, q) -> (target, roll, q)`（生成器）；
  - `VARIANTS: dict[str, tuple[tuple[str | float, ...], str] | None]`，`"hover"` 表示 `p.hover_z_m - ex.z_grasp`；
  - `look(h) -> (target, roll, q_grasp) | None`，由 twin 提供，None 表示不改道；
  - 专家属性 `recheck: GraspLoop | None = None`。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_grasp_loop.py
import numpy as np

from digital_twin_armigo.agent.grasp_loop import VARIANTS, GraspLoop


class FakeExpert:
	"""Records commanded targets; _move yields `round(secs * 30)` frames ending at the goal."""

	class P:
		hover_z_m, pregrasp_dz_m, descend_s = 0.06, 0.02, 0.5

	p, z_grasp, _active_open_g = P(), 0.01, 0.9

	def __init__(self):
		self.last = np.zeros(6)
		self.frames = []

	def _ik(self, pos, q0, roll):
		q = np.zeros(6)
		q[:3] = pos
		return q

	def _move(self, q, secs, phase):
		for _ in range(max(1, round(secs * 30))):
			self.last = q.copy()
			self.frames.append(q[:3].copy())
			yield q.copy(), phase

	def _hold_servo(self, q, secs, phase):
		yield from self._move(q, secs, phase)


def _run(gen):
	try:
		while True:
			next(gen)
	except StopIteration as stop:
		return stop.value


def test_variants_table():
	assert VARIANTS["open"] is None
	assert VARIANTS["w2s"] == (("hover", 0.035), "sync")
	assert VARIANTS["w4a"][1] == "async"


def test_sync_looks_at_each_checkpoint_and_replans():
	ex, looks = FakeExpert(), []
	new_target = np.array([0.31, 0.0, 0.01])

	def look(h):
		looks.append((round(h, 4), ex.last[:3].copy()))
		return (new_target, 0.0, np.zeros(6)) if len(looks) == 1 else None

	loop = GraspLoop(("hover", 0.035), "sync", look, fps=30)
	target, roll, q = _run(loop.descend(ex, "approach", np.array([0.30, 0.0, 0.01]), 0.0, np.zeros(6)))
	assert [h for h, _ in looks] == [0.05, 0.035]
	assert np.allclose(looks[1][1][:2], new_target[:2])  # the second look is above the replanned target
	assert np.allclose(target, new_target)


def test_async_applies_the_result_after_the_latency():
	ex, looks = FakeExpert(), []
	new_target = np.array([0.31, 0.0, 0.01])

	def look(h):
		looks.append(len(ex.frames))
		return (new_target, 0.0, np.zeros(6)) if len(looks) == 1 else None

	loop = GraspLoop(("hover", 0.035), "async", look, latency_s=0.1, segment_s=0.5, fps=30)
	target, _, _ = _run(loop.descend(ex, "approach", np.array([0.30, 0.0, 0.01]), 0.0, np.zeros(6)))
	first = looks[0]
	# the next 3 frames still head for the old target, then the arm turns to the new one
	assert np.allclose(ex.frames[first + 2][:2], [0.30, 0.0])
	assert np.allclose(ex.frames[-1][:2], new_target[:2]) and np.allclose(target, new_target)
	assert len(ex.frames) < 30 * (0.5 + 0.15) * 3  # no settle holds
```

- [ ] **Step 2：运行，确认失败**

- [ ] **Step 3：实现 `grasp_loop.py`，并给专家加钩子**

```python
# src/digital_twin_armigo/agent/grasp_loop.py
"""Closed-loop grasp descent: re-localise the brick with the wrist camera at checkpoints between hover and pre-grasp.

``sync`` settles at every checkpoint and waits for the result; ``async`` looks on arrival without settling, keeps
descending towards the old target, and turns to the replanned one ``latency_s`` later. The brick lies still on the table,
so a world-frame estimate does not go stale during the latency."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..mujoco_cell.physics_expert import FPS

__all__ = ["VARIANTS", "GraspLoop"]

VARIANTS: dict[str, tuple[tuple, str] | None] = {
	"open": None,
	"w1s": (("hover",), "sync"),
	"w2s": (("hover", 0.035), "sync"),
	"w3s": (("hover", 0.045, 0.030), "sync"),
	"w2a": (("hover", 0.035), "async"),
	"w3a": (("hover", 0.045, 0.030), "async"),
	"w4a": (("hover", 0.050, 0.040, 0.030), "async"),
}


@dataclass
class GraspLoop:
	heights: tuple
	mode: str
	look: Callable
	settle_s: float = 0.15
	latency_s: float = 0.10
	segment_s: float = 0.6
	fps: int = FPS

	def _heights(self, ex):
		hover = ex.p.hover_z_m - ex.z_grasp
		return [hover if h == "hover" else float(h) for h in self.heights]

	def _goal(self, ex, target, h, roll, q):
		goal = ex._ik(np.asarray(target) + np.array([0.0, 0.0, h]), q, roll)
		goal[5] = ex._active_open_g
		return goal

	def descend(self, ex, phase, target, roll, q):
		if self.mode == "sync":
			for h in self._heights(ex):
				goal = self._goal(ex, target, h, roll, ex.last)
				yield from ex._move(goal, self.segment_s, phase)
				yield from ex._hold_servo(goal, self.settle_s, phase)
				replan = self.look(h)
				if replan is not None:
					target, roll, _ = replan
			return target, roll, ex.last.copy()
		lat = max(1, round(self.latency_s * self.fps))
		pending = None  # [frames until usable, replan or None]
		stops = [*self._heights(ex), ex.p.pregrasp_dz_m]
		for i, h in enumerate(stops):
			while True:
				turned = False
				for frame in ex._move(self._goal(ex, target, h, roll, ex.last), self.segment_s, phase):
					yield frame
					if pending is not None:
						pending[0] -= 1
						if pending[0] == 0:
							replan, pending = pending[1], None
							if replan is not None:
								target, roll, _ = replan
								turned = True
								break
				if not turned:
					break
			if i < len(stops) - 1:
				pending = [lat, self.look(h)]
		if pending is not None and pending[1] is not None:
			target, roll, _ = pending[1]
		return target, roll, ex.last.copy()
```

`physics_expert.py` 里，`__init__` 加 `self.recheck = None`；`_approach_and_descend` 改为：

```python
	def _approach_and_descend(self, phase: str):
		"""Plan a grasp for the brick's current pose, move over it, descend and servo onto the grasp point. With a
		``recheck`` loop, re-localise between hover and pre-grasp (see agent/grasp_loop.py)."""
		p = self.p
		target, roll, q_grasp = self._grasp_plan(open_g=self._active_open_g)
		hover = target + np.array([0.0, 0.0, p.hover_z_m - self.z_grasp])
		q = self._ik(hover, q_grasp, roll)
		q[5] = self._active_open_g
		yield from self._move(q, p.hover_move_s, phase)
		if self.recheck is not None:
			target, roll, q = yield from self.recheck.descend(self, phase, target, roll, q)
		pre = target + np.array([0.0, 0.0, p.pregrasp_dz_m])
		q = self._ik(pre, q, roll)
		yield from self._move(q, p.descend_s, phase)
		q = self._ik(target, q, roll)
		yield from self._move(q, p.final_descend_s, "grasp")
		corr = yield from self._servo(target, roll, "grasp", p.servo_tol_m)
		return _Grasp(target=target, roll=roll, corr=corr)
```

`recheck is None` 时，调用顺序和参数与原来完全相同。如果 `FPS` 不在 `physics_expert` 里，就从它实际所在的模块导入，并在报告里注明。

- [ ] **Step 4：运行新测试，再跑 `tests/test_physics_expert*.py` 和 `tests/test_twin*.py`，确认旧测试逐位不变**

- [ ] **Step 5：提交** `obs C2: recheck hook in the grasp descent and the sync/async GraspLoop`

---

### Task C3：twin 接入腕部闭环，记录 `wrist_fix` 事件

**Files:**
- Modify: `src/digital_twin_armigo/agent/twin_mujoco.py`（`__init__` 的 `motion` 选项、`_grasp`、新方法 `_wrist_look`）
- Modify: `src/digital_twin_armigo/agent/step_mcp.py`（工具返回值增加 `last_wrist_fix`，只含传感器字段）
- Test: `tests/test_grasp_loop.py`（追加 MuJoCo 集成测试）、`tests/test_truth_diagnostics.py`（H1 的守护要覆盖新事件）

**Interfaces:**
- Consumes: C1 的 `localize_from_wrist`、`fuse`、`believed_wrist_from_cam`、`wrist_k`；C2 的 `GraspLoop`、`VARIANTS`。
- Produces:
  - 事件 `wrist_fix`：
    - 传感器字段：`height_m`、`found`、`reason`、`innovation_xy_mm`、`innovation_yaw_deg`、`accepted`、`area_px`、`fill`、`tilt_deg`、`latency_s`、`mode`；
    - 真值字段（只给评估器，H1 登记）：`error_xy_mm`、`error_yaw_deg`。
  - 回合结果字段 `motion`，取值见 `VARIANTS` 的键。

- [ ] **Step 1: 写失败测试**

```python
# appended to tests/test_grasp_loop.py
import pytest

from digital_twin_armigo.agent.step_mcp import assert_sensor_only


@pytest.mark.parametrize("motion", ["w2s", "w2a"])
def test_twin_closed_loop_grasp_logs_wrist_fixes(motion, tmp_path):
	from digital_twin_armigo.agent.twin_mujoco import TwinMujoco, sample_case

	case = sample_case(3_000_101, positive=False, face=0, options={"cameras": "drift"})
	twin = TwinMujoco.from_case(case, sensing="realistic", motion=motion, out_dir=tmp_path, scorer=None)
	twin.prepare()
	assert twin._grasp() is None
	fixes = [e for e in twin.events if e["kind"] == "wrist_fix"]
	assert [round(e["height_m"], 3) for e in fixes] == [0.05, 0.035]
	assert all(e["mode"] == ("sync" if motion == "w2s" else "async") for e in fixes)
	sensor = {k: v for k, v in fixes[0].items() if k not in ("error_xy_mm", "error_yaw_deg")}
	assert_sensor_only(sensor)


def test_open_motion_logs_nothing_new(tmp_path):
	from digital_twin_armigo.agent.twin_mujoco import TwinMujoco, sample_case

	case = sample_case(3_000_101, positive=False, face=0, options={"cameras": "drift"})
	twin = TwinMujoco.from_case(case, sensing="realistic", motion="open", out_dir=tmp_path, scorer=None)
	twin.prepare()
	twin._grasp()
	assert not [e for e in twin.events if e["kind"] == "wrist_fix"]
```

`TwinMujoco.from_case`、`prepare` 和 `events` 的实际名字，要先读 `twin_mujoco.py` 里现有测试（`tests/test_twin*.py`）怎样构造实例，然后照抄那里的构造方式。测试里的构造写法可以调整，但断言的内容不能改。

- [ ] **Step 2：运行，确认失败**

- [ ] **Step 3：实现**

```python
	# TwinMujoco.__init__ (after the expert is built), realistic only:
	self.motion = options.get("motion", "open")
	spec = VARIANTS[self.motion]
	if spec is not None:
		if not self.realistic:
			raise ValueError("motion variants are for realistic sensing; truth sees the brick directly")
		heights, mode = spec
		self.expert.recheck = GraspLoop(heights, mode, self._wrist_look)
	self.fuse_cfg = FuseCfg()

	def _wrist_look(self, h):
		"""One wrist-camera re-localisation during the grasp descent: fuse it into the belief and return the replanned
		grasp (target, roll, q_grasp), or None to keep the current plan."""
		cell, ex = self.cell, self.expert
		rgb = cell.rgb("wrist_cam")
		ex.kin.data.qpos[:] = cell.data.qpos
		w2c = believed_wrist_from_cam(cell, ex.kin.data)
		cid = mujoco.mj_name2id(cell.model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
		k = wrist_k(float(cell.model.cam_fovy[cid]), rgb.shape)
		size = self._believed_brick_size()
		top_z = float(self.brick_est.pos[2] + size[2] / 2)
		fix, reason = localize_from_wrist(rgb, k, w2c, self.brick_est.pos[:2], self.brick_est.yaw, size, top_z)
		info = {"accepted": False}
		if fix is not None:
			fused, info = fuse(self.brick_est, fix, self.fuse_cfg)
			info |= {"area_px": fix.area_px, "fill": round(fix.fill, 3), "tilt_deg": round(fix.tilt_deg, 2)}
			if fused is not None:
				self.brick_est = fused
		loop = ex.recheck
		rot = cell.data.xmat[cell.brick_body].reshape(3, 3)
		self.event(
			"wrist_fix",
			height_m=round(float(h), 4),
			found=fix is not None,
			reason=reason,
			mode=loop.mode,
			latency_s=loop.latency_s if loop.mode == "async" else 0.0,
			**info,
			# Truth for scoring only (registered in TRUTH_DIAGNOSTIC_KEYS); nothing in the agent reads it back.
			error_xy_mm=round(float(np.linalg.norm(self.brick_est.pos[:2] - cell.brick_pos()[:2]) * 1000), 3),
			error_yaw_deg=round(yaw_error_deg(self.brick_est.yaw, math.atan2(rot[1, 0], rot[0, 0]), self.layout.brick_size), 3),
		)
		if not info["accepted"]:
			return None
		with self._belief():
			ex.kin.data.qpos[:] = cell.data.qpos
			return ex._grasp_plan(open_g=ex._active_open_g)
```

- **规划依据**：重新规划在 `_belief()` 里做，规划器看到的是更新后的信念积木。异步模式下，`look` 在拍照那一帧就调用，结果由 `GraspLoop` 延后生效。仿真里图像和结果都是确定的，这样做与"晚一点才算出来"等价。
- **检查点与回合恢复**：`_save` 和 `restore_state` 要把 `motion` 和 `fuse_cfg` 写进 checkpoint 的 `agent` 字典，否则回退之后闭环会丢失。
- **工具返回值**：`step_mcp` 的 `session_tool` 返回值加 `last_wrist_fix`，只放传感器字段（剔掉 `error_*`），必须通过 `assert_sensor_only`。

- [ ] **Step 4：运行新测试、H1 守护测试和全量测试**

- [ ] **Step 5：提交** `obs C3: wrist re-localisation in the grasp descent, wrist_fix events and tool field`

---

### Task C4：信念相机用估计位姿，裁剪跟随图像（修 G9）

**Files:**
- Modify: `src/digital_twin_armigo/mujoco_cell/cell.py`（`camera_override`）
- Modify: `src/digital_twin_armigo/mujoco_cell/inspection_views.py:94`（`detector_crop(..., center_px=None)`）
- Modify: `src/digital_twin_armigo/agent/twin_mujoco.py`（`_belief`、`_capture`，新增选项 `belief_cameras` 和 `evidence`）
- Test: `tests/test_belief_cameras.py`

**Interfaces:**
- Produces:
  - `Cell.camera_override(name: str, world_from_cam_cv: np.ndarray, data=None)` 上下文：把模型相机临时换成给定位姿（OpenCV 约定，内部转成 MuJoCo 约定）。
  - `detector_crop(cell, camera, crop, data, image_size, pad=..., clamp=False, *, rgb=None, center_px=None)`：给 `center_px` 时，按它和渲染出的 bbox 尺寸裁剪。
  - 选项 `belief_cameras` ∈ {`actual`（默认，旧行为）, `estimated`}；`evidence` ∈ {`hold`（默认）, `img`, `dense-10`, `dense-20`}。`img` 和 `dense-*` 隐含 `belief_cameras="estimated"`。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_belief_cameras.py
import numpy as np
import pytest

from digital_twin_armigo.mujoco_cell.handeye import camera_truth


def _cell(seed=3_000_101, cameras="drift"):
	from digital_twin_armigo.agent.twin_mujoco import sample_case
	from digital_twin_armigo.mujoco_cell.cell import Cell

	layout, *_ = sample_case(seed, positive=False, face=0, options={"cameras": cameras})
	return Cell(layout)


def test_camera_override_moves_the_model_camera_and_restores_it():
	cell = _cell()
	before = camera_truth(cell.model, cell.data, "top")
	moved = before.copy()
	moved[:3, 3] += [0.01, -0.02, 0.0]
	with cell.camera_override("top", moved):
		assert np.allclose(camera_truth(cell.model, cell.data, "top"), moved, atol=1e-9)
	assert np.allclose(camera_truth(cell.model, cell.data, "top"), before, atol=1e-12)


def test_estimated_belief_camera_differs_from_actual_under_drift(tmp_path):
	"""G9: with belief_cameras='estimated' the top-view crop is placed through the stale calibration, not the truth."""
	from digital_twin_armigo.agent.twin_mujoco import TwinMujoco, sample_case

	case = sample_case(3_000_101, positive=False, face=0, options={"cameras": "drift"})
	twin = TwinMujoco.from_case(case, sensing="realistic", belief_cameras="estimated", out_dir=tmp_path, scorer=None)
	twin.prepare()
	with twin._belief():
		est = camera_truth(twin.cell.model, twin.cell.data, "top")
	assert np.allclose(est, twin._cam_est, atol=1e-9)
	assert not np.allclose(est, camera_truth(twin.cell.model, twin.cell.data, "top"), atol=1e-4)


def test_center_px_crop_is_centred_on_the_given_point():
	from digital_twin_armigo.mujoco_cell.inspection_views import detector_crop

	cell = _cell(cameras="fixed")
	img, bbox = detector_crop(cell, "top", None, cell.data, 96, clamp=True)
	img2, bbox2 = detector_crop(cell, "top", None, cell.data, 96, clamp=True, center_px=(bbox[0] + bbox[2] / 2 + 6, bbox[1] + bbox[3] / 2))
	assert bbox2[2:] == bbox[2:] and bbox2[0] == pytest.approx(bbox[0] + 6, abs=1)
```

- [ ] **Step 2：运行，确认失败**

- [ ] **Step 3：实现**

```python
	# Cell
	@contextmanager
	def camera_override(self, name, world_from_cam_cv, data=None):
		"""Show a model camera at another pose (OpenCV world-from-camera) for planners and belief renders.

		Only cameras without a parent body are supported (top, front); the wrist camera goes through nominal_wrist_cam."""
		m, d = self.model, self.data if data is None else data
		cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, name)
		if m.cam_bodyid[cid] != 0:
			raise ValueError(f"{name} is mounted on a body; use nominal_wrist_cam")
		saved = m.cam_pos[cid].copy(), m.cam_quat[cid].copy()
		rot = world_from_cam_cv[:3, :3] @ np.diag([1.0, -1.0, -1.0])
		quat = np.empty(4)
		mujoco.mju_mat2Quat(quat, rot.flatten())
		m.cam_pos[cid], m.cam_quat[cid] = world_from_cam_cv[:3, 3], quat
		mujoco.mj_camlight(m, d)
		try:
			yield
		finally:
			m.cam_pos[cid], m.cam_quat[cid] = saved
			mujoco.mj_camlight(m, d)
```

```python
	# TwinMujoco._belief, realistic branch
	if self.realistic:
		with ExitStack() as stack:
			stack.enter_context(self.cell.nominal_wrist_cam())
			if self.belief_cameras == "estimated" and self._cam_est is not None:
				stack.enter_context(self.cell.camera_override("top", self._cam_est))
			stack.enter_context(self._believed_brick())
			yield
		return
```

- **`_capture`**：在 `evidence != "hold"` 时，先在 `_belief()` 里取得 bbox，也就是用估计相机投影出的搜索窗口。然后在**真实图像**上调 `wrist_localize._blob(rgb, 窗口中心, 窗口半径)`，取色块质心，作为 `center_px` 传给 `detector_crop`；找不到色块时退回窗口中心，并记 `crop_source="projection"`，否则记 `crop_source="image"`。
- **`view` 事件**增加两个字段：`crop_source`，以及传感器字段 `crop_shift_px`（色块质心相对窗口中心的偏移）。
- **前提检查**：`mujoco.mj_camlight` 依赖 `xpos` 已经算好。`_believed_brick` 里调过 `mj_kinematics`，所以上下文的进入顺序必须是 `camera_override` 在前、`_believed_brick` 在后。
- **注意**：`camera_override` 依赖 `cam_bodyid == 0`。如果顶相机挂在某个 body 上，按 body 坐标换算，并在报告里注明。

- [ ] **Step 4：运行新测试和全量测试**（默认选项下旧测试必须逐位不变）

- [ ] **Step 5：提交** `obs C4: estimated belief cameras and image-centred evidence crops (G9)`

---

### Task C5：密集在途取证（`dense-10` / `dense-20`）

**Files:**
- Modify: `src/digital_twin_armigo/agent/twin_mujoco.py`（`_wrist` 的扫描循环、`sweep` 的回调参数）
- Test: `tests/test_grasp_loop.py`（追加）

**Interfaces:**
- Produces:
  - `sweep(expert, path, peak_speed, min_s, phase="approach", pause_s=None, on_progress=None)`：`on_progress(s_frac)` 每帧调用一次，`s_frac` ∈ [0, 1]；
  - 事件 `view` 增加 `in_motion: bool`，以及 `decided_after_frames`（结果生效时相对拍照帧的延迟帧数）。

**规则：**

- 沿扫描路径，每走过 `spacing_deg`（按关节弧长折算，用 `sweep` 里的 `arc`）在途拍一张。拍照帧的 TCP 速度超过 0.08 m/s 时顺延到下一帧，最多顺延 5 帧。
- 结果在 `round(detector_latency_s * FPS)` 帧之后才计入 `decision(views)`，默认 `detector_latency_s = 0.15`。一旦判为缺陷，就停止扫描、回到 ready 位。
- 路径末端照旧做一次 `_hold_servo` 拍照，作为兜底。
- `dense-*` 模式下不在中间视角停稳，`sweep_pause_s` 不生效。

- [ ] **Step 1：写失败测试**

用 C3 测试里同样的构造方式，在 `evidence="dense-20"` 下跑一次 `_wrist`，断言：

- 至少有 1 个 `view` 事件的 `in_motion` 为真；
- 每个 `in_motion` 视图的 `decided_after_frames` 等于 `round(0.15 * FPS)`；
- 同一案例在 `evidence="hold"` 下的 `view` 事件序列，与 C5 之前的提交完全相同（用 H8 的 `compare_runs` 判定）。

- [ ] **Step 2：运行，确认失败**

- [ ] **Step 3：实现**

`sweep` 已经按弧长参数化，在 `at(s)` 的产出循环里加 `on_progress(s / total)` 即可。`_wrist` 里的 `look(path)` 在 `dense` 模式下改为一次走完整条 `path`，用 `on_progress` 触发拍照。

- [ ] **Step 4：运行全量测试**

- [ ] **Step 5：提交** `obs C5: dense in-motion evidence capture with delayed detector decisions`

---

### Task C6：runner 接入变体，加标定误差注入

**Files:**
- Modify: `src/digital_twin_armigo/experiments/contract.py`（`MOTIONS = tuple(VARIANTS)`，`EVIDENCE = ("hold", "img", "dense-10", "dense-20")`，`EXPERIMENT_FIELDS` 增加 `motion`、`evidence`、`calib_offset_mm`）
- Modify: `src/digital_twin_armigo/experiments/runner.py`（参数、`group_name`、`agent_options`）
- Modify: `src/digital_twin_armigo/agent/twin_mujoco.py`（`sample_case` 的 `calib_offset_mm` 选项）
- Test: `tests/test_cv_motion_runner.py`

**Interfaces:**
- 组名：
  - 默认（`open`、`hold`、无误差注入）时与现在完全相同；
  - 否则在末尾追加 `-m<motion>`、`-e<evidence>`、`-c<mm>`，例如 `realistic-drift-B-v7bot-mw2a-eimg`、`realistic-fixed-B-v7bot-mopen-c10`。
- `calib_offset_mm = d` 的含义：顶相机实际位姿等于标称位姿；标定在一个偏离标称位姿 d mm 的位姿上完成。偏移方向由种子决定、在水平面内，另加 yaw 偏移 `d × 0.2°`。这样 0 mm 就等价于 `fixed`。
- 与 `--camera drift` 互斥。

- [ ] **Step 1：写失败测试**

```python
# tests/test_cv_motion_runner.py
import pytest

from digital_twin_armigo.experiments.runner import group_name


def test_default_group_names_unchanged():
	assert group_name("realistic", "drift", "B", "v7bot") == "realistic-drift-B-v7bot"


def test_variant_suffixes():
	assert group_name("realistic", "drift", "B", "v7bot", motion="w2a", evidence="img") == "realistic-drift-B-v7bot-mw2a-eimg"
	assert group_name("realistic", "fixed", "B", "v7bot", motion="open", calib_offset_mm=10) == "realistic-fixed-B-v7bot-mopen-c10"


def test_calib_offset_is_deterministic_and_sized():
	import numpy as np

	from digital_twin_armigo.agent.twin_mujoco import sample_case

	a = sample_case(3_000_101, positive=False, face=0, options={"cameras": "fixed", "calib_offset_mm": 10})
	b = sample_case(3_000_101, positive=False, face=0, options={"cameras": "fixed", "calib_offset_mm": 10})
	la, lb = a[0], b[0]
	assert np.allclose(la.calib_top.pos, lb.calib_top.pos)
	assert np.linalg.norm(np.asarray(la.calib_top.pos[:2]) - np.asarray(la.top.pos[:2])) * 1000 == pytest.approx(10, abs=0.01)


def test_drift_and_offset_are_exclusive():
	from digital_twin_armigo.agent.twin_mujoco import sample_case

	with pytest.raises(ValueError):
		sample_case(3_000_101, positive=False, face=0, options={"cameras": "drift", "calib_offset_mm": 5})
```

`calib_top` 和 `CameraPose` 的字段名以 `twin_mujoco.py:242` 附近现有的 drift 实现为准。断言的含义不能变，写法可以调整。

- [ ] **Step 2–4：先看测试失败，再实现，再全量通过。**`plan.json` 里要记录这三个新参数，这样换了参数的同名 batch 会被拒绝（现有的 `changed` 检查）。

- [ ] **Step 5：提交** `obs C6: runner motion/evidence variants and calibration-offset injection`

---

### Task C7：实验、选优与教师冻结提案【GPU】

**Files:**
- Create: `docs/experiments/cv_motion_prereg.md`（第 6 节判据原样抄入，并加批次名和种子段）
- Create: `scripts/experiments/cv_motion_matrix.sh`
- Create（运行产物，不提交）：`E:/Armigo_data_v3/loop/cv-motion/`

- [ ] **Step 1：先提交预注册文档，再跑任何东西。** `obs C7a: preregistration for the CV motion matrix`

- [ ] **Step 2：脚本**（示意阶段 2；阶段 1、3、4 同样写法）

```bash
#!/usr/bin/env bash
# Stage 2: grasp variants x {fixed, drift}, evidence=hold, strategy B, p2_dev12. Run only when scheduled (GPU).
set -euo pipefail
PY=E:/digital_twin_armigo/.venv/Scripts/python
export PYTHONPATH=src
for m in open w1s w2s w3s w2a w3a w4a; do
	for cam in fixed drift; do
		$PY -m digital_twin_armigo.experiments.runner run --root E:/Armigo_data_v3/loop/cv-motion \
			--batch cvm-s2 --sensing realistic --camera "$cam" --strategy B --suite p2_dev12 \
			--detector v7bot --motion "$m" --evidence hold --workers 4
	done
done
```

- [ ] **Step 3：【GPU，由用户或 Claude 排期】依次跑阶段 1–4。** 每个阶段结束后，用 `report.py --compare` 出表，再用 H2 的机制表看 drift 失败是否从 `brick` 机制转走了。
- [ ] **Step 4：写 `E:/Armigo_data_v3/loop/cv-motion/RESULT.md`**，内容包括：
  - 容差曲线；
  - 各阶段的表；
  - 按判据选出的胜者和确认阶段的 McNemar 结果；
  - `wrist_fix` 的接受率和新息分布；
  - 运动模糊风险的说明。
- [ ] **Step 5：教师冻结提案。** 胜者成立时，写一份提案，说明默认值改成什么、影响哪些计划：数据集计划 Task 10/11 的教师要换，F7 的 freeze 要重做。**不要自己改默认值**，等用户批准。提交 `obs C7b: CV motion results and teacher default proposal`

## 第二部分验收清单

- [ ] C1：合成测试通过；MuJoCo 集成检查的误差写进了提交信息。
- [ ] C2：`recheck=None` 时旧测试逐位不变；同步和异步的帧序列符合测试。
- [ ] C3：`wrist_fix` 的传感器字段通过 `assert_sensor_only`，真值字段已登记进 H1；checkpoint 恢复后闭环仍然有效。
- [ ] C4：drift 下估计相机和真实相机确实不同（G9 的测试）；默认 `actual` 下逐位不变。
- [ ] C5：`hold` 下的事件序列与之前相同（`compare_runs`）。
- [ ] C6：默认组名不变；`calib_offset_mm` 可以复现。
- [ ] C7：先提交预注册再跑；结果和提案已写；默认值没有被擅自修改。

## 执行总顺序（交给 Antigravity；已被第三部分末尾的版本取代）

1. 本文第一部分 H1–H8。
2. 本文第二部分 C1–C6；C7 的 Step 1–2 也做，Step 3 等排期。
3. [2026-11-24-paired-trajectory-governance.md](2026-11-24-paired-trajectory-governance.md) 的 P1–P6（全知与感知配对运行，以及奖章分层的数据治理）。
4. 回到数据集计划的 Task 10/11。正式采集用的教师，由 C7 的结论加上用户批准来定。

---

# 第三部分：神经网络分割积木各面并估计位姿（U1–U5）

> 在第二部分 C1–C6 验收之后开始（要用到 C1 的接口和 C7 的实验矩阵）。分支、worktree 和 Global Constraints 同前，提交信息第一行写 `obs U<N>: <内容>`。

## 7. 为什么做、做成什么样

**已定事项（2026-11-24 用户确认）**：

- 目标是把数字孪生做好。所有验收都在仿真里完成；和真机对照只用来说明孪生"够真实"，不设门槛。
- 仿真里的缺陷保持十字，不加胶带缺陷。真机数据里的缺陷是灰色胶带，这是已知的语义差距，本计划不处理。
- 不标定真实腕相机。仿真一律用 MJCF 的相机参数（`fovy` 61，640×480）。

**动机**：C1 的经典分割靠 Lab 颜色差，有三个弱点：

- 同色杂物、彩色桌面（v3 的桌面饱和度可到 0.35）和阴影；
- SO-101 腕相机斜装约 33°，hover 时侧面会进入画面，混进色块后拟合出的矩形会偏；
- 只能处理平放的积木。

网络可以把顶面和侧面分开，还能给出角点，用 PnP 解出完整姿态。仿真里逐像素的面标签是现成的：`Cell.face_visibility` 内部就是用 `_face_of[geom]` 把分割渲染映射到 6 个面，并且考虑了遮挡。

**模型**：一个小型 UNet，基础通道 24，四层，约 190 万参数。输入 320×240 RGB，有两组输出：

- **语义 4 类**：背景、顶面（积木自身 +z 面，带凸点）、底面、侧面。类别按积木自身坐标定义，所以翻转的积木也一致。
- **角点热图 2 通道**：顶面角点、底面角点。**不给角点编号**，因为积木有对称性：2x4 是 180°，2x2 是 90°。对应关系在 PnP 阶段枚举对称假设来解决，取重投影误差最小的那个；对称等价的假设，取离先验 yaw 最近的。

**位姿**：
- 默认用顶面 4 个角点加 `cv2.solvePnP(..., SOLVEPNP_IPPE)`，平面 PnP；
- 顶面角点不足 4 个时，用顶面掩码的轮廓多边形补齐；
- 有可见的底面角点时，用全部角点做 RANSAC-PnP，得到 6D 姿态（积木翻倒或斜靠时用）。
- 输出复用 C1 的 `WristFix`，再加 `z`、`tilt_brick_deg`、`reproj_px` 三个字段。

**运行环境**（不装包）：
- 训练在 `E:/il/lerobot-cuda`，那里有 torch；
- 导出 ONNX（`torch.onnx.export(..., dynamo=False)`，不需要 onnxscript）；
- 推理在主 `.venv`，用 OpenCV 5.0 的 `cv2.dnn.readNetFromONNX` 跑 CPU，已核实可用。
- 只用 Conv、BatchNorm、ReLU、MaxPool、最近邻 Resize、Concat 这些 `cv2.dnn` 支持的算子。
- 如果 `cv2.dnn` 加载失败，退路是仿照缺陷检测器的子进程服务（`inspection_scorer.py` 的做法），放在 lerobot-cuda 里跑，并在报告里写明。**不要为此安装包。**

**判据**（U4 之前写进 `docs/experiments/brickseg_prereg.md` 并提交）：

| 指标 | 集合 | 通过线 |
|---|---|---|
| 顶面 IoU | sim-val | ≥ 0.90 |
| 可见角点 PCK@2px（320×240） | sim-val | ≥ 0.90 |
| 腕部位姿误差中位数 | sim-val | xy ≤ 1.5 mm，yaw ≤ 1.5° |
| 定位成功率 | sim-val | ≥ 98% |
| 定位成功率和误差 | sim-ood-appearance | 不差于 C1 经典方法，配对比较 |

**判据全部在仿真里**：这张表不含真机指标。U5 只是一次外观对照，只出图，不设门槛。

**是否接入教师，由 C7 加跑的阶段 3b 决定，不由上面这张表决定**：最好的抓取变体 × `segmenter` ∈ {`lab`, `unet`} × {fixed, recal, drift}，判据沿用第 6 节。

## 第三部分文件结构

| 文件 | 改动 | 职责 |
|---|---|---|
| `src/digital_twin_armigo/mujoco_cell/cell.py` | 改 | `face_class_map(camera, data=None) -> np.ndarray[int8]`（−1 背景，0 顶，1 底，2 侧；训练时 +1） |
| `src/digital_twin_armigo/perception_nn/__init__.py` | 新 | 包（不导入 torch） |
| `src/digital_twin_armigo/perception_nn/labels.py` | 新 | `brick_corners_world`、`corner_labels`（投影和可见性），用于生成训练标签 |
| `src/digital_twin_armigo/perception_nn/model.py` | 新 | `BrickSegUNet`（torch，只在 lerobot-cuda 里导入） |
| `src/digital_twin_armigo/agent/brick_pose_nn.py` | 新 | `BrickSegNet`（cv2.dnn 推理）、`peaks`、`pose_from_keypoints`、`localize_with_net` |
| `src/digital_twin_armigo/agent/wrist_localize.py` | 改 | `localize_from_wrist(..., segmenter="lab", net=None)` |
| `scripts/perception/make_brickseg_dataset.py` | 新 | 用 MuJoCo 生成数据分片（CPU） |
| `scripts/perception/train_brickseg.py` | 新 | 训练加导出 ONNX【GPU】 |
| `scripts/perception/make_tiny_onnx_fixture.py` | 新 | 生成测试用的微型 ONNX（CPU，lerobot-cuda） |
| `scripts/perception/eval_brick_pose.py` | 新 | 离线评测：经典方法对比网络 |
| `scripts/perception/realism_compare.py` | 新 | 外观对照：真机帧与同视角仿真渲染的并排图、网络在真机帧上的叠加图（只出图） |
| `tests/test_brickseg_labels.py`、`tests/test_brick_pose_nn.py`、`tests/data/brickseg_tiny.onnx`、`tests/data/brickseg_tiny.json` | 新 | 测试与微型模型 |

---

### Task U1：逐面标签、角点标签与数据生成

**Files:**
- Modify: `src/digital_twin_armigo/mujoco_cell/cell.py`（`face_class_map`）
- Create: `src/digital_twin_armigo/perception_nn/__init__.py`, `src/digital_twin_armigo/perception_nn/labels.py`, `scripts/perception/make_brickseg_dataset.py`
- Test: `tests/test_brickseg_labels.py`

**Interfaces:**
- Produces:
  - `Cell.face_class_map(camera, data=None) -> np.ndarray(h, w) int8`：取值 −1、0、1、2；考虑机械臂遮挡；需要用 `face_probes=True` 构建的 cell。
  - `brick_corners_world(pos, rot, size) -> np.ndarray(8, 3)`：前 4 个是顶面（+z）角点，按逆时针排列。
  - `corner_labels(corners_w, k, world_from_cam, depth) -> np.ndarray(8, 3)`：每行是 `(u, v, visible)`。投影点的深度不超过深度图读数加 2 mm 时，记为可见。
  - 数据分片 `shard_XXXX.npz`：`rgb (N,240,320,3) uint8`、`cls (N,240,320) uint8`（0 背景、1 顶、2 底、3 侧）、`corners (N,8,3) float32`（坐标已经缩放到 320×240）、`meta`（JSON 字符串列表：seed、split、camera、view_kind、brick pose、K、world_from_cam、brick_size）。

**尺寸约定**：`layout.brick_size` 到底是全长还是半长，以 `scene.py` 里生成积木 geom 的代码为准（MuJoCo 的 box geom 用半长）。`brick_corners_world` 统一接收全长，在函数内部除以 2。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_brickseg_labels.py
import math

import numpy as np

from digital_twin_armigo.perception_nn.labels import brick_corners_world, corner_labels


def test_corners_order_and_extent():
	rot = np.array([[math.cos(0.3), -math.sin(0.3), 0], [math.sin(0.3), math.cos(0.3), 0], [0, 0, 1]])
	c = brick_corners_world(np.array([0.3, 0.0, 0.01]), rot, (0.032, 0.016, 0.0192))
	assert c.shape == (8, 3)
	assert np.allclose(c[:4, 2], 0.01 + 0.0096) and np.allclose(c[4:, 2], 0.01 - 0.0096)
	top = c[:4, :2] - c[:4, :2].mean(axis=0)
	cross = np.cross(top[0], top[1])
	assert cross > 0  # counter-clockwise seen from above
	assert np.isclose(np.linalg.norm(c[0] - c[1]), 0.032) or np.isclose(np.linalg.norm(c[1] - c[2]), 0.032)


def test_corner_visibility_uses_depth():
	k = np.array([[400.0, 0, 319.5], [0, 400.0, 239.5], [0, 0, 1]])
	w2c = np.eye(4)
	w2c[:3, :3] = np.diag([1.0, -1.0, -1.0])
	w2c[:3, 3] = (0.3, 0.0, 0.3)
	c = brick_corners_world(np.array([0.3, 0.0, 0.0096]), np.eye(3), (0.032, 0.016, 0.0192))
	depth = np.full((480, 640), 0.3 - 0.0192)  # a plane at the top face: bottom corners are hidden
	lab = corner_labels(c, k, w2c, depth)
	assert lab[:4, 2].all() and not lab[4:, 2].any()
	assert np.allclose(lab[0, :2], [319.5 + 400 * 0.016 / (0.3 - 0.0192), 239.5 - 400 * 0.008 / (0.3 - 0.0192)], atol=0.5)


def test_face_class_map_matches_face_visibility(tmp_path):
	"""Pixel counts per class agree with Cell.face_visibility on a rendered v3 case (CPU MuJoCo)."""
	from digital_twin_armigo.perception_nn.labels import _demo_cell  # builds a face-probe cell like the twin does

	cell = _demo_cell(3_000_101)
	vis = cell.face_visibility("top")
	cls = cell.face_class_map("top")
	assert (cls == 0).sum() == vis["faces"][4]
	assert (cls == 1).sum() == vis["faces"][5]
	assert (cls == 2).sum() == vis["faces"][:4].sum()
```

`_demo_cell(seed)` 照抄 `twin_mujoco.py` 建 cell 的方式（`Cell(layout, menagerie, ..., face_probes=True)`），放在 `labels.py` 里，只供测试和数据生成使用。

- [ ] **Step 2：运行，确认失败**

- [ ] **Step 3：实现**

```python
# src/digital_twin_armigo/perception_nn/labels.py (core)
"""Training labels for the brick segmentation network, computed from simulator truth (never used by the agent)."""

from __future__ import annotations

import numpy as np

TOP_XY = ((1, 1), (-1, 1), (-1, -1), (1, -1))  # counter-clockwise seen from +z


def brick_corners_world(pos, rot, size) -> np.ndarray:
	hx, hy, hz = (s / 2 for s in size)
	local = [(sx * hx, sy * hy, hz) for sx, sy in TOP_XY] + [(sx * hx, sy * hy, -hz) for sx, sy in TOP_XY]
	return np.asarray(local) @ np.asarray(rot).T + np.asarray(pos)


def corner_labels(corners_w, k, world_from_cam, depth, tol_m=0.002) -> np.ndarray:
	c = np.linalg.inv(world_from_cam)
	p = corners_w @ c[:3, :3].T + c[:3, 3]
	uv = (p[:, :2] / p[:, 2:3]) @ k[:2, :2].T + k[:2, 2]
	h, w = depth.shape
	out = np.zeros((8, 3), np.float32)
	for i, ((u, v), z) in enumerate(zip(uv, p[:, 2], strict=True)):
		ui, vi = round(u), round(v)
		inside = z > 0 and 0 <= ui < w and 0 <= vi < h
		out[i] = (u, v, float(inside and z <= depth[vi, ui] + tol_m))
	return out
```

`Cell.face_class_map` 复用 `face_visibility` 里那两次分割渲染和 `_drop_specks`，把 `_face_of[geom]` 按 4→0、5→1、0–3→2 映射，其余像素记 −1。抽一个私有函数让两处共用，不要复制代码。

**`make_brickseg_dataset.py`**（CPU，最多 4 个进程）：

- 案例：`sample_case(seed, positive, face, options={"scene": "v3", "cameras": "wide"})`，种子取自 v3 的 train、val、ood-appearance 段（段定义见数据集计划第 1 节）。正负样本和缺陷面随机。
- 每个案例产出 6 张图：
  - 腕部 4 张：用 `Kinematics` 加 IK，把 TCP 放到积木上方。水平偏移 ±3 cm，高度 0.04–0.14 m，roll ±90°；
  - 顶相机 1 张：积木在桌上；
  - 顶相机 1 张：把积木放到展示高度，姿态随机（30% 的概率翻转或侧放），机械臂停在附近的展示位作为遮挡。
- 只用 `mj_kinematics` 和 `mj_forward` 设置状态，不跑物理。积木状态直接写 freejoint。
- 渲染 640×480：RGB、深度、`face_class_map`。深度只用来判断角点可见性，不进数据。然后缩放到 320×240：RGB 用 `INTER_AREA`，类别图用 `INTER_NEAREST`，角点坐标乘 0.5。
- 积木在画面里少于 150 个像素的帧丢弃，并计数。
- 规模：train 2,000 个案例（12,000 帧），val 250 个案例，ood-appearance 200 个案例。
- 开发阶段输出到 `C:/scratch/brickseg-dev/`；正式输出到 `E:/Armigo_data_v3/perception/brickseg-v1/`，并写 `manifest.json`（种子段、帧数、丢弃数、提交号、各分片的 sha256）。

- [ ] **Step 4：运行测试；生成一个 50 案例的冒烟分片，抽 10 帧把类别图叠在 RGB 上存成 PNG，放到 `C:/scratch/brickseg-dev/preview/`，供人工查看**
- [ ] **Step 5：提交** `obs U1: per-face class maps, corner labels and the MuJoCo brick segmentation dataset generator`

正式数据（2,450 个案例）只用 CPU 就能生成，冒烟通过后可以直接跑。

---

### Task U2：模型、训练与 ONNX 导出

**Files:**
- Create: `src/digital_twin_armigo/perception_nn/model.py`, `scripts/perception/train_brickseg.py`, `scripts/perception/make_tiny_onnx_fixture.py`, `tests/data/brickseg_tiny.onnx`, `tests/data/brickseg_tiny.json`
- Test: `tests/test_brick_pose_nn.py`（模型部分用 `pytest.importorskip("torch")`）

**Interfaces:**
- Produces:
  - `BrickSegUNet(base=24, classes=4, heat=2)`：`forward(x[B,3,240,320]) -> [B,6,240,320]`，前 4 个通道是类别 logits，后 2 个是热图 logits；
  - 元数据 JSON：`input_hw`、`mean`、`std`、`classes`、`heat_channels`、`data_manifest_sha`、`git_commit`、`val_metrics`。

```python
# src/digital_twin_armigo/perception_nn/model.py
"""Small UNet: brick face classes (bg/top/bottom/side) and unordered top/bottom corner heatmaps.

Only ops that OpenCV's dnn module imports from ONNX: Conv, BatchNorm, ReLU, MaxPool, nearest Resize, Concat."""

import torch
from torch import nn


def block(cin, cout):
	return nn.Sequential(
		nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
		nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
	)


class BrickSegUNet(nn.Module):
	def __init__(self, base=24, classes=4, heat=2):
		super().__init__()
		c = [base, base * 2, base * 4, base * 8]
		self.enc = nn.ModuleList([block(3, c[0]), block(c[0], c[1]), block(c[1], c[2]), block(c[2], c[3])])
		self.pool = nn.MaxPool2d(2)
		self.up = nn.Upsample(scale_factor=2, mode="nearest")
		self.dec = nn.ModuleList([block(c[3] + c[2], c[2]), block(c[2] + c[1], c[1]), block(c[1] + c[0], c[0])])
		self.head = nn.Conv2d(c[0], classes + heat, 1)

	def forward(self, x):
		skips = []
		for i, enc in enumerate(self.enc):
			x = enc(x if i == 0 else self.pool(x))
			skips.append(x)
		x = skips.pop()
		for dec in self.dec:
			x = dec(torch.cat([self.up(x), skips.pop()], dim=1))
		return self.head(x)
```

**训练**（`train_brickseg.py`，在 lerobot-cuda 里跑）：

- 损失：
  - 类别交叉熵，按训练集像素频率取逆频率权重，上限 10；
  - 热图用 focal loss（CenterNet 形式，α=2、β=4），目标是 σ=1.5 px 的高斯，只放可见角点；
  - 两项权重为 1 : 1。
- 优化：AdamW，学习率 1e-3，余弦退火，批大小 32，30 个 epoch，bf16。
- 增强：亮度和对比度 ±20%、色相 ±0.03、高斯噪声、JPEG 质量 60–95。真机是 MJPEG，所以要加 JPEG 增强。几何增强只用水平翻转，并同步翻转角点。
- 按 val 上的"顶面 IoU 加 PCK@2px"选最佳 epoch；导出 ONNX（opset 17）和元数据 JSON。
- 正式产物写到 `E:/Armigo_data_v3/perception/brickseg-v1/model/`，**不提交仓库**。

- [ ] **Step 1：写测试**
  - torch 部分（在 lerobot-cuda 里用小脚本跑）：前向形状是 `[2,6,240,320]`，参数量在 150 万到 250 万之间，ONNX 能导出。
  - 主 `.venv` 部分：`cv2.dnn.readNetFromONNX("tests/data/brickseg_tiny.onnx")` 能跑出形状为 `(1,6,240,320)` 的输出。
- [ ] **Step 2：在 lerobot-cuda 里用 CPU 跑 `make_tiny_onnx_fixture.py`**：`base=4`，随机权重，种子 0。导出文件不超过 200 KB，连同 JSON 一起提交。
- [ ] **Step 3：实现训练脚本。** 用 `--epochs 1 --limit 64 --device cpu` 冒烟跑通。
- [ ] **Step 4：提交** `obs U2: brick segmentation UNet, training script and ONNX fixture`
- [ ] **Step 5【GPU，由用户或 Claude 排期】：正式训练。** 预计 1–2 小时，结果写进 `model/train_report.md`。

---

### Task U3：推理与位姿求解（主 `.venv`，无 torch）

**Files:**
- Create: `src/digital_twin_armigo/agent/brick_pose_nn.py`
- Modify: `src/digital_twin_armigo/agent/wrist_localize.py`（`segmenter` 参数）
- Test: `tests/test_brick_pose_nn.py`

**Interfaces:**
- Produces:
  - `BrickSegNet(onnx_path, meta_path)`：`predict(rgb) -> NetOut(cls, heat, scale)`；`cls` 是 (240, 320)，`heat` 是 (2, 240, 320)，已经过 sigmoid；`scale` 是 (sx, sy)，用于换算回原图坐标。
  - `peaks(heat, thr=0.3, radius=2) -> np.ndarray(n, 3)`：每行 `(u, v, score)`，原图坐标，3×3 加权质心做亚像素细化。
  - `pose_from_keypoints(top_px, bottom_px, top_mask, k, world_from_cam, size, prior_yaw) -> tuple[PoseFit | None, str | None]`；`PoseFit(pos, rot, yaw, tilt_deg, reproj_px, n_points, mode)`，`mode` ∈ {`ippe4`, `mask4`, `ransac6d`}。
  - `localize_with_net(net, rgb, k, world_from_cam, prior_xy, prior_yaw, size, *, gate_m=0.04) -> tuple[WristFix | None, str | None]`：用和 C1 相同的门限，结果填进 `WristFix`。
  - C1 的 `localize_from_wrist(..., segmenter="lab", net=None)`：`segmenter="unet"` 时转调 `localize_with_net`。

**规则：**

- **候选**：只用落在先验门限内、而且属于"离先验最近的那块顶面掩码连通域"附近（3 px 以内）的角点峰。
- **假设枚举**：顶面峰按图像中的角度排序，取 2 种方向 × 4 种循环起点，共 8 个假设。每个假设用 `SOLVEPNP_IPPE` 求解，物体点是积木坐标系里 z=+h/2 的顶面角点（顺序同 `TOP_XY`）。
- **筛选**：去掉积木 +z 轴与世界 z 轴夹角超过 60° 的解（平放模式下）。剩下的取重投影误差最小的；误差相差不到 0.5 px 的对称孪生解，取 yaw 离先验最近的。
- **补点**：顶面峰少于 4 个时，对顶面掩码的轮廓用 `approxPolyDP` 逼近成 4 个顶点再求解，`mode="mask4"`。
- **6D**：同时有 ≥ 2 个底面峰时，额外用 `solvePnPRansac` 加全部角点求 6D 解，`mode="ransac6d"`，并报告 `tilt_deg`。
- **拒绝条件**：`reproj_px > 2.5`，或解出的积木中心离先验超过门限。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_brick_pose_nn.py (pose part; no network needed)
import math

import numpy as np

from digital_twin_armigo.agent.brick_pose_nn import peaks, pose_from_keypoints
from digital_twin_armigo.perception_nn.labels import brick_corners_world

SIZE = (0.032, 0.016, 0.0192)


def _cam(tilt_deg=33.0):
	t = math.radians(tilt_deg)
	tilt = np.array([[1, 0, 0], [0, math.cos(t), -math.sin(t)], [0, math.sin(t), math.cos(t)]])
	w2c = np.eye(4)
	w2c[:3, :3] = np.diag([1.0, -1.0, -1.0]) @ tilt
	w2c[:3, 3] = (0.30, -0.09, 0.15)
	return np.array([[407.0, 0, 319.5], [0, 407.0, 239.5], [0, 0, 1]]), w2c


def _project(k, w2c, pts):
	c = np.linalg.inv(w2c)
	p = pts @ c[:3, :3].T + c[:3, 3]
	return (p[:, :2] / p[:, 2:3]) @ k[:2, :2].T + k[:2, 2]


def test_planar_pose_from_four_unordered_top_corners():
	k, w2c = _cam()
	yaw = math.radians(25)
	rot = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
	corners = brick_corners_world(np.array([0.305, 0.004, 0.0096]), rot, SIZE)
	top = _project(k, w2c, corners[:4]) + np.random.default_rng(0).normal(0, 0.3, (4, 2))
	top = top[[2, 0, 3, 1]]  # unordered
	fit, why = pose_from_keypoints(top, np.zeros((0, 2)), None, k, w2c, SIZE, prior_yaw=math.radians(20))
	assert why is None and fit.mode == "ippe4"
	assert np.linalg.norm(fit.pos[:2] - [0.305, 0.004]) < 1e-3
	assert abs(math.degrees(fit.yaw) - 25) < 1.0


def test_symmetric_twin_resolved_by_prior():
	k, w2c = _cam()
	corners = brick_corners_world(np.array([0.30, 0.0, 0.0096]), np.eye(3), SIZE)
	top = _project(k, w2c, corners[:4])
	fit, _ = pose_from_keypoints(top, np.zeros((0, 2)), None, k, w2c, SIZE, prior_yaw=math.pi + 0.05)
	assert abs(math.degrees(fit.yaw) - 180) < 1.0


def test_peaks_subpixel():
	heat = np.zeros((2, 240, 320), np.float32)
	yy, xx = np.mgrid[0:240, 0:320]
	heat[0] = np.exp(-((xx - 100.4) ** 2 + (yy - 50.6) ** 2) / (2 * 1.5**2))
	p = peaks(heat[0], scale=(2.0, 2.0))
	assert len(p) == 1 and abs(p[0, 0] - 200.8) < 0.3 and abs(p[0, 1] - 101.2) < 0.3
```

另外两条测试：

- **ONNX 包装器**：用微型 fixture 跑通 `BrickSegNet.predict`，检查输出形状和 `scale`。
- **mask4 退路**：只给 2 个角点，再给一张顶面掩码，用 `cv2.fillPoly` 画出投影出的顶面四边形，断言 `mode == "mask4"`，xy 误差 < 2 mm。

`peaks` 的签名是 `peaks(heat, thr=0.3, radius=2, scale=(1.0, 1.0))`，以测试里的写法为准。

- [ ] **Step 2：运行，确认失败**
- [ ] **Step 3：实现**
- [ ] **Step 4：运行全量测试；`segmenter="lab"`（默认）时，C1 和 C3 的测试逐位不变**
- [ ] **Step 5：提交** `obs U3: ONNX inference, corner peaks and symmetric PnP brick pose`

---

### Task U4：离线评测、接入 twin 和 runner、阶段 3b

**Files:**
- Create: `scripts/perception/eval_brick_pose.py`, `docs/experiments/brickseg_prereg.md`
- Modify:
  - `src/digital_twin_armigo/agent/twin_mujoco.py`（选项 `segmenter`；`_wrist_look` 把它传下去；网络只加载一次）
  - `src/digital_twin_armigo/experiments/runner.py`、`contract.py`（`--segmenter`，组名后缀 `-s<seg>`，默认 `lab` 时不加后缀）
  - `scripts/experiments/cv_motion_matrix.sh`（阶段 3b）
- Test: `tests/test_cv_motion_runner.py`（追加组名测试）、`tests/test_grasp_loop.py`（`segmenter="unet"` 加微型 fixture：只验证流程能走通，fixture 是随机权重，预期 `found=False`）

**`eval_brick_pose.py`**：

- 在 U1 的 val 和 ood-appearance 分片上，逐帧跑三种方法：
  - `lab`：C1；
  - `unet-rect`：网络的顶面掩码，加 C1 的矩形拟合；
  - `unet-pnp`：U3。
- 先验取真值加一个随机扰动（xy ±15 mm，yaw ±15°），模拟顶视估计。种子固定。
- 输出到 `E:/Armigo_data_v3/perception/brickseg-v1/eval.md`：
  - 每种方法的定位成功率、xy 和 yaw 误差的中位数与 p90；
  - 按相机、桌面饱和度分档、是否有同色杂物、积木是否翻倒拆开报告；
  - 各类 IoU 和 PCK；
  - CPU 推理耗时；
  - 和第 7 节判据逐项比对。
- 也评顶相机。积木在手里时，额外报告手中位姿误差（相对夹爪），作为将来检测打滑的依据。这些只报告，不接入 agent。

- [ ] **Step 1：先提交 `brickseg_prereg.md`**（第 7 节的判据表，加上阶段 3b 的设置）。`obs U4a: preregistration for the brick segmentation network`
- [ ] **Step 2：写测试（组名、流程能走通），再实现接入和评测脚本，全量测试通过**
- [ ] **Step 3：提交** `obs U4b: segmenter option in the twin and runner, offline brick pose evaluation`
- [ ] **Step 4【GPU，排期】**：U2 正式模型出来之后，先跑 `eval_brick_pose.py`（CPU 即可）；再跑阶段 3b（要用检测器）。结论写进 `cv-motion/RESULT.md`，新增"segmenter"一节。

---

### Task U5：真机外观对照（只出图，不设门槛）

**Files:**
- Create: `scripts/perception/realism_compare.py`
- Test: `tests/test_brick_pose_nn.py`（追加纯函数测试）

**数据来源**（只读，不复制进仓库，不往那里写）：

- `C:/Users/Jiang/.gemini/antigravity/scratch/hackjunction/artifacts/isaac_test/real_ref/`：同一时刻的四路真机帧 PNG，`frame_meta.json` 里有关节状态；
- `C:/Users/Jiang/.gemini/antigravity/scratch/hackjunction/data/studio/af9b173a-ee59-4e24-97c8-be71dba701ce/`：35 个真机回合，LeRobot v3 格式，AV1 视频，用主 `.venv` 里的 PyAV（19.0.2）解码；
- 真机的相机和仿真不完全对应：`top_rgb_v2` 是 512×416 的 D415；`wrist` 是 640×480。

**`realism_compare.py`**：

- 从真机录像里均匀抽 24 帧：腕相机 16 帧，以抓取前和持砖时为主；顶相机 8 帧。抽帧位置由种子固定。
- 对每一帧，取它的关节状态，让仿真的机械臂摆到同样的关节角，再从同一路相机渲染一张。
  - 真机的关节约定是 LeRobot 的度数，零位在量程中点。换算到 MJCF 弧度的方法，参考 hackjunction 的 `scripts/sim/so101_sort_sim.py` 和 `src/defect_sorter/kinematics.py`。
  - 积木和碗的位置无法从真机帧得到，仿真里用默认布局。所以这里只比较相机视角、机械臂和夹爪的样子、整体色调与亮度，不比较积木位置。
  - 换算对不上时（夹爪明显不在同一位置），就不配对：只把真机帧和一张相近姿态的仿真帧并排放，并在图上标注"未配对"。
- 有 U2 正式模型时，把网络的类别图和角点峰叠到真机腕部帧上。没有正式模型时，跳过这一项，用微型 fixture 只验证流程能走通。
- 输出到 `C:/scratch/brickseg-real/compare/`：每帧一张"真机 | 仿真 | 叠加"的拼图，再加一个 `index.html` 汇总，以及 `notes.md` 模板。模板列出要人工填写的观察项：色调与亮度、反光、景深与模糊、夹爪的样子、网络在真机上明显漏检或误检的情况。
- **不计算任何真机指标，不设门槛，不拿真机帧调参。**

- [ ] **Step 1：写纯函数测试**：关节度数到 MJCF 弧度的换算，用 `frame_meta.json` 那一帧，只断言形状、范围和往返一致性；抽帧下标由种子固定；拼图尺寸正确。然后实现脚本，并用微型 fixture 和 `real_ref` 那一帧冒烟跑通。提交 `obs U5: realism side-by-side comparison against existing real frames`
- [ ] **Step 2【U2 正式模型出来之后，CPU 即可】**：跑完整的 24 帧，结果路径写进 `eval.md`。观察记录由用户或 Claude 填写；发现明显的差距（例如色偏、焦距不对），作为下一轮外观随机化的输入写进报告。

## 第三部分验收清单

- [ ] U1：`face_class_map` 的各类像素数与 `face_visibility` 一致；冒烟预览图已生成；正式数据集的 manifest 带哈希。
- [ ] U2：微型 ONNX 能被 `cv2.dnn` 加载；训练脚本在 CPU 上冒烟跑通；正式训练等排期。
- [ ] U3：合成数据上的 PnP 测试（含对称性和 mask4 退路）通过；默认 `lab` 下逐位不变。
- [ ] U4：先提交预注册再评测；`eval.md` 与判据逐项比对；阶段 3b 等排期。
- [ ] U5：拼图和 `index.html` 已生成；没有计算任何真机指标；真机图片没有进仓库。

## 执行总顺序（更新，取代第二部分末尾的那一节）

1. 第一部分 H1–H8。
2. 第二部分 C1–C6；C7 的 Step 1–2。
3. 第三部分 U1–U5 的代码部分；U1 的正式数据可以直接生成（CPU）。
4. [配对与治理计划](2026-11-24-paired-trajectory-governance.md) P1–P6。
5. 【GPU 排队，由用户或 Claude 安排】：C7 阶段 1–4 → U2 正式训练 → U4 评测和阶段 3b → P6 容量测试。
6. 数据集计划 Task 10/11。正式采集用的教师，由 C7 和 U4 的结论加上用户批准来定。
