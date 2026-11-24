# 全知与感知配对轨迹：并行运行与奖章分层数据治理 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 同一个场景同时产出全知（truth）和感知（realistic）两条轨迹；把这些配对轨迹按奖章模型（bronze → silver → gold）管理起来，让训练集、失败库、评测集都从同一份可追溯的数据里确定性地生成。

**Architecture:**
- **运行**：配对靠确定性的 runner 完成，不另加一个 LLM 外层 agent。同一套件的 truth 组和 realistic 组作为普通任务进同一个进程池，事后按 `pair_key` 连接。LangGraph 继续只做每个回合内部的状态图。
- **数据**：
  - bronze：原始回合目录加哈希登记，只追加；
  - silver：逐回合的校验与归一化，加配对表；
  - gold：按用途划分的数据产品，每个产品是一份带版本号的 manifest，由 silver 上的谓词确定性地生成。

**Tech Stack:** Python 3.14，SQLite，numpy，pytest，ruff（tab 缩进）。

**Spec:** 本文第 0–2 节。上游：
- [2026-11-24-failure-observability-harness-learning.md](2026-11-24-failure-observability-harness-learning.md)：H2 机制列、H5 `tool_call`、H6 oracle、H8 `compare_runs`、第二部分的运动变体；
- [2026-11-24-smolvla-static-dataset.md](2026-11-24-smolvla-static-dataset.md)：种子段、Task 9 导出、Task 10/11 采集；
- [2026-11-24-agent-failure-loop.md](2026-11-24-agent-failure-loop.md)：F1 规则、F2 台账、F3 复现库、F7 冻结。

## Global Constraints

- **起点**：观测计划第一、第二部分（H1–H8、C1–C6）验收之后，在同一分支 `feat/failure-observability`、同一 worktree `E:/digital_twin_armigo-obs` 上继续做。提交信息第一行写 `gov P<N>: <内容>`，以 `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` 结尾。不推送，不合并。
- **环境**：测试用 `E:/digital_twin_armigo/.venv/Scripts/python -m pytest`，脚本加 `PYTHONPATH=src`。不安装包。标【GPU】的步骤由用户或 Claude 排期。
- **bronze 不可变**：已登记的回合目录不删、不改。哈希变了，就记一条完整性事件，并把该回合隔离；不覆盖原登记。原始 mkv 只在导出校验通过、并且已登记删除事件之后才删（沿用数据集计划第 6 节）。
- **gold 只由谓词生成**：任何 gold manifest 都由 `governance/gold.py` 里的谓词从 silver 生成，不手工增删条目。要排除某个回合，就在 silver 里加原因列。
- **隔离硬约束**（测试守护）：
  - 封存集（v3-test 种子段）只进 `eval_*` 产品，而且只在 freeze 之后；
  - oracle 回合（H6）和隔离回合永远不进训练产品；
  - realistic 成功但 truth 失败（`realistic_only`）的回合不进训练产品；
  - 训练产品只含 train 种子段，用 `dagger.assert_train_layouts_only` 再守一次。
- **真值只给评估器**：silver 可以存真值诊断列，用于归因；导出给学生的数据仍按 `FORBIDDEN_DIAGNOSTIC_KEYS` 和 `FORBIDDEN_CAMERA_NAMES` 过滤。
- **数据根**：`E:/Armigo_data_v3/governance/`。`lake.sqlite` 存 bronze 和 silver；gold 放在 `gold/<product>/<version>/`。`D:/Armigo_data_v2` 只读，可以作为 bronze 来源登记，但不往那里写。

---

## 0. 运行方式：要不要一个 LangGraph 外层 agent 同时管两路？

**结论：不要。用确定性的 runner 把两路当成两个普通任务并行跑，事后按 `pair_key` 连接。**

理由（2026-11-24 核实）：

1. **这两路本来就不需要 LLM。**
   - truth 组读仿真状态，规则规划；realistic 的策略 A、B 也是纯规则（runner 里 `use_llm=False`）；
   - 只有策略 C 在复核时调用 Qwen，每次请求一张图；
   - 第二部分把抓取和取证都脚本化以后，LLM 只剩"可选的复核顾问"这一个角色。
   - 外层再放一个 LLM agent 去调度两路，只会增加延迟和不确定性，调度本身没有需要推理的地方。
2. **两路之间没有在线交互的必要。** realistic 回合已经在事件里记录了真值诊断（`error_xy_mm` 等），所以"边跑边对比"能拿到的信息，事后对比都能拿到。锁步运行反而会把两路耦合起来：一路卡住，另一路跟着等；一路崩溃，另一路要一起重跑。
3. **runner 已经具备需要的能力**：进程池（`--workers`）、断点续跑、每次尝试单独建目录、Isaac 强制单进程、`twin_pool.ScorerBroker` 让所有 worker 共用一个检测器子进程。

**性能会不会垮。** 先区分"三个 agent 会话"指什么：

| 资源 | 现状 | 配对后的变化 | 风险 |
|---|---|---|---|
| vLLM（Qwen，`gpu_util` 0.76，权重 19.8 GiB） | 只有策略 C 的复核会请求 | truth 和 A/B 不增加任何请求；跑 C 时并发 2–4 个会话，靠 vLLM 的连续批处理消化 | 低。并发上去后单请求延迟会变长，仿真 worker 在等结果时占着进程槽，墙钟变长；图像 token 占 KV cache，可以用 `--max-num-seqs` 限制 |
| 检测器（单个子进程，GPU） | 所有 worker 排队用它 | 配对后请求量翻倍 | 中。这是最可能先饱和的点，`detector.elapsed_ms` 和排队时间要一起看 |
| 显存 | MuJoCo 回合时有余量；Isaac 加检测器加 Qwen 同驻时峰值 31.9 / 32.6 GB | Isaac 配对只能串行 | 高。Isaac 和 Qwen 同驻时不能再加进程，runner 已强制 Isaac 单进程 |
| CPU（MuJoCo 物理与渲染） | `--workers 4` | 配对后任务数翻倍，每个任务的负载不变 | 低。按物理核数加 worker，直到检测器排队时间开始上涨 |

所以真正需要实测的是**检测器排队**和 **Isaac 显存**，不是 vLLM。P6 有一个容量测试任务：workers 取 2、4、6、8，跑 16 对，记录回合/小时、检测器平均排队时间、CPU 占用，取拐点作为默认值。

## 1. 为什么不直接用"都过 = gold，一个过或用了回退 = silver，有失败 = bronze"

这个分法的方向对：按两路的结果给配对分类。但直接当作奖章层用有三个问题：

1. **奖章层是加工阶段，不是质量档位。** bronze 是原始落盘，silver 是校验和归一化，gold 是面向用途的数据产品。把"失败"放进 bronze，等于把最有价值的数据标成了最低档：
   - `truth_only`（truth 成功、realistic 失败）恰好隔离出感知缺口，失败闭环、反事实复跑、DAgger 都靠它；
   - 失败库和诊断基准（H7）也都是 gold 级别的产品。
2. **两条规则有重叠，还漏了一类。**
   - "一个通过"和"有一个失败"描述的是同一种情况；
   - realistic 成功而 truth 失败，在物理上不合理：全知应该是上限。这类回合多半说明 harness 有缺陷（真值规划器的 bug、评分错误）或者是运气，应该隔离复查，不能进 silver 或 gold。
3. **"回退"要分两种。**
   - 真值兜底：数据集计划已经规定教师不用真值兜底，出现就是违规，直接隔离。
   - 正常恢复（回退、重抓、放下重扫）：恢复后成功的回合可以训练，恢复片段本身还是纠偏样本。它应该是一个标记列，由训练决定权重，而不是降一档。

**改进后的方案**：质量结论放在 silver 的列上（配对结果、标记、隔离原因）；gold 按"拿去做什么"划分产品，每个产品写清楚谓词。

## 2. 设计

### 2.1 三层

| 层 | 内容 | 存放 | 可变性 |
|---|---|---|---|
| bronze | 每个回合目录一行：路径、各文件的 sha256、写入时间、提交号、freeze id、来源批次 | `lake.sqlite.bronze` | 只追加；哈希变了就记 `integrity_event` |
| silver | `silver_episode`：一回合一行，包括评分行（`scoring.score_episode`）、H2 机制、H6 反事实结论、最新审计裁定、种子段、各项检查结果、隔离原因。`silver_pair`：一个 `pair_key` 一行 | `lake.sqlite` | 可以从 bronze 加上代码完全重建；每次构建记 `build_id` |
| gold | 数据产品的 manifest 和数据卡 | `gold/<product>/<version>/manifest.jsonl`、`card.md` | 版本号取 manifest 内容哈希；生成后不改 |

### 2.2 配对结果（`silver_pair.outcome`）

**配对键**：`pair_key = (engine, case_id, detector, strategy, motion, evidence)`。

- truth 一侧固定用 fixed 相机。truth 的组名里没有相机字段（`group_name` 对 truth 忽略 camera）。
- realistic 一侧每种相机条件各成一对，都对同一个 truth 回合。

| realistic \ truth | 成功 | 失败 |
|---|---|---|
| 成功 | `both_pass` | `realistic_only`（隔离复查） |
| 失败 | `truth_only`（感知缺口） | `both_fail`（规划、物理或场景本身的问题） |

另有两个取值：

- `void`：任一侧 `infra_failed`，或任一侧被隔离。这不算结果，应该重跑。
- `unpaired`：只有一侧。

**标记列**：`recovered`（`n_recoveries > 0`）、`review_used`、`llm_used`、`oracle`、`truth_fallback`（必须为假，否则隔离）、`outcome_source`（`replay` 或 `verdict`）、`superseded`（已有更新的尝试）。

### 2.3 gold 产品

| 产品 | 谓词（全部要求未隔离、非 oracle，并且是最新一次尝试） | 用途 |
|---|---|---|
| `student_train_v3` | realistic；train 段；`physical_success`；配对结果 `both_pass`，或者无配对但所在批次 `realistic_only` 率 < 1%（见 2.4）；教师提交号等于 freeze 中的提交号 | SmolVLA 训练（数据集 Task 11） |
| `privileged_demos_v3` | truth；train 段；配对结果 `truth_only` | 难场景的特权示范。默认**不**混入训练，是否混入由消融实验决定 |
| `failure_bank` | realistic；配对结果 `truth_only` 或 `both_fail`；不含封存段 | F2 台账、F3 复现库、H7 诊断基准 |
| `harness_review` | 配对结果 `realistic_only`；或任何隔离回合；或同一 `pair_key` 连续 2 次 `void` | 人工复查队列 |
| `eval_<suite>` | 对应的评测种子段；封存段要求 freeze 文件存在 | 报告（只读，永不进训练） |

### 2.4 已定事项（2026-11-24 用户确认）

- **配对比例（已定）**：
  - 试采 200 个回合全部配对；之后按 25% 抽样配对。
  - 抽样配对时，如果某批次的 `realistic_only` 率 ≥ 1%，该批次所有无配对回合都暂不进入 `student_train_v3`，等补跑 truth 后再定。
  - 理由：truth 回合的仿真时间和 realistic 差不多，全配对会让采集时间接近翻倍。
- **VLA 训练暂不考虑（已定）**：
  - `student_train_v3` 和 `privileged_demos_v3` 照常按谓词生成，作为数据资产保留；
  - 回合在训练里的权重（包括 `recovered` 回合）、特权示范要不要混入训练，这两个问题都不在本计划范围内，留到重新启动 VLA 训练时再定；
  - 当前优先的产品是 `failure_bank`、`harness_review` 和 `eval_*`。

## 文件结构

| 文件 | 改动 | 职责 |
|---|---|---|
| `src/digital_twin_armigo/governance/__init__.py` | 新 | 包 |
| `src/digital_twin_armigo/governance/pairs.py` | 新 | `sensing_of(group)`、`pair_key(row)`、`pair_outcome(truth, real)`、`flags(row)` |
| `src/digital_twin_armigo/governance/lake.py` | 新 | SQLite 模式、连接、`build_id` |
| `src/digital_twin_armigo/governance/bronze.py` | 新 | `register(root)`：逐目录登记哈希，只追加 |
| `src/digital_twin_armigo/governance/silver.py` | 新 | `CHECKS`、`build_silver(lake, roots)`、隔离 |
| `src/digital_twin_armigo/governance/gold.py` | 新 | `PRODUCTS`、`build_product(lake, name)`、manifest、数据卡、守护 |
| `src/digital_twin_armigo/governance/segments.py` | 新 | 种子段到 split 的映射（读 `make_suites.py` 的 manifest） |
| `scripts/governance/lake.py` | 新 | CLI：`register`、`silver`、`gold <product>`、`report` |
| `scripts/governance/capacity_bench.sh` | 新 | 【GPU】容量测试 |
| `src/digital_twin_armigo/experiments/runner.py` | 改 | `run-pair`：同一套件依次规划 truth 组和 realistic 组，进同一个池 |
| `scripts/vla/collect_paired_carry.py` | 改 | 导出入口加 `--manifest`，只导出 gold manifest 里的回合 |
| `docs/GOVERNANCE.md` | 新 | 三层、产品、谓词、操作手册 |
| `tests/test_gov_pairs.py`、`tests/test_gov_bronze.py`、`tests/test_gov_silver.py`、`tests/test_gov_gold.py`、`tests/test_run_pair.py` | 新 | 各任务的测试 |

---

### Task P1：配对键、配对结果与标记

**Files:**
- Create: `src/digital_twin_armigo/governance/__init__.py`, `src/digital_twin_armigo/governance/pairs.py`
- Test: `tests/test_gov_pairs.py`

**Interfaces:**
- Produces:
  - `sensing_of(group: str) -> str`：返回 `"truth"` 或 `"realistic"`；
  - `pair_key(row: dict) -> tuple`；
  - `OUTCOMES = ("both_pass", "truth_only", "realistic_only", "both_fail", "void", "unpaired")`；
  - `pair_outcome(truth: dict | None, real: dict | None) -> str`；
  - `flags(row: dict) -> dict`。
- 输入的 `row` 就是 `scoring.score_episode` 的行，外加 silver 补的 `quarantined: bool`、`oracle: bool`、`motion`、`evidence`（缺省时分别为 `"open"` 和 `"hold"`）。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_gov_pairs.py
import pytest

from digital_twin_armigo.governance.pairs import OUTCOMES, flags, pair_key, pair_outcome, sensing_of


def row(group, success=True, **kw):
	base = {
		"group": group, "engine": "mujoco", "case_id": "L3000101-N", "detector": "v7bot", "strategy": "B",
		"physical_success": success, "infra_failed": False, "quarantined": False, "oracle": False,
		"n_recoveries": 0, "n_review_views": 0, "n_llm_images": 0, "physical": {"source": "replay"},
	}
	return base | kw


def test_sensing_from_group_names():
	assert sensing_of("truth-v7bot") == "truth"
	assert sensing_of("truth-B-v7bot") == "truth"
	assert sensing_of("realistic-drift-B-v7bot-mw2a-eimg") == "realistic"
	with pytest.raises(ValueError):
		sensing_of("weird-group")


def test_pair_key_ignores_sensing_and_camera():
	t = row("truth-B-v7bot")
	r = row("realistic-drift-B-v7bot", camera_condition="drift")
	assert pair_key(t) == pair_key(r) == ("mujoco", "L3000101-N", "v7bot", "B", "open", "hold")
	assert pair_key(row("realistic-drift-B-v7bot", motion="w2a")) != pair_key(t)


@pytest.mark.parametrize(
	("t_ok", "r_ok", "want"),
	[(True, True, "both_pass"), (True, False, "truth_only"), (False, True, "realistic_only"), (False, False, "both_fail")],
)
def test_outcome_matrix(t_ok, r_ok, want):
	assert pair_outcome(row("truth-B-v7bot", t_ok), row("realistic-fixed-B-v7bot", r_ok)) == want


def test_void_and_unpaired():
	assert pair_outcome(row("truth-B-v7bot", infra_failed=True), row("realistic-fixed-B-v7bot")) == "void"
	assert pair_outcome(row("truth-B-v7bot"), row("realistic-fixed-B-v7bot", quarantined=True)) == "void"
	assert pair_outcome(None, row("realistic-fixed-B-v7bot")) == "unpaired"
	assert set(OUTCOMES) == {"both_pass", "truth_only", "realistic_only", "both_fail", "void", "unpaired"}


def test_flags():
	f = flags(row("realistic-fixed-B-v7bot", n_recoveries=2, n_llm_images=1, physical={"source": "verdict"}))
	assert f == {
		"recovered": True, "review_used": False, "llm_used": True, "oracle": False, "outcome_source": "verdict",
	}
```

- [ ] **Step 2：运行，确认失败**

- [ ] **Step 3：实现**

```python
# src/digital_twin_armigo/governance/pairs.py
"""Pairing of truth and realistic trajectories of the same case, and the pair outcome matrix."""

from __future__ import annotations

__all__ = ["OUTCOMES", "flags", "pair_key", "pair_outcome", "sensing_of"]

OUTCOMES = ("both_pass", "truth_only", "realistic_only", "both_fail", "void", "unpaired")


def sensing_of(group: str) -> str:
	head = group.split("-", 1)[0]
	if head in ("truth", "realistic"):
		return head
	raise ValueError(f"cannot tell the sensing of group {group!r}")


def pair_key(row: dict) -> tuple:
	return (
		row["engine"],
		row["case_id"],
		row["detector"],
		row.get("strategy") or "A",
		row.get("motion") or "open",
		row.get("evidence") or "hold",
	)


def _void(r: dict) -> bool:
	return bool(r.get("infra_failed") or r.get("quarantined"))


def pair_outcome(truth: dict | None, real: dict | None) -> str:
	if truth is None or real is None:
		return "unpaired"
	if _void(truth) or _void(real):
		return "void"
	t, r = bool(truth.get("physical_success")), bool(real.get("physical_success"))
	return {(True, True): "both_pass", (True, False): "truth_only", (False, True): "realistic_only"}.get(
		(t, r), "both_fail"
	)


def flags(row: dict) -> dict:
	return {
		"recovered": bool(row.get("n_recoveries")),
		"review_used": bool(row.get("n_review_views")),
		"llm_used": bool(row.get("n_llm_images")),
		"oracle": bool(row.get("oracle")),
		"outcome_source": (row.get("physical") or {}).get("source") or "verdict",
	}
```

- [ ] **Step 4：运行测试和 ruff，确认通过**
- [ ] **Step 5：提交** `gov P1: pair key, pair outcome matrix and flags`

---

### Task P2：bronze 登记（只追加）

**Files:**
- Create: `src/digital_twin_armigo/governance/lake.py`, `src/digital_twin_armigo/governance/bronze.py`
- Test: `tests/test_gov_bronze.py`

**Interfaces:**
- Produces:
  - `lake.connect(path) -> sqlite3.Connection`：建表，表包括 `bronze`、`integrity_event`、`silver_episode`、`silver_pair`、`gold_build`；
  - `bronze.register(con, root: Path, *, source: str) -> dict`：返回计数 `{"new": n, "unchanged": n, "changed": n}`。
- 一个"回合目录"的判定：目录里有 `run_status.json` 或 `episode.json`。参与哈希的文件是 `episode.json`、`events.jsonl`、`run_status.json`、`task.json`、`replay/motion.npz`，缺失的文件记为 null。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_gov_bronze.py
import json

from digital_twin_armigo.governance import bronze, lake


def make_ep(root, name, events="{}\n"):
	d = root / "runs" / "b1" / "mujoco" / "truth-v7bot" / name
	d.mkdir(parents=True)
	(d / "run_status.json").write_text(json.dumps({"state": "done"}))
	(d / "events.jsonl").write_text(events)
	return d


def test_register_is_append_only_and_detects_changes(tmp_path):
	con = lake.connect(tmp_path / "lake.sqlite")
	d = make_ep(tmp_path, "L1-N")
	assert bronze.register(con, tmp_path, source="t") == {"new": 1, "unchanged": 0, "changed": 0}
	assert bronze.register(con, tmp_path, source="t") == {"new": 0, "unchanged": 1, "changed": 0}
	(d / "events.jsonl").write_text('{"kind": "edited"}\n')
	assert bronze.register(con, tmp_path, source="t") == {"new": 0, "unchanged": 0, "changed": 1}
	rows = con.execute("select count(*) from bronze").fetchone()[0]
	events = con.execute("select path, field from integrity_event").fetchall()
	assert rows == 1  # the original registration is kept, not overwritten
	assert events == [(str(d), "events.jsonl")]


def test_missing_files_are_null(tmp_path):
	con = lake.connect(tmp_path / "lake.sqlite")
	make_ep(tmp_path, "L2-N")
	bronze.register(con, tmp_path, source="t")
	sha = con.execute("select sha_motion from bronze").fetchone()[0]
	assert sha is None
```

- [ ] **Step 2：运行，确认失败**

- [ ] **Step 3：实现**

```python
# src/digital_twin_armigo/governance/lake.py
"""The governance store: bronze registrations, integrity events, silver tables and gold builds (one SQLite file)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
create table if not exists bronze (
	path text primary key, source text not null, registered_at text not null, git_commit text,
	sha_episode text, sha_events text, sha_status text, sha_task text, sha_motion text
);
create table if not exists integrity_event (
	id integer primary key, path text not null, field text not null, old_sha text, new_sha text, seen_at text not null
);
create table if not exists silver_episode (
	build_id text not null, path text not null, sensing text, pair_key text, split text, quarantined integer,
	quarantine_reasons text, row_json text not null, primary key (build_id, path)
);
create table if not exists silver_pair (
	build_id text not null, pair_key text not null, camera_condition text not null, truth_path text,
	realistic_path text, outcome text not null, primary key (build_id, pair_key, camera_condition)
);
create table if not exists gold_build (
	product text not null, version text not null, silver_build text not null, built_at text not null,
	git_commit text, predicate_sha text not null, n_items integer not null, primary key (product, version)
);
"""


def connect(path: Path) -> sqlite3.Connection:
	path.parent.mkdir(parents=True, exist_ok=True)
	con = sqlite3.connect(path)
	con.executescript(SCHEMA)
	return con
```

```python
# src/digital_twin_armigo/governance/bronze.py
"""Bronze: register raw episode directories by content hash. Append-only; a changed hash is an integrity event."""

from __future__ import annotations

import datetime as dt
import hashlib
import subprocess
from pathlib import Path

FILES = {
	"sha_episode": "episode.json",
	"sha_events": "events.jsonl",
	"sha_status": "run_status.json",
	"sha_task": "task.json",
	"sha_motion": "replay/motion.npz",
}


def _sha(p: Path) -> str | None:
	if not p.is_file():
		return None
	h = hashlib.sha256()
	with p.open("rb") as f:
		for chunk in iter(lambda: f.read(1 << 20), b""):
			h.update(chunk)
	return h.hexdigest()


def _commit() -> str | None:
	try:
		return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
	except (OSError, subprocess.CalledProcessError):
		return None


def episode_dirs(root: Path):
	for marker in ("run_status.json", "episode.json"):
		for p in sorted(root.rglob(marker)):
			yield p.parent


def register(con, root: Path, *, source: str) -> dict:
	now = dt.datetime.now(dt.UTC).isoformat()
	commit = _commit()
	counts = {"new": 0, "unchanged": 0, "changed": 0}
	cols = list(FILES)
	for d in dict.fromkeys(episode_dirs(root)):
		shas = {c: _sha(d / f) for c, f in FILES.items()}
		old = con.execute(f"select {', '.join(cols)} from bronze where path = ?", (str(d),)).fetchone()
		if old is None:
			con.execute(
				f"insert into bronze (path, source, registered_at, git_commit, {', '.join(cols)}) values (?, ?, ?, ?, {', '.join('?' * len(cols))})",
				(str(d), source, now, commit, *shas.values()),
			)
			counts["new"] += 1
			continue
		diffs = [(c, o, shas[c]) for c, o in zip(cols, old, strict=True) if o != shas[c]]
		for c, o, n in diffs:
			con.execute(
				"insert into integrity_event (path, field, old_sha, new_sha, seen_at) values (?, ?, ?, ?, ?)",
				(str(d), FILES[c].split("/")[-1], o, n, now),
			)
		counts["changed" if diffs else "unchanged"] += 1
	con.commit()
	return counts
```

- [ ] **Step 4：运行测试和 ruff，确认通过**
- [ ] **Step 5：提交** `gov P2: append-only bronze registry with integrity events`

---

### Task P3：silver 逐回合校验与隔离

**Files:**
- Create: `src/digital_twin_armigo/governance/segments.py`, `src/digital_twin_armigo/governance/silver.py`
- Test: `tests/test_gov_silver.py`

**Interfaces:**
- Consumes:
  - P1 的 `sensing_of`、`pair_key`、`flags`；
  - `experiments.scoring.score_episode(path)`；
  - H2 的 `mechanism` 和 `upstream`（已在评分行里）；
  - H5 的 `tool_call` 事件；
  - H6 的 `oracle_applied` 事件；
  - 审计库 `inventory.audit` 的最新裁定。
- Produces:
  - `CHECKS: dict[str, Callable[[dict, Path, dict], tuple[bool, str | None]]]`：参数依次是评分行、回合目录、上下文（含 freeze、bronze 行）；
  - `QUARANTINE_CHECKS = ("complete", "bronze_intact", "sensor_guard", "no_truth_fallback")`；
  - `build_silver(con, *, freeze: dict | None) -> str`：返回 `build_id`；
  - `segments.split_of(seed: int) -> str`：取值 `train`、`val`、`ood-*`、`test`、`loop-NN`、`legacy`、`unknown`。

各项检查：

| 名称 | 通过条件 | 不通过时 |
|---|---|---|
| `complete` | `run_state == "done"` 且 `has_episode` | 隔离 |
| `bronze_intact` | 该路径没有 `integrity_event` | 隔离 |
| `sensor_guard` | realistic 回合的每个 `tool_call.returned` 都通过 `step_mcp.assert_sensor_only`；没有 `tool_call` 事件的旧回合视为通过，记 `sensor_guard_basis="no_tool_calls"` | 隔离 |
| `no_truth_fallback` | realistic 回合不含 `oracle_applied` 事件；或者含有，但该回合标记为 oracle（H6） | 隔离 |
| `teacher_frozen` | 有 freeze 时，`episode.json` 的 provenance 提交号等于 freeze 中的提交号 | 只记标记，训练谓词会读 |
| `latest_attempt` | 同一 `case_id` 和组下，没有更新的尝试 | 标 `superseded` |
| `split_known` | `split_of(seed) != "unknown"` | 只记标记，训练谓词要求 `train` |

- [ ] **Step 1: 写失败测试**

```python
# tests/test_gov_silver.py
import json

from digital_twin_armigo.governance import bronze, lake, silver
from digital_twin_armigo.governance.segments import split_of


def test_split_segments():
	assert split_of(3_000_101) == "train"
	assert split_of(3_100_005) == "val"
	assert split_of(3_900_001) == "test"
	assert split_of(3_600_001) == "loop-01"
	assert split_of(700_001) == "legacy"


def _episode(root, group, name, *, events, state="done"):
	d = root / "runs" / "b1" / "mujoco" / group / name
	d.mkdir(parents=True)
	(d / "run_status.json").write_text(json.dumps({"state": state}))
	(d / "task.json").write_text(json.dumps({"case": [3000101, False, 0]}))
	(d / "episode.json").write_text(json.dumps({
		"result": {"label": "normal", "verdict": "SUCCESS"},
		"provenance": {"experiment": {"batch": "b1", "group": group, "engine": "mujoco", "detector": "v7bot", "strategy": "B", "case_id": name}},
	}))
	(d / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
	return d


def test_leaky_tool_return_is_quarantined(tmp_path):
	leak = {"kind": "tool_call", "tool": "observe", "returned": {"true_brick_pos": [0, 0, 0]}, "ok": True}
	d = _episode(tmp_path, "realistic-fixed-B-v7bot", "L3000101-N", events=[leak])
	con = lake.connect(tmp_path / "lake.sqlite")
	bronze.register(con, tmp_path, source="t")
	build = silver.build_silver(con, freeze=None)
	q, reasons = con.execute(
		"select quarantined, quarantine_reasons from silver_episode where build_id = ? and path = ?", (build, str(d))
	).fetchone()
	assert q == 1 and "sensor_guard" in json.loads(reasons)


def test_clean_episode_passes_and_changed_bronze_is_quarantined(tmp_path):
	d = _episode(tmp_path, "truth-B-v7bot", "L3000101-N", events=[])
	con = lake.connect(tmp_path / "lake.sqlite")
	bronze.register(con, tmp_path, source="t")
	b1 = silver.build_silver(con, freeze=None)
	assert con.execute("select quarantined from silver_episode where build_id = ?", (b1,)).fetchone()[0] == 0
	(d / "events.jsonl").write_text('{"kind": "x"}\n')
	bronze.register(con, tmp_path, source="t")
	b2 = silver.build_silver(con, freeze=None)
	assert b2 != b1
	assert con.execute("select quarantined from silver_episode where build_id = ?", (b2,)).fetchone()[0] == 1
```

`split_of` 的种子段：

- 以数据集计划和观测计划第 3.2 节为准：train 3000001–、val 3100001–、ood 3200001–、test 3900001–，闭环 `3_600_001 + 1000 × (NN − 1)`；
- 段长读 `make_suites.py` 的 manifest；
- 读不到 manifest 时，按"到下一段起点为止"划分；
- 3,000,000 以下记 `legacy`。

如果 manifest 里的值与这里不一致，以 manifest 为准，并在报告里注明。

- [ ] **Step 2：运行，确认失败**

- [ ] **Step 3：实现**

`build_silver` 的流程：

1. 遍历 `bronze` 表；
2. 对每个路径调用 `score_episode`；
3. 依次跑 `CHECKS`；
4. 从 `events.jsonl` 读 `tool_call` 和 `oracle_applied`；
5. 补上 `sensing`、`pair_key`（JSON 序列化的列表）、`split`、`flags`，以及最新审计裁定（`root_cause`、`mechanism`、`audit_state`）；
6. 写入 `silver_episode`。

`build_id` 取 `sha256(按路径排序的 bronze 哈希串 + 代码提交号)` 的前 12 位，所以相同输入、相同代码会得到同一个 `build_id`。再次构建同一个 `build_id` 时，先删除旧行再写入，保证幂等。

- [ ] **Step 4：运行测试和 ruff，确认通过**
- [ ] **Step 5：提交** `gov P3: silver episode checks, seed segments and quarantine`

---

### Task P4：silver 配对表与报告

**Files:**
- Modify: `src/digital_twin_armigo/governance/silver.py`（`build_pairs(con, build_id)`、`pair_report(con, build_id) -> str`）
- Test: `tests/test_gov_silver.py`（追加）

**Interfaces:**
- `silver_pair` 一行对应 (`pair_key`, `camera_condition`)；truth 一侧不分相机。
- 只使用最新一次尝试（`superseded` 为假的行）。
- `pair_report` 输出 Markdown：
  - 按批次 × 相机条件列出 `OUTCOMES` 计数；
  - `realistic_only` 的全部清单；
  - `truth_only` 按 `root_cause × mechanism` 的交叉表；
  - 隔离原因计数。

- [ ] **Step 1：写失败测试**

在临时目录里造 1 个 truth 回合，再造 fixed 和 drift 两个 realistic 回合，其中 drift 失败。断言：

- `silver_pair` 有 2 行，结果分别是 `both_pass` 和 `truth_only`；
- 报告里出现 `| drift | 0 | 1 |`（列顺序按 `OUTCOMES`）。

- [ ] **Step 2–4：先看测试失败，再实现，再通过**
- [ ] **Step 5：提交** `gov P4: silver pair table and pair outcome report`

---

### Task P5：gold 数据产品、manifest 与守护

**Files:**
- Create: `src/digital_twin_armigo/governance/gold.py`
- Test: `tests/test_gov_gold.py`

**Interfaces:**
- Produces:
  - `PRODUCTS: dict[str, Product]`；`Product(name, sensing, predicate, trainable: bool)`；
  - `build_product(con, name, *, silver_build, out_root, freeze) -> Path`：返回版本目录；
  - `assert_product_invariants(rows, product)`：违反第 Global Constraints 节的隔离硬约束时抛 `GovernanceError`。
- manifest 每行：`path`、`episode_sha256`（取 bronze 的 `sha_episode`）、`split`、`pair_outcome`、`flags`、`root_cause`、`mechanism`。按 `path` 排序。版本号取 manifest 字节的 sha256 前 12 位。
- `card.md` 内容：产品名、版本、silver build、提交号、谓词源码哈希、条目数、按 split、相机条件和配对结果的计数、各排除原因的计数。

谓词：

```python
# src/digital_twin_armigo/governance/gold.py (excerpt)
def _base(r):
	return not r["quarantined"] and not r["flags"]["oracle"] and not r.get("superseded")


PRODUCTS = {
	"student_train_v3": Product(
		"student_train_v3",
		"realistic",
		lambda r, ctx: _base(r)
		and r["split"] == "train"
		and r["physical_success"]
		and r["teacher_frozen"]
		and (
			r["pair_outcome"] == "both_pass"
			or (r["pair_outcome"] == "unpaired" and ctx["realistic_only_rate"][r["batch"]] < 0.01)
		),
		trainable=True,
	),
	"privileged_demos_v3": Product(
		"privileged_demos_v3", "truth",
		lambda r, ctx: _base(r) and r["split"] == "train" and r["pair_outcome"] == "truth_only",
		trainable=True,
	),
	"failure_bank": Product(
		"failure_bank", "realistic",
		lambda r, ctx: _base(r) and r["split"] != "test" and r["pair_outcome"] in ("truth_only", "both_fail"),
		trainable=False,
	),
	"harness_review": Product(
		"harness_review", None,
		lambda r, ctx: r["quarantined"] or r["pair_outcome"] == "realistic_only" or ctx["void_streak"].get(r["pair_key"], 0) >= 2,
		trainable=False,
	),
}
# eval_<suite> products are added per suite from the segment table: split == <suite>, and for "test" ctx["freeze"] is set.
```

`ctx["realistic_only_rate"][batch]` 的分母是该批次里有配对、且结果不是 `void` 的数量。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_gov_gold.py
import pytest

from digital_twin_armigo.governance.gold import PRODUCTS, GovernanceError, assert_product_invariants


def r(**kw):
	base = {
		"path": "p", "batch": "b", "split": "train", "physical_success": True, "teacher_frozen": True,
		"quarantined": False, "superseded": False, "pair_outcome": "both_pass", "pair_key": "k",
		"flags": {"oracle": False, "recovered": False},
	}
	return base | kw


CTX = {"realistic_only_rate": {"b": 0.0}, "void_streak": {}, "freeze": None}


def test_student_train_predicate():
	p = PRODUCTS["student_train_v3"].predicate
	assert p(r(), CTX)
	assert not p(r(pair_outcome="realistic_only"), CTX)
	assert not p(r(split="val"), CTX)
	assert not p(r(flags={"oracle": True}), CTX)
	assert p(r(pair_outcome="unpaired"), CTX)
	assert not p(r(pair_outcome="unpaired"), CTX | {"realistic_only_rate": {"b": 0.02}})
	assert p(r(flags={"oracle": False, "recovered": True}), CTX)  # recovery is a column, not a tier


def test_invariants_reject_leaks_into_trainable_products():
	prod = PRODUCTS["student_train_v3"]
	for bad in (r(split="test"), r(quarantined=True), r(flags={"oracle": True}), r(pair_outcome="realistic_only")):
		with pytest.raises(GovernanceError):
			assert_product_invariants([bad], prod)
	assert_product_invariants([r()], prod)


def test_failure_bank_never_holds_sealed_cases():
	assert not PRODUCTS["failure_bank"].predicate(r(split="test", pair_outcome="truth_only"), CTX)
```

再加一个端到端测试：用 P4 测试里造的临时湖跑 `build_product`，断言同样的输入构建两次，得到同一个版本目录和逐字节相同的 manifest。

- [ ] **Step 2：运行，确认失败**

- [ ] **Step 3：实现。** `assert_product_invariants` 对可训练的产品检查：`split == "train"`、未隔离、非 oracle、配对结果不是 `realistic_only`。写 manifest 之前调用它，然后对所有条目的 seed 调 `dagger.assert_train_layouts_only`。

- [ ] **Step 4：运行测试和 ruff，确认通过**
- [ ] **Step 5：提交** `gov P5: gold data products, versioned manifests, cards and isolation guards`

---

### Task P6：配对运行入口、导出接 manifest、容量测试与文档

**Files:**
- Modify: `src/digital_twin_armigo/experiments/runner.py`（子命令 `run-pair`）
- Modify: `scripts/vla/collect_paired_carry.py`（`--manifest`）
- Create: `scripts/governance/lake.py`, `scripts/governance/capacity_bench.sh`, `docs/GOVERNANCE.md`
- Test: `tests/test_run_pair.py`

**Interfaces:**
- `runner run-pair` 接受 `run` 的全部参数（`--sensing` 除外），外加 `--real-cameras fixed,recal,drift`。
  - 它先规划 truth 组（fixed 相机），再规划每个 realistic 组；
  - 所有任务放进同一个 `ProcessPoolExecutor`，按"同一案例的 truth 和 realistic 相邻"排序，这样中途停下时，已完成的大多是成对的；
  - Isaac 仍强制单进程。
- `collect_paired_carry.py --manifest <gold 版本目录>/manifest.jsonl`：只导出 manifest 里的回合，导出前逐条核对 `episode_sha256`，不一致就停止。

- [ ] **Step 1：写失败测试**

```python
# tests/test_run_pair.py
from digital_twin_armigo.experiments.runner import pair_jobs_order


def test_pair_jobs_are_interleaved_by_case():
	jobs = [
		{"group": "truth-B-v7bot", "case": ["L1"]},
		{"group": "truth-B-v7bot", "case": ["L2"]},
		{"group": "realistic-drift-B-v7bot", "case": ["L1"]},
		{"group": "realistic-drift-B-v7bot", "case": ["L2"]},
	]
	order = [(j["case"][0], j["group"].split("-")[0]) for j in pair_jobs_order(jobs)]
	assert order == [("L1", "truth"), ("L1", "realistic"), ("L2", "truth"), ("L2", "realistic")]
```

再测 `--manifest`：manifest 里有一条哈希不匹配时，导出抛错，并且不写任何输出文件。

- [ ] **Step 2–4：先看测试失败，再实现，再通过**

- [ ] **Step 5：`scripts/governance/lake.py` CLI**，四个子命令：

```bash
PYTHONPATH=src E:/digital_twin_armigo/.venv/Scripts/python scripts/governance/lake.py register --root E:/Armigo_data_v3/loop/cv-motion --source cv-motion
PYTHONPATH=src E:/digital_twin_armigo/.venv/Scripts/python scripts/governance/lake.py silver --freeze E:/Armigo_data_v3/freeze.json
PYTHONPATH=src E:/digital_twin_armigo/.venv/Scripts/python scripts/governance/lake.py gold student_train_v3
PYTHONPATH=src E:/digital_twin_armigo/.venv/Scripts/python scripts/governance/lake.py report
```

用 `D:/Armigo_data_v2` 做一次只读演练：登记，构建 silver 和 `failure_bank`，再出报告。湖文件写到 `C:/scratch/gov-dry/lake.sqlite`。把配对结果计数贴进提交信息正文。

注意 p2-dev-r2 里的 truth 组是否和 realistic 组在同一批次。如果不在，`pair_key` 照样能连上，因为它不含批次；但报告要按 realistic 一侧的批次分组。

- [ ] **Step 6：`capacity_bench.sh`【GPU，由用户或 Claude 排期】**
  - 设置：`run-pair`、套件取 p2_dev12 的前 8 个案例、相机 fixed、策略 B；workers 依次取 2、4、6、8，每档用独立的批次名。
  - 结果写进 `E:/Armigo_data_v3/governance/capacity.md`：回合/小时、检测器 `elapsed_ms` 的中位数和 p90、broker 排队时间（从 `detector_started` 到 `view` 事件的墙钟差）、峰值显存（用 `nvidia-smi --query-gpu=memory.used --format=csv -l 5` 采样）。
  - 取拐点作为 `run-pair` 的默认 `--workers`。
  - 另跑一档"策略 C，加 Qwen"，workers 取 4，记录 vLLM 单请求延迟的中位数和 p90。

- [ ] **Step 7：`docs/GOVERNANCE.md`**，写清楚：
  - 三层各是什么；
  - 配对结果矩阵；
  - 每个产品的谓词，用中文复述，并给出代码位置；
  - 隔离硬约束；
  - 操作顺序：采集 → register → silver → 各产品 → 导出 → 训练；
  - 2.4 节的两项待定。

- [ ] **Step 8：提交** `gov P6: run-pair, manifest-gated export, lake CLI, capacity bench and governance doc`

## 验收清单

- [ ] P1：配对结果矩阵的 4 格加 `void`、`unpaired` 全部有测试。
- [ ] P2：哈希变化不覆盖原登记，会产生 `integrity_event`。
- [ ] P3：泄漏的工具返回值和改动过的 bronze 都被隔离；`build_id` 可复现。
- [ ] P4：报告里列出了全部 `realistic_only`。
- [ ] P5：可训练产品的守护能拒绝 test 段、隔离、oracle、`realistic_only` 四类回合；同样的输入构建两次，manifest 逐字节相同。
- [ ] P6：导出只认 manifest，哈希不符就停；用 v2 数据做的只读演练计数已写进提交信息；容量测试等排期。
- [ ] 全量测试、`ruff check`、`ruff format --check` 通过；一个任务一个提交；没有推送。
