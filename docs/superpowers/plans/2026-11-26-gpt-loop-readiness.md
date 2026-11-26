# 给 GPT：闭环第 1 轮前的收尾修复

分支：在 `claude/reverse-act-demos-dbc45c`（当前 7be6abd）上新开 `gpt/loop-readiness`，worktree 自建。完成后报告提交号，**不推送、不合并**。

## 背景（已定的决定）

- 2026-11-26 用户决定：**drift 不再作为失败来源**。仍然模拟 drift 相机条件，但每个 drift 回合默认启用告警加重新标定（7be6abd 里的 `drift_alarm`，需要 `localise="vote"`）。fixed 和 recal 不变。
- 验收结论（Claude，2026-11-26）：H1–H8 已提交；封存 B/C 已落盘。fixed、recal、truth 三种条件都接近满分，失败只出现在 drift 下。闭环要正式开始，还差下面这些项。
- 人工审计由用户在 8767 前端进行，审计库为 `D:/Armigo_data_v2/reports/drift-dev-20261126-7be6abd/audit/reviews.sqlite`（viewer 的默认位置）。**不要写这个库，也不要删它。**

## 约束

- 不推送，不合并到 main，不改 `E:/digital_twin_armigo`（main checkout）。
- 不改动 `D:/Armigo_data_v2` 下已有的 runs 和 reports，新批次只追加到新目录。
- **封存段（720001– 的 p2-sealed 和 v3-test 段 3900001–）不得重跑，也不得读来调参。**
- 可以用 Qwen vLLM 跑 vote；不训练任何模型。
- 阈值只在 dev 批次上校准。诊断基准的 heldout 一半每轮只评测一次。
- 每个任务一个提交，带测试；全量 pytest、`ruff check`、`ruff format --check` 都要通过（已知的旧失败照旧排除，并在报告里列出）。

## 任务

### T1 drift 条件默认启用告警和重新标定

- `runner` 里，`camera_condition == "drift"` 时默认 `drift_alarm=True`、`localise="vote"`；新增 `--no-drift-alarm` 用于消融。
- 每个回合的 `episode.json` 和 provenance 都写清实际生效的 `drift_alarm` 与 `localise`。`contract.py` 和 freeze 的核对项要能区分新旧默认。
- 测试：drift 条件默认开启；`--no-drift-alarm` 关闭；oracle 含 brick 或 camera 时，沿用 7be6abd 的冲突检查。

### T2 台账跳转链接

`experiments/ledger.py` 的 `VIEWER_TRAJ_URL` 写的是 `http://127.0.0.1:8767/?traj=`，但 viewer 是按 `location.hash`（`/#<episode id>`）选回合的，点链接会落在默认回合上。改成 `/#`，并给出测试；已生成的 `ledger.md` 不改，下次重新生成时修正。

### T3 报告的"症状 × 机制"表加 upstream

现在的表里 `calibration_drift` 列全是 0，因为漂移记在 `upstream` 字段，读表的人会误以为没有漂移。改成按 `mechanism / upstream` 分层计数，或者另加一张"症状 × upstream"表，二选一，在报告里说明选了哪种。

### T4 最大簇的完整反事实

对 `drift-dev-20261126-7be6abd/round-01` 的 1 号簇（`grasp / no_clamp / calibration_stale`，8 例）全部跑 H6 的反事实（`repeat`、`camera`、`brick`、`bowls`、全替换）。

- 结果写到新目录下的 `counterfactual.jsonl` 和 `counterfactual.md`。
- 汇总 `minimal` 集合的分布。
- 已有的 `L3600003-N` 三个单项都能救回来，分不出主因。如果整簇大多也是这样，要在报告里写明"单项替换不具区分度"，并提出一个更强的区分方法（例如只做部分替换、减小替换幅度）。只提出，不实现。

### T5 第 0 轮基线

- 用 H7 的 `round_metrics.py`，以 `drift-dev-20261126-7be6abd` 为第 0 轮，写出 `round_metrics.json`，并在 `rounds.md` 里追加第 0 轮一行。
- 人工裁定还没完成，所以 κ 和诊断成本两项写 `null`，并注明原因。
- `make_diagnosis_bench.py` 要能指向上面那个审计库。用户裁定完成后，一条命令就能重算出 calib 和 heldout 两半；把这条命令写进 `rounds.md`。

### T6 告警修复过三道关（新种子段）

- 漂移告警是在 3600001– 段上开发的，这一段不能再用来验收。用闭环第 2 轮的种子段 **3601001–**（按计划第 3.2 节的公式），先在 `make_suites` 的 manifest 里登记。
- 用 F4 的 `verify_fix.py` 对比两臂：drift 下 `--no-drift-alarm`，对 drift 下默认（告警开启）。样本量按 F 计划第 2 节预注册的数字，**先写判据再跑**。
- 报告三道关各自的判定、配对差及区间、n10；同时报告每例多花的帧数和墙钟时间。

### T7 余量探测（只报告，不修）

recal 加 drift 告警之后，现有条件基本没有失败，闭环判据"新种子上的成功率逐轮上升"没有上升空间。

- 在 manifest 里登记一个独立的 probe 种子段，不能与 v3 四段、封存段或闭环段（3600001–3609999）重叠。
- 用 runner 已经支持的更难条件，例如 clutter 等级（只用已经有的参数，不新增场景生成代码），在默认设置下各跑一小批：fixed 和 recal；drift 加告警。
- 报告每个条件的失败率及区间、第一处错误分布，以及台账能不能聚出簇。
- **不修任何东西。** 结果交给用户选闭环的学习条件。

### T8 台账分诊改用物理结果（用户人工审计发现，优先做）

用户在 8767 前端审计了 1 号簇（`grasp / no_clamp / calibration_stale`）的 8 例，发现：

- `realistic-drift-B-v7bot/L3600001-D0` 和 `L3600001-N`：第一次 `no_clamp` 之后恢复成功，积木进了正确的碗。`scores.jsonl` 里 `physical_success=True`，但 `episode.json` 的 `result.verdict` 是 `FAILED`。原因是 `triage.py` 读的是 `verdict_final_cycle / verdict`，而 `_rule_r3` 只要见到 drift 加任意一次 grasp 的 `no_clamp` 就判 `calibration_stale`。所以台账的"失败 16 例"里混进了物理上成功的回合。
- `_rule_r3` 是同义反复：判据只看相机条件和故障码，不看证据。

要求：

- 分诊和台账的"失败"以评分的 `physical_success`（加上标签是否正确）为准，不再看 agent 自报的 verdict。中途出过故障但恢复成功的回合，单列为 `recovered`，不进失败簇。
- R3 必须有证据才归因，例如 precursors 里 `calibration_drift` 超阈值，并且有 `brick_pose_error` 一类的下游；否则判 `unclassified`。
- 重新生成 `drift-dev-20261126-7be6abd/round-02-retriage/` 下的台账，旧的 `round-01` 不动。报告新旧簇的计数差。
- 测试：用 L3600001-D0 这类"FAILED 但物理成功"的回合做夹具。

**补充给 T4 和机制表的人工观察**（用户裁定）：

- `L3600003-N`（loc-vote）：第一次抓取，爪子戳在积木顶面靠侧边的位置，积木弹起后侧翻 90° 落在桌上，第二次抓取失败。
- `L3600003-D2`：情况类似，没有第二次抓取。
- `realistic-drift-B-v7bot/L3600002-D1` 和 `L3600002-N`：同样是爪子戳到积木，积木被弹飞，重试失败。评分记为落在桌上，与观察一致。
- 汇总 1 号簇 8 例：戳翻或弹飞 4 例（L3600002、L3600003 各两例），直接没夹住 2 例（L3600005），恢复成功被误判为失败 2 例（L3600001）。
- `L3600005-D4` 和 `L3600005-N`：直接没夹住。这两例的 `brick_xy` 误差约 31 mm，是本批最大的。

由此：

- "接触后积木翻倒，恢复流程不处理翻倒的积木"是现有 `MECHANISMS` 里没有的机制。请评估是否新增一个机制，例如 `brick_toppled`：积木的倾角或高度在接触后发生突变。阈值只在 dev 上定。
- 检查恢复流程在积木翻倒后有没有重新估计姿态。

## 报告格式

用中文简短报告，每个任务一段：

- 提交号
- 测试结果
- 关键数字
- 产物路径

T6 和 T7 给出批次报告的路径。做不到或证据不足的地方直接写"证据不足"，不要猜。
