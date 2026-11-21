# 合成轨迹与场景清单（`armigo-inventory`）

一张表回答“项目里有哪些合成轨迹、场景，每条是谁生成的、能看到什么”。只读扫描，不改、不复制被扫描的数据；每次重建整张表。

```powershell
armigo-inventory scan --root data=data/catalog --root results=results --root scratch=E:/scratch
armigo-inventory summary                      # 终端里按类别 × 引擎的计数
armigo-inventory show case-0007               # 某条轨迹的全部字段
armigo-inventory label --match "*/old-run/*" --category agent_realistic --note "见运行日志"   # 人工覆盖，再 scan
```

默认根目录是仓库的 `data/catalog` 和 `results`；`--root [标签=]路径` 可重复，用来加入仓库之外的目录（例如 `E:/scratch`）。输出在 `data/inventory/`（Git 忽略，`--out` 改位置）：

| 文件 | 内容 |
| --- | --- |
| `inventory.sqlite` | 三张表 `trajectories`、`scenes`、`datasets` 加 `meta`（扫描时间、规则版本、代码提交、各根目录计数） |
| `trajectories.csv` `scenes.csv` `datasets.csv` | 同样内容，带 BOM，Excel 直接打开 |
| `SUMMARY.md` | 按类别、运行批次、场景、数据集的汇总 |

## 在浏览器里看

清单页是 `armigo-view` 服务的一部分（只读，仅本机回环）：

```powershell
armigo-inventory scan                              # 先有 data/inventory/inventory.sqlite
armigo-view serve --root data/catalog --open       # 然后打开 http://127.0.0.1:8765/inventory
armigo-view serve --inventory E:/some/dir          # 清单不在默认的 data/inventory 时
```

- 顶部是“类别 × 引擎”矩阵（内容不重复的轨迹数，括号内为 SUCCESS 数），点格子就按它筛选。
- 轨迹 / 场景 / 数据集三个页签；芯片按类别、引擎、结果、运行批次、分类依据、根目录筛选，每个芯片上的数字是再点它会得到的条数；另有搜索框、“3D 位姿 有/无”、是否隐藏字节重复的拷贝、点表头排序、分页。
- 点一行打开详情（该行全部字段）；agent 轨迹（包括 `--root` 之外的 `results/`、`E:/scratch` 里的）详情里有链接直接跳到状态机页面回放；`--root` 之外的用 `inv:<traj_id>` 作 episode id，由服务按清单里记录的路径解析，不接受客户端给的路径。目录被删掉了就不能回放。
- API（页面用的就是这些，也可直接脚本调用）：`/api/inventory/summary`、`/api/inventory/{trajectories|scenes|datasets}`（`?category=model,model_takeover&engine=mujoco&q=…&sort=-seed&limit=100&offset=0`）、`…/facets`、`/api/inventory/trajectory?id=…`。排序和筛选的列名都来自固定白名单，值走参数绑定。
- 页面读的是上一次 `scan` 的结果；重新扫描后刷新页面即可，不需要重启服务。

## 重跑失败回合（新增记录，保留原结果）

失败的 agent 回合可以按原案例、原策略、原感知模式整条重跑。原记录一个字节都不改，重跑写成**新记录**；通过了新记录是 `SUCCESS`，没通过也留着 `replay/`（`ARMIGO_RERUN=record`，见 [VIEWER.md](VIEWER.md)），可以直接开 3D 回放，不需要存视频。

```powershell
armigo-inventory rerun --dry-run                      # 先看会重跑多少、哪些被跳过及原因
armigo-inventory rerun --limit 5                      # 先试几条
armigo-inventory rerun --run-group bottom2x4          # 只重跑某个运行批次
armigo-inventory rerun                                # 全量；中断后再跑会跳过已完成的
armigo-inventory scan                                 # 把新旧记录连起来
```

- 范围：`agent_omniscient` / `agent_realistic` 里 `FAILED` / `UNKNOWN` 的回合（`--category` 可改）。没有 `task.json`（无法重建）、用了 LLM（需要模型服务）的回合会跳过，并说明原因。模型 / 接管（`model*`）的失败不在此列：它们测的是某个检查点，换成“重跑到通过”会改变含义，需要另行设计。
- 输出：`<inventory>/reruns/<运行批次>/<原目录名>-r<次数>/`，里面有 `lineage.json`（重跑了谁、第几次、代码提交、新旧检测器哈希、新旧结果）。下次 `scan` 自动把 `<inventory>/reruns` 当作根目录 `reruns`。
- 表里新增列：`rerun_of`（指向原记录）、`attempt`、`rerun_code`、`rerun_info`、`superseded_by`（原记录上，指向最新一次重跑）。原记录的 `verdict` 不变；字节完全相同的重跑加标签 `reproduced`；重跑不会被当作 `same_as` 重复。`--redo` 再来一轮，次数加一，最新一次为准。
- 页面：“重跑”列显示 `重跑 r1` 或 `已被重跑`；“只看最新一次”隐藏已被取代的原记录，“只看重跑 / 只看原记录”下拉；顶部汇总有重跑数、通过数、被取代数。
- 末轮评估（`verdict_final_cycle`）：agent 先把砖放到桌上再扫描、或原地重试过的回合，整条回合的物理健康检查会把“放到桌上”的松爪当成入盆释放，砖明明进了盆也判 `place_miss`。这类回合现在**另外**算一次“只看最后一轮抓放”的健康检查，写进 `result.health.final_cycle` 和 `result.verdict_final_cycle`；原来的 `verdict` 不改（旧结论保持一致，扰动实验本来就用末轮）。清单里 `verdict_final_cycle`（末轮结果）、`failure_mode`（整条回合的失败原因）、`failure_mode_final_cycle` 三列可筛选，页面上两个结果并排显示。只有重跑后的回合带这个字段；旧回合的遥测没存，无法离线补算。
- 注意：重跑用**当前**检测器（`--detector`，默认 `results/anomaly/models-sup/sup.json`），早期回合的检测器文件已无法恢复，所以 `rerun_info.detector_changed` 通常是 `true`；仿真本身是确定的，同一案例同一策略多半得到同一个结果。这是“当前代码下的结果”，不是对原结果的修正。

## 来源类别（`category`）

“谁在驱动机械臂”和“它能看到什么”两个问题合成一个字段：

| 类别 | 含义 | 典型来源 |
| --- | --- | --- |
| `scripted` 纯脚本 | 固定脚本（物理专家、专家 tracer、启发式），没有决策环；用特权状态 | 资料库导入的 physics-expert、Isaac stage-2、`student_rollout` 里 `student=false` 的教师 |
| `agent_omniscient` 全知 agent | 状态机 agent（规则或 LLM）+ 仿真真值（`sensing=truth`） | `data/catalog/agent/*`、agent-bench |
| `agent_realistic` 实时感知 agent | 同一个 agent，只用真实单元能感知的东西：相机、关节、力矩（`sensing=realistic`） | VLA 教师数据（`vla-pilot-*`）、`results/demos/realistic` |
| `model` 纯模型输出 | 学习到的策略（ACT / SmolVLA）产生全部动作 | `s10-*`、`s11-*-eval`、`student/` 目录 |
| `model_takeover` 模型 + 专家接管 | 模型先跑，安全门触发后交给脚本专家（DAgger） | `s11-act-roll-03/*/raw`、`s11-probe-*` |
| `agent_unknown` | agent 回合，感知模式没记录且无法推断；**需要人工补标** | — |

同时记录 `control`（`agent_rule` / `agent_llm` / `model_policy` / `script_*`：谁做的技能级决策，LLM 是否被咨询过）和 `sensing`（`truth` / `realistic` / `privileged`）。

### 分类依据（`category_basis`）

| 值 | 含义 |
| --- | --- |
| `explicit` | 回合写入时由 `EpisodeTrace.finish` 冻结在 `provenance.actor`（今后新数据都是这种），规则以后改了也不会改写它 |
| `inferred` | 老数据：由 `provenance` 里的 `student`、`dagger_takeover`、`sensing` 和 `result.llm` 推断。没有 `sensing` 字段的回合，若 `episode.json` 早于“真实感知模式”引入时间（2026-11-18 23:24 +02:00，提交 `b218d60`）则判为 `truth`（那时只有这一种），之后的判为 `agent_unknown` |
| `override` | 人工在 `config/inventory_overrides.json` 里指定（`armigo-inventory label` 追加），带 `note`。已用于 step-MCP 的两个对照批次（目录名即 profile） |

覆盖规则是对路径或 id 的 glob，后面的规则优先；未知类别会直接报错。该文件进 Git，改标签是可审计的。

## 表的主要字段

**轨迹** `trajectories`：`traj_id`、`record_kind`（`agent_episode` / `catalog_episode`）、`run_group`（运行批次）、`category`/`control`/`sensing`/`category_basis`/`category_note`、`engine`、`scene_id`、`seed`、`defective`、`defect_face`、`verdict`、`label_correct`、`sim_frames`、`sim_s`、`wall_s`、`views`/`rewinds`/`retries`、`llm_consulted`、`camera_profile`、`detector_sha256`、`takeover_gate`/`takeover_frame`、`has_actions`/`has_events`/`has_keyframes`/`has_video`/`has_motion`、`split_hint`、`catalog_role`/`quality_status`、`written_at`、`episode_sha256`、`same_as`、`tags`。

- `has_motion`：有可回放的 3D 位姿（资料库的 `motion.npz`，或 `ARMIGO_RERUN=1` 写出的 `replay/`）。没有它的轨迹只能看关键帧，渲染出来核对要先补录。
- `same_as`：`episode.json` 与更早某条字节相同的指向它——拷贝，或同案例、同配置的确定性重跑。汇总里分别给出总数和内容不重复的数。
- `split_hint`：路径里 train / val / calib / held / test 的提示，或资料库冻结划分的 `名称:划分`。只是提示，不等于正式划分。

**场景** `scenes`：每个种子案例（`case:种子:缺陷:面`）、每个资料库布局（`layout:` 几何哈希）、渲染数据集里的每个布局（`render:数据集:序号`）一行，带 `n_trajectories` 和 `categories`（哪些类别的轨迹跑过它）。agent 案例的布局由种子经 `sample_case` 现算，没有随回合保存，所以这里只存案例三元组；要看几何用资料库的录制。

**数据集** `datasets`：渲染数据集（`layouts.jsonl` + `metadata.jsonl`）和 LeRobot 导出（`meta/info.json`），带条目数、`dataset_manifest.json` 摘要（有则填，见 [MLOPS.md](MLOPS.md)）。

## 限制

- 扫描认的是 `armigo-agent-episode/v1` 的 `episode.json`、`catalog.sqlite` 的 `episodes`、渲染数据集和 LeRobot 目录。其它零散结果（`results/stage1`、`top-camera-reach` 等探索脚本的 json）不算轨迹，没有收入。
- 资料库导入全部归为 `scripted`：目前导入的都是 physics-expert、Isaac stage-2 报告和录制。以后若导入别的来源，用覆盖规则或扩展分类。
- “全知 / 实时感知”描述的是回合级 `sensing` 模式，不是 agent-bench 里 `perception` 实验的 ideal/realistic/sensor/stress 档位（那些档位是检测器输入的扰动，仍属于 `agent_omniscient`）。
- 不检查数据对不对：一条轨迹在表里是 `SUCCESS` 不代表渲染出来没问题；回放核对用 [VIEWER.md](VIEWER.md)。

## 验证

```powershell
uv run pytest tests/test_inventory.py tests/test_inventory_view.py tests/test_inventory_rerun.py -q
```

覆盖分类规则（含旧数据推断与显式冻结）、三个适配器、场景聚合、去重、覆盖规则、CSV、CLI、扫描不改被扫描文件；重跑的计划与跳过原因、`lineage.json` 链接、最新一次筛选、`--redo` 次数。
