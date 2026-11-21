# Rerun 回放与状态机实时查看（`armigo-view`）

不保存视频，现场看轨迹：HTML 页面显示 LLM agent 状态机的**当前状态**和每个阶段的**耗时统计**，一个按钮在页内打开 [Rerun](https://rerun.io) 做 3D／时间线回放。全部只读、仅监听 `127.0.0.1`。

```powershell
uv sync --extra viewer                                   # rerun-sdk + FastAPI + uvicorn
armigo-view serve --root data/catalog --open             # http://127.0.0.1:8765/
```

## 页面

- **状态机图**：当前节点高亮并闪动，走过的边变绿，每个节点标注访问次数和累计墙钟时间；运行中约每 0.7 s 刷新，结束后降到 4 s。
- **耗时表**：按状态节点（墙钟、sim）和专家阶段（rest/approach/grasp/…）给出次数、合计、均值、p50、p95、最大、占比；还有每次访问的明细，以及决策统计（来源、回退率、LLM 延迟均值/p95/最大、尝试次数）。
- **专家阶段条**：整段轨迹按阶段着色，悬停看每段时长。
- **事件流**与实时画面（`live.jpg`，运行中）。
- **打开 Rerun 回放**：后台启动一个 Rerun 进程，页内嵌入 Rerun Web Viewer；同时只保留一个会话。启动／关闭需要进程内令牌（和资料库 Web 一致），只接受本机请求。

## 数据来源与时间口径

| 来源 | 提供 |
| --- | --- |
| `events.jsonl`（agent 实时写入） | 节点进入、决策（来源／延迟／回退）、视图、检查点 → 状态机当前状态、每次访问的 sim 与墙钟时长 |
| `branch-N/actions.npz`（分支结束后） | 逐帧 `phase`、关节 `state`/`action` → 阶段耗时、关节曲线 |
| `keyframes/*.png`、`live.jpg` | Rerun 里的相机画面 |
| `motion.npz` + `scene.json` + `telemetry.npz`（资料库录制） | Rerun 里的 3D 场景回放与遥测（接触力、速度等） |

- **访问时长** = 该节点的 `node` 事件到同一分支下一个 `node` 事件之间；仍在运行的最后一次访问按“现在”累计，事件文件超过 5 分钟没更新就视为已停止，不再增长。
- **阶段墙钟时间是插值**：事件只在节点边界带墙钟时间，阶段按帧号线性插值。同一帧上有多个事件时（仿真暂停等 LLM）取最后一个，所以 LLM 思考时间算在节点上，不算进阶段。阶段的精确量是 sim 时间。
- 3D 回放要有资料库录制；agent 的 `bottom*/case-*` 目录本身没有位姿，页面里用关键帧图像，需要 3D 时用 `--recording` 或 `--catalog` 指向对应录制。

## 命令行

```powershell
armigo-view stats data/catalog/agent/bottom2x4/case-0000          # 终端里的耗时报告（--json 给机器读）
armigo-view rrd data/catalog/agent/bottom2x4/case-0000 --out ep.rrd   # 自包含 .rrd，rerun ep.rrd 打开
armigo-view rrd data/catalog/recordings/isaac-verification-1          # 录制的 3D 回放
armigo-view rrd <agent 目录> --recording <录制目录>                    # 状态机 + 3D 同一个文件
armigo-view rrd --catalog <资料库条目 id> <任意目录>                    # 用资料库的 motion/scene/telemetry
armigo-view rerun <目录> --follow                                    # 现场 Rerun Web Viewer，跟随正在运行的 episode
```

Rerun 里的实体：`state/active/<节点>`（0/1 阶梯，类似甘特图）、`state/node_index`、`state/log`（事件文本）、`state/visit_wall_s/<节点>`、`phase/index`、`robot/state|action/<关节>`、`stats/summary`（耗时表 Markdown，运行中持续刷新）、`camera/*`、`world/<geom>`、`telemetry/*`。时间线有 `sim_time` 和 `wall_time` 两条。

## 实时跟随

只要 agent 跑在同一台机器、episode 目录可读，不需要改 agent 代码：查看器读取它已经写的 `events.jsonl`、`live.jpg`，分支结束后读 `actions.npz`。这一档是状态机、耗时、事件流和相机缩略图，没有 3D 位姿。

## 实时 3D 位姿（像 Blender 一样自由视角）

给 agent 进程设环境变量 `ARMIGO_RERUN=1`，MuJoCo 回合就会把**完整 3D 场景**实时推给 Rerun：

```powershell
$env:ARMIGO_RERUN = "1"
# 然后照常跑 agent；episode 目录里出现 rerun.json，HTML 页面的按钮变成“打开实时 3D（Rerun）”
```

- 场景几何（所有 geom，含机器人、桌面、物体）只记录一次；每个控制帧只记录**动过的 geom** 的位姿（`world/<geom>`），不影响仿真（只读，任何失败只打一条警告并关闭推流）。
- 3D 视图是标准 Rerun 视图：左键轨道旋转、右键平移、滚轮缩放、`W/A/S/D` 飞行；时间轴可以在运行中回拨。状态机曲线、事件表、耗时表在同一个窗口。
- 仿真进程内起一个 Rerun gRPC + Web Viewer（每进程一次，端口自动选；`ARMIGO_RERUN_GRPC_PORT`、`ARMIGO_RERUN_WEB_PORT` 固定端口，`ARMIGO_RERUN_MEMORY` 限内存，默认 4GiB）。地址写在 `<episode>/rerun.json`。
- 回合结束后位姿落盘：`<episode>/live.rrd`（流的完整副本）和 `<episode>/replay/{scene.json,motion.npz}`（资料库的 motion 契约）。之后 `armigo-view rrd <episode>` / `rerun <episode>` 会自动带上 3D 回放，不需要进程还活着。
- 限制：实时查看器随仿真进程存在，进程退出后页面里的实时入口自动消失，改用上面的回放。当前只支持 MuJoCo；Isaac 回合走资料库录制。

## 验证

```powershell
uv run pytest tests/test_viewer_stats.py tests/test_viewer_app.py -q
```

覆盖：访问时长／阶段分段／分位数、半行事件容错、API、令牌与本机限制、路径不能越出根目录、跟随一个逐步增长的 episode、导出的 `.rrd` 里有预期实体。
