# Armigo 本地实验资料库

资料库提供本地页面、SQLite 索引和 `armigo-data` 命令。原文件保留在原位，浏览页面、导入、筛选和评估不会自动导出。只有明确选择实验并提交导出请求才产生文件包或动画。

## 启动与日常操作

在项目目录运行：

```powershell
uv sync --extra catalog --extra mujoco
.\.venv\Scripts\armigo-data.exe serve --port 8765
```

打开 <http://127.0.0.1:8765>。页面可以搜索和分页、筛选引擎／原判定／质量、查看视频与遥测、勾选实验、调整夹持力阈值、检查动画记录、冻结布局划分、导出及取消／重试任务。只监听本机。关闭服务后，已排队的任务留在账本中；重启时，运行中但未发布的任务标为中断，可明确重试。不会自动重新运行失败的实验。

命令均返回 JSON，供 agent 调用。没有隐含的“全部导出”。

```powershell
armigo-data import results/physics-expert-v2
armigo-data import results/failure-export
armigo-data import results/stage2
armigo-data import results/verification-physics-v2/stage2_d415_v2
armigo-data list --verdict FAILURE --limit 50
armigo-data show <实验ID>
armigo-data verify --ids <实验ID>
armigo-data export --ids <实验ID> --format archive
armigo-data export --ids <实验ID> --format existing-video
armigo-data rerun --ids <MuJoCo父实验ID>
armigo-data validate --ids <完整运动实验ID>
armigo-data export --ids <完整运动实验ID> --format isaac-rtx --profile quality
armigo-data export --ids <观测实验ID> --format lerobot --allow-incomplete
armigo-data split --ids <实验ID1> <实验ID2> --name dataset-v1 --seed 0
armigo-data backup --out data/catalog/backups/manual-v1
```

`--enqueue` 只排队，由页面服务的单个工作线程执行。默认命令尝试直接执行；同时已有任务占用工作锁时，应使用 `--enqueue` 并通过 `armigo-data job <任务ID>` 查询。`--request-id` 可用于幂等请求，同一编号不会重新导出；改变选择必须用新编号。失败或取消的任务不会提供下载链接。

`--allow-incomplete` 是明确的研究例外，只允许导出有质量缺失项或非成功的选定观测，不放行损坏文件、缺失相机观测或不兼容的数据结构。不会启动训练或上传数据。

## 资料和谱系

默认位置为 `data/catalog/`，不提交到 Git：

- `catalog.sqlite`：来源记录、文件索引、评估历史、派生关系和冻结划分。
- `jobs.sqlite`：独立的追加式操作账本，复用现有租约与任务状态实现。
- `exports/<任务ID>/`：独立导出目录、哈希清单、过程日志及 `bundle.zip`。

数据库保存元数据；图像、视频、遥测、网格留在文件系统。位置以项目相对路径保存，读取时按项目根目录解析。修改或移动索引文件会使验证和导出失败，不会使用错误文件继续执行。SQLite 备份通过在线 backup API 完成；**元数据备份不复制索引的大文件**，完整资料请显式导出。

来源记录 ID 来自来源文件身份和 JSON 条目位置。相同显示名称不代表同一次运行。当前八份验收报告有 1,050 条来源记录，其中包括重复报告；失败来源记录数量也不等于独立失败案例数。13 组历史动画对应三个失败和十个标记案例。

原始身份、导入身份和重跑身份分开保存。历史文件没有记录的代码哈希、种子或参数保持未知；新录制保存代码／配置／资产哈希、Git 提交与工作区状态、有效物理参数及采集信息。生产者清单是声明的运行身份；快照导入额外核验内容及发布账本，不把一个文件自称的身份当成独立认证。

每个导出包含 `manifest.json` 和说明，记录选择者、冻结成员、原始判定、质量政策、来源缺失项、文件哈希和划分。完整运动的归档还提供相对路径的 `replay.json`。源码、日志和观测的原始文件不被改写。原始 agent 日志（包括 md 和 jsonl）默认不导入、不打包。

## 质量、标签与划分

原判定保持 `SUCCESS / FAILURE / UNKNOWN / INFEASIBLE`。质量检查独立，使用 `PASS / FAIL / UNKNOWN`。默认保留 3 N 非夹爪碰撞、5 N 夹爪对环境接触、1.2 m/s 松爪速度和 890 帧标记；它们的采样对象和意义分别记录。夹持峰值阈值默认未配置；没有逐子步峰值证据时，即使设置阈值也保持未知。力矩检查只把求解器证据用于测量结论，估算力矩保持未知。

阈值可通过页面或命令调整，保留评估版本和历史：

```powershell
armigo-data evaluate --ids <实验ID> --rules '{"pinch_force_N":20}'
```

阈值是仿真资料筛选条件，不是硬件安全标定。原实验评估器未更改。

LeRobot 特征继续使用原观测和动作。特权标签单独写到 `meta/privileged.parquet`：MuJoCo 的 `defect_visible_pixels` 与 Isaac 的 `defect_visibility_proxy_score` 分列，后者保留原有分数量纲；`brick_visible_pixels` 与 `brick_color_mask_pixels` 同样分列。`meta/source.json` 保存逐实验来源映射，缺少该引擎标签的列为 null，不以零代替。合并的观测必须具有相同的帧率、状态和相机结构。

布局分组使用完整初始几何描述，忽略颜色、缺陷标签、相机和物理参数；缺少杂物／布局信息时不进行正式划分。默认固定种子 0，按布局组确定性分配 80/10/10，明确命名冻结后不因后来导入而改变。历史留出和独立验收角色会传递给重跑，禁止被自动重新分配为训练数据；历史“留出”名称不证明它尚未被查看或用于调参。

## 动画记录与 RTX

MuJoCo 重跑通过显式逐帧／逐子步回调记录，保存所有可见几何、动态杂物、相机、完整 qpos、接触和力矩。RTX 直接读取记录的世界位姿，关闭物理推进，不重新运行策略。新重跑独立于历史实验；历史动画只被核对过判定、原因、帧数等摘要，资料库将其标为 `summary_only`，不声明完整轨迹 bit-exact。

Isaac Stage 2 可在独立新目录采集：

```powershell
$env:OMNI_KIT_ACCEPT_EULA = "yes"
E:\il\v\Scripts\python.exe scripts/isaac/stage2_expert.py --out-dir data/catalog/isaac-run-new --catalog-capture data/catalog/recordings/isaac-new
armigo-data import data/catalog/recordings/isaac-new/episode.json
armigo-data validate --ids <新实验ID>
```

Isaac 3.0 后端的 USD 位姿可能保持静态。采集器从真实刚体缓冲区读位姿，USD 只提供网格及相对绑定；校验会核对积木运动与独立遥测。缺少这项证据的旧 Isaac 动画记录不能直接 RTX 导出，观测文件仍可保留。静态失败试采集被保留并标为检查失败。

默认 RTX 预览档为 1280×720、32 累计采样；质量档为 1920×1080、128 累计采样。保留源帧率，输出主机位、可用的顶视／腕部小窗、PNG 序列、USD 回放场景和 H.264。30 fps 指输出视频帧率，不承诺实时渲染。

仿真运行环境隔离；可通过 `ARMIGO_ISAAC_PYTHON` 和 `ARMIGO_LEROBOT_PYTHON` 指定现有环境。GPU 导出使用跨进程工作锁，单次执行一个任务。直接调用渲染脚本属于维护入口，应避免与队列任务同时运行。

## 现有快照

```powershell
armigo-data import-snapshot --store <项目内快照库> --ledger <发布账本> --snapshot <快照ID>
```

只接收清单、blob 哈希和生产者发布事件均可验证的快照；未发布的孤立清单被拒绝。来源身份和任务证据接入导出谱系。快照不自动变成完整动画，只有实际具备完整运动数据的记录才开放对应导出能力。

## 验证

```powershell
uv run --extra catalog --extra mujoco pytest -q
uv run ruff check src/digital_twin_armigo/catalog scripts/catalog_record_mujoco.py scripts/isaac/catalog_render.py tests/test_catalog.py
```

实现中使用独立 DeepSeek 进程生成纯函数提案，Muse Spark Standard 生成页面提案并进行独立审查；主进程整合和验证，未依赖子代理自报通过。运行报告在 `data/catalog/verification/`。页面接口及脚本已测试；Windows 浏览器视觉自动化因无法确认当前地址而停止，不能声称已完成浏览器视觉验收。
