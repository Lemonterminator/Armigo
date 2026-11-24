# SmolVLA 碗跟随探针：判据（2026-11-24）

写于 SmolVLA 训练之前。问题定位见 [BOWL_SWAP_PROBE_2026-11-23.md](BOWL_SWAP_PROBE_2026-11-23.md)：冻结的 ACT（fit30k）在 val2–val4 上跟随 0/237。

## 1. 问题

ACT 不读文字指令，碗只以状态里的 one-hot 输入，被忽略。SmolVLA 读文字指令（`carry_to_bowl`：“Carry the held brick over the {bowl} bowl…”）。问题是：**在现有数据上换成 SmolVLA，模型会不会跟着指令选碗？**这决定下一步是先做大规模多色碗静态数据集，还是先解决数据里的捷径（目标碗由缺陷标签决定）。

## 2. 方法

- **训练**：`scripts/vla/train.py --family smolvla`，从 hackjunction 项目的 `sort-depth-v2-10k` 热启动（见第 6 节），horizon 50，`--drop-unsupervised`，takeover-weight 5。数据为 fit30k 的训练集 `E:/scratch/s11v2-01/iter-01/d_union`（310 回合）加已采的配对搬运 39 对（78 回合，只含受监督的搬运和释放帧），合并为 `C:/scratch/smolvla-probe/d_union_paired39`。步数、批大小和学习率在冒烟测完速度后定，写进第 6 节，训练开始后不改。
- **探针**：`bowl_swap_probe.py --prefix teacher`（本轮新增）。到放碗之前全由 realistic 教师执行（含 `return_to_lift`），学生只跑三支 `carry_to_bowl`：`req`、`req2`、`swap`，定义与指标同 BOWL_SWAP_PROBE 第 2–3 节。teacher 前缀让每个布局都能到达搬运段，与学生抓取好坏无关，只测指令跟随。
- **布局**：`config/vla/val2.json` 全部 100 个（不碰 val5，它留给 T1 验收）。
- **对照**：ACT fit30k 用同一 teacher 前缀在 val2 上跑一遍。

## 3. 判据（运行前写定）

先看噪声：`req` 与 `req2` 去向一致率低于 90% 时，下面的分类打折扣并写明。

- **S1 语言起作用**：SmolVLA 跟随 ≥ 85%。下一步直接准备大规模静态数据集（多色碗、目标与标签解耦、指令改写）。
- **S2 部分起作用**：跟随 30%–85%。说明语言通道有用，但现有数据不足以压过捷径；数据集必须带配对（同一画面、不同目标）样本。
- **S3 不起作用**：跟随 < 30%。换架构本身不解决；先做目标与标签解耦的配对数据，再训。
- 另外报告：SmolVLA 搬运的到达率（段末距目标碗中心 ≤ `BOWL_R`）和掉砖数，以及与 ACT 对照的差。

## 4. 已知局限

- 训练步数受时间限制，SmolVLA 可能欠拟合；S3 时要先看搬运段本身的开环误差，区分“没学会搬运”和“学会搬运但不听指令”。
- 训练集中目标碗仍主要由标签决定，配对样本只占搬运样本的一小部分。

## 5. 冒烟（不看结果）

ACT fit30k、teacher 前缀、val2 前 4 个布局：4/4 到达探针，只用于检查流程。

## 6. 训练记录

- 第一次启动（`lerobot/smolvla_base` 初始化，批 16，CPU 增强）在 400 步时停掉，没有看任何结果。原因：每步 0.38 s，GPU 利用率约 45%，瓶颈是 DataLoader 里的逐样本增强（每样本约 75 ms，主要是 CPU 上生成高斯噪声和 HSV 转换）。
- 改动：`train.py --gpu-augment`，worker 只解码，增强在 GPU 上按批做（`StudentImageAugmentor.augment_batch`，变换、参数分布和顺序与逐样本路径相同；`tests/test_vla_augment_batch.py` 验证逐张一致，CPU 路径逐位不变）。
- 热启动：`C:/Users/Jiang/.gemini/antigravity/scratch/hackjunction/artifacts/training/sort-depth-v2-10k/checkpoints/best.ckpt`（global step 4120，35 回合，指令 “Pick up the brick and place it in the green/pink bowl.”，同样从 `smolvla_base` 训、VLM 冻结）。转换为 `C:/scratch/smolvla-probe/init-hackjunction-sortdepth-best/model.safetensors`：`model._model.*` → `model.*`，丢弃 4 个 snapflow 层和归一化缓冲；VLM 与 base 逐位相同，只有动作专家和投影层不同；状态投影 32 → 38 列，新列随机初始化。它的关节单位是 −100…100 归一化，我们是弧度，均按训练集统计量重新归一化。
- 正式：`--steps 20000 --batch-size 32 --num-workers 16 --gpu-augment`，lr 1e-4，warmup 500，`--drop-unsupervised --takeover-weight 5`，输出 `C:/scratch/smolvla-probe/train-20k-warm`。冒烟 0.40 s/步（批 32），约 2.2 小时。探针用 `last` 权重。
- 数据：`d_union_paired39` = 386 回合、422,302 帧（d_union 310 + 配对 38 对）。
- ACT 对照（teacher 前缀，val2）：到达探针 99/100，跟随 0，固定 94，反向 2，其他 3；`req`/`req2` 一致 95/96。

## 7. 结果

（运行后填写）
