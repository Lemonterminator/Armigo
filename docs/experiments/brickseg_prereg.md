# 积木分割网络（BrickSegUNet）预注册评测标准

写于 U2 正式训练及 U4 离线/闭环评测运行之前（Task U4 Step 1 / obs §7）。

## 1. 离线评测通过线（仿真指标，全部不含真机门槛）

离线评测脚本 `scripts/perception/eval_brick_pose.py` 在 U1 产生的 `val`（`250` 案例）与 `ood-appearance`（`200` 案例）分片上运行，对 `lab`、`unet-rect`、`unet-pnp` 三种方法配对评测，先验由真值叠加固定种子扰动（`xy ±15 mm`，`yaw ±15°`）模拟。

| 指标 | 集合 | 通过线 |
|---|---|---|
| 顶面 IoU | `sim-val` | `≥ 0.90` |
| 可见角点 PCK@2px（320×240） | `sim-val` | `≥ 0.90` |
| 腕部位姿误差中位数 | `sim-val` | `xy ≤ 1.5 mm`，`yaw ≤ 1.5°` |
| 定位成功率 | `sim-val` | `≥ 98%` |
| 定位成功率和误差 | `sim-ood-appearance` | 不差于 C1 经典 `lab` 方法（配对比较） |

注：顶相机视图及手中积木相对夹爪位姿误差仅作信息报告，不纳入上述通过线，也不接入当前 agent。

## 2. 阶段 3b（决定是否接入教师的闭环实验）

教师默认是否切换为 `segmenter="unet"` 不由上表决定，而由 C7 加跑的阶段 3b 决定：

- **批次名**：`cvm-s3b`（输出至 `E:/Armigo_data_v3/loop/cv-motion`）
- **配置**：`stage-2/3` 选出的最优抓取与取证变体（默认通过 `WINNER_MOTION` / `WINNER_EVIDENCE` 传入） × `segmenter` ∈ `{lab, unet}` × `camera` ∈ `{fixed, recal, drift}` × `suite=p2_dev12`, `strategy=B`, `detector=v7bot`
- **选优与约束**：沿用 `docs/experiments/cv_motion_prereg.md` 第 3 节全部判据（主指标 `drift` 成功率；`fixed`/`recal` 不降超过 1 例；标签准确率不降；无新增 `infra_failed`；决胜先看 `sim_s` 中位数）。

## 3. 真机外观对照范围（U5）

U5（`scripts/perception/realism_compare.py`）仅做真机 vs 仿真的并排视觉拼图对照与人工观察记录，**不计算真机数值指标、不设阈值、不以真机帧调参**。

## 4. 2026-11-25 修订（U2 正式训练、U4 正式评测与任何 `cvm-s3b` 运行之前；经用户讨论确认）

起因：WP8 冒烟评测显示 `lab` 在 val 腕相机上只成功 12.7%（主要是 `BAD_FOOTPRINT`），“不差于 `lab`”几乎没有约束力；同色杂物分层的定义过宽（val 1020/1304 帧判为同色）。第 1 节原表保留作记录，以下条款优先。

1. **新增经典候选 `silhouette`**（`src/digital_twin_armigo/agent/brick_silhouette.py`）：已知尺寸长方体的整体剪影拟合（x、y、yaw 三自由度），按区域对比打分（剪影内前景比例减去外侧 4 mm 窄带内前景比例），并按主色分开拟合。接受阈值 `min_contrast=0.8`、`min_visible=0.9`。
   - 这两个阈值是在 U1 `val`/`ood-appearance` 的腕相机直立帧上选定的，因此 `silhouette` 在这两个分片上的数字只算开发数字，不用于判定。
   - 冻结点为本修订所在提交。之后如果修改，必须在正式评测前另行登记。
2. **新的判定集：腕部检查点分层集 `wrist-checkpoint-v1`**。
   - 来源：真实闭环回合中 `_wrist_look` 实际拍到的帧，覆盖 `grasp_loop.VARIANTS` 的全部高度（hover、0.05、0.045、0.04、0.035、0.03）。
   - 套件为 v3 `val` / `ood-appearance`，相机条件 `drift`，先验取自当时的真实信念（不是人工叠加的扰动）。
   - 种子与 U1、`p2_dev12`、`loop_01_96` 均不重叠，由 `scripts/perception/make_wrist_checkpoint_set.py` 生成。
3. **通过线**（适用于 `unet-*` 和 `silhouette`，在检查点分层集上判定）：

   | 指标 | 集合 | 通过线 |
   |---|---|---|
   | 定位成功率 | `checkpoint-val` | `≥ 98%` |
   | 位姿误差中位数 | `checkpoint-val` | `xy ≤ 1.5 mm`，`yaw ≤ 1.5°` |
   | 定位成功率和误差 | `checkpoint-ood` | 不差于 `silhouette`（配对比较）；`unet-*` 才适用此条 |

   `lab` 只作参考列报告，不再作为门槛。
4. **同色杂物**：
   - 改为逐帧、逐相机标注干扰物：干扰物投影位置分为“视野外 / 视野内但在门限圈外 / 在 4 cm 门限圈内 / 与积木接触”四档。
   - 相似度用渲染像素的 Lab ΔE 衡量，与 `_blob` 的 `min_delta=35` 对照。
   - 另设压力集：在距积木 3–6 cm 处强制放置一个同色 2×2 杂物。
   - 分层结果和压力集都只作报告，不设通过线。
5. **阶段 3b**：`segmenter` ∈ `{lab, unet, silhouette}`，其余配置与第 2 节相同，判据仍沿用 `cv_motion_prereg.md` 第 3 节。离线通过线不决定是否接入教师。
6. **探索性闭环（不计入判定）**：在 CPU 上用固定评分器跑 `p2_dev12`、`realistic`、`drift`、策略 B，比较 `open`、`w2s-lab`、`w2s-silhouette` 三组。结果标为探索性，不替代阶段 3b。
