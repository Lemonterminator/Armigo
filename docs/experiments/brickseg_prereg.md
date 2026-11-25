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
