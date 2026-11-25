# CV 闭环抓取与密集取证预注册实验计划（CV Motion Matrix）

写于任何 Stage 1–4 正式实验运行之前（Task C7 Step 1 / obs §6）。

## 1. 变体定义

### 抓取闭环变体（`motion`）

| 名称 | 抓取检查点（相对抓取点高度，m） | 模式 |
|---|---|---|
| `open` | 无（开环，当前教师默认） | 开环 |
| `w1s` | `hover` | 同步（停稳 `0.15 s` + 重规划） |
| `w2s` | `hover`, `0.035` | 同步 |
| `w3s` | `hover`, `0.045`, `0.030` | 同步 |
| `w2a` | `hover`, `0.035` | 异步（不停，延迟 `0.10 s` 生效） |
| `w3a` | `hover`, `0.045`, `0.030` | 异步 |
| `w4a` | `hover`, `0.050`, `0.040`, `0.030` | 异步 |

### 取证与信念相机变体（`evidence`）

| 名称 | 说明 |
|---|---|
| `hold` | 现有停稳取证（默认，逐位兼容旧实现） |
| `img` | `belief_cameras="estimated"` + 真实图像色块质心居中裁剪 |
| `dense-10` | `img` + 腕扫路径沿关节弧长每 `10°` 在途取证（`0.08 m/s` 速度门限，检测延迟 `0.15 s`） |
| `dense-20` | `img` + 腕扫路径沿关节弧长每 `20°` 在途取证 |

## 2. 实验阶段、批次名与种子段

所有运行写入 `E:/Armigo_data_v3/loop/cv-motion/`。

| 阶段 | 批次名 | 套件 / 种子段 | 条件与矩阵 |
|---|---|---|---|
| Stage 1：标定容差曲线 | `cvm-s1-tol` | `p2_dev12` | `{open, w2s}` × `calib_offset_mm` ∈ `{0, 2, 5, 10, 20}` × `camera=fixed`, `evidence=hold`, `strategy=B`, `detector=v7bot` |
| Stage 2：抓取变体初筛 | `cvm-s2` | `p2_dev12` | 7 种 `motion` × `{fixed, recal, drift}` × `evidence=hold`, `strategy=B`, `detector=v7bot` |
| Stage 3：取证变体 | `cvm-s3` | `p2_dev12` | Stage 2 前 2 名 × `{hold, img, dense-10, dense-20}` × `{fixed, recal, drift}`, `strategy=B`, `detector=v7bot` |
| Stage 4：留出确认 | `cvm-s4-confirm` | 已登记套件 `loop_01_96`（闭环未用新种子段 `loop-01` 的 `3_600_001`–`3_600_048`，`96` 案例；与 v3 `3000001/3100001/3500001/3900001` 段不重叠） | Stage 3 最终胜者 vs `open`（`hold`） × `{fixed, recal, drift}`, `strategy=B`, `detector=v7bot` |

注：离线 H2 机制归因基准参照 `D:/Armigo_data_v2` 套件；阶段 3b（若接入 U4 分割网络）将在 `docs/experiments/brickseg_prereg.md` 中单独登记。

2026-11-25 修订（任何 cv-motion 运行之前）：Stage 2 增加 recal，使第 3 节硬约束 1 在初筛时可判；Stage 4 套件登记为 `loop_01_96`。

## 3. 选优与验收判据（逐字锁定）

- **主指标**：`drift` 下的物理成功率（`scoring` 的 `outcome == "success"`）。
- **硬约束**：
  1. `fixed` 和 `recal` 下，成功的回合数均不比 `open` 少超过 `1` 个；
  2. 标签准确率不低于 `open`；
  3. 不新增 `infra_failed`。
- **次级排序（决胜规则）**：
  - 在满足上述约束、且 `drift` 成功回合数与最好者相差不超过 `1` 个回合的变体里，取机器人仿真时间（`sim_s` 中位数）最短者；
  - 若仍打平，取抓取检查点数（K）更少者。
- **确认阶段判据**：
  - 胜者对 `open` 在 `drift` 条件下的精确 McNemar 单侧检验 `p < 0.05`，且 `fixed` / `recal` 两种条件均未出现显著退化，方算成立；
  - 若不成立，保留 `open` 为教师默认并在结果报告中写明。
- **墙钟耗时说明**：检测器推理导致的墙钟时间只作报告，不参与变体选择，因为仿真中检测器耗时不计入机器人物理执行时间（`sim_s`）。

## 4. 已知仿真与实机差异风险（仅记录，不作事后修改）

1. 在途拍照（`dense-*`）在真实相机上存在运动模糊，而 MuJoCo 无运动模糊，因此 `dense-*` 在仿真中的收益相对真机可能偏乐观。
2. 腕相机在极近检查点（如 `0.030 m`）可能遇到积木裁边，C1 通过 `CLIPPED` 拒绝此类视图。
3. 真机端腕相机安装外参偏差需独立做一次手眼/外参标定。
