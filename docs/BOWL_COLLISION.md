# MuJoCo 碗碰撞体凸分解实验与采纳记录

> 仿真名义参数（MuJoCo 4.0.1，`config/sim/physics_params.json` `v1-nominal-assumed`），结论不外推真实硬件。

## 1. 采纳方案：`R6_box_floor_wedge_walls`

- **构造**（[`scene.py`](file:///e:/digital_twin_armigo/src/digital_twin_armigo/mujoco_cell/scene.py) `BOWL_COLLISION_ID = "R6_box_floor_wedge_walls"`）：
  - **碗底**：单块解析方盒 `140 × 140 × 2.5 mm`（`pos_z = 0.00175 m`，顶面 `BOWL_FLOOR_TOP_M = 0.003 m`，带 `condim="4"`），四角半径 `98.99 mm < BOWL_R (100 mm)`，不伸出碗外，在 `plate_200mm.obj` 真实平底区（`r <= 62 mm`）内 **0 条接缝**，完全规避 MuJoCo 4.0.1 原生 box-vs-cylinder `-115.31 mm` CCD 缺陷与切分接缝水平法线问题。
  - **碗壁**：对 `assets/meshes/plate_200mm.obj` 横截面外侧（`r >= 62 mm`）做 2D Hertel–Mehlhorn 凸分解得到 7 个凸多边形，沿圆周旋转 48 份生成 336 个内联凸楔形 `<mesh>` 碰撞体（`group="3"`）。
  - **`bowl_floor_z`**（[`physics_expert.py`](file:///e:/digital_twin_armigo/src/digital_twin_armigo/mujoco_cell/physics_expert.py)）：不再搜索 `*_bowl` 圆柱 geom，改为直接返回常量 [`BOWL_FLOOR_TOP_M`](file:///e:/digital_twin_armigo/src/digital_twin_armigo/constants.py)（`0.003 m`），并由 [`test_bowl_floor_height_matches_ray_and_constant`](file:///e:/digital_twin_armigo/tests/test_physics_expert.py) 用向下 `mj_ray` 核对。

## 2. 候选与对照实验证据

| ID | 底 | 壁 | 结局 | 关键证据 |
|---|---|---|---|---|
| `R0_current`（阴性对照） | 圆柱 `R=100` | 64 倾斜盒 | FAIL G0 & G1 | `euler="0 -0.35 a"` 在 `±π/2` 形成缺口（`rim_min=3.0 mm`, `wall_miss=3`, `protrusion=6.04 mm`）；G1 `worst_dist=-159.90 mm`、`bad_normals=6` |
| `R9_slab_box_200`（阳性对照） | `200×200` 方板 | 64 倾斜盒 | FAIL G0 / PASS G1, G3 | G0 `protrusion=41.42 mm`；G1 通过；G3 `120/120` |
| `R1_coacd_full` | CoACD 32 凸块 | CoACD 32 凸块 | FAIL G0 | CoACD 把底心与曲壁合并（`r1_05.obj` 跨 `r=0..85.85 mm`），平底抬高至 `5.19 mm`、壁偏差 `7.38 mm` |
| `R2_wedge_full` | 48 楔形旋转网格 | 48 楔形旋转网格 | FAIL G1 | G0 通过（壁偏 `0.179 mm`）；G1 `bad_normals=417`（平底切分扇片径向垂直缝导致水平法线） |
| `R3_strips_floor_wedge_walls` | 24 条带盒 | 48 楔形旋转网格 | FAIL G1 | G0 通过（壁偏 `0.179 mm`、伸出 `4.10 mm`）；G1 `pose_f552_s2` 踩中条带垂直缝返回水平法线 `(-0.993, 0, 0.117)`，`bad_normals=121` |
| `R4_strips_floor_box_walls` | 24 条带盒 | 64 倾斜盒 | FAIL G0 | 继承 `R0` 64 盒壁缺陷（`rim_min=3.0 mm`, `wall_miss=3`, `protrusion=6.04 mm`、壁面与 `plate_200mm.obj` 偏差 `>1.5 mm`） |
| `R5_sdf` | MuJoCo 内置 SDF | MuJoCo 内置 SDF | FAIL G0 & G1 & 回合 | 2.2 mm 薄壳穿透（`missing_area=2.59%`, `z=6 mm` 射线 `72/72` 全漏）；`pose_f552_s2` 返回 `-0.97 mm`；`baseline_2x2` seed 1 `FAILURE ['settle_timeout']`，耗时 `2.43×` |
| **`R6_box_floor_wedge_walls`** | `140×140×2.5` 单方盒 | 48 楔形旋转网格 | **PASS G0–G5** | 见下表（G1 平底法线半径按 `plate_200mm.obj` 真实平底 `60 mm` 校验） |

`R6` 闸门实测明细（[gate_report.json](file:///e:/digital_twin_armigo/results/bowl-experiments/R6_box_floor_wedge_walls/gate_report.json)）：
- **G0**：PASS（`missing_area=0.0`，平底 `3.000 mm`，沿顶 `21.93–21.99 mm`，内壁径向偏差 `0.179 mm`，伸出 `0.0006 mm`）
- **G1**：PASS（`pose_f552_s2` `dist=-3.348 mm`、`bad_normals=0`；400 随机/交界位姿 `worst_dist=-7.091 mm`、平底 `r<60 mm` `bad_normals=0`）
- **G2**：PASS（800 次浸泡 `violations=0`，`max_speed=0.327 m/s`，`max_angvel=9.44 rad/s`，末速度 `<=9.5e-5 m/s`）
- **G3**：PASS（`high_mass` 0–29 + 200–229、`low_friction` 100–129 + 200–229：**`120/120 SUCCESS`**，`bounce_outs=0`，`max_release_peak_speed=0.300 m/s`）
- **G4**：PASS（已采纳源码复核：**`540/540`** 矩阵+留出、`119/120` 布局、`bounce_outs=0`、`fast_releases=0`、`jaw_env > 5 N` 由 `6` 降为 `2`、`frames >= 890` 为 `1`）
- **G5**：PASS（3 回合×2 逐位相同，MuJoCo 警告 `0`，墙钟 `0.877 s` = 基线 `1.19× <= 1.5×`）
- **专项碗壁碰撞**：60 次 `r=62–85 mm` 落壁 `0` 飞出（`end_v <= 0.00043 m/s`）；48 次对准全部楔形接缝 `0.5 m/s` 撞壁 `0` 飞出（`end_v = 0.0 m/s`）。

## 3. 与 Isaac 凸分解的差异

- **Isaac Sim / PhysX**（`assets/props/green_plate.usda` / `pink_plate.usda`）直接消费 USD 上的网格凸分解；
- **MuJoCo 4.0.1** 对 `plate_200mm.obj` 做了结构分离：平底由单个解析 `box`（`condim="4"`）承载以杜绝 CCD 与接缝缺陷，曲壁用基于精确 2D 剖面的 48×7 凸楔形网格匹配视觉轮廓（偏差 `<=0.18 mm`）。

## 4. 受碗几何改变而更新的测试金标值

1. [`tests/test_so101_sort_sim.py`](file:///e:/digital_twin_armigo/tests/test_so101_sort_sim.py) `test_scene_xml_uses_plate_and_brick_visual_meshes_while_hiding_collision`：
   - `<asset>/mesh`：`{el.attrib["name"]: el.attrib["file"]}` 改为 `el.attrib.get("file", "")`，因碗壁凸楔形网格使用内联 `vertex="..."` 而无 `file` 属性。
   - 碗碰撞 geom 过滤与数量：`type != "mesh"` 且数量 `65`（1 圆柱 + 64 盒）改为 `group == "3"` 且数量 **`337`**（1 方盒 + 7×48 凸楔形网格）。
2. [`tests/test_physics_expert.py`](file:///e:/digital_twin_armigo/tests/test_physics_expert.py) `OPEN_LOOP_REFERENCE`（所有 `verdict` 和 `reason_codes` 均未改变，仅以下稳态/接触浮点因碗底由圆柱变为方盒、碗壁由 64 盒变为精确楔形而变化）：
   - `baseline_2x2` seed 0：`final_dist_to_bowl_m` `0.016107775802471632 → 0.01598654973060319`；`final_brick_speed_mps` `5.375441996750679e-16 → 2.842153472593232e-16`；`final_brick_ang_vel_rps` `1.658041317683676e-08 → 9.703879612542032e-11`
   - `latency_backlash` seed 3：`max_slip_pos_m` `0.09730674675667723 → 0.09603191915451315`；`max_slip_rot_deg` `105.63162971682857 → 104.75487736639278`；`final_dist_to_bowl_m` `0.049316789239620656 → 0.047643392800519144`；`final_brick_speed_mps` `1.4449729981012896e-15 → 8.862695693548982e-06`；`final_brick_ang_vel_rps` `3.431672297239216e-07 → 0.0002500281080478251`
   - `dynamic_clutter` seed 2：`max_slip_pos_m` `0.009138321958648932 → 0.00913831868232905`；`max_slip_rot_deg` `7.797042907766439 → 7.797035364661856`；`mean_pinch_force_N` `7.129510286954424 → 7.129510087248068`；`max_pinch_force_N` `10.984975248156365 → 10.984974865402503`；`pinch_normal_peak_N` `12.471505220409892 → 12.471504782902315`；`pinch_tangential_peak_N` `2.646900659661933 → 2.6468914350585155`；`pinch_impulse_Ns` `68.87253946510539 → 68.87253315586108`；`max_tracking_error_rad` `0.5567771724721692 → 0.5567771724882659`；`final_dist_to_bowl_m` `0.0206249768196165 → 0.020523193623034414`；`final_brick_speed_mps` `2.51280850617075e-06 → 0.0006523924874616907`；`final_brick_ang_vel_rps` `1.7764555254814785e-05 → 0.055699853506223106`
   - `high_mass` seed 4：`final_dist_to_bowl_m` `0.00766919001028477 → 0.008292026414899336`；`final_brick_speed_mps` `1.704630595185858e-16 → 1.1029972226957287e-15`；`final_brick_ang_vel_rps` `1.1446425884869085e-09 → 1.4964902094279297e-07`
3. [`tests/data/gripper_state_golden.json`](file:///e:/digital_twin_armigo/tests/data/gripper_state_golden.json)：在 `R6` 碗上重新录制 Stage 1（只观测）基线 SHA-256 哈希，供 Stage 2/3 逐位不变回归使用。

## 5. 采纳后新基线与待夹爪计划处理的清单

来源：[SUMMARY_POST_ADOPTION.json](file:///e:/digital_twin_armigo/results/bowl-experiments/baseline/SUMMARY_POST_ADOPTION.json)、[audit.json](file:///e:/digital_twin_armigo/results/gripper-state/audit.json)（660 回合已采纳代码实测）：
- **矩阵 + 留出条件**：`540/540 SUCCESS`
- **随机布局**：`119/120 SUCCESS`（唯一失败：`layouts_seed1000_idx32`，`pick_miss`）
- **`jaw_env > 5 N`（2 个）**：`layouts_seed1000_idx15`（`12.5 N`、`place` 张开时被卡）、`layouts_seed1000_idx32`（`17.0 N`、`grasp` 闭合时被卡）；另 `layouts_seed1001_idx11`（`4.2 N`、`place` 张开被卡）亦被 `GripperMonitor` 检出。
- **`frames >= 890`（1 个）**：`layouts_seed1001_idx37`。

## 6. 未覆盖项

- 真实硬件上的碗表面摩擦与积木接触刚度尚未标定；
- `MuJoCo 3.12.0` 旧版底部自旋行为与第三方 `sdf` 插件未在本轮采纳范围内；
- `C:\tmp\bowl_decomp_env`（仓库外 Python 3.12 工具环境）按计划保留。

## 7. 3D 碰撞建模经验与 MuJoCo / Isaac Sim (PhysX) 对比

### 7.1 本轮踩坑与 3D 建模经验

1. **视觉网格不等于物理碰撞网格**：非凸容器（薄壁盘/碗）在 MuJoCo 中直接用 `<geom type="mesh">` 会被隐式取单凸包变成“实心盖子”，必须在设计阶段就分离 `visual` 与 `collision`（不同 `group`）并用几何/射线闸门实测内部净空。
2. **欧拉角序列陷阱（`euler="0 -0.35 a"`）**：绕多轴组合欧拉角沿圆周摆倾斜盒壁时，默认旋转顺序会随方位角引入滚动耦合，在 `±π/2` 形成 19 mm 豁口与 6.04 mm 外凸；用带精确剖面轮廓的旋转楔形或显式四元数/`axisangle` 才没有方位盲区。
3. **解析碰撞器也有深穿透退化**：MuJoCo 4.0.1 中 `box` 对大半径浅底 `cylinder` 存在特定交界姿态（`pose_f552_s2`、`high_mass_seed29`）产生 `-115.31 mm` 水平假法线并触发弹飞；换用有限尺寸解析 `box`（`140 × 140 × 2.5 mm`）后该姿态回到正确向上法线。
4. **切分平面接缝会产生虚假水平法线**：哪怕把同一平底切分得完美贴合（`R2` 48 扇楔、`R3` 24 平行条带），相邻凸块的垂直侧壁在底层仍会暴露出水平法线（`bad_normals=417` / `121`），导致微穿透时横向踢飞物体。承重平底区应保持整块无接缝，且平底采样边界必须严格按真实 `OBJ` 平底半径（`plate_200mm.obj` 为 `r <= 62 mm`）划定。
5. **通用凸分解与内置 SDF 不适合薄壳精密容器**：`CoACD` 默认阈值会把碗心平底与上翘曲壁并入同一凸包（平底抬高到 `5.19 mm`、`R1`）；MuJoCo 内置 `type="sdf"` 对 2.2 mm 薄壳在离散体素上出现表面丢失（`missing_area=2.59%`、`z=6 mm` 射线 `72/72` 全漏、`R5` 实测 `settle_timeout` 且耗时 `2.43×`）。针对轴对称容器，直接对真实剖面轮廓做 2D 凸分解再旋转成楔形（`R6` 壁偏差 `0.179 mm`）显著优于黑盒 3D 分解。
6. **物理诊断先于控制修补**：先通过 6 级闸门（`G0` 几何、`G1` 无机器人穿透与法线、`G2` 极值静置浸泡、`G3` 失败条件、`G4` 全量回归、`G5` 确定性/耗时）把环境碰撞假物理清零后，残留失败自然收敛到少数真实运动学挤压（`layouts_seed1000_idx32` 等），此时再上夹爪 `BLOCKED` 状态机才能精准对症。

### 7.2 MuJoCo vs. Isaac Sim (PhysX) 在本项目的工程差异

| 维度 | MuJoCo 4.0.1（本项目） | Isaac Sim / PhysX（`scripts/isaac/prepare_assets.py`、`green_plate.usda`） |
|---|---|---|
| **非凸网格碰撞原生支持** | `<geom type="mesh">` 默认取**单个凸包**；内置 `sdf` 对 2.2 mm 薄壳穿透且耗时 `2.43×`，需要离线拆成解析 `box` + 336 凸楔形网格 | 原生通过 `UsdPhysics.MeshCollisionAPI` 配合 `approximation="convexDecomposition"`（VHACD/CoACD 管线）直接加载网格凸分解 |
| **接触与静置稳定性** | 软约束凸优化求解器，无切分平底 + `condim="4"` 下静置残差极小（`<= 9.5e-5 m/s`），但接缝/薄壳退化会给出极端水平法线 | PhysX TGS/PGS 刚体管线 + GPU/CPU 碰撞近似（受凸包膨胀/接触偏移影响，薄壁碗仍需验证凸分解精度与接触偏移） |
| **摩擦与材质语义** | `geom_friction`（滑/旋/滚三元组）与 `condim` 直接挂在 MJCF geom 上，求解器按内置规则组合 | 通过 `UsdPhysics.MaterialAPI` + `PhysxMaterialAPI` 显式声明（如 `prepare_assets.py` 中 `physxMaterial:frictionCombineMode="max"`） |
| **关节驱动与观测语义** | `<position>`/`<motor>` 执行器模型简单确定、CPU 步进快且同机位严格逐位可复现（适合 `660` 回合快速回归） | `UsdPhysics.DriveAPI("angular")`（stiffness/damping），更贴近复杂场景与 RTX 渲染（如 `catalog_render.py` 录制的 1280×720 RTX 演示视频） |
| **本项目的协作定位** | 高速、确定性物理验证底座、660 回合基线回归与 `GRIPPER_BLOCKED` 策略迭代 | 高保真资产管线、跨引擎一致性对照与记录状态（recorded-state）RTX 视觉回放 |
