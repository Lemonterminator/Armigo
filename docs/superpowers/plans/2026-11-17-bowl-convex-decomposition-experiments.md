# MuJoCo 碗碰撞体：凸分解实验与采纳 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用一组按保真度排序的候选碗碰撞体替换「圆柱底 + 64 个倾斜盒子」，第一个通过全部鲁棒性闸门的候选立即被采纳，实验随即结束。

**Architecture:** 候选只在实验目录里通过 monkeypatch `scene._bowl` / `scene._visual_mesh_assets` / `physics_expert.bowl_floor_z` 注入，采纳之前不改 `src/`。闸门由便宜到贵依次执行，任何一道不过就淘汰该候选并转下一个。阴性对照（当前圆柱）必须被闸门判为失败，否则闸门本身无效。

**Tech Stack:** MuJoCo 4.0.1、numpy、pytest；分解工具 `coacd` + `trimesh` 只装在仓库外的独立环境（Python 3.12），运行期不依赖它，分解结果以 OBJ 形式提交进 `assets/meshes/`。

**Spec:** 本计划依据的诊断见 `results/failure-export/physics_diagnosis/REPORT.md`（第 1–3 节）。

## Global Constraints

- 只做仿真，物理参数仍是名义假设值，结论不得写成「已验证真实世界」。
- 审计评估器 `v2-audited`（`src/digital_twin_armigo/metrics.py`）不修改；`PhysicsEpisode.telemetry()` 的键不变。
- `.venv` 是 Python 3.14，分解工具不进 `pyproject.toml` / `uv.lock`；独立环境放 `C:\tmp\bowl_decomp_env`。
- 源码用 tab 缩进，行宽 120（ruff `indent-style = "tab"`）；检查范围 `src tests scripts/bench_cross_engine.py scripts/isaac/stage2_expert.py` 外的新脚本也要 `ruff format`。
- `uv` 不在 PATH：`C:\Users\Jiang\.local\bin\uv.exe`，或直接 `.venv\Scripts\python -m pytest`。MuJoCo 是可选 extra（`uv sync --extra mujoco`），否则测试被跳过。
- 不提交、不推送，除非用户明确要求。
- 采纳前 `scene.py`、`physics_expert.py`、`assets/` 不改动；实验结果写到 `results/bowl-experiments/`。
- 碗几何常量（`constants.py`）：`BOWL_R=0.10`、`BOWL_H=0.022`；碗底顶面 3.0 mm，碗沿顶 22 mm；碗摩擦 0.45/0.005/0.0005，底面 `condim=4`（有物理参数时）。

## 背景（为什么要做）

- 缺陷在 MuJoCo 4.0.1 原生 CCD：box 对 cylinder 返回 `dist=-115.31 mm`、水平法线（= R + 半宽），与厚度无关。box 对 box 走解析碰撞器，正常（`-3.35 mm`）。
- 把底换成盒子后 high_mass 0–29、low_friction 100–129 各 30/30，但那是 200×200 mm 方板，四角伸出碗外，不能直接采纳。
- 真碗网格直接做碰撞体不行：MuJoCo 把网格当实心凸包，碗心被填满，积木停在碗沿顶（z=31.5 mm，应为约 12.6 mm）。必须凸分解。
- 凸网格对积木盒子仍走通用 CCD，不能假定不触发同类缺陷，所以每个含网格的候选先过微型复现再跑回合。

## 候选方案（按保真度从高到低，依次尝试）

| id | 底 | 壁 | 需要的依赖 | 预期风险 |
|---|---|---|---|---|
| R1 `coacd_full` | coacd 凸块 | coacd 凸块 | coacd + trimesh | 薄大凸块底可能再触发 CCD 缺陷；coacd 可能把底和壁合并 |
| R2 `wedge_full` | 剖面旋转出的楔形凸网格 | 同 | trimesh（取剖面） | 底楔形仍是网格对盒子，同上 |
| R3 `strips_floor_wedge_walls` | 平行弦条盒子（解析碰撞器） | 楔形凸网格 | trimesh（取剖面） | 壁的网格对盒子路径需验证 |
| R4 `strips_floor_box_walls` | 平行弦条盒子 | 现有 64 个倾斜盒子 | 无 | 只修缺陷，壁坡度保真度不变 |
| R0 `current` | 圆柱 | 64 盒子 | 无 | **阴性对照，必须不过闸门** |
| R9 `slab_box_200` | 200×200 方板 | 64 盒子 | 无 | **阳性对照：过 G1/G3，不过 G0（伸出碗外）** |

R5（SDF 插件，非凸网格直接碰撞）不在序列内：R1–R4 全不过时停下来问用户，不自行扩展。

选择规则：R1 → R2 → R3 → R4 逐个跑，**第一个通过 G0–G5 的候选被采纳，其余不再运行**。R1 若在 Python 3.12 下装不上 coacd，记为 `blocked: tooling` 并继续，不算通过。

## 鲁棒性闸门（每道的阈值都写死在 `gates.py` 常量里）

| 闸门 | 内容 | 通过条件 |
|---|---|---|
| G0 几何 | 向下射线网格（r≤90 mm，1 mm 极坐标）测底高；沿壁剖面测坡 | 底高 3.0±0.1 mm；碗沿顶 22±1 mm；壁内表面径向偏差 ≤1.5 mm（相对 `plate_200mm.obj`，z=6/12/18 mm）；底面伸出碗外 ≤5 mm；缺失面积 ≤1% |
| G1 碰撞微型复现 | 无机器人，`mj_forward`。`pose_f552_s2.npy` + 300 个随机位姿（r≤R−20 mm，yaw 均匀，倾斜≤20°，角点穿深 0.2–4 mm）+ 100 个跨底壁交界（r∈[R−25, R] mm）；积木 2x2/2x3/2x4 | 每个积木对碗的接触 `dist ≥ −8 mm`；r<R−25 mm 的接触法线 z 分量 ≥0.5；`pose_f552_s2` 的深度在 −3.35±1 mm |
| G2 静置浸泡 | 无机器人，积木从 5 mm 落到随机位置，4 s@240 Hz，质量/摩擦取矩阵（含留出条件）的极值（`high_mass` 1.60、`low_friction` 0.65、留出条件 1.40/0.70、1.35/0.75）；每组 200 次 | 全程最大速度 ≤0.5 m/s、最大角速度 ≤10 rad/s；4 s 末速度 <0.01 m/s、偏航角速度 <0.05 rad/s；0 次违例 |
| G3 两个失败条件 | `high_mass` seed 0–29、`low_friction` seed 100–129（已知失败 29、111 在内），**再加从未跑过的 seed 200–229** | 120/120 SUCCESS；`release_peak_speed_mps` 最大 ≤0.5 m/s；0 次 bounce_out |
| G4 全量回归 | 审计矩阵 540 回合 + 留出条件 + 120 随机布局 | 无 bounce_out；矩阵 ≥538/540 且每个条件格不低于基线；布局 ≥119/120；FAST_RELEASE（>1.2 m/s）0 次；jaw-env >5 N 回合数 ≤ 基线（8）；≥890 帧回合数 ≤ 基线 |
| G5 工程指标 | 确定性与耗时 | 同一回合跑两次，telemetry 逐位相同（3 个回合）；每回合墙钟 ≤ 基线 1.5×；模型编译成功且无 MuJoCo warning |

闸门顺序即执行顺序；任一道失败立刻终止该候选，把失败证据写入 `results/bowl-experiments/<id>/gate_report.json`。

## File Structure

- Create `scripts/bowl_experiments/gates.py` — G0–G5 的实现与阈值常量。
- Create `scripts/bowl_experiments/candidates.py` — 每个候选的 MJCF 构造器与 `install(candidate)` / `uninstall()`。
- Create `scripts/bowl_experiments/decompose_plate.py` — 在独立环境运行：取剖面、coacd、写 OBJ。
- Create `scripts/bowl_experiments/run.py` — CLI；`--candidate ID` 或 `--until-pass`。
- Create `tests/test_bowl_experiments.py` — 闸门与候选构造器的单元测试（不跑整套回合）。
- Create `assets/meshes/bowl_hulls/` — 采纳时才提交的分解 OBJ。
- Create `results/bowl-experiments/` — 基线摘要、各候选 `gate_report.json`、`DECISION.md`。
- 采纳时才改：`src/digital_twin_armigo/mujoco_cell/scene.py`、`physics_expert.py`（`bowl_floor_z`）、`constants.py`、受影响的测试金标值。

---

### Task 0: 冻结基线与测量成本

**Files:**
- Create: `results/bowl-experiments/baseline/SUMMARY.json`
- Read: `results/physics-expert-v2/**/*.json`、`scripts/bench_physics_expert.py`

**Interfaces:**
- Produces: `baseline/SUMMARY.json` = `{"matrix_success": 538, "matrix_total": 540, "per_condition": {name: [ok, n]}, "layouts": [119, 120], "jaw_env_over_5N": 8, "frames_ge_890": int, "episode_wall_s": float}`；后面 G4/G5 都读它。

- [ ] **Step 1: 汇总基线**。复用 `scripts/export_failure_animations.py` 里的 `locate()` 扫描 `results/physics-expert-v2`，按条件格统计成功数、jaw-env >5 N 回合数（`JAW_ENV_OVER_5N`）、≥890 帧回合数，写入 SUMMARY.json。
- [ ] **Step 2: 测一个回合的墙钟**。

```bash
.venv/Scripts/python -c "import time,sys; sys.path[:0]=['src','scripts']; import bench_physics_expert as bp; from digital_twin_armigo.benchmarks import SENSITIVITY_MATRIX as M; from digital_twin_armigo.physics import load_physics_spec as L; t=time.time(); r=bp.run_episode(M[0], L(), bp.load_top_camera(), 1); print(r['verdict'], time.time()-t)"
```

Expected: `SUCCESS` 和秒数；把秒数写进 SUMMARY.json 的 `episode_wall_s`，并据此估算 G3（120 回合）、G4（约 660 回合）的总时长，写进 `results/bowl-experiments/PLAN_COST.md`。
- [ ] **Step 3: 确认并行度**。检查 `bench_physics_expert.py` 是否支持多进程；不支持就用 `Start-Process` 每条件一个进程（后台跑，每个写自己的 JSON）。
- [ ] **Step 4: 暂不提交**（用户未要求）。

### Task 1: 闸门库（先写闸门，再写候选）

**Files:**
- Create: `scripts/bowl_experiments/gates.py`
- Create: `tests/test_bowl_experiments.py`

**Interfaces:**
- Produces: `GateResult(name: str, passed: bool, detail: dict)`；`g1_collider_micro(model_builder, n_random=300, n_junction=100, seed=0) -> GateResult`，其中 `model_builder(extra_body_xml: str) -> mujoco.MjModel` 由候选提供，模型里已含碗和一个自由积木 `brick`。

- [ ] **Step 1: 先写失败测试（阴性对照）**。

```python
def test_g1_flags_the_current_cylinder_floor():
	from bowl_experiments import gates, candidates

	res = gates.g1_collider_micro(candidates.model_builder("R0_current"))
	assert not res.passed
	assert res.detail["worst_dist_mm"] < -50.0  # the 4.0.1 box-vs-cylinder depth bug
```

- [ ] **Step 2: 运行确认失败**：`.venv/Scripts/python -m pytest tests/test_bowl_experiments.py::test_g1_flags_the_current_cylinder_floor -q`，Expected: FAIL（`gates` 还不存在）。
- [ ] **Step 3: 实现 G1**。核心循环：

```python
def _contacts_vs_bowl(model, data, bowl_geoms):
	out = []
	for i in range(data.ncon):
		c = data.contact[i]
		if int(c.geom1) in bowl_geoms or int(c.geom2) in bowl_geoms:
			out.append((float(c.dist), np.array(c.frame[:3]), np.array(c.pos)))
	return out


def g1_collider_micro(model_builder, n_random=300, n_junction=100, seed=0) -> GateResult:
	rng = np.random.default_rng(seed)
	worst, bad_normals = 0.0, 0
	for k in range(n_random + n_junction):
		r = rng.uniform(0, BOWL_R - 0.02) if k < n_random else rng.uniform(BOWL_R - 0.025, BOWL_R)
		a = rng.uniform(0, 2 * np.pi)
		pose = random_brick_pose(rng, r, a, tilt_max_deg=20.0, pen_mm=(0.2, 4.0))
		model, data = model_builder(pose)
		mujoco.mj_forward(model, data)
		for dist, normal, pos in _contacts_vs_bowl(model, data, bowl_geom_ids(model)):
			worst = min(worst, dist)
			if np.hypot(*(pos[:2] - BOWL_XY)) < BOWL_R - 0.025 and abs(normal[2]) < 0.5:
				bad_normals += 1
	detail = {"worst_dist_mm": worst * 1e3, "bad_normals": bad_normals}
	return GateResult("G1", worst * 1e3 >= -8.0 and bad_normals == 0, detail)
```

`random_brick_pose`、`bowl_geom_ids`、`BOWL_XY` 同文件定义；`random_brick_pose` 的穿深用「最低角点相对碗底顶面 3 mm 下沉 pen_mm」计算。`pose_f552_s2.npy` 作为第 0 个固定位姿单独断言 `-3.35±1 mm`。
- [ ] **Step 4: 阳性对照测试**：`R9_slab_box_200` 通过 G1（再加断言它 G0 失败，因伸出碗外 >5 mm）。
- [ ] **Step 5: 实现 G0、G2、G3、G4、G5**。G0 用 `mj_ray` 向下射线；壁面偏差用独立环境里 trimesh 对 `plate_200mm.obj` 取的参考剖面 JSON（Task 3 产出，G0 只读该 JSON）。G2 用 `mj_step` 在 240 Hz 跑 4 s 记录速度。G3/G4 直接调用 `bench_physics_expert.run_episode` / `run_layout_episode` 并读 `SUMMARY.json`。
- [ ] **Step 6: 运行** `.venv/Scripts/python -m pytest tests/test_bowl_experiments.py -q`，Expected: PASS。
- [ ] **Step 7: 暂不提交。**

### Task 2: 候选构造器与几何单元测试

**Files:**
- Create: `scripts/bowl_experiments/candidates.py`
- Modify: `tests/test_bowl_experiments.py`

**Interfaces:**
- Consumes: `scene._bowl(name, xy, rgba, use_visual_mesh, torsional)` 的签名（`scene.py:45`）。
- Produces: `Candidate(id: str, assets_xml: str, bowl_body_xml(name, xy, rgba, torsional) -> str)`；`install(cid)` 把 `scene._bowl` 和 `scene._visual_mesh_assets` 换成候选版本，同时令 `physics_expert.bowl_floor_z = lambda *a, **k: 0.003`；`uninstall()` 还原。

- [ ] **Step 1: 弦条底（R3、R4 共用）**。

```python
def floor_strips(xy, rgba, group, condim, n=24):
	"""n parallel box strips covering the disc; analytic box-box contacts only."""
	w = 2 * BOWL_R / n
	out = []
	for j in range(n):
		x = -BOWL_R + (j + 0.5) * w
		hl = math.sqrt(max(BOWL_R**2 - x**2, 0.0))
		out.append(
			f'<geom type="box" size="{w / 2:.5f} {hl:.5f} 0.0015" '
			f'pos="{xy[0] + x:.4f} {xy[1]:.4f} 0.0015" rgba="{rgba}"{condim}{group}/>'
		)
	return "".join(out)
```

- [ ] **Step 2: 写测试**：条带无缝（相邻条带边界相接）、总面积与圆盘相差 ≤1%、最大径向伸出 ≤5 mm、所有条带顶面 z=3.0 mm。

```python
def test_floor_strips_cover_the_disc_without_gaps():
	strips = candidates.strip_params(n=24)  # [(x_center, half_len)]
	area = sum(2 * (candidates.STRIP_W / 2) * 2 * hl for _, hl in strips)
	assert abs(area - math.pi * BOWL_R**2) / (math.pi * BOWL_R**2) < 0.01
	assert max(math.hypot(x + candidates.STRIP_W / 2, hl) for x, hl in strips) - BOWL_R < 0.005
```

- [ ] **Step 3: 楔形壁（R2、R3）**：函数 `wedge_mesh_xml(profile_polys, n_wedges=48)`，对剖面里每个凸多边形 `[(r, z), ...]` 和每个角度区间 `[a_k, a_{k+1}]` 生成一个内联 `<mesh vertex="..."/>` 资产和一个碰撞 geom。剖面凸分解用 Hertel–Mehlhorn（三角化后贪心合并仍凸的相邻三角形），实现在 `decompose_plate.py`。测试：每块 `cross` 符号一致（凸）、并集面积与剖面面积相差 ≤0.1%。
- [ ] **Step 4: R1 的 OBJ 加载**：`assets_xml` 列出 `assets/meshes/bowl_hulls/r1_*.obj`（Task 3 产出）；body 里每块一个 `type="mesh"` 碰撞 geom，摩擦与 `condim` 同原碗底。
- [ ] **Step 5: R4 完全复用现有 64 盒壁**：直接调用原 `_bowl` 的壁部分（把 `bottom` 换成 `floor_strips`）。
- [ ] **Step 6: 运行** `.venv/Scripts/python -m pytest tests/test_bowl_experiments.py -q`，Expected: PASS。
- [ ] **Step 7: 暂不提交。**

### Task 3: 离线分解工具与参考剖面

**Files:**
- Create: `scripts/bowl_experiments/decompose_plate.py`
- Create: `assets/meshes/bowl_hulls/` 下的 `r1_00.obj …`（采纳 R1 时才保留）
- Create: `results/bowl-experiments/plate_profile.json`

**Interfaces:**
- Produces: `plate_profile.json` = `{"floor_top_mm": 3.0, "rim_top_mm": 22.0, "profile": [[r_m, z_m], ...], "wall_inner_r_mm": {"6": r, "12": r, "18": r}}`，供 G0 使用。

- [ ] **Step 1: 建独立环境**（需要用户先同意加依赖；不碰仓库的 `pyproject.toml`）。

```bash
C:/Users/Jiang/.local/bin/uv.exe venv --python 3.12 C:/tmp/bowl_decomp_env
C:/Users/Jiang/.local/bin/uv.exe pip install --python C:/tmp/bowl_decomp_env/Scripts/python.exe coacd trimesh numpy
```

Expected: 安装成功。若 coacd 无 Python 3.12 wheel 或安装失败，把 R1 标为 `blocked: tooling`，其余任务继续。
- [ ] **Step 2: 取剖面**：用 `trimesh.load("assets/meshes/plate_200mm.obj").section(plane_origin=[0,0,0], plane_normal=[0,1,0])`，取 x>0 半截面，输出 `(r, z)` 折线并写 `plate_profile.json`。人工核对：底顶 3.0 mm、沿顶 22 mm（与之前的发现一致）。
- [ ] **Step 3: R1 分解**：`coacd.run_coacd(mesh, threshold=0.03, max_convex_hull=32, seed=0)`；固定 seed 以保证可复现；把每块写成 OBJ。记录块数、各块包围盒，检查是否有块同时包含底和壁（若有，R1 的底仍是薄大凸块，G1 会暴露问题）。
- [ ] **Step 4: R2/R3 的剖面凸分解**：Hertel–Mehlhorn 实现，输出 `profile_polys.json`。
- [ ] **Step 5: 暂不提交。**

### Task 4: 运行器（遇到第一个通过者即停）

**Files:**
- Create: `scripts/bowl_experiments/run.py`

**Interfaces:**
- Consumes: `gates.g0 … g5`、`candidates.install/uninstall`。
- Produces: `results/bowl-experiments/<id>/gate_report.json`；`results/bowl-experiments/DECISION.md`。

- [ ] **Step 1: 实现 `run_candidate(cid) -> dict`**：`install(cid)`，按 G0→G5 顺序执行，任一失败即 `uninstall()` 并返回 `{"id", "passed": False, "failed_gate", "detail"}`；全过返回 `passed: True`。
- [ ] **Step 2: 实现 `--until-pass`**：按 `["R1", "R2", "R3", "R4"]` 顺序，第一个 `passed` 的写入 DECISION.md 后**立即退出**，不再运行后面的候选。全不过则写 DECISION.md 说明「无候选通过，列出各自失败闸门」，不改任何 `src/` 文件。
- [ ] **Step 3: 先跑对照**：`--candidate R0` 必须 FAIL 于 G1；`--candidate R9` 必须 PASS G1/G3 而 FAIL G0。对照结果不符就停下，闸门有问题，不进入候选。
- [ ] **Step 4: 后台跑 `--until-pass`**，日志写到 `results/bowl-experiments/run.log`；运行中不要轮询，等完成通知。
- [ ] **Step 5: 暂不提交。**

### Task 5: 采纳胜出候选

**Files:**
- Modify: `src/digital_twin_armigo/mujoco_cell/scene.py:45-67`（`_bowl`）及资产声明
- Modify: `src/digital_twin_armigo/mujoco_cell/physics_expert.py:302-319`（`bowl_floor_z`）
- Modify: `src/digital_twin_armigo/constants.py`（新增 `BOWL_FLOOR_TOP_M = 0.003`）
- Modify/Create: `tests/test_physics_expert.py`、`tests/test_so101_sort_sim.py`（金标值，仅限因碗几何变化而变的项）
- Create: `docs/BOWL_COLLISION.md`

**Interfaces:**
- Produces: `scene.BOWL_COLLISION_ID: str`（采纳的候选 id）；`bowl_floor_z` 不再依赖「`*_bowl` body 里的圆柱 geom」，改为返回 `BOWL_FLOOR_TOP_M`。

- [ ] **Step 1: 先写失败测试**：编译默认场景后，在碗中心向下 `mj_ray`（`bodyexclude` 取机器人与积木），命中高度须等于 `BOWL_FLOOR_TOP_M`（±0.05 mm）；并断言 `bowl_floor_z(...)` 与之相等。

```python
def test_bowl_floor_height_matches_ray_and_constant():
	cell = make_cell()  # the repo's existing test helper for a default cell
	m, d = cell.model, cell.data
	pt = np.array([*cell.layout.pink, 0.2])
	geomid = np.zeros(1, dtype=np.int32)
	dist = mujoco.mj_ray(m, d, pt, np.array([0.0, 0.0, -1.0]), None, 1, -1, geomid)
	assert abs((0.2 - dist) - BOWL_FLOOR_TOP_M) < 5e-5
	assert abs(bowl_floor_z(m, d, np.array(cell.layout.pink)) - BOWL_FLOOR_TOP_M) < 1e-9
```

- [ ] **Step 2: 运行确认失败**（常量未定义）。
- [ ] **Step 3: 改 `scene._bowl` 为胜出候选的构造；改 `bowl_floor_z`；加常量；把 `bowl_hulls` OBJ（若有）放进 `assets/meshes/`。** `cell.py:167-172` 的摩擦循环按 `geom_contype != 0` 过滤，对盒子/网格都适用，无需改；但要在测试里断言碗的所有碰撞 geom 的 `geom_friction` 都被设置。
- [ ] **Step 4: 跑整套测试**：`.venv/Scripts/python -m pytest -q`。先列出失败项，逐个判断是否由碗几何改变引起（如 `test_open_loop_path_is_bit_identical` 的金标值）；只改因此而变的金标值，并在 `docs/BOWL_COLLISION.md` 记录每个被改金标值的旧值、新值、原因。
- [ ] **Step 5: 重跑 G4 的全量回归**（用已采纳的代码而不是 monkeypatch），结果必须与实验时一致。
- [ ] **Step 6: 导出新基线**：更新 `results/bowl-experiments/baseline/SUMMARY.json` 的「采纳后」版本；重新运行 `scripts/export_failure_animations.py --list` 看失败/标记回合是否变化，把清单写进 `docs/BOWL_COLLISION.md`，作为后面夹爪状态计划的输入。
- [ ] **Step 7: `ruff check src tests` 与 `ruff format`**；暂不提交。

### Task 6: 收尾与报告

**Files:**
- Create: `results/bowl-experiments/DECISION.md`（Task 4 已创建，这里补充）

- [ ] **Step 1: 写决策记录**：每个候选的失败闸门和数值证据、胜出候选、与 Isaac 凸分解的差异（壁数、底处理）、未覆盖项（真机、3.12.0 底部自旋、SDF）。
- [ ] **Step 2: 删除实验期临时产物**（独立环境保留，`C:\tmp` 里的临时文件清理）。
- [ ] **Step 3: 向用户报告并等待是否提交。**

## Self-Review

- 目标覆盖：多个凸分解实验（R1–R4）、第一个过闸门即采纳并停止（Task 4 Step 2）、阴阳性对照（Task 4 Step 3）、独立环境不污染仓库（Task 3 Step 1）。
- 已知空白：每道闸门的耗时要等 Task 0 实测；G0 的壁面参考剖面依赖 Task 3 的 trimesh，若环境装不上则 G0 的壁偏差退化为只检查沿顶高度和底高，并在报告里写明。
- 类型一致：`GateResult`、`Candidate`、`install/uninstall`、`BOWL_FLOOR_TOP_M` 在各任务里的名字一致。
