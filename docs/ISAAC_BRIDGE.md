# Isaac 桥(双引擎)

同一套六面 agent 决策代码,可以驱动 MuJoCo 或 Isaac Lab(PhysX + RTX)。Isaac 在独立进程(`E:\il\v\Scripts\python.exe`,headless)里运行,主环境通过 localhost 连接。

## 运行

```python
from digital_twin_armigo.isaac_bridge import IsaacBridge, scene_payload

with IsaacBridge() as bridge:                      # 启动服务端,约 6-30 s
    bridge.reset(scene_payload(layout, cell))      # 新场景 6-10 s;相同场景原地重置
    st = bridge.step(ctrl, branch=0, frame=0)      # 一个 30 Hz 控制帧
    img = bridge.render("wrist_cam", depth=True)
    tok = bridge.checkpoint(); bridge.restore(tok)
    bridge.save_replay("out_dir")                  # out_dir/replay/{scene.json,motion.npz}
```

探针(7 项检查,另含腕部位姿对比和 checkpoint):

```
.venv/Scripts/python scripts/isaac/bridge_probe.py --out results/isaac-bridge/probe-600008
```

单元测试(不需要 Isaac):`pytest tests/test_isaac_bridge_codec.py`。

## 协议

- `multiprocessing.connection`,127.0.0.1,随机端口和 16 字节 authkey,请求/应答。
- 消息是 pickle(协议 4)的纯数据(dict/list/str/int/float/bytes);numpy 数组编码为
  `{"__nd__": "<f4", "shape": [...], "data": bytes}`,绝不 pickle numpy。接收端用拒绝一切类的 Unpickler。
- 命令:`hello reset step render state checkpoint restore save_replay stats shutdown`。
- 出错时服务端返回 `{"ok": False, "error", "trace"}`,客户端抛 `BridgeError`,连接保持可用。

## 约定

- 关节顺序 `JOINTS`:shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper;单位弧度。
  夹爪是位置目标(rad,张开约 0.85,夹持 0.06),不是长度。
- 物理 240 Hz,每个 30 Hz 控制帧 8 个子步;reset 后预热 12 步。
- 四元数 xyzw(Isaac Lab 3.0)。state 键:`q qd effort brick_pos brick_quat_xyzw brick_linvel brick_angvel jaw_force[2] jaw_normal[2] unintended_force frame sim_time`。
- 相机沿用 MuJoCo/OpenGL 约定(x 右,y 上,z 向后,沿 -z 看)。`cam_pose` 是 4x4 world-from-camera,等于 MuJoCo `cam_xmat` 的列。
  内参 `fx fy cx cy width height`,像素为正方形。深度是 `distance_to_image_plane`(沿光轴,米),非有限值置 0。
- 相机只在 `render` 时渲染,不推进物理时间。步进后第一次 `render` 先做 `flush` 次(默认 3,`--flush-updates`)只渲染的
  app 更新,避免 RTX 返回上一帧位姿。`render(..., flush=n)` 可显式指定。
- 缺陷:仅支持 `cross_var`,以品牌块坐标系里的盒子(USD Cube 子节点)呈现。
- `checkpoint` 仅在夹爪张开(q 和指令都 >= 0.5 rad)、夹爪力 <= 0.02 N、砖块 z <= 0.03 且近乎静止时允许,否则报错。
- `save_replay`:`scene.json`(catalog.store 格式)+ `motion.npz`(poses, timestamps, node_ids, camera_poses, branch, frame, sim_time),
  每个控制帧一行,相机有 top/front/wrist_cam。

## 探针结果(case 600008,[False,5] 与 [True,1])

| 检查 | 结果 |
|---|---|
| 1 渲染不推进物理 | q、砖块、sim_time、step 计数完全不变 |
| 2 砖角投影 | top 精度 1.0 / 召回 0.92 / 中心误差 0.7 px;front 精度 0.999 / 召回 0.85-0.88 / 中心误差 0.1-0.6 px |
| 3 顶视深度(桌面点) | 期望 0.56198 m,渲染 0.56193 m,误差 -0.05 mm |
| 4 腕部无滞后 | 大幅位姿跳变后:flush=3 与 flush=12 平均绝对差 0.36-0.47(灰度 0-255);flush=0 差 1.0-2.7,说明确有滞后,需要 flush;A/B 视图差 30;腕部位姿与 MuJoCo 差 4e-6 m |
| 5 重放确定性(310 帧,带接触,最大夹爪力 57 N) | 不是位级一致:关节最大差 7.4e-3 rad,砖块位置 0.42 mm,角度 5.8e-3 rad |
| 6 显存/延迟 | 峰值 14.0 GB(含 vLLM,基线 8.2 GB,增量约 5.8 GB);step 中位 61 ms、p95 90 ms;步进后第一次 render 约 59 ms,同状态再渲染其他相机约 15 ms |
| 7 save_replay | 310 帧,motion.npz 约 1.84 kB/帧(估计 2500 帧约 4.4 MiB);scene.json 23 MB(内嵌网格,与帧数无关) |

## 已知限制

- GPU PhysX 不保证位级确定性;三次重放不一致,且两个 case 的差值完全相同、首个差异帧为 0,
  更像第一次重放与之后重放的初始状态差异(未单独查明),不是随机噪声。
- step 约 61 ms/帧(PhysX 约 50 ms/8 子步),比 MuJoCo 慢很多。
- 只支持 `cross_var` 缺陷;标记、干扰物(markers/distractors)和杂物(clutter)未复现,`scene_payload` 会抛 NotImplementedError 或放入 `ignored`。
- 新场景构建 6-10 s(热启动),服务端冷启动约 30 s。
- 桌面为纯色;灯光沿用 stage2 强度,画面偏亮(白桌面、砖色偏淡)。
- 显存约 6-9 GB,需与 vLLM 同卡时留意余量。
- 只能用 `E:\il\v\Scripts\python.exe` 运行服务端。
