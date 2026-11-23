# 目标碗独立通道 + 配对示范 实现计划（2026-11-23）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 ACT 学生在搬运时按指令选碗。验收标准：新布局 val5 上，目标碗对照探针的跟随率 ≥ 85%。

**Architecture:**
- **碗通道。** 目标碗 one-hot 从 38 维状态里拿出来，用 LeRobot ACT 自带的 `observation.environment_state` 作为一个独立的编码器 token 输入。原状态 token 里对应的三位清零。
- **配对示范。** 训练布局上，专家在 `present_bottom` 之后的持砖检查点处分叉：一支按原目标放碗，另一支放进另一只碗。两支只监督搬运和释放帧。
- **训练与评估。** 新数据并入 fit30k 的训练集，配方与 fit30k 相同：从 long-03 初始化，30k 步，lr 3e-5。最后用探针和闭环配对评测验收。

**Tech Stack:** Python 3.14，MuJoCo，LeRobot 0.6（ACT，`E:/il/lerobot-cuda`），PyTorch bf16，pytest，ruff（tab 缩进）。

**Spec:** [BOWL_SWAP_PROBE_2026-11-23.md](BOWL_SWAP_PROBE_2026-11-23.md) 第 6 节（问题定位），以及 `E:/scratch/bowl-swap/REPORT.md` 第 5 节（处理方向）。

## Global Constraints

- **数据契约不变。** `contract.STATE_DIM = 38`，client/server 的请求格式不变。碗通道只在模型封装内部拆分。旧检查点必须照常加载、照常推理。
- **学生输入不得包含规划器诊断量**（`FORBIDDEN_DIAGNOSTIC_KEYS`）。不得把碗坐标或其他特权信息送进模型。
- **配对数据只用训练布局**（`config/vla/train.json`），用 `dagger.assert_train_layouts_only` 守护。不得使用 val、val2–val5、test、bench16。
- **验收集 val5 是全新布局**（种子 1700001–1700100）。判据在运行前写定并提交；运行后不改。
- **评测用默认设置**（等待夹稳 + 合拢退出 0.03）。
- **只提交在本地分支 `exp/grasp-tail-ab`，不推送。** 提交信息以 `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` 结尾。
- **代码风格。** tab 缩进；`ruff check` 与 `ruff format` 通过。用仓库的 `.venv/Scripts/python` 跑测试，训练和导出用 `E:/il/lerobot-cuda/Scripts/python.exe`。

## 文件结构

| 文件 | 改动 | 职责 |
| --- | --- | --- |
| `src/digital_twin_armigo/vla/models.py` | 改 | `ACTWithEndHead(bowl_token=...)`：拆分碗通道；检查点元数据；`load_act_weights_expanding()` 从无碗通道的检查点加载权重 |
| `scripts/vla/train.py` | 改 | `--bowl-token` 参数；`build_model` 用扩展加载 |
| `src/digital_twin_armigo/agent/twin_mujoco.py` | 改 | `TwinMujoco.target_bowl(label)` 钩子；`place()` 和 `evaluate()` 改用它 |
| `src/digital_twin_armigo/vla/paired_carry.py` | 新 | `TargetSwapTracer`，`restrict_supervision()`，`run_pair()` |
| `scripts/vla/collect_paired_carry.py` | 新 | 批量采集配对示范，导出 LeRobot 分片并合并 |
| `tests/test_vla_bowl_token.py` | 新 | 碗通道测试 |
| `tests/test_vla_paired_carry.py` | 新 | 目标翻转与监督掩码测试 |
| `config/vla/val5.json` | 新 | 验收布局 |
| `docs/vla/BOWL_CHANNEL_VALIDATION_2026-11-23.md` | 新 | 预注册的验收协议与结果 |

---

### Task 1：模型的碗通道

**Files:**
- Modify: `src/digital_twin_armigo/vla/models.py`（`ACTWithEndHead`，约 703–1002 行）
- Modify: `scripts/vla/train.py:131-170`（`build_model`），以及参数解析（约 394 行起）
- Test: `tests/test_vla_bowl_token.py`

**Interfaces:**
- Produces:
  - `ACTWithEndHead(..., bowl_token: bool = False)`；
  - 常量 `BOWL_SLICE = slice(31, 34)`；
  - `load_act_weights_expanding(model: ACTWithEndHead, src_state_dict: dict[str, Tensor]) -> list[str]`，返回新初始化的参数名；
  - `vla_config.json` 的 `model_config.bowl_token`（缺省时为 False）。

**背景：** LeRobot ACT 的编码器输入 token 依次为 `[latent, robot_state, env_state, 图像特征]`（`modeling_act.py` 341–361、462–468 行）。配置了 `env_state_feature` 后，它会多一个 `encoder_env_state_input_proj = Linear(env_dim, dim_model)`，`encoder_1d_feature_pos_embed` 也从 2 行变成 3 行。所以从 long-03 初始化时不能 `strict=True` 加载：投影层需要新建，位置嵌入需要扩一行。

- [ ] **Step 1：写失败的测试**

```python
# tests/test_vla_bowl_token.py
import numpy as np
import pytest
import torch

from digital_twin_armigo.vla.contract import STATE_DIM, STUDENT_CAMERAS, encode_state
from digital_twin_armigo.vla.models import BOWL_SLICE, ACTWithEndHead, load_act_weights_expanding

SHAPES = {cam: (3, 32, 32) for cam in STUDENT_CAMERAS}


def _model(bowl_token):
	torch.manual_seed(0)
	return ACTWithEndHead(horizon=8, tiny_for_test=True, camera_shapes=SHAPES, bowl_token=bowl_token)


def _batch(bowl):
	q = np.zeros(6)
	s = encode_state(q, q, q, 0.5, "carry_to_bowl", bowl=bowl)
	b = {cam: torch.rand(1, *SHAPES[cam]) for cam in STUDENT_CAMERAS}
	b["observation.state"] = torch.from_numpy(s)[None]
	return b


def test_bowl_token_splits_the_state():
	m = _model(True)
	act = m._prepare_act_batch(_batch("pink"), include_action=False)
	assert act["observation.environment_state"].tolist() == [[0.0, 0.0, 1.0]]
	assert torch.all(act["observation.state"][:, BOWL_SLICE] == 0)


def test_without_bowl_token_the_batch_is_unchanged():
	m = _model(False)
	act = m._prepare_act_batch(_batch("pink"), include_action=False)
	assert "observation.environment_state" not in act
	assert act["observation.state"].shape == (1, STATE_DIM)


def test_bowl_token_changes_the_prediction():
	m = _model(True)
	torch.nn.init.normal_(m.policy.model.encoder_env_state_input_proj.weight, std=1.0)
	a_green, _ = m.predict_chunk(_batch("green"))
	a_pink, _ = m.predict_chunk(_batch("pink"))
	assert not torch.allclose(a_green, a_pink)


def test_checkpoint_roundtrip_keeps_bowl_token(tmp_path):
	m = _model(True)
	m.save_checkpoint(tmp_path)
	back = ACTWithEndHead.load_checkpoint(tmp_path)
	assert back.bowl_token is True
	old = _model(False)
	old.save_checkpoint(tmp_path / "old")
	assert ACTWithEndHead.load_checkpoint(tmp_path / "old").bowl_token is False


def test_expanding_load_from_a_model_without_the_token():
	old, new = _model(False), _model(True)
	fresh = load_act_weights_expanding(new, old.state_dict())
	assert "policy.model.encoder_env_state_input_proj.weight" in fresh
	pe_old = old.state_dict()["policy.model.encoder_1d_feature_pos_embed.weight"]
	pe_new = new.state_dict()["policy.model.encoder_1d_feature_pos_embed.weight"]
	assert pe_new.shape[0] == pe_old.shape[0] + 1
	assert torch.equal(pe_new[: pe_old.shape[0]], pe_old)
	for k, v in old.state_dict().items():
		if k != "policy.model.encoder_1d_feature_pos_embed.weight":
			assert torch.equal(new.state_dict()[k], v), k
	with pytest.raises(ValueError):
		load_act_weights_expanding(_model(False), new.state_dict())
```

- [ ] **Step 2：运行，确认失败**

运行：`.venv/Scripts/python -m pytest tests/test_vla_bowl_token.py -q`
预期：FAIL，报 `ImportError: cannot import name 'BOWL_SLICE'`。

- [ ] **Step 3：实现**

在 `models.py` 中：

```python
from lerobot.utils.constants import OBS_ENV_STATE  # with the existing ACTION / OBS_STATE / OBS_IMAGES imports

# Target-bowl one-hot inside the 38-dim student state (contract.encode_state [31:34]). With ``bowl_token`` the
# ACT wrapper moves it to its own encoder token (LeRobot's environment-state input) and zeroes it in the state token.
BOWL_SLICE = slice(31, 34)
POS_EMBED_KEY = "policy.model.encoder_1d_feature_pos_embed.weight"
```

`ACTWithEndHead.__init__` 增加关键字参数 `bowl_token: bool = False`，并赋值 `self.bowl_token = bool(bowl_token)`。在构建 `input_features` 之后加入：

```python
		if self.bowl_token:
			input_features[OBS_ENV_STATE] = PolicyFeature(
				type=FeatureType.ENV, shape=(BOWL_SLICE.stop - BOWL_SLICE.start,)
			)
```

`_prepare_act_batch` 中，把 `OBS_STATE: self._normalize_state(state)` 改成：

```python
		state_norm = self._normalize_state(state)
		act_batch: dict[str, Any] = {OBS_STATE: state_norm, OBS_IMAGES: imgs}
		if self.bowl_token:
			act_batch[OBS_ENV_STATE] = state[:, BOWL_SLICE]  # raw one-hot: ("", green, pink)
			state_norm = state_norm.clone()
			state_norm[:, BOWL_SLICE] = 0.0
			act_batch[OBS_STATE] = state_norm
```

其余改动：

- `_pool_encoder_features` 不用改：token 1 仍是 robot_state。
- `_checkpoint_meta` 增加 `"bowl_token": self.bowl_token`。
- `load_checkpoint` 构造时传入 `bowl_token=bool(mc.get("bowl_token", False))`。

新增模块级函数：

```python
def load_act_weights_expanding(model: ACTWithEndHead, src_state_dict: Mapping[str, Tensor]) -> list[str]:
	"""Load an ACT checkpoint without the bowl token into ``model`` (which has it): every shared tensor is copied,
	the 1-d positional embedding keeps its old rows and gets one new row (left at its fresh init), and the new
	env-state projection stays freshly initialised. Returns the names of freshly initialised tensors."""
	if not model.bowl_token:
		raise ValueError("load_act_weights_expanding needs a model with bowl_token=True")
	if any("encoder_env_state_input_proj" in k for k in src_state_dict):
		raise ValueError("source already has a bowl token; load it with load_state_dict(strict=True)")
	own = model.state_dict()
	fresh = sorted(set(own) - set(src_state_dict))
	merged = dict(own)
	for k, v in src_state_dict.items():
		if k not in own:
			raise ValueError(f"unexpected tensor {k} in source checkpoint")
		if k == POS_EMBED_KEY:
			if own[k].shape[0] != v.shape[0] + 1 or own[k].shape[1:] != v.shape[1:]:
				raise ValueError(f"{k}: cannot expand {tuple(v.shape)} to {tuple(own[k].shape)}")
			merged[k] = torch.cat([v, own[k][v.shape[0] :]], dim=0)
			fresh.append(f"{k}[{v.shape[0]}]")
		elif own[k].shape != v.shape:
			raise ValueError(f"{k}: shape {tuple(v.shape)} != {tuple(own[k].shape)}")
		else:
			merged[k] = v
	model.load_state_dict(merged, strict=True)
	return fresh
```

`load_act_weights_expanding(_model(False), new.state_dict())` 必须报 ValueError（测试最后一段），由第一个 `if` 保证。

在 `train.py` 中：

- 增加 `p.add_argument("--bowl-token", action="store_true", help="ACT: target-bowl one-hot as its own encoder token (docs/vla/BOWL_CHANNEL_PAIRED_PLAN_2026-11-23.md)")`；
- 一路透传到 `build_model(..., bowl_token=args.bowl_token)` 和 `ACTWithEndHead(..., bowl_token=bowl_token)`；
- 把 init-from 分支（166–168 行）改成：

```python
		if init_from not in (None, "none", "scratch"):
			loaded = ACTWithEndHead.load_checkpoint(init_from, device="cpu")
			if bowl_token and not loaded.bowl_token:
				fresh = load_act_weights_expanding(model, loaded.state_dict())
				print(f"[init] expanded {init_from} with a bowl token; fresh tensors: {fresh}", flush=True)
			else:
				model.load_state_dict(loaded.state_dict(), strict=True)
```

- 如果 `train_summary.json` 记录了模型配置，就把 `bowl_token` 也写进去。

- [ ] **Step 4：运行，确认通过，并跑回归**

```
.venv/Scripts/python -m pytest tests/test_vla_bowl_token.py tests/test_vla_s9_losses.py tests/test_vla_s10_client.py -q
```

预期：全部 PASS。如果 `.venv` 缺 lerobot，改用 `E:/il/lerobot-cuda/Scripts/python.exe -m pytest ...`，与 `test_vla_s9_losses.py` 的现有用法一致。

- [ ] **Step 5：用真实检查点冒烟**

```
E:/il/lerobot-cuda/Scripts/python.exe scripts/vla/train.py --family act --dataset E:/scratch/s11v2-01/iter-01/d_union --out E:/scratch/bowl-token-smoke --init-from E:/scratch/s11-long-03/best --bowl-token --smoke-steps 20 --eval-every 1000000
```

预期：

- 日志打印 `[init] expanded ... fresh tensors: ['policy.model.encoder_env_state_input_proj.bias', 'policy.model.encoder_env_state_input_proj.weight', '...pos_embed.weight[2]']`；
- 生成 `last/vla_config.json`，其中 `"bowl_token": true`；
- 用 `scripts/vla/bowl_swap_probe.py --ckpt E:/scratch/bowl-token-smoke/last --max-episodes 2` 能跑通，说明 server 能加载新检查点。

- [ ] **Step 6：提交**

```
git add src/digital_twin_armigo/vla/models.py scripts/vla/train.py tests/test_vla_bowl_token.py
git commit -m "ACT bowl token: target-bowl one-hot as its own encoder token (--bowl-token)"
```

---

### Task 2：目标碗钩子与目标翻转的专家

**Files:**
- Modify: `src/digital_twin_armigo/agent/twin_mujoco.py`（`place`，1691–1756 行；`evaluate`，1758 行起）
- Create: `src/digital_twin_armigo/vla/paired_carry.py`
- Test: `tests/test_vla_paired_carry.py`

**Interfaces:**
- Produces:
  - `TwinMujoco.target_bowl(self, label: str) -> str`；
  - `TargetSwapTracer(HeldCheckpointExpertTracer)`，带属性 `target_swap: bool`（构造参数，默认 True）；
  - 翻转后的回合，`result["target_swap"] = True`。

**背景：** `place()` 有两处、`evaluate()` 有两处，都写死了 `"pink" if defective else "green"`。翻转支故意把砖放进“另一只碗”。为了让它在自己的目标下按成功计（导出资格检查 `student_frames` 要求 `verdict == SUCCESS`），这四处要改用同一个钩子，翻转只在子类里发生。结果里加 `target_swap: True`，避免被误当成一次正确分拣。

- [ ] **Step 1：写失败的测试**

```python
# tests/test_vla_paired_carry.py
from digital_twin_armigo.agent.twin_mujoco import TwinMujoco
from digital_twin_armigo.vla.paired_carry import TargetSwapTracer


class _Stub:
	target_swap = True


def test_target_bowl_default_mapping():
	assert TwinMujoco.target_bowl(object(), "defective") == "pink"
	assert TwinMujoco.target_bowl(object(), "normal") == "green"


def test_target_swap_flips_the_mapping():
	s = _Stub()
	assert TargetSwapTracer.target_bowl(s, "defective") == "green"
	assert TargetSwapTracer.target_bowl(s, "normal") == "pink"
	s.target_swap = False
	assert TargetSwapTracer.target_bowl(s, "defective") == "pink"
```

- [ ] **Step 2：运行，确认失败**

运行：`.venv/Scripts/python -m pytest tests/test_vla_paired_carry.py -q`
预期：FAIL，`AttributeError: type object 'TwinMujoco' has no attribute 'target_bowl'`，或者 `ModuleNotFoundError`。

- [ ] **Step 3：实现**

`twin_mujoco.py` 中，在 `place` 之前加：

```python
	def target_bowl(self, label: str) -> str:
		"""The bowl a brick with ``label`` goes to (subclasses may remap it, see vla/paired_carry.py)."""
		return "pink" if label == "defective" else "green"
```

然后替换写死的映射：

- `place` 中：`ex.bowl_xy = np.array(self.bowls[self.target_bowl(label)])`，以及 `target = self.target_bowl(label)`；
- `evaluate` 中：

```python
		expected = "defective" if self.layout.defective else "normal"
		true_bowl = self.target_bowl(expected)
		target = self.layout.pink if true_bowl == "pink" else self.layout.green
		...
		ok = label == expected and self.actual_destination == true_bowl
```

- 在 `result` 字典末尾加 `**({"target_swap": True} if getattr(self, "target_swap", False) else {})`。

基类行为必须完全不变：`target_bowl` 的默认映射与原先相同。

新建 `src/digital_twin_armigo/vla/paired_carry.py`：

```python
"""Paired opposite-target carry demonstrations (docs/vla/BOWL_CHANNEL_PAIRED_PLAN_2026-11-23.md).

The expert runs a train layout to its held ``present_bottom`` checkpoint; one branch places the brick by its label
(the normal demonstration), the other into the other bowl. Both branches share the physical state at the fork, so
the target-bowl instruction is the only input that tells their carries apart. Only carry and release frames are
supervised."""

from __future__ import annotations

from ..agent.perturbations import HeldCheckpointExpertTracer

SWAP = {"green": "pink", "pink": "green"}


class TargetSwapTracer(HeldCheckpointExpertTracer):
	"""Expert tracer whose brick goes to the other bowl (``target_swap=True``); its own verdict is judged against
	that bowl and its result carries ``target_swap: True``."""

	def __init__(self, *args, target_swap: bool = True, **kwargs):
		self.target_swap = bool(target_swap)
		super().__init__(*args, **kwargs)

	def target_bowl(self, label: str) -> str:
		bowl = "pink" if label == "defective" else "green"
		return SWAP[bowl] if self.target_swap else bowl
```

- [ ] **Step 4：运行，确认通过，并跑回归**

```
.venv/Scripts/python -m pytest tests/test_vla_paired_carry.py tests/test_vla_s5_perturb.py tests/test_grasp_tail.py -q
```

预期：全部 PASS，现有扰动和 grasp 测试不受影响。

- [ ] **Step 5：提交**

```
git add src/digital_twin_armigo/agent/twin_mujoco.py src/digital_twin_armigo/vla/paired_carry.py tests/test_vla_paired_carry.py
git commit -m "TwinMujoco.target_bowl hook and TargetSwapTracer for paired-target carries"
```

---

### Task 3：只监督搬运段的掩码

**Files:**
- Modify: `src/digital_twin_armigo/vla/paired_carry.py`
- Test: `tests/test_vla_paired_carry.py`

**Interfaces:**
- Produces: `restrict_supervision(arrays: dict[str, np.ndarray], segment_kinds: list[str], keep: tuple[str, ...] = ("carry_to_bowl", "release_in_bowl"), from_frame: int = 0) -> dict[str, np.ndarray]`。返回新的数组字典，其中 `supervise_action` 只在 `keep` 段内且帧号 ≥ `from_frame` 处为 True。

**背景：**
- 配对回合的前缀（扫描、抓取、检查）和基础数据集里同一布局的回合几乎相同，重复监督只会加重前缀的权重。
- 训练加载器已经遵守 `mask.supervise_action`：不监督的帧不作为锚点，chunk 也不跨过它们（`vla/dataset.py` 605–622 行）。
- `actions.npz` 的 `segment` 列是每帧所在段的编号，-1 表示不在任何段内。段编号到种类的映射来自 `segment_labels(episode)["segments"]`。

- [ ] **Step 1：写失败的测试**

```python
import numpy as np

from digital_twin_armigo.vla.paired_carry import restrict_supervision


def test_restrict_supervision_keeps_only_carry_and_release():
	arrays = {
		"segment": np.array([0, 0, 1, 1, 2, 2, 3]),
		"supervise_action": np.ones(7, bool),
		"action": np.zeros((7, 6)),
	}
	kinds = ["grasp", "return_to_lift", "carry_to_bowl", "release_in_bowl"]
	out = restrict_supervision(arrays, kinds)
	assert out["supervise_action"].tolist() == [False, False, False, False, True, True, True]
	assert arrays["supervise_action"].all()  # input untouched
	out2 = restrict_supervision(arrays, kinds, from_frame=5)
	assert out2["supervise_action"].tolist() == [False] * 5 + [True, True]


def test_restrict_supervision_respects_existing_false():
	arrays = {"segment": np.array([2, 2]), "supervise_action": np.array([False, True])}
	out = restrict_supervision(arrays, ["a", "b", "carry_to_bowl"])
	assert out["supervise_action"].tolist() == [False, True]
```

- [ ] **Step 2：运行，确认失败**

运行：`.venv/Scripts/python -m pytest tests/test_vla_paired_carry.py -q`
预期：FAIL，报 `ImportError: cannot import name 'restrict_supervision'`。

- [ ] **Step 3：实现**

```python
import numpy as np

CARRY_KINDS = ("carry_to_bowl", "release_in_bowl")


def restrict_supervision(arrays, segment_kinds, keep=CARRY_KINDS, from_frame=0):
	"""Copy of ``arrays`` whose ``supervise_action`` is True only on frames of a ``keep`` segment at or after
	``from_frame`` (and where it was True before). ``segment_kinds[i]`` is the kind of segment id ``i``."""
	seg = np.asarray(arrays["segment"], dtype=np.int64)
	n = seg.shape[0]
	prior = np.asarray(arrays.get("supervise_action", np.ones(n, bool)), dtype=bool)
	kind_ok = np.array([0 <= s < len(segment_kinds) and segment_kinds[s] in keep for s in seg], dtype=bool)
	frame_ok = np.arange(n) >= int(from_frame)
	out = {k: np.array(v, copy=True) for k, v in arrays.items()}
	out["supervise_action"] = prior & kind_ok & frame_ok
	return out
```

- [ ] **Step 4：运行，确认通过**

运行：`.venv/Scripts/python -m pytest tests/test_vla_paired_carry.py -q`
预期：PASS。

- [ ] **Step 5：提交**

```
git add src/digital_twin_armigo/vla/paired_carry.py tests/test_vla_paired_carry.py
git commit -m "restrict_supervision: carry-only supervision masks for paired demonstrations"
```

---

### Task 4：采集一对示范（单布局端到端）

**Files:**
- Modify: `src/digital_twin_armigo/vla/paired_carry.py`（加 `run_pair`）
- Create: `scripts/vla/collect_paired_carry.py`

**Interfaces:**
- Consumes:
  - `TargetSwapTracer`（Task 2）、`restrict_supervision`（Task 3）；
  - `twin_resume.load_plan(parent_dir)`、`twin_resume.restore(backend, plan)`；
  - `twin_record.stitch_perturbation_episode(parent, branch_id, out, branch_episode=...)`、`twin_record.student_frames`、`twin_record.segment_labels`。
- Produces: `run_pair(case, scorer, out_dir: Path) -> dict`。返回字段：`case`、`parent_verdict`、`swap_verdict`、`fork_frame`、`fork_state_match: bool`、`bowls: {"parent": str, "swap": str}`、`episodes: {"parent": str | None, "swap": str | None}`；后两个是可导出的回合目录。

**流程（跨进程续跑的方式，参照 `twin_mcp.py` 255–271 行）：**

1. **父回合。** 目录 `out_dir/raw/parent`，构造 `TargetSwapTracer(layout, scorer, EpisodeTrace(dir, video=True), sensing="realistic", target_swap=False)`，用 `build_graph(twin, Policy()).invoke({})` 正常跑完。持砖检查点会在 `present_bottom` 后自动存盘（`HeldCheckpointExpertTracer.inspect`）。
2. **跳过条件。** 父回合不是 SUCCESS、不 `student_eligible`，或者没有 `held is not None` 的检查点时，记录原因并返回，不出数据。
3. **翻转支。**
   - 读 `plan = twin_resume.load_plan(parent_dir)`，把 `plan.checkpoints` 截到最后一个持砖检查点（`[c for c in plan.checkpoints if c.held is not None][-1]` 及之前的检查点），并重新加载该检查点的 history 和空碗深度：`plan.history, plan.empty_bowl_depth = twin_resume.load_history(parent_dir, ck)`（这个函数返回一个二元组）；
   - 在 `out_dir/raw/swap` 新建 `TargetSwapTracer(..., target_swap=True)`，执行 `restore(backend, plan)`；
   - 用 `InMemorySaver` 建图，`graph.update_state(config, plan.graph_state(), as_node="prepare")` 后 `graph.invoke(None, config)`。
4. **分叉状态核对。** 翻转支恢复后的第一帧，`cell.tcp_pos()` 和 `cell.brick_pos()` 应与父回合在 `ck.frame` 处的记录一致，误差 ≤ 1e-6 m；把结果记入 `fork_state_match`。
5. **拼接。** `stitch_perturbation_episode(parent_dir, branch_id=ck.branch, out_dir=out_dir/"swap", branch_episode=swap_raw_dir)`，得到单分支的翻转回合。
6. **父回合导出副本。** 把 `parent_dir` 复制到 `out_dir/"parent"`。
7. **改写掩码。** 两个导出目录都要改写 `branch-0/actions.npz`：

```python
labels = segment_labels(ep_dir)["segments"]
kinds = [s["kind"] for s in labels]
with np.load(ep_dir / "branch-0/actions.npz") as z:
	arrays = {k: z[k] for k in z.files}
np.savez_compressed(ep_dir / "branch-0/actions.npz", **restrict_supervision(arrays, kinds, from_frame=ck.frame))
```

8. **导出前检查。** 两个目录都能 `next(student_frames(ep_dir))` 成功（这个函数会检查 SUCCESS、单分支和段标签），并且：
   - 翻转回合 `episode.json` 的 `result.target_swap` 为 True；
   - 两个回合 `carry_to_bowl` 段的 `params.bowl` 互为相反。

`scripts/vla/collect_paired_carry.py` 参数：

- `--suite`：默认 `config/vla/train.json`；用 `dagger.assert_train_layouts_only` 守护；
- `--out`、`--max-episodes`、`--workers`（spawn 进程池，参照 `run_dagger_iter.py` 的 `--rollout-workers`）、`--detector`；
- `--export`：对通过检查的回合调用 `twin_record.export_lerobot`，按批写分片，再用 `rollout_and_export.merge_shards_cmd` 合并到 `out/lerobot`。

每个布局写 `out/cases/case-XXXX/pair.json`。已存在的布局跳过，所以中断后可以续跑。

- [ ] **Step 1：实现 `run_pair` 和脚本**，按上面的流程写。
- [ ] **Step 2：冒烟 3 个训练布局**

```
.venv/Scripts/python scripts/vla/collect_paired_carry.py --out E:/scratch/paired-carry/smoke --max-episodes 3
```

预期：

- 至少 2 个布局的 `pair.json` 满足：`parent_verdict == swap_verdict == "SUCCESS"`、`fork_state_match` 为 true、`bowls.parent != bowls.swap`；
- 两个导出目录都通过 `student_frames` 检查。

- [ ] **Step 3：人工抽查 1 对**
  - 打开两个回合的 `events.jsonl`：翻转回合的 `placement_check` 应该在另一只碗里看到砖；
  - 两个回合搬运段起点的关节角相同；
  - `supervise_action` 只在搬运段和释放段为 True。
- [ ] **Step 4：导出冒烟**

```
E:/il/lerobot-cuda/Scripts/python.exe scripts/vla/collect_paired_carry.py --out E:/scratch/paired-carry/smoke --max-episodes 3 --export
```

预期：

- `E:/scratch/paired-carry/smoke/lerobot` 能用 `vla/dataset.py` 的加载器读出；
- 读出的训练样本中，只有 `carry_to_bowl` 和 `release_in_bowl` 帧作为锚点；
- `segment.bowl` 两种颜色各占一半。

- [ ] **Step 5：提交**

```
git add src/digital_twin_armigo/vla/paired_carry.py scripts/vla/collect_paired_carry.py
git commit -m "collect_paired_carry: same-start opposite-target carry demonstrations"
```

---

### Task 5：全量采集并合并训练集

**Files:**
- 无代码改动。数据和日志都放在 `E:/scratch/paired-carry/full`。

- [ ] **Step 1：在训练布局上全量采集**

```
E:/il/lerobot-cuda/Scripts/python.exe scripts/vla/collect_paired_carry.py --out E:/scratch/paired-carry/full --workers 4 --export
```

参考专家成功率 290/300，预计得到约 270 对、540 个只监督搬运段的回合。用 `run_in_background` 运行。

- [ ] **Step 2：数据报告**，写到 `E:/scratch/paired-carry/full/DATA.md`，包括：
  - 成功对数，以及各类失败的原因；
  - `fork_state_match` 全部为 true；
  - 两种颜色的回合数和帧数；
  - 搬运帧占比。
- [ ] **Step 3：与 fit30k 的训练集合并**

```python
from lerobot.datasets.dataset_tools import merge_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset

merge_datasets(
	[
		LeRobotDataset("local/armigo-six-face", root="E:/scratch/s11v2-01/iter-01/d_union"),
		LeRobotDataset("local/armigo-six-face", root="E:/scratch/paired-carry/full/lerobot"),
	],
	output_repo_id="local/armigo-six-face",
	output_dir="E:/scratch/paired-carry/d_union_paired",
)
```

用 lerobot-cuda 的 python 运行，并用 `vla/dataset.py` 的加载器读一遍。预期：

- 回合数为 310 加配对回合数；
- 读出的样本计数中，`carry_to_bowl` 的绿碗样本不少于粉碗样本的一半。

---

### Task 6：预注册验收协议与 val5

**Files:**
- Create: `config/vla/val5.json`
- Create: `docs/vla/BOWL_CHANNEL_VALIDATION_2026-11-23.md`

- [ ] **Step 1：生成 val5**

```
.venv/Scripts/python scripts/vla/make_suites.py --val 100 --val-start 1700001
```

把输出改名为 `config/vla/val5.json`，并把 `name` 改为 `vla-val5`、`split` 改为 `val5`，`note` 写明用途。改名方式与 val4 相同（参照 `config/vla/val4.json` 的格式）。

- [ ] **Step 2：写协议**，在任何训练结果出来之前写好，内容如下。
  - **对比组**：
    - T1 = 碗通道 + 配对数据（`E:/scratch/bowl-paired-30k`）；
    - 基线 = fit30k。
  - **主判据**：
    - val5 探针（`bowl_swap_probe.py` 与 `bowl_swap_report.py`，方法同 [BOWL_SWAP_PROBE_2026-11-23.md](BOWL_SWAP_PROBE_2026-11-23.md)）上，T1 的跟随率 ≥ 85%；
    - 同时 `req` 与 `req2` 的一致率 ≥ 90%。
    - 基线也在 val5 上跑探针，作为同一布局集上的对照，预期接近 0。
  - **次判据**（决定是否把 T1 设为默认权重）：val5 闭环配对评测，用 `eval_closed_loop.py` 默认设置，T1 对基线：
    - 最终分拣的点估计不下降；
    - 放置未确认中“落进另一只碗”的例数下降；
    - 首次夹稳下降不超过 5；
    - 不新增机械臂–环境碰撞。
    - 统计方法同 `clamp_wait_report.py`：精确 McNemar 检验和 bootstrap 区间。
  - **失败分支**：跟随率 < 85% 时不调参复跑，按本计划“若未达标”一节处理。
- [ ] **Step 3：提交**

```
git add config/vla/val5.json docs/vla/BOWL_CHANNEL_VALIDATION_2026-11-23.md
git commit -m "Bowl channel validation protocol and fresh suite val5 (pre-registered)"
```

---

### Task 7：训练 T1

- [ ] **Step 1：训练**（约 2.5 小时，用 `run_in_background`）

```
E:/il/lerobot-cuda/Scripts/python.exe scripts/vla/train.py --family act --dataset E:/scratch/paired-carry/d_union_paired --val-dataset E:/scratch/vla-pilot-val/lerobot --out E:/scratch/bowl-paired-30k --init-from E:/scratch/s11-long-03/best --bowl-token --steps 30000 --lr 3e-5 --takeover-weight 5.0 --eval-every 10000
```

除 `--dataset` 和 `--bowl-token` 外，其余参数与 fit30k 一致：lr 3e-5，takeover-weight 5，从 long-03 初始化。运行前先核对 `E:/scratch/s11v2-fit30k/train_summary.json` 里的 batch size、warmup 等，与 fit30k 不同的地方要么改成一致，要么在协议偏离记录里写明。

- [ ] **Step 2：核对**
  - `train_summary.json` 中的 `segment_kinds.carry_to_bowl` 比 fit30k 的 69885 多出约配对帧数；
  - `vla_config.json` 中 `bowl_token` 为 true；
  - 训练日志打印了扩展加载那一行。

---

### Task 8：验收评测与报告

- [ ] **Step 1：探针**，T1 和基线各跑一次，各约 35 分钟。

```
.venv/Scripts/python scripts/vla/bowl_swap_probe.py --suite config/vla/val5.json --ckpt E:/scratch/bowl-paired-30k/best --out E:/scratch/bowl-channel/val5-t1
.venv/Scripts/python scripts/vla/bowl_swap_probe.py --suite config/vla/val5.json --ckpt E:/scratch/s11v2-fit30k/best --out E:/scratch/bowl-channel/val5-base
.venv/Scripts/python scripts/vla/bowl_swap_report.py E:/scratch/bowl-channel/val5-t1
.venv/Scripts/python scripts/vla/bowl_swap_report.py E:/scratch/bowl-channel/val5-base
```

- [ ] **Step 2：闭环配对评测**，每组约 80 分钟。用 `--record-act-rgb` 和 `--pin-rgb-from` 共享前缀没有意义，因为两组权重不同，所以各自正常跑。

```
.venv/Scripts/python scripts/vla/eval_closed_loop.py --suite config/vla/val5.json --ckpt E:/scratch/bowl-paired-30k/best --out E:/scratch/bowl-channel/val5-eval-t1 --no-teacher
.venv/Scripts/python scripts/vla/eval_closed_loop.py --suite config/vla/val5.json --ckpt E:/scratch/s11v2-fit30k/best --out E:/scratch/bowl-channel/val5-eval-base --no-teacher
```

然后按逐例结果统计：首次夹稳、抬升、分拣、错碗数和碰撞，配对检验。

- [ ] **Step 3：写报告**：在协议文档里追加“结果”节，按预定判据给出结论；另把中文报告写到 `E:/scratch/bowl-channel/REPORT.md`。
- [ ] **Step 4：提交文档；更新记忆** `vla-grasp-roll-hidden-planner-state.md`。

---

## 若未达标（跟随率 < 85%）

按探针结果分流，每条都是新的一轮，需要另写协议：

1. **跟随率明显上升但不够**（比如 40–85%）：说明数据有效、通道偏弱。改用更强的条件方式，用碗嵌入对解码器的 query 或所有编码 token 做 FiLM / 加性调制；或者提高配对帧的样本权重。
2. **跟随率仍接近 0**：先确认训练时搬运段样本里 `OBS_ENV_STATE` 真的随碗变化，可以在 `_prepare_act_batch` 处抽样打印。数据确实没问题时，说明 ACT 解码器能绕开单个 token，此时转向 1 中的调制方案。
3. **跟随但闭环分拣下降**：看下降出在哪一段，搬运段以外的退化多半是训练步数被搬运样本稀释，可以按段重新配平采样。

## 可选的归因臂（不在验收路径上）

T1 达标后，如果想知道碗通道本身是否必要，可以再训一个 T0：同样的配对数据、不加 `--bowl-token`，约 2.5 小时，再在 val5 上跑探针比较跟随率。T0 也达标，就说明只靠数据就够，可以撤掉这项结构改动，以降低维护成本。

## 时间与资源估计

| 步骤 | 估计 |
| --- | --- |
| Task 1–4 编码、测试与冒烟 | 约半天 |
| Task 5 采集（4 进程）与合并 | 2–3 小时 |
| Task 7 训练 | 约 2.5 小时 |
| Task 8 探针 2 × 35 分钟，闭环 2 × 80 分钟 | 约 4 小时 |
