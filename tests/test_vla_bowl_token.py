"""ACT bowl token (docs/vla/BOWL_CHANNEL_PAIRED_PLAN_2026-11-23.md Task 1).

Needs torch + lerobot: ``E:\\il\\lerobot-cuda\\Scripts\\python.exe tests/test_vla_bowl_token.py``. Under the repo
``.venv`` (no torch) the module is skipped by pytest.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
	sys.path.insert(0, str(ROOT / "src"))

if importlib.util.find_spec("torch") is None or importlib.util.find_spec("lerobot") is None:
	import pytest

	pytest.skip("needs torch and lerobot (run under E:/il/lerobot-cuda)", allow_module_level=True)

import torch

from digital_twin_armigo.vla.contract import STATE_DIM, STUDENT_CAMERAS, encode_state
from digital_twin_armigo.vla.models import BOWL_SLICE, POS_EMBED_KEY, ACTWithEndHead, load_act_weights_expanding

SHAPES = {cam: (3, 32, 32) for cam in STUDENT_CAMERAS}
ENV_KEY = "observation.environment_state"


def _model(bowl_token: bool) -> ACTWithEndHead:
	torch.manual_seed(0)
	return ACTWithEndHead(horizon=8, tiny_for_test=True, camera_shapes=SHAPES, bowl_token=bowl_token)


def _batch(bowl: str) -> dict:
	q = np.zeros(6)
	s = encode_state(q, q, q, 0.5, "carry_to_bowl", bowl=bowl)
	torch.manual_seed(1)
	b = {cam: torch.rand(1, *SHAPES[cam]) for cam in STUDENT_CAMERAS}
	b["observation.state"] = torch.from_numpy(s)[None]
	return b


def test_bowl_token_splits_the_state():
	m = _model(True)
	act = m._prepare_act_batch(_batch("pink"), include_action=False)
	assert act[ENV_KEY].tolist() == [[0.0, 0.0, 1.0]]
	assert torch.all(act["observation.state"][:, BOWL_SLICE] == 0)


def test_without_bowl_token_the_batch_is_unchanged():
	m = _model(False)
	act = m._prepare_act_batch(_batch("pink"), include_action=False)
	assert ENV_KEY not in act
	assert act["observation.state"].shape == (1, STATE_DIM)
	assert act["observation.state"][0, BOWL_SLICE].tolist() == [0.0, 0.0, 1.0]  # identity normalizer


def test_bowl_token_changes_the_prediction():
	m = _model(True)
	torch.nn.init.normal_(m.policy.model.encoder_env_state_input_proj.weight, std=1.0)
	a_green, _ = m.predict_chunk(_batch("green"))
	a_pink, _ = m.predict_chunk(_batch("pink"))
	assert not torch.allclose(a_green, a_pink)


def test_bowl_token_forward_backward_reaches_the_projection():
	m = _model(True)
	m.train()
	b = _batch("green")
	b["action"] = torch.zeros(1, 8, 6)
	b["action_mask"] = torch.ones(1, 8, dtype=torch.bool)
	b["end_step_target"] = torch.tensor([3])
	loss, _ = m(b)
	loss.backward()
	assert m.policy.model.encoder_env_state_input_proj.weight.grad.abs().sum() > 0


def test_checkpoint_roundtrip_keeps_bowl_token(tmp_path):
	m = _model(True)
	m.save_checkpoint(tmp_path / "new")
	back = ACTWithEndHead.load_checkpoint(tmp_path / "new")
	assert back.bowl_token is True
	for k, v in m.state_dict().items():
		assert torch.equal(back.state_dict()[k], v), k
	old = _model(False)
	old.save_checkpoint(tmp_path / "old")
	assert ACTWithEndHead.load_checkpoint(tmp_path / "old").bowl_token is False


def test_expanding_load_from_a_model_without_the_token():
	old, new = _model(False), _model(True)
	torch.manual_seed(5)
	for p in old.parameters():
		torch.nn.init.normal_(p)
	fresh = load_act_weights_expanding(new, old.state_dict())
	assert "policy.model.encoder_env_state_input_proj.weight" in fresh
	pe_old = old.state_dict()[POS_EMBED_KEY]
	pe_new = new.state_dict()[POS_EMBED_KEY]
	assert pe_new.shape[0] == pe_old.shape[0] + 1
	assert torch.equal(pe_new[: pe_old.shape[0]], pe_old)
	for k, v in old.state_dict().items():
		if k != POS_EMBED_KEY:
			assert torch.equal(new.state_dict()[k], v), k
	for bad_target, src in ((_model(False), new.state_dict()), (_model(True), new.state_dict())):
		try:
			load_act_weights_expanding(bad_target, src)
		except ValueError:
			pass
		else:
			raise AssertionError("expected ValueError")


def _run_all_as_script() -> int:
	import inspect

	tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
	for name, fn in tests:
		if "tmp_path" in inspect.signature(fn).parameters:
			with tempfile.TemporaryDirectory() as td:
				fn(Path(td))
		else:
			fn()
		print(f"PASS {name}")
	print(f"\n{len(tests)}/{len(tests)} tests PASSED")
	return 0


if __name__ == "__main__":
	raise SystemExit(_run_all_as_script())
