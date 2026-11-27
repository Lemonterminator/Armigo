"""Batched (GPU) image augmentation matches the per-sample DataLoader path (docs/vla/SMOLVLA_BOWL_PROBE_2026-11-24.md)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from digital_twin_armigo.vla.dataset import StudentImageAugmentor, _batch_crop_resize


def _images(b=3, h=24, w=32, seed=0):
	g = torch.Generator().manual_seed(seed)
	return torch.rand(b, 3, h, w, generator=g)


def test_batch_crop_resize_matches_interpolate():
	x = _images()
	crops = [(2, 3, 20, 27), None, (0, 5, 22, 25)]
	got = _batch_crop_resize(x, crops)
	for i, crop in enumerate(crops):
		want = x[i] if crop is None else StudentImageAugmentor._crop_resize_torch(x[i], crop)
		assert torch.allclose(got[i], want, atol=1e-5), i


def _old_augment_rgb(aug, img):
	"""The per-sample RGB path as it was before the batched refactor (reference)."""
	import torch.nn.functional as F
	import torchvision.transforms.functional as TF

	rng = aug._rng
	if rng.random() < aug.rgb_crop_prob:
		_, h, w = img.shape
		target_area = h * w * float(rng.uniform(*aug.rgb_crop_scale))
		lr = (np.log(aug.rgb_crop_ratio[0]), np.log(aug.rgb_crop_ratio[1]))
		aspect = float(np.exp(rng.uniform(lr[0], lr[1])))
		cw = max(1, min(w, round(np.sqrt(target_area * aspect))))
		ch = max(1, min(h, round(np.sqrt(target_area / aspect))))
		top, left = int(rng.integers(0, h - ch + 1)), int(rng.integers(0, w - cw + 1))
		img = F.interpolate(TF.crop(img, top, left, ch, cw).unsqueeze(0), size=(h, w), mode="bilinear").squeeze(0)
		img = img.clamp(0.0, 1.0)
	img = TF.adjust_brightness(img, float(rng.uniform(1.0 - aug.brightness, 1.0 + aug.brightness)))
	img = TF.adjust_contrast(img, float(rng.uniform(1.0 - aug.contrast, 1.0 + aug.contrast)))
	img = TF.adjust_saturation(img, float(rng.uniform(1.0 - aug.saturation, 1.0 + aug.saturation)))
	img = TF.adjust_hue(img, float(rng.uniform(-aug.hue, aug.hue)))
	img = img + torch.randn_like(img) * aug.rgb_noise_std
	return img.clamp(0.0, 1.0)


def test_cpu_path_bitwise_unchanged_by_refactor():
	x = _images(b=6)
	new, old = StudentImageAugmentor(seed=11), StudentImageAugmentor(seed=11)
	for i in range(len(x)):
		torch.manual_seed(i)
		a = new._augment_rgb_torch(x[i])
		torch.manual_seed(i)
		b = _old_augment_rgb(old, x[i])
		assert torch.equal(a, b), i


@pytest.mark.parametrize("key", ["observation.images.wrist", "observation.images.top_depth_v2"])
def test_batch_matches_per_sample_with_same_parameters(key):
	"""Draw-for-draw: one augmentor per path with the same seed and noise off gives the same images."""
	x = _images(b=4)
	kw = {"rgb_noise_std": 0.0, "depth_noise_std": 0.0, "seed": 3}
	batch = StudentImageAugmentor(**kw).augment_batch({key: x})[key]
	per = StudentImageAugmentor(**kw)
	for i in range(len(x)):
		want = per({key: x[i]})[key]
		assert torch.allclose(batch[i], want, atol=2e-5), i


def test_batch_noise_and_range():
	x = _images(b=8)
	out = StudentImageAugmentor(seed=1).augment_batch({"observation.images.top_rgb_v2": x})
	y = out["observation.images.top_rgb_v2"]
	assert y.shape == x.shape and y.dtype == torch.float32
	assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0
	assert not torch.allclose(y, x)
