import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def test_real_joint_conversion_round_trip_and_sampling_deterministic():
	sys.path.insert(0, str(REPO / "scripts" / "perception"))
	import realism_compare as rc

	meta_path = Path(
		"C:/Users/Jiang/.gemini/antigravity/scratch/hackjunction/artifacts/isaac_test/real_ref/frame_meta.json"
	)
	state = np.asarray(json.loads(meta_path.read_text(encoding="utf-8"))["observation.state"], dtype=np.float64)
	q_rad = rc.real_state_to_mjcf_rad(state)
	assert q_rad.shape == (6,)
	assert np.all(np.abs(q_rad) < np.pi + 0.1)
	back = rc.mjcf_rad_to_real_state(q_rad)
	assert np.allclose(back, state, atol=1e-3)

	a = rc.sample_frame_indices(37515, 16, seed=7)
	b = rc.sample_frame_indices(37515, 16, seed=7)
	assert a == b and len(a) == 16 and len(set(a)) == 16

	img = np.zeros((480, 640, 3), dtype=np.uint8)
	strip = rc.make_triptych(img, np.zeros((480, 640, 3), dtype=np.uint8), img)
	assert strip.shape == (480, 640 * 3, 3)
