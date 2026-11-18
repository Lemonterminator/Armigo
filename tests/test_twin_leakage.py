import numpy as np
import pytest


def test_3d_context_does_not_encode_the_defect_or_its_face(tmp_path):
	pytest.importorskip("mujoco")
	pytest.importorskip("cv2")
	pytest.importorskip("av")
	from digital_twin_armigo.agent.twin_mujoco import TwinMujoco, sample_case
	from digital_twin_armigo.agent.twin_record import EpisodeTrace
	from digital_twin_armigo.inspection_scorer import FixedScorer

	geometries = []
	observations = []
	for i, (defective, face) in enumerate(((False, 5), (True, 5), (True, 0))):
		trace = EpisodeTrace(tmp_path / str(i), video=False)
		scorer = FixedScorer(lambda _camera, _rgb: -1, {"top": 0, "wrist_cam": 0})
		backend = TwinMujoco(sample_case(187271, defective, face), scorer, trace)
		try:
			geometries.append(backend._geometry())
			joints = backend.cell.joints().copy()
			observations.append(backend._capture("top"))
			assert np.array_equal(backend.cell.joints(), joints)
		finally:
			trace.close()
			backend.close()
	assert geometries[0] == geometries[1] == geometries[2]
	assert observations[0]["face_pixels"] == observations[1]["face_pixels"] == observations[2]["face_pixels"]
