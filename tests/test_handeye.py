import numpy as np

from digital_twin_armigo.mujoco_cell import handeye as he


def _rot(rng):
	q = rng.normal(size=4)
	q /= np.linalg.norm(q)
	w, x, y, z = q
	return np.array(
		[
			[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
			[2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
			[2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
		]
	)


def _scene(noise_px=0.0, seed=0):
	rng = np.random.default_rng(seed)
	k = np.array([[615.0, 0, 319.5], [0, 615.0, 239.5], [0, 0, 1]])
	# camera 0.6 m in front of and above the robot, looking back at it
	look = np.array([-0.6, 0.0, -0.45])
	z = look / np.linalg.norm(look)
	x = np.cross(z, [0, 0, 1.0])
	x /= np.linalg.norm(x)
	world_from_cam = he.transform(np.column_stack([x, np.cross(z, x), z]), np.array([0.65, 0.05, 0.5]))
	grip_from_card = he.transform(_rot(np.random.default_rng(7)), np.array([0.05, 0.0, 0.004]))
	obj = np.array([[a, b, 0.0] for a in (-0.03, 0.0, 0.03) for b in (-0.03, 0.0, 0.03)])
	obs = []
	for _ in range(12):
		# card roughly facing the camera, randomly tilted
		face = he.from_rvec(rng.normal(0, 0.35, 3), np.zeros(3))[:3, :3]
		world_from_card = he.transform(
			world_from_cam[:3, :3] @ np.diag([1, -1, -1]) @ face, rng.uniform([0.2, -0.15, 0.1], [0.4, 0.15, 0.3])
		)
		world_from_grip = world_from_card @ he.inverse(grip_from_card)
		img = he.project(k, he.inverse(world_from_cam), (world_from_card @ np.c_[obj, np.ones(len(obj))].T).T[:, :3])
		obs.append(he.Observation(world_from_grip, obj, img + rng.normal(0, noise_px, img.shape)))
	return k, world_from_cam, grip_from_card, obs


def test_calibrate_recovers_exact_poses():
	k, world_from_cam, grip_from_card, obs = _scene()
	cal = he.calibrate(k, obs)
	assert cal.rms_px < 1e-4
	assert np.linalg.norm(cal.world_from_cam[:3, 3] - world_from_cam[:3, 3]) < 1e-6
	assert he.rotation_error_deg(cal.world_from_cam, world_from_cam) < 1e-5
	assert np.linalg.norm(cal.grip_from_card[:3, 3] - grip_from_card[:3, 3]) < 1e-6


def test_calibrate_with_pixel_noise_stays_millimetre_accurate():
	k, world_from_cam, _, obs = _scene(noise_px=0.3, seed=3)
	cal = he.calibrate(k, obs)
	assert 0.1 < cal.rms_px < 0.6
	assert np.linalg.norm(cal.world_from_cam[:3, 3] - world_from_cam[:3, 3]) < 0.003
	assert he.rotation_error_deg(cal.world_from_cam, world_from_cam) < 0.2


def test_table_points_inverts_projection():
	k, world_from_cam, _, _ = _scene()
	pts = np.array([[0.2, -0.1, 0.01], [0.35, 0.12, 0.01]])
	uv = he.project(k, he.inverse(world_from_cam), pts)
	assert np.allclose(he.table_points(k, world_from_cam, uv, 0.01), pts)


def _toy_fk(q):
	"""A five-joint arm with the SO-101's axis layout: pan about z, three pitches about y, roll about x."""

	def rz(a):
		return he.transform(np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]), np.zeros(3))

	def ry(a):
		return he.transform(np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]), np.zeros(3))

	def rx(a):
		return he.transform(np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]]), np.zeros(3))

	def tr(x, z):
		return he.transform(np.eye(3), np.array([x, 0.0, z]))

	return rz(q[0]) @ tr(0, 0.1) @ ry(q[1]) @ tr(0.11, 0) @ ry(q[2]) @ tr(0.13, 0) @ ry(q[3]) @ tr(0.06, 0) @ rx(q[4])


def test_calibrate_recovers_pitch_joint_offsets():
	rng = np.random.default_rng(5)
	k, world_from_cam, _, _ = _scene()
	grip_from_card = he.transform(np.eye(3), np.array([0.05, 0.0, 0.004]))
	obj = np.array([[a, b, 0.0] for a in (-0.03, 0.0, 0.03) for b in (-0.03, 0.0, 0.03)])
	bias = np.radians([0.7, -0.8, 0.6, 1.1, -0.5])
	obs = []
	while len(obs) < 25:
		q = rng.uniform([-0.6, -0.4, 0.3, -0.6, -1.0], [0.6, 0.6, 1.4, 0.8, 1.0])
		card = _toy_fk(q) @ grip_from_card
		cam_from_card = he.inverse(world_from_cam) @ card
		if cam_from_card[2, 3] < 0.2 or cam_from_card[2, 2] > -0.3:  # in front of the camera, facing it
			continue
		img = he.project(k, he.inverse(world_from_cam), (card @ np.c_[obj, np.ones(len(obj))].T).T[:, :3])
		obs.append(he.Observation(_toy_fk(q + bias), obj, img, q=q + bias))
	cal = he.calibrate(k, obs, fk=_toy_fk, offset_joints=he.OFFSET_JOINTS)
	assert cal.rms_px < 1e-3, cal.rms_px
	assert np.allclose(cal.joint_offsets[list(he.OFFSET_JOINTS)], -bias[list(he.OFFSET_JOINTS)], atol=np.radians(0.01))
	# the pan offset stays in the solution as a turn of the camera about the pan axis; the roll offset in the card
	fixed = he.calibrate(k, obs)
	assert fixed.rms_px > 0.1  # the bias leaves only a small residual, which is why it goes unnoticed


def test_card_texture_write_is_atomic_under_concurrent_readers(tmp_path):
	"""Rollout workers rewrite the shared card PNG while others build cells; readers must never see a partial file."""
	import threading

	import cv2

	he.write_card_texture(tmp_path)
	target = tmp_path / he.CARD_TEXTURE
	good_size = target.stat().st_size
	stop = threading.Event()
	bad_reads: list[int] = []

	def reader():
		while not stop.is_set():
			try:
				data = target.read_bytes() if target.exists() else b""
			except PermissionError:  # Windows: the file is momentarily mid-replace; a reader just retries
				continue
			if len(data) != good_size or cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE) is None:
				bad_reads.append(len(data))

	readers = [threading.Thread(target=reader) for _ in range(2)]
	for t in readers:
		t.start()
	try:
		for _ in range(30):
			he.write_card_texture(tmp_path)
	finally:
		stop.set()
		for t in readers:
			t.join()
	assert bad_reads == []
	assert sorted(p.name for p in tmp_path.iterdir()) == [he.CARD_TEXTURE]


def test_marker_textures_survive_parallel_first_writes(tmp_path):
	"""After the cache is cleared, parallel cell builds all regenerate the marker PNGs; every file must decode."""
	import threading

	import cv2

	from digital_twin_armigo.mujoco_cell import scene

	ids = {10, 11, 16, 17}
	errors: list[BaseException] = []

	def build():
		try:
			scene.write_marker_textures(tmp_path, ids)
		except BaseException as exc:  # noqa: BLE001 - surfaced by the assertion below
			errors.append(exc)

	for _ in range(5):
		for p in tmp_path.glob("_marker_*.png"):
			p.unlink()
		threads = [threading.Thread(target=build) for _ in range(8)]
		for t in threads:
			t.start()
		for t in threads:
			t.join()
		assert errors == []
		for i in ids:
			img = cv2.imread(str(tmp_path / f"_marker_{i}.png"), cv2.IMREAD_GRAYSCALE)
			assert img is not None and img.shape == (240, 240)
		assert sorted(p.name for p in tmp_path.iterdir()) == sorted(f"_marker_{i}.png" for i in ids)
