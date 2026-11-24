"""Client of the Isaac bridge server (``scripts/isaac/bridge_server.py``): Isaac PhysX as a second physics engine.

The server runs in the Isaac Lab environment (``E:\\il\\v``); this client runs in the main environment and talks to it
over :mod:`multiprocessing.connection` on localhost (see :mod:`.codec` for the wire format). Conventions:

- joints are in :data:`digital_twin_armigo.constants.JOINTS` order (gripper last), radians, world = robot base frame
  with the table top at ``z = 0``, metres, quaternions ``xyzw``;
- ``render()`` returns ``cam_pose`` as a 4x4 world-from-camera matrix in the **MuJoCo / OpenGL camera convention**
  (camera x = image right, y = image up, z = backwards, the camera looks along ``-z``), identical to the columns of
  MuJoCo's ``cam_xmat``. To project a world point ``p``: ``c = inv(cam_pose) @ [p, 1]``, ``u = cx + fx * c.x / -c.z``,
  ``v = cy - fy * c.y / -c.z`` (pixel ``u`` right, ``v`` down, pixel centres at ``+0.5``);
- depth is metres along the optical axis (``distance_to_image_plane``), ``inf``/``0`` where nothing was hit.
"""

from __future__ import annotations

import atexit
import os
import secrets
import socket
import subprocess
import sys
import time
from dataclasses import asdict
from multiprocessing.connection import Client
from pathlib import Path

import numpy as np

from . import codec

ROOT = Path(__file__).resolve().parents[3]
SERVER_SCRIPT = ROOT / "scripts" / "isaac" / "bridge_server.py"
ISAAC_PYTHON = r"E:\il\v\Scripts\python.exe"
SCENE_VERSION = 1
CAMERAS = ("top", "front", "wrist_cam")


class BridgeError(RuntimeError):
	"""The bridge server refused or failed a request, or died."""


class IsaacBridge:
	"""One Isaac physics server process and the connection to it.

	``address=(host, port)`` plus ``authkey`` (bytes) connect to an already running server instead of spawning one
	(used by the unit tests with an in-process fake server).
	"""

	def __init__(
		self,
		*,
		python: str = ISAAC_PYTHON,
		headless: bool = True,
		log_dir: str | Path | None = None,
		timeout_s: float = 600,
		address: tuple[str, int] | None = None,
		authkey: bytes | None = None,
	) -> None:
		self.timeout_s = float(timeout_s)
		self.proc: subprocess.Popen | None = None
		self._log = None
		self._conn = None
		self.log_path: Path | None = None
		if address is None:
			if log_dir is None:
				log_dir = ROOT / "results" / "isaac-bridge" / "logs"
			log_dir = Path(log_dir)
			log_dir.mkdir(parents=True, exist_ok=True)
			self.log_path = log_dir / "bridge_server.log"
			with socket.socket() as s:
				s.bind(("127.0.0.1", 0))
				port = s.getsockname()[1]
			authkey = secrets.token_bytes(16)
			cmd = [
				str(python),
				str(SERVER_SCRIPT),
				"--port",
				str(port),
				"--authkey-hex",
				authkey.hex(),
				"--log-dir",
				str(log_dir),
			]
			if headless:
				cmd.append("--headless")
			env = {**os.environ, "OMNI_KIT_ACCEPT_EULA": "yes", "PYTHONUNBUFFERED": "1"}
			self._log = open(self.log_path, "wb")  # noqa: SIM115 - kept open for the server's lifetime, closed in close()
			flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
			self.proc = subprocess.Popen(
				cmd, stdout=self._log, stderr=subprocess.STDOUT, env=env, cwd=str(ROOT), creationflags=flags
			)
			address = ("127.0.0.1", port)
			atexit.register(self.close)
		if authkey is None:
			raise ValueError("authkey is required with an explicit address")
		self._connect(address, authkey)
		self.info = self._call("hello")

	# -- connection ---------------------------------------------------------------------------------------------
	def _log_tail(self, n: int = 40) -> str:
		if self.log_path is None or not self.log_path.is_file():
			return ""
		lines = self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
		return "\n".join(lines[-n:])

	def _connect(self, address, authkey: bytes) -> None:
		deadline = time.monotonic() + self.timeout_s
		last = None
		while time.monotonic() < deadline:
			if self.proc is not None and self.proc.poll() is not None:
				raise BridgeError(f"bridge server exited with code {self.proc.returncode}\n{self._log_tail()}")
			try:
				self._conn = Client(address, authkey=authkey)
				return
			except (ConnectionRefusedError, ConnectionResetError, OSError) as exc:
				last = exc
				time.sleep(0.5)
		self.close()
		raise BridgeError(f"could not connect to the bridge server within {self.timeout_s:.0f} s: {last}")

	def _call(self, cmd: str, timeout_s: float | None = None, **kwargs):
		if self._conn is None:
			raise BridgeError("bridge is closed")
		timeout = self.timeout_s if timeout_s is None else timeout_s
		try:
			codec.send(self._conn, {"cmd": cmd, **kwargs})
			deadline = time.monotonic() + timeout
			while not self._conn.poll(1.0):
				if self.proc is not None and self.proc.poll() is not None:
					raise BridgeError(f"bridge server died during {cmd}\n{self._log_tail()}")
				if time.monotonic() > deadline:
					raise BridgeError(f"bridge request {cmd} timed out after {timeout:.0f} s")
			reply = codec.recv(self._conn)
		except (EOFError, ConnectionError, OSError) as exc:
			raise BridgeError(f"bridge connection lost during {cmd}: {exc}\n{self._log_tail()}") from exc
		if not reply.get("ok"):
			raise BridgeError(f"{cmd}: {reply.get('error')}\n{reply.get('trace', '')}")
		return reply.get("result")

	# -- API ----------------------------------------------------------------------------------------------------
	def reset(self, scene: dict) -> dict:
		"""Build (or reset to the initial state) the Isaac scene of ``scene`` (see :func:`scene_payload`); returns the
		state. Frame counter and sim time restart at 0, the recorded replay is cleared. Building a new scene takes
		tens of seconds; an identical scene is reset in place."""
		return self._call("reset", scene=scene)

	def step(self, ctrl, *, branch: int, frame: int) -> dict:
		"""Advance one 30 Hz control frame (8 PhysX substeps at 240 Hz) towards the 6 joint position targets ``ctrl``
		(radians, JOINTS order, gripper last). ``branch`` / ``frame`` label the recorded replay row."""
		ctrl = np.asarray(ctrl, dtype=np.float64)
		if ctrl.shape != (6,) or not np.isfinite(ctrl).all():
			raise ValueError("ctrl must be 6 finite joint targets")
		return self._call("step", ctrl=ctrl, branch=int(branch), frame=int(frame))

	def render(self, camera: str, *, depth: bool = False, flush: int | None = None) -> dict:
		"""Render ``camera`` ("top", "front", "wrist_cam") of the current state; physics time does not advance.

		Returns ``rgb`` (H, W, 3 uint8), ``depth`` ((H, W) float32 metres along the optical axis, or None),
		``frame``, ``sim_time``, ``cam_pose`` (4x4 world-from-camera, MuJoCo/OpenGL camera convention) and the
		``intrinsics`` (fx, fy, cx, cy, width, height). After a ``step`` / ``restore`` the server first pumps ``flush``
		render-only updates (default: the server's ``--flush-updates``, 3) because the RTX annotators lag one update."""
		if camera not in CAMERAS:
			raise ValueError(f"unknown camera {camera!r}; expected one of {CAMERAS}")
		return self._call("render", camera=camera, depth=bool(depth), flush=flush)

	def state(self) -> dict:
		return self._call("state")

	def stats(self) -> dict:
		"""Server counters: ``render_calls``, ``step_count`` and mean server-side ms per step phase."""
		return self._call("stats")

	def checkpoint(self) -> str:
		"""Token of the current state; only allowed with the gripper open and the brick supported by the table."""
		return self._call("checkpoint")["token"]

	def restore(self, token: str) -> dict:
		"""Return to a :meth:`checkpoint`; returns the state."""
		return self._call("restore", token=str(token))

	def save_replay(self, directory: str | Path) -> None:
		"""Write ``<directory>/replay/scene.json`` and ``<directory>/replay/motion.npz`` (the MotionCapture contract plus
		``branch`` / ``frame`` / ``sim_time`` arrays, one row per control frame)."""
		self._call("save_replay", directory=str(Path(directory).resolve()))

	def close(self) -> None:
		conn, self._conn = self._conn, None
		if conn is not None:
			try:
				codec.send(conn, {"cmd": "shutdown"})
				conn.poll(10)
			except Exception:  # noqa: BLE001, S110 - best effort, the process is killed below anyway
				pass
			try:
				conn.close()
			except Exception:  # noqa: BLE001, S110
				pass
		if self.proc is not None:
			try:
				self.proc.wait(timeout=30)
			except subprocess.TimeoutExpired:
				self.proc.kill()
			self.proc = None
		if self._log is not None:
			self._log.close()
			self._log = None

	def __enter__(self) -> IsaacBridge:  # noqa: PYI034
		return self

	def __exit__(self, *exc: object) -> None:
		self.close()


# -- scene description -----------------------------------------------------------------------------------------------
def _camera_spec(model, data, name: str) -> dict:
	"""World pose (MuJoCo convention) and intrinsics of the model camera ``name`` at the current ``data`` state."""
	import math

	from digital_twin_armigo.mujoco_cell._mj import mujoco
	from digital_twin_armigo.mujoco_cell.cell import RENDER_SIZE

	cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
	if cid < 0:
		raise ValueError(f"cell has no camera {name!r}")
	width, height = (int(v) for v in model.cam_resolution[cid])
	if width * height <= 1:  # MuJoCo's default 1x1: the cell renders every camera at RENDER_SIZE
		width, height = RENDER_SIZE
	if model.cam_sensorsize[cid][0] > 0:
		fx = float(model.cam_intrinsic[cid][0] / model.cam_sensorsize[cid][0] * width)
		fy = float(model.cam_intrinsic[cid][1] / model.cam_sensorsize[cid][1] * height)
	else:
		fy = 0.5 * height / math.tan(math.radians(float(model.cam_fovy[cid])) / 2.0)
		fx = fy
	return {
		"width": width,
		"height": height,
		"fx": fx,
		"fy": fy,
		"cx": width / 2.0,
		"cy": height / 2.0,
		"pos": np.asarray(data.cam_xpos[cid], dtype=np.float64).copy(),
		"rot": np.asarray(data.cam_xmat[cid], dtype=np.float64).reshape(3, 3).copy(),
	}


def _defect_payload(layout) -> dict:
	"""The defect of ``layout`` as boxes in the brick frame (same placement as the MuJoCo ``defect_<k>`` geoms)."""
	import math

	from digital_twin_armigo.mujoco_cell import scene as mj_scene
	from digital_twin_armigo.mujoco_cell._mj import mujoco
	from digital_twin_armigo.presentation import FACES, face_frame

	out = {
		"kind": layout.defect_kind,
		"face": int(layout.defect_face),
		"seed": int(layout.defect_seed),
		"visible": bool(layout.defective),
		"boxes": [],
		"rgb": [0.05, 0.05, 0.08],
	}
	if layout.defect_kind != "cross_var":
		raise NotImplementedError(f"Isaac bridge supports defect_kind 'cross_var' only, got {layout.defect_kind!r}")
	centre, normal, u, v = face_frame(layout.brick_size, layout.defect_face)
	strokes, ink, _ = mj_scene._cross_var_sample(layout)
	stud_top_offset = 0.0035  # visual brick mesh, as in mj_scene._brick
	lift = mj_scene._DEFECT_LIFT_M + (stud_top_offset if FACES[layout.defect_face] == (2, 1) else 0.0)
	for k, ((cu, cv), ang, half_length, half_width) in enumerate(strokes):
		d = math.cos(ang) * u + math.sin(ang) * v
		s = np.cross(normal, d)
		pos = centre + cu * u + cv * v + (lift + 0.00002 * k) * normal
		rot = np.column_stack([d, s, np.cross(d, s)])
		wxyz = np.zeros(4)
		mujoco.mju_mat2Quat(wxyz, rot.flatten())
		out["boxes"].append(
			{
				"pos": pos,
				"quat_xyzw": wxyz[[1, 2, 3, 0]],
				"half": [float(half_length), float(half_width), float(mj_scene._PATCH_HALF_T_M)],
			}
		)
	out["rgb"] = [float(c) for c in ink]
	return out


def scene_payload(layout, cell) -> dict:
	"""The :meth:`IsaacBridge.reset` dict of a ``twin_mujoco`` layout (``sample_case`` output) and its MuJoCo ``Cell``.

	Everything the Isaac scene needs: brick size class, dimensions, mass and initial pose (the pose ``PhysicsEpisode``
	puts the brick at), bowl centres and radius, table, the ``cross_var`` defect as boxes in the brick frame, the three
	cameras (``top`` / ``front`` world pose + intrinsics read from the cell's model, ``wrist_cam`` mount in the
	``camera_mount`` body frame including ``layout.wrist_cam_offset`` + intrinsics), the initial joint configuration
	and the :class:`~digital_twin_armigo.physics.PhysicsSpec`. Markers, distractors (visual only) are not reproduced in
	Isaac; clutter is not supported.
	"""
	import math

	from digital_twin_armigo.constants import BOWL_R, FPS
	from digital_twin_armigo.physics import brick_spec_for_size, load_physics_spec

	spec = cell.physics or load_physics_spec()
	bspec = brick_spec_for_size(layout.brick_size, spec)
	if bspec.name not in spec.bricks:
		raise NotImplementedError(f"no Isaac brick asset for size {layout.brick_size}")
	if layout.clutter:
		raise NotImplementedError("clutter bricks are not supported by the Isaac bridge")
	from digital_twin_armigo.mujoco_cell._mj import mujoco

	m = cell.model
	d = mujoco.MjData(m)  # world-fixed cameras only: independent of the arm state
	mujoco.mj_forward(m, d)
	yaw = math.radians(layout.brick_yaw_deg)
	cameras = {"top": _camera_spec(m, d, "top"), "front": _camera_spec(m, d, "front")}
	wrist = _camera_spec(m, d, "wrist_cam")
	pos, quat = cell.wrist_cam_actual
	wrist["mount_pos"] = np.asarray(pos, dtype=np.float64).copy()
	wrist["mount_quat_wxyz"] = np.asarray(quat, dtype=np.float64).copy()
	wrist["mount_body"] = "camera_mount"
	del wrist["pos"], wrist["rot"]
	cameras["wrist_cam"] = wrist
	from digital_twin_armigo.agent.twin_mujoco import INITIAL_Q

	return {
		"version": SCENE_VERSION,
		"fps": FPS,
		"physics_hz": 240,
		"warmup_steps": 12,
		"brick": {
			"size_key": bspec.name,
			"size_m": [float(v) for v in layout.brick_size],
			"mass_kg": float(bspec.mass_kg),
			"pos": [float(layout.brick_xy[0]), float(layout.brick_xy[1]), float(layout.brick_size[2] / 2.0 + 0.0005)],
			"quat_xyzw": [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)],
			"rgb": [float(c) for c in layout.brick_rgb],
		},
		"bowls": {
			"green": {"xy": [float(layout.green[0]), float(layout.green[1])], "radius": float(BOWL_R)},
			"pink": {"xy": [float(layout.pink[0]), float(layout.pink[1])], "radius": float(BOWL_R)},
		},
		"table": (
			{"rgb": [float(c) for c in layout.table_rgb], "texture": "plain"}
			if getattr(layout, "table_texture", None) is not None
			else {"rgb": [float(c) for c in layout.table_rgb]}
		),
		"light_level": float(layout.light_level),
		"defect": _defect_payload(layout),
		"cameras": cameras,
		"q_init": np.asarray(INITIAL_Q, dtype=np.float64),
		"physics": asdict(spec),
		"ignored": {"markers": len(layout.markers), "distractors": len(layout.distractors)},
	}


def default_python() -> str:
	return ISAAC_PYTHON if Path(ISAAC_PYTHON).is_file() else sys.executable
