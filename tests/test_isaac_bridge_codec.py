"""Codec round trip and client protocol against a fake in-process server (no Isaac needed)."""

import pickle
import threading
from multiprocessing.connection import Listener

import numpy as np
import pytest

from digital_twin_armigo.isaac_bridge import BridgeError, IsaacBridge, codec


def test_roundtrip_arrays_and_scalars():
	obj = {
		"f32": np.arange(12, dtype=np.float32).reshape(3, 4),
		"u8": np.full((2, 2, 3), 7, dtype=np.uint8),
		"big_endian": np.arange(4, dtype=">f8"),
		"nested": [{"a": np.float64(1.5), "b": (1, 2)}, None, True, "x", b"raw"],
		"empty": np.zeros((0, 3)),
	}
	back = codec.loads(codec.dumps(obj))
	assert back["f32"].dtype == np.float32 and np.array_equal(back["f32"], obj["f32"])
	assert np.array_equal(back["u8"], obj["u8"])
	assert np.array_equal(back["big_endian"], np.arange(4, dtype="<f8"))
	assert back["nested"] == [{"a": 1.5, "b": [1, 2]}, None, True, "x", b"raw"]
	assert back["empty"].shape == (0, 3)
	assert back["f32"].flags.writeable


def test_non_contiguous_array():
	a = np.arange(24.0).reshape(4, 6)[:, ::2]
	assert np.array_equal(codec.loads(codec.dumps(a)), a)


def test_wire_has_no_numpy_classes():
	data = codec.dumps({"a": np.ones(3)})
	assert b"numpy" not in data


def test_rejects_unsupported():
	with pytest.raises(TypeError):
		codec.encode({1: 2})
	with pytest.raises(TypeError):
		codec.encode(np.array(["a"]))
	with pytest.raises(TypeError):
		codec.encode(object())


def test_unpickler_refuses_classes():
	class Evil:
		def __reduce__(self):
			return (print, ("pwned",))

	with pytest.raises(pickle.UnpicklingError):
		codec.loads(pickle.dumps(Evil(), protocol=4))
	with pytest.raises(pickle.UnpicklingError):
		codec.loads(pickle.dumps(np.ones(2), protocol=4))


class FakeServer(threading.Thread):
	"""Speaks the bridge protocol with a trivial 1-DoF 'physics'."""

	def __init__(self):
		super().__init__(daemon=True)
		self.key = b"k" * 16
		self.listener = Listener(("127.0.0.1", 0), authkey=self.key)
		self.address = self.listener.address
		self.q = np.zeros(6)
		self.frame = 0
		self.checkpoints = {}
		self.calls = []

	def run(self):
		conn = self.listener.accept()
		while True:
			msg = codec.recv(conn)
			cmd = msg["cmd"]
			self.calls.append(cmd)
			try:
				if cmd == "shutdown":
					codec.send(conn, {"ok": True, "result": None})
					break
				codec.send(conn, {"ok": True, "result": self.handle(cmd, msg)})
			except Exception as exc:  # noqa: BLE001
				codec.send(conn, {"ok": False, "error": str(exc), "trace": ""})
		conn.close()
		self.listener.close()

	def handle(self, cmd, msg):
		if cmd == "hello":
			return {"version": 1}
		if cmd == "reset":
			self.q = np.zeros(6)
			self.frame = 0
			return self.state()
		if cmd == "step":
			self.q = np.asarray(msg["ctrl"])
			self.frame += 1
			return self.state()
		if cmd == "state":
			return self.state()
		if cmd == "render":
			if msg["camera"] == "boom":
				raise RuntimeError("camera exploded")
			h, w = 4, 5
			return {
				"rgb": np.full((h, w, 3), self.frame, np.uint8),
				"depth": np.full((h, w), 0.5, np.float32) if msg["depth"] else None,
				"frame": self.frame,
				"sim_time": self.frame / 30,
				"cam_pose": np.eye(4),
				"intrinsics": {"fx": 1.0, "fy": 1.0, "cx": 2.0, "cy": 2.0, "width": w, "height": h},
			}
		if cmd == "checkpoint":
			tok = f"t{len(self.checkpoints)}"
			self.checkpoints[tok] = (self.q.copy(), self.frame)
			return {"token": tok}
		if cmd == "restore":
			self.q, self.frame = self.checkpoints[msg["token"]]
			self.q = self.q.copy()
			return self.state()
		if cmd == "save_replay":
			return None
		raise ValueError(f"unknown {cmd}")

	def state(self):
		return {"q": self.q.copy(), "frame": self.frame, "sim_time": self.frame / 30}


@pytest.fixture
def bridge():
	server = FakeServer()
	server.start()
	b = IsaacBridge(address=server.address, authkey=server.key, timeout_s=10)
	yield b, server
	b.close()
	server.join(5)


def test_client_protocol(bridge):
	b, server = bridge
	assert b.info == {"version": 1}
	assert b.reset({"scene": 1})["frame"] == 0
	ctrl = np.linspace(0, 0.5, 6)
	s = b.step(ctrl, branch=0, frame=0)
	assert s["frame"] == 1 and np.allclose(s["q"], ctrl)
	tok = b.checkpoint()
	b.step(ctrl * 2, branch=1, frame=1)
	assert b.state()["frame"] == 2
	assert b.restore(tok)["frame"] == 1
	r = b.render("top", depth=True)
	assert r["rgb"].shape == (4, 5, 3) and r["rgb"].dtype == np.uint8
	assert r["depth"].dtype == np.float32 and r["cam_pose"].shape == (4, 4)
	assert b.render("front")["depth"] is None
	b.save_replay("some_dir")
	assert server.calls[:3] == ["hello", "reset", "step"]


def test_client_validation_and_errors(bridge):
	b, _ = bridge
	with pytest.raises(ValueError):
		b.step(np.zeros(5), branch=0, frame=0)
	with pytest.raises(ValueError):
		b.step(np.full(6, np.nan), branch=0, frame=0)
	with pytest.raises(ValueError):
		b.render("nope")
	with pytest.raises(BridgeError, match="camera exploded"):
		b._call("render", camera="boom", depth=False, flush=None)
	assert b.state()["frame"] == 0  # the connection survives a server-side error


def test_bad_authkey_is_refused():
	server = FakeServer()
	server.start()
	with pytest.raises(Exception):  # noqa: B017 - AuthenticationError or BridgeError after the timeout
		IsaacBridge(address=server.address, authkey=b"x" * 16, timeout_s=3)
