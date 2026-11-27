"""Wire format of the Isaac bridge: plain dicts / lists / str / float / int / bytes only, arrays as ``__nd__`` dicts.

The two ends run different Python and numpy versions, so numpy objects are never pickled. Arrays travel as
``{"__nd__": "<f4", "shape": [h, w], "data": b"..."}`` (C order, little endian); every message is a pickle (protocol 4)
of such plain data, read back through an unpickler that refuses every class. This file is numpy-only and compatible
with Python 3.11 so the Isaac server can import it by path.
"""

from __future__ import annotations

import io
import pickle

import numpy as np

ND = "__nd__"
PROTOCOL = 4


def encode(obj):
	"""``obj`` with every numpy array / scalar replaced by plain data (recursively)."""
	if isinstance(obj, np.ndarray):
		arr = np.ascontiguousarray(obj)
		if arr.dtype.kind not in "biuf":
			raise TypeError(f"unsupported array dtype {arr.dtype}")
		arr = arr.astype(arr.dtype.newbyteorder("<"), copy=False)
		return {ND: arr.dtype.str, "shape": [int(s) for s in arr.shape], "data": arr.tobytes()}
	if isinstance(obj, np.generic):
		return obj.item()
	if isinstance(obj, dict):
		for k in obj:
			if not isinstance(k, str):
				raise TypeError(f"dict keys must be str, got {k!r}")
		return {k: encode(v) for k, v in obj.items()}
	if isinstance(obj, (list, tuple)):
		return [encode(v) for v in obj]
	if obj is None or isinstance(obj, (bool, int, float, str, bytes)):
		return obj
	raise TypeError(f"cannot encode {type(obj).__name__}")


def decode(obj):
	"""Inverse of :func:`encode`: ``__nd__`` dicts become (writable) numpy arrays again."""
	if isinstance(obj, dict):
		if ND in obj:
			arr = np.frombuffer(obj["data"], dtype=np.dtype(obj[ND])).reshape(obj["shape"])
			return arr.copy()
		return {k: decode(v) for k, v in obj.items()}
	if isinstance(obj, list):
		return [decode(v) for v in obj]
	return obj


class _PlainUnpickler(pickle.Unpickler):
	def find_class(self, module, name):
		raise pickle.UnpicklingError(f"bridge messages are plain data, refusing {module}.{name}")


def dumps(obj) -> bytes:
	return pickle.dumps(encode(obj), protocol=PROTOCOL)


def loads(data: bytes):
	return decode(_PlainUnpickler(io.BytesIO(data)).load())


def send(conn, obj) -> None:
	conn.send_bytes(dumps(obj))


def recv(conn):
	return loads(conn.recv_bytes())
