"""The optional ``mujoco`` import, with a headless GL backend selected where one is needed.

On Linux (WSL2 on the training PC: ``MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA``) offscreen rendering needs
``MUJOCO_GL=egl``; MuJoCo rejects that value on Windows and macOS, where its default backend already renders
offscreen. An explicit ``MUJOCO_GL`` in the environment always wins. ``mujoco`` is ``None`` when the package is not
installed, so the scene and layout code stays importable for the hardware-free tests.
"""

from __future__ import annotations

import os
import platform

if platform.system() == "Linux":
	os.environ.setdefault("MUJOCO_GL", "egl")
try:
	import mujoco
except ModuleNotFoundError:
	mujoco = None  # type: ignore[assignment]

__all__ = ["mujoco"]
