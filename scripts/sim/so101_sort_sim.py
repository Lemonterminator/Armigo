"""MuJoCo sorting cell command line (``fit-camera`` / ``layout-prior`` / ``simulate`` / ``render``).

Thin launcher for :mod:`digital_twin_armigo.mujoco_cell`, runnable without installing the package; the public names
of that package are re-exported here for scripts that imported this file as a module. See
``digital_twin_armigo/mujoco_cell/cli.py`` for the subcommands.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from digital_twin_armigo.mujoco_cell import *  # noqa: F403 - compatibility exports for existing callers
from digital_twin_armigo.mujoco_cell import main

if __name__ == "__main__":
	raise SystemExit(main())
