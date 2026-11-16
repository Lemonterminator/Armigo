"""Command line of the MuJoCo sorting cell.

Camera images follow dataset sort-depth-v2:

* ``top_rgb_v2`` / ``top_depth_v2``: the D415 colour view rendered at 640x480 and cropped to x 96..608, y 0..416, as the
  depth bridge does (``--crop 96 0 512 416``). The depth image uses the bridge's ``layers`` encoding: R = height above
  the table 0 → 4 cm, G = 0 → 30 cm, B = 255 where depth is valid.
* ``front`` and ``wrist``: 640x480 colour.

The robot model defaults to this repository's ``assets/so101_mjcf`` (MuJoCo Menagerie ``robotstudio_so101`` at commit
``370c43965323c0944b433b202aac5058b5f907a9`` plus the wrist camera mount); ``--menagerie`` points elsewhere.

Subcommands
-----------
``fit-camera``
	Fit the top camera pose to real ``top_depth_v2`` frames by matching the arm silhouette rendered at the recorded
	joint state (needs the dataset copy).
``layout-prior``
	Measure bowl centres and brick positions on the real first and last top frames (input for ``simulate --prior``).
``simulate``
	Run a scripted pick-and-place per episode with physics grasping and store the full 3D state per frame (``qpos``),
	the actions and the layout. Only the presentation planner renders here (segmentation, to score defect views).
``render``
	Replay the stored trajectories under random but plausible cameras and appearances, and write
	``offline_pipeline``-style episodes (``index.json`` + ``*.npz``) that ``lerobot_export.py`` turns into LeRobot v3.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..constants import REST_U, SO101_MJCF_DIR
from .calibration import fit_camera, layout_prior
from .pipeline import _log_setup, render_views, simulate


def main(argv: list[str] | None = None) -> int:
	parser = argparse.ArgumentParser(description="MuJoCo SO-101 sorting cell: fit, simulate and render synthetic data.")
	parser.add_argument("--menagerie", type=Path, default=SO101_MJCF_DIR, help="directory holding so101.xml")
	sub = parser.add_subparsers(dest="cmd", required=True)
	f = sub.add_parser("fit-camera")
	f.add_argument("dataset", type=Path)
	f.add_argument("--frames", type=int, nargs="+", default=[300, 1500, 4000, 9000, 15000, 22000, 30000])
	f.add_argument("--out", type=Path, required=True)
	s = sub.add_parser("simulate")
	s.add_argument("--camera", type=Path, required=True)
	s.add_argument("--out", type=Path, required=True)
	s.add_argument("--episodes", type=int, default=4)
	s.add_argument("--seed", type=int, default=0)
	s.add_argument("--rest", type=float, nargs=6, default=list(REST_U))
	s.add_argument("--prior", type=Path, help="layout prior JSON from the layout-prior subcommand")
	s.add_argument("--workers", type=int, default=1)
	lp = sub.add_parser("layout-prior")
	lp.add_argument("dataset", type=Path)
	lp.add_argument("--camera", type=Path, required=True)
	lp.add_argument("--out", type=Path, required=True)
	r = sub.add_parser("render")
	r.add_argument("--camera", type=Path, required=True)
	r.add_argument("--sim", type=Path, required=True)
	r.add_argument("--out", type=Path, required=True)
	r.add_argument("--views", type=int, default=3)
	r.add_argument("--seed", type=int, default=1)
	r.add_argument("--stride", type=int, default=1)
	r.add_argument("--workers", type=int, default=1)
	args = parser.parse_args(argv)
	_log_setup()
	if args.cmd == "fit-camera":
		fit_camera(args.dataset, args.menagerie, args.frames, args.out)
	elif args.cmd == "simulate":
		simulate(
			args.menagerie,
			args.camera,
			args.out,
			args.episodes,
			args.seed,
			np.array(args.rest),
			args.prior,
			args.workers,
		)
	elif args.cmd == "layout-prior":
		layout_prior(args.dataset, args.camera, args.out)
	else:
		render_views(args.menagerie, args.camera, args.sim, args.out, args.views, args.seed, args.stride, args.workers)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
