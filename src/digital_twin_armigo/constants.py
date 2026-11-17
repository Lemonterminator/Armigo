"""Shared constants of the SO-101 sorting cell: joints, calibration, camera crop, depth encoding, cell geometry.

Both simulators (``digital_twin_armigo.mujoco_cell`` and the Isaac Lab stages in ``scripts/isaac``) read these, so
they agree on units and on the layout of dataset sort-depth-v2. Everything is in the robot base frame (metres, table
top at z = 0) unless a name says otherwise.
"""

from __future__ import annotations

import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = REPO_ROOT / "assets"
SIM_ASSETS_DIR = ASSETS_DIR / "meshes"
SO101_MJCF_DIR = ASSETS_DIR / "so101_mjcf"
CONFIG_DIR = REPO_ROOT / "config" / "sim"

FPS = 30
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
STATE_NAMES = [f"{j}.pos" for j in JOINTS]
TASKS = {
	"green": "Pick up the brick and place it in the green bowl.",
	"pink": "Pick up the brick and place it in the pink bowl.",
}
PHASES = ("rest", "approach", "grasp", "lift", "show_move", "show_hold", "handoff", "place", "return")

# Follower calibration used for sort-depth-v2 (Studio project robot payload, ticks).
CALIBRATION = {
	"shoulder_pan": (1267, 3200),
	"shoulder_lift": (894, 3251),
	"elbow_flex": (849, 3045),
	"wrist_flex": (788, 3115),
	"wrist_roll": (2040, 3961),
	"gripper": (2041, 3508),
}
# Joint zero offsets from the camera-based joint calibration (robot.json), degrees: model = reading + offset.
OFFSETS_DEG = {"shoulder_lift": 1.5816, "elbow_flex": -4.6851, "wrist_flex": -1.3997}
TICKS_PER_DEG = 4096 / 360
GRIPPER_RAD = (-0.17453, 1.74533)  # Menagerie gripper range; assumed 0 = closed, 100 = open
# Rest pose in Studio's normalised units (body joints -100…100, gripper 0…100).
REST_U = (4.1, -98.7, 99.3, -98.2, 24.1, 33.7)
# Standard holding pose between inspection and placement: the same joint target in every episode (radians: pan,
# lift, elbow, wrist flex, wrist roll), so pick-and-inspect and place skills can be cut from one episode. Gripper
# frame at about (0.34, 0.00, 0.21) m, jaws tilted 45 deg down.
HANDOFF_Q = (0.0, 0.2, -0.9, 1.5, 0.0)

# Top camera (D415 colour view): rendered at TOP_FULL and cropped to TOP_CROP = (x, y, width, height), as the depth
# bridge does (``--crop 96 0 512 416``).
TOP_FULL = (640, 480)
TOP_CROP = (96, 0, 512, 416)
# Depth bridge ``layers`` encoding: R = height above the table 0 → FINE_MAX_M, G = 0 → COARSE_MAX_M, B = validity.
FINE_MAX_M = 0.04
COARSE_MAX_M = 0.30
# Depth bridge plane-v2: camera 0.508 m above the table, optical axis 25.8 deg from vertical.
PLANE_HEIGHT_M = 0.508
PLANE_TILT_DEG = math.degrees(math.acos(0.8996))
# Dataset image shapes (height, width, channels) per camera key.
CAMERA_SHAPES = {
	"front": [480, 640, 3],
	"top_rgb_v2": [416, 512, 3],
	"top_depth_v2": [416, 512, 3],
	"wrist": [480, 640, 3],
}

# Real cell: plates about 20 cm across, Duplo-size bricks (2x2: 31.8 mm square, 19.2 mm high; 2x4: 63.6 mm long).
BOWL_R = 0.10
BOWL_H = 0.022
BOWL_FLOOR_TOP_M = 0.003
DUPLO_2X2 = (0.0318, 0.0318, 0.0192)
DUPLO_2X3 = (0.0477, 0.0318, 0.0192)
DUPLO_2X4 = (0.0636, 0.0318, 0.0192)
STUD_PITCH = 0.016
BRICK_COLOURS = ((0.95, 0.4, 0.05), (0.95, 0.8, 0.1), (0.1, 0.45, 0.85), (0.85, 0.1, 0.1), (0.55, 0.8, 0.15))
# Printed markers seen in the cell (defect_sorter.markers): id, black-square side in metres.
CELL_MARKERS = {10: 0.040, 11: 0.040, 16: 0.060, 17: 0.060}
