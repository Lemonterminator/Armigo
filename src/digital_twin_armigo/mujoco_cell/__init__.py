"""MuJoCo sorting cell for synthetic data: SO-101, two bowls, a brick (optionally with a defect cross), random cameras.

Modules, lowest layer first:

``layout``       episode layout dataclass and its randomisation
``scene``        MJCF generation for a layout
``cell``         the compiled model: stepping, assisted grasp, dataset camera renders
``ik``           damped-least-squares inverse kinematics
``expert``       scripted pick / present / hand off / place
``views``        camera randomisation and plausibility checks
``calibration``  top camera fit and layout prior from the real recordings
``pipeline``     ``simulate`` (physics once) and ``render_views`` (many views per episode)
``cli``          the ``fit-camera`` / ``layout-prior`` / ``simulate`` / ``render`` command line
"""

from ..camera import CameraPose, camera_forward, look_from, project, table_point
from ..constants import (
	BOWL_H,
	BOWL_R,
	BRICK_COLOURS,
	CAMERA_SHAPES,
	CELL_MARKERS,
	DUPLO_2X2,
	DUPLO_2X3,
	DUPLO_2X4,
	FPS,
	JOINTS,
	PHASES,
	SIM_ASSETS_DIR,
	STATE_NAMES,
	TASKS,
	TOP_CROP,
	TOP_FULL,
)
from ..depth import decode_height, layered_height_image
from ..joints import normalized_to_rad, rad_to_normalized
from .calibration import fit_camera, layout_prior
from .cell import Cell
from .cli import main
from .expert import Grip, Plan, expert_handoff, expert_pick, expert_place, expert_present, presentation_search
from .ik import best_roll, ik, ik_held, ik_reach, jaw_direction, site_error
from .layout import Layout, release_point, sample_layout
from .pipeline import render_views, simulate
from .scene import SUBSTEPS, brick_mesh_name, scene_xml, write_marker_textures
from .views import (
	bowl_rim_points,
	perturb_main_camera,
	sample_front_camera,
	sample_side_camera,
	sample_top_camera,
	top_view_ok,
)

__all__ = [
	"BOWL_H",
	"BOWL_R",
	"BRICK_COLOURS",
	"CAMERA_SHAPES",
	"CELL_MARKERS",
	"DUPLO_2X2",
	"DUPLO_2X3",
	"DUPLO_2X4",
	"FPS",
	"JOINTS",
	"PHASES",
	"SIM_ASSETS_DIR",
	"STATE_NAMES",
	"SUBSTEPS",
	"TASKS",
	"TOP_CROP",
	"TOP_FULL",
	"CameraPose",
	"Cell",
	"Grip",
	"Layout",
	"Plan",
	"best_roll",
	"bowl_rim_points",
	"brick_mesh_name",
	"camera_forward",
	"decode_height",
	"expert_handoff",
	"expert_pick",
	"expert_place",
	"expert_present",
	"fit_camera",
	"ik",
	"ik_held",
	"ik_reach",
	"jaw_direction",
	"layered_height_image",
	"layout_prior",
	"look_from",
	"main",
	"normalized_to_rad",
	"perturb_main_camera",
	"presentation_search",
	"project",
	"rad_to_normalized",
	"release_point",
	"render_views",
	"sample_front_camera",
	"sample_layout",
	"sample_side_camera",
	"sample_top_camera",
	"scene_xml",
	"simulate",
	"site_error",
	"table_point",
	"top_view_ok",
	"write_marker_textures",
]
