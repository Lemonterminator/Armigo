"""Trusted MuJoCo skills and skill-boundary checkpoints for the six-face graph."""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import os
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields, replace
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import ClassVar

import numpy as np

from ..benchmarks import SENSITIVITY_MATRIX
from ..camera import CameraPose, look_from
from ..constants import BOWL_R, FPS, PHASES, REST_U, SO101_MJCF_DIR
from ..experiments.contract import EVIDENCE, ORACLE_PARTS
from ..inspection import assert_no_defect_leakage
from ..joints import normalized_to_rad
from ..metrics import evaluate_episode_health, motion_smoothness
from ..mujoco_cell import Cell
from ..mujoco_cell._mj import mujoco
from ..mujoco_cell.episode import PhysicsEpisode, observe
from ..mujoco_cell.gripper_state import GripperMonitor, GripperMonitorParams
from ..mujoco_cell.handeye_rig import calibrate_layout
from ..mujoco_cell.inspection_planning import path_min_z
from ..mujoco_cell.inspection_views import DETECTOR_PAD, capture_view, crop_counts, detector_crop
from ..mujoco_cell.layout import sample_layout
from ..mujoco_cell.orbit_scan import OrbitParams, plan_orbit_scan
from ..mujoco_cell.physics_expert import MIN_JERK_PEAK_RATIO, ExpertParams, _min_jerk
from ..mujoco_cell.scene_v3 import SPLITS, SceneV3, apply_scene_v3, draw_camera_v3, sample_scene_v3
from ..mujoco_cell.table_relocation import RelocateExpert, RelocateParams, plan_bottom_view, plan_table_scan
from ..mujoco_cell.visibility import workspace_visible
from ..physics import NonIdealityCfg, load_physics_spec
from .bowl_clearance import PerceivedBowlClearance, with_perceived_bowl_clearance
from .grasp_loop import VARIANTS, GraspLoop
from .perception import (
	DRIFT_CORRECTION_MM,
	DRIFT_TABLE_OFFSET_MM,
	RESTING_BRICK_CLEARANCE_M,
	BowlEstimate,
	BrickEstimate,
	PerceptionCfg,
	bands_for,
	bowl_fill,
	bowl_filled,
	locate_bowls,
	locate_brick,
	top_camera_drift,
	yaw_error_deg,
)
from .review import MAX_CANDIDATES, MIN_POSE_SEPARATION_RAD, _alarm, llm_choice, review_decision, rule_choice
from .sensing import GripperContact
from .twin_graph import RECALIBRATE, RECALIBRATION_FAILED, Policy, decision, select_skill
from .wrist_localize import FuseCfg, _blob, believed_wrist_from_cam, fuse, localize_from_wrist, wrist_k

ROOT = Path(__file__).resolve().parents[3]
INITIAL_Q = np.array([*normalized_to_rad(np.array(REST_U, dtype=np.float64))[:5], ExpertParams().open_g])
STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION

PROFILE = "mujoco-close-top-1.35"
BOTTOM_MIN_PX = 400
REVIEW_FACE_PX_CAP = 6000.0  # re-view ranking: pixels of one face beyond this add nothing (agent/review.py)
WRIST_MIN_PX = 800
MIN_BRICK_PX = 150
MAX_TOTAL_FRAMES = 6300
RECALIBRATION_S = 60
# ``truth``: skills read the simulator (brick, bowls, contact forces), optionally with the brick perceived (see
# .perception). ``realistic``: skills see only what a real cell senses, see TwinMujoco.
SENSING = ("truth", "realistic")
STRATEGIES = ("A", "B", "C")
BOUNDARIES = ("belief_cell", "legacy")
MAX_REVIEW_VIEWS = 2  # extra re-views per episode (strategies B and C)
NEUTRAL_XY = (0.25, 0.0)  # constructor target of a realistic expert; every target it uses is set later
RETRY_MOVED_M = 0.005  # a brick found this far from where it was believed has moved: its face numbering is lost
RETRY_MOVED_DEG = 3.0
BOTTOM_BORDER_PX = 40  # realistic bottom views keep the believed brick this far inside the top image


# Case options (the optional 4th element of a case, see sample_case / load_suite).
CASE_OPTIONS = ("prior", "cameras", "scene", "split", "oracle", "calib_offset_mm")
CAMERA_MODES = ("fixed", "jitter", "jitter_extrinsic", "drift", "wide")
# Plan 2026-11-24 camera conditions: ``jitter_extrinsic`` draws exactly the ``jitter`` sample but keeps every focal
# length nominal (intrinsics known), so only the mounts move; ``drift`` is ``jitter_extrinsic`` after a calibration
# made on the nominal rig: the hand-eye estimate is solved on ``FIXED_TOP`` (``Layout.calib_top``) and kept while the
# real top camera sits at its perturbed pose.
EXTRINSIC_ONLY = ("jitter_extrinsic", "drift")
# ``cameras="jitter"`` (plan decision D3): per-layout perturbation of the fixed top and front cameras, drawn from a
# separate random stream of the seed so every other layout quantity stays identical to ``cameras="fixed"``. Top
# sigmas match sample_layout's own top-camera jitter; the wrist camera rides on the arm and is never perturbed.
TOP_JITTER = {"xy_m": 0.01, "height_m": 0.01, "yaw_deg": 2.0, "tilt_deg": 1.5, "roll_deg": 1.0, "focal_rel": 0.02}
FRONT_JITTER = {"distance_m": 0.05, "height_m": 0.05, "yaw_deg": 10.0, "focal_rel": 0.05}  # uniform half-widths
CAMERA_STREAM = 0xCA3  # second word of the camera stream's seed sequence
# Wrist camera mount error under ``cameras="jitter"``: uniform half-widths per axis in the camera's own frame, drawn
# from its own stream so the top and front draws stay as they were. The agent keeps planning with the designed mount.
WRIST_JITTER = {"pos_m": 0.003, "rot_deg": 2.0}
WRIST_STREAM = 1  # third word of the wrist stream's seed sequence
FIXED_TOP = CameraPose(0.4895033468, 0.0, 0.5130680468, 180.0, 25.0712240655, 0.0, 612.47)
FIXED_FRONT = {"distance_m": 0.55, "height_m": 0.35, "yaw_deg": 35, "focal_px": 560}


@lru_cache(maxsize=8)
def _layout_prior(path: str) -> tuple[dict, ...]:
	"""Real-layout prior (``layout-prior`` output, e.g. config/sim/prior_grasp.json), as the data pipeline reads it."""
	file = Path(path) if Path(path).is_absolute() else ROOT / path
	rows = json.loads(file.read_text(encoding="utf-8"))
	prior = tuple(p for p in rows if "green" in p and "pink" in p)
	if not prior or not any(p.get("brick") for p in prior):
		raise ValueError(f"layout prior {path} has no bowl and brick rows")
	return prior


def _case_options(options) -> dict:
	if options is None:
		return {"prior": None, "cameras": "fixed"}
	if not isinstance(options, dict) or set(options) - set(CASE_OPTIONS):
		raise ValueError(f"case options must be a dict with keys from {CASE_OPTIONS}")
	prior = options.get("prior")
	if prior is not None and (type(prior) is not str or not prior):
		raise ValueError("case option prior must be a layout prior JSON path")
	scene = options.get("scene")
	cameras = options.get("cameras", "wide" if scene == "v3" else "fixed")
	if cameras not in CAMERA_MODES:
		raise ValueError(f"case option cameras must be one of {CAMERA_MODES}")
	oracle_raw = options.get("oracle")
	oracle: tuple[str, ...] = ()
	if oracle_raw is not None:
		if isinstance(oracle_raw, str):
			oracle = tuple(p.strip() for p in oracle_raw.split(",") if p.strip())
		elif isinstance(oracle_raw, (list, tuple, set, frozenset)):
			oracle = tuple(str(p) for p in oracle_raw)
		else:
			raise ValueError("case option oracle must be an iterable of ORACLE_PARTS")
		bad_oracle = [p for p in oracle if p not in ORACLE_PARTS]
		if bad_oracle:
			raise ValueError(f"case option oracle parts must be in {ORACLE_PARTS}, got {bad_oracle!r}")
	extra_oracle = {"oracle": oracle} if oracle else {}
	calib_mm = options.get("calib_offset_mm", 0)
	if calib_mm is None:
		calib_mm = 0
	if isinstance(calib_mm, bool) or not isinstance(calib_mm, (int, float)) or float(calib_mm) < 0:
		raise ValueError(f"case option calib_offset_mm must be >= 0, got {calib_mm!r}")
	calib_mm_f = float(calib_mm)
	if calib_mm_f > 0 and cameras == "drift":
		raise ValueError("case option calib_offset_mm cannot be combined with cameras='drift'")
	extra_calib = {"calib_offset_mm": calib_mm_f} if calib_mm_f > 0 else {}
	if scene is None:
		if "split" in options:
			raise ValueError("case option split requires scene='v3'")
		return {"prior": prior, "cameras": cameras, **extra_oracle, **extra_calib}
	if scene != "v3":
		raise ValueError(f"case option scene must be 'v3', got {scene!r}")
	split = options.get("split", "train")
	if split not in SPLITS:
		raise ValueError(f"case option split must be one of {SPLITS}")
	return {"prior": prior, "cameras": cameras, "scene": "v3", "split": split, **extra_oracle, **extra_calib}


def _wide_top_pose(cam: dict[str, float]) -> CameraPose:
	return CameraPose(
		x=FIXED_TOP.x + cam["dx_m"],
		y=FIXED_TOP.y + cam["dy_m"],
		height=FIXED_TOP.height + cam["dz_m"],
		yaw_deg=FIXED_TOP.yaw_deg + cam["dyaw_deg"],
		tilt_deg=FIXED_TOP.tilt_deg + cam["dtilt_deg"],
		roll_deg=FIXED_TOP.roll_deg + cam["droll_deg"],
		focal_px=FIXED_TOP.focal_px * (1.0 + cam["dfocal_rel"]),
	)


def case_cameras(
	seed: int, cameras: str = "fixed", *, scene_v3: SceneV3 | None = None
) -> tuple[CameraPose, CameraPose]:
	"""Top and front camera of a case: fixed, perturbed (``jitter``/``drift``), or wide v3 sampling."""
	f = FIXED_FRONT
	if cameras == "fixed":
		return FIXED_TOP, look_from((0.20, 0.0), f["distance_m"], f["height_m"], f["yaw_deg"], focal_px=f["focal_px"])
	if cameras == "wide":
		cam = (
			draw_camera_v3(np.random.default_rng([seed, CAMERA_STREAM, 3]), "train")
			if scene_v3 is None
			else scene_v3.camera
		)
		top = _wide_top_pose(cam)
		rng_front = np.random.default_rng([seed, CAMERA_STREAM, 2])
		j = FRONT_JITTER
		front = look_from(
			(0.20, 0.0),
			f["distance_m"] + rng_front.uniform(-j["distance_m"], j["distance_m"]),
			f["height_m"] + rng_front.uniform(-j["height_m"], j["height_m"]),
			f["yaw_deg"] + rng_front.uniform(-j["yaw_deg"], j["yaw_deg"]),
			focal_px=f["focal_px"] * (1 + rng_front.uniform(-j["focal_rel"], j["focal_rel"])),
		)
		return top, front
	rng = np.random.default_rng([seed, CAMERA_STREAM])
	keep_focal = cameras in EXTRINSIC_ONLY  # the focal draws still happen so the extrinsic draws stay identical
	t, j = FIXED_TOP, TOP_JITTER
	top = CameraPose(
		t.x + rng.normal(0, j["xy_m"]),
		t.y + rng.normal(0, j["xy_m"]),
		t.height + rng.normal(0, j["height_m"]),
		t.yaw_deg + rng.normal(0, j["yaw_deg"]),
		t.tilt_deg + rng.normal(0, j["tilt_deg"]),
		t.roll_deg + rng.normal(0, j["roll_deg"]),
		t.focal_px * (1 + (0.0 * rng.uniform() if keep_focal else rng.uniform(-j["focal_rel"], j["focal_rel"]))),
	)
	j = FRONT_JITTER
	front = look_from(
		(0.20, 0.0),
		f["distance_m"] + rng.uniform(-j["distance_m"], j["distance_m"]),
		f["height_m"] + rng.uniform(-j["height_m"], j["height_m"]),
		f["yaw_deg"] + rng.uniform(-j["yaw_deg"], j["yaw_deg"]),
		focal_px=f["focal_px"]
		* (1 + (0.0 * rng.uniform() if keep_focal else rng.uniform(-j["focal_rel"], j["focal_rel"]))),
	)
	return top, front


def case_wrist_offset(
	seed: int, cameras: str = "fixed", *, scene_v3: SceneV3 | None = None
) -> tuple[float, ...] | None:
	"""Wrist camera mount error of a case (``Layout.wrist_cam_offset``): None for the fixed rig."""
	if cameras == "fixed":
		return None
	if cameras == "wide":
		return tuple(float(v) for v in scene_v3.wrist) if scene_v3 is not None else None
	rng = np.random.default_rng([seed, CAMERA_STREAM, WRIST_STREAM])
	p, r = WRIST_JITTER["pos_m"], WRIST_JITTER["rot_deg"]
	return tuple(float(v) for v in (*rng.uniform(-p, p, 3), *rng.uniform(-r, r, 3)))


def sample_case(seed: int, positive: bool, face: int, options: dict | None = None):
	"""The layout of case ``[seed, positive, face]`` or ``[seed, positive, face, options]``.

	``options`` (all optional; omitted or None reproduces the three-element case exactly):

	- ``prior``: path (absolute or repository-relative) of a real-layout prior; bowls and brick start from measured
	  real positions (``sample_layout(prior=...)``) instead of uniform positions in reach. Changes the whole layout.
	- ``cameras``: ``"fixed"`` (default, the calibrated close-top rig) or ``"jitter"`` (D3, see ``case_cameras``).
	  Only the cameras change. In ``realistic`` sensing TwinMujoco hand-eye calibrates the top camera of every
	  episode from its own layout (``calibrate_layout`` in ``__init__``), so the skills see the jittered pose through
	  that calibration; the intrinsics are taken as known. The wrist camera mount is perturbed too (``case_wrist_offset``); the
	  agent is not told, and plans its wrist views with the designed mount (``_belief``). The bottom threshold profile (``PROFILE``) was tuned on
	  the fixed cameras.
	"""
	if type(seed) is not int or not 0 <= seed <= 2**32 - 1 or type(positive) is not bool:
		raise ValueError("invalid fixture seed or class")
	if type(face) is not int or not 0 <= face < 6:
		raise ValueError("face must be in [0, 5]")
	opts = _case_options(options)
	raw = json.loads((ROOT / "config/sim/top_camera_joint.json").read_text(encoding="utf-8"))
	prior = list(_layout_prior(opts["prior"])) if opts["prior"] else None
	layout = sample_layout(np.random.default_rng(seed), CameraPose(**raw["camera"]), defective=positive, prior=prior)
	layout.defect_kind = "cross_var"
	layout.defect_face = face
	layout.defect_seed = 19_199 + seed * 100_019
	layout.clutter = []
	layout.distractors = []
	if opts.get("scene") == "v3":
		layout.top, layout.front = case_cameras(seed, "fixed")
		scene = sample_scene_v3(seed, opts["split"])
		apply_scene_v3(layout, scene)
		layout.top, layout.front = case_cameras(seed, opts["cameras"], scene_v3=scene)
		layout.wrist_cam_offset = case_wrist_offset(seed, opts["cameras"], scene_v3=scene)
		layout.calib_top = FIXED_TOP if opts["cameras"] == "drift" else None
		if opts["cameras"] == "wide":
			resamples = 0
			ok, _ = workspace_visible(layout)
			while not ok:
				resamples += 1
				if resamples > 20:
					raise RuntimeError(
						f"v3 wide camera failed workspace_visible after 20 resamples (seed={seed}, split={opts['split']})"
					)
				rng_cam = np.random.default_rng([int(seed), 0x5C3, SPLITS.index(opts["split"]), 0xCA4, resamples])
				cam = draw_camera_v3(rng_cam, opts["split"])
				layout.top = _wide_top_pose(cam)
				layout._scene_v3 = replace(scene, camera=cam)
				ok, _ = workspace_visible(layout)
			layout._visibility_resamples = resamples
	else:
		layout.top, layout.front = case_cameras(seed, opts["cameras"])
		layout.wrist_cam_offset = case_wrist_offset(seed, opts["cameras"])
		# The pose the hand-eye calibration is solved on: the real top camera, except under ``drift``.
		layout.calib_top = FIXED_TOP if opts["cameras"] == "drift" else None
	offset_mm = float(opts.get("calib_offset_mm") or 0.0)
	if offset_mm > 0.0:
		rng_off = np.random.default_rng([seed, CAMERA_STREAM, 0xC41])
		theta = float(rng_off.uniform(0.0, 2.0 * math.pi))
		sign = -1.0 if int(rng_off.integers(0, 2)) == 0 else 1.0
		d_m = offset_mm / 1000.0
		base_cam = layout.top
		layout.calib_top = replace(
			base_cam,
			x=base_cam.x + d_m * math.cos(theta),
			y=base_cam.y + d_m * math.sin(theta),
			yaw_deg=base_cam.yaw_deg + sign * offset_mm * 0.2,
		)
	if opts.get("oracle"):
		layout.oracle = opts["oracle"]
	return layout


def load_suite(path: Path) -> list[tuple]:
	"""Cases of a suite file: a JSON array of ``[seed, positive, face]`` (optionally with a 4th options dict), or an
	object ``{"cases": [...], "options": {...}, ...metadata}`` whose suite-level options (see sample_case) are
	appended to every case as its 4th element. Default options are dropped, so such cases stay three elements."""
	raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
	options = None
	if isinstance(raw, dict):
		if not isinstance(raw.get("cases"), list):
			raise TypeError("suite object needs a 'cases' array")
		opts = _case_options(raw.get("options") or None)
		defaults = _case_options(None)
		options = {k: v for k, v in opts.items() if k not in defaults or v != defaults[k]} or None
		raw = raw["cases"]
	if not isinstance(raw, list):
		raise TypeError("suite must be a JSON array of cases or an object with 'cases'")
	cases = []
	for row in raw:
		if not isinstance(row, list) or len(row) not in (3, 4):
			raise ValueError(f"invalid case {row!r}")
		if len(row) == 4:
			if options is not None:
				raise ValueError("a case has its own options and the suite sets options too")
			_case_options(row[3])
		cases.append(tuple(row) if options is None else (*row, options))
	return cases


def _pose(a, b) -> float:
	return float(np.linalg.norm(np.asarray(a, dtype=np.float64)[:6] - np.asarray(b, dtype=np.float64)[:6]))


# Hover-scan settings of _scan_plan's fallback, reused for re-view candidates.
TABLE_SCAN_MODES = (
	{"heights": (0.06, 0.075, 0.09), "offset_m": 0.03},
	{"heights": (0.07, 0.085, 0.10), "offset_m": 0.02},
	{"heights": (0.055, 0.08, 0.095), "offset_m": 0.04},
)


def true_world_from_cam(top) -> np.ndarray:
	"""World-from-camera (OpenCV axes: x right, y down, z forward) of the top camera as mounted: the oracle that the
	hand-eye estimate approximates. Same axes as the ``actual_cam_*`` diagnostics in ``camera_calibrated``."""
	right, up = top.axes()
	forward = -np.cross(right, up)
	m = np.eye(4)
	m[:3, 0] = right / np.linalg.norm(right)
	m[:3, 1] = -up / np.linalg.norm(up)
	m[:3, 2] = forward / np.linalg.norm(forward)
	m[:3, 3] = (top.x, top.y, top.height)
	return m


def set_cv_camera(cell, name: str, world_from_cam: np.ndarray) -> None:
	"""Place ``cell``'s camera ``name`` at an OpenCV world-from-camera pose (x right, y down, z forward); MuJoCo
	cameras look along -z with y up. Intrinsics are left as they are."""
	cid = cell.camera_id(name)
	mat = np.asarray(world_from_cam, dtype=np.float64)[:3, :3] @ np.diag([1.0, -1.0, -1.0])
	quat = np.zeros(4)
	mujoco.mju_mat2Quat(quat, mat.flatten())
	cell.model.cam_pos[cid] = np.asarray(world_from_cam, dtype=np.float64)[:3, 3]
	cell.model.cam_quat[cid] = quat
	mujoco.mj_forward(cell.model, cell.data)


@contextmanager
def blind_geometry(cell):
	# Group 4 is disabled in both visibility render options. The model is session-local.
	original = cell.model.geom_group[cell.defect_geoms].copy()
	cell.model.geom_group[cell.defect_geoms] = 4
	try:
		yield
	finally:
		cell.model.geom_group[cell.defect_geoms] = original


@dataclass
class Checkpoint:
	id: str
	branch: int
	frame: int
	integration: np.ndarray
	expert: dict
	filter: dict
	wrist_count: int
	bottom_count: int
	views: list[dict]
	# Added after the first persisted checkpoints; the defaults load older checkpoint files.
	retries: int = 0  # in-place recoveries before this checkpoint (they share the recovery budget with rewinds)
	held: dict | None = None  # carried brick: lift pose and grasp roll (the grasp itself is in ``integration``)
	# Agent-side state (see TwinMujoco._save): brick belief, perceived bowls and camera, carried-brick transform,
	# contact filter, depth-noise RNG, park spot, expert parameters and variants, recovery counters, segment rows.
	agent: dict | None = None


# Persisted next to ``<id>.json`` / ``<id>.npy``: the per-frame history a resumed episode continues (telemetry rows,
# policy targets and phases), and the empty-bowl depth reference, which is fixed after ``prepare`` and stored once.
HISTORY_SUFFIX = ".history.npz"
HISTORY_KEYS = ("jaw_env_force", "phases", "targets")


def brick_to_json(est: BrickEstimate | None) -> dict | None:
	if est is None:
		return None
	return {
		"pos": np.asarray(est.pos, dtype=np.float64).tolist(),
		"yaw": float(est.yaw),
		"size": [float(v) for v in est.size],
		"footprint_m": [float(v) for v in est.footprint_m],
		"top_points": int(est.top_points),
		"extended_mm": [float(v) for v in est.extended_mm],
		"yaw_ambiguity_deg": float(est.yaw_ambiguity_deg),
		"footprint_occluded_frac": float(est.footprint_occluded_frac),
	}


def brick_from_json(raw: dict | None) -> BrickEstimate | None:
	if raw is None:
		return None
	return BrickEstimate(
		np.array(raw["pos"], dtype=np.float64),
		raw["yaw"],
		tuple(raw["size"]),
		tuple(raw["footprint_m"]),
		raw["top_points"],
		tuple(raw["extended_mm"]),
		float(raw.get("yaw_ambiguity_deg", 0.0)),
		float(raw.get("footprint_occluded_frac", 0.0)),
	)


def _tuples(value):
	return tuple(_tuples(v) for v in value) if isinstance(value, list) else value


def params_from_json(cls, raw: dict):
	"""An ``ExpertParams`` / ``RelocateParams`` from its ``asdict`` JSON (lists back to the tuples they were)."""
	names = {f.name for f in fields(cls)}
	return cls(**{k: _tuples(v) for k, v in raw.items() if k in names})


# Bounded recovery variants. Each is a fixed, named, deterministic transform of the default expert parameters. The
# agent (or LLM) only ever picks a name; no number crosses that boundary. Every value must lie in RECOVERY_BOUNDS and
# every variant only slows motion down or raises the transfer height, so a variant cannot be more aggressive than the
# defaults. Keys of "expert" / "relocate" are ExpertParams / RelocateParams fields; "scan_peak_speed" is the peak
# joint speed (rad/s) of the wrist-scan moves (default SCAN_PEAK_SPEED, the value the scan used before variants).
SCAN_PEAK_SPEED = 1.0
HINT_GATE_M = 0.015  # a pinned brick (localise="vote") is the blob fitted this close to its pinned top-view x, y
RECOVERY_BOUNDS = MappingProxyType(
	{
		"scan_peak_speed": (0.25, SCAN_PEAK_SPEED),
		"scan_move_s": (0.6, 1.5),
		"ready_move_s": (0.6, 1.2),
		"carry_peak_speed_rad_s": (0.2, 0.5),
		"show_move_s": (0.9, 2.0),
		"transfer_s": (1.8, 3.0),
		"transfer_z_m": (0.12, 0.15),
		"place_descend_s": (1.2, 2.0),
		"retreat_s": (0.8, 1.5),
		"return_s": (1.2, 2.0),
	}
)
_SLOW_SCAN = {"scan_peak_speed": 0.35, "scan_move_s": 1.0, "ready_move_s": 1.0}
_SLOW_CARRY = {"carry_peak_speed_rad_s": 0.3, "show_move_s": 1.4, "transfer_s": 2.6}
_GENTLE_PLACE = {"place_descend_s": 1.8, "retreat_s": 1.2, "return_s": 1.8}
RECOVERY_VARIANTS = MappingProxyType(
	{
		"slow_scan": MappingProxyType(_SLOW_SCAN),
		"slow_carry": MappingProxyType(_SLOW_CARRY),
		"gentle_place": MappingProxyType(_GENTLE_PLACE),
		"high_place": MappingProxyType({**_GENTLE_PLACE, "transfer_z_m": 0.14}),
		"cautious": MappingProxyType({**_SLOW_SCAN, **_SLOW_CARRY, **_GENTLE_PLACE}),
	}
)
VARIANT_NOTES = MappingProxyType(
	{
		"slow_scan": "slower wrist-scan and ready moves, so joints stop saturating",
		"slow_carry": "slower carry and show moves",
		"gentle_place": "slower place descent, retreat and return",
		"high_place": "slower place moves with a higher approach height over the bowl",
		"cautious": "slow scan, carry and place together",
	}
)
# Failure evidence -> recommended variant. A torque stall is attributed to the skill that was running when the
# saturated run started; other physical or placement failures map by code.
STALL_VARIANT_BY_NODE = MappingProxyType(
	{"scan_wrist": "slow_scan", "present_bottom": "slow_carry", "place": "gentle_place"}
)
VARIANT_BY_CODE = MappingProxyType(
	{
		"BRICK_LOST": "slow_carry",
		"PLACEMENT_UNVERIFIED": "high_place",
		"PLACE_RETURN_PATH_BLOCKED": "high_place",
		"PARK_PATH_BLOCKED": "slow_carry",
	}
)
# Orbit-scan rings per wrist-scan attempt (first scan, then repeats and scans after a rewind): later attempts keep
# the jaws higher and the ring wider, so a repeat does not retrace a ring that already failed.
ORBIT_ATTEMPTS = (
	OrbitParams(),
	OrbitParams(tip_heights_m=(0.025, 0.04, 0.055)),
	OrbitParams(tip_heights_m=(0.04, 0.06), standoffs_m=(0.04, 0.06)),
)
RELOCATE_KEYS = frozenset({"scan_move_s"})  # RelocateParams fields a variant may set
STALL_REASONS = ("torque_stall", "tracking_lag")
STALL_SECONDS = 1.5  # metrics.evaluate_episode_health max_stall_secs
STALL_FRACTION = 0.95  # fraction of the torque limit that counts as saturated


def apply_variant(name, params, relocate):
	"""Return ``(ExpertParams, RelocateParams, scan_peak_speed)`` for variant ``name`` applied to the given defaults.
	Raises ValueError for an unknown name or a value outside RECOVERY_BOUNDS."""
	if name not in RECOVERY_VARIANTS:
		raise ValueError(f"unknown recovery variant: {name}")
	values = RECOVERY_VARIANTS[name]
	for key, value in values.items():
		lo, hi = RECOVERY_BOUNDS[key]
		if not lo <= value <= hi:
			raise ValueError(f"variant {name}: {key}={value} outside [{lo}, {hi}]")
	# transfer_z_m exists in both parameter classes: only the ExpertParams one (the bowl approach) is a variant key.
	rel = {k: v for k, v in values.items() if k in RELOCATE_KEYS}
	expert = {k: v for k, v in values.items() if k not in RELOCATE_KEYS and k != "scan_peak_speed"}
	return (
		replace(params, **expert),
		replace(relocate, **rel),
		float(values.get("scan_peak_speed", SCAN_PEAK_SPEED)),
	)


def sweep(expert, path, peak_speed, min_s, phase="approach", pause_s=None, on_progress=None):
	"""Targets that move the arm from ``expert.last`` through the joint vectors of ``path`` without stopping at them:
	min-jerk timing along the piecewise-linear joint path (arc length measured as the largest joint change, so no
	joint exceeds ``peak_speed``), split into chunks of at most ``max_move_chunk_s`` with the expert's chunk pause
	between them, as :meth:`ClosedLoopExpert._move` does for one segment. Lasts at least ``min_s``."""
	p = expert.p
	points = [expert.last.copy(), *(np.clip(np.asarray(q, dtype=np.float64), expert._lo, expert._hi) for q in path)]
	if len(points) < 2:
		return
	# Arc length: the largest arm-joint change, plus a tiny share of the rest so a path that only moves the
	# gripper still advances to its end.
	steps = [float(np.max(np.abs(b[:5] - a[:5])) + 1e-6 * np.sum(np.abs(b - a))) for a, b in itertools.pairwise(points)]
	arc = np.concatenate([[0.0], np.cumsum(steps)])
	total = float(arc[-1])
	secs = max(min_s, MIN_JERK_PEAK_RATIO * total / peak_speed)
	chunks = max(1, math.ceil(secs / p.max_move_chunk_s))
	stack = np.array(points)

	def at(s):
		k = int(np.clip(np.searchsorted(arc, s, side="right") - 1, 0, len(steps) - 1))
		f = 0.0 if steps[k] <= 0 else (s - arc[k]) / steps[k]
		return stack[k] + min(max(f, 0.0), 1.0) * (stack[k + 1] - stack[k])

	for c in range(chunks):
		a, b = total * c / chunks, total * (c + 1) / chunks
		n = max(2, expert._frames(secs / chunks) + 1)
		for s in _min_jerk(n)[1:]:
			cur = a + s * (b - a)
			if on_progress is not None:
				on_progress(cur / total if total > 0 else 1.0)
			yield at(cur), phase
		if c < chunks - 1:
			for _ in range(expert._frames(p.chunk_pause_s if pause_s is None else pause_s)):
				if on_progress is not None:
					on_progress(b / total if total > 0 else 1.0)
				yield at(b), phase


def stall_start(torques, limit, fps=FPS):
	"""First frame of a continuous run of saturated arm-joint torque longer than STALL_SECONDS, else None."""
	saturated = (np.abs(np.asarray(torques)[:, :5]) >= STALL_FRACTION * limit).any(axis=1)
	run = 0
	for i, flag in enumerate(saturated):
		run = run + 1 if flag else 0
		if run / fps > STALL_SECONDS:
			return i - run + 1
	return None


def node_at(events, frame):
	"""Skill node running at active-branch ``frame``, replaying node and rewound events of the trace."""
	timeline = []
	for e in events:
		if e["kind"] == "rewound":
			timeline = [t for t in timeline if t[0] <= e["frame"]]
		elif e["kind"] == "node" and e["node"] != "recover":
			timeline.append((e["frame"], e["node"]))
	return next((node for f, node in reversed(timeline) if f <= frame), None)


def rewind_options(checkpoints, failure, stall_frame, stall_node):
	"""Ordered ``(choice, note)`` pairs for a rewind. The first entry is the deterministic default: the latest
	checkpoint taken before the stall began (a checkpoint after it keeps the stall in the health history, so it can
	never clear it) with the variant recommended for the failure, then the plain rewind of that checkpoint."""
	reasons = failure.get("reason_codes") or []
	recommended = VARIANT_BY_CODE.get(failure.get("code"))
	if stall_frame is not None and any(r in STALL_REASONS for r in reasons):
		recommended = STALL_VARIANT_BY_NODE.get(stall_node, "cautious")
	latest = list(reversed(checkpoints))
	if stall_frame is not None:
		clean = [c for c in latest if c.frame <= stall_frame]
		latest = clean + [c for c in latest if c not in clean]
	options = []
	for c in latest:
		hint = ""
		if stall_frame is not None and c.frame > stall_frame:
			hint = f"; keeps the stall that began at frame {stall_frame}"
		variants = [recommended] if recommended else []
		variants += ["cautious"] if recommended not in (None, "cautious") else []
		for v in variants:
			options.append((f"{c.id}+{v}", f"rewind to {c.id} (frame {c.frame}) with {VARIANT_NOTES[v]}{hint}"))
		options.append((c.id, f"rewind to {c.id} (frame {c.frame}) and retry with unchanged parameters{hint}"))
	return options


class ContinuousCarryExpert(RelocateExpert):
	"""Retime long carry moves along the same joint-space line."""

	smoothed_moves = 0

	def _move(self, q_to: np.ndarray, secs: float, phase: str):
		q_from = self.last.copy()
		q_to = np.clip(q_to, self._lo, self._hi)  # plan to the reachable target, as the base move does
		chunks = max(1, math.ceil(secs / self.p.max_move_chunk_s))
		if chunks == 1 or phase not in ("lift", "show_move", "place") or q_from[5] != q_to[5]:
			yield from super()._move(q_to, secs, phase)
			return

		frames = chunks * self._frames(secs / chunks) + (chunks - 1) * self._frames(self.p.chunk_pause_s)
		t = np.arange(1, frames + 1, dtype=np.float64) / frames
		ramp = 0.2
		edge = np.minimum(t, 1 - t)
		x = np.minimum(edge / ramp, 1.0)
		# Integrate a quintic velocity ramp into a constant-speed middle, then mirror at the end.
		progress = (ramp * x**4 * (2.5 - 3 * x + x * x) + np.maximum(edge - ramp, 0.0)) / (1 - ramp)
		progress[t > 0.5] = 1 - progress[t > 0.5]
		self.smoothed_moves += 1
		for s in progress[:-1]:
			yield q_from + s * (q_to - q_from), phase
		yield q_to.copy(), phase


def with_believed_brick_size(expert_type: type, size) -> type:
	"""Subclass of the expert class ``expert_type`` whose brick extents (grasp band and axes, put-down and release
	heights) come from ``size()``, the believed size, instead of the simulator's brick."""

	class BelievedSizeExpert(expert_type):
		@property
		def size(self):
			return size()

	BelievedSizeExpert.__name__ = BelievedSizeExpert.__qualname__ = f"BelievedSize{expert_type.__name__}"
	return BelievedSizeExpert


class TwinMujoco:
	"""The six-face skills on one simulated cell.

	With ``sensing="realistic"`` the skills use only what a real cell senses, the camera images, the joint and
	gripper readings and the robot model:

	- the top camera is hand-eye calibrated before the episode, and the brick and both bowls are located in its images;
	- a carried brick is believed to move rigidly with the gripper from the grasp on;
	- planners and the detector crop see the believed brick (the simulated one is moved there while they run);
	- the brick's size class (2x2 or 2x4) is read from the perceived top-face footprint, and every planner that needs
	  the brick's extents (grasp axes and band, orbit ring, held-brick table clearance, put-down and release heights)
	  uses it instead of the simulator's brick size;
	- the expert's contact signals come from the gripper position and effort (.sensing), and brick speed reads zero;
	- a placement is checked by the top camera (depth against an empty-bowl image);
	- a failure is recovered from the current state (park or drop, relocate the brick) instead of a physics rewind;
	- the motion-health check moves to the final evaluation, which, as in ``truth``, is ground truth;
	- the grasp and path clearance checks measure the robot against the nominal bowl shape at the perceived bowl
	  centres, dilated by a 5 mm margin (.bowl_clearance), not against the simulator's bowl geoms. Clutter is not
	  perceived, so realistic refuses layouts with clutter.

	``truth`` keeps the simulator's bowl and clutter collision geometry in the clearance checks.
	"""

	scan_peak_speed = SCAN_PEAK_SPEED
	sweep_pause_s = None  # pause between chunks of a multi-pose scan sweep; None: the expert's chunk pause
	recovery_variants: tuple = ()

	def __init__(
		self,
		layout,
		scorer,
		record,
		*,
		llm=None,
		bottom_threshold=None,
		crop_size=256,
		fault=None,
		perturb=None,
		on_frame=None,
		smooth_carry=False,
		sensing="truth",
		strategy="A",
		max_scored_views=None,
		max_review_views=MAX_REVIEW_VIEWS,
		boundary="belief_cell",
		cell_factory=None,
		review_llm=None,
		oracle=(),
		motion="open",
		belief_cameras="actual",
		evidence="hold",
		detector_latency_s=0.15,
		segmenter="lab",
		segmenter_onnx=None,
		cell_markers=False,
		ring_localise=False,
		localise="top",
		perception_llm=None,
		drift_alarm: bool = False,
	):
		if sensing not in SENSING:
			raise ValueError(f"sensing must be one of {SENSING}")
		self.sensing = sensing
		self.realistic = sensing == "realistic"
		if motion not in VARIANTS:
			raise ValueError(f"motion must be one of {tuple(VARIANTS)}")
		if belief_cameras not in ("actual", "estimated"):
			raise ValueError(f"belief_cameras must be 'actual' or 'estimated', got {belief_cameras!r}")
		if evidence not in EVIDENCE:
			raise ValueError(f"evidence must be one of {EVIDENCE}")
		if segmenter not in ("lab", "unet", "silhouette"):
			raise ValueError(f"segmenter must be 'lab', 'unet' or 'silhouette', got {segmenter!r}")
		self.motion = motion
		self.evidence = evidence
		self.belief_cameras = "estimated" if evidence != "hold" else belief_cameras
		self.detector_latency_s = float(detector_latency_s)
		self.segmenter = segmenter
		self.segmenter_onnx = None if segmenter_onnx is None else Path(segmenter_onnx)
		if ring_localise and sensing != "realistic":
			raise ValueError("ring_localise is for realistic sensing; truth sees the brick directly")
		# Before each grasp, re-localise the brick from a ring of wrist views (see _ring_localise); off by default.
		self.ring_localise = bool(ring_localise)
		if localise not in ("top", "vote"):
			raise ValueError(f"localise must be 'top' or 'vote', got {localise!r}")
		if localise == "vote" and sensing != "realistic":
			raise ValueError("localise='vote' is for realistic sensing")
		# "vote": the graph's localize node picks the brick (shape rule + VLM, wrist confirmation on disagreement),
		# refines it with the wrist ring and pins it (see localise_brick). "top": the top view alone (default).
		self.localise = localise
		if drift_alarm and localise != "vote":
			raise ValueError("drift_alarm requires localise='vote'")
		self.drift_alarm = bool(drift_alarm)
		self._recalibration_attempted = False
		self.perception_llm = perception_llm
		self._rest_view = None  # (top depth taken at rest in prepare, frame it stays valid at)
		self._brick_hint = None
		self._brick_net = None
		self.fuse_cfg = FuseCfg()
		# The printed cell markers on the table (ArUco 10, 11, 16, 17) have no use once the top camera is calibrated
		# (the hand-eye rig reads only the card markers 20-23); in the episode's top and wrist images they are only
		# confusers. The calibration rig is built from the layout as given, markers included (a real cell is
		# calibrated before they are taken away); the episode cell, its saved layout and the belief cells have none.
		# ``cell_markers=True`` keeps them (reproducing episodes recorded before 2026-11-25).
		self.cell_markers = bool(cell_markers)
		self._calib_layout = layout
		self.cell_markers_removed = 0 if self.cell_markers else len(getattr(layout, "markers", None) or [])
		if self.cell_markers_removed:
			layout = copy.copy(layout)
			layout.markers = []
		if isinstance(getattr(record, "scene", None), dict):
			record.scene["cell_markers"] = self.cell_markers
			if self.drift_alarm:
				record.scene["drift_alarm"] = True
		raw_oracle = oracle if oracle else getattr(layout, "oracle", ())
		if isinstance(raw_oracle, str):
			raw_oracle = [p.strip() for p in raw_oracle.split(",") if p.strip()]
		raw_oracle = tuple(raw_oracle or ())
		bad_oracle = [p for p in raw_oracle if p not in ORACLE_PARTS]
		if bad_oracle:
			raise ValueError(f"oracle parts must be in {ORACLE_PARTS}, got {bad_oracle!r}")
		self.oracle = frozenset(raw_oracle) if self.realistic else frozenset()
		if self.drift_alarm and self.oracle.intersection(("brick", "camera")):
			raise ValueError("drift_alarm requires measured brick and camera poses")
		self._oracle_logged = False
		if strategy not in STRATEGIES:
			raise ValueError(f"strategy must be one of {STRATEGIES}")
		if boundary not in BOUNDARIES:
			raise ValueError(f"boundary must be one of {BOUNDARIES}")
		# Re-view strategy (plan 2026-11-24 section 5.3): A decides on the first qualifying alarm; B and C hold an alarm
		# for review from another angle, B choosing the view by rule and C by asking the LLM with images.
		self.strategy = strategy
		self.max_scored_views = max_scored_views  # None: no episode-wide cap (phase-1 / historical behaviour)
		self.max_review_views = max_review_views
		# realistic only: "belief_cell" computes crops, visible faces and view plans on a separate MuJoCo cell built
		# from what the agent believes (perceived brick size class and pose, perceived bowls, hand-eye top camera,
		# designed wrist mount); "legacy" moves the simulator's own brick to the believed pose (development control).
		self.boundary = boundary
		self.review_llm = review_llm  # strategy C's view chooser; separate from ``llm`` (text skill selection)
		self._belief_cells = {}
		self.review_count = 0  # re-views taken (B / C)
		self.scored_count = 0  # scored views taken in the whole episode, rewound branches included
		self.review_exhausted: set[str] = set()  # alarms no reachable re-view was found for
		self._crops: dict[str, np.ndarray] = {}  # detector input of each view, for the LLM re-view chooser
		self._llm_sent: set[str] = set()  # image keys already shown to the LLM in this episode
		self._expert_xy = NEUTRAL_XY if self.realistic else layout.brick_xy
		self.layout = layout
		self.scorer = scorer
		self.record = record
		self.llm = llm
		self.bottom_threshold = bottom_threshold
		self.crop_size = crop_size
		self.on_frame = on_frame  # Called once per simulated frame; may raise to cancel the episode.
		self.fault = fault  # Internal acceptance hook. It is absent from the public MCP API.
		self.perturb = perturb  # Physical MuJoCo perturbation hook (S5); called per step in _apply_target.
		self.spec = load_physics_spec()
		base = next(c for c in SENSITIVITY_MATRIX if c.name == "baseline_2x2")
		cfg = NonIdealityCfg(
			latency_frames=base.latency_frames,
			backlash_rad=math.radians(base.backlash_deg),
			velocity_limit_rad_s=self.spec.arm_actuator.velocity_limit,
			encoder_ticks=4096,
		)
		# ``cell_factory`` swaps the physics engine (e.g. isaac_bridge.mirror.IsaacMirrorCell): the factory gets the
		# same arguments as Cell and returns a Cell whose ``step`` advances that engine and mirrors its state.
		self.cell = (cell_factory or Cell)(
			layout, SO101_MJCF_DIR, physics=self.spec, dynamic_clutter=True, assisted_grasp=False, face_probes=True
		)
		self.engine = getattr(self.cell, "engine", "mujoco")
		self.ep = PhysicsEpisode(self.cell, INITIAL_Q, layout.brick_xy, layout.brick_yaw_deg, cfg)
		# Bowl centres the skills use; realistic populates only perceived bowls in _sense_bowls (never layout truth).
		self.bowls = (
			{}
			if self.realistic
			else {
				"green": np.asarray(layout.green, dtype=np.float64),
				"pink": np.asarray(layout.pink, dtype=np.float64),
				**{
					f"extra:{i}": np.asarray((x, y), dtype=np.float64)
					for i, (_, x, y, _) in enumerate(layout.extra_bowls)
				},
			}
		)
		self.expert_type = ContinuousCarryExpert if smooth_carry else RelocateExpert
		if self.realistic:
			# Clutter is not perceived, and the clearance proxy below knows only the bowls.
			if layout.clutter and layout.bowl_names is None:
				raise ValueError("realistic sensing does not perceive clutter: use a layout without clutter")
			self.bowl_clearance = PerceivedBowlClearance(self.cell, self.bowls)
			self.expert_type = with_perceived_bowl_clearance(self.expert_type, self.bowl_clearance)
			self.expert_type = with_believed_brick_size(self.expert_type, self._believed_brick_size)
		self.expert = self._make_expert(ExpertParams(), RelocateParams(do_bottom=False, do_scan=False))
		self.node = ""
		self.phases: list[int] = []
		self.targets: list[np.ndarray] = []  # policy target per frame (before the command filter)
		self._clipped_frames = 0  # target clips of experts replaced by rewinds
		self.checkpoints: list[Checkpoint] = []
		self.branch = 0
		self.rewinds = 0  # physics restores to a checkpoint (each starts a new branch)
		self.perturbation_rewinds = 0  # scenario-forking restores for S5 perturbations (separate from online budget)
		self.retries = 0  # in-place recoveries from the current state (the branch continues)
		self._carry_epoch_start = 0
		self._parked_pose: tuple[np.ndarray, float | None] | None = None
		self.total_frames = 0
		self.wrist_count = 0
		self.bottom_count = 0
		self.held: tuple[np.ndarray, float] | None = None
		self.actual_destination = None
		self.execution_error = None
		if self.realistic:
			seed = int(getattr(layout, "defect_seed", 0))
			self._top_depth_rng = np.random.default_rng(seed + 40_000)
			self.record.top_depth_rng = self._top_depth_rng
		else:
			self._top_depth_rng = None
		self.record.start(self.cell)
		self.record.initial(self.cell, INITIAL_Q)
		# Brick perception (opt-in, see .perception): the top camera is calibrated before the episode, and the brick
		# is located in its depth image whenever a skill needs it on the table.
		self.perception = PerceptionCfg.from_env()
		if self.realistic and self.perception is None:
			self.perception = PerceptionCfg.realistic()
		self.brick_est = None
		self._bowl_estimates: dict[str, BowlEstimate | None] = {}
		self._last_carry_tcp: np.ndarray | None = None
		self._perceived_size = None  # realistic: the size class of the last brick perception (kept after a release)
		self._belief_obs = False
		self._grip_from_brick = None  # realistic: the carried brick in the gripper frame, believed at the grasp
		self._expected_yaw = None  # realistic: yaw the brick should show after the agent put it down itself
		self._park_xy = np.asarray(layout.brick_xy, dtype=np.float64)
		self.scene_error = None
		if self.perception is not None:
			seed = int(getattr(layout, "defect_seed", 0))
			calib_layout = self._calib_layout
			if getattr(layout, "calib_top", None) is not None:  # drift: calibrated on the nominal rig, then moved
				calib_layout = copy.copy(self._calib_layout)
				calib_layout.top = layout.calib_top
			self._cam_k, self._cam_est, info = calibrate_layout(calib_layout, seed, self.perception.calibration)
			info["calibrated_on"] = "nominal" if calib_layout is not self._calib_layout else "actual"
			info["cell_markers_removed"] = self.cell_markers_removed
			# Error of the estimate against the camera as it is mounted now (differs from cam_*_err under drift).
			top = layout.top
			right, up = top.axes()
			forward = -np.cross(right, up)  # MuJoCo cameras look along -z; OpenCV column 2 is the viewing axis
			info["actual_cam_pos_err_mm"] = round(
				float(np.linalg.norm(self._cam_est[:3, 3] - (top.x, top.y, top.height)) * 1000), 3
			)
			cosang = float(np.clip(np.dot(self._cam_est[:3, 2], forward / np.linalg.norm(forward)), -1.0, 1.0))
			info["actual_cam_axis_err_deg"] = round(math.degrees(math.acos(cosang)), 4)
			self._depth_rng = np.random.default_rng(seed + 30_000)
			self.event("camera_calibrated", **info)
			if "camera" in self.oracle:
				self._cam_est = true_world_from_cam(layout.top)
				self._note_oracle()
		if self.realistic:
			self._contact = GripperContact(use_effort=False)
			self._sense_bowls()

	def _note_oracle(self) -> None:
		if self.oracle and not self._oracle_logged:
			self._oracle_logged = True
			self.event("oracle_applied", parts=sorted(self.oracle))

	def _make_expert(self, params, relocate):
		ex = self.expert_type(self.cell, INITIAL_Q, self._expert_xy, params, relocate)
		if self.realistic:
			ex.gripper_monitor = GripperMonitor(GripperMonitorParams(effort_min=0.0, use_effort=False))
		self._configure_recheck(ex)
		return ex

	def _configure_recheck(self, expert=None) -> None:
		ex = self.expert if expert is None else expert
		spec = VARIANTS[self.motion]
		if spec is None:
			ex.recheck = None
			return
		if not self.realistic:
			raise ValueError("motion variants are for realistic sensing; truth sees the brick directly")
		heights, mode = spec
		ex.recheck = GraspLoop(heights, mode, self._wrist_look)

	def _get_brick_net(self):
		if self._brick_net is None:
			from .brick_pose_nn import BrickSegNet

			onnx_p = self.segmenter_onnx or Path("E:/Armigo_data_v3/perception/brickseg-v1/model/brickseg.onnx")
			self._brick_net = BrickSegNet(onnx_p)
		return self._brick_net

	def _wrist_look(self, h: float):
		"""One wrist-camera re-localisation during grasp descent: fuse and return replanned grasp or None."""
		cell, ex = self.cell, self.expert
		rgb = cell.rgb("wrist_cam")
		ex.kin.data.qpos[:] = cell.data.qpos
		w2c = believed_wrist_from_cam(cell, ex.kin.data)
		cid = mujoco.mj_name2id(cell.model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
		k = wrist_k(float(cell.model.cam_fovy[cid]), rgb.shape)
		size = self._believed_brick_size()
		top_z = float(self.brick_est.pos[2] + size[2] / 2)
		net = self._get_brick_net() if self.segmenter == "unet" else None
		fix, reason = localize_from_wrist(
			rgb,
			k,
			w2c,
			self.brick_est.pos[:2],
			self.brick_est.yaw,
			size,
			top_z,
			segmenter=self.segmenter,
			net=net,
		)
		info: dict = {"accepted": False}
		if fix is not None:
			fused, info = fuse(self.brick_est, fix, self.fuse_cfg)
			info |= {"area_px": fix.area_px, "fill": round(fix.fill, 3), "tilt_deg": round(fix.tilt_deg, 2)}
			if fix.z is not None:
				info["z_m"] = round(fix.z, 4)
			if fix.tilt_brick_deg is not None:
				info["tilt_brick_deg"] = round(fix.tilt_brick_deg, 2)
			if fix.reproj_px is not None:
				info["reproj_px"] = round(fix.reproj_px, 3)
			if fused is not None:
				self.brick_est = fused
		loop = ex.recheck
		rot = cell.data.xmat[cell.brick_body].reshape(3, 3)
		self.event(
			"wrist_fix",
			height_m=round(float(h), 4),
			found=fix is not None,
			reason=reason,
			mode=loop.mode,
			latency_s=loop.latency_s if loop.mode == "async" else 0.0,
			**info,
			# Truth diagnostics only (registered in TRUTH_DIAGNOSTIC_KEYS); never read by the agent.
			error_xy_mm=round(float(np.linalg.norm(self.brick_est.pos[:2] - cell.brick_pos()[:2]) * 1000), 3),
			error_yaw_deg=round(
				yaw_error_deg(self.brick_est.yaw, math.atan2(rot[1, 0], rot[0, 0]), self.layout.brick_size), 3
			),
		)
		if not info["accepted"]:
			return None
		with self._belief():
			ex.kin.data.qpos[:] = cell.data.qpos
			return ex._grasp_plan(open_g=ex._active_open_g)

	@property
	def recoveries(self) -> int:
		"""Recoveries used so far: physics rewinds plus in-place retries. ``Policy.max_rewinds`` bounds this sum (one
		shared recovery budget; ``truth`` only rewinds and ``realistic`` only retries)."""
		return self.rewinds + self.retries

	def _sense_bowls(self):
		"""Locate all bowls in the top images and keep the depth image as the empty-bowl reference."""
		depth = self._top_depth()
		if self.layout.bowl_rgb is not None:
			rgb_by_slot = {
				**self.layout.bowl_rgb,
				**{f"extra:{i}": rgb for i, (_, _, _, rgb) in enumerate(self.layout.extra_bowls)},
			}
			found = locate_bowls(
				self.cell.rgb("top"),
				depth,
				self._cam_k,
				self._cam_est,
				engine=self.engine,
				bands=bands_for(rgb_by_slot, engine=self.engine),
			)
		else:
			found = locate_bowls(self.cell.rgb("top"), depth, self._cam_k, self._cam_est, engine=self.engine)
		self._baseline_depth = depth
		self._bowl_estimates = dict(found)
		errors = {}
		positions_m = {}
		oracle_bowls = "bowls" in self.oracle
		if oracle_bowls:
			self._note_oracle()
		self.bowls.clear()
		for name, est in found.items():
			truth = np.asarray(self._true_bowl_xy(name), dtype=np.float64)
			if oracle_bowls:
				self.bowls[name] = truth
				if est is not None:
					self._bowl_estimates[name] = replace(est, xy=truth.copy())
				positions_m[name] = truth.round(5).tolist()
				errors[name] = 0.0
				continue
			if est is None:
				self.scene_error = "BOWL_NOT_FOUND"
				positions_m[name] = None
				continue
			self.bowls[name] = est.xy
			positions_m[name] = est.xy.round(5).tolist()
			errors[name] = round(float(np.linalg.norm(est.xy - truth) * 1000), 3)  # scoring only
		self.event(
			"bowls_perceived",
			positions_m=positions_m,
			error_xy_mm=errors,
		)

	def _grip_pose(self) -> np.ndarray:
		"""World-from-gripper frame from the joint readings (forward kinematics of the robot model)."""
		d, site = self.cell.data, self.cell.site
		# After mj_step the kinematics lag the state by one substep; recompute them so the pose does not depend on
		# whether something else happened to call mj_forward first (state is untouched: no warm-start change).
		mujoco.mj_forward(self.cell.model, d)
		tf = np.eye(4)
		tf[:3, :3] = d.site_xmat[site].reshape(3, 3)
		tf[:3, 3] = d.site_xpos[site]
		return tf

	def _brick_belief(self):
		"""``(position, quaternion xyzw)`` the realistic agent believes for the brick: carried with the gripper since
		the grasp, else as last perceived on the table; None before the first localisation."""
		if self._grip_from_brick is not None:
			tf = self._grip_pose() @ self._grip_from_brick
			quat = np.zeros(4)
			mujoco.mju_mat2Quat(quat, tf[:3, :3].flatten())
			return tf[:3, 3].copy(), quat[[1, 2, 3, 0]]
		if self.brick_est is not None:
			return self.brick_est.pos.copy(), self.brick_est.quat_xyzw()
		return None

	def _believed_brick_size(self) -> tuple[float, float, float]:
		"""Brick extents the realistic planners use: the size class perception read from the top-face footprint
		(``locate_brick``), never the simulator's brick."""
		if self.brick_est is not None:
			return tuple(self.brick_est.size)
		if self._perceived_size is not None:
			return self._perceived_size
		raise RuntimeError("realistic planning needs the brick's size: no brick perception yet")

	def _plan_brick_size(self):
		"""``brick_size`` for the planners: the believed size in ``realistic``; None (the simulator's) in ``truth``."""
		return self._believed_brick_size() if self.realistic else None

	def _table_belief_tf(self) -> np.ndarray:
		est = self.brick_est
		tf = np.eye(4)
		c, s = math.cos(est.yaw), math.sin(est.yaw)
		tf[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
		tf[:3, 3] = est.pos
		return tf

	def _top_depth(self):
		depth = self.cell.depth("top")  # MuJoCo render, or Isaac RTX through the mirror cell
		sigma = self.perception.depth_noise_m * (depth / 0.5) ** 2
		return depth + self._depth_rng.normal(0.0, 1.0, depth.shape) * sigma

	def _locate_top(self, depth, bowls):
		"""``(brick estimate or None, source)`` from one top depth image: with ``localise="vote"`` the pinned brick
		(``_brick_hint``) first, else the shape rule; otherwise the largest raised blob."""
		if self.localise != "vote":
			return locate_brick(depth, self._cam_k, self._cam_est, bowls), None
		from .brick_vote import candidates, shape_pick
		from .perception import brick_blobs

		est = source = None
		if self._brick_hint is not None:
			est = self._hinted_brick(depth, bowls)
			source = "pinned" if est is not None else None
		if est is None:
			cands = candidates(brick_blobs(depth, self._cam_k, self._cam_est, bowls))
			i = shape_pick(cands)
			source = "shape"
			if i is not None:
				est = locate_brick(depth, self._cam_k, self._cam_est, bowls, label=cands[i].label)
		return est, source

	def _arm_occludes(self, depth, bowls, est) -> bool:
		"""The blob under ``est`` borders arm-high pixels in the top image (``perception.arm_occludes_blob``)."""
		from .perception import arm_occludes_blob, brick_blobs

		blobs = brick_blobs(depth, self._cam_k, self._cam_est, bowls)
		if not blobs:
			return False
		blob = min(blobs, key=lambda b: float(np.linalg.norm(b.xy - est.pos[:2])))
		return arm_occludes_blob(depth, self._cam_k, self._cam_est, bowls, label=blob.label)

	def _clear_top_depth(self):
		"""``(top depth with the arm at rest, "prepare" | "rest")``: the image ``prepare`` took at rest when no frame
		has run since, else one taken after a move to rest (a ``prepare`` segment, ``mode="clear_view"``) before
		returning to ready (a ``return_ready`` segment). ``(None, None)`` when a move fails."""
		if self._rest_view is not None and self._rest_view[1] == self.total_frames:
			return self._rest_view[0], "prepare"
		ex = self.expert

		def clear_view():
			# Phase "approach" throughout, like the ready moves: "return" would mark the frames as a bowl release
			# for the motion-health evaluator.
			q_rest = ex.q_start.copy()
			q_rest[5] = ex.p.open_g
			yield from ex._move(q_rest, ex._carry_secs(q_rest, ex.p.ready_move_s, 1.0), "approach")
			yield from ex._hold(ex.p.final_hold_s, "approach")

		# The image is taken between the two motions, outside any generator, so a student twin (which replaces
		# generators by policy segments) takes it at the same frame with the same noise draw.
		if self._execute(clear_view()):
			return None, None
		depth = self._top_depth()
		if self._execute(ex._go_ready()):
			return None, None
		return depth, "rest"

	def _perceive(self, est=None, source=None):
		"""Locate the brick on the table through the calibrated top camera; None, or a failure code. ``est`` is a
		brick already located (``localise_brick``); with ``localise="vote"`` a pinned brick (``_brick_hint``) is
		looked up first, and otherwise the shape rule picks the blob."""
		if self.perception is None:
			return None
		if est is None:
			bowls = list(self.bowls.values())
			hint = self._brick_hint
			depth = self._top_depth()
			est, source = self._locate_top(depth, bowls)
			if self.realistic and (est is None or self._arm_occludes(depth, bowls, est)):
				# The arm at the ready pose stands between the top camera and the brick: read the top face from a
				# depth image taken with the arm at rest instead (the one of ``prepare`` when nothing has moved since).
				clear, how = self._clear_top_depth()
				if clear is not None:
					self._brick_hint = hint
					est, source = self._locate_top(clear, bowls)
					self.event("clear_view", source=how, found=est is not None)
		if "brick" in self.oracle:
			rot = self.cell.data.xmat[self.cell.brick_body].reshape(3, 3)
			true_yaw = math.atan2(rot[1, 0], rot[0, 0])
			true_pos = self.cell.brick_pos().copy()
			true_size = tuple(float(v) for v in self.layout.brick_size)
			est = (
				BrickEstimate(
					pos=true_pos,
					yaw=true_yaw,
					size=true_size,
					footprint_m=(true_size[0], true_size[1]),
					top_points=100,
				)
				if est is None
				else replace(est, pos=true_pos, yaw=true_yaw, size=true_size)
			)
			self._note_oracle()
		elif est is None:
			self.event("brick_perceived", found=False)
			return "BRICK_NOT_FOUND"
		if self._expected_yaw is not None:
			# The footprint only fixes the yaw up to its symmetry; after a put-down, keep the face numbering the
			# agent has been using by taking the symmetric yaw nearest the one it put the brick down with.
			period = math.pi if est.size[0] != est.size[1] else math.pi / 2
			est.yaw += period * round((self._expected_yaw - est.yaw) / period)
			self._expected_yaw = None
		self.brick_est = est
		self._perceived_size = tuple(est.size)
		parked_info: dict = {}
		if self._parked_pose is not None:
			park_xy, park_yaw = self._parked_pose
			self._parked_pose = None
			moved_mm = round(float(np.linalg.norm(est.pos[:2] - park_xy) * 1000), 3)
			moved = bool(
				np.linalg.norm(est.pos[:2] - park_xy) > RETRY_MOVED_M
				or (park_yaw is not None and yaw_error_deg(est.yaw, park_yaw, est.size) > RETRY_MOVED_DEG)
			)
			if self._s5_active():
				parked_info = {"moved": moved, "moved_mm": moved_mm}
		# Truth is logged for scoring only; nothing in the agent reads it back.
		rot = self.cell.data.xmat[self.cell.brick_body].reshape(3, 3)
		true_yaw = math.atan2(rot[1, 0], rot[0, 0])
		self.event(
			"brick_perceived",
			found=True,
			position_m=est.pos.round(5).tolist(),
			yaw_deg=round(math.degrees(est.yaw), 3),
			footprint_mm=[round(v * 1000, 2) for v in est.footprint_m],
			top_points=est.top_points,
			occlusion_extended_mm=list(est.extended_mm),
			error_xy_mm=round(float(np.linalg.norm(est.pos[:2] - self.cell.brick_pos()[:2]) * 1000), 3),
			error_yaw_deg=round(yaw_error_deg(est.yaw, true_yaw, self.layout.brick_size), 3),
			size_correct=bool(np.allclose(est.size, self.layout.brick_size)),
			**parked_info,
			**({"source": source} if source is not None else {}),
		)
		return None

	@contextmanager
	def _belief(self):
		"""Show planners that read the brick from the simulation the believed pose instead: the perceived one on the
		table, and in ``realistic`` also the carried one."""
		if self.realistic:
			# Realistic planners also see the designed wrist camera mount, not the actual (possibly offset) one.
			from contextlib import ExitStack

			with ExitStack() as stack:
				stack.enter_context(self.cell.nominal_wrist_cam())
				if self.belief_cameras == "estimated" and self._cam_est is not None:
					stack.enter_context(self.cell.camera_override("top", self._cam_est))
				stack.enter_context(self._believed_brick())
				yield
			return
		with self._believed_brick():
			yield

	@contextmanager
	def _believed_brick(self):
		if self.realistic:
			belief = self._brick_belief()
		elif self.brick_est is None or self.held is not None:
			belief = None
		else:
			belief = (self.brick_est.pos, self.brick_est.quat_xyzw())
		if belief is None:
			yield
			return
		m, d = self.cell.model, self.cell.data
		adr = self.cell.brick_qadr
		saved = d.qpos[adr : adr + 7].copy()
		x, y, z, w = belief[1]
		d.qpos[adr : adr + 3] = belief[0]
		d.qpos[adr + 3 : adr + 7] = (w, x, y, z)
		mujoco.mj_kinematics(m, d)
		try:
			yield
		finally:
			d.qpos[adr : adr + 7] = saved
			mujoco.mj_forward(m, d)

	def _vis_cell(self):
		"""The cell realistic crops, visible faces and view plans are computed on (``boundary="belief_cell"``).

		It is built only from what the agent believes: the perceived brick size class, the perceived bowl centres,
		the hand-eye estimate of the top camera and the designed wrist mount; at every call it is synced to the arm
		readings and the believed brick pose. ``self.cell`` in ``truth`` and under ``boundary="legacy"``."""
		if not self.realistic or self.boundary != "belief_cell":
			return self.cell
		size = tuple(round(float(v), 6) for v in self._believed_brick_size())
		bc = self._belief_cells.get(size)
		if bc is None:
			lay = copy.deepcopy(self.layout)
			lay.brick_size = size
			lay.green = tuple(float(v) for v in self.bowls["green"][:2])
			lay.pink = tuple(float(v) for v in self.bowls["pink"][:2])
			if lay.extra_bowls:
				lay.extra_bowls = [
					(cname, float(self.bowls[f"extra:{i}"][0]), float(self.bowls[f"extra:{i}"][1]), rgb)
					for i, (cname, _, _, rgb) in enumerate(lay.extra_bowls)
				]
			lay.wrist_cam_offset = None
			lay.clutter, lay.distractors = [], []
			bc = Cell(
				lay, SO101_MJCF_DIR, physics=self.spec, dynamic_clutter=False, assisted_grasp=False, face_probes=True
			)
			set_cv_camera(bc, "top", self._cam_est)
			self._belief_cells[size] = bc
			self.event("belief_cell", brick_size_m=list(size), top_camera="hand_eye_estimate")
		d = bc.data
		d.qpos[bc.qadr] = self.cell.data.qpos[self.cell.qadr]
		belief = self._brick_belief()
		if belief is not None:
			pos, q = belief
			adr = bc.brick_qadr
			d.qpos[adr : adr + 3] = pos
			d.qpos[adr + 3 : adr + 7] = (q[3], q[0], q[1], q[2])
		mujoco.mj_forward(bc.model, d)
		return bc

	@contextmanager
	def _planning_view(self):
		"""The cell planners and the detector crop render on, defect hidden: the belief cell, or (truth / legacy) the
		simulator's cell with the believed brick and the designed wrist mount shown."""
		vis = self._vis_cell()
		if vis is self.cell:
			with self._belief(), blind_geometry(self.cell):
				yield self.cell
		else:
			with blind_geometry(vis):
				yield vis

	def close(self):
		for bc in self._belief_cells.values():
			bc.close()
		self._belief_cells.clear()
		self.cell.close()

	def event(self, kind, **payload):
		self.record.event(kind, self.node, self.branch, self.ep.n_frames, **payload)

	def enter(self, node):
		self.node = node
		self.event("node")

	def _execute(self, generator):
		expert = self.expert
		expert._gen = generator
		expert.done = False
		while self.total_frames < MAX_TOTAL_FRAMES:
			if self.on_frame is not None:
				self.on_frame(self)
			obs = self._observe()
			target, phase = expert.act(obs)
			if expert.done:
				return None
			if error := self._apply_target(obs, target, phase):
				expert._gen.close()
				return error
		expert._gen.close()
		return "ACTION_BUDGET_EXHAUSTED"

	# Per-frame hooks shared by _execute and the student segment executor (agent/student_rollout.py).
	def _observe(self):
		"""The expert's per-frame observation (sensor-derived fields in ``realistic``). Updates the contact estimate."""
		obs = observe(self.cell, self.ep)
		if self.realistic:
			belief = self._brick_belief()
			if belief is None:
				obs["brick_pos"], obs["brick_quat"] = np.full(3, np.nan), np.array([0.0, 0.0, 0.0, 1.0])
			else:
				obs["brick_pos"], obs["brick_quat"] = belief
			obs["brick_vel"] = np.zeros(3)
			obs["jaw_normal"] = self._contact.update(
				float(obs["gripper_cmd"]), float(obs["q"][5]), float(obs["effort_frac"])
			)
		elif self._belief_obs and self.brick_est is not None:
			obs["brick_pos"] = self.brick_est.pos.copy()
			obs["brick_quat"] = self.brick_est.quat_xyzw()
		return obs

	def _s5_active(self):
		"""True on S5 perturbation runs (hook installed or forked branch); everything S5-specific is gated on this so
		unperturbed episodes keep their events and health verdicts exactly."""
		return self.perturb is not None or self.perturbation_rewinds > 0

	def _apply_target(self, obs, target, phase, *, executed_target=None, noisy=False):
		"""Record and apply one joint target through the command filter; returns a fault code or None."""
		exec_target = target if executed_target is None else executed_target
		# Observation at t is paired with the next unfiltered policy command.
		self.record.command(
			obs["q"],
			target,
			phase,
			gripper_cmd=float(obs["gripper_cmd"]),
			effort_frac=float(obs["effort_frac"]),
			noisy=noisy,
		)
		self.ep.step(exec_target)
		self.total_frames += 1
		self.phases.append(PHASES.index(phase))
		self.targets.append(np.asarray(exec_target, dtype=np.float64).copy())
		if self.perturb is not None:
			payload = self.perturb(self, obs, phase)
			if isinstance(payload, dict):
				self.event("perturbation", **payload)
		self.record.step(self.cell, self.node)
		if self.fault and self.fault(self.node, self.branch, self.ep.n_frames):
			return "INJECTED_MOTION_FAILURE"
		return None

	def _save(self, views):
		"""Checkpoint the skill boundary, also while the brick is held: the grasp is physical (no weld), so the
		integration state carries it. The checkpoint holds everything the agent's later behaviour depends on, so a new
		``TwinMujoco`` built from the same case continues identically (``resume_state``): the lift pose and grasp roll,
		the brick belief, perceived bowls and camera, contact filter, depth-noise RNG, expert parameters and counters.
		With a trace directory, the per-frame history and the empty-bowl depth reference are written next to it."""
		integration = np.empty(mujoco.mj_stateSize(self.cell.model, STATE_SPEC))
		mujoco.mj_getState(self.cell.model, self.cell.data, integration, STATE_SPEC)
		ex, filt = self.expert, self.ep.filt
		skill_state = {
			"last": ex.last.tolist(),
			"active_open": ex._active_open_g,
			"frame_idx": ex._frame_idx,
			"monitor_position": ex.gripper_monitor._prev_meas,
			"monitor_count": ex.gripper_monitor._run,
			# The previous observed joints (show-pose stillness) and the lost-brick and blocked-gripper counters.
			"prev_q": ex.obs["q"].tolist() if ex.obs else None,
			"no_dual": ex._no_dual,
			"last_block_dir": ex._last_block_dir,
		}
		filter_state = {
			"queue": [q.tolist() for q in filt._queue],
			"prev": filt._prev_out.tolist(),
			"backlash": filt._backlash_pos.tolist(),
		}
		held = None
		if self.held is not None:
			q_lift, roll = self.held
			held = {"q_lift": np.asarray(q_lift).tolist(), "roll": None if roll is None else float(roll)}
		grip = self._grip_from_brick
		base = getattr(self, "_base_params", None)
		directory = getattr(self.record, "checkpoints", None)
		directory = directory if isinstance(directory, Path) else None
		agent = {
			"grip_from_brick": None if grip is None else grip.tolist(),
			"contact_prev": self._contact.prev if self.realistic else None,
			"park_xy": self._park_xy.tolist(),
			"expected_yaw": self._expected_yaw,
			"carry_epoch_start": int(self._carry_epoch_start),
			"parked_pose": (
				None
				if self._parked_pose is None
				else {"xy": self._parked_pose[0].tolist(), "yaw": self._parked_pose[1]}
			),
			"brick_est": brick_to_json(self.brick_est),
			"perceived_size": None if self._perceived_size is None else [float(v) for v in self._perceived_size],
			"bowls": {name: np.asarray(xy, dtype=np.float64).tolist() for name, xy in self.bowls.items()},
			"bowl_estimates": {
				name: None
				if est is None
				else {
					"xy": np.asarray(est.xy, dtype=np.float64).tolist(),
					"radius_m": float(est.radius_m),
					"pixels": int(est.pixels),
					"rim_points": int(est.rim_points),
					"fit_residual_m": float(est.fit_residual_m),
				}
				for name, est in self._bowl_estimates.items()
			},
			"expert_params": asdict(ex.p),
			"relocate_params": asdict(ex.rp),
			"base_params": None if base is None else [asdict(base[0]), asdict(base[1])],
			"scan_peak_speed": self.scan_peak_speed,
			"recovery_variants": list(self.recovery_variants),
			"clipped_frames": int(self._clipped_frames + getattr(ex, "clipped_frames", 0)),
			"llm_stats": copy.deepcopy(getattr(self, "_llm_stats", None)),
			"oracle": sorted(self.oracle),
			"oracle_logged": self._oracle_logged,
			"motion": self.motion,
			"fuse_cfg": asdict(self.fuse_cfg),
			"belief_cameras": self.belief_cameras,
			"evidence": self.evidence,
			"segmenter": self.segmenter,
			"segmenter_onnx": None if self.segmenter_onnx is None else str(self.segmenter_onnx),
			"cell_markers": self.cell_markers,
			"ring_localise": self.ring_localise,
			"localise": self.localise,
			"brick_hint": None
			if self._brick_hint is None
			else {k: [float(x) for x in v] for k, v in self._brick_hint.items()},
			"segments": self._checkpoint_extra(),
		}
		if self.drift_alarm:
			agent["recalibration_attempted"] = self._recalibration_attempted
		if self.perception is not None:
			agent["depth_rng"] = self._depth_rng.bit_generator.state
			agent["camera"] = {"k": np.asarray(self._cam_k).tolist(), "world_from_cam": self._cam_est.tolist()}
		if self.realistic:
			agent["top_depth_rng"] = self._top_depth_rng.bit_generator.state
		if self.realistic and directory is not None:
			agent["empty_bowl_depth"] = self._write_depth_reference(directory)
		# View budget counters (episode-wide, rewound branches included): restored only by a cross-process resume.
		agent["view_counters"] = {
			"scored": self.scored_count,
			"reviews": self.review_count,
			"review_exhausted": sorted(self.review_exhausted),
		}
		snapshot = getattr(self.cell, "engine_snapshot", None)
		if snapshot is not None:
			# Isaac: a native checkpoint exists only at the validated boundary (gripper open, brick on the table);
			# None elsewhere, and rewind_options below never offers such a checkpoint.
			agent["engine_snapshot"] = snapshot()
		agent = json.loads(json.dumps(agent, allow_nan=False))  # in memory exactly as on disk (tuples become lists)
		checkpoint = Checkpoint(
			f"checkpoint-{len(self.checkpoints)}",
			self.branch,
			self.ep.n_frames,
			integration,
			skill_state,
			filter_state,
			self.wrist_count,
			self.bottom_count,
			copy.deepcopy(views),
			retries=self.retries,
			held=held,
			agent=agent,
		)
		self.record.checkpoint(checkpoint)
		if directory is not None:
			self._write_history(directory / f"{checkpoint.id}{HISTORY_SUFFIX}")
		self.checkpoints.append(checkpoint)
		self.event("checkpoint", checkpoint=checkpoint.id)

	def _checkpoint_extra(self):
		"""Subclass state a resumed episode needs (the segment twins' segment rows); JSON, or None."""
		return

	def _restore_extra(self, extra):
		del extra

	def _write_depth_reference(self, directory: Path) -> str:
		"""Store the empty-bowl depth reference once per content; returns its file name."""
		depth = np.ascontiguousarray(self._baseline_depth, dtype=np.float64)
		name = f"empty-bowl-depth-{hashlib.sha256(depth.tobytes()).hexdigest()[:16]}.npy"
		if not (directory / name).is_file():
			np.save(directory / name, depth, allow_pickle=False)
		return name

	def _write_history(self, path: Path):
		ep = self.ep
		arrays = {f"tel_{k}": np.asarray(v, dtype=np.float64) for k, v in ep._logs.items()}
		arrays["jaw_env_force"] = np.asarray(ep.jaw_env_force, dtype=np.float64)
		arrays["phases"] = np.asarray(self.phases, dtype=np.int64)
		arrays["targets"] = np.asarray(self.targets, dtype=np.float64).reshape(-1, 6)
		with path.open("wb") as f:
			np.savez(f, **arrays)

	def prepare(self):
		def ready():
			yield from self.expert._hold(self.expert.p.rest_s, "rest")
			yield from self.expert._go_ready()

		if self.scene_error:
			return self.scene_error
		# The arm starts at rest, where it hides none of the reach band from the top camera. Taken outside the
		# motion generator so expert and student twins draw the same depth noise.
		rest_depth = self._top_depth() if self.realistic else None
		error = self._execute(ready())
		self._rest_view = None if rest_depth is None else (rest_depth, self.total_frames)  # valid until a frame runs
		if not error:
			if self.realistic:  # the arm hid other pixels at rest: keep the farther reading of each
				self._baseline_depth = np.maximum(self._baseline_depth, self._top_depth())
			self._save([])
		return error

	def can_scan_wrist(self, policy):
		self.policy = policy  # read by decision_context for the remaining-budget fields
		return self.wrist_count < policy.max_wrist_views and self._views_left() > 0

	def can_present_bottom(self, policy):
		self.policy = policy
		return self.bottom_count < policy.max_bottom_views and self._views_left() > 0

	def _views_left(self) -> float:
		return math.inf if self.max_scored_views is None else self.max_scored_views - self.scored_count

	def _decide(self, views):
		"""``(label, covered faces)`` under the episode's strategy (A: first alarm decides; B / C: re-view rule)."""
		if self.strategy == "A":
			return decision(views)
		label, covered, _ = review_decision(views)
		return label, covered

	def review_pending(self, views) -> tuple[list[dict], bool]:
		"""``(unresolved alarms, a re-view can still be taken for one of them)``."""
		_, _, pending = review_decision(views)
		open_ = [p for p in pending if p["alarm"] not in self.review_exhausted]
		return pending, bool(open_) and self.review_count < self.max_review_views and self._views_left() > 0

	def _geometry(self):
		cell = self.cell
		m, d = cell.model, cell.data

		def rounded(a):
			return np.asarray(a).round(5).tolist()

		# World poses of all simulator bodies. Repeated meshes such as the bowl-wall wedges share
		# a parent, so their 336 constant local transforms need not be retransmitted at every decision.
		bodies = [
			[m.body(i).name, i, int(m.body_parentid[i]), *rounded(d.xpos[i]), *rounded(d.xquat[i][[1, 2, 3, 0]])]
			for i in range(m.nbody)
		]

		return {
			"brick_size_m": self.layout.brick_size,
			"brick_position_m": rounded(cell.brick_pos()),
			"brick_rotation_xyzw": rounded(cell.brick_quat_xyzw()),
			"joints_rad": rounded(cell.joints()),
			"tcp_position_m": rounded(cell.tcp_pos()),
			"grasped": self.held is not None,
			"joint_limits_rad": rounded(cell.jnt_range),
			"body_columns": ["name", "body_id", "parent_id", "xyz_m", "quaternion_xyzw"],
			"bodies": bodies,
			"bowl_radius_m": BOWL_R,
			"collision_model": "SO-101 / R6_box_floor_wedge_walls",
			"bowl_positions_m": {"green": self.layout.green, "pink": self.layout.pink},
			"cameras": {
				m.camera(i).name: {
					"position_m": rounded(d.cam_xpos[i]),
					"rotation": rounded(d.cam_xmat[i]),
					"intrinsic": rounded(m.cam_intrinsic[i]),
					"fovy_deg": float(m.cam_fovy[i]),
				}
				for i in range(m.ncam)
			},
			"pinch_normal_n": rounded(self.ep.last("pinch_normal")) if self.ep.n_frames else [0.0, 0.0],
		}

	SKILL_NOTES: ClassVar[dict[str, str]] = {
		"scan_wrist": "move the wrist camera over the brick to read the side and top faces (0-4) not yet cleared",
		"present_bottom": "grasp the brick and show its bottom face (5) to the top camera",
		"stop": "give up: the episode ends as a failure with the brick unsorted",
	}

	def decision_context(self, allowed, views, failures, notes=None):
		"""Compact, leak-checked state for the LLM: coverage, per-view scores, failures, budgets, checkpoints."""
		covered = decision(views)[1]
		cell = self.cell
		if self.realistic:
			belief = self._brick_belief()
			brick = np.full(3, np.nan) if belief is None else belief[0]
		else:
			brick = cell.brick_pos()
		policy = getattr(self, "policy", None)  # set by whoever owns the Policy; absent means limits are unknown
		budget = {
			"wrist_views_used": self.wrist_count,
			"bottom_views_used": self.bottom_count,
			"recoveries_used": self.recoveries,
			"frames_left": MAX_TOTAL_FRAMES - self.total_frames,
		}
		if policy is not None:
			budget["wrist_views_left"] = policy.max_wrist_views - self.wrist_count
			budget["bottom_views_left"] = policy.max_bottom_views - self.bottom_count
			budget["recoveries_left"] = policy.max_rewinds - self.recoveries
		context = {
			"covered_faces": sorted(covered),
			"missing_faces": [f for f in range(6) if f not in covered],
			"views": [
				{
					"id": v["id"],
					"camera": v["camera"],
					"faces": v["faces"],
					"score": None if v["score"] is None else round(v["score"], 3),
					"threshold": round(v["threshold"], 3),
					"margin": None if v["score"] is None else round(v["score"] - v["threshold"], 3),
				}
				for v in views
			],
			"failures": [{k: f[k] for k in ("code", "skill", "reason_codes") if k in f} for f in failures],
			"budget": budget,
			"grasped": self.held is not None,
		}
		if np.isfinite(brick).all():
			context["brick"] = {"xy_m": np.asarray(brick[:2]).round(3).tolist(), "z_m": round(float(brick[2]), 3)}
		described = {k: self.SKILL_NOTES[k] for k in allowed if k in self.SKILL_NOTES}
		checkpoints = []
		for c in self.checkpoints:
			if c.id in allowed:
				faces = sorted(decision(c.views)[1])
				checkpoints.append({"id": c.id, "frame": c.frame, "faces_covered": faces})
				described[c.id] = f"rewind to frame {c.frame} with faces {faces} covered"
		if checkpoints:
			context["checkpoints"] = checkpoints
		context["choice_notes"] = {**described, **{k: str(v) for k, v in (notes or {}).items() if k in allowed}}
		# Nothing about the defect, its face or the label may reach the model.
		assert_no_defect_leakage(context)
		return context

	def _add_decision_history(self, context):
		"""Place graph-approved views and safe trace details in chronological prompt order."""
		views = {view["id"]: {"kind": "view", **view} for view in context.pop("views")}
		failures = context.pop("failures")
		if failures:
			context["latest_failure"] = failures[-1]
		history = []
		recorded_failures = 0
		fields = {
			"failure": ("code", "skill", "reason_codes"),
			"retry": ("reason", "variant"),
			"retried": ("brick_moved", "views_kept"),
			"rewound": ("checkpoint", "reason", "variant"),
		}
		for event in getattr(self.record, "events", ()):
			kind = event.get("kind")
			node = event.get("node")
			if kind == "view":
				view = views.pop(event.get("observation", {}).get("id"), None)
				if view is not None:
					history.append(view)
			elif kind == "node" and node in {"prepare", "recalibrate", "scan_wrist", "present_bottom", "place"}:
				history.append({"kind": "skill_started", "skill": node})
			elif kind == "decision" and node in {"choose_view", "recover"}:
				entry = {"kind": kind, "node": node, "choice": event["choice"]}
				if node == "recover":
					entry["choices"] = event["choices"]
				history.append(entry)
			elif kind in fields:
				history.append({"kind": kind, **{key: event[key] for key in fields[kind] if key in event}})
				recorded_failures += kind == "failure"
		# Following a process restart, checkpoint observations and earlier failures may have no local trace rows.
		inherited_failures = failures[: -max(recorded_failures, 1)]
		context["history"] = [
			*({"kind": "failure", **failure} for failure in inherited_failures),
			*views.values(),
			*history,
		]
		assert_no_defect_leakage(context)

	def _note_llm(self, source, info):
		stats = getattr(self, "_llm_stats", None)
		if stats is None:
			stats = self._llm_stats = {
				"consulted": 0,
				"adopted": 0,
				"fallback_reasons": {},
				"prompt_tokens": [],
				"latency_s_total": 0.0,
				"attempts": 0,
			}
		stats["consulted"] += 1
		stats["adopted"] += source == "llm"
		if info.get("reason"):
			stats["fallback_reasons"][info["reason"]] = stats["fallback_reasons"].get(info["reason"], 0) + 1
		stats["prompt_tokens"].append(int(info.get("prompt_tokens") or 0))
		stats["latency_s_total"] = round(stats["latency_s_total"] + float(info.get("latency_s") or 0.0), 4)
		stats["attempts"] += int(info.get("attempts") or 0)

	def _llm_summary(self):
		stats = getattr(self, "_llm_stats", None) or {
			"consulted": 0,
			"adopted": 0,
			"fallback_reasons": {},
			"prompt_tokens": [],
			"latency_s_total": 0.0,
			"attempts": 0,
		}
		tokens = stats["prompt_tokens"]
		return {
			**stats,
			"prompt_tokens_median": float(np.median(tokens)) if tokens else None,
			"prompt_tokens_max": max(tokens) if tokens else None,
		}

	def choose(self, allowed, views, failures, notes=None):
		info = {}
		forced, self.forced_choice = getattr(self, "forced_choice", None), None
		if forced is not None:  # an outside agent (the step MCP) already chose
			if forced not in allowed:
				raise ValueError(f"{forced!r} is not one of {allowed}")
			action, source = forced, "agent"
		elif self.llm is None or len(allowed) == 1 and not failures:
			action, source = allowed[0], "policy"
		else:
			context = self.decision_context(allowed, views, failures, notes)
			if os.environ.get("ARMIGO_LLM_CONTEXT_MODE", "history") == "history":
				self._add_decision_history(context)
			action, source, info = select_skill(self.llm, allowed, context)
			self._note_llm(source, info)
		self.event(
			"decision",
			choices=allowed,
			choice=action,
			source=source,
			evidence=[v["id"] for v in views],
			prior_failure=bool(failures),
			fallback_reason=info.get("reason"),
			prompt_tokens=info.get("prompt_tokens"),
			cached_tokens=info.get("cached_tokens"),
			latency_s=info.get("latency_s"),
			attempts=info.get("attempts"),
		)
		return action

	def _capture(self, camera, *, in_motion: bool | None = None, decided_after_frames: int | None = None):
		cell = self.cell
		rgb = cell.rgb(camera)
		crop_extra: dict = {}
		if self.realistic:
			# Visible faces and the crop come from the believed brick; the image is the real one.
			with self._planning_view() as vis:
				geometry = capture_view(vis, camera)
				image, bbox = detector_crop(vis, camera, None, vis.data, self.crop_size, clamp=True, rgb=rgb)
			if self.evidence != "hold" and bbox is not None:
				win_cx, win_cy = bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0
				gate_px = max(float(max(bbox[2], bbox[3])), 20.0)
				blob_mask = _blob(rgb, (win_cx, win_cy), gate_px)
				if blob_mask is not None and blob_mask.any():
					ys, xs = np.nonzero(blob_mask)
					cx, cy = float(xs.mean()), float(ys.mean())
					with self._planning_view() as vis:
						image, bbox = detector_crop(
							vis, camera, None, vis.data, self.crop_size, clamp=True, rgb=rgb, center_px=(cx, cy)
						)
					crop_extra = {
						"crop_source": "image",
						"crop_shift_px": round(math.hypot(cx - win_cx, cy - win_cy), 2),
					}
				else:
					crop_extra = {"crop_source": "projection", "crop_shift_px": 0.0}
		else:
			with blind_geometry(cell):
				geometry = capture_view(cell, camera)
			image, bbox = detector_crop(cell, camera, None, cell.data, self.crop_size, clamp=True, rgb=rgb)
		min_px = BOTTOM_MIN_PX if camera == "top" else WRIST_MIN_PX
		faces = [i for i, n in enumerate(geometry.faces_px) if n >= min_px and ((i == 5) == (camera == "top"))]
		threshold = (
			self.bottom_threshold
			if camera == "top" and self.bottom_threshold is not None
			else self.scorer.threshold(camera)
		)
		score = elapsed_ms = None
		heads = {}
		error = None
		reason = None
		if image is None:
			reason = "NO_CROP"
		elif geometry.brick_px < MIN_BRICK_PX:
			reason = "TOO_FEW_BRICK_PIXELS"
		elif not faces:
			reason = "NO_READABLE_FACE"
		else:
			from ..inspection_scorer import ScoreServerError

			self.event("detector_started", camera=camera)
			started = time.perf_counter()
			try:
				score_details = getattr(self.scorer, "score_with_details", None)
				if score_details is None:
					score = float(self.scorer.score(camera, image))
				else:
					score, heads = score_details(camera, image)
				if not math.isfinite(score):
					score = None
					reason = "NONFINITE_SCORE"
			except ScoreServerError as exc:
				reason = "DETECTOR_UNAVAILABLE"
				error = exc
			elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
		view = {
			"id": f"{self.branch}:{self.ep.n_frames}:{camera}",
			"branch": self.branch,
			"frame": self.ep.n_frames,
			"camera": camera,
			"faces": faces,
			"face_pixels": list(geometry.faces_px),
			"score": score,
			"threshold": float(threshold),
			"crop_bbox": bbox,
			"q": [round(float(v), 5) for v in cell.joints()],
			**crop_extra,
			**({"in_motion": bool(in_motion)} if in_motion is not None else {}),
			**({"decided_after_frames": int(decided_after_frames)} if decided_after_frames is not None else {}),
		}
		if score is not None:
			self.scored_count += 1
		if image is not None:
			self._crops[view["id"]] = image
		self.record.view(view, rgb, image)
		# Scoring only (S6 grader): the true visible faces and cross, over the whole frame and inside the crop the
		# detector actually received (same bbox, margin, border shift and nearest-neighbour resize), and the true and
		# believed brick orientation so the grader can map the believed face numbering (yaw known modulo the
		# footprint symmetry) to the true one. Faces are counted with the cross hidden, the cross with it shown.
		true_quat = cell.brick_quat_xyzw()
		with blind_geometry(cell):
			true_face_map, _ = cell.visibility_labels(camera)
		cross_face_map, true_cross_map = cell.visibility_labels(camera)
		true_faces = np.bincount(true_face_map[true_face_map >= 0], minlength=len(geometry.faces_px))
		crop_truth = {}
		if bbox is not None:
			faces_crop = crop_counts(true_face_map, true_cross_map, bbox, self.crop_size, DETECTOR_PAD, clamp=True)
			cross_crop = crop_counts(cross_face_map, true_cross_map, bbox, self.crop_size, DETECTOR_PAD, clamp=True)
			crop_truth = {
				"true_face_pixels_crop": faces_crop["faces_input"],
				"true_face_pixels_crop_native": faces_crop["faces_native"],
				"true_cross_px_crop": cross_crop["cross_input"],
				"true_cross_px_crop_native": cross_crop["cross_native"],
				"truth_crop": {
					"bbox": [int(v) for v in bbox],
					"size": int(self.crop_size),
					"pad": DETECTOR_PAD,
					"clamp": True,
					"frame_hw": [int(v) for v in true_face_map.shape],
					"region": faces_crop["region"],
					"input_px_per_native_px": faces_crop["input_px_per_native_px"],
				},
			}
		if self.realistic:
			belief = self._brick_belief()
			belief_err = (
				None if belief is None else round(float(np.linalg.norm(belief[0] - cell.brick_pos()) * 1000), 2)
			)
			belief_quat = None if belief is None else [round(float(v), 6) for v in belief[1]]
		else:  # faces and crop were computed on the simulator's brick
			belief_err, belief_quat = None, [round(float(v), 6) for v in true_quat]
		self.event(
			"view_truth",
			id=view["id"],
			true_face_pixels=[int(v) for v in true_faces],
			belief_error_mm=belief_err,
			true_cross_px=int(np.count_nonzero(true_cross_map)),
			**crop_truth,
			true_brick_quat_xyzw=[round(float(v), 6) for v in true_quat],
			belief_brick_quat_xyzw=belief_quat,
		)
		# Keep diagnostics in the trace, separate from the decision evidence supplied to the agent.
		self.event(
			"view" if error is None else "detector_error",
			observation=view,
			detector={"head_scores": heads, "elapsed_ms": elapsed_ms, "reason": reason},
		)
		if error is not None:
			raise error
		return view

	def _straight_clear(self, first, last, *, held=False):
		with self._belief():
			if not self.realistic:  # truth: bowls / clutter as they are now (realistic: perceived-bowl proxy)
				self.expert.kin.data.qpos[:] = self.cell.data.qpos
			# Sample the swept arm and bowl/clutter clearance in addition to table height.
			qs = [first + a * (last - first) for a in np.linspace(0, 1, 10)]
			size = self._plan_brick_size() if held else None
			low = path_min_z(self.cell, first, last, carried=held, brick_size=size)
			return low >= 0.015 and self.expert._clearance(qs) >= 0

	def _park_held(self):
		q_lift, roll = self.held
		ex = self.expert
		# Put the brick back where it was found: the perceived spot, or its layout position without perception.
		if self.brick_est is not None:
			self._park_xy = self.brick_est.pos[:2].copy()
		if not self._straight_clear(ex.last, q_lift, held=True):
			return "PARK_PATH_BLOCKED"

		def park():
			yield from ex._carry(ex._move(q_lift, ex._carry_secs(q_lift, ex.p.show_move_s), "show_move"))
			if not ex.lost:
				yield from ex._put_down(self._park_xy, roll, q_lift)
				if self.realistic and self._grip_from_brick is not None:
					# The put-down retreats straight up, so the gripper still has the release orientation.
					rot = (self._grip_pose() @ self._grip_from_brick)[:3, :3]
					self._expected_yaw = math.atan2(rot[1, 0], rot[0, 0])
			if not ex.lost:
				yield from ex._go_ready()

		error = self._execute(park())
		if error or ex.lost:
			return error or "BRICK_LOST"
		self.held = None
		if self.realistic:  # released: where it landed is found by looking again
			self._grip_from_brick = None
			self.brick_est = None
		self._parked_pose = (self._park_xy.copy(), self._expected_yaw)
		self._carry_epoch_start = self.ep.n_frames
		return None

	def _wrist_dense(self, previous, views, steps):
		"""C5 dense in-motion scan: sweep once without mid-stops, sample every ``spacing_deg`` with 0.08 m/s TCP gate."""
		ex, cell = self.expert, self.cell
		spacing_deg = 10.0 if self.evidence == "dense-10" else 20.0
		spacing_rad = math.radians(spacing_deg)
		delay_frames = max(1, round(self.detector_latency_s * FPS))

		full_path: list[np.ndarray] = []
		pending: list[np.ndarray] = []
		cursor = ex.last.copy()
		for candidate, via in steps:
			seg = [*pending, *via, candidate.q]
			if not any(candidate.faces_px[f] >= WRIST_MIN_PX for f in range(5)):
				pending = seg if via else []
				continue
			pending = []
			if not all(self._straight_clear(a, b) for a, b in itertools.pairwise([cursor, *seg])):
				continue
			full_path.extend(seg)
			cursor = seg[-1]
		if not full_path:
			return None

		pts = [ex.last.copy(), *full_path]
		seg_arcs = [
			float(np.max(np.abs(b[:5] - a[:5])) + 1e-6 * np.sum(np.abs(b - a))) for a, b in itertools.pairwise(pts)
		]
		total_arc = float(sum(seg_arcs))
		progress = [0.0]

		def on_prog(frac: float) -> None:
			progress[0] = float(frac)

		due: list[tuple[int, dict]] = []
		next_arc = spacing_rad
		defer = 0
		prev_tcp = cell.tcp_pos()
		stop_early = False
		step_i = 0

		def _flush_due(cur_step: int) -> bool:
			remaining = []
			for ready_step, v in due:
				if cur_step >= ready_step:
					views.append(v)
				else:
					remaining.append((ready_step, v))
			due[:] = remaining
			return self._decide([*previous, *views])[0] == "defective"

		def dense_scan(path):
			nonlocal next_arc, defer, prev_tcp, stop_early, step_i
			for target, phase in sweep(
				ex, path, self.scan_peak_speed, ex.rp.scan_move_s, pause_s=0.0, on_progress=on_prog
			):
				yield target, phase
				step_i += 1
				cur_tcp = cell.tcp_pos()
				tcp_speed = float(np.linalg.norm(cur_tcp - prev_tcp) * FPS)
				prev_tcp = cur_tcp
				cur_arc = progress[0] * total_arc
				if cur_arc >= next_arc or defer > 0:
					if tcp_speed <= 0.08 or defer >= 5:
						if self._views_left() - len(due) > 0:
							v = self._capture("wrist_cam", in_motion=True, decided_after_frames=delay_frames)
							due.append((step_i + delay_frames, v))
						next_arc = max(next_arc + spacing_rad, cur_arc + spacing_rad)
						defer = 0
					else:
						defer += 1
				if _flush_due(step_i):
					stop_early = True
					return
			if self._decide([*previous, *views])[0] != "defective":
				for target, phase in ex._hold_servo(path[-1], ex.rp.scan_hold_s, "approach"):
					yield target, phase
					step_i += 1
					if _flush_due(step_i):
						stop_early = True
						return

		if err := self._execute(dense_scan(full_path)):
			return err
		if not stop_early and self._decide([*previous, *views])[0] != "defective":
			_flush_due(step_i + delay_frames + 1)
			if self._views_left() > 0 and self._decide([*previous, *views])[0] != "defective":
				views.append(self._capture("wrist_cam", in_motion=False, decided_after_frames=0))
		if not stop_early:
			for _, v in due:
				views.append(v)
		due.clear()
		if (
			self.strategy != "A"
			and views
			and _alarm(views[-1])
			and (err := self._review(previous, views, "wrist_alarm"))
		):
			return err
		return None

	def _wrist(self, previous, views):
		self.wrist_count += 1
		if self.held is not None:
			error = self._park_held()
			if error:
				return error
		if error := self._perceive():
			return error
		ex, cell = self.expert, self.cell
		with self._belief():
			ex.kin.data.qpos[:] = cell.data.qpos
		attempt = min(self.wrist_count - 1 + self.recoveries, 2)
		steps, mode, extra = self._scan_plan(previous, attempt)
		self.event("scan_plan", mode=mode, candidates=len(steps), **extra)
		if self.evidence.startswith("dense-"):
			if error := self._wrist_dense(previous, views, steps):
				return error
			if error := self._execute(ex._go_ready()):
				return error
			self._save([*previous, *views])
			return None
		pending: list[np.ndarray] = []  # ring poses of skipped views, still swept through to stay on the ring
		for candidate, via in steps:
			path = [*pending, *via, candidate.q]
			if self._views_left() <= 0:
				break
			_, covered = self._decide([*previous, *views])
			if not any(f not in covered and candidate.faces_px[f] >= WRIST_MIN_PX for f in range(5)):
				pending = path if via else []
				continue
			pending = []
			if not all(self._straight_clear(a, b) for a, b in itertools.pairwise([ex.last, *path])):
				continue

			def look(path):
				yield from sweep(
					ex,
					path,
					self.scan_peak_speed,
					ex.rp.scan_move_s,
					pause_s=self.sweep_pause_s if len(path) > 1 else None,
				)
				yield from ex._hold_servo(path[-1], ex.rp.scan_hold_s, "approach")

			if error := self._execute(look(path)):
				return error
			views.append(self._capture("wrist_cam"))
			# Decision point 1 (B / C): an alarm inside the scan is re-viewed here, before the scan can end.
			if self.strategy != "A" and _alarm(views[-1]) and (error := self._review(previous, views, "wrist_alarm")):
				return error
			if self._decide([*previous, *views])[0] == "defective":
				break
		if error := self._execute(ex._go_ready()):
			return error
		self._save([*previous, *views])
		return None

	def _scan_plan(self, previous, attempt):
		"""``([(view, via poses)], mode, event fields)``: the orbit scan around the upward face when it covers every
		side and top face not cleared yet, else the hover scan of :func:`plan_table_scan`."""
		ex = self.expert
		need = [f for f in range(5) if f not in self._decide(previous)[1]]
		with self._planning_view() as vis:
			plan = plan_orbit_scan(
				vis,
				ex.kin,
				ex.last,
				gripper=ex.p.open_g,
				params=ORBIT_ATTEMPTS[attempt],
				faces=need,
				min_px=WRIST_MIN_PX,
				brick_size=self._plan_brick_size(),
			)
		reached = {f for v in plan.views for f in need if v.faces_px[f] >= WRIST_MIN_PX}
		if plan.views and reached == set(need):
			extra = {"tip_height_m": plan.tip_height_m, "standoff_m": plan.standoff_m, "ring_poses": len(plan.ring)}
			return list(zip(plan.views, plan.via, strict=True)), plan.mode, extra
		modes = (
			{"heights": (0.06, 0.075, 0.09), "offset_m": 0.03},
			{"heights": (0.07, 0.085, 0.10), "offset_m": 0.02},
			{"heights": (0.055, 0.08, 0.095), "offset_m": 0.04},
		)
		with self._planning_view() as vis:
			candidates, mode = plan_table_scan(
				vis, ex.kin, ex.last, gripper=ex.p.open_g, min_px=WRIST_MIN_PX, **modes[attempt]
			)
		return [(c, []) for c in candidates], mode, {"orbit_faces": sorted(reached)}

	def _grasp(self):
		ex = self.expert
		ex.grasp_retries = ex.block_replans = 0
		ex._active_open_g = float(ex.p.open_g)
		if error := self._perceive():
			return error
		if self.ring_localise:
			error, _ = self._ring_localise()
			if error:
				return error

		def grasp_and_lift():
			self._belief_obs = True  # the grasp is planned on the perceived pose
			try:
				grasp = yield from ex._grasp()
			finally:
				self._belief_obs = False
			if grasp is not None and self.realistic:
				# From now on the brick is believed to move with the gripper, as it was perceived before the grasp.
				self._grip_from_brick = np.linalg.inv(self._grip_pose()) @ self._table_belief_tf()
			if grasp is not None:
				q_lift = yield from ex._lift(grasp)
				self.held = (q_lift, grasp.roll)

		error = self._execute(grasp_and_lift())
		return error or ("BRICK_LOST" if ex.lost else ex.aborted)

	def _wrist_view(self, est, q, entry: bool, view: int, views: int):
		"""Move to ring pose ``q`` (from straight above when ``entry``), hold, and fit the brick believed at ``est``
		in the wrist image (silhouette segmenter, designed mount). Returns ``(fix or None, reject reason, error)``."""
		from .wrist_ring import RING_ENTRY_DZ_M

		ex, cell = self.expert, self.cell
		size = tuple(est.size)
		top_z = float(est.pos[2] + size[2] / 2)
		q = q.copy()
		q[5] = ex.last[5]
		path = [q]
		if entry:
			ex.kin.data.qpos[ex.kin.qadr] = q
			above = ex._ik(np.array([*est.pos[:2], top_z + RING_ENTRY_DZ_M + 0.04]), q, float(q[4]))
			above[5] = q[5]
			path = [above, q]

		def ring_locate(path=path, view=view, views=views):
			yield from sweep(ex, path, self.scan_peak_speed, ex.rp.scan_move_s, pause_s=None)
			yield from ex._hold_servo(path[-1], ex.rp.scan_hold_s, "approach")

		if error := self._execute(ring_locate()):
			return None, None, error
		rgb = cell.rgb("wrist_cam")
		ex.kin.data.qpos[:] = cell.data.qpos
		cid = mujoco.mj_name2id(cell.model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
		fix, reason = localize_from_wrist(
			rgb,
			wrist_k(float(cell.model.cam_fovy[cid]), rgb.shape),
			believed_wrist_from_cam(cell, ex.kin.data),
			est.pos[:2],
			est.yaw,
			size,
			top_z,
			segmenter="silhouette",
		)
		return fix, reason, None

	def _ring_localise(self) -> tuple[str | None, bool]:
		"""Re-localise the perceived brick (x, y) from a ring of wrist views (:mod:`.wrist_ring`), then go back to
		ready. The jaws hang straight down over the believed brick, the tips ``RING_TIP_M`` above its top face, and the
		wrist roll turns the camera around it; each view is fitted with the silhouette segmenter and the median of the
		fixes replaces the top-view position (kept when fewer than ``RING_MIN_FIXES`` views fit or the median is more
		than ``RING_GATE_M`` from it). Yaw, height and size class stay as the top view read them. Returns a move
		failure code (or None) and whether the ring correction was accepted."""
		from .wrist_ring import RING_GATE_M, RING_MIN_FIXES, fuse_ring, ring_poses

		ex, cell, est = self.expert, self.cell, self.brick_est
		size = self._believed_brick_size()
		top_z = float(est.pos[2] + size[2] / 2)
		ex.kin.data.qpos[:] = cell.data.qpos
		poses = ring_poses(ex, cell, est.pos[:2], top_z)
		fixes, rejects = [], {}
		for k, q in enumerate(poses):
			fix, reason, error = self._wrist_view(est, q, k == 0, k + 1, len(poses))
			if error:
				return error, False
			if fix is None:
				rejects[reason] = rejects.get(reason, 0) + 1
			else:
				fixes.append(fix)
		info: dict = {"poses": len(poses), "fixes": len(fixes), "rejects": rejects, "accepted": False}
		before = float(np.linalg.norm(est.pos[:2] - cell.brick_pos()[:2]) * 1000)
		if len(fixes) >= RING_MIN_FIXES:
			xy = fuse_ring(fixes)
			step = float(np.linalg.norm(xy - est.pos[:2]))
			info["innovation_xy_mm"] = round(step * 1000, 3)
			info["spread_mm"] = round(float(np.linalg.norm(np.array([f.pos_xy for f in fixes]).std(0)) * 1000), 3)
			if step <= RING_GATE_M:
				pos = est.pos.copy()
				pos[:2] = xy
				self.brick_est = replace(est, pos=pos)
				info["accepted"] = True
		self.event(
			"ring_localised",
			**info,
			# Truth diagnostics only (registered in TRUTH_DIAGNOSTIC_KEYS); never read by the agent.
			error_xy_mm_before=round(before, 3),
			error_xy_mm=round(float(np.linalg.norm(self.brick_est.pos[:2] - cell.brick_pos()[:2]) * 1000), 3),
		)
		return self._execute(ex._go_ready()), info["accepted"]

	def can_localise(self) -> bool:
		"""The graph runs the ``localize`` node (``localise="vote"``) while the brick is on the table."""
		return self.localise == "vote" and self.held is None

	def _wrist_confirm(self, est):
		"""One wrist view over candidate ``est`` (mid wrist roll), then back to ready: ``(confirmed, info, error)``.
		Confirmed when the silhouette fit finds a brick of the believed size within ``RING_GATE_M`` of it."""
		from .wrist_ring import RING_GATE_M, ring_poses

		ex, cell = self.expert, self.cell
		top_z = float(est.pos[2] + est.size[2] / 2)
		ex.kin.data.qpos[:] = cell.data.qpos
		poses = ring_poses(ex, cell, est.pos[:2], top_z, n=1)
		if not poses:
			return False, {"reason": "NO_POSE"}, None
		fix, reason, error = self._wrist_view(est, poses[0], True, 1, 1)
		if error:
			return False, {}, error
		if error := self._execute(ex._go_ready()):
			return False, {}, error
		if fix is None:
			return False, {"reason": reason}, None
		step = float(np.linalg.norm(fix.pos_xy - est.pos[:2]))
		return step <= RING_GATE_M, {"reason": None, "innovation_xy_mm": round(step * 1000, 3)}, None

	def localise_brick(self):
		"""Graph node ``localize`` (``localise="vote"``): pick the brick among the top-view blobs, refine it with
		the wrist ring and pin the result for later perceptions. None, or a failure code.

		1. The shape rule and, when ``perception_llm`` is set, the VLM each pick a numbered candidate
		   (:mod:`.brick_vote`). Agreement (or no VLM vote) is accepted.
		2. On disagreement each pick (VLM's first) gets one wrist view; the first one confirmed is taken. When none is,
		   the shape pick stands (the VLM's when the shape rule found none).
		3. The ring (:meth:`_ring_localise`) refines the chosen brick's x, y. The blob's top-view position and the
		   ring's correction are pinned (``_brick_hint``): later top-view perceptions take the blob there and add the
		   correction, until no blob is found within ``HINT_GATE_M`` of it (the brick moved)."""
		from .brick_vote import candidates, shape_pick, vlm_pick
		from .perception import brick_blobs

		if self.perception is None:
			return None
		depth = self._top_depth()
		rgb = self.cell.rgb("top")
		bowls = list(self.bowls.values())
		cands = candidates(brick_blobs(depth, self._cam_k, self._cam_est, bowls))
		shape = shape_pick(cands)
		info: dict = {"candidates": len(cands), "shape": None if shape is None else shape + 1}
		vlm, voted = shape, False
		if self.perception_llm is not None and cands:
			vlm, vinfo = vlm_pick(self.perception_llm, rgb, cands)
			voted = "error" not in vinfo
			info |= {"vlm": None if vlm is None else vlm + 1, **{f"vlm_{k}": v for k, v in vinfo.items()}}
			if not voted:
				vlm = shape
		agree = vlm == shape
		chosen, confirms = (shape, []) if agree else (None, [])
		if not agree:
			for i in (vlm, shape):
				if i is None:
					continue
				est_i = locate_brick(depth, self._cam_k, self._cam_est, bowls, label=cands[i].label)
				if est_i is None:
					confirms.append({"candidate": i + 1, "confirmed": False, "reason": "NO_FIT"})
					continue
				ok, cinfo, error = self._wrist_confirm(est_i)
				if error:
					return error
				confirms.append({"candidate": i + 1, "confirmed": ok, **cinfo})
				if ok:
					chosen = i
					break
			if chosen is None:
				chosen = shape if shape is not None else vlm
		info |= {"agree": agree, "voted": voted, "confirms": confirms, "chosen": None if chosen is None else chosen + 1}
		est = (
			None
			if chosen is None
			else locate_brick(depth, self._cam_k, self._cam_est, bowls, label=cands[chosen].label)
		)
		if est is None:
			self.event("brick_localised", found=False, **info)
			if self.drift_alarm and not self._recalibration_attempted:
				# With no brick estimate there is no correction or height to check.
				self.event("recalibration_requested", reason="BRICK_NOT_FOUND")
				return RECALIBRATE
			return "BRICK_NOT_FOUND"
		if error := self._perceive(est=est, source="vote"):
			return error
		top_xy = self.brick_est.pos[:2].copy()
		error, ring_accepted = self._ring_localise()
		if error:
			return error
		delta = self.brick_est.pos[:2] - top_xy
		correction_mm = round(float(np.linalg.norm(delta)) * 1000, 3)
		self.event(
			"brick_localised",
			found=True,
			**info,
			correction_mm=correction_mm,
			# Truth diagnostic only (registered in TRUTH_DIAGNOSTIC_KEYS); never read by the agent.
			error_xy_mm=round(float(np.linalg.norm(self.brick_est.pos[:2] - self.cell.brick_pos()[:2]) * 1000), 3),
		)
		if self.drift_alarm:
			table_z_mm = float(est.pos[2] - (est.size[2] / 2 + RESTING_BRICK_CLEARANCE_M)) * 1000
			if top_camera_drift(correction_mm, ring_accepted, table_z_mm):
				self.event(
					"recalibration_needed",
					correction_mm=correction_mm,
					ring_accepted=ring_accepted,
					brick_height_mm=round(float(est.pos[2]) * 1000, 3),
					table_z_mm=round(table_z_mm, 3),
					correction_threshold_mm=DRIFT_CORRECTION_MM,
					table_offset_threshold_mm=DRIFT_TABLE_OFFSET_MM,
					already_attempted=self._recalibration_attempted,
				)
				if not self._recalibration_attempted:
					return RECALIBRATE
				# Keep the latest localisation usable when the alarm persists after calibration.
		self._brick_hint = {"top_xy": top_xy, "delta": delta}
		return None

	def recalibrate(self) -> str | None:
		"""Run a fresh hand-eye session on the currently mounted camera, once per episode."""
		if not self.drift_alarm or self._recalibration_attempted:
			return RECALIBRATE
		self._recalibration_attempted = True
		# The calibration rig renders the current mount from the original layout, including the calibration markers.
		# The solver estimates the pose from images and joint readings. The rig runs apart from episode physics.
		seed = int(getattr(self.layout, "defect_seed", 0))
		try:
			k, estimate, info = calibrate_layout(self._calib_layout, seed, self.perception.calibration)
		except (OSError, RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
			self.event("recalibration_failed", recalibration_s=RECALIBRATION_S, reason=str(exc))
			return RECALIBRATION_FAILED
		self._cam_k, self._cam_est = k, estimate
		self._brick_hint = self.brick_est = None
		self._perceived_size = None
		for cell in self._belief_cells.values():
			cell.close()
		self._belief_cells.clear()
		self.scene_error = None
		previous_depth = self._baseline_depth
		self._sense_bowls()
		# Both depth captures come from the same physical camera. Keep unoccluded reference pixels from prepare.
		self._baseline_depth = np.maximum(previous_depth, self._baseline_depth)
		self.event(
			"camera_recalibrated",
			recalibration_s=RECALIBRATION_S,
			poses=info["poses"],
			tries=info["tries"],
			rms_px=info["rms_px"],
			cam_pos_err_mm=info["cam_pos_err_mm"],
			cam_rot_err_deg=info["cam_rot_err_deg"],
		)
		if self.scene_error:
			return self.scene_error
		self._save([])
		return None

	def _hinted_brick(self, depth, bowls):
		"""The pinned brick (``_brick_hint``) in a fresh top depth image: the blob whose fit lies within
		``HINT_GATE_M`` of the pinned top-view position, with the ring's correction added; None (and the pin dropped)
		when there is none."""
		from .perception import brick_blobs

		hint = self._brick_hint
		best, best_d = None, HINT_GATE_M
		for blob in brick_blobs(depth, self._cam_k, self._cam_est, bowls):
			if float(np.linalg.norm(blob.xy - hint["top_xy"])) > 2 * HINT_GATE_M + 0.04:
				continue
			est = locate_brick(depth, self._cam_k, self._cam_est, bowls, label=blob.label)
			if est is None:
				continue
			d = float(np.linalg.norm(est.pos[:2] - hint["top_xy"]))
			if d <= best_d:
				best, best_d = est, d
		if best is None:
			self._brick_hint = None
			self.event("brick_hint_dropped")
			return None
		pos = best.pos.copy()
		pos[:2] += hint["delta"]
		return replace(best, pos=pos)

	def _bottom(self, views):
		self.bottom_count += 1
		if self.held is None and (error := self._grasp()):
			return error
		ex = self.expert
		with self._planning_view() as vis:
			candidate = plan_bottom_view(
				vis,
				ex.last,
				gripper=ex.p.hold_g,
				seed=self.recoveries * 71 + self.bottom_count - 1,
				clearance=lambda q: self._straight_clear(ex.last, q, held=True),
				border_px=BOTTOM_BORDER_PX if self.realistic else 0,
				brick_size=self._plan_brick_size(),
			)
		if candidate is None or candidate.faces_px[5] < BOTTOM_MIN_PX:
			self.event("view_unavailable", camera="top", reason="NO_BOTTOM_PATH")
			return None

		def look():
			pose = candidate.q.copy()
			yield from ex._carry(ex._move(pose, ex._carry_secs(pose, ex.p.show_move_s), "show_move"))
			yield from ex._carry(ex._hold_servo(pose, ex.rp.bottom_hold_s, "show_hold"))

		if error := self._execute(look()):
			return error
		if ex.lost:
			return "BRICK_LOST"
		views.append(self._capture("top"))
		return None

	# ---- re-view of alarms (strategies B and C; see .review) ----

	def _review(self, previous, views, trigger):
		"""Re-view the oldest unresolved alarm until none is left or the budget is spent; appends to ``views``."""
		while True:
			pending, possible = self.review_pending([*previous, *views])
			if not possible:
				return None
			target = next(p for p in pending if p["alarm"] not in self.review_exhausted)
			if error := self._review_once(target, previous, views, trigger):
				return error

	def _rank_candidates(self, pool, faces, prior_q, *, held, min_px, clears=None):
		"""Public candidate records (best first, at most MAX_CANDIDATES) and their paths; reachable ones only.

		Ranked by ``priority_px``: each target face's predicted pixels, capped at REVIEW_FACE_PX_CAP so one large
		face (usually the top) cannot outweigh the others, and divided by 1 + the face's clear views so far, so the
		faces with the least clear evidence lead (``clears``: face -> clear scored views)."""
		ex = self.expert
		clears = clears or {}
		rows = []
		for source, cand, path in pool:
			seen = [f for f in faces if cand.faces_px[f] >= min_px]
			target_px = int(sum(cand.faces_px[f] for f in seen))
			if target_px <= 0:
				continue
			priority = sum(min(float(cand.faces_px[f]), REVIEW_FACE_PX_CAP) / (1 + clears.get(f, 0)) for f in seen)
			separation = min((_pose(cand.q, q) for q in prior_q if q is not None), default=math.pi)
			if separation < MIN_POSE_SEPARATION_RAD:
				continue
			path_rad = float(sum(_pose(a, b) for a, b in itertools.pairwise([ex.last, *path])))
			rows.append((source, cand, path, target_px, separation, path_rad, int(priority)))
		rows.sort(key=lambda r: (-round(r[6] / 400.0), -round(min(r[4], 1.5), 1), r[5]))
		chosen = []
		for row in rows:
			cand, path = row[1], row[2]
			if any(_pose(cand.q, c[1].q) < 0.05 for c in chosen):
				continue
			if not all(self._straight_clear(a, b, held=held) for a, b in itertools.pairwise([ex.last, *path])):
				continue
			chosen.append(row)
			if len(chosen) == MAX_CANDIDATES:
				break
		public = [
			{
				"id": f"c{i}",
				"camera": cand.camera,
				"planner": source,
				"target_faces": [f for f in faces if cand.faces_px[f] >= min_px],
				"target_px": target_px,
				"face_px": {str(f): int(cand.faces_px[f]) for f in faces if cand.faces_px[f] >= min_px},
				"priority_px": priority,
				"brick_px": int(cand.brick_px),
				"separation_rad": round(separation, 3),
				"path_rad": round(path_rad, 3),
			}
			for i, (source, cand, path, target_px, separation, path_rad, priority) in enumerate(chosen)
		]
		return public, {f"c{i}": (row[1], row[2]) for i, row in enumerate(chosen)}

	def _wrist_review_candidates(self, faces, prior_q, clears=None):
		ex, cell = self.expert, self.cell
		with self._belief():
			ex.kin.data.qpos[:] = cell.data.qpos
		pool = []
		for k, params in enumerate(ORBIT_ATTEMPTS):
			with self._planning_view() as vis:
				plan = plan_orbit_scan(
					vis,
					ex.kin,
					ex.last,
					gripper=ex.p.open_g,
					params=params,
					faces=faces,
					min_px=WRIST_MIN_PX,
					brick_size=self._plan_brick_size(),
				)
			pending: list[np.ndarray] = []
			for cand, via in zip(plan.views, plan.via, strict=True):
				path = [*pending, *via, cand.q]
				pending = path if via else []
				pool.append((f"orbit{k}", cand, path))
		for k, mode in enumerate(TABLE_SCAN_MODES):
			with self._planning_view() as vis:
				cands, _ = plan_table_scan(vis, ex.kin, ex.last, gripper=ex.p.open_g, min_px=WRIST_MIN_PX, **mode)
			pool += [(f"hover{k}", c, [c.q]) for c in cands]
		return self._rank_candidates(pool, faces, prior_q, held=False, min_px=WRIST_MIN_PX, clears=clears)

	def _bottom_review_candidates(self, prior_q):
		ex = self.expert
		pool = []
		for k in range(6):
			with self._planning_view() as vis:
				cand = plan_bottom_view(
					vis,
					ex.last,
					gripper=ex.p.hold_g,
					seed=10_007 + 97 * k + 13 * self.review_count,
					clearance=lambda q: self._straight_clear(ex.last, q, held=True),
					border_px=BOTTOM_BORDER_PX if self.realistic else 0,
					brick_size=self._plan_brick_size(),
				)
			if cand is not None:
				pool.append((f"bottom{k}", cand, [cand.q]))
		return self._rank_candidates(pool, [5], prior_q, held=True, min_px=BOTTOM_MIN_PX)

	def _llm_images(self, target, views, camera):
		"""Images shown to the LLM chooser, one per request: the alarm crop, one earlier crop of the same faces, the
		current frame of the re-view camera, the front camera and (realistic) a top depth preview. Each is saved
		under ``llm_images/`` and logged as an ``llm_image`` event; ``cached`` marks an image shown before."""
		items = []
		if target["alarm"] in self._crops:
			items.append((f"alarm crop {target['alarm']}", target["alarm"], self._crops[target["alarm"]]))
		for v in reversed(views):
			if v["id"] != target["alarm"] and v["id"] in self._crops and set(v["faces"]) & set(target["faces"]):
				items.append((f"earlier crop {v['id']}", v["id"], self._crops[v["id"]]))
				break
		frame = self.ep.n_frames
		items.append((f"{camera} now", f"{camera}@{self.branch}:{frame}", self.cell.rgb(camera)))
		items.append(("front now", f"front@{self.branch}:{frame}", self.cell.rgb("front")))
		if self.realistic:
			depth = self._top_depth()
			lo, hi = np.nanpercentile(depth, 2), np.nanpercentile(depth, 98)
			gray = (255 * np.clip((hi - np.nan_to_num(depth, nan=hi)) / max(hi - lo, 1e-6), 0, 1)).astype(np.uint8)
			items.append(
				("top depth preview, bright is near", f"depth@{self.branch}:{frame}", np.stack([gray] * 3, -1))
			)
		from PIL import Image

		directory = self.record.path / "llm_images"
		directory.mkdir(exist_ok=True)
		out = []
		for name, key, image in items:
			cached = key in self._llm_sent
			self._llm_sent.add(key)
			file = directory / f"{self.review_count:02d}-{len(out):02d}.png"
			Image.fromarray(np.asarray(image, dtype=np.uint8)).save(file)
			self.event("llm_image", name=name, key=key, cached=cached, file=f"llm_images/{file.name}")
			out.append({"name": name, "image": image, "cached": cached})
		return out

	def _review_once(self, target, previous, views, trigger):
		ex = self.expert
		faces = list(target["faces"])
		bottom = faces == [5] or target["camera"] == "top"
		prior_q = [v.get("q") for v in [*previous, *views] if set(v.get("faces") or []) & set(faces)]
		clears = {
			f: sum(
				f in (v.get("faces") or []) and v.get("score") is not None and v["score"] < v["threshold"]
				for v in [*previous, *views]
			)
			for f in faces
		}
		if bottom:
			if self.held is None and (error := self._grasp()):
				return error
			public, paths = self._bottom_review_candidates(prior_q)
		else:
			if self.held is not None and (error := self._park_held()):
				return error
			# Re-locate only when the estimate is gone (after a park): mid-scan the arm hides the brick from the top
			# camera, and the untouched brick is where it was last perceived.
			if self.brick_est is None and (error := self._perceive()):
				return error
			public, paths = self._wrist_review_candidates([f for f in faces if f < 5], prior_q, clears)
		if not public:
			self.review_exhausted.add(target["alarm"])
			self.event("review_unavailable", alarm=target["alarm"], trigger=trigger, faces=faces)
			return None
		info, source = {}, "rule"
		if self.strategy == "B":
			choice = rule_choice(public)
		elif self.review_llm is None:
			choice, source, info = rule_choice(public), "fallback", {"reason": "no_llm"}
		else:
			camera = "top" if bottom else "wrist_cam"
			context = {
				"alarm": {"view": target["alarm"], "camera": target["camera"], "faces": faces},
				"reviews_left": self.max_review_views - self.review_count,
				"scored_views_left": None if self.max_scored_views is None else self._views_left(),
			}
			assert_no_defect_leakage(context)
			images = self._llm_images(target, [*previous, *views], camera)
			choice, source, info = llm_choice(self.review_llm, images, public, context)
		self.event(
			"review_decision",
			trigger=trigger,
			alarm=target["alarm"],
			faces=faces,
			candidates=public,
			choice=choice,
			source=source,
			fallback_reason=info.get("reason"),
			latency_s=info.get("latency_s"),
			requests=info.get("requests"),
			prompt_tokens=info.get("prompt_tokens"),
			cached_tokens=info.get("cached_tokens"),
			notes=info.get("notes"),
		)
		cand, path = paths[choice]
		self.review_count += 1
		if bottom:

			def review_bottom(pose):
				yield from ex._carry(ex._move(pose, ex._carry_secs(pose, ex.p.show_move_s), "show_move"))
				yield from ex._carry(ex._hold_servo(pose, ex.rp.bottom_hold_s, "show_hold"))

			if error := self._execute(review_bottom(cand.q.copy())):
				return error
			if ex.lost:
				return "BRICK_LOST"
			view = self._capture("top")
		else:

			def review_wrist(path):
				yield from sweep(ex, path, self.scan_peak_speed, ex.rp.scan_move_s, pause_s=None)
				yield from ex._hold_servo(path[-1], ex.rp.scan_hold_s, "approach")

			if error := self._execute(review_wrist(path)):
				return error
			view = self._capture("wrist_cam")
		view["review_of"] = target["alarm"]
		self.event("review_view", id=view["id"], alarm=target["alarm"], candidate=choice)
		views.append(view)
		return None

	def inspect(self, node, previous):
		views = []
		try:
			if node == "review":
				error = self._review(previous, views, "before_place")
				if not error and self.held is None:
					error = self._execute(self.expert._go_ready())
				if not error:
					self._save([*previous, *views])
			else:
				error = self._wrist(previous, views) if node == "scan_wrist" else self._bottom(views)
		except RuntimeError as exc:
			# The scoring process can stop independently of simulation. Keep the physical state and trace.
			from ..inspection_scorer import ScoreServerError

			if not isinstance(exc, ScoreServerError):
				raise
			error = "DETECTOR_UNAVAILABLE"
		return views, error

	def failure(self, reason, views):
		self.execution_error = reason
		failed_node = next(
			(e["node"] for e in reversed(self.record.events) if e["kind"] == "node" and e["node"] != "recover"), None
		)

		payload = {
			"code": reason,
			"skill": failed_node,
			"frame": self.ep.n_frames,
			"branch": self.branch,
			"prior_views": [{"id": v["id"], "faces": v["faces"], "score": v["score"]} for v in views],
		}
		if reason == "PHYSICAL_HEALTH_FAILED":
			payload["reason_codes"] = next(
				event["reason_codes"]
				for event in reversed(self.record.events)
				if event["kind"] == "motion_health" and event["branch"] == self.branch
			)
		self.event("failure", **{k: v for k, v in payload.items() if k not in ("frame", "branch")})
		return payload

	def restore_state(self, checkpoint, params=None, relocate=None):
		"""Physics, command filter, a rebuilt expert, the held-brick and sensing state and the view counters at a
		skill-boundary checkpoint (brick on the table or in the gripper). Shared by in-process ``restore_checkpoint``
		and cross-process ``twin_resume.restore``. The caller truncates the telemetry the next observation reads."""
		m, d = self.cell.model, self.cell.data
		mujoco.mj_setState(m, d, checkpoint.integration, STATE_SPEC)
		mujoco.mj_forward(m, d)
		mujoco.mj_setState(m, d, checkpoint.integration, STATE_SPEC)
		restore_engine = getattr(self.cell, "engine_restore", None)
		if restore_engine is not None:
			token = (checkpoint.agent or {}).get("engine_snapshot")
			if token is None:
				raise RuntimeError(f"{self.engine} cannot restore {checkpoint.id} natively")
			restore_engine(token)  # the engine's own state, then mirrored back into the MuJoCo model
		filt = self.ep.filt
		filt._queue = deque((np.array(q) for q in checkpoint.filter["queue"]), maxlen=filt._queue.maxlen)
		filt._prev_out = np.array(checkpoint.filter["prev"])
		filt._backlash_pos = np.array(checkpoint.filter["backlash"])
		params = self.expert.p if params is None else params
		relocate = self.expert.rp if relocate is None else relocate
		self._clipped_frames += getattr(self.expert, "clipped_frames", 0)
		self.expert = ex = self._make_expert(params, relocate)
		ex.last = np.array(checkpoint.expert["last"])
		ex._active_open_g = checkpoint.expert["active_open"]
		ex._frame_idx = checkpoint.expert["frame_idx"]
		ex.gripper_monitor._prev_meas = checkpoint.expert["monitor_position"]
		ex.gripper_monitor._run = checkpoint.expert["monitor_count"]
		if checkpoint.expert.get("prev_q") is not None:
			ex.obs = {"q": np.array(checkpoint.expert["prev_q"])}  # read by act() as the previous joints only
		ex._no_dual = checkpoint.expert.get("no_dual", 0)
		ex._last_block_dir = checkpoint.expert.get("last_block_dir", 0)
		held = checkpoint.held
		self.held = None if held is None else (np.array(held["q_lift"]), float(held["roll"]))
		agent = checkpoint.agent or {}
		grip = agent.get("grip_from_brick")
		self._grip_from_brick = None if grip is None else np.array(grip)
		self._expected_yaw = agent.get("expected_yaw")
		self._carry_epoch_start = int(agent.get("carry_epoch_start", 0))
		parked = agent.get("parked_pose")
		self._parked_pose = None if parked is None else (np.array(parked["xy"], dtype=np.float64), parked["yaw"])
		if agent.get("park_xy") is not None:
			self._park_xy = np.array(agent["park_xy"])
		if agent.get("depth_rng") is not None and self.perception is not None:
			self._depth_rng.bit_generator.state = copy.deepcopy(agent["depth_rng"])
		if agent.get("top_depth_rng") is not None and self.realistic:
			self._top_depth_rng.bit_generator.state = copy.deepcopy(agent["top_depth_rng"])
		if agent.get("bowls") is not None:
			for name, xy in agent["bowls"].items():
				self.bowls[name] = np.array(xy, dtype=np.float64)  # in place: the clearance proxy reads this mapping
		if agent.get("bowl_estimates") is not None:
			self._bowl_estimates = {
				name: None
				if raw is None
				else BowlEstimate(
					np.array(raw["xy"], dtype=np.float64),
					float(raw["radius_m"]),
					int(raw["pixels"]),
					int(raw["rim_points"]),
					float(raw.get("fit_residual_m", 0.0)),
				)
				for name, raw in agent["bowl_estimates"].items()
			}
		if self.drift_alarm:
			self._recalibration_attempted = bool(agent.get("recalibration_attempted", False))
		if agent.get("camera") is not None and self.perception is not None:
			self._cam_k = np.array(agent["camera"]["k"], dtype=np.float64)
			self._cam_est = np.array(agent["camera"]["world_from_cam"], dtype=np.float64)
		if agent.get("oracle") is not None:
			self.oracle = frozenset(agent["oracle"]) if self.realistic else frozenset()
			self._oracle_logged = bool(agent.get("oracle_logged", False))
		if agent.get("motion") is not None:
			self.motion = str(agent["motion"])
		if agent.get("fuse_cfg") is not None:
			self.fuse_cfg = FuseCfg(**agent["fuse_cfg"])
		if agent.get("belief_cameras") is not None:
			self.belief_cameras = str(agent["belief_cameras"])
		if agent.get("evidence") is not None:
			self.evidence = str(agent["evidence"])
		if agent.get("segmenter") is not None:
			self.segmenter = str(agent["segmenter"])
		if agent.get("segmenter_onnx") is not None:
			self.segmenter_onnx = Path(agent["segmenter_onnx"])
		self._configure_recheck()
		if self.realistic:
			self._contact.reset()
			self._contact.prev = agent.get("contact_prev")
		if self.perception is not None:
			self.brick_est = brick_from_json(agent.get("brick_est"))
			hint = agent.get("brick_hint")
			self._brick_hint = None if hint is None else {k: np.asarray(v, dtype=np.float64) for k, v in hint.items()}
			size = agent.get("perceived_size")
			self._perceived_size = None if size is None else tuple(size)
		self.actual_destination = None
		self.execution_error = None
		self.wrist_count = checkpoint.wrist_count
		self.bottom_count = checkpoint.bottom_count

	def resume_state(self, checkpoint, checkpoints, history=None, empty_bowl_depth=None):
		"""Cross-process restore (``twin_resume.restore``): put this freshly built twin of the same case at a persisted
		``checkpoint`` so that it continues exactly as the interrupted episode would have. Besides ``restore_state``:
		the expert parameters and variant bookkeeping in force at the checkpoint, the per-frame history (telemetry,
		targets, phases: the next observation reads the last row, the final health check reads them all), the
		empty-bowl depth reference and the segment rows. ``checkpoints`` become the live rewind targets."""
		agent = checkpoint.agent or {}
		params = relocate = None
		if agent.get("expert_params") is not None:
			params = params_from_json(ExpertParams, agent["expert_params"])
			relocate = params_from_json(RelocateParams, agent["relocate_params"])
		self.restore_state(checkpoint, params, relocate)
		if agent.get("base_params") is not None:
			raw = agent["base_params"]
			self._base_params = (params_from_json(ExpertParams, raw[0]), params_from_json(RelocateParams, raw[1]))
		if agent.get("scan_peak_speed") is not None:
			self.scan_peak_speed = agent["scan_peak_speed"]
		self.recovery_variants = tuple(agent.get("recovery_variants", ()))
		self._clipped_frames = agent.get("clipped_frames", 0)
		counters = agent.get("view_counters") or {}
		self.scored_count = int(counters.get("scored", 0))
		self.review_count = int(counters.get("reviews", 0))
		self.review_exhausted = set(counters.get("review_exhausted", ()))
		if agent.get("llm_stats") is not None:
			self._llm_stats = copy.deepcopy(agent["llm_stats"])
		if history is not None:
			ep = self.ep
			for key in ep._logs:
				ep._logs[key] = list(history[f"tel_{key}"])
			ep.jaw_env_force = [float(v) for v in history["jaw_env_force"]]
			self.phases = [int(v) for v in history["phases"]]
			self.targets = [row.copy() for row in history["targets"]]
		if empty_bowl_depth is not None:
			self._baseline_depth = np.array(empty_bowl_depth, dtype=np.float64)
		self._restore_extra(agent.get("segments"))
		self.checkpoints = list(checkpoints)

	def recovery_choices(self, policy: Policy, failures) -> list[tuple[str, str]]:
		"""``(choice, note)`` pairs the recover node offers after ``failures[-1]``, the default first and ``stop``
		last; empty when the budget leaves no recovery. ``truth`` rewinds the physics to a skill-boundary checkpoint;
		``realistic`` recovers from the current state (see :meth:`_retry`)."""
		stop = ("stop", "give up and report the failure")
		if self.recoveries >= policy.max_rewinds or self.total_frames >= MAX_TOTAL_FRAMES:
			return []
		failure = failures[-1]
		if self.realistic:
			if failure.get("code") in ("BOWL_NOT_FOUND", "BRICK_NOT_FOUND", "DETECTOR_UNAVAILABLE"):
				return []
			recommended = VARIANT_BY_CODE.get(failure.get("code"))
			variants = [v for v in dict.fromkeys((recommended, "cautious")) if v]
			note = "recover from the current state and retry with"
			options = [(f"retry+{v}", f"{note} {VARIANT_NOTES[v]}") for v in variants]
			return [*options, ("retry", f"{note} unchanged parameters"), stop]
		checkpoints = self.checkpoints
		if self.engine != "mujoco":  # only checkpoints the other engine can restore natively
			checkpoints = [c for c in checkpoints if (c.agent or {}).get("engine_snapshot") is not None]
		if not checkpoints:
			return []
		stall = stall_start(self.ep.telemetry()["actuator_torques"], self.spec.arm_actuator.effort_limit)
		stall_node = None if stall is None else node_at(self.record.events, stall)
		return [*rewind_options(checkpoints, failure, stall, stall_node), stop]

	def rewind(self, policy: Policy, failures):
		if self.realistic:
			return self._retry(policy, failures)
		self.policy = policy
		options = self.recovery_choices(policy, failures)
		if not options:
			return None
		choice = self.choose([c for c, _ in options], self.checkpoints[-1].views, failures, notes=dict(options))
		if choice == "stop":
			return None
		checkpoint_id, _, variant = choice.partition("+")
		checkpoint = next(c for c in self.checkpoints if c.id == checkpoint_id)
		return self.restore_checkpoint(checkpoint, variant, reason=failures[-1]["code"])

	def restore_checkpoint(self, checkpoint, variant="", *, reason=None, count_as_rewind: bool = True):
		"""Physics restore to ``checkpoint`` on a new branch: drop the later telemetry, rebuild the expert (with
		``variant`` if given) and restore the checkpoint's state, also mid-carry. Returns its views.

		With ``count_as_rewind=True`` (default), increments ``self.rewinds`` (consumed against the recovery budget).
		With ``count_as_rewind=False`` (S5 perturbation branch fork), increments ``self.perturbation_rewinds`` instead
		so ``self.rewinds`` and the online recovery budget (``self.recoveries``) remain intact."""
		if count_as_rewind:
			self.rewinds += 1
		else:
			self.perturbation_rewinds += 1
		self.branch = self.rewinds + self.perturbation_rewinds
		self.record.fork(self.branch, checkpoint)
		for rows in self.ep._logs.values():
			del rows[checkpoint.frame :]
		del self.ep.jaw_env_force[checkpoint.frame :]
		del self.phases[checkpoint.frame :]
		del self.targets[checkpoint.frame :]
		params, relocate = self.expert.p, self.expert.rp
		if not count_as_rewind:
			self.recovery_variants = tuple((checkpoint.agent or {}).get("recovery_variants", ()))
		if variant:
			# Variants transform the first attempt's defaults, so they never stack on an earlier variant.
			self._base_params = getattr(self, "_base_params", (params, relocate))
			params, relocate, self.scan_peak_speed = apply_variant(variant, *self._base_params)
		if count_as_rewind or variant:
			self.recovery_variants = (*self.recovery_variants, variant or "none")
		self.restore_state(checkpoint, params, relocate)
		if not count_as_rewind:  # S5 forks only; the rewind path stays exactly as before
			self.retries = checkpoint.retries
			self._restore_extra((checkpoint.agent or {}).get("segments"))
		del self.checkpoints[self.checkpoints.index(checkpoint) + 1 :]
		self.record.restored(self.cell)
		self.event(
			"rewound" if count_as_rewind else "perturbation_branch",
			parent_branch=checkpoint.branch,
			checkpoint=checkpoint.id,
			reason=reason,
			variant=variant or None,
		)
		return copy.deepcopy(checkpoint.views)

	def _retry(self, policy: Policy, failures):
		"""Realistic recovery: no physics rewind. Put a carried brick down (or let go of it), go to the ready pose,
		locate the brick again and carry on with a bounded variant. The views so far still count only when the brick
		is found where it was believed at the last skill boundary; otherwise its face numbering is lost."""
		self.policy = policy
		failure = failures[-1]
		options = self.recovery_choices(policy, failures)
		if not options:
			return None
		last = self.checkpoints[-1] if self.checkpoints else None
		kept = copy.deepcopy(last.views) if last else []
		choice = self.choose([c for c, _ in options], kept, failures, notes=dict(options))
		if choice == "stop":
			return None
		_, _, variant = choice.partition("+")
		self.retries += 1  # not a rewind: the physical history stays one continuous branch
		params, relocate = self.expert.p, self.expert.rp
		if variant:
			self._base_params = getattr(self, "_base_params", (params, relocate))
			params, relocate, self.scan_peak_speed = apply_variant(variant, *self._base_params)
		self.recovery_variants = (*self.recovery_variants, variant or "none")
		self.event("retry", reason=failure.get("code"), variant=variant or None)
		lost = self.expert.lost
		self._clipped_frames += getattr(self.expert, "clipped_frames", 0)
		self.expert = ex = self._make_expert(params, relocate)
		ex.last = self.cell.joints().copy()
		self._contact.reset()
		error = None
		if self.held is not None and not lost:
			error = self._park_held()
		if self.held is not None or self._grip_from_brick is not None or error:

			def let_go():  # open where the gripper is, then rise
				q = ex.last.copy()
				q[5] = ex.p.open_g
				yield from ex._move(q, ex.p.close_s, "place")
				up = ex._ik(self.cell.tcp_pos() + np.array([0.0, 0.0, 0.06]), q, float(q[4]))
				up[5] = ex.p.open_g
				yield from ex._move(up, ex.p.descend_s, "place")

			error = self._execute(let_go())
		self.held = None
		self._grip_from_brick = None
		self.brick_est = None
		if error or self._execute(ex._go_ready()):
			return None
		believed = brick_from_json((last.agent or {}).get("brick_est")) if last else None
		if self._perceive():
			return None
		moved = believed is None or bool(
			np.linalg.norm(self.brick_est.pos[:2] - believed.pos[:2]) > RETRY_MOVED_M
			or yaw_error_deg(self.brick_est.yaw, believed.yaw, self.brick_est.size) > RETRY_MOVED_DEG
			or self.brick_est.size != believed.size
		)
		views = [] if moved else kept
		self.event("retried", brick_moved=moved, views_kept=len(views))
		self._carry_epoch_start = self.ep.n_frames
		self._save(views)
		return views

	def _bowl_slots(self) -> tuple[str, ...]:
		"""All bowl slots in the layout: ``green``, ``pink``, and any ``extra:<i>``."""
		extra = getattr(self.layout, "extra_bowls", ()) or ()
		return ("green", "pink", *(f"extra:{i}" for i in range(len(extra))))

	def bowl_colour(self, slot: str) -> str:
		"""Realized colour word for bowl ``slot`` (``green``, ``pink``, or ``extra:<i>``)."""
		if slot.startswith("extra:"):
			idx = int(slot.split(":", 1)[1])
			extra = getattr(self.layout, "extra_bowls", ()) or ()
			if not 0 <= idx < len(extra):
				raise KeyError(slot)
			return str(extra[idx][0])
		names = getattr(self.layout, "bowl_names", None)
		if names and slot in names:
			return str(names[slot])
		if slot in ("green", "pink"):
			return slot
		raise KeyError(slot)

	def _true_bowl_xy(self, slot: str) -> tuple[float, float]:
		"""Ground-truth (x, y) centre for bowl ``slot`` (``green``, ``pink``, or ``extra:<i>``)."""
		if slot == "green":
			return (float(self.layout.green[0]), float(self.layout.green[1]))
		if slot == "pink":
			return (float(self.layout.pink[0]), float(self.layout.pink[1]))
		if slot.startswith("extra:"):
			idx = int(slot.split(":", 1)[1])
			extra = getattr(self.layout, "extra_bowls", ()) or ()
			if not 0 <= idx < len(extra):
				raise KeyError(slot)
			_, x, y, _ = extra[idx]
			return (float(x), float(y))
		raise KeyError(slot)

	def target_bowl(self, label: str) -> str:
		"""The bowl a brick with ``label`` goes to. Subclasses may remap it (vla/paired_carry.TargetSwapTracer); the
		placement check and the final evaluation then judge the episode against the remapped bowl."""
		return "pink" if label == "defective" else "green"

	def place(self, label):
		if label == "unknown":
			self.event("abstain", destination="inspection_table")
			error = self._park_held() if self.held is not None else None
			if error:
				return error
			if self.realistic:  # the brick must be seen back on the table
				if self.brick_est is None and self._perceive():
					return "INSPECTION_TABLE_PARK_UNVERIFIED"
				self.actual_destination = "inspection_table"
				return None
			position = self.cell.brick_pos()
			if (
				np.linalg.norm(position[:2] - self._park_xy) > 0.03
				or position[2] > self.layout.brick_size[2] / 2 + 0.01
			):
				return "INSPECTION_TABLE_PARK_UNVERIFIED"
			self.actual_destination = "inspection_table"
			return None
		if self.held is None and (error := self._grasp()):
			return error
		ex = self.expert
		q_lift, _ = self.held
		ex.bowl_xy = np.array(self.bowls[self.target_bowl(label)])
		if not self._straight_clear(ex.last, q_lift, held=True):
			return "PLACE_RETURN_PATH_BLOCKED"

		def release():
			if np.max(np.abs(ex.last - q_lift)) > 0.01:
				yield from ex._carry(ex._move(q_lift, ex._carry_secs(q_lift, ex.p.show_move_s), "show_move"))
			if not ex.lost:
				yield from ex._place(q_lift)
			if not ex.lost:
				yield from ex._go_home()

		error = self._execute(release())
		if error or ex.lost:
			return error or "BRICK_LOST"
		self.held = None
		position = self.cell.brick_pos()
		for name in self._bowl_slots():
			xy = self._true_bowl_xy(name)
			if np.linalg.norm(position[:2] - xy) < BOWL_R and position[2] < 0.055:
				self.actual_destination = name  # ground truth, read only by evaluate in realistic
				break
		target = self.target_bowl(label)
		if self.realistic:
			self._grip_from_brick = None
			self.brick_est = None
			depth = self._top_depth()
			fill = {
				n: bowl_fill(depth, self._baseline_depth, self._cam_k, self._cam_est, xy)
				for n, xy in self.bowls.items()
			}
			self.event("placement_check", fill_pixels=fill)
			if not bowl_filled(fill[target]):
				return "PLACEMENT_UNVERIFIED"
			return None  # the motion-health check is part of the final evaluation
		if self.actual_destination != target:
			return "PLACEMENT_UNVERIFIED"
		health = self._health(ex.bowl_xy)
		if health["verdict"] != "SUCCESS":
			self.event("motion_health", reason_codes=health["reason_codes"])
			return "PHYSICAL_HEALTH_FAILED"
		return None

	def evaluate(self, label, views, failures):
		# Ground truth enters only after the graph has selected and executed its final action.
		expected = "defective" if self.layout.defective else "normal"
		true_bowl = self.target_bowl(expected)
		target = self._true_bowl_xy(true_bowl)
		health = self._health(target) if self.ep.n_frames and label != "unknown" else {}
		ok = label == expected and self.actual_destination == true_bowl
		v3 = getattr(self.layout, "_scene_v3", None)
		result = {
			"verdict": "SUCCESS"
			if ok and health.get("verdict") == "SUCCESS"
			else "UNKNOWN"
			if label == "unknown" and not self.execution_error
			else "FAILED",
			**self._final_cycle_verdict(ok, health),
			"label": label,
			"label_correct": label == expected,
			"destination": self.actual_destination,
			"active_branch": self.branch,
			"rewinds": self.rewinds,
			**({"perturbation_rewinds": self.perturbation_rewinds} if self.perturbation_rewinds else {}),
			"retries": self.retries,
			"recovery_variants": list(self.recovery_variants),
			"failures": failures,
			"active_views": len(views),
			"views": sum(e["kind"] == "view" for e in self.record.events),
			"sim_frames": self.total_frames,
			"active_frames": self.ep.n_frames,
			"evidence": [v["id"] for v in views],
			"cleared_faces": sorted(self._decide(views)[1]),
			"strategy": self.strategy,
			"scored_views": self.scored_count,
			"review_views": self.review_count,
			"motion": getattr(self, "motion", "open"),
			"health": health,
			"llm": self._llm_summary(),
			"smoothness": self._smoothness(),
			# In-place retries keep one continuous physical branch (a recovery demonstration); rewinds do not.
			"student_eligible": bool(
				ok and health.get("verdict") == "SUCCESS" and not self.rewinds and not getattr(self, "oracle", None)
			),
			"placement_evidence": next(
				(v["id"] for v in views if v["score"] is not None and v["score"] >= v["threshold"]), None
			)
			if label == "defective"
			else None,
			# A paired-target demonstration put the brick in the other bowl on purpose (vla/paired_carry.py).
			**({"target_swap": True} if getattr(self, "target_swap", False) else {}),
			**({"scene": "v3", "variant": int(v3.variant)} if v3 is not None else {}),
		}
		self.record.audit(
			{"true_label": expected, "true_face": self.layout.defect_face if self.layout.defective else None}
		)
		return result

	@staticmethod
	def _final_cycle_verdict(ok: bool, health: dict) -> dict:
		"""``verdict_final_cycle``: the episode verdict if only the last pick-to-place cycle counted. Present only on
		unperturbed runs that parked or retried (see ``_health``); the ``verdict`` field itself is unchanged."""
		cycle = health.get("final_cycle")
		if not cycle:
			return {}
		return {"verdict_final_cycle": "SUCCESS" if ok and cycle.get("verdict") == "SUCCESS" else "FAILED"}

	def _smoothness(self):
		n = min(len(self.targets), self.ep.n_frames)
		if n < 4:
			return {"frames": n}
		qpos = self.ep.telemetry()["qpos"][:n]
		out = motion_smoothness(np.asarray(self.targets[:n]), qpos, np.asarray(self.phases[:n]))
		out["clipped_frames"] = int(self._clipped_frames + getattr(self.expert, "clipped_frames", 0))
		return out

	def _health(self, target):
		tel = self.ep.telemetry()
		limits = np.array([self.spec.arm_actuator.effort_limit] * 5 + [self.spec.gripper_actuator.effort_limit])
		full = evaluate_episode_health(
			q_cmd=tel["q_cmd"],
			q_act=tel["qpos"],
			tcp_pos=tel["tcp_pos"],
			brick_pos=tel["brick_pos"],
			brick_quat=tel["brick_quat"],
			jaw_forces=tel["jaw_forces"],
			unintended_forces=tel["unintended_forces"],
			phases=np.asarray(self.phases),
			target_bowl_xy=target,
			bowl_r=BOWL_R,
			brick_vel=tel["brick_vel"],
			brick_ang_vel=tel["brick_ang_vel"],
			ee_quat=tel["ee_quat"],
			fps=FPS,
			brick_size=self.layout.brick_size,
			substep_unintended_forces=tel["substep_unintended"],
			pinch_normal_forces=tel["pinch_normal"],
			pinch_tangential_forces=tel["pinch_tangential"],
			actuator_torques=tel["actuator_torques"],
			torque_limits=limits,
			torque_source="solver_measured",
			timestamps=tel["t"],
		).to_dict()
		start = int(getattr(self, "_carry_epoch_start", 0))
		if start <= 0 or start >= len(self.phases):
			return full
		# After an in-place retry or table park-and-rescan, evaluate grasp/slip/settle on the final pick-to-place
		# cycle [start, N) while still enforcing collisions, stalls, and lag across the full episode [0, N).
		# On S5 perturbation runs that is the verdict. On unperturbed runs the whole-episode ``full`` verdict stays the
		# verdict (old results keep it exactly) and the final-cycle one is added beside it as ``final_cycle``: the table
		# release of a park-and-rescan is not a bowl release, so ``full`` can fail a brick that ended in the bowl.
		sub = evaluate_episode_health(
			q_cmd=tel["q_cmd"][start:],
			q_act=tel["qpos"][start:],
			tcp_pos=tel["tcp_pos"][start:],
			brick_pos=tel["brick_pos"][start:],
			brick_quat=tel["brick_quat"][start:],
			jaw_forces=tel["jaw_forces"][start:],
			unintended_forces=tel["unintended_forces"][start:],
			phases=np.asarray(self.phases[start:]),
			target_bowl_xy=target,
			bowl_r=BOWL_R,
			brick_vel=tel["brick_vel"][start:],
			brick_ang_vel=tel["brick_ang_vel"][start:],
			ee_quat=tel["ee_quat"][start:],
			fps=FPS,
			brick_size=self.layout.brick_size,
			substep_unintended_forces=tel["substep_unintended"][start:],
			pinch_normal_forces=tel["pinch_normal"][start:],
			pinch_tangential_forces=tel["pinch_tangential"][start:],
			actuator_torques=tel["actuator_torques"][start:],
			torque_limits=limits,
			torque_source="solver_measured",
			timestamps=tel["t"][start:],
		).to_dict()
		global_reasons = [
			r for r in full["reason_codes"] if r in ("unintended_collision", "torque_stall", "tracking_lag")
		]
		if global_reasons:
			reasons = list(dict.fromkeys([*sub["reason_codes"], *global_reasons]))
			sub["reason_codes"] = reasons
			sub["process_ok"] = False
			sub["verdict"] = "FAILURE"
			sub["success"] = False
			sub["failure_mode"] = reasons[0]
		sub["unintended_collision_count"] = full["unintended_collision_count"]
		sub["max_unintended_force_N"] = full["max_unintended_force_N"]
		sub["unintended_collision_impulse_Ns"] = full["unintended_collision_impulse_Ns"]
		sub["max_stall_duration_s"] = full["max_stall_duration_s"]
		sub["max_tracking_lag_duration_s"] = full["max_tracking_lag_duration_s"]
		sub["max_tracking_error_rad"] = full["max_tracking_error_rad"]
		if self._s5_active():
			return sub
		return {**full, "final_cycle": {**sub, "start_frame": start}}


def file_digest(path):
	return hashlib.sha256(Path(path).read_bytes()).hexdigest()
