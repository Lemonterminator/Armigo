"""Verifiable deferred video re-rendering from bronze ``actions.npz`` + ``replay/motion.npz``.

Bronze episode directories are ALWAYS treated as strictly read-only: re-rendered lossless ``.mkv``
streams and their cryptographic verification manifest (``rerender_manifest.json``) are written only
to a caller-specified staging/cache directory outside the bronze episode directory.

:func:`verify_rerender_dir` uses only stdlib + numpy so it can run inside the ``lerobot-cuda``
export environment without requiring ``mujoco``.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..constants import FPS, TOP_CROP, TOP_FULL
from .twin_graph import TASK

RERENDER_MANIFEST_SCHEMA = "armigo-deferred-video/v1"
STUDENT_REQUIRED_CAMERAS: tuple[str, ...] = ("wrist", "top_rgb", "top_depth")
ALL_RENDER_CAMERAS: tuple[str, ...] = ("front", "wrist", "top_rgb", "top_depth")
MAX_GEOM_ALIGNMENT_ERR_M = 0.005


def sha256_file(path: Path) -> str:
	h = hashlib.sha256()
	with Path(path).open("rb") as fh:
		for chunk in iter(lambda: fh.read(1 << 20), b""):
			h.update(chunk)
	return h.hexdigest()


def _write_json_atomic(path: Path, value: Any) -> None:
	path = Path(path)
	tmp = path.with_suffix(path.suffix + ".tmp")
	tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
	tmp.replace(path)


def _check_episode_exportable_meta(episode_dir: Path) -> dict[str, Any]:
	ep_file = episode_dir / "episode.json"
	if not ep_file.is_file():
		raise ValueError(f"missing episode.json in {episode_dir}")
	manifest = json.loads(ep_file.read_text(encoding="utf-8"))
	result = manifest.get("result") or {}
	branches = manifest.get("branches") or []
	if (
		manifest.get("schema") != "armigo-agent-episode/v1"
		or result.get("verdict") != "SUCCESS"
		or not result.get("student_eligible")
		or len(branches) != 1
		or branches[0].get("start_frame") != 0
		or result.get("rewinds")
		or manifest.get("task") != TASK
		or manifest.get("fps") != FPS
	):
		raise ValueError(f"episode {episode_dir} is not eligible for student video rerendering")
	return manifest


def verify_rerender_dir(
	episode_dir: Path,
	video_dir: Path,
	*,
	required_cameras: Sequence[str] = STUDENT_REQUIRED_CAMERAS,
) -> dict[str, Any]:
	"""Verify that ``video_dir`` holds an intact, hash- and alignment-verified rerender for ``episode_dir``."""
	ep_dir = Path(episode_dir)
	v_dir = Path(video_dir)
	m_path = v_dir / "rerender_manifest.json"
	if not m_path.is_file():
		raise ValueError(f"missing rerender_manifest.json in {v_dir}")
	rm = json.loads(m_path.read_text(encoding="utf-8"))
	if rm.get("schema") != RERENDER_MANIFEST_SCHEMA:
		raise ValueError(f"unsupported rerender manifest schema {rm.get('schema')!r}")
	if rm.get("verified_alignment") is not True:
		raise ValueError(f"rerender manifest in {v_dir} is not alignment-verified")
	if int(rm.get("fps", -1)) != FPS:
		raise ValueError(f"rerender fps {rm.get('fps')!r} != {FPS}")

	ep_file = ep_dir / "episode.json"
	actions_file = ep_dir / "branch-0" / "actions.npz"
	motion_file = ep_dir / "replay" / "motion.npz"
	if not ep_file.is_file() or not actions_file.is_file():
		raise ValueError(f"source episode {ep_dir} missing episode.json or branch-0/actions.npz")

	actual_ep_sha = sha256_file(ep_file)
	if rm.get("episode_sha256") != actual_ep_sha:
		raise ValueError(
			f"rerender episode_sha256 mismatch for {ep_dir}: manifest={rm.get('episode_sha256')!r}, actual={actual_ep_sha!r}"
		)
	actual_act_sha = sha256_file(actions_file)
	if rm.get("actions_sha256") != actual_act_sha:
		raise ValueError(
			f"rerender actions_sha256 mismatch for {ep_dir}: manifest={rm.get('actions_sha256')!r}, actual={actual_act_sha!r}"
		)
	if rm.get("motion_sha256") and motion_file.is_file():
		actual_mot_sha = sha256_file(motion_file)
		if rm.get("motion_sha256") != actual_mot_sha:
			raise ValueError(
				f"rerender motion_sha256 mismatch for {ep_dir}: manifest={rm.get('motion_sha256')!r}, actual={actual_mot_sha!r}"
			)

	with np.load(actions_file, allow_pickle=False) as z:
		n_actions = len(z["action"])
	if int(rm.get("num_frames", -1)) != n_actions:
		raise ValueError(f"rerender num_frames {rm.get('num_frames')!r} != actions length {n_actions}")

	listed_cams = list(rm.get("cameras") or [])
	mkv_hashes = rm.get("mkv_sha256") or {}
	for cam in required_cameras:
		if cam not in listed_cams:
			raise ValueError(f"required camera {cam!r} missing from rerender cameras {listed_cams}")
		mkv_path = v_dir / f"{cam}.mkv"
		if not mkv_path.is_file():
			raise ValueError(f"missing rerendered video stream {mkv_path}")
		expected_sha = mkv_hashes.get(cam)
		actual_sha = sha256_file(mkv_path)
		if not expected_sha or actual_sha != expected_sha:
			raise ValueError(
				f"rerender mkv sha256 mismatch for {mkv_path}: expected={expected_sha!r}, actual={actual_sha!r}"
			)
	return rm


def _load_case(episode_dir: Path, manifest: dict[str, Any]) -> tuple[Any, ...]:
	task_file = episode_dir / "task.json"
	case = None
	if task_file.is_file():
		task_data = json.loads(task_file.read_text(encoding="utf-8-sig"))
		if isinstance(task_data, dict):
			case = task_data.get("case")
	if case is None:
		prov = manifest.get("provenance") or {}
		case = prov.get("case") or (prov.get("experiment") or {}).get("case")
	if not isinstance(case, list) or len(case) < 3:
		raise ValueError(f"cannot determine MuJoCo case tuple for {episode_dir} (missing task.json['case'])")
	seed = int(case[0])
	positive = bool(case[1])
	face = int(case[2])
	if len(case) >= 4 and case[3] is not None:
		return (seed, positive, face, case[3])
	return (seed, positive, face)


def _is_realistic(manifest: dict[str, Any]) -> bool:
	prov = manifest.get("provenance") or {}
	exp = prov.get("experiment") or {}
	sensing = exp.get("sensing") or prov.get("sensing")
	if sensing is not None:
		return str(sensing) == "realistic"
	group = str(exp.get("group") or "")
	return not group.startswith("truth")


def _freejoint_bindings(cell: Any, node_id_to_col: dict[str, int]) -> list[tuple[int, int, int, int]]:
	"""Return ``(qpos_adr, dof_adr, visual_geom_id, motion_col)`` for every freejoint body in ``cell``."""
	from ..mujoco_cell._mj import mujoco

	m = cell.model
	out: list[tuple[int, int, int, int]] = []
	for jid in range(m.njnt):
		if int(m.jnt_type[jid]) != int(mujoco.mjtJoint.mjJNT_FREE):
			continue
		body_id = int(m.jnt_bodyid[jid])
		qadr = int(m.jnt_qposadr[jid])
		dadr = int(m.jnt_dofadr[jid])
		# Pick the primary visual geom attached to this body at local origin (group < 3)
		chosen_geom: int | None = None
		for g in range(m.ngeom):
			if int(m.geom_bodyid[g]) != body_id or int(m.geom_group[g]) >= 3:
				continue
			gid_key = f"geom_{g}"
			if gid_key not in node_id_to_col:
				continue
			if np.allclose(m.geom_pos[g], 0.0, atol=1e-6) and np.allclose(
				m.geom_quat[g], [1.0, 0.0, 0.0, 0.0], atol=1e-6
			):
				chosen_geom = g
				break
			if chosen_geom is None:
				chosen_geom = g
		if chosen_geom is None:
			bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"
			raise ValueError(f"cannot align freejoint body {bname!r} to motion.npz node_ids")
		out.append((qadr, dadr, chosen_geom, node_id_to_col[f"geom_{chosen_geom}"]))
	return out


def _robot_check_geom(cell: Any, node_id_to_col: dict[str, int]) -> tuple[int, int] | None:
	"""Pick a moving robot visual geom (gripper) present in ``motion.npz`` for alignment verification."""
	m = cell.model
	for g in range(m.ngeom):
		if int(m.geom_bodyid[g]) == cell.grip_body and int(m.geom_group[g]) < 3:
			col = node_id_to_col.get(f"geom_{g}")
			if col is not None:
				return g, col
	return None


def rerender_student_videos(
	episode_dir: Path,
	out_video_dir: Path,
	*,
	student_cameras_only: bool = True,
) -> dict[str, Any]:
	"""Deterministically re-render student camera ``.mkv`` streams into ``out_video_dir`` without mutating ``episode_dir``.

	Rebuilds the exact MuJoCo ``Cell`` from ``task.json['case']``, verifies ``actions.npz`` and ``replay/motion.npz``
	frame/node/kinematic alignment, renders each frame's pre-action student cameras (using deterministic D415 top-depth
	noise from ``defect_seed + 40_000`` for realistic episodes), encodes lossless ``.mkv`` streams, and writes a
	cryptographically verifiable ``rerender_manifest.json``.
	"""
	ep_dir = Path(episode_dir)
	out_dir = Path(out_video_dir)
	ep_res = ep_dir.resolve()
	out_res = out_dir.resolve()
	if out_res == ep_res or out_res.is_relative_to(ep_res):
		raise ValueError(f"refusing to write rerendered videos inside bronze episode directory {ep_res}: {out_res}")

	manifest = _check_episode_exportable_meta(ep_dir)
	actions_file = ep_dir / "branch-0" / "actions.npz"
	motion_file = ep_dir / "replay" / "motion.npz"
	if not actions_file.is_file():
		raise ValueError(f"missing {actions_file}")
	if not motion_file.is_file():
		raise ValueError(f"missing {motion_file} required for deferred video rerendering")

	with np.load(actions_file, allow_pickle=False) as az:
		n_actions = len(az["action"])
		if n_actions <= 0:
			raise ValueError(f"actions.npz in {ep_dir} has 0 frames")
		q_rows = np.asarray(az["state_f64"] if "state_f64" in az.files else az["state"], dtype=np.float64).copy()
		if q_rows.shape != (n_actions, 6):
			raise ValueError(f"unexpected state shape {q_rows.shape}, expected ({n_actions}, 6)")

	with np.load(motion_file, allow_pickle=False) as mz:
		if "poses" not in mz.files or "node_ids" not in mz.files:
			raise ValueError(f"motion.npz in {ep_dir} missing 'poses' or 'node_ids'")
		poses = np.asarray(mz["poses"], dtype=np.float64).copy()
		motion_node_ids = [str(x) for x in mz["node_ids"].tolist()]

	if poses.ndim != 3 or poses.shape[2] != 7 or poses.shape[1] != len(motion_node_ids):
		raise ValueError(f"invalid motion.npz poses shape {poses.shape} for {len(motion_node_ids)} node_ids")
	if len(poses) == n_actions:
		pose_offset = -1
		pose_indexing = "post_step_with_warmup_frame0"
	elif len(poses) == n_actions + 1:
		pose_offset = 0
		pose_indexing = "pre_step_exact"
	else:
		raise ValueError(
			f"motion.npz frame count {len(poses)} misaligned with actions.npz length {n_actions} "
			f"(expected {n_actions} or {n_actions + 1})"
		)

	# Lazy imports so verification-only callers (e.g. lerobot-cuda) never need mujoco.
	from ..benchmarks import SENSITIVITY_MATRIX
	from ..constants import SO101_MJCF_DIR
	from ..depth import layered_height_image
	from ..mujoco_cell import Cell
	from ..mujoco_cell._mj import mujoco
	from ..mujoco_cell.episode import PhysicsEpisode
	from ..physics import NonIdealityCfg, load_physics_spec
	from ..sensor_noise import DEFAULT_D415_NOISE, degrade_depth_height
	from .twin_mujoco import INITIAL_Q, sample_case
	from .twin_record import Video

	case = _load_case(ep_dir, manifest)
	layout = sample_case(*case)
	spec = load_physics_spec()
	base = next(c for c in SENSITIVITY_MATRIX if c.name == "baseline_2x2")
	cfg = NonIdealityCfg(
		latency_frames=base.latency_frames,
		backlash_rad=math.radians(base.backlash_deg),
		velocity_limit_rad_s=spec.arm_actuator.velocity_limit,
		encoder_ticks=4096,
	)

	cameras = STUDENT_REQUIRED_CAMERAS if student_cameras_only else ALL_RENDER_CAMERAS
	out_dir.mkdir(parents=True, exist_ok=True)
	top_depth_rng = (
		np.random.default_rng(int(getattr(layout, "defect_seed", 0)) + 40_000) if _is_realistic(manifest) else None
	)

	cell = Cell(layout, SO101_MJCF_DIR, physics=spec, dynamic_clutter=True, assisted_grasp=False, face_probes=True)
	streams: dict[str, Video] = {}
	max_err = 0.0
	try:
		# Match TwinMujoco's exact post-warmup initial state for frame 0 when motion.npz is post-step.
		PhysicsEpisode(cell, INITIAL_Q, layout.brick_xy, layout.brick_yaw_deg, cfg)

		expected_geoms = [g for g in range(cell.model.ngeom) if int(cell.model.geom_group[g]) < 3]
		expected_node_ids = [f"geom_{g}" for g in expected_geoms]
		if motion_node_ids != expected_node_ids:
			raise ValueError(
				f"motion.npz node_ids mismatch with rebuilt scene for {ep_dir}: "
				f"got {len(motion_node_ids)} nodes vs expected {len(expected_node_ids)}"
			)
		node_id_to_col = {nid: idx for idx, nid in enumerate(motion_node_ids)}
		bindings = _freejoint_bindings(cell, node_id_to_col)
		robot_check = _robot_check_geom(cell, node_id_to_col)

		top_pose = cell._cam_poses.get("top")
		if top_depth_rng is not None and top_pose is None:
			raise ValueError("top CameraPose must be set before applying camera-frame D415 depth noise")

		for i in range(n_actions):
			q_i = q_rows[i]
			cell.data.qpos[cell.qadr] = q_i
			cell.data.ctrl[:6] = q_i
			cell.data.qvel[:] = 0.0
			pose_idx = i + pose_offset
			if pose_idx >= 0:
				frame_poses = poses[pose_idx]
				for qadr, dadr, _geom_id, col in bindings:
					p = frame_poses[col]
					cell.data.qpos[qadr : qadr + 3] = p[:3]
					qx, qy, qz, qw = p[3:]
					cell.data.qpos[qadr + 3 : qadr + 7] = (qw, qx, qy, qz)
					cell.data.qvel[dadr : dadr + 6] = 0.0
			mujoco.mj_forward(cell.model, cell.data)

			if pose_idx >= 0:
				frame_poses = poses[pose_idx]
				for _qadr, _dadr, geom_id, col in bindings:
					err = float(np.linalg.norm(cell.data.geom_xpos[geom_id] - frame_poses[col, :3]))
					max_err = max(max_err, err)
					if err > MAX_GEOM_ALIGNMENT_ERR_M:
						raise ValueError(
							f"frame {i} freejoint alignment error {err:.6f} m exceeds {MAX_GEOM_ALIGNMENT_ERR_M} m"
						)
				if robot_check is not None:
					r_geom, r_col = robot_check
					r_err = float(np.linalg.norm(cell.data.geom_xpos[r_geom] - frame_poses[r_col, :3]))
					max_err = max(max_err, r_err)
					if r_err > MAX_GEOM_ALIGNMENT_ERR_M:
						raise ValueError(
							f"frame {i} robot kinematics alignment error {r_err:.6f} m exceeds {MAX_GEOM_ALIGNMENT_ERR_M} m"
						)

			top_rgb = cell.rgb("top")
			z, valid = cell.height("top", TOP_CROP)
			if top_depth_rng is not None:
				z, valid = degrade_depth_height(
					z,
					valid,
					top_pose,
					top_depth_rng,
					DEFAULT_D415_NOISE,
					crop=TOP_CROP,
					full_shape=(TOP_FULL[1], TOP_FULL[0]),
				)
			top_depth = layered_height_image(z, valid)
			rendered: dict[str, np.ndarray] = {
				"wrist": cell.rgb("wrist_cam"),
				"top_rgb": top_rgb,
				"top_depth": top_depth,
			}
			if not student_cameras_only:
				rendered["front"] = cell.rgb("front")

			if not streams:
				streams = {cam: Video(out_dir / f"{cam}.mkv", rendered[cam].shape, lossless=True) for cam in cameras}
			for cam in cameras:
				streams[cam].put(rendered[cam])
	finally:
		for s in streams.values():
			s.close()
		cell.close()

	mkv_sha256 = {cam: sha256_file(out_dir / f"{cam}.mkv") for cam in cameras}
	rerender_manifest: dict[str, Any] = {
		"schema": RERENDER_MANIFEST_SCHEMA,
		"source_episode_dir": str(ep_dir),
		"episode_sha256": sha256_file(ep_dir / "episode.json"),
		"actions_sha256": sha256_file(actions_file),
		"motion_sha256": sha256_file(motion_file),
		"cameras": list(cameras),
		"num_frames": n_actions,
		"fps": FPS,
		"student_cameras_only": bool(student_cameras_only),
		"pose_indexing": pose_indexing,
		"max_geom_pos_err_m": round(max_err, 8),
		"verified_alignment": True,
		"mkv_sha256": mkv_sha256,
	}
	_write_json_atomic(out_dir / "rerender_manifest.json", rerender_manifest)
	return rerender_manifest
