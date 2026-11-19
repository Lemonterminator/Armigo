"""Ground-truth grader for recorded expert episodes before VLA export (``docs/VLA_DISTILLATION_PLAN.md``, S6).

The grader only filters, so it may read ground truth: the ``view_truth`` events, ``audit.private.json`` and the
per-frame contact record of :class:`~.twin_record.EpisodeTrace`. :func:`grade` returns one dict per episode with a
status per check (``pass``, ``fail`` or ``unavailable`` when the trace lacks the data), the reasons and the numbers.
An episode passes only when every check passes; ``unavailable`` never counts as a pass.

Checks:

1. ``eligible``: the result says ``SUCCESS`` and ``student_eligible`` (``rewinds`` and, when present, ``retries`` are
   reported).
2. ``decision_crop``: the views behind the label truly show what the label claims. Belief face ids are first mapped
   to true ids by the footprint symmetry (yaw known modulo 90 deg for a 2x2 brick, 180 deg for a 2x4) that best
   matches the believed brick orientation (:func:`align_view`). Defective: the ``placement_evidence`` view truly shows
   the defect face and the cross. Normal: every true face is truly visible in a clear view that claims it.
3. ``belief_error``: the believed brick centre is within a bound of the true one at every scored view.
4. ``collisions``: no frame of any branch (all attempts, recoveries and ``unknown`` episodes included) has a robot
   link or a jaw pressing on the environment (table, bowls, clutter) above a limit. Frames come from the simulator's
   own steps; no phase is used to open or close a window.
5. ``smoothness``: phase-normalized jerk indices within the expert's 95th percentile (:func:`smoothness_index`).

Thresholds and the expert reference live in ``config/vla/grader_reference.json`` (``scripts/vla/grade.py
--write-reference`` rebuilds the expert part from bench16 runs).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ..constants import FPS
from ..presentation import FACES

ROOT = Path(__file__).resolve().parents[3]
REFERENCE_PATH = ROOT / "config/vla/grader_reference.json"
CHECKS = ("eligible", "decision_crop", "belief_error", "collisions", "smoothness")
# Used when no reference file is given or a key is missing. See config/vla/grader_reference.json for the reasons.
DEFAULT_THRESHOLDS = {
	"min_true_face_px": {"wrist_cam": 800, "top": 400},  # the agent's own WRIST_MIN_PX / BOTTOM_MIN_PX
	"min_true_cross_px": 25,  # PixelOracleInspector.min_defect_px
	"max_belief_error_mm": {"wrist_cam": 6.0, "top": 12.0},
	"max_robot_contact_n": 3.0,  # metrics.evaluate_episode_health unintended_force_limit_N
	"max_jaw_env_contact_n": 2.0,
}
JERK_FLOOR = 1.0  # rad/s^3; phase medians below this (the commanded rest) are clamped so ratios stay finite


def load_reference(path: str | Path | None = None) -> dict:
	"""The thresholds and expert smoothness reference; defaults only when ``path`` is None and no file exists."""
	path = REFERENCE_PATH if path is None else Path(path)
	if not path.exists():
		return {"thresholds": DEFAULT_THRESHOLDS}
	ref = json.loads(path.read_text(encoding="utf-8"))
	ref["thresholds"] = {**DEFAULT_THRESHOLDS, **ref.get("thresholds", {})}
	return ref


# --- face symmetry -----------------------------------------------------------------------------------------------


def face_map(k: int) -> tuple[int, ...]:
	"""True face id of each belief face id when the believed brick is the true one turned by ``k`` x 90 deg about
	its own z axis (belief R_b = R_t R_z(k 90))."""
	c, s = round(math.cos(k * math.pi / 2)), round(math.sin(k * math.pi / 2))
	vectors = []
	for axis, sign in FACES:
		v = [0, 0, 0]
		v[axis] = sign
		vectors.append(tuple(v))
	out = []
	for x, y, z in vectors:
		out.append(vectors.index((c * x - s * y, s * x + c * y, z)))
	return tuple(out)


def _quat_matrix(xyzw) -> np.ndarray:
	x, y, z, w = (float(v) for v in xyzw)
	return np.array(
		[
			[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
			[2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
			[2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
		]
	)


def _rz(k: int) -> np.ndarray:
	c, s = math.cos(k * math.pi / 2), math.sin(k * math.pi / 2)
	return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def symmetry_from_quats(true_xyzw, belief_xyzw, ks=(0, 1, 2, 3)) -> tuple[int, float]:
	"""``(k, residual_deg)``: the footprint symmetry nearest to the believed orientation expressed in the true brick
	frame, and the rotation left over (the orientation error beyond the symmetry)."""
	rel = _quat_matrix(true_xyzw).T @ _quat_matrix(belief_xyzw)
	k = max(ks, key=lambda k: float(np.trace(_rz(k).T @ rel)))
	cos = (float(np.trace(_rz(k).T @ rel)) - 1.0) / 2.0
	return k, math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def symmetry_from_yaws(true_yaw_deg: float, belief_yaw_deg: float, square: bool) -> int:
	period = 90 if square else 180
	return int(round((belief_yaw_deg - true_yaw_deg) / period) * (period // 90)) % 4


def symmetry_from_pixels(belief_px, true_px, ks) -> int:
	"""The symmetry under which the believed per-face pixel counts best match the true ones (fallback when the
	trace has no orientation)."""

	def cost(k):
		m = face_map(k)
		rotated = np.zeros(6)
		for f, n in enumerate(belief_px):
			rotated[m[f]] = n
		return float(np.abs(rotated - np.asarray(true_px, dtype=float)).sum())

	return min(ks, key=lambda k: (cost(k), k))


def camera_faces(camera: str) -> tuple[int, ...]:
	return (5,) if camera == "top" else (0, 1, 2, 3, 4)


def align_view(view: dict, truth: dict, thresholds: dict, *, k: int, method: str, residual_deg=None) -> dict:
	"""Belief faces of one view, mapped to true ids by symmetry ``k``, against the truly visible faces."""
	camera = view["camera"]
	min_px = thresholds["min_true_face_px"][camera]
	true_px = truth["true_face_pixels"]
	m = face_map(k)
	true_set = sorted(f for f in camera_faces(camera) if true_px[f] >= min_px)
	belief = sorted(view["faces"])
	aligned = sorted(m[f] for f in belief)
	return {
		"id": view["id"],
		"camera": camera,
		"k": k,
		"method": method,
		"residual_deg": None if residual_deg is None else round(residual_deg, 2),
		"belief_faces": belief,
		"aligned_faces": aligned,
		"true_faces": true_set,
		"raw_match": belief == true_set,
		"aligned_match": aligned == true_set,
	}


# --- loading -----------------------------------------------------------------------------------------------------


def _read_json(path: Path):
	return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _layout(task: dict | None):
	"""The case's layout (true initial yaw and brick size) when the simulator package can rebuild it, else None."""
	if not task or "case" not in task:
		return None
	try:
		from .twin_mujoco import sample_case

		return sample_case(*task["case"])
	except Exception:  # noqa: BLE001 - optional fallback only
		return None


def _sensing(manifest: dict, events: list[dict]) -> str:
	sensing = (manifest.get("provenance") or {}).get("sensing")
	if sensing:
		return sensing
	# Traces recorded before provenance carried the mode: only realistic senses the bowls.
	return "realistic" if any(e["kind"] == "bowls_perceived" for e in events) else "truth"


def episode_class(audit: dict | None, task: dict | None) -> tuple[str | None, int | None, str]:
	if audit:
		label, face = audit.get("true_label"), audit.get("true_face")
	elif task and "case" in task:
		_, positive, face = task["case"]
		label, face = ("defective" if positive else "normal"), (face if positive else None)
	else:
		return None, None, "unknown"
	return label, face, ("normal" if label == "normal" else f"defect_face_{face}")


# --- checks ------------------------------------------------------------------------------------------------------


def _status(ok: bool | None) -> str:
	return "unavailable" if ok is None else ("pass" if ok else "fail")


def check_eligible(result: dict) -> dict:
	out = {
		"verdict": result.get("verdict"),
		"student_eligible": result.get("student_eligible"),
		"rewinds": result.get("rewinds"),
	}
	if "retries" in result:
		out["retries"] = result["retries"]
	if "student_eligible" not in result:
		return {"status": "unavailable", "reason": "result has no student_eligible", **out}
	ok = result.get("verdict") == "SUCCESS" and bool(result["student_eligible"])
	reason = None if ok else ("not student_eligible" if result.get("verdict") == "SUCCESS" else "verdict not SUCCESS")
	return {"status": _status(ok), "reason": reason, **out}


def _alignments(views, truths, events, sensing, layout, thresholds) -> dict[str, dict]:
	"""Face alignment of every view that has a truth record (or whose faces came from the simulator)."""
	square = None
	if layout is not None:
		size = layout.brick_size
		square = abs(size[0] - size[1]) < 1e-6
	else:
		for e in events:
			if e["kind"] == "brick_perceived" and e.get("found"):
				a, b = e["footprint_mm"]
				square = max(a, b) / max(min(a, b), 1e-6) < 1.5
				break
	ks = (0, 1, 2, 3) if square in (True, None) else (0, 2)
	out = {}
	belief_yaw = None
	view_by_seq = {v["seq"]: v for v in views}
	for e in events:
		if e["kind"] == "brick_perceived" and e.get("found"):
			belief_yaw = e["yaw_deg"]
		if e["seq"] not in view_by_seq:
			continue
		view = e["observation"]
		truth = truths.get(view["id"])
		if truth is None:
			if sensing == "realistic":
				continue
			# Faces and crop were computed on the simulator's brick: the observation is the truth.
			truth = {"true_face_pixels": view["face_pixels"]}
			out[view["id"]] = align_view(view, truth, thresholds, k=0, method="simulator")
			continue
		if sensing != "realistic":
			out[view["id"]] = align_view(view, truth, thresholds, k=0, method="simulator")
		elif truth.get("true_brick_quat_xyzw") and truth.get("belief_brick_quat_xyzw"):
			k, res = symmetry_from_quats(truth["true_brick_quat_xyzw"], truth["belief_brick_quat_xyzw"], ks)
			out[view["id"]] = align_view(view, truth, thresholds, k=k, method="orientation", residual_deg=res)
		elif layout is not None and belief_yaw is not None:
			k = symmetry_from_yaws(float(layout.brick_yaw_deg), belief_yaw, bool(square))
			out[view["id"]] = align_view(view, truth, thresholds, k=k, method="layout_yaw")
		else:
			k = symmetry_from_pixels(view["face_pixels"], truth["true_face_pixels"], ks)
			out[view["id"]] = align_view(view, truth, thresholds, k=k, method="face_pixels")
	return out


def check_decision_crop(result, views, truths, alignments, true_label, true_face, thresholds) -> dict:
	label = result.get("label")
	by_id = {v["observation"]["id"]: v["observation"] for v in views}
	if label not in ("defective", "normal"):
		return {"status": "fail", "reason": f"no sorting decision (label {label!r})"}
	if label != true_label:
		return {"status": "fail", "reason": f"label {label} but the brick is {true_label}"}
	if label == "defective":
		evidence_id = result.get("placement_evidence")
		if evidence_id is None:
			evidence_id = next(
				(i for i in result.get("evidence", []) if (v := by_id.get(i)) and _positive(v)),
				None,
			)
		view = by_id.get(evidence_id)
		if view is None:
			return {"status": "fail", "reason": "no view triggered the defective label", "view": evidence_id}
		truth = truths.get(evidence_id)
		if truth is None and evidence_id not in alignments:
			return {"status": "unavailable", "reason": "no truth record for the decision view", "view": evidence_id}
		true_px = (truth or {"true_face_pixels": view["face_pixels"]})["true_face_pixels"]
		face_px = int(true_px[true_face])
		min_face = thresholds["min_true_face_px"][view["camera"]]
		cross_px = None if truth is None else truth.get("true_cross_px")
		out = {"view": evidence_id, "defect_face": true_face, "true_face_px": face_px, "true_cross_px": cross_px}
		if face_px < min_face:
			return {"status": "fail", "reason": f"decision view shows {face_px} px of face {true_face}", **out}
		if cross_px is None:
			return {
				"status": "unavailable",
				"reason": "cross pixels not recorded (trace predates true_cross_px)",
				**out,
			}
		if cross_px < thresholds["min_true_cross_px"]:
			return {"status": "fail", "reason": f"decision view shows {cross_px} px of the cross", **out}
		return {"status": "pass", "reason": None, **out}
	covered: set[int] = set()
	false_claims = []
	missing_truth = []
	for vid in result.get("evidence", []):
		view = by_id.get(vid)
		if view is None or view["score"] is None or not view["faces"] or _positive(view):
			continue
		a = alignments.get(vid)
		if a is None:
			missing_truth.append(vid)
			continue
		true_px = (truths.get(vid) or {"true_face_pixels": view["face_pixels"]})["true_face_pixels"]
		min_px = thresholds["min_true_face_px"][view["camera"]]
		for f in a["aligned_faces"]:
			if true_px[f] >= min_px:
				covered.add(f)
			else:
				false_claims.append({"view": vid, "true_face": f, "true_px": int(true_px[f])})
	out = {"truly_cleared": sorted(covered), "false_claims": false_claims}
	if missing_truth:
		return {"status": "unavailable", "reason": "views without truth record", "views": missing_truth, **out}
	missing = sorted(set(range(6)) - covered)
	if missing:
		return {"status": "fail", "reason": f"faces {missing} never truly shown in a clear view", **out}
	return {"status": "pass", "reason": None, **out}


def _positive(view: dict) -> bool:
	return view["score"] is not None and view["score"] >= view["threshold"]


def check_belief_error(views, truths, sensing, thresholds) -> dict:
	scored = [v["observation"] for v in views if v["observation"]["score"] is not None]
	if sensing != "realistic":
		return {"status": "pass", "reason": "faces and crops computed on the simulator brick", "views": len(scored)}
	errors = []
	for view in scored:
		truth = truths.get(view["id"])
		if truth is None:
			return {"status": "unavailable", "reason": f"no truth record for view {view['id']}"}
		errors.append((view["id"], view["camera"], truth.get("belief_error_mm")))
	over = [
		{"view": i, "error_mm": e} for i, cam, e in errors if e is None or e > thresholds["max_belief_error_mm"][cam]
	]
	values = [e for _, _, e in errors if e is not None]
	out = {"views": len(errors), "max_error_mm": max(values) if values else None, "over": over}
	return {"status": _status(not over), "reason": f"{len(over)} views over the bound" if over else None, **out}


def _branches(path: Path, manifest: dict):
	for branch in manifest.get("branches", []):
		file = path / f"branch-{branch['id']}" / "actions.npz"
		if branch.get("actions", 1) and file.exists():
			with np.load(file, allow_pickle=False) as z:
				yield branch, {k: z[k] for k in z.files}


def check_collisions(path: Path, manifest: dict, result: dict, thresholds: dict) -> dict:
	robot_limit, jaw_limit = thresholds["max_robot_contact_n"], thresholds["max_jaw_env_contact_n"]
	frames = robot_hits = jaw_hits = 0
	peaks = [0.0, 0.0]
	first = []
	for branch, z in _branches(path, manifest):
		if "contact_robot_n" not in z or "contact_jaw_env_n" not in z:
			health = result.get("health") or {}
			return {
				"status": "unavailable",
				"reason": "trace has no per-frame contact record",
				"evaluator_unintended_collisions": health.get("unintended_collision_count"),
				"evaluator_ran": bool(health),
			}
		robot, jaw = z["contact_robot_n"], z["contact_jaw_env_n"]
		frames += len(robot)
		hit_r, hit_j = robot > robot_limit, jaw > jaw_limit
		robot_hits += int(hit_r.sum())
		jaw_hits += int(hit_j.sum())
		peaks = [max(peaks[0], float(robot.max(initial=0))), max(peaks[1], float(jaw.max(initial=0)))]
		for i in np.flatnonzero(hit_r | hit_j)[:5]:
			first.append({"branch": branch["id"], "frame": int(branch.get("start_frame", 0) + i)})
	if not frames:
		return {"status": "unavailable", "reason": "no recorded frames"}
	out = {
		"frames": frames,
		"robot_contact_frames": robot_hits,
		"jaw_env_contact_frames": jaw_hits,
		"max_robot_contact_n": round(peaks[0], 3),
		"max_jaw_env_contact_n": round(peaks[1], 3),
		"first_frames": first[:5],
	}
	ok = robot_hits == 0 and jaw_hits == 0
	return {"status": _status(ok), "reason": None if ok else "contact above limit", **out}


def phase_jerk(path: Path, manifest: dict) -> dict[str, tuple[int, float, float]]:
	"""Per phase over all branches: ``(frames, commanded jerk sum of squares, measured jerk sum of squares)`` of the
	arm joints (0-4), jerk as the third finite difference of the target and of the joint reading (rad/s^3)."""
	acc: dict[str, list] = {}
	for _, z in _branches(path, manifest):
		if len(z["action"]) < 4:
			continue
		jc = np.diff(z["action"][:, :5].astype(np.float64), 3, axis=0) * FPS**3
		jm = np.diff(z["state"][:, :5].astype(np.float64), 3, axis=0) * FPS**3
		phases = np.asarray(z["phase"]).astype(str)[3:]
		for name in np.unique(phases):
			m = phases == name
			row = acc.setdefault(str(name), [0, 0.0, 0.0])
			row[0] += int(m.sum())
			row[1] += float(np.sum(np.mean(jc[m] ** 2, axis=1)))
			row[2] += float(np.sum(np.mean(jm[m] ** 2, axis=1)))
	return {k: (n, c, m) for k, (n, c, m) in acc.items()}


def smoothness_index(per_phase: dict, medians: dict) -> dict[str, float] | None:
	"""Frame-weighted mean over phases of (episode phase jerk RMS / expert median phase jerk RMS), for the command
	and the joint reading. Normalizing per phase removes the skill mix (a defective brick skips the slow bottom
	presentation), so episodes of different classes share one reference. None if a phase has no reference."""
	total = sum(n for n, _, _ in per_phase.values())
	if not total:
		return None
	sums = [0.0, 0.0]
	for name, (n, c, m) in per_phase.items():
		if name not in medians:
			return None
		med = medians[name]
		sums[0] += n * math.sqrt(c / n) / max(med[0], JERK_FLOOR)
		sums[1] += n * math.sqrt(m / n) / max(med[1], JERK_FLOOR)
	return {"jerk_index_cmd": sums[0] / total, "jerk_index_meas": sums[1] / total}


def check_smoothness(path: Path, manifest: dict, reference: dict) -> dict:
	smooth = reference.get("smoothness")
	if not smooth:
		return {"status": "unavailable", "reason": "no expert smoothness reference"}
	per_phase = phase_jerk(path, manifest)
	index = smoothness_index(per_phase, smooth["phase_median_rms"])
	if index is None:
		return {"status": "unavailable", "reason": "a phase has no expert reference"}
	over = {k: round(v, 4) for k, v in index.items() if v > smooth["p95"][k]}
	out = {k: round(v, 4) for k, v in index.items()} | {"p95": smooth["p95"]}
	return {"status": _status(not over), "reason": f"above expert p95: {sorted(over)}" if over else None, **out}


# --- entry point -------------------------------------------------------------------------------------------------


def grade(episode_dir: str | Path, reference: dict | None = None, *, rebuild_layout: bool = True) -> dict:
	"""Grade one recorded episode directory (``episode.json``, ``events.jsonl``, ``branch-*/actions.npz``)."""
	path = Path(episode_dir)
	reference = load_reference() if reference is None else reference
	thresholds = {**DEFAULT_THRESHOLDS, **reference.get("thresholds", {})}
	manifest = json.loads((path / "episode.json").read_text(encoding="utf-8"))
	result = manifest["result"]
	events = [json.loads(line) for line in (path / "events.jsonl").read_text(encoding="utf-8").splitlines() if line]
	audit = _read_json(path / "audit.private.json")
	task = _read_json(path / "task.json")
	true_label, true_face, cls = episode_class(audit, task)
	sensing = _sensing(manifest, events)
	views = [e for e in events if e["kind"] == "view"]
	truths = {e["id"]: e for e in events if e["kind"] == "view_truth"}
	needs_layout = (
		sensing == "realistic"
		and rebuild_layout
		and any(not t.get("true_brick_quat_xyzw") or not t.get("belief_brick_quat_xyzw") for t in truths.values())
	)
	layout = _layout(task) if needs_layout else None
	alignments = _alignments(views, truths, events, sensing, layout, thresholds)
	checks = {
		"eligible": check_eligible(result),
		"decision_crop": check_decision_crop(result, views, truths, alignments, true_label, true_face, thresholds),
		"belief_error": check_belief_error(views, truths, sensing, thresholds),
		"collisions": check_collisions(path, manifest, result, thresholds),
		"smoothness": check_smoothness(path, manifest, reference),
	}
	statuses = [c["status"] for c in checks.values()]
	return {
		"episode": str(path),
		"class": cls,
		"true_label": true_label,
		"true_face": true_face,
		"label": result.get("label"),
		"verdict": result.get("verdict"),
		"sensing": sensing,
		"passed": all(s == "pass" for s in statuses),
		"passed_available": "fail" not in statuses,
		"unavailable": [k for k, c in checks.items() if c["status"] == "unavailable"],
		"checks": checks,
		"alignment": list(alignments.values()),
	}


def find_episodes(root: str | Path) -> list[Path]:
	root = Path(root)
	if (root / "episode.json").exists():
		return [root]
	return sorted(p.parent for p in root.rglob("episode.json"))


def build_smoothness_reference(episode_dirs) -> dict:
	"""Expert smoothness reference from successful episodes: per-phase median jerk RMS (over episodes), then the
	95th percentile of each episode's phase-normalized index."""
	per_episode = []
	for path in episode_dirs:
		manifest = json.loads((Path(path) / "episode.json").read_text(encoding="utf-8"))
		if manifest["result"].get("verdict") != "SUCCESS":
			continue
		per_episode.append(phase_jerk(Path(path), manifest))
	rms: dict[str, list] = {}
	for per_phase in per_episode:
		for name, (n, c, m) in per_phase.items():
			rms.setdefault(name, []).append((math.sqrt(c / n), math.sqrt(m / n)))
	medians = {
		k: [round(float(np.median([x[0] for x in v])), 4), round(float(np.median([x[1] for x in v])), 4)]
		for k, v in rms.items()
	}
	indices = [smoothness_index(p, medians) for p in per_episode]
	keys = ("jerk_index_cmd", "jerk_index_meas")
	return {
		"episodes": len(per_episode),
		"phase_median_rms": medians,
		"p95": {k: round(float(np.percentile([i[k] for i in indices], 95)), 4) for k in keys},
		"median": {k: round(float(np.median([i[k] for i in indices])), 4) for k in keys},
	}


def summarize(grades: list[dict]) -> dict:
	"""Per-class pass rates (strict: unavailable counts as not passed; and over available checks only), the
	overall rate, classes below 0.8x the overall rate, and the face-alignment counts."""
	classes: dict[str, list] = {}
	for g in grades:
		classes.setdefault(g["class"], []).append(g)
	out = {"episodes": len(grades), "classes": {}}
	for key in ("passed", "passed_available"):
		overall = sum(g[key] for g in grades) / len(grades) if grades else 0.0
		out[f"{key}_rate"] = round(overall, 4)
		for cls, gs in sorted(classes.items()):
			rate = sum(g[key] for g in gs) / len(gs)
			row = out["classes"].setdefault(cls, {"episodes": len(gs)})
			row[f"{key}_rate"] = round(rate, 4)
			row[f"{key}_flag"] = rate < 0.8 * overall
	views = [a for g in grades for a in g["alignment"]]
	out["alignment"] = {
		"views": len(views),
		"raw_mismatches": sum(not a["raw_match"] for a in views),
		"aligned_mismatches": sum(not a["aligned_match"] for a in views),
		"methods": {m: sum(a["method"] == m for a in views) for m in sorted({a["method"] for a in views})},
	}
	out["unavailable"] = {c: sum(c in g["unavailable"] for g in grades) for c in CHECKS}
	out["failed"] = {c: sum(g["checks"][c]["status"] == "fail" for g in grades) for c in CHECKS}
	return out
