"""Reverse-planned, forward-recorded grasp demonstrations for ACT (CPU feasibility study).

Per v3 train case (cameras ``drift``):

1. Probe: plan grasp candidates on ground truth (``ClosedLoopExpert.grasp_candidates``), and for the chosen one
   simulate (ideal actuation, arm teleported to the pre-grasp pose) descend -> close -> lift. Only grasps that
   really clamp and lift are kept.
2. Reverse plan: reset the brick to its original pose; from the grasp's hover pose (jaws open) plan a joint-space
   min-jerk path to a random tucked pose (``INITIAL_Q`` or ``ready_q`` plus noise), checked collision-free against
   bowls / clutter / the brick on the kinematic model.
3. Forward record: start at the tucked pose and execute the reversed path, then the last ~2 cm descent, close and
   lift, all through ``NonIdealityCfg`` (latency + backlash). Rows are (observation before command, command):
   joint state, command, wrist and top images. Episodes enter the dataset only when the brick is clamped and lifted.
4. Roll canonicalisation: the grasp roll is chosen as the candidate nearest the current wrist roll (a function of
   the visible brick yaw and the wrist roll only), instead of the planner's clearance score. Both choices are
   logged so label uniqueness can be compared.

Output (never the official data roots): ``<out>/episodes/ep_<seed>/branch-0/actions.npz`` (twin_record columns:
state, action, gripper_cmd, phase, segment, state_f64, action_f64), ``wrist.mkv`` / ``top_rgb.mkv`` (ffv1, one
frame per row), ``meta.json``; ``<out>/results.jsonl`` one line per case. ``--no-video`` skips rendering.

	CUDA_VISIBLE_DEVICES= PYTHONPATH=src python scripts/vla/reverse_grasp_demos.py --n 200 --out C:/scratch/reverse-act
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
	sys.path.insert(0, str(ROOT / "src"))

import numpy as np

TRAIN_SEED0 = 3_000_001
SEGMENTS = ("approach", "grasp", "lift")


def wrap(a: float) -> float:
	return (a + math.pi) % (2 * math.pi) - math.pi


def canonical_index(cands: list[dict], wrist_roll: float) -> int:
	"""Candidate whose grasp roll is nearest the current wrist roll (the nearest symmetric solution)."""
	return int(np.argmin([abs(c["roll"] - wrist_roll) for c in cands]))


def canonical_pick(all_cands: list[dict], wrist_roll: float, min_clear: float) -> tuple[dict | None, bool]:
	"""Nearest symmetric solution over every geometric candidate (clearance ignored, so the label depends only on
	the brick yaw and the wrist roll); when that one is blocked, the nearest clear one. Returns (candidate,
	fell_back)."""
	geo = [c for c in all_cands if c["gap_ok"]]
	if not geo:
		return None, False
	c = geo[canonical_index(geo, wrist_roll)]
	if c["clearance_m"] >= min_clear:
		return c, False
	clear = [c for c in geo if c["clearance_m"] >= min_clear]
	return (clear[canonical_index(clear, wrist_roll)], True) if clear else (None, True)


def visible_yaw(brick_quat_xyzw: np.ndarray, square: bool) -> float:
	"""Brick yaw modulo its visual symmetry (pi for a rectangle, pi/2 for a square), in degrees."""
	x, y, z, w = brick_quat_xyzw
	yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
	sym = math.pi / 2 if square else math.pi
	return math.degrees(yaw % sym)


class Case:
	"""One v3 train layout with a cell, a truth-state expert for planning and a resettable initial state."""

	def __init__(self, seed: int, video: bool) -> None:
		from digital_twin_armigo.agent.twin_mujoco import sample_case
		from digital_twin_armigo.constants import SO101_MJCF_DIR
		from digital_twin_armigo.mujoco_cell import Cell
		from digital_twin_armigo.mujoco_cell.physics_expert import ExpertParams
		from digital_twin_armigo.physics import load_physics_spec

		self.seed = seed
		opts = {"scene": "v3", "split": "train", "cameras": "drift"}
		self.layout = sample_case(seed, bool(seed % 2), seed % 6, opts)
		self.p = ExpertParams()
		self.cell = Cell(
			self.layout,
			SO101_MJCF_DIR,
			physics=load_physics_spec(),
			dynamic_clutter=True,
			assisted_grasp=False,
			face_probes=True,
			render_enabled=video,
		)
		self.qpos0 = self.cell.data.qpos.copy()
		self.lo, self.hi = self.cell.jnt_range[:, 0].copy(), self.cell.jnt_range[:, 1].copy()

	def episode(self, q_start: np.ndarray, non_ideal=None, dxy=(0.0, 0.0), dyaw_deg: float = 0.0):
		from digital_twin_armigo.mujoco_cell.episode import PhysicsEpisode

		self.cell.data.qpos[:] = self.qpos0  # clutter / bowls back to their initial state
		self.cell.data.qvel[:] = 0.0
		xy = (self.layout.brick_xy[0] + dxy[0], self.layout.brick_xy[1] + dxy[1])
		return PhysicsEpisode(self.cell, q_start, xy, self.layout.brick_yaw_deg + dyaw_deg, non_ideal)

	def expert(self, ep):
		from digital_twin_armigo.mujoco_cell.episode import observe
		from digital_twin_armigo.mujoco_cell.physics_expert import ClosedLoopExpert

		ex = ClosedLoopExpert(self.cell, ep.cell.joints(), (0.0, 0.0), self.p)
		ex.obs = observe(self.cell, ep)
		return ex

	# -- collision check on the kinematic model ---------------------------------------------------------------------

	def path_clearance(self, ex, qs: list[np.ndarray]) -> float:
		from digital_twin_armigo.mujoco_cell._mj import mujoco

		m, d = ex.kin.model, ex.kin.data
		robot = sorted(self.cell.fixed_jaw_geoms | self.cell.moving_jaw_geoms | {int(g) for g in self.cell.link_geoms})
		env = [*ex._env_geoms().tolist(), *[int(g) for g in self.cell.brick_geoms]]
		best = 0.05
		for q in qs:
			ex.kin.site_pose(q)
			for g1 in robot:
				for g2 in env:
					best = min(best, float(mujoco.mj_geomDistance(m, d, int(g1), int(g2), best, None)))
		return best


def min_jerk_path(q_from: np.ndarray, q_to: np.ndarray, secs: float, fps: int) -> list[np.ndarray]:
	from digital_twin_armigo.mujoco_cell.physics_expert import _min_jerk

	n = max(2, round(secs * fps) + 1)
	return [q_from + s * (q_to - q_from) for s in _min_jerk(n)[1:]]


def move_secs(q_from: np.ndarray, q_to: np.ndarray, min_s: float, peak: float) -> float:
	from digital_twin_armigo.mujoco_cell.physics_expert import MIN_JERK_PEAK_RATIO

	return max(min_s, float(np.max(np.abs(q_to[:5] - q_from[:5]))) * MIN_JERK_PEAK_RATIO / peak)


def grasp_tail(case: Case, ex, cand: dict, q_pre: np.ndarray) -> list[tuple[np.ndarray, int, str]]:
	"""Targets after the pre-grasp pose: final descent, close ramp, clamp wait, lift (open loop; no servo)."""
	from digital_twin_armigo.constants import FPS

	p = case.p
	out = []
	q_g = cand["q"].copy()
	q_g[5] = cand["open_g"]
	out += [(q, 1, "grasp") for q in min_jerk_path(q_pre, q_g, p.final_descend_s, FPS)]
	out += [(q_g.copy(), 1, "grasp")] * 3
	q_c = q_g.copy()
	q_c[5] = p.hold_g
	out += [(q, 1, "grasp") for q in min_jerk_path(q_g, q_c, p.close_s, FPS)]
	out += [(q_c.copy(), 1, "grasp")] * 12  # fixed clamp wait (the expert's still-check needs ~4-10 frames)
	lift = cand["target"].copy()
	lift[2] = p.lift_z_m
	q_l = ex._ik(lift, q_c, cand["roll"])
	q_l[5] = p.hold_g
	out += [(q, 2, "lift") for q in min_jerk_path(q_c, q_l, p.lift_s, FPS)]
	out += [(q_l.copy(), 2, "lift")] * 9
	return out


def clamped_and_lifted(ep, z0: float, p) -> tuple[bool, str]:
	tel = ep.telemetry()
	f = tel["pinch_normal"][-1]
	dz = float(tel["brick_pos"][-1][2] - z0)
	if dz > 0.03 and min(f) > p.lost_force_n:
		return True, "ok"
	if max(ep.jaw_env_force, default=0.0) > 5.0:
		return False, "jaw_env_collision"
	if dz > 0.005:
		return False, "slipped_in_lift"
	moved = float(np.linalg.norm(tel["brick_pos"][-1][:2] - tel["brick_pos"][0][:2]))
	return False, "pushed_brick" if moved > 0.004 else "no_clamp"


def probe(case: Case, cand: dict) -> tuple[bool, str]:
	"""Ideal-actuation forward check of one candidate from its pre-grasp pose."""
	p = case.p
	ep0 = case.episode(np.array([*cand["q"][:5], cand["open_g"]]))
	ex = case.expert(ep0)
	q_pre = ex._ik(cand["target"] + np.array([0.0, 0.0, p.pregrasp_dz_m]), cand["q"], cand["roll"])
	q_pre[5] = cand["open_g"]
	ep = case.episode(q_pre)
	z0 = float(case.cell.brick_pos()[2])
	for q, _, _ in grasp_tail(case, ex, cand, q_pre):
		ep.step(q)
	return clamped_and_lifted(ep, z0, p)


def sample_home(rng: np.random.Generator, case: Case) -> np.ndarray:
	from digital_twin_armigo.agent.twin_mujoco import INITIAL_Q

	base = INITIAL_Q[:5] if rng.uniform() < 0.5 else np.array(case.p.ready_q)
	q = base + rng.normal(0.0, [0.12, 0.12, 0.12, 0.12, 0.7])
	margin = 0.05
	q = np.clip(q, case.lo[:5] + margin, case.hi[:5] - margin)
	return np.array([*q, case.p.open_g])


def run_case(seed: int, out: Path | None, video: bool, latency: int, backlash_deg: float, roll_mode: str) -> dict:
	from digital_twin_armigo.constants import FPS
	from digital_twin_armigo.physics import NonIdealityCfg, load_physics_spec

	t0 = time.perf_counter()
	rec: dict = {"seed": seed}
	try:
		case = Case(seed, video)
	except Exception as e:  # noqa: BLE001 - a case that cannot be built is a reported failure
		return {**rec, "ok": False, "reason": f"case_build:{type(e).__name__}"}
	p = case.p
	rng = np.random.default_rng([seed, 0xAC7])
	q_home = sample_home(rng, case)
	ep = case.episode(q_home)
	ex = case.expert(ep)
	all_cands = ex.grasp_candidates()
	cands = [c for c in all_cands if c["clearance_m"] >= p.min_grasp_clearance_m and c["gap_ok"]]
	square = abs(case.layout.brick_size[0] - case.layout.brick_size[1]) < 1e-6
	bq = case.cell.brick_quat_xyzw()
	rec.update(
		brick_size=[float(v) for v in case.layout.brick_size],
		square=square,
		visible_yaw_deg=visible_yaw(bq, square),
		home_roll=float(q_home[4]),
		n_cands=len(cands),
		cand_rolls=[float(c["roll"]) for c in cands],
	)
	if not cands:
		return {**rec, "ok": False, "reason": "no_clear_candidate"}
	c_plan = cands[0]  # the planner's choice (best clearance score; independent of the wrist roll)
	c_can, fell_back = canonical_pick(all_cands, float(q_home[4]), p.min_grasp_clearance_m)
	d = sorted(abs(c["roll"] - q_home[4]) for c in all_cands if c["gap_ok"])
	rec.update(
		planner_roll=float(c_plan["roll"]),
		canonical_roll=float(c_can["roll"]),
		canonical_fell_back=fell_back,
		planner_is_canonical=c_plan is c_can,
		canonical_margin_rad=float(d[1] - d[0]) if len(d) > 1 else None,
	)
	cand = c_can if roll_mode == "canonical" else c_plan
	ok, why = probe(case, cand)
	rec["probe"] = why
	if not ok:
		return {**rec, "ok": False, "reason": f"probe_{why}", "secs": time.perf_counter() - t0}

	# reverse plan: grasp hover (open) -> tucked home, checked collision-free
	ep = case.episode(q_home)
	ex = case.expert(ep)
	q_pre = ex._ik(cand["target"] + np.array([0.0, 0.0, p.pregrasp_dz_m]), cand["q"], cand["roll"])
	q_pre[5] = cand["open_g"]
	hover = cand["target"] + np.array([0.0, 0.0, p.hover_z_m - cand["info"]["grasp_z_m"]])
	q_hov = ex._ik(hover, q_pre, cand["roll"])
	q_hov[5] = cand["open_g"]
	back = min_jerk_path(q_pre, q_hov, p.descend_s, FPS)
	back += min_jerk_path(q_hov, q_home, move_secs(q_hov, q_home, 1.2, 1.2), FPS)
	clear = case.path_clearance(ex, back[::3])
	rec["path_clearance_m"] = clear
	if clear < 0.002:
		return {**rec, "ok": False, "reason": "path_collision", "secs": time.perf_counter() - t0}
	fwd = [(q, 0, "approach") for q in [*back[::-1][1:], q_pre]]  # reversed path, ending at the pre-grasp pose
	plan = [(q_home.copy(), 0, "approach")] * 6 + fwd + grasp_tail(case, ex, cand, q_pre)

	spec = load_physics_spec()
	cfg = NonIdealityCfg(
		latency_frames=latency,
		backlash_rad=math.radians(backlash_deg),
		velocity_limit_rad_s=spec.arm_actuator.velocity_limit,
		encoder_ticks=4096,
	)
	ep = case.episode(q_home, cfg)
	z0 = float(case.cell.brick_pos()[2])
	rows = {"state": [], "action": [], "gripper_cmd": [], "phase": [], "segment": []}
	frames: dict[str, list[np.ndarray]] = {"wrist": [], "top_rgb": []}
	for q, seg, phase in plan:
		rows["state"].append(case.cell.joints())
		rows["gripper_cmd"].append(float(ep.last("q_cmd")[5]) if ep.n_frames else float(q_home[5]))
		if video:
			frames["wrist"].append(case.cell.rgb("wrist_cam"))
			frames["top_rgb"].append(case.cell.rgb("top"))
		ep.step(q)
		rows["action"].append(np.asarray(q, dtype=np.float64))
		rows["phase"].append(phase)
		rows["segment"].append(seg)
	ok, why = clamped_and_lifted(ep, z0, p)
	tel = ep.telemetry()
	pre_i = 6 + len(fwd) - 1
	rec.update(
		ok=ok,
		reason="ok" if ok else f"replay_{why}",
		n_rows=len(plan),
		pregrasp_tcp_err_mm=float(1000 * np.linalg.norm(tel["tcp_pos"][pre_i] - ex.kin.site_pose(q_pre)[0])),
		cmd_state_lag_rad=float(np.mean(np.abs(np.array(rows["action"])[:, :5] - tel["qpos"][:, :5]))),
		secs=time.perf_counter() - t0,
	)
	if ok and out is not None:
		write_episode(out / "episodes" / f"ep_{seed}", rows, frames, rec, video)
	return rec


def write_episode(path: Path, rows: dict, frames: dict, rec: dict, video: bool) -> None:
	b = path / "branch-0"
	b.mkdir(parents=True, exist_ok=True)
	state = np.array(rows["state"])
	action = np.array(rows["action"])
	np.savez_compressed(
		b / "actions.npz",
		state=state.astype("float32"),
		action=action.astype("float32"),
		gripper_cmd=np.array(rows["gripper_cmd"], dtype="float32"),
		phase=np.array(rows["phase"]),
		segment=np.array(rows["segment"], dtype=np.int32),
		state_f64=state,
		action_f64=action,
	)
	if video:
		from digital_twin_armigo.agent.twin_record import Video

		for name, imgs in frames.items():
			v = Video(b / f"{name}.mkv", imgs[0].shape, lossless=True)
			for im in imgs:
				v.put(im)
			v.close()
	meta = {k: v for k, v in rec.items() if k != "secs"}
	meta.update(schema="armigo-reverse-grasp/v0", segments=list(SEGMENTS), cameras=list(frames) if video else [])
	(path / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def summarize(recs: list[dict]) -> dict:
	from collections import Counter

	n = len(recs)
	probed = [r for r in recs if r.get("probe") == "ok"]
	replayed = [r for r in probed if "n_rows" in r]
	with_c = [r for r in recs if "planner_is_canonical" in r]
	margins = [r["canonical_margin_rad"] for r in with_c if r.get("canonical_margin_rad") is not None]
	return {
		"cases": n,
		"ok": sum(r["ok"] for r in recs),
		"probe_ok": len(probed),
		"replay_ok_given_probe": f"{sum(r['ok'] for r in replayed)}/{len(replayed)}",
		"reasons": dict(Counter(r["reason"] for r in recs)),
		"planner_equals_canonical": f"{sum(r['planner_is_canonical'] for r in with_c)}/{len(with_c)}",
		"canonical_margin_lt_0.2rad": sum(m < 0.2 for m in margins),
		"canonical_fell_back": sum(bool(r.get("canonical_fell_back")) for r in with_c),
		"pregrasp_tcp_err_mm_median": float(np.median([r["pregrasp_tcp_err_mm"] for r in replayed]))
		if replayed
		else None,
		"mean_secs": float(np.mean([r.get("secs", 0.0) for r in recs])),
	}


def label_spread(seed: int, n: int = 8, xy_mm: float = 2.0, yaw_deg: float = 2.0) -> dict:
	"""Label uniqueness under a near-identical brick pose: perturb the brick by perception-sized amounts (and the
	home pose's other joints, keeping its wrist roll) and record the spread of both roll labels.

	A flip is a label jump > 0.5 rad between perturbations (a different symmetric solution); the smooth part
	(~ the yaw perturbation) is not a flip."""
	case = Case(seed, video=False)
	rng = np.random.default_rng([seed, 0x51D])
	q_home = sample_home(rng, case)
	plan, can = [], []
	fallbacks = 0
	for i in range(n):
		d = (0.0, 0.0) if i == 0 else tuple(rng.uniform(-xy_mm, xy_mm, 2) / 1000)
		dy = 0.0 if i == 0 else float(rng.uniform(-yaw_deg, yaw_deg))
		qh = q_home.copy()
		if i:
			qh[:4] += rng.normal(0.0, 0.08, 4)
		ex = case.expert(case.episode(qh, dxy=d, dyaw_deg=dy))
		all_cands = ex.grasp_candidates()
		cands = [c for c in all_cands if c["clearance_m"] >= case.p.min_grasp_clearance_m and c["gap_ok"]]
		if not cands:
			continue
		plan.append(cands[0]["roll"])
		c_can, fb = canonical_pick(all_cands, float(qh[4]), case.p.min_grasp_clearance_m)
		can.append(c_can["roll"])
		fallbacks += fb
	spread = lambda v: float(max(v) - min(v)) if v else None
	return {
		"seed": seed,
		"n": len(plan),
		"planner_spread": spread(plan),
		"canonical_spread": spread(can),
		"fallbacks": fallbacks,
	}


def main(argv: list[str] | None = None) -> int:
	ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	ap.add_argument("--out", type=Path, default=Path("C:/scratch/reverse-act"))
	ap.add_argument("--n", type=int, default=200)
	ap.add_argument("--seed0", type=int, default=TRAIN_SEED0)
	ap.add_argument("--workers", type=int, default=8)
	ap.add_argument("--latency", type=int, default=2, help="command latency (frames)")
	ap.add_argument("--backlash-deg", type=float, default=0.5)
	ap.add_argument("--roll", choices=("canonical", "planner"), default="canonical")
	ap.add_argument("--no-video", action="store_true")
	ap.add_argument("--uniqueness", action="store_true", help="only the label-spread test (no physics replay)")
	a = ap.parse_args(argv)
	a.out.mkdir(parents=True, exist_ok=True)
	seeds = range(a.seed0, a.seed0 + a.n)
	if a.uniqueness:
		with ProcessPoolExecutor(max(1, a.workers)) as pool:
			res = [r for r in pool.map(label_spread, seeds) if r["n"]]
		summ = {
			"cases": len(res),
			"planner_flips": sum(r["planner_spread"] > 0.5 for r in res),
			"canonical_flips": sum(r["canonical_spread"] > 0.5 for r in res),
			"canonical_flips_without_fallback": sum(r["canonical_spread"] > 0.5 and not r["fallbacks"] for r in res),
			"cases_with_fallback": sum(bool(r["fallbacks"]) for r in res),
			"canonical_spread_median_rad": float(np.median([r["canonical_spread"] for r in res])),
		}
		(a.out / "uniqueness.json").write_text(json.dumps({"summary": summ, "cases": res}, indent=2), encoding="utf-8")
		print(json.dumps(summ, indent=2))
		return 0
	args = (a.out, not a.no_video, a.latency, a.backlash_deg, a.roll)
	recs = []
	with (a.out / "results.jsonl").open("w", encoding="utf-8") as fh:
		if a.workers <= 1:
			it = (run_case(s, *args) for s in seeds)
		else:
			pool = ProcessPoolExecutor(a.workers)
			it = (f.result() for f in as_completed([pool.submit(run_case, s, *args) for s in seeds]))
		for r in it:
			recs.append(r)
			fh.write(json.dumps(r) + "\n")
			fh.flush()
			print(r["seed"], r["reason"], flush=True)
	summ = summarize(recs)
	summ["config"] = {"latency": a.latency, "backlash_deg": a.backlash_deg, "roll": a.roll, "video": not a.no_video}
	(a.out / "summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
	print(json.dumps(summ, indent=2))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
