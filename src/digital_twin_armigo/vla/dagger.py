"""Takeover HG-DAgger (plan S11): train-only suite guard, realistic-only takeover gates, and ``TakeoverTwin``.

Implements S11 on top of ``StudentTwin`` / ``ExpertTracer`` (``agent/student_rollout.py``) without touching
``twin_record.py`` or ``twin_mujoco.py``:

- :func:`assert_train_layouts_only` refuses validation, sealed test, or ``bench16`` layout seeds.
- :class:`TakeoverGate` monitors ONLY realistic observations, sensors, and shadow realistic plans (never ground
  truth ``cell.brick_pos()``, ``layout.brick_xy``, or ``evaluate``).
- :class:`TakeoverTwin` runs ``SegmentPolicy`` until a takeover gate fires, closes the student segment with
  ``reason="takeover"``, recovers in place without physics rewinds (``rewinds == 0``), finishes with S7
  ``ExpertTracer``-identical segment labelling, and writes ``is_blind``, ``is_takeover``, and ``supervise_action``
  (zero back-fill before ``t_takeover``) into ``branch-0/actions.npz``.
- :func:`should_stop_dagger` implements the S11 convergence stopping criterion (< 2 percentage points gain over two
  consecutive validation evaluations).
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..agent.student_rollout import (
	ExpertTracer,
	SegmentPolicy,
	StudentObservation,
	StudentTwin,
	crop_digests,
	telemetry_digest,
	with_grasp_roll,
)
from ..agent.twin_mujoco import (
	MAX_TOTAL_FRAMES,
	TwinMujoco,
	sample_case,
	sweep,
)
from ..agent.twin_record import EpisodeTrace, Segment, policy_inputs, segment_labels
from ..constants import PHASES
from ..mujoco_cell.gripper_state import GripperMonitor, GripperMonitorParams, GripperState
from .contract import ACTION_DIM
from .contract import INITIAL_Q as CONTRACT_INITIAL_Q

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FORBID_SUITES: tuple[str, ...] = (
	"config/vla/val.json",
	"config/vla/test.json",
	"config/agent_bench/bench16.json",
)
TAKEOVER_RETURN_CODE = "TAKEOVER"


def _resolve_repo_path(path: str | Path) -> Path:
	p = Path(path)
	if p.is_absolute() or p.exists():
		return p.resolve()
	candidate = ROOT / p
	return candidate.resolve() if candidate.exists() else p.resolve()


def load_suite_cases(suite_path: str | Path) -> tuple[list[tuple], dict[str, Any]]:
	"""Load case tuples and optional metadata from a suite JSON file."""
	path = _resolve_repo_path(suite_path)
	raw = json.loads(path.read_text(encoding="utf-8-sig"))
	if isinstance(raw, dict):
		if not isinstance(raw.get("cases"), list):
			raise TypeError(f"suite {suite_path} must contain a 'cases' list")
		cases_raw = raw["cases"]
		meta = {k: v for k, v in raw.items() if k != "cases"}
		opts = meta.get("options") or {}
		default_clean = {k: v for k, v in opts.items() if not (k == "cameras" and v == "fixed") and v is not None}
		cases = [tuple(c) if not default_clean or len(c) == 4 else (*c, default_clean) for c in cases_raw]
	elif isinstance(raw, list):
		cases = [tuple(c) for c in raw]
		meta = {}
	else:
		raise TypeError(f"suite {suite_path} must be a JSON array or object with 'cases'")
	return cases, meta


def _extract_case_seeds(cases: Sequence[Any]) -> set[int]:
	seeds: set[int] = set()
	for row in cases:
		if isinstance(row, (list, tuple)):
			if not row:
				raise ValueError("empty case entry")
			seeds.add(int(row[0]))
		else:
			seeds.add(int(row))
	return seeds


def assert_train_layouts_only(
	suite_path: str | Path | None = None,
	cases: Sequence[Any] | None = None,
	forbid_suites: Sequence[str | Path] = DEFAULT_FORBID_SUITES,
) -> None:
	"""Enforce S11 train-only split isolation: reject val, sealed test, and bench16 seeds.

	Raises ``ValueError`` if ``suite_path`` is a forbidden suite, carries ``sealed=True`` or ``split`` in
	``("val", "test")``, or if ANY case seed overlaps any forbidden suite.
	"""
	resolved_forbid = [_resolve_repo_path(p) for p in forbid_suites]
	loaded_cases: Sequence[Any] | None = cases

	if suite_path is not None:
		resolved_suite = _resolve_repo_path(suite_path)
		if resolved_suite in resolved_forbid:
			raise ValueError(f"DAgger rollout forbidden on non-train suite {suite_path!s}")
		suite_cases, meta = load_suite_cases(resolved_suite)
		if meta.get("sealed") is True:
			raise ValueError(f"DAgger rollout forbidden on sealed suite {suite_path!s}")
		split = meta.get("split")
		if split is not None and str(split) != "train":
			raise ValueError(f"DAgger rollout forbidden on non-train split {split!r} ({suite_path!s})")
		if loaded_cases is None:
			loaded_cases = suite_cases

	if loaded_cases is None:
		return

	case_seeds = _extract_case_seeds(loaded_cases)
	if not case_seeds:
		return

	for raw_forbid, resolved_f in zip(forbid_suites, resolved_forbid, strict=True):
		if not resolved_f.exists():
			raise FileNotFoundError(f"forbidden reference suite not found: {raw_forbid}")
		forbid_cases, _ = load_suite_cases(resolved_f)
		forbid_seeds = _extract_case_seeds(forbid_cases)
		overlap = case_seeds & forbid_seeds
		if overlap:
			sample = sorted(overlap)[:10]
			raise ValueError(f"DAgger rollout forbidden on non-train seed(s) overlapping {raw_forbid}: {sample}")


def should_stop_dagger(
	val_rates: Sequence[float],
	*,
	min_delta: float = 0.02,
	patience: int = 2,
) -> bool:
	"""Return True when the last ``patience`` consecutive validation updates each added ``< min_delta``."""
	if patience < 1:
		raise ValueError(f"patience must be >= 1, got {patience}")
	if min_delta < 0.0:
		raise ValueError(f"min_delta must be >= 0, got {min_delta}")
	if len(val_rates) < patience + 1:
		return False
	start = len(val_rates) - patience
	deltas = [float(val_rates[i]) - float(val_rates[i - 1]) for i in range(start, len(val_rates))]
	# Guard against IEEE-754 boundary rounding (e.g. 0.74 - 0.72 == 0.019999999999999907).
	return all(d < (min_delta - 1e-9) for d in deltas)


@dataclass(frozen=True)
class TakeoverConfig:
	"""Configuration for realistic-only S11 takeover gates."""

	deviation_thresh_rad: float = 0.25
	soft_deviation_thresh_rad: float = 0.10
	stall_eps_rad: float = 0.002
	stall_frames: int = 30
	max_segment_frames: int | None = None
	enable_joint_deviation: bool = True
	enable_gripper_blocked: bool = True
	enable_brick_lost: bool = True
	enable_budget_or_stall: bool = True
	allow_multiple_takeovers: bool = False
	shadow_lag_alpha: float = 0.6
	# Per-segment-kind joint-deviation gating. The shadow plan is time-indexed, so in moves to a fixed pose (prepare,
	# return_ready) a student that merely starts earlier or later than the shadow reads as a large deviation; listing
	# such kinds in ``deviation_exempt_kinds`` leaves them to the stall/budget gate. ``grasp_deviation_thresh_rad``
	# tightens the gate where centimetre errors decide the outcome (the soft onset threshold scales with it).
	deviation_exempt_kinds: tuple[str, ...] = ()
	grasp_deviation_thresh_rad: float | None = None
	# Hand a student grasp that ends without a verified clamp to the expert (re-perceive, grasp, lift) instead of
	# failing it, so missed grasps yield expert corrections from the state the student reached.
	takeover_on_no_clamp: bool = False
	# Start the expert from the student's last commanded target instead of the measured joints: the command leads the
	# measurement by the tracking lag, so restarting from the measurement steps the command backwards (a jerk spike).
	expert_from_last_command: bool = False

	def __post_init__(self) -> None:
		if self.deviation_thresh_rad <= 0.0:
			raise ValueError("deviation_thresh_rad must be positive")
		if self.soft_deviation_thresh_rad < 0.0 or self.soft_deviation_thresh_rad > self.deviation_thresh_rad:
			raise ValueError("soft_deviation_thresh_rad must lie in [0, deviation_thresh_rad]")
		if self.stall_frames < 1:
			raise ValueError("stall_frames must be >= 1")
		if not (0.0 < self.shadow_lag_alpha <= 1.0):
			raise ValueError("shadow_lag_alpha must lie in (0, 1]")
		if self.grasp_deviation_thresh_rad is not None and self.grasp_deviation_thresh_rad <= 0.0:
			raise ValueError("grasp_deviation_thresh_rad must be positive")

	def deviation_thresholds(self, kind: str) -> tuple[float, float] | None:
		"""``(hard, soft)`` joint-deviation thresholds for a segment kind, or None if the kind is exempt."""
		if kind in self.deviation_exempt_kinds:
			return None
		if kind == "grasp" and self.grasp_deviation_thresh_rad is not None:
			hard = float(self.grasp_deviation_thresh_rad)
			return hard, hard * self.soft_deviation_thresh_rad / self.deviation_thresh_rad
		return self.deviation_thresh_rad, self.soft_deviation_thresh_rad


@dataclass(frozen=True)
class GateDecision:
	"""Result of evaluating all realistic takeover gates on one frame."""

	trigger: bool
	gate: str | None = None
	reason: str | None = None
	blind_start_frame: int | None = None
	metrics: dict[str, Any] = field(default_factory=dict)


def _sanitized_realistic_obs(
	obs: Mapping[str, Any], brick_belief: tuple[np.ndarray, np.ndarray] | None
) -> dict[str, Any]:
	"""Construct a strictly realistic observation dict containing zero simulator ground truth."""
	if brick_belief is None:
		bpos = np.full(3, np.nan, dtype=np.float64)
		bquat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
	else:
		bpos = np.asarray(brick_belief[0], dtype=np.float64).copy()
		bquat = np.asarray(brick_belief[1], dtype=np.float64).copy()
	return {
		"q": np.asarray(obs["q"], dtype=np.float64).copy(),
		"tcp": np.asarray(obs["tcp"], dtype=np.float64).copy(),
		"brick_pos": bpos,
		"brick_quat": bquat,
		"brick_vel": np.zeros(3, dtype=np.float64),
		"jaw_normal": np.asarray(obs["jaw_normal"], dtype=np.float64).copy(),
		"gripper_cmd": float(obs["gripper_cmd"]),
		"effort_frac": 0.0,  # S1 Option B: never depend on effort_frac
	}


def _put_down_carry_gen(expert, xy: np.ndarray, roll: float, q_lift: np.ndarray):
	"""Realistic lowering segment generator matching ``RelocateExpert._put_down`` carried phase."""
	p, rp = expert.p, expert.rp
	if rp.cartesian_carry:
		start = expert.obs["tcp"].copy()
		z = max(float(start[2]), rp.carry_min_z_m)
		end = np.array([xy[0], xy[1], z], dtype=np.float64)
		n = max(1, math.ceil(float(np.linalg.norm(end[:2] - start[:2])) / rp.carry_step_m))
		q = expert.last.copy()
		frames_per = max(1, expert._frames(rp.carry_step_m / rp.carry_speed_mps))
		for k in range(1, n + 1):
			q_next = expert._ik(start + (end - start) * (k / n), q, roll)
			q_next[5] = p.hold_g
			yield from expert._carry(expert._move(q_next, frames_per / 30.0, "place"))
			q = q_next
			if expert.lost:
				return
	else:
		q = expert._ik(np.array([xy[0], xy[1], rp.transfer_z_m]), q_lift, roll)
		q[5] = p.hold_g
		yield from expert._carry(expert._move(q, expert._carry_secs(q, rp.transfer_s), "place"))
	if expert.lost:
		return
	dz = expert.obs["brick_pos"][2] - expert.obs["tcp"][2]
	z_tcp = rp.release_clearance_m + expert.size[2] / 2.0 - dz
	q = expert._ik(np.array([xy[0], xy[1], z_tcp]), q, roll)
	q[5] = p.hold_g
	yield from expert._carry(expert._move(q, expert._carry_secs(q, rp.place_descend_s), "place"))
	if not expert.lost:
		yield from expert._carry(expert._servo(np.array([xy[0], xy[1], z_tcp]), roll, "place", rp.place_servo_tol_m))


def _put_down_release_gen(expert, roll: float):
	"""Realistic release-on-table segment generator matching ``RelocateExpert._put_down`` open+retreat."""
	p = expert.p
	q_open = expert.last.copy()
	q_open[5] = p.open_g
	yield from expert._move(q_open, p.open_ramp_s, "place")
	still = 0
	for _ in range(expert._frames(p.settle_timeout_s)):
		v = float(np.linalg.norm(expert.obs["brick_vel"]))
		still = still + 1 if v < p.settle_speed_mps else 0
		if still >= p.settle_frames:
			break
		yield expert.last.copy(), "place"
	q_up = expert._ik(expert.obs["tcp"] + np.array([0.0, 0.0, p.retreat_dz_m]), expert.last, roll)
	q_up[5] = p.open_g
	yield from expert._move(q_up, p.retreat_s, "place")


def _carry_to_bowl_gen(expert, q_lift: np.ndarray):
	"""Realistic carry-to-bowl segment generator matching ``ClosedLoopExpert._place`` carried phase."""
	p = expert.p
	roll = p.place_roll
	rxy = expert._release_xy()
	q = expert._ik(np.array([rxy[0], rxy[1], p.transfer_z_m]), q_lift, roll)
	q[5] = p.hold_g
	yield from expert._carry(expert._move(q, expert._carry_secs(q, p.transfer_s), "place"))
	if expert.lost:
		return
	dz = expert.obs["brick_pos"][2] - expert.obs["tcp"][2]
	z_tcp = 0.003 + p.release_clearance_m + expert.size[2] / 2.0 - dz
	q = expert._ik(np.array([rxy[0], rxy[1], z_tcp]), q, roll)
	q[5] = p.hold_g
	yield from expert._carry(expert._move(q, expert._carry_secs(q, p.place_descend_s), "place"))
	if not expert.lost:
		yield from expert._carry(expert._servo(np.array([rxy[0], rxy[1], z_tcp]), roll, "place", p.place_servo_tol_m))


def _release_in_bowl_gen(expert):
	"""Realistic release-in-bowl + return home generator matching ``ClosedLoopExpert._place`` + ``_go_home``."""
	p = expert.p
	roll = p.place_roll
	q_open = expert.last.copy()
	q_open[5] = p.open_g
	yield from expert._move(q_open, p.open_ramp_s, "place")
	still = 0
	for _ in range(expert._frames(p.settle_timeout_s)):
		v = float(np.linalg.norm(expert.obs["brick_vel"]))
		still = still + 1 if v < p.settle_speed_mps else 0
		if still >= p.settle_frames:
			break
		yield expert.last.copy(), "place"
	q_up = expert._ik(expert.obs["tcp"] + np.array([0.0, 0.0, p.retreat_dz_m]), expert.last, roll)
	q_up[5] = p.open_g
	yield from expert._move(q_up, p.retreat_s, "place")
	yield from expert._go_home()


def build_shadow_segment_generator(
	expert,
	seg: Segment,
	obs: Mapping[str, Any],
	realistic_state: Mapping[str, Any],
):
	"""Build a realistic shadow motion generator for ``seg`` from believed state ONLY (never ground truth)."""
	kind = seg.kind
	p = expert.p
	if kind == "prepare":
		yield from expert._hold(p.rest_s, "rest")
		yield from expert._go_ready()
		return
	if kind == "return_ready":
		yield from expert._go_ready()
		return
	if kind == "scan_view":
		path = realistic_state.get("scan_path")
		if not path:
			target_q = seg.params.get("target_q")
			path = [np.asarray(target_q, dtype=np.float64)] if target_q is not None else [expert.last.copy()]
		peak_speed = float(realistic_state.get("scan_peak_speed", 1.0))
		pause_s = realistic_state.get("sweep_pause_s")
		yield from sweep(expert, path, peak_speed, expert.rp.scan_move_s, pause_s=pause_s if len(path) > 1 else None)
		yield from expert._hold_servo(path[-1], expert.rp.scan_hold_s, "approach")
		return
	if kind == "grasp":
		yield from expert._grasp()
		return
	if kind == "lift":
		tcp = obs["tcp"]
		lift_target = np.array([float(tcp[0]), float(tcp[1]), p.lift_z_m], dtype=np.float64)
		q_lift = expert._ik(lift_target, expert.last, float(obs["q"][4]))
		q_lift[5] = p.hold_g
		yield from expert._move(q_lift, expert._carry_secs(q_lift, p.lift_s), "lift")
		return
	if kind == "present_bottom":
		tq = realistic_state.get("bottom_target_q")
		if tq is None:
			tq = seg.params.get("target_q")
		pose = np.asarray(tq, dtype=np.float64).copy() if tq is not None else expert.last.copy()
		pose[5] = p.hold_g
		yield from expert._move(pose, expert._carry_secs(pose, p.show_move_s), "show_move")
		yield from expert._hold_servo(pose, expert.rp.bottom_hold_s, "show_hold")
		return
	if kind == "return_to_lift":
		held = realistic_state.get("held")
		q_lift = np.asarray(held[0], dtype=np.float64).copy() if held is not None else expert.last.copy()
		yield from expert._move(q_lift, expert._carry_secs(q_lift, p.show_move_s), "show_move")
		return
	if kind == "lower_on_table":
		held = realistic_state.get("held")
		q_lift = np.asarray(held[0], dtype=np.float64).copy() if held is not None else expert.last.copy()
		roll = float(held[1]) if held is not None else float(obs["q"][4])
		park_xy = realistic_state.get("park_xy")
		if park_xy is None:
			b_belief = realistic_state.get("brick_belief")
			if b_belief is None:
				raise RuntimeError("lower_on_table shadow plan requires realistic brick belief")
			park_xy = np.asarray(b_belief[0][:2], dtype=np.float64)
		yield from _put_down_carry_gen(expert, np.asarray(park_xy, dtype=np.float64), roll, q_lift)
		return
	if kind == "release_on_table":
		held = realistic_state.get("held")
		roll = float(held[1]) if held is not None else float(obs["q"][4])
		yield from _put_down_release_gen(expert, roll)
		return
	if kind == "carry_to_bowl":
		bowls = realistic_state["bowls"]
		bowl_name = realistic_state.get("bowl_name") or seg.params.get("bowl")
		if not bowl_name or bowl_name not in bowls:
			raise RuntimeError(f"carry_to_bowl shadow plan missing perceived bowl {bowl_name!r}")
		expert.bowl_xy = np.asarray(bowls[bowl_name], dtype=np.float64).copy()
		held = realistic_state.get("held")
		q_lift = np.asarray(held[0], dtype=np.float64).copy() if held is not None else expert.last.copy()
		yield from _carry_to_bowl_gen(expert, q_lift)
		return
	if kind == "release_in_bowl":
		yield from _release_in_bowl_gen(expert)
		return
	if kind == "let_go":
		q = expert.last.copy()
		q[5] = p.open_g
		yield from expert._move(q, p.close_s, "place")
		up = expert._ik(obs["tcp"] + np.array([0.0, 0.0, 0.06]), q, float(q[4]))
		up[5] = p.open_g
		yield from expert._move(up, p.descend_s, "place")
		return
	raise ValueError(f"unknown segment kind {kind!r}")


class TakeoverGate:
	"""Realistic-only HG-DAgger takeover monitor.

	Monitors four conditions using ONLY realistic observations, realistic sensor flags, and an online shadow
	realistic expert plan computed from the current believed state:
	1. ``brick_lost``: ``seg.held and lost`` (from realistic ``GripperContact``).
	2. ``gripper_blocked``: ``blocked`` (from realistic ``GripperMonitor``).
	3. ``joint_deviation``: ``max_abs_joint_err > deviation_thresh_rad`` against the shadow realistic plan, with
	   ``deviation_onset_frame`` tracked when ``soft_deviation_thresh_rad`` is crossed.
	4. ``budget_or_stall``: ``segment_frame >= max_segment_frames`` or joint motion ``< stall_eps_rad`` for
	   ``stall_frames`` consecutive non-hold frames.
	"""

	def __init__(self, config: TakeoverConfig | None = None) -> None:
		self.config = config or TakeoverConfig()
		self._seg: Segment | None = None
		self._seg_start_frame: int = 0
		self._shadow_expert = None
		self._shadow_gen: Iterator | None = None
		self._shadow_targets: list[np.ndarray] | None = None
		self._shadow_done: bool = False
		self._shadow_step_idx: int = 0
		self._q_expected: np.ndarray | None = None
		self._prev_shadow_target: np.ndarray | None = None
		self._prev_q: np.ndarray | None = None
		self.deviation_onset_frame: int | None = None
		self._stall_run: int = 0
		self._stall_onset_frame: int | None = None

	def start_segment(
		self,
		seg: Segment,
		*,
		obs: Mapping[str, Any],
		realistic_state: Mapping[str, Any],
		segment_start_frame: int = 0,
		shadow_expert_factory: Callable[[np.ndarray, Mapping[str, Any]], Any] | None = None,
		shadow_targets: Sequence[np.ndarray] | None = None,
	) -> None:
		"""Initialise the online shadow realistic plan at the start of ``seg``."""
		self._seg = seg
		self._seg_start_frame = int(segment_start_frame)
		q0 = np.asarray(obs["q"], dtype=np.float64).copy()
		self._q_expected = q0.copy()
		self._prev_shadow_target = q0.copy()
		self._prev_q = q0.copy()
		self.deviation_onset_frame = None
		self._stall_run = 0
		self._stall_onset_frame = None
		self._shadow_done = False
		self._shadow_step_idx = 0
		self._shadow_targets = (
			[np.asarray(t, dtype=np.float64).copy() for t in shadow_targets] if shadow_targets else None
		)
		self._shadow_expert = None
		self._shadow_gen = None

		if shadow_expert_factory is not None:
			ex = shadow_expert_factory(q0, obs)
			ex.last = np.clip(q0, ex._lo, ex._hi)
			ex.obs = obs
			ex.done = False
			ex._gen = build_shadow_segment_generator(ex, seg, obs, realistic_state)
			self._shadow_expert = ex

	def _next_shadow(self, obs: Mapping[str, Any]) -> tuple[np.ndarray, str, bool]:
		"""Advance the online realistic shadow plan by one step and return ``(target, phase, done)``."""
		if self._shadow_targets is not None:
			if self._shadow_step_idx < len(self._shadow_targets):
				tgt = self._shadow_targets[self._shadow_step_idx]
				self._shadow_step_idx += 1
				return tgt.copy(), (self._seg.phase if self._seg is not None else "approach"), False
			self._shadow_done = True
			last = self._shadow_targets[-1] if self._shadow_targets else self._prev_shadow_target
			return last.copy(), "hold", True

		if self._shadow_expert is not None:
			ex = self._shadow_expert
			if ex.done:
				self._shadow_done = True
				return ex.last.copy(), "hold", True
			tgt, phase = ex.act(dict(obs))
			if ex.done:
				self._shadow_done = True
				return ex.last.copy(), "hold", True
			return np.asarray(tgt, dtype=np.float64).copy(), str(phase), False

		# Pure synthetic fallback if neither shadow_targets nor MuJoCo expert factory is provided.
		self._shadow_done = True
		return self._prev_shadow_target.copy(), "hold", True

	def _joint_deviation_error(self, q: np.ndarray, obs: Mapping[str, Any]) -> float:
		"""Max absolute joint error vs expected realistic plan, respecting physical jaw clamp on bricks."""
		arm_err = float(np.max(np.abs(q[:5] - self._q_expected[:5])))
		pinching = float(self._prev_shadow_target[5]) <= 0.20
		if pinching and float(np.min(obs.get("jaw_normal", (0.0, 0.0)))) >= 0.1:
			grip_err = max(0.0, float(q[5]) - 0.55, float(self._q_expected[5]) - float(q[5]) - 0.15)
		else:
			grip_err = abs(float(q[5]) - float(self._q_expected[5]))
		return max(arm_err, grip_err)

	def evaluate(
		self,
		*,
		frame: int,
		segment_frame: int,
		obs: Mapping[str, Any],
		blocked: bool = False,
		lost: bool = False,
		no_dual_frames: int = 0,
	) -> GateDecision:
		"""Check realistic takeover gates at observation ``frame`` before stepping the student."""
		cfg = self.config
		seg = self._seg
		if seg is None:
			raise RuntimeError("TakeoverGate.start_segment must be called before evaluate")

		q = np.asarray(obs["q"], dtype=np.float64)

		# 1. Brick lost (realistic GripperContact)
		if cfg.enable_brick_lost and seg.held and lost:
			blind_start = max(self._seg_start_frame, frame - max(0, int(no_dual_frames)))
			return GateDecision(
				trigger=True,
				gate="brick_lost",
				reason="held_brick_lost",
				blind_start_frame=blind_start,
				metrics={"no_dual_frames": int(no_dual_frames)},
			)

		# 2. Gripper blocked (realistic GripperMonitor)
		if cfg.enable_gripper_blocked and blocked:
			blind_start = max(self._seg_start_frame, frame - 3)
			return GateDecision(
				trigger=True,
				gate="gripper_blocked",
				reason="gripper_monitor_blocked",
				blind_start_frame=blind_start,
			)

		# Compute joint deviation against expected realistic trajectory BEFORE advancing filter for next step.
		max_err = self._joint_deviation_error(q, obs)
		shadow_target, shadow_phase, shadow_done = self._next_shadow(obs)
		shadow_step_norm = float(np.max(np.abs(shadow_target - self._prev_shadow_target)))

		thresholds = cfg.deviation_thresholds(seg.kind) if cfg.enable_joint_deviation else None
		if thresholds is not None:
			hard, soft = thresholds
			if max_err > soft:
				if self.deviation_onset_frame is None:
					self.deviation_onset_frame = frame
			else:
				self.deviation_onset_frame = None

			if max_err > hard:
				t_dev = self.deviation_onset_frame if self.deviation_onset_frame is not None else frame
				return GateDecision(
					trigger=True,
					gate="joint_deviation",
					reason="max_abs_joint_err_exceeded",
					blind_start_frame=t_dev,
					metrics={"max_abs_joint_err": round(max_err, 6), "threshold": hard},
				)

		# 4. Segment frame budget or non-hold stall
		max_frames = seg.max_frames if cfg.max_segment_frames is None else min(seg.max_frames, cfg.max_segment_frames)
		if cfg.enable_budget_or_stall:
			if segment_frame >= max_frames:
				return GateDecision(
					trigger=True,
					gate="budget_or_stall",
					reason="segment_frame_budget",
					blind_start_frame=self.deviation_onset_frame,
					metrics={"segment_frame": int(segment_frame), "max_frames": int(max_frames)},
				)

			in_hold = (not shadow_done) and (
				shadow_phase in ("rest", "show_hold") or shadow_step_norm < cfg.stall_eps_rad
			)
			motion = float(np.max(np.abs(q - self._prev_q))) if segment_frame > 0 else cfg.stall_eps_rad + 1.0
			if not in_hold and motion < cfg.stall_eps_rad:
				if self._stall_run == 0:
					self._stall_onset_frame = frame
				self._stall_run += 1
				if self._stall_run >= cfg.stall_frames:
					return GateDecision(
						trigger=True,
						gate="budget_or_stall",
						reason="stalled_non_hold",
						blind_start_frame=self._stall_onset_frame,
						metrics={"stall_run": self._stall_run, "motion_rad": round(motion, 6)},
					)
			else:
				self._stall_run = 0
				self._stall_onset_frame = None

		# Advance lag-compensated shadow reference for next observation (delay=2 effective: idx = max(0, t - 1)).
		cmd_for_update = shadow_target if segment_frame == 0 else self._prev_shadow_target
		self._q_expected = self._q_expected + cfg.shadow_lag_alpha * (cmd_for_update - self._q_expected)
		self._prev_shadow_target = shadow_target.copy()
		self._prev_q = q.copy()

		return GateDecision(
			trigger=False,
			metrics={"max_abs_joint_err": round(max_err, 6), "stall_run": self._stall_run},
		)


class TakeoverTwin(StudentTwin):
	"""S11 Takeover HG-DAgger backend extending ``StudentTwin`` with realistic-only takeover & S7 expert completion.

	Runs ``policy`` closed-loop until a realistic takeover gate triggers, then closes the student's open segment with
	``reason="takeover"``, recovers in place without rewinding physics (``rewinds == 0``), and executes the remainder
	of the episode using ``ExpertTracer``'s exact segment splitting.

	Per-frame boolean masks ``is_blind``, ``is_takeover``, and ``supervise_action`` are tracked cleanly and appended to
	``branch-0/actions.npz`` upon trace finish (matching ``twin_record.student_frames`` Data Contract v1 columns).
	"""

	def __init__(
		self,
		layout,
		scorer,
		record: EpisodeTrace,
		policy: SegmentPolicy,
		*,
		takeover_config: TakeoverConfig | None = None,
		segment_monitor: Callable[[Segment, int], None] | None = None,
		sensing: str = "realistic",
		**kwargs,
	) -> None:
		if sensing != "realistic":
			raise ValueError(f"TakeoverTwin requires sensing='realistic', got {sensing!r}")
		self.takeover_config = takeover_config or TakeoverConfig()
		self.gate = TakeoverGate(self.takeover_config)
		super().__init__(
			layout,
			scorer,
			record,
			policy,
			segment_monitor=segment_monitor,
			sensing=sensing,
			**kwargs,
		)
		self.taken_over: bool = False
		self._last_gate_obs: dict[str, Any] | None = None
		self.takeover_frame: int | None = None
		self.takeover_gate: str | None = None
		self.takeover_reason: str | None = None
		self.takeover_segment_id: int | None = None
		self.takeover_segment_kind: str | None = None
		self.deviation_onset_frame: int | None = None
		self._frame_is_blind: list[bool] = []
		self._frame_is_takeover: list[bool] = []
		self._frame_supervise_action: list[bool] = []
		self._active_scan_path: list[np.ndarray] | None = None
		self._active_bottom_q: np.ndarray | None = None
		self._hook_record_finish()

	def _hook_record_finish(self) -> None:
		"""Wrap ``record._finish_branch`` and ``record.finish`` so mask arrays & takeover metadata persist cleanly."""
		rec = self.record
		orig_finish_branch = rec._finish_branch
		orig_finish = rec.finish

		def wrapped_finish_branch() -> None:
			branch_id = rec.branch
			had_rows = bool(rec.rows)
			orig_finish_branch()
			if had_rows and branch_id == 0:
				self._write_takeover_masks_to_npz(rec.path / f"branch-{branch_id}" / "actions.npz")

		def wrapped_finish(result: dict[str, Any], provenance: dict[str, Any]) -> None:
			orig_finish(result, provenance)
			self._write_takeover_metadata_to_manifest(rec.path / "episode.json")

		rec._finish_branch = wrapped_finish_branch
		rec.finish = wrapped_finish

	def takeover_summary(self) -> dict[str, Any]:
		"""Return summary metadata of the takeover event and per-frame mask counts."""
		return {
			"triggered": bool(self.taken_over),
			"frame": self.takeover_frame,
			"gate": self.takeover_gate,
			"reason": self.takeover_reason,
			"segment_id": self.takeover_segment_id,
			"segment_kind": self.takeover_segment_kind,
			"deviation_onset_frame": self.deviation_onset_frame,
			"total_frames": len(self._frame_is_takeover),
			"blind_frames": int(sum(self._frame_is_blind)),
			"takeover_frames": int(sum(self._frame_is_takeover)),
			"supervised_frames": int(sum(self._frame_supervise_action)),
		}

	def _write_takeover_masks_to_npz(self, npz_path: Path) -> None:
		if not npz_path.exists() or not self._frame_is_blind:
			return
		with np.load(npz_path, allow_pickle=False) as z:
			data = {k: z[k] for k in z.files}
		n = len(data["action"])
		if len(self._frame_is_blind) != n:
			raise RuntimeError(f"takeover mask length {len(self._frame_is_blind)} != actions.npz rows {n}")
		data["is_blind"] = np.asarray(self._frame_is_blind, dtype=bool)
		data["is_takeover"] = np.asarray(self._frame_is_takeover, dtype=bool)
		data["supervise_action"] = np.asarray(self._frame_supervise_action, dtype=bool)
		tmp = npz_path.with_name(f"{npz_path.stem}.tmp.npz")
		np.savez_compressed(tmp, **data)
		tmp.replace(npz_path)

	def _write_takeover_metadata_to_manifest(self, manifest_path: Path) -> None:
		if not manifest_path.exists():
			return
		manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
		manifest["takeover"] = self.takeover_summary()
		tmp = manifest_path.with_suffix(".json.tmp")
		tmp.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
		tmp.replace(manifest_path)

	def _apply_target(self, obs, target, phase):
		# Student frames (< t_takeover) NEVER receive expert action supervision (zero back-fill!).
		is_to = bool(self.taken_over)
		self._frame_is_blind.append(False)
		self._frame_is_takeover.append(is_to)
		self._frame_supervise_action.append(is_to)
		return super()._apply_target(obs, target, phase)

	def _realistic_state_for_gate(self, seg: Segment) -> dict[str, Any]:
		"""Gather strictly realistic state for the online shadow planner (never ground truth)."""
		b_belief = self._brick_belief()
		park_xy = None
		if self.brick_est is not None:
			park_xy = self.brick_est.pos[:2].copy()
		return {
			"brick_belief": b_belief,
			"bowls": {name: np.asarray(xy, dtype=np.float64).copy() for name, xy in self.bowls.items()},
			"held": None if self.held is None else (self.held[0].copy(), float(self.held[1])),
			"park_xy": park_xy,
			"scan_path": self._active_scan_path,
			"scan_peak_speed": float(self.scan_peak_speed),
			"sweep_pause_s": self.sweep_pause_s,
			"bottom_target_q": self._active_bottom_q,
			"bowl_name": seg.params.get("bowl"),
		}

	def _make_shadow_expert_for_gate(self, q0: np.ndarray, obs: Mapping[str, Any]):
		del q0, obs
		return self._make_expert(self.expert.p, self.expert.rp)

	def _trigger_takeover(
		self, seg: Segment, sid: int, decision: GateDecision, obs: Mapping[str, Any], *, end_segment: bool = True
	) -> str:
		t_takeover = self.ep.n_frames
		t_dev = decision.blind_start_frame
		if t_dev is not None:
			t_dev = max(0, min(int(t_dev), t_takeover))
			for idx in range(t_dev, t_takeover):
				self._frame_is_blind[idx] = True

		self.taken_over = True
		self.takeover_frame = t_takeover
		self.takeover_gate = decision.gate
		self.takeover_reason = decision.reason
		self.takeover_segment_id = sid
		self.takeover_segment_kind = seg.kind
		self.deviation_onset_frame = t_dev

		if end_segment:
			self._segment_end("takeover")
		self.event(
			"takeover",
			gate=decision.gate,
			reason=decision.reason,
			segment_id=sid,
			segment_kind=seg.kind,
			deviation_onset_frame=t_dev,
			metrics=decision.metrics,
		)
		ex = self.expert
		start = self._prev_action if self.takeover_config.expert_from_last_command else obs["q"]
		ex.last = np.clip(np.asarray(start, dtype=np.float64), ex._lo, ex._hi)
		ex.done = False
		ex.obs = dict(obs)
		return TAKEOVER_RETURN_CODE

	def _takeover_after_no_clamp(self, seg: Segment) -> str | None:
		"""The student's grasp segment ended without a verified clamp: hand over to the expert in place (gate
		``no_clamp``), or return None to keep the plain ``no_clamp`` failure when the gate is off or already used."""
		cfg = self.takeover_config
		if not cfg.takeover_on_no_clamp or self._last_gate_obs is None:
			return None
		if self.taken_over and not cfg.allow_multiple_takeovers:
			return None
		decision = GateDecision(trigger=True, gate="no_clamp", reason="student_grasp_ended_without_clamp")
		self._trigger_takeover(seg, self.segments[-1]["id"], decision, self._last_gate_obs, end_segment=False)
		return TAKEOVER_RETURN_CODE

	def run_segment(self, seg: Segment) -> str | None:
		"""Run one segment under ``policy`` until ``segment_done``, failure, or realistic takeover gate trigger."""
		ex, p = self.expert, self.expert.p
		sid = self._segment_begin(seg)
		if self.segment_monitor is not None:
			self.segment_monitor(seg, sid)
		task = policy_inputs(seg)
		self.segment_policy.start(dict(task), sid)
		frames = 0
		after_step = False
		clamp_run = 0
		monitor = GripperMonitor(GripperMonitorParams(effort_min=0.0, use_effort=False))
		self._clamped = self._blocked = False
		gate_active = (not self.taken_over) or self.takeover_config.allow_multiple_takeovers
		gate_started = False

		while True:
			if self.total_frames >= MAX_TOTAL_FRAMES:
				self._segment_end("budget")
				return "ACTION_BUDGET_EXHAUSTED"
			if self.on_frame is not None:
				self.on_frame(self)
			obs = self._observe()
			if after_step:
				after_step = False
				f = np.asarray(obs["jaw_normal"], dtype=np.float64)
				if seg.held:
					dual = bool(f[0] > p.lost_force_n and f[1] > p.lost_force_n)
					ex._no_dual = 0 if dual else ex._no_dual + 1
					if ex._no_dual >= p.lost_frames:
						ex.lost = True
				if seg.kind == "grasp":
					v = float(np.linalg.norm(obs["brick_vel"]))
					clamp_run = clamp_run + 1 if (min(f) >= p.clamp_force_n and v < p.clamp_still_speed_mps) else 0
					state, _ = monitor.update(
						float(obs["gripper_cmd"]), float(obs["q"][5]), float(obs["effort_frac"]), obs["jaw_normal"]
					)
					self._blocked |= state is GripperState.BLOCKED

			safe_obs = _sanitized_realistic_obs(obs, self._brick_belief())
			self._last_gate_obs = safe_obs
			if gate_active:
				if not gate_started:
					gate_started = True
					with self._belief():
						self.gate.start_segment(
							seg,
							obs=safe_obs,
							realistic_state=self._realistic_state_for_gate(seg),
							segment_start_frame=self.ep.n_frames,
							shadow_expert_factory=self._make_shadow_expert_for_gate,
						)
				with self._belief():
					decision = self.gate.evaluate(
						frame=self.ep.n_frames,
						segment_frame=frames,
						obs=safe_obs,
						blocked=self._blocked,
						lost=ex.lost,
						no_dual_frames=ex._no_dual,
					)
				if decision.trigger:
					return self._trigger_takeover(seg, sid, decision, safe_obs)

			if seg.held and ex.lost:
				self._segment_end("brick_lost")
				return None

			q = np.asarray(obs["q"], dtype=np.float64).copy()
			q_prev = q.copy() if self._q_prev is None else self._q_prev.copy()
			prev_action = self._prev_action.copy()
			gripper_cmd = float(obs["gripper_cmd"])
			step = self.segment_policy.act(
				StudentObservation(
					frame=self.ep.n_frames,
					task=dict(task),
					segment_id=sid,
					segment_frame=frames,
					q=q,
					_images=self._images,
					q_prev=q_prev,
					prev_action=prev_action,
					gripper_cmd=gripper_cmd,
				)
			)
			if step.done:
				self._clamped = clamp_run >= p.clamp_still_frames
				self._segment_end("done")
				return None
			if frames >= seg.max_frames:
				self._segment_end("frame_budget")
				return "SEGMENT_FRAME_BUDGET"
			raw = np.asarray(step.target, dtype=np.float64)
			if raw.shape != (6,) or not np.isfinite(raw).all():
				raise ValueError("a policy target must be 6 finite joint positions")
			target = np.clip(raw, ex._lo, ex._hi)
			ex.clipped_frames += bool(np.any(target != raw))
			ex.last = target.copy()
			ex._frame_idx += 1
			phase = step.phase or seg.phase
			if phase not in PHASES:
				raise ValueError(f"unknown phase {phase!r}")
			error = self._apply_target(obs, target, phase)
			self._q_prev = q
			self._prev_action = target.copy()
			frames += 1
			after_step = True
			if error:
				self._segment_end("fault")
				return error

	# -- Expert execution & in-place recovery after takeover ------------------------------------------------------

	def _tag_carry(self):
		ExpertTracer._tag_carry(self)

	def _split(self, generator, name, segs):
		return ExpertTracer._split(self, generator, name, segs)

	def _execute_as_expert(self, generator):
		"""Execute ``generator`` identically to ``ExpertTracer._execute`` with S7 segment splitting."""
		self._tag_carry()
		form, value = self._motion(generator)
		if form == "simple":
			self._segment_begin(value)
			wrapped = generator
		else:
			wrapped = self._split(generator, *value)
		error = TwinMujoco._execute(self, wrapped)
		if self._open_segment is not None:
			lost = self._open_segment["segment"].held and self.expert.lost
			self._segment_end(error or ("brick_lost" if lost else "done"))
		if form == "composite" and value[0] == "grasp_and_lift" and self.held is not None:
			self.lift_poses.append(np.asarray(self.held[0], dtype=np.float64).copy())
		return error

	def _execute_expert_simple_segment(self, seg: Segment, generator) -> str | None:
		self._tag_carry()
		self._segment_begin(seg)
		error = TwinMujoco._execute(self, generator)
		if self._open_segment is not None:
			lost = self._open_segment["segment"].held and self.expert.lost
			self._segment_end(error or ("brick_lost" if lost else "done"))
		return error

	def _execute_expert_composite_segment(self, name: str, segs: dict[str, Segment], generator) -> str | None:
		self._tag_carry()
		wrapped = self._split(generator, name, segs)
		error = TwinMujoco._execute(self, wrapped)
		if self._open_segment is not None:
			lost = self._open_segment["segment"].held and self.expert.lost
			self._segment_end(error or ("brick_lost" if lost else "done"))
		return error

	def _execute(self, generator):
		if self.taken_over:
			return self._execute_as_expert(generator)

		local = generator.gi_frame.f_locals if generator.gi_frame is not None else {}
		gen_name = generator.__name__
		self._active_scan_path = (
			[np.asarray(q, dtype=np.float64).copy() for q in local["path"]]
			if gen_name == "look" and "path" in local
			else None
		)
		self._active_bottom_q = (
			np.asarray(local["candidate"].q, dtype=np.float64).copy()
			if gen_name == "look" and "candidate" in local
			else None
		)

		form, value = self._motion(generator)
		generator.close()
		if form == "simple":
			err = self.run_segment(value)
			if err == TAKEOVER_RETURN_CODE:
				return self._recover_simple_segment(value)
			return err
		name, segs = value
		return getattr(self, f"_takeover_{name}")(segs)

	def _recover_simple_segment(self, seg: Segment) -> str | None:
		ex = self.expert
		if seg.kind == "prepare":
			return self._execute_expert_simple_segment(seg, ex._go_ready())
		if seg.kind == "return_ready":
			return self._execute_expert_simple_segment(seg, ex._go_ready())
		if seg.kind == "scan_view":
			path = self._active_scan_path or [np.asarray(seg.params["target_q"], dtype=np.float64)]
			if not self._straight_clear(ex.last, path[0]) and (
				err := self._execute_expert_simple_segment(Segment("return_ready"), ex._go_ready())
			):
				return err
			if not self._straight_clear(ex.last, path[0]):
				return None  # skip unreachable view gracefully like TwinMujoco._wrist

			def look_recover():
				yield from sweep(
					ex,
					path,
					self.scan_peak_speed,
					ex.rp.scan_move_s,
					pause_s=self.sweep_pause_s if len(path) > 1 else None,
				)
				yield from ex._hold_servo(path[-1], ex.rp.scan_hold_s, "approach")

			return self._execute_expert_simple_segment(seg, look_recover())
		if seg.kind == "present_bottom":
			if ex.lost:
				return "BRICK_LOST"
			pose = (
				self._active_bottom_q.copy()
				if self._active_bottom_q is not None
				else np.asarray(seg.params["target_q"], dtype=np.float64)
			)
			if not self._straight_clear(ex.last, pose, held=True):
				if self.held is not None and self._straight_clear(ex.last, self.held[0], held=True):
					q_lift = self.held[0]

					def back_to_lift():
						yield from ex._carry(ex._move(q_lift, ex._carry_secs(q_lift, ex.p.show_move_s), "show_move"))

					if err := self._execute_expert_simple_segment(Segment("return_to_lift"), back_to_lift()):
						return err
				if ex.lost:
					return "BRICK_LOST"
				if not self._straight_clear(ex.last, pose, held=True):
					return "BOTTOM_PATH_BLOCKED"

			def bottom_recover():
				yield from ex._carry(ex._move(pose, ex._carry_secs(pose, ex.p.show_move_s), "show_move"))
				yield from ex._carry(ex._hold_servo(pose, ex.rp.bottom_hold_s, "show_hold"))

			return self._execute_expert_simple_segment(seg, bottom_recover())
		if seg.kind == "let_go":
			return self._execute_expert_simple_segment(seg, self._let_go_gen())
		raise RuntimeError(f"unhandled simple segment takeover for {seg.kind!r}")

	def _let_go_gen(self):
		ex = self.expert
		q = ex.last.copy()
		q[5] = ex.p.open_g
		yield from ex._move(q, ex.p.close_s, "place")
		up = ex._ik(self.cell.tcp_pos() + np.array([0.0, 0.0, 0.06]), q, float(q[4]))
		up[5] = ex.p.open_g
		yield from ex._move(up, ex.p.descend_s, "place")

	def _expert_reperceive_and_grasp_and_lift(self) -> str | None:
		"""Recover in place from a failed student grasp/lift: clear arm, re-perceive brick, grasp & lift."""
		ex = self.expert
		self.held = None
		self._grip_from_brick = None
		ex.lost = False
		ex._no_dual = 0
		ex.aborted = None
		ex.grasp_retries = ex.block_replans = 0
		ex._active_open_g = float(ex.p.open_g)
		if (self.cell.joints()[5] < ex.p.open_g - 0.05 or self._blocked) and (
			err := self._execute_expert_simple_segment(Segment("let_go"), self._let_go_gen())
		):
			return err
		if err := self._execute_expert_simple_segment(Segment("return_ready"), ex._go_ready()):
			return err
		if err := self._perceive():
			return err

		def grasp_and_lift():
			self._belief_obs = True
			try:
				grasp = yield from ex._grasp()
			finally:
				self._belief_obs = False
			if grasp is not None:
				self._grip_from_brick = np.linalg.inv(self._grip_pose()) @ self._table_belief_tf()
				q_lift = yield from ex._lift(grasp)
				self.held = (q_lift, grasp.roll)

		err = self._execute_as_expert(grasp_and_lift())
		return err or ("BRICK_LOST" if ex.lost else ex.aborted)

	def _takeover_grasp_and_lift(self, segs: dict[str, Segment]) -> str | None:
		ex = self.expert
		seg = with_grasp_roll(segs["grasp"], self._planned_grasp_roll())
		self._belief_obs = True
		try:
			error = self.run_segment(seg)
		finally:
			self._belief_obs = False
		if error == TAKEOVER_RETURN_CODE:
			return self._expert_reperceive_and_grasp_and_lift()
		if error:
			return error
		if not self._clamped:
			if not self._blocked and self._takeover_after_no_clamp(seg) == TAKEOVER_RETURN_CODE:
				return self._expert_reperceive_and_grasp_and_lift()
			ex.aborted = "gripper_blocked" if self._blocked else "no_clamp"
			return None
		self._grip_from_brick = np.linalg.inv(self._grip_pose()) @ self._table_belief_tf()
		error = self.run_segment(segs["lift"])
		if error == TAKEOVER_RETURN_CODE:
			if ex.lost:
				return self._expert_reperceive_and_grasp_and_lift()

			def lift_recover():
				self._belief_obs = True
				try:
					tcp = self.cell.tcp_pos()
					lift_target = np.array([tcp[0], tcp[1], ex.p.lift_z_m], dtype=np.float64)
					q_lift = ex._ik(lift_target, ex.last, float(ex.last[4]))
					q_lift[5] = ex.p.hold_g
					yield from ex._carry(ex._move(q_lift, ex._carry_secs(q_lift, ex.p.lift_s), "lift"))
				finally:
					self._belief_obs = False

			err_lift = self._execute_expert_composite_segment("grasp_and_lift", segs, lift_recover())
			if err_lift:
				return err_lift
			if ex.lost:
				return self._expert_reperceive_and_grasp_and_lift()
			self.held = (ex.last.copy(), float(ex.last[4]))
			self.lift_poses.append(ex.last.copy())
			return None
		if error:
			return error
		self.held = (ex.last.copy(), float(ex.last[4]))
		self.lift_poses.append(ex.last.copy())
		return None

	def _takeover_park(self, segs: dict[str, Segment]) -> str | None:
		ex = self.expert
		q_lift, roll = self.held
		park_xy = self.brick_est.pos[:2].copy() if self.brick_est is not None else self._park_xy.copy()

		error = self.run_segment(segs["return_to_lift"])
		if error == TAKEOVER_RETURN_CODE:
			if ex.lost:
				return None
			if not self._straight_clear(ex.last, q_lift, held=True):
				return "PARK_PATH_BLOCKED"

			def park_rest():
				yield from ex._carry(ex._move(q_lift, ex._carry_secs(q_lift, ex.p.show_move_s), "show_move"))
				if not ex.lost:
					yield from _put_down_carry_gen(ex, park_xy, roll, q_lift)
				if not ex.lost:
					yield from _put_down_release_gen(ex, roll)
					if self._grip_from_brick is not None:
						rot = (self._grip_pose() @ self._grip_from_brick)[:3, :3]
						self._expected_yaw = math.atan2(rot[1, 0], rot[0, 0])
				if not ex.lost:
					yield from ex._go_ready()

			return self._execute_expert_composite_segment("park", segs, park_rest())
		if error or ex.lost:
			return error

		error = self.run_segment(segs["lower_on_table"])
		if error == TAKEOVER_RETURN_CODE:
			if ex.lost:
				return None

			def lower_rest():
				yield from _put_down_carry_gen(ex, park_xy, roll, q_lift)
				if not ex.lost:
					yield from _put_down_release_gen(ex, roll)
					if self._grip_from_brick is not None:
						rot = (self._grip_pose() @ self._grip_from_brick)[:3, :3]
						self._expected_yaw = math.atan2(rot[1, 0], rot[0, 0])
				if not ex.lost:
					yield from ex._go_ready()

			return self._execute_expert_composite_segment("park", segs, lower_rest())
		if error or ex.lost:
			return error

		error = self.run_segment(segs["release_on_table"])
		if error == TAKEOVER_RETURN_CODE:

			def release_rest():
				yield from _put_down_release_gen(ex, roll)
				if self._grip_from_brick is not None:
					rot = (self._grip_pose() @ self._grip_from_brick)[:3, :3]
					self._expected_yaw = math.atan2(rot[1, 0], rot[0, 0])
				yield from ex._go_ready()

			return self._execute_expert_composite_segment("park", segs, release_rest())
		if error:
			return error

		if self._grip_from_brick is not None:
			rot = (self._grip_pose() @ self._grip_from_brick)[:3, :3]
			self._expected_yaw = math.atan2(rot[1, 0], rot[0, 0])

		if not ex.lost:
			error = self.run_segment(segs["return_ready"])
			if error == TAKEOVER_RETURN_CODE:
				return self._execute_expert_composite_segment("park", segs, ex._go_ready())
			return error
		return None

	def _takeover_release(self, segs: dict[str, Segment]) -> str | None:
		ex = self.expert
		q_lift, _ = self.held
		if np.max(np.abs(ex.last - q_lift)) > 0.01:
			error = self.run_segment(segs["return_to_lift"])
			if error == TAKEOVER_RETURN_CODE:
				if ex.lost:
					return None
				if not self._straight_clear(ex.last, q_lift, held=True):
					return "PLACE_RETURN_PATH_BLOCKED"

				def release_full():
					yield from ex._carry(ex._move(q_lift, ex._carry_secs(q_lift, ex.p.show_move_s), "show_move"))
					if not ex.lost:
						yield from _carry_to_bowl_gen(ex, q_lift)
					if not ex.lost:
						yield from _release_in_bowl_gen(ex)

				return self._execute_expert_composite_segment("release", segs, release_full())
			if error or ex.lost:
				return error

		if ex.lost:
			return None

		error = self.run_segment(segs["carry_to_bowl"])
		if error == TAKEOVER_RETURN_CODE:
			if ex.lost:
				return None

			def carry_rest():
				tcp = self.cell.tcp_pos()
				if tcp[2] < ex.p.transfer_z_m - 0.01:
					up = ex._ik(np.array([tcp[0], tcp[1], ex.p.transfer_z_m]), ex.last, float(ex.last[4]))
					up[5] = ex.p.hold_g
					yield from ex._carry(ex._move(up, ex._carry_secs(up, 0.6), "place"))
				if not ex.lost:
					yield from _carry_to_bowl_gen(ex, q_lift)
				if not ex.lost:
					yield from _release_in_bowl_gen(ex)

			return self._execute_expert_composite_segment("release", segs, carry_rest())
		if error or ex.lost:
			return error

		if not ex.lost:
			error = self.run_segment(segs["release_in_bowl"])
			if error == TAKEOVER_RETURN_CODE:
				return self._execute_expert_composite_segment("release", segs, _release_in_bowl_gen(ex))
			return error
		return None


def run_takeover_episode(
	case: tuple,
	scorer,
	out: Path,
	*,
	policy: SegmentPolicy,
	takeover_config: TakeoverConfig | None = None,
	segment_monitor: Callable[[Segment, int], None] | None = None,
	video: bool = False,
	enforce_train_only: bool = True,
) -> dict[str, Any]:
	"""Run one closed-loop ``TakeoverTwin`` episode on a training case and write trace + ``run.json``."""
	if enforce_train_only:
		assert_train_layouts_only(suite_path=None, cases=[case])

	from ..agent.twin_graph import Policy, build_graph

	out = Path(out)
	trace = EpisodeTrace(out, video=video)
	layout = sample_case(*case)
	twin = TakeoverTwin(
		layout,
		scorer,
		trace,
		policy,
		takeover_config=takeover_config,
		segment_monitor=segment_monitor,
		sensing="realistic",
	)
	try:
		result = build_graph(twin, Policy()).invoke({}, config={"recursion_limit": 64})["result"]
		trace.finish(
			result,
			{
				"engine": "mujoco",
				"sensing": "realistic",
				"case": list(case),
				"student": True,
				"dagger_takeover": True,
			},
		)
		run = {
			"case": list(case),
			"result": json.loads(json.dumps(result)),
			"takeover": twin.takeover_summary(),
			"events": trace.events,
			"digest": telemetry_digest(twin),
			"crops": crop_digests(out),
			"frames": twin.ep.n_frames,
			"lift_poses": [q.tolist() for q in twin.lift_poses],
			"clipped_frames": int(twin._smoothness().get("clipped_frames", 0)),
		}
		(out / "run.json").write_text(json.dumps(run, allow_nan=False), encoding="utf-8")
		return run
	finally:
		twin.close()


def episode_to_dataset_dict(
	episode_dir: str | Path,
	*,
	episode_index: int = 0,
	load_video: bool = False,
) -> dict[str, Any]:
	"""Load a ``TakeoverTwin`` episode directory into a Data Contract v1 dict for ``SegmentChunkDataset``.

	Reads ``is_blind``, ``is_takeover``, and ``supervise_action`` from ``branch-0/actions.npz`` directly so both
	pre-merge and post-merge environments validate chunk cutting, blind masking, and 2x takeover weighting.
	"""
	ep = Path(episode_dir)
	labels = segment_labels(ep)
	with np.load(ep / "branch-0" / "actions.npz", allow_pickle=False) as rows:
		states = np.asarray(rows["state"], dtype=np.float32).reshape(-1, ACTION_DIM)
		actions = np.asarray(rows["action"], dtype=np.float32).reshape(-1, ACTION_DIM)
		n = len(states)
		gripper_cmds = (
			np.asarray(rows["gripper_cmd"], dtype=np.float32).reshape(-1) if "gripper_cmd" in rows.files else None
		)
		is_blind = (
			np.asarray(rows["is_blind"], dtype=bool).reshape(-1)
			if "is_blind" in rows.files
			else np.zeros(n, dtype=bool)
		)
		is_takeover = (
			np.asarray(rows["is_takeover"], dtype=bool).reshape(-1)
			if "is_takeover" in rows.files
			else np.zeros(n, dtype=bool)
		)
		supervise_action = (
			np.asarray(rows["supervise_action"], dtype=bool).reshape(-1)
			if "supervise_action" in rows.files
			else np.ones(n, dtype=bool)
		)

	prev_action = np.vstack([CONTRACT_INITIAL_Q.astype(np.float32)[None, :], actions[:-1]])
	if gripper_cmds is None:
		gripper_cmds = prev_action[:, 5].astype(np.float32)

	frames: dict[str, Any] = {
		"observation.state": states,
		"observation.prev_action": prev_action,
		"observation.gripper_cmd": gripper_cmds[:, None],
		"action": actions,
		"segment.index": labels["index"],
		"segment.kind": labels["kind"],
		"segment.done": labels["done"],
		"segment.chunk_valid_len": labels["chunk_valid_len"],
		"segment.frame": labels["frame"],
		"segment.end_reason": labels["end_reason"],
		"segment.bowl": labels["bowl"],
		"segment.view": labels["view"],
		"segment.views": labels["views"],
		"mask.is_blind": is_blind,
		"mask.is_takeover": is_takeover,
		"mask.supervise_action": supervise_action,
		"task": labels["instruction"],
	}

	images: dict[str, np.ndarray] | None = None
	if load_video:
		import av

		from ..constants import TOP_CROP

		root = ep / "branch-0"
		images = {}
		x, y, w, h = TOP_CROP
		for stream_name, cam_key in (
			("top_rgb", "observation.images.top_rgb_v2"),
			("top_depth", "observation.images.top_depth_v2"),
			("wrist", "observation.images.wrist"),
		):
			with av.open(str(root / f"{stream_name}.mkv")) as container:
				buf = []
				for frame in container.decode(video=0):
					arr = frame.to_ndarray(format="rgb24")
					if stream_name == "top_rgb":
						arr = arr[y : y + h, x : x + w].copy()
					buf.append(arr)
				images[cam_key] = np.stack(buf, axis=0)

	out: dict[str, Any] = {
		"episode_index": int(episode_index),
		"segments": labels["segments"],
		"frames": frames,
	}
	if images is not None:
		out["images"] = images
	return out
