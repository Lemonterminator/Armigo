"""Generate deterministic v3 scene suites and preflight validation reports (Task 8).

Each suite contains cases formatted ``[seed, positive, face, {"scene": "v3", "split": <split>}]`` with
disjoint per-split seed ranges that never overlap legacy ``make_suites.py`` ranges or ``config/agent_bench``.
Before any episode is collected, static checks (reachability, top-camera visibility, colour separability),
balance/factor distribution tables, SmolVLM2 instruction token-length checks, and offscreen MuJoCo
perception checks are written to ``preflight.json`` and ``preflight.md``.
"""

from __future__ import annotations

import argparse
import colorsys
import dataclasses
import importlib.util
import itertools
import json
import math
import os
import re
import statistics
import subprocess
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
	sys.path.insert(0, str(ROOT / "src"))

from digital_twin_armigo.agent import perception as pc
from digital_twin_armigo.agent.twin_mujoco import INITIAL_Q, load_suite, sample_case
from digital_twin_armigo.constants import BOWL_R, BRICK_COLOURS, DUPLO_2X4, SO101_MJCF_DIR, TOP_FULL
from digital_twin_armigo.mujoco_cell import handeye as he
from digital_twin_armigo.mujoco_cell.cell import Cell
from digital_twin_armigo.mujoco_cell.grasp_geometry import Kinematics
from digital_twin_armigo.mujoco_cell.handeye_rig import calibrate_layout
from digital_twin_armigo.mujoco_cell.inspection_views import detector_crop
from digital_twin_armigo.mujoco_cell.layout import release_point
from digital_twin_armigo.mujoco_cell.orbit_scan import OrbitParams, plan_orbit_scan
from digital_twin_armigo.mujoco_cell.palette import HELD_OUT_PAIRS, PALETTE, hue_gap
from digital_twin_armigo.mujoco_cell.scene_v3 import _CAM_HALF_WIDTHS, SPLITS
from digital_twin_armigo.mujoco_cell.table_relocation import plan_table_scan
from digital_twin_armigo.mujoco_cell.visibility import workspace_visible
from digital_twin_armigo.vla.contract import (
	HELD_OUT_VARIANT,
	INSTRUCTION_VARIANTS,
	TRAIN_VARIANTS,
	format_instruction,
)


def _load_make_suites():
	path = Path(__file__).with_name("make_suites.py")
	spec = importlib.util.spec_from_file_location("_make_suites_shared", path)
	mod = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(mod)
	return mod


_MS = _load_make_suites()

DEFAULT_MIX: tuple[int, ...] = (3, 1, 1, 1, 1, 1, 1)
COLOUR_NAMES: dict[tuple[float, float, float], str] = dict(
	zip(BRICK_COLOURS, ("orange", "yellow", "blue", "red", "lime"), strict=True)
)
V3_SPLIT_SEED_RANGES: dict[str, tuple[int, int]] = {
	"train": (3_000_001, 3_099_999),
	"val": (3_100_001, 3_199_999),
	"ood-color": (3_200_001, 3_299_999),
	"ood-cam": (3_300_001, 3_399_999),
	"ood-lang": (3_400_001, 3_499_999),
	"ood-appearance": (3_500_001, 3_599_999),
	"test": (3_900_001, 3_999_999),
}
TOKEN_LIMIT = 40
SMOLVLM_MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
LEROBOT_CUDA_PYTHON = Path("E:/il/lerobot-cuda/Scripts/python.exe")

PERCEPTION_THRESHOLDS: dict[str, float | int] = {
	"bowl_median_mm_max": 10.0,
	"per_colour_median_mm_max": 10.0,
	"bowl_localization_rate_min": 0.70,
	"brick_median_mm_max": 10.0,
	"case_brick_error_mm_max": 20.0,
	"crashes_max": 0,
}

BALANCE_STRICT_MIN_EPISODES = 160
BALANCE_MAX_TARGET_DEV_FRAC = 0.15
BALANCE_MAX_ABS_CORR = 0.05

HUE_FP_BINS: tuple[tuple[str, float, float], ...] = (
	("0_60", 0.0, 60.0),
	("60_120", 60.0, 120.0),
	("120_180", 120.0, 180.0),
	("180_240", 180.0, 240.0),
	("240_300", 240.0, 300.0),
	("300_360", 300.0, 360.0),
)
HUE_FP_DEFAULT_PER_BIN = 10
HUE_FP_MAX_DELTA = 0.05
HUE_FP_GREY_RGB: tuple[float, float, float] = (0.60, 0.60, 0.60)


def parse_seed_range(spec: str, *, split: str | None = None) -> tuple[int, int]:
	"""Parse inclusive ``START-END`` seed range and validate against reserved and split boundaries."""
	m = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", spec)
	if not m:
		raise ValueError(f"Invalid --seeds {spec!r}; expected START-END (e.g. 3000001-3000050)")
	start, end = int(m.group(1)), int(m.group(2))
	if start > end:
		raise ValueError(f"Invalid --seeds {spec!r}: start ({start}) > end ({end})")
	for lo, hi in _MS.RESERVED_RANGES:
		if not (end < lo or start > hi):
			raise ValueError(f"Seed range {start}-{end} overlaps reserved range {lo}-{hi}")
	bench = _MS.bench_seeds()
	overlap_bench = [s for s in bench if start <= s <= end]
	if overlap_bench:
		raise ValueError(f"Seed range {start}-{end} overlaps config/agent_bench seeds: {overlap_bench[:5]}")
	if split is not None:
		if split not in V3_SPLIT_SEED_RANGES:
			raise ValueError(f"Unknown split {split!r}; expected one of {SPLITS}")
		s_lo, s_hi = V3_SPLIT_SEED_RANGES[split]
		if start < s_lo or end > s_hi:
			raise ValueError(f"Seed range {start}-{end} is out of bounds for split {split!r} (allowed {s_lo}-{s_hi})")
	return start, end


def check_reachability(layout) -> tuple[bool, str | None]:
	"""Verify bowl/brick reachability and pairwise clearance constraints."""
	pts = [layout.green, layout.pink, layout.brick_xy]
	if math.hypot(*layout.green) <= 0.14 or math.hypot(*layout.pink) <= 0.14:
		return False, "bowl too close to robot base"
	if not (0.14 < math.hypot(*layout.brick_xy) < 0.40):
		return False, "brick outside reach bounds"
	if math.hypot(*release_point(layout.green)) >= 0.40 or math.hypot(*release_point(layout.pink)) >= 0.40:
		return False, "bowl release point outside 0.40 m reach"
	if math.hypot(pts[0][0] - pts[1][0], pts[0][1] - pts[1][1]) <= 2 * BOWL_R - 0.01:
		return False, "green and pink bowls overlap"
	for b in (layout.green, layout.pink):
		if math.hypot(b[0] - layout.brick_xy[0], b[1] - layout.brick_xy[1]) <= BOWL_R + 0.03:
			return False, "brick too close to bowl"
	for _, ex, ey, _ in layout.extra_bowls:
		if math.hypot(ex, ey) <= 0.14 or math.hypot(*release_point((ex, ey))) >= 0.40:
			return False, "extra bowl outside reach bounds"
		if math.hypot(ex - layout.brick_xy[0], ey - layout.brick_xy[1]) <= BOWL_R + 0.03:
			return False, "extra bowl too close to brick"
		for bx, by in (layout.green, layout.pink):
			if math.hypot(ex - bx, ey - by) <= 2 * BOWL_R - 0.01:
				return False, "extra bowl overlaps primary bowl"
	return True, None


def check_visibility(layout) -> tuple[bool, str | None]:
	"""Verify workspace corners, bowls, and brick project inside the top camera margin."""
	ok, out_of_view = workspace_visible(layout)
	if not ok:
		return False, f"out of top camera view: {out_of_view}"
	return True, None


def _circular_hue_deg(a_deg: float, b_deg: float) -> float:
	d = abs(a_deg - b_deg) % 360.0
	return min(d, 360.0 - d)


def check_colour_separability(layout, split: str) -> tuple[bool, str | None]:
	"""Verify palette hue separation, split held-out pair rules, and table/bowl contrast."""
	if not layout.bowl_names or not layout.bowl_rgb:
		return False, "missing v3 bowl colour metadata"
	names = [layout.bowl_names["green"], layout.bowl_names["pink"], *(name for name, _, _, _ in layout.extra_bowls)]
	if len(set(names)) != len(names):
		return False, f"duplicate bowl colour names: {names}"

	for a, b in itertools.combinations(names, 2):
		if hue_gap(a, b) < 35.0:
			return False, f"bowl pair ({a}, {b}) hue gap < 35 deg"

	primary_pair = frozenset((layout.bowl_names["green"], layout.bowl_names["pink"]))
	if split == "ood-color":
		if primary_pair not in HELD_OUT_PAIRS:
			return False, f"ood-color primary pair {sorted(primary_pair)} not in HELD_OUT_PAIRS"
	else:
		for a, b in itertools.combinations(names, 2):
			if frozenset((a, b)) in HELD_OUT_PAIRS:
				return False, f"non-ood-color split {split!r} contains held-out pair ({a}, {b})"

	h_t, s_t, v_t = colorsys.rgb_to_hsv(*layout.table_rgb)
	if not (0.35 - 1e-6 <= v_t <= 0.80 + 1e-6):
		return False, f"table V={v_t:.3f} outside [0.35, 0.80]"
	if split == "ood-appearance":
		if not (0.35 - 1e-6 <= s_t <= 0.55 + 1e-6):
			return False, f"ood-appearance table sat={s_t:.3f} outside [0.35, 0.55]"
	elif s_t > 0.35 + 1e-6:
		return False, f"table sat={s_t:.3f} exceeds 0.35"
	if "white" in names and v_t > 0.68 + 1e-6:
		return False, f"white bowl present with bright table V={v_t:.3f} > 0.68"
	if "black" in names and v_t < 0.40 - 1e-6:
		return False, f"black bowl present with dark table V={v_t:.3f} < 0.40"
	if s_t > 0.15:
		h_t_deg = h_t * 360.0
		slots_rgb = [
			(layout.bowl_names["green"], layout.bowl_rgb["green"]),
			(layout.bowl_names["pink"], layout.bowl_rgb["pink"]),
			*((name, ergb) for name, _, _, ergb in layout.extra_bowls),
		]
		for name, brgb in slots_rgb:
			bh = PALETTE[name].hue_deg
			if bh is None:
				continue
			if _circular_hue_deg(h_t_deg, bh) < 35.0:
				return False, f"saturated table hue within 35 deg of {name}"
			jh = colorsys.rgb_to_hsv(*brgb)[0] * 360.0
			if _circular_hue_deg(h_t_deg, jh) < 35.0:
				return False, f"saturated table hue within 35 deg of jittered {name}"
	return True, None


def _classify_sample_error(exc: Exception) -> str:
	msg = str(exc).lower()
	if "workspace_visible" in msg or "visible" in msg:
		return "visibility"
	if "colour" in msg or "color" in msg or "hue" in msg or "table" in msg:
		return "colour_separability"
	return "reachability"


def evaluate_candidate_seed(seed: int, positive: bool, face: int, split: str):
	"""Sample a v3 layout and run static checks, returning ``(layout, None)`` or ``(None, rejection_reason)``."""
	opts = {"scene": "v3", "split": split}
	try:
		layout = sample_case(seed, positive, face, opts)
	except RuntimeError as exc:
		return None, _classify_sample_error(exc)

	ok_reach, _ = check_reachability(layout)
	if not ok_reach:
		return None, "reachability"
	ok_vis, _ = check_visibility(layout)
	if not ok_vis:
		return None, "visibility"
	ok_col, _ = check_colour_separability(layout, split)
	if not ok_col:
		return None, "colour_separability"
	return layout, None


def _stats(values: list[float]) -> dict[str, float]:
	v = np.asarray(values, dtype=np.float64)
	q = np.percentile(v, [5, 50, 95])
	return {
		"min": round(float(v.min()), 4),
		"p5": round(float(q[0]), 4),
		"median": round(float(q[1]), 4),
		"p95": round(float(q[2]), 4),
		"max": round(float(v.max()), 4),
		"mean": round(float(v.mean()), 4),
		"std": round(float(v.std()), 4),
	}


def _hist(values: list[float], lo: float, hi: float, bins: int) -> dict[str, Any]:
	counts, edges = np.histogram(np.clip(values, lo, hi), bins=bins, range=(lo, hi))
	return {"edges": [round(float(e), 4) for e in edges], "counts": counts.tolist()}


def _bowl_side(xy: tuple[float, float]) -> str:
	if xy[1] > 0.03:
		return "left"
	if xy[1] < -0.03:
		return "right"
	return "center"


def _instruction_specs(variants: tuple[int, ...], colours: tuple[str, ...]) -> list[tuple[int, str, str]]:
	"""Return ``(variant, colour_or_empty, formatted_text)`` for all instructions in scope."""
	specs: list[tuple[int, str, str]] = []
	for kind in INSTRUCTION_VARIANTS:
		for v in variants:
			if kind == "carry_to_bowl":
				for col in colours:
					specs.append((v, col, format_instruction(kind, {"bowl": col}, variant=v)))
			elif kind == "scan_view":
				specs.append((v, "", format_instruction(kind, {"view": 4, "views": 4, "roll_deg": -135.0}, variant=v)))
				specs.append((v, "", format_instruction(kind, {"view": 4, "views": 4}, variant=v)))
			elif kind in ("grasp", "present_bottom"):
				specs.append((v, "", format_instruction(kind, {"roll_deg": -135.0}, variant=v)))
				specs.append((v, "", format_instruction(kind, {}, variant=v)))
			else:
				specs.append((v, "", format_instruction(kind, {}, variant=v)))
	return specs


@lru_cache(maxsize=1)
def _load_full_token_table() -> tuple[str, dict[tuple[int, str, str], int]] | None:
	"""Load SmolVLM2 token lengths across all variants/colours once via local ``transformers`` or ``lerobot-cuda``."""
	all_variants = tuple(sorted({*TRAIN_VARIANTS, HELD_OUT_VARIANT}))
	all_colours = tuple(sorted(PALETTE))
	specs = _instruction_specs(all_variants, all_colours)
	texts = [t for _, _, t in specs]

	try:
		from transformers import AutoTokenizer

		tok = AutoTokenizer.from_pretrained(SMOLVLM_MODEL_ID, local_files_only=True)
		counts = [len(tok(t)["input_ids"]) for t in texts]
		return "transformers", dict(zip(specs, counts, strict=True))
	except (ImportError, OSError, ValueError, RuntimeError):
		pass

	if LEROBOT_CUDA_PYTHON.exists():
		script = (
			"import json, sys\n"
			"from transformers import AutoTokenizer\n"
			f"tok = AutoTokenizer.from_pretrained({SMOLVLM_MODEL_ID!r}, local_files_only=True)\n"
			"texts = json.loads(sys.stdin.read())\n"
			"print(json.dumps([len(tok(t)['input_ids']) for t in texts]))\n"
		)
		env = os.environ.copy()
		env["CUDA_VISIBLE_DEVICES"] = ""
		try:
			proc = subprocess.run(
				[str(LEROBOT_CUDA_PYTHON), "-c", script],
				input=json.dumps(texts),
				text=True,
				capture_output=True,
				env=env,
				timeout=25,
				check=True,
			)
			counts = [int(x) for x in json.loads(proc.stdout.strip().splitlines()[-1])]
			if len(counts) == len(specs):
				return "lerobot-cuda", dict(zip(specs, counts, strict=True))
		except (OSError, subprocess.SubprocessError, ValueError, KeyError, IndexError):
			pass

	return None


@lru_cache(maxsize=16)
def _measure_tokens_cached(variants: tuple[int, ...], colours: tuple[str, ...]) -> tuple[int, str, str]:
	"""Return ``(max_tokens, backend, worst_instruction)`` across segment kinds, variants, and colours."""
	specs = _instruction_specs(variants, colours)
	loaded = _load_full_token_table()
	if loaded is not None:
		backend, table = loaded
		counts = [table[s] for s in specs]
		idx = int(np.argmax(counts))
		return int(counts[idx]), backend, specs[idx][2]

	texts = [t for _, _, t in specs]
	counts = [len(re.findall(r"[A-Za-z]+|\d|[^\w\s]", t)) + 1 for t in texts]
	idx = int(np.argmax(counts))
	return int(counts[idx]), "subword_estimate", texts[idx]


def run_perception_check(
	cases: list[list],
	layouts: list,
	*,
	perception_sample: float = 0.1,
	max_perception: int | None = None,
	sensing: str = "realistic",
	sampled_indices: list[int] | None = None,
) -> dict[str, Any]:
	"""Render sampled cases offscreen in MuJoCo and run ``locate_bowls`` + ``locate_brick`` without truth cheat."""
	if sensing not in ("truth", "realistic"):
		raise ValueError(f"sensing must be 'truth' or 'realistic', got {sensing!r}")

	if sampled_indices is not None:
		chosen_indices = sorted(int(i) for i in sampled_indices)
		sample_n = len(chosen_indices)
	elif perception_sample <= 0.0 or not cases:
		chosen_indices = []
		sample_n = 0
	else:
		sample_n = min(len(cases), math.ceil(float(perception_sample) * len(cases)))
		if max_perception is not None:
			sample_n = min(sample_n, max(0, int(max_perception)))
		if sample_n > 0:
			rng = np.random.default_rng([int(cases[0][0]), len(cases), 0x9E2C])
			chosen_indices = sorted(int(i) for i in rng.choice(len(cases), size=sample_n, replace=False))
		else:
			chosen_indices = []

	if sample_n == 0:
		return {
			"sample_fraction": float(perception_sample),
			"max_perception": max_perception,
			"sampled_count": 0,
			"sampled_indices": [],
			"sensing": sensing,
			"camera_calibrated": sensing == "realistic",
			"thresholds": dict(PERCEPTION_THRESHOLDS),
			"crashes": 0,
			"bowls_total": 0,
			"bowls_found": 0,
			"bowl_localization_rate": 1.0,
			"bowl_median_error_mm": None,
			"per_colour_stats": {},
			"bricks_total": 0,
			"bricks_found": 0,
			"brick_localization_rate": 1.0,
			"brick_median_error_mm": None,
			"brick_all_cases_median_error_mm": None,
			"brick_max_error_mm": None,
			"cases_with_missed_bowls": 0,
			"failed_case_indices": [],
			"failed_seeds": [],
			"cases": [],
			"pass": True,
		}

	cfg = pc.PerceptionCfg.realistic() if sensing == "realistic" else None
	crashes = 0
	bowls_total = 0
	bowls_found = 0
	all_bowl_errors_mm: list[float] = []
	errors_by_colour: dict[str, list[float]] = {}
	totals_by_colour: Counter[str] = Counter()
	found_by_colour: Counter[str] = Counter()
	brick_errors_clean_mm: list[float] = []
	brick_errors_all_mm: list[float] = []
	bricks_found = 0
	per_case_records: list[dict[str, Any]] = []
	case_brick_max_mm = float(PERCEPTION_THRESHOLDS["case_brick_error_mm_max"])

	for idx in chosen_indices:
		case = cases[idx]
		lay = layouts[idx]
		slot_to_colour = {
			"green": lay.bowl_names["green"],
			"pink": lay.bowl_names["pink"],
			**{f"extra:{i}": name for i, (name, _, _, _) in enumerate(lay.extra_bowls)},
		}
		for col in slot_to_colour.values():
			totals_by_colour[col] += 1
			bowls_total += 1

		cell = None
		try:
			cell = Cell(lay, SO101_MJCF_DIR)
			cell.set_joints(INITIAL_Q)
			r = cell.renderer
			r.enable_depth_rendering()
			r.update_scene(cell.data, "top")
			depth = r.render().astype(np.float64)
			r.disable_depth_rendering()
			rgb = cell.rgb("top")

			if sensing == "realistic":
				k, cam, _ = calibrate_layout(lay, int(lay.defect_seed), cfg.calibration)
				depth_rng = np.random.default_rng(int(lay.defect_seed) + 30_000)
				sigma = cfg.depth_noise_m * (depth / 0.5) ** 2
				depth = depth + depth_rng.normal(0.0, 1.0, depth.shape) * sigma
			else:
				k = he.intrinsics(lay.top, TOP_FULL)
				cam = he.camera_truth(cell.model, cell.data, "top")

			rgb_by_slot = {
				**lay.bowl_rgb,
				**{f"extra:{i}": crgb for i, (_, _, _, crgb) in enumerate(lay.extra_bowls)},
			}
			found_bowls = pc.locate_bowls(rgb, depth, k, cam, bands=pc.bands_for(rgb_by_slot, engine="mujoco"))
			bowl_pts = [est.xy for est in found_bowls.values() if est is not None]
			brick_est = pc.locate_brick(depth, k, cam, bowl_pts)

			bowl_errors_mm: dict[str, float] = {}
			missed_bowls: list[str] = []
			for slot, col in slot_to_colour.items():
				est = found_bowls.get(slot)
				if est is None:
					missed_bowls.append(f"{slot}({col})")
					continue
				if slot.startswith("extra:"):
					ex_idx = int(slot.split(":", 1)[1])
					truth = np.asarray(lay.extra_bowls[ex_idx][1:3], dtype=np.float64)
				else:
					truth = np.asarray(getattr(lay, slot), dtype=np.float64)
				err_mm = round(float(np.linalg.norm(est.xy - truth) * 1000.0), 3)
				bowl_errors_mm[slot] = err_mm
				all_bowl_errors_mm.append(err_mm)
				errors_by_colour.setdefault(col, []).append(err_mm)
				found_by_colour[col] += 1
				bowls_found += 1

			brick_err_mm = None
			if brick_est is not None:
				bricks_found += 1
				brick_err_mm = round(float(np.linalg.norm(brick_est.pos[:2] - lay.brick_xy) * 1000.0), 3)
				brick_errors_all_mm.append(brick_err_mm)
				if not missed_bowls:
					brick_errors_clean_mm.append(brick_err_mm)

			failure_reasons: list[str] = []
			if missed_bowls:
				failure_reasons.append("missed_bowls")
			if brick_est is None:
				failure_reasons.append("brick_not_found")
			elif brick_err_mm is not None and brick_err_mm > case_brick_max_mm:
				failure_reasons.append("brick_error_exceeds_max")

			per_case_records.append(
				{
					"index": idx,
					"seed": int(case[0]),
					"light_level": round(float(lay.light_level), 4),
					"bowls": slot_to_colour,
					"bowl_errors_mm": bowl_errors_mm,
					"missed_bowls": missed_bowls,
					"brick_found": brick_est is not None,
					"brick_error_mm": brick_err_mm,
					"passed": len(failure_reasons) == 0,
					"failure_reasons": failure_reasons,
					"error": None,
				}
			)
		except (RuntimeError, ValueError, TypeError, OSError, KeyError) as exc:
			crashes += 1
			per_case_records.append(
				{
					"index": idx,
					"seed": int(case[0]),
					"light_level": round(float(lay.light_level), 4),
					"bowls": slot_to_colour,
					"bowl_errors_mm": {},
					"missed_bowls": list(slot_to_colour),
					"brick_found": False,
					"brick_error_mm": None,
					"passed": False,
					"failure_reasons": ["crash"],
					"error": f"{type(exc).__name__}: {exc}",
				}
			)
		finally:
			if cell is not None:
				cell.close()

	bowl_loc_rate = round(bowls_found / bowls_total, 4) if bowls_total else 1.0
	brick_loc_rate = round(bricks_found / len(chosen_indices), 4) if chosen_indices else 1.0
	bowl_med = round(float(statistics.median(all_bowl_errors_mm)), 3) if all_bowl_errors_mm else None
	brick_med_clean = round(float(statistics.median(brick_errors_clean_mm)), 3) if brick_errors_clean_mm else None
	brick_med_all = round(float(statistics.median(brick_errors_all_mm)), 3) if brick_errors_all_mm else None
	brick_max_mm = round(float(max(brick_errors_all_mm)), 3) if brick_errors_all_mm else None

	per_colour_stats: dict[str, dict[str, Any]] = {}
	per_colour_medians_ok = True
	for col in sorted(totals_by_colour):
		total_c = int(totals_by_colour[col])
		found_c = int(found_by_colour[col])
		errs = errors_by_colour.get(col, [])
		med = round(float(statistics.median(errs)), 3) if errs else None
		mx = round(float(max(errs)), 3) if errs else None
		if total_c == 0:
			status = "untested"
		elif found_c == 0 or med is None or med > float(PERCEPTION_THRESHOLDS["per_colour_median_mm_max"]):
			status = "failed"
			per_colour_medians_ok = False
		else:
			status = "passed"
		per_colour_stats[col] = {
			"total": total_c,
			"found": found_c,
			"missed": total_c - found_c,
			"median_error_mm": med,
			"max_error_mm": mx,
			"status": status,
		}

	failed_case_indices = [rec["index"] for rec in per_case_records if not rec["passed"]]
	failed_seeds = [rec["seed"] for rec in per_case_records if not rec["passed"]]
	cases_with_missed_bowls = sum(1 for rec in per_case_records if rec["missed_bowls"])

	passed = (
		crashes <= int(PERCEPTION_THRESHOLDS["crashes_max"])
		and bowl_loc_rate >= float(PERCEPTION_THRESHOLDS["bowl_localization_rate_min"])
		and bowl_med is not None
		and bowl_med <= float(PERCEPTION_THRESHOLDS["bowl_median_mm_max"])
		and per_colour_medians_ok
		and brick_med_clean is not None
		and brick_med_clean <= float(PERCEPTION_THRESHOLDS["brick_median_mm_max"])
		and brick_med_all is not None
		and brick_med_all <= float(PERCEPTION_THRESHOLDS["brick_median_mm_max"])
		and len(failed_case_indices) == 0
	)

	return {
		"sample_fraction": float(perception_sample),
		"max_perception": max_perception,
		"sampled_count": len(chosen_indices),
		"sampled_indices": chosen_indices,
		"sensing": sensing,
		"camera_calibrated": sensing == "realistic",
		"thresholds": dict(PERCEPTION_THRESHOLDS),
		"crashes": crashes,
		"bowls_total": bowls_total,
		"bowls_found": bowls_found,
		"bowl_localization_rate": bowl_loc_rate,
		"bowl_median_error_mm": bowl_med,
		"per_colour_stats": per_colour_stats,
		"bricks_total": len(chosen_indices),
		"bricks_found": bricks_found,
		"brick_localization_rate": brick_loc_rate,
		"brick_median_error_mm": brick_med_clean,
		"brick_all_cases_median_error_mm": brick_med_all,
		"brick_max_error_mm": brick_max_mm,
		"cases_with_missed_bowls": cases_with_missed_bowls,
		"failed_case_indices": failed_case_indices,
		"failed_seeds": failed_seeds,
		"cases": per_case_records,
		"pass": bool(passed),
	}


def _score_and_threshold(scorer, camera: str, img: np.ndarray) -> tuple[float, float]:
	"""Score a crop with either an ``AnomalyScorer`` object or a plain ``(camera, img) -> float`` callable."""
	if hasattr(scorer, "score") and hasattr(scorer, "threshold"):
		return float(scorer.score(camera, img)), float(scorer.threshold(camera))
	if callable(scorer):
		return float(scorer(camera, img)), 0.0
	raise TypeError("scorer must implement score(camera, img) + threshold(camera) or be callable")


def _plan_wrist_sweep_qs(cell: Cell, n_views: int) -> list[np.ndarray]:
	"""Compute up to ``n_views`` kinematic teacher wrist-sweep poses viewing the brick on the table."""
	kin = Kinematics(cell)
	plan = plan_orbit_scan(
		cell,
		kin,
		INITIAL_Q,
		gripper=float(INITIAL_Q[5]),
		params=OrbitParams(tip_heights_m=(0.025, 0.04), standoffs_m=(0.03,), azimuths=12, restarts=2),
		max_views=max(1, int(n_views)),
	)
	qs = [v.q.copy() for v in plan.views[: max(1, int(n_views))]]
	if not qs:
		views, _ = plan_table_scan(
			cell,
			kin,
			INITIAL_Q,
			gripper=float(INITIAL_Q[5]),
			heights=(0.075,),
			rolls=8,
			max_views=max(1, int(n_views)),
			restarts=4,
		)
		qs = [v.q.copy() for v in views[: max(1, int(n_views))]]
	return qs


def _table_hue_for_bin(
	layout, lo_deg: float, hi_deg: float, rng: np.random.Generator
) -> tuple[float, float, float] | None:
	"""Pick a valid saturated table RGB in ``[lo_deg, hi_deg)`` respecting all bowl hue/value constraints."""
	names = [layout.bowl_names["green"], layout.bowl_names["pink"], *(b[0] for b in layout.extra_bowls)]
	bowl_hues: list[float] = []
	slots_rgb = [
		(layout.bowl_names["green"], layout.bowl_rgb["green"]),
		(layout.bowl_names["pink"], layout.bowl_rgb["pink"]),
		*((b[0], b[3]) for b in layout.extra_bowls),
	]
	for name, brgb in slots_rgb:
		if PALETTE[name].hue_deg is None:
			continue
		bowl_hues.append(float(PALETTE[name].hue_deg))
		bowl_hues.append(colorsys.rgb_to_hsv(*brgb)[0] * 360.0)
	v_lo = 0.42 if "black" in names else 0.40
	v_hi = 0.66 if "white" in names else 0.72
	for _ in range(80):
		h_deg = float(rng.uniform(lo_deg + 1.0, hi_deg - 1.0))
		if all(_circular_hue_deg(h_deg, bh) >= 35.0 for bh in bowl_hues):
			sat = float(rng.uniform(0.22, 0.34))
			val = float(rng.uniform(v_lo, v_hi))
			r, g, b = colorsys.hsv_to_rgb(h_deg / 360.0, sat, val)
			return (float(np.clip(r, 0.0, 1.0)), float(np.clip(g, 0.0, 1.0)), float(np.clip(b, 0.0, 1.0)))
	return None


def _empty_hue_bins() -> dict[str, dict[str, Any]]:
	return {
		name: {
			"hue_range_deg": [lo, hi],
			"cases": 0,
			"false_positives": 0,
			"fp_rate": None,
			"delta_vs_grey": None,
			"pass": False,
		}
		for name, lo, hi in HUE_FP_BINS
	}


def run_hue_fp_control(
	cases: list[list],
	layouts: list,
	*,
	scorer=None,
	n_per_bin: int = HUE_FP_DEFAULT_PER_BIN,
	grey_rgb: tuple[float, float, float] = HUE_FP_GREY_RGB,
	max_delta: float = HUE_FP_MAX_DELTA,
	crop_size: int = 256,
	n_views: int = 2,
	skip: bool = False,
) -> dict[str, Any]:
	"""Plan §5(2) wrist-sweep false-positive control across 6 table hue bins (60 good bricks) vs grey baseline."""
	target_cases = int(n_per_bin) * len(HUE_FP_BINS)
	if skip:
		return {
			"status": "skipped",
			"pass": True,
			"target_cases": target_cases,
			"max_delta_fp_rate": float(max_delta),
			"grey_baseline": None,
			"bins": _empty_hue_bins(),
			"failing_bins": [],
			"reason": "explicitly skipped via skip_hue_fp=True / --skip-hue-fp",
		}
	if scorer is None:
		return {
			"status": "requires_detector",
			"pass": False,
			"target_cases": target_cases,
			"max_delta_fp_rate": float(max_delta),
			"grey_baseline": None,
			"bins": _empty_hue_bins(),
			"failing_bins": [],
			"reason": "no detector scorer provided; pass --detector or scorer=... to run wrist-sweep hue FP control",
		}

	good_pool = [(c, lay) for c, lay in zip(cases, layouts, strict=True) if not c[1]]
	if not good_pool:
		split = cases[0][3].get("split", "train") if cases and len(cases[0]) >= 4 else "train"
		lo_seed = V3_SPLIT_SEED_RANGES.get(split, V3_SPLIT_SEED_RANGES["train"])[0]
		s_cur = lo_seed
		while len(good_pool) < max(12, int(n_per_bin) * 2):
			lay_cand, _ = evaluate_candidate_seed(s_cur, False, 0, split)
			if lay_cand is not None:
				good_pool.append(([s_cur, False, 0, {"scene": "v3", "split": split}], lay_cand))
			s_cur += 1

	rng = np.random.default_rng([int(good_pool[0][0][0]), len(good_pool), 0x4E50])
	bin_layouts: dict[str, list[tuple[int, Any]]] = {name: [] for name, _, _ in HUE_FP_BINS}

	# 1. Assign suite good-brick cases whose saturated table already lies in a bin.
	for case, lay in good_pool:
		h_t, s_t, _ = colorsys.rgb_to_hsv(*lay.table_rgb)
		if s_t <= 0.15:
			continue
		bin_idx = int((h_t * 360.0) % 360.0 // 60.0)
		bname = HUE_FP_BINS[bin_idx][0]
		if len(bin_layouts[bname]) < int(n_per_bin):
			bin_layouts[bname].append((int(case[0]), lay))

	# 2. Fill any remaining slots per bin deterministically by re-tinting good-brick layouts into that hue bin.
	for bname, lo_deg, hi_deg in HUE_FP_BINS:
		pool_idx = 0
		attempts = 0
		while len(bin_layouts[bname]) < int(n_per_bin) and attempts < len(good_pool) * 4 + 40:
			case, base_lay = good_pool[pool_idx % len(good_pool)]
			pool_idx += 1
			attempts += 1
			new_rgb = _table_hue_for_bin(base_lay, lo_deg, hi_deg, rng)
			if new_rgb is None:
				continue
			lay_mod = dataclasses.replace(base_lay, table_rgb=new_rgb)
			if getattr(base_lay, "_scene_v3", None) is not None:
				lay_mod._scene_v3 = dataclasses.replace(base_lay._scene_v3, table_rgb=new_rgb)
			bin_layouts[bname].append((int(case[0]), lay_mod))

	grey_total = 0
	grey_fps = 0
	bins_report: dict[str, dict[str, Any]] = {}
	raw_bin_counts: dict[str, tuple[int, int]] = {}

	for bname, lo_deg, hi_deg in HUE_FP_BINS:
		b_total = 0
		b_fps = 0
		for _seed, lay in bin_layouts[bname]:
			cell = Cell(lay, SO101_MJCF_DIR, face_probes=True)
			try:
				cell.set_joints(INITIAL_Q)
				qs = _plan_wrist_sweep_qs(cell, int(n_views))
				if not qs:
					continue
				col_alarm = False
				for q in qs:
					cell.set_joints(q)
					crop, _ = detector_crop(cell, "wrist_cam", None, cell.data, int(crop_size))
					if crop is None:
						continue
					score, thr = _score_and_threshold(scorer, "wrist_cam", crop)
					if score >= thr:
						col_alarm = True
			finally:
				cell.close()

			grey_lay = dataclasses.replace(lay, table_rgb=grey_rgb)
			if getattr(lay, "_scene_v3", None) is not None:
				grey_lay._scene_v3 = dataclasses.replace(lay._scene_v3, table_rgb=grey_rgb)
			grey_cell = Cell(grey_lay, SO101_MJCF_DIR, face_probes=True)
			try:
				grey_alarm = False
				for q in qs:
					grey_cell.set_joints(q)
					crop, _ = detector_crop(grey_cell, "wrist_cam", None, grey_cell.data, int(crop_size))
					if crop is None:
						continue
					score, thr = _score_and_threshold(scorer, "wrist_cam", crop)
					if score >= thr:
						grey_alarm = True
			finally:
				grey_cell.close()

			b_total += 1
			grey_total += 1
			if col_alarm:
				b_fps += 1
			if grey_alarm:
				grey_fps += 1
		raw_bin_counts[bname] = (b_total, b_fps)

	grey_fp_rate = round(grey_fps / grey_total, 4) if grey_total > 0 else None
	failing_bins: list[str] = []
	for bname, lo_deg, hi_deg in HUE_FP_BINS:
		b_total, b_fps = raw_bin_counts[bname]
		fp_rate = round(b_fps / b_total, 4) if b_total > 0 else None
		delta = round(fp_rate - grey_fp_rate, 4) if (fp_rate is not None and grey_fp_rate is not None) else None
		bin_ok = b_total > 0 and delta is not None and delta <= float(max_delta) + 1e-9
		if not bin_ok:
			failing_bins.append(bname)
		bins_report[bname] = {
			"hue_range_deg": [lo_deg, hi_deg],
			"cases": b_total,
			"false_positives": b_fps,
			"fp_rate": fp_rate,
			"delta_vs_grey": delta,
			"pass": bool(bin_ok),
		}

	passed = grey_total > 0 and len(failing_bins) == 0
	return {
		"status": "passed" if passed else "failed",
		"pass": bool(passed),
		"target_cases": target_cases,
		"evaluated_cases": grey_total,
		"max_delta_fp_rate": float(max_delta),
		"grey_baseline": {
			"rgb": list(grey_rgb),
			"cases": grey_total,
			"false_positives": grey_fps,
			"fp_rate": grey_fp_rate,
		},
		"bins": bins_report,
		"failing_bins": failing_bins,
	}


def _target_strata_quotas(n: int) -> dict[tuple[str, bool], int]:
	"""Balanced ``(target_colour, is_left)`` quotas for ``n`` episodes achieving ``max_dev <= 0.15`` and ``|corr| < 0.05``."""
	colours = sorted(PALETTE)
	base, rem = divmod(n, len(colours))
	quotas: dict[tuple[str, bool], int] = {}
	for idx, col in enumerate(colours):
		cnt = base + (1 if idx < rem else 0)
		left_cnt = (cnt + 1) // 2 if (idx % 2 == 0) else cnt // 2
		quotas[(col, True)] = left_cnt
		quotas[(col, False)] = cnt - left_cnt
	return quotas


def _layout_target_stratum(layout, positive: bool) -> tuple[str, bool]:
	slot = "pink" if positive else "green"
	return layout.bowl_names[slot], bool(float(getattr(layout, slot)[1]) > 0.0)


def _compute_balance_and_factors(
	cases: list[list], layouts: list, split: str, *, strict_balance: bool | None = None
) -> dict[str, Any]:
	"""Compute balance metrics, colour pairs, camera histograms, visibility resamples, and factor tables."""
	n = len(cases)
	class_counts_raw = Counter("normal" if not c[1] else f"defect_face_{c[2]}" for c in cases)
	class_counts = {
		"normal": int(class_counts_raw.get("normal", 0)),
		**{f"defect_face_{f}": int(class_counts_raw.get(f"defect_face_{f}", 0)) for f in range(6)},
	}
	expected_alloc = _MS.allocate(n, DEFAULT_MIX)
	observed_alloc = [class_counts["normal"], *(class_counts[f"defect_face_{f}"] for f in range(6))]
	class_mix_ok = observed_alloc == expected_alloc

	target_colours: list[str] = []
	target_ys: list[float] = []
	normal_colours: Counter[str] = Counter()
	defect_colours: Counter[str] = Counter()
	extra_colours: Counter[str] = Counter()
	bowl_count_hist: Counter[str] = Counter()
	routing_counts: Counter[str] = Counter()
	brick_colours: Counter[str] = Counter()
	brick_sizes: Counter[str] = Counter()
	table_textures: Counter[str] = Counter()
	variants: Counter[str] = Counter()
	table_vs: list[float] = []
	table_sats: list[float] = []
	light_levels: list[float] = []
	light_ks: list[float] = []
	vis_resamples: list[int] = []
	pair_counts: Counter[str] = Counter()
	slot_pair_counts: Counter[str] = Counter()
	held_out_found: set[str] = set()
	held_out_primary_count = 0
	min_nominal_gap = 360.0
	min_chromatic_gap = 360.0

	crosstab: dict[str, dict[str, dict[str, int]]] = {
		col: {t: {s: 0 for s in ("left", "right", "center")} for t in ("target", "non_target")} for col in PALETTE
	}

	cam_series: dict[str, list[float]] = {k: [] for k in _CAM_HALF_WIDTHS}
	wrist_series: dict[str, list[float]] = {
		k: [] for k in ("wrist_x_m", "wrist_y_m", "wrist_z_m", "wrist_rx_deg", "wrist_ry_deg", "wrist_rz_deg")
	}

	for case, lay in zip(cases, layouts, strict=True):
		sc = lay._scene_v3
		target_slot = "pink" if case[1] else "green"
		target_colours.append(lay.bowl_names[target_slot])
		target_ys.append(float(getattr(lay, target_slot)[1]))

		normal_colours[sc.normal] += 1
		defect_colours[sc.defect] += 1
		extra_colours[sc.extra if sc.extra is not None else "none"] += 1
		bowl_count_hist[str(2 + len(lay.extra_bowls))] += 1
		routing_counts[f"normal:{sc.normal}|defect:{sc.defect}"] += 1

		brick_colours[COLOUR_NAMES[tuple(lay.brick_rgb)]] += 1
		brick_sizes["2x4" if tuple(lay.brick_size) == DUPLO_2X4 else "2x2"] += 1
		table_textures[sc.table_texture] += 1
		variants[str(sc.variant)] += 1

		_, s_t, v_t = colorsys.rgb_to_hsv(*sc.table_rgb)
		table_sats.append(float(s_t))
		table_vs.append(float(v_t))
		light_levels.append(float(lay.light_level))
		light_ks.append(float(sc.light_k))
		vis_resamples.append(int(getattr(lay, "_visibility_resamples", 0)))

		for k, val in sc.camera.items():
			cam_series[k].append(float(val))
		for k, val in zip(wrist_series, sc.wrist, strict=True):
			wrist_series[k].append(float(val))

		names = [sc.normal, sc.defect, *([sc.extra] if sc.extra is not None else [])]
		pair_key = "-".join(sorted((sc.normal, sc.defect)))
		pair_counts[pair_key] += 1
		slot_pair_counts[f"{sc.normal}->{sc.defect}"] += 1
		if frozenset((sc.normal, sc.defect)) in HELD_OUT_PAIRS:
			held_out_primary_count += 1
		for a, b in itertools.combinations(names, 2):
			if frozenset((a, b)) in HELD_OUT_PAIRS:
				held_out_found.add("-".join(sorted((a, b))))
			min_nominal_gap = min(min_nominal_gap, hue_gap(a, b))
			if PALETTE[a].hue_deg is not None and PALETTE[b].hue_deg is not None:
				ha = colorsys.rgb_to_hsv(*sc.rgb[a])[0] * 360.0
				hb = colorsys.rgb_to_hsv(*sc.rgb[b])[0] * 360.0
				min_chromatic_gap = min(min_chromatic_gap, _circular_hue_deg(ha, hb))

		slots_xy = [
			("green", sc.normal, lay.green),
			("pink", sc.defect, lay.pink),
			*((f"extra:{i}", name, (ex, ey)) for i, (name, ex, ey, _) in enumerate(lay.extra_bowls)),
		]
		for slot, col, xy in slots_xy:
			t_key = "target" if slot == target_slot else "non_target"
			crosstab[col][t_key][_bowl_side(xy)] += 1

	target_count_map = {col: sum(1 for tc in target_colours if tc == col) for col in sorted(PALETTE)}
	mean_target = n / len(PALETTE) if n else 0.0
	max_dev = max(abs(cnt - mean_target) / mean_target for cnt in target_count_map.values()) if mean_target > 0 else 0.0
	is_left = np.asarray([y > 0.0 for y in target_ys], dtype=np.float64)
	corrs: dict[str, float | None] = {}
	for col in sorted(PALETTE):
		ind = np.asarray([1.0 if tc == col else 0.0 for tc in target_colours], dtype=np.float64)
		if n < 2 or float(ind.std()) == 0.0 or float(is_left.std()) == 0.0:
			corrs[col] = 0.0
		else:
			corrs[col] = round(float(np.corrcoef(ind, is_left)[0, 1]), 4)
	max_abs_corr = max((abs(v) for v in corrs.values() if v is not None), default=0.0)
	enforce_strict = (n >= BALANCE_STRICT_MIN_EPISODES) if strict_balance is None else bool(strict_balance)
	all_colours_present = all(cnt > 0 for cnt in target_count_map.values()) if n >= 32 else True
	balance_pass = bool(
		class_mix_ok
		and all_colours_present
		and (not enforce_strict or (max_dev <= BALANCE_MAX_TARGET_DEV_FRAC and max_abs_corr < BALANCE_MAX_ABS_CORR))
	)

	colour_pairs_pass = (
		(held_out_primary_count == n) if split == "ood-color" else (len(held_out_found) == 0)
	) and min_nominal_gap >= 35.0

	camera_hist = {
		"top": {
			k: {
				**_stats(cam_series[k]),
				"hist": _hist(cam_series[k], -1.5 * hw, 1.5 * hw, 6),
			}
			for k, hw in _CAM_HALF_WIDTHS.items()
		},
		"wrist": {
			k: {
				**_stats(vals),
				"hist": _hist(vals, -0.005 if k.endswith("_m") else -5.0, 0.005 if k.endswith("_m") else 5.0, 5),
			}
			for k, vals in wrist_series.items()
		},
	}

	res_counts = Counter(str(r) for r in vis_resamples)
	visibility_resamples = {
		"total": int(sum(vis_resamples)),
		"max": int(max(vis_resamples, default=0)),
		"median": float(statistics.median(vis_resamples)) if vis_resamples else 0.0,
		"cases_resampled": int(sum(1 for r in vis_resamples if r > 0)),
		"counts": dict(sorted(res_counts.items(), key=lambda kv: int(kv[0]))),
		"pass": True,
	}

	used_variants = tuple(sorted(int(v) for v in variants)) or tuple(
		(HELD_OUT_VARIANT,) if split == "ood-lang" else TRAIN_VARIANTS
	)
	used_colours = tuple(sorted(set(normal_colours) | set(defect_colours) | (set(extra_colours) - {"none"}))) or tuple(
		sorted(PALETTE)
	)
	token_max, token_backend, worst_instruction = _measure_tokens_cached(used_variants, used_colours)

	factors = {
		"normal_bowl_colour": {col: int(normal_colours.get(col, 0)) for col in sorted(PALETTE)},
		"defect_bowl_colour": {col: int(defect_colours.get(col, 0)) for col in sorted(PALETTE)},
		"extra_bowl_colour": dict(sorted(extra_colours.items())),
		"bowl_count": dict(sorted(bowl_count_hist.items())),
		"routing_counts": dict(sorted(routing_counts.items())),
		"colour_x_target_x_side": crosstab,
		"brick_colour": dict(sorted(brick_colours.items())),
		"brick_size": dict(sorted(brick_sizes.items())),
		"table_texture": dict(sorted(table_textures.items())),
		"table_v": {**_stats(table_vs), "hist": _hist(table_vs, 0.35, 0.80, 9)},
		"table_sat": {**_stats(table_sats), "hist": _hist(table_sats, 0.0, 0.55, 11)},
		"light_level": _stats(light_levels),
		"light_k": {**_stats(light_ks), "hist": _hist(light_ks, 3500.0, 6500.0, 6)},
		"instruction_variant": dict(sorted(variants.items(), key=lambda kv: int(kv[0]))),
		"camera_bins": {k: v["hist"] for k, v in camera_hist["top"].items()},
	}

	return {
		"balance": {
			"episodes": n,
			"class_counts": class_counts,
			"class_mix_ok": class_mix_ok,
			"target_colour_counts": target_count_map,
			"mean_target_count": round(mean_target, 4),
			"max_target_deviation_frac": round(max_dev, 4),
			"left_right_corr_by_colour": corrs,
			"max_abs_left_right_corr": round(max_abs_corr, 4),
			"strict_thresholds_enforced": enforce_strict,
			"all_colours_present": all_colours_present,
			"pass": balance_pass,
		},
		"colour_pairs": {
			"pair_counts": dict(sorted(pair_counts.items())),
			"slot_pair_counts": dict(sorted(slot_pair_counts.items())),
			"held_out_pairs_present": sorted(held_out_found),
			"held_out_pair_count": held_out_primary_count if split == "ood-color" else len(held_out_found),
			"min_nominal_hue_gap_deg": round(float(min_nominal_gap), 3),
			"min_chromatic_hue_gap_deg": round(float(min_chromatic_gap), 3),
			"pass": bool(colour_pairs_pass),
		},
		"camera_hist": camera_hist,
		"visibility_resamples": visibility_resamples,
		"token_max": int(token_max),
		"token_check": {
			"token_max": int(token_max),
			"limit": TOKEN_LIMIT,
			"backend": token_backend,
			"worst_instruction": worst_instruction,
			"pass": int(token_max) <= TOKEN_LIMIT,
		},
		"factors": factors,
	}


def build_suite_and_preflight(
	*,
	split: str,
	start_seed: int,
	end_seed: int,
	perception_sample: float = 0.1,
	max_perception: int | None = None,
	sensing: str = "realistic",
	command: str = "",
	scorer=None,
	skip_hue_fp: bool = False,
	hue_fp_per_bin: int = HUE_FP_DEFAULT_PER_BIN,
	hue_fp_views: int = 2,
	strict_balance: bool | None = None,
	filter_perception_failures: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
	"""Generate a validated v3 suite dict and its preflight report dict."""
	if split not in V3_SPLIT_SEED_RANGES:
		raise ValueError(f"Unknown split {split!r}; expected one of {SPLITS}")
	split_lo, split_hi = V3_SPLIT_SEED_RANGES[split]
	if start_seed < split_lo or end_seed > split_hi or start_seed > end_seed:
		raise ValueError(
			f"Seed range {start_seed}-{end_seed} invalid for split {split!r} (allowed {split_lo}-{split_hi})"
		)

	target_count = end_seed - start_seed + 1
	cls = _MS.classes(DEFAULT_MIX)
	schedule = [cls[idx] for idx in _MS.interleave(_MS.allocate(target_count, DEFAULT_MIX))]
	use_stratified = (target_count >= BALANCE_STRICT_MIN_EPISODES) if strict_balance is None else bool(strict_balance)
	strata_quotas = _target_strata_quotas(target_count) if use_stratified else None
	strata_counts: Counter[tuple[str, bool]] = Counter()
	balance_skipped = 0

	rejections: dict[str, int] = {"reachability": 0, "visibility": 0, "colour_separability": 0}
	cases: list[list] = []
	layouts: list = []
	candidate_seed = start_seed

	def _next_valid(positive: bool, face: int, required_stratum: tuple[str, bool] | None = None):
		nonlocal candidate_seed, balance_skipped
		while True:
			if candidate_seed > split_hi:
				raise RuntimeError(
					f"Exhausted split {split!r} seed range at {split_hi} after {len(cases)}/{target_count} cases "
					f"(rejections={rejections})"
				)
			layout, reason = evaluate_candidate_seed(candidate_seed, positive, face, split)
			seed_used = candidate_seed
			candidate_seed += 1
			if layout is None:
				rejections[reason] += 1
				continue
			stratum = _layout_target_stratum(layout, positive)
			if required_stratum is not None:
				if stratum != required_stratum:
					balance_skipped += 1
					continue
			elif strata_quotas is not None and strata_counts[stratum] >= strata_quotas[stratum]:
				balance_skipped += 1
				continue
			return seed_used, layout, stratum

	for positive, face in schedule:
		seed_used, layout, stratum = _next_valid(positive, face)
		cases.append([seed_used, positive, face, {"scene": "v3", "split": split}])
		layouts.append(layout)
		strata_counts[stratum] += 1

	perception = run_perception_check(
		cases,
		layouts,
		perception_sample=perception_sample,
		max_perception=max_perception,
		sensing=sensing,
	)
	perception_rejected_seeds: list[int] = []
	if filter_perception_failures and perception["failed_case_indices"]:
		fixed_indices = list(perception["sampled_indices"])
		for _round in range(5):
			failed_idxs = list(perception["failed_case_indices"])
			if not failed_idxs:
				break
			for idx in failed_idxs:
				old_case = cases[idx]
				old_lay = layouts[idx]
				perception_rejected_seeds.append(int(old_case[0]))
				req_stratum = _layout_target_stratum(old_lay, bool(old_case[1])) if use_stratified else None
				seed_used, new_lay, _ = _next_valid(bool(old_case[1]), int(old_case[2]), required_stratum=req_stratum)
				cases[idx] = [seed_used, bool(old_case[1]), int(old_case[2]), {"scene": "v3", "split": split}]
				layouts[idx] = new_lay
			perception = run_perception_check(
				cases,
				layouts,
				perception_sample=perception_sample,
				max_perception=max_perception,
				sensing=sensing,
				sampled_indices=fixed_indices,
			)
	perception["rejected_seeds"] = perception_rejected_seeds
	perception["rejected_count"] = len(perception_rejected_seeds)

	sealed = split == "test"
	seed_lo_out = min(c[0] for c in cases)
	seed_hi_out = max(c[0] for c in cases)
	suite: dict[str, Any] = {
		"name": f"v3-{split}",
		"split": split,
		"sealed": sealed,
		**({"sealed_note": _MS.SEALED_NOTE} if sealed else {}),
		"generator": "scripts/vla/make_scene_suite_v3.py",
		"seed_range": [seed_lo_out, seed_hi_out],
		"requested_seed_range": [start_seed, end_seed],
		"class_mix": {"normal": DEFAULT_MIX[0], "defect_faces_0_to_5": list(DEFAULT_MIX[1:])},
		"cases": cases,
	}

	metrics = _compute_balance_and_factors(cases, layouts, split, strict_balance=strict_balance)
	metrics["balance"]["balance_skipped_seeds"] = balance_skipped
	hue_fp = run_hue_fp_control(
		cases,
		layouts,
		scorer=scorer,
		n_per_bin=hue_fp_per_bin,
		n_views=hue_fp_views,
		skip=skip_hue_fp,
	)

	overall_pass = bool(
		metrics["balance"]["pass"]
		and metrics["colour_pairs"]["pass"]
		and metrics["visibility_resamples"]["pass"]
		and metrics["token_check"]["pass"]
		and perception["pass"]
		and hue_fp["pass"]
	)

	preflight: dict[str, Any] = {
		"generator": "scripts/vla/make_scene_suite_v3.py",
		"command": command,
		"split": split,
		"episodes": len(cases),
		"seed_range": [seed_lo_out, seed_hi_out],
		"requested_seed_range": [start_seed, end_seed],
		"rejections": rejections,
		"balance": metrics["balance"],
		"colour_pairs": metrics["colour_pairs"],
		"camera_hist": metrics["camera_hist"],
		"visibility_resamples": metrics["visibility_resamples"],
		"perception_check": perception,
		"hue_fp_control": hue_fp,
		"token_max": metrics["token_max"],
		"token_check": metrics["token_check"],
		"factors": metrics["factors"],
		"pass": overall_pass,
	}
	return suite, preflight


def render_preflight_md(preflight: dict[str, Any]) -> str:
	"""Format a human-readable markdown preflight summary alongside ``preflight.json``."""
	bal = preflight["balance"]
	cp = preflight["colour_pairs"]
	pc_res = preflight["perception_check"]
	tc = preflight["token_check"]
	vis = preflight["visibility_resamples"]
	rej = preflight["rejections"]
	fac = preflight["factors"]

	lines = [
		f"# v3 Scene Suite Preflight: `{preflight['split']}`",
		"",
		f"- **Overall pass**: `{preflight['pass']}`",
		f"- **Episodes**: {preflight['episodes']} (seeds `{preflight['seed_range'][0]}-{preflight['seed_range'][1]}`)",
		(
			f"- **Rejections**: reachability={rej['reachability']}, visibility={rej['visibility']}, "
			f"colour_separability={rej['colour_separability']}"
		),
		f"- **Token max**: {tc['token_max']} / {tc['limit']} (`{tc['backend']}`, pass=`{tc['pass']}`)",
		f"- **Visibility resamples**: total={vis['total']}, max={vis['max']}, median={vis['median']}",
		"",
		"## Balance & Colour Pairs",
		"",
		f"- Class mix ok: `{bal['class_mix_ok']}` (`{bal['class_counts']}`)",
		(
			f"- Target colour counts: `{bal['target_colour_counts']}` "
			f"(max dev={bal['max_target_deviation_frac']:.3f}, "
			f"max |corr|={bal['max_abs_left_right_corr']:.4f}, strict={bal['strict_thresholds_enforced']})"
		),
		(
			f"- Held-out pairs count: {cp['held_out_pair_count']} "
			f"(min nominal hue gap={cp['min_nominal_hue_gap_deg']} deg, "
			f"min jittered hue gap={cp['min_chromatic_hue_gap_deg']} deg, pass=`{cp['pass']}`)"
		),
		"",
		"## Factors",
		"",
		f"- Bowl count: `{fac['bowl_count']}`",
		f"- Brick colour: `{fac['brick_colour']}`; size: `{fac['brick_size']}`",
		f"- Table texture: `{fac['table_texture']}`; light_k mean: `{fac['light_k']['mean']}` K",
		f"- Instruction variants: `{fac['instruction_variant']}`",
		"",
		"## Perception Check",
		"",
		f"- Sampled: {pc_res['sampled_count']} cases (sensing=`{pc_res['sensing']}`, crashes={pc_res['crashes']})",
		(
			f"- Bowls: {pc_res['bowls_found']}/{pc_res['bowls_total']} ({pc_res['bowl_localization_rate']:.3f}), "
			f"median error={pc_res['bowl_median_error_mm']} mm"
		),
		(
			f"- Bricks: {pc_res['bricks_found']}/{pc_res['bricks_total']} ({pc_res['brick_localization_rate']:.3f}), "
			f"median error={pc_res['brick_median_error_mm']} mm"
		),
		f"- Perception pass: `{pc_res['pass']}` (rejected={pc_res.get('rejected_count', 0)})",
		"",
		"## Hue False-Positive Control",
		"",
		f"- Status: `{preflight['hue_fp_control']['status']}` (pass=`{preflight['hue_fp_control']['pass']}`)",
		f"- Failing bins: `{preflight['hue_fp_control'].get('failing_bins', [])}`",
		"",
	]
	return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
	p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	p.add_argument("--split", required=True, choices=SPLITS, help="v3 split name")
	p.add_argument("--seeds", required=True, help="inclusive seed range START-END within the split range")
	p.add_argument("--out", required=True, type=Path, help="output suite JSON path")
	p.add_argument("--preflight-dir", required=True, type=Path, help="directory for preflight.json and preflight.md")
	p.add_argument(
		"--perception-sample", type=float, default=0.1, help="fraction of cases to render for perception check"
	)
	p.add_argument("--max-perception", type=int, default=None, help="optional cap on rendered perception check cases")
	p.add_argument(
		"--sensing", choices=("truth", "realistic"), default="realistic", help="camera pose/noise mode for check"
	)
	p.add_argument("--detector", type=Path, default=None, help="anomaly detector spec JSON for hue FP control")
	p.add_argument(
		"--skip-hue-fp", action="store_true", help="explicitly skip wrist-sweep hue FP control (static-only mode)"
	)
	p.add_argument("--hue-fp-per-bin", type=int, default=HUE_FP_DEFAULT_PER_BIN, help="good bricks per 60-deg hue bin")
	p.add_argument("--hue-fp-views", type=int, default=2, help="wrist-sweep views per case for hue FP control")
	args = p.parse_args(argv)

	try:
		start_seed, end_seed = parse_seed_range(args.seeds, split=args.split)
	except ValueError as exc:
		p.error(str(exc))

	if args.perception_sample < 0.0 or args.perception_sample > 1.0:
		p.error("--perception-sample must be in [0.0, 1.0]")

	cmd = (
		f"python scripts/vla/make_scene_suite_v3.py --split {args.split} --seeds {args.seeds} "
		f"--out {args.out} --preflight-dir {args.preflight_dir}"
	)
	scorer = None
	if args.detector is not None:
		from digital_twin_armigo.inspection_scorer import default_sup_scorer

		scorer = default_sup_scorer(args.detector)
	try:
		suite, preflight = build_suite_and_preflight(
			split=args.split,
			start_seed=start_seed,
			end_seed=end_seed,
			perception_sample=args.perception_sample,
			max_perception=args.max_perception,
			sensing=args.sensing,
			command=cmd,
			scorer=scorer,
			skip_hue_fp=args.skip_hue_fp,
			hue_fp_per_bin=args.hue_fp_per_bin,
			hue_fp_views=args.hue_fp_views,
		)
	finally:
		if scorer is not None and hasattr(scorer, "close"):
			scorer.close()

	args.out.parent.mkdir(parents=True, exist_ok=True)
	args.out.write_text(_MS._suite_json(suite), encoding="utf-8")
	load_suite(args.out)

	args.preflight_dir.mkdir(parents=True, exist_ok=True)
	(args.preflight_dir / "preflight.json").write_text(json.dumps(preflight, indent="\t") + "\n", encoding="utf-8")
	(args.preflight_dir / "preflight.md").write_text(render_preflight_md(preflight), encoding="utf-8")
	print(
		f"v3 suite {args.split}: {len(suite['cases'])} cases, "
		f"rejections={preflight['rejections']}, pass={preflight['pass']}"
	)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
