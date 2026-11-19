"""Train / validation / sealed-test case suites for the VLA student (plan S3) and their realized distribution.

Each split takes consecutive layout seeds from its own range; one case per seed, so splits never share a layout
(plan R5). Classes follow a mix (default 3 normal : 1 defective per face 0-5), allocated by largest remainder and
interleaved so every prefix of a suite stays close to the mix. Seeds of every suite in ``config/agent_bench`` (bench16
included) and the reserved ranges below are refused.

The report is computed from the layouts alone (``sample_case``; no episode, no scene build):

	.venv/Scripts/python scripts/vla/make_suites.py                    # default proposal into config/vla
	.venv/Scripts/python scripts/vla/make_suites.py --cameras jitter --prior config/sim/prior_grasp.json --out ...

The suites are objects ``{"cases": [...], "options": {...}, ...metadata}`` that ``armigo-mcp run --suite`` reads
(``twin_mujoco.load_suite``). The test suite is marked ``"sealed": true``: final evaluation only.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from digital_twin_armigo.agent.twin_mujoco import sample_case
from digital_twin_armigo.constants import BRICK_COLOURS, DUPLO_2X4

FACES = 6
BOTTOM_FACE = 5
COLOUR_NAMES = dict(zip(BRICK_COLOURS, ("orange", "yellow", "blue", "red", "lime"), strict=True))
# Seed ranges used elsewhere (inclusive): agent fixtures and the policy-search dev suite (7xxxxx), bottom-crop detector
# renders (``scripts/anomaly/render_bottom.py`` from 800001), bottom threshold calibration / test (661200, 772400),
# hand-eye simulation (900001). Every seed listed in a config/agent_bench suite is refused as well.
RESERVED_RANGES = ((0, 999_999),)
DEFAULT_SPLITS = {"train": (300, 1_100_001), "val": (60, 1_200_001), "test": (100, 1_300_001)}
SEALED_NOTE = (
	"Sealed final test (plan R5): never use for tuning, threshold or model selection, early stopping or DAgger. "
	"Run once per frozen student."
)


def bench_seeds(root: Path = ROOT) -> set[int]:
	"""Every seed of the case suites in config/agent_bench (bench16, dev4, bottom48, ...)."""
	seeds = set()
	for path in sorted((root / "config/agent_bench").glob("*.json")):
		rows = json.loads(path.read_text(encoding="utf-8-sig"))
		if isinstance(rows, dict):
			rows = rows.get("cases", [])
		seeds |= {int(r[0]) for r in rows if isinstance(r, list) and r}
	return seeds


def classes(mix: tuple[int, ...]) -> list[tuple[bool, int]]:
	"""The class of each mix entry: index 0 normal (face 5, the bench convention), 1 + f defective on face f."""
	return [(False, BOTTOM_FACE), *((True, f) for f in range(FACES))][: len(mix)]


def allocate(n: int, mix: tuple[int, ...]) -> list[int]:
	"""Counts per class for ``n`` cases: exact when ``n`` is a multiple of sum(mix), else largest remainder (ties to
	the lower class index)."""
	total = sum(mix)
	quotas = [Fraction(n * w, total) for w in mix]
	counts = [int(q) for q in quotas]
	order = sorted(range(len(mix)), key=lambda i: (-(quotas[i] - counts[i]), i))
	for i in order[: n - sum(counts)]:
		counts[i] += 1
	return counts


def interleave(counts: list[int]) -> list[int]:
	"""Class indices ordered so that every prefix tracks the counts: the k-th instance of class c sits at
	(k + 1/2) / counts[c] of the suite (ties by class index)."""
	keys = [((k + 0.5) / c, i) for i, c in enumerate(counts) for k in range(c)]
	return [i for _, i in sorted(keys)]


def make_cases(n: int, start: int, mix: tuple[int, ...]) -> list[list]:
	cls = classes(mix)
	return [[start + j, *cls[c]] for j, c in enumerate(interleave(allocate(n, mix)))]


def uses_bottom(case) -> bool:
	"""Default policy (wrist views first): the bottom presentation runs unless a wrist view found the defect, i.e.
	for normal bricks and bottom-face defects (assuming the wrist scan finds every face 0-4 defect)."""
	return not case[1] or case[2] == BOTTOM_FACE


def check_seeds(splits: dict[str, list], forbidden: set[int]) -> None:
	seen: dict[int, str] = {}
	for name, cases in splits.items():
		for case in cases:
			s = case[0]
			if s in seen:
				raise ValueError(f"seed {s} in both {seen[s]} and {name}")
			if s in forbidden:
				raise ValueError(f"seed {s} of {name} is used by a config/agent_bench suite")
			if any(lo <= s <= hi for lo, hi in RESERVED_RANGES):
				raise ValueError(f"seed {s} of {name} lies in a reserved range")
			seen[s] = name


# -- realized distribution ---------------------------------------------------------------------------------------


def _stats(values) -> dict:
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


def _hist(values, lo: float, hi: float, bins: int) -> dict:
	counts, edges = np.histogram(np.clip(values, lo, hi), bins=bins, range=(lo, hi))
	return {"edges": [round(float(e), 4) for e in edges], "counts": counts.tolist()}


def distribution(cases: list[list], options: dict | None) -> dict:
	"""Realized layout, camera and skill distribution of ``cases`` (layouts only)."""
	lays = [sample_case(*c[:3], options or None) for c in cases]
	bx = np.array([lay.brick_xy for lay in lays])
	reach = np.hypot(bx[:, 0], bx[:, 1])
	bearing = np.degrees(np.arctan2(bx[:, 1], bx[:, 0]))
	yaw = np.array([lay.brick_yaw_deg for lay in lays])
	bottom = sum(uses_bottom(c) for c in cases)
	cams = {}
	for name in ("top", "front"):
		poses = [getattr(lay, name) for lay in lays]
		cams[name] = {
			k: _stats([getattr(p, k) for p in poses])
			for k in ("x", "y", "height", "yaw_deg", "tilt_deg", "roll_deg", "focal_px")
		}
	cls = Counter("normal" if not c[1] else f"defect_face_{c[2]}" for c in cases)
	return {
		"episodes": len(cases),
		"seeds": [min(c[0] for c in cases), max(c[0] for c in cases)],
		"classes": dict(sorted(cls.items())),
		"skills": {
			"wrist_scan": len(cases),
			"wrist_only": len(cases) - bottom,
			"bottom_presentation": bottom,
			"bottom_share": round(bottom / len(cases), 4),
		},
		"brick": {
			"x_m": _stats(bx[:, 0]),
			"y_m": _stats(bx[:, 1]),
			"reach_m": _stats(reach),
			"bearing_deg": _stats(bearing),
			"x_hist": _hist(bx[:, 0], 0.10, 0.45, 7),
			"y_hist": _hist(bx[:, 1], -0.20, 0.20, 8),
			"yaw_deg": _stats(yaw),
			"yaw_hist": _hist(yaw, -90.0, 90.0, 6),
			"size": dict(
				sorted(Counter("2x4" if tuple(lay.brick_size) == DUPLO_2X4 else "2x2" for lay in lays).items())
			),
			"colour": dict(sorted(Counter(COLOUR_NAMES[tuple(lay.brick_rgb)] for lay in lays).items())),
		},
		"bowls": {
			f"{c}_{a}": _stats([getattr(lay, c)[i] for lay in lays])
			for c in ("green", "pink")
			for i, a in enumerate(("x_m", "y_m"))
		}
		| {f"{c}_reach_m": _stats([np.hypot(*getattr(lay, c)) for lay in lays]) for c in ("green", "pink")},
		"scene": {
			"light_level": _stats([lay.light_level for lay in lays]),
			"tile_m": _stats([lay.tile_m for lay in lays]),
		},
		"cameras": cams,
	}


# -- output -------------------------------------------------------------------------------------------------------


def _suite_json(suite: dict) -> str:
	"""Pretty JSON with one case per line."""
	head = {k: v for k, v in suite.items() if k != "cases"}
	body = json.dumps(head, indent="\t")[:-2]
	rows = ",\n".join("\t\t" + json.dumps(c) for c in suite["cases"])
	return f'{body},\n\t"cases": [\n{rows}\n\t]\n}}\n'


def _fmt(s: dict, unit: str = "", nd: int = 3) -> str:
	if s["max"] - s["min"] < 1e-9:
		return f"{s['min']:.{nd}f}{unit} (constant)"
	return (
		f"{s['min']:.{nd}f} / {s['p5']:.{nd}f} / {s['median']:.{nd}f} / {s['p95']:.{nd}f} / {s['max']:.{nd}f}{unit}"
		f" (std {s['std']:.{nd}f})"
	)


def _hist_line(h: dict, nd: int = 2) -> str:
	e = h["edges"]
	return ", ".join(f"[{e[i]:.{nd}f},{e[i + 1]:.{nd}f}): {c}" for i, c in enumerate(h["counts"]))


def report_md(report: dict) -> str:
	mix = report["class_mix"]
	expected = Fraction(mix[0] + mix[1 + BOTTOM_FACE], sum(mix))
	lines = [
		"# VLA case suites (plan S3)",
		"",
		f"Generated by `scripts/vla/make_suites.py`; regenerate with `{report['command']}`. Numbers below are the",
		"**realized** distribution of the committed suites, computed from the layouts (`sample_case`), not from",
		"episodes. `report.json` holds the same numbers in full.",
		"",
		f"- Options: `{json.dumps(report['options'])}` (prior: real-layout prior; cameras: `fixed` or `jitter`, D3).",
		f"- Class mix (normal : defect face 0..5): {' : '.join(map(str, mix))}. One case per layout seed.",
		(
			f"- Seeds: disjoint across splits; none of the {report['excluded_bench_seeds']} seeds of the"
			" `config/agent_bench` suites (bench16 included), none below 1,000,000 (reserved ranges)."
		),
		"- The test suite is **sealed**: final evaluation only (plan R5).",
		"",
		"## Splits and expected skill frequencies",
		"",
		(
			f"With wrist views first (default policy), the bottom presentation runs for normal bricks and face-5 defects:"
			f" expected share {expected} = {float(expected):.3f} for this mix (assuming the wrist scan finds every face"
			" 0-4 defect). Realized shares differ slightly when a suite size is not a multiple of the mix total."
		),
		"",
		"| Split | Seeds | Episodes | Normal | Defective (faces 0..5) | Wrist only | Bottom presentation | Share |",
		"|---|---|---|---|---|---|---|---|",
	]
	for name, d in report["splits"].items():
		cls = d["classes"]
		faces = "/".join(str(cls.get(f"defect_face_{f}", 0)) for f in range(FACES))
		sealed = " (sealed)" if name == "test" else ""
		sk = d["skills"]
		lines.append(
			f"| {name}{sealed} | {d['seeds'][0]}-{d['seeds'][1]} | {d['episodes']} | {cls.get('normal', 0)} | {faces} |"
			f" {sk['wrist_only']} | {sk['bottom_presentation']} | {sk['bottom_share']:.3f} |"
		)
	lines += ["", "## Realized layout distribution", ""]
	lines.append("Statistics: min / p5 / median / p95 / max (std). Base frame, metres and degrees.")
	for name, d in report["splits"].items():
		b, c = d["brick"], d["cameras"]
		lines += [
			"",
			f"### {name} ({d['episodes']} layouts)",
			"",
			f"- Brick x: {_fmt(b['x_m'])}; histogram {_hist_line(b['x_hist'])}",
			f"- Brick y: {_fmt(b['y_m'])}; histogram {_hist_line(b['y_hist'])}",
			f"- Brick reach: {_fmt(b['reach_m'])}; bearing {_fmt(b['bearing_deg'], ' deg', 1)}",
			f"- Brick yaw: {_fmt(b['yaw_deg'], ' deg', 1)}; histogram {_hist_line(b['yaw_hist'], 0)}",
			f"- Brick size: {b['size']}; colour: {b['colour']}",
			f"- Green bowl x {_fmt(d['bowls']['green_x_m'])}, y {_fmt(d['bowls']['green_y_m'])}",
			f"- Pink bowl x {_fmt(d['bowls']['pink_x_m'])}, y {_fmt(d['bowls']['pink_y_m'])}",
			f"- Light level {_fmt(d['scene']['light_level'])}; table tile {_fmt(d['scene']['tile_m'])} m",
		]
		for cam in ("top", "front"):
			p = c[cam]
			lines.append(
				f"- {cam} camera: x {_fmt(p['x'])}, y {_fmt(p['y'])}, height {_fmt(p['height'])},"
				f" yaw {_fmt(p['yaw_deg'], '', 2)}, tilt {_fmt(p['tilt_deg'], '', 2)},"
				f" roll {_fmt(p['roll_deg'], '', 2)}, focal {_fmt(p['focal_px'], ' px', 1)}"
			)
	if report.get("variants"):
		lines += [
			"",
			"## Option variants (same train seeds, not committed as suites)",
			"",
			"For deciding the prior and D3. Each row re-samples the train layouts with the given options.",
			"",
			(
				"| Options | Brick x (p5-p95) | Brick y (p5-p95) | Brick reach max | Pink bowl reach max |"
				" Top x std | Top yaw std | Front yaw range |"
			),
			"|---|---|---|---|---|---|---|---|",
		]
		for v in report["variants"]:
			b, t, f = v["brick"], v["cameras"]["top"], v["cameras"]["front"]
			lines.append(
				f"| `{json.dumps(v['options'])}` | {b['x_m']['p5']:.3f}-{b['x_m']['p95']:.3f} |"
				f" {b['y_m']['p5']:.3f}-{b['y_m']['p95']:.3f} | {b['reach_m']['max']:.3f} |"
				f" {v['bowls']['pink_reach_m']['max']:.3f} | {t['x']['std']:.4f} | {t['yaw_deg']['std']:.2f} |"
				f" {f['yaw_deg']['min']:.1f}-{f['yaw_deg']['max']:.1f} |"
			)
	lines += [
		"",
		"## Notes",
		"",
		(
			"- Camera jitter (`cameras: jitter`) perturbs the fixed top and front cameras per seed from a separate random"
			" stream, so all other layout quantities are unchanged. In `realistic` sensing `TwinMujoco` hand-eye calibrates"
			" the top camera of every episode from its own layout at construction (`calibrate_layout`, event"
			" `camera_calibrated`), so the skills see the jittered camera through that per-episode calibration; the"
			" intrinsics (focal) are taken as known. Spot check on six jittered train seeds: calibration converged with 40"
			" card poses, 0.9-2.1 mm and 0.08-0.25 deg camera error, about 3 s each. `truth` sensing reads no camera pose."
			" The bottom-view threshold profile and detector were tuned on the fixed camera (`camera_profile`"
			" `mujoco-close-top-1.35`, still recorded as such); a jittered suite needs its own expert check before"
			" collection."
		),
		(
			"- The real-layout prior (`config/sim/prior_grasp.json`, 35 real first frames) replaces the uniform positions:"
			" bricks start from real brick positions and the bowls from one essentially fixed real arrangement (pink bowl"
			" far out), each jittered a few cm (see the variants table). The expert has only been validated on uniform"
			" layouts (bench16, brick reach at most 0.25 m), so a prior suite also needs an expert check."
		),
		"",
	]
	return "\n".join(lines)


def main(argv=None) -> int:
	p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	p.add_argument("--out", type=Path, default=ROOT / "config/vla")
	for name, (n, start) in DEFAULT_SPLITS.items():
		p.add_argument(f"--{name}", type=int, default=n, help=f"{name} episodes (default {n})")
		p.add_argument(f"--{name}-start", type=int, default=start, help=f"first {name} seed (default {start})")
	p.add_argument("--mix", type=int, nargs=FACES + 1, default=[3, 1, 1, 1, 1, 1, 1], help="normal, faces 0..5")
	p.add_argument("--prior", help="real-layout prior JSON (repository-relative), e.g. config/sim/prior_grasp.json")
	p.add_argument("--cameras", choices=("fixed", "jitter"), default="fixed")
	p.add_argument("--no-variants", action="store_true", help="skip the option-variant comparison in the report")
	a = p.parse_args(argv)
	mix = tuple(a.mix)
	if min(mix) < 0 or sum(mix) == 0:
		p.error("mix weights must be nonnegative with a positive sum")
	options = {k: v for k, v in (("prior", a.prior), ("cameras", a.cameras)) if v not in (None, "fixed")}
	splits = {}
	for name in DEFAULT_SPLITS:
		n, start = getattr(a, name), getattr(a, f"{name}_start")
		if n < 1:
			p.error(f"--{name} must be positive")
		splits[name] = make_cases(n, start, mix)
	forbidden = bench_seeds()
	check_seeds(splits, forbidden)

	a.out.mkdir(parents=True, exist_ok=True)
	try:
		shown = a.out.resolve().relative_to(ROOT).as_posix()
	except ValueError:
		shown = str(a.out)
	cmd = "python scripts/vla/make_suites.py" + "".join(
		f" --{k} {v}" for k, v in (("prior", a.prior), ("cameras", a.cameras)) if v not in (None, "fixed")
	)
	if shown != "config/vla":
		cmd += f" --out {shown}"
	report = {
		"generator": "scripts/vla/make_suites.py",
		"command": cmd,
		"class_mix": list(mix),
		"options": options,
		"excluded_bench_seeds": len(forbidden),
		"splits": {},
	}
	for name, cases in splits.items():
		sealed = name == "test"
		suite = {
			"name": f"vla-{name}",
			"split": name,
			"sealed": sealed,
			**({"sealed_note": SEALED_NOTE} if sealed else {}),
			"generator": "scripts/vla/make_suites.py",
			"seed_range": [cases[0][0], cases[-1][0]],
			"class_mix": {"normal": mix[0], "defect_faces_0_to_5": list(mix[1:])},
			"options": options,
			"cases": cases,
		}
		(a.out / f"{name}.json").write_text(_suite_json(suite), encoding="utf-8")
		report["splits"][name] = distribution(cases, options)
	if not a.no_variants:
		report["variants"] = []
		for opts in ({}, {"cameras": "jitter"}, {"prior": "config/sim/prior_grasp.json"}):
			report["variants"].append({"options": opts, **distribution(splits["train"], opts)})
	(a.out / "report.json").write_text(json.dumps(report, indent="\t") + "\n", encoding="utf-8")
	(a.out / "SUITES.md").write_text(report_md(report), encoding="utf-8")
	for name, d in report["splits"].items():
		print(name, d["episodes"], d["classes"], d["skills"])
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
