"""Single source of truth for the CV motion & evidence matrix (WP9, prereg ``docs/experiments/cv_motion_prereg.md``).

	PYTHONPATH=src python scripts/experiments/cv_motion_plan.py --plan-only --stage all
	PYTHONPATH=src python scripts/experiments/cv_motion_plan.py --emit-commands --stage stage2

Stages mirror ``scripts/experiments/cv_motion_matrix.sh``: Stage 2 runs every motion under ``{fixed, recal, drift}``
so prereg constraint 1 (fixed and recal success counts) is decidable at the shortlist, Stage 3b stays opt-in, and
Stage 4 confirms the winner on the registered ``loop_01_96`` suite. Group names come from ``runner.group_name``;
importing this module must not pull MuJoCo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

from digital_twin_armigo.agent.grasp_loop import VARIANTS
from digital_twin_armigo.experiments.runner import group_name
from digital_twin_armigo.governance.segments import split_of

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "config/agent_bench/v2/manifest.json"

EVIDENCES = ("hold", "img", "dense-10", "dense-20")
SEGMENTERS = ("lab", "unet", "silhouette")
CAMERAS = ("fixed", "recal", "drift")
MOTIONS = tuple(VARIANTS)

DEFAULT_STAGE3_MOTIONS = ("open", "w2s")
DEFAULT_WINNER_MOTION = "w2s"
DEFAULT_WINNER_EVIDENCE = "hold"
DEFAULT_WINNER_SEGMENTER = "lab"
DEFAULT_CONFIRM_SUITE = "loop_01_96"

ALL_STAGES = ("stage1", "stage2", "stage3", "stage4")
STAGE_ORDER = ("stage1", "stage2", "stage3", "stage3b", "stage4")

STAGES: dict[str, dict] = {
	"stage1": {
		"batch": "cvm-s1-tol",
		"suite": "p2_dev12",
		"motions": ("open", "w2s"),
		"cameras": ("fixed",),
		"evidences": ("hold",),
		"offsets": (0, 2, 5, 10, 20),
	},
	"stage2": {
		"batch": "cvm-s2",
		"suite": "p2_dev12",
		"motions": MOTIONS,
		"cameras": CAMERAS,
		"evidences": ("hold",),
		"offsets": (0,),
	},
	"stage3": {
		"batch": "cvm-s3",
		"suite": "p2_dev12",
		"motions": DEFAULT_STAGE3_MOTIONS,
		"cameras": CAMERAS,
		"evidences": EVIDENCES,
		"offsets": (0,),
	},
	"stage3b": {
		"batch": "cvm-s3b",
		"suite": "p2_dev12",
		"motions": (DEFAULT_WINNER_MOTION,),
		"cameras": CAMERAS,
		"evidences": (DEFAULT_WINNER_EVIDENCE,),
		"offsets": (0,),
	},
	"stage4": {
		"batch": "cvm-s4-confirm",
		"suite": DEFAULT_CONFIRM_SUITE,
		"motions": ("open", DEFAULT_WINNER_MOTION),
		"cameras": CAMERAS,
		"evidences": ("hold",),
		"offsets": (0,),
	},
}


def _env_tokens(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
	raw = os.environ.get(name)
	if raw is None or not raw.strip():
		return default
	return tuple(raw.split())


def _run(
	stage: str,
	batch: str,
	suite: str,
	camera: str,
	motion: str,
	evidence: str,
	calib_offset_mm: float,
	segmenter: str,
) -> dict:
	return {
		"kind": "run",
		"stage": stage,
		"batch": batch,
		"suite": suite,
		"sensing": "realistic",
		"camera": camera,
		"strategy": "B",
		"detector": "v7bot",
		"motion": motion,
		"evidence": evidence,
		"calib_offset_mm": calib_offset_mm,
		"segmenter": segmenter,
		"group": group_name(
			"realistic",
			camera,
			"B",
			"v7bot",
			motion=motion,
			evidence=evidence,
			calib_offset_mm=calib_offset_mm,
			segmenter=segmenter,
		),
	}


def _compare(stage: str, batch: str, group_a: str, group_b: str) -> dict:
	return {"kind": "compare", "stage": stage, "batch": batch, "compare": [group_a, group_b]}


def plan(
	stage: str,
	*,
	stage3_motions: tuple[str, ...] | None = None,
	winner_motion: str | None = None,
	winner_evidence: str | None = None,
	winner_segmenter: str | None = None,
	confirm_suite: str | None = None,
) -> list[dict]:
	"""One dict per runner invocation (plus ``kind == "compare"`` report entries) for ``stage``."""
	if stage not in STAGES:
		raise ValueError(f"unknown stage {stage!r}; expected one of {STAGE_ORDER}")
	if stage3_motions is None:
		stage3_motions = _env_tokens("STAGE3_MOTIONS", DEFAULT_STAGE3_MOTIONS)
	if winner_motion is None:
		winner_motion = os.environ.get("WINNER_MOTION") or DEFAULT_WINNER_MOTION
	if winner_evidence is None:
		winner_evidence = os.environ.get("WINNER_EVIDENCE") or DEFAULT_WINNER_EVIDENCE
	if winner_segmenter is None:
		winner_segmenter = os.environ.get("WINNER_SEGMENTER") or DEFAULT_WINNER_SEGMENTER
	if confirm_suite is None:
		confirm_suite = os.environ.get("CONFIRM_SUITE") or DEFAULT_CONFIRM_SUITE

	spec = STAGES[stage]
	batch, suite = spec["batch"], spec["suite"]
	out: list[dict] = []
	if stage == "stage1":
		for motion in spec["motions"]:
			for offset in spec["offsets"]:
				out.append(_run(stage, batch, suite, "fixed", motion, "hold", offset, "lab"))
	elif stage == "stage2":
		for motion in spec["motions"]:
			for camera in spec["cameras"]:
				out.append(_run(stage, batch, suite, camera, motion, "hold", 0, "lab"))
		base = group_name("realistic", "drift", "B", "v7bot", motion="open", evidence="hold")
		for motion in spec["motions"]:
			if motion == "open":
				continue
			ref = group_name("realistic", "drift", "B", "v7bot", motion=motion, evidence="hold")
			out.append(_compare(stage, batch, base, ref))
	elif stage == "stage3":
		for motion in stage3_motions:
			for evidence in spec["evidences"]:
				for camera in spec["cameras"]:
					out.append(_run(stage, batch, suite, camera, motion, evidence, 0, "lab"))
	elif stage == "stage3b":
		for segmenter in SEGMENTERS:
			for camera in spec["cameras"]:
				out.append(_run(stage, batch, suite, camera, winner_motion, winner_evidence, 0, segmenter))
	elif stage == "stage4":
		for camera in spec["cameras"]:
			base = _run(stage, batch, confirm_suite, camera, "open", "hold", 0, "lab")
			win = _run(stage, batch, confirm_suite, camera, winner_motion, winner_evidence, 0, winner_segmenter)
			out.extend([base, win])
			if base["group"] != win["group"]:
				out.append(_compare(stage, batch, base["group"], win["group"]))
	return out


def _suite_path(manifest_path: Path, name: str, meta: dict) -> Path:
	rel = meta.get("path")
	base = manifest_path.parent
	if not rel:
		return base / f"{name}.json"
	p = Path(rel)
	if p.is_absolute():
		return p
	candidate = base / p
	if candidate.is_file():
		return candidate
	repo_candidate = REPO / p
	return repo_candidate if repo_candidate.is_file() else candidate


def _load_cases(path: Path) -> list:
	raw = json.loads(path.read_text(encoding="utf-8-sig"))
	return raw["cases"] if isinstance(raw, dict) else raw


def _seeds(cases: list) -> set[int]:
	return {c[0] for c in cases if isinstance(c, (list, tuple)) and c and type(c[0]) is int}


def _is_baseline(entry: dict) -> bool:
	return entry.get("motion") == "open" and entry.get("evidence") == "hold"


def validate(plans: list[dict], manifest_path) -> list[str]:
	"""Return human-readable errors; an empty list means the plan is runnable."""
	manifest_path = Path(manifest_path)
	try:
		manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
	except (OSError, ValueError) as exc:
		return [f"cannot read manifest {manifest_path}: {exc}"]
	suites_meta = manifest.get("suites", {})
	runs = [p for p in plans if p.get("kind") == "run"]
	compares = [p for p in plans if p.get("kind") == "compare"]
	errors: list[str] = []

	loaded: dict[str, list] = {}
	for name in sorted({p.get("suite") for p in runs if p.get("suite")}):
		if not isinstance(name, str) or not name or "/" in name or "\\" in name or name.endswith(".json"):
			errors.append(f"suite {name!r} must be a registered suite name, not a path")
			continue
		meta = suites_meta.get(name)
		if meta is None:
			errors.append(f"unknown suite {name!r} (not registered in {manifest_path})")
			continue
		path = _suite_path(manifest_path, name, meta)
		if not path.is_file():
			errors.append(f"suite file for {name!r} does not exist: {path}")
			continue
		if hashlib.sha256(path.read_bytes()).hexdigest() != meta.get("sha256"):
			errors.append(f"{name}: sha256 does not match the manifest")
			continue
		cases = _load_cases(path)
		if len(cases) != meta.get("cases"):
			errors.append(f"{name}: case count {len(cases)} != manifest {meta.get('cases')}")
			continue
		loaded[name] = cases

	for entry in runs:
		motion, evidence, segmenter = entry.get("motion"), entry.get("evidence"), entry.get("segmenter")
		if motion not in VARIANTS:
			errors.append(f"{entry.get('stage')}: unknown motion {motion!r} (expected one of {MOTIONS})")
		if evidence not in EVIDENCES:
			errors.append(f"{entry.get('stage')}: unknown evidence {evidence!r} (expected one of {EVIDENCES})")
		if segmenter not in SEGMENTERS:
			errors.append(f"{entry.get('stage')}: unknown segmenter {segmenter!r} (expected one of {SEGMENTERS})")
		if entry.get("camera") not in CAMERAS:
			errors.append(f"{entry.get('stage')}: unknown camera {entry.get('camera')!r} (expected one of {CAMERAS})")

	stage2 = [p for p in runs if p.get("stage") == "stage2"]
	if stage2:
		seen = {(p.get("motion"), p.get("camera")) for p in stage2}
		missing = [
			(m, c) for m in STAGES["stage2"]["motions"] for c in STAGES["stage2"]["cameras"] if (m, c) not in seen
		]
		if missing:
			errors.append(f"stage2 is missing {len(missing)} motion x camera runs: {missing[:5]}")

	produced: dict[str, set[str]] = defaultdict(set)
	for entry in runs:
		produced[entry.get("batch")].add(entry.get("group"))
	for entry in compares:
		for group in entry.get("compare") or []:
			if group not in produced.get(entry.get("batch"), set()):
				errors.append(
					f"{entry.get('stage')}: compare group {group!r} is not produced by a run in batch {entry.get('batch')!r}"
				)

	stage3 = [p for p in runs if p.get("stage") == "stage3"]
	if stage3 and not {p.get("motion") for p in stage3}:
		errors.append("stage3 matrix has no motions")
	late = [p for p in runs if p.get("stage") in ("stage3b", "stage4") and not _is_baseline(p)]
	if late:
		for motion in sorted({p.get("motion") for p in late}):
			if not motion:
				errors.append(f"{late[0].get('stage')}: winner motion is empty")
			elif motion not in STAGES["stage2"]["motions"]:
				errors.append(f"winner motion {motion!r} is not part of the stage2 matrix")
		for evidence in sorted({p.get("evidence") for p in late}):
			if not evidence:
				errors.append(f"{late[0].get('stage')}: winner evidence is empty")
			elif evidence not in STAGES["stage3"]["evidences"]:
				errors.append(f"winner evidence {evidence!r} is not part of the stage3 matrix")
		for segmenter in sorted({p.get("segmenter") for p in late}):
			if not segmenter:
				errors.append(f"{late[0].get('stage')}: winner segmenter is empty")

	stage4 = [p for p in runs if p.get("stage") == "stage4"]
	if stage4:
		baseline_cameras = {p.get("camera") for p in stage4 if _is_baseline(p)}
		for camera in STAGES["stage4"]["cameras"]:
			if camera not in baseline_cameras:
				errors.append(f"stage4 is missing the open+hold baseline for camera {camera!r}")
		confirm = {p.get("suite") for p in stage4}
		dev_seeds: set[int] = set()
		if "p2_dev12" in loaded:
			dev_seeds = _seeds(loaded["p2_dev12"])
		elif (meta := suites_meta.get("p2_dev12")) is not None:
			path = _suite_path(manifest_path, "p2_dev12", meta)
			if path.is_file():
				dev_seeds = _seeds(_load_cases(path))
		for name in sorted(confirm):
			if name not in loaded:
				continue
			seeds = _seeds(loaded[name])
			outside = sorted(s for s in seeds if split_of(s) != "loop-01")
			if outside:
				errors.append(f"{name}: seeds outside split loop-01: {outside[:5]}")
			overlap = sorted(seeds & dev_seeds)
			if overlap:
				errors.append(f"{name}: seeds overlap p2_dev12: {overlap[:5]}")
	return errors


def _fmt_offset(value) -> str:
	off = float(value or 0)
	return str(int(off)) if off.is_integer() else str(off)


def _runner_line(entry: dict) -> str:
	parts = [
		'"$PY"',
		"-m",
		"digital_twin_armigo.experiments.runner",
		"run",
		"--root",
		'"$ROOT"',
		"--batch",
		str(entry["batch"]),
		"--sensing",
		str(entry["sensing"]),
		"--camera",
		str(entry["camera"]),
		"--strategy",
		str(entry["strategy"]),
		"--suite",
		str(entry["suite"]),
		"--detector",
		str(entry["detector"]),
		"--motion",
		str(entry["motion"]),
		"--evidence",
		str(entry["evidence"]),
	]
	if float(entry.get("calib_offset_mm") or 0) != 0:
		parts += ["--calib-offset-mm", _fmt_offset(entry["calib_offset_mm"])]
	if entry.get("segmenter") != "lab" or entry.get("stage") == "stage3b":
		parts += ["--segmenter", str(entry["segmenter"])]
	parts += ["--workers", '"$W"']
	return " ".join(parts)


def _report_line(batch: str, compare: list[str] | None) -> str:
	parts = [
		'"$PY"',
		"-m",
		"digital_twin_armigo.experiments.report",
		"--root",
		'"$ROOT"',
		"--batch",
		str(batch),
	]
	if compare:
		parts += ["--compare", str(compare[0]), str(compare[1])]
	return " ".join(parts)


def emit_commands(plans: list[dict]) -> list[str]:
	lines = [_runner_line(p) for p in plans if p.get("kind") == "run"]
	batches: list[str] = []
	for entry in plans:
		if entry.get("batch") not in batches:
			batches.append(entry.get("batch"))
	compared = {p.get("batch") for p in plans if p.get("kind") == "compare"}
	for entry in plans:
		if entry.get("kind") == "compare":
			lines.append(_report_line(entry["batch"], entry.get("compare")))
	for batch in batches:
		if batch not in compared:
			lines.append(_report_line(batch, None))
	return lines


def summarize(stage: str, plans: list[dict], manifest: dict) -> dict:
	runs = [p for p in plans if p.get("kind") == "run" and p.get("stage") == stage]
	compares = [p.get("compare") for p in plans if p.get("kind") == "compare" and p.get("stage") == stage]
	suite = runs[0]["suite"] if runs else STAGES[stage]["suite"]
	batch = runs[0]["batch"] if runs else STAGES[stage]["batch"]
	meta = manifest.get("suites", {}).get(suite, {})
	return {
		"batch": batch,
		"suite": suite,
		"suite_sha256": meta.get("sha256"),
		"n_cases": meta.get("cases"),
		"n_runs": len(runs),
		"groups": sorted({p["group"] for p in runs}),
		"compares": compares,
	}


def main(argv=None) -> int:
	parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	group = parser.add_mutually_exclusive_group(required=True)
	group.add_argument("--plan-only", action="store_true", help="print a JSON summary after validating")
	group.add_argument("--emit-commands", action="store_true", help="print shell runner/report argument lines")
	parser.add_argument("--stage", default="all", choices=("all", *STAGE_ORDER))
	parser.add_argument("--manifest", default=str(MANIFEST))
	args = parser.parse_args(argv)

	stages = ALL_STAGES if args.stage == "all" else (args.stage,)
	manifest_path = Path(args.manifest)
	plans: list[dict] = []
	for stage in stages:
		plans.extend(plan(stage))
	errors = validate(plans, manifest_path)
	if errors:
		for error in errors:
			print(f"error: {error}", file=sys.stderr)
		return 1
	manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
	if args.emit_commands:
		for line in emit_commands(plans):
			print(line)
		return 0
	summary = {stage: summarize(stage, plans, manifest) for stage in stages}
	print(json.dumps(summary, indent=2, ensure_ascii=False))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
