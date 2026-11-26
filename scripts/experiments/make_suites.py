"""Write the v2 experiment suites and their manifest (plan 2026-11-24, sections 三 and 五).

	.venv/Scripts/python scripts/experiments/make_suites.py            # writes config/agent_bench/v2/
	.venv/Scripts/python scripts/experiments/make_suites.py --check    # verify files match the manifest

Suites (cases are ``[seed, positive, face]``; each layout carries one normal and one defective brick that share the
face index, as dev_suite_700001 does):

- ``smoke_600008``: the M0/M2 acceptance pair (bench16 cases 14/15).
- ``bench16``, ``dev_suite_700001``: copies of the existing regression and development suites.
- ``p1_sealed``: 24 new layouts (seeds 710001-710024), four defective bricks per face.
- ``p2_dev12``: 12 layouts of dev_suite_700001, two per face (phase-2 development).
- ``p2_sealed``: 24 new layouts (seeds 720001-720024), four per face; disjoint from p1_sealed.
- ``loop_01_96``: 48 loop-01 layouts (seeds 3600001-3600048), eight per face (Stage 4 confirmation).
- ``loop_02_400``: 200 new seed pairs from 3601001 for drift fix acceptance.
- ``probe_clutter_12``: six independent layouts with the existing v3 appearance and clutter sampler.
- ``round_00_clutter_60``: 30 new layouts (4100101–4100130) for the clutter baseline.

Seeds of the sealed suites are checked against every suite under config/ so a sealed layout never appeared before.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from digital_twin_armigo.governance.segments import split_of

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "config/agent_bench/v2"


def _paired(seeds: list[int]) -> list[list]:
	cases = []
	for i, seed in enumerate(seeds):
		face = i % 6
		cases += [[seed, False, face], [seed, True, face]]
	return cases


def _existing_seeds() -> set[int]:
	seeds: set[int] = set()
	for f in (ROOT / "config").rglob("*.json"):
		if OUT in f.parents:
			continue
		try:
			raw = json.loads(f.read_text(encoding="utf-8-sig"))
		except (ValueError, UnicodeDecodeError):
			continue
		rows = raw.get("cases") if isinstance(raw, dict) else raw
		if isinstance(rows, list):
			seeds |= {r[0] for r in rows if isinstance(r, list) and r and type(r[0]) is int}
	return seeds


def _clutter_cases(seeds: list[int]) -> list[list]:
	return [[*case, {"scene": "v3", "cameras": "fixed", "split": "train"}] for case in _paired(seeds)]


def build() -> dict[str, list]:
	bench16 = json.loads((ROOT / "config/agent_bench/bench16.json").read_text(encoding="utf-8"))
	dev = json.loads((ROOT / "config/agent_bench/dev_suite_700001.json").read_text(encoding="utf-8"))
	per_face: dict[int, list[int]] = {}
	for seed, positive, face in dev:
		if positive and len(per_face.setdefault(face, [])) < 2:
			per_face[face].append(seed)
	dev12_seeds = sorted(s for seeds in per_face.values() for s in seeds)
	dev12 = [c for c in dev if c[0] in dev12_seeds]
	suites = {
		"smoke_600008": [c for c in bench16 if c[0] == 600008],
		"bench16": bench16,
		"dev_suite_700001": dev,
		"p1_sealed": _paired(list(range(710001, 710025))),
		"p2_dev12": dev12,
		"p2_sealed": _paired(list(range(720001, 720025))),
		"loop_01_96": _paired(list(range(3_600_001, 3_600_049))),
		"loop_02_400": _paired(list(range(3_601_001, 3_601_201))),
		"probe_clutter_12": _clutter_cases(list(range(4_100_001, 4_100_007))),
		"round_00_clutter_60": _clutter_cases(list(range(4_100_101, 4_100_131))),
	}
	loop_seeds = {c[0] for c in suites["loop_01_96"]}
	assert len(loop_seeds) == 48
	assert all(split_of(seed) == "loop-01" for seed in loop_seeds), "loop_01_96 seeds must be in split loop-01"
	for name, cases in suites.items():
		if name == "loop_01_96":
			continue
		clash = loop_seeds & {c[0] for c in cases}
		assert not clash, f"loop_01_96 overlaps {name}: {sorted(clash)[:5]}"
	used = _existing_seeds()
	for name, expected_split in (
		("loop_02_400", "loop-02"),
		("probe_clutter_12", "unknown"),
		("round_00_clutter_60", "unknown"),
	):
		seeds = {c[0] for c in suites[name]}
		assert all(split_of(seed) == expected_split for seed in seeds), f"{name}: unexpected governance split"
		for other, cases in suites.items():
			if name != other:
				assert seeds.isdisjoint(c[0] for c in cases), f"{name} overlaps {other}"
		assert seeds.isdisjoint(used), f"{name} overlaps an existing config suite"
	for name in ("p1_sealed", "p2_sealed"):
		clash = {c[0] for c in suites[name]} & used
		if clash:
			raise SystemExit(f"{name}: seeds already used elsewhere: {sorted(clash)[:5]}")
	assert not {c[0] for c in suites["p1_sealed"]} & {c[0] for c in suites["p2_sealed"]}
	for name in ("p1_sealed", "p2_sealed", "p2_dev12", "dev_suite_700001", "loop_01_96", "round_00_clutter_60"):
		faces = [c[2] for c in suites[name] if c[1]]
		assert len({faces.count(f) for f in range(6)}) == 1, f"{name} is not face-balanced"
	return suites


def _dump(cases: list) -> str:
	return "[\n" + ",\n".join("  " + json.dumps(c) for c in cases) + "\n]\n"


def suite_manifest_entry(
	suite_path: Path,
	cases: list,
	*,
	sealed: bool = False,
	manifest_dir: Path | None = None,
	name: str | None = None,
) -> dict:
	entry: dict = {
		"sha256": hashlib.sha256(suite_path.read_bytes()).hexdigest(),
		"cases": len(cases),
		"layouts": len({c[0] for c in cases}),
		"sealed": bool(sealed),
	}
	if manifest_dir is not None:
		resolved_suite = suite_path.resolve()
		default_name = f"{name}.json" if name else resolved_suite.name
		if resolved_suite != (manifest_dir.resolve() / default_name):
			entry["path"] = Path(os.path.relpath(resolved_suite, manifest_dir.resolve())).as_posix()
	return entry


def register_suite(
	manifest_path: Path,
	name: str,
	suite_path: Path,
	cases: list | None = None,
	*,
	sealed: bool = False,
) -> dict:
	manifest_path = Path(manifest_path)
	suite_path = Path(suite_path)
	if cases is None:
		raw = json.loads(suite_path.read_text(encoding="utf-8-sig"))
		cases = raw["cases"] if isinstance(raw, dict) else raw
	manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {"suites": {}}
	suites = manifest.setdefault("suites", {})
	entry = suite_manifest_entry(
		suite_path,
		cases,
		sealed=sealed,
		manifest_dir=manifest_path.parent,
		name=name,
	)
	suites[name] = entry
	manifest_path.parent.mkdir(parents=True, exist_ok=True)
	manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
	return entry


def main() -> None:
	p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	p.add_argument("--check", action="store_true")
	a = p.parse_args()
	suites = build()
	manifest_path = OUT / "manifest.json"
	if a.check:
		manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
		for name, meta in manifest["suites"].items():
			digest = hashlib.sha256((OUT / f"{name}.json").read_bytes()).hexdigest()
			if digest != meta["sha256"]:
				raise SystemExit(f"{name}: sha256 differs from the manifest")
		print("all suites match the manifest")
		return
	OUT.mkdir(parents=True, exist_ok=True)
	manifest = {
		"plan": "docs/superpowers/plans/2026-11-24-dual-engine-six-face-agent-audit.md",
		"detectors": {
			# models-sup/sup.json names the same model but its checkpoint path does not resolve; sup-local.json is the
			# v6 config the agent benches ran with (sha 83128f68...).
			"v6": "results/agent-bench/sup-local.json",
			"v7bot": "results/anomaly/models-sup/sup-v7bot.json",
		},
		"suites": {},
	}
	for name, cases in suites.items():
		text = _dump(cases)
		suite_file = OUT / f"{name}.json"
		suite_file.write_text(text, encoding="utf-8", newline="\n")
		manifest["suites"][name] = suite_manifest_entry(
			suite_file,
			cases,
			sealed=name.endswith("_sealed"),
			manifest_dir=OUT,
			name=name,
		)
	manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
	print(json.dumps(manifest["suites"], indent=2))


if __name__ == "__main__":
	main()
