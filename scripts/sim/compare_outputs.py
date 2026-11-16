"""Compare simulation/render directories; fail closed on missing files or changed data.

RGB tolerance is opt-in. All other arrays, dtypes, shapes and JSON metadata must match exactly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

RGB_KEYS = {"front", "wrist", "top_rgb_v2"}


def compare_outputs(reference: Path, candidate: Path, rgb_atol: int = 0) -> dict:
	if not 0 <= rgb_atol <= 255:
		raise ValueError("rgb_atol must be between 0 and 255")
	errors, differences = [], []
	files = [{p.name for p in root.iterdir() if p.suffix in {".npz", ".json"}} for root in (reference, candidate)]
	if files[0] != files[1]:
		errors.append({"missing": sorted(files[0] - files[1]), "extra": sorted(files[1] - files[0])})
	if not any(name.endswith(".npz") for name in files[0]):
		errors.append("No reference NPZ episodes")
	for name in sorted(files[0] & files[1]):
		a, b = reference / name, candidate / name
		if a.suffix == ".json":
			if json.loads(a.read_text(encoding="utf-8")) != json.loads(b.read_text(encoding="utf-8")):
				errors.append(f"{name}: metadata differs")
			continue
		with np.load(a, allow_pickle=False) as left, np.load(b, allow_pickle=False) as right:
			if set(left.files) != set(right.files):
				errors.append(f"{name}: array keys differ")
			for key in sorted(set(left.files) & set(right.files)):
				x, y = left[key], right[key]
				if x.shape != y.shape or x.dtype != y.dtype:
					errors.append(f"{name}:{key}: shape or dtype differs")
					continue
				if np.array_equal(x, y):
					continue
				count = int(np.count_nonzero(x != y))
				# Only uint8 camera RGB may use a tolerance; state/depth/labels remain exact.
				is_rgb = key in RGB_KEYS and x.dtype == np.uint8
				max_abs = int(np.max(np.abs(x.astype(np.int16) - y.astype(np.int16)))) if is_rgb else None
				differences.append(
					{"file": name, "key": key, "changed": count, "total": int(x.size), "max_abs": max_abs}
				)
				if not is_rgb or max_abs > rgb_atol:
					errors.append(f"{name}:{key}: values differ")
	return {
		"passed": not errors,
		"bitwise_equal": not errors and not differences,
		"rgb_atol": rgb_atol,
		"errors": errors,
		"differences": differences,
		"files_checked": len(files[0] & files[1]),
	}


def main() -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("reference", type=Path)
	parser.add_argument("candidate", type=Path)
	parser.add_argument("--rgb-atol", type=int, default=0)
	parser.add_argument("--report", type=Path)
	args = parser.parse_args()
	result = compare_outputs(args.reference, args.candidate, args.rgb_atol)
	payload = json.dumps(result, indent=2) + "\n"
	if args.report:
		args.report.write_text(payload, encoding="utf-8")
	print(payload, end="")
	return 0 if result["passed"] else 1


if __name__ == "__main__":
	raise SystemExit(main())
