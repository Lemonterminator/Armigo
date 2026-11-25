"""Deterministic mapping from layout seeds to dataset/evaluation splits."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
LEGACY_UPPER = 3_000_000
DEFAULT_SEED_RANGES: dict[str, tuple[int, int]] = {
	"train": (3_000_001, 3_099_999),
	"val": (3_100_001, 3_199_999),
	"ood-color": (3_200_001, 3_299_999),
	"ood-cam": (3_300_001, 3_399_999),
	"ood-lang": (3_400_001, 3_499_999),
	"ood-appearance": (3_500_001, 3_599_999),
	"test": (3_900_001, 3_999_999),
}
LOOP_RANGE = (3_600_001, 3_899_999)
LOOP_BLOCK = 1000


def _manifest_ranges() -> dict[str, tuple[int, int]]:
	out = dict(DEFAULT_SEED_RANGES)
	for candidate in (
		REPO / "config" / "vla" / "v3" / "manifest.json",
		REPO / "config" / "agent_bench" / "v2" / "manifest.json",
	):
		if not candidate.is_file():
			continue
		try:
			data = json.loads(candidate.read_text(encoding="utf-8-sig"))
		except (OSError, ValueError):
			continue
		ranges = data.get("seed_ranges")
		if isinstance(ranges, dict):
			for name, bounds in ranges.items():
				if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
					out[str(name)] = (int(bounds[0]), int(bounds[1]))
	return out


def split_of(seed: int | None) -> str:
	"""Return split name for ``seed``: ``train``, ``val``, ``ood-*``, ``test``, ``loop-NN``, ``legacy``, ``unknown``."""
	if seed is None or isinstance(seed, bool):
		return "unknown"
	try:
		s = int(seed)
	except (TypeError, ValueError):
		return "unknown"
	if s < 0:
		return "unknown"
	if s < LEGACY_UPPER:
		return "legacy"
	if LOOP_RANGE[0] <= s <= LOOP_RANGE[1]:
		nn = (s - LOOP_RANGE[0]) // LOOP_BLOCK + 1
		return f"loop-{nn:02d}"
	for name, (lo, hi) in _manifest_ranges().items():
		if lo <= s <= hi:
			return name
	return "unknown"
