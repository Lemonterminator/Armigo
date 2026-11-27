"""Deterministic mapping from layout seeds to dataset/evaluation splits."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
_CASE_SEED_RE = re.compile(r"^L(\d+)(?:[-_]|$)")
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


def seed_of_path_or_case(
	ep_path: Path | None = None,
	task: Mapping[str, Any] | None = None,
	row: Mapping[str, Any] | None = None,
) -> int | None:
	"""Extract layout seed from ``task.json`` case, scored row ``layout_id``/``case_id``, or episode directory name."""
	if isinstance(task, Mapping):
		case = task.get("case")
		if isinstance(case, (list, tuple)) and case:
			try:
				return int(case[0])
			except (TypeError, ValueError):
				pass
	if isinstance(row, Mapping):
		for key in ("layout_id", "case_id"):
			val = str(row.get(key) or "")
			m = _CASE_SEED_RE.match(val)
			if m:
				return int(m.group(1))
	if ep_path is not None:
		m = _CASE_SEED_RE.match(Path(ep_path).name)
		if m:
			return int(m.group(1))
	return None


def is_sealed(
	seed: int | None,
	*,
	batch: str | None = None,
	options: Mapping[str, Any] | None = None,
) -> bool:
	"""Return True if an episode/case belongs to a sealed test split by seed range, batch name, or case options."""
	if split_of(seed) == "test":
		return True
	if batch is not None and "sealed" in str(batch).lower():
		return True
	if isinstance(options, Mapping):
		if options.get("sealed") is True:
			return True
		if str(options.get("split") or "") == "test":
			return True
	return False


def case_split(seed: int | None, options: Mapping[str, Any] | None = None) -> str:
	"""Cross-check seed-derived split with optional 4th-element case options."""
	s_split = split_of(seed)
	if is_sealed(seed, options=options):
		return "test"
	if isinstance(options, Mapping):
		opt_split = options.get("split")
		if opt_split is not None and str(opt_split) != "":
			opt_s = str(opt_split)
			if s_split not in ("legacy", "unknown") and opt_s != s_split:
				return f"mismatch:{s_split}:{opt_s}"
			return opt_s
	return s_split


def assert_train_case(case: Any, *, allow_legacy: bool = True) -> None:
	"""Validate one case entry (seed or tuple) against unified train-only split and sealed rules."""
	opts: Mapping[str, Any] | None = None
	if isinstance(case, (list, tuple)):
		if not case:
			raise ValueError("empty case entry")
		seed = int(case[0])
		if len(case) >= 4 and isinstance(case[3], Mapping):
			opts = case[3]
	else:
		seed = int(case)

	if is_sealed(seed, options=opts):
		raise ValueError(f"forbidden sealed/test layout seed {seed} (options={dict(opts) if opts else None})")

	s_split = split_of(seed)
	if isinstance(opts, Mapping):
		opt_split = opts.get("split")
		if opt_split is not None and str(opt_split) != "train":
			raise ValueError(f"forbidden non-train case option split={opt_split!r} for seed {seed}")

	if s_split == "legacy":
		if not allow_legacy:
			raise ValueError(f"forbidden legacy seed {seed} where v3 train split is required")
		return

	if s_split != "train":
		raise ValueError(f"forbidden non-train seed {seed} (split={s_split!r})")
