"""Shared pytest configuration for local test runs.

Speed (see docs/TESTING.md): the suite is dominated by a few dozen MuJoCo end-to-end tests. Run it in parallel with
``pytest -n 8`` (pytest-xdist, in the dev group), or skip the long ones with ``pytest -m "not slow"``.
``tests/slow_tests.txt`` lists the node ids marked ``slow`` here in addition to explicit ``@pytest.mark.slow``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_SLOW_LIST = Path(__file__).with_name("slow_tests.txt")


def _slow_seconds() -> dict[str, float]:
	"""Node id -> serial seconds, from ``slow_tests.txt`` lines ``<seconds> <node id>``."""
	if not _SLOW_LIST.is_file():
		return {}
	out = {}
	for line in _SLOW_LIST.read_text(encoding="utf-8").splitlines():
		line = line.strip()
		if not line or line.startswith("#"):
			continue
		secs, _, nodeid = line.partition(" ")
		out[nodeid.strip()] = float(secs)
	return out


def pytest_configure(config):
	config.addinivalue_line("markers", "slow: slow end-to-end simulation test")


def pytest_collection_modifyitems(config, items):
	slow = _slow_seconds()
	if not slow:
		return
	for item in items:
		# Node ids are relative to the rootdir with forward slashes, as --durations prints them.
		if item.nodeid in slow:
			item.add_marker(pytest.mark.slow)
	# Under xdist, hand out the longest tests first so the run does not end waiting on one of them. A serial run keeps
	# file order, so module-scoped fixtures are built once.
	if getattr(config.option, "numprocesses", None):
		items.sort(key=lambda it: -slow.get(it.nodeid, 0.0))  # stable: the rest keep their order
