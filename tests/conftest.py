"""Shared pytest configuration for local test runs."""

from __future__ import annotations


def pytest_configure(config):
	config.addinivalue_line("markers", "slow: slow end-to-end simulation test")
