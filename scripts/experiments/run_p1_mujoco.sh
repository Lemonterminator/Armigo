#!/usr/bin/env bash
# Phase 1, MuJoCo omniscient: regression (bench16) and development (dev_suite_700001) for both detectors.
# Resumable: rerunning skips finished cases. Usage: bash scripts/experiments/run_p1_mujoco.sh [workers]
set -u
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python
W=${1:-4}
for det in v7bot v6; do
	$PY -m digital_twin_armigo.experiments.runner run --batch p1-bench16 --engine mujoco --sensing truth \
		--suite bench16 --detector $det --workers "$W"
	$PY -m digital_twin_armigo.experiments.runner run --batch p1-dev --engine mujoco --sensing truth \
		--suite dev_suite_700001 --detector $det --workers "$W"
done
for b in p1-bench16 p1-dev; do
	$PY -m digital_twin_armigo.experiments.report --root D:/Armigo_data_v2 --batch $b --compare truth-v6 truth-v7bot
done
