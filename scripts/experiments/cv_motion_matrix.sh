#!/usr/bin/env bash
# Stages 1-4 of the CV motion & evidence matrix (Task C7 Step 2).
# Requires GPU detector/worker scheduling; do NOT run unattended during CPU-only handoff.
# Usage: bash scripts/experiments/cv_motion_matrix.sh [stage1|stage2|stage3|stage4|all] [workers]
set -euo pipefail
cd "$(dirname "$0")/../.."

PY=E:/digital_twin_armigo/.venv/Scripts/python
export PYTHONPATH=src
ROOT=E:/Armigo_data_v3/loop/cv-motion
STAGE=${1:-all}
W=${2:-4}
STAGE3_MOTIONS=${STAGE3_MOTIONS:-"open w2s"}
WINNER_MOTION=${WINNER_MOTION:-w2s}
WINNER_EVIDENCE=${WINNER_EVIDENCE:-hold}
CONFIRM_SUITE=${CONFIRM_SUITE:-config/agent_bench/v2/loop_01_96.json}

stage1() {
	for m in open w2s; do
		for d in 0 2 5 10 20; do
			"$PY" -m digital_twin_armigo.experiments.runner run \
				--root "$ROOT" --batch cvm-s1-tol --sensing realistic --camera fixed \
				--strategy B --suite p2_dev12 --detector v7bot \
				--motion "$m" --evidence hold --calib-offset-mm "$d" --workers "$W"
		done
	done
	"$PY" -m digital_twin_armigo.experiments.report --root "$ROOT" --batch cvm-s1-tol
}

stage2() {
	for m in open w1s w2s w3s w2a w3a w4a; do
		for cam in fixed drift; do
			"$PY" -m digital_twin_armigo.experiments.runner run \
				--root "$ROOT" --batch cvm-s2 --sensing realistic --camera "$cam" \
				--strategy B --suite p2_dev12 --detector v7bot \
				--motion "$m" --evidence hold --workers "$W"
		done
	done
	"$PY" -m digital_twin_armigo.experiments.report --root "$ROOT" --batch cvm-s2 \
		--compare realistic-drift-B-v7bot realistic-drift-B-v7bot-mw2s-ehold
}

stage3() {
	for m in $STAGE3_MOTIONS; do
		for ev in hold img dense-10 dense-20; do
			for cam in fixed recal drift; do
				"$PY" -m digital_twin_armigo.experiments.runner run \
					--root "$ROOT" --batch cvm-s3 --sensing realistic --camera "$cam" \
					--strategy B --suite p2_dev12 --detector v7bot \
					--motion "$m" --evidence "$ev" --workers "$W"
			done
		done
	done
	"$PY" -m digital_twin_armigo.experiments.report --root "$ROOT" --batch cvm-s3
}

stage4() {
	for cam in fixed recal drift; do
		"$PY" -m digital_twin_armigo.experiments.runner run \
			--root "$ROOT" --batch cvm-s4-confirm --sensing realistic --camera "$cam" \
			--strategy B --suite "$CONFIRM_SUITE" --detector v7bot \
			--motion open --evidence hold --workers "$W"
		"$PY" -m digital_twin_armigo.experiments.runner run \
			--root "$ROOT" --batch cvm-s4-confirm --sensing realistic --camera "$cam" \
			--strategy B --suite "$CONFIRM_SUITE" --detector v7bot \
			--motion "$WINNER_MOTION" --evidence "$WINNER_EVIDENCE" --workers "$W"
	done
	"$PY" -m digital_twin_armigo.experiments.report --root "$ROOT" --batch cvm-s4-confirm
}

case "$STAGE" in
	stage1) stage1 ;;
	stage2) stage2 ;;
	stage3) stage3 ;;
	stage4) stage4 ;;
	all) stage1; stage2; stage3; stage4 ;;
	*) echo "unknown stage: $STAGE" >&2; exit 2 ;;
esac
