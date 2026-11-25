#!/usr/bin/env bash
# Stages 1-4 + 3b of the CV motion & evidence matrix (Task C7 Step 2 + Task U4 Stage 3b).
# Requires GPU detector/worker scheduling; do NOT run unattended during CPU-only handoff.
# The matrix itself lives in scripts/experiments/cv_motion_plan.py (single source); this script validates the plan
# first and then runs the emitted runner/report lines.
# `all` = stage1 stage2 stage3 stage4; stage3b is opt-in (pass it explicitly).
# Usage: bash scripts/experiments/cv_motion_matrix.sh [--plan-only] [stage1|stage2|stage3|stage3b|stage4|all] [workers]
#        PLAN_ONLY=1 bash scripts/experiments/cv_motion_matrix.sh [stage...]   # validate only
set -euo pipefail
cd "$(dirname "$0")/../.."

PY=E:/digital_twin_armigo/.venv/Scripts/python
export PYTHONPATH=src
ROOT=E:/Armigo_data_v3/loop/cv-motion
PLAN=scripts/experiments/cv_motion_plan.py
PLAN_ONLY=${PLAN_ONLY:-0}

if [[ "${1:-}" == "--plan-only" ]]; then
	PLAN_ONLY=1
	shift
fi
STAGE=${1:-all}
W=${2:-4}
export STAGE3_MOTIONS=${STAGE3_MOTIONS:-"open w2s"}
export WINNER_MOTION=${WINNER_MOTION:-w2s}
export WINNER_EVIDENCE=${WINNER_EVIDENCE:-hold}
export WINNER_SEGMENTER=${WINNER_SEGMENTER:-lab}
export CONFIRM_SUITE=${CONFIRM_SUITE:-loop_01_96}

case "$STAGE" in
	stage1|stage2|stage3|stage3b|stage4) STAGES_TO_RUN=$STAGE ;;
	all) STAGES_TO_RUN="stage1 stage2 stage3 stage4" ;;
	*) echo "unknown stage: $STAGE" >&2; exit 2 ;;
esac

for s in $STAGES_TO_RUN; do
	"$PY" "$PLAN" --plan-only --stage "$s" >/dev/null
done

if [[ "$PLAN_ONLY" == "1" ]]; then
	echo "plan validation passed: $STAGES_TO_RUN"
	exit 0
fi

for s in $STAGES_TO_RUN; do
	echo "== $s =="
	while IFS= read -r line; do
		eval "$line"
	done < <("$PY" "$PLAN" --emit-commands --stage "$s")
done
