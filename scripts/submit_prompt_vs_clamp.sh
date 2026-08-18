#!/usr/bin/env bash
set -euo pipefail

PARTITION=${1:-a01}
MAX_STEPS=${2:-50}
SEED=${3:-42}
if [[ "$PARTITION" != "a01" && "$PARTITION" != "h01" ]]; then
  echo "Partition must be a01 or h01" >&2
  exit 2
fi

for condition in answer_prompt answer_clamp; do
  sbatch --partition="$PARTITION" \
    --export=ALL,CONDITION="$condition",MAX_STEPS="$MAX_STEPS",SEED="$SEED" \
    scripts/slurm/train_condition.sbatch
done
