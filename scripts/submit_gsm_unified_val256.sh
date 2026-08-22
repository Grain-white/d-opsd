#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/fit/alex1/WORK/Meiqi.Gu/d-opsd}
SBATCH=${SBATCH:-/rmprog/slurm/v24.05.1/bin/sbatch}
PARTITION=${PARTITION:-a01}

case "$PARTITION" in
  a01|h01) ;;
  *) echo "Refusing to submit outside a01/h01" >&2; exit 2 ;;
esac

cd "$ROOT"
for condition in answer_prompt answer_clamp self_future; do
  job_id=$(
    "$SBATCH" --parsable \
      --partition="$PARTITION" --exclude=g55 --nodes=1 --gres=gpu:1 \
      --cpus-per-task=8 --mem=80G --time=08:00:00 \
      --job-name="dopsd-${condition}-v256" \
      --output="$ROOT/logs/%x-%j.out" --error="$ROOT/logs/%x-%j.err" \
      --export="ALL,CONDITION=${condition},VALIDATION_SAMPLES=256,CHECKPOINT_STEPS=50 100 150 200 250 300,EVAL_TAG=val256-p1-seed42-v1,NUM_PROCESSES=1" \
      scripts/run_gsm_unified_val256_condition.sh
  )
  printf '%s\t%s\n' "$condition" "$job_id"
done
