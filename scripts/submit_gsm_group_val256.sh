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
for condition in group_answer_prompt group_answer_clamp; do
  short_condition=grp-prompt
  if [[ "$condition" == "group_answer_clamp" ]]; then
    short_condition=grp-clamp
  fi
  job_id=$(
    "$SBATCH" --parsable \
      --partition="$PARTITION" --exclude=g55 --nodes=1 --gres=gpu:1 \
      --cpus-per-task=8 --mem=80G --time=08:00:00 \
      --job-name="dopsd-${short_condition}-v256" \
      --output="$ROOT/logs/%x-%j.out" --error="$ROOT/logs/%x-%j.err" \
      --export="ALL,CONDITION=${condition},VALIDATION_SAMPLES=256,CHECKPOINT_STEPS=50 100 150 200 250 300,EVAL_TAG=group-val256-p1-s42,NUM_PROCESSES=1" \
      scripts/run_gsm_group_val256_condition.sh
  )
  printf '%s\t%s\n' "$condition" "$job_id"
done
