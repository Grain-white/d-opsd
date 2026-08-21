#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/fit/alex1/WORK/Meiqi.Gu/d-opsd}
SBATCH=${SBATCH:-/rmprog/slurm/v24.05.1/bin/sbatch}
PARTITION=${PARTITION:-a01}

case "$PARTITION" in
  a01|h01) ;;
  *) echo "Refusing to submit outside a01/h01: $PARTITION" >&2; exit 2 ;;
esac

cd "$REPO_ROOT"
for condition in answer_prompt answer_clamp self_future; do
  run_name="${condition}-gsm-seed42-steps300-unified-v1"
  job_id=$(
    "$SBATCH" --parsable \
      --partition="$PARTITION" \
      --exclude=g55 \
      --job-name="dopsd-${condition}-gsm-u300" \
      --export="ALL,CONDITION=${condition},DATASET=gsm8k,SEED=42,MAX_STEPS=300,PASS_K=8,EVAL_PASS_K=1,RUN_NAME=${run_name},NUM_PROCESSES=4,SAVE_STEPS=50,EVAL_STEPS=10,VALIDATION_SAMPLES=32,EVAL_ON_START=true,BETA=1,RESUME_FROM_CHECKPOINT=false,MAX_COMPLETION_LENGTH=256,BLOCK_LENGTH=32,DIFFUSION_STEPS=128,BATCH_DIVIDE=8,NUM_ITERATIONS=8,GRADIENT_CHECKPOINTING=false,SWANLAB_MODE=cloud" \
      scripts/slurm/train_condition.sbatch
  )
  printf '%s\t%s\n' "$condition" "$job_id"
done
