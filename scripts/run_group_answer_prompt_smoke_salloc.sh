#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/fit/alex1/WORK/Meiqi.Gu/d-opsd}
SLURM_BIN=${SLURM_BIN:-/rmprog/slurm/v24.05.1/bin}
PARTITION=${PARTITION:-a01}
SMOKE_TAG=${SMOKE_TAG:-group-prompt-smoke-v1}

case "$PARTITION" in
  a01|h01) ;;
  *) echo "Refusing to run outside a01/h01: $PARTITION" >&2; exit 2 ;;
esac

if [[ "${1:-}" != "--inside-allocation" ]]; then
  exec "$SLURM_BIN/salloc" \
    --nodes=1 --ntasks=1 --partition="$PARTITION" --exclude=g55 \
    --gres=gpu:1 --cpus-per-task=8 --mem=80G --time=02:00:00 \
    --job-name=dopsd-group-prompt-smoke \
    "$SLURM_BIN/srun" --export=ALL "$0" --inside-allocation
fi

cd "$REPO_ROOT"
run_name="group_answer_prompt-gsm-seed42-${SMOKE_TAG}"
env \
  CONDITION=group_answer_prompt DATASET=gsm8k SEED=42 MAX_STEPS=1 \
  PASS_K=8 EVAL_PASS_K=1 RUN_NAME="$run_name" NUM_PROCESSES=1 \
  SAVE_STEPS=1 EVAL_STEPS=1 VALIDATION_SAMPLES=2 EVAL_ON_START=true \
  BETA=1 RESUME_FROM_CHECKPOINT=false MAX_COMPLETION_LENGTH=256 \
  BLOCK_LENGTH=32 DIFFUSION_STEPS=128 BATCH_DIVIDE=8 NUM_ITERATIONS=8 \
  GRADIENT_CHECKPOINTING=false SWANLAB_MODE=disabled \
  bash scripts/slurm/train_condition.sbatch
