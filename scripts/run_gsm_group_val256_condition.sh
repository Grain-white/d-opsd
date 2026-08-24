#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/fit/alex1/WORK/Meiqi.Gu/d-opsd}
CONDITION=${CONDITION:?Set CONDITION}
VALIDATION_SAMPLES=${VALIDATION_SAMPLES:-256}
CHECKPOINT_STEPS=${CHECKPOINT_STEPS:-"50 100 150 200 250 300"}
EVAL_TAG=${EVAL_TAG:-group-val256-p1-s42}
NUM_PROCESSES=${NUM_PROCESSES:-1}

case "${SLURM_JOB_PARTITION:-}" in
  a01|h01) ;;
  *) echo "Refusing to run outside a01/h01" >&2; exit 2 ;;
esac
case "$CONDITION" in
  group_answer_prompt|group_answer_clamp) ;;
  *) echo "Unknown CONDITION=$CONDITION" >&2; exit 2 ;;
esac

cd "$ROOT"
for step in $CHECKPOINT_STEPS; do
  checkpoint="$ROOT/outputs/checkpoints/${CONDITION}-gsm-seed42-steps300-group-donor-v2/checkpoint-${step}"
  test -d "$checkpoint"
  run_name="reeval-${CONDITION}-ckpt${step}-${EVAL_TAG}"
  echo "[$(date -Is)] START $run_name"
  env \
    CONDITION="$CONDITION" DATASET=gsm8k SEED=42 MAX_STEPS="$step" \
    PASS_K=8 EVAL_PASS_K=1 RUN_NAME="$run_name" NUM_PROCESSES="$NUM_PROCESSES" \
    SAVE_STEPS=100000 EVAL_STEPS=100000 VALIDATION_SAMPLES="$VALIDATION_SAMPLES" \
    EVAL_ON_START=true BETA=1 RESUME_FROM_CHECKPOINT="$checkpoint" \
    MAX_COMPLETION_LENGTH=256 BLOCK_LENGTH=32 DIFFUSION_STEPS=128 \
    BATCH_DIVIDE=8 NUM_ITERATIONS=8 GRADIENT_CHECKPOINTING=false \
    EVAL_ONLY=true SWANLAB_MODE=cloud \
    bash scripts/slurm/train_condition.sbatch
  echo "[$(date -Is)] DONE $run_name"
done
