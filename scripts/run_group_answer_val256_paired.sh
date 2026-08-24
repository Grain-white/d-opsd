#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/fit/alex1/WORK/Meiqi.Gu/d-opsd}
CONDITION=${CONDITION:?Set CONDITION to group_answer_prompt or group_answer_clamp}
VALIDATION_SAMPLES=${VALIDATION_SAMPLES:-256}
# Put the required checkpoints first so a preempted/time-limited job still
# produces the minimum directly paired comparison requested for this study.
CHECKPOINT_STEPS=${CHECKPOINT_STEPS:-"100 200 300 50 150 250"}
EVAL_TAG=${EVAL_TAG:-val256-p1-seed42-v1}
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
train_run="${CONDITION}-gsm-seed42-steps300-group-donor-v2"
for step in $CHECKPOINT_STEPS; do
  checkpoint="$ROOT/outputs/checkpoints/$train_run/checkpoint-$step"
  test -d "$checkpoint"
  run_name="reeval-${CONDITION}-ckpt${step}-${EVAL_TAG}"
  output_dir="$ROOT/outputs/checkpoints/$run_name"
  done_marker="$output_dir/EVAL_COMPLETE"
  if [[ -f "$done_marker" ]] && [[ $(wc -l < "$output_dir/conditioning_samples.jsonl") -eq "$VALIDATION_SAMPLES" ]]; then
    echo "[$(date -Is)] SKIP complete $run_name"
    continue
  fi
  # A prior interrupted attempt must not be appended to: its rows would make
  # paired ordering ambiguous. Refuse explicitly instead of deleting evidence.
  if [[ -e "$output_dir/conditioning_samples.jsonl" ]]; then
    echo "Refusing partial/ambiguous output: $output_dir/conditioning_samples.jsonl" >&2
    exit 3
  fi
  echo "[$(date -Is)] START $run_name"
  env \
    CONDITION="$CONDITION" DATASET=gsm8k SEED=42 MAX_STEPS="$step" \
    PASS_K=8 EVAL_PASS_K=1 RUN_NAME="$run_name" NUM_PROCESSES="$NUM_PROCESSES" \
    SAVE_STEPS=100000 EVAL_STEPS=100000 VALIDATION_SAMPLES="$VALIDATION_SAMPLES" \
    EVAL_ON_START=true BETA=1 RESUME_FROM_CHECKPOINT="$checkpoint" \
    MAX_COMPLETION_LENGTH=256 BLOCK_LENGTH=32 DIFFUSION_STEPS=128 \
    BATCH_DIVIDE=8 NUM_ITERATIONS=8 GRADIENT_CHECKPOINTING=false \
    EVAL_ONLY=true SWANLAB_MODE=disabled \
    bash scripts/slurm/train_condition.sbatch
  test $(wc -l < "$output_dir/conditioning_samples.jsonl") -eq "$VALIDATION_SAMPLES"
  touch "$done_marker"
  echo "[$(date -Is)] DONE $run_name"
done
