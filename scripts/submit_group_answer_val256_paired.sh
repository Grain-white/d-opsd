#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/fit/alex1/WORK/Meiqi.Gu/d-opsd}
SBATCH=${SBATCH:-/rmprog/slurm/v24.05.1/bin/sbatch}
CONDITION=${1:?Usage: $0 CONDITION PARTITION}
PARTITION=${2:?Usage: $0 CONDITION PARTITION}

case "$CONDITION" in
  group_answer_prompt|group_answer_clamp) ;;
  *) echo "Unknown CONDITION=$CONDITION" >&2; exit 2 ;;
esac
case "$PARTITION" in
  a01|h01) ;;
  *) echo "Refusing to submit outside a01/h01" >&2; exit 2 ;;
esac

cd "$ROOT"
"$SBATCH" --parsable \
  --partition="$PARTITION" --exclude=g55 --nodes=1 --gres=gpu:1 \
  --cpus-per-task=8 --mem=80G --time=06:00:00 \
  --job-name="dopsd-${CONDITION}-v256" \
  --output="$ROOT/logs/%x-%j.out" --error="$ROOT/logs/%x-%j.err" \
  --export="ALL,CONDITION=${CONDITION},VALIDATION_SAMPLES=256,CHECKPOINT_STEPS=100 200 300 50 150 250,EVAL_TAG=val256-p1-seed42-v1,NUM_PROCESSES=1" \
  scripts/run_group_answer_val256_paired.sh
