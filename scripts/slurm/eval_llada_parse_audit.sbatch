#!/usr/bin/env bash
#SBATCH --partition=a01
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=180G
#SBATCH --time=08:00:00
#SBATCH --job-name=llada-parse-audit
#SBATCH --output=/home/fit/alex1/WORK/Meiqi.Gu/d-opsd/logs/%x-%j.out
#SBATCH --error=/home/fit/alex1/WORK/Meiqi.Gu/d-opsd/logs/%x-%j.err

set -euo pipefail
case "${SLURM_JOB_PARTITION:-}" in
  a01|h01) ;;
  *) echo "Refusing to run outside a01/h01" >&2; exit 2 ;;
esac

ROOT=/home/fit/alex1/WORK/Meiqi.Gu/d-opsd
PY=$ROOT/.venv-dopsd-py312/bin/python
MODEL=$ROOT/outputs/models/LLaDA-8B-Instruct
OUT=$ROOT/outputs/eval/parse_audit_len${GEN_LENGTH:-512}
GEN_LENGTH=${GEN_LENGTH:-512}
DIFFUSION_STEPS=${DIFFUSION_STEPS:-256}
MAX_EXAMPLES=${MAX_EXAMPLES:-256}
NUM_PROCESSES=${NUM_PROCESSES:-4}
BATCH_SIZE=${BATCH_SIZE:-4}

export PYTHONNOUSERSITE=1
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
export GSM8K_LOCAL_DIR=$ROOT/outputs/datasets/gsm8k
export HF_HOME=$ROOT/outputs/hf_home_rw
export HF_DATASETS_CACHE=$ROOT/outputs/hf_datasets_rw
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT"

cd "$ROOT/eval"
run_eval() {
  local label=$1
  local checkpoint=$2
  local checkpoint_args=()
  if [[ -n "$checkpoint" ]]; then
    checkpoint_args=(--checkpoint_path "$checkpoint")
  fi
  "$PY" -m torch.distributed.run \
    --nproc_per_node "$NUM_PROCESSES" \
    --master_port $((18000 + SLURM_JOB_ID % 20000)) \
    eval.py --dataset gsm --batch_size "$BATCH_SIZE" \
    --gen_length "$GEN_LENGTH" --block_length 32 --diffusion_steps "$DIFFUSION_STEPS" \
    --max_examples "$MAX_EXAMPLES" --model_path "$MODEL" \
    --output_dir "$OUT" --temperature 0.0 --seed 42 --split test \
    --seeded_prefix_subset --suffix "$label" "${checkpoint_args[@]}"
}

case "${AUDIT_MODEL:-all}" in
  step0)
    run_eval step0 ""
    ;;
  answer_prompt)
    run_eval answer_prompt_step300 "$ROOT/outputs/checkpoints/answer_prompt-gsm-seed42-steps300-unified-v1/checkpoint-300"
    ;;
  answer_clamp)
    run_eval answer_clamp_step300 "$ROOT/outputs/checkpoints/answer_clamp-gsm-seed42-steps300-unified-v1/checkpoint-300"
    ;;
  all)
    run_eval step0 ""
    run_eval answer_prompt_step300 "$ROOT/outputs/checkpoints/answer_prompt-gsm-seed42-steps300-unified-v1/checkpoint-300"
    run_eval answer_clamp_step300 "$ROOT/outputs/checkpoints/answer_clamp-gsm-seed42-steps300-unified-v1/checkpoint-300"
    ;;
  *) echo "Unknown AUDIT_MODEL=${AUDIT_MODEL}" >&2; exit 2 ;;
esac
