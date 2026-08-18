#!/usr/bin/env bash
#SBATCH --partition=a01
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --job-name=dopsd-eval
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

if [[ "${SLURM_JOB_PARTITION:-}" != "a01" && "${SLURM_JOB_PARTITION:-}" != "h01" ]]; then
  echo "Refusing to run outside a01/h01: ${SLURM_JOB_PARTITION:-unset}" >&2
  exit 2
fi

REPO_ROOT=${REPO_ROOT:-/home/fit/alex1/WORK/Meiqi.Gu/d-opsd}
PYTHON=${DOPSD_PYTHON:-$REPO_ROOT/.venv-dopsd-py312/bin/python}
MODEL_PATH=${MODEL_PATH:-$REPO_ROOT/outputs/models/LLaDA-8B-Instruct}
CHECKPOINT_PATH=${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to a LoRA checkpoint}
EVAL_SEED=${EVAL_SEED:-42}
NUM_PROCESSES=${NUM_PROCESSES:-4}
BATCH_SIZE=${BATCH_SIZE:-8}
GEN_LENGTH=${GEN_LENGTH:-256}
BLOCK_LENGTH=${BLOCK_LENGTH:-32}
DIFFUSION_STEPS=${DIFFUSION_STEPS:-128}
MAX_EXAMPLES=${MAX_EXAMPLES:--1}

export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="/WORK/PUBLIC/alex_work/miniconda3/envs/sdpo-full/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
export TOKENIZERS_PARALLELISM=false
export HF_HOME=${HF_HOME:-$REPO_ROOT/outputs/hf_home}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$REPO_ROOT/outputs/hf_datasets}
export GSM8K_LOCAL_DIR=${GSM8K_LOCAL_DIR:-$REPO_ROOT/outputs/datasets/gsm8k}
mkdir -p "$REPO_ROOT/logs" "$REPO_ROOT/outputs/eval"
cd "$REPO_ROOT/eval"

"$PYTHON" -m torch.distributed.run \
  --nproc_per_node "$NUM_PROCESSES" \
  --master_port $((22000 + SLURM_JOB_ID % 20000)) \
  eval.py \
  --dataset gsm \
  --batch_size "$BATCH_SIZE" \
  --gen_length "$GEN_LENGTH" \
  --block_length "$BLOCK_LENGTH" \
  --diffusion_steps "$DIFFUSION_STEPS" \
  --max_examples "$MAX_EXAMPLES" \
  --model_path "$MODEL_PATH" \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --output_dir "$REPO_ROOT/outputs/eval" \
  --temperature 0.0 \
  --seed "$EVAL_SEED" \
  --split test
