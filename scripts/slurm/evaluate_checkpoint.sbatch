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

export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="/WORK/PUBLIC/alex_work/miniconda3/envs/sdpo-full/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
export TOKENIZERS_PARALLELISM=false
mkdir -p "$REPO_ROOT/logs" "$REPO_ROOT/outputs/eval"
cd "$REPO_ROOT/eval"

"$PYTHON" -m torch.distributed.run \
  --nproc_per_node 4 \
  --master_port $((22000 + SLURM_JOB_ID % 20000)) \
  eval.py \
  --dataset gsm \
  --batch_size 8 \
  --gen_length 256 \
  --block_length 32 \
  --diffusion_steps 128 \
  --model_path "$MODEL_PATH" \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --output_dir "$REPO_ROOT/outputs/eval" \
  --temperature 0.0 \
  --seed "$EVAL_SEED" \
  --split test
