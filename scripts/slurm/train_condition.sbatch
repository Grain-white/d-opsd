#!/usr/bin/env bash
#SBATCH --partition=a01
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --job-name=dopsd-cond
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

if [[ "${SLURM_JOB_PARTITION:-}" != "a01" && "${SLURM_JOB_PARTITION:-}" != "h01" ]]; then
  echo "Refusing to run outside a01/h01: ${SLURM_JOB_PARTITION:-unset}" >&2
  exit 2
fi

REPO_ROOT=${REPO_ROOT:-/home/fit/alex1/WORK/Meiqi.Gu/d-opsd}
PYTHON=${DOPSD_PYTHON:-$REPO_ROOT/.venv-dopsd-py312/bin/python}
MODEL_PATH=${MODEL_PATH:-GSAI-ML/LLaDA-8B-Instruct}
CONDITION=${CONDITION:-answer_clamp}
SEED=${SEED:-42}
MAX_STEPS=${MAX_STEPS:-50}
PASS_K=${PASS_K:-8}
EVAL_PASS_K=${EVAL_PASS_K:-1}
RUN_NAME=${RUN_NAME:-${CONDITION}-seed${SEED}-steps${MAX_STEPS}}
NUM_PROCESSES=${NUM_PROCESSES:-4}
SAVE_STEPS=${SAVE_STEPS:-50}
EVAL_STEPS=${EVAL_STEPS:-10}
VALIDATION_SAMPLES=${VALIDATION_SAMPLES:-32}
EVAL_ON_START=${EVAL_ON_START:-true}
BETA=${BETA:-1}

case "$CONDITION" in
  self_future|answer_prompt|answer_clamp|answer_clamp_future) ;;
  *) echo "Unknown CONDITION=$CONDITION" >&2; exit 2 ;;
esac

export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="/WORK/PUBLIC/alex_work/miniconda3/envs/sdpo-full/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HOME=${DOPSD_HF_HOME:-$REPO_ROOT/outputs/hf_home}
export HF_DATASETS_CACHE=${DOPSD_HF_DATASETS_CACHE:-$REPO_ROOT/outputs/hf_datasets}
export HUGGINGFACE_HUB_CACHE=${DOPSD_HUB_CACHE:-$HF_HOME/hub}
export TRANSFORMERS_CACHE=${DOPSD_TRANSFORMERS_CACHE:-$HF_HOME/transformers}
export HF_MODULES_CACHE=${DOPSD_HF_MODULES_CACHE:-$HF_HOME/modules}
export GSM8K_LOCAL_DIR=${GSM8K_LOCAL_DIR:-$REPO_ROOT/outputs/datasets/gsm8k}
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=${WANDB_MODE:-offline}
export SWANLAB_MODE=${SWANLAB_MODE:-cloud}
export DOPSD_SWANLAB_PROJECT=${DOPSD_SWANLAB_PROJECT:-d-opsd-prompt-vs-clamp}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/dopsd-${SLURM_JOB_ID}/triton}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/tmp/dopsd-${SLURM_JOB_ID}/torchinductor}

mkdir -p "$REPO_ROOT/logs" "$REPO_ROOT/outputs" "$HF_DATASETS_CACHE" \
  "$HUGGINGFACE_HUB_CACHE" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
cd "$REPO_ROOT/d-opsd"

"$PYTHON" -m accelerate.commands.launch \
  --config_file accelerate.yaml \
  --num_processes "$NUM_PROCESSES" \
  --main_process_port $((12000 + SLURM_JOB_ID % 20000)) \
  d_opsd_train.py \
  --config opsd.yaml \
  --model_path "$MODEL_PATH" \
  --teacher_conditioning "$CONDITION" \
  --rollout_filter correct_only \
  --passk "$PASS_K" \
  --eval_passk "$EVAL_PASS_K" \
  --seed "$SEED" \
  --max_steps "$MAX_STEPS" \
  --num_iterations 8 \
  --batch_divide 8 \
  --beta "$BETA" \
  --fixed_teacher \
  --save_steps "$SAVE_STEPS" \
  --eval_strategy steps \
  --eval_steps "$EVAL_STEPS" \
  --eval_on_start "$EVAL_ON_START" \
  --validation_samples "$VALIDATION_SAMPLES" \
  --run_name "$RUN_NAME" \
  --output_dir "$REPO_ROOT/outputs/checkpoints/$RUN_NAME" \
  --report_to none
