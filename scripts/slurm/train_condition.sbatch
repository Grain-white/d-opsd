#!/usr/bin/env bash
#SBATCH --partition=a01
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --job-name=dopsd-cond
#SBATCH --output=/home/fit/alex1/WORK/Meiqi.Gu/d-opsd/logs/%x-%j.out
#SBATCH --error=/home/fit/alex1/WORK/Meiqi.Gu/d-opsd/logs/%x-%j.err

set -euo pipefail

if [[ "${SLURM_JOB_PARTITION:-}" != "a01" && "${SLURM_JOB_PARTITION:-}" != "h01" ]]; then
  echo "Refusing to run outside a01/h01: ${SLURM_JOB_PARTITION:-unset}" >&2
  exit 2
fi

REPO_ROOT=${REPO_ROOT:-/home/fit/alex1/WORK/Meiqi.Gu/d-opsd}
PYTHON=${DOPSD_PYTHON:-$REPO_ROOT/.venv-dopsd-py312/bin/python}
MODEL_PATH=${MODEL_PATH:-$REPO_ROOT/outputs/models/LLaDA-8B-Instruct}
CONDITION=${CONDITION:-answer_clamp}
DATASET=${DATASET:-gsm8k}
SEED=${SEED:-42}
MAX_STEPS=${MAX_STEPS:-50}
PASS_K=${PASS_K:-1}
EVAL_PASS_K=${EVAL_PASS_K:-1}
RUN_NAME=${RUN_NAME:-${CONDITION}-${DATASET}-seed${SEED}-steps${MAX_STEPS}}
NUM_PROCESSES=${NUM_PROCESSES:-4}
SAVE_STEPS=${SAVE_STEPS:-50}
EVAL_STEPS=${EVAL_STEPS:-10}
VALIDATION_SAMPLES=${VALIDATION_SAMPLES:-32}
EVAL_ON_START=${EVAL_ON_START:-true}
BETA=${BETA:-1}
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-false}
MAX_COMPLETION_LENGTH=${MAX_COMPLETION_LENGTH:-256}
BLOCK_LENGTH=${BLOCK_LENGTH:-32}
DIFFUSION_STEPS=${DIFFUSION_STEPS:-128}
BATCH_DIVIDE=${BATCH_DIVIDE:-8}
NUM_ITERATIONS=${NUM_ITERATIONS:-$BATCH_DIVIDE}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-false}
EVAL_ONLY=${EVAL_ONLY:-false}

case "$CONDITION" in
  self_future|answer_prompt|group_answer_prompt|answer_clamp|answer_clamp_future) ;;
  *) echo "Unknown CONDITION=$CONDITION" >&2; exit 2 ;;
esac

export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="/WORK/PUBLIC/alex_work/miniconda3/envs/sdpo-full/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy

# Load SWANLAB_API_KEY (and related) for non-interactive compute nodes.
for env_file in \
  "${DOPSD_ENV_FILE:-}" \
  "$HOME/WORK/Meiqi.Gu/.env" \
  "/WORK/PUBLIC/alex_work/Meiqi.Gu/.env" \
  "$REPO_ROOT/.env"
do
  if [[ -n "$env_file" && -f "$env_file" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$env_file"
    set +a
    break
  fi
done

export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
# Avoid root-owned incomplete caches from earlier failed hub downloads.
export HF_HOME=${DOPSD_HF_HOME:-$REPO_ROOT/outputs/hf_home_rw}
export HF_DATASETS_CACHE=${DOPSD_HF_DATASETS_CACHE:-$REPO_ROOT/outputs/hf_datasets_rw}
export HUGGINGFACE_HUB_CACHE=${DOPSD_HUB_CACHE:-$HF_HOME/hub}
export TRANSFORMERS_CACHE=${DOPSD_TRANSFORMERS_CACHE:-$HF_HOME/transformers}
export HF_MODULES_CACHE=${DOPSD_HF_MODULES_CACHE:-$HF_HOME/modules}
export GSM8K_LOCAL_DIR=${GSM8K_LOCAL_DIR:-$REPO_ROOT/outputs/datasets/gsm8k}
export MATH_LOCAL_DIR=${MATH_LOCAL_DIR:-$REPO_ROOT/outputs/datasets/math500}
# Local parquet + local MODEL_PATH: do not hit hub CDN on compute nodes.
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=${WANDB_MODE:-offline}
export SWANLAB_MODE=${SWANLAB_MODE:-cloud}
export DOPSD_SWANLAB_PROJECT=${DOPSD_SWANLAB_PROJECT:-d-opsd-prompt-vs-clamp}
# Optional: continue an existing SwanLab cloud run (set both when resuming training).
if [[ -n "${SWANLAB_RUN_ID:-}" ]]; then
  export SWANLAB_RUN_ID
  export SWANLAB_RESUME=${SWANLAB_RESUME:-must}
fi
if [[ "${SWANLAB_MODE}" != "offline" && "${SWANLAB_MODE}" != "disabled" && -z "${SWANLAB_API_KEY:-}" ]]; then
  echo "SWANLAB_API_KEY unset; cloud mode needs a key (no TTY on compute nodes)." >&2
  exit 2
fi
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/dopsd-${SLURM_JOB_ID}/triton}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/tmp/dopsd-${SLURM_JOB_ID}/torchinductor}

mkdir -p "$REPO_ROOT/logs" "$REPO_ROOT/outputs" "$HF_DATASETS_CACHE" \
  "$HUGGINGFACE_HUB_CACHE" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
cd "$REPO_ROOT/d-opsd"

ACCELERATE_CONFIG_FILE=accelerate.yaml
if [[ "$EVAL_ONLY" == "true" ]]; then
  ACCELERATE_CONFIG_FILE=accelerate_eval.yaml
  NUM_PROCESSES=1
fi
"$PYTHON" -m accelerate.commands.launch \
  --config_file "$ACCELERATE_CONFIG_FILE" \
  --num_processes "$NUM_PROCESSES" \
  --main_process_port $((12000 + SLURM_JOB_ID % 20000)) \
  d_opsd_train.py \
  --config opsd.yaml \
  --model_path "$MODEL_PATH" \
  --dataset "$DATASET" \
  --teacher_conditioning "$CONDITION" \
  --rollout_filter correct_only \
  --passk "$PASS_K" \
  --eval_passk "$EVAL_PASS_K" \
  --seed "$SEED" \
  --max_steps "$MAX_STEPS" \
  --num_iterations "$NUM_ITERATIONS" \
  --batch_divide "$BATCH_DIVIDE" \
  --beta "$BETA" \
  --fixed_teacher \
  --gradient_checkpointing "$GRADIENT_CHECKPOINTING" \
  --save_steps "$SAVE_STEPS" \
  --eval_strategy steps \
  --eval_steps "$EVAL_STEPS" \
  --eval_on_start "$EVAL_ON_START" \
  --validation_samples "$VALIDATION_SAMPLES" \
  --max_completion_length "$MAX_COMPLETION_LENGTH" \
  --block_length "$BLOCK_LENGTH" \
  --diffusion_steps "$DIFFUSION_STEPS" \
  --run_name "$RUN_NAME" \
  --output_dir "$REPO_ROOT/outputs/checkpoints/$RUN_NAME" \
  --resume_from_checkpoint "$RESUME_FROM_CHECKPOINT" \
  --eval_only "$EVAL_ONLY" \
  --report_to none
