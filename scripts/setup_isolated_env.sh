#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BASE_PYTHON=${BASE_PYTHON:-/WORK/PUBLIC/alex_work/miniconda3/envs/sdpo-full/bin/python}
ENV_DIR=${ENV_DIR:-$REPO_ROOT/.venv-dopsd-py312}
BASE_LIB=$(cd "$(dirname "$BASE_PYTHON")/../lib" && pwd)
export LD_LIBRARY_PATH="$BASE_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  "$BASE_PYTHON" -m venv --system-site-packages "$ENV_DIR"
fi

PYTHONNOUSERSITE=1 "$ENV_DIR/bin/python" -m pip install --no-deps \
  transformers==4.49.0 \
  tokenizers==0.21.4 \
  accelerate==1.4.0 \
  peft==0.15.1 \
  bitsandbytes==0.48.2 \
  deepspeed==0.16.4 \
  hjson==3.1.0 \
  datasets==3.3.2 \
  dill==0.3.8 \
  multiprocess==0.70.16 \
  fsspec==2024.12.0 \
  trl==0.16.0

TRL_DIR=$(PYTHONNOUSERSITE=1 "$ENV_DIR/bin/python" -c "import pathlib,trl; print(pathlib.Path(trl.__file__).parent)")
cp "$REPO_ROOT/grpo_trainer.py" "$TRL_DIR/trainer/grpo_trainer.py"

PYTHONNOUSERSITE=1 "$ENV_DIR/bin/python" -c \
  "import accelerate,bitsandbytes,datasets,deepspeed,peft,tokenizers,transformers,trl; print('isolated environment imports ok')"
PYTHONNOUSERSITE=1 "$ENV_DIR/bin/python" "$REPO_ROOT/tests/import_trainer.py"
