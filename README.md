<div align="center">
    <h1>Learning from the Self-future: On-policy Self-distillation for dLLMs</h1>
    <p>We introduce <strong>d-OPSD</strong>, the first OPSD framework tailored for dLLMs</p>
</div>


<div align="center">
  <hr width="100%">
</div>

**Updates:**

* 17-06-2026: We released [our paper](https://arxiv.org/abs/2606.18195)
* 15-06-2026: We released d-OPSD code.
<!-- * 04-11-2025: We released [our paper](https://dllm-reasoning.github.io/media/preprint.pdf) and [project page](https://dllm-reasoning.github.io). Additionally, the SFT code was open-sourced. -->

<div align="center">
  <hr width="100%">
</div>


## d-OPSD Environment

The environment configuration of d-OPSD is almost the same as the RLVR baseline [diffu-GRPO](https://github.com/dllm-reasoning/d1). However, there are some minor but important differences.

To set up the environment, first run (pay attention to the **trl version**):
```
cd d-opsd-code
conda env create -f env.yml
conda activate dOPSD
```

**Second, very important**, please go to your environment `/path/to/env/trl/trainer/grpo_trainer.py`, and modify line 424 to the followings:
```
# modify
# possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
possible_values = [n_gen for n_gen in range(1, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
```

Or you can choose to simply replace the original `/path/to/env/trl/trainer/grpo_trainer.py` with what we offered in this repo.

Finally, we give the real environment configuration we used for all experiments in `used-env.txt`, for debugging convenience. This configuration works like an alarm for Python 3.10 and CUDA 12.9, with A100 / H100 / B200 GPUs.


## d-OPSD Training

All training code is inside the `d-opsd` directory. To reproduce the training, run:
```
cd d-opsd-code
bash d-opsd/run/gsm/opsd.sh
bash d-opsd/run/math/opsd.sh
bash d-opsd/run/countdown/opsd.sh
bash d-opsd/run/sudoku/opsd.sh
```

Note: **Very important**, for A100 / H100 GPUs, the `BATCH_DIVIDE` in the script should be set to 8 to prevent OOM. For B200, the existing setting `BATCH_DIVIDE=4` works well.


## d-OPSD Evaluation

All evaluation code is inside the `eval` directory. First replace the checkpoint path in the scripts with your own, and run:
```
cd d-opsd-code
bash eval/run/gsm/opsd.sh
bash eval/run/math/opsd.sh
bash eval/run/countdown/opsd.sh
bash eval/run/sudoku/opsd.sh
```

This evaluation saves the generations. Second, replace the generation directory in `eval/parse_and_get_acc.py` with your owns, and run the following to obtain the accuracy:
```
cd d-opsd-code/eval
python parse_and_get_acc.py
```

## Answer prompt vs. answer clamp

This fork adds information-matched answer-conditioned teachers for LLaDA:

- `answer_prompt`: the verified answer from a correct on-policy rollout is appended to the teacher-only prompt.
- `answer_clamp`: the same answer tokens are fixed at their natural rollout positions in the teacher completion.
- `answer_clamp_future`: answer clamp plus fixed IGPO-style chunks from the same correct rollout.
- `self_future`: the original d-OPSD teacher.

The answer span comes from the same verifier path used for correctness. Ambiguous character-to-token mappings are rejected. Prompt and clamp use the same pass@8 rollout and only states in which the full answer span is still masked; privileged tokens never enter the student input or loss mask.

Create the project-local compatibility environment and run tests without modifying an existing environment:

```bash
bash scripts/setup_isolated_env.sh
PYTHONNOUSERSITE=1 .venv-dopsd-py312/bin/python tests/test_teacher_conditioning.py
PYTHONNOUSERSITE=1 .venv-dopsd-py312/bin/python tests/import_trainer.py
```

Submit matched runs only to an allowed partition:

```bash
bash scripts/submit_prompt_vs_clamp.sh a01 50 42
# or: bash scripts/submit_prompt_vs_clamp.sh h01 50 42
```

`MODEL_PATH`, `DOPSD_PYTHON`, `MAX_STEPS`, and `SEED` can be passed through `sbatch --export`. Checkpoints are written below `outputs/checkpoints`; logs and human-readable conditioning samples are written below `logs` and the corresponding run directory.

The main comparison uses seed 42, `pass_k=8` for finding correct on-policy
training trajectories, and `eval_passk=1` on a fixed 32-example held-out GSM8K
subset. Validation runs before training and every 10 optimizer steps; checkpoints
are saved every 50 steps. An explicit `SwanLabCallback` records the runs under
project `d-opsd-prompt-vs-clamp`. Use `SWANLAB_MODE=cloud` after `swanlab login`,
or use `offline` and later run the printed `swanlab sync ...` command.
