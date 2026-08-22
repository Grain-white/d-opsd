import os

import torch
import wandb
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from trl import TrlParser, ModelConfig
from peft import LoraConfig
import warnings
from swanlab.integration.transformers import SwanLabCallback

# Custom imports
from d_opsd_trainer import dOPSDTrainer
from d_opsd_config import dOPSDConfig
from reward_func import (
    xmlcount_reward_func,
    soft_format_reward_func,
    strict_format_reward_func,
    int_reward_func,
    correctness_reward_func,
    countdown_reward_func,
    correctness_reward_func_math,
    sudoku_reward_func,
    boxed_and_answer_tags_format_reward,
)
from data_utils import (
    get_gsm8k_questions,
    get_countdown_questions,
    get_sudoku_questions,
    get_math_questions,
)
from utils import set_random_seed


def main(opsd_config, model_config):
    # Set seed for reproducibility
    set_random_seed(opsd_config.seed)

    # Load dataset based on configuration
    if opsd_config.dataset == "gsm8k":
        dataset = get_gsm8k_questions(split="train", add_ref=opsd_config.add_ref)
        eval_dataset = get_gsm8k_questions(split="test", add_ref=False)
        reward_functions = [
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func,
        ]
    elif opsd_config.dataset == "countdown":
        dataset = get_countdown_questions("train")
        eval_dataset = None
        reward_functions = [countdown_reward_func]
    elif opsd_config.dataset == "sudoku":
        dataset = get_sudoku_questions()
        eval_dataset = None
        reward_functions = [sudoku_reward_func]
    elif opsd_config.dataset == "math":
        dataset = get_math_questions("train", add_ref=opsd_config.add_ref)
        # Prefer a dedicated test split when available; otherwise hold out from train.
        try:
            eval_dataset = get_math_questions("test", add_ref=False)
        except (ValueError, KeyError, FileNotFoundError, OSError):
            eval_dataset = None
        reward_functions = [
            correctness_reward_func_math,
            boxed_and_answer_tags_format_reward,
        ]
    # Shuffle dataset with fixed seed for reproducibility
    dataset = dataset.shuffle(seed=opsd_config.seed)
    if eval_dataset is not None:
        eval_dataset = eval_dataset.shuffle(seed=opsd_config.seed).select(
            range(min(opsd_config.validation_samples, len(eval_dataset)))
        )

    # Split dataset if needed
    if opsd_config.dataset in ["countdown", "sudoku"]:
        train_set = dataset.select(range(0, len(dataset) - 500))  # Leave last 500 for evaluation
    elif opsd_config.dataset == "math" and eval_dataset is None:
        n_val = min(opsd_config.validation_samples, max(1, len(dataset) // 5))
        eval_dataset = dataset.select(range(len(dataset) - n_val, len(dataset)))
        train_set = dataset.select(range(0, len(dataset) - n_val))
    else:
        train_set = dataset

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 4 bit quantization configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load model and tokenizer
    model = AutoModel.from_pretrained(
        opsd_config.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(opsd_config.model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = False

    # Configure LoRA for parameter-efficient fine-tuning
    peft_config = LoraConfig(
        r=model_config.lora_r,
        lora_alpha=model_config.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        task_type="CAUSAL_LM",
        lora_dropout=model_config.lora_dropout,
    )
    # Initialize and run trainer
    callbacks = []
    swanlab_mode = os.environ.get("SWANLAB_MODE", "cloud")
    if swanlab_mode != "disabled":
        swanlab_callback = SwanLabCallback(
            project=os.environ.get("DOPSD_SWANLAB_PROJECT", "d-opsd-prompt-vs-clamp"),
            workspace=os.environ.get("DOPSD_SWANLAB_WORKSPACE") or None,
            experiment_name=opsd_config.run_name,
            description=(
                f"teacher_conditioning={opsd_config.teacher_conditioning}; "
                f"seed={opsd_config.seed}; pass_k={opsd_config.passk}"
            ),
            log_dir=os.path.join(opsd_config.output_dir, "swanlab"),
            mode=swanlab_mode,
            tags=[opsd_config.teacher_conditioning, opsd_config.dataset, f"seed-{opsd_config.seed}"],
        )
        # SwanLabCallback drops unknown kwargs; inject resume id explicitly so
        # SWANLAB_RUN_ID continues the same cloud experiment instead of creating a new one.
        swanlab_run_id = os.environ.get("SWANLAB_RUN_ID")
        if swanlab_run_id:
            swanlab_callback._init_kwargs["id"] = swanlab_run_id
            swanlab_callback._init_kwargs["resume"] = os.environ.get("SWANLAB_RESUME", "must")
        callbacks.append(swanlab_callback)

    trainer = dOPSDTrainer(
        args=opsd_config,
        model=model,
        peft_config=peft_config,
        reward_funcs=reward_functions,
        train_dataset=train_set,
        eval_dataset=eval_dataset,
        callbacks=callbacks,
    )

    if opsd_config.save_steps % opsd_config.num_iterations != 0:
        warnings.warn(
            f"save_steps ({opsd_config.save_steps}) is not divisible by num_iterations ({opsd_config.num_iterations}). If resuming training from a checkpoint, you might need to manually specify a checkpoint whose step is divisible by {opsd_config.num_iterations}."
        )

    resume_from_checkpoint = opsd_config.resume_from_checkpoint
    if isinstance(resume_from_checkpoint, str):
        lowered = resume_from_checkpoint.strip().lower()
        if lowered in {"", "false", "none", "null"}:
            resume_from_checkpoint = False
        elif lowered in {"true", "1", "yes"}:
            resume_from_checkpoint = True
    if opsd_config.eval_only:
        if resume_from_checkpoint is True:
            raise ValueError("eval_only requires an explicit checkpoint path, not resume_from_checkpoint=true")
        if resume_from_checkpoint:
            trainer._load_from_checkpoint(resume_from_checkpoint)
        trainer.evaluate()
        return
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)


if __name__ == "__main__":
    parser = TrlParser((dOPSDConfig, ModelConfig))
    opsd_config, model_config = parser.parse_args_and_config()
    main(opsd_config=opsd_config, model_config=model_config)
