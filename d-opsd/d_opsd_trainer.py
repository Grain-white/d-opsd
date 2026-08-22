import hashlib
import torch
from trl.trainer.grpo_trainer import GRPOTrainer
from typing import Any, Callable, Optional, Union, Sized
import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TrainerCallback, Trainer
from datasets import Dataset, IterableDataset
import warnings
import random
import time
import json
from pathlib import Path
import torch.nn.functional as F
from trl.trainer.grpo_config import GRPOConfig
from trl.extras.profiling import profiling_decorator, profiling_context
from transformers.utils import is_peft_available
from torch import nn
from trl.import_utils import is_rich_available, is_vllm_available
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.utils import (
    generate_model_card,
    get_comet_experiment_url,
    pad,
    print_prompt_completions_sample,
    selective_log_softmax,
)

from utils import (
    main_print,
    generate,
    get_all_parsed_answer,
    get_all_parsed_answer_with_metadata,
    get_parsed_answer_sudoku,
    get_parsed_answer_countdown,
)
from teacher_conditioning import (
    build_answer_prompt,
    fully_masked_rows,
    map_char_span_to_token_span,
    sample_future_hint_positions,
    select_group_rollout_pair,
    split_bounds,
)


if is_peft_available():
    from peft import PeftConfig, get_peft_model
# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class dOPSDTrainer(GRPOTrainer):
    """
    On-policy Self-distillation (OPSD) Trainer for Diffusion Language Models.

    This class extends from the GRPOTrainer. Very Important: Make Sure You Have Replaced the trl File with Ours.

    Key features:
    - Learn from the self-generated future: retain a part of the teacher's trajectory to provide a learning signal.
    - Efficient per-step divergence supervision for diffusion language models
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[
            Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]
        ] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[
            Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]
        ] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (
            None,
            None,
        ),
        peft_config: Optional["PeftConfig"] = None,
    ):
        # Initialize the parent class
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
        )

        self.log_completions = True
        self.batch_divide = args.batch_divide
        self.debug1 = args.debug1 # cnannot replace with debug, there is already one "debug" attribute existed.
        self.passk = args.passk
        self.eval_passk = args.eval_passk
        self.passk_temperature = args.passk_temperature
        self.teacher_retain_ratio = args.teacher_retain_ratio
        self.teacher_conditioning = args.teacher_conditioning
        self.rollout_filter = args.rollout_filter
        self.fixed_teacher_tokens_remask = args.fixed_teacher_tokens_remask
        self.future_hint_ratio_min = args.future_hint_ratio_min
        self.future_hint_ratio_max = args.future_hint_ratio_max
        self.future_hint_chunk_min = args.future_hint_chunk_min
        self.future_hint_chunk_max = args.future_hint_chunk_max
        self.fixed_teacher = args.fixed_teacher
        self.top_k_loss = args.top_k_loss
        self.jsd_token_clip = args.jsd_token_clip
        self.add_ref = args.add_ref
        self.diff_student_mask = args.diff_student_mask
        self.dataset_name = args.dataset
        self.sudoku_threshold = args.sudoku_threshold
        valid_conditioning = {
            "self_future",
            "answer_prompt",
            "group_answer_prompt",
            "answer_clamp",
            "answer_clamp_future",
        }
        if self.add_ref:
            warnings.warn(
                "add_ref is a legacy full-solution baseline and is not information-matched. "
                "It is kept for compatibility only.",
                stacklevel=2,
            )
            self.teacher_conditioning = "reference_prompt"
        elif self.teacher_conditioning not in valid_conditioning:
            raise ValueError(f"teacher_conditioning must be one of {sorted(valid_conditioning)}")
        if self.rollout_filter not in {"correct_only", "all"}:
            raise ValueError("rollout_filter must be 'correct_only' or 'all'")
        if self.fixed_teacher_tokens_remask and self.teacher_conditioning in {
            "answer_prompt",
            "group_answer_prompt",
            "answer_clamp",
            "answer_clamp_future",
        }:
            raise NotImplementedError(
                "Stochastic remasking is intentionally excluded from the primary experiment. "
                "Set fixed_teacher_tokens_remask=false."
            )
        if not (0 <= self.future_hint_ratio_min <= self.future_hint_ratio_max <= 1):
            raise ValueError("Future hint ratios must satisfy 0 <= min <= max <= 1")
        if not (1 <= self.future_hint_chunk_min <= self.future_hint_chunk_max):
            raise ValueError("Future hint chunk sizes must satisfy 1 <= min <= max")
        if self.teacher_conditioning in {"answer_prompt", "group_answer_prompt", "reference_prompt"}:
            self.teacher_max_prompt_length = args.teacher_max_prompt_length
        if args.max_grad_norm is not None:
            main_print(f'max_grad is {args.max_grad_norm}')
        else:
            main_print(f'no max_grad')
        main_print(f"Batch divide to prevent OOM: {self.batch_divide}")
        main_print(f"Debug mode: {self.debug1}")
        main_print(f"PassK (number of reasoning trajectories): {self.passk}")
        main_print(f"PassK temperature: {self.passk_temperature}")
        main_print(f"Teacher retain ratio: {self.teacher_retain_ratio}")
        main_print(f"Teacher conditioning: {self.teacher_conditioning}")
        main_print(f"Rollout filter: {self.rollout_filter}")
        main_print(f"Fixed teacher: {self.fixed_teacher}")
        main_print(f"Top-k for loss computation: {self.top_k_loss}")
        main_print(f"JSD token clip value: {self.jsd_token_clip}")
        main_print(f"Add reference solutions to prompts: {self.add_ref}")
        main_print(f"Diff student mask: {self.diff_student_mask}")
        main_print(f"Dataset name: {self.dataset_name}")
        main_print(f"Sudoku accuracy threshold: {self.sudoku_threshold}")
        main_print(f'gen_length: {self.args.max_completion_length}, block_length: {self.args.block_length}, diffusion_steps: {self.args.diffusion_steps}')

    def _stable_input_seed(self, inputs, salt: int = 0) -> int:
        """Condition-independent seed for a held-out example."""
        example = inputs[0] if isinstance(inputs, list) and inputs else inputs
        if isinstance(example, dict):
            canonical_keys = ("question", "answer", "puzzle", "solution", "numbers", "target")
            example = {key: example[key] for key in canonical_keys if key in example}
        payload = json.dumps(example, sort_keys=True, ensure_ascii=False, default=str)
        digest = int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")
        return (int(self.args.seed) + digest + int(salt)) % (2**31 - 1)

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        """Route metrics by the actual log payload, including eval-on-start."""
        is_eval_log = any(key.startswith("eval_") for key in logs)
        previous_should_evaluate = self.control.should_evaluate
        self.control.should_evaluate = is_eval_log
        if is_eval_log:
            self._last_eval_metrics = {
                f"eval_{key}": sum(values) / len(values)
                for key, values in self._metrics["eval"].items()
                if values
            }
        try:
            return super().log(logs, start_time)
        finally:
            self.control.should_evaluate = previous_should_evaluate

    def get_logits(self, model, batch, prompt_index, cfg_scale, mask_id):
        input = batch
        logits = model(input).logits
        if cfg_scale > 0.0:
            main_print(f'cfg>0, Wrong')
            raise NotImplementedError("CFG is not implemented for dOPSDTrainer yet")
        return logits

    @staticmethod
    def topk_kendall_tau(student_logits, teacher_logits, top_k=20):
        """Kendall tau of student ranks over the teacher's top-k candidates."""
        k = min(top_k, teacher_logits.shape[-1])
        teacher_indices = torch.topk(teacher_logits, k=k, dim=-1).indices
        student_scores = torch.gather(student_logits, dim=-1, index=teacher_indices)
        upper = torch.triu_indices(k, k, offset=1, device=student_logits.device)
        pair_differences = student_scores[..., upper[0]] - student_scores[..., upper[1]]
        return pair_differences.sign().float().mean()
    
    def generalized_jsd_loss(
        self,
        student_logits,
        teacher_logits,
        beta=0.5,
        reduction="batchmean",
        top_k=None,
        token_clip=None,
    ):
        """
        Compute the generalized Jensen-Shannon Divergence loss for knowledge distillation using F.kl_div. See Eq. (1)
        of https://huggingface.co/papers/2306.13649 for the definition.

        Args:
            student_logits:
                Tensor of shape (batch_size, sequence_length, vocab_size)
            teacher_logits:
                Tensor of shape (batch_size, sequence_length, vocab_size)
            beta:
                Interpolation coefficient between 0 and 1 (default: 0.5)
            reduction:
                Specifies the reduction to apply to the output (default: 'batchmean')
            top_k:
                If set, restricts the loss to only the top-k tokens of the teacher distribution. Both student and
                teacher distributions are renormalized over these k tokens before computing JSD. This reduces memory
                and focuses distillation on the teacher's most probable tokens. (default: None = full vocabulary)
            token_clip:
                if set, clips per-token divergence values to this maximum before reduction. Prevents style tokens from dominating the gradient signal over math tokens.

        Returns:
            loss: Scalar tensor with the generalized JSD loss
        """

        if top_k is not None and top_k > 0:
            # Restrict to top-k tokens of the teacher distribution and renormalize.
            # Also compute the overlap between student top-k and teacher top-k.
            # Shape: [batch, seq_len, top_k]
            _, teacher_top_k_indices = torch.topk(teacher_logits, k=top_k, dim=-1)
            _, student_top_k_indices = torch.topk(student_logits, k=top_k, dim=-1)

            student_top_k_mask = student_top_k_indices.unsqueeze(-1) == teacher_top_k_indices.unsqueeze(-2)
            top_k_overlap = student_top_k_mask.any(dim=-1).float().mean(dim=-1)

            student_logits = torch.gather(student_logits, dim=-1, index=teacher_top_k_indices)
            teacher_logits = torch.gather(teacher_logits, dim=-1, index=teacher_top_k_indices)
        else:
            top_k_overlap = torch.zeros(
                student_logits.shape[:-1], dtype=student_logits.dtype, device=student_logits.device
            )

        # KL reductions are numerically fragile in bf16.  Compute the
        # distributions in fp32 even when the model forward uses bf16.
        student_log_probs = F.log_softmax(student_logits.float(), dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits.float(), dim=-1)

        if beta == 0: # forward
            jsd = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
        elif beta == 1: # reverse
            jsd = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
        else:
            # Compute the log of the mixture distribution
            # log(a + b) = log(exp(log(a)) + exp(log(b))) -> for mixture
            beta = torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device)
            mixture_log_probs = torch.logsumexp(
                torch.stack([student_log_probs + torch.log1p(-beta), teacher_log_probs + torch.log(beta)]),
                dim=0,
            )

            # Compute KL divergences using F.kl_div
            # PyTorch differs from the standard mathematical definition, so the order of the probability distributions is swapped compared to that defined in the paper.
            kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
            kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)
            # Compute the Generalized Jensen-Shannon Divergence
            jsd = beta * kl_teacher + (1 - beta) * kl_student

        # A distillation target is one sequence token, whose divergence is the
        # sum over vocabulary entries.  Individual KL summands may be negative;
        # clipping them before this sum can therefore make the total divergence
        # negative.  Reduce vocabulary first, then apply token-level clipping.
        jsd = jsd.sum(dim=-1).clamp_min(0.0)

        # Per-token clipping: cap each selected sequence token's divergence.
        if token_clip is not None:
            clipped_mask = jsd > token_clip
            clip_ratio = clipped_mask.float().mean()
            jsd = jsd.clamp(max=token_clip)
        else:
            clip_ratio = torch.zeros((), dtype=jsd.dtype, device=jsd.device)

        # Apply reduction
        if reduction == "batchmean":
            loss = jsd.sum() / jsd.size(0)
        elif reduction == "sum":
            loss = jsd.sum()
        elif reduction == "mean":
            loss = jsd.mean()
        else:
            loss = jsd

        return loss, clip_ratio, top_k_overlap
    
    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        prompt_length = inputs["prompt_length"]
        teacher_prompt_length = inputs["teacher_prompt_length"]
        block_length = inputs["block_length"]
        is_correct = inputs["is_correct"]
        pair_count = inputs["student_inputs"].shape[0]
        microstep = (self._step - 1) % self.batch_divide
        start_pos, end_pos = split_bounds(pair_count, self.batch_divide, microstep)
        if start_pos == end_pos:
            raise ValueError(
                f"Only {pair_count} eligible state pairs for batch_divide={self.batch_divide}. "
                "Reduce batch_divide or increase diffusion states."
            )
        student_input = inputs["student_inputs"][start_pos:end_pos].to(model.device)
        student_output = inputs["student_outputs"][start_pos:end_pos].to(model.device)
        teacher_input = inputs["teacher_inputs"][start_pos:end_pos].to(model.device)
        privileged_mask = inputs["privileged_mask"][start_pos:end_pos].to(model.device)

        # teacher forward
        teacher_started = time.perf_counter()
        if self.fixed_teacher and is_peft_model(model):
            with torch.no_grad(), self.accelerator.unwrap_model(model).disable_adapter():
                teacher_logits = self.get_logits(model, teacher_input, None, self.args.cfg_scale, self.args.mask_id)
        else:
            with torch.no_grad():
                teacher_logits = self.get_logits(model, teacher_input, None, self.args.cfg_scale, self.args.mask_id)
        teacher_logits = teacher_logits.detach()
        teacher_seconds = time.perf_counter() - teacher_started

        prompt_delta = teacher_prompt_length - prompt_length
        selected_rows = []
        teacher_positions = []
        student_positions = []
        if self.diff_student_mask:
            if prompt_delta != 0:
                raise ValueError("diff_student_mask is not supported with a length-changing teacher prompt")
            changed = student_input != student_output
            for row in range(changed.shape[0]):
                positions = torch.where(changed[row] & ~privileged_mask[row])[0][:2]
                for position in positions.tolist():
                    selected_rows.append(row)
                    teacher_positions.append(position)
                    student_positions.append(position)
        else:
            mask_id = self.args.mask_id
            seq_length = teacher_input.size(1)
            teacher_confidence = teacher_logits.max(dim=-1).values
            for i in range(teacher_input.size(0)):
                masked_positions = torch.where((teacher_input[i] == mask_id) & ~privileged_mask[i])[0]
                if masked_positions.numel() == 0:
                    continue
                first_mask_pos = masked_positions[0]
                relative_pos = first_mask_pos - teacher_prompt_length
                if relative_pos < 0:
                    raise ValueError(
                        f"First mask position ({first_mask_pos.item()}) is before teacher prompt_length ({teacher_prompt_length}) at row {i}."
                    )

                block_idx = relative_pos // block_length
                block_start = teacher_prompt_length + block_idx * block_length
                block_end = min(block_start + block_length, seq_length)
                block_positions = torch.arange(block_start, block_end, device=teacher_input.device)
                block_masked_positions = block_positions[
                    (teacher_input[i, block_positions] == mask_id)
                    & ~privileged_mask[i, block_positions]
                ]
                if block_masked_positions.numel() == 0:
                    continue
                block_confidence = teacher_confidence[i, block_masked_positions]
                count = min(2, block_masked_positions.numel())
                chosen = block_masked_positions[torch.topk(block_confidence, k=count).indices]
                for teacher_position in chosen.tolist():
                    student_position = teacher_position - prompt_delta
                    if not prompt_length <= student_position < student_input.shape[1]:
                        raise AssertionError("Teacher/student target alignment escaped the completion")
                    selected_rows.append(i)
                    teacher_positions.append(teacher_position)
                    student_positions.append(student_position)
        if not selected_rows:
            raise ValueError("No non-privileged masked targets remain in this microbatch")
        row_index = torch.tensor(selected_rows, device=model.device, dtype=torch.long)
        teacher_index = torch.tensor(teacher_positions, device=model.device, dtype=torch.long)
        student_index = torch.tensor(student_positions, device=model.device, dtype=torch.long)
        assert not privileged_mask[row_index, teacher_index].any(), "Privileged tokens entered the loss"
        if self.debug1:
            main_print(f'step is:{self._step}')
            main_print(f'global step is: {self.state.global_step}')
            main_print(f'start_pos: {start_pos}, end_pos: {end_pos}')
            main_print(f'trajectory shape: {student_input.shape}')
            main_print(f'selected target count: {len(selected_rows)}')

        # student forward
        student_started = time.perf_counter()
        student_logits = self.get_logits(model, student_input, None, self.args.cfg_scale, self.args.mask_id)
        student_seconds = time.perf_counter() - student_started
        if self.debug1:
            main_print(f'Before logits cutting')
            main_print(f'student_logits shape: {student_logits.shape}')
            main_print(f'teacher_logits shape: {teacher_logits.shape}')
        student_logits = student_logits[row_index, student_index].unsqueeze(1)
        teacher_logits = teacher_logits[row_index, teacher_index].unsqueeze(1)
        kendall_tau = self.topk_kendall_tau(student_logits, teacher_logits, self.top_k_loss or 20)
        completion_positions = student_index - prompt_length
        final_completion_ids = inputs["final_completion_ids"].to(model.device)
        target_ids = final_completion_ids[completion_positions]
        student_target_prob = torch.gather(
            F.softmax(student_logits[:, 0].float(), dim=-1), 1, target_ids[:, None]
        ).mean()
        teacher_target_prob = torch.gather(
            F.softmax(teacher_logits[:, 0].float(), dim=-1), 1, target_ids[:, None]
        ).mean()
        if self.debug1:
            main_print(f'After logits cutting')
            main_print(f'student_logits shape: {student_logits.shape}')
            main_print(f'teacher_logits shape: {teacher_logits.shape}')

        loss, clip_ratio, top_k_overlap = self.generalized_jsd_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                beta=self.beta,
                top_k=self.top_k_loss,
                token_clip=self.jsd_token_clip,
            )   
        if self.debug1:
            main_print(f'After generalized_jsd_loss: loss={loss.item():.6f}, clip_ratio={clip_ratio.item():.6f}, top_k_overlap_shape={tuple(top_k_overlap.shape)}')
        assert top_k_overlap.shape == (student_logits.size(0), 1)
        
        mode = "train" if self.model.training else "eval"
        self._metrics[mode]["loss"].append(self.accelerator.gather_for_metrics(loss).mean().item())
        self._metrics[mode]["clip_ratio"].append(
            self.accelerator.gather_for_metrics(clip_ratio).mean().item()
        )
        self._metrics[mode]["teacher_forward_seconds"].append(teacher_seconds)
        self._metrics[mode]["student_forward_seconds"].append(student_seconds)
        self._metrics[mode]["distillation_targets"].append(float(len(selected_rows)))
        self._metrics[mode]["kendall_tau_topk"].append(
            self.accelerator.gather_for_metrics(kendall_tau).mean().item()
        )
        self._metrics[mode]["student_correct_token_probability"].append(
            self.accelerator.gather_for_metrics(student_target_prob).mean().item()
        )
        self._metrics[mode]["teacher_correct_token_probability"].append(
            self.accelerator.gather_for_metrics(teacher_target_prob).mean().item()
        )
        # top_k_overlap has shape [local_batch, local_seq_len], which may vary across ranks.
        # Reduce locally to a scalar first to avoid distributed gather shape mismatch / deadlock.
        top_k_overlap_local = top_k_overlap.mean()
        top_k_overlap_value = self.accelerator.gather_for_metrics(top_k_overlap_local).mean().item()
        self._metrics[mode]["top_k_overlap"].append(top_k_overlap_value)
        if self.debug1:
            main_print(f'After metrics gather: top_k_overlap_value={top_k_overlap_value:.6f}')

        del student_logits, teacher_logits, student_input, student_output, teacher_input, privileged_mask
        torch.cuda.empty_cache()

        if self.dataset_name == "sudoku":
            accepted = is_correct >= self.sudoku_threshold
        else:
            accepted = bool(is_correct)
        if self.teacher_conditioning in {
            "answer_prompt",
            "group_answer_prompt",
            "answer_clamp",
            "answer_clamp_future",
        } and not inputs.get("conditioning_available", False):
            return loss * 0.0
        if self.rollout_filter == "all" or accepted or self.teacher_conditioning == "reference_prompt":
            return loss
        return loss * 0.0

    def _prepare_inputs(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        mode = "train" if self.model.training else "eval"

        if mode == "train":
            # Very important, due to the RepeatSampler from GPPOTrainer.
            if self._step % self.batch_divide == 0:
                inputs = self._generate_and_score_completions(inputs)
                self._buffered_inputs[0] = inputs
            else:
                inputs = self._buffered_inputs[0]
            self._step += 1
        else:
            # Each held-out example gets the same rollout RNG under every
            # conditioning method. Restore the training RNG after evaluation.
            cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
            with torch.random.fork_rng(devices=cuda_devices, enabled=True):
                eval_seed = self._stable_input_seed(inputs, salt=17)
                torch.manual_seed(eval_seed)
                inputs = self._generate_and_score_completions(inputs)
        return inputs

    def _generate_and_score_completions_legacy(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        active_passk = self.eval_passk if mode == "eval" else self.passk

        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs
        ]
        if self.debug1:
            main_print(f'prompts_text is: {prompts_text}')
        prompt_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
        prompt_ids = prompt_inputs["input_ids"]
        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
        if self.debug1:
            main_print(f'prompt_ids shape is: {prompt_ids.shape}')  

        # Configuration for the diffusion generation
        gen_length = self.args.max_completion_length
        block_length = self.args.block_length
        steps = self.args.diffusion_steps
        temperature = self.args.temperature or 0.0
        cfg_scale = self.args.cfg_scale

        generation_started = time.perf_counter()
        with unwrap_model_for_generation(self.model_wrapped, self.accelerator) as unwrapped_model:
            generation_batch_size = 1 # we fix it here. It almost won't slow the training..
            with torch.no_grad():
                for i in range(0, prompt_ids.size(0), generation_batch_size):
                    end_idx = min(i + generation_batch_size, prompt_ids.size(0))
                    batch_prompt_ids = prompt_ids[i:end_idx] # [1, prompt_length]
                    # WARNING: Attention masks are not currently used during generation.
                    # This works fine here as long as the generation batch only consists of same prompts (our case a single prompt).
                    
                    batch_prompt_completion_ids, batch_trajectory = generate(
                        model=unwrapped_model,
                        prompt=batch_prompt_ids,
                        steps=steps,
                        gen_length=gen_length,
                        block_length=block_length,
                        temperature=temperature,
                        cfg_scale=cfg_scale,
                        remasking=self.args.remasking,
                        mask_id=self.args.mask_id,
                        debug1=self.debug1,
                        fp16=self.args.fp16
                    )
                    completions_text = self.processing_class.batch_decode(batch_prompt_completion_ids[:, -gen_length:], skip_special_tokens=False)
                    # if self.debug1:
                    #     main_print(f'prompt type: {type(inputs[0]["prompt"])}') # list
                    if self.dataset_name == "sudoku":
                        parsed_answer, accuracy = get_parsed_answer_sudoku(completions_text[0], inputs[0]["solution"], inputs[0]["puzzle"])
                        best_accuracy = accuracy
                        best_parsed_answer = parsed_answer
                        best_completion_text = completions_text
                        best_batch_prompt_completion_ids = batch_prompt_completion_ids
                        best_batch_trajectory = batch_trajectory
                        is_correct = False
                    elif self.dataset_name == "countdown":
                        parsed_answer, is_correct = get_parsed_answer_countdown(completions_text[0], inputs[0]["numbers"], inputs[0]["target"])
                    else:
                        parsed_answer, is_correct = get_all_parsed_answer(completions_text[0], inputs[0]["answer"], self.dataset_name)
                    if self.dataset_name == "sudoku":
                        verifier_pass_at_1 = accuracy >= self.sudoku_threshold
                    else:
                        verifier_pass_at_1 = bool(is_correct)
                    if self.debug1:
                        main_print(f'input is:{inputs}')  
                        '''
                        [
                            {
                                "question": "",
                                "answer": "",
                                "prompt": [ {"role": "user", "content": "prompt+question"} ]
                            }
                        ]
                        '''
                        main_print(f'completions_text is: {completions_text[0]}')
                        main_print(f'parsed_answer is: {parsed_answer}')
                        if self.dataset_name == "sudoku":
                            gt_answer = inputs[0]["solution"]
                        elif self.dataset_name == "countdown":
                            gt_answer = inputs[0]["target"]
                        else:
                            gt_answer = inputs[0]["answer"]
                        main_print(f'ground truth answer is: {gt_answer}')
                        main_print(f'is_correct is: {is_correct if self.dataset_name != "sudoku" else accuracy}')
                    
                    # Refer to the "pass@k" in the paper: extend to more reasoning trajectories if needed.
                    iter_num = 1
                    while (not self.add_ref) and iter_num < active_passk and (not is_correct or self.dataset_name == "sudoku"):
                        iter_num = iter_num + 1
                        batch_prompt_completion_ids, batch_trajectory = generate(
                            model=unwrapped_model,
                            prompt=batch_prompt_ids,
                            steps=steps,
                            gen_length=gen_length,
                            block_length=block_length,
                            temperature=self.passk_temperature,
                            cfg_scale=cfg_scale,
                            remasking=self.args.remasking,
                            mask_id=self.args.mask_id,
                            debug1=self.debug1,
                            fp16=self.args.fp16
                        )
                        completions_text = self.processing_class.batch_decode(batch_prompt_completion_ids[:, -gen_length:], skip_special_tokens=False)
                        if self.dataset_name == "sudoku":
                            parsed_answer, accuracy = get_parsed_answer_sudoku(completions_text[0], inputs[0]["solution"], inputs[0]["puzzle"])
                            if accuracy >= best_accuracy:
                                best_accuracy = accuracy
                                best_parsed_answer = parsed_answer
                                best_completion_text = completions_text
                                best_batch_prompt_completion_ids = batch_prompt_completion_ids
                                best_batch_trajectory = batch_trajectory
                        elif self.dataset_name == "countdown":
                            parsed_answer, is_correct = get_parsed_answer_countdown(completions_text[0], inputs[0]["numbers"], inputs[0]["target"])
                        else:
                            parsed_answer, is_correct = get_all_parsed_answer(completions_text[0], inputs[0]["answer"], self.dataset_name)
                        if self.debug1:
                            main_print(f'now at iteration: {iter_num}')
                            main_print(f'completions_text is: {completions_text[0]}')
                            main_print(f'parsed_answer is: {parsed_answer}')
                            if self.dataset_name == "sudoku":
                                gt_answer = inputs[0]["solution"]
                            elif self.dataset_name == "countdown":
                                gt_answer = inputs[0]["target"]
                            else:
                                gt_answer = inputs[0]["answer"]
                            main_print(f'ground truth answer is: {gt_answer}')
                            main_print(f'is_correct is: {is_correct if self.dataset_name != "sudoku" else best_accuracy}')    
                    '''
                    batch_prompt_completion_ids: [1, prompt_length + gen_length]
                    batch_trajectory: [x0, x1, ..., x_steps_till_eos], each of shape [1, prompt_length + gen_length]
                    '''
            # The correct here is for pass@k. If correct, we only keep the first succesful trajectory; Otherwise we keep the last sampled trajectory.
            local_is_correct = torch.tensor(iter_num, device=device, dtype=torch.float32)
        generation_seconds = time.perf_counter() - generation_started

        if self.dataset_name == "sudoku":
            accuracy = best_accuracy
            parsed_answer = best_parsed_answer
            completions_text = best_completion_text
            batch_prompt_completion_ids = best_batch_prompt_completion_ids
            batch_trajectory = best_batch_trajectory
            accuracy_tensor = torch.tensor(accuracy, device=device, dtype=torch.float32)
        prompt_length = prompt_ids.size(1)
        completion_part_ids = batch_prompt_completion_ids[:, prompt_length:]
        eos_id = 126081 # we fix it here for LLADA. Please adjust it to yours.
        eos_positions = torch.where(completion_part_ids[0] == eos_id)[0]
        pure_gen_length_val = eos_positions[0].item() if eos_positions.numel() > 0 else completion_part_ids.size(1)
        pure_gen_length = torch.tensor(pure_gen_length_val, device=device, dtype=torch.float32)
        trajectory = torch.cat(batch_trajectory, dim=0) # [steps_till_eos, length], student

        steps_till_eos, full_seq_length = trajectory.shape
        if steps_till_eos <= self.batch_divide:
            print(f'completions_text is: {completions_text[0]}')
            raise ValueError(f"Steps till EOS {steps_till_eos} is not greater than batch_divide {self.batch_divide}, which may cause issues with batch splitting. Consider reducing batch_divide or checking the generation process for early EOS.")

        # construct the teacher
        teacher_trajectory = trajectory.clone()
        if self.add_ref: # AR-style counterpart that adds reference solutions to the prompt
            teacher_prompts_text = [
                maybe_apply_chat_template(example, self.processing_class)["teacher_prompt"] for example in inputs
            ]
            if self.debug1:
                main_print(f'teacher_prompts_text is: {teacher_prompts_text}')
            teacher_prompt_inputs = self.processing_class(
                text=teacher_prompts_text,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
            )
            teacher_prompt_inputs = Trainer._prepare_inputs(self, teacher_prompt_inputs)
            teacher_prompt_ids = teacher_prompt_inputs["input_ids"]
            if self.teacher_max_prompt_length is not None:
                teacher_prompt_ids = teacher_prompt_ids[:, -self.teacher_max_prompt_length:]

            teacher_prompt_length = teacher_prompt_ids.size(1)
            teacher_prompt_ids = teacher_prompt_ids.to(teacher_trajectory.device)
            teacher_prompt_ids = teacher_prompt_ids[0:1]
            teacher_completion_part = teacher_trajectory[:, prompt_length:]
            teacher_prompt_part = teacher_prompt_ids.expand(steps_till_eos, -1)
            teacher_trajectory = torch.cat([teacher_prompt_part, teacher_completion_part], dim=1)
        else:
            teacher_prompt_length = prompt_length
            final_sequence = batch_prompt_completion_ids[0]  # [seq_length]
            
            for step_idx in range(steps_till_eos - 1):
                n = step_idx // (block_length // (gen_length // steps))
                start_pos = prompt_length + (n + 1) * block_length
                if start_pos > full_seq_length - block_length:
                    raise ValueError(f"start_pos {start_pos} for step {step_idx} is out of bounds for full_seq_length {full_seq_length}")

                # Skip if start_pos is already at or beyond the EOS position
                # print(f'step_idx: {step_idx}, n: {n}, start_pos: {start_pos}, pure_gen_length_val: {pure_gen_length_val}, prompt_length: {prompt_length}')
                if start_pos >= pure_gen_length_val + prompt_length:
                    continue

                # candidate_positions = torch.arange(start_pos, pure_gen_length_val + prompt_length, device=teacher_trajectory.device)
                candidate_positions = torch.arange(start_pos, full_seq_length, device=teacher_trajectory.device)
                num_candidates = candidate_positions.numel()
                num_replace = int(num_candidates * self.teacher_retain_ratio)
                if num_replace <= 0:
                    continue
                # Teacher-only hint sampling must not perturb the global RNG used by
                # subsequent student rollouts.  A stable per-example/per-step seed also
                # makes the conditioning comparison reproducible across runs.
                teacher_generator = torch.Generator(device=teacher_trajectory.device)
                teacher_generator.manual_seed(
                    self._stable_input_seed(
                        inputs,
                        salt=int(self.state.global_step) * 1_000_003 + step_idx + 29,
                    )
                )
                selected_relative = torch.randperm(
                    num_candidates,
                    device=teacher_trajectory.device,
                    generator=teacher_generator,
                )[:num_replace]
                selected_positions = candidate_positions[selected_relative]
                teacher_trajectory[step_idx, selected_positions] = final_sequence[selected_positions]
        
        if self.debug1:
            main_print(f'trajectory shape: {trajectory.shape}')
            main_print(f'teacher_trajectory shape: {teacher_trajectory.shape}')
            main_print(f"Prompt length: {prompt_length}")
            completions_text_10step = self.processing_class.batch_decode(trajectory[10:11, -gen_length:], skip_special_tokens=False)
            teacher_completions_text_10step = self.processing_class.batch_decode(teacher_trajectory[10:11, -gen_length:], skip_special_tokens=False)
            main_print(f'trajectory[10]: {completions_text_10step[0]}')
            main_print(f'teacher_trajectory[10]: {teacher_completions_text_10step[0]}')
        

        # Log the metrics
        mode = "train" if self.model.training else "eval"
        completion_length = self.accelerator.gather_for_metrics(pure_gen_length).float().mean().item()
        self._metrics[mode]["completion_length"].append(completion_length)
        mean_is_correct = self.accelerator.gather_for_metrics(local_is_correct).mean().item()
        self._metrics[mode]["iter_num"].append(mean_is_correct)
        if self.dataset_name != "sudoku":
            verifier_metrics = {
                "rollout_attempts": float(iter_num),
                "passk_success": float(bool(is_correct)),
                # Back-compat with answer-conditioned validation metrics.
                "verifier_accuracy": float(bool(is_correct)),
                "verifier_pass_at_1": float(verifier_pass_at_1),
                "conditioning_available": 1.0,
                "eligible_state_ratio": 1.0,
                "generation_seconds": generation_seconds,
                "rollout_tokens_per_second": float(
                    gen_length * iter_num / max(generation_seconds, 1e-6)
                ),
            }
            if active_passk >= 8:
                verifier_metrics["verifier_pass_at_8"] = float(bool(is_correct))
            elif active_passk != 1:
                verifier_metrics[f"verifier_pass_at_{active_passk}"] = float(bool(is_correct))
            for name, value in verifier_metrics.items():
                gathered = self.accelerator.gather_for_metrics(
                    torch.tensor(value, device=device, dtype=torch.float32)
                ).mean().item()
                self._metrics[mode][name].append(gathered)
        if self.dataset_name == "sudoku":
            if accuracy >= self.sudoku_threshold:
                effective_num = 1
            else:
                effective_num = 0
            accuracy_value = self.accelerator.gather_for_metrics(accuracy_tensor).mean().item()
            effective_num_gathered = self.accelerator.gather_for_metrics(
                torch.tensor(effective_num, device=device, dtype=torch.float32)
            ).mean().item()
            self._metrics[mode]["accuracy"].append(accuracy_value)
            self._metrics[mode]["effective_num"].append(effective_num_gathered)
            is_correct = accuracy

        if self.log_completions and self.state.global_step % self.args.completion_logging_steps == 0:
            prompts_to_log = gather_object(prompts_text)
            completions_to_log = gather_object(completions_text)
            if self.dataset_name == "sudoku":
                rewards_to_log = [accuracy]
            elif is_correct:
                rewards_to_log = [1.0]
            else:
                rewards_to_log = [0.0]
            rewards_to_log = gather_object(rewards_to_log)
            if self.add_ref:
                teacher_prompts_to_log = gather_object(teacher_prompts_text)
                
            if self.accelerator.is_main_process:
                    print_prompt_completions_sample(
                        prompts_to_log,
                        completions_to_log,
                        rewards_to_log,
                        self._step,
                    )

        return {
            "prompt_length": prompt_length,
            "teacher_prompt_length": teacher_prompt_length,
            "trajectory": trajectory.cpu(),
            "teacher_trajectory": teacher_trajectory.cpu(),
            "steps": steps, 
            "gen_length": gen_length,
            "block_length": block_length,
            "is_correct": is_correct,
        }

    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        """Generate one shared rollout and construct the configured teacher."""
        if self.teacher_conditioning in {"self_future", "reference_prompt"}:
            legacy = self._generate_and_score_completions_legacy(inputs)
            trajectory = legacy.pop("trajectory")
            teacher_trajectory = legacy.pop("teacher_trajectory")
            legacy.update(
                student_inputs=trajectory[:-1],
                student_outputs=trajectory[1:],
                teacher_inputs=teacher_trajectory[:-1],
                privileged_mask=torch.zeros_like(teacher_trajectory[:-1], dtype=torch.bool),
                conditioning_available=True,
                eligible_state_ratio=1.0,
                answer_token_span=None,
                span_status="not_required",
                final_completion_ids=trajectory[-1, legacy["prompt_length"]:].clone(),
            )
            return legacy

        if len(inputs) != 1:
            raise ValueError("Answer-conditioned rollout currently requires per-device batch size 1")
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        generation_started = time.perf_counter()
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompt_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
        prompt_ids = prompt_inputs["input_ids"]
        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length:]
        prompt_length = prompt_ids.size(1)
        gen_length = self.args.max_completion_length
        block_length = self.args.block_length
        steps = self.args.diffusion_steps

        def run_once(unwrapped_model, temperature):
            return generate(
                model=unwrapped_model,
                prompt=prompt_ids,
                steps=steps,
                gen_length=gen_length,
                block_length=block_length,
                temperature=temperature,
                cfg_scale=self.args.cfg_scale,
                remasking=self.args.remasking,
                mask_id=self.args.mask_id,
                debug1=self.debug1,
                fp16=self.args.fp16,
            )

        verification = None
        answer_token_span = None
        span_status = "not_checked"
        verifier_correct = False  # pass@k: True if any attempt is verifier-correct
        verifier_pass_at_1 = False  # first-attempt (usually T=0) verifier correctness
        group_candidates = []
        group_pair_available = False
        group_positive_fallback = False
        group_donor_attempt = 0
        group_recipient_attempt = 0
        recipient_is_correct = False
        with unwrap_model_for_generation(self.model_wrapped, self.accelerator) as unwrapped_model:
            with torch.no_grad():
                active_passk = self.eval_passk if mode == "eval" else self.passk
                for attempt in range(1, active_passk + 1):
                    temperature = (self.args.temperature or 0.0) if attempt == 1 else self.passk_temperature
                    batch_prompt_completion_ids, batch_trajectory = run_once(unwrapped_model, temperature)
                    completion_ids = batch_prompt_completion_ids[0, prompt_length:].tolist()
                    completion_text = self.processing_class.decode(completion_ids, skip_special_tokens=False)
                    verification = get_all_parsed_answer_with_metadata(
                        completion_text, inputs[0]["answer"], self.dataset_name
                    )
                    if verification.is_correct:
                        verifier_correct = True
                        if attempt == 1:
                            verifier_pass_at_1 = True
                    candidate_token_span = None
                    candidate_span_status = "verifier_span_missing"
                    if verification.char_span is not None:
                        candidate_token_span, candidate_span_status = map_char_span_to_token_span(
                            self.processing_class,
                            completion_ids,
                            completion_text,
                            verification.char_span,
                            verification.answer_text,
                        )
                    if self.teacher_conditioning == "group_answer_prompt":
                        candidate = {
                            "attempt": attempt,
                            "is_correct": bool(verification.is_correct),
                            "answer_text": verification.answer_text,
                            "verification": verification,
                            "token_span": candidate_token_span,
                            "span_status": candidate_span_status,
                            "prompt_completion_ids": batch_prompt_completion_ids,
                            "trajectory": batch_trajectory,
                            "completion_text": completion_text,
                        }
                        group_candidates.append(candidate)
                        # Preserve exact compute parity with answer_prompt.  A
                        # correct first rollout follows the original positive-
                        # trajectory path.  Only when the first rollout is wrong
                        # do later attempts donate their correct answer to it.
                        if attempt == 1 and verification.is_correct and candidate_token_span is not None:
                            group_positive_fallback = True
                            break
                        donor_index, recipient_index = select_group_rollout_pair(group_candidates)
                        if donor_index is not None and recipient_index is not None:
                            group_pair_available = True
                            break
                    else:
                        answer_token_span = candidate_token_span
                        span_status = candidate_span_status
                        if verification.is_correct and answer_token_span is not None:
                            break

        if self.teacher_conditioning == "group_answer_prompt" and (
            group_pair_available or group_positive_fallback
        ):
            if group_positive_fallback:
                donor = recipient = group_candidates[0]
            else:
                donor_index, recipient_index = select_group_rollout_pair(group_candidates)
                donor = group_candidates[donor_index]
                recipient = group_candidates[recipient_index]
            verification = donor["verification"]
            answer_token_span = recipient["token_span"]
            span_status = recipient["span_status"]
            batch_prompt_completion_ids = recipient["prompt_completion_ids"]
            batch_trajectory = recipient["trajectory"]
            completion_text = recipient["completion_text"]
            group_donor_attempt = donor["attempt"]
            group_recipient_attempt = recipient["attempt"]
            recipient_is_correct = bool(recipient["is_correct"])

        if self.teacher_conditioning == "group_answer_prompt":
            conditioning_available = bool(
                (group_pair_available or group_positive_fallback)
                and answer_token_span is not None
            )
        else:
            conditioning_available = bool(verifier_correct and answer_token_span is not None)
        is_correct = conditioning_available
        trajectory = torch.cat(batch_trajectory, dim=0)
        final_sequence = batch_prompt_completion_ids[0]
        student_inputs = trajectory[:-1]
        student_outputs = trajectory[1:]
        pair_count = student_inputs.shape[0]
        teacher_prompt_length = prompt_length
        teacher_inputs = student_inputs.clone()
        privileged_mask = torch.zeros_like(teacher_inputs, dtype=torch.bool)
        eligible_rows = torch.arange(pair_count, device=student_inputs.device)
        answer_absolute_span = None
        teacher_prompts_text = None

        if conditioning_available:
            answer_start, answer_end = answer_token_span
            answer_absolute_span = (prompt_length + answer_start, prompt_length + answer_end)
            eligible = fully_masked_rows(student_inputs, answer_absolute_span, self.args.mask_id)
            eligible_rows = torch.where(eligible)[0]
            if self.teacher_conditioning in {"answer_prompt", "group_answer_prompt"}:
                teacher_example = dict(inputs[0])
                teacher_example["prompt"] = build_answer_prompt(
                    inputs[0]["prompt"], verification.answer_text
                )
                teacher_prompts_text = [
                    maybe_apply_chat_template(teacher_example, self.processing_class)["prompt"]
                ]
                teacher_prompt_inputs = self.processing_class(
                    text=teacher_prompts_text,
                    return_tensors="pt",
                    padding=True,
                    padding_side="left",
                    add_special_tokens=False,
                )
                teacher_prompt_inputs = Trainer._prepare_inputs(self, teacher_prompt_inputs)
                teacher_prompt_ids = teacher_prompt_inputs["input_ids"]
                if self.teacher_max_prompt_length is not None:
                    teacher_prompt_ids = teacher_prompt_ids[:, -self.teacher_max_prompt_length:]
                teacher_prompt_length = teacher_prompt_ids.size(1)
                teacher_completion = student_inputs[:, prompt_length:]
                teacher_inputs = torch.cat(
                    [teacher_prompt_ids.expand(pair_count, -1), teacher_completion], dim=1
                )
                privileged_mask = torch.zeros_like(teacher_inputs, dtype=torch.bool)
                assert torch.all(
                    teacher_completion[eligible_rows, answer_start:answer_end] == self.args.mask_id
                ), "Answer prompt accidentally revealed completion answer tokens"
            else:
                absolute_start, absolute_end = answer_absolute_span
                answer_tokens = final_sequence[absolute_start:absolute_end]
                teacher_inputs[:, absolute_start:absolute_end] = answer_tokens
                privileged_mask[:, absolute_start:absolute_end] = True
                assert torch.equal(
                    teacher_inputs[:, absolute_start:absolute_end],
                    answer_tokens.expand(pair_count, -1),
                ), "Clamp answer differs from verified rollout"

                if self.teacher_conditioning == "answer_clamp_future":
                    pure_completion = final_sequence[prompt_length:]
                    eos_id = self.processing_class.eos_token_id
                    eos_positions = torch.where(pure_completion == eos_id)[0] if eos_id is not None else []
                    pure_end = prompt_length + (
                        int(eos_positions[0]) if len(eos_positions) else pure_completion.numel()
                    )
                    reasoning_positions = [
                        position
                        for position in range(prompt_length, pure_end)
                        if not absolute_start <= position < absolute_end
                    ]
                    rng = random.Random(self.args.seed + int(self.state.global_step))
                    hint_positions = sample_future_hint_positions(
                        reasoning_positions,
                        self.future_hint_ratio_min,
                        self.future_hint_ratio_max,
                        self.future_hint_chunk_min,
                        self.future_hint_chunk_max,
                        rng,
                    )
                    for row in range(pair_count):
                        current_targets = student_inputs[row] != student_outputs[row]
                        for position in hint_positions:
                            if student_inputs[row, position] == self.args.mask_id and not current_targets[position]:
                                teacher_inputs[row, position] = final_sequence[position]
                                privileged_mask[row, position] = True

        eligible_state_ratio = float(eligible_rows.numel() / max(pair_count, 1))
        if conditioning_available and eligible_rows.numel() < self.batch_divide:
            conditioning_available = False
            is_correct = False
            span_status = "insufficient_eligible_states"
            eligible_rows = torch.arange(pair_count, device=student_inputs.device)
            teacher_inputs = student_inputs.clone()
            privileged_mask = torch.zeros_like(teacher_inputs, dtype=torch.bool)
            teacher_prompt_length = prompt_length

        student_inputs = student_inputs[eligible_rows]
        student_outputs = student_outputs[eligible_rows]
        teacher_inputs = teacher_inputs[eligible_rows]
        privileged_mask = privileged_mask[eligible_rows]
        if self.teacher_conditioning.startswith("answer_clamp") and conditioning_available:
            assert torch.all(
                student_inputs[:, answer_absolute_span[0]:answer_absolute_span[1]] == self.args.mask_id
            ), "Student received answer clamp tokens"

        completion_ids_tensor = batch_prompt_completion_ids[0, prompt_length:]
        eos_id = self.processing_class.eos_token_id
        eos_positions = torch.where(completion_ids_tensor == eos_id)[0] if eos_id is not None else []
        completion_length = int(eos_positions[0]) if len(eos_positions) else completion_ids_tensor.numel()
        generation_seconds = time.perf_counter() - generation_started
        metrics = {
            "completion_length": float(completion_length),
            "iter_num": float(attempt),
            "rollout_attempts": float(attempt),
            "passk_success": float(verifier_correct),
            # Back-compat: verifier_accuracy == pass@{active_passk} (eval uses eval_passk).
            "verifier_accuracy": float(verifier_correct),
            "verifier_pass_at_1": float(verifier_pass_at_1),
            "conditioning_available": float(conditioning_available),
            "span_locatable": float(answer_token_span is not None),
            "eligible_state_ratio": eligible_state_ratio,
            "generation_seconds": generation_seconds,
            "rollout_tokens_per_second": float(gen_length * attempt / max(generation_seconds, 1e-6)),
            "answer_source_boxed": float(verification is not None and verification.source == "boxed"),
            "answer_source_answer_tag": float(
                verification is not None and verification.source == "answer_tag"
            ),
        }
        if self.teacher_conditioning == "group_answer_prompt":
            metrics.update({
                "group_pair_available": float(group_pair_available),
                "group_positive_fallback": float(group_positive_fallback),
                "group_donor_attempt": float(group_donor_attempt),
                "group_recipient_attempt": float(group_recipient_attempt),
                "group_recipient_correct": float(recipient_is_correct),
            })
        if active_passk >= 8:
            metrics["verifier_pass_at_8"] = float(verifier_correct)
        elif active_passk != 1:
            metrics[f"verifier_pass_at_{active_passk}"] = float(verifier_correct)
        for name, value in metrics.items():
            gathered = self.accelerator.gather_for_metrics(
                torch.tensor(value, device=device, dtype=torch.float32)
            ).mean().item()
            self._metrics[mode][name].append(gathered)
        completions_text = [completion_text]
        if self.log_completions and self.state.global_step % self.args.completion_logging_steps == 0:
            prompts_to_log = gather_object(prompts_text)
            completions_to_log = gather_object(completions_text)
            logged_reward = recipient_is_correct if group_pair_available else verifier_correct
            rewards_to_log = gather_object([float(logged_reward)])
            if self.accelerator.is_main_process:
                print_prompt_completions_sample(
                    prompts_to_log,
                    completions_to_log,
                    rewards_to_log,
                    self._step,
                )
                artifact_path = Path(self.args.output_dir) / "conditioning_samples.jsonl"
                artifact_path.parent.mkdir(parents=True, exist_ok=True)
                teacher_completion_text = self.processing_class.decode(
                    teacher_inputs[0, teacher_prompt_length:].tolist(), skip_special_tokens=False
                )
                with artifact_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({
                        "global_step": self.state.global_step,
                        "teacher_conditioning": self.teacher_conditioning,
                        "prompt": prompts_text[0],
                        "teacher_prompt": teacher_prompts_text[0] if teacher_prompts_text else prompts_text[0],
                        "completion": completion_text,
                        "teacher_completion_state": teacher_completion_text,
                        "parsed_answer": str(verification.parsed_answer) if verification else None,
                        "answer_text": verification.answer_text if verification else None,
                        "answer_source": verification.source if verification else None,
                        "answer_token_span": answer_token_span,
                        "span_status": span_status,
                        "conditioning_available": conditioning_available,
                        "eligible_state_ratio": eligible_state_ratio,
                        "group_pair_available": group_pair_available,
                        "group_positive_fallback": group_positive_fallback,
                        "group_donor_attempt": group_donor_attempt,
                        "group_recipient_attempt": group_recipient_attempt,
                    }, ensure_ascii=False) + "\n")

        return {
            "prompt_length": prompt_length,
            "teacher_prompt_length": teacher_prompt_length,
            "student_inputs": student_inputs.cpu(),
            "student_outputs": student_outputs.cpu(),
            "teacher_inputs": teacher_inputs.cpu(),
            "privileged_mask": privileged_mask.cpu(),
            "steps": steps,
            "gen_length": gen_length,
            "block_length": block_length,
            "is_correct": is_correct,
            "conditioning_available": conditioning_available,
            "eligible_state_ratio": eligible_state_ratio,
            "answer_token_span": answer_token_span,
            "answer_source": verification.source if verification else None,
            "span_status": span_status,
            "rollout_attempts": attempt,
            "final_completion_ids": final_sequence[prompt_length:].cpu(),
        }
