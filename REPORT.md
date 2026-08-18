# LLaDA Answer Prompt vs. Answer Clamp Distillation

## Question and hypotheses

This experiment asks whether giving a diffusion-LM teacher the verified final answer at its natural completion position produces a better distillation signal than giving the same answer in a teacher-only prompt. The primary comparison is information matched: both arms use the same correct on-policy rollout, pass@8 budget, eligible diffusion states, target-selection rule, optimizer, and checkpoint schedule.

The main hypothesis is that answer clamp will improve accuracy or accuracy AUC at equal GPU-hours because the privileged evidence is geometrically aligned with the answer region. Secondary mechanism hypotheses are higher teacher probability on correct-trajectory tokens and a different teacher/student token-ranking signal. Response length and parse rate are treated as possible confounders rather than evidence by themselves.

## Implemented conditions

| Condition | Teacher prompt | Fixed completion tokens | Primary? |
|---|---|---|---|
| `answer_prompt` | Verified rollout answer only | None | Yes |
| `answer_clamp` | Original prompt | Natural verified answer span | Yes |
| `answer_clamp_future` | Original prompt | Answer plus fixed future-reasoning chunks | Ablation |
| `self_future` | Original prompt | Original d-OPSD random future tokens | Baseline |

The verifier returns the parsed answer, correctness, exact character span, answer text, and format source. Token offsets are accepted only when re-tokenization reproduces the rollout IDs; otherwise only a unique answer-token subsequence is accepted. Unlocatable answers and trajectories with too few eligible states receive zero training weight.

Future-hint chunks are sampled once per trajectory with lengths in `[5, 10]` and a ratio in `[0.2, 0.6]`. They exclude the answer and current denoising targets, remain fixed, and are excluded from distillation loss.

## Metrics

Training logs include completion length, pass@k attempts and success, verifier accuracy, span-locatable rate, answer-format source, eligible-state ratio, rollout throughput, teacher/student forward time, JSD/KL loss, top-k overlap, top-k Kendall tau, and teacher/student probability of the correct-trajectory token.

Final evaluation reports raw accuracy, parse rate, conditional accuracy, response-length distribution, paired bootstrap confidence intervals, checkpoint learning curves, accuracy AUC, wall time, and GPU-hours. Clamp is declared better only if accuracy/AUC improve at matched compute with consistent seed direction and without a response-length or parse-rate confound.

## Results

Results will be inserted here from `outputs/analysis` after the matched Slurm runs and checkpoint evaluations complete.
