#!/usr/bin/env python3
"""Regression test: token clipping must not turn a KL/JSD loss negative."""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "d-opsd"))

from d_opsd_trainer import dOPSDTrainer  # noqa: E402


def main():
    trainer = dOPSDTrainer.__new__(dOPSDTrainer)
    student = torch.tensor(
        [[8.0, -2.0, 1.0, 0.0], [-4.0, 7.0, 2.0, -1.0]], dtype=torch.bfloat16
    )
    teacher = torch.tensor(
        [[-1.0, 7.0, 0.0, 2.0], [6.0, -3.0, 1.0, 0.0]], dtype=torch.bfloat16
    )
    token_clip = 0.05

    for beta in (0, 1):
        loss, clip_ratio, _ = trainer.generalized_jsd_loss(
            student,
            teacher,
            beta=beta,
            reduction="batchmean",
            token_clip=token_clip,
        )
        input_logits, target_logits = (
            (student, teacher) if beta == 0 else (teacher, student)
        )
        pointwise = F.kl_div(
            F.log_softmax(input_logits.float(), dim=-1),
            F.log_softmax(target_logits.float(), dim=-1),
            reduction="none",
            log_target=True,
        )
        expected = pointwise.sum(dim=-1).clamp_min(0.0).clamp(max=token_clip).mean()

        assert loss.item() >= 0.0
        assert torch.allclose(loss, expected, atol=1e-7), (beta, loss, expected)
        assert 0.0 <= clip_ratio.item() <= 1.0
        print({"beta": beta, "loss": loss.item(), "clip_ratio": clip_ratio.item()})


if __name__ == "__main__":
    main()
