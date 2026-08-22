#!/usr/bin/env python3
"""Upload completed GSM8K unified val-256 evaluations to SwanLab.

The historical evaluation jobs ran with SWANLAB_MODE=disabled.  Their stdout
contains a START marker followed by one EVAL_ONLY_METRICS JSON object per
checkpoint.  This script reconstructs one evaluation curve per conditioning
method without modifying or resuming the original training runs.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


SOURCES = {
    "answer_prompt": ("dopsd-answer_prompt-v256-474476.out", "474476"),
    "answer_clamp": ("dopsd-answer_clamp-v256-474477.out", "474477"),
    "self_future": ("dopsd-self_future-v256-474478.out", "474478"),
}

START_RE = re.compile(r"START reeval-(answer_prompt|answer_clamp|self_future)-ckpt(\d+)-val256-p1-seed42-v1$")
METRICS_PREFIX = "EVAL_ONLY_METRICS="


def parse_log(path: Path, expected_method: str) -> list[tuple[int, dict[str, float]]]:
    records: list[tuple[int, dict[str, float]]] = []
    pending_step: int | None = None
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = START_RE.search(raw_line)
        if match:
            method, step_text = match.groups()
            if method != expected_method:
                raise ValueError(f"Unexpected method {method!r} in {path}")
            pending_step = int(step_text)
            continue
        if raw_line.startswith(METRICS_PREFIX):
            if pending_step is None:
                raise ValueError(f"Metrics without START marker in {path}")
            metrics = json.loads(raw_line[len(METRICS_PREFIX) :])
            numeric = {
                f"val256/{key.removeprefix('eval_')}": float(value)
                for key, value in metrics.items()
                if isinstance(value, (int, float))
            }
            numeric["val256/checkpoint_step"] = float(pending_step)
            records.append((pending_step, numeric))
            pending_step = None

    if pending_step is not None:
        # The old jobs failed after beginning the next checkpoint.  Never upload
        # an incomplete point that lacks EVAL_ONLY_METRICS.
        print(f"warning: ignoring incomplete checkpoint {pending_step} in {path}")
    if not records:
        raise ValueError(f"No complete evaluation records found in {path}")
    steps = [step for step, _ in records]
    if steps != sorted(set(steps)):
        raise ValueError(f"Duplicate or unordered checkpoints in {path}: {steps}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-dir", type=Path, default=Path("logs"))
    parser.add_argument("--project", default="d-opsd-prompt-vs-clamp")
    parser.add_argument("--mode", choices=("online", "offline", "disabled"), default="disabled")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=tuple(SOURCES),
        default=list(SOURCES),
        help="Subset of conditioning methods to parse/upload.",
    )
    args = parser.parse_args()

    parsed: dict[str, tuple[str, list[tuple[int, dict[str, float]]]]] = {}
    for method in args.methods:
        filename, job_id = SOURCES[method]
        records = parse_log(args.logs_dir / filename, method)
        parsed[method] = (job_id, records)
        summary = ", ".join(
            f"{step}:{metrics['val256/verifier_accuracy']:.6f}" for step, metrics in records
        )
        print(f"{method}: {summary}")

    if args.mode == "disabled":
        print("dry-run only; pass --mode online to upload")
        return

    import swanlab

    for method, (job_id, records) in parsed.items():
        run = swanlab.init(
            reinit=True,
            mode=args.mode,
            project=args.project,
            name=f"unified-val256-p1-{method}-seed42-v1",
            description=(
                "Post-hoc upload of the completed GSM8K unified validation-256 "
                "pass@1 checkpoint evaluations. Original evaluation ran with "
                "SwanLab disabled; incomplete checkpoints are excluded."
            ),
            job_type="evaluation",
            group="gsm8k-unified-val256-p1-seed42-v1",
            tags=["gsm8k", "unified", "val256", "pass@1", method, "posthoc-sync"],
            config={
                "dataset": "gsm8k",
                "method": method,
                "seed": 42,
                "validation_samples": 256,
                "eval_pass_k": 1,
                "source_slurm_job_id": job_id,
                "source_log": str(args.logs_dir / SOURCES[method][0]),
                "training_run_name": f"{method}-gsm-seed42-steps300-unified-v1",
                "completed_checkpoint_steps": [step for step, _ in records],
            },
        )
        for step, metrics in records:
            run.log(metrics, step=step)
        run.finish()


if __name__ == "__main__":
    main()
