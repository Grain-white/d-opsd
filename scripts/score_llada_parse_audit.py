#!/usr/bin/env python3
"""Score saved LLaDA generations with the official OPSD/RLCSD verifier."""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--rlcsd-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(args.rlcsd_root).resolve()))
    from src.opsd_format import extract_boxed_answer, grade_boxed_answer

    rows = []
    for pattern in args.input:
        paths = sorted(glob.glob(pattern))
        if not paths:
            raise FileNotFoundError(pattern)
        for path in paths:
            if path.endswith(".jsonl"):
                source_rows = [json.loads(line) for line in open(path, encoding="utf-8")]
                source_rows = [
                    {
                        "question": row["prompt"],
                        "generation": row["completion"],
                        "ground_truth": None,
                    }
                    for row in source_rows
                ]
            else:
                payload = json.loads(Path(path).read_text(encoding="utf-8"))
                source_rows = [
                    {
                        "question": row["question"],
                        "generation": row["generations"],
                        "ground_truth": str(row["ground_truth"]),
                    }
                    for row in payload["generations"]
                ]
            rows.extend(source_rows)

    # JSONL trainer artifacts do not repeat the gold answer. Recover it by
    # question identity from the same locally cached, seed-42 shuffled split.
    missing_gold = any(row["ground_truth"] is None for row in rows)
    if missing_gold:
        os.environ.setdefault(
            "HF_DATASETS_CACHE",
            "/home/fit/alex1/WORK/Meiqi.Gu/d-opsd/outputs/hf_datasets_rw",
        )
        from datasets import load_dataset

        dataset_dir = Path(
            "/home/fit/alex1/WORK/Meiqi.Gu/d-opsd/outputs/datasets/gsm8k/main/test-00000-of-00001.parquet"
        )
        dataset = load_dataset("parquet", data_files={"test": str(dataset_dir)})["test"]
        answer_by_question = {
            example["question"]: example["answer"].split("####", 1)[1].strip()
            for example in dataset
        }
        for row in rows:
            matches = [question for question in answer_by_question if question in row["question"]]
            if len(matches) != 1:
                raise ValueError(f"Could not uniquely recover GSM8K question: {len(matches)} matches")
            row["question"] = matches[0]
            row["ground_truth"] = answer_by_question[matches[0]]

    seen = set()
    scored = []
    for row in rows:
        if row["question"] in seen:
            raise ValueError(f"Duplicate question: {row['question'][:80]}")
        seen.add(row["question"])
        answer = extract_boxed_answer(row["generation"])
        correct = grade_boxed_answer(answer, row["ground_truth"])
        scored.append(
            {
                **row,
                "official_answer": answer,
                "official_parsed": answer is not None,
                "official_correct": bool(correct),
                "has_answer_tag": "<answer>" in row["generation"],
                "has_eot": "<|eot_id|>" in row["generation"],
            }
        )

    n = len(scored)
    parsed = sum(row["official_parsed"] for row in scored)
    correct = sum(row["official_correct"] for row in scored)
    metrics = {
        "samples": n,
        "parsed": parsed,
        "correct": correct,
        "parse_rate": parsed / n,
        "raw_accuracy": correct / n,
        "accuracy_given_parsed": correct / parsed if parsed else 0.0,
        "answer_tag_rate": sum(row["has_answer_tag"] for row in scored) / n,
        "eot_rate": sum(row["has_eot"] for row in scored) / n,
    }
    payload = {"label": args.label, "metrics": metrics, "rows": scored}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
