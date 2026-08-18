#!/usr/bin/env python3
"""Paired prompt-vs-clamp evaluation with bootstrap confidence intervals."""

import argparse
import glob
import json
import random
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "d-opsd"))
from utils import get_all_parsed_answer_with_metadata  # noqa: E402


def load_records(path, dataset):
    records = {}
    source = Path(path)
    paths = sorted(source.glob("*_generations.json")) if source.is_dir() else [Path(p) for p in glob.glob(path)]
    if not paths:
        raise FileNotFoundError(f"No generation JSON files matched {path}")
    for json_path in paths:
        with json_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        for item in payload.get("generations", payload):
            question = item.get("question", item.get("prompt"))
            generation = item.get("generations", item.get("generation", ""))
            ground_truth = item.get("ground_truth", item.get("answer"))
            result = get_all_parsed_answer_with_metadata(generation, ground_truth, dataset)
            records[question] = {
                "correct": int(result.is_correct),
                "parsed": int(result.parsed_answer is not None),
                "response_chars": len(generation),
                "response_words": len(generation.split()),
            }
    return records


def summarize(records):
    values = list(records.values())
    correct = [item["correct"] for item in values]
    parsed = [item["parsed"] for item in values]
    lengths = [item["response_words"] for item in values]
    parsed_correct = [item["correct"] for item in values if item["parsed"]]
    quantiles = statistics.quantiles(lengths, n=4) if len(lengths) >= 4 else [None] * 3
    return {
        "n": len(values),
        "raw_accuracy": statistics.mean(correct) if correct else None,
        "parse_rate": statistics.mean(parsed) if parsed else None,
        "conditional_accuracy": statistics.mean(parsed_correct) if parsed_correct else None,
        "response_words_mean": statistics.mean(lengths) if lengths else None,
        "response_words_median": statistics.median(lengths) if lengths else None,
        "response_words_q25": quantiles[0],
        "response_words_q75": quantiles[2],
    }


def paired_bootstrap(prompt, clamp, samples, seed):
    questions = sorted(set(prompt) & set(clamp))
    rng = random.Random(seed)
    observed = statistics.mean(clamp[q]["correct"] - prompt[q]["correct"] for q in questions)
    deltas = []
    for _ in range(samples):
        draw = [rng.choice(questions) for _ in questions]
        deltas.append(statistics.mean(clamp[q]["correct"] - prompt[q]["correct"] for q in draw))
    deltas.sort()
    lower = deltas[int(0.025 * (len(deltas) - 1))]
    upper = deltas[int(0.975 * (len(deltas) - 1))]
    return {"paired_n": len(questions), "clamp_minus_prompt": observed, "ci95": [lower, upper]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--clamp", required=True)
    parser.add_argument("--dataset", choices=["gsm8k", "math"], default="gsm8k")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prompt = load_records(args.prompt, args.dataset)
    clamp = load_records(args.clamp, args.dataset)
    result = {
        "prompt": summarize(prompt),
        "clamp": summarize(clamp),
        "paired_bootstrap": paired_bootstrap(
            prompt, clamp, args.bootstrap_samples, args.seed
        ),
        "length_definition": "whitespace-delimited words; training logs retain model-token completion length",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
