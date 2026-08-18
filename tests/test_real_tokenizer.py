import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "d-opsd"))
os.environ.setdefault("LOCAL_RANK", "0")

from teacher_conditioning import map_char_span_to_token_span  # noqa: E402
from utils import get_all_parsed_answer_with_metadata  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    original = "<reasoning>Compute carefully.</reasoning><answer>\\boxed{1,234}</answer>"
    token_ids = tokenizer.encode(original, add_special_tokens=False)
    decoded = tokenizer.decode(token_ids, skip_special_tokens=False)
    verification = get_all_parsed_answer_with_metadata(decoded, "1234", "gsm8k")
    assert verification.is_correct and verification.char_span
    token_span, status = map_char_span_to_token_span(
        tokenizer,
        token_ids,
        decoded,
        verification.char_span,
        verification.answer_text,
    )
    assert token_span is not None, status
    recovered = tokenizer.decode(token_ids[slice(*token_span)], skip_special_tokens=False)
    assert verification.answer_text.strip() in recovered.strip() or recovered.strip() in verification.answer_text.strip()
    print({"token_span": token_span, "status": status, "recovered": recovered})


if __name__ == "__main__":
    main()
