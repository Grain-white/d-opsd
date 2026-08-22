import os
import random
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "d-opsd"))
os.environ.setdefault("LOCAL_RANK", "0")

from teacher_conditioning import (  # noqa: E402
    build_answer_prompt,
    fully_masked_rows,
    map_char_span_to_token_span,
    sample_future_hint_positions,
    select_group_rollout_pair,
    split_bounds,
)
from utils import get_all_parsed_answer_with_metadata  # noqa: E402


class CharacterTokenizer:
    eos_token_id = 999

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        result = {"input_ids": [ord(char) for char in text]}
        if return_offsets_mapping:
            result["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return result

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(token) for token in ids)


def test_gsm_verifier_returns_the_exact_span_it_used():
    text = "first \\boxed{1,234}; later <answer>-9</answer>"
    result = get_all_parsed_answer_with_metadata(text, "1234", "gsm8k")
    assert result.is_correct
    assert result.parsed_answer == 1234.0
    assert result.answer_text == "1,234"
    assert text[slice(*result.char_span)] == result.answer_text
    assert result.source == "boxed"


def test_answer_tag_uses_last_numeric_value_like_existing_verifier():
    text = "<answer>calculation 3, final -2.5</answer>"
    result = get_all_parsed_answer_with_metadata(text, "-2.5", "gsm8k")
    assert result.is_correct
    assert result.answer_text == "-2.5"
    assert text[slice(*result.char_span)] == "-2.5"


def test_math_verifier_locates_nested_last_box():
    text = "trial \\boxed{1}; final \\boxed{\\frac{1}{2}}"
    result = get_all_parsed_answer_with_metadata(text, "solution \\boxed{\\frac{1}{2}}", "math")
    assert result.is_correct
    assert result.answer_text == "\\frac{1}{2}"
    assert text[slice(*result.char_span)] == result.answer_text


def test_char_to_token_mapping_is_verified():
    tokenizer = CharacterTokenizer()
    text = "reason <answer>-12</answer>"
    start = text.index("-12")
    span, status = map_char_span_to_token_span(
        tokenizer,
        tokenizer.encode(text),
        text,
        (start, start + 3),
        "-12",
    )
    assert status == "offset_mapping"
    assert tokenizer.decode(tokenizer.encode(text)[slice(*span)]) == "-12"


def test_prompt_is_copied_and_contains_answer_only():
    messages = [{"role": "user", "content": "Solve 1+1"}]
    teacher = build_answer_prompt(messages, "2")
    assert messages[0]["content"] == "Solve 1+1"
    content = teacher[0]["content"]
    # Formal OPSD answer_only injection (RLCSD).
    assert "Here is a reference solution to this problem:" in content
    assert "=== Reference Solution Begin ===" in content
    assert "Correct final answer: 2" in content
    assert "=== Reference Solution End ===" in content
    assert "After reading the reference solution above" in content
    assert "Privileged information for the teacher only" not in content


def test_clamp_eligibility_and_balanced_split():
    mask = 9
    trajectory = torch.tensor([[1, mask, mask], [1, 2, mask], [1, 2, 3]])
    assert fully_masked_rows(trajectory, (1, 3), mask).tolist() == [True, False, False]
    assert [split_bounds(10, 3, index) for index in range(3)] == [(0, 4), (4, 7), (7, 10)]


def test_future_hints_are_eligible_and_reproducible():
    eligible = list(range(10, 40))
    first = sample_future_hint_positions(eligible, 0.2, 0.6, 5, 10, random.Random(7))
    second = sample_future_hint_positions(eligible, 0.2, 0.6, 5, 10, random.Random(7))
    assert first == second
    assert first
    assert set(first).issubset(eligible)


def test_group_pair_uses_correct_donor_and_incorrect_recipient():
    candidates = [
        {"is_correct": False, "answer_text": "11", "token_span": (20, 21)},
        {"is_correct": True, "answer_text": "42", "token_span": (18, 19)},
        {"is_correct": False, "answer_text": None, "token_span": None},
    ]
    assert select_group_rollout_pair(candidates) == (1, 0)


def test_group_pair_requires_a_parseable_incorrect_recipient():
    candidates = [
        {"is_correct": True, "answer_text": "42", "token_span": (18, 19)},
        {"is_correct": False, "answer_text": None, "token_span": None},
    ]
    assert select_group_rollout_pair(candidates) == (0, None)


def test_group_pair_keeps_the_earliest_parseable_wrong_rollout():
    candidates = [
        {"is_correct": False, "answer_text": "11", "token_span": (20, 21)},
        {"is_correct": False, "answer_text": "12", "token_span": (22, 23)},
        {"is_correct": True, "answer_text": "42", "token_span": (18, 19)},
    ]
    assert select_group_rollout_pair(candidates) == (2, 0)


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"{len(tests)} teacher-conditioning tests passed")
