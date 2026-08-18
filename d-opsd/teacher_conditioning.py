"""Pure helpers for answer-conditioned d-OPSD teachers.

The helpers in this module deliberately do not know the ground-truth answer.  They
operate only on the answer span selected by the verifier from a correct on-policy
rollout.
"""

from __future__ import annotations

import copy
import random
from typing import Sequence

import torch


# Formal OPSD answer-only privileged block (RLCSD `opsd_format.build_teacher_messages`
# with `privileged_text_mode=answer_only` / `TEACHER_PROMPT_TEMPLATE_ANSWER_ONLY`):
# wrap the verified final answer as a "reference solution" and ask the teacher to
# internalize the reasoning before producing its own solution.
ANSWER_PROMPT_FRAMING = "Here is a reference solution to this problem:"
ANSWER_PROMPT_TRANSITION = (
    "\n\nAfter reading the reference solution above, make sure you understand "
    "the reasoning behind each step.\n"
)
ANSWER_PROMPT_TEMPLATE = (
    "\n\n"
    f"{ANSWER_PROMPT_FRAMING}\n"
    "=== Reference Solution Begin ===\n"
    "Correct final answer: {answer}\n"
    "=== Reference Solution End ==="
    f"{ANSWER_PROMPT_TRANSITION}"
)


def build_answer_prompt(messages, answer_text: str):
    """Return a copy of a conversational prompt with OPSD answer-only injection."""
    teacher_messages = copy.deepcopy(messages)
    if not teacher_messages or "content" not in teacher_messages[-1]:
        raise ValueError("Expected a non-empty conversational prompt")
    teacher_messages[-1]["content"] += ANSWER_PROMPT_TEMPLATE.format(answer=answer_text)
    return teacher_messages


def _find_subsequence(sequence: Sequence[int], needle: Sequence[int]):
    if not needle or len(needle) > len(sequence):
        return []
    return [
        index
        for index in range(len(sequence) - len(needle) + 1)
        if list(sequence[index:index + len(needle)]) == list(needle)
    ]


def map_char_span_to_token_span(tokenizer, token_ids, decoded_text, char_span, answer_text):
    """Map a verified character span to a half-open token span.

    Fast-tokenizer offsets are used only when re-tokenizing reproduces the exact
    rollout token IDs.  Otherwise a unique answer-token subsequence is accepted.
    Ambiguous mappings return ``None`` rather than guessing.
    """
    token_ids = [int(token) for token in token_ids]
    start_char, end_char = char_span
    if not (0 <= start_char < end_char <= len(decoded_text)):
        return None, "invalid_char_span"
    if decoded_text[start_char:end_char] != answer_text:
        return None, "char_text_mismatch"

    try:
        encoded = tokenizer(
            decoded_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        encoded_ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]
        if encoded_ids and isinstance(encoded_ids[0], list):
            encoded_ids, offsets = encoded_ids[0], offsets[0]
        if [int(token) for token in encoded_ids] == token_ids:
            covered = [
                index for index, (left, right) in enumerate(offsets)
                if left < end_char and right > start_char
            ]
            if covered and covered == list(range(covered[0], covered[-1] + 1)):
                token_span = (covered[0], covered[-1] + 1)
                recovered = tokenizer.decode(token_ids[token_span[0]:token_span[1]], skip_special_tokens=False)
                if answer_text.strip() in recovered.strip() or recovered.strip() in answer_text.strip():
                    return token_span, "offset_mapping"
    except (KeyError, TypeError, ValueError, NotImplementedError):
        pass

    answer_ids = tokenizer.encode(answer_text, add_special_tokens=False)
    matches = _find_subsequence(token_ids, answer_ids)
    if len(matches) == 1:
        return (matches[0], matches[0] + len(answer_ids)), "unique_subsequence"
    return None, "ambiguous_subsequence" if matches else "subsequence_not_found"


def sample_future_hint_positions(
    eligible_positions,
    ratio_min: float,
    ratio_max: float,
    chunk_min: int,
    chunk_max: int,
    rng: random.Random,
):
    """Sample IGPO-style contiguous chunks from eligible future positions."""
    positions = sorted({int(position) for position in eligible_positions})
    if not positions:
        return []
    target = max(1, round(len(positions) * rng.uniform(ratio_min, ratio_max)))
    eligible = set(positions)
    selected = set()
    attempts = 0
    while len(selected) < target and eligible and attempts < len(positions) * 4:
        attempts += 1
        start = rng.choice(sorted(eligible))
        length = rng.randint(chunk_min, chunk_max)
        chunk = []
        for position in range(start, start + length):
            if position not in eligible:
                break
            chunk.append(position)
        if not chunk:
            eligible.discard(start)
            continue
        selected.update(chunk[: target - len(selected)])
        eligible.difference_update(chunk)
    return sorted(selected)


def split_bounds(total: int, parts: int, part_index: int):
    """Return balanced half-open bounds without dropping a remainder."""
    if parts <= 0 or not 0 <= part_index < parts:
        raise ValueError("Invalid split request")
    quotient, remainder = divmod(total, parts)
    start = part_index * quotient + min(part_index, remainder)
    size = quotient + (1 if part_index < remainder else 0)
    return start, start + size


def fully_masked_rows(trajectory, absolute_span, mask_id: int):
    """Rows where the rollout has not naturally revealed any answer token."""
    start, end = absolute_span
    if not 0 <= start < end <= trajectory.shape[1]:
        raise ValueError(f"Answer span {absolute_span} is outside trajectory width {trajectory.shape[1]}")
    return torch.all(trajectory[:, start:end] == mask_id, dim=1)
