"""Answer extraction and deterministic grading for alignment rollouts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation


_SOLUTION_CLIP_CHARS = 300


@dataclass(frozen=True)
class GradeResult:
    raw_answer: str | None
    normalized_answer: str | None
    normalized_ground_truth: str | None
    is_correct: bool
    invalid_parse: bool


def grade_answer(completion: str, ground_truth: str, source: str = "gsm8k") -> GradeResult:
    source = source.lower()
    if source in {"gsm8k", "openai/gsm8k"}:
        raw_answer = extract_gsm8k_solution(completion, method="flexible")
    elif source in {"math", "synthetic"}:
        raw_answer = extract_boxed_or_last_number(completion)
    else:
        raw_answer = extract_boxed_or_last_number(completion)

    normalized_answer = normalize_answer(raw_answer)
    normalized_ground_truth = normalize_answer(ground_truth)
    return GradeResult(
        raw_answer=raw_answer,
        normalized_answer=normalized_answer,
        normalized_ground_truth=normalized_ground_truth,
        is_correct=normalized_answer is not None and normalized_answer == normalized_ground_truth,
        invalid_parse=normalized_answer is None,
    )


def extract_gsm8k_solution(solution_str: str, method: str = "strict") -> str | None:
    if method not in {"strict", "flexible"}:
        raise ValueError(f"Unknown GSM8K extraction method: {method}")

    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

    if method == "strict":
        solutions = re.findall(r"#### (\-?[0-9\.\,]+)", solution_str)
        return solutions[-1].replace(",", "").replace("$", "") if solutions else None

    answers = re.findall(r"(\-?[0-9\.\,]+)", solution_str)
    for answer in reversed(answers):
        if answer not in {"", "."}:
            return answer
    return None


def extract_boxed_or_last_number(text: str) -> str | None:
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        return boxed[-1].strip()

    numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text[-500:])
    if numbers:
        return numbers[-1]
    return None


def normalize_answer(value: str | None) -> str | None:
    if value is None:
        return None

    normalized = value.strip()
    normalized = normalized.replace("$", "").replace(",", "")
    normalized = normalized.rstrip(".")
    normalized = re.sub(r"\s+", "", normalized)
    if not normalized:
        return None

    try:
        decimal = Decimal(normalized)
    except InvalidOperation:
        return normalized.lower()

    return format(decimal.normalize(), "f")
