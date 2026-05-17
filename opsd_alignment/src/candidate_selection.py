"""Candidate-token helpers for selected diagnostic nodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class CandidateToken:
    token_id: int
    student_logprob: float | None = None
    teacher_logprob: float | None = None
    in_student_topk: bool = False
    in_teacher_topk: bool = False


def union_topk_candidates(
    student_token_ids: Sequence[int],
    teacher_token_ids: Sequence[int],
    student_logprobs: Sequence[float] | None = None,
    teacher_logprobs: Sequence[float] | None = None,
) -> list[CandidateToken]:
    student_logprobs = student_logprobs or [None] * len(student_token_ids)
    teacher_logprobs = teacher_logprobs or [None] * len(teacher_token_ids)
    if len(student_token_ids) != len(student_logprobs):
        raise ValueError("student token/logprob length mismatch")
    if len(teacher_token_ids) != len(teacher_logprobs):
        raise ValueError("teacher token/logprob length mismatch")

    merged: dict[int, CandidateToken] = {}
    for token_id, logprob in zip(student_token_ids, student_logprobs, strict=True):
        merged[int(token_id)] = CandidateToken(
            token_id=int(token_id),
            student_logprob=None if logprob is None else float(logprob),
            in_student_topk=True,
        )

    for token_id, logprob in zip(teacher_token_ids, teacher_logprobs, strict=True):
        token_id = int(token_id)
        existing = merged.get(token_id)
        merged[token_id] = CandidateToken(
            token_id=token_id,
            student_logprob=existing.student_logprob if existing else None,
            teacher_logprob=None if logprob is None else float(logprob),
            in_student_topk=existing.in_student_topk if existing else False,
            in_teacher_topk=True,
        )

    return list(merged.values())

