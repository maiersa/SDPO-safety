"""Candidate-set gradient calculations for the OPSD alignment diagnostic."""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Iterable, Sequence


class DistillationObjective(StrEnum):
    FORWARD_KL = "forward_kl"
    REVERSE_KL = "reverse_kl"
    JSD = "jsd"


def normalize_probs(values: Sequence[float], epsilon: float = 1e-12) -> list[float]:
    """Normalize non-negative probability-like values over a candidate set."""
    if not values:
        raise ValueError("values must be non-empty")

    clipped = [max(float(value), epsilon) for value in values]
    total = sum(clipped)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("values must have a positive finite sum")
    return [value / total for value in clipped]


def softmax(logits: Sequence[float]) -> list[float]:
    if not logits:
        raise ValueError("logits must be non-empty")

    max_logit = max(logits)
    exp_values = [math.exp(float(logit) - max_logit) for logit in logits]
    total = sum(exp_values)
    return [value / total for value in exp_values]


def renormalize_logprobs(logprobs: Sequence[float]) -> list[float]:
    """Convert candidate-set log probabilities into renormalized probabilities."""
    return softmax(logprobs)


def ideal_gradient(p_student: Sequence[float], p_success: Sequence[float]) -> list[float]:
    """Compute g_ideal[k] = p_s[k] * (P_success[k] - E_s[P_success])."""
    _check_same_length(p_student, p_success)
    p_s = normalize_probs(p_student)
    success = [float(value) for value in p_success]
    baseline = sum(prob * value for prob, value in zip(p_s, success, strict=True))
    return [prob * (value - baseline) for prob, value in zip(p_s, success, strict=True)]


def distillation_gradient(
    p_student: Sequence[float],
    p_teacher: Sequence[float],
    objective: str | DistillationObjective = DistillationObjective.FORWARD_KL,
    jsd_alpha: float = 0.5,
) -> list[float]:
    """Return the student-logit descent direction for a candidate-set distillation loss.

    The returned vector has the same sign convention as the paper's distillation
    gradient: it is the direction that minimizing the distillation objective
    would push the student logits.
    """
    _check_same_length(p_student, p_teacher)
    p_s = normalize_probs(p_student)
    p_t = normalize_probs(p_teacher)
    objective = DistillationObjective(objective)

    if objective == DistillationObjective.FORWARD_KL:
        return _forward_kl_descent_direction(p_s, p_t)
    if objective == DistillationObjective.REVERSE_KL:
        return [teacher - student for student, teacher in zip(p_s, p_t, strict=True)]
    if objective == DistillationObjective.JSD:
        return _jsd_descent_direction(p_s, p_t, alpha=jsd_alpha)

    raise ValueError(f"Unsupported distillation objective: {objective}")


def alignment(
    g_ideal: Sequence[float],
    g_distill: Sequence[float],
    min_gradient_norm: float = 1e-8,
) -> float | None:
    """Compute cosine alignment, returning None when either vector is too small."""
    _check_same_length(g_ideal, g_distill)
    dot = sum(float(a) * float(b) for a, b in zip(g_ideal, g_distill, strict=True))
    ideal_norm = math.sqrt(sum(float(value) ** 2 for value in g_ideal))
    distill_norm = math.sqrt(sum(float(value) ** 2 for value in g_distill))
    if ideal_norm < min_gradient_norm or distill_norm < min_gradient_norm:
        return None
    return dot / (ideal_norm * distill_norm)


def student_teacher_kl(p_student: Sequence[float], p_teacher: Sequence[float]) -> float:
    """Compute KL(student || teacher) on a renormalized candidate set."""
    _check_same_length(p_student, p_teacher)
    p_s = normalize_probs(p_student)
    p_t = normalize_probs(p_teacher)
    return sum(student * math.log(student / teacher) for student, teacher in zip(p_s, p_t, strict=True))


def entropy(probs: Sequence[float]) -> float:
    p = normalize_probs(probs)
    return -sum(prob * math.log(prob) for prob in p)


def summarize_alignment(values: Iterable[float | None]) -> dict[str, float | int | None]:
    valid = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not valid:
        return {
            "count_nodes": 0,
            "mean_alignment": None,
            "median_alignment": None,
            "std_alignment": None,
            "standard_error": None,
            "fraction_positive_alignment": None,
        }

    sorted_values = sorted(valid)
    count = len(sorted_values)
    mean = sum(sorted_values) / count
    mid = count // 2
    median = sorted_values[mid] if count % 2 else (sorted_values[mid - 1] + sorted_values[mid]) / 2
    variance = sum((value - mean) ** 2 for value in sorted_values) / (count - 1) if count > 1 else 0.0
    std = math.sqrt(variance)
    return {
        "count_nodes": count,
        "mean_alignment": mean,
        "median_alignment": median,
        "std_alignment": std,
        "standard_error": std / math.sqrt(count) if count > 1 else 0.0,
        "fraction_positive_alignment": sum(value > 0 for value in sorted_values) / count,
    }


def _forward_kl_descent_direction(p_student: Sequence[float], p_teacher: Sequence[float]) -> list[float]:
    ell = [math.log(student) - math.log(teacher) for student, teacher in zip(p_student, p_teacher, strict=True)]
    ell_bar = sum(student * value for student, value in zip(p_student, ell, strict=True))
    return [-student * (value - ell_bar) for student, value in zip(p_student, ell, strict=True)]


def _jsd_descent_direction(p_student: Sequence[float], p_teacher: Sequence[float], alpha: float) -> list[float]:
    if not 0.0 < alpha < 1.0:
        raise ValueError("jsd_alpha must be in (0, 1)")

    mixture = [
        (1.0 - alpha) * student + alpha * teacher
        for student, teacher in zip(p_student, p_teacher, strict=True)
    ]
    d_loss_d_prob = [
        (1.0 - alpha) * math.log(student / mixed)
        for student, mixed in zip(p_student, mixture, strict=True)
    ]
    expected = sum(student * value for student, value in zip(p_student, d_loss_d_prob, strict=True))
    grad_logits = [
        student * (value - expected)
        for student, value in zip(p_student, d_loss_d_prob, strict=True)
    ]
    return [-value for value in grad_logits]


def _check_same_length(left: Sequence[object], right: Sequence[object]) -> None:
    if len(left) != len(right):
        raise ValueError(f"length mismatch: {len(left)} != {len(right)}")
    if len(left) == 0:
        raise ValueError("vectors must be non-empty")

