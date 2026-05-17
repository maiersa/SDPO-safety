"""Node selection policy for compute-conscious alignment diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class NodeScore:
    token_position: int
    student_entropy: float
    student_teacher_kl: float
    special_token: bool = False
    after_final_answer: bool = False


def select_diagnostic_nodes(
    node_scores: Iterable[NodeScore],
    nodes_per_rollout: int,
    min_generated_position: int = 3,
    num_high_kl: int | None = None,
) -> list[tuple[NodeScore, str]]:
    """Select high-KL and high-entropy nodes from scored positions."""
    candidates = [
        score
        for score in node_scores
        if score.token_position >= min_generated_position and not score.special_token and not score.after_final_answer
    ]
    if nodes_per_rollout <= 0:
        return []

    num_high_kl = num_high_kl if num_high_kl is not None else max(nodes_per_rollout - 1, 0)
    selected: list[tuple[NodeScore, str]] = []
    used_positions: set[int] = set()

    for score in sorted(candidates, key=lambda item: item.student_teacher_kl, reverse=True):
        if len(selected) >= num_high_kl:
            break
        selected.append((score, "high_kl"))
        used_positions.add(score.token_position)

    for score in sorted(candidates, key=lambda item: item.student_entropy, reverse=True):
        if len(selected) >= nodes_per_rollout:
            break
        if score.token_position in used_positions:
            continue
        selected.append((score, "high_entropy"))
        used_positions.add(score.token_position)

    return selected

