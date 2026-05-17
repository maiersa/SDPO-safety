"""Prompt templates for student and privileged-context teacher scoring."""

from __future__ import annotations

from enum import StrEnum


class TeacherContext(StrEnum):
    CONTROL = "control"
    ANSWER_ONLY = "answer_only"
    FULL_SOLUTION = "full_solution"


STUDENT_PROMPT_TEMPLATE = """Question:
{question}

Solve the problem step by step and give the final answer.
"""

ANSWER_ONLY_TEACHER_TEMPLATE = """Question:
{question}

The correct final answer is:
{answer}

The student solution so far is:
{student_prefix}

Continue the solution in a way that leads to the correct answer.
"""

FULL_SOLUTION_TEACHER_TEMPLATE = """Question:
{question}

Reference solution:
{reference_solution}

The student solution so far is:
{student_prefix}

Continue or correct the reasoning in a way that follows the reference solution.
"""


def build_student_prompt(question: str) -> str:
    return STUDENT_PROMPT_TEMPLATE.format(question=question).strip()


def build_teacher_prompt(
    context: str | TeacherContext,
    question: str,
    answer: str,
    reference_solution: str,
    student_prefix: str,
    student_prompt: str | None = None,
) -> str:
    context = TeacherContext(context)
    if context == TeacherContext.CONTROL:
        return student_prompt or build_student_prompt(question)
    if context == TeacherContext.ANSWER_ONLY:
        return ANSWER_ONLY_TEACHER_TEMPLATE.format(
            question=question,
            answer=answer,
            student_prefix=student_prefix,
        ).strip()
    if context == TeacherContext.FULL_SOLUTION:
        return FULL_SOLUTION_TEACHER_TEMPLATE.format(
            question=question,
            reference_solution=reference_solution,
            student_prefix=student_prefix,
        ).strip()
    raise ValueError(f"Unsupported teacher context: {context}")

