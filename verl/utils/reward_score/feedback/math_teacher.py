"""
Teacher prompt builder for math OPSD / SDPO runs with privileged reference solutions.

For math teacher rollouts, the teacher policy receives:
1. The original problem text
2. A worked reference solution from the dataset
3. An instruction to solve the problem again in its own way
"""

TEACHER_PROMPT_TEMPLATE = """Problem: {problem_text}
Here is a reference solution: {solution_text}

After understanding the reference solution, solve the problem again using your own approach.
Keep the reasoning brief and focused on the essential steps.
Your final line must be exactly:
{final_answer_instruction}
Do not write anything after that final line.

Answer:"""


def answer_format_for_data_source(data_source: str | None) -> str:
    """Return the final-answer format expected by the reward function."""
    if data_source in {"gsm8k", "openai/gsm8k"}:
        return "gsm8k_hash"
    return "boxed"


def final_answer_instruction(answer_format: str) -> str:
    if answer_format == "gsm8k_hash":
        return "#### <answer>"
    if answer_format == "boxed":
        return r"\boxed{<answer>}"
    raise ValueError(f"Unsupported math teacher answer format: {answer_format}")


def build_teacher_prompt(
    problem_text: str,
    solution_text: str,
    data_source: str | None = None,
    answer_format: str | None = None,
) -> str:
    """Build the privileged math-teacher prompt."""
    answer_format = answer_format or answer_format_for_data_source(data_source)
    return TEACHER_PROMPT_TEMPLATE.format(
        problem_text=problem_text,
        solution_text=solution_text,
        final_answer_instruction=final_answer_instruction(answer_format),
    )
