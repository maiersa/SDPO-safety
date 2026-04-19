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
#### <answer>
Do not write anything after that final line.

Answer:"""


def build_teacher_prompt(problem_text: str, solution_text: str) -> str:
    """Build the privileged math-teacher prompt."""
    return TEACHER_PROMPT_TEMPLATE.format(
        problem_text=problem_text,
        solution_text=solution_text,
    )
