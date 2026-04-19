import re

_SOLUTION_CLIP_CHARS = 300


def extract_solution(solution_str: str):
    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

    solutions = re.findall(r"####\s*(-?[$0-9.,]+)", solution_str)
    if len(solutions) == 0:
        return None
    return solutions[-1].replace(",", "").replace("$", "")


def has_strict_final_line_format(solution_str: str) -> bool:
    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

    lines = [line.strip() for line in solution_str.splitlines() if line.strip()]
    if not lines:
        return False

    return re.fullmatch(r"####\s*-?[$0-9.,]+", lines[-1]) is not None


def compute_score(solution_str: str, ground_truth: str, extra_info=None) -> dict:
    extra_info = extra_info or {}
    pred = extract_solution(solution_str)
    correct = pred == ground_truth if pred is not None else False
    was_truncated = extra_info.get("truncated", False)
    incorrect_format = pred is None
    strict_final_line_format = has_strict_final_line_format(solution_str)

    feedback = ""
    if incorrect_format and not was_truncated:
        feedback = 'Your answer had the wrong format. The solution must be given in the format: #### your_answer.'
    elif was_truncated:
        feedback = "Your response was truncated because it exceeded the maximum length."

    return {
        "score": 1.0 if correct else 0.0,
        "acc": 1.0 if correct else 0.0,
        "pred": pred or "",
        "incorrect_format": 1 if incorrect_format else 0,
        "strict_final_line_format": 1 if strict_final_line_format else 0,
        "truncated": 1 if was_truncated else 0,
        "truncated_and_missing_answer": 1 if incorrect_format and was_truncated else 0,
        "feedback": feedback,
    }
