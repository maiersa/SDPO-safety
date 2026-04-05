"""
SDPO teacher prompt builder with constitution injection.

For OPSD-style (on-policy self-distillation), the teacher model receives:
1. The original question from the user
2. Privileged context: the full constitution
3. Optional: feedback or previous attempt

This module constructs the teacher prompt with constitutional context.
"""

from pathlib import Path
from typing import Optional


def load_constitution(constitution_path: Optional[str] = None) -> str:
    """Load constitution text from disk."""
    if constitution_path is None:
        constitution_path = Path(__file__).resolve().parents[4] / "data" / "constitution.txt"
    else:
        constitution_path = Path(constitution_path).expanduser()

    if constitution_path.exists():
        return constitution_path.read_text()

    return "Follow safety, truthfulness, and helpfulness principles."


TEACHER_PROMPT_TEMPLATE = """You are an AI assistant designed to generate high-quality, aligned responses.

<CONSTITUTION>
{constitution}
</CONSTITUTION>

Your task is to generate an answer that is as aligned as possible with the constitution above.
The response should be helpful, honest, and safe.

User Question:
{question}

Generate an answer:"""


def build_teacher_prompt(
    question: str,
    constitution: Optional[str] = None,
    feedback: Optional[str] = None,
    previous_attempt: Optional[str] = None
) -> str:
    """
    Build a teacher prompt with constitutional context for SDPO distillation.
    
    Args:
        question: User's question
        constitution: Constitution text (loaded from file if not provided)
        feedback: Optional feedback on previous attempt (e.g., test errors)
        previous_attempt: Optional previous response to improve upon
    
    Returns:
        Full prompt for teacher model
    """
    if constitution is None:
        constitution = load_constitution()
    
    prompt = TEACHER_PROMPT_TEMPLATE.format(
        constitution=constitution,
        question=question
    )
    
    # Optionally add context about previous attempt
    if previous_attempt is not None:
        prompt += f"\n\nPrevious attempt (to improve on):\n{previous_attempt}"
    
    if feedback is not None:
        prompt += f"\n\nFeedback to address:\n{feedback}"
    
    return prompt


def build_student_prompt_with_teacher_context(
    question: str,
    teacher_response: Optional[str] = None,
    feedback: Optional[str] = None
) -> str:
    """
    Build a student prompt that includes teacher-generated demonstration.
    Used for the reprompting/demonstration phase of SDPO.
    
    Args:
        question: Original user question
        teacher_response: High-quality response from teacher
        feedback: Optional feedback on why this is good
    
    Returns:
        Prompt for student to learn from teacher
    """
    prompt = f"Question:\n{question}\n\n"
    
    if teacher_response is not None:
        prompt += f"High-quality answer (aligned with constitution):\n{teacher_response}\n\n"
    
    if feedback is not None:
        prompt += f"Why this is good:\n{feedback}\n\n"
    
    prompt += "Provide another response that is similarly aligned and helpful:"
    
    return prompt
