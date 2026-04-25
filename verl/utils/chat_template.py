# Copyright 2025 Bytedance Ltd. and/or its affiliates
import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def has_chat_template(tokenizer) -> bool:
    return getattr(tokenizer, "chat_template", None) is not None


def _stringify_message_content(content) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif "text" in item:
                    parts.append(item.get("text", ""))
            else:
                parts.append(str(item))
        return "".join(parts)

    return str(content)


def render_plain_prompt(messages, add_generation_prompt: bool = True) -> str:
    parts = []
    for message in messages:
        role = str(message.get("role", "user")).strip().lower()
        content = _stringify_message_content(message.get("content", ""))
        if role == "system":
            parts.append(f"System: {content}".strip())
        elif role == "assistant":
            parts.append(f"Assistant: {content}".strip())
        else:
            parts.append(f"User: {content}".strip())

    if add_generation_prompt:
        parts.append("Assistant:")

    return "\n\n".join(parts)


def initialize_system_prompt(tokenizer, **apply_chat_template_kwargs) -> list[int]:
    """
    Initialize system prompt tokens for chat templates that support them.

    Args:
        tokenizer: The tokenizer with a chat template
        **apply_chat_template_kwargs: Additional arguments for apply_chat_template

    Returns:
        List of token IDs for the system prompt, or empty list if not supported
    """
    if not has_chat_template(tokenizer):
        return []

    token1 = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}], add_generation_prompt=False, tokenize=True
    )
    token2 = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}] * 2, add_generation_prompt=False, tokenize=True
    )
    # get system prompt tokens
    system_prompt = token1[: -(len(token2) - len(token1))]
    return system_prompt


def extract_system_prompt_and_generation(tokenizer):
    if not has_chat_template(tokenizer):
        return [], []

    token1 = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}], add_generation_prompt=False, tokenize=True
    )
    token2 = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}] * 2, add_generation_prompt=False, tokenize=True
    )
    # get system prompt tokens
    system_prompt = token1[: -(len(token2) - len(token1))]
    # get generate prompt tokens
    token3 = tokenizer.apply_chat_template([{"role": "user", "content": ""}], add_generation_prompt=True, tokenize=True)
    generate_prompt = token3[len(token1) :]

    return system_prompt, generate_prompt
