"""Model inference interfaces and local runners for the OPSD alignment diagnostic.

The runner interface is intentionally narrow: later stages need exact token IDs,
next-token log probabilities, and continuations from already-tokenized prefixes.
This keeps the diagnostic portable across HF, vLLM, and SGLang backends.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence


@dataclass(frozen=True)
class Generation:
    text: str
    token_ids: list[int]
    seed: int


@dataclass(frozen=True)
class NextTokenDistribution:
    token_ids: list[int]
    logprobs: list[float]


class LanguageModelRunner(Protocol):
    checkpoint: str

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ...

    def decode(self, token_ids: Sequence[int]) -> str:
        ...

    def generate(self, prompt: str, *, seed: int, max_new_tokens: int, temperature: float, top_p: float) -> Generation:
        ...

    def continue_from_tokens(
        self,
        prefix_token_ids: Sequence[int],
        *,
        seed: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> Generation:
        ...

    def next_token_logprobs(self, prefix_token_ids: Sequence[int], candidate_token_ids: Sequence[int]) -> list[float]:
        ...

    def topk_next_token_distribution(self, prefix_token_ids: Sequence[int], k: int) -> NextTokenDistribution:
        ...


class HuggingFaceModelRunner:
    """Simple Transformers-backed runner for local smoke tests and diagnostics."""

    checkpoint: str

    def __init__(
        self,
        checkpoint: str,
        model_path: str | None = None,
        *,
        device: str = "auto",
        torch_dtype: str = "auto",
        trust_remote_code: bool = True,
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers and torch are required for HuggingFaceModelRunner") from exc

        self.torch = torch
        self.checkpoint = checkpoint
        self.model_path = model_path or checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {"trust_remote_code": trust_remote_code}
        if torch_dtype != "auto":
            model_kwargs["torch_dtype"] = _resolve_torch_dtype(torch, torch_dtype)
        else:
            model_kwargs["torch_dtype"] = "auto"

        if device == "auto":
            model_kwargs["device_map"] = "auto"
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **model_kwargs)
            self.input_device = self._infer_input_device()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **model_kwargs)
            self.input_device = torch.device(device)
            self.model.to(self.input_device)

        self.model.eval()

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, token_ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(token_ids), skip_special_tokens=True)

    def generate(self, prompt: str, *, seed: int, max_new_tokens: int, temperature: float, top_p: float) -> Generation:
        prefix_token_ids = self.encode(prompt, add_special_tokens=True)
        return self.continue_from_tokens(
            prefix_token_ids,
            seed=seed,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    def continue_from_tokens(
        self,
        prefix_token_ids: Sequence[int],
        *,
        seed: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> Generation:
        torch = self.torch
        input_ids = torch.tensor([list(prefix_token_ids)], dtype=torch.long, device=self.input_device)
        attention_mask = torch.ones_like(input_ids)
        generator = torch.Generator(device=self.input_device)
        generator.manual_seed(int(seed))
        do_sample = temperature > 0

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                generator=generator,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        new_token_ids = output_ids[0, input_ids.shape[1] :].detach().cpu().tolist()
        return Generation(text=self.decode(new_token_ids), token_ids=new_token_ids, seed=seed)

    def next_token_logprobs(self, prefix_token_ids: Sequence[int], candidate_token_ids: Sequence[int]) -> list[float]:
        if not candidate_token_ids:
            return []
        logprobs = self._next_token_logprobs_tensor(prefix_token_ids)
        candidate_ids = self.torch.tensor(list(candidate_token_ids), dtype=self.torch.long, device=logprobs.device)
        return logprobs.index_select(0, candidate_ids).detach().cpu().tolist()

    def topk_next_token_distribution(self, prefix_token_ids: Sequence[int], k: int) -> NextTokenDistribution:
        if k <= 0:
            raise ValueError("k must be positive")
        logprobs = self._next_token_logprobs_tensor(prefix_token_ids)
        values, indices = self.torch.topk(logprobs, k=k)
        return NextTokenDistribution(
            token_ids=indices.detach().cpu().tolist(),
            logprobs=values.detach().cpu().tolist(),
        )

    def _next_token_logprobs_tensor(self, prefix_token_ids: Sequence[int]):
        torch = self.torch
        input_ids = torch.tensor([list(prefix_token_ids)], dtype=torch.long, device=self.input_device)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits[0, -1, :]
        return torch.log_softmax(logits.float(), dim=-1)

    def _infer_input_device(self):
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return self.torch.device("cpu")


def build_model_runner(model_config: dict, *, device: str = "auto", torch_dtype: str = "auto") -> LanguageModelRunner:
    backend = model_config.get("backend", "hf")
    if backend != "hf":
        raise ValueError(f"Unsupported model backend for now: {backend!r}")
    model_path = resolve_model_path(model_config)
    return HuggingFaceModelRunner(
        checkpoint=model_config["name"],
        model_path=model_path,
        device=device,
        torch_dtype=torch_dtype,
        trust_remote_code=bool(model_config.get("trust_remote_code", True)),
    )


def resolve_model_path(model_config: dict) -> str:
    raw_path = model_config.get("path") or model_config.get("model_path") or model_config["name"]
    expanded = os.path.expandvars(os.path.expanduser(str(raw_path)))
    if "$" in expanded or expanded == "":
        raise ValueError(
            f"Model path for {model_config.get('name', '<unnamed>')!r} is not resolved: {raw_path!r}. "
            "Set the referenced environment variable or use a config with a concrete path."
        )
    if raw_path != model_config["name"] and not _looks_like_remote_model_id(expanded) and not Path(expanded).exists():
        raise FileNotFoundError(
            f"Model path for {model_config.get('name', '<unnamed>')!r} does not exist: {expanded}"
        )
    return expanded


def _looks_like_remote_model_id(path: str) -> bool:
    return not path.startswith(("/", ".")) and "/" in path


def _resolve_torch_dtype(torch, dtype_name: str):
    dtype_name = dtype_name.lower()
    aliases = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if dtype_name not in aliases:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return aliases[dtype_name]
