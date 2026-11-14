"""Inference & AST syntax validation engine for MiniMind Copilot Studio."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import time
from typing import Any, Callable
import warnings

import torch


@dataclass
class CheckpointSpec:
    """Metadata and callable inference hooks for a registered model stage."""

    key: str
    label: str
    stage: str
    model: Any
    supports_sft_template: bool = False


class CopilotInferenceEngine:
    """Manages multiple MiniMind checkpoints and runs L2R, FIM, and SFT code completion."""

    def __init__(
        self,
        tokenizer: Any,
        complete_fn: Callable[..., str],
        fim_complete_fn: Callable[..., str],
        sft_complete_fn: Callable[..., str],
    ) -> None:
        self.tokenizer = tokenizer
        self._complete_fn = complete_fn
        self._fim_complete_fn = fim_complete_fn
        self._sft_complete_fn = sft_complete_fn
        self.checkpoints: list[CheckpointSpec] = []

    def register_checkpoint(
        self,
        key: str,
        label: str,
        stage: str,
        model: Any,
        supports_sft_template: bool = False,
    ) -> None:
        self.checkpoints.append(
            CheckpointSpec(
                key=key,
                label=label,
                stage=stage,
                model=model,
                supports_sft_template=supports_sft_template,
            )
        )

    @staticmethod
    def is_valid_python(code: str) -> bool:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                ast.parse(code)
            return True
        except SyntaxError:
            return False

    @torch.no_grad()
    def complete_all(
        self,
        prefix: str,
        mode: str = "sft",
        suffix: str = "",
        max_new_tokens: int = 64,
    ) -> dict[str, Any]:
        """Run code completion across all registered checkpoints and return structured metrics."""
        models_out: list[dict[str, Any]] = []
        for spec in self.checkpoints:
            t0 = time.perf_counter()
            if mode == "fim":
                text = self._fim_complete_fn(
                    spec.model, prefix, suffix, max_new_tokens=min(max_new_tokens, 48)
                )
                candidate = prefix + text + suffix
            elif mode == "sft" and spec.supports_sft_template:
                text = self._sft_complete_fn(
                    spec.model, prefix, max_new_tokens=max_new_tokens
                )
                sep = "" if (prefix.endswith("\n") or text.startswith("\n")) else "\n"
                candidate = prefix + sep + text
            else:
                text = self._complete_fn(
                    spec.model, prefix, max_new_tokens=max_new_tokens
                )
                sep = "" if (prefix.endswith("\n") or text.startswith("\n")) else "\n"
                candidate = prefix + sep + text

            latency_ms = int((time.perf_counter() - t0) * 1000)
            n_tokens = len(
                self.tokenizer(text, add_special_tokens=False).input_ids
            )
            models_out.append(
                {
                    "key": spec.key,
                    "label": spec.label,
                    "stage": spec.stage,
                    "text": text.rstrip("\n"),
                    "valid_syntax": self.is_valid_python(candidate),
                    "latency_ms": latency_ms,
                    "tokens_generated": n_tokens,
                }
            )
        return {"mode": mode, "models": models_out}
