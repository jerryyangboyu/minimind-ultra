"""Inference & AST syntax validation engine for MiniMind Copilot Studio."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import os
import sys
import time
from typing import Any, Callable
import warnings

import torch


FIM_PREFIX = "<|buffer1|>"
FIM_SUFFIX = "<|buffer2|>"
FIM_MIDDLE = "<|buffer3|>"


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
        complete_fn: Callable[..., str] | None = None,
        fim_complete_fn: Callable[..., str] | None = None,
        sft_complete_fn: Callable[..., str] | None = None,
        device: str | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.bos_id = tokenizer.bos_token_id
        self.eos_id = tokenizer.eos_token_id
        self.fim_ids = tokenizer.convert_tokens_to_ids(
            [FIM_PREFIX, FIM_SUFFIX, FIM_MIDDLE]
        )
        self._complete_fn = complete_fn or self.default_complete
        self._fim_complete_fn = fim_complete_fn or self.default_fim_complete
        self._sft_complete_fn = sft_complete_fn or self.default_sft_complete
        self.checkpoints: list[CheckpointSpec] = []

    @classmethod
    def from_pretrained(
        cls,
        minimind_dir: str = "/content/minimind",
        hf_repo_id: str = "boyuyang/minimind-course-weights",
        hidden_size: int = 768,
        num_hidden_layers: int = 8,
        device: str | None = None,
        reuse_globals: dict[str, Any] | None = None,
    ) -> "CopilotInferenceEngine":
        """Standalone factory: downloads any missing checkpoints from HF Hub and registers all 4 stages."""
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        if minimind_dir not in sys.path:
            sys.path.insert(0, minimind_dir)

        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import disable_progress_bars
        from transformers import AutoTokenizer
        from model.model_lora import apply_lora, load_lora
        from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

        disable_progress_bars()
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            try:
                from google.colab import userdata  # type: ignore
                hf_token = userdata.get("HF_TOKEN")
                if hf_token:
                    os.environ["HF_TOKEN"] = hf_token
            except Exception:
                hf_token = None

        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        out_dir = os.path.join(minimind_dir, "out")
        os.makedirs(out_dir, exist_ok=True)

        required_files = [
            f"pretrain_{hidden_size}.pth",
            f"code_pretrain_{hidden_size}.pth",
            f"code_sft_{hidden_size}.pth",
            f"lora_code_{hidden_size}.pth",
        ]
        for fname in required_files:
            fpath = os.path.join(out_dir, fname)
            if not os.path.exists(fpath):
                print(f"⬇️ Downloading {fname} from {hf_repo_id} ...")
                hf_hub_download(
                    repo_id=hf_repo_id,
                    repo_type="model",
                    filename=fname,
                    local_dir=out_dir,
                    token=hf_token,
                )

        tokenizer = AutoTokenizer.from_pretrained(os.path.join(minimind_dir, "model"))
        engine = cls(tokenizer=tokenizer, device=dev)

        def _load_ckpt(stem: str) -> Any:
            cfg = MiniMindConfig(
                hidden_size=hidden_size, num_hidden_layers=num_hidden_layers
            )
            m = MiniMindForCausalLM(cfg).to(dev)
            state = torch.load(
                os.path.join(out_dir, f"{stem}_{hidden_size}.pth"), map_location=dev
            )
            m.load_state_dict(state, strict=False)
            return m.eval()

        g = reuse_globals or {}
        base_m = g.get("base_model") or _load_ckpt("pretrain")
        code_m = g.get("code_base_model") or _load_ckpt("code_pretrain")
        sft_m = g.get("code_sft_model") or _load_ckpt("code_sft")
        lora_m = g.get("lora_code_model")
        if lora_m is None:
            lora_m = _load_ckpt("pretrain")
            apply_lora(lora_m, rank=16)
            load_lora(lora_m, os.path.join(out_dir, f"lora_code_{hidden_size}.pth"))
            lora_m.eval()

        engine.register_checkpoint(
            "base",
            f"① General Base (pretrain_{hidden_size})",
            "Pretrain",
            base_m,
            supports_sft_template=False,
        )
        engine.register_checkpoint(
            "code",
            f"② Code Base (code_pretrain_{hidden_size})",
            "Code DAPT",
            code_m,
            supports_sft_template=False,
        )
        engine.register_checkpoint(
            "sft",
            f"③ Code SFT (code_sft_{hidden_size}, Full FT)",
            "Full SFT",
            sft_m,
            supports_sft_template=True,
        )
        engine.register_checkpoint(
            "lora",
            f"④ LoRA Adapter (lora_code_{hidden_size})",
            "LoRA (0.6%)",
            lora_m,
            supports_sft_template=True,
        )
        return engine

    @torch.no_grad()
    def _generate_raw(
        self,
        model: Any,
        ids: list[int],
        max_new_tokens: int,
        repetition_penalty: float = 1.12,
    ) -> str:
        x = torch.tensor([ids], device=self.device)
        saved = model.lm_head.weight.data[self.fim_ids].clone()
        model.lm_head.weight.data[self.fim_ids] = 0.0
        try:
            out = model.generate(
                x,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=repetition_penalty,
                eos_token_id=self.eos_id,
            )[0, len(ids) :].tolist()
        finally:
            model.lm_head.weight.data[self.fim_ids] = saved
        if self.eos_id in out:
            out = out[: out.index(self.eos_id)]
        return self.tokenizer.decode(out)

    def default_complete(
        self, model: Any, prefix: str, max_new_tokens: int = 64
    ) -> str:
        ids = [self.bos_id] + self.tokenizer(
            prefix, add_special_tokens=False
        ).input_ids
        return self._generate_raw(model, ids, max_new_tokens=max_new_tokens)

    def default_fim_complete(
        self, model: Any, prefix: str, suffix: str, max_new_tokens: int = 48
    ) -> str:
        prompt = f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}"
        ids = [self.bos_id] + self.tokenizer(
            prompt, add_special_tokens=False
        ).input_ids
        return self._generate_raw(model, ids, max_new_tokens=max_new_tokens)

    def default_sft_complete(
        self, model: Any, header: str, max_new_tokens: int = 96
    ) -> str:
        prompt = f"<|im_start|>user\n{header}<|im_end|>\n<|im_start|>assistant\n"
        ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        text = self._generate_raw(model, ids, max_new_tokens=max_new_tokens)
        if "</think>" in text:
            text = text.split("</think>", 1)[1].lstrip("\n")
        if text and not text[0].isspace():
            text = "    " + text
        return text

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
