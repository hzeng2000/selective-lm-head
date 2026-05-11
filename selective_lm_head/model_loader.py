"""Transformers model/tokenizer loader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from selective_lm_head.config import ModelConfig, resolve_model
from selective_lm_head.dependencies import require_module


@dataclass
class ModelBundle:
    config: ModelConfig
    tokenizer: Any
    model: Any
    model_path: Path
    device: str
    dtype: str


def torch_dtype_from_name(name: str | None) -> Any:
    torch = require_module(
        "torch",
        "Run `bash scripts/setup_env.sh ml` to install PyTorch.",
    )
    if name is None or name == "auto":
        return "auto"
    normalized = name.lower()
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[normalized]


def load_model_bundle(
    model: str,
    device: str = "cuda",
    dtype: str | None = None,
    trust_remote_code: bool = True,
    local_files_only: bool = True,
) -> ModelBundle:
    transformers = require_module(
        "transformers",
        "Run `bash scripts/setup_env.sh ml` to install Transformers.",
    )
    torch = require_module("torch", "Run `bash scripts/setup_env.sh ml` to install PyTorch.")
    cfg = resolve_model(model)
    model_path = cfg.local_path
    requested_dtype = dtype or cfg.recommended_dtype
    torch_dtype = torch_dtype_from_name(requested_dtype)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    loaded_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        device_map=None,
    )
    loaded_model.eval()
    loaded_model.to(device)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return ModelBundle(
        config=cfg,
        tokenizer=tokenizer,
        model=loaded_model,
        model_path=model_path,
        device=device,
        dtype=requested_dtype,
    )

