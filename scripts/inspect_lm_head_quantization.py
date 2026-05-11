#!/usr/bin/env python
"""Inspect whether checkpoint LM-head tensors appear quantized."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from safetensors import safe_open


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def _read_config(model_dir: Path) -> dict[str, Any]:
    config_path = model_dir / "config.json"
    return json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}


def _safetensor_files(model_dir: Path) -> list[Path]:
    return sorted(model_dir.glob("*.safetensors"))


def inspect_model(model_path: str | Path) -> dict[str, Any]:
    model_dir = Path(model_path)
    config = _read_config(model_dir)
    tensors: dict[str, dict[str, Any]] = {}
    for file_path in _safetensor_files(model_dir):
        with safe_open(file_path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if (
                    key.startswith("lm_head")
                    or key.startswith("model.embed_tokens")
                    or key.startswith("transformer.wte")
                ):
                    tensor = handle.get_tensor(key)
                    tensors[key] = {
                        "file": file_path.name,
                        "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype).replace("torch.", ""),
                    }

    lm_quant_keys = [
        key
        for key in tensors
        if key.startswith("lm_head.")
        and any(part in key for part in ["qweight", "qzeros", "scales", "g_idx"])
    ]
    lm_float_keys = [
        key
        for key in tensors
        if key.startswith("lm_head.") and key.endswith((".weight", ".bias"))
    ]
    embed_keys = [key for key in tensors if key.startswith("model.embed_tokens")]
    if lm_quant_keys:
        lm_head_status = "quantized"
    elif lm_float_keys:
        lm_head_status = "float"
    elif config.get("tie_word_embeddings") and embed_keys:
        lm_head_status = "tied_to_embedding_float_or_unpacked"
    else:
        lm_head_status = "not_found"

    return {
        "model_path": str(model_dir),
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures"),
        "vocab_size": config.get("vocab_size"),
        "hidden_size": config.get("hidden_size"),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "num_experts": config.get("num_experts"),
        "num_experts_per_tok": config.get("num_experts_per_tok"),
        "tie_word_embeddings": config.get("tie_word_embeddings"),
        "quantization_config": config.get("quantization_config"),
        "lm_head_status": lm_head_status,
        "lm_quant_keys": lm_quant_keys[:20],
        "lm_float_keys": lm_float_keys[:20],
        "embed_keys": embed_keys[:20],
        "tensors": tensors,
    }


def main() -> None:
    args = parse_args()
    rows = [inspect_model(path) for path in args.models]
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    for row in rows:
        print(
            f"{row['model_path']}: lm_head_status={row['lm_head_status']} "
            f"tie_word_embeddings={row['tie_word_embeddings']} "
            f"quant_method={(row['quantization_config'] or {}).get('quant_method')}"
        )


if __name__ == "__main__":
    main()
