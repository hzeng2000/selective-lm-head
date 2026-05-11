#!/usr/bin/env python
"""Download the model matrix used by the profiling experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_MODEL_IDS = [
    "Qwen/Qwen2.5-0.5B-Instruct-AWQ",
    "Qwen/Qwen2.5-1.5B-Instruct-AWQ",
    "Qwen/Qwen2.5-3B-Instruct-AWQ",
    "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4",
    "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4",
    "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4",
    "Qwen/Qwen1.5-MoE-A2.7B-Chat",
    "Qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4",
]

OPTIONAL_LARGE_MODEL_IDS = [
    "Qwen/Qwen3-30B-A3B-GPTQ-Int4",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/hzeng/models")
    parser.add_argument("--include-large-qwen3", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    model_ids = list(DEFAULT_MODEL_IDS)
    if args.include_large_qwen3:
        model_ids.extend(OPTIONAL_LARGE_MODEL_IDS)
    for model_id in model_ids:
        local_dir = root / model_id
        print(f"downloading {model_id} -> {local_dir}", flush=True)
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print(f"done {model_id}", flush=True)


if __name__ == "__main__":
    main()
