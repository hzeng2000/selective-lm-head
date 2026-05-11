"""Environment metadata recorded into traces."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from typing import Any

from selective_lm_head.dependencies import module_available


def _run(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def collect_environment() -> dict[str, Any]:
    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
    }

    if module_available("torch"):
        import torch

        env["torch_version"] = torch.__version__
        env["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            env["cuda_version"] = torch.version.cuda
            env["gpu_count"] = torch.cuda.device_count()
            env["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]

    for module_name in ["transformers", "xgrammar", "vllm", "sglang", "jsonschema", "pandas"]:
        if module_available(module_name):
            module = __import__(module_name)
            env[f"{module_name}_version"] = getattr(module, "__version__", "unknown")

    nvidia_smi = _run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]
    )
    if nvidia_smi:
        env["nvidia_smi"] = nvidia_smi.splitlines()

    return env

