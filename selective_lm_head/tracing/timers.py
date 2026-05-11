"""Wall-clock and CUDA timing helpers."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator


def perf_counter_ms() -> float:
    return time.perf_counter_ns() / 1_000_000


def synchronize_if_needed(device: str | None = None) -> None:
    if not device or not str(device).startswith("cuda"):
        return
    try:
        import torch
    except ModuleNotFoundError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@dataclass
class Stopwatch:
    device: str | None = None
    sync: bool = False
    _start_ms: float = 0.0

    def start(self) -> None:
        if self.sync:
            synchronize_if_needed(self.device)
        self._start_ms = perf_counter_ms()

    def stop(self) -> float:
        if self.sync:
            synchronize_if_needed(self.device)
        return perf_counter_ms() - self._start_ms


@contextmanager
def timed(device: str | None = None, sync: bool = False) -> Iterator[dict[str, float]]:
    sw = Stopwatch(device=device, sync=sync)
    bucket: dict[str, float] = {}
    sw.start()
    try:
        yield bucket
    finally:
        bucket["ms"] = sw.stop()

