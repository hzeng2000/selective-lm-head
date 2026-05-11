"""JSONL trace writer and reader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

from selective_lm_head.tracing.trace_schema import RequestTrace


class JsonlTraceWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")

    def write(self, trace: RequestTrace | dict) -> None:
        data = trace.to_dict() if isinstance(trace, RequestTrace) else trace
        self._handle.write(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "JsonlTraceWriter":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

