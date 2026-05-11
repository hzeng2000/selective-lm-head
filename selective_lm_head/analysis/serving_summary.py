"""Serving baseline aggregation."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from selective_lm_head.tracing.writer import read_jsonl


def _median(values: list[float]) -> float:
    return float(median(values)) if values else 0.0


def summarize_serving_trace(trace_path: str | Path) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str, str], dict[str, list[float] | int]] = defaultdict(
        lambda: {
            "requests": 0,
            "valid": 0,
            "latency_ms": [],
            "output_tokens": [],
            "prompt_tokens": [],
            "ms_per_output_token": [],
        }
    )
    for request in read_jsonl(trace_path):
        key = (
            str(request.get("benchmark")),
            str(request.get("category")),
            str(request.get("model")),
            str(request.get("framework")),
            str(request.get("device")),
            str(request.get("head")),
        )
        bucket = grouped[key]
        bucket["requests"] = int(bucket["requests"]) + 1
        if request.get("valid"):
            bucket["valid"] = int(bucket["valid"]) + 1
        latency_ms = float(request.get("latency_ms_total") or 0.0)
        output_tokens = float(request.get("output_tokens") or 0.0)
        prompt_tokens = float(request.get("prompt_tokens") or 0.0)
        ms_per_output_token = float(request.get("latency_ms_per_decode_token") or 0.0)
        bucket["latency_ms"].append(latency_ms)  # type: ignore[union-attr]
        bucket["output_tokens"].append(output_tokens)  # type: ignore[union-attr]
        bucket["prompt_tokens"].append(prompt_tokens)  # type: ignore[union-attr]
        if ms_per_output_token:
            bucket["ms_per_output_token"].append(ms_per_output_token)  # type: ignore[union-attr]

    rows = []
    for key, values in sorted(grouped.items()):
        latencies = values["latency_ms"]
        output_tokens = values["output_tokens"]
        prompt_tokens = values["prompt_tokens"]
        per_token = values["ms_per_output_token"]
        assert isinstance(latencies, list)
        assert isinstance(output_tokens, list)
        assert isinstance(prompt_tokens, list)
        assert isinstance(per_token, list)
        rows.append(
            {
                "benchmark": key[0],
                "category": key[1],
                "model": key[2],
                "framework": key[3],
                "device": key[4],
                "head": key[5],
                "requests": int(values["requests"]),
                "valid": int(values["valid"]),
                "median_latency_ms": _median(latencies),
                "min_latency_ms": min(latencies) if latencies else 0.0,
                "max_latency_ms": max(latencies) if latencies else 0.0,
                "median_prompt_tokens": _median(prompt_tokens),
                "median_output_tokens": _median(output_tokens),
                "total_output_tokens": int(sum(output_tokens)),
                "median_ms_per_output_token": _median(per_token),
            }
        )
    return rows


def write_csv(rows: list[dict[str, Any]], output: str | Path) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["benchmark", "category", "model", "framework"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
