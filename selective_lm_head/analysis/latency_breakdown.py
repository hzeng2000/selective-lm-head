"""Latency breakdown aggregation."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from selective_lm_head.tracing.writer import read_jsonl


def summarize_latency_breakdown(trace_path: str | Path) -> list[dict[str, object]]:
    sums: dict[tuple[str, str, str, str, str], dict[str, float]] = defaultdict(
        lambda: {
            "transformer_ms": 0.0,
            "lm_head_ms": 0.0,
            "mask_or_allowed_ms": 0.0,
            "bitmask_to_list_ms": 0.0,
            "sampler_ms": 0.0,
            "decode_ms": 0.0,
            "requests": 0.0,
            "steps": 0.0,
        }
    )
    for request in read_jsonl(trace_path):
        key = (
            str(request.get("benchmark")),
            str(request.get("category")),
            str(request.get("model")),
            str(request.get("device")),
            str(request.get("head")),
        )
        bucket = sums[key]
        bucket["requests"] += 1
        bucket["decode_ms"] += float(request.get("latency_ms_decode") or 0.0)
        for step in request.get("steps", []):
            bucket["steps"] += 1
            for name in [
                "transformer_ms",
                "lm_head_ms",
                "mask_or_allowed_ms",
                "bitmask_to_list_ms",
                "sampler_ms",
            ]:
                bucket[name] += float(step.get(name) or 0.0)

    rows = []
    for key, values in sorted(sums.items()):
        measured = (
            values["transformer_ms"]
            + values["lm_head_ms"]
            + values["mask_or_allowed_ms"]
            + values["sampler_ms"]
        )
        denom = measured if measured > 0 else 1.0
        rows.append(
            {
                "benchmark": key[0],
                "category": key[1],
                "model": key[2],
                "device": key[3],
                "head": key[4],
                "requests": int(values["requests"]),
                "steps": int(values["steps"]),
                "decode_ms": values["decode_ms"],
                "transformer_ms": values["transformer_ms"],
                "lm_head_ms": values["lm_head_ms"],
                "grammar_list_ms": values["mask_or_allowed_ms"],
                "bitmask_to_list_ms": values["bitmask_to_list_ms"],
                "sampler_ms": values["sampler_ms"],
                "transformer_pct": values["transformer_ms"] / denom,
                "lm_head_pct": values["lm_head_ms"] / denom,
                "grammar_list_pct": values["mask_or_allowed_ms"] / denom,
                "sampler_pct": values["sampler_ms"] / denom,
            }
        )
    return rows


def write_csv(rows: list[dict[str, object]], output: str | Path) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["benchmark", "category", "model", "device", "head"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

