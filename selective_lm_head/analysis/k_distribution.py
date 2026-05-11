"""K distribution aggregation."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

from selective_lm_head.grammar.bitmask import bucket_k
from selective_lm_head.tracing.writer import read_jsonl

BUCKETS = ["K=1", "K<=8", "K<=32", "K<=128", "K<=512", "K<=2048", "K<=8192", "K>8192"]


def summarize_k_distribution(trace_path: str | Path) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str, str], Counter[str]] = defaultdict(Counter)
    fallbacks: Counter[tuple[str, str, str, str, str]] = Counter()
    totals: Counter[tuple[str, str, str, str, str]] = Counter()
    for request in read_jsonl(trace_path):
        key = (
            str(request.get("benchmark")),
            str(request.get("category")),
            str(request.get("model")),
            str(request.get("device")),
            str(request.get("head")),
        )
        for step in request.get("steps", []):
            bucket = bucket_k(int(step.get("K", 0)))
            grouped[key][bucket] += 1
            totals[key] += 1
            if step.get("fallback"):
                fallbacks[key] += 1

    rows = []
    for key, counter in sorted(grouped.items()):
        total = totals[key]
        row: dict[str, object] = {
            "benchmark": key[0],
            "category": key[1],
            "model": key[2],
            "device": key[3],
            "head": key[4],
            "total_steps": total,
            "fallback_rate": fallbacks[key] / total if total else 0.0,
        }
        for bucket in BUCKETS:
            row[bucket] = counter[bucket]
            row[bucket + "_rate"] = counter[bucket] / total if total else 0.0
        rows.append(row)
    return rows


def write_csv(rows: list[dict[str, object]], output: str | Path) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else [
        "benchmark",
        "category",
        "model",
        "device",
        "head",
        "total_steps",
        "fallback_rate",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

