"""Paired request-level speedup aggregation."""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import median

from selective_lm_head.tracing.writer import read_jsonl


def summarize_paired_speedup(baseline_path: str | Path, selective_path: str | Path) -> list[dict[str, object]]:
    baseline = {row["request_id"]: row for row in read_jsonl(baseline_path)}
    selective = {row["request_id"]: row for row in read_jsonl(selective_path)}
    rows = []
    for request_id in sorted(set(baseline) & set(selective)):
        base = baseline[request_id]
        opt = selective[request_id]
        base_latency = float(base.get("latency_ms_total") or 0.0)
        opt_latency = float(opt.get("latency_ms_total") or 0.0)
        rows.append(
            {
                "request_id": request_id,
                "benchmark": base.get("benchmark"),
                "category": base.get("category"),
                "model": base.get("model"),
                "device": base.get("device"),
                "baseline_latency_ms": base_latency,
                "selective_latency_ms": opt_latency,
                "speedup": base_latency / opt_latency if opt_latency > 0 else 0.0,
                "baseline_output_tokens": base.get("output_tokens"),
                "selective_output_tokens": opt.get("output_tokens"),
                "same_output": base.get("output_text") == opt.get("output_text"),
                "baseline_valid": base.get("valid"),
                "selective_valid": opt.get("valid"),
            }
        )
    if rows:
        rows.append(
            {
                "request_id": "__summary__",
                "benchmark": "",
                "category": "",
                "model": "",
                "device": "",
                "baseline_latency_ms": median(float(row["baseline_latency_ms"]) for row in rows),
                "selective_latency_ms": median(float(row["selective_latency_ms"]) for row in rows),
                "speedup": median(float(row["speedup"]) for row in rows),
                "baseline_output_tokens": "",
                "selective_output_tokens": "",
                "same_output": all(bool(row["same_output"]) for row in rows),
                "baseline_valid": "",
                "selective_valid": "",
            }
        )
    return rows


def write_csv(rows: list[dict[str, object]], output: str | Path) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["request_id", "speedup"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

