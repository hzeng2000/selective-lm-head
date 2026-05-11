"""Aggregate vLLM LM-head profiling JSONL records."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from selective_lm_head.tracing.writer import read_jsonl


def _median(values: list[float]) -> float:
    return float(median(values)) if values else 0.0


def _sum(values: list[float]) -> float:
    return float(sum(values)) if values else 0.0


def summarize_vllm_lm_head_profile(profile_path: str | Path) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "steps": 0,
            "num_reqs": 0,
            "scheduled_tokens": 0,
            "sample_rows": 0,
            "logits_rows": 0,
            "num_reqs_per_step": [],
            "scheduled_tokens_per_step": [],
            "forward_ms": [],
            "logits_ms": [],
            "grammar_ms": [],
            "sample_ms": [],
            "bookkeeping_ms": [],
            "core_ms": [],
            "step_ms": [],
            "lm_head_share_model_logits": [],
            "lm_head_share_core": [],
            "lm_head_share_step": [],
        }
    )

    for record in read_jsonl(profile_path):
        phase = str(record.get("phase") or "unknown")
        bucket = grouped[phase]
        bucket["steps"] += 1
        bucket["num_reqs"] += int(record.get("num_reqs") or 0)
        bucket["scheduled_tokens"] += int(record.get("num_scheduled_tokens") or 0)
        bucket["sample_rows"] += int(record.get("sample_rows") or 0)
        bucket["logits_rows"] += int(record.get("logits_rows") or 0)
        bucket["num_reqs_per_step"].append(float(record.get("num_reqs") or 0))
        bucket["scheduled_tokens_per_step"].append(
            float(record.get("num_scheduled_tokens") or 0)
        )
        for key in [
            "forward_ms",
            "logits_ms",
            "grammar_ms",
            "sample_ms",
            "bookkeeping_ms",
            "core_ms",
            "step_ms",
            "lm_head_share_model_logits",
            "lm_head_share_core",
            "lm_head_share_step",
        ]:
            bucket[key].append(float(record.get(key) or 0.0))

    rows: list[dict[str, Any]] = []
    for phase, values in sorted(grouped.items()):
        forward_sum = _sum(values["forward_ms"])
        logits_sum = _sum(values["logits_ms"])
        grammar_sum = _sum(values["grammar_ms"])
        sample_sum = _sum(values["sample_ms"])
        bookkeeping_sum = _sum(values["bookkeeping_ms"])
        core_sum = _sum(values["core_ms"])
        step_sum = _sum(values["step_ms"])
        model_logits_sum = forward_sum + logits_sum
        rows.append(
            {
                "phase": phase,
                "steps": values["steps"],
                "num_reqs": values["num_reqs"],
                "scheduled_tokens": values["scheduled_tokens"],
                "sample_rows": values["sample_rows"],
                "logits_rows": values["logits_rows"],
                "avg_reqs_per_step": (
                    values["num_reqs"] / values["steps"] if values["steps"] else 0.0
                ),
                "median_reqs_per_step": _median(values["num_reqs_per_step"]),
                "max_reqs_per_step": (
                    max(values["num_reqs_per_step"])
                    if values["num_reqs_per_step"]
                    else 0.0
                ),
                "avg_scheduled_tokens_per_step": (
                    values["scheduled_tokens"] / values["steps"]
                    if values["steps"]
                    else 0.0
                ),
                "median_scheduled_tokens_per_step": _median(
                    values["scheduled_tokens_per_step"]
                ),
                "sum_forward_ms": forward_sum,
                "sum_logits_ms": logits_sum,
                "sum_grammar_ms": grammar_sum,
                "sum_sample_ms": sample_sum,
                "sum_bookkeeping_ms": bookkeeping_sum,
                "sum_core_ms": core_sum,
                "sum_step_ms": step_sum,
                "median_forward_ms": _median(values["forward_ms"]),
                "median_logits_ms": _median(values["logits_ms"]),
                "median_grammar_ms": _median(values["grammar_ms"]),
                "median_sample_ms": _median(values["sample_ms"]),
                "median_bookkeeping_ms": _median(values["bookkeeping_ms"]),
                "median_step_ms": _median(values["step_ms"]),
                "lm_head_pct_model_logits_sum": (
                    logits_sum / model_logits_sum if model_logits_sum > 0 else 0.0
                ),
                "lm_head_pct_core_sum": logits_sum / core_sum if core_sum > 0 else 0.0,
                "lm_head_pct_step_sum": logits_sum / step_sum if step_sum > 0 else 0.0,
                "median_lm_head_share_model_logits": _median(
                    values["lm_head_share_model_logits"]
                ),
                "median_lm_head_share_core": _median(values["lm_head_share_core"]),
                "median_lm_head_share_step": _median(values["lm_head_share_step"]),
            }
        )
    return rows


def write_csv(rows: list[dict[str, Any]], output: str | Path) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["phase", "steps"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
