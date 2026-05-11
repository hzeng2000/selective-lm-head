#!/usr/bin/env python
"""Head-only replay and threshold sweep from a trace."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from selective_lm_head.config import resolve_model
from selective_lm_head.grammar.bitmask import bucket_k
from selective_lm_head.lm_head import (
    allowed_ids_tensor,
    apply_allowed_mask,
    full_lm_head,
    get_lm_head_weight_and_bias,
    greedy_from_logits,
    selective_lm_head,
    verify_selective_exactness,
)
from selective_lm_head.model_loader import load_model_bundle
from selective_lm_head.tracing.timers import Stopwatch
from selective_lm_head.tracing.writer import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--thresholds", default="1,8,32,128,512,2048,8192")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--no-sync-cuda", action="store_true")
    parser.add_argument(
        "--allow-synthetic-ids",
        action="store_true",
        help="Use range(K) when traces do not contain allowed_ids.",
    )
    return parser.parse_args()


def _iter_steps(trace_path: str | Path, allow_synthetic_ids: bool, vocab_size: int, max_steps: int | None):
    count = 0
    for request in read_jsonl(trace_path):
        for step in request.get("steps", []):
            allowed_ids = step.get("allowed_ids")
            k = int(step.get("K") or 0)
            if allowed_ids is None:
                if not allow_synthetic_ids:
                    continue
                allowed_ids = list(range(min(k, vocab_size)))
            yield request, step, allowed_ids
            count += 1
            if max_steps is not None and count >= max_steps:
                return


def main() -> None:
    args = parse_args()
    thresholds = [int(value) for value in args.thresholds.split(",") if value.strip()]
    cfg = resolve_model(args.model)
    bundle = load_model_bundle(args.model, device=args.device, dtype=args.dtype)

    import torch

    weight, bias = get_lm_head_weight_and_bias(bundle.model)
    sync = not args.no_sync_cuda
    rows: list[dict[str, object]] = []

    for request, step, allowed_list in _iter_steps(
        args.trace,
        allow_synthetic_ids=args.allow_synthetic_ids,
        vocab_size=cfg.vocab_size,
        max_steps=args.max_steps,
    ):
        ids = allowed_ids_tensor(allowed_list, device=weight.device)
        if ids.numel() == 0:
            continue
        hidden = torch.randn((1, cfg.hidden_size), device=weight.device, dtype=weight.dtype)

        full_timer = Stopwatch(device=args.device, sync=sync)
        full_timer.start()
        full_logits = full_lm_head(hidden, weight, bias=bias)
        full_masked = apply_allowed_mask(full_logits, ids)
        full_token, _ = greedy_from_logits(full_masked)
        full_ms = full_timer.stop()

        exactness = verify_selective_exactness(hidden, weight, ids, bias=bias)
        for threshold in thresholds:
            k = int(ids.numel())
            if k == 1:
                selected_token = int(ids.reshape(-1)[0].item())
                selective_ms = 0.0
                path = "direct"
            elif k <= threshold:
                selective_timer = Stopwatch(device=args.device, sync=sync)
                selective_timer.start()
                selected_logits = selective_lm_head(hidden, weight, ids, bias=bias)
                selected_token, _ = greedy_from_logits(selected_logits, allowed_ids=ids)
                selective_ms = selective_timer.stop()
                path = "selective"
            else:
                selected_token = full_token
                selective_ms = full_ms
                path = "fallback_full"
            rows.append(
                {
                    "request_id": request.get("request_id"),
                    "step": step.get("step"),
                    "threshold": threshold,
                    "K": k,
                    "bucket": bucket_k(k),
                    "path": path,
                    "full_head_ms": full_ms,
                    "selective_or_fallback_ms": selective_ms,
                    "head_speedup": full_ms / selective_ms if selective_ms > 0 else "",
                    "argmax_match": selected_token == full_token,
                    "logits_match": exactness["logits_match"],
                    "max_abs_diff": exactness["max_abs_diff"],
                    "synthetic_ids": step.get("allowed_ids") is None,
                }
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else [
        "request_id",
        "step",
        "threshold",
        "K",
        "bucket",
        "path",
        "full_head_ms",
        "selective_or_fallback_ms",
        "head_speedup",
        "argmax_match",
        "logits_match",
        "max_abs_diff",
        "synthetic_ids",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
