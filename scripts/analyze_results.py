#!/usr/bin/env python
"""Analyze trace JSONL files."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from selective_lm_head.analysis.k_distribution import (
    summarize_k_distribution,
    write_csv as write_k_csv,
)
from selective_lm_head.analysis.latency_breakdown import (
    summarize_latency_breakdown,
    write_csv as write_latency_csv,
)
from selective_lm_head.analysis.paired_speedup import (
    summarize_paired_speedup,
    write_csv as write_speedup_csv,
)
from selective_lm_head.analysis.serving_summary import (
    summarize_serving_trace,
    write_csv as write_serving_csv,
)
from selective_lm_head.analysis.vllm_lm_head_profile import (
    summarize_vllm_lm_head_profile,
    write_csv as write_vllm_profile_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    k_parser = sub.add_parser("k-distribution")
    k_parser.add_argument("--trace", required=True)
    k_parser.add_argument("--output", required=True)

    latency_parser = sub.add_parser("latency-breakdown")
    latency_parser.add_argument("--trace", required=True)
    latency_parser.add_argument("--output", required=True)

    speedup_parser = sub.add_parser("paired-speedup")
    speedup_parser.add_argument("--baseline", required=True)
    speedup_parser.add_argument("--selective", required=True)
    speedup_parser.add_argument("--output", required=True)

    serving_parser = sub.add_parser("serving-summary")
    serving_parser.add_argument("--trace", required=True)
    serving_parser.add_argument("--output", required=True)

    vllm_profile_parser = sub.add_parser("vllm-lm-head-profile")
    vllm_profile_parser.add_argument("--profile", required=True)
    vllm_profile_parser.add_argument("--output", required=True)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "k-distribution":
        rows = summarize_k_distribution(args.trace)
        write_k_csv(rows, args.output)
    elif args.command == "latency-breakdown":
        rows = summarize_latency_breakdown(args.trace)
        write_latency_csv(rows, args.output)
    elif args.command == "paired-speedup":
        rows = summarize_paired_speedup(args.baseline, args.selective)
        write_speedup_csv(rows, args.output)
    elif args.command == "serving-summary":
        rows = summarize_serving_trace(args.trace)
        write_serving_csv(rows, args.output)
    elif args.command == "vllm-lm-head-profile":
        rows = summarize_vllm_lm_head_profile(args.profile)
        write_vllm_profile_csv(rows, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
