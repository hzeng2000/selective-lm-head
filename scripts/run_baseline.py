#!/usr/bin/env python
"""Run controlled full-head constrained baseline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from selective_lm_head.benchmarks import load_requests
from selective_lm_head.decode_loop import ControlledDecoder, DecodeSettings
from selective_lm_head.model_loader import load_model_bundle
from selective_lm_head.tracing.writer import JsonlTraceWriter


def build_parser(default_head: str = "full", default_record_allowed_ids: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Model alias from configs/models.yaml or local path.")
    parser.add_argument("--benchmark", default="jsonschemabench", choices=["jsonschemabench", "jsonschema", "bfcl"])
    parser.add_argument("--data-path", default=None, help="Optional JSON/JSONL benchmark file.")
    parser.add_argument("--categories", default=None, help="Comma-separated BFCL categories.")
    parser.add_argument("--splits", default=None, help="Comma-separated JSONSchemaBench splits.")
    parser.add_argument("--device", default="cuda", help="cuda, cuda:0, or cpu.")
    parser.add_argument("--dtype", default=None, help="bf16, fp16, fp32, or auto.")
    parser.add_argument("--head", default=default_head, choices=["full", "selective"])
    parser.add_argument("--k-threshold", type=int, default=2048)
    parser.add_argument("--allowed-provider", default="cached_bitmask", choices=["bitmask", "cached_bitmask"])
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output", required=True)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--no-sync-cuda", action="store_true")
    parser.add_argument("--record-allowed-ids", action="store_true", default=default_record_allowed_ids)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    return parser


def run(args: argparse.Namespace) -> None:
    output = Path(args.output)
    if output.exists() and not args.append:
        output.unlink()

    categories = [item.strip() for item in args.categories.split(",") if item.strip()] if args.categories else None
    splits = [item.strip() for item in args.splits.split(",") if item.strip()] if args.splits else None
    requests = load_requests(
        args.benchmark,
        data_path=args.data_path,
        max_requests=args.max_requests,
        categories=categories,
        splits=splits,
    )
    bundle = load_model_bundle(
        model=args.model,
        device=args.device,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
    )
    settings = DecodeSettings(
        head=args.head,
        max_new_tokens=args.max_new_tokens,
        k_threshold=args.k_threshold,
        temperature=args.temperature,
        allowed_provider=args.allowed_provider,
        record_allowed_ids=args.record_allowed_ids,
        sync_cuda=not args.no_sync_cuda,
    )
    decoder = ControlledDecoder(bundle=bundle, settings=settings)

    with JsonlTraceWriter(output) as writer:
        for request in requests:
            trace = decoder.decode(request)
            writer.write(trace)
            status = "ok" if not trace.error else f"error={trace.error}"
            print(
                f"{request.request_id} {status} tokens={trace.output_tokens} "
                f"latency_ms={trace.latency_ms_total:.2f}"
            )


def main(default_head: str = "full", default_record_allowed_ids: bool = False) -> None:
    parser = build_parser(
        default_head=default_head,
        default_record_allowed_ids=default_record_allowed_ids,
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
