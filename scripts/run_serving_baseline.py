#!/usr/bin/env python
"""Run vLLM/SGLang OpenAI-compatible structured-output baseline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from selective_lm_head.benchmarks import load_requests
from selective_lm_head.environment import collect_environment
from selective_lm_head.eval.json_validity import validate_json_output
from selective_lm_head.serving_baseline.openai_api_runner import run_structured_request
from selective_lm_head.tracing.trace_schema import RequestTrace
from selective_lm_head.tracing.writer import JsonlTraceWriter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", choices=["vllm", "sglang"], required=True)
    parser.add_argument("--base-url", required=True, help="Example: http://127.0.0.1:8000/v1")
    parser.add_argument("--model", required=True, help="Served model name.")
    parser.add_argument("--benchmark", default="jsonschemabench", choices=["jsonschemabench", "jsonschema", "bfcl"])
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--categories", default=None, help="Comma-separated BFCL categories.")
    parser.add_argument("--splits", default=None, help="Comma-separated JSONSchemaBench splits.")
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-s", type=int, default=120)
    parser.add_argument("--output", required=True)
    parser.add_argument("--append", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
    with JsonlTraceWriter(output) as writer:
        for request in requests:
            result = run_structured_request(
                base_url=args.base_url,
                model=args.model,
                prompt=request.prompt,
                schema=request.schema,
                timeout_s=args.timeout_s,
                temperature=args.temperature,
            )
            validity = validate_json_output(result.output_text, schema=request.schema)
            usage = result.raw.get("usage", {}) if isinstance(result.raw, dict) else {}
            prompt_tokens = int(usage.get("prompt_tokens") or 0)
            output_tokens = int(usage.get("completion_tokens") or 0)
            latency_per_token = result.latency_ms / output_tokens if output_tokens else 0.0
            trace = RequestTrace(
                request_id=request.request_id,
                model=args.model,
                framework=f"{args.server}_structured",
                device="server",
                benchmark=request.benchmark,
                category=request.category,
                head="serving_structured",
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                latency_ms_total=result.latency_ms,
                latency_ms_decode=result.latency_ms,
                latency_ms_per_decode_token=latency_per_token,
                valid=bool(validity["valid"]) and result.status_code < 400,
                output_text=result.output_text,
                environment=collect_environment(),
                extra={"status_code": result.status_code, "raw": result.raw, "error": validity["error"]},
            )
            writer.write(trace)
            print(f"{request.request_id} status={result.status_code} latency_ms={result.latency_ms:.2f}")


if __name__ == "__main__":
    main()
