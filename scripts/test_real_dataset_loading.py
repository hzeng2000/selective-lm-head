#!/usr/bin/env python
"""Fast smoke test for real BFCL and JSONSchemaBench loaders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from selective_lm_head.benchmarks import load_requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bfcl-dir", default="data/repos/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data")
    parser.add_argument("--jsonschema-dir", default="data/repos/jsonschemabench/data")
    parser.add_argument("--max-requests", type=int, default=3)
    parser.add_argument("--output", default="results/tables/real_dataset_loading.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bfcl = load_requests(
        "bfcl",
        data_path=args.bfcl_dir,
        categories=["simple_python", "simple_java", "simple_javascript"],
        max_requests=args.max_requests,
    )
    jsonschema = load_requests(
        "jsonschemabench",
        data_path=args.jsonschema_dir,
        splits=["GlaiveAI-2K", "Kubernetes", "Github-Easy"],
        max_requests=args.max_requests,
    )
    summary = {
        "bfcl_loaded": len(bfcl),
        "bfcl_first": bfcl[0].__dict__ if bfcl else None,
        "jsonschemabench_loaded": len(jsonschema),
        "jsonschemabench_first": jsonschema[0].__dict__ if jsonschema else None,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if not k.endswith("_first")}, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
