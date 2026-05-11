#!/usr/bin/env python
"""Write a small manifest for downloaded real datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bfcl-dir", default="data/repos/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data")
    parser.add_argument("--jsonschema-dir", default="data/repos/jsonschemabench/data")
    parser.add_argument("--output", default="data/dataset_manifest.json")
    return parser.parse_args()


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def main() -> None:
    args = parse_args()
    bfcl_dir = Path(args.bfcl_dir)
    jsonschema_dir = Path(args.jsonschema_dir)

    bfcl = {}
    for path in sorted(bfcl_dir.glob("BFCL_v4_*.json")):
        bfcl[path.name] = {"path": str(path), "requests": count_jsonl(path)}

    jsonschema = {}
    for split_dir in sorted(child for child in jsonschema_dir.iterdir() if child.is_dir()):
        jsonschema[split_dir.name] = {
            "path": str(split_dir),
            "schemas": len(list(split_dir.glob("*.json"))),
        }

    manifest = {
        "bfcl": bfcl,
        "jsonschemabench": jsonschema,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    print(f"BFCL files: {len(bfcl)}")
    print(f"JSONSchemaBench splits: {len(jsonschema)}")


if __name__ == "__main__":
    main()

