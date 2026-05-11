"""Benchmark request loading.

The real BFCL/JSONSchemaBench datasets are optional. When no dataset path is
provided, the runners use small smoke requests that exercise the same JSON
schema constrained-decoding path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from selective_lm_head.config import CONFIG_DIR, load_yaml
from selective_lm_head.grammar.bfcl_adapter import (
    bfcl_request_to_prompt,
    function_definitions_to_schema,
    smoke_bfcl_requests,
)
from selective_lm_head.grammar.json_schema_adapter import smoke_json_schema_requests


@dataclass(frozen=True)
class BenchmarkRequest:
    request_id: str
    benchmark: str
    prompt: str
    schema: dict[str, Any] | None = None
    category: str | None = None
    expected: Any = None
    raw: dict[str, Any] | None = None


def _read_json_or_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if source.suffix == ".jsonl":
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
        return data["data"]
    raise ValueError(f"Unsupported benchmark file shape: {path}")


def _dataset_config() -> dict[str, Any]:
    path = CONFIG_DIR / "datasets.yaml"
    return load_yaml(path) if path.exists() else {}


def _normalize_split_name(split: str) -> str:
    return split.strip().replace("_", "-").lower()


def _jsonschema_split_dir(data_dir: Path, split: str) -> Path:
    configured = _dataset_config().get("jsonschemabench", {}).get("splits", {})
    for public_name, dirname in configured.items():
        if _normalize_split_name(split) == _normalize_split_name(public_name):
            return data_dir / dirname
    candidates = {
        split,
        split.replace("-", "_"),
        split.replace("_", "-"),
        split.replace("Github-", "Github_"),
        split.replace("GlaiveAI-2K", "Glaiveai2K"),
    }
    for candidate in candidates:
        path = data_dir / candidate
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find JSONSchemaBench split {split!r} under {data_dir}")


def _bfcl_category_file(data_dir: Path, category: str) -> Path:
    configured = _dataset_config().get("bfcl", {}).get("categories", {})
    filename = configured.get(category) or configured.get(category.strip())
    if filename:
        path = data_dir / filename
        if path.exists():
            return path
    path = data_dir / f"BFCL_v4_{category}.json"
    if path.exists():
        return path
    raise FileNotFoundError(f"Could not find BFCL category {category!r} under {data_dir}")


def _load_jsonschema_from_path(path: str | Path, max_requests: int | None) -> list[BenchmarkRequest]:
    source = Path(path)
    if source.is_dir():
        files = sorted(source.glob("*.json"))
        requests: list[BenchmarkRequest] = []
        for file_path in files:
            schema = json.loads(file_path.read_text(encoding="utf-8"))
            if not isinstance(schema, dict):
                continue
            split = source.name
            request_id = f"jsonschemabench/{split}/{file_path.stem}"
            requests.append(
                BenchmarkRequest(
                    request_id=request_id,
                    benchmark="jsonschemabench",
                    prompt=(
                        "Generate one JSON object that satisfies the given JSON Schema. Return only JSON.\n"
                        "JSON Schema:\n"
                        + json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
                    ),
                    schema=schema,
                    category=split,
                    raw={"path": str(file_path)},
                )
            )
            if max_requests is not None and len(requests) >= max_requests:
                break
        return requests

    rows = _read_json_or_jsonl(source)
    requests: list[BenchmarkRequest] = []
    for idx, row in enumerate(rows):
        schema = row.get("schema") or row.get("json_schema") or row.get("output_schema")
        if isinstance(schema, str):
            schema = json.loads(schema)
        if not isinstance(schema, dict):
            continue
        request_id = str(row.get("request_id") or row.get("id") or f"jsonschema/file/{idx}")
        prompt = row.get("prompt") or (
            "Generate one JSON object that satisfies the given JSON Schema. Return only JSON.\n"
            + json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
        )
        requests.append(
            BenchmarkRequest(
                request_id=request_id,
                benchmark="jsonschemabench",
                prompt=str(prompt),
                schema=schema,
                category=str(row.get("split") or row.get("category") or "file"),
                raw=row,
            )
        )
        if max_requests is not None and len(requests) >= max_requests:
            break
    return requests


def _load_bfcl_from_path(path: str | Path, max_requests: int | None) -> list[BenchmarkRequest]:
    rows = _read_json_or_jsonl(path)
    requests: list[BenchmarkRequest] = []
    for idx, row in enumerate(rows):
        functions = row.get("functions") or row.get("tools") or row.get("function")
        if isinstance(functions, dict):
            functions = [functions]
        if not isinstance(functions, list):
            continue
        prompt = row.get("prompt") or row.get("question") or row.get("messages") or ""
        prompt = _flatten_prompt(prompt)
        prompt = (
            prompt
            + "\nFunction definitions:\n"
            + json.dumps(functions, ensure_ascii=False, separators=(",", ":"))
        )
        schema = function_definitions_to_schema(functions)
        request_id = str(row.get("request_id") or row.get("id") or f"bfcl/file/{idx}")
        requests.append(
            BenchmarkRequest(
                request_id=request_id,
                benchmark="bfcl",
                prompt=str(prompt),
                schema=schema,
                category=str(row.get("category") or "file"),
                expected=row.get("expected") or row.get("answer"),
                raw=row,
            )
        )
        if max_requests is not None and len(requests) >= max_requests:
            break
    return requests


def _flatten_prompt(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, dict):
        return str(prompt.get("content", prompt))
    if isinstance(prompt, list):
        parts: list[str] = []
        for item in prompt:
            text = _flatten_prompt(item)
            if text:
                parts.append(text)
        return "\n".join(parts)
    return str(prompt)


def load_requests(
    benchmark: str,
    data_path: str | Path | None = None,
    max_requests: int | None = None,
    categories: list[str] | None = None,
    splits: list[str] | None = None,
) -> list[BenchmarkRequest]:
    normalized = benchmark.lower()
    if data_path:
        source = Path(data_path)
        if normalized in {"jsonschema", "jsonschemabench"}:
            if source.is_dir() and any(child.is_dir() for child in source.iterdir()):
                selected_splits = splits or [
                    "GlaiveAI-2K",
                    "Kubernetes",
                    "Github-Easy",
                    "Github-Medium",
                    "Github-Hard",
                ]
                requests: list[BenchmarkRequest] = []
                for split in selected_splits:
                    split_dir = _jsonschema_split_dir(source, split)
                    remaining = None if max_requests is None else max_requests - len(requests)
                    if remaining is not None and remaining <= 0:
                        break
                    requests.extend(_load_jsonschema_from_path(split_dir, max_requests=remaining))
                return requests
            return _load_jsonschema_from_path(source, max_requests=max_requests)
        if normalized == "bfcl":
            if source.is_dir():
                selected_categories = categories or [
                    "simple_python",
                    "simple_java",
                    "simple_javascript",
                    "multiple",
                    "parallel",
                    "irrelevance",
                    "live_simple",
                ]
                requests = []
                for category in selected_categories:
                    file_path = _bfcl_category_file(source, category)
                    remaining = None if max_requests is None else max_requests - len(requests)
                    if remaining is not None and remaining <= 0:
                        break
                    requests.extend(_load_bfcl_from_path(file_path, max_requests=remaining))
                return requests
            return _load_bfcl_from_path(source, max_requests=max_requests)
        raise ValueError(f"Unsupported benchmark: {benchmark}")

    if normalized in {"jsonschema", "jsonschemabench"}:
        rows = smoke_json_schema_requests(limit=max_requests)
        return [
            BenchmarkRequest(
                request_id=row.request_id,
                benchmark="jsonschemabench",
                prompt=row.prompt,
                schema=row.schema,
                category=row.category,
            )
            for row in rows
        ]

    if normalized == "bfcl":
        rows = smoke_bfcl_requests(limit=max_requests)
        return [
            BenchmarkRequest(
                request_id=row.request_id,
                benchmark="bfcl",
                prompt=bfcl_request_to_prompt(row),
                schema=function_definitions_to_schema(row.functions),
                category=row.category,
                expected=row.expected,
            )
            for row in rows
        ]

    raise ValueError(f"Unsupported benchmark: {benchmark}")
