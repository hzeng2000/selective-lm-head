"""Small BFCL/function-call helpers.

The official BFCL evaluator is intentionally optional. These helpers normalize
OpenAI-style function definitions into a JSON schema that can be passed to
structured-output backends for smoke and controlled runs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BFCLRequest:
    request_id: str
    prompt: str
    functions: list[dict[str, Any]]
    category: str
    expected: Any = None


def _function_name(function: dict[str, Any]) -> str:
    if "name" in function:
        return str(function["name"])
    if "function" in function and isinstance(function["function"], dict):
        return str(function["function"].get("name", "unknown"))
    return "unknown"


def normalize_bfcl_json_schema(schema: Any) -> Any:
    """Normalize BFCL's Python-ish schema dialect into JSON Schema."""

    if isinstance(schema, list):
        return [normalize_bfcl_json_schema(item) for item in schema]
    if not isinstance(schema, dict):
        return schema

    normalized: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "type":
            type_value = value
            if isinstance(type_value, str):
                type_map = {
                    "dict": "object",
                    "Dict": "object",
                    "object": "object",
                    "list": "array",
                    "List": "array",
                    "array": "array",
                    "tuple": "array",
                    "str": "string",
                    "string": "string",
                    "int": "integer",
                    "integer": "integer",
                    "float": "number",
                    "double": "number",
                    "number": "number",
                    "bool": "boolean",
                    "boolean": "boolean",
                    "None": "null",
                    "null": "null",
                }
                normalized[key] = type_map.get(type_value, type_value)
            else:
                normalized[key] = normalize_bfcl_json_schema(type_value)
        elif key == "properties" and isinstance(value, dict):
            normalized[key] = {
                str(prop): normalize_bfcl_json_schema(prop_schema)
                for prop, prop_schema in value.items()
            }
        elif key in {"items", "additionalProperties", "prefixItems", "oneOf", "anyOf", "allOf"}:
            normalized[key] = normalize_bfcl_json_schema(value)
        else:
            normalized[key] = normalize_bfcl_json_schema(value)

    if normalized.get("type") == "object":
        normalized.setdefault("properties", {})
        normalized.setdefault("additionalProperties", False)
    return normalized


def function_definitions_to_schema(functions: list[dict[str, Any]]) -> dict[str, Any]:
    names = [_function_name(function) for function in functions]
    argument_schemas: list[dict[str, Any]] = []
    for function in functions:
        body = function.get("function", function)
        parameters = normalize_bfcl_json_schema(body.get("parameters") or {"type": "object"})
        argument_schemas.append(
            {
                "type": "object",
                "properties": {
                    "name": {"const": str(body.get("name", "unknown"))},
                    "arguments": parameters,
                },
                "required": ["name", "arguments"],
                "additionalProperties": False,
            }
        )

    if len(argument_schemas) == 1:
        return argument_schemas[0]
    return {
        "oneOf": argument_schemas,
        "description": "One tool call matching the provided function definitions.",
        "x-function-names": names,
    }


def smoke_bfcl_requests(limit: int | None = None) -> list[BFCLRequest]:
    functions = [
        {
            "name": "get_weather",
            "description": "Get weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city"],
                "additionalProperties": False,
            },
        },
        {
            "name": "search_file",
            "description": "Search for files by query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 10},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    ]
    rows = [
        BFCLRequest(
            request_id="bfcl/smoke/weather",
            prompt=(
                "Call the correct tool for this request: what is the weather in Shanghai? "
                "Return only a JSON tool call."
            ),
            functions=[functions[0]],
            category="simple_python",
        ),
        BFCLRequest(
            request_id="bfcl/smoke/search",
            prompt="Call the correct tool to search project files for README. Return only JSON.",
            functions=[functions[1]],
            category="simple_python",
        ),
    ]
    return rows[:limit] if limit is not None else rows


def bfcl_request_to_prompt(request: BFCLRequest) -> str:
    return (
        request.prompt
        + "\nFunction definitions:\n"
        + json.dumps(request.functions, ensure_ascii=False, separators=(",", ":"))
    )
