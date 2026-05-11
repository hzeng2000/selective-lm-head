"""JSON schema helpers for controlled and serving runners."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class JSONSchemaRequest:
    request_id: str
    schema: dict[str, Any]
    prompt: str
    category: str = "smoke"

    @property
    def schema_string(self) -> str:
        return minify_json_schema(self.schema)


def minify_json_schema(schema: dict[str, Any] | str) -> str:
    if isinstance(schema, str):
        parsed = json.loads(schema)
    else:
        parsed = schema
    return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def smoke_json_schema_requests(limit: int | None = None) -> list[JSONSchemaRequest]:
    schemas: list[tuple[str, dict[str, Any]]] = [
        (
            "weather_tool_args",
            {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city", "unit"],
                "additionalProperties": False,
            },
        ),
        (
            "router_intent",
            {
                "type": "object",
                "properties": {
                    "intent": {
                        "type": "string",
                        "enum": ["get_weather", "search_file", "send_email"],
                    }
                },
                "required": ["intent"],
                "additionalProperties": False,
            },
        ),
        (
            "todo_item",
            {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "priority": {"type": "string", "enum": ["low", "medium", "high"]},
                    "done": {"type": "boolean"},
                },
                "required": ["title", "priority", "done"],
                "additionalProperties": False,
            },
        ),
    ]
    requests = [
        JSONSchemaRequest(
            request_id=f"jsonschema/smoke/{name}",
            schema=schema,
            prompt=(
                "Generate one JSON object that satisfies the given JSON Schema. "
                "Return only JSON.\nJSON Schema:\n" + minify_json_schema(schema)
            ),
        )
        for name, schema in schemas
    ]
    return requests[:limit] if limit is not None else requests

