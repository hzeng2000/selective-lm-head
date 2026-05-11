"""JSON validity helpers."""

from __future__ import annotations

import json
from typing import Any


def parse_json_output(output: str) -> tuple[Any | None, str | None]:
    try:
        return json.loads(output), None
    except json.JSONDecodeError as exc:
        return None, str(exc)


def validate_json_output(output: str, schema: dict[str, Any] | None = None) -> dict[str, Any]:
    parsed, error = parse_json_output(output)
    if error:
        return {"valid": False, "parsed": None, "error": error}
    if schema is not None:
        try:
            import jsonschema

            jsonschema.validate(instance=parsed, schema=schema)
        except Exception as exc:
            return {"valid": False, "parsed": parsed, "error": str(exc)}
    return {"valid": True, "parsed": parsed, "error": None}

