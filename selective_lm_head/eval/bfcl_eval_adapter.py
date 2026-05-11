"""Optional BFCL evaluator adapter."""

from __future__ import annotations

import json
from typing import Any


def exact_json_match(output: str, expected: Any) -> dict[str, Any]:
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError as exc:
        return {"score": 0, "valid": False, "error": str(exc)}
    return {"score": int(parsed == expected), "valid": True, "parsed": parsed}


def evaluate_with_official_bfcl(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    raise NotImplementedError(
        "The official BFCL evaluator is not vendored in this repository. "
        "Install/clone BFCL and wire it here for leaderboard-compatible scoring."
    )

