"""OpenAI-compatible structured-output serving baseline client."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests
from urllib.parse import urlparse

from selective_lm_head.tracing.timers import Stopwatch


@dataclass(frozen=True)
class ServingResult:
    output_text: str
    latency_ms: float
    status_code: int
    raw: dict[str, Any]


def _response_format(schema: dict[str, Any] | None) -> dict[str, Any] | None:
    if schema is None:
        return {"type": "json_object"}
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "selective_lm_head_schema",
            "schema": schema,
            "strict": True,
        },
    }


def run_structured_request(
    base_url: str,
    model: str,
    prompt: str,
    schema: dict[str, Any] | None = None,
    timeout_s: int = 120,
    temperature: float = 0.0,
) -> ServingResult:
    url = base_url.rstrip("/") + "/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    response_format = _response_format(schema)
    if response_format is not None:
        payload["response_format"] = response_format

    timer = Stopwatch()
    timer.start()
    session = requests.Session()
    parsed = urlparse(url)
    if parsed.hostname in {"127.0.0.1", "localhost", "::1"}:
        session.trust_env = False
    response = session.post(url, json=payload, timeout=timeout_s)
    elapsed = timer.stop()
    raw = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
    output = ""
    try:
        output = raw["choices"][0]["message"]["content"]
    except Exception:
        output = response.text
    return ServingResult(output_text=output, latency_ms=elapsed, status_code=response.status_code, raw=raw)
