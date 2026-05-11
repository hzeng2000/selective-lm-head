"""Trace schema used by every runner."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class StepTrace:
    step: int
    K: int
    K_over_V: float
    transformer_ms: float = 0.0
    lm_head_ms: float = 0.0
    mask_or_allowed_ms: float = 0.0
    sampler_ms: float = 0.0
    token_id: int | None = None
    path: str = "full"
    provider: str | None = None
    fallback: bool = False
    cache_hit: bool | None = None
    bitmask_to_list_ms: float = 0.0
    allowed_ids: list[int] | None = None
    selected_index: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RequestTrace:
    request_id: str
    model: str
    framework: str
    device: str
    benchmark: str
    head: str
    category: str | None = None
    prompt_tokens: int = 0
    output_tokens: int = 0
    latency_ms_total: float = 0.0
    latency_ms_prefill: float = 0.0
    latency_ms_decode: float = 0.0
    latency_ms_per_decode_token: float = 0.0
    valid: bool | None = None
    score: float | int | None = None
    output_text: str | None = None
    error: str | None = None
    environment: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    steps: list[StepTrace] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["steps"] = [step.to_dict() for step in self.steps]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RequestTrace":
        copied = dict(data)
        copied["steps"] = [StepTrace(**step) for step in copied.get("steps", [])]
        return cls(**copied)

