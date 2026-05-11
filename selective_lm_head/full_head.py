"""Full-head constrained token selection."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from selective_lm_head.lm_head import (
    apply_allowed_mask,
    full_lm_head,
    greedy_from_logits,
    sample_from_logits,
)


@dataclass
class HeadSelection:
    token_id: int
    selected_index: int
    path: str
    logits: Any | None = None
    fallback: bool = False


def full_head_select(
    hidden: Any,
    lm_head_weight: Any,
    allowed_ids: Sequence[int] | Any,
    bias: Any | None = None,
    temperature: float = 0.0,
    keep_logits: bool = False,
) -> HeadSelection:
    logits = full_lm_head(hidden, lm_head_weight, bias=bias)
    masked = apply_allowed_mask(logits, allowed_ids)
    if temperature <= 0:
        token_id, selected_index = greedy_from_logits(masked)
    else:
        token_id, selected_index = sample_from_logits(masked, temperature=temperature)
    return HeadSelection(
        token_id=token_id,
        selected_index=selected_index,
        path="full",
        logits=masked if keep_logits else None,
        fallback=True,
    )

