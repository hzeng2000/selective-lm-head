"""Threshold-controlled selective LM head."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from selective_lm_head.full_head import HeadSelection, full_head_select
from selective_lm_head.lm_head import greedy_from_logits, sample_from_logits, selective_lm_head


@dataclass(frozen=True)
class SelectiveHeadConfig:
    k_threshold: int = 2048
    direct_when_k1: bool = True
    temperature: float = 0.0
    keep_logits: bool = False


class SelectiveLMHead:
    def __init__(
        self,
        lm_head_weight: Any,
        bias: Any | None = None,
        config: SelectiveHeadConfig | None = None,
    ):
        self.lm_head_weight = lm_head_weight
        self.bias = bias
        self.config = config or SelectiveHeadConfig()

    def select(self, hidden: Any, allowed_ids: Sequence[int] | Any) -> HeadSelection:
        k = int(allowed_ids.numel()) if hasattr(allowed_ids, "numel") else len(allowed_ids)
        if k <= 0:
            raise ValueError("allowed_ids is empty; grammar state is unsatisfiable.")

        if self.config.direct_when_k1 and k == 1:
            if hasattr(allowed_ids, "reshape"):
                token_id = int(allowed_ids.reshape(-1)[0].item())
            else:
                token_id = int(list(allowed_ids)[0])
            return HeadSelection(token_id=token_id, selected_index=0, path="direct", fallback=False)

        if k > self.config.k_threshold:
            return full_head_select(
                hidden=hidden,
                lm_head_weight=self.lm_head_weight,
                allowed_ids=allowed_ids,
                bias=self.bias,
                temperature=self.config.temperature,
                keep_logits=self.config.keep_logits,
            )

        logits = selective_lm_head(hidden, self.lm_head_weight, allowed_ids, bias=self.bias)
        if self.config.temperature <= 0:
            token_id, selected_index = greedy_from_logits(logits, allowed_ids=allowed_ids)
        else:
            token_id, selected_index = sample_from_logits(
                logits, allowed_ids=allowed_ids, temperature=self.config.temperature
            )
        return HeadSelection(
            token_id=token_id,
            selected_index=selected_index,
            path="selective",
            logits=logits if self.config.keep_logits else None,
            fallback=False,
        )

