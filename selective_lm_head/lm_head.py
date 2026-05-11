"""LM-head math primitives."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from selective_lm_head.dependencies import require_module


def require_torch() -> Any:
    return require_module(
        "torch",
        "Run `bash scripts/setup_env.sh ml` to install the PyTorch/Transformers stack.",
    )


def get_lm_head_weight_and_bias(model: Any) -> tuple[Any, Any | None]:
    """Return `(weight, bias)` for common CausalLM implementations."""

    module = None
    if hasattr(model, "get_output_embeddings"):
        module = model.get_output_embeddings()
    if module is None and hasattr(model, "lm_head"):
        module = model.lm_head
    if module is None:
        raise AttributeError("Could not locate model output embeddings or lm_head.")
    weight = getattr(module, "weight", None)
    if weight is None:
        raise AttributeError("LM head module has no weight.")
    return weight, getattr(module, "bias", None)


def allowed_ids_tensor(allowed_ids: Sequence[int] | Any, device: Any | None = None) -> Any:
    torch = require_torch()
    if hasattr(allowed_ids, "to"):
        tensor = allowed_ids
        return tensor.to(device=device, dtype=torch.long) if device is not None else tensor.to(dtype=torch.long)
    return torch.tensor(list(allowed_ids), device=device, dtype=torch.long)


def normalize_hidden(hidden: Any) -> Any:
    if hidden.ndim == 1:
        return hidden.view(1, -1)
    if hidden.ndim == 2:
        return hidden
    if hidden.ndim == 3:
        return hidden[:, -1, :]
    raise ValueError(f"Expected hidden with 1, 2, or 3 dimensions, got shape {tuple(hidden.shape)}")


def full_lm_head(hidden: Any, lm_head_weight: Any, bias: Any | None = None) -> Any:
    torch = require_torch()
    import torch.nn.functional as F

    with torch.no_grad():
        return F.linear(normalize_hidden(hidden), lm_head_weight, bias)


def selective_lm_head(
    hidden: Any,
    lm_head_weight: Any,
    allowed_ids: Sequence[int] | Any,
    bias: Any | None = None,
) -> Any:
    """Compute logits only for `allowed_ids`.

    Returns shape `[batch, K]`. For batch=1 callers can use `.reshape(-1)`.
    """

    torch = require_torch()
    import torch.nn.functional as F

    with torch.no_grad():
        hidden_2d = normalize_hidden(hidden)
        ids = allowed_ids_tensor(allowed_ids, device=lm_head_weight.device)
        selected_weight = lm_head_weight.index_select(0, ids)
        selected_bias = bias.index_select(0, ids) if bias is not None else None
        return F.linear(hidden_2d, selected_weight, selected_bias)


def apply_allowed_mask(logits: Any, allowed_ids: Sequence[int] | Any, fill_value: float = float("-inf")) -> Any:
    """Return logits with everything outside allowed_ids set to -inf."""

    torch = require_torch()
    with torch.no_grad():
        if logits.ndim == 1:
            ids = allowed_ids_tensor(allowed_ids, device=logits.device)
            masked = torch.full_like(logits, fill_value)
            masked.index_copy_(0, ids, logits.index_select(0, ids))
            return masked
        if logits.ndim == 2:
            ids = allowed_ids_tensor(allowed_ids, device=logits.device)
            masked = torch.full_like(logits, fill_value)
            masked.index_copy_(1, ids, logits.index_select(1, ids))
            return masked
        raise ValueError(f"Expected 1D or 2D logits, got shape {tuple(logits.shape)}")


def greedy_from_logits(logits: Any, allowed_ids: Sequence[int] | Any | None = None) -> tuple[int, int]:
    """Return `(token_id, selected_index)` for greedy decoding."""

    torch = require_torch()
    with torch.no_grad():
        flat = logits.reshape(-1)
        selected_index = int(torch.argmax(flat).item())
        if allowed_ids is None:
            return selected_index, selected_index
        if hasattr(allowed_ids, "detach"):
            token_id = int(allowed_ids.reshape(-1)[selected_index].item())
        else:
            token_id = int(list(allowed_ids)[selected_index])
        return token_id, selected_index


def sample_from_logits(
    logits: Any,
    allowed_ids: Sequence[int] | Any | None = None,
    temperature: float = 1.0,
) -> tuple[int, int]:
    """Sample from logits and return `(token_id, selected_index)`."""

    torch = require_torch()
    with torch.no_grad():
        flat = logits.reshape(-1)
        if temperature <= 0:
            return greedy_from_logits(flat, allowed_ids=allowed_ids)
        probs = torch.softmax(flat / temperature, dim=-1)
        selected_index = int(torch.multinomial(probs, num_samples=1).item())
        if allowed_ids is None:
            return selected_index, selected_index
        if hasattr(allowed_ids, "detach"):
            token_id = int(allowed_ids.reshape(-1)[selected_index].item())
        else:
            token_id = int(list(allowed_ids)[selected_index])
        return token_id, selected_index


def verify_selective_exactness(
    hidden: Any,
    lm_head_weight: Any,
    allowed_ids: Sequence[int] | Any,
    bias: Any | None = None,
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> dict[str, Any]:
    """Compare full masked logits and selective logits for one hidden state."""

    torch = require_torch()
    with torch.no_grad():
        full_logits = full_lm_head(hidden, lm_head_weight, bias=bias)
        ids = allowed_ids_tensor(allowed_ids, device=full_logits.device)
        selected_from_full = full_logits.index_select(-1, ids)
        selected_logits = selective_lm_head(hidden, lm_head_weight, ids, bias=bias)
        logits_match = bool(torch.allclose(selected_from_full, selected_logits, atol=atol, rtol=rtol))
        full_token, _ = greedy_from_logits(apply_allowed_mask(full_logits, ids))
        selective_token, _ = greedy_from_logits(selected_logits, ids)
        max_abs_diff = float((selected_from_full - selected_logits).abs().max().item())
        return {
            "logits_match": logits_match,
            "argmax_match": full_token == selective_token,
            "full_token": full_token,
            "selective_token": selective_token,
            "max_abs_diff": max_abs_diff,
            "K": int(ids.numel()),
        }

