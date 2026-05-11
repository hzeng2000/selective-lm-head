"""Allowed-token bitmask conversion utilities."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from typing import Any


def bucket_k(k: int) -> str:
    if k == 1:
        return "K=1"
    if k <= 8:
        return "K<=8"
    if k <= 32:
        return "K<=32"
    if k <= 128:
        return "K<=128"
    if k <= 512:
        return "K<=512"
    if k <= 2048:
        return "K<=2048"
    if k <= 8192:
        return "K<=8192"
    return "K>8192"


def _is_torch_tensor(value: Any) -> bool:
    return value.__class__.__module__.startswith("torch") and hasattr(value, "detach")


def _torch_bitmask_to_allowed_ids(mask: Any, vocab_size: int | None, allowed_bit: int) -> list[int]:
    return bitmask_to_allowed_ids_tensor(
        mask, vocab_size=vocab_size, allowed_bit=allowed_bit, device="cpu"
    ).tolist()


def _torch_packed_bitmask_to_allowed_ids_tensor(
    mask: Any,
    vocab_size: int | None = None,
    allowed_bit: int = 1,
    device: Any | None = None,
) -> Any:
    import torch

    tensor = mask.detach()
    if device is not None:
        tensor = tensor.to(device=device, non_blocking=True)
    words = tensor.reshape(-1).to(dtype=torch.int64)
    shifts = torch.arange(32, device=words.device, dtype=torch.int64)
    bits = ((words.unsqueeze(1) >> shifts) & 1).reshape(-1)
    if vocab_size is not None:
        bits = bits[:vocab_size]
    return (bits == int(allowed_bit)).nonzero(as_tuple=False).reshape(-1).to(dtype=torch.long)


def bitmask_to_allowed_ids_tensor(
    bitmask: Any,
    vocab_size: int | None = None,
    allowed_bit: int = 1,
    device: Any | None = None,
) -> Any:
    """Convert a bitmask to a torch LongTensor of allowed token ids.

    For torch bitmasks this avoids Python per-bit loops and can return ids
    directly on the target device used by the LM head.
    """

    import torch

    if _is_torch_tensor(bitmask):
        tensor = bitmask.detach()
        if tensor.ndim > 1:
            tensor = tensor.reshape(-1)
        if tensor.dtype == torch.bool:
            if device is not None:
                tensor = tensor.to(device=device, non_blocking=True)
            if vocab_size is not None:
                tensor = tensor[:vocab_size]
            return tensor.nonzero(as_tuple=False).reshape(-1).to(dtype=torch.long)
        return _torch_packed_bitmask_to_allowed_ids_tensor(
            tensor, vocab_size=vocab_size, allowed_bit=allowed_bit, device=device
        )

    ids = bitmask_to_allowed_ids(bitmask, vocab_size=vocab_size, allowed_bit=allowed_bit)
    return torch.tensor(ids, device=device, dtype=torch.long)


def _packed_ints_to_allowed_ids(
    values: Sequence[int], vocab_size: int | None = None, allowed_bit: int = 1
) -> list[int]:
    allowed: list[int] = []
    max_bits = vocab_size if vocab_size is not None else len(values) * 32
    for word_index, raw_value in enumerate(values):
        value = int(raw_value)
        for bit in range(32):
            token_id = word_index * 32 + bit
            if token_id >= max_bits:
                return allowed
            if ((value >> bit) & 1) == allowed_bit:
                allowed.append(token_id)
    return allowed


def bitmask_to_allowed_ids(
    bitmask: Any, vocab_size: int | None = None, allowed_bit: int = 1
) -> list[int]:
    """Convert a bool mask or packed int32 mask to a sorted allowed token list.

    XGrammar-style masks are usually packed int32 arrays; simple tests often use
    bool lists. A set bit is treated as allowed by default.
    """

    if _is_torch_tensor(bitmask):
        return _torch_bitmask_to_allowed_ids(bitmask, vocab_size=vocab_size, allowed_bit=allowed_bit)

    if isinstance(bitmask, (bytes, bytearray)):
        allowed: list[int] = []
        max_bits = vocab_size if vocab_size is not None else len(bitmask) * 8
        for byte_index, value in enumerate(bitmask):
            for bit in range(8):
                token_id = byte_index * 8 + bit
                if token_id >= max_bits:
                    return allowed
                if ((value >> bit) & 1) == allowed_bit:
                    allowed.append(token_id)
        return allowed

    if isinstance(bitmask, Iterable):
        values = list(bitmask)
        if not values:
            return []
        if all(isinstance(value, bool) for value in values):
            limit = vocab_size if vocab_size is not None else len(values)
            return [idx for idx, value in enumerate(values[:limit]) if value]
        if vocab_size is not None and len(values) * 32 >= vocab_size and any(int(v) > 1 for v in values):
            return _packed_ints_to_allowed_ids(values, vocab_size=vocab_size, allowed_bit=allowed_bit)
        if all(int(value) in (0, 1) for value in values):
            limit = vocab_size if vocab_size is not None else len(values)
            return [idx for idx, value in enumerate(values[:limit]) if int(value) == allowed_bit]
        return _packed_ints_to_allowed_ids(values, vocab_size=vocab_size, allowed_bit=allowed_bit)

    raise TypeError(f"Unsupported bitmask type: {type(bitmask)!r}")


def allowed_ids_to_bool_mask(allowed_ids: Sequence[int], vocab_size: int) -> list[bool]:
    mask = [False] * vocab_size
    for token_id in allowed_ids:
        if token_id < 0 or token_id >= vocab_size:
            raise ValueError(f"token_id out of range: {token_id}")
        mask[token_id] = True
    return mask


def hash_allowed_ids(allowed_ids: Sequence[int]) -> str:
    digest = hashlib.blake2b(digest_size=16)
    for token_id in allowed_ids:
        digest.update(int(token_id).to_bytes(8, byteorder="little", signed=False))
    return digest.hexdigest()
