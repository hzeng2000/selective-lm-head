"""XGrammar integration wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from selective_lm_head.dependencies import require_module
from selective_lm_head.grammar.bitmask import bitmask_to_allowed_ids, bitmask_to_allowed_ids_tensor
from selective_lm_head.tracing.timers import Stopwatch


@dataclass
class AllowedSetResult:
    allowed_ids: Any
    provider_ms: float
    bitmask_to_list_ms: float
    cache_hit: bool | None = None


class CachedAllowedSetProvider:
    """Cache bitmask-to-list results by packed-mask bytes."""

    name = "cached_bitmask"

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.cache: dict[tuple[bytes, str], Any] = {}
        self.hits = 0
        self.misses = 0

    def convert(self, bitmask: Any) -> tuple[list[int], bool, float]:
        key = (_mask_to_bytes(bitmask), "list")
        if key in self.cache:
            self.hits += 1
            return self.cache[key], True, 0.0
        timer = Stopwatch()
        timer.start()
        allowed = bitmask_to_allowed_ids(bitmask, vocab_size=self.vocab_size)
        elapsed = timer.stop()
        self.cache[key] = allowed
        self.misses += 1
        return allowed, False, elapsed

    def convert_tensor(self, bitmask: Any, device: Any | None = None) -> tuple[Any, bool, float]:
        device_key = str(device) if device is not None else "none"
        key = (_mask_to_bytes(bitmask), f"tensor:{device_key}")
        if key in self.cache:
            self.hits += 1
            return self.cache[key], True, 0.0
        device_name = str(device) if device is not None else None
        timer = Stopwatch(device=device_name, sync=bool(device_name and device_name.startswith("cuda")))
        timer.start()
        allowed = bitmask_to_allowed_ids_tensor(bitmask, vocab_size=self.vocab_size, device=device)
        elapsed = timer.stop()
        self.cache[key] = allowed
        self.misses += 1
        return allowed, False, elapsed

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


def _mask_to_bytes(bitmask: Any) -> bytes:
    if hasattr(bitmask, "detach"):
        tensor = bitmask.detach().to(device="cpu")
        return tensor.numpy().tobytes()
    if isinstance(bitmask, bytes):
        return bitmask
    if isinstance(bitmask, bytearray):
        return bytes(bitmask)
    if isinstance(bitmask, (list, tuple)):
        return bytes(str(tuple(bitmask)), encoding="utf-8")
    return bytes(str(bitmask), encoding="utf-8")


class XGrammarBackend:
    def __init__(self, tokenizer: Any, vocab_size: int):
        self.xgr = require_module(
            "xgrammar",
            "Run `bash scripts/setup_env.sh ml` to install XGrammar with the ML stack.",
        )
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        tokenizer_info = self.xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size)
        self.compiler = self.xgr.GrammarCompiler(tokenizer_info)

    def compile_json_schema(self, schema_string: str) -> "XGrammarMatcher":
        compiled = self.compiler.compile_json_schema(schema_string)
        return XGrammarMatcher(self.xgr, compiled, self.vocab_size)

    def compile_builtin_json(self) -> "XGrammarMatcher":
        compiled = self.compiler.compile_builtin_json_grammar()
        return XGrammarMatcher(self.xgr, compiled, self.vocab_size)


class XGrammarMatcher:
    name = "xgrammar"

    def __init__(self, xgr: Any, compiled_grammar: Any, vocab_size: int):
        self.xgr = xgr
        self.vocab_size = vocab_size
        self.matcher = xgr.GrammarMatcher(compiled_grammar)
        self.bitmask = xgr.allocate_token_bitmask(1, vocab_size)
        self.cache = CachedAllowedSetProvider(vocab_size=vocab_size)

    def fill_bitmask(self) -> Any:
        self.matcher.fill_next_token_bitmask(self.bitmask)
        try:
            return self.bitmask[0]
        except Exception:
            return self.bitmask

    def allowed_token_ids(
        self,
        cached: bool = True,
        as_tensor: bool = False,
        device: Any | None = None,
    ) -> AllowedSetResult:
        device_name = str(device) if device is not None else None
        sync_device = bool(as_tensor and device_name and device_name.startswith("cuda"))
        timer = Stopwatch(device=device_name, sync=sync_device)
        timer.start()
        bitmask = self.fill_bitmask()
        if cached:
            if as_tensor:
                allowed, hit, bitmask_ms = self.cache.convert_tensor(bitmask, device=device)
            else:
                allowed, hit, bitmask_ms = self.cache.convert(bitmask)
        else:
            list_timer = Stopwatch(device=device_name, sync=sync_device)
            list_timer.start()
            if as_tensor:
                allowed = bitmask_to_allowed_ids_tensor(
                    bitmask, vocab_size=self.vocab_size, device=device
                )
            else:
                allowed = bitmask_to_allowed_ids(bitmask, vocab_size=self.vocab_size)
            bitmask_ms = list_timer.stop()
            hit = None
        provider_ms = timer.stop()
        return AllowedSetResult(
            allowed_ids=allowed,
            provider_ms=provider_ms,
            bitmask_to_list_ms=bitmask_ms,
            cache_hit=hit,
        )

    def apply_token_bitmask_inplace(self, logits: Any) -> None:
        self.xgr.apply_token_bitmask_inplace(logits, self.bitmask)

    def accept_token(self, token_id: int) -> None:
        self.matcher.accept_token(int(token_id))

    def is_terminated(self) -> bool:
        return bool(self.matcher.is_terminated())
