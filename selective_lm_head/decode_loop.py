"""Controlled full/selective constrained decode loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from selective_lm_head.benchmarks import BenchmarkRequest
from selective_lm_head.environment import collect_environment
from selective_lm_head.grammar.json_schema_adapter import minify_json_schema
from selective_lm_head.grammar.xgrammar_adapter import XGrammarBackend
from selective_lm_head.lm_head import (
    allowed_ids_tensor,
    apply_allowed_mask,
    full_lm_head,
    get_lm_head_weight_and_bias,
    greedy_from_logits,
    sample_from_logits,
    selective_lm_head,
)
from selective_lm_head.model_loader import ModelBundle
from selective_lm_head.tracing.timers import Stopwatch
from selective_lm_head.tracing.trace_schema import RequestTrace, StepTrace


@dataclass(frozen=True)
class DecodeSettings:
    head: str = "full"
    max_new_tokens: int = 256
    k_threshold: int = 2048
    temperature: float = 0.0
    allowed_provider: str = "cached_bitmask"
    record_allowed_ids: bool = False
    sync_cuda: bool = True


class ControlledDecoder:
    def __init__(self, bundle: ModelBundle, settings: DecodeSettings):
        self.bundle = bundle
        self.settings = settings
        self.weight, self.bias = get_lm_head_weight_and_bias(bundle.model)
        self.xgrammar = XGrammarBackend(bundle.tokenizer, vocab_size=bundle.config.vocab_size)

    def _compile_matcher(self, request: BenchmarkRequest) -> Any:
        if request.schema is None:
            return self.xgrammar.compile_builtin_json()
        return self.xgrammar.compile_json_schema(minify_json_schema(request.schema))

    def _allowed_len(self, allowed_ids: Any) -> int:
        return int(allowed_ids.numel()) if hasattr(allowed_ids, "numel") else len(allowed_ids)

    def _record_allowed_ids(self, allowed_ids: Any) -> list[int] | None:
        if not self.settings.record_allowed_ids:
            return None
        if hasattr(allowed_ids, "detach"):
            return allowed_ids.detach().to(device="cpu", dtype=__import__("torch").long).reshape(-1).tolist()
        return list(allowed_ids)

    def _select_token(self, hidden: Any, allowed_ids: Any) -> tuple[int, int, str, bool, float, float]:
        torch = __import__("torch")
        ids = allowed_ids_tensor(allowed_ids, device=self.weight.device)
        k = int(ids.numel())
        if k == 0:
            raise ValueError("Grammar returned an empty allowed token set.")

        if self.settings.head == "selective" and k == 1:
            return int(ids.reshape(-1)[0].item()), 0, "direct", False, 0.0, 0.0

        use_selective = self.settings.head == "selective" and k <= self.settings.k_threshold
        fallback = self.settings.head == "selective" and not use_selective

        head_timer = Stopwatch(device=self.bundle.device, sync=self.settings.sync_cuda)
        sample_timer = Stopwatch(device=self.bundle.device, sync=self.settings.sync_cuda)

        if use_selective:
            head_timer.start()
            logits = selective_lm_head(hidden, self.weight, ids, bias=self.bias)
            lm_head_ms = head_timer.stop()
            sample_timer.start()
            if self.settings.temperature <= 0:
                token_id, selected_index = greedy_from_logits(logits, allowed_ids=ids)
            else:
                token_id, selected_index = sample_from_logits(
                    logits, allowed_ids=ids, temperature=self.settings.temperature
                )
            sampler_ms = sample_timer.stop()
            return token_id, selected_index, "selective", False, lm_head_ms, sampler_ms

        head_timer.start()
        full_logits = full_lm_head(hidden, self.weight, bias=self.bias)
        masked_logits = apply_allowed_mask(full_logits, ids)
        lm_head_ms = head_timer.stop()
        sample_timer.start()
        if self.settings.temperature <= 0:
            token_id, selected_index = greedy_from_logits(masked_logits)
        else:
            token_id, selected_index = sample_from_logits(
                masked_logits, temperature=self.settings.temperature
            )
        sampler_ms = sample_timer.stop()
        return token_id, selected_index, "full", fallback, lm_head_ms, sampler_ms

    def decode(self, request: BenchmarkRequest) -> RequestTrace:
        torch = __import__("torch")
        matcher = self._compile_matcher(request)
        tokenizer = self.bundle.tokenizer
        model = self.bundle.model
        device = self.bundle.device
        sync_cuda = self.settings.sync_cuda and str(device).startswith("cuda")

        trace = RequestTrace(
            request_id=request.request_id,
            model=self.bundle.config.hf_name,
            framework="transformers_xgrammar",
            device=device,
            benchmark=request.benchmark,
            category=request.category,
            head=self.settings.head,
            environment=collect_environment(),
            extra={
                "k_threshold": self.settings.k_threshold,
                "temperature": self.settings.temperature,
                "allowed_provider": self.settings.allowed_provider,
            },
        )

        total_timer = Stopwatch(device=device, sync=sync_cuda)
        prefill_timer = Stopwatch(device=device, sync=sync_cuda)
        total_timer.start()

        try:
            encoded = tokenizer(request.prompt, return_tensors="pt")
            if hasattr(encoded, "to"):
                encoded = encoded.to(device)
            trace.prompt_tokens = int(encoded["input_ids"].shape[-1])

            with torch.no_grad():
                prefill_timer.start()
                outputs = model(**encoded, use_cache=True, output_hidden_states=True)
                trace.latency_ms_prefill = prefill_timer.stop()
                past_key_values = outputs.past_key_values
                hidden = outputs.hidden_states[-1][:, -1, :]

                generated: list[int] = []
                decode_timer = Stopwatch(device=device, sync=sync_cuda)
                decode_timer.start()
                for step in range(self.settings.max_new_tokens):
                    allowed_result = matcher.allowed_token_ids(
                        cached=self.settings.allowed_provider == "cached_bitmask",
                        as_tensor=True,
                        device=self.weight.device,
                    )
                    allowed_ids = allowed_result.allowed_ids
                    k = self._allowed_len(allowed_ids)
                    token_id, selected_index, path, fallback, head_ms, sampler_ms = self._select_token(
                        hidden, allowed_ids
                    )
                    matcher.accept_token(token_id)
                    generated.append(token_id)
                    trace.steps.append(
                        StepTrace(
                            step=step,
                            K=k,
                            K_over_V=k / max(self.bundle.config.vocab_size, 1),
                            transformer_ms=0.0,
                            lm_head_ms=head_ms,
                            mask_or_allowed_ms=allowed_result.provider_ms,
                            bitmask_to_list_ms=allowed_result.bitmask_to_list_ms,
                            sampler_ms=sampler_ms,
                            token_id=token_id,
                            selected_index=selected_index,
                            path=path,
                            provider=self.settings.allowed_provider,
                            fallback=fallback,
                            cache_hit=allowed_result.cache_hit,
                            allowed_ids=self._record_allowed_ids(allowed_ids),
                        )
                    )
                    if matcher.is_terminated():
                        break

                    transformer_timer = Stopwatch(device=device, sync=sync_cuda)
                    transformer_timer.start()
                    next_input = torch.tensor([[token_id]], device=device, dtype=torch.long)
                    outputs = model(
                        input_ids=next_input,
                        past_key_values=past_key_values,
                        use_cache=True,
                        output_hidden_states=True,
                    )
                    transformer_ms = transformer_timer.stop()
                    trace.steps[-1].transformer_ms = transformer_ms
                    past_key_values = outputs.past_key_values
                    hidden = outputs.hidden_states[-1][:, -1, :]

                trace.latency_ms_decode = decode_timer.stop()
                trace.output_tokens = len(generated)
                trace.output_text = tokenizer.decode(generated, skip_special_tokens=False)
                trace.valid = matcher.is_terminated()
        except Exception as exc:
            trace.error = f"{type(exc).__name__}: {exc}"
            trace.valid = False

        trace.latency_ms_total = total_timer.stop()
        if trace.output_tokens:
            trace.latency_ms_per_decode_token = trace.latency_ms_decode / trace.output_tokens
        return trace
