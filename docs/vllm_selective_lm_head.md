# vLLM Selective LM Head Notes

This project can benchmark vLLM's public structured-output path, but a selective
LM head cannot be implemented as an ordinary OpenAI-compatible client feature or
as a post-logits processor. In vLLM 0.16.0 the full vocabulary projection has
already happened before grammar masking runs.

## Verified Baseline

Server used for the smoke run:

```bash
CUDA_VISIBLE_DEVICES=0 /hzeng/miniconda3/envs/vllm==0.16/bin/vllm serve \
  /hzeng/models/Qwen/Qwen2.5-0.5B-Instruct \
  --served-model-name qwen25-0p5b \
  --host 127.0.0.1 \
  --port 18000 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.20 \
  --disable-uvicorn-access-log
```

Trace generation:

```bash
/hzeng/miniconda3/envs/vllm==0.16/bin/python scripts/run_serving_baseline.py \
  --server vllm \
  --base-url http://127.0.0.1:18000/v1 \
  --model qwen25-0p5b \
  --benchmark bfcl \
  --data-path data/repos/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data \
  --categories simple_python \
  --max-requests 5 \
  --output results/traces/vllm_bfcl_qwen25_0p5b_a100.jsonl

/hzeng/miniconda3/envs/vllm==0.16/bin/python scripts/run_serving_baseline.py \
  --server vllm \
  --base-url http://127.0.0.1:18000/v1 \
  --model qwen25-0p5b \
  --benchmark jsonschemabench \
  --data-path data/repos/jsonschemabench/data \
  --splits GlaiveAI-2K \
  --max-requests 3 \
  --output results/traces/vllm_jsonschema_qwen25_0p5b_a100.jsonl
```

Summaries:

```bash
python scripts/analyze_results.py serving-summary \
  --trace results/traces/vllm_bfcl_qwen25_0p5b_a100.jsonl \
  --output results/tables/vllm_bfcl_qwen25_0p5b_a100_summary.csv

python scripts/analyze_results.py serving-summary \
  --trace results/traces/vllm_jsonschema_qwen25_0p5b_a100.jsonl \
  --output results/tables/vllm_jsonschema_qwen25_0p5b_a100_summary.csv
```

Measured on A100 with a warm vLLM server:

| Benchmark | Requests | Valid | Median latency | Median output tokens | Median ms/output token |
| --- | ---: | ---: | ---: | ---: | ---: |
| BFCL `simple_python` | 5 | 5 | 99.38 ms | 41 | 2.41 |
| JSONSchemaBench `GlaiveAI-2K` | 3 | 3 | 177.62 ms | 77 | 2.31 |

## vLLM Entry Points

Relevant local source paths for vLLM 0.16.0:

- `/hzeng/miniconda3/envs/vllm==0.16/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py`
- `/hzeng/miniconda3/envs/vllm==0.16/lib/python3.12/site-packages/vllm/model_executor/layers/logits_processor.py`
- `/hzeng/miniconda3/envs/vllm==0.16/lib/python3.12/site-packages/vllm/v1/structured_output/utils.py`
- `/hzeng/miniconda3/envs/vllm==0.16/lib/python3.12/site-packages/vllm/v1/structured_output/backend_xgrammar.py`

The call order in `GPUModelRunner.execute_model` / `sample_tokens` is:

1. select the hidden states that need logits
2. call `self.model.compute_logits(sample_hidden_states)`
3. apply `apply_grammar_bitmask(...)`
4. call the sampler

`LogitsProcessor._get_logits` computes the full projection with
`lm_head.quant_method.apply(...)`, gathers tensor-parallel logits, then trims the
padding vocab. `StructuredOutputsWorker` then copies the XGrammar bitmask to GPU
and masks the already-computed full logits with a Triton kernel.

## LM Head Profiling

The local vLLM environment also has opt-in profiling instrumentation for the
original full-head path. The patch is saved at
[patches/vllm_0_16_lm_head_profile.diff](../patches/vllm_0_16_lm_head_profile.diff).
It only records timings when `VLLM_LM_HEAD_PROFILE=1` is set and does not enable
the selective prototype.

Example server command:

```bash
unset VLLM_SELECTIVE_LM_HEAD_THRESHOLD
CUDA_VISIBLE_DEVICES=0 \
VLLM_LM_HEAD_PROFILE=1 \
VLLM_LM_HEAD_PROFILE_PATH=results/traces/vllm_lm_head_profile_bfcl_qwen25_0p5b_a100.jsonl \
/hzeng/miniconda3/envs/vllm==0.16/bin/vllm serve \
  /hzeng/models/Qwen/Qwen2.5-0.5B-Instruct \
  --served-model-name qwen25-0p5b \
  --host 127.0.0.1 \
  --port 18000 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.20 \
  --disable-uvicorn-access-log
```

Summary command:

```bash
python scripts/analyze_results.py vllm-lm-head-profile \
  --profile results/traces/vllm_lm_head_profile_bfcl_qwen25_0p5b_a100.jsonl \
  --output results/tables/vllm_lm_head_profile_bfcl_qwen25_0p5b_a100_summary.csv
```

The instrumentation uses CUDA synchronizations around profiled sections, so use
the proportions and per-section timings, not the request wall latency, as the
main signal.

Measured on A100 with real structured-output requests from BFCL
`simple_python` and JSONSchemaBench `GlaiveAI-2K`:

| Model | Benchmark | Phase | Steps | Median forward | Median LM head | LM head / core | LM head / step |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-0.5B | BFCL | decode | 193 | 1.904 ms | 0.216 ms | 7.81% | 5.76% |
| Qwen2.5-0.5B | BFCL | prefill | 5 | 3.153 ms | 0.233 ms | 2.66% | 2.30% |
| Qwen2.5-0.5B | JSONSchemaBench | decode | 225 | 1.903 ms | 0.214 ms | 7.90% | 5.81% |
| Qwen2.5-0.5B | JSONSchemaBench | prefill | 3 | 3.231 ms | 0.272 ms | 6.25% | 4.68% |
| Qwen2.5-3B | BFCL | decode | 148 | 6.388 ms | 0.395 ms | 5.31% | 4.65% |
| Qwen2.5-3B | BFCL | prefill | 5 | 7.875 ms | 0.419 ms | 2.85% | 2.59% |
| Qwen2.5-3B | JSONSchemaBench | decode | 208 | 6.389 ms | 0.396 ms | 5.34% | 4.67% |
| Qwen2.5-3B | JSONSchemaBench | prefill | 3 | 7.788 ms | 0.463 ms | 4.58% | 3.98% |

For these workloads the decode phase is the more relevant target: it repeats
many more times, and the LM head accounts for roughly 5-8% of the measured core
GPU path. Prefill has only one logits row per request in normal generation, so
the LM-head share is usually lower and can be dominated by first-step structured
output setup.

### Concurrency Profiling

Tensor parallelism was intentionally not used for the few-billion-parameter
models in this repo because it would mostly measure extra communication and
scheduling overhead. Instead, the serving pressure test below uses TP=1 and
increases OpenAI-compatible request concurrency.

Concurrent runner:

```bash
/hzeng/miniconda3/envs/vllm==0.16/bin/python scripts/run_serving_concurrent.py \
  --server vllm \
  --base-url http://127.0.0.1:18000/v1 \
  --model qwen25-3b \
  --benchmark bfcl \
  --data-path data/repos/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data \
  --categories simple_python \
  --max-requests 16 \
  --concurrency 8 \
  --temperature 0 \
  --output results/traces/vllm_concurrent_bfcl_qwen25_3b_tp1_c8.jsonl
```

Measured on Qwen2.5-3B, A100, TP=1:

| Benchmark | Concurrency | Phase | Steps | Median reqs/step | Max reqs/step | Median LM head | LM head / core | LM head / step |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BFCL | 1 | decode | 500 | 1 | 1 | 0.397 ms | 5.34% | 4.66% |
| BFCL | 4 | decode | 139 | 4 | 4 | 0.404 ms | 3.86% | 3.46% |
| BFCL | 8 | decode | 85 | 6 | 8 | 0.410 ms | 5.77% | 4.89% |
| JSONSchemaBench | 1 | decode | 705 | 1 | 1 | 0.398 ms | 5.37% | 4.68% |
| JSONSchemaBench | 4 | decode | 200 | 4 | 4 | 0.404 ms | 5.71% | 4.86% |
| JSONSchemaBench | 8 | decode | 121 | 7 | 8 | 0.411 ms | 5.78% | 4.87% |

Increasing concurrency clearly reduces the number of runner steps and increases
the effective batch size, but it does not make the LM head a dominant cost for
this model size. The measured decode LM-head share stays around 4-6% of the
runner step and 4-6% of the profiled core path. This narrows the practical target
for selective LM head: it is unlikely to be compelling for single-GPU serving of
3B-class models unless the implementation overhead is extremely small.

### Quantized And MoE Profiling

The local model matrix uses the existing `/hzeng/models` root. No symlinks are
needed.

Downloaded or already-present model paths:

| Model | Path | Size |
| --- | --- | ---: |
| Qwen2.5-0.5B-Instruct | `/hzeng/models/Qwen/Qwen2.5-0.5B-Instruct` | 954 MB |
| Qwen2.5-0.5B-Instruct-AWQ | `/hzeng/models/Qwen/Qwen2.5-0.5B-Instruct-AWQ` | 708 MB |
| Qwen2.5-0.5B-Instruct-GPTQ-Int4 | `/hzeng/models/Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4` | 450 MB |
| Qwen2.5-1.5B-Instruct | `/hzeng/models/Qwen/Qwen2.5-1.5B-Instruct` | 2.9 GB |
| Qwen2.5-1.5B-Instruct-AWQ | `/hzeng/models/Qwen/Qwen2.5-1.5B-Instruct-AWQ` | 2.1 GB |
| Qwen2.5-1.5B-Instruct-GPTQ-Int4 | `/hzeng/models/Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4` | 1.1 GB |
| Qwen2.5-3B-Instruct | `/hzeng/models/Qwen/Qwen2.5-3B-Instruct` | 5.8 GB |
| Qwen2.5-3B-Instruct-AWQ | `/hzeng/models/Qwen/Qwen2.5-3B-Instruct-AWQ` | 2.6 GB |
| Qwen2.5-3B-Instruct-GPTQ-Int4 | `/hzeng/models/Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4` | 2.0 GB |
| Qwen1.5-MoE-A2.7B-Chat | `/hzeng/models/Qwen/Qwen1.5-MoE-A2.7B-Chat` | 27 GB |
| Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4 | `/hzeng/models/Qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4` | 7.9 GB |
| Qwen3-30B-A3B-Instruct-2507 | `/hzeng/models/Qwen/Qwen3-30B-A3B-Instruct-2507` | 57 GB |

Local checkpoint inspection shows that these common Qwen GPTQ/AWQ checkpoints do
not quantize `lm_head` to int4:

- Qwen2.5 AWQ has float16 `lm_head.weight`.
- Qwen2.5 GPTQ ties the head to float16 `model.embed_tokens.weight`.
- Qwen1.5 MoE GPTQ has float16 `lm_head.weight`.
- Dense MoE checkpoints keep `lm_head.weight` in bfloat16.

The inspection outputs are saved in
`results/tables/lm_head_quantization_inspection_qwen25.json` and
`results/tables/lm_head_quantization_inspection_moe.json`.

BFCL `simple_python`, A100, TP=1, request concurrency 1:

| Model | Variant | Decode steps | Median forward | Median LM head | LM head / core | LM head / step |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-0.5B | dense | 193 | 1.904 ms | 0.216 ms | 7.81% | 5.76% |
| Qwen2.5-0.5B | AWQ | 620 | 2.278 ms | 0.217 ms | 6.95% | 5.27% |
| Qwen2.5-0.5B | GPTQ | 637 | 2.176 ms | 0.219 ms | 7.38% | 5.43% |
| Qwen2.5-3B | dense | 148 | 6.388 ms | 0.395 ms | 5.31% | 4.65% |
| Qwen2.5-3B | AWQ | 532 | 4.073 ms | 0.395 ms | 7.76% | 6.44% |
| Qwen2.5-3B | GPTQ | 544 | 3.919 ms | 0.400 ms | 8.04% | 6.60% |
| Qwen1.5-MoE-A2.7B | BF16 | 557 | 4.609 ms | 0.396 ms | 7.02% | 5.88% |
| Qwen1.5-MoE-A2.7B | GPTQ | 576 | 3.415 ms | 0.402 ms | 8.96% | 7.17% |

The quantized and MoE cases raise the relative LM-head share, especially for the
3B and MoE GPTQ cases, because the transformer block path gets cheaper while the
float LM head stays about the same absolute cost. Even in the best case measured
here, the LM head is still only about 7% of the runner step. That makes a
vLLM-level selective LM-head optimization hard to justify unless a future target
has a much smaller transformer path, a slower edge GPU/CPU head projection, or a
sampler implementation that can consume compact selected logits without
scattering back to a full vocabulary tensor.

## Expected Effect

Selective LM head can still help inside vLLM only if it is inserted before
`compute_logits`, because post-logits hooks already paid the full LM-head matmul
cost. The public structured-output API is therefore useful as a baseline, not as
the optimization point.

A practical first vLLM prototype should be intentionally narrow:

1. V1 engine only.
2. tensor parallel size 1.
3. structured-output requests only.
4. greedy or ordinary sampling without logprobs.
5. fallback to `compute_logits` when `K` exceeds a threshold or a batch mixes
   constrained and unconstrained rows.

The easiest compatibility prototype is to compute selected logits for the
allowed token IDs, scatter them into a full logits tensor initialized to
`-inf`, and then reuse vLLM's existing sampler. This still allocates the full
logits tensor, but it removes the full-vocab matrix multiply, which is the part
the selective LM head is meant to test. A production patch would also need a
sampler path over compact selected logits, tensor-parallel handling, logprob
support, batching policy, and CUDA/Triton kernels to avoid Python-side K
materialization.

Given the controlled A100 results in this repo, the end-to-end win is expected
to be small for 0.5B-3B models and short structured outputs unless `K` is very
small for most steps or the batch is dominated by constrained rows. The head-only
microbenchmark still shows the local projection can be several times faster, so
the remaining question is whether that local gain is large enough relative to
vLLM's transformer, scheduler, sampler, and serving overheads on the target
workload.

## Prototype Result

An experimental patch for vLLM 0.16.0 is saved at
[patches/vllm_0_16_selective_lm_head_prototype.diff](../patches/vllm_0_16_selective_lm_head_prototype.diff).
It was also applied to the local `/hzeng/miniconda3/envs/vllm==0.16`
environment for the smoke run. The patch is disabled by default and only turns
on when `VLLM_SELECTIVE_LM_HEAD_THRESHOLD` is set.

The prototype defers `compute_logits` until `sample_tokens`, where the grammar
bitmask is available. It supports only TP=1, no speculative decode, unquantized
LM head, and batches where every logits row has a structured-output mask.

Smoke command:

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_SELECTIVE_LM_HEAD_THRESHOLD=8192 \
  /hzeng/miniconda3/envs/vllm==0.16/bin/vllm serve \
  /hzeng/models/Qwen/Qwen2.5-0.5B-Instruct \
  --served-model-name qwen25-0p5b \
  --host 127.0.0.1 \
  --port 18000 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.20 \
  --disable-uvicorn-access-log
```

Measured against the unpatched serving baseline:

| Benchmark | Baseline median | Prototype median | Paired speedup |
| --- | ---: | ---: | ---: |
| BFCL `simple_python` | 99.38 ms | 140.04 ms | 0.72x |
| JSONSchemaBench `GlaiveAI-2K` | 177.62 ms | 252.33 ms | 0.69x |

The output text and JSON validity matched the baseline in these smoke runs.
Logs confirmed the selective path was used for many steps with average K around
200 after warmup, while free-string regions fell back when K exceeded the 8192
threshold. This prototype is therefore a useful correctness probe but not a
performance win: the Python/NumPy bit unpacking, per-step `index_select`, and
scatter into full logits are slower than vLLM's optimized full-head plus
XGrammar bitmask path for these small real-data samples.
