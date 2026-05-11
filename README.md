# Selective LM Head

Trace-first experiments for constrained decoding with a selective LM head.

## Project Status

Status as of 2026-05-11: closed as a negative result.

The main hypothesis was that structured decoding often permits only a small
token set, so computing only selected LM-head logits could produce a meaningful
serving speedup. Profiling on the target vLLM path did not support that as a
useful project direction:

- In original vLLM structured decoding, the full LM head is already only about
  `5%-8%` of the measured decode core path for Qwen2.5 0.5B-3B on A100.
- TP=1 concurrency profiling did not make the LM head dominant; it stayed around
  `4%-6%` of the runner step for Qwen2.5-3B.
- Quantized and MoE cases raise the relative share because the transformer block
  path gets cheaper while `lm_head` usually stays float, but the best measured
  case here, Qwen1.5-MoE-A2.7B-GPTQ, was still only `8.96%` of core and `7.17%`
  of runner step.
- The vLLM prototype that moved structured-output masking before full logits was
  slower than baseline (`0.72x` on BFCL, `0.69x` on JSONSchemaBench), because the
  Python/NumPy bit unpacking, per-step selection, and scatter back to full logits
  outweighed the saved matmul.

The practical conclusion is that a vLLM selective LM-head optimization is not
worth continuing for the current single-A100, 0.5B-3B, structured-decoding
serving scenario. This repository is kept as an experiment record with runnable
profiling scripts, patches, downloaded-model notes, and result tables. Future
work would need a different bottleneck profile, such as CPU serving, a weaker
edge GPU, or a sampler that consumes compact selected logits directly without
materializing full-vocabulary logits.

CPU was checked as a last possible direction. On the controlled Transformers +
XGrammar CPU path with `OMP_NUM_THREADS=4` and `MKL_NUM_THREADS=4`, Qwen2.5-0.5B
reaches about `15.6%-15.7%` LM-head share of the measured decode core on BFCL
and JSONSchemaBench. The same BFCL sample had a selective/full median speedup of
about `1.16x` with identical outputs. The share decreases with model size:
Qwen2.5-1.5B measured `11.59%`, and Qwen2.5-3B measured `8.39%`. This is a
stronger signal than A100/vLLM, but still not enough to justify continuing the
project: it is limited to a small controlled CPU path, the speedup ceiling is
modest, and production value would require a separate compact-logits sampling
stack. This direction is therefore also closed.

The design document is [selective_lm_head_project_design_full.md](selective_lm_head_project_design_full.md). This repository implemented the project in the same order:

1. environment setup and fixed configs
2. serving structured-output baseline clients
3. controlled Transformers + XGrammar full-head baseline
4. per-step trace logging with `K = |allowed_ids|`
5. selective LM head with threshold fallback
6. head-only replay and threshold sweep
7. analysis tables for K distribution, latency breakdown, and paired speedup

## Environment

The host default Python is 3.13, but this project is configured for Python 3.10 because the ML stack is more stable there.

```bash
cd /hzeng/prj/selective-lm-head

# Lightweight dev environment, enough for package import and unit tests.
bash scripts/setup_env.sh dev
source .venv/bin/activate

# CPU ML environment, enough for real-data controlled decode smoke tests.
bash scripts/setup_env.sh cpu-ml
source .venv/bin/activate

# Full local ML environment for A100 model runs.
bash scripts/setup_env.sh ml
source .venv/bin/activate
```

Installed local model paths are declared in [configs/models.yaml](configs/models.yaml). The default root is `/hzeng/models`.

The `ml` profile pins PyTorch to CUDA 12.4 (`torch==2.6.0+cu124`) for the current 550-series driver. If your network is slow, use `cpu-ml` first to verify the full project flow, then rerun `ml` before A100 experiments.

## Datasets

```bash
bash scripts/download_datasets.sh

python scripts/test_real_dataset_loading.py \
  --max-requests 10 \
  --output results/tables/real_dataset_loading.json
```

Downloaded paths:

- BFCL V4: `data/repos/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data`
- JSONSchemaBench: `data/repos/jsonschemabench/data`

## Smoke Checks

```bash
python -m unittest discover -s tests
python scripts/analyze_results.py --help
python scripts/run_baseline.py --help
python scripts/replay_head.py --help
```

## Controlled Full-Head Baseline

```bash
python scripts/run_baseline.py \
  --model qwen25_0p5b \
  --benchmark jsonschemabench \
  --data-path data/repos/jsonschemabench/data \
  --splits GlaiveAI-2K \
  --device cuda \
  --max-requests 5 \
  --max-new-tokens 128 \
  --output results/traces/qwen25_0p5b_jsonschema_full_a100.jsonl
```

## End-to-End Selective Decode

```bash
python scripts/run_selective.py \
  --model qwen25_0p5b \
  --benchmark jsonschemabench \
  --data-path data/repos/jsonschemabench/data \
  --splits GlaiveAI-2K \
  --device cuda \
  --k-threshold 2048 \
  --max-requests 5 \
  --max-new-tokens 128 \
  --output results/traces/qwen25_0p5b_jsonschema_selective_a100.jsonl
```

## Head-Only Replay

```bash
python scripts/replay_head.py \
  --trace results/traces/qwen25_0p5b_jsonschema_full_a100.jsonl \
  --model qwen25_0p5b \
  --device cuda \
  --thresholds 1,8,32,128,512,2048,8192 \
  --output results/tables/head_replay_qwen25_0p5b_a100.csv
```

## Analysis

```bash
python scripts/analyze_results.py k-distribution \
  --trace results/traces/qwen25_0p5b_jsonschema_full_a100.jsonl \
  --output results/tables/k_distribution.csv

python scripts/analyze_results.py latency-breakdown \
  --trace results/traces/qwen25_0p5b_jsonschema_full_a100.jsonl \
  --output results/tables/latency_breakdown.csv

python scripts/analyze_results.py paired-speedup \
  --baseline results/traces/qwen25_0p5b_jsonschema_full_a100.jsonl \
  --selective results/traces/qwen25_0p5b_jsonschema_selective_a100.jsonl \
  --output results/tables/paired_speedup.csv

python scripts/analyze_results.py serving-summary \
  --trace results/traces/vllm_jsonschema_qwen25_0p5b_a100.jsonl \
  --output results/tables/vllm_jsonschema_qwen25_0p5b_a100_summary.csv

python scripts/analyze_results.py vllm-lm-head-profile \
  --profile results/traces/vllm_lm_head_profile_jsonschema_qwen25_0p5b_a100.jsonl \
  --output results/tables/vllm_lm_head_profile_jsonschema_qwen25_0p5b_a100_summary.csv
```

## vLLM Baseline

vLLM structured output is supported through the OpenAI-compatible serving runner:

```bash
/hzeng/miniconda3/envs/vllm==0.16/bin/vllm serve \
  /hzeng/models/Qwen/Qwen2.5-0.5B-Instruct \
  --served-model-name qwen25-0p5b \
  --host 127.0.0.1 \
  --port 18000 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.20 \
  --disable-uvicorn-access-log

/hzeng/miniconda3/envs/vllm==0.16/bin/python scripts/run_serving_baseline.py \
  --server vllm \
  --base-url http://127.0.0.1:18000/v1 \
  --model qwen25-0p5b \
  --benchmark bfcl \
  --data-path data/repos/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data \
  --categories simple_python \
  --max-requests 5 \
  --output results/traces/vllm_bfcl_qwen25_0p5b_a100.jsonl
```

Implementation notes, the vLLM 0.16 prototype patch, and measured results are in
[docs/vllm_selective_lm_head.md](docs/vllm_selective_lm_head.md).

## Verified Smoke Results

On this machine, `cpu-ml` was used to run real-data smoke tests with `Qwen2.5-0.5B-Instruct`, CPU, `fp32`, `OMP_NUM_THREADS=4`:

- BFCL `simple_python_0`, 23 output tokens: full `2055.42 ms`, selective `1799.08 ms`, speedup `1.14x`, identical output, grammar terminated.
- JSONSchemaBench `Glaiveai2K/analyze_health_data_4ad104b4`, 32-token cap: full `2625.49 ms`, selective `2363.41 ms`, speedup `1.11x`, identical prefix; grammar did not terminate because the cap stopped inside a JSON array.
- Head-only replay on BFCL trace: all checked rows had matching selected logits and matching argmax.

GPU smoke notes after tensorized allowed-id conversion:

- A100 BFCL `simple_python`, Qwen2.5-3B, 3 requests: grammar/list time dropped from `864.24 ms` to `44.50 ms`; decode time dropped from `2471.93 ms` to `1628.97 ms`.
- A100 JSONSchemaBench `GlaiveAI-2K`, Qwen2.5-0.5B, 3 requests: grammar/list time dropped from `1117.62 ms` to `84.46 ms`; decode time dropped from `4696.79 ms` to `3387.69 ms`.
- Selective vs full remains close on A100 for these smoke runs: `~1.00x-1.01x`, because both paths now avoid the old Python list bottleneck and LM head is only a small fraction of end-to-end decode time in this prototype.
- vLLM 0.16.0 structured-output serving baseline, Qwen2.5-0.5B on A100: BFCL `simple_python` median `99.38 ms` for 5 valid requests, median `2.41 ms/output token`; JSONSchemaBench `GlaiveAI-2K` median `177.62 ms` for 3 valid requests, median `2.31 ms/output token`.
- vLLM original full-head profiling with CUDA synchronization instrumentation on real BFCL/JSONSchemaBench samples: decode LM-head share is `7.81%-7.90%` of the measured core GPU path on Qwen2.5-0.5B and `5.31%-5.34%` on Qwen2.5-3B; prefill is generally lower because only one logits row is projected per request.
- vLLM TP=1 concurrency profiling on Qwen2.5-3B with real BFCL/JSONSchemaBench samples: raising request concurrency from `1` to `8` increases effective decode batch size to roughly `6-7` rows/step, but LM-head share remains around `4%-6%` of the runner step.
- Qwen2.5 AWQ/GPTQ and Qwen1.5 MoE GPTQ checkpoint inspection shows the LM head is not int4 quantized: AWQ and MoE GPTQ keep float `lm_head.weight`, while Qwen2.5 GPTQ ties the head to the float embedding weight. The BFCL vLLM profiling matrix across dense, AWQ, GPTQ, and MoE variants is saved at `results/tables/vllm_lm_head_profile_bfcl_model_matrix_a100.csv`; the highest measured decode LM-head share is Qwen1.5-MoE-A2.7B-GPTQ at `8.96%` of core and `7.17%` of runner step.
- vLLM 0.16.0 selective LM-head prototype with `VLLM_SELECTIVE_LM_HEAD_THRESHOLD=8192`: BFCL median `140.04 ms`, paired speedup `0.72x`; JSONSchemaBench median `252.33 ms`, paired speedup `0.69x`. Outputs matched baseline, but this Python/NumPy prototype is slower than vLLM's optimized full-head + bitmask path.

## Notes

The core Python modules use lazy imports for Torch, Transformers, XGrammar, pandas, and matplotlib. This keeps the project testable before the heavyweight ML environment is installed, while the runtime scripts fail with actionable messages when an optional dependency is missing.
