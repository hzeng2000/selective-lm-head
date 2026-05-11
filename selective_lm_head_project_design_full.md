# 项目设计：Constrained Decoding 场景下的 Selective LM Head 加速

> 版本：2026-05-09  
> 优化主题：Selective LM Head  
> 实验形态：batch=1、单请求延迟评估  
> 首轮设备：A100 单卡机器 + 同机 CPU  
> 模型范围：Dense 0.5B / 1.5B / 3B 与 A3B-class MoE  
> 主实验框架：vLLM/SGLang serving baseline + PyTorch/Transformers/XGrammar 可控实现

---

## 0. 项目摘要

结构化输出、工具调用、JSON schema 生成、函数调用等 constrained decoding 场景中，常见执行路径是：模型先通过 LM head 计算完整词表 logits，再由 grammar / schema / regex 约束将非法 token 屏蔽，最后在合法 token 集合上进行 greedy 或 sampling。

Selective LM Head 的核心思想是利用 grammar state 已经给出的合法 token 集合 `A`：在当前 decode step 只读取 `A` 对应的 LM head rows，只计算合法 token 的 logits，并在合法集合上完成 argmax / softmax / sampling。

项目采用 trace-first 的实验流程：

1. 在 BFCL 与 JSONSchemaBench 上运行 full-head constrained baseline；
2. 记录每个 request 的端到端延迟、每个 decode step 的 `K = |A|`、LM head 耗时、grammar/list 耗时、sampler 耗时；
3. 基于真实 trace 做 head-only replay，验证 selective head 的局部收益与数值一致性；
4. 接入 end-to-end selective decoding，比较 request-level latency；
5. 在 dense 小模型与 A3B-class MoE 上完成统一实验矩阵。

---

## 1. 背景与动机

### 1.1 当前 constrained decoding 的典型路径

主流 structured output / guided decoding 流程可以抽象为：

```text
hidden state h
    -> full-vocab LM head: logits over V
    -> grammar / schema / regex mask invalid tokens
    -> softmax / argmax / sampling over valid tokens
    -> accept token and advance grammar state
```

公开文档中的相关证据：

- vLLM 的 structured outputs 文档展示了 `choice`、`regex`、`json`、`grammar` 等结构化输出模式，并说明 structured outputs 使用 xgrammar 或 guidance 作为 backend。[S1]
- vLLM 的 logits processor 文档描述，processor 消费 `(num_requests) x (vocab_size)` 的 raw logits，并把处理后的 logits 传给 softmax。[S2]
- SGLang structured outputs 支持 JSON schema、regex、EBNF，并支持 XGrammar、Outlines、llguidance 三类 grammar backend，默认使用 XGrammar。[S3]
- XGrammar workflow 与 engine integration 文档描述，每个 autoregressive step 填充 token bitmask，并把 bitmask apply 到模型 logits 上；非法 token logits 被设为 negative infinity。[S4][S5]

因此，本项目关注的实际执行形态是：

```text
full logits first -> mask invalid tokens -> sample from valid tokens
```

### 1.2 项目场景

本项目聚焦 batch=1 的单请求 constrained decoding latency。首轮实验在 A100 单卡机器上完成，同时使用该机器 CPU 进行 CPU 路径测量。

实验覆盖两类 active-compute 较小、LM head 仍需完整词表投影的模型：

```text
Dense small models:
    Qwen2.5-0.5B-Instruct
    Qwen2.5-1.5B-Instruct
    Qwen2.5-3B-Instruct

A3B-class MoE models:
    Qwen3-30B-A3B-Instruct-2507
    Qwen3-30B-A3B quantized variants
```

结构化输出中存在大量小合法集合位置，例如：

```text
JSON punctuation: { } [ ] : , "
fixed keys: "name", "arguments", "location"
tool names: get_weather, search_file, send_email
argument names: city, date, unit, limit
enum values: celsius, fahrenheit, high, low
boolean/null: true, false, null
```

这些 step 的 `K` 远小于完整 vocab，具备 selective LM head 的计算空间。

### 1.3 MoE 实验动机

A3B-class MoE 每 token 激活的中间层参数规模接近 dense 3B，但输出层仍面对完整词表。以 Qwen3-30B-A3B-Instruct-2507 为例，官方 model card 写明其总参数约 30.5B、激活参数约 3.3B、48 层、128 experts、每 token 激活 8 experts。[S12]

Qwen3-30B-A3B-Instruct-2507 config 可见 `hidden_size=2048`、`num_experts=128`、`num_experts_per_tok=8` 等配置。[S13] 若使用 `vocab_size=151936` 与 `hidden_size=2048` 估算 LM head 规模：

```text
151,936 * 2,048 ≈ 311M parameters
FP16/BF16 full-head weight read ≈ 622 MB / token
INT4 full-head weight read ≈ 155 MB / token, ignoring metadata and dequant overhead
```

Dense Qwen2.5-3B 与 A3B MoE 的 hidden size 和 vocab size 量级接近，因此二者在 selective LM head 评估中具有直接可比性。

---

## 2. 项目目标

### 2.1 目标 A：真实任务 full-head constrained baseline

在开源 benchmark 上运行 baseline，记录 request-level 与 step-level 指标：

```text
request end-to-end latency
prefill latency
decode latency
per-token decode latency
full LM head latency
grammar mask / allowed set latency
sampler latency
output token count
correctness / validity
```

### 2.2 目标 B：合法 token 集合分析

在真实任务上记录每个 decode step 的合法 token 数：

```text
K = |allowed token set at current grammar state|
K / vocab_size
K bucket: 1, <=8, <=32, <=128, <=512, <=2048, <=8192, full-like
```

该分析用于决定 selective head 的启用阈值与 workload 覆盖率。

### 2.3 目标 C：Selective LM Head 实现

将 baseline 的 full-head 路径：

```text
logits = h @ W_vocab.T
logits[invalid] = -inf
token = sample_or_argmax(logits)
```

替换为 selective 路径：

```text
allowed_ids = grammar.allowed_token_ids()
selected_logits = h @ W_vocab[allowed_ids].T
token = sample_or_argmax(selected_logits)
map selected index back to token id
```

当 `K` 超过经验阈值时使用 full-head constrained 路径。

### 2.4 目标 D：Dense + MoE 统一评估

在同一套 harness、benchmark、trace schema 和指标下评估：

```text
Qwen2.5-0.5B-Instruct
Qwen2.5-1.5B-Instruct
Qwen2.5-3B-Instruct
Qwen3-30B-A3B-Instruct-2507 / quantized variant
```

每个模型记录：

```text
full-head constrained baseline latency
selective-head constrained latency
speedup by request
speedup by decode token
speedup by K bucket
LM head time ratio
grammar/list time ratio
validity / BFCL correctness
```

---

## 3. 核心研究问题

### RQ1：真实 constrained decoding 任务里，full LM head 占 decode latency 的比例是多少？

对比维度：

```text
模型：dense 0.5B / 1.5B / 3B / A3B MoE
设备：A100 GPU / 同机 CPU
任务：BFCL / JSONSchemaBench
阶段：prefill / decode / LM head / grammar-list / sampler
```

### RQ2：真实任务中的 `K` 分布是什么样？

重点报告 bucket 分布：

```text
K = 1
2 <= K <= 8
9 <= K <= 32
33 <= K <= 128
129 <= K <= 512
513 <= K <= 2048
2049 <= K <= 8192
K > 8192
```

### RQ3：Selective LM Head 在 request-level latency 上带来多少收益？

主指标：

```text
speedup = baseline_request_latency / selective_request_latency
```

辅助指标：

```text
ms / generated token
ms / decode step
head-local speedup by K bucket
fallback rate by threshold
```

### RQ4：A3B MoE 与 dense 3B 的 selective head 收益如何对齐？

分析点：

```text
LM head 参数量
LM head latency ratio
transformer / expert latency ratio
selective head 局部收益
request-level speedup
```

### RQ5：Allowed set provider 对收益的影响有多大？

比较三种 allowed set 获取路径：

```text
bitmask-to-list
cached allowed_ids
direct allowed_ids from trie / grammar state
```

---

## 4. Benchmark 设计

### 4.1 主任务一：BFCL，Berkeley Function Calling Leaderboard

BFCL 用于评估工具调用 / function calling 场景。官方页面称 BFCL V4 评估模型准确调用 functions/tools 的能力，数据由 real-world data 构成，并会周期性更新。[S6] BFCL repo 将其描述为 comprehensive and executable function call evaluation，覆盖多种 function call 形式、场景和可执行性。[S7]

首轮选择 categories：

```text
simple_python
simple_java
simple_javascript
multiple
parallel
irrelevance
live_simple
```

BFCL 在本项目中的角色：

| 项 | 说明 |
|---|---|
| 场景 | 工具调用 / function calling / local agent action |
| 输入 | BFCL 官方 prompt + function definitions |
| 输出 | function call / tool call JSON 或 AST-compatible 格式 |
| 约束 | 从 function definitions 自动生成 grammar / JSON schema / token constraints |
| 指标 | request latency + BFCL correctness |
| 价值 | 覆盖真实工具调用任务 |

本项目使用 BFCL 官方 evaluator 计算 correctness，同时使用自定义 step-by-step harness 记录 LM head、mask/list、sampler 和 `K`。

### 4.2 主任务二：JSONSchemaBench

JSONSchemaBench 是 structured output / constrained decoding benchmark。官方 repo 描述其包含约 10,000 个真实 JSON schemas，用于评估 structured output generation，并覆盖多样约束和复杂度。[S8]

JSONSchemaBench repo 说明数据来源包括 GitHub、Kubernetes configurations、API specifications 等，并将 schemas 按 complexity 和 domain 分类。[S8]

首轮选择 splits：

```text
GlaiveAI-2K Function Call
Kubernetes
Github-Easy
Github-Medium
Github-Hard
```

JSONSchemaBench 在本项目中的角色：

| 项 | 说明 |
|---|---|
| 场景 | 真实 schema / typed extraction / config generation / API schema |
| 输入 | JSONSchemaBench 官方 schema |
| 输出 | 符合 schema 的 JSON object |
| 约束 | JSON schema constrained decoding |
| 指标 | request latency、validity、`K` 分布、schema complexity bucket |
| 价值 | 覆盖函数调用之外的大量 schema 结构 |

统一运行模板：

```text
Generate one JSON object that satisfies the given JSON Schema. Return only JSON.
```

### 4.3 补充任务：Intent Router / Classifier

补充任务用于构造稳定的低 `K` constrained generation 场景。可选数据集：

```text
BANKING77
CLINC150
```

输出格式：

```json
{"intent": "<one_of_labels>"}
```

该任务用于补充 router / classifier 场景，并为 selective head 提供固定 label trie 的 direct allowed_ids 路径。

---

## 5. 实验框架

### 5.1 两套互补框架

本项目使用两套互补实验框架。

第一套是官方 structured-output serving baseline。使用 vLLM 或 SGLang 运行 BFCL 和 JSONSchemaBench，记录 batch=1 下每个 request 的端到端延迟、输出长度、正确率和 schema validity。该 baseline 用于反映主流 structured-output 推理框架的实际表现。

第二套是可控 selective LM head 实验框架。使用 PyTorch + Hugging Face Transformers 执行模型 forward，并使用 XGrammar Python API 作为 structured decoding backend。XGrammar 负责 JSON schema / grammar 编译、grammar state 维护、token bitmask 生成和 token 接受检查。实验 harness 在每个 decode step 中记录 transformer forward 时间、LM head 时间、grammar mask 时间、sampling 时间、allowed token 数 `K` 和 request-level latency。

Selective LM head 在第二套框架中实现。baseline 路径计算 full-vocab LM head 后应用 XGrammar bitmask；optimized 路径先根据 XGrammar bitmask 或 trie state 得到 allowed token set，再只对 allowed token rows 计算 LM head logits，并在 allowed token set 上完成 argmax 或 sampling。

### 5.2 框架分工

| 层级 | 框架 / 工具 | 作用 |
|---|---|---|
| Serving baseline | vLLM / SGLang | 运行官方 structured-output API，获得实际 serving latency |
| 可控模型执行 | PyTorch + Transformers | 加载 dense 与 MoE 模型，手写 step-by-step decode loop |
| Structured backend | XGrammar Python API | JSON schema / grammar 编译、matcher 状态、bitmask 生成 |
| Function-call adapter | BFCL adapter + trie provider | 从 function definitions 生成 JSON schema / grammar / allowed_ids |
| Evaluation | BFCL official evaluator | 计算 function calling correctness |
| JSON validation | jsonschema | 检查 JSON output 与 schema validity |
| Timing | CUDA Event + perf_counter_ns | A100 与 CPU 的 request-level / step-level timing |
| Analysis | pandas + matplotlib | `K` 分布、latency breakdown、paired speedup |

### 5.3 Decode 路径

Full-head constrained path：

```text
Transformers model forward
    -> hidden state h_t
    -> lm_head over full vocab
    -> XGrammar bitmask
    -> masked logits
    -> argmax / sample
    -> matcher.accept(token)
```

Selective-head constrained path：

```text
Transformers model forward
    -> hidden state h_t
    -> allowed_ids from bitmask / cache / trie
    -> lm_head over W[allowed_ids]
    -> argmax / sample over allowed_ids
    -> matcher.accept(token)
```

Serving baseline path：

```text
vLLM or SGLang server
    -> structured_outputs / json_schema / regex / grammar request
    -> OpenAI-compatible API response
    -> latency + correctness logging
```

### 5.4 XGrammar 接入方式

XGrammar 提供以下对象和方法：

```text
TokenizerInfo.from_huggingface(tokenizer, vocab_size)
GrammarCompiler(tokenizer_info)
compiler.compile_json_schema(json_schema_string)
compiler.compile_builtin_json_grammar()
GrammarMatcher(compiled_grammar)
allocate_token_bitmask(batch_size, vocab_size)
matcher.fill_next_token_bitmask(bitmask)
apply_token_bitmask_inplace(logits, bitmask)
matcher.accept_token(token_id)
matcher.is_terminated()
```

可控实验中的 baseline 使用 XGrammar 的 bitmask apply 路径；selective 实现使用同一个 matcher state 产生当前 step 的合法 token 信息。

---

## 6. 模型矩阵

### 6.1 Dense 模型

主 dense 模型使用 Qwen2.5-Instruct 系列：

| 模型 | hidden size | vocab size | LM head row 参数量 | FP16/BF16 full-head 读权重量级 | 角色 |
|---|---:|---:|---:|---:|---|
| Qwen2.5-0.5B-Instruct | 896 | 151,936 | 约 136M | 约 272 MB / token | 小型 dense |
| Qwen2.5-1.5B-Instruct | 1536 | 151,936 | 约 233M | 约 467 MB / token | 中型 dense |
| Qwen2.5-3B-Instruct | 2048 | 151,936 | 约 311M | 约 622 MB / token | dense 上限 |

Hugging Face config 页面可见 Qwen2.5-0.5B-Instruct 的 `hidden_size=896` 与 `vocab_size=151936`，Qwen2.5-1.5B-Instruct 的 `hidden_size=1536`，Qwen2.5-3B-Instruct 的 `hidden_size=2048` 与 `vocab_size=151936`。[S9][S10][S11]

Qwen2.5 dense 模型通常 tied embedding，即 input embedding 和 output embedding 共享权重。Selective head 可直接从 shared embedding weight 中取 `allowed_ids` 对应 rows。

### 6.2 MoE 模型

MoE 主模型：

```text
Qwen3-30B-A3B-Instruct-2507
Qwen3-30B-A3B-GPTQ-Int4 / AWQ / 其他可用量化版本
```

MoE 记录项：

```text
total parameters
activated parameters
num_experts
num_experts_per_token
hidden_size
vocab_size
lm_head dtype
model quantization
peak GPU memory
```

Qwen3-30B-A3B-Instruct-2507 model card 写明：30.5B total parameters、3.3B activated parameters、48 层、128 experts、每 token 激活 8 experts。[S12] Qwen3-30B-A3B-Instruct-2507 config 可见 `hidden_size=2048`、`model_type=qwen3_moe`、`num_experts=128`、`num_experts_per_tok=8` 等配置。[S13]

### 6.3 模型运行优先级

| 优先级 | 模型 | A100 GPU | 同机 CPU | 说明 |
|---|---|---:|---:|---|
| P0 | Qwen2.5-0.5B-Instruct | ✓ | ✓ | 快速迭代、完整 trace |
| P0 | Qwen2.5-1.5B-Instruct | ✓ | ✓ | 主力 dense 模型 |
| P0 | Qwen2.5-3B-Instruct | ✓ | ✓ | dense 上限 |
| P0 | Qwen3-30B-A3B-Instruct-2507 / quantized | ✓ | ✓ | MoE 主模型 |
| P1 | Llama-3.2-3B-Instruct | ✓ | ✓ | tokenizer / vocab 对照 |

---

## 7. 设备与运行环境

### 7.1 A100 GPU 环境

首轮 GPU 实验使用 A100 机器：

```text
batch = 1
num_gpus = 1
single request execution
same prompt order across runs
greedy decoding as main setting
```

记录环境信息：

```text
GPU model
GPU memory capacity
CUDA version
NVIDIA driver version
PyTorch version
Transformers version
XGrammar version
vLLM / SGLang version
model dtype
lm_head dtype
peak GPU memory
```

推荐运行配置：
模型均已经下载完成，路径为：/hzeng/models
| 模型 | A100 dtype / quantization | 备注 |
|---|---|---|
| Qwen2.5-0.5B | FP16/BF16 | baseline 与 selective 全量运行 |
| Qwen2.5-1.5B | FP16/BF16 | baseline 与 selective 全量运行 |
| Qwen2.5-3B | FP16/BF16 | baseline 与 selective 全量运行 |
| Qwen3-30B-A3B | BF16/FP16 或 4-bit/8-bit | 记录 `lm_head` dtype 与量化方式 |

GPU timing：

```text
torch.cuda.synchronize() before measured region
torch.cuda.Event for transformer / lm_head / grammar-list / sampler
torch.cuda.synchronize() after measured region
warmup before measurement
fixed request order
```

### 7.2 同机 CPU 环境

同机 CPU 用于 CPU 路径测量：

```text
batch = 1
single process
fixed thread count
same request subset as GPU runs
same decoding setting
```

记录 CPU 环境：

```text
CPU model
physical cores
logical cores
RAM
OMP_NUM_THREADS
MKL_NUM_THREADS
PyTorch CPU backend
quantization setting
```

CPU timing 使用：

```text
time.perf_counter_ns()
per-request timing
per-step timing
same JSONL trace schema as GPU
```

---

## 8. Baseline 流程

### 8.1 Baseline variants

| 名称 | 路径 | 目的 |
|---|---|---|
| B1: serving structured baseline | vLLM/SGLang structured-output API | 主流 serving 端表现 |
| B2: controlled full-head baseline | Transformers + XGrammar full head + mask | selective head 的主对比对象 |
| B3: controlled full-head + tracing | 与 B2 相同，增加 step-level timing 与 `K` trace | 获得完整 breakdown |

### 8.2 Full-head constrained decode path

```text
for each request:
    prefill prompt
    initialize grammar matcher
    for t in range(max_new_tokens):
        h_t = transformer_decode_step(...)
        logits_full = lm_head(h_t)
        mask = grammar.next_token_mask(...)
        logits_masked = apply_mask(logits_full, mask)
        token = argmax_or_sample(logits_masked)
        grammar.accept(token)
        append token
        stop when grammar finished or EOS
```

### 8.3 Baseline logging schema

每个 request 记录：

```json
{
  "request_id": "bfcl/simple_python/xxx",
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "framework": "transformers_xgrammar",
  "device": "cuda:0",
  "gpu": "A100",
  "benchmark": "bfcl",
  "category": "simple_python",
  "prompt_tokens": 512,
  "output_tokens": 48,
  "latency_ms_total": 1234.5,
  "latency_ms_prefill": 300.1,
  "latency_ms_decode": 934.4,
  "latency_ms_per_decode_token": 19.46,
  "valid": true,
  "score": 1,
  "steps": [
    {
      "step": 0,
      "K": 1,
      "K_over_V": 0.00000658,
      "transformer_ms": 15.2,
      "lm_head_ms": 2.1,
      "mask_or_allowed_ms": 0.3,
      "sampler_ms": 0.05,
      "token_id": 123
    }
  ]
}
```

---

## 9. Selective LM Head 设计

### 9.1 数学定义

当前 hidden state：

```text
h ∈ R^d
```

LM head 权重：

```text
W ∈ R^{V x d}
```

grammar 当前允许 token 集合：

```text
A ⊂ {0, ..., V-1}, K = |A|
```

Full-head constrained decoding：

```text
logits_full = W h
logits_full[j] = -inf, for j outside A
p(i | h, i in A) = exp(logits_full[i]) / sum_{j in A} exp(logits_full[j])
```

Selective LM Head：

```text
W_A = W[A]
logits_A = W_A h
p_A(i) = exp(logits_A[i]) / sum_j exp(logits_A[j])
```

两种路径在合法集合 `A` 上得到相同 logits 与相同 constrained distribution。

### 9.2 Selective decode path

```text
for each request:
    prefill prompt
    initialize grammar matcher
    for t in range(max_new_tokens):
        h_t = transformer_decode_step(...)
        allowed_ids = grammar.allowed_token_ids(...)
        K = len(allowed_ids)

        if K == 1:
            token = allowed_ids[0]
        elif K <= K_THRESHOLD:
            logits_A = selective_lm_head(h_t, allowed_ids)
            token = sample_or_argmax_over_A(logits_A, allowed_ids)
        else:
            logits_full = lm_head(h_t)
            logits_masked = apply_mask(logits_full, allowed_ids or bitmask)
            token = sample_or_argmax(logits_masked)

        grammar.accept(token)
        append token
        stop when grammar finished or EOS
```

`K == 1` 时，受限分布为单点分布，可直接选择唯一合法 token。该路径仍逐 token 更新 KV 与 grammar state。

### 9.3 Adaptive threshold

Selective head 的启用阈值通过实验 sweep 决定：

```text
K_THRESHOLD ∈ {1, 8, 32, 128, 512, 2048, 8192, 16384}
```

报告内容：

```text
threshold -> fallback rate
threshold -> head-local speedup
threshold -> request-level speedup
threshold -> correctness / validity
```

### 9.4 Allowed set provider

Selective LM Head 需要 `allowed_ids`。本项目实现三条 provider 路径：

| 路径 | 说明 | 适用场景 |
|---|---|---|
| P1: bitmask-to-list | 从 grammar bitmask 得到 allowed_ids | 与现有 grammar backend 对接 |
| P2: cached allowed_ids | 对重复 grammar state / bitmask hash 做 cache | JSON schema / fixed grammar state |
| P3: direct allowed_ids | grammar / trie 原生返回 list | BFCL tool name / parameter name / label trie |

每条路径单独记录：

```text
allowed_provider_ms
bitmask_to_list_ms
cache_hit_rate
allowed_ids_length
```

### 9.5 PyTorch prototype

最小原型：

```python
import torch
import torch.nn.functional as F

@torch.no_grad()
def selective_lm_head(hidden: torch.Tensor,
                      lm_head_weight: torch.Tensor,
                      allowed_ids: torch.Tensor,
                      bias: torch.Tensor | None = None) -> torch.Tensor:
    # hidden: [d] or [1, d]
    # lm_head_weight: [V, d]
    # allowed_ids: [K]
    selected_weight = lm_head_weight.index_select(0, allowed_ids)  # [K, d]
    logits = F.linear(hidden.view(1, -1), selected_weight, None).view(-1)  # [K]
    if bias is not None:
        logits = logits + bias.index_select(0, allowed_ids)
    return logits
```

验证项：

```text
full-head masked argmax == selective argmax
selected logits match full logits on allowed_ids
K bucket latency
index_select time
dot product time
CPU / A100 comparison
```

### 9.6 Quantization 与 MoE 配置

Dense Qwen2.5：

```text
FP16/BF16 full-head baseline
FP16/BF16 selective head
INT8/INT4 runtime with recorded lm_head dtype
```

Qwen3-30B-A3B：

```text
BF16/FP16 configuration when available
4-bit/8-bit quantized configuration
lm_head dtype recorded separately
selected-row dequant timing when quantized head is used
peak memory recorded for each configuration
```

---

## 10. 实验矩阵

### 10.1 主实验矩阵

| Workload | Dense 0.5B | Dense 1.5B | Dense 3B | MoE A3B | A100 GPU | 同机 CPU |
|---|---:|---:|---:|---:|---:|---:|
| BFCL simple_python | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| BFCL simple_java | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| BFCL simple_javascript | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| BFCL multiple | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| BFCL parallel | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| BFCL irrelevance | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| JSONSchemaBench GlaiveAI-2K | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| JSONSchemaBench Kubernetes | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| JSONSchemaBench Github-Easy/Medium/Hard | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### 10.2 每个组合的运行版本

| 版本 | 名称 | 说明 |
|---|---|---|
| V1 | Serving structured baseline | vLLM/SGLang structured-output API |
| V2 | Controlled full-head constrained | Transformers + XGrammar full head + mask |
| V3 | Controlled full-head constrained + tracing | step-level timing 与 `K` trace |
| V4 | Selective head, fixed threshold | 固定阈值，例如 `K<=2048` |
| V5 | Selective head, threshold sweep | 阈值 sweep 后选择配置 |

### 10.3 主采样设置

主实验使用 greedy：

```text
temperature = 0
do_sample = false
```

Greedy 设置用于保证 baseline 与 selective 的输出 token 序列一致，方便 paired request latency 对比。

---

## 11. Measurement protocol

### 11.1 Request subset

BFCL：

```text
Smoke run: 每个 category 100 requests
Main run: selected categories full run
Same request order across all models and devices
```

JSONSchemaBench：

```text
Smoke run: 每个 split 100 schemas
Main run: GlaiveAI-2K full or 500+
Main run: Kubernetes / Github-Easy / Medium / Hard each 200+
Same schema subset across all models and devices
```

MoE CPU 运行采用与 dense 相同的 trace schema，并记录 request subset 大小。

### 11.2 Warmup and repeats

每个 model-device-workload：

```text
warmup requests: 5–10
measured repeats: 3
random seed fixed
same request order
same max_new_tokens
same grammar backend
same allowed provider
```

GPU timing：

```text
torch.cuda.synchronize() before and after measured region
CUDA events for lm_head and selected operations
NVTX ranges for profiler run
```

CPU timing：

```text
time.perf_counter_ns()
fixed thread count
process affinity recorded
CPU model and RAM recorded
```

### 11.3 Metrics

Primary metrics：

| Metric | 说明 |
|---|---|
| request_latency_ms_p50 / p90 / p95 | 用户感知延迟 |
| decode_latency_ms | decode 阶段耗时 |
| ms_per_generated_token | 输出长度归一化 |
| paired_request_speedup | 同一个 request 的 baseline/selective 比值 |
| correctness / validity | BFCL score / JSON validity |

Breakdown metrics：

| Metric | 说明 |
|---|---|
| transformer_step_ms | 中间层 decode 成本 |
| lm_head_ms | full 或 selective head 成本 |
| grammar_mask_ms | grammar mask / allowed set 成本 |
| bitmask_to_list_ms | 从 bitmask 得到 allowed_ids 的成本 |
| sampler_ms | argmax / softmax / sampling 成本 |
| fallback_rate | selective 路径转 full head 的 step 占比 |
| k_eq_1_rate | `K=1` step 占比 |
| k_bucket_distribution | `K` 分布 |

Resource metrics：

| Metric | 说明 |
|---|---|
| peak_gpu_memory_mb | A100 显存峰值 |
| cpu_rss_mb | CPU 内存 |
| lm_head_dtype | FP16/BF16/INT4/other |
| model_quantization | none / GPTQ / AWQ / bitsandbytes / GGUF |
| wall_clock_total | 完整运行耗时 |

---

## 12. Trace-first implementation plan

### Milestone M0：环境和数据准备

交付：

```text
BFCL selected categories loader
JSONSchemaBench selected splits loader
Qwen2.5 0.5B / 1.5B / 3B loader
Qwen3-30B-A3B loader
vLLM/SGLang serving smoke test
Transformers + XGrammar smoke test
A100 batch=1 single request smoke test
同机 CPU batch=1 single request smoke test
统一 config 与 trace output path
```

### Milestone M1：Serving structured baseline

交付：

```text
vLLM structured_outputs request script
SGLang structured outputs request script
BFCL request latency
JSONSchemaBench request latency
BFCL correctness
JSON validity
per-request serving log
```

### Milestone M2：Controlled full-head constrained baseline

交付：

```text
自定义 step-by-step decode loop
full LM head + grammar mask
greedy decoding
BFCL request latency
JSONSchemaBench request latency
BFCL correctness
JSON validity
per-request JSONL log
```

### Milestone M3：`K` 分布与耗时 breakdown

交付：

```text
每 decode step 的 K
每 decode step 的 full LM head latency
每 decode step 的 grammar mask / allowed-provider latency
K histogram
latency breakdown
A100 vs CPU 对比
Dense vs MoE 对比
```

### Milestone M4：Head-only replay

基于 M2/M3 记录的 hidden states 和 allowed_ids 做 replay：

```text
for each recorded step:
    compare full_head_masked_argmax vs selective_head_argmax
    compare full_head_time vs selective_head_time
```

交付：

```text
argmax exactness check
selected logits equality check
selective head local speedup by K bucket
index_select / gather timing
threshold sweep 初步结果
```

### Milestone M5：End-to-end selective decoding

交付：

```text
decode loop 中实际使用 selective head
K==1 direct token path
adaptive fallback
BFCL request-level speedup
JSONSchemaBench request-level speedup
Dense + MoE 主矩阵结果
A100 + CPU 主矩阵结果
```

### Milestone M6：MoE 运行配置固化

交付：

```text
MoE checkpoint name
quantization method
lm_head dtype
peak GPU memory
selected-row dequant time
MoE vs dense 3B head ratio comparison
MoE vs dense 3B request latency comparison
```

### Milestone M7：最终报告

交付：

```text
benchmark setup table
K distribution figures
latency breakdown figures
request-level speedup table
dense vs MoE comparison
A100 vs CPU comparison
implementation notes
reproducibility scripts
```

---

## 13. 预期图表

### Figure 1：系统路径对比

```text
Baseline:
    h -> full LM head over V -> mask -> sample

Selective:
    grammar state -> allowed_ids A
    h -> W[A] dot h -> sample over A
    K threshold controls full-head path
```

### Figure 2：K distribution by workload

每个 workload 一组 stacked bars：

```text
K=1
K<=8
K<=32
K<=128
K<=512
K<=2048
K<=8192
K>8192
```

### Figure 3：Latency breakdown

按模型和设备展示：

```text
transformer
lm_head
grammar / allowed set
sampler
other
```

### Figure 4：Selective head speedup by K bucket

展示 full head vs selective head 的局部耗时比。

### Figure 5：Request-level paired speedup

每个点是一个 request：

```text
x = baseline request latency
y = selective request latency
```

### Figure 6：Dense vs MoE

比较：

```text
Qwen2.5-3B-Instruct
Qwen3-30B-A3B-Instruct-2507 / quantized
```

分析项：

```text
LM head 占比
selective speedup
K distribution
MoE routing / expert latency ratio
```

### Figure 7：A100 vs CPU

比较同一模型、同一 workload、同一 request subset 下的：

```text
request latency
ms/token
LM head ratio
selective head speedup
allowed-provider overhead
```

---

## 14. 成功指标

### 14.1 基础交付指标

项目完成时应具备：

1. BFCL 与 JSONSchemaBench 的 dense + MoE serving baseline；
2. BFCL 与 JSONSchemaBench 的 dense + MoE controlled full-head baseline；
3. request-level latency、decode breakdown 与 `K` 分布；
4. selective head 与 full-head constrained 在 greedy 下输出一致；
5. A100 与同机 CPU 的主结果；
6. dense 3B 与 A3B MoE 的对比结果；
7. 可复现实验脚本与 JSONL trace。

### 14.2 量化目标

| 场景 | 目标 |
|---|---|
| Qwen2.5-0.5B A100 / CPU | request p50 speedup >= 1.10x |
| Qwen2.5-1.5B A100 / CPU | request p50 speedup >= 1.08x |
| Qwen2.5-3B A100 | request p50 speedup >= 1.05x |
| Qwen3-30B-A3B A100 | request p50 speedup >= 1.03x–1.08x |
| K<=512 steps | LM head local speedup > 2x |
| correctness / validity | BFCL score 与 JSON validity 保持一致 |

---

## 15. 实现关注点

### 15.1 Allowed set 生成成本

记录并优化：

```text
bitmask_to_list_ms
allowed_provider_ms
cache_hit_rate
K bucket by provider
provider choice by workload
```

BFCL 的 tool name、parameter name、enum value 可使用 trie 直接产生 allowed_ids。JSONSchemaBench 可使用 bitmask-to-list 与 cache 两种路径进行对比。

### 15.2 A100 上的 selective row access

A100 上重点记录：

```text
index_select time
selected_weight materialization time
dot product time
CUDA kernel count
CUDA event timing
NVTX profiler trace
```

阈值 sweep 用于选择：

```text
K_THRESHOLD for A100
K_THRESHOLD for CPU
K_THRESHOLD by model
K_THRESHOLD by workload
```

### 15.3 MoE 量化与 lm_head dtype

MoE 报告中固定记录：

```text
checkpoint
quantization method
lm_head dtype
transformer dtype
expert dtype
peak GPU memory
selected-row dequant latency
```

MoE selective head 结果与 dense 3B 结果放入同一主表。

### 15.4 Correctness 与 exactness

Greedy 主实验的检查项：

```text
same prompt
same grammar
same max_new_tokens
same decoding params
same output token sequence
same BFCL score / JSON validity
```

Step-level 检查项：

```text
selected_logits == full_logits[allowed_ids] within dtype tolerance
argmax(selected_logits) maps to argmax(masked_full_logits)
```

---

## 16. 推荐代码结构

```text
selective-lm-head/
  README.md
  pyproject.toml
  configs/
    models.yaml
    devices.yaml
    bfcl.yaml
    jsonschemabench.yaml
    thresholds.yaml
  selective_lm_head/
    __init__.py
    decode_loop.py
    model_loader.py
    lm_head.py
    selective_head.py
    full_head.py
    serving_baseline/
      vllm_client.py
      sglang_client.py
      openai_api_runner.py
    grammar/
      __init__.py
      xgrammar_adapter.py
      json_schema_adapter.py
      bfcl_adapter.py
      bitmask.py
      trie_allowed_set.py
    tracing/
      timers.py
      trace_schema.py
      writer.py
    eval/
      bfcl_eval_adapter.py
      json_validity.py
    analysis/
      k_distribution.py
      latency_breakdown.py
      paired_speedup.py
  scripts/
    run_serving_baseline.py
    run_baseline.py
    run_trace.py
    run_selective.py
    replay_head.py
    analyze_results.py
  results/
    traces/
    tables/
    figures/
  notebooks/
    01_k_distribution.ipynb
    02_latency_breakdown.ipynb
    03_speedup.ipynb
```

---

## 17. 示例命令

### 17.1 vLLM serving baseline

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --served-model-name qwen25-1p5b \
  --max-model-len 4096

python scripts/run_serving_baseline.py \
  --server vllm \
  --base-url http://127.0.0.1:8000/v1 \
  --model qwen25-1p5b \
  --benchmark bfcl \
  --categories simple_python,simple_java,simple_javascript \
  --batch-size 1 \
  --output results/traces/vllm_qwen25_1p5b_bfcl_a100.jsonl
```

### 17.2 SGLang serving baseline

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-1.5B-Instruct \
  --host 0.0.0.0 \
  --port 30000 \
  --grammar-backend xgrammar

python scripts/run_serving_baseline.py \
  --server sglang \
  --base-url http://127.0.0.1:30000/v1 \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --benchmark jsonschemabench \
  --splits GlaiveAI-2K,Kubernetes,Github-Easy \
  --batch-size 1 \
  --output results/traces/sglang_qwen25_1p5b_jsonschema_a100.jsonl
```

### 17.3 A100 controlled full-head baseline

```bash
python scripts/run_baseline.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --benchmark bfcl \
  --categories simple_python,simple_java,simple_javascript \
  --device cuda \
  --batch-size 1 \
  --decoding greedy \
  --max-new-tokens 256 \
  --grammar-backend xgrammar \
  --output results/traces/qwen25_1p5b_bfcl_full_a100.jsonl
```

### 17.4 CPU controlled full-head baseline

```bash
OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 \
python scripts/run_baseline.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --benchmark bfcl \
  --categories simple_python,simple_java,simple_javascript \
  --device cpu \
  --batch-size 1 \
  --decoding greedy \
  --max-new-tokens 256 \
  --grammar-backend xgrammar \
  --output results/traces/qwen25_1p5b_bfcl_full_cpu.jsonl
```

### 17.5 Trace `K` distribution

```bash
python scripts/run_trace.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --benchmark jsonschemabench \
  --splits GlaiveAI-2K,Kubernetes,Github-Easy \
  --device cuda \
  --head full \
  --max-requests-per-split 200 \
  --output results/traces/qwen25_1p5b_jsonschema_trace_a100.jsonl
```

### 17.6 Head-only replay

```bash
python scripts/replay_head.py \
  --trace results/traces/qwen25_1p5b_bfcl_full_a100.jsonl \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --device cuda \
  --thresholds 1,8,32,128,512,2048,8192 \
  --output results/tables/head_replay_qwen25_1p5b_a100.csv
```

### 17.7 End-to-end selective head

```bash
python scripts/run_selective.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --benchmark bfcl \
  --categories simple_python,simple_java,simple_javascript \
  --device cuda \
  --batch-size 1 \
  --decoding greedy \
  --k-threshold 2048 \
  --allowed-provider cached_bitmask \
  --output results/traces/qwen25_1p5b_bfcl_selective_a100.jsonl
```

### 17.8 MoE selective head on A100

```bash
python scripts/run_selective.py \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --quantization int4 \
  --lm-head-dtype fp16 \
  --benchmark bfcl \
  --categories simple_python,simple_java,simple_javascript,parallel,multiple \
  --device cuda \
  --batch-size 1 \
  --decoding greedy \
  --k-threshold 2048 \
  --output results/traces/qwen3_30b_a3b_bfcl_selective_a100.jsonl
```

---

## 18. 结果表模板

### 18.1 Main latency table

| Model | Framework | Device | Workload | Head | p50 latency | p95 latency | ms/token | Valid / Acc | Speedup |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | Transformers+XGrammar | CPU | BFCL simple | full |  |  |  |  | 1.00x |
| Qwen2.5-0.5B | Transformers+XGrammar | CPU | BFCL simple | selective |  |  |  |  |  |
| Qwen2.5-1.5B | vLLM/SGLang | A100 | BFCL simple | structured |  |  |  |  |  |
| Qwen2.5-1.5B | Transformers+XGrammar | A100 | BFCL simple | full |  |  |  |  | 1.00x |
| Qwen2.5-1.5B | Transformers+XGrammar | A100 | BFCL simple | selective |  |  |  |  |  |
| Qwen3-30B-A3B | Transformers+XGrammar | A100 | BFCL simple | full |  |  |  |  | 1.00x |
| Qwen3-30B-A3B | Transformers+XGrammar | A100 | BFCL simple | selective |  |  |  |  |  |

### 18.2 K distribution table

| Workload | Model | Device | K=1 | K<=8 | K<=32 | K<=128 | K<=512 | K<=2048 | K>2048 | fallback rate |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BFCL simple | Qwen2.5-1.5B | A100 |  |  |  |  |  |  |  |  |
| JSONSchema GlaiveAI | Qwen2.5-1.5B | A100 |  |  |  |  |  |  |  |  |
| BFCL simple | Qwen3-30B-A3B | A100 |  |  |  |  |  |  |  |  |

### 18.3 Breakdown table

| Model | Device | Workload | Head | Transformer % | LM head % | Grammar/list % | Sampler % | Other % |
|---|---|---|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | CPU | BFCL | full |  |  |  |  |  |
| Qwen2.5-0.5B | CPU | BFCL | selective |  |  |  |  |  |
| Qwen2.5-1.5B | A100 | BFCL | full |  |  |  |  |  |
| Qwen2.5-1.5B | A100 | BFCL | selective |  |  |  |  |  |
| Qwen3-30B-A3B | A100 | BFCL | full |  |  |  |  |  |
| Qwen3-30B-A3B | A100 | BFCL | selective |  |  |  |  |  |

---

## 19. 最终项目叙事

Structured output decoding improves reliability but often computes full-vocabulary logits before masking invalid tokens. This is especially visible in batch-1 constrained generation, where grammar states frequently permit a small subset of tokens. The project first characterizes real workloads from BFCL and JSONSchemaBench, measuring request latency, LM-head contribution, and grammar-induced allowed-token-set sizes. It then implements an exact selective LM head that computes logits only for grammar-allowed tokens and uses a threshold-controlled full-head path for large candidate sets. The evaluation covers dense 0.5B–3B models and A3B-class MoE models on A100 GPU and the same host CPU.

中文版本：

结构化输出提升了格式可靠性，但 constrained decoding 中常见路径仍会先计算完整词表 logits，再由 grammar 屏蔽非法 token。batch=1 结构化生成中，许多 grammar state 只允许很小的 token 子集。本项目先在 BFCL 和 JSONSchemaBench 上刻画真实 request latency、LM head 占比和合法 token 集合分布，再实现 exact selective LM head：只对当前合法 token 计算 logits，并通过阈值控制 large-`K` state 的 full-head 路径。实验覆盖 dense 0.5B–3B 与 A3B-class MoE，并在 A100 GPU 与同机 CPU 上完成测量。

---

## 20. 参考来源

[S1] vLLM Structured Outputs: https://docs.vllm.ai/en/latest/features/structured_outputs/  
[S2] vLLM Logits Processors: https://docs.vllm.ai/en/latest/design/logits_processors/  
[S3] SGLang Structured Outputs: https://docs.sglang.ai/advanced_features/structured_outputs.html  
[S4] XGrammar Workflow: https://xgrammar.mlc.ai/docs/tutorials/workflow_of_xgrammar.html  
[S5] XGrammar Engine Integration: https://xgrammar.mlc.ai/docs/tutorials/engine_integration.html  
[S6] BFCL V4 Leaderboard: https://gorilla.cs.berkeley.edu/leaderboard.html  
[S7] BFCL GitHub README: https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/README.md  
[S8] JSONSchemaBench GitHub repo: https://github.com/guidance-ai/jsonschemabench  
[S9] Qwen2.5-0.5B-Instruct config: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blame/main/config.json  
[S10] Qwen2.5-1.5B-Instruct config: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/config.json  
[S11] Qwen2.5-3B-Instruct config: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/14d7620ba47cf51be0b176e14e27e38a34d4ff88/config.json  
[S12] Qwen3-30B-A3B-Instruct-2507 model card: https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507  
[S13] Qwen3-30B-A3B-Instruct-2507 config: https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507/blob/main/config.json
