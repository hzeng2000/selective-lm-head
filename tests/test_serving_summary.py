from selective_lm_head.analysis.serving_summary import summarize_serving_trace
from selective_lm_head.tracing.writer import write_jsonl


def test_serving_summary_groups_trace_rows(tmp_path):
    trace_path = tmp_path / "serving.jsonl"
    write_jsonl(
        trace_path,
        [
            {
                "request_id": "a",
                "benchmark": "bfcl",
                "category": "simple_python",
                "model": "qwen",
                "framework": "vllm_structured",
                "device": "server",
                "head": "serving_structured",
                "valid": True,
                "latency_ms_total": 100.0,
                "prompt_tokens": 10,
                "output_tokens": 20,
                "latency_ms_per_decode_token": 5.0,
            },
            {
                "request_id": "b",
                "benchmark": "bfcl",
                "category": "simple_python",
                "model": "qwen",
                "framework": "vllm_structured",
                "device": "server",
                "head": "serving_structured",
                "valid": False,
                "latency_ms_total": 80.0,
                "prompt_tokens": 12,
                "output_tokens": 10,
                "latency_ms_per_decode_token": 8.0,
            },
        ],
    )

    rows = summarize_serving_trace(trace_path)

    assert rows == [
        {
            "benchmark": "bfcl",
            "category": "simple_python",
            "model": "qwen",
            "framework": "vllm_structured",
            "device": "server",
            "head": "serving_structured",
            "requests": 2,
            "valid": 1,
            "median_latency_ms": 90.0,
            "min_latency_ms": 80.0,
            "max_latency_ms": 100.0,
            "median_prompt_tokens": 11.0,
            "median_output_tokens": 15.0,
            "total_output_tokens": 30,
            "median_ms_per_output_token": 6.5,
        }
    ]
