from pathlib import Path

from selective_lm_head.analysis.k_distribution import summarize_k_distribution
from selective_lm_head.tracing.trace_schema import RequestTrace, StepTrace
from selective_lm_head.tracing.writer import JsonlTraceWriter


def test_trace_writer_and_k_distribution(tmp_path: Path):
    trace_path = tmp_path / "trace.jsonl"
    trace = RequestTrace(
        request_id="r1",
        model="m",
        framework="f",
        device="cpu",
        benchmark="jsonschemabench",
        category="smoke",
        head="full",
        steps=[
            StepTrace(step=0, K=1, K_over_V=0.1),
            StepTrace(step=1, K=9, K_over_V=0.9),
        ],
    )
    with JsonlTraceWriter(trace_path) as writer:
        writer.write(trace)

    rows = summarize_k_distribution(trace_path)
    assert rows[0]["total_steps"] == 2
    assert rows[0]["K=1"] == 1
    assert rows[0]["K<=32"] == 1

