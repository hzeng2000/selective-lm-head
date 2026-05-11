from pathlib import Path

from selective_lm_head.benchmarks import load_requests


def test_real_bfcl_loader_if_downloaded():
    data_dir = Path("data/repos/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data")
    if not data_dir.exists():
        return
    requests = load_requests(
        "bfcl",
        data_path=data_dir,
        categories=["simple_python"],
        max_requests=2,
    )
    assert len(requests) == 2
    assert requests[0].schema is not None
    assert "JSON" in requests[0].prompt or "Function definitions" in requests[0].prompt


def test_real_jsonschemabench_loader_if_downloaded():
    data_dir = Path("data/repos/jsonschemabench/data")
    if not data_dir.exists():
        return
    requests = load_requests(
        "jsonschemabench",
        data_path=data_dir,
        splits=["GlaiveAI-2K"],
        max_requests=2,
    )
    assert len(requests) == 2
    assert requests[0].schema is not None
    assert requests[0].prompt.startswith("Generate one JSON object")

