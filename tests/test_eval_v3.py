"""eval_v3: closed-loop runner works end to end on CPU with the mock model."""

from training import eval_v3


def test_mock_model_closed_loop_runs_and_parses():
    res = eval_v3.main(["--mock-model", "--n-episodes", "2", "--baselines", "wait", "--no-log"])
    mock = res["mock(heuristic_v3)"]
    assert mock["metrics"]["parse_rate"] == 1.0
    assert mock["metrics"]["model_calls"] > 0
    assert set(res) == {"mock(heuristic_v3)", "wait"}
    # same seeds for every row
    assert [e["seed"] for e in mock["episodes"]] == [e["seed"] for e in res["wait"]["episodes"]]


def test_baseline_needs_no_generator():
    res = eval_v3.main(["--no-model", "--n-episodes", "2", "--baselines", "heuristic_v3", "--no-log"])
    assert res["heuristic_v3"]["metrics"]["parse_rate"] is None
