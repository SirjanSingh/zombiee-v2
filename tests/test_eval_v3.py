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


def test_dagger_labels_model_states(tmp_path):
    out = tmp_path / "dagger.jsonl"
    eval_v3.main(["--mock-model", "--n-episodes", "2", "--baselines", "--no-log",
                  "--dagger-out", str(out)])
    import json
    from training.inference import parse_actions
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert rows, "no DAgger rows written"
    assert all(len(parse_actions(r["completion"], agent_id=0, max_actions=5)) == 5 for r in rows)
    assert all(r["prompt"].rstrip().endswith("commentary.") for r in rows)


def test_replay_recording(tmp_path):
    import json
    eval_v3.main(["--no-model", "--baselines", "camp", "--n-episodes", "1", "--no-log",
                  "--record-replays", "1", "--replay-dir", str(tmp_path), "--tag", "t"])
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    rep = json.loads(files[0].read_text())
    assert rep["frames"][0]["actor"] is None            # reset frame first
    assert rep["frames"][-1]["t"] == rep["meta"]["result"]["final_t"]
    assert {"walls", "food", "water", "safehouse"} <= set(rep["layout"])
    assert len(rep["frames"][1]["agents"]) == 5
    types = {e["type"] for f in rep["frames"] for e in f["events"]}
    assert "drink" in types                              # camp drinks within an episode
