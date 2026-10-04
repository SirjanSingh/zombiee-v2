"""tools/calibrate.py runs end to end and summarises correctly (no research_log writes)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import calibrate as C  # noqa: E402
from survivecity_v2_env.balance import V2_2  # noqa: E402


def test_calibrate_small_run_all_policies():
    res = C.calibrate(V2_2, list(C.POLICIES), n=3, seed=1, progress=False)
    assert set(res) == set(C.POLICIES)
    for r in res.values():
        m = r["metrics"]
        assert 0.0 <= m["survival"] <= 1.0
        assert m["ep_len"] > 0
        assert len(r["episodes"]) == 3
    assert "| oracle |" in C.format_table(res)


def test_short_episode_counts_as_survival():
    # With max_steps=10 nobody starves: every policy should reach the end alive.
    res = C.calibrate(V2_2.with_(max_steps=10), ["heuristic_v3"], n=3, seed=1, progress=False)
    assert res["heuristic_v3"]["metrics"]["reached_max"] == 1.0


def test_parse_overrides():
    ov = C.parse_overrides(["starve_threshold=20", 'wave_schedule={"50": 2}', "zombie_chase_radius=null"])
    assert ov == {"starve_threshold": 20, "wave_schedule": {50: 2}, "zombie_chase_radius": None}
    cfg = V2_2.with_(**ov)
    assert cfg.waves == {50: 2}


def test_extraction_metrics_reported_for_rc2():
    from survivecity_v2_env.balance import V3_RC2
    res = C.calibrate(V3_RC2, ["heuristic_v3"], n=2, seed=1, progress=False)
    assert {"extraction", "failed_flight", "n_extracted"} <= set(res["heuristic_v3"]["metrics"])
    assert {"extracted", "failed_flight", "n_extracted"} <= set(res["heuristic_v3"]["episodes"][0])
    assert "| extract |" in C.format_table(res)
    off = C.calibrate(V2_2, ["random"], n=1, seed=1, progress=False)
    assert "extraction" not in off["random"]["metrics"]
