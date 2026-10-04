"""forage_heuristic_v3: the v2 bugs are fixed and it avoids zombies."""

from survivecity_v2_env import layout
from training import inference as I
from training.inference import forage_heuristic_v3 as H


def _obs(row, col, hunger=0, thirst=0, hp=3, inv=None, zombies=(), step=5, meta=None):
    return {
        "step_count": step,
        "agents": [{"agent_id": 0, "row": row, "col": col, "hunger": hunger, "thirst": thirst,
                    "hp": hp, "inventory": inv or [], "is_alive": True}],
        "zombies": [{"zombie_id": i, "row": r, "col": c} for i, (r, c) in enumerate(zombies)],
        "metadata": meta or {},
    }


def test_layout_mirrors_match():
    assert set(I._FOOD_CELLS_TUPLE) == layout.FOOD_CELLS
    assert set(I._WATER_CELLS_TUPLE) == layout.WATER_CELLS
    assert set(I._WALL_CELLS_TUPLE) == layout.WALL_CELLS
    assert set(I._SAFEHOUSE_CELLS_TUPLE) == layout.SAFEHOUSE_CELLS


def test_rollout_policy_is_v3():
    assert I.forage_heuristic_action is I.forage_heuristic_v3


def test_does_not_camp_on_water_when_not_thirsty():
    a = H(0, _obs(5, 4, thirst=0, hunger=0))
    assert a["action_type"] != "drink"
    assert H(0, _obs(5, 4, thirst=3))["action_type"] == "drink"


def test_ignores_depleted_food_cell():
    a = H(0, _obs(4, 5, hunger=6, meta={"depleted_food": [[4, 5]]}))
    assert a["action_type"] != "eat"
    assert H(0, _obs(4, 5, hunger=6))["action_type"] == "eat"


def test_steps_around_wall():
    # From (6,5) toward water (5,4): v2 picked move_left into the wall at (6,4).
    a = H(0, _obs(6, 5, thirst=6, hunger=0))
    assert a["action_type"] == "move_up"
    assert I.forage_heuristic_v2(0, _obs(6, 5, thirst=6, hunger=0))["action_type"] == "move_left"


def test_waits_instead_of_stepping_next_to_zombie():
    # Wants water at (5,4) via (5,5); a zombie at (4,5) makes (5,5) dangerous.
    a = H(0, _obs(5, 6, thirst=6, zombies=[(4, 5)]))
    assert a["action_type"] == "wait"


def test_heads_to_extraction_zone_after_radio():
    zone = [[r, c] for r in range(3) for c in range(12, 15)]
    a = H(0, _obs(7, 7, step=61, meta={"extraction_zone": zone}))
    assert a["action_type"] in ("move_up", "move_right")
    assert H(0, _obs(0, 13, step=71, meta={"extraction_zone": zone}))["action_type"] == "wait"


def test_extraction_departure_is_timed():
    zone = [[r, c] for r in range(3) for c in range(12, 15)]
    meta = {"extraction_zone": zone, "extraction": {"extraction_step": 90}}
    # 29 turns left, zone 9 away: too early, stays in the safehouse
    assert H(0, _obs(7, 7, step=61, meta=meta))["action_type"] == "wait"
    # 15 turns left: leaves
    assert H(0, _obs(7, 7, step=75, meta=meta))["action_type"] in ("move_up", "move_right")
