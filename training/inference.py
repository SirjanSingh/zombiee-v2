"""Inference helpers — JSON action parsing + LLM action functions.

Reused by:
    - training.train (reward function rolls out random actions after the
      model's first action; needs robust JSON parsing)
    - training.eval (drives the trained policy across full episodes)
    - training.simulator (single-episode pretty playthrough)
"""

from __future__ import annotations

import json
import logging
import random
import re
from typing import Any, Callable, Optional

logger = logging.getLogger("survivecity_v2.inference")


VALID_ACTION_TYPES = frozenset({
    "move_up", "move_down", "move_left", "move_right",
    "eat", "wait", "vote_lockout", "broadcast",
    "drink", "scan", "pickup", "drop", "give", "inject",
})

# Used for random rollouts in the GRPO reward function and as the
# random-baseline policy in eval / simulator.
RANDOM_NON_VOTE_ACTIONS = [
    "move_up", "move_down", "move_left", "move_right",
    "eat", "drink", "wait", "pickup",
]

# Regex fallbacks for lenient parsing (used when strict JSON fails).
# Pattern matches: action_type: "move_up" / action_type="move_up" /
# "action_type": move_up / action_type = "move_up" / etc.
_ACTION_TYPE_RE = re.compile(
    r"""action_type[\s"']*[:=]+[\s"']*(\w+)""", re.IGNORECASE
)
_VOTE_TARGET_RE   = re.compile(r"""vote_target[\s"']*[:=]+[\s"']*(\d+)""", re.IGNORECASE)
_SCAN_TARGET_RE   = re.compile(r"""scan_target[\s"']*[:=]+[\s"']*(\d+)""", re.IGNORECASE)
_INJECT_TARGET_RE = re.compile(r"""inject_target[\s"']*[:=]+[\s"']*(\d+)""", re.IGNORECASE)
_GIFT_TARGET_RE   = re.compile(r"""gift_target[\s"']*[:=]+[\s"']*(\d+)""", re.IGNORECASE)
_ITEM_SLOT_RE     = re.compile(r"""item_slot[\s"']*[:=]+[\s"']*(\d+)""", re.IGNORECASE)
_MESSAGE_RE       = re.compile(r"""message[\s"']*[:=]+[\s"']*["']([^"']{1,40})["']""", re.IGNORECASE)


def parse_action(text: str, agent_id: int) -> Optional[dict]:
    """Parse an action from model output.

    Tries three strategies in order:
      1. Strict JSON object extraction (any {...} substring that loads).
      2. Regex extraction of `action_type: <word>` and optional fields
         from prose like "I'll move_up" or `action_type=eat`.
      3. Last-resort: scan for any literal valid action_type word
         anywhere in the text (catches "I will move_up to the food cell").

    Returns None only if NONE of the 14 valid action_types appears
    anywhere in the completion. This is critical for GRPO — if every
    completion in a group returns None, all rewards floor at 0.01,
    `reward_std=0`, and the gradient signal is dead.
    """
    if not text:
        return None
    text = text.strip()

    # Strip markdown code fences if present
    fenced = text
    if fenced.startswith("```"):
        parts = fenced.split("```")
        if len(parts) >= 2:
            inner = parts[1]
            if inner.startswith("json"):
                inner = inner[4:]
            fenced = inner.strip()

    # --- Strategy 1: strict JSON extraction ---
    for start in range(len(fenced)):
        if fenced[start] == "{":
            for end in range(len(fenced), start, -1):
                if fenced[end - 1] == "}":
                    try:
                        d = json.loads(fenced[start:end])
                        if not isinstance(d, dict):
                            continue
                    except (json.JSONDecodeError, TypeError):
                        continue
                    d.setdefault("agent_id", agent_id)
                    if d.get("action_type") in VALID_ACTION_TYPES:
                        return d

    # --- Strategy 2: regex-extract action_type + optional fields ---
    m = _ACTION_TYPE_RE.search(text)
    if m:
        atype = m.group(1).lower()
        if atype in VALID_ACTION_TYPES:
            result: dict[str, Any] = {"agent_id": agent_id, "action_type": atype}
            for field, regex in (
                ("vote_target",   _VOTE_TARGET_RE),
                ("scan_target",   _SCAN_TARGET_RE),
                ("inject_target", _INJECT_TARGET_RE),
                ("gift_target",   _GIFT_TARGET_RE),
                ("item_slot",     _ITEM_SLOT_RE),
            ):
                fm = regex.search(text)
                if fm:
                    try:
                        result[field] = int(fm.group(1))
                    except ValueError:
                        pass
            mm = _MESSAGE_RE.search(text)
            if mm:
                result["message"] = mm.group(1)[:40]
            return result

    # --- Strategy 3: last-resort word scan ---
    # Lowercase scan so "Move_up" / "MOVE_UP" still match. Sort by length
    # so "move_up" matches before "move" (no false-positive on "move").
    # Word-boundary required: bare "scan" must NOT match inside "scan_target"
    # (which would return a no-op scan with no target — the model exploited
    # this to dodge the anti_camp rubric for ~50% of training actions).
    text_lower = text.lower()
    for atype in sorted(VALID_ACTION_TYPES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(atype)}\b", text_lower):
            return {"agent_id": agent_id, "action_type": atype}

    return None


def parse_actions(
    text: str,
    agent_id: int,
    max_actions: int = 8,
) -> list[dict]:
    """Parse one or more actions from model output.

    Two modes:
      1. JSON array `[{...}, {...}, ...]` — each element treated as an action.
         Each element runs through the same validation as `parse_action`.
      2. Single object — falls back to `parse_action`. Returns a 1-element
         list (or [] if nothing parseable).

    Used by multi-action GRPO rollouts where the model emits K decisions
    per completion. With max_actions=1 (or completion containing a single
    object), behaves identically to wrapping `parse_action` in a list.

    Caps output at `max_actions`. Returns [] when no valid actions found.
    """
    if not text:
        return []
    text = text.strip()

    # Strip markdown code fences (same handling as parse_action).
    fenced = text
    if fenced.startswith("```"):
        parts = fenced.split("```")
        if len(parts) >= 2:
            inner = parts[1]
            if inner.startswith("json"):
                inner = inner[4:]
            fenced = inner.strip()

    # Try to find a top-level JSON array first.
    for start in range(len(fenced)):
        if fenced[start] == "[":
            for end in range(len(fenced), start, -1):
                if fenced[end - 1] == "]":
                    try:
                        arr = json.loads(fenced[start:end])
                    except (json.JSONDecodeError, TypeError):
                        continue
                    if not isinstance(arr, list):
                        continue
                    out: list[dict] = []
                    for elem in arr[:max_actions]:
                        if not isinstance(elem, dict):
                            continue
                        elem.setdefault("agent_id", agent_id)
                        if elem.get("action_type") in VALID_ACTION_TYPES:
                            out.append(elem)
                    if out:
                        return out
                    # Empty array or array with no valid actions — keep searching
                    # (could be a stray "[" earlier in the text).

    # No valid array — fall back to single-action parse.
    single = parse_action(text, agent_id=agent_id)
    return [single] if single is not None else []


def random_action(agent_id: int, obs: dict, rng: Optional[random.Random] = None) -> dict:
    """Random baseline action. Casts a random vote at vote-phase steps."""
    rng = rng or random
    s = obs.get("step_count", 0)
    if s in (30, 50, 70, 90):
        return {
            "agent_id": agent_id,
            "action_type": "vote_lockout",
            "vote_target": rng.choice([0, 1, 2, 3, 4]),
        }
    return {"agent_id": agent_id, "action_type": rng.choice(RANDOM_NON_VOTE_ACTIONS)}


# Cell coordinates copied from layout.py to avoid circular imports during
# training (training/ is a leaf package; survivecity_v2_env layout is fine
# to import here, but inlining keeps this helper self-contained for tests).
# Mirror of survivecity_v2_env.layout.FOOD_CELLS / WATER_CELLS. Inlined to
# keep the heuristic self-contained for tests (training/ is a leaf package).
# v2.2: added inner-ring food/water (rows 4-5 and 9-10, cols 4-5 and 9-10)
# so safehouse-bound agents have nearby resupply within hunger budget.
_FOOD_CELLS_TUPLE: tuple[tuple[int, int], ...] = (
    (1, 1), (1, 13), (13, 1), (13, 13),
    (1, 7), (13, 7), (7, 1), (7, 13),
    (4, 5), (4, 9), (10, 5), (10, 9),
)
_WATER_CELLS_TUPLE: tuple[tuple[int, int], ...] = (
    (3, 3), (3, 11), (11, 3), (11, 11),
    (5, 4), (5, 10), (9, 4), (9, 10),
)
_SAFEHOUSE_CELLS_TUPLE: tuple[tuple[int, int], ...] = tuple(
    (r, c) for r in range(6, 9) for c in range(6, 9)
)


def _nearest_cell(my_r: int, my_c: int, cells) -> Optional[tuple[int, int]]:
    best = None
    best_d = 999
    for (r, c) in cells:
        d = abs(my_r - r) + abs(my_c - c)
        if d < best_d:
            best_d = d
            best = (r, c)
    return best


def _step_toward(my_r: int, my_c: int, target: tuple[int, int]) -> Optional[str]:
    tr, tc = target
    dr, dc = tr - my_r, tc - my_c
    # Prefer the larger axis so we close distance fastest
    if abs(dr) > abs(dc):
        return "move_down" if dr > 0 else "move_up"
    if dc != 0:
        return "move_right" if dc > 0 else "move_left"
    if dr != 0:
        return "move_down" if dr > 0 else "move_up"
    return None


def forage_heuristic_action(
    agent_id: int,
    obs: dict,
    rng: Optional[random.Random] = None,
) -> dict:
    """A scripted forage policy used as the rollout opponent during GRPO.

    v2.2 rewrite (2026-05-10) — fixes the early-game zombie deaths that
    pinned eval episodes at ~17 steps regardless of the trained model.

    Why the rewrite: under the v2.1 heuristic, agents at episode start had
    hunger=0 / thirst=0 / hp=3, which fell through every threshold and hit
    the random-action fallback. Random walking exposed agents to zombie
    attacks for ~5 steps until hp dropped enough to trigger the safehouse
    rule — by which time many were already dead. Eval baselines ran
    ~16-18 steps regardless. The v2.2 priorities below default agents to
    the safehouse from step 0 and only forage when resources are actually
    needed.

    Decision rules (highest priority first):
      1. On water cell                            → drink (free, +1 thirst clear)
      2. On food cell + hungry                    → eat (clears hunger)
      3. On food cell + inv space + not hungry    → pickup food (carry to safehouse)
      4. Vote phase                               → vote_lockout (random non-self)
      5. In safehouse + has food + hunger>=4      → eat from inventory
      6. In safehouse + has water + thirst>=4     → drink from inventory
      7. HP <= 1                                  → step toward safehouse (emergency)
      8. Thirst >= 4                              → step toward nearest water
      9. Hunger >= 4                              → step toward nearest food
     10. Default                                  → wait if in safehouse, else move to it

    Default-to-safehouse is the key v2.2 change — random fallback was a
    silent killer of v2.1. Forage threshold 4 matters because latent
    infected agents tick hunger at 1.5x rate (game.py line 261); waiting
    until 10 inside the safehouse left infected teammates dead by
    env-step 10. The inner-ring food/water (layout.py v2.2) makes the
    threshold-4 forage trip survivable: ~3-4 cells from any safehouse
    cell, ~6 round-trip rounds — within the hunger=4-to-15 budget.
    """
    rng = rng or random
    s = obs.get("step_count", 0)

    # Find self in agents list
    me = None
    for a in obs.get("agents", []):
        if a.get("agent_id") == agent_id and a.get("is_alive", True):
            me = a
            break
    if me is None:
        return {"agent_id": agent_id, "action_type": "wait"}

    my_r, my_c = me.get("row", 0), me.get("col", 0)
    hunger = me.get("hunger", 0)
    thirst = me.get("thirst", 0)
    hp = me.get("hp", 3)
    inv = me.get("inventory", []) or []
    on_water = (my_r, my_c) in _WATER_CELLS_TUPLE
    on_food = (my_r, my_c) in _FOOD_CELLS_TUPLE
    in_safehouse = (my_r, my_c) in _SAFEHOUSE_CELLS_TUPLE
    has_food = "food" in inv
    has_water = "water" in inv
    inv_full = len(inv) >= 3

    def _move_to(cells):
        target = _nearest_cell(my_r, my_c, cells)
        if target is None:
            return None
        return _step_toward(my_r, my_c, target)

    # 1-2. Always drink/eat on the cell — free and immediate.
    if on_water:
        return {"agent_id": agent_id, "action_type": "drink"}
    if on_food and hunger > 0:
        return {"agent_id": agent_id, "action_type": "eat"}
    # 3. On food cell but not hungry: stash for the trip back to safehouse.
    if on_food and not has_food and not inv_full:
        return {"agent_id": agent_id, "action_type": "pickup", "item_type": "food"}

    # 4. Vote phase always takes precedence over forage routing.
    if s in (30, 50, 70, 90):
        choices = [i for i in range(5) if i != agent_id]
        return {
            "agent_id": agent_id,
            "action_type": "vote_lockout",
            "vote_target": rng.choice(choices),
        }

    # 5-6. In safehouse: drain inventory before stepping out.
    if in_safehouse:
        if hunger >= 4 and has_food:
            return {"agent_id": agent_id, "action_type": "eat"}
        if thirst >= 4 and has_water:
            return {"agent_id": agent_id, "action_type": "drink"}

    # 7. Emergency HP — get inside safehouse before anything else.
    if hp <= 1:
        if in_safehouse:
            return {"agent_id": agent_id, "action_type": "wait"}
        mv = _move_to(_SAFEHOUSE_CELLS_TUPLE)
        if mv is not None:
            return {"agent_id": agent_id, "action_type": mv}

    # 8-9. Forage at threshold 4. CRITICAL — latent infected agents tick
    # hunger at 1.5x rate (game.py line 261), so they hit hunger=15 starvation
    # by env_step=10 if the heuristic waits. The earlier v2.2 attempt waited
    # until hunger=10 inside the safehouse and watched infected teammates
    # die. Threshold 4 (matches v2.1) gives agents 11 game-rounds to walk
    # to a food cell — the inner ring is 3-4 steps from any safehouse cell,
    # so a forage trip is comfortably inside the survival budget.
    if thirst >= 4 and thirst >= hunger:
        mv = _move_to(_WATER_CELLS_TUPLE)
        if mv is not None:
            return {"agent_id": agent_id, "action_type": mv}
    if hunger >= 4:
        mv = _move_to(_FOOD_CELLS_TUPLE)
        if mv is not None:
            return {"agent_id": agent_id, "action_type": mv}

    # 10. Default: wait in safehouse, otherwise step toward it. NEVER random-
    # walk — random motion is what got agents killed by zombies in v2.1.
    if in_safehouse:
        return {"agent_id": agent_id, "action_type": "wait"}
    mv = _move_to(_SAFEHOUSE_CELLS_TUPLE)
    if mv is not None:
        return {"agent_id": agent_id, "action_type": mv}
    return {"agent_id": agent_id, "action_type": "wait"}


def make_llm_action_fn(
    model: Any,
    tokenizer: Any,
    max_new_tokens: int = 96,
    prefix_actions: int = 1,
    trained_agent_id: Optional[int] = None,
) -> Callable:
    """Build an action_fn that calls the LLM and parses output as JSON.

    Modes:
      - prefix_actions=1, trained_agent_id=None (default): legacy behaviour.
        Every agent gets a single-action LLM call per turn. Used when the
        model was trained with K=1 GRPO (run 1, 2, 3).
      - prefix_actions=K (>1): model is queried with the multi-action prompt
        and emits a JSON array of K actions. The first action is returned;
        the remaining K-1 are queued and emitted on the agent's next K-1
        turns. The cache resets at episode boundaries (detected by step==0).
        Used when the model was trained with K>1 GRPO (run 4+) so the eval
        distribution matches training.
      - trained_agent_id=A (int): only agent A uses the LLM. The other
        agents use forage_heuristic_action. Mirrors the GRPO training setup
        where only A0 is model-controlled. Without this flag, all 5 agents
        would be model-controlled — an OOD setup the model never trained on.

    On parse failure, falls back to forage_heuristic_action (random_action
    used to be the fallback but produced parse-failure-dominated baselines).
    """
    import torch
    from collections import deque
    from survivecity_v2_env.prompts import build_system_prompt

    # Per-agent queue of remaining actions from a multi-action plan.
    # Empty/missing → query the model fresh.
    plans: dict[int, deque] = {}
    # Track last observed step per agent so we can clear stale plans when a
    # new episode starts (step_count drops to 0).
    last_step_seen = {"value": -1}

    def _generate(agent_id: int, description: str) -> str:
        prompt = build_system_prompt(
            agent_id, description, prefix_actions=prefix_actions,
        )
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user",
             "content": (
                 "What is your next action? Respond with JSON only."
                 if prefix_actions == 1
                 else f"What is your {prefix_actions}-action plan? "
                      "Respond with a JSON array only."
             )},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
        ).strip()

    def _sanitize(action: dict) -> dict:
        if action.get("action_type") == "broadcast":
            msg = action.get("message")
            if isinstance(msg, str):
                action["message"] = msg[:40]
            else:
                action["message"] = "alert"
        return action

    def llm_action(agent_id: int, obs: dict) -> dict:
        # Reset the per-episode plan cache when step rolls back to 0
        # (env.reset() called between episodes).
        step = obs.get("step_count", 0)
        if step < last_step_seen["value"]:
            plans.clear()
        last_step_seen["value"] = step

        # If only a specific agent is trained, others go straight to heuristic.
        if trained_agent_id is not None and agent_id != trained_agent_id:
            return forage_heuristic_action(agent_id, obs)

        # Use a queued planned action if available.
        if agent_id in plans and plans[agent_id]:
            return _sanitize(plans[agent_id].popleft())

        description = obs.get("description", "")
        try:
            response = _generate(agent_id, description)
        except Exception as e:
            logger.debug(f"llm generation error: {e}")
            return forage_heuristic_action(agent_id, obs)

        if prefix_actions > 1:
            actions = parse_actions(
                response, agent_id=agent_id, max_actions=prefix_actions,
            )
            if not actions:
                return forage_heuristic_action(agent_id, obs)
            # Queue remaining actions for the agent's next prefix_actions-1 turns.
            plans[agent_id] = deque(actions[1:])
            return _sanitize(actions[0])

        parsed = parse_action(response, agent_id=agent_id)
        if parsed is None:
            return forage_heuristic_action(agent_id, obs)
        return _sanitize(parsed)

    return llm_action
