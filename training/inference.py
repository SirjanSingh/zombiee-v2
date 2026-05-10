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
_FOOD_CELLS_TUPLE: tuple[tuple[int, int], ...] = (
    (1, 1), (1, 13), (13, 1), (13, 13),
    (1, 7), (13, 7), (7, 1), (7, 13),
)
_WATER_CELLS_TUPLE: tuple[tuple[int, int], ...] = (
    (3, 3), (3, 11), (11, 3), (11, 11),
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

    Why this exists: the v2.0 reward_fn rolled out 30 random actions after
    the model's single action. Random policy on a 15×15 grid with 8 food
    cells and 100-step horizon almost never reaches food, so rollouts
    universally end in starvation around step 18-25 — and every group's
    cumulative reward floors at the same negative value, killing GRPO
    variance. A simple forage heuristic raises the baseline survival from
    ~step 18 to ~step 50, so the model's first action's downstream
    consequences become visible in the cumulative reward.

    Decision rules (highest priority first):
      1. On a water cell                         → drink
      2. On a food cell (and hunger > 0)         → eat
      3. Vote phase                              → vote_lockout (random non-self)
      4. HP < 3 and resources OK                 → step toward safehouse (heal +1/turn)
      5. Thirst >= 4 (and >= hunger)             → step toward nearest water
      6. Hunger >= 4                             → step toward nearest food
      7. Else                                    → random non-vote action

    The safehouse-return rule is what keeps the rollouts from dying at
    step 11-13. Without it, agents leave the safehouse to forage and never
    come back; their HP slowly attrits from zombie hits and they die
    before any voting phase even starts. With it, agents top up at the
    safehouse between forage trips, episodes routinely reach step 50+,
    and the GRPO reward signal sees vote phases / late-game waves /
    terminal rubrics — which is what we actually want to train on.

    Thresholds default to 4 (out of 15-to-death) so agents start foraging
    well before they're in danger. With 5 agents acting once per round and
    hunger ticking +1/action, at hunger=4 the agent has ~10 actions before
    starvation HP loss starts — enough time to walk 5-7 cells to food
    even if they need to detour around walls.
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
    on_water = (my_r, my_c) in _WATER_CELLS_TUPLE
    on_food = (my_r, my_c) in _FOOD_CELLS_TUPLE
    in_safehouse = (my_r, my_c) in _SAFEHOUSE_CELLS_TUPLE

    # Always drink/eat on the cell — there's no cost to doing so even if
    # we're not yet hungry/thirsty (eat resets hunger to 0; the depot
    # respawn is a future-step concern, not this rollout's problem).
    if on_water:
        return {"agent_id": agent_id, "action_type": "drink"}
    if on_food and hunger > 0:
        return {"agent_id": agent_id, "action_type": "eat"}

    # Vote phase always takes precedence over forage routing — the rollout
    # needs to actually exercise the vote rubric for the gradient to see it.
    if s in (30, 50, 70, 90):
        choices = [i for i in range(5) if i != agent_id]
        return {
            "agent_id": agent_id,
            "action_type": "vote_lockout",
            "vote_target": rng.choice(choices),
        }

    # Damaged but not in immediate need? Head back to safehouse to heal.
    # Inside the safehouse the env adds +1 HP per turn (capped at 3) and
    # zombies can't path-find inside, so a few turns of waiting here is
    # cheap insurance. Without this, agents drained by zombies in transit
    # never recover and the rollout dies around step 13.
    if hp < 3 and hunger < 6 and thirst < 6:
        if in_safehouse:
            return {"agent_id": agent_id, "action_type": "wait"}
        target = _nearest_cell(my_r, my_c, _SAFEHOUSE_CELLS_TUPLE)
        if target is not None:
            mv = _step_toward(my_r, my_c, target)
            if mv is not None:
                return {"agent_id": agent_id, "action_type": mv}

    # Whichever resource is more urgent gets routed first. Tie-break:
    # whichever is higher absolute value. Threshold lowered to 4 so we
    # start foraging early enough to actually arrive in time.
    if thirst >= 4 and thirst >= hunger:
        target = _nearest_cell(my_r, my_c, _WATER_CELLS_TUPLE)
        if target is not None:
            mv = _step_toward(my_r, my_c, target)
            if mv is not None:
                return {"agent_id": agent_id, "action_type": mv}

    if hunger >= 4:
        target = _nearest_cell(my_r, my_c, _FOOD_CELLS_TUPLE)
        if target is not None:
            mv = _step_toward(my_r, my_c, target)
            if mv is not None:
                return {"agent_id": agent_id, "action_type": mv}

    return {"agent_id": agent_id, "action_type": rng.choice(RANDOM_NON_VOTE_ACTIONS)}


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
