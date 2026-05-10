"""System prompts for SurviveCity v2 agents.

Phase-aware: the prompt template includes the current phase, day/night, and
vote-round information. Past failures (post-mortems) are prepended.
"""

from __future__ import annotations


SYSTEM_PROMPT_TEMPLATE_SINGLE = """You are agent A{agent_id} in a 5-agent zombie-apocalypse simulation called SurviveCity v2.

== WORLD RULES ==
- 15x15 grid. 5 agents (A0..A4) share the map with zombies.
- Safehouse (S) at the centre (rows 6-8, cols 6-8): heals 1 HP/step, blocks zombies.
- Food (F): 12 cells (8 outer + 4 inner ring near safehouse). Use "eat" on cell to clear hunger, or "pickup" to carry.
- Water (W): 8 cells, persistent (4 outer + 4 inner ring). Use "drink" on cell to clear thirst, or "pickup" to carry.
- Medicine (M): 2 cells, scarce. Use "pickup" to carry. Use "inject" to cure latent infection.
- Walls (#): block movement.
- Hunger and thirst tick +1 each step. >=15 deals 1 HP/step.
- Inventory: 3 slots per agent. Items: food | water | medicine.

== HIDDEN INFECTION ==
- 2 of 5 agents start INFECTED: one BITER, one SABOTEUR. Roles assigned at episode start.
- BITER: revealed at step 25. Post-reveal, attacks adjacent agents. Each attack has a 35% chance to BITE-INFECT (latent for 15 steps, then revealed). Bitten agents become biters too.
- SABOTEUR: revealed at step 60. Post-reveal, food they eat depletes 2x slower to respawn (silently hurts the team).
- LATENT infection: hidden 15-step countdown. If you are bitten you receive a private "You were bitten" message — go grab medicine and inject yourself.
- You only see your OWN infection_state. Other agents' status is inferred via noisy behavioural cues (30% false-positive on healthy, 30% miss on infected).

== DAY / NIGHT + WAVES ==
- Day: steps 0-24 and 50-74. Visibility full.
- Night: steps 25-49 and 75-99. Visibility reduced (filtered description).
- Zombie waves spawn at steps 25, 50, 75 (+2, +3, +3, capped at 12 total).

== VOTING (3 ROUNDS) ==
- vote_lockout is valid at exactly step 30, 50, 70, 90. Plurality with at least 2 votes locks one agent out of the safehouse for the rest of the episode.
- Voting wrong (locking out a healthy agent) hurts the team and ends them faster.

== BROADCAST ECONOMY ==
- "broadcast" sends a 40-char message to all agents.
- Each broadcast adds +1 to the noise meter. Above threshold (3), zombies get a free extra step toward agents next turn. Decays every 10 steps.

== YOUR ACTIONS ==
Output EXACTLY ONE action as JSON:
{{"action_type": "<type>", "vote_target": <int>, "message": "<str>", "scan_target": <int>, "inject_target": <int>, "gift_target": <int>, "item_slot": <int>, "item_type": "<type>"}}

Valid action_types:
  v1: "move_up", "move_down", "move_left", "move_right", "eat", "wait", "vote_lockout", "broadcast"
  v2 NEW: "drink", "scan", "pickup", "drop", "give", "inject"

Field rules:
  - vote_lockout: requires vote_target in {{0,1,2,3,4}}. Only valid at t=30,50,70,90.
  - broadcast: requires message (max 40 chars).
  - scan: requires scan_target. Costs 1 thirst. Returns a noisy infection-status hint.
  - pickup: optional item_type ("food" | "water" | "medicine"). Picks up from current cell.
  - drop / give / inject: requires item_slot in {{0,1,2}}. give also requires gift_target (adjacent agent).
  - inject: target is inject_target (None or self_id == self-inject). Cures latent infection.

Unused fields can be null/omitted.

== EXAMPLES (copy this exact format) ==
{{"action_type": "move_up"}}
{{"action_type": "eat"}}
{{"action_type": "drink"}}
{{"action_type": "vote_lockout", "vote_target": 2}}
{{"action_type": "broadcast", "message": "I think A2 is infected"}}
{{"action_type": "pickup", "item_type": "medicine"}}
{{"action_type": "inject", "inject_target": 0, "item_slot": 0}}

{past_failures}
== CURRENT SITUATION ==
{situation}

Respond with ONLY the JSON action on a single line, like the examples above. No explanation, no markdown, no commentary.
"""


# Multi-action variant. The model commits to a SHORT plan of K consecutive
# actions; the env applies them in order at agent A0's turns. Used by
# multi-action GRPO (--prefix-actions K) so the gradient sees the consequence
# of a multi-step strategy, not just a single first move.
SYSTEM_PROMPT_TEMPLATE_MULTI = """You are agent A{agent_id} in a 5-agent zombie-apocalypse simulation called SurviveCity v2.

== WORLD RULES ==
- 15x15 grid. 5 agents (A0..A4) share the map with zombies.
- Safehouse (S) at the centre (rows 6-8, cols 6-8): heals 1 HP/step, blocks zombies.
- Food (F): 12 cells (8 outer + 4 inner ring near safehouse). Use "eat" on cell to clear hunger, or "pickup" to carry.
- Water (W): 8 cells, persistent (4 outer + 4 inner ring). Use "drink" on cell to clear thirst, or "pickup" to carry.
- Medicine (M): 2 cells, scarce. Use "pickup" to carry. Use "inject" to cure latent infection.
- Walls (#): block movement.
- Hunger and thirst tick +1 each step. >=15 deals 1 HP/step.
- Inventory: 3 slots per agent. Items: food | water | medicine.

== HIDDEN INFECTION ==
- 2 of 5 agents start INFECTED: one BITER, one SABOTEUR.
- BITER: revealed at step 25. Post-reveal, attacks adjacent agents with 35% bite chance.
- SABOTEUR: revealed at step 60. Food they eat depletes 2x slower to respawn.
- LATENT infection: hidden 15-step countdown. If bitten you receive a private "You were bitten" message.
- You only see your OWN infection_state. Other agents inferred via noisy behavioural cues.

== DAY / NIGHT + WAVES ==
- Day: steps 0-24 and 50-74. Visibility full.
- Night: steps 25-49 and 75-99. Visibility reduced (filtered description).
- Zombie waves spawn at steps 25, 50, 75 (+2, +3, +3, capped at 12 total).

== VOTING (3 ROUNDS) ==
- vote_lockout is valid at exactly step 30, 50, 70, 90.
- Voting wrong (locking out a healthy agent) hurts the team.

== BROADCAST ECONOMY ==
- "broadcast" sends a 40-char message to all agents.
- Each broadcast adds +1 to the noise meter. Above threshold (3), zombies get a free extra step.

== YOUR PLAN ==
Output a JSON ARRAY of EXACTLY {prefix_k} actions you commit to take in order on your next {prefix_k} turns. The env applies them in sequence; you cannot revise mid-plan.

Each element has the shape:
{{"action_type": "<type>", "vote_target": <int>, "message": "<str>", "scan_target": <int>, "inject_target": <int>, "gift_target": <int>, "item_slot": <int>, "item_type": "<type>"}}

Valid action_types:
  v1: "move_up", "move_down", "move_left", "move_right", "eat", "wait", "vote_lockout", "broadcast"
  v2 NEW: "drink", "scan", "pickup", "drop", "give", "inject"

Field rules: same as single-action mode. Unused fields can be null/omitted.

== EXAMPLES (copy this exact format) ==
[{{"action_type": "move_up"}}, {{"action_type": "eat"}}, {{"action_type": "drink"}}]
[{{"action_type": "move_left"}}, {{"action_type": "move_left"}}, {{"action_type": "pickup", "item_type": "medicine"}}]
[{{"action_type": "vote_lockout", "vote_target": 2}}, {{"action_type": "wait"}}, {{"action_type": "wait"}}]

{past_failures}
== CURRENT SITUATION ==
{situation}

Respond with ONLY the JSON array on a single line, like the examples above. No explanation, no markdown, no commentary.
"""


def build_system_prompt(
    agent_id: int,
    situation: str,
    postmortem_buffer: dict[int, list[str]] | None = None,
    prefix_actions: int = 1,
) -> str:
    """Build the system prompt for an agent.

    Args:
        agent_id: 0..4
        situation: NL description of current observation
        postmortem_buffer: optional {agent_id -> [past_postmortem_strings]}

    Returns:
        Full prompt string.
    """
    if postmortem_buffer and agent_id in postmortem_buffer:
        past = postmortem_buffer[agent_id][-3:]
        if past:
            past_block = "== PAST FAILURES (learn from these!) ==\n" + "\n".join(past) + "\n"
        else:
            past_block = ""
    else:
        past_block = ""
    if prefix_actions > 1:
        return SYSTEM_PROMPT_TEMPLATE_MULTI.format(
            agent_id=agent_id,
            past_failures=past_block,
            situation=situation,
            prefix_k=prefix_actions,
        )
    return SYSTEM_PROMPT_TEMPLATE_SINGLE.format(
        agent_id=agent_id,
        past_failures=past_block,
        situation=situation,
    )


# Backward-compat alias for code that imports SYSTEM_PROMPT_TEMPLATE directly.
SYSTEM_PROMPT_TEMPLATE = SYSTEM_PROMPT_TEMPLATE_SINGLE


def format_observation_description(
    agent_id: int,
    state_dict: dict,
    phase: str,
    day_phase: str,
    step: int,
    broadcasts: list[str],
    behavioral_cues: list[str],
    last_scan: dict | None,
    own_inventory: list[str],
    own_infection_state: str,
    own_bite_at_step: int | None,
    noise_meter: int,
    noise_threshold: int,
) -> str:
    """Format the observation into an LLM-readable description.

    At night, far-away agents/zombies are filtered out (Manhattan > 5).
    """
    agents = state_dict.get("agents", [])
    zombies = state_dict.get("zombies", [])

    self_a = next((a for a in agents if a["agent_id"] == agent_id), None)
    if self_a is None:
        return "You are dead."

    is_night = day_phase == "night"
    visibility_radius = 5 if is_night else 99

    def _dist(other_r, other_c):
        return abs(other_r - self_a["row"]) + abs(other_c - self_a["col"])

    lines = []
    lines.append(f"Step {step}/100 | Phase: {phase} | {day_phase.upper()}")
    lines.append(
        f"You are A{agent_id} at ({self_a['row']},{self_a['col']}) "
        f"HP={self_a['hp']}/3 hunger={self_a['hunger']} thirst={self_a['thirst']}"
    )
    lines.append(f"Inventory: {own_inventory if own_inventory else 'empty'} (cap=3)")
    lines.append(f"Noise meter: {noise_meter}/{noise_threshold} (decays every 10 steps)")

    # Self infection state
    if own_infection_state == "latent":
        if own_bite_at_step is not None:
            lines.append(
                f"⚠ YOU ARE LATENT-INFECTED (bitten at step {own_bite_at_step}, "
                f"reveals at step {own_bite_at_step + 15}). Get medicine + inject self."
            )
        else:
            lines.append("⚠ YOU ARE LATENT-INFECTED. Get medicine + inject self.")
    elif own_infection_state == "revealed":
        lines.append(
            "⚠⚠ YOU ARE REVEALED-INFECTED. Strategy: avoid suspicion, frame healthy "
            "agents in votes, survive to step 100."
        )

    if self_a.get("locked_out"):
        lines.append("⚠ You are LOCKED OUT — cannot enter safehouse.")

    # Other agents
    visible_others = []
    for a in agents:
        if a["agent_id"] == agent_id:
            continue
        d = _dist(a["row"], a["col"])
        if not a["is_alive"]:
            visible_others.append(f"  A{a['agent_id']}: DEAD")
            continue
        if d > visibility_radius:
            visible_others.append(f"  A{a['agent_id']}: out_of_sight (night)")
            continue
        locked = " [LOCKED OUT]" if a.get("locked_out") else ""
        visible_others.append(
            f"  A{a['agent_id']}: ({a['row']},{a['col']}) "
            f"HP={a['hp']} hunger={a['hunger']} thirst={a['thirst']}{locked}"
        )
    if visible_others:
        lines.append("Other agents:")
        lines.extend(visible_others)

    # Zombies
    visible_zombies = []
    for z in zombies:
        d = _dist(z["row"], z["col"])
        if d > visibility_radius:
            continue
        danger = " ⚠CLOSE!" if d <= 2 else ""
        visible_zombies.append(f"  Z{z['zombie_id']}: ({z['row']},{z['col']}) dist={d}{danger}")
    if visible_zombies:
        lines.append(f"Zombies (within {visibility_radius}):")
        lines.extend(visible_zombies)
    elif zombies and is_night:
        lines.append(f"Zombies: none within {visibility_radius} cells (night)")

    # Behavioural cues (already noise-filtered)
    if behavioral_cues:
        lines.append("Cues:")
        for c in behavioral_cues[:5]:
            lines.append(f"  • {c}")

    # Last scan result
    if last_scan is not None:
        target = last_scan.get("target_id")
        rep = last_scan.get("reported_infected")
        lines.append(
            f"Last scan: A{target} reported as "
            f"{'INFECTED' if rep else 'healthy'} (noisy, ~70% accurate)"
        )

    # Broadcasts
    if broadcasts:
        lines.append("Recent broadcasts:")
        for b in broadcasts[-5:]:
            lines.append(f"  📢 {b}")

    # Phase-specific reminders
    if step in (30, 50, 70, 90):
        lines.append(f"⚠ VOTE PHASE (round at step {step}) — use vote_lockout NOW.")
    elif step in (25, 50, 75):
        lines.append(f"⚠ ZOMBIE WAVE INCOMING (this step). Cluster near safehouse.")
    elif step == 24:
        lines.append("⚠ Sun setting. Wave at step 25.")
    elif step == 49:
        lines.append("⚠ Day breaking. Vote phase 2 + wave at step 50.")
    elif step == 89:
        lines.append("⚠ Final vote at step 90.")

    return "\n".join(lines)
