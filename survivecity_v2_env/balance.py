"""Balance knobs for SurviveCity v2 in one place.

Every number that decides how hard the game is (survival clocks, zombie
pressure, infection timing, resource respawns) lives on `BalanceConfig`.
`game.py` reads them from `EpisodeState.balance`, so an episode's difficulty
is fixed at `create_episode()` time and calibration scripts can sweep
configs without monkeypatching module constants.

Layout (cell positions) stays in layout.py; vote steps stay in rubric/game
because they are part of the social-deduction rules, not the balance.

This module imports nothing from the package so any module can import it.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Optional


@dataclass(frozen=True)
class BalanceConfig:
    # Episode length
    max_steps: int = 100

    # Survival clocks. Rates are per turn and may be fractional: an agent's
    # meter rises by ceil((t+1)*rate) - ceil(t*rate) on step t, so a rate of
    # 1.5 alternates +2/+1 and 0.5 alternates +1/+0.
    hp_max: int = 3
    hunger_rate: float = 1.0
    thirst_rate: float = 1.0
    infected_hunger_mult: float = 1.5   # latent or revealed agents get hungry faster
    starve_threshold: int = 15          # hunger >= this costs 1 HP per turn
    dehydrate_threshold: int = 15       # thirst >= this costs 1 HP per turn
    safehouse_heal: int = 1             # HP per turn inside (not if locked out)
    scan_thirst_cost: int = 1

    # Zombies
    zombie_chase_radius: Optional[int] = None   # None = chase any agent outside the safehouse
    zombie_contact_damage: int = 1
    wave_schedule: tuple[tuple[int, int], ...] = ((25, 2), (50, 3), (75, 3))
    max_zombies: int = 12
    noise_threshold: int = 3            # broadcasts above this give zombies an extra step
    noise_decay_period: int = 10

    # Resources
    food_respawn_delay: int = 10
    saboteur_food_respawn_delay: int = 20
    medicine_respawn_delay: int = 25

    # Infection timing (role mechanics themselves are not balance knobs)
    p_bite: float = 0.35
    bite_damage: int = 1
    latent_duration: int = 15
    biter_reveal_step: int = 25
    saboteur_reveal_step: int = 60
    infection_death_after: int = 30     # steps after a bite with no medicine -> death

    @property
    def waves(self) -> dict[int, int]:
        return dict(self.wave_schedule)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["wave_schedule"] = {str(k): v for k, v in self.wave_schedule}
        return d

    def with_(self, **changes) -> "BalanceConfig":
        if "wave_schedule" in changes and isinstance(changes["wave_schedule"], dict):
            changes["wave_schedule"] = tuple(sorted(changes["wave_schedule"].items()))
        return replace(self, **changes)


def meter_tick(step: int, rate: float) -> int:
    """How much a meter rising at `rate`/turn grows on `step` (integer, deterministic)."""
    eps = 1e-9
    return math.ceil((step + 1) * rate - eps) - math.ceil(step * rate - eps)


# The tuning every run before v3 used. Kept explicit so it never drifts
# when the dataclass defaults change.
V2_2 = BalanceConfig(
    max_steps=100,
    hp_max=3,
    hunger_rate=1.0,
    thirst_rate=1.0,
    infected_hunger_mult=1.5,
    starve_threshold=15,
    dehydrate_threshold=15,
    safehouse_heal=1,
    scan_thirst_cost=1,
    zombie_chase_radius=None,
    zombie_contact_damage=1,
    wave_schedule=((25, 2), (50, 3), (75, 3)),
    max_zombies=12,
    noise_threshold=3,
    noise_decay_period=10,
    food_respawn_delay=10,
    saboteur_food_respawn_delay=20,
    medicine_respawn_delay=25,
    p_bite=0.35,
    bite_damage=1,
    latent_duration=15,
    biter_reveal_step=25,
    saboteur_reveal_step=60,
    infection_death_after=30,
)

PRESETS: dict[str, BalanceConfig] = {
    "v2.2": V2_2,
}

DEFAULT_PRESET = "v2.2"
DEFAULT_BALANCE: BalanceConfig = PRESETS[DEFAULT_PRESET]


def get_balance(name_or_cfg: "str | BalanceConfig | None" = None) -> BalanceConfig:
    """Resolve a preset name (or pass a config through). None -> default."""
    if name_or_cfg is None:
        return DEFAULT_BALANCE
    if isinstance(name_or_cfg, BalanceConfig):
        return name_or_cfg
    try:
        return PRESETS[name_or_cfg]
    except KeyError:
        raise ValueError(f"unknown balance preset {name_or_cfg!r}; have {sorted(PRESETS)}")
