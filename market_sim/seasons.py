"""Calendar and seasonal system.

Pure constants and two tiny helper functions — no side effects, safe to import anywhere.
"""

TICKS_PER_SEASON = 10
TICKS_PER_YEAR   = TICKS_PER_SEASON * 4

SEASON_SPRING = 0
SEASON_SUMMER = 1
SEASON_AUTUMN = 2
SEASON_WINTER = 3
SEASON_NAMES  = {0: "Spring", 1: "Summer", 2: "Autumn", 3: "Winter"}

# Food consumption multiplier per season
SEASON_FOOD_CONSUMPTION_MULT: dict = {
    SEASON_SPRING: 1.0,
    SEASON_SUMMER: 0.9,
    SEASON_AUTUMN: 0.9,
    SEASON_WINTER: 1.5,
}

# Farm production progress multiplier per season
FARM_SEASON_MULT: dict = {
    SEASON_SPRING: 0.4,
    SEASON_SUMMER: 1.0,
    SEASON_AUTUMN: 1.8,
    SEASON_WINTER: 0.0,
}

# Non-farm output reduction in winter
BUILDING_WINTER_MULT = 0.8


def getCurrentSeason(tick: int) -> int:
    """Return season (0–3) for the given global tick."""
    return (tick % TICKS_PER_YEAR) // TICKS_PER_SEASON


def getEffectiveSeason(ctrl) -> int:
    """Return effective season, suppressing spring/summer during a long winter."""
    base = getCurrentSeason(ctrl._current_tick)
    if getattr(ctrl, 'long_winter_remaining', 0) > 0:
        if base in (SEASON_SPRING, SEASON_SUMMER):
            return SEASON_WINTER
    return base
