"""Economy metrics and the composite reward signal.

All functions are pure reads — no side effects on the simulation state.

Depends on: types, structures, _globals, tick (for computeCoinValue)
"""
from __future__ import annotations
import math

from .types      import itemTag, itemFlag
from .structures import ALL_BUILDING_TYPES, MAX_TECH_LEVEL
from ._globals   import globalControllers
from .tick       import (
    computeCoinValue, _INFRA_DRAIN_PER_POP, _INFRA_DRAIN_PER_BUILDING,
    _ownedBuildings,
)

_POP_CAP = 100_000.0

# GDP smoothing — exponential moving average applied to the growth signal so
# single-tick spikes (rare-item crafts, market floods) don't masquerade as
# sustained growth.  Alpha ≈ 0.15 ≈ 6-tick half-life.
_GDP_EMA_ALPHA = 0.15

# Treasury target — the agent should keep treasury near 20 % of the rolling
# 30-tick average GDP.  Hoarding beyond 2× that target incurs a per-tick
# penalty that scales with the excess.
_TREASURY_TARGET_FRACTION = 0.20
_TREASURY_HISTORY_LEN     = 30
_TREASURY_GRACE_MULTIPLE  = 2.0    # allow up to 2× target before penalising
_TREASURY_HOARD_RATE      = 0.003  # penalty per unit of excess ratio
_TREASURY_HOARD_CAP       = 0.05   # max penalty per tick

# Strategic resource tags that matter most for ruler-level health
_STRATEGIC_TAGS = (
    itemTag.food, itemTag.fuel, itemTag.tools,
    itemTag.metal, itemTag.lumber,
)


# =============================================================================
# METRIC FUNCTIONS
# =============================================================================
def computeGDP(ctrl) -> float:
    """Sum of (price × supply) across all tracked items."""
    return sum(u.itemPrice * u.supply for u in ctrl.market.itemTracker.values())


def computeDiversityScore(ctrl) -> float:
    """Shannon entropy of active production output tags, normalised to [0, 1]."""
    counts: dict = {}
    for building in ctrl.market.buildingList.values():
        chain = building.activeChain()
        if chain:
            for tag in chain.outputDict:
                counts[tag] = counts.get(tag, 0) + 1
    if not counts:
        return 0.0
    total      = sum(counts.values())
    entropy    = -sum((c / total) * math.log2(c / total) for c in counts.values())
    max_entropy = math.log2(len(counts)) if len(counts) > 1 else 1.0
    return entropy / max_entropy if max_entropy > 0 else 1.0


def computeUnemploymentRate(ctrl) -> float:
    """Fraction of the labour force without employment."""
    total_employed = sum(
        sum(b.workers.values()) for b in ctrl.market.buildingList.values()
    )
    participation = {1: 0.95, 2: 0.90, 3: 0.70, 4: 0.20}
    labour_force  = sum(pc.count * participation.get(pc.tier, 0.9)
                        for pc in ctrl.pop_classes)
    return max(0.0, 1.0 - total_employed / max(labour_force, 1.0))


def computeAvgHappiness(ctrl) -> float:
    """Population-weighted average happiness (0–1)."""
    if ctrl.population <= 0:
        return 0.0
    return sum(pc.happiness * pc.count for pc in ctrl.pop_classes) / ctrl.population


def computeAvgLiteracy(ctrl) -> float:
    """Population-weighted average literacy (0–1)."""
    if ctrl.population <= 0:
        return 0.0
    return sum(pc.literacy * pc.count for pc in ctrl.pop_classes) / ctrl.population


def computeTierRatio(ctrl) -> float:
    """Fraction of population in tier 3+ (merchants and nobles)."""
    upper = sum(pc.count for pc in ctrl.pop_classes if pc.tier >= 3)
    return upper / max(ctrl.population, 1.0)


def computeProductionEfficiency(ctrl) -> float:
    """Average efficiency_rating across owned active chains."""
    ratings = [b.activeChain().efficiency_rating
               for b in _ownedBuildings(ctrl)
               if b.activeChain() is not None]
    return sum(ratings) / len(ratings) if ratings else 0.0


def computeEventPenalty(ctrl) -> float:
    """Penalty score [0, 1] from active disasters and economic distress."""
    penalty = 0.0
    if ctrl.crop_failure_remaining > 0:
        penalty += 0.30 * min(ctrl.crop_failure_remaining / 30.0, 1.0)
    if ctrl.long_winter_remaining > 0:
        penalty += 0.20 * min(ctrl.long_winter_remaining / 25.0, 1.0)
    penalty += 0.20 * ctrl.market.infrastructure.wear
    starving = sum(1 for pc in ctrl.pop_classes
                   if getattr(pc, '_food_satisfaction', 0.5) < 0.35)
    penalty += 0.20 * (starving / max(len(ctrl.pop_classes), 1))
    penalty += 0.10 * min(computeUnemploymentRate(ctrl), 1.0)
    return min(penalty, 1.0)


def computePrivateProfit(ctrl) -> float:
    """Local-AI metric: net profit from owned building operations this tick.

    Positive = buildings are earning more than they cost to run.
    Useful as a per-tick signal; normalised by treasury to stay bounded.
    """
    revenue = sum(b.last_revenue for b in _ownedBuildings(ctrl))
    costs   = sum(b.last_cost    for b in _ownedBuildings(ctrl))
    return revenue - costs


def computeSupplyChainBalance(ctrl) -> float:
    """Score [0, 1] measuring how evenly supply meets demand across all tracked items.

    1.0 = all items perfectly balanced.  Low values indicate bottlenecks.
    """
    items = list(ctrl.market.itemTracker.values())
    if not items:
        return 0.0
    scores = []
    for itu in items:
        if itu.demand <= 0:
            continue
        ratio = min(itu.supply / itu.demand, 2.0)   # cap at 2× over-supply
        scores.append(1.0 - abs(ratio - 1.0) / 2.0)
    return sum(scores) / len(scores) if scores else 0.0


def computeEconomicStrength(ctrl) -> float:
    """Ruler-AI composite metric [0, 1]: GDP, tech, diversity, infra, strategic supply.

    Weights:
      GDP growth   30%
      Tech level   20%
      Diversity    20%
      Infra        15%
      Strategic    15%
    """
    gdp_score   = math.log1p(computeGDP(ctrl)) / math.log1p(1_000_000.0)
    gdp_score   = min(gdp_score, 1.0)
    tech_score  = ctrl.techLevel / MAX_TECH_LEVEL
    div_score   = computeDiversityScore(ctrl)
    infra       = ctrl.market.infrastructure
    infra_score = min(infra.level / 5.0, 1.0)

    # Strategic resource coverage: fraction of strategic tags with supply > demand
    strat_scores = []
    for tag in _STRATEGIC_TAGS:
        units = ctrl.market.getItemsByTag(tag)
        if not units:
            strat_scores.append(0.0)
            continue
        total_supply = sum(u.supply for u in units)
        total_demand = sum(u.demand for u in units)
        strat_scores.append(min(total_supply / max(total_demand, 1.0), 1.0))
    strat_score = sum(strat_scores) / len(strat_scores)

    return (0.30 * gdp_score
            + 0.20 * tech_score
            + 0.20 * div_score
            + 0.15 * infra_score
            + 0.15 * strat_score)


def computeTaxBurden(ctrl) -> float:
    """Local-AI penalty: fraction of last tick's treasury that was taxed away.

    0.0 = no taxes paid.  1.0 = full treasury seized (catastrophic).
    """
    history = getattr(ctrl, 'tax_history', [])   # tax_history lives on ruler
    # For locals, we compute implicitly from treasury delta — approximate via
    # the ruler's record for this controller
    ruler_id = getattr(ctrl, 'ruler_id', None)
    if ruler_id is None:
        return 0.0
    from ._globals import globalControllers
    ruler = globalControllers.get(ruler_id)
    if ruler is None:
        return 0.0
    recent = [r for r in ruler.tax_history if r.local_ctrl_id == ctrl.id]
    if not recent:
        return 0.0
    last = recent[-1]
    if ctrl.treasury <= 0:
        return 0.0
    return min(last.total / max(ctrl.treasury, 0.01), 1.0)


def computeEventMitigationScore(ctrl) -> float:
    """Preparedness score [0, 1] — food buffer, infra headroom, diversity, trade."""
    food_units   = ctrl.market.getItemsByTag(itemTag.food)
    food_supply  = sum(u.supply for u in food_units) if food_units else 0.0
    food_per_tick = ctrl.population * 0.005
    food_score   = min(food_supply / max(food_per_tick, 0.001) / 20.0, 1.0)

    infra         = ctrl.market.infrastructure
    passive_drain = (ctrl.population * _INFRA_DRAIN_PER_POP
                     + len(ctrl.market.buildingList) * _INFRA_DRAIN_PER_BUILDING)
    headroom      = max(0.0, infra.effective_capacity - passive_drain)
    infra_score   = min(headroom / max(infra.max_capacity, 1.0), 1.0)

    active_btypes = {b.building_type.name for b in ctrl.market.buildingList.values()}
    btype_score   = len(active_btypes) / max(len(ALL_BUILDING_TYPES), 1)

    trade_score   = min(len(ctrl.activeTradeDeals) / 3.0, 1.0)

    return (0.35 * food_score + 0.25 * infra_score
            + 0.25 * btype_score + 0.15 * trade_score)


# =============================================================================
# DELTA / STAGNATION / DECLINE PENALTY
# =============================================================================

# Grace periods before penalties kick in
_STAGNATION_GRACE = 8    # ticks of no change allowed before stagnation cost
_DECLINE_GRACE    = 4    # consecutive declining ticks before escalation penalty

# Max expected delta per tick — used to normalise penalties to a consistent scale
_METRIC_MAX_DELTA = {
    'gdp':       2000.0,
    'happiness': 0.20,
    'literacy':  0.10,
    'diversity': 0.30,
    'population':200.0,
    'tech':      1.0,
    'infra':     1.0,
    'emp_rate':  0.20,    # employment rate = 1 − unemployment; higher is better
}

# Per-tick stagnation cost applied after the grace period (capped per metric)
_STAGNATION_COST = {
    'gdp':       0.004,
    'happiness': 0.003,
    'literacy':  0.003,
    'diversity': 0.003,
    'population':0.002,
    'tech':      0.003,
    'infra':     0.003,
    'emp_rate':  0.007,   # employment stagnation hurts more — 100 % unemp must be costly
}

_STAGNATION_CAP = 0.04   # max stagnation penalty per metric per tick
_DECLINE_CAP    = 0.04   # max escalation penalty per metric per tick


def _metric_snapshot(ctrl) -> dict:
    return {
        'gdp':        computeGDP(ctrl),
        'happiness':  computeAvgHappiness(ctrl),
        'literacy':   computeAvgLiteracy(ctrl),
        'diversity':  computeDiversityScore(ctrl),
        'population': float(ctrl.population),
        'tech':       float(ctrl.techLevel),
        'infra':      float(ctrl.market.infrastructure.level),
        'emp_rate':   1.0 - computeUnemploymentRate(ctrl),
    }


def _delta_stagnation_penalty(ctrl) -> float:
    """Return a <=0 penalty that penalises:
      • any metric moving in the wrong direction (proportional delta penalty),
      • any metric stagnating beyond the grace period,
      • any metric in sustained decline (escalating per extra tick).
    State is stored on the controller object and resets naturally with each
    episode because reset() creates fresh controllers.
    """
    current = _metric_snapshot(ctrl)

    # First call this episode — initialise tracking state, no penalty yet.
    if not hasattr(ctrl, '_metric_prev'):
        ctrl._metric_prev    = current.copy()
        ctrl._metric_stag    = {k: 0 for k in current}
        ctrl._metric_decline = {k: 0 for k in current}
        return 0.0

    prev    = ctrl._metric_prev
    stag    = ctrl._metric_stag
    decline = ctrl._metric_decline
    penalty = 0.0

    for k, cur in current.items():
        p     = prev[k]
        delta = cur - p
        norm  = delta / _METRIC_MAX_DELTA[k]   # normalised change

        if delta < -1e-6:                                    # metric declined
            penalty += min(abs(norm), 1.0) * 0.05            # delta penalty
            stag[k]    = 0
            decline[k] = decline[k] + 1
            excess = decline[k] - _DECLINE_GRACE
            if excess > 0:
                penalty += min(0.003 * excess, _DECLINE_CAP)  # escalating decline

        elif delta > 1e-6:                                   # metric improved
            stag[k]    = 0
            decline[k] = max(0, decline[k] - 1)

        else:                                                # stagnant tick
            stag[k]    = stag[k] + 1
            decline[k] = max(0, decline[k] - 1)
            excess = stag[k] - _STAGNATION_GRACE
            if excess > 0:
                penalty += min(_STAGNATION_COST[k] * excess, _STAGNATION_CAP)

    ctrl._metric_prev    = current
    ctrl._metric_stag    = stag
    ctrl._metric_decline = decline
    return -penalty


# =============================================================================
# GDP EMA + TREASURY TARGET HELPERS
# =============================================================================

def _update_gdp_tracking(ctrl, gdp: float) -> tuple:
    """Update EMA and 30-tick history; return (prev_ema, curr_ema).

    Call once per tick inside each reward function before computing
    growth_score so the growth signal is smoothed.
    """
    if not hasattr(ctrl, '_gdp_ema'):
        ctrl._gdp_ema     = gdp
        ctrl._gdp_history = []
    prev_ema      = ctrl._gdp_ema
    curr_ema      = _GDP_EMA_ALPHA * gdp + (1.0 - _GDP_EMA_ALPHA) * prev_ema
    ctrl._gdp_ema = curr_ema
    ctrl._gdp_history.append(gdp)
    if len(ctrl._gdp_history) > _TREASURY_HISTORY_LEN:
        ctrl._gdp_history.pop(0)
    return prev_ema, curr_ema


def _treasury_target_penalty(ctrl) -> float:
    """Per-tick penalty when treasury is hoarded well above the target level.

    Target = TREASURY_TARGET_FRACTION × mean(GDP over last 30 ticks).
    No penalty until we have 10 ticks of history (early-game grace).
    Penalty scales linearly once treasury exceeds TREASURY_GRACE_MULTIPLE × target,
    capped at TREASURY_HOARD_CAP per tick.
    """
    history = getattr(ctrl, '_gdp_history', [])
    if len(history) < 10:
        return 0.0
    avg_gdp = sum(history) / len(history)
    target  = _TREASURY_TARGET_FRACTION * avg_gdp
    if target < 100.0:          # too small to be meaningful (very early game)
        return 0.0
    grace = target * _TREASURY_GRACE_MULTIPLE
    if ctrl.treasury <= grace:
        return 0.0
    excess_ratio = (ctrl.treasury - grace) / max(grace, 1.0)
    return min(excess_ratio * _TREASURY_HOARD_RATE, _TREASURY_HOARD_CAP)


# =============================================================================
# REWARD  (role-aware)
# =============================================================================

# Local AI: prioritise private profit and supply-chain balance.
_LOCAL_REWARD_WEIGHTS = {
    "profit":    0.30,   # private building profit (main driver)
    "balance":   0.15,   # supply-chain balance (avoid bottlenecks)
    "growth":    0.15,   # GDP growth (rising tide lifts all boats)
    "penalty":   0.15,   # disaster penalties
    "population":0.15,   # population size (workforce = earning power)
    "happiness": 0.10,   # worker happiness (affects productivity)
}

# Ruler AI: prioritise overall economic strength and strategic resources.
_RULER_REWARD_WEIGHTS = {
    "strength":  0.35,   # overall economic strength
    "tax_rev":   0.20,   # tax revenue (incentive to grow the local economy)
    "growth":    0.15,   # GDP growth
    "tech":      0.15,   # technology advancement
    "penalty":   0.10,   # event penalties
    "mitigation":0.05,   # disaster preparedness
}


def _computeLocalReward(ctrl, prev_gdp: float) -> float:
    """Local-AI reward: profit + supply balance + welfare penalties."""
    dsp = _delta_stagnation_penalty(ctrl)

    # No buildings → no productive economy.  Passive market signals (population,
    # happiness, GDP from spawned items) must not make it profitable to ignore
    # BUILD.  Return a flat negative so any no-building policy is strictly losing.
    gdp = computeGDP(ctrl)
    prev_ema, curr_ema = _update_gdp_tracking(ctrl, gdp)
    if not _ownedBuildings(ctrl):
        return -0.10 + dsp

    # Use EMA-smoothed GDP for the growth signal so single-tick rare-item
    # spikes don't look like sustained growth to the policy.
    gdp_growth   = math.log1p(curr_ema) - math.log1p(max(prev_ema, 1.0))
    growth_score = max(-1.0, min(1.0, gdp_growth * 5.0))

    # Profit signal: normalised log-scale so high profits still reward
    profit = computePrivateProfit(ctrl)
    profit_score = math.log1p(max(profit, 0.0)) / math.log1p(500.0)
    profit_score -= math.log1p(max(-profit, 0.0)) / math.log1p(500.0)
    profit_score = max(-1.0, min(1.0, profit_score))

    # Tax burden is a mild penalty — locals accept reasonable taxes but
    # the penalty incentivises the local AI to negotiate lighter terms
    tax_penalty = computeTaxBurden(ctrl) * 0.1

    return (
        _LOCAL_REWARD_WEIGHTS["profit"]     * profit_score                          +
        _LOCAL_REWARD_WEIGHTS["balance"]    * computeSupplyChainBalance(ctrl)        +
        _LOCAL_REWARD_WEIGHTS["growth"]     * growth_score                           +
       -_LOCAL_REWARD_WEIGHTS["penalty"]    * computeEventPenalty(ctrl)              +
        _LOCAL_REWARD_WEIGHTS["population"] * (math.log1p(ctrl.population)
                                               / math.log1p(_POP_CAP))               +
        _LOCAL_REWARD_WEIGHTS["happiness"]  * computeAvgHappiness(ctrl)
        - tax_penalty
        - _treasury_target_penalty(ctrl)
        + dsp
    )


def _computeRulerReward(ctrl, prev_gdp: float) -> float:
    """Ruler-AI reward: economic strength + tax revenue + tech progression."""
    dsp = _delta_stagnation_penalty(ctrl)

    gdp = computeGDP(ctrl)
    prev_ema, curr_ema = _update_gdp_tracking(ctrl, gdp)
    if not _ownedBuildings(ctrl):
        return -0.10 + dsp

    gdp_growth   = math.log1p(curr_ema) - math.log1p(max(prev_ema, 1.0))
    growth_score = max(-1.0, min(1.0, gdp_growth * 5.0))

    tax_rev_score = math.log1p(ctrl._last_tax_revenue) / math.log1p(50.0)
    tax_rev_score = min(tax_rev_score, 1.0)

    return (
        _RULER_REWARD_WEIGHTS["strength"]   * computeEconomicStrength(ctrl)          +
        _RULER_REWARD_WEIGHTS["tax_rev"]    * tax_rev_score                           +
        _RULER_REWARD_WEIGHTS["growth"]     * growth_score                            +
        _RULER_REWARD_WEIGHTS["tech"]       * (ctrl.techLevel / MAX_TECH_LEVEL)       +
       -_RULER_REWARD_WEIGHTS["penalty"]    * computeEventPenalty(ctrl)               +
        _RULER_REWARD_WEIGHTS["mitigation"] * computeEventMitigationScore(ctrl)
        - _treasury_target_penalty(ctrl)
        + dsp
    )


def computeReward(ctrl, prev_gdp: float) -> float:
    """Role-aware composite reward. Dispatches to local or ruler variant."""
    if getattr(ctrl, 'ai_role', 'local') == "ruler":
        return _computeRulerReward(ctrl, prev_gdp)
    return _computeLocalReward(ctrl, prev_gdp)
