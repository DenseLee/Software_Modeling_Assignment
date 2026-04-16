"""All per-tick simulation functions.

Call order each tick (enforced by MarketEnvironment.step):
  1. tickInfrastructureDrain
  2. tickLogisticsInput
  3. tickSubsistence
  4. tickConsumption
  5. tickPopQueue
  6. tickProductionProgress
  7. tickLogisticsOutput
  8. tickWageBalancing
  9. tickBuildingViability
 10. tickProduction
 11. updateMarketPrices
 12. globalTick  (tech advances + random events + spawns)

Depends on: types, structures, _globals, _rng, production, seasons
"""
from __future__ import annotations
import math
from typing import Optional

from .types      import itemTag, itemFlag
from .structures import (
    Market, Building, ProductionChain, productionGraphs,
    PopChangeEvent, BUILDING_FUNDS_CAP,
)
from ._globals   import globalItemIndex, globalControllers
from ._rng       import _rng
from .production import simulateProduction, scoreItemForSlot, trySpawnItems
from .taxation   import tickTaxCollection
from .seasons    import (
    SEASON_WINTER, SEASON_FOOD_CONSUMPTION_MULT, FARM_SEASON_MULT,
    BUILDING_WINTER_MULT, getEffectiveSeason,
)


# =============================================================================
# CONSUMPTION CONSTANTS
# =============================================================================
_CONSUMABLE_TAGS          = (itemTag.food, itemTag.consumable, itemTag.alcohol, itemTag.luxury)


def _ownedBuildings(ctrl) -> list:
    """Return only the buildings owned by ctrl (shared-market aware)."""
    return [b for b in ctrl.market.buildingList.values() if b.owner_id == ctrl.id]
_NUTRITION_GROWTH_BONUS   = 0.005
_COMFORT_MORTALITY_REDUCTION = 0.0001

_SUBSISTENCE_FOOD_RATE    = 0.0025
_SUBSISTENCE_SEASON_MULT  = {0: 0.3, 1: 0.7, 2: 1.0, 3: 0.0}   # spring/summer/autumn/winter

_BIRTH_LAG_TICKS  = 12
_IMMEDIATE_MORT   = 0.0002

_PROGRESS_BASE    = 1.0


# =============================================================================
# SUBSISTENCE FARMING
# =============================================================================
def tickSubsistence(ctrl) -> None:
    """Unemployed workers subsistence-farm food directly into the market."""
    season = getEffectiveSeason(ctrl)
    smult  = _SUBSISTENCE_SEASON_MULT.get(season, 0.0)
    if smult <= 0.0:
        return

    participation = {1: 0.95, 2: 0.90, 3: 0.70, 4: 0.20}
    employed_by_tier: dict = {}
    for b in ctrl.market.buildingList.values():
        for tier, cnt in b.workers.items():
            employed_by_tier[tier] = employed_by_tier.get(tier, 0.0) + cnt

    total_unemployed = sum(
        max(0.0, pc.count * participation.get(pc.tier, 0.9)
            - employed_by_tier.get(pc.tier, 0.0))
        for pc in ctrl.pop_classes
    )
    food_produced = total_unemployed * _SUBSISTENCE_FOOD_RATE * smult
    food_items    = ctrl.market.getItemsByTag(itemTag.food)
    if food_items and food_produced > 0.0:
        cheapest = min(food_items, key=lambda u: u.itemPrice)
        cheapest.supply = min(cheapest.supply + food_produced, 5000.0)


# =============================================================================
# CONSUMPTION
# =============================================================================
def tickConsumption(ctrl) -> None:
    """Each pop class buys and consumes goods from the market."""
    for pc in ctrl.pop_classes:
        pc._nutrition_bonus = 0.0
        pc._comfort_bonus   = 0.0
        tag_satisfactions: list = []

        for tag in _CONSUMABLE_TAGS:
            cap = pc.consumption_caps.get(tag, 0.0)
            if cap <= 0 or pc.count <= 0:
                continue
            candidates = ctrl.market.getItemsByTag(tag)
            if not candidates:
                tag_satisfactions.append(0.0)
                continue

            eligible = [u for u in candidates
                        if not (tag == itemTag.food
                                and u.itemInstance.level < pc.food_level_min)]
            if not eligible:
                eligible = candidates

            def value_score(u):
                base_val = 2.0 ** (u.itemInstance.level - 1)
                if itemFlag.isValuable   in u.itemInstance.flags: base_val *= 1.5
                if itemFlag.isLuxury     in u.itemInstance.flags: base_val *= 2.0
                if itemFlag.isNutritional in u.itemInstance.flags: base_val *= 1.3
                return base_val / max(u.itemPrice, 0.001)

            eligible.sort(key=value_score, reverse=True)

            food_mult = (SEASON_FOOD_CONSUMPTION_MULT.get(getEffectiveSeason(ctrl), 1.0)
                         if tag == itemTag.food else 1.0)
            total_needed   = (pc.count
                              * pc.effectiveConsumption(tag, eligible[0].itemInstance)
                              * food_mult)
            total_consumed = 0.0

            for itu in eligible:
                if total_consumed >= total_needed:
                    break
                per_unit_sat = 1.0 + 0.5 * (itu.itemInstance.level - 1)
                each_needed  = max(0.0,
                    pc.count * pc.effectiveConsumption(tag, itu.itemInstance)
                    - total_consumed / per_unit_sat)
                consume = min(each_needed, max(0.0, itu.supply))
                itu.supply  = max(0.0, itu.supply - consume)
                itu.demand  = max(itu.demand,
                                  pc.count * pc.effectiveConsumption(tag, itu.itemInstance))

                # Deduct consumption cost from citizen funds.
                # The ruler's subsidy rate (0.0–0.5) covers that fraction
                # from treasury; citizens pay the rest.  Market prices are
                # NOT changed — only the citizen wallet is affected.
                if consume > 0.0 and pc.count > 0:
                    total_cost     = consume * itu.itemPrice
                    sub_rate       = min(getattr(ctrl, '_subsidy_rate', 0.0), 0.50)
                    treasury_share = total_cost * sub_rate
                    citizen_share  = total_cost - treasury_share
                    ctrl.treasury  = max(0.0, ctrl.treasury - treasury_share)
                    pc.avg_funds   = max(0.0, pc.avg_funds - citizen_share / pc.count)

                total_consumed += consume * per_unit_sat

            satisfaction = min(total_consumed / max(total_needed, 0.001), 1.0)
            tag_satisfactions.append(satisfaction)

            if tag == itemTag.food:
                pc._food_satisfaction = satisfaction
            if tag == itemTag.food and satisfaction >= 0.8:
                nutri = any(u.itemInstance.hasFlag(itemFlag.isNutritional)
                            for u in eligible[:3])
                pc._nutrition_bonus = (_NUTRITION_GROWTH_BONUS * satisfaction
                                       * (1.2 if nutri else 1.0))
            elif tag in (itemTag.consumable, itemTag.luxury) and satisfaction >= 0.5:
                pc._comfort_bonus += _COMFORT_MORTALITY_REDUCTION * satisfaction

        pc.satisfaction = sum(tag_satisfactions) / max(len(tag_satisfactions), 1)


# =============================================================================
# POPULATION DYNAMICS
# =============================================================================
def tickPopQueue(ctrl) -> None:
    """Mortality, birth queue, literacy, happiness, and social mobility."""
    current_tick = ctrl._current_tick

    # Phase 1: mortality
    for pc in ctrl.pop_classes:
        mort     = _IMMEDIATE_MORT + pc.mortality_rate
        comfort  = getattr(pc, '_comfort_bonus', 0.0)
        food_sat = getattr(pc, '_food_satisfaction', 0.5)
        starvation_mort = 0.0
        if food_sat < 0.35:
            starvation_mort = (0.35 - food_sat) / 0.35 * 0.008
        pc.count = max(10.0, pc.count * (1 - mort + comfort - starvation_mort))

    # Queued events
    due = [e for e in ctrl.pop_change_queue if e.apply_at_tick <= current_tick]
    ctrl.pop_change_queue = [e for e in ctrl.pop_change_queue
                             if e.apply_at_tick > current_tick]
    for event in due:
        pc = next((p for p in ctrl.pop_classes if p.tier == event.class_tier), None)
        if pc:
            pc.count = max(10.0, pc.count + event.delta)

    # Phase 2: births (food-suppressed)
    for pc in ctrl.pop_classes:
        nutrition  = getattr(pc, '_nutrition_bonus', 0.0)
        food_sat   = getattr(pc, '_food_satisfaction', 0.5)
        birth_mult = max(0.0, min(1.0, (food_sat - 0.15) / 0.55))
        births     = pc.count * (pc.birth_rate + nutrition) * birth_mult
        ctrl.pop_change_queue.append(PopChangeEvent(
            delta=births, apply_at_tick=current_tick + _BIRTH_LAG_TICKS,
            class_tier=pc.tier, reason="birth_cohort",
        ))

    # Phase 3: literacy
    for pc in ctrl.pop_classes:
        wr = getattr(pc, 'wage_ratio', 1.0)
        if wr > 1.2:
            pc.literacy = min(1.0, pc.literacy + 0.00005 * (wr - 1.0))
        elif wr < 0.8:
            pc.literacy = max(0.0, pc.literacy - 0.00002)

    # Phase 4: happiness
    # Happiness from funds: citizens are content when they can afford
    # several ticks of living costs in savings.  This replaces the old
    # wage-ratio signal — funds reflect actual purchasing power and
    # respond immediately to subsidies and unemployment.
    _FUNDS_HAPPY_TICKS = 10   # avg_funds target = 10× min living cost
    for pc in ctrl.pop_classes:
        min_w          = _computeMinLivingWage(ctrl, pc.tier)
        target_funds   = max(min_w * _FUNDS_HAPPY_TICKS, 0.001)
        funds_happiness = min(1.0, max(0.0, pc.avg_funds / target_funds))
        pc.happiness   = 0.6 * pc.satisfaction + 0.4 * funds_happiness

    # Phase 5: social mobility
    total_wealth = sum(pc.avg_funds * pc.count for pc in ctrl.pop_classes)
    total_pop    = max(sum(pc.count for pc in ctrl.pop_classes), 1.0)
    avg_wealth   = total_wealth / total_pop

    UPGRADE_MULT   = {1: 4.0, 2: 8.0, 3: 15.0}
    DOWNGRADE_MULT = {2: 0.4, 3: 0.7, 4: 1.2}

    tiers = {pc.tier: pc for pc in ctrl.pop_classes}
    for lower_tier, upper_tier in [(1, 2), (2, 3), (3, 4)]:
        lower = tiers.get(lower_tier)
        upper = tiers.get(upper_tier)
        if not lower or not upper:
            continue
        upgrade_threshold = avg_wealth * UPGRADE_MULT.get(lower_tier, 4.0)
        if lower.avg_funds >= upgrade_threshold and lower.count > 20:
            migrate      = lower.count * 0.002
            lower.count  = max(10.0, lower.count - migrate)
            upper.count += migrate
            upper.avg_funds = (upper.avg_funds * (upper.count - migrate)
                               + lower.avg_funds * migrate) / max(upper.count, 1)
        downgrade_threshold = avg_wealth * DOWNGRADE_MULT.get(upper_tier, 0.5)
        if upper.avg_funds < downgrade_threshold and upper.count > 10:
            migrate      = upper.count * 0.001
            upper.count  = max(10.0, upper.count - migrate)
            lower.count += migrate

    ctrl.population = sum(pc.count for pc in ctrl.pop_classes)


# =============================================================================
# PRODUCTION PROGRESS
# =============================================================================
_LOGISTICS_COST_PER_TRANSFER = 1.0
_BUFFER_TICKS_AHEAD          = 2
_INFRA_DRAIN_PER_POP         = 0.00005
_INFRA_DRAIN_PER_BUILDING    = 0.50


def _maintenanceProgressRate(building: Building, market: Market) -> float:
    """Return a progress rate multiplier [0.02, ~1.3] based on tools, infra, workers."""
    chain       = building.activeChain()
    chain_level = int(chain.timeToProduce / 10) if chain else 1
    needed      = max(0, chain_level - 1)

    # Tool availability
    if needed == 0:
        tool_rate = _PROGRESS_BASE
    else:
        tool_units = market.getItemsByTag(itemTag.tools)
        available  = sum(u.supply for u in tool_units)
        # Tool wear
        if tool_units and available > 0:
            wear     = min(available, needed * 0.08)
            total    = sum(u.supply for u in tool_units)
            for u in tool_units:
                fraction = u.supply / max(total, 0.001)
                u.supply = max(0.0, u.supply - wear * fraction)
        missing = max(0.0, needed - available)
        tool_rate = (_PROGRESS_BASE if missing <= 0
                     else max(0.02, 1.0 / (1.0 + math.log1p(missing) ** 1.5)))

    # Infra congestion
    infra       = market.infrastructure
    infra_util  = 1.0 - (infra.effective_capacity / max(infra.max_capacity, 1.0))
    congestion_penalty = max(0.0, (infra_util - 0.60) * 0.625)

    # Worker tier quality bonus
    tier_bonus    = 0.0
    total_workers = building.totalWorkers()
    if chain_level >= 3 and total_workers > 0:
        skilled    = sum(cnt for tier, cnt in building.workers.items() if tier >= 3)
        tier_bonus = 0.20 * (skilled / total_workers)

    return max(0.02, tool_rate * (1.0 + tier_bonus - congestion_penalty))


def tickProductionProgress(ctrl) -> None:
    """Advance production progress in owned buildings; trigger output cycles."""
    for building in _ownedBuildings(ctrl):
        chain = building.activeChain()
        if chain is None:
            continue

        is_farm = building.building_type.name == "Farm"
        season  = getEffectiveSeason(ctrl)
        if is_farm:
            if ctrl.crop_failure_remaining > 0:
                building.production_progress = 0.0
                continue
            season_mult = FARM_SEASON_MULT.get(season, 1.0)
        else:
            season_mult = BUILDING_WINTER_MULT if season == SEASON_WINTER else 1.0

        rate       = _maintenanceProgressRate(building, ctrl.market)
        efficiency = building.workerEfficiency()
        building.production_progress += (_PROGRESS_BASE * rate
                                         * max(efficiency, 0.1) * season_mult)

        if building.production_progress < chain.timeToProduce:
            continue

        building.production_progress = 0.0

        # Update efficiency rating (EMA)
        actual_rate = (rate * max(efficiency, 0.1) * season_mult)
        chain.efficiency_rating = (0.80 * chain.efficiency_rating
                                   + 0.20 * min(actual_rate, 1.0))

        # Check input buffer
        rule = next(
            (r for rules in productionGraphs.values()
             for r in rules
             if set(r.inputTags) == set(chain.inputDict.keys())
             and r.outputTag in chain.outputDict),
            None,
        )
        prefs = rule.inputPreferences if rule else {}

        input_ids: dict = {}
        buffer_ok = True
        for tag, needed in chain.inputDict.items():
            buffered_items = [
                (iid, qty) for iid, qty in building.itemInputBuffer.items()
                if iid in globalItemIndex and tag in globalItemIndex[iid].tags and qty > 0
            ]
            buffered_total = sum(q for _, q in buffered_items)
            if buffered_total < needed * 0.5:
                buffer_ok = False
                break
            pf      = prefs.get(tag, [])
            best_iid = max(
                buffered_items,
                key=lambda x: scoreItemForSlot(globalItemIndex[x[0]], tag, pf),
            )[0]
            input_ids[tag] = best_iid

        if not buffer_ok or not input_ids:
            chain.stall_ticks    += 1
            chain.efficiency_rating = max(0.0, chain.efficiency_rating - 0.05)
            continue

        for tag, needed in chain.inputDict.items():
            iid = input_ids.get(tag)
            if iid is not None:
                building.itemInputBuffer[iid] = max(
                    0.0, building.itemInputBuffer.get(iid, 0.0) - needed)

        produced = simulateProduction(building, ctrl.market, input_ids,
                                      mutation_chance=0.05, split_chance=0.02)
        for iid, qty in produced.items():
            building.outputBuffer[iid] = building.outputBuffer.get(iid, 0.0) + qty
        chain.cycle_count        += 1
        building._completed_cycle = True


# =============================================================================
# LOGISTICS
# =============================================================================
def tickInfrastructureDrain(ctrl) -> None:
    """Apply passive infrastructure wear from population and buildings."""
    infra = ctrl.market.infrastructure
    drain = (ctrl.population * _INFRA_DRAIN_PER_POP
             + len(ctrl.market.buildingList) * _INFRA_DRAIN_PER_BUILDING)
    infra.try_consume(drain)
    infra.tick_repair()


def tickLogisticsInput(ctrl) -> None:
    """Pull market items into owned building input buffers (priority-ordered)."""
    infra     = ctrl.market.infrastructure
    buildings = sorted(
        _ownedBuildings(ctrl),
        key=lambda b: b.activeChain().priority if b.activeChain() else 0,
        reverse=True,
    )
    for building in buildings:
        chain = building.activeChain()
        if chain is None:
            continue
        rule  = next(
            (r for rules in productionGraphs.values()
             for r in rules
             if set(r.inputTags) == set(chain.inputDict.keys())
             and r.outputTag in chain.outputDict),
            None,
        )
        prefs = rule.inputPreferences if rule else {}

        for tag, needed_per_cycle in chain.inputDict.items():
            target_qty = needed_per_cycle * _BUFFER_TICKS_AHEAD
            already    = sum(
                q for iid, q in building.itemInputBuffer.items()
                if iid in globalItemIndex
                and tag in globalItemIndex[iid].tags and q > 0
            )
            deficit = max(0.0, target_qty - already)
            if deficit <= 0:
                continue
            candidates = ctrl.market.getItemsByTag(tag)
            if not candidates:
                continue
            pf       = prefs.get(tag, [])
            best_itu = max(candidates,
                           key=lambda u: scoreItemForSlot(u.itemInstance, tag, pf))
            transfer = min(deficit, best_itu.supply)
            if transfer <= 0:
                continue
            granted = infra.try_consume(_LOGISTICS_COST_PER_TRANSFER)
            if granted < _LOGISTICS_COST_PER_TRANSFER * 0.01:
                return  # infrastructure exhausted
            fraction = granted / _LOGISTICS_COST_PER_TRANSFER
            actual   = transfer * fraction
            best_itu.supply = max(0.0, best_itu.supply - actual)
            iid = best_itu.itemInstance.id
            building.itemInputBuffer[iid] = (building.itemInputBuffer.get(iid, 0.0) + actual)


def tickLogisticsOutput(ctrl) -> None:
    """Move owned building output buffers into market supply."""
    infra      = ctrl.market.infrastructure
    _SUPPLY_CAP = 5000.0
    for building in _ownedBuildings(ctrl):
        if not building.outputBuffer:
            continue
        for iid, qty in list(building.outputBuffer.items()):
            if qty <= 0:
                building.outputBuffer[iid] = 0.0
                continue
            granted = infra.try_consume(_LOGISTICS_COST_PER_TRANSFER)
            if granted < _LOGISTICS_COST_PER_TRANSFER * 0.01:
                break
            fraction = granted / _LOGISTICS_COST_PER_TRANSFER
            transfer = qty * fraction
            itu = ctrl.market.itemTracker.get(iid)
            if itu is not None:
                itu.supply = min(_SUPPLY_CAP, itu.supply + transfer)
                building.outputBuffer[iid] = max(0.0, qty - transfer)
            else:
                building.outputBuffer[iid] = 0.0


# =============================================================================
# WAGE BALANCING
# =============================================================================
def _computeMinLivingWage(ctrl, tier: int) -> float:
    pc = next((p for p in ctrl.pop_classes if p.tier == tier), None)
    if pc is None:
        return 0.0
    wage = 0.0
    for tag in (itemTag.food, itemTag.consumable):
        units = ctrl.market.getItemsByTag(tag)
        if not units:
            continue
        cheapest  = min(units, key=lambda u: u.itemPrice)
        per_tick  = pc.effectiveConsumption(tag, cheapest.itemInstance)
        wage     += cheapest.itemPrice * per_tick
    return max(0.01, wage)


_LABOUR_PARTICIPATION = {1: 0.95, 2: 0.90, 3: 0.70, 4: 0.20}
_HIRE_RATE            = 0.01    # fraction of available unemployed hired per building per tick
_HIRE_FUNDS_FRACTION  = 0.20    # fraction of building funds that can be committed to new hires


def tickWageBalancing(ctrl) -> None:
    """Re-hire available workers, pay wages, and shed workers if building is broke.

    Hiring is gated by two real-world constraints:
      1. Labour availability — only unemployed workers can be hired; the pool
         is shared across all buildings so competition is natural.
      2. Affordability — a building can only hire as many workers as it can
         sustain; it spends at most _HIRE_FUNDS_FRACTION of its current funds
         on new hires this tick.
    There is no hard worker cap: workforce grows organically as the building
    earns revenue and the labour market tightens as employment rises.
    """
    # ── Market-wide labour availability ──────────────────────────────────
    # Count workers already employed across ALL buildings so we draw only
    # from the genuinely unemployed pool.
    employed_by_tier: dict = {}
    for b in ctrl.market.buildingList.values():
        for tier, count in b.workers.items():
            employed_by_tier[tier] = employed_by_tier.get(tier, 0.0) + count

    available_by_tier: dict = {
        pc.tier: max(0.0,
                     pc.count * _LABOUR_PARTICIPATION.get(pc.tier, 0.9)
                     - employed_by_tier.get(pc.tier, 0.0))
        for pc in ctrl.pop_classes
    }

    # ── Hire into owned buildings ─────────────────────────────────────────
    for building in _ownedBuildings(ctrl):
        low_complexity = building.building_type.max_chain_level <= 3
        min_tier       = 1 if low_complexity else 2

        for pc in sorted(ctrl.pop_classes, key=lambda p: p.tier):
            if pc.tier < min_tier:
                continue
            available = available_by_tier.get(pc.tier, 0.0)
            if available < 1.0:
                continue
            wage = _computeMinLivingWage(ctrl, pc.tier)
            # Can only hire what the building's funds can sustain
            max_affordable = int(building.funds * _HIRE_FUNDS_FRACTION
                                 / max(wage, 0.001))
            if max_affordable < 1:
                continue
            # Incremental: hire at most _HIRE_RATE of available unemployed this tick
            hire = float(int(min(available * _HIRE_RATE, float(max_affordable))))
            if hire < 1.0:
                continue
            building.workers[pc.tier]     = building.workers.get(pc.tier, 0.0) + hire
            building.worker_wage[pc.tier] = wage
            # Reduce the shared pool so later buildings compete for the same workers
            available_by_tier[pc.tier]    = max(0.0, available - hire)

    # Collect market wages — look across ALL buildings to reflect market-wide rates
    market_wages: dict = {}
    for building in ctrl.market.buildingList.values():
        for tier, count in building.workers.items():
            if count > 0:
                min_w = _computeMinLivingWage(ctrl, tier)
                market_wages[tier] = max(market_wages.get(tier, 0.0), min_w * 1.1)

    # Pay wages / fire if broke (owned buildings only)
    for building in _ownedBuildings(ctrl):
        chain = building.activeChain()
        revenue_est = 0.0
        if chain:
            for tag, qty in chain.outputDict.items():
                units = ctrl.market.getItemsByTag(tag)
                if units:
                    avg_p = sum(u.itemPrice for u in units) / len(units)
                    revenue_est += avg_p * qty / max(chain.timeToProduce, 1)

        profit_margin = (building.last_revenue / max(building.last_cost, 0.001)) - 1.0
        profit_bonus  = max(0.0, min(profit_margin * 0.15, 0.5))
        total_wage_bill = 0.0

        for tier in list(building.workers.keys()):
            count = building.workers[tier]
            if count <= 0:
                continue
            min_w = _computeMinLivingWage(ctrl, tier)
            wage  = max(min_w, market_wages.get(tier, 0.0)) * (1.0 + profit_bonus)
            building.worker_wage[tier] = wage
            pc = next((p for p in ctrl.pop_classes if p.tier == tier), None)
            if pc:
                pc.wage_ratio = wage / max(min_w, 0.001)
            bill = wage * count
            if building.funds >= bill:
                building.funds      -= bill
                total_wage_bill     += bill
                if pc and pc.count > 0:
                    pc.avg_funds += wage / pc.count
            else:
                affordable = max(0, int(building.funds / max(wage, 0.001)))
                building.workers[tier] = float(affordable)
                if affordable > 0:
                    building.funds      -= wage * affordable
                    total_wage_bill     += wage * affordable

        building.last_revenue = revenue_est
        building.last_cost    = total_wage_bill

        # Distribute estimated per-tick revenue:
        # 90 % → building operational fund (capped at BUILDING_FUNDS_CAP)
        # 10 % → controller's local fund (treasury)
        # If the building buffer is already at or above cap, 100 % goes to treasury.
        if revenue_est > 0:
            cap_room = max(0.0, BUILDING_FUNDS_CAP - building.funds)
            if cap_room <= 0:
                ctrl.treasury += revenue_est
            else:
                to_building = revenue_est * 0.90
                to_treasury = revenue_est * 0.10
                deposited   = min(to_building, cap_room)
                building.funds += deposited
                ctrl.treasury  += to_treasury + (to_building - deposited)

        profit = revenue_est - total_wage_bill
        if profit < 0 and building.totalWorkers() > 0 and building.productionChains:
            building.unprofitable_ticks += 1
        else:
            building.unprofitable_ticks = 0


# =============================================================================
# BUILDING VIABILITY
# =============================================================================
_UNPROFITABLE_SWITCH_THRESHOLD  = 5
_UNPROFITABLE_DESTROY_THRESHOLD = 10


def tickBuildingViability(ctrl) -> None:
    """Auto-switch or demolish chronically unprofitable owned buildings."""
    to_demolish = []
    for building in _ownedBuildings(ctrl):
        bid = building.id
        if (building.unprofitable_ticks >= _UNPROFITABLE_DESTROY_THRESHOLD
                and building.totalWorkers() == 0):
            to_demolish.append(bid)
            continue
        if (building.unprofitable_ticks >= _UNPROFITABLE_SWITCH_THRESHOLD
                and len(building.productionChains) > 1):
            best_idx = max(
                (i for i in range(len(building.productionChains))
                 if i != building.selectedChain),
                key=lambda i: building.productionChains[i].priority,
                default=None,
            )
            if best_idx is not None:
                building.selectedChain      = best_idx
                building.unprofitable_ticks = 0
                building.production_progress = 0.0
    for bid in to_demolish:
        print(f"[viability] Building #{bid} in {ctrl.name} demolished (chronic loss)")
        del ctrl.market.buildingList[bid]


# =============================================================================
# CURRENCY & PRICES
# =============================================================================
_CURRENCY_BASKET  = {"wheat": 0.40, "iron_ingot": 0.40, "salt": 0.20}
_BASE_ITEM_PRICE  = 1.0
_PRICE_ELASTICITY = 0.15
# Speculative pricing
_SPEC_HORIZON     = 50.0   # ticks of depletion lookahead: below this urgency kicks in
_SPEC_EMA_ALPHA   = 0.20   # smoothing factor for per-tick supply delta (≈5-tick window)


def computeCoinValue() -> float:
    """1 coin = weighted-average market price of basket commodities."""
    total_weight = 0.0
    total_value  = 0.0
    for name, weight in _CURRENCY_BASKET.items():
        ref = next((i for i in globalItemIndex.values() if i.name == name), None)
        if ref is None:
            continue
        prices = [
            ctrl.market.itemTracker[ref.id].itemPrice
            for ctrl in globalControllers.values()
            if ref.id in ctrl.market.itemTracker
            and ctrl.market.itemTracker[ref.id].itemPrice > 0
        ]
        if prices:
            total_value  += weight * (sum(prices) / len(prices))
            total_weight += weight
    return total_value / total_weight if total_weight > 0 else 1.0


def _initPrice(itu) -> float:
    base = _BASE_ITEM_PRICE * (2.0 ** (itu.itemInstance.level - 1))
    if itemFlag.isLuxury   in itu.itemInstance.flags: base *= 5.0
    if itemFlag.isValuable in itu.itemInstance.flags: base *= 2.0
    if itemFlag.isRare     in itu.itemInstance.flags: base *= 3.0
    if itemFlag.isCrude    in itu.itemInstance.flags: base *= 0.6
    return max(0.01, base)


def updateMarketPrices(market) -> None:
    """Adjust prices based on supply/demand ratio with speculative depletion weighting.

    Speculative logic: if supply is falling, estimate ticks-to-empty from the
    smoothed delta.  Within _SPEC_HORIZON ticks of running out, effective demand
    is linearly interpolated from actual demand toward supply (ratio → 1), so
    prices stabilise and rise *before* the item actually runs out rather than
    crashing to zero and spiking only at the last moment.
    """
    for itu in market.itemTracker.values():
        itu.supply = max(itu.supply, 0.1)
        if itu.itemPrice <= 0:
            itu.itemPrice = _initPrice(itu)

        # Update smoothed supply delta
        delta = itu.supply - itu.prev_supply
        itu.supply_delta_ema = (_SPEC_EMA_ALPHA * delta
                                + (1.0 - _SPEC_EMA_ALPHA) * itu.supply_delta_ema)
        itu.prev_supply = itu.supply

        # Speculative effective demand
        effective_demand = itu.demand
        if itu.supply_delta_ema < 0:
            ticks_to_empty = itu.supply / max(-itu.supply_delta_ema, 0.001)
            if ticks_to_empty < _SPEC_HORIZON:
                # urgency: 0 at horizon → 1 at zero ticks left
                urgency = 1.0 - ticks_to_empty / _SPEC_HORIZON
                # spec_demand bridges actual demand up toward supply level
                spec_demand = itu.demand + urgency * max(itu.supply - itu.demand, 0.0)
                effective_demand = max(itu.demand, spec_demand)

        ratio         = effective_demand / itu.supply
        raw           = itu.itemPrice * (1 + _PRICE_ELASTICITY * (ratio - 1))
        price_ceiling = _BASE_ITEM_PRICE * (2.0 ** (itu.itemInstance.level - 1)) * 500
        itu.itemPrice    = max(0.01, min(raw, price_ceiling))
        itu.itemScarcity = (itu.demand - itu.supply) / itu.supply


_ACCOUNTING_PRICE_MULT = 10.0   # cap at 10× item base value — prevents price-spike disasters


def _cappedPrice(units: list) -> float:
    """Average market price, capped at 10× the item's base value.

    Prevents volatile spot-price spikes from producing runaway treasury swings
    in the once-per-cycle accounting.
    """
    if not units:
        return 0.0
    avg_price = sum(u.itemPrice for u in units) / len(units)
    avg_level = sum(u.itemInstance.level for u in units) / len(units)
    base = _BASE_ITEM_PRICE * (2.0 ** (avg_level - 1))
    return min(avg_price, base * _ACCOUNTING_PRICE_MULT)


def tickProduction(ctrl) -> None:
    """Update chain metrics; credit treasury only when an owned building's cycle completes."""
    for building in _ownedBuildings(ctrl):
        chain = building.activeChain()
        if chain is None:
            continue
        ticks   = max(chain.timeToProduce, 1)
        revenue = 0.0
        for tag, qty in chain.outputDict.items():
            units = ctrl.market.getItemsByTag(tag)
            revenue += _cappedPrice(units) * qty / ticks
        cost = 0.0
        for tag, qty in chain.inputDict.items():
            units = ctrl.market.getItemsByTag(tag)
            cost += _cappedPrice(units) * qty / ticks
        # Store full-cycle estimates for AI/metrics display.
        # Revenue is distributed per-tick in tickWageBalancing (10/90 split),
        # so no lump-sum treasury credit here — just reset the cycle flag.
        chain.operating_revenue = revenue * ticks
        chain.operating_cost    = cost    * ticks
        if building._completed_cycle:
            building._completed_cycle = False


def tickPopulation(ctrl) -> None:
    """Grow/shrink population based on food availability."""
    food_units  = ctrl.market.getItemsByTag(itemTag.food)
    food_supply = sum(u.supply for u in food_units) if food_units else 0.0
    food_factor = min(food_supply / max(ctrl.population * 0.1, 1.0), 2.0)

    effective_rate = ctrl.growthRate * food_factor
    if food_factor < 0.5:
        effective_rate -= 0.02
    ctrl.population = max(100.0, ctrl.population * (1 + effective_rate))

    for tag in (itemTag.food, itemTag.consumable, itemTag.alcohol):
        for u in ctrl.market.getItemsByTag(tag):
            u.demand = (ctrl.population * 0.01
                        * (1 + 0.5 * (itemFlag.isLuxury in u.itemInstance.flags)))


# =============================================================================
# RANDOM EVENTS
# =============================================================================
_EVENT_CHANCE = 0.004


def _applyEarthquake(ctrl) -> None:
    buildings = list(ctrl.market.buildingList.keys())
    if not buildings:
        return
    n_destroy = max(1, int(len(buildings) * 0.15))
    victims   = _rng.sample(buildings, min(n_destroy, len(buildings)))
    for bid in victims:
        del ctrl.market.buildingList[bid]
    for pc in ctrl.pop_classes:
        pc.count = max(10.0, pc.count * 0.94)
    msg = f"Earthquake! {len(victims)} building(s) destroyed, ~6% population casualties."
    ctrl.active_events.append(msg)
    print(f"[EVENT] {ctrl.name}: {msg}")


def _applyCropFailure(ctrl) -> None:
    duration = _rng.randint(15, 30)
    ctrl.crop_failure_remaining = max(ctrl.crop_failure_remaining, duration)
    for b in ctrl.market.buildingList.values():
        if b.building_type.name == "Farm":
            b.production_progress = 0.0
    msg = f"Crop failure! Farms halted for {duration} ticks."
    ctrl.active_events.append(msg)
    print(f"[EVENT] {ctrl.name}: {msg}")


def _applyLongWinter(ctrl) -> None:
    extension = _rng.randint(10, 25)
    ctrl.long_winter_remaining = max(ctrl.long_winter_remaining, extension)
    msg = f"Long winter! {extension} extra winter ticks."
    ctrl.active_events.append(msg)
    print(f"[EVENT] {ctrl.name}: {msg}")


def randomEventFunction() -> None:
    """Roll for random natural events on each controller."""
    for ctrl in sorted(globalControllers.values(), key=lambda c: c.id):
        if _rng.random() > _EVENT_CHANCE:
            continue
        roll = _rng.random()
        if roll < 0.30:
            _applyEarthquake(ctrl)
        elif roll < 0.65:
            _applyCropFailure(ctrl)
        else:
            _applyLongWinter(ctrl)


def globalTick() -> None:
    """Tech advancement, tax collection, item spawning, events — once per tick."""
    ctrls = sorted(globalControllers.values(), key=lambda c: c.id)

    # Reset per-tick tax counters
    for ctrl in ctrls:
        ctrl._last_tax_revenue = 0.0

    # Tax collection: each local controller pays its ruler
    for ctrl in ctrls:
        if ctrl.ruler_id is None:
            continue
        ruler = globalControllers.get(ctrl.ruler_id)
        if ruler is None or ruler.tax_policy is None:
            continue
        record = tickTaxCollection(ctrl, ruler)
        ruler._last_tax_revenue     += record.total
        ruler._cumulative_tax_revenue += record.total
        # Keep a rolling window of the last 20 records for analysis
        ruler.tax_history.append(record)
        if len(ruler.tax_history) > 20:
            ruler.tax_history.pop(0)

    # Tech advancement (tax revenue already updated above)
    for ctrl in ctrls:
        ctrl.tryAdvanceTechLevel()

    trySpawnItems()

    for ctrl in ctrls:
        if ctrl.crop_failure_remaining > 0:
            ctrl.crop_failure_remaining -= 1
        if ctrl.long_winter_remaining > 0:
            ctrl.long_winter_remaining -= 1
        if ctrl.active_events:
            ctrl.active_events.clear()

    randomEventFunction()
