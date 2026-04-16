"""Rule-based AI agents: LocalAI and RulerAI.

LocalAI — private-profit focused
    • Maximises building revenues and supply-chain balance
    • Builds what is most demanded / scarce (demand-biased expansion)
    • Switches production chains to maximise profit margin

RulerAI — economic-strength focused
    • Ensures strategic resources (food, fuel, metal, tools) are produced first
    • Manages tax policy by economic phase (tech level)
    • Invests aggressively in infrastructure
    • Diversifies building types before duplicating

Both AIs use the same action-dict format consumed by MarketEnvironment._dispatch.

Depends on: types, structures, _globals, tick, metrics, taxation, environment
"""
from __future__ import annotations
import math

from .types      import itemTag
from .structures import ALL_BUILDING_TYPES, BT_FARM
from .tick       import (
    _INFRA_DRAIN_PER_POP, _INFRA_DRAIN_PER_BUILDING,
    _LOGISTICS_COST_PER_TRANSFER,
)
from .metrics    import computePrivateProfit, computeEconomicStrength
from .taxation   import TaxPolicy, TAX_POLICY_EARLY, TAX_POLICY_MID, TAX_POLICY_LATE
from .environment import MarketEnvironment


# =============================================================================
# SHARED HELPERS
# =============================================================================

def _infraStressed(ctrl) -> bool:
    """True if infra capacity is below the expected demand + logistics headroom."""
    infra     = ctrl.market.infrastructure
    buildings = list(ctrl.market.buildingList.values())
    expected  = (ctrl.population * _INFRA_DRAIN_PER_POP
                 + len(buildings) * _INFRA_DRAIN_PER_BUILDING)
    logistics = len(buildings) * 2 * _LOGISTICS_COST_PER_TRANSFER
    return infra.effective_capacity < expected + logistics


def _canAffordBuild(ctrl, btype) -> bool:
    """True only if ctrl has enough coins AND all required physical resources."""
    if ctrl.treasury < btype.construction_cost:
        return False
    for tag, qty in btype.resource_cost.items():
        available = sum(u.supply for u in ctrl.market.getItemsByTag(tag))
        if available < qty:
            return False
    return True


def _demandScoreForTag(ctrl, output_tag: itemTag) -> float:
    """Private-profit signal: price × (1 + scarcity) for a given output tag.

    High price + high scarcity = very profitable to produce.
    """
    units = ctrl.market.getItemsByTag(output_tag)
    if not units:
        return 0.0
    avg_price    = sum(u.itemPrice        for u in units) / len(units)
    avg_scarcity = sum(max(u.itemScarcity, 0.0) for u in units) / len(units)
    return avg_price * (1.0 + avg_scarcity)


def _supplyDeficitForTag(ctrl, output_tag: itemTag) -> float:
    """Supply-deficit ratio: max(0, demand-supply)/demand.

    1.0 = completely unmet demand.  0.0 = over-supplied.
    Used by both AIs for supply-chain-balance decisions.
    """
    units = ctrl.market.getItemsByTag(output_tag)
    if not units:
        return 1.0   # unknown tag → assume fully scarce
    total_demand = sum(u.demand for u in units)
    total_supply = sum(u.supply for u in units)
    if total_demand <= 0:
        return 0.0
    return max(0.0, (total_demand - total_supply) / total_demand)


def _bestChainForBuilding(ctrl, building):
    """Return (rule, score) for the most profitable chain for an idle building.

    Prefers chains that produce goods not yet produced anywhere (diversify first),
    then falls back to highest demand score.
    """
    already_produced = {tag
                        for b in ctrl.market.buildingList.values()
                        for chain in b.productionChains
                        for tag in chain.outputDict.keys()}
    bt = building.building_type
    visible = [
        r for r in ctrl.getVisibleRules()
        if r.outputTag in bt.allowed_output_tags
        and all(t in bt.allowed_input_tags for t in r.inputTags)
        and r.level <= bt.max_chain_level
    ]
    if not visible:
        return None, 0.0

    unique     = [r for r in visible if r.outputTag not in already_produced]
    candidates = unique if unique else visible

    def score(r):
        return _demandScoreForTag(ctrl, r.outputTag)

    best = max(candidates, key=score)
    return best, score(best)


# =============================================================================
# LOCAL AI
# =============================================================================
class LocalAI:
    """Private-profit focused agent for local market controllers.

    Goal: maximise building revenues relative to costs, balanced supply chains.

    Priority order each tick:
      0. Infrastructure if stressed (logistics bottleneck kills profit)
      1. First building → Farm (food is the foundation of any economy)
      2. Assign a chain to any chainless building (demand-biased)
      3. Expand every EXPAND_INTERVAL ticks — demand+deficit weighted
      4. Switch active chain to the highest profit option
      5. IDLE
    """
    EXPAND_INTERVAL  = 15
    _TREASURY_RESERVE = 0.25   # keep 25% of treasury before building

    def act(self, ctrl, tick: int) -> dict:
        buildings = list(ctrl.market.buildingList.values())

        # 0. Infrastructure
        if _infraStressed(ctrl) and ctrl.treasury >= ctrl.market.infrastructure.upgrade_cost():
            return {"type": MarketEnvironment.BUILD_INFRA, "params": {}}

        # 1. No buildings yet
        if not buildings:
            if _canAffordBuild(ctrl, BT_FARM):
                return {"type": MarketEnvironment.BUILD,
                        "params": {"building_type": BT_FARM}}
            return {"type": MarketEnvironment.IDLE, "params": {}}

        # 2. Chainless building → assign most-profitable chain
        for building in buildings:
            if not building.productionChains:
                rule, _ = _bestChainForBuilding(ctrl, building)
                if rule is not None:
                    return {"type": MarketEnvironment.ADD_CHAIN,
                            "params": {"building_id": building.id,
                                       "output_tag":  rule.outputTag,
                                       "input_tags":  rule.inputTags}}

        # 3. Expand — demand+deficit weighted, supply-chain balanced
        if tick % self.EXPAND_INTERVAL == ctrl.id % self.EXPAND_INTERVAL:
            action = self._expandAction(ctrl)
            if action is not None:
                return action

        # 4. Switch to highest-demand chain in a building with alternatives
        for building in buildings:
            if len(building.productionChains) > 1:
                best_idx = max(
                    range(len(building.productionChains)),
                    key=lambda i: _demandScoreForTag(
                        ctrl,
                        next(iter(building.productionChains[i].outputDict), itemTag.food),
                    ),
                )
                if best_idx != building.selectedChain:
                    return {"type": MarketEnvironment.SELECT_CHAIN,
                            "params": {"building_id": building.id,
                                       "chain_index": best_idx}}

        return {"type": MarketEnvironment.IDLE, "params": {}}

    def _expandAction(self, ctrl) -> dict | None:
        avg_lit = (sum(pc.literacy * pc.count for pc in ctrl.pop_classes)
                   / max(sum(pc.count for pc in ctrl.pop_classes), 1))
        active_outputs = {
            tag
            for b in ctrl.market.buildingList.values()
            if b.activeChain()
            for tag in b.activeChain().outputDict.keys()
        }

        def _affordable(btype) -> bool:
            reserve = ctrl.treasury * self._TREASURY_RESERVE
            if ctrl.treasury - btype.construction_cost < reserve:
                return False
            return _canAffordBuild(ctrl, btype)

        # Score each buildable type by demand + supply deficit
        best_btype, best_score = None, 0.0
        for btype in ALL_BUILDING_TYPES:
            if ctrl.techLevel < btype.required_tech:
                continue
            if avg_lit < btype.required_literacy:
                continue
            if not _affordable(btype):
                continue

            # Focus on output tags not yet covered first; otherwise any tag
            potential_tags = btype.allowed_output_tags - active_outputs
            if not potential_tags:
                potential_tags = btype.allowed_output_tags

            visible_tags = [t for t in potential_tags if ctrl.market.hasItemWithTag(t)]
            if not visible_tags:
                continue

            # Combined score: demand profit + supply deficit (biases toward gaps)
            demand_scores  = [_demandScoreForTag(ctrl, t) for t in visible_tags]
            deficit_scores = [_supplyDeficitForTag(ctrl, t) for t in visible_tags]
            avg_demand  = sum(demand_scores)  / len(demand_scores)
            avg_deficit = sum(deficit_scores) / len(deficit_scores)
            score = avg_demand * (1.0 + 2.0 * avg_deficit)   # deficit strongly biases

            if score > best_score:
                best_score = score
                best_btype = btype

        if best_btype is not None:
            return {"type": MarketEnvironment.BUILD,
                    "params": {"building_type": best_btype}}
        return None


# =============================================================================
# RULER AI
# =============================================================================

# Strategic tags: ruler ensures supply for these before anything else
_STRATEGIC_TAGS = {
    itemTag.food:   5.0,   # highest priority — collapse without it
    itemTag.fuel:   3.0,   # powers forge, kiln, brewery
    itemTag.tools:  4.0,   # every high-level chain needs tools
    itemTag.metal:  3.5,   # industrial base
    itemTag.lumber: 2.5,   # construction + fuel source
}


class RulerAI:
    """Economic-strength focused agent for ruler controllers.

    Goal: maximise GDP, tech advancement, and strategic resource coverage.
    Tax policy is adjusted each phase to balance growth vs. revenue.

    Priority order each tick:
      0. Infrastructure — ruler maintains 30% headroom (aggressive investment)
      1. Adjust tax policy to current tech phase
      2. First building → Farm
      3. Assign strategic chain to chainless buildings
      4. Expand every EXPAND_INTERVAL ticks — strategic gaps first
      5. Switch to most-strategic chain
      6. IDLE
    """
    EXPAND_INTERVAL = 10

    def act(self, ctrl, tick: int) -> dict:
        buildings = list(ctrl.market.buildingList.values())

        # 0. Infrastructure — ruler maintains 30% headroom over expected drain
        infra = ctrl.market.infrastructure
        expected = (ctrl.population * _INFRA_DRAIN_PER_POP
                    + len(buildings) * _INFRA_DRAIN_PER_BUILDING)
        if infra.effective_capacity < expected * 1.30:
            if ctrl.treasury >= infra.upgrade_cost():
                return {"type": MarketEnvironment.BUILD_INFRA, "params": {}}

        # 1. Phase-tune tax policy
        self._updateTaxPolicy(ctrl)

        # 2. First building → Farm
        if not buildings:
            if _canAffordBuild(ctrl, BT_FARM):
                return {"type": MarketEnvironment.BUILD,
                        "params": {"building_type": BT_FARM}}
            return {"type": MarketEnvironment.IDLE, "params": {}}

        # 3. Chainless building → strategic chain
        for building in buildings:
            if not building.productionChains:
                rule = self._strategicChainForBuilding(ctrl, building)
                if rule is None:
                    rule, _ = _bestChainForBuilding(ctrl, building)
                if rule is not None:
                    return {"type": MarketEnvironment.ADD_CHAIN,
                            "params": {"building_id": building.id,
                                       "output_tag":  rule.outputTag,
                                       "input_tags":  rule.inputTags}}

        # 4. Expand
        if tick % self.EXPAND_INTERVAL == ctrl.id % self.EXPAND_INTERVAL:
            action = self._expandAction(ctrl)
            if action is not None:
                return action

        # 5. Switch to most-strategic chain
        for building in buildings:
            if len(building.productionChains) > 1:
                best_idx = max(
                    range(len(building.productionChains)),
                    key=lambda i: self._strategicScore(
                        ctrl,
                        next(iter(building.productionChains[i].outputDict), itemTag.food),
                    ),
                )
                if best_idx != building.selectedChain:
                    return {"type": MarketEnvironment.SELECT_CHAIN,
                            "params": {"building_id": building.id,
                                       "chain_index": best_idx}}

        return {"type": MarketEnvironment.IDLE, "params": {}}

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _strategicScore(self, ctrl, output_tag: itemTag) -> float:
        """Score combining strategic importance and supply deficit."""
        weight  = _STRATEGIC_TAGS.get(output_tag, 1.0)
        deficit = _supplyDeficitForTag(ctrl, output_tag)
        demand  = _demandScoreForTag(ctrl, output_tag)
        return deficit * weight + demand * 0.2

    def _strategicChainForBuilding(self, ctrl, building):
        """Return the highest-strategic-priority rule compatible with this building.

        Strongly prefers producing output tags NOT already covered by another building,
        so each building diversifies the ruler's portfolio rather than stacking food.
        """
        bt = building.building_type
        visible = [
            r for r in ctrl.getVisibleRules()
            if r.outputTag in bt.allowed_output_tags
            and all(t in bt.allowed_input_tags for t in r.inputTags)
            and r.level <= bt.max_chain_level
        ]
        strategic = [r for r in visible if r.outputTag in _STRATEGIC_TAGS]
        if not strategic:
            return None

        already_covered = {tag
                           for b in ctrl.market.buildingList.values()
                           if b is not building
                           for chain in b.productionChains
                           for tag in chain.outputDict}

        # Prefer strategic outputs not yet covered; fall back to all strategic
        uncovered = [r for r in strategic if r.outputTag not in already_covered]
        candidates = uncovered if uncovered else strategic
        return max(candidates, key=lambda r: self._strategicScore(ctrl, r.outputTag))

    def _updateTaxPolicy(self, ctrl) -> None:
        """Set tax policy based on technology phase.

        Early game: low taxes — let local economy grow first.
        Mid game: moderate taxes — fund tech and infrastructure.
        Late game: higher taxes — accelerate industrialisation.
        """
        if ctrl.tax_policy is None:
            ctrl.tax_policy = TaxPolicy()
        if ctrl.techLevel <= 1:
            p = TAX_POLICY_EARLY
        elif ctrl.techLevel <= 3:
            p = TAX_POLICY_MID
        else:
            p = TAX_POLICY_LATE
        ctrl.tax_policy.income_tax_rate   = p.income_tax_rate
        ctrl.tax_policy.building_tax_rate = p.building_tax_rate
        ctrl.tax_policy.trade_tax_rate    = p.trade_tax_rate

    # Keep at least this fraction of treasury as reserve before building
    _TREASURY_RESERVE = 0.35

    def _expandAction(self, ctrl) -> dict | None:
        avg_lit = (sum(pc.literacy * pc.count for pc in ctrl.pop_classes)
                   / max(sum(pc.count for pc in ctrl.pop_classes), 1))
        existing_types = {b.building_type.name for b in ctrl.market.buildingList.values()}

        def _affordable_with_reserve(btype) -> bool:
            reserve = ctrl.treasury * self._TREASURY_RESERVE
            if ctrl.treasury - btype.construction_cost < reserve:
                return False
            return _canAffordBuild(ctrl, btype)

        # 4a. Build one of each new strategic-capable building type first.
        # No deficit threshold — ruler invests proactively for economic diversity.
        for btype in ALL_BUILDING_TYPES:
            if btype.name in existing_types:
                continue
            if ctrl.techLevel < btype.required_tech:
                continue
            if avg_lit < btype.required_literacy:
                continue
            if not _affordable_with_reserve(btype):
                continue
            # Does this building type produce at least one strategic tag?
            if btype.allowed_output_tags & _STRATEGIC_TAGS.keys():
                return {"type": MarketEnvironment.BUILD,
                        "params": {"building_type": btype}}

        # 4b. Duplicate only the most supply-deficit bottleneck building.
        # Requires actual scarcity signal before duplicating.
        best_bld, best_score = None, 0.40   # higher threshold than LocalAI
        for b in ctrl.market.buildingList.values():
            chain = b.activeChain()
            if chain is None or not _affordable_with_reserve(b.building_type):
                continue
            score = sum(
                self._strategicScore(ctrl, tag) for tag in chain.outputDict
            ) / max(len(chain.outputDict), 1)
            if score > best_score:
                best_score = score
                best_bld   = b

        if best_bld is not None:
            return {"type": MarketEnvironment.BUILD,
                    "params": {"building_type": best_bld.building_type}}
        return None
