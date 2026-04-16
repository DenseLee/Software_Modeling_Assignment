"""MarketController — one agent = one economy.

Depends on: types, structures, _globals
"""
from __future__ import annotations
import math
from typing import Optional

from .types       import itemTag, itemFlag, ActionResult, TradeDeal
from .structures  import (
    Market, PopClass, PopChangeEvent,
    productionGraphs, ProductionChain,
    BuildingType, Building, BT_WORKSHOP,
    MAX_TECH_LEVEL, TECH_TIER_NAMES,
    _BASE_TECH_ADVANCE_RATE, _POP_SCALE,
    BUILDING_FUNDS_CAP, BUILDING_SEED_CAPITAL,
)
from ._globals    import globalItemIndex, globalControllers
from ._rng        import _rng
from .taxation    import TaxPolicy


class MarketController:

    def __init__(self, id: int, name: str = "", starting_tech: int = 1,
                 starting_population: float = 1000,
                 shared_market: "Market | None" = None):
        self.id         = id
        self.name       = name or f"Controller_{id}"
        self.techLevel  = max(1, min(starting_tech, MAX_TECH_LEVEL))
        self.growthRate = 0.02

        if shared_market is not None:
            # Join an existing market — share items, buildings, population
            self._market = shared_market
            # Merge this controller's starting population into the shared pool
            new_classes = PopClass.defaultClasses(starting_population)
            if shared_market._shared_pop_classes:
                for new_pc, existing_pc in zip(new_classes, shared_market._shared_pop_classes):
                    existing_pc.count += new_pc.count
            else:
                shared_market._shared_pop_classes = new_classes
            shared_market._shared_pop_change_queue = shared_market._shared_pop_change_queue or []
        else:
            # Standalone market — owns its own population
            self._market = Market(id)
            self._market._shared_pop_classes      = PopClass.defaultClasses(starting_population)
            self._market._shared_pop_change_queue = []

        self.activeTradeDeals:  dict = {}
        self.pendingTradeDeals: dict = {}
        self.embargos:          set  = set()

        self.treasury: float = starting_population * 0.5

        self._current_tick:   int = 0

        self.crop_failure_remaining: int  = 0
        self.long_winter_remaining:  int  = 0
        self.active_events:          list = []

        # ── Role & taxation ───────────────────────────────────────────────────
        # "ruler"  — sets tax policy, receives tax revenue, advances tech faster
        # "local"  — pays taxes to ruler, focuses on private market profit
        self.ai_role:   str          = "local"
        self.ruler_id:  Optional[int] = None     # local → which controller ID is ruler
        self.tax_policy: Optional[TaxPolicy] = None   # ruler only
        self.tax_history: list       = []        # recent TaxRecord objects (last 20)
        self._last_tax_revenue: float    = 0.0   # tax income this tick
        self._cumulative_tax_revenue: float = 0.0  # lifetime tax income

        # Ruler strategic production targets: itemTag → "high"/"medium"/"low"
        self.strategic_targets: dict = {}

        # Register visible items — skip IDs already tracked (shared market may
        # have items from the ruler's earlier initialisation)
        for i in globalItemIndex.values():
            if i.level <= self.techLevel and i.id not in self._market.itemTracker:
                self._market.addItem(i)

    # ── Properties for shared-market access ──────────────────────────────────

    @property
    def market(self) -> "Market":
        return self._market

    @market.setter
    def market(self, value: "Market") -> None:
        self._market = value

    @property
    def pop_classes(self) -> list:
        return self._market._shared_pop_classes

    @pop_classes.setter
    def pop_classes(self, value: list) -> None:
        self._market._shared_pop_classes = value

    @property
    def population(self) -> float:
        return sum(pc.count for pc in self._market._shared_pop_classes)

    @population.setter
    def population(self, value: float) -> None:
        # Scale all class counts proportionally to hit the target total
        current = sum(pc.count for pc in self._market._shared_pop_classes)
        if current > 0 and value > 0:
            scale = value / current
            for pc in self._market._shared_pop_classes:
                pc.count = max(10.0, pc.count * scale)

    @property
    def pop_change_queue(self) -> list:
        return self._market._shared_pop_change_queue

    @pop_change_queue.setter
    def pop_change_queue(self, value: list) -> None:
        self._market._shared_pop_change_queue = value

    @property
    def _nextBuildingId(self) -> int:
        return self._market._nextBuildingId

    @_nextBuildingId.setter
    def _nextBuildingId(self, value: int) -> None:
        self._market._nextBuildingId = value

    # ── Visibility helpers ────────────────────────────────────────────────────

    def canSeeItem(self, item_instance) -> bool:
        return item_instance.level <= self.techLevel

    def getVisibleItems(self) -> list:
        return [i for i in globalItemIndex.values() if i.level <= self.techLevel]

    def canUseRule(self, rule) -> bool:
        visible = self.getVisibleItems()
        visible_tags = {tag for i in visible for tag in i.tags}
        return (rule.outputTag in visible_tags
                and all(t in visible_tags for t in rule.inputTags))

    def getVisibleRules(self) -> list:
        return [r for rules in productionGraphs.values() for r in rules if self.canUseRule(r)]

    # ── Building actions ──────────────────────────────────────────────────────

    def action_buildBuilding(self, building_type: BuildingType = None,
                             initial_chains: list = None) -> ActionResult:
        btype  = building_type or BT_WORKSHOP
        chains = initial_chains or []

        if self.techLevel < btype.required_tech:
            return ActionResult(False,
                f"{btype.name} requires tech level {btype.required_tech} "
                f"(current: {self.techLevel})")

        avg_literacy = (
            sum(pc.literacy * pc.count for pc in self.pop_classes)
            / max(sum(pc.count for pc in self.pop_classes), 1)
        )
        if avg_literacy < btype.required_literacy:
            return ActionResult(False,
                f"{btype.name} requires avg literacy {btype.required_literacy:.2f} "
                f"(current: {avg_literacy:.3f})")

        for chain in chains:
            if not Building(0, btype).isChainCompatible(chain):
                return ActionResult(False,
                    f"chain input/output tags incompatible with {btype.name}")
            for tag in chain.inputDict:
                if not any(tag in i.tags for i in self.getVisibleItems()):
                    return ActionResult(False,
                        f"tag '{tag.name}' not visible at tech level {self.techLevel}")

        total_cost = btype.construction_cost + BUILDING_SEED_CAPITAL
        if self.treasury < total_cost:
            return ActionResult(False,
                f"insufficient treasury: {btype.name} costs {btype.construction_cost:.0f} "
                f"+ {BUILDING_SEED_CAPITAL:.0f} seed capital = {total_cost:.0f}, "
                f"have {self.treasury:.1f}")

        # Check physical resource requirements (lumber, stone, metal …)
        for tag, qty in btype.resource_cost.items():
            available = sum(u.supply for u in self.market.getItemsByTag(tag))
            if available < qty:
                return ActionResult(False,
                    f"insufficient {tag.name} to build {btype.name}: "
                    f"need {qty:.1f}, have {available:.1f}")

        # Deduct construction cost + seed capital for the building's operational fund
        self.treasury -= btype.construction_cost + BUILDING_SEED_CAPITAL

        # Consume physical resources (cheapest units consumed first to preserve
        # higher-quality stock for production)
        for tag, qty in btype.resource_cost.items():
            remaining = qty
            for u in sorted(self.market.getItemsByTag(tag), key=lambda u: u.itemPrice):
                take = min(remaining, u.supply)
                u.supply = max(0.0, u.supply - take)
                remaining -= take
                if remaining <= 0:
                    break

        building = Building(self._nextBuildingId, btype, chains, selectedChain=0)
        building.owner_id = self.id
        building.funds    = BUILDING_SEED_CAPITAL
        self._nextBuildingId += 1
        self.market.buildingList[building.id] = building

        # Seed workforce — same availability + affordability logic as tickWageBalancing
        _PARTICIPATION = {1: 0.95, 2: 0.90, 3: 0.70, 4: 0.20}
        employed_by_tier: dict = {}
        for b in self.market.buildingList.values():
            for tier, count in b.workers.items():
                employed_by_tier[tier] = employed_by_tier.get(tier, 0.0) + count
        available_at_build: dict = {
            pc.tier: max(0.0, pc.count * _PARTICIPATION.get(pc.tier, 0.9)
                         - employed_by_tier.get(pc.tier, 0.0))
            for pc in self.pop_classes
        }
        min_tier = 1 if btype.max_chain_level <= 3 else 2
        for pc in sorted(self.pop_classes, key=lambda p: p.tier):
            if pc.tier < min_tier:
                continue
            available = available_at_build.get(pc.tier, 0.0)
            if available < 1.0:
                continue
            wage = _computeMinLivingWage(self, pc.tier)
            max_affordable = int(BUILDING_SEED_CAPITAL * 0.20 / max(wage, 0.001))
            hire = float(int(min(available * 0.01, float(max_affordable))))
            if hire < 1.0:
                continue
            building.workers[pc.tier]     = hire
            building.worker_wage[pc.tier] = wage
            available_at_build[pc.tier]   = max(0.0, available - hire)

        return ActionResult(True,
            f"{btype.name} #{building.id} constructed "
            f"(workers: {dict(building.workers)})", building)

    def action_demolishBuilding(self, building_id: int) -> ActionResult:
        if building_id not in self.market.buildingList:
            return ActionResult(False, f"building {building_id} not found")
        del self.market.buildingList[building_id]
        return ActionResult(True, f"building {building_id} demolished")

    def action_selectProductionChain(self, building_id: int, chain_index: int) -> ActionResult:
        building = self.market.buildingList.get(building_id)
        if building is None:
            return ActionResult(False, f"building {building_id} not found")
        if not (0 <= chain_index < len(building.productionChains)):
            return ActionResult(False,
                f"chain index {chain_index} out of range "
                f"(building has {len(building.productionChains)} chains)")
        building.selectedChain = chain_index
        chain = building.productionChains[chain_index]
        return ActionResult(True,
            f"building {building_id} switched to chain {chain_index}: "
            f"{[t.name for t in chain.inputDict]} → {[t.name for t in chain.outputDict]}", chain)

    def action_addChainToBuilding(self, building_id: int, chain: ProductionChain) -> ActionResult:
        building = self.market.buildingList.get(building_id)
        if building is None:
            return ActionResult(False, f"building {building_id} not found")
        if not building.isChainCompatible(chain):
            return ActionResult(False,
                f"chain tags incompatible with {building.building_type.name}: "
                f"inputs={[t.name for t in chain.inputDict]} "
                f"outputs={[t.name for t in chain.outputDict]}")
        for tag in chain.inputDict:
            if not any(tag in i.tags for i in self.getVisibleItems()):
                return ActionResult(False,
                    f"tag '{tag.name}' not accessible at tech level {self.techLevel}")
        building.productionChains.append(chain)
        return ActionResult(True,
            f"chain added to building {building_id} at index {len(building.productionChains)-1}", chain)

    def action_removeChainFromBuilding(self, building_id: int, chain_index: int) -> ActionResult:
        building = self.market.buildingList.get(building_id)
        if building is None:
            return ActionResult(False, f"building {building_id} not found")
        if not (0 <= chain_index < len(building.productionChains)):
            return ActionResult(False, f"chain index {chain_index} out of range")
        building.productionChains.pop(chain_index)
        if building.selectedChain >= len(building.productionChains):
            building.selectedChain = max(0, len(building.productionChains) - 1)
        return ActionResult(True, f"chain {chain_index} removed from building {building_id}")

    def action_setPriority(self, building_id: int, chain_index: int, priority: int) -> ActionResult:
        building = self.market.buildingList.get(building_id)
        if building is None:
            return ActionResult(False, f"building {building_id} not found")
        if not (0 <= chain_index < len(building.productionChains)):
            return ActionResult(False, f"chain index {chain_index} out of range")
        building.productionChains[chain_index].priority = priority
        return ActionResult(True, f"building {building_id} chain {chain_index} priority → {priority}")

    def action_requestProductionChain(
        self, output_tag: itemTag, proposed_input_tags: list,
        target_building_id: int = None,
    ) -> ActionResult:
        from .production import requestProductionChainCreation   # local to avoid circular
        output_items = [i for i in self.getVisibleItems() if output_tag in i.tags]
        if not output_items:
            return ActionResult(False,
                f"output tag '{output_tag.name}' not accessible at tech level {self.techLevel}")
        for tag in proposed_input_tags:
            if not any(tag in i.tags for i in self.getVisibleItems()):
                return ActionResult(False,
                    f"input tag '{tag.name}' not accessible at tech level {self.techLevel}")
        chain = requestProductionChainCreation(
            self.market, output_tag, proposed_input_tags, target_building_id)
        if chain is None:
            return ActionResult(False,
                f"environment rejected chain "
                f"{[t.name for t in proposed_input_tags]} → {output_tag.name}")
        return ActionResult(True,
            f"chain approved: {[t.name for t in proposed_input_tags]} → {output_tag.name}", chain)

    # ── Trade actions ─────────────────────────────────────────────────────────

    def action_proposeTradeDeal(
        self, target: MarketController,
        item_tag: itemTag, quantity_per_tick: float, price_per_unit: float,
    ) -> ActionResult:
        if target.id in self.embargos or self.id in target.embargos:
            return ActionResult(False,
                f"embargo blocks trade between {self.name} and {target.name}")
        items_for_tag = [i for i in globalItemIndex.values() if item_tag in i.tags]
        if not items_for_tag:
            return ActionResult(False, f"no item registered under tag '{item_tag.name}'")
        min_level = min(i.level for i in items_for_tag)
        if min_level > self.techLevel:
            return ActionResult(False,
                f"'{item_tag.name}' (tier {min_level}) above sender tech level {self.techLevel}")
        if min_level > target.techLevel:
            return ActionResult(False,
                f"'{item_tag.name}' (tier {min_level}) above receiver tech level {target.techLevel}")
        deal = TradeDeal(self.id, target.id, item_tag, quantity_per_tick, price_per_unit)
        target.pendingTradeDeals[deal.id] = deal
        return ActionResult(True, f"trade deal proposed to {target.name}: {deal}", deal)

    def action_respondToTradeDeal(self, deal_id: int, accept: bool) -> ActionResult:
        deal = self.pendingTradeDeals.pop(deal_id, None)
        if deal is None:
            return ActionResult(False, f"no pending deal #{deal_id}")
        if accept:
            deal.active = True
            self.activeTradeDeals[deal_id] = deal
            proposer = globalControllers.get(deal.from_controller_id)
            if proposer:
                proposer.activeTradeDeals[deal_id] = deal
            return ActionResult(True, f"deal #{deal_id} accepted: {deal}", deal)
        return ActionResult(True, f"deal #{deal_id} rejected")

    def action_cancelTradeDeal(self, deal_id: int) -> ActionResult:
        deal = self.activeTradeDeals.pop(deal_id, None)
        if deal is None:
            return ActionResult(False, f"no active deal #{deal_id}")
        other_id = (deal.to_controller_id
                    if deal.from_controller_id == self.id
                    else deal.from_controller_id)
        other = globalControllers.get(other_id)
        if other:
            other.activeTradeDeals.pop(deal_id, None)
        return ActionResult(True, f"deal #{deal_id} cancelled")

    def action_setEmbargo(self, target: MarketController) -> ActionResult:
        self.embargos.add(target.id)
        cancelled = []
        for deal_id, deal in list(self.activeTradeDeals.items()):
            if deal.from_controller_id == target.id or deal.to_controller_id == target.id:
                self.action_cancelTradeDeal(deal_id)
                cancelled.append(deal_id)
        return ActionResult(True,
            f"{self.name} embargoed {target.name}; cancelled deals: {cancelled}")

    def action_liftEmbargo(self, target: MarketController) -> ActionResult:
        if target.id not in self.embargos:
            return ActionResult(False, f"no embargo on {target.name}")
        self.embargos.discard(target.id)
        return ActionResult(True, f"{self.name} lifted embargo on {target.name}")

    def action_buildInfrastructure(self) -> ActionResult:
        infra = self.market.infrastructure
        cost  = infra.upgrade_cost()
        if self.techLevel < infra.level:
            return ActionResult(False,
                f"tech level {self.techLevel} too low to upgrade infra "
                f"to level {infra.level + 1} (need tech ≥ {infra.level})")
        if self.treasury < cost:
            return ActionResult(False,
                f"insufficient treasury: need {cost:.1f}, have {self.treasury:.1f}")
        self.treasury -= cost
        infra.level  += 1
        infra.wear    = max(0.0, infra.wear - 0.20)
        return ActionResult(True,
            f"infrastructure upgraded to level {infra.level} "
            f"(capacity {infra.effective_capacity:.0f}/{infra.max_capacity:.0f}, "
            f"cost {cost:.1f})", infra)

    # ── Tech advancement ──────────────────────────────────────────────────────

    def tryAdvanceTechLevel(self) -> bool:
        if self.techLevel >= MAX_TECH_LEVEL:
            return False
        controllers = list(globalControllers.values())
        if not controllers:
            return False
        avg_global_tech  = sum(c.techLevel for c in controllers) / len(controllers)
        pop_factor       = math.log1p(self.population) / math.log1p(_POP_SCALE)
        diffusion_factor = max(0.0, avg_global_tech - self.techLevel) / MAX_TECH_LEVEL

        # Rulers gain a technology bonus from their tax revenue —
        # high tax income means the ruler is investing in civilisational growth.
        # Locals advance slightly slower (they rely on the ruler for R&D).
        if self.ai_role == "ruler" and self._cumulative_tax_revenue > 0:
            # log-scaled bonus: doubles chance at ~100 cumulative revenue
            tax_factor = min(
                math.log1p(self._cumulative_tax_revenue) / math.log1p(200.0), 1.5
            )
        elif self.ai_role == "local":
            tax_factor = -0.15   # locals advance 15% slower
        else:
            tax_factor = 0.0

        chance = (_BASE_TECH_ADVANCE_RATE
                  * (1 + pop_factor)
                  * (1 + diffusion_factor)
                  * (1 + tax_factor))
        if _rng.random() < chance:
            self.techLevel += 1
            newly_unlocked = []
            for i in globalItemIndex.values():
                if i.level == self.techLevel and i.id not in self.market.itemTracker:
                    self.market.addItem(i)
                    newly_unlocked.append(i.name)
            tier_name = TECH_TIER_NAMES.get(self.techLevel, f"tier {self.techLevel}")
            print(f"[tech] {self.name} advanced to level {self.techLevel} ({tier_name})")
            if newly_unlocked:
                print(f"       unlocked items: {newly_unlocked}")
            return True
        return False

    def __repr__(self):
        return (f"MarketController({self.name} | tech={self.techLevel} "
                f"({TECH_TIER_NAMES.get(self.techLevel)}) | pop={self.population:.0f})")


# ── Helper used by action_buildBuilding (defined here to avoid circular import) ──

def _computeMinLivingWage(ctrl: MarketController, tier: int) -> float:
    pc = next((p for p in ctrl.pop_classes if p.tier == tier), None)
    if pc is None:
        return 0.0
    wage = 0.0
    for tag in (itemTag.food, itemTag.consumable):
        units = ctrl.market.getItemsByTag(tag)
        if not units:
            continue
        cheapest = min(units, key=lambda u: u.itemPrice)
        per_tick = pc.effectiveConsumption(tag, cheapest.itemInstance)
        wage    += cheapest.itemPrice * per_tick
    return max(0.01, wage)
