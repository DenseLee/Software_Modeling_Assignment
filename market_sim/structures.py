"""Mid-level data structures: production rules, buildings, market, population.

Depends on: types.py
"""
from __future__ import annotations
from typing import Optional
from dataclasses import dataclass, field
import math



from ._rng import _rng
from .types import itemTag, itemFlag, item, qualityTier


# =============================================================================
# PRODUCTION RULE & CHAIN
# =============================================================================
class productionRule:
    def __init__(
        self,
        inputTags:        list,
        outputTag:        itemTag,
        level:            int,
        inputPreferences: dict = None,
        qualityTiers:     list = None,
    ):
        self.inputTags        = inputTags
        self.outputTag        = outputTag
        self.level            = level
        self.inputPreferences = inputPreferences or {}
        self.qualityTiers     = sorted(qualityTiers or [], key=lambda t: t.threshold, reverse=True)


productionGraphs: dict = {}     # itemTag → list[productionRule]

def registerProductionRule(rule: productionRule):
    productionGraphs.setdefault(rule.outputTag, []).append(rule)


class ProductionChain:
    def __init__(
        self,
        inputDict:         dict,
        outputDict:        dict,
        timeToProduce:     float,
        priority:          int,
        inputPreferences:  dict = None,
        qualityTiers:      list = None,
    ):
        self.inputDict        = inputDict
        self.outputDict       = outputDict
        self.timeToProduce    = timeToProduce
        self.priority         = priority
        self.inputPreferences = inputPreferences or {}
        self.qualityTiers     = qualityTiers or []
        # Efficiency & cost tracking
        self.cycle_count:       int   = 0
        self.stall_ticks:       int   = 0
        self.efficiency_rating: float = 1.0
        self.operating_cost:    float = 0.0
        self.operating_revenue: float = 0.0


# =============================================================================
# BUILDING TYPE & BUILDING
# =============================================================================
@dataclass
class BuildingType:
    name:                  str
    allowed_input_tags:    frozenset
    allowed_output_tags:   frozenset
    base_tool_requirement: int
    max_chain_level:       int   = 5
    required_tech:         int   = 1
    required_literacy:     float = 0.0
    construction_cost:     float = 100.0
    # Physical resources consumed from market at build time.
    # Keys are itemTag; values are quantities drawn from market supply.
    # Low-tier buildings use lumber; high-tier add stone/metal/machinery.
    resource_cost:         dict  = field(default_factory=dict)


BT_FARM      = BuildingType("Farm",
    frozenset({itemTag.crops, itemTag.livestock}),
    frozenset({itemTag.food, itemTag.fiber, itemTag.livestock, itemTag.alcohol}),
    base_tool_requirement=0, max_chain_level=3, required_tech=1, required_literacy=0.00,
    construction_cost=80.0,
    resource_cost={itemTag.lumber: 5.0})                   # timber frame + fencing

BT_FORGE     = BuildingType("Forge",
    frozenset({itemTag.ore, itemTag.metal, itemTag.fuel, itemTag.coke, itemTag.alloy}),
    frozenset({itemTag.metal, itemTag.alloy, itemTag.tools, itemTag.weapons, itemTag.ammunition}),
    base_tool_requirement=2, max_chain_level=5, required_tech=1, required_literacy=0.05,
    construction_cost=250.0,
    resource_cost={itemTag.lumber: 4.0, itemTag.ore: 4.0})  # structure + heat-sink stone

BT_WORKSHOP  = BuildingType("Workshop",
    frozenset({itemTag.metal, itemTag.alloy, itemTag.lumber, itemTag.leather, itemTag.fabric}),
    frozenset({itemTag.tools, itemTag.consumable, itemTag.weapons, itemTag.armor, itemTag.machinery}),
    base_tool_requirement=1, max_chain_level=5, required_tech=2, required_literacy=0.15,
    construction_cost=200.0,
    resource_cost={itemTag.lumber: 8.0, itemTag.metal: 2.0})  # benches + metal fixtures

BT_MILL      = BuildingType("Mill",
    frozenset({itemTag.crops, itemTag.fiber, itemTag.lumber}),
    frozenset({itemTag.food, itemTag.fabric, itemTag.paper}),
    base_tool_requirement=1, max_chain_level=4, required_tech=1, required_literacy=0.05,
    construction_cost=150.0,
    resource_cost={itemTag.lumber: 12.0})                  # millwheel frame + gears

BT_TANNERY   = BuildingType("Tannery",
    frozenset({itemTag.livestock, itemTag.leather}),
    frozenset({itemTag.leather, itemTag.consumable, itemTag.armor}),
    base_tool_requirement=1, max_chain_level=3, required_tech=1, required_literacy=0.05,
    construction_cost=120.0,
    resource_cost={itemTag.lumber: 6.0})                   # vats + drying racks

BT_BREWERY   = BuildingType("Brewery",
    frozenset({itemTag.crops, itemTag.spice, itemTag.fuel}),
    frozenset({itemTag.alcohol, itemTag.food}),
    base_tool_requirement=0, max_chain_level=3, required_tech=1, required_literacy=0.00,
    construction_cost=100.0,
    resource_cost={itemTag.lumber: 5.0})                   # casks + structure

BT_KILN      = BuildingType("Kiln",
    frozenset({itemTag.stone, itemTag.lumber, itemTag.fuel, itemTag.ceramic, itemTag.miscellaneous}),
    frozenset({itemTag.ceramic, itemTag.glass, itemTag.stone}),
    base_tool_requirement=1, max_chain_level=4, required_tech=2, required_literacy=0.10,
    construction_cost=180.0,
    resource_cost={itemTag.stone: 6.0, itemTag.lumber: 4.0})  # stone chamber + scaffold

BT_ALCHEMIST = BuildingType("Alchemist",
    frozenset({itemTag.chemical, itemTag.ore, itemTag.sulfur, itemTag.saltpeter, itemTag.fuel, itemTag.miscellaneous}),
    frozenset({itemTag.chemical, itemTag.explosive, itemTag.medicine, itemTag.dye}),
    base_tool_requirement=2, max_chain_level=5, required_tech=3, required_literacy=0.40,
    construction_cost=400.0,
    resource_cost={itemTag.stone: 8.0, itemTag.metal: 4.0})   # reinforced lab + equipment

BT_FACTORY   = BuildingType("Factory",
    frozenset({itemTag.metal, itemTag.alloy, itemTag.fuel, itemTag.coal, itemTag.rubber, itemTag.machinery}),
    frozenset({itemTag.machinery, itemTag.tools, itemTag.weapons, itemTag.ammunition}),
    base_tool_requirement=3, max_chain_level=5, required_tech=4, required_literacy=0.60,
    construction_cost=600.0,
    resource_cost={itemTag.metal: 8.0, itemTag.stone: 10.0})  # heavy frame + foundations

ALL_BUILDING_TYPES: list = [
    BT_FARM, BT_FORGE, BT_WORKSHOP, BT_MILL, BT_TANNERY,
    BT_BREWERY, BT_KILN, BT_ALCHEMIST, BT_FACTORY,
]


BUILDING_FUNDS_CAP    = 2000.0   # max operational reserve a building may hold
BUILDING_SEED_CAPITAL =  200.0   # coins transferred from treasury at construction


class Building:
    def __init__(self, id: int, building_type: BuildingType,
                 productionChains: list = None, selectedChain: int = 0):
        self.id               = id
        self.building_type    = building_type
        self.productionChains = productionChains or []
        self.selectedChain    = selectedChain

        self.itemInputBuffer: dict = {}   # item_id → quantity
        self.outputBuffer:    dict = {}   # item_id → quantity

        self.production_progress: float = 0.0
        self.funds:          float = 0.0   # funded explicitly at construction
        self.workers:        dict  = {}   # tier → count
        self.worker_wage:    dict  = {}   # tier → wage

        self.unprofitable_ticks: int   = 0
        self.last_revenue:       float = 0.0
        self.last_cost:          float = 0.0
        self._completed_cycle:   bool  = False  # set True in tickProductionProgress when cycle fires
        self.owner_id:           int   = 0      # controller id that built this building

    def activeChain(self) -> Optional[ProductionChain]:
        if 0 <= self.selectedChain < len(self.productionChains):
            return self.productionChains[self.selectedChain]
        return None

    def isChainCompatible(self, chain: ProductionChain) -> bool:
        bt = self.building_type
        for tag in chain.inputDict:
            if tag not in bt.allowed_input_tags:
                return False
        for tag in chain.outputDict:
            if tag not in bt.allowed_output_tags:
                return False
        return True

    def totalWorkers(self) -> float:
        return sum(self.workers.values())

    @property
    def target_workers(self) -> int:
        """Workforce needed for 100 % production efficiency (soft guide, not a cap).
        Simple low-complexity buildings absorb many workers; specialist buildings
        need fewer but more skilled people."""
        mcl = self.building_type.max_chain_level
        if mcl <= 3:   return 400   # farms, breweries, tanneries — labour-intensive
        elif mcl == 4: return 300   # mills, kilns — moderate staffing
        else:          return 200   # forges, workshops, factories — skill over numbers

    def workerEfficiency(self) -> float:
        return min(self.totalWorkers() / max(self.target_workers, 1), 1.0)


# =============================================================================
# ITEM TRACK UNIT & MARKET
# =============================================================================
class itemTrackUnit:
    def __init__(self, item_instance: item, initial_supply: float = 200.0):
        self.itemInstance  = item_instance
        self.supply:  float = initial_supply
        self.demand:  float = 10
        self.itemPrice:    float = 0.0
        self.itemScarcity: float = 0.0
        # Speculative pricing state
        self.prev_supply:       float = initial_supply  # supply at previous tick
        self.supply_delta_ema:  float = 0.0             # EMA of per-tick supply change


class Market:
    def __init__(self, id: int):
        self.id           = id
        self.buildingList: dict  = {}   # building_id → Building
        self.itemTracker:  dict  = {}   # item_id → itemTrackUnit
        self._tagIndex:    dict  = {}   # itemTag → set[item_id]
        self.infrastructure = Infrastructure()
        # Shared state — used when multiple controllers share this market
        self._shared_pop_classes:      list  = []   # shared PopClass list
        self._shared_pop_change_queue: list  = []   # shared birth/event queue
        self._nextBuildingId:          int   = 1    # monotone ID counter across all owners

    def addItem(self, item_instance: item, initial_supply: float = 200.0):
        self.itemTracker[item_instance.id] = itemTrackUnit(item_instance, initial_supply)
        for tag in item_instance.tags:
            self._tagIndex.setdefault(tag, set()).add(item_instance.id)

    def hasItemWithTag(self, tag: itemTag) -> bool:
        return bool(self._tagIndex.get(tag))

    def getItemsByTag(self, tag: itemTag) -> list:
        return [self.itemTracker[iid] for iid in self._tagIndex.get(tag, set())]


# =============================================================================
# INFRASTRUCTURE
# =============================================================================
class Infrastructure:
    BASE_CAPACITY      = 60.0
    CAPACITY_PER_LEVEL = 60.0
    WEAR_PER_UNIT      = 0.0002
    PASSIVE_REPAIR     = 0.005
    UPGRADE_COST_BASE  = 500.0

    def __init__(self):
        self.level: int   = 1
        self.wear:  float = 0.0

    @property
    def max_capacity(self) -> float:
        return self.BASE_CAPACITY + self.CAPACITY_PER_LEVEL * (self.level - 1)

    @property
    def effective_capacity(self) -> float:
        return max(0.0, self.max_capacity * (1.0 - self.wear))

    def try_consume(self, units: float) -> float:
        granted   = min(units, self.effective_capacity)
        self.wear = min(1.0, self.wear + granted * self.WEAR_PER_UNIT)
        return granted

    def tick_repair(self):
        self.wear = max(0.0, self.wear - self.PASSIVE_REPAIR)

    def upgrade_cost(self) -> float:
        return self.UPGRADE_COST_BASE * (self.level ** 2)

    def __repr__(self):
        return (f"Infrastructure(level={self.level}  "
                f"capacity={self.effective_capacity:.1f}/{self.max_capacity:.1f}  "
                f"wear={self.wear:.3f})")


# =============================================================================
# TECH TIER
# =============================================================================
MAX_TECH_LEVEL = 5
TECH_TIER_NAMES = {
    1: "Early Medieval",
    2: "Medieval",
    3: "Early Modern",
    4: "Early Industrial",
    5: "Late Industrial",
}

_BASE_TECH_ADVANCE_RATE = 0.0008
_POP_SCALE              = 5000.0


# =============================================================================
# POPULATION
# =============================================================================
@dataclass
class PopChangeEvent:
    delta:         float
    apply_at_tick: int
    class_tier:    int
    reason:        str = ""


class PopClass:
    def __init__(
        self,
        tier:           int,
        name:           str,
        count:          float,
        literacy:       float,
        avg_funds:      float,
        birth_rate:     float,
        mortality_rate: float,
        food_level_min: int,
        consumption_caps: dict = None,
    ):
        self.tier           = tier
        self.name           = name
        self.count          = count
        self.literacy       = literacy
        self.avg_funds      = avg_funds
        self.birth_rate     = birth_rate
        self.mortality_rate = mortality_rate
        self.food_level_min = food_level_min
        # Per-person per-tick consumption.  Calibrated so 2 level-1 farms
        # (each producing 80 food per ~17-tick cycle ≈ 4.7/tick) can feed
        # a 1 000-person population before additional farms come online.
        default_caps = {
            itemTag.food:       0.003 * tier,     # was 0.005 — halved for stability
            itemTag.consumable: 0.002 * tier,
            itemTag.alcohol:    0.001 * tier,
            itemTag.luxury:     0.001 * (tier - 1),
        }
        self.consumption_caps: dict = consumption_caps or default_caps
        self.satisfaction:  float = 0.5
        self.happiness:     float = 0.5
        self.wage_ratio:    float = 1.0

    def effectiveConsumption(self, tag: itemTag, item_instance: item) -> float:
        if tag == itemTag.food and item_instance.level < self.food_level_min:
            return 0.0
        base = self.consumption_caps.get(tag, 0.0)
        if base <= 0:
            return 0.0
        level_mult = 1.0 + 0.5 * (item_instance.level - 1)
        flag_mult  = 1.0
        if item_instance.hasFlag(itemFlag.isNutritional): flag_mult *= 0.80
        if item_instance.hasFlag(itemFlag.isLuxury):      flag_mult *= 0.70
        if item_instance.hasFlag(itemFlag.isValuable):    flag_mult *= 0.90
        return base / (level_mult * flag_mult)

    @staticmethod
    def defaultClasses(total_pop: float) -> list:
        return [
            PopClass(1, "Peasant",  total_pop * 0.60, literacy=0.05, avg_funds=5.0,
                     birth_rate=0.0004, mortality_rate=0.00004, food_level_min=1),
            PopClass(2, "Artisan",  total_pop * 0.25, literacy=0.35, avg_funds=25.0,
                     birth_rate=0.0003, mortality_rate=0.00003, food_level_min=1),
            PopClass(3, "Merchant", total_pop * 0.10, literacy=0.75, avg_funds=150.0,
                     birth_rate=0.0002, mortality_rate=0.00002, food_level_min=2),
            PopClass(4, "Noble",    total_pop * 0.05, literacy=0.95, avg_funds=800.0,
                     birth_rate=0.0001, mortality_rate=0.00001, food_level_min=2),
        ]

    def __repr__(self):
        return f"PopClass({self.name} ×{self.count:.0f} funds={self.avg_funds:.1f})"
