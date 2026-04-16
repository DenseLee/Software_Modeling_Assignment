"""Economy bootstrap: calibrated initial buildings, production chains, and supplies.

Calling bootstrapEconomy(ctrl) gives a controller a running economy from tick 0
rather than starting with a blank market.  This prevents the early-game
food-scarcity spiral that collapses economies before the AI can react.

Baseline buildings per role
────────────────────────────
Both (before role split):
  Farm  #1  crops       → food       (feed the population)
  Forge #1  ore + fuel  → metal      (industrial foundation)

Ruler (Kingdom) additionally:
  Farm  #2  fiber       → fabric     (clothing → happiness)
  Mill  #1  crops       → food       (extra food throughput via milling)

Local (Merchant) additionally:
  Farm  #2  livestock   → fiber      (raw material for traders)
  Brewery#1 crops+fuel  → alcohol    (high-margin consumer good)

Key calibration decisions
────────────────────────────
• Starting supplies of basket commodities boosted so coin_value ≈ 1.0
  and the economy survives 200+ ticks of AI ramp-up without collapse.
• Bootstrap buildings hire at 2 % of pop (10× normal rate).
• Building input buffers pre-loaded with 3 cycles of raw material so
  production starts on tick 0 without waiting for logistics.
• Starting treasury raised to 1 500 so role-specific buildings are
  affordable with a meaningful reserve remaining.

Depends on: types, structures, _globals, production
"""
from __future__ import annotations

from .types      import itemTag
from .structures import (
    Building,
    BT_FARM, BT_FORGE, BT_MILL, BT_BREWERY,
    BUILDING_FUNDS_CAP, BUILDING_SEED_CAPITAL,
)
from ._globals   import globalItemIndex
from .production import requestProductionChainCreation


# =============================================================================
# SUPPLY TABLE
# =============================================================================
# Absolute supply values set at bootstrap.  Items not listed keep their
# default (200).  Basket commodities (wheat, iron_ingot, salt) are set high
# enough that the initial coin_value stays near 1.0.

_INITIAL_SUPPLIES: dict = {
    # ── Food & Crops ───────────────────────────────────────────────────────
    # Bread/ale have only the 'food' tag, not 'crops', so they serve as a
    # pure consumption buffer without being pulled into farm input buffers.
    "wheat":        2500.0,   # primary food crop — also farm input
    "rye":          1500.0,   # secondary grain
    "bread":        2000.0,   # processed buffer: consumed but never used as input
    "ale":           800.0,   # alcohol-food buffer
    "mead":          500.0,   # alcohol-food buffer
    # ── Fiber & Livestock ──────────────────────────────────────────────────
    "raw_wool":      800.0,   # fiber → fabric → clothing
    "hide":          500.0,   # tannery input
    # ── Fuel & Construction ────────────────────────────────────────────────
    "timber":        800.0,   # lumber: buildings + charcoal
    "charcoal":      700.0,   # forge + kiln fuel
    # ── Ores & Metals ──────────────────────────────────────────────────────
    "iron_ore":      600.0,   # forge input
    "iron_ingot":    400.0,   # coin basket + tools input (pre-smelted)
    "iron_tool":     200.0,   # tools → production efficiency bonus
    # ── Spice / Trade ──────────────────────────────────────────────────────
    "salt":          400.0,   # coin basket + trade good
    # ── Consumer Goods ─────────────────────────────────────────────────────
    "clay_pots":     300.0,
    "wax_candle":    250.0,
    "basic_clothes": 300.0,   # consumable → satisfaction
}


# =============================================================================
# INTERNAL HELPERS
# =============================================================================
_BOOTSTRAP_HIRE_RATE = 0.020   # 2 % of class count, vs 0.2 % normally


def _boostSupplies(ctrl) -> None:
    """Set initial commodity supply levels from the table above."""
    name_to_id = {item.name: item.id for item in globalItemIndex.values()}
    for name, qty in _INITIAL_SUPPLIES.items():
        iid = name_to_id.get(name)
        if iid is None:
            continue
        itu = ctrl.market.itemTracker.get(iid)
        if itu is not None:
            itu.supply = max(itu.supply, qty)


def _hireWorkers(ctrl, building: Building) -> None:
    """Hire workers into a bootstrap building from the available unemployed pool."""
    bt        = building.building_type
    min_tier  = 1 if bt.max_chain_level <= 3 else 2

    _PARTICIPATION = {1: 0.95, 2: 0.90, 3: 0.70, 4: 0.20}
    employed_by_tier: dict = {}
    for b in ctrl.market.buildingList.values():
        for tier, count in b.workers.items():
            employed_by_tier[tier] = employed_by_tier.get(tier, 0.0) + count
    available: dict = {
        pc.tier: max(0.0, pc.count * _PARTICIPATION.get(pc.tier, 0.9)
                     - employed_by_tier.get(pc.tier, 0.0))
        for pc in ctrl.pop_classes
    }

    for pc in sorted(ctrl.pop_classes, key=lambda p: p.tier):
        if pc.tier < min_tier:
            continue
        pool = available.get(pc.tier, 0.0)
        if pool < 1.0:
            continue
        hire = int(min(pool, pc.count * _BOOTSTRAP_HIRE_RATE))
        if hire < 1:
            continue
        building.workers[pc.tier]     = float(hire)
        building.worker_wage[pc.tier] = max(0.01, pc.avg_funds * 0.002)
        available[pc.tier]            = max(0.0, pool - hire)


def _preloadBuffer(ctrl, building: Building) -> None:
    """Pull 3 cycles of input material directly into the building buffer."""
    chain = building.activeChain()
    if chain is None:
        return
    for tag, qty_per_cycle in chain.inputDict.items():
        needed = qty_per_cycle * 3
        for itu in sorted(ctrl.market.getItemsByTag(tag), key=lambda u: u.itemPrice):
            take = min(needed, itu.supply)
            if take <= 0:
                continue
            itu.supply = max(0.0, itu.supply - take)
            iid = itu.itemInstance.id
            building.itemInputBuffer[iid] = (
                building.itemInputBuffer.get(iid, 0.0) + take
            )
            needed -= take
            if needed <= 0:
                break


def _place(ctrl, btype, output_tag: itemTag, input_tags: list) -> Building | None:
    """Place a building and assign a production chain, bypassing action checks.

    Used only during bootstrap — avoids the treasury/resource deduction so the
    caller controls exactly what costs are applied.
    """
    building = Building(ctrl._nextBuildingId, btype)
    building.owner_id = ctrl.id
    ctrl._nextBuildingId += 1
    ctrl.market.buildingList[building.id] = building

    # Transfer seed capital from treasury to the building's operational fund
    seed = min(BUILDING_SEED_CAPITAL, max(0.0, ctrl.treasury))
    ctrl.treasury  -= seed
    building.funds  = seed

    _hireWorkers(ctrl, building)

    chain = requestProductionChainCreation(
        ctrl.market, output_tag, input_tags, building.id
    )
    if chain is not None:
        building.productionChains.append(chain)
        building.selectedChain = 0
        _preloadBuffer(ctrl, building)
    else:
        # Chain creation failed — building stays chainless for AI to assign
        pass

    return building


def _deductBuildCost(ctrl, btype) -> bool:
    """Deduct coins + physical resources for one building; return False if cannot afford.

    Total coin cost = construction_cost + BUILDING_SEED_CAPITAL (operational reserve).
    The seed capital is transferred to the building in _place().
    """
    total_cost = btype.construction_cost + BUILDING_SEED_CAPITAL
    if ctrl.treasury < total_cost:
        return False
    for tag, qty in btype.resource_cost.items():
        avail = sum(u.supply for u in ctrl.market.getItemsByTag(tag))
        if avail < qty:
            return False
    ctrl.treasury -= btype.construction_cost   # seed capital deducted in _place
    for tag, qty in btype.resource_cost.items():
        rem = qty
        for u in sorted(ctrl.market.getItemsByTag(tag), key=lambda u: u.itemPrice):
            take = min(rem, u.supply)
            u.supply = max(0.0, u.supply - take)
            rem -= take
            if rem <= 0:
                break
    return True


# =============================================================================
# ROLE-SPECIFIC BASELINES
# =============================================================================

def _buildRulerBaseline(ctrl) -> None:
    """
    Kingdom / Ruler starting economy — strategic resource base.

    Farm  #1  crops       → food       sustain population
    Forge #1  ore+fuel    → metal      military + tool base
    Farm  #2  fiber       → fabric     clothing → population happiness
    Mill  #1  crops       → food       extra food throughput
    """
    print(f"\n[bootstrap] {ctrl.name} (RULER) — placing strategic baseline")

    # Farm #1: staple food
    if _deductBuildCost(ctrl, BT_FARM):
        _place(ctrl, BT_FARM, itemTag.food, [itemTag.crops])

    # Forge #1: metal foundation
    if _deductBuildCost(ctrl, BT_FORGE):
        _place(ctrl, BT_FORGE, itemTag.metal, [itemTag.ore, itemTag.fuel])

    # Farm #2: fiber for fabric (happiness & consumables)
    if _deductBuildCost(ctrl, BT_FARM):
        b = _place(ctrl, BT_FARM, itemTag.fabric, [itemTag.fiber])
        # Farm can't produce fabric directly — leave chainless for AI
        # (RulerAI will assign the best strategic chain next tick)

    # Mill #1: extra food capacity once more workers come online
    if _deductBuildCost(ctrl, BT_MILL):
        _place(ctrl, BT_MILL, itemTag.food, [itemTag.crops])


def _buildLocalBaseline(ctrl) -> None:
    """
    Merchant / Local starting economy — profit-oriented base.

    Farm    #1  crops       → food       sustain workforce
    Forge   #1  ore+fuel    → metal      high-value output
    Farm    #2  crops+fuel  → alcohol    premium consumer good
    Brewery #1  crops+fuel  → alcohol    diversified luxury production
    """
    print(f"\n[bootstrap] {ctrl.name} (LOCAL) — placing profit baseline")

    # Farm #1: food
    if _deductBuildCost(ctrl, BT_FARM):
        _place(ctrl, BT_FARM, itemTag.food, [itemTag.crops])

    # Forge #1: metal
    if _deductBuildCost(ctrl, BT_FORGE):
        _place(ctrl, BT_FORGE, itemTag.metal, [itemTag.ore, itemTag.fuel])

    # Farm #2: alcohol chain (crops → alcohol needs crops + fuel, assign via brewery instead)
    # Use this second farm for fiber → textile supply chain
    if _deductBuildCost(ctrl, BT_FARM):
        b = _place(ctrl, BT_FARM, itemTag.food, [itemTag.crops])
        # LocalAI will reassign to best-profit chain

    # Brewery: high-margin alcohol for merchants and nobles
    if _deductBuildCost(ctrl, BT_BREWERY):
        _place(ctrl, BT_BREWERY, itemTag.alcohol, [itemTag.crops, itemTag.fuel])


# =============================================================================
# PUBLIC ENTRY POINT
# =============================================================================

def bootstrapEconomy(ctrl) -> None:
    """Set up a controller's initial economy based on its ai_role.

    Must be called after:
      1. env.reset()        — controllers created, market initialised
      2. ctrl.ai_role set   — role determines which baseline is built

    Side effects:
      • ctrl.treasury raised to _BOOTSTRAP_TREASURY if currently lower
      • Item supplies boosted per _INITIAL_SUPPLIES table
      • 3–4 buildings placed with pre-loaded input buffers
    """
    _BOOTSTRAP_TREASURY = 2000.0
    ctrl.treasury = max(ctrl.treasury, _BOOTSTRAP_TREASURY)

    # Level-2 infrastructure so 4 baseline buildings don't instantly saturate
    # logistics capacity.  Also reduces wear accumulation in the early game.
    infra = ctrl.market.infrastructure
    if infra.level < 2:
        infra.level = 2
        infra.wear  = 0.0

    # Boost commodity supplies BEFORE building so resource checks pass
    _boostSupplies(ctrl)

    if ctrl.ai_role == "ruler":
        _buildRulerBaseline(ctrl)
    else:
        _buildLocalBaseline(ctrl)

    n_bldgs = len(ctrl.market.buildingList)
    n_chains = sum(1 for b in ctrl.market.buildingList.values() if b.activeChain())
    print(f"         → {n_bldgs} buildings, {n_chains} active chains, "
          f"treasury={ctrl.treasury:.0f}")
