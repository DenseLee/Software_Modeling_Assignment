"""Flag inheritance, quality scoring, random item spawning, and simulateProduction.

Depends on: types, structures, _globals, _rng
"""
from __future__ import annotations
from typing import Optional

from .types      import item, itemTag, itemFlag, qualityTier
from .structures import (
    Building, Market, ProductionChain, productionGraphs,
    itemTrackUnit,
)
from ._globals   import globalItemIndex
from ._rng       import _rng
from . import _globals


# =============================================================================
# PREFERENCE SCORING
# =============================================================================
def scoreItemForSlot(item_instance: item, required_tag: itemTag,
                     preferred_flags: list) -> float:
    if required_tag not in item_instance.tags:
        return -1.0
    if not preferred_flags:
        return 0.0
    return sum(1 for f in preferred_flags if item_instance.hasFlag(f)) / len(preferred_flags)


def resolveOutputFlags(avg_score: float, tiers: list) -> set:
    for tier in tiers:
        if avg_score >= tier.threshold:
            return set(tier.outputFlags)
    return set()


def bestItemsForRule(market: Market, rule) -> dict:
    result = {}
    for tag in rule.inputTags:
        preferred  = rule.inputPreferences.get(tag, [])
        candidates = market.getItemsByTag(tag)
        if not candidates:
            result[tag] = (None, -1.0)
            continue
        best  = max(candidates, key=lambda u: scoreItemForSlot(u.itemInstance, tag, preferred))
        score = scoreItemForSlot(best.itemInstance, tag, preferred)
        result[tag] = (best, score)
    return result


def requestProductionChainCreation(
    market: Market,
    output_tag: itemTag,
    proposed_input_tags: list,
    target_building_id: int = None,
) -> Optional[ProductionChain]:
    """Validate and instantiate a new production chain.

    Returns None if rejected; returns a ProductionChain ready to be assigned
    to a building via action_addChainToBuilding.
    """
    rules = productionGraphs.get(output_tag, [])
    matching_rule = next(
        (r for r in rules if set(r.inputTags) == set(proposed_input_tags)), None
    )
    if matching_rule is None:
        print(f"[rejected] no rule: {[t.name for t in proposed_input_tags]} → {output_tag.name}")
        return None

    for tag in proposed_input_tags:
        if not market.hasItemWithTag(tag):
            print(f"[rejected] '{tag.name}' not in market")
            return None

    check_buildings = (
        [market.buildingList[target_building_id]]
        if target_building_id is not None and target_building_id in market.buildingList
        else market.buildingList.values()
    )
    for building in check_buildings:
        for chain in building.productionChains:
            if (set(chain.inputDict.keys()) == set(proposed_input_tags)
                    and output_tag in chain.outputDict):
                print(f"[rejected] identical chain already on building #{building.id}")
                return None

    best      = bestItemsForRule(market, matching_rule)
    scores    = [s for (_, s) in best.values() if s >= 0.0]
    avg_score = sum(scores) / len(scores) if scores else 0.0
    output_flags = resolveOutputFlags(avg_score, matching_rule.qualityTiers)

    print(f"\n[approved] {[t.name for t in proposed_input_tags]} → {output_tag.name}")
    for tag, (itu, score) in best.items():
        print(f"  [{tag.name}] best: {itu.itemInstance.name if itu else 'NONE'}  score: {score:.2f}")
    print(f"  avg score: {avg_score:.2f}  output flags: {[f.name for f in output_flags]}\n")

    # Cycle quantities scale with rule level.
    # Level 1 is calibrated so 2 well-staffed farms (12 workers, 60% eff,
    # cycle ≈ 17 ticks) produce ~10 food/tick — enough for 1 000 people.
    # Higher tiers produce fewer units of more refined goods.
    _OUT = {1: 80, 2: 50, 3: 30, 4: 20, 5: 12}
    _IN  = {1: 40, 2: 30, 3: 20, 4: 15, 5: 10}
    out_qty = _OUT.get(matching_rule.level, 12)
    in_qty  = _IN.get(matching_rule.level, 10)

    return ProductionChain(
        inputDict={tag: in_qty for tag in proposed_input_tags},
        outputDict={output_tag: out_qty},
        timeToProduce=matching_rule.level * 10,
        priority=matching_rule.level,
        inputPreferences=matching_rule.inputPreferences,
        qualityTiers=matching_rule.qualityTiers,
    )


# =============================================================================
# FLAG INHERITANCE & MUTATION
# =============================================================================
INHERITABLE_FLAGS: frozenset = frozenset({
    itemFlag.isSmooth, itemFlag.isFlexible, itemFlag.isRough,
    itemFlag.isHardened, itemFlag.isTough, itemFlag.isBrittle,
    itemFlag.isPure, itemFlag.isCrude, itemFlag.isReactive, itemFlag.isFlammable,
    itemFlag.isValuable, itemFlag.isRare, itemFlag.isToxic,
    itemFlag.isConductive, itemFlag.isHighCalorie,
})

FLAG_CONFLICTS: list = [
    (itemFlag.isSmooth,   itemFlag.isRough),
    (itemFlag.isFlexible, itemFlag.isBrittle),
    (itemFlag.isHardened, itemFlag.isBrittle),
    (itemFlag.isPure,     itemFlag.isCrude),
    (itemFlag.isTough,    itemFlag.isFlexible),
]

FLAG_MUTATIONS: dict = {
    itemFlag.isSmooth:      [itemFlag.isRough],
    itemFlag.isRough:       [itemFlag.isSmooth],
    itemFlag.isFlexible:    [itemFlag.isTough, itemFlag.isBrittle],
    itemFlag.isTough:       [itemFlag.isFlexible, itemFlag.isBrittle],
    itemFlag.isBrittle:     [itemFlag.isHardened, itemFlag.isTough],
    itemFlag.isHardened:    [itemFlag.isBrittle],
    itemFlag.isPure:        [itemFlag.isCrude],
    itemFlag.isCrude:       [itemFlag.isPure],
    itemFlag.isReactive:    [itemFlag.isFlammable, itemFlag.isToxic],
    itemFlag.isFlammable:   [itemFlag.isReactive],
    itemFlag.isValuable:    [itemFlag.isRare],
    itemFlag.isRare:        [itemFlag.isValuable],
    itemFlag.isToxic:       [itemFlag.isReactive],
    itemFlag.isConductive:  [itemFlag.isFlexible],
    itemFlag.isHighCalorie: [itemFlag.isFlammable],
}


def _resolveConflicts(flags: set) -> set:
    resolved = set(flags)
    for keep, remove in FLAG_CONFLICTS:
        if keep in resolved and remove in resolved:
            resolved.discard(remove)
    return resolved


def inheritFlags(
    input_items: list,
    base_output_flags: set,
    mutation_chance: float = 0.08,
    split_chance: float = 0.04,
) -> tuple:
    """Merge, inherit, and mutate flags from input items onto a crafted output."""
    flag_counts: dict = {}
    for inp in input_items:
        for f in inp.flags:
            if f in INHERITABLE_FLAGS:
                flag_counts[f] = flag_counts.get(f, 0) + 1

    inherited: set = set()
    for f, count in flag_counts.items():
        if count >= 2 or _rng.random() < 0.70:
            inherited.add(f)

    merged  = _resolveConflicts(base_output_flags | inherited)
    primary = set(merged)
    for f in list(merged):
        if f in INHERITABLE_FLAGS and f in FLAG_MUTATIONS and _rng.random() < mutation_chance:
            primary.discard(f)
            primary.add(_rng.choice(FLAG_MUTATIONS[f]))
    primary = _resolveConflicts(primary)

    split_flags: Optional[set] = None
    if _rng.random() < split_chance:
        variant = set(primary)
        mutable = [f for f in variant if f in FLAG_MUTATIONS and f in INHERITABLE_FLAGS]
        if mutable:
            pivot      = _rng.choice(mutable)
            candidates = [m for m in FLAG_MUTATIONS[pivot] if m not in primary]
            if not candidates:
                candidates = FLAG_MUTATIONS[pivot]
            variant.discard(pivot)
            variant.add(_rng.choice(candidates))
            variant = _resolveConflicts(variant)
            if variant != primary:
                split_flags = variant

    return primary, split_flags


# =============================================================================
# RANDOM ITEM SPAWNING
# =============================================================================
_FLAG_ADJECTIVES: dict = {
    itemFlag.isSmooth:      "smooth",   itemFlag.isFlexible:  "flexible",
    itemFlag.isRough:       "rough",    itemFlag.isHardened:  "hardened",
    itemFlag.isTough:       "tough",    itemFlag.isBrittle:   "brittle",
    itemFlag.isPure:        "pure",     itemFlag.isCrude:     "crude",
    itemFlag.isReactive:    "reactive", itemFlag.isFlammable: "volatile",
    itemFlag.isValuable:    "rich",     itemFlag.isRare:      "rare",
    itemFlag.isToxic:       "toxic",    itemFlag.isConductive:"conductive",
    itemFlag.isHighCalorie: "dense",    itemFlag.isLiquid:    "liquid",
    itemFlag.isPowdered:    "powdered", itemFlag.isDurable:   "hard",
    itemFlag.isBulky:       "heavy",
}

_TAG_NOUNS: dict = {
    itemTag.ore: "ore", itemTag.fiber: "fiber", itemTag.lumber: "timber",
    itemTag.stone: "stone", itemTag.crops: "grain", itemTag.metal: "metal",
    itemTag.alloy: "alloy", itemTag.fabric: "cloth", itemTag.fuel: "fuel",
    itemTag.livestock: "beast", itemTag.chemical: "reagent", itemTag.gem: "gem",
    itemTag.leather: "hide", itemTag.miscellaneous: "material",
}

_NAME_TAG_PRIORITY: list = [
    itemTag.gem, itemTag.ore, itemTag.fiber, itemTag.lumber, itemTag.stone,
    itemTag.chemical, itemTag.metal, itemTag.alloy, itemTag.fabric,
    itemTag.fuel, itemTag.crops, itemTag.livestock, itemTag.leather,
    itemTag.miscellaneous,
]

_FLAG_NAME_PRIORITY: list = [
    itemFlag.isRare, itemFlag.isPure, itemFlag.isToxic, itemFlag.isConductive,
    itemFlag.isReactive, itemFlag.isSmooth, itemFlag.isHardened, itemFlag.isBrittle,
    itemFlag.isFlexible, itemFlag.isTough, itemFlag.isRough, itemFlag.isCrude,
    itemFlag.isFlammable, itemFlag.isHighCalorie, itemFlag.isDurable,
]


def _generateItemName(tags: set, flags: set) -> str:
    primary_tag = next((t for t in _NAME_TAG_PRIORITY if t in tags), next(iter(tags)))
    noun = _TAG_NOUNS.get(primary_tag, primary_tag.name)
    adjs = [_FLAG_ADJECTIVES[f] for f in _FLAG_NAME_PRIORITY
            if f in flags and f in _FLAG_ADJECTIVES][:2]
    return "_".join(adjs + [noun])


class SpawnProfile:
    def __init__(self, tags, guaranteed_flags, optional_flags,
                 min_optional, max_optional, min_tech, max_tech):
        self.tags              = tags
        self.guaranteed_flags  = guaranteed_flags
        self.optional_flags    = optional_flags
        self.min_optional      = min_optional
        self.max_optional      = max_optional
        self.min_tech          = min_tech
        self.max_tech          = max_tech


SPAWN_PROFILES: list = [
    SpawnProfile(
        tags={itemTag.ore},
        guaranteed_flags={itemFlag.isDurable, itemFlag.isBulky},
        optional_flags=[itemFlag.isPure, itemFlag.isCrude, itemFlag.isValuable,
                        itemFlag.isRare, itemFlag.isConductive],
        min_optional=1, max_optional=3, min_tech=1, max_tech=4,
    ),
    SpawnProfile(
        tags={itemTag.fiber, itemTag.crops},
        guaranteed_flags={itemFlag.isPerishable},
        optional_flags=[itemFlag.isSmooth, itemFlag.isRough, itemFlag.isFlexible,
                        itemFlag.isValuable, itemFlag.isRare],
        min_optional=1, max_optional=2, min_tech=1, max_tech=3,
    ),
    SpawnProfile(
        tags={itemTag.lumber},
        guaranteed_flags={itemFlag.isBulky},
        optional_flags=[itemFlag.isTough, itemFlag.isDurable,
                        itemFlag.isFlammable, itemFlag.isValuable],
        min_optional=1, max_optional=2, min_tech=1, max_tech=2,
    ),
    SpawnProfile(
        tags={itemTag.stone},
        guaranteed_flags={itemFlag.isDurable, itemFlag.isBulky},
        optional_flags=[itemFlag.isTough, itemFlag.isPure, itemFlag.isValuable,
                        itemFlag.isRare, itemFlag.isBrittle],
        min_optional=1, max_optional=2, min_tech=1, max_tech=2,
    ),
    SpawnProfile(
        tags={itemTag.chemical},
        guaranteed_flags={itemFlag.isRare},
        optional_flags=[itemFlag.isReactive, itemFlag.isToxic, itemFlag.isFlammable,
                        itemFlag.isPure, itemFlag.isValuable, itemFlag.isLiquid],
        min_optional=2, max_optional=3, min_tech=3, max_tech=5,
    ),
    SpawnProfile(
        tags={itemTag.gem},
        guaranteed_flags={itemFlag.isValuable, itemFlag.isDurable},
        optional_flags=[itemFlag.isPure, itemFlag.isRare,
                        itemFlag.isFragile, itemFlag.isConductive],
        min_optional=1, max_optional=2, min_tech=1, max_tech=3,
    ),
    SpawnProfile(
        tags={itemTag.fuel},
        guaranteed_flags={itemFlag.isFlammable},
        optional_flags=[itemFlag.isHighCalorie, itemFlag.isLiquid, itemFlag.isValuable,
                        itemFlag.isBulky, itemFlag.isToxic],
        min_optional=1, max_optional=3, min_tech=2, max_tech=5,
    ),
]

ITEM_SPAWN_CHANCE: float = 0.05


def _spawnItemOnController(ctrl, profile: SpawnProfile) -> Optional[item]:
    if not (profile.min_tech <= ctrl.techLevel <= profile.max_tech):
        return None

    n     = _rng.randint(profile.min_optional,
                         min(profile.max_optional, len(profile.optional_flags)))
    picked = set(_rng.sample(profile.optional_flags, n))
    flags  = _resolveConflicts(profile.guaranteed_flags | picked)
    level  = min(ctrl.techLevel, profile.max_tech)

    base_name     = _generateItemName(profile.tags, flags)
    existing_names = {i.name for i in globalItemIndex.values()}
    name, suffix  = base_name, 1
    while name in existing_names:
        suffix += 1
        name = f"{base_name}_{suffix}"

    new_item = item(
        id=_globals._nextItemId,
        tags=set(profile.tags),
        flags=flags,
        level=level,
        rawResource=True,
        name=name,
    )
    _globals._nextItemId += 1

    globalItemIndex[new_item.id] = new_item
    ctrl.market.addItem(new_item)
    return new_item


def trySpawnItems() -> list:
    from ._globals import globalControllers
    if not globalControllers or _rng.random() > ITEM_SPAWN_CHANCE:
        return []
    ctrl     = _rng.choice(list(globalControllers.values()))
    eligible = [p for p in SPAWN_PROFILES
                if p.min_tech <= ctrl.techLevel <= p.max_tech]
    if not eligible:
        return []
    profile  = _rng.choice(eligible)
    new_item = _spawnItemOnController(ctrl, profile)
    if new_item is None:
        return []
    print(f"[spawn] {ctrl.name}: '{new_item.name}' discovered  "
          f"tags={[t.name for t in new_item.tags]}  "
          f"flags={[f.name for f in new_item.flags]}")
    return [(ctrl, new_item)]


# =============================================================================
# SIMULATE PRODUCTION (one crafting cycle with inheritance)
# =============================================================================
def simulateProduction(
    building: Building,
    market: Market,
    input_item_ids: dict,
    mutation_chance: float = 0.08,
    split_chance: float = 0.04,
) -> dict:
    """Run one production cycle with flag inheritance. Returns {item_id: qty}."""
    chain = building.activeChain()
    if chain is None:
        return {}

    input_items_used = []
    for tag, iid in input_item_ids.items():
        inp = globalItemIndex.get(iid)
        if inp and tag in inp.tags:
            input_items_used.append(inp)

    output: dict = {}

    for output_tag, quantity in chain.outputDict.items():
        base_candidates = [
            i for i in globalItemIndex.values()
            if output_tag in i.tags and not i.rawResource
        ]
        if not base_candidates:
            continue
        base = min(base_candidates, key=lambda i: i.level)

        primary_flags, split_flags = inheritFlags(
            input_items_used, base.flags, mutation_chance, split_chance)

        # ── Primary output ───────────────────────────────────────────────────
        p_name   = _generateItemName(base.tags, primary_flags)
        existing = {i.name for i in globalItemIndex.values()}
        if p_name in existing:
            p_name = f"{p_name}_{_globals._nextItemId}"

        primary_item = item(
            id=_globals._nextItemId,
            tags=set(base.tags),
            flags=primary_flags,
            level=base.level,
            rawResource=False,
            name=p_name,
        )
        _globals._nextItemId += 1
        globalItemIndex[primary_item.id] = primary_item
        market.addItem(primary_item, initial_supply=0.0)
        output[primary_item.id] = float(quantity)

        inherited_display = [f.name for f in primary_flags if f in INHERITABLE_FLAGS]
        base_display      = [f.name for f in primary_flags if f not in INHERITABLE_FLAGS]
        print(f"[crafted]  '{primary_item.name}'  "
              f"base_flags={base_display}  inherited={inherited_display}")

        # ── Split variant (byproduct) ────────────────────────────────────────
        if split_flags is not None:
            s_name = _generateItemName(base.tags, split_flags)
            if s_name in {i.name for i in globalItemIndex.values()}:
                s_name = f"{s_name}_{_globals._nextItemId}_variant"
            split_item_obj = item(
                id=_globals._nextItemId,
                tags=set(base.tags),
                flags=split_flags,
                level=base.level,
                rawResource=False,
                name=s_name,
            )
            _globals._nextItemId += 1
            globalItemIndex[split_item_obj.id] = split_item_obj
            market.addItem(split_item_obj, initial_supply=0.0)
            output[split_item_obj.id] = max(1.0, float(quantity) // 4)
            diff = split_flags.symmetric_difference(primary_flags)
            print(f"[split!]   '{split_item_obj.name}'  "
                  f"mutation_diff={[f.name for f in diff]}")

    return output
