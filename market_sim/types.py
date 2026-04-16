"""Core enums and lightweight value types.

Nothing in this file imports from the rest of the package — it is the
dependency root that every other module can safely import from.
"""
from __future__ import annotations
from enum import Enum
from dataclasses import dataclass, field


# =============================================================================
# ACTION TYPE
# =============================================================================
class ActionType(Enum):
    """All valid action types available to a controller agent."""
    IDLE              = 0
    BUILD_BUILDING    = 1
    SELECT_CHAIN      = 2
    ADD_CHAIN         = 3
    PROPOSE_TRADE     = 4
    SET_EMBARGO       = 5
    LIFT_EMBARGO      = 6
    BUILD_INFRA       = 7
    DEMOLISH_BUILDING = 8
    REMOVE_CHAIN      = 9
    SET_PRIORITY      = 10
    CANCEL_TRADE      = 11
    RESPOND_TRADE     = 12
    SUBSIDIZE         = 13


@dataclass
class ActionParamSpec:
    """Schema definition for one action parameter (used for RL encoding)."""
    name:        str
    dtype:       str      # 'int'|'float'|'bool'|'itemTag'|'itemTag_list'|'BuildingType'|'list'
    required:    bool = True
    default:     object = None
    description: str = ""


# =============================================================================
# ITEM TAGS  (Early Medieval → Late Industrial era range)
# =============================================================================
class itemTag(Enum):
    # ── Food & Agriculture ───────────────────────────────────────────────────
    food        = 1
    crops       = 2
    livestock   = 3
    alcohol     = 4
    spice       = 5
    # ── Raw Materials ────────────────────────────────────────────────────────
    ore         = 6
    fiber       = 7
    lumber      = 8
    stone       = 9
    gem         = 10
    # ── Processed / Intermediate ─────────────────────────────────────────────
    metal       = 11
    alloy       = 12
    fabric      = 13
    leather     = 14
    ceramic     = 15
    glass       = 16
    paper       = 17
    dye         = 18
    adhesive    = 19
    chemical    = 20
    rubber      = 21
    coal        = 22
    # ── Energy ───────────────────────────────────────────────────────────────
    fuel        = 23
    # ── Finished Goods ───────────────────────────────────────────────────────
    tools       = 24
    weapons     = 25
    ammunition  = 26
    armor       = 27
    consumable  = 28
    luxury      = 29
    medicine    = 30
    machinery   = 31
    # ── Exclusive / Precursor Tags ───────────────────────────────────────────
    explosive   = 32
    saltpeter   = 33
    sulfur      = 34
    nitroglycerin = 35
    coke        = 36
    # ── Misc ─────────────────────────────────────────────────────────────────
    miscellaneous = 37


# =============================================================================
# ITEM FLAGS  (properties of an item, not its category)
# =============================================================================
class itemFlag(Enum):
    # ── Nutritional ──────────────────────────────────────────────────────────
    isNutritional   = 1
    # ── Durability / Decay ───────────────────────────────────────────────────
    isDurable       = 2
    isPerishable    = 3
    # ── Economic ─────────────────────────────────────────────────────────────
    isValuable      = 4
    isLuxury        = 5
    isRare          = 6
    # ── Textile / Material Feel ───────────────────────────────────────────────
    isSmooth        = 7
    isFlexible      = 8
    isRough         = 9
    # ── Structural / Mechanical ──────────────────────────────────────────────
    isHardened      = 10
    isTough         = 11
    isBrittle       = 12
    # ── Purity / Refinement ──────────────────────────────────────────────────
    isPure          = 13
    isCrude         = 14
    isRefined       = 15
    isAlloyed       = 16
    # ── Chemical / Reactive ──────────────────────────────────────────────────
    isReactive      = 17
    isFlammable     = 18
    isToxic         = 19
    # ── Physical Form ────────────────────────────────────────────────────────
    isPowdered      = 20
    isLiquid        = 21
    isBulky         = 22
    isFragile       = 23
    # ── Industrial ───────────────────────────────────────────────────────────
    isConductive    = 24
    isHighCalorie   = 25


# =============================================================================
# ITEM
# =============================================================================
class item:
    def __init__(self, id: int, tags: set, flags: set,
                 level: int, rawResource: bool, name: str = ""):
        self.id          = id
        self.tags        = tags
        self.flags       = flags
        self.level       = level
        self.rawResource = rawResource
        self.name        = name or f"item_{id}"

    def hasFlag(self, flag: itemFlag) -> bool:
        return flag in self.flags

    def hasTag(self, tag: itemTag) -> bool:
        return tag in self.tags

    def __repr__(self):
        return self.name


# =============================================================================
# SMALL VALUE TYPES
# =============================================================================
@dataclass
class qualityTier:
    threshold:   float        # 0.0–1.0 fraction of preferred flags matched
    outputFlags: set          # flags added to output at this tier


@dataclass
class ActionResult:
    success: bool
    message: str
    data:    object = None


class TradeDeal:
    _next_id: int = 0

    def __init__(self, from_id: int, to_id: int, item_tag: itemTag,
                 quantity_per_tick: float, price_per_unit: float):
        TradeDeal._next_id += 1
        self.id                   = TradeDeal._next_id
        self.from_controller_id   = from_id
        self.to_controller_id     = to_id
        self.item_tag             = item_tag
        self.quantity_per_tick    = quantity_per_tick
        self.price_per_unit       = price_per_unit
        self.active               = False

    def __repr__(self):
        status = "active" if self.active else "pending"
        return (f"TradeDeal#{self.id}({status}: "
                f"{self.from_controller_id}→{self.to_controller_id} "
                f"{self.item_tag.name} x{self.quantity_per_tick} "
                f"@ {self.price_per_unit:.2f}/u)")


# =============================================================================
# ACTION_SPECS — full parameter schema per action
# =============================================================================
ACTION_SPECS: dict = {
    ActionType.IDLE: [],
    ActionType.BUILD_BUILDING: [
        ActionParamSpec("building_type", "BuildingType", required=False,
                        default="BT_WORKSHOP", description="BuildingType to construct"),
        ActionParamSpec("chains", "list", required=False, default=[],
                        description="Initial ProductionChains to install"),
    ],
    ActionType.SELECT_CHAIN: [
        ActionParamSpec("building_id",  "int", description="Target building ID"),
        ActionParamSpec("chain_index",  "int", description="Index into productionChains list"),
    ],
    ActionType.ADD_CHAIN: [
        ActionParamSpec("output_tag",  "itemTag",       description="Desired output tag"),
        ActionParamSpec("input_tags",  "itemTag_list",  description="Input tags matching a rule"),
        ActionParamSpec("building_id", "int", required=False, default=None,
                        description="Specific building to assign chain (auto-select if None)"),
    ],
    ActionType.PROPOSE_TRADE: [
        ActionParamSpec("target_id", "int",     description="Target controller ID"),
        ActionParamSpec("item_tag",  "itemTag", description="Tag of item to trade"),
        ActionParamSpec("quantity",  "float",   required=False, default=10.0),
        ActionParamSpec("price",     "float",   required=False, default=1.0),
    ],
    ActionType.SET_EMBARGO: [
        ActionParamSpec("target_id", "int", description="Controller ID to embargo"),
    ],
    ActionType.LIFT_EMBARGO: [
        ActionParamSpec("target_id", "int", description="Controller ID to un-embargo"),
    ],
    ActionType.BUILD_INFRA: [],
    ActionType.DEMOLISH_BUILDING: [
        ActionParamSpec("building_id", "int", description="Building ID to demolish"),
    ],
    ActionType.REMOVE_CHAIN: [
        ActionParamSpec("building_id",  "int"),
        ActionParamSpec("chain_index",  "int"),
    ],
    ActionType.SET_PRIORITY: [
        ActionParamSpec("building_id",  "int"),
        ActionParamSpec("chain_index",  "int"),
        ActionParamSpec("priority",     "int"),
    ],
    ActionType.CANCEL_TRADE: [
        ActionParamSpec("deal_id", "int"),
    ],
    ActionType.RESPOND_TRADE: [
        ActionParamSpec("deal_id", "int"),
        ActionParamSpec("accept",  "bool"),
    ],
    ActionType.SUBSIDIZE: [
        ActionParamSpec("level", "int", required=False, default=1,
                        description="Subsidy level 0-5 (0=off, 1=10%, 2=20%, 3=30%, 4=40%, 5=50%)"),
    ],
}
